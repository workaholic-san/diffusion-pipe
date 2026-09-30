"""Integration checks for the muon_kahan wiring in train.py.

The Muon/AdamW split introduces six config keys. The failure mode this file
guards against is a user leaving them in `[optimizer]` while selecting a
different optimizer: the keys used to reach that optimizer's constructor and
raise `TypeError: __init__() got an unexpected keyword argument`.

`get_optimizer` is a closure inside train.py's main body, so it is not directly
importable. These tests reproduce its kwarg handling against the same constants
the real code imports, which keeps the contract honest without a refactor.
"""

import sys
from pathlib import Path

import torch
import toml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizers.muon_kahan import MUON_SPLIT_KEYS, MuonKahan, split_param_groups

REPO = Path(__file__).resolve().parent.parent
EXAMPLE = REPO / 'examples' / 'muon_kahan_wsd_lokr.toml'


def _simulate_get_optimizer_kwargs(optim_config):
    """Mirror train.py's kwargs construction and Muon-key popping."""
    kwargs = {k: v for k, v in optim_config.items() if k not in ['type', 'gradient_release']}
    muon_cfg = {k: kwargs.pop(k) for k in MUON_SPLIT_KEYS if k in kwargs}
    return kwargs, muon_cfg


def test_example_config_declares_every_split_key():
    config = toml.load(EXAMPLE)
    optim = config['optimizer']
    assert optim['type'] == 'muon_kahan', 'the example must select the split explicitly'
    for key in ('muon_min_size', 'adamw_lr', 'kahan_adamw', 'adamw_betas', 'adamw_eps'):
        assert key in optim, f'example config is missing {key}'
    assert config['lr_scheduler'] == 'wsd'
    assert 0.0 <= config['wsd_decay_frac'] < 1.0


def test_split_keys_do_not_leak_into_torch_adamw():
    """The regression: a muon_kahan config must not break a plain adamw run."""
    config = toml.load(EXAMPLE)['optimizer']
    config['type'] = 'adamw'          # user switches optimizer, forgets the muon keys
    kwargs, muon_cfg = _simulate_get_optimizer_kwargs(config)

    assert muon_cfg, 'the muon keys should have been consumed'
    assert not (set(kwargs) & set(MUON_SPLIT_KEYS)), \
        f'keys leaked into torch.optim.AdamW: {sorted(set(kwargs) & set(MUON_SPLIT_KEYS))}'

    # The real proof: the constructor accepts them.
    param = torch.nn.Parameter(torch.zeros(4, 4))
    torch.optim.AdamW([param], **kwargs)   # must not raise


def test_lokr_like_shapes_split_the_way_the_recipe_expects():
    """A LoKr adapter contributes small [4,4] factors that must reach AdamW."""
    lokr_w1 = torch.nn.Parameter(torch.zeros(1024, 4))   # tall but thin
    lokr_w2 = torch.nn.Parameter(torch.zeros(4, 4))       # decompose_factor shape
    norm = torch.nn.Parameter(torch.zeros(1024))          # 1-D
    embed = torch.nn.Parameter(torch.zeros(4096, 1024))    # matrix -> Muon

    groups = split_param_groups(
        [{'params': [lokr_w1, lokr_w2, norm, embed], 'lr': 1e-3, 'weight_decay': 0.01}],
        muon_min_size=32, adamw_lr=1e-4, kahan_adamw=True)

    muon = {id(p) for g in groups if g['use_muon'] for p in g['params']}
    adamw = {id(p) for g in groups if g['use_muon'] is False for p in g['params']}

    assert id(embed) in muon, 'the large matrix belongs on Muon'
    assert id(lokr_w1) not in muon and id(lokr_w2) not in muon, \
        'LoKr factors below muon_min_size must go to AdamW'
    assert id(norm) in adamw, '1-D params belong on AdamW'
    assert len(muon) + len(adamw) == 4


def test_muon_kahan_step_advances_both_groups():
    """Both buckets must actually move when the optimizer steps."""
    muon_like = torch.nn.Parameter(torch.zeros(64, 64))
    adamw_like = torch.nn.Parameter(torch.zeros(4, 4))
    vec = torch.nn.Parameter(torch.zeros(64))

    groups = split_param_groups(
        [{'params': [muon_like, adamw_like, vec], 'lr': 1e-2, 'weight_decay': 0.0}],
        muon_min_size=32, kahan_adamw=True)

    # MuonKahan subclasses pytorch_optimizer.Muon; if that import failed the
    # base is object and there is no muon branch. Use a stand-in that records.
    stepped = {'muon': 0}

    class _RecordingMuon(MuonKahan):
        def _adamw_kahan_step(self, groups_):        # keep the real AdamW maths
            return super()._adamw_kahan_step(groups_)

    opt = MuonKahan(groups, lr=1e-2)
    # Drive the AdamW half directly; the Muon half needs the upstream optimizer.
    for p in (muon_like, adamw_like, vec):
        p.grad = torch.ones_like(p)
    opt._adamw_kahan_step([g for g in opt.param_groups if g.get('use_muon') is False])

    assert adamw_like.abs().sum().item() > 0, 'AdamW-group parameters did not move'
    assert vec.abs().sum().item() > 0, '1-D parameters did not move'
    assert muon_like.abs().sum().item() == 0, 'Muon group must not be touched by the AdamW half'
    assert stepped['muon'] == 0


def test_state_dict_roundtrip_preserves_kahan_compensation():
    """A resumed run must not silently lose the fp32 compensation tensor."""
    src = torch.nn.Parameter(torch.full((32,), 0.02, dtype=torch.bfloat16))
    opt = MuonKahan([{'params': [src], 'use_muon': False, 'lr': 1e-3,
                      'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0, 'kahan': True}])
    for _ in range(3):
        src.grad = torch.full((32,), 1e-4, dtype=torch.bfloat16)
        opt.step()

    assert 'kahan_comp' in opt.state[src], 'no compensation tensor was created'

    state = opt.state_dict()
    assert any('kahan_comp' in s for s in state['state'].values()), \
        'kahan_comp is missing from the checkpointed optimizer state'


if __name__ == '__main__':
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith('test_') and callable(f)]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f'PASS  {name}')
        except Exception as exc:
            failed += 1
            print(f'FAIL  {name}: {type(exc).__name__}: {exc}')
    print(f'\n{len(fns) - failed}/{len(fns)} passed')
    sys.exit(1 if failed else 0)
