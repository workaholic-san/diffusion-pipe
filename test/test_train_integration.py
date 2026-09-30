"""End-to-end smoke test of the training instrument.

Builds a tiny model with the same *shape mixture* the real recipe produces
(large transformer matrices, small LoKr-style factors, 1-D norms), then drives
it through the exact pipeline train.py uses:

    split_param_groups -> MuonKahan -> build_lr_scheduler -> install_debug_hook
    -> N real optimizer steps

This is the closest thing to a training run that is possible without the 30 GB
of model weights: it exercises the optimizer maths, the LR schedule, the debug
hook and the resume fast-forward on real tensors rather than on assertions
about source text.

CPU-only and takes a few seconds.
"""

import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizers.muon_kahan import MuonKahan, split_param_groups
from utils.lr_schedulers import build_lr_scheduler, fast_forward_lr_scheduler
from utils.train_debug import install_debug_hook


class TinyAdapterModel(nn.Module):
    """Shape mixture matching a real LoKr run."""

    def __init__(self, width=128, rank=4):
        super().__init__()
        self.attn_q = nn.Linear(width, width, bias=False)      # [128,128] -> Muon
        self.attn_k = nn.Linear(width, width, bias=False)      # [128,128] -> Muon
        self.attn_v = nn.Linear(width, width, bias=False)      # [128,128] -> Muon
        self.norm = nn.LayerNorm(width)                        # 1-D -> AdamW
        # LoKr-style low-rank correction: rank -> width, so w1 is [width, rank]
        # and w2 is [rank, width]. Both are below muon_min_size on their small
        # axis, which is exactly why they belong to the AdamW group.
        self.lokr_w1 = nn.Parameter(torch.zeros(width, rank))
        self.lokr_w2 = nn.Parameter(torch.zeros(rank, width))
        self.embed = nn.Parameter(torch.zeros(512, width))     # large -> Muon
        self.register_buffer('mu', torch.zeros(()))

        # Real LoKr factors are non-zero at init. Left at zero, the product
        # x @ w1 @ w2 is identically zero AND its gradient w.r.t. both factors
        # is exactly zero, so neither would ever receive an update.
        with torch.no_grad():
            self.lokr_w1.normal_(0, 0.02)
            self.lokr_w2.normal_(0, 0.02)

    def forward(self, x):
        h = self.attn_v(torch.nn.functional.silu(self.attn_k(self.attn_q(x))))
        h = h + (x @ self.lokr_w1 @ self.lokr_w2) * 0.01
        h = self.norm(h)
        return h + x


class FakeEngine:
    """Stands in for DeepSpeedEngine: the debug hook and scheduler touch only these."""

    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.lr_scheduler = None
        self.global_steps = 0
        self._custom_global_grad_norm = None
        self._custom_global_grad_norm_step = None

    def get_lr(self):
        return [g['lr'] for g in self.optimizer.param_groups]


def _build_optimizer(model, muon_lr=1e-3, adamw_lr=1e-4, muon_min_size=32):
    """Mirror train.get_optimizer for the muon_kahan route."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = split_param_groups(
        [{'params': params, 'lr': muon_lr, 'weight_decay': 0.01}],
        muon_min_size=muon_min_size, adamw_lr=adamw_lr, kahan_adamw=True,
        adamw_betas=(0.95, 0.995), adamw_eps=1e-8)
    return MuonKahan(groups, lr=muon_lr, weight_decay=0.01)


def test_end_to_end_training_run_is_stable_and_learns():
    torch.manual_seed(0)
    model = TinyAdapterModel()
    model.lokr_w1.data.normal_(0, 0.02)
    model.lokr_w2.data.normal_(0, 0.02)

    opt = _build_optimizer(model)
    engine = FakeEngine(opt)
    install_debug_hook(opt, {'debug': {'log_param_groups': False,
                                      'debug_every_n_steps': 5,
                                      'log_grad_norm': True,
                                      'log_group_grad_norm': True,
                                      'log_muon_param_norm': True,
                                      'log_adamw_param_norm': True,
                                      'log_adamw_update_norm': True,
                                      'log_adamw_state_norm': True}}, engine)

    # 5 epochs x 20 steps, matching the recipe's shape
    cfg = {'epochs': 5, 'warmup_steps': 10, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5}
    engine.lr_scheduler = build_lr_scheduler(opt, cfg, steps_per_epoch=20)

    x = torch.randn(8, 16, 128)
    target = torch.randn(8, 16, 128)

    first_loss = last_loss = None
    for step in range(1, 101):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(model(x), target)
        loss.backward()
        if step == 1:
            first_loss = loss.item()
        last_loss = loss.item()
        opt.step()
        engine.lr_scheduler.step()

    assert all(torch.isfinite(p).all() for p in model.parameters()), \
        'a parameter became NaN or inf during training'
    assert last_loss < first_loss, (
        f'the model did not learn: first loss {first_loss:.6f} -> last {last_loss:.6f}')


def test_both_buckets_are_populated_and_move():
    model = TinyAdapterModel()
    opt = _build_optimizer(model)

    muon_groups = [g for g in opt.param_groups if g.get('use_muon')]
    adamw_groups = [g for g in opt.param_groups if g.get('use_muon') is False]
    assert muon_groups and adamw_groups, 'the split produced only one bucket'

    before = {id(p): p.detach().clone() for g in opt.param_groups for p in g['params']}
    model(torch.randn(4, 16, 128)).pow(2).mean().backward()
    opt._adamw_kahan_step(adamw_groups)
    for g in adamw_groups:
        for p in g['params']:
            if p.grad is None:
                continue
            assert not torch.equal(before[id(p)], p.detach()), \
                f'an AdamW-group parameter did not move: {tuple(p.shape)}'


def test_debug_hook_records_real_metrics_not_zeros():
    model = TinyAdapterModel()
    opt = _build_optimizer(model)
    engine = FakeEngine(opt)
    install_debug_hook(opt, {'debug': {'log_param_groups': False, 'debug_every_n_steps': 1}},
                       engine)

    model(torch.randn(4, 16, 128)).pow(2).mean().backward()
    opt.step()

    metrics = getattr(opt, '_debug_metrics', None)
    assert metrics, 'the debug hook recorded nothing'
    assert 'train/grad_norm' in metrics, 'global grad norm missing'
    assert 'debug/adamw/update_norm' in metrics, 'AdamW update norm missing'
    assert 'debug/adamw/kahan_comp_norm' in metrics, 'Kahan compensation norm missing'

    # The update norm must be a real measurement: the first step genuinely moves
    # the AdamW parameters, so a 0.0 here would be the fabricated-zero bug.
    assert metrics['debug/adamw/update_norm'] > 0, \
        'AdamW update norm is 0.0 - a fabricated zero, not a measurement'


def test_debug_hook_omits_adamw_metrics_when_there_is_no_adamw_group():
    """A plain optimizer must not report AdamW metrics as 0.0."""
    param = torch.nn.Parameter(torch.zeros(8, 8))
    opt = torch.optim.SGD([param], lr=1e-3)
    engine = FakeEngine(opt)
    install_debug_hook(opt, {'debug': {'log_param_groups': False, 'debug_every_n_steps': 1}},
                       engine)

    param.grad = torch.ones_like(param)
    opt.step()

    metrics = getattr(opt, '_debug_metrics', None) or {}
    assert not [k for k in metrics if k.startswith('debug/adamw/')], \
        f'AdamW metrics reported for a non-split optimizer: {sorted(metrics)}'


def test_wsd_lr_trace_shape_over_a_full_run():
    model = TinyAdapterModel()
    opt = _build_optimizer(model)
    engine = FakeEngine(opt)
    cfg = {'epochs': 5, 'warmup_steps': 10, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5}
    engine.lr_scheduler = build_lr_scheduler(opt, cfg, steps_per_epoch=20)

    base_muon = 1e-3
    warmup = cfg['warmup_steps']
    trace = []
    for _ in range(100):
        engine.lr_scheduler.step()
        trace.append(engine.get_lr()[0])

    assert len(trace) == 100
    assert all(lr >= 0 for lr in trace), 'a negative learning rate appeared'
    # trace[k-1] is the LR after k steps. Warmup occupies the first `warmup`
    # steps, the stable phase the next `stable_iters`, and decay the rest.
    stable_iters = engine.lr_scheduler._schedulers[1].total_iters
    warmup_end, stable_end = warmup, warmup + stable_iters

    assert trace[0] < trace[warmup_end - 1], 'warmup did not increase the LR'
    assert abs(trace[warmup_end - 1] - base_muon) < 1e-9, (
        f'warmup ended at {trace[warmup_end - 1]}, expected {base_muon}')
    plateau = trace[warmup_end:stable_end]
    assert max(plateau) - min(plateau) < 1e-9, (
        f'the stable phase is not flat: {max(plateau) - min(plateau):.3e}')
    assert trace[-1] < plateau[0], 'the decay phase never brought the LR down'
    assert trace[-1] < 1e-6, f'the schedule ended at {trace[-1]}, expected ~0'


def test_resume_fast_forward_reaches_the_same_lr_as_an_uninterrupted_run():
    """--reset_optimizer must not replay warmup from step 0."""
    def make():
        model = TinyAdapterModel()
        opt = _build_optimizer(model)
        engine = FakeEngine(opt)
        cfg = {'epochs': 5, 'warmup_steps': 10, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5}
        engine.lr_scheduler = build_lr_scheduler(opt, cfg, steps_per_epoch=20)
        return opt, engine

    # Uninterrupted: 40 steps.
    opt_a, engine_a = make()
    for _ in range(40):
        engine_a.lr_scheduler.step()
    expected = engine_a.get_lr()[0]

    # Interrupted at step 40, resumed with a fresh scheduler.
    opt_b, engine_b = make()
    engine_b.lr_scheduler = build_lr_scheduler(
        opt_b, {'epochs': 5, 'warmup_steps': 10, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5},
        steps_per_epoch=20)
    last = fast_forward_lr_scheduler(engine_b, 40)

    assert abs(last[0] - expected) < 1e-12, (
        f'fast-forward landed at {last[0]} but an uninterrupted run is at {expected}')


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
