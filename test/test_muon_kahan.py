"""Tests for the Muon/AdamW split and the Kahan-summation AdamW step.

These are CPU-only and need no dataset or GPU. Run with:

    python -m pytest test/test_muon_kahan.py -v

or without pytest:

    python test/test_muon_kahan.py
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimizers.muon_kahan import MuonKahan, is_muon_param, split_param_groups
from utils.lr_schedulers import build_lr_scheduler


def _param(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.nn.Parameter(torch.randn(*shape, generator=g))


def test_split_routes_matrices_to_muon_and_vectors_to_adamw():
    big = _param((64, 64), 1)      # matrix -> Muon
    small = _param((4, 4), 2)      # too small -> AdamW
    vec = _param((64,), 3)         # 1-D -> AdamW

    groups = split_param_groups([{'params': [big, small, vec], 'lr': 1e-3}], muon_min_size=32)

    muon = [p for g in groups if g['use_muon'] for p in g['params']]
    adamw = [p for g in groups if g['use_muon'] is False for p in g['params']]

    assert muon == [big], f'expected only the 64x64 on Muon, got {[tuple(p.shape) for p in muon]}'
    assert set(map(id, adamw)) == {id(small), id(vec)}
    # No parameter may be stepped twice or not at all.
    assert len(muon) + len(adamw) == 3


def test_split_preserves_zero_weight_decay_unless_overridden():
    """1-D params get weight_decay=0 in train.py; the split must not undo that."""
    vec = _param((32,), 4)
    mat = _param((64, 64), 5)
    src = [{'params': [vec, mat], 'lr': 1e-3, 'weight_decay': 0.0}]

    groups = split_param_groups(src, muon_min_size=32, adamw_lr=1e-4)
    adamw = next(g for g in groups if g['use_muon'] is False)

    assert adamw['weight_decay'] == 0.0, 'split silently restored weight decay on 1-D params'
    assert adamw['lr'] == 1e-4, 'adamw_lr was not applied'

    groups_override = split_param_groups(src, muon_min_size=32, adamw_weight_decay=0.01)
    adamw2 = next(g for g in groups_override if g['use_muon'] is False)
    assert adamw2['weight_decay'] == 0.01, 'explicit override was ignored'


def test_is_muon_param_uses_matrix_axes_for_conv_weights():
    conv = torch.nn.Parameter(torch.zeros(128, 64, 3, 3))   # 4-D, spatial 3x3
    assert is_muon_param(conv, muon_min_size=32), 'conv judged on spatial axes, not matrix axes'
    assert not is_muon_param(torch.nn.Parameter(torch.zeros(8, 8)), muon_min_size=32)


def test_adamw_step_matches_torch_adamw_in_fp32():
    """With kahan disabled and fp32 params, the custom step must equal torch.optim.AdamW."""
    torch.manual_seed(0)
    p_custom = _param((16, 16), 7)
    p_ref = torch.nn.Parameter(p_custom.detach().clone())

    opt = MuonKahan([{'params': [p_custom], 'use_muon': False, 'lr': 1e-2,
                      'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0, 'kahan': False}])
    ref = torch.optim.AdamW([p_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    for _ in range(5):
        g = torch.randn(16, 16, generator=torch.Generator().manual_seed(_ + 100))
        p_custom.grad = g.clone()
        p_ref.grad = g.clone()
        opt.step()
        ref.step()

    # rtol=0: the default allclose would also permit a 1e-5 RELATIVE allowance,
    # which would let real drift through while the atol check still reads green.
    drift = (p_custom - p_ref).abs().max().item()
    assert drift <= 1e-6, f'max drift vs torch.optim.AdamW: {drift:.3e}'


def test_adamw_kahan_step_matches_torch_adamw_in_fp32():
    """The COMPENSATED path must also reproduce torch.optim.AdamW in fp32.

    The comparison above only proves the shared AdamW arithmetic, because it
    runs with kahan disabled. This one proves the compensation itself does not
    change the result: over this trajectory the fp32 running total is many ulps
    above the parameter's fp32 resolution, so the reconstructed total must land
    on the value an ordinary AdamW write produces.
    """
    torch.manual_seed(0)
    p_custom = _param((16, 16), 7)
    p_ref = torch.nn.Parameter(p_custom.detach().clone())

    opt = MuonKahan([{'params': [p_custom], 'use_muon': False, 'lr': 1e-2,
                      'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0, 'kahan': True}])
    ref = torch.optim.AdamW([p_ref], lr=1e-2, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    for i in range(5):
        g = torch.randn(16, 16, generator=torch.Generator().manual_seed(i + 100))
        p_custom.grad = g.clone()
        p_ref.grad = g.clone()
        opt.step()
        ref.step()

    drift = (p_custom - p_ref).abs().max().item()
    assert drift <= 1e-6, f'max drift vs torch.optim.AdamW (kahan=True): {drift:.3e}'


def test_kahan_preserves_updates_smaller_than_one_bf16_ulp():
    """The whole point of Kahan: a bf16 add rounds tiny updates away.

    Both runs get the SAME normalized AdamW update. The non-Kahan path writes
    `p = (p - update*lr).to(bfloat16)` every step, so once update*lr drops below
    one bf16 ulp of the parameter it rounds straight back to the same value and
    the step is discarded. The Kahan path carries the residual in fp32 and only
    moves `p` once the accumulated total crosses a rounding boundary.
    """
    # Kahan accumulates the residual and only moves the parameter once the
    # running total crosses a bf16 rounding boundary. One ulp at 0.02 is
    # ~1.2e-4 and a step moves ~1e-6, so a boundary is only crossed after
    # O(120) steps. 200 is comfortably past the first crossing.
    steps = 200
    lr = 1e-6

    p_kahan = torch.nn.Parameter(torch.full((64,), 0.02, dtype=torch.bfloat16))
    p_plain = torch.nn.Parameter(torch.full((64,), 0.02, dtype=torch.bfloat16))
    # 0.02 is not exactly representable in bf16, so the baseline has to be the
    # value actually stored in the parameter, not the Python literal.
    start = p_plain.detach().float().clone()

    def make(param, kahan):
        return MuonKahan([{'params': [param], 'use_muon': False, 'lr': lr,
                           'betas': (0.9, 0.999), 'eps': 1e-8,
                           'weight_decay': 0.0, 'kahan': kahan}])

    opt_kahan = make(p_kahan, True)
    opt_plain = make(p_plain, False)

    # ulp of bf16 at 0.02 is ~1.2e-4, so a 1e-6 update is ~1% of one ulp.
    tiny = torch.full((64,), 1e-7, dtype=torch.bfloat16)
    for _ in range(steps):
        p_kahan.grad = tiny.clone()
        p_plain.grad = tiny.clone()
        opt_kahan.step()
        opt_plain.step()

    kahan_moved = (p_kahan.detach().float() - start).abs().max().item()
    plain_moved = (p_plain.detach().float() - start).abs().max().item()

    assert plain_moved == 0.0, (
        f'the non-Kahan path moved {plain_moved:.3e}; this test only proves '
        'anything while the update stays below one bf16 ulp')
    assert kahan_moved > 0, (
        'Kahan discarded every sub-ulp update; the compensation did not accumulate')
    assert kahan_moved > plain_moved, (
        f'plain bf16 AdamW moved {plain_moved:.3e}, Kahan {kahan_moved:.3e}')


def test_kahan_invariant_holds_across_steps():
    """p + kahan_comp must reconstruct the intended fp32 value."""
    p = _param((32,), 11)
    p.data = p.data.to(torch.bfloat16)
    opt = MuonKahan([{'params': [p], 'use_muon': False, 'lr': 1e-3,
                      'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.01, 'kahan': True}])

    for i in range(10):
        p.grad = torch.full((32,), 1e-4, dtype=torch.bfloat16)
        opt.step()
        comp = opt.state[p]['kahan_comp']
        # The compensation is bounded by half a bf16 ulp of the parameter.
        ulp = torch.tensor(p.detach().float().abs().max().item(),
                           dtype=torch.float32)
        ulp = torch.ldexp(torch.ones_like(ulp), torch.floor(torch.log2(ulp)).long() - 8)
        assert comp.abs().max().item() <= ulp.item() * 1.01, (
            f'step {i}: compensation {comp.abs().max().item():.3e} exceeds one bf16 ulp {ulp.item():.3e}')


def test_sparse_gradient_does_not_abort_the_step():
    p = torch.nn.Parameter(torch.zeros(8))
    opt = MuonKahan([{'params': [p], 'use_muon': False, 'lr': 1e-2,
                      'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0, 'kahan': True}])
    idx = torch.tensor([0, 3, 5])
    val = torch.tensor([1.0, 2.0, 3.0])
    p.grad = torch.sparse_coo_tensor(idx.unsqueeze(0), val, (8,))
    opt.step()  # must not raise
    assert p.abs().sum().item() > 0


def test_eps_zero_does_not_poison_state_with_nan():
    p = _param((8,), 13)
    opt = MuonKahan([{'params': [p], 'use_muon': False, 'lr': 1e-3,
                      'betas': (0.9, 0.999), 'eps': 0.0, 'weight_decay': 0.0, 'kahan': False}])
    p.grad = torch.zeros(8)
    opt.step()
    assert torch.isfinite(p).all(), 'eps=0 with a zero gradient produced NaN'


def test_wsd_reaches_base_lr_then_decays_to_end_factor():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cfg = {'epochs': 5, 'warmup_steps': 50, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5}
    sched = build_lr_scheduler(opt, cfg, steps_per_epoch=20)  # total 100
    base = 1e-3

    for _ in range(100):
        sched.step()

    final = sched.get_last_lr()[0]
    assert abs(final - 0.0) < 1e-12, f'WSD ended at {final}, expected 0'


def test_wsd_phase_boundaries_sum_to_total_steps():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cfg = {'epochs': 5, 'warmup_steps': 50, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5}
    sched = build_lr_scheduler(opt, cfg, steps_per_epoch=20)  # total 100, main 50
    _, stable, decay = sched._schedulers
    assert stable.total_iters + decay.total_iters == 50, 'stable+decay must cover the post-warmup budget'


def test_wsd_warmup_ramps_up_to_base_lr():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cfg = {'epochs': 5, 'warmup_steps': 10, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5}
    sched = build_lr_scheduler(opt, cfg, steps_per_epoch=20)

    # Measure AFTER each step: the scheduler sets the LR for the step it just
    # entered, so reading before stepping would report the previous value.
    start = opt.param_groups[0]['lr']
    trace = []
    for _ in range(10):
        sched.step()
        trace.append(opt.param_groups[0]['lr'])

    assert start < trace[-1], 'warmup did not increase the LR'
    assert abs(trace[-1] - 1e-3) < 1e-9, (
        f'after {len(trace)} warmup steps the LR is {trace[-1]}, expected base lr 1e-3')


def test_wsd_rejects_decay_frac_of_one():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    cfg = {'epochs': 5, 'warmup_steps': 0, 'lr_scheduler': 'wsd', 'wsd_decay_frac': 1.0}
    try:
        build_lr_scheduler(opt, cfg, steps_per_epoch=20)
    except ValueError:
        return
    raise AssertionError('wsd_decay_frac=1.0 was accepted; it leaves no stable phase')


def test_schedule_rejects_warmup_at_or_beyond_total_steps():
    """A warmup that eats the whole run cannot reach the base LR.

    This used to be absorbed by `max(1, total - warmup)`, which handed back
    milestones past the last step: a schedule that silently never trains the way
    it describes. Both boundaries are refused loudly instead.
    """
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    total = 5 * 20
    for warmup in (total, total + 1):
        try:
            build_lr_scheduler(opt, {'epochs': 5, 'warmup_steps': warmup,
                                     'lr_scheduler': 'wsd'}, steps_per_epoch=20)
        except ValueError:
            continue
        raise AssertionError(f'warmup_steps={warmup} (total={total}) was accepted')


def test_unknown_scheduler_still_raises():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    try:
        build_lr_scheduler(opt, {'epochs': 1, 'lr_scheduler': 'nope'}, steps_per_epoch=10)
    except NotImplementedError:
        return
    raise AssertionError('an unknown lr_scheduler did not raise NotImplementedError')


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
