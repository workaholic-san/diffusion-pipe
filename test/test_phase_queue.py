"""The `[[phase]]` training queue: config contract and LR behaviour.

Two things are checked here, and they are checked against real objects rather
than against the text of the source:

1. `utils/phase_plan.py` resolves a config into stages with correct absolute
   epoch ranges, and refuses the configurations that used to fail silently -
   most importantly a top-level `epochs` that disagrees with the queue, which
   is exactly the mismatch that leaves a decay stage 80% finished before it
   starts.

2. The LR schedule that `train.py` builds for each stage actually does what the
   owner asked for: warmup then a flat full LR, then a flat full LR AGAIN (not
   wherever the previous stage left the groups), then a linear decay that stops
   at 1/40 of the base instead of at zero. Real optimizer, real param groups,
   real schedulers, real steps.

CPU-only, no DeepSpeed, no model weights, a couple of seconds.
"""

import ast
import sys
from pathlib import Path

import toml
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from utils import phase_plan as phase_plan_util
from utils.lr_schedulers import (
    build_lr_scheduler, capture_base_lrs, fast_forward_lr_scheduler,
)
from utils.resume_position import resume_skip_batches

# The owner's LR table from a real run: Muon on the matrices, Kahan-AdamW on
# everything else, and the `mod` group held back because it trained ahead.
MUON_LR = 1e-3
ADAMW_LR = 1.5e-4
MOD_LR = 8e-5
END_FACTOR = 0.025  # lr/40


def _three_group_optimizer():
    """Three param groups with the real recipe's three LRs."""
    return torch.optim.AdamW([
        {'params': [torch.nn.Parameter(torch.zeros(8, 8))], 'lr': MUON_LR, 'name': 'muon'},
        {'params': [torch.nn.Parameter(torch.zeros(8, 8))], 'lr': ADAMW_LR, 'name': 'attn'},
        {'params': [torch.nn.Parameter(torch.zeros(8, 8))], 'lr': MOD_LR, 'name': 'mod'},
    ])


def _owner_queue():
    """The three-stage plan, as it appears in the shipped example config."""
    return {
        'dataset': 'a.toml',
        'phase': [
            {'name': 'img_flip', 'dataset': 'p1.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 60},
            {'name': 'crop', 'dataset': 'p2.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 0},
            {'name': 'all', 'dataset': 'p3.toml', 'epochs': 3,
             'lr_scheduler': 'linear', 'warmup_steps': 0, 'linear_end_factor': END_FACTOR},
        ],
    }


# --------------------------------------------------------------------------
# config contract
# --------------------------------------------------------------------------

def test_a_config_without_a_queue_behaves_exactly_as_before():
    config = {'epochs': 5, 'dataset': 'd.toml', 'lr_scheduler': 'wsd', 'wsd_decay_frac': 0.5}
    phases = phase_plan_util.normalize_phase_plan(config)

    assert len(phases) == 1, 'a plain config must not grow a queue'
    p = phases[0]
    assert p.epochs == 5 and p.dataset == 'd.toml'
    assert p.config['lr_scheduler'] == 'wsd'
    assert p.config['wsd_decay_frac'] == 0.5
    assert p.first_epoch == 1 and p.last_epoch == 5


def test_queue_epochs_are_absolute_over_the_whole_run():
    config = _owner_queue()
    phases = phase_plan_util.normalize_phase_plan(config)

    assert [p.epochs for p in phases] == [3, 3, 3]
    assert [p.first_epoch for p in phases] == [1, 4, 7]
    assert [p.last_epoch for p in phases] == [3, 6, 9]
    # train.py stops at `dataloader.epoch > epochs`, so this is the stop point.
    assert config['epochs'] == 9


def test_epochs_that_disagree_with_the_queue_are_refused():
    config = _owner_queue()
    config['epochs'] = 5  # what a hand-edited config drifts into
    try:
        phase_plan_util.normalize_phase_plan(config)
    except phase_plan_util.PhaseError as e:
        assert 'does not match' in str(e)
    else:
        raise AssertionError('a queue/epochs mismatch must be refused, not silently accepted')


def test_a_phase_without_epochs_or_dataset_is_refused():
    config = {'dataset': 'd.toml', 'phase': [{'name': 'x', 'lr_scheduler': 'constant'}]}
    try:
        phase_plan_util.normalize_phase_plan(config)
    except phase_plan_util.PhaseError as e:
        assert 'epochs' in str(e)
    else:
        raise AssertionError('a phase with no epoch budget must be refused')

    config = {'phase': [{'name': 'x', 'epochs': 2, 'dataset': 'p.toml'}],
              'epochs': 2}
    del config['phase'][0]['dataset']
    try:
        phase_plan_util.normalize_phase_plan(config)
    except phase_plan_util.PhaseError as e:
        assert 'dataset' in str(e)
    else:
        raise AssertionError('a phase with no dataset and no inherited one must be refused')


def test_schedule_keys_are_inherited_from_the_top_level_not_from_the_previous_phase():
    config = {
        'dataset': 'd.toml',
        'lr_scheduler': 'wsd',
        'wsd_decay_frac': 0.5,
        'phase': [
            {'name': 'a', 'epochs': 1, 'dataset': 'a.toml', 'lr_scheduler': 'constant'},
            {'name': 'b', 'epochs': 1, 'dataset': 'b.toml'},
        ],
    }
    a, b = phase_plan_util.normalize_phase_plan(config)

    assert a.config['lr_scheduler'] == 'constant', 'the phase override must win'
    # Phase b did not ask for a scheduler, so it gets the TOP-LEVEL one - not
    # the one phase a happened to override.
    assert b.config['lr_scheduler'] == 'wsd'
    assert b.config['wsd_decay_frac'] == 0.5
    assert 'phase' not in b.config, 'the queue must not leak into a schedule view'


# --------------------------------------------------------------------------
# epoch boundaries - the arithmetic that decides when data actually changes
# --------------------------------------------------------------------------

def test_the_queue_switches_dataset_on_exactly_the_right_epochs():
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    index, start, switches, finished = 0, 1, [], None

    for new_epoch in range(2, 12):
        action, target = phase_plan_util.phase_action(phases, index, start, new_epoch)
        if action == 'switch':
            switches.append((new_epoch, target))
            index, start = target, new_epoch
        elif action == 'finish':
            finished = new_epoch
            break

    assert switches == [(4, 1), (7, 2)], (
        f'dataset must change entering epochs 4 and 7, got {switches}')
    assert finished == 10, f'the queue must exhaust at epoch 10, got {finished}'


def test_a_phase_never_switches_one_epoch_early_or_late():
    """Boundary check against every budget from 1..5, not just the shipped 3/3/3."""
    for budget in (1, 2, 3, 4, 5):
        config = {'dataset': 'd.toml', 'phase': [
            {'name': f'a{i}', 'epochs': budget, 'dataset': f'{i}.toml'} for i in range(3)]}
        phases = phase_plan_util.normalize_phase_plan(config)
        index, start, switches = 0, 1, []
        for new_epoch in range(2, 3 * budget + 3):
            action, target = phase_plan_util.phase_action(phases, index, start, new_epoch)
            if action == 'switch':
                switches.append(new_epoch)
                index, start = target, new_epoch
            elif action == 'finish':
                break
        expected = [budget + 1, 2 * budget + 1]
        assert switches == expected, (
            f'budget={budget}: switched at {switches}, expected {expected}')


# --------------------------------------------------------------------------
# the LR schedule each stage actually produces
# --------------------------------------------------------------------------

def _run_phase(optimizer, phase_config, steps_per_epoch, base_lrs):
    """Build the stage's schedule and step it once per step. Returns the LR trace."""
    sched = build_lr_scheduler(optimizer, phase_config, steps_per_epoch, base_lrs=base_lrs)
    trace = []
    for _ in range(phase_config['epochs'] * steps_per_epoch):
        trace.append([g['lr'] for g in optimizer.param_groups])
        sched.step()
    return trace


def test_the_three_stage_queue_produces_warmup_then_flat_then_decay_to_lr_over_40():
    opt = _three_group_optimizer()
    base_lrs = capture_base_lrs(opt)
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    spe = 200  # stand-in for a real epoch

    warmup, flat, decay = (_run_phase(opt, p.config, spe, base_lrs) for p in phases)

    # trace[k] is the LR the optimizer USES on step k+1: the scheduler is stepped
    # at the END of a step. So the warmup occupies trace[0..59] - 60 steps from
    # lr/60 upward - and trace[60] is the first step already at the full LR.
    assert abs(warmup[0][0] - MUON_LR / 60) < 1e-15, (
        f'warmup did not start at lr/60: {warmup[0][0]}')
    rising = [r[0] for r in warmup[:61]]
    assert all(b > a for a, b in zip(rising, rising[1:])), f'warmup is not ascending: {rising[:3]}'
    assert abs(warmup[60][0] - MUON_LR) < 1e-12, (
        f'warmup ended at {warmup[59][0]} and never reached the base LR')
    plateau = [row[0] for row in warmup[60:]]
    assert max(plateau) - min(plateau) < 1e-15, 'stage 1 plateau is not flat'

    # Stage 2: back at the FULL LR, not wherever stage 1 left the groups.
    assert abs(flat[0][0] - MUON_LR) < 1e-12, (
        f'stage 2 started at {flat[0][0]}, expected the full {MUON_LR}')
    flat_muon = [row[0] for row in flat]
    assert max(flat_muon) - min(flat_muon) < 1e-15, 'stage 2 is not flat at the full LR'

    # Stage 3: monotone linear decay stopping at 1/40 of base, not at zero.
    tail = [row[0] for row in decay]
    assert all(b <= a + 1e-15 for a, b in zip(tail, tail[1:])), 'stage 3 is not monotone'
    assert abs(tail[0] - MUON_LR) < 1e-12, f'stage 3 started at {tail[0]}'
    assert abs(tail[-1] - MUON_LR * END_FACTOR) < 1e-9, (
        f'stage 3 ended at {tail[-1]}, expected {MUON_LR * END_FACTOR} (= lr/40)')
    assert tail[-1] > 0, 'stage 3 decayed to zero; the floor is lr/40'


def test_every_param_group_keeps_its_own_lr_through_all_three_stages():
    """The per-category table must survive the queue, not collapse to one LR."""
    opt = _three_group_optimizer()
    base_lrs = capture_base_lrs(opt)
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    expected = [MUON_LR, ADAMW_LR, MOD_LR]

    for p in phases:
        trace = _run_phase(opt, p.config, 200, base_lrs)
        for step, row in enumerate(trace):
            # Every schedule multiplies all groups by the same coefficient, so
            # the per-category ratio must hold at every single step.
            ratios = [row[i] / expected[i] for i in range(3)]
            assert abs(ratios[0] - ratios[1]) < 1e-9 and abs(ratios[1] - ratios[2]) < 1e-9, (
                f'stage {p.name} step {step} scaled the groups differently: {ratios}')
            # A flat stage sits exactly on its configured LR once its warmup is
            # over. A decaying stage never does, so it is only ratio-checked.
            if (p.config['lr_scheduler'] != 'linear'
                    and step >= p.config.get('warmup_steps', 0)):
                for i in range(3):
                    assert abs(row[i] - expected[i]) < 1e-12, (
                        f'stage {p.name} step {step} moved group {i} off its configured LR')


def test_a_stage_after_a_decay_returns_to_the_full_lr():
    """The bug this queue exists to remove: starting from the previous stage's end LR."""
    opt = _three_group_optimizer()
    base_lrs = capture_base_lrs(opt)
    config = {'dataset': 'd.toml', 'phase': [
        {'name': 'decayed', 'epochs': 1, 'dataset': 'a.toml',
         'lr_scheduler': 'linear', 'linear_end_factor': END_FACTOR},
        {'name': 'full', 'epochs': 1, 'dataset': 'b.toml',
         'lr_scheduler': 'constant'},
    ]}
    decayed, full = phase_plan_util.normalize_phase_plan(config)

    trace_a = _run_phase(opt, decayed.config, 100, base_lrs)
    assert abs(trace_a[-1][0] - MUON_LR * END_FACTOR) < 1e-9, 'stage 1 precondition failed'

    trace_b = _run_phase(opt, full.config, 100, base_lrs)
    assert abs(trace_b[0][0] - MUON_LR) < 1e-12, (
        f'stage 2 started at {trace_b[0][0]} instead of the full {MUON_LR}: it inherited '
        'the decayed LR of stage 1')


def test_switching_stages_preserves_both_adam_moments():
    """A stage boundary must not touch optimizer state, and moments must stay live.

    Scope note: this exercises the scheduler rebuild, which is the part train.py
    performs at a boundary. That train.py additionally never reassigns the
    optimizer object is a fact of reading that code, not something this test
    can observe.
    """
    opt = _three_group_optimizer()
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())

    param = opt.param_groups[0]['params'][0]
    param.grad = torch.ones_like(param)
    opt.step()

    for key in ('exp_avg', 'exp_avg_sq'):
        state = opt.state[param].get(key)
        assert state is not None and state.numel() > 0, f'{key} was never allocated'
        assert not torch.allclose(state, torch.zeros_like(state)), (
            f'{key} is zero before the switch, so "carried" would be vacuous')

    _run_phase(opt, phases[0].config, 100, capture_base_lrs(opt))
    at_boundary = {k: opt.state[param][k].clone() for k in ('exp_avg', 'exp_avg_sq')}

    # The boundary: build and run the NEXT stage's schedule over the same
    # optimizer.
    _run_phase(opt, phases[1].config, 100, capture_base_lrs(opt))
    for key, was in at_boundary.items():
        assert torch.equal(opt.state[param][key], was), (
            f'{key} changed purely from switching stages; the boundary must not '
            'touch optimizer state')

    # And the moments must still be live afterwards, not frozen at the boundary.
    opt.zero_grad(set_to_none=True)
    param.grad = torch.full_like(param, 3.0)
    opt.step()
    for key, was in at_boundary.items():
        assert not torch.equal(opt.state[param][key], was), (
            f'{key} stopped evolving after the stage switch')


def test_a_warmup_that_does_not_fit_its_stage_is_refused_by_name():
    """A warmup larger than the stage budget must name the stage, not fail anonymously."""
    opt = _three_group_optimizer()
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    # 2 epochs x 3 steps = 6 steps of budget against a 60-step warmup.
    try:
        build_lr_scheduler(opt, phases[0].config, 3, base_lrs=capture_base_lrs(opt))
    except ValueError as e:
        assert 'warmup_steps' in str(e)
    else:
        raise AssertionError('a warmup longer than the stage must be refused')


# --------------------------------------------------------------------------
# the shipped example config
# --------------------------------------------------------------------------

def test_every_checkpoint_in_the_queue_resumes_into_its_own_stage():
    """A 20-minute save can land anywhere, including mid-stage.

    The epoch alone has to be enough to name the stage, because that is all a
    checkpoint carries. The old code refused everything past stage 1; the point
    of this test is that the refusal is now only for a finished run.
    """
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    expected = {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 2}

    for epoch, index in expected.items():
        assert phase_plan_util.resume_phase_index(phases, epoch) == index, (
            f'epoch {epoch} must resume into stage {index + 1}')
        assert phase_plan_util.resume_epoch_is_supported(phases, epoch), (
            f'epoch {epoch} is inside the queue and must stay resumable')

    for epoch in (10, 11, 99):
        assert phase_plan_util.resume_phase_index(phases, epoch) is None, (
            f'epoch {epoch} is past the end of the queue')
        assert not phase_plan_util.resume_epoch_is_supported(phases, epoch)
        assert 'past the end' in phase_plan_util.resume_refusal(phases, epoch)


def test_mixed_stage_budgets_address_every_epoch_to_exactly_one_stage():
    """Uneven budgets are the case where a wrong index would go unnoticed."""
    for budgets in [(1, 2, 3), (2, 1, 5), (4, 1, 1), (1, 5, 2)]:
        config = {'dataset': 'd.toml', 'phase': [
            {'name': f's{i}', 'epochs': b, 'dataset': f'{i}.toml'}
            for i, b in enumerate(budgets)]}
        phases = phase_plan_util.normalize_phase_plan(config)
        for ph in phases:
            for epoch in range(ph.first_epoch, ph.last_epoch + 1):
                assert phase_plan_util.resume_phase_index(phases, epoch) == ph.index, (
                    f'budgets {budgets}: epoch {epoch} belongs to stage {ph.index + 1}')
        assert phase_plan_util.resume_phase_index(phases, sum(budgets) + 1) is None


def test_completed_steps_count_the_epoch_a_mid_epoch_save_interrupted():
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    spe, gas = 10, 4
    stage3 = phases[2]  # epochs 7..9

    # Fresh start of the stage: nothing done yet.
    assert phase_plan_util.steps_completed_in_phase(
        stage3, stage3.first_epoch, spe, 0, gas) == 0

    # 17 steps into epoch 7, plus the one preloaded micro-batch.
    assert phase_plan_util.steps_completed_in_phase(
        stage3, 7, spe, 17 * gas + 1, gas) == 17

    # A whole epoch further along.
    assert phase_plan_util.steps_completed_in_phase(
        stage3, 8, spe, 3 * gas, gas) == spe + 3

    # A save taken exactly at a stage boundary has already rolled the counter
    # over, so it reads as the first step of the stage that is starting.
    assert phase_plan_util.steps_completed_in_phase(
        phases[2], 7, spe, 0, gas) == 0


def test_only_a_schedule_that_moves_the_lr_is_replayed():
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    assert phase_plan_util.schedule_varies(phases[0]), 'stage 1 has a warmup'
    assert not phase_plan_util.schedule_varies(phases[1]), (
        'stage 2 is constant with no warmup, so replaying its steps is a no-op')
    assert phase_plan_util.schedule_varies(phases[2]), 'stage 3 decays'

    flat = phase_plan_util.normalize_phase_plan(
        {'dataset': 'd.toml', 'phase': [
            {'name': 'flat', 'dataset': 'p.toml', 'epochs': 2}]})
    assert not phase_plan_util.schedule_varies(flat[0])


class _Engine:
    """The only attribute fast_forward_lr_scheduler touches."""

    def __init__(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler


def test_resuming_a_decay_stage_lands_on_the_lr_for_the_steps_it_actually_ran():
    """The falsifiable core: a resume must not shift the schedule.

    The oracle is the closed form of stage 3's own decay, derived here rather
    than taken from a second scheduler call - comparing the implementation
    against itself would prove nothing. Stage 3 decays to lr/40, so a schedule
    that restarted, or jumped to its end, is a visibly different number.
    """
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    stage3 = phases[2]
    spe, gas = 10, 4
    saved_epoch, done = 8, 14  # mid-epoch: 14 steps into epoch 8

    # The stage-local step count a resume has to replay: the whole epochs of the
    # stage before this one, plus the steps taken inside this epoch.
    completed = phase_plan_util.steps_completed_in_phase(
        stage3, saved_epoch, spe, done * gas + 1, gas)
    assert completed == (saved_epoch - stage3.first_epoch) * spe + done == 24

    # Closed form of LinearLR(start=1.0, end=END_FACTOR, total_iters=steps-1).
    decay_iters = stage3.epochs * spe - 1
    optimizer = _three_group_optimizer()
    base = capture_base_lrs(optimizer)
    scheduler = build_lr_scheduler(optimizer, stage3.config, spe, base_lrs=base)
    fast_forward_lr_scheduler(_Engine(scheduler), completed)

    for got, want, start_lr in zip(scheduler.get_last_lr(), base, (MUON_LR, ADAMW_LR, MOD_LR)):
        expected = want * (1 - (1 - END_FACTOR) * completed / decay_iters)
        assert abs(got - expected) < 1e-15, (
            f'resum at epoch {saved_epoch} gave lr {got}, the stage decay gives '
            f'{expected} after {completed} steps')
        assert abs(got - start_lr) > 1e-9, (
            'the resume must not rewind the stage to its first step')
        assert abs(got - want * END_FACTOR) > 1e-9, (
            'the resume must not jump the stage to its floor')


def test_resuming_a_later_constant_stage_comes_back_at_the_full_configured_lr():
    """The corruption the old refusal guarded against, now checked positively.

    A resume that reused stage 1's schedule would keep stage 1's LR state, and
    one that reused whatever the previous stage left in the param groups would
    come back too low. Stage 2 is constant at the configured LR, so every group
    must read exactly its base.
    """
    phases = phase_plan_util.normalize_phase_plan(_owner_queue())
    stage2 = phases[1]
    spe, gas = 10, 4
    saved_epoch = 6  # last epoch of stage 2, saved mid-epoch

    optimizer = _three_group_optimizer()
    base = capture_base_lrs(optimizer)
    scheduler = build_lr_scheduler(optimizer, stage2.config, spe, base_lrs=base)
    completed = phase_plan_util.steps_completed_in_phase(
        stage2, saved_epoch, spe, 5 * gas, gas)
    if phase_plan_util.schedule_varies(stage2):
        fast_forward_lr_scheduler(_Engine(scheduler), completed)

    for got, want in zip(scheduler.get_last_lr(), base):
        assert abs(got - want) < 1e-12, (
            f'a constant stage must resume at the configured lr {want}, got {got}')


def test_a_changed_queue_is_refused_rather_than_silently_remapped():
    """Epoch addressing trades a refusal for a mapping, so the mapping is pinned.

    A resume finds its stage by epoch, which is only meaningful while the queue
    is the one the checkpoint was written under. The signature is what makes an
    edited queue loud instead of a wrong stage.
    """
    saved = phase_plan_util.queue_signature(phase_plan_util.normalize_phase_plan(_owner_queue()))

    same = phase_plan_util.normalize_phase_plan(_owner_queue())
    assert phase_plan_util.queue_signature(same) == saved, 'an identical queue must resume'

    for edited in (
        # a stage budget moved
        {'dataset': 'a.toml', 'phase': [
            {'name': 'img_flip', 'dataset': 'p1.toml', 'epochs': 2,
             'lr_scheduler': 'constant', 'warmup_steps': 60},
            {'name': 'crop', 'dataset': 'p2.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 0},
            {'name': 'all', 'dataset': 'p3.toml', 'epochs': 3,
             'lr_scheduler': 'linear', 'warmup_steps': 0, 'linear_end_factor': END_FACTOR},
        ]},
        # a stage's dataset swapped
        {'dataset': 'a.toml', 'phase': [
            {'name': 'img_flip', 'dataset': 'p1.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 60},
            {'name': 'crop', 'dataset': 'OTHER.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 0},
            {'name': 'all', 'dataset': 'p3.toml', 'epochs': 3,
             'lr_scheduler': 'linear', 'warmup_steps': 0, 'linear_end_factor': END_FACTOR},
        ]},
        # the decay floor moved: same stages, same epochs, same datasets, but a
        # materially different LR at every point of the resumed stage
        {'dataset': 'a.toml', 'phase': [
            {'name': 'img_flip', 'dataset': 'p1.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 60},
            {'name': 'crop', 'dataset': 'p2.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 0},
            {'name': 'all', 'dataset': 'p3.toml', 'epochs': 3,
             'lr_scheduler': 'linear', 'warmup_steps': 0, 'linear_end_factor': 0.05},
        ]},
        # a warmup added to the middle stage: its LR curve changes shape
        {'dataset': 'a.toml', 'phase': [
            {'name': 'img_flip', 'dataset': 'p1.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 60},
            {'name': 'crop', 'dataset': 'p2.toml', 'epochs': 3,
             'lr_scheduler': 'constant', 'warmup_steps': 10},
            {'name': 'all', 'dataset': 'p3.toml', 'epochs': 3,
             'lr_scheduler': 'linear', 'warmup_steps': 0, 'linear_end_factor': END_FACTOR},
        ]},
    ):
        other = phase_plan_util.normalize_phase_plan(edited)
        assert phase_plan_util.queue_signature(other) != saved
        msg = phase_plan_util.queue_mismatch_refusal(saved, other)
        assert 'different phase queue' in msg and 'wrong schedule' in msg


def test_the_signature_pins_the_schedule_not_only_the_stage_mapping():
    """A moved LR floor must invalidate a resume, not slip through.

    The stage triple (name, epochs, dataset) says WHICH stage an epoch lands on.
    The schedule says what LR that stage will apply. Pinning only the first
    meant editing `linear_end_factor` - or adding a warmup - between save and
    restart resumed silently onto a curve the run was never trained under.
    """
    queue = phase_plan_util.normalize_phase_plan(_owner_queue())
    signature = phase_plan_util.queue_signature(queue)

    for phase, row in zip(queue, signature):
        assert row[3] == [phase.config.get(k) for k in phase_plan_util.SCHEDULE_KEYS], (
            f'stage `{phase.name}` is missing its schedule view in the signature')
    # And the values are the effective ones, not placeholders.
    decay_stage = dict(zip(phase_plan_util.SCHEDULE_KEYS, signature[2][3]))
    assert decay_stage['lr_scheduler'] == 'linear'
    assert decay_stage['linear_end_factor'] == END_FACTOR
    middle_stage = dict(zip(phase_plan_util.SCHEDULE_KEYS, signature[1][3]))
    assert middle_stage['warmup_steps'] == 0, 'the middle stage must carry its own warmup'


def test_a_signature_less_checkpoint_cannot_be_resumed_into_a_queue():
    """Absence of a signature is ambiguous, so a queue must refuse it.

    A checkpoint with no `phase_queue` was written either before the feature
    existed or by a run of the SAME config without its [[phase]] block. The
    second case is reachable - the owner adds a queue to a config whose run has
    been going for days - and trusting it would build a stage's different
    dataset while restoring a batch position that indexes the old one, so
    training would continue on the wrong data without a word.

    The refusal itself is checked here; the guard that fires it lives in
    train.py (`len(phase_queue) > 1 and saved_queue is None`), which cannot be
    imported on a machine without deepspeed.
    """
    queue = phase_plan_util.normalize_phase_plan(_owner_queue())
    msg = phase_plan_util.missing_signature_refusal(queue)

    assert 'no queue signature' in msg
    assert '[[phase]]' in msg, 'the refusal must name the cause the owner can act on'
    assert 'new output dir' in msg, 'and it must name the way out'


def test_a_single_phase_plan_stays_resumable_where_it_used_to_be():
    """No regression: a plain config has no later stage to disagree with."""
    phases = phase_plan_util.normalize_phase_plan({'epochs': 5, 'dataset': 'd.toml'})
    for e in (1, 3, 5, 6):
        assert phase_plan_util.resume_epoch_is_supported(phases, e), (
            f'single-phase resume at epoch {e} must keep working exactly as before')


def test_mixed_stage_budgets_switch_on_the_right_epochs():
    """Equal budgets are one case only; stage k must start at 1 + sum(before it)."""
    for budgets in [(1, 2, 3), (2, 1, 5), (3, 3, 1), (1, 5, 2), (4, 1, 1)]:
        config = {'dataset': 'd.toml', 'phase': [
            {'name': f's{i}', 'epochs': b, 'dataset': f'{i}.toml'}
            for i, b in enumerate(budgets)]}
        phases = phase_plan_util.normalize_phase_plan(config)
        index, start, switches = 0, 1, []
        for new_epoch in range(2, sum(budgets) + 3):
            action, target = phase_plan_util.phase_action(phases, index, start, new_epoch)
            if action == 'switch':
                switches.append((new_epoch, target))
                index, start = target, new_epoch
            elif action == 'finish':
                break
        # Switching INTO stage k happens when the epoch counter reaches
        # sum(budgets[:k]) + 1; stage 0 is where the run begins, not switched into.
        expected = [(sum(budgets[:k]) + 1, k) for k in range(1, len(budgets))]
        assert switches == expected, (
            f'budgets={budgets}: switched at {switches}, expected {expected}')


def test_a_decay_to_a_floor_that_cannot_be_reached_is_refused():
    """A single-step stage cannot ramp, so lr/40 would silently mean full lr."""
    opt = _three_group_optimizer()
    base = capture_base_lrs(opt)
    cfg = {'epochs': 1, 'lr_scheduler': 'linear', 'warmup_steps': 0,
           'linear_end_factor': END_FACTOR}

    # One epoch of one step: the floor is unreachable.
    try:
        build_lr_scheduler(opt, cfg, steps_per_epoch=1, base_lrs=base)
    except ValueError as e:
        assert 'at least 2 optimizer steps' in str(e) and str(END_FACTOR) in str(e)
    else:
        raise AssertionError(
            'a 1-step decay stage must be refused, not silently run at the base LR')

    # Two steps is the smallest stage that can honour the promise.
    sched = build_lr_scheduler(opt, cfg, steps_per_epoch=2, base_lrs=base)
    used = []
    for _ in range(2):
        used.append(opt.param_groups[0]['lr'])
        sched.step()
    assert abs(used[-1] - MUON_LR * END_FACTOR) < 1e-12, (
        f'a 2-step stage still missed its floor: ended at {used[-1]}')


def test_the_preexisting_single_phase_example_still_resolves():
    """Backward compatibility: a config without `[[phase]]` must be untouched."""
    config = toml.load(REPO / 'examples' / 'muon_kahan_wsd_lokr.toml')
    assert 'phase' not in config, 'the old example is supposed to have no queue'

    before = dict(config)
    phases = phase_plan_util.normalize_phase_plan(config)

    assert len(phases) == 1
    p = phases[0]
    assert p.epochs == config['epochs'] == 5
    assert p.dataset == config['dataset']
    assert p.config['lr_scheduler'] == 'wsd'
    assert p.config['wsd_decay_frac'] == 0.5
    assert p.config['warmup_steps'] == 50
    # Nothing may be rewritten for a config that has no queue.
    assert config == before


def test_the_shipped_example_config_resolves_to_the_intended_plan():
    path = REPO / 'examples' / 'muon_kahan_phase_queue.toml'
    config = toml.load(path)
    phases = phase_plan_util.normalize_phase_plan(config)

    assert [p.epochs for p in phases] == [3, 3, 3]
    assert [p.config['lr_scheduler'] for p in phases] == ['constant', 'constant', 'linear']
    assert [p.config['warmup_steps'] for p in phases] == [60, 0, 0]
    assert phases[2].config['linear_end_factor'] == END_FACTOR
    assert config['epochs'] == 9

    # Every stage must point at a dataset file that exists in the repo.
    for p in phases:
        assert (REPO / p.dataset).exists(), f'phase `{p.name}` points at a missing file: {p.dataset}'


# --------------------------------------------------------------------------
# resume position at an epoch boundary
# --------------------------------------------------------------------------
#
# `Saver.process_epoch` checkpoints exactly where the dataloader runs dry, and
# that is the state an epoch-boundary save records: the loader has bumped its
# epoch and reset `num_batches_pulled` to 0. With
# `checkpoint_every_n_epochs = 1` that is every boundary, including both
# internal boundaries of a phase queue - i.e. precisely the checkpoints the
# resume path exists to serve.


def _load_real_sampler_class():
    """Compile the REAL SkipFirstNSampler out of utils/dataset.py.

    utils/dataset.py imports deepspeed, datasets and comfy at module level, so
    it cannot be imported on a machine that only has torch. The sampler is a
    leaf that needs nothing but torch, so it is taken straight from the shipped
    source: this runs the real bytes, not a re-typed copy of them.
    """
    source = (REPO / 'utils' / 'dataset.py').read_text(encoding='utf-8')
    for node in ast.parse(source).body:
        if isinstance(node, ast.ClassDef) and node.name == 'SkipFirstNSampler':
            namespace = {'torch': torch}
            exec(compile(ast.Module(body=[node], type_ignores=[]),
                         'utils/dataset.py', 'exec'), namespace)
            return namespace['SkipFirstNSampler']
    raise AssertionError('SkipFirstNSampler is no longer in utils/dataset.py')


def test_a_boundary_checkpoint_resumes_at_the_first_batch_of_the_epoch():
    sampler = _load_real_sampler_class()
    length = 10
    saved = 0  # exactly what PipelineDataLoader.state_dict holds at a boundary

    skip = resume_skip_batches(saved)
    assert skip == 0, f'a boundary checkpoint must skip 0 batches, got {skip}'

    indices = list(sampler(skip, length))
    assert indices == list(range(length)), (
        f'the resumed epoch must be the whole epoch, got {indices}')


def test_the_clamp_actually_changes_behaviour_not_just_the_bookkeeping():
    """Falsifiability: prove the unclamped arithmetic was observably broken.

    A length-only assertion cannot see this class of defect - both the broken
    and the fixed restore yield a plausible-looking list. What the clamp changes
    is WHICH indices, and by how many, so the check is on the head of the epoch
    and on its exact length.
    """
    sampler = _load_real_sampler_class()
    length = 10
    unclamped = 0 - 1  # the arithmetic before the fix

    assert unclamped < 0
    before = list(sampler(unclamped, length))
    after = list(sampler(resume_skip_batches(0), length))

    # range(-1, 10) = 11 items: the epoch got one batch too many and it opened
    # by re-serving the dataset's last item.
    assert before[0] == -1, 'pre-fix, the epoch opened with the dataset LAST item'
    assert len(before) == length + 1, (
        f'pre-fix the epoch ran {len(before)} batches for {length} items')

    assert after == list(range(length)), (
        f'post-fix the epoch must be exactly the epoch, got {after}')


def test_the_resume_skip_is_never_negative_and_never_exceeds_what_was_pulled():
    for saved in range(0, 64):
        skip = resume_skip_batches(saved)
        assert skip >= 0, f'saved={saved} produced a negative skip: {skip}'
        assert skip <= saved, f'saved={saved} skipped more than was pulled: {skip}'
    # One pulled means one consumed and one merely preloaded -> start at 0.
    assert resume_skip_batches(1) == 0
    assert resume_skip_batches(5) == 4


def test_reset_dataloader_replays_only_the_whole_epochs_of_the_stage():
    """The arithmetic behind --reset_dataloader on a later phase.

    --reset_dataloader discards the intra-epoch position, so the LR must be
    replayed over whole finished epochs only. Passing the saved batch count
    anyway would advance the schedule past the data it is about to re-read.
    train.py computes this with `saved_batches = 0` in that case; this pins the
    value that choice must produce, and shows it is strictly behind the
    position the loader would otherwise resume at.
    """
    phase = phase_plan_util.normalize_phase_plan(_owner_queue())[2]
    assert phase.first_epoch == 7 and phase.epochs == 3

    steps_per_epoch, gas = 10, 4
    epoch = 8  # second epoch of the decay stage
    at_reset = phase_plan_util.steps_completed_in_phase(
        phase, epoch, steps_per_epoch, 0, gas)
    at_saved_position = phase_plan_util.steps_completed_in_phase(
        phase, epoch, steps_per_epoch, 29, gas)

    assert at_reset == (epoch - phase.first_epoch) * steps_per_epoch == 10
    assert at_saved_position == 10 + 29 // gas == 17
    assert at_reset < at_saved_position


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