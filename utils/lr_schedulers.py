"""Learning-rate schedules for train.py.

Two things the inline version in train.py got wrong, and this module fixes:

1. `warmup_steps` are the FIRST steps of `epochs * steps_per_epoch`, not extra
   steps bolted on the front. The decay therefore has to be sized on the
   remaining budget, or the run stops decaying early and sits at a non-zero LR
   for its last stretch.

2. PyTorch's `SequentialLR` cannot be nested. At a milestone it calls
   `scheduler.step(0)` on the child, and a nested `SequentialLR` inherits
   `LRScheduler.get_lr`, which raises `NotImplementedError`. WSD is therefore
   built as ONE flat `SequentialLR` of (warmup, stable, decay) rather than
   wrapping a warmup around a warmup+stable+decay chain.
"""

import torch


def _is_main_process():
    """Rank check that does not import deepspeed.

    utils.common imports deepspeed at module scope, so reusing its
    is_main_process would make this module unimportable without a working
    deepspeed install — and would make the schedule untestable on its own.
    """
    try:
        import torch.distributed as dist
        return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0
    except Exception:
        return True


def capture_base_lrs(optimizer):
    """The per-group LR a schedule is supposed to decay FROM.

    Read before the first schedule of a run is built, and handed back to
    `build_lr_scheduler` for every later phase. A phase that follows a decayed
    phase must still start from the configured base LR, not from wherever the
    previous phase happened to leave its groups.
    """
    return [group.get('initial_lr', group['lr']) for group in optimizer.param_groups]


def set_base_lrs(optimizer, base_lrs):
    """Pin every param group to its base LR before a schedule is constructed.

    PyTorch's LRScheduler reads `base_lrs` from `group['initial_lr']`, so this
    is what actually makes a freshly built schedule start where the config says
    instead of where the previous phase left off.
    """
    if base_lrs is None:
        return
    if len(base_lrs) != len(optimizer.param_groups):
        raise ValueError(
            f'base_lrs has {len(base_lrs)} entries but the optimizer has '
            f'{len(optimizer.param_groups)} param groups')
    for group, lr in zip(optimizer.param_groups, base_lrs):
        group['initial_lr'] = lr
        group['lr'] = lr


def build_lr_scheduler(optimizer, config, steps_per_epoch, base_lrs=None):
    set_base_lrs(optimizer, base_lrs)
    scheduler_type = config.get('lr_scheduler', 'constant')
    warmup_steps = config.get('warmup_steps', 0)
    total_steps = config['epochs'] * steps_per_epoch

    # A warmup at or beyond the end of the run cannot reach the base LR, and the
    # `max(1, ...)` below would then hand back milestones past the last step:
    # a schedule that silently never trains the way it says it does. Refuse it
    # loudly instead of inventing a phase budget.
    if not 0 <= warmup_steps < total_steps:
        raise ValueError(
            f'warmup_steps={warmup_steps} must satisfy '
            f'0 <= warmup_steps < epochs * steps_per_epoch ({total_steps}); '
            f'the warmup has to leave room for at least one step of the schedule.'
        )
    main_total_iters = total_steps - warmup_steps

    if scheduler_type == 'constant':
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)
    elif scheduler_type == 'linear':
        # A decay that stops at a floor instead of at zero. The default keeps
        # the original end_factor=0.0, so an existing `linear` config is
        # unaffected; `linear_end_factor = 0.025` is a decay to lr/40.
        end_factor = float(config.get('linear_end_factor', 0.0))
        if not 0.0 <= end_factor <= 1.0:
            raise ValueError(
                f'linear_end_factor must be in [0, 1], got {end_factor}')
        # LinearLR reaches `end_factor` only AFTER `total_iters` step() calls,
        # and the scheduler is stepped at the end of an optimizer step - so the
        # last LR the optimizer actually uses is the coefficient at
        # last_epoch = total_iters - 1, i.e. base * (1 - (1-end)/total_iters).
        # With total_iters = main_total_iters a run advertised as decaying to
        # lr/40 ends at lr/37.6 instead. Shortening the ramp by one makes the
        # floor exact on the final step.
        if end_factor > 0.0 and main_total_iters < 2:
            # A floor of lr*<end_factor> is a promise about the LAST step. With a
            # single step there is no ramp to place it on: LinearLR would use the
            # base LR and never reach the floor, so the run would quietly do the
            # opposite of what the config says. Refuse instead.
            raise ValueError(
                f'lr_scheduler="linear" with linear_end_factor={end_factor} needs at '
                f'least 2 optimizer steps after warmup to reach its floor, but this '
                f'schedule has {main_total_iters}. Give the stage more epochs or '
                f'steps per epoch, drop linear_end_factor, or use lr_scheduler="constant".')
        #
        # end_factor == 0.0 is left alone on purpose: it is the pre-existing
        # default, which never reached zero either (it stopped at 1/main_total_iters),
        # and changing it would silently alter every existing `linear` config.
        decay_iters = main_total_iters - 1 if end_factor > 0.0 else main_total_iters
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=decay_iters)
    elif scheduler_type == 'cosine':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=main_total_iters, eta_min=1e-6)
    elif scheduler_type == 'wsd':
        decay_frac = float(config.get('wsd_decay_frac', 0.25))
        if not 0.0 <= decay_frac < 1.0:
            # decay_frac == 1.0 would leave no stable phase at all; the phase
            # boundaries would then overshoot the run by one step.
            raise ValueError(
                f'wsd_decay_frac must be in [0, 1) to leave a stable phase, got {decay_frac}')
        end_factor = float(config.get('wsd_end_factor', 0.0))

        decay_iters = max(1, int(main_total_iters * decay_frac))
        stable_iters = max(1, main_total_iters - decay_iters)

        stable_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=stable_iters)
        decay_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=end_factor, total_iters=decay_iters)

        if warmup_steps > 0:
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, stable_scheduler, decay_scheduler],
                milestones=[warmup_steps, warmup_steps + stable_iters])
        else:
            lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[stable_scheduler, decay_scheduler],
                milestones=[stable_iters])

        if _is_main_process():
            print(f'WSD schedule: warmup={warmup_steps}, stable={stable_iters}, '
                  f'decay={decay_iters} (total {total_steps})')
    else:
        raise NotImplementedError(f'Unknown lr_scheduler: {scheduler_type}')

    # WSD already contains its warmup phase in the flat chain above; re-wrapping
    # would reintroduce the nested-SequentialLR crash.
    if warmup_steps > 0 and scheduler_type != 'wsd':
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1/warmup_steps, total_iters=warmup_steps)
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, lr_scheduler],
            milestones=[warmup_steps])

    return lr_scheduler


def fast_forward_lr_scheduler(model_engine, steps):
    """Advance a freshly built scheduler by `steps` already-completed steps.

    Needed when resuming with --reset_optimizer: DeepSpeed does not restore the
    scheduler in that mode, so a brand-new schedule (last_epoch=0) would replay
    warmup from the beginning no matter how far into training we are.
    """
    steps = max(0, int(steps))
    if _is_main_process():
        print(f'Fast-forwarding lr_scheduler by {steps} completed steps')
    for _ in range(steps):
        model_engine.lr_scheduler.step()
    return model_engine.lr_scheduler.get_last_lr()
