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


def build_lr_scheduler(optimizer, config, steps_per_epoch):
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
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.0, total_iters=main_total_iters)
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
