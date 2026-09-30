"""A queue of training phases inside one process.

train.py builds its dataset once at startup, sizes the LR schedule from
`epochs * steps_per_epoch`, and stops when the dataloader's epoch counter
passes `epochs`. That describes exactly one regime: one dataset, one schedule,
one stop point. A staged fine-tune - train on one mixture, switch to another,
then decay the LR to a floor - cannot be written in it.

Running the stages as separate processes instead costs a manual restart between
them, and each restart then has to rebuild its schedule against a global step
counter that no longer matches the stage it is starting: the new schedule is
sized for `epochs * steps_per_epoch` while the process is already deep into the
run, so a decay phase ends up mostly finished before it begins.

This module resolves a `[[phase]]` queue from the config into a list of
independently validated schedule views. It is deliberately pure - no torch, no
deepspeed, no filesystem - so the config contract can be tested on a machine
that cannot run the trainer at all.
"""

# Keys a phase may override. Everything else in a phase is inherited from the
# top-level config, so a phase block only has to name what actually differs.
SCHEDULE_KEYS = (
    'lr_scheduler',
    'warmup_steps',
    'linear_end_factor',
    'wsd_decay_frac',
    'wsd_end_factor',
)


class PhaseError(ValueError):
    """A `[[phase]]` queue that cannot be turned into a runnable schedule.

    A ValueError subclass so the existing `except ValueError` around schedule
    construction keeps working; the phase name is prepended by train.py.
    """


class Phase:
    """One stage of a run: a dataset, an epoch budget, and a schedule view.

    `first_epoch` is absolute over the whole run (1-based), which is what keeps
    checkpoint and model names unique across stages.
    """

    __slots__ = ('index', 'name', 'dataset', 'epochs', 'first_epoch', 'config')

    def __init__(self, index, name, dataset, epochs, first_epoch, config):
        self.index = index
        self.name = name
        self.dataset = dataset
        self.epochs = epochs
        self.first_epoch = first_epoch
        self.config = config

    @property
    def last_epoch(self):
        return self.first_epoch + self.epochs - 1

    def __repr__(self):
        return (f'Phase({self.index}, {self.name!r}, epochs={self.epochs}, '
                f'dataset={self.dataset!r})')


def _positive_int(value, what):
    # bool is an int subclass, and `epochs = true` is a typo, not a request.
    if isinstance(value, bool) or not isinstance(value, int):
        raise PhaseError(f'{what} must be an integer number of epochs, got {value!r}')
    if value < 1:
        raise PhaseError(f'{what} must be >= 1, got {value}')
    return value


def normalize_phase_plan(config):
    """Resolve `config` into a list of `Phase`, one per stage of the run.

    With no `[[phase]]` table this returns the single implicit phase the config
    already describes, so every existing config keeps working untouched.

    With a queue, each phase inherits the whole top-level config and overrides
    only its dataset, epoch budget and schedule keys. `config['epochs']` is
    rewritten to the sum of the phase budgets: train.py's own stop condition is
    `dataloader.epoch > epochs`, and leaving the two numbers disagree would end
    the run somewhere the plan does not describe.
    """
    raw = config.get('phase')

    if raw is None:
        entries = [{}]
    else:
        if not isinstance(raw, list) or not raw:
            raise PhaseError(
                '`phase` must be a non-empty array of tables - declare stages as '
                f'[[phase]] blocks, got {type(raw).__name__}')
        for i, entry in enumerate(raw):
            if not isinstance(entry, dict):
                raise PhaseError(
                    f'phase[{i}] must be a table ([[phase]]), got {type(entry).__name__}')
        entries = raw

    default_dataset = config.get('dataset')
    phases = []
    total_epochs = 0

    for index, entry in enumerate(entries):
        name = str(entry.get('name') or f'phase{index + 1}')

        if 'epochs' in entry:
            epochs = _positive_int(entry['epochs'], f'phase `{name}`: epochs')
        elif raw is None:
            epochs = _positive_int(config.get('epochs'), 'top-level `epochs`')
        else:
            raise PhaseError(
                f'phase `{name}` must declare `epochs`; every stage of a queue owns '
                'its own epoch budget')

        dataset = entry.get('dataset') or default_dataset
        if not dataset:
            raise PhaseError(
                f'phase `{name}` has no `dataset`, and there is no top-level `dataset` '
                'for it to inherit')

        view = dict(config)
        view.pop('phase', None)
        view['epochs'] = epochs
        view['dataset'] = dataset
        for key in SCHEDULE_KEYS:
            if key in entry:
                view[key] = entry[key]

        phases.append(Phase(index, name, dataset, epochs, total_epochs + 1, view))
        total_epochs += epochs

    declared = config.get('epochs')
    if declared is not None and declared != total_epochs:
        raise PhaseError(
            f'top-level `epochs = {declared}` does not match the sum of the phase '
            f'budgets ({total_epochs}). train.py ends the run at `epochs`, so a '
            'mismatch silently truncates or overruns the queue - drop the key and '
            'let the queue be the single source of truth.')

    if raw is not None:
        config['epochs'] = total_epochs

    return phases


def resume_epoch_is_supported(phases, epoch):
    """Can a checkpoint at `epoch` be resumed by this plan?

    Every checkpoint inside the queue can. Epoch numbering is absolute over the
    whole run, so the saved epoch names the stage the run died in, and that
    stage's dataset and schedule are rebuilt on resume. Only a checkpoint past
    the end of the queue is refused: there is no stage left to run.

    A single-phase plan is the same rule with one stage, so it stays resumable
    exactly where it used to be.
    """
    return resume_phase_index(phases, epoch) is not None


def resume_phase_index(phases, epoch):
    """Index of the stage that owns absolute `epoch`, or None past the end.

    This is the whole of resume addressing. The stages partition 1..last_epoch
    with no gaps and no overlap, so the epoch alone is sufficient - including an
    epoch that was only partly trained, which is what a `checkpoint_every_n_minutes`
    save leaves behind when a run dies between two epoch boundaries.

    A run that is not a queue resolves to stage 0 for every epoch, which is the
    pre-existing contract: with one dataset and one schedule there is no later
    stage a resume could disagree with, so it stays permissive where it always
    was.
    """
    if len(phases) == 1:
        return 0
    for ph in phases:
        if ph.first_epoch <= epoch <= ph.last_epoch:
            return ph.index
    return None


def resume_refusal(phases, epoch):
    """Why `resume_epoch_is_supported` said no, phrased for the operator."""
    return (f'Checkpoint is at epoch {epoch}, which is past the end of the queue: '
            f'this plan covers epochs 1..{phases[-1].last_epoch} across '
            f'{len(phases)} phase(s). That is a finished run rather than a resumable '
            'one - point --resume_from_checkpoint at an unfinished run, or start a '
            'new output dir.')


def steps_completed_in_phase(phase, epoch, steps_per_epoch, batches_pulled,
                             gradient_accumulation_steps):
    """How many optimizer steps of `phase` the saved checkpoint had already run.

    `batches_pulled` is the loader's micro-batch counter, which is what pins
    the position inside an epoch that a `checkpoint_every_n_minutes` save
    interrupted. It runs one batch ahead of the finished steps because the next
    micro-batch is preloaded; the integer division drops that one, and it is the
    same offset `PipelineDataLoader.load_state_dict` uses to skip batches - so a
    replayed schedule and the restored data agree on where the run stopped.
    """
    whole = max(0, epoch - phase.first_epoch) * steps_per_epoch
    partial = max(0, int(batches_pulled)) // max(1, int(gradient_accumulation_steps))
    return whole + partial


def schedule_varies(phase):
    """Does this stage's schedule actually change the LR as it steps?

    A `constant` stage with no warmup returns the same LR on every step, so
    replaying its completed steps through the scheduler cannot change
    anything - it would only cost time, and on a long stage a wall of PyTorch
    deprecation warnings from ConstantLR.
    """
    if phase.config.get('warmup_steps', 0) > 0:
        return True
    return phase.config.get('lr_scheduler', 'constant') != 'constant'


def _schedule_view(phase):
    """The stage's effective schedule keys, in the fixed SCHEDULE_KEYS order.

    A list rather than a dict so the value stored in a checkpoint stays stable
    and readable, and so the comparison is order-independent of how it was built.
    """
    return [phase.config.get(key) for key in SCHEDULE_KEYS]


def queue_signature(phases):
    """A compact fingerprint of the queue's shape, stored beside a checkpoint.

    A resume maps the saved epoch onto a stage by position, so if the queue was
    edited between the save and the restart - a budget moved, a stage added, a
    dataset swapped - that mapping silently lands on a DIFFERENT stage and the
    replayed LR belongs to a schedule the run was never on. Carrying the shape
    in the checkpoint turns that into a refusal instead.

    Each stage contributes its schedule view on top of its name, budget and
    dataset, because those three only pin WHICH stage an epoch lands on while
    the schedule is what makes the replayed LR right or wrong. Editing
    `linear_end_factor`, or adding a warmup, changes the LR at the resume point
    without moving any epoch, so a signature that omitted it would resume onto a
    schedule the run was never trained under - silently.
    """
    return [[ph.name, ph.epochs, ph.dataset, _schedule_view(ph)] for ph in phases]


def missing_signature_refusal(phases):
    """Why a checkpoint cannot be proven to belong to this queue.

    Absence of a signature reads as two different things: written before the
    queue feature existed, or written by a run of THIS SAME config with no
    [[phase]] block. The second is the dangerous one - resuming that checkpoint
    would build a stage's different dataset and restore a batch position that
    indexes the old one, so training would continue on the wrong data without a
    word. A queue's own checkpoints always carry the signature, so refusing
    here costs nothing that used to work.
    """
    return ('This run is configured with a multi-phase queue, but the checkpoint '
            'carries no queue signature, so nothing proves which stage its epoch '
            'belongs to.\n'
            f'  configured queue: {queue_signature(phases)}\n'
            'A checkpoint without a signature was written either before the queue '
            'feature existed, or by a run of this config WITHOUT its [[phase]] '
            'block. The second case is the dangerous one: resuming it would build '
            "this queue's stage datasets and restore a batch position that "
            'indexes a different one. Resume with the original config, or start '
            'a new output dir.')


def queue_mismatch_refusal(saved_signature, phases):
    """Why a checkpoint's queue shape is not the one being resumed with."""
    return ('This checkpoint was written by a different phase queue than the one '
            f'now configured.\n  checkpoint: {saved_signature}\n  config:     '
            f'{queue_signature(phases)}\n'
            'Epochs are absolute over the run, so a changed queue maps the saved '
            'epoch onto a different stage and the resumed learning rate would '
            'belong to the wrong schedule. Resume with the original config, or '
            'start a new output dir.')


def phase_action(phases, index, start_epoch, new_epoch):
    """What the run should do when the dataloader rolls into `new_epoch`.

    Returns `('continue', index)`, `('switch', next_index)` or `('finish', None)`.
    Kept here rather than inline in train.py so the epoch arithmetic - the part
    that decides on which step a dataset actually changes - is testable without
    DeepSpeed, a GPU or the model weights.
    """
    if new_epoch - start_epoch < phases[index].epochs:
        return ('continue', index)
    nxt = index + 1
    if nxt >= len(phases):
        return ('finish', None)
    return ('switch', nxt)


def describe(phase, steps_per_epoch=None):
    """One line a human can read off the log before the GPU gets busy."""
    scheduler = phase.config.get('lr_scheduler', 'constant')
    parts = [
        f'phase {phase.index + 1} `{phase.name}`',
        f'epochs={phase.epochs}',
        f'lr_scheduler={scheduler}',
        f'warmup_steps={phase.config.get("warmup_steps", 0)}',
    ]
    if scheduler == 'linear':
        parts.append(f'linear_end_factor={phase.config.get("linear_end_factor", 0.0)}')
    elif scheduler == 'wsd':
        parts.append(f'wsd_decay_frac={phase.config.get("wsd_decay_frac", 0.25)}')
        parts.append(f'wsd_end_factor={phase.config.get("wsd_end_factor", 0.0)}')
    if steps_per_epoch is not None:
        parts.append(f'steps/epoch={steps_per_epoch}')
        parts.append(f'total_steps={phase.epochs * steps_per_epoch}')
    parts.append(f'dataset={phase.dataset}')
    return ', '.join(parts)