"""Where inside an epoch a checkpoint left the dataloader.

Pure arithmetic about a resume position, kept out of utils/dataset.py on
purpose: that module imports deepspeed, datasets and comfy at import time, none
of which install on a machine that only needs to check the arithmetic.
"""


def resume_skip_batches(num_batches_pulled):
    """How many batches `SkipFirstNSampler` must drop to restore a position.

    `num_batches_pulled` counts the micro-batches the loader has pulled, and it
    runs ONE AHEAD of the batches the caller actually consumed, because the
    next micro-batch is preloaded - hence the -1.

    The clamp is not cosmetic; it is what makes an epoch-boundary checkpoint
    resume correctly. `PipelineDataLoader.__next__` resets the counter to 0 when
    the underlying dataloader is exhausted and bumps the epoch, and
    `Saver.process_epoch` checkpoints exactly there. With
    `checkpoint_every_n_epochs = 1` that is every epoch boundary, including both
    internal boundaries of a phase queue - the checkpoints the resume path is
    built for.

    `0 - 1` is -1, and `SkipFirstNSampler.__iter__` is `range(n, length)`, so an
    unclamped restore made the resumed epoch run over `range(-1, length)`: one
    batch TOO MANY, opening on index -1 - the dataset's last item, served again
    as if it were the epoch's first. A counter of 0 means the start of an epoch,
    so 0 is what it has to restore to.
    """
    return max(0, int(num_batches_pulled) - 1)
