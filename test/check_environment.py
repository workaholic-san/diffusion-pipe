"""Environment check: every import the training stack needs, on the real venv.

`train.py` imports deepspeed at module scope, so on a host without a working
deepspeed the whole script is unimportable. This file reports the ACTUAL state
of each dependency rather than assuming it, so a gap is named instead of
discovered mid-run.

Run:  python test/check_environment.py
Exit: 0 when every REQUIRED import is present; 1 otherwise.
"""

import importlib
import sys

#: (module, required?) - `False` marks a dependency that is genuinely optional
#: on this platform rather than quietly tolerated.
MODULES = [
    ('torch', True),
    ('numpy', True),
    ('scipy', True),
    ('toml', True),
    ('transformers', True),
    ('diffusers', True),
    ('datasets', True),
    ('PIL', True),
    ('sentencepiece', True),
    ('peft', True),
    ('optimi', True),
    ('tensorboard', True),
    ('safetensors', True),
    ('einops', True),
    ('accelerate', True),
    ('loguru', True),
    ('omegaconf', True),
    ('iopath', True),
    ('hydra', True),
    ('ftfy', True),
    ('pytorch_optimizer', True),
    ('wandb', True),
    ('imageio', True),
    ('av', True),
    ('optimum.quanto', True),
]

#: Dependencies that ARE supported upstream on this platform, but for which
#: THIS environment has no installable distribution. The facts below are
#: measured, not assumed:
#:
#:   pip install --only-binary=:all: deepspeed==0.18.4
#:     -> "No matching distribution found". PyPI publishes ZERO Windows wheels
#:        for 0.18.4, and the only Windows wheel for any DeepSpeed release is
#:        deepspeed-0.14.5-cp311-cp311-win_amd64.whl - while this venv is
#:        Python 3.13.
#:
#: DeepSpeed has supported Windows natively since 0.14.5 (its own Windows post,
#: blogs/windows/08-2024): `pip install deepspeed` ships prebuilt operators with
#: no CUDA SDK needed, and `ds_report` validates the install. So "not
#: installable on Windows" would be the wrong claim - the accurate one is "this
#: pin has no wheel for this interpreter". pip therefore falls back to an sdist
#: build, which dies on a missing bin\\deepspeed.bat and then on CUDA_HOME.
#: It installs normally on the owner's Linux/Colab training host.
NO_DISTRIBUTION = {
    'deepspeed': (
        'no wheel for this pin on this interpreter (see the note above); a '
        'Python 3.11 environment would get the prebuilt deepspeed-0.14.5-cp311 '
        'wheel, or build from source with DeepSpeed\'s build_win.bat.'
    ),
}

#: Repo modules this fork touches, checked separately because they import the
#: heavy stack transitively.
REPO_MODULES = [
    'optimizers.muon_kahan',
    'utils.lr_schedulers',
    'utils.train_debug',
]


def main():
    print(f'python: {sys.version.split()[0]}  ({sys.executable})\n')

    missing = []
    blocked = []

    for name, reason in NO_DISTRIBUTION.items():
        try:
            mod = importlib.import_module(name)
            print(f'  OK      {name:<22} {getattr(mod, "__version__", "?")}')
        except Exception as exc:
            print(f'  MISSING {name:<22} {type(exc).__name__}: {exc}')
            missing.append(name)
            blocked.append((name, reason))

    for name, required in MODULES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, '__version__', '?')
            print(f'  OK      {name:<22} {version}')
        except Exception as exc:
            tag = 'MISSING' if required else 'optional'
            print(f'  {tag:<8} {name:<22} {type(exc).__name__}: {exc}')
            if required:
                missing.append(name)

    print()
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    for name in REPO_MODULES:
        try:
            importlib.import_module(name)
            print(f'  OK      {name}')
        except Exception as exc:
            print(f'  MISSING {name:<22} {type(exc).__name__}: {exc}')
            missing.append(name)

    try:
        import torch
        print(f'\ntorch: {torch.__version__}')
        print(f'cuda available: {torch.cuda.is_available()}')
        if torch.cuda.is_available():
            print(f'cuda device: {torch.cuda.get_device_name(0)}')
    except Exception as exc:
        print(f'\ntorch probe failed: {exc}')

    print()
    if blocked:
        for name, reason in blocked:
            print(f'NO DISTRIBUTION: {name}')
            print(f'  {reason}\n')

    if missing:
        print(f'FAILED: {len(missing)} required module(s) missing: {missing}')
        return 1
    if blocked:
        print(f'Core stack complete, but {len(blocked)} module(s) have no '
              'installable distribution here (see above). train.py cannot run '
              'on this host.')
        return 1
    print('All required modules import cleanly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
