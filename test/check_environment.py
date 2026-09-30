"""Environment check: every import the training stack needs, on the real venv.

This file reports the ACTUAL state of each dependency rather than assuming it,
so a gap is named instead of discovered mid-run.

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
    if missing:
        print(f'FAILED: {len(missing)} required module(s) missing: {missing}')
        return 1
    print('All required modules import cleanly.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
