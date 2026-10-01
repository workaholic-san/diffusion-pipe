"""Standalone check for the opt-in tiled VAE encode in utils/vae_tiling.py.

The property that matters: once the user turns tiling ON, it must be
TRANSPARENT. The real encoder is nonlinear, so a fake one stands in - a linear,
local, stride-8 encoder (average pool plus fixed channel weights). Such an
encoder is exactly reproducible by encoding tiles and blending them, so if the
blend weights are right the tiled result equals the untiled result. A seam that
does not sum to weight 1 shows up here as a nonzero deviation instead of as a
faint band in someone's training data months later.

Tiling itself is off unless the config sets a positive [model].vae_max_area, so
this check states its budget explicitly rather than reading a default.

Run it with any interpreter that has torch:

    python tools/cosmos_tiled_vae_test.py

Exits nonzero on the first failed invariant.
"""

import contextlib
import io
import pathlib
import sys

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    sys.exit('this check needs torch; run it with the training interpreter')

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils.vae_tiling import (  # noqa: E402
    DEFAULT_VAE_MIN_OVERLAP,
    VAE_SPATIAL_STRIDE,
    axis_fade_lengths,
    get_tiling_indices_v2,
    plan_tiling_grid,
    resolve_vae_tiling,
    vae_encode,
)

# The budget a user would set for roughly 15 GB of VRAM; tiling is off by
# default, so the value cannot come from a default any more.
MAX_AREA = 1638400


class FakeVaeModel:
    """Linear local encoder: 3 -> 16 channels, stride-8 spatial average pool."""

    def __init__(self, latent_channels=16):
        gen = torch.Generator().manual_seed(0)
        self.weights = torch.randn(latent_channels, 3, generator=gen) * 0.2
        self.latent_channels = latent_channels

    def encode(self, x, scale):
        b, c, t, h, w = x.shape
        pooled = F.avg_pool2d(x.reshape(b * t, c, h, w), STRIDE, STRIDE)
        pooled = pooled.reshape(b, t, c, h // STRIDE, w // STRIDE)
        out = torch.einsum('btchw,kc->btkhw', pooled, self.weights.to(pooled.dtype))
        return out.permute(0, 2, 1, 3, 4).contiguous()


class FakeVae:
    def __init__(self):
        self.model = FakeVaeModel()
        self.scale = [torch.zeros(16), torch.ones(16)]


def main():
    global STRIDE
    STRIDE = VAE_SPATIAL_STRIDE

    max_area = MAX_AREA
    overlap = DEFAULT_VAE_MIN_OVERLAP
    print('module: utils/vae_tiling.py')
    print(f'tiling: max_area={max_area}px  min_overlap={overlap}px  stride={STRIDE}')

    failures = check_opt_in_default(max_area, overlap)
    failures += check_geometry(plan_tiling_grid, get_tiling_indices_v2,
                               axis_fade_lengths, max_area, overlap)
    failures += check_transparency(vae_encode, plan_tiling_grid, max_area, overlap)

    print()
    if failures:
        print(f'FAILED ({len(failures)})')
        for line in failures:
            print(f'  - {line}')
        return 1
    print('OK: tiling is transparent, no seams')
    return 0


def check_opt_in_default(max_area, overlap):
    """Tiling must stay off until the user asks for it."""
    print()
    print('opt-in default')
    failures = []
    cases = [
        ({}, 0),
        ({'vae_min_overlap': 64}, 0),
        ({'vae_max_area': 0}, 0),
        ({'vae_max_area': -1}, 0),
        ({'vae_max_area': max_area}, max_area),
    ]
    for config, want in cases:
        got, got_overlap = resolve_vae_tiling(config)
        if got != want:
            failures.append(f'{config} -> max_area {got}, expected {want}')
        if config.get('vae_min_overlap', overlap) != got_overlap:
            failures.append(f'{config} -> min_overlap {got_overlap}')
        print(f'  {str(config):<40} max_area={got}  min_overlap={got_overlap}')
    if not failures:
        print('  no config tiles; a positive vae_max_area is the only way in')
    return failures


def check_geometry(plan, indices, fades, max_area, overlap):
    """Every frame must be fully covered, aligned to the stride, and within budget."""
    print()
    print('geometry')
    failures = []
    for h in (1024, 1536, 2048, 2560, 4096):
        for w in (1024, 1536, 2048, 2560, 4096):
            rows, cols = plan(h, w, max_area, overlap)
            worst_h, bad = check_axis('H', h, rows, indices, fades, overlap)
            worst_w, bad_w = check_axis('W', w, cols, indices, fades, overlap)
            bad += bad_w
            if h * w > max_area and worst_h * worst_w > max_area:
                bad.append(f'budget: tile {worst_h}x{worst_w} exceeds {max_area}px')
            for line in bad:
                failures.append(f'{h}x{w}: {line}')
            if (h, w) in ((2048, 2048), (4096, 4096), (1024, 1024)):
                verdict = 'FAIL' if bad else 'ok'
                print(f'  {h:5}x{w:<5} grid {rows}x{cols}  tile {worst_h}x{worst_w}  {verdict}')
    if not failures:
        print('  25 frames: full coverage, overlap kept, stride-aligned, within budget')
    return failures


def check_axis(axis, length, num, indices, fades, overlap):
    tiles, usable = indices(length, num, overlap)
    bad = []
    if len(tiles) != num:
        return length, [f'{axis}: {len(tiles)} tiles, expected {num}']
    if num == 1:
        return tiles[0][1] - tiles[0][0], bad
    if tiles[0][0] != 0:
        bad.append(f'{axis}: first tile starts at {tiles[0][0]}')
    if tiles[-1][1] != usable:
        bad.append(f'{axis}: last tile ends at {tiles[-1][1]}, not {usable}')
    for (_, end), (start, _) in zip(tiles, tiles[1:]):
        if start >= end:
            bad.append(f'{axis}: gap at {end}')
        if (end - start) < overlap:
            bad.append(f'{axis}: overlap {end - start} below {overlap}')
    for start, end in tiles:
        if start % STRIDE or end % STRIDE:
            bad.append(f'{axis}: edge ({start}, {end}) not a multiple of {STRIDE}')
        if end <= start:
            bad.append(f'{axis}: empty tile ({start}, {end})')

    # a fade must not swallow more than half of a tile, or the middle of the
    # tile would be pure blend with no own contribution
    leading, trailing = fades(tiles, STRIDE)
    for i, ((start, end), lead, trail) in enumerate(zip(tiles, leading, trailing)):
        span = (end - start) // STRIDE
        if lead + trail > span:
            bad.append(f'{axis}: tile {i} fades {lead}+{trail} of {span} rows')
    return max(end - start for start, end in tiles), bad


def check_transparency(vae_encode, plan, max_area, overlap):
    """Tiled encoding must reproduce the untiled encoding."""
    print()
    print('transparency (tiled vs untiled, fake linear encoder)')
    failures = []
    vae = FakeVae()
    torch.manual_seed(1234)

    frames = [(1024, 4096), (1536, 4096), (2048, 4096), (2560, 2560), (4096, 1024),
              (4096, 1536), (4096, 2048), (2048, 2048), (1024, 1024), (2560, 4096)]
    for h, w in frames:
        rows, cols = plan(h, w, max_area, overlap)
        x = torch.rand(1, 3, 1, h, w, dtype=torch.float32)
        with torch.no_grad():
            untiled = vae.model.encode(x * 2 - 1, vae.scale)
            with contextlib.redirect_stdout(io.StringIO()):   # the code is chatty per tile
                tiled = vae_encode(x.clone(), vae, max_area, overlap)
        if tiled.shape != untiled.shape or tiled.dtype != untiled.dtype:
            failures.append(f'{h}x{w}: shape/dtype {tuple(tiled.shape)}/{tiled.dtype} '
                            f'!= {tuple(untiled.shape)}/{untiled.dtype}')
            continue
        deviation = (tiled.float() - untiled.float()).abs().max().item()
        ok = deviation < 2e-4
        if not ok:
            failures.append(f'{h}x{w} grid {rows}x{cols}: deviation {deviation:.3e}')
        print(f'  {h:5}x{w:<5} grid {rows}x{cols}  max deviation {deviation:.2e}  {"ok" if ok else "FAIL"}')

    # a frame that does not need tiling must take the plain single-encode path
    x = torch.rand(1, 3, 1, 1024, 1024, dtype=torch.float32)
    with torch.no_grad():
        direct = vae.model.encode(x * 2 - 1, vae.scale)
        with contextlib.redirect_stdout(io.StringIO()):
            under_budget = vae_encode(x.clone(), vae, max_area, overlap)
            disabled = vae_encode(x.clone(), vae, 0, overlap)
    for label, got in (('under budget', under_budget), ('max_area=0', disabled)):
        deviation = (got.float() - direct.float()).abs().max().item()
        if deviation != 0.0:
            failures.append(f'{label} path is not a plain encode: deviation {deviation:.3e}')
        print(f'  no tiling via {label:<12} max deviation {deviation:.2e}  '
              f'{"ok" if deviation == 0.0 else "FAIL"}')

    # a frame whose size is not a multiple of the stride drops the tail, and must
    # say so, and must still return the truncated latent size
    x = torch.rand(1, 3, 1, 1000, 1000, dtype=torch.float32)
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        out = vae_encode(x.clone(), vae, 96 * 96, overlap)
    if tuple(out.shape[-2:]) != (125, 125):
        failures.append(f'1000x1000: latent {tuple(out.shape[-2:])} != (125, 125)')
    if 'WARNING' not in log.getvalue():
        failures.append('1000x1000: dropped tail was not reported')
    print(f'  unaligned 1000x1000 -> latent {tuple(out.shape[-2:])}  '
          f'{"ok" if tuple(out.shape[-2:]) == (125, 125) else "FAIL"}')
    return failures


if __name__ == '__main__':
    sys.exit(main())
