"""Tiling is OPT-IN: utils/vae_tiling.py must encode whole frames until the
config asks for tiles with a positive [model].vae_max_area.

The contract has two halves and they are not equally strong.

The strong half is the default. Frames are encoded whole until the user sets a
positive area, and that is observable directly: the encoder is called once, on
the whole frame, whatever the frame size. A frame the retired 1280*1280 default
would have split is now encoded whole.

The weaker half is the tiling that can be switched ON. A tiled encode is not the
same computation as a whole-frame encode - a convolutional encoder sees other
neighbourhoods at a tile edge - so it is only trustworthy if the split covers the
frame exactly once, keeps every edge on the stride, and normalises the blend so
the weights sum to 1 everywhere. Those three are checked here directly.

What is NOT checked, and cannot be with a weight-free fake: that the blend
weights are the RIGHT ones. The fake encoder is linear and local, so two tiles
that straddle a seam produce the same value per pixel and any positive weights
that sum to 1 give the same answer - including the wrong ones. The real encoder
is not that forgiving, so this file is not evidence that the blend is optimal;
it is evidence that the geometry and the normalisation hold. See the note on
test_a_tiled_encode_reproduces_the_untiled_encode.
"""

import contextlib
import io
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.vae_tiling as vae_tiling
from utils.vae_tiling import (
    DEFAULT_VAE_MAX_AREA,
    DEFAULT_VAE_MIN_OVERLAP,
    VAE_SPATIAL_STRIDE,
    axis_fade_lengths,
    get_tiling_indices_v2,
    plan_tiling_grid,
    resolve_vae_tiling,
    vae_encode,
)

STRIDE = VAE_SPATIAL_STRIDE

# A budget a user would set for roughly 15 GB of VRAM, and a frame above it.
BUDGET = 1280 * 1280
OVERLAP = 128


class RecordingVae:
    """Wraps a fake encoder and reports the spatial shape of every call."""

    def __init__(self, encoder):
        self.model = encoder
        self.scale = [torch.zeros(16), torch.ones(16)]

    def calls(self):
        return list(self.model.shapes)


def _pool_encoder(latent_channels=16):
    class PoolEncoder:
        """Linear local encoder: 3 -> 16 channels, stride-8 spatial average pool."""

        def __init__(self):
            gen = torch.Generator().manual_seed(0)
            self.weights = torch.randn(latent_channels, 3, generator=gen) * 0.2
            self.shapes = []

        def encode(self, x, scale):
            self.shapes.append((x.shape[-2], x.shape[-1]))
            b, c, t, h, w = x.shape
            pooled = F.avg_pool2d(x.reshape(b * t, c, h, w), STRIDE, STRIDE)
            pooled = pooled.reshape(b, t, c, h // STRIDE, w // STRIDE)
            out = torch.einsum('btchw,kc->btkhw', pooled, self.weights.to(pooled.dtype))
            return out.permute(0, 2, 1, 3, 4).contiguous()

    return PoolEncoder()


class _ConstantEncoderBase:
    """Base for encoders that return a constant latent."""

    def encode(self, x, scale):
        b, _, t, h, w = x.shape
        return torch.ones((b, 16, t, h // STRIDE, w // STRIDE),
                          dtype=torch.float32, device=x.device)

    shapes = []


def _vae(encoder):
    return RecordingVae(encoder)


def _pool_vae():
    return RecordingVae(_pool_encoder())


def _untiled(vae, x):
    """The single whole-frame encode, for comparison against a tiled one."""
    return vae.model.encode(x * 2 - 1, vae.scale)


def _frame(h, w, seed=0):
    gen = torch.Generator().manual_seed(seed)
    return torch.rand(1, 3, 1, h, w, generator=gen, dtype=torch.float32)


# --------------------------------------------------------------------------
# the default: off, and explicitly enabled by a positive vae_max_area
# --------------------------------------------------------------------------

def test_the_shipped_default_is_no_tiling():
    assert DEFAULT_VAE_MAX_AREA == 0, 'tiling must be opt-in; 0 is the off switch'


@pytest.mark.parametrize('config, want_area, want_overlap', [
    ({}, 0, DEFAULT_VAE_MIN_OVERLAP),
    ({'vae_min_overlap': 64}, 0, 64),
    ({'vae_max_area': 0}, 0, DEFAULT_VAE_MIN_OVERLAP),
    ({'vae_max_area': -1}, 0, DEFAULT_VAE_MIN_OVERLAP),
    ({'vae_max_area': BUDGET}, BUDGET, DEFAULT_VAE_MIN_OVERLAP),
    ({'vae_max_area': '1048576', 'vae_min_overlap': '32'}, 1048576, 32),
])
def test_resolve_vae_tiling_reads_the_user_opt_in(config, want_area, want_overlap):
    assert resolve_vae_tiling(config) == (want_area, want_overlap)


def test_no_config_means_one_whole_frame_encode():
    vae = _pool_vae()
    out = vae_encode(_frame(1024, 1024), vae, *resolve_vae_tiling({}))
    assert vae.calls() == [(1024, 1024)]
    assert out.shape == (1, 16, 1, 128, 128)


def test_the_old_default_area_no_longer_tiles_by_itself():
    """1536x1536 is above the retired 1280*1280 default, and is now whole."""
    vae = _pool_vae()
    out = vae_encode(_frame(1536, 1536), vae, *resolve_vae_tiling({}))
    assert vae.calls() == [(1536, 1536)]
    assert out.shape[-2:] == (192, 192)


@pytest.mark.parametrize('area', [0, -4096])
def test_an_explicit_non_positive_area_never_tiles(area):
    vae = _pool_vae()
    vae_encode(_frame(512, 4096), vae, area, DEFAULT_VAE_MIN_OVERLAP)
    assert vae.calls() == [(512, 4096)]


def test_a_positive_area_below_the_frame_area_tiles():
    vae = _pool_vae()
    out = vae_encode(_frame(1536, 1536), vae, 512 * 512, 128)
    calls = vae.calls()
    assert len(calls) > 1, 'the frame exceeds the budget, so it must be split'
    assert all(h * w <= 512 * 512 for h, w in calls), calls
    assert out.shape[-2:] == (192, 192), 'the tiles still cover the whole frame'


def test_a_positive_area_at_or_above_the_frame_area_keeps_it_whole():
    vae = _pool_vae()
    vae_encode(_frame(512, 512), vae, 512 * 512, 128)
    assert vae.calls() == [(512, 512)]


# --------------------------------------------------------------------------
# the tiling that IS switched on: geometry and normalisation
# --------------------------------------------------------------------------

@pytest.mark.parametrize('h, w', [(1536, 1536), (2048, 1536), (1536, 2048), (2048, 2048)])
def test_a_tiled_encode_reproduces_the_untiled_encode(h, w):
    """The blend is normalised, so a linear encoder tiles to the same answer.

    Note the limit of this check: because the fake encoder is linear and local,
    it cannot tell a correct fade from any other set of weights that sums to 1.
    It proves the split is geometrically sound; it does not certify the blend.
    The normalisation itself is pinned by the two tests below.
    """
    vae = _pool_vae()
    x = _frame(h, w, seed=h + w)
    with torch.no_grad():
        untiled = _untiled(vae, x)
        with contextlib.redirect_stdout(io.StringIO()):
            tiled = vae_encode(x.clone(), vae, BUDGET, OVERLAP)
    assert tiled.shape == untiled.shape, f'{h}x{w}: the tile grid changed the latent shape'
    deviation = (tiled.float() - untiled.float()).abs().max().item()
    assert deviation < 2e-4, f'{h}x{w}: tiled encode deviates by {deviation:.3e}'


@pytest.mark.parametrize('h, w', [(1536, 1536), (2048, 1536), (2048, 2048)])
def test_every_latent_row_is_normalised_to_one(h, w):
    """A constant encoder turns the blend weights into the returned value.

    Where the weights of the tiles covering a cell sum to 1 the result is exactly
    1; where a cell is left uncovered (weight 0) the divisor clamps and the
    result collapses to 0. So this pins both facts at once: no gap, and no
    double-counted band.
    """
    vae = _vae(_ConstantEncoderBase())
    out = vae_encode(_frame(h, w), vae, BUDGET, OVERLAP)
    assert out.shape[-2:] == (h // STRIDE, w // STRIDE)
    assert torch.equal(out, torch.ones_like(out)), (
        f'{h}x{w}: {int((out != 1).sum())} latent cells are not normalised to 1')


@pytest.mark.parametrize('h, w', [(1536, 1536), (2048, 1536), (2048, 2048)])
def test_the_masks_vae_encode_builds_sum_to_one(h, w, monkeypatch):
    """The real masks, lifted out of vae_encode, must sum to 1 on every cell.

    Reading them off torch.outer means this checks the masks the code actually
    builds rather than a reconstruction of them: a trailing ramp that no longer
    complements its neighbour's leading ramp leaves a band where the weights
    sum to something other than 1, which is what shows up as a bright seam.
    """
    vae = _vae(_ConstantEncoderBase())
    masks = []
    real_outer = vae_tiling.torch.outer

    def recording_outer(mask_h, mask_w):
        masks.append((mask_h.detach().clone(), mask_w.detach().clone()))
        return real_outer(mask_h, mask_w)

    monkeypatch.setattr(vae_tiling.torch, 'outer', recording_outer)
    with contextlib.redirect_stdout(io.StringIO()):
        vae_encode(_frame(h, w), vae, BUDGET, OVERLAP)

    rows, cols = plan_tiling_grid(h, w, BUDGET, OVERLAP)
    h_tiles = get_tiling_indices_v2(h, rows, OVERLAP)[0]
    w_tiles = get_tiling_indices_v2(w, cols, OVERLAP)[0]
    assert len(masks) == rows * cols, f'{len(masks)} masks for a {rows}x{cols} grid'

    total = torch.zeros(h // STRIDE, w // STRIDE, dtype=torch.float64)
    recorded = iter(masks)
    for h_start, h_end in h_tiles:
        for w_start, w_end in w_tiles:
            mask_h, mask_w = next(recorded)
            tile_rows, tile_cols = (h_end - h_start) // STRIDE, (w_end - w_start) // STRIDE
            assert mask_h.shape == (tile_rows,) and mask_w.shape == (tile_cols,), (
                f'tile ({h_start}:{h_end}, {w_start}:{w_end}) got masks of '
                f'{tuple(mask_h.shape)}/{tuple(mask_w.shape)}')
            total[h_start // STRIDE:h_end // STRIDE,
                  w_start // STRIDE:w_end // STRIDE] += torch.outer(mask_h.double(), mask_w.double())

    # The masks are float32, so complementary ramps are complementary to within a
    # few ULP rather than exactly; the tolerance is far below anything a real
    # blend error would produce (a misaligned ramp is off by ~0.5).
    worst = (total - 1.0).abs().max().item()
    assert worst < 1e-6, f'{h}x{w}: blend weights deviate from 1 by {worst:.3e}'


def test_the_seam_ramps_are_exact_complements():
    """The per-latent-row weight profile is 1 everywhere, seams included.

    This is the invariant the seam fade exists for: a tile's trailing ramp and
    its neighbour's leading ramp are the same rows with mirrored values, so the
    two add up to exactly 1. If either ramp moved by a row, the band between
    them would sum to less or more than 1 and show as a bright or dark seam.
    """
    tiles, _ = get_tiling_indices_v2(2048, 4, 128)
    leading, trailing = axis_fade_lengths(tiles, STRIDE)
    total_rows = tiles[-1][1] // STRIDE
    profile = torch.zeros(total_rows, dtype=torch.float64)

    for (start, end), lead, trail in zip(tiles, leading, trailing):
        span = (end - start) // STRIDE
        weights = torch.ones(span, dtype=torch.float64)
        if trail:
            weights[-trail:] = torch.linspace(1.0, 0.0, trail, dtype=torch.float64)
        if lead:
            weights[:lead] = torch.linspace(0.0, 1.0, lead, dtype=torch.float64)
        profile[start // STRIDE:end // STRIDE] += weights

    assert torch.allclose(profile, torch.ones_like(profile), atol=1e-12), profile
    assert leading[0] == 0 and trailing[-1] == 0, 'the outer border keeps a flat weight of 1'


def test_a_seam_fade_never_swallows_more_than_half_a_tile():
    tiles, _ = get_tiling_indices_v2(2048, 4, 128)
    leading, trailing = axis_fade_lengths(tiles, STRIDE)
    for (start, end), lead, trail in zip(tiles, leading, trailing):
        span = (end - start) // STRIDE
        assert 0 < lead + trail <= span


def test_a_frame_that_is_not_stride_aligned_is_truncated_and_says_so():
    vae = _pool_vae()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        out = vae_encode(_frame(1000, 1000), vae, 96 * 96, 128)
    assert tuple(out.shape[-2:]) == (125, 125)
    assert 'WARNING' in log.getvalue()


def test_the_tiling_log_names_the_budget_and_the_grid():
    vae = _pool_vae()
    log = io.StringIO()
    with contextlib.redirect_stdout(log):
        vae_encode(_frame(1024, 1024), vae, 256 * 256, 128)
    printed = log.getvalue()
    assert f'exceeds the {256 * 256}px limit' in printed
    assert 'Tiling' in printed


# --------------------------------------------------------------------------
# the geometry itself
# --------------------------------------------------------------------------

@pytest.mark.parametrize('h, w', [(1024, 1024), (1536, 2048), (2048, 1536), (4096, 4096)])
def test_planned_tiles_cover_the_frame_with_stride_aligned_edges(h, w):
    budget, overlap = BUDGET, OVERLAP
    rows, cols = plan_tiling_grid(h, w, budget, overlap)
    h_tiles, _ = get_tiling_indices_v2(h, rows, overlap)
    w_tiles, _ = get_tiling_indices_v2(w, cols, overlap)

    assert h_tiles[0][0] == 0 and h_tiles[-1][1] == h, 'the frame must be covered end to end'
    assert w_tiles[0][0] == 0 and w_tiles[-1][1] == w
    for tiles in (h_tiles, w_tiles):
        for start, end in tiles:
            assert start % STRIDE == 0 and end % STRIDE == 0, 'the stride must not cross an edge'
            assert end > start, 'a tile may not be empty'
        for (_, end), (start, _) in zip(tiles, tiles[1:]):
            assert start < end, 'a seam may overlap but must not leave a gap'


@pytest.mark.parametrize('h, w', [(1024, 1024), (1536, 2048), (2048, 1536), (4096, 4096)])
def test_planned_tiles_stay_inside_the_budget(h, w):
    rows, cols = plan_tiling_grid(h, w, BUDGET, OVERLAP)
    h_tiles, _ = get_tiling_indices_v2(h, rows, OVERLAP)
    w_tiles, _ = get_tiling_indices_v2(w, cols, OVERLAP)
    tile_h = max(end - start for start, end in h_tiles)
    tile_w = max(end - start for start, end in w_tiles)
    assert tile_h * tile_w <= BUDGET, f'{h}x{w}: tile {tile_h}x{tile_w} over {BUDGET}px'


def test_a_frame_under_budget_plans_a_single_tile():
    assert plan_tiling_grid(256, 256, 256 * 256, OVERLAP) == (1, 1)
    assert get_tiling_indices_v2(1024, 1, OVERLAP)[0] == [(0, 1024)]


# --------------------------------------------------------------------------
# the wiring: the model module must read the opt-in, not carry its own default
# --------------------------------------------------------------------------

def test_the_user_table_reaches_vae_encode_through_the_caching_chain():
    """The [model] table the user writes is the one vae_encode is configured from.

    A default that is off in utils/vae_tiling.py is worthless if the pipeline
    reads its area from somewhere else, so the whole seam is pinned: the [model]
    table becomes self.model_config, the knobs are resolved from it, the encode
    closure gets them, and the caching path takes that closure.
    """
    root = Path(__file__).resolve().parent.parent
    model = (root / 'models' / 'cosmos_predict2.py').read_text(encoding='utf-8')
    dataset = (root / 'utils' / 'dataset.py').read_text(encoding='utf-8')
    train = (root / 'train.py').read_text(encoding='utf-8')

    assert "model_config = config['model']" in train, 'config[model] is the [model] table'
    assert "self.model_config = self.config['model']" in model
    assert 'resolve_vae_tiling(self.model_config)' in model, (
        'the pipeline must resolve the tiling knobs from the user table')
    assert ('vae_encode(tensor, self.vae, self.vae_max_area, self.vae_min_overlap)'
            in model), 'the encode closure must pass the resolved knobs through'
    assert 'self.model.get_call_vae_fn(self.vae)' in dataset, (
        'the caching path must take the closure that configures the encode')


def test_the_model_module_reads_the_opt_in_instead_of_a_local_default():
    source = (Path(__file__).resolve().parent.parent / 'models' / 'cosmos_predict2.py').read_text(
        encoding='utf-8')
    assert 'from utils.vae_tiling import' in source, 'tiling must be imported, not re-copied'
    assert 'resolve_vae_tiling(self.model_config)' in source
    assert 'DEFAULT_VAE_MAX_AREA' not in source, (
        'a second default here would silently re-enable tiling')
    assert 'def vae_encode(' not in source, 'the tiling code belongs in utils/vae_tiling.py'
