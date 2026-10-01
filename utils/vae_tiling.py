"""Tiled VAE encode: split an oversized frame into overlapping tiles and blend
the latent tiles back together.

The VAE is fed whole frames while latents are pre-cached, so a large frame can
blow past the available VRAM. vae_encode() below splits an oversized frame into
a grid of overlapping tiles, encodes each tile on its own, and blends the latent
tiles back together with a linear feather across the overlap bands.

Tiling is OPT-IN. It runs only when the config sets [model].vae_max_area to a
positive number; without that key (or with 0 / a negative value) the frame is
always encoded whole. A whole-frame encode is exact, while a tiled one is not
bit-identical: a convolutional encoder sees different neighbourhoods at a tile
edge, so a tiled encode is an approximation of the same frame and its cached
latents differ slightly from the untiled ones.

Two knobs drive the split, both from the [model] table of the training config:

  vae_max_area    - largest pixel area handed to the VAE in one call. A frame at
                    or below this is encoded whole, anything above it is tiled.
                    1280*1280 (1638400) is a safe fit for roughly 15 GB of VRAM.
                    Omit it, or set it to 0, to never tile.
  vae_min_overlap - overlap between neighbouring tiles, in pixels. A larger
                    overlap hides seams better but costs extra VRAM. Only read
                    when tiling is on.

Every tile edge sits on a multiple of the VAE's spatial stride so that the
stride never crosses an edge. get_tiling_indices_v2() may snap a tile up to a
larger multiple when the arithmetic allows it, so plan_tiling_grid() has to
budget for the largest of them - otherwise the tile the VAE actually sees can
come out bigger than the one we solved for.

This module deliberately imports only math and torch: it is the VAE encode
geometry, independent of the model plumbing in models/cosmos_predict2.py, and
keeping it importable on its own is what lets the tiling invariants be tested
without loading the transformers/accelerate stack or any weights.
"""

import math

import torch

DEFAULT_VAE_MAX_AREA = 0  # 0 disables tiling; see module docstring.
DEFAULT_VAE_MIN_OVERLAP = 128

# The VAE downsamples by 8 in each spatial dimension. This is an architectural
# constraint of the encoder, not a tuning knob.
VAE_SPATIAL_STRIDE = 8

# Multiples a tile may be snapped up to, largest first.
VAE_TILE_DIVISORS = (32, 16)


def resolve_vae_tiling(model_config):
    """Read the tiling knobs out of a [model] config table.

    Returns (max_area, min_overlap). A missing (or non-positive) vae_max_area
    means no tiling, so the user opts in by setting a positive area.
    """
    max_area = int(model_config.get('vae_max_area', DEFAULT_VAE_MAX_AREA))
    min_overlap = int(model_config.get('vae_min_overlap', DEFAULT_VAE_MIN_OVERLAP))
    if max_area <= 0:
        max_area = 0
    return max_area, min_overlap


def plan_tiling_grid(H, W, max_area, min_overlap, max_iter=1000):
    """Pick the smallest (rows, cols) tile grid whose tiles fit in max_area."""
    if H * W <= max_area:
        return 1, 1

    R = 1
    C = 1
    safety_divisor = max(VAE_TILE_DIVISORS)

    for _ in range(max_iter):
        min_H_tile = (H + (R - 1) * min_overlap) / R
        min_W_tile = (W + (C - 1) * min_overlap) / C

        # A tile gets rounded up to safety_divisor, so the area solved for here
        # is a lower bound on the area the VAE actually sees.
        H_tile = int(math.ceil(min_H_tile / safety_divisor) * safety_divisor)
        W_tile = int(math.ceil(min_W_tile / safety_divisor) * safety_divisor)

        if H_tile * W_tile <= max_area:
            return R, C

        if H_tile >= W_tile:
            R += 1
        else:
            C += 1

    print(f"[Smart Tiled VAE] WARNING: no grid fits {H}x{W} within {max_area}px in "
          f"{max_iter} iterations, using the last one: {R}x{C}")
    return R, C


def get_tiling_indices_v2(length, num_tiles, min_overlap, hard_divisor=VAE_SPATIAL_STRIDE):
    """Return (tile spans, usable length) for one axis.

    Spans are cut on a multiple of hard_divisor so the encoder's stride never
    crosses a tile edge. When the arithmetic allows it we snap to a larger
    multiple instead, which keeps tiles from ending on odd offsets.
    """
    if num_tiles == 1:
        return [(0, length)], length

    usable_length = (length // hard_divisor) * hard_divisor

    divisor = hard_divisor
    for candidate in VAE_TILE_DIVISORS:
        if candidate % hard_divisor == 0 and usable_length % candidate == 0:
            divisor = candidate
            break

    min_S = (usable_length + (num_tiles - 1) * min_overlap) / num_tiles
    S = min(int(math.ceil(min_S / divisor) * divisor), usable_length)

    stride = (usable_length - S) / (num_tiles - 1) if num_tiles > 1 else 0

    tiles = []
    for i in range(num_tiles):
        if i == num_tiles - 1:
            start = usable_length - S
        else:
            start = int(round((i * stride) / divisor) * divisor)

        start = max(0, min(start, usable_length - S))
        tiles.append((start, start + S))

    return tiles, usable_length


def axis_fade_lengths(tiles, stride):
    """Per tile, how many latent rows/cols to fade at its leading and trailing edge.

    A tile overlaps its neighbour by more than the nominal minimum, so fading a
    fixed width at each edge leaves the two ramps misaligned - they cover
    different rows and do not sum to 1, which shows up as a bright band along
    the seam. Fading each seam over its own overlap makes the two ramps exact
    complements. The outer edges of the frame get no fade, so the border keeps a
    flat weight of 1.
    """
    leading = [0] * len(tiles)
    trailing = [0] * len(tiles)
    for i in range(len(tiles) - 1):
        span = (tiles[i][1] - tiles[i][0]) // stride
        rows = (tiles[i][1] - tiles[i + 1][0]) // stride
        rows = max(0, min(rows, span // 2))
        trailing[i] = rows
        leading[i + 1] = rows
    return leading, trailing


def vae_encode(tensor, vae, max_area=DEFAULT_VAE_MAX_AREA, min_overlap=DEFAULT_VAE_MIN_OVERLAP):
    """Encode one frame, tiling it first if it is too large to fit in VRAM.

    max_area <= 0 (the default) disables tiling and always encodes the whole
    frame; the user opts in by setting a positive max_area.
    """
    tensor = tensor * 2 - 1  # Normalize [0, 1] -> [-1, 1], same as transforms.Normalize.

    B, C, T, H, W = tensor.shape
    device = tensor.device
    scale_factor = VAE_SPATIAL_STRIDE

    if max_area <= 0 or H * W <= max_area:
        with torch.no_grad():
            return vae.model.encode(tensor, vae.scale)

    if H % scale_factor != 0 or W % scale_factor != 0:
        print(f"[Smart Tiled VAE] WARNING: {H}x{W} is not a multiple of {scale_factor}. "
              f"The trailing {H % scale_factor}px of height and {W % scale_factor}px of "
              f"width are dropped.")

    R, C_grid = plan_tiling_grid(H, W, max_area, min_overlap)

    print(f"\n[Smart Tiled VAE] {H}x{W} ({H * W}px) exceeds the {max_area}px limit.")
    print(f"[Smart Tiled VAE] Tiling {R} rows x {C_grid} cols.")

    latent_H = H // scale_factor
    latent_W = W // scale_factor

    h_tiles = get_tiling_indices_v2(H, R, min_overlap)[0]
    w_tiles = get_tiling_indices_v2(W, C_grid, min_overlap)[0]

    h_lead, h_trail = axis_fade_lengths(h_tiles, scale_factor)
    w_lead, w_trail = axis_fade_lengths(w_tiles, scale_factor)

    output_latent = None
    weights = None
    out_dtype = tensor.dtype

    for h_index, (h_start, h_end) in enumerate(h_tiles):
        for w_index, (w_start, w_end) in enumerate(w_tiles):
            tile_input = tensor[:, :, :, h_start:h_end, w_start:w_end]

            print(f" -> tile [{h_start}:{h_end}, {w_start}:{w_end}] "
                  f"({h_end - h_start}x{w_end - w_start})")

            with torch.no_grad():
                tile_latent = vae.model.encode(tile_input, vae.scale)

            if output_latent is None:
                latent_C = tile_latent.shape[1]
                out_dtype = tile_latent.dtype
                output_latent = torch.zeros((B, latent_C, T, latent_H, latent_W), device=device, dtype=torch.float32)
                weights = torch.zeros_like(output_latent)

            lh_start, lh_end = h_start // scale_factor, h_end // scale_factor
            lw_start, lw_end = w_start // scale_factor, w_end // scale_factor
            tile_h_latent = lh_end - lh_start
            tile_w_latent = lw_end - lw_start

            # Each interior edge fades over its own overlap so the two ramps on a
            # seam are exact complements and the weights sum to 1. The outer
            # edges of the frame keep a flat weight of 1, so the border is not
            # darkened.
            mask_h = torch.ones(tile_h_latent, device=device, dtype=torch.float32)
            mask_w = torch.ones(tile_w_latent, device=device, dtype=torch.float32)

            for n, from_zero in ((h_lead[h_index], True), (h_trail[h_index], False)):
                if n > 0:
                    if from_zero:
                        mask_h[:n] = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float32)
                    else:
                        mask_h[-n:] = torch.linspace(1.0, 0.0, n, device=device, dtype=torch.float32)

            for n, from_zero in ((w_lead[w_index], True), (w_trail[w_index], False)):
                if n > 0:
                    if from_zero:
                        mask_w[:n] = torch.linspace(0.0, 1.0, n, device=device, dtype=torch.float32)
                    else:
                        mask_w[-n:] = torch.linspace(1.0, 0.0, n, device=device, dtype=torch.float32)

            mask_2d = torch.outer(mask_h, mask_w).view(1, 1, 1, tile_h_latent, tile_w_latent)

            output_latent[:, :, :, lh_start:lh_end, lw_start:lw_end] += tile_latent.float() * mask_2d
            weights[:, :, :, lh_start:lh_end, lw_start:lw_end] += mask_2d

            del tile_latent
            torch.cuda.empty_cache()

    return (output_latent / weights.clamp(min=1e-5)).to(out_dtype)
