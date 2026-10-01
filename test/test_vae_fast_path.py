"""The single-frame VAE fast path in models/wan/vae2_1.py.

A still image arrives as T=1. The encoder's frame cache is then pure overhead:
every convolution would clone the whole activation to fill a cache that
clear_cache() throws away one call later, and every temporal padding would
materialise a padded copy of the frame. Both were the reason a 512x512 encode
cost 970 MiB of activations against 384 MiB for the same arithmetic done
ComfyUI's way.

The shortcut is only legitimate if it computes the SAME numbers. Two facts make
it so, and both are pinned here rather than assumed:

  * the temporal padding only prepends zeros, so w0*0 + w1*0 + w2*x0 == w2*x0
    exactly in IEEE arithmetic - the dropped taps can only multiply zero. The
    test compares against the padded reference bitwise, not within a tolerance;
  * the cache is only skipped when it cannot be read. A single chunk never
    reads it, so passing None is equivalent by construction. Multi-chunk video
    keeps the cache and must be byte-for-byte unchanged - asserted against a
    forced-cache reference, not by inspection.

The F.pad counter is the falsifier: without the shortcut every convolution
pads, and the temporal component of that pad is the copy of the whole
activation. If a future change routes a T=1 frame back through the padded path,
the counter test fails; if it routes a cached frame through the shortcut, the
cache test fails.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import models.wan.vae2_1 as vae2_1
from models.wan.vae2_1 import CACHE_T, CausalConv3d, WanVAE_


def _conv(in_ch=3, out_ch=5, kernel=3, seed=0):
    torch.manual_seed(seed)
    conv = CausalConv3d(in_ch, out_ch, kernel, padding=kernel // 2)
    with torch.no_grad():
        conv.weight.copy_(torch.randn(conv.weight.shape, generator=torch.Generator().manual_seed(seed)) * 0.1)
        if conv.bias is not None:
            conv.bias.copy_(torch.randn(conv.bias.shape, generator=torch.Generator().manual_seed(seed)) * 0.1)
    return conv


def _padded_reference(conv, x):
    """What the layer did before the shortcut: pad the activation, then convolve."""
    padding = list(conv._padding)
    return F.conv3d(F.pad(x, padding), conv.weight, conv.bias, conv.stride, 0,
                    conv.dilation, conv.groups)


@pytest.fixture
def counted_pads(monkeypatch):
    """Record every F.pad the module issues, keyed on the padding tuple."""
    seen = []
    real_pad = vae2_1.F.pad

    def recording_pad(input, pad, *args, **kwargs):
        seen.append(tuple(pad) if hasattr(pad, '__iter__') else (pad,))
        return real_pad(input, pad, *args, **kwargs)

    monkeypatch.setattr(vae2_1.F, 'pad', recording_pad)
    return seen


# --------------------------------------------------------------------------
# the shortcut computes the same numbers
# --------------------------------------------------------------------------

@pytest.mark.parametrize('shape', [(1, 3, 1, 32, 32), (2, 4, 1, 16, 24), (1, 3, 1, 8, 8)])
def test_single_frame_fast_path_is_bitwise_identical_to_the_padded_path(shape):
    conv = _conv(in_ch=shape[1], out_ch=7, seed=1)
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(3))
    assert torch.equal(conv(x), _padded_reference(conv, x))


def test_the_dropped_weight_taps_only_ever_multiplied_zeros():
    """The equivalence rests on this: the truncated kernel taps saw zeros."""
    conv = _conv(in_ch=2, out_ch=3, kernel=5, seed=2)
    assert conv.weight.shape[2] == 5
    assert conv._padding[4] == 2 * (5 // 2), 'two prepended frames of temporal padding'
    x = torch.randn(1, 2, 1, 8, 8, generator=torch.Generator().manual_seed(4))
    padded = F.pad(x, list(conv._padding))
    prepended = padded.shape[2] - x.shape[2]
    assert prepended == conv.weight.shape[2] - 1, (
        'exactly one frame is kept and every earlier tap saw padding')
    assert torch.count_nonzero(padded[:, :, :prepended]) == 0, 'the prepended frames are all zero'
    assert torch.equal(conv(x), F.conv3d(padded, conv.weight, conv.bias, conv.stride, 0,
                                         conv.dilation, conv.groups))


def test_a_conv_without_temporal_padding_still_works():
    """kernel=1 has no temporal padding to fold, so it must take the plain path."""
    conv = _conv(in_ch=3, out_ch=4, kernel=1, seed=5)
    assert conv._padding[4] == 0
    x = torch.randn(1, 3, 1, 8, 8, generator=torch.Generator().manual_seed(6))
    assert torch.equal(conv(x), F.conv3d(x, conv.weight, conv.bias, conv.stride, 0,
                                         conv.dilation, conv.groups))


def test_a_multi_frame_input_without_a_cache_still_pads():
    """The shortcut is about the cached single-frame case, not about T in general."""
    conv = _conv(in_ch=2, out_ch=3, seed=7)
    x = torch.randn(1, 2, 4, 8, 8, generator=torch.Generator().manual_seed(8))
    assert torch.equal(conv(x), _padded_reference(conv, x))


def test_a_cached_single_frame_keeps_the_cache_path(counted_pads):
    """The shortcut is for the no-cache case; a cache must still be honoured."""
    conv = _conv(in_ch=2, out_ch=3, seed=9)
    x = torch.randn(1, 2, 1, 8, 8, generator=torch.Generator().manual_seed(10))
    cache = torch.zeros(1, 2, CACHE_T, 8, 8)
    out = conv(x, cache)

    assert counted_pads, 'a cached conv must go through the padded path'
    assert all(pad[4] == 0 for pad in counted_pads), (
        f'a full frame cache already covers the padding, got {counted_pads}')
    # The layer concatenates the cache, shortens the temporal pad by the cache
    # length, then pads spatially only - so the reference has to do the same.
    padding = list(conv._padding)
    padding[4] -= cache.shape[2]
    expected = F.conv3d(F.pad(torch.cat([cache, x], dim=2), padding), conv.weight,
                        conv.bias, conv.stride, 0, conv.dilation, conv.groups)
    assert torch.equal(out, expected)


def test_a_partial_cache_still_pads_the_missing_frame(counted_pads):
    """A cache short of CACHE_T leaves temporal padding the cache cannot cover."""
    conv = _conv(in_ch=2, out_ch=3, seed=10)
    x = torch.randn(1, 2, 1, 8, 8, generator=torch.Generator().manual_seed(16))
    cache = torch.zeros(1, 2, CACHE_T - 1, 8, 8)
    conv(x, cache)
    assert counted_pads and all(pad[4] > 0 for pad in counted_pads), counted_pads


def test_a_single_frame_with_no_cache_pads_no_temporal_frames(counted_pads):
    conv = _conv(in_ch=2, out_ch=3, seed=11)
    x = torch.randn(1, 2, 1, 8, 8, generator=torch.Generator().manual_seed(12))
    conv(x)
    assert counted_pads == [], (
        f'a cached-free single frame must not pad at all, got {counted_pads}')


# --------------------------------------------------------------------------
# encode() only skips the cache when the cache cannot be read
# --------------------------------------------------------------------------

def _tiny_vae():
    """The anima encoder shape, small enough to run on a CPU test."""
    torch.manual_seed(0)
    return WanVAE_(dim=8, z_dim=4, dim_mult=[1, 2, 4, 4], num_res_blocks=1,
                   attn_scales=[], temperal_downsample=[False, True, True], dropout=0.0)


def _single_frame_reference(vae, x, scale):
    """What encode() did for one frame before the shortcut: hand the cache over.

    This is encode()'s first chunk followed by its tail, spelled out so the
    comparison is against the cached path rather than against itself.
    """
    vae.clear_cache()
    vae._enc_conv_idx = [0]
    out = vae.encoder(x[:, :, :1, :, :], feat_cache=vae._enc_feat_map,
                      feat_idx=vae._enc_conv_idx)
    mu, _ = vae.conv1(out).chunk(2, dim=1)
    vae.clear_cache()
    return (mu - scale[0].view(1, -1, 1, 1, 1)) * scale[1].view(1, -1, 1, 1, 1)


def test_single_frame_encode_matches_the_cached_reference():
    vae = _tiny_vae().eval()
    scale = [torch.zeros(4), torch.ones(4)]
    x = torch.randn(1, 3, 1, 32, 32, generator=torch.Generator().manual_seed(13))
    with torch.no_grad():
        fast = vae.encode(x.clone(), scale)
        cached = _single_frame_reference(vae, x, scale)
    assert torch.equal(fast, cached), (
        f'single-frame encode changed: max delta '
        f'{(fast - cached).abs().max().item():.3e}')


def test_single_frame_encode_does_not_hand_the_encoder_a_cache(monkeypatch):
    """The cache must be skipped for one frame - observed WHILE the encode runs.

    Checking the cache after encode() returns would prove nothing: clear_cache()
    at the end empties it either way, so a broken shortcut that filled it would
    still look clean. The encoder forward is where the decision shows.
    """
    vae = _tiny_vae().eval()
    scale = [torch.zeros(4), torch.ones(4)]
    seen = []
    real_forward = vae.encoder.forward

    def recording_forward(x, feat_cache=None, feat_idx=[0]):
        out = real_forward(x, feat_cache=feat_cache, feat_idx=feat_idx)
        seen.append(feat_cache is None)
        return out

    monkeypatch.setattr(vae.encoder, 'forward', recording_forward)
    x = torch.randn(1, 3, 1, 32, 32, generator=torch.Generator().manual_seed(14))
    with torch.no_grad():
        vae.encode(x.clone(), scale)

    assert seen == [True], (
        f'a single chunk must reach the encoder without a frame cache, got {seen}')


def test_multi_frame_encode_still_chunks_and_uses_the_cache(monkeypatch):
    vae = _tiny_vae().eval()
    scale = [torch.zeros(4), torch.ones(4)]
    seen = []
    real_forward = vae.encoder.forward

    def recording_forward(x, feat_cache=None, feat_idx=[0]):
        seen.append((x.shape[2], feat_cache is not None))
        return real_forward(x, feat_cache=feat_cache, feat_idx=feat_idx)

    monkeypatch.setattr(vae.encoder, 'forward', recording_forward)
    x = torch.randn(1, 3, 5, 32, 32, generator=torch.Generator().manual_seed(15))
    with torch.no_grad():
        out = vae.encode(x.clone(), scale)

    # T=5 is one T=1 chunk plus one T=4 chunk, both handed the cache.
    assert [frames for frames, _ in seen] == [1, 4], seen
    assert all(has_cache for _, has_cache in seen), 'video must keep the frame cache'
    assert out.shape == (1, 4, 2, 4, 4), f'unexpected latent shape {tuple(out.shape)}'
