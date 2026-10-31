# SPDX-License-Identifier: Apache-2.0
"""Tests for the LFM3 fast RoPE and fused decode launches."""

from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
lfm3 = pytest.importorskip("mlx_vlm.models.lfm3.language")

from mlx_vlm.models.lfm3.config import ModelConfig  # noqa: E402

from omlx.patches import lfm3_decode  # noqa: E402

pytestmark = pytest.mark.skipif(
    not mx.metal.is_available(), reason="LFM3 decode kernels need Metal"
)

_STOCK_ROPE = lfm3.RotaryEmbedding.__call__
_STOCK_MODEL = lfm3.Lfm3Model.__call__


@pytest.fixture
def patched():
    lfm3_decode._APPLIED = False
    lfm3_decode._DISABLED = False
    assert lfm3_decode.apply_lfm3_decode_patch()
    yield
    lfm3.RotaryEmbedding.__call__ = _STOCK_ROPE
    lfm3.Lfm3Model.__call__ = _STOCK_MODEL
    lfm3_decode._APPLIED = False
    lfm3_decode._DISABLED = False


def _config(**overrides):
    kwargs = dict(
        hidden_size=512,
        intermediate_size=1024,
        num_hidden_layers=6,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=128,
        vocab_size=256,
        conv_L_cache=7,
        layer_types=[
            "conv",
            "full_attention",
            "conv",
            "sliding_attention",
            "conv",
            "full_attention",
        ],
        sliding_window=8,
    )
    kwargs.update(overrides)
    return ModelConfig(**kwargs)


def _model(quantize_bits=8, dtype=mx.bfloat16, seed=0, **overrides):
    mx.random.seed(seed)
    model = lfm3.LanguageModel(_config(**overrides))
    # Random init gives norms of weight 1; perturb them so a wrong norm
    # weight (or order) shows up in the outputs.
    params = nn.utils.tree_flatten(model.parameters())
    model.update(
        nn.utils.tree_unflatten(
            [
                (
                    k,
                    v + 0.1 * mx.random.normal(v.shape)
                    if k.endswith("norm.weight")
                    else v,
                )
                for k, v in params
            ]
        )
    )
    if quantize_bits:
        nn.quantize(
            model,
            bits=quantize_bits,
            group_size=64,
            class_predicate=lambda p, m: isinstance(m, nn.Linear),
        )
    model.set_dtype(dtype)
    model.eval()  # loaded checkpoints are in eval mode; training keeps stock
    mx.eval(model.parameters())
    return model


def _close(a, b, tol):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return (mx.abs(a - b).max() / (mx.abs(b).max() + 1e-6)).item() < tol


# ---------------------------------------------------------------- RoPE
@pytest.mark.parametrize(
    "params", [{"rope_theta": 1e6, "partial_rotary_factor": 0.5}, {"rope_theta": 1e4}]
)
@pytest.mark.parametrize(
    "shape,offset",
    [((1, 4, 1, 128), 37), ((1, 4, 9, 128), 0), ((3, 4, 1, 128), "array")],
)
def test_fast_rope_matches_stock(patched, params, shape, offset):
    rope = lfm3.RotaryEmbedding(128, params)
    if offset == "array":
        offset = mx.array([3, 500, 41])
    x = mx.random.normal(shape, key=mx.random.key(1))
    ref = _STOCK_ROPE(rope, x, offset)
    out = rope(x, offset)
    assert mx.allclose(out, ref, atol=1e-4, rtol=1e-4).item()


def test_fast_rope_leaves_nope_pairs_unrotated(patched):
    rope = lfm3.RotaryEmbedding(128, {"rope_theta": 1e6, "partial_rotary_factor": 0.5})
    x = mx.random.normal((1, 2, 5, 128), key=mx.random.key(2))
    out = rope(x, 1000)
    for half in (0, 64):
        nope = slice(half + 32, half + 64)
        assert mx.array_equal(out[..., nope], x[..., nope]).item()


def test_rope_periods_are_infinite_past_partial_factor():
    periods = lfm3_decode.rope_periods(128, 1e6, 32)
    assert periods.shape == (64,)
    assert all(math.isinf(v) for v in periods[32:].tolist())
    assert periods[1].item() == pytest.approx(1e6 ** (2 / 128), rel=1e-6)


def test_fast_rope_keeps_position_ids_path(patched):
    rope = lfm3.RotaryEmbedding(128, {"rope_theta": 1e4})
    x = mx.random.normal((1, 2, 3, 128), key=mx.random.key(3))
    pos = mx.array([[4, 9, 30]])
    assert mx.array_equal(rope(x, 0, pos), _STOCK_ROPE(rope, x, 0, pos)).item()


def test_rope_periods_are_not_parameters(patched):
    model = _model(quantize_bits=0)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache=cache)
    names = [k for k, _ in nn.utils.tree_flatten(model.parameters())]
    assert not any("periods" in k for k in names)


# ---------------------------------------------------------------- kernels
@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("rows", [1, 3])
def test_add_norm_matches_stock(dtype, rows):
    mx.random.seed(rows)
    hidden, eps = 512, 1e-5
    x = mx.random.normal((rows, 1, hidden)).astype(dtype)
    h = (3 * mx.random.normal((rows, 1, hidden))).astype(dtype)
    w1 = (1 + 0.1 * mx.random.normal((hidden,))).astype(dtype)
    w2 = (1 + 0.1 * mx.random.normal((hidden,))).astype(dtype)
    ref_x = x + mx.fast.rms_norm(h, w1, eps)
    ref_y = mx.fast.rms_norm(ref_x, w2, eps)
    xo, yo = lfm3_decode.add_norm(x, h, w1, w2, eps)
    assert xo.dtype == dtype and yo.dtype == dtype
    # Same arithmetic; only the fp32 sum-of-squares order differs.
    assert _close(xo, ref_x, 1e-2)
    assert _close(yo, ref_y, 1e-2)
    assert (
        mx.abs(xo.astype(mx.float32) - ref_x.astype(mx.float32)) > 0
    ).mean().item() < 0.02


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("rows", [1, 4])
def test_gate_up_matches_stock(bits, rows):
    model = _model(quantize_bits=bits)
    mlp = model.model.layers[0].feed_forward
    plan = lfm3_decode._build_mlp_plan(mlp)
    assert plan is not None
    x = mx.random.normal((rows, 1, 512), key=mx.random.key(rows)).astype(mx.bfloat16)
    ref = nn.silu(mlp.w1(x)) * mlp.w3(x)
    out = lfm3_decode.gate_up(plan, x)
    assert out.shape == ref.shape
    assert _close(out, ref, 1e-2)


def test_mlp_plan_rejects_unquantized_and_mixed_bits():
    assert (
        lfm3_decode._build_mlp_plan(
            _model(quantize_bits=0).model.layers[0].feed_forward
        )
        is None
    )
    model = _model(quantize_bits=0)
    mlp = model.model.layers[0].feed_forward
    mlp.w1 = nn.QuantizedLinear.from_linear(mlp.w1, group_size=64, bits=4)
    mlp.w3 = nn.QuantizedLinear.from_linear(mlp.w3, group_size=64, bits=8)
    assert lfm3_decode._build_mlp_plan(mlp) is None


@pytest.mark.parametrize("taps", [3, 7])
@pytest.mark.parametrize("seqs", [1, 3])
def test_conv_step_matches_stock(taps, seqs):
    model = _model(conv_L_cache=taps)
    conv = model.model.layers[0].conv
    plan = lfm3_decode._build_conv_plan(conv)
    assert plan is not None and plan.taps == taps
    stock_cache = lfm3.ArraysCache(size=1)
    fused_cache = lfm3.ArraysCache(size=1)
    prompt = mx.random.normal((seqs, 5, 512), key=mx.random.key(seqs)).astype(
        mx.bfloat16
    )
    conv(prompt, None, stock_cache)
    conv(prompt, None, fused_cache)
    for step in range(4):
        x = mx.random.normal((seqs, 1, 512), key=mx.random.key(10 + step)).astype(
            mx.bfloat16
        )
        ref = conv(x, None, stock_cache)
        out = conv.out_proj(lfm3_decode.conv_step(plan, x, fused_cache))
        assert _close(out, ref, 2e-2)
        assert _close(fused_cache[0], stock_cache[0], 1e-2)


# ---------------------------------------------------------------- model
def _decode(model, prompt, steps):
    cache = model.make_cache()
    logits = [model(prompt, cache=cache).logits[:, -1]]
    tok = prompt[:, -1:]
    for i in range(steps):
        tok = (tok + 7 * i + 1) % 256  # teacher-forced, same for both arms
        logits.append(model(tok, cache=cache).logits[:, -1])
    out = mx.stack(logits, 1).astype(mx.float32)
    mx.eval(out)
    return out


def _kl(ref, out):
    a = ref - mx.logsumexp(ref, -1, keepdims=True)
    b = out - mx.logsumexp(out, -1, keepdims=True)
    return (mx.exp(a) * (a - b)).sum(-1)


@pytest.mark.parametrize("bits", [8, 4, 0])
@pytest.mark.parametrize("batch", [1, 3])
def test_fused_decode_matches_stock(bits, batch):
    model = _model(quantize_bits=bits)
    prompt = mx.random.randint(0, 256, (batch, 12), key=mx.random.key(batch))
    ref = _decode(model, prompt, 20)  # crosses the 8-token sliding window
    lfm3_decode._APPLIED = False
    try:
        assert lfm3_decode.apply_lfm3_decode_patch()
        out = _decode(model, prompt, 20)
        assert not lfm3_decode._DISABLED
        assert lfm3_decode._plans(model.model)[0][1] is not None or not bits
    finally:
        lfm3.RotaryEmbedding.__call__ = _STOCK_ROPE
        lfm3.Lfm3Model.__call__ = _STOCK_MODEL
        lfm3_decode._APPLIED = False
    # Random tiny models have near-tied logits, so top-1 is no signal here;
    # the distributions must agree to rounding.
    assert _kl(ref, out).max().item() < 1e-3


def test_fused_decode_runs_custom_kernels(patched, tmp_path):
    model = _model(quantize_bits=8)
    cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3, 4]]), cache=cache).logits, cache[0][0])
    hidden = model.model(mx.array([[5]]), cache=cache)
    # 2 add_norm per block, gate_up per block, the conv step per conv block.
    layers = model.model.layers
    expected = 3 * len(layers) + sum(not layer.is_attention_layer for layer in layers)
    path = tmp_path / "decode.dot"
    mx.export_to_dot(str(path), hidden)
    assert path.read_text().count('label ="CustomKernel') == expected


def test_stock_paths_without_sandwich_norm(patched):
    model = _model(quantize_bits=8, use_sandwich_norm=False)
    cache = model.make_cache()
    model(mx.array([[1, 2, 3]]), cache=cache)
    out = model(mx.array([[4]]), cache=cache).logits
    mx.eval(out)
    assert lfm3_decode._norms_ok(model.model) is None
    assert not lfm3_decode._DISABLED


def test_prefill_rows_keep_stock_forward(patched):
    model = _model(quantize_bits=8)
    prompt = mx.random.randint(0, 256, (1, 40), key=mx.random.key(5))
    cache_a, cache_b = model.make_cache(), model.make_cache()
    out = model(prompt, cache=cache_a).logits
    lfm3.Lfm3Model.__call__ = _STOCK_MODEL
    ref = model(prompt, cache=cache_b).logits
    # Prefill differs from stock only through RoPE rounding.
    assert _close(out, ref, 5e-2)
