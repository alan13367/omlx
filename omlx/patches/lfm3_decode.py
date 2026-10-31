# SPDX-License-Identifier: Apache-2.0
"""Fewer launches for LFM3 (mlx-vlm ``models/lfm3``) text decode.

At batch one a decode step of LFM3-1.7B is short (about 1.7 GB of 8-bit
weights) and its kernels are tiny and serially dependent, so launches, not
bandwidth, set the step time. The stock step issues about 800 kernels; this
module removes most of the ones that move no weights.

RoPE (every call, prefill included). mlx-vlm writes the rotation out in array
ops: per call it builds positions and inverse frequencies (``arange``,
``power``, ``pad``), the angles, ``cos``/``sin`` and their casts, then rotates
with a split, negate, concatenate and three elementwise ops, about fifteen
launches for each of q and k. The rotation is non-traditional (half-split)
RoPE over the full head dim whose frequencies past ``partial_rotary_factor``
are zero, i.e. ``mx.fast.rope`` with a frequency table that gives those pairs
an infinite period (angle 0, identity): one launch per call. The fast kernel
forms the angle and rotates in fp32 with one rounding, where the stock path
rounds cos/sin to the activation dtype first, so outputs differ by rounding
only (and sit closer to an fp32 reference). Integer offsets and the per-row
offset arrays of batched caches are both served; explicit ``position_ids``
keep the stock body.

Decode rows (``B * L <= MAX_ROWS``; one token per sequence, or a short
chunk of one sequence) of a sandwich-norm model run each block as:

* the mixer: attention layers unchanged (with the fast RoPE); short-conv
  layers ``in_proj`` rows B, C and x of a channel block plus the K-tap
  depthwise conv step and state shift in one launch (one-token rows only),
  then the stock ``out_proj``;
* one ``add_norm`` launch: ``x + post_operator_norm(h)`` and the
  ``ffn_norm`` of that sum;
* the MLP: gate and up rows plus SwiGLU in one launch, then the stock
  ``w2``;
* one ``add_norm`` launch: ``x + post_ffn_norm(m)`` and the next block's
  ``operator_norm`` of that sum (the final ``embedding_norm`` after the last
  block), so no block runs its own pre-norm.

Norms use MLX's ``rms_norm`` arithmetic and the quantized dot products MLX's
``qmv`` arithmetic (``qwen35_moe_routed_decode.qmv_rows``); the conv taps
accumulate in fp32 like ``conv1d``. Anything outside the supported layout
(no sandwich norm, masks on conv layers, ragged batch caches, an
unquantized or mixed-format MLP, ...) keeps the stock call for that piece,
and a failure while building a launch disables the fused path for the
process.
"""

from __future__ import annotations

import logging
import math
from functools import cache
from typing import NamedTuple

import mlx.core as mx
import mlx.nn as nn

from .module_cache import cached_per_module
from .qwen35_moe_routed_decode import (
    _COMMON,
    _Format,
    _format_header,
    _mlx_format,
    _name,
    _quantized_ok,
)

logger = logging.getLogger(__name__)

MAX_ROWS = 8

_APPLIED = False
_DISABLED = False
_NORM_SIMDGROUPS = 8
_MLP_ROWS = 2
_MLP_SIMDGROUPS = 2
_CONV_ROWS = 2
_CONV_SIMDGROUPS = 2


def _float(v: float) -> str:
    return repr(float(v)) + "f"


# ---------------------------------------------------------------- RoPE
def rope_periods(dim: int, base: float, angles: int) -> mx.array:
    """Per-pair periods for ``mx.fast.rope``: ``base ** (2 i / dim)`` for the
    first ``angles`` pairs, infinity (no rotation) for the rest."""
    i = mx.arange(dim // 2, dtype=mx.float32)
    periods = mx.where(
        i < angles, base ** (2 * i / dim), mx.array(math.inf, dtype=mx.float32)
    )
    mx.eval(periods)
    return periods


def _patched_rope_call(orig):
    def patched(self, x, offset=0, position_ids=None):
        if position_ids is not None or x.ndim != 4:
            return orig(self, x, offset, position_ids)
        periods = self.__dict__.get("_omlx_periods")
        if periods is None:
            periods = rope_periods(self.dim, float(self.base), int(self.angles))
            # Instance attribute, not a module member, so it never reaches
            # parameters(), the weights or quantization.
            object.__setattr__(self, "_omlx_periods", periods)
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=periods,
        )

    patched._omlx_lfm3_fast_rope = True
    return patched


# ---------------------------------------------------------------- kernels
# One threadgroup per row: xo = x + w1 * rms(h), yo = w2 * rms(xo), each
# with MLX rms_norm's arithmetic (fp32 sum of squares, precise rsqrt, the
# normalized value rounded to T before the weight product in T).
_ADD_NORM_SRC = r"""
    constexpr int NT = NSG * 32;
    constexpr int CH = H / NT;
    const uint sg = simdgroup_index_in_threadgroup;
    const uint lane = thread_index_in_simdgroup;
    const int tid = int(sg) * 32 + int(lane);
    const int row = int(threadgroup_position_in_grid.x);
    x += row * H;
    h += row * H;
    xo += row * H;
    yo += row * H;
    threadgroup float red1[NSG];
    threadgroup float red2[NSG];
    float hv[CH];
    float ss = 0;
    for (int c = 0; c < CH; c++) {
      hv[c] = float(h[c * NT + tid]);
      ss += hv[c] * hv[c];
    }
    ss = simd_sum(ss);
    if (lane == 0) {
      red1[sg] = ss;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot = 0;
    for (int s = 0; s < NSG; s++) {
      tot += red1[s];
    }
    const float inv1 = metal::precise::rsqrt(tot / H + EPS);
    float xv[CH];
    float ss2 = 0;
    for (int c = 0; c < CH; c++) {
      const int i = c * NT + tid;
      const T p = w1[i] * static_cast<T>(hv[c] * inv1);
      const T s = static_cast<T>(float(x[i]) + float(p));
      xo[i] = s;
      xv[c] = float(s);
      ss2 += xv[c] * xv[c];
    }
    ss2 = simd_sum(ss2);
    if (lane == 0) {
      red2[sg] = ss2;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float tot2 = 0;
    for (int s = 0; s < NSG; s++) {
      tot2 += red2[s];
    }
    const float inv2 = metal::precise::rsqrt(tot2 / H + EPS);
    for (int c = 0; c < CH; c++) {
      const int i = c * NT + tid;
      yo[i] = w2[i] * static_cast<T>(xv[c] * inv2);
    }
"""

# Simdgroup rows d.. of w1 (gate) and w3 (up) for each of the R input rows,
# then silu(gate) * up as mlx's nn.silu(w1(x)) * w3(x) rounds it.
_GATE_UP_SRC = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int d = (int(threadgroup_position_in_grid.y) * NSG + int(simd_gid)) * RPS;
    for (int r = 0; r < R; r++) {
      float result[2 * RPS];
      gu::qmv_rows<T, K, RPS, RPS>(
          (const device uint8_t*)w1, s1, b1, d,
          (const device uint8_t*)w3, s3, b3, d,
          x + r * K, simd_lid, result);
      swiglu_store<T, RPS>(result, y + r * N + d, simd_lid);
    }
"""

# Simdgroup rows d.., H + d.., 2H + d.. of in_proj (B, C, x of a channel
# block) for each of the S sequences (one token each), then the depthwise
# conv over the KM1 state taps and the new B * x, and the state shift:
#   bx = B * x; y = C * sum_k w_k s_k + w_KM1 bx; state <- (s_1.., bx)
_CONV_SRC = r"""
    const uint simd_gid = simdgroup_index_in_threadgroup;
    const uint simd_lid = thread_index_in_simdgroup;
    const int d = (int(threadgroup_position_in_grid.y) * NSG + int(simd_gid)) * RPS;
    for (int b = 0; b < S; b++) {
      float rb[2 * RPS];
      float rx[RPS];
      ct::qmv_rows<T, H, RPS, RPS>(
          (const device uint8_t*)w, scales, biases, d,
          (const device uint8_t*)w, scales, biases, H + d,
          x + b * H, simd_lid, rb);
      ct::qmv_rows<T, H, RPS, 0>(
          (const device uint8_t*)w, scales, biases, 2 * H + d,
          (const device uint8_t*)w, scales, biases, 2 * H + d,
          x + b * H, simd_lid, rx);
      if (simd_lid == 0) {
        const device T* st = state + size_t(b) * KM1 * H;
        device T* ns = new_state + size_t(b) * KM1 * H;
        for (int r = 0; r < RPS; r++) {
          const int c = d + r;
          const T bx = static_cast<T>(rb[r]) * static_cast<T>(rx[r]);
          float acc = 0;
          for (int k = 0; k < KM1; k++) {
            acc += float(cw[c * (KM1 + 1) + k]) * float(st[k * H + c]);
          }
          acc += float(cw[c * (KM1 + 1) + KM1]) * float(bx);
          y[b * H + c] = static_cast<T>(rb[RPS + r]) * static_cast<T>(acc);
          for (int k = 0; k + 1 < KM1; k++) {
            ns[k * H + c] = st[(k + 1) * H + c];
          }
          ns[(KM1 - 1) * H + c] = bx;
        }
      }
    }
"""


@cache
def _add_norm_kernel(eps: float):
    return mx.fast.metal_kernel(
        name=f"omlx_lfm3_add_norm_{abs(hash(eps)):x}",
        input_names=["x", "h", "w1", "w2"],
        output_names=["xo", "yo"],
        header="using namespace metal;\n",
        source=_ADD_NORM_SRC.replace("EPS", _float(eps)),
    )


@cache
def _gate_up_kernel(fmt: _Format):
    return mx.fast.metal_kernel(
        name=f"omlx_lfm3_gate_up_swiglu_{_name(fmt)}",
        input_names=["x", "w1", "s1", "b1", "w3", "s3", "b3"],
        output_names=["y"],
        header=_COMMON + _format_header("gu", fmt),
        source=_GATE_UP_SRC,
    )


@cache
def _conv_kernel(fmt: _Format):
    return mx.fast.metal_kernel(
        name=f"omlx_lfm3_conv_in_step_{_name(fmt)}",
        input_names=["x", "w", "scales", "biases", "cw", "state"],
        output_names=["y", "new_state"],
        header=_COMMON + _format_header("ct", fmt),
        source=_CONV_SRC,
    )


# ---------------------------------------------------------------- plans
def _operands(layer) -> tuple:
    return (layer["weight"], layer["scales"], layer["biases"])


class _MLPPlan(NamedTuple):
    hidden: int
    inter: int
    dtype: mx.Dtype
    fmt: _Format
    operands: tuple


class _ConvPlan(NamedTuple):
    hidden: int
    taps: int
    dtype: mx.Dtype
    fmt: _Format
    operands: tuple
    conv_weight: mx.array


def _build_mlp_plan(mlp) -> _MLPPlan | None:
    w1, w3, w2 = mlp.get("w1"), mlp.get("w3"), mlp.get("w2")
    if not (
        _quantized_ok(w1, nn.QuantizedLinear)
        and _quantized_ok(w3, nn.QuantizedLinear)
        and w2 is not None
        and w1.bits == w3.bits
        and w1.group_size == w3.group_size
    ):
        return None
    dtype = w1["scales"].dtype
    hidden = w1["weight"].shape[1] * 32 // w1.bits
    inter = w1["weight"].shape[0]
    fmt = _mlx_format(w1, hidden, inter)
    if (
        fmt is None
        or fmt != _mlx_format(w3, hidden, inter)
        or w3["scales"].dtype != dtype
        or inter % (_MLP_ROWS * _MLP_SIMDGROUPS)
    ):
        return None
    return _MLPPlan(hidden, inter, dtype, fmt, _operands(w1) + _operands(w3))


def _build_conv_plan(conv) -> _ConvPlan | None:
    in_proj, conv1d = conv.get("in_proj"), conv.get("conv")
    if not (
        _quantized_ok(in_proj, nn.QuantizedLinear)
        and type(conv1d) is nn.Conv1d
        and "bias" not in conv1d
        and getattr(conv, "causal", True)
    ):
        return None
    dtype = in_proj["scales"].dtype
    hidden = int(conv.args.hidden_size)
    taps = int(conv.L_cache)
    fmt = _mlx_format(in_proj, hidden, 3 * hidden)
    weight = conv1d["weight"]
    if (
        fmt is None
        or taps < 2
        or weight.shape != (hidden, taps, 1)
        or weight.dtype != dtype
        or hidden % (_CONV_ROWS * _CONV_SIMDGROUPS)
    ):
        return None
    return _ConvPlan(hidden, taps, dtype, fmt, _operands(in_proj), weight)


# ---------------------------------------------------------------- launches
def add_norm(x: mx.array, h: mx.array, w1: mx.array, w2: mx.array, eps: float):
    """``(x + rms_norm(h, w1), rms_norm(that, w2))`` for ``[..., H]`` rows."""
    hidden = x.shape[-1]
    rows = x.size // hidden
    return _add_norm_kernel(eps)(
        inputs=[x, h, w1, w2],
        template=[("T", x.dtype), ("H", hidden), ("NSG", _NORM_SIMDGROUPS)],
        grid=(32 * _NORM_SIMDGROUPS * rows, 1, 1),
        threadgroup=(32 * _NORM_SIMDGROUPS, 1, 1),
        output_shapes=[x.shape, x.shape],
        output_dtypes=[x.dtype, x.dtype],
    )


def gate_up(plan: _MLPPlan, x: mx.array) -> mx.array:
    """``silu(w1(x)) * w3(x)`` for ``x`` [..., H]."""
    rows = x.size // plan.hidden
    return _gate_up_kernel(plan.fmt)(
        inputs=[x, *plan.operands],
        template=[
            ("T", plan.dtype),
            ("K", plan.hidden),
            ("N", plan.inter),
            ("RPS", _MLP_ROWS),
            ("NSG", _MLP_SIMDGROUPS),
            ("R", rows),
        ],
        grid=(32, plan.inter // _MLP_ROWS, 1),
        threadgroup=(32, _MLP_SIMDGROUPS, 1),
        output_shapes=[(*x.shape[:-1], plan.inter)],
        output_dtypes=[plan.dtype],
    )[0]


def conv_step(plan: _ConvPlan, x: mx.array, cache) -> mx.array:
    """The short-conv mixer before ``out_proj`` for one token per sequence
    (``x`` [S, 1, H]), advancing ``cache``."""
    seqs, hidden, dtype = x.shape[0], plan.hidden, plan.dtype
    state = cache[0]
    if state is None:
        state = mx.zeros((seqs, plan.taps - 1, hidden), dtype=dtype)
    y, new_state = _conv_kernel(plan.fmt)(
        inputs=[x, *plan.operands, plan.conv_weight, state],
        template=[
            ("T", dtype),
            ("H", hidden),
            ("KM1", plan.taps - 1),
            ("RPS", _CONV_ROWS),
            ("NSG", _CONV_SIMDGROUPS),
            ("S", seqs),
        ],
        grid=(32, hidden // _CONV_ROWS, 1),
        threadgroup=(32, _CONV_SIMDGROUPS, 1),
        output_shapes=[x.shape, (seqs, plan.taps - 1, hidden)],
        output_dtypes=[dtype, dtype],
    )
    # Only after the launch was built, so a failure leaves the cache as is.
    cache[0] = new_state
    cache.advance(1)
    return y


# ---------------------------------------------------------------- model
def _conv_ok(cache, mask, x, plan) -> bool:
    # One sequence only: the kernel re-reads in_proj per sequence, which
    # loses to MLX's multi-row qmv once the weights fall out of cache.
    if plan is None or mask is not None or cache is None or x.shape[:2] != (1, 1):
        return False
    state = cache[0]
    return (
        getattr(cache, "lengths", None) is None
        and getattr(cache, "left_padding", None) is None
        and (
            state is None
            or tuple(state.shape) == (x.shape[0], plan.taps - 1, plan.hidden)
        )
        and plan.dtype == x.dtype
    )


def _norms_ok(model) -> float | None:
    """The shared norm eps when every block is a sandwich-norm block whose
    norms are plain ``nn.RMSNorm`` with one eps, else None."""
    eps = model.embedding_norm.eps
    if type(model.embedding_norm) is not nn.RMSNorm:
        return None
    for layer in model.layers:
        if not getattr(layer, "use_sandwich_norm", False):
            return None
        for name in (
            "operator_norm",
            "post_operator_norm",
            "ffn_norm",
            "post_ffn_norm",
        ):
            norm = layer.get(name)
            if type(norm) is not nn.RMSNorm or norm.eps != eps:
                return None
    return float(eps)


def _decode_rows(h: mx.array) -> bool:
    return (
        h.ndim == 3
        and h.shape[0] * h.shape[1] <= MAX_ROWS
        and h.dtype in (mx.float16, mx.bfloat16)
        and h.shape[-1] % (32 * _NORM_SIMDGROUPS) == 0
    )


def _plans(model) -> list:
    """(conv plan or None, MLP plan or None) per block, cached per module."""
    out = []
    for layer in model.layers:
        conv = None
        if not layer.is_attention_layer:
            conv = cached_per_module(
                layer.conv, "_omlx_lfm3_conv_plan", _build_conv_plan
            )
        mlp = cached_per_module(
            layer.feed_forward, "_omlx_lfm3_mlp_plan", _build_mlp_plan
        )
        out.append((conv, mlp))
    return out


def _fused_forward(model, lfm3, h, cache, eps, plans):
    layers = model.layers
    config = model.config
    xn = layers[0].operator_norm(h)
    for i, (layer, c, (conv_plan, plan)) in enumerate(zip(layers, cache, plans)):
        if layer.is_attention_layer:
            window = (
                config.sliding_window
                if layer.layer_type == "sliding_attention"
                else None
            )
            mask = lfm3.create_attention_mask(h, c, window_size=window)
            o = layer.self_attn(xn, mask, c, None)
        else:
            mask = lfm3.create_ssm_mask(h, c)
            conv = layer.conv
            if _conv_ok(c, mask, xn, conv_plan):
                o = conv.out_proj(conv_step(conv_plan, xn, c))
            else:
                o = conv(xn, mask, c)
        h, y = add_norm(
            h, o, layer.post_operator_norm.weight, layer.ffn_norm.weight, eps
        )
        mlp = layer.feed_forward
        # One row only: the gate/up kernel re-reads the weights per row
        # (batch-8 decode measured -31% against MLX's multi-row qmv).
        if plan is not None and plan.dtype == y.dtype and y.size == plan.hidden:
            m = mlp.w2(gate_up(plan, y))
        else:
            m = mlp(y)
        nxt = (
            layers[i + 1].operator_norm.weight
            if i + 1 < len(layers)
            else model.embedding_norm.weight
        )
        h, xn = add_norm(h, m, layer.post_ffn_norm.weight, nxt, eps)
    return xn


def _patched_model_call(orig, lfm3):
    def patched(
        self, inputs=None, cache=None, input_embeddings=None, position_ids=None
    ):
        global _DISABLED
        if _DISABLED or self.training or position_ids is not None or cache is None:
            return orig(self, inputs, cache, input_embeddings, position_ids)
        h = self.embed_tokens(inputs) if input_embeddings is None else input_embeddings
        if not _decode_rows(h):
            return orig(self, inputs, cache, input_embeddings, position_ids)
        eps = cached_per_module(self, "_omlx_lfm3_norm_eps", _norms_ok, depth=2)
        try:
            plans = None if eps is None else _plans(self)
        except Exception:
            _DISABLED = True
            logger.warning("LFM3 fused decode disabled", exc_info=True)
            plans = None
        if plans is None:
            return orig(self, inputs, cache, input_embeddings, position_ids)
        try:
            return _fused_forward(self, lfm3, h, cache, eps, plans)
        except Exception:
            # An attention layer may already have appended this step's keys,
            # so the step cannot be rerun stock; later steps run stock.
            _DISABLED = True
            logger.warning("LFM3 fused decode disabled", exc_info=True)
            raise

    patched._omlx_lfm3_fused_decode = True
    return patched


def apply_lfm3_decode_patch() -> bool:
    """Install the LFM3 fast paths on mlx-vlm's ``models.lfm3.language``.
    Idempotent; returns whether they are installed."""
    global _APPLIED
    if _APPLIED:
        return True
    if not mx.metal.is_available():
        return False
    try:
        from mlx_vlm.models.lfm3 import language as lfm3
    except ImportError:
        return False
    rope = lfm3.RotaryEmbedding
    if not getattr(rope.__call__, "_omlx_lfm3_fast_rope", False):
        rope.__call__ = _patched_rope_call(rope.__call__)
    model = lfm3.Lfm3Model
    if not getattr(model.__call__, "_omlx_lfm3_fused_decode", False):
        model.__call__ = _patched_model_call(model.__call__, lfm3)
    _APPLIED = True
    logger.info("LFM3 decode patch applied")
    return True


__all__ = [
    "MAX_ROWS",
    "add_norm",
    "apply_lfm3_decode_patch",
    "conv_step",
    "gate_up",
    "rope_periods",
]
