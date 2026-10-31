# SPDX-License-Identifier: MIT
"""Token-normalized Engram lookup with request-local lookback."""

import math
from functools import cache

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .gemv import mxfp8_gemv_each
from .quantization import QuantizedProjection, quantize_activation
from .storage import dedupe_tuple_ids


def build_compressed_token_map(tokenizer) -> tuple[list[int], int]:
    """Map every token id onto a smaller id space where tokens that normalize alike collapse together.

    N-grams are hashed over these compressed ids, so " The", "the" and "THE" all hash the same way.
    Returns the lookup plus the size of the compressed vocab -- and that size matters beyond bounds
    checking, because every hash multiplier is derived from it.
    """
    from tokenizers import Regex, normalizers

    # a private-use char, so a token that is exactly one space survives Strip() instead of
    # collapsing to the empty string and merging with unrelated tokens
    sentinel = "\ue000"
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )

    # the raw Rust tokenizer, matching what training decodes with (no clean_up_tokenization_spaces)
    backend = tokenizer.backend_tokenizer
    key_to_new: dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "\ufffd" in text:
            # a partial UTF-8 byte token: nothing to normalize, so key it by its raw form
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text

        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id

    return lookup, len(key_to_new)


def _prime(n):
    return n >= 2 and all(n % d for d in range(2, math.isqrt(n) + 1))


class NgramHash:
    """CPU int64 hashing matches the official NumPy RNG, not Qwen's SplitMix64.

    Only the last n-1 compressed token ids live in the request cache. Image
    positions stop lookback; they are not removed from the token timeline.
    """

    def __init__(self, config, token_map):
        self.config = config
        self.token_map = np.asarray(token_map, dtype=np.int64)
        vocab = int(self.token_map.max()) + 1
        if vocab != config.engram_compressed_vocab_size:
            raise ValueError(f"Engram compressed vocabulary mismatch: {vocab}")
        self.pad_id = int(self.token_map[config.engram_pad_id])
        primes, multipliers, seen = [], [], set()
        for layer_id in config.engram_layer_ids:
            groups = []
            for _ in range(config.engram_max_ngram_size - 1):
                current, group = config.engram_vocab_size - 1, []
                for _ in range(config.engram_n_heads):
                    current += 1
                    while not _prime(current) or current in seen:
                        current += 1
                    seen.add(current)
                    group.append(current)
                groups.append(group)
            primes.append(groups)
            rng = np.random.default_rng(10007 * layer_id)
            bound = max(1, (np.iinfo(np.int64).max // vocab) // 2)
            multipliers.append(
                rng.integers(0, bound, config.engram_max_ngram_size, dtype=np.int64) * 2
                + 1
            )
        self.primes = np.array(primes, dtype=np.int64)
        self.multipliers = np.array(multipliers, dtype=np.int64)
        flat = self.primes.reshape(len(primes), -1)
        self.offsets = np.cumsum(
            np.concatenate(
                [np.zeros((len(primes), 1), dtype=np.int64), flat[:, :-1]], -1
            ),
            -1,
        )
        if list(flat.sum(-1)) != list(config.engram_num_embeddings):
            raise ValueError("Engram table rows do not match the official prime layout")

    def __call__(self, ids, history=None, image_mask=None):
        ids = np.asarray(ids, dtype=np.int64)
        if ids.min() < 0 or ids.max() >= len(self.token_map):
            raise ValueError("Token id outside Engram tokenizer vocabulary")
        tokens = self.token_map[ids]
        if image_mask is not None:
            tokens = np.where(np.asarray(image_mask), -1, tokens)
        depth = self.config.engram_max_ngram_size - 1
        if history is None:
            history = np.full((len(ids), depth), -1, dtype=np.int64)
        joined = np.concatenate([np.asarray(history), tokens], axis=1)
        positions = np.arange(tokens.shape[1]) + depth
        blocked = np.zeros_like(tokens, dtype=bool)
        lookback = []
        for shift in range(depth + 1):
            source = joined[:, positions - shift]
            blocked |= source == -1
            lookback.append(np.where(blocked, self.pad_id, source))
        product = np.stack(lookback, -1)[:, :, None, :] * self.multipliers
        rolling, hashes = product[..., 0], []
        for shift in range(1, depth + 1):
            rolling = np.bitwise_xor(rolling, product[..., shift])
            hashes.append(rolling[..., None] % self.primes[:, shift - 1])
        return np.concatenate(hashes, -1) + self.offsets, joined[:, -depth:]


# Inputs of up to this many rows (batched decode, MTP verify) are projected row by row.
_ROWWISE_ROWS = 8


def _project_each(p, flat):
    """p applied to each row of flat ([count, k]) on its own, with one weight read for all
    rows (mxfp8_gemv_each), or None if p is not an MXFP8 projection."""
    if not (isinstance(p, QuantizedProjection) and p.mode == "mxfp8" and p.group_size == 32):
        return None
    return mxfp8_gemv_each(quantize_activation(flat) if p.quantize_input else flat, p.weight, p.scales)


_GATE_HEADER = r"""
#pragma clang fp reassociate(off)
#pragma clang fp contract(off)

inline float v41_engram_sigmoid(float g) {
    const float neg = 1.0f / (1.0f + metal::precise::exp(abs(g)));
    return g < 0 ? neg : 1.0f - neg;
}
inline float v41_engram_rsqrt_mean(float sumsq, float n, float eps) {
    return metal::precise::rsqrt(sumsq * (1.0f / n) + eps);
}
"""

_GATE_SOURCE = r"""
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup;
    const uint simd = simdgroup_index_in_threadgroup;
    const uint token = row / HC;
    const uint c = row % HC;
    const uint kv_row = uint(inv[token]);
    const uint hrow = row * D;
    const uint krow = kv_row * (HC + 1) * D + c * D;
    const uint vrow = kv_row * (HC + 1) * D + HC * D;

    threadgroup float part[TG / 32][3];
    float sx = 0.0f, sk = 0.0f, dot = 0.0f;
    // Match MLX's row reduction: adjacent reads per thread, then SIMD sums.
    constexpr uint reads = D <= 64 ? D : 4;
    for (uint t = 0; t < (D + TG * reads - 1) / (TG * reads); ++t) {
        for (uint j = 0; j < reads; ++j) {
            const uint d = tid * reads + TG * reads * t + j;
            if (d < D) {
                const float x = float(h[hrow + d]);
                const float key = float(kv[krow + d]);
                const float w = q_weight[c * D + d] * k_weight[c * D + d];
                sx += x * x;
                sk += key * key;
                dot += x * w * key;
            }
        }
    }
    sx = simd_sum(sx);
    sk = simd_sum(sk);
    dot = simd_sum(dot);
    if (lane == 0) {
        part[simd][0] = sx;
        part[simd][1] = sk;
        part[simd][2] = dot;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0) {
        float xsum = lane < TG / 32 ? part[lane][0] : 0.0f;
        float ksum = lane < TG / 32 ? part[lane][1] : 0.0f;
        float dsum = lane < TG / 32 ? part[lane][2] : 0.0f;
        xsum = simd_sum(xsum);
        ksum = simd_sum(ksum);
        dsum = simd_sum(dsum);
        if (lane == 0) {
            part[0][0] = xsum;
            part[0][1] = ksum;
            part[0][2] = dsum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float gatev[1];
    if (tid == 0) {
        const float rx = v41_engram_rsqrt_mean(part[0][0], float(D), eps[0]);
        const float rk = v41_engram_rsqrt_mean(part[0][1], float(D), eps[0]);
        const float scaled = part[0][2] * (rx * rk) * scalep[0];
        float gate = 0.0f;
        if (!(MASK && mask[token] != 0)) {
            if (scaled == 0.0f) {
                gate = v41_engram_sigmoid(0.001f);
            } else {
                const float sign = scaled > 0.0f ? 1.0f : -1.0f;
                gate = v41_engram_sigmoid(
                    sign * metal::precise::sqrt(max(abs(scaled), 1e-6f))
                );
            }
        }
        gatev[0] = gate;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float gate = gatev[0];
    for (uint t = 0; t < (D + TG - 1) / TG; ++t) {
        const uint d = tid + TG * t;
        if (d < D) {
            out[hrow + d] = T(float(h[hrow + d]) + gate * float(kv[vrow + d]));
        }
    }
"""


@cache
def _gate_kernel():
    return mx.fast.metal_kernel(
        name="deepseek_v41_engram_gate",
        input_names=[
            "h", "kv", "inv", "q_weight", "k_weight", "mask", "eps", "scalep"
        ],
        output_names=["out"],
        source=_GATE_SOURCE,
        header=_GATE_HEADER,
    )


@cache
def _gate_scale(dim):
    value = mx.array([dim**-0.5], mx.float32)
    mx.eval(value)
    return value


@cache
def _gate_eps(eps):
    value = mx.array([eps], mx.float32)
    mx.eval(value)
    return value


def engram_gate_reference(h, kv, q_weight, k_weight, eps, dim, hc, image_mask=None):
    """The official per-row gate arithmetic, used for decode and fallback."""
    key, value = mx.split(kv, [dim * hc], axis=-1)
    key = key.astype(mx.float32).reshape(*h.shape)
    x = h.astype(mx.float32)
    inv = mx.rsqrt(mx.mean(x * x, -1) + eps) * mx.rsqrt(
        mx.mean(key * key, -1) + eps
    )
    weight = q_weight.astype(mx.float32) * k_weight.astype(mx.float32)
    dot = mx.sum(x * weight * key, -1) * inv * dim**-0.5
    gate = mx.sigmoid(mx.sign(dot) * mx.sqrt(mx.maximum(mx.abs(dot), 1e-6)))
    gate = mx.where(dot == 0, mx.sigmoid(mx.array(0.001)), gate)
    if image_mask is not None:
        gate = mx.where(image_mask[..., None], 0, gate)
    return (x + gate[..., None] * value.astype(mx.float32)[..., None, :]).astype(
        h.dtype
    )


def fused_engram_gate(
    h, kv, inverse, q_weight, k_weight, eps, dim, hc, image_mask=None
):
    """Gate and value add read only the unique tuple's kv row and BF16 h.

    ``kv`` is [unique tuples, (hc + 1) * dim], ``inverse`` maps every token to
    its unique tuple. Avoids materialising key/value in FP32 and gathering the
    full kv matrix; FP32 reductions and arithmetic match the reference.
    """
    if (
        mx.default_device() != mx.gpu
        or h.dtype not in (mx.bfloat16, mx.float16, mx.float32)
        or h.shape[-1] != dim
        or h.shape[-2] != hc
        or kv.shape[-1] != (hc + 1) * dim
        or int(inverse.size) != int(h.shape[0] * h.shape[1])
    ):
        return None
    if image_mask is not None and tuple(image_mask.shape) != tuple(h.shape[:-2]):
        return None
    mask = (
        mx.array(image_mask, dtype=mx.uint8)
        if image_mask is not None
        else mx.zeros((1, 1), mx.uint8)
    )
    rows = h.size // dim
    # Keep in sync with MLX 0.32.4's threadgroup_size_from_row_size.
    threads = 32 if dim <= 512 else 128 if dim <= 1024 else min(1024, (dim + 127) // 128 * 32)
    return _gate_kernel()(
        inputs=[
            h,
            kv,
            mx.array(inverse, dtype=mx.int32),
            q_weight.astype(mx.float32),
            k_weight.astype(mx.float32),
            mask,
            _gate_eps(eps),
            _gate_scale(dim),
        ],
        template=[
            ("T", h.dtype),
            ("HC", hc),
            ("D", dim),
            ("TG", threads),
            ("MASK", image_mask is not None),
        ],
        grid=(rows * threads, 1, 1),
        threadgroup=(threads, 1, 1),
        output_shapes=[h.shape],
        output_dtypes=[h.dtype],
    )[0]


class Engram(nn.Module):
    def __init__(self, config, table_index):
        super().__init__()
        self.embed = nn.Embedding(
            config.engram_num_embeddings[table_index], config.engram_head_dim
        )
        self.wkv = nn.Linear(
            (config.engram_max_ngram_size - 1)
            * config.engram_n_heads
            * config.engram_head_dim,
            config.dim * (config.hc_mult + 1),
            bias=False,
        )
        self.q_weight = mx.ones((config.hc_mult, config.dim))
        self.k_weight = mx.ones((config.hc_mult, config.dim))
        self._config = config

    def __call__(self, h, ids, image_mask=None):
        c = self._config
        ids_np = np.asarray(ids, dtype=np.int64)
        ids_host, inverse = dedupe_tuple_ids(ids_np)
        rows = self.embed(mx.array(ids_host)).flatten(-2)
        count = math.prod(ids_np.shape[:-1]) if ids_np.ndim >= 2 else 1
        if 1 < count <= _ROWWISE_ROWS:
            # Decode-sized inputs (one token for each of B requests, or an MTP verify block
            # of one request): project each row on its own, exactly as a single-token
            # decode does. A multi-row quantized_matmul runs a different MLX kernel whose
            # fp32 sums round differently (the reference leaves GEMM order free, but a
            # token's output would then depend on its batch or on speculation).
            kv = _project_each(self.wkv, rows.reshape(count, -1))
            if kv is None:
                flat = rows.reshape(count, 1, -1)
                kv = mx.concatenate([self.wkv(flat[r : r + 1]) for r in range(count)])
            kv = kv.reshape(*rows.shape[:2], -1)
        else:
            kv = self.wkv(rows)
        if count <= _ROWWISE_ROWS or ids_np.ndim != 3:
            return engram_gate_reference(
                h, kv, self.q_weight, self.k_weight, c.norm_eps, c.dim, c.hc_mult,
                image_mask,
            )
        token_inverse = (
            inverse
            if inverse is not None
            else np.arange(count, dtype=np.int64).reshape(ids_host.shape[:-1])
        )
        fused = fused_engram_gate(
            h, kv, token_inverse, self.q_weight, self.k_weight, c.norm_eps,
            c.dim, c.hc_mult, image_mask,
        )
        if fused is not None:
            return fused
        if inverse is not None:
            kv = kv[mx.array(inverse, dtype=mx.int32)]
        return engram_gate_reference(
            h, kv, self.q_weight, self.k_weight, c.norm_eps, c.dim, c.hc_mult,
            image_mask,
        )
