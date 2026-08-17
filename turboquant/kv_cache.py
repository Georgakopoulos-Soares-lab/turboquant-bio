"""kv_cache.py — TurboQuant-style KV-cache quantization for Evo2 / StripedHyena.

Evo2 is a StripedHyena2 model: out of N layers, only a handful are *attention*
layers (40B: 8 of 50; 7B: 5 of 32).  The KV cache exists ONLY in those attention
layers; the Hyena layers keep a small fixed recurrent state instead.  This module
therefore targets exactly those attention layers.

Two capabilities are provided:

1. Simulated quantization (accuracy measurement)
   ------------------------------------------------
   `install_kv_quant(model, bits=4, ...)` monkeypatches every
   `FlashSelfAttention.forward` inside the attention layers so that the K and V
   tensors are quantize→dequantized ("fake quant") before the attention scores
   are computed.  This measures the accuracy impact of storing K/V in `bits`-bit
   precision, and works in an ordinary forward pass (the path used by
   perplexity evaluation, which otherwise never touches the runtime KV cache).

   Quantization scheme (KIVI / KVQuant best practice, NOT a blind reuse of the
   weight quantizer, because K/V activations have outlier channels):
       K  →  per-channel  (each head_dim coordinate shares a scale across tokens)
       V  →  per-token    (each token's vector quantized independently)
   Asymmetric min-max, `bits` levels.  An optional `residual` window keeps the
   most recent tokens in full precision (helps a lot at very low bit-widths).

2. Analytic memory accounting
   ---------------------------
   `kv_cache_bytes(...)` / `print_kv_memory_report(...)` compute the bf16 vs
   quantized KV-cache footprint for a given (model, context length, batch).

The patch is installed at runtime via monkeypatching so the vortex site-package
is never modified.  Call `remove_kv_quant(model)` to restore the originals.
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch


GB = 1 << 30
_ATTN_LAYER_IDXS = {
    "evo2_40b": [3, 10, 17, 24, 31, 35, 42, 49],
    "evo2_40b_base": [3, 10, 17, 24, 31, 35, 42, 49],
    "evo2_7b": [3, 10, 17, 24, 31],
    "evo2_7b_base": [3, 10, 17, 24, 31],
}


# ---------------------------------------------------------------------------
# Quantization primitives (fake quant: returns same dtype/shape)
# ---------------------------------------------------------------------------

def _asym_fake_quant(x: torch.Tensor, bits: int, reduce_dim: int) -> torch.Tensor:
    """Asymmetric min-max quantize→dequantize along `reduce_dim` groups.

    The scale/zero-point are computed by reducing over `reduce_dim`; every other
    axis keeps its own scale.  Returns a tensor of the same shape and dtype as
    `x`, holding the de-quantized (lossy) values.
    """
    if bits >= 16:
        return x
    qmax = (1 << bits) - 1
    x_f = x.float()
    x_min = x_f.amin(dim=reduce_dim, keepdim=True)
    x_max = x_f.amax(dim=reduce_dim, keepdim=True)
    scale = (x_max - x_min).clamp_(min=1e-8) / qmax
    q = torch.clamp(torch.round((x_f - x_min) / scale), 0, qmax)
    deq = q * scale + x_min
    return deq.to(x.dtype)


def fake_quant_k(k: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-channel quant for K.  k: (B, S, H, D) → reduce over S (token axis)."""
    # Each (head, head_dim) channel gets its own scale across the sequence.
    return _asym_fake_quant(k, bits, reduce_dim=1)


def fake_quant_v(v: torch.Tensor, bits: int) -> torch.Tensor:
    """Per-token quant for V.  v: (B, S, H, D) → reduce over D (head_dim axis)."""
    # Each (token, head) vector gets its own scale.
    return _asym_fake_quant(v, bits, reduce_dim=3)


def _quant_kv_packed(
    qkv: torch.Tensor, bits: int, residual: int
) -> torch.Tensor:
    """Fake-quantize K and V inside a packed qkv tensor of shape (B, S, 3, H, D).

    The most recent `residual` tokens are left in full precision.
    """
    if qkv.dim() != 5 or qkv.shape[2] != 3:
        # Unpadded/varlen layout — skip (only used during cu_seqlens training).
        return qkv
    b, s, _, h, d = qkv.shape
    q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

    if residual > 0 and s > residual:
        k_old, k_new = k[:, :-residual], k[:, -residual:]
        v_old, v_new = v[:, :-residual], v[:, -residual:]
        k_q = torch.cat([fake_quant_k(k_old, bits), k_new], dim=1)
        v_q = torch.cat([fake_quant_v(v_old, bits), v_new], dim=1)
    else:
        k_q = fake_quant_k(k, bits)
        v_q = fake_quant_v(v, bits)

    return torch.stack([q, k_q, v_q], dim=2)


# ---------------------------------------------------------------------------
# Monkeypatch installer
# ---------------------------------------------------------------------------

def install_kv_quant(
    model: torch.nn.Module,
    bits: int = 4,
    residual: int = 0,
    verbose: bool = True,
) -> int:
    """Patch every FlashSelfAttention.forward to fake-quantize K and V.

    Returns the number of attention modules patched.  Idempotent: calling again
    re-patches with the new (bits, residual).  Use `remove_kv_quant` to restore.
    """
    from vortex.model.attention import FlashSelfAttention

    patched = 0
    for module in model.modules():
        if not isinstance(module, FlashSelfAttention):
            continue
        # Preserve the genuine original once.
        if not hasattr(module, "_orig_forward_kvq"):
            module._orig_forward_kvq = module.forward

        orig = module._orig_forward_kvq

        def make_patched(orig_fn):
            def patched_forward(qkv, causal=None, cu_seqlens=None, max_seqlen=None):
                if cu_seqlens is None:
                    qkv = _quant_kv_packed(qkv, bits, residual)
                return orig_fn(qkv, causal=causal, cu_seqlens=cu_seqlens,
                               max_seqlen=max_seqlen)
            return patched_forward

        module.forward = make_patched(orig)
        patched += 1

    if verbose:
        print(f"  [kv-quant] patched {patched} attention modules "
              f"(bits={bits}, residual={residual})")
    return patched


def remove_kv_quant(model: torch.nn.Module) -> int:
    """Restore the original FlashSelfAttention.forward on every patched module."""
    from vortex.model.attention import FlashSelfAttention

    restored = 0
    for module in model.modules():
        if isinstance(module, FlashSelfAttention) and hasattr(module, "_orig_forward_kvq"):
            module.forward = module._orig_forward_kvq
            del module._orig_forward_kvq
            restored += 1
    return restored


@contextlib.contextmanager
def kv_quant(model: torch.nn.Module, bits: int = 4, residual: int = 0,
             verbose: bool = False):
    """Context manager: temporarily fake-quantize K/V inside `with` block."""
    install_kv_quant(model, bits=bits, residual=residual, verbose=verbose)
    try:
        yield model
    finally:
        remove_kv_quant(model)


# ---------------------------------------------------------------------------
# Analytic memory accounting
# ---------------------------------------------------------------------------

def kv_cache_bytes(
    n_attn_layers: int,
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    bits: int = 16,
) -> int:
    """Bytes for the KV cache.

    bf16 baseline (bits=16): 2 (k+v) * heads * head_dim * 2 bytes per token.
    Quantized: `bits`/8 bytes per code + per-(token,head) fp16 scale & zero.
    """
    per_token_per_layer_codes = 2 * num_heads * head_dim  # K + V elements
    if bits >= 16:
        store = per_token_per_layer_codes * 2  # bf16
        meta = 0
    else:
        store = per_token_per_layer_codes * bits / 8.0
        # one fp16 scale + one fp16 zero per (head) for K (per-channel reduces
        # over tokens so amortized ~per head_dim) and per (token,head) for V.
        # Use a conservative per-(token,head) estimate for both: 2 heads-vectors
        # * 2 (scale+zero) * 2 bytes.
        meta = 2 * num_heads * 2 * 2
    return int((store + meta) * seq_len * n_attn_layers * batch)


def print_kv_memory_report(
    model_name: str,
    seq_lens=(8192, 40000, 131072, 1048576),
    batch: int = 1,
    bits: int = 4,
) -> None:
    cfg = {
        "evo2_40b": dict(n=8, heads=64, d=128),
        "evo2_7b":  dict(n=5, heads=32, d=128),
    }
    key = "evo2_40b" if "40b" in model_name else "evo2_7b"
    c = cfg[key]
    print(f"\n  KV-cache memory — {model_name} "
          f"({c['n']} attn layers, {c['heads']} heads, head_dim {c['d']})")
    print(f"  {'ctx_len':>10} | {'bf16':>10} | {f'{bits}-bit':>10} | {'freed':>10} | ratio")
    print("  " + "-" * 60)
    for s in seq_lens:
        b16 = kv_cache_bytes(c["n"], c["heads"], c["d"], s, batch, 16)
        bq  = kv_cache_bytes(c["n"], c["heads"], c["d"], s, batch, bits)
        print(f"  {s:>10} | {b16/GB:>8.2f}GB | {bq/GB:>8.2f}GB | "
              f"{(b16-bq)/GB:>8.2f}GB | {b16/bq:>4.2f}x")


# ===========================================================================
# REAL int4/int2/int8 KV storage (actual byte-level packing → measurable VRAM)
# ===========================================================================
#
# Unlike the fake-quant above (which dequantizes immediately and therefore never
# saves a single byte), the classes below physically store the K/V cache as
# packed `bits`-bit integer codes (uint8 nibble/crumb packing).  The persistent
# footprint of the cache is genuinely reduced; the full-precision bf16 tensor is
# only materialized *transiently*, one attention layer at a time, during the
# attention call, then freed.  This is what produces a real, measurable drop in
# `torch.cuda.max_memory_allocated` during live generation.
#
# Scheme (matches the validated fake-quant):
#   K -> per-channel  (scale/zero reduced over the token axis)
#   V -> per-token    (scale/zero reduced over the head_dim axis)
# Asymmetric min-max.  The prefill chunk is quantized once; subsequent decode
# tokens accumulate in a small fp16 residual buffer (KIVI-style), so the dominant
# long-context portion is stored at `bits` precision.


def _pack_nbit(codes: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack a uint8 tensor of `bits`-bit codes into a dense uint8 byte string.

    codes: arbitrary-shape uint8 tensor with values in [0, 2**bits - 1].
    Returns a 1-D uint8 tensor of length ceil(numel * bits / 8).
    """
    if bits == 8:
        return codes.reshape(-1).contiguous()
    per_byte = 8 // bits  # 2 for 4-bit, 4 for 2-bit
    flat = codes.reshape(-1)
    n = flat.numel()
    pad = (-n) % per_byte
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    flat = flat.view(-1, per_byte)
    out = torch.zeros(flat.shape[0], dtype=torch.uint8, device=codes.device)
    for i in range(per_byte):
        out |= (flat[:, i] << (bits * i)).to(torch.uint8)
    return out


def _unpack_nbit(packed: torch.Tensor, bits: int, numel: int) -> torch.Tensor:
    """Inverse of `_pack_nbit`.  Returns a 1-D uint8 tensor of length `numel`."""
    if bits == 8:
        return packed[:numel]
    per_byte = 8 // bits
    mask = (1 << bits) - 1
    cols = []
    for i in range(per_byte):
        cols.append(((packed >> (bits * i)) & mask).to(torch.uint8))
    out = torch.stack(cols, dim=1).reshape(-1)
    return out[:numel]


class QuantizedKVCache:
    """Per-layer packed int-N store for one attention layer's K/V cache."""

    __slots__ = ("bits", "device", "segments", "residual", "_contig_view")

    def __init__(self, bits: int = 4):
        self.bits = bits
        self.reset()

    def reset(self):
        # segments: list of independently-quantized packed blocks (dicts).
        # residual: list of fp16 (B, s, 2, H, D) most-recent (decode) chunks.
        self.segments = []
        self.residual = []
        self.device = None
        self._contig_view = None   # cached contiguous view for the batched prefill kernel

    # -- quantize one (B, s, 2, H, D) block into a packed segment ----------
    def _quantize_block(self, kv: torch.Tensor) -> dict:
        """Quantize+pack a block with its OWN per-channel-K / per-token-V scales.

        The largest transient is one fp32 copy of THIS block, so flushing only
        ever costs O(block) memory — never O(context).  That is what keeps int4
        prefill memory bounded at long context.

        Packed-byte layout is (B, S, H, D) (token-major). A Phase-3 tuning
        attempt reordered this to (B, H, S, D) so a fixed head's tokens were
        contiguous, motivated by Nsight Compute measuring 31.84 sectors/request
        (~8x worse than the ~4 of a fully-coalesced access). REVERTED: sector
        efficiency was measured unchanged (31.95, still ~32) after the
        reorder — Triton's codegen for this kernel's (BLOCK_N, D) load doesn't
        tile threads along the token axis the way the layout change assumed, so
        it bought no coalescing — and it correlated with a real-model gate
        regression on the 40B (two attention layers moved from ~3e-8 to
        ~1-2e-4 vs the fp32 reference; still small in absolute terms but a
        genuine regression from the stated baseline, and not worth carrying
        for zero measured benefit). See TUNING.md.
        """
        k = kv[:, :, 0]  # (B, s, H, D)
        v = kv[:, :, 1]
        qmax = (1 << self.bits) - 1

        # K per-channel: reduce over the token axis (dim=1)
        kf = k.float()
        kmin = kf.amin(dim=1, keepdim=True)
        kmax = kf.amax(dim=1, keepdim=True)
        kscale = (kmax - kmin).clamp_(min=1e-8) / qmax
        kcodes = torch.clamp(torch.round((kf - kmin) / kscale), 0, qmax).to(torch.uint8)
        seg = {
            "k_shape": tuple(k.shape),
            "k_packed": _pack_nbit(kcodes, self.bits),
            "k_scale": kscale.to(torch.float16),
            "k_zero": kmin.to(torch.float16),
        }
        del kf, kmin, kmax, kscale, kcodes

        # V per-token: reduce over the head_dim axis (dim=3)
        vf = v.float()
        vmin = vf.amin(dim=3, keepdim=True)
        vmax = vf.amax(dim=3, keepdim=True)
        vscale = (vmax - vmin).clamp_(min=1e-8) / qmax
        vcodes = torch.clamp(torch.round((vf - vmin) / vscale), 0, qmax).to(torch.uint8)
        seg["v_shape"] = tuple(v.shape)
        seg["v_packed"] = _pack_nbit(vcodes, self.bits)
        seg["v_scale"] = vscale.to(torch.float16)
        seg["v_zero"] = vmin.to(torch.float16)
        del vf, vmin, vmax, vscale, vcodes
        return seg

    def set_prefill(self, kv: torch.Tensor):
        """kv: (B, S, 2, H, D) post-rotary.  Reset and quantize as segment 0."""
        self.reset()
        self.device = kv.device
        self.segments = [self._quantize_block(kv)]

    def append_decode(self, kv: torch.Tensor):
        """kv: (B, s, 2, H, D) — keep recent tokens in fp16 residual."""
        if not self.segments and not self.residual:
            # No prefill happened yet (generation started at len 0).
            self.set_prefill(kv)
            return
        if self.device is None:
            self.device = kv.device
        self.residual.append(kv.to(torch.float16))

    def flush_residual(self):
        """Quantize the accumulated residual into a NEW segment.

        Only the residual block is dequantized/re-quantized, so the transient is
        O(residual) (= one prefill chunk) — never the whole context.  This is the
        key difference from a single-segment store, which re-quantized the entire
        cache on every flush and thus blew up int4 prefill memory at long context.
        """
        if not self.residual:
            return
        kv_block = torch.cat([r.to(self.device) for r in self.residual], dim=1)
        self.segments.append(self._quantize_block(kv_block))
        self.residual = []
        del kv_block

    # -- token-block dequant (streaming attention; never materialize full S) --
    def _seg_slice_codes(self, packed, k_shape, s0: int, s1: int) -> torch.Tensor:
        """Unpack codes for token range [s0, s1) of ONE segment → (B, blk, H, D).

        Relies on H*D being divisible by the pack factor (8 // bits), which holds
        for Evo2 (H*D = 8192 for 40B, 4096 for 7B), so every token boundary is a
        byte boundary and we can slice the packed buffer directly without
        unpacking the whole segment.
        """
        B, S, H, D = k_shape
        HD = H * D
        per_byte = 1 if self.bits == 8 else 8 // self.bits
        blk = s1 - s0
        out = []
        for b in range(B):
            start = (b * S + s0) * HD
            stop = (b * S + s1) * HD
            sub = packed[start // per_byte: stop // per_byte]
            codes = _unpack_nbit(sub, self.bits, blk * HD).view(blk, H, D)
            out.append(codes)
        return torch.stack(out, dim=0)  # (B, blk, H, D)

    def _seg_dequant_k(self, seg, s0: int, s1: int, dtype) -> torch.Tensor:
        codes = self._seg_slice_codes(seg["k_packed"], seg["k_shape"], s0, s1).to(dtype)
        # K is per-channel: scale/zero shape (B, 1, H, D) broadcast over tokens.
        return codes * seg["k_scale"].to(dtype) + seg["k_zero"].to(dtype)

    def _seg_dequant_v(self, seg, s0: int, s1: int, dtype) -> torch.Tensor:
        codes = self._seg_slice_codes(seg["v_packed"], seg["v_shape"], s0, s1).to(dtype)
        # V is per-token: scale/zero shape (B, S, H, 1) → slice the token axis.
        return codes * seg["v_scale"][:, s0:s1].to(dtype) + seg["v_zero"][:, s0:s1].to(dtype)

    def iter_key_blocks(self, key_block: int, dtype):
        """Yield (k_blk, v_blk, abs_s0) over every segment then the residual,
        tiled by `key_block`.  Only one block is materialized at a time, so the
        streaming attention that consumes this is O(block) in peak memory and
        independent of how the cache was segmented.
        """
        s0 = 0
        for seg in self.segments:
            S = int(seg["k_shape"][1])
            for a in range(0, S, key_block):
                b = min(a + key_block, S)
                yield (self._seg_dequant_k(seg, a, b, dtype),
                       self._seg_dequant_v(seg, a, b, dtype),
                       s0 + a)
            s0 += S
        for r in self.residual:
            S = int(r.shape[1])
            for a in range(0, S, key_block):
                b = min(a + key_block, S)
                yield (r[:, a:b, 0].to(dtype), r[:, a:b, 1].to(dtype), s0 + a)
            s0 += S

    @property
    def packed_len(self) -> int:
        return sum(int(seg["k_shape"][1]) for seg in self.segments)

    @property
    def total_len(self) -> int:
        return self.packed_len + sum(int(r.shape[1]) for r in self.residual)

    @property
    def num_heads_kv(self) -> int:
        if self.segments:
            return int(self.segments[0]["k_shape"][2])
        if self.residual:
            return int(self.residual[0].shape[3])
        return 0

    def dequantize(self, dtype=torch.bfloat16, device=None) -> torch.Tensor:
        """Materialize the full (B, S_total, 2, H, D) cache (transient).

        Used only by the non-streaming `install_real_kv_quant` path and by
        correctness checks; the streaming path never calls this.
        """
        parts = []
        for seg in self.segments:
            S = int(seg["k_shape"][1])
            k = self._seg_dequant_k(seg, 0, S, dtype)
            v = self._seg_dequant_v(seg, 0, S, dtype)
            parts.append(torch.stack([k, v], dim=2))  # (B, S, 2, H, D)
        for r in self.residual:
            parts.append(r.to(dtype))
        return torch.cat(parts, dim=1)

    def nbytes(self) -> int:
        n = 0
        for seg in self.segments:
            for key in ("k_packed", "k_scale", "k_zero",
                        "v_packed", "v_scale", "v_zero"):
                t = seg[key]
                n += t.numel() * t.element_size()
        for r in self.residual:
            n += r.numel() * r.element_size()
        return n


# ---------------------------------------------------------------------------
# Real-storage installer: replaces bf16 cache with packed int-N store
# ---------------------------------------------------------------------------

def install_real_kv_quant(
    model: torch.nn.Module,
    bits: int = 4,
    model_name: str = "evo2_7b",
    verbose: bool = True,
) -> int:
    """Patch attention MHA modules to store the KV cache as packed int-N codes.

    This produces a *real* reduction in the persistent KV-cache footprint during
    live generation (measurable via torch.cuda.max_memory_allocated), at the cost
    of a transient bf16 materialization of one layer's cache during its attention.

    Returns the number of attention modules patched.
    """
    import types
    from vortex.model.attention import MHA

    key = "evo2_40b" if "40b" in model_name else "evo2_7b"
    attn_idxs = set(_ATTN_LAYER_IDXS[key])

    patched = 0
    for module in model.modules():
        if not isinstance(module, MHA):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or layer_idx not in attn_idxs:
            continue

        # Force the non-fused python-rotary path so rotary is applied before the
        # cache write and our override of _update_kvcache_attention is used.
        if not hasattr(module, "_orig_use_flash_attn"):
            module._orig_use_flash_attn = module.use_flash_attn
        module.use_flash_attn = False

        module._qkv_cache = QuantizedKVCache(bits)
        module._kvq_bits = bits

        if not hasattr(module, "_orig_update_kvcache_attention"):
            module._orig_update_kvcache_attention = module._update_kvcache_attention

        def _quant_update_kvcache_attention(self, q, kv, inference_params):
            so = inference_params.seqlen_offset
            if so == 0:
                self._qkv_cache.set_prefill(kv)
            else:
                self._qkv_cache.append_decode(kv)
            kv_full = self._qkv_cache.dequantize(dtype=q.dtype, device=q.device)
            return self.inner_cross_attn(q, kv_full)

        module._update_kvcache_attention = types.MethodType(
            _quant_update_kvcache_attention, module)
        patched += 1

    if verbose:
        print(f"  [real-kv-quant] patched {patched} attention MHA modules "
              f"(bits={bits}, model={model_name})")
    return patched


def remove_real_kv_quant(model: torch.nn.Module) -> int:
    """Restore the original bf16 KV-cache path on every patched MHA module."""
    from vortex.model.attention import MHA

    restored = 0
    for module in model.modules():
        if not isinstance(module, MHA):
            continue
        if hasattr(module, "_orig_update_kvcache_attention"):
            module._update_kvcache_attention = module._orig_update_kvcache_attention
            del module._orig_update_kvcache_attention
            restored += 1
        if hasattr(module, "_orig_use_flash_attn"):
            module.use_flash_attn = module._orig_use_flash_attn
            del module._orig_use_flash_attn
        if hasattr(module, "_qkv_cache"):
            del module._qkv_cache
    return restored


# ===========================================================================
# Streaming (tiled) dequantized attention — bounds PEAK decode memory
# ===========================================================================
#
# The real-storage installer above shrinks the *resident* KV footprint, but its
# attention step rebuilds the entire (B, S, 2, H, D) bf16 cache transiently
# before calling `inner_cross_attn`.  At long context that transient blob (~2.6
# GB at 32k on 40B) coexists with the packed store and pushes *peak* memory
# above the bf16 baseline (and tips int4 into OOM at 32k).
#
# The streaming path below never materializes the full-sequence bf16 K/V.  It
# tiles over key positions: for each block it dequantizes only that block from
# the packed int store, computes partial attention scores against q, and folds
# them into a running (max, sum, output) accumulator using a numerically stable
# online softmax (FlashAttention-style).  The block is freed before the next is
# unpacked, so the largest live bf16 tensor is a single key block, not the whole
# sequence.  The KV dequant is a plain scale+zero (no rotation), so per-block
# decode is exact and the result matches the full-dequant path to float rounding.


@torch.inference_mode()
def streaming_kv_attention(
    cache: "QuantizedKVCache",
    q: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    key_block: int = 2048,
) -> torch.Tensor:
    """Online-softmax attention over a packed int-N KV cache, tiled over keys.

    q:    (B, Sq, Hq, D)  query (decode: Sq == 1; prefill/extend: Sq == block).
    Returns context of shape (B, Sq, Hq, D), matching vortex CrossAttention.

    Only one key block is dequantized to bf16/fp16 at a time, so peak memory is
    O(block) rather than O(sequence).
    """
    B, Sq, Hq, D = q.shape
    device, dtype = q.device, q.dtype
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)

    Sk = cache.total_len
    Hkv = cache.num_heads_kv
    g = Hq // max(Hkv, 1)  # GQA / MQA group size

    # fp32 running accumulators in (B, Hq, Sq, ...) layout for the einsums below.
    acc = torch.zeros(B, Hq, Sq, D, device=device, dtype=torch.float32)
    m = torch.full((B, Hq, Sq), float("-inf"), device=device, dtype=torch.float32)
    l = torch.zeros(B, Hq, Sq, device=device, dtype=torch.float32)
    qf = (q.to(torch.float32) * scale)  # (B, Sq, Hq, D)

    def _consume(k_blk: torch.Tensor, v_blk: torch.Tensor, s0: int):
        nonlocal acc, m, l
        blk = k_blk.shape[1]
        if g > 1:  # expand KV heads to match query heads (GQA/MQA)
            k_blk = k_blk.repeat_interleave(g, dim=2)
            v_blk = v_blk.repeat_interleave(g, dim=2)
        kf = k_blk.to(torch.float32)
        vf = v_blk.to(torch.float32)
        scores = torch.einsum("bthd,bshd->bhts", qf, kf)  # (B, Hq, Sq, blk)
        if causal:
            row = torch.arange(Sq, device=device).view(Sq, 1)
            col = torch.arange(s0, s0 + blk, device=device).view(1, blk)
            # vortex rule: key masked if col_idx > row_idx + Sk - Sq
            mask = col > (row + (Sk - Sq))
            if mask.any():
                scores = scores.masked_fill(mask.view(1, 1, Sq, blk), float("-inf"))
        blk_max = scores.amax(dim=-1)              # (B, Hq, Sq)
        m_new = torch.maximum(m, blk_max)
        p = torch.exp(scores - m_new.unsqueeze(-1))  # masked → exp(-inf)=0
        alpha = torch.exp(m - m_new)                 # finite after first block
        l = l * alpha + p.sum(dim=-1)
        acc = acc * alpha.unsqueeze(-1) + torch.einsum("bhts,bshd->bhtd", p, vf)
        m = m_new

    # every segment then residual, one key block at a time (O(block) peak)
    for k_blk, v_blk, s0 in cache.iter_key_blocks(key_block, dtype):
        _consume(k_blk, v_blk, s0)
        del k_blk, v_blk

    out = acc / l.clamp_min(1e-20).unsqueeze(-1)     # (B, Hq, Sq, D)
    return out.permute(0, 2, 1, 3).contiguous().to(dtype)  # (B, Sq, Hq, D)


def install_streaming_kv_quant(
    model: torch.nn.Module,
    bits: int = 4,
    model_name: str = "evo2_7b",
    key_block: int = 2048,
    verbose: bool = True,
) -> int:
    """Like `install_real_kv_quant`, but consumes the cache with streaming
    (tiled online-softmax) attention so the full-sequence bf16 K/V is never
    materialized — bounding *peak* decode memory.

    Storage format is identical (`QuantizedKVCache`); only the attention step
    differs, so the two installers are directly comparable.  Mirrors
    `install_real_kv_quant`'s monkeypatch so the vendored vortex package is
    never edited.
    """
    import types
    from vortex.model.attention import MHA

    key = "evo2_40b" if "40b" in model_name else "evo2_7b"
    attn_idxs = set(_ATTN_LAYER_IDXS[key])

    patched = 0
    for module in model.modules():
        if not isinstance(module, MHA):
            continue
        layer_idx = getattr(module, "layer_idx", None)
        if layer_idx is None or layer_idx not in attn_idxs:
            continue

        if not hasattr(module, "_orig_use_flash_attn"):
            module._orig_use_flash_attn = module.use_flash_attn
        module.use_flash_attn = False

        module._qkv_cache = QuantizedKVCache(bits)
        module._kvq_bits = bits
        module._kvq_key_block = key_block

        if not hasattr(module, "_orig_update_kvcache_attention"):
            module._orig_update_kvcache_attention = module._update_kvcache_attention

        def _stream_update_kvcache_attention(self, q, kv, inference_params):
            so = inference_params.seqlen_offset
            if so == 0:
                self._qkv_cache.set_prefill(kv)
            else:
                self._qkv_cache.append_decode(kv)
            return streaming_kv_attention(
                self._qkv_cache, q,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                key_block=self._kvq_key_block,
            )

        module._update_kvcache_attention = types.MethodType(
            _stream_update_kvcache_attention, module)
        patched += 1

    if verbose:
        print(f"  [stream-kv-quant] patched {patched} attention MHA modules "
              f"(bits={bits}, key_block={key_block}, model={model_name})")
    return patched


def remove_streaming_kv_quant(model: torch.nn.Module) -> int:
    """Restore the original bf16 KV-cache path (same as remove_real_kv_quant)."""
    return remove_real_kv_quant(model)

