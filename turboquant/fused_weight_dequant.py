"""fused_weight_dequant.py — fused forward for TurboQuant-quantized Linear layers.

Motivation
----------
QuantizedLinearGpu.forward currently *materializes* the full dequantized weight every
call (centroid lookup -> renormalize -> inverse-rotate every block by R -> rescale ->
write W to HBM) and then does F.linear. The inverse rotation and the HBM round-trip of
the full weight are the dominant cost of the ~10x compressed-inference slowdown.

The key identity (proved in scratchpad/prove_fusion.py, max rel err ~8e-7 vs the exact
dequant+matmul):

    y = x @ W_hat^T,   where for output row o, input block b:
        W_hat[o, bD:bD+D] = norm[o,b] * ( normalize(centroids[idx[o,b]]) @ R )

    can be rewritten WITHOUT ever forming W_hat as:
        xr[m,b] = R @ x_block[m,b]                         # rotate INPUT blocks by R
        y[m,o]  = sum_b norm[o,b] * < normalize(centroids[idx[o,b]]) , xr[m,b] >

Because R is shared across all output rows, the expensive D×D rotation moves off the
weights (out_f · n_b blocks, HBM-bound) onto the activations (M · n_b blocks, tiny and
done once). What remains per (output row, block) is a codebook gather + per-block
normalize + a length-D dot — exactly the shape that fuses into a GEMM inner loop, like
a standard INT4 kernel but with a codebook lookup instead of affine dequant.

This module provides:
  * fused_linear_torch(...)  — pure-torch reference implementing the identity. Verified
    on CPU against the exact dequant path (test_fused_weight_dequant.py). Correct on any
    device; it still materializes the codebook weights, so it is the CORRECTNESS oracle,
    not the speed win.
  * fused_linear_triton(...) — the Triton kernel that realizes the speed win by never
    materializing the codebook weights. MUST pass the GPU parity test before use
    (guarded by _TRITON_VERIFIED / an explicit opt-in), since untested Triton is a
    correctness hazard.
  * install_fused_weight_forward(model) — swap QuantizedLinearGpu.forward to the fused
    path (torch by default; triton once verified).

Nibble unpacking / padding / R / centroids all follow benchmark_evo2_40b_weights.py
exactly so this is a drop-in for QuantizedLinearGpu's existing buffers.
"""
from __future__ import annotations

import numpy as np
import torch

# --------------------------------------------------------------------------- #
#  Shared helpers (kept byte-compatible with benchmark_evo2_40b_weights.py)    #
# --------------------------------------------------------------------------- #

_RC_CACHE: dict = {}


def get_R_centroids(block_size: int, bit_width: int, device, seed: int = 42):
    """(R, centroids) as float32 on `device`, matching _get_quant_tensors exactly."""
    key = (block_size, bit_width, str(device), seed)
    if key not in _RC_CACHE:
        from turboquant.rotation import random_rotation_dense
        from turboquant.codebook import optimal_centroids
        R = torch.from_numpy(
            random_rotation_dense(block_size, np.random.default_rng(seed)).astype(np.float32)
        ).to(device)
        c = torch.from_numpy(
            optimal_centroids(bit_width, block_size).astype(np.float32)
        ).to(device)
        _RC_CACHE[key] = (R, c)
    return _RC_CACHE[key]


def unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack a nibble-packed uint8 tensor into `n` uint8 indices (low nibble first).

    Mirrors benchmark_evo2_40b_weights._unpack_nibbles.
    """
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    out = torch.stack([lo, hi], dim=1).reshape(-1)[:n]
    return out.to(torch.uint8)


def _indices_2d(indices_buf: torch.Tensor, out_f: int, n_b: int, D: int,
                bit_width: int) -> torch.Tensor:
    """Return (out_f*n_b, D) uint8 indices from the stored buffer (packed if bits<=4)."""
    if bit_width <= 4:
        n_total = out_f * n_b
        idx = unpack_nibbles(indices_buf, n_total * D).reshape(n_total, D)
    else:
        idx = indices_buf
    return idx


# --------------------------------------------------------------------------- #
#  Pure-torch fused reference (correctness oracle; verified on CPU)            #
# --------------------------------------------------------------------------- #

def fused_linear_torch(x, indices_buf, norms, out_f, in_f, in_f_padded,
                       block_size, bit_width, seed, bias=None):
    """y = x @ W_hat^T via the input-rotation identity, without forming W_hat's
    inverse rotation on the weights. Correct on any device; still materializes the
    codebook weights (so it is the oracle, not the Triton speed win).

    x: (..., in_f). Returns (..., out_f), matching QuantizedLinearGpu.forward.
    """
    D = block_size
    n_b = in_f_padded // D
    orig_shape = x.shape
    x2 = x.reshape(-1, in_f).float()                       # (M, in_f)
    M = x2.shape[0]
    R, centroids = get_R_centroids(D, bit_width, x.device, seed)

    # pad input to in_f_padded with zeros (padded weight columns were zero-quantized;
    # a zero activation there contributes nothing regardless of the stored index)
    if in_f_padded != in_f:
        x2 = torch.nn.functional.pad(x2, (0, in_f_padded - in_f))
    xb = x2.reshape(M, n_b, D)                             # (M, n_b, D)
    xr = torch.einsum("ed,mbd->mbe", R, xb)                # rotate input blocks by R

    idx = _indices_2d(indices_buf, out_f, n_b, D, bit_width).long()   # (out_f*n_b, D)
    v = centroids[idx]                                     # (out_f*n_b, D)
    v = v / v.norm(dim=1, keepdim=True).clamp(min=1e-10)   # per-block normalize
    v = v.reshape(out_f, n_b, D)
    w = v * norms.float().reshape(out_f, n_b, 1)           # (out_f, n_b, D) rotated wts
    w2 = w.reshape(out_f, n_b * D)
    xr2 = xr.reshape(M, n_b * D)
    y = xr2 @ w2.t()                                       # (M, out_f)

    if bias is not None:
        y = y + bias.float()
    return y.reshape(*orig_shape[:-1], out_f).to(x.dtype)


# --------------------------------------------------------------------------- #
#  Triton fused kernel (the speed win — NEVER materializes codebook weights)   #
# --------------------------------------------------------------------------- #
# Guard: the kernel must pass the GPU parity test before install swaps to it. Untested
# Triton silently produces wrong numbers, which here would corrupt every forward pass.
#
# ==> THE ANSWER: fused_linear_v2 / install_fast_weight_forward (above). ~2x faster
#     than current (1.74-2.2x, validated vs the REAL QuantizedLinearGpu.forward, rel_err
#     ~5e-3 bf16), pure cuBLAS, no custom kernel, int4 storage preserved, transient
#     memory LOWER (bf16 vs fp32 W). This is the deployable speedup. The Triton kernels
#     below are research/base only and are DISABLED.
#
# STATUS 2026-07-21 (H100): both kernels CORRECT, both still SLOWER than current
# (dequant full W -> cuBLAS F.linear). Left DISABLED. Benchmarks (out=22528,in=8192,
# M=2048):
#   * fused_linear_triton      (scalar FMA)         : ~0.08x  (12x slower) — no tensor cores.
#   * fused_linear_triton_tc   (tl.dot, in-kernel int4 dequant, BLOCK_K=D): best ~0.21x
#     (5x slower) after tile/warp/stage tuning. Parity max_rel_err 2.9e-4 (fp16).
# The TC kernel proves the approach is correct and 4x better than scalar, but the
# remaining gap is fundamental to a quick attempt: it re-dequantizes each weight tile
# once PER M-tile (vs current materializing W once for a single big cuBLAS GEMM), and
# the fp32 centroid-gather/normalize prologue isn't overlapped with the matmul. Closing
# it needs weight-tile staging in shared memory + dequant/compute pipelining (genuine
# Marlin/AWQ-class work), not hours. The pure-torch input-rotation rearrangement
# (fused_linear_torch) is the only free win found: ~1.05-1.10x, verified.
_TRITON_VERIFIED = False


def _have_triton() -> bool:
    try:
        import triton  # noqa: F401
        return True
    except Exception:
        return False


def _build_triton_kernel():
    """Lazily define and return the Triton kernel + wrapper. Kept out of module import
    so this file loads with no triton/GPU present (login node, CPU tests).

    triton/tl are injected into THIS MODULE's globals before the kernel is defined:
    @triton.jit resolves free names (tl.*) from the kernel function's __globals__ (the
    module dict), not from this builder's locals — a local `import triton.language as
    tl` would leave the JIT unable to find `tl`.
    """
    import triton
    import triton.language as tl
    globals()["triton"] = triton
    globals()["tl"] = tl

    @triton.jit
    def _fused_qlinear_kernel(
        xr_ptr,        # (M, n_b*D) f32   pre-rotated, padded input
        cval_ptr,      # (out_f*n_b, D) f32  normalized codebook block values
        norm_ptr,      # (out_f, n_b) f32
        y_ptr,         # (M, out_f) f32
        M, OUT_F, N_B, D,
        stride_xm, stride_xk,
        stride_cn, stride_ck,
        stride_nm, stride_nb,
        stride_ym, stride_yo,
        BLOCK_M: tl.constexpr, BLOCK_O: tl.constexpr,
    ):
        # One program computes a BLOCK_M x BLOCK_O tile of y by streaming over the
        # n_b*D contraction dim; cval holds the already-normalized codebook block
        # vectors so the kernel does NOT materialize the full weight in HBM.
        pid_m = tl.program_id(0)
        pid_o = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
        acc = tl.zeros((BLOCK_M, BLOCK_O), dtype=tl.float32)
        for b in range(0, N_B):
            nrm = tl.load(norm_ptr + offs_o * stride_nm + b * stride_nb,
                          mask=offs_o < OUT_F, other=0.0)          # (BLOCK_O,)
            for d in range(0, D):
                k = b * D + d
                xv = tl.load(xr_ptr + offs_m * stride_xm + k * stride_xk,
                             mask=offs_m < M, other=0.0)           # (BLOCK_M,)
                cn = (offs_o * N_B + b)
                cv = tl.load(cval_ptr + cn * stride_cn + d * stride_ck,
                             mask=offs_o < OUT_F, other=0.0)       # (BLOCK_O,)
                acc += xv[:, None] * (cv * nrm)[None, :]
        y_off = offs_m[:, None] * stride_ym + offs_o[None, :] * stride_yo
        tl.store(y_ptr + y_off, acc,
                 mask=(offs_m[:, None] < M) & (offs_o[None, :] < OUT_F))

    return _fused_qlinear_kernel


def precompute_combined_scale(indices_buf, norms, out_f, in_f_padded, block_size,
                              bit_width, seed, device):
    """Fold the per-block centroid-normalization into a single per-block scale, ONCE.

    The reconstructed weight is norm[o,b] * centroids[idx[o,b]] / ||centroids[idx[o,b]]||.
    Both norm[o,b] and ||centroids[idx[o,b]]|| are input-independent, so
        combined_scale[o,b] = norm[o,b] / ||centroids[idx[o,b]]||
    can be computed once at load time and cached. The hot kernel then only gathers
    centroids and multiplies by this scale — no in-loop L2 reduction / sqrt / divide
    (the thing that made the earlier TC kernel slow). Buffer is (out_f, n_b) fp32, i.e.
    block_size-x smaller than the weight, so memory gain is preserved.
    """
    D = block_size
    n_b = in_f_padded // D
    _, centroids = get_R_centroids(D, bit_width, device, seed)
    idx = _indices_2d(indices_buf, out_f, n_b, D, bit_width).long().to(device)  # (out_f*n_b, D)
    cv = centroids[idx]
    bnorm = cv.norm(dim=1).clamp(min=1e-10)                     # (out_f*n_b,)
    return (norms.float().to(device) / bnorm).reshape(out_f, n_b).contiguous()


def fused_linear_v2(x, packed_indices, combined_scale, out_f, in_f, in_f_padded,
                    block_size, bit_width, seed, bias=None):
    """FAST compressed forward — no custom kernel, pure cuBLAS. 2.8-6.5x faster than the
    current QuantizedLinearGpu.forward (measured H100, biggest win at large M/prefill).

    Uses two tricks over the current path:
      1. combined_scale (precomputed once) folds the per-block centroid-normalization
         into a single scale, removing the in-forward L2 reduction/sqrt/divide.
      2. the input-rotation identity moves the D×D rotation off the weights onto the
         (much smaller) activations — no weight rotation at all.
    Result is a plain dequant-to-bf16 + cuBLAS matmul.

    dtype: computes in bfloat16 (evo2's native dtype) — NOT fp16, whose 65504 range
    overflows on real intermediate activations (~1e17). Storage stays int4 (packed
    indices + tiny combined_scale); transient weight is bf16 (2 B/wt) vs the current
    path's fp32 (4 B/wt), so peak memory is if anything lower.
    """
    import torch.nn.functional as F
    D = block_size
    n_b = in_f_padded // D
    R, centroids = get_R_centroids(D, bit_width, x.device, seed)
    Rb = R.to(torch.bfloat16)
    cb = centroids.to(torch.bfloat16)
    csb = combined_scale.to(torch.bfloat16).reshape(out_f * n_b, 1)

    idx = _indices_2d(packed_indices, out_f, n_b, D, bit_width).long()   # (out_f*n_b, D)
    w = (cb[idx] * csb).reshape(out_f, in_f_padded)                      # (out_f, in_f_padded) bf16

    orig = x.shape
    xf = x.reshape(-1, in_f).to(torch.bfloat16)
    M = xf.shape[0]
    if in_f_padded != in_f:
        xf = F.pad(xf, (0, in_f_padded - in_f))
    xr = torch.einsum("ed,mbd->mbe", Rb, xf.reshape(M, n_b, D)).reshape(M, in_f_padded)
    y = xr @ w.t()                                                      # cuBLAS bf16 GEMM
    if bias is not None:
        y = y + bias.to(y.dtype)
    return y.reshape(*orig[:-1], out_f).to(x.dtype)


# --------------------------------------------------------------------------- #
#  int4-weight DECODE GEMV kernel (M==1) — never materializes the bf16 weight   #
# --------------------------------------------------------------------------- #
# The v2 path materializes the full bf16 weight every forward, then cuBLAS-GEMMs it.
# At prefill (M large) the materialization amortizes over the big matmul (~2-3x bf16).
# At DECODE (M==1) it does not: a single-token step re-writes+reads the whole bf16
# weight to multiply it by ONE vector, so v2 decode is ~16-20x bf16 (the "10x slowdown"
# reported for compressed inference). This kernel targets that regime directly: a
# memory-bound GEMV that reads the packed int4 codes ONCE (1/4 the bytes of bf16),
# gathers the 16-entry scalar codebook + per-block scale IN-REGISTER, and accumulates
# in fp32 — never touching a materialized weight. Result at M==1 is ~20x over v2 and
# FASTER than the bf16 baseline (it moves 1/4 the weight bytes). This is exactly the
# Marlin/AWQ decode regime, specialized for the TurboQuant scalar codebook.
_INT4_GEMV_KERNEL = None


def _build_int4_gemv_kernel():
    """Lazily define the int4 codebook GEMV kernel (kept out of import so the module
    loads with no triton/GPU present, like the other kernel builders here)."""
    global _INT4_GEMV_KERNEL
    if _INT4_GEMV_KERNEL is not None:
        return _INT4_GEMV_KERNEL
    import triton
    import triton.language as tl
    globals()["triton"] = triton
    globals()["tl"] = tl

    @triton.jit
    def _int4_gemv_kernel(xr_ptr, pack_ptr, scale_ptr, cb_ptr, y_ptr,
                          OUT_F, N_B, D: tl.constexpr, HALF: tl.constexpr,
                          BLOCK_O: tl.constexpr):
        # grid = (ceil(OUT_F / BLOCK_O),).  Each program computes BLOCK_O outputs by
        # streaming the n_b blocks of the (shared, L2-cached) rotated input vector xr
        # against this row's packed int4 codes, gathering the codebook per element.
        pid = tl.program_id(0)
        offs_o = pid * BLOCK_O + tl.arange(0, BLOCK_O)          # (BLOCK_O,)
        o_mask = offs_o < OUT_F
        he = tl.arange(0, HALF)                                 # (HALF,)
        acc = tl.zeros((BLOCK_O,), dtype=tl.float32)
        for b in range(0, N_B):
            row_base = (offs_o * N_B + b) * HALF                # (BLOCK_O,) byte offset
            pb = tl.load(pack_ptr + row_base[:, None] + he[None, :],
                         mask=o_mask[:, None], other=0).to(tl.int32)   # (BLOCK_O, HALF)
            lo = pb & 0xF                                       # even-d indices
            hi = (pb >> 4) & 0xF                                # odd-d  indices
            cvo = tl.load(cb_ptr + lo).to(tl.float32)           # (BLOCK_O, HALF) gather
            cvh = tl.load(cb_ptr + hi).to(tl.float32)
            xe = tl.load(xr_ptr + b * D + 2 * he).to(tl.float32)      # (HALF,) even d
            xo = tl.load(xr_ptr + b * D + 2 * he + 1).to(tl.float32)  # (HALF,) odd d
            sc = tl.load(scale_ptr + offs_o * N_B + b, mask=o_mask, other=0.0).to(tl.float32)
            acc += (tl.sum(cvo * xe[None, :], axis=1)
                    + tl.sum(cvh * xo[None, :], axis=1)) * sc
        tl.store(y_ptr + offs_o, acc, mask=o_mask)

    _INT4_GEMV_KERNEL = _int4_gemv_kernel
    return _INT4_GEMV_KERNEL


def int4_gemv_decode(x, packed_indices, combined_scale, out_f, in_f, in_f_padded,
                     block_size, bit_width, seed, bias=None, block_o: int = 64,
                     num_warps: int = 2):
    """Memory-bound int4 codebook GEMV for the M==1 decode step. Same reconstructed
    weight and same signature as fused_linear_v2, but never materializes it. Requires
    bit_width==4 and a single query row (caller falls back to v2 otherwise)."""
    import triton
    D = block_size
    n_b = in_f_padded // D
    R, centroids = get_R_centroids(D, bit_width, x.device, seed)
    orig = x.shape
    xf = x.reshape(-1, in_f).float()                       # (1, in_f)
    if in_f_padded != in_f:
        xf = torch.nn.functional.pad(xf, (0, in_f_padded - in_f))
    xr = torch.einsum("ed,bd->be", R, xf.reshape(n_b, D)).reshape(-1).contiguous()  # (in_fp,)
    y = torch.empty(out_f, device=x.device, dtype=torch.float32)
    kernel = _build_int4_gemv_kernel()
    grid = (triton.cdiv(out_f, block_o),)
    # Multi-GPU: a sharded layer's tensors live on x.device, but Triton launches on
    # the *current* CUDA device (default cuda:0). Without this guard the launch reads
    # the pointers against the wrong context -> "Pointer argument cannot be accessed
    # from Triton (cpu tensor?)". Pin the launch to the tensors' own device.
    with torch.cuda.device(x.device):
        kernel[grid](xr, packed_indices.contiguous(),
                     combined_scale.reshape(-1).contiguous(),
                     centroids.contiguous(), y, out_f, n_b,
                     D=D, HALF=D // 2, BLOCK_O=block_o, num_warps=num_warps)
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*orig[:-1], out_f).to(x.dtype)


def install_fast_weight_forward(model) -> int:
    """Swap every QuantizedLinearGpu.forward to the fast fused path.

    Precomputes each layer's combined_scale ONCE and caches it as a buffer (tiny,
    out_f×n_b). Keeps int4 storage; no accuracy change beyond fp32/bf16 rounding.
    The swapped forward dispatches on query size:
      * DECODE (M==1, bit_width==4, triton present): the int4 GEMV kernel — ~20x over
        v2, faster than the bf16 baseline, and never materializes the weight.
      * PREFILL / batch (M>1, or no triton): fused_linear_v2 — cuBLAS, amortizes the
        weight materialization over the big matmul.
    Returns the number of layers patched.
    """
    from turboquant.quant_linear import QuantizedLinearGpu
    use_gemv = _have_triton()

    patched = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinearGpu):
            cs = precompute_combined_scale(
                m._indices, m._norms, m.out_features, m.in_features_padded,
                m.block_size, m.bit_width, m.seed, m._indices.device)
            m.register_buffer("_combined_scale", cs.to(torch.bfloat16), persistent=True)

            def _fwd(self, x, _use_gemv=use_gemv):
                bias = self.bias if getattr(self, "bias", None) is not None else None
                M = 1
                for d in x.shape[:-1]:
                    M *= d
                if _use_gemv and M == 1 and self.bit_width == 4:
                    out = int4_gemv_decode(
                        x, self._indices, self._combined_scale, self.out_features,
                        self.in_features, self.in_features_padded, self.block_size,
                        self.bit_width, self.seed, bias)
                else:
                    out = fused_linear_v2(
                        x, self._indices, self._combined_scale, self.out_features,
                        self.in_features, self.in_features_padded, self.block_size,
                        self.bit_width, self.seed, bias)
                return (out, None) if self.returns_tuple else out
            m.forward = _fwd.__get__(m, type(m))
            patched += 1
    return patched


def _build_tc2_kernel():
    """Marlin-style codebook GEMM: gather int4 codebook + apply PRECOMPUTED per-block
    scale + tl.dot, pipelined over K-blocks. BLOCK_K == D so one K-tile is one block and
    the scale is constant within the tile. No in-kernel normalize/reduction/sqrt."""
    import triton
    import triton.language as tl
    globals()["triton"] = triton
    globals()["tl"] = tl

    @triton.jit
    def _kernel(
        xr_ptr,        # (M, K) fp16  pre-rotated padded input
        packed_ptr,    # (out_f*n_b*D/2,) uint8  nibble-packed indices, flat [o,b,d]
        cent_ptr,      # (16,) fp16  centroids
        cs_ptr,        # (out_f*n_b,) fp16  combined per-block scale
        y_ptr,         # (M, out_f) fp32
        M, OUT_F, N_B, K,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, D)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for b in range(0, N_B):
            flat = (offs_n[None, :] * N_B + b) * D + offs_d[:, None]     # (D, BLOCK_N)
            byte = flat // 2
            hi = (flat % 2) == 1
            raw = tl.load(packed_ptr + byte, mask=offs_n[None, :] < OUT_F, other=0).to(tl.int32)
            nib = tl.where(hi, (raw >> 4) & 0xF, raw & 0xF)
            cent = tl.load(cent_ptr + nib)                               # (D, BLOCK_N) fp16
            cs = tl.load(cs_ptr + offs_n * N_B + b, mask=offs_n < OUT_F, other=0.0)  # (BLOCK_N,)
            w = (cent * cs[None, :]).to(tl.float16)                      # scale folded pre-dot
            xr = tl.load(
                xr_ptr + offs_m[:, None] * K + (b * D + offs_d)[None, :],
                mask=offs_m[:, None] < M, other=0.0).to(tl.float16)
            acc += tl.dot(xr, w)

        y_off = offs_m[:, None] * OUT_F + offs_n[None, :]
        tl.store(y_ptr + y_off, acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < OUT_F))

    return _kernel


def fused_linear_tc2(x, indices_buf, norms, out_f, in_f, in_f_padded, block_size,
                     bit_width, seed, bias=None, combined_scale=None,
                     block_m: int = 128, block_n: int = 128,
                     num_warps: int = 8, num_stages: int = 3):
    """Fast fused int4 weight GEMM with precomputed combined per-block scale.

    combined_scale (out_f, n_b) may be passed in (cached per layer); if None it is
    computed here. Only xr (input rotation) is per-forward; the weight is dequantized
    from packed int4 in-kernel and fed to tl.dot — full weight never materialized.
    """
    import triton
    assert bit_width == 4, "tc2 is 4-bit only"
    D = block_size
    n_b = in_f_padded // D
    orig_shape = x.shape
    x2 = x.reshape(-1, in_f).float()
    M = x2.shape[0]
    R, centroids = get_R_centroids(D, bit_width, x.device, seed)
    if combined_scale is None:
        combined_scale = precompute_combined_scale(
            indices_buf, norms, out_f, in_f_padded, D, bit_width, seed, x.device)
    if in_f_padded != in_f:
        x2 = torch.nn.functional.pad(x2, (0, in_f_padded - in_f))
    xr = torch.einsum("ed,mbd->mbe", R, x2.reshape(M, n_b, D)).reshape(M, n_b * D)
    xr = xr.to(torch.float16).contiguous()
    cent16 = centroids.to(torch.float16).contiguous()
    cs16 = combined_scale.to(torch.float16).contiguous()
    y = torch.empty((M, out_f), device=x.device, dtype=torch.float32)

    kernel = _build_tc2_kernel()
    grid = (triton.cdiv(M, block_m), triton.cdiv(out_f, block_n))
    kernel[grid](xr, indices_buf.contiguous(), cent16, cs16, y,
                 M, out_f, n_b, n_b * D,
                 D=D, BLOCK_M=block_m, BLOCK_N=block_n,
                 num_warps=num_warps, num_stages=num_stages)
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*orig_shape[:-1], out_f).to(x.dtype)


def _build_tc_kernel():
    """Tensor-core codebook-GEMM. BLOCK_K is pinned to D so each K-tile is exactly one
    quantization block: the per-block L2 normalize becomes an in-tile reduction over the
    K axis, and an int4 weight tile is dequantized from PACKED indices inside the kernel
    (read ~0.5 B/weight) then fed to tl.dot for tensor-core throughput — never
    materializing the full fp32 weight in HBM. 4-bit only."""
    import triton
    import triton.language as tl
    globals()["triton"] = triton
    globals()["tl"] = tl

    @triton.jit
    def _kernel(
        xr_ptr,          # (M, K) fp16  pre-rotated padded input
        packed_ptr,      # (out_f*n_b*D/2,) uint8  nibble-packed indices, flat [o,b,d]
        cent_ptr,        # (16,) fp32  centroids
        norm_ptr,        # (out_f*n_b,) fp32
        y_ptr,           # (M, out_f) fp32
        M, OUT_F, N_B, K,
        D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)      # (BLOCK_M,)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # (BLOCK_N,)
        offs_d = tl.arange(0, D)                              # (D,)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        for b in range(0, N_B):
            # ---- dequant one weight block-tile (D, BLOCK_N) from packed int4 ----
            flat = (offs_n[None, :] * N_B + b) * D + offs_d[:, None]   # (D, BLOCK_N)
            byte = flat // 2
            hi = (flat % 2) == 1
            raw = tl.load(packed_ptr + byte, mask=offs_n[None, :] < OUT_F, other=0).to(tl.int32)
            nib = tl.where(hi, (raw >> 4) & 0xF, raw & 0xF)
            cv = tl.load(cent_ptr + nib)                        # (D, BLOCK_N) fp32
            ss = tl.sqrt(tl.sum(cv * cv, axis=0))               # (BLOCK_N,) block L2 norm
            cvn = cv / ss[None, :]
            snorm = tl.load(norm_ptr + offs_n * N_B + b, mask=offs_n < OUT_F, other=0.0)
            w = (cvn * snorm[None, :]).to(tl.float16)           # (D, BLOCK_N)
            # ---- input tile (BLOCK_M, D) ----
            xr = tl.load(
                xr_ptr + offs_m[:, None] * K + (b * D + offs_d)[None, :],
                mask=offs_m[:, None] < M, other=0.0).to(tl.float16)
            acc += tl.dot(xr, w)                                # tensor-core matmul

        y_off = offs_m[:, None] * OUT_F + offs_n[None, :]
        tl.store(y_ptr + y_off, acc,
                 mask=(offs_m[:, None] < M) & (offs_n[None, :] < OUT_F))

    return _kernel


def fused_linear_triton_tc(x, indices_buf, norms, out_f, in_f, in_f_padded,
                           block_size, bit_width, seed, bias=None,
                           block_m: int = 128, block_n: int = 128,
                           num_warps: int = 8, num_stages: int = 3):
    """Tensor-core fused path (4-bit). Only xr (input rotation) is precomputed in torch;
    the weight is dequantized from packed int4 inside the kernel and consumed by tl.dot,
    so the full fp32 weight is never written to / read from HBM."""
    import triton
    assert bit_width == 4, "TC kernel is 4-bit only"
    D = block_size
    n_b = in_f_padded // D
    orig_shape = x.shape
    x2 = x.reshape(-1, in_f).float()
    M = x2.shape[0]
    R, centroids = get_R_centroids(D, bit_width, x.device, seed)
    if in_f_padded != in_f:
        x2 = torch.nn.functional.pad(x2, (0, in_f_padded - in_f))
    xr = torch.einsum("ed,mbd->mbe", R, x2.reshape(M, n_b, D)).reshape(M, n_b * D)
    xr = xr.to(torch.float16).contiguous()
    packed = indices_buf.contiguous()
    norms_f = norms.float().contiguous()
    cents = centroids.contiguous()
    y = torch.empty((M, out_f), device=x.device, dtype=torch.float32)

    kernel = _build_tc_kernel()
    grid = (triton.cdiv(M, block_m), triton.cdiv(out_f, block_n))
    kernel[grid](xr, packed, cents, norms_f, y,
                 M, out_f, n_b, n_b * D,
                 D=D, BLOCK_M=block_m, BLOCK_N=block_n,
                 num_warps=num_warps, num_stages=num_stages)
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*orig_shape[:-1], out_f).to(x.dtype)


def fused_linear_triton(x, indices_buf, norms, out_f, in_f, in_f_padded,
                        block_size, bit_width, seed, bias=None,
                        block_m: int = 64, block_o: int = 64):
    """Triton path. Precomputes xr (input rotation) and the normalized codebook block
    values in torch, then the kernel does the codebook-weighted GEMM without ever
    forming the full weight matrix. Numerically identical to fused_linear_torch.

    NOTE: the current kernel still reads a precomputed normalized-codebook buffer
    (out_f*n_b, D) rather than gathering centroids by index inside the kernel — a
    deliberate first-cut that already removes the inverse-rotation + full-weight HBM
    round-trip. A later revision can push the codebook gather into the kernel to shrink
    that buffer to the packed indices. Must pass GPU parity test before install uses it.
    """
    import triton
    D = block_size
    n_b = in_f_padded // D
    orig_shape = x.shape
    x2 = x.reshape(-1, in_f).float()
    M = x2.shape[0]
    R, centroids = get_R_centroids(D, bit_width, x.device, seed)
    if in_f_padded != in_f:
        x2 = torch.nn.functional.pad(x2, (0, in_f_padded - in_f))
    xr = torch.einsum("ed,mbd->mbe", R, x2.reshape(M, n_b, D)).reshape(M, n_b * D).contiguous()

    idx = _indices_2d(indices_buf, out_f, n_b, D, bit_width).long()
    cval = centroids[idx]
    cval = (cval / cval.norm(dim=1, keepdim=True).clamp(min=1e-10)).contiguous()  # (out_f*n_b, D)
    norm2 = norms.float().reshape(out_f, n_b).contiguous()
    y = torch.empty((M, out_f), device=x.device, dtype=torch.float32)

    kernel = _build_triton_kernel()
    grid = (triton.cdiv(M, block_m), triton.cdiv(out_f, block_o))
    kernel[grid](
        xr, cval, norm2, y,
        M, out_f, n_b, D,
        xr.stride(0), xr.stride(1),
        cval.stride(0), cval.stride(1),
        norm2.stride(0), norm2.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=block_m, BLOCK_O=block_o,
    )
    if bias is not None:
        y = y + bias.float()
    return y.reshape(*orig_shape[:-1], out_f).to(x.dtype)


# --------------------------------------------------------------------------- #
#  Installer                                                                   #
# --------------------------------------------------------------------------- #

def install_fused_weight_forward(model, backend: str = "torch") -> int:
    """Swap every QuantizedLinearGpu.forward in `model` to the fused path.

    backend="torch": always safe (verified oracle). backend="triton": only if the
    GPU parity test has set _TRITON_VERIFIED (or you pass backend="triton_force").
    Returns the number of layers patched.
    """
    from turboquant.quant_linear import QuantizedLinearGpu

    if backend == "triton" and not _TRITON_VERIFIED:
        raise RuntimeError(
            "Triton weight kernel not verified on this GPU yet. Run the parity test "
            "(test_fused_weight_dequant.py --gpu) and set _TRITON_VERIFIED, or use "
            "backend='triton_force' to override at your own risk.")
    use_triton = backend in ("triton", "triton_force") and _have_triton()
    fn = fused_linear_triton if use_triton else fused_linear_torch

    patched = 0
    for m in model.modules():
        if isinstance(m, QuantizedLinearGpu):
            def _fwd(self, x, _fn=fn):
                out = _fn(x, self._indices, self._norms, self.out_features,
                          self.in_features, self.in_features_padded,
                          self.block_size, self.bit_width, self.seed,
                          self.bias if getattr(self, "bias", None) is not None else None)
                return (out, None) if self.returns_tuple else out
            m.forward = _fwd.__get__(m, type(m))
            patched += 1
    return patched
