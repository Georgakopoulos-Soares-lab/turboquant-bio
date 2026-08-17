"""fused_kv_attention.py — fused 4-bit-KV dequant + decode attention (Triton).

This is the fused counterpart to `turboquant.kv_cache.streaming_kv_attention`.
The streaming path dequantizes each packed K/V block into a bf16 tensor in HBM
and then re-reads it for the attention matmul (a dequant HBM round-trip every
decode step — the source of the decode-latency regression reported in the
TurboQuant-Bio paper).  The kernel here fuses the two: it loads the packed
4-bit codes and the per-channel(K)/per-token(V) fp16 scales/zero-points, unpacks
and dequantizes **in registers**, and immediately consumes the dequantized K/V
with a numerically-stable FlashAttention-style online softmax accumulated in
fp32.  The dequantized K/V never touch global memory.

RoPE / rotation note (verified against the code, see README_fused_kernel.md):
Evo2 attention applies RoPE to Q/K, but the KV quantizer applies NO orthogonal
rotation to the cache (pure asymmetric per-channel/per-token min-max), and the
cache stores the already-post-RoPE K.  The decode query is likewise post-RoPE.
Because the dequant is a scalar affine map (code*scale + zero) with no Pi matrix,
the RoPE-then-rotation ordering problem that TurboESM had to solve does NOT
arise: the kernel dequantizes and attends directly, with no in-kernel rotation.

Scope: decode (Sq == 1) over the packed int4 store.  Prefill / extend (Sq > 1)
and any non-4-bit width fall back to the reference torch streaming path.
"""

from __future__ import annotations

import contextlib as _contextlib
import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

# The Tier-1 prefill optimization attends the fresh chunk with torch's built-in
# scaled_dot_product_attention (flash-backed on H100), NOT the standalone flash_attn
# package: importing flash_attn at module load collides with vortex's bundled,
# version-pinned flash-attn op registration (vortex supports <=2.7.4; envs ship 2.8+)
# and corrupts the stock attention path. SDPA is always present in torch and has no
# such conflict.
_HAS_SDPA = hasattr(torch.nn.functional, "scaled_dot_product_attention")

import os as _os
_DEBUG_NO_AUTOTUNE = bool(int(_os.environ.get("TQ_DEBUG_NO_AUTOTUNE", "0")))


# ---------------------------------------------------------------------------
# Triton kernel: one packed segment's contribution to the online softmax.
# grid = (B * Hq,).  Each program streams one (batch, query-head) over all the
# key tokens of ONE segment, dequantizing K/V tiles in registers and folding
# them into the running (m, l, acc) accumulators held in HBM.  Launch once per
# packed segment (accumulators carry across launches); the fp16 residual is
# folded in fp32 by the wrapper.
#
# NOTE: superseded by the split-K kernel below (`_fused_seg_splitk_kernel`),
# which `fused_decode_attention` always uses. Kept only as the original,
# simpler reference; UNUSED and NOT updated for the Phase-3 (B,H,S,D) packed
# layout, so its row_byte addressing below still assumes the old (B,S,H,D)
# layout and would silently mis-read a Phase-3 cache if ever revived.
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _fused_seg_decode_kernel(
        Q_ptr,          # (B, Hq, D) fp32, query pre-scaled by softmax_scale
        Kc_ptr,         # packed K codes, uint8, 1-D  (B*S*Hkv*D // 2 bytes)
        Vc_ptr,         # packed V codes, uint8, 1-D
        Ksc_ptr, Kz_ptr,  # (B, Hkv, D) fp32   per-channel K scale / zero
        Vsc_ptr, Vz_ptr,  # (B, S, Hkv) fp32   per-token   V scale / zero
        M_ptr, L_ptr,   # (B, Hq) fp32  running max / denom
        Acc_ptr,        # (B, Hq, D) fp32 running weighted sum
        B, S, Hq, Hkv, G,
        BITS: tl.constexpr,
        D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // Hq
        h = pid % Hq
        hk = h // G

        offs_d = tl.arange(0, D)               # (D,)
        HD2 = D // (8 // BITS)                  # bytes per (token,head) row = D/2 for 4-bit
        per = 8 // BITS                          # codes per byte (2 for 4-bit)

        q = tl.load(Q_ptr + (b * Hq + h) * D + offs_d)          # (D,) fp32
        ksc = tl.load(Ksc_ptr + (b * Hkv + hk) * D + offs_d)    # (D,)
        kz = tl.load(Kz_ptr + (b * Hkv + hk) * D + offs_d)      # (D,)

        m_i = tl.load(M_ptr + b * Hq + h)                       # scalar
        l_i = tl.load(L_ptr + b * Hq + h)
        acc = tl.load(Acc_ptr + (b * Hq + h) * D + offs_d)      # (D,)

        # nibble selection: byte column = d // per, shift = (d % per) * BITS
        byte_col = (offs_d // per).to(tl.int64)                 # (D,)
        shift = ((offs_d % per) * BITS).to(tl.int32)            # (D,)
        mask_bits = (1 << BITS) - 1

        base_bh = (b.to(tl.int64) * S) * Hkv + hk               # will add token*Hkv
        for n0 in range(0, S, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)                 # (BLOCK_N,) tokens
            mask_n = offs_n < S
            tok = offs_n.to(tl.int64)
            # byte base per token for this (b, hk): ((b*S + tok)*Hkv + hk) * HD2
            row_byte = ((b.to(tl.int64) * S + tok) * Hkv + hk) * HD2   # (BLOCK_N,)
            addr = row_byte[:, None] + byte_col[None, :]        # (BLOCK_N, D)

            # ---- K dequant in registers ----
            rawk = tl.load(Kc_ptr + addr, mask=mask_n[:, None], other=0).to(tl.int32)
            codek = (rawk >> shift[None, :]) & mask_bits        # (BLOCK_N, D)
            kf = codek.to(tl.float32) * ksc[None, :] + kz[None, :]

            scores = tl.sum(q[None, :] * kf, axis=1)            # (BLOCK_N,)
            scores = tl.where(mask_n, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)                          # (BLOCK_N,)
            p = tl.where(mask_n, p, 0.0)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            # ---- V dequant in registers ----
            rawv = tl.load(Vc_ptr + addr, mask=mask_n[:, None], other=0).to(tl.int32)
            codev = (rawv >> shift[None, :]) & mask_bits        # (BLOCK_N, D)
            vsc = tl.load(Vsc_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vz = tl.load(Vz_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vf = codev.to(tl.float32) * vsc[:, None] + vz[:, None]   # (BLOCK_N, D)

            acc = acc * alpha + tl.sum(p[:, None] * vf, axis=0)      # (D,)
            m_i = m_new

        tl.store(M_ptr + b * Hq + h, m_i)
        tl.store(L_ptr + b * Hq + h, l_i)
        tl.store(Acc_ptr + (b * Hq + h) * D + offs_d, acc)


def _launch_segment(qf, seg, m, l, acc, B, Hq, Hkv, G, D, bits, block_n):
    """Fold one packed segment into (m, l, acc) via the (v1) 1-program kernel."""
    S = int(seg["k_shape"][1])
    ksc, kz, vsc, vz = _seg_scales_f32(seg)
    grid = (B * Hq,)
    _fused_seg_decode_kernel[grid](
        qf, seg["k_packed"], seg["v_packed"],
        ksc, kz, vsc, vz,
        m, l, acc,
        B, S, Hq, Hkv, G,
        BITS=bits, D=D, BLOCK_N=block_n,
        num_warps=4,
    )


def _seg_scales_f32(seg):
    """Cache fp32-contiguous scales/zeros on the segment (converted once, not
    every decode step)."""
    if "k_scale_f32" not in seg:
        seg["k_scale_f32"] = seg["k_scale"].squeeze(1).float().contiguous()   # (B,Hkv,D)
        seg["k_zero_f32"] = seg["k_zero"].squeeze(1).float().contiguous()
        seg["v_scale_f32"] = seg["v_scale"].squeeze(-1).float().contiguous()  # (B,S,Hkv)
        seg["v_zero_f32"] = seg["v_zero"].squeeze(-1).float().contiguous()
    return (seg["k_scale_f32"], seg["k_zero_f32"],
            seg["v_scale_f32"], seg["v_zero_f32"])


if _HAS_TRITON:

    # Phase 2 (tuning): the kernel was hardcoded to BLOCK_N=min(key_block,128),
    # num_warps=4, num_stages unset (Triton default) -- never swept. Nsight
    # Compute (Phase 1) showed occupancy capped at 18.75% by register pressure
    # (Block Limit Registers=3 of a possible 32) and 31.84 sectors/request on
    # the packed-K/V loads (~8x worse than the ~4 sectors/request of a fully
    # coalesced access) -- both BLOCK_N and num_warps directly affect the
    # per-thread register budget and how loads vectorize, so both are real
    # tuning levers here, not just launch-config bookkeeping.
    _AUTOTUNE_CONFIGS = [
        triton.Config({"BLOCK_N": bn}, num_warps=nw, num_stages=ns)
        for bn in (32, 64, 128, 256)
        for nw in (1, 2, 4, 8)
        for ns in (1, 2, 3, 4)
    ]

    if _DEBUG_NO_AUTOTUNE:
        def _no_autotune_decorator(fn):
            return fn
        _autotune_decorator = _no_autotune_decorator
        _autotune_decorator_v4 = _no_autotune_decorator
    else:
        _autotune_decorator = triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["S", "D", "BITS"])
        # v4 kernel is 4-bit-only (no BITS arg) -> key on (S, D)
        _autotune_decorator_v4 = triton.autotune(configs=_AUTOTUNE_CONFIGS, key=["S", "D"])

    @_autotune_decorator
    @triton.jit
    def _fused_seg_splitk_kernel(
        Q_ptr, Kc_ptr, Vc_ptr,
        Ksc_ptr, Kz_ptr, Vsc_ptr, Vz_ptr,
        Mp_ptr, Lp_ptr, Accp_ptr,     # (B, Hq, TS) / (B, Hq, TS, D) scratch
        B, S, Hq, Hkv, G, TS, SPLIT_BASE,
        BITS: tl.constexpr, D: tl.constexpr, BLOCK_N: tl.constexpr,
        NSPLIT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // Hq
        h = pid % Hq
        hk = h // G
        sp = tl.program_id(1)
        gsplit = SPLIT_BASE + sp                       # global split index

        # contiguous token range this split owns within the segment
        chunk = (S + NSPLIT - 1) // NSPLIT
        lo = sp * chunk
        hi = tl.minimum(lo + chunk, S)

        offs_d = tl.arange(0, D)
        per = 8 // BITS
        HD2 = D // per
        byte_col = (offs_d // per).to(tl.int64)
        shift = ((offs_d % per) * BITS).to(tl.int32)
        mask_bits = (1 << BITS) - 1

        q = tl.load(Q_ptr + (b * Hq + h) * D + offs_d)
        ksc = tl.load(Ksc_ptr + (b * Hkv + hk) * D + offs_d)
        kz = tl.load(Kz_ptr + (b * Hkv + hk) * D + offs_d)

        m_i = tl.full((), float("-inf"), tl.float32)
        l_i = tl.zeros((), tl.float32)
        acc = tl.zeros((D,), tl.float32)

        for n0 in range(lo, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < hi
            tok = offs_n.to(tl.int64)
            # Packed layout is (B, S, Hkv, D) (token-major); see kv_cache.py
            # _quantize_block for why a Phase-3 (B,Hkv,S,D) reorder was tried
            # and reverted (no coalescing gain, real-model gate regression).
            row_byte = ((b.to(tl.int64) * S + tok) * Hkv + hk) * HD2
            addr = row_byte[:, None] + byte_col[None, :]

            rawk = tl.load(Kc_ptr + addr, mask=mask_n[:, None], other=0).to(tl.int32)
            codek = (rawk >> shift[None, :]) & mask_bits
            kf = codek.to(tl.float32) * ksc[None, :] + kz[None, :]
            scores = tl.sum(q[None, :] * kf, axis=1)
            scores = tl.where(mask_n, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            p = tl.exp(scores - m_new)
            p = tl.where(mask_n, p, 0.0)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=0)

            rawv = tl.load(Vc_ptr + addr, mask=mask_n[:, None], other=0).to(tl.int32)
            codev = (rawv >> shift[None, :]) & mask_bits
            vsc = tl.load(Vsc_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vz = tl.load(Vz_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vf = codev.to(tl.float32) * vsc[:, None] + vz[:, None]
            acc = acc * alpha + tl.sum(p[:, None] * vf, axis=0)
            m_i = m_new

        base = (b * Hq + h) * TS + gsplit
        tl.store(Mp_ptr + base, m_i)
        tl.store(Lp_ptr + base, l_i)
        tl.store(Accp_ptr + base * D + offs_d, acc)


if _HAS_TRITON:

    # -- 4-bit-optimized split-K kernel (coalesced/vectorized packed loads) --
    #
    # The generic kernel above addresses the packed bytes with
    #   byte_col = offs_d // per   ->  0,0,1,1,2,2,...  (repeats, NOT unit-stride)
    # so Triton cannot prove the D axis contiguous and falls back to per-byte
    # ld.global.b8 scalar gathers (verified via PTX: 256 b8 loads at the selected
    # BLOCK_N=32/num_warps=8 config), with each thread striding across tokens ->
    # 31.84 sectors/request (~one 32-B sector per thread, ~8x DRAM waste).
    #
    # This version loads HD2 = D/2 CONTIGUOUS bytes per (token,head) with a
    # unit-stride byte axis (exactly the FlashAttention K-load pattern), then
    # unpacks the two nibbles per byte into even (d=2j) / odd (d=2j+1) planes.
    # The dot product and P@V accumulation are split even/odd accordingly
    # (numerically identical: same terms, summed in even-then-odd order). q,
    # kscale, kzero are pre-split even/odd once per program.
    @_autotune_decorator_v4
    @triton.jit
    def _fused_seg_splitk_kernel_v4(
        Q_ptr, Kc_ptr, Vc_ptr,
        Ksc_ptr, Kz_ptr, Vsc_ptr, Vz_ptr,
        Mp_ptr, Lp_ptr, Accp_ptr,
        B, S, Hq, Hkv, G, TS, SPLIT_BASE,
        D: tl.constexpr, BLOCK_N: tl.constexpr, NSPLIT: tl.constexpr,
    ):
        pid = tl.program_id(0)
        b = pid // Hq
        h = pid % Hq
        hk = h // G
        sp = tl.program_id(1)
        gsplit = SPLIT_BASE + sp

        chunk = (S + NSPLIT - 1) // NSPLIT
        lo = sp * chunk
        hi = tl.minimum(lo + chunk, S)

        HD2: tl.constexpr = D // 2
        byte_offs = tl.arange(0, HD2)                  # 0..HD2-1, UNIT STRIDE

        qbase = (b * Hq + h) * D
        kbase = (b * Hkv + hk) * D
        q_lo = tl.load(Q_ptr + qbase + 2 * byte_offs)         # d = 2j (even)
        q_hi = tl.load(Q_ptr + qbase + 2 * byte_offs + 1)     # d = 2j+1 (odd)
        ksc_lo = tl.load(Ksc_ptr + kbase + 2 * byte_offs)
        ksc_hi = tl.load(Ksc_ptr + kbase + 2 * byte_offs + 1)
        kz_lo = tl.load(Kz_ptr + kbase + 2 * byte_offs)
        kz_hi = tl.load(Kz_ptr + kbase + 2 * byte_offs + 1)

        m_i = tl.full((), float("-inf"), tl.float32)
        l_i = tl.zeros((), tl.float32)
        acc_lo = tl.zeros((HD2,), tl.float32)
        acc_hi = tl.zeros((HD2,), tl.float32)

        for n0 in range(lo, hi, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < hi
            tok = offs_n.to(tl.int64)
            # (B, S, Hkv, D) token-major: HD2 contiguous bytes per (token, head)
            row_byte = ((b.to(tl.int64) * S + tok) * Hkv + hk) * HD2   # (BLOCK_N,)
            addr = row_byte[:, None] + byte_offs[None, :]              # (BLOCK_N, HD2) contiguous

            rawk = tl.load(Kc_ptr + addr, mask=mask_n[:, None], other=0).to(tl.int32)
            klo = (rawk & 0xF).to(tl.float32)
            khi = ((rawk >> 4) & 0xF).to(tl.float32)
            kf_lo = klo * ksc_lo[None, :] + kz_lo[None, :]
            kf_hi = khi * ksc_hi[None, :] + kz_hi[None, :]
            scores = tl.sum(q_lo[None, :] * kf_lo + q_hi[None, :] * kf_hi, axis=1)
            scores = tl.where(mask_n, scores, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(scores, axis=0))
            pe = tl.exp(scores - m_new)
            pe = tl.where(mask_n, pe, 0.0)
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(pe, axis=0)

            rawv = tl.load(Vc_ptr + addr, mask=mask_n[:, None], other=0).to(tl.int32)
            vlo = (rawv & 0xF).to(tl.float32)
            vhi = ((rawv >> 4) & 0xF).to(tl.float32)
            vsc = tl.load(Vsc_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vz = tl.load(Vz_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vf_lo = vlo * vsc[:, None] + vz[:, None]
            vf_hi = vhi * vsc[:, None] + vz[:, None]
            acc_lo = acc_lo * alpha + tl.sum(pe[:, None] * vf_lo, axis=0)
            acc_hi = acc_hi * alpha + tl.sum(pe[:, None] * vf_hi, axis=0)
            m_i = m_new

        base = (b * Hq + h) * TS + gsplit
        tl.store(Mp_ptr + base, m_i)
        tl.store(Lp_ptr + base, l_i)
        tl.store(Accp_ptr + base * D + 2 * byte_offs, acc_lo)       # even d
        tl.store(Accp_ptr + base * D + 2 * byte_offs + 1, acc_hi)   # odd d


if _HAS_TRITON:

    @triton.jit
    def _splitk_merge_kernel(
        Mp_ptr, Lp_ptr, Accp_ptr,   # (B, Hq, TS) / (B, Hq, TS, D)
        Mo_ptr, Lo_ptr, Acco_ptr,    # (B, Hq) / (B, Hq) / (B, Hq, D) -- UNnormalized
        TS, D: tl.constexpr, BLOCK_TS: tl.constexpr,
    ):
        """Phase 4: the split-K log-sum-exp merge (amax, exp, mul, sum x2) was
        five separate PyTorch ops on the (B, Hq, TS[, D]) scratch -- Nsight/
        torch.profiler (Phase 1) showed this costs ~2.35ms of real GPU compute
        but ~27ms/step of CPU dispatch overhead across ~5 small kernel launches
        per attention layer, i.e. almost all its wall-clock cost was launch
        overhead, not the (cheap) arithmetic itself. One program per (b, h)
        does the whole reduction over the TS split axis in a single launch.

        Outputs the UNnormalized running (m, l, acc) -- NOT acc/l -- so the
        caller can still fold the fp16 residual chunk into the same online-
        softmax state afterward, exactly as the multi-op version did.
        """
        pid = tl.program_id(0)   # 0 .. B*Hq-1
        base_row = pid * TS

        m = tl.full((), float("-inf"), tl.float32)
        offs_ts0 = tl.arange(0, BLOCK_TS)
        for t0 in range(0, TS, BLOCK_TS):
            offs_ts = t0 + offs_ts0
            mask_ts = offs_ts < TS
            mv = tl.load(Mp_ptr + base_row + offs_ts, mask=mask_ts, other=float("-inf"))
            m = tl.maximum(m, tl.max(mv, axis=0))

        offs_d = tl.arange(0, D)
        acc = tl.zeros((D,), tl.float32)
        l = tl.zeros((), tl.float32)
        for t0 in range(0, TS, BLOCK_TS):
            offs_ts = t0 + offs_ts0
            mask_ts = offs_ts < TS
            mv = tl.load(Mp_ptr + base_row + offs_ts, mask=mask_ts, other=float("-inf"))
            lv = tl.load(Lp_ptr + base_row + offs_ts, mask=mask_ts, other=0.0)
            alpha = tl.exp(mv - m)
            alpha = tl.where(mask_ts, alpha, 0.0)
            l += tl.sum(lv * alpha, axis=0)
            av = tl.load(
                Accp_ptr + (base_row + offs_ts)[:, None] * D + offs_d[None, :],
                mask=mask_ts[:, None], other=0.0)
            acc += tl.sum(av * alpha[:, None], axis=0)

        tl.store(Mo_ptr + pid, m)
        tl.store(Lo_ptr + pid, l)
        tl.store(Acco_ptr + pid * D + offs_d, acc)


def _splitk_merge(Mp, Lp, Accp, B, Hq, TS, D, device):
    """Fused (single-launch) equivalent of the amax/exp/mul/sum merge.
    Returns UNnormalized (m, l, acc) of shape (B,Hq)/(B,Hq)/(B,Hq,D)."""
    m = torch.empty((B, Hq), device=device, dtype=torch.float32)
    l = torch.empty((B, Hq), device=device, dtype=torch.float32)
    acc = torch.empty((B, Hq, D), device=device, dtype=torch.float32)
    block_ts = min(triton.next_power_of_2(max(TS, 1)), 1024)
    _splitk_merge_kernel[(B * Hq,)](Mp, Lp, Accp, m, l, acc, TS, D=D, BLOCK_TS=block_ts)
    return m, l, acc


_H100_SMS = 132


def _choose_nsplit(S, B, Hq):
    """Pick the split-K split count.

    Re-derived (Phase 2, from Nsight Compute occupancy/launch-stats data, not a
    guess) from the old ~256-total-program target, which measured "Waves Per
    SM: 0.69" -- UNDER one full wave, i.e. many SMs got zero blocks. The
    kernel's register-limited capacity ranges 3-12 blocks/SM depending on the
    autotuned num_warps (3 at num_warps=4, 12 at num_warps=1 -- same ~12
    warps/SM theoretical ceiling either way, since that's set by per-thread
    register count, not block/warp shape).

    NOTE: an even larger target (24x SMs, TS~512/layer) was tried and reverted
    -- it raised Waves/SM to 1.29 and isolated-kernel bandwidth only ~7%
    further (174 vs 162 GB/s), but broke the real-model numerical gate on the
    40B (layer 2: 1.22e-4 vs the 1e-4 gate, up from the 3.0e-8 baseline) by
    combining many more independently-rounded split-K partials through the
    log-sum-exp merge. 8x SMs recovers the gate (see TUNING.md) for a small
    performance cost -- accuracy is the hard constraint, not negotiable for a
    ~7% isolated-kernel gain that barely moved end-to-end throughput anyway.
    """
    if S <= 256:
        return 1
    max_by_work = (S + 255) // 256           # >= 256 tokens of work per split
    target_total = 8 * _H100_SMS
    target_ns = max(1, (target_total + B * Hq - 1) // (B * Hq))
    return max(1, min(max_by_work, target_ns))


def _fold_residual_fp32(qf, r, m, l, acc, G):
    """Fold one fp16 residual chunk (B, s, 2, Hkv, D) into (m, l, acc) in fp32.

    Mirrors the online-softmax math of streaming_kv_attention._consume exactly,
    but for a single query (Sq == 1) with state shape (B, Hq, D)/(B, Hq).
    """
    k = r[:, :, 0].float()          # (B, s, Hkv, D)
    v = r[:, :, 1].float()
    if G > 1:
        k = k.repeat_interleave(G, dim=2)
        v = v.repeat_interleave(G, dim=2)
    # qf: (B, Hq, D)
    scores = torch.einsum("bhd,bshd->bhs", qf, k)           # (B, Hq, s)
    blk_max = scores.amax(dim=-1)                           # (B, Hq)
    m_new = torch.maximum(m, blk_max)
    p = torch.exp(scores - m_new.unsqueeze(-1))             # (B, Hq, s)
    alpha = torch.exp(m - m_new)
    l.mul_(alpha).add_(p.sum(dim=-1))
    acc.mul_(alpha.unsqueeze(-1)).add_(torch.einsum("bhs,bshd->bhd", p, v))
    m.copy_(m_new)


@torch.inference_mode()
def fused_decode_attention(
    cache,
    q: torch.Tensor,
    softmax_scale: Optional[float],
    causal: bool,
    key_block: int = 2048,
) -> torch.Tensor:
    """Fused int4-KV online-softmax decode attention.  Drop-in for
    streaming_kv_attention when Sq == 1 and bits == 4/2/8.

    q: (B, 1, Hq, D).  Returns (B, 1, Hq, D) in q.dtype.
    """
    B, Sq, Hq, D = q.shape
    assert Sq == 1, "fused_decode_attention handles decode (Sq==1) only"
    device, dtype = q.device, q.dtype
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    bits = cache.bits
    Hkv = cache.num_heads_kv
    G = Hq // max(Hkv, 1)

    # For the tensor-parallel / layer-sharded 40B, each attention layer's cache,
    # query and scales live on that layer's own GPU.  Triton launches on the
    # *current* CUDA device, so set it to the tensors' device (no-op for 7B).
    dev_ctx = torch.cuda.device(device) if device.type == "cuda" else _nullctx()
    with dev_ctx:
        return _fused_decode_impl(cache, q, qf_scale=scale, bits=bits, Hkv=Hkv,
                                  G=G, B=B, Hq=Hq, D=D, device=device, dtype=dtype,
                                  key_block=key_block)


def _nullctx():
    return _contextlib.nullcontext()


@torch.inference_mode()
def _fused_decode_impl(cache, q, qf_scale, bits, Hkv, G, B, Hq, D, device, dtype,
                       key_block):
    scale = qf_scale
    qf = (q[:, 0].float() * scale).contiguous()             # (B, Hq, D)

    use_triton = _HAS_TRITON and device.type == "cuda" and bits in (2, 4, 8)
    if use_triton:
        segs = cache.segments
        # split-K: give every segment NSPLIT programs per (b, h) -> full occupancy
        nsplits = [_choose_nsplit(int(s["k_shape"][1]), B, Hq) for s in segs]
        TS = sum(nsplits) if segs else 1
        Mp = torch.full((B, Hq, TS), float("-inf"), device=device, dtype=torch.float32)
        Lp = torch.zeros((B, Hq, TS), device=device, dtype=torch.float32)
        Accp = torch.zeros((B, Hq, TS, D), device=device, dtype=torch.float32)
        base = 0
        for seg, ns in zip(segs, nsplits):
            S = int(seg["k_shape"][1])
            ksc, kz, vsc, vz = _seg_scales_f32(seg)
            # BLOCK_N / num_warps / num_stages are autotuned; only algorithmic
            # args are passed explicitly. 4-bit uses the coalesced/vectorized
            # even-odd kernel (`_v4`); 2-/8-bit use the generic kernel.
            extra = {"BLOCK_N": min(key_block, 128), "num_warps": 4} \
                if _DEBUG_NO_AUTOTUNE else {}
            if bits == 4:
                _fused_seg_splitk_kernel_v4[(B * Hq, ns)](
                    qf, seg["k_packed"], seg["v_packed"], ksc, kz, vsc, vz,
                    Mp, Lp, Accp,
                    B, S, Hq, Hkv, G, TS, base,
                    D=D, NSPLIT=ns, **extra)
            else:
                _fused_seg_splitk_kernel[(B * Hq, ns)](
                    qf, seg["k_packed"], seg["v_packed"], ksc, kz, vsc, vz,
                    Mp, Lp, Accp,
                    B, S, Hq, Hkv, G, TS, base,
                    BITS=bits, D=D, NSPLIT=ns, **extra)
            base += ns
        # combine all splits (log-sum-exp merge over the split axis) -- Phase 4:
        # one Triton kernel launch instead of 5 separate PyTorch ops (amax,
        # exp, mul, sum, mul, sum), which Phase 1 showed cost ~27ms/step of
        # CPU dispatch overhead for ~2.35ms of actual GPU compute.
        if segs:
            m, l, acc = _splitk_merge(Mp, Lp, Accp, B, Hq, TS, D, device)
        else:
            m = torch.full((B, Hq), float("-inf"), device=device, dtype=torch.float32)
            l = torch.zeros((B, Hq), device=device, dtype=torch.float32)
            acc = torch.zeros((B, Hq, D), device=device, dtype=torch.float32)
    else:
        m = torch.full((B, Hq), float("-inf"), device=device, dtype=torch.float32)
        l = torch.zeros((B, Hq), device=device, dtype=torch.float32)
        acc = torch.zeros((B, Hq, D), device=device, dtype=torch.float32)
        for k_blk, v_blk, s0 in _iter_segment_blocks_fp32(cache, key_block):
            _fold_block_fp32(qf, k_blk, v_blk, m, l, acc, G)

    # residual (fp16, full precision) folded in fp32
    for r in cache.residual:
        _fold_residual_fp32(qf, r, m, l, acc, G)

    out = acc / l.clamp_min(1e-20).unsqueeze(-1)            # (B, Hq, D)
    return out.unsqueeze(1).to(dtype)                       # (B, 1, Hq, D)


# ---- pure-torch helpers (fp32 reference / CPU fallback) --------------------

def _iter_segment_blocks_fp32(cache, key_block):
    for seg in cache.segments:
        S = int(seg["k_shape"][1])
        for a in range(0, S, key_block):
            b = min(a + key_block, S)
            yield (cache._seg_dequant_k(seg, a, b, torch.float32),
                   cache._seg_dequant_v(seg, a, b, torch.float32), a)


def _fold_block_fp32(qf, k_blk, v_blk, m, l, acc, G):
    k = k_blk.float()
    v = v_blk.float()
    if G > 1:
        k = k.repeat_interleave(G, dim=2)
        v = v.repeat_interleave(G, dim=2)
    scores = torch.einsum("bhd,bshd->bhs", qf, k)
    blk_max = scores.amax(dim=-1)
    m_new = torch.maximum(m, blk_max)
    p = torch.exp(scores - m_new.unsqueeze(-1))
    alpha = torch.exp(m - m_new)
    l.mul_(alpha).add_(p.sum(dim=-1))
    acc.mul_(alpha.unsqueeze(-1)).add_(torch.einsum("bhs,bshd->bhd", p, v))
    m.copy_(m_new)


@torch.inference_mode()
def reference_attention_fp32(cache, q, softmax_scale, causal=True):
    """Ground-truth: full-precision fp32 attention over the fp32-dequantized
    real cache (non-tiled).  The numerical-gate reference (TurboESM-style)."""
    B, Sq, Hq, D = q.shape
    assert Sq == 1
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    Hkv = cache.num_heads_kv
    G = Hq // max(Hkv, 1)
    kv = cache.dequantize(dtype=torch.float32, device=q.device)   # (B, Stot, 2, Hkv, D)
    k = kv[:, :, 0]
    v = kv[:, :, 1]
    if G > 1:
        k = k.repeat_interleave(G, dim=2)
        v = v.repeat_interleave(G, dim=2)
    qf = q[:, 0].float() * scale                        # (B, Hq, D)
    scores = torch.einsum("bhd,bshd->bhs", qf, k)       # (B, Hq, Stot)
    attn = torch.softmax(scores, dim=-1)
    out = torch.einsum("bhs,bshd->bhd", attn, v)        # (B, Hq, D)
    return out.unsqueeze(1).to(q.dtype)


# ---------------------------------------------------------------------------
# Installer: dispatch decode (Sq==1) to the fused kernel, prefill/extend to the
# reference streaming path.  Storage format is identical to the streaming path.
# ---------------------------------------------------------------------------

def _bf16_flash_fresh_attention(q, kv, softmax_scale, causal: bool):
    """Exact bf16 FlashAttention over the FRESH chunk's own K/V (self-attention).

    Tier-1 prefill optimization: the first prefill chunk (seqlen_offset == 0) has
    no cached prefix, so its attention is entirely self-attention over the chunk
    we are about to store.  There is no reason to quantize→dequant that chunk to
    attend it — we attend it exactly in bf16 (fast, memory O(chunk), numerically
    exact), then quantize it for the int4 cache.  For a prompt that fits one chunk
    (the common generation case) this makes the ENTIRE prefill run at bf16-flash
    speed at *zero* extra memory: the fresh bf16 K/V already exist as `kv` before
    we store them, and the flash-backed SDPA never materializes the S×S scores.

    q:  (B, Sq, Hq, D)   kv: (B, Sq, 2, Hkv, D)  post-rotary, bf16.
    Returns context (B, Sq, Hq, D) matching vortex CrossAttention's layout.
    Uses torch SDPA (flash kernel on H100); GQA/MQA handled by expanding KV heads.
    For so==0 the fresh chunk is the whole sequence, so is_causal matches the vortex
    mask (Sk==Sq → col > row).
    """
    import torch.nn.functional as F
    B, Sq, Hq, D = q.shape
    k = kv[:, :, 0]                              # (B, Sq, Hkv, D)
    v = kv[:, :, 1]
    Hkv = k.shape[2]
    qh = q.permute(0, 2, 1, 3)                   # (B, Hq, Sq, D)
    kh = k.permute(0, 2, 1, 3)
    vh = v.permute(0, 2, 1, 3)
    if Hkv != Hq:                                # GQA/MQA: expand KV heads
        g = Hq // Hkv
        kh = kh.repeat_interleave(g, dim=1)
        vh = vh.repeat_interleave(g, dim=1)
    o = F.scaled_dot_product_attention(qh, kh, vh, is_causal=causal,
                                       scale=softmax_scale)
    return o.permute(0, 2, 1, 3).contiguous().to(q.dtype)   # (B, Sq, Hq, D)


def install_fused_kv_quant(
    model: torch.nn.Module,
    bits: int = 4,
    model_name: str = "evo2_7b",
    key_block: int = 2048,
    verbose: bool = True,
    fused_prefill: bool = False,
) -> int:
    import types
    from vortex.model.attention import MHA
    from turboquant.kv_cache import (
        QuantizedKVCache, streaming_kv_attention, _ATTN_LAYER_IDXS,
    )

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
        module._use_fused_prefill = fused_prefill

        if not hasattr(module, "_orig_update_kvcache_attention"):
            module._orig_update_kvcache_attention = module._update_kvcache_attention

        def _fused_update_kvcache_attention(self, q, kv, inference_params):
            so = inference_params.seqlen_offset
            scale = self.inner_cross_attn.softmax_scale
            causal = self.inner_cross_attn.causal
            # Tier-1 prefill optimization: the first prefill chunk (so == 0) has no
            # cached prefix, so its attention is pure self-attention over the fresh
            # chunk.  Attend it exactly in bf16 via flash-backed torch SDPA (fast,
            # memory O(chunk)) rather than through the int4 dequant path, THEN
            # quantize+store as segment 0.  For a prompt that fits one chunk this
            # makes the whole prefill bf16-fast at no memory cost (the int4 store is
            # unaffected — it is written identically after attending).
            if (so == 0 and q.shape[1] > 1 and _HAS_SDPA
                    and getattr(self, "_use_fused_prefill", False)):
                out = _bf16_flash_fresh_attention(q, kv, scale, causal)
                self._qkv_cache.set_prefill(kv)
                return out
            if so == 0:
                self._qkv_cache.set_prefill(kv)
            else:
                self._qkv_cache.append_decode(kv)
            if q.shape[1] == 1:  # decode -> fused Triton kernel
                return fused_decode_attention(
                    self._qkv_cache, q,
                    softmax_scale=scale, causal=causal,
                    key_block=self._kvq_key_block)
            # prefill / extend -> fused kernel (opt-in) or reference streaming path
            if getattr(self, "_use_fused_prefill", False):
                return fused_prefill_attention(
                    self._qkv_cache, q,
                    softmax_scale=self.inner_cross_attn.softmax_scale,
                    causal=self.inner_cross_attn.causal,
                    key_block=self._kvq_key_block)
            return streaming_kv_attention(
                self._qkv_cache, q,
                softmax_scale=self.inner_cross_attn.softmax_scale,
                causal=self.inner_cross_attn.causal,
                key_block=self._kvq_key_block)

        module._update_kvcache_attention = types.MethodType(
            _fused_update_kvcache_attention, module)
        patched += 1

    if verbose:
        print(f"  [fused-kv-quant] patched {patched} attention MHA modules "
              f"(bits={bits}, key_block={key_block}, model={model_name})")
    return patched


def remove_fused_kv_quant(model: torch.nn.Module) -> int:
    from turboquant.kv_cache import remove_real_kv_quant
    return remove_real_kv_quant(model)


def set_fused_prefill(model: torch.nn.Module, enabled: bool) -> int:
    """Toggle the fused prefill-attention path on all patched MHA modules at runtime.
    Returns the number of modules toggled. Lets a validation compare fused vs eager
    streaming on the SAME loaded model."""
    n = 0
    for m in model.modules():
        if hasattr(m, "_use_fused_prefill"):
            m._use_fused_prefill = enabled
            n += 1
    return n


# ===========================================================================
# Fused int4-KV PREFILL attention (FlashAttention + in-register dequant).
#
# Generalizes the decode kernel above (Sq==1) to Sq==block: query-blocked, tl.dot
# for QK^T and P@V (tensor cores), causal mask by absolute key position, online
# softmax. This is the prefill/extend counterpart the dispatch in
# _fused_update_kvcache_attention currently sends to the eager streaming_kv_attention
# path — swapping that one call to fused_prefill_attention accelerates the
# prefill-dominated workloads (e.g. contact-map extraction) that the decode kernel
# does not touch.
#
# Validated on H100 vs streaming_kv_attention (the eager oracle): parity max_rel_err
# ~3e-4 (fp16 tl.dot). Tuned speedup: ~6.5x single-segment (large key block), 3.1-4.8x
# on multi-segment caches (degrades with segment COUNT — per-segment launch overhead;
# a stress test with tiny 256-tok segments hit 3.15x at 64 segments, but that
# exaggerates launch cost — at real long context the O(context^2) work dominates the
# fixed per-launch cost, so expect closer to the 6.5x ceiling). vs bf16 flash-attn the
# eager path was ~15x slower, so ~6.5x over eager lands the kernel ~2.3x off bf16.
#
# Perf came from the canonical flash accumulate (acc = tl.dot(p, v, acc), 3-arg) which
# is Hopper-safe and UNLOCKED pipelining (num_stages=2) + BLOCK_M=256 — those took it
# from 2.5x to 6.5x. Config note: BLOCK_N=128 or BLOCK_N=128+pipelining still
# hard-aborts on Hopper ("mma->mma layout conversion only supported on Ampere"), so
# BLOCK_N is pinned to 64. Future: batch segments into fewer launches to hold 6.5x
# regardless of segmentation.
# ===========================================================================
if _HAS_TRITON:

    @triton.jit
    def _fused_prefill_seg_kernel(
        Q_ptr, Kc_ptr, Vc_ptr, Ksc_ptr, Kz_ptr, Vsc_ptr, Vz_ptr,
        M_ptr, L_ptr, Acc_ptr,        # (B,Hq,Sq)/(B,Hq,Sq,D) carried online-softmax state
        B, Sq, S, Hq, Hkv, G, SK_MINUS_SQ, SEG_ABS_S0,
        BITS: tl.constexpr, D: tl.constexpr,
        BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        b = pid_bh // Hq
        h = pid_bh % Hq
        hk = h // G
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        per = 8 // BITS
        HD2 = D // per
        byte_col = (offs_d // per)
        shift = ((offs_d % per) * BITS).to(tl.int32)
        mask_bits = (1 << BITS) - 1
        m_ok = offs_m < Sq

        q_off = ((b * Sq + offs_m)[:, None] * Hq + h) * D + offs_d[None, :]
        q = tl.load(Q_ptr + q_off, mask=m_ok[:, None], other=0.0).to(tl.float16)
        ksc = tl.load(Ksc_ptr + (b * Hkv + hk) * D + offs_d)
        kz = tl.load(Kz_ptr + (b * Hkv + hk) * D + offs_d)

        ml_off = (b * Hq + h) * Sq + offs_m
        m_i = tl.load(M_ptr + ml_off, mask=m_ok, other=float("-inf"))
        l_i = tl.load(L_ptr + ml_off, mask=m_ok, other=0.0)
        acc_off = ((b * Hq + h) * Sq + offs_m)[:, None] * D + offs_d[None, :]
        acc = tl.load(Acc_ptr + acc_off, mask=m_ok[:, None], other=0.0)

        q_abs = offs_m + SK_MINUS_SQ
        for n0 in range(0, S, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S
            key_abs = offs_n + SEG_ABS_S0
            row_byte_k = ((b * S + offs_n) * Hkv + hk) * HD2
            addr_k = row_byte_k[None, :] + byte_col[:, None]
            rawk = tl.load(Kc_ptr + addr_k, mask=mask_n[None, :], other=0).to(tl.int32)
            codek = (rawk >> shift[:, None]) & mask_bits
            kf = (codek.to(tl.float32) * ksc[:, None] + kz[:, None]).to(tl.float16)
            scores = tl.dot(q, kf)
            causal = key_abs[None, :] > q_abs[:, None]
            scores = tl.where(mask_n[None, :] & (~causal), scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            row_byte_v = ((b * S + offs_n) * Hkv + hk) * HD2
            addr_v = row_byte_v[:, None] + byte_col[None, :]
            rawv = tl.load(Vc_ptr + addr_v, mask=mask_n[:, None], other=0).to(tl.int32)
            codev = (rawv >> shift[None, :]) & mask_bits
            vsc = tl.load(Vsc_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vz = tl.load(Vz_ptr + (b * S + offs_n) * Hkv + hk, mask=mask_n, other=0.0)
            vf = (codev.to(tl.float32) * vsc[:, None] + vz[:, None]).to(tl.float16)
            # canonical flash accumulate: scale acc, then tl.dot INTO acc (3-arg) keeps
            # acc in the mma-accumulator layout — Hopper-safe and unlocks pipelining
            # (num_stages) + BLOCK_M=256, which took this kernel from 2.5x to ~6.5x.
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), vf, acc)
            m_i = m_new

        tl.store(M_ptr + ml_off, m_i, mask=m_ok)
        tl.store(L_ptr + ml_off, l_i, mask=m_ok)
        tl.store(Acc_ptr + acc_off, acc, mask=m_ok[:, None])


if _HAS_TRITON:

    @triton.jit
    def _fused_prefill_batched_kernel(
        Q_ptr, Kc_ptr, Vc_ptr, Ksc_ptr, Kz_ptr, Vsc_ptr, Vz_ptr, TileSeg_ptr,
        M_ptr, L_ptr, Acc_ptr,                   # un-finalized online-softmax state, B==1
        Sq, S, Hq, Hkv, G, SK_MINUS_SQ,
        BITS: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    ):
        # ONE launch over ALL keys; (m,l,acc) stay in registers across every segment.
        # Per-segment K-scale via a tile->segment lookup (segments 2048-aligned, tiles 64).
        pid_m = tl.program_id(0)
        h = tl.program_id(1)
        hk = h // G
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, D)
        per = 8 // BITS
        HD2 = D // per
        byte_col = (offs_d // per)
        shift = ((offs_d % per) * BITS).to(tl.int32)
        mask_bits = (1 << BITS) - 1
        m_ok = offs_m < Sq
        q = tl.load(Q_ptr + (offs_m[:, None] * Hq + h) * D + offs_d[None, :],
                    mask=m_ok[:, None], other=0.0).to(tl.float16)
        acc = tl.zeros((BLOCK_M, D), dtype=tl.float32)
        m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
        l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
        q_abs = offs_m + SK_MINUS_SQ
        for n0 in range(0, S, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            mask_n = offs_n < S
            seg = tl.load(TileSeg_ptr + (n0 // BLOCK_N))
            ksc = tl.load(Ksc_ptr + (seg * Hkv + hk) * D + offs_d)
            kz = tl.load(Kz_ptr + (seg * Hkv + hk) * D + offs_d)
            # int64: offs_n is int32 from tl.arange, and row_byte reaches
            # total_S * Hkv * HD2. For the 40B (Hkv*HD2 = 4096) that overflows
            # int32 at total_S = 2**31 / 4096 = 524,288 tokens, wrapping negative
            # and faulting with an illegal memory access. The per-segment kernel
            # already promotes (b.to(tl.int64)); this batched contiguous-view
            # kernel is the one that runs at long context, so it needs it more.
            row_byte = (offs_n.to(tl.int64) * Hkv + hk) * HD2
            rawk = tl.load(Kc_ptr + (row_byte[None, :] + byte_col[:, None]),
                           mask=mask_n[None, :], other=0).to(tl.int32)
            kf = (((rawk >> shift[:, None]) & mask_bits).to(tl.float32) * ksc[:, None] + kz[:, None]).to(tl.float16)
            scores = tl.dot(q, kf)
            scores = tl.where(mask_n[None, :] & (offs_n[None, :] <= q_abs[:, None]), scores, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_new[:, None])
            alpha = tl.exp(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, axis=1)
            rawv = tl.load(Vc_ptr + (row_byte[:, None] + byte_col[None, :]),
                           mask=mask_n[:, None], other=0).to(tl.int32)
            vsc = tl.load(Vsc_ptr + offs_n * Hkv + hk, mask=mask_n, other=0.0)
            vz = tl.load(Vz_ptr + offs_n * Hkv + hk, mask=mask_n, other=0.0)
            vf = (((rawv >> shift[None, :]) & mask_bits).to(tl.float32) * vsc[:, None] + vz[:, None]).to(tl.float16)
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.float16), vf, acc)
            m_i = m_new
        ml_off = h * Sq + offs_m                       # B==1: (Hq, Sq)
        tl.store(M_ptr + ml_off, m_i, mask=m_ok)
        tl.store(L_ptr + ml_off, l_i, mask=m_ok)
        tl.store(Acc_ptr + (h * Sq + offs_m)[:, None] * D + offs_d[None, :], acc, mask=m_ok[:, None])


def _get_contiguous_view(cache, block_n):
    """Build+cache the contiguous (single-launch) view of the flushed segments, keyed by
    segment count so it rebuilds only when a segment is added. B==1. Per-block K-scales
    are preserved in a (n_seg, Hkv, D) table — accuracy-neutral. NOTE: this holds a
    CONCATENATED copy of the packed codes (≈ +1x the flushed KV bytes) — affordable at
    ≤524k, but at 1M prefer the per-segment path (batched=False) or a contiguous-primary
    cache. Returns None if any segment isn't BLOCK_N-aligned (falls back to per-seg)."""
    nseg = len(cache.segments)
    cv = getattr(cache, "_contig_view", None)
    if cv is not None and cv["nseg"] == nseg:
        return cv
    kcs, vcs, ksc_t, kz_t, vsc_t, vz_t, tile_seg = [], [], [], [], [], [], []
    for si, seg in enumerate(cache.segments):
        S = int(seg["k_shape"][1])
        if S % block_n != 0 or seg["k_shape"][0] != 1:
            return None
        kcs.append(seg["k_packed"]); vcs.append(seg["v_packed"])
        ksc_t.append(seg["k_scale"].squeeze(1).squeeze(0).float())
        kz_t.append(seg["k_zero"].squeeze(1).squeeze(0).float())
        vsc_t.append(seg["v_scale"].squeeze(-1).squeeze(0).float())
        vz_t.append(seg["v_zero"].squeeze(-1).squeeze(0).float())
        tile_seg += [si] * (S // block_n)
    dev = cache.segments[0]["k_packed"].device
    cv = {"nseg": nseg,
          "kc": torch.cat(kcs).contiguous(), "vc": torch.cat(vcs).contiguous(),
          "ksc": torch.stack(ksc_t).contiguous(), "kz": torch.stack(kz_t).contiguous(),
          "vsc": torch.cat(vsc_t).contiguous(), "vz": torch.cat(vz_t).contiguous(),
          "tile_seg": torch.tensor(tile_seg, dtype=torch.int32, device=dev),
          "total_S": sum(int(s["k_shape"][1]) for s in cache.segments)}
    cache._contig_view = cv
    return cv


@torch.inference_mode()
def fused_prefill_attention(cache, q, softmax_scale, causal: bool = True,
                            key_block: int = 2048, block_m: int = 256,
                            num_warps: int = 8, num_stages: int = 2, batched: bool = True):
    """Fused int4-KV prefill attention — drop-in for streaming_kv_attention (Sq>1).

    Falls back to streaming_kv_attention when Triton/CUDA is unavailable, bits!=4, or
    causal is False. q: (B, Sq, Hq, D).

    batched=True (B==1): ONE kernel launch over all segments — holds ~6.5x over eager
    regardless of segment count (the per-segment path degrades to ~3x at many segments),
    at the cost of a concatenated code copy (see _get_contiguous_view; use batched=False
    at 1M if memory-bound). batched=False: one launch per segment (proven, +1.3GB only).

    Tuned config (H100): BLOCK_M=256, BLOCK_N=64, num_warps=8, num_stages=2.
    """
    import os as _os
    from turboquant.kv_cache import streaming_kv_attention
    B, Sq, Hq, D = q.shape
    if not (_HAS_TRITON and q.device.type == "cuda" and cache.bits == 4 and causal):
        return streaming_kv_attention(cache, q, softmax_scale, causal, key_block)
    if _os.environ.get("TQ_KV_BATCHED", "1") == "0":
        batched = False   # global override: force per-segment (memory-safe, e.g. 1M)

    Hkv = cache.num_heads_kv
    G = Hq // max(Hkv, 1)
    Sk = cache.total_len
    scale = softmax_scale if softmax_scale is not None else 1.0 / math.sqrt(D)
    dev = q.device
    block_n = 64  # pinned (Hopper mma-layout constraint)
    qf = (q.float() * scale).to(torch.float16).contiguous()
    M = torch.full((B, Hq, Sq), float("-inf"), device=dev, dtype=torch.float32)
    L = torch.zeros((B, Hq, Sq), device=dev, dtype=torch.float32)
    Acc = torch.zeros((B, Hq, Sq, D), device=dev, dtype=torch.float32)
    grid = (triton.cdiv(Sq, block_m), B * Hq)
    dev_ctx = torch.cuda.device(dev) if dev.type == "cuda" else _nullctx()

    cv = _get_contiguous_view(cache, block_n) if (batched and B == 1 and cache.segments) else None
    with dev_ctx:
        if cv is not None:
            # ONE launch over all flushed segments -> fills M/L/Acc (un-finalized).
            _fused_prefill_batched_kernel[(triton.cdiv(Sq, block_m), Hq)](
                qf[0], cv["kc"], cv["vc"], cv["ksc"], cv["kz"], cv["vsc"], cv["vz"],
                cv["tile_seg"], M[0], L[0], Acc[0], Sq, cv["total_S"], Hq, Hkv, G, Sk - Sq,
                BITS=cache.bits, D=D, BLOCK_M=block_m, BLOCK_N=block_n,
                num_warps=num_warps, num_stages=num_stages)
            s0 = cv["total_S"]
        else:
            s0 = 0
            for seg in cache.segments:
                S = int(seg["k_shape"][1])
                ksc, kz, vsc, vz = _seg_scales_f32(seg)
                _fused_prefill_seg_kernel[grid](
                    qf, seg["k_packed"], seg["v_packed"], ksc, kz, vsc, vz, M, L, Acc,
                    B, Sq, S, Hq, Hkv, G, Sk - Sq, s0,
                    BITS=cache.bits, D=D, BLOCK_M=block_m, BLOCK_N=block_n,
                    num_warps=num_warps, num_stages=num_stages)
                s0 += S

    # residual (fp16, exact) folded eager in fp32 (same as the decode wrapper)
    for r in cache.residual:
        S = r.shape[1]
        k = r[:, :, 0].float(); v = r[:, :, 1].float()
        if G > 1:
            k = k.repeat_interleave(G, dim=2); v = v.repeat_interleave(G, dim=2)
        scores = torch.einsum("bthd,bshd->bhts", qf.float(), k)
        q_abs = torch.arange(Sq, device=dev).view(Sq, 1) + (Sk - Sq)
        key_abs = torch.arange(s0, s0 + S, device=dev).view(1, S)
        scores = scores.masked_fill((key_abs > q_abs).view(1, 1, Sq, S), float("-inf"))
        m_new = torch.maximum(M, scores.amax(-1))
        p = torch.exp(scores - m_new.unsqueeze(-1))
        alpha = torch.exp(M - m_new)
        L = L * alpha + p.sum(-1)
        Acc = Acc * alpha.unsqueeze(-1) + torch.einsum("bhts,bshd->bhtd", p, v)
        M = m_new
        s0 += S

    out = Acc / L.clamp_min(1e-20).unsqueeze(-1)
    return out.permute(0, 2, 1, 3).contiguous().to(q.dtype)
