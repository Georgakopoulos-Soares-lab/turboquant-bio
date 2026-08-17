#!/usr/bin/env python3
"""int4 nibble-packed nn.Linear replacement and the in-place model quantizer.

Extracted from benchmarks/benchmark_evo2_40b_weights.py so that the installable
package is self-contained: turboquant.int4_checkpoint and turboquant.api need
QuantizedLinearGpu and quantize_model_weights_gpu, and importing them from a
benchmark script meant the published package could import successfully and then
fail at tier2 load time.

Memory per layer (weight [out_f, in_f], bit_width <= 4):
    bf16       : out_f * in_f * 2 bytes
    compressed : out_f * ceil(in_f/D) * D * 0.5 byte  (nibble-packed indices)
               + out_f * ceil(in_f/D) * 2 bytes       (fp16 norms)
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from turboquant.codebook import optimal_centroids
from turboquant.rotation import random_rotation_dense

GB = 1024 ** 3

def _pack_nibbles(x: torch.Tensor) -> torch.Tensor:
    """Pack (N,) uint8 tensor with values 0-15 into (ceil(N/2),) uint8.

    Low nibble  ← x[0], x[2], x[4], ...
    High nibble ← x[1], x[3], x[5], ...
    Halves storage versus plain uint8.
    """
    flat = x.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, flat.new_zeros(1)])
    return ((flat[0::2] & 0xF) | ((flat[1::2] & 0xF) << 4)).to(torch.uint8)


def _unpack_nibbles(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack (ceil(N/2),) uint8 → (N,) uint8, reversing _pack_nibbles."""
    low  = packed & 0xF
    high = (packed >> 4) & 0xF
    return torch.stack([low, high], dim=1).reshape(-1)[:n].to(torch.uint8)


# ---------------------------------------------------------------------------
# Per-GPU cache: rotation matrix + centroids
# ---------------------------------------------------------------------------


_QUANT_CACHE: dict = {}


def _get_quant_tensors(
    block_size: int,
    bit_width: int,
    device: torch.device,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, None]:
    """Return (R, centroids, None) as GPU float32 tensors (cached).

    R         : (D, D) — Haar-distributed orthogonal rotation matrix
    centroids : (n_centroids,) — optimal MSE centroids for N(0, 1/D)
    """
    key = (block_size, bit_width, str(device), seed)
    if key not in _QUANT_CACHE:
        rng = np.random.default_rng(seed)
        R_np = random_rotation_dense(block_size, rng).astype(np.float32)
        c_np = optimal_centroids(bit_width, block_size).astype(np.float32)
        R = torch.from_numpy(R_np).to(device)
        c = torch.from_numpy(c_np).to(device)
        _QUANT_CACHE[key] = (R, c, None)
    return _QUANT_CACHE[key]


# ---------------------------------------------------------------------------
# CPU-based PolarQuant quantize  (one-time, reliable across all GPU devices)
# GPU-based dequantize           (hot path during inference)
# ---------------------------------------------------------------------------

# Cache TurboQuantMSE instances per (block_size, bit_width, seed) — CPU objects
_CPU_QUANT_CACHE: dict = {}


def _get_cpu_quantizer(block_size: int, bit_width: int, seed: int = 42):
    """Return a cached TurboQuantMSE instance for CPU quantization."""
    from turboquant.turboquant import TurboQuantMSE
    key = (block_size, bit_width, seed)
    if key not in _CPU_QUANT_CACHE:
        _CPU_QUANT_CACHE[key] = TurboQuantMSE(d=block_size, bit_width=bit_width, seed=seed)
    return _CPU_QUANT_CACHE[key]


def _cpu_quantize_blocks(
    W_blocks_np: np.ndarray,   # (n_total, D) float32, CPU numpy
    block_size: int,
    bit_width: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight blocks on CPU, store results on target GPU device.

    Uses the same rotation matrix and centroids as _gpu_dequantize_blocks
    (both derive from random_rotation_dense + optimal_centroids with the same
    seed), so CPU quantize + GPU dequantize are fully compatible.

    Returns:
        indices : (n_total, D) uint8   on device
        norms   : (n_total,)  float16  on device
    """
    q = _get_cpu_quantizer(block_size, bit_width, seed)
    indices_np, norms_np = q.quantize(W_blocks_np)     # numpy, CPU

    indices = torch.from_numpy(indices_np.astype(np.uint8)).to(device)
    norms   = torch.from_numpy(norms_np.astype(np.float16)).to(device)
    return indices, norms


def _gpu_dequantize_blocks(
    indices:   torch.Tensor,  # (n_total, D) uint8 on GPU
    norms:     torch.Tensor,  # (n_total,)  float16 on GPU
    R:         torch.Tensor,  # (D, D) on GPU  (orthogonal: inverse = R^T)
    centroids: torch.Tensor,  # (n_centroids,) on GPU
) -> torch.Tensor:
    """Reconstruct (n_total, D) float32 weight blocks from compressed storage."""
    # Centroid lookup → rotated unit vectors
    W_r    = centroids[indices.long()]                  # (n_total, D) float32

    # Norm correction: renormalize reconstructed vectors to unit sphere
    rn     = W_r.norm(dim=1, keepdim=True).clamp(min=1e-10)
    W_r    = W_r / rn

    # Inverse rotation:  W_n = W_r @ R   (R^{-1} = R^T, row-vector convention)
    W_n    = W_r @ R                                    # (n_total, D)

    # Rescale by original norms
    return W_n * norms.float().unsqueeze(1)             # (n_total, D)


# ---------------------------------------------------------------------------
# QuantizedLinearGpu — nn.Linear replacement with GPU-resident compression
# ---------------------------------------------------------------------------


class QuantizedLinearGpu(nn.Module):
    """nn.Linear replacement that stores weights as nibble-packed int4 + fp16 on GPU.

    All dequantization math runs on GPU (torch matmul + centroid lookup) so
    there is no CPU bounce during forward passes.

    Memory (per layer, weight shape [out_f, in_f], bit_width ≤ 4):
        Original bf16  : out_f × in_f × 2 bytes
        Compressed     : out_f × ceil(in_f/D) × D × 0.5 byte  (nibble-packed indices)
                       + out_f × ceil(in_f/D) × 2 bytes        (fp16 norms)
        ≈ bits/16 fraction of original bf16 memory (true 4-bit storage)
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        in_features_padded: int,
        block_size: int,
        bit_width: int,
        seed: int,
        indices: torch.Tensor,    # (n_total, D) uint8, on target GPU
        norms: torch.Tensor,      # (n_total,)  fp16,  on target GPU
        bias_data,                # Tensor or None
        returns_tuple: bool = False,
    ):
        super().__init__()
        self.out_features       = out_features
        self.in_features        = in_features
        self.in_features_padded = in_features_padded
        self.block_size         = block_size
        self.bit_width          = bit_width
        self.seed               = seed
        self.returns_tuple      = returns_tuple

        self.register_buffer("_indices", indices)
        self.register_buffer("_norms",   norms)

        if bias_data is not None:
            self.register_buffer("bias", bias_data.clone())
        else:
            self.bias = None

    def _dequant_weight(self) -> torch.Tensor:
        """Reconstruct the weight matrix entirely on GPU."""
        device = self._indices.device
        R, centroids, _ = _get_quant_tensors(
            self.block_size, self.bit_width, device, self.seed
        )
        if self.bit_width <= 4:
            # Unpack nibbles → (n_total, D) uint8
            n_total    = self.out_features * (self.in_features_padded // self.block_size)
            n_elements = n_total * self.block_size
            idx = _unpack_nibbles(self._indices, n_elements).reshape(n_total, self.block_size)
        else:
            idx = self._indices
        blocks = _gpu_dequantize_blocks(idx, self._norms, R, centroids)
        # blocks : (out_f * n_b, D)  → (out_f, in_f_padded) → (out_f, in_f)
        W = blocks.reshape(self.out_features, self.in_features_padded)
        return W[:, : self.in_features]

    def forward(self, x: torch.Tensor):
        W   = self._dequant_weight().to(dtype=x.dtype)
        out = F.linear(x, W, self.bias)
        if self.returns_tuple:
            return out, None
        return out

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"bits={self.bit_width}, block={self.block_size}"
        )

    def compressed_bytes(self) -> int:
        idx  = self._indices.numel() * 1    # uint8 = 1 byte
        nrm  = self._norms.numel()   * 2    # fp16  = 2 bytes
        bias = (
            self.bias.numel() * self.bias.element_size()
            if self.bias is not None else 0
        )
        return idx + nrm + bias

    def original_bytes(self, dtype_bytes: int = 2) -> int:
        """Bytes the original bf16 weight would occupy (default bf16 = 2 B)."""
        return self.out_features * self.in_features * dtype_bytes

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bit_width: int,
        block_size: int = 128,
        seed: int = 42,
    ) -> "QuantizedLinearGpu":
        """Quantize an nn.Linear layer on its current GPU device, in-place safe."""
        W               = linear.weight.data           # (out_f, in_f) on GPU
        out_f, in_f     = W.shape
        device          = W.device

        # Pull weight to CPU numpy for reliable quantization (avoids CUDA
        # cross-device context bugs when the model spans multiple GPUs).
        # NOTE: .cpu() BEFORE .float() to avoid allocating a fp32 copy on GPU
        # (critical when running on a single GPU with tight VRAM headroom).
        W_np = W.cpu().float().numpy()           # GPU→CPU, then bf16→fp32

        # Free the GPU bf16 tensor NOW — before pushing uint8 indices back.
        # Without this, both the old bf16 weight AND the new uint8 tensor must
        # fit simultaneously, causing OOM on a fully-loaded single H100.
        del W
        linear.weight.data = torch.empty(0, device=device, dtype=torch.bfloat16)
        torch.cuda.empty_cache()

        pad = (-in_f) % block_size
        if pad:
            W_np = np.pad(W_np, ((0, 0), (0, pad)))
        in_f_padded = W_np.shape[1]

        n_b         = in_f_padded // block_size
        W_blocks_np = W_np.reshape(out_f * n_b, block_size)  # (n_total, D)

        # Quantize on CPU; results are immediately moved to the original device
        # so VRAM savings are real.  _get_quant_tensors (for forward/dequantize)
        # uses the same seed → same rotation + centroids → compatible.
        indices, norms = _cpu_quantize_blocks(
            W_blocks_np, block_size, bit_width, seed, device
        )

        # For 4-bit (and below), pack two indices per byte to achieve true
        # half-precision storage (0.5 byte/weight instead of 1 byte/weight).
        if bit_width <= 4:
            indices = _pack_nibbles(indices.reshape(-1)).to(device)

        # Detect TELinear-style (out, None) return contract used by vortex
        returns_tuple = (
            getattr(linear, "te_return_bias", None) is not None
            or getattr(linear, "skip_bias_add", False)
        )

        return cls(
            out_features       = out_f,
            in_features        = in_f,
            in_features_padded = in_f_padded,
            block_size         = block_size,
            bit_width          = bit_width,
            seed               = seed,
            indices            = indices,
            norms              = norms,
            bias_data          = linear.bias.data if linear.bias is not None else None,
            returns_tuple      = returns_tuple,
        )


# ---------------------------------------------------------------------------
# RTN (Round-To-Nearest) int4 — naive baseline for comparison with TurboQuant
# ---------------------------------------------------------------------------


def quantize_model_weights_gpu(
    model: nn.Module,
    bit_width: int = 4,
    block_size: int = 128,
    seed: int = 42,
    min_params: int = 1024,
    verbose: bool = True,
) -> dict:
    """Replace every large nn.Linear in model with QuantizedLinearGpu in-place.

    Returns a stats dict with memory accounting.
    """
    # Collect all (parent, attr, child) — do NOT modify the tree while iterating
    # Skip any nn.Linear whose weight is shared with an nn.Embedding (weight tying).
    # Destroying a tied weight (e.g. lm_head.decoder in ESM-2) zeroes the embedding
    # table in-place and crashes the subsequent forward pass.
    embedding_weight_ids = {
        id(m.weight) for m in model.modules() if isinstance(m, nn.Embedding)
    }
    to_replace = []
    for name, module in model.named_modules():
        for attr, child in module.named_children():
            if isinstance(child, nn.Linear):
                n_params = child.out_features * child.in_features
                if n_params >= min_params:
                    if id(child.weight) in embedding_weight_ids:
                        if verbose:
                            full_name = f"{name}.{attr}" if name else attr
                            print(f"  [skip] {full_name} — weight tied to embedding")
                        continue
                    full_name = f"{name}.{attr}" if name else attr
                    to_replace.append((full_name, module, attr, child))

    total            = len(to_replace)
    native_bytes     = 0
    compressed_bytes = 0
    t_start          = time.time()

    for i, (full_name, parent, attr, linear) in enumerate(to_replace, 1):
        orig_b = linear.out_features * linear.in_features * linear.weight.element_size()
        native_bytes += orig_b

        ql = QuantizedLinearGpu.from_linear(
            linear, bit_width=bit_width, block_size=block_size, seed=seed
        )
        setattr(parent, attr, ql)

        # Explicitly free the original weight tensor immediately
        del linear
        torch.cuda.empty_cache()

        compressed_bytes += ql.compressed_bytes()
        elapsed = time.time() - t_start

        if verbose:
            print(
                f"  [{i:>3}/{total}] {full_name:<60} "
                f"{ql.out_features}×{ql.in_features:>6}  "
                f"{orig_b / 1e6:>6.1f} MB → {ql.compressed_bytes() / 1e6:>5.1f} MB  "
                f"[{elapsed:>6.0f}s]",
                flush=True,
            )

    savings = native_bytes - compressed_bytes
    ratio   = native_bytes / max(compressed_bytes, 1)

    return {
        "replaced":         total,
        "native_bytes":     native_bytes,
        "compressed_bytes": compressed_bytes,
        "savings_gb":       savings / GB,
        "ratio":            ratio,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
