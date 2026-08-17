"""TurboQuantMSE — MSE-optimal scalar quantization with random rotation.

Algorithm (PolarQuant / TurboQuant-MSE):
  Quantize:
    1. Compute L2 norm of each weight block → save as scale.
    2. Normalize each block to unit sphere.
    3. Apply forward rotation:  W_r = W_norm @ R^T  (row-vector convention).
    4. Scalar-quantize each coordinate to nearest centroid index.

  Dequantize (GPU side in benchmark_evo2_40b_weights._gpu_dequantize_blocks):
    1. Centroid lookup → W_r_approx.
    2. Renormalize W_r_approx to unit sphere.
    3. Inverse rotation: W_n = W_r_approx @ R  ((R^T)^{-1} = R for orthogonal R).
    4. Rescale by saved norms → W_reconstructed.

Seed and block_size must match the values used by _get_quant_tensors in
benchmark_evo2_40b_weights.py so that CPU quantize and GPU dequantize are
compatible.
"""

import numpy as np

from turboquant.codebook import nearest_centroid_indices, optimal_centroids
from turboquant.rotation import random_rotation_dense


class TurboQuantMSE:
    """CPU quantizer for one (block_size, bit_width) configuration.

    Args:
        d:         Block size — number of weights per quantization unit.
        bit_width: Bits per coordinate (e.g. 2, 4, 8).
        seed:      RNG seed; must match the seed used by _get_quant_tensors on
                   the GPU side so rotation matrices are identical.
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42) -> None:
        self.d = d
        self.bit_width = bit_width
        self.seed = seed

        rng = np.random.default_rng(seed)
        # R is (d, d) orthogonal.  Forward rotation uses R^T; inverse uses R.
        self.R = random_rotation_dense(d, rng).astype(np.float32)       # (d, d)
        self.centroids = optimal_centroids(bit_width, d).astype(np.float32)  # (2^b,)

    def quantize(
        self, W_blocks: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantize a batch of weight blocks.

        Args:
            W_blocks: float32 array of shape (n_total, d).

        Returns:
            indices: uint8  array of shape (n_total, d) — centroid index per coord.
            norms:   float32 array of shape (n_total,)  — L2 norm of each block.
        """
        if W_blocks.ndim != 2 or W_blocks.shape[1] != self.d:
            raise ValueError(
                f"Expected shape (n, {self.d}), got {W_blocks.shape}"
            )

        W = W_blocks.astype(np.float32, copy=False)

        # 1. Per-block L2 norms
        norms = np.linalg.norm(W, axis=1)                   # (n_total,)
        safe_norms = np.where(norms > 1e-10, norms, 1.0)

        # 2. Normalize to unit sphere
        W_norm = W / safe_norms[:, np.newaxis]               # (n_total, d)

        # 3. Forward rotation (row-vector convention): W_r = W_norm @ R^T
        W_rotated = W_norm @ self.R.T                        # (n_total, d)

        # 4. Scalar-quantize each coordinate to nearest centroid
        indices = nearest_centroid_indices(W_rotated, self.centroids)  # (n_total, d)

        return indices.astype(np.uint8), norms.astype(np.float32)
