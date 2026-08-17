"""Codebook construction for PolarQuant.

After random rotation, each coordinate follows Beta(d/2, d/2) on [-1/√d, 1/√d],
which converges to N(0, 1/d) for large d. We use optimal scalar quantizers for this
distribution.

Paper provides closed-form centroids for 1-bit and 2-bit. For higher bit-widths,
we use Lloyd's algorithm on the Gaussian approximation.
"""

import math
import numpy as np


def _norm_ppf(p: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Inverse normal CDF (Acklam rational approximation) — no scipy needed."""
    # Coefficients from Peter Acklam's algorithm (max abs error < 1.15e-9)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00)
    d = ( 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00)

    def _ppf_scalar(x: float) -> float:
        p_low, p_high = 0.02425, 1.0 - 0.02425
        if x < p_low:
            q = math.sqrt(-2.0 * math.log(x))
            return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                   ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)
        elif x <= p_high:
            q = x - 0.5;  r = q * q
            return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / \
                   (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1.0)
        else:
            q = math.sqrt(-2.0 * math.log(1.0 - x))
            return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                    ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1.0)

    result = np.vectorize(_ppf_scalar)(np.asarray(p, dtype=float))
    return scale * result


def optimal_centroids(bit_width: int, d: int) -> np.ndarray:
    """Compute optimal MSE centroids for the post-rotation coordinate distribution.

    Args:
        bit_width: Number of bits per coordinate (1, 2, 3, 4, ...).
        d: Vector dimension (affects centroid scale).

    Returns:
        Sorted array of 2^bit_width centroids.
    """
    n_centroids = 1 << bit_width

    if bit_width == 1:
        c = np.sqrt(2.0 / (np.pi * d))
        return np.array([-c, c])

    if bit_width == 2:
        return np.array([-1.51, -0.453, 0.453, 1.51]) / np.sqrt(d)

    # For b >= 3, use Lloyd's algorithm on N(0, 1/d)
    return _lloyds_gaussian(n_centroids, sigma=1.0 / np.sqrt(d))


def _lloyds_gaussian(n_centroids: int, sigma: float, n_iter: int = 100) -> np.ndarray:
    """Lloyd's algorithm (iterative k-means) for optimal scalar quantization of N(0, sigma²).

    Args:
        n_centroids: Number of quantization levels (2^b).
        sigma: Standard deviation of the Gaussian.
        n_iter: Number of Lloyd iterations.

    Returns:
        Sorted array of optimal centroids.
    """
    # Initialize boundary positions from uniform quantiles
    boundaries = _norm_ppf(
        np.linspace(0, 1, n_centroids + 1)[1:-1], scale=sigma
    )
    centroids = np.zeros(n_centroids)

    # Initial centroids: conditional expectations within each region
    centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
    for i in range(1, n_centroids - 1):
        centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
    centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    for _ in range(n_iter):
        # Update boundaries (midpoints between consecutive centroids)
        boundaries = (centroids[:-1] + centroids[1:]) / 2.0

        # Update centroids (conditional expectations within each region)
        centroids[0] = _gaussian_conditional_expectation(sigma, -np.inf, boundaries[0])
        for i in range(1, n_centroids - 1):
            centroids[i] = _gaussian_conditional_expectation(sigma, boundaries[i - 1], boundaries[i])
        centroids[-1] = _gaussian_conditional_expectation(sigma, boundaries[-1], np.inf)

    return np.sort(centroids)


def _gaussian_conditional_expectation(sigma: float, a: float, b: float) -> float:
    """E[X | a < X < b] where X ~ N(0, sigma²).

    Uses the formula: E[X | a < X < b] = sigma² * (φ(a/σ) - φ(b/σ)) / (Φ(b/σ) - Φ(a/σ))
    where φ is the PDF and Φ is the CDF of standard normal.
    """
    _sqrt2 = math.sqrt(2.0)
    _sqrt2pi = math.sqrt(2.0 * math.pi)

    def _ncdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / _sqrt2))

    def _nsf(x: float) -> float:          # survival = 1 - cdf
        return 0.5 * math.erfc(x / _sqrt2)

    def _npdf(x: float) -> float:
        return math.exp(-0.5 * x * x) / _sqrt2pi

    a_std = a / sigma if math.isfinite(a) else a
    b_std = b / sigma if math.isfinite(b) else b

    # Use sf() for upper tail to avoid CDF cancellation at extreme values
    if not math.isfinite(a_std):
        prob = _ncdf(b_std)
    elif not math.isfinite(b_std):
        prob = _nsf(a_std)
    else:
        prob = _ncdf(b_std) - _ncdf(a_std)

    if prob < 1e-15:
        if math.isfinite(a) and not math.isfinite(b):
            return a + sigma
        elif not math.isfinite(a) and math.isfinite(b):
            return b - sigma
        elif math.isfinite(a) and math.isfinite(b):
            return (a + b) / 2.0
        else:
            return 0.0

    pdf_diff = _npdf(a_std) - _npdf(b_std)
    return sigma * pdf_diff / prob


def nearest_centroid_indices(values: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Find nearest centroid index for each value. Vectorized.

    Args:
        values: Array of values to quantize, shape (...).
        centroids: Sorted centroid array, shape (n_centroids,).

    Returns:
        Integer indices into centroids array, same shape as values.
    """
    # Use searchsorted for sorted centroids — O(n log k) instead of O(n * k)
    # Find the insertion point, then check left and right neighbors
    boundaries = (centroids[:-1] + centroids[1:]) / 2.0
    return np.searchsorted(boundaries, values.ravel()).reshape(values.shape)
