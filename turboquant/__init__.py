"""TurboQuant-Bio: compressed, correct Evo 2 inference.

    from turboquant import load_evo2, score
    model, tok = load_evo2("evo2_40b", tier="tier2",
                           int4_ckpt="evo2_40b_int4.pt")
    print(score(model, tok, "ACGT...", model_name="evo2_40b"))

The heavy entry points are imported lazily so that `import turboquant` stays
cheap and does not require torch/evo2 to be importable for the pure
quantization utilities.
"""

from turboquant.codebook import optimal_centroids
from turboquant.rotation import random_rotation_dense

__all__ = [
    "optimal_centroids",
    "random_rotation_dense",
    "load_evo2",
    "score",
    "EFFECTIVE_CONTEXT",
    "save_int4_checkpoint",
    "load_int4_model",
    "install_block_continuation",
]


def __getattr__(name):
    if name in ("load_evo2", "score", "EFFECTIVE_CONTEXT"):
        from turboquant import api
        return getattr(api, name)
    if name in ("save_int4_checkpoint", "load_int4_model"):
        from turboquant import int4_checkpoint
        return getattr(int4_checkpoint, name)
    if name == "install_block_continuation":
        from turboquant import block_continuation
        return block_continuation.install_block_continuation
    raise AttributeError(f"module 'turboquant' has no attribute {name!r}")
