"""Workaround for a pre-existing bug in transformer_engine's global FP8
delayed-scaling amax buffer, triggered by vortex's multi-GPU layer sharding.

Not part of the KV-cache dequant+attention fusion work. vortex assigns the
40B's layers to devices with plain `.to(cuda:N)` inside ONE process (not
torch.distributed) — see StripedHyena.__init__ "Assigned layer_idx=X to
device=cuda:Y". Every layer's `te.fp8_autocast(..., fp8_recipe=self.fp8_recipe)`
shares the SAME recipe object, so TE's FP8GlobalStateManager buffers all
layers' amax tensors under one key regardless of which GPU produced them.
`reduce_and_update_fp8_tensors` then does `torch.cat(amax_buffer)`, which
crashes with "Expected all tensors to be on the same device" as soon as two
buffered tensors come from different GPUs — reproducible on this node during
plain forward passes (prefill and decode), independent of the KV kernel.

This patches the buffer merge to move every tensor to a common device first
(the same thing a real distributed all-reduce would do via NCCL), which is
what TE's *intent* already is for the multi-GPU case — it just assumes
torch.distributed with one device per rank, which vortex's single-process
layer-sharding doesn't fit.
"""
from __future__ import annotations


def patch_te_multi_gpu_amax_reduce() -> bool:
    """Idempotent. Returns True if the patch was applied (TE import succeeded)."""
    try:
        from transformer_engine.pytorch.fp8 import FP8GlobalStateManager
    except ImportError:
        return False

    if getattr(FP8GlobalStateManager, "_turboquant_multigpu_patched", False):
        return True

    orig = FP8GlobalStateManager.reduce_and_update_fp8_tensors.__func__

    def patched(cls, forward=True):
        for buffer_key, amax_buffer in cls.global_amax_buffer.items():
            if len(amax_buffer) > 1:
                dev0 = amax_buffer[0].device
                if any(t.device != dev0 for t in amax_buffer):
                    cls.global_amax_buffer[buffer_key] = [t.to(dev0) for t in amax_buffer]
        return orig(cls, forward=forward)

    FP8GlobalStateManager.reduce_and_update_fp8_tensors = classmethod(patched)
    FP8GlobalStateManager._turboquant_multigpu_patched = True
    return True
