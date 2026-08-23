#!/usr/bin/env python3
"""Save and load a pre-quantized int4 Evo 2, so the 40B loads on ONE GPU.

Why this exists. `quantize_model_weights_gpu` compresses an ALREADY-LOADED
model in place, so reaching int4 normally requires the full bf16 model resident
first -- 82 GB for the 40B, which is exactly what does not fit on a single 80 GB
card. Measured: bf16 evo2_40b OOMs on one 80 GB GPU at block 46 of 50. So the
compression is real but unreachable for the user who needs it most.

Building the model CPU-resident and quantizing there does not work either:
TransformerEngine asserts `torch.cuda.is_available()` when a layer is
constructed, and `fixup_fp8_extra_states` asserts its fp8 metadata sits on the
parameters' device while TE forces that metadata onto CUDA. A CPU-resident TE
layer is not supported.

What does work is to never hold more than one bf16 block at a time. vortex builds
blocks one by one and calls `move_to_device` on each, so we intercept that call
and swap the block's Linears for their int4 replacements immediately. Peak GPU
memory becomes (int4 accumulated so far) + (one bf16 block), roughly 40 GB for
the 40B rather than 82 GB.

Two entry points:

    save_int4_checkpoint(model, path)
        Call once, after quantize_model_weights_gpu, on hardware big enough to
        hold bf16 (e.g. 4 GPUs). Writes int4 buffers + the metadata needed to
        rebuild the quantized layers. This is the artifact a user downloads.

    load_int4_model(model_name, path, device="cuda:0")
        Rebuilds the model on one GPU straight into int4, never materialising
        the full bf16 model.
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _quantized_cls():
    from turboquant.quant_linear import QuantizedLinearGpu
    return QuantizedLinearGpu


def save_int4_checkpoint(model: nn.Module, path: str) -> dict:
    """Serialize an already-quantized model. Returns a small stats dict."""
    QL = _quantized_cls()
    meta = {}
    for name, mod in model.named_modules():
        if isinstance(mod, QL):
            meta[name] = {
                "out_features": mod.out_features,
                "in_features": mod.in_features,
                "in_features_padded": mod.in_features_padded,
                "block_size": mod.block_size,
                "bit_width": mod.bit_width,
                "seed": mod.seed,
                "returns_tuple": getattr(mod, "returns_tuple", False),
                "has_bias": hasattr(mod, "bias") and mod.bias is not None,
            }
    # TransformerEngine layers put non-tensor values (fp8 `_extra_state`, often
    # None or a bytes blob) into state_dict, so this cannot blindly call .cpu().
    state = {}
    for k, v in model.state_dict().items():
        state[k] = v.detach().cpu() if torch.is_tensor(v) else v
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"format": "turboquant-int4-v1",
                "quant_meta": meta, "state": state}, path)
    n_bytes = sum(v.numel() * v.element_size()
                  for v in state.values() if torch.is_tensor(v))
    return {"quantized_layers": len(meta), "tensors": len(state),
            "bytes": n_bytes, "path": path}


def _swap_block(block: nn.Module, prefix: str, meta: dict, state: dict,
                device: str) -> int:
    """Replace this block's nn.Linear children with int4 layers from `state`.

    Runs immediately after the block is built, so the bf16 weights it was born
    with are freed before the next block is constructed.
    """
    QL = _quantized_cls()
    swapped = 0
    for mod_name, module in list(block.named_modules()):
        for attr, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{prefix}.{mod_name}.{attr}" if mod_name else f"{prefix}.{attr}"
            m = meta.get(full)
            if m is None:
                continue
            idx = state.get(f"{full}._indices")
            nrm = state.get(f"{full}._norms")
            if idx is None or nrm is None:
                continue
            bias = state.get(f"{full}.bias") if m["has_bias"] else None
            ql = QL(out_features=m["out_features"], in_features=m["in_features"],
                    in_features_padded=m["in_features_padded"],
                    block_size=m["block_size"], bit_width=m["bit_width"],
                    seed=m["seed"], indices=idx.to(device), norms=nrm.to(device),
                    bias_data=None if bias is None else bias.to(device),
                    returns_tuple=m["returns_tuple"])
            setattr(module, attr, ql)
            del child
            swapped += 1
    torch.cuda.empty_cache()
    return swapped


def _build_skeleton(model_name: str):
    """Construct the model from its config alone, downloading nothing.

    `evo2.Evo2(...)` downloads the ORIGINAL bf16 checkpoint before it ever calls
    load_checkpoint -- ~80 GB for the 40B -- and we then discard it, because the
    int4 file already holds every weight. So we replicate the two things Evo2's
    constructor actually needs (the yaml config and the tokenizer) and skip the
    fetch. Without this, tier2 costs a user 80 GB of useless download on top of
    the 33.8 GB they need.
    """
    import pkgutil

    import yaml
    from evo2.utils import CONFIG_MAP
    from vortex.model.model import StripedHyena
    from vortex.model.tokenizer import CharLevelTokenizer
    from vortex.model.utils import dotdict

    cfg = yaml.safe_load(pkgutil.get_data("evo2.models", CONFIG_MAP[model_name]))
    model = StripedHyena(dotdict(cfg, Loader=yaml.FullLoader))
    return model, CharLevelTokenizer(512)


def _warm_te_workspaces() -> None:
    """Allocate Transformer Engine's cuBLAS workspace on every visible device.

    TE keeps one workspace per device and allocates it lazily, on that device's
    first GEMM. Under sharding that first GEMM happens in the middle of a
    forward pass, and creating the cuBLAS handle there fails with

        cuBLAS Error: an internal operation failed   (CreateCublasHandle)

    after the first two devices. Touching each device once here, while nothing
    else is in flight, gets the workspaces built up front. No-op without TE.
    """
    try:
        import transformer_engine.pytorch.module.linear as telinear
    except ImportError:
        return
    for i in range(torch.cuda.device_count()):
        with torch.cuda.device(i):
            try:
                telinear.get_workspace()
            except Exception as e:                      # pragma: no cover
                print(f"[int4-load] WARNING: could not pre-allocate TE "
                      f"workspace on cuda:{i}: {e}", flush=True)


def load_int4_model(model_name: str, path: str, device: str = "cuda:0",
                    verbose: bool = True):
    """Build `model_name` directly in int4, without ever materializing bf16.

    Downloads nothing except the int4 checkpoint itself.

    `device` is either an explicit device string, which pins every block there,
    or the string "auto", which honours the placement vortex computes from the
    visible GPUs and therefore shards the model across all of them.

    "auto" is what makes the checkpoint usable for long context. The weights fit
    on one card, but the KV cache does not: past a few hundred kilobases it
    needs more memory than any single accelerator has. Without sharding here,
    reaching those lengths in int4 meant loading the 82 GB bf16 model first and
    quantizing it in place -- which requires exactly the hardware the
    compression exists to avoid.
    """
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "turboquant-int4-v1":
        raise ValueError(f"not a turboquant int4 checkpoint: {path}")
    meta, state = ck["quant_meta"], ck["state"]

    import vortex.model.model as vm

    orig_move = vm.move_to_device
    counter = {"i": 0, "swapped": 0}
    auto = (device == "auto")

    def patched_move(mod, dev):
        # In "auto" mode `dev` is vortex's own choice for this block; pinning
        # every block to one device is what previously made the checkpoint
        # single-GPU only.
        tgt = dev if auto else device
        orig_move(mod, tgt)
        i = counter["i"]
        counter["i"] += 1
        counter["swapped"] += _swap_block(mod, f"blocks.{i}", meta, state, tgt)

    vm.move_to_device = patched_move
    try:
        model, tok = _build_skeleton(model_name)
    finally:
        vm.move_to_device = orig_move

    # Load everything that was NOT quantized (embeddings, norms, biases, ...).
    #
    # Deliberately NOT model.load_state_dict(): that COPIES into the existing
    # parameter and so keeps the DESTINATION dtype, whereas vortex's own
    # custom_load_state_dict replaces .data and therefore adopts the
    # checkpoint's dtype. The difference is not cosmetic -- a freshly built
    # RMSNorm has an fp32 scale, so copy-semantics leave it fp32, its output
    # promotes to fp32, and the next TransformerEngine Linear dies with
    # "Data types for parameters must match ... input dtype: torch.float32 and
    # 'weight' dtype: torch.bfloat16". Assigning .data reproduces vortex.
    own = dict(model.named_parameters())
    own.update(dict(model.named_buffers()))
    missing = []
    for k, v in state.items():
        if not torch.is_tensor(v):
            continue                      # TE fp8 `_extra_state` and friends
        tgt = own.get(k)
        if tgt is None:
            missing.append(k)
            continue
        # Follow the destination, not one fixed card: under sharding the
        # embeddings, norms and biases live on whichever device their block
        # was placed on. (Identical to the old behaviour when pinned.)
        tgt.data = v.to(tgt.device)
    unexpected = []

    # Second pass, and it is NOT redundant. `_extra_state` (TransformerEngine's
    # fp8 scaling metadata) is neither a parameter nor a buffer -- it moves only
    # through get_extra_state/set_extra_state, which the .data loop above cannot
    # reach. Skipping it leaves 42 `projections` layers running on default fp8
    # scales: the model still produces plausible-looking numbers, which is
    # precisely why this has to be restored rather than assumed harmless.
    # Safe to run after the .data pass: load_state_dict COPIES, so it preserves
    # the dtypes just fixed, and additionally invokes set_extra_state.
    try:
        model.load_state_dict(state, strict=False)
    except Exception as e:                      # pragma: no cover
        print(f"[int4-load] WARNING: extra-state pass failed: {e}", flush=True)

    # TE restores fp8 extra state onto the DEFAULT cuda device, not the device
    # the layer actually lives on, so under sharding every block's fp8 scales
    # and amax history stay on cuda:0 while its weights sit on cuda:1..N. The
    # FP8 `projections` GEMM then dies with
    #     cuBLAS Error: an internal operation failed  (CreateCublasHandle)
    # vortex hits the same problem and ships the repair; it calls this straight
    # after load_state_dict in custom_load_state_dict, which we bypass, so we
    # have to call it ourselves. A no-op when everything is on one device.
    try:
        from vortex.model.utils import fixup_fp8_extra_states
        fixup_fp8_extra_states(model)
    except ImportError:                         # pragma: no cover
        pass
    if hasattr(model, "block_idx_to_device") and not auto:
        model.block_idx_to_device = {k: device for k in model.block_idx_to_device}

    if auto:
        _warm_te_workspaces()
    if verbose:
        print(f"[int4-load] swapped {counter['swapped']} layers, "
              f"{len(missing)} missing / {len(unexpected)} unexpected keys, "
              f"resident {sum(torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count()))/1e9:.1f} GB "
              f"across {torch.cuda.device_count() if auto else 1} device(s)",
              flush=True)
        if missing:
            # Never wave these through on a checkpoint meant for distribution:
            # a tensor that lands nowhere is a silently wrong model.
            print(f"[int4-load] UNMATCHED keys ({len(missing)}), first 8: "
                  f"{missing[:8]}", flush=True)
    return model, tok
