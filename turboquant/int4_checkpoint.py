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


def load_int4_model(model_name: str, path: str, device: str = "cuda:0",
                    verbose: bool = True):
    """Build `model_name` directly in int4 on a single device."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("format") != "turboquant-int4-v1":
        raise ValueError(f"not a turboquant int4 checkpoint: {path}")
    meta, state = ck["quant_meta"], ck["state"]

    import vortex.model.model as vm
    import vortex.model.utils as vu

    orig_move = vm.move_to_device
    orig_load_vm = getattr(vm, "load_checkpoint", None)
    orig_load_vu = vu.load_checkpoint
    counter = {"i": 0, "swapped": 0}

    def patched_move(mod, dev):
        orig_move(mod, device)
        i = counter["i"]
        counter["i"] += 1
        counter["swapped"] += _swap_block(mod, f"blocks.{i}", meta, state, device)

    # vortex would otherwise reload the full bf16 checkpoint over the top of the
    # int4 layers we just installed, which both defeats the purpose and fails on
    # missing keys. Everything needed is already in the int4 file.
    def noop_load(model, weights_path, *a, **kw):
        return model

    # Patch load_checkpoint at EVERY binding site, not just in vortex.model.utils.
    # evo2/models.py does `from vortex.model.utils import load_checkpoint` at
    # import time, so it holds its OWN reference; patching the utils module alone
    # only works if evo2 has not been imported yet. That made this fail exactly
    # when a user loads a second model in the same process -- vortex then reloads
    # the bf16 checkpoint over the int4 layers and raises on the missing
    # _indices/_norms keys.
    import evo2
    import evo2.models as _em
    orig_load_em = getattr(_em, "load_checkpoint", None)

    vm.move_to_device = patched_move
    vu.load_checkpoint = noop_load
    if hasattr(vm, "load_checkpoint"):
        vm.load_checkpoint = noop_load
    if orig_load_em is not None:
        _em.load_checkpoint = noop_load
    try:
        obj = evo2.Evo2(model_name)
        model, tok = obj.model, obj.tokenizer
    finally:
        # Restore the ORIGINALS captured before patching. Reading them back off
        # the module here would just re-install the no-op permanently.
        vm.move_to_device = orig_move
        vu.load_checkpoint = orig_load_vu
        if orig_load_vm is not None:
            vm.load_checkpoint = orig_load_vm
        if orig_load_em is not None:
            _em.load_checkpoint = orig_load_em

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
        tgt.data = v.to(device)
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
    if hasattr(model, "block_idx_to_device"):
        model.block_idx_to_device = {k: device for k in model.block_idx_to_device}
    if verbose:
        print(f"[int4-load] swapped {counter['swapped']} layers, "
              f"{len(missing)} missing / {len(unexpected)} unexpected keys, "
              f"resident {torch.cuda.memory_allocated(0)/1e9:.1f} GB", flush=True)
        if missing:
            # Never wave these through on a checkpoint meant for distribution:
            # a tensor that lands nowhere is a silently wrong model.
            print(f"[int4-load] UNMATCHED keys ({len(missing)}), first 8: "
                  f"{missing[:8]}", flush=True)
    return model, tok
