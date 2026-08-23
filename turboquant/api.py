#!/usr/bin/env python3
"""TurboQuant-Bio: one entry point for running Evo 2 compressed and correct.

    from turboquant import load_evo2, score

    model, tok = load_evo2("evo2_40b", tier="tier2")     # one 80 GB GPU
    logl = score(model, tok, "ACGT...")                  # mean log-likelihood

Why this wrapper exists rather than "call the three installers yourself":

* `install_block_continuation` is NOT optional. Without it, any sequence longer
  than one chunk is silently wrong -- not an approximation, essentially
  uncorrelated with the truth (see README_chunk_prefill.md). It is applied here
  unconditionally so a user cannot forget it.
* The order of operations matters. A full-precision reference must be computed
  BEFORE a KV tier is installed, because the fused attention does not accept
  `key_padding_mask` and the stateless path then raises TypeError.
* Chunk size, not the KV cache, drives peak memory: the Hyena modal-FFT buffer
  is ~8.6 GB at chunk 4096 for the 40B. We pick a chunk that fits the model and
  device rather than letting the user discover this via an OOM.

Tiers:
    baseline : bf16 weights + bf16 KV        (stock, no compression)
    tier1    : bf16 weights + int4 KV        (same weights, 4x smaller KV cache)
    tier2    : int4 weights + int4 KV        (max compression; needs an int4
                                              checkpoint, see int4_checkpoint.py)
"""
from __future__ import annotations

import os
import sys
import warnings

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Evo 2's benefit from context peaks near here and DECLINES beyond it (8 loci,
# p = 0.0078; README_chunk_prefill.md §10). Feeding more is not merely wasteful,
# it measurably hurts.
EFFECTIVE_CONTEXT = 32_768

# Chunk defaults chosen so peak memory fits a single 80 GB card. The 40B needs
# the smaller value because the modal-FFT buffer scales with chunk. Fidelity is
# independent of chunk size, so this costs nothing.
_DEFAULT_CHUNK = {"evo2_40b": 1024, "evo2_7b": 4096}

_TIERS = ("baseline", "tier1", "tier2")

# Pre-quantized int4 checkpoints on the Hub. tier2 fetches from here when no
# explicit int4_ckpt path is given, so a user never has to manage 34 GB by hand.
INT4_REPO = "michalakis99/turboquant-evo2-int4"
INT4_FILES = {"evo2_40b": "evo2_40b_int4.pt", "evo2_7b": "evo2_7b_int4.pt"}


def fetch_int4_checkpoint(model_name: str, repo: str = INT4_REPO) -> str:
    """Download (and cache) the int4 checkpoint for `model_name`.

    Uses the standard HF cache, so it downloads once. Returns a local path.
    """
    fname = INT4_FILES.get(model_name)
    if fname is None:
        raise ValueError(f"no int4 checkpoint published for {model_name!r}; "
                         f"build one with "
                         f"experiments/single_gpu/make_int4_checkpoint.py")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=repo, filename=fname)


def default_chunk(model_name: str) -> int:
    for k, v in _DEFAULT_CHUNK.items():
        if k in model_name:
            return v
    return 1024


def load_evo2(model_name: str = "evo2_7b", tier: str = "baseline",
              device: str = "cuda:0", int4_ckpt: str | None = None,
              verbose: bool = True):
    """Load Evo 2 at the requested compression tier, correct by construction.

    model_name : "evo2_7b" or "evo2_40b"
    tier       : "baseline" | "tier1" | "tier2"
    device     : where tier2 puts the int4 weights. An explicit device string
                 ("cuda:0") pins the whole model there; "auto" shards it across
                 every visible GPU, which is what long context needs -- the
                 weights fit on one card but the KV cache does not.
    int4_ckpt  : path to a pre-quantized int4 checkpoint (tier2). If omitted,
                 the published checkpoint is downloaded and cached from
                 INT4_REPO. Only if that also fails does tier2 fall back to
                 quantizing in place, which needs the full bf16 model resident
                 FIRST (~82 GB for the 40B, i.e. not a single card). Build your
                 own with experiments/single_gpu/make_int4_checkpoint.py.

    Returns (model, tokenizer).
    """
    if tier not in _TIERS:
        raise ValueError(f"tier must be one of {_TIERS}, got {tier!r}")

    # Evo 2 is intentionally NOT a dependency of this package (vortex,
    # transformer_engine and flash-attn are built against a specific CUDA
    # version and GPU, so installing them automatically breaks more
    # environments than it fixes). Fail with an actionable message rather than
    # a bare ImportError from three frames down.
    try:
        import evo2  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "TurboQuant-Bio needs a working Evo 2 install, which it does not "
            "provide. Install Evo 2 first (https://github.com/ArcInstitute/evo2), "
            "check `python -c \"from evo2 import Evo2\"` succeeds, then retry."
        ) from e

    if "40b" in model_name:
        from turboquant._te_multigpu_patch import patch_te_multi_gpu_amax_reduce
        patch_te_multi_gpu_amax_reduce()

    # tier2 needs int4 weights. Prefer an explicit path; otherwise fetch the
    # published checkpoint. Quantizing in place is the last resort because it
    # requires the full bf16 model resident FIRST (~82 GB for the 40B).
    if tier == "tier2" and not int4_ckpt:
        try:
            int4_ckpt = fetch_int4_checkpoint(model_name)
            if verbose:
                print(f"[turboquant] using int4 checkpoint {int4_ckpt}",
                      flush=True)
        except Exception as e:
            warnings.warn(
                f"could not fetch a published int4 checkpoint ({e}); falling "
                f"back to in-place quantization, which needs the full bf16 "
                f"model on GPU first.", stacklevel=2)

    if tier == "tier2" and int4_ckpt:
        from turboquant.int4_checkpoint import load_int4_model
        model, tok = load_int4_model(model_name, int4_ckpt, device=device,
                                     verbose=verbose)
        from turboquant.fused_weight_dequant import install_fast_weight_forward
        install_fast_weight_forward(model)
    else:
        from evo2 import Evo2
        if tier == "tier2" and not int4_ckpt:
            warnings.warn(
                "tier2 without int4_ckpt quantizes in place, which needs the "
                "full bf16 model resident on GPU first (~82 GB for the 40B). "
                "Build a checkpoint with make_int4_checkpoint.py to load on a "
                "single card.", stacklevel=2)
        obj = Evo2(model_name)
        model, tok = obj.model, obj.tokenizer
        if tier == "tier2":
            from turboquant.quant_linear import quantize_model_weights_gpu
            from turboquant.fused_weight_dequant import install_fast_weight_forward
            quantize_model_weights_gpu(model, bit_width=4, block_size=128,
                                       verbose=False)
            install_fast_weight_forward(model)

    if tier in ("tier1", "tier2"):
        from turboquant.fused_kv_attention import install_fused_kv_quant
        install_fused_kv_quant(model, bits=4, model_name=model_name,
                               fused_prefill=True, verbose=False)

    # Unconditional: correctness is not opt-in.
    from turboquant.block_continuation import install_block_continuation
    install_block_continuation(model, verbose=False)

    if verbose:
        alloc = torch.cuda.memory_allocated(0) / 1e9
        print(f"[turboquant] {model_name} tier={tier} loaded, "
              f"{alloc:.1f} GB on cuda:0, block-continuation active", flush=True)
    return model, tok


@torch.inference_mode()
def score(model, tok, sequence: str, chunk: int | None = None,
          model_name: str = "evo2_40b", return_per_token: bool = False):
    """Mean log-likelihood per token of `sequence` (higher = better).

    Uses chunked prefill with block continuation, so arbitrarily long sequences
    are both correct and fast. Sequences longer than the model's effective
    context are accepted but warned about.
    """
    if chunk is None:
        chunk = default_chunk(model_name)
    if len(sequence) > EFFECTIVE_CONTEXT * 1.5:
        warnings.warn(
            f"sequence is {len(sequence)} bp but Evo 2's measured effective "
            f"context is ~{EFFECTIVE_CONTEXT} bp; beyond that, extra context "
            f"reduces accuracy (README_chunk_prefill.md §10).", stacklevel=2)

    ids = torch.tensor(tok.tokenize(sequence), dtype=torch.int,
                       device="cuda:0").unsqueeze(0)
    L = ids.shape[1]
    ip = model.initialize_inference_params(max_seqlen=L + 64)
    try:
        ip["mha"].max_batch_size = 1
    except Exception:
        pass

    out = torch.full((L - 1,), float("nan"), dtype=torch.float32)
    pos = 0
    while pos < L:
        end = min(pos + chunk, L)
        logits, _ = model(ids[:, pos:end], inference_params_dict=ip)
        for g in ("mha", "hcl", "hcm", "hcs"):
            if g in ip:
                ip[g].seqlen_offset = end
        for m in model.modules():
            c = getattr(m, "_qkv_cache", None)
            if c is not None and hasattr(c, "flush_residual"):
                c.flush_residual()
        n_pred = min(end - pos, L - 1 - pos)
        if n_pred > 0:
            lp = torch.log_softmax(logits[0, :n_pred].float(), dim=-1)
            tgt = ids[0, pos + 1:pos + 1 + n_pred].long()
            out[pos:pos + n_pred] = lp[
                torch.arange(n_pred, device=lp.device), tgt].cpu()
        pos = end
        del logits
        torch.cuda.empty_cache()

    return out if return_per_token else float(out.mean())
