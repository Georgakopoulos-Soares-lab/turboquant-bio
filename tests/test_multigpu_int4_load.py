#!/usr/bin/env python3
"""Load the int4 checkpoint sharded across every visible GPU and check it
scores identically to the single-device path.

Before `device="auto"` existed, `load_int4_model` pinned every block to one
device, so the published checkpoint could only be used on a single GPU. Reaching
long context in int4 therefore meant loading the 82 GB bf16 model and quantizing
it in place -- which needs exactly the hardware the compression exists to avoid.

The reference value is the one in the model card and the paper: 32,768 bases of
BRCA1 (hg38 chr17:43,044,295), scored in 8,192-token chunks, mean log-likelihood
-0.83931, at chunk 1024. Chunk size sets the number of block boundaries and
moves the score slightly, so the comparison is only meaningful at the chunk the
reference used -- scoring at 8192 shifts it by ~2e-4 and also needs a 16 GiB
Hyena FFT buffer, which will not fit on one 80 GB card.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

# Site-specific paths, overridable so this runs somewhere other than where it
# was written. The reference value below is tied to this exact locus, so a
# different genome build or locus will not reproduce it.
HG38 = os.environ.get(
    "TQ_HG38", "/scratch/10917/michalakis/dnalongbench_data/eqtl/hg38.fa")
DEFAULT_CKPT = os.environ.get(
    "TQ_INT4_CKPT", "/scratch/10917/michalakis/hf_cache/evo2_40b_int4.pt")
CHROM, START = "chr17", 43_044_295
REFERENCE_MEAN_LOGL = -0.839311991422749
_GROUPS = ("mha", "hcl", "hcm", "hcs")


def _flush(model):
    for m in model.modules():
        c = getattr(m, "_qkv_cache", None)
        if c is not None and hasattr(c, "flush_residual"):
            c.flush_residual()


@torch.inference_mode()
def score(model, ids, chunk):
    ip = model.initialize_inference_params(max_seqlen=ids.shape[1] + 64)
    try:
        ip["mha"].max_batch_size = 1
    except Exception:
        pass
    tot, cnt, pos, L = 0.0, 0, 0, ids.shape[1]
    while pos < L:
        end = min(pos + chunk, L)
        logits, _ = model(ids[:, pos:end], inference_params_dict=ip)
        for g in _GROUPS:
            if g in ip:
                ip[g].seqlen_offset = end
        _flush(model)
        n_pred = min(end - pos, L - 1 - pos)
        if n_pred > 0:
            lp = torch.log_softmax(logits[0, :n_pred].float(), dim=-1)
            tgt = ids[0, pos + 1:pos + 1 + n_pred].long()
            tot += float(lp[torch.arange(n_pred, device=lp.device), tgt].sum())
            cnt += n_pred
        pos = end
        del logits
        torch.cuda.empty_cache()
    return tot / cnt, cnt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_40b")
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--context", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=1024)
    ap.add_argument("--tol", type=float, default=1e-5)
    ap.add_argument("--device", default="auto",
                    help='"auto" to shard, or an explicit device to pin')
    ap.add_argument("--min-gpus", type=int, default=2,
                    help="skip unless at least this many GPUs are visible")
    args = ap.parse_args()

    import turboquant
    print(f"turboquant package: {turboquant.__file__}", flush=True)
    n_dev = torch.cuda.device_count()
    print(f"visible GPUs: {n_dev}", flush=True)
    if n_dev < args.min_gpus:
        print(f"SKIP: need >={args.min_gpus} GPUs, have {n_dev}", flush=True)
        return 0

    if "40b" in args.model:
        from turboquant._te_multigpu_patch import patch_te_multi_gpu_amax_reduce
        patch_te_multi_gpu_amax_reduce()

    from turboquant.int4_checkpoint import load_int4_model
    t0 = time.perf_counter()
    model, tok = load_int4_model(args.model, args.ckpt, device=args.device,
                                 verbose=True)
    print(f"loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    per_dev = [torch.cuda.memory_allocated(i) / 1e9 for i in range(n_dev)]
    print("resident per GPU (GB): "
          + ", ".join(f"{i}:{g:.1f}" for i, g in enumerate(per_dev)), flush=True)
    print(f"total resident: {sum(per_dev):.1f} GB", flush=True)

    used = [i for i, g in enumerate(per_dev) if g > 1.0]
    if args.device == "auto" and n_dev > 1 and len(used) < 2:
        print(f"\nFAIL: weights landed on {len(used)} device(s) -- not sharded",
              flush=True)
        return 1

    from turboquant.fused_weight_dequant import install_fast_weight_forward
    install_fast_weight_forward(model)
    from turboquant.fused_kv_attention import install_fused_kv_quant
    install_fused_kv_quant(model, bits=4, model_name=args.model,
                           fused_prefill=True, verbose=False)
    from turboquant.block_continuation import install_block_continuation
    install_block_continuation(model)

    if not os.path.exists(HG38):
        print(f"SKIP: no genome at {HG38} (set TQ_HG38)", flush=True)
        return 0
    import pyfaidx
    seq = str(pyfaidx.Fasta(HG38)[CHROM][START:START + args.context]).upper()
    ids = torch.tensor(tok.tokenize(seq), dtype=torch.int,
                       device="cuda:0").unsqueeze(0)

    mean_logl, n = score(model, ids, args.chunk)
    delta = abs(mean_logl - REFERENCE_MEAN_LOGL)
    print(f"\nscored {n} positions")
    print(f"  sharded ({len(used)} GPUs) mean logL = {mean_logl:.9f}")
    print(f"  reference (1 GPU)         = {REFERENCE_MEAN_LOGL:.9f}")
    print(f"  |delta|                   = {delta:.3e}")

    ok = delta < args.tol
    print(f"\n{'PASS' if ok else 'FAIL'}: sharded int4 load "
          f"{'matches' if ok else 'DIVERGES FROM'} the single-GPU reference",
          flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
