#!/usr/bin/env python3
"""Produce the pre-quantized int4 Evo 2 checkpoint (run once, on big hardware).

This is the publishing step. It loads bf16 across whatever GPUs are visible,
quantizes to int4, and writes a checkpoint that `turboquant.int4_checkpoint.
load_int4_model` can rebuild on a SINGLE GPU without ever materialising the
82 GB bf16 model.

    python tools/make_int4_checkpoint.py \
        --model evo2_40b --out /scratch/.../evo2_40b_int4.pt

Then, on one card:

    from turboquant.int4_checkpoint import load_int4_model
    model, tok = load_int4_model("evo2_40b", "evo2_40b_int4.pt")
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE))):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="evo2_40b")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--block-size", type=int, default=128)
    args = ap.parse_args()

    from evo2 import Evo2
    if "40b" in args.model:
        from turboquant._te_multigpu_patch import patch_te_multi_gpu_amax_reduce
        patch_te_multi_gpu_amax_reduce()

    print(f"loading {args.model} (bf16, all visible GPUs) ...", flush=True)
    t0 = time.perf_counter()
    obj = Evo2(args.model)
    model = obj.model
    print(f"  loaded in {time.perf_counter()-t0:.0f}s", flush=True)

    from turboquant.quant_linear import quantize_model_weights_gpu
    print(f"quantizing to int{args.bits} ...", flush=True)
    t1 = time.perf_counter()
    stats = quantize_model_weights_gpu(model, bit_width=args.bits,
                                       block_size=args.block_size, verbose=False)
    print(f"  {stats['replaced']} layers, "
          f"{stats['native_bytes']/1e9:.1f} -> {stats['compressed_bytes']/1e9:.1f} GB "
          f"({stats['ratio']:.2f}x) in {time.perf_counter()-t1:.0f}s", flush=True)

    from turboquant.int4_checkpoint import save_int4_checkpoint
    print(f"writing {args.out} ...", flush=True)
    t2 = time.perf_counter()
    info = save_int4_checkpoint(model, args.out)
    print(f"  wrote {info['quantized_layers']} quantized layers, "
          f"{info['tensors']} tensors, {info['bytes']/1e9:.1f} GB "
          f"in {time.perf_counter()-t2:.0f}s", flush=True)
    print(f"on-disk size: {os.path.getsize(args.out)/1e9:.1f} GB", flush=True)


if __name__ == "__main__":
    main()
