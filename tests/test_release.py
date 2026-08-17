#!/usr/bin/env python3
"""Release gate for TurboQuant-Bio. Run this before tagging or publishing.

Three layers, cheapest first, so a broken build fails in seconds rather than
after a 40 GB download:

  1. math      : the IIR block-continuation decomposition, pure torch, no GPU
  2. layers    : real vortex Hyena layers (hcl/hcm/hcs) on CPU, random weights
  3. numerical : a fixed sequence must score a PINNED value on real weights

Layer 3 is the one that matters most. Every bug in this project's history
produced plausible-looking numbers rather than a crash:

  * chunked prefill dropped 4095 of every 4096 tokens  -> output looked fine
  * 42 fp8 extra states were silently not restored     -> output looked fine
  * an fp32 RMSNorm scale promoted the activation dtype -> that one DID crash

A pinned expected value is the only thing that catches the first two.

    python tests/test_release.py --level fast          # 1+2, no GPU, seconds
    python tests/test_release.py --level full \\
        --int4-ckpt evo2_40b_int4.pt                   # adds 3, needs a GPU
"""
from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Pinned references, BRCA1 chr17:43,044,295. Measured on this codebase; a change
# here means the numerics moved and you must find out why before releasing.
PINNED = {
    # (model, tier, context, chunk): (mean_logL, tolerance)
    ("evo2_7b",  "baseline", 8192, 4096): (-0.90374, 2e-4),
    ("evo2_7b",  "tier1",    8192, 4096): (-0.90361, 2e-4),
    ("evo2_7b",  "tier2",    8192, 4096): (-0.90698, 2e-4),
    ("evo2_40b", "tier2",   32768, 1024): (-0.83931, 2e-4),
}
# Checkpoints needed per model for tier2; tier1 needs none (KV quantization is
# runtime-only, the weights are untouched).
CKPT_ARG = {"evo2_7b": "--int4-ckpt-7b", "evo2_40b": "--int4-ckpt"}
# The pinned numerical layer needs a reference genome. Point HG38 at your own
# hg38 FASTA (or set the TURBOQUANT_HG38 environment variable); the pinned values
# were measured on GRCh38 at CHROM:START below.
HG38 = os.environ.get("TURBOQUANT_HG38", "hg38.fa")
CHROM, START = "chr17", 43_044_295


def run_math() -> bool:
    print("\n[1/3] IIR block-continuation math (no GPU) ...", flush=True)
    from turboquant.block_continuation import _unit_test
    ok = all(_unit_test(B=b, D=d, S=s, L=l, seed=i, device="cpu")
             for i, (b, d, s, l) in enumerate(
                 [(2, 8, 16, 64), (1, 4, 8, 256), (3, 16, 16, 16)]))
    print("      PASS" if ok else "      FAIL", flush=True)
    return ok


def run_layers() -> bool:
    print("\n[2/3] real Hyena layers, CPU, all three variants ...", flush=True)
    import subprocess
    r = subprocess.run(
        [sys.executable,
         os.path.join(_HERE, "test_block_continuation_layers.py")],
        capture_output=True, text=True)
    ok = "PASS" in r.stdout and "FAIL" not in r.stdout
    print("      PASS" if ok else f"      FAIL\n{r.stdout[-1500:]}", flush=True)
    return ok


def run_numerical(int4_ckpt: str | None, int4_ckpt_7b: str | None) -> bool:
    print("\n[3/3] pinned numerical check on real weights ...", flush=True)
    import torch
    if not torch.cuda.is_available():
        print("      SKIP (no GPU)", flush=True)
        return True
    if not os.path.exists(HG38):
        print(f"      SKIP (no reference genome at {HG38}; set TURBOQUANT_HG38)",
              flush=True)
        return True
    import pyfaidx
    from turboquant import load_evo2, score

    all_ok = True
    for (model_name, tier, ctx, chunk), (expected, tol) in PINNED.items():
        ck = int4_ckpt_7b if model_name == "evo2_7b" else int4_ckpt
        if tier == "tier2" and not ck:
            print(f"      SKIP {model_name}/{tier} "
                  f"(no {CKPT_ARG[model_name]})", flush=True)
            continue
        fa = pyfaidx.Fasta(HG38)
        seq = str(fa[CHROM][START:START + ctx]).upper()
        model, tok = load_evo2(model_name, tier=tier,
                               int4_ckpt=ck if tier == "tier2" else None,
                               verbose=False)
        got = score(model, tok, seq, chunk=chunk, model_name=model_name)
        delta = abs(got - expected)
        ok = delta <= tol
        all_ok &= ok
        print(f"      {model_name}/{tier} ctx={ctx} chunk={chunk}: "
              f"got {got:.5f} expected {expected:.5f} "
              f"(|d|={delta:.2e} tol={tol:.0e}) "
              f"{'PASS' if ok else 'FAIL'}", flush=True)
        del model
        import gc; gc.collect(); torch.cuda.empty_cache()
    return all_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["fast", "full"], default="fast")
    ap.add_argument("--int4-ckpt", default=None, help="evo2_40b int4 checkpoint")
    ap.add_argument("--int4-ckpt-7b", default=None, help="evo2_7b int4 checkpoint")
    args = ap.parse_args()

    results = [("math", run_math()), ("layers", run_layers())]
    if args.level == "full":
        results.append(("numerical",
                        run_numerical(args.int4_ckpt, args.int4_ckpt_7b)))

    print("\n" + "=" * 46)
    for name, ok in results:
        print(f"  {name:>10}: {'PASS' if ok else 'FAIL'}")
    ok = all(v for _, v in results)
    print(f"  {'OVERALL':>10}: {'PASS' if ok else 'FAIL'}")
    print("=" * 46)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
