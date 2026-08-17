#!/usr/bin/env python3
"""Score a FASTA sequence with Evo 2, compressed and correct.

    python examples/score_sequence.py --fasta my.fa --model evo2_7b
    python examples/score_sequence.py --fasta my.fa --model evo2_40b --tier tier2

The 40B/tier2 form runs on a single 80 GB GPU; the int4 checkpoint is fetched
and cached on first use.
"""
import argparse

from turboquant import EFFECTIVE_CONTEXT, load_evo2, score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--model", default="evo2_7b",
                    choices=["evo2_7b", "evo2_40b"])
    ap.add_argument("--tier", default="baseline",
                    choices=["baseline", "tier1", "tier2"])
    ap.add_argument("--int4-ckpt", default=None,
                    help="local int4 checkpoint; omit to auto-download")
    ap.add_argument("--max-bp", type=int, default=EFFECTIVE_CONTEXT,
                    help=f"truncate each record to this many bp "
                         f"(default {EFFECTIVE_CONTEXT}, the model's measured "
                         f"effective context -- more than this reduces accuracy)")
    args = ap.parse_args()

    import pyfaidx
    fa = pyfaidx.Fasta(args.fasta)

    model, tok = load_evo2(args.model, tier=args.tier,
                           int4_ckpt=args.int4_ckpt)

    print(f"\n{'record':<30} {'bp':>8} {'mean logL':>11}")
    for name in fa.keys():
        seq = str(fa[name][:args.max_bp]).upper()
        if "N" in seq:
            seq = seq.replace("N", "A")     # Evo 2 has no N token
        v = score(model, tok, seq, model_name=args.model)
        print(f"{name[:30]:<30} {len(seq):>8} {v:>11.5f}", flush=True)


if __name__ == "__main__":
    main()
