#!/usr/bin/env python3
"""Does a chunked walk through a REAL HyenaCascade reproduce one parallel pass?

The standalone math test in turboquant/block_continuation.py proves the IIR
decomposition. It says nothing about whether the patch is wired into the model
correctly -- and the first two attempts failed exactly there: once because
HyenaCascade.forward routed every block to sequential_forward (making the patch
dead code), once because parallel_iir returns (B, L, D) while the correction was
built in (B, D, L).

This runs the actual vortex layers on CPU with random weights, small enough to
need no GPU, and checks all three Hyena variants:

    hcl -- long IIR   (the correction that needs real math)
    hcm -- inner FIR, 128 taps, via fftconv_func
    hcs -- inner FIR, 7 taps, via conv1d

For each: one parallel pass over L tokens (no inference params, so no boundaries
-- the ground truth) versus the same tokens fed as blocks through the stateful
path. Without the patch the blocked walk must diverge; with it, it must match.
"""
from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vortex.model.cache import HyenaCascadeFIRInferenceParams
from vortex.model.model import HyenaCascade
from vortex.model.utils import dotdict

HIDDEN = 64
HEADS = 8
STATE = 16
L = 256
BLOCKS = (64, 128, 256)


def make_config():
    return dotdict(
        dict(
            hidden_size=HIDDEN,
            num_filters=HIDDEN,
            num_attention_heads=HEADS,
            state_size=STATE,
            short_filter_length=3,
            short_filter_bias=False,
            column_split_hyena=False,
            interleave=True,
            hyena_flip_x1x2=False,
            use_flashfft=False,
            use_flash_depthwise=False,
            inference_mode=True,
            print_activations=False,
            prefill_style="fft",
            eps=1e-6,
        )
    )


def make_layer(kind: str) -> HyenaCascade:
    cfg = make_config()
    inner = {"hcl": None, "hcm": 128, "hcs": 7}[kind]
    groups = HIDDEN if kind == "hcl" else HIDDEN // 4
    torch.manual_seed(0)
    layer = HyenaCascade(cfg, layer_idx=0, hyena_filter_groups=groups,
                         fir_inner_filter_length=inner).to(torch.float32)
    with torch.no_grad():
        layer.short_filter_weight.normal_(0, 0.3)
        if kind == "hcl":
            # A trained model's poles sit inside the unit disc; random init does
            # not, and exp(+t) overflows long before the boundary matters.
            layer.log_poles.copy_(-0.02 - 0.10 * torch.rand_like(layer.log_poles))
            layer.residues.normal_(0, 0.2)
            layer.D.normal_(0, 0.1)
        else:
            layer.h.normal_(0, 0.1)
            if layer.D is not None:
                layer.D.normal_(0, 0.1)
    return layer.eval()


def fresh_params():
    return HyenaCascadeFIRInferenceParams(
        fir_filter_length=3, fir_inner_filter_length=128, seqlen_offset=0)


@torch.inference_mode()
def blocked(layer, u, block):
    ip = fresh_params()
    outs = []
    for a in range(0, u.shape[1], block):
        y, _ = layer(u[:, a:a + block], inference_params=ip)
        ip.seqlen_offset = a + y.shape[1]
        outs.append(y)
    return torch.cat(outs, dim=1)


@torch.inference_mode()
def blocks_then_tokens(layer, u, block, n_tokens):
    """Blocks up to L-n_tokens, then single-token steps -- the transition
    generation actually makes. Checks the state a corrected block HANDS ON is
    the state step_iir/step_fir expect, not just that the block's outputs are
    right."""
    ip = fresh_params()
    L = u.shape[1]
    cut = L - n_tokens
    outs = []
    for a in range(0, cut, block):
        y, _ = layer(u[:, a:min(a + block, cut)], inference_params=ip)
        ip.seqlen_offset = a + y.shape[1]
        outs.append(y)
    for j in range(cut, L):
        y, _ = layer(u[:, j:j + 1], inference_params=ip)
        ip.seqlen_offset = j + 1
        outs.append(y)
    return torch.cat(outs, dim=1)


@torch.inference_mode()
def main():
    torch.manual_seed(1)
    u = torch.randn(1, L, 3 * HIDDEN) * 0.5

    from turboquant.block_continuation import (install_block_continuation,
                                               uninstall_block_continuation)

    print(f"hidden={HIDDEN} heads={HEADS} state={STATE} L={L}\n")
    all_ok = True
    for kind in ("hcl", "hcm", "hcs"):
        layer = make_layer(kind)
        exact = layer.parallel_forward(u, None, None)[0]

        def err(y):
            """Stock blocks after the first come back length-1: HyenaCascade
            .forward sends them to sequential_forward, which does u = u[:, -1].
            In the full model that length-1 output then BROADCASTS over the
            residual stream instead of raising, which is why chunked prefill
            produced plausible-looking garbage rather than a crash."""
            if y.shape != exact.shape:
                return None, f"len {y.shape[1]} != {exact.shape[1]}"
            e = float((y - exact).abs().max())
            return e, f"{e:.3e}"

        uninstall_block_continuation()
        stock = {b: err(blocked(layer, u, b))[1] for b in BLOCKS}

        install_block_continuation(layer, verbose=False)
        try:
            fixed = {b: err(blocked(layer, u, b)) for b in BLOCKS}
            handoff = err(blocks_then_tokens(layer, u, 64, 8))
        finally:
            uninstall_block_continuation()

        scale = float(exact.abs().max())
        print(f"{kind}: |exact| max = {scale:.4f}")
        print(f"  {'block':>7} {'stock':>22} {'fixed max|err|':>15} {'rel':>10}")
        for b in BLOCKS:
            e, txt = fixed[b]
            ok = e is not None and e / max(scale, 1e-12) < 1e-4
            all_ok &= ok
            rel = "-" if e is None else f"{e / max(scale, 1e-12):.2e}"
            print(f"  {b:>7} {stock[b]:>22} {txt:>15} {rel:>10} "
                  f"{'ok' if ok else 'FAIL'}")
        # block == L is the no-boundary control: it must be exact either way

        he, htxt = handoff
        hok = he is not None and he / max(scale, 1e-12) < 1e-4
        all_ok &= hok
        print(f"  {'64+step':>7} {'-':>22} {htxt:>15} "
              f"{'-' if he is None else f'{he / max(scale, 1e-12):.2e}':>10} "
              f"{'ok' if hok else 'FAIL'}   (blocks then single tokens)")
        print()

    print("PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
