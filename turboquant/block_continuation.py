"""Correct block-wise continuation for Evo 2's Hyena layers.

Problem this solves. Evo 2 can be driven two ways: one parallel pass over a
whole block (fast, but every implemented variant assumes it starts from a ZERO
state), or single-token steps (continues correctly, but one token at a time).
There is no supported way to advance an EXISTING state across a BLOCK -- the
routine that would do it, `prefill_via_hybrid_recurrence` ("recurrence-
convolution over blocks"), is a `NotImplementedError` stub upstream. Feeding
multi-token blocks anyway silently produces garbage (measured: Pearson ~0
against an exact forward; see FINDINGS.md).

Three state carries have to be fixed for a block to be equivalent to one
continuous pass:

  1. short FIR   (all Hyena layers, 3 taps)      -- finite receptive field
  2. inner FIR   (hcm 128 taps, hcs 7 taps)      -- finite receptive field
  3. IIR         (hcl, the long convolution)     -- infinite receptive field

(1) and (2) are exact once the previous block's trailing inputs are prepended
instead of zero-padding, because the filter only reaches back `fir_length-1`
samples.

(3) needs real math, but it is simple because the recurrence is LINEAR. In
modal form each pole evolves as

    state_{t+1} = pole * state_t + x_t

so unrolling across a block of length L starting from s0:

    state_t   = pole^(t+1) * s0  +  sum_{i<=t} pole^(t-i) * x_i
                └── correction ──┘  └── exactly what the existing FFT computes ──┘

Therefore, given the stock (zero-state) results:

    y_corrected[b,d,t] = y_zero[b,d,t] + sum_s residues[d,s] * pole[d,s]^(t+1) * s0[b,d,s]
    state_L            = pole^L * s0   + state_zero

`pole^t` is already materialised inside `prefill_via_modal_fft` as
`exp(poles * t)`, so the correction reuses tensors the model computes anyway.

Use it with `install_block_continuation(model)` before feeding blocks.

Running this module directly exercises the IIR decomposition alone, against L
single-token steps from the same starting state -- the decisive check on the
math, independent of the rest of the model. Two further tests cover the wiring,
which is where the first two attempts actually failed:

    experiments/satmut/test_block_continuation_layers.py   real layers, CPU
    experiments/satmut/diag_chunk_fidelity.py --block-fix  full model, GPU

Verified on evo2_7b at 8192 bp in bf16: Pearson 0.99992-0.99994 against an exact
single forward at every chunk size, versus ~0 unpatched, with the zero-boundary
case still bit-exact. Not yet verified on the 40B, at long context, or with
quantized weights/KV.
"""
from __future__ import annotations

import torch


# --------------------------------------------------------------------------
# reference recurrence (ground truth for the unit test)
# --------------------------------------------------------------------------

def iir_reference_steps(x: torch.Tensor, poles: torch.Tensor,
                        residues: torch.Tensor, s0: torch.Tensor):
    """Advance the modal recurrence one token at a time -- the definition.

    x:        (B, D, L)         complex or real input to the long conv
    poles:    (D, S)            complex
    residues: (D, S)            complex
    s0:       (B, D, S)         complex incoming state (zeros = fresh block)

    Returns (y, state_L) with y (B, D, L) and state_L (B, D, S).
    """
    B, D, L = x.shape
    state = s0.clone()
    y = torch.zeros((B, D, L), dtype=torch.complex64, device=x.device)
    for t in range(L):
        # state_{t} = pole * state_{t-1} + x_t
        state = poles[None] * state + x[:, :, t].unsqueeze(-1)
        y[:, :, t] = (residues[None] * state).sum(dim=-1)
    return y, state


def iir_block_zero_state(x: torch.Tensor, poles: torch.Tensor,
                         residues: torch.Tensor):
    """What the stock implementation effectively computes: the same recurrence
    but forced to start from zero. Used here as the 'stock' term the correction
    is added to, so the unit test exercises exactly the decomposition the real
    patch will rely on."""
    B, D, _ = x.shape
    S = poles.shape[1]
    zero = torch.zeros((B, D, S), dtype=torch.complex64, device=x.device)
    return iir_reference_steps(x, poles, residues, zero)


# --------------------------------------------------------------------------
# the correction
# --------------------------------------------------------------------------

def iir_state_correction(L: int, poles: torch.Tensor, residues: torch.Tensor,
                         s0: torch.Tensor):
    """Correction terms that turn a zero-state block result into a continuation.

    Returns (y_corr, state_decay) where

        y_corr[b,d,t]  = sum_s residues[d,s] * pole[d,s]^(t+1) * s0[b,d,s]
        state_decay    = pole^L * s0

    so that
        y_true      = y_zero_state + y_corr
        state_true  = state_zero_state + state_decay
    """
    device = poles.device
    # pole^(t+1) for t = 0..L-1  ->  exponents 1..L
    exps = torch.arange(1, L + 1, device=device, dtype=torch.float32)
    # (D, S, L); computed in log space for numerical stability, matching the
    # stock code's own `state_s = (poles * t).exp()` construction.
    pole_pow = torch.exp(torch.log(poles).unsqueeze(-1) * exps)  # (D, S, L)

    # y_corr = sum_s residues[d,s] * pole_pow[d,s,t] * s0[b,d,s]
    weighted = residues.unsqueeze(-1) * pole_pow            # (D, S, L)
    y_corr = torch.einsum("bds,dsl->bdl", s0, weighted)     # (B, D, L)

    state_decay = (poles ** L).unsqueeze(0) * s0            # (B, D, S)
    return y_corr, state_decay


# --------------------------------------------------------------------------
# unit test: one corrected block  ==  L single-token steps
# --------------------------------------------------------------------------

def _unit_test(B=2, D=8, S=16, L=64, seed=0, device="cpu"):
    torch.manual_seed(seed)

    def cplx(*shape):
        return torch.complex(torch.randn(*shape), torch.randn(*shape)).to(device)

    # Poles must lie inside the unit disc for a stable recurrence, as they do in
    # a trained model; sample magnitudes in [0.80, 0.999].
    mag = (0.80 + 0.199 * torch.rand(D, S)).to(device)
    ang = (2 * torch.pi * torch.rand(D, S)).to(device)
    poles = torch.polar(mag, ang)
    residues = cplx(D, S) * 0.1
    x = cplx(B, D, L)
    s0 = cplx(B, D, S)

    # ground truth: continue the recurrence token by token from s0
    y_true, state_true = iir_reference_steps(x, poles, residues, s0)

    # stock path (zero state) + our correction
    y_zero, state_zero = iir_block_zero_state(x, poles, residues)
    y_corr, state_decay = iir_state_correction(L, poles, residues, s0)
    y_fixed = y_zero + y_corr
    state_fixed = state_zero + state_decay

    y_err = (y_fixed - y_true).abs().max().item()
    s_err = (state_fixed - state_true).abs().max().item()
    # how wrong the stock path is on its own, for contrast
    y_stock_err = (y_zero - y_true).abs().max().item()
    s_stock_err = (state_zero - state_true).abs().max().item()

    print(f"  B={B} D={D} S={S} L={L}")
    print(f"    stock (zero-state)  max|y err| = {y_stock_err:.3e}   "
          f"max|state err| = {s_stock_err:.3e}")
    print(f"    with correction     max|y err| = {y_err:.3e}   "
          f"max|state err| = {s_err:.3e}")
    ok = y_err < 1e-4 and s_err < 1e-4
    print(f"    -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    print("IIR block-continuation correction: one block vs L single-token steps\n")
    results = [
        _unit_test(B=2, D=8, S=16, L=64, seed=0),
        _unit_test(B=1, D=4, S=8, L=256, seed=1),
        _unit_test(B=3, D=16, S=16, L=16, seed=2),
    ]
    print(f"\n{sum(results)}/{len(results)} passed")
    if all(results):
        print("\nThe decomposition holds: a zero-state block result plus the")
        print("pole^t * s0 correction reproduces a true continuation exactly.")
        print("This tests the math only. For the wiring, run")
        print("  experiments/satmut/test_block_continuation_layers.py")


# ==========================================================================
# wiring the correction into the model
# ==========================================================================
#
# Three call sites, patched onto HyenaInferenceEngine:
#
#   parallel_fir  -- prepend the saved trailing inputs instead of zero-padding.
#                    Handles BOTH the short filter (gate=False -> fir_state_dict)
#                    and the inner filter (gate=True -> fir_inner_state_dict);
#                    `gate` is what distinguishes them, since parallel_fir itself
#                    is not told which filter it is serving.
#   parallel_iir  -- add the pole^t * s0 output term and the pole^L * s0 state
#                    term. Wrapped rather than reimplemented: the stock function
#                    ends with  y = (y_conv + x1v*D) * x2,  so adding
#                    y_corr * x2 afterwards is algebraically identical to adding
#                    y_corr to y_conv before that line.
#
# Every patch falls back to stock behaviour when no prior state exists (the
# first block of a sequence), so a fresh prefill is bit-identical to before.

_ORIG = {}

# How many output positions of the IIR correction to materialise at once. The
# intermediate pole^t tensor is (D, S, chunk); at D=8192, S=16 a 512-wide chunk
# is 268 MB in fp32, which is the same order as the FFT workspace the stock
# prefill already allocates.
_CORR_CHUNK = 512


def _get_state(inference_params, gate, layer_idx):
    if inference_params is None:
        return None
    # HyenaCascadeIIRInferenceParams has no fir_inner_state_dict at all (hcl
    # layers never run the gated inner filter), so read it defensively.
    d = (getattr(inference_params, "fir_inner_state_dict", None) if gate
         else getattr(inference_params, "fir_state_dict", None))
    return None if d is None else d.get(layer_idx, None)


def _as_real(x):
    """State is stored real (engine.py casts the ifft result to float32), but be
    tolerant of a complex tensor: poles and residues are real parameters, so the
    imaginary part is numerical noise either way."""
    return (x.real if x.is_complex() else x).to(torch.float32)


def install_block_continuation(model, verbose: bool = True):
    """Patch the Hyena engine so multi-token blocks continue an existing state
    correctly. Idempotent."""
    import torch.nn.functional as F
    from vortex.model.engine import HyenaInferenceEngine
    from vortex.model.utils import column_split
    try:
        from vortex.model.engine import fftconv_func
    except Exception:
        fftconv_func = None

    if "parallel_fir" in _ORIG:
        if verbose:
            print("[block-continuation] already installed")
        return model

    _ORIG["parallel_fir"] = HyenaInferenceEngine.parallel_fir
    _ORIG["parallel_iir"] = HyenaInferenceEngine.parallel_iir

    def parallel_fir(self, fir_fn, u, weight, bias, L, dims, groups=None,
                     gated_bias=False, column_split_hyena=False, dim_last=True,
                     fir_length=3, gate=False, inference_params=None,
                     prefill_mode=None, padding_mask=None):
        L = u.shape[1] if dim_last else u.shape[2]

        if gate:
            hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
            if column_split_hyena:
                x2, x1, v = column_split(u, num_attention_heads,
                                          hidden_size_per_attention_head)
            else:
                x2, x1, v = u.split([hidden_size, hidden_size, hidden_size], dim=1)
            if self.hyena_flip_x1x2:
                x1, x2 = x2, x1
            u = x1 * v            # (B, D, L)
        elif dim_last:
            u = u.permute(0, 2, 1)  # (B, D, L)

        # --- the fix: real history instead of zero padding -----------------
        prev = _get_state(inference_params, gate, self.layer_idx)
        P = 0
        u_conv = u
        if prev is not None and prev.shape[-1] > 0 and prev.shape[1] == u.shape[1]:
            P = prev.shape[-1]
            u_conv = torch.cat([prev.to(u.dtype), u], dim=-1)
        Lc = L + P

        if fir_length >= 128:
            if fftconv_func is None:
                raise RuntimeError("fftconv_func unavailable for the inner filter")
            with torch.autocast("cuda"):
                z = fftconv_func(u_conv.to(torch.float32),
                                 weight[:, :, :Lc].to(torch.float32), bias, None,
                                 gelu=False, bidirectional=False,
                                 print_activations=False, groups=groups,
                                 layer_idx=self.layer_idx)
                z = z.to(u.dtype)
        else:
            z = fir_fn(u_conv.to(torch.float32), weight.to(torch.float32), bias=None,
                       stride=1, padding=fir_length - 1,
                       groups=u_conv.shape[1])[..., :Lc]
            z = z.to(u.dtype)
            if bias is not None:
                if gated_bias:
                    z = z + bias[None, :, None] * u_conv
                else:
                    z = z + bias[None, :, None]

        if P:
            z = z[..., P:]        # drop the prepended history

        if type(padding_mask) == torch.Tensor:
            z = z * padding_mask[:, None]
        if gate:
            z = x2 * z

        # Take the trailing samples from the CONCATENATED signal, not just this
        # block, so a block shorter than the filter still hands on a full window.
        fir_state = (u_conv[..., -fir_length + 1:]
                     if inference_params is not None else None)
        return z, fir_state

    def parallel_iir(self, z_pre, h, D, L, poles, residues, t, dims, layer_idx,
                     inference_params=None, prefill_style="fft", fftconv_fn=None,
                     padding_mask=None, use_flashfft=False,
                     column_split_hyena=False, long_fir_threshold=None):
        s0 = None
        if inference_params is not None:
            s0 = inference_params.state_dict.get(layer_idx, None)

        y = _ORIG["parallel_iir"](self, z_pre, h, D, L, poles, residues, t, dims,
                                   layer_idx, inference_params, prefill_style,
                                   fftconv_fn, padding_mask, use_flashfft,
                                   column_split_hyena, long_fir_threshold)
        if s0 is None:
            return y

        # x2, recomputed exactly as the stock function splits it
        hidden_size, num_attention_heads, hidden_size_per_attention_head, _, _ = dims
        if column_split_hyena:
            zz = z_pre.reshape(z_pre.shape[0], num_attention_heads,
                               3 * hidden_size_per_attention_head, z_pre.shape[2])
            x2 = zz[:, :, :hidden_size_per_attention_head]
            x1 = zz[:, :, hidden_size_per_attention_head:2 * hidden_size_per_attention_head]
            x2 = x2.reshape(x2.shape[0], -1, x2.shape[-1])
            x1 = x1.reshape(x1.shape[0], -1, x1.shape[-1])
        else:
            x2, x1, _v = z_pre.split([hidden_size, hidden_size, hidden_size], dim=1)
        if self.hyena_flip_x1x2:
            x1, x2 = x2, x1
        # x2 is (B, D, L) -- the stock function's own working layout.

        # `poles` are LOG-poles (a real nn.Parameter of shape (G, S, 1); the
        # trailing axis is the dummy seqlen dim step_iir squeezes off). Residues
        # are real too, so the whole correction is real arithmetic.
        lp = poles.to(torch.float32)
        rs = residues.to(torch.float32)
        if lp.dim() == 3 and lp.shape[-1] == 1:
            lp = lp[..., 0]                       # (G, S)
        if rs.dim() == 3 and rs.shape[-1] == 1:
            rs = rs[..., 0]                       # (G, S)
        Dh = x2.shape[1]
        if lp.shape[0] != Dh:                     # grouped filters
            rep = Dh // lp.shape[0]
            lp = lp.repeat_interleave(rep, 0)
            rs = rs.repeat_interleave(rep, 0)

        s0r = _as_real(s0)                        # (B, D, S)
        c = rs.unsqueeze(0) * s0r                 # (B, D, S), residue-weighted

        # y_corr[b,d,t] = sum_s residues[d,s] * pole[d,s]^(t+1) * s0[b,d,s],
        # built a slice of t at a time so pole^t never exists at full length.
        y_corr = torch.empty((s0r.shape[0], Dh, L), dtype=torch.float32,
                             device=lp.device)
        for a in range(0, L, _CORR_CHUNK):
            b = min(a + _CORR_CHUNK, L)
            e = torch.arange(a + 1, b + 1, device=lp.device, dtype=torch.float32)
            pw = torch.exp(lp.unsqueeze(-1) * e)          # (D, S, b-a)
            y_corr[:, :, a:b] = torch.einsum("bds,dsl->bdl", c, pw)
            del pw

        # The stock function ends with  y = (y_conv + x1v*D) * x2  and then
        # PERMUTES to (B, L, D). So the correction is gated by x2 in (B, D, L)
        # and permuted to match before it is added.
        corr = (y_corr.to(x2.dtype) * x2).permute(0, 2, 1)   # (B, L, D)
        if corr.shape != y.shape:
            raise RuntimeError(
                f"block-continuation layout mismatch: correction {tuple(corr.shape)} "
                f"vs stock output {tuple(y.shape)} (layer {layer_idx}, L={L})")
        y = y + corr.to(y.dtype)

        # final state: pole^L * s0 on top of the zero-state result the stock
        # prefill just stored
        st = inference_params.state_dict.get(layer_idx, None)
        if st is not None:
            decay = torch.exp(lp * float(L))                 # (D, S)
            new_st = _as_real(st) + decay.unsqueeze(0) * s0r
            inference_params.state_dict[layer_idx] = new_st.to(
                st.real.dtype if st.is_complex() else st.dtype)
        return y

    # Routing. HyenaCascade.forward sends ANY call to sequential_forward once
    # state exists for that layer -- so blocks 2..N of a chunked prefill never
    # reach parallel_forward at all, and the corrections above would be dead
    # code. Send multi-token inputs back to the parallel path (now
    # state-aware); keep single tokens on the recurrent step, which is already
    # correct and much cheaper for decode.
    from vortex.model.model import HyenaCascade
    _ORIG["hc_forward"] = HyenaCascade.forward

    def hc_forward(self, u, inference_params=None, padding_mask=None, *a, **kw):
        single = (u.shape[1] == 1)
        if (inference_params is not None and single
                and self.layer_idx in inference_params.fir_state_dict.keys()):
            return self.sequential_forward(u, inference_params)
        return self.parallel_forward(u, inference_params, padding_mask)

    HyenaCascade.forward = hc_forward
    HyenaInferenceEngine.parallel_fir = parallel_fir
    HyenaInferenceEngine.parallel_iir = parallel_iir
    if verbose:
        print("[block-continuation] patched parallel_fir + parallel_iir")
    return model


def uninstall_block_continuation():
    from vortex.model.engine import HyenaInferenceEngine
    if "parallel_fir" in _ORIG:
        HyenaInferenceEngine.parallel_fir = _ORIG.pop("parallel_fir")
        HyenaInferenceEngine.parallel_iir = _ORIG.pop("parallel_iir")
    if "hc_forward" in _ORIG:
        from vortex.model.model import HyenaCascade
        HyenaCascade.forward = _ORIG.pop("hc_forward")
