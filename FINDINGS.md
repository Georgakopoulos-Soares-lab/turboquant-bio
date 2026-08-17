# Chunked prefill is numerically invalid on Evo 2 — findings, impact, and the fix

**Status: action required.** Our chunked-prefill helper produces wrong numbers on
any sequence longer than one chunk. It does not crash and it does not warn; the
outputs look plausible. Every long-context experiment we have run with it is
invalid and must be re-run.

**A fix is now implemented and verified** (§6): with
`install_block_continuation(model)`, chunked prefill reproduces the exact
single-pass forward to round-off, against r ≈ 0 before. Verified on `evo2_7b`
(r = 0.9999), on `evo2_40b` across three loci (r = 0.9965–0.9999, worst on
low-entropy sequence), and under all three deployment tiers (r ≥ 0.991) — all at
8,192 bp, and at long context (7B to 131,072 bp with 127 boundaries; 40B to
65,536). Error does **not** grow with boundary count.

This document records what was measured, why it happens, which results are and
are not affected, the fix, and what the fix then made measurable:
**§10** Evo 2's effective context (peaks at ~32 kb, then declines),
**§11** a controlled negative on long-range enhancer detection,
**§12** running the 40B on a single 80 GB GPU where bf16 cannot even load,
**§13** open items.

---

## 1. TL;DR

| | verdict |
|---|---|
| Chunked prefill (feeding a long sequence as repeated multi-token blocks), **as shipped** | **invalid** — per-token log-probs are essentially uncorrelated with the truth |
| Chunked prefill **with `install_block_continuation`** | **valid** — r = 0.991–0.9999 across 7B, 40B, three loci and all three tiers; at tier2 quantization dominates and chunking is nearly free (§6) |
| One parallel forward over the whole sequence | **exact** (bit-identical reference) |
| One parallel prefill, then **single-token** steps | **valid** (Δ = +0.049 nats over 2047 tokens, r = 0.99995) |
| Paper accuracy figures (perplexity, GUE, splice, BRCA1, gene completion) | **unaffected** — all short-context, single pass |
| Paper memory figures (Fig 3, Table 2) | **essentially valid** — see §5 |
| Long-context biological experiments (ciliate, eQTL, long-genes, contact-map) | **invalid, must be re-run** |
| Max single parallel prefill, evo2_40b on 4× H100 | **65,536 tokens — identical for bf16 and int4-W** (§7) |
| Extra prefill length bought by weight compression on 40B, **single pass** | **none** — that ceiling is a PyTorch index limit, not memory (§7) |
| Extra context bought by compression, **chunked/incremental** | **real** — there the binding constraint is weights + KV, both of which we compress |
| Evo 2's effective context | **~32 kb**, and *declines* beyond; 8 loci, p = 0.0078 (§10) |
| Long-range enhancer detection (CRISPRi K562) | **absent past 50 kb**; gene-level p = 0.102 (§11) |
| bf16 evo2_40b on ONE 80 GB GPU | **cannot load** — OOM at block 46/50 (§12) |
| int4 evo2_40b on ONE 80 GB GPU | **33.8 GB, 883 tok/s, output identical to 4-GPU**, and 38 % faster than 4 GPUs (§12) |

---

## 2. The measurement

`experiments/satmut/diag_chunk_fidelity.py` scores one fixed 8,192 bp human
sequence (BRCA1 locus, chr17) two ways and compares per-token log-probabilities:

* **exact**: a single `model(input_ids)` forward — no inference params, no
  boundaries. This is the same path `evo2.scoring.score_sequences` uses and the
  path all of the paper's short-context evaluations already rely on.
* **chunked**: the same sequence through our chunked-prefill loop, at several
  chunk sizes.

evo2_7b, bf16, no quantization (so the effect is isolated from compression):

| chunk | boundaries | total logL | Δ vs exact | Pearson vs exact |
|------:|-----------:|-----------:|-----------:|-----------------:|
|   256 |         31 | −13641.75  |   −6238.58 | **−0.033** |
|   512 |         15 | −13087.26  |   −5684.10 | **0.003** |
|  1024 |          7 | −11991.74  |   −4588.57 | **0.062** |
|  2048 |          3 | −17836.29  |  −10433.12 | **0.033** |
|  4096 |          1 | −16796.51  |   −9393.34 | **0.297** |
|  8192 |          0 |  **−7403.17** | **0.000** | **1.000** |

Read the last row first: with **zero** boundaries the chunked path reproduces the
exact forward *bit for bit*, which confirms the comparison itself is sound. Every
other row crosses at least one boundary, and the correlation with the truth
collapses to approximately zero. Perplexity roughly doubles to triples
(exact −0.904 nats/token, i.e. ppl ≈ 2.5, consistent with the ~2.7 reported for
human; chunked −1.6 to −2.05, ppl 4.9–7.8).

This is **not** a subtle approximation that degrades gracefully with more
boundaries. **One boundary is already fatal** (chunk 4096, r = 0.297).

Two controls rule out the obvious alternative explanations
(`experiments/satmut/diag_noise_floor.py`):

* **Not nondeterminism.** Three identical runs at chunk 512 gave
  −712.954376 each — bit-identical, spread 0.0.
* **Not quantization.** The disagreement is *larger* with KV quantization
  disabled than with it enabled.

---

## 3. Why it happens

Evo 2 (StripedHyena) has exactly two internal modes. `HyenaCascade.forward`
dispatches between them:

```python
def forward(self, u, inference_params=None, padding_mask=None, *args, **kwargs):
    if inference_params is not None and self.layer_idx in inference_params.fir_state_dict.keys():
        return self.sequential_forward(u, inference_params)      # single-token recurrent step
    else:
        return self.parallel_forward(u, inference_params, padding_mask)  # FFT prefill
```

* **First call**: no saved state, so it takes `parallel_forward` → the FFT
  prefill. Correct.
* **Every later call**: state now exists in `fir_state_dict`, so it takes
  `sequential_forward` — the path built to advance **exactly one token**.

Our loop hands that single-token path a **4,096-token block** on every iteration
after the first. It accepts the input without error and returns plausible
numbers, but the recurrence is not advanced correctly.

**What actually happens is worse than a state error, and it is worth stating
precisely** (confirmed by running the real layers in
`experiments/satmut/test_block_continuation_layers.py`). `sequential_forward`
opens with

```python
if len(u.shape) > 2:
    u = u[:, -1]        # keep only the LAST token of the block
```

so a 4,096-token block produces a **single** output token. That length-1 tensor
then meets the residual stream in `ParallelGatedConvBlock.forward`:

```python
z = self.out_filter_dense(z) + u     # (B, 1, D) + (B, L, D)  ->  broadcasts
```

PyTorch broadcasts it silently over all L positions. So for every block after the
first, **4,095 of 4,096 tokens are never processed by any Hyena layer** — the
mixer contribution at every position is a copy of the one computed from the last
token of the block. Nothing raises. The eight attention layers keep working
normally (flash-attn's `flash_attn_with_kvcache` handles multi-token chunked
prefill correctly), which is why the output stays in a plausible numeric range
instead of collapsing to obvious garbage. That combination — 42 of 50 blocks
degenerate, 8 correct, no error — is what a Pearson of ~0 against the exact
forward looks like.

Running the layers directly reproduces it: a 256-token sequence fed as four
64-token blocks comes back with **67** output positions (64 + 1 + 1 + 1).

`vortex/model/generation.py` confirms the model is only ever driven two ways:
one parallel pass over the whole prompt, then single-token steps advancing
`seqlen_offset` by exactly 1. For prompts too large to prefill, vortex's own
answer is `force_prompt_threshold`, documented as *"avoids OOM errors through
teacher forcing"* — i.e. it teacher-forces the remainder **one token at a time**.
There is no supported multi-token continuation.

Crucially, the four routines that *would* provide one are unimplemented upstream
(`vortex/model/engine.py`):

```python
def prefill_via_fir_caching(...):        raise NotImplementedError(":)")
def prefill_via_hybrid_recurrence(...):  raise NotImplementedError(":)")   # "recurrence-convolution over blocks"
def prefill_via_scan(...):               raise NotImplementedError
def prefill_via_canonical_fft(...):      raise NotImplementedError(":)")
```

and the two that do work both start from a zero state, so neither can continue
from a previous block:

```python
state = 0 * x1v_[:, :, 0]        # prefill_via_direct_recurrence
```

**Attribution.** The missing block-wise prefill is an upstream gap in
Evo 2/vortex. Our bug was calling the model as though that gap were filled. The
131k ceiling in §5 is a separate, unrelated PyTorch limitation.

---

## 4. Why the biological experiments were wrong

All of our long-context scoring used this loop
(`experiments/ciliate_codon/run_ciliate.py`,
`experiments/dnalongbench_etgp/eval_eqtl_zeroshot.py`,
`experiments/long_genes/run_long_genes.py`):

```python
while pos < L:
    end = min(pos + chunk, L)
    logits, _ = model(ids[:, pos:end], inference_params_dict=ip)   # a 4096-token BLOCK, every iteration
    for k in ("mha", "hcl", "hcm", "hcs"):
        ip[k].seqlen_offset = end
    _flush_kv_residual(model)
    pos = end
```

Every iteration feeds a block. We never stepped one token at a time anywhere. So
the first block was correct and blocks 2…N were not.

Affected runs, all at contexts far beyond one chunk:

| experiment | context | outcome we observed |
|---|---|---|
| ciliate genetic-code recognition | 180k / 300k / 450k | degraded with more context |
| eQTL causal-variant delta scoring | 450k | chance-level (AUROC ≈ 0.46–0.51) |
| long-gene completion (C vs 2C) | 180k / 360k | mixed, no clean trend |
| DNALongBench contact map (earlier pilot) | 262k / 524k / 1M | SCC ≈ 0 |

These results say nothing about the underlying biology. The model was being fed
corrupted state, so the experiments were never a valid test of whether Evo 2
benefits from long context.

**Re-running them was expensive; with the fix it should not be.** Token-by-token
stepping at ~7 tokens/s costs roughly 10 hours *per sequence* at 300k context,
which made these designs impractical. The §6 fix restores prefill speed, so they
become affordable again — but every number in the table above has to be
regenerated before any of it is quoted.

---

## 5. What is NOT affected

**Accuracy results in the paper are safe.** Every accuracy evaluation is
short-context and single-pass, so no boundary is ever crossed:

* T4 phage and cross-taxa perplexity — 512 bp windows
* GUE classification, splice-site prediction — short sequences
* BRCA1 zero-shot VEP — one 8,192 bp `evo2_obj(input_ids)` call, no chunking
  (`benchmarks/eval_evo2_variant_effect.py`)
* Gene completion — goes through the official `generate()` path

Independently reproduced here: 40B mean log-likelihood −0.8663 (bf16) vs −0.8695
(int4 weights), a 0.37 % difference.

**Memory results are essentially valid too.** The large FFT workspace is
allocated only on the *first* call (all later calls take the recurrent path), so
the memory profile of chunked prefill — one FFT, then a growing KV cache — has
the same shape as the correct method. Resident memory (weights + KV) does not
depend on how the tokens were fed in at all.

One caveat for Fig 3A: the "bf16 OOMs at 1M" point was measured at chunk 2048,
where that first FFT buffer tipped it over. bf16 resident at 1M is 360 GB against
a 383 GB pool — genuinely marginal, but with a smaller first block it may fit. The
honest statement is "94 % of the pool and fails in practice," not "cannot run."

**What changed is the time — and the §6 fix changes it back.** Using only the
paths vortex supports, a megabase is one parallel prefill plus ~1M single-token
steps: about **40 hours** at the measured 7.2 tokens/s (~35 h if the first 131k is
prefilled in parallel), against the ~15 minutes chunked prefill appeared to take.
That ~160× gap was the cost of correctness, and it is the regime our KV
compression and fused decode kernel address — without int4 KV the 1M cache does
not fit, and without the fused kernel the stepping is 8.2× slower still
(~14 days). With block continuation working, long context runs at prefill speed
again *and* correctly, so the 40-hour figure is a description of the unfixed
code, not a standing constraint.

Note the memory profile does shift with the fix: every block now takes the
parallel path, so the FFT workspace is allocated once per block rather than once
per sequence. It is bounded by the chunk size, which is the whole point of
chunking, but Fig 3-style numbers should be re-measured under the fixed path
rather than carried over.

---

## 6. The fix: implement block-wise continuation

The Hyena IIR recurrence is **linear**:

```
state_{t+1} = pole · state_t + x_t
```

so advancing it across a block of L tokens from a non-zero initial state
decomposes exactly:

```
state_L = pole^L · state_0  +  (what the existing FFT already computes assuming state_0 = 0)
```

The existing `prefill_via_modal_fft` gives the second term; what is missing is the
`pole^L · state_0` correction and the matching per-position correction on the
block's outputs.

The FIR layers need a fix too, though a trivial one. `fir_state_dict` /
`fir_inner_state_dict` do hold the right history, but only `step_fir` ever reads
them — `parallel_fir` zero-pads (`conv1d(..., padding=fir_length-1)`), so a block
starts every filter from silence. Since the receptive field is finite, prepending
the saved trailing samples and dropping the corresponding outputs is exact, with
no math required.

**Status: implemented and verified end to end** on the real 7B model. See
"Acceptance test" below for the numbers. Enable it with
`install_block_continuation(model)` from `turboquant/block_continuation.py`.

The math is unit-tested standalone in the same module. Running it directly
compares one corrected block call against L single-token steps from the same
starting state:

| shape | stock (zero-state) max err | with correction |
|---|---|---|
| D=8, S=16, L=64  | 2.57 | **2.3e-06** |
| D=4, S=8, L=256  | 1.02 | **1.8e-06** |
| D=16, S=16, L=16 | 1.80 | **1.2e-06** |

So the decomposition is exact. Wiring it into the model touches three call
sites:

1. `engine.parallel_iir` — add `y_corr` to the convolution output *before* the
   `y = (y + x1v * D) * x2` line, and add `pole^L * s0` to the state that
   `prefill_via_modal_fft` stores.
2. `engine.parallel_fir` (short filter, 3 taps) — prepend the saved
   `fir_state` instead of zero-padding `conv1d(..., padding=fir_length-1)`.
3. `engine.parallel_fir` (inner filter: hcm 128 taps, hcs 7 taps) — same
   change on the fftconv branch, reading `fir_inner_state_dict`.

All three are required for end-to-end correctness: hcl layers use the IIR, hcm
and hcs use the inner FIR, and every Hyena layer uses the short FIR. Fixing a
subset leaves the model wrong.

`turboquant/block_continuation.py` provides `install_block_continuation()`,
which patches `parallel_fir` (state prepend, short + inner), `parallel_iir`
(output + state correction), and `HyenaCascade.forward` (routing).

Three things had to be right, and the first two were wrong on the first passes —
worth recording so they are not rediscovered:

1. **Routing.** Patching `parallel_fir`/`parallel_iir` alone is a silent no-op.
   `HyenaCascade.forward` sends a layer to `sequential_forward` as soon as state
   exists for it, so blocks 2…N never reach `parallel_forward` and the
   corrections are dead code. The dispatch has to be patched too: multi-token
   inputs go to `parallel_forward` (now state-aware), single tokens keep taking
   the cheaper recurrent step. This was only detectable because the result was
   *bit-identical* to the unfixed baseline — plausible-looking output would have
   hidden it.

2. **Layout.** `parallel_iir` does its work in `(B, D, L)` but ends with
   `return y.permute(0, 2, 1)`, so it hands back `(B, L, D)`. The correction is
   naturally built in `(B, D, L)` (it is gated by `x2`, which is `(B, D, L)`), so
   it must be permuted before being added. Getting this wrong raised
   `size of tensor a (4096) must match tensor b (512) at dim 2` — 4096 being 7B's
   hidden size, 512 the chunk. `parallel_iir` now raises a labelled error if the
   two shapes disagree, rather than risking a silent broadcast.

3. **Real, not complex, arithmetic.** `log_poles` and `residues` are real
   parameters; the state only looks complex because `prefill_via_modal_fft`
   routes it through an FFT and then casts back with
   `state[..., L-1].to(torch.float32)`. The correction is therefore all real.

### Acceptance test

§2's table recomputed with the patch installed — same locus, same model, same
exact reference (`evo2_7b`, 8,192 bp of BRCA1, exact total log L = −7403.1685):

| chunk | boundaries | delta vs exact (before) | pearson (before) | delta vs exact (**after**) | pearson (**after**) |
|---|---|---|---|---|---|
| 512  | 15 | −5684.10 | 0.003 | **+0.0571** | **0.999940** |
| 1024 |  7 | −4588.57 | 0.062 | **−0.3120** | **0.999920** |
| 2048 |  3 | −10433.12 | 0.033 | **−0.2368** | **0.999934** |
| 4096 |  1 | −9393.34 | 0.297 | **+0.6660** | **0.999937** |
| 8192 |  0 | 0.00 | 1.000 | **0.0000** | **1.000000** |

And the same on **evo2_40b** (4× H100, bf16, exact total log L = −7043.0923):

| chunk | boundaries | delta vs exact | nats/token | pearson | max\|err\| |
|---|---|---|---|---|---|
| 512  | 15 | −0.0791 | −1.0e-05 | 0.999860 | 0.2634 |
| 1024 |  7 | −0.6318 | −7.7e-05 | 0.999832 | 0.4707 |
| 2048 |  3 | −1.8823 | −2.3e-04 | 0.999835 | 0.3329 |
| 4096 |  1 | −2.8794 | −3.5e-04 | 0.999836 | 0.3286 |
| 8192 |  0 | **+0.0000** | 0 | **1.000000** | **0.0000** |

The zero-boundary row is bit-exact on both models. That is the regression
control: the patch does not perturb the path that was already correct.

**On the residual drift — and how much it varies by locus.** The BRCA1 table
above happens to show drift growing monotonically (0.08 → 2.88 nats) as
boundaries *fall* (15 → 1). That pattern does not reproduce: repeating the sweep
at two other loci (`--start-offset 200000 / 400000`) gives non-monotone drift
that changes sign between chunk sizes. So the drift is round-off, not a residual
boundary bias — every locus lands at exactly 0.0000 with zero boundaries, and
error that grew with *fewer* boundaries would make no mechanical sense anyway.
The specific reason one decomposition rounds differently from another is not
pinned down; do not quote a mechanism.

Fidelity is locus-dependent, and the dependence is worth planning around:

| locus (mean log L/token) | pearson | max\|err\| | worst drift |
|---|---|---|---|
| BRCA1, chr17:43,044,295 (−0.860) | 0.99983 – 0.99986 | 0.26 – 0.47 | −2.88 |
| +200 kb (−0.192, low-entropy) | 0.99651 – 0.99708 | 0.82 – 1.10 | −8.65 |
| +400 kb (−0.789) | 0.99993 – 0.99995 | 0.18 – 0.43 | −0.90 |

The low-entropy locus is the outlier by roughly an order of magnitude on every
measure. Part of the correlation drop is Pearson's denominator shrinking when
per-token log-probs barely vary, but `max|err|` genuinely triples, so it is not
only an artifact. **Budget for r ≈ 0.997, not 0.9999, on repetitive or
low-complexity sequence.** Total drift stays ≤1.1e-3 nats/token everywhere
measured.

### At long context — does error accumulate?

The 8,192 bp tables above cross at most 15 boundaries. The question that decides
whether any of this is usable is whether error grows with the *number* of
boundaries, because a megabase at chunk 8,192 crosses 127 of them and at chunk
1,024 crosses 1,023.

An exact reference requires a single unchunked forward, which is capped at
131,072 (7B) and 65,536 (40B) — so those lengths are the longest at which the
question can be answered against truth at all.

| model | seq len | chunk | boundaries | delta | nats/token | pearson | tok/s |
|---|---|---|---|---|---|---|---|
| 7B  | 131,072 | 1,024  | **127** | −4.27 | −3.3e-05 | 0.999849 | 4,789 |
| 7B  | 131,072 | 4,096  | 31 | +1.90 | +1.4e-05 | 0.999850 | 4,851 |
| 7B  | 131,072 | 16,384 | 7  | +3.52 | +2.7e-05 | 0.999853 | 4,400 |
| 40B | 65,536  | 1,024  | 63 | −4.80 | −7.3e-05 | 0.999829 | 1,670 |
| 40B | 65,536  | 4,096  | 15 | −1.82 | −2.8e-05 | 0.999828 | 1,680 |

**Error does not accumulate.** Pearson is flat to five decimals across 7, 31 and
127 boundaries on the 7B (0.99985) and across 15 and 63 on the 40B (0.99983),
and the drift changes sign. Crossing 16× more boundaries costs nothing
measurable. Per-token drift at 131 kb is *smaller* than at 8 kb, which is what
you expect if the error is round-off rather than a per-boundary leak.

**Throughput is the other half.** 4,789 tok/s (7B) and 1,670 tok/s (40B) against
the 7.2 tok/s of single-token stepping: 665× and 232×. That turns a megabase on
the 40B from ~40 hours into ~10 minutes.

One caveat on reading these rows: the `peakGB` column in this table is not
trustworthy. `torch.cuda.reset_peak_memory_stats()` with no argument resets only
the *current* device, so on a sharded model the other ranks kept reporting the
exact reference's peak. Fixed in the script (it now resets every device), but the
runs above predate the fix.

### Under the compressed tiers

Same 40B, same BRCA1 locus, with the fix installed, at each deployment tier. The
reference is always the full-precision unchunked forward, so a quantized tier is
measured against real truth rather than against itself. The **chunk = 8,192 row
has zero boundaries, so it is that tier's own quantization floor**; the excess
over it at smaller chunks is what chunking costs.

| tier | quantization floor (chunk 8192) | chunked (512 – 4096) |
|---|---|---|
| baseline (bf16 W, bf16 KV) | bit-exact: r = 1.000000, max\|err\| 0.0000 | r 0.99983 – 0.99986, max\|err\| 0.26 – 0.47 |
| tier1 (bf16 W, int4 KV) | r = 0.999859, max\|err\| 0.3114 | r 0.99842 – 0.99933, max\|err\| 0.69 – 1.10 |
| tier2 (int4 W, int4 KV) | r = 0.992944, max\|err\| 2.1160 | r 0.99121 – 0.99248, max\|err\| 2.08 – 2.18 |

Read across the rows and the error budget is clear:

* **The fix holds under compression.** Every tier stays at r ≥ 0.991 with drift
  ≤3.6e-3 nats/token, against r ≈ 0 unpatched.
* **At tier2, chunking is nearly free.** Quantization alone already costs
  max|err| 2.12 with zero boundaries; adding 15 boundaries moves it to 2.18. If
  you are running int4 weights, chunking is not what limits your fidelity.
* **At tier1, chunking is the larger term.** The int4-KV floor is max|err| 0.31,
  and chunking roughly triples it. Still r ≥ 0.998, but this is the tier where
  a larger chunk actually buys accuracy.

Two practical notes from running these:

* **Compute the exact reference before installing any KV tier.**
  `install_fused_kv_quant` swaps in a `FlashSelfAttention` whose `forward` does
  not accept `key_padding_mask`, which vortex's stateless path passes — so the
  full-precision reference cannot be computed at all once the tier is live. Any
  eval script that mixes `stateless_forward` with a fused-KV tier will hit this.
* **tier1 needs more memory than baseline at the largest chunk**, despite a 4×
  smaller KV cache: chunk 8,192 OOMs in a sweep where plain bf16 fits, so the
  row above was measured in its own process. The fused int4 path evidently
  carries extra buffers (packed store + bf16 residual + fused-prefill
  workspace). tier2 has no such trouble, since int4 weights free ~48 GB.

Reproduce (drop `--model evo2_40b` for the 7B, add `--tier tier1|tier2`):

```bash
python experiments/satmut/diag_chunk_fidelity.py --model evo2_40b \
    --seq-len 8192 --chunk-list 512,1024,2048,4096,8192 --block-fix
```

The 40B needs a clean allocator to reach the largest chunk: the sweep frees
between chunk sizes, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
helps. Without the free, chunk 8192 OOMs on a size it runs fine from cold.

Add `--start-offset N` to move the locus, which is how the table above was
produced.

A second, cheaper test isolates the layers themselves — no GPU, no checkpoint,
random weights, all three Hyena variants, against a single parallel pass:

```bash
python experiments/satmut/test_block_continuation_layers.py
```

| layer | stock, 4 blocks of 64 | with fix (rel err) | blocks → single tokens |
|---|---|---|---|
| hcl (long IIR) | returns 67 of 256 positions | 2.8e-07 | 2.8e-07 |
| hcm (inner FIR, 128 taps) | returns 67 of 256 positions | 2.5e-07 | 2.7e-07 |
| hcs (inner FIR, 7 taps) | returns 67 of 256 positions | 0.0 | 1.2e-07 |

The last column matters for generation: it blocks up to L−8, then takes eight
single-token steps, confirming the state a corrected block *hands on* is the
state `step_iir`/`step_fir` expect.

### What this buys

* Multi-token blocks are now correct, so long context runs at **prefill speed**
  instead of ~40 hours of single-token stepping at 1M.
* The biological experiments in §4 become affordable to re-run — and they must
  be re-run before any of their conclusions are quoted.
* It fills in the routine (`prefill_via_hybrid_recurrence`) that upstream left as
  a stub.

Caveats before leaning on it: verified on `evo2_7b` at 8,192 bp with baseline
(bf16) weights. Not yet verified on the 40B, at long context, or in combination
with int4 weights/KV — those are the next runs, and the same
`diag_chunk_fidelity.py` invocation covers them (`--model evo2_40b`,
`--tier tier2`).

---

## 7. Maximum single parallel prefill

The largest sequence that can be fed in **one** call — the fast path, valid by
construction because it crosses no boundary.

**evo2_7b, 1× H100 95.8 GB** (`experiments/satmut/results/valid_context_7b.json`):

| length | bf16 peak | int4-W peak |
|-------:|----------:|------------:|
|  32,768 | 32.60 GB | 24.94 GB |
|  65,536 | 51.97 GB | 44.30 GB |
| 131,072 | 90.70 GB | 83.04 GB |
| 262,144 | RuntimeError | RuntimeError |

Both cap at **131,072**, and the failure at 262,144 is *not* OOM — it is a
PyTorch 32-bit indexing overflow (`canUse32BitIndexMath`), the same limit that
caps batch size at 8 in `benchmarks/bench_vep_batching.py`.

Note the activation cost: 90.70 − 13.17 = 77.5 GB (bf16) and 83.04 − 5.57 =
77.5 GB (int4) — **identical**. At long context memory is dominated by
activations, which quantization does not touch; weight compression saves a flat
amount (7.6 GB on 7B, ~45 GB on 40B) regardless of length. This is why weight
compression buys proportionally the most at *short* context (int4 peak is 47 % of
bf16 at 2k, but 92 % at 131k).

**evo2_40b, 4× H100** (`experiments/satmut/results/valid_context_40b.json`).
Peaks are summed over the four devices, following the paper's convention:

| length | bf16 peak | int4-W peak | difference |
|-------:|----------:|------------:|-----------:|
|   8,192 | 120.77 GB |  72.82 GB | 47.95 GB |
|  16,384 | 159.04 GB | 111.09 GB | 47.95 GB |
|  32,768 | 235.58 GB | 187.63 GB | 47.95 GB |
|  65,536 | 388.64 GB | 340.69 GB | 47.95 GB |
| 131,072 | RuntimeError | RuntimeError | — |

Resident after load: 82.24 GB (bf16) vs 34.46 GB (int4-W).

**Both configurations cap at 65,536 tokens, and weight compression buys no extra
prefill length whatsoever.** The ceiling is again the PyTorch 32-bit indexing
overflow, not memory — and it arrives at half the 7B length because 40B's hidden
dimension is 8192 versus 4096, so the offending tensor reaches 2^31 elements
twice as fast.

The difference between the two columns is **exactly 47.95 GB at every length**,
matching the weight delta (82.24 − 34.46 = 47.78 GB) to within rounding. That
constant offset is direct evidence that activations are identical in the two
configurations and only the weights differ: quantization shifts the whole curve
down by a fixed amount, it does not change its slope. Since the binding
constraint is an index limit rather than memory, that fixed offset translates
into zero additional context.

**Token-by-token verified on 40B as well** (prefill 1024, then single-token
steps, 2048-token sequence): exact −1681.787 vs stepwise −1681.260, i.e.
Δ = +0.528 over 2047 tokens, r = 0.99956, mean |err| = 0.0085. Valid — and about
four orders of magnitude closer to exact than chunked prefill's −4589 to −10433.

---

## 8. Reproducing

```bash
# chunked vs exact, by chunk size (add --block-fix for the corrected path)
python3 experiments/satmut/diag_chunk_fidelity.py --model evo2_7b --seq-len 8192 \
    --chunk-list 256,512,1024,2048,4096,8192

# the block-continuation fix, layer by layer -- CPU only, no checkpoint, seconds
python3 experiments/satmut/test_block_continuation_layers.py

# the IIR decomposition on its own, against L single-token steps
python3 turboquant/block_continuation.py

# determinism and chunk-dependence controls
python3 experiments/satmut/diag_noise_floor.py

# max single parallel prefill + verification that token-by-token reproduces exact
python3 experiments/satmut/diag_valid_context.py --model evo2_40b \
    --configs bf16,int4w --len-list 8192,16384,32768,65536,131072,262144
```

---

## 9. Recommended actions

1. **Never call the chunked-prefill helper without
   `install_block_continuation(model)`.** Unpatched, anything longer than a
   single parallel call must use prefill + single-token stepping.
2. **Extend the §6 verification to long context.** Confirmed on `evo2_7b` and
   `evo2_40b`, three loci, all three tiers — but only at 8,192 bp. The memory
   argument for chunking lives at 100 kb+, which is exactly where it is
   untested, and error that is harmless over 4 boundaries may not be over 400.
3. **Re-run the long-context biological experiments** (§4) with the fix
   installed. Every result in that table is currently void.
4. **Re-frame the long-context claim** in the paper. Compression does not extend
   how much can be prefilled in one pass (activation-bound, §7). Its value is
   that in the token-by-token regime memory is *only* weights + KV — both of
   which we compress — taking 1M context from ~360 GB (4 GPUs, marginal) to
   ~107 GB (comfortable on 2), with the fused kernel keeping the step rate at
   bf16 parity.

---

## 10. Effective context: how much long range does Evo 2 actually use?

With chunked prefill correct, this became measurable for the first time. Past the
single-pass ceiling the only correct path was ~7 tok/s stepping, and the fast path
was silently wrong -- so nobody could previously run this experiment honestly.

`experiments/context_util/context_utilization.py` fixes an 8 kb target window and
scores the *byte-identical* window while varying only what precedes it:

* **real** -- the genuine N bp immediately 5' of the target
* **shuffled** -- those same bases, mononucleotide-shuffled (composition kept)
* **foreign** -- real DNA of length N from a different chromosome

8 loci on distinct chromosomes (BRCA1/EGFR/KRAS/BRCA2/TERT/CDKN2A/PTEN/WT1),
paired Wilcoxon across loci. `evo2_7b`, all arms:

| N | real | shuffled | foreign | real - none | p | real - foreign | p |
|---|---|---|---|---|---|---|---|
| 0 | -1.06749 | - | - | - | - | - | - |
| 8,192 | -1.06469 | -1.08394 | -1.06824 | +0.00280 | 0.0078 | +0.00354 | 0.0078 |
| 32,768 | -1.06444 | -1.09107 | -1.06982 | **+0.00305** | 0.0078 | +0.00538 | 0.0078 |
| 131,072 | -1.06479 | -1.10038 | -1.07009 | +0.00270 | 0.0156 | +0.00530 | 0.0078 |
| 524,288 | -1.06531 | -1.10764 | -1.07106 | +0.00218 | 0.0234 | +0.00575 | 0.0078 |

p = 0.0078 is the floor for n = 8, i.e. as significant as 8 loci allow.

**Evo 2's benefit from real context peaks at ~32 kb and then DECLINES.** At
524 kb it retains ~70 % of its 32 kb value and its significance decays. The 40B
behaves identically (3 loci: +0.00470 / **+0.00521** / +0.00501 at 8k / 32k /
131k), so the larger model is no better at distance.

**Read the `real - none` column, not `real - shuffled`.** The real-vs-shuffled gap
widens monotonically (+0.019 -> +0.042) and, on its own, appears to say that more
context helps more. It does not: the *control* is degrading, because ever more
out-of-distribution sequence is being fed in. Against a no-context baseline the
curve is flat then falling. The `foreign` arm is what separates "the model uses
THIS locus" from "the model likes realistic DNA" -- at 8 kb, foreign DNA is nearly
as good as the true locus (+0.0009 on the 40B), so most of the short-range benefit
is not locus-specific at all.

```bash
python experiments/context_util/context_utilization.py --model evo2_7b --loci 8 \
    --n-list 0,8192,32768,131072,524288 --arms real,shuffled,foreign \
    --out results/full_7b.json
python experiments/context_util/analyze_context.py results/full_7b.json
```

**Consequence.** The long-context biological experiments in §4 were doomed
independently of the chunking bug -- there was no signal beyond ~32 kb to find.
"More context -> better biology" is not supportable for this model.

---

## 11. Long-range regulatory signal: a controlled negative

`experiments/enhancer/enhancer_dependency.py` asks the sharper question an
8 kb average cannot: does a *specific*, experimentally validated enhancer register
at distance? Using the DNALongBench CRISPRi K562 set (133 validated
enhancer->gene links), each TSS window is scored twice -- real sequence, versus
the same sequence with **only the enhancer's bases shuffled in place** (same
length, position, distance, composition). Effect = logL(wt) - logL(scrambled),
compared against distance-matched non-enhancers.

Three constraints the design must respect, each of which silently fabricates a
result if ignored:

1. **Evo 2 is causal.** An enhancer 3' of the TSS is invisible when the TSS is
   scored. Only upstream-in-transcription-orientation pairs are usable (80 of
   133); minus-strand windows are reverse-complemented so the enhancer is read
   first.
2. **Positives are much closer than negatives** (median 26 kb vs 183 kb). Effect
   size falls with distance, so an unmatched comparison hands positives a win on
   geometry alone. Negatives are drawn from the same log-distance stratum.
3. **Enhancers overlapping the scored window must be dropped** -- scrambling
   would rewrite the bases being scored, measuring self-corruption rather than
   regulation. (Caught in validation on NUCB1.)

Result, `evo2_7b`, 63 positives / 57 matched negatives:

| distance | validated enhancers | matched controls |
|---|---|---|
| 0-10 kb | +0.00045 (n=23) | +0.00014 (n=16) |
| 10-50 kb | +0.00021 (n=23) | -0.00004 (n=25) |
| **50-150 kb** | **-0.00007 (n=17)** | +0.00005 (n=15) |

Pair-level Mann-Whitney p = 0.043, but that is pseudo-replicated: several
enhancers share a gene and therefore a TSS window. **Aggregated to one value per
gene, p = 0.102 -- not significant.** The effect is present at short range,
halves by 10-50 kb, and is gone beyond 50 kb.

Consistent with §10. Effect sizes are ~1e-4 and n = 17 in the far stratum, so
this is "weak or absent", not proof of zero -- but nothing here supports a
long-range biological claim.

---

## 12. Running the 40B on ONE GPU (the accessibility result)

**bf16 `evo2_40b` cannot be loaded on a single 80 GB GPU at all** -- measured, it
OOMs during construction at block 46 of 50. int4 can, and this is the practical
payoff of compression.

| config (one 80 GB card) | outcome |
|---|---|
| bf16 weights | **OOM during load, block 46/50** |
| int4 weights (from checkpoint) | 33.8 GB resident, 36.5 GB peak load, **49.0 GB peak scoring** |
| | 32,767 tokens in 37 s = **883 tok/s**, mean logL -0.83931 |

**Verified faithful.** The identical 32,768 tokens through the standard 4-GPU
tier2 path give mean logL **-0.83931**, matching to five decimals. And the single
GPU is *faster*: 883 tok/s versus 641 tok/s on 4 GPUs (8.9 GB/GPU resident,
18.2 GB/GPU peak), because splitting the model across devices ships activations
between cards at every layer and that costs more than the parallelism returns.
For single-sequence scoring, one card is the better configuration, not merely a
sufficient one.

Context is set to 32,768 = the model's **measured effective context** (§10), so
this is Evo 2's full useful capability on one card, not a truncated version.

### Using it

```python
from turboquant.int4_checkpoint import load_int4_model
model, tok = load_int4_model("evo2_40b", "evo2_40b_int4.pt", device="cuda:0")
```

Build the checkpoint once, on a machine large enough to hold bf16:

```bash
python experiments/single_gpu/make_int4_checkpoint.py \
    --model evo2_40b --out evo2_40b_int4.pt      # 33.8 GB, ~18 min

CUDA_VISIBLE_DEVICES=0 python experiments/single_gpu/load_int4_single.py \
    --model evo2_40b --ckpt evo2_40b_int4.pt \
    --budget-gb 80 --context 32768 --chunk 1024
```

**Quote 33.8 GB, not 16.8 GB.** Only `nn.Linear` layers are quantized: 208 of
them, 65.3 -> 16.8 GB (3.88x). Hyena filters, embeddings and norms are not Linear
and stay bf16 -- the other ~17 GB. Half the resident footprint is therefore still
uncompressed, which is real headroom for future work.

### Four traps, all hit and fixed

1. **The model cannot be built CPU-resident.** TransformerEngine asserts
   `torch.cuda.is_available()` at layer construction, and `fixup_fp8_extra_states`
   asserts its fp8 metadata sits on the parameters' device while TE forces that
   metadata onto CUDA. vortex's own `device="cpu"` branch is dead code -- it
   computes `"cpu"` and then calls `torch.cuda.device("cpu")`, which raises.
   *Solution:* intercept `move_to_device` and swap each block's Linears to int4
   immediately after that block is built. Peak = int4-so-far + one bf16 block.
2. **Do not use `model.load_state_dict` for the non-quantized tensors.** It
   *copies* and keeps the DESTINATION dtype, whereas vortex's
   `custom_load_state_dict` replaces `.data` and adopts the CHECKPOINT dtype.
   Copy-semantics leave a fresh RMSNorm scale in fp32, its output promotes to
   fp32, and the next TE Linear dies with "Data types for parameters must match".
3. **But you still need a `load_state_dict` pass afterwards**, for
   `_extra_state` -- TE's fp8 scaling metadata, which is neither parameter nor
   buffer and moves only via `set_extra_state`. Without it, 42 `projections`
   layers run on default fp8 scales and produce a plausible but WRONG answer
   (-0.83951 instead of -0.83931; the same order as the context effects in §10).
4. **Chunk size, not the KV cache, drives peak memory.**
   `prefill_via_modal_fft` allocates `hidden x state_dim x fft_size` complex
   ~= 8.6 GB at chunk 4096 for the 40B. Chunks 8192 and 4096 both OOM at 80 GB;
   chunk 1024 peaks at 49 GB. This is free: error is independent of boundary
   count (§6).

Also: moving a 4-GPU-sharded vortex model onto one card with `.to("cuda:0")` does
**not** work -- device-bound state (flash-FFT plans, cuBLAS workspaces) is not
relocated, and you get an illegal memory access inside `prefill_via_modal_fft`.

---

## 13. Known open items

1. **An int32 overflow in our own fused int4-KV kernel caps context at
   524,288 tokens.** `fused_kv_attention.py` computed
   `row_byte = (offs_n * Hkv + hk) * HD2` with an int32 `offs_n`; for the 40B
   (`Hkv*HD2 = 4096`) that crosses 2^31 at exactly 524,288 tokens, wraps
   negative, and faults with an illegal memory access. The per-segment kernel
   already promoted to int64; the batched contiguous-view kernel -- the one used
   at long context -- did not. Patched to int64, **not yet bisected**
   (262,144 should pass and 524,288 should fail on the unpatched kernel).
2. **The previously reported ~756k max context must be re-measured.** It was
   obtained on the broken path, where blocks after the first skipped their FFT
   entirely and therefore used materially less transient memory than correct
   execution. Correct execution may reach *less*. Do not quote 756k.
3. **All long-context biological results (§4) remain void** until re-run.
