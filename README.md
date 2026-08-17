# TurboQuant-Bio

Run [Evo 2](https://github.com/ArcInstitute/evo2) compressed, on less hardware,
and — importantly — **correctly**.

* **Evo 2 40B on a single 80 GB GPU**, where bf16 cannot be loaded at all
  (33.8 GB resident, 883 tok/s). It is also *faster* than running across 4 GPUs.
* **A correctness fix** for Evo 2's chunked-prefill path, which is silently
  wrong upstream — it returns plausible numbers that are essentially
  uncorrelated with the truth on any sequence longer than one chunk.
* **A measurement of Evo 2's effective context**: the benefit from real
  upstream context peaks near **32 kb** and then *declines*.

Full findings, tables and derivations: **[FINDINGS.md](FINDINGS.md)**.

---

## Install

```bash
git clone https://github.com/michalispatsakis/turboquant-bio.git
cd turboquant-bio
pip install -e .
```

You need a working Evo 2 install first (`evo2`, `vortex`, and for the 40B
`transformer_engine`). TurboQuant-Bio sits on top of it and does not try to
manage those environment-specific packages for you.

## Quickstart

```python
from turboquant import load_evo2, score

# 7B, unmodified weights, with the chunked-prefill fix applied
model, tok = load_evo2("evo2_7b", tier="baseline")
print(score(model, tok, my_sequence, model_name="evo2_7b"))
```

```python
# 40B on ONE 80 GB GPU. The int4 checkpoint is downloaded and cached on first use.
model, tok = load_evo2("evo2_40b", tier="tier2", device="cuda:0")
print(score(model, tok, my_sequence, model_name="evo2_40b"))
```

`score()` returns mean log-likelihood per base (higher is better; human DNA
typically lands near −0.85 to −1.1). `return_per_token=True` gives the per-base
array.

---

## Tiers

| tier | weights | KV cache | needs a checkpoint? | use when |
|---|---|---|---|---|
| `baseline` | bf16 | bf16 | no | you have the memory and want reference numbers |
| `tier1` | bf16 | int4 | **no** | long context on a big card: 4× smaller KV cache, weights untouched |
| `tier2` | int4 | int4 | yes (auto-downloaded) | the model does not otherwise fit — e.g. 40B on one GPU |

Tier 1 needs no download because KV quantization happens at runtime.

Accuracy cost, measured against a full-precision unchunked forward with **zero**
chunk boundaries, so this isolates each tier's own quantization error (40B, 8 kb):

| tier | Pearson vs exact | max abs error |
|---|---|---|
| baseline | 1.000000 (bit-exact) | 0.0000 |
| tier1 | 0.999859 | 0.31 |
| tier2 | 0.992944 | 2.12 |

On the 7B at 8 kb, through this API against a known exact value of −0.90382:
baseline −0.90374, tier1 −0.90361, tier2 −0.90698.

---

## Hardware

Measured on H100s, capped to an 80 GB budget so the numbers transfer to the card
most labs have. "Peak" is at 32 kb context.

| model | tier | GPUs | resident | peak | throughput |
|---|---|---|---|---|---|
| evo2_40b | baseline | 1 × 80 GB | — | — | **cannot load** (OOM at block 46/50) |
| evo2_40b | tier2 | 1 × 80 GB | 33.8 GB | 49.0 GB | **883 tok/s** |
| evo2_40b | tier2 | 4 × 80 GB | 8.9 GB/GPU | 18.2 GB/GPU | 641 tok/s |
| evo2_7b | baseline | 1 × 80 GB | 3.3 GB | comfortable | 4,789 tok/s |
| evo2_7b | tier2 | 1 × 16 GB | 5.5 GB | comfortable | — |

**One GPU is faster than four** for single-sequence scoring. Sharding ships
activations between cards at every layer, and that costs more than the
parallelism returns. Use one card unless you need the memory.

The 7B already fits a single A100/H100 in bf16 — use tier2 for it only on
smaller cards.

---

## Checkpoints

Pre-quantized int4 weights:
[michalakis99/turboquant-evo2-int4](https://huggingface.co/michalakis99/turboquant-evo2-int4)

| file | size |
|---|---|
| `evo2_40b_int4.pt` | 33.8 GB |
| `evo2_7b_int4.pt` | 5.4 GB |

Fetched automatically by `load_evo2(..., tier="tier2")`. To build your own on a
machine large enough to hold bf16:

```bash
python tools/make_int4_checkpoint.py --model evo2_40b --out evo2_40b_int4.pt
python tools/make_int4_checkpoint.py --model evo2_7b  --out evo2_7b_int4.pt
```

Note the 40B checkpoint is 33.8 GB, not the 16.8 GB the Linear-only compression
ratio suggests: only `nn.Linear` layers are quantized (208 of them, 65.3 → 16.8
GB, 3.88×), while Hyena filters, embeddings and norms stay bf16.

---

## How much context should I use?

**About 32 kb.** Evo 2's benefit from real upstream context peaks near 32 kb and
then declines — measured across 8 loci on both the 7B and 40B, paired Wilcoxon
p = 0.0078 ([FINDINGS.md §10](FINDINGS.md)). Feeding 500 kb is slightly *worse*
than feeding 32 kb.

`score()` warns above ~49 kb; the warning is advice, not a limit.

## Chunk size

Defaults are chosen for you (1024 for the 40B, 4096 for the 7B). If you override
it: **chunk size, not the KV cache, drives peak memory** — the Hyena modal-FFT
buffer is ≈8.6 GB at chunk 4096 for the 40B, so 4096 and 8192 both OOM on an
80 GB card while 1024 peaks at 49 GB.

Fidelity is **independent** of chunk size (verified to 127 boundaries), so
lowering it costs nothing but wall-clock.

---

## Verifying your install

```bash
# seconds, no GPU, no checkpoint
python tests/test_release.py --level fast

# adds a pinned numerical check on real weights
export TURBOQUANT_HG38=/path/to/hg38.fa
python tests/test_release.py --level full \
    --int4-ckpt evo2_40b_int4.pt --int4-ckpt-7b evo2_7b_int4.pt
```

The pinned check matters: every bug found while building this produced
plausible-looking numbers rather than a crash, so only a pinned expected value
catches a regression.

---

## Gotchas

* **Do not call the stateless path under tier1/tier2.** The fused attention does
  not accept `key_padding_mask`, so `model(ids)` with no inference params raises
  `TypeError`. Compute any full-precision reference *before* installing a KV tier.
* **Do not move a multi-GPU model onto one card with `.to("cuda:0")`.**
  Device-bound state (flash-FFT plans, cuBLAS workspaces) is not relocated and
  you get an illegal memory access. Load on the target device instead.
* **Context beyond ~524 kb currently faults** in the fused int4-KV kernel — an
  int32 indexing overflow, patched but not yet re-verified at that scale.
* **If the checkpoint repo is private, you must be authenticated** (`hf auth
  login`). Note that setting `HF_HOME` moves where the token is looked for: a
  token written to `~/.cache/huggingface/token` is invisible if `HF_HOME` points
  elsewhere, and the download then fails with a bare `401`. Either log in after
  setting `HF_HOME`, or export `HF_TOKEN`.

---

## Credit and licence

The models are [Evo 2](https://github.com/ArcInstitute/evo2) by Arc Institute;
this project provides compression, a correctness fix, and measurements on top of
them. The published checkpoints are quantized re-encodings of Arc Institute's
weights and inherit their licence. Code here is Apache-2.0.
