# TurboQuant-Bio

Compressed [Evo 2](https://github.com/ArcInstitute/evo2) inference, so you can run
the big model on the hardware you actually have — plus a fix for a correctness
problem in Evo 2's long-sequence path.

Three things this gives you:

**The 40B model on a single GPU.** In bf16 the 40B needs about 82 GB just for
weights, so it will not load on an 80 GB card at all. With int4 weights it takes
33.8 GB and runs comfortably on one card, at 883 tokens/s. It turns out to be
*faster* on one GPU than on four, because splitting the model across devices means
copying activations between cards at every layer.

**The 7B on a single A100.** It already fits in bf16 (3.3 GB of weights), and int4
brings it down far enough for much smaller cards (5.5 GB).

**Long sequences that are actually correct.** A single forward pass in Evo 2 is
capped at 65,536 tokens for the 40B and 131,072 for the 7B. Going beyond that
means feeding the sequence in chunks — and Evo 2's chunked path is broken
upstream. It doesn't crash or warn; it just returns numbers that look reasonable
and are wrong. We fixed it, and verified that **quality is unaffected**: chunked
scoring matches a single exact pass to within round-off, and the error does not
grow as you add more chunks (checked up to 127 of them). The fix is applied
automatically — you don't have to remember anything.

Details and measurements: **[FINDINGS.md](FINDINGS.md)**.

---

## What a new user actually does

Three steps, and **no Hugging Face account is needed at anything** — every model
file involved is public and downloads anonymously.

**1. Install Evo 2.** Follow [Arc Institute's
instructions](https://github.com/ArcInstitute/evo2). This is the part that takes
effort, because `vortex`, `transformer_engine` and `flash-attn` compile against
your CUDA version. Check it worked:

```bash
python -c "from evo2 import Evo2; print('ok')"
```

**2. Install this on top.**

```bash
git clone https://github.com/Georgakopoulos-Soares-lab/turboquant-bio.git
cd turboquant-bio && pip install -e .
python tests/test_release.py --level fast      # seconds, no GPU
```

**3. Run it.** Weights download automatically on first use and are cached:

```python
from turboquant import load_evo2, score
model, tok = load_evo2("evo2_40b", tier="tier2")     # one 80 GB GPU
print(score(model, tok, my_sequence, model_name="evo2_40b"))
```

What gets downloaded depends on the tier, and it is worth knowing before you
start, because the sizes differ a lot:

| tier | what it downloads | 7B | 40B |
|---|---|---|---|
| `baseline`, `tier1` | Arc's original bf16 weights (Evo 2 fetches these itself) | ~15 GB | ~80 GB |
| `tier2` | **only** our int4 checkpoint — the original weights are never needed | 5.4 GB | 33.8 GB |

So tier 2 is not just smaller in memory, it is a much smaller download: the 40B
comes down as 33.8 GB instead of 80 GB, and works on a machine that could never
hold the bf16 model.

---

## Why Evo 2 isn't installed for you

`pip install -e .` does **not** pull in Evo 2, and that is deliberate rather than
an oversight. Evo 2 needs `vortex`, `transformer_engine` and `flash-attn`, all of
which build against your specific CUDA version and GPU architecture. Declaring
them as dependencies would let pip reinstall and break environments that already
work. `load_evo2()` checks for Evo 2 up front and tells you what to do if it is
missing.

---

## Using it

```python
from turboquant import load_evo2, score

model, tok = load_evo2("evo2_7b", tier="baseline")
print(score(model, tok, my_sequence, model_name="evo2_7b"))
```

The 40B on one GPU — the int4 weights download automatically the first time
(33.8 GB, cached afterwards):

```python
model, tok = load_evo2("evo2_40b", tier="tier2", device="cuda:0")
print(score(model, tok, my_sequence, model_name="evo2_40b"))
```

**More than one GPU, for long sequences.** Pass `device="auto"` and the int4
model is sharded across every visible GPU:

```python
model, tok = load_evo2("evo2_40b", tier="tier2", device="auto")
```

Use this once your sequence gets long. The weights fit on a single card, but the
KV cache grows with the sequence and eventually will not: scoring a complete
580 kb bacterial genome in one context peaks at about 45 GB in total. `"auto"`
spreads that across the GPUs you have, and it still loads straight from the int4
checkpoint, so the 82 GB bf16 model is never materialized.

`score()` gives you the mean log-likelihood per base; higher is better, and human
DNA usually lands around −0.85 to −1.1. Ask for `return_per_token=True` if you
want the value at every position.

There is also a small command-line example:

```bash
python examples/score_sequence.py --fasta my.fa --model evo2_40b --tier tier2
```

---

## Which tier?

| tier | weights | KV cache | download needed | when to use it |
|---|---|---|---|---|
| `baseline` | bf16 | bf16 | no | you have the memory and want reference numbers |
| `tier1` | bf16 | int4 | no | long sequences on a big card — the cache is 4× smaller, weights untouched |
| `tier2` | int4 | int4 | yes, automatic | the model doesn't otherwise fit, e.g. the 40B on one GPU — or `device="auto"` across several for long context |

Tier 1 needs no download because the cache is quantized as you go. Only tier 2
uses the pre-quantized weights.

What it costs you in accuracy, measured against an exact full-precision pass
(40B, 8 kb, no chunk boundaries so this is purely the quantization):

| tier | correlation with exact | worst single-base error |
|---|---|---|
| baseline | 1.000000 (identical) | 0.0000 |
| tier1 | 0.999859 | 0.31 |
| tier2 | 0.992944 | 2.12 |

On the 7B, against a known exact value of −0.90382: baseline −0.90374,
tier1 −0.90361, tier2 −0.90698.

---

## What runs where

Measured on H100s with the process capped to 80 GB, so the numbers apply to a
normal 80 GB card. Peak is at 32 kb of context.

| model | tier | GPUs | weights | peak memory | speed |
|---|---|---|---|---|---|
| 40B | baseline | 1 × 80 GB | — | — | **will not load** |
| 40B | tier2 | 1 × 80 GB | 33.8 GB | 49.0 GB | 883 tok/s |
| 40B | tier2 | 4 × 80 GB | 8.9 GB each | 18.2 GB each | 641 tok/s |
| 7B | baseline | 1 × A100 | 3.3 GB | comfortable | 4,789 tok/s |
| 7B | tier2 | 1 × 16 GB | 5.5 GB | comfortable | — |

---

## Long sequences

Anything longer than one forward pass (65,536 tokens on the 40B, 131,072 on the
7B) is fed in chunks, and `score()` handles that for you. Two things worth
knowing:

**Quality holds.** Chunked scoring reproduces a single exact pass to round-off,
and crucially the error does not accumulate: results are the same whether the
sequence is split into 7 chunks or 127. So there is no accuracy reason to prefer
larger chunks.

**About 32 kb of context is the sweet spot.** We measured how much Evo 2 actually
uses, and the benefit from real upstream sequence peaks near 32 kb and then gets
slightly *worse*. Feeding half a megabase is not an improvement over 32 kb.
`score()` mentions this if you go well past it, but it will still run.

Chunk size is chosen for you (1024 for the 40B, 4096 for the 7B). If you override
it, note that chunk size — not the cache — is what drives peak memory: for the
40B, chunks of 4096 or larger will exhaust an 80 GB card.

---

## Checkpoints

The int4 weights live on the Hub and are fetched automatically:
[michalakis99/turboquant-evo2-int4](https://huggingface.co/michalakis99/turboquant-evo2-int4)
(`evo2_40b_int4.pt`, 33.8 GB; `evo2_7b_int4.pt`, 5.4 GB).

Pass `int4_ckpt="/path/to/file.pt"` to use a local copy, or build your own on a
machine big enough to hold bf16:

```bash
python tools/make_int4_checkpoint.py --model evo2_40b --out evo2_40b_int4.pt
```

The 40B file is 33.8 GB rather than the ~17 GB you might expect from the 3.88×
compression, because only the linear layers are quantized — Evo 2's Hyena filters
and embeddings stay in bf16.

---

## Checking it works

```bash
python tests/test_release.py --level fast        # seconds, no GPU

export TURBOQUANT_HG38=/path/to/hg38.fa         # for the full check
python tests/test_release.py --level full \
    --int4-ckpt evo2_40b_int4.pt --int4-ckpt-7b evo2_7b_int4.pt
```

The full check compares against recorded log-likelihoods rather than just looking
for a crash. That matters here: every bug we hit while building this produced
plausible-looking output, so agreement with a known value is the only real signal.

---

## A few things to watch out for

* Under tier1/tier2, don't call the model without inference params
  (`model(ids)`) — the fused attention doesn't accept that path and raises a
  `TypeError`. Compute any full-precision reference before switching tier.
* Don't load the model across several GPUs and then move it to one with
  `.to("cuda:0")`. Some internal state is tied to the original device and you
  will get a memory fault. Load it on the device you want.
* Context beyond roughly 524 kb currently fails in the int4 cache kernel (an
  indexing overflow we've patched but not yet re-verified at that size).
* All weights used here are public, so no Hugging Face login is required. If you
  do log in *and* set `HF_HOME`, set `HF_HOME` first — the token is looked up
  relative to it, and a mismatch surfaces as a bare `401`.

---

## Credit

The models are [Evo 2](https://github.com/ArcInstitute/evo2) by Arc Institute —
all credit for them belongs there. This project adds compression, the
long-sequence fix, and the measurements above. The published checkpoints are
quantized re-encodings of Arc Institute's weights and inherit their licence; the
code here is Apache-2.0.
