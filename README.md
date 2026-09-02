# kimodo-fast

Makes NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) motion diffusion model small and
fast enough to keep resident. Measured on an RTX 3090 against `nv-tlabs/kimodo @ 1aece8c`.

| | stock Kimodo | this |
|---|---:|---:|
| startup | 101 s | **12 s**, once |
| per clip (5 s of motion) | 3.465 s | **687 ms** at batch 16 |
| resident VRAM | 15.3 GB | **5.4 GB** |
| throughput | 17 clips/min | **87 clips/min** |

Fits an 8 GB card. `--steps 100 --cfg-type separated` reproduces stock output — every
speedup here is one you can back out of.

Measured against **Kimodo's own benchmark**, that speedup costs about **one point of R@3**
(87.77 vs 88.65) with foot-skate and foot-contact unchanged — see
[`BENCHMARK.md`](BENCHMARK.md).

## Setup

```bash
git clone https://github.com/suzuenhasa/kimodo-perf-patch.git && cd kimodo-perf-patch
bash bootstrap.sh
source ../env.sh
```

Takes 20–40 minutes, mostly the 18 GB of weights. It creates a venv, installs torch, clones
and patches Kimodo, installs this package, downloads the weights, and sets up an ungated
mirror for the text encoder (**no `HF_TOKEN` needed**). It is idempotent, so re-run it if it
dies. Run it detached if your ssh times out:

```bash
nohup bash bootstrap.sh > boot.log 2>&1 &
```

Needs a GPU, Python ≥ 3.10 and ~40 GB free. **8 GB of VRAM is enough** — verified end to
end on an RTX 2070, which also has no native bf16 (compute capability 7.5, so it is
emulated) and works anyway. Any recent torch works; set
`TORCH_INDEX=https://download.pytorch.org/whl/cu128` to pin the build the timings were
measured on.

### About the weights

The text encoder is LLM2Vec, which ships **adapters only** — its `adapter_config.json`
points at `meta-llama/Meta-Llama-3-8B-Instruct`, which is **gated**. Rather than require a
token and manual approval, `bootstrap.sh` defaults to
[`NousResearch/Meta-Llama-3-8B-Instruct`](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct),
an ungated mirror of the same bf16 weights, and rewrites that one field to point at it.

Either way these are Llama-3 weights and the **Meta Llama 3 Community License** applies —
a mirror changes how you download them, not what governs them. That matters if you
redistribute a built encoder; see [`NOTICE`](NOTICE).

Nothing in the normal path needs a token. If you have gated access — either to
`meta-llama` itself, or to `bones-studio/seed` for the benchmark ground truth — pass it in
the environment and `bootstrap.sh` will write it to `$HF_HOME/token`:

```bash
HF_TOKEN=hf_... bash bootstrap.sh
```

It is read from the environment rather than an argument, so it does not land in your shell
history. Use a **read-only** token, and delete it when you are done.

To use the official Llama repo instead of the mirror:

```bash
huggingface-cli login
MIRROR=meta-llama/Meta-Llama-3-8B-Instruct bash bootstrap.sh
```

## Build the encoder

One-time, and required before anything will run. It turns Kimodo's 15 GB text encoder into a
4.35 GB nf4 checkpoint that loads in 7 s.

```bash
kimodo-stream-encoder --out ../enc_nf4      # ~2 min, 0.17 GB VRAM, 3.7 GB RAM
```

The checkpoint is not in this repo — you build it, which also matches it to your own
bitsandbytes version.

There is a second builder, `kimodo-build-encoder`, which produces the same checkpoint but
needs ~6 GB of VRAM, ~16 GB of RAM and ~15 GB of scratch. Prefer `kimodo-stream-encoder` —
it holds one tensor at a time rather than the whole model. Given `--like <an existing
enc_nf4>` it reproduces that checkpoint byte for byte; this was verified tensor-by-tensor
across all 1410 tensors and by sha256 on every shard.

To publish a built checkpoint for others, see `kimodo-publish-encoder --help` — note the
Llama-3 licence obligations it enforces.

## Run

**Single request:**

```bash
kimodo-fast "a person walks forward." --encoder ../enc_nf4 --out ./out --duration 5.0
```

```python
from kimodo_fast import KimodoFast

k = KimodoFast("../enc_nf4")                 # 12 s, once, 5.4 GB resident
clips = k.generate(["a person walks forward."], duration=5.0)
# clips[i] is Kimodo's own dict: posed_joints (T,77,3), local/global_rot_mats, root_positions
```

**Batch request** — pass many prompts and they go through one denoiser pass:

```bash
kimodo-fast "a person walks forward." "a zombie staggers." "a person jumps." \
    --encoder ../enc_nf4 --out ./out --batch 16
```

```python
clips = k.generate(prompts, batch=16, duration=5.0, seed=0)
```

Or put them in a file, one per line — blank lines and `#` comments are skipped:

```bash
kimodo-fast --prompts-file prompts.txt --encoder ../enc_nf4 --out ./out
```

```
# prompts.txt
a person walks forward.
a zombie staggers.
a person jumps.
```

Duplicates are dropped, including ones that differ only in the capitalisation and trailing
period Kimodo adds for you, so the same clip is never generated twice.

Batching is where most of the throughput comes from, so use it whenever you have more than
one prompt. `generate()` batches prompts directly: `patches/0005` fixes the encoder's
bidirectional mask under padding, so a prompt embeds identically whether it is batched with
others or not.

**How much batching buys depends on how much GPU you have spare.** 24 prompts, 35 steps,
5 s clips:

| `--batch` | RTX 3090 (24 GB) | RTX 2070 (8 GB) |
|---:|---:|---:|
| 1 | 1085 ms | 2164 ms |
| 2 | — | 1971 ms |
| 4 | — | 1774 ms |
| 8 | — | 1808 ms |
| 16 | **687 ms** | 1799 ms |
| 24 | — | **1742 ms** |

The 3090 keeps gaining out to 16; the 2070 is saturated by 4 and flat after that, so
batching is worth 1.58× on one card and 1.24× on the other. The 3090 was not swept across
batch sizes, hence the gaps — its two points are a warm single call and the
87-clip run. A card with fewer cores is
already busy at small batches — there is no idle capacity for a bigger batch to fill. Pick
`--batch` accordingly; the default of 16 costs nothing on a small card, it just stops
helping.

Memory is not what limits it: batch 24 ran fine in 8 GB, because the resident 5.43 GB
leaves enough for activations at any batch this model uses.

**Timelines** — an ordered sequence, generated as one continuous motion rather than
stitched clips. Each segment is conditioned on the tail of the one before it, so the
character continues rather than cutting:

```python
clip = k.generate_sequence([("a person jumps", 3.0),
                            ("a person walks to the left", 5.0),
                            ("a person stands still", 5.0)])
```

From the command line, segments are separated by `|`, each optionally `:<seconds>`
(default 5). A colon inside a prompt is safe — only a trailing colon followed by a number
reads as a duration:

```bash
kimodo-fast --timeline "a person walks forward:3 | a person stops and waves:3 | a person jumps:2"

# several timelines: repeat the flag, or one per line in a file
kimodo-fast --timeline "a person runs:4|a person stops:2" --timeline "a person crouches:3"
kimodo-fast --timelines-file timelines.txt
```

This writes `timeline_00.npz…` plus a `timelines.json` recording each timeline's segments,
durations and frame count.

**What batches and what does not.** Segments within one timeline are sequential by
construction — each is conditioned on the tail of the last. Kimodo's multi-prompt path
also replicates one segment's text across the batch (`kimodo_model.py:164`), so the batch
dimension carries *alternative takes of a single timeline*, not different timelines:

```bash
kimodo-fast --timeline "a person walks:3|a person jumps:2" --variations 4   # batched
```

`--variations 4` costs **1332 ms per take** against 1737 ms generating them one at a time,
and the takes are genuinely sampled, not duplicated. Several *different* timelines run
serially and cost **1.18×** the batched-clip path for the same segment count — measured, and
the reason this wrapper does not patch Kimodo to batch them.

Kimodo is trained to ~10 s, so split anything longer into consecutive segments.

`transition_frames` defaults to **2**, not the demo's 5. At 5, later segments ignore their
own prompt — "a person stands still" runs at 0.02 m/s alone and 0.88 m/s as a third segment
after a walk, i.e. it just keeps walking. Raising text guidance makes it worse, not better.
At 2, a walk → wave → jump timeline holds each segment to its own prompt (2.31 m/s, then
0.18 m/s, then the largest vertical range) with no discontinuity at either splice.

**Paths and IK** — constrain where the character goes, or pin a hand or foot:

```python
path = k.root_path([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)], nframes=180)
clip = k.generate(["a person walks"], duration=6.0, constraints=[[path]])
```

Followed to **4 cm** mean error, against 297 cm unconstrained. `constraints=` takes one
entry per prompt, each a list of constraint sets — Kimodo also ships
`EndEffectorConstraintSet` (with hand/foot subclasses) and `FullBodyConstraintSet` for
keyframing.

**Cheap previews** — `steps=` overrides the step count for one call, for progressive
refinement while a user drags something:

```python
preview = k.generate_sequence(segs, steps=15)     # 2.1x faster
```

Unconstrained, 15 steps is the floor worth showing: below it the root drifts most of a metre
from where the final lands. **With a path constrained, 5 steps lands within 2 cm** — the
constraint does the positioning the extra steps would have to. Repeat requests are cached
and return in 0.3 ms.

Useful flags:

| flag | |
|---|---|
| `--prompts-file`, `-f` | read prompts from a file, one per line |
| `--duration` | seconds of motion, default 5. Kimodo is trained to ~10; beyond that it degrades |
| `--steps` | denoising steps, default 35. `100` is stock |
| `--cfg-type` | `regular` (default) or `separated`, which is stock |
| `--batch` | prompts per denoiser pass, default 16 |
| `--seed` | default 0 |
| `--encoder bf16` | load the original 15 GB encoder instead, for comparison |

`--steps 100 --cfg-type separated` gives you stock Kimodo output.

## Also here

- [`BENCHMARK.md`](BENCHMARK.md) — FID and R-precision against NVIDIA's own benchmark.
- [`patches/`](patches/) — six patches against Kimodo, five applied by `bootstrap.sh`;
  [`patches/README.md`](patches/README.md) says what each one is for.
- `scripts/generate_eval_fast.py` — drives Kimodo's `benchmark/generate_eval.py` unmodified,
  swapping in this stack so the benchmark numbers are comparable.
- `scripts/extract_seed.py` — pulls just the referenced BVH out of the SEED tarball.

Apache-2.0. Not affiliated with NVIDIA. Kimodo's weights and the Llama-3 / LLM2Vec weights
carry their own separate licences — see [`NOTICE`](NOTICE).
