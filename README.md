# kimodo-fast

Makes NVIDIA's [Kimodo](https://github.com/nv-tlabs/kimodo) motion diffusion model small and
fast enough to keep resident. RTX 3090, against `nv-tlabs/kimodo @ 1aece8c`.

| | stock | this |
|---|---:|---:|
| startup | 101 s | **12 s**, once |
| per clip (5 s of motion) | 3.465 s | **687 ms** at batch 16 |
| resident VRAM | 15.3 GB | **5.4 GB** |
| throughput | 17 clips/min | **87 clips/min** |

Fits an 8 GB card. `--steps 100 --cfg-type separated` reproduces stock output.
Those are a warm 5 s clip at batch 16 and resident memory; [`BENCHMARK.md`](BENCHMARK.md)
reports 938 ms and 6,882 MiB because NVIDIA's harness adds post-processing and measures peak.

Against Kimodo's own benchmark it costs about **one point of R@3** (87.77 vs 88.65), with
foot-skate and contact unchanged. On constrained generation — pinned paths, hands, feet,
keyframes — it costs **nothing**: ours matches or beats stock on every metric.
See [`BENCHMARK.md`](BENCHMARK.md).

## Setup

```bash
git clone https://github.com/suzuenhasa/kimodo-perf-patch.git && cd kimodo-perf-patch
bash bootstrap.sh          # 20-40 min, mostly the 18 GB of weights
source ../env.sh
```

Idempotent — re-run it if it dies. Run detached if ssh times out:
`nohup bash bootstrap.sh > boot.log 2>&1 &`

| needs | |
|---|---|
| GPU | 8 GB is enough; verified on an RTX 2070 (no native bf16) |
| Python | ≥ 3.10 |
| disk | ~40 GB |
| HF token | **not needed** — the text encoder uses an ungated mirror |

The encoder is LLM2Vec, which ships adapters only and points at the gated
`meta-llama/Meta-Llama-3-8B-Instruct`. `bootstrap.sh` substitutes
[`NousResearch/Meta-Llama-3-8B-Instruct`](https://huggingface.co/NousResearch/Meta-Llama-3-8B-Instruct),
the same bf16 weights, ungated. Either way the **Meta Llama 3 Community License** applies —
see [`NOTICE`](NOTICE).

For gated extras (`meta-llama` itself, or `bones-studio/seed` for benchmark ground truth),
pass a **read-only** token in the environment so it stays out of your shell history:

```bash
HF_TOKEN=hf_... bash bootstrap.sh
MIRROR=meta-llama/Meta-Llama-3-8B-Instruct bash bootstrap.sh   # official repo instead
```

## Build the encoder

One-time, required before anything runs. Turns the 15 GB encoder into a 4.35 GB nf4
checkpoint that loads in 7 s.

```bash
kimodo-stream-encoder --out ../enc_nf4      # ~2 min, 0.17 GB VRAM, 3.7 GB RAM
```

Holds one tensor at a time, so it builds on any card. `kimodo-build-encoder` produces the
identical checkpoint but wants ~6 GB VRAM, ~16 GB RAM and ~15 GB scratch. Given
`--like <existing enc_nf4>` the streaming builder reproduces that checkpoint byte for byte
— verified across all 1410 tensors and by sha256 per shard.

Sharing a build: `kimodo-publish-encoder --help` (it enforces the Llama-3 obligations).

## Run

```bash
# one prompt
kimodo-fast "a person walks forward." --encoder ../enc_nf4 --out ./out

# many — batching is where the throughput is
kimodo-fast "a person jumps." "a zombie staggers." --encoder ../enc_nf4
kimodo-fast --prompts-file prompts.txt --encoder ../enc_nf4
```

`prompts.txt` is one per line; `#` and blank lines are skipped. Duplicates are dropped,
including ones differing only in the capitalisation and trailing period Kimodo adds.

### Timelines

One continuous motion from several prompts — each segment is conditioned on the tail of the
last, so the character carries momentum across the joins rather than cutting.

```bash
kimodo-fast --timeline "a person walks forward:3 | a person stops and waves:3 | a person jumps:2"
kimodo-fast --timelines-file timelines.txt          # one timeline per line
kimodo-fast --timeline "a person walks:3|a person jumps:2" --variations 4
```

Segments split on `|`, `:seconds` optional (default 5). A colon inside a prompt is safe —
only a trailing colon followed by a number reads as a duration. Writes `timeline_00.npz…`
plus a `timelines.json` of segments and frame counts.

| | |
|---|---|
| a 9 s timeline (3 segments) | **1.88 s** — about 4.8× faster than real time |
| `--variations 4` | 1332 ms/take vs 1737 ms serial; batched, genuinely sampled |
| several *different* timelines | serial at 628 ms/segment, ~1.6× the batched-clip path |

Only variations batch. Kimodo replicates one segment's text across the batch
(`kimodo_model.py:164`), so the batch dimension carries takes of *one* timeline.

`--transition-frames` defaults to **2**, not the demo's 5. At 5 later segments ignore their
own prompt — "a person stands still" runs 0.02 m/s alone and 0.88 m/s after a walk, i.e. it
just keeps walking.

### Python

```python
from kimodo_fast.serve import KimodoFast

k       = KimodoFast("../enc_nf4")
clips   = k.generate(["a person walks forward."], duration=5.0, batch=16)

segs    = [("a person jumps", 3.0), ("a person walks left", 5.0)]
tl      = k.generate_sequence(segs)
takes   = k.generate_sequence(segs, variations=4)   # 4 alternative takes, batched

# constrain where it goes, or pin a hand or foot
path  = k.root_path([(0, 0), (3, 0), (3, 3), (0, 3), (0, 0)], nframes=180)
clip  = k.generate(["a person walks"], duration=6.0, constraints=[[path]])
```

`clips[i]` is Kimodo's own dict — `posed_joints (T,77,3)`, `local/global_rot_mats`,
`root_positions`. A pinned path is followed to **4 cm** mean error against 297 cm
unconstrained. `constraints=` takes one entry per prompt, each a list of constraint sets;
Kimodo also ships `EndEffectorConstraintSet` (hand/foot subclasses) and
`FullBodyConstraintSet`.

**Cheap previews** — `steps=` overrides for one call:

```python
preview = k.generate_sequence(segs, steps=15)     # 2.1x faster
```

15 steps is the floor worth showing unconstrained. **With a path pinned, 5 steps lands
within 2 cm** of the final — the constraint does the positioning. Repeats are cached and
return in 0.3 ms.

## Flags

| | |
|---|---|
| `--prompts-file`, `-f` | prompts from a file, one per line |
| `--timeline` | segments split on `\|`, each optionally `:seconds`. Repeatable |
| `--timelines-file` | one timeline per line |
| `--variations` | alternative takes of one timeline, batched |
| `--transition-frames` | frames blended between segments, default 2 |
| `--duration` | seconds, default 5. Kimodo is trained to ~10 |
| `--steps` | denoising steps, default 35. `100` is stock |
| `--cfg-type` | `regular` (default) or `separated` (stock) |
| `--batch` | prompts per denoiser pass, default 16 |
| `--seed` | default 0 |
| `--encoder bf16` | the original 15 GB encoder, for comparison |

## Also here

- [`BENCHMARK.md`](BENCHMARK.md) — FID, R-precision and constrained metrics against
  NVIDIA's own benchmark, plus the quality-vs-steps curve.
- [`patches/`](patches/) — six patches, five applied by `bootstrap.sh`;
  [`patches/README.md`](patches/README.md) says what each is for.
- `scripts/generate_eval_fast.py` — drives Kimodo's `benchmark/generate_eval.py` unmodified.
- `scripts/extract_seed.py` — pulls only the referenced BVH out of the SEED tarball.

Apache-2.0. Not affiliated with NVIDIA. Kimodo's weights and the Llama-3 / LLM2Vec weights
carry their own separate licences — see [`NOTICE`](NOTICE).
