# Benchmark results

Every other number in this repo is speed, or a proxy for quality — jerk, embedding cosine,
joint drift, and watching clips. This page is the one that puts the whole stack against
**Kimodo's own benchmark**: FID, R-precision, foot-skate and foot-contact, computed by
NVIDIA's `benchmark/` pipeline rather than anything written here.

**Result: 2.5× faster and 2.5× less VRAM costs about one point of R@3.**

## Setup

Both arms go through `benchmark/generate_eval.py` unmodified — same discovery, same
per-case frame counts, same seeding, same cropping, same npz layout. Only the model
differs, swapped by [`scripts/generate_eval_fast.py`](scripts/generate_eval_fast.py).

| arm | encoder | steps | cfg | ms/clip | VRAM |
|---|---|---:|---|---:|---:|
| **ours** | nf4, 4.35 GB | 35 | `regular` | **938** | **6,882 MiB** |
| **stock** | bf16, 15 GB | 100 | `separated` | 2,344 | 17,506 MiB |

1,311 test cases — a deterministic every-other-leaf half of `content/text2motion`
(overview 458, timeline_single 458, timeline_multi 395). RTX 3090, `--postprocess` on both.

## Text-following

| | | ours | stock | ground truth |
|---|---|---:|---:|---:|
| **Overview** | R@3 ↑ | 87.77 | 88.65 | 95.41 |
| | FID ↓ | 0.039 | 0.032 | 0.000 |
| | Skate ↓ | 2.593 | 2.585 | 1.848 |
| | Contact ↑ | 0.972 | 0.972 | 1.000 |
| **Timeline single** | R@3 ↑ | 83.19 | 84.06 | 93.23 |
| | FID ↓ | 0.045 | 0.040 | 0.000 |
| | Skate ↓ | 2.568 | 2.526 | 1.778 |
| | Contact ↑ | 0.977 | 0.976 | 1.000 |
| **Timeline multi** | R@3 ↑ | **91.14** | 89.87 | 93.16 |
| | FID ↓ | 0.045 | 0.044 | 0.000 |
| | Skate ↓ | **2.263** | 2.429 | 1.743 |
| | Contact ↑ | 0.979 | 0.977 | 1.000 |

Read it against the ceiling. Ground truth is 95.41; stock reaches 88.65, so the model
already concedes 6.76 points to reality. Ours concedes a further **0.88** — 13% of the gap
stock had already given up, and 1% of its score. On timeline-multi ours is *better* on both
R@3 and skate, which is probably noise but bounds the damage.

**Physical plausibility is untouched**: skate 2.593 vs 2.585 and contact 0.972 vs 0.972.
That is what would have degraded first if 35 steps were producing sloppy motion.

**FID is the one number that moved** — 0.032 → 0.039, +22% relative. Tiny in absolute terms
against a floor of 0, and invisible in every physical metric, but it is real and it is the
measure most sensitive to distribution shift. Reporting R@3 and skate alone would be
cherry-picking.

## What this does and does not establish

**It validates the stack, not any single change.** nf4, 35 steps and `cfg regular` were
tested together, so the 0.88 cannot be attributed among them. Earlier claims here that "nf4
is indistinguishable" rested on cosine and joint drift; the honest version is that the
*combination* is nearly indistinguishable, which is weaker.

**The benchmark build is verified.** Our ground-truth foot-skate is **1.848** against
NVIDIA's published **1.849**. Ground truth is built from SEED BVH and never touches our
code, so that near-exact match says the skeleton conversion, benchmark construction and
metric plumbing are all correct. Without it the rest would not be worth reading.

**Do not compare these R@3 numbers to the published table.** R@3 is retrieval against a
gallery, and ours is half the size, which makes it easier — our *ground truth* row is 95.41
against their 89.09 for exactly that reason. Between our two arms the comparison is exact,
because gallery, cases and seeds are identical. Against their table it is not. Their
batch-1 seeding caveat applies as well: both arms here used batch 16.

## Reproducing

```bash
# ground truth: 917 BVH files pulled out of soma_uniform.tar.gz, then
python benchmark/create_benchmark.py <testsuite>/content/text2motion \
    --dataset <seed>/soma_uniform --workers 24            # 2623 cases in 46 s

python scripts/generate_eval_fast.py --benchmark <subset> --output gen_ours \
    --encoder ../enc_nf4 --diffusion_steps 35 --cfg_type regular --batch_size 16 --postprocess
python scripts/generate_eval_fast.py --benchmark <subset> --output gen_stock \
    --diffusion_steps 100 --cfg_type separated --batch_size 16 --postprocess

python benchmark/embed_folder.py gen_ours --model tmr-soma-rp
python benchmark/evaluate_folder.py gen_ours
python benchmark/parse_folder.py <root> --format md
```

Four things that are not in NVIDIA's docs and cost hours to find:

- **SEED is three tarballs and you need one.** `soma_uniform.tar.gz` is 42 GB of the 106.
- **Do not extract it.** 42 GB of gzipped BVH text overflows a 150 GB box. Stream the
  tarball once and pull only the files your subset references — 917 files, 1.94 GB.
- **`git clone` the testsuite, don't `snapshot_download` it.** 58,265 tiny files means
  58,265 HTTP requests and hard rate-limiting; git took **4 seconds** for what
  `snapshot_download` could not finish in 20 minutes.
- **`evaluate_folder.py` crashes on any current scipy** — see `patches/0006`.
- `parse_folder.py` needs paths shaped `<split>/<category>/<testcase>`, so the generated
  tree has to sit under `content/text2motion/` before it will parse.
