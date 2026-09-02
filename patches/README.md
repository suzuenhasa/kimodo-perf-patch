# Patches

Diffs against [`nv-tlabs/kimodo`](https://github.com/nv-tlabs/kimodo) at **`1aece8c`**, the
commit `bootstrap.sh` pins. Five are applied automatically; `0004` is opt-in.

| | file(s) touched | applied | what it is |
|---|---|:-:|---|
| **0001** | `setup.py` | yes | MotionCorrection's CMake reads `Python3_EXECUTABLE`, but `setup.py` passes the legacy `-DPYTHON_EXECUTABLE`. Inside a venv the extension builds against the *system* interpreter, and the failure only surfaces much later as an import error when `post_processing=True`. |
| **0002** | `model/diffusion.py`, `model/kimodo_model.py`, `motion_rep/feature_utils.py`, `motion_rep/reps/base.py` | yes | Device and diffusion-step count are hardcoded. This makes both parameters, which is what lets you run anywhere but `cuda:0` and sweep step counts at all. |
| **0003** | `motion_rep/smooth_root.py` | yes | ADMM root smoother. |
| **0004** | `model/backbone.py` | **no** | CUDA-graph capture of the denoiser step. Experimental — measurable, but not something to run by default. Apply by hand if you want it. |
| **0005** | `model/llm2vec/models/bidirectional_llama.py` | yes | **Correctness fix.** transformers 5.x moved mask construction into `masking_utils`; without this the "bidirectional" encoder silently builds a *causal* mask, so a padded prompt embeds differently depending on what it was batched with. |
| **0006** | `metrics/tmr.py` | yes | scipy ≥ 1.18 removed the `disp` keyword from `sqrtm`, which makes FID raise. Only matters if you run the benchmark. |

`0005` is the one that changes output rather than speed. It is why prompts can be batched
directly instead of grouped by token length.

## Applying by hand

```bash
cd /path/to/kimodo
patch -p1 < /path/to/patches/0002-device-and-diffusion-vars.patch
```

Upstream moves; these are pinned. If a patch stops applying, that is the pin drifting, not
a bug in your setup — `bootstrap.sh` now fails loudly rather than reporting "already applied".
