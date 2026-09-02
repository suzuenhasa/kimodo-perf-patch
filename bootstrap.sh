#!/usr/bin/env bash
# One-shot setup for kimodo-fast on a fresh GPU box. Idempotent; safe to re-run.
#
#   git clone https://github.com/suzuenhasa/kimodo-perf.git && cd kimodo-perf
#   bash bootstrap.sh                      # ~20-40 min, mostly weights
#   source ../env.sh && kimodo-fast "a person walks forward." --encoder ../enc_nf4
#
# Every step here exists because running the README on a clean box failed without it.
# Run it DETACHED if your ssh times out: nohup bash bootstrap.sh > boot.log 2>&1 &
set -euo pipefail

PERF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=${ROOT:-$(dirname "$PERF")}
VENV=$ROOT/venv
SRC=$ROOT/kimodo
TE_DIR=$ROOT/text_encoders
# Some GPU images preset HF_HOME (often via /etc/environment). Honour that -- the
# operator usually pointed it at the big volume on purpose -- but remember we did, because
# it means the weights do NOT land under $ROOT and the space check has to follow them.
HF_PRESET=${HF_HOME:+inherited from the environment}
export HF_HOME=${HF_HOME:-$ROOT/hf}
# McGill ship ADAPTERS ONLY, and their adapter_config.json points at
# meta-llama/Meta-Llama-3-8B-Instruct, which is GATED -- without this mirror the build
# dies with "You are trying to access a gated repo". Same bf16 weights, ungated.
MIRROR=${MIRROR:-NousResearch/Meta-Llama-3-8B-Instruct}
SKIP_VENV=${SKIP_VENV:-0}
TORCH_INDEX=${TORCH_INDEX:-}

say() { printf "\n=== %s ===\n" "$*"; }
die() { printf "\nFAILED: %s\n" "$*" >&2; exit 1; }

say "0. preflight"
command -v git   >/dev/null || die "git not installed"
command -v python3 >/dev/null || die "python3 not installed"   # note: 'python' often absent
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || die "no NVIDIA GPU visible"
python3 -c 'import sys; assert sys.version_info >= (3,10), sys.version' || die "need Python >= 3.10"
mkdir -p "$HF_HOME"
echo "  weights -> $HF_HOME${HF_PRESET:+  ($HF_PRESET)}"

# Nothing in the normal path needs a token: the text-encoder mirror is ungated. HF_TOKEN is
# only for the gated extras -- meta-llama itself, or bones-studio/seed for the benchmark
# ground truth. Passed in the environment so it never becomes a shell argument, and written
# only to HF_HOME, which is where huggingface_hub already looks.
if [ -n "${HF_TOKEN:-}" ]; then
    printf '%s' "$HF_TOKEN" > "$HF_HOME/token"
    chmod 600 "$HF_HOME/token"
    echo "  HF token written to \$HF_HOME/token (gated repos available; delete it when done)"
    unset HF_TOKEN
fi
df -h "$ROOT" | tail -1

# ~18 GB of weights land in HF_HOME; ~7 GB of torch/deps plus the kimodo source land under
# $ROOT; kimodo-build-encoder additionally wants ~15 GB of scratch (kimodo-stream-encoder,
# the recommended builder, wants none). Those are two different filesystems whenever the
# image preset HF_HOME, so check whichever ones are actually involved rather than assuming
# $ROOT covers it. MIN_FREE_GB overrides the combined figure, e.g. when HF_HOME is already
# populated.
free_gb() { df -BG --output=avail "$1" | tail -1 | tr -dc '0-9'; }
dev_of()  { df --output=source "$1" | tail -1; }

MIN_FREE_GB=${MIN_FREE_GB:-40}
if [ "$(dev_of "$ROOT")" = "$(dev_of "$HF_HOME")" ]; then
    FREE_GB=$(free_gb "$ROOT")
    [ "$FREE_GB" -ge "$MIN_FREE_GB" ] || die "need >= ${MIN_FREE_GB} GB free on the filesystem holding both $ROOT and $HF_HOME (weights ~18 GB, deps ~7 GB, build-encoder scratch ~15 GB); have ${FREE_GB} GB. Override with MIN_FREE_GB=n if the weights are already cached."
else
    echo "  note: $HF_HOME is on a different filesystem to $ROOT; checking both"
    df -h "$HF_HOME" | tail -1
    W=$(free_gb "$HF_HOME"); R=$(free_gb "$ROOT")
    [ "$W" -ge "${MIN_WEIGHTS_GB:-20}" ] || die "need >= ${MIN_WEIGHTS_GB:-20} GB free on $HF_HOME for the weights; have ${W} GB. Point HF_HOME somewhere larger, or set MIN_WEIGHTS_GB=n if they are already cached there."
    [ "$R" -ge "${MIN_ROOT_GB:-12}" ] || die "need >= ${MIN_ROOT_GB:-12} GB free under $ROOT for the venv and sources; have ${R} GB."
fi

if [ "$SKIP_VENV" = "1" ]; then
    PY=python3; PIP="python3 -m pip"
    say "1. venv SKIPPED (SKIP_VENV=1) -- installing into system python"
else
    say "1. venv + torch"
    [ -d "$VENV" ] || python3 -m venv "$VENV"
    PY=$VENV/bin/python; PIP="$VENV/bin/pip"
    $PIP install -q --upgrade pip
    $PIP install -q numpy    # before torch, or its first import warns "Failed to initialize NumPy"
    # Whatever torch pip offers is fine. Verified on 2.11.0+cu128 and 2.13.0+cu130;
    # both produce a byte-identical nf4 encoder. Set TORCH_INDEX to reproduce the exact
    # build the timings were measured on:
    #   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash bootstrap.sh
    $PIP install -q torch ${TORCH_INDEX:+--index-url $TORCH_INDEX}
fi
$PY -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__, 'sm', torch.cuda.get_device_capability())" \
    || die "torch cannot see the GPU"

say "2. kimodo source + patches"
# Pinned: every patch below is a diff against this exact commit, and the measurements in
# BENCHMARK.md were taken on it. Override to try a newer upstream, but expect patch fallout.
KIMODO_REV=${KIMODO_REV:-1aece8c}
if [ ! -d "$SRC" ]; then
    git clone -q https://github.com/nv-tlabs/kimodo.git "$SRC"
    git -C "$SRC" checkout -q "$KIMODO_REV" || die "cannot check out kimodo @ $KIMODO_REV"
fi
cd "$SRC"; git log -1 --format='kimodo @ %h %s'
apt-get -qq update >/dev/null 2>&1 || true
apt-get -qq install -y cmake build-essential python3-dev >/dev/null 2>&1 || true
for p in 0001-motioncorrection-cmake-python3-var 0002-device-and-diffusion-vars \
         0003-admm-smoother 0005-bidirectional-mask-transformers5 \
         0006-fid-sqrtm-scipy-compat; do
    f=$PERF/patches/$p.patch
    [ -f "$f" ] || die "missing patch $f"
    if patch -p1 --dry-run --forward -s < "$f" >/dev/null 2>&1; then
        patch -p1 --forward -s < "$f" || die "patch $p failed to apply"
        echo "  applied $p"
    elif patch -p1 --dry-run --reverse -s < "$f" >/dev/null 2>&1; then
        echo "  $p already applied"
    else
        die "patch $p neither applies nor is already applied -- is $SRC at $KIMODO_REV?"
    fi
done
# 0001 matters only inside a venv: setup.py passes the legacy -DPYTHON_EXECUTABLE while
# CMakeLists uses find_package(Python3), which reads Python3_EXECUTABLE.
$PIP install -q -e . || die "kimodo install failed"
$PY -c "import motion_correction" 2>/dev/null && echo "  motion_correction OK" \
    || die "MotionCorrection did not build -- post_processing=True will raise"

say "3. kimodo-fast"
$PIP install -q -e "$PERF" || die "kimodo-fast install failed"
$PY -c "import kimodo_fast; print('  kimodo_fast', kimodo_fast.__version__, kimodo_fast.__file__)"

say "4. weights (~18 GB) + ungated-mirror shim"
mkdir -p "$HF_HOME"
$PY - <<PY || die "weight download failed"
from huggingface_hub import snapshot_download
for repo in ["nvidia/Kimodo-SOMA-RP-v1.1",
             "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
             "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised",
             "$MIRROR"]:
    print(f"  --- {repo}", flush=True); snapshot_download(repo_id=repo)
PY
$PY - <<PY || die "text-encoder shim failed"
import json, pathlib
from huggingface_hub import snapshot_download
te = pathlib.Path("$TE_DIR")
base_local = snapshot_download(repo_id="$MIRROR")
for name in ["LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
             "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"]:
    src = pathlib.Path(snapshot_download(repo_id=f"McGill-NLP/{name}"))
    dst = te / "McGill-NLP" / name; dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "adapter_config.json":
            cfg = json.loads(f.read_text())
            cfg["base_model_name_or_path"] = base_local      # the ONLY edit
            (dst / f.name).write_text(json.dumps(cfg, indent=2))
        else:
            link = dst / f.name
            if link.is_symlink() or link.exists(): link.unlink()
            link.symlink_to(f.resolve())
    print(f"  shimmed {dst}")
# The shim must NOT touch config.json: llm2vec branches on _name_or_path to add the
# Llama-3 chat header, and losing it changes the tokenization, hence every embedding.
cfg = json.loads((te / "McGill-NLP" / "LLM2Vec-Meta-Llama-3-8B-Instruct-mntp" / "config.json").read_text())
assert cfg["_name_or_path"] == "meta-llama/Meta-Llama-3-8B-Instruct", cfg["_name_or_path"]
print("  _name_or_path preserved -> chat template unchanged")
PY

cat > "$ROOT/env.sh" <<ENV
$([ "$SKIP_VENV" = "1" ] || echo "source $VENV/bin/activate")
export HF_HOME=$HF_HOME
export TEXT_ENCODERS_DIR=$TE_DIR
export TEXT_ENCODER_MODE=local
export LOCAL_CACHE=true
ENV

say "done"
df -h "$ROOT" | tail -1
cat <<NEXT

  source $ROOT/env.sh
  kimodo-stream-encoder --out $ROOT/enc_nf4       # ~2 min, 0.2 GB VRAM
  kimodo-fast "a person walks forward." --encoder $ROOT/enc_nf4 --out $ROOT/out

NEXT
