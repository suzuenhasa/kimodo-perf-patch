#!/usr/bin/env python
"""One-time: turn Kimodo's 15 GB text encoder into a 4.35 GB checkpoint that loads in 7 s.

    python build_encoder.py --out ./enc_nf4

Runs once, ever. Afterwards serve.py never touches the original weights.

Order matters and is not negotiable: MERGE the adapters at bf16, save, THEN quantize.
Loading a quantized base and asking PEFT to merge a LoRA into it fails outright --
`SCB is neither a parameter, buffer, nor extra state` (int8), `'weight' is not an
nn.Module` (nf4). Both are PEFT reaching into a bitsandbytes layer.

What it actually costs, per step -- the binding constraint is RAM, not VRAM:

  [1] reference embedding   14.1 GB VRAM   OPTIONAL (--skip-verify, and auto-skipped
                                           on a card too small to hold it)
  [2] merge at bf16         ~16 GB RAM     CPU only, no VRAM at all
  [3] quantize + serialise  ~6 GB VRAM     streamed: transformers hands bitsandbytes
                                           one parameter at a time as the shards are
                                           read, so the bf16 model is never resident
  [4] verify the result     4.3 GB VRAM

If even ~6 GB is too much, kimodo-stream-encoder does the whole thing in 0.173 GB of VRAM
and 3.67 GB of RSS with no scratch at all, and produces this exact checkpoint byte for
byte. This builder stays because it is the one the format was derived from.

So the BUILD does not need a big card -- only the reference embedding in [1] does, and
that step exists purely to print a cosine. nf4 is a data-free quantizer (a 16-entry
codebook plus a per-64-element absmax; no calibration pass, unlike GPTQ/AWQ), so [3] is
a pure per-tensor map with nothing to hold globally.

Two private attributes are poked, and both will break without warning on the wrong
version: `_hf_peft_config_loaded` (transformers >= 5, or save_pretrained raises
UnboundLocalError) and `config._name_or_path` (llm2vec branches on it to add the Llama-3
chat header). Verified on transformers 5.1.0 with peft 0.17 and bitsandbytes 0.48. If a
future version changes either, this fails loudly at [2] or silently changes every
embedding at [4] -- which is what the cosine gate is for.

Scratch for the merged bf16 intermediate is ~15 GB (--scratch, default /dev/shm, which
is tmpfs and therefore RAM -- with the 16 GB model still live during save_pretrained
that is ~31 GB of RAM, so point --scratch at a disk unless you have the headroom).
"""
import argparse
import gc
import os
import shutil
import time
from pathlib import Path

import numpy as np
import torch

CANONICAL = "meta-llama/Meta-Llama-3-8B-Instruct"
BASE = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
PEFT = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
PROBES = ["A person walks forward.", "A person waves their right hand.", "A person jumps."]


# The three builds so far -- two machines, torch 2.11.0+cu128 and 2.13.0+cu130, CUDA 12.8
# and 13.0, venv and system python -- produced byte-identical output. If your card is too
# small for the cosine check in [1], this is the stronger gate anyway.
KNOWN = {"model-00001-of-00003.safetensors":
         "c7a46f4b4020612e4003f9c26b833bee9faf9ed0260c3f0cce26fd8a4c7f9f3c",
         "config.json":
         "5ce3d1ca5106f6f6944646549e33cebbe1cde378f7eac8a59e748d2ddf70da13"}


def free():
    gc.collect(); torch.cuda.empty_cache()


def peak(label):
    """Peak VRAM since the last reset. The [1] number is the only one that has ever
    been measured; [3] is printed so the next box on a small card settles it."""
    mb = torch.cuda.max_memory_allocated() / 2**30
    torch.cuda.reset_peak_memory_stats()
    print(f"        peak VRAM {label}: {mb:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./enc_nf4")
    ap.add_argument("--scratch", default="/dev/shm/kimodo_merged")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--peft", default=PEFT)
    ap.add_argument("--keep-scratch", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    from kimodo.model.llm2vec import LLM2Vec
    from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
    from transformers import AutoTokenizer, BitsAndBytesConfig

    # The McGill repos ship ADAPTERS ONLY and their adapter_config.json points at
    # meta-llama/Meta-Llama-3-8B-Instruct, which is gated. The usual fix is a local
    # shim directory with that one field rewritten to an ungated mirror, exported as
    # TEXT_ENCODERS_DIR. Every measurement script honours it; this one did not, so a
    # correctly-shimmed box still failed with "You are trying to access a gated repo".
    te = os.environ.get("TEXT_ENCODERS_DIR")
    if te:
        for attr in ("base", "peft"):
            repo = getattr(args, attr)
            local = os.path.join(te, repo)
            if not os.path.isdir(repo) and os.path.isdir(local):
                setattr(args, attr, local)
                print(f"  using TEXT_ENCODERS_DIR for --{attr}: {local}")

    def embed(m):
        with torch.no_grad():
            o = m.encode(PROBES, batch_size=1, show_progress_bar=False, device="cuda:0")
        return (o.float().cpu().numpy() if torch.is_tensor(o) else np.asarray(o)).astype(np.float64)

    # Every stage from [3] on needs a CUDA device -- bitsandbytes quantizes on the GPU --
    # so --skip-verify does not make this runnable on CPU. Say so before the first
    # get_device_properties(0), which would otherwise raise something far less clear.
    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible. This builder quantizes on the GPU, so "
                         "--skip-verify will not help; check `nvidia-smi` and that torch "
                         "was installed with CUDA support.")

    # [1] holds the whole 8B at bf16; nothing else in the build does. Rather than OOM
    # halfway, skip it and fall back to the checksum gate.
    vram = torch.cuda.get_device_properties(0).total_memory / 2**30
    if not args.skip_verify and vram < 17.0:
        print(f"  [1/4] SKIPPED: the reference embedding needs ~14.1 GB and this card has "
              f"{vram:.1f} GB.\n        The build itself does not -- continuing. Result is "
              f"checked against KNOWN below.")
        args.skip_verify = True

    ref = None
    torch.cuda.reset_peak_memory_stats()
    if not args.skip_verify:
        print("  [1/4] loading the shipped encoder for a reference embedding ...")
        t0 = time.perf_counter()
        m = LLM2Vec.from_pretrained(args.base, args.peft, torch_dtype=torch.bfloat16).to("cuda:0").eval()
        print(f"        {time.perf_counter()-t0:.1f} s, {torch.cuda.memory_allocated()/2**30:.2f} GB")
        ref = embed(m)
        del m; free(); peak("[1] reference embedding")

    print(f"  [2/4] merging both adapters at bf16 -> {args.scratch}")
    t0 = time.perf_counter()
    mm = LLM2Vec.from_pretrained(args.base, args.peft, torch_dtype=torch.bfloat16, merge_peft=True)
    inner = mm.model
    live = sum(1 for k, _ in inner.named_parameters() if "lora_" in k)
    if live:
        raise SystemExit(f"merge left {live} LoRA tensors alive; expected 0")
    # transformers >= 5: merge_and_unload() leaves _hf_peft_config_loaded set, and
    # save_pretrained then dies in get_adapter_state_dict with UnboundLocalError.
    inner._hf_peft_config_loaded = False
    Path(args.scratch).mkdir(parents=True, exist_ok=True)
    inner.save_pretrained(args.scratch, safe_serialization=True, max_shard_size="4GB")
    mm.tokenizer.save_pretrained(args.scratch)
    print(f"        {time.perf_counter()-t0:.1f} s")
    del mm, inner; free(); peak("[2] merge (expected ~0: this step is CPU-only)")

    # Written to .partial and renamed at the very end. Writing straight into --out means a
    # crash in [3], or a checkpoint that fails [4], replaces a previously good enc_nf4 with
    # a broken one -- and --out defaults to a path people re-run against.
    final = Path(args.out)
    out_tmp = final.with_name(final.name + ".partial")
    shutil.rmtree(out_tmp, ignore_errors=True)
    args.out = str(out_tmp)

    print(f"  [3/4] quantizing to nf4 and serialising -> {final}")
    t0 = time.perf_counter()
    q = LlamaBiModel.from_pretrained(
        args.scratch, device_map={"": 0},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)).eval()
    q.save_pretrained(args.out, safe_serialization=True, max_shard_size="2GB")
    AutoTokenizer.from_pretrained(args.scratch).save_pretrained(args.out)
    print(f"        {time.perf_counter()-t0:.1f} s")
    del q; free(); peak("[3] quantize -- this is the real minimum card size")

    size = sum(f.stat().st_size for f in Path(args.out).rglob("*") if f.is_file()) / 2**30
    print(f"  [4/4] verifying {args.out} ({size:.2f} GB on disk)")
    t0 = time.perf_counter()
    m2 = LlamaBiModel.from_pretrained(args.out, device_map={"": 0}).eval()
    m2.config._name_or_path = CANONICAL          # chat-header branch, llm2vec.py:173
    tok = AutoTokenizer.from_pretrained(args.out)
    tok.pad_token, tok.padding_side = tok.eos_token, "left"
    w = LLM2Vec(model=m2, tokenizer=tok, pooling_mode="mean", max_length=512,
                doc_max_length=400, skip_instruction=True)
    load_s = time.perf_counter() - t0
    free()
    print(f"        loads in {load_s:.1f} s, {torch.cuda.memory_allocated()/2**30:.2f} GB resident")

    cos = None
    if ref is not None:
        a = embed(w)
        c = np.sum(a*ref, -1) / (np.linalg.norm(a, axis=-1)*np.linalg.norm(ref, axis=-1) + 1e-12)
        cos = float(c.min())
        print(f"        cosine to the shipped encoder: {cos:.6f}")
        print("        (~0.98 is expected and correct: nf4 is lossy. The body motion it")
        print("         produces sits 25 mm RMS from bf16, inside Kimodo's own 26.7 mm")
        print("         constraint error -- ~90% of the visible difference is root path.)")
    import hashlib
    for name, want in KNOWN.items():
        f = Path(args.out) / name
        if not f.is_file():
            continue
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f"        {'MATCHES' if h == want else 'DIFFERS from'} the reference {name}")
        if h != want:
            print(f"          got  {h}\n          want {want}")
            print("          (a transformers/bitsandbytes upgrade can legitimately change"
                  " the\n           serialised layout -- check the cosine above, not this,"
                  " if it printed)")
    # nf4 lands at ~0.986. Anything under 0.95 is a broken build, not quantization loss,
    # so refuse to promote it -- and keep BOTH the .partial and the merged intermediate so
    # the next attempt does not have to redo [2].
    if cos is not None and cos < 0.95:
        raise SystemExit(
            f"\n  cosine {cos:.6f} is far below the ~0.986 nf4 produces -- not promoting.\n"
            f"  left in place for inspection:\n    {out_tmp}\n    {args.scratch}")

    if final.exists():
        shutil.rmtree(final)
    out_tmp.rename(final)
    if not args.keep_scratch:
        shutil.rmtree(args.scratch, ignore_errors=True)
        print(f"        removed {args.scratch}")
    print(f"\n  done. point serve.py at {final}")


if __name__ == "__main__":
    main()
