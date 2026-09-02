#!/usr/bin/env python
"""Build the nf4 encoder one tensor at a time. Peak memory is one tensor, not one model.

    kimodo-stream-encoder --out ./enc_nf4_stream --like ./enc_nf4

The normal builder (build_encoder.py) materialises the merged 8B at bf16 in CPU RAM, writes
~15 GB of scratch, then reloads it to quantize: ~16 GB RAM, ~15 GB disk, ~6 GB VRAM.

None of that is inherent. Both operations it performs are per-tensor local:

  merge      W' = W + (B @ A) * (alpha / r)          needs only this layer's weights
  quantize   nf4 = 16-entry codebook + one absmax    data-free -- no calibration pass,
             per 64 weights                          unlike GPTQ/AWQ, so nothing global

So this reads the base shards lazily, applies both LoRA deltas to a tensor, quantizes it,
writes it, and drops it. Peak RSS is the largest single tensor -- embed_tokens at
128256x4096 bf16, ~1.0 GB -- plus the two adapters, which are ~65 MB each. Peak VRAM is
the largest *quantized* tensor, down_proj at 4096x14336 bf16, ~117 MB.

This is not a different quantizer. It produces the byte-identical checkpoint the big-RAM
path produces, by the shorter route -- verified tensor-by-tensor with --verify-against,
which is the gate and is worth re-running after any transformers/bitsandbytes upgrade:

    kimodo-stream-encoder --out ./enc_stream --like ./enc_nf4 --verify-against ./enc_nf4
"""
import argparse
import json
import os
import resource
import struct
import time
from pathlib import Path

import torch

QUANT_SUFFIX = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def resolve(repo, te=None):
    """Accept a path, or a repo id to look up under TEXT_ENCODERS_DIR / the HF cache."""
    if os.path.isdir(repo):
        return Path(repo)
    te = te or os.environ.get("TEXT_ENCODERS_DIR")
    if te and os.path.isdir(Path(te) / repo):
        return Path(te) / repo
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo))


def shards(d):
    """Every safetensors file in a checkpoint dir, single-file or sharded."""
    idx = d / "model.safetensors.index.json"
    if idx.is_file():
        names = sorted(set(json.loads(idx.read_text())["weight_map"].values()))
        return [d / n for n in names]
    one = d / "model.safetensors"
    return [one] if one.is_file() else sorted(d.glob("*.safetensors"))


# safetensors' own safe_open mmaps the file, and every tensor we touch leaves its pages
# resident -- reading all 16 GB of base shards showed up as 16 GB of RSS. Reading the byte
# ranges by hand keeps RSS to the tensor in flight.
DT = {"BF16": torch.bfloat16, "F16": torch.float16, "F32": torch.float32,
      "F64": torch.float64, "U8": torch.uint8, "I8": torch.int8, "I16": torch.int16,
      "I32": torch.int32, "I64": torch.int64, "BOOL": torch.bool}


def header(p):
    """safetensors header without opening the payload -- 8-byte length, then JSON."""
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def open_shard(p):
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_tensor(fh, hdr, base, key):
    e = hdr[key]
    s, end = e["data_offsets"]
    fh.seek(base + s)
    buf = bytearray(end - s)
    if fh.readinto(buf) != len(buf):
        raise SystemExit(f"short read on {key}")
    return torch.frombuffer(buf, dtype=DT[e["dtype"]]).reshape(e["shape"])


class Adapter:
    """One LoRA, held in RAM. Both of Kimodo's are ~65 MB, so this is not the problem."""

    def __init__(self, d):
        from safetensors.torch import load_file
        cfg = json.loads((d / "adapter_config.json").read_text())
        self.dir, self.cfg = d, cfg

        # Anything below changes the merge math, and silently producing a subtly wrong
        # encoder is worse than refusing to build one.
        for bad, why in (("use_dora", "DoRA decomposes the delta; W + BA is then wrong"),
                         ("use_rslora", "rsLoRA scales by alpha/sqrt(r), not alpha/r")):
            if cfg.get(bad):
                raise SystemExit(f"{d.name}: {bad}=True is not supported here -- {why}. "
                                 f"Use kimodo-build-encoder instead.")
        if cfg.get("modules_to_save"):
            raise SystemExit(f"{d.name}: modules_to_save={cfg['modules_to_save']} replaces "
                             f"whole modules rather than adding a delta. Not handled here; "
                             f"use kimodo-build-encoder.")
        self.rank_pattern = cfg.get("rank_pattern") or {}
        self.alpha_pattern = cfg.get("alpha_pattern") or {}
        self.r, self.alpha = cfg["r"], cfg["lora_alpha"]
        self.fan_in_fan_out = bool(cfg.get("fan_in_fan_out"))

        w = {}
        for f in sorted(d.glob("adapter_model.safetensors")) or sorted(d.glob("*.safetensors")):
            w.update(load_file(str(f)))
        # PEFT writes base_model.model.<key>.lora_A.weight, sometimes with a .<adapter>
        # segment before .weight. Normalise to the base model's own key.
        self.A, self.B = {}, {}
        for k, v in w.items():
            if ".lora_A" not in k and ".lora_B" not in k:
                continue
            side = "A" if ".lora_A" in k else "B"
            base = k.split(f".lora_{side}")[0]
            for pre in ("base_model.model.model.", "base_model.model.", "base_model."):
                if base.startswith(pre):
                    base = base[len(pre):]
                    break
            (self.A if side == "A" else self.B)[base + ".weight"] = v
        missing = set(self.A) ^ set(self.B)
        if missing:
            raise SystemExit(f"{d.name}: {len(missing)} unpaired LoRA tensors, e.g. "
                             f"{sorted(missing)[:3]}")

    def scaling(self, key):
        mod = key[: -len(".weight")]
        r = next((v for p, v in self.rank_pattern.items() if mod.endswith(p)), self.r)
        a = next((v for p, v in self.alpha_pattern.items() if mod.endswith(p)), self.alpha)
        return a / r

    def delta(self, key):
        """PEFT's get_delta_weight, in fp32 and left there -- see apply() for why."""
        A, B = self.A.get(key), self.B.get(key)
        if A is None:
            return None
        d = (B.float() @ A.float()) * self.scaling(key)
        return d.T if self.fan_in_fan_out else d

    def apply(self, W, key):
        """W + delta, in the exact arithmetic the reference checkpoint was built with.

        Neither cast here is cosmetic. Measured against the reference, on one 4096x14336
        tensor, out of 29,360,128 quantized bytes:

            fp32 matmul, fp32 add, cast after each adapter          0 differ
            fp32 matmul, bf16 add                              34,003 differ
            bf16 matmul                                       324,581 differ
            fp32 throughout, single cast at the end           324,581 differ

        So: promote to fp32, add, and come back to bf16 once per adapter. Accumulating
        both adapters in fp32 before a single cast is as wrong as never promoting at all.
        The GPU and CPU fp32 matmuls agree exactly, and thread count is irrelevant --
        both were checked."""
        d = self.delta(key)
        return W if d is None else (W.float() + d).to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="./enc_nf4_stream")
    ap.add_argument("--base", default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
    ap.add_argument("--peft", default="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
    ap.add_argument("--weights", default=None,
                    help="base bf16 weights; default: read from the adapter's "
                         "base_model_name_or_path (the mirror, after bootstrap.sh)")
    ap.add_argument("--like", default=None,
                    help="an existing enc_nf4 to copy shard layout and config.json from, "
                         "so the output is byte-identical rather than merely equivalent")
    ap.add_argument("--verify-against", default=None,
                    help="an existing enc_nf4 to compare tensor-by-tensor after building")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    import bitsandbytes as bnb
    from safetensors.torch import save_file
    from safetensors import safe_open

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    a_dir, b_dir = resolve(args.base), resolve(args.peft)
    print(f"  adapters: {a_dir.name}\n            {b_dir.name}")
    ada, adb = Adapter(a_dir), Adapter(b_dir)
    print(f"  {len(ada.A)} + {len(adb.A)} LoRA pairs, "
          f"scaling {ada.scaling(next(iter(ada.A)))} / {adb.scaling(next(iter(adb.A)))}")

    w_dir = Path(args.weights) if args.weights else resolve(
        json.loads((a_dir / "adapter_config.json").read_text())["base_model_name_or_path"])
    print(f"  base weights: {w_dir}")

    # Which key lands in which shard. Reusing the reference index reproduces the big-RAM
    # builder's packing exactly; without it we emit one shard per input shard, which loads
    # fine but will not checksum the same.
    layout = None
    if args.like:
        layout = json.loads((Path(args.like) / "model.safetensors.index.json").read_text())["weight_map"]

    dev = args.device if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("  no CUDA -- quantize_4bit needs a GPU; this will fail. Aborting early.")
        raise SystemExit(2)
    torch.cuda.reset_peak_memory_stats()

    # A target shard is written the moment its last key arrives, so at most one output
    # shard is ever held. Without --like each input shard maps to one output shard, which
    # flushes at the end of its pass.
    expect = {}
    if layout:
        for k, v in layout.items():
            expect.setdefault(v, set()).add(k)

    written, wmap, buckets, done, t0 = {}, {}, {}, 0, time.perf_counter()
    src = shards(w_dir)
    if not src:
        raise SystemExit(f"no safetensors under {w_dir}")

    def flush(name):
        save_file(buckets.pop(name), str(out / name), metadata={"format": "pt"})
        written[name] = (out / name).stat().st_size

    for sp in src:
        hdr, base = open_shard(sp)
        keys = [k for k in hdr if k != "__metadata__"]
        with open(sp, "rb") as f:
            for k in keys:
                # LlamaBiModel *is* the model, so save_pretrained emits embed_tokens.weight,
                # not model.embed_tokens.weight. The mirror is a LlamaForCausalLM and does
                # carry the prefix; strip it, and drop the lm_head it has and we do not.
                nk = k[len("model."):] if k.startswith("model.") else k
                if nk == "lm_head.weight" or k == "lm_head.weight":
                    continue
                w = read_tensor(f, hdr, base, k)
                mod = nk[: -len(".weight")] if nk.endswith(".weight") else nk
                if mod.endswith(QUANT_SUFFIX):
                    for ad in (ada, adb):
                        w = ad.apply(w, nk)
                    qw, qs = bnb.functional.quantize_4bit(
                        w.to(torch.bfloat16).to(dev), blocksize=64,
                        compress_statistics=True, quant_type="nf4")
                    group = {nk: qw.cpu()}
                    for sk, sv in qs.as_dict(packed=True).items():
                        group[f"{nk}.{sk}"] = sv.cpu() if torch.is_tensor(sv) else sv
                    done += 1
                else:
                    if nk in ada.A or nk in adb.A:
                        raise SystemExit(f"{nk} has a LoRA but is not in QUANT_SUFFIX")
                    group = {nk: w}
                for gk, gv in group.items():
                    tgt = layout.get(gk) if layout else sp.name
                    if layout and tgt is None:
                        raise SystemExit(f"--like has no slot for {gk}")
                    buckets.setdefault(tgt, {})[gk] = gv
                    wmap[gk] = tgt
                    if layout and set(buckets[tgt]) == expect[tgt]:
                        flush(tgt)
                del w
        if not layout and sp.name in buckets:
            flush(sp.name)
        print(f"    {sp.name}: {done} quantized so far, peak VRAM "
              f"{torch.cuda.max_memory_allocated()/2**30:.3f} GB, RSS "
              f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f} GB")

    for name in list(buckets):                 # anything --like left short
        flush(name)
    total = len(wmap)
    index = {"metadata": {"total_size": sum(written.values())}, "weight_map": wmap}
    (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    SIDE = ("config.json", "tokenizer.json", "tokenizer_config.json",
            "special_tokens_map.json", "chat_template.jinja")
    if args.like:
        for f in SIDE:
            s = Path(args.like) / f
            if s.is_file():
                (out / f).write_bytes(s.read_bytes())
    else:
        # Without a reference to copy from, synthesise the sidecar files, or the result is
        # a directory of weights that nothing can load. The tokenizer comes from the
        # adapter repo (which ships one); the config from the base, relabelled.
        for f in SIDE[1:]:
            s = a_dir / f
            if s.is_file():
                (out / f).write_bytes(s.read_bytes())
        cfg = json.loads((w_dir / "config.json").read_text())
        cfg["architectures"] = ["LlamaBiModel"]
        cfg["dtype"] = cfg["torch_dtype"] = "bfloat16"
        cfg["quantization_config"] = {
            "quant_method": "bitsandbytes", "load_in_4bit": True, "load_in_8bit": False,
            "_load_in_4bit": True, "_load_in_8bit": False,
            "bnb_4bit_quant_type": "nf4", "bnb_4bit_use_double_quant": True,
            "bnb_4bit_compute_dtype": "bfloat16", "bnb_4bit_quant_storage": "uint8",
            "llm_int8_enable_fp32_cpu_offload": False, "llm_int8_has_fp16_weight": False,
            "llm_int8_skip_modules": None, "llm_int8_threshold": 6.0}
        (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
        print("  wrote a synthesised config.json (no --like given), so this loads; it will "
              "not\n  checksum against a --like build, whose config came from transformers "
              "itself")
    # The whole point of this path is that neither number is the model's size.
    print(f"\n  {done} tensors quantized, {total} written, {time.perf_counter()-t0:.1f} s"
          f"\n  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.3f} GB, "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20:.2f} GB")

    if args.verify_against:
        ref = Path(args.verify_against)
        rmap = json.loads((ref / "model.safetensors.index.json").read_text())["weight_map"]
        omap = index["weight_map"]
        if set(rmap) != set(omap):
            only = sorted(set(rmap) ^ set(omap))[:5]
            print(f"  KEY SET DIFFERS ({len(set(rmap)^set(omap))}), e.g. {only}")
            raise SystemExit(1)
        handles, bad = {}, 0
        for k in sorted(rmap):
            for d, m in ((ref, rmap), (out, omap)):
                key = (d, m[k])
                if key not in handles:
                    handles[key] = safe_open(str(d / m[k]), framework="pt")
            a = handles[(ref, rmap[k])].get_tensor(k)
            b = handles[(out, omap[k])].get_tensor(k)
            if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                bad += 1
                if bad <= 5:
                    print(f"  DIFFERS {k}  {tuple(a.shape)}/{a.dtype} vs "
                          f"{tuple(b.shape)}/{b.dtype}")
        print(f"  verify: {len(rmap)-bad}/{len(rmap)} tensors bit-identical to {ref}")
        raise SystemExit(1 if bad else 0)


if __name__ == "__main__":
    main()
