#!/usr/bin/env python
"""Prepare (and optionally upload) the nf4 encoder for Hugging Face.

Writes the three files the Llama 3 Community License requires alongside the weights, plus
a model card with the metadata the Hub actually reads, then uploads the folder.

    kimodo-publish-encoder --repo <you>/Llama-3-8B-LLM2Vec-kimodo-nf4 --dir ./enc_nf4
    kimodo-publish-encoder --repo ... --dir ./enc_nf4 --private --upload

Redistributing a Llama 3 derivative carries four obligations, and this script refuses to
run if it cannot satisfy them:

  1. ship a copy of the licence           -> LICENSE, fetched from Meta
  2. display "Built with Meta Llama 3"    -> first line of the model card body
  3. a NOTICE file with Meta's wording    -> NOTICE, verbatim
  4. the model NAME must begin "Llama 3"  -> asserted against --repo

https://www.llama.com/llama3/license/
"""
import argparse
import json
import sys
import urllib.request
from pathlib import Path

# Meta ships the licence as plain text in the reference repo; llama.com serves it as a
# 300 KB HTML page, which is a poor thing to drop next to the weights as LICENSE.
LICENSE_URLS = [
    "https://raw.githubusercontent.com/meta-llama/llama3/main/LICENSE",
    "https://www.llama.com/llama3/license/",
]
NOTICE = ("Meta Llama 3 is licensed under the Meta Llama 3 Community License, "
          "Copyright © Meta Platforms, Inc. All Rights Reserved.\n")
BASE = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"

CARD = """---
license: llama3
license_link: https://www.llama.com/llama3/license/
base_model: {base}
base_model_relation: quantized
library_name: transformers
pipeline_tag: feature-extraction
tags:
- llm2vec
- bitsandbytes
- nf4
- 4-bit
- text-embeddings
- motion-generation
extra_gated_prompt: >-
  This is a 4-bit quantization of a Meta Llama 3 derivative. By downloading it you agree to
  the Meta Llama 3 Community License at https://www.llama.com/llama3/license/
extra_gated_fields:
  I agree to the Meta Llama 3 Community License: checkbox
---

**Built with Meta Llama 3**

# {name}

A 4-bit (NF4, double-quantized) build of [`{base}`]({base_url}), the LLM2Vec text encoder
that [NVIDIA Kimodo](https://github.com/nv-tlabs/kimodo) uses for text conditioning.

Kimodo ships this encoder at bf16 with two LoRA adapters applied at load time. Both adapters
are merged here before quantization, because the reverse order does not work: loading a
quantized base and asking PEFT to merge a LoRA into it fails outright.

| | stock bf16 + 2 adapters | this |
|---|---:|---:|
| resident VRAM | 14.14 GB | **4.33 GB** |
| time to load | 101.1 s | **6.7 s** |
| encode, 1 prompt | 181.1 ms | **115.6 ms** |
| on disk | ~15 GB | **4.35 GB** |

It is *faster* as well as smaller: merging removes 448 unmerged LoRA matmuls from every
forward pass, which outweighs the dequantization cost. (int8 goes the other way -- 1.66x
slower, because `LLM.int8()` routes outlier columns through a separate fp16 path and this
encoder runs at batch 1.)

## Use

```python
from transformers import AutoTokenizer
from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
from kimodo.model.llm2vec import LLM2Vec

model = LlamaBiModel.from_pretrained("{repo}", device_map={{"": 0}})   # no quantization_config
model.config._name_or_path = "meta-llama/Meta-Llama-3-8B-Instruct"     # see the trap below
tok = AutoTokenizer.from_pretrained("{repo}")
tok.pad_token, tok.padding_side = tok.eos_token, "left"
enc = LLM2Vec(model=model, tokenizer=tok, pooling_mode="mean",
              max_length=512, doc_max_length=400, skip_instruction=True)
```

The quantization config travels in `config.json`, so do **not** pass a `BitsAndBytesConfig`.
Requires `bitsandbytes` and an NVIDIA Pascal-or-newer GPU (or Intel XPU/Gaudi, or CPU).

**The trap.** `prepare_for_tokenization` branches on
`config._name_or_path == "meta-llama/Meta-Llama-3-8B-Instruct"` to add the Llama 3 chat
header. Saving and reloading rewrites that field to the repo path, which silently drops the
header, changes the tokenization, and changes every embedding. Restore it, and assert on it.

**Batching.** Keep `batch_size=1` unless every prompt in the batch has the same token length.
Batching mixed lengths returns cosine ~0.83 -- a different prompt, not a different last bit.
The cause is padding, not precision: the bidirectional attention mask override drops the
padding mask, so pad tokens shift the real tokens' hidden states. Equal-length batches have
nothing to pad and hold cosine >= 0.99996.

## Quality

Against the bf16 original on 16 probe prompts, worst-case cosine **0.9839**. In motion terms,
driving Kimodo from this encoder moves the body **24.9 mm RMS** -- the same order as Kimodo's
own 26.7 mm full-body constraint error -- while the character's root path can end up 64 cm
away. About 90% of the total difference is that path, not the pose.

Not evaluated against FID or R-precision.

## Licence

Derived from Meta Llama 3 via [`{base}`]({base_url}). Use is governed by the
[Meta Llama 3 Community License](https://www.llama.com/llama3/license/), a copy of which is
included as `LICENSE`. See `NOTICE`.

Built with `kimodo-fast`: {project}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="e.g. you/Llama-3-8B-LLM2Vec-kimodo-nf4")
    ap.add_argument("--dir", default="./enc_nf4")
    ap.add_argument("--project", default="https://github.com/suzuenhasa/kimodo-perf")
    # Private by default. Accidentally publishing 4.35 GB of Llama-derived weights is a
    # great deal worse than accidentally keeping them to yourself, so make the safe choice
    # the one you get by saying nothing.
    ap.add_argument("--public", action="store_true",
                    help="publish publicly (default: private)")
    ap.add_argument("--upload", action="store_true", help="actually push; otherwise dry run")
    args = ap.parse_args()

    d = Path(args.dir)
    if not (d / "config.json").exists():
        raise SystemExit(f"{d} has no config.json -- run kimodo-build-encoder first")

    name = args.repo.split("/")[-1]
    # Obligation 4: derivative model names must begin with "Llama 3".
    if not name.lower().replace("_", "-").startswith("llama-3"):
        raise SystemExit(
            f"repo name {name!r} must begin with 'Llama-3'. The Meta Llama 3 Community "
            f"License requires derivative model names to start with 'Llama 3'.")

    cfg = json.loads((d / "config.json").read_text())
    if "quantization_config" not in cfg:
        raise SystemExit(f"{d}/config.json carries no quantization_config -- consumers would "
                         f"load this as full precision. Rebuild with kimodo-build-encoder.")

    # Obligation 1: ship a copy of the licence.
    lic = d / "LICENSE"
    if lic.exists() and lic.stat().st_size > 5000 and b"<html" not in lic.read_bytes()[:2000].lower():
        print("  LICENSE already present")
    else:
        err = None
        for url in LICENSE_URLS:
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    body = r.read()
                if b"<html" in body[:2000].lower():
                    err = f"{url} served HTML, not the licence text"
                    continue
                lic.write_bytes(body)
                print(f"  fetched LICENSE ({len(body)/1000:.0f} KB) from {url}")
                break
            except Exception as e:
                err = f"{url}: {e}"
        else:
            raise SystemExit(
                f"could not fetch the licence text ({err}). The Llama 3 licence requires "
                f"shipping a copy, so save it yourself and re-run:\n"
                f"    curl -Lo {lic} {LICENSE_URLS[0]}")

    # Obligation 3: NOTICE, verbatim.
    (d / "NOTICE").write_text(NOTICE)
    print("  wrote NOTICE")

    # Obligation 2 lives in the card body, first line.
    card = CARD.format(base=BASE, base_url=f"https://huggingface.co/{BASE}",
                       name=name, repo=args.repo, project=args.project)
    (d / "README.md").write_text(card)
    print("  wrote README.md (model card)")
    assert "Built with Meta Llama 3" in card

    private = not args.public
    size = sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2**30
    print(f"\n  {d} is ready: {size:.2f} GB -> {args.repo}, "
          + ("private" if private else "*** PUBLIC ***"))

    if not args.upload:
        print("\n  dry run. To publish:")
        print(f"    kimodo-publish-encoder --repo {args.repo} --dir {d} "
              f"{'--public ' if args.public else ''}--upload")
        print("  or by hand:")
        print(f"    huggingface-cli upload {args.repo} {d} "
              + ("" if args.public else "--private"))
        return

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    api = HfApi()
    api.create_repo(args.repo, private=private, exist_ok=True, repo_type="model")

    # create_repo(exist_ok=True) does NOT change the visibility of a repo that already
    # exists, so pushing to one you made public earlier would silently stay public while
    # this script printed "private". Set it explicitly, then read it back.
    try:
        api.update_repo_settings(repo_id=args.repo, repo_type="model", private=private)
    except (AttributeError, TypeError):
        api.update_repo_visibility(repo_id=args.repo, repo_type="model", private=private)

    actual = api.repo_info(args.repo, repo_type="model").private
    if actual != private:
        raise SystemExit(f"repo visibility is private={actual}, wanted private={private}. "
                         f"Refusing to upload weights into the wrong visibility.")
    print(f"  verified: repo is {'private' if actual else 'PUBLIC'} before any upload")

    api.upload_folder(folder_path=str(d), repo_id=args.repo, repo_type="model")
    final = api.repo_info(args.repo, repo_type="model").private
    print(f"\n  https://huggingface.co/{args.repo}  ({'private' if final else 'PUBLIC'})")


if __name__ == "__main__":
    main()
