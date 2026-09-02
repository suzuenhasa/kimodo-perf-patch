#!/usr/bin/env python
"""Run Kimodo's own benchmark generation, but with OUR configuration.

Answering "does what we ship hurt quality?" means generating with what we ship, not with
stock Kimodo at a lower step count. This drives `benchmark/generate_eval.py` unchanged --
same discovery, same per-case frame counts, same seeding, same cropping, same npz layout --
and swaps only the model:

    --encoder <dir>   the nf4 checkpoint, instead of the 15 GB bf16 one
    --cfg_type        'regular' drops the third guidance arm, which is zero without
                      constraints. Kimodo's default is 'separated'.
    --diffusion_steps 35 rather than 100.
    --encoder_batch   lifts the batch_size=1 pin in llm2vec_wrapper.py, which patches/0005
                      makes safe.

Everything the benchmark does around the model is therefore identical between arms, and the
only thing that varies is the thing under test.

    python scripts/generate_eval_fast.py --benchmark <testsuite> --output <dir> \
        --encoder ../enc_nf4 --diffusion_steps 35 --cfg_type regular --postprocess
"""
import argparse
import os
import sys
from pathlib import Path

import torch

CANONICAL = "meta-llama/Meta-Llama-3-8B-Instruct"


class NF4Encoder:
    """LLM2VecEncoder's interface, backed by the pre-quantized checkpoint.

    Kimodo's own wrapper pins the internal batch to 1 and blames transformers precision.
    It is not precision -- see FINDINGS.md -- and with patches/0005 applied a larger batch
    matches one-at-a-time to cosine 0.99996, so --encoder_batch is safe to raise.
    """

    def __init__(self, enc_dir, llm_dim=4096, device="cuda:0", batch_size=1):
        import numpy as np
        from kimodo.model.llm2vec import LLM2Vec
        from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
        from transformers import AutoTokenizer

        self.np, self.llm_dim, self._device, self._bs = np, llm_dim, device, batch_size
        m = LlamaBiModel.from_pretrained(str(enc_dir), device_map={"": 0}).eval()
        m.config._name_or_path = CANONICAL      # chat-header branch, llm2vec.py:173
        tok = AutoTokenizer.from_pretrained(str(enc_dir))
        tok.pad_token, tok.padding_side = tok.eos_token, "left"
        self.model = LLM2Vec(model=m, tokenizer=tok, pooling_mode="mean", max_length=512,
                             doc_max_length=400, skip_instruction=True)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    def to(self, device):
        return self

    def eval(self):
        return self

    def get_device(self):
        return self.model.model.device

    def __call__(self, text):
        is_string = isinstance(text, str)
        if is_string:
            text = [text]
        with torch.no_grad():
            enc = self.model.encode(text, batch_size=self._bs, show_progress_bar=False,
                                    device=self._device)
        assert self.llm_dim == enc.shape[-1]
        enc = enc[:, None]
        lengths = self.np.ones(len(enc), dtype=int).tolist()
        if is_string:
            return enc[0], lengths[0]
        return enc, lengths


class Arm:
    """Forces our guidance settings onto every call the benchmark makes.

    cfg_type and cfg_weight travel together: cfg.py asserts a scalar for 'regular' and a
    pair for 'separated', so setting one without the other fails loudly at the first batch.
    """

    def __init__(self, model, cfg_type, cfg_weight):
        self._m, self._t = model, cfg_type
        self._w = cfg_weight if cfg_type == "separated" else float(cfg_weight[0])

    def __call__(self, *a, **kw):
        kw["cfg_type"], kw["cfg_weight"] = self._t, self._w
        return self._m(*a, **kw)

    def __getattr__(self, n):
        return getattr(self._m, n)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--encoder", default=None,
                    help="nf4 checkpoint dir; omit to use Kimodo's own 15 GB bf16 encoder")
    ap.add_argument("--diffusion_steps", type=int, default=35)
    ap.add_argument("--cfg_type", default="regular", choices=["regular", "separated"])
    ap.add_argument("--cfg_weight", type=float, nargs="+", default=[2.0, 2.0])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--encoder_batch", type=int, default=1,
                    help="internal llm2vec batch; >1 needs patches/0005")
    ap.add_argument("--postprocess", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    # bootstrap.sh clones kimodo to $ROOT/kimodo, i.e. a sibling of this repo.
    _default_src = Path(__file__).resolve().parents[2] / "kimodo"
    sys.path.insert(0, str(Path(os.environ.get("KIMODO_SRC", _default_src)) / "benchmark"))
    import generate_eval as ge

    real_load = ge.load_model

    def load(*args, **kw):
        enc = None
        if a.encoder:
            print(f"  encoder: {a.encoder} (nf4, internal batch {a.encoder_batch})")
            enc = NF4Encoder(a.encoder, batch_size=a.encoder_batch)
        else:
            print("  encoder: Kimodo's own bf16")
        kw["text_encoder"] = enc
        r = real_load(*args, **kw)
        print(f"  cfg_type={a.cfg_type}  steps={a.diffusion_steps}")
        # load_model returns (model, name) when called with return_resolved_name=True,
        # which generate_eval does. Wrap the model, keep the tuple shape.
        if isinstance(r, tuple):
            return Arm(r[0], a.cfg_type, a.cfg_weight), *r[1:]
        return Arm(r, a.cfg_type, a.cfg_weight)

    ge.load_model = load
    sys.argv = ["generate_eval.py",
                "--benchmark", a.benchmark, "--output", a.output,
                "--diffusion_steps", str(a.diffusion_steps),
                "--batch_size", str(a.batch_size)]
    if a.postprocess:
        sys.argv.append("--postprocess")
    if a.overwrite:
        sys.argv.append("--overwrite")
    ge.main()


if __name__ == "__main__":
    main()
