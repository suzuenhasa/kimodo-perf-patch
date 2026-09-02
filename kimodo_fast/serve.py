#!/usr/bin/env python
"""Kimodo, resident and batched. ~690 ms per clip, 5.8 GB, arbitrary prompts.

    from serve import KimodoFast
    k = KimodoFast("./enc_nf4")                  # ~12 s, once
    clips = k.generate(["a person walks forward.", "a zombie staggers."])
    # each clip is Kimodo's own output dict: posed_joints, global_rot_mats, root_positions...

or from the shell:

    python serve.py "a person walks forward." "a person waves hello." --out ./out

Three things are doing the work, all measured on an RTX 3090 against stock Kimodo's
3.465 s per clip and 101 s startup:

  nf4 encoder      15 GB -> 4.33 GB, 101 s -> 6.9 s to load, and 1.6x FASTER per encode
                   than stock, because merging the adapters removes 448 LoRA matmuls
                   from every forward
  regular CFG      the shipped `separated` mode evaluates a third guidance term that is
                   identically zero when there are no constraints -- 2 denoiser arms, not 3
  35 DDIM steps    the paper fixes 100 and never ablates it; jerk and foot-skate hold flat
                   down to 35, and the knee is between 35 and 25

plus batching: distinct prompts go through one denoiser pass, and prompts are grouped by
TOKEN LENGTH before encoding.

That last one is not cosmetic. Kimodo pins the encoder to batch_size=1 because batching
changes the embedding (cosine 0.83). The cause is not precision, as the shipped comment
says -- it is padding: LlamaBiModel's bidirectional mask override drops the padding mask,
so pad tokens are visible to attention and corrupt the real tokens' hidden states by ~169%
relative. Equal-length prompts have nothing to pad, so bucketing sidesteps it entirely and
measures cosine >= 0.99996 against batch 1. Mixed-length batching is NEVER done here.

For the denoiser speedups also apply, in your kimodo checkout:
    patch -p1 < patches/0002-device-and-diffusion-vars.patch
    patch -p1 < patches/0003-admm-smoother.patch
Both are bit-identical to stock output; without them you lose ~10%.
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

CANONICAL = "meta-llama/Meta-Llama-3-8B-Instruct"
BF16_BASE = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp"
BF16_PEFT = "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised"
# --encoder bf16 loads Kimodo's original 15 GB encoder instead of the nf4 one. It is the
# slower, larger arm and it is not the default: nf4 sits 0.9839 cosine from it, and the
# body motion that falls out lands 24.9 mm RMS away -- inside Kimodo's own 26.7 mm
# constraint error, i.e. under the model's own noise floor. Keep it for A/B work.
BF16_ALIASES = {"bf16", "full", "original", "fp16", "unquantized"}


class _Table:
    """Stands in for the 8B at generation time. load_model() sets model_cfg['text_encoder']
    = None when given one, so the encoder is never instantiated inside Kimodo."""

    def __init__(self, device):
        self.table, self.device = {}, device

    def __call__(self, texts):
        texts = [texts] if isinstance(texts, str) else texts
        missing = [t for t in texts if t not in self.table]
        if missing:
            raise KeyError(
                f"prompt not embedded: {missing!r}. Kimodo runs sanitize_text() on prompts "
                f"before they reach the encoder, so cache keys must be sanitised too.")
        return torch.cat([self.table[t] for t in texts], 0).to(self.device), [1] * len(texts)

    def to(self, device=None, dtype=None):
        return self

    def eval(self):
        return self


class KimodoFast:
    def __init__(self, encoder_dir, model="kimodo-soma-rp", device="cuda:0",
                 steps=35, cfg_weight=2.0, cfg_type="regular", verbose=True):
        from kimodo import load_model
        from kimodo.model.llm2vec import LLM2Vec
        from kimodo.model.llm2vec.models.bidirectional_llama import LlamaBiModel
        from transformers import AutoTokenizer

        self.device, self.steps, self.cfg_weight = device, steps, cfg_weight
        # 'regular' drops Kimodo's third guidance arm, which is provably zero with no
        # constraints; 'separated' is stock. Kept switchable so the speedup is reversible.
        self.cfg_type = cfg_type
        t0 = time.perf_counter()
        if str(encoder_dir).lower() in BF16_ALIASES:
            import os
            base, peft = BF16_BASE, BF16_PEFT
            te = os.environ.get("TEXT_ENCODERS_DIR")
            if te:            # same gated-repo shim the builders use
                base, peft = (os.path.join(te, r) if os.path.isdir(os.path.join(te, r))
                              else r for r in (base, peft))
            if verbose:
                print("  loading the ORIGINAL bf16 encoder (~15 GB, ~101 s). nf4 is the "
                      "default;\n  this arm exists for A/B work.")
            self.enc = LLM2Vec.from_pretrained(base, peft, torch_dtype=torch.bfloat16,
                                               pooling_mode="mean", max_length=512,
                                               doc_max_length=400, skip_instruction=True)
            self.enc.model = self.enc.model.to(device).eval()
        else:
            m = LlamaBiModel.from_pretrained(str(encoder_dir), device_map={"": 0}).eval()
            # llm2vec.py:173 branches on this to add the Llama-3 chat header.
            # save_pretrained rewrote it to a local path; losing it silently changes
            # every embedding.
            m.config._name_or_path = CANONICAL
            tok = AutoTokenizer.from_pretrained(str(encoder_dir))
            tok.pad_token, tok.padding_side = tok.eos_token, "left"
            self.enc = LLM2Vec(model=m, tokenizer=tok, pooling_mode="mean", max_length=512,
                               doc_max_length=400, skip_instruction=True)
        for p in self.enc.parameters():
            p.requires_grad = False
        enc_s = time.perf_counter() - t0

        self._stub = _Table(device)
        t0 = time.perf_counter()
        self.model = load_model(model, device=device, default_family="Kimodo",
                                text_encoder=self._stub)
        den_s = time.perf_counter() - t0
        self.fps = self.model.fps
        self._cache = {}
        self._out_cache = {}
        # patches/0005 makes padded batches safe. Detect it rather than assume, so an
        # unpatched checkout silently keeps the bucketing workaround instead of silently
        # producing corrupted embeddings.
        try:
            import kimodo.model.llm2vec.models.bidirectional_llama as _bl
            self._bidi_fixed = getattr(_bl, "_CREATE_BIDIRECTIONAL_MASK", None) is not None
        except Exception:
            self._bidi_fixed = False
        if verbose and not self._bidi_fixed:
            print("  note: patches/0005 not applied -- length-bucketing prompts to avoid "
                  "padding.\n        Apply it and mixed-length batches go through as one call.")
        if verbose:
            gb = torch.cuda.memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            print(f"  KimodoFast ready in {enc_s + den_s:.1f} s "
                  f"({enc_s:.1f} encoder + {den_s:.1f} denoiser), {gb:.2f} GB resident, "
                  f"{self.steps} steps")

    # ---- text -----------------------------------------------------------------------
    @staticmethod
    def sanitize(prompts):
        """Kimodo rewrites prompts (capitalisation, trailing period) in sanitize_text before
        they reach the text encoder, so anything keyed by the raw string misses. Key
        everything by the sanitised form instead -- it is idempotent, so passing already
        sanitised prompts through changes nothing."""
        try:
            from kimodo.sanitize import sanitize_texts
            return sanitize_texts(list(prompts))
        except Exception:                       # keep working if that helper ever moves
            out = []
            for t in prompts:
                t = " ".join(str(t).split())
                if t and t[0].islower():
                    t = t[0].upper() + t[1:]
                if t and t[-1] not in ".!?":
                    t += "."
                out.append(t)
            return out

    def _token_len(self, text):
        return len(self.enc.tokenizer(self.enc.prepare_for_tokenization(text))["input_ids"])

    def embed(self, prompts, max_bucket=16, bucket=None):
        """Embed the prompts, caching by sanitised text.

        `bucket` groups prompts by token length so no batch ever contains padding. That was
        a workaround: padding used to corrupt the encoder badly (cosine 0.839 against
        one-at-a-time), because the bidirectional mask override silently went causal
        whenever the attention mask held a zero. patches/0005 fixes that at the source and
        restores 0.999957, so bucketing is no longer needed for correctness -- and dropping
        it lets a mixed-length set go through as ONE call instead of one per distinct
        length. Defaults to off when the patch is present, on when it is not.
        """
        prompts = self.sanitize(prompts)
        todo = [p for p in dict.fromkeys(prompts) if p not in self._cache]
        if not todo:
            return
        if bucket is None:
            bucket = not self._bidi_fixed
        buckets = defaultdict(list)
        for p in todo:
            buckets[self._token_len(p) if bucket else 0].append(p)
        for _, group in sorted(buckets.items()):
            for i in range(0, len(group), max_bucket):
                chunk = group[i:i + max_bucket]
                with torch.no_grad():
                    out = self.enc.encode(chunk, batch_size=len(chunk),
                                          show_progress_bar=False, device=self.device)
                a = out.float().cpu() if torch.is_tensor(out) else torch.as_tensor(out).float()
                for j, p in enumerate(chunk):
                    self._cache[p] = a[j:j + 1, None, :]      # [1, 1, D] -- the contract

    # ---- constraints ----------------------------------------------------------------
    def root_path(self, waypoints, nframes, every=10, heading=None):
        """A Root2DConstraintSet from (x, z) waypoints -- "walk this path".

        Waypoints are resampled to one target per `every` frames by walking the polyline at
        constant speed, which is what makes a drawn path behave like a route rather than a
        set of poses to teleport between. Kimodo is Y-up, so the constrained pair is (x, z)
        and height stays free.
        """
        import torch
        from kimodo.constraints import Root2DConstraintSet

        w = np.asarray(waypoints, dtype=np.float64)
        if w.ndim != 2 or w.shape[1] not in (2, 3):
            raise ValueError(f"waypoints must be (N,2) or (N,3), got {w.shape}")
        if w.shape[1] == 3:
            w = w[:, [0, 2]]                       # drop height; Root2D constrains x and z
        seg = np.linalg.norm(np.diff(w, axis=0), axis=1)
        if not seg.sum():
            raise ValueError("waypoints do not move")
        s = np.concatenate([[0.0], np.cumsum(seg)]) / seg.sum()

        idx = np.arange(0, nframes, every)
        u = idx / max(nframes - 1, 1)
        path = np.stack([np.interp(u, s, w[:, 0]), np.interp(u, s, w[:, 1])], -1)
        # Built on the model's device: conditioning.py indexes the data with these and
        # torch refuses a cpu/cuda mix, so a CPU constraint fails deep inside
        # create_conditions rather than at the call site.
        dev = self.device
        return Root2DConstraintSet(
            skeleton=self.model.skeleton,
            frame_indices=torch.as_tensor(idx, dtype=torch.long, device=dev),
            smooth_root_2d=torch.as_tensor(path, dtype=torch.float32, device=dev),
            global_root_heading=(None if heading is None
                                 else torch.as_tensor(heading, dtype=torch.float32, device=dev)))

    # ---- motion ---------------------------------------------------------------------
    def generate_sequence(self, segments, transition_frames=2, seed=0, variations=1,
                          post_processing=True, constraints=None, first_heading=None,
                          steps=None, cache=True):
        # steps= overrides the instance default for one call: the point is progressive
        # refinement -- a cheap preview while the user drags, the full count on release.
        # See FINDINGS.md; a CONSTRAINED preview is far cheaper than an unconstrained one,
        # because the constraint does the positioning the extra steps would have to.
        """One continuous motion from an ordered list of (prompt, seconds) segments.

            k.generate_sequence([("a person jumps", 3.0),
                                 ("a person walks to the left", 5.0),
                                 ("a person stands still thinking", 5.0)])

        This is `multi_prompt=True`: each segment is denoised conditioned on the tail of the
        previous one, with heading carried across, so it is a continuation rather than a
        crossfade. Segments are therefore SEQUENTIAL -- this cannot be batched, and is the
        opposite trade to generate(), which batches independent clips.

        Kimodo is trained to ~10 s; the demo caps a segment there and so does this.

        `transition_frames` defaults to 2 rather than the demo's 5, because 5 makes later
        segments ignore their own prompt. "A person stands still" runs at 0.02 m/s alone
        and 0.88 m/s as the third segment after a walk -- it just keeps walking. Measured
        across the triple (jump 3 s, walk left 5 s, stand still 5 s), summed error against
        each prompt's standalone speed, and the worst jerk in a +/-5 frame window at each
        seam relative to the clip's median:

            tf    jumps  walks  still   err    seam@90  seam@240
             1     0.19   0.93   0.12   0.21      6.86     23.55
             2     0.19   0.88   0.08   0.22      4.53      7.03
             3     0.19   0.90   0.17   0.29      3.09     14.84
             5     0.19   0.88   0.88   1.02      1.63      7.40
            10     0.19   0.62   0.79   1.19
            30     0.19   0.09   0.03      -                          all motion suppressed

        1 follows the prompts best but pops at the seam. 2 matches it on prompts while its
        worse seam is better than the default's. 30 looks perfect on "stands still" only
        because it flattens everything, including the walk -- a trap worth naming.

        One prompt triple, one seed, and the seam numbers are not monotonic in tf, so treat
        this as a starting point rather than a law. Raise it if you see a discontinuity;
        lower it if a segment ignores its text.
        """
        from kimodo.tools import seed_everything

        segs = [(s, 5.0) if isinstance(s, str) else tuple(s) for s in segments]
        if not segs:
            raise ValueError("no segments")
        over = [p for p, d in segs if d > 10.0]
        if over:
            raise ValueError(f"segment longer than the 10 s Kimodo is trained for: {over[0]!r}. "
                             f"Split it into consecutive segments instead.")
        prompts = self.sanitize([p for p, _ in segs])
        frames = [int(d * self.fps) for _, d in segs]

        self.embed(prompts)
        self._stub.table = {p: self._cache[p] for p in prompts}
        seed_everything(seed)
        # _multiprompt reads bs from num_samples (kimodo_model.py:491) and then does
        # [0.0] * bs for the starting heading, so leaving it None fails with a TypeError
        # rather than defaulting to one sample.
        key = ("seq", tuple(prompts), tuple(frames), transition_frames, seed,
               steps or self.steps, post_processing, first_heading,
               constraints is not None, variations)
        if cache and constraints is None and key in self._out_cache:
            return self._out_cache[key]

        out = self.model(prompts, frames, num_denoising_steps=steps or self.steps,
                         num_samples=variations,
                         multi_prompt=True, num_transition_frames=transition_frames,
                         constraint_lst=constraints or [],
                         post_processing=post_processing, return_numpy=True,
                         cfg_type=self.cfg_type, cfg_weight=self._cfgw(),
                         first_heading_angle=first_heading)
        def _slice(i):
            return {k: (v[i] if isinstance(v, np.ndarray) and v.ndim and v.shape[0] == variations
                        else v) for k, v in out.items()}
        # variations>1 returns a list: these are alternative takes of the SAME timeline,
        # which is the only thing Kimodo's batch dimension can carry here.
        clip = _slice(0) if variations == 1 else [_slice(i) for i in range(variations)]
        if cache and constraints is None:
            # Small on purpose: these are megabytes each, and the hit that matters is a UI
            # replaying the same timeline, not a long history.
            if len(self._out_cache) >= 8:
                self._out_cache.pop(next(iter(self._out_cache)))
            self._out_cache[key] = clip
        return clip

    def generate_timelines(self, timelines, **kw):
        """Several DIFFERENT timelines.

        Sequential, and that is not an oversight. Kimodo's multi-prompt path does
        `texts_bs = [text for _ in range(num_samples)]` (kimodo_model.py:164), i.e. it
        replicates one segment's text across the batch, so the batch dimension carries
        alternative takes of a single timeline -- not different timelines. Batching
        distinct timelines would mean patching the model to accept per-sample prompts.

        For alternative takes of ONE timeline, use generate_sequence(..., variations=N),
        which is genuinely batched and close to free.
        """
        return [self.generate_sequence(t, **kw) for t in timelines]

    def _cfgw(self):
        """separated takes [text_cfg, constraint_cfg]; regular takes a scalar. cfg.py
        asserts both, so they have to travel together."""
        if self.cfg_type == "separated" and not isinstance(self.cfg_weight, (list, tuple)):
            return [self.cfg_weight] * 2
        return self.cfg_weight

    def generate(self, prompts, duration=5.0, batch=16, seed=0, post_processing=True,
                 constraints=None, steps=None):
        """Independent clips, batched. For a timeline use generate_sequence().

        `constraints` is per-prompt: a list as long as `prompts`, each entry a list of
        constraint sets (root path, end-effector, keyframe). Constrained batches are split
        to match, so they do not mix with unconstrained ones.
        """
        from kimodo.tools import seed_everything
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = self.sanitize(prompts)
        if constraints is not None and len(constraints) != len(prompts):
            raise ValueError(f"constraints must be one entry per prompt: "
                             f"{len(constraints)} vs {len(prompts)}")
        self.embed(prompts)
        self._stub.table = {p: self._cache[p] for p in prompts}
        nframes = int(duration * self.fps)

        cfgw = self._cfgw()

        clips = []
        for i in range(0, len(prompts), batch):
            chunk = prompts[i:i + batch]
            ccl = [] if constraints is None else constraints[i:i + batch]
            seed_everything(seed)
            out = self.model(chunk, [nframes] * len(chunk),
                             num_denoising_steps=steps or self.steps, multi_prompt=False,
                             constraint_lst=ccl,
                             post_processing=post_processing, return_numpy=True,
                             cfg_type=self.cfg_type, cfg_weight=cfgw)
            for j in range(len(chunk)):
                clips.append({k: (v[j] if isinstance(v, np.ndarray) and v.shape[:1] ==
                                  (len(chunk),) else v) for k, v in out.items()})
        return clips


def parse_timeline(spec):
    """'a person walks:3 | a person waves:2 | a person jumps' -> [(prompt, seconds), ...]

    Segments are separated by '|'. A trailing ':<seconds>' sets that segment's duration;
    without one it defaults to 5 s. Prompts may themselves contain a colon -- only a
    trailing colon followed by a number is read as a duration.
    """
    segs = []
    for part in spec.split("|"):
        part = part.strip()
        if not part:
            continue
        secs = 5.0
        if ":" in part:
            head, tail = part.rsplit(":", 1)
            try:
                secs = float(tail.strip())
                part = head.strip()
            except ValueError:
                pass
        if not part:
            raise SystemExit(f"empty prompt in timeline segment: {spec!r}")
        segs.append((part, secs))
    if not segs:
        raise SystemExit(f"no segments in timeline: {spec!r}")
    return segs


def _run_timelines(args, timelines):
    k = KimodoFast(args.encoder, steps=args.steps, cfg_type=args.cfg_type)
    total = sum(len(t) for t in timelines)
    print(f"  {len(timelines)} timeline(s), {total} segment(s)"
          + (f", {args.variations} variations" if args.variations > 1 else ""))
    t0 = time.perf_counter()
    if args.variations > 1:
        # batched: alternative takes of one timeline
        clips = k.generate_sequence(timelines[0], transition_frames=args.transition_frames,
                                    seed=args.seed, variations=args.variations)
        labels = [(timelines[0], i) for i in range(args.variations)]
    else:
        # sequential: the model cannot batch distinct timelines (see generate_timelines)
        clips = k.generate_timelines(timelines,
                                     transition_frames=args.transition_frames,
                                     seed=args.seed)
        labels = [(t, 0) for t in timelines]
    dt = time.perf_counter() - t0
    print(f"  {len(clips)} clip(s) in {dt:.3f} s  ({1000*dt/len(clips):.0f} ms each, "
          f"{1000*dt/max(1,total):.0f} ms/segment)")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(clips) - 1)))
    meta = []
    for i, (c, (segs, vi)) in enumerate(zip(clips, labels)):
        f = outdir / f"timeline_{i:0{width}d}.npz"
        np.savez_compressed(f, **{kk: v for kk, v in c.items() if isinstance(v, np.ndarray)})
        desc = " | ".join(f"{p}:{d:g}" for p, d in segs)
        frames = int(c["posed_joints"].shape[0]) if "posed_joints" in c else None
        meta.append({"file": f.name, "segments": [[p, d] for p, d in segs],
                     "variation": vi, "frames": frames})
        print(f"    {f}  {frames} frames  {desc}")
    (outdir / "timelines.json").write_text(json.dumps(meta, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="*", help="prompts on the command line; omit if "
                    "using --prompts-file")
    ap.add_argument("--prompts-file", "-f", type=Path,
                    help="text file with one prompt per line. Blank lines and lines "
                         "starting with # are skipped. Combined with any prompts given "
                         "on the command line, which come first.")
    ap.add_argument("--encoder", default="./enc_nf4",
                    help="path to an nf4 encoder (default), or 'bf16' for the original "
                         "15 GB one")
    ap.add_argument("--timeline", action="append", default=[], metavar="SPEC",
                    help="one continuous motion from segments separated by '|', each "
                         "optionally ':<seconds>'. Repeat the flag for several timelines. "
                         "e.g. --timeline \'a person walks:3|a person jumps:2\'")
    ap.add_argument("--timelines-file", type=Path,
                    help="one timeline per line, same syntax as --timeline. Blank lines "
                         "and lines starting with # are skipped.")
    ap.add_argument("--variations", type=int, default=1,
                    help="alternative takes of a timeline, generated in one batch. Only "
                         "valid with a single timeline.")
    ap.add_argument("--transition-frames", type=int, default=2,
                    help="frames blended between timeline segments (default 2; the demo's "
                         "5 makes later segments ignore their own prompt)")
    ap.add_argument("--out", default="./out")
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=35)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cfg-type", default="regular", choices=["regular", "separated"],
                    help="'separated' is stock Kimodo; 'regular' drops the third "
                         "guidance arm, which is zero without constraints")
    args = ap.parse_args()

    timelines = [parse_timeline(t) for t in args.timeline]
    if args.timelines_file:
        if not args.timelines_file.is_file():
            raise SystemExit(f"no such file: {args.timelines_file}")
        for line in args.timelines_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                timelines.append(parse_timeline(line))

    if timelines:
        if args.prompts or args.prompts_file:
            raise SystemExit("--timeline/--timelines-file and plain prompts are separate "
                             "modes; run them separately")
        if args.variations > 1 and len(timelines) > 1:
            raise SystemExit("--variations applies to a single timeline; got "
                             f"{len(timelines)}")
        _run_timelines(args, timelines)
        return

    prompts = list(args.prompts)
    if args.prompts_file:
        if not args.prompts_file.is_file():
            raise SystemExit(f"no such file: {args.prompts_file}")
        for line in args.prompts_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                prompts.append(line)
    if not prompts:
        raise SystemExit("no prompts: pass them as arguments or with --prompts-file")
    # Kimodo rewrites prompts before they reach the encoder, so a file and the command line
    # must agree on the sanitised form or the same prompt would embed twice.
    seen, unique = set(), []
    for p_ in KimodoFast.sanitize(prompts):
        if p_ not in seen:
            seen.add(p_); unique.append(p_)
    if len(unique) != len(prompts):
        print(f"  {len(prompts) - len(unique)} duplicate prompt(s) dropped")
    prompts = unique
    print(f"  {len(prompts)} prompt(s)")

    k = KimodoFast(args.encoder, steps=args.steps, cfg_type=args.cfg_type)
    t0 = time.perf_counter()
    clips = k.generate(prompts, duration=args.duration, batch=args.batch, seed=args.seed)
    dt = time.perf_counter() - t0
    print(f"  {len(clips)} clip(s) in {dt:.3f} s  ({1000*dt/len(clips):.0f} ms each, "
          f"{60*len(clips)/dt:.0f} clips/min)")

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    width = max(2, len(str(len(clips) - 1)))
    for i, (p_, c) in enumerate(zip(prompts, clips)):
        f = outdir / f"clip_{i:0{width}d}.npz"
        np.savez_compressed(f, **{kk: v for kk, v in c.items() if isinstance(v, np.ndarray)})
        print(f"    {f}  {p_!r}")
    (outdir / "prompts.json").write_text(json.dumps(prompts, indent=2))


if __name__ == "__main__":
    main()
