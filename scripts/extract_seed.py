"""Pull only the BVH files a testsuite subset references out of soma_uniform.tar.gz.

The tarball is ~42 GB of gzipped BVH text. Fully extracted it would not fit beside the
weights on a 150 GB box, and we need a few thousand files out of it, so extract by name.

A .tar.gz is a single gzip stream, so this reads it once, start to finish, and writes out
members as it meets them -- no seeking, no full extraction.
"""
import json, os, sys, tarfile, time
from pathlib import Path

SUBSET = Path(sys.argv[1] if len(sys.argv) > 1
              else "testsuite/content/text2motion")
TARBALL = Path(sys.argv[2] if len(sys.argv) > 2 else "soma_uniform.tar.gz")
OUT = Path(sys.argv[3] if len(sys.argv) > 3 else "soma_uniform")

wanted = {}
for dp, _, fn in os.walk(SUBSET):
    if "seed_motion.json" in fn:
        bp = json.load(open(Path(dp) / "seed_motion.json"))["bvh_path"]
        # metadata says BVH/<date>/<file>; the tarball holds soma_uniform/bvh/<date>/<file>
        member = "soma_uniform/" + bp.replace("BVH/", "bvh/", 1)
        # The metadata says BVH/, the tarball and create_benchmark.py both say bvh/.
        # Write the lowercase form, or create_benchmark dies with FileNotFoundError.
        wanted[member] = bp.replace("BVH/", "bvh/", 1)
print(f"  {len(wanted)} unique BVH files referenced by {SUBSET.name}")

OUT.mkdir(parents=True, exist_ok=True)
found = 0
t0 = time.perf_counter()
with tarfile.open(TARBALL, "r|gz") as tf:          # streaming: no random access
    for m in tf:
        if m.name not in wanted:
            continue
        dest = OUT / wanted[m.name]                 # keep the metadata's own layout
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = tf.extractfile(m)
        if src is None:
            continue
        with open(dest, "wb") as f:
            while chunk := src.read(1 << 20):
                f.write(chunk)
        found += 1
        if found % 200 == 0:
            print(f"    {found}/{len(wanted)}  {time.perf_counter()-t0:.0f} s")
        if found == len(wanted):
            break                                   # stop early; no need to read the rest

sz = sum(f.stat().st_size for f in OUT.rglob("*.bvh")) / 2**30
print(f"  extracted {found}/{len(wanted)} in {time.perf_counter()-t0:.0f} s, {sz:.2f} GB -> {OUT}")
if found < len(wanted):
    missing = [v for k, v in wanted.items() if not (OUT / v).is_file()]
    print(f"  MISSING {len(missing)}, e.g. {missing[:3]}")
