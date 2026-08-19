#!/usr/bin/env python3
"""
add_flashhead.py — build a FlashHead into an EXISTING gemma4 or lfm25 container.

The converters build one from the checkpoint. This does the same job from a container
that already exists, which is the common case once you have converted a model: the
clustering needs the lm_head and nothing else, and re-running a full conversion to get
it would re-quantise 5-16 GB of experts for no reason.

What it costs is that the clustering sees the DEQUANTISED head rather than the
checkpoint's bf16. That is a real difference from `--flash` at conversion time but a
small one, and it is not the requant trap `TODO.md` warns about: nothing is
re-quantised here. The stored head is untouched, and the clustering only uses it to
decide which rows sit near each other. A q4_0 row is within d/2 of the original with
d ~ 1/8 of the row's max magnitude, which moves a direction by far less than the gap
between neighbouring clusters.

Both engines tie the head to the embedding table, so `embed_tokens` is what gets
clustered in either container.

    python3 tools/add_flashhead.py ./lfm-ct --src /path/to/LFM2.5-8B-A1B
    python3 tools/add_flashhead.py ./g4 --src /path/to/gemma-4 --probes 512

--src is only read for the control-token ids that must never be pruned (EOS and the
chat template's markers). Without it the head is built with none forced, which is
usually wrong: a model that cannot emit EOS does not stop.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashhead
from convert_lfm25 import (FMT_F32, FMT_Q40, FMT_Q80, dequant_rows, quant_rows)

FMT_I32 = 5
FLASH_TENSORS = ("flash_centroids", "flash_token_map", "flash_cluster_scale",
                 "flash_force")


def read_manifest(path):
    """-> (list of raw lines, cfg dict, list of dense entries)."""
    lines = open(path).read().splitlines()
    cfg, dense = {}, []
    for ln in lines:
        f = ln.split()
        if not f:
            continue
        if f[0] == "cfg" and len(f) >= 3:
            cfg[f[1]] = " ".join(f[2:])
        elif f[0] == "dense" and len(f) == 7:
            dense.append({"name": f[1], "off": int(f[2]), "len": int(f[3]),
                          "fmt": int(f[4]), "O": int(f[5]), "I": int(f[6])})
    return lines, cfg, dense


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", help="container directory (dense.bin + manifest.txt)")
    ap.add_argument("--src", help="checkpoint dir, read only for the forced "
                                  "control-token ids")
    ap.add_argument("--probes", type=int, default=0,
                    help="clusters probed per token (default: ~11%% of them)")
    ap.add_argument("--iters", type=int, default=10, help="k-means iterations")
    ap.add_argument("--cluster-size", type=int, default=flashhead.CLUSTER_SIZE)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    mpath = os.path.join(a.dir, "manifest.txt")
    dpath = os.path.join(a.dir, "dense.bin")
    if not os.path.exists(mpath) or not os.path.exists(dpath):
        sys.exit(f"{a.dir}: not a container (need manifest.txt and dense.bin)")
    lines, cfg, dense = read_manifest(mpath)

    emb = next((d for d in dense if d["name"] == "embed_tokens"), None)
    if emb is None:
        sys.exit("no embed_tokens in this container; both engines tie the lm_head to "
                 "it, so there is nothing to cluster")
    if emb["fmt"] not in (FMT_F32, FMT_Q40, FMT_Q80):
        sys.exit(f"embed_tokens has format {emb['fmt']}, which this tool cannot read "
                 f"(maple's q4a head already ships a FlashHead of its own)")
    V, D = emb["O"], emb["I"]

    with open(dpath, "rb") as f:
        f.seek(emb["off"])
        buf = f.read(emb["len"])
    rows = np.ascontiguousarray(dequant_rows(emb["fmt"], buf, V, D), dtype=np.float32)
    del buf

    print(f"clustering {V} x {D} lm_head rows "
          f"({'q4_0' if emb['fmt'] == FMT_Q40 else 'q8_0' if emb['fmt'] == FMT_Q80 else 'f32'} "
          f"in the container)", flush=True)
    fh = flashhead.build(rows, cluster_size=a.cluster_size, n_probes=a.probes,
                         iters=a.iters, seed=a.seed)
    force = np.asarray(flashhead.force_tokens(a.src, V), np.int32)
    if not force.size:
        print("!! no forced control tokens found%s: EOS can be pruned, and a model "
              "that cannot emit EOS does not stop" %
              ("" if a.src else " (pass --src)"))

    # Append. Existing offsets are untouched, so a container that a running engine has
    # open stays valid until it is reloaded.
    if any(d["name"] in FLASH_TENSORS for d in dense):
        sys.exit("this container already has a FlashHead; the tensors are appended "
                 "rather than replaced, so rebuild it from the checkpoint instead")
    blobs = [
        ("flash_centroids", quant_rows(FMT_Q40, fh["centroids"]), FMT_Q40,
         fh["centroids"].shape),
        ("flash_token_map", np.ascontiguousarray(fh["token_map"], np.int32).tobytes(),
         FMT_I32, fh["token_map"].shape),
        ("flash_cluster_scale",
         np.ascontiguousarray(fh["cluster_scale"], np.float32).tobytes(), FMT_F32,
         (1, fh["cluster_scale"].size)),
        ("flash_force",
         np.ascontiguousarray(force if force.size else np.zeros(1, np.int32),
                              np.int32).tobytes(), FMT_I32,
         (1, max(1, force.size))),
    ]
    off = os.path.getsize(dpath)
    added = []
    with open(dpath, "ab") as f:
        for name, b, fmt, shape in blobs:
            f.write(b)
            added.append((name, off, len(b), fmt, int(shape[0]), int(shape[1])))
            off += len(b)

    fcfg = dict(fh["cfg"], n_force=int(force.size))
    out = []
    for ln in lines:
        f = ln.split()
        if f and f[0] == "ndense":
            out.append(f"ndense {int(f[1]) + len(added)}")
            continue
        if f and f[0] == "cfg" and f[1].startswith("flash_"):
            continue
        out.append(ln)
    # cfg first, dense rows after the existing ones: the engine's parser is
    # order-independent within a section but reads cfg keys as it goes, and
    # slots_per_layer must already be set when the dense rows are bound.
    ins = max(i for i, ln in enumerate(out) if ln.startswith("cfg ")) + 1
    out[ins:ins] = [f"cfg flash_{k} {v}" for k, v in fcfg.items()]
    ins = max(i for i, ln in enumerate(out) if ln.startswith("dense ")) + 1
    out[ins:ins] = [f"dense {n} {o} {l} {fmt} {O} {I}" for n, o, l, fmt, O, I in added]
    open(mpath, "w").write("\n".join(out) + "\n")

    p = os.path.join(a.dir, "cfg.json")
    if os.path.exists(p):
        c = json.load(open(p))
        c.update({f"flash_{k}": v for k, v in fcfg.items()})
        json.dump(c, open(p, "w"), indent=1)

    grew = sum(len(b) for _, b, _, _ in blobs)
    print(f"appended {grew / 2**20:.1f} MiB to dense.bin; run with --flash "
          f"(and --flash-check to see how often it agrees with the exact head)")


if __name__ == "__main__":
    main()
