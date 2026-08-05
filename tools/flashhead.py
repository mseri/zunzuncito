#!/usr/bin/env python3
"""
flashhead.py — build a FlashHead (clustered output head) from an lm_head matrix.

Maple ships its FlashHead in the checkpoint: DeepGrove clustered the head offline and
`convert_maple.py` only repacks the result. Gemma-4 and LFM2.5 ship nothing of the
sort, and both tie the head to the embedding table, so the clustering has to be built
here. This module is the shared half of that: the two converters call `build()` and
write whatever it returns.

The construction follows Maple's, which is the only published description of what the
runtime expects:

    balanced spherical k-means, `cluster_size` rows per cluster, a cluster ranked by
    centroid similarity times the largest member-row norm

Balanced matters twice over. `token_map` is a rectangular [n_clusters, cluster_size]
int32 table, so equal sizes are what makes the gather a fixed-size block read rather
than a ragged one, and equal sizes are also what keeps the probe budget meaningful:
with free-form k-means a handful of clusters absorb most of the vocabulary and 512
probes stop bounding the candidate count.

The scale is not a nicety either. Spherical k-means throws away row norms, but a
logit is a dot product, not a cosine: a cluster holding one large-norm embedding can
win the argmax while its centroid looks unremarkable. Multiplying the similarity by
the cluster's largest member norm is the heuristic Maple uses, and it is a heuristic
rather than an upper bound -- the true bound would need the norm of the *component*
along h, which is what we are trying to avoid computing.

Two phases, because exact balanced k-means is an assignment problem and this runs on
a laptop:

    1. `iters` unconstrained Lloyd iterations to find the geometry
    2. one capacity-constrained pass that fixes the sizes at `cluster_size`

Phase 2 is greedy, in descending order of the margin between a row's best and second
best cluster: a row that is nearly indifferent between two clusters can be displaced
cheaply, a row with a clear winner cannot, so the indifferent ones go last.
"""
import json
import os
import sys

import numpy as np

# Maple's own numbers: 4748 clusters of 32, 512 probed. The cluster size carries over
# unchanged (it is a memory-layout choice, not a vocabulary-dependent one), and the
# probe count carries over as the FRACTION it represents, so a different vocabulary
# scores the same ~11% of its rows.
CLUSTER_SIZE = 32
PROBE_FRACTION = 512 / 4748


def default_probes(n_clusters):
    return max(1, int(round(n_clusters * PROBE_FRACTION)))


def build(rows, cluster_size=CLUSTER_SIZE, n_probes=0, iters=10, seed=0,
          chunk=4096, log=print):
    """Cluster an lm_head [V, D] into a FlashHead.

    `rows` is consumed: it is normalised in place, because at Gemma-4's 262144 x 2816
    a float32 copy is 2.8 GiB and this converter is meant to run on the machines the
    engine targets. The caller must be done with it.

    Returns a dict of tensors and the config the engine needs:
        token_map     int32 [n_clusters, cluster_size], -1 in the padding slots
        centroids     f32   [n_clusters, D], with cluster_scale already folded in
        cluster_scale f32   [n_clusters], kept for inspection (see `scaled_centroids`)
    """
    V, D = rows.shape
    ncl = (V + cluster_size - 1) // cluster_size
    if n_probes <= 0:
        n_probes = default_probes(ncl)
    n_probes = min(n_probes, ncl)
    rng = np.random.default_rng(seed)

    norms = np.linalg.norm(rows, axis=1).astype(np.float32)
    unit = rows                                   # normalised in place, see docstring
    unit /= np.maximum(norms, 1e-12)[:, None]

    # k-means++ over 262144 points would cost more than the Lloyd iterations that
    # follow it; a random sample of distinct rows is the usual substitute and the
    # empty-cluster reseeding below repairs the bad draws.
    C = np.ascontiguousarray(unit[rng.choice(V, ncl, replace=False)], dtype=np.float32)

    assign = np.zeros(V, np.int32)
    for it in range(iters):
        sums = np.zeros((ncl, D), np.float32)
        best = np.empty(V, np.float32)
        for i in range(0, V, chunk):
            sim = unit[i:i + chunk] @ C.T
            a = np.argmax(sim, axis=1).astype(np.int32)
            assign[i:i + chunk] = a
            best[i:i + chunk] = sim[np.arange(a.size), a]
            np.add.at(sums, a, unit[i:i + chunk])
        counts = np.bincount(assign, minlength=ncl)

        # An empty cluster is a wasted probe AND a hole in the capacity budget that
        # phase 2 would have to fill with whatever is left over, so reseed it on the
        # rows that fit their own cluster worst.
        empty = np.flatnonzero(counts == 0)
        if empty.size:
            worst = np.argsort(best)[:empty.size]
            sums[empty] = unit[worst]
        n = np.linalg.norm(sums, axis=1)
        n[n < 1e-12] = 1.0
        C = (sums / n[:, None]).astype(np.float32)
        log(f"  k-means iter {it + 1}/{iters}: mean sim {float(best.mean()):.4f}, "
            f"{empty.size} empty, max cluster {int(counts.max())}")

    # Phase 2. Candidates first: a row only ever wants one of its nearest clusters,
    # so keep the top few and fall back to a global pass for the rows that lose all
    # of them to capacity.
    NCAND = min(ncl, 32)
    cand = np.empty((V, NCAND), np.int32)
    margin = np.empty(V, np.float32)
    for i in range(0, V, chunk):
        sim = unit[i:i + chunk] @ C.T
        part = np.argpartition(-sim, NCAND - 1, axis=1)[:, :NCAND]
        rowsel = np.arange(part.shape[0])[:, None]
        order = np.argsort(-sim[rowsel, part], axis=1)
        part = part[rowsel, order]
        cand[i:i + chunk] = part
        top = sim[rowsel, part[:, :2]]
        margin[i:i + chunk] = top[:, 0] - (top[:, 1] if NCAND > 1 else 0.0)

    cap = np.full(ncl, cluster_size, np.int32)
    slack = ncl * cluster_size - V
    if slack:
        # The vocabulary does not fill the last cluster. Spread the empty slots
        # rather than leaving one runt cluster: a probe that reads padding is a probe
        # that scored nothing.
        cap[rng.choice(ncl, slack, replace=False)] -= 1
    member = [[] for _ in range(ncl)]
    leftover = []
    for r in np.argsort(-margin):
        for c in cand[r]:
            if cap[c]:
                cap[c] -= 1
                member[c].append(r)
                break
        else:
            leftover.append(int(r))
    log(f"  balanced pass: {len(leftover)} rows past their {NCAND} candidates")

    if leftover:
        free = np.flatnonzero(cap > 0)
        lo = np.asarray(leftover, np.int32)
        sim = unit[lo] @ C[free].T
        for j in np.argsort(-(sim.max(axis=1) - np.median(sim, axis=1))):
            r = int(lo[j])
            k = int(np.argmax(np.where(cap[free] > 0, sim[j], -np.inf)))
            cap[free[k]] -= 1
            member[free[k]].append(r)

    token_map = np.full((ncl, cluster_size), -1, np.int32)
    centroids = np.zeros((ncl, D), np.float32)
    scale = np.zeros(ncl, np.float32)
    for c in range(ncl):
        ids = np.asarray(member[c], np.int32)
        if ids.size == 0:
            sys.exit("flashhead: empty cluster survived the balanced pass")
        token_map[c, :ids.size] = ids
        v = unit[ids].sum(axis=0)
        nv = float(np.linalg.norm(v))
        centroids[c] = v / nv if nv > 1e-12 else unit[ids[0]]
        scale[c] = float(norms[ids].max())

    seen = np.bincount(token_map[token_map >= 0], minlength=V)
    if not (seen == 1).all():
        sys.exit(f"flashhead: {int((seen != 1).sum())} vocabulary ids are not in "
                 f"exactly one cluster")

    # Fold the scale into the centroid rows. The engine has a flag for the unfolded
    # form because Maple's older checkpoints ship that way; building it here, there is
    # no reason to leave a per-cluster multiply in the decode path.
    centroids *= scale[:, None]

    sizes = np.array([len(m) for m in member])
    log(f"  {ncl} clusters x {cluster_size} (sizes {sizes.min()}..{sizes.max()}), "
        f"probe {n_probes} -> {n_probes * cluster_size}/{V} rows scored "
        f"({100.0 * n_probes * cluster_size / V:.1f}%)")
    return {
        "token_map": token_map,
        "centroids": centroids,
        "cluster_scale": scale,
        "cfg": {"n_clusters": int(ncl), "cluster_size": int(cluster_size),
                "n_probes": int(n_probes), "scaled_centroids": 1},
    }


def force_tokens(src, vocab):
    """The token ids that must be scored no matter which clusters are probed.

    An unprobed token is -inf and cannot be sampled, which is a tail cut everywhere
    except on the control tokens: a model that cannot emit EOS does not stop. This
    reads the checkpoint's own declarations rather than guessing ids, and returns []
    for a fixture directory that has none of these files.
    """
    ids = set()

    def take(x):
        if isinstance(x, int):
            ids.add(x)
        elif isinstance(x, (list, tuple)):
            for y in x:
                take(y)

    for fn in ("config.json", "generation_config.json"):
        p = os.path.join(src, fn) if src else None
        if p and os.path.exists(p):
            d = json.load(open(p))
            for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
                take(d.get(k))
                take(d.get("text_config", {}).get(k) if isinstance(
                    d.get("text_config"), dict) else None)

    p = os.path.join(src, "tokenizer_config.json") if src else None
    if p and os.path.exists(p):
        d = json.load(open(p))
        for k, v in (d.get("added_tokens_decoder") or {}).items():
            if isinstance(v, dict) and v.get("special"):
                take(int(k))

    # The chat template's control tokens are usually only declared here -- LFM2.5
    # leaves added_tokens_decoder empty and lists <|im_end|>, <|endoftext|>, the
    # think markers and the tool-call markers in tokenizer.json instead. Forcing a
    # couple of hundred of them costs nothing against the probed set.
    p = os.path.join(src, "tokenizer.json") if src else None
    if p and os.path.exists(p):
        d = json.load(open(p))
        for a in d.get("added_tokens") or []:
            if isinstance(a, dict) and a.get("special"):
                take(a.get("id"))

    return sorted(i for i in ids if 0 <= i < vocab)
