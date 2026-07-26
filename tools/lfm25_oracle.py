#!/usr/bin/env python3
"""
lfm25_oracle.py — an independent numpy forward pass for LFM2.5, written from the
architecture, run on the DEQUANTISED container weights.

WHY THIS AND NOT HF. Comparing lfm25.c against fp32 HF logits would conflate two
completely different things: a bug in the engine, and the quantiser's own error
(which on a random fixture is percent-level and would drown any bug worth finding).
So the oracle reads the SAME container the engine reads, dequantises it, and runs
the math in float64. Any disagreement beyond float noise is then unambiguously an
engine bug -- there is nothing else left for it to be.

It is also written to be structurally different from the C: whole-sequence
matrices, an explicit attention matrix, no streaming, no online softmax, no conv
state machine (the conv is a plain shifted sum here). Two implementations that
share a bug tend to share it because they share code; these share none.

    python3 tools/convert_lfm25.py --fixture --ctx 64 /tmp/lfmfix
    python3 tools/lfm25_oracle.py /tmp/lfmfix
    ./lfm25-exact /tmp/lfmfix --check
"""
import argparse, json, os, sys
import numpy as np

QK = 32
FMT_F32, FMT_Q40, FMT_Q80 = 0, 1, 2


def deq(fmt, buf, O, I):
    if fmt == FMT_F32:
        return np.frombuffer(buf, np.float32).reshape(O, I).astype(np.float64)
    nb = I // QK
    if fmt == FMT_Q40:
        a = np.frombuffer(buf, np.uint8).reshape(O, nb, 18)
        d = a[:, :, 0:2].copy().view(np.float16).astype(np.float64).reshape(O, nb)
        nib = a[:, :, 2:]
        lo = (nib & 0x0F).astype(np.int32) - 8
        hi = (nib >> 4).astype(np.int32) - 8
        w = np.concatenate([lo, hi], 2).astype(np.float64)
    else:
        a = np.frombuffer(buf, np.uint8).reshape(O, nb, 34)
        d = a[:, :, 0:2].copy().view(np.float16).astype(np.float64).reshape(O, nb)
        w = a[:, :, 2:].view(np.int8).astype(np.float64)
    return (w * d[:, :, None]).reshape(O, I)


class Container:
    def __init__(self, d):
        self.d = d
        self.cfg = {}
        self.dense = {}
        self.eoff = {}
        self.esz, self.gb, self.db, self.efmt = {}, {}, {}, {}
        for line in open(os.path.join(d, "manifest.txt")):
            t = line.split()
            if not t:
                continue
            if t[0] == "cfg":
                if t[1] == "layer_types":
                    self.cfg["layer_types"] = [int(x) for x in t[2:]]
                else:
                    v = t[2]
                    try:
                        self.cfg[t[1]] = int(v)
                    except ValueError:
                        self.cfg[t[1]] = float(v)
            elif t[0] == "eszl":
                li = int(t[1])
                self.esz[li], self.gb[li], self.db[li] = int(t[2]), int(t[3]), int(t[4])
                self.efmt[li] = int(t[5])
            elif t[0] == "dense":
                self.dense[t[1]] = dict(off=int(t[2]), len=int(t[3]), fmt=int(t[4]),
                                        O=int(t[5]), I=int(t[6]))
            elif t[0] == "expert":
                self.eoff[(int(t[1]), int(t[2]))] = int(t[3])
        self.blob = open(os.path.join(d, "dense.bin"), "rb").read()
        self.efile = open(os.path.join(d, "experts.bin"), "rb")

    def w(self, name):
        e = self.dense[name]
        return deq(e["fmt"], self.blob[e["off"]:e["off"] + e["len"]], e["O"], e["I"])

    def expert(self, li, eid):
        """[gate | up | down] for one expert, as three dequantised matrices."""
        D, MI = self.cfg["hidden"], self.cfg["moe_inter"]
        f, gb = self.efmt[li], self.gb[li]
        self.efile.seek(self.eoff[(li, eid)])
        raw = self.efile.read(self.esz[li])
        return (deq(f, raw[:gb], MI, D),
                deq(f, raw[gb:2 * gb], MI, D),
                deq(f, raw[2 * gb:], D, MI))


def rmsnorm(x, w, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def forward(C, ids):
    c = C.cfg
    D, H, hd = c["hidden"], c["n_heads"], c["head_dim"]
    nkv, NE, K = c["n_kv_heads"], c["n_experts"], c["topk"]
    MI, CL, ND = c["moe_inter"], c["conv_L"], c["n_dense_layers"]
    eps, theta = c["eps"], c["rope_theta"]
    S = len(ids)

    emb = C.w("embed_tokens")
    x = emb[list(ids)].copy()                       # [S, D] -- no embed scaling

    # RoPE tables, rotate_half convention
    inv = theta ** (-np.arange(0, hd // 2) * 2.0 / hd)
    ang = np.arange(S)[:, None] * inv[None, :]      # [S, hd/2]
    cos, sin = np.cos(ang), np.sin(ang)

    def rope(v):                                    # v: [S, nh, hd]
        a, b = v[..., :hd // 2], v[..., hd // 2:]
        return np.concatenate([a * cos[:, None, :] - b * sin[:, None, :],
                               b * cos[:, None, :] + a * sin[:, None, :]], -1)

    for li in range(c["n_layers"]):
        p = f"layers.{li}."
        h = rmsnorm(x, C.w(p + "operator_norm")[0], eps)

        if c["layer_types"][li]:
            # ---- GQA attention, full causal, scaled by 1/sqrt(head_dim) ----
            q = (h @ C.w(p + "q_proj").T).reshape(S, H, hd)
            k = (h @ C.w(p + "k_proj").T).reshape(S, nkv, hd)
            v = (h @ C.w(p + "v_proj").T).reshape(S, nkv, hd)
            q = rmsnorm(q, C.w(p + "q_norm")[0], eps)
            k = rmsnorm(k, C.w(p + "k_norm")[0], eps)
            q, k = rope(q), rope(k)
            k = np.repeat(k, H // nkv, axis=1)      # GQA: share kv heads
            v = np.repeat(v, H // nkv, axis=1)
            att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd)
            att = np.where(np.tril(np.ones((S, S), bool))[None], att, -np.inf)
            att -= att.max(-1, keepdims=True)
            att = np.exp(att)
            att /= att.sum(-1, keepdims=True)
            o = np.einsum("hqk,khd->qhd", att, v).reshape(S, H * hd)
            y = o @ C.w(p + "o_proj").T
        else:
            # ---- short causal depthwise conv: y = C * conv(B * x) ----
            bcx = h @ C.w(p + "conv_in").T          # [S, 3D]
            Bg, Cg, xv = bcx[:, :D], bcx[:, D:2 * D], bcx[:, 2 * D:]
            g = Bg * xv                             # [S, D]
            cw = C.w(p + "conv_w")                  # [D, CL]
            pad = np.concatenate([np.zeros((CL - 1, D)), g], 0)
            conv = sum(pad[t:t + S] * cw[:, t][None, :] for t in range(CL))
            y = (Cg * conv) @ C.w(p + "conv_out").T

        x = x + y

        # ---- feed forward ----
        h = rmsnorm(x, C.w(p + "ffn_norm")[0], eps)
        if li < ND:
            y = (silu(h @ C.w(p + "mlp_gate").T) * (h @ C.w(p + "mlp_up").T)) \
                @ C.w(p + "mlp_down").T
        else:
            logits = h @ C.w(p + "router").T        # [S, NE]
            probs = 1.0 / (1.0 + np.exp(-logits))   # SIGMOID, not softmax
            bias = C.w(p + "expert_bias")[0]
            sel = probs + bias if c["use_expert_bias"] else probs
            idx = np.argsort(-sel, axis=-1, kind="stable")[:, :K]
            wt = np.take_along_axis(probs, idx, -1)  # the UNBIASED sigmoid
            if c["norm_topk_prob"]:
                wt = wt / (wt.sum(-1, keepdims=True) + 1e-6)
            wt = wt * c["routed_scaling"]
            y = np.zeros_like(x)
            cache = {}
            for s in range(S):
                for j in range(K):
                    e = int(idx[s, j])
                    if e not in cache:
                        cache[e] = C.expert(li, e)
                    g_, u_, d_ = cache[e]
                    hs = h[s]
                    y[s] += wt[s, j] * ((silu(g_ @ hs) * (u_ @ hs)) @ d_.T)
        x = x + y

    x = rmsnorm(x, C.w("embedding_norm")[0], eps)
    return x @ emb.T                                # tied lm_head, no softcap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    C = Container(a.dir)
    ids = json.load(open(os.path.join(a.dir, "ref.json")))["prompt"]
    out = forward(C, ids).astype(np.float32)
    out.tofile(os.path.join(a.dir, "deq_logits.f32"))
    print(f"oracle: {out.shape[0]} rows x {out.shape[1]} vocab -> deq_logits.f32")
    print(f"  logit range [{out.min():.4f}, {out.max():.4f}]  argmax {out.argmax(-1)}")


if __name__ == "__main__":
    main()
