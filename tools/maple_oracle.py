#!/usr/bin/env python3
"""
maple_oracle.py — an independent numpy forward pass for Maple, written from the
architecture, run on the DEQUANTISED container weights.

WHY NOT THE MLX REFERENCE. Comparing against mlx-lm's logits would conflate a bug in
the engine with the difference between two quantised arithmetics (and on a random
fixture there is no mlx-lm to run at all). So the oracle reads the SAME container,
dequantises it, and runs the math in float64: any disagreement beyond float noise is
then unambiguously an engine bug. tools/maple_mlx_check.py does the other comparison,
against the real checkpoint.

It is deliberately structured unlike the C -- whole-sequence matrices, an explicit
attention matrix with an explicit sliding-window mask, no streaming, no online
softmax, no ring buffer, no expert cache -- so the two share no code to share a bug
in.

    python3 tools/convert_maple.py --fixture --ctx 64 /tmp/mapfix
    python3 tools/maple_oracle.py /tmp/mapfix
    ./maple-exact /tmp/mapfix --check
"""
import argparse, json, os
import numpy as np

QK = 32
TQ2_GRP = 64
Q4A_GRP, Q4A_GRP_BYTES = 64, 36
FMT_F32, FMT_TQ2, FMT_Q4A, FMT_I32 = 0, 3, 4, 5


def deq(fmt, buf, O, I):
    """Dequantise one container tensor to float64, from the format description in
    tq2.h rather than from the converter's packing code."""
    if fmt == FMT_F32:
        return np.frombuffer(buf, np.float32).reshape(O, I).astype(np.float64)
    if fmt == FMT_I32:
        return np.frombuffer(buf, np.int32).reshape(O, I)
    if fmt == FMT_TQ2:
        # row scales, then code rows; byte (g,k) holds elements g*64 + t*16 + k at
        # bit position 2t
        alpha = np.frombuffer(buf[:O * 4], np.float32).astype(np.float64)
        p = np.frombuffer(buf[O * 4:], np.uint8).reshape(O, I // TQ2_GRP, 16)
        c = np.stack([(p >> 0) & 3, (p >> 2) & 3, (p >> 4) & 3, (p >> 6) & 3], axis=2)
        return (c.reshape(O, I).astype(np.float64) - 1.0) * alpha[:, None]
    # FMT_Q4A: groups of 64 = fp16 d, fp16 m, two 32-weight nibble blocks
    ng = I // Q4A_GRP
    a = np.frombuffer(buf, np.uint8).reshape(O, ng, Q4A_GRP_BYTES)
    d = a[:, :, 0:2].copy().view(np.float16).astype(np.float64).reshape(O, ng)
    m = a[:, :, 2:4].copy().view(np.float16).astype(np.float64).reshape(O, ng)
    nib = a[:, :, 4:].reshape(O, ng, 2, 16)
    lo = (nib & 0x0F).astype(np.int32)
    hi = (nib >> 4).astype(np.int32)
    # within a 32-block: low nibble of byte j is w[j], high nibble is w[j+16]
    q = np.concatenate([lo, hi], axis=3).reshape(O, ng, Q4A_GRP).astype(np.float64)
    return (q * d[:, :, None] + m[:, :, None]).reshape(O, I)


class Container:
    def __init__(self, d):
        self.d = d
        self.cfg = {}
        self.dense = {}
        self.eoff = {}
        for line in open(os.path.join(d, "manifest.txt")):
            t = line.split()
            if not t:
                continue
            if t[0] == "cfg":
                if t[1] == "layer_swa":
                    self.cfg["layer_swa"] = [int(x) for x in t[2:]]
                else:
                    v = t[2]
                    try:
                        self.cfg[t[1]] = int(v)
                    except ValueError:
                        self.cfg[t[1]] = float(v)
            elif t[0] == "esz":
                self.esz, self.gb, self.db = int(t[1]), int(t[2]), int(t[3])
            elif t[0] == "dense":
                self.dense[t[1]] = dict(off=int(t[2]), len=int(t[3]), fmt=int(t[4]),
                                        O=int(t[5]), I=int(t[6]))
            elif t[0] == "expert":
                self.eoff[(int(t[1]), int(t[2]))] = int(t[3])
        self.blob = open(os.path.join(d, "dense.bin"), "rb").read()
        self.efile = open(os.path.join(d, "experts.bin"), "rb")

    def has(self, name):
        return name in self.dense

    def w(self, name):
        e = self.dense[name]
        return deq(e["fmt"], self.blob[e["off"]:e["off"] + e["len"]], e["O"], e["I"])

    def expert(self, li, eid):
        """[gate | up | down] for one expert, as three dequantised matrices."""
        D, MI = self.cfg["hidden"], self.cfg["moe_inter"]
        self.efile.seek(self.eoff[(li, eid)])
        raw = self.efile.read(self.esz)
        return (deq(FMT_TQ2, raw[:self.gb], MI, D),
                deq(FMT_TQ2, raw[self.gb:2 * self.gb], MI, D),
                deq(FMT_TQ2, raw[2 * self.gb:], D, MI))


def rmsnorm(x, w, eps):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def forward(C, ids):
    c = C.cfg
    D, H, hd = c["hidden"], c["n_heads"], c["head_dim"]
    nkv, NE, K = c["n_kv_heads"], c["n_experts"], c["topk"]
    MI = c["moe_inter"]
    eps, theta = c["eps"], c["rope_theta"]
    rd, win, clamp = c["rope_dim"], c["sliding_window"], c["mlp_clamp"]
    S = len(ids)

    x = C.w("embed")[list(ids)].copy()               # [S, D] -- no embed scaling

    # Partial rotary: only the first `rd` of each head rotates, non-traditional
    # pairing (i, i + rd/2). inv_freq has rd/2 entries, not hd/2.
    half = rd // 2
    inv = theta ** (-np.arange(half) / half)
    ang = np.arange(S)[:, None] * inv[None, :]       # [S, rd/2]
    cos, sin = np.cos(ang), np.sin(ang)

    def rope(v):                                     # v: [S, nh, hd]
        out = v.copy()
        a, b = v[..., :half], v[..., half:rd]
        out[..., :half] = a * cos[:, None, :] - b * sin[:, None, :]
        out[..., half:rd] = b * cos[:, None, :] + a * sin[:, None, :]
        return out                                   # dims rd.. pass through

    causal = np.tril(np.ones((S, S), bool))
    # sliding window: query q attends to keys in [q-win+1, q]
    qi, ki = np.arange(S)[:, None], np.arange(S)[None, :]
    swa_mask = causal & (qi - ki < win)

    for li in range(c["n_layers"]):
        p = f"layers.{li}."
        h = rmsnorm(x, C.w(p + "input_layernorm")[0], eps)

        q = (h @ C.w(p + "q_proj").T).reshape(S, H, hd)
        k = (h @ C.w(p + "k_proj").T).reshape(S, nkv, hd)
        v = (h @ C.w(p + "v_proj").T).reshape(S, nkv, hd)
        q = rmsnorm(q, C.w(p + "q_norm")[0], eps)
        k = rmsnorm(k, C.w(p + "k_norm")[0], eps)
        # RoPE on the sliding layers only; the full-attention layers are NoPE
        if c["layer_swa"][li]:
            q, k = rope(q), rope(k)
        k = np.repeat(k, H // nkv, axis=1)           # GQA: share kv heads
        v = np.repeat(v, H // nkv, axis=1)
        att = np.einsum("qhd,khd->hqk", q, k) / np.sqrt(hd)
        mask = swa_mask if c["layer_swa"][li] else causal
        att = np.where(mask[None], att, -np.inf)
        att -= att.max(-1, keepdims=True)
        att = np.exp(att)
        att /= att.sum(-1, keepdims=True)
        o = np.einsum("hqk,khd->qhd", att, v).reshape(S, H * hd)
        x = x + o @ C.w(p + "o_proj").T

        # MoE (every layer): plain softmax router, top-k, renormalise
        h = rmsnorm(x, C.w(p + "post_attention_layernorm")[0], eps)
        logits = h @ C.w(p + "router").T             # [S, NE]
        mx = logits.max(-1, keepdims=True)
        probs = np.exp(logits - mx)
        probs /= probs.sum(-1, keepdims=True)
        idx = np.argsort(-probs, axis=-1, kind="stable")[:, :K]
        wt = np.take_along_axis(probs, idx, -1)
        if c["norm_topk_prob"]:
            wt = wt / (wt.sum(-1, keepdims=True) + 1e-20)
        y = np.zeros_like(x)
        cache = {}
        for s in range(S):
            for j in range(K):
                e = int(idx[s, j])
                if e not in cache:
                    cache[e] = C.expert(li, e)
                g_, u_, d_ = cache[e]
                hs = h[s]
                # clamped SwiGLU -- part of the trained forward pass
                gate = np.minimum(g_ @ hs, clamp)
                up = np.clip(u_ @ hs, -clamp, clamp)
                y[s] += wt[s, j] * ((silu(gate) * up) @ d_.T)
        x = x + y

    x = rmsnorm(x, C.w("final_norm")[0], eps)
    return x @ C.w("lm_head").T                      # untied head, no softcap


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
