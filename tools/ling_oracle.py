#!/usr/bin/env python3
"""
ling_oracle.py — an independent numpy forward pass for Ling-3.0-tiny, written from the
architecture, run on the DEQUANTISED container weights.

WHY NOT HF. Comparing against fp32 HF logits would conflate a bug in the engine with
the quantiser's own error, which on a random fixture is percent-level. So the oracle
reads the SAME container, dequantises it, and runs the math in float64: any
disagreement beyond float noise is then unambiguously an engine bug.

It is also deliberately structured unlike the C -- whole-sequence matrices, an explicit
attention matrix, no streaming, no online softmax, no conv state machine (the conv is a
plain shifted sum), no partitioned reduction -- so the two share no code to share a bug
in. The KDA recurrence is the one place where a loop over positions is unavoidable, and
even there the state update is written as an outer product rather than the C's two
fused passes.

What it does NOT check, and cannot: whether this reading of the architecture is the
right one. The engine and this file were written from the same reading of
modeling_bailing_moe_v3.py and of fla's KDA kernel, so they agree by construction on
anything they both get wrong. tools/ling_hf_check.py is the thing that closes that gap.

    python3 tools/convert_ling.py --fixture --ctx 64 /tmp/lingfix
    python3 tools/ling_oracle.py /tmp/lingfix
    ./ling-exact /tmp/lingfix --check
"""
import argparse, json, os
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def route(c, h, router, bias):
    """Grouped top-k (topk_method noaux_tc), returning (idx, weight) per row.

    Written with argsort over whole arrays rather than the engine's insertion sort, and
    with the group mask built explicitly, so the two agree only if the RULE agrees. Ties
    resolve to the lower expert index in both, which is what a stable descending sort
    gives."""
    NE, K = c["n_experts"], c["topk"]
    NG, TG = c["n_group"], c["topk_group"]
    EG = NE // NG
    S = h.shape[0]

    probs = sigmoid(h @ router.T)                       # sigmoid, not softmax
    sel = probs + bias

    # group score = sum of the group's two best biased scores
    g = sel.reshape(S, NG, EG)
    top2 = -np.sort(-g, axis=-1)[:, :, :min(2, EG)]
    gscore = top2.sum(-1)                               # [S, NG]
    keep = np.argsort(-gscore, axis=-1, kind="stable")[:, :TG]
    mask = np.zeros((S, NG), bool)
    np.put_along_axis(mask, keep, True, axis=-1)
    live = np.repeat(mask, EG, axis=-1)                 # [S, NE]

    masked = np.where(live, sel, -np.inf)
    idx = np.argsort(-masked, axis=-1, kind="stable")[:, :K]
    wt = np.take_along_axis(probs, idx, -1)             # the UNBIASED sigmoid
    if c["norm_topk_prob"] and K > 1:
        wt = wt / (wt.sum(-1, keepdims=True) + 1e-20)
    return idx, wt * c["routed_scale"]


def forward(C, ids, route_log=None):
    c = C.cfg
    D, H, hd = c["hidden"], c["n_heads"], c["head_dim"]
    P = H * hd
    KL, RD, NOPE, VH = c["kv_lora"], c["qk_rope"], c["qk_nope"], c["v_head"]
    QH = NOPE + RD
    NE, K, CL, ND = c["n_experts"], c["topk"], c["conv_L"], c["n_dense_layers"]
    eps, theta, LB = c["eps"], c["rope_theta"], c["kda_lower_bound"]
    S = len(ids)

    x = C.w("embed")[list(ids)].copy()               # [S, D] -- no embed scaling

    # RoPE over the rotary half only, interleaved: the checkpoint stores the pair
    # members adjacent, so de-interleave first and then rotate halves.
    inv = theta ** (-np.arange(0, RD // 2) * 2.0 / RD)
    ang = np.arange(S)[:, None] * inv[None, :]       # [S, RD/2]
    cos, sin = np.cos(ang), np.sin(ang)

    def rope(v):                                     # v: [S, nh, RD]
        a, b = v[..., 0::2], v[..., 1::2]            # de-interleave
        return np.concatenate([a * cos[:, None, :] - b * sin[:, None, :],
                               b * cos[:, None, :] + a * sin[:, None, :]], -1)

    for li in range(c["n_layers"]):
        p = f"layers.{li}."
        h = rmsnorm(x, C.w(p + "input_layernorm")[0], eps)

        if c["layer_types"][li]:
            # ---- Multi-head Latent Attention, in the SAME absorbed form the engine
            # uses. Not the textbook expansion: the container only ships W_k^T and W_v,
            # so there is no unabsorbed path to write.
            q = rmsnorm(h @ C.w(p + "q_a_proj").T, C.w(p + "q_a_norm")[0], eps)
            q = (q @ C.w(p + "q_b_proj").T).reshape(S, H, QH)
            kva = h @ C.w(p + "kv_a_proj").T                      # [S, KL+RD]
            cl = rmsnorm(kva[:, :KL], C.w(p + "kv_a_norm")[0], eps)   # [S, KL]
            kr = rope(kva[:, KL:].reshape(S, 1, RD))[:, 0, :]     # [S, RD]
            qr = rope(q[:, :, NOPE:])                             # [S, H, RD]

            wkt = C.w(p + "kv_b_kt").reshape(H, KL, NOPE)
            wv = C.w(p + "kv_b_v").reshape(H, VH, KL)
            qc = np.einsum("hlk,shk->shl", wkt, q[:, :, :NOPE])    # [S, H, KL]

            att = (np.einsum("shl,tl->hst", qc, cl)
                   + np.einsum("shr,tr->hst", qr, kr)) / np.sqrt(QH)
            att = np.where(np.tril(np.ones((S, S), bool))[None], att, -np.inf)
            att -= att.max(-1, keepdims=True)
            att = np.exp(att)
            att /= att.sum(-1, keepdims=True)
            acc = np.einsum("hst,tl->shl", att, cl)               # [S, H, KL]
            o = np.einsum("hvl,shl->shv", wv, acc)                # [S, H, VH]
            o = o * sigmoid(h @ C.w(p + "g_proj").T)[:, :, None]  # head-wise gate
            y = o.reshape(S, H * VH) @ C.w(p + "dense").T
        else:
            # ---- Kimi Delta Attention
            def conv(sig, name):
                """causal depthwise conv of width CL, then silu, as a shifted sum"""
                cw = C.w(p + name)                                # [P, CL]
                pad = np.concatenate([np.zeros((CL - 1, P)), sig], 0)
                return silu(sum(pad[t:t + S] * cw[:, t][None, :] for t in range(CL)))

            q = conv(h @ C.w(p + "q_proj").T, "q_conv").reshape(S, H, hd)
            k = conv(h @ C.w(p + "k_proj").T, "k_conv").reshape(S, H, hd)
            v = conv(h @ C.w(p + "v_proj").T, "v_conv").reshape(S, H, hd)
            f = (h @ C.w(p + "f_proj").T).reshape(S, H, hd)
            # kg_proj is KDA's output gate. It is named apart from the MLA layers'
            # g_proj because they are different shapes doing different jobs: this one
            # is [P, hidden] and gates every channel, that one is [H, hidden] and gates
            # whole heads.
            gt = (h @ C.w(p + "kg_proj").T).reshape(S, H, hd)
            beta = sigmoid(h @ C.w(p + "b_proj").T)               # [S, H]

            A = np.exp(C.w(p + "A_log")[0])                       # [H]
            dtb = C.w(p + "dt_bias")[0].reshape(H, hd)
            # the BOUNDED decay branch: g in (lower_bound, 0), not -exp(A)*softplus(f)
            gk = np.exp(LB * sigmoid(A[None, :, None] * (f + dtb[None])))  # [S,H,hd]

            qn = q / np.sqrt((q ** 2).sum(-1, keepdims=True) + 1e-6)
            kn = k / np.sqrt((k ** 2).sum(-1, keepdims=True) + 1e-6)
            qn = qn / np.sqrt(hd)

            St = np.zeros((H, hd, hd))                            # S[h][v][k]
            o = np.zeros((S, H, hd))
            for s in range(S):
                St = St * gk[s][:, None, :]                       # decay along k
                u = beta[s][:, None] * (v[s] - np.einsum("hvk,hk->hv", St, kn[s]))
                St = St + u[:, :, None] * kn[s][:, None, :]       # rank-1 update
                o[s] = np.einsum("hvk,hk->hv", St, qn[s])
            # FusedRMSNormGated, activation sigmoid: normalise, weight, THEN gate
            o = rmsnorm(o, C.w(p + "o_norm")[0], eps) * sigmoid(gt)
            y = o.reshape(S, P) @ C.w(p + "o_proj").T

        x = x + y

        # ---- feed forward
        h = rmsnorm(x, C.w(p + "post_attention_layernorm")[0], eps)
        if li < ND:
            y = (silu(h @ C.w(p + "mlp_gate").T) * (h @ C.w(p + "mlp_up").T)) \
                @ C.w(p + "mlp_down").T
        else:
            idx, wt = route(c, h, C.w(p + "router"), C.w(p + "expert_bias")[0])
            if route_log is not None:
                route_log[li] = idx.astype(np.int32)
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
            # the always-on shared expert, added on top of the routed sum
            y += (silu(h @ C.w(p + "shared_gate").T) * (h @ C.w(p + "shared_up").T)) \
                 @ C.w(p + "shared_down").T
        x = x + y

    x = rmsnorm(x, C.w("final_norm")[0], eps)
    return x @ C.w("lm_head").T                       # untied head, no softcap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    a = ap.parse_args()
    C = Container(a.dir)
    ids = json.load(open(os.path.join(a.dir, "ref.json")))["prompt"]
    S, L, K = len(ids), C.cfg["n_layers"], C.cfg["topk"]
    # Which experts the oracle routed to, so --check can tell an arithmetic
    # disagreement from a different expert firing. -1 on the dense layers.
    route_log = {}
    out = forward(C, ids, route_log).astype(np.float32)
    out.tofile(os.path.join(a.dir, "deq_logits.f32"))

    routes = np.full((L, S, K), -1, np.int32)
    for li, idx in route_log.items():
        routes[li] = idx
    routes.tofile(os.path.join(a.dir, "deq_route.i32"))

    print(f"oracle: {out.shape[0]} rows x {out.shape[1]} vocab -> deq_logits.f32")
    print(f"  logit range [{out.min():.4f}, {out.max():.4f}]  argmax {out.argmax(-1)}")
    print(f"  routing: {len(route_log)} MoE layers x {S} rows x top-{K} "
          f"-> deq_route.i32")


if __name__ == "__main__":
    main()
