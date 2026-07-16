#!/usr/bin/env python3
"""
gemma4_oracle.py — pure-numpy Gemma-4 text forward pass.

PURPOSE. This is the spec that gemma4.c is written against, and the teacher-forcing
oracle it is validated with (same role tools/make_glm_oracle.py plays for GLM).
It is deliberately written from the architecture, NOT by calling transformers, so
that `--selftest` is a real independent check: it builds a tiny random Gemma-4,
runs the HF reference, runs this, and diffs. If my reading of the decoder layer is
wrong, that diff explodes.

The subtleties this encodes, each of which is a plausible way to get it silently
wrong:
  * RMSNorm is plain `w`, NOT Gemma-3's `(1 + w)`.
  * attention scaling is 1.0 — no 1/sqrt(d); q_norm/k_norm carry it.
  * RoPE is rotate_half (split-half), NOT interleaved.
  * p-RoPE on global layers = ordinary rotate_half with an inv_freq whose tail is
    ZERO (rope_angles = int(partial * head_dim // 2) nonzero, rest zeros). A zero
    frequency is cos=1,sin=0, i.e. identity — so NO special code path is needed.
  * global layers (attention_k_eq_v) have no v_proj: V = v_norm(k_proj(x)) with
    v_norm UNSCALED and UNROPED, while K = rope(k_norm(k_proj(x))). Same
    projection, different post-processing.
  * the dense MLP and the MoE are PARALLEL branches fed DIFFERENT inputs: the MLP
    gets pre_ffn_ln(residual), the MoE gets pre_ffn_ln_2(residual), and the router
    reads the RAW residual. (This is why expert ids are known before the MLP runs.)
  * router: rmsnorm(no scale) -> *scale * hidden**-0.5 -> proj -> softmax over all
    experts -> top-k -> renormalise to sum 1 -> multiply by per_expert_scale[idx].
  * embeddings are scaled by embed_scale; final logits get tanh softcapping.
"""
import argparse
import numpy as np


# ------------------------------------------------------------------ primitives
def rmsnorm(x, w, eps):
    """Gemma4RMSNorm. Computed in f32. w=None => with_scale=False."""
    x = x.astype(np.float32)
    ms = (x ** 2).mean(-1, keepdims=True) + eps
    out = x * (ms ** -0.5)
    return out * w.astype(np.float32) if w is not None else out


def gelu_tanh(x):
    """gelu_pytorch_tanh"""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def rotate_half(x):
    d = x.shape[-1] // 2
    return np.concatenate([-x[..., d:], x[..., :d]], axis=-1)


def rope_tables(head_dim, theta, positions, partial=1.0):
    """inv_freq per _compute_proportional_rope_parameters.

    rope_angles frequencies are live; the remaining (head_dim//2 - rope_angles)
    are ZERO -> cos=1, sin=0 -> those dims pass through unrotated. partial=1.0
    reduces to plain RoPE, so one function covers both layer types.
    """
    n = head_dim // 2
    k = int(partial * head_dim // 2)
    inv = np.zeros(n, dtype=np.float32)
    inv[:k] = 1.0 / (theta ** (np.arange(0, 2 * k, 2, dtype=np.float32) / head_dim))
    fr = positions[:, None].astype(np.float32) * inv[None, :]     # [S, n]
    emb = np.concatenate([fr, fr], axis=-1)                       # [S, head_dim]
    return np.cos(emb), np.sin(emb)


def apply_rope(x, cos, sin):
    """x: [S, H, D]; cos/sin: [S, D]"""
    c, s = cos[:, None, :], sin[:, None, :]
    return x * c + rotate_half(x) * s


def softmax(x, axis=-1):
    x = x.astype(np.float32)
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


# ------------------------------------------------------------------ the model
class Gemma4Oracle:
    def __init__(self, W, cfg):
        self.W, self.c = W, cfg

    def layer(self, h, li, positions):
        c, W = self.c, self.W
        p = f"layers.{li}."
        is_global = c["layer_types"][li] == 1

        head_dim = c["global_head_dim"] if is_global else c["head_dim"]
        n_kv = c["n_global_kv_heads"] if is_global else c["n_kv_heads"]
        n_h = c["n_heads"]
        S = h.shape[0]
        k_eq_v = is_global and c["k_eq_v_global"]

        # ---- attention ----
        x = rmsnorm(h, W[p + "input_layernorm.weight"], c["eps"])

        q = (x @ W[p + "self_attn.q_proj.weight"].T).reshape(S, n_h, head_dim)
        q = rmsnorm(q, W[p + "self_attn.q_norm.weight"], c["eps"])

        kraw = (x @ W[p + "self_attn.k_proj.weight"].T).reshape(S, n_kv, head_dim)
        k = rmsnorm(kraw, W[p + "self_attn.k_norm.weight"], c["eps"])
        if k_eq_v:
            # V aliases the RAW k projection (pre-k_norm), then v_norm (no scale).
            v = rmsnorm(kraw, None, c["eps"])
        else:
            vraw = (x @ W[p + "self_attn.v_proj.weight"].T).reshape(S, n_kv, head_dim)
            v = rmsnorm(vraw, None, c["eps"])

        theta = c["rope_theta_global"] if is_global else c["rope_theta_local"]
        partial = c["rope_partial_global"] if is_global else 1.0
        cos, sin = rope_tables(head_dim, theta, positions, partial)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)          # V is NOT roped

        rep = n_h // n_kv
        kk = np.repeat(k, rep, axis=1)       # [S, n_h, D]
        vv = np.repeat(v, rep, axis=1)

        att = np.einsum("qhd,khd->hqk", q, kk).astype(np.float32)   # scaling = 1.0
        i = np.arange(S)
        mask = i[:, None] < i[None, :]                               # causal
        if not is_global:                                            # sliding window
            mask |= (i[:, None] - i[None, :]) >= c["sliding_window"]
        att = np.where(mask[None], -np.inf, att)
        att = softmax(att, -1)
        o = np.einsum("hqk,khd->qhd", att, vv).reshape(S, n_h * head_dim)
        o = o @ W[p + "self_attn.o_proj.weight"].T

        o = rmsnorm(o, W[p + "post_attention_layernorm.weight"], c["eps"])
        h = h + o

        # ---- feed-forward: dense MLP and MoE are PARALLEL on DIFFERENT inputs ----
        residual = h

        y = rmsnorm(residual, W[p + "pre_feedforward_layernorm.weight"], c["eps"])
        g = y @ W[p + "mlp.gate_proj.weight"].T
        u = y @ W[p + "mlp.up_proj.weight"].T
        mlp_out = (gelu_tanh(g) * u) @ W[p + "mlp.down_proj.weight"].T
        h1 = rmsnorm(mlp_out, W[p + "post_feedforward_layernorm_1.weight"], c["eps"])

        # router reads the RAW residual -> expert ids are known BEFORE the MLP above
        idx, wts = self.route(residual, li)
        y2 = rmsnorm(residual, W[p + "pre_feedforward_layernorm_2.weight"], c["eps"])
        moe_out = self.experts(y2, idx, wts, li)
        h2 = rmsnorm(moe_out, W[p + "post_feedforward_layernorm_2.weight"], c["eps"])

        h = residual + rmsnorm(h1 + h2, W[p + "post_feedforward_layernorm.weight"], c["eps"])
        return h * W[p + "layer_scalar"]

    def route(self, x, li):
        c, W = self.c, self.W
        p = f"layers.{li}.router."
        y = rmsnorm(x, None, c["eps"])                       # no scale
        y = y * W[p + "scale"] * (c["hidden"] ** -0.5)
        scores = y @ W[p + "proj.weight"].T                  # [S, E]
        prob = softmax(scores, -1)
        k = c["topk"]
        idx = np.argpartition(-prob, k - 1, axis=-1)[:, :k]
        take = np.take_along_axis(prob, idx, -1)
        order = np.argsort(-take, axis=-1)                   # topk returns sorted
        idx = np.take_along_axis(idx, order, -1)
        wts = np.take_along_axis(take, order, -1)
        wts = wts / wts.sum(-1, keepdims=True)               # renormalise to 1
        wts = wts * W[p + "per_expert_scale"][idx]           # then per-expert scale
        return idx, wts

    def experts(self, x, idx, wts, li):
        c, W = self.c, self.W
        gu = W[f"layers.{li}.experts.gate_up_proj"]          # [E, 2*MI, D]
        dn = W[f"layers.{li}.experts.down_proj"]             # [E, D, MI]
        MI = c["moe_inter"]
        out = np.zeros_like(x, dtype=np.float32)
        for t in range(x.shape[0]):
            for j in range(c["topk"]):
                e = idx[t, j]
                z = gu[e] @ x[t]                             # [2*MI]
                g, u = z[:MI], z[MI:]                        # chunk on the OUTPUT
                out[t] += (dn[e] @ (gelu_tanh(g) * u)) * wts[t, j]
        return out

    def forward(self, ids):
        c, W = self.c, self.W
        S = len(ids)
        pos = np.arange(S)
        h = W["embed_tokens.weight"][ids] * c["embed_scale"]
        for li in range(c["n_layers"]):
            h = self.layer(h, li, pos)
        h = rmsnorm(h, W["norm.weight"], c["eps"])
        logits = h @ W["embed_tokens.weight"].T              # tied
        cap = c["final_logit_softcap"]
        if cap:
            logits = np.tanh(logits / cap) * cap
        return logits


# ------------------------------------------------------------------- self-test
def selftest():
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    torch.manual_seed(0)
    L = 6
    hf = Gemma4TextConfig(
        vocab_size=100, hidden_size=64, intermediate_size=48, num_hidden_layers=L,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        global_head_dim=32, num_global_key_value_heads=1, attention_k_eq_v=True,
        num_experts=8, top_k_experts=2, moe_intermediate_size=32,
        enable_moe_block=True, sliding_window=4, hidden_size_per_layer_input=0,
        num_kv_shared_layers=0, final_logit_softcapping=30.0,
        layer_types=["sliding_attention"] * 5 + ["full_attention"],
        rope_parameters={
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                               "partial_rotary_factor": 0.25},
        },
        rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
    )
    ref = Gemma4TextModel(hf).eval().float()
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.08)
    for n, b in ref.named_buffers():
        if n.endswith("layer_scalar"):
            b.copy_(torch.tensor([0.9]))

    sd = {k: v.detach().numpy().astype(np.float32) for k, v in ref.state_dict().items()}
    W = {k: v for k, v in sd.items()}
    cfg = dict(
        hidden=64, n_layers=L, n_heads=4, head_dim=16, global_head_dim=32,
        n_kv_heads=2, n_global_kv_heads=1, k_eq_v_global=True,
        n_experts=8, topk=2, moe_inter=32, eps=1e-6, sliding_window=4,
        layer_types=[0] * 5 + [1],
        rope_theta_local=10000.0, rope_theta_global=1000000.0,
        rope_partial_global=0.25, final_logit_softcap=None,
        embed_scale=float(ref.embed_tokens.embed_scale),
    )

    ids = np.array([7, 3, 42, 5, 19, 88, 2, 61, 30, 11])
    with torch.no_grad():
        got = ref(input_ids=torch.tensor(ids)[None]).last_hidden_state[0].numpy()

    ora = Gemma4Oracle(W, cfg)
    h = W["embed_tokens.weight"][ids] * cfg["embed_scale"]
    for li in range(L):
        h = ora.layer(h, li, np.arange(len(ids)))
    mine = rmsnorm(h, W["norm.weight"], cfg["eps"])

    num = np.abs(got - mine).max()
    den = np.abs(got).max()
    print(f"reference |h| max = {den:.4f}")
    print(f"max abs diff       = {num:.3e}   (rel {num/den:.3e})")
    ok = num / den < 1e-5
    print("MATCH" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    raise SystemExit(selftest() if a.selftest else 0)
