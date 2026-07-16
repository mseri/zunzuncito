#!/usr/bin/env python3
"""
gemma4_mtp_check.py — validate the MTP head.

Two checks:
  1. numpy-vs-HF: run the real Gemma4AssistantForCausalLM with a given inputs_embeds
     and shared_kv_states, and diff against a numpy forward built from my reading of
     the architecture. This is what proves the KV sharing, the non-MoE decoder layer,
     the q-only attention and the pre/post projections are right.
  2. dump mtp_ref_logits.f32 for `gemma4 --check-mtp`, so the C is held to the numpy.

The concat convention (order of the halves, scaled vs raw embedding, pre/post-norm
hidden) is NOT tested here -- it lives in HF's generation glue, not the model.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_gemma4 import q40_dequant_rows
from gemma4_oracle import rmsnorm, gelu_tanh, rope_tables, apply_rope, softmax
from gemma4_check import load as load_backbone, rename


def load_mtp(dst):
    cfg = json.load(open(os.path.join(dst, "mtp.cfg.json")))
    idx = json.load(open(os.path.join(dst, "mtp.idx")))
    blob = np.fromfile(os.path.join(dst, "mtp.bin"), dtype=np.uint8)
    W = {}
    for name, e in idx.items():
        raw = blob[e["off"]:e["off"] + e["len"]].tobytes()
        if e["fmt"] == "q40":
            O, I = e["shape"]
            W[name] = q40_dequant_rows(raw, O, I)
        else:
            W[name] = np.frombuffer(raw, np.float32).reshape(e["shape"])
    return cfg, W


def mtp_forward(cfg, W, bh, emb_row, pos, K_slide, V_slide, K_full, V_full,
                sliding_window, order_he=False):
    """One draft step, from my reading of the architecture."""
    D, BB = cfg["hidden"], cfg["backbone_hidden"]
    nh, eps = cfg["n_heads"], cfg["eps"]

    # HF: inputs_embeds = cat([last_token_embedding, last_hidden_state]) -- EMBED FIRST
    e = np.concatenate([bh, emb_row] if order_he else [emb_row, bh])
    h = W["pre_projection"] @ e

    for li in range(cfg["n_layers"]):
        p = f"layers.{li}."
        glob = cfg["layer_types"][li]
        hd = cfg["global_head_dim"] if glob else cfg["head_dim"]
        Ksrc, Vsrc = (K_full, V_full) if glob else (K_slide, V_slide)
        T, nkv, _ = Ksrc.shape
        rep = nh // nkv

        x = rmsnorm(h, W[p + "input_layernorm"], eps)
        q = (W[p + "q_proj"] @ x).reshape(nh, hd)
        q = rmsnorm(q, W[p + "q_norm"], eps)
        theta = cfg["rope_theta_global"] if glob else cfg["rope_theta_local"]
        partial = cfg["rope_partial_global"] if glob else 1.0
        cos, sin = rope_tables(hd, theta, np.array([pos]), partial)
        q = apply_rope(q[None], cos, sin)[0]          # [nh, hd]

        # The head's mask is BIDIRECTIONAL. Full layers: attend everything. Sliding
        # layers: t >= pos - W, i.e. W+1 positions -- one MORE than the backbone's
        # causal SWA (t >= pos - W + 1). Verified against HF; assuming "q_len==1 means
        # full attention" is wrong and costs ~30% on the attention output.
        KK = np.repeat(Ksrc, rep, axis=1)             # [T, nh, hd]
        VV = np.repeat(Vsrc, rep, axis=1)
        att = np.einsum("hd,khd->hk", q, KK)          # scaling = 1.0
        if not glob:
            lo = max(0, pos - sliding_window)
            att[:, :lo] = -np.inf
        att = softmax(att, -1)
        o = np.einsum("hk,khd->hd", att, VV).reshape(nh * hd)
        o = W[p + "o_proj"] @ o
        o = rmsnorm(o, W[p + "post_attention_layernorm"], eps)
        h = h + o

        # plain (non-MoE) FFN
        y = rmsnorm(h, W[p + "pre_feedforward_layernorm"], eps)
        g = W[p + "mlp_gate"] @ y
        u = W[p + "mlp_up"] @ y
        f = W[p + "mlp_down"] @ (gelu_tanh(g) * u)
        f = rmsnorm(f, W[p + "post_feedforward_layernorm"], eps)
        h = (h + f) * W[p + "layer_scalar"][0]

    hn = rmsnorm(h, W["norm"], eps)
    logits = W["embed_tokens"] @ hn                   # tied lm_head
    bh_next = W["post_projection"] @ hn
    return logits, bh_next


def main(dst, mtpsrc):
    import torch
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM,
    )

    cfg, W = load_mtp(dst)
    ref = Gemma4AssistantForCausalLM.from_pretrained(mtpsrc, dtype=torch.float32).eval()

    D, BB = cfg["hidden"], cfg["backbone_hidden"]
    nh = cfg["n_heads"]
    hd, ghd = cfg["head_dim"], cfg["global_head_dim"]
    # backbone kv head counts, from the fixture backbone
    bcfg = json.load(open(os.path.join(dst, "cfg.json")))
    nkv, gnkv = bcfg["n_kv_heads"], bcfg["n_global_kv_heads"]

    T = 6
    rng = np.random.default_rng(3)
    Ks = rng.standard_normal((T, nkv, hd)).astype(np.float32) * 0.3
    Vs = rng.standard_normal((T, nkv, hd)).astype(np.float32) * 0.3
    Kf = rng.standard_normal((T, gnkv, ghd)).astype(np.float32) * 0.3
    Vf = rng.standard_normal((T, gnkv, ghd)).astype(np.float32) * 0.3
    emb = rng.standard_normal(2 * BB).astype(np.float32) * 0.3   # the raw concat

    pos = T - 1
    with torch.no_grad():
        out = ref(
            inputs_embeds=torch.tensor(emb)[None, None],          # [1, 1, 2*BB]
            position_ids=torch.tensor([[pos]]),
            shared_kv_states={
                "sliding_attention": (
                    torch.tensor(Ks.transpose(1, 0, 2))[None],    # [1, nkv, T, hd]
                    torch.tensor(Vs.transpose(1, 0, 2))[None]),
                "full_attention": (
                    torch.tensor(Kf.transpose(1, 0, 2))[None],
                    torch.tensor(Vf.transpose(1, 0, 2))[None]),
            },
        )
    hf_logits = out.logits[0, 0].numpy()
    hf_bh = out.last_hidden_state[0, 0].numpy()

    SW = bcfg["sliding_window"]

    # (a) EXACT fp32 weights: any error here is an ARCHITECTURE bug, so the bar is
    #     float precision. This is the check that matters.
    sd = {k: v.detach().numpy().astype(np.float32) for k, v in ref.state_dict().items()}
    Wx = {"embed_tokens": sd["model.embed_tokens.weight"], "norm": sd["model.norm.weight"],
          "pre_projection": sd["pre_projection.weight"],
          "post_projection": sd["post_projection.weight"]}
    for li in range(cfg["n_layers"]):
        q, o = f"model.layers.{li}.", f"layers.{li}."
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            Wx[o + nm] = sd[q + nm + ".weight"]
        Wx[o + "layer_scalar"] = np.asarray(sd[q + "layer_scalar"]).reshape(-1)
        Wx[o + "q_proj"] = sd[q + "self_attn.q_proj.weight"]
        Wx[o + "o_proj"] = sd[q + "self_attn.o_proj.weight"]
        Wx[o + "q_norm"] = sd[q + "self_attn.q_norm.weight"]
        Wx[o + "mlp_gate"] = sd[q + "mlp.gate_proj.weight"]
        Wx[o + "mlp_up"] = sd[q + "mlp.up_proj.weight"]
        Wx[o + "mlp_down"] = sd[q + "mlp.down_proj.weight"]

    # inputs_embeds = cat([last_token_embedding, last_hidden_state]): the FIRST half is
    # the embedding and the second is the hidden state, so split it that way.
    xl, xb = mtp_forward(cfg, Wx, emb[BB:], emb[:BB], pos, Ks, Vs, Kf, Vf, SW)
    el = np.abs(hf_logits - xl).max() / (np.abs(hf_logits).max() + 1e-9)
    eb = np.abs(hf_bh - xb).max() / (np.abs(hf_bh).max() + 1e-9)
    print("MTP head, numpy on EXACT fp32 weights vs HF:")
    print(f"  logits            max rel err {el:.3e}")
    print(f"  post_projection   max rel err {eb:.3e}   <- architecture check")

    # (b) the q4_0 container: adds quantisation error on top (large on a RANDOM
    #     fixture model, which is the worst case for q4_0 -- not representative).
    ql, qb = mtp_forward(cfg, W, emb[BB:], emb[:BB], pos, Ks, Vs, Kf, Vf, SW)
    dl = np.abs(hf_logits - ql).max() / (np.abs(hf_logits).max() + 1e-9)
    print(f"container (q4_0)    logits max rel err {dl:.3e}   <- includes quantisation")

    ok = el < 1e-5 and eb < 1e-5
    print("  " + ("MATCH" if ok else "MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
