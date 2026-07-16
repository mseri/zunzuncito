#!/usr/bin/env python3
"""
convert_gemma4_mtp.py — Gemma-4 MTP drafter (Gemma4AssistantForCausalLM) -> container.

WHAT THIS THING IS
  Not a model. An EAGLE/MTP-style draft HEAD that runs inside the backbone's forward:

    * its layers have NO k_proj / v_proj / k_norm. num_kv_shared_layers == num_layers,
      so `is_kv_shared_layer` is true for every layer and each one takes K and V
      straight from the BACKBONE's `shared_kv_states[layer_type]`. Concretely: the
      assistant's sliding layers attend into the backbone's LAST sliding layer's KV,
      and its full layer into the backbone's LAST full layer's KV. (The backbone
      publishes exactly those two via `store_full_length_kv`.)
    * pre_projection is [hidden, 2 * backbone_hidden]: it consumes a concat of two
      backbone-space (2816-dim) vectors.
    * post_projection maps hidden -> backbone_hidden, so the head can iterate in
      backbone space across draft steps.
    * enable_moe_block = false: the plain decoder layer, no router, no experts, and
      none of the extra FFN norms.

  So the draft loop is:
      e_0 = concat(backbone_hidden, backbone_embed(tok))     -> pre_projection
      4 layers (attending into the TARGET's KV) -> norm
      logits = lm_head(h)            -> the drafted token
      backbone_hidden' = post_projection(h)                  -> feeds the next step

  Everything is small: 4 layers, hidden 1024, ~0.4 B params. It is resident, never
  streamed.

OUTPUT: mtp.bin / mtp.idx / mtp.cfg.json, same conventions as the main container
(q4_0 matrices, f32 norms).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_gemma4 import Shards, Dense, q40_bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="the -assistant checkpoint dir")
    ap.add_argument("dst", help="the TARGET container dir (mtp.* is written into it)")
    a = ap.parse_args()

    raw = json.load(open(os.path.join(a.src, "config.json")))
    arch = (raw.get("architectures") or [""])[0]
    if raw.get("model_type") != "gemma4_assistant" and "Assistant" not in arch:
        sys.exit(f"not a Gemma-4 assistant checkpoint (got {arch or raw.get('model_type')})")

    t = raw["text_config"]
    L = t["num_hidden_layers"]
    D = t["hidden_size"]
    BB = raw["backbone_hidden_size"]
    types = [1 if x == "full_attention" else 0 for x in t["layer_types"]]

    if t.get("num_kv_shared_layers") != L:
        sys.exit(f"expected every layer to share KV (num_kv_shared_layers={L}), "
                 f"got {t.get('num_kv_shared_layers')}")
    if t.get("enable_moe_block"):
        sys.exit("assistant should not have a MoE block")
    if raw.get("use_ordered_embeddings"):
        sys.exit("use_ordered_embeddings=True (centroid LM head) is not supported; "
                 "this converter assumes the plain tied lm_head")

    S = Shards(a.src)
    os.makedirs(a.dst, exist_ok=True)
    dn = Dense(os.path.join(a.dst, "mtp.bin"))

    # tied: model.embed_tokens doubles as lm_head [vocab, 1024]
    dn.add_q40("embed_tokens", S.get("model.embed_tokens.weight"))
    dn.add_f32("norm", S.get("model.norm.weight"))
    dn.add_q40("pre_projection", S.get("pre_projection.weight"))    # [D, 2*BB]
    dn.add_q40("post_projection", S.get("post_projection.weight"))  # [BB, D]

    for li in range(L):
        q, o = f"model.layers.{li}.", f"layers.{li}."
        for nm in ("input_layernorm", "post_attention_layernorm",
                   "pre_feedforward_layernorm", "post_feedforward_layernorm"):
            dn.add_f32(o + nm, S.get(q + nm + ".weight"))
        dn.add_f32(o + "layer_scalar", np.asarray(S.get(q + "layer_scalar")).reshape(-1))

        # q_proj and o_proj only -- there is no k_proj/v_proj/k_norm, by design:
        # K and V come from the backbone's cache.
        dn.add_q40(o + "q_proj", S.get(q + "self_attn.q_proj.weight"))
        dn.add_q40(o + "o_proj", S.get(q + "self_attn.o_proj.weight"))
        dn.add_f32(o + "q_norm", S.get(q + "self_attn.q_norm.weight"))
        for bad in ("k_proj", "v_proj", "k_norm"):
            if S.has(q + f"self_attn.{bad}.weight"):
                sys.exit(f"unexpected {bad} in the assistant: it should share the "
                         f"backbone's KV, not compute its own")

        dn.add_q40(o + "mlp_gate", S.get(q + "mlp.gate_proj.weight"))
        dn.add_q40(o + "mlp_up", S.get(q + "mlp.up_proj.weight"))
        dn.add_q40(o + "mlp_down", S.get(q + "mlp.down_proj.weight"))

    dn.close(os.path.join(a.dst, "mtp.idx"))

    cfg = dict(
        hidden=D, backbone_hidden=BB, n_layers=L,
        n_heads=t["num_attention_heads"], head_dim=t["head_dim"],
        global_head_dim=t["global_head_dim"], dense_inter=t["intermediate_size"],
        vocab=t["vocab_size"], eps=t["rms_norm_eps"],
        layer_types=types,
        rope_theta_local=t["rope_parameters"]["sliding_attention"]["rope_theta"],
        rope_theta_global=t["rope_parameters"]["full_attention"]["rope_theta"],
        rope_partial_global=t["rope_parameters"]["full_attention"].get("partial_rotary_factor", 1.0),
        final_logit_softcap=float(t.get("final_logit_softcapping") or 0.0),
    )
    json.dump(cfg, open(os.path.join(a.dst, "mtp.cfg.json"), "w"), indent=1)

    FMT = {"f32": 0, "q40": 1}
    with open(os.path.join(a.dst, "mtp.manifest.txt"), "w") as m:
        for k, v in cfg.items():
            if k == "layer_types":
                m.write("cfg layer_types " + " ".join(str(x) for x in v) + "\n")
            else:
                m.write(f"cfg {k} {v}\n")
        m.write(f"ndense {len(dn.idx)}\n")
        for k, v in dn.idx.items():
            sh = v["shape"]
            O = sh[0]
            I = sh[1] if len(sh) > 1 else 1
            m.write(f"dense {k} {v['off']} {v['len']} {FMT[v['fmt']]} {O} {I}\n")

    print(f"mtp: {L} layers, hidden {D}, backbone {BB}, {dn.off/2**20:.1f} MiB")
    print("layer types:", " ".join("full" if x else "slide" for x in types))
    print("\nRun with:  ./gemma4 <dir> --mtp --ndraft 4")


if __name__ == "__main__":
    main()
