#!/usr/bin/env python3
"""
gemma4_mtp_fixture.py — a tiny Gemma-4 MTP head with the REAL architecture, and the
HF reference outputs to check gemma4.c's mtp_forward against.

What this pins down (each is a way to be silently wrong):
  * every assistant layer takes K and V from the BACKBONE's shared_kv_states, keyed by
    layer_type -- not from its own projections, which do not exist;
  * the plain (non-MoE) decoder layer: no router, no experts, no extra FFN norms;
  * pre_projection consumes 2 * backbone_hidden, post_projection returns backbone_hidden;
  * the tied lm_head, and no final logit softcap.

It does NOT pin down the ORDER of the two halves of the concat, whether the embedding
is scaled, or whether the backbone hidden is pre/post final norm -- those live in the
generation glue, not the model.
"""
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gemma4_fixture import TEXT_CFG


def make(dst):
    import torch
    from transformers.models.gemma4_assistant.modeling_gemma4_assistant import (
        Gemma4AssistantForCausalLM,
    )
    from transformers.models.gemma4_assistant.configuration_gemma4_assistant import (
        Gemma4AssistantConfig,
    )

    BB = TEXT_CFG["hidden_size"]          # backbone hidden (the fixture's)
    L = 4
    tcfg = dict(
        vocab_size=TEXT_CFG["vocab_size"], hidden_size=32, intermediate_size=64,
        num_hidden_layers=L, num_attention_heads=TEXT_CFG["num_attention_heads"],
        num_key_value_heads=TEXT_CFG["num_key_value_heads"],
        head_dim=TEXT_CFG["head_dim"], global_head_dim=TEXT_CFG["global_head_dim"],
        num_global_key_value_heads=TEXT_CFG["num_global_key_value_heads"],
        attention_k_eq_v=True, enable_moe_block=False, num_experts=None,
        moe_intermediate_size=None, top_k_experts=None,
        num_kv_shared_layers=L,                     # every layer shares the backbone's KV
        sliding_window=TEXT_CFG["sliding_window"],
        hidden_size_per_layer_input=0, vocab_size_per_layer_input=0,
        final_logit_softcapping=None,
        layer_types=["sliding_attention"] * (L - 1) + ["full_attention"],
        rope_parameters=TEXT_CFG["rope_parameters"],
        rms_norm_eps=1e-6, hidden_activation="gelu_pytorch_tanh",
    )
    cfg = Gemma4AssistantConfig(text_config=tcfg, backbone_hidden_size=BB,
                                use_ordered_embeddings=False,
                                num_centroids=8, centroid_intermediate_top_k=2)

    torch.manual_seed(1)
    m = Gemma4AssistantForCausalLM(cfg).eval().float()
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.08)
    for n, b in m.named_buffers():
        if n.endswith("layer_scalar"):
            b.copy_(torch.tensor([0.9]))

    os.makedirs(dst, exist_ok=True)
    from safetensors.torch import save_file
    sd = {k: v.detach().contiguous() for k, v in m.state_dict().items()
          if not k.startswith("lm_head")}          # tied to model.embed_tokens
    save_file(sd, os.path.join(dst, "model.safetensors"))
    json.dump({
        "architectures": ["Gemma4AssistantForCausalLM"],
        "model_type": "gemma4_assistant",
        "backbone_hidden_size": BB,
        "use_ordered_embeddings": False,
        "num_centroids": 8, "centroid_intermediate_top_k": 2,
        "tie_word_embeddings": True,
        "text_config": {**tcfg, "model_type": "gemma4_text"},
    }, open(os.path.join(dst, "config.json"), "w"), indent=1)
    return m, cfg


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mtpfix")
    print("mtp fixture written")
