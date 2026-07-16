#!/usr/bin/env python3
"""
gemma4_fixture.py — a tiny random Gemma-4 with the REAL architecture.

The 26B weights are 50 GB and cannot live in CI. But every structural feature that
gemma4.c can get wrong is present at any size: sliding vs global layers, K=V on the
global layers, p-RoPE, the softmax router with per_expert_scale, the parallel
MLP/MoE branches, the tied embedding, the logit softcap. So we build a ~1 MB random
model with all of them, convert it through the real container path, and require the
C engine to reproduce the HF reference logits.

Dims are chosen so every matmul's reduction dim is a multiple of 32 (q4_0's block).
"""
import json, os
import numpy as np


TEXT_CFG = dict(
    vocab_size=128, hidden_size=64, intermediate_size=64, num_hidden_layers=6,
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

PROMPT = [7, 3, 42, 5, 19, 88, 2, 61, 30, 11]


class _SD:
    """Shards-compatible view over a torch state_dict."""
    def __init__(self, sd):
        self.sd = sd

    def has(self, n):
        return n in self.sd

    def get(self, n):
        return self.sd[n]


def make_fixture(dst):
    import torch
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    torch.manual_seed(0)
    hf = Gemma4TextConfig(**TEXT_CFG)
    ref = Gemma4ForCausalLM(hf).eval().float()
    for p in ref.parameters():
        torch.nn.init.normal_(p, std=0.08)
    for n, b in ref.named_buffers():
        if n.endswith("layer_scalar"):
            b.copy_(torch.tensor([0.9]))

    ids = torch.tensor(PROMPT)[None]
    with torch.no_grad():
        logits = ref(input_ids=ids).logits[0].float().numpy()

    os.makedirs(dst, exist_ok=True)
    np.save(os.path.join(dst, "ref_logits.npy"), logits)
    logits.astype(np.float32).tofile(os.path.join(dst, "ref_logits.f32"))
    json.dump({"prompt": PROMPT, "argmax": [int(x) for x in logits.argmax(-1)]},
              open(os.path.join(dst, "ref.json"), "w"))

    sd = {k.replace("model.", "", 1): v.detach().numpy().astype(np.float32)
          for k, v in ref.state_dict().items()}
    # embed_scale is a buffer on the embedding, not a weight; the converter derives
    # it as sqrt(hidden), so assert that is what the reference actually uses.
    es = float(ref.model.embed_tokens.embed_scale)
    assert abs(es - TEXT_CFG["hidden_size"] ** 0.5) < 1e-3, \
        f"embed_scale {es} != sqrt(hidden) {TEXT_CFG['hidden_size']**0.5}"

    return _SD(sd), TEXT_CFG, ""


if __name__ == "__main__":
    import sys
    make_fixture(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fix")
    print("fixture written")
