#!/usr/bin/env python3
"""
maple_mlx_check.py — diff the engine against the mlx-lm reference on the REAL
checkpoint, end to end.

tools/maple_oracle.py already proves the engine reproduces a numpy forward pass over
the container weights to float precision. What it cannot catch is a systematic
misreading of the architecture, because the oracle is a second transcription of the
same reading -- if maple.py's RoPE convention, sliding-window boundary or up/gate
assignment were understood wrongly, both would be wrong together. This script closes
that gap by running the actual reference implementation.

The comparison is greedy generation, not logits. The two engines quantise and
accumulate differently (mlx computes in bf16 on the GPU, this one in f32 with int8
activations), so their logits differ by percent-level amounts that say nothing; what
must agree is the token stream. A divergence in the first few tokens is an
architecture bug. A divergence deep into a long generation, at a position where the
top two logits are near-tied, is not.

Needs mlx and mlx-lm, and therefore a Metal GPU -- it will not run in a headless or
sandboxed session.

    python3 tools/maple_mlx_check.py /path/to/maple-preview-mlx ./maple-ct
"""
import argparse, json, os, subprocess, sys

PROMPTS = [
    "What is the capital of France?",
    "Write the numbers one to five, comma separated.",
    "Explain in one sentence why the sky is blue.",
    "def fibonacci(n):",
    "Le chat s'est assis sur le tapis. Traduis en anglais.",
]


def mlx_generate(model_dir, prompts, n, use_flash):
    from mlx_lm import load
    from mlx_lm.generate import generate
    from mlx_lm.sample_utils import make_sampler

    cfg = {"use_flash_head": True} if use_flash else {}
    model, tok = load(model_dir, model_config=cfg)
    out = []
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}],
                                       tokenize=False, add_generation_prompt=True)
        out.append(generate(model, tok, prompt=text, max_tokens=n, verbose=False,
                            sampler=make_sampler(temp=0.0)))
    return out


def engine_generate(binary, container, prompts, n, extra):
    out = []
    for p in prompts:
        r = subprocess.run([binary, container, "--temp", "0", "--max_tokens", str(n),
                            *extra, "--", p],
                           capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"{binary} failed: {r.stderr}")
        # strip the trailing timing block the CLI prints after the completion
        txt = r.stdout
        cut = txt.find("\n\nprefill ")
        out.append(txt[:cut] if cut >= 0 else txt)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir", help="the HF/MLX checkpoint")
    ap.add_argument("container", help="the converted maple container")
    ap.add_argument("--binary", default="./maple")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--flash", action="store_true",
                    help="compare the FlashHead paths against each other instead of "
                         "the exact heads")
    a = ap.parse_args()

    extra = [] if a.flash else ["--noflash"]
    print(f"generating {a.tokens} greedy tokens for {len(PROMPTS)} prompts "
          f"({'flash' if a.flash else 'exact'} head) ...", flush=True)
    mine = engine_generate(a.binary, a.container, PROMPTS, a.tokens, extra)
    theirs = mlx_generate(a.model_dir, PROMPTS, a.tokens, a.flash)

    bad = 0
    for p, m, t in zip(PROMPTS, mine, theirs):
        # compare on the common prefix length: the two stop conditions differ
        k = min(len(m), len(t))
        if m[:k].strip() == t[:k].strip():
            print(f"  ok        {p!r}")
            continue
        bad += 1
        common = 0
        while common < k and m[common] == t[common]:
            common += 1
        print(f"  DIVERGES  {p!r}\n    after {common} chars\n"
              f"    mlx   {t[max(0,common-40):common+80]!r}\n"
              f"    maple {m[max(0,common-40):common+80]!r}")

    print(f"\n{len(PROMPTS) - bad}/{len(PROMPTS)} prompts identical")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
