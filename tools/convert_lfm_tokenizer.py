#!/usr/bin/env python3
"""
convert_lfm_tokenizer.py — HF ByteLevel-BPE tokenizer.json -> flat binary for lfm25.c.

LFM2.5's tokenizer is a GPT-2-style ByteLevel BPE, NOT the SentencePiece one gemma4
uses, so it needs its own container (magic "LFTK") and its own engine-side reader
(lfmtok.h). Three things differ and each of them changes the output if you get it
wrong:

  1. BYTE LEVEL. There is no byte_fallback and no <unk>. Every input byte is mapped
     through the GPT-2 byte->unicode table (0x20 -> U+0120 'Ġ', etc.) and BPE runs
     in THAT alphabet. So the vocab strings are the byte-level spellings ("ĠThe"),
     and a raw UTF-8 byte never reaches the merge table directly.

  2. A PRE-TOKENIZER SPLIT. The text is first cut by the Qwen2/Llama-3 regex

       (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}
       | ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+

     and BPE runs INDEPENDENTLY inside each piece. Merges never cross a piece
     boundary, so a tokenizer that skips this step produces different (wrong) ids.
     The engine hand-codes the regex, but it needs Unicode \\p{L} / \\p{N}
     membership to do it -- and shipping a Unicode database in C is silly. So we
     emit the letter and number ranges HERE, as sorted [lo,hi] pairs, and the C
     side binary-searches them. \\s is small enough (25 codepoints) to hardcode.

  3. ignore_merges. When set, a piece whose byte-level spelling is ITSELF a vocab
     token is emitted as that token directly, without running BPE. This is not an
     optimisation: for these vocabs the BPE result can differ.

As in convert_tokenizer.py, merge rules are resolved to token IDs offline so the
engine's inner loop is integer-only.

Format (little-endian):
  magic "LFTK"           u32
  n_vocab               u32
  n_merges              u32
  n_special             u32
  n_lrange              u32     \\p{L} ranges
  n_nrange              u32     \\p{N} ranges
  ignore_merges         u32
  bos_id                i32     (-1 if none)
  eos_id                i32
  vocab: n_vocab x { len u16, bytes }        (byte-level spellings, UTF-8)
  merges: n_merges x { a i32, b i32, ab i32 }   (rank = index)
  special: n_special x { id i32, len u16, bytes }   (RAW text, not byte-level)
  lranges: n_lrange x { lo u32, hi u32 }
  nranges: n_nrange x { lo u32, hi u32 }
"""
import argparse, json, struct, sys, unicodedata

EXPECTED_PATTERN_HINT = "\\p{L}"


def bytes_to_unicode():
    """The GPT-2 byte<->unicode table. Printable-ASCII-ish bytes map to themselves;
    the remaining 68 map to U+0100.. so that no byte is ever a control char or a
    space in the BPE alphabet."""
    bs = (list(range(ord("!"), ord("~") + 1)) +
          list(range(0xA1, 0xAC + 1)) +
          list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, cs))


def category_ranges(major):
    """Sorted [lo,hi] ranges of codepoints whose general category starts with
    `major` ('L' or 'N'). Built from Python's own Unicode database."""
    out = []
    start = None
    for c in range(0x110000):
        hit = unicodedata.category(chr(c))[0] == major
        if hit:
            if start is None:
                start = c
        elif start is not None:
            out.append((start, c - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tokenizer_json")
    ap.add_argument("out")
    a = ap.parse_args()

    d = json.load(open(a.tokenizer_json, encoding="utf-8"))
    mdl = d["model"]
    if mdl["type"] != "BPE":
        sys.exit(f"expected a BPE model, got {mdl['type']}")
    if mdl.get("byte_fallback"):
        sys.exit("byte_fallback ByteLevel BPE is not supported; use convert_tokenizer.py")

    # The engine hand-codes ONE regex. Refuse anything else rather than silently
    # tokenising differently from HF.
    pt = d.get("pre_tokenizer") or {}
    subs = pt.get("pretokenizers", [pt]) if pt.get("type") == "Sequence" else [pt]
    kinds = [s.get("type") for s in subs]
    if kinds != ["Split", "ByteLevel"]:
        sys.exit(f"expected pre_tokenizer Split+ByteLevel, got {kinds}")
    pattern = subs[0].get("pattern", {}).get("Regex", "")
    if EXPECTED_PATTERN_HINT not in pattern:
        sys.exit(f"unexpected split pattern: {pattern!r}")
    if subs[1].get("add_prefix_space"):
        sys.exit("add_prefix_space is not supported")
    if d.get("normalizer"):
        sys.exit(f"normaliser {d['normalizer'].get('type')} not supported")

    vocab = mdl["vocab"]                        # byte-level string -> id
    n = max(vocab.values()) + 1
    toks = [None] * n
    for s, i in vocab.items():
        toks[i] = s
    for aa in d.get("added_tokens", []):        # added tokens may sit outside `vocab`
        if aa["id"] < n:
            toks[aa["id"]] = aa["content"]
        else:
            toks += [None] * (aa["id"] - n + 1)
            n = aa["id"] + 1
            toks[aa["id"]] = aa["content"]
    for i in range(n):
        if toks[i] is None:
            toks[i] = ""                        # holes: unreachable, but keep ids dense

    # Every single byte-level character must be a token, or a piece containing that
    # byte cannot be seeded at all. Unlike byte_fallback there is no second chance.
    b2u = bytes_to_unicode()
    missing = [b for b in range(256) if chr(b2u[b]) not in vocab]
    if missing:
        sys.exit(f"vocab is missing {len(missing)} byte-level characters, e.g. {missing[:8]}")

    merges = []
    dropped = 0
    for m in mdl["merges"]:
        x, y = (m if isinstance(m, list) else m.split(" ", 1))
        ab = vocab.get(x + y)
        ia, ib = vocab.get(x), vocab.get(y)
        if ab is None or ia is None or ib is None:
            dropped += 1
            continue
        merges.append((ia, ib, ab))

    special = [(aa["id"], aa["content"]) for aa in d.get("added_tokens", [])]
    ignore_merges = 1 if mdl.get("ignore_merges") else 0

    # bos/eos are only used to seed the engine's defaults; the chat template names
    # them explicitly anyway.
    def find_id(name):
        for i, s in special:
            if s == name:
                return i
        return -1
    bos = find_id("<|startoftext|>")
    eos = find_id("<|im_end|>")

    lr = category_ranges("L")
    nr = category_ranges("N")

    with open(a.out, "wb") as f:
        f.write(b"LFTK")
        f.write(struct.pack("<IIIIIIii", len(toks), len(merges), len(special),
                            len(lr), len(nr), ignore_merges, bos, eos))
        for t in toks:
            b = t.encode("utf-8")
            f.write(struct.pack("<H", len(b)))
            f.write(b)
        for x, y, z in merges:
            f.write(struct.pack("<iii", x, y, z))
        for i, s in special:
            b = s.encode("utf-8")
            f.write(struct.pack("<iH", i, len(b)))
            f.write(b)
        for lo, hi in lr:
            f.write(struct.pack("<II", lo, hi))
        for lo, hi in nr:
            f.write(struct.pack("<II", lo, hi))

    print(f"vocab {len(toks)}  merges {len(merges)} (dropped {dropped})  "
          f"special {len(special)}  ignore_merges {ignore_merges}  "
          f"L-ranges {len(lr)}  N-ranges {len(nr)}  bos {bos}  eos {eos}")


if __name__ == "__main__":
    main()
