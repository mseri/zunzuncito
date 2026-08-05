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

     The digit run is the one part that varies between checkpoints; it travels in
     the container as digit_max (see _PATTERN_TEMPLATE).

  3. ignore_merges. When set, a piece whose byte-level spelling is ITSELF a vocab
     token is emitted as that token directly, without running BPE. This is not an
     optimisation: for these vocabs the BPE result can differ.

As in convert_tokenizer.py, merge rules are resolved to token IDs offline so the
engine's inner loop is integer-only.

Format (little-endian):
  magic "LFT2"           u32     ("LFTK" is the same minus digit_max, digit_max = 3)
  n_vocab               u32
  n_merges              u32
  n_special             u32
  n_lrange              u32     \\p{L} ranges
  n_nrange              u32     \\p{N} ranges
  ignore_merges         u32
  bos_id                i32     (-1 if none)
  eos_id                i32
  digit_max             i32     longest \\p{N} run one piece may take (LFT2 only)
  n_qcrange             u32     (LFT2 only)
  vocab: n_vocab x { len u16, bytes }        (byte-level spellings, UTF-8)
  merges: n_merges x { a i32, b i32, ab i32 }   (rank = index)
  special: n_special x { id i32, len u16, bytes }   (RAW text, not byte-level)
  lranges: n_lrange x { lo u32, hi u32 }
  nranges: n_nrange x { lo u32, hi u32 }
  qcranges: n_qcrange x { lo u32, hi u32 }   codepoints NFC may rewrite (LFT2 only)
"""
import argparse, json, re, struct, sys, unicodedata

# The engine hand-codes one regex, with exactly one degree of freedom: how many
# digits a number piece may take. LFM2.5 uses \p{N}{1,3}, Maple/Qwen2 a bare \p{N}.
# Everything else must match character for character, so the pattern is compared
# against these templates rather than sniffed for a substring: a checkpoint whose
# split differs anywhere else must fail loudly, not tokenise subtly wrong.
_PATTERN_TEMPLATE = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}%s"
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)


def digit_run(pattern):
    """Longest digit run the split allows, or None if the pattern is not one we
    hand-code."""
    for suffix, dmax in (("", 1), ("{1,3}", 3)):
        if pattern == _PATTERN_TEMPLATE % suffix:
            return dmax
    return None


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


def ranges(codepoints):
    """Collapse a sorted iterable of codepoints into sorted [lo,hi] ranges, which is
    what the engine binary-searches."""
    out = []
    lo = prev = None
    for c in codepoints:
        if prev is not None and c == prev + 1:
            prev = c
            continue
        if lo is not None:
            out.append((lo, prev))
        lo = prev = c
    if lo is not None:
        out.append((lo, prev))
    return out


def category_ranges(major):
    """Sorted [lo,hi] ranges of codepoints whose general category starts with
    `major` ('L' or 'N'). Built from Python's own Unicode database."""
    return ranges(c for c in range(0x110000)
                  if unicodedata.category(chr(c))[0] == major)


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
    dmax = digit_run(pattern)
    if dmax is None:
        sys.exit(f"unexpected split pattern: {pattern!r}")
    if subs[1].get("add_prefix_space"):
        sys.exit("add_prefix_space is not supported")

    # Normalisation. LFM2.5 has none; Maple/Qwen2 declares NFC. A full NFC
    # implementation in C means canonical decomposition, combining-class reordering
    # and a composition table -- disproportionate here, because NFC is the identity
    # on any string that is already in NFC, which is essentially all prompt text.
    #
    # So instead of pretending, we ship the set where it might NOT be the identity:
    # the codepoints whose NFC quick-check is not Yes. A string containing none of
    # them is provably unchanged by NFC, and the engine's ids provably match HF's.
    # The engine checks that per prompt and says so when it fails, rather than
    # quietly tokenising differently.
    qr = []
    norm = (d.get("normalizer") or {}).get("type")
    if norm == "NFC":
        qr = ranges(c for c in range(0x110000)
                    if not unicodedata.is_normalized("NFC", chr(c)))
    elif norm:
        sys.exit(f"normaliser {norm} not supported")

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
    # Only used to seed the engine's defaults; both engines name their stop tokens
    # explicitly. LFM2.5 opens with <|startoftext|>, Maple/Qwen2 with <|endoftext|>.
    bos = find_id("<|startoftext|>")
    if bos < 0:
        bos = find_id("<|endoftext|>")
    eos = find_id("<|im_end|>")

    lr = category_ranges("L")
    nr = category_ranges("N")

    with open(a.out, "wb") as f:
        f.write(b"LFT2")
        f.write(struct.pack("<IIIIIIiiiI", len(toks), len(merges), len(special),
                            len(lr), len(nr), ignore_merges, bos, eos, dmax,
                            len(qr)))
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
        for lo, hi in qr:
            f.write(struct.pack("<II", lo, hi))

    print(f"vocab {len(toks)}  merges {len(merges)} (dropped {dropped})  "
          f"special {len(special)}  ignore_merges {ignore_merges}  "
          f"L-ranges {len(lr)}  N-ranges {len(nr)}  bos {bos}  eos {eos}  "
          f"digits 1..{dmax}  normaliser {norm or 'none'} "
          f"({len(qr)} non-NFC ranges)")


if __name__ == "__main__":
    main()
