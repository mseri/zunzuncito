/* g4tok.h — SentencePiece-BPE tokenizer for Gemma-4 (262K vocab, byte_fallback).
 *
 * Loads the flat container produced by tools/convert_tokenizer.py. Every merge rule
 * was resolved to token IDs offline, so the BPE loop here is integer-only: no string
 * comparisons, no hashing of substrings. Merge lookup is a single open-addressed
 * probe on the 64-bit key (a << 32 | b).
 *
 * ENCODE
 *   1. normalise: apply the Replace rules (Gemma's is ' ' -> U+2581 "▁").
 *   2. carve out special/added tokens by exact match (control tokens like <|think|>
 *      must never be BPE'd).
 *   3. seed symbols: one per UTF-8 codepoint. A codepoint absent from the vocab
 *      falls back to its raw BYTES as <0xNN> tokens -- that is what byte_fallback
 *      means, and it is why this tokenizer can never emit <unk> on valid UTF-8.
 *   4. BPE: repeatedly apply the lowest-RANK applicable merge. Rank, not greed:
 *      picking the leftmost or the longest merge gives different (wrong) output.
 *
 * Symbols live in a doubly-linked list so a merge is O(1) and the scan is O(n) per
 * round. Prompts are short; this is not the bottleneck (a 3.19 MiB expert read is).
 */
#ifndef G4TOK_H
#define G4TOK_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

typedef struct { int32_t a, b, ab; } G4Merge;

typedef struct {
    int n_vocab, n_merges, n_special, n_replace;
    int32_t unk, byte_fallback;
    int32_t byte_tok[256];

    char **tok;            /* [n_vocab] NUL-terminated (may contain no NULs itself) */
    uint16_t *tlen;

    /* merge table: open-addressed map (a,b) -> rank, and rank -> merged id */
    uint64_t *mkey;        /* 0 = empty; key is ((a+1)<<32 | (b+1)) so 0 is free */
    int32_t *mrank;
    int32_t *mid;
    size_t mcap;

    int32_t *sp_id;  char **sp_str;  uint16_t *sp_len;      /* special tokens */
    char **rf; uint16_t *rfl; char **rt; uint16_t *rtl;     /* replace rules */

    /* id lookup for a literal string (used to seed symbols) */
    uint64_t *vkey;  int32_t *vval;  size_t vcap;
} G4Tok;

/* FNV-1a over bytes */
static uint64_t g4_hash(const char *s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ULL; }
    return h ? h : 1;
}

static int32_t g4_lookup(const G4Tok *t, const char *s, int n) {
    uint64_t h = g4_hash(s, n);
    size_t i = h & (t->vcap - 1);
    for (;;) {
        if (!t->vkey[i]) return -1;
        if (t->vkey[i] == h) {
            int32_t id = t->vval[i];
            if (t->tlen[id] == n && !memcmp(t->tok[id], s, n)) return id;
        }
        i = (i + 1) & (t->vcap - 1);
    }
}

static int g4_merge_find(const G4Tok *t, int32_t a, int32_t b, int32_t *id) {
    uint64_t k = ((uint64_t)(a + 1) << 32) | (uint32_t)(b + 1);
    size_t i = (k * 1099511628211ULL) & (t->mcap - 1);
    for (;;) {
        if (!t->mkey[i]) return -1;
        if (t->mkey[i] == k) { *id = t->mid[i]; return t->mrank[i]; }
        i = (i + 1) & (t->mcap - 1);
    }
}

static size_t g4_pow2(size_t n) { size_t c = 8; while (c < n * 2) c <<= 1; return c; }

static G4Tok *g4tok_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return NULL; }
    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "G4TK", 4)) {
        fprintf(stderr, "%s: bad magic\n", path); fclose(f); return NULL;
    }
    G4Tok *t = calloc(1, sizeof *t);
    uint32_t nv, nm, ns, nr, bf; int32_t unk;
    if (fread(&nv, 4, 1, f) != 1 || fread(&nm, 4, 1, f) != 1 ||
        fread(&ns, 4, 1, f) != 1 || fread(&nr, 4, 1, f) != 1 ||
        fread(&unk, 4, 1, f) != 1 || fread(&bf, 4, 1, f) != 1) goto bad;
    t->n_vocab = nv; t->n_merges = nm; t->n_special = ns; t->n_replace = nr;
    t->unk = unk; t->byte_fallback = bf;
    if (fread(t->byte_tok, 4, 256, f) != 256) goto bad;

    t->tok = calloc(nv, sizeof(char *));
    t->tlen = calloc(nv, sizeof(uint16_t));
    for (uint32_t i = 0; i < nv; i++) {
        uint16_t L;
        if (fread(&L, 2, 1, f) != 1) goto bad;
        t->tok[i] = malloc(L + 1);
        if (L && fread(t->tok[i], 1, L, f) != L) goto bad;
        t->tok[i][L] = 0;
        t->tlen[i] = L;
    }

    t->mcap = g4_pow2(nm ? nm : 1);
    t->mkey = calloc(t->mcap, 8);
    t->mrank = calloc(t->mcap, 4);
    t->mid = calloc(t->mcap, 4);
    for (uint32_t r = 0; r < nm; r++) {
        G4Merge m;
        if (fread(&m, sizeof m, 1, f) != 1) goto bad;
        uint64_t k = ((uint64_t)(m.a + 1) << 32) | (uint32_t)(m.b + 1);
        size_t i = (k * 1099511628211ULL) & (t->mcap - 1);
        while (t->mkey[i] && t->mkey[i] != k) i = (i + 1) & (t->mcap - 1);
        if (!t->mkey[i]) {                 /* first rule for this pair wins: lowest rank */
            t->mkey[i] = k; t->mrank[i] = r; t->mid[i] = m.ab;
        }
    }

    t->sp_id = calloc(ns ? ns : 1, 4);
    t->sp_str = calloc(ns ? ns : 1, sizeof(char *));
    t->sp_len = calloc(ns ? ns : 1, 2);
    for (uint32_t i = 0; i < ns; i++) {
        uint16_t L;
        if (fread(&t->sp_id[i], 4, 1, f) != 1 || fread(&L, 2, 1, f) != 1) goto bad;
        t->sp_str[i] = malloc(L + 1);
        if (L && fread(t->sp_str[i], 1, L, f) != L) goto bad;
        t->sp_str[i][L] = 0;
        t->sp_len[i] = L;
    }

    t->rf = calloc(nr ? nr : 1, sizeof(char *));
    t->rt = calloc(nr ? nr : 1, sizeof(char *));
    t->rfl = calloc(nr ? nr : 1, 2);
    t->rtl = calloc(nr ? nr : 1, 2);
    for (uint32_t i = 0; i < nr; i++) {
        uint16_t L;
        if (fread(&L, 2, 1, f) != 1) goto bad;
        t->rf[i] = malloc(L + 1);
        if (L && fread(t->rf[i], 1, L, f) != L) goto bad;
        t->rf[i][L] = 0; t->rfl[i] = L;
        if (fread(&L, 2, 1, f) != 1) goto bad;
        t->rt[i] = malloc(L + 1);
        if (L && fread(t->rt[i], 1, L, f) != L) goto bad;
        t->rt[i][L] = 0; t->rtl[i] = L;
    }
    fclose(f);

    t->vcap = g4_pow2(nv);
    t->vkey = calloc(t->vcap, 8);
    t->vval = calloc(t->vcap, 4);
    for (uint32_t i = 0; i < nv; i++) {
        if (!t->tlen[i]) continue;
        uint64_t h = g4_hash(t->tok[i], t->tlen[i]);
        size_t j = h & (t->vcap - 1);
        while (t->vkey[j]) j = (j + 1) & (t->vcap - 1);
        t->vkey[j] = h; t->vval[j] = i;
    }
    return t;
bad:
    fprintf(stderr, "%s: truncated\n", path);
    fclose(f);
    return NULL;
}

/* ------------------------------------------------------------------ encode */
typedef struct { int32_t id; int prev, next; } G4Sym;

/* BPE one span of normalised text (no special tokens inside) */
static int g4_bpe(const G4Tok *t, const char *s, int n, int *out, int cap, int nout) {
    if (!n) return nout;
    G4Sym *sy = malloc(sizeof(G4Sym) * (size_t)(n + 1));
    int ns = 0;

    for (int i = 0; i < n;) {
        /* one UTF-8 codepoint */
        unsigned char c = s[i];
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + len > n) len = 1;
        int32_t id = g4_lookup(t, s + i, len);
        if (id >= 0) {
            sy[ns].id = id; sy[ns].prev = ns - 1; sy[ns].next = ns + 1; ns++;
        } else if (t->byte_fallback) {
            /* not in vocab: emit its raw bytes as <0xNN>. This is why a
             * byte_fallback tokenizer never produces <unk> on valid UTF-8.
             * If the vocab lacks the byte tokens (only possible in a degenerate
             * vocab -- the real Gemma one has all 256), fall back to a single unk
             * for the whole codepoint rather than DROPPING the input. */
            int have = 1;
            for (int k = 0; k < len; k++)
                if (t->byte_tok[(unsigned char)s[i + k]] < 0) { have = 0; break; }
            if (have) {
                for (int k = 0; k < len; k++) {
                    sy[ns].id = t->byte_tok[(unsigned char)s[i + k]];
                    sy[ns].prev = ns - 1; sy[ns].next = ns + 1; ns++;
                }
            } else if (t->unk >= 0) {
                sy[ns].id = t->unk; sy[ns].prev = ns - 1; sy[ns].next = ns + 1; ns++;
            }
        } else if (t->unk >= 0) {
            sy[ns].id = t->unk; sy[ns].prev = ns - 1; sy[ns].next = ns + 1; ns++;
        }
        i += len;
    }
    if (ns) sy[ns - 1].next = -1;

    /* repeatedly apply the LOWEST-RANK applicable merge (not the leftmost, not the
     * longest -- rank order is what defines BPE, and the other two give wrong output) */
    for (;;) {
        int best = -1, brank = 0x7fffffff;
        int32_t bid = -1;
        for (int i = 0; i != -1 && i < ns; i = sy[i].next) {
            int j = sy[i].next;
            if (j == -1 || j >= ns) break;
            int32_t id;
            int r = g4_merge_find(t, sy[i].id, sy[j].id, &id);
            if (r >= 0 && r < brank) { brank = r; best = i; bid = id; }
        }
        if (best < 0) break;
        int j = sy[best].next;
        sy[best].id = bid;
        sy[best].next = sy[j].next;
        if (sy[j].next != -1) sy[sy[j].next].prev = best;
    }

    for (int i = 0; i != -1 && i < ns; i = sy[i].next) {
        if (nout < cap) out[nout++] = sy[i].id;
        if (sy[i].next == i) break;
    }
    free(sy);
    return nout;
}

/* normalise via the Replace rules, then BPE around any special tokens */
static int g4tok_encode(const G4Tok *t, const char *text, int *out, int cap) {
    /* 1. normalise */
    size_t n = strlen(text);
    size_t bcap = n * 4 + 16;
    char *buf = malloc(bcap);
    size_t bn = 0;
    for (size_t i = 0; i < n;) {
        int hit = -1;
        for (int r = 0; r < t->n_replace; r++) {
            if (!strncmp(t->rf[r], "\x00PREPEND", 8)) continue;
            if (t->rfl[r] && i + t->rfl[r] <= n && !memcmp(text + i, t->rf[r], t->rfl[r])) {
                hit = r; break;
            }
        }
        if (hit >= 0) {
            memcpy(buf + bn, t->rt[hit], t->rtl[hit]);
            bn += t->rtl[hit];
            i += t->rfl[hit];
        } else {
            buf[bn++] = text[i++];
        }
    }

    /* 2. carve out special tokens; BPE the gaps */
    int nout = 0;
    size_t i = 0, seg = 0;
    while (i < bn) {
        int hit = -1;
        for (int s = 0; s < t->n_special; s++) {
            uint16_t L = t->sp_len[s];
            if (L && i + L <= bn && !memcmp(buf + i, t->sp_str[s], L)) {
                if (hit < 0 || L > t->sp_len[hit]) hit = s;   /* longest match wins */
            }
        }
        if (hit >= 0) {
            nout = g4_bpe(t, buf + seg, (int)(i - seg), out, cap, nout);
            if (nout < cap) out[nout++] = t->sp_id[hit];
            i += t->sp_len[hit];
            seg = i;
        } else {
            i++;
        }
    }
    nout = g4_bpe(t, buf + seg, (int)(bn - seg), out, cap, nout);
    free(buf);
    return nout;
}

/* ------------------------------------------------------------------ decode */
static int g4tok_decode(const G4Tok *t, const int *ids, int n, char *out, int cap) {
    int m = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= t->n_vocab) continue;
        const char *s = t->tok[id];
        int L = t->tlen[id];

        /* <0xNN> -> the raw byte it stands for */
        if (t->byte_fallback && L == 6 && s[0] == '<' && s[1] == '0' && s[2] == 'x' && s[5] == '>') {
            int hi = s[3], lo = s[4];
            #define HEX(ch) ((ch) >= '0' && (ch) <= '9' ? (ch) - '0' : \
                             (ch) >= 'A' && (ch) <= 'F' ? (ch) - 'A' + 10 : \
                             (ch) >= 'a' && (ch) <= 'f' ? (ch) - 'a' + 10 : -1)
            int h = HEX(hi), l = HEX(lo);
            #undef HEX
            if (h >= 0 && l >= 0) {
                if (m < cap - 1) out[m++] = (char)(h * 16 + l);
                continue;
            }
        }
        for (int k = 0; k < L && m < cap - 1; k++) out[m++] = s[k];
    }
    out[m] = 0;

    /* undo the Replace rules (U+2581 -> ' ') */
    for (int r = 0; r < t->n_replace; r++) {
        if (!t->rtl[r] || !strncmp(t->rf[r], "\x00PREPEND", 8)) continue;
        char *p;
        int fl = t->rtl[r], tl = t->rfl[r];
        while ((p = strstr(out, t->rt[r]))) {
            memmove(p + tl, p + fl, strlen(p + fl) + 1);
            memcpy(p, t->rf[r], tl);
        }
    }
    return (int)strlen(out);
}

#endif /* G4TOK_H */
