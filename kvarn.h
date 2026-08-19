/* kvarn.h — KVarN variance-normalised KV-cache quantisation.
 *
 * From huawei-csl/KVarN (Huawei CSL, arXiv 2606.03458), the vLLM backend's
 * `kvarn_k4v2_g128` recipe: Hadamard rotation, log-domain Sinkhorn variance
 * normalisation over a tile, then asymmetric round-to-nearest with the tile's
 * row scales absorbed into the quantiser's own.
 *
 * What it buys: Gemma-4's sliding layers cap their KV at 1024 positions, so 25 of 30
 * layers cost a fixed 400 MiB whatever the context, and all KV growth is in the 5
 * global layers. At 4 GB and 4K ctx, quantising KV buys about one extra expert slot.
 * At 32K it takes the cache from 9 slots/layer to 21, and at 128K it is the
 * difference between running and not running. So this buys context length rather
 * than RAM in general.
 *
 * Why this and not TurboQuant. TurboQuant treats each KV vector on its own: divide
 * out its norm, rotate, and round every coordinate against one standard-normal
 * Lloyd-Max codebook. That is optimal for a vector drawn from an isotropic
 * distribution, and real KV is not isotropic: a handful of channels carry an order
 * of magnitude more variance than the rest, in every layer, for every model. The
 * rotation smears those outliers across all coordinates rather than removing them,
 * so the per-coordinate scale is set by the outlier energy and the bulk of the
 * vector gets a fraction of the codebook. Upstream's own table shows where that
 * lands: K4/V2 without a residual window misses the needle at 2K.
 *
 * KVarN quantises a tile, `group` consecutive tokens by `d` channels, and before
 * rounding drives that tile towards equal variance along both axes by alternating
 * column-wise and row-wise standard-deviation normalisation. That is a Sinkhorn
 * iteration in log space. A hot channel is divided down by its own row scale, a hot
 * token by its own column scale, and both scales are kept, so the map stays exact
 * and nothing is clipped away.
 *
 * The algorithm, per tile:
 *   encode:  rotate every token's vector by the orthonormal Hadamard H
 *            -> lay the tile out as [d, group] for K, [group, d] for V
 *            -> Sinkhorn: 16 alternating col/row std normalisations in log space,
 *               keeping the iterate with the lowest imbalance seen
 *            -> asymmetric RTN per row: q = round((x - lo) / ((hi - lo) / qmax))
 *            -> fold the row's Sinkhorn scale into the RTN scale and zero point,
 *               pack b bits/coord, store the folded row scales and the column scale
 *   decode:  x[r][c] = (q[r][c] * rs[r] + rz[r]) * cs[c]  ->  inverse H
 *
 * Which way up the tile goes decides what the per-row quantiser is per, and that is
 * the one asymmetry between K and V. K is balanced as [d, group], so its rows are
 * channels: a key meets a query channel by channel, and a badly scaled channel
 * damages every score it takes part in. V is balanced as [group, d], so its rows are
 * tokens, and softmax weighting dilutes an error confined to a single token. Hence
 * 4 bits on keys against 2 on values. Same code either way; see kvarn_init's `is_k`.
 *
 * The rotation is a plain Sylvester Hadamard scaled by 1/sqrt(d): orthonormal,
 * symmetric, and its own inverse, so one routine encodes and decodes and there is no
 * seed or sign vector to reproduce. We run it as a fast Walsh-Hadamard transform,
 * O(d log d) rather than the reference's O(d^2) GEMM, which at d=512 and 240 KV
 * writes/token matters. It requires d to be a power of two, and Gemma-4's head dims
 * are 256 and 512 with LFM2.5's at 128, so this is exact rather than a compromise.
 * Unlike TurboQuant's randomised Hadamard there are no sign flips: the balancing is
 * what deals with outliers here, and a deterministic H means an encoder and a
 * decoder built from the same config agree without sharing any state.
 *
 * Tiles are not free. A vector cannot be quantised until the other `group - 1`
 * tokens of its tile exist, so the f32 residual ring has to be a whole number of
 * tiles (kvarn_window) and a speculative caller needs a tile's worth of unsealed
 * slack on top of its draft length. The scale planes also cost more than
 * TurboQuant's one f32 norm per vector: 2*d + group halves per K tile against group
 * f32s, which at d=512 and group=128 is 18 bytes per token rather than 4, and is
 * what a per-channel scale costs.
 *
 * There is no bit-width dial. Upstream exposes four configurations through one vLLM
 * flag (`--kv-cache-dtype kvarn_k4v2_g128` and three siblings), all of them K4, and
 * reports FP16-parity accuracy at the shipped K4/V2. Those four are what the engines
 * offer, under upstream's own names; see kvarn_presets below.
 */
#ifndef KVARN_H
#define KVARN_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#include "q40.h"        /* fp16 <-> f32 for the scale planes */

/* Gemma-4 head dims are 256 (sliding) and 512 (global); LFM2.5 and Maple are 128.
 * Bounding these lets the hot paths use fixed stack arrays instead of alloca, which
 * is a GNU-ism and would overflow the stack if a head dim ever grew. */
#define KVARN_MAXD     1024
#define KVARN_MAXGROUP 512
#define KVARN_MAXBITS  8

/* Upstream's constants (sinkhorn.py). The std clamp keeps a degenerate row from
 * producing an infinite scale; the log-scale clip is deliberately asymmetric, so a
 * scale may grow by e^10 but never shrink below e^-0.3. Normalisation is there to
 * divide outliers down, not to amplify quiet rows. */
#define KVARN_ITERS     16
#define KVARN_STD_MIN   1e-3f
#define KVARN_STD_MAX   1e3f
#define KVARN_LOGS_MIN  (-0.3f)
#define KVARN_LOGS_MAX  10.0f

#define KVARN_CLIP_BITS 4         /* rows narrower than this get the percentile range */
#define KVARN_CLIP_Q    0.005f    /* upstream's example value */
#define KVARN_CLIP_MAXK 16        /* order statistics we are willing to select */

/* ============================================================================
 * presets
 *
 * Upstream's four `--kv-cache-dtype` values, verbatim, and the only KV settings
 * the engines accept. All four are K4, for the reason the orientation note above
 * gives. 128 is the design point; 64 trades a little KV capacity (more per-tile
 * scale bytes per token) for finer granularity at much the same speed.
 *
 * The engines add one thing upstream has no analogue for, fixed rather than
 * exposed: an f32 residual window over the most recent positions. It exists because
 * a tile cannot be sealed until all of its tokens exist, so the newest ones need
 * somewhere to live. Nothing else is layered on top, and in particular every layer
 * gets the preset's own bit widths, so asking for kvarn_k4v2_g128 gets you
 * kvarn_k4v2_g128.
 * ==========================================================================*/

#define KVARN_RWIN 128      /* f32 residual window, in positions */

typedef struct { const char *name; int kbits, vbits, group; } KvarnPreset;

static const KvarnPreset kvarn_presets[] = {
    { "kvarn_k4v2_g128", 4, 2, 128 },   /* what upstream ships */
    { "kvarn_k4v4_g128", 4, 4, 128 },
    { "kvarn_k4v2_g64",  4, 2,  64 },
    { "kvarn_k4v4_g64",  4, 4,  64 },
};

/* Named rather than taken from the head of the table, so reordering the presets
 * cannot quietly change what everyone runs. */
#define KVARN_DEFAULT "kvarn_k4v2_g128"
#define KVARN_NPRESETS ((int)(sizeof kvarn_presets / sizeof *kvarn_presets))

/* NULL if `name` is not one of them. */
static const KvarnPreset *kvarn_preset(const char *name) {
    for (int i = 0; i < KVARN_NPRESETS; i++)
        if (!strcmp(name, kvarn_presets[i].name)) return &kvarn_presets[i];
    return NULL;
}

/* "off | kvarn_k4v2_g128 | ...", for usage text and the one error message. */
static const char *kvarn_preset_list(void) {
    static char buf[256];
    size_t n = 0;
    n += (size_t)snprintf(buf + n, sizeof buf - n, "off");
    for (int i = 0; i < KVARN_NPRESETS && n < sizeof buf; i++)
        n += (size_t)snprintf(buf + n, sizeof buf - n, " | %s", kvarn_presets[i].name);
    return buf;
}

/* ============================================================================
 * Hadamard
 * ==========================================================================*/

/* in-place fast Walsh-Hadamard; d must be a power of two */
static void kvarn_fwht_scalar(float *a, int d, int len0) {
    for (int len = len0; len < d; len <<= 1)
        for (int i = 0; i < d; i += len << 1)
            for (int j = i; j < i + len; j++) {
                float u = a[j], v = a[j + len];
                a[j] = u + v;
                a[j + len] = u - v;
            }
}

/* The transform is 96% of a decode once the packing and the scale planes are out
 * of the way, and a decode happens once per cached position per head per layer per
 * token, so it is the hottest loop in the engine and worth the intrinsics.
 *
 * The stages split in two. Everything from len=8 up is a plain vertical add and
 * subtract of two vectors, eight butterflies at a time with no data movement at
 * all. The three stages below that live inside a single vector, and each is one
 * shuffle plus one multiply-add against a constant sign pattern: at len=1 the
 * partner of lane j is its neighbour, at len=2 the other half of its quad, at
 * len=4 the other 128-bit half, and in each case
 *     out = a * (+1 on the low half of every pair, -1 on the high half) + swapped
 * reproduces u+v and u-v in the right lanes at once. */
#if defined(__AVX2__)
#include <immintrin.h>
static inline __m256 kvarn_fwht8(__m256 a) {
    const __m256 s1 = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    const __m256 s2 = _mm256_setr_ps(1, 1, -1, -1, 1, 1, -1, -1);
    const __m256 s4 = _mm256_setr_ps(1, 1, 1, 1, -1, -1, -1, -1);
    __m256 t = _mm256_permute_ps(a, 0xB1);               /* swap neighbours */
    a = _mm256_add_ps(_mm256_mul_ps(a, s1), t);
    t = _mm256_permute_ps(a, 0x4E);                      /* swap pairs */
    a = _mm256_add_ps(_mm256_mul_ps(a, s2), t);
    t = _mm256_permute2f128_ps(a, a, 0x01);              /* swap halves */
    return _mm256_add_ps(_mm256_mul_ps(a, s4), t);
}
static void kvarn_fwht(float *a, int d) {
    if (d < 8) { kvarn_fwht_scalar(a, d, 1); return; }
    for (int i = 0; i < d; i += 8)
        _mm256_storeu_ps(a + i, kvarn_fwht8(_mm256_loadu_ps(a + i)));
    for (int len = 8; len < d; len <<= 1)
        for (int i = 0; i < d; i += len << 1)
            for (int j = i; j < i + len; j += 8) {
                __m256 u = _mm256_loadu_ps(a + j), v = _mm256_loadu_ps(a + j + len);
                _mm256_storeu_ps(a + j, _mm256_add_ps(u, v));
                _mm256_storeu_ps(a + j + len, _mm256_sub_ps(u, v));
            }
}
static inline void kvarn_rot(float *a, int d) {
    if (d < 8) {
        kvarn_fwht_scalar(a, d, 1);
        float s = 1.0f / sqrtf((float)d);
        for (int i = 0; i < d; i++) a[i] *= s;
        return;
    }
    kvarn_fwht(a, d);
    __m256 sc = _mm256_set1_ps(1.0f / sqrtf((float)d));
    for (int i = 0; i < d; i += 8)
        _mm256_storeu_ps(a + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), sc));
}

#elif defined(__ARM_NEON)
#include <arm_neon.h>
static inline float32x4_t kvarn_fwht4(float32x4_t a) {
    const float32x4_t s1 = { 1, -1, 1, -1 };
    const float32x4_t s2 = { 1, 1, -1, -1 };
    float32x4_t t = vrev64q_f32(a);                      /* swap neighbours */
    a = vmlaq_f32(t, a, s1);
    t = vextq_f32(a, a, 2);                              /* swap pairs */
    return vmlaq_f32(t, a, s2);
}
static void kvarn_fwht(float *a, int d) {
    if (d < 4) { kvarn_fwht_scalar(a, d, 1); return; }
    for (int i = 0; i < d; i += 4)
        vst1q_f32(a + i, kvarn_fwht4(vld1q_f32(a + i)));
    for (int len = 4; len < d; len <<= 1)
        for (int i = 0; i < d; i += len << 1)
            for (int j = i; j < i + len; j += 4) {
                float32x4_t u = vld1q_f32(a + j), v = vld1q_f32(a + j + len);
                vst1q_f32(a + j, vaddq_f32(u, v));
                vst1q_f32(a + j + len, vsubq_f32(u, v));
            }
}
static inline void kvarn_rot(float *a, int d) {
    if (d < 4) {
        kvarn_fwht_scalar(a, d, 1);
        float s = 1.0f / sqrtf((float)d);
        for (int i = 0; i < d; i++) a[i] *= s;
        return;
    }
    kvarn_fwht(a, d);
    float32x4_t sc = vdupq_n_f32(1.0f / sqrtf((float)d));
    for (int i = 0; i < d; i += 4) vst1q_f32(a + i, vmulq_f32(vld1q_f32(a + i), sc));
}

#else
static inline void kvarn_fwht(float *a, int d) { kvarn_fwht_scalar(a, d, 1); }
static inline void kvarn_rot(float *a, int d) {
    kvarn_fwht_scalar(a, d, 1);
    float s = 1.0f / sqrtf((float)d);
    for (int i = 0; i < d; i++) a[i] *= s;
}
#endif

/* ============================================================================
 * bit packing — dense, row-major, `bits` per code
 *
 * Upstream packs 8/bits codes per byte, which forces bits into {1,2,4,8}. The
 * presets only ever ask for 4 and 2, so that would do, but the byte layout is ours
 * alone (nothing reads this store but us) and a dense packer costs nothing and keeps
 * the header a general codec the tests can sweep over 1..8.
 *
 * The code plane is token-major whichever way the tile was balanced: `group` rows
 * of `d` codes, each byte-aligned because kvarn_init requires d*bits to be a
 * multiple of 8. A K tile is not quantised that way round, but storage order is
 * free to differ from arithmetic order and it decides decode speed. Attention reads
 * one token at a time, and with the K plane stored channels-major that is a
 * stride-(group*bits/8) gather: one cache line per element, 512 of them to rebuild
 * a single d=512 vector. Token-major makes it one contiguous run of d*bits/8 bytes,
 * and encode pays a transpose once per 128 tokens instead.
 * ==========================================================================*/

/* Whole rows only, since that is all the codec ever needs. */
static void kvarn_pack_row(uint8_t *dst, const uint8_t *code, int n, int bits) {
    switch (bits) {
    case 8:
        memcpy(dst, code, (size_t)n);
        return;
    case 4:
        for (int i = 0, j = 0; i < n; i += 2, j++)
            dst[j] = (uint8_t)(code[i] | (code[i + 1] << 4));
        return;
    case 2:
        for (int i = 0, j = 0; i < n; i += 4, j++)
            dst[j] = (uint8_t)(code[i] | (code[i + 1] << 2) |
                               (code[i + 2] << 4) | (code[i + 3] << 6));
        return;
    case 1:
        for (int i = 0, j = 0; i < n; i += 8, j++) {
            uint8_t b = 0;
            for (int k = 0; k < 8; k++) b |= (uint8_t)(code[i + k] << k);
            dst[j] = b;
        }
        return;
    default: {                       /* 3, 5, 6, 7: codes straddle bytes */
        memset(dst, 0, ((size_t)n * bits + 7) / 8);
        for (int i = 0; i < n; i++) {
            size_t bit = (size_t)i * bits;
            for (int b = 0; b < bits; b++)
                if (code[i] & (1u << b))
                    dst[(bit + b) >> 3] |= (uint8_t)(1u << ((bit + b) & 7));
        }
        return;
    }
    }
}

static void kvarn_unpack_row(const uint8_t *src, uint8_t *code, int n, int bits) {
    switch (bits) {
    case 8:
        memcpy(code, src, (size_t)n);
        return;
    case 4:
        for (int i = 0, j = 0; i < n; i += 2, j++) {
            uint8_t b = src[j];
            code[i] = (uint8_t)(b & 0x0f);
            code[i + 1] = (uint8_t)(b >> 4);
        }
        return;
    case 2:
        for (int i = 0, j = 0; i < n; i += 4, j++) {
            uint8_t b = src[j];
            code[i]     = (uint8_t)(b & 3);
            code[i + 1] = (uint8_t)((b >> 2) & 3);
            code[i + 2] = (uint8_t)((b >> 4) & 3);
            code[i + 3] = (uint8_t)(b >> 6);
        }
        return;
    case 1:
        for (int i = 0, j = 0; i < n; i += 8, j++) {
            uint8_t b = src[j];
            for (int k = 0; k < 8; k++) code[i + k] = (uint8_t)((b >> k) & 1);
        }
        return;
    default: {
        uint32_t mask = (1u << bits) - 1;
        for (int i = 0; i < n; i++) {
            size_t bit = (size_t)i * bits;
            uint32_t v = (uint32_t)src[bit >> 3] >> (bit & 7);
            v |= (uint32_t)src[(bit >> 3) + 1] << (8 - (bit & 7));
            code[i] = (uint8_t)(v & mask);
        }
        return;
    }
    }
}

/* fp16 -> f32 for a whole scale plane.
 *
 * q40.h's converter is the right shape for a per-block scale read once per 32
 * weights. Here the per-channel plane is read once per element of every decoded
 * vector, so its branches land in the engine's hottest loop. F16C does eight at a
 * time and every -march=native x86 since Ivy Bridge has it; aarch64 NEON has
 * vcvt_f32_f16 unconditionally. The scalar fallback is branch-free for the same
 * reason. */
#if defined(__F16C__)
#include <immintrin.h>
static inline void kvarn_h2f_row(const uint16_t *h, float *f, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(f + i, _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)(h + i))));
    for (; i < n; i++) f[i] = q40_fp16_to_f32(h[i]);
}
#elif defined(__ARM_NEON) && defined(__aarch64__)
#include <arm_neon.h>
static inline void kvarn_h2f_row(const uint16_t *h, float *f, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4)
        vst1q_f32(f + i, vcvt_f32_f16(vreinterpret_f16_u16(vld1_u16(h + i))));
    for (; i < n; i++) f[i] = q40_fp16_to_f32(h[i]);
}
#else
/* Branch-free (the classic magic-number decode): shift the halfword into single
 * position, add the exponent bias difference, and fix inf/nan and subnormals with
 * arithmetic rather than a jump. */
static inline float kvarn_h2f(uint16_t h) {
    union { uint32_t u; float f; } o, magic = { 113u << 23 };
    const uint32_t shifted_exp = 0x7c00u << 13;
    o.u = (uint32_t)(h & 0x7fffu) << 13;
    uint32_t exp = shifted_exp & o.u;
    o.u += (uint32_t)(127 - 15) << 23;
    if (exp == shifted_exp) o.u += (uint32_t)(128 - 16) << 23;   /* inf/nan */
    else if (exp == 0) { o.u += 1u << 23; o.f -= magic.f; }      /* subnormal */
    o.u |= (uint32_t)(h & 0x8000u) << 16;
    return o.f;
}
static inline void kvarn_h2f_row(const uint16_t *h, float *f, int n) {
    for (int i = 0; i < n; i++) f[i] = kvarn_h2f(h[i]);
}
#endif

/* ============================================================================
 * the codec
 * ==========================================================================*/

typedef struct {
    int d;              /* head dim */
    int bits;           /* codes per element, 1..8 */
    int group;          /* tile width in tokens */
    int iters;          /* Sinkhorn iterations */
    int clip;           /* take the RTN range from percentiles, not from the extremes */
    int is_k;           /* 1: tile is [d, group]; 0: tile is [group, d] */
    int R, C;           /* tile rows and columns after orientation */
    size_t row_bytes;   /* one token's codes: d * bits / 8 */
    size_t qbytes;      /* packed payload bytes */
    size_t bytes;       /* whole tile record */
    size_t off_rs;      /* fp16[R] row scale, RTN scale folded in */
    size_t off_rz;      /* fp16[R] row zero point, likewise folded */
    size_t off_cs;      /* fp16[C] column scale */
    /* Encoder scratch. One tile is d*group floats, 256 KiB at d=512 and group=128,
     * far too much for the stack, and mallocing per tile in the write path is not
     * free either. Owned by the codec, so encoding one (layer, K/V) from two threads
     * at once is not allowed; the models write KV serially per layer. */
    float *work, *cur;
    float *srow, *scol;             /* the scales the balancer settled on */
    float *lrow, *lcol;             /* log scales, live */
    float *lrow_best, *lcol_best;   /* ... and at the lowest imbalance seen */
    float *brow, *bcol;             /* per-sweep reciprocal ratios */
    float *rlo, *rinv;              /* per-row RTN zero point and 1/scale */
    float *csum, *csq;              /* column accumulators, kept out of the sweep */
    uint8_t *code;                  /* one token's codes, pre-pack */
} Kvarn;

/* Bytes for one tile of one head. */
static inline size_t kvarn_tile_bytes(int d, int bits, int group, int is_k) {
    int R = is_k ? d : group, C = is_k ? group : d;
    return (size_t)d * bits / 8 * group + (size_t)(2 * R + C) * sizeof(uint16_t);
}

/* The f32 residual ring, in positions: `rwin` rounded up to whole tiles.
 *
 * A tile can only be sealed once all `group` of its tokens have been written, and
 * the tile about to be evicted has to sit in contiguous ring slots for the encoder
 * to walk it, so the ring must be a whole number of tiles. Rounding up is all that
 * is needed. Sealing a tile does not retire its f32 copies, since the reader prefers
 * the ring whenever the slot still holds the position it wants, so a position keeps
 * full precision for the whole W writes until its slot is reused, exactly as it did
 * under a per-vector codec.
 *
 * What tiling does cost is the confirmation margin. A tile sealed as position `pos`
 * arrives covers positions [pos-W, pos-W+group), so its youngest token is only
 * W-group+1 positions behind the write head, and a sealed tile cannot be taken back.
 * A caller that writes speculatively must therefore ask for a window at least
 * `group + ndraft` wide; see gemma4's floor. Without speculation, one tile is
 * enough. */
static inline int kvarn_window(int rwin, int group) {
    int t = (rwin + group - 1) / group;
    if (t < 1) t = 1;
    return t * group;
}

/* Tiles in a layer's packed ring.
 *
 * `cap` live positions span ceil(cap/group) tiles when they happen to start on a
 * tile boundary and one more when they do not, and for a sliding layer they never
 * do. Two spare tiles cover both cases with one line and cost a few KiB. */
static inline int kvarn_ntiles(int cap, int group) {
    return cap / group + 2;
}

static void kvarn_init(Kvarn *q, int d, int bits, int group, int is_k) {
    if (d < 2 || d > KVARN_MAXD || (d & (d - 1))) {
        fprintf(stderr, "kvarn: head_dim %d must be a power of two <= %d\n",
                d, KVARN_MAXD);
        exit(1);
    }
    if (group < 2 || group > KVARN_MAXGROUP) {
        fprintf(stderr, "kvarn: tile %d outside 2..%d\n", group, KVARN_MAXGROUP);
        exit(1);
    }
    if (bits < 1 || bits > KVARN_MAXBITS) {
        fprintf(stderr, "kvarn: %d bits outside 1..%d\n", bits, KVARN_MAXBITS);
        exit(1);
    }
    q->d = d;
    q->bits = bits;
    q->group = group;
    q->iters = KVARN_ITERS;
    q->clip = bits < KVARN_CLIP_BITS;
    q->is_k = is_k;
    q->R = is_k ? d : group;
    q->C = is_k ? group : d;
    if (((size_t)d * bits) % 8) {
        fprintf(stderr, "kvarn: %d codes of %d bits per token is not a whole "
                "number of bytes\n", d, bits);
        exit(1);
    }
    q->row_bytes = (size_t)d * bits / 8;
    q->qbytes = q->row_bytes * (size_t)group;
    q->off_rs = q->qbytes;
    q->off_rz = q->off_rs + (size_t)q->R * sizeof(uint16_t);
    q->off_cs = q->off_rz + (size_t)q->R * sizeof(uint16_t);
    q->bytes  = q->off_cs + (size_t)q->C * sizeof(uint16_t);

    size_t n = (size_t)q->R * q->C;
    q->work = malloc(sizeof(float) * n);
    q->cur  = malloc(sizeof(float) * n);
    q->srow = malloc(sizeof(float) * q->R);
    q->brow = malloc(sizeof(float) * q->R);
    q->lrow = malloc(sizeof(float) * q->R);
    q->scol = malloc(sizeof(float) * q->C);
    q->bcol = malloc(sizeof(float) * q->C);
    q->lcol = malloc(sizeof(float) * q->C);
    q->lrow_best = malloc(sizeof(float) * q->R);
    q->lcol_best = malloc(sizeof(float) * q->C);
    q->rlo  = malloc(sizeof(float) * q->R);
    q->rinv = malloc(sizeof(float) * q->R);
    q->csum = malloc(sizeof(float) * q->C);
    q->csq  = malloc(sizeof(float) * q->C);
    q->code = malloc((size_t)d);
    if (!q->work || !q->cur || !q->srow || !q->brow || !q->lrow ||
        !q->scol || !q->bcol || !q->lcol || !q->lrow_best || !q->lcol_best ||
        !q->rlo || !q->rinv || !q->csum || !q->csq || !q->code) {
        fprintf(stderr, "kvarn: OOM allocating a %dx%d tile\n", q->R, q->C);
        exit(1);
    }
}

static inline void kvarn_free(Kvarn *q) {
    free(q->work); free(q->cur);
    free(q->srow); free(q->brow); free(q->lrow);
    free(q->scol); free(q->bcol); free(q->lcol);
    free(q->lrow_best); free(q->lcol_best);
    free(q->rlo); free(q->rinv); free(q->csum); free(q->csq); free(q->code);
    memset(q, 0, sizeof *q);
}

/* ---------------------------------------------------------------------------
 * Sinkhorn variance normalisation
 * -------------------------------------------------------------------------*/

/* Sample standard deviations, Bessel-corrected to match the reference's torch.std.
 *
 * Both axes in one row-major sweep. Reducing down a column with a `for c { for r }`
 * loop walks memory with a 512-byte stride at C=128 and takes a cache miss per
 * element; carrying C accumulators keeps the traversal row-major and the
 * accumulators in L1.
 *
 * The row reduction carries four partial sums, for the reason head_dot in each
 * engine spells out: one accumulator makes it a serial chain of float adds and
 * bounds the loop at the latency of a single add. The combine is done in double,
 * which is where precision actually matters, since ss - s*s/n cancels. */
static inline float kvarn_var_to_std(double ss, double s, int n) {
    double var = (ss - s * s / n) / (n - 1);
    return var > 0 ? (float)sqrt(var) : 0.0f;
}

static inline float kvarn_row_std(const float *row, int C) {
    float s0 = 0, s1 = 0, s2 = 0, s3 = 0, q0 = 0, q1 = 0, q2 = 0, q3 = 0;
    int c = 0;
    for (; c + 4 <= C; c += 4) {
        float a = row[c], b = row[c + 1], e = row[c + 2], f = row[c + 3];
        s0 += a; s1 += b; s2 += e; s3 += f;
        q0 += a * a; q1 += b * b; q2 += e * e; q3 += f * f;
    }
    for (; c < C; c++) { s0 += row[c]; q0 += row[c] * row[c]; }
    return kvarn_var_to_std((double)q0 + q1 + q2 + q3, (double)s0 + s1 + s2 + s3, C);
}

static void kvarn_stds(const float *m, int R, int C, float *csd, float *rsd,
                       float *cs, float *cq) {
    for (int c = 0; c < C; c++) { cs[c] = 0.0f; cq[c] = 0.0f; }
    for (int r = 0; r < R; r++) {
        const float *row = m + (size_t)r * C;
        rsd[r] = kvarn_row_std(row, C);
        for (int c = 0; c < C; c++) { cs[c] += row[c]; cq[c] += row[c] * row[c]; }
    }
    for (int c = 0; c < C; c++) csd[c] = kvarn_var_to_std(cq[c], cs[c], R);
}

/* Upstream's objective: the spread of the column stds plus the spread of the row
 * stds. A perfectly balanced tile scores 2. The loop keeps the best iterate rather
 * than the last because alternating normalisation is not monotone: fixing the rows
 * can un-fix the columns, and a late iteration sometimes lands worse than an early
 * one. */
static double kvarn_spread(const float *csd, const float *rsd, int R, int C) {
    float cmax = csd[0], cmin = csd[0], rmax = rsd[0], rmin = rsd[0];
    for (int c = 1; c < C; c++) {
        if (csd[c] > cmax) cmax = csd[c];
        if (csd[c] < cmin) cmin = csd[c];
    }
    for (int r = 1; r < R; r++) {
        if (rsd[r] > rmax) rmax = rsd[r];
        if (rsd[r] < rmin) rmin = rsd[r];
    }
    return (double)cmax / (cmin > 1e-8f ? cmin : 1e-8f)
         + (double)rmax / (rmin > 1e-8f ? rmin : 1e-8f);
}

static inline float kvarn_clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* Balance q->work into q->cur, returning the scales that produced it:
 * cur[r][c] = work[r][c] / srow[r] / scol[c].
 *
 * Structurally this is upstream's loop with the passes fused. The reference
 * rebuilds the whole normalised tile from the original after each half-step and
 * then measures it, eight sweeps over R*C per iteration, of which six recompute
 * something the sweep before could have produced on the way past. Applying a scale
 * and measuring the result is one traversal rather than two; the column stds a step
 * opens with are the ones the previous step left behind; and scaling a whole row by
 * a constant scales that row's std by the same constant, so the row stds after the
 * second sweep are arithmetic rather than a third pass. That leaves two. */
static void kvarn_balance(Kvarn *q) {
    const float *m = q->work;
    float *cur = q->cur;
    int R = q->R, C = q->C;
    size_t n = (size_t)R * C;
    float *cs = q->csum, *cq = q->csq;
    float *csd = q->scol, *rsd = q->srow;   /* live stds of `cur` */

    for (int r = 0; r < R; r++) { q->lrow[r] = 0.0f; q->lrow_best[r] = 0.0f; }
    for (int c = 0; c < C; c++) { q->lcol[c] = 0.0f; q->lcol_best[c] = 0.0f; }
    memcpy(cur, m, sizeof(float) * n);

    /* the unnormalised tile is a candidate like any other */
    kvarn_stds(cur, R, C, csd, rsd, cs, cq);
    double best = kvarn_spread(csd, rsd, R, C);

    for (int it = 0; it < q->iters; it++) {
        /* Columns. The log clip makes the update non-multiplicative, so apply the
         * ratio the clipped log actually moved by rather than the raw std. The
         * reciprocal is hoisted: a divide in the inner loop costs several times the
         * multiply-add around it, and the last bit of a scale that is about to be
         * rounded to fp16 is not worth paying for. */
        for (int c = 0; c < C; c++) {
            float sd = kvarn_clampf(csd[c], KVARN_STD_MIN, KVARN_STD_MAX);
            float nl = kvarn_clampf(q->lcol[c] + logf(sd),
                                    KVARN_LOGS_MIN, KVARN_LOGS_MAX);
            q->bcol[c] = 1.0f / expf(nl - q->lcol[c]);   /* scratch: the ratio */
            q->lcol[c] = nl;
        }
        /* Fusing these two sweeps into one was tried and is about 10% slower. A
         * row's own ratio depends on nothing but that row, so in principle it could
         * all happen while the row is still in L1, but a whole tile is 256 KiB and
         * lives in L2 anyway: nothing was bandwidth-bound to begin with, and making
         * each row wait on its own standard deviation drains the pipeline between
         * the halves. Kept separate so the rows stay independent.
         *
         * sweep 1: apply the column ratios, measure the rows they leave behind */
        for (int r = 0; r < R; r++) {
            float *row = cur + (size_t)r * C;
            for (int c = 0; c < C; c++) row[c] *= q->bcol[c];
            rsd[r] = kvarn_row_std(row, C);
        }

        for (int r = 0; r < R; r++) {
            float sd = kvarn_clampf(rsd[r], KVARN_STD_MIN, KVARN_STD_MAX);
            float nl = kvarn_clampf(q->lrow[r] + logf(sd),
                                    KVARN_LOGS_MIN, KVARN_LOGS_MAX);
            q->brow[r] = 1.0f / expf(nl - q->lrow[r]);   /* scratch: the ratio */
            q->lrow[r] = nl;
        }
        /* sweep 2: apply the row ratios and accumulate the column stats of the
         * result, which are both this iterate's imbalance and the stds the next
         * iteration opens with. The row stds come along for free: scaling a row by
         * rr scales its std by rr. */
        for (int c = 0; c < C; c++) { cs[c] = 0.0f; cq[c] = 0.0f; }
        for (int r = 0; r < R; r++) {
            float *row = cur + (size_t)r * C;
            float rr = q->brow[r];
            for (int c = 0; c < C; c++) {
                float v = row[c] * rr;
                row[c] = v;
                cs[c] += v; cq[c] += v * v;
            }
            rsd[r] *= rr;
        }
        for (int c = 0; c < C; c++) csd[c] = kvarn_var_to_std(cq[c], cs[c], R);

        double imb = kvarn_spread(csd, rsd, R, C);
        if (imb <= best) {
            best = imb;
            memcpy(q->lrow_best, q->lrow, sizeof(float) * R);
            memcpy(q->lcol_best, q->lcol, sizeof(float) * C);
        }
    }

    /* Rebuild from the original at the best iterate. Doing this from `m` rather
     * than unwinding `cur` is what keeps the result independent of how many
     * iterations ran after the best one. */
    for (int r = 0; r < R; r++) q->srow[r] = expf(q->lrow_best[r]);
    for (int c = 0; c < C; c++) {
        q->scol[c] = expf(q->lcol_best[c]);
        q->bcol[c] = 1.0f / q->scol[c];
    }
    for (int r = 0; r < R; r++) {
        const float *src = m + (size_t)r * C;
        float *dst = cur + (size_t)r * C;
        float ir = 1.0f / q->srow[r];
        for (int c = 0; c < C; c++) dst[c] = src[c] * ir * q->bcol[c];
    }
}

/* Cache-blocked transpose, src[rows][cols] -> dst[cols][rows].
 *
 * The naive version walks one operand with a row-length stride, which on a
 * [512,128] f32 tile means touching 512 distinct cache lines per column and
 * streaming the whole 256 KiB past the cache once per column. A square block small
 * enough that both its source rows and its destination rows stay resident pays for
 * the tile once instead. */
static void kvarn_transpose(const float *src, float *dst, int rows, int cols) {
    const int B = 32;
    for (int r0 = 0; r0 < rows; r0 += B)
        for (int c0 = 0; c0 < cols; c0 += B) {
            int r1 = r0 + B < rows ? r0 + B : rows;
            int c1 = c0 + B < cols ? c0 + B : cols;
            for (int r = r0; r < r1; r++)
                for (int c = c0; c < c1; c++)
                    dst[(size_t)c * rows + r] = src[(size_t)r * cols + c];
        }
}

/* ---------------------------------------------------------------------------
 * the RTN range
 * -------------------------------------------------------------------------*/

/* Below 4 bits the row's range is taken from percentiles rather than from its
 * extremes, and the handful of values outside get clamped by the rounder.
 *
 * This is upstream's KVARN_RTN_QUANTILE, and at 2 bits it is not optional. Four
 * levels spread over the full min..max of a 128-sample row put roughly +/-3 sigma
 * between the outer two, so one loud coordinate costs the other 127 samples most of
 * their resolution, and after the balancing there is usually exactly one left:
 * Sinkhorn equalises variance, not kurtosis. Clamping it is the cheaper of the two
 * errors. At 4 bits and up there are enough
 * levels that the extremes are worth keeping exactly, so min/max stands.
 *
 * Upstream reaches for this on models whose K rows have max/std around 6, which is
 * exactly the shape a rotated-then-balanced tile has. */
/* The two order statistics torch.quantile(row, p) interpolates between.
 *
 * No sort: p*(n-1) is under 6 at both ends for every (row length, q) this ships
 * with, so a bounded insertion pass keeping the few smallest and few largest is
 * both simpler and faster than a quickselect, and it touches the row once. */
static void kvarn_row_range(const float *row, int n, float q, float *lo, float *hi) {
    float pl = q * (float)(n - 1), ph = (1.0f - q) * (float)(n - 1);
    int kl = (int)pl, kh = (int)ph;
    float fl = pl - (float)kl, fh = ph - (float)kh;
    int ml = kl + 2, mh = n - kh;               /* extremes needed from each end */

    if (ml > KVARN_CLIP_MAXK || mh > KVARN_CLIP_MAXK || n < 2) {
        float a = row[0], b = row[0];           /* defensive: never taken as shipped */
        for (int c = 1; c < n; c++) {
            if (row[c] < a) a = row[c];
            if (row[c] > b) b = row[c];
        }
        *lo = a; *hi = b;
        return;
    }

    float s[KVARN_CLIP_MAXK], G[KVARN_CLIP_MAXK];    /* ascending / descending */
    for (int i = 0; i < ml; i++) s[i] = INFINITY;
    for (int i = 0; i < mh; i++) G[i] = -INFINITY;
    for (int c = 0; c < n; c++) {
        float v = row[c];
        if (v < s[ml - 1]) {
            int i = ml - 1;
            while (i > 0 && s[i - 1] > v) { s[i] = s[i - 1]; i--; }
            s[i] = v;
        }
        if (v > G[mh - 1]) {
            int i = mh - 1;
            while (i > 0 && G[i - 1] < v) { G[i] = G[i - 1]; i--; }
            G[i] = v;
        }
    }
    /* G is descending, so G[j] is the (n-1-j)-th smallest. */
    *lo = s[kl] + fl * (s[kl + 1] - s[kl]);
    *hi = G[n - 1 - kh] + fh * (G[n - 2 - kh] - G[n - 1 - kh]);
    if (*hi < *lo) { float t = *lo; *lo = *hi; *hi = t; }
}

/* ---------------------------------------------------------------------------
 * encode / decode
 * -------------------------------------------------------------------------*/

/* Quantise one tile of one head.
 *
 * `src` is the tile in the natural token-major layout: token i's d floats live at
 * src + i*stride, which is how both models hold a KV ring (stride = n_kv_heads*d).
 * `ntok` may be short of `group`; the missing tokens are filled by repeating the
 * last real one so the tile statistics stay defined, and their codes are never
 * read back. `dst` must have room for q->bytes. */
static void kvarn_encode_tile(Kvarn *q, const float *src, size_t stride, int ntok,
                              uint8_t *dst) {
    int d = q->d, g = q->group, R = q->R, C = q->C, bits = q->bits;

    /* Rotate every token into a token-major staging plane, then transpose once if
     * the balancer wants channels down the rows. Scattering each rotated vector
     * straight into [d, group] would be the same cache disaster the store layout
     * avoids, 128 times over. `cur` is free until kvarn_balance overwrites it. */
    float *stage = q->is_k ? q->cur : q->work;
    if (ntok > g) ntok = g;
    for (int t = 0; t < g; t++) {
        float *dstrow = stage + (size_t)t * d;
        if (ntok <= 0) {
            memset(dstrow, 0, sizeof(float) * d);
        } else {
            memcpy(dstrow, src + (size_t)(t < ntok ? t : ntok - 1) * stride,
                   sizeof(float) * d);
            kvarn_rot(dstrow, d);
        }
    }
    if (q->is_k) kvarn_transpose(q->cur, q->work, g, d);   /* -> [d, group] */

    kvarn_balance(q);

    int qmax = (1 << bits) - 1;
    uint16_t *rs = (uint16_t *)(dst + q->off_rs);
    uint16_t *rz = (uint16_t *)(dst + q->off_rz);
    uint16_t *cs = (uint16_t *)(dst + q->off_cs);

    /* Per-row RTN parameters first, then one pass that emits whole tokens. */
    for (int r = 0; r < R; r++) {
        const float *row = q->cur + (size_t)r * C;
        float lo, hi;
        if (q->clip) {
            kvarn_row_range(row, C, KVARN_CLIP_Q, &lo, &hi);
        } else {
            lo = hi = row[0];
            for (int c = 1; c < C; c++) {
                if (row[c] < lo) lo = row[c];
                if (row[c] > hi) hi = row[c];
            }
        }
        float scale = (hi - lo) / (float)qmax;
        if (!(scale > 1e-10f)) scale = 1e-10f;
        q->rlo[r] = lo;
        q->rinv[r] = 1.0f / scale;
        /* Fold the row's Sinkhorn scale into the quantiser's own, so the decoder
         * needs one multiply and one add per element rather than three. */
        rs[r] = q40_f32_to_fp16(q->srow[r] * scale);
        rz[r] = q40_f32_to_fp16(q->srow[r] * lo);
    }
    for (int c = 0; c < C; c++) cs[c] = q40_f32_to_fp16(q->scol[c]);

    /* The store is token-major whichever way the tile was balanced, so a K tile is
     * transposed here. Blocked, not column-by-column: walking one column of a
     * [512,128] f32 tile touches 512 distinct cache lines, and doing that 128 times
     * streams the whole 256 KiB tile past the cache once per token. A 32x32 block
     * pays for the tile once. `work` is free to scribble on -- the balancer has
     * already consumed it -- so the transpose needs no extra memory. */
    const float *plane = q->cur;
    if (q->is_k) {
        kvarn_transpose(q->cur, q->work, d, g);   /* -> [group, d], token-major */
        plane = q->work;
    }

    uint8_t *code = q->code;
    for (int t = 0; t < g; t++) {
        const float *row = plane + (size_t)t * d;
        if (q->is_k) {
            /* per-channel parameters, which are now the contiguous axis */
            for (int i = 0; i < d; i++) {
                float e = roundf((row[i] - q->rlo[i]) * q->rinv[i]);
                code[i] = (uint8_t)(e < 0.0f ? 0 : (e > (float)qmax ? qmax : (int)e));
            }
        } else {
            float lo = q->rlo[t], inv = q->rinv[t];
            for (int i = 0; i < d; i++) {
                float e = roundf((row[i] - lo) * inv);
                code[i] = (uint8_t)(e < 0.0f ? 0 : (e > (float)qmax ? qmax : (int)e));
            }
        }
        kvarn_pack_row(dst + (size_t)t * q->row_bytes, code, d, bits);
    }
}

/* A tile's per-channel scale plane is the same for all `group` of its tokens, so
 * converting it inside the decode does `group` times the work the tile needs, and
 * touches a second region of the record tens of KiB from the codes on every
 * position. A caller walking positions in order holds one of these instead and
 * pays for the conversion once per tile. Zero it, and kvarn_decode_raw refills it
 * whenever the tile changes. */
typedef struct {
    const uint8_t *rec;         /* the record a and b were built from */
    float a[KVARN_MAXD];        /* K: per-channel scale;  V: per-channel column scale */
    float b[KVARN_MAXD];        /* K: per-channel zero;   V: unused */
} KvarnPlanes;

/* Recover token `tok` of a tile into x[d], in the rotated frame, H not applied.
 *
 * One contiguous run of codes either way; the packing note above says why the K
 * plane is stored transposed relative to the frame it was quantised in. All that is
 * left of the orientation here is which scale plane is indexed by the token and
 * which by the channel: K carries a scale and a zero per channel and one scale for
 * the whole token, V the other way round.
 *
 * Callers that read a lot of positions should stay in this frame and pay for H once
 * rather than once per position; kvarn_decode_vec is the convenience wrapper that
 * does the transform, and the engines deliberately do not use it in attention. H is
 * orthonormal and symmetric, so q.k = (Hq).(Hk) and sum_t w_t v_t = H(sum_t w_t Hv_t):
 * rotate the query once, accumulate in the rotated frame, transform the output once.
 * At 32K that turns a d log d transform per cached position into one per token. */
static void kvarn_decode_raw(const Kvarn *q, const uint8_t *src, int tok,
                             KvarnPlanes *p, float *x) {
    int d = q->d;
    const uint16_t *rs = (const uint16_t *)(src + q->off_rs);
    const uint16_t *rz = (const uint16_t *)(src + q->off_rz);
    const uint16_t *cs = (const uint16_t *)(src + q->off_cs);
    uint8_t code[KVARN_MAXD];

    if (p->rec != src) {                  /* first token of a new tile */
        p->rec = src;
        if (q->is_k) {
            kvarn_h2f_row(rs, p->a, d);
            kvarn_h2f_row(rz, p->b, d);
        } else {
            kvarn_h2f_row(cs, p->a, d);
        }
    }
    kvarn_unpack_row(src + (size_t)tok * q->row_bytes, code, d, q->bits);

    if (q->is_k) {
        float col = q40_fp16_to_f32(cs[tok]);
        const float *a = p->a, *b = p->b;
        for (int i = 0; i < d; i++) x[i] = ((float)code[i] * a[i] + b[i]) * col;
    } else {
        float sc = q40_fp16_to_f32(rs[tok]), zp = q40_fp16_to_f32(rz[tok]);
        const float *a = p->a;
        for (int i = 0; i < d; i++) x[i] = ((float)code[i] * sc + zp) * a[i];
    }
}

/* The same, brought back to the model's frame. H is its own inverse. */
static inline void kvarn_decode_vec(const Kvarn *q, const uint8_t *src, int tok,
                                    float *x) {
    KvarnPlanes p;
    p.rec = NULL;
    kvarn_decode_raw(q, src, tok, &p, x);
    kvarn_rot(x, q->d);
}

#endif /* KVARN_H */
