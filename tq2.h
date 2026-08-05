/* tq2.h — the two Maple weight formats: ternary (tq2) and 4-bit affine (q4a).
 *
 * Maple ships a genuinely ternary checkpoint. Every matrix in the model, the four
 * attention projections and all three expert projections, is stored as 2-bit codes
 * with ONE scale per output row, and the codes only ever take the values 0, 1, 2:
 *
 *     w_j = alpha_row * (code_j - 1)        code in {0,1,2}, so w in {-a, 0, +a}
 *
 * That is not a quantisation we chose, it is what was trained (checked: across the
 * whole checkpoint no code 3 occurs). So unlike the q4_0/q8_0 tiers in q40.h there
 * is no fidelity dial here and nothing to tune -- the format is the checkpoint, and
 * a converter that re-blocked these onto q4_0 would spend 2.25x the bytes to
 * represent strictly less.
 *
 * Two consequences shape this file.
 *
 * 1. The scale leaves the inner loop. q4_0 and q8_0 carry an fp16 scale per 32
 *    weights, so their kernels do a per-block fp16 decode and an FMA. Here the scale
 *    is per row, so the whole row reduces to ONE integer accumulation and one
 *    multiply at the end. Consequently the layout is per TENSOR rather than per row:
 *
 *        alpha[O]         f32
 *        codes[O][I/4]    u8, row stride exactly I/4
 *
 *    Splitting them keeps every code row 4-byte aligned at a power-of-two-ish stride
 *    (512 B at I=2048, 128 B at I=512) instead of q4_0's awkward 18-bytes-per-block
 *    interleave, and it costs nothing for streaming: an expert is still one
 *    contiguous pread, alphas and codes together.
 *
 * 2. The -1 offset is free. With u8 codes and i8 activations,
 *
 *        <c - 1, x> = <c, x> - sum(x)
 *
 *    and sum(x) does not depend on the row -- or on the tensor, or on the expert.
 *    Over a whole matvec it is a single scalar. So the caller quantises the
 *    activation once with tq2_quant_act(), which also returns
 *    Sx = sum_b sx[b] * sum_j xq[b][j], and every row of every tensor consuming that
 *    activation subtracts alpha*Sx at the end. q4_0 pays the equivalent correction
 *    (its -8) per block, per row; here it is paid once per vector.
 *
 *    On NEON the offset is cheaper still: sdot multiplies signed by signed, so the
 *    codes are turned into {-1,0,+1} with one vsubq_s8 and Sx is not needed at all.
 *    Both routes are exact integer arithmetic on the same operands, so the two ISAs
 *    agree bit for bit; only AVX2 and the scalar fallback read Sx.
 *
 * PACKING. Codes are packed 4 per byte in groups of 64 weights = 16 bytes, where
 * byte k of a group holds weights k, k+16, k+32, k+48 at bit positions 0, 2, 4, 6.
 * That is the ternary analogue of q4_0's low-nibble/high-nibble split, chosen so one
 * 16-byte vector load yields four vectors that are already in the order the
 * activation bytes sit in -- no shuffles on either ISA. A group of 64 spans exactly
 * two 32-weight activation blocks, so I must be a multiple of 64.
 *
 * Q4A is the second format, and it exists only because Maple's embedding table, its
 * lm_head and the FlashHead centroids are NOT ternary: they are 4-bit affine with a
 * scale AND a bias per group of 64,
 *
 *     w_j = d_g * q_j + m_g                 q in [0,15]
 *
 * Re-quantising those onto q4_0 (symmetric, blocks of 32) would throw away the zero
 * point for no saving -- 36 bytes per 64 weights either way -- so we carry the
 * checkpoint's own format through to the kernel instead, exactly as q40.h argues for
 * Gemma-4's QAT blocks. The bias term needs a per-block sum(xq), which unlike the
 * ternary case does not factor out of the row (it is weighted by m_g), but it is two
 * extra instructions per block and the q4_0 kernel already computes the same
 * quantity for its -8.
 *
 * Both formats consume the q8_0-style int8 activations from q40.h, in blocks of 32.
 */
#ifndef COLI_TQ2_H
#define COLI_TQ2_H

#include <stdint.h>
#include <string.h>
#include <math.h>
#include "q40.h"

/* ============================================================================
 * TQ2 — ternary, one f32 scale per output row
 * ==========================================================================*/

#define TQ2_GRP 64          /* weights per packing group (16 bytes) */

static inline int64_t tq2_row_bytes(int I) { return (int64_t)I / 4; }
/* alphas first, then the code rows */
static inline int64_t tq2_tensor_bytes(int O, int I) {
    return (int64_t)O * 4 + (int64_t)O * tq2_row_bytes(I);
}
static inline const float *tq2_alphas(const uint8_t *t) { return (const float *)t; }
static inline const uint8_t *tq2_codes(const uint8_t *t, int O) {
    return t + (size_t)O * 4;
}

/* Quantise an activation vector for a ternary matvec.
 *
 * Identical to q40_quant_act (blocks of 32, one f32 scale each) plus the one scalar
 * the -1 offset needs. Returning it here rather than recomputing it per row is the
 * whole point: it turns an O*I-sized correction into an I-sized one. */
static inline float tq2_quant_act(const float *x, int8_t *xq, float *sx, int I) {
    q40_quant_act(x, xq, sx, I);
    double tot = 0;
    for (int b = 0; b < I / Q40_BLK; b++) {
        int32_t s = 0;
        const int8_t *q = xq + (size_t)b * Q40_BLK;
        for (int j = 0; j < Q40_BLK; j++) s += q[j];
        tot += (double)sx[b] * (double)s;
    }
    return (float)tot;
}

#if defined(__AVX2__)
#include <immintrin.h>
/* One 64-weight group, folded into two integer accumulators (one per activation
 * block, since each block carries its own f32 scale and they cannot be summed in
 * int). Codes stay unsigned so maddubs applies; the -1 is settled once by the
 * caller via sxtot. Saturation is not a risk: 2*127*2 = 508 against i16. */
static inline void tq2_grp_acc(__m256 *acc, const uint8_t *codes, const int8_t *xq,
                               const float *sx, int g, __m256i m3, __m256i ones16) {
    __m128i raw = _mm_loadu_si128((const __m128i *)(codes + (size_t)g * 16));
    /* byte k -> weights k, k+16, k+32, k+48; pairing (0,1) and (2,3) puts each
     * 32-weight half in the order the activation bytes already sit in. */
    __m256i lo = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 2), raw), m3);
    __m256i hi = _mm256_and_si256(
        _mm256_set_m128i(_mm_srli_epi16(raw, 6), _mm_srli_epi16(raw, 4)), m3);

    const int8_t *x = xq + (size_t)g * TQ2_GRP;
    __m256i x0 = _mm256_loadu_si256((const __m256i *)x);
    __m256i x1 = _mm256_loadu_si256((const __m256i *)(x + 32));

    __m256i d0 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, x0), ones16);
    __m256i d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, x1), ones16);
    *acc = _mm256_fmadd_ps(_mm256_set1_ps(sx[2 * g]), _mm256_cvtepi32_ps(d0), *acc);
    *acc = _mm256_fmadd_ps(_mm256_set1_ps(sx[2 * g + 1]), _mm256_cvtepi32_ps(d1), *acc);
}
static inline float tq2_dot(const uint8_t *codes, float alpha, const int8_t *xq,
                            const float *sx, float sxtot, int I) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    const __m256i m3 = _mm256_set1_epi8(0x03);
    const __m256i ones16 = _mm256_set1_epi16(1);
    int ng = I / TQ2_GRP, g = 0;
    /* four independent accumulators, for the reason q40_blk_acc documents: one
     * chain of FMA latencies is what bounds this loop, not the op count. */
    for (; g + 3 < ng; g += 4) {
        tq2_grp_acc(&a0, codes, xq, sx, g + 0, m3, ones16);
        tq2_grp_acc(&a1, codes, xq, sx, g + 1, m3, ones16);
        tq2_grp_acc(&a2, codes, xq, sx, g + 2, m3, ones16);
        tq2_grp_acc(&a3, codes, xq, sx, g + 3, m3, ones16);
    }
    for (; g < ng; g++) tq2_grp_acc(&a0, codes, xq, sx, g, m3, ones16);
    __m256 acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
    __m128 l = _mm256_castps256_ps128(acc), h = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(l, h);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return alpha * (_mm_cvtss_f32(s) - sxtot);
}

#elif defined(__ARM_NEON)
#include <arm_neon.h>
/* NEON subtracts the 1 into the codes and lets sdot do signed x signed, so sxtot is
 * unused on this path. The integer products are identical to the AVX2 route's
 * <c,x> - sum(x), so the two ISAs return the same float. */
static inline void tq2_grp_acc(float32x4_t *acc, const uint8_t *codes,
                               const int8_t *xq, const float *sx, int g,
                               uint8x16_t m3, int8x16_t one) {
    uint8x16_t raw = vld1q_u8(codes + (size_t)g * 16);
    int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m3)), one);
    int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 2), m3)), one);
    int8x16_t q2 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(raw, 4), m3)), one);
    int8x16_t q3 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 6)), one);

    const int8_t *x = xq + (size_t)g * TQ2_GRP;
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t s0 = vdotq_s32(vdupq_n_s32(0), q0, vld1q_s8(x));
    s0 = vdotq_s32(s0, q1, vld1q_s8(x + 16));
    int32x4_t s1 = vdotq_s32(vdupq_n_s32(0), q2, vld1q_s8(x + 32));
    s1 = vdotq_s32(s1, q3, vld1q_s8(x + 48));
#else
    int8x16_t x0 = vld1q_s8(x), x1 = vld1q_s8(x + 16);
    int8x16_t x2 = vld1q_s8(x + 32), x3 = vld1q_s8(x + 48);
    int16x8_t p0 = vmull_s8(vget_low_s8(q0), vget_low_s8(x0));
    p0 = vmlal_s8(p0, vget_high_s8(q0), vget_high_s8(x0));
    int16x8_t p1 = vmull_s8(vget_low_s8(q1), vget_low_s8(x1));
    p1 = vmlal_s8(p1, vget_high_s8(q1), vget_high_s8(x1));
    int16x8_t p2 = vmull_s8(vget_low_s8(q2), vget_low_s8(x2));
    p2 = vmlal_s8(p2, vget_high_s8(q2), vget_high_s8(x2));
    int16x8_t p3 = vmull_s8(vget_low_s8(q3), vget_low_s8(x3));
    p3 = vmlal_s8(p3, vget_high_s8(q3), vget_high_s8(x3));
    int32x4_t s0 = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
    int32x4_t s1 = vaddq_s32(vpaddlq_s16(p2), vpaddlq_s16(p3));
#endif
    *acc = vmlaq_n_f32(*acc, vcvtq_f32_s32(s0), sx[2 * g]);
    *acc = vmlaq_n_f32(*acc, vcvtq_f32_s32(s1), sx[2 * g + 1]);
}
static inline float tq2_dot(const uint8_t *codes, float alpha, const int8_t *xq,
                            const float *sx, float sxtot, int I) {
    (void)sxtot;                       /* folded into the codes on this path */
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0, a2 = a0, a3 = a0;
    const uint8x16_t m3 = vdupq_n_u8(0x03);
    const int8x16_t one = vdupq_n_s8(1);
    int ng = I / TQ2_GRP, g = 0;
    for (; g + 3 < ng; g += 4) {
        tq2_grp_acc(&a0, codes, xq, sx, g + 0, m3, one);
        tq2_grp_acc(&a1, codes, xq, sx, g + 1, m3, one);
        tq2_grp_acc(&a2, codes, xq, sx, g + 2, m3, one);
        tq2_grp_acc(&a3, codes, xq, sx, g + 3, m3, one);
    }
    for (; g < ng; g++) tq2_grp_acc(&a0, codes, xq, sx, g, m3, one);
    return alpha * vaddvq_f32(vaddq_f32(vaddq_f32(a0, a1), vaddq_f32(a2, a3)));
}

#else
static inline float tq2_dot(const uint8_t *codes, float alpha, const int8_t *xq,
                            const float *sx, float sxtot, int I) {
    float a[4] = {0.f, 0.f, 0.f, 0.f};
    int ng = I / TQ2_GRP;
    for (int g = 0; g < ng; g++) {
        const uint8_t *c = codes + (size_t)g * 16;
        const int8_t *x = xq + (size_t)g * TQ2_GRP;
        int32_t s0 = 0, s1 = 0;
        for (int k = 0; k < 16; k++) {
            s0 += (int)(c[k] & 3) * (int)x[k];
            s0 += (int)((c[k] >> 2) & 3) * (int)x[k + 16];
            s1 += (int)((c[k] >> 4) & 3) * (int)x[k + 32];
            s1 += (int)(c[k] >> 6) * (int)x[k + 48];
        }
        a[(2 * g) & 3] += sx[2 * g] * (float)s0;
        a[(2 * g + 1) & 3] += sx[2 * g + 1] * (float)s1;
    }
    return alpha * (((a[0] + a[1]) + (a[2] + a[3])) - sxtot);
}
#endif

/* Exact-activation reference (COLI_F32ACT), so the engine can be validated against
 * the numpy oracle without the int8 activation error in the way. */
static inline float tq2_dot_f32(const uint8_t *codes, float alpha, const float *x,
                                int I) {
    double acc = 0;
    int ng = I / TQ2_GRP;
    for (int g = 0; g < ng; g++) {
        const uint8_t *c = codes + (size_t)g * 16;
        const float *v = x + (size_t)g * TQ2_GRP;
        double s = 0;
        for (int k = 0; k < 16; k++) {
            s += (double)((int)(c[k] & 3) - 1) * v[k];
            s += (double)((int)((c[k] >> 2) & 3) - 1) * v[k + 16];
            s += (double)((int)((c[k] >> 4) & 3) - 1) * v[k + 32];
            s += (double)((int)(c[k] >> 6) - 1) * v[k + 48];
        }
        acc += s;
    }
    return (float)((double)alpha * acc);
}

static inline void tq2_dequant_row(const uint8_t *codes, float alpha, float *w,
                                   int I) {
    for (int g = 0; g < I / TQ2_GRP; g++) {
        const uint8_t *c = codes + (size_t)g * 16;
        float *v = w + (size_t)g * TQ2_GRP;
        for (int k = 0; k < 16; k++) {
            v[k]      = alpha * (float)((int)(c[k] & 3) - 1);
            v[k + 16] = alpha * (float)((int)((c[k] >> 2) & 3) - 1);
            v[k + 32] = alpha * (float)((int)((c[k] >> 4) & 3) - 1);
            v[k + 48] = alpha * (float)((int)(c[k] >> 6) - 1);
        }
    }
}

/* ============================================================================
 * Q4A — 4-bit affine, one fp16 scale AND one fp16 bias per group of 64
 *
 *   group = 64 weights = 36 bytes
 *     [0..1]    fp16 d      (scale)
 *     [2..3]    fp16 m      (bias)
 *     [4..19]   16 bytes    block 0: low nibble -> w[j], high -> w[j+16]
 *     [20..35]  16 bytes    block 1: low nibble -> w[j+32], high -> w[j+48]
 *
 *   w_j = d * q_j + m,  q in [0,15]
 *
 * A row of I weights is I/64 groups laid out contiguously, so unlike tq2 there is
 * nothing to split out: this format is only ever used for resident dense tensors
 * (embeddings, lm_head, FlashHead centroids), never for streamed experts.
 * ==========================================================================*/

#define Q4A_GRP 64
#define Q4A_GRP_BYTES 36

static inline int64_t q4a_row_bytes(int I) {
    return (int64_t)(I / Q4A_GRP) * Q4A_GRP_BYTES;
}
static inline int64_t q4a_tensor_bytes(int O, int I) {
    return (int64_t)O * q4a_row_bytes(I);
}

/* The lm_head is the widest tensor in the model (151936 x 2048) and, with the
 * FlashHead off, the single biggest cost of a decode step, so this gets the same
 * SIMD treatment q40_dot has. The bias term needs sum(xq) per block; that is one
 * extra maddubs against a vector of ones, exactly what q40_blk_acc already computes
 * to apply its -8. */
#if defined(__AVX2__)
static inline void q4a_blk_acc(__m256 *acc, const uint8_t *nib, const int8_t *v,
                               float sd, float sm, __m256i lomask, __m256i ones16) {
    __m128i raw = _mm_loadu_si128((const __m128i *)nib);
    /* low nibbles are weights 0..15, high nibbles 16..31 -- the order the
     * activation bytes already sit in. */
    __m256i q = _mm256_and_si256(_mm256_set_m128i(_mm_srli_epi16(raw, 4), raw), lomask);
    __m256i x = _mm256_loadu_si256((const __m256i *)v);
    __m256i dp = _mm256_madd_epi16(_mm256_maddubs_epi16(q, x), ones16);
    __m256i sv = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_set1_epi8(1), x), ones16);
    *acc = _mm256_fmadd_ps(_mm256_set1_ps(sd), _mm256_cvtepi32_ps(dp), *acc);
    *acc = _mm256_fmadd_ps(_mm256_set1_ps(sm), _mm256_cvtepi32_ps(sv), *acc);
}
static inline float q4a_dot(const uint8_t *w, const int8_t *xq, const float *sx,
                            int I) {
    __m256 a0 = _mm256_setzero_ps(), a1 = a0;
    const __m256i lomask = _mm256_set1_epi8(0x0f);
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int g = 0; g < I / Q4A_GRP; g++) {
        const uint8_t *grp = w + (size_t)g * Q4A_GRP_BYTES;
        uint16_t hd, hm;
        memcpy(&hd, grp, 2);
        memcpy(&hm, grp + 2, 2);
        float d = q40_fp16_to_f32(hd), m = q40_fp16_to_f32(hm);
        const int8_t *x = xq + (size_t)g * Q4A_GRP;
        float s0 = sx[2 * g], s1 = sx[2 * g + 1];
        q4a_blk_acc(&a0, grp + 4, x, s0 * d, s0 * m, lomask, ones16);
        q4a_blk_acc(&a1, grp + 20, x + 32, s1 * d, s1 * m, lomask, ones16);
    }
    __m256 acc = _mm256_add_ps(a0, a1);
    __m128 l = _mm256_castps256_ps128(acc), h = _mm256_extractf128_ps(acc, 1);
    __m128 s = _mm_add_ps(l, h);
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 1));
    return _mm_cvtss_f32(s);
}
#elif defined(__ARM_NEON)
static inline void q4a_blk_acc(float32x4_t *acc, const uint8_t *nib, const int8_t *v,
                               float sd, float sm, uint8x16_t lomask, int8x16_t one) {
    uint8x16_t raw = vld1q_u8(nib);
    int8x16_t q0 = vreinterpretq_s8_u8(vandq_u8(raw, lomask));
    int8x16_t q1 = vreinterpretq_s8_u8(vshrq_n_u8(raw, 4));
    int8x16_t x0 = vld1q_s8(v), x1 = vld1q_s8(v + 16);
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t dp = vdotq_s32(vdupq_n_s32(0), q0, x0);
    dp = vdotq_s32(dp, q1, x1);
    int32x4_t sv = vdotq_s32(vdupq_n_s32(0), one, x0);
    sv = vdotq_s32(sv, one, x1);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(q0), vget_low_s8(x0));
    p0 = vmlal_s8(p0, vget_high_s8(q0), vget_high_s8(x0));
    int16x8_t p1 = vmull_s8(vget_low_s8(q1), vget_low_s8(x1));
    p1 = vmlal_s8(p1, vget_high_s8(q1), vget_high_s8(x1));
    int32x4_t dp = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
    int32x4_t sv = vaddq_s32(vpaddlq_s16(vaddl_s8(vget_low_s8(x0), vget_high_s8(x0))),
                             vpaddlq_s16(vaddl_s8(vget_low_s8(x1), vget_high_s8(x1))));
#endif
    *acc = vmlaq_n_f32(*acc, vcvtq_f32_s32(dp), sd);
    *acc = vmlaq_n_f32(*acc, vcvtq_f32_s32(sv), sm);
}
static inline float q4a_dot(const uint8_t *w, const int8_t *xq, const float *sx,
                            int I) {
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = a0;
    const uint8x16_t lomask = vdupq_n_u8(0x0f);
    const int8x16_t one = vdupq_n_s8(1);
    for (int g = 0; g < I / Q4A_GRP; g++) {
        const uint8_t *grp = w + (size_t)g * Q4A_GRP_BYTES;
        uint16_t hd, hm;
        memcpy(&hd, grp, 2);
        memcpy(&hm, grp + 2, 2);
        float d = q40_fp16_to_f32(hd), m = q40_fp16_to_f32(hm);
        const int8_t *x = xq + (size_t)g * Q4A_GRP;
        float s0 = sx[2 * g], s1 = sx[2 * g + 1];
        q4a_blk_acc(&a0, grp + 4, x, s0 * d, s0 * m, lomask, one);
        q4a_blk_acc(&a1, grp + 20, x + 32, s1 * d, s1 * m, lomask, one);
    }
    return vaddvq_f32(vaddq_f32(a0, a1));
}
#else
static inline float q4a_dot(const uint8_t *w, const int8_t *xq, const float *sx,
                            int I) {
    double acc = 0;
    for (int g = 0; g < I / Q4A_GRP; g++) {
        const uint8_t *grp = w + (size_t)g * Q4A_GRP_BYTES;
        uint16_t hd, hm;
        memcpy(&hd, grp, 2);
        memcpy(&hm, grp + 2, 2);
        float d = q40_fp16_to_f32(hd), m = q40_fp16_to_f32(hm);
        const int8_t *x = xq + (size_t)g * Q4A_GRP;
        /* two activation blocks per group, each with its own f32 scale */
        for (int h = 0; h < 2; h++) {
            const uint8_t *nib = grp + 4 + h * 16;
            const int8_t *v = x + h * 32;
            int32_t dp = 0, sv = 0;
            for (int j = 0; j < 16; j++) {
                dp += (int)(nib[j] & 0x0f) * (int)v[j];
                dp += (int)(nib[j] >> 4) * (int)v[j + 16];
                sv += (int)v[j] + (int)v[j + 16];
            }
            acc += (double)sx[2 * g + h] * ((double)d * dp + (double)m * sv);
        }
    }
    return (float)acc;
}
#endif

static inline float q4a_dot_f32(const uint8_t *w, const float *x, int I) {
    double acc = 0;
    for (int g = 0; g < I / Q4A_GRP; g++) {
        const uint8_t *grp = w + (size_t)g * Q4A_GRP_BYTES;
        uint16_t hd, hm;
        memcpy(&hd, grp, 2);
        memcpy(&hm, grp + 2, 2);
        double d = q40_fp16_to_f32(hd), m = q40_fp16_to_f32(hm);
        const float *v = x + (size_t)g * Q4A_GRP;
        for (int h = 0; h < 2; h++) {
            const uint8_t *nib = grp + 4 + h * 16;
            const float *u = v + h * 32;
            double s = 0, t = 0;
            for (int j = 0; j < 16; j++) {
                s += (double)(nib[j] & 0x0f) * u[j];
                s += (double)(nib[j] >> 4) * u[j + 16];
                t += (double)u[j] + (double)u[j + 16];
            }
            acc += d * s + m * t;
        }
    }
    return (float)acc;
}

static inline void q4a_dequant_row(const uint8_t *w, float *o, int I) {
    for (int g = 0; g < I / Q4A_GRP; g++) {
        const uint8_t *grp = w + (size_t)g * Q4A_GRP_BYTES;
        uint16_t hd, hm;
        memcpy(&hd, grp, 2);
        memcpy(&hm, grp + 2, 2);
        float d = q40_fp16_to_f32(hd), m = q40_fp16_to_f32(hm);
        float *v = o + (size_t)g * Q4A_GRP;
        for (int h = 0; h < 2; h++) {
            const uint8_t *nib = grp + 4 + h * 16;
            float *u = v + h * 32;
            for (int j = 0; j < 16; j++) {
                u[j]      = d * (float)(nib[j] & 0x0f) + m;
                u[j + 16] = d * (float)(nib[j] >> 4) + m;
            }
        }
    }
}

#endif /* COLI_TQ2_H */
