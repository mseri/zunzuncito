/* q40.h — Q4_0 block-scale int4 for colibrì (fmt=4).
 *
 * WHY a fourth format. colibrì's native int4 (fmt=2) uses ONE f32 scale per row.
 * Gemma-4's QAT checkpoints are trained against Q4_0: 32-weight blocks, each with
 * its own fp16 scale. Re-blocking those weights onto per-row scales throws away
 * exactly the structure QAT optimised for — you'd pay for QAT and get RTN quality.
 * So we carry the block scales through to the kernel instead.
 *
 * LAYOUT (bit-identical to llama.cpp's block_q4_0, so a GGUF's expert tensors can
 * be memcpy'd straight into the container):
 *
 *   block = 32 weights = 18 bytes
 *     [0..1]   fp16 d          (scale)
 *     [2..17]  16 bytes qs     nibbles, q in [0,15]; w_j = d * (q_j - 8)
 *              qs[j] low nibble  -> weight j        (j = 0..15)
 *              qs[j] high nibble -> weight j + 16
 *
 * A row of I weights is I/32 such blocks, laid out contiguously; a tensor [O,I]
 * is O such rows. I MUST be a multiple of 32 (Gemma-4's hidden/inter dims are).
 * Unlike fmt=2 there is no separate `.qs` scale tensor: the scales are interleaved
 * in the weight bytes. That is a feature for a streaming engine — the scales ride
 * along in the SAME pread as the weights, so an expert load is ONE contiguous read
 * instead of weights + a second seek for the scale array.
 *
 * ACTIVATIONS are quantised Q8_0-style, also in blocks of 32 (one f32 scale per
 * block, held separately), matching the weight blocking so the dot decomposes as
 *   sum_blocks  d_w[b] * d_x[b] * <q_w[b] - 8, q_x[b]>
 * and the inner product is pure integer.
 *
 * The (q-8) offset is never materialised: with u8 weights and i8 activations,
 *   <q-8, x> = <q, x> - 8 * sum(x)
 * and <q,x> is exactly what AVX2 _mm256_maddubs_epi16 computes (u8 x i8 -> i16).
 */
#ifndef COLI_Q40_H
#define COLI_Q40_H

#include <stdint.h>
#include <string.h>
#include <math.h>

#define Q40_BLK 32
#define Q40_BLK_BYTES 18   /* 2 (fp16 d) + 16 (nibbles) */

/* ---- fp16 <-> f32. Scalar, no F16C dependency: used only on scales (I/32 per
 * row), never in the inner loop, so a table-free bit-twiddle is plenty. ---- */
static inline float q40_fp16_to_f32(uint16_t h){
    uint32_t s=(uint32_t)(h>>15)&1u, e=(uint32_t)(h>>10)&0x1fu, m=(uint32_t)h&0x3ffu;
    uint32_t bits;
    if(e==0){
        if(m==0) bits=s<<31;                       /* +/- 0 */
        else{                                      /* subnormal -> normalise */
            e=127-15+1;
            while(!(m&0x400u)){ m<<=1; e--; }
            m&=0x3ffu; bits=(s<<31)|(e<<23)|(m<<13);
        }
    } else if(e==31) bits=(s<<31)|0x7f800000u|(m<<13);   /* inf / nan */
    else bits=(s<<31)|((e-15+127)<<23)|(m<<13);
    float f; memcpy(&f,&bits,4); return f;
}
static inline uint16_t q40_f32_to_fp16(float f){
    uint32_t bits; memcpy(&bits,&f,4);
    uint32_t s=(bits>>31)&1u; int32_t e=(int32_t)((bits>>23)&0xffu)-127+15;
    uint32_t m=bits&0x7fffffu;
    if(((bits>>23)&0xffu)==0xffu) return (uint16_t)((s<<15)|0x7c00u|(m?0x200u:0u));
    if(e>=31) return (uint16_t)((s<<15)|0x7c00u);           /* overflow -> inf */
    if(e<=0){                                               /* subnormal / zero */
        if(e<-10) return (uint16_t)(s<<15);
        m|=0x800000u;
        uint32_t sh=(uint32_t)(14-e);
        uint32_t half=m>>sh, rem=m&((1u<<sh)-1u), tie=1u<<(sh-1);
        if(rem>tie || (rem==tie && (half&1u))) half++;       /* round-half-even */
        return (uint16_t)((s<<15)|half);
    }
    uint32_t half=m>>13, rem=m&0x1fffu;
    if(rem>0x1000u || (rem==0x1000u && (half&1u))){
        half++;
        if(half==0x400u){ half=0; e++; if(e>=31) return (uint16_t)((s<<15)|0x7c00u); }
    }
    return (uint16_t)((s<<15)|((uint32_t)e<<10)|half);
}

/* ---- quantise one f32 row of I weights into I/32 q4_0 blocks ---- */
static inline void q40_quant_row(const float *w, uint8_t *dst, int I){
    for(int b=0;b<I/Q40_BLK;b++){
        const float *x=w+b*Q40_BLK;
        uint8_t *blk=dst+(size_t)b*Q40_BLK_BYTES;
        /* q4_0 is a SYMMETRIC codebook centred on 8: the representable range is
         * [-8d, +7d]. Scale off the max-|x| element, signed, exactly as llama.cpp
         * does — using amax/7 instead would clip the negative tail. */
        float amax=0.f, mx=0.f;
        for(int j=0;j<Q40_BLK;j++){ float a=fabsf(x[j]); if(a>amax){ amax=a; mx=x[j]; } }
        /* Round d to fp16 BEFORE deriving id. The decoder only ever sees the fp16
         * value; quantising against the unrounded f32 d (as llama.cpp does) means
         * encoder and decoder disagree by the fp16 rounding, which leaks straight
         * into the weight error and pushes it just past the d/2 bound. Rounding
         * first makes the two agree bit-for-bit and restores the bound. */
        uint16_t hd=q40_f32_to_fp16(mx / -8.0f); memcpy(blk,&hd,2);
        float d = q40_fp16_to_f32(hd);
        float id = d ? 1.0f/d : 0.0f;
        for(int j=0;j<16;j++){
            float v0=x[j]*id + 8.5f, v1=x[j+16]*id + 8.5f;
            int q0=(int)v0, q1=(int)v1;
            if(q0<0) q0=0;
            if(q0>15) q0=15;
            if(q1<0) q1=0;
            if(q1>15) q1=15;
            blk[2+j]=(uint8_t)(q0 | (q1<<4));
        }
    }
}
static inline void q40_dequant_row(const uint8_t *src, float *w, int I){
    for(int b=0;b<I/Q40_BLK;b++){
        const uint8_t *blk=src+(size_t)b*Q40_BLK_BYTES;
        uint16_t hd; memcpy(&hd,blk,2); float d=q40_fp16_to_f32(hd);
        float *o=w+b*Q40_BLK;
        for(int j=0;j<16;j++){
            o[j]    = d * (float)((blk[2+j] & 0x0f) - 8);
            o[j+16] = d * (float)((blk[2+j] >> 4)   - 8);
        }
    }
}

/* ---- Q8_0-style activation quantisation: blocks of 32, one f32 scale each ---- */
static inline void q40_quant_act(const float *x, int8_t *xq, float *sx, int I){
    for(int b=0;b<I/Q40_BLK;b++){
        const float *v=x+b*Q40_BLK; int8_t *o=xq+b*Q40_BLK;
        float amax=0.f;
        for(int j=0;j<Q40_BLK;j++){ float a=fabsf(v[j]); if(a>amax) amax=a; }
        float d=amax/127.0f, id=d?1.0f/d:0.0f;
        sx[b]=d;
        for(int j=0;j<Q40_BLK;j++){
            int q=(int)lrintf(v[j]*id);
            if(q<-127) q=-127;
            if(q>127) q=127;
            o[j]=(int8_t)q;
        }
    }
}

/* ================= inner product: one q4_0 row . one q8_0 activation ========= */

#if defined(__AVX2__)
#include <immintrin.h>
static inline int q40_hsum_i32(__m256i v){
    __m128i lo=_mm256_castsi256_si128(v), hi=_mm256_extracti128_si256(v,1);
    __m128i s=_mm_add_epi32(lo,hi);
    s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0x4e));
    s=_mm_add_epi32(s,_mm_shuffle_epi32(s,0xb1));
    return _mm_cvtsi128_si32(s);
}
/* One block's contribution, folded into `acc`. Split out so the dot below can feed
 * FOUR independent accumulators. There is exactly one fmadd per block, so a single
 * accumulator makes the loop a serial chain of 4-cycle FMA latencies -- ~6.4
 * measured cycles/block, most of it the chain. Four chains in flight cover that
 * latency: 1.13x / 1.21x / 1.12x at 1 / 2 / 4 threads, best-of-7 on a 2.1 GHz Xeon.
 * q8_0 gains much less (1.03-1.07x): at 34 bytes/block it is already near bandwidth,
 * where q4_0 at 18 has arithmetic to spare.
 *
 * It is the chain and not the op count. Hoisting the redundant per-row sum(x) out of
 * the loop, ~15% of the inner-loop uops, is worth 1.01x. The arithmetic here is
 * unchanged and the split sum is, if anything, slightly better conditioned. */
static inline __m256 q40_blk_acc(__m256 acc, const uint8_t *w, const int8_t *xq,
                                 const float *sx, int b,
                                 __m256i lomask, __m256i ones16){
    const uint8_t *blk=w+(size_t)b*Q40_BLK_BYTES;
    uint16_t hd; memcpy(&hd,blk,2);
    float d=q40_fp16_to_f32(hd)*sx[b];

    /* 16 packed bytes -> 32 nibbles. Low nibbles are weights 0..15, high
     * nibbles are weights 16..31: broadcast the 16 bytes into both lanes and
     * mask, which puts w[0..15] in lane 0 and w[16..31] in lane 1 — exactly
     * the order the activation bytes already sit in. */
    __m128i raw=_mm_loadu_si128((const __m128i*)(blk+2));
    __m256i both=_mm256_set_m128i(_mm_srli_epi16(raw,4), raw);
    __m256i q=_mm256_and_si256(both,lomask);           /* u8, 0..15 */

    __m256i x=_mm256_loadu_si256((const __m256i*)(xq+(size_t)b*Q40_BLK));

    /* <q,x> with u8 x i8; and sum(x) to apply the -8 offset without
     * materialising q-8 (maddubs needs the u8 operand first). */
    __m256i p =_mm256_maddubs_epi16(q,x);              /* i16, may saturate? no:
                                                         15*127*2 = 3810 < 32767 */
    __m256i sx8=_mm256_maddubs_epi16(_mm256_set1_epi8(1),x);   /* sum pairs of x */
    __m256i acc_i=_mm256_sub_epi32(_mm256_madd_epi16(p,ones16),
                   _mm256_slli_epi32(_mm256_madd_epi16(sx8,ones16),3)); /* -8*sum(x) */
    return _mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(acc_i),acc);
}
static inline float q40_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I){
    __m256 a0=_mm256_setzero_ps(), a1=a0, a2=a0, a3=a0;
    const __m256i lomask=_mm256_set1_epi8(0x0f);
    const __m256i ones16=_mm256_set1_epi16(1);
    int nb=I/Q40_BLK, b=0;
    for(; b+3<nb; b+=4){
        a0=q40_blk_acc(a0,w,xq,sx,b+0,lomask,ones16);
        a1=q40_blk_acc(a1,w,xq,sx,b+1,lomask,ones16);
        a2=q40_blk_acc(a2,w,xq,sx,b+2,lomask,ones16);
        a3=q40_blk_acc(a3,w,xq,sx,b+3,lomask,ones16);
    }
    for(; b<nb; b++) a0=q40_blk_acc(a0,w,xq,sx,b,lomask,ones16);
    /* horizontal sum of the f32 accumulators */
    __m256 acc=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
    __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
    __m128 s=_mm_add_ps(lo,hi);
    s=_mm_add_ps(s,_mm_movehl_ps(s,s));
    s=_mm_add_ss(s,_mm_shuffle_ps(s,s,1));
    return _mm_cvtss_f32(s);
}
#elif defined(__ARM_NEON)
#include <arm_neon.h>
/* Apple silicon path. M1 and later implement ARMv8.2 dotprod (vdotq_s32), which
 * does exactly what maddubs+madd does on AVX2 in one instruction; the generic NEON
 * fallback below uses widening multiply-accumulate for older cores.
 * Unlike AVX2 we can subtract the 8 offset directly (NEON multiplies signed x
 * signed), so no sum(x) correction term is needed. */
/* One block's contribution, folded into a VECTOR accumulator. Two departures from the
 * obvious loop, both about the dependency chain rather than the arithmetic: the
 * cross-lane vaddvq_s32 lives outside the loop (accumulate lane-wise, reduce once at
 * the end), and the caller keeps FOUR accumulators so the per-block FMA latency is
 * covered instead of forming one serial chain.
 *
 * NEON carries both problems -- the serial chain AVX2 has, plus a cross-lane
 * reduction per block -- so it should gain at least what AVX2's 1.13-1.21x did. But
 * that is UNMEASURED: no arm64 hardware was available, only a cross-compile that
 * checks it builds and emits sdot. test_q40 covers correctness, not speed. */
static inline float32x4_t q40_blk_acc(float32x4_t acc, const uint8_t *w,
                                      const int8_t *xq, const float *sx, int b,
                                      uint8x16_t lomask, int8x16_t eight){
    const uint8_t *blk=w+(size_t)b*Q40_BLK_BYTES;
    uint16_t hd; memcpy(&hd,blk,2);
    float d=q40_fp16_to_f32(hd)*sx[b];

    uint8x16_t raw = vld1q_u8(blk+2);                  /* 16 bytes = 32 nibbles */
    /* low nibbles are weights 0..15, high nibbles are weights 16..31 --
     * matching the order the activation bytes already sit in. */
    int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, lomask)), eight);
    int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);

    int8x16_t x0 = vld1q_s8(xq + (size_t)b*Q40_BLK);
    int8x16_t x1 = vld1q_s8(xq + (size_t)b*Q40_BLK + 16);

#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t s = vdotq_s32(vdupq_n_s32(0), q0, x0);
    s = vdotq_s32(s, q1, x1);
#else
    int16x8_t p0 = vmull_s8(vget_low_s8(q0),  vget_low_s8(x0));
    p0 = vmlal_s8(p0, vget_high_s8(q0), vget_high_s8(x0));
    int16x8_t p1 = vmull_s8(vget_low_s8(q1),  vget_low_s8(x1));
    p1 = vmlal_s8(p1, vget_high_s8(q1), vget_high_s8(x1));
    int32x4_t s = vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1));
#endif
    return vmlaq_n_f32(acc, vcvtq_f32_s32(s), d);
}
static inline float q40_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I){
    float32x4_t a0=vdupq_n_f32(0.0f), a1=a0, a2=a0, a3=a0;
    const uint8x16_t lomask = vdupq_n_u8(0x0f);
    const int8x16_t eight = vdupq_n_s8(8);
    int nb=I/Q40_BLK, b=0;
    for(; b+3<nb; b+=4){
        a0=q40_blk_acc(a0,w,xq,sx,b+0,lomask,eight);
        a1=q40_blk_acc(a1,w,xq,sx,b+1,lomask,eight);
        a2=q40_blk_acc(a2,w,xq,sx,b+2,lomask,eight);
        a3=q40_blk_acc(a3,w,xq,sx,b+3,lomask,eight);
    }
    for(; b<nb; b++) a0=q40_blk_acc(a0,w,xq,sx,b,lomask,eight);
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1), vaddq_f32(a2,a3)));
}
#else
/* Four accumulators here too: the scalar path has the same serial-FP-add chain as
 * the SIMD ones, and this costs nothing to spell out. */
static inline float q40_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I){
    float a[4]={0.f,0.f,0.f,0.f};
    for(int b=0;b<I/Q40_BLK;b++){
        const uint8_t *blk=w+(size_t)b*Q40_BLK_BYTES;
        uint16_t hd; memcpy(&hd,blk,2);
        const int8_t *x=xq+(size_t)b*Q40_BLK;
        int32_t s=0;
        for(int j=0;j<16;j++){
            s += ((int)(blk[2+j]&0x0f)-8)*(int)x[j];
            s += ((int)(blk[2+j]>>4)  -8)*(int)x[j+16];
        }
        a[b&3] += q40_fp16_to_f32(hd)*sx[b]*(float)s;
    }
    return (a[0]+a[1])+(a[2]+a[3]);
}
#endif

/* EXACT-ACTIVATION reference dot: q4_0 weights, f32 activations. Not used in the
 * hot path -- it exists so the engine can be built with COLI_F32ACT to separate
 * "int8 activation error" from "bug" when validating against the numpy oracle. */
static inline float q40_dot_f32(const uint8_t *w, const float *x, int I){
    double acc=0;
    for(int b=0;b<I/Q40_BLK;b++){
        const uint8_t *blk=w+(size_t)b*Q40_BLK_BYTES;
        uint16_t hd; memcpy(&hd,blk,2);
        float d=q40_fp16_to_f32(hd);
        const float *v=x+b*Q40_BLK;
        double s=0;
        for(int j=0;j<16;j++){
            s += (double)((int)(blk[2+j]&0x0f)-8)*v[j];
            s += (double)((int)(blk[2+j]>>4)  -8)*v[j+16];
        }
        acc += (double)d*s;
    }
    return (float)acc;
}

/* row-major [O,I] q4_0 matvec: y[o] = <W[o,:], x> */
static inline void q40_matvec(float *y, const uint8_t *W, const int8_t *xq,
                              const float *sx, int O, int I){
    size_t rb=(size_t)(I/Q40_BLK)*Q40_BLK_BYTES;
    for(int o=0;o<O;o++) y[o]=q40_dot(W+(size_t)o*rb, xq, sx, I);
}
static inline int64_t q40_row_bytes(int I){ return (int64_t)(I/Q40_BLK)*Q40_BLK_BYTES; }
static inline int64_t q40_tensor_bytes(int O,int I){ return (int64_t)O*q40_row_bytes(I); }

/* ============================================================================
 * Q8_0 WEIGHTS (fmt=8) — the higher-precision tier for the apex-quant gradient.
 *
 * Bit-identical to llama.cpp's block_q8_0: 32 weights per block, one fp16 scale,
 * 32 signed int8 codes. w_j = d * q_j, q in [-127, 127].
 *
 *   block = 32 weights = 34 bytes
 *     [0..1]   fp16 d          (scale)
 *     [2..33]  32 int8 qs      w_j = d * q_j
 *
 * Same streaming property as q4_0: the scale rides inline, one contiguous read.
 * Activations are the SAME Q8_0-style block quantisation (q40_quant_act), so the
 * dot decomposes as sum_blocks d_w[b] * d_x[b] * <q_w[b], q_x[b]> (int8 x int8).
 * ========================================================================== */
#define Q80_BLK_BYTES 34   /* 2 (fp16 d) + 32 (int8) */

static inline void q80_quant_row(const float *w, uint8_t *dst, int I){
    for(int b=0;b<I/Q40_BLK;b++){
        const float *x=w+b*Q40_BLK;
        uint8_t *blk=dst+(size_t)b*Q80_BLK_BYTES;
        float amax=0.f;
        for(int j=0;j<Q40_BLK;j++){ float a=fabsf(x[j]); if(a>amax) amax=a; }
        uint16_t hd=q40_f32_to_fp16(amax/127.0f); memcpy(blk,&hd,2);
        float d=q40_fp16_to_f32(hd), id=d?1.0f/d:0.0f;
        int8_t *o=(int8_t*)(blk+2);
        for(int j=0;j<Q40_BLK;j++){
            int q=(int)lrintf(x[j]*id);
            if(q<-127) q=-127;
            if(q>127) q=127;
            o[j]=(int8_t)q;
        }
    }
}
static inline void q80_dequant_row(const uint8_t *src, float *w, int I){
    for(int b=0;b<I/Q40_BLK;b++){
        const uint8_t *blk=src+(size_t)b*Q80_BLK_BYTES;
        uint16_t hd; memcpy(&hd,blk,2); float d=q40_fp16_to_f32(hd);
        const int8_t *q=(const int8_t*)(blk+2); float *o=w+b*Q40_BLK;
        for(int j=0;j<Q40_BLK;j++) o[j]=d*(float)q[j];
    }
}

#if defined(__AVX2__)
static inline __m256 q80_blk_acc(__m256 acc, const uint8_t *w, const int8_t *xq,
                                 const float *sx, int b, __m256i ones16){
    const uint8_t *blk=w+(size_t)b*Q80_BLK_BYTES;
    uint16_t hd; memcpy(&hd,blk,2);
    float d=q40_fp16_to_f32(hd)*sx[b];
    __m256i a=_mm256_loadu_si256((const __m256i*)(blk+2));
    __m256i x=_mm256_loadu_si256((const __m256i*)(xq+(size_t)b*Q40_BLK));
    /* signed x signed via the sign trick: maddubs needs an unsigned operand */
    __m256i ax=_mm256_sign_epi8(a,a);          /* |a| (unsigned magnitudes) */
    __m256i sx8=_mm256_sign_epi8(x,a);         /* x carrying sign of a */
    __m256i p=_mm256_maddubs_epi16(ax,sx8);    /* i16 pairs */
    __m256i s=_mm256_madd_epi16(p,ones16);     /* i32 */
    return _mm256_fmadd_ps(_mm256_set1_ps(d),_mm256_cvtepi32_ps(s),acc);
}
/* four accumulators, same reasoning as q40_dot above */
static inline float q80_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I){
    __m256 a0=_mm256_setzero_ps(), a1=a0, a2=a0, a3=a0;
    const __m256i ones16=_mm256_set1_epi16(1);
    int nb=I/Q40_BLK, b=0;
    for(; b+3<nb; b+=4){
        a0=q80_blk_acc(a0,w,xq,sx,b+0,ones16);
        a1=q80_blk_acc(a1,w,xq,sx,b+1,ones16);
        a2=q80_blk_acc(a2,w,xq,sx,b+2,ones16);
        a3=q80_blk_acc(a3,w,xq,sx,b+3,ones16);
    }
    for(; b<nb; b++) a0=q80_blk_acc(a0,w,xq,sx,b,ones16);
    __m256 acc=_mm256_add_ps(_mm256_add_ps(a0,a1),_mm256_add_ps(a2,a3));
    __m128 lo=_mm256_castps256_ps128(acc), hi=_mm256_extractf128_ps(acc,1);
    __m128 s=_mm_add_ps(lo,hi);
    s=_mm_add_ps(s,_mm_movehl_ps(s,s));
    s=_mm_add_ss(s,_mm_shuffle_ps(s,s,1));
    return _mm_cvtss_f32(s);
}
#elif defined(__ARM_NEON)
static inline float32x4_t q80_blk_acc(float32x4_t acc, const uint8_t *w,
                                      const int8_t *xq, const float *sx, int b){
    const uint8_t *blk=w+(size_t)b*Q80_BLK_BYTES;
    uint16_t hd; memcpy(&hd,blk,2);
    float d=q40_fp16_to_f32(hd)*sx[b];
    int8x16_t a0=vld1q_s8((const int8_t*)(blk+2));
    int8x16_t a1=vld1q_s8((const int8_t*)(blk+2+16));
    int8x16_t x0=vld1q_s8(xq+(size_t)b*Q40_BLK);
    int8x16_t x1=vld1q_s8(xq+(size_t)b*Q40_BLK+16);
#if defined(__ARM_FEATURE_DOTPROD)
    int32x4_t s=vdotq_s32(vdupq_n_s32(0),a0,x0);
    s=vdotq_s32(s,a1,x1);
#else
    int16x8_t p0=vmull_s8(vget_low_s8(a0),vget_low_s8(x0));
    p0=vmlal_s8(p0,vget_high_s8(a0),vget_high_s8(x0));
    int16x8_t p1=vmull_s8(vget_low_s8(a1),vget_low_s8(x1));
    p1=vmlal_s8(p1,vget_high_s8(a1),vget_high_s8(x1));
    int32x4_t s=vaddq_s32(vpaddlq_s16(p0),vpaddlq_s16(p1));
#endif
    return vmlaq_n_f32(acc, vcvtq_f32_s32(s), d);
}
/* four accumulators, vaddvq hoisted out -- same reasoning as q40_dot above */
static inline float q80_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I){
    float32x4_t a0=vdupq_n_f32(0.0f), a1=a0, a2=a0, a3=a0;
    int nb=I/Q40_BLK, b=0;
    for(; b+3<nb; b+=4){
        a0=q80_blk_acc(a0,w,xq,sx,b+0);
        a1=q80_blk_acc(a1,w,xq,sx,b+1);
        a2=q80_blk_acc(a2,w,xq,sx,b+2);
        a3=q80_blk_acc(a3,w,xq,sx,b+3);
    }
    for(; b<nb; b++) a0=q80_blk_acc(a0,w,xq,sx,b);
    return vaddvq_f32(vaddq_f32(vaddq_f32(a0,a1), vaddq_f32(a2,a3)));
}
#else
static inline float q80_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I){
    float acc[4]={0.f,0.f,0.f,0.f};
    for(int b=0;b<I/Q40_BLK;b++){
        const uint8_t *blk=w+(size_t)b*Q80_BLK_BYTES;
        uint16_t hd; memcpy(&hd,blk,2);
        const int8_t *a=(const int8_t*)(blk+2);
        const int8_t *x=xq+(size_t)b*Q40_BLK;
        int32_t s=0;
        for(int j=0;j<Q40_BLK;j++) s+=(int)a[j]*(int)x[j];
        acc[b&3]+=q40_fp16_to_f32(hd)*sx[b]*(float)s;
    }
    return (acc[0]+acc[1])+(acc[2]+acc[3]);
}
#endif

static inline float q80_dot_f32(const uint8_t *w, const float *x, int I){
    double acc=0;
    for(int b=0;b<I/Q40_BLK;b++){
        const uint8_t *blk=w+(size_t)b*Q80_BLK_BYTES;
        uint16_t hd; memcpy(&hd,blk,2); float d=q40_fp16_to_f32(hd);
        const int8_t *a=(const int8_t*)(blk+2);
        const float *v=x+b*Q40_BLK;
        double s=0;
        for(int j=0;j<Q40_BLK;j++) s+=(double)a[j]*v[j];
        acc+=(double)d*s;
    }
    return (float)acc;
}
static inline void q80_matvec(float *y, const uint8_t *W, const int8_t *xq,
                              const float *sx, int O, int I){
    size_t rb=(size_t)(I/Q40_BLK)*Q80_BLK_BYTES;
    for(int o=0;o<O;o++) y[o]=q80_dot(W+(size_t)o*rb, xq, sx, I);
}
static inline int64_t q80_row_bytes(int I){ return (int64_t)(I/Q40_BLK)*Q80_BLK_BYTES; }
static inline int64_t q80_tensor_bytes(int O,int I){ return (int64_t)O*q80_row_bytes(I); }

#endif /* COLI_Q40_H */
