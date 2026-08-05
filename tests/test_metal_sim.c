/* Transliteration of the Metal kernel body, lane-for-lane, to validate the SHADER
 * LOGIC (nibble order, unaligned fp16 scale assembly, 32-lane strided reduction)
 * on a machine with no Metal. Any mismatch here is a real shader bug. */
#include "q40.h"
#include "tq2.h"
#include <stdio.h>
#include <stdlib.h>
#define TG 32
/* maple.c's FMT_TQ2 / FMT_Q4A, local copies so this test need not include the engine */
#define FMT_TQ2_T 3
#define FMT_Q4A_T 4

static float half_from_bytes(unsigned char b0, unsigned char b1){
    unsigned short bits = (unsigned short)b0 | ((unsigned short)b1 << 8);
    return q40_fp16_to_f32(bits);              /* as_type<half>(bits) */
}

static void sim_kernel(float *Y, const uint8_t *W, const float *X,
                       unsigned O, unsigned I, unsigned S){
    unsigned nb = I / 32u;
    unsigned long rb = (unsigned long)nb * 18u;
    for (unsigned s = 0; s < S; s++)
      for (unsigned row = 0; row < O; row++) {
        float part[TG];
        for (unsigned lane = 0; lane < TG; lane++) {   /* one threadgroup */
            const uint8_t *w = W + (unsigned long)row * rb;
            const float *x = X + (unsigned long)s * I;
            float acc = 0.0f;
            for (unsigned b = lane; b < nb; b += TG) {
                const uint8_t *blk = w + (unsigned long)b * 18u;
                float d = half_from_bytes(blk[0], blk[1]);
                const float *xv = x + (unsigned long)b * 32u;
                float s0 = 0.0f;
                for (unsigned j = 0; j < 16u; ++j) {
                    unsigned char q = blk[2u + j];
                    s0 += (float)((int)(q & 0x0F) - 8) * xv[j];
                    s0 += (float)((int)(q >> 4)   - 8) * xv[j + 16u];
                }
                acc += d * s0;
            }
            part[lane] = acc;
        }
        for (unsigned off = TG/2u; off > 0u; off >>= 1u)
            for (unsigned lane = 0; lane < off; lane++) part[lane] += part[lane+off];
        Y[(unsigned long)s * O + row] = part[0];
      }
}

/* Same, for the q8_0 kernel: 34-byte blocks, fp16 scale, 32 signed int8 codes. */
static void sim_kernel8(float *Y, const uint8_t *W, const float *X,
                        unsigned O, unsigned I, unsigned S){
    unsigned nb = I / 32u;
    unsigned long rb = (unsigned long)nb * 34u;
    for (unsigned s = 0; s < S; s++)
      for (unsigned row = 0; row < O; row++) {
        float part[TG];
        for (unsigned lane = 0; lane < TG; lane++) {
            const uint8_t *w = W + (unsigned long)row * rb;
            const float *x = X + (unsigned long)s * I;
            float acc = 0.0f;
            for (unsigned b = lane; b < nb; b += TG) {
                const uint8_t *blk = w + (unsigned long)b * 34u;
                float d = half_from_bytes(blk[0], blk[1]);
                const float *xv = x + (unsigned long)b * 32u;
                float s0 = 0.0f;
                for (unsigned j = 0; j < 32u; ++j)
                    s0 += (float)(int)(signed char)blk[2u + j] * xv[j];
                acc += d * s0;
            }
            part[lane] = acc;
        }
        for (unsigned off = TG/2u; off > 0u; off >>= 1u)
            for (unsigned lane = 0; lane < off; lane++) part[lane] += part[lane+off];
        Y[(unsigned long)s * O + row] = part[0];
      }
}

/* Same, for maple's tq2 kernel: alpha[O] f32 at the front of the tensor, then code
 * rows of I/4 bytes, groups of 64 weights = 16 bytes with byte k holding weights
 * k, k+16, k+32, k+48 at bit positions 0, 2, 4, 6. */
static void sim_kernel_tq2(float *Y, const uint8_t *W, const float *X,
                           unsigned O, unsigned I, unsigned S){
    unsigned ng = I / 64u;
    const float *alpha = (const float *)W;
    for (unsigned s = 0; s < S; s++)
      for (unsigned row = 0; row < O; row++) {
        float part[TG];
        for (unsigned lane = 0; lane < TG; lane++) {
            const uint8_t *w = W + (unsigned long)O * 4u
                                 + (unsigned long)row * (unsigned long)(I / 4u);
            const float *x = X + (unsigned long)s * I;
            float acc = 0.0f;
            for (unsigned g = lane; g < ng; g += TG) {
                const uint8_t *c = w + (unsigned long)g * 16u;
                const float *xv = x + (unsigned long)g * 64u;
                float s0 = 0.0f;
                for (unsigned k = 0; k < 16u; ++k) {
                    unsigned char b = c[k];
                    s0 += (float)((int)( b       & 3) - 1) * xv[k];
                    s0 += (float)((int)((b >> 2) & 3) - 1) * xv[k + 16u];
                    s0 += (float)((int)((b >> 4) & 3) - 1) * xv[k + 32u];
                    s0 += (float)((int)( b >> 6)      - 1) * xv[k + 48u];
                }
                acc += s0;
            }
            part[lane] = acc;
        }
        for (unsigned off = TG/2u; off > 0u; off >>= 1u)
            for (unsigned lane = 0; lane < off; lane++) part[lane] += part[lane+off];
        Y[(unsigned long)s * O + row] = alpha[row] * part[0];
      }
}

/* Same, for maple's q4a kernel: 36-byte groups of 64 weights, fp16 scale AND fp16
 * bias, then two 16-byte nibble blocks. */
static void sim_kernel_q4a(float *Y, const uint8_t *W, const float *X,
                           unsigned O, unsigned I, unsigned S){
    unsigned ng = I / 64u;
    unsigned long rb = (unsigned long)ng * 36u;
    for (unsigned s = 0; s < S; s++)
      for (unsigned row = 0; row < O; row++) {
        float part[TG];
        for (unsigned lane = 0; lane < TG; lane++) {
            const uint8_t *w = W + (unsigned long)row * rb;
            const float *x = X + (unsigned long)s * I;
            float acc = 0.0f;
            for (unsigned g = lane; g < ng; g += TG) {
                const uint8_t *grp = w + (unsigned long)g * 36u;
                float d = half_from_bytes(grp[0], grp[1]);
                float m = half_from_bytes(grp[2], grp[3]);
                const float *xv = x + (unsigned long)g * 64u;
                float dp = 0.0f, sv = 0.0f;
                for (unsigned h = 0; h < 2u; ++h) {
                    const uint8_t *nib = grp + 4u + h * 16u;
                    const float *u = xv + h * 32u;
                    for (unsigned j = 0; j < 16u; ++j) {
                        unsigned char q = nib[j];
                        dp += (float)(q & 0x0F) * u[j];
                        dp += (float)(q >>   4) * u[j + 16u];
                        sv += u[j] + u[j + 16u];
                    }
                }
                acc += d * dp + m * sv;
            }
            part[lane] = acc;
        }
        for (unsigned off = TG/2u; off > 0u; off >>= 1u)
            for (unsigned lane = 0; lane < off; lane++) part[lane] += part[lane+off];
        Y[(unsigned long)s * O + row] = part[0];
      }
}

/* maple's shapes. The tensors are built already packed rather than quantised from
 * floats: tq2/q4a are the checkpoint's own formats, so there is no packer in the C
 * (only in tools/convert_maple.py), and the reference to diff against is the CPU
 * kernel reading the same bytes. */
static int check_maple(void){
    struct { int fmt,O,I; const char*n; } shp[] = {
        {FMT_TQ2_T, 2048,2048,"square proj"}, {FMT_TQ2_T,  512,2048,"narrow proj"},
        {FMT_TQ2_T,  512,2048,"expert gate/up"}, {FMT_TQ2_T, 2048, 512,"expert down"},
        {FMT_Q4A_T, 4096,2048,"lm_head slab"},   {FMT_Q4A_T, 4736,2048,"flash centroids"},
    };
    int fail=0;
    unsigned seed=11;
    #define NXT() (seed=seed*1664525u+1013904223u)
    #define UNI() ((float)(NXT()>>8)/8388608.0f-1.0f)
    for (unsigned t=0;t<sizeof shp/sizeof*shp;t++){
        int fmt=shp[t].fmt,O=shp[t].O,I=shp[t].I,S=3;
        size_t wb=(size_t)(fmt==FMT_TQ2_T?tq2_tensor_bytes(O,I):q4a_tensor_bytes(O,I));
        uint8_t *W=malloc(wb);
        if(fmt==FMT_TQ2_T){
            float *al=(float*)W;
            for(int o=0;o<O;o++) al[o]=0.01f+0.04f*fabsf(UNI());
            uint8_t *codes=W+(size_t)O*4;
            /* codes are {0,1,2} in the checkpoint; 3 never occurs, so do not
             * manufacture it here either */
            for(size_t i=0;i<(size_t)O*(I/4);i++){
                unsigned b=0;
                for(int k=0;k<4;k++) b |= (unsigned)((NXT()>>8)%3u)<<(2*k);
                codes[i]=(uint8_t)b;
            }
        } else {
            for(int o=0;o<O;o++){
                uint8_t *row=W+(size_t)o*q4a_row_bytes(I);
                for(int g=0;g<I/Q4A_GRP;g++){
                    uint8_t *grp=row+(size_t)g*Q4A_GRP_BYTES;
                    uint16_t hd=q40_f32_to_fp16(0.002f+0.01f*fabsf(UNI()));
                    uint16_t hm=q40_f32_to_fp16(0.05f*UNI());
                    memcpy(grp,&hd,2); memcpy(grp+2,&hm,2);
                    for(int j=0;j<32;j++) grp[4+j]=(uint8_t)(NXT()>>8);
                }
            }
        }
        float *X=malloc(sizeof(float)*(size_t)S*I);
        for(size_t i=0;i<(size_t)S*I;i++) X[i]=UNI();
        float *yg=malloc(sizeof(float)*(size_t)S*O), *yc=malloc(sizeof(float)*(size_t)S*O);
        if(fmt==FMT_TQ2_T){
            sim_kernel_tq2(yg,W,X,O,I,S);
            const float *al=tq2_alphas(W); const uint8_t *cd=tq2_codes(W,O);
            for(int s=0;s<S;s++) for(int o=0;o<O;o++)
                yc[(size_t)s*O+o]=tq2_dot_f32(cd+(size_t)o*tq2_row_bytes(I),al[o],
                                              X+(size_t)s*I,I);
        } else {
            sim_kernel_q4a(yg,W,X,O,I,S);
            for(int s=0;s<S;s++) for(int o=0;o<O;o++)
                yc[(size_t)s*O+o]=q4a_dot_f32(W+(size_t)o*q4a_row_bytes(I),
                                              X+(size_t)s*I,I);
        }
        double worst=0,mag=0;
        for(size_t i=0;i<(size_t)S*O;i++){
            double d=fabs((double)yg[i]-(double)yc[i]);
            if(!(d<=worst))worst=d;                /* NaN-safe */
            if(fabs(yc[i])>mag)mag=fabs(yc[i]);
        }
        double rel=worst/(mag+1e-9);
        printf("  %-16s [%5d x %5d] S=%d  max rel err %.3e  %s\n",
               shp[t].n,O,I,S,rel, rel<1e-5?"ok":"MISMATCH");
        if(!(rel<1e-5)) fail=1;
        free(W);free(X);free(yg);free(yc);
    }
    #undef UNI
    #undef NXT
    return fail;
}

/* lfm25's shapes, in q8_0: the always-on tensors and the edge-layer experts. */
static int check_q80(void){
    struct { int O,I; const char*n; } shp[] = {
        {2048,2048,"q_proj"}, {512,2048,"k/v_proj"}, {6144,2048,"conv in_proj"},
        {8192,2048,"dense gate/up"}, {2048,8192,"dense down"},
        {1792,2048,"edge expert g/u"},
    };
    int fail=0;
    unsigned seed=7;
    for (unsigned t=0;t<sizeof shp/sizeof*shp;t++){
        int O=shp[t].O,I=shp[t].I,S=3;
        size_t rb=(size_t)(I/Q40_BLK)*Q80_BLK_BYTES;
        float *w=malloc(sizeof(float)*(size_t)O*I);
        for(size_t i=0;i<(size_t)O*I;i++){seed=seed*1664525u+1013904223u;
            w[i]=((float)(seed>>8)/8388608.0f-1.0f)*0.05f;}
        uint8_t *W=malloc(rb*(size_t)O);
        for(int o=0;o<O;o++) q80_quant_row(w+(size_t)o*I, W+(size_t)o*rb, I);
        float *X=malloc(sizeof(float)*(size_t)S*I);
        for(size_t i=0;i<(size_t)S*I;i++){seed=seed*1664525u+1013904223u;
            X[i]=(float)(seed>>8)/8388608.0f-1.0f;}
        float *yg=malloc(sizeof(float)*(size_t)S*O), *yc=malloc(sizeof(float)*(size_t)S*O);
        sim_kernel8(yg,W,X,O,I,S);
        for(int s=0;s<S;s++) for(int o=0;o<O;o++)
            yc[(size_t)s*O+o]=q80_dot_f32(W+(size_t)o*rb, X+(size_t)s*I, I);
        double worst=0,mag=0;
        for(size_t i=0;i<(size_t)S*O;i++){
            double d=fabs(yg[i]-yc[i]); if(d>worst)worst=d;
            if(fabs(yc[i])>mag)mag=fabs(yc[i]);
        }
        double rel=worst/(mag+1e-9);
        printf("  %-16s [%5d x %5d] S=%d  max rel err %.3e  %s\n",
               shp[t].n,O,I,S,rel, rel<1e-5?"ok":"MISMATCH");
        if(rel>=1e-5) fail=1;
        free(w);free(W);free(X);free(yg);free(yc);
    }
    return fail;
}

int main(void){
    struct { int O,I; const char*n; } shp[] = {
        {4096,2816,"q_proj sliding"}, {8192,2816,"q_proj global"},
        {2112,2816,"mlp gate/up"}, {2816,2112,"mlp down"},
        {704,2816,"expert gate/up"}, {2816,704,"expert down"},
    };
    int fail=0;
    unsigned seed=1;
    for (unsigned t=0;t<sizeof shp/sizeof*shp;t++){
        int O=shp[t].O,I=shp[t].I,S=3;
        float *w=malloc(sizeof(float)*(size_t)O*I);
        for(size_t i=0;i<(size_t)O*I;i++){seed=seed*1664525u+1013904223u;
            w[i]=((float)(seed>>8)/8388608.0f-1.0f)*0.05f;}
        uint8_t *W=malloc(q40_tensor_bytes(O,I));
        for(int o=0;o<O;o++) q40_quant_row(w+(size_t)o*I, W+(size_t)o*q40_row_bytes(I), I);
        float *X=malloc(sizeof(float)*(size_t)S*I);
        for(size_t i=0;i<(size_t)S*I;i++){seed=seed*1664525u+1013904223u;
            X[i]=(float)(seed>>8)/8388608.0f-1.0f;}
        float *yg=malloc(sizeof(float)*(size_t)S*O), *yc=malloc(sizeof(float)*(size_t)S*O);
        sim_kernel(yg,W,X,O,I,S);
        for(int s=0;s<S;s++) for(int o=0;o<O;o++)
            yc[(size_t)s*O+o]=q40_dot_f32(W+(size_t)o*q40_row_bytes(I), X+(size_t)s*I, I);
        double worst=0,mag=0;
        for(size_t i=0;i<(size_t)S*O;i++){
            double d=fabs(yg[i]-yc[i]); if(d>worst)worst=d;
            if(fabs(yc[i])>mag)mag=fabs(yc[i]);
        }
        double rel=worst/(mag+1e-9);
        printf("  %-16s [%5d x %5d] S=%d  max rel err %.3e  %s\n",
               shp[t].n,O,I,S,rel, rel<1e-5?"ok":"MISMATCH");
        if(rel>=1e-5) fail=1;
        free(w);free(W);free(X);free(yg);free(yc);
    }
    printf("\nq8_0 (lfm25):\n");
    fail |= check_q80();
    printf("\ntq2 / q4a (maple):\n");
    fail |= check_maple();
    printf("\n%s\n", fail?"SHADER LOGIC FAILED":"shader logic ok");
    return fail;
}
