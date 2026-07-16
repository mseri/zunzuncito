/* Transliteration of the Metal kernel body, lane-for-lane, to validate the SHADER
 * LOGIC (nibble order, unaligned fp16 scale assembly, 32-lane strided reduction)
 * on a machine with no Metal. Any mismatch here is a real shader bug. */
#include "q40.h"
#include <stdio.h>
#include <stdlib.h>
#define TG 32

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
    printf("\n%s\n", fail?"SHADER LOGIC FAILED":"shader logic ok");
    return fail;
}
