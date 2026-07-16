/* test_q40.c — dependency-free checks for the q4_0 block-scale path.
 * Build: cc -O2 -mavx2 -mfma tests/test_q40.c -lm -o /tmp/t && /tmp/t
 *    or: cc -O2          tests/test_q40.c -lm -o /tmp/t && /tmp/t   (scalar path)
 */
#include "../q40.h"
#include <stdio.h>
#include <stdlib.h>

static uint32_t rs=12345u;
static float frand(void){ rs=rs*1664525u+1013904223u; return ((float)(rs>>8)/8388608.0f-1.0f); }

/* independent scalar reference, never the header's SIMD path */
static float ref_dot(const uint8_t *w, const int8_t *xq, const float *sx, int I){
    double acc=0;
    for(int b=0;b<I/Q40_BLK;b++){
        const uint8_t *blk=w+(size_t)b*Q40_BLK_BYTES;
        uint16_t hd; memcpy(&hd,blk,2);
        const int8_t *x=xq+b*Q40_BLK;
        long s=0;
        for(int j=0;j<16;j++){
            s += ((int)(blk[2+j]&0x0f)-8)*(int)x[j];
            s += ((int)(blk[2+j]>>4)  -8)*(int)x[j+16];
        }
        acc += (double)q40_fp16_to_f32(hd)*(double)sx[b]*(double)s;
    }
    return (float)acc;
}

int main(void){
    int fail=0;

    /* 1. fp16 roundtrip on values in the range scales actually occupy */
    for(int i=0;i<200000;i++){
        float f=frand()*0.5f;
        float g=q40_fp16_to_f32(q40_f32_to_fp16(f));
        float tol=fabsf(f)*(1.0f/1024.0f)+1e-7f;   /* fp16 has 11 bits of mantissa */
        if(fabsf(f-g)>tol){ printf("FAIL fp16 %g -> %g\n",f,g); fail=1; break; }
    }
    if(q40_fp16_to_f32(q40_f32_to_fp16(0.0f))!=0.0f){ printf("FAIL fp16 zero\n"); fail=1; }

    /* 2. quantise/dequantise. NOTE the q4_0 codebook is ASYMMETRIC: with
     *    d = mx / -8 (mx = the signed max-|.| element), the representable set is
     *    d*[-8..+7]. So the max-magnitude element is always exact, but the tail
     *    on the OPPOSITE side clips at 7|d| and can carry ~|d| of error. That is
     *    the format, not a defect — assert exactly that contract. */
    const int I=4096;
    float *w=malloc(I*sizeof(float)), *wd=malloc(I*sizeof(float));
    uint8_t *wq=malloc(q40_row_bytes(I));
    for(int i=0;i<I;i++) w[i]=frand()*0.1f;
    q40_quant_row(w,wq,I);
    q40_dequant_row(wq,wd,I);
    for(int b=0;b<I/Q40_BLK;b++){
        uint16_t hd; memcpy(&hd,wq+(size_t)b*Q40_BLK_BYTES,2);
        float ds=q40_fp16_to_f32(hd), d=fabsf(ds);
        float lo, hi;                                  /* representable interval */
        if(ds>=0){ lo=-8.0f*ds; hi=7.0f*ds; } else { hi=-8.0f*ds; lo=7.0f*ds; }
        float amax=0.f, mx=0.f;
        for(int j=0;j<Q40_BLK;j++){ float a=fabsf(w[b*Q40_BLK+j]); if(a>amax){ amax=a; mx=w[b*Q40_BLK+j]; } }
        /* the max-magnitude element must be reproduced essentially exactly */
        for(int j=0;j<Q40_BLK;j++){
            int k=b*Q40_BLK+j;
            if(w[k]==mx && fabsf(w[k]-wd[k]) > 0.01f*d + 1e-6f){
                printf("FAIL max elem not preserved: %g vs %g\n",w[k],wd[k]); fail=1;
            }
        }
        for(int j=0;j<Q40_BLK;j++){
            int k=b*Q40_BLK+j; float t=w[k];
            if(t>=lo && t<=hi){                        /* in range: half-step bound */
                if(fabsf(t-wd[k]) > 0.5f*d + 1e-6f){
                    printf("FAIL quant in-range blk %d j %d: %g vs %g (d=%g)\n",b,j,t,wd[k],d); fail=1; b=I; break;
                }
            } else {                                   /* out of range: must clamp to the endpoint */
                float end = (t>hi)?hi:lo;
                if(fabsf(wd[k]-end) > 1e-6f){
                    printf("FAIL quant clamp blk %d j %d: %g -> %g, expected %g\n",b,j,t,wd[k],end); fail=1; b=I; break;
                }
            }
        }
    }

    /* 3. the kernel must agree with the scalar reference to f32 rounding, and
     *    with an exact f32 dot of the DEQUANTISED weights (this is the real
     *    contract: the kernel introduces no error beyond activation quant). */
    float *x=malloc(I*sizeof(float));
    int8_t *xq=malloc(I);
    float *sx=malloc((I/Q40_BLK)*sizeof(float));
    double worst_ref=0, worst_deq=0;
    for(int trial=0;trial<200;trial++){
        for(int i=0;i<I;i++) x[i]=frand();
        q40_quant_act(x,xq,sx,I);

        float got=q40_dot(wq,xq,sx,I);
        float ref=ref_dot(wq,xq,sx,I);

        /* exact f32 dot of dequantised weights against dequantised activations:
         * isolates the kernel from the quantiser */
        double deq=0;
        for(int b=0;b<I/Q40_BLK;b++)
            for(int j=0;j<Q40_BLK;j++){
                int k=b*Q40_BLK+j;
                deq += (double)wd[k]*(double)xq[k]*(double)sx[b];
            }

        /* the dot of two zero-mean vectors cancels almost completely, so |ref| is
         * a meaningless denominator (f32 accumulation noise dwarfs it). Normalise
         * by the SUMMAND magnitude — that is what f32 accumulation error scales with. */
        double mag=0;
        for(int b=0;b<I/Q40_BLK;b++)
            for(int j=0;j<Q40_BLK;j++){
                int k=b*Q40_BLK+j;
                mag += fabs((double)wd[k]*(double)xq[k]*(double)sx[b]);
            }
        double scale=mag+1e-6;
        double e1=fabs(got-ref)/scale, e2=fabs(got-deq)/scale;
        if(e1>worst_ref) worst_ref=e1;
        if(e2>worst_deq) worst_deq=e2;
    }
    printf("kernel vs scalar ref : max rel err %.3e\n", worst_ref);
    printf("kernel vs f32 dequant: max rel err %.3e\n", worst_deq);
    if(worst_ref>1e-5){ printf("FAIL kernel disagrees with reference\n"); fail=1; }
    if(worst_deq>1e-5){ printf("FAIL kernel disagrees with f32 dequant\n"); fail=1; }

    /* 4. saturation guard: the maddubs i16 accumulator must not overflow at the
     *    extremes (q=15, x=+/-127). Construct the worst case explicitly. */
    for(int b=0;b<I/Q40_BLK;b++){
        uint8_t *blk=wq+(size_t)b*Q40_BLK_BYTES;
        uint16_t hd=q40_f32_to_fp16(1.0f); memcpy(blk,&hd,2);
        for(int j=0;j<16;j++) blk[2+j]=0xff;      /* every nibble = 15 */
    }
    for(int i=0;i<I;i++) xq[i]=(int8_t)127;
    for(int b=0;b<I/Q40_BLK;b++) sx[b]=1.0f;
    {
        float got=q40_dot(wq,xq,sx,I), ref=ref_dot(wq,xq,sx,I);
        if(fabsf(got-ref)/fabsf(ref) > 1e-5f){
            printf("FAIL saturation: %g vs %g\n",got,ref); fail=1;
        } else printf("saturation extreme ok (%.0f)\n", got);
    }

    printf(fail?"FAILED\n":"ok\n");
    return fail;
}
