/* gpu.h — Metal backend interface (Apple silicon / Intel Mac with a Metal GPU).
 *
 * Only the quantised matvec/matmul is offloaded -- q4_0 and q8_0, one kernel each --
 * which is ~95% of the arithmetic: the attention projections, the dense MLP, and the
 * expert GEMMs. (gemma4 is q4_0 throughout; lfm25's apex gradient makes its always-on
 * tensors q8_0, hence the second kernel.)
 *
 * Attention, routing, the KV codec and the expert cache stay on the CPU on purpose:
 * they are small, branchy, or I/O-bound, and at a 4-8 GB budget this engine is
 * disk-bound anyway (~800 MB of expert reads per token against 7.6 GFLOP of compute),
 * so moving arithmetic to the GPU does not help you wait on NVMe.
 *
 * Metal pays for itself in two places:
 *   * prefill, which is batched and genuinely compute-bound;
 *   * a 16 GB machine, where the whole container is resident, there is no disk in
 *     the loop, and the ~400 GB/s of unified memory bandwidth beats the CPU's ~100.
 * Expect little from it during 4 GB decode.
 *
 * Expert slots and the dense blob are page-aligned (posix_memalign, 4096) and wrapped
 * with newBufferWithBytesNoCopy, so the GPU reads the same pages the streaming cache
 * filled. There is no host->device copy anywhere in the hot path; one would cost more
 * than the matmul saves.
 *
 * The Metal kernel consumes f32 activations and decodes q4_0 weights on the fly, so
 * it is numerically equivalent to the CPU's q40_dot_f32: it matches the COLI_F32ACT
 * reference path and is more accurate than the default int8 activation path.
 * `--check-gpu` diffs the two and prints the max relative error; run it once on your
 * machine before trusting any output.
 *
 * Every entry point is fallible. No Metal device, a shader compile failure or a
 * buffer allocation failure all leave gpu_init returning 0 and the engine running
 * entirely on the CPU, so correctness never depends on Metal.
 */
#ifndef COLI_GPU_H
#define COLI_GPU_H

#include <stdint.h>
#include <stddef.h>

/* Weight formats the kernels speak. The values match lfm25.c's FMT_* on purpose so
 * that engine can pass its tensor format straight through. */
#define GPU_FMT_Q40 1
#define GPU_FMT_Q80 2

#ifdef COLI_METAL

#ifdef __cplusplus
extern "C" {
#endif

/* 1 if a Metal device exists and the shaders compiled. Safe to call always. */
int  gpu_init(void);
void gpu_shutdown(void);
int  gpu_ready(void);
const char *gpu_name(void);

/* Register a host allocation as a GPU-visible buffer with NO copy.
 * `p` must be page-aligned and `n` a whole number of pages (xmalloc guarantees the
 * former). Returns 0 on failure, in which case the caller must stay on the CPU. */
int  gpu_map(const void *p, size_t n);

/* y[S,O] = W[O,I] (q4_0) * x[S,I] (f32), row-major, I % 32 == 0.
 * W must lie inside a previously gpu_map'd region. Returns 0 if it could not run,
 * and the caller falls back to the CPU. */
int  gpu_q40_matmul(float *y, const uint8_t *W, const float *x, int O, int I, int S);

/* Same, for either GPU_FMT_Q40 or GPU_FMT_Q80 weights. Returns 0 (caller falls back
 * to the CPU) for any other format, or if that kernel failed to compile. */
int  gpu_matmul(int fmt, float *y, const uint8_t *W, const float *x,
                int O, int I, int S);

#ifdef __cplusplus
}
#endif

#else
/* Non-Apple builds, or Metal explicitly disabled: every entry point becomes a no-op
 * that declines. gemma4.c then needs no #ifdefs in its hot path -- it asks the GPU,
 * is told no, and runs on the CPU. */
static inline int  gpu_init(void)        { return 0; }
static inline void gpu_shutdown(void)    { }
static inline int  gpu_ready(void)       { return 0; }
static inline const char *gpu_name(void) { return "none"; }
static inline int  gpu_map(const void *p, size_t n) { (void)p; (void)n; return 0; }
static inline int  gpu_q40_matmul(float *y, const uint8_t *W, const float *x,
                                  int O, int I, int S) {
    (void)y; (void)W; (void)x; (void)O; (void)I; (void)S; return 0;
}
static inline int  gpu_matmul(int fmt, float *y, const uint8_t *W, const float *x,
                              int O, int I, int S) {
    (void)fmt; (void)y; (void)W; (void)x; (void)O; (void)I; (void)S; return 0;
}
#endif /* COLI_METAL */

#endif /* COLI_GPU_H */
