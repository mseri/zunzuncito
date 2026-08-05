/* metal.mm — Metal backend for the quantised matmul. Built only on Darwin.
 *
 * Objective-C++ (not .m) so the shader can be a raw string literal; plain
 * Objective-C has none, and escaping 60 lines of Metal by hand is a bug farm.
 *
 * The shader is compiled at runtime with newLibraryWithSource, so there is no
 * .metallib to build, ship, or keep in sync with the binary. Costs ~50 ms once.
 *
 * Built without ARC (-fno-objc-arc), deliberately. ARC restricts Objective-C pointers
 * as members of C structs, and the pointer->MTLBuffer map below is exactly that.
 * Under manual retain the lifetimes here are trivial: every object we create (device,
 * library, pipeline, queue, buffers) is kept for the life of the process and never
 * released; the only transient objects are the per-call command buffer and encoder,
 * which are autoreleased inside an @autoreleasepool.
 *
 * See gpu.h for what is offloaded and why. Short version: the quantised
 * matvec/matmul only (~95% of the FLOPs), one kernel per weight format; attention,
 * routing, KV codec and the expert cache stay on the CPU; and any failure here falls
 * back to the CPU rather than aborting.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "gpu.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* shader */
static const char *kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

// q4_0: 32 weights per 18-byte block -> fp16 scale d, then 16 bytes of nibbles.
//   qs[j] low  nibble -> weight j        (j = 0..15)
//   qs[j] high nibble -> weight j + 16
//   w = d * (q - 8)
//
// One threadgroup per (output row, batch element); its 32 lanes stride over the
// row's blocks and then reduce. One row per thread would be simpler but leaves the
// memory system idle -- the weights are the bandwidth here, and we want 32 lanes
// streaming them concurrently.
//
// Activations are consumed as f32 (no int8 quantisation on the GPU), so this is
// numerically identical to the CPU's q40_dot_f32 reference, and more accurate than
// the CPU's default int8-activation path.

constant uint TG = 32;

kernel void q40_matmul(
    device const uchar  *W   [[buffer(0)]],   // [O, (I/32)*18]
    device const float  *X   [[buffer(1)]],   // [S, I]
    device       float  *Y   [[buffer(2)]],   // [S, O]
    constant     uint   &O   [[buffer(3)]],
    constant     uint   &I   [[buffer(4)]],
    constant     uint   &S   [[buffer(5)]],
    // Both position attributes must have the same dimensionality: Intel Macs' Metal
    // compiler rejects a uint2/uint mix outright ("all scalar types or all vector
    // types with the same number of elements"), while Apple silicon's accepts it.
    // uint3 for both is portable across every Metal device.
    uint3 gid  [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]])
{
    const uint row  = gid.x;
    const uint s    = gid.y;
    const uint lane = tid.x;
    if (row >= O || s >= S) return;

    const uint nb = I / 32u;
    const ulong rb = (ulong)nb * 18u;
    device const uchar *w = W + (ulong)row * rb;
    device const float *x = X + (ulong)s * I;

    float acc = 0.0f;
    for (uint b = lane; b < nb; b += TG) {
        device const uchar *blk = w + (ulong)b * 18u;

        // the fp16 scale is not 2-byte aligned inside an 18-byte block, so assemble
        // it from bytes rather than reinterpreting the pointer
        ushort bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        float  d    = (float)as_type<half>(bits);

        device const float *xv = x + (ulong)b * 32u;
        float s0 = 0.0f;
        for (uint j = 0; j < 16u; ++j) {
            uchar q = blk[2u + j];
            s0 += (float)((int)(q & 0x0F) - 8) * xv[j];
            s0 += (float)((int)(q >>   4) - 8) * xv[j + 16u];
        }
        acc += d * s0;
    }

    threadgroup float part[TG];
    part[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) part[lane] += part[lane + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) Y[(ulong)s * O + row] = part[0];
}

// q8_0: 32 weights per 34-byte block -> fp16 scale d, then 32 signed int8 codes.
//   w = d * q
//
// Same threadgroup shape and reduction as q40_matmul above; only the block decode
// differs. It is a separate kernel rather than a branch inside one because the
// per-block byte stride is a compile-time constant in both, and the inner loop is
// where every cycle of this thing goes.
kernel void q80_matmul(
    device const uchar  *W   [[buffer(0)]],   // [O, (I/32)*34]
    device const float  *X   [[buffer(1)]],   // [S, I]
    device       float  *Y   [[buffer(2)]],   // [S, O]
    constant     uint   &O   [[buffer(3)]],
    constant     uint   &I   [[buffer(4)]],
    constant     uint   &S   [[buffer(5)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]])
{
    const uint row  = gid.x;
    const uint s    = gid.y;
    const uint lane = tid.x;
    if (row >= O || s >= S) return;

    const uint nb = I / 32u;
    const ulong rb = (ulong)nb * 34u;
    device const uchar *w = W + (ulong)row * rb;
    device const float *x = X + (ulong)s * I;

    float acc = 0.0f;
    for (uint b = lane; b < nb; b += TG) {
        device const uchar *blk = w + (ulong)b * 34u;

        // 34 is even, but the tensor base need not be 2-byte aligned relative to the
        // buffer origin, so assemble the fp16 from bytes here too
        ushort bits = (ushort)blk[0] | ((ushort)blk[1] << 8);
        float  d    = (float)as_type<half>(bits);

        device const float *xv = x + (ulong)b * 32u;
        float s0 = 0.0f;
        for (uint j = 0; j < 32u; ++j)
            s0 += (float)(int)(char)blk[2u + j] * xv[j];
        acc += d * s0;
    }

    threadgroup float part[TG];
    part[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) part[lane] += part[lane + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) Y[(ulong)s * O + row] = part[0];
}

// tq2: ternary, ONE f32 scale per output row and 2-bit codes packed 4 per byte.
//   tensor = alpha[O] (f32) followed by codes[O][I/4]
//   group  = 64 weights = 16 bytes; byte k holds weights k, k+16, k+32, k+48 at bit
//            positions 0, 2, 4, 6
//   w = alpha_row * (code - 1),  code in {0,1,2}
//
// The row-wise scale is what makes this the cheapest inner loop of the four: the
// q4_0/q8_0 kernels above assemble an fp16 from bytes once per 32 weights, whereas
// here the scale leaves the loop entirely and the body is a mask, a subtract and an
// FMA.
//
// The CPU's int8 path folds the -1 into a precomputed sum(x) (see tq2.h); with f32
// activations there is nothing to hoist, so the -1 is applied inline and this
// reproduces tq2_dot_f32 instead.
kernel void tq2_matmul(
    device const uchar  *W   [[buffer(0)]],   // alpha[O] f32, then codes[O][I/4]
    device const float  *X   [[buffer(1)]],   // [S, I]
    device       float  *Y   [[buffer(2)]],   // [S, O]
    constant     uint   &O   [[buffer(3)]],
    constant     uint   &I   [[buffer(4)]],
    constant     uint   &S   [[buffer(5)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]])
{
    const uint row  = gid.x;
    const uint s    = gid.y;
    const uint lane = tid.x;
    if (row >= O || s >= S) return;

    const uint ng = I / 64u;                       // groups of 64 weights
    device const float *alpha = (device const float *)W;
    device const uchar *w = W + (ulong)O * 4u + (ulong)row * (ulong)(I / 4u);
    device const float *x = X + (ulong)s * I;

    float acc = 0.0f;
    for (uint g = lane; g < ng; g += TG) {
        device const uchar *c = w + (ulong)g * 16u;
        device const float *xv = x + (ulong)g * 64u;
        float s0 = 0.0f;
        for (uint k = 0; k < 16u; ++k) {
            uchar b = c[k];
            s0 += (float)((int)( b       & 3) - 1) * xv[k];
            s0 += (float)((int)((b >> 2) & 3) - 1) * xv[k + 16u];
            s0 += (float)((int)((b >> 4) & 3) - 1) * xv[k + 32u];
            s0 += (float)((int)( b >> 6)      - 1) * xv[k + 48u];
        }
        acc += s0;
    }

    threadgroup float part[TG];
    part[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) part[lane] += part[lane + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) Y[(ulong)s * O + row] = alpha[row] * part[0];
}

// q4a: 4-bit affine, an fp16 scale AND an fp16 bias per group of 64 weights.
//   group = 36 bytes: d, m, then two 16-byte nibble blocks
//     bytes  4..19  low nibble -> w[j],      high -> w[j + 16]
//     bytes 20..35  low nibble -> w[j + 32], high -> w[j + 48]
//   w = d * q + m,  q in [0,15]
//
// The bias needs sum(x) over the group, which unlike tq2's -1 does not factor out of
// the row (it is weighted by m), but it is one extra add in the same loop.
kernel void q4a_matmul(
    device const uchar  *W   [[buffer(0)]],   // [O, (I/64)*36]
    device const float  *X   [[buffer(1)]],   // [S, I]
    device       float  *Y   [[buffer(2)]],   // [S, O]
    constant     uint   &O   [[buffer(3)]],
    constant     uint   &I   [[buffer(4)]],
    constant     uint   &S   [[buffer(5)]],
    uint3 gid  [[threadgroup_position_in_grid]],
    uint3 tid  [[thread_position_in_threadgroup]])
{
    const uint row  = gid.x;
    const uint s    = gid.y;
    const uint lane = tid.x;
    if (row >= O || s >= S) return;

    const uint ng = I / 64u;
    const ulong rb = (ulong)ng * 36u;
    device const uchar *w = W + (ulong)row * rb;
    device const float *x = X + (ulong)s * I;

    float acc = 0.0f;
    for (uint g = lane; g < ng; g += TG) {
        device const uchar *grp = w + (ulong)g * 36u;

        // 36 is even, but the tensor base need not be 2-byte aligned relative to the
        // buffer origin, so assemble both fp16s from bytes
        ushort db = (ushort)grp[0] | ((ushort)grp[1] << 8);
        ushort mb = (ushort)grp[2] | ((ushort)grp[3] << 8);
        float d = (float)as_type<half>(db);
        float m = (float)as_type<half>(mb);

        device const float *xv = x + (ulong)g * 64u;
        float dp = 0.0f, sv = 0.0f;
        for (uint h = 0; h < 2u; ++h) {
            device const uchar *nib = grp + 4u + h * 16u;
            device const float *u = xv + h * 32u;
            for (uint j = 0; j < 16u; ++j) {
                uchar q = nib[j];
                dp += (float)(q & 0x0F) * u[j];
                dp += (float)(q >>   4) * u[j + 16u];
                sv += u[j] + u[j + 16u];
            }
        }
        acc += d * dp + m * sv;
    }

    threadgroup float part[TG];
    part[lane] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint off = TG / 2u; off > 0u; off >>= 1u) {
        if (lane < off) part[lane] += part[lane + off];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (lane == 0u) Y[(ulong)s * O + row] = part[0];
}
)METAL";

/* state */
static id<MTLDevice>               g_dev  = nil;
static id<MTLCommandQueue>         g_q    = nil;
static id<MTLComputePipelineState> g_pipe = nil;   /* q4_0 */
static id<MTLComputePipelineState> g_pipe8 = nil;  /* q8_0 */
static id<MTLComputePipelineState> g_pipet = nil;  /* tq2  */
static id<MTLComputePipelineState> g_pipea = nil;  /* q4a  */
static char g_name[128] = "none";
static int  g_ok = 0;

/* host pointer -> MTLBuffer. Lets us find which buffer a weight pointer lives in
 * without ever copying weights. One entry for the dense blob, one per expert slot. */
#define GPU_MAXMAP 8192
static struct { const uint8_t *base; size_t len; id<MTLBuffer> buf; } g_map[GPU_MAXMAP];
static int g_nmap = 0;

static id<MTLBuffer> g_x = nil, g_y = nil;
static size_t g_xcap = 0, g_ycap = 0;

int gpu_init(void) {
    if (g_ok) return 1;
    @autoreleasepool {
        g_dev = MTLCreateSystemDefaultDevice();
        if (!g_dev) return 0;                       /* no Metal GPU -> CPU path */
        snprintf(g_name, sizeof g_name, "%s", [[g_dev name] UTF8String]);

        NSError *err = nil;
        MTLCompileOptions *opt = [[[MTLCompileOptions alloc] init] autorelease];
        id<MTLLibrary> lib =
            [g_dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc]
                                options:opt
                                  error:&err];
        if (!lib) {
            fprintf(stderr, "metal: shader compile failed (%s); using CPU\n",
                    err ? [[err localizedDescription] UTF8String] : "?");
            g_dev = nil;
            return 0;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"q40_matmul"];
        if (!fn) { g_dev = nil; return 0; }

        g_pipe = [g_dev newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        if (!g_pipe) {
            fprintf(stderr, "metal: pipeline creation failed; using CPU\n");
            g_dev = nil;
            return 0;
        }
        /* q8_0 is optional: gemma4 never asks for it, so a failure here only costs
         * lfm25 its GPU path, and gpu_matmul declines for fmt 2. */
        id<MTLFunction> fn8 = [lib newFunctionWithName:@"q80_matmul"];
        if (fn8) {
            g_pipe8 = [g_dev newComputePipelineStateWithFunction:fn8 error:&err];
            [fn8 release];
        }
        /* tq2/q4a are maple's and equally optional: gemma4 and lfm25 never ask for
         * them, and a failure here only costs maple its GPU path. */
        id<MTLFunction> fnt = [lib newFunctionWithName:@"tq2_matmul"];
        if (fnt) {
            g_pipet = [g_dev newComputePipelineStateWithFunction:fnt error:&err];
            [fnt release];
        }
        id<MTLFunction> fna = [lib newFunctionWithName:@"q4a_matmul"];
        if (fna) {
            g_pipea = [g_dev newComputePipelineStateWithFunction:fna error:&err];
            [fna release];
        }
        g_q = [g_dev newCommandQueue];
        if (!g_q) {
            g_dev = nil; g_pipe = nil; g_pipe8 = nil; g_pipet = nil; g_pipea = nil;
            return 0;
        }
        g_ok = 1;
        return 1;
    }
}

void gpu_shutdown(void) {
    g_ok = 0;
    g_nmap = 0;
    g_pipe = nil; g_pipe8 = nil; g_pipet = nil; g_pipea = nil;
    g_q = nil; g_dev = nil; g_x = nil; g_y = nil;
}

int gpu_ready(void)        { return g_ok; }
const char *gpu_name(void) { return g_name; }

int gpu_map(const void *p, size_t n) {
    if (!g_ok || g_nmap >= GPU_MAXMAP || !p || !n) return 0;
    @autoreleasepool {
        /* newBufferWithBytesNoCopy needs a page-aligned pointer and a page-multiple
         * length. Our allocator (posix_memalign, 4096) guarantees the alignment;
         * round the length up. The GPU then reads the same pages the expert cache
         * streamed into, with no host->device copy anywhere in the hot path. */
        size_t pg = (size_t)getpagesize();
        if ((uintptr_t)p & (uintptr_t)(pg - 1)) return 0;
        size_t len = (n + pg - 1) & ~(pg - 1);
        id<MTLBuffer> b = [g_dev newBufferWithBytesNoCopy:(void *)p
                                                   length:len
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
        if (!b) return 0;
        g_map[g_nmap].base = (const uint8_t *)p;
        g_map[g_nmap].len  = len;
        g_map[g_nmap].buf  = b;
        g_nmap++;
        return 1;
    }
}

static id<MTLBuffer> find_map(const uint8_t *p, size_t need, size_t *off) {
    for (int i = 0; i < g_nmap; i++) {
        const uint8_t *b = g_map[i].base;
        if (p >= b && (size_t)(p - b) + need <= g_map[i].len) {
            *off = (size_t)(p - b);
            return g_map[i].buf;
        }
    }
    return nil;
}

static int ensure(id<MTLBuffer> *buf, size_t *cap, size_t need) {
    if (*buf && *cap >= need) return 1;
    id<MTLBuffer> b = [g_dev newBufferWithLength:need
                                         options:MTLResourceStorageModeShared];
    if (!b) return 0;
    *buf = b;
    *cap = need;
    return 1;
}

int gpu_matmul(int fmt, float *y, const uint8_t *W, const float *x,
               int O, int I, int S) {
    if (!g_ok || (I & 31) || O <= 0 || S <= 0) return 0;

    /* pipeline, plus how many bytes of `W` the kernel will touch. tq2 is the odd one
     * out on both counts: its groups are 64 wide, and the tensor carries O row scales
     * ahead of the code rows. */
    id<MTLComputePipelineState> pipe;
    size_t need;
    switch (fmt) {
    case GPU_FMT_Q40: pipe = g_pipe;  need = (size_t)(I / 32) * 18 * (size_t)O; break;
    case GPU_FMT_Q80: pipe = g_pipe8; need = (size_t)(I / 32) * 34 * (size_t)O; break;
    case GPU_FMT_TQ2:
        if (I & 63) return 0;
        pipe = g_pipet; need = (size_t)O * 4 + (size_t)O * (size_t)(I / 4);
        break;
    case GPU_FMT_Q4A:
        if (I & 63) return 0;
        pipe = g_pipea; need = (size_t)(I / 64) * 36 * (size_t)O;
        break;
    default: return 0;
    }
    if (!pipe) return 0;

    size_t off = 0;
    id<MTLBuffer> wb = find_map(W, need, &off);
    if (!wb) return 0;                     /* weights not GPU-mapped -> CPU path */

    @autoreleasepool {
        size_t xn = sizeof(float) * (size_t)S * (size_t)I;
        size_t yn = sizeof(float) * (size_t)S * (size_t)O;
        if (!ensure(&g_x, &g_xcap, xn) || !ensure(&g_y, &g_ycap, yn)) return 0;

        /* only the activations are copied (S*I floats -- kilobytes). The weights,
         * which are the actual volume, are read in place. */
        memcpy([g_x contents], x, xn);

        id<MTLCommandBuffer> cb = [g_q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:pipe];
        [e setBuffer:wb  offset:off atIndex:0];
        [e setBuffer:g_x offset:0   atIndex:1];
        [e setBuffer:g_y offset:0   atIndex:2];
        uint32_t uo = (uint32_t)O, ui = (uint32_t)I, us = (uint32_t)S;
        [e setBytes:&uo length:sizeof uo atIndex:3];
        [e setBytes:&ui length:sizeof ui atIndex:4];
        [e setBytes:&us length:sizeof us atIndex:5];
        [e dispatchThreadgroups:MTLSizeMake((NSUInteger)O, (NSUInteger)S, 1)
          threadsPerThreadgroup:MTLSizeMake(32, 1, 1)];
        [e endEncoding];
        [cb commit];
        [cb waitUntilCompleted];

        if ([cb status] != MTLCommandBufferStatusCompleted) return 0;
        memcpy(y, [g_y contents], yn);
        return 1;
    }
}

int gpu_q40_matmul(float *y, const uint8_t *W, const float *x, int O, int I, int S) {
    return gpu_matmul(GPU_FMT_Q40, y, W, x, O, I, S);
}
