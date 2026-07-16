/* metal.mm — Metal backend for the q4_0 matmul. Built only on Darwin.
 *
 * Objective-C++ (not .m) so the shader can be a raw string literal; plain
 * Objective-C has none, and escaping 60 lines of Metal by hand is a bug farm.
 *
 * The shader is compiled at runtime with newLibraryWithSource, so there is no
 * .metallib to build, ship, or keep in sync with the binary. Costs ~50 ms once.
 *
 * Built WITHOUT ARC (-fno-objc-arc), deliberately. ARC restricts Objective-C pointers
 * as members of C structs, and the pointer->MTLBuffer map below is exactly that.
 * Under manual retain the lifetimes here are trivial: every object we create (device,
 * library, pipeline, queue, buffers) is kept for the life of the process and never
 * released; the only transient objects are the per-call command buffer and encoder,
 * which are autoreleased inside an @autoreleasepool.
 *
 * See gpu.h for what is offloaded and why. Short version: the q4_0 matvec/matmul
 * only (~95% of the FLOPs); attention, routing, KV codec and the expert cache stay
 * on the CPU; and ANY failure here falls back to the CPU rather than aborting.
 */
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "gpu.h"
#include <stdio.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ shader */
static const char *kSrc = R"METAL(
#include <metal_stdlib>
using namespace metal;

// q4_0: 32 weights per 18-byte block -> fp16 scale d, then 16 bytes of nibbles.
//   qs[j] low  nibble -> weight j        (j = 0..15)
//   qs[j] high nibble -> weight j + 16
//   w = d * (q - 8)
//
// One THREADGROUP per (output row, batch element); its 32 lanes stride over the
// row's blocks and then reduce. One-row-per-thread would be simpler but leaves the
// memory system idle -- the weights are the bandwidth here, and we want 32 lanes
// streaming them concurrently.
//
// Activations are consumed as f32 (no int8 quantisation on the GPU), so this is
// numerically identical to the CPU's q40_dot_f32 reference, and strictly MORE
// accurate than the CPU's default int8-activation path.

constant uint TG = 32;

kernel void q40_matmul(
    device const uchar  *W   [[buffer(0)]],   // [O, (I/32)*18]
    device const float  *X   [[buffer(1)]],   // [S, I]
    device       float  *Y   [[buffer(2)]],   // [S, O]
    constant     uint   &O   [[buffer(3)]],
    constant     uint   &I   [[buffer(4)]],
    constant     uint   &S   [[buffer(5)]],
    // Both position attributes MUST have the same dimensionality: Intel Macs' Metal
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
)METAL";

/* ------------------------------------------------------------------ state */
static id<MTLDevice>               g_dev  = nil;
static id<MTLCommandQueue>         g_q    = nil;
static id<MTLComputePipelineState> g_pipe = nil;
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
        g_q = [g_dev newCommandQueue];
        if (!g_q) { g_dev = nil; g_pipe = nil; return 0; }
        g_ok = 1;
        return 1;
    }
}

void gpu_shutdown(void) {
    g_ok = 0;
    g_nmap = 0;
    g_pipe = nil; g_q = nil; g_dev = nil; g_x = nil; g_y = nil;
}

int gpu_ready(void)        { return g_ok; }
const char *gpu_name(void) { return g_name; }

int gpu_map(const void *p, size_t n) {
    if (!g_ok || g_nmap >= GPU_MAXMAP || !p || !n) return 0;
    @autoreleasepool {
        /* newBufferWithBytesNoCopy needs a page-aligned pointer and a page-multiple
         * length. Our allocator (posix_memalign, 4096) guarantees the alignment;
         * round the length up. The GPU then reads the SAME pages the expert cache
         * streamed into -- no host->device copy anywhere in the hot path. */
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

int gpu_q40_matmul(float *y, const uint8_t *W, const float *x, int O, int I, int S) {
    if (!g_ok || (I & 31) || O <= 0 || S <= 0) return 0;

    size_t rb = (size_t)(I / 32) * 18;
    size_t off = 0;
    id<MTLBuffer> wb = find_map(W, rb * (size_t)O, &off);
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
        [e setComputePipelineState:g_pipe];
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
