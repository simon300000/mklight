// Philox4x32-10 kernels (CONVENTIONS §6). Bit-identical to
// mklight_core::rng::philox4x32_10 / Philox and to the CPU-f32 emulation.
#include "mklight_common.h"

// One thread per 4-word block of the stream (seed, stream): block index
// first_block + t, counter (block lo, block hi, stream lo, stream hi),
// key (seed lo, seed hi).
kernel void mkl_philox_stream(constant MklPhiloxParams& p [[buffer(0)]],
                              device uint4* raw [[buffer(1)]],
                              device float4* uniform [[buffer(2)]],
                              uint t [[thread_position_in_grid]]) {
    if (t >= p.num_blocks) return;
    // 64-bit block index with carry.
    uint lo = p.first_block[0] + t;
    uint hi = p.first_block[1] + (lo < t ? 1u : 0u);
    uint4 r = mkl_philox4x32_10(uint4(lo, hi, p.stream[0], p.stream[1]),
                                uint2(p.key[0], p.key[1]));
    raw[t] = r;
    uniform[t] = float4(mkl_u32_to_unit(r.x), mkl_u32_to_unit(r.y),
                        mkl_u32_to_unit(r.z), mkl_u32_to_unit(r.w));
}

// Arbitrary counters and keys, one block per thread.
kernel void mkl_philox_blocks(device const uint4* counters [[buffer(0)]],
                              device const uint2* keys [[buffer(1)]],
                              device uint4* raw [[buffer(2)]],
                              device float4* uniform [[buffer(3)]],
                              constant uint& n [[buffer(4)]],
                              uint t [[thread_position_in_grid]]) {
    if (t >= n) return;
    uint4 r = mkl_philox4x32_10(counters[t], keys[t]);
    raw[t] = r;
    uniform[t] = float4(mkl_u32_to_unit(r.x), mkl_u32_to_unit(r.y),
                        mkl_u32_to_unit(r.z), mkl_u32_to_unit(r.w));
}
