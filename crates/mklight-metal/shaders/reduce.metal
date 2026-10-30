// Deterministic compensated parallel reduction (CONVENTIONS §7.5), used for
// spot centroids/RMS, energy and other statistics. The order of operations is
// fixed by the data size only (no atomics), and is specified and emulated
// exactly by mklight_gpu::reduce; see that module for the full description.
//
// Pass 1: grid (k, channels) with k <= num_groups, MKL_REDUCE_THREADS threads
//         per group, computing groups first_group .. first_group + k - 1.
// Pass 2: grid (1, channels), MKL_REDUCE_THREADS threads per group.
// The host binds x, partials and sums at the first channel of a launch.
#include "mklight_common.h"

// Accumulator (MKL_REDUCE_*), specialised per pipeline.
constant uint mkl_fc_reduce_mode [[function_constant(1)]];

static inline float2 mkl_reduce_combine(float2 a, float2 b) {
    if (mkl_fc_reduce_mode == MKL_REDUCE_PLAIN) {
        return float2(a.x + b.x, 0.0f);
    }
    return mkl_df_add(a, b);
}

// Binary tree over the group's MKL_REDUCE_THREADS values: for stride =
// T/2 .. 1, v[i] <- v[i] (+) v[i + stride].
static inline float2 mkl_group_tree(threadgroup float2* v, uint tid) {
    for (uint stride = MKL_REDUCE_THREADS / 2u; stride > 0u; stride >>= 1) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < stride) {
            v[tid] = mkl_reduce_combine(v[tid], v[tid + stride]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return v[0];
}

kernel void mkl_reduce_pass1(device const float* x [[buffer(0)]],
                             device float2* partials [[buffer(1)]],
                             constant MklReduceParams& p [[buffer(2)]],
                             uint2 lid [[thread_position_in_threadgroup]],
                             uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float2 v[MKL_REDUCE_THREADS];
    uint tid = lid.x;
    uint channel = group.y;
    device const float* xc = x + channel * p.channel_stride;
    uint stride = p.num_groups * MKL_REDUCE_THREADS;
    uint gidx = p.first_group + group.x;
    uint g = gidx * MKL_REDUCE_THREADS + tid;

    float2 acc;
    if (mkl_fc_reduce_mode == MKL_REDUCE_PLAIN) {
        float s = 0.0f;
        for (uint i = g; i < p.count; i += stride) s += xc[i];
        acc = float2(s, 0.0f);
    } else if (mkl_fc_reduce_mode == MKL_REDUCE_KAHAN) {
        float s = 0.0f, c = 0.0f;
        for (uint i = g; i < p.count; i += stride) mkl_neumaier_add(s, c, xc[i]);
        acc = mkl_two_sum(s, c);
    } else {
        acc = float2(0.0f);
        for (uint i = g; i < p.count; i += stride) acc = mkl_df_add_f32(acc, xc[i]);
    }
    v[tid] = acc;
    float2 r = mkl_group_tree(v, tid);
    if (tid == 0u) {
        partials[channel * p.num_groups + gidx] = r;
    }
}

kernel void mkl_reduce_pass2(device const float2* partials [[buffer(0)]],
                             device float2* sums [[buffer(1)]],
                             constant MklReduceParams& p [[buffer(2)]],
                             uint2 lid [[thread_position_in_threadgroup]],
                             uint2 group [[threadgroup_position_in_grid]]) {
    threadgroup float2 v[MKL_REDUCE_THREADS];
    uint tid = lid.x;
    uint channel = group.y;
    float2 acc = float2(0.0f);
    for (uint j = tid; j < p.num_groups; j += MKL_REDUCE_THREADS) {
        acc = mkl_reduce_combine(acc, partials[channel * p.num_groups + j]);
    }
    v[tid] = acc;
    float2 r = mkl_group_tree(v, tid);
    if (tid == 0u) {
        sums[channel] = r;
    }
}
