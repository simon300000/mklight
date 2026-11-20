// Deterministic compensated two-pass reduction (reduce.wgsl's interface,
// reduce.metal's order, specified by mklight_gpu::reduce): MKL_REDUCE_THREADS
// threads per group, a grid-stride accumulation per thread, a binary tree
// over the group (strides T/2 .. 32 in shared memory, 16 .. 1 as warp
// shuffles: the same order) and a single-group tree over the partials.
//
// Pass 1: grid (k, channels), groups first_group .. first_group + k - 1.
// Pass 2: grid (1, channels).
#include "mkl_cuda.h"

struct MklWgslReduceParams {
    uint count, channel_stride, num_groups, first_group, mode;
    // Offsets (elements) of the launch's first channel in x, partials, sums.
    uint x_offset, partial_offset, sum_offset;
};

__device__ inline float2 mkl_reduce_combine(float2 a, float2 b, uint mode) {
    if (mode == MKL_REDUCE_PLAIN) return float2(a.x + b.x, 0.0f);
    return mkl_df_add(a, b);
}

__device__ inline float2 mkl_group_tree(float2* v, uint tid, uint mode) {
    for (uint stride = MKL_REDUCE_THREADS / 2u; stride >= 32u; stride >>= 1) {
        __syncthreads();
        if (tid < stride) v[tid] = mkl_reduce_combine(v[tid], v[tid + stride], mode);
    }
    __syncthreads();
    float2 r = v[tid];
    for (uint stride = 16u; stride > 0u; stride >>= 1) {
        float2 o = float2(__shfl_down_sync(0xffffffffu, r.x, stride),
                          __shfl_down_sync(0xffffffffu, r.y, stride));
        r = mkl_reduce_combine(r, o, mode);
    }
    return r;
}

MKL_KERNEL __launch_bounds__(MKL_REDUCE_THREADS)
mkl_reduce_pass1(const MklWgslReduceParams* u, const float* rx, float2* rpartials, float2* rsums) {
    __shared__ float2 v[MKL_REDUCE_THREADS];
    const MklWgslReduceParams p = *u;
    const uint tid = threadIdx.x;
    const uint channel = blockIdx.y;
    const float* xc = rx + p.x_offset + channel * p.channel_stride;
    const uint stride = p.num_groups * MKL_REDUCE_THREADS;
    const uint gidx = p.first_group + blockIdx.x;
    const uint g = gidx * MKL_REDUCE_THREADS + tid;
    float2 acc;
    if (p.mode == MKL_REDUCE_PLAIN) {
        float s = 0.0f;
        for (uint i = g; i < p.count; i += stride) s += xc[i];
        acc = float2(s, 0.0f);
    } else if (p.mode == MKL_REDUCE_KAHAN) {
        float s = 0.0f, c = 0.0f;
        for (uint i = g; i < p.count; i += stride) mkl_neumaier_add(s, c, xc[i]);
        acc = mkl_two_sum(s, c);
    } else {
        acc = float2(0.0f);
        for (uint i = g; i < p.count; i += stride) acc = mkl_df_add_f32(acc, xc[i]);
    }
    v[tid] = acc;
    const float2 r = mkl_group_tree(v, tid, p.mode);
    if (tid == 0u) rpartials[p.partial_offset + channel * p.num_groups + gidx] = r;
}

MKL_KERNEL __launch_bounds__(MKL_REDUCE_THREADS)
mkl_reduce_pass2(const MklWgslReduceParams* u, const float* rx, float2* rpartials, float2* rsums) {
    __shared__ float2 v[MKL_REDUCE_THREADS];
    const MklWgslReduceParams p = *u;
    const uint tid = threadIdx.x;
    const uint channel = blockIdx.y;
    float2 acc = float2(0.0f);
    for (uint j = tid; j < p.num_groups; j += MKL_REDUCE_THREADS)
        acc = mkl_reduce_combine(acc, rpartials[p.partial_offset + channel * p.num_groups + j], p.mode);
    v[tid] = acc;
    const float2 r = mkl_group_tree(v, tid, p.mode);
    if (tid == 0u) rsums[p.sum_offset + channel] = r;
}
