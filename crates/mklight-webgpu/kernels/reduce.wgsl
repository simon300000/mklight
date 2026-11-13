// Deterministic compensated two-pass reduction (reduce.metal's twin,
// specified by mklight_gpu::reduce). The order depends only on the data
// size: MKL_REDUCE_THREADS threads per group, a grid-stride accumulation per
// thread, a binary tree over the group in workgroup memory (strides
// T/2 .. 1; Metal runs strides below 32 as SIMD shuffles, which is the same
// order) and a single-group tree over the partials. No atomics, no
// subgroups.
//
// Pass 1: grid (k, channels), groups first_group .. first_group + k - 1.
// Pass 2: grid (1, channels).
// The host binds x, partials and sums at word offsets in the params.

struct MklWgslReduceParams {
    count: u32,
    channel_stride: u32,
    num_groups: u32,
    first_group: u32,
    mode: u32,
    // Offsets (elements) of the launch's first channel in x, partials, sums.
    x_offset: u32,
    partial_offset: u32,
    sum_offset: u32,
}
@group(0) @binding(1) var<uniform> rp: MklWgslReduceParams;
@group(0) @binding(2) var<storage, read> rx: array<f32>;
@group(0) @binding(3) var<storage, read_write> rpartials: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> rsums: array<vec2<f32>>;

var<workgroup> rv: array<vec2<f32>, MKL_REDUCE_THREADS>;

fn mkl_reduce_combine(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (rp.mode == MKL_REDUCE_PLAIN) {
        return vec2<f32>(mkl_add(a.x, b.x), 0.0);
    }
    return mkl_df_add(a, b);
}

// Binary tree over the group's values, strides T/2 .. 1. Every thread calls
// it (uniform control flow for the barriers); thread 0's result is final.
fn mkl_group_tree(tid: u32) -> vec2<f32> {
    var stride = MKL_REDUCE_THREADS / 2u;
    loop {
        workgroupBarrier();
        if (tid < stride) {
            rv[tid] = mkl_reduce_combine(rv[tid], rv[tid + stride]);
        }
        if (stride == 1u) {
            break;
        }
        stride = stride >> 1u;
    }
    workgroupBarrier();
    return rv[0];
}

@compute @workgroup_size(MKL_REDUCE_THREADS)
fn mkl_reduce_pass1(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let tid = lid.x;
    let channel = group.y;
    let base = rp.x_offset + channel * rp.channel_stride;
    let stride = rp.num_groups * MKL_REDUCE_THREADS;
    let gidx = rp.first_group + group.x;
    let g = gidx * MKL_REDUCE_THREADS + tid;
    var acc: vec2<f32>;
    if (rp.mode == MKL_REDUCE_PLAIN) {
        var s = 0.0;
        for (var i = g; i < rp.count; i += stride) {
            s = mkl_add(s, rx[base + i]);
        }
        acc = vec2<f32>(s, 0.0);
    } else if (rp.mode == MKL_REDUCE_KAHAN) {
        var sc = vec2<f32>(0.0, 0.0);
        for (var i = g; i < rp.count; i += stride) {
            sc = mkl_neumaier_add(sc, rx[base + i]);
        }
        acc = mkl_two_sum(sc.x, sc.y);
    } else {
        acc = vec2<f32>(0.0, 0.0);
        for (var i = g; i < rp.count; i += stride) {
            acc = mkl_df_add_f32(acc, rx[base + i]);
        }
    }
    rv[tid] = acc;
    let r = mkl_group_tree(tid);
    if (tid == 0u) {
        rpartials[rp.partial_offset + channel * rp.num_groups + gidx] = r;
    }
}

@compute @workgroup_size(MKL_REDUCE_THREADS)
fn mkl_reduce_pass2(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let tid = lid.x;
    let channel = group.y;
    var acc = vec2<f32>(0.0, 0.0);
    for (var j = tid; j < rp.num_groups; j += MKL_REDUCE_THREADS) {
        acc = mkl_reduce_combine(acc, rpartials[rp.partial_offset + channel * rp.num_groups + j]);
    }
    rv[tid] = acc;
    let r = mkl_group_tree(tid);
    if (tid == 0u) {
        rsums[rp.sum_offset + channel] = r;
    }
}
