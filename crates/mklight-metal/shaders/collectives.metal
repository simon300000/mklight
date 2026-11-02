// Fixed-order resident collectives; mklight_gpu::collectives is the CPU
// specification. Word-indexed bindings allow element-aligned views.
#include "mklight_common.h"

static inline float2 mkl_c_load(device const uint* a, uint i, uint words) {
    return mkl_df_input(float2(as_type<float>(a[i * words]), words == 2u ? as_type<float>(a[i * words + 1u]) : 0.0f));
}
static inline void mkl_c_store(device uint* a, uint i, float2 v) {
    a[2u * i] = as_type<uint>(v.x);
    a[2u * i + 1u] = as_type<uint>(v.y);
}
// Exact double-float of a u32.
static inline float2 mkl_c_uint(uint i) {
    return mkl_df_add(float2(float(i >> 16) * 65536.0f, 0.0f), float2(float(i & 65535u), 0.0f));
}
// Order-preserving u32 of a float: signed zeros tie, NaNs sort last.
static inline uint mkl_c_fkey(float x) {
    uint b = as_type<uint>(x), m = b & 0x7fffffffu;
    uint k = (b & 0x80000000u) != 0u && m != 0u ? 0x7fffffffu - m : m | 0x80000000u;
    return m > 0x7f800000u ? 0xffffffffu : k;
}
// Three-way numeric order of double-float values; NaN in either word sorts last.
static inline int mkl_c_cmp(float2 a, float2 b) {
    bool an = isnan(a.x) || isnan(a.y), bn = isnan(b.x) || isnan(b.y);
    uint2 ka = an ? uint2(0xffffffffu) : uint2(mkl_c_fkey(a.x), mkl_c_fkey(a.y));
    uint2 kb = bn ? uint2(0xffffffffu) : uint2(mkl_c_fkey(b.x), mkl_c_fkey(b.y));
    bool less = ka.x < kb.x || (ka.x == kb.x && ka.y < kb.y);
    bool greater = ka.x > kb.x || (ka.x == kb.x && ka.y > kb.y);
    return int(greater) - int(less);
}
// Extremum keeping the first operand on ties; the first NaN wins.
static inline float2 mkl_c_ext(float2 a, float2 b, bool maximum) {
    if (isnan(a.x)) return a;
    if (isnan(b.x)) return b;
    int c = mkl_c_cmp(a, b);
    return (maximum ? c < 0 : c > 0) ? b : a;
}

// ---------------------------------------------------------------------------
// Fixed reduction tree (mklight_gpu::reduce): 256 threads, binary tree with
// strides 128..1. Strides below 32 run as SIMD shuffles, bit-identically.
// ---------------------------------------------------------------------------
static inline float2 mkl_c_combine(float2 a, float2 b, uint op) {
    if (op >= MKL_COLLECT_MIN) return mkl_c_ext(a, b, op == MKL_COLLECT_MAX);
    return op == MKL_REDUCE_PLAIN ? float2(a.x + b.x, 0.0f) : mkl_df_add(a, b);
}
struct MklAcc {
    float2 v;
    float c;
};
static inline void mkl_c_acc(thread MklAcc& a, float2 x, uint op) {
    if (op >= MKL_COLLECT_MIN) {
        a.v = mkl_c_ext(a.v, x, op == MKL_COLLECT_MAX);
    } else if (op == MKL_REDUCE_PLAIN) {
        a.v.x += x.x + x.y;
    } else if (op == MKL_REDUCE_KAHAN) {
        float sum = a.v.x;
        mkl_neumaier_add(sum, a.c, x.x);
        if (x.y != 0.0f) mkl_neumaier_add(sum, a.c, x.y);
        a.v.x = sum;
    } else {
        a.v = mkl_df_add(a.v, x);
    }
}
static inline float2 mkl_c_value(MklAcc a, uint op) {
    return op == MKL_REDUCE_KAHAN ? mkl_two_sum(a.v.x, a.c) : a.v;
}
// Strides 16..1 inside a SIMD group; the result is valid in lane 0.
static inline float2 mkl_c_simd_levels(float2 v, uint op) {
    for (ushort s = 16; s > 0; s >>= 1) v = mkl_c_combine(v, simd_shuffle_down(v, s), op);
    return v;
}
// One 256-thread tree emulated by a SIMD group: t[j] is thread lane + 32j.
// Threads from `used` on hold the identity, and combining with the identity
// returns the other operand unchanged, so those combines are skipped.
static inline float2 mkl_c_tree8(thread float2* t, uint used, uint op) {
    for (uint j = 0u; j + 4u < used; ++j) t[j] = mkl_c_combine(t[j], t[j + 4u], op);
    if (used > 2u) t[0] = mkl_c_combine(t[0], t[2], op);
    if (used > 3u) t[1] = mkl_c_combine(t[1], t[3], op);
    return mkl_c_simd_levels(used > 1u ? mkl_c_combine(t[0], t[1], op) : t[0], op);
}
// The same tree over 256 threads; valid in thread 0.
static inline float2 mkl_c_tree256(threadgroup float2* t, float2 v, uint tid, uint op) {
    t[tid] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 128u) t[tid] = mkl_c_combine(t[tid], t[tid + 128u], op);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 64u) t[tid] = mkl_c_combine(t[tid], t[tid + 64u], op);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 32u) v = mkl_c_combine(t[tid], t[tid + 32u], op);
    return mkl_c_simd_levels(v, op);
}
static inline uint mkl_c_groups(uint n) {
    return clamp((n + MKL_REDUCE_THREADS * MKL_REDUCE_ITEMS_PER_THREAD - 1u) /
                     (MKL_REDUCE_THREADS * MKL_REDUCE_ITEMS_PER_THREAD), 1u, MKL_REDUCE_MAX_GROUPS);
}

// ---------------------------------------------------------------------------
// Segmented reductions. Terms per kind and phase are listed in SegmentKind.
// ---------------------------------------------------------------------------
struct MklSegIo {
    device const uint* vals;
    device const uint* refs;
    device const uint* mask;
    device const uint* weights;
};
static inline uint mkl_c_seg_terms(constant MklSegmentParams& p, uint phase) {
    if (p.kind == MKL_COLLECT_MOMENTS2) return phase == 0u ? 4u : 6u;
    return p.kind == MKL_COLLECT_SUMMARY ? 4u * p.channels : p.channels;
}
static inline uint mkl_c_seg_op(constant MklSegmentParams& p, uint phase, uint term) {
    if (p.kind == MKL_COLLECT_SUMMARY) {
        uint k = term % 4u;
        return k == 0u ? p.mode : (k == 1u ? MKL_REDUCE_DF : MKL_COLLECT_MIN + k - 2u);
    }
    if (p.kind == MKL_COLLECT_MOMENTS2) {
        if (phase == 0u && term == 0u) return MKL_REDUCE_DF;
        if (phase == 1u && term >= 4u) return MKL_COLLECT_MAX;
    }
    return p.mode;
}
static inline float2 mkl_c_seg_identity(constant MklSegmentParams& p, uint op) {
    if (op == MKL_COLLECT_MIN) return float2(INFINITY, 0.0f);
    if (op == MKL_COLLECT_MAX && p.kind == MKL_COLLECT_SUMMARY) return float2(-INFINITY, 0.0f);
    return float2(0.0f);
}
// Term value of element i of segment seg; masked records give the identity
// without loading their (possibly non-finite) coordinates.
static inline float2 mkl_c_seg_value(constant MklSegmentParams& p, MklSegIo io, uint seg, uint phase,
                                     uint term, uint op, uint i, float4 mean) {
    if (p.masked != 0u && io.mask[i] != p.mask_value) return mkl_c_seg_identity(p, op);
    if (p.kind == MKL_COLLECT_SUMMARY) {
        return term % 4u == 1u ? float2(1.0f, 0.0f) : mkl_c_load(io.vals, (term / 4u) * p.count + i, p.value_words);
    }
    if (p.kind != MKL_COLLECT_MOMENTS2) return mkl_c_load(io.vals, term * p.count + i, p.value_words);
    float2 w = p.weight_words != 0u ? mkl_c_load(io.weights, i, p.weight_words)
             : (p.channels == 3u ? mkl_c_load(io.vals, 2u * p.count + i, p.value_words) : float2(1.0f, 0.0f));
    if (phase == 0u && term < 2u) return term == 0u ? float2(1.0f, 0.0f) : w;
    float2 x = mkl_c_load(io.vals, i, p.value_words), y = mkl_c_load(io.vals, p.count + i, p.value_words);
    if (p.reference_words != 0u) {
        x = mkl_df_sub(x, mkl_c_load(io.refs, 2u * seg, p.reference_words));
        y = mkl_df_sub(y, mkl_c_load(io.refs, 2u * seg + 1u, p.reference_words));
    }
    if (phase == 0u) return mkl_df_mul(w, term == 2u ? x : y);
    if (term == 0u || term == 4u) {
        float2 r = mkl_df_add(mkl_df_mul(x, x), mkl_df_mul(y, y));
        return term == 0u ? mkl_df_mul(w, r) : (w.x > 0.0f ? r : float2(0.0f));
    }
    x = mkl_df_sub(x, mean.xy);
    y = mkl_df_sub(y, mean.zw);
    if (term == 5u) return w.x > 0.0f ? mkl_df_add(mkl_df_mul(x, x), mkl_df_mul(y, y)) : float2(0.0f);
    return mkl_df_mul(w, term == 1u ? mkl_df_mul(x, x) : (term == 2u ? mkl_df_mul(x, y) : mkl_df_mul(y, y)));
}
static inline uint mkl_c_seg_slot(constant MklSegmentParams& p, uint seg, uint phase, uint term) {
    if (p.kind == MKL_COLLECT_SUM) return term * p.segments + seg;
    if (p.kind == MKL_COLLECT_SUMMARY) return ((term / 4u) * p.segments + seg) * 4u + term % 4u;
    if (p.kind == MKL_COLLECT_MOMENTS2) return seg * 10u + phase * 4u + term;
    return seg * (p.channels + p.channels * (p.channels + 1u) / 2u) + term;
}
// After a phase's sums: moments form the means, source products the
// upper-triangular products of the source sums. Returns the means.
static inline float4 mkl_c_seg_finish(device uint* out, constant MklSegmentParams& p, uint seg, uint phase) {
    if (p.kind == MKL_COLLECT_MOMENTS2 && phase == 0u) {
        float2 w = mkl_c_load(out, seg * 10u + 1u, 2u);
        float4 mean = float4(NAN, 0.0f, NAN, 0.0f);
        if (w.x > 0.0f) {
            mean.xy = mkl_df_div(mkl_c_load(out, seg * 10u + 2u, 2u), w);
            mean.zw = mkl_df_div(mkl_c_load(out, seg * 10u + 3u, 2u), w);
        }
        mkl_c_store(out, seg * 10u + 2u, mean.xy);
        mkl_c_store(out, seg * 10u + 3u, mean.zw);
        return mean;
    }
    if (p.kind == MKL_COLLECT_SOURCE_PRODUCTS) {
        uint c = p.channels, base = seg * (c + c * (c + 1u) / 2u), dest = base + c;
        for (uint a = 0u; a < c; ++a)
            for (uint b = a; b < c; ++b)
                mkl_c_store(out, dest++, mkl_df_mul(mkl_c_load(out, base + a, 2u), mkl_c_load(out, base + b, 2u)));
    }
    return float4(0.0f);
}
static inline void mkl_c_seg_invalid(device uint* out, constant MklSegmentParams& p, uint seg) {
    for (uint ph = 0u; ph < (p.kind == MKL_COLLECT_MOMENTS2 ? 2u : 1u); ++ph)
        for (uint t = 0u; t < mkl_c_seg_terms(p, ph); ++t) mkl_c_store(out, mkl_c_seg_slot(p, seg, ph, t), float2(NAN, 0.0f));
    if (p.kind == MKL_COLLECT_SOURCE_PRODUCTS)
        for (uint t = p.channels; t < p.channels + p.channels * (p.channels + 1u) / 2u; ++t)
            mkl_c_store(out, mkl_c_seg_slot(p, seg, 0u, t), float2(NAN, 0.0f));
}
static inline bool mkl_c_seg_range(device const uint* offsets, constant MklSegmentParams& p, uint seg,
                                   thread uint& a, thread uint& n) {
    uint b = offsets[seg + 1u];
    a = offsets[seg];
    bool valid = a <= b && b <= p.count && b - a <= p.max_len;
    n = valid ? b - a : 0u;
    return valid;
}

// One term of one segment (at most one tree group) by a SIMD group.
template <uint OP>
static inline float2 mkl_c_seg_simd(constant MklSegmentParams& p, MklSegIo io, uint seg, uint phase, uint term,
                                    uint a, uint n, uint lane, float4 mean) {
    float2 t[8];
    uint used = min(8u, (n + 31u) / 32u);
    for (uint j = 0u; j < max(used, 1u); ++j) {
        MklAcc acc = {mkl_c_seg_identity(p, OP), 0.0f};
        for (uint i = lane + 32u * j; i < n; i += 256u)
            mkl_c_acc(acc, mkl_c_seg_value(p, io, seg, phase, term, OP, a + i, mean), OP);
        t[j] = mkl_c_value(acc, OP);
    }
    return mkl_c_tree8(t, used, OP);
}
// Segments of at most 4096 records (one tree group): one SIMD group per
// segment computes every term and phase without partials.
kernel void mkl_c_seg_short(device const uint* vals [[buffer(0)]], device const uint* offsets [[buffer(1)]],
                            device const uint* refs [[buffer(2)]], device const uint* mask [[buffer(3)]],
                            device const uint* weights [[buffer(4)]], device uint* out [[buffer(5)]],
                            device uint* status [[buffer(6)]], constant MklSegmentParams& p [[buffer(7)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint seg = p.first + group * 8u + simd;
    if (seg >= p.end) return;
    uint a, n;
    bool valid = mkl_c_seg_range(offsets, p, seg, a, n);
    if (lane == 0u) status[seg] = valid ? 0u : 1u;
    if (!valid) {
        if (lane == 0u) mkl_c_seg_invalid(out, p, seg);
        return;
    }
    MklSegIo io = {vals, refs, mask, weights};
    float4 mean = float4(0.0f);
    for (uint phase = 0u; phase < (p.kind == MKL_COLLECT_MOMENTS2 ? 2u : 1u); ++phase) {
        for (uint term = 0u; term < mkl_c_seg_terms(p, phase); ++term) {
            float2 v;
            switch (mkl_c_seg_op(p, phase, term)) {
            case MKL_REDUCE_PLAIN: v = mkl_c_seg_simd<MKL_REDUCE_PLAIN>(p, io, seg, phase, term, a, n, lane, mean); break;
            case MKL_REDUCE_KAHAN: v = mkl_c_seg_simd<MKL_REDUCE_KAHAN>(p, io, seg, phase, term, a, n, lane, mean); break;
            case MKL_REDUCE_DF: v = mkl_c_seg_simd<MKL_REDUCE_DF>(p, io, seg, phase, term, a, n, lane, mean); break;
            case MKL_COLLECT_MIN: v = mkl_c_seg_simd<MKL_COLLECT_MIN>(p, io, seg, phase, term, a, n, lane, mean); break;
            default: v = mkl_c_seg_simd<MKL_COLLECT_MAX>(p, io, seg, phase, term, a, n, lane, mean); break;
            }
            if (lane == 0u) mkl_c_store(out, mkl_c_seg_slot(p, seg, phase, term), v);
        }
        if (lane == 0u) mean = mkl_c_seg_finish(out, p, seg, phase);
        mean = simd_broadcast(mean, 0);
    }
}
// Long segments, pass 1: grid (max_groups, terms, segments); one tree group
// each, partials [segment, term, max_groups]. Phase 1 reads phase-0 means.
kernel void mkl_c_seg_pass1(device const uint* vals [[buffer(0)]], device const uint* offsets [[buffer(1)]],
                            device const uint* refs [[buffer(2)]], device const uint* mask [[buffer(3)]],
                            device const uint* weights [[buffer(4)]], device const uint* out [[buffer(5)]],
                            device uint* partials [[buffer(6)]], constant MklSegmentParams& p [[buffer(7)]],
                            uint3 group [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 tree[256];
    uint g = group.x, term = group.y, seg = p.first + group.z, a, n;
    if (!mkl_c_seg_range(offsets, p, seg, a, n)) return;
    uint G = mkl_c_groups(n);
    if (g >= G) return;
    uint op = mkl_c_seg_op(p, p.phase, term);
    float4 mean = float4(0.0f);
    if (p.phase == 1u) mean = float4(mkl_c_load(out, seg * 10u + 2u, 2u), mkl_c_load(out, seg * 10u + 3u, 2u));
    MklSegIo io = {vals, refs, mask, weights};
    MklAcc acc = {mkl_c_seg_identity(p, op), 0.0f};
    for (uint i = g * 256u + tid; i < n; i += G * 256u)
        mkl_c_acc(acc, mkl_c_seg_value(p, io, seg, p.phase, term, op, a + i, mean), op);
    float2 v = mkl_c_tree256(tree, mkl_c_value(acc, op), tid, op);
    if (tid == 0u) mkl_c_store(partials, (group.z * p.terms + term) * p.max_groups + g, v);
}
// Long segments, pass 2: one SIMD group per segment folds the partials.
kernel void mkl_c_seg_pass2(device const uint* offsets [[buffer(0)]], device const uint* partials [[buffer(1)]],
                            device uint* out [[buffer(2)]], device uint* status [[buffer(3)]],
                            constant MklSegmentParams& p [[buffer(4)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint simd [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint seg = p.first + group * 8u + simd, a, n;
    if (seg >= p.end) return;
    bool valid = mkl_c_seg_range(offsets, p, seg, a, n);
    if (lane == 0u) status[seg] = valid ? 0u : 1u;
    if (!valid) {
        if (lane == 0u) mkl_c_seg_invalid(out, p, seg);
        return;
    }
    uint G = mkl_c_groups(n), local = seg - p.first;
    for (uint term = 0u; term < p.terms; ++term) {
        uint op = mkl_c_seg_op(p, p.phase, term);
        float2 id = mkl_c_seg_identity(p, op), t[8];
        uint used = (G + 31u) / 32u;
        for (uint j = 0u; j < used; ++j) {
            uint q = lane + 32u * j;
            t[j] = q < G ? mkl_c_combine(id, mkl_c_load(partials, (local * p.terms + term) * p.max_groups + q, 2u), op) : id;
        }
        float2 v = mkl_c_tree8(t, used, op);
        if (lane == 0u) mkl_c_store(out, mkl_c_seg_slot(p, seg, p.phase, term), v);
    }
    if (lane == 0u) mkl_c_seg_finish(out, p, seg, p.phase);
}

// ---------------------------------------------------------------------------
// Scans: 256-element Hillis-Steele blocks, recursively scanned block totals,
// then one finishing pass. kind: 0 u32 (flag 1: x != 0), 1 f32, 2 df.
// ---------------------------------------------------------------------------
kernel void mkl_c_scan_blocks(device const uint* src [[buffer(0)]], device uint* dst [[buffer(1)]],
                              device uint* totals [[buffer(2)]], constant MklCollectParams& p [[buffer(3)]],
                              uint tid [[thread_index_in_threadgroup]], uint g [[threadgroup_position_in_grid]]) {
    threadgroup float2 v[256];
    threadgroup uint u[256];
    uint i = g * 256u + tid;
    bool integer = p.kind == 0u;
    if (integer) {
        uint x = i < p.n ? src[i] : 0u;
        u[tid] = p.flag != 0u ? uint(x != 0u) : x;
    } else {
        v[tid] = i < p.n ? mkl_c_load(src, i, p.kind) : float2(0.0f);
    }
    for (uint step = 1u; step < 256u; step <<= 1u) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint pu = 0u;
        float2 pv = float2(0.0f);
        if (tid >= step) {
            if (integer) pu = u[tid - step];
            else pv = v[tid - step];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid >= step) {
            if (integer) u[tid] = pu + u[tid];
            else v[tid] = mkl_df_add(pv, v[tid]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i < p.n) {
        if (integer) dst[i] = u[tid];
        else mkl_c_store(dst, i, v[tid]);
    }
    if (tid == 255u) {
        if (integer) totals[g] = u[tid];
        else mkl_c_store(totals, g, v[tid]);
    }
}
// out[i] = inclusive prefix at i (flag 0) or i - 1 (flag 1, exclusive).
kernel void mkl_c_scan_finish(device const uint* local [[buffer(0)]], device const uint* prefix [[buffer(1)]],
                              device uint* out [[buffer(2)]], constant MklCollectParams& p [[buffer(3)]],
                              uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    bool integer = p.kind == 0u;
    if (p.flag != 0u && i == 0u) {
        if (integer) out[0] = 0u;
        else mkl_c_store(out, 0u, float2(0.0f));
        return;
    }
    uint j = i - p.flag, b = j / 256u;
    if (integer) {
        out[i] = local[j] + (b > 0u ? prefix[b - 1u] : 0u);
    } else {
        float2 v = mkl_c_load(local, j, 2u);
        mkl_c_store(out, i, b > 0u ? mkl_df_add(mkl_c_load(prefix, b - 1u, 2u), v) : v);
    }
}
// Stable compaction from the inclusive count of nonzero mask words. Kept
// records and the cleared tail write disjoint slots.
kernel void mkl_c_compact(device const uint* src [[buffer(0)]], device const uint* ids [[buffer(1)]],
                          device const uint* mask [[buffer(2)]], device const uint* prefix [[buffer(3)]],
                          device uint* dst [[buffer(4)]], device uint* outids [[buffer(5)]],
                          device uint* count [[buffer(6)]], constant MklCollectParams& p [[buffer(7)]],
                          uint i [[thread_position_in_grid]]) {
    uint kept = p.n == 0u ? 0u : prefix[p.n - 1u];
    if (i == 0u) count[0] = kept;
    if (i >= p.n) return;
    if (i >= kept) {
        outids[i] = 0xffffffffu;
        for (uint w = 0u; w < p.words; ++w) dst[i * p.words + w] = 0u;
    }
    if (mask[i] != 0u) {
        uint j = prefix[i] - 1u;
        outids[j] = ids[i];
        for (uint w = 0u; w < p.words; ++w) dst[j * p.words + w] = src[i * p.words + w];
    }
}

// ---------------------------------------------------------------------------
// Stable sort: LSD radix sort of records (key words, logical ID, position)
// in 8-bit digits, ID digits first. Every record has a distinct position, so
// the result is the unique (key, ID, position) order.
// ---------------------------------------------------------------------------
kernel void mkl_c_sort_init(device const uint* keys [[buffer(0)]], device const uint* ids [[buffer(1)]],
                            device uint4* rec [[buffer(2)]], constant MklCollectParams& p [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
    if (i >= p.n) return;
    uint k0 = keys[i * p.words], k1 = 0u;
    if (p.kind == 1u) {
        k0 = mkl_c_fkey(as_type<float>(k0));
    } else if (p.kind == 2u) {
        float2 k = mkl_c_load(keys, i, 2u);
        bool nan = isnan(k.x) || isnan(k.y);
        k0 = nan ? 0xffffffffu : mkl_c_fkey(k.x);
        k1 = nan ? 0xffffffffu : mkl_c_fkey(k.y);
    }
    rec[i] = uint4(k0, k1, ids[i], i);
}
static inline uint mkl_c_digit(uint4 r, constant MklCollectParams& p) {
    return (r[p.flag] >> p.shift) & 255u;
}
// Block digit counts, digit-major: hist[digit * blocks + block].
kernel void mkl_c_radix_count(device const uint4* rec [[buffer(0)]], device uint* hist [[buffer(1)]],
                              constant MklCollectParams& p [[buffer(2)]],
                              uint tid [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]]) {
    threadgroup atomic_uint h[256];
    atomic_store_explicit(&h[tid], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint r = 0u; r < 4u; ++r) {
        uint i = block * MKL_COLLECT_SORT_BLOCK + r * 256u + tid;
        if (i < p.n) atomic_fetch_add_explicit(&h[mkl_c_digit(rec[i], p)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    hist[tid * p.blocks + block] = atomic_load_explicit(&h[tid], memory_order_relaxed);
}
static inline uint mkl_c_ballot(bool b) {
    return uint(ulong(simd_ballot(b)));
}
// Stable scatter: block-local ranks follow block order (round, SIMD group,
// lane); start[digit * blocks + block] is the exclusive digit-major scan.
kernel void mkl_c_radix_scatter(device const uint4* src [[buffer(0)]], device const uint* start [[buffer(1)]],
                                device uint4* dst [[buffer(2)]], constant MklCollectParams& p [[buffer(3)]],
                                uint tid [[thread_index_in_threadgroup]], uint block [[threadgroup_position_in_grid]],
                                uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup uint counts[8][256];
    threadgroup uint base[256];
    base[tid] = start[tid * p.blocks + block];
    for (uint r = 0u; r < 4u; ++r) {
        for (uint s = 0u; s < 8u; ++s) counts[s][tid] = 0u;
        uint i = block * MKL_COLLECT_SORT_BLOCK + r * 256u + tid;
        bool valid = i < p.n;
        uint4 rec = valid ? src[i] : uint4(0u);
        uint d = mkl_c_digit(rec, p), peers = mkl_c_ballot(valid);
        for (uint b = 0u; b < 8u; ++b) {
            bool bit = ((d >> b) & 1u) != 0u;
            uint vote = mkl_c_ballot(bit);
            peers &= bit ? vote : ~vote;
        }
        uint rank = popcount(peers & ((1u << lane) - 1u));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid && rank == 0u) counts[simd][d] = popcount(peers);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint run = base[tid];
        for (uint s = 0u; s < 8u; ++s) {
            uint c = counts[s][tid];
            counts[s][tid] = run;
            run += c;
        }
        base[tid] = run;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (valid) dst[counts[simd][d] + rank] = rec;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
kernel void mkl_c_sort_gather(device const uint* keys [[buffer(0)]], device const uint* ids [[buffer(1)]],
                              device const uint* vals [[buffer(2)]], device const uint4* rec [[buffer(3)]],
                              device uint* outkeys [[buffer(4)]], device uint* outids [[buffer(5)]],
                              device uint* outvals [[buffer(6)]], constant MklCollectParams& p [[buffer(7)]],
                              uint j [[thread_position_in_grid]]) {
    if (j >= p.n) return;
    uint i = rec[j].w;
    outids[j] = ids[i];
    for (uint w = 0u; w < p.words; ++w) outkeys[j * p.words + w] = keys[i * p.words + w];
    for (uint w = 0u; w < p.value_words; ++w) outvals[j * p.value_words + w] = vals[i * p.value_words + w];
}

// ---------------------------------------------------------------------------
// Pivoted QR (modified Gram-Schmidt, two projections per step). Work is
// column-major [batch, cols + 1, rows]; column `cols` is the right-hand side.
// aux per batch: residual norms [cols + 1], Q^T b [cols], first pivot norm,
// row sums of |R| and |R^-1| [cols each], reorthogonalisation dots [cols].
// Q columns are also kept column-major [batch, cols, rows].
// ---------------------------------------------------------------------------
static inline uint mkl_c_qf(constant MklQrParams& p, uint b, uint i, uint j) {
    return (b * (p.rows + p.cols + 2u) + i) * p.cols + j;
}
static inline uint mkl_c_qw(constant MklQrParams& p, uint b, uint j, uint i) {
    return (b * (p.cols + 1u) + j) * p.rows + i;
}
static inline float mkl_c_maxnan(float a, float b) {
    return (isnan(a) || isnan(b)) ? NAN : max(a, b);
}
// Sum over the group's threads: SIMD levels, then the same levels over the
// zero-padded SIMD partials. Every thread receives the result.
static inline float2 mkl_c_qr_sum(threadgroup float2* s, float2 v, uint simd, uint lane, uint simds) {
    v = mkl_c_simd_levels(v, MKL_REDUCE_DF);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) s[simd] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        v = mkl_c_simd_levels(lane < simds ? s[lane] : float2(0.0f), MKL_REDUCE_DF);
        if (lane == 0u) s[32] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s[32];
}
static inline float mkl_c_qr_max(threadgroup float2* s, float v, uint simd, uint lane, uint simds) {
    for (ushort d = 16; d > 0; d >>= 1) v = mkl_c_maxnan(v, simd_shuffle_down(v, d));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0u) s[simd].x = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd == 0u) {
        v = lane < simds ? s[lane].x : 0.0f;
        for (ushort d = 16; d > 0; d >>= 1) v = mkl_c_maxnan(v, simd_shuffle_down(v, d));
        if (lane == 0u) s[32].x = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s[32].x;
}
// x * 2^e through two normal powers of two (exact without underflow).
static inline float2 mkl_c_ldexp(float2 x, int e) {
    int h = e / 2;
    return x * as_type<float>(uint(127 + h) << 23) * as_type<float>(uint(127 + e - h) << 23);
}
// Exponent e with |x| = f 2^e, f in [0.5, 1), from the bits.
static inline int mkl_c_exponent(float x) {
    return int((as_type<uint>(x) >> 23) & 255u) - 126;
}
// 2-norm of work column j scaled by the exponent of its maximum; NaN when
// any entry is non-finite.
static inline float2 mkl_c_qr_norm(device const uint* work, constant MklQrParams& p, uint b, uint j, float mx,
                                   threadgroup float2* s, uint tid, uint simd, uint lane) {
    if (!isfinite(mx)) return float2(NAN, 0.0f);
    if (mx == 0.0f) return float2(0.0f);
    int e = mkl_c_exponent(mx);
    float2 acc = float2(0.0f);
    for (uint i = tid; i < p.rows; i += p.threads) {
        float2 v = mkl_c_ldexp(mkl_c_load(work, mkl_c_qw(p, b, j, i), 2u), -e);
        acc = mkl_df_add(acc, mkl_df_mul(v, v));
    }
    return mkl_c_ldexp(mkl_df_sqrt(mkl_c_qr_sum(s, acc, simd, lane, p.threads / 32u)), e);
}
static inline float mkl_c_qr_colmax(device const uint* work, constant MklQrParams& p, uint b, uint j,
                                    threadgroup float2* s, uint tid, uint simd, uint lane) {
    float mx = 0.0f;
    for (uint i = tid; i < p.rows; i += p.threads) mx = mkl_c_maxnan(mx, fabs(mkl_c_load(work, mkl_c_qw(p, b, j, i), 2u).x));
    return mkl_c_qr_max(s, mx, simd, lane, p.threads / 32u);
}
// Grid (cols + 1, batch). Copies a column, records its norm as the scale,
// equilibrates it and publishes the scaled norm; clears its Q/R columns.
kernel void mkl_c_qr_prepare(device const uint* a [[buffer(0)]], device const uint* rhs [[buffer(1)]],
                             device uint* work [[buffer(2)]], device uint* factors [[buffer(3)]],
                             device uint* aux [[buffer(4)]], device uint* diag [[buffer(5)]],
                             constant MklQrParams& p [[buffer(6)]], uint2 group [[threadgroup_position_in_grid]],
                             uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float2 s[33];
    uint j = group.x, b = group.y, m = p.rows, n = p.cols, aux0 = b * p.aux_stride;
    for (uint i = tid; i < m; i += p.threads) {
        uint src = b * m + i;
        float2 v = j == n ? mkl_c_load(rhs, src, p.input_words) : mkl_c_load(a, src * n + j, p.input_words);
        mkl_c_store(work, mkl_c_qw(p, b, j, i), v);
    }
    float2 scale = mkl_c_qr_norm(work, p, b, j, mkl_c_qr_colmax(work, p, b, j, s, tid, simd, lane), s, tid, simd, lane);
    if (j == n) {
        if (tid == 0u) {
            mkl_c_store(aux, aux0 + n, scale);
            mkl_c_store(aux, aux0 + 2u * n + 1u, float2(0.0f));
            mkl_c_store(diag, 4u * b, float2(0.0f));
            mkl_c_store(diag, 4u * b + 1u, float2(INFINITY, 0.0f));
            mkl_c_store(diag, 4u * b + 2u, float2(0.0f));
            mkl_c_store(diag, 4u * b + 3u, float2(0.0f));
        }
        return;
    }
    for (uint i = tid; i < m + n; i += p.threads) mkl_c_store(factors, mkl_c_qf(p, b, i, j), float2(0.0f));
    if (tid == 0u) {
        mkl_c_store(factors, mkl_c_qf(p, b, m + n, j), scale);
        mkl_c_store(factors, mkl_c_qf(p, b, m + n + 1u, j), mkl_c_uint(j));
        mkl_c_store(aux, aux0 + n + 1u + j, float2(0.0f));
    }
    if (scale.x != 0.0f) {
        for (uint i = tid; i < m; i += p.threads) {
            uint w = mkl_c_qw(p, b, j, i);
            mkl_c_store(work, w, mkl_df_div(mkl_c_load(work, w, 2u), scale));
        }
    }
    float2 norm = mkl_c_qr_norm(work, p, b, j, mkl_c_qr_colmax(work, p, b, j, s, tid, simd, lane), s, tid, simd, lane);
    if (tid == 0u) mkl_c_store(aux, aux0 + j, norm);
}
// Grid (batch). Chooses pivot `step` (largest residual norm, then lowest
// original column), checks rank/finiteness and swaps it into place.
kernel void mkl_c_qr_pivot(device uint* work [[buffer(0)]], device uint* factors [[buffer(1)]],
                           device uint* aux [[buffer(2)]], device uint* diag [[buffer(3)]],
                           constant MklQrParams& p [[buffer(4)]],
                           uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup uint chosen[1];
    uint m = p.rows, n = p.cols, k = p.step, aux0 = b * p.aux_stride;
    if (as_type<float>(diag[8u * b + 6u]) != 0.0f) return;
    if (tid == 0u) {
        uint best = k;
        for (uint j = k + 1u; j < n; ++j) {
            int c = mkl_c_cmp(mkl_c_load(aux, aux0 + j, 2u), mkl_c_load(aux, aux0 + best, 2u));
            if (c > 0 || (c == 0 && mkl_c_load(factors, mkl_c_qf(p, b, m + n + 1u, j), 2u).x <
                                        mkl_c_load(factors, mkl_c_qf(p, b, m + n + 1u, best), 2u).x))
                best = j;
        }
        float2 d = mkl_c_load(aux, aux0 + best, 2u);
        if (k == 0u) mkl_c_store(aux, aux0 + 2u * n + 1u, d);
        bool finite = isfinite(d.x) && isfinite(d.y);
        for (uint j = 0u; k == 0u && j < n; ++j) {
            float2 s = mkl_c_load(factors, mkl_c_qf(p, b, m + n, j), 2u);
            finite = finite && isfinite(s.x) && isfinite(s.y);
        }
        float tol = as_type<float>(p.tolerance_bits);
        float limit = mkl_df_mul_f32(mkl_c_load(aux, aux0 + 2u * n + 1u, 2u), tol).x;
        chosen[0] = UINT_MAX;
        if (!finite) {
            mkl_c_store(diag, 4u * b + 3u, float2(2.0f, 0.0f));
        } else if (d.x <= limit) {
            mkl_c_store(diag, 4u * b + 3u, float2(1.0f, 0.0f));
        } else {
            chosen[0] = best;
            mkl_c_store(diag, 4u * b, mkl_c_uint(k + 1u));
            for (uint r = m; r <= m + k; ++r) {
                uint row = r < m + k ? r : m + n + 1u;
                float2 x = mkl_c_load(factors, mkl_c_qf(p, b, row, k), 2u);
                mkl_c_store(factors, mkl_c_qf(p, b, row, k), mkl_c_load(factors, mkl_c_qf(p, b, row, best), 2u));
                mkl_c_store(factors, mkl_c_qf(p, b, row, best), x);
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint best = chosen[0];
    if (best == UINT_MAX || best == k) return;
    for (uint i = tid; i < m; i += p.threads) {
        float2 v = mkl_c_load(work, mkl_c_qw(p, b, best, i), 2u);
        mkl_c_store(work, mkl_c_qw(p, b, best, i), mkl_c_load(work, mkl_c_qw(p, b, k, i), 2u));
        mkl_c_store(work, mkl_c_qw(p, b, k, i), v);
    }
}
// Grid (step, batch): c_i = q_i . v_step for every earlier q (one classical
// Gram-Schmidt pass of the pivot column), into aux.
kernel void mkl_c_qr_reorth_dot(device const uint* work [[buffer(0)]], device uint* aux [[buffer(1)]],
                                device const uint* diag [[buffer(2)]], device const uint* qcol [[buffer(3)]],
                                constant MklQrParams& p [[buffer(4)]], uint2 group [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float2 s[33];
    uint m = p.rows, n = p.cols, k = p.step, i = group.x, b = group.y;
    if (as_type<float>(diag[8u * b + 6u]) != 0.0f) return;
    float2 acc = float2(0.0f);
    for (uint r = tid; r < m; r += p.threads)
        acc = mkl_df_add(acc, mkl_df_mul(mkl_c_load(qcol, (b * n + i) * m + r, 2u), mkl_c_load(work, mkl_c_qw(p, b, k, r), 2u)));
    float2 c = mkl_c_qr_sum(s, acc, simd, lane, p.threads / 32u);
    if (tid == 0u) mkl_c_store(aux, b * p.aux_stride + 4u * n + 2u + i, c);
}
// Grid (row blocks, batch): v_step -= sum_i q_i c_i in order of i; the first
// thread adds c_i to R[i][step].
kernel void mkl_c_qr_reorth_update(device uint* work [[buffer(0)]], device uint* factors [[buffer(1)]],
                                   device const uint* aux [[buffer(2)]], device const uint* diag [[buffer(3)]],
                                   device const uint* qcol [[buffer(4)]], constant MklQrParams& p [[buffer(5)]],
                                   uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    uint m = p.rows, n = p.cols, k = p.step, b = group.y, r = group.x * 256u + tid, c0 = b * p.aux_stride + 4u * n + 2u;
    if (as_type<float>(diag[8u * b + 6u]) != 0.0f) return;
    if (r < m) {
        float2 v = mkl_c_load(work, mkl_c_qw(p, b, k, r), 2u);
        for (uint i = 0u; i < k; ++i)
            v = mkl_df_sub(v, mkl_df_mul(mkl_c_load(qcol, (b * n + i) * m + r, 2u), mkl_c_load(aux, c0 + i, 2u)));
        mkl_c_store(work, mkl_c_qw(p, b, k, r), v);
    }
    if (group.x == 0u && tid == 0u) {
        for (uint i = 0u; i < k; ++i) {
            uint f = mkl_c_qf(p, b, p.rows + i, k);
            mkl_c_store(factors, f, mkl_df_add(mkl_c_load(factors, f, 2u), mkl_c_load(aux, c0 + i, 2u)));
        }
    }
}
// Grid (batch): R[step][step] = ||v_step||, q_step = v_step / R[step][step].
kernel void mkl_c_qr_normalise(device const uint* work [[buffer(0)]], device uint* factors [[buffer(1)]],
                               device const uint* diag [[buffer(2)]], device uint* qcol [[buffer(3)]],
                               constant MklQrParams& p [[buffer(4)]], uint b [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]], uint simd [[simdgroup_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float2 s[33];
    uint m = p.rows, n = p.cols, k = p.step;
    if (as_type<float>(diag[8u * b + 6u]) != 0.0f) return;
    float2 d = mkl_c_qr_norm(work, p, b, k, mkl_c_qr_colmax(work, p, b, k, s, tid, simd, lane), s, tid, simd, lane);
    if (tid == 0u) mkl_c_store(factors, mkl_c_qf(p, b, m + k, k), d);
    for (uint i = tid; i < m; i += p.threads) {
        float2 q = mkl_df_div(mkl_c_load(work, mkl_c_qw(p, b, k, i), 2u), d);
        mkl_c_store(qcol, (b * n + k) * m + i, q);
        mkl_c_store(factors, mkl_c_qf(p, b, i, k), q);
    }
}
// Grid (cols - step, batch): columns step+1 .. cols (the last is the right-hand
// side). Two projections against q, then the new residual norm.
kernel void mkl_c_qr_project(device uint* work [[buffer(0)]], device uint* factors [[buffer(1)]],
                             device uint* aux [[buffer(2)]], device const uint* diag [[buffer(3)]],
                             device const uint* qcol [[buffer(4)]], constant MklQrParams& p [[buffer(5)]],
                             uint2 group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                             uint simd [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float2 s[33];
    uint m = p.rows, n = p.cols, k = p.step, b = group.y, j = k + 1u + group.x, aux0 = b * p.aux_stride;
    if (as_type<float>(diag[8u * b + 6u]) != 0.0f) return;
    uint simds = p.threads / 32u;
    float2 acc = float2(0.0f);
    for (uint i = tid; i < m; i += p.threads)
        acc = mkl_df_add(acc, mkl_df_mul(mkl_c_load(qcol, (b * n + k) * m + i, 2u), mkl_c_load(work, mkl_c_qw(p, b, j, i), 2u)));
    float2 dot1 = mkl_c_qr_sum(s, acc, simd, lane, simds);
    acc = float2(0.0f);
    for (uint i = tid; i < m; i += p.threads) {
        float2 q = mkl_c_load(qcol, (b * n + k) * m + i, 2u);
        float2 v = mkl_df_sub(mkl_c_load(work, mkl_c_qw(p, b, j, i), 2u), mkl_df_mul(q, dot1));
        acc = mkl_df_add(acc, mkl_df_mul(q, v));
    }
    float2 dot2 = mkl_c_qr_sum(s, acc, simd, lane, simds);
    float mx = 0.0f;
    for (uint i = tid; i < m; i += p.threads) {
        uint w = mkl_c_qw(p, b, j, i);
        float2 q = mkl_c_load(qcol, (b * n + k) * m + i, 2u);
        float2 v = mkl_df_sub(mkl_c_load(work, w, 2u), mkl_df_mul(q, dot1));
        v = mkl_df_sub(v, mkl_df_mul(q, dot2));
        mkl_c_store(work, w, v);
        mx = mkl_c_maxnan(mx, fabs(v.x));
    }
    mx = mkl_c_qr_max(s, mx, simd, lane, simds);
    float2 norm = mkl_c_qr_norm(work, p, b, j, mx, s, tid, simd, lane);
    if (tid == 0u) {
        float2 r = mkl_df_add(dot1, dot2);
        if (j < n) mkl_c_store(factors, mkl_c_qf(p, b, m + k, j), r);
        else mkl_c_store(aux, aux0 + n + 1u + k, r);
        mkl_c_store(aux, aux0 + j, norm);
    }
}
static inline float2 mkl_c_absdf(float2 x) {
    return x.x < 0.0f ? -x : x;
}
// Grid (batch). Back substitution, then ||R||_inf ||R^-1||_inf: one row of
// R and of R^-1 (forward substitution on R^T into `inverse`) per thread.
kernel void mkl_c_qr_finish(device const uint* factors [[buffer(0)]], device uint* aux [[buffer(1)]],
                            device uint* diag [[buffer(2)]], device uint* solution [[buffer(3)]],
                            device uint* inverse [[buffer(4)]], constant MklQrParams& p [[buffer(5)]],
                            uint b [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup bool ok[1];
    uint m = p.rows, n = p.cols, aux0 = b * p.aux_stride;
    if (tid == 0u) {
        float2 residual = mkl_c_load(aux, aux0 + n, 2u);
        mkl_c_store(diag, 4u * b + 2u, residual);
        bool finite = isfinite(residual.x) && isfinite(residual.y);
        bool good = finite && as_type<float>(diag[8u * b + 6u]) == 0.0f;
        for (uint ri = n; good && ri > 0u; --ri) {
            uint i = ri - 1u;
            float2 v = mkl_c_load(aux, aux0 + n + 1u + i, 2u);
            for (uint j = i + 1u; j < n; ++j)
                v = mkl_df_sub(v, mkl_df_mul(mkl_c_load(factors, mkl_c_qf(p, b, m + i, j), 2u), mkl_c_load(solution, b * n + j, 2u)));
            mkl_c_store(solution, b * n + i, mkl_df_div(v, mkl_c_load(factors, mkl_c_qf(p, b, m + i, i), 2u)));
        }
        for (uint i = 0u; good && i < n; ++i) {
            uint j = uint(mkl_c_load(factors, mkl_c_qf(p, b, m + n + 1u, i), 2u).x);
            float2 x = mkl_df_div(mkl_c_load(solution, b * n + i, 2u), mkl_c_load(factors, mkl_c_qf(p, b, m + n, j), 2u));
            mkl_c_store(aux, aux0 + 2u * n + 2u + j, x);
            good = isfinite(x.x) && isfinite(x.y);
        }
        if (!finite || (!good && as_type<float>(diag[8u * b + 6u]) == 0.0f))
            mkl_c_store(diag, 4u * b + 3u, float2(2.0f, 0.0f));
        ok[0] = good;
    }
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    if (!ok[0]) {
        for (uint j = tid; j < n; j += p.threads) mkl_c_store(solution, b * n + j, float2(NAN, 0.0f));
        return;
    }
    for (uint j = tid; j < n; j += p.threads) mkl_c_store(solution, b * n + j, mkl_c_load(aux, aux0 + 2u * n + 2u + j, 2u));
    for (uint i = tid; i < n; i += p.threads) {
        float2 row = float2(0.0f);
        for (uint j = i; j < n; ++j) row = mkl_df_add(row, mkl_c_absdf(mkl_c_load(factors, mkl_c_qf(p, b, m + i, j), 2u)));
        uint base = (b * n + i) * n;
        float2 z = mkl_df_div(float2(1.0f, 0.0f), mkl_c_load(factors, mkl_c_qf(p, b, m + i, i), 2u));
        float2 sum = mkl_c_absdf(z);
        mkl_c_store(inverse, base + i, z);
        for (uint j = i + 1u; j < n; ++j) {
            float2 t = float2(0.0f);
            for (uint l = i; l < j; ++l)
                t = mkl_df_add(t, mkl_df_mul(mkl_c_load(inverse, base + l, 2u), mkl_c_load(factors, mkl_c_qf(p, b, m + l, j), 2u)));
            z = mkl_df_div(-t, mkl_c_load(factors, mkl_c_qf(p, b, m + j, j), 2u));
            mkl_c_store(inverse, base + j, z);
            sum = mkl_df_add(sum, mkl_c_absdf(z));
        }
        mkl_c_store(aux, aux0 + 2u * n + 2u + i, row);
        mkl_c_store(aux, aux0 + 3u * n + 2u + i, sum);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) {
        float2 normr = float2(0.0f), norminv = float2(0.0f);
        for (uint i = 0u; i < n; ++i) {
            normr = mkl_c_ext(normr, mkl_c_load(aux, aux0 + 2u * n + 2u + i, 2u), true);
            norminv = mkl_c_ext(norminv, mkl_c_load(aux, aux0 + 3u * n + 2u + i, 2u), true);
        }
        mkl_c_store(diag, 4u * b + 1u, mkl_df_mul(normr, norminv));
    }
}
