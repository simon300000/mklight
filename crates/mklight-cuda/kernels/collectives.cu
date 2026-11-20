// Segmented reductions (collectives::Request::SegmentReduce): collectives.wgsl's
// interface and order, term for term (mklight_gpu::collectives::cpu's
// `segments` over the fixed two-pass tree of mklight_gpu::reduce::tree_in).
//
// One block of MKL_REDUCE_THREADS threads per segment runs every term of that
// segment in output order: for each pass-1 group g of the tree thread t folds
// records g·T + t, g·T + t + G·T, … of the segment from the term's identity,
// the group's binary tree (strides T/2 … 1, shared memory) leaves the group
// partial, and pass 2 folds partial t into the identity and runs the same
// tree. No atomics, no shuffles.
//
// Arguments: parameters (binding 1), values (channel-major words, f32 or
// double-float), offsets, mask, weights, reference (absent inputs bind a
// placeholder), output (double-float), status.
#include "mkl_cuda.h"

struct MklWgslSegmentParams {
    uint count, channels, segments;
    // First segment of this dispatch (block x adds to it).
    uint first;
    uint kind, mode, max_len, masked, mask_value;
    // Words per element of values, weights, reference (0: absent).
    uint value_words, weight_words, reference_words;
};

#define SEG_T MKL_REDUCE_THREADS
#define SEG_MIN MKL_COLLECT_MIN
#define SEG_MAX MKL_COLLECT_MAX

// The segment's inputs (one per block).
struct SegIn {
    const MklWgslSegmentParams* p;
    const uint* values;
    const uint* mask;
    const uint* weights;
    const uint* reference;
};

__device__ inline float2 seg_word_df(const uint* w, uint words, uint i) {
    if (words == 2u) return float2(__uint_as_float(w[2u * i]), __uint_as_float(w[2u * i + 1u]));
    return float2(__uint_as_float(w[i]), 0.0f);
}
__device__ inline float2 seg_load(const SegIn& s, uint ch, uint i) {
    return seg_word_df(s.values, s.p->value_words, ch * s.p->count + i);
}
__device__ inline float2 seg_weight(const SegIn& s, uint i) {
    const uint ww = s.p->weight_words;
    if (ww == 2u || ww == 1u) return seg_word_df(s.weights, ww, i);
    if (s.p->channels == 3u) return seg_load(s, 2u, i);
    return float2(1.0f, 0.0f);
}
__device__ inline float2 seg_ref(const SegIn& s, uint j) {
    return seg_word_df(s.reference, s.p->reference_words == 2u ? 2u : 1u, j);
}
__device__ inline float2 seg_coordinate(const SegIn& s, uint ch, uint i, uint z) {
    if (s.p->reference_words != 0u) return mkl_df_add(seg_load(s, ch, i), mkl_df_neg(seg_ref(s, 2u * z + ch)));
    return seg_load(s, ch, i);
}

// df_cmp: numeric order, a NaN word sorts last.
__device__ inline int seg_cmp(float2 a, float2 b) {
    const bool an = isnan(a.x) || isnan(a.y);
    const bool bn = isnan(b.x) || isnan(b.y);
    if (an || bn) return (int)an - (int)bn;
    if (a.x < b.x) return -1;
    if (a.x > b.x) return 1;
    if (a.y < b.y) return -1;
    if (a.y > b.y) return 1;
    return 0;
}
// Ties keep the earlier operand, the first NaN wins.
__device__ inline float2 seg_extremum(float2 a, float2 b, bool mx) {
    if (isnan(a.x)) return a;
    if (isnan(b.x)) return b;
    const int c = seg_cmp(a, b);
    if ((mx && c < 0) || (!mx && c > 0)) return b;
    return a;
}
__device__ inline float2 seg_combine(uint op, float2 a, float2 b) {
    if (op == MKL_REDUCE_PLAIN) return float2(a.x + b.x, 0.0f);
    if (op == SEG_MIN) return seg_extremum(a, b, false);
    if (op == SEG_MAX) return seg_extremum(a, b, true);
    return mkl_df_add(a, b);
}

// Term count of a phase, accumulator and identity of a term.
__device__ inline uint seg_terms_of(const MklWgslSegmentParams& p, uint phase) {
    if (p.kind == MKL_COLLECT_MOMENTS2) return phase == 0u ? 4u : 6u;
    if (p.kind == MKL_COLLECT_SUMMARY) return 4u * p.channels;
    return p.channels;
}
__device__ inline uint seg_op(const MklWgslSegmentParams& p, uint phase, uint term) {
    if (p.kind == MKL_COLLECT_SUMMARY) {
        const uint t = term % 4u;
        if (t == 1u) return MKL_REDUCE_DF;
        if (t == 2u) return SEG_MIN;
        if (t == 3u) return SEG_MAX;
    }
    if (p.kind == MKL_COLLECT_MOMENTS2) {
        if (phase == 0u && term == 0u) return MKL_REDUCE_DF;
        if (phase == 1u && term >= 4u) return SEG_MAX;
    }
    return p.mode;
}
__device__ inline float2 seg_identity(const MklWgslSegmentParams& p, uint op) {
    if (op == SEG_MIN) return float2(INFINITY, 0.0f);
    if (op == SEG_MAX && p.kind == MKL_COLLECT_SUMMARY) return float2(-INFINITY, 0.0f);
    return float2(0.0f);
}
// Output index of a term (collectives::cpu's `slot`).
__device__ inline uint seg_slot(const MklWgslSegmentParams& p, uint z, uint phase, uint term) {
    const uint c = p.channels;
    if (p.kind == MKL_COLLECT_SUM) return term * p.segments + z;
    if (p.kind == MKL_COLLECT_SUMMARY) return ((term / 4u) * p.segments + z) * 4u + term % 4u;
    if (p.kind == MKL_COLLECT_MOMENTS2) return z * 10u + phase * 4u + term;
    return z * (c + c * (c + 1u) / 2u) + term;
}

__device__ inline float2 seg_r2(float2 a, float2 b) {
    return mkl_df_add(mkl_df_mul(a, a), mkl_df_mul(b, b));
}
// The term value of record i of segment z (mkl_c_seg_value).
__device__ float2 seg_value(const SegIn& s, uint z, uint phase, uint term, uint i, float2 id,
                            float2 mean, float2 mean_y) {
    const MklWgslSegmentParams& p = *s.p;
    if (p.masked != 0u && s.mask[i] != p.mask_value) return id;
    if (p.kind == MKL_COLLECT_SUMMARY) {
        if (term % 4u == 1u) return float2(1.0f, 0.0f);
        return seg_load(s, term / 4u, i);
    }
    if (p.kind != MKL_COLLECT_MOMENTS2) return seg_load(s, term, i);
    const float2 w = seg_weight(s, i);
    if (phase == 0u && term < 2u) return term == 0u ? float2(1.0f, 0.0f) : w;
    float2 px = seg_coordinate(s, 0u, i, z);
    float2 py = seg_coordinate(s, 1u, i, z);
    if (phase == 0u) return mkl_df_mul(w, term == 2u ? px : py);
    const bool positive = w.x > 0.0f;
    if (term == 0u) return mkl_df_mul(w, seg_r2(px, py));
    if (term == 4u) return positive ? seg_r2(px, py) : float2(0.0f);
    px = mkl_df_add(px, mkl_df_neg(mean));
    py = mkl_df_add(py, mkl_df_neg(mean_y));
    if (term == 5u) return positive ? seg_r2(px, py) : float2(0.0f);
    if (term == 1u) return mkl_df_mul(w, mkl_df_mul(px, px));
    if (term == 2u) return mkl_df_mul(w, mkl_df_mul(px, py));
    return mkl_df_mul(w, mkl_df_mul(py, py));
}

// Neumaier's step on the state (s, c) (common.wgsl's mkl_neumaier_add).
__device__ inline float2 seg_neumaier(float2 sc, float x) {
    float s = sc.x, c = sc.y;
    mkl_neumaier_add(s, c, x);
    return float2(s, c);
}

// The in-block binary tree over v (every thread calls it).
__device__ inline float2 seg_tree(float2* v, uint tid, uint op) {
    for (uint stride = SEG_T / 2u; stride >= 1u; stride >>= 1) {
        __syncthreads();
        if (tid < stride) v[tid] = seg_combine(op, v[tid], v[tid + stride]);
    }
    __syncthreads();
    const float2 r = v[0];
    __syncthreads();
    return r;
}

MKL_KERNEL __launch_bounds__(MKL_REDUCE_THREADS)
mkl_segment_reduce(const MklWgslSegmentParams* u, const uint* s_values, const uint* s_offsets,
                   const uint* s_mask, const uint* s_weights, const uint* s_reference,
                   float2* s_out, uint* s_status) {
    __shared__ float2 seg_v[MKL_REDUCE_THREADS];
    __shared__ float2 seg_partials[MKL_REDUCE_MAX_GROUPS];
    // Results of the segment's moment terms (summary and source products
    // publish as they go).
    __shared__ float2 seg_terms[10];
    __shared__ uint seg_len, seg_start, seg_valid;
    const MklWgslSegmentParams& p = *u;
    const SegIn s = {u, s_values, s_mask, s_weights, s_reference};
    const uint tid = threadIdx.x;
    const uint z = p.first + blockIdx.x;
    const uint c = p.channels;
    const uint phases = p.kind == MKL_COLLECT_MOMENTS2 ? 2u : 1u;
    const uint products = p.kind == MKL_COLLECT_SOURCE_PRODUCTS ? c * (c + 1u) / 2u : 0u;
    if (tid == 0u) {
        const uint a = s_offsets[z];
        const uint b = s_offsets[z + 1u];
        const bool ok = a <= b && b <= p.count && b - a <= p.max_len;
        seg_valid = (uint)ok;
        seg_start = a;
        seg_len = ok ? b - a : 0u;
        s_status[z] = (uint)!ok;
    }
    __syncthreads();
    if (seg_valid == 0u) {
        if (tid == 0u) {
            for (uint phase = 0u; phase < phases; phase++)
                for (uint t = 0u; t < seg_terms_of(p, phase) + products; t++)
                    s_out[seg_slot(p, z, phase, t)] = float2(mkl_qnan(), 0.0f);
        }
        return;
    }
    const uint len = seg_len;
    const uint a = seg_start;
    const uint groups = clamp((len + SEG_T * MKL_REDUCE_ITEMS_PER_THREAD - 1u) / (SEG_T * MKL_REDUCE_ITEMS_PER_THREAD),
                              1u, MKL_REDUCE_MAX_GROUPS);
    const uint step = groups * SEG_T;
    float2 mean = float2(0.0f), mean_y = float2(0.0f);
    for (uint phase = 0u; phase < phases; phase++) {
        const uint terms = seg_terms_of(p, phase);
        for (uint term = 0u; term < terms; term++) {
            const uint op = seg_op(p, phase, term);
            const float2 id = seg_identity(p, op);
            for (uint g = 0u; g < groups; g++) {
                float2 acc = id;
                if (op == MKL_REDUCE_PLAIN) {
                    float sum = id.x;
                    for (uint i = g * SEG_T + tid; i < len; i += step) {
                        const float2 q = seg_value(s, z, phase, term, a + i, id, mean, mean_y);
                        sum = sum + (q.x + q.y);
                    }
                    acc = float2(sum, 0.0f);
                } else if (op == MKL_REDUCE_KAHAN) {
                    float2 sc = float2(0.0f);
                    for (uint i = g * SEG_T + tid; i < len; i += step) {
                        const float2 q = seg_value(s, z, phase, term, a + i, id, mean, mean_y);
                        sc = seg_neumaier(sc, q.x);
                        if (q.y != 0.0f) sc = seg_neumaier(sc, q.y);
                    }
                    acc = mkl_two_sum(sc.x, sc.y);
                } else {
                    for (uint i = g * SEG_T + tid; i < len; i += step)
                        acc = seg_combine(op, acc, seg_value(s, z, phase, term, a + i, id, mean, mean_y));
                }
                seg_v[tid] = acc;
                const float2 r = seg_tree(seg_v, tid, op);
                if (tid == 0u) seg_partials[g] = r;
            }
            __syncthreads();
            float2 acc = id;
            if (tid < groups) acc = seg_combine(op, id, seg_partials[tid]);
            seg_v[tid] = acc;
            const float2 r = seg_tree(seg_v, tid, op);
            if (tid == 0u) {
                if (p.kind == MKL_COLLECT_MOMENTS2) seg_terms[phase * 4u + term] = r;
                else s_out[seg_slot(p, z, phase, term)] = r;
            }
        }
        if (p.kind == MKL_COLLECT_MOMENTS2 && phase == 0u) {
            if (tid == 0u) {
                const float2 w = seg_terms[1];
                for (uint k = 0u; k < 2u; k++) {
                    float2 m = float2(mkl_qnan(), 0.0f);
                    if (w.x > 0.0f) m = mkl_df_div(seg_terms[2u + k], w);
                    seg_terms[2u + k] = m;
                }
            }
            __syncthreads();
            mean = seg_terms[2];
            mean_y = seg_terms[3];
        }
    }
    if (tid == 0u) {
        if (p.kind == MKL_COLLECT_MOMENTS2) {
            for (uint k = 0u; k < 10u; k++) s_out[z * 10u + k] = seg_terms[k];
        } else if (p.kind == MKL_COLLECT_SOURCE_PRODUCTS) {
            const uint base = seg_slot(p, z, 0u, 0u);
            uint dest = base + c;
            for (uint i = 0u; i < c; i++)
                for (uint j = i; j < c; j++) s_out[dest++] = mkl_df_mul(s_out[base + i], s_out[base + j]);
        }
    }
}
