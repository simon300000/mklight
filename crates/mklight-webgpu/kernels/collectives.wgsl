// Segmented reductions (collectives::Request::SegmentReduce): the WGSL twin
// of mklight_gpu::collectives::cpu's `segments`, term for term, and of the
// fixed two-pass tree of mklight_gpu::reduce::tree_in for every term.
//
// One workgroup of MKL_REDUCE_THREADS threads per segment runs every term of
// that segment in output order: for each pass-1 group g of the tree (the
// group count depends on the segment length only) thread t folds records
// g·T + t, g·T + t + G·T, … of the segment from the term's identity, the
// group's binary tree (strides T/2 … 1) leaves the group partial in
// workgroup memory, and pass 2 folds partial t into the identity and runs
// the same tree. The order is the specification's, so every term is bitwise
// the CPU-f32 result; no atomics, no subgroups.
//
// Bindings: 0 env, 1 parameters, 2 values (channel-major words, f32 or
// double-float), 3 offsets, 4 mask, 5 weights, 6 reference (absent inputs
// bind a placeholder), 7 output (double-float), 8 status.

struct MklWgslSegmentParams {
    count: u32,
    channels: u32,
    segments: u32,
    // First segment of this dispatch (workgroup x adds to it).
    first: u32,
    kind: u32,
    mode: u32,
    max_len: u32,
    masked: u32,
    mask_value: u32,
    // Words per element of values, weights, reference (0: absent).
    value_words: u32,
    weight_words: u32,
    reference_words: u32,
}

@group(0) @binding(1) var<uniform> sp: MklWgslSegmentParams;
@group(0) @binding(2) var<storage, read> s_values: array<u32>;
@group(0) @binding(3) var<storage, read> s_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> s_mask: array<u32>;
@group(0) @binding(5) var<storage, read> s_weights: array<u32>;
@group(0) @binding(6) var<storage, read> s_reference: array<u32>;
@group(0) @binding(7) var<storage, read_write> s_out: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> s_status: array<u32>;

const SEG_T: u32 = MKL_REDUCE_THREADS;
// Accumulators: MKL_REDUCE_PLAIN / KAHAN / DF sums, then the extrema.
const SEG_MIN: u32 = MKL_COLLECT_MIN;
const SEG_MAX: u32 = MKL_COLLECT_MAX;

var<workgroup> seg_v: array<vec2<f32>, MKL_REDUCE_THREADS>;
var<workgroup> seg_partials: array<vec2<f32>, MKL_REDUCE_MAX_GROUPS>;
// Results of the segment's terms (at most 10 for moments; summary and
// source products publish as they go).
var<workgroup> seg_terms: array<vec2<f32>, 10>;
var<workgroup> seg_len: u32;
var<workgroup> seg_start: u32;
var<workgroup> seg_valid: u32;

fn seg_df(words: u32, i: u32) -> vec2<f32> {
    if (words == 2u) {
        return vec2<f32>(bitcast<f32>(s_values[2u * i]), bitcast<f32>(s_values[2u * i + 1u]));
    }
    return vec2<f32>(bitcast<f32>(s_values[i]), 0.0);
}
fn seg_load(ch: u32, i: u32) -> vec2<f32> {
    return seg_df(sp.value_words, ch * sp.count + i);
}
fn seg_weight(i: u32) -> vec2<f32> {
    if (sp.weight_words == 2u) {
        return vec2<f32>(bitcast<f32>(s_weights[2u * i]), bitcast<f32>(s_weights[2u * i + 1u]));
    }
    if (sp.weight_words == 1u) {
        return vec2<f32>(bitcast<f32>(s_weights[i]), 0.0);
    }
    if (sp.channels == 3u) {
        return seg_load(2u, i);
    }
    return vec2<f32>(1.0, 0.0);
}
fn seg_ref(j: u32) -> vec2<f32> {
    if (sp.reference_words == 2u) {
        return vec2<f32>(bitcast<f32>(s_reference[2u * j]), bitcast<f32>(s_reference[2u * j + 1u]));
    }
    return vec2<f32>(bitcast<f32>(s_reference[j]), 0.0);
}
fn seg_coordinate(ch: u32, i: u32, z: u32) -> vec2<f32> {
    if (sp.reference_words != 0u) {
        return mkl_df_add(seg_load(ch, i), mkl_df_neg(seg_ref(2u * z + ch)));
    }
    return seg_load(ch, i);
}

// df_cmp: numeric order, a NaN word sorts last.
fn seg_cmp(a: vec2<f32>, b: vec2<f32>) -> i32 {
    let an = mkl_is_nan(a.x) || mkl_is_nan(a.y);
    let bn = mkl_is_nan(b.x) || mkl_is_nan(b.y);
    if (an || bn) {
        return i32(an) - i32(bn);
    }
    if (a.x < b.x) {
        return -1;
    }
    if (a.x > b.x) {
        return 1;
    }
    if (a.y < b.y) {
        return -1;
    }
    if (a.y > b.y) {
        return 1;
    }
    return 0;
}
// Ties keep the earlier operand, the first NaN wins.
fn seg_extremum(a: vec2<f32>, b: vec2<f32>, mx: bool) -> vec2<f32> {
    if (mkl_is_nan(a.x)) {
        return a;
    }
    if (mkl_is_nan(b.x)) {
        return b;
    }
    let c = seg_cmp(a, b);
    if ((mx && c < 0) || (!mx && c > 0)) {
        return b;
    }
    return a;
}
fn seg_combine(op: u32, a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    if (op == MKL_REDUCE_PLAIN) {
        return vec2<f32>(mkl_add(a.x, b.x), 0.0);
    }
    if (op == SEG_MIN) {
        return seg_extremum(a, b, false);
    }
    if (op == SEG_MAX) {
        return seg_extremum(a, b, true);
    }
    return mkl_df_add(a, b);
}

// Term count of a phase, accumulator and identity of a term.
fn seg_terms_of(phase: u32) -> u32 {
    if (sp.kind == MKL_COLLECT_MOMENTS2) {
        return select(6u, 4u, phase == 0u);
    }
    if (sp.kind == MKL_COLLECT_SUMMARY) {
        return 4u * sp.channels;
    }
    return sp.channels;
}
fn seg_op(phase: u32, term: u32) -> u32 {
    if (sp.kind == MKL_COLLECT_SUMMARY) {
        let t = term % 4u;
        if (t == 1u) {
            return MKL_REDUCE_DF;
        }
        if (t == 2u) {
            return SEG_MIN;
        }
        if (t == 3u) {
            return SEG_MAX;
        }
    }
    if (sp.kind == MKL_COLLECT_MOMENTS2) {
        if (phase == 0u && term == 0u) {
            return MKL_REDUCE_DF;
        }
        if (phase == 1u && term >= 4u) {
            return SEG_MAX;
        }
    }
    return sp.mode;
}
fn seg_identity(op: u32) -> vec2<f32> {
    if (op == SEG_MIN) {
        return vec2<f32>(mkl_inf(), 0.0);
    }
    if (op == SEG_MAX && sp.kind == MKL_COLLECT_SUMMARY) {
        return vec2<f32>(-mkl_inf(), 0.0);
    }
    return vec2<f32>(0.0, 0.0);
}
// Output index of a term (collectives::cpu's `slot`).
fn seg_slot(z: u32, phase: u32, term: u32) -> u32 {
    let c = sp.channels;
    if (sp.kind == MKL_COLLECT_SUM) {
        return term * sp.segments + z;
    }
    if (sp.kind == MKL_COLLECT_SUMMARY) {
        return ((term / 4u) * sp.segments + z) * 4u + term % 4u;
    }
    if (sp.kind == MKL_COLLECT_MOMENTS2) {
        return z * 10u + phase * 4u + term;
    }
    return z * (c + c * (c + 1u) / 2u) + term;
}

fn seg_r2(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_add(mkl_df_mul(a, a), mkl_df_mul(b, b));
}
// The term value of record i of segment z (mkl_c_seg_value).
fn seg_value(z: u32, phase: u32, term: u32, i: u32, id: vec2<f32>, mean: vec2<f32>, mean_y: vec2<f32>) -> vec2<f32> {
    if (sp.masked != 0u && s_mask[i] != sp.mask_value) {
        return id;
    }
    if (sp.kind == MKL_COLLECT_SUMMARY) {
        if (term % 4u == 1u) {
            return vec2<f32>(1.0, 0.0);
        }
        return seg_load(term / 4u, i);
    }
    if (sp.kind != MKL_COLLECT_MOMENTS2) {
        return seg_load(term, i);
    }
    let w = seg_weight(i);
    if (phase == 0u && term < 2u) {
        return select(w, vec2<f32>(1.0, 0.0), term == 0u);
    }
    var px = seg_coordinate(0u, i, z);
    var py = seg_coordinate(1u, i, z);
    if (phase == 0u) {
        return mkl_df_mul(w, select(py, px, term == 2u));
    }
    let positive = w.x > 0.0;
    if (term == 0u) {
        return mkl_df_mul(w, seg_r2(px, py));
    }
    if (term == 4u) {
        return select(vec2<f32>(0.0, 0.0), seg_r2(px, py), positive);
    }
    px = mkl_df_add(px, mkl_df_neg(mean));
    py = mkl_df_add(py, mkl_df_neg(mean_y));
    if (term == 5u) {
        return select(vec2<f32>(0.0, 0.0), seg_r2(px, py), positive);
    }
    if (term == 1u) {
        return mkl_df_mul(w, mkl_df_mul(px, px));
    }
    if (term == 2u) {
        return mkl_df_mul(w, mkl_df_mul(px, py));
    }
    return mkl_df_mul(w, mkl_df_mul(py, py));
}

// The in-group binary tree over seg_v (every thread calls it).
fn seg_tree(tid: u32, op: u32) -> vec2<f32> {
    var stride = SEG_T / 2u;
    loop {
        workgroupBarrier();
        if (tid < stride) {
            seg_v[tid] = seg_combine(op, seg_v[tid], seg_v[tid + stride]);
        }
        if (stride == 1u) {
            break;
        }
        stride = stride >> 1u;
    }
    workgroupBarrier();
    let r = seg_v[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(MKL_REDUCE_THREADS)
fn mkl_segment_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let tid = lid.x;
    let z = sp.first + group.x;
    let c = sp.channels;
    let phases = select(1u, 2u, sp.kind == MKL_COLLECT_MOMENTS2);
    let products = select(0u, c * (c + 1u) / 2u, sp.kind == MKL_COLLECT_SOURCE_PRODUCTS);
    if (tid == 0u) {
        let a = s_offsets[z];
        let b = s_offsets[z + 1u];
        let ok = a <= b && b <= sp.count && b - a <= sp.max_len;
        seg_valid = u32(ok);
        seg_start = a;
        seg_len = select(0u, b - a, ok);
        s_status[z] = u32(!ok);
    }
    let valid = workgroupUniformLoad(&seg_valid);
    if (valid == 0u) {
        if (tid == 0u) {
            for (var phase = 0u; phase < phases; phase++) {
                for (var t = 0u; t < seg_terms_of(phase) + products; t++) {
                    s_out[seg_slot(z, phase, t)] = vec2<f32>(mkl_qnan(), 0.0);
                }
            }
        }
        return;
    }
    let len = workgroupUniformLoad(&seg_len);
    let a = seg_start;
    let groups = clamp((len + SEG_T * MKL_REDUCE_ITEMS_PER_THREAD - 1u) / (SEG_T * MKL_REDUCE_ITEMS_PER_THREAD), 1u, MKL_REDUCE_MAX_GROUPS);
    let step = groups * SEG_T;
    var mean = vec2<f32>(0.0, 0.0);
    var mean_y = vec2<f32>(0.0, 0.0);
    for (var phase = 0u; phase < phases; phase++) {
        let terms = seg_terms_of(phase);
        for (var term = 0u; term < terms; term++) {
            let op = seg_op(phase, term);
            let id = seg_identity(op);
            for (var g = 0u; g < groups; g++) {
                var acc = id;
                if (op == MKL_REDUCE_PLAIN) {
                    var s = id.x;
                    for (var i = g * SEG_T + tid; i < len; i += step) {
                        let q = seg_value(z, phase, term, a + i, id, mean, mean_y);
                        s = mkl_add(s, mkl_add(q.x, q.y));
                    }
                    acc = vec2<f32>(s, 0.0);
                } else if (op == MKL_REDUCE_KAHAN) {
                    var sc = vec2<f32>(0.0, 0.0);
                    for (var i = g * SEG_T + tid; i < len; i += step) {
                        let q = seg_value(z, phase, term, a + i, id, mean, mean_y);
                        sc = mkl_neumaier_add(sc, q.x);
                        if (q.y != 0.0) {
                            sc = mkl_neumaier_add(sc, q.y);
                        }
                    }
                    acc = mkl_two_sum(sc.x, sc.y);
                } else {
                    for (var i = g * SEG_T + tid; i < len; i += step) {
                        acc = seg_combine(op, acc, seg_value(z, phase, term, a + i, id, mean, mean_y));
                    }
                }
                seg_v[tid] = acc;
                let r = seg_tree(tid, op);
                if (tid == 0u) {
                    seg_partials[g] = r;
                }
            }
            workgroupBarrier();
            var acc = id;
            if (tid < groups) {
                acc = seg_combine(op, id, seg_partials[tid]);
            }
            seg_v[tid] = acc;
            let r = seg_tree(tid, op);
            if (tid == 0u) {
                if (sp.kind == MKL_COLLECT_MOMENTS2) {
                    seg_terms[phase * 4u + term] = r;
                } else {
                    s_out[seg_slot(z, phase, term)] = r;
                }
            }
        }
        if (sp.kind == MKL_COLLECT_MOMENTS2 && phase == 0u) {
            if (tid == 0u) {
                let w = seg_terms[1];
                for (var k = 0u; k < 2u; k++) {
                    var m = vec2<f32>(mkl_qnan(), 0.0);
                    if (w.x > 0.0) {
                        m = mkl_df_div_rn(seg_terms[2u + k], w);
                    }
                    seg_terms[2u + k] = m;
                }
            }
            workgroupBarrier();
            mean = seg_terms[2];
            mean_y = seg_terms[3];
        }
    }
    if (tid == 0u) {
        if (sp.kind == MKL_COLLECT_MOMENTS2) {
            for (var k = 0u; k < 10u; k++) {
                s_out[z * 10u + k] = seg_terms[k];
            }
        } else if (sp.kind == MKL_COLLECT_SOURCE_PRODUCTS) {
            let base = seg_slot(z, 0u, 0u);
            var dest = base + c;
            for (var i = 0u; i < c; i++) {
                for (var j = i; j < c; j++) {
                    s_out[dest] = mkl_df_mul(s_out[base + i], s_out[base + j]);
                    dest++;
                }
            }
        }
    }
}
