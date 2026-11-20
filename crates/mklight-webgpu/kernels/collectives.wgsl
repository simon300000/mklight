// Segmented reductions (collectives::Request::SegmentReduce): the WGSL twin
// of mklight_gpu::collectives::cpu's `segments`, term for term, and of the
// fixed two-pass tree of mklight_gpu::reduce::tree_in for every term.
//
// Two entries per phase (moments have two phases, the second centred on the
// first's means), so that a few long segments fill the device:
// * mkl_segment_pass1, workgroup (g, term, segment): if g is below the
//   segment's group count G (a function of its length only), thread t folds
//   records g·T + t, g·T + t + G·T, … of the segment from the term's
//   identity, and the group's binary tree (strides T/2 … 1) leaves the group
//   partial in partials[(segment · terms + term) · max_groups + g] (segment
//   relative to `first`).
// * mkl_segment_pass2, one workgroup per segment: for every term of the
//   phase thread t folds partial t into the identity and runs the same tree;
//   it writes the output, the status and the moments' means or the source
//   products.
// Each partial depends only on (segment, term, g), so the order is the
// specification's and every term is bitwise the CPU-f32 result; no atomics,
// no subgroups.
//
// Bindings: 0 env, 1 parameters, 2 values (channel-major words, f32 or
// double-float), 3 offsets, 4 mask, 5 weights, 6 reference (absent inputs
// bind a placeholder), 7 output (double-float), 8 status, 9 partials
// (double-float scratch).

struct MklWgslSegmentParams {
    count: u32,
    channels: u32,
    segments: u32,
    // First segment of this dispatch (the segment workgroup index adds to it).
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
    // Phase of this pass pair, pass-1 group cap of the segments
    // (reduce::num_groups(max_len)) and terms of the phase (partial strides).
    phase: u32,
    max_groups: u32,
    terms: u32,
}

@group(0) @binding(1) var<uniform> sp: MklWgslSegmentParams;
@group(0) @binding(2) var<storage, read> s_values: array<u32>;
@group(0) @binding(3) var<storage, read> s_offsets: array<u32>;
@group(0) @binding(4) var<storage, read> s_mask: array<u32>;
@group(0) @binding(5) var<storage, read> s_weights: array<u32>;
@group(0) @binding(6) var<storage, read> s_reference: array<u32>;
@group(0) @binding(7) var<storage, read_write> s_out: array<vec2<f32>>;
@group(0) @binding(8) var<storage, read_write> s_status: array<u32>;
@group(0) @binding(9) var<storage, read_write> s_partials: array<vec2<f32>>;

const SEG_T: u32 = MKL_REDUCE_THREADS;
// Accumulators: MKL_REDUCE_PLAIN / KAHAN / DF sums, then the extrema.
const SEG_MIN: u32 = MKL_COLLECT_MIN;
const SEG_MAX: u32 = MKL_COLLECT_MAX;

var<workgroup> seg_v: array<vec2<f32>, MKL_REDUCE_THREADS>;
var<workgroup> seg_len: u32;
var<workgroup> seg_start: u32;

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

const SEG_INVALID: u32 = 0xffffffffu;

// Length of segment z (SEG_INVALID if its offsets are not valid) as a
// workgroup-uniform value; seg_start holds its first record.
fn seg_range(tid: u32, z: u32) -> u32 {
    if (tid == 0u) {
        let a = s_offsets[z];
        let b = s_offsets[z + 1u];
        let ok = a <= b && b <= sp.count && b - a <= sp.max_len;
        seg_start = a;
        seg_len = select(SEG_INVALID, b - a, ok);
    }
    return workgroupUniformLoad(&seg_len);
}
fn seg_groups(len: u32) -> u32 {
    return clamp((len + SEG_T * MKL_REDUCE_ITEMS_PER_THREAD - 1u) / (SEG_T * MKL_REDUCE_ITEMS_PER_THREAD), 1u, MKL_REDUCE_MAX_GROUPS);
}

@compute @workgroup_size(MKL_REDUCE_THREADS)
fn mkl_segment_pass1(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let tid = lid.x;
    let g = group.x;
    let term = group.y;
    let z = sp.first + group.z;
    let phase = sp.phase;
    let len = seg_range(tid, z);
    if (len == SEG_INVALID) {
        return;
    }
    let groups = seg_groups(len);
    if (g >= groups) {
        return;
    }
    let a = seg_start;
    let step = groups * SEG_T;
    var mean = vec2<f32>(0.0, 0.0);
    var mean_y = vec2<f32>(0.0, 0.0);
    if (phase == 1u) {
        mean = s_out[z * 10u + 2u];
        mean_y = s_out[z * 10u + 3u];
    }
    let op = seg_op(phase, term);
    let id = seg_identity(op);
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
        s_partials[(group.z * sp.terms + term) * sp.max_groups + g] = r;
    }
}

@compute @workgroup_size(MKL_REDUCE_THREADS)
fn mkl_segment_pass2(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let tid = lid.x;
    let z = sp.first + group.x;
    let c = sp.channels;
    let phase = sp.phase;
    let len = seg_range(tid, z);
    if (tid == 0u) {
        s_status[z] = u32(len == SEG_INVALID);
    }
    if (len == SEG_INVALID) {
        if (tid == 0u) {
            let phases = select(1u, 2u, sp.kind == MKL_COLLECT_MOMENTS2);
            let products = select(0u, c * (c + 1u) / 2u, sp.kind == MKL_COLLECT_SOURCE_PRODUCTS);
            for (var ph = 0u; ph < phases; ph++) {
                for (var t = 0u; t < seg_terms_of(ph) + products; t++) {
                    s_out[seg_slot(z, ph, t)] = vec2<f32>(mkl_qnan(), 0.0);
                }
            }
        }
        return;
    }
    let groups = seg_groups(len);
    for (var term = 0u; term < sp.terms; term++) {
        let op = seg_op(phase, term);
        let id = seg_identity(op);
        var acc = id;
        if (tid < groups) {
            acc = seg_combine(op, id, s_partials[(group.x * sp.terms + term) * sp.max_groups + tid]);
        }
        seg_v[tid] = acc;
        let r = seg_tree(tid, op);
        if (tid == 0u) {
            s_out[seg_slot(z, phase, term)] = r;
        }
    }
    if (tid == 0u) {
        if (sp.kind == MKL_COLLECT_MOMENTS2 && phase == 0u) {
            // The means replace the weighted sums (slots 2, 3).
            let w = s_out[z * 10u + 1u];
            for (var k = 0u; k < 2u; k++) {
                var m = vec2<f32>(mkl_qnan(), 0.0);
                if (w.x > 0.0) {
                    m = mkl_df_div_rn(s_out[z * 10u + 2u + k], w);
                }
                s_out[z * 10u + 2u + k] = m;
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
