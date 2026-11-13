// Resident geometric algebra (geometry::Request::{Coordinates, FocusTerms,
// Characteristic}): the WGSL twin of crates/mklight-metal/shaders/
// geometry.metal and of mklight_gpu::geometry::{cpu, spec}, operation for
// operation. Every division is the correctly rounded mkl_df_div_rn, so
// image coordinates and focus terms are bitwise the CPU-f32 emulation. The
// characteristic function evaluates cos/sin in double-float from a
// three-part pi/2 reduction and rounds once (the CPU uses the platform
// libm; both meet the characteristic-function tolerance, and agree bit for
// bit wherever libm rounds correctly).
//
// Bindings: 0 env, 1 parameters, 2..6 inputs (u32 words), 7..8 outputs
// (u32 words); each entry names its use of them. Thread `tid` of a pass
// handles element `offset + tid`.

struct MklWgslGeometryParams {
    // Elements of this pass, first element.
    count: u32,
    offset: u32,
    // Rays (plane stride) and chief rays (or the segment count).
    rays: u32,
    chiefs: u32,
    segment: u32,
    centroid: u32,
    reserved0: u32,
    reserved1: u32,
    defocus: vec2<f32>,
}

@group(0) @binding(1) var<uniform> gp: MklWgslGeometryParams;
@group(0) @binding(2) var<storage, read> g_in0: array<u32>;
@group(0) @binding(3) var<storage, read> g_in1: array<u32>;
@group(0) @binding(4) var<storage, read> g_in2: array<u32>;
@group(0) @binding(5) var<storage, read> g_in3: array<u32>;
@group(0) @binding(6) var<storage, read> g_in4: array<u32>;
@group(0) @binding(7) var<storage, read_write> g_out0: array<u32>;
@group(0) @binding(8) var<storage, read_write> g_out1: array<u32>;

const GEO_GROUP: u32 = 256u;

fn geo_put0(i: u32, v: vec2<f32>) {
    g_out0[2u * i] = bitcast<u32>(v.x);
    g_out0[2u * i + 1u] = bitcast<u32>(v.y);
}

// ---------------------------------------------------------------------------
// Coordinates: in0 hit floats [15, rays], in1 hit status, in2 chief floats
// [15, chiefs], in3 chief status, in4 chief index; out0 DF [5, rays], out1
// mask.
// ---------------------------------------------------------------------------
fn geo_hit_f(chief: bool, n: u32, i: u32, k: u32) -> f32 {
    if (chief) {
        return bitcast<f32>(g_in2[k * n + i]);
    }
    return bitcast<f32>(g_in0[k * n + i]);
}
fn geo_hit(chief: bool, n: u32, i: u32, k: u32) -> vec2<f32> {
    return vec2<f32>(geo_hit_f(chief, n, i, k), geo_hit_f(chief, n, i, k + 9u));
}

@compute @workgroup_size(GEO_GROUP)
fn mkl_geometry_coordinates(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    if (gid.x >= gp.count) {
        return;
    }
    let i = gp.offset + gid.x;
    let n = gp.rays;
    let nc = gp.chiefs;
    let ok = g_in1[i] == 0u;
    let j = g_in4[i];
    let cok = j < nc && g_in3[j] == 0u;
    g_out1[i] = u32(ok);
    for (var k = 0u; k < 5u; k++) {
        geo_put0(k * n + i, vec2<f32>(0.0, 0.0));
    }
    if (!ok) {
        return;
    }
    let dz = geo_hit(false, n, i, 5u);
    var c = array<vec2<f32>, 4>();
    if (cok) {
        let cdz = geo_hit(true, nc, j, 5u);
        c[0] = geo_hit(true, nc, j, 0u);
        c[1] = geo_hit(true, nc, j, 1u);
        c[2] = mkl_df_div_rn(geo_hit(true, nc, j, 3u), cdz);
        c[3] = mkl_df_div_rn(geo_hit(true, nc, j, 4u), cdz);
    }
    for (var k = 0u; k < 2u; k++) {
        let t = mkl_df_div_rn(geo_hit(false, n, i, k + 3u), dz);
        let dt = mkl_df_sub(t, c[k + 2u]);
        let q = mkl_df_add(mkl_df_sub(geo_hit(false, n, i, k), c[k]), mkl_df_mul(gp.defocus, dt));
        geo_put0(k * n + i, q);
        geo_put0((k + 2u) * n + i, dt);
    }
    geo_put0(4u * n + i, vec2<f32>(bitcast<f32>(g_in0[8u * n + i]), 0.0));
}

// ---------------------------------------------------------------------------
// FocusTerms: in0 points DF [5, rays], in1 mask, in2 position moments DF
// [segments, 10], in3 slope moments; out0 DF [3, rays].
// ---------------------------------------------------------------------------
fn geo_df(which: u32, i: u32) -> vec2<f32> {
    if (which == 0u) {
        return vec2<f32>(bitcast<f32>(g_in0[2u * i]), bitcast<f32>(g_in0[2u * i + 1u]));
    }
    if (which == 2u) {
        return vec2<f32>(bitcast<f32>(g_in2[2u * i]), bitcast<f32>(g_in2[2u * i + 1u]));
    }
    return vec2<f32>(bitcast<f32>(g_in3[2u * i]), bitcast<f32>(g_in3[2u * i + 1u]));
}

@compute @workgroup_size(GEO_GROUP)
fn mkl_geometry_focus_terms(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    if (gid.x >= gp.count) {
        return;
    }
    let i = gp.offset + gid.x;
    let n = gp.rays;
    for (var k = 0u; k < 3u; k++) {
        geo_put0(k * n + i, vec2<f32>(0.0, 0.0));
    }
    if (g_in1[i] == 0u) {
        return;
    }
    let row = (i / gp.segment) * 10u;
    var q = array<vec2<f32>, 2>();
    var t = array<vec2<f32>, 2>();
    for (var k = 0u; k < 2u; k++) {
        var qc = vec2<f32>(0.0, 0.0);
        var tc = vec2<f32>(0.0, 0.0);
        if (gp.centroid != 0u) {
            qc = geo_df(2u, row + 2u + k);
            tc = geo_df(3u, row + 2u + k);
        }
        q[k] = mkl_df_sub(geo_df(0u, k * n + i), qc);
        t[k] = mkl_df_sub(geo_df(0u, (k + 2u) * n + i), tc);
    }
    let w = geo_df(0u, 4u * n + i);
    geo_put0(i, mkl_df_mul(mkl_df_add(mkl_df_mul(q[0], q[0]), mkl_df_mul(q[1], q[1])), w));
    geo_put0(n + i, mkl_df_mul(mkl_df_add(mkl_df_mul(q[0], t[0]), mkl_df_mul(q[1], t[1])), w));
    geo_put0(2u * n + i, mkl_df_mul(mkl_df_add(mkl_df_mul(t[0], t[0]), mkl_df_mul(t[1], t[1])), w));
}

// ---------------------------------------------------------------------------
// Characteristic: in0 points DF [5, rays], in1 mask, in2 frequencies DF
// [frequencies, segments, 2]; out0 DF [2, frequencies · rays] (the second
// plane starts at element gp.chiefs = frequencies · rays).
// ---------------------------------------------------------------------------

// f32 nearest to x, ties away from zero (Rust f32::round).
fn geo_round(x: f32) -> f32 {
    let t = trunc(x);
    if (abs(mkl_sub(x, t)) >= 0.5) {
        return mkl_add(t, select(-1.0, 1.0, x > 0.0));
    }
    return t;
}

// cos and sin of |x| <= 4 in double-float: r = x - k pi/2 with a
// three-part pi/2 (k·P exact), Taylor polynomials through r^18 / r^19 on
// |r| <= pi/4 (truncation below 2^-55) in Horner form, then the quadrant's
// signs and swap; the high words are the rounded results.
const GEO_COS = array<vec2<f32>, 10>(
    vec2<f32>(0x1p+0f, 0.0f),
    vec2<f32>(-0x1p-1f, 0.0f),
    vec2<f32>(0x1.555556p-5f, -0x1.555556p-30f),
    vec2<f32>(-0x1.6c16c2p-10f, 0x1.27d27ep-35f),
    vec2<f32>(0x1.a01a02p-16f, -0x1.7f97fap-42f),
    vec2<f32>(-0x1.27e4fcp-22f, 0x1.10ec14p-47f),
    vec2<f32>(0x1.1eed8ep-29f, 0x1.ff1b12p-54f),
    vec2<f32>(-0x1.93974ap-37f, -0x1.180f94p-62f),
    vec2<f32>(0x1.ae7f3ep-45f, 0x1.ccee08p-71f),
    vec2<f32>(-0x1.682786p-53f, -0x1.dcbeccp-80f),
);
const GEO_SIN = array<vec2<f32>, 10>(
    vec2<f32>(0x1p+0f, 0.0f),
    vec2<f32>(-0x1.555556p-3f, 0x1.555556p-28f),
    vec2<f32>(0x1.111112p-7f, -0x1.dddddep-32f),
    vec2<f32>(-0x1.a01a02p-13f, 0x1.7f97fap-39f),
    vec2<f32>(0x1.71de3ap-19f, 0x1.55b1ccp-45f),
    vec2<f32>(-0x1.ae6456p-26f, -0x1.fd5138p-52f),
    vec2<f32>(0x1.612462p-33f, -0x1.8af25ep-58f),
    vec2<f32>(-0x1.ae7f3ep-41f, -0x1.ccee08p-67f),
    vec2<f32>(0x1.952c78p-49f, -0x1.f9ea56p-74f),
    vec2<f32>(-0x1.2f49b4p-57f, -0x1.a05056p-83f),
);
fn geo_cos_sin(x: f32) -> vec2<f32> {
    let k = geo_round(mkl_mul(x, bitcast<f32>(0x3f22f983u)));
    var r = mkl_two_sum(x, -mkl_mul(k, bitcast<f32>(0x3fc90fdbu)));
    r = mkl_df_add_f32(r, -mkl_mul(k, bitcast<f32>(0xb33bbd2eu)));
    r = mkl_df_add_f32(r, -mkl_mul(k, bitcast<f32>(0xa6f72cedu)));
    let r2 = mkl_df_mul(r, r);
    var c = GEO_COS[9];
    var s = GEO_SIN[9];
    for (var j = 8; j >= 0; j--) {
        c = mkl_df_add(GEO_COS[j], mkl_df_mul(r2, c));
        s = mkl_df_add(GEO_SIN[j], mkl_df_mul(r2, s));
    }
    s = mkl_df_mul(r, s);
    let q = (i32(k) % 4 + 4) % 4;
    var cs = vec2<f32>(c.x, s.x);
    if (q == 1) {
        cs = vec2<f32>(-s.x, c.x);
    } else if (q == 2) {
        cs = vec2<f32>(-c.x, -s.x);
    } else if (q == 3) {
        cs = vec2<f32>(s.x, -c.x);
    }
    return cs;
}

@compute @workgroup_size(GEO_GROUP)
fn mkl_geometry_characteristic(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    if (gid.x >= gp.count) {
        return;
    }
    let i = gp.offset + gid.x;
    let nr = gp.rays;
    let total = gp.chiefs;
    let ray = i % nr;
    let frequency = i / nr;
    geo_put0(i, vec2<f32>(0.0, 0.0));
    geo_put0(total + i, vec2<f32>(0.0, 0.0));
    if (g_in1[ray] == 0u) {
        return;
    }
    let j = 2u * (frequency * (nr / gp.segment) + ray / gp.segment);
    var a = mkl_df_add(mkl_df_mul(geo_df(0u, ray), geo_df(2u, j)),
        mkl_df_mul(geo_df(0u, nr + ray), geo_df(2u, j + 1u)));
    a = mkl_df_sub(a, vec2<f32>(geo_round(a.x), 0.0));
    let tau = vec2<f32>(bitcast<f32>(0x40c90fdbu), bitcast<f32>(0xb43bbd2eu));
    let phase = mkl_df_mul(a, tau).x;
    let w = geo_df(0u, 4u * nr + ray);
    let cs = geo_cos_sin(phase);
    geo_put0(i, mkl_df_mul_f32(w, cs.x));
    geo_put0(total + i, mkl_df_mul_f32(w, cs.y));
}
