// The sequential / path ray-trace kernel: the WGSL twin, operation for
// operation, of crates/mklight-metal/shaders/{surface_math.h,
// trace_shared.h, trace.metal} and of the resident trace (mkl_trace_resident)
// of aiming.metal. The Rust specification is mklight-gpu's trace_kernel.rs,
// surface_math.rs and fmath.rs; change all of them together.
//
// src/kernels.rs compiles this file twice, prefixed with
// `const MKL_OVERLAYS: bool = false;` (module "trace") or `true` (module
// "trace_overlays", for systems with sag overlays), as Metal compiles
// trace_overlays.metal: the overlay code costs systems without overlays
// nothing.
//
// Numerics (common.wgsl): every floating-point operation is a protected
// mkl_* operation (one IEEE rounding, never fused or reassociated); plain
// operators only negate, take absolute values, compare and select. Tests of
// values that may be NaN use the bit tests (mkl_is_nan / mkl_is_finite) or
// test NaN explicitly before a negated comparison, since a fast-math
// compiler may rewrite !(a <= b) as a > b. Hardware `/` and `sqrt` are not
// correctly rounded on every WebGPU compiler, so the kernel divides and takes
// square roots with mkl_div_rn / mkl_sqrt_rn (and the double-float
// operations built on them): every output and record is bitwise the CPU-f32
// emulation's (NaN payloads aside).
//
// Templates: surface_math.h is generic in the scalar R (float or the
// double-float MklDf). Below it appears twice, with the suffix _f for
// R = f32 and _d for R = vec2<f32> (hi, lo); the two families are the same
// text and must stay so.
//
// Bindings (group 0): 0 mkl_env (common.wgsl), 1 the parameters, then
// 2 headers, 3 surfaces, 4 coefficient pool, 5 path program (step words),
// 6 input planes, 7 output planes, 8 records, every plane `stride` long and
// starting at the launch's first ray (the host packs each launch's rays):
//   mkl_trace           in:  MKL_TRACE_IN_FLOATS float planes (bits), then
//                            MKL_TRACE_IN_UINTS planes (wavelength, system)
//                       out: MKL_TRACE_OUT_FLOATS float planes, then
//                            MKL_TRACE_OUT_UINTS planes
//                       rec: [slot][ray][MKL_TRACE_RECORD_FLOATS] floats,
//                            then [slot][ray] status words
//   mkl_trace_resident  in:  15 float planes (aiming ray layout), system,
//                            wavelength, 2 id planes, 4 report planes
//   (+ _records)        out: the same 23 planes, then 3 status planes
//                            (status, last surface, steps)
//                       rec: [slot][ray][MKL_AIM_RECORD_FLOATS] hi, then lo

struct MklWgslTraceParams {
    trace: MklTraceParams,
    // Systems in the bound set (resident rays carry device indices).
    systems: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

@group(0) @binding(1) var<uniform> prm: MklWgslTraceParams;
@group(0) @binding(2) var<storage, read> mkl_headers: array<MklPackedHeader>;
@group(0) @binding(3) var<storage, read> mkl_surfaces: array<MklPackedSurface>;
@group(0) @binding(4) var<storage, read> mkl_pool: array<f32>;
@group(0) @binding(5) var<storage, read> mkl_steps: array<u32>;
@group(0) @binding(6) var<storage, read> mkl_in: array<u32>;
@group(0) @binding(7) var<storage, read_write> mkl_out: array<u32>;
@group(0) @binding(8) var<storage, read_write> mkl_records: array<u32>;

// The ray of this invocation and the plane stride.
var<private> g_ray: u32;
var<private> g_stride: u32;

// 2^-20: vertex-sheet tolerance (trace_kernel::SHEET_TOL).
fn mkl_sheet_tol() -> f32 {
    return bitcast<f32>(0x35800000u);
}
fn mkl_flt_min() -> f32 {
    return bitcast<f32>(0x00800000u);
}

// min() that ignores a NaN operand like MSL fmin / Rust f32::min.
fn mkl_min_num(a: f32, b: f32) -> f32 {
    if (mkl_is_nan(a)) {
        return b;
    }
    if (mkl_is_nan(b)) {
        return a;
    }
    return select(b, a, a <= b);
}

// a <= b, false when either is NaN.
fn mkl_le_num(a: f32, b: f32) -> bool {
    return !mkl_is_nan(a) && !mkl_is_nan(b) && a <= b;
}

// ---------------------------------------------------------------------------
// Correctly rounded division and square root. WebGPU compilers may lower `/`
// and `sqrt` to approximations (wgpu on Metal: fast math), and in the
// trace's f32 decisions (conic roots, Newton iterations, TIR, clipping) one
// ulp can change a ray's outcome. The hardware result is corrected to the
// nearest float by exact residual tests (one fma each), so the kernel is
// bitwise the CPU-f32 emulation. Values whose residuals could leave the
// normal range keep the hardware result.
// ---------------------------------------------------------------------------

// The float one ulp from finite nonzero q, toward +inf (`up`) or -inf.
fn mkl_step_ulp(q: f32, up: bool) -> f32 {
    let b = bitcast<u32>(q);
    if ((q > 0.0) == up) {
        return bitcast<f32>(b + 1u);
    }
    return bitcast<f32>(b - 1u);
}

// |x| within [2^-100, 2^100]: residuals of x-sized operands stay normal.
fn mkl_rn_range(x: f32) -> bool {
    let e = (bitcast<u32>(x) >> 23u) & 0xFFu;
    return e >= 27u && e <= 227u;
}

// a / b rounded to nearest. From the hardware quotient, step toward a/b
// while the neighbour's residual a - q b (exact for quotients within an ulp
// of a/b) is smaller; a/b is never a midpoint.
fn mkl_div_rn(a: f32, b: f32) -> f32 {
    var q = mkl_div(a, b);
    if (!(mkl_rn_range(a) && mkl_rn_range(b) && mkl_rn_range(q))) {
        return q;
    }
    var r = mkl_fma(-q, b, a);
    for (var k = 0; k < 3; k++) {
        let q2 = mkl_step_ulp(q, (r > 0.0) == (b > 0.0));
        let r2 = mkl_fma(-q2, b, a);
        if (!(abs(r2) < abs(r))) {
            break;
        }
        q = q2;
        r = r2;
    }
    return q;
}

// sqrt(a) rounded to nearest. With s' the neighbour of s and m their
// midpoint, a - m^2 = (a - s s') - ((s' - s)/2)^2, where a - s s' is a
// multiple of a grid coarser than the square: a lies above m exactly when
// fma(-s, s', a) > 0 (no midpoint is a square root of a float).
fn mkl_sqrt_rn(a: f32) -> f32 {
    var s = mkl_sqrt(a);
    if (!(a > 0.0 && mkl_rn_range(a) && mkl_is_finite(s))) {
        return s;
    }
    for (var k = 0; k < 4; k++) {
        let up = mkl_step_ulp(s, true);
        if (mkl_fma(-s, up, a) > 0.0) {
            s = up;
            continue;
        }
        let down = mkl_step_ulp(s, false);
        if (mkl_fma(-down, s, a) <= 0.0) {
            s = down;
            continue;
        }
        break;
    }
    return s;
}

// common.wgsl's mkl_df_div, mkl_df_sqrt and mkl_hypot3 on the rounded
// operations.
fn mkl_df_div_rn(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let th = mkl_div_rn(x.x, y.x);
    let r = mkl_df_mul_f32(y, th);
    let pi_h = mkl_sub(x.x, r.x);
    let dl = mkl_sub(x.y, r.y);
    let d = mkl_add(pi_h, dl);
    let tl = mkl_div_rn(d, y.x);
    return mkl_fast_two_sum(th, tl);
}

fn mkl_df_sqrt_rn(x: vec2<f32>) -> vec2<f32> {
    if (x.x <= 0.0) {
        if (x.x == 0.0) {
            return vec2<f32>(0.0, 0.0);
        }
        return vec2<f32>(mkl_qnan(), mkl_qnan());
    }
    let s = mkl_sqrt_rn(x.x);
    let p = mkl_two_prod(s, s);
    let r = mkl_add(mkl_sub(mkl_sub(x.x, p.x), p.y), x.y);
    let e = mkl_div_rn(r, mkl_mul(2.0, s));
    return mkl_fast_two_sum(s, e);
}

fn mkl_hypot3_rn(x: f32, y: f32, z: f32) -> f32 {
    let m = mkl_max_num(mkl_max_num(abs(x), abs(y)), abs(z));
    let sp = mkl_hypot_special(m, mkl_is_nan(x) || mkl_is_nan(y) || mkl_is_nan(z));
    if (sp.y != 0.0) {
        return sp.x;
    }
    let sc = mkl_hypot_scale(m);
    let v = vec3<f32>(mkl_mul(x, sc.x), mkl_mul(y, sc.x), mkl_mul(z, sc.x));
    return mkl_mul(mkl_sqrt_rn(mkl_dot3(v, v)), sc.y);
}

// ---------------------------------------------------------------------------
// Scalar primitives of the two R types (surface_math.h's MklDf operators,
// mkl_c, mkl_parts, mkl_sqrt, mkl_le, mkl_zero, mkl_load).
// ---------------------------------------------------------------------------
fn add_f(a: f32, b: f32) -> f32 {
    return mkl_add(a, b);
}
fn sub_f(a: f32, b: f32) -> f32 {
    return mkl_sub(a, b);
}
fn mul_f(a: f32, b: f32) -> f32 {
    return mkl_mul(a, b);
}
fn div_f(a: f32, b: f32) -> f32 {
    return mkl_div_rn(a, b);
}
fn neg_f(a: f32) -> f32 {
    return -a;
}
fn sqrt_f(a: f32) -> f32 {
    return mkl_sqrt_rn(a);
}
fn c_f(v: f32) -> f32 {
    return v;
}
fn parts_f(hi: f32, lo: f32) -> f32 {
    return hi;
}
fn le_f(a: f32, b: f32) -> bool {
    return mkl_le_num(a, b);
}
fn zero_f(a: f32) -> bool {
    return a == 0.0;
}
// Element i of the double-float array at float index `base` of the pool.
fn load_f(base: u32, i: u32) -> f32 {
    return mkl_pool[base + 2u * i];
}

fn add_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_add(a, b);
}
fn sub_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_sub(a, b);
}
fn mul_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_mul(a, b);
}
fn div_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_div_rn(a, b);
}
fn neg_d(a: vec2<f32>) -> vec2<f32> {
    return mkl_df_neg(a);
}
fn sqrt_d(a: vec2<f32>) -> vec2<f32> {
    return mkl_df_sqrt_rn(a);
}
fn c_d(v: f32) -> vec2<f32> {
    return vec2<f32>(v, 0.0);
}
fn parts_d(hi: f32, lo: f32) -> vec2<f32> {
    return vec2<f32>(hi, lo);
}
fn le_d(a: vec2<f32>, b: f32) -> bool {
    return mkl_le_num(mkl_df_add_f32(a, -b).x, 0.0);
}
fn zero_d(a: vec2<f32>) -> bool {
    return a.x == 0.0;
}
fn load_d(base: u32, i: u32) -> vec2<f32> {
    return vec2<f32>(mkl_pool[base + 2u * i], mkl_pool[base + 2u * i + 1u]);
}

// First-order jets (value, d/dx, d/dy), the sag of a surface with its
// slopes, and the scaled pupil coordinates u, v and s = u^2 + v^2.
struct MklJetF {
    v: f32,
    x: f32,
    y: f32,
}
struct MklJetD {
    v: vec2<f32>,
    x: vec2<f32>,
    y: vec2<f32>,
}
struct MklSagF {
    ok: bool,
    j: MklJetF,
}
struct MklSagD {
    ok: bool,
    j: MklJetD,
}
struct MklUvsF {
    u: MklJetF,
    v: MklJetF,
    s: MklJetF,
}
struct MklUvsD {
    u: MklJetD,
    v: MklJetD,
    s: MklJetD,
}

// Float index of a surface's coefficient data in the pool.
fn mkl_surface_data(s: u32) -> u32 {
    return 2u * mkl_surfaces[s].coefficient_offset;
}

// ---- R = f32 (suffix _f) --------------------------------------------------
fn mkl_jconst_f(v: f32) -> MklJetF {
    return MklJetF(v, c_f(0.0), c_f(0.0));
}
fn mkl_jvar_f(v: f32, is_x: bool) -> MklJetF {
    if (is_x) {
        return MklJetF(v, c_f(1.0), c_f(0.0));
    }
    return MklJetF(v, c_f(0.0), c_f(1.0));
}
fn mkl_jchain_f(a: MklJetF, v: f32, d: f32) -> MklJetF {
    return MklJetF(v, mul_f(d, a.x), mul_f(d, a.y));
}
fn mkl_jscale_f(a: MklJetF, b: f32) -> MklJetF {
    return MklJetF(mul_f(a.v, b), mul_f(a.x, b), mul_f(a.y, b));
}
fn mkl_jadd_f(a: MklJetF, b: f32) -> MklJetF {
    return MklJetF(add_f(a.v, b), a.x, a.y);
}
fn mkl_jsqrt_f(a: MklJetF) -> MklJetF {
    let v = sqrt_f(a.v);
    return mkl_jchain_f(a, v, div_f(c_f(0.5), v));
}
fn mkl_jrecip_f(a: MklJetF) -> MklJetF {
    let r = div_f(c_f(1.0), a.v);
    return mkl_jchain_f(a, r, mul_f(neg_f(r), r));
}
fn mkl_jsum_f(a: MklJetF, b: MklJetF) -> MklJetF {
    return MklJetF(add_f(a.v, b.v), add_f(a.x, b.x), add_f(a.y, b.y));
}
fn mkl_jneg_f(a: MklJetF) -> MklJetF {
    return mkl_jscale_f(a, c_f(-1.0));
}
fn mkl_jdiff_f(a: MklJetF, b: MklJetF) -> MklJetF {
    return mkl_jsum_f(a, mkl_jneg_f(b));
}
fn mkl_jmul_f(a: MklJetF, b: MklJetF) -> MklJetF {
    return MklJetF(
        mul_f(a.v, b.v),
        add_f(mul_f(a.x, b.v), mul_f(a.v, b.x)),
        add_f(mul_f(a.y, b.v), mul_f(a.v, b.y)),
    );
}
fn mkl_jdiv_f(a: MklJetF, b: MklJetF) -> MklJetF {
    return mkl_jmul_f(a, mkl_jrecip_f(b));
}
fn mkl_le_value_f(a: f32, b: f32) -> bool {
    return le_f(sub_f(a, b), 0.0);
}
fn mkl_sag_fail_f() -> MklSagF {
    return MklSagF(false, mkl_jconst_f(c_f(0.0)));
}
fn mkl_poly_f(pool: u32, start: u32, n: u32, t: MklJetF) -> MklJetF {
    var p = mkl_jconst_f(c_f(0.0));
    for (var j = i32(n) - 1; j >= 0; j--) {
        p = mkl_jadd_f(mkl_jmul_f(p, t), load_f(pool, start + u32(j)));
    }
    return p;
}
fn mkl_even_f(pool: u32, start: u32, n: u32, t: MklJetF) -> MklJetF {
    return mkl_jmul_f(mkl_jmul_f(mkl_poly_f(pool, start, n, t), t), t);
}
fn mkl_conic_sag_jet_f(c: f32, k: f32, t: MklJetF) -> MklSagF {
    let arg = sub_f(c_f(1.0), mul_f(mul_f(mul_f(add_f(c_f(1.0), k), c), c), t.v));
    if (!mkl_le_value_f(c_f(0.0), arg)) {
        return mkl_sag_fail_f();
    }
    let w = sqrt_f(arg);
    return MklSagF(true, mkl_jchain_f(t, div_f(mul_f(c, t.v), add_f(c_f(1.0), w)), div_f(c, mul_f(c_f(2.0), w))));
}
fn mkl_scaled_jet_f(x: MklJetF, y: MklJetF, radius: f32) -> MklUvsF {
    let inv = div_f(c_f(1.0), radius);
    let u = mkl_jscale_f(x, inv);
    let v = mkl_jscale_f(y, inv);
    return MklUvsF(u, v, mkl_jsum_f(mkl_jmul_f(u, u), mkl_jmul_f(v, v)));
}
// mkl_normalized_jet's test on a scaled jet.
fn mkl_normalized_f(q: MklUvsF) -> bool {
    return mkl_le_value_f(q.s.v, parts_f(1.0, 2e-12));
}
fn mkl_jacobi_f(n: u32, b: f32, t: MklJetF) -> MklJetF {
    var p0 = mkl_jconst_f(c_f(1.0));
    if (n == 0u) {
        return p0;
    }
    var p1 = mkl_jadd_f(mkl_jscale_f(mkl_jadd_f(t, c_f(-1.0)), div_f(add_f(b, c_f(2.0)), c_f(2.0))), c_f(1.0));
    for (var ki = 1u; ki < n; ki++) {
        let k = c_f(f32(ki));
        let s = add_f(mul_f(c_f(2.0), k), b);
        let a1 = mul_f(mul_f(mul_f(c_f(2.0), add_f(k, c_f(1.0))), add_f(add_f(k, b), c_f(1.0))), s);
        let a2 = mkl_jscale_f(mkl_jadd_f(mkl_jscale_f(t, mul_f(add_f(s, c_f(2.0)), s)), neg_f(mul_f(b, b))), add_f(s, c_f(1.0)));
        let a3 = mul_f(mul_f(mul_f(c_f(2.0), k), add_f(k, b)), add_f(s, c_f(2.0)));
        let next = mkl_jscale_f(mkl_jdiff_f(mkl_jmul_f(a2, p1), mkl_jscale_f(p0, a3)), div_f(c_f(1.0), a1));
        p0 = p1;
        p1 = next;
    }
    return p1;
}
// Q-con recurrence: v = 2k+4, d = 2(k+1)(k+5)v.
fn mkl_qcon_d_f(ki: u32, v: f32) -> f32 {
    let k = c_f(f32(ki));
    return mul_f(mul_f(mul_f(c_f(2.0), add_f(k, c_f(1.0))), add_f(k, c_f(5.0))), v);
}
fn mkl_qcon_a_f(ki: u32, s: MklJetF) -> MklJetF {
    let v = add_f(mul_f(c_f(2.0), c_f(f32(ki))), c_f(4.0));
    let d = mkl_qcon_d_f(ki, v);
    return mkl_jscale_f(
        mkl_jadd_f(mkl_jscale_f(mkl_jadd_f(mkl_jscale_f(s, c_f(2.0)), c_f(-1.0)), mul_f(add_f(v, c_f(2.0)), v)), c_f(-16.0)),
        div_f(add_f(v, c_f(1.0)), d),
    );
}
fn mkl_qcon_b_f(ki: u32) -> f32 {
    let k = c_f(f32(ki));
    let v = add_f(mul_f(c_f(2.0), k), c_f(4.0));
    return div_f(mul_f(mul_f(mul_f(c_f(2.0), k), add_f(k, c_f(4.0))), add_f(v, c_f(2.0))), mkl_qcon_d_f(ki, v));
}
fn mkl_qcon_f(pool: u32, n: u32, s: MklJetF) -> MklJetF {
    var b1 = mkl_jconst_f(c_f(0.0));
    var b2 = b1;
    for (var k = i32(n) - 1; k >= 0; k--) {
        let b = mkl_jadd_f(
            mkl_jdiff_f(mkl_jmul_f(mkl_qcon_a_f(u32(k), s), b1), mkl_jscale_f(b2, mkl_qcon_b_f(u32(k + 1)))),
            load_f(pool, u32(k)),
        );
        b2 = b1;
        b1 = b;
    }
    return mkl_jmul_f(mkl_jmul_f(b1, s), s);
}
fn mkl_qbfs_f(pool: u32, n: u32, s: MklJetF) -> MklJetF {
    var a1 = mkl_jconst_f(c_f(0.0));
    var a2 = a1;
    for (var m = i32(n) - 1; m >= 0; m--) {
        let a = mkl_jadd_f(
            mkl_jdiff_f(mkl_jmul_f(mkl_jadd_f(mkl_jscale_f(s, c_f(-4.0)), c_f(2.0)), a1), a2),
            load_f(pool, u32(m)),
        );
        a2 = a1;
        a1 = a;
    }
    return mkl_jscale_f(mkl_jsum_f(a1, a2), c_f(2.0));
}
fn mkl_zernike_f(pool: u32, n: u32, standard: bool, x: MklJetF, y: MklJetF, s: MklJetF) -> MklJetF {
    var z = mkl_jconst_f(c_f(0.0));
    for (var j = 1u; j <= n; j++) {
        let a = load_f(pool, j - 1u);
        if (zero_f(a)) {
            continue;
        }
        var degree: u32;
        var m: i32;
        if (standard) {
            degree = 0u;
            while ((degree + 1u) * (degree + 2u) / 2u < j) {
                degree++;
            }
            let k = j - degree * (degree + 1u) / 2u - 1u;
            var am: i32;
            if (degree % 2u == 0u) {
                am = i32(2u * ((k + 1u) / 2u));
            } else {
                am = i32(2u * (k / 2u) + 1u);
            }
            m = select(-am, am, j % 2u == 0u);
        } else if (j == 37u) {
            degree = 12u;
            m = 0;
        } else {
            var k = 0u;
            while ((k + 1u) * (k + 1u) < j) {
                k++;
            }
            let off = j - 1u - k * k;
            let am = i32(k - off / 2u);
            degree = 2u * k - u32(am);
            m = select(am, -am, off % 2u == 1u);
        }
        let am = u32(abs(m));
        var re = mkl_jconst_f(c_f(1.0));
        var im = mkl_jconst_f(c_f(0.0));
        for (var i = 0u; i < am; i++) {
            let r = mkl_jdiff_f(mkl_jmul_f(re, x), mkl_jmul_f(im, y));
            im = mkl_jsum_f(mkl_jmul_f(re, y), mkl_jmul_f(im, x));
            re = r;
        }
        var scale = c_f(1.0);
        if (standard) {
            scale = sqrt_f(c_f(f32((degree + 1u) * select(2u, 1u, m == 0))));
        }
        var part = im;
        if (m >= 0) {
            part = re;
        }
        let radial = mkl_jacobi_f((degree - am) / 2u, c_f(f32(am)), mkl_jadd_f(mkl_jscale_f(s, c_f(2.0)), c_f(-1.0)));
        z = mkl_jsum_f(z, mkl_jscale_f(mkl_jmul_f(radial, part), mul_f(a, scale)));
    }
    return z;
}
fn mkl_surface_domain_f(s: u32, x: f32, y: f32) -> bool {
    let data = mkl_surface_data(s);
    return mkl_normalized_f(mkl_scaled_jet_f(mkl_jvar_f(x, true), mkl_jvar_f(y, false), load_f(data, 0u)));
}
fn mkl_phase_f(s: u32, x: f32, y: f32) -> MklJetF {
    let data = mkl_surface_data(s);
    let q = mkl_scaled_jet_f(mkl_jvar_f(x, true), mkl_jvar_f(y, false), load_f(data, 0u));
    let n = u32(mkl_pool[data + 2u * MKL_COEFF_NA]);
    return mkl_jmul_f(mkl_poly_f(data, MKL_COEFF_PREFIX + MKL_MAX_WAVELENGTHS, n, q.s), q.s);
}
fn mkl_extended_sag_f(s: u32, xv: f32, yv: f32) -> MklSagF {
    let data = mkl_surface_data(s);
    let radius = load_f(data, 0u);
    let na = u32(mkl_pool[data + 2u * MKL_COEFF_NA]);
    let nb = u32(mkl_pool[data + 2u * MKL_COEFF_NB]);
    let coeff = data + 2u * MKL_COEFF_PREFIX;
    let x = mkl_jvar_f(xv, true);
    let y = mkl_jvar_f(yv, false);
    let r2 = mkl_jsum_f(mkl_jmul_f(x, x), mkl_jmul_f(y, y));
    let c = parts_f(mkl_surfaces[s].curvature, mkl_surfaces[s].curvature_lo);
    let k = parts_f(mkl_surfaces[s].conic, mkl_surfaces[s].conic_lo);
    let shape = mkl_surfaces[s].shape;
    if (shape == MKL_SHAPE_ODD_ASPHERE) {
        let base = mkl_conic_sag_jet_f(c, k, r2);
        if (!base.ok) {
            return base;
        }
        if (zero_f(x.v) && zero_f(y.v)) {
            return base;
        }
        var radial: MklJetF;
        if (zero_f(r2.v)) {
            let ax = select(x.v, neg_f(x.v), le_f(x.v, 0.0));
            let ay = select(y.v, neg_f(y.v), le_f(y.v, 0.0));
            let scale = select(ax, ay, mkl_le_value_f(ax, ay));
            let u = div_f(x.v, scale);
            let v = div_f(y.v, scale);
            let w = sqrt_f(add_f(mul_f(u, u), mul_f(v, v)));
            radial = MklJetF(mul_f(scale, w), div_f(u, w), div_f(v, w));
        } else {
            radial = mkl_jsqrt_f(r2);
        }
        let r = mkl_jscale_f(radial, div_f(c_f(1.0), radius));
        return MklSagF(true, mkl_jsum_f(base.j, mkl_jmul_f(mkl_poly_f(coeff, 0u, na, r), r)));
    }
    if (shape == MKL_SHAPE_TOROID || shape == MKL_SHAPE_BICONIC) {
        let cx = load_f(data, MKL_COEFF_CX);
        let inv = div_f(c_f(1.0), radius);
        let xx = mkl_jscale_f(x, inv);
        let yy = mkl_jscale_f(y, inv);
        if (shape == MKL_SHAPE_TOROID) {
            let base = mkl_conic_sag_jet_f(c, k, mkl_jmul_f(y, y));
            if (!base.ok) {
                return base;
            }
            let zy = mkl_jsum_f(base.j, mkl_even_f(coeff, na, nb, mkl_jmul_f(yy, yy)));
            let a = mkl_jadd_f(mkl_jscale_f(mkl_jneg_f(zy), cx), c_f(1.0));
            let q = mkl_jdiff_f(mkl_jmul_f(a, a), mkl_jscale_f(mkl_jmul_f(x, x), mul_f(cx, cx)));
            if (le_f(a.v, 0.0) || !mkl_le_value_f(c_f(0.0), q.v)) {
                return mkl_sag_fail_f();
            }
            return MklSagF(true, mkl_jsum_f(zy, mkl_jdiv_f(mkl_jscale_f(mkl_jmul_f(x, x), cx), mkl_jsum_f(a, mkl_jsqrt_f(q)))));
        }
        let kx = load_f(data, MKL_COEFF_KX);
        let arg = mkl_jadd_f(
            mkl_jsum_f(
                mkl_jscale_f(mkl_jmul_f(x, x), mul_f(mul_f(neg_f(add_f(c_f(1.0), kx)), cx), cx)),
                mkl_jscale_f(mkl_jmul_f(y, y), mul_f(mul_f(neg_f(add_f(c_f(1.0), k)), c), c)),
            ),
            c_f(1.0),
        );
        if (!mkl_le_value_f(c_f(0.0), arg.v)) {
            return mkl_sag_fail_f();
        }
        let num = mkl_jsum_f(mkl_jscale_f(mkl_jmul_f(x, x), cx), mkl_jscale_f(mkl_jmul_f(y, y), c));
        let z = mkl_jsum_f(
            mkl_jsum_f(mkl_jdiv_f(num, mkl_jadd_f(mkl_jsqrt_f(arg), c_f(1.0))), mkl_even_f(coeff, 0u, na, mkl_jmul_f(xx, xx))),
            mkl_even_f(coeff, na, nb, mkl_jmul_f(yy, yy)),
        );
        return MklSagF(true, z);
    }
    let q = mkl_scaled_jet_f(x, y, radius);
    if (!mkl_normalized_f(q)) {
        return mkl_sag_fail_f();
    }
    let base = mkl_conic_sag_jet_f(c, k, r2);
    if (!base.ok) {
        return base;
    }
    let u = q.u;
    let v = q.v;
    let t = q.s;
    if (shape == MKL_SHAPE_Q_CON) {
        return MklSagF(true, mkl_jsum_f(base.j, mkl_qcon_f(coeff, na, t)));
    }
    if (shape == MKL_SHAPE_Q_BFS) {
        if (na == 0u) {
            return base;
        }
        let w = mkl_jsqrt_f(mkl_jadd_f(mkl_jscale_f(r2, mul_f(neg_f(c), c)), c_f(1.0)));
        if (zero_f(w.v)) {
            return mkl_sag_fail_f();
        }
        let tt = mkl_jmul_f(mkl_jmul_f(t, mkl_jadd_f(mkl_jneg_f(t), c_f(1.0))), mkl_qbfs_f(coeff, na, t));
        return MklSagF(true, mkl_jsum_f(base.j, mkl_jdiv_f(tt, w)));
    }
    if (shape == MKL_SHAPE_XY) {
        var start = 0u;
        var degree = 1u;
        var z = base.j;
        while (start < na) {
            var row = mkl_jconst_f(c_f(0.0));
            var vn = mkl_jconst_f(c_f(1.0));
            for (var n = 0u; n <= degree; n++) {
                var a = c_f(0.0);
                if (start + n < na) {
                    a = load_f(coeff, start + n);
                }
                row = mkl_jsum_f(mkl_jmul_f(row, u), mkl_jscale_f(vn, a));
                vn = mkl_jmul_f(vn, v);
            }
            z = mkl_jsum_f(z, row);
            start += degree + 1u;
            degree++;
        }
        return MklSagF(true, z);
    }
    let standard = mkl_pool[data + 2u * MKL_COEFF_KIND] != 0.0;
    return MklSagF(true, mkl_jsum_f(base.j, mkl_zernike_f(coeff, na, standard, u, v, t)));
}
// The base of an overlaid surface may be legacy (or a DOE's legacy base).
fn mkl_legacy_sag_f(s: u32, xv: f32, yv: f32) -> MklSagF {
    let x = mkl_jvar_f(xv, true);
    let y = mkl_jvar_f(yv, false);
    let r2 = mkl_jsum_f(mkl_jmul_f(x, x), mkl_jmul_f(y, y));
    let sf = &mkl_surfaces[s];
    let base = mkl_conic_sag_jet_f(parts_f((*sf).curvature, (*sf).curvature_lo), parts_f((*sf).conic, (*sf).conic_lo), r2);
    if (!base.ok) {
        return base;
    }
    let t = mkl_jscale_f(r2, c_f((*sf).asphere_inv_norm2));
    var p = mkl_jconst_f(c_f(0.0));
    for (var j = i32((*sf).num_asphere) - 1; j >= 0; j--) {
        p = mkl_jadd_f(mkl_jmul_f(p, t), parts_f((*sf).asphere[j], (*sf).asphere_lo[j]));
    }
    return MklSagF(true, mkl_jsum_f(base.j, mkl_jmul_f(mkl_jmul_f(p, t), t)));
}
// Sag jet of an extended surface or, with MKL_OVERLAYS, of any surface with
// MKL_SURF_HAS_SAG_OVERLAY. Code without MKL_OVERLAYS never sees the overlay.
fn mkl_surface_eval_f(s: u32, xv: f32, yv: f32) -> MklSagF {
    if (!MKL_OVERLAYS || (mkl_surfaces[s].flags & MKL_SURF_HAS_SAG_OVERLAY) == 0u) {
        return mkl_extended_sag_f(s, xv, yv);
    }
    let data = mkl_surface_data(s);
    var shape = mkl_surfaces[s].shape;
    if (shape == MKL_SHAPE_DIFFRACTIVE) {
        shape = u32(mkl_pool[data + 2u * MKL_COEFF_KIND]);
    }
    var base: MklSagF;
    if (shape <= MKL_SHAPE_EVEN_ASPHERE) {
        base = mkl_legacy_sag_f(s, xv, yv);
    } else {
        base = mkl_extended_sag_f(s, xv, yv);
    }
    if (!base.ok) {
        return base;
    }
    let d = data + 2u * (mkl_surfaces[s].coefficient_count - MKL_OVERLAY_COUNT);
    let inv = load_f(d, MKL_OVERLAY_INV_RADIUS);
    let c = load_f(d, MKL_OVERLAY_COS);
    let sn = load_f(d, MKL_OVERLAY_SIN);
    let s4 = load_f(d, MKL_OVERLAY_S4);
    let two = c_f(2.0);
    let u = mul_f(xv, inv);
    let v = mul_f(yv, inv);
    let t = add_f(mul_f(u, u), mul_f(v, v));
    let q = add_f(load_f(d, MKL_OVERLAY_S2), mul_f(t, s4));
    let w = add_f(q, mul_f(t, s4));
    let o = MklJetF(
        add_f(add_f(mul_f(c, mul_f(sub_f(u, v), add_f(u, v))), mul_f(mul_f(two, sn), mul_f(u, v))), mul_f(t, q)),
        mul_f(mul_f(two, inv), add_f(mul_f(u, add_f(c, w)), mul_f(sn, v))),
        mul_f(mul_f(two, inv), add_f(mul_f(v, sub_f(w, c)), mul_f(sn, u))),
    );
    return MklSagF(true, mkl_jsum_f(base.j, o));
}

// ---- R = double-float (suffix _d) -----------------------------------------
fn mkl_jconst_d(v: vec2<f32>) -> MklJetD {
    return MklJetD(v, c_d(0.0), c_d(0.0));
}
fn mkl_jvar_d(v: vec2<f32>, is_x: bool) -> MklJetD {
    if (is_x) {
        return MklJetD(v, c_d(1.0), c_d(0.0));
    }
    return MklJetD(v, c_d(0.0), c_d(1.0));
}
fn mkl_jchain_d(a: MklJetD, v: vec2<f32>, d: vec2<f32>) -> MklJetD {
    return MklJetD(v, mul_d(d, a.x), mul_d(d, a.y));
}
fn mkl_jscale_d(a: MklJetD, b: vec2<f32>) -> MklJetD {
    return MklJetD(mul_d(a.v, b), mul_d(a.x, b), mul_d(a.y, b));
}
fn mkl_jadd_d(a: MklJetD, b: vec2<f32>) -> MklJetD {
    return MklJetD(add_d(a.v, b), a.x, a.y);
}
fn mkl_jsqrt_d(a: MklJetD) -> MklJetD {
    let v = sqrt_d(a.v);
    return mkl_jchain_d(a, v, div_d(c_d(0.5), v));
}
fn mkl_jrecip_d(a: MklJetD) -> MklJetD {
    let r = div_d(c_d(1.0), a.v);
    return mkl_jchain_d(a, r, mul_d(neg_d(r), r));
}
fn mkl_jsum_d(a: MklJetD, b: MklJetD) -> MklJetD {
    return MklJetD(add_d(a.v, b.v), add_d(a.x, b.x), add_d(a.y, b.y));
}
fn mkl_jneg_d(a: MklJetD) -> MklJetD {
    return mkl_jscale_d(a, c_d(-1.0));
}
fn mkl_jdiff_d(a: MklJetD, b: MklJetD) -> MklJetD {
    return mkl_jsum_d(a, mkl_jneg_d(b));
}
fn mkl_jmul_d(a: MklJetD, b: MklJetD) -> MklJetD {
    return MklJetD(
        mul_d(a.v, b.v),
        add_d(mul_d(a.x, b.v), mul_d(a.v, b.x)),
        add_d(mul_d(a.y, b.v), mul_d(a.v, b.y)),
    );
}
fn mkl_jdiv_d(a: MklJetD, b: MklJetD) -> MklJetD {
    return mkl_jmul_d(a, mkl_jrecip_d(b));
}
fn mkl_le_value_d(a: vec2<f32>, b: vec2<f32>) -> bool {
    return le_d(sub_d(a, b), 0.0);
}
fn mkl_sag_fail_d() -> MklSagD {
    return MklSagD(false, mkl_jconst_d(c_d(0.0)));
}
fn mkl_poly_d(pool: u32, start: u32, n: u32, t: MklJetD) -> MklJetD {
    var p = mkl_jconst_d(c_d(0.0));
    for (var j = i32(n) - 1; j >= 0; j--) {
        p = mkl_jadd_d(mkl_jmul_d(p, t), load_d(pool, start + u32(j)));
    }
    return p;
}
fn mkl_even_d(pool: u32, start: u32, n: u32, t: MklJetD) -> MklJetD {
    return mkl_jmul_d(mkl_jmul_d(mkl_poly_d(pool, start, n, t), t), t);
}
fn mkl_conic_sag_jet_d(c: vec2<f32>, k: vec2<f32>, t: MklJetD) -> MklSagD {
    let arg = sub_d(c_d(1.0), mul_d(mul_d(mul_d(add_d(c_d(1.0), k), c), c), t.v));
    if (!mkl_le_value_d(c_d(0.0), arg)) {
        return mkl_sag_fail_d();
    }
    let w = sqrt_d(arg);
    return MklSagD(true, mkl_jchain_d(t, div_d(mul_d(c, t.v), add_d(c_d(1.0), w)), div_d(c, mul_d(c_d(2.0), w))));
}
fn mkl_scaled_jet_d(x: MklJetD, y: MklJetD, radius: vec2<f32>) -> MklUvsD {
    let inv = div_d(c_d(1.0), radius);
    let u = mkl_jscale_d(x, inv);
    let v = mkl_jscale_d(y, inv);
    return MklUvsD(u, v, mkl_jsum_d(mkl_jmul_d(u, u), mkl_jmul_d(v, v)));
}
// mkl_normalized_jet's test on a scaled jet.
fn mkl_normalized_d(q: MklUvsD) -> bool {
    return mkl_le_value_d(q.s.v, parts_d(1.0, 2e-12));
}
fn mkl_jacobi_d(n: u32, b: vec2<f32>, t: MklJetD) -> MklJetD {
    var p0 = mkl_jconst_d(c_d(1.0));
    if (n == 0u) {
        return p0;
    }
    var p1 = mkl_jadd_d(mkl_jscale_d(mkl_jadd_d(t, c_d(-1.0)), div_d(add_d(b, c_d(2.0)), c_d(2.0))), c_d(1.0));
    for (var ki = 1u; ki < n; ki++) {
        let k = c_d(f32(ki));
        let s = add_d(mul_d(c_d(2.0), k), b);
        let a1 = mul_d(mul_d(mul_d(c_d(2.0), add_d(k, c_d(1.0))), add_d(add_d(k, b), c_d(1.0))), s);
        let a2 = mkl_jscale_d(mkl_jadd_d(mkl_jscale_d(t, mul_d(add_d(s, c_d(2.0)), s)), neg_d(mul_d(b, b))), add_d(s, c_d(1.0)));
        let a3 = mul_d(mul_d(mul_d(c_d(2.0), k), add_d(k, b)), add_d(s, c_d(2.0)));
        let next = mkl_jscale_d(mkl_jdiff_d(mkl_jmul_d(a2, p1), mkl_jscale_d(p0, a3)), div_d(c_d(1.0), a1));
        p0 = p1;
        p1 = next;
    }
    return p1;
}
// Q-con recurrence: v = 2k+4, d = 2(k+1)(k+5)v.
fn mkl_qcon_d_d(ki: u32, v: vec2<f32>) -> vec2<f32> {
    let k = c_d(f32(ki));
    return mul_d(mul_d(mul_d(c_d(2.0), add_d(k, c_d(1.0))), add_d(k, c_d(5.0))), v);
}
fn mkl_qcon_a_d(ki: u32, s: MklJetD) -> MklJetD {
    let v = add_d(mul_d(c_d(2.0), c_d(f32(ki))), c_d(4.0));
    let d = mkl_qcon_d_d(ki, v);
    return mkl_jscale_d(
        mkl_jadd_d(mkl_jscale_d(mkl_jadd_d(mkl_jscale_d(s, c_d(2.0)), c_d(-1.0)), mul_d(add_d(v, c_d(2.0)), v)), c_d(-16.0)),
        div_d(add_d(v, c_d(1.0)), d),
    );
}
fn mkl_qcon_b_d(ki: u32) -> vec2<f32> {
    let k = c_d(f32(ki));
    let v = add_d(mul_d(c_d(2.0), k), c_d(4.0));
    return div_d(mul_d(mul_d(mul_d(c_d(2.0), k), add_d(k, c_d(4.0))), add_d(v, c_d(2.0))), mkl_qcon_d_d(ki, v));
}
fn mkl_qcon_d(pool: u32, n: u32, s: MklJetD) -> MklJetD {
    var b1 = mkl_jconst_d(c_d(0.0));
    var b2 = b1;
    for (var k = i32(n) - 1; k >= 0; k--) {
        let b = mkl_jadd_d(
            mkl_jdiff_d(mkl_jmul_d(mkl_qcon_a_d(u32(k), s), b1), mkl_jscale_d(b2, mkl_qcon_b_d(u32(k + 1)))),
            load_d(pool, u32(k)),
        );
        b2 = b1;
        b1 = b;
    }
    return mkl_jmul_d(mkl_jmul_d(b1, s), s);
}
fn mkl_qbfs_d(pool: u32, n: u32, s: MklJetD) -> MklJetD {
    var a1 = mkl_jconst_d(c_d(0.0));
    var a2 = a1;
    for (var m = i32(n) - 1; m >= 0; m--) {
        let a = mkl_jadd_d(
            mkl_jdiff_d(mkl_jmul_d(mkl_jadd_d(mkl_jscale_d(s, c_d(-4.0)), c_d(2.0)), a1), a2),
            load_d(pool, u32(m)),
        );
        a2 = a1;
        a1 = a;
    }
    return mkl_jscale_d(mkl_jsum_d(a1, a2), c_d(2.0));
}
fn mkl_zernike_d(pool: u32, n: u32, standard: bool, x: MklJetD, y: MklJetD, s: MklJetD) -> MklJetD {
    var z = mkl_jconst_d(c_d(0.0));
    for (var j = 1u; j <= n; j++) {
        let a = load_d(pool, j - 1u);
        if (zero_d(a)) {
            continue;
        }
        var degree: u32;
        var m: i32;
        if (standard) {
            degree = 0u;
            while ((degree + 1u) * (degree + 2u) / 2u < j) {
                degree++;
            }
            let k = j - degree * (degree + 1u) / 2u - 1u;
            var am: i32;
            if (degree % 2u == 0u) {
                am = i32(2u * ((k + 1u) / 2u));
            } else {
                am = i32(2u * (k / 2u) + 1u);
            }
            m = select(-am, am, j % 2u == 0u);
        } else if (j == 37u) {
            degree = 12u;
            m = 0;
        } else {
            var k = 0u;
            while ((k + 1u) * (k + 1u) < j) {
                k++;
            }
            let off = j - 1u - k * k;
            let am = i32(k - off / 2u);
            degree = 2u * k - u32(am);
            m = select(am, -am, off % 2u == 1u);
        }
        let am = u32(abs(m));
        var re = mkl_jconst_d(c_d(1.0));
        var im = mkl_jconst_d(c_d(0.0));
        for (var i = 0u; i < am; i++) {
            let r = mkl_jdiff_d(mkl_jmul_d(re, x), mkl_jmul_d(im, y));
            im = mkl_jsum_d(mkl_jmul_d(re, y), mkl_jmul_d(im, x));
            re = r;
        }
        var scale = c_d(1.0);
        if (standard) {
            scale = sqrt_d(c_d(f32((degree + 1u) * select(2u, 1u, m == 0))));
        }
        var part = im;
        if (m >= 0) {
            part = re;
        }
        let radial = mkl_jacobi_d((degree - am) / 2u, c_d(f32(am)), mkl_jadd_d(mkl_jscale_d(s, c_d(2.0)), c_d(-1.0)));
        z = mkl_jsum_d(z, mkl_jscale_d(mkl_jmul_d(radial, part), mul_d(a, scale)));
    }
    return z;
}
fn mkl_surface_domain_d(s: u32, x: vec2<f32>, y: vec2<f32>) -> bool {
    let data = mkl_surface_data(s);
    return mkl_normalized_d(mkl_scaled_jet_d(mkl_jvar_d(x, true), mkl_jvar_d(y, false), load_d(data, 0u)));
}
fn mkl_phase_d(s: u32, x: vec2<f32>, y: vec2<f32>) -> MklJetD {
    let data = mkl_surface_data(s);
    let q = mkl_scaled_jet_d(mkl_jvar_d(x, true), mkl_jvar_d(y, false), load_d(data, 0u));
    let n = u32(mkl_pool[data + 2u * MKL_COEFF_NA]);
    return mkl_jmul_d(mkl_poly_d(data, MKL_COEFF_PREFIX + MKL_MAX_WAVELENGTHS, n, q.s), q.s);
}
fn mkl_extended_sag_d(s: u32, xv: vec2<f32>, yv: vec2<f32>) -> MklSagD {
    let data = mkl_surface_data(s);
    let radius = load_d(data, 0u);
    let na = u32(mkl_pool[data + 2u * MKL_COEFF_NA]);
    let nb = u32(mkl_pool[data + 2u * MKL_COEFF_NB]);
    let coeff = data + 2u * MKL_COEFF_PREFIX;
    let x = mkl_jvar_d(xv, true);
    let y = mkl_jvar_d(yv, false);
    let r2 = mkl_jsum_d(mkl_jmul_d(x, x), mkl_jmul_d(y, y));
    let c = parts_d(mkl_surfaces[s].curvature, mkl_surfaces[s].curvature_lo);
    let k = parts_d(mkl_surfaces[s].conic, mkl_surfaces[s].conic_lo);
    let shape = mkl_surfaces[s].shape;
    if (shape == MKL_SHAPE_ODD_ASPHERE) {
        let base = mkl_conic_sag_jet_d(c, k, r2);
        if (!base.ok) {
            return base;
        }
        if (zero_d(x.v) && zero_d(y.v)) {
            return base;
        }
        var radial: MklJetD;
        if (zero_d(r2.v)) {
            let ax = select(x.v, neg_d(x.v), le_d(x.v, 0.0));
            let ay = select(y.v, neg_d(y.v), le_d(y.v, 0.0));
            let scale = select(ax, ay, mkl_le_value_d(ax, ay));
            let u = div_d(x.v, scale);
            let v = div_d(y.v, scale);
            let w = sqrt_d(add_d(mul_d(u, u), mul_d(v, v)));
            radial = MklJetD(mul_d(scale, w), div_d(u, w), div_d(v, w));
        } else {
            radial = mkl_jsqrt_d(r2);
        }
        let r = mkl_jscale_d(radial, div_d(c_d(1.0), radius));
        return MklSagD(true, mkl_jsum_d(base.j, mkl_jmul_d(mkl_poly_d(coeff, 0u, na, r), r)));
    }
    if (shape == MKL_SHAPE_TOROID || shape == MKL_SHAPE_BICONIC) {
        let cx = load_d(data, MKL_COEFF_CX);
        let inv = div_d(c_d(1.0), radius);
        let xx = mkl_jscale_d(x, inv);
        let yy = mkl_jscale_d(y, inv);
        if (shape == MKL_SHAPE_TOROID) {
            let base = mkl_conic_sag_jet_d(c, k, mkl_jmul_d(y, y));
            if (!base.ok) {
                return base;
            }
            let zy = mkl_jsum_d(base.j, mkl_even_d(coeff, na, nb, mkl_jmul_d(yy, yy)));
            let a = mkl_jadd_d(mkl_jscale_d(mkl_jneg_d(zy), cx), c_d(1.0));
            let q = mkl_jdiff_d(mkl_jmul_d(a, a), mkl_jscale_d(mkl_jmul_d(x, x), mul_d(cx, cx)));
            if (le_d(a.v, 0.0) || !mkl_le_value_d(c_d(0.0), q.v)) {
                return mkl_sag_fail_d();
            }
            return MklSagD(true, mkl_jsum_d(zy, mkl_jdiv_d(mkl_jscale_d(mkl_jmul_d(x, x), cx), mkl_jsum_d(a, mkl_jsqrt_d(q)))));
        }
        let kx = load_d(data, MKL_COEFF_KX);
        let arg = mkl_jadd_d(
            mkl_jsum_d(
                mkl_jscale_d(mkl_jmul_d(x, x), mul_d(mul_d(neg_d(add_d(c_d(1.0), kx)), cx), cx)),
                mkl_jscale_d(mkl_jmul_d(y, y), mul_d(mul_d(neg_d(add_d(c_d(1.0), k)), c), c)),
            ),
            c_d(1.0),
        );
        if (!mkl_le_value_d(c_d(0.0), arg.v)) {
            return mkl_sag_fail_d();
        }
        let num = mkl_jsum_d(mkl_jscale_d(mkl_jmul_d(x, x), cx), mkl_jscale_d(mkl_jmul_d(y, y), c));
        let z = mkl_jsum_d(
            mkl_jsum_d(mkl_jdiv_d(num, mkl_jadd_d(mkl_jsqrt_d(arg), c_d(1.0))), mkl_even_d(coeff, 0u, na, mkl_jmul_d(xx, xx))),
            mkl_even_d(coeff, na, nb, mkl_jmul_d(yy, yy)),
        );
        return MklSagD(true, z);
    }
    let q = mkl_scaled_jet_d(x, y, radius);
    if (!mkl_normalized_d(q)) {
        return mkl_sag_fail_d();
    }
    let base = mkl_conic_sag_jet_d(c, k, r2);
    if (!base.ok) {
        return base;
    }
    let u = q.u;
    let v = q.v;
    let t = q.s;
    if (shape == MKL_SHAPE_Q_CON) {
        return MklSagD(true, mkl_jsum_d(base.j, mkl_qcon_d(coeff, na, t)));
    }
    if (shape == MKL_SHAPE_Q_BFS) {
        if (na == 0u) {
            return base;
        }
        let w = mkl_jsqrt_d(mkl_jadd_d(mkl_jscale_d(r2, mul_d(neg_d(c), c)), c_d(1.0)));
        if (zero_d(w.v)) {
            return mkl_sag_fail_d();
        }
        let tt = mkl_jmul_d(mkl_jmul_d(t, mkl_jadd_d(mkl_jneg_d(t), c_d(1.0))), mkl_qbfs_d(coeff, na, t));
        return MklSagD(true, mkl_jsum_d(base.j, mkl_jdiv_d(tt, w)));
    }
    if (shape == MKL_SHAPE_XY) {
        var start = 0u;
        var degree = 1u;
        var z = base.j;
        while (start < na) {
            var row = mkl_jconst_d(c_d(0.0));
            var vn = mkl_jconst_d(c_d(1.0));
            for (var n = 0u; n <= degree; n++) {
                var a = c_d(0.0);
                if (start + n < na) {
                    a = load_d(coeff, start + n);
                }
                row = mkl_jsum_d(mkl_jmul_d(row, u), mkl_jscale_d(vn, a));
                vn = mkl_jmul_d(vn, v);
            }
            z = mkl_jsum_d(z, row);
            start += degree + 1u;
            degree++;
        }
        return MklSagD(true, z);
    }
    let standard = mkl_pool[data + 2u * MKL_COEFF_KIND] != 0.0;
    return MklSagD(true, mkl_jsum_d(base.j, mkl_zernike_d(coeff, na, standard, u, v, t)));
}
// The base of an overlaid surface may be legacy (or a DOE's legacy base).
fn mkl_legacy_sag_d(s: u32, xv: vec2<f32>, yv: vec2<f32>) -> MklSagD {
    let x = mkl_jvar_d(xv, true);
    let y = mkl_jvar_d(yv, false);
    let r2 = mkl_jsum_d(mkl_jmul_d(x, x), mkl_jmul_d(y, y));
    let sf = &mkl_surfaces[s];
    let base = mkl_conic_sag_jet_d(parts_d((*sf).curvature, (*sf).curvature_lo), parts_d((*sf).conic, (*sf).conic_lo), r2);
    if (!base.ok) {
        return base;
    }
    let t = mkl_jscale_d(r2, c_d((*sf).asphere_inv_norm2));
    var p = mkl_jconst_d(c_d(0.0));
    for (var j = i32((*sf).num_asphere) - 1; j >= 0; j--) {
        p = mkl_jadd_d(mkl_jmul_d(p, t), parts_d((*sf).asphere[j], (*sf).asphere_lo[j]));
    }
    return MklSagD(true, mkl_jsum_d(base.j, mkl_jmul_d(mkl_jmul_d(p, t), t)));
}
// Sag jet of an extended surface or, with MKL_OVERLAYS, of any surface with
// MKL_SURF_HAS_SAG_OVERLAY. Code without MKL_OVERLAYS never sees the overlay.
fn mkl_surface_eval_d(s: u32, xv: vec2<f32>, yv: vec2<f32>) -> MklSagD {
    if (!MKL_OVERLAYS || (mkl_surfaces[s].flags & MKL_SURF_HAS_SAG_OVERLAY) == 0u) {
        return mkl_extended_sag_d(s, xv, yv);
    }
    let data = mkl_surface_data(s);
    var shape = mkl_surfaces[s].shape;
    if (shape == MKL_SHAPE_DIFFRACTIVE) {
        shape = u32(mkl_pool[data + 2u * MKL_COEFF_KIND]);
    }
    var base: MklSagD;
    if (shape <= MKL_SHAPE_EVEN_ASPHERE) {
        base = mkl_legacy_sag_d(s, xv, yv);
    } else {
        base = mkl_extended_sag_d(s, xv, yv);
    }
    if (!base.ok) {
        return base;
    }
    let d = data + 2u * (mkl_surfaces[s].coefficient_count - MKL_OVERLAY_COUNT);
    let inv = load_d(d, MKL_OVERLAY_INV_RADIUS);
    let c = load_d(d, MKL_OVERLAY_COS);
    let sn = load_d(d, MKL_OVERLAY_SIN);
    let s4 = load_d(d, MKL_OVERLAY_S4);
    let two = c_d(2.0);
    let u = mul_d(xv, inv);
    let v = mul_d(yv, inv);
    let t = add_d(mul_d(u, u), mul_d(v, v));
    let q = add_d(load_d(d, MKL_OVERLAY_S2), mul_d(t, s4));
    let w = add_d(q, mul_d(t, s4));
    let o = MklJetD(
        add_d(add_d(mul_d(c, mul_d(sub_d(u, v), add_d(u, v))), mul_d(mul_d(two, sn), mul_d(u, v))), mul_d(t, q)),
        mul_d(mul_d(two, inv), add_d(mul_d(u, add_d(c, w)), mul_d(sn, v))),
        mul_d(mul_d(two, inv), add_d(mul_d(v, sub_d(w, c)), mul_d(sn, u))),
    );
    return MklSagD(true, mkl_jsum_d(base.j, o));
}

// ---------------------------------------------------------------------------
// trace_shared.h: local-frame trace helpers.
// ---------------------------------------------------------------------------

// A double-float 3-vector.
struct MklDv {
    x: vec2<f32>,
    y: vec2<f32>,
    z: vec2<f32>,
}

struct MklHit {
    code: u32,
    ta: vec2<f32>,
    tb: vec2<f32>,
    x: MklDv,
    g: MklDv,
    normal: vec3<f32>,
}

// A ray state (MklState and MklRawState): position, direction, the
// unnormalised normal of the last hit, OPL, medium index.
struct MklRaw {
    pos: MklDv,
    dir: MklDv,
    normal: MklDv,
    opl: vec2<f32>,
    index: vec2<f32>,
    weight: f32,
    transmittance: f32,
    status: u32,
    last: u32,
    done: u32,
}

fn mkl_hi3(p: MklDv) -> vec3<f32> {
    return vec3<f32>(p.x.x, p.y.x, p.z.x);
}
fn mkl_lo3(p: MklDv) -> vec3<f32> {
    return vec3<f32>(p.x.y, p.y.y, p.z.y);
}
fn mkl_finite3(v: vec3<f32>) -> bool {
    return mkl_is_finite(v.x) && mkl_is_finite(v.y) && mkl_is_finite(v.z);
}
fn mkl_norm3(v: vec3<f32>) -> f32 {
    return mkl_hypot3_rn(v.x, v.y, v.z);
}
fn mkl_normalized(v: vec3<f32>) -> vec3<f32> {
    let n = mkl_norm3(v);
    if (n == 0.0) {
        return v;
    }
    return vec3<f32>(mkl_div_rn(v.x, n), mkl_div_rn(v.y, n), mkl_div_rn(v.z, n));
}

fn mkl_ddot(a: MklDv, b: MklDv) -> vec2<f32> {
    var s = mkl_df_mul(a.z, b.z);
    s = mkl_df_add(s, mkl_df_mul(a.y, b.y));
    return mkl_df_add(s, mkl_df_mul(a.x, b.x));
}

fn mkl_renorm_along(d: MklDv, g: MklDv) -> MklDv {
    let m = mkl_df_add_f32(mkl_ddot(d, d), -1.0);
    let h = mkl_dot3(mkl_hi3(d), mkl_hi3(g));
    if (h == 0.0) {
        return d;
    }
    let b = mkl_div_rn(m.x, mkl_mul(2.0, h));
    return MklDv(
        mkl_df_sub(d.x, mkl_df_mul_f32(g.x, b)),
        mkl_df_sub(d.y, mkl_df_mul_f32(g.y, b)),
        mkl_df_sub(d.z, mkl_df_mul_f32(g.z, b)),
    );
}

fn mkl_renorm(d: MklDv) -> MklDv {
    let s = mkl_mul(0.5, mkl_df_add_f32(mkl_ddot(d, d), -1.0).x);
    return MklDv(
        mkl_df_sub(d.x, mkl_df_mul_f32(d.x, s)),
        mkl_df_sub(d.y, mkl_df_mul_f32(d.y, s)),
        mkl_df_sub(d.z, mkl_df_mul_f32(d.z, s)),
    );
}

fn mkl_rot_row(s: u32, row: u32, transpose: bool, v: MklDv) -> vec2<f32> {
    let sf = &mkl_surfaces[s];
    let a = select(3u * row, row, transpose);
    let b = select(3u * row + 1u, row + 3u, transpose);
    let c = select(3u * row + 2u, row + 6u, transpose);
    let lo = mkl_fma(
        (*sf).to_next_rot_lo[a],
        v.x.x,
        mkl_fma((*sf).to_next_rot_lo[b], v.y.x, mkl_mul((*sf).to_next_rot_lo[c], v.z.x)),
    );
    var acc = mkl_df_mul_f32(v.z, (*sf).to_next_rot[c]);
    acc = mkl_df_add(acc, mkl_df_mul_f32(v.y, (*sf).to_next_rot[b]));
    acc = mkl_df_add(acc, mkl_df_mul_f32(v.x, (*sf).to_next_rot[a]));
    return mkl_df_add_f32(acc, lo);
}

// Position and direction (pos, dir).
struct MklPd {
    p: MklDv,
    d: MklDv,
}

fn mkl_xf_forward(s: u32, p: MklDv, d: MklDv) -> MklPd {
    let sf = &mkl_surfaces[s];
    if (((*sf).flags & MKL_SURF_AXIAL) != 0u) {
        var q = p;
        q.z = mkl_df_sub(p.z, vec2<f32>((*sf).to_next_origin[2], (*sf).to_next_origin_lo[2]));
        return MklPd(q, d);
    }
    let px = mkl_df_sub(mkl_rot_row(s, 0u, false, p), vec2<f32>((*sf).to_next_shift[0], (*sf).to_next_shift_lo[0]));
    let dx = mkl_rot_row(s, 0u, false, d);
    let py = mkl_df_sub(mkl_rot_row(s, 1u, false, p), vec2<f32>((*sf).to_next_shift[1], (*sf).to_next_shift_lo[1]));
    let dy = mkl_rot_row(s, 1u, false, d);
    let pz = mkl_df_sub(mkl_rot_row(s, 2u, false, p), vec2<f32>((*sf).to_next_shift[2], (*sf).to_next_shift_lo[2]));
    let dz = mkl_rot_row(s, 2u, false, d);
    return MklPd(MklDv(px, py, pz), MklDv(dx, dy, dz));
}

fn mkl_xf_backward(s: u32, p: MklDv, d: MklDv) -> MklPd {
    let sf = &mkl_surfaces[s];
    if (((*sf).flags & MKL_SURF_AXIAL) != 0u) {
        var q = p;
        q.z = mkl_df_add(p.z, vec2<f32>((*sf).to_next_origin[2], (*sf).to_next_origin_lo[2]));
        return MklPd(q, d);
    }
    let px = mkl_df_add(mkl_rot_row(s, 0u, true, p), vec2<f32>((*sf).to_next_origin[0], (*sf).to_next_origin_lo[0]));
    let dx = mkl_rot_row(s, 0u, true, d);
    let py = mkl_df_add(mkl_rot_row(s, 1u, true, p), vec2<f32>((*sf).to_next_origin[1], (*sf).to_next_origin_lo[1]));
    let dy = mkl_rot_row(s, 1u, true, d);
    let pz = mkl_df_add(mkl_rot_row(s, 2u, true, p), vec2<f32>((*sf).to_next_origin[2], (*sf).to_next_origin_lo[2]));
    let dz = mkl_rot_row(s, 2u, true, d);
    return MklPd(MklDv(px, py, pz), MklDv(dx, dy, dz));
}

fn mkl_discriminant(a: f32, b: f32, c: f32) -> f32 {
    let w = mkl_mul(a, c);
    let e = mkl_fma(-a, c, w);
    let ff = mkl_fma(b, b, -w);
    return mkl_add(ff, e);
}

fn mkl_at(t: f32, d: vec3<f32>, q: vec3<f32>) -> vec3<f32> {
    return vec3<f32>(mkl_fma(t, d.x, q.x), mkl_fma(t, d.y, q.y), mkl_fma(t, d.z, q.z));
}

fn mkl_on_sheet(kap: f32, c: f32, pt: vec3<f32>) -> bool {
    return mkl_sub(1.0, mkl_mul(mkl_mul(kap, c), pt.z)) >= -mkl_sheet_tol();
}

// A conic root: whether there is one, its t and point.
struct MklRoot {
    ok: bool,
    t: f32,
    p: vec3<f32>,
}

// Conic steps 2-5 of mklight_trace::surface from q; ok false for a miss.
fn mkl_conic_from(c: f32, k: f32, q: vec3<f32>, d: vec3<f32>) -> MklRoot {
    let kap = mkl_add(1.0, k);
    let a = mkl_mul(c, mkl_add(mkl_fma(d.x, d.x, mkl_mul(d.y, d.y)), mkl_mul(mkl_mul(kap, d.z), d.z)));
    let b = mkl_sub(mkl_mul(c, mkl_add(mkl_fma(q.x, d.x, mkl_mul(q.y, d.y)), mkl_mul(mkl_mul(kap, q.z), d.z))), d.z);
    let cc = mkl_sub(
        mkl_mul(c, mkl_add(mkl_fma(q.x, q.x, mkl_mul(q.y, q.y)), mkl_mul(mkl_mul(kap, q.z), q.z))),
        mkl_mul(2.0, q.z),
    );
    let disc = mkl_discriminant(a, b, cc);
    if (mkl_is_nan(disc) || disc < 0.0) {
        return MklRoot(false, 0.0, vec3<f32>(0.0));
    }
    let sq = mkl_sqrt_rn(disc);
    let qq = -mkl_add(b, select(-sq, sq, b >= 0.0));
    var have = false;
    var bt = 0.0;
    var bp = vec3<f32>(0.0);
    if (qq != 0.0) {
        let t = mkl_div_rn(cc, qq);
        let pt = mkl_at(t, d, q);
        if (mkl_is_finite(t) && mkl_finite3(pt) && mkl_on_sheet(kap, c, pt)) {
            have = true;
            bt = t;
            bp = pt;
        }
    } else if (cc == 0.0) {
        have = true;
        bt = 0.0;
        bp = q;
    }
    if (a != 0.0 && qq != 0.0) {
        let t = mkl_div_rn(qq, a);
        let pt = mkl_at(t, d, q);
        if (mkl_is_finite(t) && mkl_finite3(pt) && mkl_on_sheet(kap, c, pt)) {
            let r2 = mkl_fma(pt.x, pt.x, mkl_mul(pt.y, pt.y));
            let keep = have && mkl_fma(bp.x, bp.x, mkl_mul(bp.y, bp.y)) <= r2;
            if (!keep) {
                have = true;
                bt = t;
                bp = pt;
            }
        }
    }
    return MklRoot(have, bt, bp);
}

// (P(r^2), dP/d(r^2)) of the packed even polynomial.
fn mkl_asphere_poly(s: u32, r2: f32) -> vec2<f32> {
    let sf = &mkl_surfaces[s];
    let n = min((*sf).num_asphere, MKL_MAX_ASPHERE);
    if (n == 0u) {
        return vec2<f32>(0.0);
    }
    let inv = (*sf).asphere_inv_norm2;
    let rho2 = mkl_mul(r2, inv);
    var p = (*sf).asphere[n - 1u];
    var dp = mkl_mul((*sf).asphere[n - 1u], f32(n + 1u));
    for (var j = i32(n) - 2; j >= 0; j--) {
        p = mkl_fma(p, rho2, (*sf).asphere[j]);
        dp = mkl_fma(dp, rho2, mkl_mul((*sf).asphere[j], f32(j + 2)));
    }
    let rho4 = mkl_mul(rho2, rho2);
    return vec2<f32>(mkl_mul(p, rho4), mkl_mul(mkl_mul(dp, rho2), inv));
}

// (P(r^2), dP/d(r^2)) in double-float from the packed hi + lo coefficients
// (trace_kernel::asphere_poly_dp_df), as (P.hi, P.lo, dP.hi, dP.lo).
fn mkl_asphere_poly_dp_df(s: u32, r2: vec2<f32>) -> vec4<f32> {
    let sf = &mkl_surfaces[s];
    let n = min((*sf).num_asphere, MKL_MAX_ASPHERE);
    if (n == 0u) {
        return vec4<f32>(0.0);
    }
    let rho2 = mkl_df_mul_f32(r2, (*sf).asphere_inv_norm2);
    let an = vec2<f32>((*sf).asphere[n - 1u], (*sf).asphere_lo[n - 1u]);
    var p = an;
    var dp = mkl_df_mul_f32(an, f32(n + 1u));
    for (var j = i32(n) - 2; j >= 0; j--) {
        let a = vec2<f32>((*sf).asphere[j], (*sf).asphere_lo[j]);
        p = mkl_df_add(mkl_df_mul(p, rho2), a);
        dp = mkl_df_add(mkl_df_mul(dp, rho2), mkl_df_mul_f32(a, f32(j + 2)));
    }
    let po = mkl_df_mul(p, mkl_df_mul(rho2, rho2));
    let dpo = mkl_df_mul_f32(mkl_df_mul(dp, rho2), (*sf).asphere_inv_norm2);
    return vec4<f32>(po, dpo);
}

fn mkl_asphere_domain(s: u32, p: vec3<f32>) -> bool {
    let c = mkl_surfaces[s].curvature;
    let kap = mkl_add(1.0, mkl_surfaces[s].conic);
    let r2 = mkl_fma(p.x, p.x, mkl_mul(p.y, p.y));
    let arg = mkl_fma(mkl_mul(mkl_mul(-kap, c), c), r2, 1.0);
    return !(mkl_is_nan(arg) || arg < 0.0);
}

// A Newton root: MKL_RAY_* code and t.
struct MklNewton {
    code: u32,
    t: f32,
}

// Newton on the sag residual from q; MKL_RAY_OK and t on success, else the
// MKL_RAY_* failure code.
fn mkl_asphere_from(s: u32, q: vec3<f32>, d: vec3<f32>) -> MklNewton {
    let c = mkl_surfaces[s].curvature;
    let k = mkl_surfaces[s].conic;
    let seed = mkl_conic_from(c, k, q, d);
    var t = select(0.0, seed.t, seed.ok);
    let kap = mkl_add(1.0, k);
    var g_first = 0.0;
    var sign_changed = false;
    var prev_abs_g = mkl_inf();
    for (var it = 0u; ; it++) {
        let x = mkl_fma(t, d.x, q.x);
        let y = mkl_fma(t, d.y, q.y);
        let z = mkl_fma(t, d.z, q.z);
        let r2 = mkl_fma(x, x, mkl_mul(y, y));
        let arg = mkl_fma(mkl_mul(mkl_mul(-kap, c), c), r2, 1.0);
        if (mkl_is_nan(arg) || arg < 0.0) {
            return MklNewton(select(MKL_RAY_OUTSIDE_DOMAIN, MKL_RAY_MISS, !seed.ok && !sign_changed), t);
        }
        let w = mkl_sqrt_rn(arg);
        let pp = mkl_asphere_poly(s, r2);
        let base = mkl_div_rn(mkl_mul(c, r2), mkl_add(1.0, w));
        let g = mkl_sub(z, mkl_add(base, pp.x));
        var breakdown = select(MKL_RAY_MISS, MKL_RAY_NOT_CONVERGED, sign_changed);
        if (!mkl_is_finite(g)) {
            return MklNewton(breakdown, t);
        }
        if (it == 0u) {
            g_first = g;
        } else if (g != 0.0 && (g > 0.0) != (g_first > 0.0)) {
            sign_changed = true;
            breakdown = MKL_RAY_NOT_CONVERGED;
        }
        let slope_w = mkl_fma(mkl_mul(2.0, w), pp.y, c);
        let gp = mkl_sub(d.z, mkl_mul(mkl_div_rn(slope_w, max(w, mkl_flt_min())), mkl_fma(x, d.x, mkl_mul(y, d.y))));
        let gp_ok = mkl_is_finite(gp) && gp != 0.0;
        let tol = mkl_add(
            prm.trace.tolerance_abs,
            mkl_mul(prm.trace.tolerance_rel, mkl_add(mkl_add(mkl_add(abs(t), abs(z)), abs(base)), abs(pp.x))),
        );
        if (abs(g) <= tol) {
            if (gp_ok) {
                t = mkl_sub(t, mkl_div_rn(g, gp));
            }
            let code = select(MKL_RAY_OUTSIDE_DOMAIN, MKL_RAY_OK, mkl_asphere_domain(s, mkl_at(t, d, q)));
            return MklNewton(code, t);
        }
        if (it == prm.trace.max_iterations) {
            let converging = abs(g) <= mkl_mul(0.5, prev_abs_g);
            let cap_fails = (prm.trace.flags & MKL_TRACE_NEWTON_CAP_NOT_CONVERGED) != 0u;
            return MklNewton(select(breakdown, MKL_RAY_NOT_CONVERGED, converging || cap_fails), t);
        }
        if (!gp_ok) {
            return MklNewton(breakdown, t);
        }
        t = mkl_sub(t, mkl_div_rn(g, gp));
        if (!mkl_is_finite(t)) {
            return MklNewton(breakdown, t);
        }
        prev_abs_g = abs(g);
    }
    return MklNewton(MKL_RAY_NOT_CONVERGED, t);
}

fn mkl_point_on(q: MklDv, t: vec2<f32>, d: MklDv) -> MklDv {
    return MklDv(
        mkl_df_add(q.x, mkl_df_mul(d.x, t)),
        mkl_df_add(q.y, mkl_df_mul(d.y, t)),
        mkl_df_add(q.z, mkl_df_mul(d.z, t)),
    );
}

fn mkl_refine_t(t1: f32, r: f32, rp: f32) -> vec2<f32> {
    let stp = mkl_div_rn(r, rp);
    if (rp != 0.0 && mkl_is_finite(stp)) {
        return mkl_two_sum(t1, -stp);
    }
    return vec2<f32>(t1, 0.0);
}

fn mkl_zero_dv() -> MklDv {
    return MklDv(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(0.0));
}

fn mkl_miss(code: u32) -> MklHit {
    var h: MklHit;
    h.code = code;
    return h;
}

// mklight_trace::surface::intersect from the double-float point p along d.
fn mkl_intersect_legacy(s: u32, shape: u32, p: MklDv, d: MklDv) -> MklHit {
    let zg = MklDv(vec2<f32>(0.0), vec2<f32>(0.0), vec2<f32>(1.0, 0.0));
    let zn = vec3<f32>(0.0, 0.0, 1.0);
    var hit: MklHit;
    hit.code = MKL_RAY_OK;
    if (shape == MKL_SHAPE_COORD_BREAK) {
        hit.ta = vec2<f32>(0.0);
        hit.tb = vec2<f32>(0.0);
        hit.x = p;
        hit.g = zg;
        hit.normal = zn;
        return hit;
    }
    let sf = &mkl_surfaces[s];
    let conic_like = (shape == MKL_SHAPE_CONIC && (*sf).curvature != 0.0)
        || (shape == MKL_SHAPE_EVEN_ASPHERE && ((*sf).num_asphere > 0u || (*sf).curvature != 0.0));
    if (!conic_like) {
        if (d.z.x == 0.0) {
            return mkl_miss(MKL_RAY_MISS);
        }
        let t = mkl_df_div_rn(mkl_df_neg(p.z), d.z);
        let x = MklDv(mkl_df_add(p.x, mkl_df_mul(d.x, t)), mkl_df_add(p.y, mkl_df_mul(d.y, t)), vec2<f32>(0.0));
        if (!(mkl_is_finite(x.x.x) && mkl_is_finite(x.y.x))) {
            return mkl_miss(MKL_RAY_MISS);
        }
        hit.ta = t;
        hit.tb = vec2<f32>(0.0);
        hit.x = x;
        hit.g = zg;
        hit.normal = zn;
        return hit;
    }
    let t0 = mkl_df_neg(mkl_ddot(p, d));
    let q = mkl_point_on(p, t0, d);
    let qf = mkl_hi3(q);
    let dfv = mkl_hi3(d);
    hit.ta = t0;
    let c = (*sf).curvature;
    let k = (*sf).conic;
    let kap = mkl_add(1.0, k);
    let cd = vec2<f32>((*sf).curvature, (*sf).curvature_lo);
    let kapd = mkl_df_add_f32(vec2<f32>((*sf).conic, (*sf).conic_lo), 1.0);
    let kcd = mkl_df_mul(kapd, cd);
    if (shape == MKL_SHAPE_EVEN_ASPHERE && (*sf).num_asphere > 0u) {
        // trace_kernel::asphere_refine_df: two double-float Newton steps and
        // the normal from the double-float slope.
        let r = mkl_asphere_from(s, qf, dfv);
        if (r.code != MKL_RAY_OK) {
            return mkl_miss(r.code);
        }
        var tb = vec2<f32>(r.t, 0.0);
        var w = vec2<f32>(1.0, 0.0);
        var gs = cd;
        for (var it = 0; it < 2; it++) {
            let xi = mkl_point_on(q, tb, d);
            let r2 = mkl_df_add(mkl_df_mul(xi.x, xi.x), mkl_df_mul(xi.y, xi.y));
            w = mkl_df_sqrt_rn(mkl_df_sub(vec2<f32>(1.0, 0.0), mkl_df_mul(r2, mkl_df_mul(kcd, cd))));
            let csag = mkl_df_div_rn(mkl_df_mul(r2, cd), mkl_df_add_f32(w, 1.0));
            let pd = mkl_asphere_poly_dp_df(s, r2);
            let res = mkl_df_sub(mkl_df_sub(xi.z, csag), pd.xy);
            gs = mkl_df_add(mkl_df_mul_f32(mkl_df_mul(w, pd.zw), 2.0), cd);
            let gp = mkl_df_sub(
                d.z,
                mkl_df_mul(mkl_df_div_rn(gs, w), mkl_df_add(mkl_df_mul(xi.x, d.x), mkl_df_mul(xi.y, d.y))),
            );
            let stp = mkl_df_div_rn(res, gp);
            if (mkl_is_finite(stp.x)) {
                tb = mkl_df_sub(tb, stp);
            }
        }
        let x = mkl_point_on(q, tb, d);
        hit.tb = tb;
        hit.x = x;
        hit.g = MklDv(mkl_df_neg(mkl_df_mul(x.x, gs)), mkl_df_neg(mkl_df_mul(x.y, gs)), w);
    } else {
        let root = mkl_conic_from(c, k, qf, dfv);
        if (!root.ok) {
            return mkl_miss(MKL_RAY_MISS);
        }
        let t1 = root.t;
        let x0 = mkl_point_on(q, vec2<f32>(t1, 0.0), d);
        let r2 = mkl_df_add(mkl_df_mul(x0.x, x0.x), mkl_df_mul(x0.y, x0.y));
        let fv = mkl_df_sub(
            mkl_df_mul(mkl_df_add(r2, mkl_df_mul(mkl_df_mul(x0.z, x0.z), kapd)), cd),
            mkl_df_mul_f32(x0.z, 2.0),
        );
        let xh = mkl_hi3(x0);
        let fp = mkl_mul(
            2.0,
            mkl_sub(mkl_mul(c, mkl_add(mkl_fma(xh.x, dfv.x, mkl_mul(xh.y, dfv.y)), mkl_mul(mkl_mul(kap, xh.z), dfv.z))), dfv.z),
        );
        let tb = mkl_refine_t(t1, fv.x, fp);
        let x = mkl_point_on(q, tb, d);
        hit.tb = tb;
        hit.x = x;
        hit.g = MklDv(
            mkl_df_neg(mkl_df_mul(x.x, cd)),
            mkl_df_neg(mkl_df_mul(x.y, cd)),
            mkl_df_sub(vec2<f32>(1.0, 0.0), mkl_df_mul(x.z, kcd)),
        );
    }
    hit.normal = mkl_normalized(mkl_hi3(hit.g));
    return hit;
}

// Safeguarded Newton on z - sag from the conic seed (trace_kernel::sag_from):
// the reference's bracket, backtracking and Miss / NoConvergence /
// OutsideDomain decisions in f32.
fn mkl_sag_from(s: u32, q: vec3<f32>, d: vec3<f32>) -> MklNewton {
    let seed = mkl_conic_from(mkl_surfaces[s].curvature, mkl_surfaces[s].conic, q, d);
    var t = select(0.0, seed.t, seed.ok);
    var have_pos = false;
    var have_neg = false;
    var positive = 0.0;
    var negative = 0.0;
    var previous = mkl_inf();
    let max_it = prm.trace.max_iterations;
    for (var it = 0u; it <= max_it; it++) {
        let bracketed = have_pos && have_neg;
        let left_domain = select(MKL_RAY_OUTSIDE_DOMAIN, MKL_RAY_MISS, !seed.ok && !bracketed);
        let pt = mkl_at(t, d, q);
        let v = mkl_surface_eval_f(s, pt.x, pt.y);
        if (!v.ok) {
            return MklNewton(left_domain, t);
        }
        let g = mkl_sub(pt.z, v.j.v);
        let gp = mkl_sub(mkl_sub(d.z, mkl_mul(v.j.x, d.x)), mkl_mul(v.j.y, d.y));
        if (g > 0.0) {
            positive = t;
            have_pos = true;
        } else if (g < 0.0) {
            negative = t;
            have_neg = true;
        }
        let bracket = have_pos && have_neg;
        let lo = min(positive, negative);
        let hi = max(positive, negative);
        let failure = select(MKL_RAY_MISS, MKL_RAY_NOT_CONVERGED, bracket);
        let tol = mkl_add(
            prm.trace.tolerance_abs,
            mkl_mul(prm.trace.tolerance_rel, mkl_add(mkl_add(abs(t), abs(pt.z)), abs(v.j.v))),
        );
        if (abs(g) <= tol) {
            let corrected = mkl_at(mkl_sub(t, mkl_div_rn(g, gp)), d, q);
            let vc = mkl_surface_eval_f(s, corrected.x, corrected.y);
            if (vc.ok) {
                t = mkl_sub(t, mkl_div_rn(g, gp));
            }
            return MklNewton(MKL_RAY_OK, t);
        }
        if (it == max_it) {
            let cap = (prm.trace.flags & MKL_TRACE_NEWTON_CAP_NOT_CONVERGED) != 0u;
            return MklNewton(select(failure, MKL_RAY_NOT_CONVERGED, abs(g) <= mkl_mul(0.5, previous) || cap), t);
        }
        if (!mkl_is_finite(g) || gp == 0.0 || !mkl_is_finite(gp)) {
            return MklNewton(failure, t);
        }
        var next = mkl_sub(t, mkl_div_rn(g, gp));
        if (bracket && (mkl_is_nan(next) || !(next > lo && next < hi))) {
            next = mkl_add(lo, mkl_mul(0.5, mkl_sub(hi, lo)));
        }
        var outside = false;
        var accepted = false;
        for (var j = 0u; j < 20u; j++) {
            let p = mkl_at(next, d, q);
            let vn = mkl_surface_eval_f(s, p.x, p.y);
            if (vn.ok) {
                if (bracket || abs(mkl_sub(p.z, vn.j.v)) < abs(g)) {
                    accepted = true;
                    break;
                }
            } else {
                outside = true;
            }
            next = mkl_add(t, mkl_mul(0.5, mkl_sub(next, t)));
        }
        if (!accepted) {
            return MklNewton(select(failure, left_domain, outside && !bracket), t);
        }
        previous = abs(g);
        t = next;
    }
    return MklNewton(MKL_RAY_NOT_CONVERGED, t);
}

fn mkl_intersect(s: u32, p: MklDv, d: MklDv) -> MklHit {
    let shape = mkl_surfaces[s].shape;
    let overlay = MKL_OVERLAYS && (mkl_surfaces[s].flags & MKL_SURF_HAS_SAG_OVERLAY) != 0u;
    if (shape <= MKL_SHAPE_COORD_BREAK && !overlay) {
        return mkl_intersect_legacy(s, shape, p, d);
    }
    let data = mkl_surface_data(s);
    if (shape == MKL_SHAPE_DIFFRACTIVE && !overlay) {
        var h = mkl_intersect_legacy(s, u32(mkl_pool[data + 2u * MKL_COEFF_KIND]), p, d);
        if (h.code != MKL_RAY_OK) {
            return h;
        }
        if (!mkl_surface_domain_d(s, h.x.x, h.x.y)) {
            h.code = MKL_RAY_OUTSIDE_DOMAIN;
        }
        return h;
    }
    if (!overlay && mkl_pool[data + 2u * MKL_COEFF_EXACT_CONIC] != 0.0) {
        var h = mkl_intersect_legacy(s, MKL_SHAPE_CONIC, p, d);
        if (h.code != MKL_RAY_OK) {
            return h;
        }
        if (!mkl_surface_eval_d(s, h.x.x, h.x.y).ok) {
            h.code = MKL_RAY_OUTSIDE_DOMAIN;
        }
        return h;
    }
    let ta = mkl_df_neg(mkl_ddot(p, d));
    let q = mkl_point_on(p, ta, d);
    let r = mkl_sag_from(s, mkl_hi3(q), mkl_hi3(d));
    if (r.code != MKL_RAY_OK) {
        return mkl_miss(r.code);
    }
    var tb = vec2<f32>(r.t, 0.0);
    for (var i = 0u; i < 2u; i++) {
        let x = mkl_point_on(q, tb, d);
        let v = mkl_surface_eval_d(s, x.x, x.y);
        if (!v.ok) {
            return mkl_miss(MKL_RAY_OUTSIDE_DOMAIN);
        }
        let gp = mkl_df_sub(mkl_df_sub(d.z, mkl_df_mul(v.j.x, d.x)), mkl_df_mul(v.j.y, d.y));
        let stp = mkl_df_div_rn(mkl_df_sub(x.z, v.j.v), gp);
        if (mkl_is_finite(stp.x)) {
            tb = mkl_df_sub(tb, stp);
        }
    }
    var hit: MklHit;
    hit.x = mkl_point_on(q, tb, d);
    let v = mkl_surface_eval_d(s, hit.x.x, hit.x.y);
    if (!v.ok) {
        return mkl_miss(MKL_RAY_OUTSIDE_DOMAIN);
    }
    if (overlay && shape == MKL_SHAPE_DIFFRACTIVE && !mkl_surface_domain_d(s, hit.x.x, hit.x.y)) {
        return mkl_miss(MKL_RAY_OUTSIDE_DOMAIN);
    }
    hit.g = MklDv(mkl_df_neg(v.j.x), mkl_df_neg(v.j.y), vec2<f32>(1.0, 0.0));
    if (!mkl_finite3(mkl_hi3(hit.g))) {
        return mkl_miss(MKL_RAY_OUTSIDE_DOMAIN);
    }
    hit.code = MKL_RAY_OK;
    hit.ta = ta;
    hit.tb = tb;
    hit.normal = mkl_normalized(mkl_hi3(hit.g));
    return hit;
}

// A new direction, or ok false (evanescent order, total internal reflection
// or a degenerate lens).
struct MklTurn {
    ok: bool,
    d: MklDv,
}

// mklight_trace::interact::diffract in double-float (trace_kernel::diffract):
// the order's momentum grad projected on the tangent plane of g, then the
// normal component for the hemisphere; ok false for an evanescent order.
fn mkl_diffract(k: MklDv, g: MklDv, n1: vec2<f32>, n2: vec2<f32>, gx: vec2<f32>, gy: vec2<f32>, reflected: bool) -> MklTurn {
    let len = mkl_df_sqrt_rn(mkl_ddot(g, g));
    let normal = MklDv(mkl_df_div_rn(g.x, len), mkl_df_div_rn(g.y, len), mkl_df_div_rn(g.z, len));
    let kn = mkl_ddot(k, normal);
    let grad = MklDv(gx, gy, vec2<f32>(0.0));
    let gn = mkl_ddot(grad, normal);
    let inv = mkl_df_div_rn(vec2<f32>(1.0, 0.0), n2);
    let tangent = MklDv(
        mkl_df_mul(mkl_df_sub(mkl_df_add(mkl_df_mul(mkl_df_sub(k.x, mkl_df_mul(normal.x, kn)), n1), grad.x), mkl_df_mul(normal.x, gn)), inv),
        mkl_df_mul(mkl_df_sub(mkl_df_add(mkl_df_mul(mkl_df_sub(k.y, mkl_df_mul(normal.y, kn)), n1), grad.y), mkl_df_mul(normal.y, gn)), inv),
        mkl_df_mul(mkl_df_sub(mkl_df_add(mkl_df_mul(mkl_df_sub(k.z, mkl_df_mul(normal.z, kn)), n1), grad.z), mkl_df_mul(normal.z, gn)), inv),
    );
    let disc = mkl_df_sub(vec2<f32>(1.0, 0.0), mkl_ddot(tangent, tangent));
    if (disc.x < 0.0) {
        return MklTurn(false, k);
    }
    let sgn = select(1.0, -1.0, (kn.x < 0.0) != reflected);
    let root = mkl_df_mul_f32(mkl_df_sqrt_rn(disc), sgn);
    return MklTurn(
        true,
        MklDv(
            mkl_df_add(tangent.x, mkl_df_mul(normal.x, root)),
            mkl_df_add(tangent.y, mkl_df_mul(normal.y, root)),
            mkl_df_add(tangent.z, mkl_df_mul(normal.z, root)),
        ),
    );
}

// Stable vector Snell in f32: ok false on TIR; cos i and the signed
// coefficient gamma of `normal` in k' = mu k + gamma normal.
struct MklSnell {
    ok: bool,
    cos_i: f32,
    gamma: f32,
}

fn mkl_refract(k: vec3<f32>, normal: vec3<f32>, n1: f32, n2: f32) -> MklSnell {
    var cos_i = mkl_dot3(k, normal);
    let flip = cos_i < 0.0;
    if (flip) {
        cos_i = -cos_i;
    }
    cos_i = mkl_min_num(cos_i, 1.0);
    let mu = mkl_div_rn(n1, n2);
    let sin2 = mkl_mul(mkl_sub(1.0, cos_i), mkl_add(1.0, cos_i));
    let disc = mkl_fma(mkl_mul(-mu, mu), sin2, 1.0);
    if (disc < 0.0) {
        return MklSnell(false, cos_i, 0.0);
    }
    let root = mkl_sqrt_rn(disc);
    let denom = mkl_fma(mu, cos_i, root);
    var gamma: f32;
    if (denom > 0.0) {
        gamma = mkl_div_rn(mkl_div_rn(mkl_mul(mkl_sub(n2, n1), mkl_add(n2, n1)), mkl_mul(n2, n2)), denom);
    } else {
        gamma = mkl_sub(root, mkl_mul(mu, cos_i));
    }
    return MklSnell(true, cos_i, select(gamma, -gamma, flip));
}

// The incidence cosine of mkl_reflect (its direction is not needed).
fn mkl_reflect_cos(k: vec3<f32>, n: vec3<f32>) -> f32 {
    let kn = mkl_dot3(k, n);
    return mkl_min_num(abs(kn), 1.0);
}

fn mkl_dnorm(v: MklDv) -> vec2<f32> {
    return mkl_df_sqrt_rn(mkl_ddot(v, v));
}

fn mkl_dabs(x: vec2<f32>) -> vec2<f32> {
    if (x.x < 0.0) {
        return mkl_df_neg(x);
    }
    return x;
}

// sphere_term in double-float for X = (x0, x1, 0).
fn mkl_sphere_term(x0: vec2<f32>, x1: vec2<f32>, t: MklDv, rho: vec2<f32>) -> vec2<f32> {
    let tn = mkl_dnorm(t);
    let xx = mkl_df_add(mkl_df_mul(x1, x1), mkl_df_mul(x0, x0));
    let tx = mkl_df_add(mkl_df_mul(t.y, x1), mkl_df_mul(t.x, x0));
    if (mkl_mul(abs(rho.x), mkl_sqrt_rn(xx.x)) <= tn.x) {
        let u = MklDv(mkl_df_sub(t.x, mkl_df_mul(x0, rho)), mkl_df_sub(t.y, mkl_df_mul(x1, rho)), t.z);
        let den = mkl_df_add(mkl_dnorm(u), tn);
        return mkl_df_div_rn(mkl_df_sub(mkl_df_mul(rho, xx), mkl_df_mul_f32(tx, 2.0)), den);
    }
    let a = mkl_df_div_rn(vec2<f32>(1.0, 0.0), rho);
    let o = MklDv(
        mkl_df_sub(x0, mkl_df_mul(t.x, a)),
        mkl_df_sub(x1, mkl_df_mul(t.y, a)),
        mkl_df_neg(mkl_df_mul(t.z, a)),
    );
    let num = mkl_df_sub(mkl_df_mul_f32(mkl_df_add(mkl_df_mul(o.y, x1), mkl_df_mul(o.x, x0)), 2.0), xx);
    let q = mkl_df_div_rn(num, mkl_df_add(mkl_dnorm(o), mkl_df_mul(mkl_dabs(a), tn)));
    if (a.x < 0.0) {
        return mkl_df_neg(q);
    }
    return q;
}

// The ideal lens in double-float: direction and phase delta.
struct MklLens {
    ok: bool,
    d: MklDv,
    delta: vec2<f32>,
}

fn mkl_ideal_lens(px: vec2<f32>, py: vec2<f32>, k: MklDv, phi: vec2<f32>, n1: vec2<f32>, n2: vec2<f32>, vergence: vec2<f32>) -> MklLens {
    if (k.z.x == 0.0) {
        return MklLens(false, k, vec2<f32>(0.0));
    }
    // copysign(1, k.z): the sign bit of k.z on 1.
    let sz = vec2<f32>(bitcast<f32>(0x3F800000u | (bitcast<u32>(k.z.x) & 0x80000000u)), 0.0);
    let az = mkl_dabs(k.z);
    let r = mkl_df_div_rn(n1, n2);
    let t = MklDv(mkl_df_div_rn(k.x, az), mkl_df_div_rn(k.y, az), sz);
    let tx2 = mkl_df_sub(mkl_df_mul(r, t.x), mkl_df_mul(px, phi));
    let ty2 = mkl_df_sub(mkl_df_mul(r, t.y), mkl_df_mul(py, phi));
    let t_out = MklDv(tx2, ty2, sz);
    let inv = mkl_df_div_rn(vec2<f32>(1.0, 0.0), mkl_dnorm(t_out));
    let d = MklDv(mkl_df_mul(tx2, inv), mkl_df_mul(ty2, inv), mkl_df_mul(sz, inv));
    let rho_b = mkl_df_sub(phi, mkl_df_mul(r, vergence));
    let delta = mkl_df_sub(
        mkl_df_mul(n1, mkl_sphere_term(px, py, t, vergence)),
        mkl_df_mul(n2, mkl_sphere_term(px, py, t_out, mkl_df_neg(rho_b))),
    );
    return MklLens(true, d, delta);
}

// Unpolarised (R, T) of a bare interface.
fn mkl_fresnel(n1: f32, n2: f32, cos_i: f32) -> vec2<f32> {
    let ci = mkl_min_num(mkl_max_num(cos_i, 0.0), 1.0);
    let sin2_i = mkl_mul(mkl_sub(1.0, ci), mkl_add(1.0, ci));
    let rr = mkl_div_rn(n1, n2);
    let sin2_t = mkl_mul(mkl_mul(rr, rr), sin2_i);
    if (sin2_t >= 1.0) {
        return vec2<f32>(1.0, 0.0);
    }
    let ct = mkl_sqrt_rn(mkl_max_num(mkl_sub(1.0, sin2_t), 0.0));
    let a = mkl_mul(n1, ci);
    let b = mkl_mul(n2, ct);
    var rs = 1.0;
    if (mkl_add(a, b) != 0.0) {
        rs = mkl_div_rn(mkl_sub(a, b), mkl_add(a, b));
    }
    let c = mkl_mul(n2, ci);
    let dd = mkl_mul(n1, ct);
    var rp = 1.0;
    if (mkl_add(c, dd) != 0.0) {
        rp = mkl_div_rn(mkl_sub(c, dd), mkl_add(c, dd));
    }
    let rs2 = mkl_mul(rs, rs);
    let rp2 = mkl_mul(rp, rp);
    return vec2<f32>(mkl_mul(0.5, mkl_add(rs2, rp2)), mkl_mul(0.5, mkl_add(mkl_sub(1.0, rs2), mkl_sub(1.0, rp2))));
}

// Unpolarised (R, T) at a surface for incidence cosine cos_i
// (trace_kernel::trace_ray `rt`; never a transmission at a mirror).
fn mkl_surface_rt(s: u32, mirror: bool, n1: f32, n2: f32, cos_i: f32) -> vec2<f32> {
    if (mkl_surfaces[s].coating == MKL_COATING_IDEAL) {
        return vec2<f32>(mkl_surfaces[s].coating_r, mkl_surfaces[s].coating_t);
    }
    if (mirror) {
        return vec2<f32>(1.0, 0.0);
    }
    return mkl_fresnel(n1, n2, cos_i);
}

// Aperture test in double-float with the packed sizes' residuals.
fn mkl_passes_aperture(s: u32, x: vec2<f32>, y: vec2<f32>) -> bool {
    let sf = &mkl_surfaces[s];
    let a0 = vec2<f32>((*sf).aperture_params[0], (*sf).aperture_params_lo[0]);
    let a1 = vec2<f32>((*sf).aperture_params[1], (*sf).aperture_params_lo[1]);
    switch ((*sf).aperture) {
        case MKL_APERTURE_CIRCULAR: {
            let r2 = mkl_df_add(mkl_df_mul(x, x), mkl_df_mul(y, y));
            return mkl_df_sub(r2, mkl_df_mul(a0, a0)).x <= 0.0;
        }
        case MKL_APERTURE_ANNULAR: {
            let r2 = mkl_df_add(mkl_df_mul(x, x), mkl_df_mul(y, y));
            return mkl_df_sub(r2, mkl_df_mul(a0, a0)).x >= 0.0 && mkl_df_sub(r2, mkl_df_mul(a1, a1)).x <= 0.0;
        }
        case MKL_APERTURE_RECTANGULAR: {
            return mkl_df_sub(mkl_dabs(x), a0).x <= 0.0 && mkl_df_sub(mkl_dabs(y), a1).x <= 0.0;
        }
        case MKL_APERTURE_ELLIPTICAL: {
            let u = mkl_df_div_rn(x, a0);
            let v = mkl_df_div_rn(y, a1);
            return mkl_df_add_f32(mkl_df_add(mkl_df_mul(u, u), mkl_df_mul(v, v)), -1.0).x <= 0.0;
        }
        default: {
            return true;
        }
    }
}

// The reported OPL: the point-rounding correction of mkl_report.
fn mkl_report_opl(p: MklDv, d: MklDv, opl: vec2<f32>, n: vec2<f32>) -> vec2<f32> {
    let corr = mkl_mul(n.x, mkl_dot3(mkl_lo3(p), mkl_hi3(d)));
    return mkl_df_add_f32(opl, -corr);
}

fn mkl_index(s: u32, w: u32) -> vec2<f32> {
    return vec2<f32>(mkl_surfaces[s].n_after[w], mkl_surfaces[s].n_after_lo[w]);
}

fn mkl_plane(k: u32, i: u32) -> u32 {
    return k * g_stride + i;
}
fn mkl_in_f(k: u32) -> f32 {
    return bitcast<f32>(mkl_in[mkl_plane(k, g_ray)]);
}
fn mkl_put_f(k: u32, v: f32) {
    mkl_out[mkl_plane(k, g_ray)] = bitcast<u32>(v);
}

// Record modes of mkl_trace_core.
const MKL_REC_NONE: u32 = 0u;
// mkl_trace's records: the rounded state (mkl_report) and status.
const MKL_REC_TRACE: u32 = 1u;
// mkl_trace_resident's paired hi / lo records (MklAimRecords).
const MKL_REC_RESIDENT: u32 = 2u;

fn mkl_rec(mode: u32, slot: u32, st: MklRaw) {
    if (mode == MKL_REC_TRACE) {
        let o = (slot * g_stride + g_ray) * MKL_TRACE_RECORD_FLOATS;
        let opl = mkl_report_opl(st.pos, st.dir, st.opl, st.index);
        mkl_records[o] = bitcast<u32>(st.pos.x.x);
        mkl_records[o + 1u] = bitcast<u32>(st.pos.y.x);
        mkl_records[o + 2u] = bitcast<u32>(st.pos.z.x);
        mkl_records[o + 3u] = bitcast<u32>(st.dir.x.x);
        mkl_records[o + 4u] = bitcast<u32>(st.dir.y.x);
        mkl_records[o + 5u] = bitcast<u32>(st.dir.z.x);
        mkl_records[o + 6u] = bitcast<u32>(opl.x);
        mkl_records[o + 7u] = bitcast<u32>(opl.y);
        mkl_records[o + 8u] = bitcast<u32>(st.weight);
        let slots = prm.trace.num_steps + 1u;
        mkl_records[slots * g_stride * MKL_TRACE_RECORD_FLOATS + slot * g_stride + g_ray] = st.status;
    } else if (mode == MKL_REC_RESIDENT) {
        let a = (slot * g_stride + g_ray) * MKL_AIM_RECORD_FLOATS;
        let b = (prm.trace.num_steps + 1u) * g_stride * MKL_AIM_RECORD_FLOATS + a;
        mkl_records[a] = bitcast<u32>(st.pos.x.x);
        mkl_records[b] = bitcast<u32>(st.pos.x.y);
        mkl_records[a + 1u] = bitcast<u32>(st.pos.y.x);
        mkl_records[b + 1u] = bitcast<u32>(st.pos.y.y);
        mkl_records[a + 2u] = bitcast<u32>(st.pos.z.x);
        mkl_records[b + 2u] = bitcast<u32>(st.pos.z.y);
        mkl_records[a + 3u] = bitcast<u32>(st.dir.x.x);
        mkl_records[b + 3u] = bitcast<u32>(st.dir.x.y);
        mkl_records[a + 4u] = bitcast<u32>(st.dir.y.x);
        mkl_records[b + 4u] = bitcast<u32>(st.dir.y.y);
        mkl_records[a + 5u] = bitcast<u32>(st.dir.z.x);
        mkl_records[b + 5u] = bitcast<u32>(st.dir.z.y);
        mkl_records[a + 6u] = bitcast<u32>(st.opl.x);
        mkl_records[b + 6u] = bitcast<u32>(st.opl.y);
        mkl_records[a + 7u] = bitcast<u32>(st.weight);
        mkl_records[b + 7u] = 0u;
        mkl_records[a + 8u] = bitcast<u32>(st.normal.x.x);
        mkl_records[b + 8u] = bitcast<u32>(st.normal.x.y);
        mkl_records[a + 9u] = bitcast<u32>(st.normal.y.x);
        mkl_records[b + 9u] = bitcast<u32>(st.normal.y.y);
        mkl_records[a + 10u] = bitcast<u32>(st.normal.z.x);
        mkl_records[b + 10u] = bitcast<u32>(st.normal.z.y);
        mkl_records[a + 11u] = bitcast<u32>(st.index.x);
        mkl_records[b + 11u] = bitcast<u32>(st.index.y);
        mkl_records[a + 12u] = st.status;
        mkl_records[b + 12u] = 0u;
    }
}

// The trace core (mkl_trace_body and mkl_trace_raw): unrounded hi+lo
// position, direction and OPL, the unnormalised normal and the medium index
// after every step of the path program, from the input ray of system `hdr`
// at wavelength `w`. Records go to slot 0 (the input), slot k+1 after step
// k, and the final state repeated for skipped slots of a failed ray. The
// host has validated the program, the systems and the wavelengths.
fn mkl_trace_core(hdr: u32, w: u32, input: MklRaw, mode: u32) -> MklRaw {
    let clip = (prm.trace.flags & MKL_TRACE_CLIP) != 0u;
    let radiometry = (prm.trace.flags & MKL_TRACE_RADIOMETRY) != 0u;
    let at_infinity = (mkl_headers[hdr].flags & MKL_SYS_OBJECT_AT_INFINITY) != 0u;
    let sb = mkl_headers[hdr].surface_offset;
    let num_steps = prm.trace.num_steps;

    var pos = input.pos;
    var dir = input.dir;
    var opl = input.opl;
    let w_in = input.weight;
    var tw = 1.0;
    var absorb = 0.0;
    let start = prm.trace.start_surface;
    var n_cur = mkl_index(sb + start, w);
    var last = start;
    var status = MKL_RAY_OK;
    var done = 0u;
    var normal = mkl_zero_dv();

    mkl_rec(mode, 0u, MklRaw(pos, dir, normal, opl, n_cur, w_in, 1.0, MKL_RAY_OK, last, 0u));
    if (mkl_finite3(mkl_hi3(pos)) && mkl_finite3(mkl_hi3(dir)) && mkl_finite3(mkl_lo3(pos))
        && mkl_finite3(mkl_lo3(dir))) {
        dir = mkl_renorm(dir);
    } else {
        status = MKL_RAY_NONFINITE | (start << MKL_RAY_SURFACE_SHIFT);
    }

    for (var k = 0u; status == MKL_RAY_OK && k < num_steps; k++) {
        let word = mkl_steps[k];
        let si = word & MKL_STEP_SURFACE_MASK;
        let step_mode = (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK;
        let back = (word & MKL_STEP_BACKWARD) != 0u;
        var gap = si - 1u;
        if (back) {
            gap = si;
        }
        let n1 = mkl_index(sb + gap, w);
        var alpha = 0.0;
        if (!(gap == 0u && at_infinity)) {
            alpha = mkl_surfaces[sb + gap].alpha_after[w];
        }
        var n_other = n1;
        if (!back) {
            n_other = mkl_index(sb + si, w);
        } else if (si >= 1u) {
            n_other = mkl_index(sb + si - 1u, w);
        }
        // 1. Transfer into the surface's frame.
        var pd: MklPd;
        if (back) {
            pd = mkl_xf_backward(sb + si, pos, dir);
        } else {
            pd = mkl_xf_forward(sb + si - 1u, pos, dir);
        }
        pos = pd.p;
        dir = pd.d;
        let surf = sb + si;
        last = si;
        n_cur = n1;
        let fail_s = si << MKL_RAY_SURFACE_SHIFT;
        // 2. Intersect; optical path and absorption of the segment.
        let hit = mkl_intersect(surf, pos, dir);
        if (hit.code != MKL_RAY_OK) {
            status = hit.code | fail_s;
            break;
        }
        opl = mkl_df_add(opl, mkl_df_mul(hit.ta, n1));
        opl = mkl_df_add(opl, mkl_df_mul(hit.tb, n1));
        if (alpha != 0.0) {
            absorb = mkl_fma(alpha, mkl_add(hit.ta.x, hit.tb.x), absorb);
        }
        pos = hit.x;
        normal = hit.g;
        if (!mkl_finite3(mkl_hi3(pos))) {
            status = MKL_RAY_NONFINITE | fail_s;
            break;
        }
        // 3. Clip.
        let shape = mkl_surfaces[surf].shape;
        let is_cb = shape == MKL_SHAPE_COORD_BREAK;
        if (clip && !is_cb && !mkl_passes_aperture(surf, pos.x, pos.y)) {
            status = MKL_RAY_VIGNETTED | fail_s;
            break;
        }
        // 4. Interact (0 none, 1 refract, 2 reflect, 3 lens, 4 reflecting
        // lens). A DOE's selected nonzero order replaces 1 and 2
        // (trace_kernel's Act::Diffract), tested first.
        let is_image = (mkl_surfaces[surf].flags & MKL_SURF_IMAGE) != 0u;
        let paraxial = shape == MKL_SHAPE_PARAXIAL;
        let mirror = mkl_surfaces[surf].interaction == MKL_INTERACT_REFLECT;
        let dummy = mkl_surfaces[surf].interaction == MKL_INTERACT_DUMMY;
        var act = 0u;
        if (!is_cb) {
            if (step_mode == MKL_STEP_SEQUENTIAL) {
                if (is_image || dummy) {
                    act = 0u;
                } else if (paraxial) {
                    act = select(3u, 4u, mirror);
                } else {
                    act = select(1u, 2u, mirror);
                }
            } else if (step_mode == MKL_STEP_TRANSMIT) {
                if (is_image || dummy) {
                    act = 0u;
                } else {
                    act = select(1u, 3u, paraxial);
                }
            } else {
                act = select(2u, 4u, paraxial && mirror);
            }
        }
        let coating = mkl_surfaces[surf].coating;
        let dfv = mkl_hi3(dir);
        if ((act == 1u || act == 2u) && shape == MKL_SHAPE_DIFFRACTIVE && !dummy && !is_image
            && mkl_pool[mkl_surface_data(surf) + 2u * MKL_COEFF_TRACE_ORDER] != 0.0) {
            // m lambda/(2 pi) scales grad Phi to momentum and Phi to optical
            // path.
            let scale = load_d(mkl_surface_data(surf), MKL_COEFF_PREFIX + w);
            let v = mkl_phase_d(surf, pos.x, pos.y);
            let refl = act == 2u;
            var n2 = n_other;
            if (refl) {
                n2 = n1;
            }
            let turn = mkl_diffract(dir, hit.g, n1, n2, mkl_df_mul(v.x, scale), mkl_df_mul(v.y, scale), refl);
            if (!turn.ok) {
                status = MKL_RAY_EVANESCENT | fail_s;
                break;
            }
            dir = turn.d;
            n_cur = n2;
            opl = mkl_df_add(opl, mkl_df_mul(scale, v.v));
            if (radiometry) {
                let cos_i = mkl_reflect_cos(dfv, hit.normal);
                let rt = mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i);
                tw = mkl_mul(tw, select(rt.y, rt.x, refl));
            }
        } else if (act == 1u) {
            let sn = mkl_refract(dfv, hit.normal, n1.x, n_other.x);
            if (!sn.ok) {
                status = MKL_RAY_TIR | fail_s;
                break;
            }
            let mu = mkl_df_div_rn(n1, n_other);
            let b = mkl_div_rn(sn.gamma, mkl_norm3(mkl_hi3(hit.g)));
            let d2 = MklDv(
                mkl_df_add(mkl_df_mul(dir.x, mu), mkl_df_mul_f32(hit.g.x, b)),
                mkl_df_add(mkl_df_mul(dir.y, mu), mkl_df_mul_f32(hit.g.y, b)),
                mkl_df_add(mkl_df_mul(dir.z, mu), mkl_df_mul_f32(hit.g.z, b)),
            );
            dir = mkl_renorm_along(mkl_renorm_along(d2, hit.g), hit.g);
            n_cur = n_other;
            if (radiometry) {
                tw = mkl_mul(tw, mkl_surface_rt(surf, mirror, n1.x, n_other.x, sn.cos_i).y);
            }
        } else if (act == 2u) {
            let cos_i = mkl_reflect_cos(dfv, hit.normal);
            let gh = mkl_hi3(hit.g);
            let b = mkl_div_rn(mkl_mul(-2.0, mkl_dot3(dfv, gh)), mkl_dot3(gh, gh));
            let d2 = MklDv(
                mkl_df_add(dir.x, mkl_df_mul_f32(hit.g.x, b)),
                mkl_df_add(dir.y, mkl_df_mul_f32(hit.g.y, b)),
                mkl_df_add(dir.z, mkl_df_mul_f32(hit.g.z, b)),
            );
            dir = mkl_renorm_along(d2, hit.g);
            if (radiometry) {
                tw = mkl_mul(tw, mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i).x);
            }
        } else if (act >= 3u) {
            let refl = act == 4u;
            var kk = dir;
            if (refl) {
                kk.z = mkl_df_neg(kk.z);
            }
            var n2 = n_other;
            if (refl) {
                n2 = n1;
            }
            var verg = vec2<f32>(0.0);
            if (!back) {
                verg = vec2<f32>(mkl_surfaces[surf].vergence[w], mkl_surfaces[surf].vergence_lo[w]);
            }
            let phi = vec2<f32>(mkl_surfaces[surf].paraxial_power, mkl_surfaces[surf].paraxial_power_lo);
            let lens = mkl_ideal_lens(pos.x, pos.y, kk, phi, n1, n2, verg);
            if (!lens.ok) {
                status = MKL_RAY_DEGENERATE | fail_s;
                break;
            }
            dir = lens.d;
            n_cur = n2;
            opl = mkl_df_add(opl, lens.delta);
            if (radiometry && coating == MKL_COATING_IDEAL) {
                tw = mkl_mul(tw, select(mkl_surfaces[surf].coating_t, mkl_surfaces[surf].coating_r, refl));
            }
        }
        done = k + 1u;
        if (mode != MKL_REC_NONE) {
            let tr = mkl_exp(-absorb);
            var wnow = w_in;
            if (radiometry) {
                wnow = mkl_mul(mkl_mul(w_in, tw), tr);
            }
            mkl_rec(mode, done, MklRaw(pos, dir, normal, opl, n_cur, wnow, tr, MKL_RAY_OK, last, done));
        }
    }

    let transmittance = mkl_exp(-absorb);
    var weight = w_in;
    if (radiometry) {
        weight = mkl_mul(mkl_mul(w_in, tw), transmittance);
    }
    let st = MklRaw(pos, dir, normal, opl, n_cur, weight, transmittance, status, last, done);
    if (mode != MKL_REC_NONE) {
        for (var slot = done + 1u; slot <= num_steps; slot++) {
            mkl_rec(mode, slot, st);
        }
    }
    return st;
}

// One thread per ray (mkl_trace). The host has validated the program, the
// systems and the wavelengths (PreparedTrace::new).
@compute @workgroup_size(64)
fn mkl_trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    let i = gid.x;
    if (i >= prm.trace.num_rays) {
        return;
    }
    g_ray = i;
    g_stride = prm.trace.stride;
    let w = mkl_in[mkl_plane(MKL_TRACE_IN_FLOATS + MKL_TRACE_INU_WAVELENGTH, i)];
    let hdr = mkl_in[mkl_plane(MKL_TRACE_IN_FLOATS + MKL_TRACE_INU_SYSTEM, i)];
    var input: MklRaw;
    input.pos = MklDv(
        vec2<f32>(mkl_in_f(MKL_TRACE_IN_X), mkl_in_f(MKL_TRACE_IN_X_LO)),
        vec2<f32>(mkl_in_f(MKL_TRACE_IN_Y), mkl_in_f(MKL_TRACE_IN_Y_LO)),
        vec2<f32>(mkl_in_f(MKL_TRACE_IN_Z), mkl_in_f(MKL_TRACE_IN_Z_LO)),
    );
    input.dir = MklDv(
        vec2<f32>(mkl_in_f(MKL_TRACE_IN_L), mkl_in_f(MKL_TRACE_IN_L_LO)),
        vec2<f32>(mkl_in_f(MKL_TRACE_IN_M), mkl_in_f(MKL_TRACE_IN_M_LO)),
        vec2<f32>(mkl_in_f(MKL_TRACE_IN_N), mkl_in_f(MKL_TRACE_IN_N_LO)),
    );
    input.opl = vec2<f32>(mkl_in_f(MKL_TRACE_IN_OPL_HI), mkl_in_f(MKL_TRACE_IN_OPL_LO));
    input.weight = mkl_in_f(MKL_TRACE_IN_WEIGHT);
    var st: MklRaw;
    if ((prm.trace.flags & MKL_TRACE_RECORD) != 0u) {
        st = mkl_trace_core(hdr, w, input, MKL_REC_TRACE);
    } else {
        st = mkl_trace_core(hdr, w, input, MKL_REC_NONE);
    }
    let opl = mkl_report_opl(st.pos, st.dir, st.opl, st.index);
    mkl_put_f(MKL_TRACE_OUT_X, st.pos.x.x);
    mkl_put_f(MKL_TRACE_OUT_Y, st.pos.y.x);
    mkl_put_f(MKL_TRACE_OUT_Z, st.pos.z.x);
    mkl_put_f(MKL_TRACE_OUT_L, st.dir.x.x);
    mkl_put_f(MKL_TRACE_OUT_M, st.dir.y.x);
    mkl_put_f(MKL_TRACE_OUT_N, st.dir.z.x);
    mkl_put_f(MKL_TRACE_OUT_OPL_HI, opl.x);
    mkl_put_f(MKL_TRACE_OUT_OPL_LO, opl.y);
    mkl_put_f(MKL_TRACE_OUT_WEIGHT, st.weight);
    mkl_put_f(MKL_TRACE_OUT_TRANSMITTANCE, st.transmittance);
    mkl_out[mkl_plane(MKL_TRACE_OUT_FLOATS + MKL_TRACE_OUTU_STATUS, i)] = st.status;
    mkl_out[mkl_plane(MKL_TRACE_OUT_FLOATS + MKL_TRACE_OUTU_LAST_SURFACE, i)] = st.last;
    mkl_out[mkl_plane(MKL_TRACE_OUT_FLOATS + MKL_TRACE_OUTU_STEPS, i)] = st.done;
}

// Planes of the resident ray layout (aiming/mod.rs): 15 floats, system,
// wavelength, 2 logical ids, 4 report words; outputs add 3 status planes.
const MKL_RES_SYSTEM: u32 = 15u;
const MKL_RES_WAVELENGTH: u32 = 16u;
const MKL_RES_IDS: u32 = 17u;
const MKL_RES_REPORT: u32 = 19u;
const MKL_RES_STATUS: u32 = 23u;

// Resident rays carry device-computed system/wavelength indices and may run
// a device-built path program, so both are checked before they index data.
fn mkl_trace_resident_ray(i: u32, mode: u32) {
    g_ray = i;
    g_stride = prm.trace.stride;
    let system = mkl_in[mkl_plane(MKL_RES_SYSTEM, i)];
    let wl = mkl_in[mkl_plane(MKL_RES_WAVELENGTH, i)];
    let report = vec4<u32>(
        mkl_in[mkl_plane(MKL_RES_REPORT, i)],
        mkl_in[mkl_plane(MKL_RES_REPORT + 1u, i)],
        mkl_in[mkl_plane(MKL_RES_REPORT + 2u, i)],
        mkl_in[mkl_plane(MKL_RES_REPORT + 3u, i)],
    );
    var ray: MklRaw;
    ray.pos = MklDv(
        vec2<f32>(mkl_in_f(0u), mkl_in_f(9u)),
        vec2<f32>(mkl_in_f(1u), mkl_in_f(10u)),
        vec2<f32>(mkl_in_f(2u), mkl_in_f(11u)),
    );
    ray.dir = MklDv(
        vec2<f32>(mkl_in_f(3u), mkl_in_f(12u)),
        vec2<f32>(mkl_in_f(4u), mkl_in_f(13u)),
        vec2<f32>(mkl_in_f(5u), mkl_in_f(14u)),
    );
    ray.opl = vec2<f32>(mkl_in_f(6u), mkl_in_f(7u));
    ray.weight = mkl_in_f(8u);
    ray.status = report.x;
    ray.last = report.x >> MKL_RAY_SURFACE_SHIFT;
    if (ray.status == 0u) {
        var valid = system < prm.systems;
        if (valid) {
            let h = &mkl_headers[system];
            valid = wl < (*h).num_wavelengths && prm.trace.start_surface < (*h).num_surfaces;
            var current = prm.trace.start_surface;
            var backwards = false;
            for (var k = 0u; valid && k < prm.trace.num_steps; k++) {
                let word = mkl_steps[k];
                let si = word & MKL_STEP_SURFACE_MASK;
                let step_mode = (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK;
                let back = (word & MKL_STEP_BACKWARD) != 0u;
                var linked = si == current + 1u;
                if (back) {
                    linked = current > 0u && si + 1u == current;
                }
                valid = (word & ~(MKL_STEP_SURFACE_MASK | (MKL_STEP_MODE_MASK << MKL_STEP_MODE_SHIFT)
                    | MKL_STEP_BACKWARD)) == 0u && si < (*h).num_surfaces && step_mode <= MKL_STEP_REFLECT
                    && linked && (k == 0u || back == backwards);
                if (valid) {
                    let s = (*h).surface_offset + si;
                    let cb = mkl_surfaces[s].shape == MKL_SHAPE_COORD_BREAK;
                    let mirror = !cb && mkl_surfaces[s].interaction == MKL_INTERACT_REFLECT;
                    valid = !(back && step_mode == MKL_STEP_SEQUENTIAL)
                        && !(step_mode == MKL_STEP_TRANSMIT && mirror)
                        && !(step_mode == MKL_STEP_REFLECT && cb);
                    backwards = back;
                    if (step_mode == MKL_STEP_REFLECT && !mirror) {
                        backwards = !backwards;
                    }
                }
                current = si;
            }
            if (valid) {
                ray = mkl_trace_core(system, wl, ray, mode);
            }
        }
        if (!valid) {
            ray.status = MKL_AIM_INVALID;
        }
    }
    if (report.x != 0u || ray.status == MKL_AIM_INVALID) {
        for (var k = 0u; k <= prm.trace.num_steps; k++) {
            mkl_rec(mode, k, ray);
        }
    }
    mkl_put_f(0u, ray.pos.x.x);
    mkl_put_f(1u, ray.pos.y.x);
    mkl_put_f(2u, ray.pos.z.x);
    mkl_put_f(3u, ray.dir.x.x);
    mkl_put_f(4u, ray.dir.y.x);
    mkl_put_f(5u, ray.dir.z.x);
    mkl_put_f(6u, ray.opl.x);
    mkl_put_f(7u, ray.opl.y);
    mkl_put_f(8u, ray.weight);
    mkl_put_f(9u, ray.pos.x.y);
    mkl_put_f(10u, ray.pos.y.y);
    mkl_put_f(11u, ray.pos.z.y);
    mkl_put_f(12u, ray.dir.x.y);
    mkl_put_f(13u, ray.dir.y.y);
    mkl_put_f(14u, ray.dir.z.y);
    mkl_out[mkl_plane(MKL_RES_SYSTEM, i)] = system;
    mkl_out[mkl_plane(MKL_RES_WAVELENGTH, i)] = wl;
    mkl_out[mkl_plane(MKL_RES_IDS, i)] = mkl_in[mkl_plane(MKL_RES_IDS, i)];
    mkl_out[mkl_plane(MKL_RES_IDS + 1u, i)] = mkl_in[mkl_plane(MKL_RES_IDS + 1u, i)];
    for (var k = 0u; k < 4u; k++) {
        mkl_out[mkl_plane(MKL_RES_REPORT + k, i)] = report[k];
    }
    mkl_out[mkl_plane(MKL_RES_STATUS, i)] = ray.status;
    mkl_out[mkl_plane(MKL_RES_STATUS + 1u, i)] = ray.last;
    mkl_out[mkl_plane(MKL_RES_STATUS + 2u, i)] = ray.done;
}

@compute @workgroup_size(64)
fn mkl_trace_resident(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    if (gid.x < prm.trace.num_rays) {
        mkl_trace_resident_ray(gid.x, MKL_REC_NONE);
    }
}

// Records cost the plain resident trace time: a separate entry point.
@compute @workgroup_size(64)
fn mkl_trace_resident_records(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    if (gid.x < prm.trace.num_rays) {
        mkl_trace_resident_ray(gid.x, MKL_REC_RESIDENT);
    }
}
