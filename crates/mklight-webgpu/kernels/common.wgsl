// common.wgsl — numerics shared by every MKLight WGSL kernel.
//
// Each function is the WGSL twin of a function in crates/mklight-gpu
// (df.rs, fmath.rs, mklight_core::rng) and of crates/mklight-metal/shaders/
// mklight_common.h, written operation for operation alike.
//
// Numerics rules (docs/CONVENTIONS.md §7, GPU.md §11):
//   * WGSL has no fast-math or contraction switch, and the compilers that run
//     it differ: wgpu on Metal compiles with Metal's default fast math, which
//     reassociates (folding the error term of two_sum to zero) and contracts
//     a*b + c into fma. Every value an error-free transform depends on is
//     therefore passed through mkl_opq, an XOR with a run-time zero the
//     compiler cannot see through: no algebraic rewrite or fusion crosses it.
//     Fusion is always written explicitly with fma().
//   * The backend runs mkl_numerics_probe at start-up (raw and protected
//     operations) and refuses double-float work if the protected operations
//     are not IEEE.
//   * WGSL has no isnan/isinf; mkl_is_nan/mkl_is_finite test the bits (fast
//     math may fold float comparisons with NaN).
//   * Exact float equality only against the sentinels of mklight_gpu_types.h.
//   * Devices may flush subnormals (Apple GPUs do); never rely on gradual
//     underflow.
//
// Binding 0 of group 0 is the environment block every pass receives:
// x = 0 (the opaque zero), yzw reserved. Kernels call mkl_enter() first.

@group(0) @binding(0) var<uniform> mkl_env: vec4<u32>;

var<private> mkl_zero: u32;

fn mkl_enter() {
    mkl_zero = mkl_env.x;
}

// The opaque identity: x bit for bit, invisible to the optimiser.
fn mkl_opq(x: f32) -> f32 {
    return bitcast<f32>(bitcast<u32>(x) ^ mkl_zero);
}

// Exact powers of two from their bit patterns: 2^e = (e + 127) << 23.
fn mkl_pow2(e: i32) -> f32 {
    return bitcast<f32>(u32(e + 127) << 23u);
}

const MKL_INF_BITS: u32 = 0x7F800000u;
fn mkl_qnan() -> f32 {
    return bitcast<f32>(0x7FC00000u ^ mkl_zero);
}
fn mkl_inf() -> f32 {
    return bitcast<f32>(MKL_INF_BITS ^ mkl_zero);
}
fn mkl_is_nan(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7FFFFFFFu) > MKL_INF_BITS;
}
fn mkl_is_inf(x: f32) -> bool {
    return (bitcast<u32>(x) & 0x7FFFFFFFu) == MKL_INF_BITS;
}
fn mkl_is_finite(x: f32) -> bool {
    return (bitcast<u32>(x) & MKL_INF_BITS) != MKL_INF_BITS;
}

// ---------------------------------------------------------------------------
// Protected scalar operations: one IEEE rounding each, never fused or
// reassociated with a neighbour.
// ---------------------------------------------------------------------------
fn mkl_add(a: f32, b: f32) -> f32 {
    return mkl_opq(a + b);
}
fn mkl_sub(a: f32, b: f32) -> f32 {
    return mkl_opq(a - b);
}
fn mkl_mul(a: f32, b: f32) -> f32 {
    return mkl_opq(a * b);
}
fn mkl_fma(a: f32, b: f32, c: f32) -> f32 {
    return mkl_opq(fma(a, b, c));
}
fn mkl_div(a: f32, b: f32) -> f32 {
    return mkl_opq(a / b);
}
fn mkl_sqrt(a: f32) -> f32 {
    return mkl_opq(sqrt(a));
}

// ---------------------------------------------------------------------------
// Error-free transforms
// ---------------------------------------------------------------------------

// s + t == a + b exactly (Knuth 2Sum, no precondition).
fn mkl_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = mkl_add(a, b);
    let a1 = mkl_sub(s, b);
    let b1 = mkl_sub(s, a1);
    let da = mkl_sub(a, a1);
    let db = mkl_sub(b, b1);
    return vec2<f32>(s, mkl_add(da, db));
}

// s + t == a + b exactly when |a| >= |b| or a == 0 (Dekker Fast2Sum).
fn mkl_fast_two_sum(a: f32, b: f32) -> vec2<f32> {
    let s = mkl_add(a, b);
    let z = mkl_sub(s, a);
    return vec2<f32>(s, mkl_sub(b, z));
}

// p + e == a * b exactly (barring underflow), using the hardware FMA.
fn mkl_two_prod(a: f32, b: f32) -> vec2<f32> {
    let p = mkl_mul(a, b);
    return vec2<f32>(p, mkl_fma(a, b, -p));
}

// ---------------------------------------------------------------------------
// Double-float arithmetic: x = (hi, lo) = x.x + x.y, |lo| <= ulp(hi)/2
// (Joldes, Muller, Popescu, ACM TOMS 44(2), 2017; bounds in df.rs).
// ---------------------------------------------------------------------------

// DWPlusFP (Alg. 4), relative error <= 2u^2.
fn mkl_df_add_f32(x: vec2<f32>, y: f32) -> vec2<f32> {
    let s = mkl_two_sum(x.x, y);
    let v = mkl_add(x.y, s.y);
    return mkl_fast_two_sum(s.x, v);
}

// AccurateDWPlusDW (Alg. 6), relative error <= 3u^2 + 13u^3.
fn mkl_df_add(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let s = mkl_two_sum(x.x, y.x);
    let t = mkl_two_sum(x.y, y.y);
    let c = mkl_add(s.y, t.x);
    let v = mkl_fast_two_sum(s.x, c);
    let w = mkl_add(t.y, v.y);
    return mkl_fast_two_sum(v.x, w);
}

fn mkl_df_neg(x: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-x.x, -x.y);
}
fn mkl_df_sub(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    return mkl_df_add(x, mkl_df_neg(y));
}

// DWTimesFP3 (Alg. 9), relative error <= 2u^2.
fn mkl_df_mul_f32(x: vec2<f32>, y: f32) -> vec2<f32> {
    let c = mkl_two_prod(x.x, y);
    let cl3 = mkl_fma(x.y, y, c.y);
    return mkl_fast_two_sum(c.x, cl3);
}

// DWTimesDW3 (Alg. 12), relative error <= 4u^2.
fn mkl_df_mul(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let c = mkl_two_prod(x.x, y.x);
    let tl0 = mkl_mul(x.y, y.y);
    let tl1 = mkl_fma(x.x, y.y, tl0);
    let cl2 = mkl_fma(x.y, y.x, tl1);
    let cl3 = mkl_add(c.y, cl2);
    return mkl_fast_two_sum(c.x, cl3);
}

// DWDivDW2 (Alg. 17), relative error <= 15u^2 + 56u^3.
fn mkl_df_div(x: vec2<f32>, y: vec2<f32>) -> vec2<f32> {
    let th = mkl_div(x.x, y.x);
    let r = mkl_df_mul_f32(y, th);
    let pi_h = mkl_sub(x.x, r.x);
    let dl = mkl_sub(x.y, r.y);
    let d = mkl_add(pi_h, dl);
    let tl = mkl_div(d, y.x);
    return mkl_fast_two_sum(th, tl);
}

// sqrt with one Newton (Karp) correction from an exact residual.
fn mkl_df_sqrt(x: vec2<f32>) -> vec2<f32> {
    if (x.x <= 0.0) {
        if (x.x == 0.0) {
            return vec2<f32>(0.0, 0.0);
        }
        return vec2<f32>(mkl_qnan(), mkl_qnan());
    }
    let s = mkl_sqrt(x.x);
    let p = mkl_two_prod(s, s);
    let r = mkl_add(mkl_sub(mkl_sub(x.x, p.x), p.y), x.y);
    let e = mkl_div(r, mkl_mul(2.0, s));
    return mkl_fast_two_sum(s, e);
}

// ---------------------------------------------------------------------------
// Compensated summation (CONVENTIONS §7.5): Neumaier's step; the result is
// two_sum(s, c). State is (s, c).
// ---------------------------------------------------------------------------
fn mkl_neumaier_add(sc: vec2<f32>, x: f32) -> vec2<f32> {
    let s = sc.x;
    let t = mkl_add(s, x);
    var c = sc.y;
    if (abs(s) >= abs(x)) {
        c = mkl_add(c, mkl_add(mkl_sub(s, t), x));
    } else {
        c = mkl_add(c, mkl_add(mkl_sub(x, t), s));
    }
    return vec2<f32>(t, c);
}

// ---------------------------------------------------------------------------
// Philox4x32-10 (CONVENTIONS §6): exactly mklight_core::rng::philox4x32_10.
// ---------------------------------------------------------------------------
fn mkl_mulhi(a: u32, b: u32) -> u32 {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let p00 = a0 * b0;
    let p01 = a0 * b1;
    let p10 = a1 * b0;
    let p11 = a1 * b1;
    let mid = (p00 >> 16u) + (p01 & 0xFFFFu) + (p10 & 0xFFFFu);
    return p11 + (p01 >> 16u) + (p10 >> 16u) + (mid >> 16u);
}

fn mkl_philox4x32_10(c_in: vec4<u32>, k_in: vec2<u32>) -> vec4<u32> {
    var c = c_in;
    var k = k_in;
    for (var round = 0; round < 10; round++) {
        if (round > 0) {
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        let hi0 = mkl_mulhi(0xD2511F53u, c.x);
        let lo0 = 0xD2511F53u * c.x;
        let hi1 = mkl_mulhi(0xCD9E8D57u, c.z);
        let lo1 = 0xCD9E8D57u * c.z;
        c = vec4<u32>(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    return c;
}

// Uniform in (0, 1) from the top 24 bits (mklight_gpu::fmath::u32_to_unit_f32).
fn mkl_u32_to_unit(x: u32) -> f32 {
    let k = x >> 8u;
    var half_ulp = 0.0;
    if (k < 0x800000u) {
        half_ulp = mkl_pow2(-25);
    }
    return mkl_fma(f32(k), mkl_pow2(-24), half_ulp);
}

// ---------------------------------------------------------------------------
// Utility math (CONVENTIONS §7.3, §7.6)
// ---------------------------------------------------------------------------

// Same nesting as mklight_core::Vec3::dot.
fn mkl_dot3(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return mkl_fma(a.x, b.x, mkl_fma(a.y, b.y, mkl_mul(a.z, b.z)));
}

// Power-of-two scaling into [2^-60, 2^60] so squares cannot overflow or
// underflow; exact.
fn mkl_hypot_scale(m: f32) -> vec2<f32> {
    if (m > mkl_pow2(60)) {
        return vec2<f32>(mkl_pow2(-80), mkl_pow2(80));
    }
    if (m < mkl_pow2(-60)) {
        return vec2<f32>(mkl_pow2(80), mkl_pow2(-80));
    }
    return vec2<f32>(1.0, 1.0);
}

// IEEE hypot special values (mklight_common.h): (value, 1) for a special
// case, (0, 0) when m is regular. A flag, not a negative sentinel, because a
// fast-math compiler may assume comparisons never see a NaN.
fn mkl_hypot_special(m: f32, any_nan: bool) -> vec2<f32> {
    if (mkl_is_inf(m)) {
        return vec2<f32>(mkl_inf(), 1.0);
    }
    if (any_nan) {
        return vec2<f32>(mkl_qnan(), 1.0);
    }
    if (m == 0.0) {
        return vec2<f32>(0.0, 1.0);
    }
    return vec2<f32>(0.0, 0.0);
}

// max() that ignores a NaN operand like MSL/IEEE maxNum (only used on |x|).
fn mkl_max_num(a: f32, b: f32) -> f32 {
    if (mkl_is_nan(a)) {
        return b;
    }
    if (mkl_is_nan(b)) {
        return a;
    }
    return select(b, a, a >= b);
}

fn mkl_hypot2(x: f32, y: f32) -> f32 {
    let m = mkl_max_num(abs(x), abs(y));
    let sp = mkl_hypot_special(m, mkl_is_nan(x) || mkl_is_nan(y));
    if (sp.y != 0.0) {
        return sp.x;
    }
    let sc = mkl_hypot_scale(m);
    let xs = mkl_mul(x, sc.x);
    let ys = mkl_mul(y, sc.x);
    return mkl_mul(mkl_sqrt(mkl_fma(xs, xs, mkl_mul(ys, ys))), sc.y);
}

fn mkl_hypot3(x: f32, y: f32, z: f32) -> f32 {
    let m = mkl_max_num(mkl_max_num(abs(x), abs(y)), abs(z));
    let sp = mkl_hypot_special(m, mkl_is_nan(x) || mkl_is_nan(y) || mkl_is_nan(z));
    if (sp.y != 0.0) {
        return sp.x;
    }
    let sc = mkl_hypot_scale(m);
    let v = vec3<f32>(mkl_mul(x, sc.x), mkl_mul(y, sc.x), mkl_mul(z, sc.x));
    return mkl_mul(mkl_sqrt(mkl_dot3(v, v)), sc.y);
}

// e^x in f32 (mklight_gpu::fmath::exp): Cody-Waite reduction, degree-7
// Taylor polynomial, two exact power-of-two factors.
fn mkl_exp(x: f32) -> f32 {
    if (mkl_is_nan(x)) {
        return x;
    }
    if (x > 88.72284) {
        return mkl_inf();
    }
    if (x < -103.97208) {
        return 0.0;
    }
    let kf = round(mkl_mul(x, bitcast<f32>(0x3FB8AA3Bu)));
    let k = i32(kf);
    var r = mkl_fma(-kf, bitcast<f32>(0x3F317200u), x);
    r = mkl_fma(-kf, bitcast<f32>(0x35BFBE8Eu), r);
    var p = bitcast<f32>(0x39500D01u);
    p = mkl_fma(p, r, bitcast<f32>(0x3AB60B61u));
    p = mkl_fma(p, r, bitcast<f32>(0x3C088889u));
    p = mkl_fma(p, r, bitcast<f32>(0x3D2AAAABu));
    p = mkl_fma(p, r, bitcast<f32>(0x3E2AAAABu));
    p = mkl_fma(p, r, 0.5);
    p = mkl_fma(p, r, 1.0);
    p = mkl_fma(p, r, 1.0);
    let k1 = k / 2;
    return mkl_mul(mkl_mul(p, mkl_pow2(k1)), mkl_pow2(k - k1));
}
