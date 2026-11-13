
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
