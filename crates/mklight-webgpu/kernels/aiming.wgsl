// ---------------------------------------------------------------------------
// Pupil aiming (aiming::Request::GenerateAim from host chief seeds, with a
// host seed table or none): the WGSL twin of mklight_gpu::aiming::kernel's
// FieldKernel::{generate, aim, start} and of its double-float Newton–Broyden
// solver, operation for operation; every evaluation traces the stop with the
// trace core above (mkl_trace_core, no records). build.rs appends this file
// to trace.wgsl and rounds every division and square root like the trace's,
// so generated rays and reports are bitwise the CPU-f32 emulation's.
//
// One thread per sample runs the sample's whole aiming schedule. Bindings
// are the trace's: mkl_in holds a 16-word header (AIM_* below) and the
// field rows, chief seeds, seed table and the launch's samples at the
// header's word offsets; mkl_out receives the ResidentRays planes of the
// launch (15 floats, system, wavelength, 2 logical IDs, 4 report words),
// each `count` long; mkl_steps is the sequential path 1, 2, …; the uniform
// carries the aiming trace settings (newton_params: no clipping, start
// surface 0). The host has validated the field rows and folded culled
// samples into MKL_AIM_SKIP.
// ---------------------------------------------------------------------------

const AIM_COUNT: u32 = 0u;
const AIM_CAP: u32 = 1u;
const AIM_TOLERANCE: u32 = 2u;
const AIM_TABLE: u32 = 3u;
const AIM_SIDE: u32 = 4u;
const AIM_SCHEDULE_CAP: u32 = 5u;
const AIM_FIELDS_AT: u32 = 6u;
const AIM_SEEDS_AT: u32 = 7u;
const AIM_TABLE_AT: u32 = 8u;
const AIM_SAMPLES_AT: u32 = 9u;

// MklAimField word offsets.
const AF_SYSTEM: u32 = 0u;
const AF_WAVELENGTH: u32 = 1u;
const AF_FLAGS: u32 = 2u;
const AF_PUPIL: u32 = 12u;
const AF_VIGNETTE: u32 = 18u;
const AF_ORIGIN: u32 = 26u;
const AF_ROTATION: u32 = 32u;

struct AimV2 {
    x: vec2<f32>,
    y: vec2<f32>,
}
// [[a, b], [c, d]]
struct AimM2 {
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
    d: vec2<f32>,
}
struct AimEval {
    ok: bool,
    r: AimV2,
}
struct AimSolve {
    x: AimV2,
    j: AimM2,
    has_j: bool,
    residual: vec2<f32>,
    iterations: u32,
    converged: bool,
}
struct AimStart {
    ok: bool,
    ray: MklRaw,
}
struct AimSeed {
    ok: bool,
    x: AimV2,
    j: AimM2,
}

// The sample's field row (word offset), system, wavelength, chief source
// and current Newton target.
var<private> a_field: u32;
var<private> a_system: u32;
var<private> a_wl: u32;
var<private> a_src: MklDv;
var<private> a_target: AimV2;

fn aim_df(o: u32) -> vec2<f32> {
    return vec2<f32>(bitcast<f32>(mkl_in[o]), bitcast<f32>(mkl_in[o + 1u]));
}
fn aim_fd(k: u32) -> vec2<f32> {
    return aim_df(a_field + k);
}
fn aim_flags() -> u32 {
    return mkl_in[a_field + AF_FLAGS];
}
fn aim_pupil(k: u32) -> vec2<f32> {
    return aim_fd(AF_PUPIL + 2u * k);
}
fn aim_rot(k: u32) -> vec2<f32> {
    return aim_fd(AF_ROTATION + 2u * k);
}
fn aim_c(x: f32) -> vec2<f32> {
    return vec2<f32>(x, 0.0);
}
// Comparisons of double-float high words against zero that are false for
// NaN, as in Rust, whatever the compiler assumes about NaN.
fn aim_le0(v: vec2<f32>) -> bool {
    return !mkl_is_nan(v.x) && v.x <= 0.0;
}
fn aim_lt0(v: vec2<f32>) -> bool {
    return !mkl_is_nan(v.x) && v.x < 0.0;
}
fn aim_gt0(v: vec2<f32>) -> bool {
    return !mkl_is_nan(v.x) && v.x > 0.0;
}

fn aim_add(a: AimV2, b: AimV2) -> AimV2 {
    return AimV2(mkl_df_add(a.x, b.x), mkl_df_add(a.y, b.y));
}
fn aim_sub(a: AimV2, b: AimV2) -> AimV2 {
    return AimV2(mkl_df_sub(a.x, b.x), mkl_df_sub(a.y, b.y));
}
fn aim_scale(a: AimV2, b: vec2<f32>) -> AimV2 {
    return AimV2(mkl_df_mul(a.x, b), mkl_df_mul(a.y, b));
}
fn aim_finite(a: AimV2) -> bool {
    return mkl_is_finite(a.x.x) && mkl_is_finite(a.y.x);
}
fn aim_norm2(v: AimV2) -> vec2<f32> {
    return mkl_df_sqrt(mkl_df_add(mkl_df_mul(v.x, v.x), mkl_df_mul(v.y, v.y)));
}
fn aim_unit(v: MklDv) -> MklDv {
    let n = mkl_df_sqrt(mkl_ddot(v, v));
    return MklDv(mkl_df_div(v.x, n), mkl_df_div(v.y, n), mkl_df_div(v.z, n));
}
fn aim_neg3(v: MklDv) -> MklDv {
    return MklDv(mkl_df_neg(v.x), mkl_df_neg(v.y), mkl_df_neg(v.z));
}

// Object-local to entrance frame: row k of the result is column k of the
// rotation times p (less the origin for a point).
fn aim_local(p: MklDv, point: bool) -> MklDv {
    var a = p;
    if (point) {
        a = MklDv(
            mkl_df_sub(p.x, aim_fd(AF_ORIGIN)),
            mkl_df_sub(p.y, aim_fd(AF_ORIGIN + 2u)),
            mkl_df_sub(p.z, aim_fd(AF_ORIGIN + 4u)),
        );
    }
    return MklDv(
        mkl_ddot(MklDv(aim_rot(0u), aim_rot(3u), aim_rot(6u)), a),
        mkl_ddot(MklDv(aim_rot(1u), aim_rot(4u), aim_rot(7u)), a),
        mkl_ddot(MklDv(aim_rot(2u), aim_rot(5u), aim_rot(8u)), a),
    );
}

// The launch ray of aim point `a` from source `src` (FieldKernel::start).
fn aim_start(src: MklDv, a: AimV2) -> AimStart {
    var out: AimStart;
    out.ok = true;
    let flags = aim_flags();
    var pos: MklDv;
    var dir: MklDv;
    if ((flags & MKL_AIM_FINITE) == 0u) {
        let ap = MklDv(a.x, a.y, aim_pupil(0u));
        let t = mkl_ddot(ap, src);
        pos = MklDv(
            mkl_df_sub(ap.x, mkl_df_mul(src.x, t)),
            mkl_df_sub(ap.y, mkl_df_mul(src.y, t)),
            mkl_df_sub(ap.z, mkl_df_mul(src.z, t)),
        );
        dir = src;
    } else if ((flags & MKL_AIM_TELECENTRIC) != 0u) {
        pos = src;
        dir = aim_local(aim_unit(MklDv(a.x, a.y, aim_c(1.0))), false);
    } else {
        let ap = aim_local(MklDv(a.x, a.y, aim_pupil(0u)), true);
        var d = aim_unit(MklDv(mkl_df_sub(ap.x, src.x), mkl_df_sub(ap.y, src.y), mkl_df_sub(ap.z, src.z)));
        let z = mkl_ddot(MklDv(aim_rot(6u), aim_rot(7u), aim_rot(8u)), d);
        if (z.x == 0.0 || !mkl_is_finite(z.x) || !mkl_finite3(mkl_hi3(d))) {
            out.ok = false;
            return out;
        }
        if (aim_lt0(z)) {
            d = aim_neg3(d);
        }
        pos = src;
        dir = d;
    }
    let zero = vec2<f32>(0.0, 0.0);
    out.ray = MklRaw(pos, dir, MklDv(zero, zero, zero), zero, zero, 1.0, 1.0, 0u, 0u, 0u);
    return out;
}

// Stop residual of aim point `a` (the Newton function).
fn aim_fun(a: AimV2) -> AimEval {
    var e: AimEval;
    e.ok = false;
    let s = aim_start(a_src, a);
    if (!s.ok) {
        return e;
    }
    let r = mkl_trace_core(a_system, a_wl, s.ray, MKL_REC_NONE, mkl_headers[a_system].stop);
    if (r.status != 0u) {
        return e;
    }
    e.ok = true;
    e.r = AimV2(mkl_df_sub(r.pos.x, a_target.x), mkl_df_sub(r.pos.y, a_target.y));
    return e;
}

fn aim_solve2(j: AimM2, r: AimV2) -> AimEval {
    var out: AimEval;
    out.ok = false;
    let det = mkl_df_sub(mkl_df_mul(j.a, j.d), mkl_df_mul(j.b, j.c));
    var s = 0.0;
    s = mkl_max_num(s, abs(j.a.x));
    s = mkl_max_num(s, abs(j.b.x));
    s = mkl_max_num(s, abs(j.c.x));
    s = mkl_max_num(s, abs(j.d.x));
    if (!mkl_is_finite(det.x) || abs(det.x) <= mkl_mul(mkl_mul(1e-14, s), s)) {
        return out;
    }
    out.r = AimV2(
        mkl_df_div(mkl_df_sub(mkl_df_mul(r.x, j.d), mkl_df_mul(j.b, r.y)), det),
        mkl_df_div(mkl_df_sub(mkl_df_mul(j.a, r.y), mkl_df_mul(r.x, j.c)), det),
    );
    out.ok = aim_finite(out.r);
    return out;
}

fn aim_budget(spent: u32) -> u32 {
    let sc = mkl_in[AIM_SCHEDULE_CAP];
    let cap = mkl_in[AIM_CAP];
    if (sc == 0u) {
        return cap;
    }
    return min(cap, select(0u, sc - spent, sc > spent));
}
fn aim_capped(spent: u32) -> bool {
    let sc = mkl_in[AIM_SCHEDULE_CAP];
    return sc != 0u && spent >= sc;
}

// Catmull–Rom weights at t (x) and their derivatives (y).
fn aim_spline(t: f32) -> array<vec2<f32>, 4> {
    let t2 = mkl_mul(t, t);
    let t3 = mkl_mul(t2, t);
    return array<vec2<f32>, 4>(
        vec2<f32>(
            mkl_mul(mkl_sub(mkl_sub(mkl_mul(t2, 2.0), t3), t), 0.5),
            mkl_mul(mkl_sub(mkl_sub(mkl_mul(t, 4.0), mkl_mul(t2, 3.0)), 1.0), 0.5),
        ),
        vec2<f32>(
            mkl_mul(mkl_add(mkl_sub(mkl_mul(t3, 3.0), mkl_mul(t2, 5.0)), 2.0), 0.5),
            mkl_mul(mkl_sub(mkl_mul(t2, 9.0), mkl_mul(t, 10.0)), 0.5),
        ),
        vec2<f32>(
            mkl_mul(mkl_add(mkl_sub(mkl_mul(t2, 4.0), mkl_mul(t3, 3.0)), t), 0.5),
            mkl_mul(mkl_add(mkl_sub(mkl_mul(t, 8.0), mkl_mul(t2, 9.0)), 1.0), 0.5),
        ),
        vec2<f32>(
            mkl_mul(mkl_sub(t3, t2), 0.5),
            mkl_mul(mkl_sub(mkl_mul(t2, 3.0), mkl_mul(t, 2.0)), 0.5),
        ),
    );
}

// Bicubic table seed at p and its stop Jacobian (kernel::table_seed); the
// table rows of the field start at word `base`.
fn aim_table_seed(base: u32, side: u32, p: AimV2, r_stop: vec2<f32>) -> AimSeed {
    var out: AimSeed;
    out.ok = false;
    let step = mkl_div(2.25, f32(side - 1u));
    let fx = mkl_div(mkl_add(p.x.x, 1.125), step);
    let fy = mkl_div(mkl_add(p.y.x, 1.125), step);
    if (mkl_is_nan(fx) || mkl_is_nan(fy)) {
        return out;
    }
    let ix = floor(fx);
    let iy = floor(fy);
    let last = f32(side - 3u);
    if (!(ix >= 1.0 && iy >= 1.0 && ix <= last && iy <= last)) {
        return out;
    }
    var w = aim_spline(mkl_sub(fx, ix));
    var v = aim_spline(mkl_sub(fy, iy));
    var a = AimV2(aim_c(0.0), aim_c(0.0));
    var ax = a;
    var ay = a;
    for (var j = 0u; j < 4u; j++) {
        for (var i = 0u; i < 4u; i++) {
            let node = base + ((u32(iy) + j - 1u) * side + u32(ix) + i - 1u) * MKL_AIM_NODE_WORDS;
            if (mkl_in[node + 4u] != 0u) {
                return out;
            }
            let wv = mkl_mul(w[i].x, v[j].x);
            let dxv = mkl_mul(w[i].y, v[j].x);
            let dyv = mkl_mul(w[i].x, v[j].y);
            let n0 = aim_df(node);
            let n1 = aim_df(node + 2u);
            a = AimV2(mkl_df_add(a.x, mkl_df_mul_f32(n0, wv)), mkl_df_add(a.y, mkl_df_mul_f32(n1, wv)));
            ax = AimV2(mkl_df_add(ax.x, mkl_df_mul_f32(n0, dxv)), mkl_df_add(ax.y, mkl_df_mul_f32(n1, dxv)));
            ay = AimV2(mkl_df_add(ay.x, mkl_df_mul_f32(n0, dyv)), mkl_df_add(ay.y, mkl_df_mul_f32(n1, dyv)));
        }
    }
    let det = mkl_df_sub(mkl_df_mul(ax.x, ay.y), mkl_df_mul(ay.x, ax.y));
    let q = mkl_df_div(r_stop, mkl_df_mul_f32(det, mkl_div(f32(side - 1u), 2.25)));
    out.x = a;
    out.j = AimM2(
        mkl_df_mul(ay.y, q),
        mkl_df_neg(mkl_df_mul(ay.x, q)),
        mkl_df_neg(mkl_df_mul(ax.y, q)),
        mkl_df_mul(ax.x, q),
    );
    out.ok = mkl_is_finite(out.j.a.x) && mkl_is_finite(out.j.b.x) && mkl_is_finite(out.j.c.x)
        && mkl_is_finite(out.j.d.x) && det.x != 0.0;
    return out;
}

// ---------------------------------------------------------------------------
// The aiming schedule as a state machine. Every Newton evaluation traces
// the stop, and GPU compilers inline every call of the trace core: a solver
// written with nested calls (schedule, Newton, line search, Jacobian
// columns) inlines the core a dozen times and does not compile in
// reasonable time. So the schedule of FieldKernel::aim and the
// Newton–Broyden solve of kernel::newton run as one machine (aim_advance)
// that stops whenever it needs the stop residual of a point (a_req) and
// resumes with it; aim_point's loop evaluates it at the one call site of
// the trace (aim_fun). The arithmetic and its order are the specification's.
// ---------------------------------------------------------------------------

// Newton states.
const AN_F0: u32 = 1u;
const AN_LOOP: u32 = 2u;
const AN_LINE: u32 = 3u;
const AN_LINE_DONE: u32 = 4u;
// Jacobian states (differences of columns 0 and 1).
const AJ_PLUS: u32 = 5u;
const AJ_MINUS: u32 = 6u;
// Schedule states (where a Newton result continues).
const AS_TABLE_DONE: u32 = 7u;
const AS_PREDICT: u32 = 8u;
const AS_PREDICT_DONE: u32 = 9u;
const AS_CONTINUE: u32 = 10u;
const AS_CONTINUE_DONE: u32 = 11u;
const AS_FINISHED: u32 = 12u;
// Why a Jacobian is built (kernel::newton's four call sites).
const AJ_INIT: u32 = 0u;
const AJ_SOLVE_FAILED: u32 = 1u;
const AJ_REJECTED: u32 = 2u;
const AJ_SLOW: u32 = 3u;

var<private> a_state: u32;
var<private> a_req: AimV2;
// Newton solve: start (x0, j0), current point, residual, Jacobian, settings,
// step and line search; the state its result returns to and the result.
var<private> n_x0: AimV2;
var<private> n_j0: AimM2;
var<private> n_has_j0: bool;
var<private> n_x: AimV2;
var<private> n_r: AimV2;
var<private> n_rn: vec2<f32>;
var<private> n_j: AimM2;
var<private> n_fresh: bool;
var<private> n_it: u32;
var<private> n_cap: u32;
var<private> n_h: vec2<f32>;
var<private> n_tol: vec2<f32>;
var<private> n_dx: AimV2;
var<private> n_lam: vec2<f32>;
var<private> n_k: u32;
var<private> n_then: u32;
var<private> n_result: AimSolve;
// Jacobian: purpose, column, its plus-side evaluation, column 0.
var<private> j_why: u32;
var<private> j_col: u32;
var<private> j_plus: AimEval;
var<private> j_first: AimV2;
// Schedule: target, iterations spent, chief prediction, continuation.
var<private> s_goal: AimV2;
var<private> s_it: u32;
var<private> s_chief: AimV2;
var<private> s_j0: AimM2;
var<private> s_has_j: bool;
var<private> s_p: AimV2;
var<private> s_xx: AimV2;
var<private> s_jj: AimM2;
var<private> s_hj: bool;
var<private> s_prev: AimV2;
var<private> s_t: AimV2;
var<private> s_k: u32;

// Start a Newton solve (kernel::solve_capped) towards `goal`; its result
// continues in state `then`. Requests f(x).
fn aim_newton(goal: AimV2, x: AimV2, j: AimM2, has_j: bool, cap: u32, then: u32) {
    a_target = goal;
    n_h = mkl_df_mul(aim_c(1e-6), aim_c(mkl_max_num(aim_pupil(1u).x, 1e-6)));
    n_tol = mkl_df_mul(aim_c(bitcast<f32>(mkl_in[AIM_TOLERANCE])), aim_pupil(2u));
    n_x0 = x;
    n_j0 = j;
    n_has_j0 = has_j;
    n_cap = cap;
    n_then = then;
    a_req = x;
    a_state = AN_F0;
}
// Leave the solve with `s`.
fn aim_newton_return(s: AimSolve) {
    n_result = s;
    a_state = n_then;
}
// The solve's result after its loop.
fn aim_newton_done() {
    aim_newton_return(AimSolve(n_x, n_j, true, n_rn, n_it, aim_le0(mkl_df_sub(n_rn, n_tol))));
}
// Start a Jacobian at (n_x, n_r): requests f(x + h e0).
fn aim_jacobian(why: u32) {
    j_why = why;
    j_col = 0u;
    a_req = n_x;
    a_req.x = mkl_df_add(a_req.x, n_h);
    a_state = AJ_PLUS;
}
// A finished Jacobian attempt (ok: jj built).
fn aim_jacobian_done(ok: bool, jj: AimM2) {
    if (j_why == AJ_INIT && !ok) {
        aim_newton_return(AimSolve(n_x0, n_j0, n_has_j0, n_rn, 0u, false));
        return;
    }
    if (j_why != AJ_SLOW && !ok) {
        aim_newton_done();
        return;
    }
    if (ok) {
        n_j = jj;
        n_fresh = true;
    }
    a_state = AN_LOOP;
}

// Advance the schedule with `e`, the evaluation of the last request (if
// any). Returns true with a new request in a_req, false when the schedule
// has finished (its result in n_result).
fn aim_advance(e: AimEval) -> bool {
    loop {
        switch (a_state) {
            case AN_F0: {
                if (!e.ok) {
                    aim_newton_return(AimSolve(n_x0, n_j0, n_has_j0, aim_c(mkl_inf()), 0u, false));
                    break;
                }
                n_x = n_x0;
                n_r = e.r;
                n_rn = aim_norm2(n_r);
                n_j = n_j0;
                n_fresh = false;
                n_it = 0u;
                if (!n_has_j0) {
                    aim_jacobian(AJ_INIT);
                    return true;
                }
                a_state = AN_LOOP;
            }
            case AN_LOOP: {
                if (n_it >= n_cap || aim_le0(mkl_df_sub(n_rn, n_tol))) {
                    aim_newton_done();
                    break;
                }
                n_it += 1u;
                let dx = aim_solve2(n_j, aim_scale(n_r, aim_c(-1.0)));
                if (!dx.ok) {
                    if (n_fresh) {
                        aim_newton_done();
                        break;
                    }
                    aim_jacobian(AJ_SOLVE_FAILED);
                    return true;
                }
                n_dx = dx.r;
                n_lam = aim_c(1.0);
                n_k = 0u;
                a_state = AN_LINE;
            }
            case AN_LINE: {
                if (n_k == 16u) {
                    // No step accepted.
                    if (n_fresh) {
                        aim_newton_done();
                        break;
                    }
                    aim_jacobian(AJ_REJECTED);
                    return true;
                }
                a_req = aim_add(n_x, aim_scale(n_dx, n_lam));
                a_state = AN_LINE_DONE;
                return true;
            }
            case AN_LINE_DONE: {
                var accepted = false;
                var nn = n_rn;
                if (e.ok) {
                    nn = aim_norm2(e.r);
                    accepted = mkl_is_finite(nn.x)
                        && (aim_lt0(mkl_df_sub(nn, n_rn)) || aim_le0(mkl_df_sub(nn, n_tol)));
                }
                if (!accepted) {
                    n_lam = mkl_df_mul(n_lam, aim_c(0.5));
                    n_k += 1u;
                    a_state = AN_LINE;
                    break;
                }
                let xn = a_req;
                let rr = e.r;
                let s = aim_sub(xn, n_x);
                let ss = mkl_df_add(mkl_df_mul(s.x, s.x), mkl_df_mul(s.y, s.y));
                if (aim_gt0(ss)) {
                    let y = aim_sub(rr, n_r);
                    let js0 = mkl_df_add(mkl_df_mul(n_j.a, s.x), mkl_df_mul(n_j.b, s.y));
                    let js1 = mkl_df_add(mkl_df_mul(n_j.c, s.x), mkl_df_mul(n_j.d, s.y));
                    let e0 = mkl_df_sub(y.x, js0);
                    let e1 = mkl_df_sub(y.y, js1);
                    n_j.a = mkl_df_add(n_j.a, mkl_df_div(mkl_df_mul(e0, s.x), ss));
                    n_j.b = mkl_df_add(n_j.b, mkl_df_div(mkl_df_mul(e0, s.y), ss));
                    n_j.c = mkl_df_add(n_j.c, mkl_df_div(mkl_df_mul(e1, s.x), ss));
                    n_j.d = mkl_df_add(n_j.d, mkl_df_div(mkl_df_mul(e1, s.y), ss));
                }
                let slow = aim_gt0(mkl_df_sub(nn, mkl_df_mul(aim_c(0.5), n_rn)));
                n_x = xn;
                n_r = rr;
                n_rn = nn;
                n_fresh = false;
                if (slow && aim_gt0(mkl_df_sub(n_rn, n_tol))) {
                    aim_jacobian(AJ_SLOW);
                    return true;
                }
                a_state = AN_LOOP;
            }
            case AJ_PLUS: {
                j_plus = e;
                a_req = n_x;
                if (j_col == 0u) {
                    a_req.x = mkl_df_sub(a_req.x, n_h);
                } else {
                    a_req.y = mkl_df_sub(a_req.y, n_h);
                }
                a_state = AJ_MINUS;
                return true;
            }
            case AJ_MINUS: {
                var col: AimV2;
                if (j_plus.ok && e.ok) {
                    col = aim_scale(aim_sub(j_plus.r, e.r), mkl_df_div(aim_c(1.0), mkl_df_mul(aim_c(2.0), n_h)));
                } else if (j_plus.ok) {
                    col = aim_scale(aim_sub(j_plus.r, n_r), mkl_df_div(aim_c(1.0), n_h));
                } else if (e.ok) {
                    col = aim_scale(aim_sub(n_r, e.r), mkl_df_div(aim_c(1.0), n_h));
                } else {
                    aim_jacobian_done(false, n_j);
                    break;
                }
                if (j_col == 0u) {
                    j_first = col;
                    j_col = 1u;
                    a_req = n_x;
                    a_req.y = mkl_df_add(a_req.y, n_h);
                    a_state = AJ_PLUS;
                    return true;
                }
                aim_jacobian_done(true, AimM2(j_first.x, col.x, j_first.y, col.y));
            }
            case AS_TABLE_DONE: {
                if (n_result.converged) {
                    a_state = AS_FINISHED;
                    return false;
                }
                s_it = n_result.iterations;
                if (aim_capped(s_it)) {
                    a_state = AS_FINISHED;
                    return false;
                }
                a_state = AS_PREDICT;
            }
            case AS_PREDICT: {
                var x = aim_scale(s_p, aim_pupil(1u));
                if (s_has_j) {
                    let d = aim_solve2(s_j0, s_goal);
                    if (d.ok) {
                        x = d.r;
                    }
                }
                aim_newton(s_goal, aim_add(s_chief, x), s_j0, s_has_j, aim_budget(s_it), AS_PREDICT_DONE);
                return true;
            }
            case AS_PREDICT_DONE: {
                s_it += n_result.iterations;
                if (n_result.converged || aim_capped(s_it)) {
                    n_result.iterations = s_it;
                    a_state = AS_FINISHED;
                    return false;
                }
                s_xx = s_chief;
                s_jj = s_j0;
                s_hj = s_has_j;
                s_prev = AimV2(aim_c(0.0), aim_c(0.0));
                s_k = 1u;
                a_state = AS_CONTINUE;
            }
            case AS_CONTINUE: {
                s_t = aim_scale(s_goal, aim_c(0.25 * f32(s_k)));
                var start = s_xx;
                if (s_hj) {
                    let d = aim_solve2(s_jj, aim_sub(s_t, s_prev));
                    if (d.ok) {
                        start = aim_add(s_xx, d.r);
                    }
                }
                aim_newton(s_t, start, s_jj, s_hj, aim_budget(s_it), AS_CONTINUE_DONE);
                return true;
            }
            case AS_CONTINUE_DONE: {
                s_it += n_result.iterations;
                if (!n_result.converged || s_k == 4u) {
                    n_result.iterations = s_it;
                    a_state = AS_FINISHED;
                    return false;
                }
                s_xx = n_result.x;
                s_jj = n_result.j;
                s_hj = n_result.has_j;
                s_prev = s_t;
                s_k += 1u;
                a_state = AS_CONTINUE;
            }
            default: {
                return false;
            }
        }
    }
    return false;
}

// Start the real-aim schedule for vignetted pupil point p
// (FieldKernel::aim). True when its first request is in a_req (a table
// seed); otherwise it starts by advancing.
fn aim_begin(seed: u32, table: u32, p: AimV2) -> bool {
    s_chief = AimV2(aim_df(seed + 6u), aim_df(seed + 8u));
    s_j0 = AimM2(aim_df(seed + 10u), aim_df(seed + 12u), aim_df(seed + 14u), aim_df(seed + 16u));
    s_has_j = mkl_in[seed + 22u] != 0u;
    s_goal = aim_scale(p, aim_pupil(2u));
    s_p = p;
    s_it = 0u;
    if (mkl_in[AIM_TABLE] != 0u) {
        let ts = aim_table_seed(table, mkl_in[AIM_SIDE], p, aim_pupil(2u));
        if (ts.ok) {
            aim_newton(s_goal, ts.x, ts.j, true, aim_budget(0u), AS_TABLE_DONE);
            return true;
        }
    }
    a_state = AS_PREDICT;
    return false;
}

// The whole schedule, with the one call site of the trace.
fn aim_point(seed: u32, table: u32, p: AimV2) -> AimSolve {
    var pending = aim_begin(seed, table, p);
    var e: AimEval;
    e.ok = false;
    loop {
        if (pending) {
            e = aim_fun(a_req);
        }
        if (!aim_advance(e)) {
            break;
        }
        pending = true;
    }
    return n_result;
}

fn aim_put(k: u32, v: u32) {
    mkl_out[k * g_stride + g_ray] = v;
}

@compute @workgroup_size(64)
fn mkl_aim_generate(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    let n = mkl_in[AIM_COUNT];
    if (gid.x >= n) {
        return;
    }
    g_ray = gid.x;
    g_stride = n;
    let smp = mkl_in[AIM_SAMPLES_AT] + gid.x * MKL_AIM_SAMPLE_WORDS;
    let field_word = mkl_in[smp + 4u];
    let row = field_word & ~MKL_AIM_SKIP;
    a_field = mkl_in[AIM_FIELDS_AT] + row * MKL_AIM_FIELD_WORDS;
    a_system = mkl_in[a_field + AF_SYSTEM];
    a_wl = mkl_in[a_field + AF_WAVELENGTH];
    let seed = mkl_in[AIM_SEEDS_AT] + row * MKL_AIM_CACHE_WORDS;
    let side = mkl_in[AIM_SIDE];
    let table = mkl_in[AIM_TABLE_AT] + row * side * side * MKL_AIM_NODE_WORDS;
    let seed_status = mkl_in[seed + 20u];
    let stop_fail = MKL_AIM_FAILED | (mkl_headers[a_system].stop << MKL_RAY_SURFACE_SHIFT);
    let zero = vec2<f32>(0.0, 0.0);
    var ray = MklRaw(MklDv(zero, zero, zero), MklDv(zero, zero, zero), MklDv(zero, zero, zero),
        zero, zero, 0.0, 0.0, 0u, 0u, 0u);
    var report = vec4<u32>(0u, 0u, 0u, 0u);
    let hard = seed_status == MKL_AIM_FIELD_UNREACHABLE || seed_status == MKL_AIM_PRIMARY_UNAVAILABLE
        || seed_status == MKL_AIM_INVALID;
    if ((field_word & MKL_AIM_SKIP) != 0u) {
        report.x = MKL_AIM_OUTSIDE;
    } else if (hard) {
        report.x = seed_status;
    } else {
        let real = (aim_flags() & MKL_AIM_REAL) != 0u;
        var p: AimV2;
        p.x = mkl_df_add(aim_fd(AF_VIGNETTE), mkl_df_mul(mkl_df_sub(aim_c(1.0), aim_fd(AF_VIGNETTE + 4u)), aim_df(smp)));
        p.y = mkl_df_add(aim_fd(AF_VIGNETTE + 2u), mkl_df_mul(mkl_df_sub(aim_c(1.0), aim_fd(AF_VIGNETTE + 6u)), aim_df(smp + 2u)));
        a_src = MklDv(aim_df(seed), aim_df(seed + 2u), aim_df(seed + 4u));
        var s: AimSolve;
        if (real) {
            s = aim_point(seed, table, p);
        } else {
            s.x = aim_scale(p, aim_pupil(1u));
            s.residual = zero;
            s.iterations = 0u;
            s.converged = true;
        }
        let st = aim_start(a_src, s.x);
        if (!st.ok) {
            report = vec4<u32>(stop_fail, s.iterations, bitcast<u32>(s.residual.x), bitcast<u32>(s.residual.y));
        } else {
            ray = st.ray;
            ray.weight = bitcast<f32>(mkl_in[smp + 7u]);
            var residual = zero;
            if (real) {
                residual = mkl_df_div(s.residual, aim_pupil(2u));
            }
            report = vec4<u32>(select(stop_fail, 0u, s.converged), s.iterations, bitcast<u32>(residual.x),
                bitcast<u32>(residual.y));
        }
    }
    aim_put(0u, bitcast<u32>(ray.pos.x.x));
    aim_put(1u, bitcast<u32>(ray.pos.y.x));
    aim_put(2u, bitcast<u32>(ray.pos.z.x));
    aim_put(3u, bitcast<u32>(ray.dir.x.x));
    aim_put(4u, bitcast<u32>(ray.dir.y.x));
    aim_put(5u, bitcast<u32>(ray.dir.z.x));
    aim_put(6u, bitcast<u32>(ray.opl.x));
    aim_put(7u, bitcast<u32>(ray.opl.y));
    aim_put(8u, bitcast<u32>(ray.weight));
    aim_put(9u, bitcast<u32>(ray.pos.x.y));
    aim_put(10u, bitcast<u32>(ray.pos.y.y));
    aim_put(11u, bitcast<u32>(ray.pos.z.y));
    aim_put(12u, bitcast<u32>(ray.dir.x.y));
    aim_put(13u, bitcast<u32>(ray.dir.y.y));
    aim_put(14u, bitcast<u32>(ray.dir.z.y));
    aim_put(15u, a_system);
    aim_put(16u, a_wl);
    aim_put(17u, mkl_in[smp + 5u]);
    aim_put(18u, mkl_in[smp + 6u]);
    aim_put(19u, report.x);
    aim_put(20u, report.y);
    aim_put(21u, report.z);
    aim_put(22u, report.w);
}
