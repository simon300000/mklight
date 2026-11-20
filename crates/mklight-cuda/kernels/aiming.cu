// Pupil aiming (aiming::Request::GenerateAim from host chief seeds, with a
// host seed table or none): mkl_aim_generate of kernels/aiming.wgsl, the
// twin of mklight_gpu::aiming::kernel's FieldKernel::{generate, aim, start}
// and of its double-float Newton–Broyden solver, operation for operation.
// Every evaluation traces the stop with trace_shared.h's mkl_trace_raw (no
// records), so generated rays and reports are bitwise the CPU-f32
// emulation's.
//
// kernels.rs compiles this file twice, prefixed with `#define MKL_OVERLAYS
// false` (module "aiming") or `true` (module "aiming_overlays"), like the
// WGSL modules of the same names.
//
// Bindings are the trace's (trace_bindings.h): `in` holds a 16-word header
// (AIM_* below) and the field rows (MklAimField), chief seeds (MklAimCache),
// seed table (MklAimNode) and the launch's samples (MklAimSample) at the
// header's word offsets; `out` receives the ResidentRays planes of the
// launch (15 floats, system, wavelength, 2 logical IDs, 4 report words),
// each `count` long; `steps` is the sequential path 1, 2, …; the uniform
// carries the aiming trace settings (kernel::newton_params, start surface
// 0). The host has validated the field rows and folded culled samples into
// MKL_AIM_SKIP.
//
// The WGSL kernel runs the schedule as a state machine so that its compilers
// inline the trace once; here the solver keeps the specification's
// structure and the one trace call site is a noinline function (aim_eval).
#include "mkl_cuda.h"
#include "trace_bindings.h"

#ifndef MKL_OVERLAYS
#error "aiming.cu is compiled with MKL_OVERLAYS defined (kernels.rs)"
#endif

#define AIM_COUNT 0u
#define AIM_CAP 1u
#define AIM_TOLERANCE 2u
#define AIM_TABLE 3u
#define AIM_SIDE 4u
#define AIM_SCHEDULE_CAP 5u
#define AIM_FIELDS_AT 6u
#define AIM_SEEDS_AT 7u
#define AIM_TABLE_AT 8u
#define AIM_SAMPLES_AT 9u

// A double-float 2-vector and 2x2 matrix [[a, b], [c, d]].
struct AimV2 {
    float2 x, y;
};
struct AimM2 {
    float2 a, b, c, d;
};
struct AimSolve {
    AimV2 x;
    AimM2 j;
    bool has_j;
    float2 residual;
    uint iterations;
    bool converged;
};

static inline float2 aim_c(float x) { return float2(x, 0.0f); }
static inline float2 aim_df(const float* p) { return float2(p[0], p[1]); }
static inline AimV2 aim_v2(float2 x, float2 y) { return {x, y}; }
static inline AimV2 aim_add(AimV2 a, AimV2 b) { return {mkl_df_add(a.x, b.x), mkl_df_add(a.y, b.y)}; }
static inline AimV2 aim_sub(AimV2 a, AimV2 b) { return {mkl_df_sub(a.x, b.x), mkl_df_sub(a.y, b.y)}; }
static inline AimV2 aim_scale(AimV2 a, float2 b) { return {mkl_df_mul(a.x, b), mkl_df_mul(a.y, b)}; }
static inline bool aim_finite(AimV2 a) { return isfinite(a.x.x) && isfinite(a.y.x); }
static inline float2 aim_norm2(AimV2 v) { return mkl_df_sqrt(mkl_df_add(mkl_df_mul(v.x, v.x), mkl_df_mul(v.y, v.y))); }
static inline MklDv aim_dv(float2 x, float2 y, float2 z) { return {{x, y, z}}; }
static inline MklDv aim_unit(MklDv v) {
    const float2 n = mkl_df_sqrt(mkl_ddot(v, v));
    return aim_dv(mkl_df_div(v.v[0], n), mkl_df_div(v.v[1], n), mkl_df_div(v.v[2], n));
}
// Comparisons of high words against zero are false for NaN (IEEE, as in
// the specification).

// The sample's launch context: bindings, field row, chief source and the
// solver settings.
struct AimCtx {
    const MklTraceParams* prm;
    const MklPackedHeader* h;
    const MklPackedSurface* s;
    const float* pool;
    const uint* steps;
    const uint* in;
    const MklAimField* field;
    MklDv src;

    uint flags() const { return field->flags; }
    float2 pupil(uint k) const { return aim_df(field->pupil + 2 * k); }
    float2 rot(uint k) const { return aim_df(field->rotation + 2 * k); }
    float2 origin(uint k) const { return aim_df(field->origin + 2 * k); }
    // Newton step and tolerance (kernel::FieldKernel::{step, tol}).
    float2 step() const { return mkl_df_mul(aim_c(1e-6f), aim_c(fmaxf(pupil(1).x, 1e-6f))); }
    float2 tol() const { return mkl_df_mul(aim_c(as_type<float>(in[AIM_TOLERANCE])), pupil(2)); }
    uint budget(uint spent) const {
        const uint sc = in[AIM_SCHEDULE_CAP], cap = in[AIM_CAP];
        return sc == 0u ? cap : min(cap, sc > spent ? sc - spent : 0u);
    }
    bool capped(uint spent) const {
        const uint sc = in[AIM_SCHEDULE_CAP];
        return sc != 0u && spent >= sc;
    }

    // Object-local to entrance frame: row k of the result is column k of the
    // rotation times p (less the origin for a point).
    MklDv local(MklDv p, bool point) const {
        MklDv a = p;
        if (point)
            a = aim_dv(mkl_df_sub(p.v[0], origin(0)), mkl_df_sub(p.v[1], origin(1)), mkl_df_sub(p.v[2], origin(2)));
        return aim_dv(mkl_ddot(aim_dv(rot(0), rot(3), rot(6)), a), mkl_ddot(aim_dv(rot(1), rot(4), rot(7)), a),
                      mkl_ddot(aim_dv(rot(2), rot(5), rot(8)), a));
    }

    // The launch ray of aim point `a` (FieldKernel::start); false when the
    // direction is undefined.
    bool start(AimV2 a, MklRawState& ray) const {
        MklDv pos, dir;
        if ((flags() & MKL_AIM_FINITE) == 0u) {
            const MklDv ap = aim_dv(a.x, a.y, pupil(0));
            const float2 t = mkl_ddot(ap, src);
            pos = aim_dv(mkl_df_sub(ap.v[0], mkl_df_mul(src.v[0], t)), mkl_df_sub(ap.v[1], mkl_df_mul(src.v[1], t)),
                         mkl_df_sub(ap.v[2], mkl_df_mul(src.v[2], t)));
            dir = src;
        } else if ((flags() & MKL_AIM_TELECENTRIC) != 0u) {
            pos = src;
            dir = local(aim_unit(aim_dv(a.x, a.y, aim_c(1.0f))), false);
        } else {
            const MklDv ap = local(aim_dv(a.x, a.y, pupil(0)), true);
            MklDv d = aim_unit(aim_dv(mkl_df_sub(ap.v[0], src.v[0]), mkl_df_sub(ap.v[1], src.v[1]),
                                      mkl_df_sub(ap.v[2], src.v[2])));
            const float2 z = mkl_ddot(aim_dv(rot(6), rot(7), rot(8)), d);
            if (z.x == 0.0f || !isfinite(z.x) || !mkl_finite3(mkl_hi3(d))) return false;
            if (z.x < 0.0f)
                for (uint k = 0; k < 3; ++k) d.v[k] = mkl_df_neg(d.v[k]);
            pos = src;
            dir = d;
        }
        ray = {};
        ray.pos = pos;
        ray.dir = dir;
        ray.weight = 1.0f;
        ray.transmittance = 1.0f;
        return true;
    }
};

// Stop residual of aim point `a` towards `target` (the Newton function):
// the one call site of the trace.
static __noinline__ bool aim_eval(const AimCtx& c, AimV2 target, AimV2 a, AimV2& r) {
    MklRawState ray;
    if (!c.start(a, ray)) return false;
    MklNoRecords none;
    const MklRawState t = mkl_trace_raw<MKL_OVERLAYS>(*c.h, c.s, c.pool, c.steps, *c.prm, c.h->stop, c.field->wavelength,
                                                       ray, none);
    if (t.status != 0u) return false;
    r = aim_v2(mkl_df_sub(t.pos.v[0], target.x), mkl_df_sub(t.pos.v[1], target.y));
    return true;
}

static inline bool aim_solve2(AimM2 j, AimV2 r, AimV2& out) {
    const float2 det = mkl_df_sub(mkl_df_mul(j.a, j.d), mkl_df_mul(j.b, j.c));
    float s = 0.0f;
    s = fmaxf(s, fabsf(j.a.x));
    s = fmaxf(s, fabsf(j.b.x));
    s = fmaxf(s, fabsf(j.c.x));
    s = fmaxf(s, fabsf(j.d.x));
    if (!isfinite(det.x) || fabsf(det.x) <= 1e-14f * s * s) return false;
    out = aim_v2(mkl_df_div(mkl_df_sub(mkl_df_mul(r.x, j.d), mkl_df_mul(j.b, r.y)), det),
                 mkl_df_div(mkl_df_sub(mkl_df_mul(j.a, r.y), mkl_df_mul(r.x, j.c)), det));
    return aim_finite(out);
}

// Central differences, one-sided where one side fails (kernel::jacobian);
// `j` is written only on success.
static __noinline__ bool aim_jacobian(const AimCtx& c, AimV2 target, AimV2 x, AimV2 fx, float2 h, AimM2& j) {
    AimV2 cols[2];
    for (uint k = 0; k < 2; ++k) {
        AimV2 xp = x, xm = x, a, b;
        if (k == 0) {
            xp.x = mkl_df_add(xp.x, h);
            xm.x = mkl_df_sub(xm.x, h);
        } else {
            xp.y = mkl_df_add(xp.y, h);
            xm.y = mkl_df_sub(xm.y, h);
        }
        const bool ap = aim_eval(c, target, xp, a);
        const bool bp = aim_eval(c, target, xm, b);
        if (ap && bp)
            cols[k] = aim_scale(aim_sub(a, b), mkl_df_div(aim_c(1.0f), mkl_df_mul(aim_c(2.0f), h)));
        else if (ap)
            cols[k] = aim_scale(aim_sub(a, fx), mkl_df_div(aim_c(1.0f), h));
        else if (bp)
            cols[k] = aim_scale(aim_sub(fx, b), mkl_df_div(aim_c(1.0f), h));
        else
            return false;
    }
    j = {cols[0].x, cols[1].x, cols[0].y, cols[1].y};
    return true;
}

// Newton–Broyden with backtracking and Jacobian refresh (kernel::newton).
static __noinline__ AimSolve aim_newton(const AimCtx& c, AimV2 target, AimV2 x0, AimM2 j0, bool has_j0, uint cap) {
    const float2 h = c.step(), tol = c.tol();
    AimSolve out = {x0, j0, has_j0, aim_c(INFINITY), 0u, false};
    AimV2 r;
    if (!aim_eval(c, target, x0, r)) return out;
    AimV2 x = x0;
    float2 rn = aim_norm2(r);
    AimM2 j = j0;
    bool fresh = false;
    if (!has_j0) {
        if (!aim_jacobian(c, target, x, r, h, j)) {
            out.residual = rn;
            return out;
        }
        fresh = true;
    }
    uint it = 0;
    while (it < cap) {
        if (mkl_df_sub(rn, tol).x <= 0.0f) break;
        ++it;
        AimV2 dx;
        if (!aim_solve2(j, aim_scale(r, aim_c(-1.0f)), dx)) {
            if (fresh) break;
            if (!aim_jacobian(c, target, x, r, h, j)) break;
            fresh = true;
            continue;
        }
        float2 lam = aim_c(1.0f);
        bool accepted = false;
        AimV2 xn, rr;
        float2 nn = rn;
        for (uint k = 0; k < 16; ++k) {
            xn = aim_add(x, aim_scale(dx, lam));
            if (aim_eval(c, target, xn, rr)) {
                nn = aim_norm2(rr);
                if (isfinite(nn.x) && (mkl_df_sub(nn, rn).x < 0.0f || mkl_df_sub(nn, tol).x <= 0.0f)) {
                    accepted = true;
                    break;
                }
            }
            lam = mkl_df_mul(lam, aim_c(0.5f));
        }
        if (!accepted) {
            if (fresh) break;
            if (!aim_jacobian(c, target, x, r, h, j)) break;
            fresh = true;
            continue;
        }
        const AimV2 s = aim_sub(xn, x);
        const float2 ss = mkl_df_add(mkl_df_mul(s.x, s.x), mkl_df_mul(s.y, s.y));
        if (ss.x > 0.0f) {
            const AimV2 y = aim_sub(rr, r);
            const float2 js0 = mkl_df_add(mkl_df_mul(j.a, s.x), mkl_df_mul(j.b, s.y));
            const float2 js1 = mkl_df_add(mkl_df_mul(j.c, s.x), mkl_df_mul(j.d, s.y));
            const float2 e0 = mkl_df_sub(y.x, js0), e1 = mkl_df_sub(y.y, js1);
            j.a = mkl_df_add(j.a, mkl_df_div(mkl_df_mul(e0, s.x), ss));
            j.b = mkl_df_add(j.b, mkl_df_div(mkl_df_mul(e0, s.y), ss));
            j.c = mkl_df_add(j.c, mkl_df_div(mkl_df_mul(e1, s.x), ss));
            j.d = mkl_df_add(j.d, mkl_df_div(mkl_df_mul(e1, s.y), ss));
        }
        const bool slow = mkl_df_sub(nn, mkl_df_mul(aim_c(0.5f), rn)).x > 0.0f;
        x = xn;
        r = rr;
        rn = nn;
        fresh = false;
        if (slow && mkl_df_sub(rn, tol).x > 0.0f && aim_jacobian(c, target, x, r, h, j)) fresh = true;
    }
    return {x, j, true, rn, it, mkl_df_sub(rn, tol).x <= 0.0f};
}

// Catmull–Rom weights at t (x) and their derivatives (y).
static inline void aim_spline(float t, float2 w[4]) {
    const float t2 = t * t, t3 = t2 * t;
    w[0] = float2(((t2 * 2.0f - t3) - t) * 0.5f, ((t * 4.0f - t2 * 3.0f) - 1.0f) * 0.5f);
    w[1] = float2(((t3 * 3.0f - t2 * 5.0f) + 2.0f) * 0.5f, (t2 * 9.0f - t * 10.0f) * 0.5f);
    w[2] = float2(((t2 * 4.0f - t3 * 3.0f) + t) * 0.5f, ((t * 8.0f - t2 * 9.0f) + 1.0f) * 0.5f);
    w[3] = float2((t3 - t2) * 0.5f, (t2 * 3.0f - t * 2.0f) * 0.5f);
}

// Bicubic table seed at p and its stop Jacobian (kernel::table_seed); the
// field's table starts at `table`.
static inline bool aim_table_seed(const MklAimNode* table, uint side, AimV2 p, float2 r_stop, AimV2& x, AimM2& j) {
    const float step = 2.25f / float(side - 1u);
    const float fx = (p.x.x + 1.125f) / step;
    const float fy = (p.y.x + 1.125f) / step;
    if (isnan(fx) || isnan(fy)) return false;
    const float ix = floorf(fx), iy = floorf(fy), last = float(side - 3u);
    if (!(ix >= 1.0f && iy >= 1.0f && ix <= last && iy <= last)) return false;
    float2 w[4], v[4];
    aim_spline(fx - ix, w);
    aim_spline(fy - iy, v);
    AimV2 a = aim_v2(aim_c(0.0f), aim_c(0.0f)), ax = a, ay = a;
    for (uint jj = 0; jj < 4; ++jj)
        for (uint ii = 0; ii < 4; ++ii) {
            const MklAimNode& n = table[(uint(iy) + jj - 1u) * side + uint(ix) + ii - 1u];
            if (n.status != 0u) return false;
            const float wv = w[ii].x * v[jj].x, dxv = w[ii].y * v[jj].x, dyv = w[ii].x * v[jj].y;
            const float2 n0 = aim_df(n.aim), n1 = aim_df(n.aim + 2);
            a = aim_v2(mkl_df_add(a.x, mkl_df_mul_f32(n0, wv)), mkl_df_add(a.y, mkl_df_mul_f32(n1, wv)));
            ax = aim_v2(mkl_df_add(ax.x, mkl_df_mul_f32(n0, dxv)), mkl_df_add(ax.y, mkl_df_mul_f32(n1, dxv)));
            ay = aim_v2(mkl_df_add(ay.x, mkl_df_mul_f32(n0, dyv)), mkl_df_add(ay.y, mkl_df_mul_f32(n1, dyv)));
        }
    const float2 det = mkl_df_sub(mkl_df_mul(ax.x, ay.y), mkl_df_mul(ay.x, ax.y));
    const float2 q = mkl_df_div(r_stop, mkl_df_mul_f32(det, float(side - 1u) / 2.25f));
    x = a;
    j = {mkl_df_mul(ay.y, q), mkl_df_neg(mkl_df_mul(ay.x, q)), mkl_df_neg(mkl_df_mul(ax.y, q)), mkl_df_mul(ax.x, q)};
    return isfinite(j.a.x) && isfinite(j.b.x) && isfinite(j.c.x) && isfinite(j.d.x) && det.x != 0.0f;
}

// Real-aim schedule for vignetted pupil point p (FieldKernel::aim): table
// seed, prediction from the chief, then continuation in the pupil radius.
static inline AimSolve aim_point(const AimCtx& c, const MklAimCache& seed, const MklAimNode* table, AimV2 p) {
    const AimV2 chief = aim_v2(aim_df(seed.chief), aim_df(seed.chief + 2));
    const AimM2 j0 = {aim_df(seed.jacobian), aim_df(seed.jacobian + 2), aim_df(seed.jacobian + 4),
                      aim_df(seed.jacobian + 6)};
    const bool has_j0 = seed.has_jacobian != 0u;
    const AimV2 target = aim_scale(p, c.pupil(2));
    uint it = 0;
    AimV2 tx;
    AimM2 tj;
    if (table && aim_table_seed(table, c.in[AIM_SIDE], p, c.pupil(2), tx, tj)) {
        const AimSolve s = aim_newton(c, target, tx, tj, true, c.budget(it));
        if (s.converged) return s;
        it = s.iterations;
        if (c.capped(it)) return s;
    }
    AimV2 x = aim_scale(p, c.pupil(1)), d;
    if (has_j0 && aim_solve2(j0, target, d)) x = d;
    AimSolve s = aim_newton(c, target, aim_add(chief, x), j0, has_j0, c.budget(it));
    it += s.iterations;
    if (!s.converged && !c.capped(it)) {
        AimV2 xx = chief, prev = aim_v2(aim_c(0.0f), aim_c(0.0f));
        AimM2 jj = j0;
        bool hj = has_j0;
        for (uint k = 1; k <= 4; ++k) {
            const AimV2 t = aim_scale(target, aim_c(0.25f * float(k)));
            AimV2 start = xx;
            if (hj && aim_solve2(jj, aim_sub(t, prev), d)) start = aim_add(xx, d);
            s = aim_newton(c, t, start, jj, hj, c.budget(it));
            it += s.iterations;
            if (!s.converged) break;
            xx = s.x;
            jj = s.j;
            hj = s.has_j;
            prev = t;
        }
    }
    s.iterations = it;
    return s;
}

// One thread per sample (FieldKernel::generate).
MKL_KERNEL mkl_aim_generate(MKL_TRACE_ARGS) {
    const uint i = mkl_gid_x();
    const uint n = in[AIM_COUNT];
    if (i >= n) return;
    const MklAimSample& smp =
        *reinterpret_cast<const MklAimSample*>(in + in[AIM_SAMPLES_AT] + i * MKL_AIM_SAMPLE_WORDS);
    const uint row = smp.field & ~MKL_AIM_SKIP;
    const MklAimField& field = *reinterpret_cast<const MklAimField*>(in + in[AIM_FIELDS_AT] + row * MKL_AIM_FIELD_WORDS);
    const MklAimCache& seed = *reinterpret_cast<const MklAimCache*>(in + in[AIM_SEEDS_AT] + row * MKL_AIM_CACHE_WORDS);
    const uint side = in[AIM_SIDE];
    const MklAimNode* table =
        in[AIM_TABLE] != 0u
            ? reinterpret_cast<const MklAimNode*>(in + in[AIM_TABLE_AT] + row * side * side * MKL_AIM_NODE_WORDS)
            : nullptr;
    const MklPackedHeader& h = headers[field.system];
    const uint stop_fail = MKL_AIM_FAILED | (h.stop << MKL_RAY_SURFACE_SHIFT);
    const bool hard = seed.status == MKL_AIM_FIELD_UNREACHABLE || seed.status == MKL_AIM_PRIMARY_UNAVAILABLE ||
                      seed.status == MKL_AIM_INVALID;
    MklRawState ray = {};
    uint report[4] = {0u, 0u, 0u, 0u};
    if ((smp.field & MKL_AIM_SKIP) != 0u) {
        report[0] = MKL_AIM_OUTSIDE;
    } else if (hard) {
        report[0] = seed.status;
    } else {
        const AimCtx c = {&u->trace,
                          &h,
                          surfaces + h.surface_offset,
                          pool,
                          steps,
                          in,
                          &field,
                          aim_dv(aim_df(seed.source), aim_df(seed.source + 2), aim_df(seed.source + 4))};
        const bool real = (field.flags & MKL_AIM_REAL) != 0u;
        AimV2 p;
        p.x = mkl_df_add(aim_df(field.vignette),
                         mkl_df_mul(mkl_df_sub(aim_c(1.0f), aim_df(field.vignette + 4)), aim_df(smp.pupil)));
        p.y = mkl_df_add(aim_df(field.vignette + 2),
                         mkl_df_mul(mkl_df_sub(aim_c(1.0f), aim_df(field.vignette + 6)), aim_df(smp.pupil + 2)));
        AimSolve s;
        if (real) {
            s = aim_point(c, seed, table, p);
        } else {
            s.x = aim_scale(p, c.pupil(1));
            s.residual = aim_c(0.0f);
            s.iterations = 0u;
            s.converged = true;
        }
        if (!c.start(s.x, ray)) {
            ray = {};
            report[0] = stop_fail;
            report[2] = as_type<uint>(s.residual.x);
            report[3] = as_type<uint>(s.residual.y);
        } else {
            ray.weight = smp.weight;
            const float2 residual = real ? mkl_df_div(s.residual, c.pupil(2)) : aim_c(0.0f);
            report[0] = s.converged ? 0u : stop_fail;
            report[2] = as_type<uint>(residual.x);
            report[3] = as_type<uint>(residual.y);
        }
        report[1] = s.iterations;
    }
    float* out_f = reinterpret_cast<float*>(out);
    for (uint k = 0; k < 3; ++k) {
        out_f[mkl_plane(k, n, i)] = ray.pos.v[k].x;
        out_f[mkl_plane(3 + k, n, i)] = ray.dir.v[k].x;
        out_f[mkl_plane(9 + k, n, i)] = ray.pos.v[k].y;
        out_f[mkl_plane(12 + k, n, i)] = ray.dir.v[k].y;
    }
    out_f[mkl_plane(6, n, i)] = ray.opl.x;
    out_f[mkl_plane(7, n, i)] = ray.opl.y;
    out_f[mkl_plane(8, n, i)] = ray.weight;
    out[mkl_plane(15, n, i)] = field.system;
    out[mkl_plane(16, n, i)] = field.wavelength;
    out[mkl_plane(17, n, i)] = smp.id_lo;
    out[mkl_plane(18, n, i)] = smp.id_hi;
    for (uint k = 0; k < 4; ++k) out[mkl_plane(19 + k, n, i)] = report[k];
}
