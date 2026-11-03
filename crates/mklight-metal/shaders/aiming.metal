// Resident generation, aiming, trace and pupil projection: the MSL twin of
// aiming/kernel.rs (operation for operation; no fast math). Kernels taking
// systems come in two variants, `<name>` and `<name>_overlays`, so systems
// without sag overlays do not carry the overlay code. noinline boundaries keep
// nested Newton objectives from duplicating the whole trace program.
#include "trace_shared.h"

static inline MklDf mkl_aim_c(float x) { return mkl_c<MklDf>(x); }
static inline MklDf mkl_aim_load(constant float* x, uint k) { return MklDf{mkl_df_input(float2(x[2 * k], x[2 * k + 1]))}; }
static inline MklDf mkl_aim_load(thread const float* x, uint k) { return MklDf{mkl_df_input(float2(x[2 * k], x[2 * k + 1]))}; }
static inline void mkl_aim_store(thread float* p, uint k, MklDf x) {
    p[2 * k] = x.v.x;
    p[2 * k + 1] = x.v.y;
}
static inline MklDf mkl_aim_scale(MklDf a, float w) { return MklDf{mkl_df_mul_f32(a.v, w)}; }

struct MklAimVec2 {
    MklDf v[2];
};
static inline MklAimVec2 mkl_aim_vec2(MklDf x, MklDf y) { return {{x, y}}; }
static inline MklAimVec2 mkl_aim_zero() { return mkl_aim_vec2(mkl_aim_c(0), mkl_aim_c(0)); }
static inline MklAimVec2 operator+(MklAimVec2 a, MklAimVec2 b) { return mkl_aim_vec2(a.v[0] + b.v[0], a.v[1] + b.v[1]); }
static inline MklAimVec2 operator-(MklAimVec2 a, MklAimVec2 b) { return mkl_aim_vec2(a.v[0] - b.v[0], a.v[1] - b.v[1]); }
static inline MklAimVec2 operator*(MklAimVec2 a, MklDf b) { return mkl_aim_vec2(a.v[0] * b, a.v[1] * b); }
static inline MklDf mkl_aim_norm(MklAimVec2 a) { return mkl_sqrt(a.v[0] * a.v[0] + a.v[1] * a.v[1]); }
static inline MklDv mkl_aim_vec3(MklDf x, MklDf y, MklDf z) { return {{x.v, y.v, z.v}}; }
static inline MklDv mkl_aim_unit(MklDv p) {
    MklDf n = mkl_sqrt(MklDf{mkl_ddot(p, p)});
    for (uint k = 0; k < 3; ++k) p.v[k] = (MklDf{p.v[k]} / n).v;
    return p;
}
struct MklAimMat2 {
    MklDf v[2][2];
};
static inline bool mkl_aim_solve2(MklAimMat2 j, MklAimVec2 r, thread MklAimVec2& out) {
    MklDf det = j.v[0][0] * j.v[1][1] - j.v[0][1] * j.v[1][0];
    float s = 0;
    for (uint row = 0; row < 2; ++row)
        for (uint col = 0; col < 2; ++col) s = max(s, fabs(j.v[row][col].v.x));
    if (!isfinite(det.v.x) || fabs(det.v.x) <= 1e-14f * s * s) return false;
    out = mkl_aim_vec2((r.v[0] * j.v[1][1] - j.v[0][1] * r.v[1]) / det,
                       (j.v[0][0] * r.v[1] - r.v[0] * j.v[1][0]) / det);
    return isfinite(out.v[0].v.x) && isfinite(out.v[1].v.x);
}

// Central differences, one-sided where one side fails (numeric::fd_jacobian).
template <typename Fun>
static __attribute__((noinline)) bool mkl_aim_jacobian(thread Fun& fun, MklAimVec2 x, MklAimVec2 fx,
                                                       MklDf h, thread MklAimMat2& j) {
    for (uint k = 0; k < 2; ++k) {
        MklAimVec2 xp = x, xm = x, a, b, col;
        xp.v[k] = xp.v[k] + h;
        xm.v[k] = xm.v[k] - h;
        bool ap = fun.eval(xp, a), bp = fun.eval(xm, b);
        if (ap && bp)
            col = (a - b) * (mkl_aim_c(1) / (mkl_aim_c(2) * h));
        else if (ap)
            col = (a - fx) * (mkl_aim_c(1) / h);
        else if (bp)
            col = (fx - b) * (mkl_aim_c(1) / h);
        else
            return false;
        j.v[0][k] = col.v[0];
        j.v[1][k] = col.v[1];
    }
    return true;
}
struct MklAimSolve {
    MklAimVec2 x;
    MklAimMat2 j;
    MklDf residual;
    uint iterations;
    bool converged, has_j;
};
// Newton–Broyden with backtracking and Jacobian refresh (numeric::newton2).
template <typename Fun>
static inline MklAimSolve mkl_aim_newton(Fun fun, MklAimVec2 x0, MklAimMat2 j0, bool has_j, MklDf h,
                                         MklDf tol, uint cap) {
    MklAimSolve out = {x0, j0, mkl_aim_c(INFINITY), 0, false, has_j};
    MklAimVec2 r;
    if (!fun.eval(x0, r)) return out;
    MklAimVec2 x = x0;
    MklDf rn = mkl_aim_norm(r);
    MklAimMat2 j = j0;
    bool fresh = false;
    if (!has_j) {
        if (!mkl_aim_jacobian(fun, x, r, h, j)) {
            out.residual = rn;
            return out;
        }
        fresh = true;
    }
    uint it = 0;
    while (it < cap) {
        if ((rn - tol).v.x <= 0) break;
        ++it;
        MklAimVec2 dx;
        if (!mkl_aim_solve2(j, r * mkl_aim_c(-1), dx)) {
            if (fresh) break;
            if (mkl_aim_jacobian(fun, x, r, h, j)) {
                fresh = true;
                continue;
            }
            break;
        }
        MklDf lam = mkl_aim_c(1);
        bool accepted = false;
        MklAimVec2 xn, rr;
        MklDf nn = mkl_aim_c(0);
        for (uint k = 0; k < 16; ++k) {
            xn = x + dx * lam;
            if (fun.eval(xn, rr)) {
                nn = mkl_aim_norm(rr);
                if (isfinite(nn.v.x) && ((nn - rn).v.x < 0 || (nn - tol).v.x <= 0)) {
                    accepted = true;
                    break;
                }
            }
            lam = lam * mkl_aim_c(0.5f);
        }
        if (accepted) {
            MklAimVec2 s = xn - x;
            MklDf ss = s.v[0] * s.v[0] + s.v[1] * s.v[1];
            if (ss.v.x > 0) {
                MklAimVec2 y = rr - r, js = mkl_aim_vec2(j.v[0][0] * s.v[0] + j.v[0][1] * s.v[1],
                                                         j.v[1][0] * s.v[0] + j.v[1][1] * s.v[1]);
                for (uint row = 0; row < 2; ++row)
                    for (uint col = 0; col < 2; ++col)
                        j.v[row][col] = j.v[row][col] + (y.v[row] - js.v[row]) * s.v[col] / ss;
            }
            bool slow = (nn - mkl_aim_c(0.5f) * rn).v.x > 0;
            x = xn;
            r = rr;
            rn = nn;
            fresh = false;
            if (slow && (rn - tol).v.x > 0) {
                MklAimMat2 jn;
                if (mkl_aim_jacobian(fun, x, r, h, jn)) {
                    j = jn;
                    fresh = true;
                }
            }
        } else {
            if (fresh) break;
            if (mkl_aim_jacobian(fun, x, r, h, j))
                fresh = true;
            else
                break;
        }
    }
    return {x, j, rn, it, (rn - tol).v.x <= 0, true};
}

// Immutable optical data of one field row (kernel::FieldKernel).
template <bool OV>
struct MklAimFieldKernel {
    constant MklPackedHeader* h;
    constant MklPackedSurface* s;
    constant float* pool;
    constant MklAimField* field;
    constant MklTraceParams* newton;
    float tolerance;
    uint cap;
    bool finite() const { return (field->flags & MKL_AIM_FINITE) != 0; }
    bool real() const { return (field->flags & MKL_AIM_REAL) != 0; }
    bool tele() const { return (field->flags & MKL_AIM_TELECENTRIC) != 0; }
    MklDf pupil(uint k) const { return mkl_aim_load(field->pupil, k); }
    MklDf step() const { return mkl_aim_c(1e-6f) * mkl_aim_c(max(pupil(1).v.x, 1e-6f)); }
    MklDf tol() const { return mkl_aim_c(tolerance) * pupil(2); }
    MklDv local(MklDv p, bool point) const {
        MklDv a = p;
        if (point)
            for (uint k = 0; k < 3; ++k) a.v[k] = (MklDf{p.v[k]} - mkl_aim_load(field->origin, k)).v;
        MklDv out;
        for (uint k = 0; k < 3; ++k) {
            MklDv r;
            for (uint j = 0; j < 3; ++j) r.v[j] = mkl_aim_load(field->rotation, j * 3 + k).v;
            out.v[k] = mkl_ddot(r, a);
        }
        return out;
    }
    bool object(MklAimVec2 p, thread MklDv& out) const {
        // The reference evaluates extended object sags on +y at the radius.
        MklDv at = s[0].shape > MKL_SHAPE_COORD_BREAK ? mkl_aim_vec3(mkl_aim_c(0), mkl_aim_norm(p), mkl_aim_c(0))
                                                      : mkl_aim_vec3(p.v[0], p.v[1], mkl_aim_c(0));
        MklHit hit;
        uint code = mkl_intersect<OV>(s[0], pool, at, mkl_aim_vec3(mkl_aim_c(0), mkl_aim_c(0), mkl_aim_c(1)),
                                      *newton, hit);
        out = mkl_aim_vec3(p.v[0], p.v[1], MklDf{hit.x.v[2]});
        return code == 0;
    }
    bool source(MklAimVec2 p, thread MklDv& out) const {
        if (finite()) return object(p, out);
        out = mkl_aim_unit(mkl_aim_vec3(p.v[0], p.v[1], mkl_aim_c(1)));
        return true;
    }
    bool initial(thread MklDv& out) const {
        MklAimVec2 p = mkl_aim_vec2(mkl_aim_load(field->source, 0), mkl_aim_load(field->source, 1));
        if (finite() && (field->flags & MKL_AIM_ANGLE) != 0) {
            if (tele()) return false;
            MklDv dir = local(mkl_aim_unit(mkl_aim_vec3(p.v[0], p.v[1], mkl_aim_c(1))), false),
                  point = local(mkl_aim_vec3(mkl_aim_c(0), mkl_aim_c(0), pupil(0)), true);
            MklHit hit;
            uint code = mkl_intersect<OV>(s[0], pool, point, dir, *newton, hit);
            out = hit.x;
            return code == 0;
        }
        return source(p, out);
    }
    MklRawState start(MklDv src, MklAimVec2 a) const {
        MklRawState out = {};
        out.weight = 1;
        if (!finite()) {
            MklDv ap = mkl_aim_vec3(a.v[0], a.v[1], pupil(0));
            MklDf t = MklDf{mkl_ddot(ap, src)};
            for (uint k = 0; k < 3; ++k) out.pos.v[k] = (MklDf{ap.v[k]} - MklDf{src.v[k]} * t).v;
            out.dir = src;
        } else if (tele()) {
            out.pos = src;
            out.dir = local(mkl_aim_unit(mkl_aim_vec3(a.v[0], a.v[1], mkl_aim_c(1))), false);
        } else {
            MklDv ap = local(mkl_aim_vec3(a.v[0], a.v[1], pupil(0)), true), d;
            for (uint k = 0; k < 3; ++k) d.v[k] = (MklDf{ap.v[k]} - MklDf{src.v[k]}).v;
            d = mkl_aim_unit(d);
            MklDv rot = mkl_aim_vec3(mkl_aim_load(field->rotation, 6), mkl_aim_load(field->rotation, 7),
                                     mkl_aim_load(field->rotation, 8));
            float z = mkl_ddot(rot, d).x;
            if (z == 0 || !isfinite(z) || !mkl_finite3(mkl_hi3(d))) {
                MklRawState failed = {};
                failed.status = MKL_AIM_FAILED | (h->stop << MKL_RAY_SURFACE_SHIFT);
                return failed;
            }
            if (z < 0)
                for (uint k = 0; k < 3; ++k) d.v[k] = mkl_df_neg(d.v[k]);
            out.pos = src;
            out.dir = d;
        }
        return out;
    }
    __attribute__((noinline)) bool trace(MklDv src, MklAimVec2 a, uint last, uint wl,
                                         thread MklRawState& out) const {
        MklNoRecords rec;
        MklRawState input = start(src, a);
        if (input.status != 0) return false;
        out = mkl_trace_raw<OV>(*h, s, pool, nullptr, *newton, last, wl, input, rec);
        return out.status == 0;
    }
};
template <bool OV>
struct MklAimObjective {
    MklAimFieldKernel<OV> f;
    MklDv src;
    uint wl;
    MklAimVec2 target;
    bool eval(MklAimVec2 a, thread MklAimVec2& out) {
        MklRawState r;
        if (!f.trace(src, a, f.h->stop, wl, r)) return false;
        out = mkl_aim_vec2(MklDf{r.pos.v[0]} - target.v[0], MklDf{r.pos.v[1]} - target.v[1]);
        return true;
    }
};
template <bool OV>
static __attribute__((noinline)) MklAimSolve mkl_aim_solve(MklAimFieldKernel<OV> f, MklDv src, uint wl,
                                                           MklAimVec2 target, MklAimVec2 x, MklAimMat2 j,
                                                           bool has_j) {
    return mkl_aim_newton(MklAimObjective<OV>{f, src, wl, target}, x, j, has_j, f.step(), f.tol(), f.cap);
}
template <bool OV>
static __attribute__((noinline)) MklAimSolve mkl_aim_chief(MklAimFieldKernel<OV> f, MklDv src, uint wl) {
    MklAimMat2 zero = {};
    if (!f.real()) return {mkl_aim_zero(), zero, mkl_aim_c(0), 0, true, false};
    MklAimSolve s = mkl_aim_solve(f, src, wl, mkl_aim_zero(), mkl_aim_zero(), zero, false);
    if (!s.converged) {
        uint n = f.finite() ? 16 : 64;
        MklAimVec2 x = mkl_aim_zero();
        MklAimMat2 j = {};
        bool has_j = false, has_entrance = false;
        MklDv entrance = {};
        for (uint k = 0; k <= n; ++k) {
            MklDf frac = mkl_aim_c(float(k)) / mkl_aim_c(float(n));
            MklDv step;
            if (k == n)
                step = src;
            else if (f.finite()) {
                if (!f.source(mkl_aim_vec2(MklDf{src.v[0]} * frac, MklDf{src.v[1]} * frac), step)) break;
            } else
                step = mkl_aim_unit(mkl_aim_vec3(MklDf{src.v[0]} * frac, MklDf{src.v[1]} * frac,
                                                 mkl_aim_c(1) - frac + MklDf{src.v[2]} * frac));
            if (has_entrance) {
                MklDf t = (f.pupil(0) - MklDf{entrance.v[2]}) / MklDf{step.v[2]};
                x = mkl_aim_vec2(MklDf{entrance.v[0]} + t * MklDf{step.v[0]},
                                 MklDf{entrance.v[1]} + t * MklDf{step.v[1]});
            }
            MklAimSolve sn = mkl_aim_solve(f, step, wl, mkl_aim_zero(), x, j, has_j);
            if (!sn.converged) break;
            x = sn.x;
            j = sn.j;
            has_j = sn.has_j;
            if (k == n)
                s = sn;
            else if (!f.finite()) {
                MklRawState r;
                if (f.trace(step, x, 1, wl, r)) {
                    entrance = r.pos;
                    MklDv d = {};
                    mkl_xf_backward(f.s[0], entrance, d);
                    has_entrance = true;
                }
            }
        }
    }
    MklAimObjective<OV> fun = {f, src, wl, mkl_aim_zero()};
    MklAimVec2 fx;
    MklAimMat2 j;
    if (fun.eval(s.x, fx) && mkl_aim_jacobian(fun, s.x, fx, f.step(), j)) {
        s.j = j;
        s.has_j = true;
    }
    return s;
}
template <bool OV>
struct MklImageHeightObjective {
    MklAimFieldKernel<OV> f;
    MklAimVec2 target;
    bool eval(MklAimVec2 p, thread MklAimVec2& out) {
        MklDv src;
        if (!f.source(p, src)) return false;
        MklAimSolve aim = mkl_aim_chief(f, src, f.h->primary_wavelength);
        if (!aim.converged) return false;
        MklRawState r;
        if (!f.trace(src, aim.x, f.h->num_surfaces - 1, f.h->primary_wavelength, r)) return false;
        out = mkl_aim_vec2(MklDf{r.pos.v[0]} - target.v[0], MklDf{r.pos.v[1]} - target.v[1]);
        return true;
    }
};
template <bool OV>
static inline MklAimCache mkl_aim_prepare(MklAimFieldKernel<OV> f) {
    MklAimCache out = {};
    out.status = MKL_AIM_FIELD_UNREACHABLE;
    if ((f.field->flags & MKL_AIM_IMAGE_HEIGHT) != 0 && f.h->primary_wavelength >= f.h->num_wavelengths) {
        out.status = MKL_AIM_PRIMARY_UNAVAILABLE;
        return out;
    }
    MklDv src;
    if (!f.initial(src)) return out;
    MklAimVec2 target = mkl_aim_vec2(mkl_aim_load(f.field->target, 0), mkl_aim_load(f.field->target, 1));
    if ((f.field->flags & MKL_AIM_IMAGE_HEIGHT) != 0 && (target.v[0].v.x != 0 || target.v[1].v.x != 0)) {
        MklAimVec2 p = f.finite() ? mkl_aim_vec2(MklDf{src.v[0]}, MklDf{src.v[1]})
                                  : mkl_aim_vec2(MklDf{src.v[0]} / MklDf{src.v[2]}, MklDf{src.v[1]} / MklDf{src.v[2]});
        MklDf h = mkl_aim_c(1e-6f) * mkl_aim_c(max(mkl_aim_norm(p).v.x, 1e-6f)),
              tol = mkl_aim_c(1e-12f) * mkl_aim_c(max(mkl_aim_norm(target).v.x, 1e-3f));
        MklAimMat2 j = {};
        MklAimSolve result = mkl_aim_newton(MklImageHeightObjective<OV>{f, target}, p, j, false, h, tol, 60);
        if (!result.converged || !f.source(result.x, src)) return out;
    }
    MklAimSolve s = mkl_aim_chief(f, src, f.field->wavelength);
    for (uint k = 0; k < 3; ++k) mkl_aim_store(out.source, k, MklDf{src.v[k]});
    for (uint k = 0; k < 2; ++k) mkl_aim_store(out.chief, k, s.x.v[k]);
    for (uint k = 0; k < 4; ++k) mkl_aim_store(out.jacobian, k, s.has_j ? s.j.v[k / 2][k % 2] : mkl_aim_c(0));
    mkl_aim_store(out.residual, 0, s.residual);
    out.status = s.converged ? 0 : (MKL_AIM_FAILED | (f.h->stop << MKL_RAY_SURFACE_SHIFT));
    out.iterations = s.iterations;
    out.has_jacobian = s.has_j;
    return out;
}

static inline bool mkl_aim_hard_failure(uint status) {
    return status == MKL_AIM_FIELD_UNREACHABLE || status == MKL_AIM_PRIMARY_UNAVAILABLE ||
           status == MKL_AIM_INVALID;
}

#define MKL_AIM_TABLE_EXTENT 1.125f
// Node spacing of a `side`-node table over [-1.125, 1.125] (kernel::table_step):
// one rounded f32 division, so every table side matches the specification.
static inline float mkl_aim_table_step(uint side) { return (2.0f * MKL_AIM_TABLE_EXTENT) / float(side - 1); }
static inline float mkl_aim_table_inv_step(uint side) { return float(side - 1) / (2.0f * MKL_AIM_TABLE_EXTENT); }

// Catmull–Rom weights at t in [0, 1) and their derivatives (kernel::spline).
static inline void mkl_aim_spline(float t, thread float* w, thread float* d) {
    float t2 = t * t, t3 = t2 * t;
    w[0] = (t2 * 2.0f - t3 - t) * 0.5f;
    w[1] = (t3 * 3.0f - t2 * 5.0f + 2.0f) * 0.5f;
    w[2] = (t2 * 4.0f - t3 * 3.0f + t) * 0.5f;
    w[3] = (t3 - t2) * 0.5f;
    d[0] = (t * 4.0f - t2 * 3.0f - 1.0f) * 0.5f;
    d[1] = (t2 * 9.0f - t * 10.0f) * 0.5f;
    d[2] = (t * 8.0f - t2 * 9.0f + 1.0f) * 0.5f;
    d[3] = (t2 * 3.0f - t * 2.0f) * 0.5f;
}
// Interpolated aim and stop Jacobian from the seed table (kernel::table_seed).
static inline bool mkl_aim_table_seed(device const MklAimNode* table, uint side, MklAimVec2 p, MklDf r_stop,
                                      thread MklAimVec2& a, thread MklAimMat2& j) {
    float step = mkl_aim_table_step(side);
    float fx = (p.v[0].v.x + MKL_AIM_TABLE_EXTENT) / step;
    float fy = (p.v[1].v.x + MKL_AIM_TABLE_EXTENT) / step;
    float ix = floor(fx), iy = floor(fy), last = float(side - 3);
    if (!(ix >= 1.0f && iy >= 1.0f && ix <= last && iy <= last)) return false;
    float wx[4], dx[4], wy[4], dy[4];
    mkl_aim_spline(fx - ix, wx, dx);
    mkl_aim_spline(fy - iy, wy, dy);
    MklAimVec2 ax = mkl_aim_zero(), ay = mkl_aim_zero();
    a = mkl_aim_zero();
    for (uint jj = 0; jj < 4; ++jj)
        for (uint ii = 0; ii < 4; ++ii) {
            MklAimNode n = table[(uint(iy) + jj - 1) * side + uint(ix) + ii - 1];
            if (n.status != 0) return false;
            for (uint k = 0; k < 2; ++k) {
                MklDf v = MklDf{float2(n.aim[2 * k], n.aim[2 * k + 1])};
                a.v[k] = a.v[k] + mkl_aim_scale(v, wx[ii] * wy[jj]);
                ax.v[k] = ax.v[k] + mkl_aim_scale(v, dx[ii] * wy[jj]);
                ay.v[k] = ay.v[k] + mkl_aim_scale(v, wx[ii] * dy[jj]);
            }
        }
    MklDf det = ax.v[0] * ay.v[1] - ay.v[0] * ax.v[1];
    MklDf q = r_stop / MklDf{mkl_df_mul_f32(det.v, mkl_aim_table_inv_step(side))};
    j.v[0][0] = ay.v[1] * q;
    j.v[0][1] = -(ay.v[0] * q);
    j.v[1][0] = -(ax.v[1] * q);
    j.v[1][1] = ax.v[0] * q;
    bool finite = true;
    for (uint k = 0; k < 4; ++k) finite = finite && isfinite(j.v[k / 2][k % 2].v.x);
    return finite && det.v.x != 0.0f;
}

// ---- Resumable pupil aiming ----------------------------------------------------
// kernel::FieldKernel::aim (table seed, chief prediction, four continuation
// stages) as a state machine whose unit of work is one phase entry
// (objective evaluation and, without a Jacobian, finite differences) or one
// Newton–Broyden iteration (line search, update, optional refresh). A command
// runs at most `budget` units per lane and parks the rest in MklAimResume
// (MKL_AIM_STATE_WORDS words) for a continuation command, so no command
// waits for the hardest ray's whole schedule. The arithmetic and its order
// are the monolithic specification's; a resumed solve is bitwise the same.
struct MklAimResume {
    MklAimVec2 x, r;
    MklAimMat2 j;
    MklDf rn;
    MklAimVec2 xc, prev;
    MklAimMat2 jc;
    uint phase, it, total, flags;
};
#define MKL_AIM_RS_ACTIVE 1u
#define MKL_AIM_RS_FRESH 2u
#define MKL_AIM_RS_HAS_J 4u
#define MKL_AIM_RS_HAS_JC 8u
#define MKL_AIM_RS_CONVERGED 16u
#define MKL_AIM_PHASE_DONE 6u
#define MKL_AIM_UNIT_MORE 0u
#define MKL_AIM_UNIT_CONVERGED 1u
#define MKL_AIM_UNIT_FAILED 2u

static inline void mkl_aim_put2(device float* w, MklAimVec2 v) {
    for (uint k = 0; k < 2; ++k) {
        w[2 * k] = v.v[k].v.x;
        w[2 * k + 1] = v.v[k].v.y;
    }
}
static inline MklAimVec2 mkl_aim_get2(device const float* w) {
    return mkl_aim_vec2(MklDf{float2(w[0], w[1])}, MklDf{float2(w[2], w[3])});
}
static inline void mkl_aim_put22(device float* w, MklAimMat2 m) {
    for (uint k = 0; k < 4; ++k) {
        w[2 * k] = m.v[k / 2][k % 2].v.x;
        w[2 * k + 1] = m.v[k / 2][k % 2].v.y;
    }
}
static inline MklAimMat2 mkl_aim_get22(device const float* w) {
    MklAimMat2 m;
    for (uint k = 0; k < 4; ++k) m.v[k / 2][k % 2] = MklDf{float2(w[2 * k], w[2 * k + 1])};
    return m;
}
static inline void mkl_aim_save(device uint* words, thread const MklAimResume& s) {
    device float* f = (device float*)words;
    mkl_aim_put2(f, s.x);
    mkl_aim_put2(f + 4, s.r);
    mkl_aim_put22(f + 8, s.j);
    f[16] = s.rn.v.x;
    f[17] = s.rn.v.y;
    mkl_aim_put2(f + 18, s.xc);
    mkl_aim_put2(f + 22, s.prev);
    mkl_aim_put22(f + 26, s.jc);
    words[34] = s.phase;
    words[35] = s.it;
    words[36] = s.total;
    words[37] = s.flags;
}
static inline MklAimResume mkl_aim_restore(device const uint* words) {
    device const float* f = (device const float*)words;
    MklAimResume s;
    s.x = mkl_aim_get2(f);
    s.r = mkl_aim_get2(f + 4);
    s.j = mkl_aim_get22(f + 8);
    s.rn = MklDf{float2(f[16], f[17])};
    s.xc = mkl_aim_get2(f + 18);
    s.prev = mkl_aim_get2(f + 22);
    s.jc = mkl_aim_get22(f + 26);
    s.phase = words[34];
    s.it = words[35];
    s.total = words[36];
    s.flags = words[37];
    return s;
}

template <bool OV>
struct MklAimMachine {
    MklAimFieldKernel<OV> f;
    MklDv src;
    uint wl;
    MklAimVec2 chief, p, target;
    MklAimMat2 sj;
    bool has_sj;
    device const MklAimNode* table;
    uint side;

    MklAimVec2 stage_target(uint phase) const {
        return phase <= 1 ? target : target * mkl_aim_c(float(phase - 1) * 0.25f);
    }
    // Seed the solve of phase s.phase as kernel::aim does; phase 0 falls
    // through to 1 when the table gives no seed (no trace is spent).
    void seed(thread MklAimResume& s) const {
        s.flags &= ~(MKL_AIM_RS_ACTIVE | MKL_AIM_RS_FRESH | MKL_AIM_RS_HAS_J);
        s.it = 0;
        if (s.phase == 0) {
            MklAimVec2 a;
            MklAimMat2 jt;
            if (table && mkl_aim_table_seed(table, side, p, f.pupil(2), a, jt)) {
                s.x = a;
                s.j = jt;
                s.flags |= MKL_AIM_RS_HAS_J;
                return;
            }
            s.phase = 1;
        }
        bool has_jc = (s.flags & MKL_AIM_RS_HAS_JC) != 0;
        MklAimVec2 delta;
        if (s.phase == 1) {
            bool have = has_jc && mkl_aim_solve2(s.jc, target, delta);
            s.x = chief + (have ? delta : p * f.pupil(1));
        } else {
            bool have = has_jc && mkl_aim_solve2(s.jc, stage_target(s.phase) - s.prev, delta);
            s.x = have ? s.xc + delta : s.xc;
        }
        s.j = s.jc;
        if (has_jc) s.flags |= MKL_AIM_RS_HAS_J;
    }
    void init(thread MklAimResume& s) const {
        s.x = s.r = s.xc = s.prev = mkl_aim_zero();
        s.j = s.jc = sj;
        s.rn = mkl_aim_c(0);
        s.phase = 0;
        s.it = s.total = 0;
        s.flags = has_sj ? MKL_AIM_RS_HAS_JC : 0u;
        seed(s);
    }
    uint check(thread const MklAimResume& s, MklDf tol) const {
        if ((s.rn - tol).v.x <= 0) return MKL_AIM_UNIT_CONVERGED;
        return s.it >= f.cap ? MKL_AIM_UNIT_FAILED : MKL_AIM_UNIT_MORE;
    }
    // One unit of the current solve (mkl_aim_newton's entry or one pass of
    // its loop body, followed by its loop-top test).
    uint unit(thread MklAimResume& s) const {
        MklAimObjective<OV> fun = {f, src, wl, stage_target(s.phase)};
        MklDf h = f.step(), tol = f.tol();
        if ((s.flags & MKL_AIM_RS_ACTIVE) == 0) {
            if (!fun.eval(s.x, s.r)) {
                s.rn = mkl_aim_c(INFINITY);
                return MKL_AIM_UNIT_FAILED;
            }
            s.rn = mkl_aim_norm(s.r);
            if ((s.flags & MKL_AIM_RS_HAS_J) == 0) {
                if (!mkl_aim_jacobian(fun, s.x, s.r, h, s.j)) return MKL_AIM_UNIT_FAILED;
                s.flags |= MKL_AIM_RS_FRESH | MKL_AIM_RS_HAS_J;
            }
            s.it = 0;
            s.flags |= MKL_AIM_RS_ACTIVE;
            return check(s, tol);
        }
        ++s.it;
        MklAimVec2 dx;
        if (!mkl_aim_solve2(s.j, s.r * mkl_aim_c(-1), dx)) {
            if (s.flags & MKL_AIM_RS_FRESH) return MKL_AIM_UNIT_FAILED;
            if (!mkl_aim_jacobian(fun, s.x, s.r, h, s.j)) return MKL_AIM_UNIT_FAILED;
            s.flags |= MKL_AIM_RS_FRESH;
            return check(s, tol);
        }
        MklDf lam = mkl_aim_c(1);
        bool accepted = false;
        MklAimVec2 xn, rr;
        MklDf nn = mkl_aim_c(0);
        for (uint k = 0; k < 16; ++k) {
            xn = s.x + dx * lam;
            if (fun.eval(xn, rr)) {
                nn = mkl_aim_norm(rr);
                if (isfinite(nn.v.x) && ((nn - s.rn).v.x < 0 || (nn - tol).v.x <= 0)) {
                    accepted = true;
                    break;
                }
            }
            lam = lam * mkl_aim_c(0.5f);
        }
        if (accepted) {
            MklAimVec2 d = xn - s.x;
            MklDf ss = d.v[0] * d.v[0] + d.v[1] * d.v[1];
            if (ss.v.x > 0) {
                MklAimVec2 y = rr - s.r, js = mkl_aim_vec2(s.j.v[0][0] * d.v[0] + s.j.v[0][1] * d.v[1],
                                                           s.j.v[1][0] * d.v[0] + s.j.v[1][1] * d.v[1]);
                for (uint row = 0; row < 2; ++row)
                    for (uint col = 0; col < 2; ++col)
                        s.j.v[row][col] = s.j.v[row][col] + (y.v[row] - js.v[row]) * d.v[col] / ss;
            }
            bool slow = (nn - mkl_aim_c(0.5f) * s.rn).v.x > 0;
            s.x = xn;
            s.r = rr;
            s.rn = nn;
            s.flags &= ~MKL_AIM_RS_FRESH;
            if (slow && (s.rn - tol).v.x > 0) {
                MklAimMat2 jn;
                if (mkl_aim_jacobian(fun, s.x, s.r, h, jn)) {
                    s.j = jn;
                    s.flags |= MKL_AIM_RS_FRESH;
                }
            }
        } else {
            if (s.flags & MKL_AIM_RS_FRESH) return MKL_AIM_UNIT_FAILED;
            if (!mkl_aim_jacobian(fun, s.x, s.r, h, s.j)) return MKL_AIM_UNIT_FAILED;
            s.flags |= MKL_AIM_RS_FRESH;
        }
        return check(s, tol);
    }
    // Phase transitions of kernel::aim; true once the aim is finished.
    bool advance(thread MklAimResume& s, uint outcome) const {
        s.total += s.it;
        if (outcome == MKL_AIM_UNIT_CONVERGED) {
            if (s.phase <= 1 || s.phase == 5) {
                s.phase = MKL_AIM_PHASE_DONE;
                s.flags |= MKL_AIM_RS_CONVERGED;
                return true;
            }
            s.xc = s.x;
            s.jc = s.j;
            s.flags |= MKL_AIM_RS_HAS_JC;
            s.prev = stage_target(s.phase);
            s.phase += 1;
            seed(s);
            return false;
        }
        if (s.phase == 0) {
            s.phase = 1;
            seed(s);
            return false;
        }
        if (s.phase == 1) {
            s.phase = 2;
            s.xc = chief;
            s.prev = mkl_aim_zero();
            seed(s);
            return false;
        }
        s.phase = MKL_AIM_PHASE_DONE;
        return true;
    }
    // Run at most `budget` units; true when finished.
    bool run(thread MklAimResume& s, uint budget) const {
        for (uint u = 0; u < budget && s.phase != MKL_AIM_PHASE_DONE; ++u) {
            uint o = unit(s);
            if (o != MKL_AIM_UNIT_MORE) advance(s, o);
        }
        return s.phase == MKL_AIM_PHASE_DONE;
    }
};
template <bool OV>
static inline MklAimMachine<OV> mkl_aim_machine(MklAimFieldKernel<OV> f, constant MklAimCache& seed,
                                                device const MklAimNode* table, uint side, MklAimVec2 p) {
    MklAimMachine<OV> m;
    m.f = f;
    m.src = mkl_aim_vec3(mkl_aim_load(seed.source, 0), mkl_aim_load(seed.source, 1), mkl_aim_load(seed.source, 2));
    m.wl = f.field->wavelength;
    m.chief = mkl_aim_vec2(mkl_aim_load(seed.chief, 0), mkl_aim_load(seed.chief, 1));
    m.p = p;
    m.target = p * f.pupil(2);
    for (uint k = 0; k < 4; ++k) m.sj.v[k / 2][k % 2] = mkl_aim_load(seed.jacobian, k);
    m.has_sj = seed.has_jacobian != 0;
    m.table = table;
    m.side = side;
    return m;
}

struct MklAimParams {
    uint count, offset, stride, fields, systems, cap;
    float tolerance;
    uint table;
    uint pattern, side;
    float extent[2];
    uint budget, table_side, flags, reserved;
};
#define MKL_AIM_CULL 1u
// Looser than the 1e-12 disk inclusion of the lattice masks (kernel::CULL_RADIUS2).
#define MKL_AIM_CULL_RADIUS2 float2(1.0f, 1.0000001e-9f)

#define MKL_AIM_SYSTEMS                                                                     \
    constant MklPackedHeader *headers [[buffer(0)]], constant MklPackedSurface *surfaces [[buffer(1)]], \
        constant float *pool [[buffer(2)]], constant MklAimField *fields [[buffer(3)]]
// Continuation plumbing shared by generation and table kernels: resumable
// state per lane of the tile, the pending list this command appends, and the
// pending list a continuation command consumes.
#define MKL_AIM_RESUME_ARGS                                                                 \
    device uint *state [[buffer(14)]], device uint *pending [[buffer(15)]],                  \
        device atomic_uint *pending_count [[buffer(16)]], device const uint *list [[buffer(17)]], \
        device const uint *list_count [[buffer(18)]]

#define MKL_AIM_CHIEFS_ARGS                                                                 \
    MKL_AIM_SYSTEMS, device MklAimCache *seeds [[buffer(4)]], constant MklAimParams &p [[buffer(5)]], \
        constant MklTraceParams &newton [[buffer(6)]], uint tid [[thread_position_in_grid]]
template <bool OV>
kernel void mkl_aim_chiefs(MKL_AIM_CHIEFS_ARGS) {
    if (tid >= p.count) return;
    uint i = tid + p.offset;
    constant MklAimField& f = fields[i];
    MklAimCache out = {};
    out.status = MKL_AIM_INVALID;
    if (f.system < p.systems && f.wavelength < headers[f.system].num_wavelengths) {
        constant auto& h = headers[f.system];
        out = mkl_aim_prepare(MklAimFieldKernel<OV>{&h, surfaces + h.surface_offset, pool, &f, &newton,
                                                    p.tolerance, p.cap});
    }
    seeds[i] = out;
}
template [[host_name("mkl_aim_chiefs")]] kernel void mkl_aim_chiefs<false>(MKL_AIM_CHIEFS_ARGS);
template [[host_name("mkl_aim_chiefs_overlays")]] kernel void mkl_aim_chiefs<true>(MKL_AIM_CHIEFS_ARGS);

// Threadgroup count of the next continuation command from a pending count,
// and a zeroed counter for the list that command appends.
kernel void mkl_aim_indirect(device const uint* count [[buffer(0)]], device uint* args [[buffer(1)]],
                             device uint* next_count [[buffer(2)]], constant uint& width [[buffer(3)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    args[0] = (count[0] + width - 1) / width;
    args[1] = 1;
    args[2] = 1;
    next_count[0] = 0;
}

// Seed-table node i (row i / side², node i % side², `p.table_side` nodes per
// axis), solved without a table; nodes beyond pupil radius 1.25 stay
// unsolved (kernel::FieldKernel::node). `start` initialises the lane's
// state; a continuation resumes it.
template <bool OV>
static inline void mkl_aim_table_node(MKL_AIM_SYSTEMS, constant MklAimCache* seeds, constant MklAimParams& p,
                                      constant MklTraceParams& newton, device MklAimNode* table,
                                      device uint* state, device uint* pending, device atomic_uint* pending_count,
                                      uint i, uint lane, bool start) {
    const uint side = p.table_side, nodes = side * side;
    uint row = i / nodes, k = i % nodes;
    constant MklAimField& f = fields[row];
    constant MklAimCache& seed = seeds[row];
    MklAimNode node = {};
    node.status = 1;
    float step = mkl_aim_table_step(side);
    float x = float(k % side) * step - MKL_AIM_TABLE_EXTENT, y = float(k / side) * step - MKL_AIM_TABLE_EXTENT;
    if ((f.flags & MKL_AIM_REAL) != 0 && !mkl_aim_hard_failure(seed.status) && x * x + y * y <= 1.5625f) {
        constant auto& h = headers[f.system];
        MklAimFieldKernel<OV> fk = {&h, surfaces + h.surface_offset, pool, &f, &newton, p.tolerance, p.cap};
        MklAimMachine<OV> m = mkl_aim_machine(fk, seed, nullptr, side, mkl_aim_vec2(mkl_aim_c(x), mkl_aim_c(y)));
        MklAimResume s;
        if (start)
            m.init(s);
        else
            s = mkl_aim_restore(state + lane * MKL_AIM_STATE_WORDS);
        if (!m.run(s, p.budget)) {
            mkl_aim_save(state + lane * MKL_AIM_STATE_WORDS, s);
            pending[atomic_fetch_add_explicit(pending_count, 1u, memory_order_relaxed)] = i;
            return;
        }
        for (uint c = 0; c < 2; ++c) {
            node.aim[2 * c] = s.x.v[c].v.x;
            node.aim[2 * c + 1] = s.x.v[c].v.y;
        }
        node.status = (s.flags & MKL_AIM_RS_CONVERGED) ? 0 : 1;
    }
    table[i] = node;
}
#define MKL_AIM_TABLE_ARGS                                                                  \
    MKL_AIM_SYSTEMS, constant MklAimCache *seeds [[buffer(4)]], constant MklAimParams &p [[buffer(5)]], \
        constant MklTraceParams &newton [[buffer(6)]], device MklAimNode *table [[buffer(7)]],          \
        MKL_AIM_RESUME_ARGS, uint tid [[thread_position_in_grid]]
template <bool OV>
kernel void mkl_aim_table(MKL_AIM_TABLE_ARGS) {
    if (tid >= p.count) return;
    mkl_aim_table_node<OV>(headers, surfaces, pool, fields, seeds, p, newton, table, state, pending, pending_count,
                           tid + p.offset, tid, true);
}
template <bool OV>
kernel void mkl_aim_table_continue(MKL_AIM_TABLE_ARGS) {
    if (tid >= list_count[0]) return;
    uint i = list[tid];
    mkl_aim_table_node<OV>(headers, surfaces, pool, fields, seeds, p, newton, table, state, pending, pending_count,
                           i, i - p.offset, false);
}
template [[host_name("mkl_aim_table")]] kernel void mkl_aim_table<false>(MKL_AIM_TABLE_ARGS);
template [[host_name("mkl_aim_table_overlays")]] kernel void mkl_aim_table<true>(MKL_AIM_TABLE_ARGS);
template [[host_name("mkl_aim_table_continue")]] kernel void mkl_aim_table_continue<false>(MKL_AIM_TABLE_ARGS);
template [[host_name("mkl_aim_table_continue_overlays")]] kernel void mkl_aim_table_continue<true>(MKL_AIM_TABLE_ARGS);

static inline void mkl_aim_write(device float* ff, device uint* uu, device uint* ids, device uint* reports,
                                 uint n, uint i, MklRawState ray, uint system, uint wl, uint2 id,
                                 uint4 report) {
    for (uint k = 0; k < 3; ++k) {
        ff[mkl_plane(k, n, i)] = ray.pos.v[k].x;
        ff[mkl_plane(k + 9, n, i)] = ray.pos.v[k].y;
        ff[mkl_plane(k + 3, n, i)] = ray.dir.v[k].x;
        ff[mkl_plane(k + 12, n, i)] = ray.dir.v[k].y;
    }
    ff[mkl_plane(6, n, i)] = ray.opl.x;
    ff[mkl_plane(7, n, i)] = ray.opl.y;
    ff[mkl_plane(8, n, i)] = ray.weight;
    uu[i] = system;
    uu[mkl_plane(1, n, i)] = wl;
    ids[i] = id.x;
    ids[mkl_plane(1, n, i)] = id.y;
    for (uint k = 0; k < 4; ++k) reports[mkl_plane(k, n, i)] = report[k];
}
static inline MklRawState mkl_aim_read(device const float* ff, uint n, uint i) {
    MklRawState r = {};
    for (uint k = 0; k < 3; ++k) {
        r.pos.v[k] = float2(ff[mkl_plane(k, n, i)], ff[mkl_plane(k + 9, n, i)]);
        r.dir.v[k] = float2(ff[mkl_plane(k + 3, n, i)], ff[mkl_plane(k + 12, n, i)]);
    }
    r.opl = float2(ff[mkl_plane(6, n, i)], ff[mkl_plane(7, n, i)]);
    r.weight = ff[mkl_plane(8, n, i)];
    return r;
}
// Sample i: explicit, or the procedural lattice / fan (kernel::sample_at).
static inline MklAimSample mkl_aim_sample(constant MklAimSample* samples, constant MklAimParams& prm, uint i) {
    MklAimSample sample = samples[prm.pattern == 0 ? i : 0];
    if (prm.pattern != 0) {
        MklDf extent = mkl_aim_load(prm.extent, 0);
        uint side = prm.pattern == 1 ? prm.side : prm.stride;
        MklDf px = side <= 1 ? mkl_aim_c(0)
                             : (mkl_aim_c(-1) + mkl_aim_c(2) * mkl_aim_c(float(prm.pattern == 1 ? i % side : i)) /
                                                    mkl_aim_c(float(side - 1))) *
                                   extent;
        MklDf py = prm.pattern == 1 ? (side <= 1 ? mkl_aim_c(0)
                                                 : (mkl_aim_c(-1) + mkl_aim_c(2) * mkl_aim_c(float(i / side)) /
                                                                        mkl_aim_c(float(side - 1))) *
                                                       extent)
                                    : mkl_aim_c(0);
        if (prm.pattern == 2 && prm.side == 1) {
            py = px;
            px = mkl_aim_c(0);
        }
        mkl_aim_store(sample.pupil, 0, px);
        mkl_aim_store(sample.pupil, 1, py);
        ulong id = ((ulong(sample.id_hi) << 32) | ulong(sample.id_lo)) + ulong(i);
        sample.id_lo = uint(id);
        sample.id_hi = uint(id >> 32);
    }
    return sample;
}
// Samples never aimed: skip-flagged, or outside the unit disk when culling
// (kernel::excluded).
static inline bool mkl_aim_excluded(MklAimSample sample, constant MklAimParams& prm) {
    if ((sample.field & MKL_AIM_SKIP) != 0) return true;
    if ((prm.flags & MKL_AIM_CULL) == 0) return false;
    MklDf x = mkl_aim_load(sample.pupil, 0), y = mkl_aim_load(sample.pupil, 1);
    MklDf r2 = x * x + y * y;
    return (r2 - MklDf{MKL_AIM_CULL_RADIUS2}).v.x > 0;
}

#define MKL_GENERATE_AIM_ARGS                                                                         \
    MKL_AIM_SYSTEMS, constant MklAimCache *seeds [[buffer(4)]], constant MklAimParams &prm [[buffer(5)]], \
        constant MklAimSample *samples [[buffer(6)]], device float *ff [[buffer(7)]],                 \
        device uint *uu [[buffer(8)]], device uint *ids [[buffer(9)]], device uint *reports [[buffer(10)]], \
        device atomic_uint *failures [[buffer(11)]], constant MklTraceParams &newton [[buffer(12)]],   \
        device const MklAimNode *tables [[buffer(13)]], MKL_AIM_RESUME_ARGS, uint tid [[thread_position_in_grid]]
// Ray i of the generation; `lane` indexes the tile's resumable state.
template <bool OV>
static inline void mkl_aim_ray(MKL_AIM_SYSTEMS, constant MklAimCache* seeds, constant MklAimParams& prm,
                               constant MklAimSample* samples, device float* ff, device uint* uu, device uint* ids,
                               device uint* reports, device atomic_uint* failures, constant MklTraceParams& newton,
                               device const MklAimNode* tables, device uint* state, device uint* pending,
                               device atomic_uint* pending_count, uint i, uint lane, bool start) {
    MklAimSample sample = mkl_aim_sample(samples, prm, i);
    uint row = sample.field & ~MKL_AIM_SKIP;
    MklRawState ray = {};
    uint4 report = uint4(MKL_AIM_INVALID, 0, 0, 0);
    uint system = 0, wl = 0;
    if (row < prm.fields) {
        system = fields[row].system;
        wl = fields[row].wavelength;
    }
    if (mkl_aim_excluded(sample, prm))
        report.x = MKL_AIM_OUTSIDE;
    else if (row < prm.fields) {
        constant auto& meta = fields[row];
        constant auto& seed = seeds[row];
        if (mkl_aim_hard_failure(seed.status))
            report.x = seed.status;
        else {
            constant auto& h = headers[system];
            MklAimFieldKernel<OV> f = {&h, surfaces + h.surface_offset, pool, &meta, &newton, prm.tolerance, prm.cap};
            MklAimVec2 p;
            for (uint k = 0; k < 2; ++k)
                p.v[k] = mkl_aim_load(meta.vignette, k) +
                         (mkl_aim_c(1) - mkl_aim_load(meta.vignette, k + 2)) * mkl_aim_load(sample.pupil, k);
            MklAimVec2 x = p * f.pupil(1);
            MklDf residual = mkl_aim_c(0);
            bool converged = true;
            uint iterations = 0;
            if (f.real()) {
                const uint nodes = prm.table_side * prm.table_side;
                MklAimMachine<OV> m =
                    mkl_aim_machine(f, seed, prm.table != 0 ? tables + row * nodes : nullptr, prm.table_side, p);
                MklAimResume s;
                if (start)
                    m.init(s);
                else
                    s = mkl_aim_restore(state + lane * MKL_AIM_STATE_WORDS);
                if (!m.run(s, prm.budget)) {
                    mkl_aim_save(state + lane * MKL_AIM_STATE_WORDS, s);
                    pending[atomic_fetch_add_explicit(pending_count, 1u, memory_order_relaxed)] = i;
                    return;
                }
                x = s.x;
                residual = s.rn;
                converged = (s.flags & MKL_AIM_RS_CONVERGED) != 0;
                iterations = s.total;
            }
            ray = f.start(mkl_aim_vec3(mkl_aim_load(seed.source, 0), mkl_aim_load(seed.source, 1),
                                       mkl_aim_load(seed.source, 2)),
                          x);
            if (ray.status == 0) ray.weight = sample.weight;
            MklDf reported = ray.status != 0 ? residual : f.real() ? residual / f.pupil(2) : mkl_aim_c(0);
            report = uint4(ray.status != 0 ? ray.status
                           : converged     ? 0
                                           : (MKL_AIM_FAILED | (f.h->stop << MKL_RAY_SURFACE_SHIFT)),
                           iterations, as_type<uint>(reported.v.x), as_type<uint>(reported.v.y));
        }
    }
    mkl_aim_write(ff, uu, ids, reports, prm.stride, i, ray, system, wl, uint2(sample.id_lo, sample.id_hi), report);
    if (report.x != 0 && report.x != MKL_AIM_OUTSIDE) atomic_fetch_add_explicit(failures, 1u, memory_order_relaxed);
}
template <bool OV>
kernel void mkl_generate_aim(MKL_GENERATE_AIM_ARGS) {
    if (tid >= prm.count) return;
    mkl_aim_ray<OV>(headers, surfaces, pool, fields, seeds, prm, samples, ff, uu, ids, reports, failures, newton, tables,
                    state, pending, pending_count, tid + prm.offset, tid, true);
}
template <bool OV>
kernel void mkl_aim_continue(MKL_GENERATE_AIM_ARGS) {
    if (tid >= list_count[0]) return;
    uint i = list[tid];
    mkl_aim_ray<OV>(headers, surfaces, pool, fields, seeds, prm, samples, ff, uu, ids, reports, failures, newton, tables,
                    state, pending, pending_count, i, i - prm.offset, false);
}
template [[host_name("mkl_generate_aim")]] kernel void mkl_generate_aim<false>(MKL_GENERATE_AIM_ARGS);
template [[host_name("mkl_generate_aim_overlays")]] kernel void mkl_generate_aim<true>(MKL_GENERATE_AIM_ARGS);
template [[host_name("mkl_aim_continue")]] kernel void mkl_aim_continue<false>(MKL_GENERATE_AIM_ARGS);
template [[host_name("mkl_aim_continue_overlays")]] kernel void mkl_aim_continue<true>(MKL_GENERATE_AIM_ARGS);

// Optional paired record planes: [slot, ray, MKL_AIM_RECORD_FLOATS] hi and
// lo. A compile-time switch: record support costs the plain trace ~15 %.
struct MklAimRecords {
    device float* hi;
    device float* lo;
    uint n, i;
    bool enabled;
    void write(uint slot, MklRawState s) {
        if (!enabled) return;
        ulong a = (ulong(slot) * n + i) * MKL_AIM_RECORD_FLOATS;
        for (uint k = 0; k < 3; ++k) {
            hi[a + k] = s.pos.v[k].x;
            lo[a + k] = s.pos.v[k].y;
            hi[a + 3 + k] = s.dir.v[k].x;
            lo[a + 3 + k] = s.dir.v[k].y;
            hi[a + 8 + k] = s.normal.v[k].x;
            lo[a + 8 + k] = s.normal.v[k].y;
        }
        hi[a + 6] = s.opl.x;
        lo[a + 6] = s.opl.y;
        hi[a + 7] = s.weight;
        lo[a + 7] = 0;
        hi[a + 11] = s.index.x;
        lo[a + 11] = s.index.y;
        hi[a + 12] = as_type<float>(s.status);
        lo[a + 12] = 0;
    }
};
struct MklResidentTraceParams {
    MklTraceParams trace;
    uint offset, systems, reserved0, reserved1;
};
#define MKL_TRACE_RESIDENT_ARGS                                                                       \
    constant MklPackedHeader *headers [[buffer(0)]], constant MklPackedSurface *surfaces [[buffer(1)]],  \
        constant float *pool [[buffer(2)]], constant uint *steps [[buffer(3)]],                        \
        constant MklResidentTraceParams &prm [[buffer(4)]], device const float *in_f [[buffer(5)]],      \
        device const uint *in_u [[buffer(6)]], device const uint *in_ids [[buffer(7)]],                \
        device const uint *in_report [[buffer(8)]], device float *out_f [[buffer(9)]],                 \
        device uint *out_u [[buffer(10)]], device uint *out_ids [[buffer(11)]],                        \
        device uint *out_report [[buffer(12)]], device uint *status [[buffer(13)]],                    \
        device float *rec_hi [[buffer(14)]], device float *rec_lo [[buffer(15)]],                      \
        uint tid [[thread_position_in_grid]]
// Resident rays carry device-computed system/wavelength indices and may run
// a device-built path program, so both are checked before they index data.
template <bool OV, bool REC>
kernel void mkl_trace_resident(MKL_TRACE_RESIDENT_ARGS) {
    if (tid >= prm.trace.num_rays) return;
    uint i = tid + prm.offset, n = prm.trace.stride, system = in_u[i], wl = in_u[mkl_plane(1, n, i)];
    uint4 report;
    for (uint k = 0; k < 4; ++k) report[k] = in_report[mkl_plane(k, n, i)];
    MklRawState ray = mkl_aim_read(in_f, n, i);
    ray.status = report.x;
    ray.last = report.x >> MKL_RAY_SURFACE_SHIFT;
    MklAimRecords rec = {rec_hi, rec_lo, n, i, (prm.trace.flags & MKL_TRACE_RECORD) != 0};
    if (ray.status == 0) {
        bool valid = system < prm.systems;
        if (valid) {
            constant auto& h = headers[system];
            valid = wl < h.num_wavelengths && prm.trace.start_surface < h.num_surfaces;
            uint current = prm.trace.start_surface;
            bool backwards = false;
            for (uint k = 0; valid && k < prm.trace.num_steps; ++k) {
                uint word = steps[k], si = word & MKL_STEP_SURFACE_MASK,
                     mode = (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK;
                bool back = (word & MKL_STEP_BACKWARD) != 0;
                valid = (word & ~(MKL_STEP_SURFACE_MASK | (MKL_STEP_MODE_MASK << MKL_STEP_MODE_SHIFT) |
                                  MKL_STEP_BACKWARD)) == 0 &&
                        si < h.num_surfaces && mode <= MKL_STEP_REFLECT &&
                        (back ? current > 0 && si + 1 == current : si == current + 1) && (k == 0 || back == backwards);
                if (valid) {
                    constant auto& s = surfaces[h.surface_offset + si];
                    const bool cb = s.shape == MKL_SHAPE_COORD_BREAK;
                    const bool mirror = !cb && s.interaction == MKL_INTERACT_REFLECT;
                    valid = !(back && mode == MKL_STEP_SEQUENTIAL) && !(mode == MKL_STEP_TRANSMIT && mirror) &&
                            !(mode == MKL_STEP_REFLECT && cb);
                    backwards = back;
                    if (mode == MKL_STEP_REFLECT && !mirror) backwards = !backwards;
                }
                current = si;
            }
            if (valid) {
                if (REC)
                    ray = mkl_trace_raw<OV>(h, surfaces + h.surface_offset, pool, steps, prm.trace,
                                            prm.trace.num_steps, wl, ray, rec);
                else {
                    MklNoRecords none;
                    ray = mkl_trace_raw<OV>(h, surfaces + h.surface_offset, pool, steps, prm.trace,
                                            prm.trace.num_steps, wl, ray, none);
                }
            }
        }
        if (!valid) ray.status = MKL_AIM_INVALID;
    }
    if (report.x != 0 || ray.status == MKL_AIM_INVALID)
        for (uint k = 0; k <= prm.trace.num_steps; ++k) rec.write(k, ray);
    mkl_aim_write(out_f, out_u, out_ids, out_report, n, i, ray, system, wl, uint2(in_ids[i], in_ids[mkl_plane(1, n, i)]),
                  report);
    status[i] = ray.status;
    status[mkl_plane(1, n, i)] = ray.last;
    status[mkl_plane(2, n, i)] = ray.done;
}
template [[host_name("mkl_trace_resident")]] kernel void mkl_trace_resident<false, false>(MKL_TRACE_RESIDENT_ARGS);
template [[host_name("mkl_trace_resident_overlays")]] kernel void mkl_trace_resident<true, false>(MKL_TRACE_RESIDENT_ARGS);
template [[host_name("mkl_trace_resident_records")]] kernel void mkl_trace_resident<false, true>(MKL_TRACE_RESIDENT_ARGS);
template [[host_name("mkl_trace_resident_overlays_records")]] kernel void mkl_trace_resident<true, true>(MKL_TRACE_RESIDENT_ARGS);

struct MklPupilParams {
    float radius[2], index[2], wavelength[2];
    int sheet;
    uint telecentric, count, offset, stride, reserved;
};
// Rationalised reference-sphere correction (kernel::project).
static inline bool mkl_aim_correction(MklDv pos, MklDv dir, MklDv center, MklDf r, MklDf sign, bool tele,
                                      thread MklDf& corr, thread MklDv& pupil) {
    MklDv w;
    for (uint k = 0; k < 3; ++k) w.v[k] = (MklDf{pos.v[k]} - MklDf{center.v[k]}).v;
    MklDf b = MklDf{mkl_ddot(dir, w)};
    if (tele) {
        corr = b;
        for (uint k = 0; k < 3; ++k) pupil.v[k] = mkl_df_neg(dir.v[k]);
        return true;
    }
    MklDv cross = mkl_aim_vec3(MklDf{w.v[1]} * MklDf{dir.v[2]} - MklDf{w.v[2]} * MklDf{dir.v[1]},
                               MklDf{w.v[2]} * MklDf{dir.v[0]} - MklDf{w.v[0]} * MklDf{dir.v[2]},
                               MklDf{w.v[0]} * MklDf{dir.v[1]} - MklDf{w.v[1]} * MklDf{dir.v[0]});
    MklDf p2 = MklDf{mkl_ddot(cross, cross)}, delta = r * r - p2;
    if (delta.v.x < 0) return false;
    MklDf root = mkl_sqrt(delta);
    corr = b + sign * p2 / (r + root);
    MklDf t = -b + sign * root;
    for (uint k = 0; k < 3; ++k) pupil.v[k] = ((MklDf{w.v[k]} + MklDf{dir.v[k]} * t) / r).v;
    return true;
}
kernel void mkl_project_pupil(device const float* rays [[buffer(0)]], device const uint* hit_status [[buffer(1)]],
                              constant float* chief [[buffer(2)]], constant float* centre [[buffer(3)]],
                              constant MklPupilParams& p [[buffer(4)]], device float2* output [[buffer(5)]],
                              device uint* status [[buffer(6)]], uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    uint i = tid + p.offset, n = p.stride;
    uint code = hit_status[i];
    MklRawState ray = mkl_aim_read(rays, n, i);
    MklDv cp, cd, center, pupil = {}, unused;
    for (uint k = 0; k < 3; ++k) {
        cp.v[k] = mkl_aim_load(chief, k).v;
        cd.v[k] = mkl_aim_load(chief, 3 + k).v;
        center.v[k] = mkl_aim_load(centre, k).v;
    }
    MklDf corr, cc, opd = mkl_aim_c(NAN), r = mkl_aim_load(p.radius, 0), sign = mkl_aim_c(float(p.sheet));
    if (code == 0) {
        if (mkl_aim_correction(ray.pos, ray.dir, center, r, sign, p.telecentric != 0, corr, pupil) &&
            mkl_aim_correction(cp, cd, center, r, sign, p.telecentric != 0, cc, unused)) {
            opd = (mkl_aim_load(chief, 6) - MklDf{ray.opl}) + mkl_aim_load(p.index, 0) * (corr - cc);
            // 1e-3 mm per µm as a double-float.
            opd = opd / (mkl_aim_load(p.wavelength, 0) * MklDf{float2(0.0010000000474974513f, -4.7497450835765065e-11f)});
            if (cd.v[2].x < 0) pupil.v[2] = mkl_df_neg(pupil.v[2]);
        } else
            code = MKL_PUPIL_MISS;
    }
    status[i] = code;
    if (code != 0) {
        opd = mkl_aim_c(NAN);
        for (uint k = 0; k < 3; ++k) pupil.v[k] = float2(NAN, 0);
    }
    output[i] = opd.v;
    for (uint k = 0; k < 3; ++k) output[mkl_plane(k + 1, n, i)] = pupil.v[k];
}

// Aim convergence statistics over a [4,n] report, 256 rays per thread:
// stats = [failed, iterations (low word), max iterations, histogram of
// 0, 1, 2, 3-4, 5-8, 9-16, 17-60, >60 iterations, iterations (high word)].
kernel void mkl_aim_stats(device const uint* report [[buffer(0)]], device atomic_uint* stats [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint g [[thread_position_in_grid]]) {
    uint first = g * 256;
    if (first >= n) return;
    uint failed = 0, total = 0, most = 0;
    uint hist[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (uint i = first; i < min(first + 256, n); ++i) {
        if (report[i] == MKL_AIM_OUTSIDE) continue;
        uint it = report[n + i];
        failed += report[i] != 0 ? 1u : 0u;
        total += it;
        most = max(most, it);
        uint b = it <= 2 ? it : it <= 4 ? 3 : it <= 8 ? 4 : it <= 16 ? 5 : it <= 60 ? 6 : 7;
        hist[b] += 1;
    }
    atomic_fetch_add_explicit(&stats[0], failed, memory_order_relaxed);
    uint old = atomic_fetch_add_explicit(&stats[1], total, memory_order_relaxed);
    if (old + total < old) atomic_fetch_add_explicit(&stats[11], 1u, memory_order_relaxed);
    atomic_fetch_max_explicit(&stats[2], most, memory_order_relaxed);
    for (uint k = 0; k < 8; ++k)
        if (hist[k] != 0) atomic_fetch_add_explicit(&stats[3 + k], hist[k], memory_order_relaxed);
}

// Stable unresolved compaction: fixed 256-ray groups, exclusive prefix,
// original-order scatter.
kernel void mkl_aim_count(device const uint* report [[buffer(0)]], device uint* counts [[buffer(1)]],
                          device uint* sparse [[buffer(2)]], constant uint& n [[buffer(3)]],
                          uint g [[thread_position_in_grid]]) {
    uint first = g * 256;
    if (first >= n) return;
    uint count = 0;
    for (uint i = first; i < min(first + 256, n); ++i) {
        count += report[i] != 0 && report[i] != MKL_AIM_OUTSIDE;
        sparse[i + 1] = 0xffffffffu;
    }
    counts[g] = count;
}
kernel void mkl_aim_prefix(device uint* counts [[buffer(0)]], device uint* sparse [[buffer(1)]],
                           constant uint& groups [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint sum = 0;
    for (uint g = 0; g < groups; ++g) {
        uint count = counts[g];
        counts[g] = sum;
        sum += count;
    }
    sparse[0] = sum;
}
kernel void mkl_aim_scatter(device const uint* report [[buffer(0)]], device const uint* counts [[buffer(1)]],
                            device uint* sparse [[buffer(2)]], constant uint& n [[buffer(3)]],
                            uint g [[thread_position_in_grid]]) {
    uint first = g * 256;
    if (first >= n) return;
    uint index = counts[g] + 1;
    for (uint i = first; i < min(first + 256, n); ++i)
        if (report[i] != 0 && report[i] != MKL_AIM_OUTSIDE) sparse[index++] = i;
}
