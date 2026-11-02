// Local-frame trace helpers shared by trace.metal (mkl_trace) and aiming.metal.
//
// The MSL twin of crates/mklight-gpu/src/trace_kernel.rs, which documents
// the algorithm and is the bit-exact CPU emulation: every function below has
// a Rust counterpart of the same name (without the mkl_ prefix) written
// operation for operation alike. Change both together.
//
// In short: f32 decisions mirroring mklight-trace (conic roots and vertex
// sheet, asphere Newton classification, TIR, clipping) and double-float
// geometry (point, direction, unnormalised normal, indices, one Newton
// refinement of every hit), with the optical path in double-float.
#ifndef MKLIGHT_TRACE_SHARED
#define MKLIGHT_TRACE_SHARED
#include "surface_math.h"

// 2^-20: vertex-sheet tolerance (trace_kernel::SHEET_TOL).
#define MKL_SHEET_TOL as_type<float>(0x35800000u)
#define MKL_FLT_MIN as_type<float>(0x00800000u)

// A double-float 3-vector.
struct MklDv {
    float2 v[3];
};

struct MklHit {
    float2 ta;
    float2 tb;
    MklDv x;
    MklDv g;
    float3 normal;
};

struct MklState {
    float3 pos;
    float3 dir;
    float2 opl;
    float weight;
    uint status;
};

static inline float3 mkl_hi3(MklDv p) { return float3(p.v[0].x, p.v[1].x, p.v[2].x); }
static inline float3 mkl_lo3(MklDv p) { return float3(mkl_df_low(p.v[0].y), mkl_df_low(p.v[1].y), mkl_df_low(p.v[2].y)); }
static inline bool mkl_finite3(float3 v) { return isfinite(v.x) && isfinite(v.y) && isfinite(v.z); }
static inline float mkl_norm3(float3 v) { return mkl_hypot3(v.x, v.y, v.z); }

static inline float3 mkl_normalized(float3 v) {
    float n = mkl_norm3(v);
    if (n == 0.0f) return v;
    return float3(v.x / n, v.y / n, v.z / n);
}

static inline float2 mkl_ddot(MklDv a, MklDv b) {
    float2 s = mkl_df_mul(a.v[2], b.v[2]);
    s = mkl_df_add(s, mkl_df_mul(a.v[1], b.v[1]));
    return mkl_df_add(s, mkl_df_mul(a.v[0], b.v[0]));
}

static inline MklDv mkl_renorm_along(MklDv d, MklDv g) {
    float2 m = mkl_df_add_f32(mkl_ddot(d, d), -1.0f);
    float h = mkl_dot3(mkl_hi3(d), mkl_hi3(g));
    if (h == 0.0f) return d;
    float b = m.x / (2.0f * h);
    MklDv o;
    for (int k = 0; k < 3; ++k) o.v[k] = mkl_df_sub(d.v[k], mkl_df_mul_f32(g.v[k], b));
    return o;
}

static inline MklDv mkl_renorm(MklDv d) {
    float s = 0.5f * mkl_df_add_f32(mkl_ddot(d, d), -1.0f).x;
    MklDv o;
    for (int k = 0; k < 3; ++k) o.v[k] = mkl_df_sub(d.v[k], mkl_df_mul_f32(d.v[k], s));
    return o;
}

static inline float2 mkl_rot_row(constant MklPackedSurface& s, uint row, bool transpose, MklDv v) {
    uint a = transpose ? row : 3u * row;
    uint b = transpose ? row + 3u : 3u * row + 1u;
    uint c = transpose ? row + 6u : 3u * row + 2u;
    float lo = fma(s.to_next_rot_lo[a], v.v[0].x,
                   fma(s.to_next_rot_lo[b], v.v[1].x, s.to_next_rot_lo[c] * v.v[2].x));
    float2 acc = mkl_df_mul_f32(v.v[2], s.to_next_rot[c]);
    acc = mkl_df_add(acc, mkl_df_mul_f32(v.v[1], s.to_next_rot[b]));
    acc = mkl_df_add(acc, mkl_df_mul_f32(v.v[0], s.to_next_rot[a]));
    return mkl_df_add_f32(acc, mkl_df_low(lo));
}

static inline void mkl_xf_forward(constant MklPackedSurface& s, thread MklDv& p, thread MklDv& d) {
    if ((s.flags & MKL_SURF_AXIAL) != 0u) {
        p.v[2] = mkl_df_sub(p.v[2], float2(s.to_next_origin[2], s.to_next_origin_lo[2]));
        return;
    }
    MklDv p0 = p, d0 = d;
    for (uint row = 0; row < 3u; ++row) {
        p.v[row] = mkl_df_sub(mkl_rot_row(s, row, false, p0),
                              float2(s.to_next_shift[row], s.to_next_shift_lo[row]));
        d.v[row] = mkl_rot_row(s, row, false, d0);
    }
}

static inline void mkl_xf_backward(constant MklPackedSurface& s, thread MklDv& p, thread MklDv& d) {
    if ((s.flags & MKL_SURF_AXIAL) != 0u) {
        p.v[2] = mkl_df_add(p.v[2], float2(s.to_next_origin[2], s.to_next_origin_lo[2]));
        return;
    }
    MklDv p0 = p, d0 = d;
    for (uint row = 0; row < 3u; ++row) {
        p.v[row] = mkl_df_add(mkl_rot_row(s, row, true, p0),
                              float2(s.to_next_origin[row], s.to_next_origin_lo[row]));
        d.v[row] = mkl_rot_row(s, row, true, d0);
    }
}

static inline float mkl_discriminant(float a, float b, float c) {
    float w = a * c;
    float e = fma(-a, c, w);
    float ff = fma(b, b, -w);
    return ff + mkl_df_low(e);
}

static inline float3 mkl_at(float t, float3 d, float3 q) {
    return float3(fma(t, d.x, q.x), fma(t, d.y, q.y), fma(t, d.z, q.z));
}

static inline bool mkl_on_sheet(float kap, float c, float3 pt) {
    return 1.0f - kap * c * pt.z >= -MKL_SHEET_TOL;
}

// Conic steps 2-5 of mklight_trace::surface from q; false for a miss.
static inline bool mkl_conic_from(float c, float k, float3 q, float3 d,
                                  thread float& t_out, thread float3& p_out) {
    float kap = 1.0f + k;
    float a = c * (fma(d.x, d.x, d.y * d.y) + kap * d.z * d.z);
    float b = c * (fma(q.x, d.x, q.y * d.y) + kap * q.z * d.z) - d.z;
    float cc = c * (fma(q.x, q.x, q.y * q.y) + kap * q.z * q.z) - 2.0f * q.z;
    float disc = mkl_discriminant(a, b, cc);
    if (isnan(disc) || disc < 0.0f) return false;
    float sq = sqrt(disc);
    float qq = -(b + (b >= 0.0f ? sq : -sq));
    bool have = false;
    float bt = 0.0f;
    float3 bp = float3(0.0f);
    if (qq != 0.0f) {
        float t = cc / qq;
        float3 pt = mkl_at(t, d, q);
        if (isfinite(t) && mkl_finite3(pt) && mkl_on_sheet(kap, c, pt)) {
            have = true;
            bt = t;
            bp = pt;
        }
    } else if (cc == 0.0f) {
        have = true;
        bt = 0.0f;
        bp = q;
    }
    if (a != 0.0f && qq != 0.0f) {
        float t = qq / a;
        float3 pt = mkl_at(t, d, q);
        if (isfinite(t) && mkl_finite3(pt) && mkl_on_sheet(kap, c, pt)) {
            float r2 = fma(pt.x, pt.x, pt.y * pt.y);
            bool keep = have && fma(bp.x, bp.x, bp.y * bp.y) <= r2;
            if (!keep) {
                have = true;
                bt = t;
                bp = pt;
            }
        }
    }
    t_out = bt;
    p_out = bp;
    return have;
}

// (P(r^2), dP/d(r^2)) of the packed even polynomial.
static inline float2 mkl_asphere_poly(constant MklPackedSurface& s, float r2) {
    uint n = min(s.num_asphere, MKL_MAX_ASPHERE);
    if (n == 0u) return float2(0.0f);
    float rho2 = r2 * s.asphere_inv_norm2;
    float p = s.asphere[n - 1u];
    float dp = s.asphere[n - 1u] * float(n + 1u);
    for (int j = int(n) - 2; j >= 0; --j) {
        p = fma(p, rho2, s.asphere[j]);
        dp = fma(dp, rho2, s.asphere[j] * float(j + 2));
    }
    float rho4 = rho2 * rho2;
    return float2(p * rho4, dp * rho2 * s.asphere_inv_norm2);
}

// P(r^2) in double-float from the packed hi + lo coefficients.
static inline float2 mkl_asphere_poly_df(constant MklPackedSurface& s, float2 r2) {
    uint n = min(s.num_asphere, MKL_MAX_ASPHERE);
    if (n == 0u) return float2(0.0f);
    float2 rho2 = mkl_df_mul_f32(r2, s.asphere_inv_norm2);
    float2 p = float2(s.asphere[n - 1u], s.asphere_lo[n - 1u]);
    for (int j = int(n) - 2; j >= 0; --j) {
        p = mkl_df_add(mkl_df_mul(p, rho2), float2(s.asphere[j], s.asphere_lo[j]));
    }
    return mkl_df_mul(p, mkl_df_mul(rho2, rho2));
}

static inline bool mkl_asphere_domain(constant MklPackedSurface& s, float3 p) {
    float c = s.curvature;
    float kap = 1.0f + s.conic;
    float r2 = fma(p.x, p.x, p.y * p.y);
    float arg = fma(-kap * c * c, r2, 1.0f);
    return !(isnan(arg) || arg < 0.0f);
}

// Newton on the sag residual from q; 0 and t_out on success, else the
// MKL_RAY_* failure code.
static inline uint mkl_asphere_from(constant MklPackedSurface& s, float3 q, float3 d,
                                    constant MklTraceParams& p, thread float& t_out) {
    float c = s.curvature, k = s.conic;
    float st;
    float3 sp;
    bool seed_hit = mkl_conic_from(c, k, q, d, st, sp);
    float t = seed_hit ? st : 0.0f;
    float kap = 1.0f + k;
    float g_first = 0.0f;
    bool sign_changed = false;
    float prev_abs_g = INFINITY;
    for (uint it = 0;; ++it) {
        float x = fma(t, d.x, q.x);
        float y = fma(t, d.y, q.y);
        float z = fma(t, d.z, q.z);
        float r2 = fma(x, x, y * y);
        float arg = fma(-kap * c * c, r2, 1.0f);
        if (isnan(arg) || arg < 0.0f) {
            return (!seed_hit && !sign_changed) ? MKL_RAY_MISS : MKL_RAY_OUTSIDE_DOMAIN;
        }
        float w = sqrt(arg);
        float2 pp = mkl_asphere_poly(s, r2);
        float base = c * r2 / (1.0f + w);
        float g = z - (base + pp.x);
        uint breakdown = sign_changed ? MKL_RAY_NOT_CONVERGED : MKL_RAY_MISS;
        if (!isfinite(g)) return breakdown;
        if (it == 0u) {
            g_first = g;
        } else if (g != 0.0f && (g > 0.0f) != (g_first > 0.0f)) {
            sign_changed = true;
            breakdown = MKL_RAY_NOT_CONVERGED;
        }
        float slope_w = fma(2.0f * w, pp.y, c);
        float gp = d.z - slope_w / max(w, MKL_FLT_MIN) * fma(x, d.x, y * d.y);
        bool gp_ok = isfinite(gp) && gp != 0.0f;
        float tol = p.tolerance_abs
                  + p.tolerance_rel * (((fabs(t) + fabs(z)) + fabs(base)) + fabs(pp.x));
        if (fabs(g) <= tol) {
            if (gp_ok) t = t - g / gp;
            t_out = t;
            return mkl_asphere_domain(s, mkl_at(t, d, q)) ? MKL_RAY_OK : MKL_RAY_OUTSIDE_DOMAIN;
        }
        if (it == p.max_iterations) {
            bool converging = fabs(g) <= 0.5f * prev_abs_g;
            bool cap_fails = (p.flags & MKL_TRACE_NEWTON_CAP_NOT_CONVERGED) != 0u;
            return (converging || cap_fails) ? MKL_RAY_NOT_CONVERGED : breakdown;
        }
        if (!gp_ok) return breakdown;
        t = t - g / gp;
        if (!isfinite(t)) return breakdown;
        prev_abs_g = fabs(g);
    }
}

static inline MklDv mkl_point_on(MklDv q, float2 t, MklDv d) {
    MklDv x;
    for (int k = 0; k < 3; ++k) x.v[k] = mkl_df_add(q.v[k], mkl_df_mul(d.v[k], t));
    return x;
}

static inline float2 mkl_refine_t(float t1, float r, float rp) {
    float step = r / rp;
    if (rp != 0.0f && isfinite(step)) return mkl_two_sum(t1, -step);
    return float2(t1, 0.0f);
}

// mklight_trace::surface::intersect from the double-float point p along d.
static inline uint mkl_intersect_legacy(constant MklPackedSurface& s, uint shape, MklDv p, MklDv d,
                                 constant MklTraceParams& prm, thread MklHit& hit) {
    MklDv zg;
    zg.v[0] = float2(0.0f);
    zg.v[1] = float2(0.0f);
    zg.v[2] = float2(1.0f, 0.0f);
    float3 zn = float3(0.0f, 0.0f, 1.0f);
    if (shape == MKL_SHAPE_COORD_BREAK) {
        hit.ta = float2(0.0f);
        hit.tb = float2(0.0f);
        hit.x = p;
        hit.g = zg;
        hit.normal = zn;
        return MKL_RAY_OK;
    }
    bool conic_like = (shape == MKL_SHAPE_CONIC && s.curvature != 0.0f)
        || (shape == MKL_SHAPE_EVEN_ASPHERE && (s.num_asphere > 0u || s.curvature != 0.0f));
    if (!conic_like) {
        if (d.v[2].x == 0.0f) return MKL_RAY_MISS;
        float2 t = mkl_df_div(mkl_df_neg(p.v[2]), d.v[2]);
        MklDv x;
        x.v[0] = mkl_df_add(p.v[0], mkl_df_mul(d.v[0], t));
        x.v[1] = mkl_df_add(p.v[1], mkl_df_mul(d.v[1], t));
        x.v[2] = float2(0.0f);
        if (!(isfinite(x.v[0].x) && isfinite(x.v[1].x))) return MKL_RAY_MISS;
        hit.ta = t;
        hit.tb = float2(0.0f);
        hit.x = x;
        hit.g = zg;
        hit.normal = zn;
        return MKL_RAY_OK;
    }
    float2 t0 = mkl_df_neg(mkl_ddot(p, d));
    MklDv q;
    for (int k = 0; k < 3; ++k) q.v[k] = mkl_df_add(p.v[k], mkl_df_mul(d.v[k], t0));
    float3 qf = mkl_hi3(q);
    float3 dfv = mkl_hi3(d);
    hit.ta = t0;
    float c = s.curvature, k = s.conic;
    float kap = 1.0f + k;
    float2 cd = float2(s.curvature, s.curvature_lo);
    float2 kapd = mkl_df_add_f32(float2(s.conic, s.conic_lo), 1.0f);
    float2 kcd = mkl_df_mul(kapd, cd);
    if (shape == MKL_SHAPE_EVEN_ASPHERE && s.num_asphere > 0u) {
        float t1;
        uint code = mkl_asphere_from(s, qf, dfv, prm, t1);
        if (code != MKL_RAY_OK) return code;
        MklDv x0 = mkl_point_on(q, float2(t1, 0.0f), d);
        float2 r2 = mkl_df_add(mkl_df_mul(x0.v[0], x0.v[0]), mkl_df_mul(x0.v[1], x0.v[1]));
        float2 w = mkl_df_sqrt(mkl_df_sub(float2(1.0f, 0.0f), mkl_df_mul(r2, mkl_df_mul(kcd, cd))));
        float2 csag = mkl_df_div(mkl_df_mul(r2, cd), mkl_df_add_f32(w, 1.0f));
        float2 pp = mkl_asphere_poly(s, r2.x);
        float2 res = mkl_df_sub(mkl_df_sub(x0.v[2], csag), mkl_asphere_poly_df(s, r2));
        float3 xh = mkl_hi3(x0);
        float wh = w.x;
        float slope_w = fma(2.0f * wh, pp.y, c);
        float gp = dfv.z - slope_w / max(wh, MKL_FLT_MIN) * fma(xh.x, dfv.x, xh.y * dfv.y);
        float2 tb = mkl_refine_t(t1, res.x, gp);
        MklDv x = mkl_point_on(q, tb, d);
        float2 gs = mkl_df_add(mkl_df_mul_f32(mkl_df_mul_f32(w, pp.y), 2.0f), cd);
        hit.tb = tb;
        hit.x = x;
        hit.g.v[0] = mkl_df_neg(mkl_df_mul(x.v[0], gs));
        hit.g.v[1] = mkl_df_neg(mkl_df_mul(x.v[1], gs));
        hit.g.v[2] = w;
    } else {
        float t1;
        float3 pt;
        if (!mkl_conic_from(c, k, qf, dfv, t1, pt)) return MKL_RAY_MISS;
        MklDv x0 = mkl_point_on(q, float2(t1, 0.0f), d);
        float2 r2 = mkl_df_add(mkl_df_mul(x0.v[0], x0.v[0]), mkl_df_mul(x0.v[1], x0.v[1]));
        float2 fv = mkl_df_sub(
            mkl_df_mul(mkl_df_add(r2, mkl_df_mul(mkl_df_mul(x0.v[2], x0.v[2]), kapd)), cd),
            mkl_df_mul_f32(x0.v[2], 2.0f));
        float3 xh = mkl_hi3(x0);
        float fp = 2.0f * (c * (fma(xh.x, dfv.x, xh.y * dfv.y) + kap * xh.z * dfv.z) - dfv.z);
        float2 tb = mkl_refine_t(t1, fv.x, fp);
        MklDv x = mkl_point_on(q, tb, d);
        hit.tb = tb;
        hit.x = x;
        hit.g.v[0] = mkl_df_neg(mkl_df_mul(x.v[0], cd));
        hit.g.v[1] = mkl_df_neg(mkl_df_mul(x.v[1], cd));
        hit.g.v[2] = mkl_df_sub(float2(1.0f, 0.0f), mkl_df_mul(x.v[2], kcd));
    }
    hit.normal = mkl_normalized(mkl_hi3(hit.g));
    return MKL_RAY_OK;
}

// Safeguarded Newton on z - sag from the conic seed (trace_kernel::sag_from):
// the reference's bracket, backtracking and Miss / NoConvergence /
// OutsideDomain decisions in f32.
template<bool OVERLAYS> static inline uint mkl_sag_from(constant MklPackedSurface& s,constant float* pool,float3 q,float3 d,constant MklTraceParams& prm,thread float& t) {
    float3 seed_point;
    bool seed_hit=mkl_conic_from(s.curvature,s.conic,q,d,t,seed_point);
    if (!seed_hit) t=0.0f;
    bool have_pos=false,have_neg=false; float positive=0.0f,negative=0.0f,previous=INFINITY;
    for (uint it=0;it<=prm.max_iterations;++it) {
        bool bracketed=have_pos&&have_neg;
        uint left_domain=!seed_hit&&!bracketed?MKL_RAY_MISS:MKL_RAY_OUTSIDE_DOMAIN;
        float3 pt=mkl_at(t,d,q); MklJet<float> v;
        if (!mkl_surface_eval<OVERLAYS>(s,pool,pt.x,pt.y,v)) return left_domain;
        float g=pt.z-v.v,gp=d.z-v.x*d.x-v.y*d.y;
        if (g>0.0f) {positive=t;have_pos=true;} else if (g<0.0f) {negative=t;have_neg=true;}
        bool bracket=have_pos&&have_neg; float lo=min(positive,negative),hi=max(positive,negative);
        uint failure=bracket?MKL_RAY_NOT_CONVERGED:MKL_RAY_MISS;
        float tol=prm.tolerance_abs+prm.tolerance_rel*((fabs(t)+fabs(pt.z))+fabs(v.v));
        if (fabs(g)<=tol) {
            float3 corrected=mkl_at(t-g/gp,d,q); MklJet<float> vc;
            if (mkl_surface_eval<OVERLAYS>(s,pool,corrected.x,corrected.y,vc)) t=t-g/gp;
            return MKL_RAY_OK;
        }
        if (it==prm.max_iterations) return fabs(g)<=0.5f*previous || (prm.flags&MKL_TRACE_NEWTON_CAP_NOT_CONVERGED)!=0u ? MKL_RAY_NOT_CONVERGED:failure;
        if (!isfinite(g)||gp==0.0f||!isfinite(gp)) return failure;
        float next=t-g/gp;
        if (bracket&&!(next>lo&&next<hi)) next=lo+0.5f*(hi-lo);
        bool outside=false,accepted=false;
        for (uint j=0;j<20;++j) {
            float3 p=mkl_at(next,d,q); MklJet<float> vn;
            if (mkl_surface_eval<OVERLAYS>(s,pool,p.x,p.y,vn)) {
                if (bracket||fabs(p.z-vn.v)<fabs(g)) {accepted=true;break;}
            } else outside=true;
            next=t+0.5f*(next-t);
        }
        if (!accepted) return outside&&!bracket?left_domain:failure;
        previous=fabs(g); t=next;
    }
    return MKL_RAY_NOT_CONVERGED;
}
template<bool OVERLAYS> static inline uint mkl_intersect(constant MklPackedSurface& s,constant float* pool,MklDv p,MklDv d,constant MklTraceParams& prm,thread MklHit& hit) {
    bool overlay=OVERLAYS && (s.flags&MKL_SURF_HAS_SAG_OVERLAY)!=0u;
    if (s.shape<=MKL_SHAPE_COORD_BREAK && !overlay) return mkl_intersect_legacy(s,s.shape,p,d,prm,hit);
    constant float* data=mkl_surface_data(s,pool);
    if (s.shape==MKL_SHAPE_DIFFRACTIVE && !overlay) {
        uint code=mkl_intersect_legacy(s,uint(data[2*MKL_COEFF_KIND]),p,d,prm,hit);
        if (code!=MKL_RAY_OK) return code;
        return mkl_surface_domain(s,pool,MklDf{hit.x.v[0]},MklDf{hit.x.v[1]})?MKL_RAY_OK:MKL_RAY_OUTSIDE_DOMAIN;
    }
    if (!overlay && data[2*MKL_COEFF_EXACT_CONIC]!=0.0f) {
        uint code=mkl_intersect_legacy(s,MKL_SHAPE_CONIC,p,d,prm,hit);
        if (code!=MKL_RAY_OK) return code;
        MklJet<MklDf> v;
        return mkl_surface_eval<OVERLAYS>(s,pool,MklDf{hit.x.v[0]},MklDf{hit.x.v[1]},v)?MKL_RAY_OK:MKL_RAY_OUTSIDE_DOMAIN;
    }
    float2 ta=mkl_df_neg(mkl_ddot(p,d)); MklDv q=mkl_point_on(p,ta,d);
    float t; uint code=mkl_sag_from<OVERLAYS>(s,pool,mkl_hi3(q),mkl_hi3(d),prm,t);
    if (code!=MKL_RAY_OK) return code;
    MklDf tb={float2(t,0)};
    for (uint i=0;i<2;++i) {
        MklDv x=mkl_point_on(q,tb.v,d); MklJet<MklDf> v;
        if (!mkl_surface_eval<OVERLAYS>(s,pool,MklDf{x.v[0]},MklDf{x.v[1]},v)) return MKL_RAY_OUTSIDE_DOMAIN;
        MklDf gp=MklDf{d.v[2]}-v.x*MklDf{d.v[0]}-v.y*MklDf{d.v[1]};
        MklDf step=(MklDf{x.v[2]}-v.v)/gp;
        if (isfinite(step.v.x)) tb=tb-step;
    }
    hit.x=mkl_point_on(q,tb.v,d); MklJet<MklDf> v;
    if (!mkl_surface_eval<OVERLAYS>(s,pool,MklDf{hit.x.v[0]},MklDf{hit.x.v[1]},v)) return MKL_RAY_OUTSIDE_DOMAIN;
    if (overlay && s.shape==MKL_SHAPE_DIFFRACTIVE && !mkl_surface_domain(s,pool,MklDf{hit.x.v[0]},MklDf{hit.x.v[1]})) return MKL_RAY_OUTSIDE_DOMAIN;
    hit.g.v[0]=mkl_df_neg(v.x.v);hit.g.v[1]=mkl_df_neg(v.y.v);hit.g.v[2]=float2(1,0);
    if (!mkl_finite3(mkl_hi3(hit.g))) return MKL_RAY_OUTSIDE_DOMAIN;
    hit.ta=ta;hit.tb=tb.v;hit.normal=mkl_normalized(mkl_hi3(hit.g)); return MKL_RAY_OK;
}
// mklight_trace::interact::diffract in double-float (trace_kernel::diffract):
// the order's momentum grad projected on the tangent plane of g, then the
// normal component for the hemisphere; false for an evanescent order.
static inline bool mkl_diffract(MklDv k,MklDv g,float2 n1,float2 n2,MklDf gx,MklDf gy,bool reflected,thread MklDv& out) {
    float2 length=mkl_df_sqrt(mkl_ddot(g,g)); MklDv normal,grad,tangent;
    for (uint i=0;i<3;++i) normal.v[i]=mkl_df_div(g.v[i],length);
    MklDf kn={mkl_ddot(k,normal)};
    grad.v[0]=gx.v;grad.v[1]=gy.v;grad.v[2]=float2(0);
    MklDf gn={mkl_ddot(grad,normal)},inv=mkl_c<MklDf>(1)/MklDf{n2};
    for (uint i=0;i<3;++i) tangent.v[i]=(((MklDf{k.v[i]}-MklDf{normal.v[i]}*kn)*MklDf{n1}+MklDf{grad.v[i]}-MklDf{normal.v[i]}*gn)*inv).v;
    float2 disc=mkl_df_sub(float2(1,0),mkl_ddot(tangent,tangent));
    if (disc.x<0.0f) return false;
    float sign=(kn.v.x<0.0f)!=reflected?-1.0f:1.0f;
    float2 root=mkl_df_mul_f32(mkl_df_sqrt(disc),sign);
    for (uint i=0;i<3;++i) out.v[i]=mkl_df_add(tangent.v[i],mkl_df_mul(normal.v[i],root));
    return true;
}

// Stable vector Snell in f32: false on TIR; cos i and the signed coefficient
// gamma of `normal` in k' = mu k + gamma normal.
static inline bool mkl_refract(float3 k, float3 normal, float n1, float n2,
                               thread float& cos_out, thread float& gamma_out) {
    float cos_i = mkl_dot3(k, normal);
    bool flip = cos_i < 0.0f;
    if (flip) cos_i = -cos_i;
    cos_i = min(cos_i, 1.0f);
    float mu = n1 / n2;
    float sin2 = (1.0f - cos_i) * (1.0f + cos_i);
    float disc = fma(-mu * mu, sin2, 1.0f);
    if (disc < 0.0f) return false;
    float root = sqrt(disc);
    float denom = fma(mu, cos_i, root);
    float gamma = denom > 0.0f ? ((n2 - n1) * (n2 + n1)) / (n2 * n2) / denom : root - mu * cos_i;
    cos_out = cos_i;
    gamma_out = flip ? -gamma : gamma;
    return true;
}

static inline float3 mkl_reflect(float3 k, float3 n, thread float& cos_out) {
    float kn = mkl_dot3(k, n);
    float s = -2.0f * kn;
    cos_out = min(fabs(kn), 1.0f);
    return float3(fma(s, n.x, k.x), fma(s, n.y, k.y), fma(s, n.z, k.z));
}

static inline float2 mkl_dnorm(MklDv v) { return mkl_df_sqrt(mkl_ddot(v, v)); }

static inline float2 mkl_dabs(float2 x) { return x.x < 0.0f ? mkl_df_neg(x) : x; }

// sphere_term in double-float for X = (x0, x1, 0).
static inline float2 mkl_sphere_term(float2 x0, float2 x1, MklDv t, float2 rho) {
    float2 tn = mkl_dnorm(t);
    float2 xx = mkl_df_add(mkl_df_mul(x1, x1), mkl_df_mul(x0, x0));
    float2 tx = mkl_df_add(mkl_df_mul(t.v[1], x1), mkl_df_mul(t.v[0], x0));
    if (fabs(rho.x) * sqrt(xx.x) <= tn.x) {
        MklDv u;
        u.v[0] = mkl_df_sub(t.v[0], mkl_df_mul(x0, rho));
        u.v[1] = mkl_df_sub(t.v[1], mkl_df_mul(x1, rho));
        u.v[2] = t.v[2];
        float2 den = mkl_df_add(mkl_dnorm(u), tn);
        return mkl_df_div(mkl_df_sub(mkl_df_mul(rho, xx), mkl_df_mul_f32(tx, 2.0f)), den);
    }
    float2 a = mkl_df_div(float2(1.0f, 0.0f), rho);
    MklDv o;
    o.v[0] = mkl_df_sub(x0, mkl_df_mul(t.v[0], a));
    o.v[1] = mkl_df_sub(x1, mkl_df_mul(t.v[1], a));
    o.v[2] = mkl_df_neg(mkl_df_mul(t.v[2], a));
    float2 num = mkl_df_sub(
        mkl_df_mul_f32(mkl_df_add(mkl_df_mul(o.v[1], x1), mkl_df_mul(o.v[0], x0)), 2.0f), xx);
    float2 q = mkl_df_div(num, mkl_df_add(mkl_dnorm(o), mkl_df_mul(mkl_dabs(a), tn)));
    return a.x < 0.0f ? mkl_df_neg(q) : q;
}

// The ideal lens in double-float: direction d_out and phase delta.
static inline bool mkl_ideal_lens(float2 px, float2 py, MklDv k, float2 phi, float2 n1,
                                  float2 n2, float2 vergence, thread MklDv& d_out,
                                  thread float2& delta) {
    if (k.v[2].x == 0.0f) return false;
    float2 sz = float2(copysign(1.0f, k.v[2].x), 0.0f);
    float2 az = mkl_dabs(k.v[2]);
    float2 r = mkl_df_div(n1, n2);
    MklDv t;
    t.v[0] = mkl_df_div(k.v[0], az);
    t.v[1] = mkl_df_div(k.v[1], az);
    t.v[2] = sz;
    float2 tx2 = mkl_df_sub(mkl_df_mul(r, t.v[0]), mkl_df_mul(px, phi));
    float2 ty2 = mkl_df_sub(mkl_df_mul(r, t.v[1]), mkl_df_mul(py, phi));
    MklDv t_out;
    t_out.v[0] = tx2;
    t_out.v[1] = ty2;
    t_out.v[2] = sz;
    float2 inv = mkl_df_div(float2(1.0f, 0.0f), mkl_dnorm(t_out));
    d_out.v[0] = mkl_df_mul(tx2, inv);
    d_out.v[1] = mkl_df_mul(ty2, inv);
    d_out.v[2] = mkl_df_mul(sz, inv);
    float2 rho_b = mkl_df_sub(phi, mkl_df_mul(r, vergence));
    delta = mkl_df_sub(mkl_df_mul(n1, mkl_sphere_term(px, py, t, vergence)),
                       mkl_df_mul(n2, mkl_sphere_term(px, py, t_out, mkl_df_neg(rho_b))));
    return true;
}

// Unpolarised (R, T) of a bare interface.
static inline float2 mkl_fresnel(float n1, float n2, float cos_i) {
    float ci = min(max(cos_i, 0.0f), 1.0f);
    float sin2_i = (1.0f - ci) * (1.0f + ci);
    float rr = n1 / n2;
    float sin2_t = rr * rr * sin2_i;
    if (sin2_t >= 1.0f) return float2(1.0f, 0.0f);
    float ct = sqrt(max(1.0f - sin2_t, 0.0f));
    float a = n1 * ci, b = n2 * ct;
    float rs = (a + b == 0.0f) ? 1.0f : (a - b) / (a + b);
    float c = n2 * ci, dd = n1 * ct;
    float rp = (c + dd == 0.0f) ? 1.0f : (c - dd) / (c + dd);
    float rs2 = rs * rs, rp2 = rp * rp;
    return float2(0.5f * (rs2 + rp2), 0.5f * ((1.0f - rs2) + (1.0f - rp2)));
}

// Unpolarised (R, T) at a surface for incidence cosine cos_i
// (trace_kernel::trace_ray `rt`; never a transmission at a mirror).
static inline float2 mkl_surface_rt(constant MklPackedSurface& s, bool mirror, float n1, float n2, float cos_i) {
    if (s.coating == MKL_COATING_IDEAL) return float2(s.coating_r, s.coating_t);
    if (mirror) return float2(1.0f, 0.0f);
    return mkl_fresnel(n1, n2, cos_i);
}

// Aperture test in double-float with the packed sizes' residuals.
static inline bool mkl_passes_aperture(constant MklPackedSurface& s, float2 x, float2 y) {
    float2 a0 = float2(s.aperture_params[0], s.aperture_params_lo[0]);
    float2 a1 = float2(s.aperture_params[1], s.aperture_params_lo[1]);
    switch (s.aperture) {
        case MKL_APERTURE_CIRCULAR: {
            float2 r2 = mkl_df_add(mkl_df_mul(x, x), mkl_df_mul(y, y));
            return mkl_df_sub(r2, mkl_df_mul(a0, a0)).x <= 0.0f;
        }
        case MKL_APERTURE_ANNULAR: {
            float2 r2 = mkl_df_add(mkl_df_mul(x, x), mkl_df_mul(y, y));
            return mkl_df_sub(r2, mkl_df_mul(a0, a0)).x >= 0.0f
                && mkl_df_sub(r2, mkl_df_mul(a1, a1)).x <= 0.0f;
        }
        case MKL_APERTURE_RECTANGULAR:
            return mkl_df_sub(mkl_dabs(x), a0).x <= 0.0f && mkl_df_sub(mkl_dabs(y), a1).x <= 0.0f;
        case MKL_APERTURE_ELLIPTICAL: {
            float2 u = mkl_df_div(x, a0);
            float2 v = mkl_df_div(y, a1);
            return mkl_df_add_f32(mkl_df_add(mkl_df_mul(u, u), mkl_df_mul(v, v)), -1.0f).x <= 0.0f;
        }
        default: return true;
    }
}

static inline MklState mkl_report(MklDv p, MklDv d, float2 opl, float2 n, float weight,
                                  uint status) {
    float corr = n.x * mkl_dot3(mkl_lo3(p), mkl_hi3(d));
    MklState st;
    st.pos = mkl_hi3(p);
    st.dir = mkl_hi3(d);
    st.opl = mkl_df_add_f32(opl, -corr);
    st.weight = weight;
    st.status = status;
    return st;
}

// Element i of plane k of `stride` values, in 64-bit arithmetic: record
// buffers (slots · stride · 9 floats) and input planes of large batches
// exceed 2^32 elements.
static inline ulong mkl_plane(uint k, uint stride, uint i) { return ulong(k) * ulong(stride) + ulong(i); }

static inline void mkl_record(device float* rf, device uint* ru, uint slot, uint stride, uint i,
                              MklState st) {
    device float* o = rf + mkl_plane(slot, stride, i) * ulong(MKL_TRACE_RECORD_FLOATS);
    o[0] = st.pos.x;
    o[1] = st.pos.y;
    o[2] = st.pos.z;
    o[3] = st.dir.x;
    o[4] = st.dir.y;
    o[5] = st.dir.z;
    o[6] = st.opl.x;
    o[7] = st.opl.y;
    o[8] = st.weight;
    ru[mkl_plane(slot, stride, i)] = st.status;
}

static inline float2 mkl_index(constant MklPackedSurface& s, uint w) {
    return mkl_df_input(float2(s.n_after[w], s.n_after_lo[w]));
}

// Resident trace core (trace_kernel::trace_resident): unrounded hi+lo
// position, direction and OPL, the unnormalised normal and the medium index
// after every step. `rec.write(slot, state)` receives slot 0 (the input),
// slot k+1 after step k, and the final state repeated for skipped slots of a
// failed ray. steps == nullptr runs the sequential program start+1..start+n.
struct MklRawState {
    MklDv pos, dir, normal;
    float2 opl, index;
    float weight, transmittance;
    uint status, last, done;
};
struct MklNoRecords { void write(uint, MklRawState) {} };
template<bool OVERLAYS, typename Recorder>
static __attribute__((always_inline)) MklRawState mkl_trace_raw(constant MklPackedHeader& h,
    constant MklPackedSurface* s, constant float* coefficients,
    constant uint* steps, constant MklTraceParams& prm, uint num_steps, uint w,
    MklRawState input, thread Recorder& rec) {
    const bool clip = (prm.flags & MKL_TRACE_CLIP) != 0u;
    const bool radiometry = (prm.flags & MKL_TRACE_RADIOMETRY) != 0u;
    const bool at_infinity = (h.flags & MKL_SYS_OBJECT_AT_INFINITY) != 0u;

    MklDv pos = input.pos, dir = input.dir;
    for (uint k = 0; k < 3; ++k) {
        pos.v[k] = mkl_df_input(pos.v[k]);
        dir.v[k] = mkl_df_input(dir.v[k]);
    }
    float2 opl = mkl_df_input(input.opl);
    const float w_in = input.weight;
    float tw = 1.0f;
    float absorb = 0.0f;
    const uint start = prm.start_surface;
    float2 n_cur = mkl_index(s[start], w);
    uint last = start;
    uint status = MKL_RAY_OK;
    uint done = 0u;
    MklDv normal = {};

    rec.write(0u, MklRawState{pos, dir, normal, opl, n_cur, w_in, 1.0f, MKL_RAY_OK, last, 0u});
    if (mkl_finite3(mkl_hi3(pos)) && mkl_finite3(mkl_hi3(dir)) && mkl_finite3(mkl_lo3(pos))
        && mkl_finite3(mkl_lo3(dir))) {
        dir = mkl_renorm(dir);
    } else {
        status = MKL_RAY_NONFINITE | (start << MKL_RAY_SURFACE_SHIFT);
    }

    for (uint k = 0u; status == MKL_RAY_OK && k < num_steps; ++k) {
        const uint word = steps ? steps[k] : start + k + 1u;
        const uint si = word & MKL_STEP_SURFACE_MASK;
        const uint mode = (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK;
        const bool back = (word & MKL_STEP_BACKWARD) != 0u;
        const uint gap = back ? si : si - 1u;
        const float2 n1 = mkl_index(s[gap], w);
        const float alpha = (gap == 0u && at_infinity) ? 0.0f : s[gap].alpha_after[w];
        const float2 n_other = back ? (si >= 1u ? mkl_index(s[si - 1u], w) : n1) : mkl_index(s[si], w);
        // 1. Transfer into the surface's frame.
        if (back) {
            mkl_xf_backward(s[si], pos, dir);
        } else {
            mkl_xf_forward(s[si - 1u], pos, dir);
        }
        constant MklPackedSurface& surf = s[si];
        last = si;
        n_cur = n1;
        const uint fail_s = si << MKL_RAY_SURFACE_SHIFT;
        // 2. Intersect; optical path and absorption of the segment.
        MklHit hit;
        uint code = mkl_intersect<OVERLAYS>(surf, coefficients, pos, dir, prm, hit);
        if (code != MKL_RAY_OK) {
            status = code | fail_s;
            break;
        }
        opl = mkl_df_add(opl, mkl_df_mul(hit.ta, n1));
        opl = mkl_df_add(opl, mkl_df_mul(hit.tb, n1));
        if (alpha != 0.0f) absorb = fma(alpha, hit.ta.x + hit.tb.x, absorb);
        pos = hit.x;
        normal = hit.g;
        if (!mkl_finite3(mkl_hi3(pos))) {
            status = MKL_RAY_NONFINITE | fail_s;
            break;
        }
        // 3. Clip.
        const bool is_cb = surf.shape == MKL_SHAPE_COORD_BREAK;
        if (clip && !is_cb && !mkl_passes_aperture(surf, pos.v[0], pos.v[1])) {
            status = MKL_RAY_VIGNETTED | fail_s;
            break;
        }
        // 4. Interact (0 none, 1 refract, 2 reflect, 3 lens, 4 reflecting lens).
        // A DOE's selected nonzero order replaces 1 and 2 (trace_kernel's
        // Act::Diffract), tested first: faster than a separate act code.
        const bool is_image = (surf.flags & MKL_SURF_IMAGE) != 0u;
        const bool paraxial = surf.shape == MKL_SHAPE_PARAXIAL;
        const bool mirror = surf.interaction == MKL_INTERACT_REFLECT;
        const bool dummy = surf.interaction == MKL_INTERACT_DUMMY;
        uint act = 0u;
        if (!is_cb) {
            if (mode == MKL_STEP_SEQUENTIAL) {
                act = (is_image || dummy) ? 0u : paraxial ? (mirror ? 4u : 3u) : mirror ? 2u : 1u;
            } else if (mode == MKL_STEP_TRANSMIT) {
                act = (is_image || dummy) ? 0u : paraxial ? 3u : 1u;
            } else {
                act = (paraxial && mirror) ? 4u : 2u;
            }
        }
        const uint coating = surf.coating;
        const float3 dfv = mkl_hi3(dir);
        if ((act == 1u || act == 2u) && surf.shape == MKL_SHAPE_DIFFRACTIVE && !dummy && !is_image &&
            mkl_surface_data(surf, coefficients)[2 * MKL_COEFF_TRACE_ORDER] != 0.0f) {
            // m lambda/(2 pi) scales grad Phi to momentum and Phi to optical path.
            const MklDf scale = mkl_load<MklDf>(mkl_surface_data(surf, coefficients), MKL_COEFF_PREFIX + w);
            const MklJet<MklDf> v = mkl_phase(surf, coefficients, MklDf{pos.v[0]}, MklDf{pos.v[1]});
            const bool refl = act == 2u;
            const float2 n2 = refl ? n1 : n_other;
            MklDv d2;
            if (!mkl_diffract(dir, hit.g, n1, n2, v.x * scale, v.y * scale, refl, d2)) {
                status = MKL_RAY_EVANESCENT | fail_s;
                break;
            }
            dir = d2;
            n_cur = n2;
            opl = mkl_df_add(opl, (scale * v.v).v);
            if (radiometry) {
                float cos_i;
                mkl_reflect(dfv, hit.normal, cos_i);
                const float2 rt = mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i);
                tw = tw * (refl ? rt.x : rt.y);
            }
        } else if (act == 1u) {
            float cos_i, gamma;
            if (!mkl_refract(dfv, hit.normal, n1.x, n_other.x, cos_i, gamma)) {
                status = MKL_RAY_TIR | fail_s;
                break;
            }
            float2 mu = mkl_df_div(n1, n_other);
            float b = gamma / mkl_norm3(mkl_hi3(hit.g));
            MklDv d2;
            for (int c = 0; c < 3; ++c) {
                d2.v[c] = mkl_df_add(mkl_df_mul(dir.v[c], mu), mkl_df_mul_f32(hit.g.v[c], b));
            }
            dir = mkl_renorm_along(mkl_renorm_along(d2, hit.g), hit.g);
            n_cur = n_other;
            if (radiometry) tw = tw * mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i).y;
        } else if (act == 2u) {
            float cos_i;
            mkl_reflect(dfv, hit.normal, cos_i);
            float3 gh = mkl_hi3(hit.g);
            float b = -2.0f * mkl_dot3(dfv, gh) / mkl_dot3(gh, gh);
            MklDv d2;
            for (int c = 0; c < 3; ++c) d2.v[c] = mkl_df_add(dir.v[c], mkl_df_mul_f32(hit.g.v[c], b));
            dir = mkl_renorm_along(d2, hit.g);
            if (radiometry) tw = tw * mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i).x;
        } else if (act >= 3u) {
            const bool refl = act == 4u;
            MklDv kk = dir;
            if (refl) kk.v[2] = mkl_df_neg(kk.v[2]);
            const float2 n2 = refl ? n1 : n_other;
            const float2 verg = back ? float2(0.0f) : float2(surf.vergence[w], surf.vergence_lo[w]);
            const float2 phi = float2(surf.paraxial_power, surf.paraxial_power_lo);
            MklDv d2;
            float2 delta;
            if (!mkl_ideal_lens(pos.v[0], pos.v[1], kk, phi, n1, n2, verg, d2, delta)) {
                status = MKL_RAY_DEGENERATE | fail_s;
                break;
            }
            dir = d2;
            n_cur = n2;
            opl = mkl_df_add(opl, delta);
            if (radiometry && coating == MKL_COATING_IDEAL) {
                tw = tw * (refl ? surf.coating_r : surf.coating_t);
            }
        }
        done = k + 1u;
        float wnow = radiometry ? w_in * tw * mkl_exp(-absorb) : w_in;
        rec.write(done, MklRawState{pos, dir, normal, opl, n_cur, wnow, mkl_exp(-absorb), MKL_RAY_OK, last, done});
    }

    const float transmittance = mkl_exp(-absorb);
    const float weight = radiometry ? w_in * tw * transmittance : w_in;
    MklRawState st = {pos, dir, normal, opl, n_cur, weight, transmittance, status, last, done};
    for (uint slot = done + 1u; slot <= num_steps; ++slot) rec.write(slot, st);
    return st;
}
#endif
