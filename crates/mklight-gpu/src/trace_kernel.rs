//! Specification and CPU emulation of the sequential / path trace kernel
//! (`mkl_trace` in `crates/mklight-metal/shaders/trace.metal`).
//!
//! Every function here has an MSL twin (`mkl_` prefix) in `trace.metal`,
//! written operation for operation alike, so the CPU-f32 emulation (generic
//! over the f32 environment [`FpEnv`]) reproduces the GPU bit for bit.
//! Change both sides together.
//!
//! # Algorithm
//!
//! The per-ray algorithm is `mklight_trace::trace_ray` step for step (that
//! crate's `trace`, `surface` and `interact` modules): the same transfer,
//! closest-approach conic intersection with stable roots and vertex-sheet
//! selection, even-asphere Newton iteration with the same Miss /
//! NoConvergence / OutsideDomain classification, clipping, stable vector
//! Snell, reflection, TIR and ideal lens. Decisions (which root, which
//! status) are taken by an **f32 solve** that mirrors the reference
//! operation for operation. The geometry the ray carries from surface to
//! surface is **double-float** (CONVENTIONS §7.4), because f32 alone cannot
//! meet the CONVENTIONS §7 budget over long paths: one f32 ulp of a
//! direction (6e-8) moves the image point by 3e-5 mm over 500 mm, and a
//! direction whose norm is off by 2 ulp lengthens a 100 mm segment by
//! 1.6e-5 mm (0.03 waves). Per step:
//!
//! 1. **Transfer** of the double-float point `P` and direction `D` into the
//!    surface frame: `P − (0, 0, t)` on the axial fast path, else
//!    `rot·P − shift` and `rot·D` with `rot`, `shift` as packed hi + lo.
//! 2. **Closest approach** `t₀ = −P·D`, `Q = P + t₀D` in double-float; `Q`
//!    is O(aperture) and is rounded once (`Q_f`); the residual's component
//!    along the ray moves into the distance, `t₀' = t₀ − (Q − Q_f)·D`.
//! 3. **f32 solve** from `Q_f` along `fl(D)`: exactly the reference's conic
//!    or asphere algorithm (f32 tolerances: vertex sheet [`SHEET_TOL`],
//!    Newton converged at `|g| ≤ tol_abs + tol_rel·(|t| + |z| + |conic sag|
//!    + |poly|)`, the f32 resolution of the residual, within
//!    `max_iterations` steps). It gives the status and `t₁`.
//! 4. **Double-float refinement**: one Newton step on the surface equation
//!    evaluated in double-float at `X = Q_f + t₁D` (`c(x² + y² + κz²) − 2z`
//!    for conics, `z − sag(r²)` with the conic part in double-float for
//!    aspheres), giving `t₁'` (hi + lo) and the hit `X`.
//! 5. **Unnormalised normal in double-float**, `G = (−cx, −cy, 1 − κcz)` or
//!    `(−x·g, −y·g, w)`, `g = c + 2wP'(r²)` (the f32 unit normal `fl(G)/|·|`
//!    serves the f32 decisions).
//! 6. **Interaction in double-float**: refraction `D' = μD + bG`
//!    (`μ = n₁/n₂` from double-float indices) and reflection `D' = D + bG`,
//!    with `b` from the f32 stable Snell / reflection law and then corrected
//!    along `G` so that `|D'| = 1` to O(ε²) — the tangential component, i.e.
//!    Snell's law, is exact by construction. The ideal lens is evaluated in
//!    f32 and its output direction renormalised in double-float. TIR is the
//!    f32 test.
//! 7. **Optical path** `Σ n·(t₀' + t₁')` with double-float `n`, plus the
//!    ideal-lens phase; bulk absorption `Σ α·t` in f32.
//!
//! The **reported state** is `X` and `D` rounded to f32; the rounding
//! residual of `X` along `D` is moved into the reported optical path
//! (`OPL − n·((X − fl(X))·D)`), so `(point, direction, OPL)` describe one ray
//! consistently and an OPD formed from them on the host does not see the
//! rounding of the point.
//!
//! # Path programs
//!
//! A trace runs a list of steps (`mklight_gpu_types.h`, `batch::PathStep`):
//! each moves the ray from the current surface `c` to `c ± 1` and interacts
//! there. The medium of the gap between surfaces `j` and `j + 1` is
//! `n_after[j]`, `alpha_after[j]` (no absorption in gap 0 for an object at
//! infinity). `Sequential` steps reproduce `mklight_trace::trace_ray`;
//! `Transmit` and `Reflect` steps force the interaction (ghost paths);
//! coordinate breaks only change the frame. An ideal lens transmits with
//! its packed object vergence going forward and with zero vergence going
//! backward, and reflects as a plane mirror. Radiometry (unpolarised):
//! bulk absorption, bare Fresnel `T`/`R` on refracting surfaces, `R = 1`,
//! `T = 0` on bare mirrors, constant `R`/`T` for ideal coatings; ideal
//! lenses are lossless unless coated.

use crate::consts::*;
use crate::df::Df32;
use crate::fenv::FpEnv;
use crate::fmath::{exp_in, hypot3_in};
use crate::fx::{D, F, f};
use crate::layout::{MAX_ASPHERE, PackedHeader, PackedSurface, TraceParams};

type V<E> = [F<E>; 3];
type DV<E> = [D<E>; 3];

/// Vertex-sheet tolerance in f32: a point is on the vertex sheet when
/// `1 − κcz ≥ −SHEET_TOL` (2⁻²⁰, about 16 ulp of 1; the f64 reference uses
/// 1e-12).
pub const SHEET_TOL: f32 = 9.536_743e-7;

/// Default asphere Newton step cap of the GPU kernel.
pub const DEFAULT_MAX_ITERATIONS: u32 = 8;
/// Default absolute sag-residual tolerance (mm).
pub const DEFAULT_TOL_ABS: f32 = 1e-9;
/// Default relative sag-residual tolerance (2⁻²⁰).
pub const DEFAULT_TOL_REL: f32 = 9.536_743e-7;

/// One input ray (double-float position and direction as hi + lo).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KRayIn {
    pub pos: [f32; 3],
    pub pos_lo: [f32; 3],
    pub dir: [f32; 3],
    pub dir_lo: [f32; 3],
    pub opl: Df32,
    pub weight: f32,
    pub wavelength: u32,
}

/// The ray state after a step (one record slot) or at the end.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KState {
    pub pos: [f32; 3],
    pub dir: [f32; 3],
    pub opl: Df32,
    pub weight: f32,
    pub status: u32,
}

/// Kernel output for one ray.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct KRayOut {
    pub state: KState,
    /// Internal (bulk) transmittance `exp(−Σ α t)`.
    pub transmittance: f32,
    pub last_surface: u32,
    /// Steps completed.
    pub steps: u32,
}

#[inline(always)]
fn raw<E: FpEnv>(a: V<E>) -> [f32; 3] {
    [a[0].v(), a[1].v(), a[2].v()]
}

/// `mkl_dot3`: `fma(a0, b0, fma(a1, b1, a2·b2))`.
#[inline(always)]
pub fn dot<E: FpEnv>(a: V<E>, b: V<E>) -> F<E> {
    a[0].fma(b[0], a[1].fma(b[1], a[2] * b[2]))
}

/// `mkl_ddot`: double-float dot product, `((a2b2 + a1b1) + a0b0)`.
#[inline(always)]
fn ddot<E: FpEnv>(a: DV<E>, b: DV<E>) -> D<E> {
    let mut s = a[2] * b[2];
    s = s + a[1] * b[1];
    s + a[0] * b[0]
}

#[inline(always)]
fn norm<E: FpEnv>(v: V<E>) -> F<E> {
    f(hypot3_in::<E>(v[0].v(), v[1].v(), v[2].v()))
}

/// `v / |v|` (componentwise division, like `mklight_core::Vec3::normalized`).
#[inline(always)]
fn normalized<E: FpEnv>(v: V<E>) -> V<E> {
    let n = norm(v);
    if n == 0.0 {
        v
    } else {
        [v[0] / n, v[1] / n, v[2] / n]
    }
}

#[inline(always)]
fn finite3<E: FpEnv>(v: V<E>) -> bool {
    v[0].is_finite() && v[1].is_finite() && v[2].is_finite()
}

#[inline(always)]
fn hi3<E: FpEnv>(p: DV<E>) -> V<E> {
    [p[0].hi(), p[1].hi(), p[2].hi()]
}

#[inline(always)]
fn lo3<E: FpEnv>(p: DV<E>) -> V<E> {
    [p[0].lo(), p[1].lo(), p[2].lo()]
}

/// `mkl_renorm_along`: move `d` along `g` so that `|d| = 1` to O(ε²); the
/// component of `d` perpendicular to `g` is unchanged.
#[inline(always)]
fn renorm_along<E: FpEnv>(d: DV<E>, g: DV<E>) -> DV<E> {
    let m = ddot(d, d) - F::ONE;
    let h = dot(hi3(d), hi3(g));
    if h == 0.0 {
        return d;
    }
    let b = m.hi() / (2.0 * h);
    [d[0] - g[0] * b, d[1] - g[1] * b, d[2] - g[2] * b]
}

/// `mkl_renorm`: `d·(1 − (|d|² − 1)/2)`, unit to O((|d|² − 1)²).
#[inline(always)]
fn renorm<E: FpEnv>(d: DV<E>) -> DV<E> {
    let s = 0.5 * (ddot(d, d) - F::ONE).hi();
    [d[0] - d[0] * s, d[1] - d[1] * s, d[2] - d[2] * s]
}

/// `mkl_rot_row`: row `row` of `rot · v` (or `rotᵀ · v`) in double-float,
/// with the rotation's f32 residual folded in as an f32 sum.
#[inline(always)]
fn rot_row<E: FpEnv>(s: &PackedSurface, row: usize, transpose: bool, v: DV<E>) -> D<E> {
    let (a, b, c) = if transpose {
        (row, row + 3, row + 6)
    } else {
        (3 * row, 3 * row + 1, 3 * row + 2)
    };
    let r = |k: usize| f::<E>(s.to_next_rot[k]);
    let rl = |k: usize| f::<E>(s.to_next_rot_lo[k]);
    let lo = rl(a).fma(v[0].hi(), rl(b).fma(v[1].hi(), rl(c) * v[2].hi()));
    let mut acc = v[2] * r(c);
    acc = acc + v[1] * r(b);
    acc = acc + v[0] * r(a);
    acc + lo
}

/// `mkl_xf_forward`: from this surface's frame into the next one's,
/// `P' = rot·P − shift`, `D' = rot·D`.
#[inline(always)]
pub fn xf_forward<E: FpEnv>(s: &PackedSurface, p: &mut DV<E>, d: &mut DV<E>) {
    if s.flags & MKL_SURF_AXIAL != 0 {
        p[2] = p[2] - D::new(s.to_next_origin[2], s.to_next_origin_lo[2]);
        return;
    }
    let (p0, d0) = (*p, *d);
    for row in 0..3 {
        p[row] = rot_row(s, row, false, p0) - D::new(s.to_next_shift[row], s.to_next_shift_lo[row]);
        d[row] = rot_row(s, row, false, d0);
    }
}

/// `mkl_xf_backward`: from the next surface's frame back into this one's,
/// `P = rotᵀ·P' + origin`, `D = rotᵀ·D'`.
#[inline(always)]
pub fn xf_backward<E: FpEnv>(s: &PackedSurface, p: &mut DV<E>, d: &mut DV<E>) {
    if s.flags & MKL_SURF_AXIAL != 0 {
        p[2] = p[2] + D::new(s.to_next_origin[2], s.to_next_origin_lo[2]);
        return;
    }
    let (p0, d0) = (*p, *d);
    for row in 0..3 {
        p[row] = rot_row(s, row, true, p0) + D::new(s.to_next_origin[row], s.to_next_origin_lo[row]);
        d[row] = rot_row(s, row, true, d0);
    }
}

/// `mkl_discriminant`: Kahan's `b² − a·c` (`mklight_trace::numeric::discriminant`).
#[inline(always)]
fn discriminant<E: FpEnv>(a: F<E>, b: F<E>, c: F<E>) -> F<E> {
    let w = a * c;
    let e = (-a).fma(c, w);
    let ff = b.fma(b, -w);
    ff + e
}

#[inline(always)]
fn at<E: FpEnv>(t: F<E>, d: V<E>, q: V<E>) -> V<E> {
    [t.fma(d[0], q[0]), t.fma(d[1], q[1]), t.fma(d[2], q[2])]
}

/// `mkl_conic_from`: conic intersection steps 2–5 of `mklight_trace::surface`
/// from the point `q`; the distance from `q` and the f32 hit point, or `None`
/// for a miss. Also valid for `c = 0` (the plane `z = 0`).
#[inline(always)]
pub fn conic_from<E: FpEnv>(c: F<E>, k: F<E>, q: V<E>, d: V<E>) -> Option<(F<E>, V<E>)> {
    let kap = 1.0 + k;
    let a = c * (d[0].fma(d[0], d[1] * d[1]) + kap * d[2] * d[2]);
    let b = c * (q[0].fma(d[0], q[1] * d[1]) + kap * q[2] * d[2]) - d[2];
    let cc = c * (q[0].fma(q[0], q[1] * q[1]) + kap * q[2] * q[2]) - 2.0 * q[2];
    let disc = discriminant(a, b, cc);
    if disc.is_nan() || disc < 0.0 {
        return None;
    }
    let sq = disc.sqrt();
    let qq = -(b + if b >= 0.0 { sq } else { -sq });
    let on_sheet = |pt: V<E>| 1.0 - kap * c * pt[2] >= -SHEET_TOL;
    let mut best: Option<(F<E>, V<E>)> = None;
    if qq != 0.0 {
        let t = cc / qq;
        let pt = at(t, d, q);
        if t.is_finite() && finite3(pt) && on_sheet(pt) {
            best = Some((t, pt));
        }
    } else if cc == 0.0 {
        best = Some((F::ZERO, q));
    }
    if a != 0.0 && qq != 0.0 {
        let t = qq / a;
        let pt = at(t, d, q);
        if t.is_finite() && finite3(pt) && on_sheet(pt) {
            let r2 = pt[0].fma(pt[0], pt[1] * pt[1]);
            match best {
                Some((_, b1)) if b1[0].fma(b1[0], b1[1] * b1[1]) <= r2 => {}
                _ => best = Some((t, pt)),
            }
        }
    }
    best
}

/// `mkl_asphere_poly`: the packed even polynomial `P(r²)` and `dP/d(r²)` by
/// Horner in `ρ² = r²·asphere_inv_norm2`.
#[inline(always)]
pub fn asphere_poly<E: FpEnv>(s: &PackedSurface, r2: F<E>) -> (F<E>, F<E>) {
    let n = (s.num_asphere as usize).min(MAX_ASPHERE);
    if n == 0 {
        return (F::ZERO, F::ZERO);
    }
    let rho2 = r2 * s.asphere_inv_norm2;
    let mut p = f::<E>(s.asphere[n - 1]);
    let mut dp = s.asphere[n - 1] * f::<E>((n + 1) as f32);
    for j in (0..n - 1).rev() {
        p = p.fma(rho2, f(s.asphere[j]));
        dp = dp.fma(rho2, s.asphere[j] * f::<E>((j + 2) as f32));
    }
    let rho4 = rho2 * rho2;
    (p * rho4, dp * rho2 * s.asphere_inv_norm2)
}

/// `mkl_asphere_domain`: whether the asphere normal is defined at `p`
/// (`1 − κc²r² ≥ 0`), as `mklight_trace::surface::asphere_normal`.
#[inline(always)]
fn asphere_domain<E: FpEnv>(s: &PackedSurface, p: V<E>) -> bool {
    let (c, kap) = (f::<E>(s.curvature), 1.0 + f::<E>(s.conic));
    let r2 = p[0].fma(p[0], p[1] * p[1]);
    let arg = (-kap * c * c).fma(r2, F::ONE);
    !(arg.is_nan() || arg < 0.0)
}

/// Intersection failure codes (`MKL_RAY_*`).
const MISS: u32 = MKL_RAY_MISS;
const NO_CONV: u32 = MKL_RAY_NOT_CONVERGED;
const DOMAIN: u32 = MKL_RAY_OUTSIDE_DOMAIN;

/// `mkl_asphere_from`: Newton on the sag residual from `q` (the conic seed
/// first), with the reference's failure classification. Returns the
/// distance from `q`.
#[inline(always)]
pub fn asphere_from<E: FpEnv>(
    s: &PackedSurface,
    q: V<E>,
    d: V<E>,
    p: &TraceParams,
) -> Result<F<E>, u32> {
    let (c, k) = (f::<E>(s.curvature), f::<E>(s.conic));
    let seed = conic_from(c, k, q, d);
    let seed_hit = seed.is_some();
    let mut t = seed.map_or(F::ZERO, |(t, _)| t);
    let kap = 1.0 + k;
    let mut g_first = F::<E>::ZERO;
    let mut sign_changed = false;
    let mut prev_abs_g = f::<E>(f32::INFINITY);
    let breakdown = |sc: bool| if sc { NO_CONV } else { MISS };
    let mut it = 0u32;
    loop {
        let x = t.fma(d[0], q[0]);
        let y = t.fma(d[1], q[1]);
        let z = t.fma(d[2], q[2]);
        let r2 = x.fma(x, y * y);
        let arg = (-kap * c * c).fma(r2, F::ONE);
        if arg.is_nan() || arg < 0.0 {
            return Err(if !seed_hit && !sign_changed {
                MISS
            } else {
                DOMAIN
            });
        }
        let w = arg.sqrt();
        let (poly, dpoly) = asphere_poly(s, r2);
        let base = c * r2 / (1.0 + w);
        let g = z - (base + poly);
        if !g.is_finite() {
            return Err(breakdown(sign_changed));
        }
        if it == 0 {
            g_first = g;
        } else if g != 0.0 && (g > 0.0) != (g_first > 0.0) {
            sign_changed = true;
        }
        let slope_w = (2.0 * w).fma(dpoly, c);
        let gp = d[2] - slope_w / w.max(f(f32::MIN_POSITIVE)) * x.fma(d[0], y * d[1]);
        let gp_ok = gp.is_finite() && gp != 0.0;
        let tol = f::<E>(p.tolerance_abs)
            + p.tolerance_rel * (((t.abs() + z.abs()) + base.abs()) + poly.abs());
        if g.abs() <= tol {
            if gp_ok {
                t = t - g / gp;
            }
            return if asphere_domain(s, at(t, d, q)) {
                Ok(t)
            } else {
                Err(DOMAIN)
            };
        }
        if it == p.max_iterations {
            let converging = g.abs() <= 0.5 * prev_abs_g;
            return Err(if converging {
                NO_CONV
            } else {
                breakdown(sign_changed)
            });
        }
        if !gp_ok {
            return Err(breakdown(sign_changed));
        }
        t = t - g / gp;
        if !t.is_finite() {
            return Err(breakdown(sign_changed));
        }
        prev_abs_g = g.abs();
        it += 1;
    }
}

/// A valid intersection: the distance from the incoming point in two
/// double-float parts, the hit point, the unnormalised normal `G` (double-
/// float) and the f32 unit normal.
#[derive(Clone, Copy)]
struct Hit<E: FpEnv> {
    ta: D<E>,
    tb: D<E>,
    x: DV<E>,
    g: DV<E>,
    normal: V<E>,
}

/// `mkl_point_on`: `q + t·d` in double-float.
#[inline(always)]
fn point_on<E: FpEnv>(q: V<E>, t: D<E>, d: DV<E>) -> DV<E> {
    [d[0] * t + q[0], d[1] * t + q[1], d[2] * t + q[2]]
}

/// `mkl_refine_t`: `t₁ − r/r'` as a double-float (`r'` and the step are f32;
/// a degenerate slope keeps `t₁`).
#[inline(always)]
fn refine_t<E: FpEnv>(t1: F<E>, r: F<E>, rp: F<E>) -> D<E> {
    let step = r / rp;
    if rp != 0.0 && step.is_finite() {
        D::two_sum(t1, -step)
    } else {
        D::from_f(t1)
    }
}

/// `mkl_intersect`: `mklight_trace::surface::intersect` from the
/// double-float point `p` along `d` (module docs, steps 2–5).
#[inline(always)]
fn intersect<E: FpEnv>(
    s: &PackedSurface,
    p: DV<E>,
    d: DV<E>,
    prm: &TraceParams,
) -> Result<Hit<E>, u32> {
    let zg = [D::ZERO, D::ZERO, D::from_f(F::ONE)];
    let zn = [F::ZERO, F::ZERO, F::ONE];
    let shape = s.shape;
    if shape == MKL_SHAPE_COORD_BREAK {
        return Ok(Hit {
            ta: D::ZERO,
            tb: D::ZERO,
            x: p,
            g: zg,
            normal: zn,
        });
    }
    let conic_like = (shape == MKL_SHAPE_CONIC && s.curvature != 0.0)
        || (shape == MKL_SHAPE_EVEN_ASPHERE && (s.num_asphere > 0 || s.curvature != 0.0));
    if !conic_like {
        // Plane, ideal lens, zero-curvature conic: t = −P_z / D_z.
        if d[2].hi() == 0.0 {
            return Err(MISS);
        }
        let t = (-p[2]) / d[2];
        let x = [p[0] + d[0] * t, p[1] + d[1] * t, D::ZERO];
        if !(x[0].hi().is_finite() && x[1].hi().is_finite()) {
            return Err(MISS);
        }
        return Ok(Hit {
            ta: t,
            tb: D::ZERO,
            x,
            g: zg,
            normal: zn,
        });
    }
    // Closest approach to the vertex, in double-float.
    let t0 = -ddot(p, d);
    let q = [p[0] + d[0] * t0, p[1] + d[1] * t0, p[2] + d[2] * t0];
    let qf = hi3(q);
    let df = hi3(d);
    let ta = t0 - dot(lo3(q), df);
    let (c, k) = (f::<E>(s.curvature), f::<E>(s.conic));
    let kap = 1.0 + k;
    let (tb, x, g) = if shape == MKL_SHAPE_EVEN_ASPHERE && s.num_asphere > 0 {
        let t1 = asphere_from(s, qf, df, prm)?;
        // Refinement on z − sag(r²), conic part in double-float.
        let x0 = point_on(qf, D::from_f(t1), d);
        let r2 = x0[0] * x0[0] + x0[1] * x0[1];
        let w = (D::from_f(F::ONE) - r2 * (kap * c * c)).sqrt();
        let csag = (r2 * c) / (w + F::ONE);
        let (poly, dpoly) = asphere_poly(s, r2.hi());
        let res = (x0[2] - csag) - poly;
        let (xh, wh) = (hi3(x0), w.hi());
        let slope_w = (2.0 * wh).fma(dpoly, c);
        let gp = df[2] - slope_w / wh.max(f(f32::MIN_POSITIVE)) * xh[0].fma(df[0], xh[1] * df[1]);
        let tb = refine_t(t1, res.hi(), gp);
        let x = point_on(qf, tb, d);
        let gs = (w * dpoly) * 2.0 + c;
        (tb, x, [-(x[0] * gs), -(x[1] * gs), w])
    } else {
        let (t1, _) = conic_from(c, k, qf, df).ok_or(MISS)?;
        // Refinement on F = c(x² + y² + κz²) − 2z.
        let x0 = point_on(qf, D::from_f(t1), d);
        let r2 = x0[0] * x0[0] + x0[1] * x0[1];
        let fv = (r2 + (x0[2] * x0[2]) * kap) * c - x0[2] * 2.0;
        let xh = hi3(x0);
        let fp = 2.0 * (c * (xh[0].fma(df[0], xh[1] * df[1]) + kap * xh[2] * df[2]) - df[2]);
        let tb = refine_t(t1, fv.hi(), fp);
        let x = point_on(qf, tb, d);
        let kc = kap * c;
        (tb, x, [x[0] * -c, x[1] * -c, D::from_f(F::ONE) - x[2] * kc])
    };
    Ok(Hit {
        ta,
        tb,
        x,
        g,
        normal: normalized(hi3(g)),
    })
}

/// `mkl_refract`: stable vector Snell (`mklight_trace::interact::refract`)
/// in f32. Returns `(k', cos i, γ)` with `k' = μk + γ·normal` (`γ` signed
/// for the given normal's orientation), or `None` on TIR.
#[inline(always)]
pub fn refract<E: FpEnv>(
    k: V<E>,
    normal: V<E>,
    n1: F<E>,
    n2: F<E>,
) -> Option<(V<E>, F<E>, F<E>)> {
    let mut cos_i = dot(k, normal);
    let flip = cos_i < 0.0;
    let n = if flip {
        cos_i = -cos_i;
        [-normal[0], -normal[1], -normal[2]]
    } else {
        normal
    };
    let cos_i = cos_i.min(F::ONE);
    let mu = n1 / n2;
    let sin2 = (1.0 - cos_i) * (1.0 + cos_i);
    let disc = (-mu * mu).fma(sin2, F::ONE);
    if disc < 0.0 {
        return None;
    }
    let root = disc.sqrt();
    let denom = mu.fma(cos_i, root);
    let gamma = if denom > 0.0 {
        ((n2 - n1) * (n2 + n1)) / (n2 * n2) / denom
    } else {
        root - mu * cos_i
    };
    let out = [
        mu.fma(k[0], gamma * n[0]),
        mu.fma(k[1], gamma * n[1]),
        mu.fma(k[2], gamma * n[2]),
    ];
    Some((out, cos_i, if flip { -gamma } else { gamma }))
}

/// `mkl_reflect`: `k − 2(k·n)n`, and `cos i`.
#[inline(always)]
pub fn reflect<E: FpEnv>(k: V<E>, n: V<E>) -> (V<E>, F<E>) {
    let kn = dot(k, n);
    let s = -2.0 * kn;
    (
        [s.fma(n[0], k[0]), s.fma(n[1], k[1]), s.fma(n[2], k[2])],
        kn.abs().min(F::ONE),
    )
}

/// `mkl_sphere_term` (`mklight_trace::interact::sphere_term`).
#[inline(always)]
fn sphere_term<E: FpEnv>(x: V<E>, t: V<E>, rho: F<E>) -> F<E> {
    let tn = norm(t);
    let xx = dot(x, x);
    let tx = dot(t, x);
    if rho.abs() * xx.sqrt() <= tn {
        let den = norm([t[0] - x[0] * rho, t[1] - x[1] * rho, t[2] - x[2] * rho]) + tn;
        rho.fma(xx, -2.0 * tx) / den
    } else {
        let a = 1.0 / rho;
        let o = [x[0] - t[0] * a, x[1] - t[1] * a, x[2] - t[2] * a];
        F::ONE.copysign(a) * f::<E>(2.0).fma(dot(o, x), -xx) / (norm(o) + a.abs() * tn)
    }
}

/// `mkl_ideal_lens` (`mklight_trace::interact::ideal_lens`, with the power
/// `φ = 1/f` instead of `f`): outgoing direction and added optical path.
#[inline(always)]
pub fn ideal_lens<E: FpEnv>(
    point: V<E>,
    k: V<E>,
    phi: F<E>,
    n1: F<E>,
    n2: F<E>,
    vergence: F<E>,
) -> Option<(V<E>, F<E>)> {
    if k[2] == 0.0 {
        return None;
    }
    let sz = F::ONE.copysign(k[2]);
    let az = k[2].abs();
    let r = n1 / n2;
    let t = [k[0] / az, k[1] / az, sz];
    let tx2 = r.fma(t[0], -point[0] * phi);
    let ty2 = r.fma(t[1], -point[1] * phi);
    let inv = 1.0 / tx2.fma(tx2, ty2.fma(ty2, F::ONE)).sqrt();
    let out = [tx2 * inv, ty2 * inv, sz * inv];
    let t_out = [tx2, ty2, sz];
    let x = [point[0], point[1], F::ZERO];
    let rho_b = (-r).fma(vergence, phi);
    let delta = n1.fma(
        sphere_term(x, t, vergence),
        -n2 * sphere_term(x, t_out, -rho_b),
    );
    Some((out, delta))
}

/// `mkl_fresnel`: unpolarised `(R, T)` of a bare interface
/// (`mklight_core::materials::coating::fresnel`).
#[inline(always)]
pub fn fresnel<E: FpEnv>(n1: F<E>, n2: F<E>, cos_i: F<E>) -> (F<E>, F<E>) {
    let ci = cos_i.max(F::ZERO).min(F::ONE);
    let sin2_i = (1.0 - ci) * (1.0 + ci);
    let rr = n1 / n2;
    let sin2_t = rr * rr * sin2_i;
    if sin2_t >= 1.0 {
        return (F::ONE, F::ZERO);
    }
    let ct = (1.0 - sin2_t).max(F::ZERO).sqrt();
    let (a, b) = (n1 * ci, n2 * ct);
    let rs = if a + b == 0.0 {
        F::ONE
    } else {
        (a - b) / (a + b)
    };
    let (c, dd) = (n2 * ci, n1 * ct);
    let rp = if c + dd == 0.0 {
        F::ONE
    } else {
        (c - dd) / (c + dd)
    };
    let (rs2, rp2) = (rs * rs, rp * rp);
    (0.5 * (rs2 + rp2), 0.5 * ((1.0 - rs2) + (1.0 - rp2)))
}

/// `mkl_passes_aperture` (`mklight_core::ResolvedSurface::passes_aperture`).
#[inline(always)]
pub fn passes_aperture<E: FpEnv>(s: &PackedSurface, x: F<E>, y: F<E>) -> bool {
    let a0 = f::<E>(s.aperture_params[0]);
    let a1 = f::<E>(s.aperture_params[1]);
    match s.aperture {
        MKL_APERTURE_CIRCULAR => x * x + y * y <= a0 * a0,
        MKL_APERTURE_ANNULAR => {
            let r2 = x * x + y * y;
            r2 >= a0 * a0 && r2 <= a1 * a1
        }
        MKL_APERTURE_RECTANGULAR => x.abs() <= a0 && y.abs() <= a1,
        MKL_APERTURE_ELLIPTICAL => {
            let u = x / a0;
            let v = y / a1;
            u * u + v * v <= 1.0
        }
        _ => true,
    }
}

/// What a step does at its surface.
#[derive(Clone, Copy)]
enum Act {
    None,
    Refract,
    Reflect,
    /// Ideal lens (reflect first for a mirror-type ideal lens).
    Lens { reflect: bool },
}

/// The reported state: the f32 point and direction, with the rounding
/// residual of the point along the direction moved into the optical path.
#[inline(always)]
fn report<E: FpEnv>(
    p: DV<E>,
    d: DV<E>,
    opl: D<E>,
    n: D<E>,
    weight: F<E>,
    status: u32,
) -> KState {
    let corr = n.hi() * dot(lo3(p), hi3(d));
    KState {
        pos: raw(hi3(p)),
        dir: raw(hi3(d)),
        opl: (opl - corr).0,
        weight: weight.v(),
        status,
    }
}

#[inline(always)]
fn index<E: FpEnv>(s: &PackedSurface, w: usize) -> D<E> {
    D::new(s.n_after[w], s.n_after_lo[w])
}

/// Trace one ray through the path program `steps` (`mkl_trace`).
///
/// `h`/`s` are the ray's system (`s` its `num_surfaces` records); the
/// program, wavelength and surfaces must have been validated by the host
/// (`PreparedTrace::new`). `records`, when given, has `steps.len() + 1`
/// slots: slot 0 the input, slot `k + 1` the state after step `k` (a failed
/// ray repeats its final state and status).
pub fn trace_ray<E: FpEnv>(
    h: &PackedHeader,
    s: &[PackedSurface],
    steps: &[u32],
    prm: &TraceParams,
    ray: &KRayIn,
    mut records: Option<&mut [KState]>,
) -> KRayOut {
    let w = ray.wavelength as usize;
    let clip = prm.flags & MKL_TRACE_CLIP != 0;
    let radiometry = prm.flags & MKL_TRACE_RADIOMETRY != 0;
    let at_infinity = h.flags & MKL_SYS_OBJECT_AT_INFINITY != 0;
    let num_steps = (prm.num_steps as usize).min(steps.len());

    let mut pos: DV<E> = [
        D::new(ray.pos[0], ray.pos_lo[0]),
        D::new(ray.pos[1], ray.pos_lo[1]),
        D::new(ray.pos[2], ray.pos_lo[2]),
    ];
    let mut dir: DV<E> = [
        D::new(ray.dir[0], ray.dir_lo[0]),
        D::new(ray.dir[1], ray.dir_lo[1]),
        D::new(ray.dir[2], ray.dir_lo[2]),
    ];
    let mut opl = D::<E>::from_df(ray.opl);
    let w_in = f::<E>(ray.weight);
    let mut tw = F::<E>::ONE;
    let mut absorb = F::<E>::ZERO;
    let start = prm.start_surface as usize;
    let mut n_cur = index::<E>(&s[start], w);
    let mut last = start as u32;
    let mut status = MKL_RAY_OK;
    let mut done = 0usize;

    let weight_now = |tw: F<E>, absorb: F<E>| {
        if radiometry {
            w_in * tw * f::<E>(exp_in::<E>((-absorb).v()))
        } else {
            w_in
        }
    };
    if let Some(r) = records.as_deref_mut() {
        r[0] = report(pos, dir, opl, n_cur, w_in, MKL_RAY_OK);
    }
    if finite3(hi3(pos)) && finite3(hi3(dir)) && finite3(lo3(pos)) && finite3(lo3(dir)) {
        dir = renorm(dir);
    } else {
        status = MKL_RAY_NONFINITE | ((start as u32) << MKL_RAY_SURFACE_SHIFT);
    }

    let mut k = 0usize;
    while status == MKL_RAY_OK && k < num_steps {
        let word = steps[k];
        let si = (word & MKL_STEP_SURFACE_MASK) as usize;
        let mode = (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK;
        let back = word & MKL_STEP_BACKWARD != 0;
        let gap = if back { si } else { si - 1 };
        let n1 = index::<E>(&s[gap], w);
        let alpha = if gap == 0 && at_infinity {
            F::ZERO
        } else {
            f::<E>(s[gap].alpha_after[w])
        };
        let n_other = if back {
            if si >= 1 { index::<E>(&s[si - 1], w) } else { n1 }
        } else {
            index::<E>(&s[si], w)
        };
        // 1. Transfer into the surface's frame.
        if back {
            xf_backward(&s[si], &mut pos, &mut dir);
        } else {
            xf_forward(&s[si - 1], &mut pos, &mut dir);
        }
        let surf = &s[si];
        last = si as u32;
        n_cur = n1;
        let fail = |code: u32| code | ((si as u32) << MKL_RAY_SURFACE_SHIFT);
        // 2. Intersect; optical path and absorption of the segment.
        let hit = match intersect(surf, pos, dir, prm) {
            Ok(h) => h,
            Err(code) => {
                status = fail(code);
                break;
            }
        };
        opl = opl + hit.ta * n1;
        opl = opl + hit.tb * n1;
        if alpha != 0.0 {
            absorb = alpha.fma(hit.ta.hi() + hit.tb.hi(), absorb);
        }
        pos = hit.x;
        if !finite3(hi3(pos)) {
            status = fail(MKL_RAY_NONFINITE);
            break;
        }
        // 3. Clip.
        let is_cb = surf.shape == MKL_SHAPE_COORD_BREAK;
        if clip && !is_cb && !passes_aperture(surf, pos[0].hi(), pos[1].hi()) {
            status = fail(MKL_RAY_VIGNETTED);
            break;
        }
        // 4. Interact.
        let is_image = surf.flags & MKL_SURF_IMAGE != 0;
        let paraxial = surf.shape == MKL_SHAPE_PARAXIAL;
        let mirror = surf.interaction == MKL_INTERACT_REFLECT;
        let dummy = surf.interaction == MKL_INTERACT_DUMMY;
        let act = if is_cb {
            Act::None
        } else {
            match mode {
                MKL_STEP_SEQUENTIAL => {
                    if is_image || dummy {
                        Act::None
                    } else if paraxial {
                        Act::Lens { reflect: mirror }
                    } else if mirror {
                        Act::Reflect
                    } else {
                        Act::Refract
                    }
                }
                MKL_STEP_TRANSMIT => {
                    if is_image || dummy {
                        Act::None
                    } else if paraxial {
                        Act::Lens { reflect: false }
                    } else {
                        Act::Refract
                    }
                }
                _ => Act::Reflect,
            }
        };
        let coating = surf.coating;
        let df = hi3(dir);
        match act {
            Act::None => {}
            Act::Refract => {
                let Some((_, cos, gamma)) = refract(df, hit.normal, n1.hi(), n_other.hi()) else {
                    status = fail(MKL_RAY_TIR);
                    break;
                };
                let mu = n1 / n_other;
                let b = gamma / norm(hi3(hit.g));
                let d2 = [
                    dir[0] * mu + hit.g[0] * b,
                    dir[1] * mu + hit.g[1] * b,
                    dir[2] * mu + hit.g[2] * b,
                ];
                dir = renorm_along(d2, hit.g);
                n_cur = n_other;
                if radiometry {
                    let t = if mirror {
                        F::ZERO
                    } else if coating == MKL_COATING_IDEAL {
                        f(surf.coating_t)
                    } else {
                        fresnel(n1.hi(), n_other.hi(), cos).1
                    };
                    tw = tw * t;
                }
            }
            Act::Reflect => {
                let (_, cos) = reflect(df, hit.normal);
                let gh = hi3(hit.g);
                let b = -2.0 * dot(df, gh) / dot(gh, gh);
                let d2 = [
                    dir[0] + hit.g[0] * b,
                    dir[1] + hit.g[1] * b,
                    dir[2] + hit.g[2] * b,
                ];
                dir = renorm_along(d2, hit.g);
                if radiometry {
                    let r = if coating == MKL_COATING_IDEAL {
                        f(surf.coating_r)
                    } else if mirror {
                        F::ONE
                    } else {
                        fresnel(n1.hi(), n_other.hi(), cos).0
                    };
                    tw = tw * r;
                }
            }
            Act::Lens { reflect: refl } => {
                let mut kk = df;
                if refl {
                    kk = reflect(kk, hit.normal).0;
                }
                let n2 = if refl { n1 } else { n_other };
                let verg = if back { F::ZERO } else { f(surf.vergence[w]) };
                let point = [pos[0].hi(), pos[1].hi(), F::ZERO];
                let Some((d2, delta)) = ideal_lens(
                    point,
                    kk,
                    f(surf.paraxial_power),
                    n1.hi(),
                    n2.hi(),
                    verg,
                ) else {
                    status = fail(MKL_RAY_DEGENERATE);
                    break;
                };
                dir = renorm([D::from_f(d2[0]), D::from_f(d2[1]), D::from_f(d2[2])]);
                n_cur = n2;
                opl = opl + delta;
                if radiometry && coating == MKL_COATING_IDEAL {
                    tw = tw * f::<E>(if refl { surf.coating_r } else { surf.coating_t });
                }
            }
        }
        k += 1;
        done = k;
        if let Some(r) = records.as_deref_mut() {
            r[k] = report(pos, dir, opl, n_cur, weight_now(tw, absorb), MKL_RAY_OK);
        }
    }

    let transmittance = f::<E>(exp_in::<E>((-absorb).v()));
    let weight = if radiometry {
        w_in * tw * transmittance
    } else {
        w_in
    };
    let state = report(pos, dir, opl, n_cur, weight, status);
    if let Some(r) = records.as_deref_mut() {
        for slot in r.iter_mut().take(num_steps + 1).skip(done + 1) {
            *slot = state;
        }
    }
    KRayOut {
        state,
        transmittance: transmittance.v(),
        last_surface: last,
        steps: done as u32,
    }
}
