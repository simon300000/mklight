//! Field curvature (astigmatic foci along the real chief ray) and distortion.
//!
//! # Astigmatic foci: generalised Coddington along the real chief ray
//!
//! The chief ray of a field is traced (unclipped) with per-surface records.
//! Along it, the thin pencil around the chief ray is followed in an
//! orthonormal basis `U = (u₁, u₂)` transverse to the ray, as a family of
//! rays `x_⊥ = X a`, `δk_⊥ = T a` (2×2 matrices, parameter `a`). Its wavefront
//! **curvature matrix** is `C = −T X⁻¹`: near the chief ray the wavefront is
//! `S(x) = n k·x − (n/2) x_⊥ᵀ C x_⊥`, and a pencil converging to a point a
//! distance `t` ahead has `C = I/t` (`t > 0` ahead). The pair `(X, T)` stays
//! finite both for collimated pencils (`T = 0`, the start for objects at
//! infinity) and at a focus (`X = 0`, e.g. on the image surface). A finite
//! object starts with `X = d I`, `T = I` (`C = −I/d`), a pencil diverging
//! from the object point a distance `d` behind surface 1.
//!
//! * transfer over a distance `d` along the ray: `X ← X + d T`;
//! * refraction or reflection at a surface with unit normal `N` and second
//!   fundamental form `K` in a tangent basis `E` (both relative to `N`). The
//!   phase of the incident and the outgoing wave match on the surface to
//!   second order, which gives
//!   `n' P'ᵀ C' P' = n Pᵀ C P + (n' k'·N − n k·N) K`, with
//!   `P = Uᵀ E`, `P' = U'ᵀ E`. In ray form, with surface coordinates
//!   `δs = P⁻¹ X`: `X' = P' δs`, `T' = P'⁻ᵀ (n Pᵀ T − g K δs)/n'`,
//!   `g = n' k'·N − n k·N`. Reflection uses `n' = n` and the mirrored basis
//!   `U' = (I − 2NNᵀ) U`. Refraction carries `U` by the minimal rotation
//!   taking `k` to `k'`. For a chief ray in a principal plane this reduces
//!   to Coddington's equations,
//!   `n' cos²I'/t' = n cos²I/t + (n' cos I' − n cos I) κ_t` and
//!   `n'/s' = n/s + (n' cos I' − n cos I) κ_s`;
//! * ideal lens (`Paraxial` shape), which maps the direction tangents
//!   `τ = (k_x, k_y)/|k_z|` at lens-plane point `s` to `τ' = (n/n') τ − s/f`:
//!   with `δs = P⁻¹ X` and `B = ∂τ/∂k`, `T' = (B'U')⁻¹ ((n/n') BU T − δs/f)`
//!   and `X' = P' δs`. In a meridional section this is
//!   `V' = cos U' (V / cos U + n'/f)` for the vergence `V = n/t`;
//! * dummies and coordinate breaks: nothing (the basis is carried into each
//!   surface frame).
//!
//! `K` comes from the principal curvatures of the surface of revolution at
//! radius `r`, the meridional `κ_m = f''/(1 + f'²)^{3/2}` and the sagittal
//! `κ_s = f'/(r √(1 + f'²))` ([`principal_curvatures`], evaluated without
//! dividing by `r` or by the conic square root). Their directions are the
//! radial direction projected onto the tangent plane and its perpendicular.
//! The equations hold for skew chief rays and any tilts. No principal-plane
//! assumption is made.
//!
//! At the image the **focal lines** follow from the eigen-decomposition of
//! the symmetric focal-distance matrix `F = −X T⁻¹ = C⁻¹` (or of `C` for an
//! afocal pencil, `T` singular). Rays displaced along an eigenvector with
//! eigenvalue `t` meet a distance `t` along the chief ray from its image
//! point. Following CONVENTIONS §2, the
//! *tangential* focus is the one whose collapsing direction (eigenvector) is
//! closer to the image surface's local `y` (meridional) direction, projected
//! normal to the chief ray. The *sagittal* focus is the other one. On a
//! surface tilted about `y` the chief ray's plane of incidence is `x–z`, so
//! the surface's own "tangential" section images the sagittal (x) direction.
//! The labels follow the image frame, not the plane of incidence, and match
//! the close-ray pairs below. The reported **field curvature** is the
//! projection of each focus on the image surface's local `z` axis,
//! `Δz = t · d_z`, i.e. the axial focus shift from the image surface
//! (positive along local `+z`).
//!
//! # Close-ray cross-check
//!
//! Two pairs of rays at pupil offsets `(0, ±h)` and `(±h, 0)` from the chief
//! are traced to the image. Each ray is moved onto the plane through the
//! chief image point `I` normal to the chief direction `c`, giving a point
//! `Q` and a transverse slope `T = d/(d·c) − c`. With `X` and `T` the 2×2
//! matrices of the pair differences `ΔQ`, `ΔT` (columns: the `y` pair and
//! the `x` pair; rows: the transverse basis), the pencil at distance `s`
//! along `c` is `X + sT`. It degenerates into a focal line where
//! `det(X + sT) = 0`, i.e. at the eigenvalues of the focal-distance matrix
//! `F = −X T⁻¹ = C⁻¹`. `F` is symmetrised and decomposed like `C` above (for
//! afocal pencils, `C = −T X⁻¹` is used instead), and the foci are labelled
//! by the same rule. Central differences give an error of `O(h²)`. Both
//! values are reported, and [`field_curvature`] adds a warning where they
//! disagree by more than `10⁻³ (1 + |Δz|)` mm.
//!
//! # Distortion
//!
//! The real chief-ray height `y` (image local frame) is compared with the
//! paraxial chief height `y_p` of the same object source at the same
//! wavelength: `k·tan θ` for an object at infinity (`k` = paraxial image
//! height per unit `tan θ` on the actual image surface) and `m·h` for a
//! finite object. `distortion = 100 (y − y_p)/y_p` %. For objects at
//! infinity, the `f·tan θ` and `f·θ` distortions use `f = n₀·EFL`, the image
//! height per unit `tan θ` at the paraxial focal plane. All are 0 on axis
//! (their limit) and NaN where undefined.
//!
//! The grid distortion map traces the chief rays of a square grid of
//! normalised fields `(Hx, Hy) ∈ [−a, a]²` (`a = 1/√2` by default, so the
//! corners are at the maximum radial field) and reports real and paraxial
//! image points and the radial distortion `100 (|r| − |r_p|)/|r_p|`.

use mklight_core::job::RunTimer;
use mklight_core::{Interaction, JobContext, MkResult, RunInfo, SurfaceShape, Vec3};
use mklight_trace::trace::transfer_into;
use mklight_trace::{FieldPoint, ObjectSource, PreparedField, Ray, SequentialTracer, TraceOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::util::{
    WavelengthSelection, check_count, check_finite, field_scale, field_unit, linspace, m2, nf,
    paraxial_image_point, run_info, wavelength_or_primary,
};

/// Principal curvatures `(κ_m, κ_s)` (meridional, sagittal) of a surface of
/// revolution at radius `r ≥ 0`, positive when the centre of curvature lies
/// at local `+z`. `None` outside the sag domain. Planes, ideal lenses and
/// coordinate breaks have zero curvature.
///
/// With `w = √(1 − (1+k)c²r²)` the conic has `f'/r = c/w` and `f'' = c/w³`,
/// so `κ_m = c/D^{3/2}` and `κ_s = c/√D` with `D = 1 − k c² r²`. An even
/// asphere adds `Σ aᵢ(2i+4) r^{2i+2}` to `f'/r` and `Σ aᵢ(2i+4)(2i+3) r^{2i+2}`
/// to `f''`.
pub fn principal_curvatures(shape: &SurfaceShape, r: f64) -> Option<(f64, f64)> {
    let r2 = r * r;
    match shape {
        SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::CoordinateBreak => {
            Some((0.0, 0.0))
        }
        SurfaceShape::Conic {
            curvature: c,
            conic: k,
        } => {
            let w2 = 1.0 - (1.0 + k) * c * c * r2;
            if w2 < 0.0 {
                return None;
            }
            let d = (-k * c * c).mul_add(r2, 1.0);
            if d <= 0.0 {
                return None;
            }
            Some((c / (d * d.sqrt()), c / d.sqrt()))
        }
        SurfaceShape::EvenAsphere {
            curvature: c,
            conic: k,
            coefficients,
        } => {
            let w2 = 1.0 - (1.0 + k) * c * c * r2;
            if w2 < 0.0 {
                return None;
            }
            let w = w2.sqrt();
            let (mut g, mut f2) = if *c == 0.0 {
                (0.0, 0.0)
            } else {
                (c / w, c / (w * w2))
            };
            // Σ aᵢ (2i+4) r^{2i+2} and Σ aᵢ (2i+4)(2i+3) r^{2i+2}.
            let mut p = r2; // r^{2i+2}
            for (i, a) in coefficients.iter().enumerate() {
                let e = (2 * i + 4) as f64;
                g = (a * e).mul_add(p, g);
                f2 = (a * e * (e - 1.0)).mul_add(p, f2);
                p *= r2;
            }
            let q = (r2 * g).mul_add(g, 1.0); // 1 + f'²
            let sq = q.sqrt();
            Some((f2 / (q * sq), g / sq))
        }
    }
}

/// Tangential and sagittal foci of the pencil around a chief ray: signed
/// distances (mm) along the image-space chief ray from its image point
/// (`±∞` for a collimated pencil), and the chief direction's local `z`
/// component on the image surface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct AstigmaticFocus {
    #[serde(with = "nf")]
    pub tangential: f64,
    #[serde(with = "nf")]
    pub sagittal: f64,
    pub dir_z: f64,
}

impl AstigmaticFocus {
    /// Axial (image local `z`) focus shifts `(Δz_t, Δz_s)` from the image surface.
    pub fn axial(&self) -> (f64, f64) {
        (self.tangential * self.dir_z, self.sagittal * self.dir_z)
    }
}

/// Tangent basis `E` (principal directions) and second fundamental form `K`
/// (diagonal in `E`, relative to the normal `n`) of a surface of revolution
/// at local hit `p`.
fn surface_tensor(shape: &SurfaceShape, p: Vec3, n: Vec3) -> Option<([Vec3; 2], m2::M)> {
    let r = p.x.hypot(p.y);
    let (km, ks) = principal_curvatures(shape, r)?;
    let m = if r > 0.0 && km != ks {
        let er = Vec3::new(p.x / r, p.y / r, 0.0);
        er - n * er.dot(n)
    } else {
        let t = Vec3::X - n * n.x;
        if t.norm() > 1e-6 {
            t
        } else {
            Vec3::Y - n * n.y
        }
    };
    let m = m.normalized();
    Some(([m, n.cross(m)], [[km, 0.0], [0.0, ks]]))
}

/// `P_ab = u_a · e_b`: transverse coordinates of tangent vectors.
fn proj(u: &[Vec3; 2], e: &[Vec3; 2]) -> m2::M {
    [
        [u[0].dot(e[0]), u[0].dot(e[1])],
        [u[1].dot(e[0]), u[1].dot(e[1])],
    ]
}

/// An orthonormal basis transverse to unit `k`.
fn transverse_basis(k: Vec3) -> [Vec3; 2] {
    let mut a = Vec3::X - k * k.x;
    if a.norm() < 1e-6 {
        a = Vec3::Y - k * k.y;
    }
    let a = a.normalized();
    [a, k.cross(a)]
}

/// Carry the transverse basis `u` of `k` to `k2` by the minimal rotation
/// (Rodrigues with the unnormalised axis `k × k2`), or by re-projection for
/// near-reversals.
fn carry(u: &[Vec3; 2], k: Vec3, k2: Vec3) -> [Vec3; 2] {
    let c = k.dot(k2);
    if c > -0.9 {
        let a = k.cross(k2);
        let rot = |v: Vec3| v * c + a.cross(v) + a * (a.dot(v) / (1.0 + c));
        let u0 = rot(u[0]).normalized();
        let u1 = (rot(u[1]) - u0 * rot(u[1]).dot(u0)).normalized();
        [u0, u1]
    } else {
        let u0 = (u[0] - k2 * u[0].dot(k2)).normalized();
        [u0, k2.cross(u0)]
    }
}

/// Mirror image of the basis in the plane normal to unit `n`.
fn householder(u: &[Vec3; 2], n: Vec3) -> [Vec3; 2] {
    [
        u[0] - n * (2.0 * u[0].dot(n)),
        u[1] - n * (2.0 * u[1].dot(n)),
    ]
}

/// `(BU)_{ib} = B_i · u_b` with `B = ∂τ/∂k`, `τ = (k_x, k_y)/|k_z|`.
fn tangent_jacobian(k: Vec3, u: &[Vec3; 2]) -> Option<m2::M> {
    let az = k.z.abs();
    if az == 0.0 {
        return None;
    }
    let sz = k.z.signum();
    let b = [
        Vec3::new(1.0, 0.0, -sz * k.x / az) * (1.0 / az),
        Vec3::new(0.0, 1.0, -sz * k.y / az) * (1.0 / az),
    ];
    Some([
        [b[0].dot(u[0]), b[0].dot(u[1])],
        [b[1].dot(u[0]), b[1].dot(u[1])],
    ])
}

/// The two foci `(tangential, sagittal)` from a symmetric pencil matrix in
/// the transverse basis `u`: a curvature matrix (`curvature = true`, foci
/// `1/λ`) or a focal-distance matrix (foci `λ`). The tangential focus is the
/// one whose eigenvector lies closer to `e_t` (module docs).
fn label_foci(a: &m2::M, curvature: bool, u: &[Vec3; 2], e_t: Vec3) -> (f64, f64) {
    let (lp, lm, v) = m2::sym_eigen(a);
    let w = [u[0].dot(e_t), u[1].dot(e_t)];
    let plus_is_t = (v[0] * w[0] + v[1] * w[1]).abs() >= std::f64::consts::FRAC_1_SQRT_2;
    let f = |l: f64| if curvature { 1.0 / l } else { l };
    if plus_is_t {
        (f(lp), f(lm))
    } else {
        (f(lm), f(lp))
    }
}

/// The image surface's meridional (local `y`) direction projected normal
/// to the chief direction `k`.
fn meridional(k: Vec3) -> Option<Vec3> {
    let e = Vec3::Y - k * k.y;
    (e.norm() > 1e-9).then(|| e.normalized())
}

/// Generalised-Coddington tangential/sagittal foci along the real chief ray
/// of `field` at wavelength `wl` (module docs). `None` if the chief ray
/// fails or a surface curvature is undefined at its hit.
pub fn coddington_focus(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
) -> Option<AstigmaticFocus> {
    let sys = tracer.system();
    let img = sys.image_index();
    let mut rec = Vec::with_capacity(sys.num_surfaces());
    let r = tracer.trace_with(
        field,
        wl,
        0.0,
        0.0,
        &TraceOptions::unclipped(),
        Some(&mut rec),
    );
    if !r.status.is_ok() || rec.len() != img + 1 {
        return None;
    }
    // Pencil rays `x_⊥ = X a`, `δk_⊥ = T a` (module docs): collimated.
    let (mut x, mut t) = (m2::I, [[0.0; 2]; 2]);
    let mut u = [Vec3::X, Vec3::Y];
    for i in 1..=img {
        let inc = transfer_into(sys, i, Ray::new(rec[i - 1].point, rec[i - 1].dir));
        let k = inc.dir;
        if i == 1 {
            u = transverse_basis(k);
        } else {
            let xf = |v: Vec3| transfer_into(sys, i, Ray::new(Vec3::ZERO, v)).dir;
            u = [xf(u[0]), xf(u[1])];
        }
        let d = (rec[i].point - inc.pos).dot(k);
        if i == 1 && !sys.object_at_infinity {
            // Diverging from the object point `d` behind: C = −I/d.
            (x, t) = ([[d, 0.0], [0.0, d]], m2::I);
        } else {
            x = m2::add(&x, &m2::scale(&t, d));
        }
        if i == img {
            let e_t = meridional(k)?;
            let (ft, fs) = pencil_foci(&x, &t, &u, e_t)?;
            return Some(AstigmaticFocus {
                tangential: ft,
                sagittal: fs,
                dir_z: k.z,
            });
        }
        let s = &sys.surfaces[i];
        if s.is_passive() {
            continue;
        }
        let n = sys.n_before(i, wl);
        let nrm = rec[i].normal;
        let kout = rec[i].dir;
        let reflect = s.interaction == Interaction::Reflect;
        if let SurfaceShape::Paraxial { focal_length } = s.shape {
            // A reflecting ideal lens is a plane mirror (which leaves the
            // pencil unchanged in the mirrored basis) followed by the lens.
            let (k, u_in, n2) = if reflect {
                (k - Vec3::Z * (2.0 * k.z), householder(&u, Vec3::Z), n)
            } else {
                (k, u, s.n_after[wl])
            };
            let e = [Vec3::X, Vec3::Y];
            let u_out = carry(&u_in, k, kout);
            // δs = P⁻¹X a, δτ = (BU) T a, δτ' = (n/n') δτ − δs/f.
            let ds = m2::mul(&m2::inv(&proj(&u_in, &e))?, &x);
            let dtau = m2::add(
                &m2::scale(&m2::mul(&tangent_jacobian(k, &u_in)?, &t), n / n2),
                &m2::scale(&ds, -1.0 / focal_length),
            );
            t = m2::mul(&m2::inv(&tangent_jacobian(kout, &u_out)?)?, &dtau);
            x = m2::mul(&proj(&u_out, &e), &ds);
            u = u_out;
        } else {
            let (e, kmat) = surface_tensor(&s.shape, rec[i].point, nrm)?;
            let (n2, u_out) = if reflect {
                (n, householder(&u, nrm))
            } else {
                (s.n_after[wl], carry(&u, k, kout))
            };
            // Surface coordinates δs = P⁻¹X a; with C = −T X⁻¹ the matching
            // condition gives T' = P'⁻ᵀ (n Pᵀ T − g K δs)/n', X' = P' δs.
            let p = proj(&u, &e);
            let p_out = proj(&u_out, &e);
            let ds = m2::mul(&m2::inv(&p)?, &x);
            let g = n2.mul_add(kout.dot(nrm), -n * k.dot(nrm));
            let rhs = m2::add(
                &m2::scale(&m2::mul(&m2::t(&p), &t), n),
                &m2::scale(&m2::mul(&kmat, &ds), -g),
            );
            t = m2::scale(&m2::mul(&m2::t(&m2::inv(&p_out)?), &rhs), 1.0 / n2);
            x = m2::mul(&p_out, &ds);
            u = u_out;
        }
    }
    None
}

/// Labelled foci `(tangential, sagittal)` of the pencil `(X, T)` in the
/// transverse basis `u`: from the focal-distance matrix `F = −X T⁻¹`, or
/// the curvature matrix `C = −T X⁻¹` when `T` is singular (afocal).
fn pencil_foci(x: &m2::M, t: &m2::M, u: &[Vec3; 2], e_t: Vec3) -> Option<(f64, f64)> {
    if let Some(ti) = m2::inv(t) {
        let f = m2::scale(&m2::mul(x, &ti), -1.0);
        Some(label_foci(&f, false, u, e_t))
    } else {
        let xi = m2::inv(x)?;
        let c = m2::scale(&m2::mul(t, &xi), -1.0);
        Some(label_foci(&c, true, u, e_t))
    }
}

/// Close-ray (finite-difference) foci of `field` at `wl` with pupil step `h`
/// (module docs). `None` if any of the five rays fails or the pencil is
/// degenerate.
pub fn close_ray_focus(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    h: f64,
) -> Option<AstigmaticFocus> {
    let opts = TraceOptions::unclipped();
    let c = tracer.trace(field, wl, 0.0, 0.0, &opts);
    if !c.status.is_ok() {
        return None;
    }
    let ip = c.ray.pos;
    let cd = c.ray.dir;
    let proj_ray = |px: f64, py: f64| -> Option<(Vec3, Vec3)> {
        let r = tracer.trace(field, wl, px, py, &opts);
        if !r.status.is_ok() {
            return None;
        }
        let d = r.ray.dir;
        let dc = d.dot(cd);
        if dc <= 0.0 {
            return None;
        }
        let q = r.ray.pos + d * ((ip - r.ray.pos).dot(cd) / dc);
        Some((q, d / dc - cd))
    };
    let e_t = meridional(cd)?;
    let u = [e_t, cd.cross(e_t)];
    let (ya, yb) = (proj_ray(0.0, h)?, proj_ray(0.0, -h)?);
    let (xa, xb) = (proj_ray(h, 0.0)?, proj_ray(-h, 0.0)?);
    let col = |a: Vec3, b: Vec3| [a.dot(u[0]), a.dot(u[1]), b.dot(u[0]), b.dot(u[1])];
    let qy = col(ya.0 - yb.0, xa.0 - xb.0);
    let ty = col(ya.1 - yb.1, xa.1 - xb.1);
    let xm = [[qy[0], qy[2]], [qy[1], qy[3]]];
    let tm = [[ty[0], ty[2]], [ty[1], ty[3]]];
    let (t, s) = pencil_foci(&xm, &tm, &u, e_t)?;
    Some(AstigmaticFocus {
        tangential: t,
        sagittal: s,
        dir_z: cd.z,
    })
}

// ----- Field curvature and distortion vs field ----------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FieldCurvatureSettings {
    /// Field samples from the axis to the maximum field along `+y`.
    pub field_samples: usize,
    pub wavelengths: WavelengthSelection,
    /// Normalised pupil step of the close-ray cross-check.
    pub close_ray_step: f64,
}

impl Default for FieldCurvatureSettings {
    fn default() -> Self {
        FieldCurvatureSettings {
            field_samples: 21,
            wavelengths: WavelengthSelection::All,
            close_ray_step: 1e-3,
        }
    }
}

/// Field curvature and distortion of one wavelength; one value per field
/// sample, NaN (`null`) where undefined.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldCurvatureCurve {
    pub wavelength: usize,
    pub wavelength_um: f64,
    /// Tangential axial focus shift from the image surface (mm, Coddington).
    #[serde(with = "nf::vec")]
    pub tangential: Vec<f64>,
    /// Sagittal axial focus shift (mm, Coddington).
    #[serde(with = "nf::vec")]
    pub sagittal: Vec<f64>,
    /// Close-ray cross-checks of `tangential` / `sagittal`.
    #[serde(with = "nf::vec")]
    pub tangential_close_rays: Vec<f64>,
    #[serde(with = "nf::vec")]
    pub sagittal_close_rays: Vec<f64>,
    /// Real chief-ray height on the image surface (mm).
    #[serde(with = "nf::vec")]
    pub real_height: Vec<f64>,
    /// Paraxial chief height of the same source (mm).
    #[serde(with = "nf::vec")]
    pub paraxial_height: Vec<f64>,
    /// `100 (y − y_p)/y_p` (%).
    #[serde(with = "nf::vec")]
    pub distortion_percent: Vec<f64>,
    /// Against `f·tan θ` (objects at infinity; NaN otherwise).
    #[serde(with = "nf::vec")]
    pub f_tan_theta_distortion_percent: Vec<f64>,
    /// Against `f·θ` (objects at infinity; NaN otherwise).
    #[serde(with = "nf::vec")]
    pub f_theta_distortion_percent: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldCurvatureResult {
    /// Field values (in `field_unit`) from the axis to the maximum field.
    pub field_values: Vec<f64>,
    /// The same samples as normalised `Hy`.
    pub normalized_field: Vec<f64>,
    pub field_unit: String,
    pub curves: Vec<FieldCurvatureCurve>,
    pub wavelengths_um: Vec<f64>,
    pub run_info: RunInfo,
}

impl FieldCurvatureResult {
    pub fn curve(&self, wavelength: usize) -> Option<&FieldCurvatureCurve> {
        self.curves.iter().find(|c| c.wavelength == wavelength)
    }
}

fn percent(real: f64, reference: f64) -> f64 {
    100.0 * (real - reference) / reference
}

/// Field curvature and distortion vs field (module docs).
pub fn field_curvature(
    tracer: &SequentialTracer,
    settings: &FieldCurvatureSettings,
) -> MkResult<FieldCurvatureResult> {
    field_curvature_job(tracer, settings, &JobContext::new())
}

/// [`field_curvature`] with cancellation and progress.
pub fn field_curvature_job(
    tracer: &SequentialTracer,
    settings: &FieldCurvatureSettings,
    ctx: &JobContext,
) -> MkResult<FieldCurvatureResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("field samples", settings.field_samples, 2, 100_001)?;
    check_finite("close-ray step", settings.close_ray_step)?;
    if !(settings.close_ray_step > 0.0 && settings.close_ray_step < 0.5) {
        return Err(mklight_core::MkError::InvalidModel(
            "close-ray step must be in (0, 0.5)".into(),
        ));
    }
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let hs = linspace(0.0, 1.0, settings.field_samples);
    let mut warnings = Vec::new();
    ctx.report(0.0, "field curvature");
    // Prepare every field sample once (real image heights solve here).
    let fields: Vec<Option<PreparedField>> = hs
        .par_iter()
        .map(|&h| tracer.prepare_field(FieldPoint::normalized(0.0, h)).ok())
        .collect();
    for (h, f) in hs.iter().zip(&fields) {
        if f.is_none() {
            warnings.push(format!("field Hy = {h}: could not be resolved"));
        }
    }
    ctx.check()?;
    let n0 = |w: usize| sys.n_object[w];
    let mut curves = Vec::with_capacity(wls.len());
    for (k, &w) in wls.iter().enumerate() {
        let fo = tracer.paraxial().at(w);
        let f_eff = n0(w) * fo.efl;
        let rows: Vec<[f64; 10]> = fields
            .par_iter()
            .zip(&hs)
            .map(|(f, &h)| {
                let mut row = [f64::NAN; 10];
                let Some(f) = f else { return row };
                if let Some(a) = coddington_focus(tracer, f, w) {
                    (row[0], row[1]) = a.axial();
                }
                if let Some(a) = close_ray_focus(tracer, f, w, settings.close_ray_step) {
                    (row[2], row[3]) = a.axial();
                }
                let c = tracer.chief(f, w);
                if c.status.is_ok() {
                    row[4] = c.ray.pos.y;
                }
                if let Some(p) = paraxial_image_point(tracer, f, w) {
                    row[5] = p[1];
                }
                if h == 0.0 {
                    row[6] = 0.0;
                    row[7] = if sys.object_at_infinity {
                        0.0
                    } else {
                        f64::NAN
                    };
                    row[8] = row[7];
                } else {
                    row[6] = percent(row[4], row[5]);
                    if let ObjectSource::Direction { dir } = f.source
                        && sys.object_at_infinity
                    {
                        let t = dir.y / dir.z;
                        row[7] = percent(row[4], f_eff * t);
                        row[8] = percent(row[4], f_eff * t.atan());
                    }
                }
                row
            })
            .collect();
        // Independent estimates of the same foci: flag disagreements.
        let bad: Vec<f64> = rows
            .iter()
            .zip(&hs)
            .filter(|(r, _)| (0..2).any(|j| (r[j] - r[j + 2]).abs() > 1e-3 * (1.0 + r[j].abs())))
            .map(|(_, &h)| h)
            .collect();
        if let (Some(first), Some(last)) = (bad.first(), bad.last()) {
            warnings.push(format!(
                "wavelength {w}: Coddington and close-ray foci differ by more than \
                 1e-3 (1 + |Δz|) mm at {} field sample(s), Hy = {first}…{last}",
                bad.len()
            ));
        }
        let col = |j: usize| rows.iter().map(|r| r[j]).collect::<Vec<f64>>();
        curves.push(FieldCurvatureCurve {
            wavelength: w,
            wavelength_um: sys.wavelengths_um[w],
            tangential: col(0),
            sagittal: col(1),
            tangential_close_rays: col(2),
            sagittal_close_rays: col(3),
            real_height: col(4),
            paraxial_height: col(5),
            distortion_percent: col(6),
            f_tan_theta_distortion_percent: col(7),
            f_theta_distortion_percent: col(8),
        });
        ctx.check()?;
        ctx.report((k + 1) as f64 / wls.len() as f64, "field curvature");
    }
    let scale = field_scale(tracer);
    Ok(FieldCurvatureResult {
        field_values: hs.iter().map(|h| h * scale).collect(),
        normalized_field: hs,
        field_unit: field_unit(sys.system().fields.kind).to_string(),
        curves,
        wavelengths_um: sys.wavelengths_um.clone(),
        run_info: run_info(
            timer,
            tracer,
            "field_curvature_distortion",
            settings,
            None,
            warnings,
        ),
    })
}

// ----- Grid distortion ------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GridDistortionSettings {
    /// Grid points per side (≥ 2).
    pub grid_size: usize,
    /// Half-width `a` of the normalised field square.
    pub half_width: f64,
    /// System wavelength index; the primary when `None`.
    pub wavelength: Option<usize>,
}

impl Default for GridDistortionSettings {
    fn default() -> Self {
        GridDistortionSettings {
            grid_size: 11,
            half_width: std::f64::consts::FRAC_1_SQRT_2,
            wavelength: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridPoint {
    /// Normalised field `(Hx, Hy)`.
    pub field: [f64; 2],
    /// Real chief-ray image point (image local frame, mm).
    #[serde(with = "nf::arr2")]
    pub real: [f64; 2],
    /// Paraxial image point (mm).
    #[serde(with = "nf::arr2")]
    pub paraxial: [f64; 2],
    /// Radial distortion (%); 0 at the centre.
    #[serde(with = "nf")]
    pub distortion_percent: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GridDistortionResult {
    pub grid_size: usize,
    pub wavelength: usize,
    /// Row-major, `Hy` outer and ascending, `Hx` inner and ascending.
    pub points: Vec<GridPoint>,
    /// Largest `|distortion|` over the grid (%).
    #[serde(with = "nf")]
    pub max_abs_distortion_percent: f64,
    pub run_info: RunInfo,
}

/// Grid distortion map (module docs).
pub fn grid_distortion(
    tracer: &SequentialTracer,
    settings: &GridDistortionSettings,
) -> MkResult<GridDistortionResult> {
    grid_distortion_job(tracer, settings, &JobContext::new())
}

/// [`grid_distortion`] with cancellation and progress (one block per grid row).
pub fn grid_distortion_job(
    tracer: &SequentialTracer,
    settings: &GridDistortionSettings,
    ctx: &JobContext,
) -> MkResult<GridDistortionResult> {
    let timer = RunTimer::start();
    check_count("grid size", settings.grid_size, 2, 1001)?;
    check_finite("grid half-width", settings.half_width)?;
    if !(settings.half_width > 0.0) {
        return Err(mklight_core::MkError::InvalidModel(
            "grid half-width must be positive".into(),
        ));
    }
    let w = wavelength_or_primary(tracer, settings.wavelength)?;
    let n = settings.grid_size;
    let t = linspace(-settings.half_width, settings.half_width, n);
    let mut points = Vec::with_capacity(n * n);
    let mut failed = 0usize;
    for (j, &hy) in t.iter().enumerate() {
        ctx.check()?;
        let row: Vec<GridPoint> = t
            .par_iter()
            .map(|&hx| {
                let mut g = GridPoint {
                    field: [hx, hy],
                    real: [f64::NAN; 2],
                    paraxial: [f64::NAN; 2],
                    distortion_percent: f64::NAN,
                };
                let Ok(f) = tracer.prepare_field(FieldPoint::normalized(hx, hy)) else {
                    return g;
                };
                let c = tracer.chief(&f, w);
                if c.status.is_ok() {
                    g.real = [c.ray.pos.x, c.ray.pos.y];
                }
                if let Some(p) = paraxial_image_point(tracer, &f, w) {
                    g.paraxial = p;
                }
                let rp = g.paraxial[0].hypot(g.paraxial[1]);
                g.distortion_percent = if hx == 0.0 && hy == 0.0 {
                    0.0
                } else {
                    percent(g.real[0].hypot(g.real[1]), rp)
                };
                g
            })
            .collect();
        failed += row.iter().filter(|g| g.real[0].is_nan()).count();
        points.extend(row);
        ctx.report((j + 1) as f64 / n as f64, "grid distortion");
    }
    let mut warnings = Vec::new();
    if failed > 0 {
        warnings.push(format!("{failed} grid chief rays failed"));
    }
    let max = points
        .iter()
        .map(|g| g.distortion_percent.abs())
        .filter(|v| v.is_finite())
        .fold(f64::NAN, f64::max);
    Ok(GridDistortionResult {
        grid_size: n,
        wavelength: w,
        points,
        max_abs_distortion_percent: max,
        run_info: run_info(timer, tracer, "grid_distortion", settings, None, warnings),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn principal_curvatures_of_known_surfaces() {
        // Sphere: both curvatures equal c everywhere.
        let s = SurfaceShape::sphere(40.0);
        let (km, ks) = principal_curvatures(&s, 25.0).unwrap();
        assert!((km - 0.025).abs() < 1e-15 && (ks - 0.025).abs() < 1e-15);
        // Paraboloid z = r²/(2R): κ_m = c/(1 + c²r²)^{3/2}, κ_s = c/√(1 + c²r²).
        let c = 0.01;
        let p = SurfaceShape::Conic {
            curvature: c,
            conic: -1.0,
        };
        let r: f64 = 30.0;
        let q = 1.0 + c * c * r * r;
        let (km, ks) = principal_curvatures(&p, r).unwrap();
        assert!((km - c / q.powf(1.5)).abs() < 1e-16);
        assert!((ks - c / q.sqrt()).abs() < 1e-16);
        // The same paraboloid as an even asphere with c = 0 and A4 = 0 must
        // differ; with c given as conic −1 plus zero terms it must agree.
        let a = SurfaceShape::EvenAsphere {
            curvature: c,
            conic: -1.0,
            coefficients: vec![0.0, 0.0],
        };
        let (am, as_) = principal_curvatures(&a, r).unwrap();
        assert!((am - km).abs() < 1e-16 && (as_ - ks).abs() < 1e-16);
        // Pure quartic z = A r⁴: f' = 4Ar³, f'' = 12Ar².
        let a4 = 1e-5;
        let q = SurfaceShape::EvenAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: vec![a4],
        };
        let r: f64 = 7.0;
        let fp = 4.0 * a4 * r.powi(3);
        let fpp = 12.0 * a4 * r * r;
        let (qm, qs) = principal_curvatures(&q, r).unwrap();
        assert!((qm - fpp / (1.0 + fp * fp).powf(1.5)).abs() < 1e-15);
        assert!((qs - fp / (r * (1.0 + fp * fp).sqrt())).abs() < 1e-15);
        // Outside the sphere's domain.
        assert!(principal_curvatures(&s, 41.0).is_none());
    }
}
