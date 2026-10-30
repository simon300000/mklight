//! Third-order (Seidel) aberrations, Petzval curvature, longitudinal
//! spherical aberration, chromatic focal shift and lateral colour.
//!
//! # Seidel sums
//!
//! The coefficients are computed surface by surface from the paraxial
//! marginal ray `(y, u)` and chief ray `(ȳ, ū)` of `mklight-trace` (scaled to
//! the full aperture and the maximum field) on the unfolded system, with the
//! signed reduced indices `n` (before) and `n'` (after) of the paraxial model,
//! so mirrors need no special case (`n' = −n`). With the refraction
//! invariants `A = n(u + yc)`, `Ā = n(ū + ȳc)` (`u`, `ū` before the surface),
//! the Lagrange invariant `H = n(ū y − u ȳ)` and `Δ(x) = x' − x`:
//!
//! | term | formula |
//! |---|---|
//! | `S_I` spherical | `−A² y Δ(u/n)` |
//! | `S_II` coma | `−A Ā y Δ(u/n)` |
//! | `S_III` astigmatism | `−Ā² y Δ(u/n)` |
//! | `S_IV` Petzval | `−H² c Δ(1/n)` |
//! | `S_V` distortion | `−Ā³ y Δ(1/n²) + Ā ȳ c (Ā y + H) Δ(1/n)` |
//! | `C_L` axial colour | `A y Δ(δn/n)` |
//! | `C_T` lateral colour | `Ā y Δ(δn/n)` |
//!
//! `S_V` is the division-free form of `(Ā/A)(S_III + S_IV)` (it follows from
//! `Δ(u/n) = A Δ(1/n²) − yc Δ(1/n)`), so it stays finite where `A = 0`.
//! `δn = n(λ_short) − n(λ_long)` carries the sign of the reduced index, so
//! mirrors contribute no colour. A conic or even asphere adds, with the
//! fourth-order sag coefficient beyond the base sphere `G = k c³/8 + A₄`:
//! `δS_I = 8G y⁴ Δn`, `δS_II = 8G y³ȳ Δn`, `δS_III = 8G y²ȳ² Δn`,
//! `δS_V = 8G y ȳ³ Δn` (`Δn = n' − n`). Ideal lenses (`Paraxial`) image
//! without aberration and contribute nothing; planes contribute only through
//! `Δ(u/n)` and `Δ(1/n²)` terms.
//!
//! Extended rotational sag bases use their exact r⁴ expansion (ANALYSIS §2.3).
//! Odd r¹/r³ terms are omitted with a warning. DOE contributions use the
//! algebraic infinite-index thin-plate limit, including phase dispersion;
//! its power adds no Petzval curvature. Non-rotational systems return
//! `NotApplicable` and undefined coefficients, with real-ray guidance.
//!
//! The **Petzval sum** is `P = Σ φ/(n n')` with `φ = c (n' − n)`
//! (`S_IV = H² P` summed), and the Petzval surface has radius
//! `R_p = −1/(n'_k P)` in the image surface's local `z` sense, with the
//! **signed** image-space reduced index `n'_k` (negative after an odd number
//! of reflections, where the light travels towards local `−z`): its sag
//! `h²/(2R_p)` is the axial Petzval focus shift at paraxial image height `h`.
//!
//! Wavefront coefficients (mm, and waves at the Seidel wavelength):
//! `W040 = S_I/8`, `W131 = S_II/2`, `W222 = S_III/2`,
//! `W220 = (S_III + S_IV)/4` (sagittal), `W220P = S_IV/4`, `W311 = S_V/2`,
//! `W020 = C_L/2`, `W111 = C_T`. With this module's signs a simple positive
//! lens has `S_I > 0` (under-corrected) and positive `W040`, which matches the
//! OPD sign of CONVENTIONS §4.
//!
//! Transverse third-order ray aberrations on the paraxial image plane follow
//! as (`H` normalised field, `(ρ, φ)` pupil polar with `ρ cos φ = P_y`,
//! `ω' = n'_k u'_k` the image-space reduced marginal slope)
//! `εy = (1/(2ω')) [S_I ρ³ cos φ + S_II H ρ² (2 + cos 2φ) + (3S_III + S_IV) H² ρ cos φ + S_V H³]`,
//! `εx = (1/(2ω')) [S_I ρ³ sin φ + S_II H ρ² sin 2φ + (S_III + S_IV) H² ρ sin φ]`
//! ([`SeidelResult::transverse_aberration`]; validated against real rays).
//! The colour terms give the third-order focal shift
//! `z(λ_short) − z(λ_long) = −n'_k C_L/ω'²` and lateral colour
//! `y(λ_short) − y(λ_long) = C_T H/ω'`.
//!
//! # Longitudinal aberration, focal shift and lateral colour
//!
//! * **Longitudinal aberration**: on-axis rays at pupil zones `P_y = ρ` are
//!   traced (unclipped) and their crossing with the chief ray is found in
//!   the plane normal to the chief ray (as for close rays in
//!   [`crate::curvature`]); the reported focus is its axial (`z`) distance
//!   from the image surface, and the aberration is that focus minus the
//!   primary wavelength's paraxial focus `−y_m/u'_m` (from the image surface).
//! * **Chromatic focal shift**: the paraxial focus of a wavelength sweep,
//!   relative to the primary wavelength's, from a copy of the system with the
//!   sweep as its wavelength set (same material library).
//! * **Lateral colour**: the real chief-ray height of every wavelength minus
//!   the primary's, and `y(λ_short) − y(λ_long)`, vs field along `+y`.

use mklight_core::job::RunTimer;
use mklight_core::{
    FieldKind, Interaction, JobContext, MkError, MkResult, ResolvedSystem, RunInfo, Snapshot,
    SurfaceShape, Vec3, Wavelength, WavelengthSet,
};
use mklight_trace::{FieldPoint, Paraxial, ParaxialModel, SequentialTracer, TraceOptions};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::util::{
    WavelengthSelection, check_count, field_scale, field_unit, linspace, nf, run_info,
    wavelength_or_primary,
};

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct SeidelSettings {
    /// Wavelength of the paraxial rays; the primary when `None`.
    pub wavelength: Option<usize>,
    /// Short and long wavelengths of `δn`; the shortest / longest system
    /// wavelengths when `None`.
    pub short_wavelength: Option<usize>,
    pub long_wavelength: Option<usize>,
}

/// The seven third-order coefficients (mm).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct SeidelTerms {
    #[serde(with = "nf")]
    pub s1: f64,
    #[serde(with = "nf")]
    pub s2: f64,
    #[serde(with = "nf")]
    pub s3: f64,
    #[serde(with = "nf")]
    pub s4: f64,
    #[serde(with = "nf")]
    pub s5: f64,
    #[serde(with = "nf")]
    pub cl: f64,
    #[serde(with = "nf")]
    pub ct: f64,
}

impl SeidelTerms {
    fn as_array(&self) -> [f64; 7] {
        [
            self.s1, self.s2, self.s3, self.s4, self.s5, self.cl, self.ct,
        ]
    }
    fn from_array(a: [f64; 7]) -> Self {
        SeidelTerms {
            s1: a[0],
            s2: a[1],
            s3: a[2],
            s4: a[3],
            s5: a[4],
            cl: a[5],
            ct: a[6],
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceSeidel {
    pub surface: usize,
    #[serde(flatten)]
    pub terms: SeidelTerms,
}

/// Wavefront aberration coefficients (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct WavefrontCoefficients {
    #[serde(with = "nf")]
    pub w040: f64,
    #[serde(with = "nf")]
    pub w131: f64,
    #[serde(with = "nf")]
    pub w222: f64,
    #[serde(with = "nf")]
    pub w220: f64,
    #[serde(with = "nf")]
    pub w220p: f64,
    #[serde(with = "nf")]
    pub w311: f64,
    #[serde(with = "nf")]
    pub w020: f64,
    #[serde(with = "nf")]
    pub w111: f64,
}

impl WavefrontCoefficients {
    fn from_sums(s: &SeidelTerms, scale: f64) -> Self {
        WavefrontCoefficients {
            w040: s.s1 / 8.0 * scale,
            w131: s.s2 / 2.0 * scale,
            w222: s.s3 / 2.0 * scale,
            w220: (s.s3 + s.s4) / 4.0 * scale,
            w220p: s.s4 / 4.0 * scale,
            w311: s.s5 / 2.0 * scale,
            w020: s.cl / 2.0 * scale,
            w111: s.ct * scale,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SeidelStatus {
    Applicable,
    NotApplicable,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeidelResult {
    pub status: SeidelStatus,
    pub reason: Option<String>,
    pub wavelength: usize,
    pub wavelength_um: f64,
    pub short_wavelength: usize,
    pub long_wavelength: usize,
    /// Per surface `1..N−1` (the image contributes nothing).
    pub surfaces: Vec<SurfaceSeidel>,
    pub sums: SeidelTerms,
    /// `H = n(ū y − u ȳ)` (mm·rad).
    #[serde(with = "nf")]
    pub lagrange_invariant: f64,
    /// `Σ φ/(n n')` (1/mm).
    #[serde(with = "nf")]
    pub petzval_sum: f64,
    /// Petzval radius (mm, image local `z` sense); ∞ when flat.
    #[serde(with = "nf")]
    pub petzval_radius: f64,
    /// Image-space reduced marginal slope `ω' = n'_k u'_k`.
    #[serde(with = "nf")]
    pub image_marginal_slope: f64,
    /// Wavefront coefficients in mm.
    pub wavefront_mm: WavefrontCoefficients,
    /// Wavefront coefficients in waves at `wavelength_um`.
    pub wavefront_waves: WavefrontCoefficients,
    pub run_info: RunInfo,
}

impl SeidelResult {
    /// Third-order transverse aberration `[εx, εy]` (mm) on the paraxial
    /// image plane at normalised field `H` (along `+y`) and pupil `(px, py)`
    /// (module docs). `include_distortion` adds the `S_V H³` chief-ray term,
    /// i.e. the aberration relative to the paraxial chief ray rather than the
    /// real one.
    pub fn transverse_aberration(
        &self,
        h: f64,
        px: f64,
        py: f64,
        include_distortion: bool,
    ) -> [f64; 2] {
        let s = &self.sums;
        let k = 1.0 / (2.0 * self.image_marginal_slope);
        let r2 = px.mul_add(px, py * py);
        // ρ³cos φ = r² py, ρ²(2 + cos 2φ) = 3py² + px², ρ² sin 2φ = 2 px py.
        let ey = s.s1 * r2 * py
            + s.s2 * h * (3.0 * py * py + px * px)
            + (3.0 * s.s3 + s.s4) * h * h * py
            + if include_distortion {
                s.s5 * h * h * h
            } else {
                0.0
            };
        let ex = s.s1 * r2 * px + s.s2 * h * 2.0 * px * py + (s.s3 + s.s4) * h * h * px;
        [k * ex, k * ey]
    }
}

fn select_short_long(
    sys: &ResolvedSystem,
    s: Option<usize>,
    l: Option<usize>,
) -> MkResult<(usize, usize)> {
    let wl = &sys.wavelengths_um;
    let check = |w: usize| {
        if w < wl.len() {
            Ok(w)
        } else {
            Err(MkError::InvalidModel(format!(
                "wavelength {w} does not exist"
            )))
        }
    };
    let argmin = (0..wl.len())
        .min_by(|&a, &b| wl[a].total_cmp(&wl[b]))
        .unwrap_or(0);
    let argmax = (0..wl.len())
        .max_by(|&a, &b| wl[a].total_cmp(&wl[b]))
        .unwrap_or(0);
    Ok((check(s.unwrap_or(argmin))?, check(l.unwrap_or(argmax))?))
}

/// Vertex curvature and total r⁴ sag coefficient, without numerical differencing.
fn fourth_order(shape: &SurfaceShape) -> (f64, f64) {
    let c = mklight_core::surface_math::vertex_curvatures(shape)[2];
    let base = (1.0 + shape.conic()) * shape.curvature().powi(3) / 8.0;
    let fourth = match shape {
        SurfaceShape::EvenAsphere { coefficients, .. } => {
            base + coefficients.first().copied().unwrap_or(0.0)
        }
        SurfaceShape::OddAsphere { coefficients, .. } => {
            base + coefficients.get(3).copied().unwrap_or(0.0)
        }
        SurfaceShape::QCon {
            coefficients,
            norm_radius,
            ..
        } => {
            // P_m^(0,4)(-1) = (-1)^m binomial(m+4,4).
            let mut q = 1.0;
            let mut sum = mklight_core::math::KahanSum::default();
            for (m, a) in coefficients.iter().enumerate() {
                sum.add(a * q);
                q *= -((m + 5) as f64) / (m + 1) as f64;
            }
            base + sum.value() / norm_radius.powi(4)
        }
        SurfaceShape::XYPolynomial {
            coefficients,
            norm_radius,
            ..
        } => {
            base + coefficients
                .get(mklight_core::surface_math::xy_index(0, 4))
                .copied()
                .unwrap_or(0.0)
                / norm_radius.powi(4)
        }
        SurfaceShape::ZernikeSag {
            coefficients,
            norm_radius,
            ordering,
            ..
        } => {
            let mut sum = mklight_core::math::KahanSum::default();
            for (j, a) in coefficients.iter().enumerate() {
                let (n, m) = mklight_core::surface_math::zernike_term(*ordering, j + 1)
                    .expect("validated Zernike count");
                let p = n / 2;
                if m == 0 && p >= 2 {
                    let p = p as f64;
                    let q = (p - 1.0) * p * (p + 1.0) * (p + 2.0) / 4.0;
                    let sign = if (n / 2).is_multiple_of(2) { 1.0 } else { -1.0 };
                    let norm =
                        if *ordering == mklight_core::model::surface::ZernikeOrdering::Standard {
                            ((n + 1) as f64).sqrt()
                        } else {
                            1.0
                        };
                    sum.add(a * sign * norm * q);
                }
            }
            base + sum.value() / norm_radius.powi(4)
        }
        SurfaceShape::QBfs {
            coefficients,
            norm_radius,
            curvature,
        } => {
            let (q0, q1) = qbfs_vertex(coefficients);
            base + (q1 - q0) / norm_radius.powi(4)
                + 0.5 * curvature.powi(2) * q0 / norm_radius.powi(2)
        }
        _ => base,
    };
    (c, fourth)
}

/// Qbfs sum and its t derivative at t=0, from the defining triangular basis.
fn qbfs_vertex(coefficients: &[f64]) -> (f64, f64) {
    let mut f = [0.0; mklight_core::surface_math::MAX_Q_TERMS];
    let mut g = f;
    let mut h = f;
    f[0] = 2.0;
    f[1] = 19.0_f64.sqrt() / 2.0;
    g[0] = -0.5;
    for m in 2..f.len() {
        h[m - 2] = -((m * (m - 1)) as f64) / (2.0 * f[m - 2]);
        g[m - 1] = -(1.0 + g[m - 2] * h[m - 2]) / f[m - 1];
        f[m] = ((m * (m + 1) + 3) as f64 - g[m - 1].powi(2) - h[m - 2].powi(2)).sqrt();
    }
    let (mut p0, mut p1) = ([2.0, 0.0], [6.0, -8.0]);
    let (mut q0, mut q1) = ([0.0; 2], [0.0; 2]);
    let mut sums = [mklight_core::math::KahanSum::default(); 2];
    for (m, a) in coefficients.iter().enumerate() {
        let p = if m == 0 { p0 } else { p1 };
        let q = std::array::from_fn(|i| {
            (p[i]
                - if m > 0 { g[m - 1] * q1[i] } else { 0.0 }
                - if m > 1 { h[m - 2] * q0[i] } else { 0.0 })
                / f[m]
        });
        for i in 0..2 {
            sums[i].add(a * q[i]);
        }
        (q0, q1) = (q1, q);
        if m > 0 {
            (p0, p1) = (
                p1,
                [2.0 * p1[0] - p0[0], -4.0 * p1[0] + 2.0 * p1[1] - p0[1]],
            );
        }
    }
    (sums[0].value(), sums[1].value())
}

/// Seidel sums (module docs).
pub fn seidel(tracer: &SequentialTracer, settings: &SeidelSettings) -> MkResult<SeidelResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    let w = wavelength_or_primary(tracer, settings.wavelength)?;
    let (ws, wlong) = select_short_long(sys, settings.short_wavelength, settings.long_wavelength)?;
    if mklight_core::surface_math::system_symmetry(sys.system())
        != mklight_core::surface_math::Symmetry::Rotational
    {
        let reason = "Seidel coefficients are not applicable to non-rotational systems; use full real-ray spots, fans, grid distortion and field curvature".to_string();
        let sums = SeidelTerms::from_array([f64::NAN; 7]);
        return Ok(SeidelResult {
            status: SeidelStatus::NotApplicable,
            reason: Some(reason.clone()),
            wavelength: w,
            wavelength_um: sys.wavelengths_um[w],
            short_wavelength: ws,
            long_wavelength: wlong,
            surfaces: Vec::new(),
            lagrange_invariant: f64::NAN,
            petzval_sum: f64::NAN,
            petzval_radius: f64::NAN,
            image_marginal_slope: f64::NAN,
            wavefront_mm: WavefrontCoefficients::from_sums(&sums, 1.0),
            wavefront_waves: WavefrontCoefficients::from_sums(&sums, 1.0),
            sums,
            run_info: run_info(timer, tracer, "seidel", settings, None, vec![reason]),
        });
    }
    let fo = tracer.paraxial().at(w);
    let model = ParaxialModel::new(sys, w);
    let ns = ParaxialModel::new(sys, ws).reduced_index;
    let nl = ParaxialModel::new(sys, wlong).reduced_index;
    let nn = &model.reduced_index;
    let m = &fo.marginal;
    let c = &fo.chief;
    let h = fo.lagrange_invariant;
    let img = sys.image_index();
    let mut warnings = Vec::new();
    let mut surfaces = Vec::with_capacity(img);
    let mut petzval = 0.0;
    for j in 1..img {
        let s = &sys.surfaces[j];
        let mut t = [0.0f64; 7];
        let ideal = matches!(s.shape, SurfaceShape::Paraxial { .. });
        if ideal && s.interaction != Interaction::Dummy {
            warnings.push(format!(
                "surface {j}: ideal lens treated as free of third-order aberrations"
            ));
        }
        if !s.is_passive() && !ideal {
            let (n, n2) = (nn[j - 1], nn[j]);
            let base = match &s.shape {
                SurfaceShape::Diffractive { base, .. } => base.as_ref(),
                shape => shape,
            };
            let (cv, fourth) = fourth_order(base);
            if let SurfaceShape::OddAsphere { coefficients, .. } = base
                && [0, 2]
                    .iter()
                    .any(|&i| coefficients.get(i).is_some_and(|v| *v != 0.0))
            {
                warnings.push(format!("surface {j}: non-analytic r¹/r³ terms omitted; Seidel describes only the regular fourth-order equivalent, use real rays for their aberrations"));
            }
            let (y, yb) = (m.y[j], c.y[j]);
            let a = n.mul_add(y * cv, m.nu[j - 1]);
            let ab = n.mul_add(yb * cv, c.nu[j - 1]);
            let du_n = m.nu[j] / (n2 * n2) - m.nu[j - 1] / (n * n);
            let d1n = 1.0 / n2 - 1.0 / n;
            let d1n2 = 1.0 / (n2 * n2) - 1.0 / (n * n);
            t[0] = -a * a * y * du_n;
            t[1] = -a * ab * y * du_n;
            t[2] = -ab * ab * y * du_n;
            t[3] = -h * h * cv * d1n;
            t[4] = -ab * ab * ab * y * d1n2 + ab * yb * cv * ab.mul_add(y, h) * d1n;
            let dn = n2 - n;
            let g = fourth - cv.powi(3) / 8.0;
            if g != 0.0 {
                let e = 8.0 * g * dn * y;
                t[0] += e * y * y * y;
                t[1] += e * y * y * yb;
                t[2] += e * y * yb * yb;
                t[4] += e * yb * yb * yb;
            }
            let dr = |k: usize| (ns[k] - nl[k]) / nn[k];
            let dd = dr(j) - dr(j - 1);
            t[5] = a * y * dd;
            t[6] = ab * y * dd;
            petzval += cv * (n2 - n) / (n * n2);
            if let SurfaceShape::Diffractive { phase, .. } = &s.shape {
                let scale =
                    phase.trace_order as f64 * sys.wavelengths_um[w] * 1e-3 / std::f64::consts::TAU;
                let p = -2.0 * scale * phase.coefficients.first().copied().unwrap_or(0.0)
                    / phase.norm_radius.powi(2);
                let q4 = scale * phase.coefficients.get(1).copied().unwrap_or(0.0)
                    / phase.norm_radius.powi(4);
                let (a2, ab2) = (a - p * y, ab - p * yb);
                let (vi, vo) = (m.nu[j - 1] / n.powi(2), m.nu[j] / n2.powi(2));
                // Exact N -> infinity limit of two coincident interfaces:
                // c1=c, c2=c-p/N, z2_4=z1_4+q4/N. No large index subtraction.
                let gg = dn * g - q4 - 3.0 * cv * cv * p / 8.0;
                t[0] = y * (a * a * vi - a2 * a2 * vo) + 8.0 * gg * y.powi(4);
                t[1] = y * (a * ab * vi - a2 * ab2 * vo) + 8.0 * gg * y.powi(3) * yb;
                t[2] = y * (ab * ab * vi - ab2 * ab2 * vo) + 8.0 * gg * y.powi(2) * yb.powi(2);
                t[4] = y * (ab.powi(3) / n.powi(2) - ab2.powi(3) / n2.powi(2))
                    + yb * cv * (ab2 * (ab2 * y + h) / n2 - ab * (ab * y + h) / n)
                    + 8.0 * gg * y * yb.powi(3);
                let dispersion =
                    (sys.wavelengths_um[ws] - sys.wavelengths_um[wlong]) / sys.wavelengths_um[w];
                t[5] = y * (a2 * dr(j) - a * dr(j - 1) + p * y * dispersion);
                t[6] = y * (ab2 * dr(j) - ab * dr(j - 1) + p * yb * dispersion);
            }
        }
        surfaces.push(SurfaceSeidel {
            surface: j,
            terms: SeidelTerms::from_array(t),
        });
    }
    // Compensated column sums in surface order.
    let mut sums = [0.0f64; 7];
    for (k, v) in sums.iter_mut().enumerate() {
        *v = surfaces
            .iter()
            .map(|s| s.terms.as_array()[k])
            .collect::<mklight_core::math::KahanSum>()
            .value();
    }
    let sums = SeidelTerms::from_array(sums);
    // Signed: the image-space light direction sets the side of the surface.
    let n_img = nn[img - 1];
    let wl_mm = sys.wavelengths_um[w] * 1e-3;
    Ok(SeidelResult {
        status: SeidelStatus::Applicable,
        reason: None,
        wavelength: w,
        wavelength_um: sys.wavelengths_um[w],
        short_wavelength: ws,
        long_wavelength: wlong,
        surfaces,
        lagrange_invariant: h,
        petzval_sum: petzval,
        petzval_radius: if petzval != 0.0 {
            -1.0 / (n_img * petzval)
        } else {
            f64::INFINITY
        },
        image_marginal_slope: m.nu[img - 1],
        wavefront_mm: WavefrontCoefficients::from_sums(&sums, 1.0),
        wavefront_waves: WavefrontCoefficients::from_sums(&sums, 1.0 / wl_mm),
        sums,
        run_info: run_info(timer, tracer, "seidel", settings, None, warnings),
    })
}

// ----- Longitudinal aberration ----------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LongitudinalSettings {
    pub axes: crate::util::FieldAxes,
    /// Zones 0..1 for rotational systems; signed -1..1 otherwise.
    pub pupil_samples: usize,
    pub wavelengths: WavelengthSelection,
}

impl Default for LongitudinalSettings {
    fn default() -> Self {
        LongitudinalSettings {
            axes: Default::default(),
            pupil_samples: 21,
            wavelengths: WavelengthSelection::All,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalCurve {
    pub axis: mklight_trace::ParaxialAxis,
    pub wavelength: usize,
    pub wavelength_um: f64,
    /// This wavelength's paraxial focus from the image surface (mm, local z).
    #[serde(with = "nf")]
    pub paraxial_focus: f64,
    /// Real focus of each zone from the image surface (mm).
    #[serde(with = "nf::vec")]
    pub focus: Vec<f64>,
    /// `focus − paraxial focus of the primary wavelength` (mm).
    #[serde(with = "nf::vec")]
    pub aberration: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LongitudinalResult {
    /// Pupil zones along each curve's selected axis.
    pub pupil: Vec<f64>,
    #[serde(with = "nf")]
    pub primary_paraxial_focus: f64,
    pub curves: Vec<LongitudinalCurve>,
    pub run_info: RunInfo,
}

/// Paraxial focus of wavelength `w`, from the image surface along local z.
fn paraxial_focus(p: &Paraxial, w: usize) -> f64 {
    let m = &p.at(w).marginal;
    let img = m.y.len() - 1;
    -m.y[img] / m.u[img - 1]
}

/// Where a ray `(p, d)` crosses the chief ray `(pc, c)`, as the distance
/// along `c` from `pc` (least-squares crossing in the plane normal to `c`).
pub(crate) fn crossing_along_chief(pc: Vec3, c: Vec3, p: Vec3, d: Vec3) -> f64 {
    let dc = d.dot(c);
    let q = p + d * ((pc - p).dot(c) / dc) - pc;
    let t = d / dc - c;
    let t2 = t.norm2();
    if t2 > 0.0 {
        -q.dot(t) / t2
    } else {
        f64::INFINITY
    }
}

/// Longitudinal (spherical) aberration vs pupil zone (module docs).
pub fn longitudinal_aberration(
    tracer: &SequentialTracer,
    settings: &LongitudinalSettings,
) -> MkResult<LongitudinalResult> {
    longitudinal_aberration_job(tracer, settings, &JobContext::new())
}

pub fn longitudinal_aberration_job(
    tracer: &SequentialTracer,
    settings: &LongitudinalSettings,
    ctx: &JobContext,
) -> MkResult<LongitudinalResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("pupil samples", settings.pupil_samples, 2, 100_001)?;
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let field = tracer.prepare_field(FieldPoint::normalized(0.0, 0.0))?;
    let lo = if mklight_core::surface_math::system_symmetry(sys.system())
        == mklight_core::surface_math::Symmetry::Rotational
    {
        0.0
    } else {
        -1.0
    };
    let pupil = linspace(lo, 1.0, settings.pupil_samples);
    let p0 = paraxial_focus(tracer.paraxial(), sys.primary_wavelength);
    let opts = TraceOptions::unclipped();
    let mut curves = Vec::new();
    let axes = settings.axes.components(tracer);
    let focus = |fo: &mklight_trace::FirstOrder| {
        let m = &fo.marginal;
        -m.y[sys.image_index()] / m.u[sys.image_index() - 1]
    };
    for (axis_index, &component) in axes.iter().enumerate() {
        let axis = if component == 0 {
            mklight_trace::ParaxialAxis::X
        } else {
            mklight_trace::ParaxialAxis::Y
        };
        let primary = mklight_trace::paraxial::first_order_axis(sys, sys.primary_wavelength, axis)?;
        let primary_focus = focus(&primary);
        for (wi, &w) in wls.iter().enumerate() {
            ctx.check()?;
            let pf = focus(&mklight_trace::paraxial::first_order_axis(sys, w, axis)?);
            let chief = tracer.trace(&field, w, 0.0, 0.0, &opts);
            let focus: Vec<f64> = pupil
                .par_iter()
                .map(|&rho| {
                    ctx.check()?;
                    if rho == 0.0 {
                        return Ok(pf);
                    }
                    if !chief.status.is_ok() {
                        return Ok(f64::NAN);
                    }
                    let (px, py) = if component == 0 {
                        (rho, 0.0)
                    } else {
                        (0.0, rho)
                    };
                    let r = tracer.trace(&field, w, px, py, &opts);
                    if !r.status.is_ok() {
                        return Ok(f64::NAN);
                    }
                    let (pc, c) = (chief.ray.pos, chief.ray.dir);
                    let s = crossing_along_chief(pc, c, r.ray.pos, r.ray.dir);
                    Ok(pc.z + s * c.z)
                })
                .collect::<MkResult<_>>()?;
            curves.push(LongitudinalCurve {
                axis,
                wavelength: w,
                wavelength_um: sys.wavelengths_um[w],
                paraxial_focus: pf,
                aberration: focus.iter().map(|f| f - primary_focus).collect(),
                focus,
            });
            ctx.report(
                (axis_index * wls.len() + wi + 1) as f64 / (axes.len() * wls.len()) as f64,
                "longitudinal aberration",
            );
        }
    }
    Ok(LongitudinalResult {
        pupil,
        primary_paraxial_focus: p0,
        curves,
        run_info: run_info(
            timer,
            tracer,
            "longitudinal_aberration",
            settings,
            None,
            Vec::new(),
        ),
    })
}

// ----- Chromatic focal shift ---------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ChromaticFocalShiftSettings {
    pub samples: usize,
    /// Sweep range (µm); the system's shortest / longest wavelength when `None`.
    pub min_um: Option<f64>,
    pub max_um: Option<f64>,
}

impl Default for ChromaticFocalShiftSettings {
    fn default() -> Self {
        ChromaticFocalShiftSettings {
            samples: 41,
            min_um: None,
            max_um: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChromaticFocalShiftResult {
    #[serde(with = "nf::vec")]
    pub focal_shift_x: Vec<f64>,
    #[serde(with = "nf::vec")]
    pub paraxial_focus_x: Vec<f64>,
    pub wavelengths_um: Vec<f64>,
    /// Paraxial focus minus the primary wavelength's (mm).
    #[serde(with = "nf::vec")]
    pub focal_shift: Vec<f64>,
    /// Paraxial focus from the image surface (mm, local z).
    #[serde(with = "nf::vec")]
    pub paraxial_focus: Vec<f64>,
    pub primary_wavelength_um: f64,
    /// `max − min` of `focal_shift` over the sweep (mm).
    #[serde(with = "nf")]
    pub range: f64,
    pub run_info: RunInfo,
}

/// Chromatic focal shift vs wavelength (module docs).
pub fn chromatic_focal_shift(
    tracer: &SequentialTracer,
    settings: &ChromaticFocalShiftSettings,
) -> MkResult<ChromaticFocalShiftResult> {
    chromatic_focal_shift_job(tracer, settings, &JobContext::new())
}

pub fn chromatic_focal_shift_job(
    tracer: &SequentialTracer,
    settings: &ChromaticFocalShiftSettings,
    ctx: &JobContext,
) -> MkResult<ChromaticFocalShiftResult> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("wavelength samples", settings.samples, 2, 10_001)?;
    let lo = settings.min_um.unwrap_or_else(|| {
        sys.wavelengths_um
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
    });
    let hi = settings
        .max_um
        .unwrap_or_else(|| sys.wavelengths_um.iter().copied().fold(0.0, f64::max));
    if !(lo > 0.0 && hi >= lo && hi.is_finite()) {
        return Err(MkError::InvalidModel(
            "wavelength range must be positive and ascending".into(),
        ));
    }
    let sweep = linspace(lo, hi, settings.samples);
    let primary_um = sys.wavelengths_um[sys.primary_wavelength];
    let mut osys = sys.system().clone();
    let mut wl: Vec<Wavelength> = sweep
        .iter()
        .map(|&um| Wavelength { um, weight: 1.0 })
        .collect();
    wl.push(Wavelength {
        um: primary_um,
        weight: 1.0,
    });
    osys.wavelengths = WavelengthSet {
        primary: wl.len() - 1,
        wavelengths: wl,
    };
    let snap = Snapshot::new(osys)?;
    let rs = ResolvedSystem::new(&snap, sys.library.clone())?;
    ctx.check()?;
    let xy = mklight_trace::ParaxialXY::new_job(&rs, ctx)?;
    let par = Paraxial {
        wavelengths: xy.y,
        primary: rs.primary_wavelength,
        warnings: xy.run_info.warnings,
    };
    let foci: Vec<f64> = (0..rs.wavelengths_um.len())
        .map(|w| paraxial_focus(&par, w))
        .collect();
    let p0 = foci[foci.len() - 1];
    let focus: Vec<f64> = foci[..sweep.len()].to_vec();
    let shift: Vec<f64> = focus.iter().map(|f| f - p0).collect();
    let range = shift.iter().copied().fold(f64::NEG_INFINITY, f64::max)
        - shift.iter().copied().fold(f64::INFINITY, f64::min);
    let x_focus = |fo: &mklight_trace::FirstOrder| {
        let m = &fo.marginal;
        -m.y[rs.image_index()] / m.u[rs.image_index() - 1]
    };
    let x0 = x_focus(xy.x.last().unwrap());
    let paraxial_focus_x: Vec<f64> = xy.x[..sweep.len()].iter().map(x_focus).collect();
    let focal_shift_x = paraxial_focus_x.iter().map(|v| v - x0).collect();
    Ok(ChromaticFocalShiftResult {
        paraxial_focus_x,
        focal_shift_x,
        wavelengths_um: sweep,
        focal_shift: shift,
        paraxial_focus: focus,
        primary_wavelength_um: primary_um,
        range,
        run_info: run_info(
            timer,
            tracer,
            "chromatic_focal_shift",
            settings,
            None,
            rs.warnings.clone(),
        ),
    })
}

// ----- Lateral colour -----------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LateralColourSettings {
    pub field_axes: crate::util::FieldAxes,
    pub field_samples: usize,
    pub short_wavelength: Option<usize>,
    pub long_wavelength: Option<usize>,
}

impl Default for LateralColourSettings {
    fn default() -> Self {
        LateralColourSettings {
            field_axes: Default::default(),
            field_samples: 21,
            short_wavelength: None,
            long_wavelength: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LateralColourCurve {
    pub wavelength: usize,
    pub wavelength_um: f64,
    /// Real chief height minus the primary wavelength's (mm).
    #[serde(with = "nf::vec")]
    pub offset: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LateralColourResult {
    pub field_coordinates: Vec<[f64; 2]>,
    pub field_values: Vec<f64>,
    pub normalized_field: Vec<f64>,
    pub field_unit: String,
    pub short_wavelength: usize,
    pub long_wavelength: usize,
    /// `y(λ_short) − y(λ_long)` (mm).
    #[serde(with = "nf::vec")]
    pub lateral_colour: Vec<f64>,
    pub curves: Vec<LateralColourCurve>,
    /// Airy radius at the primary wavelength (mm), for scale.
    #[serde(with = "nf")]
    pub airy_radius: f64,
    pub run_info: RunInfo,
}

/// Lateral colour vs field (module docs).
pub fn lateral_colour(
    tracer: &SequentialTracer,
    settings: &LateralColourSettings,
) -> MkResult<LateralColourResult> {
    lateral_colour_job(tracer, settings, &JobContext::new())
}

/// [`lateral_colour`] with cancellation and progress.
pub fn lateral_colour_job(
    tracer: &SequentialTracer,
    settings: &LateralColourSettings,
    ctx: &JobContext,
) -> MkResult<LateralColourResult> {
    ctx.check()?;
    let timer = RunTimer::start();
    let sys = tracer.system();
    check_count("field samples", settings.field_samples, 2, 100_001)?;
    let (ws, wl) = select_short_long(sys, settings.short_wavelength, settings.long_wavelength)?;
    let sweep = crate::util::field_sweep(tracer, settings.field_samples, settings.field_axes);
    let hs = &sweep.heights;
    let nw = sys.wavelengths_um.len();
    let primary = sys.primary_wavelength;
    ctx.report(0.0, "lateral colour");
    let heights: Vec<Vec<f64>> = sweep
        .coordinates
        .par_iter()
        .zip(&sweep.components)
        .map(
            |(&[hx, hy], &axis)| match tracer.prepare_field(FieldPoint::normalized(hx, hy)) {
                Ok(f) => (0..nw)
                    .map(|w| {
                        let c = tracer.chief(&f, w);
                        if c.status.is_ok() {
                            if axis == 0 { c.ray.pos.x } else { c.ray.pos.y }
                        } else {
                            f64::NAN
                        }
                    })
                    .collect(),
                Err(_) => vec![f64::NAN; nw],
            },
        )
        .collect();
    ctx.check()?;
    ctx.report(1.0, "lateral colour");
    let curves = (0..nw)
        .map(|w| LateralColourCurve {
            wavelength: w,
            wavelength_um: sys.wavelengths_um[w],
            offset: heights.iter().map(|y| y[w] - y[primary]).collect(),
        })
        .collect();
    let fo = tracer.paraxial().primary();
    let scale = field_scale(tracer);
    let kind: FieldKind = sys.system().fields.kind;
    Ok(LateralColourResult {
        field_values: hs.iter().map(|h| h * scale).collect(),
        normalized_field: hs.clone(),
        field_coordinates: sweep.coordinates,
        field_unit: field_unit(kind).to_string(),
        short_wavelength: ws,
        long_wavelength: wl,
        lateral_colour: heights.iter().map(|y| y[ws] - y[wl]).collect(),
        curves,
        airy_radius: 1.22 * sys.wavelengths_um[primary] * 1e-3 * fo.working_f_number,
        run_info: run_info(timer, tracer, "lateral_colour", settings, None, Vec::new()),
    })
}
