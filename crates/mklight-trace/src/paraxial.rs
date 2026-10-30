//! Paraxial (first-order) optics by y–nu tracing.
//!
//! # Model
//!
//! The paraxial trace works on the **unfolded, rotationally symmetric
//! equivalent** of the system (the y section for anamorphic shapes): tilts, decenters and coordinate breaks are
//! ignored (a warning is issued when any exist), thicknesses are used with
//! their signs, and mirrors are handled with **signed reduced indices**
//! `N_j = σ_j n_j`, where `σ` starts at `+1` in object space and flips at
//! every mirror (CONVENTIONS §2). The coordinate `z` along the unfolded axis
//! is each surface's local `z`, so all positions below are in the same sign
//! convention as thicknesses (e.g. a focus in front of a concave mirror has a
//! negative distance).
//!
//! With `y_j` the height at surface `j` and `ω_j = N_j u_j` the reduced slope
//! after it (`u = dy/dz`):
//!
//! * transfer: `y_{j+1} = y_j + t_j ω_j / N_j`;
//! * refraction: `ω_j = ω_{j−1} − y_j φ_j` with `φ_j = c_j (N_j − N_{j−1})` for
//!   conics/even aspheres; extended shapes use their vertex Hessian and
//!   add DOE power (SURFACES.md). The formal odd-asphere model omits r¹.
//!   `φ_j = |N_j| / f` for an ideal lens, and `0` for planes, dummies (any shape),
//!   coordinate breaks and the image surface.
//!
//! Two basis rays span every paraxial ray: `A` (`y₁ = 1`, `u₀ = 0`) and `B`
//! (`y₁ = 0`, `u₀ = 1`), where index 1 is the first surface and `u₀` is the
//! object-space slope.
//!
//! # Definitions (per wavelength)
//!
//! * EFL `= −1/ω'_A` (system power referenced to unit index, so the rear focal
//!   length is `n'·EFL`); BFL `= −y_A/u'_A` at the last surface before the
//!   image; FFL `= ω'_B/ω'_A` is the *signed position* of the front focal point
//!   relative to surface 1 (negative: in front).
//! * Entrance pupil position `z_EP = y_B(stop)/y_A(stop)`, relative to surface
//!   1 in object space; `±∞` (object-space telecentric) when `y_A(stop) = 0`.
//! * The marginal ray starts at the axial object point (parallel to the axis
//!   at infinity) and is scaled by the aperture specification:
//!   EPD → `r_EP`; image-space F/# → `r_EP = f'/(2F)` with the rear focal
//!   length `f' = |N'|·|EFL|` (core: F/# `= f'/EPD`); object NA →
//!   `u₀ = tan(asin(NA/n₀))` (so a real ray to the pupil edge has exactly that
//!   NA); paraxial working F/# → `|ω'_m| = 1/(2F_w)`; float by stop size →
//!   `|y_m(stop)|` equals the stop's fixed semi-diameter.
//! * The chief ray passes through the stop centre and is scaled to the maximum
//!   radial field: `u₀ = tan θ_max` (angles), `y₀ = h_max` (object heights), or
//!   image-surface height `= h'_max` (paraxial and real image heights; real
//!   image height uses its paraxial equivalent here).
//! * Exit pupil position `z_XP = −y_c/u'_c` relative to the image surface;
//!   diameter `2|y_m + u'_m z_XP|`.
//! * Image-space F/# `= f' / EPD = |N'||EFL| / EPD` (equal to the working F/#
//!   at infinity only when image space is air); paraxial working F/#
//!   `= 1/(2|ω'_m|)`;
//!   object NA `= n₀ sin(atan|u_m|)`; image NA `= |N'| sin(atan|u'_m|)`;
//!   lateral magnification `m = ω_m(object)/ω'_m` (0 for an object at
//!   infinity); Lagrange invariant `H = ω_c y_m − ω_m y_c` (at surface 1);
//!   total track `= Σ_{j=1}^{N−2} t_j` (signed).
//!
//! Each wavelength's values are computed with that wavelength's indices.
//! Ray generation uses the primary wavelength's pupil (CONVENTIONS §3).

use mklight_core::{
    ApertureSpec, FieldKind, Interaction, MkError, MkResult, ResolvedSystem, SurfaceShape,
};
use serde::{Deserialize, Serialize};

/// Beyond this magnitude (mm) a pupil position is treated as infinite.
const TELECENTRIC_LIMIT: f64 = 1e12;

/// Per-surface paraxial data of one wavelength.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParaxialModel {
    /// Signed reduced index after each surface (`N_0` is object space).
    pub reduced_index: Vec<f64>,
    /// Surface power `φ_j` (1/mm, reduced).
    pub power: Vec<f64>,
    /// Thickness after each surface.
    pub thickness: Vec<f64>,
    pub object_at_infinity: bool,
}

/// Heights, slopes and reduced slopes of one paraxial ray at every surface.
/// `y[0]` is the object height (0 for an object at infinity, where it is
/// undefined); `u[j]`, `nu[j]` are the slope and reduced slope after surface `j`
/// (`u[0]` is the object-space slope).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ParaxialRayTable {
    pub y: Vec<f64>,
    pub u: Vec<f64>,
    pub nu: Vec<f64>,
}

impl ParaxialRayTable {
    fn scaled(&self, s: f64) -> Self {
        ParaxialRayTable {
            y: self.y.iter().map(|v| v * s).collect(),
            u: self.u.iter().map(|v| v * s).collect(),
            nu: self.nu.iter().map(|v| v * s).collect(),
        }
    }
    fn combine(a: &Self, ka: f64, b: &Self, kb: f64) -> Self {
        let f = |x: &[f64], y: &[f64]| -> Vec<f64> {
            x.iter()
                .zip(y)
                .map(|(p, q)| ka.mul_add(*p, kb * q))
                .collect()
        };
        ParaxialRayTable {
            y: f(&a.y, &b.y),
            u: f(&a.u, &b.u),
            nu: f(&a.nu, &b.nu),
        }
    }
}

/// Local principal coordinate used for an unfolded first-order sweep.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ParaxialAxis {
    X,
    Y,
}

impl ParaxialModel {
    pub fn new(sys: &ResolvedSystem, wl: usize) -> Self {
        Self::for_axis(sys, wl, ParaxialAxis::Y)
    }
    pub fn for_axis(sys: &ResolvedSystem, wl: usize, axis: ParaxialAxis) -> Self {
        let n = sys.surfaces.len();
        let mut nn = vec![0.0; n];
        let mut phi = vec![0.0; n];
        let mut t = vec![0.0; n];
        nn[0] = sys.n_object[wl];
        t[0] = sys.surfaces[0].thickness;
        for j in 1..n {
            (nn[j], phi[j]) = surface_step_axis(sys, j, wl, nn[j - 1], axis);
            t[j] = if j < n - 1 {
                sys.surfaces[j].thickness
            } else {
                0.0
            };
        }
        ParaxialModel {
            reduced_index: nn,
            power: phi,
            thickness: t,
            object_at_infinity: sys.object_at_infinity,
        }
    }

    /// Trace the paraxial ray with height `y1` at surface 1 and object-space
    /// slope `u0`.
    pub fn trace(&self, y1: f64, u0: f64) -> ParaxialRayTable {
        let n = self.power.len();
        let nn = &self.reduced_index;
        let mut y = vec![0.0; n];
        let mut w = vec![0.0; n];
        w[0] = nn[0] * u0;
        y[0] = if self.object_at_infinity {
            0.0
        } else {
            (-self.thickness[0]).mul_add(u0, y1)
        };
        for j in 1..n {
            y[j] = if j == 1 {
                y1
            } else {
                (self.thickness[j - 1] / nn[j - 1]).mul_add(w[j - 1], y[j - 1])
            };
            w[j] = (-y[j]).mul_add(self.power[j], w[j - 1]);
        }
        let u = w.iter().zip(nn).map(|(w, n)| w / n).collect();
        ParaxialRayTable { y, u, nu: w }
    }
}

/// Signed reduced index `N_j` after surface `j` and its power `φ_j`, given
/// `N_{j−1}` (module docs). Dummy interactions, coordinate breaks and the
/// image surface have no power.
#[inline]
fn surface_step(sys: &ResolvedSystem, j: usize, wl: usize, n_prev: f64) -> (f64, f64) {
    surface_step_axis(sys, j, wl, n_prev, ParaxialAxis::Y)
}

fn surface_step_axis(
    sys: &ResolvedSystem,
    j: usize,
    wl: usize,
    n_prev: f64,
    axis: ParaxialAxis,
) -> (f64, f64) {
    let n = sys.surfaces.len();
    let s = &sys.surfaces[j];
    let is_cb = matches!(s.shape, SurfaceShape::CoordinateBreak);
    let mirror = !is_cb && s.interaction == Interaction::Reflect && j < n - 1;
    let sign = if (n_prev < 0.0) != mirror { -1.0 } else { 1.0 };
    let nj = sign * s.n_after[wl];
    let phi = if j == n - 1 || is_cb || s.interaction == Interaction::Dummy {
        0.0
    } else {
        match &s.shape {
            SurfaceShape::Paraxial { focal_length } => nj.abs() / focal_length,
            _ => {
                let h = mklight_core::surface_math::vertex_curvatures(&s.shape);
                let mut power = h[if axis == ParaxialAxis::X { 0 } else { 2 }] * (nj - n_prev);
                if let SurfaceShape::Diffractive { phase, .. } = &s.shape {
                    // Reduced slope is n*k_transverse to first order, also
                    // on reflected paths. Positive A₁ therefore diverges.
                    power -= phase.trace_order as f64
                        * sys.wavelengths_um[wl]
                        * 1e-3
                        * phase.coefficients.first().copied().unwrap_or(0.0)
                        / (std::f64::consts::PI * phase.norm_radius * phase.norm_radius);
                }
                power
            }
        }
    };
    (nj, phi)
}

/// Paraxial **object vergence** `1/a` at ideal-lens surface `i` for
/// wavelength `wl`: `a` is the distance, measured against the propagation
/// direction, from the lens to the paraxial image of the axial object point
/// formed by surfaces `1..i` (`a > 0`: real object in front of the lens,
/// `a < 0`: virtual, `1/a = 0`: collimated).
///
/// It is traced with the axial y–nu ray (`y₀ = 0`, `u₀ = 1` for a finite
/// object, `y₁ = 1`, `u₀ = 0` at infinity): with `y_i` and the reduced slope
/// `ω_{i−1}` before the lens, `1/a = ω_{i−1}/(y_i |N_{i−1}|)`. The value is a
/// pure function of `(system, i, wl)`; GPU kernels receive it per
/// (surface, wavelength) instead of recomputing it per ray.
/// See [`crate::interact::ideal_lens`].
pub fn ideal_lens_object_vergence(sys: &ResolvedSystem, i: usize, wl: usize) -> f64 {
    let mut n_prev = sys.n_object[wl];
    let (mut y, mut w) = if sys.object_at_infinity {
        (1.0, 0.0)
    } else {
        (sys.surfaces[0].thickness, n_prev)
    };
    for j in 1..i {
        let (nj, phi) = surface_step(sys, j, wl, n_prev);
        w = (-y).mul_add(phi, w);
        y = (sys.surfaces[j].thickness / nj).mul_add(w, y);
        n_prev = nj;
    }
    w / (y * n_prev.abs())
}

/// First-order properties at one wavelength (module docs for definitions).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FirstOrder {
    pub wavelength_um: f64,
    pub efl: f64,
    pub bfl: f64,
    /// Signed position of the front focal point relative to surface 1.
    pub ffl: f64,
    /// Paraxial image distance from the last surface before the image
    /// (the thickness that puts the image surface at paraxial focus).
    pub image_distance: f64,
    /// Entrance-pupil position relative to surface 1 (object space).
    pub entrance_pupil_position: f64,
    pub entrance_pupil_diameter: f64,
    /// Exit-pupil position relative to the image surface.
    pub exit_pupil_position: f64,
    pub exit_pupil_diameter: f64,
    /// `f'/EPD` with the rear focal length `f' = |n'|·|EFL|`.
    pub image_f_number: f64,
    pub working_f_number: f64,
    pub object_na: f64,
    pub image_na: f64,
    pub magnification: f64,
    pub lagrange_invariant: f64,
    pub total_track: f64,
    /// Paraxial marginal-ray height at the stop.
    pub stop_semi_diameter: f64,
    /// Chief-ray height on the image surface at the maximum field.
    pub paraxial_image_height: f64,
    /// Image-surface height of the chief ray per unit object-space slope
    /// (`tan θ`), when the field can be given as an angle.
    pub image_height_per_tan: Option<f64>,
    /// Image-surface height of the chief ray per unit object height (finite
    /// objects).
    pub image_height_per_object_height: Option<f64>,
    /// Object-space slope of the marginal ray (finite objects; 0 at infinity).
    pub marginal_object_slope: f64,
    /// Marginal ray scaled by the aperture specification.
    pub marginal: ParaxialRayTable,
    /// Chief ray scaled to the maximum field.
    pub chief: ParaxialRayTable,
}

impl FirstOrder {
    pub fn entrance_pupil_radius(&self) -> f64 {
        0.5 * self.entrance_pupil_diameter
    }
    pub fn object_space_telecentric(&self) -> bool {
        !self.entrance_pupil_position.is_finite()
    }
    pub fn image_space_telecentric(&self) -> bool {
        !self.exit_pupil_position.is_finite()
    }
}

/// First-order data at every system wavelength.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Paraxial {
    pub wavelengths: Vec<FirstOrder>,
    pub primary: usize,
    pub warnings: Vec<String>,
}

impl Paraxial {
    pub fn new(sys: &ResolvedSystem) -> MkResult<Self> {
        let mut warnings = extended_paraxial_warnings(sys);
        if sys
            .system()
            .surfaces
            .iter()
            .any(|s| s.placement.is_some_and(|p| !p.is_identity()))
        {
            warnings.push(
                "system has tilts or decenters: paraxial values are for the unfolded, \
                 rotationally symmetric equivalent"
                    .to_string(),
            );
        }
        let wavelengths = (0..sys.wavelengths_um.len())
            .map(|w| first_order(sys, w))
            .collect::<MkResult<Vec<_>>>()?;
        Ok(Paraxial {
            wavelengths,
            primary: sys.primary_wavelength,
            warnings,
        })
    }
    pub fn primary(&self) -> &FirstOrder {
        &self.wavelengths[self.primary]
    }
    pub fn at(&self, wl: usize) -> &FirstOrder {
        &self.wavelengths[wl]
    }
}

fn extended_paraxial_warnings(sys: &ResolvedSystem) -> Vec<String> {
    let mut warnings = Vec::new();
    for (i, s) in sys
        .surfaces
        .iter()
        .enumerate()
        .filter(|(_, s)| !s.shape.is_legacy())
    {
        let v = mklight_core::surface_math::evaluate(&s.shape, 0.0, 0.0).expect("validated vertex");
        if v.sag != 0.0 || v.gradient != [0.0; 2] {
            warnings.push(format!("surface {i}: piston or vertex tilt; first order is a nominal-vertex quadratic approximation"));
        }
        if mklight_core::surface_math::vertex_curvatures(&s.shape)[1] != 0.0 {
            warnings.push(format!("surface {i}: mixed xy power; x/y are diagonal projections, not independent focal lengths"));
        }
        if let SurfaceShape::OddAsphere { coefficients, .. } = &s.shape
            && coefficients.first().is_some_and(|a| *a != 0.0)
        {
            warnings.push(format!(
                "surface {i}: cone vertex; formal quadratic first order omits the r term"
            ));
        }
    }
    warnings
}

/// Both unfolded first-order traces. `squeeze = EFL_y/EFL_x` (signed).
/// x/y are independent only for zero mixed vertex curvature. General
/// freeforms with xy power require a later coupled 4×4 first-order model;
/// their two diagonal projections are still reported, with a warning.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParaxialXY {
    pub x: Vec<FirstOrder>,
    pub y: Vec<FirstOrder>,
    pub squeeze: Vec<f64>,
    pub primary: usize,
    pub symmetry: mklight_core::surface_math::Symmetry,
    pub run_info: mklight_core::RunInfo,
}
impl ParaxialXY {
    pub fn new(sys: &ResolvedSystem) -> MkResult<Self> {
        Self::new_job(sys, &mklight_core::JobContext::new())
    }
    pub fn new_job(sys: &ResolvedSystem, ctx: &mklight_core::JobContext) -> MkResult<Self> {
        use mklight_core::job::{Precision, RunTimer};
        let timer = RunTimer::start();
        let mut warnings = extended_paraxial_warnings(sys);
        if sys
            .system()
            .surfaces
            .iter()
            .any(|s| s.placement.is_some_and(|p| !p.is_identity()))
        {
            warnings.push("placements are ignored by unfolded x/y first order".into());
        }
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut squeeze = Vec::new();
        for wl in 0..sys.wavelengths_um.len() {
            ctx.check()?;
            let fx = first_order_axis(sys, wl, ParaxialAxis::X)?;
            let fy = first_order_axis(sys, wl, ParaxialAxis::Y)?;
            squeeze.push(fy.efl / fx.efl);
            x.push(fx);
            y.push(fy);
            ctx.report(
                (wl + 1) as f64 / sys.wavelengths_um.len() as f64,
                "paraxial x/y",
            );
        }
        Ok(Self {
            x,
            y,
            squeeze,
            primary: sys.primary_wavelength,
            symmetry: mklight_core::surface_math::system_symmetry(sys.system()),
            run_info: timer.finish(
                &sys.snapshot,
                "paraxial_xy",
                &serde_json::json!({}),
                "cpu-f64",
                Precision::F64Reference,
                None,
                warnings,
            ),
        })
    }
}

fn inf_if_large(z: f64) -> f64 {
    if z.is_finite() && z.abs() <= TELECENTRIC_LIMIT {
        z
    } else {
        f64::INFINITY
    }
}

/// Compute the first-order properties at wavelength index `wl`.
pub fn first_order(sys: &ResolvedSystem, wl: usize) -> MkResult<FirstOrder> {
    first_order_axis(sys, wl, ParaxialAxis::Y)
}

/// Existing first-order definitions applied along one local coordinate.
/// Cross-axis quadratic terms are not included; see [`ParaxialXY`].
pub fn first_order_axis(
    sys: &ResolvedSystem,
    wl: usize,
    axis: ParaxialAxis,
) -> MkResult<FirstOrder> {
    let model = ParaxialModel::for_axis(sys, wl, axis);
    let nn = &model.reduced_index;
    let n = nn.len();
    let img = n - 1;
    let last = n - 2;
    let stop = sys.stop;
    let n0 = nn[0];
    let at_inf = sys.object_at_infinity;
    let t0 = sys.surfaces[0].thickness;
    let osys = sys.system();
    let bad = |m: &str| MkError::InvalidModel(format!("paraxial: {m}"));

    let ra = model.trace(1.0, 0.0);
    let rb = model.trace(0.0, 1.0);
    let wa = ra.nu[last];
    let efl = if wa != 0.0 { -1.0 / wa } else { f64::INFINITY };
    let ua = ra.u[last];
    let bfl = if ua != 0.0 {
        -ra.y[last] / ua
    } else {
        f64::INFINITY
    };
    // Rear focal length f' = |N'|·|EFL| (image space after the last surface).
    let f_rear = nn[last].abs() * efl.abs();
    let ffl = if wa != 0.0 {
        rb.nu[last] / wa
    } else {
        f64::INFINITY
    };

    // Entrance pupil.
    let z_ep = inf_if_large(rb.y[stop] / ra.y[stop]);
    let obj_tele = !z_ep.is_finite();
    if obj_tele && at_inf {
        return Err(bad(
            "entrance pupil at infinity with the object at infinity",
        ));
    }

    // Unit chief ray through the stop centre, slope parameterised (u₀ = 1),
    // or the axis-parallel basis ray when object space is telecentric.
    let chief_u = if obj_tele {
        ra.clone()
    } else {
        ParaxialRayTable::combine(&ra, -z_ep, &rb, 1.0)
    };
    // Unit chief from an object point of height 1 (finite objects).
    let chief_h = (!at_inf).then(|| {
        let u0 = if obj_tele { 0.0 } else { -1.0 / (z_ep + t0) };
        model.trace(u0.mul_add(t0, 1.0), u0)
    });

    // Unit marginal ray and its height in the entrance-pupil plane.
    let (m1, e1) = if at_inf {
        (ra.clone(), 1.0)
    } else {
        (
            ParaxialRayTable::combine(&ra, t0, &rb, 1.0),
            if obj_tele { f64::INFINITY } else { z_ep + t0 },
        )
    };
    let sigma = match osys.aperture {
        ApertureSpec::EntrancePupilDiameter { value } => {
            if !e1.is_finite() {
                return Err(bad(
                    "entrance-pupil diameter undefined for a telecentric object space",
                ));
            }
            0.5 * value / e1.abs()
        }
        ApertureSpec::ImageSpaceFNumber { value } => {
            if !efl.is_finite() || !e1.is_finite() {
                return Err(bad("image-space F/# needs a finite EFL and entrance pupil"));
            }
            f_rear / (2.0 * value) / e1.abs()
        }
        ApertureSpec::ObjectSpaceNa { value } => {
            if at_inf {
                return Err(bad("object-space NA needs a finite object"));
            }
            let s = value / n0;
            if !(s > 0.0 && s < 1.0) {
                return Err(bad("object-space NA must be in (0, n₀)"));
            }
            s.asin().tan()
        }
        ApertureSpec::ParaxialWorkingFNumber { value } => {
            let w = m1.nu[last].abs();
            if w == 0.0 {
                return Err(bad("working F/# undefined for an afocal system"));
            }
            1.0 / (2.0 * value * w)
        }
        ApertureSpec::FloatByStopSize => {
            let s = &sys.surfaces[stop];
            let r = s
                .aperture
                .as_ref()
                .map(|a| a.bounding_radius())
                .or(s.fixed_semi_diameter)
                .ok_or_else(|| {
                    bad("aperture floats by stop size, but the stop has no fixed semi-diameter")
                })?;
            r / m1.y[stop].abs()
        }
    };
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(bad(
            "aperture specification gives no finite, non-zero marginal ray",
        ));
    }
    let marginal = m1.scaled(sigma);
    let r_ep = sigma * e1.abs();
    let stop_sd = marginal.y[stop].abs();

    // Chief ray scaled to the maximum field.
    let fmax = osys.fields.max_radial();
    let has_field = osys.fields.fields.iter().any(|f| f.x != 0.0 || f.y != 0.0);
    let per_tan = (!obj_tele).then(|| chief_u.y[img]);
    let per_h = chief_h.as_ref().map(|c| c.y[img]);
    let chief = if !has_field {
        chief_u.scaled(0.0)
    } else {
        match osys.fields.kind {
            FieldKind::ObjectAngle => {
                if obj_tele {
                    return Err(bad("angle fields need a finite entrance pupil"));
                }
                chief_u.scaled(fmax.to_radians().tan())
            }
            FieldKind::ObjectHeight => chief_h
                .as_ref()
                .ok_or_else(|| bad("object-height fields need a finite object"))?
                .scaled(fmax),
            FieldKind::ParaxialImageHeight | FieldKind::RealImageHeight => {
                let unit = chief_h.as_ref().unwrap_or(&chief_u);
                let h = unit.y[img];
                if h == 0.0 {
                    return Err(bad("chief ray does not reach a non-zero image height"));
                }
                unit.scaled(fmax / h)
            }
        }
    };

    // Exit pupil from the unit chief ray.
    let uc = chief_u.u[last];
    let z_xp = if uc != 0.0 {
        inf_if_large(-chief_u.y[img] / uc)
    } else {
        f64::INFINITY
    };
    let um = marginal.u[last];
    let r_xp = if z_xp.is_finite() {
        um.mul_add(z_xp, marginal.y[img]).abs()
    } else {
        f64::INFINITY
    };

    let wm_img = marginal.nu[last];
    let image_distance = if at_inf {
        bfl
    } else if um != 0.0 {
        -marginal.y[last] / um
    } else {
        f64::INFINITY
    };
    let total_track = (1..last + 1).map(|j| model.thickness[j]).sum::<f64>();
    let lagrange = chief.nu[0].mul_add(marginal.y[1], -(marginal.nu[0] * chief.y[1]));

    Ok(FirstOrder {
        wavelength_um: sys.wavelengths_um[wl],
        efl,
        bfl,
        ffl,
        image_distance,
        entrance_pupil_position: z_ep,
        entrance_pupil_diameter: 2.0 * r_ep,
        exit_pupil_position: z_xp,
        exit_pupil_diameter: 2.0 * r_xp,
        image_f_number: if efl.is_finite() && r_ep.is_finite() {
            f_rear / (2.0 * r_ep)
        } else {
            f64::INFINITY
        },
        working_f_number: if wm_img != 0.0 {
            1.0 / (2.0 * wm_img.abs())
        } else {
            f64::INFINITY
        },
        object_na: if at_inf {
            0.0
        } else {
            n0 * marginal.u[0].abs().atan().sin()
        },
        image_na: nn[last].abs() * um.abs().atan().sin(),
        magnification: if at_inf || wm_img == 0.0 {
            0.0
        } else {
            marginal.nu[0] / wm_img
        },
        lagrange_invariant: lagrange,
        total_track,
        stop_semi_diameter: stop_sd,
        paraxial_image_height: chief.y[img],
        image_height_per_tan: per_tan,
        image_height_per_object_height: per_h,
        marginal_object_slope: if at_inf { 0.0 } else { marginal.u[0] },
        marginal,
        chief,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::{
        Field, FieldSet, MaterialRef, OpticalSystem, Snapshot, Surface, Wavelength, WavelengthSet,
    };
    use std::sync::Arc;

    fn resolve(s: OpticalSystem) -> ResolvedSystem {
        let snap = Snapshot::new(s).unwrap();
        ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap()
    }

    fn mono() -> WavelengthSet {
        WavelengthSet {
            wavelengths: vec![Wavelength {
                um: 0.5875618,
                weight: 1.0,
            }],
            primary: 0,
        }
    }

    #[test]
    fn every_aperture_spec_sizes_the_same_pupil() {
        // Ideal f = 100 lens, stop 25 mm in front: each specification below
        // describes an entrance pupil of diameter 25 mm.
        let build = |aperture: ApertureSpec| {
            let mut s = OpticalSystem::new_empty();
            s.wavelengths = mono();
            s.aperture = aperture;
            let mut stop = Surface::new(1, SurfaceShape::Plane, 25.0, MaterialRef::Air);
            stop.semi_diameter = mklight_core::SemiDiameter::Fixed { value: 12.5 };
            s.surfaces = vec![
                Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
                stop,
                Surface::new(
                    2,
                    SurfaceShape::Paraxial {
                        focal_length: 100.0,
                    },
                    100.0,
                    MaterialRef::Air,
                ),
                Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
            ];
            s.stop = 1;
            first_order(&resolve(s), 0).unwrap()
        };
        for spec in [
            ApertureSpec::EntrancePupilDiameter { value: 25.0 },
            ApertureSpec::ImageSpaceFNumber { value: 4.0 },
            ApertureSpec::ParaxialWorkingFNumber { value: 4.0 },
            ApertureSpec::FloatByStopSize,
        ] {
            let fo = build(spec);
            assert!(
                (fo.entrance_pupil_diameter - 25.0).abs() < 1e-12,
                "{spec:?}: {}",
                fo.entrance_pupil_diameter
            );
            assert!((fo.stop_semi_diameter - 12.5).abs() < 1e-12);
            assert!((fo.working_f_number - 4.0).abs() < 1e-12);
        }
    }

    #[test]
    fn image_space_f_number_uses_rear_focal_length() {
        // Ideal lens of image-space focal length f' = 100 mm with glass after
        // it: EFL = f'/n', and F/4 means f'/EPD = 4, i.e. EPD = 25 mm.
        let mut s = OpticalSystem::new_empty();
        s.wavelengths = mono();
        s.aperture = ApertureSpec::ImageSpaceFNumber { value: 4.0 };
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(1, SurfaceShape::Plane, 10.0, MaterialRef::Air),
            Surface::new(
                2,
                SurfaceShape::Paraxial {
                    focal_length: 100.0,
                },
                100.0,
                MaterialRef::glass("N-BK7"),
            ),
            Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::glass("N-BK7")),
        ];
        s.stop = 1;
        let sys = resolve(s);
        let n_img = sys.surfaces[2].n_after[0];
        assert!(n_img > 1.5);
        let fo = first_order(&sys, 0).unwrap();
        assert!((fo.efl - 100.0 / n_img).abs() < 1e-12);
        assert!((fo.bfl - 100.0).abs() < 1e-12);
        assert!((fo.entrance_pupil_diameter - 25.0).abs() < 1e-12);
        assert!((fo.image_f_number - 4.0).abs() < 1e-12);
        // Working F/# = 1/(2 n' u') = 4/n' at infinity.
        assert!((fo.working_f_number - 4.0 / n_img).abs() < 1e-12);
    }

    #[test]
    fn dummy_ideal_lens_has_no_power() {
        let mut s = OpticalSystem::new_empty();
        s.wavelengths = mono();
        s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
        let mut lens = Surface::new(
            2,
            SurfaceShape::Paraxial { focal_length: 50.0 },
            100.0,
            MaterialRef::Air,
        );
        lens.interaction = Interaction::Dummy;
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, 200.0, MaterialRef::Air),
            Surface::new(1, SurfaceShape::Plane, 10.0, MaterialRef::Air),
            lens,
            Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        let model = ParaxialModel::new(&resolve(s), 0);
        assert!(model.power.iter().all(|p| *p == 0.0));
    }

    #[test]
    fn ideal_lens_object_vergence_known_answers() {
        // Object 150 mm before an f = 50 lens, then an f = 100 lens 100 mm
        // behind: lens 1 sees a = 150; its image is at b = 75, i.e. 25 mm in
        // front of lens 2 (a = 25). At infinity lens 1 sees ρ = 0.
        let build = |t0: f64| {
            let mut s = OpticalSystem::new_empty();
            s.wavelengths = mono();
            s.aperture = if t0.is_finite() {
                ApertureSpec::ObjectSpaceNa { value: 0.05 }
            } else {
                ApertureSpec::EntrancePupilDiameter { value: 5.0 }
            };
            s.surfaces = vec![
                Surface::new(0, SurfaceShape::Plane, t0, MaterialRef::Air),
                Surface::new(
                    1,
                    SurfaceShape::Paraxial { focal_length: 50.0 },
                    100.0,
                    MaterialRef::Air,
                ),
                Surface::new(
                    2,
                    SurfaceShape::Paraxial {
                        focal_length: 100.0,
                    },
                    100.0,
                    MaterialRef::Air,
                ),
                Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
            ];
            s.stop = 1;
            resolve(s)
        };
        let sys = build(150.0);
        assert!((ideal_lens_object_vergence(&sys, 1, 0) - 1.0 / 150.0).abs() < 1e-15);
        assert!((ideal_lens_object_vergence(&sys, 2, 0) - 1.0 / 25.0).abs() < 1e-15);
        let sys = build(f64::INFINITY);
        assert_eq!(ideal_lens_object_vergence(&sys, 1, 0), 0.0);
        // Collimated in, focus 50 behind lens 1 = 50 in front of lens 2.
        assert!((ideal_lens_object_vergence(&sys, 2, 0) - 1.0 / 50.0).abs() < 1e-15);
    }

    #[test]
    fn ideal_lens_first_order() {
        let mut s = OpticalSystem::new_empty();
        s.wavelengths = mono();
        s.aperture = ApertureSpec::EntrancePupilDiameter { value: 20.0 };
        s.fields = FieldSet {
            kind: FieldKind::ObjectAngle,
            fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 5.0)],
        };
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(1, SurfaceShape::Plane, 25.0, MaterialRef::Air),
            Surface::new(
                2,
                SurfaceShape::Paraxial {
                    focal_length: 100.0,
                },
                100.0,
                MaterialRef::Air,
            ),
            Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        let fo = first_order(&resolve(s), 0).unwrap();
        assert!((fo.efl - 100.0).abs() < 1e-12);
        assert!((fo.bfl - 100.0).abs() < 1e-12);
        assert!((fo.ffl + 75.0).abs() < 1e-12); // front focal point 100 mm before the lens
        assert!(fo.entrance_pupil_position.abs() < 1e-12);
        assert!((fo.entrance_pupil_diameter - 20.0).abs() < 1e-12);
        assert!((fo.image_f_number - 5.0).abs() < 1e-12);
        // Stop 25 mm before the lens: exit pupil = image of the stop.
        // 1/b = 1/f − 1/a with a = 25 → b = −100/3 (virtual), i.e. 100/3 mm
        // before the lens and 100 + 100/3 before the image.
        assert!((fo.exit_pupil_position + (100.0 + 100.0 / 3.0)).abs() < 1e-9);
        // Chief at +5° (object at infinity): image height +f tan θ.
        let h = 100.0 * 5f64.to_radians().tan();
        assert!((fo.paraxial_image_height - h).abs() < 1e-12);
        // Lagrange invariant H = n u_c y_m − n u_m y_c = tan θ · r_EP (u_m = 0 in object space).
        assert!((fo.lagrange_invariant - 5f64.to_radians().tan() * 10.0).abs() < 1e-12);
    }

    #[test]
    fn concave_mirror_signs() {
        let mut s = OpticalSystem::new_empty();
        s.wavelengths = mono();
        s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
        let mut m = Surface::new(1, SurfaceShape::sphere(-200.0), -100.0, MaterialRef::Air);
        m.interaction = Interaction::Reflect;
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            m,
            Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        let fo = first_order(&resolve(s), 0).unwrap();
        assert!((fo.efl - 100.0).abs() < 1e-12);
        assert!((fo.bfl + 100.0).abs() < 1e-12); // focus 100 mm in front of the mirror
        assert!((fo.exit_pupil_position - 100.0).abs() < 1e-12); // mirror is 100 mm "behind" in local z
        assert!((fo.working_f_number - 10.0).abs() < 1e-12);
        assert!((fo.total_track + 100.0).abs() < 1e-12);
    }

    #[test]
    fn finite_conjugate_magnification_and_na() {
        let mut s = OpticalSystem::new_empty();
        s.wavelengths = mono();
        s.aperture = ApertureSpec::ObjectSpaceNa { value: 0.1 };
        s.fields = FieldSet {
            kind: FieldKind::ObjectHeight,
            fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 2.0)],
        };
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, 150.0, MaterialRef::Air),
            Surface::new(
                1,
                SurfaceShape::Paraxial {
                    focal_length: 100.0,
                },
                300.0,
                MaterialRef::Air,
            ),
            Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        let fo = first_order(&resolve(s), 0).unwrap();
        assert!((fo.image_distance - 300.0).abs() < 1e-9);
        assert!((fo.magnification + 2.0).abs() < 1e-12);
        assert!((fo.object_na - 0.1).abs() < 1e-14);
        assert!((fo.paraxial_image_height + 4.0).abs() < 1e-9);
        // Paraxial NA scales with 1/|m| in the slope domain.
        let u_img = 0.1f64.asin().tan() / 2.0;
        assert!((fo.image_na - u_img.atan().sin()).abs() < 1e-12);
    }
}
