//! Paraxial ghost analysis: a fast first-order ranking of every ghost path.
//!
//! # y–nu trace along a step sequence
//!
//! [`trace_steps`] runs the y–nu trace of `mklight_trace::paraxial` along an
//! arbitrary [`Step`] list, on the *unfolded* sequence. Local `z` is each
//! surface's own axis (tilts and decenters ignored, as in the reference
//! paraxial model). The signed reduced index `N = n·σ`, `σ = sign(dz)` of the
//! propagation, starts at `+n_object` and flips at **every** reflection
//! (designed mirror or ghost reflection). With `ω = N u` after each surface:
//!
//! * pass at a refracting surface: `N' = σ·n_far`, `φ = c(N' − N)` (the same
//!   surface power `c(n_after − n_before)` in both directions); ideal lens
//!   `φ = n_after/f` in both directions;
//! * reflection (ghost or mirror): `N' = −N`, `φ = c(N' − N) = −2cN`
//!   (ideal-lens mirror: `|N|/f`), the sensor included (with its own
//!   curvature, as the real tracer reflects about the curved normal);
//! * dummies, coordinate breaks and detection: `φ = 0`, `N' = N`;
//! * refraction `ω' = ω − yφ`, transfer `y' = y + Δz·ω'/N'` with
//!   `Δz = t_i` from `i` to `i + 1` and `Δz = −t_{i−1}` from `i` to `i − 1`.
//!
//! The empty path reproduces the reference first-order rays exactly
//! (tested), so ghost and primary are on the same footing.
//!
//! # Per ghost (primary wavelength unless set)
//!
//! The marginal and chief rays start at surface 1 with the reference
//! first-order values (`y₁`, `ω₀`), so the ghost beam is the designed beam.
//! At the image surface (local frame):
//!
//! * **ghost focus** `z_g = −y_m/u_m`: signed distance of the paraxial ghost
//!   image from the sensor along the image's local `z` (`±∞`: collimated);
//! * **spot diameter** `D_g = 2|y_m|`: the defocused ghost's blur disc;
//! * **ghost height** `y_c`: centre of the ghost blur for the maximum field;
//! * **irradiance ratio**
//!   `E_g/E_s ≈ (Π R)(Π T_g/Π T_p)·(D_s/max(D_g, D_s))²·V`, where `Π R` is
//!   the reflectance product, `Π T` the pass transmittances and bulk
//!   transmittance along the axis (normal incidence, unpolarised), `D_s` the
//!   signal (primary) spot diameter and `V` the paraxial vignetting estimate:
//!   the smallest `(r_clip/|y_m|)²` over the surfaces of the path whose
//!   clipping radius `r_clip` the ghost's marginal ray overfills (on-axis
//!   beams are concentric discs). The signal diameter is the largest of the
//!   primary paraxial blur `2|y_m|`, the Airy diameter `2.44·λ·N_w` and the
//!   sensor pixel pitch, unless set explicitly;
//! * **near focus**: `D_g ≤ near_focus_factor · D_s`, the ghosts that form a
//!   concentrated image near the sensor.
//!
//! Ghosts are ranked by the irradiance ratio (descending).

use mklight_core::job::{Precision, RunTimer};
use mklight_core::{Interaction, MkError, MkResult, RunInfo, SurfaceShape};
use serde::{Deserialize, Serialize};

use crate::model::{GhostModel, GhostSettings};
use crate::path::{Action, EnumerationSettings, GhostPath, Step, enumerate};
use crate::signature::GhostCoverage;

/// One y–nu ray along a step list: values at each step after its
/// interaction (height at the surface, reduced slope and signed reduced
/// index after it).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct ParaxialTrace {
    pub y: Vec<f64>,
    pub nu: Vec<f64>,
    pub n: Vec<f64>,
}

impl ParaxialTrace {
    /// Slope `u = ω/N` after step `k`.
    pub fn u(&self, k: usize) -> f64 {
        self.nu[k] / self.n[k]
    }
}

/// Signed reduced index after a step and the step's power (module docs).
fn step_power(model: &GhostModel, st: &Step, n_in: f64, wl: usize) -> (f64, f64) {
    let sys = model.system();
    let i = st.surface as usize;
    let s = &sys.surfaces[i];
    let c = s.shape.curvature();
    if i == sys.image_index() {
        return match st.action {
            Action::Pass => (n_in, 0.0),
            // The sensor reflects about its own (possibly curved) normal.
            Action::Reflect => (-n_in, -2.0 * c * n_in),
        };
    }
    if matches!(s.shape, SurfaceShape::CoordinateBreak) || s.interaction == Interaction::Dummy {
        return (n_in, 0.0);
    }
    let ideal = match s.shape {
        SurfaceShape::Paraxial { focal_length } => Some(focal_length),
        _ => None,
    };
    if s.interaction == Interaction::Reflect || st.action == Action::Reflect {
        let n_out = -n_in;
        let phi = match ideal {
            Some(f) => n_in.abs() / f,
            None => c * (n_out - n_in),
        };
        return (n_out, phi);
    }
    let (_, n_far) = model.media(i, st.direction, wl);
    let n_out = n_far.copysign(n_in);
    let phi = match ideal {
        Some(f) => s.n_after[wl] / f,
        None => c * (n_out - n_in),
    };
    (n_out, phi)
}

/// Axial distance from step surface `from` to the next step's surface.
fn delta_z(model: &GhostModel, from: usize, to: usize) -> f64 {
    let s = &model.system().surfaces;
    if to == from + 1 {
        s[from].thickness
    } else {
        -s[to].thickness
    }
}

/// y–nu trace along `steps` (starting at surface 1) of the ray with height
/// `y1` at surface 1 and object-space reduced slope `nu0` (module docs).
pub fn trace_steps(
    model: &GhostModel,
    steps: &[Step],
    wl: usize,
    y1: f64,
    nu0: f64,
) -> ParaxialTrace {
    let sys = model.system();
    let mut out = ParaxialTrace {
        y: Vec::with_capacity(steps.len()),
        nu: Vec::with_capacity(steps.len()),
        n: Vec::with_capacity(steps.len()),
    };
    let (mut y, mut w, mut nn) = (y1, nu0, sys.n_object[wl]);
    for (k, st) in steps.iter().enumerate() {
        let (n_out, phi) = step_power(model, st, nn, wl);
        w = (-y).mul_add(phi, w);
        nn = n_out;
        out.y.push(y);
        out.nu.push(w);
        out.n.push(nn);
        if let Some(next) = steps.get(k + 1) {
            let dz = delta_z(model, st.surface as usize, next.surface as usize);
            y = (dz / nn).mul_add(w, y);
        }
    }
    out
}

/// Normal-incidence, unpolarised `(Π R, Π T·τ)` along a step list: the
/// reflectance product of the reflections and the transmittance of the
/// passes times the axial bulk transmittance.
pub fn axial_throughput(model: &GhostModel, steps: &[Step], wl: usize) -> (f64, f64) {
    let sys = model.system();
    let lambda = sys.wavelengths_um[wl];
    let (mut r, mut t, mut absorb) = (1.0, 1.0, 0.0f64);
    for (k, st) in steps.iter().enumerate() {
        let i = st.surface as usize;
        let s = &sys.surfaces[i];
        if k > 0 {
            let prev = steps[k - 1].surface as usize;
            let gap = prev.min(i);
            absorb += sys.surfaces[gap].alpha_after[wl] * s_thick(model, gap).abs();
        }
        let passive =
            matches!(s.shape, SurfaceShape::CoordinateBreak) || s.interaction == Interaction::Dummy;
        if i == sys.image_index() {
            if let Some(rs) = model.settings().sensor_reflectance {
                match st.action {
                    Action::Reflect => r *= rs,
                    // Detection: the absorbed fraction (as the real tracer).
                    Action::Pass => t *= 1.0 - rs,
                }
            }
            continue;
        }
        if passive {
            continue;
        }
        let (n_in, n_far) = model.media(i, st.direction, wl);
        if s.interaction == Interaction::Reflect {
            t *= s.coating.rt(lambda, n_in, n_in, 1.0, true).r_unpolarized();
        } else {
            let rt = s.coating.rt(lambda, n_in, n_far, 1.0, false);
            match st.action {
                Action::Pass => t *= rt.t_unpolarized(),
                Action::Reflect => r *= rt.r_unpolarized(),
            }
        }
    }
    (r, t * (-absorb).exp())
}

fn s_thick(model: &GhostModel, i: usize) -> f64 {
    model.system().surfaces[i].thickness
}

/// Settings of [`paraxial_ghosts`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParaxialGhostSettings {
    pub enumeration: EnumerationSettings,
    /// Explicit paths instead of the enumeration.
    #[serde(default)]
    pub paths: Option<Vec<GhostPath>>,
    /// System wavelength index; `None` = primary.
    #[serde(default)]
    pub wavelength: Option<usize>,
    /// Signal spot diameter (mm); `None` = automatic (module docs).
    #[serde(default)]
    pub signal_diameter_mm: Option<f64>,
    /// Ghosts whose spot is at most this many signal diameters are flagged.
    pub near_focus_factor: f64,
}

impl Default for ParaxialGhostSettings {
    fn default() -> Self {
        ParaxialGhostSettings {
            enumeration: EnumerationSettings::default(),
            paths: None,
            wavelength: None,
            signal_diameter_mm: None,
            near_focus_factor: 10.0,
        }
    }
}

/// First-order data of one ghost (module docs).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParaxialGhost {
    pub path: GhostPath,
    pub label: String,
    /// Π R, normal incidence, unpolarised.
    pub reflectance_product: f64,
    /// Π T of the passes times axial bulk transmittance.
    pub transmittance: f64,
    /// Signed distance of the ghost focus from the sensor (image local z).
    pub focus_from_sensor: f64,
    /// Ghost blur diameter on the sensor (full aperture, on axis).
    pub spot_diameter: f64,
    /// Ghost centre height on the sensor at the maximum field.
    pub image_height: f64,
    /// Paraxial vignetting estimate `V` in (0, 1].
    pub vignetting: f64,
    /// Estimated ghost irradiance relative to the signal.
    pub relative_irradiance: f64,
    pub near_focus: bool,
}

/// Result of [`paraxial_ghosts`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ParaxialGhostReport {
    pub run: RunInfo,
    pub wavelength: usize,
    pub signal_diameter: f64,
    /// Primary transmittance (normal incidence, axial absorption).
    pub primary_transmittance: f64,
    /// Primary paraxial focus from the sensor (≈ 0 for a focused design).
    pub primary_focus_from_sensor: f64,
    pub primary_image_height: f64,
    /// Ranked by `relative_irradiance`, strongest first.
    pub ghosts: Vec<ParaxialGhost>,
    pub truncated: bool,
    pub coverage: GhostCoverage,
}

#[derive(Serialize)]
struct RunSettings<'a, T: Serialize> {
    model: &'a GhostSettings,
    analysis: &'a T,
}

pub(crate) fn run_info<T: Serialize>(
    model: &GhostModel,
    timer: RunTimer,
    analysis: &str,
    settings: &T,
    warnings: Vec<String>,
) -> RunInfo {
    timer.finish(
        &model.system().snapshot,
        analysis,
        &RunSettings {
            model: model.settings(),
            analysis: settings,
        },
        "cpu-f64",
        Precision::F64Reference,
        None,
        warnings,
    )
}

fn focus(y: f64, u: f64) -> f64 {
    if u != 0.0 && (y / u).is_finite() {
        -y / u
    } else if y == 0.0 {
        0.0
    } else {
        f64::INFINITY
    }
}

/// Rank every ghost path by its paraxial irradiance estimate (module docs).
pub fn paraxial_ghosts(
    model: &GhostModel,
    settings: &ParaxialGhostSettings,
) -> MkResult<ParaxialGhostReport> {
    let timer = RunTimer::start();
    let sys = model.system();
    let wl = settings.wavelength.unwrap_or(sys.primary_wavelength);
    if wl >= sys.wavelengths_um.len() {
        return Err(MkError::InvalidModel(format!(
            "wavelength {wl} out of range"
        )));
    }
    let image = sys.image_index();
    let fo = model.tracer().paraxial().at(wl);
    let (ym1, wm0) = (fo.marginal.y[1], fo.marginal.nu[0]);
    let (yc1, wc0) = (fo.chief.y[1], fo.chief.nu[0]);

    let (paths, truncated, coverage) = match &settings.paths {
        Some(p) => {
            let coverage = model.coverage(p.iter().cloned());
            (p.clone(), false, coverage)
        }
        None => {
            let e = enumerate(model, &settings.enumeration)?;
            (
                e.paths.into_iter().map(|p| p.path).collect(),
                e.truncated,
                e.coverage,
            )
        }
    };

    let pstep = model.check_path(&GhostPath::primary())?;
    let pm = trace_steps(model, &pstep, wl, ym1, wm0);
    let pc = trace_steps(model, &pstep, wl, yc1, wc0);
    let last = pstep.len() - 1;
    let (_, primary_t) = axial_throughput(model, &pstep, wl);
    let lambda_mm = sys.wavelengths_um[wl] * 1e-3;
    let pixel = sys
        .system()
        .sensor
        .as_ref()
        .map_or(0.0, |s| s.pitch_x.max(s.pitch_y));
    let airy = if fo.working_f_number.is_finite() {
        2.44 * lambda_mm * fo.working_f_number
    } else {
        0.0
    };
    let d_sig = settings
        .signal_diameter_mm
        .unwrap_or_else(|| (2.0 * pm.y[last].abs()).max(airy).max(pixel));
    if !(d_sig > 0.0 && d_sig.is_finite()) {
        return Err(MkError::InvalidModel(
            "signal diameter must be positive and finite".into(),
        ));
    }

    let mut ghosts = Vec::with_capacity(paths.len());
    for path in paths {
        if !path.ends_at_image() || path.is_primary() {
            continue;
        }
        let steps = model.check_path(&path)?;
        let m = trace_steps(model, &steps, wl, ym1, wm0);
        let c = trace_steps(model, &steps, wl, yc1, wc0);
        let k = steps.len() - 1;
        let (r, t) = axial_throughput(model, &steps, wl);
        let ym = m.y[k];
        let d_g = 2.0 * ym.abs();
        let mut vig = 1.0f64;
        for (j, st) in steps.iter().enumerate() {
            if let Some(rc) = model.clip_radius(st.surface as usize) {
                let h = m.y[j].abs();
                if h > rc {
                    vig = vig.min((rc / h) * (rc / h));
                }
            }
        }
        let conc = (d_sig / d_g.max(d_sig)).powi(2);
        let rel = if primary_t > 0.0 {
            r * (t / primary_t) * conc * vig
        } else {
            0.0
        };
        ghosts.push(ParaxialGhost {
            label: path.label(image),
            path,
            reflectance_product: r,
            transmittance: t,
            focus_from_sensor: focus(ym, m.u(k)),
            spot_diameter: d_g,
            image_height: c.y[k],
            vignetting: vig,
            relative_irradiance: rel,
            near_focus: d_g <= settings.near_focus_factor * d_sig,
        });
    }
    ghosts.sort_by(|a, b| {
        b.relative_irradiance
            .total_cmp(&a.relative_irradiance)
            .then_with(|| a.path.cmp(&b.path))
    });
    let warnings = model.tracer().warnings().to_vec();
    Ok(ParaxialGhostReport {
        run: run_info(model, timer, "ghost.paraxial", settings, warnings),
        wavelength: wl,
        signal_diameter: d_sig,
        primary_transmittance: primary_t,
        primary_focus_from_sensor: focus(pm.y[last], pm.u(last)),
        primary_image_height: pc.y[last],
        ghosts,
        truncated,
        coverage,
    })
}

/// Real-ray ghost focus for comparison with the paraxial one: the on-axis
/// meridional ray at normalised pupil height `rho` is traced along `path`
/// (with the model's clipping) and the signed distance from the sensor to
/// where it crosses the axis, along the image's local `z`, is returned.
/// As `rho → 0` it tends to [`ParaxialGhost::focus_from_sensor`].
pub fn real_ghost_focus(
    model: &GhostModel,
    path: &GhostPath,
    wl: usize,
    rho: f64,
) -> MkResult<f64> {
    let tracer = model.tracer();
    let field = tracer.prepare_field(mklight_trace::FieldPoint::normalized(0.0, 0.0))?;
    let g = tracer
        .generate(&field, wl, 0.0, rho)
        .map_err(|_| MkError::Trace("pupil ray could not be generated".into()))?;
    let steps = model.check_path(path)?;
    let r = model.trace_steps(&g.ray, wl, &steps, None);
    if !r.status.is_ok() {
        return Err(MkError::Trace(format!("ghost ray lost: {:?}", r.status)));
    }
    let (y, d) = (r.ray.pos.y, r.ray.dir);
    Ok(if d.y != 0.0 {
        -y * d.z / d.y
    } else {
        f64::INFINITY
    })
}
