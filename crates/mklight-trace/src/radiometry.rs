//! Radiometry along the sequential primary path (CONVENTIONS §5, §9).
//!
//! # Per-ray throughput
//!
//! The power fraction a ray carries from the entrance pupil to the image is
//! `T = T_interfaces · τ_internal`, where `τ_internal = exp(−Σ α_k L_k)`
//! (Beer–Lambert over the real path lengths in every medium, from
//! [`crate::trace`]) and `T_interfaces` follows the [`RadiometryModel`]:
//!
//! * [`RadiometryModel::Unpolarized`]: the product over interacting surfaces
//!   of the incoherent s/p average `½(T_s + T_p)` (`½(R_s + R_p)` at mirrors,
//!   1 at ideal lenses). This is the model of the GPU `Unpolarized`
//!   radiometry mode: cheap, scalar, and it treats the light as re-depolarised
//!   after every interface. For a meridional ray of a coaxial system, whose
//!   s/p planes coincide at every surface, the exact unpolarised value is
//!   `½(Π T_s + Π T_p)`; the two differ only at second order in the
//!   per-surface diattenuations (tested). It needs only the power
//!   coefficients, so [`SequentialTracer::throughput`] evaluates it from the
//!   surface records without building amplitude or PRT data
//!   ([`unpolarized_path_product`]).
//! * [`RadiometryModel::Polarized`]: polarisation ray tracing
//!   ([`crate::polarization`]): `T_interfaces = |J e|²` for a pure input
//!   state, or `½ S₀ tr(J†J) + m·S` for a Stokes state, normalised by `S₀`.
//!   `PolarizationState::Unpolarized` here is the exact unpolarised
//!   throughput, `½ tr(J†J)` (the average over two orthogonal states).
//!
//! # System transmission
//!
//! [`system_transmission`] averages the per-ray throughput over the pupil
//! for each (field, wavelength): equal weight per pupil sample, uniform in
//! normalised pupil coordinates (a grid inside the unit disk, or seeded
//! Philox points `r = √u₁`, `φ = 2πu₂`; a sampling with no points, such as
//! `Grid { n: 2 }` whose four nodes lie outside the disk, is rejected).
//! Rays blocked by an aperture, missing
//! a surface or totally internally reflected transmit nothing; rays that
//! fail numerically are excluded and reported. Pupil-shape and projection
//! effects (relative illumination) belong to the geometric analyses; this is
//! the coating, Fresnel and bulk-absorption transmission of the pupil
//! samples. Sums are compensated and taken in fixed order, so results are
//! deterministic.
//!
//! # Polarisation pupil map
//!
//! [`polarization_pupil_map`] evaluates, on a pupil grid, the amplitude
//! Jones matrix, the transmitted intensity for an input state, the
//! unpolarised transmittance, diattenuation and proper retardance with
//! their axes.

use std::f64::consts::PI;

use mklight_core::job::{Precision, RunTimer};
use mklight_core::math::KahanSum;
use mklight_core::rng::Philox;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use serde::{Deserialize, Serialize};

use crate::batch::run_blocks;
use crate::polarization::{
    InterfaceStep, JonesMatrix, PolarizationProperties, PolarizationState, PolarizedRay,
    unpolarized_path_product,
};
use crate::pupil;
use crate::trace::{TraceOptions, TraceStatus};
use crate::tracer::{FieldPoint, PreparedField, SequentialTracer};

/// Backend name reported in [`RunInfo`].
const BACKEND: &str = "cpu-f64";

/// How interface losses of a ray are combined (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RadiometryModel {
    /// Product of per-interface s/p averages (GPU-compatible).
    #[default]
    Unpolarized,
    /// Polarisation ray tracing with the given input state.
    Polarized { state: PolarizationState },
}

/// Product of the incoherent s/p path factors `½(X_s + X_p)` of the steps.
pub fn unpolarized_interface_product(steps: &[InterfaceStep]) -> f64 {
    steps.iter().map(InterfaceStep::path_unpolarized).product()
}

/// Scalar DOE factor of the selected sequential orders, excluding passive/image surfaces.
pub fn diffraction_product(sys: &mklight_core::ResolvedSystem, wl: usize) -> f64 {
    (1..sys.image_index())
        .filter_map(|i| {
            let s = &sys.surfaces[i];
            if s.is_passive() {
                return None;
            }
            if let mklight_core::SurfaceShape::Diffractive { phase, .. } = &s.shape {
                Some(mklight_core::materials::diffraction::order_efficiency(
                    sys,
                    i,
                    wl,
                    phase.trace_order,
                    s.is_mirror(),
                    false,
                ))
            } else {
                None
            }
        })
        .product()
}

/// Throughput of one ray.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RayThroughput {
    pub status: TraceStatus,
    /// Interface (coating / Fresnel) factor.
    pub interfaces: f64,
    /// Bulk-absorption factor `exp(−Σ αL)`.
    pub internal: f64,
    /// `interfaces · internal` (0 for a ray that did not reach the image).
    pub total: f64,
}

impl RayThroughput {
    fn blocked(status: TraceStatus) -> Self {
        RayThroughput {
            status,
            interfaces: 0.0,
            internal: 0.0,
            total: 0.0,
        }
    }

    /// `Unpolarized`-model throughput of a recorded ray (records as filled
    /// by [`crate::trace_ray`]), from the power coefficients only. Equal to
    /// `RayThroughput::of(&PolarizedRay::from_trace(..), &Unpolarized)`.
    pub fn unpolarized(
        sys: &mklight_core::ResolvedSystem,
        wl: usize,
        trace: &crate::trace::TraceResult,
        records: &[crate::trace::SurfaceRecord],
    ) -> Self {
        if !trace.status.is_ok() {
            return Self::blocked(trace.status);
        }
        let interfaces = unpolarized_path_product(sys, wl, records) * diffraction_product(sys, wl);
        let internal = records
            .last()
            .map_or(trace.transmittance, |r| r.transmittance);
        RayThroughput {
            status: trace.status,
            interfaces,
            internal,
            total: interfaces * internal,
        }
    }

    /// Throughput of a polarisation-traced ray under `model`.
    pub fn of(ray: &PolarizedRay, model: &RadiometryModel) -> Self {
        let status = ray.trace.status;
        if !status.is_ok() {
            return Self::blocked(status);
        }
        let interfaces = match model {
            RadiometryModel::Unpolarized => unpolarized_interface_product(&ray.steps),
            RadiometryModel::Polarized { state } => {
                let s0 = state.stokes().s0;
                if s0 > 0.0 {
                    ray.jones().intensity_of(state) / s0
                } else {
                    0.0
                }
            }
        };
        let internal = ray.internal_transmittance;
        RayThroughput {
            status,
            interfaces,
            internal,
            total: interfaces * internal,
        }
    }
}

impl SequentialTracer {
    /// Throughput of pupil point `(px, py)` under `model`. The
    /// `Unpolarized` model uses the power coefficients only
    /// ([`RayThroughput::unpolarized`]); `Polarized` traces with PRT.
    pub fn throughput(
        &self,
        field: &PreparedField,
        wl: usize,
        px: f64,
        py: f64,
        opts: &TraceOptions,
        model: &RadiometryModel,
    ) -> RayThroughput {
        match model {
            RadiometryModel::Unpolarized => {
                let mut rec = Vec::with_capacity(self.system().surfaces.len());
                let r = self.trace_with(field, wl, px, py, opts, Some(&mut rec));
                RayThroughput::unpolarized(self.system(), wl, &r, &rec)
            }
            RadiometryModel::Polarized { .. } => {
                let mut r =
                    RayThroughput::of(&self.trace_polarized(field, wl, px, py, opts), model);
                let eta = diffraction_product(self.system(), wl);
                r.interfaces *= eta;
                r.total *= eta;
                r
            }
        }
    }
}

// ---------------------------------------------------------------------------
// System transmission.

/// Pupil sampling for pupil averages.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PupilSamples {
    /// `n × n` square grid inside the unit disk ([`pupil::grid`]); `n` must
    /// be 1 (the pupil centre) or at least 3 (`n = 2` has no node inside
    /// the disk).
    Grid { n: usize },
    /// `count` uniform points on the unit disk from Philox stream
    /// `(seed, 0)`: draws `2j`, `2j + 1` give `r = √u₁`, `φ = 2πu₂`.
    Random { count: usize, seed: u64 },
}

impl PupilSamples {
    pub fn points(&self) -> Vec<[f64; 2]> {
        match *self {
            PupilSamples::Grid { n } => pupil::grid(n),
            PupilSamples::Random { count, seed } => {
                let mut rng = Philox::new(seed, 0);
                (0..count)
                    .map(|_| {
                        let r = rng.uniform().sqrt();
                        let (s, c) = (2.0 * PI * rng.uniform()).sin_cos();
                        [r * c, r * s]
                    })
                    .collect()
            }
        }
    }
    fn seed(&self) -> Option<u64> {
        match *self {
            PupilSamples::Random { seed, .. } => Some(seed),
            PupilSamples::Grid { .. } => None,
        }
    }
}

/// Settings of [`system_transmission`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransmissionSettings {
    /// Field points; empty means the system fields.
    #[serde(default)]
    pub fields: Vec<FieldPoint>,
    /// System wavelength indices; empty means all.
    #[serde(default)]
    pub wavelengths: Vec<usize>,
    pub pupil: PupilSamples,
    #[serde(default)]
    pub model: RadiometryModel,
    /// Apply apertures (vignetted rays transmit nothing).
    pub clip: bool,
}

impl Default for TransmissionSettings {
    fn default() -> Self {
        TransmissionSettings {
            fields: Vec::new(),
            wavelengths: Vec::new(),
            pupil: PupilSamples::Grid { n: 32 },
            model: RadiometryModel::Unpolarized,
            clip: true,
        }
    }
}

/// Pupil-averaged transmission of one (field, wavelength).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransmissionPoint {
    /// Index into [`TransmissionResult::fields`].
    pub field: usize,
    /// System wavelength index.
    pub wavelength: usize,
    pub wavelength_um: f64,
    /// Mean throughput over all usable samples (blocked rays count 0).
    pub transmission: f64,
    /// Mean throughput of the rays that reach the image.
    pub mean_unvignetted: f64,
    /// Fraction of usable samples that reach the image.
    pub unvignetted_fraction: f64,
    /// Usable samples (generated and traced without numerical failure).
    pub samples: usize,
    /// Samples excluded because generation or tracing failed numerically.
    pub failed: usize,
}

/// Result of [`system_transmission`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransmissionResult {
    pub fields: Vec<FieldPoint>,
    pub wavelengths: Vec<usize>,
    /// Field-major: entry `f · W + w`.
    pub points: Vec<TransmissionPoint>,
    /// Per field: spectral-weight average of `transmission` over the
    /// requested wavelengths.
    pub polychromatic: Vec<f64>,
    pub run_info: RunInfo,
}

impl TransmissionResult {
    pub fn at(&self, field: usize, wavelength_pos: usize) -> &TransmissionPoint {
        &self.points[field * self.wavelengths.len() + wavelength_pos]
    }
}

fn resolve_fields(
    tracer: &SequentialTracer,
    fields: &[FieldPoint],
) -> MkResult<(Vec<FieldPoint>, Vec<PreparedField>)> {
    if fields.is_empty() {
        let p = tracer.fields().to_vec();
        Ok((p.iter().map(|f| f.point).collect(), p))
    } else {
        let p = fields
            .iter()
            .map(|f| tracer.prepare_field(*f))
            .collect::<MkResult<Vec<_>>>()?;
        Ok((fields.to_vec(), p))
    }
}

fn resolve_wavelengths(tracer: &SequentialTracer, w: &[usize]) -> MkResult<Vec<usize>> {
    let n = tracer.num_wavelengths();
    if let Some(bad) = w.iter().find(|&&i| i >= n) {
        return Err(mklight_core::MkError::InvalidModel(format!(
            "wavelength index {bad} out of range (system has {n})"
        )));
    }
    Ok(if w.is_empty() {
        (0..n).collect()
    } else {
        w.to_vec()
    })
}

/// Pupil points of `samples`, rejecting samplings without any point (they
/// would report zero transmission, indistinguishable from an opaque system).
fn nonempty_points(samples: &PupilSamples) -> MkResult<Vec<[f64; 2]>> {
    let pts = samples.points();
    if pts.is_empty() {
        return Err(MkError::InvalidModel(format!(
            "pupil sampling {samples:?} has no points inside the unit disk \
             (grid n must be 1 or at least 3, random count at least 1)"
        )));
    }
    Ok(pts)
}

fn is_numerical_failure(s: &TraceStatus) -> bool {
    matches!(s, TraceStatus::Failed { .. })
}

/// System transmission versus field and wavelength (module docs).
/// Cancellable, with monotone progress; carries a [`RunInfo`].
pub fn system_transmission(
    tracer: &SequentialTracer,
    settings: &TransmissionSettings,
    ctx: &JobContext,
) -> MkResult<TransmissionResult> {
    let timer = RunTimer::start();
    let (field_points, fields) = resolve_fields(tracer, &settings.fields)?;
    let wls = resolve_wavelengths(tracer, &settings.wavelengths)?;
    if let RadiometryModel::Polarized { state } = &settings.model {
        state.validate()?;
    }
    let pts = nonempty_points(&settings.pupil)?;
    let opts = TraceOptions {
        clip: settings.clip,
        ..tracer.options()
    };
    let (nw, np) = (wls.len(), pts.len());
    let total = fields.len() * nw * np;
    let rays = run_blocks(total, 4096, ctx, "transmission", |i| {
        let (f, w, p) = (i / (nw * np), (i / np) % nw, i % np);
        let pp = pts[p];
        tracer.throughput(&fields[f], wls[w], pp[0], pp[1], &opts, &settings.model)
    })?;
    let sys = tracer.system();
    let mut points = Vec::with_capacity(fields.len() * nw);
    let mut warnings = tracer.warnings().to_vec();
    for f in 0..fields.len() {
        for (wi, &wl) in wls.iter().enumerate() {
            let block = &rays[(f * nw + wi) * np..(f * nw + wi + 1) * np];
            let mut sum = KahanSum::default();
            let (mut usable, mut ok, mut failed) = (0usize, 0usize, 0usize);
            for r in block {
                if is_numerical_failure(&r.status) {
                    failed += 1;
                    continue;
                }
                usable += 1;
                if r.status.is_ok() {
                    ok += 1;
                    sum.add(r.total);
                }
            }
            if failed > 0 {
                warnings.push(format!(
                    "field {f}, wavelength {wl}: {failed} of {np} pupil samples failed and were excluded"
                ));
            }
            let s = sum.value();
            points.push(TransmissionPoint {
                field: f,
                wavelength: wl,
                wavelength_um: sys.wavelengths_um[wl],
                transmission: if usable > 0 { s / usable as f64 } else { 0.0 },
                mean_unvignetted: if ok > 0 { s / ok as f64 } else { 0.0 },
                unvignetted_fraction: if usable > 0 {
                    ok as f64 / usable as f64
                } else {
                    0.0
                },
                samples: usable,
                failed,
            });
        }
    }
    let polychromatic = (0..fields.len())
        .map(|f| {
            let mut num = KahanSum::default();
            let mut den = KahanSum::default();
            for (wi, &wl) in wls.iter().enumerate() {
                let wt = sys.wavelength_weights[wl];
                num.add(wt * points[f * nw + wi].transmission);
                den.add(wt);
            }
            let d = den.value();
            if d > 0.0 { num.value() / d } else { 0.0 }
        })
        .collect();
    let run_info = timer.finish(
        &sys.snapshot,
        "system_transmission",
        settings,
        BACKEND,
        Precision::F64Reference,
        settings.pupil.seed(),
        warnings,
    );
    Ok(TransmissionResult {
        fields: field_points,
        wavelengths: wls,
        points,
        polychromatic,
        run_info,
    })
}

// ---------------------------------------------------------------------------
// Polarisation pupil map.

/// Settings of [`polarization_pupil_map`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolarizationMapSettings {
    pub field: FieldPoint,
    /// System wavelength index.
    pub wavelength: usize,
    /// Pupil grid size (`n × n` inside the unit disk; 1 or at least 3).
    pub grid: usize,
    /// Input state for the `intensity` map.
    pub state: PolarizationState,
    pub clip: bool,
}

/// Polarisation of one pupil sample.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PupilPolarization {
    pub pupil: [f64; 2],
    pub status: TraceStatus,
    /// Amplitude Jones matrix `√τ·J` in the double-pole bases.
    pub jones: JonesMatrix,
    /// Transmitted power fraction for the input state (normalised by `S₀`).
    pub intensity: f64,
    /// Diattenuation and proper retardance (`Qᵀ P`); `t_*` include bulk
    /// absorption.
    pub properties: PolarizationProperties,
    /// Retardance of `jones` itself (double-pole bases; includes the
    /// geometric rotation and mirror parity).
    pub retardance_jones: f64,
}

/// Result of [`polarization_pupil_map`]. Samples whose ray did not reach
/// the image have zero Jones matrices.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolarizationPupilMap {
    pub samples: Vec<PupilPolarization>,
    /// Pupil-mean intensity over the samples reaching the image.
    pub mean_intensity: f64,
    pub max_diattenuation: f64,
    /// Maximum and RMS proper retardance over the samples reaching the
    /// image whose retarder is defined
    /// ([`PolarizationProperties::retarder_defined`]).
    pub max_retardance: f64,
    pub rms_retardance: f64,
    /// Samples reaching the image whose retarder is undefined (nearly
    /// singular Jones matrix), excluded from the retardance statistics.
    #[serde(default)]
    pub undefined_retarder: usize,
    pub run_info: RunInfo,
}

/// Diattenuation / retardance / intensity over the pupil of one (field,
/// wavelength). Cancellable, with monotone progress.
pub fn polarization_pupil_map(
    tracer: &SequentialTracer,
    settings: &PolarizationMapSettings,
    ctx: &JobContext,
) -> MkResult<PolarizationPupilMap> {
    let timer = RunTimer::start();
    let wl = resolve_wavelengths(tracer, &[settings.wavelength])?[0];
    settings.state.validate()?;
    let field = tracer.prepare_field(settings.field)?;
    let pts = nonempty_points(&PupilSamples::Grid { n: settings.grid })?;
    let opts = TraceOptions {
        clip: settings.clip,
        ..tracer.options()
    };
    let s0 = settings.state.stokes().s0;
    let samples = run_blocks(pts.len(), 2048, ctx, "polarization pupil map", |i| {
        let p = pts[i];
        let r = tracer.trace_polarized(&field, wl, p[0], p[1], &opts);
        if !r.is_ok() {
            let z = JonesMatrix {
                m: [[Default::default(); 2]; 2],
            };
            return PupilPolarization {
                pupil: p,
                status: r.trace.status,
                jones: z,
                intensity: 0.0,
                properties: z.properties(),
                retardance_jones: 0.0,
            };
        }
        let eta = diffraction_product(tracer.system(), wl);
        let jones = r.jones_amplitude().scale(eta.sqrt());
        let a = (r.internal_transmittance.max(0.0) * eta).sqrt();
        PupilPolarization {
            pupil: p,
            status: r.trace.status,
            jones,
            intensity: if s0 > 0.0 {
                jones.intensity_of(&settings.state) / s0
            } else {
                0.0
            },
            properties: r.proper_jones().scale(a).properties(),
            retardance_jones: jones.properties().retardance,
        }
    })?;
    let mut sum = KahanSum::default();
    let mut sum_r2 = KahanSum::default();
    let (mut n, mut nr, mut undefined) = (0usize, 0usize, 0usize);
    let (mut dmax, mut rmax) = (0.0f64, 0.0f64);
    for s in samples.iter().filter(|s| s.status.is_ok()) {
        n += 1;
        sum.add(s.intensity);
        dmax = dmax.max(s.properties.diattenuation);
        if s.properties.retarder_defined {
            nr += 1;
            sum_r2.add(s.properties.retardance * s.properties.retardance);
            rmax = rmax.max(s.properties.retardance);
        } else {
            undefined += 1;
        }
    }
    let nf = n.max(1) as f64;
    let mut warnings = tracer.warnings().to_vec();
    if undefined > 0 {
        warnings.push(format!(
            "{undefined} pupil samples have a singular Jones matrix (no defined retarder) \
             and are excluded from the retardance statistics"
        ));
    }
    let run_info = timer.finish(
        &tracer.system().snapshot,
        "polarization_pupil_map",
        settings,
        BACKEND,
        Precision::F64Reference,
        None,
        warnings,
    );
    Ok(PolarizationPupilMap {
        mean_intensity: sum.value() / nf,
        max_diattenuation: dmax,
        max_retardance: rmax,
        rms_retardance: (sum_r2.value() / nr.max(1) as f64).sqrt(),
        undefined_retarder: undefined,
        samples,
        run_info,
    })
}

#[cfg(test)]
mod diffraction_tests {
    use super::*;
    use mklight_core::materials::{MaterialLibrary, diffraction::DiffractionEfficiency};
    use mklight_core::{ResolvedSystem, Snapshot};
    use std::sync::Arc;
    #[test]
    fn system_transmission_includes_selected_order_for_both_models() {
        let mut sys = mklight_core::samples::hybrid_diffractive_achromat();
        sys.wavelengths.wavelengths[0].um = 0.5875618;
        let calculate = |s| {
            let snapshot = Snapshot::new(s).unwrap();
            let resolved =
                ResolvedSystem::new(&snapshot, Arc::new(MaterialLibrary::with_builtin())).unwrap();
            let tracer = SequentialTracer::new(Arc::new(resolved)).unwrap();
            [
                RadiometryModel::Unpolarized,
                RadiometryModel::Polarized {
                    state: PolarizationState::Unpolarized,
                },
            ]
            .map(|model| {
                system_transmission(
                    &tracer,
                    &TransmissionSettings {
                        fields: vec![FieldPoint::normalized(0.0, 0.0)],
                        wavelengths: vec![0],
                        model,
                        pupil: PupilSamples::Grid { n: 1 },
                        ..Default::default()
                    },
                    &JobContext::new(),
                )
                .unwrap()
                .points[0]
                    .transmission
            })
        };
        let ideal = calculate(sys.clone());
        sys.surfaces[2].diffraction_efficiency = DiffractionEfficiency::Multilevel {
            levels: 2.try_into().unwrap(),
        };
        let binary = calculate(sys);
        for i in 0..2 {
            assert!((binary[i] / ideal[i] - 4.0 / PI.powi(2)).abs() < 1e-14);
        }
    }
}
