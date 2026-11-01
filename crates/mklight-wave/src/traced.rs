//! The link to the sequential reference tracer: a [`PupilFunction`] for one
//! (field, wavelength) of a [`SequentialTracer`] (one configuration of a
//! system; see [`tracer_for_configuration`]).
//!
//! # Steps
//!
//! 1. **Reference.** The unclipped chief ray gives the image point `I`, its
//!    direction `d_c` and its optical path. The paraxial exit pupil (at the
//!    ray's wavelength) fixes the exit-pupil point `E = I + s_E·d_c` on the
//!    chief ray, as in `mklight_trace::opd`. The reference sphere is centred
//!    on the reference point `P` ([`ReferencePoint`]: the chief-ray image
//!    point, or the centroid of the traced rays on the image surface) and
//!    passes through `E`: `R = |E − P|`. An image-space telecentric system
//!    (paraxial exit pupil at infinity) has `R = ∞` and is computed in that
//!    limit (below), not rejected.
//! 2. **Rays.** A square lattice of `m` entrance-pupil points across the
//!    unit disk (plus one guard ring outside it) is traced with clipping.
//!    Surviving rays are placed at their reference-sphere position and carry
//!    their OPD and amplitude ([`AmplitudeModel`]).
//! 3. **Rim.** The clear aperture is bounded by marching squares on the
//!    survival of the lattice nodes; every lattice edge that the boundary
//!    crosses is refined by bisection to `rim_tolerance` (the disk edge is
//!    taken analytically when its point survives), and the bracketing
//!    surviving ray is placed on the pupil. The closed loops become the
//!    [`PupilMask`] (even–odd rule, so obscurations are holes), so the
//!    resampled pupil covers the true clear aperture, not the hull of the
//!    lattice rays. Saddle cells are resolved by tracing their centre.
//! 4. **Resampling** onto the `n × n` grid by
//!    [`crate::grid_from_structured`] with that mask.
//!
//! # Optical path difference
//!
//! For a ray reaching the image surface at `Q` with direction `d` and path
//! `OPL_r`, with `w = Q − P`, `b = d·w`, `p² = |w × d|²` and
//! `S = √(R² − p²)`, the ray meets the sphere at `s = −b + σ·S` (`σ` the sign
//! of `s_E`, the side of the exit pupil). With the chief ray's `b_c`, `S_c`:
//!
//! `OPD = (OPL_c + n's_c) − (OPL_r + n's_r)
//!      = (OPL_c − OPL_r) + n'·[b_r − b_c + σ·(p_r²/(R + S_r) − p_c²/(R + S_c))]`
//!
//! (CONVENTIONS §4 sign, cancellation free; for `P = I` it is exactly the
//! formula of `mklight_trace::opd`). As `R → ∞` the bracket tends to
//! `b_r − b_c`: the phase of each ray's plane wave at the reference point.
//! This is the **telecentric** reference (the quantity the Debye integral
//! needs); `mklight_trace::opd` takes the same limit, so the two agree on
//! every ray (a defocused image beyond focus has positive OPD,
//! `n'·δz·(1 − cosθ)`).
//!
//! # Pupil coordinates and mirrors
//!
//! Positions on the sphere relative to `P` are scaled to the normalised
//! pupil (see [`crate::pupil`]): `(u, v) = (x·ê_u, x·ê_v)/(R·sinθ'max)` with
//! `x` the sphere point (for `R = ∞`, `x/R → −d`). `sinθ'max` is the
//! largest `|(α, β)|` over the traced rays and rim points, so the grid
//! always holds the whole beam; it is a scale, not a claim about the F/#
//! (vignetted and decentred pupils keep their true shape in the mask).
//!
//! When the light reaches the image travelling toward local **−z** (after
//! an odd number of mirrors in an unfolded layout), every vector is
//! mirrored to `(x, y, −z)` before the wave-optics frame is built. A
//! reflection preserves distances, so OPD, PSF and MTF are unchanged, and
//! the transverse axes stay the image surface's `x` (sagittal) and `y`
//! (tangential): maps, PSFs and Zernike angles are reported in image-surface
//! coordinates for refracting and reflecting systems alike.
//!
//! # Amplitudes
//!
//! The field amplitude on the reference sphere is `A = √(irradiance per
//! unit sphere area)`. With a uniformly filled entrance pupil (object at
//! infinity) or an isotropic point source (finite object) the flux per
//! unit `(u, v)` area is `T·m/J`, where `T` is the ray's throughput
//! (coatings, Fresnel, absorption: `mklight_trace::radiometry`, unpolarised
//! model), `m` the object-space measure per unit entrance-pupil area (area
//! of the plane wavefront, or direction-cosine area over `cos θ` for a point
//! source) and `J = |∂(u,v)/∂(P_x,P_y)|`. Since the sphere area per unit
//! `(u, v)` area is `∝ 1/γ`, `A ∝ √(T·m·γ/J)` ([`AmplitudeModel::Radiometric`]).
//! [`AmplitudeModel::Transmission`] keeps `√T` only (pupil-distortion and
//! obliquity factors left out, as is customary), [`AmplitudeModel::Uniform`]
//! sets `A = 1`. Amplitudes are normalised to 1 at the chief ray (or the
//! largest value when the chief ray is vignetted).
//!
//! # Image surface
//!
//! The PSF plane is the plane through `P` normal to the image surface's
//! local `z` axis (for a curved image surface, the plane of the vertex
//! frame through `P`; a warning says so).

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem, RunInfo, Snapshot, Vec3};
use mklight_trace::radiometry::RayThroughput;
use mklight_trace::trace::{Ray, TraceResult, trace_ray};
use mklight_trace::{
    ObjectSource, OpdReference, PreparedField, SequentialTracer, SolveSettings,
    apply_solves_configuration,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::interp::{RaySample, StructuredSamples, grid_from_structured};
use crate::pupil::{PupilFunction, PupilGeometry, PupilMask};

/// Centre of the reference sphere.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReferencePoint {
    /// The chief ray's intersection with the image surface (CONVENTIONS §4).
    #[default]
    Chief,
    /// The flux-weighted centroid of the traced rays on the image surface
    /// (removes the wavefront tilt of the centroid shift).
    Centroid,
}

/// How ray amplitudes on the reference sphere are set (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmplitudeModel {
    /// `A = 1`.
    Uniform,
    /// `A = √T` (coating, Fresnel and bulk transmission of each ray).
    #[default]
    Transmission,
    /// `A = √(T·m·γ/J)`: transmission, ray density and obliquity.
    Radiometric,
}

/// Sampling of a traced pupil.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct PupilSettings {
    /// Pupil-function samples across `[−1, 1]`; zero selects from traced slopes.
    pub grid: usize,
    /// Lattice rays across the entrance-pupil diameter (`m`).
    pub rays: usize,
    /// Sub-samples per grid cell and axis in the resampling.
    pub supersample: usize,
    pub reference: ReferencePoint,
    pub amplitude: AmplitudeModel,
    /// Trace the vignetting rim and use it as the clear aperture. Without
    /// it the aperture is the union of the lattice triangles (up to one ray
    /// spacing too small).
    pub rim: bool,
    /// Bisection tolerance of the rim search (normalised entrance-pupil
    /// units).
    pub rim_tolerance: f64,
}

impl Default for PupilSettings {
    fn default() -> Self {
        PupilSettings {
            grid: 0,
            rays: 65,
            supersample: 4,
            reference: ReferencePoint::Chief,
            amplitude: AmplitudeModel::Transmission,
            rim: true,
            rim_tolerance: 1e-10,
        }
    }
}

impl PupilSettings {
    pub fn validate(&self) -> MkResult<()> {
        let ok = (self.grid == 0 || (8..=4096).contains(&self.grid))
            && (8..=2048).contains(&self.rays)
            && (1..=32).contains(&self.supersample)
            && self.rim_tolerance > 0.0
            && self.rim_tolerance < 1e-2;
        if ok {
            Ok(())
        } else {
            Err(MkError::InvalidParam(format!(
                "pupil sampling out of range: {self:?}"
            )))
        }
    }
}

/// Exit-pupil and reference-sphere data of one (field, wavelength).
/// Points and directions are in the image surface's local frame.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExitPupilInfo {
    pub wavelength_um: f64,
    /// Paraxial exit-pupil position relative to the image surface along its
    /// local `z` (mm; `"inf"` when image-space telecentric).
    #[serde(with = "mklight_core::serde_util::f64_inf")]
    pub paraxial_position_mm: f64,
    /// Paraxial exit-pupil radius (mm; `"inf"` when telecentric).
    #[serde(with = "mklight_core::serde_util::f64_inf")]
    pub paraxial_radius_mm: f64,
    pub telecentric: bool,
    /// Chief-ray intersection with the image surface.
    pub chief_point_mm: [f64; 3],
    /// Chief-ray direction in image space.
    pub chief_direction: [f64; 3],
    /// Exit-pupil point `E` on the chief ray (absent when telecentric).
    pub exit_pupil_point_mm: Option<[f64; 3]>,
    pub reference: ReferencePoint,
    /// Centre `P` of the reference sphere.
    pub reference_point_mm: [f64; 3],
    /// `R = |E − P|` (mm; `"inf"` when telecentric).
    #[serde(with = "mklight_core::serde_util::f64_inf")]
    pub reference_radius_mm: f64,
    /// `+1` when the light reaches the image toward local `+z`, `−1` toward
    /// `−z` (the wave-optics frame is then mirrored, module docs).
    pub propagation_sign: f64,
    pub image_index: f64,
    /// Pupil scale `sinθ'max` (module docs) and the matching `n'·sinθ'max`.
    pub sin_max: f64,
    pub numerical_aperture: f64,
    /// Clear area of the traced pupil in normalised `(u, v)` units (`π` for
    /// a full circular pupil at the scale's edge).
    pub clear_area: f64,
    /// Fraction of the entrance-pupil disk that reaches the image, from the
    /// traced rim (or the lattice count without rim tracing).
    pub transmitted_fraction: f64,
    /// Effective working F/# `1/(2n'·sinθ_e)` of the clear area,
    /// `sinθ_e = sinθ'max·√(area/π)`.
    pub effective_f_number: f64,
    pub lattice_rays: usize,
    pub surviving_rays: usize,
    pub rim_points: usize,
}

/// A traced pupil: the sampled [`PupilFunction`] plus everything it was
/// built from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TracedPupil {
    pub info: ExitPupilInfo,
    pub pupil: PupilFunction,
    /// Clear aperture in normalised exit-pupil coordinates (`None` without
    /// rim tracing).
    pub mask: Option<PupilMask>,
    /// The same boundary in normalised entrance-pupil coordinates.
    pub entrance_rim: Option<PupilMask>,
    /// Lattice rays at their pupil positions (for Zernike fits).
    pub samples: StructuredSamples,
    pub warnings: Vec<String>,
}

impl TracedPupil {
    /// Resample the original traced lattice, retaining its reference and mask.
    pub(crate) fn resample(
        &mut self,
        n: usize,
        supersample: usize,
        ctx: &JobContext,
    ) -> MkResult<Vec<String>> {
        if n == self.pupil.n() {
            return Ok(Vec::new());
        }
        let p = &self.pupil;
        let (mut refined, warnings) = grid_from_structured(
            &self.samples,
            n,
            p.wavelength_um(),
            *p.geometry(),
            self.mask.as_ref(),
            supersample,
            ctx,
        )?;
        refined.set_image_offset_mm(p.image_offset_mm());
        self.pupil = refined;
        Ok(warnings.iter().map(ToString::to_string).collect())
    }
}

/// A solved configuration `k` of `system` with its reference tracer
/// (configuration 0 or an empty list is the base system). The tracer's
/// snapshot is that of the solved, configured system and the solve
/// warnings are dropped; [`configured_tracer`] keeps both.
pub fn tracer_for_configuration(
    system: &OpticalSystem,
    library: Arc<MaterialLibrary>,
    k: usize,
) -> MkResult<SequentialTracer> {
    Ok(configured_tracer(system, library, k)?.tracer)
}

/// The reference tracer of one configuration of a system, with what a
/// `RunInfo` needs to say which system and configuration it analyses.
pub struct ConfiguredTracer {
    /// Tracer of the solved configuration.
    pub tracer: SequentialTracer,
    pub configuration: usize,
    /// Hash of the system as given (all configurations, before solves):
    /// the session snapshot an engine checks results against.
    pub base_snapshot_hash: String,
    /// Warnings of applying the configuration and its solves.
    pub solve_warnings: Vec<String>,
}

impl ConfiguredTracer {
    /// Stamp a `RunInfo` of an analysis run on [`Self::tracer`]: the base
    /// snapshot hash replaces the configured one, `"configuration": k` is
    /// added to the settings object, and the solve warnings are prepended.
    pub fn stamp(&self, info: &mut RunInfo) {
        info.snapshot_hash = self.base_snapshot_hash.clone();
        if let serde_json::Value::Object(m) = &mut info.settings {
            m.insert("configuration".into(), self.configuration.into());
        }
        let mut w = self.solve_warnings.clone();
        for x in info.warnings.drain(..) {
            if !w.contains(&x) {
                w.push(x);
            }
        }
        info.warnings = w;
    }

    /// [`Self::stamp`] on the `run_info` of a result serialised as JSON.
    pub fn stamp_json(&self, result: &mut serde_json::Value) -> MkResult<()> {
        let ri = result
            .get_mut("run_info")
            .ok_or_else(|| MkError::Numerical("result has no run_info".into()))?;
        let mut info: RunInfo = serde_json::from_value(ri.clone())
            .map_err(|e| MkError::Numerical(format!("run_info: {e}")))?;
        self.stamp(&mut info);
        *ri = serde_json::to_value(&info)
            .map_err(|e| MkError::Numerical(format!("run_info: {e}")))?;
        Ok(())
    }
}

/// The solved configuration `k` of `system` with its tracer, the base
/// snapshot hash and the solve warnings ([`ConfiguredTracer`]).
pub fn configured_tracer(
    system: &OpticalSystem,
    library: Arc<MaterialLibrary>,
    k: usize,
) -> MkResult<ConfiguredTracer> {
    configured_tracer_job(system, library, k, &JobContext::new())
}

/// [`configured_tracer`] with cancellation and solve progress on the caller's job.
pub fn configured_tracer_job(
    system: &OpticalSystem,
    library: Arc<MaterialLibrary>,
    k: usize,
    ctx: &JobContext,
) -> MkResult<ConfiguredTracer> {
    let base_snapshot_hash = mklight_core::snapshot::hash_system(system)?.to_hex();
    let (solved, report) =
        apply_solves_configuration(system, &library, k, &SolveSettings::default(), ctx)?;
    let snap = Snapshot::new(solved)?;
    let sys = mklight_core::ResolvedSystem::new(&snap, library)?;
    Ok(ConfiguredTracer {
        tracer: SequentialTracer::new(Arc::new(sys))?,
        configuration: k,
        base_snapshot_hash,
        solve_warnings: report.warnings,
    })
}

/// One traced lattice or rim ray.
#[derive(Clone, Copy, Debug)]
struct NodeRay {
    trace: TraceResult,
    start: Ray,
    throughput: f64,
}

/// Evaluate `f(0..total)` in parallel blocks with cancellation and
/// progress (fixed output order).
pub(crate) fn par_blocks<T: Send>(
    total: usize,
    block: usize,
    ctx: &JobContext,
    stage: &str,
    f: impl Fn(usize) -> T + Sync,
) -> MkResult<Vec<T>> {
    let mut out = Vec::with_capacity(total);
    let mut start = 0;
    ctx.report(0.0, stage);
    while start < total {
        ctx.check()?;
        let end = (start + block).min(total);
        out.par_extend((start..end).into_par_iter().map(&f));
        start = end;
        ctx.report(start as f64 / total as f64, stage);
    }
    ctx.report(1.0, stage);
    Ok(out)
}

struct Tracing<'a> {
    tracer: &'a SequentialTracer,
    field: &'a PreparedField,
    wl: usize,
    records: bool,
}

impl Tracing<'_> {
    /// Trace entrance-pupil point `p` with clipping; `None` unless it is
    /// inside the unit disk and reaches the image.
    fn ray(&self, p: [f64; 2]) -> Option<NodeRay> {
        if p[0] * p[0] + p[1] * p[1] > 1.0 + 1e-12 {
            return None;
        }
        let g = self.tracer.generate(self.field, self.wl, p[0], p[1]).ok()?;
        let sys = self.tracer.system();
        let opts = self.tracer.options();
        let (trace, throughput) = if self.records {
            let mut rec = Vec::with_capacity(sys.surfaces.len());
            let t = trace_ray(sys, &g.ray, self.wl, &opts, Some(&mut rec));
            let tp = RayThroughput::unpolarized(sys, self.wl, &t, &rec).total;
            (t, tp)
        } else {
            let t = trace_ray(sys, &g.ray, self.wl, &opts, None);
            (t, 1.0)
        };
        let ok = trace.status.is_ok() && trace.ray.pos.is_finite() && trace.ray.dir.is_finite();
        ok.then_some(NodeRay {
            trace,
            start: g.ray,
            throughput,
        })
    }
}

/// The reference sphere in the wave-optics frame (module docs).
struct Sphere {
    /// Centre `P`.
    p: Vec3,
    /// `R` (`∞` when telecentric).
    r: f64,
    /// Side of the exit pupil, `sgn(s_E)`.
    sheet: f64,
    /// Chief ray's `b_c` and `p_c²/(R + S_c)`.
    b_c: f64,
    q_c: f64,
    n: f64,
    reference: OpdReference,
}

impl Sphere {
    /// `(b, p²/(R + S), S)` of a ray from `q` along `d`; `None` when it
    /// misses the sphere.
    fn terms(&self, q: Vec3, d: Vec3) -> Option<(f64, f64, f64)> {
        let w = q - self.p;
        let b = d.dot(w);
        if !self.r.is_finite() {
            return Some((b, 0.0, f64::INFINITY));
        }
        let p2 = w.cross(d).norm2();
        let rr = self.r * self.r;
        if p2 > rr {
            return None;
        }
        let s = (rr - p2).sqrt();
        Some((b, p2 / (self.r + s), s))
    }

    /// OPD (mm) of a ray with image point `q`, direction `d` (wave frame).
    fn opd_mm(&self, t: &TraceResult, q: Vec3, d: Vec3) -> Option<f64> {
        let (b, qr, _) = self.terms(q, d)?;
        let d_opl = self.reference.chief_opl.diff(&t.opl);
        Some(
            self.n
                .mul_add(b - self.b_c + self.sheet * (qr - self.q_c), d_opl),
        )
    }

    /// Sphere point relative to `P`, mapped to the exit-pupil side and
    /// divided by `R` (for `R = ∞` the limit `−d`).
    fn unit_point(&self, q: Vec3, d: Vec3) -> Option<Vec3> {
        if !self.r.is_finite() {
            return Some(-d);
        }
        let (b, _, s) = self.terms(q, d)?;
        let x = (q - self.p) + d * (-b + self.sheet * s);
        Some(x * (-self.sheet / self.r))
    }
}

/// Signed area of a closed polygon.
fn polygon_area(l: &[[f64; 2]]) -> f64 {
    let mut a = 0.0;
    let mut j = l.len() - 1;
    for i in 0..l.len() {
        a += l[j][0] * l[i][1] - l[i][0] * l[j][1];
        j = i;
    }
    0.5 * a
}

/// Area of the even–odd union of loops: each loop counts with sign
/// `(−1)^depth`, `depth` the number of other loops that contain it.
pub(crate) fn even_odd_area(loops: &[Vec<[f64; 2]>]) -> f64 {
    loops
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let v = l[0];
            let depth = loops
                .iter()
                .enumerate()
                .filter(|(j, o)| *j != i && PupilMask::polygon((*o).clone()).contains(v[0], v[1]))
                .count();
            let a = polygon_area(l).abs();
            if depth % 2 == 0 { a } else { -a }
        })
        .sum()
}

/// Lattice geometry: `side` nodes at `(k − c)·h`, `h = 2/(m − 1)`, so the
/// nodes `1` and `side − 2` sit on `±1` and nodes `0`, `side − 1` form a
/// guard ring outside the unit disk.
pub(crate) struct Lattice {
    side: usize,
    h: f64,
    c: f64,
}

impl Lattice {
    pub(crate) fn new(m: usize) -> Self {
        let side = m + 2;
        Lattice {
            side,
            h: 2.0 / (m - 1) as f64,
            c: 0.5 * (side - 1) as f64,
        }
    }
    fn coord(&self, k: usize) -> f64 {
        (k as f64 - self.c) * self.h
    }
    pub(crate) fn point(&self, i: usize, j: usize) -> [f64; 2] {
        [self.coord(i), self.coord(j)]
    }
}

/// One lattice edge crossed by the boundary: inside node, outside node.
#[derive(Clone, Copy)]
struct Crossing {
    a: [f64; 2],
    b: [f64; 2],
    ray_a: usize,
}

/// Point of `[a, b]` at which the beam's survival ends, by bisection; the
/// returned point and ray are the last surviving ones.
fn bisect_rim(
    tr: &Tracing,
    a: [f64; 2],
    b: [f64; 2],
    ray_a: NodeRay,
    tol: f64,
) -> ([f64; 2], NodeRay) {
    let at = |t: f64| [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
    let len = (b[0] - a[0]).hypot(b[1] - a[1]);
    let mut hi = 1.0;
    if b[0] * b[0] + b[1] * b[1] > 1.0 {
        // Disk crossing |a + t(b − a)| = 1 (a is inside): the larger root.
        let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
        let qa = dx * dx + dy * dy;
        let qb = a[0] * dx + a[1] * dy;
        let qc = a[0] * a[0] + a[1] * a[1] - 1.0;
        let disc = (qb * qb - qa * qc).max(0.0);
        // Stable form of (−qb + √disc)/qa for qc ≤ 0.
        let t = if qb <= 0.0 {
            (-qb + disc.sqrt()) / qa
        } else {
            -qc / (qb + disc.sqrt())
        };
        let t = t.clamp(0.0, 1.0);
        let p = at(t);
        let p = {
            let r = p[0].hypot(p[1]);
            if r > 1.0 { [p[0] / r, p[1] / r] } else { p }
        };
        if let Some(r) = tr.ray(p) {
            return (p, r);
        }
        hi = t;
    }
    let (mut lo, mut ray) = (0.0, ray_a);
    while (hi - lo) * len > tol {
        let mid = 0.5 * (lo + hi);
        match tr.ray(at(mid)) {
            Some(r) => {
                lo = mid;
                ray = r;
            }
            None => hi = mid,
        }
    }
    (at(lo), ray)
}

/// Marching-squares segments of the survival field: pairs of edge ids.
/// Horizontal edge `(i, j)–(i+1, j)` has id `2(j·side + i)`, vertical edge
/// `(i, j)–(i, j+1)` id `2(j·side + i) + 1`.
pub(crate) fn rim_segments(
    lat: &Lattice,
    inside: &[bool],
    centre_inside: impl Fn(usize, usize) -> bool + Sync,
) -> Vec<(usize, usize)> {
    let s = lat.side;
    let hid = |i: usize, j: usize| 2 * (j * s + i);
    let vid = |i: usize, j: usize| 2 * (j * s + i) + 1;
    let cells: Vec<Vec<(usize, usize)>> = (0..(s - 1) * (s - 1))
        .into_par_iter()
        .map(|k| {
            let (i, j) = (k % (s - 1), k / (s - 1));
            let b = |a: usize, c: usize| inside[c * s + a] as usize;
            let case = b(i, j) | b(i + 1, j) << 1 | b(i + 1, j + 1) << 2 | b(i, j + 1) << 3;
            let (e0, e1, e2, e3) = (hid(i, j), vid(i + 1, j), hid(i, j + 1), vid(i, j));
            match case {
                0 | 15 => vec![],
                1 | 14 => vec![(e3, e0)],
                2 | 13 => vec![(e0, e1)],
                3 | 12 => vec![(e3, e1)],
                4 | 11 => vec![(e1, e2)],
                6 | 9 => vec![(e0, e2)],
                7 | 8 => vec![(e3, e2)],
                5 => {
                    if centre_inside(i, j) {
                        vec![(e0, e1), (e2, e3)]
                    } else {
                        vec![(e3, e0), (e1, e2)]
                    }
                }
                10 => {
                    if centre_inside(i, j) {
                        vec![(e3, e0), (e1, e2)]
                    } else {
                        vec![(e0, e1), (e2, e3)]
                    }
                }
                _ => unreachable!(),
            }
        })
        .collect();
    cells.into_iter().flatten().collect()
}

/// Chain segments into closed loops of edge ids (deterministic order).
pub(crate) fn chain_loops(segments: &[(usize, usize)]) -> Vec<Vec<usize>> {
    use std::collections::BTreeMap;
    let mut adj: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (k, &(a, b)) in segments.iter().enumerate() {
        adj.entry(a).or_default().push(k);
        adj.entry(b).or_default().push(k);
    }
    let mut used = vec![false; segments.len()];
    let mut loops = Vec::new();
    for start in 0..segments.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let (first, mut cur) = segments[start];
        let mut lp = vec![first];
        while cur != first {
            lp.push(cur);
            let next = adj[&cur].iter().copied().find(|&k| !used[k]);
            let Some(k) = next else { break };
            used[k] = true;
            let (a, b) = segments[k];
            cur = if a == cur { b } else { a };
        }
        if lp.len() >= 3 {
            loops.push(lp);
        }
    }
    loops
}

/// Trace the pupil of `field` at wavelength index `wl` (module docs).
pub fn trace_pupil(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    settings: &PupilSettings,
    ctx: &JobContext,
) -> MkResult<TracedPupil> {
    settings.validate()?;
    let sys = tracer.system();
    if wl >= sys.wavelengths_um.len() {
        return Err(MkError::InvalidParam(format!(
            "wavelength {wl} does not exist"
        )));
    }
    if !field.is_valid() {
        return Err(MkError::Trace(format!(
            "field unusable: {}",
            field.error.clone().unwrap_or_default()
        )));
    }
    let mut warnings = Vec::new();
    let reference = tracer
        .opd_reference(field, wl)
        .map_err(|st| MkError::Trace(format!("chief ray failed: {st:?}")))?;
    let img = sys.image_index();
    if !matches!(sys.surfaces[img].shape, mklight_core::SurfaceShape::Plane) {
        warnings.push(
            "curved image surface: the PSF plane is normal to the image's vertex axis".into(),
        );
    }
    let fo = tracer.paraxial().at(wl);
    let dc = reference.chief_dir;
    let sign = if dc.z > 0.0 {
        1.0
    } else if dc.z < 0.0 {
        -1.0
    } else {
        return Err(MkError::Trace(
            "chief ray runs parallel to the image surface".into(),
        ));
    };
    let fl = |v: Vec3| Vec3::new(v.x, v.y, sign * v.z);
    let lam_mm = sys.wavelengths_um[wl] * 1e-3;

    // --- Lattice rays -------------------------------------------------------
    let needs_t = settings.amplitude != AmplitudeModel::Uniform;
    let tr = Tracing {
        tracer,
        field,
        wl,
        records: needs_t,
    };
    let lat = Lattice::new(settings.rays);
    let side = lat.side;
    let rays: Vec<Option<NodeRay>> = par_blocks(
        side * side,
        1024,
        &ctx.subrange(0.0, 0.55),
        "tracing pupil rays",
        |k| tr.ray(lat.point(k % side, k / side)),
    )?;
    let inside: Vec<bool> = rays.iter().map(Option::is_some).collect();
    let surviving = inside.iter().filter(|&&b| b).count();
    let lattice_rays = (0..side * side)
        .filter(|&k| {
            let p = lat.point(k % side, k / side);
            p[0] * p[0] + p[1] * p[1] <= 1.0 + 1e-12
        })
        .count();
    if surviving == 0 {
        return Err(MkError::Trace(
            "no ray of the pupil lattice reaches the image".into(),
        ));
    }

    // --- Rim ----------------------------------------------------------------
    // Rim points (entrance-pupil point, its ray) and loops of indices into
    // them.
    type Rim = (Vec<([f64; 2], NodeRay)>, Vec<Vec<usize>>);
    let (rim, rim_loops): Rim = if settings.rim {
        let segs = rim_segments(&lat, &inside, |i, j| {
            let h = 0.5 * lat.h;
            let p = lat.point(i, j);
            tr.ray([p[0] + h, p[1] + h]).is_some()
        });
        ctx.check()?;
        let loops = chain_loops(&segs);
        let mut ids: Vec<usize> = loops.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        let crossings: Vec<Crossing> = ids
            .iter()
            .map(|&id| {
                let k = id / 2;
                let (i, j) = (k % side, k / side);
                let (i2, j2) = if id % 2 == 0 { (i + 1, j) } else { (i, j + 1) };
                let (n1, n2) = (j * side + i, j2 * side + i2);
                let (na, nb) = if inside[n1] { (n1, n2) } else { (n2, n1) };
                Crossing {
                    a: lat.point(na % side, na / side),
                    b: lat.point(nb % side, nb / side),
                    ray_a: na,
                }
            })
            .collect();
        let tol = settings.rim_tolerance;
        let pts = par_blocks(
            crossings.len(),
            64,
            &ctx.subrange(0.55, 0.85),
            "tracing the vignetting rim",
            |q| {
                let c = crossings[q];
                let ra = rays[c.ray_a].expect("inside node has a ray");
                bisect_rim(&tr, c.a, c.b, ra, tol)
            },
        )?;
        // Loops as indices into `pts` (ids are sorted).
        let loops = loops
            .iter()
            .map(|l| {
                l.iter()
                    .map(|id| ids.binary_search(id).expect("edge id"))
                    .collect()
            })
            .collect();
        (pts, loops)
    } else {
        (Vec::new(), Vec::new())
    };

    // --- Reference point and sphere -----------------------------------------
    let weight = |r: &NodeRay| if needs_t { r.throughput } else { 1.0 };
    let centre = match settings.reference {
        ReferencePoint::Chief => reference.center,
        ReferencePoint::Centroid => {
            let (mut sx, mut sy, mut sw) = (
                mklight_core::math::KahanSum::default(),
                mklight_core::math::KahanSum::default(),
                0.0,
            );
            for r in rays.iter().flatten() {
                let w = weight(r);
                let q = r.trace.ray.pos - reference.center;
                sx.add(w * q.x);
                sy.add(w * q.y);
                sw += w;
            }
            if !(sw > 0.0) {
                return Err(MkError::Numerical("no flux reaches the image".into()));
            }
            let (x, y) = (
                reference.center.x + sx.value() / sw,
                reference.center.y + sy.value() / sw,
            );
            let z = sys.surfaces[img]
                .shape
                .sag(x * x + y * y)
                .unwrap_or(reference.center.z);
            Vec3::new(x, y, z)
        }
    };
    let s_e = reference.s_exit_pupil;
    let telecentric = !s_e.is_finite();
    let (p, e_pt, r, sheet, axis) = if telecentric {
        (fl(centre), None, f64::INFINITY, -1.0, fl(dc))
    } else {
        if s_e == 0.0 {
            return Err(MkError::Trace("exit pupil lies on the image".into()));
        }
        let e = reference.center + dc * s_e;
        let r = (e - centre).norm();
        let sheet = s_e.signum();
        let ax = (fl(centre) - fl(e)) * (-sheet / r);
        (fl(centre), Some(e), r, sheet, ax)
    };
    let mut sphere = Sphere {
        p,
        r,
        sheet,
        b_c: 0.0,
        q_c: 0.0,
        n: reference.n_image,
        reference,
    };
    let (b_c, q_c, _) = sphere
        .terms(fl(reference.center), fl(dc))
        .ok_or_else(|| MkError::Trace("chief ray misses the reference sphere".into()))?;
    sphere.b_c = b_c;
    sphere.q_c = q_c;
    let geom0 = if telecentric {
        PupilGeometry::telecentric(0.5, sphere.n)
    } else {
        PupilGeometry::new(0.5 * r, r, sphere.n)
    }?
    .with_chief_ray([axis.x, axis.y, axis.z])?;
    let [eu, ev, ew] = geom0.chief_frame();

    // --- Ray samples ----------------------------------------------------------
    // (α, β, γ, OPD waves) per surviving ray.
    let place = |nr: &NodeRay| -> Option<(f64, f64, f64, f64)> {
        let q = fl(nr.trace.ray.pos);
        let d = fl(nr.trace.ray.dir);
        let x = sphere.unit_point(q, d)?;
        let opd = sphere.opd_mm(&nr.trace, q, d)? / lam_mm;
        // The ray's γ against the axis (Debye factor of its direction).
        Some((x.dot(eu), x.dot(ev), d.dot(ew).abs(), opd))
    };
    let placed: Vec<Option<(f64, f64, f64, f64)>> =
        rays.iter().map(|r| r.as_ref().and_then(place)).collect();
    let rim_placed: Vec<Option<(f64, f64, f64, f64)>> =
        rim.iter().map(|(_, nr)| place(nr)).collect();
    let mut sin_max: f64 = 0.0;
    for (a, b, _, _) in placed.iter().chain(&rim_placed).flatten() {
        sin_max = sin_max.max(a.hypot(*b));
    }
    if placed.iter().filter(|p| p.is_some()).count() < 3 {
        return Err(MkError::Trace(
            "too few rays reach the reference sphere".into(),
        ));
    }
    let sin_max = sin_max * (1.0 + 1e-9);
    if !(sin_max > 0.0 && sin_max < 1.0) {
        return Err(MkError::Trace(format!(
            "beam half-angle sine {sin_max} out of range"
        )));
    }
    let geometry = if telecentric {
        PupilGeometry::telecentric(sin_max, sphere.n)
    } else {
        PupilGeometry::new(sin_max * r, r, sphere.n)
    }?
    .with_chief_ray([axis.x, axis.y, axis.z])?;

    // --- Amplitudes -----------------------------------------------------------
    let amp: Vec<f64> = match settings.amplitude {
        AmplitudeModel::Uniform => vec![1.0; side * side],
        AmplitudeModel::Transmission => rays
            .iter()
            .map(|r| r.as_ref().map_or(0.0, |r| r.throughput.max(0.0).sqrt()))
            .collect(),
        AmplitudeModel::Radiometric => radiometric_amplitudes(field, &lat, &rays, &placed, sin_max),
    };
    let centre_node = (side / 2) * side + side / 2;
    let norm = if side % 2 == 1 && placed[centre_node].is_some() && amp[centre_node] > 0.0 {
        amp[centre_node]
    } else {
        placed
            .iter()
            .zip(&amp)
            .filter(|(p, _)| p.is_some())
            .map(|(_, a)| *a)
            .fold(0.0, f64::max)
    };
    if !(norm > 0.0) {
        return Err(MkError::Numerical("pupil transmits no light".into()));
    }
    let samples = StructuredSamples {
        nx: side,
        ny: side,
        samples: placed
            .iter()
            .zip(&amp)
            .map(|(p, a)| {
                p.map(|(al, be, _, w)| RaySample {
                    u: al / sin_max,
                    v: be / sin_max,
                    opd_waves: w,
                    amplitude: a / norm,
                })
            })
            .collect(),
    };

    // --- Mask -------------------------------------------------------------------
    let to_uv = |k: usize| rim_placed[k].map(|(a, b, _, _)| [a / sin_max, b / sin_max]);
    let mut uv_loops = Vec::new();
    let mut ep_loops = Vec::new();
    for l in &rim_loops {
        let uv: Vec<[f64; 2]> = l.iter().filter_map(|&k| to_uv(k)).collect();
        let ep: Vec<[f64; 2]> = l.iter().map(|&k| rim[k].0).collect();
        if uv.len() >= 3 {
            uv_loops.push(uv);
            ep_loops.push(ep);
        }
    }
    let (mask, entrance_rim) = if uv_loops.is_empty() {
        if settings.rim {
            warnings.push("no vignetting rim found; the clear aperture is the ray hull".into());
        }
        (None, None)
    } else {
        (
            Some(PupilMask::Polygons {
                loops: uv_loops.clone(),
            }),
            Some(PupilMask::Polygons {
                loops: ep_loops.clone(),
            }),
        )
    };
    let grid = if settings.grid == 0 {
        crate::sampling::pupil_grid(&samples)
    } else {
        settings.grid
    };
    let (mut pupil, wv) = grid_from_structured(
        &samples,
        grid,
        sys.wavelengths_um[wl],
        geometry,
        mask.as_ref(),
        settings.supersample,
        &ctx.subrange(0.85, 1.0),
    )?;
    pupil.set_image_offset_mm([0.0, 0.0]);
    warnings.extend(wv.iter().map(|w| w.to_string()));
    let failed = (0..side * side)
        .filter(|&k| {
            let p = lat.point(k % side, k / side);
            p[0] * p[0] + p[1] * p[1] <= 1.0 + 1e-12 && placed[k].is_none() && rays[k].is_some()
        })
        .count();
    if failed > 0 {
        warnings.push(format!("{failed} rays miss the reference sphere"));
    }

    let clear_area = if uv_loops.is_empty() {
        pupil.coverage().iter().sum::<f64>() * pupil.spacing().powi(2)
    } else {
        even_odd_area(&uv_loops)
    };
    let transmitted_fraction = if ep_loops.is_empty() {
        surviving as f64 / lattice_rays.max(1) as f64
    } else {
        even_odd_area(&ep_loops) / std::f64::consts::PI
    };
    let sin_e = sin_max * (clear_area / std::f64::consts::PI).sqrt();
    let info = ExitPupilInfo {
        wavelength_um: sys.wavelengths_um[wl],
        paraxial_position_mm: fo.exit_pupil_position,
        paraxial_radius_mm: 0.5 * fo.exit_pupil_diameter,
        telecentric,
        chief_point_mm: v3(reference.center),
        chief_direction: v3(dc),
        exit_pupil_point_mm: e_pt.map(v3),
        reference: settings.reference,
        reference_point_mm: v3(centre),
        reference_radius_mm: r,
        propagation_sign: sign,
        image_index: sphere.n,
        sin_max,
        numerical_aperture: sphere.n * sin_max,
        clear_area,
        transmitted_fraction,
        effective_f_number: 0.5 / (sphere.n * sin_e),
        lattice_rays,
        surviving_rays: surviving,
        rim_points: rim.len(),
    };
    Ok(TracedPupil {
        info,
        pupil,
        mask,
        entrance_rim,
        samples,
        warnings,
    })
}

fn v3(v: Vec3) -> [f64; 3] {
    [v.x, v.y, v.z]
}

/// `√(T·m·γ/J)` per lattice node (module docs), from central (or
/// one-sided) differences between neighbouring surviving rays.
fn radiometric_amplitudes(
    field: &PreparedField,
    lat: &Lattice,
    rays: &[Option<NodeRay>],
    placed: &[Option<(f64, f64, f64, f64)>],
    sin_max: f64,
) -> Vec<f64> {
    let side = lat.side;
    // Object-space coordinates of each start ray: the transverse position
    // on the field's plane wavefront, or the direction cosines (L, M) of a
    // point source, whose measure carries 1/cosθ.
    let (e1, e2, at_inf) = match field.source {
        ObjectSource::Direction { dir } => {
            let a = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
            let e1 = (a - dir * a.dot(dir)).normalized();
            (e1, dir.cross(e1), true)
        }
        ObjectSource::Point { .. } => (Vec3::X, Vec3::Y, false),
    };
    let obj = |r: &NodeRay| -> [f64; 2] {
        if at_inf {
            [r.start.pos.dot(e1), r.start.pos.dot(e2)]
        } else {
            [r.start.dir.x, r.start.dir.y]
        }
    };
    let uv = |k: usize| placed[k].map(|(a, b, _, _)| [a / sin_max, b / sin_max]);
    let h = lat.h;
    // Derivative of f along the lattice axis `axis` at node (i, j).
    let deriv = |i: usize, j: usize, axis: usize, f: &dyn Fn(usize) -> Option<[f64; 2]>| {
        let idx = |a: isize, b: isize| -> Option<usize> {
            (a >= 0 && b >= 0 && (a as usize) < side && (b as usize) < side)
                .then(|| b as usize * side + a as usize)
        };
        let (di, dj) = if axis == 0 { (1, 0) } else { (0, 1) };
        let (i, j) = (i as isize, j as isize);
        let fp = idx(i + di, j + dj).and_then(f);
        let fm = idx(i - di, j - dj).and_then(f);
        let f0 = idx(i, j).and_then(f)?;
        match (fp, fm) {
            (Some(a), Some(b)) => Some([(a[0] - b[0]) / (2.0 * h), (a[1] - b[1]) / (2.0 * h)]),
            (Some(a), None) => Some([(a[0] - f0[0]) / h, (a[1] - f0[1]) / h]),
            (None, Some(b)) => Some([(f0[0] - b[0]) / h, (f0[1] - b[1]) / h]),
            _ => None,
        }
    };
    let fobj = |k: usize| rays[k].as_ref().map(obj);
    let raw: Vec<Option<f64>> = (0..side * side)
        .map(|k| {
            let r = rays[k].as_ref()?;
            let (_, _, gamma, _) = placed[k]?;
            let (i, j) = (k % side, k / side);
            let (ux, uy) = (deriv(i, j, 0, &uv)?, deriv(i, j, 1, &uv)?);
            let (ox, oy) = (deriv(i, j, 0, &fobj)?, deriv(i, j, 1, &fobj)?);
            let jac = (ux[0] * uy[1] - ux[1] * uy[0]).abs();
            let mut m = (ox[0] * oy[1] - ox[1] * oy[0]).abs();
            if !at_inf {
                m /= r.start.dir.z.abs().max(1e-12);
            }
            (jac > 0.0).then(|| (r.throughput * m * gamma / jac).max(0.0).sqrt())
        })
        .collect();
    // Nodes without a usable difference stencil take the mean of the rest.
    let vals: Vec<f64> = raw.iter().flatten().copied().collect();
    let mean = if vals.is_empty() {
        1.0
    } else {
        vals.iter().sum::<f64>() / vals.len() as f64
    };
    raw.iter()
        .zip(rays)
        .map(|(v, r)| match (v, r) {
            (Some(v), _) => *v,
            (None, Some(_)) => mean,
            _ => 0.0,
        })
        .collect()
}
