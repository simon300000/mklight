//! The connection to the sequential reference tracer: a non-sequential
//! scene sized by the sequential model, and sources aimed by its paraxial
//! pupil data.
//!
//! [`StraySystem::new`] builds a `mklight_trace::SequentialTracer` for the
//! resolved system, computes its automatic clear semi-diameters
//! (`SequentialTracer::semi_diameters`) and builds the [`Scene`] with them
//! (`SceneOptions::clear_semi_diameters`):
//!
//! * an `Auto` surface gets its largest traced ray height over all system
//!   fields and wavelengths;
//! * the `Auto` aperture stop gets `max(paraxial stop radius, traced
//!   height)`, the same freeze rule as `mklight_trace::fix_semi_diameters`,
//!   so the stop plate never vignettes the design's own fields;
//! * an `Auto` image surface without a sensor gets its traced height times
//!   `1 + detector_margin`, so the detector covers the whole image;
//! * fixed semi-diameters, explicit apertures and values supplied by the
//!   caller are kept.
//!
//! The stop and the other optical surfaces take their radii from
//! `mklight_ghost::clip_apertures` (clear apertures, no margin), the single
//! definition of the apertures an explicit ghost run clips at, so the
//! stray solver's `Ghost` class and a ghost run with default settings
//! partition at identical apertures (CONVENTIONS §10).
//! [`StraySystem::ghost_apertures`] reports the clipping of the built scene
//! in the ghost solver's terms, and [`StraySystem::check_ghost_coverage`]
//! checks it against a ghost run's [`GhostCoverage`].
//!
//! The same values define the *clipped sequential model*
//! ([`StraySystem::clipped_system`]): the resolved system with every
//! automatic semi-diameter frozen to the scene's clear aperture. Its
//! transmitted rays are the `Primary` paths of the Monte Carlo **unless
//! non-sequential geometry obstructs them** (the back of a fold mirror, a
//! secondary mount, a baffle in the beam), which the sequential model cannot
//! see. [`crate::reference`] checks every designed path against the scene;
//! [`StraySystem::new`] measures the obstructed fraction of every system
//! field ([`StraySystem::obstruction`]) and warns when it is not negligible.
//!
//! # Sources
//!
//! All margins are relative overfills of the entrance-pupil radius (or of
//! the scene's projected radius), so the stop and the mounts, not the
//! source, limit the beam.
//!
//! * [`StraySystem::field_source`], object at infinity: a uniform collimated
//!   beam along the field direction that fills a disk of radius
//!   `r_EP·(1 + margin)` on the paraxial entrance-pupil plane `z = z_EP`,
//!   centred on the chief ray's aim point
//!   ([`SourceSpec::CollimatedThrough`]). `strength` is the irradiance on a
//!   plane normal to the beam (W/mm²), so the power is
//!   `E·π r²·|d_z|`.
//! * [`StraySystem::field_source`], finite object: a point source at the
//!   object point that emits uniformly in solid angle into a cone around the
//!   line to the chief ray's aim point, whose half-angle `α` has
//!   `tan α = (1 + margin)·tan α_EP` (`α_EP`: the largest angle from the
//!   axis to the entrance-pupil rim). An object-space telecentric system
//!   uses the chief direction and the marginal slope instead. `strength` is
//!   the intensity (W/sr).
//! * [`StraySystem::out_of_field_source`]: a collimated beam from angles
//!   `(θx, θy)` (for example the sun outside the field of view) on a disk
//!   normal to the beam that covers the projection of the **whole scene**
//!   along the beam: every corner of every primitive's bounding box is
//!   projected onto the plane normal to the beam, the disk is centred on
//!   the centre of the projections' bounding rectangle and its radius is
//!   `(1 + margin)` times the largest distance of a projected corner from
//!   that centre. Every ray that meets any geometry, including a lens hood
//!   or telescope tube far in front of the first vertex, therefore starts
//!   inside the disk.
//!
//! Each source comes with a *reference flux* ([`Beam::reference_flux`]),
//! the usual normalisation of stray-light ratios: the flux through the
//! entrance pupil for field sources (the margin excluded; for a finite
//! object, the flux into the circular cone of half-angle `α_EP`, which off
//! axis slightly exceeds the flux through the elliptical pupil), or through
//! the first interface's mechanical aperture for out-of-field sources.

use std::f64::consts::PI;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use mklight_core::error::{MkError, MkResult};
use mklight_core::job::JobContext;
use mklight_core::math::Vec3;
use mklight_core::model::SurfaceShape;
use mklight_core::resolved::ResolvedSystem;
use mklight_ghost::{Clip, ClipMode, GhostCoverage, clip_apertures, interface_kinds};
use mklight_trace::{
    FieldPoint, ObjectSource, SemiDiameterSettings, SemiDiameterSource, SemiDiameters,
    SequentialTracer,
};

use crate::geom::onb;
use crate::reference::primary_reference;
use crate::scene::{Scene, SceneOptions};
use crate::source::SourceSpec;

/// Largest detector map, in pixels (`nx·ny`). The tallies hold about
/// 0.5 kB per pixel (per-class sums and squares, result maps and errors),
/// so this bounds them near 0.5 GB.
pub const MAX_DETECTOR_PIXELS: u64 = 1 << 20;

/// Obstructed fraction of a system field's designed flux above which
/// [`StraySystem::new`] warns.
pub const OBSTRUCTION_WARNING: f64 = 1e-3;

/// Grid size of the obstruction check in [`StraySystem::new`].
const OBSTRUCTION_GRID: usize = 48;

/// Options of [`StraySystem::new`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SystemOptions {
    /// Scene options. Entries of `clear_semi_diameters` that are `Some`
    /// override the traced values.
    pub scene: SceneOptions,
    pub semi_diameters: SemiDiameterSettings,
    /// Relative margin on the traced image height that sizes the detector
    /// of an automatic image surface without a sensor.
    pub detector_margin: f64,
}

impl Default for SystemOptions {
    fn default() -> Self {
        SystemOptions {
            scene: SceneOptions::default(),
            semi_diameters: SemiDiameterSettings::default(),
            detector_margin: 0.05,
        }
    }
}

impl SystemOptions {
    pub fn validate(&self) -> MkResult<()> {
        if !(self.detector_margin.is_finite() && self.detector_margin >= 0.0) {
            return Err(MkError::InvalidModel(
                "detector margin must be finite and ≥ 0".into(),
            ));
        }
        let (nx, ny) = self.scene.detector_pixels;
        if nx == 0 || ny == 0 {
            return Err(MkError::InvalidModel("detector needs pixels".into()));
        }
        if nx as u64 * ny as u64 > MAX_DETECTOR_PIXELS {
            return Err(MkError::InvalidModel(format!(
                "detector map {nx} × {ny} exceeds {MAX_DETECTOR_PIXELS} pixels"
            )));
        }
        Ok(())
    }
}

/// Shape of the detector in its local `(x, y)`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DetectorShape {
    /// The sensor rectangle.
    Rect { half_x: f64, half_y: f64 },
    /// The image surface's clear disk.
    Disk { radius: f64 },
}

impl DetectorShape {
    pub fn contains(&self, x: f64, y: f64) -> bool {
        match *self {
            DetectorShape::Rect { half_x, half_y } => x.abs() <= half_x && y.abs() <= half_y,
            DetectorShape::Disk { radius } => x * x + y * y <= radius * radius,
        }
    }
}

/// A source with its normalisation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Beam {
    pub spec: SourceSpec,
    /// Flux through the entrance pupil (field sources, without the margin)
    /// or the front aperture (out-of-field sources), W.
    pub reference_flux: f64,
    pub description: String,
}

/// A resolved system together with its sequential tracer, automatic
/// semi-diameters and non-sequential scene (module docs).
#[derive(Clone, Debug)]
pub struct StraySystem {
    tracer: SequentialTracer,
    pub semi_diameters: SemiDiameters,
    /// Clear semi-diameters passed to the scene, per surface (`None` where
    /// the surface's own fixed value or aperture applies).
    pub clear_semi_diameters: Vec<Option<f64>>,
    pub scene: Scene,
    pub detector_shape: DetectorShape,
    /// The options the system was built with.
    pub options: SystemOptions,
    /// Per system field: the fraction of the designed (sequential) flux of a
    /// pupil-filling beam that non-sequential geometry obstructs
    /// ([`crate::reference`]), `None` where no field source exists.
    pub obstruction: Vec<Option<f64>>,
    /// Warnings of the sizing traces, the scene and the obstruction check.
    pub warnings: Vec<String>,
}

/// Per-system-field obstruction and its warnings ([`StraySystem::obstruction_checks`]).
#[derive(Clone, Debug, Default)]
pub(crate) struct ObstructionChecks {
    pub(crate) obstruction: Vec<Option<f64>>,
    pub(crate) warnings: Vec<String>,
}

impl StraySystem {
    /// Builds the tracer, the semi-diameters and the scene.
    pub fn new(rs: Arc<ResolvedSystem>, opts: &SystemOptions) -> MkResult<StraySystem> {
        Self::build(rs, opts, &JobContext::new())
    }

    /// As [`StraySystem::new`], with cancellation and progress (module docs).
    pub fn build(
        rs: Arc<ResolvedSystem>,
        opts: &SystemOptions,
        ctx: &JobContext,
    ) -> MkResult<StraySystem> {
        ctx.check()?;
        Self::with_tracer_ctx(SequentialTracer::new(rs)?, opts, ctx)
    }

    /// As [`StraySystem::new`] with an existing tracer (custom aiming).
    pub fn with_tracer(tracer: SequentialTracer, opts: &SystemOptions) -> MkResult<StraySystem> {
        Self::with_tracer_ctx(tracer, opts, &JobContext::new())
    }

    /// As [`StraySystem::with_tracer`], with cancellation and progress.
    pub fn with_tracer_ctx(
        tracer: SequentialTracer,
        opts: &SystemOptions,
        ctx: &JobContext,
    ) -> MkResult<StraySystem> {
        Self::with_tracer_impl(tracer, opts, false, ctx)
    }

    fn with_tracer_impl(
        tracer: SequentialTracer,
        opts: &SystemOptions,
        defer_checks: bool,
        ctx: &JobContext,
    ) -> MkResult<StraySystem> {
        opts.validate()?;
        ctx.check()?;
        ctx.report(0.0, "stray light: sizing apertures");
        let sd = tracer.semi_diameters_with(&opts.semi_diameters, &ctx.subrange(0.0, 0.4))?;
        ctx.check()?;
        ctx.report(0.4, "stray light: building the scene");
        let rs = tracer.system();
        let n = rs.num_surfaces();
        let image = n - 1;
        let has_sensor = rs.system().sensor.is_some();
        // The ghost solver's clear apertures, shared so that both solvers
        // clip at bit-identical radii.
        let ghost = clip_apertures(
            rs,
            &interface_kinds(rs),
            &sd,
            ClipMode::ClearApertures { margin_mm: 0.0 },
        );
        let mut clear = vec![None; n];
        for (j, c) in clear.iter_mut().enumerate() {
            let user = opts
                .scene
                .clear_semi_diameters
                .get(j)
                .copied()
                .flatten()
                .filter(|r| r.is_finite() && *r > 0.0);
            let traced = match (sd.sources[j], ghost[j]) {
                (SemiDiameterSource::Traced, _) if j == image => {
                    Some(sd.values[j] * (1.0 + opts.detector_margin))
                }
                (
                    SemiDiameterSource::Traced | SemiDiameterSource::Stop,
                    Clip::Radius { radius },
                ) => Some(radius),
                // Passive surfaces (no geometry unless they are the stop).
                (SemiDiameterSource::Traced, _) => Some(sd.values[j]),
                (SemiDiameterSource::Stop, _) => Some(sd.values[j].max(sd.traced[j])),
                (SemiDiameterSource::Fixed | SemiDiameterSource::None, _) => None,
            };
            *c = user.or(traced.filter(|r| r.is_finite() && *r > 0.0));
        }
        let mut sopts = opts.scene.clone();
        sopts.clear_semi_diameters = clear.clone();
        let scene = Scene::from_system(rs, &sopts)?;
        ctx.check()?;
        ctx.report(0.6, "stray light: checking for obstruction");
        let detector_shape = if has_sensor {
            DetectorShape::Rect {
                half_x: scene.detector.half_x,
                half_y: scene.detector.half_y,
            }
        } else {
            DetectorShape::Disk {
                radius: scene.detector.half_x,
            }
        };
        let mut sys = StraySystem {
            tracer,
            semi_diameters: sd,
            clear_semi_diameters: clear,
            scene,
            detector_shape,
            options: opts.clone(),
            obstruction: Vec::new(),
            warnings: Vec::new(),
        };
        let mut warnings: Vec<String> = sys.tracer.warnings().to_vec();
        let f = sys.semi_diameters.failed_rays;
        if f > 0 {
            warnings.push(format!(
                "{f} sizing rays failed in the sequential trace; automatic semi-diameters may be too small"
            ));
        }
        sys.warnings = warnings;
        if defer_checks {
            return Ok(sys);
        }
        let checks = sys.obstruction_checks(&ctx.subrange(0.6, 1.0))?;
        sys.finish_checks(checks);
        ctx.report(1.0, "");
        Ok(sys)
    }

    /// As [`StraySystem::build`] without the obstruction check of the system
    /// fields: `obstruction` is empty and `warnings` holds only the sizing
    /// warnings until [`Self::finish_checks`] gets the checks of
    /// [`Self::obstruction_checks`], which can run beside other work.
    pub(crate) fn build_scene(
        rs: Arc<ResolvedSystem>,
        opts: &SystemOptions,
        ctx: &JobContext,
    ) -> MkResult<StraySystem> {
        ctx.check()?;
        Self::with_tracer_impl(SequentialTracer::new(rs)?, opts, true, ctx)
    }

    /// The obstruction of a pupil-filling beam for every system field and
    /// the warnings it raises (the check [`StraySystem::build`] makes).
    pub(crate) fn obstruction_checks(&self, ctx: &JobContext) -> MkResult<ObstructionChecks> {
        use rayon::prelude::*;
        let mut out = ObstructionChecks::default();
        let nf = self.resolved().system().fields.fields.len();
        // Fields in parallel (each reference is parallel over its rows but
        // too small to fill the cores), kept in field order.
        let done = std::sync::Mutex::new(0usize);
        let checks = (0..nf)
            .into_par_iter()
            .map(|k| {
                let quiet = JobContext::with(ctx.cancel.clone(), Default::default());
                let o = match self.system_field_source(k, 0.1, 1.0) {
                    Ok(beam) => Some(
                        primary_reference(self, &beam.spec, OBSTRUCTION_GRID, &quiet)?
                            .obstruction(),
                    ),
                    Err(_) => None,
                };
                // Counted and reported under the lock: monotone progress.
                let mut d = done.lock().unwrap_or_else(|e| e.into_inner());
                *d += 1;
                ctx.report(*d as f64 / nf as f64, "");
                Ok(o)
            })
            .collect::<MkResult<Vec<_>>>()?;
        for (k, o) in checks.into_iter().enumerate() {
            if let Some(o) = o
                && o > OBSTRUCTION_WARNING
            {
                out.warnings.push(format!(
                    "field {k}: {:.2} % of the designed flux is obstructed by non-sequential geometry; a purely sequential Primary overcounts it (use the Monte-Carlo Primary or primary_reference)",
                    100.0 * o
                ));
            }
            out.obstruction.push(o);
        }
        Ok(out)
    }

    /// Completes a system from [`Self::build_scene`] with its checks.
    pub(crate) fn finish_checks(&mut self, checks: ObstructionChecks) {
        self.obstruction = checks.obstruction;
        self.warnings.extend(checks.warnings);
        for w in &self.scene.warnings {
            if !self.warnings.contains(w) {
                self.warnings.push(w.clone());
            }
        }
    }

    pub fn tracer(&self) -> &SequentialTracer {
        &self.tracer
    }

    /// How the scene clips each surface, in the ghost solver's terms
    /// (`mklight_ghost::Clip`): `Surface` for a fixed semi-diameter or
    /// explicit aperture, `Radius` for an automatic clear aperture of an
    /// interface or the stop plate (outside it the scene has an opaque
    /// mount, or a polished zone whose hits are `Sneak`), `None` for the
    /// object, coordinate breaks, surfaces without geometry and an
    /// automatic image surface (the detector bounds detection, not a path).
    /// The stray `Ghost` class holds only paths inside these apertures.
    pub fn ghost_apertures(&self) -> Vec<Clip> {
        let rs = self.resolved();
        let image = rs.image_index();
        rs.surfaces
            .iter()
            .enumerate()
            .map(|(j, s)| {
                if j == 0 || matches!(s.shape, SurfaceShape::CoordinateBreak) {
                    Clip::None
                } else if s.aperture.is_some() || s.fixed_semi_diameter.is_some() {
                    Clip::Surface
                } else if j == image {
                    Clip::None
                } else {
                    match &self.scene.surfaces[j] {
                        Some(g) => Clip::Radius {
                            radius: g.clear_semi_diameter,
                        },
                        None => Clip::None,
                    }
                }
            })
            .collect()
    }

    /// Checks that this scene partitions consistently with the ghost run of
    /// `coverage` (CONVENTIONS §10): the same interface kinds and the same
    /// clipping apertures ([`GhostCoverage::same_apertures`] with
    /// [`Self::ghost_apertures`]). Run the stray solver with the `Ghost`
    /// class excluded only when this holds; otherwise paths clipped by one
    /// solver and not the other are counted twice or not at all.
    pub fn check_ghost_coverage(&self, coverage: &GhostCoverage) -> MkResult<()> {
        let kinds = interface_kinds(self.resolved());
        if coverage.interfaces != kinds {
            return Err(MkError::InvalidModel(
                "the ghost run used a different system (interface kinds differ)".into(),
            ));
        }
        let mine = self.ghost_apertures();
        if !coverage.same_apertures(&mine) {
            let j = (0..mine.len())
                .find(|&j| coverage.apertures.get(j) != Some(&mine[j]))
                .unwrap_or(mine.len());
            return Err(MkError::InvalidModel(format!(
                "the ghost run clips at different apertures than the stray scene \
                 (first at surface {j}: ghost {:?}, stray {:?}); use the default ghost clipping \
                 (clear apertures, no margin) and this scene's semi-diameters \
                 (GhostModel::with_semi_diameters)",
                coverage.apertures.get(j),
                mine.get(j)
            )));
        }
        Ok(())
    }
    pub fn resolved(&self) -> &ResolvedSystem {
        self.tracer.system()
    }

    /// The clipped sequential model (module docs): the resolved system with
    /// every automatic semi-diameter frozen to the scene's clear aperture.
    /// The image surface is left unclipped (use [`Self::detector_shape`]).
    pub fn clipped_system(&self) -> ResolvedSystem {
        let mut rs = self.resolved().clone();
        let image = rs.image_index();
        for (j, s) in rs.surfaces.iter_mut().enumerate() {
            if j == 0 || j == image || matches!(s.shape, SurfaceShape::CoordinateBreak) {
                continue;
            }
            if s.fixed_semi_diameter.is_none()
                && s.aperture.is_none()
                && let Some(r) = self.clear_semi_diameters[j]
            {
                s.fixed_semi_diameter = Some(r);
            }
        }
        rs
    }

    /// Source for a normalised field point (module docs). `strength` is the
    /// irradiance normal to the beam (W/mm²) for an object at infinity and
    /// the intensity (W/sr) for a finite object.
    pub fn field_source(&self, field: FieldPoint, margin: f64, strength: f64) -> MkResult<Beam> {
        check_margin(margin)?;
        let t = &self.tracer;
        let pf = t.prepare_field(field)?;
        let rs = t.system();
        let pupil = *t.pupil();
        let chief = pf.aims[rs.primary_wavelength].chief;
        let (sc, sr) = self.scene.bounding_sphere();
        let what = format!("field ({}, {})", field.hx, field.hy);
        match pf.source {
            ObjectSource::Direction { dir } => {
                if !(pupil.z_ep.is_finite() && pupil.r_ep > 0.0) {
                    return Err(MkError::InvalidModel(
                        "collimated field source needs a finite entrance pupil".into(),
                    ));
                }
                let center = Vec3::new(chief[0], chief[1], pupil.z_ep);
                let radius = pupil.r_ep * (1.0 + margin);
                let cos = dir.z.abs();
                let spec = SourceSpec::CollimatedThrough {
                    direction: dir,
                    center,
                    normal: Vec3::Z,
                    radius,
                    start_distance: (center - sc).norm() + sr + radius + 1.0,
                    power: strength * PI * radius * radius * cos,
                };
                spec.validate()?;
                Ok(Beam {
                    spec,
                    reference_flux: strength * PI * pupil.r_ep * pupil.r_ep * cos,
                    description: format!(
                        "{what}: collimated beam filling the entrance pupil (r = {:.6} mm) with margin {margin}",
                        pupil.r_ep
                    ),
                })
            }
            ObjectSource::Point { point } => {
                let p = rs.surfaces[0].frame.to_global_point(point);
                let (axis, tan_ep) = if pupil.telecentric {
                    let axis = Vec3::new(chief[0], chief[1], 1.0).normalized();
                    (axis, pupil.u_marginal.abs())
                } else {
                    let aim = Vec3::new(chief[0], chief[1], pupil.z_ep);
                    let mut axis = (aim - p).normalized();
                    if axis.z < 0.0 {
                        // Virtual entrance pupil behind the object.
                        axis = -axis;
                    }
                    // Largest angle from the axis to the pupil rim.
                    let mut max_tan = 0.0f64;
                    for k in 0..360 {
                        let a = 2.0 * PI * k as f64 / 360.0;
                        let q = aim + Vec3::new(pupil.r_ep * a.cos(), pupil.r_ep * a.sin(), 0.0);
                        let mut v = (q - p).normalized();
                        if v.dot(axis) < 0.0 {
                            v = -v;
                        }
                        let c = v.dot(axis).clamp(-1.0, 1.0);
                        let s = ((1.0 - c) * (1.0 + c)).max(0.0).sqrt();
                        max_tan = max_tan.max(s / c);
                    }
                    (axis, max_tan)
                };
                if !(tan_ep.is_finite() && tan_ep > 0.0 && axis.is_finite()) {
                    return Err(MkError::InvalidModel(format!(
                        "{what}: entrance-pupil cone is undefined"
                    )));
                }
                let half = (tan_ep * (1.0 + margin)).atan();
                let spec = SourceSpec::Point {
                    position: p,
                    axis,
                    half_angle_deg: half.to_degrees(),
                    intensity: strength,
                };
                spec.validate()?;
                let a_ep = tan_ep.atan();
                Ok(Beam {
                    spec,
                    reference_flux: strength * 2.0 * PI * 2.0 * (0.5 * a_ep).sin().powi(2),
                    description: format!(
                        "{what}: point source filling the entrance-pupil cone (half-angle {:.6}°) with margin {margin}",
                        a_ep.to_degrees()
                    ),
                })
            }
        }
    }

    /// Source for the system's field `index` with its vignetting factors
    /// ignored (the beam always fills the whole pupil).
    pub fn system_field_source(&self, index: usize, margin: f64, strength: f64) -> MkResult<Beam> {
        let sys = self.resolved().system();
        if index >= sys.fields.fields.len() {
            return Err(MkError::InvalidModel(format!("no field {index}")));
        }
        let f = FieldPoint::system(sys, index);
        self.field_source(FieldPoint::normalized(f.hx, f.hy), margin, strength)
    }

    /// Out-of-field collimated source from angles `(θx, θy)` in degrees
    /// (direction `∝ (tan θx, tan θy, 1)`, CONVENTIONS §2) on a disk that
    /// covers the whole scene's projection along the beam (module docs).
    /// `irradiance` is normal to the beam.
    pub fn out_of_field_source(
        &self,
        angle_x_deg: f64,
        angle_y_deg: f64,
        margin: f64,
        irradiance: f64,
    ) -> MkResult<Beam> {
        check_margin(margin)?;
        if !(angle_x_deg.abs() < 90.0 && angle_y_deg.abs() < 90.0) {
            return Err(MkError::InvalidModel(
                "out-of-field source angles must lie within ±90°".into(),
            ));
        }
        let scene = &self.scene;
        let front = scene
            .sequence
            .first()
            .and_then(|&s| scene.surfaces.get(s).cloned().flatten())
            .ok_or_else(|| MkError::InvalidModel("scene has no entrance surface".into()))?;
        let d = Vec3::new(
            angle_x_deg.to_radians().tan(),
            angle_y_deg.to_radians().tan(),
            1.0,
        )
        .normalized();
        // Projections of every primitive's bounding-box corners on the plane
        // normal to the beam, relative to the front vertex.
        let c0 = front.frame.origin;
        let (t1, t2) = onb(d);
        let mut corners = Vec::with_capacity(8 * scene.primitives().len());
        for p in scene.primitives() {
            let b = p.aabb();
            for i in 0..8 {
                let q = Vec3::new(
                    if i & 1 == 0 { b.min.x } else { b.max.x },
                    if i & 2 == 0 { b.min.y } else { b.max.y },
                    if i & 4 == 0 { b.min.z } else { b.max.z },
                ) - c0;
                corners.push((q.dot(t1), q.dot(t2)));
            }
        }
        let (mut lo, mut hi) = (
            (f64::INFINITY, f64::INFINITY),
            (f64::NEG_INFINITY, f64::NEG_INFINITY),
        );
        for &(u, v) in &corners {
            lo = (lo.0.min(u), lo.1.min(v));
            hi = (hi.0.max(u), hi.1.max(v));
        }
        let (mu, mv) = (0.5 * (lo.0 + hi.0), 0.5 * (lo.1 + hi.1));
        let r_scene = corners
            .iter()
            .map(|&(u, v)| (u - mu).hypot(v - mv))
            .fold(0.0, f64::max);
        if !(r_scene.is_finite() && r_scene > 0.0) {
            return Err(MkError::InvalidModel("scene has no finite extent".into()));
        }
        let radius = r_scene * (1.0 + margin);
        let c = c0 + t1 * mu + t2 * mv;
        let (sc, sr) = scene.bounding_sphere();
        let spec = SourceSpec::Collimated {
            direction: d,
            center: c,
            radius,
            start_distance: (c - sc).norm() + sr + radius + 1.0,
            power: irradiance * PI * radius * radius,
        };
        spec.validate()?;
        let r_front = front.mechanical_semi_diameter;
        let axis = front.frame.to_global_dir(Vec3::Z);
        Ok(Beam {
            spec,
            reference_flux: irradiance * PI * r_front * r_front * d.dot(axis).abs(),
            description: format!(
                "out-of-field collimated source at ({angle_x_deg}°, {angle_y_deg}°) covering the whole scene (r = {radius:.6} mm)"
            ),
        })
    }
}

fn check_margin(margin: f64) -> MkResult<()> {
    if margin.is_finite() && margin >= 0.0 {
        Ok(())
    } else {
        Err(MkError::InvalidModel(
            "source margin must be finite and ≥ 0".into(),
        ))
    }
}
