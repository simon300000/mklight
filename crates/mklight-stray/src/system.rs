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
//! The same values define the *clipped sequential model*
//! ([`StraySystem::clipped_system`]): the resolved system with every
//! automatic semi-diameter frozen to the scene's clear aperture. It is what
//! the `Primary` class of the Monte Carlo must reproduce
//! ([`crate::reference`]).
//!
//! # Sources
//!
//! All margins are relative overfills of the entrance-pupil radius (or of
//! the front aperture), so the stop and the mounts, not the source, limit
//! the beam.
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
//!   `(θx, θy)` (for example the sun outside the field of view), on a disk
//!   normal to the beam centred on the first interface's vertex, with radius
//!   `(1 + margin)` times the largest mechanical radius in the scene
//!   (surfaces, barrels, baffles). Any ray that can enter the lens through
//!   its front opening crosses that disk.
//!
//! Each source comes with a *reference flux* ([`Beam::reference_flux`]):
//! the flux through the entrance pupil (field sources, the margin
//! excluded), or through the first interface's mechanical aperture
//! (out-of-field sources), the usual normalisation of stray-light ratios.

use std::f64::consts::PI;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use mklight_core::error::{MkError, MkResult};
use mklight_core::math::Vec3;
use mklight_core::model::{MechanicalPart, SurfaceShape};
use mklight_core::resolved::ResolvedSystem;
use mklight_trace::{
    FieldPoint, ObjectSource, SemiDiameterSettings, SemiDiameterSource, SemiDiameters,
    SequentialTracer,
};

use crate::scene::{Scene, SceneOptions};
use crate::source::SourceSpec;

/// Options of [`StraySystem::new`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
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
}

impl StraySystem {
    /// Builds the tracer, the semi-diameters and the scene.
    pub fn new(rs: Arc<ResolvedSystem>, opts: &SystemOptions) -> MkResult<StraySystem> {
        Self::with_tracer(SequentialTracer::new(rs)?, opts)
    }

    /// As [`StraySystem::new`] with an existing tracer (custom aiming).
    pub fn with_tracer(tracer: SequentialTracer, opts: &SystemOptions) -> MkResult<StraySystem> {
        if !(opts.detector_margin.is_finite() && opts.detector_margin >= 0.0) {
            return Err(MkError::InvalidModel(
                "detector margin must be finite and ≥ 0".into(),
            ));
        }
        let sd = tracer.semi_diameters(&opts.semi_diameters);
        let rs = tracer.system();
        let n = rs.num_surfaces();
        let image = n - 1;
        let has_sensor = rs.system().sensor.is_some();
        let mut clear = vec![None; n];
        for (j, c) in clear.iter_mut().enumerate() {
            let user = opts
                .scene
                .clear_semi_diameters
                .get(j)
                .copied()
                .flatten()
                .filter(|r| r.is_finite() && *r > 0.0);
            let traced = match sd.sources[j] {
                SemiDiameterSource::Traced if j == image => {
                    Some(sd.values[j] * (1.0 + opts.detector_margin))
                }
                SemiDiameterSource::Traced => Some(sd.values[j]),
                SemiDiameterSource::Stop => Some(sd.values[j].max(sd.traced[j])),
                SemiDiameterSource::Fixed | SemiDiameterSource::None => None,
            };
            *c = user.or(traced.filter(|r| r.is_finite() && *r > 0.0));
        }
        let mut sopts = opts.scene.clone();
        sopts.clear_semi_diameters = clear.clone();
        let scene = Scene::from_system(rs, &sopts)?;
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
        Ok(StraySystem {
            tracer,
            semi_diameters: sd,
            clear_semi_diameters: clear,
            scene,
            detector_shape,
        })
    }

    pub fn tracer(&self) -> &SequentialTracer {
        &self.tracer
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
    /// (direction `∝ (tan θx, tan θy, 1)`, CONVENTIONS §2) that overfills the
    /// front aperture (module docs). `irradiance` is normal to the beam.
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
        let mut r_max = scene
            .surfaces
            .iter()
            .flatten()
            .map(|g| g.mechanical_semi_diameter)
            .fold(0.0, f64::max);
        let include = scene.options.as_ref().is_none_or(|o| o.include_mechanics);
        if include {
            for part in &self.resolved().system().mechanics {
                r_max = r_max.max(match *part {
                    MechanicalPart::Barrel { inner_radius, .. } => inner_radius,
                    MechanicalPart::AnnularBaffle { outer_radius, .. } => outer_radius,
                });
            }
        }
        let d = Vec3::new(
            angle_x_deg.to_radians().tan(),
            angle_y_deg.to_radians().tan(),
            1.0,
        )
        .normalized();
        let radius = r_max * (1.0 + margin);
        let c = front.frame.origin;
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
                "out-of-field collimated source at ({angle_x_deg}°, {angle_y_deg}°) overfilling the front aperture (r = {radius:.6} mm)"
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
