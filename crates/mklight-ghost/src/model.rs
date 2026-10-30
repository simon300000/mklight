//! [`GhostModel`]: a sequential tracer plus everything the ghost solvers
//! share: interface classification, clipping apertures, the inverse
//! surface-to-surface transforms for backward travel, and sensor
//! reflectance.
//!
//! # Clipping
//!
//! Ghost light leaves the designed beam footprint, so the apertures that
//! clip it matter much more than for the primary path, where `Auto`
//! semi-diameters never clip. [`ClipMode::ClearApertures`] (the default)
//! clips at
//!
//! * fixed semi-diameters and explicit apertures as given (any surface);
//! * the stop at `max(paraxial stop radius, traced height)` (the value
//!   `fix_semi_diameters` freezes, so the primary beam is never vignetted);
//! * every other `Auto` surface that is physically present (not
//!   [`InterfaceKind::Passive`]) at its traced clear semi-diameter plus
//!   `margin_mm`.
//!
//! A passive surface (dummy, coordinate break, bare same-index plane) that
//! is neither the stop nor has an explicit aperture never clips: inserting a
//! reference plane leaves every ghost result unchanged (tested). The image
//! surface clips only with an explicit aperture; a sensor reflection happens
//! only inside the sensor rectangle when the model has a
//! [`Sensor`](mklight_core::Sensor). `FixedApertures` clips at the fixed
//! apertures only, plus an `Auto` stop at the same radius as above (the
//! reference trace aims every ray inside the stop; an overfilled ghost grid
//! must not let light around an unbounded stop). `None` disables clipping.
//!
//! The apertures used are part of the [`GhostCoverage`](crate::GhostCoverage)
//! of a run: a stray-light solver combined with it must clip at the same
//! apertures (CONVENTIONS §10).

use std::sync::Arc;

use mklight_core::math::Frame;
use mklight_core::{JobContext, MkError, MkResult, ResolvedSystem, SurfaceShape};
use mklight_trace::semidia::SemiDiameterSettings;
use mklight_trace::{IntersectSettings, SemiDiameters, SequentialTracer};
use serde::{Deserialize, Serialize};

use crate::path::{Action, Direction, GhostPath, Step};
use crate::signature::{GhostCoverage, InterfaceKind, interface_kinds};

/// Which apertures clip ghost (and primary) rays (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum ClipMode {
    None,
    FixedApertures,
    ClearApertures { margin_mm: f64 },
}

/// Settings shared by every ghost analysis of one model.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct GhostSettings {
    /// Unpolarised, angle-independent reflectance of the sensor. `None`: the
    /// sensor does not reflect and no path reflects at the image.
    pub sensor_reflectance: Option<f64>,
    pub clip: ClipMode,
    /// Treat a segment with a negative geometric length (an intersection
    /// behind the ray) as a lost ray. Physical ghost light cannot travel
    /// backwards in time, so this is on by default.
    pub reject_virtual: bool,
    pub intersect: IntersectSettings,
}

impl Default for GhostSettings {
    fn default() -> Self {
        GhostSettings {
            sensor_reflectance: None,
            clip: ClipMode::ClearApertures { margin_mm: 0.0 },
            reject_virtual: true,
            intersect: IntersectSettings::default(),
        }
    }
}

/// Clipping at one surface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Clip {
    None,
    /// The surface's own fixed semi-diameter or explicit aperture.
    Surface,
    /// A circular clear aperture of this radius.
    Radius {
        radius: f64,
    },
}

/// The clipping of every surface for `mode` (module docs), from the
/// interface kinds and the clear semi-diameters of the system. This is the
/// single definition of the apertures a ghost run clips at:
/// [`GhostModel::new`] uses it with `SequentialTracer::semi_diameters` at
/// default settings ([`GhostModel::sized`] and
/// [`GhostModel::with_semi_diameters`] with others), and a stray-light
/// solver that is combined with a ghost
/// run builds its clear apertures from it, so both partition at identical
/// apertures (CONVENTIONS §10, `GhostCoverage::same_apertures`).
pub fn clip_apertures(
    sys: &ResolvedSystem,
    kinds: &[InterfaceKind],
    semi: &SemiDiameters,
    mode: ClipMode,
) -> Vec<Clip> {
    let n = sys.num_surfaces();
    let stop_radius = semi.values[sys.stop].max(semi.traced[sys.stop]) * (1.0 + 1e-12);
    (0..n)
        .map(|i| {
            let s = &sys.surfaces[i];
            let has_fixed = s.aperture.is_some() || s.fixed_semi_diameter.is_some();
            let is_cb = matches!(s.shape, SurfaceShape::CoordinateBreak);
            if i == 0 || is_cb {
                return Clip::None;
            }
            if has_fixed {
                return match mode {
                    ClipMode::None => Clip::None,
                    _ => Clip::Surface,
                };
            }
            let is_stop = i == sys.stop;
            match mode {
                ClipMode::None => Clip::None,
                ClipMode::FixedApertures if is_stop => Clip::Radius {
                    radius: stop_radius,
                },
                ClipMode::FixedApertures => Clip::None,
                ClipMode::ClearApertures { .. } if is_stop => Clip::Radius {
                    radius: stop_radius,
                },
                // Physically absent surfaces (dummies, bare same-index
                // planes) and the image never clip on their own.
                ClipMode::ClearApertures { .. }
                    if i == n - 1 || kinds[i] == InterfaceKind::Passive =>
                {
                    Clip::None
                }
                ClipMode::ClearApertures { margin_mm } => Clip::Radius {
                    radius: semi.values[i] + margin_mm,
                },
            }
        })
        .collect()
}

/// The shared ghost-solver model (module docs). Immutable and `Sync`.
#[derive(Clone, Debug)]
pub struct GhostModel {
    tracer: Arc<SequentialTracer>,
    settings: GhostSettings,
    kinds: Vec<InterfaceKind>,
    clip: Vec<Clip>,
    /// `back[i]`: maps surface `i + 1`'s local frame into surface `i`'s
    /// (inverse of `surfaces[i].to_next`), in subtract-then-rotate form.
    back: Vec<Option<Frame>>,
    semi: SemiDiameters,
}

fn check_settings(settings: &GhostSettings) -> MkResult<()> {
    if let Some(r) = settings.sensor_reflectance
        && !(0.0..=1.0).contains(&r)
    {
        return Err(MkError::InvalidModel(
            "sensor reflectance must lie in [0, 1]".into(),
        ));
    }
    Ok(())
}

impl GhostModel {
    /// Sizes the clear apertures with the default
    /// [`SemiDiameterSettings`] (module docs).
    pub fn new(tracer: Arc<SequentialTracer>, settings: GhostSettings) -> MkResult<Self> {
        Self::sized(
            tracer,
            settings,
            &SemiDiameterSettings::default(),
            &JobContext::new(),
        )
    }

    /// As [`GhostModel::new`], sizing the clear apertures with `sizing`,
    /// with cancellation and progress (`SequentialTracer::semi_diameters_with`).
    pub fn sized(
        tracer: Arc<SequentialTracer>,
        settings: GhostSettings,
        sizing: &SemiDiameterSettings,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        check_settings(&settings)?;
        let semi = tracer.semi_diameters_with(sizing, ctx)?;
        Self::with_semi_diameters(tracer, settings, semi)
    }

    /// With clear apertures sized elsewhere by the same tracer, e.g. a
    /// stray-light scene's (`StraySystem::semi_diameters`). Both solvers
    /// then clip at bit-identical radii whatever the sizing settings, which
    /// partitioning requires (CONVENTIONS §10: the apertures are compared
    /// exactly, so even a rounding difference of an independent sizing
    /// fails `GhostCoverage::same_apertures`).
    pub fn with_semi_diameters(
        tracer: Arc<SequentialTracer>,
        settings: GhostSettings,
        semi: SemiDiameters,
    ) -> MkResult<Self> {
        check_settings(&settings)?;
        let sys = tracer.system();
        let n = sys.num_surfaces();
        if semi.values.len() != n || semi.traced.len() != n || semi.sources.len() != n {
            return Err(MkError::InvalidParam(format!(
                "semi-diameters for {} surfaces given for a system of {n}",
                semi.values.len()
            )));
        }
        let kinds = interface_kinds(sys);
        let clip = clip_apertures(sys, &kinds, &semi, settings.clip);
        // Inverse transforms. With to_next = (o, R) (next frame in this
        // one: p_i = o + R p_{i+1}), the inverse in the same
        // subtract-then-rotate form as the forward map is
        // p_i = R (p_{i+1} − q) with q = −Rᵀ o, the vertex of surface i
        // expressed in surface i+1's frame. As a `Frame` whose
        // `to_local_point(p) = rotᵀ (p − origin)`: origin = q, rot = Rᵀ.
        let back = sys
            .surfaces
            .iter()
            .map(|s| {
                s.to_next.map(|f| Frame {
                    origin: f.to_local_point(mklight_core::Vec3::ZERO),
                    rot: f.rot.transpose(),
                })
            })
            .collect();
        Ok(GhostModel {
            tracer,
            settings,
            kinds,
            clip,
            back,
            semi,
        })
    }

    /// Build the tracer with default settings as well.
    pub fn from_system(sys: Arc<ResolvedSystem>, settings: GhostSettings) -> MkResult<Self> {
        Self::new(Arc::new(SequentialTracer::new(sys)?), settings)
    }

    pub fn tracer(&self) -> &SequentialTracer {
        &self.tracer
    }
    pub fn system(&self) -> &ResolvedSystem {
        self.tracer.system()
    }
    pub fn settings(&self) -> &GhostSettings {
        &self.settings
    }
    pub fn interface_kinds(&self) -> &[InterfaceKind] {
        &self.kinds
    }
    pub fn clip(&self, surface: usize) -> Clip {
        self.clip[surface]
    }
    /// Clipping of every surface (module docs).
    pub fn clips(&self) -> &[Clip] {
        &self.clip
    }
    /// Clear semi-diameters of the underlying tracer.
    pub fn semi_diameters(&self) -> &SemiDiameters {
        &self.semi
    }
    pub(crate) fn back_frame(&self, surface: usize) -> Option<&Frame> {
        self.back[surface].as_ref()
    }

    /// Radius that bounds the clipping aperture of a surface (`None`: no
    /// clipping), for the paraxial vignetting estimate.
    pub fn clip_radius(&self, surface: usize) -> Option<f64> {
        let s = &self.system().surfaces[surface];
        match self.clip[surface] {
            Clip::None => None,
            Clip::Radius { radius } => Some(radius),
            Clip::Surface => s
                .aperture
                .as_ref()
                .map(|a| a.bounding_radius())
                .or(s.fixed_semi_diameter),
        }
    }

    /// Does a local point pass the clipping aperture of a surface?
    #[inline]
    pub fn passes(&self, surface: usize, x: f64, y: f64) -> bool {
        match self.clip[surface] {
            Clip::None => true,
            Clip::Surface => self.system().surfaces[surface].passes_aperture(x, y),
            Clip::Radius { radius } => x.mul_add(x, y * y) <= radius * radius,
        }
    }

    /// May this surface carry a ghost reflection? Refracting interfaces,
    /// and the sensor when it has a reflectance.
    pub fn is_reflector(&self, surface: usize) -> bool {
        match self.kinds.get(surface) {
            Some(InterfaceKind::Refractive) => true,
            Some(InterfaceKind::Image) => self.settings.sensor_reflectance.is_some(),
            _ => false,
        }
    }

    /// Unpolarised normal-incidence reflectance of a ghost reflection at
    /// `surface` for light arriving in `direction` (0 for non-reflectors).
    pub fn normal_reflectance(&self, surface: usize, direction: Direction, wl: usize) -> f64 {
        if !self.is_reflector(surface) {
            return 0.0;
        }
        if self.kinds[surface] == InterfaceKind::Image {
            return self.settings.sensor_reflectance.unwrap_or(0.0);
        }
        let sys = self.system();
        let s = &sys.surfaces[surface];
        let (n_in, n_out) = self.media(surface, direction, wl);
        s.coating
            .rt(sys.wavelengths_um[wl], n_in, n_out, 1.0, false)
            .r_unpolarized()
    }

    /// `(incident medium, medium on the far side)` of a surface for light
    /// arriving in `direction`: forward, `(n_before, n_after)`; backward,
    /// `(n_after, n_before)`.
    #[inline]
    pub fn media(&self, surface: usize, direction: Direction, wl: usize) -> (f64, f64) {
        let sys = self.system();
        let (a, b) = (sys.n_before(surface, wl), sys.surfaces[surface].n_after[wl]);
        match direction {
            Direction::Forward => (a, b),
            Direction::Backward => (b, a),
        }
    }

    /// The coverage record of a run that traced exactly `paths` with this
    /// model (interface kinds and clipping apertures included).
    pub fn coverage(&self, paths: impl IntoIterator<Item = GhostPath>) -> GhostCoverage {
        GhostCoverage {
            interfaces: self.kinds.clone(),
            apertures: self.clip.clone(),
            paths: paths.into_iter().collect(),
        }
    }

    /// Steps of a path, after checking that every reflection is at a
    /// reflector of this model.
    pub fn check_path(&self, path: &GhostPath) -> MkResult<Vec<Step>> {
        let steps = path.steps(self.system().num_surfaces())?;
        for st in &steps {
            if let Some(order) = st.order {
                match &self.system().surfaces[st.surface as usize].shape {
                    SurfaceShape::Diffractive { phase, .. } if order != phase.trace_order => {}
                    _ => {
                        return Err(MkError::InvalidModel(
                            "order override must name a non-selected DOE order".into(),
                        ));
                    }
                }
            }
            if st.action == Action::Reflect && !self.is_reflector(st.surface as usize) {
                return Err(MkError::InvalidModel(format!(
                    "ghost path {:?}: surface {} cannot reflect ({:?})",
                    path.reflections, st.surface, self.kinds[st.surface as usize]
                )));
            }
        }
        Ok(steps)
    }
}
