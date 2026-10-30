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
//! clips at every surface `1 … N−2`:
//!
//! * fixed semi-diameters and explicit apertures as given;
//! * the stop at `max(paraxial stop radius, traced height)` (the value
//!   `fix_semi_diameters` freezes, so the primary beam is never vignetted);
//! * other `Auto` surfaces at their traced clear semi-diameter plus
//!   `margin_mm`.
//!
//! Coordinate breaks never clip. The image surface clips only with an
//! explicit aperture; a sensor reflection happens only inside the sensor
//! rectangle when the model has a [`Sensor`](mklight_core::Sensor).
//! `FixedApertures` clips like the reference trace (fixed apertures only);
//! `None` disables clipping.

use std::sync::Arc;

use mklight_core::math::Frame;
use mklight_core::{MkError, MkResult, ResolvedSystem, SurfaceShape};
use mklight_trace::semidia::SemiDiameterSettings;
use mklight_trace::{IntersectSettings, SemiDiameters, SequentialTracer};
use serde::{Deserialize, Serialize};

use crate::path::{Action, Direction, GhostPath, Step};
use crate::signature::{InterfaceKind, interface_kinds};

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

impl GhostModel {
    pub fn new(tracer: Arc<SequentialTracer>, settings: GhostSettings) -> MkResult<Self> {
        if let Some(r) = settings.sensor_reflectance
            && !(0.0..=1.0).contains(&r)
        {
            return Err(MkError::InvalidModel(
                "sensor reflectance must lie in [0, 1]".into(),
            ));
        }
        let sys = tracer.system();
        let n = sys.num_surfaces();
        let kinds = interface_kinds(sys);
        let semi = tracer.semi_diameters(&SemiDiameterSettings::default());
        let clip = (0..n)
            .map(|i| {
                let s = &sys.surfaces[i];
                let has_fixed = s.aperture.is_some() || s.fixed_semi_diameter.is_some();
                let is_cb = matches!(s.shape, SurfaceShape::CoordinateBreak);
                if i == 0 || is_cb {
                    return Clip::None;
                }
                match settings.clip {
                    ClipMode::None => Clip::None,
                    ClipMode::FixedApertures => Clip::Surface,
                    ClipMode::ClearApertures { margin_mm } => {
                        if has_fixed {
                            Clip::Surface
                        } else if i == n - 1 {
                            Clip::None
                        } else if i == sys.stop {
                            Clip::Radius {
                                radius: semi.values[i].max(semi.traced[i]) * (1.0 + 1e-12),
                            }
                        } else {
                            Clip::Radius {
                                radius: semi.values[i] + margin_mm,
                            }
                        }
                    }
                }
            })
            .collect();
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

    /// Steps of a path, after checking that every reflection is at a
    /// reflector of this model.
    pub fn check_path(&self, path: &GhostPath) -> MkResult<Vec<Step>> {
        let steps = path.steps(self.system().num_surfaces())?;
        for st in &steps {
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
