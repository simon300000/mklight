//! Automatic clear semi-diameters.
//!
//! For every surface whose semi-diameter is `Auto`, the clear semi-diameter
//! is the largest radial ray height `√(x² + y²)` in the surface's local frame
//! over all system fields (with their vignetting factors), all system
//! wavelengths, the chief ray and the pupil edge sampled at `azimuths`
//! equally spaced angles. Rays clipped by fixed apertures upstream contribute
//! only to the surfaces they reach. The stop's semi-diameter is the paraxial
//! stop radius from the aperture specification; fixed semi-diameters and
//! explicit apertures report their own (bounding) radius; coordinate breaks
//! report 0. The object surface reports the largest object height (0 for an
//! object at infinity).
//!
//! The raw traced maxima are kept for every surface in
//! [`SemiDiameters::traced`]. With ray aiming off, pupil aberration makes real
//! rays reach the stop slightly above or below its paraxial radius, so
//! [`fix_semi_diameters`] freezes the stop at the larger of the two values:
//! freezing never vignettes the current design.

use mklight_core::{OpticalSystem, SemiDiameter, SurfaceShape};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pupil::ring;
use crate::trace::{SurfaceRecord, TraceOptions};
use crate::tracer::SequentialTracer;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemiDiameterSettings {
    /// Pupil-edge samples around the azimuth.
    pub azimuths: usize,
    /// Let fixed apertures clip rays while sizing.
    pub clip: bool,
}

impl Default for SemiDiameterSettings {
    fn default() -> Self {
        SemiDiameterSettings {
            azimuths: 64,
            clip: true,
        }
    }
}

/// Where a semi-diameter value came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SemiDiameterSource {
    /// Computed from traced rays.
    Traced,
    /// The stop, sized by the aperture specification.
    Stop,
    /// A fixed semi-diameter or explicit aperture.
    Fixed,
    /// Coordinate break (no geometry).
    None,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SemiDiameters {
    /// Per-surface semi-diameter (mm).
    pub values: Vec<f64>,
    /// Largest traced ray height per surface (mm), whatever the source.
    pub traced: Vec<f64>,
    pub sources: Vec<SemiDiameterSource>,
    /// Sample rays that failed before the image (counted, not fatal).
    pub failed_rays: usize,
}

impl SequentialTracer {
    /// Compute clear semi-diameters (module docs).
    pub fn semi_diameters(&self, settings: &SemiDiameterSettings) -> SemiDiameters {
        let sys = self.system();
        let n = sys.surfaces.len();
        let mut pupil = vec![[0.0, 0.0]];
        pupil.extend(ring(settings.azimuths.max(4), 1.0));
        let opts = TraceOptions {
            clip: settings.clip,
            ..self.options()
        };
        let nw = self.num_wavelengths();
        let tasks: Vec<(usize, usize)> = (0..self.fields().len())
            .flat_map(|f| (0..nw).map(move |w| (f, w)))
            .collect();
        let (maxes, failed) = tasks
            .par_iter()
            .map(|&(f, w)| {
                let field = self.field(f);
                let mut m = vec![0.0f64; n];
                let mut failed = 0usize;
                let mut rec: Vec<SurfaceRecord> = Vec::with_capacity(n);
                for p in &pupil {
                    let r = self.trace_with(field, w, p[0], p[1], &opts, Some(&mut rec));
                    if !r.status.is_ok() {
                        failed += 1;
                    }
                    for (j, s) in rec.iter().enumerate() {
                        m[j] = m[j].max(s.point.x.hypot(s.point.y));
                    }
                }
                (m, failed)
            })
            .reduce(
                || (vec![0.0; n], 0),
                |(a, fa), (b, fb)| (a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect(), fa + fb),
            );
        let mut values = Vec::with_capacity(n);
        let mut sources = Vec::with_capacity(n);
        for (j, s) in sys.surfaces.iter().enumerate() {
            let fixed = s
                .aperture
                .as_ref()
                .map(|a| a.bounding_radius())
                .or(s.fixed_semi_diameter);
            let (v, src) = if let Some(v) = fixed {
                (v, SemiDiameterSource::Fixed)
            } else if matches!(s.shape, SurfaceShape::CoordinateBreak) {
                (0.0, SemiDiameterSource::None)
            } else if j == sys.stop {
                (self.pupil().r_stop, SemiDiameterSource::Stop)
            } else if j == 0 && sys.object_at_infinity {
                (0.0, SemiDiameterSource::None)
            } else {
                (maxes[j], SemiDiameterSource::Traced)
            };
            values.push(v);
            sources.push(src);
        }
        SemiDiameters {
            values,
            traced: maxes,
            sources,
            failed_rays: failed,
        }
    }
}

/// A copy of `system` with every `Auto` semi-diameter (except on coordinate
/// breaks, and on the object surface at infinity) replaced by
/// `Fixed { value + margin }`, e.g. to freeze apertures before tolerancing.
/// The stop uses `max(paraxial radius, traced height)` (module docs).
pub fn fix_semi_diameters(
    system: &OpticalSystem,
    sd: &SemiDiameters,
    margin_mm: f64,
) -> OpticalSystem {
    let mut out = system.clone();
    for (j, s) in out.surfaces.iter_mut().enumerate() {
        let v = match sd.sources[j] {
            SemiDiameterSource::Traced => sd.values[j],
            SemiDiameterSource::Stop => sd.values[j].max(sd.traced[j]),
            _ => continue,
        };
        if matches!(s.semi_diameter, SemiDiameter::Auto) && v + margin_mm > 0.0 {
            s.semi_diameter = SemiDiameter::Fixed {
                value: v + margin_mm,
            };
        }
    }
    out
}
