//! Automatic clear semi-diameters.
//!
//! For every surface whose semi-diameter is `Auto`, the clear semi-diameter
//! is the largest radial ray height `√(x² + y²)` in the surface's local frame
//! over all system fields (with their vignetting factors), all system
//! wavelengths, the chief ray and the pupil edge sampled at `azimuths`
//! equally spaced angles. Rays clipped by fixed apertures upstream contribute
//! only to the surfaces they reach. The stop's semi-diameter is the stop
//! radius of the aperture specification ([`crate::PupilGeometry::r_stop`],
//! CONVENTIONS §3); fixed semi-diameters and
//! explicit apertures report their own (bounding) radius; coordinate breaks
//! report 0. The object surface reports the largest object height (0 for an
//! object at infinity).
//!
//! The raw traced maxima are kept for every surface in
//! [`SemiDiameters::traced`]. With ray aiming off, pupil aberration makes real
//! rays reach the stop slightly above or below that radius, so
//! [`fix_semi_diameters`] freezes the stop at the larger of the two values:
//! freezing never vignettes the current design.
//!
//! [`SequentialTracer::semi_diameters_with`] is the cancellable variant for
//! jobs and solvers: it checks the [`JobContext`] before every (field,
//! wavelength) task and every [`CANCEL_CHUNK`] rays within one, and reports
//! monotone progress over the traced rays. The result does not depend on
//! the thread count (maxima and counts are exact reductions).

use std::sync::atomic::{AtomicUsize, Ordering};

use mklight_core::{JobContext, MkResult, OpticalSystem, SemiDiameter, SurfaceShape};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pupil::ring;
use crate::trace::{SurfaceRecord, TraceOptions, trace_rays_recorded};
use crate::tracer::{RayRequest, SequentialTracer};

/// Rays traced between two cancellation checks (and progress reports) of
/// [`SequentialTracer::semi_diameters_with`]: one packet batch.
pub const CANCEL_CHUNK: usize = crate::batch::CHUNK;

/// Settings of [`SequentialTracer::semi_diameters`]. In JSON every field is
/// optional (defaults below) and unknown keys are rejected, so a misspelt
/// key is reported instead of silently ignored.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SemiDiameterSettings {
    /// Pupil-edge samples around the azimuth (at least 4 are used).
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
        self.semi_diameters_with(settings, &JobContext::new())
            .expect("a fresh JobContext is never cancelled")
    }

    /// As [`SequentialTracer::semi_diameters`], with cancellation (checked
    /// before every (field, wavelength) task and every [`CANCEL_CHUNK`] rays)
    /// and monotone progress (the fraction of rays traced).
    pub fn semi_diameters_with(
        &self,
        settings: &SemiDiameterSettings,
        ctx: &JobContext,
    ) -> MkResult<SemiDiameters> {
        const STAGE: &str = "semi-diameters";
        ctx.check()?;
        ctx.report(0.0, STAGE);
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
        let done = AtomicUsize::new(0);
        let total = (tasks.len() * pupil.len()).max(1);
        let advance = |rays: usize| {
            let k = done.fetch_add(rays, Ordering::Relaxed) + rays;
            ctx.report(k as f64 / total as f64, STAGE);
        };
        let (maxes, failed) = tasks
            .par_iter()
            .map(|&(f, w)| -> MkResult<(Vec<f64>, usize)> {
                ctx.check()?;
                let field = self.field(f);
                let mut m = vec![0.0f64; n];
                let mut failed = 0usize;
                let mut rec: Vec<Vec<SurfaceRecord>> = Vec::new();
                for (c, chunk) in pupil.chunks(CANCEL_CHUNK).enumerate() {
                    if c > 0 {
                        ctx.check()?;
                        advance(CANCEL_CHUNK);
                    }
                    // Aimed and traced together, with records (the same
                    // results as `trace_with`, one by one).
                    let requests: Vec<RayRequest> = chunk.iter().map(|&p| (field, w, p)).collect();
                    let generated = self.generate_many(&requests);
                    failed += generated.iter().filter(|g| g.is_err()).count();
                    let starts: Vec<_> = generated.iter().flatten().map(|g| g.ray).collect();
                    rec.resize_with(starts.len(), || Vec::with_capacity(n));
                    let traced =
                        trace_rays_recorded(sys, &starts, &[w], &opts, &mut rec[..starts.len()]);
                    for (r, rec) in traced.iter().zip(&rec) {
                        if !r.status.is_ok() {
                            failed += 1;
                        }
                        for (j, s) in rec.iter().enumerate() {
                            m[j] = m[j].max(s.point.x.hypot(s.point.y));
                        }
                    }
                }
                advance(pupil.len() - (pupil.len() - 1) / CANCEL_CHUNK * CANCEL_CHUNK);
                Ok((m, failed))
            })
            .try_reduce(
                || (vec![0.0; n], 0),
                |(a, fa), (b, fb)| {
                    Ok((a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect(), fa + fb))
                },
            )?;
        ctx.check()?;
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
        ctx.report(1.0, STAGE);
        Ok(SemiDiameters {
            values,
            traced: maxes,
            sources,
            failed_rays: failed,
        })
    }
}

/// A copy of `system` with every `Auto` semi-diameter (except on coordinate
/// breaks, and on the object surface at infinity) replaced by
/// `Fixed { value + margin }`, e.g. to freeze apertures before tolerancing.
/// The stop uses `max(stop radius, traced height)` (module docs).
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
