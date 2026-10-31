//! MKLight geometric image-quality analyses on top of the CPU f64 reference
//! tracer (`mklight-trace`). Design notes: `docs/ANALYSIS.md`.
//!
//! Every analysis is a pure function of a [`SequentialTracer`] (which owns
//! the `ResolvedSystem` and its snapshot) and a serde settings struct with
//! `Default`, returning a serde result that carries a `RunInfo` (snapshot
//! hash, settings, backend `cpu-f64` or the backend of a reduced ray batch,
//! seed where random, warnings). Long computations have a `*_job` variant
//! taking a `JobContext` for cancellation and monotone progress. Results are deterministic: fixed
//! ordering, compensated sums, Philox for random pupil sampling.
//!
//! | module | analyses |
//! |---|---|
//! | [`spot`] | spot diagrams (square, hexapolar, random, dithered sampling), centroid / RMS / GEO about centroid and chief, polychromatic weighting, Airy radius, vignetted fraction, through-focus; the public ray-batch reduction layer (`RayBatch`, `FieldRays`, `*_from_rays`) shared with other backends |
//! | [`fans`] | transverse ray fans and OPD fans |
//! | [`curvature`] | field curvature (generalised Coddington along the real chief ray, close-ray cross-check), distortion (paraxial, `f·tan θ`, `f·θ`), grid distortion |
//! | [`seidel`] | Seidel sums per surface, Petzval, wavefront coefficients, third-order ray aberrations, longitudinal aberration, chromatic focal shift, lateral colour |
//! | [`illumination`] | relative illumination (projected solid angle of the exit pupil seen from the image point, real aiming), vignetting vs field, surface footprints |
//! | [`energy`] | geometric encircled / ensquared energy, geometric MTF, best focus (Brent) |
//! | [`layout`] | the viewport scene in global coordinates, with compact buffers |
//! | [`system_data`] | first-order report table |
//! | [`util`] | selections, pupil patterns, weighted moments, Brent, Gauss–Legendre, non-finite-safe serde (`nf`) |
//!
//! Non-finite result numbers serialise as `"inf"` / `"-inf"` (core
//! `serde_util` convention) and NaN ("undefined", e.g. a vignetted field) as
//! `null` ([`util::nf`]).

// `!(x > 0.0)` deliberately rejects NaN as well; keep that form.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{MkResult, OpticalSystem, ResolvedSystem, Snapshot};
use mklight_trace::SequentialTracer;

pub mod curvature;
pub mod energy;
pub mod fans;
pub mod field_maps;
pub mod footprint_overlay;
pub mod glass_map;
pub mod illumination;
pub mod layout;
pub mod seidel;
pub mod spectrum;
pub mod spot;
pub mod system_data;
pub mod util;

/// Convenience for tests, tools and examples: snapshot, resolve against the
/// built-in material library and build the reference tracer.
pub fn tracer_for(system: OpticalSystem) -> MkResult<SequentialTracer> {
    let snap = Snapshot::new(system)?;
    let sys = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin()))?;
    SequentialTracer::new(Arc::new(sys))
}

/// Resident GPU adapter seam (enabled by the solver package).
pub mod gpu;
