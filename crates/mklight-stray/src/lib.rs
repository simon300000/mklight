//! MKLight non-sequential Monte-Carlo stray-light transport.
//!
//! The pieces, bottom-up:
//!
//! * [`bsdf`] — Lambertian, Harvey–Shack ABg and Gaussian-microfacet surface
//!   scatter: evaluation, exact importance sampling, adaptive total integrated
//!   scatter (TIS), energy normalisation (`TIS ≤ 1`).
//! * [`geom`] — robust ray intersection with bounded sag patches (planes,
//!   conics in closed form, even aspheres by bracketing), cylinders, placed
//!   primitives, and the [`geom::RayCaster`] spatial-search interface with a
//!   CPU BVH.
//! * [`scene`] — the non-sequential scene built from an `OpticalSystem` and
//!   its `ResolvedSystem`: optical interfaces bounded by their apertures
//!   (holes, masks), opaque non-clear zones, merged double-pass surfaces,
//!   lens edges and flats, aperture-stop plates and rings, barrels, baffles
//!   and the detector.
//! * [`source`] — collimated, point and extended Lambertian sources.
//! * [`path`] — path histories and the disjoint path classes shared with the
//!   ghost solver (ARCHITECTURE "Solvers and double counting").
//! * [`tally`] — per-class detector maps with Monte-Carlo standard errors,
//!   and the exact energy ledger.
//! * [`tracer`] — the rayon-parallel, thread-count-independent path tracer
//!   with rare-branch forcing, Russian roulette, splitting and weight
//!   windows.
//! * [`system`] — the connection to the sequential reference tracer
//!   (`mklight-trace`): scenes sized with its automatic clear
//!   semi-diameters (stop included), field sources aimed at the paraxial
//!   entrance pupil, out-of-field (sun) sources that overfill the front
//!   aperture.
//! * [`reference`] — the deterministic sequential expectation of the
//!   `Primary` class (flux, centroid, RMS radius) for cross-validation.
//! * [`analysis`] — the engine entry point: settings, per-class maps with
//!   standard errors, totals, top path signatures, `RunInfo`.
//!
//! The partition of detected light between the sequential, ghost and
//! stray solvers is specified in `docs/STRAY.md` and [`path`].
//!
//! # Integration points (later phase)
//!
//! * A hardware ray-tracing [`geom::RayCaster`] (Metal acceleration
//!   structures); it also needs the BSDF's TIS table and lobe parameters.
//! * Rays carry unpolarised (s/p-averaged) flux, while the ghost solver may
//!   use per-polarisation `R`.
//!
//! Units follow CONVENTIONS §1: mm, µm, W, W/mm².

pub mod analysis;
pub mod bsdf;
pub mod geom;
pub mod integrate;
pub mod path;
pub mod reference;
pub mod scene;
pub mod source;
pub mod system;
pub mod tally;
pub mod tracer;

pub use analysis::{
    ClassMap, ClassTotal, StrayAnalysisResult, StrayAnalysisSettings, StraySource,
    TransportOptions, analyze, analyze_system,
};
pub use reference::{PrimaryReference, primary_reference};
pub use system::{Beam, DetectorShape, StraySystem, SystemOptions};

pub use bsdf::Bsdf;
pub use path::{ClassMask, EventKind, History, PathClass, Signature, SurfaceOp};
pub use scene::{Scene, SceneBuilder, SceneOptions};
pub use source::SourceSpec;
pub use tally::{DetectorResult, EnergyReport, ImageMoments, SignatureFlux, Sink};
pub use tracer::{SplitMode, StrayResult, StraySettings, run, run_with};
