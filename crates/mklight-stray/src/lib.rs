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
//!
//! # Integration points (later phase)
//!
//! * Clear semi-diameters of automatic surfaces, including the stop, come
//!   from the sequential tracer through `SceneOptions::clear_semi_diameters`.
//! * Sources aimed at the entrance pupil (paraxial tracer) instead of the
//!   first surface's mechanical aperture.
//! * A hardware ray-tracing [`geom::RayCaster`] (Metal acceleration
//!   structures); it also needs the BSDF's TIS table and lobe parameters.
//! * Rays carry unpolarised (s/p-averaged) flux, while the ghost solver may
//!   use per-polarisation `R`.
//!
//! Units follow CONVENTIONS §1: mm, µm, W, W/mm².

pub mod bsdf;
pub mod geom;
pub mod integrate;
pub mod path;
pub mod scene;
pub mod source;
pub mod tally;
pub mod tracer;

pub use bsdf::Bsdf;
pub use path::{ClassMask, History, PathClass};
pub use scene::{Scene, SceneBuilder, SceneOptions};
pub use source::SourceSpec;
pub use tally::{DetectorResult, EnergyReport, Sink};
pub use tracer::{SplitMode, StrayResult, StraySettings, run, run_with};
