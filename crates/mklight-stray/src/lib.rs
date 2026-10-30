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
//!   its `ResolvedSystem`: optical interfaces, lens edges and flats,
//!   aperture-stop plates, barrels, baffles and the detector.
//! * [`source`] — collimated, point and extended Lambertian sources.
//! * [`path`] — path histories and the disjoint path classes shared with the
//!   ghost solver (ARCHITECTURE "Solvers and double counting").
//! * [`tally`] — per-class detector maps and the exact energy ledger.
//! * [`tracer`] — the rayon-parallel, thread-count-independent path tracer
//!   with Russian roulette and splitting.
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
