//! MKLight explicit ghost-path tracing.
//!
//! Ghosts are the deterministic, purely specular paths that reach the image
//! after an even number of extra reflections at refracting interfaces (or
//! the sensor). This crate owns path class `Ghost(k)` of the solver
//! partition (ARCHITECTURE "Solvers and double counting"); the partition
//! rule and the path signature that the stray-light solver reuses are in
//! [`signature`]. Design notes: `docs/GHOST.md`.
//!
//! | module | contents |
//! |---|---|
//! | [`path`] | [`GhostPath`] (ordered extra reflections), expansion into [`Step`]s, [`enumerate`] with filters and budget |
//! | [`signature`] | [`InterfaceKind`], [`PathSignature`], [`classify`], [`GhostCoverage`]: the partition rule |
//! | [`model`] | [`GhostModel`]: tracer, settings, clipping apertures, inverse frame transforms |
//! | [`sequence`] | the general sequence tracer (forward/backward, s/p radiometry from the shared coating code) |
//! | [`paraxial`] | y–nu along unfolded ghost sequences: ghost focus, spot size, irradiance estimate, ranking |
//! | [`irradiance`] | real-ray ghost irradiance maps and ghost-to-signal ratios (rayon, deterministic, cancellable) |
//!
//! # Usage
//!
//! ```
//! use std::sync::Arc;
//! use mklight_core::{materials::MaterialLibrary, samples, JobContext, ResolvedSystem, Snapshot};
//! use mklight_ghost::{GhostModel, GhostSettings, ParaxialGhostSettings, IrradianceSettings};
//!
//! let snap = Snapshot::new(samples::double_gauss()).unwrap();
//! let sys = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
//! let model = GhostModel::from_system(Arc::new(sys), GhostSettings::default()).unwrap();
//! let ranked = mklight_ghost::paraxial_ghosts(&model, &ParaxialGhostSettings::default()).unwrap();
//! assert!(!ranked.ghosts.is_empty());
//! let settings = IrradianceSettings {
//!     pupil_samples: 16,
//!     map_bins: [32, 32],
//!     fields: vec![0],
//!     wavelengths: vec![1],
//!     ..Default::default()
//! };
//! let maps = mklight_ghost::ghost_irradiance(&model, &settings, &JobContext::new()).unwrap();
//! assert!(maps.primary.flux > 0.0);
//! ```

pub mod irradiance;
pub mod model;
pub mod paraxial;
pub mod path;
pub mod sequence;
pub mod signature;

pub use irradiance::{
    FieldIrradiance, FieldRef, GhostIrradiance, IrradianceMap, IrradianceSettings, PathIrradiance,
    ghost_irradiance,
};
pub use model::{Clip, ClipMode, GhostModel, GhostSettings};
pub use paraxial::{
    ParaxialGhost, ParaxialGhostReport, ParaxialGhostSettings, ParaxialTrace, paraxial_ghosts,
    real_ghost_focus,
};
pub use path::{
    Action, Direction, EnumeratedPath, EnumerationSettings, GhostPath, PathEnumeration, Step,
    enumerate,
};
pub use sequence::{SequenceFail, SequenceResult, SequenceStatus, StepRecord};
pub use signature::{
    Event, GhostCoverage, InterfaceKind, Owner, PathClass, PathEvent, PathSignature, classify,
    interface_kinds,
};
