//! MKLight core: the single physical model shared by every solver.
//!
//! This crate contains *data and physics primitives only* — no solvers. See
//! `docs/ARCHITECTURE.md` and `docs/CONVENTIONS.md` at the repository root.

pub mod error;
pub mod job;
pub mod materials;
pub mod math;
pub mod memo;
pub mod model;
pub mod param;
pub mod resolved;
pub mod rng;
pub mod samples;
pub mod serde_util;
pub mod snapshot;
pub mod surface_math;

pub use error::{MkError, MkResult};
pub use job::{CancelToken, JobContext, Progress, RunInfo};
pub use math::{Mat3, Vec3};
pub use model::*;
pub use resolved::ResolvedSystem;
pub use snapshot::Snapshot;

/// Engine version reported in every `RunInfo` and project file.
pub const ENGINE_VERSION: &str = env!("CARGO_PKG_VERSION");
