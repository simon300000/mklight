//! MKLight engine facade, used by every front end (macOS app via
//! `mklight-ffi`, Python, CLI).
//!
//! * `project` — the `.mklight` file format, schema migrations, atomic save.
//! * `session` — one open document: project, material library, immutable
//!   snapshots per edit, LRU cache of resolved systems.
//! * `scheduler` — worker pool with cancellation, progress and supersession.
//! * `analysis` — the pluggable analysis registry and the built-in analyses.
//! * `backend` — compute-backend selection (CPU reference, GPU, `auto`).
//! * `gpu` — GPU backend providers (Metal, CPU-f32 emulation) with
//!   per-system precision qualification.
//! * `validation` — the engine-level model checks of `validate`.
//! * `engine` — ties these together.
//!
//! ```
//! use mklight_engine::{Engine, AnalysisRequest};
//! let engine = Engine::default();
//! let session = engine.new_session();
//! let job = session
//!     .submit_analysis(&AnalysisRequest::new("validate", serde_json::Value::Null))
//!     .unwrap();
//! engine.wait_job(job, None).unwrap();
//! let result = engine.take_result(job).unwrap();
//! assert_eq!(result.json["ok"], true);
//! ```

pub mod analysis;
pub mod backend;
pub mod buffer;
pub mod engine;
pub mod error;
pub mod fsutil;
pub mod gpu;
pub mod lru;
pub mod project;
pub mod scheduler;
pub mod session;
pub mod settings;
pub mod time;
pub mod validation;

pub use analysis::{
    Analysis, AnalysisInfo, AnalysisInput, AnalysisOutput, AnalysisResult, BackendReport,
};
pub use backend::{Backend, BackendInfo};
pub use buffer::{Buffer, BufferData, DType};
pub use engine::{Engine, EngineConfig};
pub use error::{EngineError, EngineResult};
pub use project::{PROJECT_SCHEMA_VERSION, Project};
pub use scheduler::{JobEvent, JobId, JobState, JobStatus};
pub use session::{AnalysisRequest, ParamEdit, Session};
