//! A monotonic clock: `std::time::Instant` natively; on the web
//! (wasm32-unknown-unknown has no clock in `std`) `performance.now()`.
#[cfg(not(target_arch = "wasm32"))]
pub use std::time::Instant;
#[cfg(target_arch = "wasm32")]
pub use web_time::Instant;
