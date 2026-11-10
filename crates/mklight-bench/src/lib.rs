//! Measurement and qualification of the engine: the harnesses in `tests/`
//! (ignored, run by hand on a real Metal device, next to the two regression
//! suites that need the comparator) and the programs in `examples/`. The crate's
//! `README.md` has the invocation and the environment variables of each.
//!
//! This library is what they share: the result comparator and its contract
//! tables ([`comparison`]), the allowance of the solver-outcome checks
//! ([`solver_spread`]), the solver jobs' variables and settings ([`solver`]),
//! the unique tiny edits of a session ([`edit`]), the host's load and CPU
//! time ([`host`]) and the check for a real Metal device
//! ([`require_metal`]).

pub mod comparison;
pub mod edit;
pub mod host;
pub mod solver;
pub mod solver_spread;

use mklight_engine::Engine;

/// Panics ("real Metal required") unless `engine` has an available Metal
/// device.
pub fn require_metal(engine: &Engine) {
    assert!(
        engine
            .backends()
            .iter()
            .any(|b| b.id == "gpu:metal" && b.available),
        "real Metal required"
    );
}
