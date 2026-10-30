//! MKLight geometric image-quality analyses (WIP)

use std::sync::Arc;

use mklight_core::materials::MaterialLibrary;
use mklight_core::{MkResult, OpticalSystem, ResolvedSystem, Snapshot};
use mklight_trace::SequentialTracer;

pub mod curvature;
pub mod fans;
pub mod illumination;
pub mod seidel;
pub mod spot;
pub mod util;

/// Convenience for tests, tools and examples: snapshot, resolve against the
/// built-in material library and build the reference tracer.
pub fn tracer_for(system: OpticalSystem) -> MkResult<SequentialTracer> {
    let snap = Snapshot::new(system)?;
    let sys = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin()))?;
    SequentialTracer::new(Arc::new(sys))
}
