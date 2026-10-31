//! Resident GPU adapter. No solver capability is enabled by this seam.
use mklight_core::{JobContext, MkResult};
use mklight_gpu::resident::{PlanResult, ResidentPlan};
use mklight_gpu::{ComputeBackend, Pending};
pub fn submit(
    _: &dyn ComputeBackend,
    _: ResidentPlan,
    _: &JobContext,
) -> MkResult<Pending<PlanResult>> {
    mklight_gpu::resident::unsupported("solver adapter")
}
