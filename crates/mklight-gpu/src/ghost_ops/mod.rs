//! Frozen ghost_ops seam; unsupported until its package qualifies it.
use crate::resident::{BufferId, PlanView};
use mklight_core::MkResult;
/// Exact ordered surface/interaction/order words, shared with stray.
#[derive(Clone, Debug)]
pub struct Coverage {
    pub aperture_words: PlanView,
    pub path_offsets: PlanView,
    pub path_events: PlanView,
    pub diffraction_visits: PlanView,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub systems: crate::resident::SystemSet,
    pub coverage: Coverage,
    pub probes: PlanView,
    pub settings: PlanView,
    pub output: BufferId,
    pub status: BufferId,
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("ghost_ops")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("ghost_ops")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
