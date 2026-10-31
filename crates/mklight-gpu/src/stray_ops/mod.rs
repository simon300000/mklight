//! Frozen stray_ops seam; unsupported until its package qualifies it.
#[allow(unused_imports)]
use crate::{
    SystemHandle,
    resident::{BufferId, PlanView},
};
use mklight_core::MkResult;
#[derive(Clone, Debug)]
pub struct Request {
    pub systems: crate::resident::SystemSet,
    pub coverage: crate::ghost_ops::Coverage,
    pub scene: PlanView,
    pub sources: PlanView,
    pub settings: PlanView,
    pub seed: u64,
    pub first_ray: u64,
    pub ray_count: usize,
    pub output: BufferId,
    pub ledger: BufferId,
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("stray_ops")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("stray_ops")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
