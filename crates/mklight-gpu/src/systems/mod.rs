//! Frozen systems seam; unsupported until its package qualifies it.
#[allow(unused_imports)]
use crate::{
    SystemHandle,
    resident::{BufferId, PlanView},
};
use mklight_core::MkResult;
#[derive(Clone, Debug)]
pub struct CandidateMap {
    pub candidate_ids: BufferId,
    pub configuration_ids: BufferId,
    pub status: BufferId,
    pub count: usize,
}
#[derive(Clone, Debug)]
pub struct SystemBatchPlan {
    pub base: SystemHandle,
    pub parameters: PlanView,
    pub slots: PlanView,
    pub bounds: PlanView,
    pub dependencies: PlanView,
    pub material_tables: PlanView,
    pub headers: BufferId,
    pub surfaces: BufferId,
    pub coefficients: BufferId,
    pub first_order: BufferId,
    pub map: CandidateMap,
}
#[derive(Clone, Debug)]
pub enum Request {
    Prepare(SystemBatchPlan),
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("systems")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("systems")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
