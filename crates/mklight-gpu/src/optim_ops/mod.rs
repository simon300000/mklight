//! Frozen optim_ops seam; unsupported until its package qualifies it.
#[allow(unused_imports)]
use crate::{
    SystemHandle,
    resident::{BufferId, PlanView},
};
use mklight_core::MkResult;
#[derive(Clone, Copy, Debug)]
pub enum OptimKind {
    Residuals,
    FiniteDifference,
    JacobianProducts,
    LeastSquares,
    Tolerance,
    Compensators,
    Glass,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub kind: OptimKind,
    pub candidates: crate::systems::CandidateMap,
    pub values: PlanView,
    pub operands: PlanView,
    pub settings: PlanView,
    pub output: BufferId,
    pub status: BufferId,
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("optim_ops")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("optim_ops")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
