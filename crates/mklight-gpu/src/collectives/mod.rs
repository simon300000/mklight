//! Frozen collectives seam; unsupported until its package qualifies it.
#[allow(unused_imports)]
use crate::{
    SystemHandle,
    resident::{BufferId, PlanView},
};
use mklight_core::MkResult;
#[derive(Clone, Debug)]
pub struct SegmentReduce {
    pub values: PlanView,
    pub offsets: PlanView,
    pub reference: Option<PlanView>,
    pub mode: crate::ReduceMode,
    pub output: BufferId,
}
#[derive(Clone, Debug)]
pub struct StableCompact {
    pub values: PlanView,
    pub ids: PlanView,
    pub mask: PlanView,
    pub output: BufferId,
    pub output_ids: BufferId,
    pub count: BufferId,
}
#[derive(Clone, Debug)]
pub struct StableSort {
    pub keys: PlanView,
    pub ids: PlanView,
    pub values: Option<PlanView>,
    pub output_keys: BufferId,
    pub output_ids: BufferId,
    pub output_values: Option<BufferId>,
}
#[derive(Clone, Debug)]
pub struct Scan {
    pub values: PlanView,
    pub inclusive: bool,
    pub output: BufferId,
}
#[derive(Clone, Debug)]
pub struct QrFit {
    pub matrix: PlanView,
    pub rhs: PlanView,
    pub rank_tolerance: f32,
    pub factors: BufferId,
    pub solution: BufferId,
    pub rank_condition: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    SegmentReduce(SegmentReduce),
    StableCompact(StableCompact),
    StableSort(StableSort),
    Scan(Scan),
    QrFit(QrFit),
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("collectives")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("collectives")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
