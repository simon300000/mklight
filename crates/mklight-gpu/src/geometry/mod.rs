//! Frozen geometry seam; unsupported until its package qualifies it.
#[allow(unused_imports)]
use crate::{
    SystemHandle,
    resident::{BufferId, PlanView},
};
use mklight_core::MkResult;
#[derive(Clone, Copy, Debug)]
pub enum GeometryKind {
    SemiDiameters,
    SurfaceSamples,
    Layout,
    FieldMaps,
    Illumination,
    Spot,
    Energy,
    Focus,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub kind: GeometryKind,
    pub systems: crate::resident::SystemSet,
    pub samples: PlanView,
    pub settings: PlanView,
    pub output: BufferId,
    pub status: BufferId,
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("geometry")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("geometry")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
