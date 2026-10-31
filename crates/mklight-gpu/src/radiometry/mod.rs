//! Frozen radiometry seam; unsupported until its package qualifies it.
use crate::resident::{BufferId, PlanView};
use mklight_core::MkResult;
#[derive(Clone, Copy, Debug)]
pub enum RadiometryKind {
    Coating,
    Transmission,
    Polarization,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub kind: RadiometryKind,
    pub systems: crate::resident::SystemSet,
    pub hits: PlanView,
    pub stacks: PlanView,
    pub wavelength_incidence: PlanView,
    pub powers: BufferId,
    pub amplitudes: BufferId,
    pub status: BufferId,
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("radiometry")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("radiometry")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
