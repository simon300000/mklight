//! Frozen wave_ops seam; unsupported until its package qualifies it.
use crate::resident::{BufferId, PlanView};
use mklight_core::MkResult;
#[derive(Clone, Copy, Debug)]
pub enum WaveKind {
    Pupil,
    Zernike,
    Psf,
    Huygens,
    Mtf,
    Energy,
    Detector,
    Image,
}
#[derive(Clone, Debug)]
pub struct Request {
    pub kind: WaveKind,
    pub input: PlanView,
    pub sampling: PlanView,
    pub settings: PlanView,
    pub output: BufferId,
    pub status: BufferId,
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("wave_ops")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("wave_ops")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
