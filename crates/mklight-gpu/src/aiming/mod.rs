//! Frozen aiming seam; unsupported until its package qualifies it.
#[allow(unused_imports)]
use crate::{
    SystemHandle,
    resident::{BufferId, PlanView},
};
use mklight_core::MkResult;
#[derive(Clone, Debug)]
pub struct ResidentRays {
    pub floats: BufferId,
    pub indices: BufferId,
    pub count: usize,
    pub logical_ids: BufferId,
}
#[derive(Clone, Debug)]
pub struct ResidentHits {
    pub rays: ResidentRays,
    pub status: BufferId,
    pub records: Option<BufferId>,
    pub record_low: Option<BufferId>,
}
#[derive(Clone, Debug)]
pub struct GenerateAim {
    pub systems: crate::resident::SystemSet,
    pub samples: PlanView,
    pub field_wavelength: PlanView,
    pub seed: u64,
    pub tolerance: f32,
    pub max_iterations: u32,
    pub output: ResidentRays,
}
#[derive(Clone, Debug)]
pub struct TraceResident {
    pub systems: crate::resident::SystemSet,
    pub input: ResidentRays,
    pub steps: PlanView,
    pub start_surface: u32,
    pub apply_apertures: bool,
    pub radiometry: u32,
    pub newton: crate::NewtonSettings,
    pub output: ResidentHits,
}
#[derive(Clone, Debug)]
pub struct PupilReference {
    pub chief: PlanView,
    pub centre: PlanView,
    pub radius: crate::Df32,
    pub image_index: crate::Df32,
    pub wavelength_um: crate::Df32,
    pub sphere_sheet: i32,
    pub telecentric: bool,
}
#[derive(Clone, Debug)]
pub struct ProjectPupil {
    pub hits: ResidentHits,
    pub reference: PupilReference,
    pub output: BufferId,
    pub status: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    GenerateAim(GenerateAim),
    TraceResident(TraceResident),
    ProjectPupil(ProjectPupil),
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("aiming")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("aiming")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
