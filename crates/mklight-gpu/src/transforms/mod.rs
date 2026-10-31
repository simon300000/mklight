//! Frozen transforms seam; unsupported until its package qualifies it.
use crate::resident::{BufferId, PlanView};
use mklight_core::MkResult;
#[derive(Clone, Copy, Debug)]
pub enum TransformKind {
    Fft,
    ChirpZ,
    AffineMft,
}
#[derive(Clone, Debug)]
pub struct TransformPlan {
    pub kind: TransformKind,
    pub input: PlanView,
    pub output: BufferId,
    pub dimensions: Vec<usize>,
    pub output_dimensions: Vec<usize>,
    pub sign: i32,
    pub scale: f32,
    pub affine: [crate::Df32; 6],
}
#[derive(Clone, Debug)]
pub struct CorrelationStencil {
    pub pupil: PlanView,
    pub frequencies: PlanView,
    pub mapping: [crate::Df32; 6],
    pub coverage: Option<PlanView>,
    pub output: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    Transform(TransformPlan),
    Correlation(CorrelationStencil),
}
pub fn validate(_: &Request, _: &crate::resident::ResidentPlan) -> MkResult<()> {
    crate::resident::unsupported("transforms")
}
pub fn execute(
    _: &Request,
    _: &mut crate::resident::cpu::Execution,
    _: &crate::pending::JobControl,
) -> MkResult<()> {
    crate::resident::unsupported("transforms")
}
pub fn accesses(_: &Request) -> crate::resident::Access {
    crate::resident::Access::default()
}
