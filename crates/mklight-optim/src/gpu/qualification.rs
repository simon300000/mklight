//! Account for the legacy trace adapter used by the shared qualification
//! service. These owned-input/output copies are separate from resident plans.
use super::GpuContext;
use mklight_core::{JobContext, MkResult};
use mklight_gpu::resident::{PlanStats, StageStats};
use mklight_gpu::*;
use std::sync::atomic::{AtomicU64, Ordering};
pub(super) struct QualificationBackend<'a> {
    pub context: &'a GpuContext,
    system_bytes: AtomicU64,
}
impl<'a> QualificationBackend<'a> {
    pub fn new(context: &'a GpuContext) -> Self {
        Self {
            context,
            system_bytes: AtomicU64::new(0),
        }
    }
}
impl ComputeBackend for QualificationBackend<'_> {
    fn info(&self) -> &DeviceInfo {
        self.context.backend.info()
    }
    fn precision(&self) -> &PrecisionDescriptor {
        self.context.backend.precision()
    }
    fn supports(&self, k: JobKind) -> bool {
        self.context.backend.supports(k)
    }
    fn upload_systems(&self, s: &[PackedSystem]) -> MkResult<SystemHandle> {
        let h = self.context.backend.upload_systems(s)?;
        let bytes = s
            .iter()
            .map(|s| {
                size_of_val(&s.header)
                    + size_of_val(s.surfaces.as_slice())
                    + size_of_val(s.coefficients.as_slice())
            })
            .sum::<usize>() as u64;
        self.system_bytes.store(bytes, Ordering::Relaxed);
        if self.info().kind == BackendKind::Metal {
            self.context.observe(PlanStats {
                copied_bytes: bytes,
                uploaded_bytes: bytes,
                allocations: 3,
                peak_bytes: bytes,
                ..Default::default()
            });
        }
        Ok(h)
    }
    fn release_systems(&self, h: SystemHandle) -> MkResult<()> {
        self.context.backend.release_systems(h)
    }
    fn submit(&self, j: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>> {
        self.context.backend.submit(j, ctx)
    }
    fn run(&self, j: Job, ctx: &JobContext) -> MkResult<JobResult> {
        let input = match &j {
            Job::Trace(t) => {
                let r = &t.rays;
                let planes = [
                    &r.x, &r.y, &r.z, &r.l, &r.m, &r.n, &r.opl_hi, &r.opl_lo, &r.weight, &r.x_lo,
                    &r.y_lo, &r.z_lo, &r.l_lo, &r.m_lo, &r.n_lo,
                ];
                4 * (planes.iter().map(|v| v.len()).sum::<usize>()
                    + r.wavelength.len()
                    + r.system.len()
                    + t.program().len()) as u64
            }
            _ => 0,
        };
        let steps = match &j {
            Job::Trace(t) => 4 * t.program().len() as u64,
            _ => 0,
        };
        let result = self.context.backend.run(j, ctx)?;
        let output = match &result.output {
            JobOutput::Trace(t) => {
                4 * (13 * t.len() + t.recorded.len() + t.recorded_status.len()) as u64
            }
            _ => 0,
        };
        let copies = self.info().kind == BackendKind::Metal;
        self.context.observe(PlanStats {
            wall_seconds: result.stats.wall_seconds,
            device_seconds: result.stats.device_seconds,
            command_buffers: result.stats.command_buffers,
            copied_bytes: if copies { steps + output } else { 0 },
            uploaded_bytes: input,
            readback_bytes: output,
            allocations: if copies { 7 } else { 0 },
            peak_bytes: input + output + self.system_bytes.load(Ordering::Relaxed),
            stages: vec![StageStats {
                name: "legacy geometry qualification".into(),
                wall_seconds: result.stats.wall_seconds,
                device_seconds: result.stats.device_seconds,
                command_buffers: result.stats.command_buffers,
                ..Default::default()
            }],
            ..Default::default()
        });
        Ok(result)
    }
}
