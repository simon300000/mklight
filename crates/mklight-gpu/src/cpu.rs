//! CPU-f32 emulation backend.
//!
//! Runs the GPU algorithms on the CPU with the same f32 operations in the
//! same order, so its results are bit-identical to a conforming GPU (see
//! `NumericsProbe`). Comparing *reference f64 ↔ CPU-f32* isolates precision
//! error; comparing *CPU-f32 ↔ GPU* isolates backend/compiler error.
//!
//! Kernels available now: Philox (stream and raw blocks), double-float
//! operations, compensated reduction and the packed-sag probe. The
//! sequential trace kernel is added together with its GPU port.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use mklight_core::job::Precision;
use mklight_core::rng::philox4x32_10;
use mklight_core::{JobContext, MkError, MkResult};
use rayon::prelude::*;

use crate::backend::*;
use crate::df::Df32;
use crate::fmath::{packed_sag, u32_to_unit_f32};
use crate::layout::{PackedSystem, PackedSystemSet};
use crate::pending::{Pending, pending};
use crate::reduce;

/// Work items between cancellation checks / progress reports.
const CHUNK: usize = 1 << 16;

pub struct CpuF32Backend {
    inner: Arc<Inner>,
}

struct Inner {
    info: DeviceInfo,
    precision: PrecisionDescriptor,
    instance: u64,
    next_id: AtomicU64,
    systems: Mutex<HashMap<u64, Arc<PackedSystemSet>>>,
}

impl Default for CpuF32Backend {
    fn default() -> Self {
        Self::new()
    }
}

impl CpuF32Backend {
    pub fn new() -> Self {
        let inner = Inner {
            info: DeviceInfo {
                kind: BackendKind::CpuF32,
                name: "CPU f32 emulation".into(),
                vendor: std::env::consts::ARCH.into(),
                max_buffer_bytes: isize::MAX as u64,
                recommended_working_set_bytes: None,
                unified_memory: true,
                ray_tracing_api: false,
                hardware_ray_tracing: false,
                native_fp64: true,
                simd_width: 1,
                max_threads_per_group: reduce::THREADS as u32,
            },
            precision: PrecisionDescriptor {
                model: Precision::F32Gpu,
                fast_math_disabled: true,
                opl_double_float: true,
                compensated_reductions: true,
                probe: NumericsProbe::IEEE,
                budget: ErrorBudget::CONVENTIONS,
            },
            instance: next_backend_instance(),
            next_id: AtomicU64::new(1),
            systems: Mutex::new(HashMap::new()),
        };
        CpuF32Backend {
            inner: Arc::new(inner),
        }
    }
}

impl Inner {
    fn lookup(&self, h: SystemHandle) -> MkResult<Arc<PackedSystemSet>> {
        if h.backend_instance != self.instance {
            return Err(MkError::Backend(
                "system handle belongs to another backend".into(),
            ));
        }
        self.systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&h.id)
            .cloned()
            .ok_or_else(|| MkError::Backend("unknown or released system handle".into()))
    }
}

/// Run `f` over `0..total` in chunks, checking cancellation and reporting
/// progress between chunks (the CPU analogue of command buffers).
fn chunked<T: Send>(
    total: usize,
    ctx: &JobContext,
    stage: &str,
    stats: &mut JobStats,
    f: impl Fn(usize) -> T + Sync,
) -> MkResult<Vec<T>> {
    let mut out = Vec::with_capacity(total);
    let mut start = 0;
    while start < total {
        ctx.check()?;
        let end = (start + CHUNK).min(total);
        out.par_extend((start..end).into_par_iter().map(&f));
        stats.command_buffers += 1;
        start = end;
        ctx.report(end as f64 / total as f64, stage);
    }
    ctx.check()?;
    Ok(out)
}

fn philox_output(blocks: Vec<[u32; 4]>) -> JobOutput {
    let raw: Vec<u32> = blocks.into_iter().flatten().collect();
    let uniform = raw.iter().map(|&w| u32_to_unit_f32(w)).collect();
    JobOutput::Philox { raw, uniform }
}

impl Inner {
    fn execute(&self, job: Job, ctx: &JobContext) -> MkResult<JobResult> {
        let t0 = Instant::now();
        let mut stats = JobStats::default();
        let output = match job {
            Job::PhiloxStream {
                seed,
                stream,
                first_block,
                num_blocks,
            } => {
                let key = [seed as u32, (seed >> 32) as u32];
                let blocks = chunked(num_blocks as usize, ctx, "philox", &mut stats, |t| {
                    let b = first_block.wrapping_add(t as u64);
                    let ctr = [
                        b as u32,
                        (b >> 32) as u32,
                        stream as u32,
                        (stream >> 32) as u32,
                    ];
                    philox4x32_10(ctr, key)
                })?;
                philox_output(blocks)
            }
            Job::PhiloxBlocks { counters, keys } => {
                let blocks = chunked(counters.len(), ctx, "philox", &mut stats, |i| {
                    philox4x32_10(counters[i], keys[i])
                })?;
                philox_output(blocks)
            }
            Job::DfOps { op, a, b } => {
                JobOutput::Df(chunked(a.len(), ctx, "df", &mut stats, |i| {
                    op.apply(a[i], b[i])
                })?)
            }
            Job::Reduce {
                mode,
                count,
                values,
            } => {
                let channels = values.len() / count;
                let mut sums = Vec::with_capacity(channels);
                for c in 0..channels {
                    ctx.check()?;
                    sums.push(reduce::reduce_channel(
                        mode,
                        &values[c * count..(c + 1) * count],
                    ));
                    stats.command_buffers += 1;
                    ctx.report((c + 1) as f64 / channels as f64, "reduce");
                }
                JobOutput::Sums(sums)
            }
            Job::SagProbe {
                systems,
                system,
                surface,
                r2,
            } => {
                let set = self.lookup(systems)?;
                let h = &set.headers[system as usize];
                if surface >= h.num_surfaces {
                    return Err(MkError::Backend("sag probe: surface out of range".into()));
                }
                let s = set.surfaces[(h.surface_offset + surface) as usize];
                JobOutput::F32(chunked(r2.len(), ctx, "sag", &mut stats, |i| {
                    packed_sag(&s, r2[i])
                })?)
            }
            Job::Trace(_) => {
                return Err(MkError::Unsupported(
                    "cpu-f32: sequential trace kernel not implemented yet".into(),
                ));
            }
        };
        stats.wall_seconds = t0.elapsed().as_secs_f64();
        Ok(JobResult { output, stats })
    }
}

impl ComputeBackend for CpuF32Backend {
    fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }
    fn precision(&self) -> &PrecisionDescriptor {
        &self.inner.precision
    }
    fn supports(&self, kind: JobKind) -> bool {
        kind != JobKind::Trace
    }
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle> {
        let set = PackedSystemSet::new(systems)?;
        let id = self.inner.next_id.fetch_add(1, Ordering::Relaxed);
        let count = set.len() as u32;
        self.inner
            .systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, Arc::new(set));
        Ok(SystemHandle {
            backend_instance: self.inner.instance,
            id,
            count,
        })
    }
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()> {
        self.inner.lookup(handle)?;
        self.inner
            .systems
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&handle.id);
        Ok(())
    }
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>> {
        job.validate()?;
        if !self.supports(job.kind()) {
            return Err(MkError::Unsupported(format!(
                "cpu-f32 does not implement {:?}",
                job.kind()
            )));
        }
        // Resolve handles now so a released handle fails at submission.
        if let Job::SagProbe { systems, .. } = &job {
            self.inner.lookup(*systems)?;
        }
        let (p, c) = pending(ctx.cancel.clone());
        // The emulation is CPU-bound and parallel inside; run it on its own
        // thread so `submit` returns immediately, like a GPU queue.
        let ctx = ctx.clone();
        let inner = self.inner.clone();
        std::thread::Builder::new()
            .name("mklight-cpu-f32".into())
            .spawn(move || c.complete(inner.execute(job, &ctx)))
            .map_err(|e| MkError::Backend(e.to_string()))?;
        Ok(p)
    }
}

/// Convenience: run a job and return its output, for tests and tools.
pub fn run_output(b: &dyn ComputeBackend, job: Job) -> MkResult<JobOutput> {
    Ok(b.run(job, &JobContext::new())?.output)
}

/// Convenience: the double-float sums of a reduction job.
pub fn reduce_sums(
    b: &dyn ComputeBackend,
    mode: ReduceMode,
    count: usize,
    values: Vec<f32>,
) -> MkResult<Vec<Df32>> {
    match run_output(
        b,
        Job::Reduce {
            mode,
            count,
            values,
        },
    )? {
        JobOutput::Sums(s) => Ok(s),
        other => Err(MkError::Backend(format!("unexpected output {other:?}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::rng::Philox;

    #[test]
    fn philox_stream_matches_core() {
        let b = CpuF32Backend::new();
        let JobOutput::Philox { raw, uniform } = run_output(
            &b,
            Job::PhiloxStream {
                seed: 0x1234_5678_9abc_def0,
                stream: 77,
                first_block: 3,
                num_blocks: 1000,
            },
        )
        .unwrap() else {
            panic!()
        };
        // Skip draws 0..12 (blocks 0..3) by drawing them.
        let mut r = Philox::new(0x1234_5678_9abc_def0, 77);
        for _ in 0..12 {
            r.next_u32();
        }
        for (i, w) in raw.iter().enumerate() {
            assert_eq!(*w, r.next_u32(), "draw {i}");
            assert_eq!(uniform[i], u32_to_unit_f32(*w));
        }
    }

    #[test]
    fn cancellation_between_chunks() {
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        ctx.cancel.cancel();
        let r = b.run(
            Job::PhiloxStream {
                seed: 1,
                stream: 0,
                first_block: 0,
                num_blocks: 1 << 20,
            },
            &ctx,
        );
        assert_eq!(r, Err(MkError::Cancelled));
    }

    #[test]
    fn progress_reaches_one() {
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        let r = b
            .run(
                Job::DfOps {
                    op: DfOp::Add,
                    a: vec![Df32::from_f32(1.0); 200_000],
                    b: vec![Df32::from_f32(2.0); 200_000],
                },
                &ctx,
            )
            .unwrap();
        assert_eq!(ctx.progress.fraction(), 1.0);
        assert!(r.stats.command_buffers >= 4);
        let JobOutput::Df(v) = r.output else { panic!() };
        assert!(v.iter().all(|z| *z == Df32::from_f32(3.0)));
    }

    #[test]
    fn handles_are_checked() {
        let a = CpuF32Backend::new();
        let b = CpuF32Backend::new();
        let rs = crate::testutil::singlet_resolved();
        let p = PackedSystem::pack_all(&rs).unwrap();
        let h = a.upload_system(&p).unwrap();
        let probe = |h| Job::SagProbe {
            systems: h,
            system: 0,
            surface: 1,
            r2: vec![0.0, 100.0],
        };
        assert!(b.submit(probe(h), &JobContext::new()).is_err());
        let JobOutput::F32(s) = run_output(&a, probe(h)).unwrap() else {
            panic!()
        };
        assert_eq!(s[0], 0.0);
        assert!((s[1] as f64 - rs.surfaces[1].shape.sag(100.0).unwrap()).abs() < 1e-6);
        a.release_systems(h).unwrap();
        assert!(a.submit(probe(h), &JobContext::new()).is_err());
        assert!(matches!(
            a.submit(
                Job::Trace(Box::new(crate::batch::TraceRequest {
                    systems: h,
                    rays: Default::default(),
                    start_surface: 0,
                    end_surface: 3,
                    record: Default::default(),
                    apply_apertures: true,
                })),
                &JobContext::new()
            ),
            Err(MkError::Unsupported(_))
        ));
    }

    #[test]
    fn reduce_via_backend() {
        let b = CpuF32Backend::new();
        let n = 10_000;
        let mut v: Vec<f32> = (0..n).map(|i| i as f32).collect();
        v.extend((0..n).map(|i| -(i as f32) * 0.5));
        let s = reduce_sums(&b, ReduceMode::DoubleFloat, n, v).unwrap();
        let want = (n * (n - 1) / 2) as f64;
        assert_eq!(s[0].to_f64(), want);
        assert_eq!(s[1].to_f64(), -want * 0.5);
    }
}
