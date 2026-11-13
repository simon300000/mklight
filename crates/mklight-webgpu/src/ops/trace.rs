//! The sequential / path trace kernel on the device: `Job::Trace`,
//! `ResidentOp::TraceUpload` and `aiming::Request::TraceResident`
//! (`kernels/trace.wgsl`).
//!
//! Every launch is one batch: the host packs the launch's rays into planes
//! of the launch's length (one upload of every input plane, a 32-bit word
//! each: float bits and indices together, so the kernel binds 7 storage
//! buffers), the kernel writes its output planes and records into two
//! scratch buffers, and the batch reads each plane back into its place in
//! the caller's storage. Launches hold at most [`chunk_rays`] rays and fit
//! every binding into `maxStorageBufferBindingSize` and the dispatch into
//! `maxComputeWorkgroupsPerDimension`; cancellation and progress are checked
//! between them. Requests are validated by the shared host code
//! (`PreparedTrace`); resident traces check their device-made system,
//! wavelength and path indices in the kernel, as Metal's do.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use mklight_core::MkResult;
use mklight_gpu::aiming::{Request, TraceResident};
use mklight_gpu::consts::*;
use mklight_gpu::layout::TraceParams;
use mklight_gpu::pending::JobControl;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::resident::{BufferId, ResidentOp, TraceUpload};
use mklight_gpu::{JobStats, PackedSystemSet, PreparedTrace, TraceRequest, TraceResult};

use super::WebGpuOps;
use super::resident::Mirrors;
use crate::backend::{DeviceSystems, WebGpuBackend};
use crate::device::*;

/// Whether the kernel is ported.
pub const PORTED: bool = true;

/// Threads per workgroup of every trace entry point.
const GROUP: usize = 64;
/// Planes of the plain trace's packed input and output.
const IN_PLANES: usize = (MKL_TRACE_IN_FLOATS + MKL_TRACE_IN_UINTS) as usize;
const OUT_PLANES: usize = (MKL_TRACE_OUT_FLOATS + MKL_TRACE_OUT_UINTS) as usize;
const RECORD: usize = MKL_TRACE_RECORD_FLOATS as usize;
/// Planes of the resident trace's packed input (15 floats, system and
/// wavelength, 2 logical ids, 4 report words) and output (the same and 3
/// status planes); words per record slot (hi and lo).
const RES_IN_PLANES: usize = 23;
const RAY_FLOATS: usize = MKL_TRACE_IN_FLOATS as usize;
const RES_OUT_PLANES: usize = 26;
const RES_RECORD: usize = MKL_AIM_RECORD_FLOATS as usize;

static CHUNK_RAYS: AtomicUsize = AtomicUsize::new(1 << 18);

/// The most rays one launch traces (before the binding limits).
pub fn chunk_rays() -> usize {
    CHUNK_RAYS.load(Ordering::Relaxed)
}

/// Set [`chunk_rays`] (at least 1); returns the previous value. Launch
/// boundaries never change a result.
pub fn set_chunk_rays(n: usize) -> usize {
    CHUNK_RAYS.swap(n.max(1), Ordering::Relaxed)
}

fn module(overlays: bool) -> &'static str {
    if overlays { "trace_overlays" } else { "trace" }
}

pub(super) fn has_overlays(set: &PackedSystemSet) -> bool {
    set.surfaces
        .iter()
        .any(|s| s.flags & MKL_SURF_HAS_SAG_OVERLAY != 0)
}

pub(super) fn shape() -> PassShape {
    use Access::*;
    PassShape {
        uniform: true,
        storage: vec![Read, Read, Read, Read, Read, ReadWrite, ReadWrite],
    }
}

/// The trace pipelines (`prepare_kernels`).
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    let mut v = Vec::new();
    for overlays in [false, true] {
        for entry in [
            "mkl_trace",
            "mkl_trace_resident",
            "mkl_trace_resident_records",
        ] {
            v.push((Kernel::new(module(overlays), entry), shape()));
        }
    }
    v
}

/// Rays per launch so that a binding of `words` words per ray fits.
fn launch_rays(gpu: &dyn Gpu, words: usize) -> usize {
    let s = gpu.summary();
    let binding = (s.max_storage_buffer_binding_size / 4) as usize / words.max(1);
    let groups = s.max_compute_workgroups_per_dimension as usize * GROUP;
    chunk_rays().min(binding).min(groups).max(1)
}

/// The parameter block (`MklWgslTraceParams`).
pub(super) fn uniform(p: &TraceParams, systems: u32) -> Vec<u32> {
    let mut w = bytemuck::cast_slice::<TraceParams, u32>(std::slice::from_ref(p)).to_vec();
    w.extend([systems, 0, 0, 0]);
    uniform_words(&w)
}

/// A launch's packed input: uploaded words or a device buffer.
#[derive(Clone, Copy)]
enum Input<'a> {
    Words(&'a [u32]),
    Key(BufferKey),
}

/// A temporary device buffer, released on drop.
struct Temp {
    gpu: Arc<dyn Gpu>,
    key: BufferKey,
}

impl Temp {
    fn new(gpu: &Arc<dyn Gpu>, words: usize) -> MkResult<Self> {
        Ok(Self {
            gpu: gpu.clone(),
            key: gpu.allocate(words.max(1))?,
        })
    }
}

impl Drop for Temp {
    fn drop(&mut self) {
        self.gpu.release(self.key);
    }
}

/// One launch of `kernel` over `len` rays: packed `input`, `out_words` output
/// and `rec_words` record words; `reads` copies them into place.
#[allow(clippy::too_many_arguments)]
fn launch<'a>(
    gpu: &dyn Gpu,
    dev: &DeviceSystems,
    kernel: Kernel,
    params: Vec<u32>,
    steps: &'a [u32],
    input: Input<'a>,
    len: usize,
    out_words: usize,
    rec_words: usize,
    reads: impl FnOnce(&mut Batch<'a>, usize, usize) -> MkResult<()>,
    stats: &mut JobStats,
) -> MkResult<()> {
    let mut batch = Batch::default();
    let storage = vec![
        Binding::read(batch.persistent(dev.headers)),
        Binding::read(batch.persistent(dev.surfaces)),
        Binding::read(batch.persistent(dev.coefficients)),
        Binding::read(batch.upload(steps)),
        Binding::read(match input {
            Input::Words(w) => batch.upload(w),
            Input::Key(k) => batch.persistent(k),
        }),
    ];
    let out = batch.scratch(out_words);
    let rec = batch.scratch(rec_words.max(4));
    let storage = storage
        .into_iter()
        .chain([Binding::write(out), Binding::write(rec)])
        .collect();
    batch.passes.push(Pass {
        kernel,
        uniform: Some(params),
        storage,
        workgroups: [len.div_ceil(GROUP) as u32, 1, 1],
    });
    reads(&mut batch, out, rec)?;
    let b = gpu.run(batch)?;
    stats.device_seconds += b.device_seconds;
    stats.command_buffers += 1;
    Ok(())
}

/// The plain trace's outputs: `MKL_TRACE_OUT_FLOATS` and
/// `MKL_TRACE_OUT_UINTS` planes of `n`, and the records (`slots · n ·
/// MKL_TRACE_RECORD_FLOATS` floats, `slots · n` status words).
struct TraceOut<'a> {
    floats: &'a mut [u32],
    indices: &'a mut [u32],
    records: &'a mut [u32],
    record_status: &'a mut [u32],
}

/// Where the plain trace's outputs go.
enum Dest<'d> {
    /// Host slices (`Job::Trace`).
    Host(TraceOut<'d>),
    /// The device mirrors of plan buffers (floats, indices, records,
    /// record status), never read back by the trace.
    Device(&'d mut Mirrors, &'d Execution, [BufferId; 4]),
}

/// The host slices of [`Dest::Host`], one per plane.
struct HostPlanes<'a> {
    floats: Vec<&'a mut [u32]>,
    indices: Vec<&'a mut [u32]>,
    records: Vec<&'a mut [u32]>,
    record_status: Vec<&'a mut [u32]>,
}

/// The plain trace of the request's rays (`f`, `u`: its input planes) into
/// `dest`, in launches.
#[allow(clippy::too_many_arguments)]
fn trace_planes(
    backend: &WebGpuBackend,
    set: &PackedSystemSet,
    prep: &PreparedTrace,
    n: usize,
    f: &[f32],
    u: &[u32],
    dest: Dest<'_>,
    ctl: &JobControl,
    stats: &mut JobStats,
    stage: &str,
) -> MkResult<()> {
    let gpu = &**backend.gpu();
    let dev = backend.device_systems(set)?;
    let slots = prep.record_slots;
    let kernel = Kernel::new(module(has_overlays(set)), "mkl_trace");
    let chunk = launch_rays(gpu, IN_PLANES.max(slots * (RECORD + 1)));
    let f: &[u32] = bytemuck::cast_slice(f);
    let steps: &[u32] = if prep.steps.is_empty() {
        &[0]
    } else {
        &prep.steps
    };
    let (mut host, mut device) = match dest {
        Dest::Host(TraceOut {
            floats,
            indices,
            records,
            record_status,
        }) => (
            Some(HostPlanes {
                floats: floats[..MKL_TRACE_OUT_FLOATS as usize * n]
                    .chunks_mut(n.max(1))
                    .collect(),
                indices: indices[..3 * n].chunks_mut(n.max(1)).collect(),
                records: records[..slots * n * RECORD]
                    .chunks_mut((n * RECORD).max(1))
                    .collect(),
                record_status: record_status[..slots * n].chunks_mut(n.max(1)).collect(),
            }),
            None,
        ),
        Dest::Device(mirrors, x, ids) => (None, Some((mirrors, x, ids))),
    };
    let mut input = Vec::new();
    let mut start = 0;
    while start < n {
        ctl.check()?;
        let len = chunk.min(n - start);
        input.clear();
        for k in 0..MKL_TRACE_IN_FLOATS as usize {
            input.extend_from_slice(&f[k * n + start..][..len]);
        }
        for k in 0..MKL_TRACE_IN_UINTS as usize {
            input.extend_from_slice(&u[k * n + start..][..len]);
        }
        let p = TraceParams {
            num_rays: len as u32,
            stride: len as u32,
            ..prep.params
        };
        let (host, device) = (host.as_mut(), device.as_mut());
        launch(
            gpu,
            &dev,
            kernel,
            uniform(&p, dev.count),
            steps,
            Input::Words(&input),
            len,
            OUT_PLANES * len,
            slots * len * (RECORD + 1),
            move |batch, out, rec| {
                if let Some(h) = host {
                    let planes = h.floats.iter_mut().chain(h.indices.iter_mut());
                    for (k, plane) in planes.enumerate() {
                        batch.read(out, k * len, &mut plane[start..start + len]);
                    }
                    for (slot, plane) in h.records.iter_mut().enumerate() {
                        let r = &mut plane[start * RECORD..(start + len) * RECORD];
                        batch.read(rec, slot * len * RECORD, r);
                    }
                    let status = slots * len * RECORD;
                    for (slot, plane) in h.record_status.iter_mut().enumerate() {
                        let r = &mut plane[start..start + len];
                        batch.read(rec, status + slot * len, r);
                    }
                }
                if let Some((mirrors, x, ids)) = device {
                    let floats = MKL_TRACE_OUT_FLOATS as usize;
                    for k in 0..floats {
                        mirrors.store(batch, x, out, k * len, ids[0], k * n + start, len)?;
                    }
                    for k in 0..3 {
                        let from = (floats + k) * len;
                        mirrors.store(batch, x, out, from, ids[1], k * n + start, len)?;
                    }
                    for slot in 0..slots {
                        let (from, to) = (slot * len * RECORD, (slot * n + start) * RECORD);
                        mirrors.store(batch, x, rec, from, ids[2], to, len * RECORD)?;
                        let from = slots * len * RECORD + slot * len;
                        mirrors.store(batch, x, rec, from, ids[3], slot * n + start, len)?;
                    }
                }
                Ok(())
            },
            stats,
        )?;
        start += len;
        ctl.report(start as f64 / n as f64, stage);
    }
    ctl.check()
}

/// `Job::Trace` on the device.
pub fn job(
    backend: &WebGpuBackend,
    req: &TraceRequest,
    set: &PackedSystemSet,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<TraceResult> {
    let prep = PreparedTrace::new(req, set)?;
    let n = req.rays.len();
    let (f, u) = PreparedTrace::input_planes(&req.rays);
    let mut out_f = vec![0f32; MKL_TRACE_OUT_FLOATS as usize * n];
    let mut out_u = vec![0u32; MKL_TRACE_OUT_UINTS as usize * n];
    let mut rec_f = vec![0f32; prep.record_slots * n * RECORD];
    let mut rec_u = vec![0u32; prep.record_slots * n];
    trace_planes(
        backend,
        set,
        &prep,
        n,
        &f,
        &u,
        Dest::Host(TraceOut {
            floats: bytemuck::cast_slice_mut(&mut out_f),
            indices: &mut out_u,
            records: bytemuck::cast_slice_mut(&mut rec_f),
            record_status: &mut rec_u,
        }),
        ctl,
        stats,
        "webgpu trace",
    )?;
    Ok(TraceResult::from_planes(&out_f, &out_u, n, n, rec_f, rec_u))
}

/// Run a trace operation of a plan on the device, or `Ok(None)` for its
/// CPU-f32 specification.
pub fn run(
    ops: &mut WebGpuOps,
    _i: usize,
    op: &ResidentOp,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    match op {
        ResidentOp::TraceUpload(t) => trace_upload(ops, t, x, ctl, stats),
        ResidentOp::Aiming(Request::TraceResident(t)) => trace_resident(ops, t, x, ctl, stats),
        _ => Ok(None),
    }
}

/// `ResidentOp::TraceUpload` (Metal's `resident::trace`): the rays are
/// request data and uploaded; the outputs stay on the device.
fn trace_upload(
    ops: &mut WebGpuOps,
    t: &TraceUpload,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    let req = t.request();
    let set = x.system(req.systems)?;
    let prep = PreparedTrace::new(req, &set)?;
    let n = req.rays.len();
    let (f, u) = PreparedTrace::input_planes(&req.rays);
    let o = &t.output;
    trace_planes(
        &ops.backend,
        &set,
        &prep,
        n,
        &f,
        &u,
        Dest::Device(
            &mut ops.mirrors,
            x,
            [o.floats, o.indices, o.records, o.record_status],
        ),
        ctl,
        stats,
        "resident trace",
    )?;
    x.stats.uploaded_bytes += (n * IN_PLANES * 4) as u64;
    Ok(Some("webgpu:trace"))
}

/// `aiming::Request::TraceResident` (Metal's resident trace): rays and
/// path program from plan buffers; the kernel validates each ray's system,
/// wavelength and the program as Metal's does.
fn trace_resident(
    ops: &mut WebGpuOps,
    t: &TraceResident,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    let set = x.system(t.systems)?;
    let inputs = [
        t.input.floats,
        t.input.indices,
        t.input.logical_ids,
        t.input.report,
    ];
    let mut outputs = vec![
        t.output.rays.floats,
        t.output.rays.indices,
        t.output.rays.logical_ids,
        t.output.rays.report,
        t.output.status,
    ];
    outputs.extend(t.output.records);
    outputs.extend(t.output.record_low);
    // The specification reports the unsupported coating; it also runs a
    // trace in place (an input buffer is an output).
    if (t.radiometry != 0 && set.surfaces.iter().any(|s| s.coating == MKL_COATING_TABLE))
        || inputs.iter().any(|i| outputs.contains(i))
        || outputs.contains(&t.steps.buffer)
    {
        return Ok(None);
    }
    // The path program is read by the host (its length sizes the records).
    ops.mirrors.read_back(t.steps.buffer, x, stats)?;
    let (backend, mirrors) = (&ops.backend, &mut ops.mirrors);
    let x: &Execution = x;
    let gpu_arc = backend.gpu().clone();
    let gpu = &*gpu_arc;
    let dev = backend.device_systems(&set)?;
    let n = t.input.count;
    let steps = x.view(&t.steps).to_vec();
    let recorded = t.output.records.is_some();
    let slots = if recorded { steps.len() + 1 } else { 0 };
    let kernel = Kernel::new(
        module(has_overlays(&set)),
        if recorded {
            "mkl_trace_resident_records"
        } else {
            "mkl_trace_resident"
        },
    );
    let chunk = launch_rays(
        gpu,
        RES_IN_PLANES
            .max(RES_OUT_PLANES)
            .max(slots * 2 * RES_RECORD),
    );
    let flags = if t.apply_apertures { MKL_TRACE_CLIP } else { 0 }
        | if t.radiometry != 0 {
            MKL_TRACE_RADIOMETRY
        } else {
            0
        }
        | if recorded { MKL_TRACE_RECORD } else { 0 };
    // Input planes in kernel order: the kernel binds one packed buffer. When
    // any input is device-resident the launch's planes are packed on the
    // device into a temporary buffer (a copy batch before the launch).
    let in_ids = [
        (t.input.floats, RAY_FLOATS),
        (t.input.indices, 2),
        (t.input.logical_ids, 2),
        (t.input.report, 4),
    ];
    let packed = if in_ids.iter().any(|&(id, _)| mirrors.on_device(id)) {
        Some(Temp::new(&gpu_arc, RES_IN_PLANES * chunk.min(n.max(1)))?)
    } else {
        None
    };
    let rays = [
        t.output.rays.floats,
        t.output.rays.indices,
        t.output.rays.logical_ids,
        t.output.rays.report,
        t.output.status,
    ];
    let halves: Vec<BufferId> = t
        .output
        .records
        .into_iter()
        .chain(t.output.record_low)
        .collect();
    let mut input = Vec::new();
    let mut start = 0;
    while start < n {
        ctl.check()?;
        let len = chunk.min(n - start);
        let words = if let Some(tmp) = &packed {
            let mut b = Batch::default();
            let mut at = 0;
            for &(id, k) in &in_ids {
                let res = mirrors.resource(&mut b, id);
                for j in 0..k {
                    match res {
                        Some(r) => {
                            b.copy(r, j * n + start, tmp.key, at, len);
                            mirrors.saved_bytes += (len * 4) as u64;
                        }
                        None => {
                            let up = b.upload(&x.words(id)[j * n + start..][..len]);
                            b.copy(up, 0, tmp.key, at, len);
                        }
                    }
                    at += len;
                }
            }
            let r = gpu.run(b)?;
            stats.device_seconds += r.device_seconds;
            stats.command_buffers += 1;
            Input::Key(tmp.key)
        } else {
            input.clear();
            for &(id, k) in &in_ids {
                let p = &x.words(id)[..k * n];
                for j in 0..k {
                    input.extend_from_slice(&p[j * n + start..][..len]);
                }
            }
            Input::Words(&input)
        };
        let p = TraceParams {
            num_rays: len as u32,
            stride: len as u32,
            num_steps: steps.len() as u32,
            start_surface: t.start_surface,
            flags,
            max_iterations: t.newton.max_iterations,
            tolerance_abs: t.newton.tolerance_abs,
            tolerance_rel: t.newton.tolerance_rel,
        };
        let (mirrors, halves, rays) = (&mut *mirrors, &halves, &rays);
        launch(
            gpu,
            &dev,
            kernel,
            uniform(&p, dev.count),
            &steps,
            words,
            len,
            RES_OUT_PLANES * len,
            slots * len * 2 * RES_RECORD,
            move |batch, out, records| {
                // Output planes in kernel order, records per slot.
                let mut k = 0;
                for (&id, planes) in rays.iter().zip([RAY_FLOATS, 2, 2, 4, 3]) {
                    for j in 0..planes {
                        mirrors.store(batch, x, out, k * len, id, j * n + start, len)?;
                        k += 1;
                    }
                }
                for (half, &id) in halves.iter().enumerate() {
                    let base = half * slots * len * RES_RECORD;
                    for slot in 0..slots {
                        let from = base + slot * len * RES_RECORD;
                        let to = (slot * n + start) * RES_RECORD;
                        mirrors.store(batch, x, records, from, id, to, len * RES_RECORD)?;
                    }
                }
                Ok(())
            },
            stats,
        )?;
        start += len;
        ctl.report(start as f64 / n as f64, "resident trace");
    }
    ctl.check()?;
    Ok(Some("webgpu:trace_resident"))
}
