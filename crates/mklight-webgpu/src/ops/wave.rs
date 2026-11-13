//! Scalar-wave operations on the device: `ResidentOp::Wave`
//! (`kernels/wave.wgsl`).
//!
//! Ported: `FocusCorrelation` (the dominant cost of the direct and
//! through-focus MTF plans), `Focus`, `Prepare`, `Intensity`, `LineSpread`,
//! `Spectral` and `ImageIntegral`. The other requests (`Huygens`, `Shells`,
//! `FftMtf`, `Placement`, `Resample`, `Lattice`, `Diagnostics`,
//! `Simulation`) return `Ok(None)` and run their CPU-f32 specification.
//!
//! Each operation evaluates the specification's arithmetic
//! (`mklight_gpu::wave_ops::spec`) in its order with correctly rounded
//! division and square root, and its sums in the specification's fixed
//! blocks (`spec::SUM_BLOCK` samples per thread, then the blocks in order).
//! Results without trigonometry are therefore bitwise the CPU-f32
//! specification (focus phases, phase steps, moments, intensities, line
//! spreads, image metrics); the others differ from it only through the
//! kernels' polynomial `cis` in place of the specification's f32 sin/cos.
//!
//! The parameters are the specification's own structs (`spec::params`,
//! `spectral::params`, `image_integrals::params`) bound as read-only
//! storage. Operations whose buffers exceed a storage binding return
//! `Ok(None)`.
//!
//! Focus correlations run in batches of at most [`BATCH_TERMS`] sample
//! terms (bounded command time and cancellation latency): each batch
//! evaluates the focused phases of its planes, their phase steps, and the
//! blocked correlation sums of its outputs.

use std::sync::Arc;

use mklight_core::MkResult;
use mklight_gpu::JobStats;
use mklight_gpu::pending::JobControl;
use mklight_gpu::resident::ResidentOp;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::wave_ops::image_integrals::{self, ImageIntegral};
use mklight_gpu::wave_ops::{Request, spec, spectral};

use super::WebGpuOps;
use super::transforms::{Builder, DeviceTable, Res, grid_1d};
use crate::device::*;

const FOCUS: Kernel = Kernel::new("wave", "mkl_wave_focus");
const GAMMA: Kernel = Kernel::new("wave", "mkl_wave_gamma");
const STEP_BLOCKS: Kernel = Kernel::new("wave", "mkl_wave_step_blocks");
const STEP_FINISH: Kernel = Kernel::new("wave", "mkl_wave_step_finish");
const CORRELATION: Kernel = Kernel::new("wave", "mkl_wave_focus_correlation");
const CORRELATION_FINISH: Kernel = Kernel::new("wave", "mkl_wave_focus_correlation_finish");
const PREPARE: Kernel = Kernel::new("wave", "mkl_wave_prepare");
const INTENSITY: Kernel = Kernel::new("wave", "mkl_wave_intensity");
const LSF: Kernel = Kernel::new("wave", "mkl_wave_lsf");
const SPECTRAL: Kernel = Kernel::new("wave", "mkl_wave_spectral");
const METRICS: Kernel = Kernel::new("wave", "mkl_wave_metrics");
const METRICS_FINISH: Kernel = Kernel::new("wave", "mkl_wave_metrics_finish");
const IMAGE_COMPLEX: Kernel = Kernel::new("wave", "mkl_wave_image_complex");
const QUERY: Kernel = Kernel::new("wave", "mkl_wave_image_query");
const QUERY_FINISH: Kernel = Kernel::new("wave", "mkl_wave_image_query_finish");

/// Sample terms (outputs × pupil samples) per focus-correlation batch.
pub const BATCH_TERMS: usize = 1 << 25;

/// Lower batch and stencil-binding limits, so tests reach the batched and
/// sliced paths on small inputs (results do not depend on them).
#[doc(hidden)]
pub mod limits {
    use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
    pub static BATCH_TERMS: AtomicUsize = AtomicUsize::new(super::BATCH_TERMS);
    pub static STENCIL_BYTES: AtomicUsize = AtomicUsize::new(usize::MAX);
    pub fn batch_terms() -> usize {
        BATCH_TERMS.load(Relaxed).max(1)
    }
    pub fn stencil_bytes() -> usize {
        STENCIL_BYTES.load(Relaxed)
    }
}
/// Threads of one blocked-sum dispatch (outputs × sample blocks); its
/// block sums are scratch reused by the next dispatch.
const BLOCK_THREADS: usize = 1 << 18;

/// The wave pipelines (`prepare_kernels`).
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    use Access::*;
    let shape = |storage: &[Access]| PassShape {
        uniform: false,
        storage: storage.to_vec(),
    };
    vec![
        (FOCUS, shape(&[Read, Read, Read, Read, ReadWrite])),
        (GAMMA, shape(&[Read, ReadWrite])),
        (
            STEP_BLOCKS,
            shape(&[Read, Read, Read, ReadWrite, ReadWrite]),
        ),
        (
            STEP_FINISH,
            shape(&[Read, Read, Read, ReadWrite, ReadWrite]),
        ),
        (
            CORRELATION,
            shape(&[Read, Read, Read, Read, Read, Read, ReadWrite]),
        ),
        (CORRELATION_FINISH, shape(&[Read, Read, ReadWrite])),
        (
            PREPARE,
            shape(&[Read, Read, Read, Read, ReadWrite, ReadWrite]),
        ),
        (INTENSITY, shape(&[Read, Read, Read, ReadWrite])),
        (LSF, shape(&[Read, Read, ReadWrite])),
        (SPECTRAL, shape(&[Read, Read, Read, ReadWrite])),
        (METRICS, shape(&[Read, Read, ReadWrite, ReadWrite])),
        (METRICS_FINISH, shape(&[Read, Read, ReadWrite, ReadWrite])),
        (IMAGE_COMPLEX, shape(&[Read, Read, ReadWrite])),
        (QUERY, shape(&[Read, Read, Read, ReadWrite])),
        (QUERY_FINISH, shape(&[Read, Read, Read, ReadWrite])),
    ]
}

/// Run a wave operation of a plan on the device, or `Ok(None)` for its
/// CPU-f32 specification.
pub fn run(
    ops: &mut WebGpuOps,
    _i: usize,
    op: &ResidentOp,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    let ResidentOp::Wave(r) = op else {
        return Ok(None);
    };
    let gpu = ops.backend.gpu().clone();
    let mut d = Device {
        gpu: &gpu,
        ctl,
        stats,
    };
    match r {
        Request::FocusCorrelation { .. } => d.focus_correlation(r, x),
        Request::Focus { .. } => d.focus(r, x),
        Request::Prepare { .. } => d.prepare(r, x),
        Request::Intensity { .. } => d.intensity(r, x),
        Request::LineSpread { .. } => d.line_spread(r, x),
        Request::Spectral(s) => d.spectral(s, x),
        Request::ImageIntegral(i) => d.image_integral(i, x),
        _ => Ok(None),
    }
}

fn words<T: bytemuck::Pod>(p: &T) -> &[u32] {
    bytemuck::cast_slice(std::slice::from_ref(p))
}

struct Device<'a> {
    gpu: &'a Arc<dyn Gpu>,
    ctl: &'a JobControl,
    stats: &'a mut JobStats,
}

impl Device<'_> {
    /// Whether buffers of these word counts fit storage bindings.
    fn fits(&self, words: &[usize]) -> bool {
        let max = self.gpu.summary().max_storage_buffer_binding_size;
        words.iter().all(|&w| (w as u64) * 4 <= max)
    }

    fn submit(&mut self, mut bld: Builder<'_>, reads: Vec<(usize, &mut [u32])>) -> MkResult<()> {
        let mut batch = bld.batch();
        for (res, into) in reads {
            batch.read(res, 0, into);
        }
        let b = self.gpu.run(batch)?;
        self.stats.device_seconds += b.device_seconds;
        self.stats.command_buffers += 1;
        Ok(())
    }

    fn focus_correlation(
        &mut self,
        r: &Request,
        x: &mut Execution,
    ) -> MkResult<Option<&'static str>> {
        let Request::FocusCorrelation {
            on_axis,
            pupil,
            stencil,
            shifts,
            output,
            phase_steps,
            ..
        } = r
        else {
            unreachable!()
        };
        let base = spec::params(r);
        let count = (base.n * base.n) as usize;
        let freqs = base.field_side as usize;
        let planes = base.size as usize;
        let blocks = count.div_ceil(spec::SUM_BLOCK);
        let max_binding = self.gpu.summary().max_storage_buffer_binding_size as usize;
        // Frequencies per stencil binding (the plan's stencil may be up to
        // twice the 128 MiB binding limit).
        let chunk = (max_binding / (count.max(1) * 20)).min(freqs);
        if count == 0 || freqs == 0 || chunk == 0 || !self.fits(&[count * 2]) {
            return Ok(None);
        }
        let chunk = chunk.min((limits::stencil_bytes() / (count * 20)).max(1));
        let plane_fit = (max_binding / (count * 8)).max(1);
        // Batches: whole planes when a plane's terms fit a batch (and its
        // stencil one binding), else frequency ranges of one plane.
        let batch_terms = limits::batch_terms();
        let per_freqs = (batch_terms / count).clamp(1, chunk);
        let per_planes = if per_freqs == freqs {
            (batch_terms / (freqs * count)).clamp(1, plane_fit.min(planes))
        } else {
            1
        };
        // (stencil slice, first plane, planes, first frequency, frequencies)
        let mut units = vec![];
        for (slice, c0) in (0..freqs).step_by(chunk).enumerate() {
            let c1 = (c0 + chunk).min(freqs);
            for z0 in (0..planes).step_by(per_planes) {
                let nz = per_planes.min(planes - z0);
                for f0 in (c0..c1).step_by(per_freqs) {
                    units.push((slice, z0, nz, f0, per_freqs.min(c1 - f0)));
                }
            }
        }
        let zero = [0u32; 2];
        let tables = [
            x.view(&pupil.amplitude),
            x.view(&pupil.coverage),
            x.view(&pupil.opd),
            shifts.as_ref().map_or(&zero[..], |v| x.view(v)),
        ]
        .into_iter()
        .map(|w| DeviceTable::new(self.gpu, w))
        .collect::<MkResult<Vec<_>>>()?;
        let stencils = x
            .view(stencil)
            .chunks(chunk * count * 5)
            .map(|w| DeviceTable::new(self.gpu, w))
            .collect::<MkResult<Vec<_>>>()?;
        let mut outs = vec![*output];
        outs.extend(*phase_steps);
        let tile = (BLOCK_THREADS / blocks).max(1);
        let total = units.len();
        x.with_outputs(&outs, |_, out| -> MkResult<()> {
            let (first, rest) = out.split_at_mut(1);
            let result = &mut *first[0];
            let mut steps = rest.first_mut().map(|s| &mut **s);
            for (done, &(slice, z0, nz, f0, nf)) in units.iter().enumerate() {
                self.ctl.check()?;
                let mut bld = Builder::new(&**self.gpu);
                let [amp, cov, opd, sh] =
                    [0, 1, 2, 3].map(|k| bld.add(Res::Persistent(tables[k].key)));
                let st = bld.add(Res::Persistent(stencils[slice].key));
                let phases = bld.add(Res::Scratch(nz * count * 2));
                let mut p = base;
                (p.first, p.count) = (z0 as u32, nz as u32);
                bld.pass(
                    FOCUS,
                    words(&p),
                    &[
                        (cov, Access::Read),
                        (opd, Access::Read),
                        (sh, Access::Read),
                        (phases, Access::ReadWrite),
                    ],
                    grid_1d(nz * count),
                );
                let gamma = if *on_axis {
                    let g = bld.add(Res::Scratch(count));
                    bld.pass(
                        GAMMA,
                        words(&base),
                        &[(g, Access::ReadWrite)],
                        grid_1d(count),
                    );
                    g
                } else {
                    amp
                };
                let mut reads = vec![];
                let step_out = (f0 == 0 && steps.is_some()).then(|| {
                    let partial = bld.add(Res::Scratch(nz * blocks));
                    let s = bld.add(Res::Scratch(nz));
                    let storage = [
                        (cov, Access::Read),
                        (phases, Access::Read),
                        (partial, Access::ReadWrite),
                        (s, Access::ReadWrite),
                    ];
                    bld.pass(STEP_BLOCKS, words(&p), &storage, grid_1d(nz * blocks));
                    bld.pass(
                        STEP_FINISH,
                        words(&p),
                        &storage,
                        [nz.div_ceil(64) as u32, 1, 1],
                    );
                    s
                });
                // Outputs o0.. of this batch, contiguous (plane-major).
                let o0 = z0 * freqs + f0;
                let n_out = if nz > 1 { nz * freqs } else { nf };
                let result_res = bld.add(Res::Scratch(n_out * 2));
                let acc = bld.add(Res::Scratch(tile.min(n_out) * blocks * 6));
                for t0 in (0..n_out).step_by(tile) {
                    let nt = tile.min(n_out - t0);
                    let mut q = base;
                    q.first = (o0 + t0) as u32;
                    q.count = nt as u32;
                    q.sample_first = blocks as u32;
                    q.sample_count = (slice * chunk) as u32;
                    q.reserved = [z0 as u32, o0 as u32];
                    bld.pass(
                        CORRELATION,
                        words(&q),
                        &[
                            (amp, Access::Read),
                            (cov, Access::Read),
                            (phases, Access::Read),
                            (st, Access::Read),
                            (gamma, Access::Read),
                            (acc, Access::ReadWrite),
                        ],
                        grid_1d(nt * blocks),
                    );
                    bld.pass(
                        CORRELATION_FINISH,
                        words(&q),
                        &[(acc, Access::Read), (result_res, Access::ReadWrite)],
                        [nt.div_ceil(64) as u32, 1, 1],
                    );
                }
                reads.push((result_res, &mut result[o0 * 2..(o0 + n_out) * 2]));
                if let (Some(s), Some(dst)) = (step_out, steps.as_deref_mut()) {
                    reads.push((s, &mut dst[z0..z0 + nz]));
                }
                self.submit(bld, reads)?;
                self.ctl
                    .report((done + 1) as f64 / total as f64, "focus correlation");
            }
            Ok(())
        })?;
        Ok(Some("webgpu:focus-correlation"))
    }

    fn focus(&mut self, r: &Request, x: &mut Execution) -> MkResult<Option<&'static str>> {
        let Request::Focus {
            pupil,
            shifts,
            output,
            phase_steps,
            ..
        } = r
        else {
            unreachable!()
        };
        let base = spec::params(r);
        let count = (base.n * base.n) as usize;
        let planes = base.size as usize;
        let blocks = count.div_ceil(spec::SUM_BLOCK);
        if count == 0 || !self.fits(&[count * 2]) {
            return Ok(None);
        }
        let max_binding = self.gpu.summary().max_storage_buffer_binding_size as usize;
        let per = (max_binding / (count * 8)).clamp(1, planes.max(1));
        let mut outs = vec![*output];
        outs.extend(*phase_steps);
        x.with_outputs(&outs, |x, out| -> MkResult<()> {
            let (first, rest) = out.split_at_mut(1);
            let result = &mut *first[0];
            let mut steps = rest.first_mut().map(|s| &mut **s);
            for z0 in (0..planes).step_by(per) {
                self.ctl.check()?;
                let nz = per.min(planes - z0);
                let mut bld = Builder::new(&**self.gpu);
                let cov = bld.add(Res::Upload(x.view(&pupil.coverage)));
                let opd = bld.add(Res::Upload(x.view(&pupil.opd)));
                let sh = bld.add(Res::Upload(x.view(shifts)));
                let phases = bld.add(Res::Scratch(nz * count * 2));
                let mut p = base;
                (p.first, p.count) = (z0 as u32, nz as u32);
                bld.pass(
                    FOCUS,
                    words(&p),
                    &[
                        (cov, Access::Read),
                        (opd, Access::Read),
                        (sh, Access::Read),
                        (phases, Access::ReadWrite),
                    ],
                    grid_1d(nz * count),
                );
                let mut reads = vec![(phases, &mut result[z0 * count * 2..(z0 + nz) * count * 2])];
                if let Some(dst) = steps.as_deref_mut() {
                    let partial = bld.add(Res::Scratch(nz * blocks));
                    let s = bld.add(Res::Scratch(nz));
                    let storage = [
                        (cov, Access::Read),
                        (phases, Access::Read),
                        (partial, Access::ReadWrite),
                        (s, Access::ReadWrite),
                    ];
                    bld.pass(STEP_BLOCKS, words(&p), &storage, grid_1d(nz * blocks));
                    bld.pass(
                        STEP_FINISH,
                        words(&p),
                        &storage,
                        [nz.div_ceil(64) as u32, 1, 1],
                    );
                    reads.push((s, &mut dst[z0..z0 + nz]));
                }
                self.submit(bld, reads)?;
                self.ctl.report((z0 + nz) as f64 / planes as f64, "focus");
            }
            Ok(())
        })?;
        Ok(Some("webgpu:focus"))
    }

    fn prepare(&mut self, r: &Request, x: &mut Execution) -> MkResult<Option<&'static str>> {
        let Request::Prepare {
            pupil,
            field,
            moments,
            ..
        } = r
        else {
            unreachable!()
        };
        let mut p = spec::params(r);
        let count = (p.n * p.n) as usize;
        let field_words = x.words(*field).len();
        let side = (field_words / 2).isqrt();
        p.field_side = side as u32;
        if !self.fits(&[field_words, 3 * count, 2 * count]) {
            return Ok(None);
        }
        self.ctl.check()?;
        x.with_outputs(&[*field, *moments], |x, out| -> MkResult<()> {
            let mut bld = Builder::new(&**self.gpu);
            let amp = bld.add(Res::Upload(x.view(&pupil.amplitude)));
            let cov = bld.add(Res::Upload(x.view(&pupil.coverage)));
            let opd = bld.add(Res::Upload(x.view(&pupil.opd)));
            let f = bld.add(Res::Scratch(field_words));
            let m = bld.add(Res::Scratch(3 * count));
            bld.pass(
                PREPARE,
                words(&p),
                &[
                    (amp, Access::Read),
                    (cov, Access::Read),
                    (opd, Access::Read),
                    (f, Access::ReadWrite),
                    (m, Access::ReadWrite),
                ],
                grid_1d(side * side),
            );
            let (a, b) = out.split_at_mut(1);
            self.submit(bld, vec![(f, &mut *a[0]), (m, &mut *b[0])])
        })?;
        self.ctl.report(1.0, "wave");
        Ok(Some("webgpu:wave-prepare"))
    }

    fn intensity(&mut self, r: &Request, x: &mut Execution) -> MkResult<Option<&'static str>> {
        let Request::Intensity {
            field,
            sums,
            output,
            ..
        } = r
        else {
            unreachable!()
        };
        let p = spec::params(r);
        let size = p.size as usize;
        let field_words = field.desc.len()? * 2;
        if !self.fits(&[field_words, size * size]) {
            return Ok(None);
        }
        self.ctl.check()?;
        x.with_outputs(&[*output], |x, out| -> MkResult<()> {
            let mut bld = Builder::new(&**self.gpu);
            let f = bld.add(Res::Upload(x.view(field)));
            let s = bld.add(Res::Upload(x.view(sums)));
            let o = bld.add(Res::Scratch(size * size));
            bld.pass(
                INTENSITY,
                words(&p),
                &[(f, Access::Read), (s, Access::Read), (o, Access::ReadWrite)],
                grid_1d(size * size),
            );
            self.submit(bld, vec![(o, &mut *out[0])])
        })?;
        self.ctl.report(1.0, "wave");
        Ok(Some("webgpu:wave-intensity"))
    }

    fn line_spread(&mut self, r: &Request, x: &mut Execution) -> MkResult<Option<&'static str>> {
        let Request::LineSpread { image, output, .. } = r else {
            unreachable!()
        };
        let p = spec::params(r);
        let n = p.n as usize;
        if !self.fits(&[n * n]) {
            return Ok(None);
        }
        self.ctl.check()?;
        x.with_outputs(&[*output], |x, out| -> MkResult<()> {
            let mut bld = Builder::new(&**self.gpu);
            let a = bld.add(Res::Upload(x.view(image)));
            let o = bld.add(Res::Scratch(2 * n));
            bld.pass(
                LSF,
                words(&p),
                &[(a, Access::Read), (o, Access::ReadWrite)],
                [(2 * n).div_ceil(64) as u32, 1, 1],
            );
            self.submit(bld, vec![(o, &mut *out[0])])
        })?;
        self.ctl.report(1.0, "wave");
        Ok(Some("webgpu:line-spread"))
    }

    fn spectral(
        &mut self,
        r: &spectral::Spectral,
        x: &mut Execution,
    ) -> MkResult<Option<&'static str>> {
        let p = spectral::params(r);
        let width = if p.complex != 0 { 2 } else { 1 };
        let len = p.count as usize * width;
        if !self.fits(&[len * r.inputs.len()]) {
            return Ok(None);
        }
        self.ctl.check()?;
        // The inputs concatenated: up to 16 of them exceed the storage
        // bindings of a pass.
        let mut inputs = Vec::with_capacity(len * r.inputs.len());
        for v in &r.inputs {
            inputs.extend_from_slice(x.view(v));
        }
        x.with_outputs(&[r.output], |x, out| -> MkResult<()> {
            let mut bld = Builder::new(&**self.gpu);
            let a = bld.add(Res::Upload(&inputs));
            let f = match &r.frequencies {
                Some(v) => bld.add(Res::Upload(x.view(v))),
                None => a,
            };
            let o = bld.add(Res::Scratch(len));
            bld.pass(
                SPECTRAL,
                words(&p),
                &[(a, Access::Read), (f, Access::Read), (o, Access::ReadWrite)],
                grid_1d(p.count as usize),
            );
            self.submit(bld, vec![(o, &mut *out[0])])
        })?;
        self.ctl.report(1.0, "spectral");
        Ok(Some("webgpu:spectral"))
    }

    fn image_integral(
        &mut self,
        r: &ImageIntegral,
        x: &mut Execution,
    ) -> MkResult<Option<&'static str>> {
        let mut p = image_integrals::params(r);
        let count = (p.n * p.n) as usize;
        let blocks = count.div_ceil(spec::SUM_BLOCK);
        if !self.fits(&[count * 2]) {
            return Ok(None);
        }
        self.ctl.check()?;
        match r {
            ImageIntegral::Metrics { input, output } => {
                x.with_outputs(&[*output], |x, out| -> MkResult<()> {
                    let mut bld = Builder::new(&**self.gpu);
                    let a = bld.add(Res::Upload(x.view(input)));
                    let part = bld.add(Res::Scratch(blocks * 10));
                    let o = bld.add(Res::Scratch(10));
                    let storage = [
                        (a, Access::Read),
                        (part, Access::ReadWrite),
                        (o, Access::ReadWrite),
                    ];
                    bld.pass(METRICS, words(&p), &storage, grid_1d(blocks));
                    bld.pass(METRICS_FINISH, words(&p), &storage, [1, 1, 1]);
                    self.submit(bld, vec![(o, &mut *out[0])])
                })?;
            }
            ImageIntegral::Complex { input, output, .. } => {
                x.with_outputs(&[*output], |x, out| -> MkResult<()> {
                    let mut bld = Builder::new(&**self.gpu);
                    let a = bld.add(Res::Upload(x.view(input)));
                    let o = bld.add(Res::Scratch(count * 2));
                    bld.pass(
                        IMAGE_COMPLEX,
                        words(&p),
                        &[(a, Access::Read), (o, Access::ReadWrite)],
                        grid_1d(count),
                    );
                    self.submit(bld, vec![(o, &mut *out[0])])
                })?;
            }
            ImageIntegral::Query {
                input,
                queries,
                output,
                ..
            } => {
                let nq = p.queries as usize;
                // Queries per batch (bounded command time) and per dispatch
                // (bounded block sums).
                let per_batch = (limits::batch_terms() / count.max(1)).clamp(1, nq.max(1));
                let tile = (BLOCK_THREADS / blocks.max(1)).max(1);
                x.with_outputs(&[*output], |x, out| -> MkResult<()> {
                    let input = DeviceTable::new(self.gpu, x.view(input))?;
                    let queries = DeviceTable::new(self.gpu, x.view(queries))?;
                    for q0 in (0..nq).step_by(per_batch) {
                        self.ctl.check()?;
                        let nb = per_batch.min(nq - q0);
                        let mut bld = Builder::new(&**self.gpu);
                        let a = bld.add(Res::Persistent(input.key));
                        let q = bld.add(Res::Persistent(queries.key));
                        let acc = bld.add(Res::Scratch(tile.min(nb) * blocks * 4));
                        let o = bld.add(Res::Scratch(nq * 2));
                        for t0 in (0..nb).step_by(tile) {
                            (p.first, p.count) = ((q0 + t0) as u32, tile.min(nb - t0) as u32);
                            bld.pass(
                                QUERY,
                                words(&p),
                                &[
                                    (a, Access::Read),
                                    (q, Access::Read),
                                    (acc, Access::ReadWrite),
                                ],
                                grid_1d(p.count as usize * blocks),
                            );
                            bld.pass(
                                QUERY_FINISH,
                                words(&p),
                                &[
                                    (q, Access::Read),
                                    (acc, Access::Read),
                                    (o, Access::ReadWrite),
                                ],
                                [p.count.div_ceil(64), 1, 1],
                            );
                        }
                        let mut part = vec![0u32; nq * 2];
                        self.submit(bld, vec![(o, &mut part[..])])?;
                        out[0][q0 * 2..(q0 + nb) * 2].copy_from_slice(&part[q0 * 2..(q0 + nb) * 2]);
                        self.ctl
                            .report((q0 + nb) as f64 / nq as f64, "image integral");
                    }
                    Ok(())
                })?;
            }
        }
        self.ctl.report(1.0, "image integral");
        Ok(Some("webgpu:image-integral"))
    }
}
