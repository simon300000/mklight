//! Regular transforms on the device: `ResidentOp::Transforms`
//! (`kernels/transforms.wgsl`).
//!
//! A transform runs the backend-neutral recipe of
//! `mklight_gpu::transforms::recipe` (the steps the CPU-f32 specification
//! evaluates and Metal encodes) as one device batch: the operation's inputs
//! are uploaded, every recipe step is one or more passes over batch scratch
//! (buffers are reused once their last reader ran, `Recipe::liveness`), and
//! only the output is read back. Twiddle and chirp-factor tables (f64 host
//! tables) and chirp spectra are persistent device buffers in the
//! execution's operation cache, content-addressed by the recipe's keys like
//! Metal's.
//!
//! WebGPU differs from Metal in its limits only:
//!
//! * 16 KiB of workgroup storage hold rows of up to [`LOCAL_ROW`] = 2048
//!   points (Metal: 4096). Longer FFT rows run the global Stockham stages
//!   (`recipe::global_stages`) and longer fused chirp rows the same
//!   arithmetic unfused: table pre-chirp, global stages, broadcast spectrum
//!   product, global stages, table post-chirp.
//! * A storage binding holds at most `maxStorageBufferBindingSize` (128 MiB:
//!   a 4096² complex grid). An operation with a larger buffer, or a
//!   correlation grid beyond the dispatch limits, returns `Ok(None)` and runs
//!   its CPU-f32 specification.
//!
//! Correlations run in frequency tiles (≤4M samples·frequencies, as Metal):
//! terms, the double-float reduction of `reduce.wgsl` and normalisation, one
//! batch per tile over persistent copies of the inputs.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use mklight_core::MkResult;
use mklight_gpu::JobStats;
use mklight_gpu::consts::*;
use mklight_gpu::pending::JobControl;
use mklight_gpu::reduce;
use mklight_gpu::resident::ResidentOp;
use mklight_gpu::resident::cpu::Execution;
use mklight_gpu::resident::immutable::CacheCategory::TransformPlans;
use mklight_gpu::transforms::correlation;
use mklight_gpu::transforms::recipe::*;
use mklight_gpu::transforms::{Request, accesses};

use super::WebGpuOps;
use crate::device::*;

/// Longest row transformed in one shared-memory pass (16 KiB of workgroup
/// storage: 2048 complex f32).
pub const LOCAL_ROW: usize = 2048;
/// Pupil samples × frequencies per correlation batch (Metal's tile).
const TILE: usize = 1 << 22;
/// Cached chirp spectra are at most this large (the operation cache's entry
/// limit); larger ones are recomputed as batch scratch.
const CACHE_ENTRY_BYTES: usize = 64 << 20;

const ELEMENTWISE: Kernel = Kernel::new("transforms", "mkl_transform");
const CHIRP_PRE: Kernel = Kernel::new("transforms", "mkl_chirp_pre");
const CHIRP_POST: Kernel = Kernel::new("transforms", "mkl_chirp_post");
const TRANSPOSE: Kernel = Kernel::new("transforms", "mkl_transform_transpose");
const FFT_ROWS: Kernel = Kernel::new("transforms", "mkl_fft_rows");
const CHIRP_ROWS: Kernel = Kernel::new("transforms", "mkl_chirp_rows");
const FFT_STAGE: Kernel = Kernel::new("transforms", "mkl_fft_stage");
const STENCIL: Kernel = Kernel::new("transforms", "mkl_optical_stencil");
const TERMS: Kernel = Kernel::new("transforms", "mkl_correlation_terms");
const NORMALIZE: Kernel = Kernel::new("transforms", "mkl_correlation_normalize");
const REDUCE_PASS1: Kernel = Kernel::new("reduce", "mkl_reduce_pass1");
const REDUCE_PASS2: Kernel = Kernel::new("reduce", "mkl_reduce_pass2");

/// The transform pipelines (`prepare_kernels`); the reduction passes the
/// correlations use are the backend's own.
pub fn pipelines() -> Vec<(Kernel, PassShape)> {
    use Access::*;
    let shape = |storage: &[Access]| PassShape {
        uniform: false,
        storage: storage.to_vec(),
    };
    vec![
        (ELEMENTWISE, shape(&[Read, Read, Read, ReadWrite])),
        (CHIRP_PRE, shape(&[Read, Read, Read, ReadWrite])),
        (CHIRP_POST, shape(&[Read, Read, Read, ReadWrite])),
        (TRANSPOSE, shape(&[Read, Read, ReadWrite])),
        (FFT_ROWS, shape(&[Read, Read, Read, ReadWrite])),
        (FFT_STAGE, shape(&[Read, Read, Read, ReadWrite])),
        (
            CHIRP_ROWS,
            shape(&[Read, Read, Read, ReadWrite, Read, Read]),
        ),
        (STENCIL, shape(&[Read, Read, ReadWrite])),
        (
            TERMS,
            shape(&[Read, Read, Read, Read, Read, Read, ReadWrite]),
        ),
        (NORMALIZE, shape(&[Read, Read, ReadWrite])),
    ]
}

/// Run a transform operation of a plan on the device, or `Ok(None)` for its
/// CPU-f32 specification.
pub fn run(
    ops: &mut WebGpuOps,
    _i: usize,
    op: &ResidentOp,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
) -> MkResult<Option<&'static str>> {
    let ResidentOp::Transforms(r) = op else {
        return Ok(None);
    };
    let gpu = ops.backend.gpu().clone();
    match r {
        Request::Correlation(_) => correlate(&gpu, r, x, ctl, stats, "webgpu:correlation"),
        Request::OpticalCorrelation(_) => {
            correlate(&gpu, r, x, ctl, stats, "webgpu:optical-correlation")
        }
        Request::PrepareOpticalStencil(_) => {
            correlate(&gpu, r, x, ctl, stats, "webgpu:optical-stencil")
        }
        Request::Transform(_) => transform(&gpu, r, x, ctl, stats, "webgpu:transform"),
        Request::Transpose { .. } => transform(&gpu, r, x, ctl, stats, "webgpu:transpose"),
        Request::Convolve { .. } => transform(&gpu, r, x, ctl, stats, "webgpu:convolve"),
        Request::Phase { .. } => transform(&gpu, r, x, ctl, stats, "webgpu:phase"),
    }
}

/// A persistent read-only device buffer, released when dropped (an
/// operation-cache entry or an operation's input copy).
pub struct DeviceTable {
    gpu: Arc<dyn Gpu>,
    key: BufferKey,
}

impl DeviceTable {
    fn new(gpu: &Arc<dyn Gpu>, words: &[u32]) -> MkResult<Self> {
        Ok(Self {
            gpu: gpu.clone(),
            key: gpu.persist(words)?,
        })
    }
}

impl Drop for DeviceTable {
    fn drop(&mut self) {
        self.gpu.release(self.key);
    }
}

/// The cached device table `name` (content-addressed), built by `fill`.
fn table(
    gpu: &Arc<dyn Gpu>,
    x: &Execution,
    name: String,
    fill: impl FnOnce() -> Vec<[f32; 2]>,
) -> MkResult<Arc<DeviceTable>> {
    let key = x.content_key(TransformPlans, name);
    if let Some(t) = x.cached::<DeviceTable>(&key)? {
        return Ok(t);
    }
    let built = Instant::now();
    let v = fill();
    let t = Arc::new(DeviceTable::new(gpu, bytemuck::cast_slice(&v))?);
    x.cache(
        key,
        (v.len() * 8).max(16),
        built.elapsed().as_secs_f64(),
        t.clone(),
    );
    Ok(t)
}

/// A buffer of the batch being assembled.
enum Res<'a> {
    Upload(&'a [u32]),
    Scratch(usize),
    Persistent(BufferKey),
}

/// A batch under construction: resources, passes and the parameter block
/// every pass binds a range of (binding 2).
struct Builder<'a> {
    res: Vec<Res<'a>>,
    passes: Vec<Pass>,
    params: Vec<u32>,
    /// Words between parameter records (the storage offset alignment).
    stride: usize,
    /// Free scratch (resource, words), reused first-fit.
    free: Vec<(usize, usize)>,
}

impl<'a> Builder<'a> {
    fn new(gpu: &dyn Gpu) -> Self {
        let align = gpu.summary().min_storage_buffer_offset_alignment as usize / 4;
        Self {
            // Resource 0 is the parameter block, uploaded when finished.
            res: vec![Res::Scratch(0)],
            passes: vec![],
            params: vec![],
            stride: align.max(1),
            free: vec![],
        }
    }
    fn add(&mut self, r: Res<'a>) -> usize {
        self.res.push(r);
        self.res.len() - 1
    }
    /// Scratch of at least `words`, reusing a freed buffer.
    fn take(&mut self, words: usize) -> usize {
        if let Some(at) = self.free.iter().position(|&(_, w)| w >= words) {
            return self.free.swap_remove(at).0;
        }
        self.add(Res::Scratch(words))
    }
    fn release(&mut self, r: usize) {
        if let Res::Scratch(w) = self.res[r] {
            self.free.push((r, w));
        }
    }
    /// A pass binding `params` first, then `storage` (resource, access).
    fn pass(&mut self, kernel: Kernel, params: &[u32], storage: &[(usize, Access)], g: [u32; 3]) {
        let off = self.params.len().div_ceil(self.stride) * self.stride;
        self.params.resize(off, 0);
        self.params.extend_from_slice(params);
        let mut bindings = vec![Binding {
            resource: 0,
            access: Access::Read,
            range: Some((off, params.len())),
        }];
        bindings.extend(storage.iter().map(|&(resource, access)| Binding {
            resource,
            access,
            range: None,
        }));
        self.passes.push(Pass {
            kernel,
            uniform: None,
            storage: bindings,
            workgroups: g,
        });
    }
    /// The batch: the parameter block, then the resources in order.
    fn batch(&mut self) -> Batch<'_> {
        let passes = std::mem::take(&mut self.passes);
        let mut b = Batch::default();
        for (i, r) in self.res.iter().enumerate() {
            match r {
                _ if i == 0 => b.upload(&self.params),
                Res::Upload(w) => b.upload(w),
                Res::Scratch(w) => b.scratch(*w),
                Res::Persistent(k) => b.persistent(*k),
            };
        }
        b.passes = passes;
        b
    }
}

/// Groups of 256 covering `count` elements (`p2_linear`): x up to the
/// dispatch limit, y the rest.
fn grid_1d(count: usize) -> [u32; 3] {
    let groups = count.div_ceil(256).max(1);
    let gx = groups.min(65535);
    [gx as u32, groups.div_ceil(gx) as u32, 1]
}

fn param_words(p: &Params) -> &[u32] {
    bytemuck::cast_slice(std::slice::from_ref(p))
}

/// The global Stockham stages of an unnormalised FFT of `h` rows of `n`
/// points, ping-ponging `cur` → `other`; the result ends in `cur`.
fn stages(
    bld: &mut Builder,
    n: usize,
    h: usize,
    sign: i32,
    tw: usize,
    cur: &mut usize,
    other: &mut usize,
) {
    for (q, _) in global_stages(&Params::fft(n, h, sign)) {
        let r = q.radices[0] as usize;
        bld.pass(
            FFT_STAGE,
            param_words(&q),
            &[
                (tw, Access::Read),
                (*cur, Access::Read),
                (*other, Access::ReadWrite),
            ],
            [(n / r).div_ceil(STAGE_THREADS) as u32, h as u32, 1],
        );
        std::mem::swap(cur, other);
    }
}

/// Whether every buffer of the recipe fits a storage binding.
fn fits(recipe: &Recipe, inputs: &[usize], max_binding: u64) -> bool {
    let ok = |elements: usize| (elements as u64) * 8 <= max_binding;
    inputs.iter().all(|&n| ok(n))
        && recipe.steps.iter().all(|s| {
            ok(s.p.count as usize)
                && (!matches!(s.p.mode, MKL_TRANSFORM_FFT_ROWS | MKL_TRANSFORM_CHIRP_ROWS)
                    || ok(s.p.width as usize * s.p.height as usize))
        })
}

fn transform(
    gpu: &Arc<dyn Gpu>,
    r: &Request,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
    name: &'static str,
) -> MkResult<Option<&'static str>> {
    let access = accesses(r);
    let key = x.content_key(TransformPlans, format!("transform-recipe:{r:?}"));
    let recipe = match x.cached::<Recipe>(&key)? {
        Some(v) => v,
        None => {
            let built = Instant::now();
            let v = Arc::new(request(r)?);
            let bytes = v.steps.len() * size_of::<Step>();
            x.cache(key, bytes, built.elapsed().as_secs_f64(), v.clone());
            v
        }
    };
    let input_lens: Vec<usize> = access
        .reads
        .iter()
        .map(|v| v.desc.len())
        .collect::<MkResult<_>>()?;
    if !fits(
        &recipe,
        &input_lens,
        gpu.summary().max_storage_buffer_binding_size,
    ) {
        return Ok(None);
    }
    ctl.check()?;
    let steps = &recipe.steps;
    let names = recipe.cache_keys();
    // The CPU-f32 specification caches host spectra under the recipe's own
    // names; device copies get theirs.
    let spectrum_key =
        |x: &Execution, i: usize| x.content_key(TransformPlans, format!("webgpu:{}", names[i]));
    let mut held: HashMap<usize, Arc<DeviceTable>> = HashMap::new();
    let Liveness { active, mut uses } = recipe.liveness(|i| {
        Ok(match x.cached::<DeviceTable>(&spectrum_key(x, i))? {
            Some(t) => {
                held.insert(i, t);
                true
            }
            None => false,
        })
    })?;
    // Immutable f64-derived tables: twiddles per row length, chirp factors
    // per fused chirp step.
    let mut twiddle_tables: HashMap<usize, Arc<DeviceTable>> = HashMap::new();
    let mut chirp_tables: HashMap<usize, Arc<DeviceTable>> = HashMap::new();
    for (i, s) in steps.iter().enumerate().filter(|(i, _)| active[*i]) {
        let w = s.p.width as usize;
        if matches!(s.p.mode, MKL_TRANSFORM_FFT_ROWS | MKL_TRANSFORM_CHIRP_ROWS)
            && !twiddle_tables.contains_key(&w)
        {
            let t = table(gpu, x, format!("webgpu:p2-twiddles:{w}"), || twiddles(w))?;
            twiddle_tables.insert(w, t);
        }
        if s.p.mode == MKL_TRANSFORM_CHIRP_ROWS {
            let t = table(gpu, x, format!("webgpu:{}:chirp", names[i]), || {
                chirp_factors(&s.p)
            })?;
            chirp_tables.insert(i, t);
        }
    }
    let output = access.writes[0];
    let reads = access.reads.clone();
    let (b, spectra) = x.with_outputs(&[output], |x, out| -> MkResult<_> {
        let mut bld = Builder::new(&**gpu);
        let first = bld.add(Res::Upload(x.view(&reads[0])));
        let second = if reads.len() > 1 {
            bld.add(Res::Upload(x.view(&reads[1])))
        } else {
            first
        };
        let mut slot = vec![usize::MAX; steps.len() + 2];
        (slot[0], slot[1]) = (first, second);
        for (&i, t) in &held {
            slot[i + 2] = bld.add(Res::Persistent(t.key));
        }
        let tw: HashMap<usize, usize> = twiddle_tables
            .iter()
            .map(|(&w, t)| (w, bld.add(Res::Persistent(t.key))))
            .collect();
        let chirp: HashMap<usize, usize> = chirp_tables
            .iter()
            .map(|(&i, t)| (i, bld.add(Res::Persistent(t.key))))
            .collect();
        // Fresh spectra to read back and cache: (step, resource).
        let mut fresh = vec![];
        use Access::{Read as R, ReadWrite as W};
        for (i, s) in steps.iter().enumerate() {
            if !active[i] {
                continue;
            }
            let p = s.p;
            let (n, h) = (p.width as usize, p.height as usize);
            let words = p.count as usize * 2;
            let keep = s.cache && words * 4 <= CACHE_ENTRY_BYTES;
            let dst = if keep {
                bld.add(Res::Scratch(words))
            } else {
                bld.take(words)
            };
            if keep {
                fresh.push((i, dst));
            }
            slot[i + 2] = dst;
            let a = slot[s.a];
            let mut temps = vec![];
            match p.mode {
                MKL_TRANSFORM_FFT_ROWS if n > LOCAL_ROW => {
                    let tmp = bld.take(words);
                    temps.push(tmp);
                    let mut src = a;
                    for (q, into_dst) in global_stages(&p) {
                        let to = if into_dst { dst } else { tmp };
                        let r = q.radices[0] as usize;
                        bld.pass(
                            FFT_STAGE,
                            param_words(&q),
                            &[(tw[&n], R), (src, R), (to, W)],
                            [(n / r).div_ceil(STAGE_THREADS) as u32, h as u32, 1],
                        );
                        src = to;
                    }
                }
                MKL_TRANSFORM_FFT_ROWS => {
                    let (_, _, g) = row_geometry(LOCAL_ROW, n);
                    bld.pass(
                        FFT_ROWS,
                        param_words(&p),
                        &[(tw[&n], R), (a, R), (dst, W)],
                        [h.div_ceil(g) as u32, 1, 1],
                    );
                }
                MKL_TRANSFORM_CHIRP_ROWS if n > LOCAL_ROW => {
                    // The fused row's arithmetic in global passes.
                    let padded = n * h * 2;
                    let (mut cur, mut other) = (bld.take(padded), bld.take(padded));
                    temps.extend([cur, other]);
                    let mut q = p;
                    q.count = (n * h) as u32;
                    bld.pass(
                        CHIRP_PRE,
                        param_words(&q),
                        &[(a, R), (chirp[&i], R), (cur, W)],
                        grid_1d(n * h),
                    );
                    stages(&mut bld, n, h, -1, tw[&n], &mut cur, &mut other);
                    let mut m = Params::new(MKL_TRANSFORM_MULTIPLY, n, h);
                    m.flags = BROADCAST;
                    bld.pass(
                        ELEMENTWISE,
                        param_words(&m),
                        &[(cur, R), (slot[s.b], R), (other, W)],
                        grid_1d(n * h),
                    );
                    (cur, other) = (other, cur);
                    stages(&mut bld, n, h, 1, tw[&n], &mut cur, &mut other);
                    bld.pass(
                        CHIRP_POST,
                        param_words(&p),
                        &[(cur, R), (chirp[&i], R), (dst, W)],
                        grid_1d(p.count as usize),
                    );
                }
                MKL_TRANSFORM_CHIRP_ROWS => {
                    let (_, _, g) = row_geometry(LOCAL_ROW, n);
                    bld.pass(
                        CHIRP_ROWS,
                        param_words(&p),
                        &[
                            (tw[&n], R),
                            (a, R),
                            (dst, W),
                            (slot[s.b], R),
                            (chirp[&i], R),
                        ],
                        [h.div_ceil(g) as u32, 1, 1],
                    );
                }
                MKL_TRANSFORM_TRANSPOSE => {
                    bld.pass(
                        TRANSPOSE,
                        param_words(&p),
                        &[(a, R), (dst, W)],
                        [n.div_ceil(32) as u32, h.div_ceil(32) as u32, 1],
                    );
                }
                _ => {
                    bld.pass(
                        ELEMENTWISE,
                        param_words(&p),
                        &[(a, R), (slot[s.b], R), (dst, W)],
                        grid_1d(p.count as usize),
                    );
                }
            }
            for d in dependencies(s) {
                uses[d] -= 1;
                let r = slot[d];
                if d >= 2
                    && uses[d] == 0
                    && !held.contains_key(&(d - 2))
                    && !fresh.iter().any(|&(_, f)| f == r)
                {
                    bld.release(r);
                }
            }
            for t in temps {
                bld.release(t);
            }
        }
        let mut spectra: Vec<(usize, Vec<u32>)> = fresh
            .iter()
            .map(|&(i, _)| (i, vec![0u32; steps[i].p.count as usize * 2]))
            .collect();
        let mut batch = bld.batch();
        for ((_, res), (_, v)) in fresh.iter().zip(spectra.iter_mut()) {
            batch.read(*res, 0, v);
        }
        batch.read(slot[recipe.output], 0, out[0]);
        let b = gpu.run(batch)?;
        Ok((b, spectra))
    })?;
    stats.device_seconds += b.device_seconds;
    stats.command_buffers += 1;
    for (i, v) in spectra {
        let t = Arc::new(DeviceTable::new(gpu, &v)?);
        x.cache(spectrum_key(x, i), v.len() * 4, b.device_seconds, t);
    }
    ctl.report(1.0, "transform");
    Ok(Some(name))
}

fn correlate(
    gpu: &Arc<dyn Gpu>,
    r: &Request,
    x: &mut Execution,
    ctl: &JobControl,
    stats: &mut JobStats,
    name: &'static str,
) -> MkResult<Option<&'static str>> {
    let mut p = correlation::params(r);
    let count = (p.n * p.height) as usize;
    let (freq, output) = match r {
        Request::Correlation(c) => (&c.frequencies, c.output),
        Request::OpticalCorrelation(c) | Request::PrepareOpticalStencil(c) => {
            (&c.frequencies, c.output)
        }
        _ => unreachable!(),
    };
    let nf = freq.desc.shape[0];
    let s = gpu.summary();
    let max_binding = s.max_storage_buffer_binding_size;
    let max_groups = s.max_compute_workgroups_per_dimension as usize;
    // Grids beyond one binding or one dispatch row run the specification.
    if (count as u64) * 8 > max_binding || count.div_ceil(256) > max_groups {
        return Ok(None);
    }
    // Frequencies per batch: Metal's tile, three reduction channels each
    // within the dispatch limit.
    let tile = (TILE / count.max(1))
        .clamp(1, nf.max(1))
        .min(max_groups / 3);
    let stencil = matches!(r, Request::PrepareOpticalStencil(_));
    // Persistent copies of the inputs for every tile's batch.
    let views = match r {
        Request::Correlation(c) => {
            let cov = c.coverage.as_ref().unwrap_or(&c.pupil);
            vec![&c.pupil, &c.pupil, cov, freq]
        }
        Request::OpticalCorrelation(c) => vec![&c.amplitude, &c.opd, &c.coverage, freq],
        Request::PrepareOpticalStencil(_) => vec![freq],
        _ => unreachable!(),
    };
    let mut copies: Vec<DeviceTable> = vec![];
    for v in &views {
        copies.push(DeviceTable::new(gpu, x.view(v))?);
    }
    let prepared = match r {
        Request::OpticalCorrelation(c) => c.stencil.clone(),
        _ => None,
    };
    let groups = reduce::num_groups(count);
    let out_words = if stencil { count * 5 } else { 2 };
    x.with_outputs(&[output], |x, out| -> MkResult<()> {
        let out = &mut *out[0];
        for first in (0..nf).step_by(tile) {
            ctl.check()?;
            let nt = tile.min(nf - first);
            (p.first, p.frequencies) = (first as u32, nt as u32);
            let params: &[u32] = bytemuck::cast_slice(std::slice::from_ref(&p));
            let mut batch = Batch::default();
            let pr = batch.upload(params);
            let dst = &mut out[first * out_words..(first + nt) * out_words];
            if stencil {
                let f = batch.persistent(copies[0].key);
                let d = batch.scratch(nt * count * 5);
                batch.passes.push(Pass {
                    kernel: STENCIL,
                    uniform: None,
                    storage: vec![Binding::read(pr), Binding::read(f), Binding::write(d)],
                    workgroups: [count.div_ceil(256) as u32, nt as u32, 1],
                });
                batch.read(d, 0, dst);
            } else {
                let [field, opd, cov, f] = [0, 1, 2, 3].map(|k| batch.persistent(copies[k].key));
                let st = match &prepared {
                    Some(v) => {
                        let w = &x.view(v)[first * count * 5..(first + nt) * count * 5];
                        batch.upload(w)
                    }
                    None => field,
                };
                let channels = 3 * nt;
                let terms = batch.scratch(channels * count);
                let partials = batch.scratch(channels * groups * 2);
                let sums = batch.scratch(channels * 2);
                let result = batch.scratch(nt * 2);
                batch.passes.push(Pass {
                    kernel: TERMS,
                    uniform: None,
                    storage: [pr, field, opd, cov, f, st]
                        .into_iter()
                        .map(Binding::read)
                        .chain([Binding::write(terms)])
                        .collect(),
                    workgroups: [count.div_ceil(256) as u32, nt as u32, 1],
                });
                let rp = uniform_words(&[
                    count as u32,
                    count as u32,
                    groups as u32,
                    0,
                    MKL_REDUCE_DF,
                    0,
                    0,
                    0,
                ]);
                let reduce = vec![
                    Binding::read(terms),
                    Binding::write(partials),
                    Binding::write(sums),
                ];
                batch.passes.push(Pass {
                    kernel: REDUCE_PASS1,
                    uniform: Some(rp.clone()),
                    storage: reduce.clone(),
                    workgroups: [groups as u32, channels as u32, 1],
                });
                batch.passes.push(Pass {
                    kernel: REDUCE_PASS2,
                    uniform: Some(rp),
                    storage: reduce,
                    workgroups: [1, channels as u32, 1],
                });
                batch.passes.push(Pass {
                    kernel: NORMALIZE,
                    uniform: None,
                    storage: vec![
                        Binding::read(pr),
                        Binding::read(sums),
                        Binding::write(result),
                    ],
                    workgroups: [nt.div_ceil(64) as u32, 1, 1],
                });
                batch.read(result, 0, dst);
            }
            let b = gpu.run(batch)?;
            stats.device_seconds += b.device_seconds;
            stats.command_buffers += 1;
            ctl.report((first + nt) as f64 / nf as f64, "correlation");
        }
        Ok(())
    })?;
    Ok(Some(name))
}
