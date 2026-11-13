//! Pupil-aiming designs of the resident geometric adapters
//! ([`Route::aiming`]).
//!
//! Every generation of a [`Geometry`] (grid rays, chief rays, layout fans,
//! the sizing and field-map grids built on `append_grid`) is aimed before
//! its plan is built, by the design of the adapter's route
//! ([`Geometry::with_route`]):
//!
//! * [`AimDesign::HostTable`] (default): the reference tracer aims every ray
//!   in f64 and writes the starting rays straight into the shared ray planes
//!   the device trace reads (four `fill_buffer` buffers filled in place by
//!   the rayon workers; no sample or field records). Generations of at least
//!   [`HOST_TABLE_MIN_PER_ROW`] rays per field row first solve a coarse f64
//!   seed table per row ([`HostTable`]) and start every ray's Newton solve
//!   from its interpolated aim and stop Jacobian
//!   (`SequentialTracer::generate_many_from`; the reference schedule takes
//!   over where a started solve does not converge). Paraxial launches
//!   (aiming off) need no solve: the device generates them inside the plan.
//!
//!   **Split generations.** In the analyses that opt in
//!   ([`Geometry::split_large_generations`]: spot diagram and geometric
//!   MTF), a generation with a seed table and at least [`SPLIT_MIN_WORK`]
//!   rays × surfaces is aimed by the host and the device together
//!   ([`Geometry::aim_split`]): a fixed, evenly interleaved share of its
//!   tiles ([`Route::device_share`], by default [`SPLIT_DEVICE_SHARE`]) goes
//!   to the device's double-float Newton solve, seeded from the same table,
//!   except for the 128-ray chunks with a ray outside the table's solved
//!   nodes; the rest goes to the host's rayon workers. The assignment
//!   depends only on the generation and the share, never on timing, so
//!   identical input gives identical rays.
//! * [`AimDesign::Table`]: a coarse host f64 aim map per field row (a seed
//!   table whose side follows the rays per row) and device Newton
//!   corrections of every ray from its interpolated seeds, with a bounded
//!   schedule, followed by host completion.
//! * [`AimDesign::Device`]: double-float Newton–Broyden on the device from
//!   host chief rows (and a device seed table for large generations), in its
//!   own plan, followed by host completion.
//!
//! **Host completion.** A device aim that does not certify the requested
//! stop tolerance (`MKL_AIM_FAILED`) is aimed again in f64 by the reference
//! tracer; a ray it aims replaces the device's, one it cannot aim stays
//! failed exactly as on the CPU. Hard field failures are the host rows' own
//! and are kept. With the double-float even-asphere slope of the trace, the
//! device aims converge except for rays the reference cannot aim either.

use super::*;
use mklight_core::RayAiming;
use mklight_core::platform::Instant;
use mklight_gpu::{consts::*, resident};
use mklight_trace::{AimReport, FieldPoint, GeneratedRay, PreparedField, RayRequest};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc;

/// How a [`Geometry`] aims its generations (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AimDesign {
    #[default]
    HostTable,
    Table,
    Device,
}
impl AimDesign {
    /// The design's name in the run evidence (`aiming.design`).
    pub fn name(self) -> &'static str {
        match self {
            Self::HostTable => "host-table",
            Self::Table => "table",
            Self::Device => "device",
        }
    }
}

/// One ray of a generation: its field row, requested (pre-vignetting) pupil
/// point, weight and logical ID.
#[derive(Clone, Copy, Debug)]
pub struct RaySpec {
    pub row: u32,
    pub pupil: [f64; 2],
    pub weight: f32,
    pub id: u64,
}

/// The field rows of a generation, prepared by the reference tracer of its
/// aiming mode: system fields are the tracer's own, other points are
/// prepared once each.
#[derive(Clone, Debug)]
pub struct AimRows {
    pub aiming: RayAiming,
    /// Unique prepared field points.
    pub fields: Arc<Vec<PreparedField>>,
    /// (index into `fields`, system wavelength) of every row.
    pub rows: Vec<(usize, usize)>,
}
impl AimRows {
    pub fn field(&self, row: usize) -> &PreparedField {
        &self.fields[self.rows[row].0]
    }
    pub fn wavelength(&self, row: usize) -> usize {
        self.rows[row].1
    }
    pub fn len(&self) -> usize {
        self.rows.len()
    }
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }
}

/// Aiming work of a [`Geometry`] since its last plan, attached to that plan's
/// run evidence (`run_info.settings.aiming`).
#[derive(Clone, Debug, Default, serde::Serialize)]
pub struct AimEvidence {
    pub design: &'static str,
    /// The device's share of split generations (0 where the adapter aims
    /// every generation of the design on one side).
    pub device_share: f64,
    pub generations: u64,
    pub rays: u64,
    /// Rays aimed by the reference tracer.
    pub host_rays: u64,
    pub host_failed: u64,
    pub host_iterations: u64,
    pub host_seconds: f64,
    /// Rays aimed or launched on the device, their unconverged aims and
    /// Newton iterations.
    pub device_rays: u64,
    pub device_failed: u64,
    pub device_iterations: u64,
    /// Device aiming plans and their wall seconds (submission to completion).
    pub device_plans: u64,
    pub device_seconds: f64,
    /// Device failures the host aimed again, and how many it resolved.
    pub completion_rays: u64,
    pub completion_resolved: u64,
    pub completion_seconds: f64,
    /// Host seed-table nodes and their seconds.
    pub table_nodes: u64,
    pub table_seconds: f64,
}
impl AimEvidence {
    fn host(&mut self, s: &HostAims) {
        self.host_rays += s.rays;
        self.host_failed += s.failed;
        self.host_iterations += s.iterations;
    }
}

/// Counters of host-aimed rays.
#[derive(Clone, Copy, Debug, Default)]
struct HostAims {
    rays: u64,
    failed: u64,
    iterations: u64,
}
impl HostAims {
    fn add(mut self, o: Self) -> Self {
        self.rays += o.rays;
        self.failed += o.failed;
        self.iterations += o.iterations;
        self
    }
}

/// Rays per host aiming task (one `generate_many` batch).
const HOST_CHUNK: usize = 128;
/// Planes of the resident ray layout (`ResidentRays`).
const FLOAT_PLANES: usize = 15;

/// Column block `[start, start + len)` of plane-major resident ray storage:
/// one mutable slice per plane of each buffer.
pub(crate) struct RayColumns<'a> {
    pub start: usize,
    floats: Vec<&'a mut [f32]>,
    indices: Vec<&'a mut [u32]>,
    ids: Vec<&'a mut [u32]>,
    report: Vec<&'a mut [u32]>,
}
impl RayColumns<'_> {
    pub fn len(&self) -> usize {
        self.report[0].len()
    }
    /// Write ray `k` of the block: the double-float starting ray (zero for
    /// an unaimed ray), its system/wavelength slot, logical ID and report.
    fn write(
        &mut self,
        k: usize,
        ray: Option<(&mklight_trace::Ray, f32)>,
        slot: (u32, u32),
        id: u64,
        report: [u32; 4],
    ) {
        let mut planes = [0f32; FLOAT_PLANES];
        if let Some((r, weight)) = ray {
            let p = [r.pos.x, r.pos.y, r.pos.z].map(Df32::from_f64);
            let d = [r.dir.x, r.dir.y, r.dir.z].map(Df32::from_f64);
            for j in 0..3 {
                planes[j] = p[j].hi;
                planes[3 + j] = d[j].hi;
                planes[9 + j] = p[j].lo;
                planes[12 + j] = d[j].lo;
            }
            planes[8] = weight;
        }
        for (plane, v) in self.floats.iter_mut().zip(planes) {
            plane[k] = v;
        }
        self.indices[0][k] = slot.0;
        self.indices[1][k] = slot.1;
        self.ids[0][k] = id as u32;
        self.ids[1][k] = (id >> 32) as u32;
        for (plane, v) in self.report.iter_mut().zip(report) {
            plane[k] = v;
        }
    }
    /// Copy this block's columns from columns `off..` of device ray planes
    /// of `count` rays (`[floats, indices, ids, report]` views).
    fn copy_from(&mut self, off: usize, count: usize, views: &[ReadView]) -> MkResult<()> {
        let len = self.len();
        let part = |k: usize| k * count + off..k * count + off + len;
        let f = views[0].as_slice::<f32>()?;
        for (k, plane) in self.floats.iter_mut().enumerate() {
            plane.copy_from_slice(&f[part(k)]);
        }
        for (planes, view) in [&mut self.indices, &mut self.ids, &mut self.report]
            .into_iter()
            .zip(&views[1..])
        {
            let u = view.as_slice::<u32>()?;
            for (k, plane) in planes.iter_mut().enumerate() {
                plane.copy_from_slice(&u[part(k)]);
            }
        }
        Ok(())
    }
}

/// Fill four new shared buffers of `n` rays in the resident generated-ray
/// layout (`ResidentRays`) in place: `fill` receives the storage as column
/// blocks of `chunk` rays. Returns `[floats, indices, logical_ids, report]`
/// (published without a copy) and `fill`'s result.
fn ray_buffers<R>(
    b: &dyn ComputeBackend,
    n: usize,
    chunk: usize,
    fill: impl for<'c> FnOnce(Vec<RayColumns<'c>>) -> MkResult<R>,
) -> MkResult<([BufferHandle; 4], R)> {
    fn planes<'a, T>(data: &'a mut [T], n: usize, chunk: usize, out: &mut [Vec<&'a mut [T]>]) {
        for plane in data.chunks_mut(n) {
            for (c, part) in plane.chunks_mut(chunk).enumerate() {
                out[c].push(part);
            }
        }
    }
    let mut fill = Some(fill);
    let mut result = None;
    let mut handles: Vec<BufferHandle> = Vec::new();
    let desc = |planes: usize, e: ElementType| BufferDesc::new(e, [planes, n]);
    let report = b.fill_buffer(desc(4, ElementType::U32), &mut |mut rf| {
        let report = rf.as_mut_slice::<u32>()?;
        let ids = b.fill_buffer(desc(2, ElementType::U32), &mut |mut xf| {
            let ids = xf.as_mut_slice::<u32>()?;
            let indices = b.fill_buffer(desc(2, ElementType::U32), &mut |mut uf| {
                let indices = uf.as_mut_slice::<u32>()?;
                let floats =
                    b.fill_buffer(desc(FLOAT_PLANES, ElementType::F32), &mut |mut ff| {
                        let floats = ff.as_mut_slice::<f32>()?;
                        let blocks = n.div_ceil(chunk);
                        let mut f: Vec<Vec<&mut [f32]>> = (0..blocks).map(|_| Vec::new()).collect();
                        let mut u: Vec<Vec<&mut [u32]>> = (0..blocks).map(|_| Vec::new()).collect();
                        let mut i: Vec<Vec<&mut [u32]>> = (0..blocks).map(|_| Vec::new()).collect();
                        let mut r: Vec<Vec<&mut [u32]>> = (0..blocks).map(|_| Vec::new()).collect();
                        if n > 0 {
                            planes(floats, n, chunk, &mut f);
                            planes(indices, n, chunk, &mut u);
                            planes(ids, n, chunk, &mut i);
                            planes(report, n, chunk, &mut r);
                        }
                        let columns = f
                            .into_iter()
                            .zip(u)
                            .zip(i)
                            .zip(r)
                            .enumerate()
                            .map(|(c, (((floats, indices), ids), report))| RayColumns {
                                start: c * chunk,
                                floats,
                                indices,
                                ids,
                                report,
                            })
                            .collect();
                        let fill = fill.take().ok_or_else(|| bad("ray buffers filled twice"))?;
                        result = Some(fill(columns)?);
                        Ok(())
                    })?;
                handles.push(floats);
                Ok(())
            })?;
            handles.push(indices);
            Ok(())
        })?;
        handles.push(ids);
        Ok(())
    })?;
    let [floats, indices, ids]: [BufferHandle; 3] =
        handles.try_into().map_err(|_| bad("ray buffers"))?;
    let result = result.ok_or_else(|| bad("ray buffers not filled"))?;
    Ok(([floats, indices, ids, report], result))
}

/// Report words of a generated ray, as the device writes them: status
/// (`MKL_AIM_FIELD_UNREACHABLE` for an unusable field, `MKL_AIM_FAILED` at
/// the stop for an unconverged aim), Newton iterations and the relative stop
/// residual as double-float bits.
fn report_words(g: &Result<GeneratedRay, AimReport>, valid: bool, stop: u32) -> [u32; 4] {
    let r = match g {
        Ok(g) => &g.report,
        Err(r) => r,
    };
    let residual = if r.residual.is_finite() {
        Df32::from_f64(r.residual)
    } else {
        Df32::new(r.residual as f32, 0.0)
    };
    let status = match g {
        Ok(_) => 0,
        Err(_) if !valid => MKL_AIM_FIELD_UNREACHABLE,
        Err(_) => MKL_AIM_FAILED | (stop << MKL_RAY_SURFACE_SHIFT),
    };
    [
        status,
        r.iterations,
        residual.hi.to_bits(),
        residual.lo.to_bits(),
    ]
}

/// Everything a host aim of one generation needs, shared by the tasks.
struct HostJob<'a> {
    tracer: &'a SequentialTracer,
    rows: &'a AimRows,
    /// (packed system, wavelength slot) of every row.
    slots: &'a [(u32, u32)],
    spec: &'a (dyn Fn(usize) -> RaySpec + Sync),
    /// Start points from a host seed table.
    table: Option<&'a HostTable>,
}
impl HostJob<'_> {
    /// The reference tracer's rays for `specs`, started from the host seed
    /// table where there is one. Each ray's result is independent of the
    /// others in the batch (`generate_many_from`).
    fn generate(&self, specs: &[RaySpec]) -> (Vec<Result<GeneratedRay, AimReport>>, u64) {
        let requests: Vec<RayRequest> = specs
            .iter()
            .map(|s| {
                let row = s.row as usize;
                (self.rows.field(row), self.rows.wavelength(row), s.pupil)
            })
            .collect();
        let mut stats = mklight_trace::AimStats::default();
        let rays = match self.table {
            Some(table) => {
                let r_stop = self.tracer.pupil().r_stop;
                let starts: Vec<_> = specs
                    .iter()
                    .map(|s| {
                        let row = s.row as usize;
                        let v = self.rows.field(row).point.vignetting;
                        let (x, y) = v.apply(s.pupil[0], s.pupil[1]);
                        table.start(row, [x, y], r_stop)
                    })
                    .collect();
                self.tracer
                    .generate_many_from(&requests, &starts, &mut stats)
            }
            None => self.tracer.generate_many_stats(&requests, &mut stats),
        };
        (rays, stats.iterations)
    }
    /// Aim the rays of `cols` with the reference tracer and write them.
    fn aim(&self, cols: &mut RayColumns<'_>) -> HostAims {
        let stop = self.tracer.system().stop as u32;
        let specs: Vec<RaySpec> = (cols.start..cols.start + cols.len())
            .map(self.spec)
            .collect();
        let (generated, iterations) = self.generate(&specs);
        let mut out = HostAims {
            rays: specs.len() as u64,
            iterations,
            ..Default::default()
        };
        for (k, (s, g)) in specs.iter().zip(&generated).enumerate() {
            let row = s.row as usize;
            let report = report_words(g, self.rows.field(row).is_valid(), stop);
            out.failed += u64::from(report[0] != 0);
            cols.write(
                k,
                g.as_ref().ok().map(|g| (&g.ray, s.weight)),
                self.slots[row],
                s.id,
                report,
            );
        }
        out
    }
    /// Aim the rays of `cols` (column blocks) in parallel.
    fn aim_all(&self, cols: Vec<RayColumns<'_>>, ctx: &JobContext) -> MkResult<HostAims> {
        cols.into_par_iter()
            .map(|mut c| {
                ctx.check()?;
                Ok(self.aim(&mut c))
            })
            .try_reduce(HostAims::default, |a, b| Ok(a.add(b)))
    }
    /// Aim the given rays (by index) in parallel, for host completion: the
    /// index, spec and ray of each ray the reference aims.
    fn aim_indices(&self, indices: &[u32]) -> Vec<(usize, RaySpec, GeneratedRay)> {
        indices
            .par_chunks(HOST_CHUNK)
            .flat_map_iter(|batch| {
                let specs: Vec<RaySpec> = batch.iter().map(|&i| (self.spec)(i as usize)).collect();
                let (rays, _) = self.generate(&specs);
                batch
                    .iter()
                    .zip(specs)
                    .zip(rays)
                    .filter_map(|((&i, s), g)| Some((i as usize, s, g.ok()?)))
                    .collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Split generations ([`Geometry::split_large_generations`]) below this
/// many rays × surfaces (rays of the generation times the image surface
/// index) are aimed on the host alone: the device's share does not pay for
/// its plans there. Measured on an M2 Pro (8 + 4 CPU cores, 19 GPU cores):
/// spot diagrams and geometric MTFs from 1.5 M up were 0.96 of host-only
/// aiming (geometric mean), from 4 M up 0.94; below 1.5 M they lost up to 18 %.
pub const SPLIT_MIN_WORK: usize = 1_500_000;
/// The device's share of a split generation's tiles unless the route sets
/// another ([`Route::device_share`]). On the M2 Pro, 0.35 gave the same mean
/// gain as 0.5 and 0.65 with the smallest worst case (+8 % against +26 % and
/// +32 % on one lens); a GPU with more cores per CPU core may take more.
pub const SPLIT_DEVICE_SHARE: f64 = 0.35;
/// Rays per split tile at most (the unit in which tiles are dealt to host
/// and device).
const DEVICE_TILE: usize = 1 << 17;
/// The smallest split tile.
pub(super) const MIN_DEVICE_TILE: usize = 1 << 13;
/// Device plans per split generation: few and large, since each plan's
/// Newton rounds cost a fixed latency (small plans left the device 4–5×
/// slower per ray than the host); two let the host complete the first while
/// the second runs.
const DEVICE_PLANS: usize = 2;

/// The tile of a split generation of `n` rays: a sixteenth of it, at least
/// `min` ([`MIN_DEVICE_TILE`] unless tuned) and at most [`DEVICE_TILE`]
/// rays, in whole host chunks.
fn split_tile(n: usize, min: usize) -> usize {
    let min = min.max(1);
    (n / 16)
        .clamp(min, DEVICE_TILE.max(min))
        .next_multiple_of(HOST_CHUNK)
}

/// Whether tile `k` of a split generation is the device's: a fraction
/// `share` of the tiles, spread evenly.
fn device_tile(k: usize, share: f64) -> bool {
    ((k + 1) as f64 * share).floor() > (k as f64 * share).floor()
}

/// The next message of `rx`, `None` once every sender is gone. While it
/// waits, a rayon worker runs other pending pool work, so a device side
/// that uses the pool cannot wait on this thread.
fn recv_helping<T>(rx: &mpsc::Receiver<T>) -> Option<T> {
    loop {
        match rx.try_recv() {
            Ok(v) => return Some(v),
            Err(mpsc::TryRecvError::Disconnected) => return None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
        if rayon::yield_now() != Some(rayon::Yield::Executed) {
            match rx.recv_timeout(std::time::Duration::from_micros(200)) {
                Ok(v) => return Some(v),
                Err(mpsc::RecvTimeoutError::Disconnected) => return None,
                Err(mpsc::RecvTimeoutError::Timeout) => {}
            }
        }
    }
}

/// A submitted device aiming plan.
pub(crate) struct DeviceAims {
    pending: mklight_gpu::Pending<PlanResult>,
    submitted: Instant,
}

/// Seed-table nodes per axis of a host table for `per_row` rays per field
/// row: about one node per four rays, odd, within 5..=37 (37 is the device
/// table's side).
pub(crate) fn table_side(per_row: usize) -> u32 {
    let s = ((per_row as f64 / 4.0).sqrt().round() as u32).clamp(5, 37);
    s | 1
}
/// Newton iterations a table-seeded device aim may spend over its schedule
/// before it is left to host completion: table seeds converge in one to
/// three, so a ray still unconverged after this is cheaper on the host.
const TABLE_SCHEDULE_CAP: u32 = 12;

/// The aiming adapters of [`Geometry`].
impl Geometry<'_> {
    /// Run `f` with the reference tracer of aiming mode `aiming`.
    pub(crate) fn with_tracer<R>(
        &self,
        aiming: RayAiming,
        f: impl FnOnce(&SequentialTracer) -> R,
    ) -> MkResult<R> {
        Ok(if aiming == self.tracer.aiming() {
            f(self.tracer)
        } else {
            f(&*self.tracer_for(aiming)?)
        })
    }
    /// Prepared field rows for (field point, system wavelength) pairs: the
    /// tracer's own system fields are reused, other points are prepared once
    /// each, in parallel. A point that cannot be resolved gives unusable rows
    /// (unreachable chief seeds, failed aims), as on the CPU.
    pub(crate) fn aim_rows(
        &self,
        p: &mut ResidentPlan,
        rows: &[(FieldPoint, usize)],
        aiming: RayAiming,
    ) -> MkResult<AimRows> {
        let start = Instant::now();
        // Unique points by coordinate bits, then by equality: a map grid has
        // up to a million rows, so no scan of the points per row.
        let mut points: Vec<FieldPoint> = Vec::new();
        let mut seen: HashMap<(u64, u64), Vec<usize>> = HashMap::new();
        let index: Vec<(usize, usize)> = rows
            .iter()
            .map(|(f, w)| {
                let same = seen.entry((f.hx.to_bits(), f.hy.to_bits())).or_default();
                let i = match same.iter().copied().find(|&i| points[i] == *f) {
                    Some(i) => i,
                    None => {
                        points.push(*f);
                        same.push(points.len() - 1);
                        points.len() - 1
                    }
                };
                (i, *w)
            })
            .collect();
        let prepared: Vec<(PreparedField, bool)> = self.with_tracer(aiming, |t| {
            points
                .par_iter()
                .map(
                    |&point| match t.fields().iter().find(|pf| pf.point == point) {
                        Some(pf) => (pf.clone(), true),
                        None => {
                            let pf = t.prepare_field(point).unwrap_or_else(|e| PreparedField {
                                point,
                                source: mklight_trace::ObjectSource::Direction {
                                    dir: mklight_core::Vec3::Z,
                                },
                                aims: vec![],
                                error: Some(e.to_string()),
                            });
                            (pf, false)
                        }
                    },
                )
                .collect()
        })?;
        let reused = prepared.iter().filter(|p| p.1).count() as u64;
        p.host.add(&resident::HostPreparation {
            seed_rows: rows.len() as u64,
            reused_fields: reused,
            solved_fields: prepared.len() as u64 - reused,
            seconds: start.elapsed().as_secs_f64(),
        });
        Ok(AimRows {
            aiming,
            fields: Arc::new(prepared.into_iter().map(|p| p.0).collect()),
            rows: index,
        })
    }
    /// Packed (system, wavelength slot) of every row.
    fn slots(&self, rows: &AimRows) -> MkResult<Vec<(u32, u32)>> {
        rows.rows.iter().map(|&(_, w)| self.chunk(w)).collect()
    }
    /// The stop tolerance of a device generation (CONVENTIONS §7: the
    /// tightest derived stop of the prepared fields, or the requested one).
    pub(super) fn device_tolerance(&self, aiming: RayAiming) -> MkResult<f32> {
        Ok(match aiming {
            RayAiming::Real { .. } => {
                self.with_tracer(aiming, |t| t.device_aim_tolerance() as f32)?
            }
            _ => 1e-10,
        })
    }
    /// Aim `n` rays of `rows` (ray `i`: `spec(i)`) by this adapter's design
    /// and import the aimed starting rays into `p`.
    pub(crate) fn aim_rays(
        &self,
        p: &mut ResidentPlan,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        ctx: &JobContext,
    ) -> MkResult<ResidentRays> {
        ctx.check()?;
        let handles = match self.route.aiming {
            AimDesign::HostTable if !matches!(rows.aiming, RayAiming::Real { .. }) => {
                // Paraxial launches need no solve: the device generates them
                // in the plan itself (they fail only where the reference
                // fails).
                self.evidence(|e| {
                    e.generations += 1;
                    e.rays += n as u64;
                    e.device_rays += n as u64;
                });
                return self.append_generation(p, rows, n, spec, None, ctx);
            }
            AimDesign::HostTable => {
                let table = self.host_seed_table(rows, n, ctx)?;
                let work = n.saturating_mul(self.tracer.system().image_index());
                let share = self.split_share();
                match table {
                    Some(table) if share > 0.0 && work >= self.tuning.split_min_work => {
                        self.aim_split(rows, n, spec, &table, share, ctx)?
                    }
                    table => self.aim_host(rows, n, spec, table.as_ref(), ctx)?,
                }
            }
            AimDesign::Table => {
                let table = self.host_table(rows, n, ctx)?;
                self.aim_device(rows, n, spec, Some(table), ctx)?
            }
            AimDesign::Device => self.aim_device(rows, n, spec, None, ctx)?,
        };
        let [floats, indices, logical_ids, report] = handles.map(|h| p.import(h.view()));
        Ok(ResidentRays {
            floats,
            indices,
            count: n,
            logical_ids,
            report,
        })
    }
    /// The device's share of this adapter's split generations.
    fn split_share(&self) -> f64 {
        if self.split && self.route.aiming == AimDesign::HostTable {
            self.route.device_share
        } else {
            0.0
        }
    }
    fn evidence(&self, f: impl FnOnce(&mut AimEvidence)) {
        let mut e = self.aim_evidence.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut e);
    }
    /// The aiming evidence since the previous plan (cleared).
    pub(crate) fn take_aim_evidence(&self) -> AimEvidence {
        let mut e = self.aim_evidence.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = std::mem::take(&mut *e);
        out.design = self.route.aiming.name();
        out.device_share = self.split_share();
        out
    }
    /// Aim every ray of the generation on the host.
    fn aim_host(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        table: Option<&HostTable>,
        ctx: &JobContext,
    ) -> MkResult<[BufferHandle; 4]> {
        let start = Instant::now();
        let slots = self.slots(rows)?;
        let (handles, stats) = self.with_tracer(rows.aiming, |tracer| {
            let job = HostJob {
                tracer,
                rows,
                slots: &slots,
                spec,
                table,
            };
            ray_buffers(self.backend, n, HOST_CHUNK, |cols| job.aim_all(cols, ctx))
        })??;
        self.evidence(|e| {
            e.generations += 1;
            e.rays += n as u64;
            e.host(&stats);
            e.host_seconds += start.elapsed().as_secs_f64();
        });
        Ok(handles)
    }
    /// Aim a large generation on the host and the device together (module
    /// docs). The generation is cut into tiles of whole host chunks; the
    /// device takes the chunks of its tiles (`device_share` of them,
    /// [`device_tile`]) whose rays all start inside the seed table's solved
    /// nodes (rays the reference cannot aim lie at unsolved nodes; they stay
    /// the host's). Its chunks form [`DEVICE_PLANS`] plans, driven from a
    /// thread of its own outside the rayon pool (it mostly waits for the
    /// GPU), each copied into the shared planes as it lands; the pool's
    /// workers aim the host's chunks, then complete the device plans'
    /// unconverged aims.
    fn aim_split(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        table: &HostTable,
        device_share: f64,
        ctx: &JobContext,
    ) -> MkResult<[BufferHandle; 4]> {
        let start = Instant::now();
        let slots = self.slots(rows)?;
        let nodes = (table.seed_nodes(), table.side as u32);
        let per = split_tile(n, self.tuning.split_min_tile) / HOST_CHUNK;
        let device_chunk: Vec<bool> = (0..n.div_ceil(HOST_CHUNK))
            .into_par_iter()
            .map(|c| {
                device_tile(c / per, device_share)
                    && (c * HOST_CHUNK..((c + 1) * HOST_CHUNK).min(n)).all(|i| {
                        let s = spec(i);
                        let row = s.row as usize;
                        let v = rows.field(row).point.vignetting;
                        let (x, y) = v.apply(s.pupil[0], s.pupil[1]);
                        table.covers(row, [x, y])
                    })
            })
            .collect();
        let (handles, (host, host_seconds, completion)) =
            self.with_tracer(rows.aiming, |tracer| {
                let job = HostJob {
                    tracer,
                    rows,
                    slots: &slots,
                    spec,
                    table: Some(table),
                };
                ray_buffers(self.backend, n, HOST_CHUNK, |chunks| {
                    let (mut device, mut host) = (Vec::new(), Vec::new());
                    for (c, cols) in chunks.into_iter().enumerate() {
                        if device_chunk[c] {
                            device.push(cols);
                        } else {
                            host.push(cols);
                        }
                    }
                    let per_plan = device.len().div_ceil(DEVICE_PLANS).max(1);
                    let mut plans: Vec<Vec<RayColumns<'_>>> = Vec::new();
                    for cols in device {
                        match plans.last_mut() {
                            Some(p) if p.len() < per_plan => p.push(cols),
                            _ => plans.push(vec![cols]),
                        }
                    }
                    std::thread::scope(|scope| {
                        let (tx, rx) = mpsc::channel();
                        let driver = scope.spawn(move || -> MkResult<()> {
                            // Submit every plan, then copy each as it lands.
                            let mut submitted = Vec::with_capacity(plans.len());
                            for cols in plans {
                                let rays: Vec<usize> = cols
                                    .iter()
                                    .flat_map(|c| c.start..c.start + c.len())
                                    .collect();
                                let d = self.submit_device(
                                    rows,
                                    rays.len(),
                                    &|k| spec(rays[k]),
                                    Some(&nodes),
                                    ctx,
                                )?;
                                submitted.push((d, rays, cols));
                            }
                            for (d, rays, mut cols) in submitted {
                                let views = self
                                    .finish_device(d)?
                                    .iter()
                                    .map(|h| self.backend.read_view(h.view()))
                                    .collect::<MkResult<Vec<_>>>()?;
                                let mut offset = 0;
                                for c in &mut cols {
                                    c.copy_from(offset, rays.len(), &views)?;
                                    offset += c.len();
                                }
                                let report = views[3].as_slice::<u32>()?;
                                let failed: Vec<u32> = (0..rays.len())
                                    .filter(|&k| report[k] & MKL_RAY_CODE_MASK == MKL_AIM_FAILED)
                                    .map(|k| rays[k] as u32)
                                    .collect();
                                if tx.send((cols, failed)).is_err() {
                                    return Ok(());
                                }
                            }
                            Ok(())
                        });
                        let aimed = job.aim_all(host, ctx);
                        let host_seconds = start.elapsed().as_secs_f64();
                        // Host completion of each device plan as it lands: its
                        // rays, the resolved ones and the seconds they took.
                        let mut completion = (0u64, 0u64, 0.0f64);
                        while let Some((mut cols, failed)) = recv_helping(&rx) {
                            let landed = Instant::now();
                            let resolved = job.aim_indices(&failed);
                            completion.0 += failed.len() as u64;
                            completion.1 += resolved.len() as u64;
                            for (i, s, g) in resolved {
                                let c = cols.partition_point(|c| c.start <= i) - 1;
                                let k = i - cols[c].start;
                                cols[c].write(
                                    k,
                                    Some((&g.ray, s.weight)),
                                    slots[s.row as usize],
                                    s.id,
                                    report_words(&Ok(g), true, 0),
                                );
                            }
                            completion.2 += landed.elapsed().as_secs_f64();
                        }
                        driver
                            .join()
                            .map_err(|_| bad("device aiming thread panicked"))??;
                        Ok((aimed?, host_seconds, completion))
                    })
                })
            })??;
        self.evidence(|e| {
            e.generations += 1;
            e.rays += n as u64;
            e.host(&host);
            e.host_seconds += host_seconds;
            e.completion_rays += completion.0;
            e.completion_resolved += completion.1;
            e.completion_seconds += completion.2;
        });
        Ok(handles)
    }
    /// Submit a device aiming plan for `n` rays (ray `k`: `spec(k)`), seeded
    /// from host chief rows and a host seed table's nodes and side, if any.
    fn submit_device(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        table: Option<&(Vec<SeedNode>, u32)>,
        ctx: &JobContext,
    ) -> MkResult<DeviceAims> {
        let mut a = ResidentPlan::new(self.tracer.system().snapshot.clone());
        let out = self.append_generation(&mut a, rows, n, spec, table, ctx)?;
        a.outputs = vec![out.floats, out.indices, out.logical_ids, out.report];
        Ok(DeviceAims {
            submitted: Instant::now(),
            pending: self.backend.submit_plan(a, ctx)?,
        })
    }
    /// Append a device generation (`GenerateAim`) of `n` rays (ray `k`:
    /// `spec(k)`) to plan `a`: field rows, samples, host chief rows and, for
    /// table-seeded aims, the seed table with its bounded schedule. Returns
    /// its output rays.
    fn append_generation(
        &self,
        a: &mut ResidentPlan,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        table: Option<&(Vec<SeedNode>, u32)>,
        ctx: &JobContext,
    ) -> MkResult<ResidentRays> {
        let slots = self.slots(rows)?;
        let metadata = self.records::<FieldSpec>(a, rows.len(), |i| {
            FieldSpec::new(
                self.tracer.system(),
                self.tracer,
                rows.field(i).point,
                slots[i].0,
                slots[i].1,
                rows.aiming,
            )
        })?;
        let samples = self.records::<Sample>(a, n, |k| {
            if k % 4096 == 0 {
                ctx.check()?;
            }
            let s = spec(k);
            let mut x = Sample::new(s.pupil, s.row, s.id);
            x.weight = s.weight;
            Ok(x)
        })?;
        let field_key = self.field_key(a, metadata)?;
        let seeds = self.with_tracer(rows.aiming, |t| {
            self.records::<ChiefSeed>(a, rows.len(), |i| {
                Ok(ChiefSeed::host(t, rows.field(i), rows.wavelength(i)))
            })
        })??;
        let options = match table {
            Some((nodes, side)) => {
                let id = self.records::<SeedNode>(a, nodes.len(), |i| Ok(nodes[i]))?;
                AimOptions {
                    table: Some(a.view(id)?),
                    table_side: *side,
                    schedule_cap: TABLE_SCHEDULE_CAP,
                    ..Default::default()
                }
            }
            None => Default::default(),
        };
        let out = rays(a, n);
        a.operations
            .push(ResidentOp::Aiming(Request::GenerateAim(GenerateAim {
                systems: self.systems,
                samples: a.view(samples)?,
                field_wavelength: a.view(metadata)?,
                field_key,
                seeds: Some(a.view(seeds)?),
                tolerance: self.device_tolerance(rows.aiming)?,
                max_iterations: 60,
                output: out.clone(),
                options,
            })));
        Ok(out)
    }
    /// Wait for a device aiming plan: its four ray buffers.
    fn finish_device(&self, d: DeviceAims) -> MkResult<[BufferHandle; 4]> {
        let r = d.pending.wait()?;
        let seconds = d.submitted.elapsed().as_secs_f64();
        let stats = &r.stats.aiming;
        self.evidence(|e| {
            e.device_plans += 1;
            e.device_seconds += seconds;
            e.device_rays += stats.rays;
            e.device_failed += stats.failed;
            e.device_iterations += stats.iterations;
        });
        r.outputs.try_into().map_err(|_| bad("device aim outputs"))
    }
    /// A host f64 seed table as device nodes for the rows of a generation of
    /// `n` rays ([`AimDesign::Table`]).
    fn host_table(
        &self,
        rows: &AimRows,
        n: usize,
        ctx: &JobContext,
    ) -> MkResult<(Vec<SeedNode>, u32)> {
        ctx.check()?;
        let start = Instant::now();
        let side = table_side(n / rows.len().max(1));
        let table = self.with_tracer(rows.aiming, |t| {
            let refs: Vec<(&PreparedField, usize)> = (0..rows.len())
                .map(|r| (rows.field(r), rows.wavelength(r)))
                .collect();
            host_tables(t, &refs, side)
        })?;
        self.evidence(|e| {
            e.table_nodes += table.len() as u64;
            e.table_seconds += start.elapsed().as_secs_f64();
        });
        Ok((table, side))
    }
    /// Aim every ray of the generation on the device ([`AimDesign::Table`],
    /// [`AimDesign::Device`]), in a plan of its own, then complete it.
    fn aim_device(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        table: Option<(Vec<SeedNode>, u32)>,
        ctx: &JobContext,
    ) -> MkResult<[BufferHandle; 4]> {
        let d = self.submit_device(rows, n, spec, table.as_ref(), ctx)?;
        let handles = self.finish_device(d)?;
        self.evidence(|e| {
            e.generations += 1;
            e.rays += n as u64;
        });
        self.complete(rows, n, spec, handles, ctx)
    }
    /// Host completion (module docs) of a device generation of `n` rays: the
    /// device's buffers when no unconverged aim resolves on the host,
    /// otherwise copies of its position and report planes with the host's
    /// rays in place.
    fn complete(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        handles: [BufferHandle; 4],
        ctx: &JobContext,
    ) -> MkResult<[BufferHandle; 4]> {
        let start = Instant::now();
        let report = self.backend.read_view(handles[3].view())?;
        let report = report.as_slice::<u32>()?;
        let failed: Vec<u32> = (0..n)
            .filter(|&i| report[i] & MKL_RAY_CODE_MASK == MKL_AIM_FAILED)
            .map(|i| i as u32)
            .collect();
        if failed.is_empty() {
            return Ok(handles);
        }
        ctx.check()?;
        let slots = self.slots(rows)?;
        let resolved = self.with_tracer(rows.aiming, |tracer| {
            HostJob {
                tracer,
                rows,
                slots: &slots,
                spec,
                table: None,
            }
            .aim_indices(&failed)
        })?;
        ctx.check()?;
        let fixed = resolved.len() as u64;
        let out = if resolved.is_empty() {
            handles
        } else {
            // Copies of the device planes with the host rays written in.
            let views = handles
                .iter()
                .map(|h| self.backend.read_view(h.view()))
                .collect::<MkResult<Vec<_>>>()?;
            ray_buffers(self.backend, n, HOST_CHUNK, |mut cols| {
                cols.par_iter_mut()
                    .try_for_each(|c| c.copy_from(c.start, n, &views))?;
                for (i, s, g) in &resolved {
                    cols[i / HOST_CHUNK].write(
                        i % HOST_CHUNK,
                        Some((&g.ray, s.weight)),
                        slots[s.row as usize],
                        s.id,
                        report_words(&Ok(*g), true, 0),
                    );
                }
                Ok(())
            })?
            .0
        };
        self.evidence(|e| {
            e.completion_rays += failed.len() as u64;
            e.completion_resolved += fixed;
            e.completion_seconds += start.elapsed().as_secs_f64();
        });
        Ok(out)
    }
}

/// Rays per field row below which a generation is aimed without a seed
/// table: a coarse table's interpolation saves too few iterations to pay
/// for its nodes.
pub const HOST_TABLE_MIN_PER_ROW: usize = 2048;
const HOST_TABLE_EXTENT: f64 = 1.125;

/// A host f64 seed table: for every field row, the reference tracer's aims
/// at a `side`² lattice of vignetted pupil points over [−1.125, 1.125]²
/// (nodes beyond pupil radius 1.25, unusable rows and unreachable nodes
/// unsolved), as the device table ([`host_tables`]) but kept in f64.
pub(crate) struct HostTable {
    side: usize,
    nodes: Vec<Option<[f64; 2]>>,
}
impl HostTable {
    fn build(tracer: &SequentialTracer, rows: &AimRows, side: usize) -> Self {
        let nodes = side * side;
        let step = 2.0 * HOST_TABLE_EXTENT / (side - 1) as f64;
        let mut out = vec![None; rows.len() * nodes];
        out.par_chunks_mut(nodes)
            .enumerate()
            .for_each(|(row, table)| {
                let field = rows.field(row);
                let v = field.point.vignetting;
                if !field.is_valid() || v.vcx >= 1. || v.vcy >= 1. {
                    return;
                }
                let mut solved = Vec::with_capacity(nodes);
                let mut requests = Vec::with_capacity(nodes);
                for k in 0..nodes {
                    let p = [
                        (k % side) as f64 * step - HOST_TABLE_EXTENT,
                        (k / side) as f64 * step - HOST_TABLE_EXTENT,
                    ];
                    if p[0] * p[0] + p[1] * p[1] > 1.5625 {
                        continue;
                    }
                    // Nodes lie in vignetted coordinates; `generate` applies
                    // the field's vignetting to the point it is given.
                    let px = (p[0] - v.vdx) / (1. - v.vcx);
                    let py = (p[1] - v.vdy) / (1. - v.vcy);
                    solved.push(k);
                    requests.push((field, rows.wavelength(row), [px, py]));
                }
                for (k, g) in solved.into_iter().zip(tracer.generate_many(&requests)) {
                    table[k] = g.ok().map(|g| g.aim);
                }
            });
        Self { side, nodes: out }
    }
    /// The table as device seed nodes (double-float aims; status 1 where
    /// unsolved), for [`AimOptions::table`].
    fn seed_nodes(&self) -> Vec<SeedNode> {
        self.nodes
            .iter()
            .map(|n| match n {
                Some(a) => SeedNode {
                    aim: a.map(Df32::from_f64),
                    status: 0,
                    reserved: 0,
                },
                None => SeedNode {
                    aim: [Df32::ZERO; 2],
                    status: 1,
                    reserved: 0,
                },
            })
            .collect()
    }
    /// The table cell of vignetted pupil point `p`: its fractional node
    /// coordinates and the node below and left of it, if its 4 × 4
    /// interpolation stencil lies inside the table.
    fn cell(&self, p: [f64; 2]) -> Option<(f64, f64, f64, f64)> {
        let inv = (self.side - 1) as f64 / (2.0 * HOST_TABLE_EXTENT);
        let fx = (p[0] + HOST_TABLE_EXTENT) * inv;
        let fy = (p[1] + HOST_TABLE_EXTENT) * inv;
        let (ix, iy) = (fx.floor(), fy.floor());
        let last = (self.side - 3) as f64;
        (ix >= 1.0 && iy >= 1.0 && ix <= last && iy <= last).then_some((fx, fy, ix, iy))
    }
    /// Whether all 16 nodes around vignetted pupil point `p` of row `row`
    /// are solved (the condition for [`Self::start`]).
    fn covers(&self, row: usize, p: [f64; 2]) -> bool {
        let side = self.side;
        self.cell(p).is_some_and(|(_, _, ix, iy)| {
            let base = row * side * side + (iy as usize - 1) * side + ix as usize - 1;
            (0..4).all(|j| (0..4).all(|i| self.nodes[base + j * side + i].is_some()))
        })
    }
    /// The Catmull–Rom interpolated aim of row `row` at vignetted pupil
    /// point `p`, and the stop Jacobian `r_stop · (d aim / dp)⁻¹` of the
    /// interpolant; `None` unless all 16 nodes around `p` are solved.
    pub fn start(&self, row: usize, p: [f64; 2], r_stop: f64) -> Option<([f64; 2], [[f64; 2]; 2])> {
        let side = self.side;
        let inv = (side - 1) as f64 / (2.0 * HOST_TABLE_EXTENT);
        let (fx, fy, ix, iy) = self.cell(p)?;
        let spline = |t: f64| {
            let (t2, t3) = (t * t, t * t * t);
            (
                [
                    (2.0 * t2 - t3 - t) * 0.5,
                    (3.0 * t3 - 5.0 * t2 + 2.0) * 0.5,
                    (4.0 * t2 - 3.0 * t3 + t) * 0.5,
                    (t3 - t2) * 0.5,
                ],
                [
                    (4.0 * t - 3.0 * t2 - 1.0) * 0.5,
                    (9.0 * t2 - 10.0 * t) * 0.5,
                    (8.0 * t - 9.0 * t2 + 1.0) * 0.5,
                    (3.0 * t2 - 2.0 * t) * 0.5,
                ],
            )
        };
        let (wx, dx) = spline(fx - ix);
        let (wy, dy) = spline(fy - iy);
        let base = row * side * side;
        let (mut a, mut ax, mut ay) = ([0.0; 2], [0.0; 2], [0.0; 2]);
        for j in 0..4 {
            for i in 0..4 {
                let n = self.nodes[base + (iy as usize + j - 1) * side + ix as usize + i - 1]?;
                for k in 0..2 {
                    a[k] += n[k] * (wx[i] * wy[j]);
                    ax[k] += n[k] * (dx[i] * wy[j]);
                    ay[k] += n[k] * (wx[i] * dy[j]);
                }
            }
        }
        // d aim / dp = inv · [ax ay]; the stop target is p · r_stop.
        let det = ax[0] * ay[1] - ay[0] * ax[1];
        let q = r_stop / (det * inv);
        let j = [[ay[1] * q, -ay[0] * q], [-ax[1] * q, ax[0] * q]];
        (det != 0.0 && j.iter().flatten().all(|v| v.is_finite()) && a.iter().all(|v| v.is_finite()))
            .then_some((a, j))
    }
}
impl Geometry<'_> {
    /// The host f64 seed table of a generation of `n` rays, or `None` below
    /// [`HOST_TABLE_MIN_PER_ROW`] rays per row.
    fn host_seed_table(
        &self,
        rows: &AimRows,
        n: usize,
        ctx: &JobContext,
    ) -> MkResult<Option<HostTable>> {
        let per_row = n / rows.len().max(1);
        if per_row < HOST_TABLE_MIN_PER_ROW {
            return Ok(None);
        }
        ctx.check()?;
        let start = Instant::now();
        let side = table_side(per_row) as usize;
        let table = self.with_tracer(rows.aiming, |t| HostTable::build(t, rows, side))?;
        self.evidence(|e| {
            e.table_nodes += table.nodes.len() as u64;
            e.table_seconds += start.elapsed().as_secs_f64();
        });
        Ok(Some(table))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_gpu::CpuF32Backend;

    fn tracer() -> SequentialTracer {
        let mut m = mklight_core::samples::double_gauss();
        m.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        crate::tracer_for(m).unwrap()
    }

    #[test]
    fn host_table_starts_are_close_to_the_reference_aims() {
        let t = tracer();
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        let geo = Geometry::new(&b, &t, &ctx).unwrap();
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let points: Vec<_> = (0..t.fields().len())
            .map(|f| (t.field(f).point, t.system().primary_wavelength))
            .collect();
        let rows = geo.aim_rows(&mut p, &points, t.aiming()).unwrap();
        let table = HostTable::build(&t, &rows, 25);
        let r_stop = t.pupil().r_stop;
        let mut seen = 0;
        for row in 0..rows.len() {
            for &q in &[[0.13, -0.41], [0.52, 0.31], [-0.77, 0.12], [0.0, 0.9]] {
                // The table lies in vignetted pupil coordinates.
                let (x, y) = rows.field(row).point.vignetting.apply(q[0], q[1]);
                let Some((a, j)) = table.start(row, [x, y], r_stop) else {
                    continue;
                };
                seen += 1;
                let g = t
                    .generate(rows.field(row), rows.wavelength(row), q[0], q[1])
                    .unwrap();
                let scale = g.aim[0].hypot(g.aim[1]).max(r_stop);
                for k in 0..2 {
                    assert!(
                        (a[k] - g.aim[k]).abs() < 1e-4 * scale,
                        "{row} {q:?} {a:?} {:?}",
                        g.aim
                    );
                }
                assert!(j.iter().flatten().all(|v| v.is_finite()));
            }
        }
        assert!(seen > 0);
        // Outside the table's interior the start is left to the reference.
        assert!(table.start(0, [1.2, 0.0], r_stop).is_none());
    }

    #[test]
    fn completion_replaces_unconverged_device_aims_with_reference_aims() {
        let t = tracer();
        let b = CpuF32Backend::new();
        let ctx = JobContext::new();
        let geo = Geometry::new(&b, &t, &ctx).unwrap();
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let points = [
            (t.field(1).point, t.system().primary_wavelength),
            (t.field(2).point, 0),
        ];
        let rows = geo.aim_rows(&mut p, &points, t.aiming()).unwrap();
        let pupil = [[0.0, 0.0], [0.5, 0.3], [-0.7, 0.6], [0.9, -0.1], [3.0, 3.0]];
        let n = 2 * pupil.len();
        let spec = |i: usize| RaySpec {
            row: (i / pupil.len()) as u32,
            pupil: pupil[i % pupil.len()],
            weight: 0.5,
            id: 100 + i as u64,
        };
        let slots = geo.slots(&rows).unwrap();
        let stop = (t.system().stop as u32) << MKL_RAY_SURFACE_SHIFT;
        // A device result: ray 0 converged (a sentinel the completion must
        // keep), every other aim unconverged with a zero ray.
        let floats = b
            .fill_buffer(BufferDesc::new(ElementType::F32, [15, n]), &mut |mut f| {
                let f = f.as_mut_slice::<f32>()?;
                f.fill(0.0);
                f[0] = 42.0;
                Ok(())
            })
            .unwrap();
        let indices = b
            .fill_buffer(BufferDesc::new(ElementType::U32, [2, n]), &mut |mut f| {
                let f = f.as_mut_slice::<u32>()?;
                for i in 0..n {
                    let s = slots[spec(i).row as usize];
                    f[i] = s.0;
                    f[n + i] = s.1;
                }
                Ok(())
            })
            .unwrap();
        let ids = b
            .fill_buffer(BufferDesc::new(ElementType::U32, [2, n]), &mut |mut f| {
                let f = f.as_mut_slice::<u32>()?;
                for i in 0..n {
                    f[i] = spec(i).id as u32;
                    f[n + i] = 0;
                }
                Ok(())
            })
            .unwrap();
        let report = b
            .fill_buffer(BufferDesc::new(ElementType::U32, [4, n]), &mut |mut f| {
                let f = f.as_mut_slice::<u32>()?;
                f.fill(0);
                for v in &mut f[1..n] {
                    *v = MKL_AIM_FAILED | stop;
                }
                Ok(())
            })
            .unwrap();
        let out = geo
            .complete(&rows, n, &spec, [floats, indices, ids, report], &ctx)
            .unwrap();
        let f = b.read_view(out[0].view()).unwrap();
        let f = f.as_slice::<f32>().unwrap();
        let r = b.read_view(out[3].view()).unwrap();
        let r = r.as_slice::<u32>().unwrap();
        assert_eq!((f[0], r[0]), (42.0, 0), "a converged device ray is kept");
        let mut resolved = 0;
        for i in 1..n {
            let s = spec(i);
            let row = s.row as usize;
            let g = t.generate(
                rows.field(row),
                rows.wavelength(row),
                s.pupil[0],
                s.pupil[1],
            );
            match g {
                Ok(g) => {
                    resolved += 1;
                    assert_eq!(r[i], 0, "ray {i}");
                    let p = [g.ray.pos.x, g.ray.pos.y, g.ray.pos.z].map(Df32::from_f64);
                    let d = [g.ray.dir.x, g.ray.dir.y, g.ray.dir.z].map(Df32::from_f64);
                    for k in 0..3 {
                        assert_eq!((f[k * n + i], f[(9 + k) * n + i]), (p[k].hi, p[k].lo));
                        assert_eq!(
                            (f[(3 + k) * n + i], f[(12 + k) * n + i]),
                            (d[k].hi, d[k].lo)
                        );
                    }
                    assert_eq!(f[8 * n + i], 0.5);
                }
                Err(_) => {
                    assert_eq!(r[i], MKL_AIM_FAILED | stop, "ray {i} stays failed");
                    assert!((0..15).all(|k| f[k * n + i] == 0.0));
                }
            }
        }
        assert!(resolved >= 6);
        let e = geo.take_aim_evidence();
        assert_eq!(e.completion_rays, n as u64 - 1);
        assert_eq!(e.completion_resolved, resolved);
    }

    #[test]
    fn split_tiles_spread_the_device_share_over_whole_chunks() {
        let half = |k| device_tile(k, 0.5);
        assert!(!half(0) && half(1) && !half(2) && half(3));
        for share in [0.0, 0.25, 0.5, 0.7, 1.0] {
            let device = (0..1000).filter(|&k| device_tile(k, share)).count();
            assert_eq!(device, (1000.0 * share) as usize, "{share}");
        }
        for n in [1usize, 5000, 1 << 20, 1 << 24] {
            let t = split_tile(n, MIN_DEVICE_TILE);
            assert_eq!(t % HOST_CHUNK, 0);
            assert!((MIN_DEVICE_TILE..=DEVICE_TILE).contains(&t), "{n}: {t}");
        }
    }
}
