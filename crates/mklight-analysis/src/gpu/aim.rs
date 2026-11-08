//! Pupil-aiming designs of the resident geometric adapters (study key
//! `aiming`, [`mklight_core::split`]).
//!
//! Every generation of a [`Geometry`] (grid rays, chief rays, layout fans)
//! is aimed by one design, read once per adapter in [`Geometry::new`]:
//!
//! * [`AimDesign::Host`] (default): the reference tracer aims every ray in
//!   f64 (`SequentialTracer::generate_many`: batched Newton with analytic
//!   Jacobians, the chunks in parallel on rayon) and writes the starting
//!   rays straight into the shared ray planes the device trace reads; no
//!   device aiming, no sample or field records.
//! * [`AimDesign::Device`]: the resident design before the work split:
//!   double-float Newton–Broyden on the device from host chief rows (and a
//!   device seed table for large generations), in its own plan, followed by
//!   host completion.
//! * [`AimDesign::Table`]: a coarse host f64 aim map per field row (a seed
//!   table whose side follows the rays per row) and device Newton
//!   corrections from its interpolated seeds, with a bounded schedule,
//!   followed by host completion.
//! * [`AimDesign::Split`]: host and device aim one generation at the same
//!   time: the device ([`AimDesign::Table`]) takes tiles from the front, the
//!   host's rayon workers take chunks from the back until they meet; the
//!   host completes the device tiles.
//!
//! **Host completion.** A device double-float aim cannot always certify the
//! requested stop tolerance: the double-float trace carries the even-asphere
//! slope in f32, so on wide-angle aspheric designs the evaluated stop hit
//! jitters by ~1e-9–1e-8 of the stop radius, above the 1e-10 the reference
//! reaches (a third of a 35-surface ultra-wide's rays failed, and with them
//! some chief rays). Every ray the device reports as an unconverged aim
//! (`MKL_AIM_FAILED`) is therefore aimed again by the reference tracer in
//! f64; a ray the reference aims replaces the device's, one it cannot aim
//! stays failed exactly as on the CPU. Hard field failures are the host
//! rows' own and are kept. A failed aim is never accepted.

use super::*;
use mklight_core::{RayAiming, split};
use mklight_gpu::{consts::*, resident};
use mklight_trace::{AimReport, FieldPoint, GeneratedRay, PreparedField, RayRequest};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// The study key of the aiming design.
pub const KEY: &str = "aiming";

/// How a [`Geometry`] aims its generations (module docs).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AimDesign {
    #[default]
    Host,
    Device,
    Table,
    Split,
}
impl AimDesign {
    /// The design selected for this run: `aiming` = `host`, `device`,
    /// `table` or `split`; the default design without a selection.
    pub fn selected() -> MkResult<Self> {
        Ok(match split::variant(KEY).as_deref() {
            None => Self::default(),
            Some("host") => Self::Host,
            Some("device") => Self::Device,
            Some("table") => Self::Table,
            Some("split") => Self::Split,
            Some(other) => {
                return Err(mklight_core::MkError::InvalidParam(format!(
                    "unknown aiming design '{other}' (host, device, table, split)"
                )));
            }
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Host => "host",
            Self::Device => "device",
            Self::Table => "table",
            Self::Split => "split",
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
    pub generations: u64,
    pub rays: u64,
    /// Rays aimed by the reference tracer (host design and host shares).
    pub host_rays: u64,
    pub host_failed: u64,
    pub host_iterations: u64,
    pub host_seconds: f64,
    /// Rays aimed on the device, their unconverged aims and Newton iterations.
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
    /// Host seed-table nodes ([`AimDesign::Table`]/[`AimDesign::Split`]).
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
}

/// Split plane-major storage of `n` rays into column blocks of `chunk` rays.
pub(crate) fn columns<'a>(
    floats: &'a mut [f32],
    indices: &'a mut [u32],
    ids: &'a mut [u32],
    report: &'a mut [u32],
    n: usize,
    chunk: usize,
) -> Vec<RayColumns<'a>> {
    fn planes<'a, T>(data: &'a mut [T], n: usize, chunk: usize, out: &mut [Vec<&'a mut [T]>]) {
        for plane in data.chunks_mut(n) {
            for (c, part) in plane.chunks_mut(chunk).enumerate() {
                out[c].push(part);
            }
        }
    }
    if n == 0 {
        return Vec::new();
    }
    let blocks = n.div_ceil(chunk);
    let mut f: Vec<Vec<&mut [f32]>> = (0..blocks)
        .map(|_| Vec::with_capacity(FLOAT_PLANES))
        .collect();
    let mut u: Vec<Vec<&mut [u32]>> = (0..blocks).map(|_| Vec::with_capacity(2)).collect();
    let mut i: Vec<Vec<&mut [u32]>> = (0..blocks).map(|_| Vec::with_capacity(2)).collect();
    let mut r: Vec<Vec<&mut [u32]>> = (0..blocks).map(|_| Vec::with_capacity(4)).collect();
    planes(floats, n, chunk, &mut f);
    planes(indices, n, chunk, &mut u);
    planes(ids, n, chunk, &mut i);
    planes(report, n, chunk, &mut r);
    f.into_iter()
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
        .collect()
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
pub(crate) struct HostJob<'a> {
    pub tracer: &'a SequentialTracer,
    pub rows: &'a AimRows,
    /// (packed system, wavelength slot) of every row.
    pub slots: &'a [(u32, u32)],
    pub spec: &'a (dyn Fn(usize) -> RaySpec + Sync),
}
impl HostJob<'_> {
    /// Aim the rays of `cols` with the reference tracer and write them.
    fn aim(&self, cols: &mut RayColumns<'_>) -> HostAims {
        let stop = self.tracer.system().stop as u32;
        let specs: Vec<RaySpec> = (cols.start..cols.start + cols.len())
            .map(self.spec)
            .collect();
        let requests: Vec<RayRequest> = specs
            .iter()
            .map(|s| {
                let row = s.row as usize;
                (self.rows.field(row), self.rows.wavelength(row), s.pupil)
            })
            .collect();
        let mut stats = mklight_trace::AimStats::default();
        let generated = self.tracer.generate_many_stats(&requests, &mut stats);
        let mut out = HostAims {
            rays: specs.len() as u64,
            iterations: stats.iterations,
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
    /// [`Self::aim_indices`] on the calling thread (one batch).
    fn aim_indices_serial(&self, indices: &[u32]) -> Vec<(u32, Result<GeneratedRay, AimReport>)> {
        let specs: Vec<RaySpec> = indices.iter().map(|&i| (self.spec)(i as usize)).collect();
        let requests: Vec<RayRequest> = specs
            .iter()
            .map(|s| {
                let row = s.row as usize;
                (self.rows.field(row), self.rows.wavelength(row), s.pupil)
            })
            .collect();
        indices
            .iter()
            .copied()
            .zip(self.tracer.generate_many(&requests))
            .collect()
    }
    /// Aim the given rays (by index) with the reference tracer, in parallel:
    /// the generated ray or report of each.
    fn aim_indices(&self, indices: &[u32]) -> Vec<(u32, Result<GeneratedRay, AimReport>)> {
        indices
            .par_chunks(HOST_CHUNK)
            .flat_map_iter(|chunk| {
                let specs: Vec<RaySpec> = chunk.iter().map(|&i| (self.spec)(i as usize)).collect();
                let requests: Vec<RayRequest> = specs
                    .iter()
                    .map(|s| {
                        let row = s.row as usize;
                        (self.rows.field(row), self.rows.wavelength(row), s.pupil)
                    })
                    .collect();
                let generated = self.tracer.generate_many(&requests);
                chunk.iter().copied().zip(generated).collect::<Vec<_>>()
            })
            .collect()
    }
}

/// Aim every ray of `n` with the reference tracer into four new shared
/// buffers (filled in place, published without a copy).
fn host_fill(
    b: &dyn ComputeBackend,
    job: &HostJob<'_>,
    n: usize,
    ctx: &JobContext,
) -> MkResult<([BufferHandle; 4], HostAims)> {
    let mut stats = HostAims::default();
    let mut handles: Vec<BufferHandle> = Vec::new();
    let report = b.fill_buffer(BufferDesc::new(ElementType::U32, [4, n]), &mut |mut rf| {
        let report = rf.as_mut_slice::<u32>()?;
        let ids = b.fill_buffer(BufferDesc::new(ElementType::U32, [2, n]), &mut |mut xf| {
            let ids = xf.as_mut_slice::<u32>()?;
            let indices =
                b.fill_buffer(BufferDesc::new(ElementType::U32, [2, n]), &mut |mut uf| {
                    let indices = uf.as_mut_slice::<u32>()?;
                    let floats = b.fill_buffer(
                        BufferDesc::new(ElementType::F32, [FLOAT_PLANES, n]),
                        &mut |mut ff| {
                            let floats = ff.as_mut_slice::<f32>()?;
                            let blocks = columns(floats, indices, ids, report, n, HOST_CHUNK);
                            stats = blocks
                                .into_par_iter()
                                .map(|mut cols| {
                                    ctx.check()?;
                                    Ok::<_, mklight_core::MkError>(job.aim(&mut cols))
                                })
                                .try_reduce(HostAims::default, |a, b| Ok(a.add(b)))?;
                            Ok(())
                        },
                    )?;
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
        handles.try_into().map_err(|_| bad("host aim buffers"))?;
    Ok(([floats, indices, ids, report], stats))
}

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
        let mut points: Vec<FieldPoint> = Vec::new();
        let index: Vec<(usize, usize)> = rows
            .iter()
            .map(|(f, w)| {
                let i = points.iter().position(|q| q == f).unwrap_or_else(|| {
                    points.push(*f);
                    points.len() - 1
                });
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
    fn device_tolerance(&self, aiming: RayAiming) -> MkResult<f32> {
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
        let handles = match self.aim {
            AimDesign::Host => self.aim_host(rows, n, spec, ctx)?,
            AimDesign::Device => self.aim_device(rows, n, spec, None, ctx)?,
            AimDesign::Table => {
                let table = self.host_table(rows, n, ctx)?;
                self.aim_device(rows, n, spec, Some(table), ctx)?
            }
            AimDesign::Split => self.aim_split(rows, n, spec, ctx)?,
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
    fn evidence(&self, f: impl FnOnce(&mut AimEvidence)) {
        let mut e = self.aim_evidence.lock().unwrap_or_else(|e| e.into_inner());
        e.design = self.aim.name();
        f(&mut e);
    }
    /// The aiming evidence since the previous plan (cleared).
    pub(crate) fn take_aim_evidence(&self) -> AimEvidence {
        let mut e = self.aim_evidence.lock().unwrap_or_else(|e| e.into_inner());
        let mut out = std::mem::take(&mut *e);
        out.design = self.aim.name();
        out
    }
    fn aim_host(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
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
            };
            host_fill(self.backend, &job, n, ctx)
        })??;
        self.evidence(|e| {
            e.generations += 1;
            e.rays += n as u64;
            e.host(&stats);
            e.host_seconds += start.elapsed().as_secs_f64();
        });
        Ok(handles)
    }
}

/// A submitted device aiming plan of rays `[start, start + count)` of a
/// generation.
pub(crate) struct DeviceAims {
    pending: Option<mklight_gpu::Pending<PlanResult>>,
    pub start: usize,
    pub count: usize,
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
pub(crate) const TABLE_SCHEDULE_CAP: u32 = 12;

impl Geometry<'_> {
    /// A host f64 seed table for the rows of a generation of `n` rays
    /// (module docs, [`AimDesign::Table`]).
    pub(crate) fn host_table(
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
    /// Submit a device aiming plan for rays `range` of a generation: host
    /// chief rows, the design's seed table and schedule.
    pub(crate) fn submit_device(
        &self,
        rows: &AimRows,
        range: std::ops::Range<usize>,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        table: Option<&(Vec<SeedNode>, u32)>,
        ctx: &JobContext,
    ) -> MkResult<DeviceAims> {
        let n = range.len();
        let mut a = ResidentPlan::new(self.tracer.system().snapshot.clone());
        let slots = self.slots(rows)?;
        let metadata = self.records::<FieldSpec>(&mut a, rows.len(), |i| {
            FieldSpec::new(
                self.tracer.system(),
                self.tracer.paraxial(),
                rows.field(i).point,
                slots[i].0,
                slots[i].1,
                rows.aiming,
            )
        })?;
        let samples = self.records::<Sample>(&mut a, n, |k| {
            if k % 4096 == 0 {
                ctx.check()?;
            }
            let s = spec(range.start + k);
            let mut x = Sample::new(s.pupil, s.row, s.id);
            x.weight = s.weight;
            Ok(x)
        })?;
        let field_key = self.field_key(&a, metadata)?;
        let seeds = self.with_tracer(rows.aiming, |t| {
            self.records::<ChiefSeed>(&mut a, rows.len(), |i| {
                Ok(ChiefSeed::host(t, rows.field(i), rows.wavelength(i)))
            })
        })??;
        let options = match table {
            Some((nodes, side)) => {
                let id = self.records::<SeedNode>(&mut a, nodes.len(), |i| Ok(nodes[i]))?;
                AimOptions {
                    table: Some(a.view(id)?),
                    table_side: *side,
                    schedule_cap: TABLE_SCHEDULE_CAP,
                    ..Default::default()
                }
            }
            None => Default::default(),
        };
        let out = rays(&mut a, n);
        a.operations
            .push(ResidentOp::Aiming(Request::GenerateAim(GenerateAim {
                systems: SystemSet::Uploaded(self.systems),
                samples: a.view(samples)?,
                sampling: Sampling::Explicit,
                unresolved: None,
                field_wavelength: a.view(metadata)?,
                field_key,
                seeds: Some(a.view(seeds)?),
                tolerance: self.device_tolerance(rows.aiming)?,
                max_iterations: 60,
                output: out.clone(),
                options,
            })));
        a.outputs = vec![out.floats, out.indices, out.logical_ids, out.report];
        let submitted = Instant::now();
        Ok(DeviceAims {
            pending: Some(self.backend.submit_plan(a, ctx)?),
            start: range.start,
            count: n,
            submitted,
        })
    }
    /// Wait for a device aiming plan: its four ray buffers and statistics.
    pub(crate) fn finish_device(
        &self,
        mut d: DeviceAims,
    ) -> MkResult<([BufferHandle; 4], resident::AimStats)> {
        let r = d.pending.take().unwrap().wait()?;
        let seconds = d.submitted.elapsed().as_secs_f64();
        let stats = r.stats.aiming.clone();
        self.evidence(|e| {
            e.device_plans += 1;
            e.device_seconds += seconds;
            e.device_rays += stats.rays;
            e.device_failed += stats.failed;
            e.device_iterations += stats.iterations;
        });
        let [f, u, i, rep]: [BufferHandle; 4] = r
            .outputs
            .try_into()
            .map_err(|_| bad("device aim outputs"))?;
        Ok(([f, u, i, rep], stats))
    }
    fn aim_device(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        table: Option<(Vec<SeedNode>, u32)>,
        ctx: &JobContext,
    ) -> MkResult<[BufferHandle; 4]> {
        let d = self.submit_device(rows, 0..n, spec, table.as_ref(), ctx)?;
        let (handles, _) = self.finish_device(d)?;
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
        let resolved: Vec<(u32, GeneratedRay)> = self.with_tracer(rows.aiming, |tracer| {
            HostJob {
                tracer,
                rows,
                slots: &slots,
                spec,
            }
            .aim_indices(&failed)
            .into_iter()
            .filter_map(|(i, g)| g.ok().map(|g| (i, g)))
            .collect()
        })?;
        ctx.check()?;
        let count = failed.len() as u64;
        let fixed = resolved.len() as u64;
        let out = if resolved.is_empty() {
            handles
        } else {
            let floats = self.backend.read_view(handles[0].view())?;
            let floats = floats.as_slice::<f32>()?;
            let patched_floats = self.backend.fill_buffer(
                BufferDesc::new(ElementType::F32, [FLOAT_PLANES, n]),
                &mut |mut f| {
                    let out = f.as_mut_slice::<f32>()?;
                    copy_parallel(out, floats);
                    for (i, g) in &resolved {
                        let i = *i as usize;
                        let s = spec(i);
                        let p = [g.ray.pos.x, g.ray.pos.y, g.ray.pos.z].map(Df32::from_f64);
                        let d = [g.ray.dir.x, g.ray.dir.y, g.ray.dir.z].map(Df32::from_f64);
                        for j in 0..3 {
                            out[j * n + i] = p[j].hi;
                            out[(3 + j) * n + i] = d[j].hi;
                            out[(9 + j) * n + i] = p[j].lo;
                            out[(12 + j) * n + i] = d[j].lo;
                        }
                        out[6 * n + i] = 0.0;
                        out[7 * n + i] = 0.0;
                        out[8 * n + i] = s.weight;
                    }
                    Ok(())
                },
            )?;
            let patched_report = self.backend.fill_buffer(
                BufferDesc::new(ElementType::U32, [4, n]),
                &mut |mut f| {
                    let out = f.as_mut_slice::<u32>()?;
                    copy_parallel(out, report);
                    for (i, g) in &resolved {
                        let i = *i as usize;
                        let w = report_words(&Ok(*g), true, 0);
                        for (k, v) in w.into_iter().enumerate() {
                            out[k * n + i] = v;
                        }
                    }
                    Ok(())
                },
            )?;
            let [_, indices, ids, _] = handles;
            [patched_floats, indices, ids, patched_report]
        };
        self.evidence(|e| {
            e.completion_rays += count;
            e.completion_resolved += fixed;
            e.completion_seconds += start.elapsed().as_secs_f64();
        });
        Ok(out)
    }
}

/// `dst.copy_from_slice(src)` in parallel blocks (bulk plane copies of
/// hundreds of megabytes on dense generations).
pub(crate) fn copy_parallel<T: Copy + Send + Sync>(dst: &mut [T], src: &[T]) {
    const BLOCK: usize = 1 << 16;
    dst.par_chunks_mut(BLOCK)
        .zip(src.par_chunks(BLOCK))
        .for_each(|(d, s)| d.copy_from_slice(s));
}

/// The split design ([`AimDesign::Split`]): host and device aim one
/// generation concurrently.
mod split_design {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Rays below which a generation is aimed on the host alone: the device
    /// plan's latency (seed table, submission, completion) exceeds the host
    /// time of the whole generation.
    pub const MIN_DEVICE_RAYS: usize = 1 << 15;
    /// Rays per device tile at most; two tiles stay in flight.
    pub const DEVICE_TILE: usize = 1 << 17;
    /// The device tile of a generation of `n` rays: a sixteenth of it within
    /// [2¹³, [`DEVICE_TILE`]], so the host keeps most of a small generation
    /// and a slow device never holds more than two tiles.
    pub fn device_tile(n: usize) -> usize {
        (n / 16).clamp(1 << 13, DEVICE_TILE)
    }

    /// Shared claim state: the device claims tiles from the front, host
    /// workers claim chunks from the back, until they meet.
    pub struct Claims {
        front: AtomicUsize,
        back: AtomicUsize,
        lock: Mutex<()>,
    }
    impl Claims {
        pub fn new(n: usize) -> Self {
            Self {
                front: AtomicUsize::new(0),
                back: AtomicUsize::new(n),
                lock: Mutex::new(()),
            }
        }
        /// The next device tile `[front, front + len)`, if it fits.
        pub fn device(&self, tile: usize) -> Option<std::ops::Range<usize>> {
            let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            let (f, b) = (
                self.front.load(Ordering::Relaxed),
                self.back.load(Ordering::Relaxed),
            );
            let len = tile.min(b - f);
            // A remainder smaller than a host chunk round is left to the host.
            if len < tile / 4 {
                return None;
            }
            self.front.store(f + len, Ordering::Relaxed);
            Some(f..f + len)
        }
        /// The next host chunk `[back − len, back)`, if any remains.
        pub fn host(&self, chunk: usize) -> Option<std::ops::Range<usize>> {
            let _g = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            let (f, b) = (
                self.front.load(Ordering::Relaxed),
                self.back.load(Ordering::Relaxed),
            );
            if b == f {
                return None;
            }
            let len = chunk.min(b - f);
            self.back.store(b - len, Ordering::Relaxed);
            Some(b - len..b)
        }
    }
}
pub(crate) use split_design::{MIN_DEVICE_RAYS, device_tile};

/// Raw plane pointers for the split design's concurrent writers: device
/// tiles and host chunks claim disjoint column ranges ([`split_design::Claims`]),
/// so no two writers touch the same element.
struct SharedColumns {
    floats: *mut f32,
    indices: *mut u32,
    ids: *mut u32,
    report: *mut u32,
    n: usize,
}
// SAFETY: writers only touch the disjoint column ranges they claimed.
unsafe impl Send for SharedColumns {}
unsafe impl Sync for SharedColumns {}
impl SharedColumns {
    /// Column block `range` (claimed exclusively by the caller).
    ///
    /// # Safety
    /// No other live block may overlap `range`.
    unsafe fn block(&self, range: std::ops::Range<usize>) -> RayColumns<'_> {
        let n = self.n;
        let len = range.len();
        // SAFETY: each plane of `n` elements is split at `range`, which the
        // caller holds exclusively; the storage outlives `self`.
        unsafe {
            let plane = |base: *mut f32, k: usize| {
                std::slice::from_raw_parts_mut(base.add(k * n + range.start), len)
            };
            let uplane = |base: *mut u32, k: usize| {
                std::slice::from_raw_parts_mut(base.add(k * n + range.start), len)
            };
            RayColumns {
                start: range.start,
                floats: (0..FLOAT_PLANES).map(|k| plane(self.floats, k)).collect(),
                indices: (0..2).map(|k| uplane(self.indices, k)).collect(),
                ids: (0..2).map(|k| uplane(self.ids, k)).collect(),
                report: (0..4).map(|k| uplane(self.report, k)).collect(),
            }
        }
    }
}

impl Geometry<'_> {
    fn aim_split(
        &self,
        rows: &AimRows,
        n: usize,
        spec: &(dyn Fn(usize) -> RaySpec + Sync),
        ctx: &JobContext,
    ) -> MkResult<[BufferHandle; 4]> {
        if n < MIN_DEVICE_RAYS {
            return self.aim_host(rows, n, spec, ctx);
        }
        let start = Instant::now();
        let slots = self.slots(rows)?;
        let table = self.host_table(rows, n, ctx)?;
        let claims = split_design::Claims::new(n);
        let mut host_stats = HostAims::default();
        let mut completion = (0u64, 0u64);
        let mut handles: Vec<BufferHandle> = Vec::new();
        let b = self.backend;
        let report = b.fill_buffer(BufferDesc::new(ElementType::U32, [4, n]), &mut |mut rf| {
            let report = rf.as_mut_slice::<u32>()?;
            let ids = b.fill_buffer(BufferDesc::new(ElementType::U32, [2, n]), &mut |mut xf| {
                let ids = xf.as_mut_slice::<u32>()?;
                let indices =
                    b.fill_buffer(BufferDesc::new(ElementType::U32, [2, n]), &mut |mut uf| {
                        let indices = uf.as_mut_slice::<u32>()?;
                        let floats = b.fill_buffer(
                            BufferDesc::new(ElementType::F32, [FLOAT_PLANES, n]),
                            &mut |mut ff| {
                                let floats = ff.as_mut_slice::<f32>()?;
                                let shared = SharedColumns {
                                    floats: floats.as_mut_ptr(),
                                    indices: indices.as_mut_ptr(),
                                    ids: ids.as_mut_ptr(),
                                    report: report.as_mut_ptr(),
                                    n,
                                };
                                let (h, c) = self.with_tracer(rows.aiming, |tracer| {
                                    let job = HostJob {
                                        tracer,
                                        rows,
                                        slots: &slots,
                                        spec,
                                    };
                                    self.split_run(&job, &claims, &shared, &table, ctx)
                                })??;
                                host_stats = h;
                                completion = c;
                                Ok(())
                            },
                        )?;
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
            handles.try_into().map_err(|_| bad("split aim buffers"))?;
        self.evidence(|e| {
            e.generations += 1;
            e.rays += n as u64;
            e.host(&host_stats);
            e.host_seconds += start.elapsed().as_secs_f64();
            e.completion_rays += completion.0;
            e.completion_resolved += completion.1;
        });
        Ok([floats, indices, ids, report])
    }
    /// The concurrent part of [`AimDesign::Split`]: this thread drives the
    /// device tiles (two in flight) and copies each completed tile into the
    /// shared planes, queueing its unconverged aims; rayon workers aim host
    /// chunks from the back meanwhile and, once the claims meet, complete
    /// the queued device failures as they arrive.
    fn split_run(
        &self,
        job: &HostJob<'_>,
        claims: &split_design::Claims,
        shared: &SharedColumns,
        table: &(Vec<SeedNode>, u32),
        ctx: &JobContext,
    ) -> MkResult<(HostAims, (u64, u64))> {
        use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
        let job_n = shared.n;
        let queue: Mutex<Vec<u32>> = Mutex::new(Vec::new());
        let done = AtomicBool::new(false);
        let (queued, resolved) = (AtomicU64::new(0), AtomicU64::new(0));
        let host = Mutex::new(HostAims::default());
        let error: Mutex<Option<mklight_core::MkError>> = Mutex::new(None);
        let workers = rayon::current_num_threads().max(1);
        rayon::scope(|s| {
            for _ in 0..workers {
                s.spawn(|_| {
                    let mut local = HostAims::default();
                    loop {
                        if ctx.check().is_err() {
                            break;
                        }
                        if let Some(range) = claims.host(HOST_CHUNK) {
                            // SAFETY: the claim is exclusive (`Claims`).
                            let mut cols = unsafe { shared.block(range) };
                            local = local.add(job.aim(&mut cols));
                            continue;
                        }
                        let finished = done.load(Ordering::Acquire);
                        let batch: Vec<u32> = {
                            let mut q = queue.lock().unwrap_or_else(|e| e.into_inner());
                            let k = q.len().saturating_sub(HOST_CHUNK);
                            q.split_off(k)
                        };
                        if batch.is_empty() {
                            if finished {
                                break;
                            }
                            std::thread::sleep(std::time::Duration::from_micros(200));
                            continue;
                        }
                        for (i, g) in job.aim_indices_serial(&batch) {
                            let Ok(g) = g else { continue };
                            resolved.fetch_add(1, Ordering::Relaxed);
                            let i = i as usize;
                            let sp = (job.spec)(i);
                            // SAFETY: the device copied this tile before
                            // queueing it, and each index is queued once.
                            let mut cols = unsafe { shared.block(i..i + 1) };
                            cols.write(
                                0,
                                Some((&g.ray, sp.weight)),
                                job.slots[sp.row as usize],
                                sp.id,
                                report_words(&Ok(g), true, 0),
                            );
                        }
                    }
                    let mut h = host.lock().unwrap_or_else(|e| e.into_inner());
                    *h = h.add(local);
                });
            }
            // The device side, on this thread.
            let r = (|| -> MkResult<()> {
                let mut inflight: std::collections::VecDeque<DeviceAims> = Default::default();
                loop {
                    while inflight.len() < 2 {
                        let Some(range) = claims.device(device_tile(job_n)) else {
                            break;
                        };
                        inflight.push_back(self.submit_device(
                            job.rows,
                            range,
                            job.spec,
                            Some(table),
                            ctx,
                        )?);
                    }
                    let Some(d) = inflight.pop_front() else {
                        break;
                    };
                    let (tile_start, count) = (d.start, d.count);
                    let (handles, _) = self.finish_device(d)?;
                    let views = handles
                        .iter()
                        .map(|h| self.backend.read_view(h.view()))
                        .collect::<MkResult<Vec<_>>>()?;
                    // SAFETY: the device claimed this tile exclusively.
                    let mut cols = unsafe { shared.block(tile_start..tile_start + count) };
                    let f = views[0].as_slice::<f32>()?;
                    let u = views[1].as_slice::<u32>()?;
                    let i = views[2].as_slice::<u32>()?;
                    let rep = views[3].as_slice::<u32>()?;
                    for (k, plane) in cols.floats.iter_mut().enumerate() {
                        plane.copy_from_slice(&f[k * count..(k + 1) * count]);
                    }
                    for (k, plane) in cols.indices.iter_mut().enumerate() {
                        plane.copy_from_slice(&u[k * count..(k + 1) * count]);
                    }
                    for (k, plane) in cols.ids.iter_mut().enumerate() {
                        plane.copy_from_slice(&i[k * count..(k + 1) * count]);
                    }
                    for (k, plane) in cols.report.iter_mut().enumerate() {
                        plane.copy_from_slice(&rep[k * count..(k + 1) * count]);
                    }
                    drop(cols);
                    let failed: Vec<u32> = (0..count)
                        .filter(|&k| rep[k] & MKL_RAY_CODE_MASK == MKL_AIM_FAILED)
                        .map(|k| (tile_start + k) as u32)
                        .collect();
                    queued.fetch_add(failed.len() as u64, Ordering::Relaxed);
                    queue
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .extend(failed);
                }
                Ok(())
            })();
            done.store(true, Ordering::Release);
            if let Err(e) = r {
                *error.lock().unwrap_or_else(|e| e.into_inner()) = Some(e);
            }
        });
        if let Some(e) = error.into_inner().unwrap_or_else(|e| e.into_inner()) {
            return Err(e);
        }
        ctx.check()?;
        Ok((
            host.into_inner().unwrap_or_else(|e| e.into_inner()),
            (queued.into_inner(), resolved.into_inner()),
        ))
    }
}
