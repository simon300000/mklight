//! P1 generation/trace/projection and P7 scalar transport, retaining lattice
//! channels for resampling and fits. Only small rim-loop topology returns to host.
//! Every pupil of a batch (fields × wavelengths) shares the chief, lattice,
//! saddle and rim-bisection commands; placement and resampling stay per pupil.
use super::*;
use crate::{AmplitudeModel, ExitPupilInfo, PupilMask, PupilSettings, ReferencePoint};
use mklight_core::Vec3;
use mklight_gpu::{
    PackOptions, PackedSystem, SystemHandle, aiming as aim, collectives as col,
    layout::MAX_WAVELENGTHS,
    radiometry as rad,
    wave_ops::{lattice::Lattice as Op, placement::Placement},
};
use mklight_trace::{PreparedField, SequentialTracer};

/// Lattice rays per batch. Bounds the per-pupil split copies and the rim
/// probe planes; larger lattices trace one pupil at a time.
const BATCH_RAYS: usize = 1 << 16;
/// Bracket subdivisions per rim-bisection round, as a power of two: each
/// round of `2^RIM_LEVELS` probes per open edge gains that many bits. One
/// level (plain bisection) aims the fewest rays; rounds are device
/// dispatches of one plan, not host round trips.
const RIM_LEVELS: u32 = 1;
/// Nodes per axis of the host f64 seed tables handed to every generation of
/// a batch (chief probe, lattice, saddle and rim probes alike, so a sample's
/// converged ray does not depend on the batch it was aimed in).
/// Measured on the complex lens: 13 nodes need 2.9 Newton iterations per
/// ray, 25 need 1.55 and 37 need 1.27 at the same device time, while the
/// host cost grows with the node count; a device-built table costs ~0.8 s
/// per edit.
const TABLE_SIDE: u32 = 25;
/// Continuation rounds committed without a pending-count readback: rays
/// seeded from the tables or the chief rows converge within the first
/// command (≤ 2 Newton iterations on the complex lens), so a plan's
/// generations need no host round trip; the backend verifies that at
/// completion and reruns the plan with read-back rounds otherwise. The
/// round shares the first command's command buffer (at most twice its
/// bounded time when every lane continued).
const SPECULATIVE_ROUNDS: u32 = 1;
/// Host seed-table rows and the seconds spent solving them.
struct HostTable {
    nodes: Vec<aim::SeedNode>,
    seconds: f64,
}
/// Relative stop residual (fraction of the stop radius) above which a
/// failed aim is a physically unaimable ray rather than a convergence
/// problem: four orders above the 1e-10 tolerance, far below the 5e-3–1e-1
/// residuals of vignetted off-axis rays on a fast lens.
const UNAIMABLE_RESIDUAL: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct ResidentTracedPupil {
    pub pupil: ResidentPupil,
    pub info: ExitPupilInfo,
    /// Resident lattice nodes (DF `[side, side, 4]`: u, v, unwrapped OPD,
    /// amplitude) and survival (u32) of a pupil the device traced; None
    /// when the host traced it (its lattice is in `host`).
    pub nodes: Option<BufferHandle>,
    pub valid: Option<BufferHandle>,
    pub mask: Option<PupilMask>,
    pub entrance_rim: Option<PupilMask>,
    /// Content key of this traced lattice ([`lattice_key`]): the shared
    /// analysis graph's `pupil_lattice` key for the same (tracer, field,
    /// wavelength, lattice settings), in the Metal double-float family; for
    /// a host-traced pupil the CPU f64 `pupil_lattice` key.
    pub key: mklight_core::memo::NodeKey,
    /// The reference lattice (and pupil) of a pupil the host traced (the
    /// native analyses, `gpu::split`); None when the device traced it.
    pub host: Option<super::split::HostPupil>,
}
/// The `pupil_lattice` node key of a Metal traced pupil: the CPU f64
/// [`crate::shared::lattice_key`] of the same inputs, in a family of its own
/// (double-float lattices are never substituted for f64 ones), scoped by the
/// radiometry library identity. Every generation of a batch aims from the
/// same host seeds, so a lattice does not depend on the batch it was traced
/// in and the key identifies its rays, masks and automatic grid exactly.
pub fn lattice_key(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    settings: &PupilSettings,
    library: &str,
) -> mklight_core::memo::NodeKey {
    mklight_core::memo::KeyHasher::new("mklight-wave/pupil_lattice/metal-df/1")
        .key(&crate::shared::lattice_key(tracer, field, wl, settings))
        .str(library)
        .finish()
}
/// Probe rays: generated starts, the five hit channels, survival and power.
struct Batch {
    starts: BufferHandle,
    hits: [BufferHandle; 5],
    samples: BufferHandle,
    mask: Option<BufferHandle>,
    power: Option<BufferHandle>,
    count: usize,
}
/// Outputs of an appended probe chain. IDs remain plan-local until the
/// complete composed plan publishes; validation reads only completed leases.
struct ProbeLayout {
    first: usize,
    len: usize,
    samples: Option<BufferHandle>,
    count: usize,
    survival: bool,
    power: bool,
}
impl Batch {
    fn import(&self, p: &mut ResidentPlan) -> aim::ResidentHits {
        let h = self.hits.each_ref().map(|h| p.import(h.view()));
        aim::ResidentHits {
            rays: aim::ResidentRays {
                floats: h[0],
                indices: h[1],
                logical_ids: h[2],
                report: h[3],
                count: self.count,
            },
            status: h[4],
            records: None,
            record_low: None,
        }
    }
    fn mask(&self) -> &BufferHandle {
        self.mask.as_ref().expect("survival mask of a final probe")
    }
    /// Channels that per-pupil projection reads, in `from_split` order.
    fn channels(&self) -> Vec<(BufferHandle, bool)> {
        let mut v: Vec<_> = self.hits.iter().map(|h| (h.clone(), false)).collect();
        v.push((self.starts.clone(), false));
        v.push((self.mask().clone(), false));
        v.extend(self.power.clone().map(|h| (h, true)));
        v
    }
    fn from_split(
        parts: &mut impl Iterator<Item = BufferHandle>,
        samples: &BufferHandle,
        power: bool,
        count: usize,
    ) -> Self {
        let mut next = || parts.next().unwrap();
        let hits = std::array::from_fn(|_| next());
        let starts = next();
        let mask = Some(next());
        Self {
            hits,
            starts,
            mask,
            power: power.then(next),
            samples: samples.clone(),
            count,
        }
    }
}
/// Field rows of one batch, their host f64 chief rows and seed tables, and
/// their content key.
struct Fields {
    buffer: BufferHandle,
    seeds: BufferHandle,
    /// Host f64 seed tables (`TABLE_SIDE` nodes per axis) of the rows, when
    /// the batch's lattice amortises them.
    table: Option<BufferHandle>,
    /// Reported by the first plan that uses the rows.
    host: std::cell::RefCell<Option<HostPreparation>>,
    key: String,
}
/// Per-pupil state carried between the batched stages.
struct Item<'f> {
    position: usize,
    /// Field row (and lattice copy) in the batch's lattice probe.
    row: usize,
    field: &'f PreparedField,
    wl: usize,
    spec: aim::FieldSpec,
    chief: [Df32; 7],
    inside: Vec<bool>,
    survived: usize,
    loops: Vec<Vec<usize>>,
    ids: Vec<usize>,
    /// First edge of this pupil in the shared rim batch.
    rim_start: usize,
}
pub struct PupilTracer<'a> {
    backend: &'a dyn ComputeBackend,
    tracer: &'a SequentialTracer,
    systems: SystemHandle,
    wavelengths: Vec<Vec<usize>>,
    /// [chunks, MAX_WAVELENGTHS] global wavelength of each chunk slot.
    table: Option<BufferHandle>,
    steps: BufferHandle,
    radiometry: rad::PreparedRadiometry,
}
impl Drop for PupilTracer<'_> {
    fn drop(&mut self) {
        let _ = self.backend.release_systems(self.systems);
    }
}
fn fill<T: BufferElement>(
    b: &dyn ComputeBackend,
    shape: impl Into<Vec<usize>>,
    mut f: impl FnMut(&mut [T]) -> MkResult<()>,
) -> MkResult<BufferHandle> {
    b.fill_buffer(BufferDesc::new(T::ELEMENT, shape), &mut |mut a| {
        f(a.as_mut_slice::<T>()?)
    })
}
fn rows<T: bytemuck::Pod>(b: &dyn ComputeBackend, v: &[T]) -> MkResult<BufferHandle> {
    fill::<u32>(b, [v.len(), size_of::<T>() / 4], |a| {
        a.copy_from_slice(bytemuck::cast_slice(v));
        Ok(())
    })
}
pub(crate) fn accumulate(a: &mut PlanStats, b: &PlanStats) {
    let offset = a.stages.len();
    a.stages.extend(b.stages.iter().cloned().map(|mut s| {
        s.operation += offset;
        s
    }));
    a.copied_bytes += b.copied_bytes;
    a.wall_seconds += b.wall_seconds;
    a.device_seconds += b.device_seconds;
    a.command_buffers += b.command_buffers;
    a.allocations += b.allocations;
    a.reused_allocations += b.reused_allocations;
    a.uploaded_bytes += b.uploaded_bytes;
    a.parameter_bytes += b.parameter_bytes;
    a.readback_bytes += b.readback_bytes;
    a.peak_bytes = a.peak_bytes.max(b.peak_bytes);
    a.fallback_count += b.fallback_count;
    a.polished_rays += b.polished_rays;
    a.host_waits += b.host_waits;
    a.host_wait_seconds += b.host_wait_seconds;
    a.long_commands += b.long_commands;
    a.add_activity(b);
}
fn subset(h: &BufferHandle, shape: impl Into<Vec<usize>>) -> BufferView {
    let mut v = h.view();
    v.desc.shape = shape.into();
    v
}
/// Trace and Numerical errors make one pupil (and its field) unusable;
/// every other error fails the whole batch, as it would the job.
fn per_pupil(e: MkError) -> MkResult<MkError> {
    match e {
        MkError::Trace(_) | MkError::Numerical(_) => Ok(e),
        e => Err(e),
    }
}
impl<'a> PupilTracer<'a> {
    pub fn new(
        backend: &'a dyn ComputeBackend,
        tracer: &'a SequentialTracer,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        ctx.check()?;
        let total = tracer.num_wavelengths();
        let primary = tracer.system().primary_wavelength;
        let wavelengths: Vec<Vec<usize>> = if total <= MAX_WAVELENGTHS {
            vec![(0..total).collect()]
        } else {
            (0..total)
                .step_by(MAX_WAVELENGTHS - 1)
                .map(|start| {
                    let mut w: Vec<_> = (start..(start + MAX_WAVELENGTHS - 1).min(total)).collect();
                    if !w.contains(&primary) {
                        w.push(primary);
                    }
                    w
                })
                .collect()
        };
        let packed: Vec<_> = wavelengths
            .iter()
            .map(|w| {
                PackedSystem::pack(
                    tracer.system(),
                    &PackOptions {
                        wavelengths: Some(w.clone()),
                        radius_hints: None,
                    },
                )
            })
            .collect::<MkResult<_>>()?;
        let table = if wavelengths.len() > 1 {
            Some(fill::<u32>(
                backend,
                [wavelengths.len() * MAX_WAVELENGTHS],
                |a| {
                    a.fill(u32::MAX);
                    for (c, w) in wavelengths.iter().enumerate() {
                        for (s, &wl) in w.iter().enumerate() {
                            a[c * MAX_WAVELENGTHS + s] = wl as u32;
                        }
                    }
                    Ok(())
                },
            )?)
        } else {
            None
        };
        let steps = fill::<u32>(backend, [tracer.system().image_index()], |a| {
            for (i, v) in a.iter_mut().enumerate() {
                *v = (i + 1) as u32;
            }
            Ok(())
        })?;
        let radiometry = rad::PreparedRadiometry::new(backend, tracer.system(), ctx)?;
        let key = SystemCacheKey {
            snapshot: tracer.system().snapshot.hash().to_hex(),
            library: radiometry.identity.clone(),
            configuration: String::new(),
            spectrum: String::new(),
            layout: mklight_gpu::consts::MKL_LAYOUT_VERSION,
            kernel_build: backend.precision().kernel_build.clone(),
            operation_settings: "P6 pupil tracing v1".into(),
        };
        let systems = backend.upload_systems_cached(&key, &packed)?;
        Ok(Self {
            backend,
            tracer,
            systems,
            wavelengths,
            table,
            steps,
            radiometry,
        })
    }
    fn plan(&self) -> ResidentPlan {
        let mut p = ResidentPlan::new(self.tracer.system().snapshot.clone());
        p.identity.library_hash = self.radiometry.identity.clone();
        p
    }
    /// Rows of (field, wavelength) pupils with their metadata `specs`. Chief
    /// rows come from the prepared fields (f64 source once per field, chief
    /// and Jacobian per wavelength); the device solves none of them. With
    /// `rays_per_row` lattice rays per row the host also solves the seed
    /// tables when the lattice amortises them (twice the node count).
    fn fields(
        &self,
        pupils: &[(&PreparedField, usize)],
        specs: &[aim::FieldSpec],
        table: Option<HostTable>,
    ) -> MkResult<Fields> {
        let start = std::time::Instant::now();
        let seeds: Vec<_> = pupils
            .iter()
            .map(|(f, w)| aim::ChiefSeed::host(self.tracer, f, *w))
            .collect();
        let table_seconds = table.as_ref().map_or(0., |t| t.seconds);
        let table = table.map(|t| rows(self.backend, &t.nodes)).transpose()?;
        let mut points: Vec<_> = Vec::new();
        for (f, _) in pupils {
            if !points.contains(&f.point) {
                points.push(f.point);
            }
        }
        let host = HostPreparation {
            seed_rows: seeds.len() as u64,
            reused_fields: points.len() as u64,
            solved_fields: 0,
            seconds: start.elapsed().as_secs_f64() + table_seconds,
        };
        Ok(Fields {
            buffer: rows(self.backend, specs)?,
            seeds: rows(self.backend, &seeds)?,
            table,
            host: std::cell::RefCell::new(Some(host)),
            key: blake3::hash(bytemuck::cast_slice(specs))
                .to_hex()
                .to_string(),
        })
    }
    /// Host f64 seed tables of `pupils` when real aiming with `rays_per_row`
    /// lattice rays per row amortises them (twice the node count).
    fn tables(&self, pupils: &[(&PreparedField, usize)], rays_per_row: usize) -> Option<HostTable> {
        let real = matches!(self.tracer.aiming(), mklight_core::RayAiming::Real { .. });
        (real && rays_per_row >= 2 * aim::table_nodes(TABLE_SIDE)).then(|| {
            let start = std::time::Instant::now();
            let nodes = aim::host_tables(self.tracer, pupils, TABLE_SIDE);
            HostTable {
                nodes,
                seconds: start.elapsed().as_secs_f64(),
            }
        })
    }
    fn spec(&self, field: &PreparedField, wl: usize) -> MkResult<aim::FieldSpec> {
        let (chunk, slot) = self
            .wavelengths
            .iter()
            .enumerate()
            .find_map(|(i, w)| w.iter().position(|v| *v == wl).map(|j| (i, j)))
            .unwrap();
        aim::FieldSpec::new(
            self.tracer.system(),
            self.tracer.paraxial(),
            field.point,
            chunk as u32,
            slot as u32,
            self.tracer.aiming(),
        )
    }
    /// Chief rays of every field row: hi+lo position, direction and OPL, or
    /// None where the chief fails.
    fn chief_rays(
        &self,
        pupils: &[(&PreparedField, usize)],
        specs: &[aim::FieldSpec],
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<Vec<Option<[Df32; 7]>>> {
        let fields = self.fields(pupils, specs, None)?;
        let samples: Vec<_> = (0..specs.len())
            .map(|q| aim::Sample::new([0.; 2], q as u32, q as u64))
            .collect();
        let samples = rows(self.backend, &samples)?;
        let mut p = self.plan();
        let s = p.import(samples.view());
        let (chief, _, s) = self.probe(
            p,
            s,
            specs.len(),
            &fields,
            false,
            false,
            false,
            false,
            &[],
            ctx,
        )?;
        accumulate(stats, &s);
        let status = self.backend.read_view(chief.hits[4].view())?;
        let status = status.as_slice::<u32>()?;
        let c = self.backend.read_view(chief.hits[0].view())?;
        let c = c.as_slice::<f32>()?;
        let m = specs.len();
        Ok((0..m)
            .map(|q| {
                (status[q] == 0).then(|| {
                    std::array::from_fn(|j| {
                        let (h, l) = if j < 6 { (j, j + 9) } else { (6, 7) };
                        Df32::new(c[h * m + q], c[l * m + q])
                    })
                })
            })
            .collect())
    }
    #[allow(clippy::too_many_arguments)]
    fn probe(
        &self,
        mut p: ResidentPlan,
        samples: BufferId,
        count: usize,
        fields: &Fields,
        cull: bool,
        clip: bool,
        survival: bool,
        power: bool,
        extra: &[BufferId],
        ctx: &JobContext,
    ) -> MkResult<(Batch, Vec<BufferHandle>, PlanStats)> {
        let layout = self.append_probe(
            &mut p, samples, count, fields, cull, clip, survival, power, extra,
        )?;
        let r = self.backend.submit_plan(p, ctx)?.wait()?;
        let (batch, extra) = self.finish_probe(&layout, &r, ctx)?;
        Ok((batch, extra, r.stats))
    }
    /// Append a dependent chain without submitting or exposing a host view.
    /// `cull` excludes samples outside the unit pupil disk from aiming (the
    /// lattice guard nodes, which the survival mask discards anyway).
    #[allow(clippy::too_many_arguments)]
    fn append_probe(
        &self,
        p: &mut ResidentPlan,
        samples: BufferId,
        count: usize,
        fields: &Fields,
        cull: bool,
        clip: bool,
        survival: bool,
        power: bool,
        extra: &[BufferId],
    ) -> MkResult<ProbeLayout> {
        if let Some(h) = fields.host.borrow_mut().take() {
            p.host.add(&h);
        }
        let steps = p.import(self.steps.view());
        let generated = self.append_aim(p, samples, count, fields, cull)?;
        let hits = aim::hits(
            p,
            count,
            if power {
                self.tracer.system().surfaces.len()
            } else {
                0
            },
        );
        p.operations
            .push(ResidentOp::Aiming(aim::Request::TraceResident(
                aim::TraceResident {
                    systems: SystemSet::Uploaded(self.systems),
                    input: generated.clone(),
                    steps: p.view(steps)?,
                    start_surface: 0,
                    apply_apertures: clip,
                    radiometry: 0,
                    newton: Default::default(),
                    output: hits.clone(),
                },
            )));
        let mut outputs = vec![generated.floats];
        outputs.extend([
            hits.rays.floats,
            hits.rays.indices,
            hits.rays.logical_ids,
            hits.rays.report,
            hits.status,
        ]);
        if survival {
            let mask = p.scratch(BufferDesc::new(ElementType::U32, [count]));
            p.operations
                .push(ResidentOp::Wave(w::Request::Lattice(Op::Survival {
                    samples: p.view(samples)?,
                    status: p.view(hits.status)?,
                    output: mask,
                })));
            outputs.push(mask);
        }
        if power {
            let mut radiometric_hits = hits.clone();
            if let Some(table) = &self.table {
                let table = p.import(table.view());
                let indices = p.scratch(BufferDesc::new(ElementType::U32, [2, count]));
                p.operations
                    .push(ResidentOp::Wave(w::Request::Lattice(Op::Wavelength {
                        indices: p.view(hits.rays.indices)?,
                        table: p.view(table)?,
                        output: indices,
                    })));
                radiometric_hits.rays.indices = indices;
            }
            outputs.push(self.radiometry.append(
                p,
                &radiometric_hits,
                &mklight_trace::PolarizationState::Unpolarized,
            )?);
            if let Some(ResidentOp::Radiometry(rad::Request::Polarization(r))) =
                p.operations.last_mut()
            {
                r.scalar = true;
            }
        }
        // Imported sample leases already survive; generated samples must publish.
        let sample_handle = match &p.buffers[samples.0] {
            PlanBuffer::Import(v) => Some(v.buffer.clone()),
            _ => {
                outputs.push(samples);
                None
            }
        };
        outputs.extend(extra);
        let first = p.outputs.len();
        let len = outputs.len();
        p.outputs.extend(outputs);
        Ok(ProbeLayout {
            first,
            len,
            samples: sample_handle,
            count,
            survival,
            power,
        })
    }
    /// Append device aiming of `samples` (GenerateAim with the host seed
    /// rows and tables of `fields`).
    fn append_aim(
        &self,
        p: &mut ResidentPlan,
        samples: BufferId,
        count: usize,
        fields: &Fields,
        cull: bool,
    ) -> MkResult<aim::ResidentRays> {
        let metadata = p.import(fields.buffer.view());
        let seeds = p.import(fields.seeds.view());
        let table = fields.table.as_ref().map(|t| p.import(t.view()));
        let generated = aim::rays(p, count);
        // The tightest derived stop of the prepared fields, or the requested
        // tolerance (CONVENTIONS §7, the aim-stop policy).
        let tolerance = match self.tracer.aiming() {
            mklight_core::RayAiming::Real { .. } => self.tracer.device_aim_tolerance() as f32,
            _ => 1e-10,
        };
        p.operations
            .push(ResidentOp::Aiming(aim::Request::GenerateAim(
                aim::GenerateAim {
                    systems: SystemSet::Uploaded(self.systems),
                    samples: p.view(samples)?,
                    sampling: aim::Sampling::Explicit,
                    unresolved: None,
                    field_wavelength: p.view(metadata)?,
                    field_key: Some(fields.key.clone()),
                    seeds: Some(p.view(seeds)?),
                    tolerance,
                    max_iterations: 60,
                    output: generated.clone(),
                    options: aim::AimOptions {
                        table: table.map(|t| p.view(t)).transpose()?,
                        table_side: TABLE_SIDE,
                        cull_outside: cull,
                        speculative_rounds: SPECULATIVE_ROUNDS,
                        // The wave lattice keeps the full schedule: every
                        // unresolved ray is checked against the reference.
                        schedule_cap: 0,
                    },
                },
            )));
        Ok(generated)
    }
    fn finish_probe(
        &self,
        layout: &ProbeLayout,
        result: &PlanResult,
        ctx: &JobContext,
    ) -> MkResult<(Batch, Vec<BufferHandle>)> {
        let r = &result.outputs[layout.first..layout.first + layout.len];
        // The device-aimed starts are the first output.
        let (starts, r) = (r[0].clone(), &r[1..]);
        let (count, survival, power) = (layout.count, layout.survival, layout.power);
        let sample_handle = layout.samples.clone();
        let mut rest = r[5..].iter().cloned();
        let mask = survival.then(|| rest.next().unwrap());
        let power = power.then(|| rest.next().unwrap());
        let samples = sample_handle.unwrap_or_else(|| rest.next().unwrap());
        let extra = rest.collect();
        self.check_aim(&samples, &r[3], ctx)?;
        if let Some(power) = &power {
            let v = self.backend.read_view(power.view())?;
            let rows: &[rad::PolarizationValue] = bytemuck::cast_slice(v.words());
            let count = rows
                .iter()
                .filter(|v| v.status == mklight_gpu::consts::MKL_RAD_REFERENCE)
                .count();
            if count > 0 {
                return Err(MkError::Unsupported(format!(
                    "P7 pupil transport requires reference completion for {count} rows"
                )));
            }
        }
        Ok((
            Batch {
                starts,
                hits: std::array::from_fn(|i| r[i].clone()),
                samples,
                mask,
                power,
                count,
            },
            extra,
        ))
    }
    fn check_aim(
        &self,
        samples: &BufferHandle,
        reports: &BufferHandle,
        ctx: &JobContext,
    ) -> MkResult<()> {
        let sample_guard = self.backend.read_view(samples.view())?;
        let sample_rows: &[aim::Sample] = bytemuck::cast_slice(sample_guard.words());
        let reports = self.backend.read_view(reports.view())?;
        let reports = reports.as_slice::<u32>()?;
        let n = sample_rows.len();
        let mut unresolved = 0usize;
        let mut detail = String::new();
        for (k, sample) in sample_rows.iter().enumerate() {
            if k % 4096 == 0 {
                ctx.check()?;
            }
            // Excluded samples (culled guard nodes, probes of converged rim
            // edges) were never aimed. A failed aim whose stop residual is
            // far above the tolerance, or whose stop is unreachable (an
            // infinite residual, NaN once divided by the stop radius in
            // double-float), is a ray that cannot be aimed through the stop
            // at this field: the reference treats it as vignetted
            // (`Tracing::ray` is None), and so does the survival mask here.
            // Only tolerance-level non-convergence (a precision symptom)
            // needs the reference.
            let residual = f32::from_bits(reports[2 * n + k]);
            let unaimable = residual.is_nan() || residual > UNAIMABLE_RESIDUAL;
            if !matches!(reports[k], 0 | mklight_gpu::consts::MKL_AIM_OUTSIDE)
                && !unaimable
                && (sample.pupil[0] * sample.pupil[0] + sample.pupil[1] * sample.pupil[1]).to_f64()
                    <= 1. + 1e-12
            {
                if unresolved < 4 {
                    detail += &format!(
                        " row {} p=({:.4}, {:.4}) status {} after {} iterations, residual {:.2e};",
                        sample.field,
                        sample.pupil[0].to_f64(),
                        sample.pupil[1].to_f64(),
                        reports[k],
                        reports[n + k],
                        f32::from_bits(reports[2 * n + k])
                    );
                }
                unresolved += 1;
            }
        }
        if unresolved > 0 {
            return Err(MkError::Unsupported(format!(
                "P6 has {unresolved} unresolved in-pupil aims; reference completion required (e.g.{detail})"
            )));
        }
        Ok(())
    }
    /// Copy each pupil's range of a shared probe batch into its own channels.
    /// A single pupil spanning the batch keeps the batch itself.
    fn split(
        &self,
        batch: &Batch,
        ranges: &[[usize; 2]],
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<Vec<Batch>> {
        let power = batch.power.is_some();
        if let [[0, n]] = ranges
            && *n == batch.count
        {
            let mut parts = batch.channels().into_iter().map(|(h, _)| h);
            return Ok(vec![Batch::from_split(
                &mut parts,
                &batch.samples,
                power,
                *n,
            )]);
        }
        let mut p = self.plan();
        let channels: Vec<_> = batch
            .channels()
            .into_iter()
            .map(|(h, rows)| (p.import(h.view()), rows))
            .collect();
        let mut views = vec![];
        let mut outputs = vec![];
        for &(id, rows) in &channels {
            let v = p.view(id)?;
            for &[_, len] in ranges {
                let mut shape = v.desc.shape.clone();
                let axis = if rows { 0 } else { shape.len() - 1 };
                shape[axis] = len;
                outputs.push(p.scratch(BufferDesc::new(v.desc.element, shape)));
            }
            views.push((v, rows));
        }
        p.operations
            .push(ResidentOp::Wave(w::Request::Lattice(Op::Split {
                channels: views,
                ranges: ranges.to_vec(),
                outputs: outputs.clone(),
            })));
        p.outputs = outputs;
        let r = self.backend.submit_plan(p, ctx)?.wait()?;
        accumulate(stats, &r.stats);
        // Outputs are channel-major; regroup them per range.
        let m = ranges.len();
        Ok(ranges
            .iter()
            .enumerate()
            .map(|(k, &[_, len])| {
                let mut parts = (0..channels.len()).map(|c| r.outputs[c * m + k].clone());
                Batch::from_split(&mut parts, &batch.samples, power, len)
            })
            .collect())
    }
    pub fn trace(
        &self,
        field: &PreparedField,
        wl: usize,
        settings: &PupilSettings,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<ResidentTracedPupil>> {
        let (mut r, shared) = self.trace_many(&[(field, wl)], settings, ctx)?;
        let mut r = r.pop().unwrap()?;
        let mut stats = shared;
        accumulate(&mut stats, &r.stats);
        r.stats = stats;
        Ok(r)
    }
    /// Trace every (field, wavelength) pupil with batched probes and rim
    /// bisection. Trace/Numerical failures are returned per pupil; any other
    /// error fails the batch. The second value holds the shared stages' stats.
    #[allow(clippy::type_complexity)]
    pub fn trace_many(
        &self,
        pupils: &[(&PreparedField, usize)],
        settings: &PupilSettings,
        ctx: &JobContext,
    ) -> MkResult<(Vec<MkResult<WaveResult<ResidentTracedPupil>>>, PlanStats)> {
        settings.validate()?;
        ctx.check()?;
        let side = settings.rays + 2;
        let per = (BATCH_RAYS / (side * side)).max(1);
        let mut results = Vec::with_capacity(pupils.len());
        let mut stats = PlanStats::default();
        let chunks = pupils.len().div_ceil(per);
        for (c, chunk) in pupils.chunks(per).enumerate() {
            let sub = ctx.subrange(c as f64 / chunks as f64, (c + 1) as f64 / chunks as f64);
            results.extend(self.trace_batch(chunk, settings, &sub, &mut stats)?);
        }
        Ok((results, stats))
    }
    fn trace_batch(
        &self,
        pupils: &[(&PreparedField, usize)],
        settings: &PupilSettings,
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<Vec<MkResult<WaveResult<ResidentTracedPupil>>>> {
        let sys = self.tracer.system();
        let side = settings.rays + 2;
        let n = side * side;
        let mut results: Vec<Option<MkResult<_>>> = pupils.iter().map(|_| None).collect();
        let mut items = vec![];
        for (position, &(field, wl)) in pupils.iter().enumerate() {
            if wl >= sys.wavelengths_um.len() {
                return Err(MkError::InvalidParam(format!(
                    "wavelength {wl} does not exist"
                )));
            }
            // A wavelength-only selected-order factor cancels in chief-normalized
            // amplitudes and centroid weights. Zero throughput remains unusable.
            if settings.amplitude != AmplitudeModel::Uniform
                && mklight_trace::radiometry::diffraction_product(sys, wl) <= 0.
            {
                results[position] = Some(Err(MkError::Numerical(
                    "pupil transmits no selected-order light".into(),
                )));
                continue;
            }
            match self.spec(field, wl) {
                Ok(spec) => items.push(Item {
                    position,
                    field,
                    wl,
                    spec,
                    row: 0,
                    chief: [Df32::ZERO; 7],
                    inside: vec![],
                    survived: 0,
                    loops: vec![],
                    ids: vec![],
                    rim_start: 0,
                }),
                Err(e) => results[position] = Some(Err(per_pupil(e)?)),
            }
        }
        // Chief rays, one per field row. The rows' host seed tables are
        // solved meanwhile (rows are independent, so the kept rows are
        // bitwise those of a table of the kept pupils alone).
        let mut table = None;
        if !items.is_empty() {
            let specs: Vec<_> = items.iter().map(|i| i.spec).collect();
            let pupils: Vec<_> = items.iter().map(|i| (i.field, i.wl)).collect();
            let (chiefs, tables) = std::thread::scope(|s| {
                let tables = s.spawn(|| self.tables(&pupils, n));
                let chiefs = self.chief_rays(&pupils, &specs, &ctx.subrange(0., 0.03), stats);
                (chiefs, tables.join().expect("host seed tables"))
            });
            let chiefs = chiefs?;
            let nodes = aim::table_nodes(TABLE_SIDE);
            let mut kept_rows = vec![];
            let mut kept = vec![];
            for (q, (mut item, chief)) in items.into_iter().zip(chiefs).enumerate() {
                if matches!(&chief, Some(c) if c[5].to_f64() != 0.) {
                    kept_rows.push(q);
                }
                let error = match chief {
                    None => "chief ray failed",
                    Some(c) if c[5].to_f64() == 0. => {
                        "chief ray runs parallel to the image surface"
                    }
                    Some(c) => {
                        item.chief = c;
                        kept.push(item);
                        continue;
                    }
                };
                results[item.position] = Some(Err(MkError::Trace(error.into())));
            }
            items = kept;
            table = tables.map(|t| HostTable {
                nodes: kept_rows
                    .iter()
                    .flat_map(|&q| t.nodes[q * nodes..(q + 1) * nodes].iter().copied())
                    .collect(),
                seconds: t.seconds,
            });
        }
        // Guard-ringed lattices; field row q is item q.
        let mut lattice_batch = None;
        if !items.is_empty() {
            let specs: Vec<_> = items.iter().map(|i| i.spec).collect();
            let pupils: Vec<_> = items.iter().map(|i| (i.field, i.wl)).collect();
            let fields = self.fields(&pupils, &specs, table.take())?;
            let mut p = self.plan();
            let probes = p.scratch(BufferDesc::new(ElementType::U32, [items.len() * n, 8]));
            p.operations
                .push(ResidentOp::Wave(w::Request::Lattice(Op::Generate {
                    side,
                    copies: items.len(),
                    samples: probes,
                })));
            // Guard nodes outside the disk are never aimed: the survival
            // mask excludes them whatever their trace, so the lattice
            // topology and rim edges are unchanged.
            let (lat, _, s) = self.probe(
                p,
                probes,
                items.len() * n,
                &fields,
                true,
                true,
                true,
                settings.amplitude != AmplitudeModel::Uniform,
                &[],
                &ctx.subrange(0.03, 0.25),
            )?;
            accumulate(stats, &s);
            let valid = self.backend.read_view(lat.mask().view())?;
            let valid = valid.as_slice::<u32>()?;
            let mut kept = vec![];
            for (q, mut item) in items.into_iter().enumerate() {
                item.row = q;
                item.inside = valid[q * n..(q + 1) * n].iter().map(|v| *v != 0).collect();
                item.survived = item.inside.iter().filter(|&&v| v).count();
                if item.survived == 0 {
                    results[item.position] = Some(Err(MkError::Trace(
                        "no ray of pupil lattice reaches image".into(),
                    )));
                } else {
                    kept.push(item);
                }
            }
            items = kept;
            lattice_batch = Some((lat, fields));
        }
        let lattice = crate::traced::Lattice::new(settings.rays);
        let mut rim_batch = None;
        if let Some((_, fields)) = &lattice_batch
            && settings.rim
            && !items.is_empty()
        {
            // Saddle cells resolve their centre with one more probe each.
            let mut saddles = Vec::new();
            for (t, item) in items.iter().enumerate() {
                let inside = &item.inside;
                for j in 0..side - 1 {
                    for i in 0..side - 1 {
                        let k = j * side + i;
                        let case = usize::from(inside[k])
                            | (usize::from(inside[k + 1]) << 1)
                            | (usize::from(inside[k + side + 1]) << 2)
                            | (usize::from(inside[k + side]) << 3);
                        if case == 5 || case == 10 {
                            saddles.push((t, i, j));
                        }
                    }
                }
            }
            let mut centres = vec![std::collections::BTreeMap::new(); items.len()];
            if !saddles.is_empty() {
                let h = 1. / (settings.rays - 1) as f64;
                let samples: Vec<_> = saddles
                    .iter()
                    .enumerate()
                    .map(|(k, &(t, i, j))| {
                        let p = lattice.point(i, j);
                        aim::Sample::new([p[0] + h, p[1] + h], items[t].row as u32, k as u64)
                    })
                    .collect();
                let samples = rows(self.backend, &samples)?;
                let mut plan = self.plan();
                let samples = plan.import(samples.view());
                let (probes, _, s) = self.probe(
                    plan,
                    samples,
                    saddles.len(),
                    fields,
                    true,
                    true,
                    true,
                    false,
                    &[],
                    &ctx.subrange(0.25, 0.25),
                )?;
                accumulate(stats, &s);
                let mask = self.backend.read_view(probes.mask().view())?;
                for (&(t, i, j), &v) in saddles.iter().zip(mask.as_slice::<u32>()?) {
                    centres[t].insert((i, j), v != 0);
                }
            }
            let mut edges = vec![];
            let mut rows_of = vec![];
            let mut nodes_of = vec![];
            for (t, item) in items.iter_mut().enumerate() {
                let loops = crate::traced::chain_loops(&crate::traced::rim_segments(
                    &lattice,
                    &item.inside,
                    |i, j| centres[t][&(i, j)],
                ));
                let mut ids: Vec<usize> = loops.iter().flatten().copied().collect();
                ids.sort_unstable();
                ids.dedup();
                item.loops = loops
                    .iter()
                    .map(|l| l.iter().map(|i| ids.binary_search(i).unwrap()).collect())
                    .collect();
                item.rim_start = rows_of.len();
                for &id in &ids {
                    let k = id / 2;
                    let q = k + if id % 2 == 0 { 1 } else { side };
                    let (a, b) = if item.inside[k] { (k, q) } else { (q, k) };
                    nodes_of.push((item.row * n + a) as u32);
                    let a = lattice.point(a % side, a / side);
                    let b = lattice.point(b % side, b / side);
                    edges.extend([a[0], a[1], b[0], b[1]].map(Df32::from_f64));
                    rows_of.push(item.row as u32);
                }
                item.ids = ids;
            }
            if !rows_of.is_empty() {
                let (lat, _) = lattice_batch.as_ref().unwrap();
                rim_batch = Some(self.bisect(
                    &edges,
                    &rows_of,
                    &nodes_of,
                    lat,
                    fields,
                    settings,
                    &items,
                    &ctx.subrange(0.25, 0.6),
                    stats,
                )?);
            }
        }
        // Per-pupil copies of the shared probes for placement.
        let mut lats = vec![];
        let mut rims = vec![];
        if let Some((lat, _)) = &lattice_batch
            && !items.is_empty()
        {
            let ranges: Vec<_> = items.iter().map(|i| [i.row * n, n]).collect();
            lats = self.split(lat, &ranges, &ctx.subrange(0.6, 0.6), stats)?;
            if let Some(rim) = &rim_batch {
                let ranges: Vec<_> = items
                    .iter()
                    .filter(|i| !i.ids.is_empty())
                    .map(|i| [i.rim_start, i.ids.len()])
                    .collect();
                rims = self.split(rim, &ranges, &ctx.subrange(0.6, 0.6), stats)?;
            }
        }
        let total = items.len();
        let mut rims = rims.into_iter();
        let work: Vec<_> = items
            .iter()
            .zip(lats)
            .map(|(item, lat)| {
                let rim = (!item.ids.is_empty()).then(|| rims.next().unwrap());
                (item, lat, rim)
            })
            .collect();
        // Pupils place independently: their plan chains run side by side.
        let placed = super::concurrent(&work, |k, (item, lat, rim)| {
            let sub = ctx.subrange(
                0.6 + 0.4 * k as f64 / total as f64,
                0.6 + 0.4 * (k + 1) as f64 / total as f64,
            );
            self.place(item, lat, rim.as_ref(), settings, &sub)
        });
        for ((item, _, _), r) in work.iter().zip(placed) {
            results[item.position] = Some(r.or_else(|e| Err(per_pupil(e)?)));
        }
        Ok(results.into_iter().map(Option::unwrap).collect())
    }
    /// Batched rim bisection of every pupil's crossed lattice edges,
    /// `RIM_LEVELS` levels (`2^RIM_LEVELS` probes) per open edge and command
    /// round; converged edges emit skip-flagged probes. Each edge's rim ray
    /// is the surviving bracket end's ray as traced in its own round (first
    /// the inside lattice node), kept with `Op::Keep` exactly as the
    /// reference keeps the bracketing ray: tracing the final point again
    /// would aim it from other seeds, and a ray converged within the aiming
    /// tolerance of an aperture edge may then clip, leaving a hole in the rim.
    #[allow(clippy::too_many_arguments)]
    fn bisect(
        &self,
        edges: &[Df32],
        rows_of: &[u32],
        nodes_of: &[u32],
        lat: &Batch,
        fields: &Fields,
        settings: &PupilSettings,
        items: &[Item],
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<Batch> {
        let count = rows_of.len();
        let edges = fill::<Df32>(self.backend, [count, 4], |a| {
            a.copy_from_slice(edges);
            Ok(())
        })?;
        let edge_rows = fill::<u32>(self.backend, [count], |a| {
            a.copy_from_slice(rows_of);
            Ok(())
        })?;
        let nodes = fill::<u32>(self.backend, [count], |a| {
            a.copy_from_slice(nodes_of);
            Ok(())
        })?;
        // Every kept ray survived by construction.
        let survived = fill::<u32>(self.backend, [count], |a| {
            a.fill(1);
            Ok(())
        })?;
        let levels = RIM_LEVELS;
        let iterations = ((2. / (settings.rays - 1) as f64 / settings.rim_tolerance)
            .log2()
            .ceil() as usize)
            .div_ceil(levels as usize)
            + 1;
        let parts = 1usize << levels;
        let mut p = self.plan();
        let edge = p.import(edges.view());
        let edge_rows = p.import(edge_rows.view());
        // Kept channels in `Batch` order: starts, the five hit channels and
        // the sample rows; first those of the inside lattice nodes.
        let mut lattice = vec![];
        for h in std::iter::once(&lat.starts).chain(&lat.hits) {
            let id = p.import(h.view());
            lattice.push((p.view(id)?, false));
        }
        let samples = p.import(lat.samples.view());
        lattice.push((p.view(samples)?, true));
        let nodes = p.import(nodes.view());
        let nodes = p.view(nodes)?;
        let mut kept = Self::keep(&mut p, lattice, None, nodes);
        let mut previous = None;
        let mut status = None;
        let mut probes = vec![];
        let mut traced = None;
        for iteration in 0..=iterations {
            ctx.check()?;
            let last = iteration == iterations;
            let state = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [count, 8]));
            let source = p.scratch(BufferDesc::new(ElementType::U32, [count]));
            let probe_count = count * if last { 1 } else { parts };
            let samples = p.scratch(BufferDesc::new(ElementType::U32, [probe_count, 8]));
            p.operations
                .push(ResidentOp::Wave(w::Request::Lattice(Op::Rim {
                    edges: p.view(edge)?,
                    fields: p.view(edge_rows)?,
                    previous: previous.map(|i| p.view(i)).transpose()?,
                    status: status.map(|i| p.view(i)).transpose()?,
                    tolerance: Df32::from_f64(settings.rim_tolerance),
                    levels,
                    final_point: last,
                    state,
                    samples,
                    source,
                })));
            if let Some(channels) = traced.take() {
                let source = p.view(source)?;
                kept = Self::keep(&mut p, channels, Some(kept), source);
            }
            previous = Some(state);
            if last {
                break;
            }
            let layout = self.append_probe(
                &mut p,
                samples,
                probe_count,
                fields,
                false,
                true,
                false,
                false,
                &[],
            )?;
            // Probe publication order: starts, four ray channels, trace
            // status, generated samples.
            let ids = p.outputs[layout.first..layout.first + layout.len].to_vec();
            status = Some(ids[5]);
            let mut channels = vec![];
            for &id in &ids[..6] {
                channels.push((p.view(id)?, false));
            }
            channels.push((p.view(ids[6])?, true));
            traced = Some(channels);
            // Intermediate rounds publish only aim reports and input samples
            // for the qualification checks. Keep hit/status IDs plan-local.
            p.outputs.truncate(layout.first);
            probes.push((p.outputs.len(), p.outputs.len() + 1));
            p.outputs.extend([ids[4], ids[6]]);
        }
        let state_output = p.outputs.len();
        p.outputs.push(previous.unwrap());
        let kept_output = p.outputs.len();
        p.outputs.extend(kept);
        let result = self.backend.submit_plan(p, ctx)?.wait()?;
        accumulate(stats, &result.stats);
        for &(reports, samples) in &probes {
            self.check_aim(&result.outputs[samples], &result.outputs[reports], ctx)?;
        }
        let state = self
            .backend
            .read_view(result.outputs[state_output].view())?;
        let state = state.as_slice::<Df32>()?;
        for item in items {
            let rows = &state[item.rim_start * 8..(item.rim_start + item.ids.len()) * 8];
            let unresolved = rows.chunks(8).filter(|v| v[7].hi == 0.).count();
            if unresolved > 0 {
                return Err(MkError::Unsupported(format!(
                    "{unresolved} rim edges did not reach tolerance {}; reference completion required",
                    settings.rim_tolerance
                )));
            }
        }
        let out = &result.outputs[kept_output..];
        Ok(Batch {
            starts: out[0].clone(),
            hits: std::array::from_fn(|i| out[i + 1].clone()),
            samples: out[6].clone(),
            mask: Some(survived),
            power: None,
            count,
        })
    }
    /// Append `Op::Keep` of `channels` into fresh outputs with `source`'s
    /// column count (`previous` supplies the unchanged columns).
    fn keep(
        p: &mut ResidentPlan,
        channels: Vec<(PlanView, bool)>,
        previous: Option<Vec<BufferId>>,
        source: PlanView,
    ) -> Vec<BufferId> {
        let n = source.desc.shape[0];
        let outputs: Vec<_> = channels
            .iter()
            .map(|(v, rows)| {
                let mut shape = v.desc.shape.clone();
                let axis = if *rows { 0 } else { shape.len() - 1 };
                shape[axis] = n;
                p.scratch(BufferDesc::new(v.desc.element, shape))
            })
            .collect();
        let previous = previous.map(|ids| {
            ids.iter()
                .map(|&id| p.view(id).expect("kept scratch"))
                .collect()
        });
        p.operations
            .push(ResidentOp::Wave(w::Request::Lattice(Op::Keep {
                channels,
                previous,
                source,
                outputs: outputs.clone(),
            })));
        outputs
    }
    /// Reference sphere, projection, lattice nodes and automatic-grid resampling
    /// of one traced pupil.
    fn place(
        &self,
        item: &Item,
        lat: &Batch,
        rim: Option<&Batch>,
        settings: &PupilSettings,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<ResidentTracedPupil>> {
        let sys = self.tracer.system();
        let wl = item.wl;
        let side = settings.rays + 2;
        let n = side * side;
        let mut stats = PlanStats::default();
        let chief_df = item.chief;
        let v3 = |a: [f64; 3]| Vec3::new(a[0], a[1], a[2]);
        let cp = v3(std::array::from_fn(|j| chief_df[j].to_f64()));
        let dc = v3(std::array::from_fn(|j| chief_df[j + 3].to_f64()));
        let sign = dc.z.signum();
        let mut centre = cp;
        if settings.reference == ReferencePoint::Centroid {
            let mut p = self.plan();
            let hits = p.import(lat.hits[0].view());
            let power = lat.power.as_ref().map(|h| p.import(h.view()));
            let points = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3, n]));
            p.operations.push(ResidentOp::Wave(w::Request::Placement(
                Placement::Centroid {
                    hits: p.view(hits)?,
                    power: power.map(|i| p.view(i).unwrap()),
                    chief: [chief_df[0], chief_df[1]],
                    count: n,
                    output: points,
                },
            )));
            let mask = p.import(subset(lat.mask(), [n]));
            let mut v = p.view(points)?;
            v.desc.shape = vec![3, n];
            v.stride = 1;
            let summary = super::sampling::reduce(
                self.backend,
                &mut p,
                v,
                Some(p_view(mask, &BufferDesc::new(ElementType::U32, [n]))),
                None,
                col::SegmentKind::Moments2,
            )?;
            p.outputs = vec![summary];
            let r = self
                .backend
                .submit_plan(p, &ctx.subrange(0.60, 0.62))?
                .wait()?;
            accumulate(&mut stats, &r.stats);
            let a = self.backend.read_view(r.outputs[0].view())?;
            let a = a.as_slice::<Df32>()?;
            centre.x += a[2].to_f64();
            centre.y += a[3].to_f64();
            centre.z = sys.surfaces[sys.image_index()]
                .shape
                .sag(centre.x * centre.x + centre.y * centre.y)
                .unwrap_or(cp.z);
        }
        let fo = self.tracer.paraxial().at(wl);
        let se = (fo.exit_pupil_position - cp.z) / dc.z;
        let tele = !se.is_finite();
        if se == 0. {
            return Err(MkError::Trace("exit pupil lies on the image".into()));
        }
        let sheet = if tele { -1. } else { se.signum() };
        let e = cp + dc * se;
        let radius = if tele {
            f64::INFINITY
        } else {
            (e - centre).norm()
        };
        let flip = |v: Vec3| Vec3::new(v.x, v.y, sign * v.z);
        let axis = if tele {
            flip(dc)
        } else {
            (flip(centre) - flip(e)) * (-sheet / radius)
        };
        let geometry = if tele {
            PupilGeometry::telecentric(0.5, sys.n_before(sys.image_index(), wl))
        } else {
            PupilGeometry::new(radius * 0.5, radius, sys.n_before(sys.image_index(), wl))
        }?
        .with_chief_ray([axis.x, axis.y, axis.z])?;
        let chief_buffer = fill::<Df32>(self.backend, [7], |a| {
            a.copy_from_slice(&chief_df);
            Ok(())
        })?;
        let centre_buffer = fill::<Df32>(self.backend, [3], |a| {
            a.copy_from_slice(&[centre.x, centre.y, centre.z].map(Df32::from_f64));
            Ok(())
        })?;
        let mut p = self.plan();
        let cb = p.import(chief_buffer.view());
        let ce = p.import(centre_buffer.view());
        let reference = aim::PupilReference {
            chief: p.view(cb)?,
            centre: p.view(ce)?,
            radius: Df32::from_f64(if tele { 1. } else { radius }),
            image_index: Df32::from_f64(geometry.image_index),
            wavelength_um: Df32::from_f64(sys.wavelengths_um[wl]),
            sphere_sheet: sheet as i32,
            telecentric: tele,
        };
        let (obj1, obj2) = match item.field.source {
            mklight_trace::ObjectSource::Direction { dir } => {
                let a = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                let e = (a - dir * a.dot(dir)).normalized();
                (e, dir.cross(e))
            }
            _ => (Vec3::X, Vec3::Y),
        };
        let frame = geometry
            .chief_frame()
            .map(|v| [v.x, v.y, v.z].map(Df32::from_f64));
        let object_axes = [obj1, obj2].map(|v| [v.x, v.y, v.z].map(Df32::from_f64));
        let mut raw_ids = vec![];
        for b in std::iter::once(lat).chain(rim) {
            let hits = b.import(&mut p);
            let projection = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4, b.count]));
            let status = p.scratch(BufferDesc::new(ElementType::U32, [b.count]));
            p.operations
                .push(ResidentOp::Aiming(aim::Request::ProjectPupil(
                    aim::ProjectPupil {
                        hits: hits.clone(),
                        reference: reference.clone(),
                        output: projection,
                        status,
                    },
                )));
            let starts = p.import(b.starts.view());
            let mask = p.import(subset(b.mask(), [b.count]));
            let power = b.power.as_ref().map(|h| p.import(h.view()));
            let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b.count, 8]));
            let radius = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1]));
            p.operations.push(ResidentOp::Wave(w::Request::Placement(
                Placement::Project {
                    projection: p.view(projection)?,
                    status: p.view(mask)?,
                    starts: p.view(starts)?,
                    hits: p.view(hits.rays.floats)?,
                    power: power.map(|i| p.view(i).unwrap()),
                    frame,
                    object_axes,
                    infinite: sys.object_at_infinity,
                    propagation: sign as f32,
                    sheet_factor: if tele { 1. } else { -sheet as f32 },
                    output,
                    radius,
                },
            )));
            raw_ids.push((output, radius, status));
        }
        p.outputs = raw_ids.iter().flat_map(|(a, b, c)| [*a, *b, *c]).collect();
        let r = self
            .backend
            .submit_plan(p, &ctx.subrange(0.62, 0.72))?
            .wait()?;
        accumulate(&mut stats, &r.stats);
        let mut sin = 0f64;
        for i in (1..r.outputs.len()).step_by(3) {
            let v = self.backend.read_view(r.outputs[i].view())?;
            sin = sin.max(v.as_slice::<Df32>()?[0].to_f64());
        }
        for (i, batch) in std::iter::once(lat).chain(rim).enumerate() {
            let failures = self.backend.read_view(r.outputs[i * 3 + 2].view())?;
            let mask = self.backend.read_view(batch.mask().view())?;
            let failed = failures
                .as_slice::<u32>()?
                .iter()
                .zip(mask.as_slice::<u32>()?)
                .filter(|(s, m)| **m != 0 && **s != 0)
                .count();
            if failed > 0 {
                return Err(MkError::Unsupported(format!(
                    "{failed} pupil rays miss the reference sphere; reference completion required"
                )));
            }
        }
        let sin = sin * (1. + 1e-9);
        let geometry = if tele {
            PupilGeometry::telecentric(sin, geometry.image_index)
        } else {
            PupilGeometry::new(sin * radius, radius, geometry.image_index)
        }?
        .with_chief_ray([axis.x, axis.y, axis.z])?;
        let mut uv_loops = vec![];
        let mut ep_loops = vec![];
        if let Some(b) = rim {
            let raw = self.backend.read_view(r.outputs[3].view())?;
            let raw = raw.as_slice::<Df32>()?;
            let probes = self.backend.read_view(b.samples.view())?;
            let probes: &[aim::Sample] = bytemuck::cast_slice(probes.words());
            for l in &item.loops {
                uv_loops.push(
                    l.iter()
                        .map(|&k| [raw[k * 8].to_f64() / sin, raw[k * 8 + 1].to_f64() / sin])
                        .collect(),
                );
                ep_loops.push(
                    l.iter()
                        .map(|&k| probes[item.rim_start + k].pupil.map(Df32::to_f64))
                        .collect(),
                );
            }
        }
        let mask = (!uv_loops.is_empty()).then(|| PupilMask::Polygons {
            loops: uv_loops.clone(),
        });
        let entrance_rim = (!ep_loops.is_empty()).then(|| PupilMask::Polygons {
            loops: ep_loops.clone(),
        });
        let mut p = self.plan();
        let placed = p.import(subset(&r.outputs[0], [n, 8]));
        let valid = p.import(subset(lat.mask(), [side, side]));
        let nodes = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [side, side, 4]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Placement(Placement::Nodes {
                placed: p.view(placed)?,
                valid: p.view(valid)?,
                sin_max: Df32::from_f64(sin),
                side,
                model: match settings.amplitude {
                    AmplitudeModel::Uniform => 0,
                    AmplitudeModel::Transmission => 1,
                    AmplitudeModel::Radiometric => 2,
                },
                infinite: sys.object_at_infinity,
                output: nodes,
            })));
        let maximum = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Placement(Placement::Slopes {
                nodes: p.view(nodes)?,
                valid: p.view(valid)?,
                output: maximum,
            })));
        p.outputs = vec![nodes, maximum];
        let r = self
            .backend
            .submit_plan(p, &ctx.subrange(0.72, 0.8))?
            .wait()?;
        accumulate(&mut stats, &r.stats);
        let gradient = self
            .backend
            .read_view(r.outputs[1].view())?
            .as_slice::<Df32>()?[0]
            .to_f64();
        let grid = if settings.grid == 0 {
            ((2. * gradient / 0.35).clamp(64., 1024.).ceil() as usize).next_power_of_two()
        } else {
            settings.grid
        };
        let nodes = r.outputs[0].clone();
        let valid = lat.mask().clone();
        let key = lattice_key(
            self.tracer,
            item.field,
            wl,
            settings,
            &self.radiometry.identity,
        );
        let identity = key.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let mut result = from_resident_lattice(
            self.backend,
            sys.snapshot.clone(),
            nodes.view(),
            subset(&valid, [side, side]),
            identity,
            grid,
            sys.wavelengths_um[wl],
            geometry,
            mask.as_ref(),
            settings.supersample,
            &ctx.subrange(0.8, 1.),
        )?;
        accumulate(&mut stats, &result.stats);
        let clear_area = if uv_loops.is_empty() {
            result.value.clear_area
        } else {
            crate::traced::even_odd_area(&uv_loops)
        };
        // The validated integer lattice has no off-disk point within the
        // 1e-12 inclusion tolerance. Count its rows without a host grid scan.
        let lattice_radius = settings.rays - 1;
        let lattice_rays = (0..settings.rays)
            .map(|j| {
                let y = (2 * j).abs_diff(lattice_radius);
                let x = (lattice_radius * lattice_radius - y * y).isqrt();
                if lattice_radius.is_multiple_of(2) {
                    2 * (x / 2) + 1
                } else {
                    2 * x.div_ceil(2)
                }
            })
            .sum::<usize>();
        let fraction = if ep_loops.is_empty() {
            item.survived as f64 / lattice_rays as f64
        } else {
            crate::traced::even_odd_area(&ep_loops) / std::f64::consts::PI
        };
        let arr = |v: Vec3| [v.x, v.y, v.z];
        let info = ExitPupilInfo {
            wavelength_um: sys.wavelengths_um[wl],
            paraxial_position_mm: fo.exit_pupil_position,
            paraxial_radius_mm: fo.exit_pupil_diameter * 0.5,
            telecentric: tele,
            chief_point_mm: arr(cp),
            chief_direction: arr(dc),
            exit_pupil_point_mm: (!tele).then(|| arr(e)),
            reference: settings.reference,
            reference_point_mm: arr(centre),
            reference_radius_mm: radius,
            propagation_sign: sign,
            image_index: geometry.image_index,
            sin_max: sin,
            numerical_aperture: sin * geometry.image_index,
            clear_area,
            transmitted_fraction: fraction,
            effective_f_number: 0.5
                / (geometry.image_index * sin * (clear_area / std::f64::consts::PI).sqrt()),
            lattice_rays,
            surviving_rays: item.survived,
            rim_points: item.ids.len(),
        };
        result.run_info.settings = serde_json::json!({"p6":"traced_pupil","field":item.field.point,"wavelength":wl,"pupil":settings,"grid":grid,"resident_stats":stats});
        Ok(WaveResult {
            value: ResidentTracedPupil {
                pupil: result.value,
                info,
                nodes: Some(nodes),
                valid: Some(valid),
                mask,
                entrance_rim,
                key,
                host: None,
            },
            run_info: result.run_info,
            stats,
        })
    }
}
