//! Explicit resident ghost jobs. Engine capability flags remain with P11.
//! Immutable path programs use the solver's checked expansion and exact clips.
use crate::{Action, Clip, Direction, GhostModel, GhostPath, InterfaceKind};
use bytemuck::Pod;
use mklight_core::{JobContext, MkError, MkResult, RunInfo, SurfaceShape};
use mklight_gpu::{
    ComputeBackend, Df32, NewtonSettings, PackOptions, PackedSystem, SystemHandle,
    aiming::*,
    consts::*,
    ghost_ops::{self as op, LaunchSample, Probe, Step, Value},
    radiometry::{
        self, CoatingDomain, CoatingQuery, CoatingRequest, CoatingValue, PreparedCoatings,
    },
    resident::*,
};

/// Step rows per lattice plan. This bounds the step, factor and coating
/// storage of one plan to about 160 MB, which the scratch pool can retain.
pub(crate) const ROW_TILE: usize = 1 << 20;
pub struct GhostTransport<'a> {
    pub backend: &'a dyn ComputeBackend,
    model: &'a GhostModel,
    systems: SystemHandle,
    offsets: BufferHandle,
    steps: BufferHandle,
    coatings: PreparedCoatings,
    identity: String,
    /// Steps per path; each probe owns that many compact step rows.
    lengths: Vec<u32>,
    max_steps: usize,
    wavelengths: usize,
    chiefs: Option<(BufferHandle, BufferHandle)>,
    /// Every submission, including device time, command count and memory.
    pub stats: Vec<PlanStats>,
    map_stats: std::cell::RefCell<Vec<PlanStats>>,
    pub(crate) resident_probes: Vec<BufferHandle>,
    pub distinct_probes: usize,
    pub(crate) lattice_keys: usize,
    pub fallback_probes: usize,
    pub(crate) reference_seconds: f64,
    pub(crate) boundary_probes: usize,
    pub(crate) pixel_migrations: usize,
    pub run: Option<RunInfo>,
}
impl Drop for GhostTransport<'_> {
    fn drop(&mut self) {
        let _ = self.backend.release_systems(self.systems);
    }
}
pub struct ProbeValues(ReadView);
impl std::ops::Deref for ProbeValues {
    type Target = [Value];
    fn deref(&self) -> &[Value] {
        bytemuck::cast_slice(self.0.words())
    }
}
pub struct ProbeResult {
    pub values: ProbeValues,
    pub run: RunInfo,
    pub stats: PlanStats,
}
fn fill<T: Pod>(
    b: &dyn ComputeBackend,
    n: usize,
    mut f: impl FnMut(usize) -> MkResult<T>,
) -> MkResult<BufferHandle> {
    b.fill_buffer(op::record_desc::<T>(n), &mut |mut out| {
        for (i, v) in bytemuck::cast_slice_mut::<_, T>(out.as_mut_slice::<u32>()?)
            .iter_mut()
            .enumerate()
        {
            *v = f(i)?;
        }
        Ok(())
    })
}
impl<'a> GhostTransport<'a> {
    pub fn new(
        backend: &'a dyn ComputeBackend,
        model: &'a GhostModel,
        paths: &[GhostPath],
        ctx: &JobContext,
    ) -> MkResult<Self> {
        ctx.check()?;
        if model.settings().intersect != mklight_trace::IntersectSettings::default() {
            return Err(MkError::Unsupported(
                "ghost custom intersection settings require reference qualification".into(),
            ));
        }
        let sys = model.system();
        let nw = sys.wavelengths_um.len();
        if nw > mklight_gpu::layout::MAX_WAVELENGTHS {
            return Err(MkError::Unsupported(
                "ghost wavelength chunks above 16".into(),
            ));
        }
        let programs = paths
            .iter()
            .map(|p| {
                ctx.check()?;
                model.check_path(p)
            })
            .collect::<MkResult<Vec<_>>>()?;
        let max_steps = programs.iter().map(Vec::len).max().unwrap_or(0);
        if max_steps == 0 {
            return Err(MkError::InvalidParam("ghost paths are empty".into()));
        }
        let mut packed = PackedSystem::pack(sys, &PackOptions::default())?;
        for (i, s) in packed.surfaces.iter_mut().enumerate() {
            match model.clip(i) {
                Clip::None => s.aperture = MKL_APERTURE_NONE,
                Clip::Surface => {}
                Clip::Radius { radius } => {
                    let r = Df32::from_f64(radius);
                    s.aperture = MKL_APERTURE_CIRCULAR;
                    s.aperture_params = [r.hi, 0., 0., 0.];
                    s.aperture_params_lo = [r.lo, 0., 0., 0.];
                }
            }
        }
        let plane = programs.iter().map(Vec::len).sum::<usize>();
        let ns = sys.num_surfaces();
        let (coatings, coating_ids) =
            prepare_coatings(backend, model, &(0..nw).collect::<Vec<_>>(), ctx)?;
        let steps = backend.fill_buffer(op::record_desc::<Step>(nw * plane), &mut |mut out| {
            let rows = bytemuck::cast_slice_mut::<_, Step>(out.as_mut_slice::<u32>()?);
            for (w, plane_rows) in rows.chunks_exact_mut(plane).enumerate() {
                let mut at = 0;
                for path in &programs {
                    ctx.check()?;
                    for st in path {
                        let coating = coating_ids[w * ns + st.surface as usize]
                            [(st.direction == Direction::Backward) as usize];
                        plane_rows[at] = pack_step(model, st, w, coating);
                        at += 1;
                    }
                }
            }
            Ok(())
        })?;
        let offsets = backend.fill_buffer(
            BufferDesc::new(ElementType::U32, [programs.len() + 1]),
            &mut |mut out| {
                let rows = out.as_mut_slice::<u32>()?;
                rows[0] = 0;
                for (i, path) in programs.iter().enumerate() {
                    rows[i + 1] = rows[i] + path.len() as u32;
                }
                Ok(())
            },
        )?;
        let identity = material_identity(backend, &coatings, &steps, ctx)?;
        let key = SystemCacheKey {
            snapshot: sys.snapshot.hash().to_hex(),
            library: coatings.identity.clone(),
            configuration: String::new(),
            spectrum: serde_json::to_string(&sys.wavelengths_um).unwrap(),
            layout: MKL_LAYOUT_VERSION,
            kernel_build: backend.precision().kernel_build.clone(),
            operation_settings: serde_json::to_string(model.clips()).unwrap(),
        };
        let systems = backend.upload_systems_cached(&key, &[packed])?;
        Ok(Self {
            backend,
            model,
            systems,
            offsets,
            steps,
            coatings,
            identity,
            lengths: programs.iter().map(|p| p.len() as u32).collect(),
            max_steps,
            wavelengths: nw,
            chiefs: None,
            stats: vec![],
            map_stats: Default::default(),
            resident_probes: vec![],
            distinct_probes: 0,
            lattice_keys: 0,
            fallback_probes: 0,
            reference_seconds: 0.,
            boundary_probes: 0,
            pixel_migrations: 0,
            run: None,
        })
    }
    pub(crate) fn path_rows(&self, path: usize) -> usize {
        self.lengths[path] as usize
    }
    fn plan(&self) -> ResidentPlan {
        let mut p = ResidentPlan::new(self.model.system().snapshot.clone());
        p.identity.library_hash = self.identity.clone();
        p.settings = serde_json::json!({"operation":"ghost","version":MKL_GHOST_VERSION,"max_steps":self.max_steps,"critical_discriminant_band":f32::from_bits(MKL_GHOST_CRITICAL_BAND_BITS),"max_segment_mm":MKL_GHOST_MAX_SEGMENT_MM,"model":self.model.settings()});
        p
    }
    /// Prefix of the probes' path lengths: their compact step rows. Step
    /// storage is rounded up to [`ROW_TILE`] (or a power of two below it) so
    /// the exact-size pool reuses it across tiles; spare rows are zeroed.
    fn step_rows(&self, n: usize, path: impl Fn(usize) -> u32) -> MkResult<(BufferHandle, usize)> {
        let mut total = 0;
        let rows = self.backend.fill_buffer(
            BufferDesc::new(ElementType::U32, [n + 1]),
            &mut |mut f| {
                let out = f.as_mut_slice::<u32>()?;
                out[0] = 0;
                for i in 0..n {
                    out[i + 1] = out[i] + self.lengths[path(i) as usize];
                }
                total = out[n] as usize;
                Ok(())
            },
        )?;
        let capacity = if total > ROW_TILE / 2 {
            total.next_multiple_of(ROW_TILE)
        } else {
            total.next_power_of_two()
        };
        Ok((rows, capacity))
    }
    fn append(
        &self,
        p: &mut ResidentPlan,
        probes: BufferId,
        (rows, capacity): (BufferHandle, usize),
        powers: bool,
    ) -> MkResult<BufferId> {
        let n = p.desc(probes)?.shape[0];
        let offsets = p.import(self.offsets.view());
        let steps = p.import(self.steps.view());
        let rows = p.import(rows.view());
        let values = p.scratch(op::record_desc::<Value>(n));
        let queries = p.scratch(op::record_desc::<CoatingQuery>(capacity));
        let factors = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [capacity, 2]));
        let sensor = self
            .model
            .system()
            .system()
            .sensor
            .as_ref()
            .map(|s| [s.half_width(), s.half_height()])
            .unwrap_or([f32::MAX as f64; 2])
            .map(Df32::from_f64);
        p.operations.push(ResidentOp::Ghost(op::Request::Trace {
            systems: SystemSet::Uploaded(self.systems),
            probes: p.view(probes)?,
            offsets: p.view(offsets)?,
            steps: p.view(steps)?,
            wavelengths: self.wavelengths,
            reject_virtual: self.model.settings().reject_virtual,
            sensor,
            newton: NewtonSettings::default(),
            step_rows: p.view(rows)?,
            output: values,
            queries,
            factors,
        }));
        if !powers {
            return Ok(values);
        }
        let specs = p.import(self.coatings.specs.view());
        let layers = p.import(self.coatings.layers.view());
        let coatings = p.scratch(op::record_desc::<CoatingValue>(capacity));
        p.operations
            .push(ResidentOp::Radiometry(radiometry::Request::Coating(
                CoatingRequest {
                    specs: p.view(specs)?,
                    layers: p.view(layers)?,
                    queries: p.view(queries)?,
                    output: coatings,
                    tile: radiometry::MAX_TILE,
                },
            )));
        let output = p.scratch(op::record_desc::<Value>(n));
        p.operations.push(ResidentOp::Ghost(op::Request::Weights {
            values: p.view(values)?,
            coatings: p.view(coatings)?,
            factors: p.view(factors)?,
            offsets: p.view(offsets)?,
            steps: p.view(steps)?,
            probes: p.view(probes)?,
            wavelengths: self.wavelengths,
            step_rows: p.view(rows)?,
            sensor_reflectance: Df32::from_f64(
                self.model.settings().sensor_reflectance.unwrap_or(0.),
            ),
            output,
        }));
        Ok(output)
    }
    /// Explicit start rays are useful for same-sample transport and combined
    /// conservation checks. Inputs are copied once from the caller's slice.
    pub fn trace(&self, probes: &[Probe], ctx: &JobContext) -> MkResult<ProbeResult> {
        for (i, q) in probes.iter().enumerate() {
            if i % 4096 == 0 {
                ctx.check()?;
            }
            if q.wavelength as usize >= self.wavelengths
                || q.path as usize + 1 >= self.offsets.desc().shape[0]
                || !q.position.iter().chain(&q.direction).all(|v| v.is_finite())
            {
                return Err(MkError::InvalidParam(
                    "ghost probe index or geometry".into(),
                ));
            }
        }
        let input = fill(self.backend, probes.len(), |i| Ok(probes[i]))?;
        let rows = self.step_rows(probes.len(), |i| probes[i].path)?;
        let mut p = self.plan();
        p.settings["input_copy_bytes"] = serde_json::json!(std::mem::size_of_val(probes));
        let input = p.import(input.view());
        let output = self.append(&mut p, input, rows, true)?;
        p.outputs = vec![output];
        let r = self.backend.submit_plan(p, ctx)?.wait()?;
        let v = self.backend.read_view(r.outputs[0].view())?;
        Ok(ProbeResult {
            values: ProbeValues(v),
            run: r.run_info,
            stats: r.stats,
        })
    }
    pub(crate) fn prepare_fields(
        &mut self,
        points: &[mklight_trace::FieldPoint],
        ctx: &JobContext,
    ) -> MkResult<()> {
        self.prepare_pupils(points, [0.; 2], ctx)
    }
    fn prepare_pupils(
        &mut self,
        points: &[mklight_trace::FieldPoint],
        pupil: [f64; 2],
        ctx: &JobContext,
    ) -> MkResult<()> {
        let nw = self.wavelengths;
        let fields = fill(self.backend, points.len() * nw, |i| {
            let mut f = FieldSpec::new(
                self.model.system(),
                self.model.tracer().paraxial(),
                points[i / nw],
                0,
                (i % nw) as u32,
                self.model.tracer().aiming(),
            )?;
            f.vignette = [Df32::ZERO; 4];
            Ok(f)
        })?;
        let samples = fill(self.backend, points.len() * nw, |i| {
            Ok(Sample::new(pupil, i as u32, i as u64))
        })?;
        // Chief rows from the reference tracer (packed wavelength slot i % nw
        // is system wavelength i % nw): no device chief/image-height solve.
        let rows: Vec<_> = (0..points.len() * nw)
            .map(|i| (points[i / nw], i % nw))
            .collect();
        let (seeds, host) = mklight_gpu::aiming::host_seeds(self.model.tracer(), &rows);
        let seeds = fill(self.backend, seeds.len(), |i| Ok(seeds[i]))?;
        let mut p = self.plan();
        p.host = host;
        let metadata = p.import(fields.view());
        let samples = p.import(samples.view());
        let seeds = p.import(seeds.view());
        let n = points.len() * nw;
        let rays = ResidentRays {
            floats: p.scratch(BufferDesc::new(ElementType::F32, [15, n])),
            indices: p.scratch(BufferDesc::new(ElementType::U32, [2, n])),
            logical_ids: p.scratch(BufferDesc::new(ElementType::U32, [2, n])),
            report: p.scratch(BufferDesc::new(ElementType::U32, [4, n])),
            count: n,
        };
        let tolerance = match self.model.tracer().aiming() {
            mklight_core::RayAiming::Real { tolerance } => tolerance as f32,
            _ => 1e-10,
        };
        p.operations.push(ResidentOp::Aiming(
            mklight_gpu::aiming::Request::GenerateAim(GenerateAim {
                systems: SystemSet::Uploaded(self.systems),
                samples: p.view(samples)?,
                sampling: Sampling::Explicit,
                unresolved: None,
                field_wavelength: p.view(metadata)?,
                field_key: None,
                seeds: Some(p.view(seeds)?),
                tolerance,
                max_iterations: 40,
                output: rays.clone(),
            }),
        ));
        p.outputs = vec![rays.floats, rays.report];
        let r = self.backend.submit_plan(p, ctx)?.wait()?;
        let report = self.backend.read_view(r.outputs[1].view())?;
        let words = report.as_slice::<u32>()?;
        if words[..n].iter().any(|&v| v != 0) {
            return Err(MkError::Unsupported(
                "ghost chief aiming requires reference completion".into(),
            ));
        }
        self.stats.push(r.stats);
        self.run = Some(r.run_info);
        self.chiefs = Some((r.outputs[0].clone(), fields));
        Ok(())
    }
    pub(crate) fn lattice(
        &mut self,
        n: usize,
        sample: impl FnMut(usize) -> MkResult<LaunchSample>,
        powers: bool,
        ctx: &JobContext,
    ) -> MkResult<ProbeValues> {
        let (chiefs, fields) = self.chiefs.as_ref().unwrap();
        let input = fill(self.backend, n, sample)?;
        let rows = {
            let samples = self.backend.read_view(input.view())?;
            let samples = bytemuck::cast_slice::<_, LaunchSample>(samples.words());
            self.step_rows(n, |i| samples[i].path)?
        };
        let mut p = self.plan();
        let chiefs = p.import(chiefs.view());
        let fields = p.import(fields.view());
        let samples_id = p.import(input.view());
        let probes = p.scratch(op::record_desc::<Probe>(n));
        p.operations.push(ResidentOp::Ghost(op::Request::Launch {
            chiefs: p.view(chiefs)?,
            fields: p.view(fields)?,
            samples: p.view(samples_id)?,
            output: probes,
        }));
        let output = self.append(&mut p, probes, rows, powers)?;
        p.outputs = vec![output];
        let r = self.backend.submit_plan(p, ctx)?.wait()?;
        let v = self.backend.read_view(r.outputs[0].view())?;
        self.resident_probes.push(r.outputs[0].clone());
        self.distinct_probes += n;
        self.stats.push(r.stats);
        self.run = Some(r.run_info);
        Ok(ProbeValues(v))
    }
    pub(crate) fn stamp(&self, run: &mut RunInfo) {
        if let Some(r) = &self.run {
            run.backend = r.backend.clone();
            run.precision = r.precision;
            run.settings["resident_backend"] = r.settings.clone();
        }
        run.settings["gpu"] = serde_json::json!({"system_cache":self.backend.resident_cache_stats(),"resident_bytes":self.resident_probes.iter().map(|h|h.desc().byte_len().unwrap()).sum::<usize>(),"version":MKL_GHOST_VERSION,"distinct_probes":self.distinct_probes,"lattice_keys":self.lattice_keys,"reference_probes":self.fallback_probes,"reference_seconds":self.reference_seconds,"boundary_probes":self.boundary_probes,"pixel_migrations":self.pixel_migrations,"plans":self.stats,"map_plans":*self.map_stats.borrow()});
        if self.fallback_probes > 0 {
            run.warnings.push(format!(
                "P9: {} probes completed by CPU f64 reference",
                self.fallback_probes
            ));
        }
    }
}

pub(crate) fn ray_out(v: Value) -> crate::irradiance::RayOut {
    let w = 0.5 * (v.weight_s.to_f64() + v.weight_p.to_f64()) * v.internal.to_f64();
    match v.status & MKL_RAY_CODE_MASK {
        0 => Ok(Some((v.position[0].to_f64(), v.position[1].to_f64(), w))),
        MKL_GHOST_VIRTUAL => Err(w),
        _ => Ok(None),
    }
}
/// Explicit GPU job; unsupported setup is returned to the engine adapter.
/// Per-probe P7 reference masks are completed and counted by the scheduler.
pub fn ghost_irradiance(
    backend: &dyn ComputeBackend,
    model: &GhostModel,
    settings: &crate::IrradianceSettings,
    ctx: &JobContext,
) -> MkResult<crate::GhostIrradiance> {
    crate::irradiance::ghost_irradiance_with_backend(model, settings, ctx, Some(backend))
}

pub(crate) struct MapData {
    pub values: Vec<f64>,
    pub moments: Vec<[f64; 10]>,
    pub summaries: Vec<crate::irradiance::LeafSummary>,
}
impl GhostTransport<'_> {
    /// P3 stable sort and fixed-tree pixel segments preserve leaf IDs. All
    /// bins, offsets and sums are produced on device without floating atomics.
    pub(crate) fn map(
        &self,
        mut leaves: Vec<op::Leaf>,
        template: &crate::IrradianceMap,
        summaries: Vec<crate::irradiance::LeafSummary>,
        ctx: &JobContext,
    ) -> MkResult<MapData> {
        use mklight_gpu::collectives::*;
        let maps = summaries.len();
        let nb = template.values.len() * maps;
        let n = leaves.len();
        let mut p = self.plan();
        for (i, l) in leaves.iter_mut().enumerate() {
            l.destination = i as u32;
        }
        let mut field_counts = vec![0u32; maps];
        for l in &leaves {
            field_counts[l.map as usize] += 1;
        }
        let max_field = field_counts.iter().copied().max().unwrap_or(0) as usize;
        let field_offsets = self.backend.fill_buffer(
            BufferDesc::new(ElementType::U32, [maps + 1]),
            &mut |mut f| {
                let out = f.as_mut_slice::<u32>()?;
                out[0] = 0;
                for (i, &count) in field_counts.iter().enumerate() {
                    out[i + 1] = out[i] + count;
                }
                Ok(())
            },
        )?;

        leaves.sort_by_key(|l| l.batch);
        let mut batches = leaves.iter().map(|l| l.batch as usize).collect::<Vec<_>>();
        batches.dedup();
        let sources = batches
            .iter()
            .map(|&b| {
                let id = p.import(self.resident_probes[b].view());
                p.view(id)
            })
            .collect::<MkResult<Vec<_>>>()?;
        let records = fill(self.backend, n, |i| {
            if i % 4096 == 0 {
                ctx.check()?;
            }
            let mut l = leaves[i];
            l.batch = batches.binary_search(&(l.batch as usize)).unwrap() as u32;
            Ok(l)
        })?;
        let records = p.import(records.view());
        let data = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n, 5]));
        p.operations.push(ResidentOp::Ghost(op::Request::Gather {
            sources,
            leaves: p.view(records)?,
            output: data,
        }));

        let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3, n]));
        p.operations
            .push(ResidentOp::Ghost(op::Request::MomentTerms {
                hits: p.view(data)?,
                output: terms,
            }));
        let field_offsets = p.import(field_offsets.view());
        let moments = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [maps, 10]));
        let moment_status = p.scratch(BufferDesc::new(ElementType::U32, [maps]));
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::SegmentReduce(SegmentReduce {
                values: p.view(terms)?,
                offsets: p.view(field_offsets)?,
                mask: None,
                mask_value: 0,
                weights: None,
                max_segment_len: max_field,
                reference: None,
                mode: mklight_gpu::ReduceMode::DoubleFloat,
                kind: SegmentKind::Moments2,
                output: moments,
                status: moment_status,
            }),
        ));
        let keys = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let ids = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let weights = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
        p.operations.push(ResidentOp::Ghost(op::Request::PixelKeys {
            hits: p.view(data)?,
            bins: template.bins.map(|v| v as u32),
            maps,
            half: template.half_size.map(Df32::from_f64),
            keys,
            ids,
            weights,
        }));
        let sk = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let si = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let sw = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::StableSort(StableSort {
                keys: p.view(keys)?,
                ids: p.view(ids)?,
                values: Some(p.view(weights)?),
                output_keys: sk,
                output_ids: si,
                output_values: Some(sw),
            }),
        ));
        let flags = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations
            .push(ResidentOp::Ghost(op::Request::SegmentFlags {
                keys: p.view(sk)?,
                bins: nb,
                output: flags,
            }));
        let prefix = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::Scan(Scan {
                values: p.view(flags)?,
                inclusive: true,
                output: prefix,
            }),
        ));
        let offsets = p.scratch(BufferDesc::new(ElementType::U32, [n + 1]));
        let pixels = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let count = p.scratch(BufferDesc::new(ElementType::U32, [2]));
        p.operations
            .push(ResidentOp::Ghost(op::Request::PixelSegments {
                keys: p.view(sk)?,
                prefix: p.view(prefix)?,
                bins: nb,
                offsets,
                pixels,
                count,
            }));
        p.outputs = vec![sw, offsets, pixels, count, moments];
        let prepared = self
            .backend
            .submit_plan(p, &ctx.subrange(0., 0.7))?
            .wait()?;
        self.map_stats.borrow_mut().push(prepared.stats);
        let moment_view = self.backend.read_view(prepared.outputs[4].view())?;
        let moments = moment_view
            .as_slice::<Df32>()?
            .as_chunks::<10>()
            .0
            .iter()
            .map(|v| v.map(|v| v.to_f64()))
            .collect();
        let counts = self.backend.read_view(prepared.outputs[3].view())?;
        let counts = counts.as_slice::<u32>()?;
        let active = counts[0] as usize;
        let maxlen = counts[1] as usize;
        let mut p = self.plan();
        let mut weights = prepared.outputs[0].view();
        weights.desc.shape = vec![1, n];
        let weights = p.import(weights);
        let mut os = prepared.outputs[1].view();
        os.desc.shape = vec![active + 1];
        let offsets = p.import(os);
        let mut ps = prepared.outputs[2].view();
        ps.desc.shape = vec![active];
        let pixels = p.import(ps);
        let sums = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, active]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [active]));
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::SegmentReduce(SegmentReduce {
                values: p.view(weights)?,
                offsets: p.view(offsets)?,
                mask: None,
                mask_value: 0,
                weights: None,
                max_segment_len: maxlen,
                reference: None,
                mode: mklight_gpu::ReduceMode::DoubleFloat,
                kind: SegmentKind::Sum,
                output: sums,
                status,
            }),
        ));
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [nb]));
        p.operations
            .push(ResidentOp::Ghost(op::Request::MapScatter {
                values: p.view(sums)?,
                pixels: p.view(pixels)?,
                output,
            }));
        p.outputs = vec![output];
        let r = self
            .backend
            .submit_plan(p, &ctx.subrange(0.7, 1.))?
            .wait()?;
        self.map_stats.borrow_mut().push(r.stats);
        let guard = self.backend.read_view(r.outputs[0].view())?;
        Ok(MapData {
            values: guard
                .as_slice::<Df32>()?
                .iter()
                .map(|v| v.to_f64() / template.bin_area())
                .collect(),
            moments,
            summaries,
        })
    }
}

impl GhostTransport<'_> {
    pub(crate) fn reference_batch(
        &mut self,
        replacements: &[(usize, crate::irradiance::RayOut)],
        ctx: &JobContext,
    ) -> MkResult<usize> {
        use bytemuck::Zeroable;
        self.stats.last_mut().unwrap().polished_rays += replacements.len() as u64;
        let handle = fill(self.backend, replacements.len(), |i| {
            if i % 256 == 0 {
                ctx.check()?;
            }
            let mut v = Value::zeroed();
            v.internal = Df32::from_f32(1.);
            match replacements[i].1 {
                Ok(Some((x, y, w))) => {
                    v.position[0] = Df32::from_f64(x);
                    v.position[1] = Df32::from_f64(y);
                    v.weight_s = Df32::from_f64(w);
                    v.weight_p = v.weight_s;
                }
                Ok(None) => v.status = MKL_RAY_MISS,
                Err(w) => {
                    v.status = MKL_GHOST_VIRTUAL;
                    v.weight_s = Df32::from_f64(w);
                    v.weight_p = v.weight_s;
                }
            }
            Ok(v)
        })?;
        let batch = self.resident_probes.len();
        self.resident_probes.push(handle);
        Ok(batch)
    }
}
/// Batched real meridional ghost foci. P1 aims the requested pupil ray;
/// paths share those resident launches. Lost rays keep their status and NaN.
pub struct GhostFocus {
    pub run: RunInfo,
    /// Path-major, wavelength-minor; signed image-local z from the sensor.
    pub focus: Vec<f64>,
    pub status: Vec<u32>,
    pub stats: Vec<PlanStats>,
}
pub fn real_ghost_focus(
    backend: &dyn ComputeBackend,
    model: &GhostModel,
    paths: &[GhostPath],
    rho: f64,
    ctx: &JobContext,
) -> MkResult<GhostFocus> {
    let timer = mklight_core::job::RunTimer::start();
    if !rho.is_finite() {
        return Err(MkError::InvalidParam("ghost focus pupil height".into()));
    }
    let mut g = GhostTransport::new(backend, model, paths, &ctx.subrange(0., 0.02))?;
    g.prepare_pupils(
        &[mklight_trace::FieldPoint::normalized(0., 0.)],
        [0., rho],
        &ctx.subrange(0.02, 0.2),
    )?;
    let nw = g.wavelengths;
    let v = g.lattice(
        paths.len() * nw,
        |i| {
            Ok(LaunchSample {
                pupil: [Df32::ZERO; 2],
                lattice: [0; 2],
                field: (i % nw) as u32,
                path: (i / nw) as u32,
                wavelength: (i % nw) as u32,
                reserved: 0,
            })
        },
        false,
        &ctx.subrange(0.2, 0.99),
    )?;
    let focus = v
        .iter()
        .map(|v| {
            if v.status != 0 {
                f64::NAN
            } else {
                let d = v.direction[1].to_f64();
                if d == 0. {
                    f64::INFINITY
                } else {
                    -v.position[1].to_f64() * v.direction[2].to_f64() / d
                }
            }
        })
        .collect();
    let status = v.iter().map(|v| v.status).collect();
    let mut run = crate::paraxial::run_info(
        model,
        timer,
        "ghost.real_focus",
        &serde_json::json!({"paths":paths,"rho":rho}),
        Vec::new(),
    );
    g.stamp(&mut run);
    ctx.check()?;
    ctx.report(1., "ghost focus complete");
    Ok(GhostFocus {
        run,
        focus,
        status,
        stats: g.stats.clone(),
    })
}

pub struct ParaxialBatch {
    pub run: RunInfo,
    /// Path-major, wavelength-minor; marginal and chief, with all step records.
    pub rays: Vec<[crate::ParaxialTrace; 2]>,
    pub throughput: Vec<[f64; 2]>,
    pub stats: PlanStats,
}
/// Compile path/wavelength constitutive metadata once, then evaluate both
/// first-order rays in double-float on the backend. Enumeration stays host.
pub fn paraxial_batch(
    backend: &dyn ComputeBackend,
    model: &GhostModel,
    paths: &[GhostPath],
    wavelengths: &[usize],
    ctx: &JobContext,
) -> MkResult<ParaxialBatch> {
    use op::FirstOrderStep;
    let timer = mklight_core::job::RunTimer::start();
    let programs = paths
        .iter()
        .map(|p| {
            ctx.check()?;
            model.check_path(p)
        })
        .collect::<MkResult<Vec<_>>>()?;
    if wavelengths
        .iter()
        .any(|&w| w >= model.system().wavelengths_um.len())
    {
        return Err(MkError::InvalidParam("ghost paraxial wavelength".into()));
    }
    let (coatings, coating_ids) = prepare_coatings(backend, model, wavelengths, ctx)?;
    let mut offsets = vec![0u32];
    for path in &programs {
        for _ in wavelengths {
            offsets.push(offsets.last().unwrap() + path.len() as u32);
        }
    }
    let step_count = *offsets.last().unwrap() as usize;
    let rows = paths.len() * wavelengths.len();
    let mut indices = Vec::new();
    let steps = backend.fill_buffer(
        op::record_desc::<FirstOrderStep>(step_count),
        &mut |mut f| {
            let output = bytemuck::cast_slice_mut::<_, FirstOrderStep>(f.as_mut_slice::<u32>()?);
            let mut at = 0;
            indices.clear();
            for path in &programs {
                for &w in wavelengths {
                    ctx.check()?;
                    let mut n = model.system().n_object[w];
                    let mut ns = Vec::with_capacity(path.len());
                    for (k, st) in path.iter().enumerate() {
                        let (no, power) = crate::paraxial::step_power(model, st, n, w);
                        n = no;
                        ns.push(n);
                        let si = st.surface as usize;
                        let surface = &model.system().surfaces[si];
                        let back = st.direction == Direction::Backward;
                        let reflected = surface.is_mirror() || st.action == Action::Reflect;
                        let channel = if si == model.system().image_index() {
                            if st.action == Action::Reflect {
                                MKL_GHOST_AXIAL_SENSOR
                            } else {
                                MKL_GHOST_AXIAL_DETECT
                            }
                        } else if surface.is_passive() {
                            MKL_GHOST_AXIAL_NONE
                        } else if surface.is_mirror() {
                            MKL_GHOST_AXIAL_MIRROR
                        } else if st.action == Action::Reflect {
                            MKL_GHOST_AXIAL_REFLECT
                        } else {
                            MKL_GHOST_AXIAL_TRANSMIT
                        };
                        let efficiency = if matches!(
                            channel,
                            MKL_GHOST_AXIAL_NONE | MKL_GHOST_AXIAL_SENSOR | MKL_GHOST_AXIAL_DETECT
                        ) {
                            1.
                        } else {
                            mklight_core::materials::diffraction::order_efficiency(
                                model.system(),
                                si,
                                w,
                                st.order,
                                reflected,
                                back,
                            )
                        };
                        let absorption = if k == 0 {
                            0.
                        } else {
                            let gap = si.min(path[k - 1].surface as usize);
                            model.system().surfaces[gap].alpha_after[w]
                                * model.system().surfaces[gap].thickness.abs()
                        };
                        let distance = path.get(k + 1).map_or(0., |next| {
                            crate::paraxial::delta_z(model, si, next.surface as usize)
                        });
                        output[at] = FirstOrderStep {
                            power: Df32::from_f64(power),
                            index: Df32::from_f64(n),
                            distance: Df32::from_f64(distance),
                            efficiency: Df32::from_f64(efficiency),
                            absorption: Df32::from_f64(absorption),
                            coating: coating_ids[w * model.system().num_surfaces() + si]
                                [back as usize],
                            channel,
                        };
                        at += 1;
                    }
                    indices.push(ns);
                }
            }
            Ok(())
        },
    )?;
    let os = backend.fill_buffer(
        BufferDesc::new(ElementType::U32, [offsets.len()]),
        &mut |mut f| {
            let output = f.as_mut_slice::<u32>()?;
            output[0] = 0;
            let mut i = 0;
            for path in &programs {
                for _ in wavelengths {
                    output[i + 1] = output[i] + path.len() as u32;
                    i += 1;
                }
            }
            Ok(())
        },
    )?;
    let initial = backend.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [rows, 4]),
        &mut |mut f| {
            for (i, row) in f
                .as_mut_slice::<Df32>()?
                .as_chunks_mut::<4>()
                .0
                .iter_mut()
                .enumerate()
            {
                if i % 256 == 0 {
                    ctx.check()?;
                }
                let fo = model
                    .tracer()
                    .paraxial()
                    .at(wavelengths[i % wavelengths.len()]);
                for (dest, value) in row.iter_mut().zip([
                    fo.marginal.y[1],
                    fo.marginal.nu[0],
                    fo.chief.y[1],
                    fo.chief.nu[0],
                ]) {
                    *dest = Df32::from_f64(value);
                }
            }
            Ok(())
        },
    )?;
    let mut p = ResidentPlan::new(model.system().snapshot.clone());
    p.identity.library_hash = material_identity(backend, &coatings, &steps, ctx)?;
    p.settings = serde_json::json!({"operation":"ghost.paraxial_batch","paths":paths,"wavelengths":wavelengths,"version":MKL_GHOST_VERSION,"model":model.settings(),"minimum_reduced_slope":f32::from_bits(MKL_GHOST_FIRSTORDER_SLOPE_BITS)});
    let nq = coatings.specs.desc().shape[0];
    let queries = fill(backend, nq, |i| {
        Ok(CoatingQuery {
            coating: i as u32,
            reserved: 0,
            cosine: Df32::from_f32(1.),
        })
    })?;
    let specs = p.import(coatings.specs.view());
    let layers = p.import(coatings.layers.view());
    let queries = p.import(queries.view());
    let coating_values = p.scratch(op::record_desc::<CoatingValue>(nq));
    p.operations
        .push(ResidentOp::Radiometry(radiometry::Request::Coating(
            CoatingRequest {
                specs: p.view(specs)?,
                layers: p.view(layers)?,
                queries: p.view(queries)?,
                output: coating_values,
                tile: radiometry::MAX_TILE,
            },
        )));
    let steps = p.import(steps.view());
    let os = p.import(os.view());
    let initial = p.import(initial.view());
    let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [step_count, 4]));
    let throughput = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [rows, 2]));
    let status = p.scratch(BufferDesc::new(ElementType::U32, [rows]));
    p.operations
        .push(ResidentOp::Ghost(op::Request::FirstOrder {
            steps: p.view(steps)?,
            offsets: p.view(os)?,
            initial: p.view(initial)?,
            coatings: p.view(coating_values)?,
            sensor_reflectance: Df32::from_f64(model.settings().sensor_reflectance.unwrap_or(0.)),
            output,
            throughput,
            status,
        }));
    p.outputs = vec![output, throughput, status];
    let mut r = backend.submit_plan(p, &ctx.subrange(0., 0.9))?.wait()?;
    let data = backend.read_view(r.outputs[0].view())?;
    let data = data.as_slice::<Df32>()?;
    let rt_view = backend.read_view(r.outputs[1].view())?;
    let rt = rt_view.as_slice::<Df32>()?;
    let flag_view = backend.read_view(r.outputs[2].view())?;
    let flags = flag_view.as_slice::<u32>()?;
    let (mut rays, mut throughputs, mut reference_rows) = (Vec::new(), Vec::new(), Vec::new());
    let reference_start = std::time::Instant::now();
    for (i, ns) in indices.into_iter().enumerate() {
        ctx.check()?;
        let (a, z) = (offsets[i] as usize, offsets[i + 1] as usize);
        if flags[i] != 0 {
            let path = &programs[i / wavelengths.len()];
            let w = wavelengths[i % wavelengths.len()];
            let fo = model.tracer().paraxial().at(w);
            rays.push([
                crate::paraxial::trace_steps(model, path, w, fo.marginal.y[1], fo.marginal.nu[0]),
                crate::paraxial::trace_steps(model, path, w, fo.chief.y[1], fo.chief.nu[0]),
            ]);
            let (r, t) = crate::paraxial::axial_throughput(model, path, w);
            throughputs.push([r, t]);
            reference_rows.push(i);
        } else {
            rays.push(std::array::from_fn(|j| crate::ParaxialTrace {
                y: (a..z).map(|k| data[4 * k + 2 * j].to_f64()).collect(),
                nu: (a..z).map(|k| data[4 * k + 2 * j + 1].to_f64()).collect(),
                n: ns.clone(),
            }));
            throughputs.push([rt[2 * i].to_f64(), rt[2 * i + 1].to_f64()]);
        }
    }
    r.stats.polished_rays += 2 * reference_rows.len() as u64;
    r.run_info.settings["reference_rows"] = serde_json::json!(reference_rows);
    r.run_info.settings["reference_and_result_seconds"] =
        serde_json::json!(reference_start.elapsed().as_secs_f64());
    if !reference_rows.is_empty() {
        r.run_info.warnings.push(format!("P9 first order: {} path/wavelength rows completed by CPU f64 for coating, range or afocal conditioning",reference_rows.len()));
    }
    let run = timer.finish(
        &model.system().snapshot,
        "ghost.paraxial_batch",
        &r.run_info.settings,
        &r.run_info.backend,
        r.run_info.precision,
        None,
        r.run_info.warnings,
    );
    ctx.check()?;
    ctx.report(1., "ghost paraxial batch complete");
    Ok(ParaxialBatch {
        run,
        rays,
        throughput: throughputs,
        stats: r.stats,
    })
}
/// Native paraxial ranking with batched resident ray recurrences. Small
/// enumeration, axial material resolution and the final ranking stay host.
pub fn paraxial_ghosts(
    backend: &dyn ComputeBackend,
    model: &GhostModel,
    settings: &crate::ParaxialGhostSettings,
    ctx: &JobContext,
) -> MkResult<crate::ParaxialGhostReport> {
    crate::paraxial::paraxial_ghosts_with_backend(model, settings, ctx, Some(backend))
}

fn pack_step(model: &GhostModel, st: &crate::Step, w: usize, coating: u32) -> Step {
    let sys = model.system();
    let i = st.surface as usize;
    let s = &sys.surfaces[i];
    let back = st.direction == Direction::Backward;
    let reflect = st.action == Action::Reflect || s.is_mirror();
    let (n1, n2) = model.media(i, st.direction, w);
    let channel = if i == sys.image_index() {
        if reflect { 4 } else { 3 }
    } else if s.is_passive() {
        0
    } else if reflect {
        2
    } else {
        1
    };
    let scale = match &s.shape {
        SurfaceShape::Diffractive { phase, .. } => {
            st.order.unwrap_or(phase.trace_order) as f64 * sys.wavelengths_um[w] * 1e-3
                / std::f64::consts::TAU
        }
        _ => 0.,
    };
    let eta = if i != sys.image_index() && !s.is_passive() {
        mklight_core::materials::diffraction::order_efficiency(sys, i, w, st.order, reflect, back)
    } else {
        1.
    };
    let alpha = if !back && i == 1 && sys.object_at_infinity {
        0.
    } else {
        sys.surfaces[if back { i } else { i - 1 }].alpha_after[w]
    };
    let power = match s.shape {
        SurfaceShape::Paraxial { focal_length } => {
            if back && !reflect {
                n1 / (n2 * focal_length)
            } else {
                1. / focal_length
            }
        }
        _ => 0.,
    };
    let mode = if st.action == Action::Reflect {
        MKL_STEP_REFLECT
    } else {
        MKL_STEP_SEQUENTIAL
    };
    Step {
        word: st.surface | (mode << MKL_STEP_MODE_SHIFT) | if back { MKL_STEP_BACKWARD } else { 0 },
        physical: (model.interface_kinds()[i] != InterfaceKind::Passive) as u32,
        coating,
        channel,
        scale: Df32::from_f64(scale),
        efficiency: Df32::from_f64(eta),
        alpha: Df32::from_f64(alpha),
        power: Df32::from_f64(power),
    }
}

fn prepare_coatings(
    backend: &dyn ComputeBackend,
    model: &GhostModel,
    wavelengths: &[usize],
    ctx: &JobContext,
) -> MkResult<(PreparedCoatings, Vec<[u32; 2]>)> {
    let sys = model.system();
    let nw = sys.wavelengths_um.len();
    let mut domains = vec![CoatingDomain {
        coating: mklight_core::materials::coating::ResolvedCoating::Ideal {
            reflectance: 0.,
            transmittance: 1.,
        },
        wavelength: sys.wavelengths_um[0],
        n_in: 1.,
        n_out: 1.,
        mirror: false,
    }];
    let ns = sys.num_surfaces();
    let mut coating_ids = vec![[0u32; 2]; nw * ns];
    let mut selected = vec![false; nw];
    for &w in wavelengths {
        selected[w] = true;
    }
    for w in (0..nw).filter(|&w| selected[w]) {
        for i in 1..sys.image_index() {
            let surface = &sys.surfaces[i];
            if surface.is_passive() {
                continue;
            }
            for (side, direction) in [Direction::Forward, Direction::Backward]
                .into_iter()
                .enumerate()
            {
                ctx.check()?;
                let (n1, n2) = model.media(i, direction, w);
                coating_ids[w * ns + i][side] = domains.len() as u32;
                domains.push(CoatingDomain {
                    coating: surface.coating.clone(),
                    wavelength: sys.wavelengths_um[w],
                    n_in: n1,
                    n_out: if surface.is_mirror() { n1 } else { n2 },
                    mirror: surface.is_mirror(),
                });
            }
        }
    }
    let coatings = PreparedCoatings::new(backend, &domains, ctx)?;
    Ok((coatings, coating_ids))
}

fn material_identity(
    backend: &dyn ComputeBackend,
    coatings: &PreparedCoatings,
    steps: &BufferHandle,
    ctx: &JobContext,
) -> MkResult<String> {
    let mut hash = blake3::Hasher::new();
    hash.update(coatings.identity.as_bytes());
    let view = backend.read_view(steps.view())?;
    for block in bytemuck::cast_slice::<_, u8>(view.words()).chunks(65536) {
        ctx.check()?;
        hash.update(block);
    }
    Ok(hash.finalize().to_hex().to_string())
}
