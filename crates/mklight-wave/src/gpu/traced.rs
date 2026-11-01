//! P1 generation/trace/projection and P7 scalar transport, retaining lattice
//! channels for resampling and fits. Only small rim-loop topology returns to host.
use super::*;
use crate::{AmplitudeModel, ExitPupilInfo, PupilMask, PupilSettings, ReferencePoint};
use mklight_core::Vec3;
use mklight_gpu::{
    PackOptions, PackedSystem, SystemHandle, aiming as aim, collectives as col, radiometry as rad,
    wave_ops::{lattice::Lattice as Op, placement::Placement},
};
use mklight_trace::{PreparedField, SequentialTracer};

#[derive(Clone, Debug)]
pub struct ResidentTracedPupil {
    pub pupil: ResidentPupil,
    pub info: ExitPupilInfo,
    pub nodes: BufferHandle,
    pub valid: BufferHandle,
    pub mask: Option<PupilMask>,
    pub entrance_rim: Option<PupilMask>,
}
impl ResidentTracedPupil {
    pub fn resample(
        &mut self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        n: usize,
        supersample: usize,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<ResidentPupil>> {
        let mut r = from_resident_lattice(
            b,
            snapshot,
            self.nodes.view(),
            subset(
                &self.valid,
                [self.nodes.desc().shape[0], self.nodes.desc().shape[1]],
            ),
            self.pupil.input_hash.clone(),
            n,
            self.pupil.wavelength_um,
            self.pupil.geometry,
            self.mask.as_ref(),
            supersample,
            ctx,
        )?;
        r.value.image_offset_mm = self.pupil.image_offset_mm;
        self.pupil = r.value.clone();
        Ok(r)
    }
}
struct Batch {
    starts: BufferHandle,
    hits: [BufferHandle; 5],
    records: Option<[BufferHandle; 2]>,
    samples: BufferHandle,
    mask: BufferHandle,
    power: Option<BufferHandle>,
    count: usize,
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
            records: self.records.as_ref().map(|r| p.import(r[0].view())),
            record_low: self.records.as_ref().map(|r| p.import(r[1].view())),
        }
    }
}
pub struct PupilTracer<'a> {
    backend: &'a dyn ComputeBackend,
    tracer: &'a SequentialTracer,
    systems: SystemHandle,
    wavelengths: Vec<Vec<usize>>,
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
fn record<T: bytemuck::Pod>(b: &dyn ComputeBackend, v: T) -> MkResult<BufferHandle> {
    fill::<u32>(b, [1, size_of::<T>() / 4], |a| {
        a.copy_from_slice(bytemuck::cast_slice(&[v]));
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
}
fn subset(h: &BufferHandle, shape: impl Into<Vec<usize>>) -> BufferView {
    let mut v = h.view();
    v.desc.shape = shape.into();
    v
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
        let wavelengths: Vec<Vec<usize>> = if total <= mklight_gpu::layout::MAX_WAVELENGTHS {
            vec![(0..total).collect()]
        } else {
            (0..total)
                .step_by(mklight_gpu::layout::MAX_WAVELENGTHS - 1)
                .map(|start| {
                    let mut w: Vec<_> = (start
                        ..(start + mklight_gpu::layout::MAX_WAVELENGTHS - 1).min(total))
                        .collect();
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
            radiometry,
        })
    }
    fn plan(&self) -> ResidentPlan {
        let mut p = ResidentPlan::new(self.tracer.system().snapshot.clone());
        p.identity.library_hash = self.radiometry.identity.clone();
        p
    }
    fn probe(
        &self,
        mut p: ResidentPlan,
        samples: BufferId,
        count: usize,
        field: &BufferHandle,
        clip: bool,
        power: bool,
        ctx: &JobContext,
    ) -> MkResult<(Batch, PlanStats)> {
        let metadata = p.import(field.view());
        let steps = fill::<u32>(self.backend, [self.tracer.system().image_index()], |a| {
            for (i, v) in a.iter_mut().enumerate() {
                *v = (i + 1) as u32;
            }
            Ok(())
        })?;
        let steps = p.import(steps.view());
        let generated = aim::rays(&mut p, count);
        let tolerance = match self.tracer.aiming() {
            mklight_core::RayAiming::Real { tolerance } => tolerance as f32,
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
                    field_key: Some(
                        blake3::hash(bytemuck::cast_slice(
                            self.backend.read_view(field.view())?.words(),
                        ))
                        .to_hex()
                        .to_string(),
                    ),
                    tolerance,
                    max_iterations: 60,
                    output: generated.clone(),
                },
            )));
        let hits = aim::hits(
            &mut p,
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
        let mask = p.scratch(BufferDesc::new(ElementType::U32, [count]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Lattice(Op::Survival {
                samples: p.view(samples)?,
                status: p.view(hits.status)?,
                output: mask,
            })));
        let power_id = if power {
            let mut radiometric_hits = hits.clone();
            if self.wavelengths.len() > 1 {
                let guard = self.backend.read_view(field.view())?;
                let spec: &[aim::FieldSpec] = bytemuck::cast_slice(guard.words());
                let spec = spec[0];
                let wavelength = self.wavelengths[spec.system as usize][spec.wavelength as usize];
                let indices = p.scratch(BufferDesc::new(ElementType::U32, [2, count]));
                p.operations
                    .push(ResidentOp::Wave(w::Request::Lattice(Op::Wavelength {
                        count,
                        wavelength: wavelength as u32,
                        output: indices,
                    })));
                radiometric_hits.rays.indices = indices;
            }
            let id = self.radiometry.append(
                &mut p,
                &radiometric_hits,
                &mklight_trace::PolarizationState::Unpolarized,
            )?;
            if let Some(ResidentOp::Radiometry(rad::Request::Polarization(r))) =
                p.operations.last_mut()
            {
                r.scalar = true;
            }
            Some(id)
        } else {
            None
        };
        p.outputs = vec![
            generated.floats,
            hits.rays.floats,
            hits.rays.indices,
            hits.rays.logical_ids,
            hits.rays.report,
            hits.status,
            mask,
        ];
        // Imported sample leases already survive; generated samples must publish.
        let sample_handle = match &p.buffers[samples.0] {
            PlanBuffer::Import(v) => Some(v.buffer.clone()),
            _ => {
                p.outputs.push(samples);
                None
            }
        };
        if let Some(id) = power_id {
            p.outputs.push(id);
        }
        if let (Some(h), Some(l)) = (hits.records, hits.record_low) {
            p.outputs.extend([h, l]);
        }
        let r = self.backend.submit_plan(p, ctx)?.wait()?;
        let mut i = 7;
        let samples = sample_handle.unwrap_or_else(|| {
            let h = r.outputs[i].clone();
            i += 1;
            h
        });
        let sample_guard = self.backend.read_view(samples.view())?;
        let sample_rows: &[aim::Sample] = bytemuck::cast_slice(sample_guard.words());
        let reports = self.backend.read_view(r.outputs[4].view())?;
        let reports = reports.as_slice::<u32>()?;
        let mut unresolved = 0usize;
        for (k, sample) in sample_rows.iter().enumerate() {
            if k % 4096 == 0 {
                ctx.check()?;
            }
            if reports[k] != 0
                && (sample.pupil[0] * sample.pupil[0] + sample.pupil[1] * sample.pupil[1]).to_f64()
                    <= 1. + 1e-12
            {
                unresolved += 1;
            }
        }
        if unresolved > 0 {
            return Err(MkError::Unsupported(format!(
                "P6 has {unresolved} unresolved in-pupil aims; reference completion required"
            )));
        }
        let power = power_id.map(|_| {
            let h = r.outputs[i].clone();
            i += 1;
            h
        });
        let records = hits
            .records
            .map(|_| [r.outputs[i].clone(), r.outputs[i + 1].clone()]);
        if let Some(power) = &power {
            let v = self.backend.read_view(power.view())?;
            let rows: &[rad::PolarizationValue] = bytemuck::cast_slice(v.words());
            if rows
                .iter()
                .any(|v| v.status == mklight_gpu::consts::MKL_RAD_REFERENCE)
            {
                return Err(MkError::Unsupported(
                    "P7 pupil transport requires reference completion".into(),
                ));
            }
        }
        Ok((
            Batch {
                starts: r.outputs[0].clone(),
                hits: std::array::from_fn(|i| r.outputs[i + 1].clone()),
                records,
                samples,
                mask: r.outputs[6].clone(),
                power,
                count,
            },
            r.stats,
        ))
    }
    fn summary(
        &self,
        p: &mut ResidentPlan,
        mut values: PlanView,
        count: usize,
        mask: Option<PlanView>,
        kind: col::SegmentKind,
    ) -> MkResult<BufferId> {
        if values.desc.shape.len() == 1 {
            values.desc.shape = vec![1, count];
        }
        let offsets = fill::<u32>(self.backend, [2], |a| {
            a.copy_from_slice(&[0, count as u32]);
            Ok(())
        })?;
        let offsets = p.import(offsets.view());
        let output = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            if kind == col::SegmentKind::Moments2 {
                vec![1, 10]
            } else {
                vec![values.desc.shape[0], 1, 4]
            },
        ));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [1]));
        p.operations
            .push(ResidentOp::Collectives(col::Request::SegmentReduce(
                col::SegmentReduce {
                    values,
                    offsets: p.view(offsets)?,
                    mask,
                    mask_value: 1,
                    weights: None,
                    max_segment_len: count,
                    reference: None,
                    mode: ReduceMode::DoubleFloat,
                    kind,
                    output,
                    status,
                },
            )));
        Ok(output)
    }
    pub fn trace(
        &self,
        field: &PreparedField,
        wl: usize,
        settings: &PupilSettings,
        ctx: &JobContext,
    ) -> MkResult<WaveResult<ResidentTracedPupil>> {
        settings.validate()?;
        ctx.check()?;
        let mut stats = PlanStats::default();
        let sys = self.tracer.system();
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
            return Err(MkError::Numerical(
                "pupil transmits no selected-order light".into(),
            ));
        }
        let side = settings.rays + 2;
        let n = side * side;
        let (chunk, slot) = self
            .wavelengths
            .iter()
            .enumerate()
            .find_map(|(i, w)| w.iter().position(|v| *v == wl).map(|j| (i, j)))
            .unwrap();
        let field_spec = aim::FieldSpec::new(
            sys,
            self.tracer.paraxial(),
            field.point,
            chunk as u32,
            slot as u32,
            self.tracer.aiming(),
        )?;
        let metadata = record(self.backend, field_spec)?;
        let chief_sample = record(self.backend, aim::Sample::new([0.; 2], 0, 0))?;
        let mut p = self.plan();
        let c = p.import(chief_sample.view());
        let (chief, s) = self.probe(p, c, 1, &metadata, false, false, &ctx.subrange(0., 0.03))?;
        accumulate(&mut stats, &s);
        let st = self.backend.read_view(chief.hits[4].view())?;
        if st.as_slice::<u32>()?[0] != 0 {
            return Err(MkError::Trace("chief ray failed".into()));
        }
        let c = self.backend.read_view(chief.hits[0].view())?;
        let c = c.as_slice::<f32>()?;
        let chief_df: [Df32; 7] = std::array::from_fn(|j| {
            if j < 6 {
                Df32::new(c[j], c[j + 9])
            } else {
                Df32::new(c[6], c[7])
            }
        });
        let v3 = |a: [f64; 3]| Vec3::new(a[0], a[1], a[2]);
        let cp = v3(std::array::from_fn(|j| chief_df[j].to_f64()));
        let dc = v3(std::array::from_fn(|j| chief_df[j + 3].to_f64()));
        let sign = dc.z.signum();
        if sign == 0. {
            return Err(MkError::Trace("chief ray parallel to image".into()));
        }
        let mut p = self.plan();
        let count = n;
        let probes = p.scratch(BufferDesc::new(ElementType::U32, [count, 8]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Lattice(Op::Generate {
                side,
                centres: false,
                samples: probes,
            })));
        let (lat, s) = self.probe(
            p,
            probes,
            count,
            &metadata,
            true,
            settings.amplitude != AmplitudeModel::Uniform,
            &ctx.subrange(0.03, 0.25),
        )?;
        accumulate(&mut stats, &s);
        let valid = self.backend.read_view(lat.mask.view())?;
        let valid = valid.as_slice::<u32>()?;
        let inside: Vec<bool> = valid[..n].iter().map(|v| *v != 0).collect();
        let survived = inside.iter().filter(|&&v| v).count();
        if survived == 0 {
            return Err(MkError::Trace(
                "no ray of pupil lattice reaches image".into(),
            ));
        }
        let lattice = crate::traced::Lattice::new(settings.rays);
        let loops = if settings.rim {
            let mut saddles = Vec::new();
            for j in 0..side - 1 {
                for i in 0..side - 1 {
                    let k = j * side + i;
                    let case = usize::from(inside[k])
                        | (usize::from(inside[k + 1]) << 1)
                        | (usize::from(inside[k + side + 1]) << 2)
                        | (usize::from(inside[k + side]) << 3);
                    if case == 5 || case == 10 {
                        saddles.push((i, j));
                    }
                }
            }
            let mut centres = std::collections::BTreeMap::new();
            if !saddles.is_empty() {
                let samples = fill::<u32>(self.backend, [saddles.len(), 8], |a| {
                    for (k, (row, &(i, j))) in bytemuck::cast_slice_mut::<_, aim::Sample>(a)
                        .iter_mut()
                        .zip(&saddles)
                        .enumerate()
                    {
                        let p = lattice.point(i, j);
                        let h = 1. / (settings.rays - 1) as f64;
                        *row = aim::Sample::new([p[0] + h, p[1] + h], 0, k as u64);
                    }
                    Ok(())
                })?;
                let mut plan = self.plan();
                let samples = plan.import(samples.view());
                let (probes, s) = self.probe(
                    plan,
                    samples,
                    saddles.len(),
                    &metadata,
                    true,
                    false,
                    &ctx.subrange(0.25, 0.25),
                )?;
                accumulate(&mut stats, &s);
                let mask = self.backend.read_view(probes.mask.view())?;
                for (&cell, &v) in saddles.iter().zip(mask.as_slice::<u32>()?) {
                    centres.insert(cell, v != 0);
                }
            }
            crate::traced::chain_loops(&crate::traced::rim_segments(&lattice, &inside, |i, j| {
                centres[&(i, j)]
            }))
        } else {
            vec![]
        };
        let mut ids: Vec<usize> = loops.iter().flatten().copied().collect();
        ids.sort_unstable();
        ids.dedup();
        let loops: Vec<Vec<usize>> = loops
            .iter()
            .map(|l| l.iter().map(|i| ids.binary_search(i).unwrap()).collect())
            .collect();
        let rim = if ids.is_empty() {
            None
        } else {
            let edges = fill::<Df32>(self.backend, [ids.len(), 4], |a| {
                for (row, &id) in a.chunks_mut(4).zip(&ids) {
                    let k = id / 2;
                    let q = k + if id % 2 == 0 { 1 } else { side };
                    let (a, b) = if inside[k] { (k, q) } else { (q, k) };
                    let a = lattice.point(a % side, a / side);
                    let b = lattice.point(b % side, b / side);
                    row.copy_from_slice(&[a[0], a[1], b[0], b[1]].map(Df32::from_f64));
                }
                Ok(())
            })?;
            let iterations = ((2. / (settings.rays - 1) as f64 / settings.rim_tolerance)
                .log2()
                .ceil() as usize)
                .div_ceil(4)
                + 1;
            let mut previous = None;
            let mut batch: Option<Batch> = None;
            for iteration in 0..=iterations {
                let mut p = self.plan();
                let edge = p.import(edges.view());
                let state_in = previous.as_ref().map(|h: &BufferHandle| p.import(h.view()));
                let status = batch.as_ref().map(|b| p.import(b.hits[4].view()));
                let state = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [ids.len(), 8]));
                let probe_count = ids.len() * if iteration == iterations { 1 } else { 16 };
                let samples = p.scratch(BufferDesc::new(ElementType::U32, [probe_count, 8]));
                p.operations
                    .push(ResidentOp::Wave(w::Request::Lattice(Op::Rim {
                        edges: p.view(edge)?,
                        previous: state_in.map(|i| p.view(i).unwrap()),
                        status: status.map(|i| p.view(i).unwrap()),
                        tolerance: Df32::from_f64(settings.rim_tolerance),
                        final_point: iteration == iterations,
                        state,
                        samples,
                    })));
                p.outputs = vec![state, samples];
                let range = ctx.subrange(
                    0.25 + 0.35 * iteration as f64 / (iterations + 1) as f64,
                    0.25 + 0.35 * (iteration + 1) as f64 / (iterations + 1) as f64,
                );
                let r = self
                    .backend
                    .submit_plan(p, &range.subrange(0., 0.1))?
                    .wait()?;
                accumulate(&mut stats, &r.stats);
                previous = Some(r.outputs[0].clone());
                let mut p = self.plan();
                let samples = p.import(r.outputs[1].view());
                let (b, s) = self.probe(
                    p,
                    samples,
                    probe_count,
                    &metadata,
                    true,
                    false,
                    &range.subrange(0.1, 1.),
                )?;
                accumulate(&mut stats, &s);
                batch = Some(b);
            }
            let state = self.backend.read_view(previous.as_ref().unwrap().view())?;
            let unresolved = state
                .as_slice::<Df32>()?
                .chunks(8)
                .filter(|v| v[7].hi == 0.)
                .count();
            if unresolved > 0 {
                return Err(MkError::Unsupported(format!(
                    "{unresolved} rim edges did not reach tolerance {}; reference completion required",
                    settings.rim_tolerance
                )));
            }
            batch
        };
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
            let mask = p.import(subset(&lat.mask, [n]));
            let mut v = p.view(points)?;
            v.desc.shape = vec![3, n];
            v.stride = 1;
            let summary = self.summary(
                &mut p,
                v,
                n,
                Some(p_view(mask, &BufferDesc::new(ElementType::U32, [n]))),
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
        let (obj1, obj2) = match field.source {
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
        for b in std::iter::once(&lat).chain(rim.as_ref()) {
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
            let mask = p.import(b.mask.view());
            let power = b.power.as_ref().map(|h| p.import(h.view()));
            let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b.count, 8]));
            let radii = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [b.count]));
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
                    radii,
                },
            )));
            let max = self.summary(
                &mut p,
                p_view(
                    radii,
                    &BufferDesc::new(
                        ElementType::DoubleFloat,
                        [if std::ptr::eq(b, &lat) { n } else { b.count }],
                    ),
                ),
                if std::ptr::eq(b, &lat) { n } else { b.count },
                None,
                col::SegmentKind::Summary,
            )?;
            raw_ids.push((output, max, status));
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
            sin = sin.max(v.as_slice::<Df32>()?[3].to_f64());
        }
        for (i, batch) in std::iter::once(&lat).chain(rim.as_ref()).enumerate() {
            let failures = self.backend.read_view(r.outputs[i * 3 + 2].view())?;
            let mask = self.backend.read_view(batch.mask.view())?;
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
        if let Some(b) = &rim {
            let raw = self.backend.read_view(r.outputs[3].view())?;
            let raw = raw.as_slice::<Df32>()?;
            let probes = self.backend.read_view(b.samples.view())?;
            let probes: &[aim::Sample] = bytemuck::cast_slice(probes.words());
            for l in &loops {
                uv_loops.push(
                    l.iter()
                        .map(|&k| [raw[k * 8].to_f64() / sin, raw[k * 8 + 1].to_f64() / sin])
                        .collect(),
                );
                ep_loops.push(
                    l.iter()
                        .map(|&k| probes[k].pupil.map(Df32::to_f64))
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
        let valid = p.import(subset(&lat.mask, [side, side]));
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
        let slopes = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [(side - 1) * (side - 1)],
        ));
        p.operations
            .push(ResidentOp::Wave(w::Request::Placement(Placement::Slopes {
                nodes: p.view(nodes)?,
                valid: p.view(valid)?,
                output: slopes,
            })));
        let maximum = self.summary(
            &mut p,
            p_view(
                slopes,
                &BufferDesc::new(ElementType::DoubleFloat, [(side - 1) * (side - 1)]),
            ),
            (side - 1) * (side - 1),
            None,
            col::SegmentKind::Summary,
        )?;
        p.outputs = vec![nodes, maximum];
        let r = self
            .backend
            .submit_plan(p, &ctx.subrange(0.72, 0.8))?
            .wait()?;
        accumulate(&mut stats, &r.stats);
        let gradient = self
            .backend
            .read_view(r.outputs[1].view())?
            .as_slice::<Df32>()?[3]
            .to_f64();
        let grid = if settings.grid == 0 {
            ((2. * gradient / 0.35).clamp(64., 1024.).ceil() as usize).next_power_of_two()
        } else {
            settings.grid
        };
        let nodes = r.outputs[0].clone();
        let valid = lat.mask.clone();
        let identity = blake3::hash(
            &serde_json::to_vec(&(
                sys.snapshot.hash().to_hex(),
                &self.radiometry.identity,
                wl,
                field.point,
                settings,
            ))
            .unwrap(),
        )
        .to_hex()
        .to_string();
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
        let lattice_rays = (0..n)
            .filter(|&k| {
                let p = lattice.point(k % side, k / side);
                p[0] * p[0] + p[1] * p[1] <= 1. + 1e-12
            })
            .count();
        let fraction = if ep_loops.is_empty() {
            survived as f64 / lattice_rays as f64
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
            surviving_rays: survived,
            rim_points: ids.len(),
        };
        result.run_info.settings = serde_json::json!({"p6":"traced_pupil","field":field.point,"wavelength":wl,"pupil":settings,"grid":grid,"resident_stats":stats});
        Ok(WaveResult {
            value: ResidentTracedPupil {
                pupil: result.value,
                info,
                nodes,
                valid,
                mask,
                entrance_rim,
            },
            run_info: result.run_info,
            stats,
        })
    }
}
