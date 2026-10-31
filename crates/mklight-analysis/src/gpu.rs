//! Backend-neutral resident geometric adapters. These are deliberately not
//! wired into engine routing; qualification and automatic policy belong there.
use mklight_core::{JobContext, MkResult, RunInfo};
use mklight_gpu::{
    ComputeBackend, Df32, PackOptions, PackedSystem, SystemHandle, aiming::*, resident::*,
};
use mklight_trace::{FieldPoint, SequentialTracer};
mod energy;
mod spot;
pub use energy::{encircled_energy_job, geometric_mtf_job};
pub use spot::spot_diagram_job;

/// A reusable system upload. Host work is immutable resolution/packing and
/// small first-order field metadata; generation and tracing run on the backend.
pub struct Geometry<'a> {
    pub backend: &'a dyn ComputeBackend,
    pub tracer: &'a SequentialTracer,
    systems: SystemHandle,
    wavelengths: Vec<Vec<usize>>,
}
impl Drop for Geometry<'_> {
    fn drop(&mut self) {
        let _ = self.backend.release_systems(self.systems);
    }
}
impl<'a> Geometry<'a> {
    pub fn new(
        backend: &'a dyn ComputeBackend,
        tracer: &'a SequentialTracer,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        ctx.check()?;
        let primary = tracer.system().primary_wavelength;
        let mut chunks = Vec::new();
        let mut packed = Vec::new();
        // Include the primary in every chunk: real-image-height fields use it
        // even when a different wavelength was selected for the analysis.
        for start in (0..tracer.num_wavelengths()).step_by(15) {
            let mut w: Vec<_> = (start..(start + 15).min(tracer.num_wavelengths())).collect();
            if !w.contains(&primary) {
                w.push(primary);
            }
            packed.push(PackedSystem::pack(
                tracer.system(),
                &PackOptions {
                    wavelengths: Some(w.clone()),
                    radius_hints: None,
                },
            )?);
            chunks.push(w);
        }
        let systems = backend.upload_systems(&packed)?;
        Ok(Self {
            backend,
            tracer,
            systems,
            wavelengths: chunks,
        })
    }
    fn import<T: BufferElement>(
        &self,
        p: &mut ResidentPlan,
        shape: impl Into<Vec<usize>>,
        mut fill: impl FnMut(&mut [T]) -> MkResult<()>,
    ) -> MkResult<BufferId> {
        let h = self
            .backend
            .fill_buffer(BufferDesc::new(T::ELEMENT, shape), &mut |mut f| {
                fill(f.as_mut_slice::<T>()?)
            })?;
        Ok(p.import(h.view()))
    }
    fn records<T: bytemuck::Pod>(
        &self,
        p: &mut ResidentPlan,
        count: usize,
        mut fill: impl FnMut(usize) -> MkResult<T>,
    ) -> MkResult<BufferId> {
        self.import::<u32>(p, [count, size_of::<T>() / 4], |words| {
            for (i, v) in bytemuck::cast_slice_mut::<_, T>(words)
                .iter_mut()
                .enumerate()
            {
                *v = fill(i)?;
            }
            Ok(())
        })
    }
    /// Append resident generation and trace for a field/wavelength/pupil grid.
    /// Rows follow fields, wavelengths, then pupil; chiefs include primary.
    pub fn append_grid(
        &self,
        p: &mut ResidentPlan,
        fields: &[FieldPoint],
        wls: &[usize],
        pupil: &[[f64; 2]],
        aiming: mklight_core::RayAiming,
        last: usize,
        clip: bool,
        records: bool,
        ctx: &JobContext,
    ) -> MkResult<ResidentGrid> {
        ctx.check()?;
        let mut cw = wls.to_vec();
        cw.push(self.tracer.system().primary_wavelength);
        cw.sort_unstable();
        cw.dedup();
        let nf = fields.len() * cw.len();
        let metadata = self.records::<FieldSpec>(p, nf, |i| {
            let w = cw[i % cw.len()];
            let (si, wi) = self
                .wavelengths
                .iter()
                .enumerate()
                .find_map(|(s, ws)| ws.iter().position(|&v| v == w).map(|k| (s, k)))
                .unwrap();
            FieldSpec::new(
                self.tracer.system(),
                self.tracer.paraxial(),
                fields[i / cw.len()],
                si as u32,
                wi as u32,
                aiming,
            )
        })?;
        let steps = self.import::<u32>(p, [last], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i + 1) as u32;
            }
            Ok(())
        })?;
        let c_samples =
            self.records::<Sample>(p, nf, |i| Ok(Sample::new([0.0; 2], i as u32, i as u64)))?;
        let count = fields.len() * wls.len() * pupil.len();
        let samples = self.records::<Sample>(p, count, |i| {
            if i % 4096 == 0 {
                ctx.check()?;
            }
            let row = i / pupil.len();
            let wi = row % wls.len();
            let fi = row / wls.len();
            let index = fi * cw.len() + cw.iter().position(|&w| w == wls[wi]).unwrap();
            let mut s = Sample::new(pupil[i % pupil.len()], index as u32, i as u64);
            s.weight = self.tracer.system().wavelength_weights[wls[wi]] as f32;
            Ok(s)
        })?;
        let chiefs = self.append_trace(p, metadata, c_samples, nf, steps, aiming, false, false)?;
        let hits = self.append_trace(p, metadata, samples, count, steps, aiming, clip, records)?;
        Ok(ResidentGrid {
            hits,
            chiefs,
            chief_wavelengths: cw,
            fields: fields.len(),
            wavelengths: wls.to_vec(),
            pupil_count: pupil.len(),
        })
    }
    fn append_trace(
        &self,
        p: &mut ResidentPlan,
        metadata: BufferId,
        samples: BufferId,
        n: usize,
        steps: BufferId,
        aiming: mklight_core::RayAiming,
        clip: bool,
        records: bool,
    ) -> MkResult<ResidentHits> {
        let generated = rays(p, n);
        let tolerance = match aiming {
            mklight_core::RayAiming::Real { tolerance } => tolerance as f32,
            _ => 1e-10,
        };
        p.operations
            .push(ResidentOp::Aiming(Request::GenerateAim(GenerateAim {
                systems: SystemSet::Uploaded(self.systems),
                samples: p.view(samples)?,
                sampling: Sampling::Explicit,
                unresolved: None,
                field_wavelength: p.view(metadata)?,
                field_key: None,
                tolerance,
                max_iterations: 60,
                output: generated.clone(),
            })));
        let out = hits(
            p,
            n,
            if records {
                p.desc(steps)?.len()? + 1
            } else {
                0
            },
        );
        p.operations
            .push(ResidentOp::Aiming(Request::TraceResident(TraceResident {
                systems: SystemSet::Uploaded(self.systems),
                input: generated,
                steps: p.view(steps)?,
                start_surface: 0,
                apply_apertures: clip,
                radiometry: 0,
                newton: mklight_gpu::NewtonSettings::default(),
                output: out.clone(),
            })));
        Ok(out)
    }
    pub fn coordinates(
        &self,
        p: &mut ResidentPlan,
        g: &ResidentGrid,
        delta: f64,
        same_wavelength: bool,
    ) -> MkResult<(BufferId, BufferId)> {
        let n = g.hits.rays.count;
        let map = self.import::<u32>(p, [n], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                let row = i / g.pupil_count;
                let wi = row % g.wavelengths.len();
                let fi = row / g.wavelengths.len();
                let w = if same_wavelength {
                    g.wavelengths[wi]
                } else {
                    self.tracer.system().primary_wavelength
                };
                *v = (fi * g.chief_wavelengths.len()
                    + g.chief_wavelengths.iter().position(|&v| v == w).unwrap())
                    as u32;
            }
            Ok(())
        })?;
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5, n]));
        let mask = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations.push(ResidentOp::Geometry(
            mklight_gpu::geometry::Request::Coordinates(mklight_gpu::geometry::Coordinates {
                hits: g.hits.clone(),
                chiefs: g.chiefs.clone(),
                chief_index: p.view(map)?,
                defocus: Df32::from_f64(delta),
                output,
                mask,
            }),
        ));
        Ok((output, mask))
    }
    pub fn moments(
        &self,
        p: &mut ResidentPlan,
        points: BufferId,
        mask: BufferId,
        segment: usize,
        weighted: bool,
    ) -> MkResult<BufferId> {
        use mklight_gpu::collectives::*;
        let n = p.desc(mask)?.len()?;
        let ns = n / segment;
        let offsets = self.import::<u32>(p, [ns + 1], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i * segment) as u32;
            }
            Ok(())
        })?;
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [ns, 10]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [ns]));
        let mut values = p.view(points)?;
        values.desc.shape = vec![2, n];
        let weights = if weighted {
            let mut w = p.view(points)?;
            w.offset = 4 * n;
            w.desc.shape = vec![n];
            Some(w)
        } else {
            None
        };
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::SegmentReduce(SegmentReduce {
                values,
                offsets: p.view(offsets)?,
                mask: Some(p.view(mask)?),
                mask_value: 1,
                weights,
                max_segment_len: segment,
                reference: None,
                mode: mklight_gpu::ReduceMode::DoubleFloat,
                kind: SegmentKind::Moments2,
                output,
                status,
            }),
        ));
        Ok(output)
    }
    fn run(&self, p: ResidentPlan, ctx: &JobContext) -> MkResult<PlanResult> {
        self.backend.submit_plan(p, ctx)?.wait()
    }
}
#[derive(Clone, Debug)]
pub struct ResidentGrid {
    pub hits: ResidentHits,
    pub chiefs: ResidentHits,
    pub chief_wavelengths: Vec<usize>,
    pub fields: usize,
    pub wavelengths: Vec<usize>,
    pub pupil_count: usize,
}
fn stamp(
    mut run: RunInfo,
    name: &str,
    s: &impl serde::Serialize,
    warnings: Vec<String>,
) -> RunInfo {
    run.analysis = name.into();
    // Preserve the plan's backend/probe/build/stage evidence under execution.
    let execution = run.settings;
    run.settings = serde_json::to_value(s).unwrap();
    run.settings
        .as_object_mut()
        .unwrap()
        .insert("resident_execution".into(), execution);
    run.warnings.extend(warnings);
    run
}
fn value(f: &[f32], n: usize, i: usize, k: usize) -> f64 {
    f[k * n + i] as f64 + f[(k + 9) * n + i] as f64
}
fn chief(f: &[f32], status: &[u32], n: usize, i: usize, delta: f64) -> [f64; 2] {
    if status[i] != 0 {
        return [f64::NAN; 2];
    }
    let dz = value(f, n, i, 5);
    std::array::from_fn(|k| value(f, n, i, k) + delta * value(f, n, i, k + 3) / dz)
}
