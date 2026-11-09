//! Backend-neutral resident geometric adapters. These are deliberately not
//! wired into engine routing; qualification and automatic policy belong there.
use mklight_core::{JobContext, MkResult, RunInfo};
use mklight_gpu::{
    ComputeBackend, Df32, PackOptions, PackedSystem, SystemHandle, aiming::*, resident::*,
};
use mklight_trace::{FieldPoint, SequentialTracer};
pub mod aim;
mod chromatic;
mod energy;
pub mod plans;
pub use chromatic::{lateral_colour_job, longitudinal_aberration_job};
mod curvature;
pub use curvature::{field_curvature_job, grid_distortion_job};
mod chief_maps;
mod layout;
mod maps;
pub use chief_maps::ChiefDesign;
pub use layout::layout_job;
mod illumination;
mod illumination_split;
pub use illumination::{
    IlluminationBuffers, IlluminationRows, RiDesign, illumination_rows, relative_illumination_job,
};
/// Study key of the field-map, illumination and radiometry work-split
/// designs (`mklight_core::split`).
pub const SPLIT_KEY: &str = "fieldmaps";
mod records;
pub use records::{
    GridResult, footprint_job, footprint_overlay_job, ray_grid_job, semi_diameters_job,
};
mod fans;
pub use fans::{opd_fan_job, ray_fan_job};
pub use maps::field_map_job;
mod spot;
pub use energy::{
    best_focus_job, encircled_energy_job, encircled_energy_with_uncertainty_job, geometric_mtf_job,
};
pub use spot::{spot_diagram_job, spot_diagram_packed_job};

/// A reusable system upload. Host work is immutable resolution/packing, small
/// first-order field metadata and f64 chief rows from the reference tracer's
/// field preparation; pupil generation and tracing run on the backend.
pub struct Geometry<'a> {
    pub backend: &'a dyn ComputeBackend,
    pub tracer: &'a SequentialTracer,
    systems: SystemHandle,
    wavelengths: Vec<Vec<usize>>,
    /// Reference tracers for aiming modes other than the tracer's own (maps
    /// and illumination force real aiming, pupil aberration none).
    others: std::sync::Mutex<Vec<(mklight_core::RayAiming, std::sync::Arc<SequentialTracer>)>>,
    /// How generations are aimed (work-split key `aiming`, [`aim`]).
    aim: aim::AimDesign,
    /// The device's share of large generations aimed by host and device
    /// together: the selected share once the analysis opts in
    /// ([`Self::split_large_generations`]), otherwise 0.
    split_share: f64,
    /// Aiming work since the last plan, reported with it.
    aim_evidence: std::sync::Mutex<aim::AimEvidence>,
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
        let chunk = mklight_gpu::layout::MAX_WAVELENGTHS - 1;
        for start in (0..tracer.num_wavelengths()).step_by(chunk) {
            let mut w: Vec<_> = (start..(start + chunk).min(tracer.num_wavelengths())).collect();
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
        let mut numeric = blake3::Hasher::new();
        for system in &packed {
            numeric.update(bytemuck::bytes_of(&system.header));
            numeric.update(bytemuck::cast_slice(&system.surfaces));
            numeric.update(bytemuck::cast_slice(&system.coefficients));
        }
        let systems = backend.upload_systems_cached(
            &SystemCacheKey {
                snapshot: tracer.system().snapshot.hash().to_hex(),
                library: numeric.finalize().to_hex().to_string(),
                configuration: "resolved".into(),
                spectrum: "all_chunks".into(),
                layout: mklight_gpu::consts::MKL_LAYOUT_VERSION,
                kernel_build: backend.precision().kernel_build.clone(),
                operation_settings: "geometry-v1".into(),
            },
            &packed,
        )?;
        Ok(Self {
            backend,
            tracer,
            systems,
            wavelengths: chunks,
            others: Default::default(),
            aim: aim::AimDesign::selected()?,
            split_share: 0.0,
            aim_evidence: Default::default(),
        })
    }
    /// Aim large generations by host and device together (`gpu::aim`, split
    /// generations): for the analyses where that was measured to pay, the
    /// spot diagram and the geometric MTF.
    pub(crate) fn split_large_generations(mut self) -> MkResult<Self> {
        self.split_share = aim::AimDesign::selection()?.1;
        Ok(self)
    }
    /// The aiming design of this adapter's generations.
    pub fn aim_design(&self) -> aim::AimDesign {
        self.aim
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
    /// Rows follow fields, wavelengths, then pupil. Chief rays (including the
    /// primary wavelength) are traced only by [`Self::append_chiefs`]. The
    /// rays are aimed now, by this adapter's design ([`aim`]); the plan only
    /// traces them.
    #[allow(clippy::too_many_arguments)]
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
        let steps = self.import::<u32>(p, [last], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i + 1) as u32;
            }
            Ok(())
        })?;
        // Field rows: fields × chief wavelengths (requested plus primary).
        let rows: Vec<_> = (0..nf)
            .map(|i| (fields[i / cw.len()], cw[i % cw.len()]))
            .collect();
        let rows = std::sync::Arc::new(self.aim_rows(p, &rows, aiming)?);
        // Field row of each requested wavelength.
        let columns: Vec<_> = wls
            .iter()
            .map(|w| cw.iter().position(|v| v == w).unwrap())
            .collect();
        let weights: Vec<f32> = wls
            .iter()
            .map(|&w| self.tracer.system().wavelength_weights[w] as f32)
            .collect();
        let count = fields.len() * wls.len() * pupil.len();
        let np = pupil.len();
        let nw = wls.len();
        let ncw = cw.len();
        let spec = |i: usize| {
            let row = i / np;
            let wi = row % nw;
            aim::RaySpec {
                row: (row / nw * ncw + columns[wi]) as u32,
                pupil: pupil[i % np],
                weight: weights[wi],
                id: i as u64,
            }
        };
        let input = self.aim_rays(p, &rows, count, &spec, ctx)?;
        let hits = self.trace_rays(p, input, steps, clip, records)?;
        Ok(ResidentGrid {
            hits,
            chiefs: None,
            chief_wavelengths: cw,
            fields: fields.len(),
            wavelengths: wls.to_vec(),
            pupil_count: pupil.len(),
            chief_source: Some(ChiefSource { rows, steps }),
        })
    }
    /// Chief rows for (field point, system wavelength) pairs, solved in f64 by
    /// the reference tracer for `aiming` (its prepared system fields are
    /// reused), so the backend skips its chief/real-image-height solve.
    pub fn host_seeds(
        &self,
        p: &mut ResidentPlan,
        rows: &[(FieldPoint, usize)],
        aiming: mklight_core::RayAiming,
    ) -> MkResult<BufferId> {
        let (seeds, host) = if aiming == self.tracer.aiming() {
            host_seeds(self.tracer, rows)
        } else {
            host_seeds(&*self.tracer_for(aiming)?, rows)
        };
        p.host.add(&host);
        self.records::<ChiefSeed>(p, rows.len(), |i| Ok(seeds[i]))
    }
    fn tracer_for(
        &self,
        aiming: mklight_core::RayAiming,
    ) -> MkResult<std::sync::Arc<SequentialTracer>> {
        let mut others = self.others.lock().unwrap_or_else(|e| e.into_inner());
        if let Some((_, t)) = others.iter().find(|(a, _)| *a == aiming) {
            return Ok(t.clone());
        }
        let t = std::sync::Arc::new(SequentialTracer::with_settings(
            self.tracer.system_arc(),
            mklight_trace::TracerSettings {
                aiming: Some(aiming),
                ..*self.tracer.settings()
            },
        )?);
        others.push((aiming, t.clone()));
        Ok(t)
    }
    /// Append the unclipped chief traces of a grid, one per field and chief
    /// wavelength (requested wavelengths plus the primary), aimed now as the
    /// grid's rays are.
    pub fn append_chiefs(&self, p: &mut ResidentPlan, g: &mut ResidentGrid) -> MkResult<()> {
        let Some(ChiefSource { rows, steps }) = g.chief_source.clone() else {
            return Err(mklight_core::MkError::InvalidParam(
                "grid has no chief metadata".into(),
            ));
        };
        let nf = g.fields * g.chief_wavelengths.len();
        let spec = |i: usize| aim::RaySpec {
            row: i as u32,
            pupil: [0.0; 2],
            weight: 1.0,
            id: i as u64,
        };
        let input = self.aim_rays(p, &rows, nf, &spec, &JobContext::new())?;
        g.chiefs = Some(self.trace_rays(p, input, steps, false, false)?);
        Ok(())
    }
    /// Packed wavelength chunk and slot of system wavelength `w`.
    fn chunk(&self, w: usize) -> MkResult<(u32, u32)> {
        self.wavelengths
            .iter()
            .enumerate()
            .find_map(|(s, ws)| {
                ws.iter()
                    .position(|&v| v == w)
                    .map(|k| (s as u32, k as u32))
            })
            .ok_or_else(|| mklight_core::MkError::InvalidParam(format!("no wavelength {w}")))
    }
    /// Content identity of imported field metadata, for chief-seed reuse.
    fn field_key(&self, p: &ResidentPlan, metadata: BufferId) -> MkResult<Option<String>> {
        Ok(match &p.buffers[metadata.0] {
            PlanBuffer::Import(v) => {
                let view = self.backend.read_view(v.clone())?;
                Some(
                    blake3::hash(bytemuck::cast_slice(view.words()))
                        .to_hex()
                        .to_string(),
                )
            }
            _ => None,
        })
    }
    /// Append the resident trace of aimed starting rays (`last` steps from
    /// `steps`), clipped at the apertures when `clip`, with per-surface
    /// records when `records`.
    pub(crate) fn trace_rays(
        &self,
        p: &mut ResidentPlan,
        input: ResidentRays,
        steps: BufferId,
        clip: bool,
        records: bool,
    ) -> MkResult<ResidentHits> {
        let out = hits(
            p,
            input.count,
            if records {
                p.desc(steps)?.len()? + 1
            } else {
                0
            },
        );
        p.operations
            .push(ResidentOp::Aiming(Request::TraceResident(TraceResident {
                systems: SystemSet::Uploaded(self.systems),
                input,
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
        let map = self.chief_map(p, g, same_wavelength)?;
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5, n]));
        let mask = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations.push(ResidentOp::Geometry(
            mklight_gpu::geometry::Request::Coordinates(mklight_gpu::geometry::Coordinates {
                hits: g.hits.clone(),
                chiefs: g.chiefs()?.clone(),
                chief_index: p.view(map)?,
                defocus: Df32::from_f64(delta),
                output,
                mask,
            }),
        ));
        Ok((output, mask))
    }
    /// U32 [rays]: the chief row of each hit, at its own wavelength or the primary.
    fn chief_map(
        &self,
        p: &mut ResidentPlan,
        g: &ResidentGrid,
        same_wavelength: bool,
    ) -> MkResult<BufferId> {
        let nc = g.chief_wavelengths.len();
        let column = |w: usize| g.chief_wavelengths.iter().position(|&v| v == w).unwrap();
        let primary = column(self.tracer.system().primary_wavelength);
        let columns: Vec<_> = g
            .wavelengths
            .iter()
            .map(|&w| if same_wavelength { column(w) } else { primary })
            .collect();
        self.import::<u32>(p, [g.hits.rays.count], |out| {
            for (row, rays) in out.chunks_mut(g.pupil_count.max(1)).enumerate() {
                let fi = row / g.wavelengths.len();
                rays.fill((fi * nc + columns[row % g.wavelengths.len()]) as u32);
            }
            Ok(())
        })
    }
    pub fn moments(
        &self,
        p: &mut ResidentPlan,
        points: BufferId,
        mask: BufferId,
        segment: usize,
        weighted: bool,
    ) -> MkResult<BufferId> {
        self.moments_at(p, points, mask, segment, weighted, 0)
    }
    fn moments_at(
        &self,
        p: &mut ResidentPlan,
        points: BufferId,
        mask: BufferId,
        segment: usize,
        weighted: bool,
        plane: usize,
    ) -> MkResult<BufferId> {
        use mklight_gpu::collectives::*;
        let n = p.desc(mask)?.len()?;
        let ns = n / segment;
        let offsets = self.offsets(p, ns, segment)?;
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [ns, 10]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [ns]));
        let mut values = p.view(points)?;
        values.desc.shape = vec![2, n];
        values.offset = plane * n;
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
    /// Weighted defocus quadratic per segment, DF [3, segments]: Σw|q|²,
    /// Σw q·t and Σw|t|² of the image offsets q and slope differences t from
    /// [`Self::coordinates`], centred on the segment means when `centroid`.
    /// The mean square at defocus d is (a + 2bd + cd²)/Σw; with the slope
    /// moments (also returned) the centroid moves as q̄ + d t̄.
    pub fn focus_quadratic(
        &self,
        p: &mut ResidentPlan,
        points: BufferId,
        mask: BufferId,
        moments: BufferId,
        segment: usize,
        centroid: bool,
    ) -> MkResult<(BufferId, BufferId)> {
        use mklight_gpu::collectives::*;
        let n = p.desc(mask)?.len()?;
        let ns = n / segment;
        let slope = self.moments_at(p, points, mask, segment, true, 2)?;
        let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3, n]));
        p.operations.push(ResidentOp::Geometry(
            mklight_gpu::geometry::Request::FocusTerms(mklight_gpu::geometry::FocusTerms {
                points: p.view(points)?,
                mask: p.view(mask)?,
                position_moments: p.view(moments)?,
                slope_moments: p.view(slope)?,
                segment_len: segment,
                centroid,
                output: terms,
            }),
        ));
        let offsets = self.offsets(p, ns, segment)?;
        let quad = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3, ns]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [ns]));
        p.operations.push(ResidentOp::Collectives(
            mklight_gpu::collectives::Request::SegmentReduce(SegmentReduce {
                values: p.view(terms)?,
                offsets: p.view(offsets)?,
                mask: None,
                mask_value: 0,
                weights: None,
                max_segment_len: segment,
                reference: None,
                mode: mklight_gpu::ReduceMode::DoubleFloat,
                kind: SegmentKind::Sum,
                output: quad,
                status,
            }),
        ));
        Ok((quad, slope))
    }
    /// U32 [segments+1] offsets of equal-length segments.
    fn offsets(&self, p: &mut ResidentPlan, segments: usize, len: usize) -> MkResult<BufferId> {
        self.import::<u32>(p, [segments + 1], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i * len) as u32;
            }
            Ok(())
        })
    }
    fn run(&self, p: ResidentPlan, ctx: &JobContext) -> MkResult<PlanResult> {
        let s = self.submit(p, ctx)?;
        self.finish(s)
    }
}
#[derive(Clone, Debug)]
pub struct ResidentGrid {
    pub hits: ResidentHits,
    /// Present after [`Geometry::append_chiefs`].
    pub chiefs: Option<ResidentHits>,
    pub chief_wavelengths: Vec<usize>,
    pub fields: usize,
    pub wavelengths: Vec<usize>,
    pub pupil_count: usize,
    /// Prepared field rows and trace steps, for the chief rays.
    chief_source: Option<ChiefSource>,
}
#[derive(Clone, Debug)]
struct ChiefSource {
    rows: std::sync::Arc<aim::AimRows>,
    steps: BufferId,
}
impl ResidentGrid {
    pub fn chiefs(&self) -> MkResult<&ResidentHits> {
        self.chiefs.as_ref().ok_or_else(|| {
            mklight_core::MkError::InvalidParam("grid chief rays were not appended".into())
        })
    }
}
fn stamp(
    mut run: RunInfo,
    name: &str,
    s: &impl serde::Serialize,
    warnings: Vec<String>,
) -> RunInfo {
    run.analysis = name.into();
    run.settings["geometry_version"] = serde_json::json!(mklight_gpu::geometry::VERSION);
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
