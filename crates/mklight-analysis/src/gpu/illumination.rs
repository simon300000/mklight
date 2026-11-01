use super::*;
use crate::{
    illumination::{RelativeIlluminationResult, RelativeIlluminationSettings},
    util::{
        check_count, field_scale, field_sweep, field_unit, gauss_legendre_01, wavelength_or_primary,
    },
};
use mklight_gpu::{
    ReduceMode,
    collectives::{self, SegmentKind, SegmentReduce},
    geometry::{self, FieldData, Illumination},
};
/// Integrals are provisional until the sparse unresolved list is handled.
#[derive(Clone, Debug)]
pub struct IlluminationBuffers {
    pub integrals: BufferId,
    pub field_data: BufferId,
    pub unresolved: BufferId,
    pub unresolved_count: BufferId,
}
impl Geometry<'_> {
    /// Resident projected-solid-angle integral for P6/P8. The last field is
    /// the +y scale probe; all preceding fields have quadrature outputs.
    pub fn illumination(
        &self,
        p: &mut ResidentPlan,
        coordinates: &[[f64; 2]],
        wl: usize,
        radial: usize,
        azimuth: usize,
        ctx: &JobContext,
    ) -> MkResult<IlluminationBuffers> {
        check_count("radial samples", radial, 1, 512)?;
        check_count("azimuth samples", azimuth, 4, 4096)?;
        let mut fields: Vec<_> = coordinates
            .iter()
            .map(|&[x, y]| FieldPoint::normalized(x, y))
            .collect();
        fields.push(FieldPoint::normalized(0.0, 1.0));
        let real = mklight_core::RayAiming::Real {
            tolerance: match self.tracer.aiming() {
                mklight_core::RayAiming::Real { tolerance } => tolerance,
                _ => 1e-12,
            },
        };
        let g = self.append_grid(
            p,
            &fields,
            &[wl],
            &[[0.0; 2]],
            real,
            self.tracer.system().image_index(),
            false,
            true,
            ctx,
        )?;
        let (si, wi) = self.chunk(wl)?;
        let metadata = self.records::<FieldSpec>(p, fields.len(), |i| {
            FieldSpec::new(
                self.tracer.system(),
                self.tracer.paraxial(),
                fields[i],
                si,
                wi,
                real,
            )
        })?;
        let (rx, rw) = gauss_legendre_01(radial);
        let per = radial * azimuth;
        let n = coordinates.len() * per;
        let dphi = std::f64::consts::TAU / azimuth as f64;
        let samples = self.records::<Sample>(p, n, |i| {
            if i % 4096 == 0 {
                ctx.check()?;
            }
            let k = i % per;
            let r = k / azimuth;
            let a = k % azimuth;
            let (sn, cs) = (a as f64 * dphi).sin_cos();
            let mut node = Sample::new([rx[r] * cs, rx[r] * sn], (i / per) as u32, i as u64);
            node.weight = (rw[r] * rx[r] * dphi) as f32;
            Ok(node)
        })?;
        let ids = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4, n]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations
            .push(ResidentOp::Geometry(geometry::Request::Illumination(
                Illumination {
                    ids,
                    systems: SystemSet::Uploaded(self.systems),
                    fields: p.view(metadata)?,
                    chiefs: g.hits.clone(),
                    slots: self.tracer.system().num_surfaces(),
                    nodes: p.view(samples)?,
                    output: terms,
                    status,
                },
            )));
        let node_status = status;
        let offsets = self.import::<u32>(p, [coordinates.len() + 1], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i * per) as u32;
            }
            Ok(())
        })?;
        let sum = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [4, coordinates.len()],
        ));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [coordinates.len()]));
        p.operations.push(ResidentOp::Collectives(
            collectives::Request::SegmentReduce(SegmentReduce {
                values: p.view(terms)?,
                offsets: p.view(offsets)?,
                mask: None,
                mask_value: 0,
                weights: None,
                max_segment_len: per,
                reference: None,
                mode: ReduceMode::DoubleFloat,
                kind: SegmentKind::Sum,
                output: sum,
                status,
            }),
        ));
        let angles = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [fields.len(), 5]));
        p.operations
            .push(ResidentOp::Geometry(geometry::Request::FieldData(
                FieldData {
                    record_slots: self.tracer.system().num_surfaces(),
                    hits: g.hits,
                    kind: 0,
                    pupil_step: Df32::ZERO,
                    output: angles,
                },
            )));
        let values = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let unresolved = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        let count = p.scratch(BufferDesc::new(ElementType::U32, [1]));
        p.operations.push(ResidentOp::Collectives(
            collectives::Request::StableCompact(collectives::StableCompact {
                values: p.view(ids)?,
                ids: p.view(ids)?,
                mask: p.view(node_status)?,
                output: values,
                output_ids: unresolved,
                count,
            }),
        ));
        Ok(IlluminationBuffers {
            integrals: sum,
            field_data: angles,
            unresolved,
            unresolved_count: count,
        })
    }
}
/// Matched RI rows `[solid angle, surviving area share, CRA, cos4]`, plus
/// on-axis normalisation, image coordinates and reproducibility evidence.
pub type IlluminationRows = (Vec<[f64; 4]>, f64, Vec<[f64; 2]>, RunInfo);
/// Resident RI rows for field coordinates (see [`IlluminationRows`]).
pub fn illumination_rows(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    coordinates: &[[f64; 2]],
    wl: usize,
    radial: usize,
    azimuth: usize,
    ctx: &JobContext,
) -> MkResult<IlluminationRows> {
    check_count("radial samples", radial, 1, 512)?;
    check_count("azimuth samples", azimuth, 4, 4096)?;
    let geo = Geometry::new(b, t, ctx)?;
    let per = radial * azimuth;
    let tile = (65536 / per).max(1);
    let mut rows = Vec::new();
    let mut images = Vec::new();
    let mut omega0 = f64::NAN;
    let mut run: Option<RunInfo> = None;
    let mut attempted = 0;
    let mut resolved = 0;
    let mut polish_seconds = 0.0;
    for start in (0..coordinates.len().max(1)).step_by(tile) {
        let end = (start + tile).min(coordinates.len());
        let mut coords = coordinates[start.min(end)..end].to_vec();
        coords.push([0.0; 2]);
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        p.settings = serde_json::json!({"coordinates":coords,"wavelength":wl,"radial_samples":radial,"azimuth_samples":azimuth});
        let buffers = geo.illumination(&mut p, &coords, wl, radial, azimuth, ctx)?;
        p.outputs = vec![
            buffers.integrals,
            buffers.field_data,
            buffers.unresolved,
            buffers.unresolved_count,
        ];
        let a = start as f64 / coordinates.len().max(1) as f64;
        let z = end as f64 / coordinates.len().max(1) as f64;
        let r = geo.run(p, &ctx.subrange(0.95 * a, 0.95 * (a + 0.9 * (z - a))))?;
        let v = b.read_view(r.outputs[0].view())?;
        let v = v.as_slice::<Df32>()?;
        let ca = b.read_view(r.outputs[1].view())?;
        let ca = ca.as_slice::<Df32>()?;
        let n = coords.len();
        let count = b.read_view(r.outputs[3].view())?.as_slice::<u32>()?[0] as usize;
        let mut corrections = vec![[mklight_core::math::KahanSum::default(); 3]; n];
        if count > 0 {
            ctx.check()?;
            attempted += count;
            let indices = b.read_view(r.outputs[2].view())?;
            let indices = &indices.as_slice::<u32>()?[..count];
            let clock = std::time::Instant::now();
            let corrected = crate::illumination::polish_illumination_nodes(
                t,
                &coords,
                wl,
                radial,
                azimuth,
                indices,
                &ctx.subrange(0.95 * (a + 0.9 * (z - a)), 0.95 * z),
            )?;
            polish_seconds += clock.elapsed().as_secs_f64();
            for (index, value, failed) in corrected {
                if !failed {
                    resolved += 1;
                    let field = index as usize / per;
                    corrections[field][0].add(value[0]);
                    corrections[field][1].add(value[1]);
                    corrections[field][2].add(1.0);
                }
            }
        }

        let mut warnings = Vec::new();
        for i in 0..n {
            let failed = (v[2 * n + i].to_f64() - corrections[i][2].value()) as usize;
            let omega = v[i].to_f64() + corrections[i][0].value();
            let row = if ca[5 * i].hi.is_finite() {
                [
                    omega,
                    (v[n + i].to_f64() + corrections[i][1].value()) / std::f64::consts::PI,
                    ca[5 * i + 2].to_f64(),
                    v[3 * n + i].to_f64() / std::f64::consts::PI,
                ]
            } else {
                [f64::NAN; 4]
            };
            if failed > 0 {
                warnings.push(format!("field ({}, {}): {failed} of {per} pupil nodes have no ray through the image point",coords[i][0],coords[i][1]));
            }
            if i + 1 == n {
                omega0 = row[0];
            } else {
                rows.push(row);
                images.push([ca[5 * i].to_f64(), ca[5 * i + 1].to_f64()]);
            }
        }
        if let Some(run) = &mut run {
            run.duration_ms += r.run_info.duration_ms;
            run.settings["extra_plans"]
                .as_array_mut()
                .unwrap()
                .push(r.run_info.settings);
            run.warnings.extend(warnings);
        } else {
            let mut first = r.run_info;
            first.settings["extra_plans"] = serde_json::json!([]);
            first.warnings.extend(warnings);
            run = Some(first);
        }
    }
    if !(omega0 > 0.0) {
        return Err(mklight_core::MkError::Numerical(
            "on-axis beam has no projected solid angle".into(),
        ));
    }
    ctx.check()?;
    ctx.report(1.0, "relative illumination");
    let mut run = run.unwrap();
    run.duration_ms += 1000.0 * polish_seconds;
    run.settings["ri_polish"] = serde_json::json!({"attempted_nodes":attempted,"resolved_nodes":resolved,"remaining_failed_nodes":attempted-resolved,"seconds":polish_seconds,"reference_backend":"cpu-f64"});
    if attempted > 0 {
        run.warnings.push(format!(
            "RI: {attempted} unresolved device nodes used the f64 reference; {resolved} resolved"
        ));
    }
    Ok((rows, omega0, images, run))
}
pub fn relative_illumination_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &RelativeIlluminationSettings,
    ctx: &JobContext,
) -> MkResult<RelativeIlluminationResult> {
    check_count("field samples", s.field_samples, 2, 10_001)?;
    let wl = wavelength_or_primary(t, s.wavelength)?;
    let sweep = field_sweep(t, s.field_samples, s.field_axes);
    let (rows, omega0, _, run) = illumination_rows(
        b,
        t,
        &sweep.coordinates,
        wl,
        s.radial_samples,
        s.azimuth_samples,
        &ctx.subrange(0.0, 0.95),
    )?;
    ctx.check()?;
    ctx.report(1.0, "relative illumination");
    Ok(RelativeIlluminationResult {
        field_values: sweep.heights.iter().map(|h| h * field_scale(t)).collect(),
        field_unit: field_unit(t.system().system().fields.kind).into(),
        normalized_field: sweep.heights,
        field_coordinates: sweep.coordinates,
        wavelength: wl,
        relative_illumination: rows.iter().map(|r| r[0] / omega0).collect(),
        projected_solid_angle: rows.iter().map(|r| r[0]).collect(),
        vignetting_fraction: rows.iter().map(|r| 1.0 - r[1]).collect(),
        chief_ray_angle_deg: rows.iter().map(|r| r[2]).collect(),
        cos4_object: rows.iter().map(|r| r[3]).collect(),
        run_info: stamp(run, "relative_illumination", s, t.warnings().to_vec()),
    })
}
