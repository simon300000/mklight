use super::*;
use crate::{
    curvature::*,
    util::{
        check_count, check_finite, field_scale, field_sweep, field_unit, first_order_x, linspace,
        wavelength_or_primary,
    },
};
use mklight_core::surface_math::{Symmetry, system_symmetry};
use mklight_gpu::geometry::{self, Coddington, FieldData};
use rayon::prelude::*;
use std::sync::atomic::AtomicUsize;
impl Geometry<'_> {
    /// Resident Coddington output for P6/P8; signed ray distances and dz.
    pub fn coddington(&self, p: &mut ResidentPlan, hits: ResidentHits) -> MkResult<BufferId> {
        let n = hits.rays.count;
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n, 3]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [n]));
        p.operations
            .push(ResidentOp::Geometry(geometry::Request::Coddington(
                Coddington {
                    systems: SystemSet::Uploaded(self.systems),
                    hits,
                    slots: self.tracer.system().num_surfaces(),
                    output,
                    status,
                },
            )));
        Ok(output)
    }
    /// Resident image x/y and launch-source coordinates for distortion consumers.
    pub fn image_source(&self, p: &mut ResidentPlan, hits: ResidentHits) -> MkResult<BufferId> {
        let output = p.scratch(BufferDesc::new(
            ElementType::DoubleFloat,
            [hits.rays.count, 5],
        ));
        p.operations
            .push(ResidentOp::Geometry(geometry::Request::FieldData(
                FieldData {
                    record_slots: self.tracer.system().num_surfaces(),
                    hits,
                    kind: 2,
                    pupil_step: Df32::from_f32(if self.tracer.system().object_at_infinity {
                        0.0
                    } else {
                        1.0
                    }),
                    output,
                },
            )));
        Ok(output)
    }
}
fn scales(t: &SequentialTracer, w: usize) -> MkResult<([f64; 2], [f64; 2])> {
    let fx = first_order_x(t, w)?;
    let fy = t.paraxial().at(w);
    Ok((
        if t.system().object_at_infinity {
            [
                fx.image_height_per_tan.unwrap_or(f64::NAN),
                fy.image_height_per_tan.unwrap_or(f64::NAN),
            ]
        } else {
            [
                fx.image_height_per_object_height.unwrap_or(f64::NAN),
                fy.image_height_per_object_height.unwrap_or(f64::NAN),
            ]
        },
        [fx.efl, fy.efl],
    ))
}
pub fn grid_distortion_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &GridDistortionSettings,
    ctx: &JobContext,
) -> MkResult<GridDistortionResult> {
    check_count("grid size", s.grid_size, 2, 1001)?;
    check_finite("grid half-width", s.half_width)?;
    for v in s.field_center {
        check_finite("grid field centre", v)?;
    }
    if s.half_width <= 0.0 {
        return Err(mklight_core::MkError::InvalidModel(
            "grid half-width must be positive".into(),
        ));
    }
    let w = wavelength_or_primary(t, s.wavelength)?;
    let x = linspace(-s.half_width, s.half_width, s.grid_size);
    let coords: Vec<_> = x
        .iter()
        .flat_map(|&y| {
            x.iter()
                .map(move |&x| [x + s.field_center[0], y + s.field_center[1]])
        })
        .collect();
    let (design, name) = super::chief_maps::ChiefDesign::selected()?;
    let fields: Vec<_> = coords
        .iter()
        .map(|&[x, y]| FieldPoint::normalized(x, y))
        .collect();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
    let hits = geo.append_chief_grid(
        &mut p,
        &fields,
        w,
        &[[0.0; 2]],
        t.aiming(),
        t.system().image_index(),
        true,
        design == super::chief_maps::ChiefDesign::DeviceChief,
        ctx,
    )?;
    let source = geo.image_source(&mut p, hits)?;
    p.outputs = vec![source];
    let mut r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    r.run_info.settings["fieldmaps_variant"] = serde_json::json!(name);
    let v = b.read_view(r.outputs[0].view())?;
    let v = v.as_slice::<Df32>()?;
    let failed = (0..coords.len())
        .filter(|&i| !v[5 * i].to_f64().is_finite())
        .count();
    if design == super::chief_maps::ChiefDesign::DeviceChief && failed > 0 {
        // A device chief solve that fails is completed by the reference.
        let mut out = crate::curvature::grid_distortion_job(t, s, ctx)?;
        out.run_info.settings["fieldmaps_variant"] = serde_json::json!(name);
        out.run_info.settings["device_chief_failures"] = serde_json::json!(failed);
        out.run_info.warnings.push(format!(
            "{failed} device chief solves failed; the grid was completed by the reference"
        ));
        return Ok(out);
    }
    let (scale, _) = scales(t, w)?;
    let rotational = system_symmetry(t.system().system()) == Symmetry::Rotational;
    let points: Vec<_> = coords
        .iter()
        .enumerate()
        .map(|(i, &field)| {
            let real = [v[5 * i].to_f64(), v[5 * i + 1].to_f64()];
            let paraxial = [
                scale[0] * v[5 * i + 2].to_f64(),
                scale[1] * v[5 * i + 3].to_f64(),
            ];
            let distortion_percent = if field == [0.0; 2] && rotational {
                0.0
            } else {
                100.0 * (real[0].hypot(real[1]) - paraxial[0].hypot(paraxial[1]))
                    / paraxial[0].hypot(paraxial[1])
            };
            GridPoint {
                field,
                real,
                paraxial,
                distortion_percent,
            }
        })
        .collect();
    let failed = points.iter().filter(|p| p.real[0].is_nan()).count();
    let mut warnings = t.warnings().to_vec();
    if failed > 0 {
        warnings.push(format!("{failed} grid chief rays failed"));
    }
    let max = points
        .iter()
        .map(|p| p.distortion_percent.abs())
        .filter(|v| v.is_finite())
        .fold(f64::NAN, f64::max);
    ctx.report(1.0, "grid distortion");
    Ok(GridDistortionResult {
        grid_size: s.grid_size,
        wavelength: w,
        points,
        max_abs_distortion_percent: max,
        run_info: stamp(r.run_info, "grid_distortion", s, warnings),
    })
}
pub fn field_curvature_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &FieldCurvatureSettings,
    ctx: &JobContext,
) -> MkResult<FieldCurvatureResult> {
    check_count("field samples", s.field_samples, 2, 100_001)?;
    check_finite("close-ray step", s.close_ray_step)?;
    if !(s.close_ray_step > 0.0 && s.close_ray_step < 0.5) {
        return Err(mklight_core::MkError::InvalidModel(
            "close-ray step must be in (0, 0.5)".into(),
        ));
    }
    let sweep = field_sweep(t, s.field_samples, s.field_axes);
    let fields: Vec<_> = sweep
        .coordinates
        .iter()
        .map(|&[x, y]| FieldPoint::normalized(x, y))
        .collect();
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let n = fields.len() * wls.len();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
    let g = geo.append_grid(
        &mut p,
        &fields,
        &wls,
        &[[0.0; 2]],
        t.aiming(),
        t.system().image_index(),
        false,
        true,
        ctx,
    )?;
    let cod = geo.coddington(&mut p, g.hits.clone())?;
    let source = geo.image_source(&mut p, g.hits)?;
    p.outputs = vec![cod, source];
    // The cross-check column is the host f64 differential pencil (the
    // close rays' limit, `crate::curvature`): device close rays differenced
    // in double-float lost significance against the f64 reference.
    let close_fallbacks = AtomicUsize::new(0);
    let fb = &close_fallbacks;
    let cross: Vec<[f64; 2]> = fields
        .par_iter()
        .flat_map_iter(|&fp| {
            let prepared = t.prepare_field(fp).ok();
            wls.iter().map(move |&w| {
                prepared
                    .as_ref()
                    .and_then(|f| cross_check_focus(t, f, w, s.close_ray_step, fb))
                    .map_or([f64::NAN; 2], |a| a.axial().into())
            })
        })
        .collect();
    debug_assert_eq!(cross.len(), n);
    let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    let views: Vec<_> = r
        .outputs
        .iter()
        .map(|h| b.read_view(h.view()))
        .collect::<MkResult<_>>()?;
    let v: Vec<_> = views
        .iter()
        .map(|v| v.as_slice::<Df32>())
        .collect::<MkResult<_>>()?;
    let rotational = system_symmetry(t.system().system()) == Symmetry::Rotational;
    let mut warnings = t.warnings().to_vec();
    let mut curves = Vec::new();
    let mut fallback_count = 0;
    for (wi, &w) in wls.iter().enumerate() {
        let (scale, efl) = scales(t, w)?;
        let mut rows = Vec::new();
        let mut bad = Vec::new();
        for fi in 0..fields.len() {
            ctx.check()?;
            let i = fi * wls.len() + wi;
            let a = &v[0][3 * i..];
            let c = cross[i];
            let im = &v[1][5 * i..];
            let axis = sweep.components[fi];
            let mut real = im[axis].to_f64();
            let mut q = im[2 + axis].to_f64();
            let mut paraxial = scale[axis] * q;
            if paraxial == 0.0 && !(sweep.heights[fi] == 0.0 && rotational) {
                // A zero denominator makes the sign of an arbitrarily small
                // chief residual observable as +/-infinity. Keep the reference
                // decision at this singular point and report the scalar fallback.
                let [hx, hy] = sweep.coordinates[fi];
                if let Ok(field) = t.prepare_field(FieldPoint::normalized(hx, hy)) {
                    let chief = t.chief(&field, w);
                    real = if chief.status.is_ok() {
                        [chief.ray.pos.x, chief.ray.pos.y][axis]
                    } else {
                        f64::NAN
                    };
                    q = match field.source {
                        mklight_trace::ObjectSource::Direction { dir } => {
                            [dir.x, dir.y][axis] / dir.z
                        }
                        mklight_trace::ObjectSource::Point { point } => [point.x, point.y][axis],
                    };
                    paraxial = scale[axis] * q;
                }
                fallback_count += 1;
            }
            let d = if sweep.heights[fi] == 0.0 && rotational {
                0.0
            } else {
                100.0 * (real - paraxial) / paraxial
            };
            let f = t.system().n_object[w] * efl[axis];
            let (dt, da) = if !t.system().object_at_infinity {
                (f64::NAN, f64::NAN)
            } else if sweep.heights[fi] == 0.0 && rotational {
                (0.0, 0.0)
            } else {
                (
                    100.0 * (real - f * q) / (f * q),
                    100.0 * (real - f * q.atan()) / (f * q.atan()),
                )
            };
            let row = [
                a[0].to_f64() * a[2].to_f64(),
                a[1].to_f64() * a[2].to_f64(),
                c[0],
                c[1],
                real,
                paraxial,
                d,
                dt,
                da,
            ];
            if (0..2).any(|j| (row[j] - row[j + 2]).abs() > 1e-3 * (1.0 + row[j].abs())) {
                bad.push(sweep.coordinates[fi]);
            }
            rows.push(row);
        }
        if let (Some(first), Some(last)) = (bad.first(), bad.last()) {
            warnings.push(format!("wavelength {w}: Coddington and close-ray foci differ by more than 1e-3 (1 + |Δz|) mm at {} field sample(s), [Hx, Hy] = {first:?}…{last:?}",bad.len()));
        }
        let col = |j| rows.iter().map(|r| r[j]).collect();
        curves.push(FieldCurvatureCurve {
            wavelength: w,
            wavelength_um: t.system().wavelengths_um[w],
            tangential: col(0),
            sagittal: col(1),
            tangential_close_rays: col(2),
            sagittal_close_rays: col(3),
            real_height: col(4),
            paraxial_height: col(5),
            distortion_percent: col(6),
            f_tan_theta_distortion_percent: col(7),
            f_theta_distortion_percent: col(8),
        });
    }
    if fallback_count > 0 {
        warnings.push(format!(
            "{fallback_count} singular distortion points used the f64 reference"
        ));
    }
    let close_fallbacks = close_fallbacks.into_inner();
    if close_fallbacks > 0 {
        warnings.push(format!(
            "{close_fallbacks} field sample(s) used finite-difference close rays (differential \
             tangent undefined)"
        ));
    }
    let mut run = stamp(r.run_info, "field_curvature_distortion", s, warnings);
    run.settings["adapter_fallbacks"] =
        serde_json::json!({"stage":"singular_distortion","count":fallback_count});
    run.settings["host_stages"] =
        serde_json::json!({"close_rays":"f64 differential pencil","fd_fallbacks":close_fallbacks});
    ctx.report(1.0, "field curvature");
    Ok(FieldCurvatureResult {
        field_coordinates: sweep.coordinates,
        field_values: sweep.heights.iter().map(|h| h * field_scale(t)).collect(),
        normalized_field: sweep.heights,
        field_unit: field_unit(t.system().system().fields.kind).into(),
        curves,
        wavelengths_um: t.system().wavelengths_um.clone(),
        run_info: run,
    })
}
