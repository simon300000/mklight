use super::*;
use crate::{field_maps::*, util::wavelength_or_primary};
use mklight_gpu::geometry::{self, FieldData};
pub fn field_map_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    kind: FieldAnalysis,
    s: &FieldMapSettings,
    ctx: &JobContext,
) -> MkResult<FieldMapResult> {
    validate_map(s)?;
    let wl = wavelength_or_primary(t, s.wavelength)?;
    let (shape, coords) = map_coordinates(s);
    if matches!(kind, FieldAnalysis::RelativeIllumination) {
        let inside: Vec<_> = coords
            .iter()
            .copied()
            .filter(|p| p[0].hypot(p[1]) <= 1.0 + 1e-12)
            .collect();
        let (rows, omega0, images, run) = super::illumination::illumination_rows(
            b,
            t,
            &inside,
            wl,
            s.radial_samples,
            s.azimuth_samples,
            &ctx.subrange(0.0, 0.95),
        )?;
        let mut ri = rows.iter().zip(images);
        let mut image_mm = Vec::new();
        let mut values = Vec::new();
        for p in &coords {
            if p[0].hypot(p[1]) > 1.0 + 1e-12 {
                image_mm.push([f64::NAN; 2]);
                values.extend([f64::NAN; 3]);
            } else {
                let (r, im) = ri.next().unwrap();
                image_mm.push(im);
                values.extend([r[0] / omega0, r[0], 1.0 - r[1]]);
            }
        }
        ctx.check()?;
        ctx.report(1.0, kind.name());
        return Ok(FieldMapResult {
            shape,
            wavelength: wl,
            columns: kind.columns(),
            field_coordinates: coords,
            image_mm,
            values,
            run_info: stamp(run, kind.name(), s, t.warnings().to_vec()),
        });
    }
    let (selected, name) = super::chief_maps::ChiefDesign::selected();
    let design = selected.resolve(coords.len());
    if design == super::chief_maps::ChiefDesign::Cpu {
        // The CPU-only division: the reference path from the GPU route.
        let mut r = crate::field_maps::field_map_job(t, kind, s, ctx)?;
        r.run_info.settings["fieldmaps_variant"] = serde_json::json!(name);
        return Ok(r);
    }
    let device_chiefs = design == super::chief_maps::ChiefDesign::DeviceChief;
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
    let mut points: Vec<_> = coords
        .iter()
        .map(|&[x, y]| FieldPoint::normalized(x, y))
        .collect();
    points.push(FieldPoint::normalized(0.0, 0.0));
    let real = mklight_core::RayAiming::Real {
        tolerance: match t.aiming() {
            mklight_core::RayAiming::Real { tolerance } => tolerance,
            _ => 1e-12,
        },
    };
    let hits = geo.append_chief_grid(
        &mut p,
        &points,
        wl,
        &[[0.0; 2]],
        real,
        t.system().image_index(),
        true,
        device_chiefs,
        ctx,
    )?;
    let angles = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [points.len(), 5]));
    p.operations
        .push(ResidentOp::Geometry(geometry::Request::FieldData(
            FieldData {
                record_slots: t.system().num_surfaces(),
                hits,
                kind: 0,
                pupil_step: Df32::ZERO,
                output: angles,
            },
        )));
    let mut outputs = vec![angles];
    if matches!(kind, FieldAnalysis::PupilAberration) {
        let hits = geo.append_chief_grid(
            &mut p,
            &points,
            wl,
            &[
                [0.0, 0.0],
                [1e-4, 0.0],
                [-1e-4, 0.0],
                [0.0, 1e-4],
                [0.0, -1e-4],
            ],
            mklight_core::RayAiming::Off,
            t.system().stop,
            false,
            device_chiefs,
            ctx,
        )?;
        let out = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [points.len(), 5]));
        p.operations
            .push(ResidentOp::Geometry(geometry::Request::FieldData(
                FieldData {
                    record_slots: t.system().num_surfaces(),
                    hits,
                    kind: 1,
                    pupil_step: if t.pupil().r_ep.is_finite() {
                        Df32::from_f64(2e-4 * t.pupil().r_ep)
                    } else {
                        Df32::from_f32(f32::INFINITY)
                    },
                    output: out,
                },
            )));
        outputs.push(out);
    }
    p.outputs = outputs;
    let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    let v = b.read_view(r.outputs[0].view())?;
    let a = v.as_slice::<Df32>()?;
    let images: Vec<_> = (0..coords.len())
        .map(|i| [a[5 * i].to_f64(), a[5 * i + 1].to_f64()])
        .collect();
    if device_chiefs && images.iter().any(|p| !p[0].is_finite()) {
        // A device chief solve that fails is completed by the reference:
        // the whole map, so a design never returns fewer chiefs.
        let failed = images.iter().filter(|p| !p[0].is_finite()).count();
        let mut out = crate::field_maps::field_map_job(t, kind, s, ctx)?;
        out.run_info.settings["fieldmaps_variant"] = serde_json::json!(name);
        out.run_info.settings["device_chief_failures"] = serde_json::json!(failed);
        out.run_info.warnings.push(format!(
            "{failed} device chief solves failed; the map was completed by the reference"
        ));
        return Ok(out);
    }
    let values = match kind {
        FieldAnalysis::ChiefRayAngle => (0..coords.len())
            .flat_map(|i| {
                let angle = a[5 * i + 2].to_f64();
                let p = images[i];
                let limit = interpolate_limit(&s.sensor_limit, p[0].hypot(p[1]));
                [
                    angle,
                    a[5 * i + 3].to_f64(),
                    a[5 * i + 4].to_f64(),
                    limit,
                    limit - angle,
                ]
            })
            .collect(),
        FieldAnalysis::PupilAberration => {
            let v = b.read_view(r.outputs[1].view())?;
            let v = v.as_slice::<Df32>()?;
            let a = &v[5 * coords.len()..];
            (0..coords.len())
                .flat_map(|i| {
                    let p = &v[5 * i..];
                    let mx = p[2].to_f64();
                    let my = p[3].to_f64();
                    [
                        p[0].to_f64(),
                        p[1].to_f64(),
                        mx,
                        my,
                        mx / a[2].to_f64() - 1.0,
                        my / a[3].to_f64() - 1.0,
                    ]
                })
                .collect()
        }
        FieldAnalysis::RelativeIllumination => unreachable!(),
    };
    let mut warnings = t.warnings().to_vec();
    for (i, p) in images.iter().enumerate() {
        if !p[0].is_finite() {
            warnings.push(format!(
                "field ({},{}): chief ray failed",
                coords[i][0], coords[i][1]
            ));
        }
    }
    ctx.check()?;
    ctx.report(1.0, kind.name());
    Ok(FieldMapResult {
        shape,
        wavelength: wl,
        columns: kind.columns(),
        field_coordinates: coords,
        image_mm: images,
        values,
        run_info: {
            let mut run = stamp(r.run_info, kind.name(), s, warnings);
            run.settings["resident_execution"]["fieldmaps_variant"] = serde_json::json!(name);
            run
        },
    })
}
