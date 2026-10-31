use super::*;
use crate::{
    seidel::*,
    util::{check_count, field_scale, field_sweep, field_unit, linspace},
};
pub fn lateral_colour_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &LateralColourSettings,
    ctx: &JobContext,
) -> MkResult<LateralColourResult> {
    check_count("field samples", s.field_samples, 2, 100_001)?;
    let (ws, wl) = select_short_long(t.system(), s.short_wavelength, s.long_wavelength)?;
    let sweep = field_sweep(t, s.field_samples, s.field_axes);
    let fields: Vec<_> = sweep
        .coordinates
        .iter()
        .map(|&[x, y]| FieldPoint::normalized(x, y))
        .collect();
    let wls: Vec<_> = (0..t.num_wavelengths()).collect();
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
        false,
        ctx,
    )?;
    let (q, mask) = geo.coordinates(&mut p, &g, 0.0, false)?;
    p.outputs = vec![q, mask];
    let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    let v = b.read_view(r.outputs[0].view())?;
    let v = v.as_slice::<Df32>()?;
    let mask = b.read_view(r.outputs[1].view())?;
    let mask = mask.as_slice::<u32>()?;
    let n = g.hits.rays.count;
    let nw = wls.len();
    let offsets: Vec<Vec<_>> = (0..fields.len())
        .map(|f| {
            (0..nw)
                .map(|w| {
                    let i = f * nw + w;
                    if mask[i] != 0 {
                        v[sweep.components[f] * n + i].to_f64()
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect();
    ctx.report(1.0, "lateral colour");
    Ok(LateralColourResult {
        field_values: sweep.heights.iter().map(|h| h * field_scale(t)).collect(),
        normalized_field: sweep.heights,
        field_coordinates: sweep.coordinates,
        field_unit: field_unit(t.system().system().fields.kind).into(),
        short_wavelength: ws,
        long_wavelength: wl,
        lateral_colour: offsets.iter().map(|r| r[ws] - r[wl]).collect(),
        curves: wls
            .iter()
            .map(|&w| LateralColourCurve {
                wavelength: w,
                wavelength_um: t.system().wavelengths_um[w],
                offset: offsets.iter().map(|r| r[w]).collect(),
            })
            .collect(),
        airy_radius: 1.22
            * t.system().wavelengths_um[t.system().primary_wavelength]
            * 1e-3
            * t.paraxial().primary().working_f_number,
        run_info: stamp(r.run_info, "lateral_colour", s, t.warnings().to_vec()),
    })
}
pub fn longitudinal_aberration_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &LongitudinalSettings,
    ctx: &JobContext,
) -> MkResult<LongitudinalResult> {
    use mklight_core::surface_math::{Symmetry, system_symmetry};
    use mklight_trace::paraxial::{ParaxialAxis, first_order_axis};
    check_count("pupil samples", s.pupil_samples, 2, 100_001)?;
    let symmetry = system_symmetry(t.system().system());
    let rho = linspace(
        if symmetry == Symmetry::Rotational {
            0.0
        } else {
            -1.0
        },
        1.0,
        s.pupil_samples,
    );
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
    let axes = s.axes.components(symmetry);
    let pupil: Vec<_> = axes
        .iter()
        .flat_map(|&a| {
            rho.iter()
                .map(move |&r| if a == 0 { [r, 0.0] } else { [0.0, r] })
        })
        .collect();
    let g = geo.append_grid(
        &mut p,
        &[FieldPoint::normalized(0.0, 0.0)],
        &wls,
        &pupil,
        t.aiming(),
        t.system().image_index(),
        false,
        false,
        ctx,
    )?;
    let n = g.hits.rays.count;
    let index = geo.import::<u32>(&mut p, [n], |out| {
        for (i, v) in out.iter_mut().enumerate() {
            let w = wls[i / pupil.len()];
            *v = g.chief_wavelengths.iter().position(|&v| v == w).unwrap() as u32;
        }
        Ok(())
    })?;
    let reference = geo.import::<Df32>(&mut p, [g.chiefs.rays.count, 3], |out| {
        out.fill(Df32::ZERO);
        Ok(())
    })?;
    let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
    let status = p.scratch(BufferDesc::new(ElementType::U32, [n]));
    p.operations
        .push(ResidentOp::Geometry(mklight_gpu::geometry::Request::Opd(
            mklight_gpu::geometry::Opd {
                axial: true,
                hits: g.hits,
                chiefs: g.chiefs,
                chief_index: p.view(index)?,
                reference: p.view(reference)?,
                output,
                status,
            },
        )));
    p.outputs = vec![output];
    let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    let v = b.read_view(r.outputs[0].view())?;
    let v = v.as_slice::<Df32>()?;
    let mut curves = Vec::new();
    for (ai, &a) in axes.iter().enumerate() {
        let axis = [ParaxialAxis::X, ParaxialAxis::Y][a];
        let primary = paraxial_focus(&first_order_axis(
            t.system(),
            t.system().primary_wavelength,
            axis,
        )?);
        for (wi, &w) in wls.iter().enumerate() {
            let pf = paraxial_focus(&first_order_axis(t.system(), w, axis)?);
            let focus: Vec<_> = rho
                .iter()
                .enumerate()
                .map(|(i, &rho)| {
                    if rho == 0.0 {
                        pf
                    } else {
                        v[wi * pupil.len() + ai * s.pupil_samples + i].to_f64()
                    }
                })
                .collect();
            curves.push(LongitudinalCurve {
                axis,
                wavelength: w,
                wavelength_um: t.system().wavelengths_um[w],
                paraxial_focus: pf,
                aberration: focus.iter().map(|f| f - primary).collect(),
                focus,
            });
        }
    }
    ctx.report(1.0, "longitudinal aberration");
    Ok(LongitudinalResult {
        pupil: rho,
        primary_paraxial_focus: paraxial_focus(t.paraxial().primary()),
        curves,
        run_info: stamp(
            r.run_info,
            "longitudinal_aberration",
            s,
            t.warnings().to_vec(),
        ),
    })
}
