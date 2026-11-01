use super::*;
use crate::{
    fans::*,
    util::{check_count, default_fields, field_value, linspace},
};
pub fn ray_fan_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &FanSettings,
    ctx: &JobContext,
) -> MkResult<FanResult> {
    check_count("fan samples", s.samples, 2, 100_001)?;
    let aug = default_fields(t, &s.fields)?;
    let t = aug.as_ref();
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let coords = linspace(-1.0, 1.0, s.samples);
    let pupil: Vec<_> = coords
        .iter()
        .map(|&y| [0.0, y])
        .chain(coords.iter().map(|&x| [x, 0.0]))
        .collect();
    let points: Vec<_> = fields.iter().map(|&f| t.field(f).point).collect();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
    let mut g = geo.append_grid(
        &mut p,
        &points,
        &wls,
        &pupil,
        t.aiming(),
        t.system().image_index(),
        true,
        false,
        ctx,
    )?;
    geo.append_chiefs(&mut p, &mut g)?;
    let (q, mask) =
        geo.coordinates(&mut p, &g, 0.0, s.reference == FanReference::SameWavelength)?;
    p.outputs = vec![q, mask];
    let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    let q = b.read_view(r.outputs[0].view())?;
    let q = q.as_slice::<Df32>()?;
    let mask = b.read_view(r.outputs[1].view())?;
    let mask = mask.as_slice::<u32>()?;
    let n = g.hits.rays.count;
    let mut out = Vec::new();
    for (fi, &field) in fields.iter().enumerate() {
        ctx.check()?;
        let mut axes = [Vec::new(), Vec::new()];
        for (wi, &w) in wls.iter().enumerate() {
            for (axis, curves) in axes.iter_mut().enumerate() {
                let component = 1 - axis;
                let start = (fi * wls.len() + wi) * 2 * s.samples + axis * s.samples;
                let val = |k: usize| {
                    (0..s.samples)
                        .map(|i| {
                            if mask[start + i] != 0 {
                                q[k * n + start + i].to_f64()
                            } else {
                                f64::NAN
                            }
                        })
                        .collect()
                };
                curves.push(FanCurve {
                    wavelength: w,
                    pupil: coords.clone(),
                    values: val(component),
                    cross: val(axis),
                });
            }
        }
        let [tangential, sagittal] = axes;
        out.push(FieldFan {
            field,
            field_value: field_value(t, field),
            tangential,
            sagittal,
        });
    }
    ctx.report(1.0, "ray fan");
    Ok(FanResult {
        quantity: FanQuantity::TransverseAberration,
        fields: out,
        wavelengths_um: t.system().wavelengths_um.clone(),
        run_info: stamp(r.run_info, "ray_fan", s, t.warnings().to_vec()),
    })
}

pub fn opd_fan_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &FanSettings,
    ctx: &JobContext,
) -> MkResult<FanResult> {
    check_count("fan samples", s.samples, 2, 100_001)?;
    let aug = default_fields(t, &s.fields)?;
    let t = aug.as_ref();
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let coords = linspace(-1.0, 1.0, s.samples);
    let pupil: Vec<_> = coords
        .iter()
        .map(|&y| [0.0, y])
        .chain(coords.iter().map(|&x| [x, 0.0]))
        .collect();
    let pts: Vec<_> = fields.iter().map(|&f| t.field(f).point).collect();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
    let mut g = geo.append_grid(
        &mut p,
        &pts,
        &wls,
        &pupil,
        t.aiming(),
        t.system().image_index(),
        true,
        false,
        ctx,
    )?;
    geo.append_chiefs(&mut p, &mut g)?;
    let chiefs = g.chiefs()?.clone();
    let n = g.hits.rays.count;
    let map = geo.chief_map(&mut p, &g, true)?;
    let reference = geo.import::<Df32>(&mut p, [chiefs.rays.count, 3], |out| {
        for i in 0..chiefs.rays.count {
            let w = g.chief_wavelengths[i % g.chief_wavelengths.len()];
            let z = t.paraxial().at(w).exit_pupil_position;
            out[3 * i] = if z.is_finite() {
                Df32::from_f64(z)
            } else {
                Df32::from_f32(z as f32)
            };
            out[3 * i + 1] = Df32::from_f64(t.system().n_before(t.system().image_index(), w));
            out[3 * i + 2] = Df32::from_f64(t.system().wavelengths_um[w]);
        }
        Ok(())
    })?;
    let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
    let status = p.scratch(BufferDesc::new(ElementType::U32, [n]));
    p.operations
        .push(ResidentOp::Geometry(mklight_gpu::geometry::Request::Opd(
            mklight_gpu::geometry::Opd {
                axial: false,
                hits: g.hits,
                chiefs,
                chief_index: p.view(map)?,
                reference: p.view(reference)?,
                output,
                status,
            },
        )));
    p.outputs = vec![output];
    let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    let v = b.read_view(r.outputs[0].view())?;
    let v = v.as_slice::<Df32>()?;
    let mut out = Vec::new();
    let mut warnings = t.warnings().to_vec();
    for (fi, &field) in fields.iter().enumerate() {
        let mut axes = [Vec::new(), Vec::new()];
        for (wi, &w) in wls.iter().enumerate() {
            let start = (fi * wls.len() + wi) * pupil.len();
            if v[start..start + pupil.len()].iter().all(|v| v.hi.is_nan()) {
                warnings.push(format!(
                    "field {field}, wavelength {w}: no OPD (chief ray or reference sphere failed)"
                ));
            }
            for axis in 0..2 {
                axes[axis].push(FanCurve {
                    wavelength: w,
                    pupil: coords.clone(),
                    values: v[start + axis * s.samples..start + (axis + 1) * s.samples]
                        .iter()
                        .map(|v| v.to_f64())
                        .collect(),
                    cross: Vec::new(),
                });
            }
        }
        let [tangential, sagittal] = axes;
        out.push(FieldFan {
            field,
            field_value: field_value(t, field),
            tangential,
            sagittal,
        });
    }
    ctx.report(1.0, "OPD fan");
    Ok(FanResult {
        quantity: FanQuantity::Opd,
        fields: out,
        wavelengths_um: t.system().wavelengths_um.clone(),
        run_info: stamp(r.run_info, "opd_fan", s, warnings),
    })
}
