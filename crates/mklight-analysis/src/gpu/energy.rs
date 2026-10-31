use super::*;
use crate::{
    energy::*,
    spot::SpotReference,
    util::{curved_image_warning, default_fields, field_value, linspace, pupil_points},
};
use mklight_gpu::{
    ReduceMode,
    collectives::{self, Scan, SegmentKind, SegmentReduce, StableSort},
    geometry::{self, Characteristic, Distribution, Radial},
};
fn setup<'a>(
    b: &'a dyn ComputeBackend,
    t: &'a SequentialTracer,
    s: &RaySampling,
    delta: f64,
    ctx: &JobContext,
) -> MkResult<(
    Geometry<'a>,
    ResidentPlan,
    ResidentGrid,
    BufferId,
    BufferId,
    BufferId,
    Vec<usize>,
)> {
    crate::spot::validate_sampling(s.pattern, s.rings)?;
    crate::util::check_finite("defocus", delta)?;
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let points: Vec<_> = fields.iter().map(|&i| t.field(i).point).collect();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.seed = Some(s.seed);
    let g = geo.append_grid(
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
    let (q, m) = geo.coordinates(&mut p, &g, delta, false)?;
    let moments = geo.moments(&mut p, q, m, g.pupil_count * wls.len(), true)?;
    Ok((geo, p, g, q, m, moments, fields))
}
fn evidence(run: &mut RunInfo, r: &PlanResult) {
    run.duration_ms += r.run_info.duration_ms;
    run.settings["extra_plans"]
        .as_array_mut()
        .unwrap()
        .push(r.run_info.settings.clone());
}
fn begin(mut run: RunInfo) -> RunInfo {
    run.settings["extra_plans"] = serde_json::json!([]);
    run
}
fn warnings(t: &SequentialTracer, delta: f64) -> Vec<String> {
    let mut w = t.warnings().to_vec();
    if delta != 0.0 {
        w.extend(curved_image_warning(t));
    }
    w
}
fn part(h: &BufferHandle, start: usize, n: usize) -> BufferView {
    BufferView {
        buffer: h.clone(),
        offset: start,
        stride: 1,
        desc: BufferDesc::new(h.desc().element, [n]),
    }
}

pub fn encircled_energy_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &EncircledEnergySettings,
    ctx: &JobContext,
) -> MkResult<EncircledEnergyResult> {
    crate::energy::validate_ee(s)?;
    if s.max_radius.is_some_and(|r| r <= 0.0 || !r.is_finite()) {
        return Err(mklight_core::MkError::InvalidModel(
            "maximum radius must be positive".into(),
        ));
    }
    let augmented = default_fields(t, &s.sampling.fields)?;
    let t = augmented.as_ref();
    let (geo, mut p, g, q, mask, moments, fields) = setup(b, t, &s.sampling, s.defocus, ctx)?;
    p.settings = serde_json::to_value(s).unwrap();
    let n = g.hits.rays.count;
    let len = g.pupil_count * g.wavelengths.len();
    let radius = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
    let square = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
    let weight = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
    let ids = p.scratch(BufferDesc::new(ElementType::U32, [n]));
    p.operations
        .push(ResidentOp::Geometry(geometry::Request::Radial(Radial {
            points: p.view(q)?,
            mask: p.view(mask)?,
            moments: p.view(moments)?,
            segment_len: len,
            centroid: s.reference == SpotReference::Centroid,
            radius,
            square,
            weight,
            ids,
        })));
    p.outputs = vec![radius, square, weight, ids, moments];
    let r = geo.run(p, &ctx.subrange(0.0, 0.5))?;
    let v = b.read_view(r.outputs[4].view())?;
    let mom = v.as_slice::<Df32>()?;
    let max = fields
        .iter()
        .enumerate()
        .map(|(i, _)| {
            mom[i * 10
                + if s.reference == SpotReference::Centroid {
                    9
                } else {
                    8
                }]
            .to_f64()
            .sqrt()
        })
        .fold(0.0, f64::max);
    let radii = linspace(0.0, s.max_radius.unwrap_or(max), s.samples);
    let mut run = begin(r.run_info.clone());
    let mut out = Vec::new();
    for (fi, &field) in fields.iter().enumerate() {
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let rid = geo.import::<Df32>(&mut p, [radii.len()], |out| {
            for (v, &r) in out.iter_mut().zip(&radii) {
                *v = Df32::from_f64(r);
            }
            Ok(())
        })?;
        let mut queries = Vec::new();
        for kind in 0..2 {
            let keys = p.import(part(&r.outputs[kind], fi * len, len));
            let weights = p.import(part(&r.outputs[2], fi * len, len));
            let ids = p.import(part(&r.outputs[3], fi * len, len));
            let sk = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [len]));
            let si = p.scratch(BufferDesc::new(ElementType::U32, [len]));
            let sw = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [len]));
            p.operations
                .push(ResidentOp::Collectives(collectives::Request::StableSort(
                    StableSort {
                        keys: p.view(keys)?,
                        ids: p.view(ids)?,
                        values: Some(p.view(weights)?),
                        output_keys: sk,
                        output_ids: si,
                        output_values: Some(sw),
                    },
                )));
            let cumulative = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [len]));
            p.operations
                .push(ResidentOp::Collectives(collectives::Request::Scan(Scan {
                    values: p.view(sw)?,
                    inclusive: true,
                    output: cumulative,
                })));
            let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [radii.len() + 4]));
            p.operations
                .push(ResidentOp::Geometry(geometry::Request::Distribution(
                    Distribution {
                        keys: p.view(sk)?,
                        cumulative: p.view(cumulative)?,
                        radii: p.view(rid)?,
                        output,
                    },
                )));
            queries.push(output);
        }
        p.outputs = queries;
        let rr = geo.run(
            p,
            &ctx.subrange(
                0.5 + 0.5 * fi as f64 / fields.len() as f64,
                0.5 + 0.5 * (fi + 1) as f64 / fields.len() as f64,
            ),
        )?;
        evidence(&mut run, &rr);
        let c = b.read_view(rr.outputs[0].view())?;
        let q = b.read_view(rr.outputs[1].view())?;
        let c = c.as_slice::<Df32>()?;
        let q = q.as_slice::<Df32>()?;
        let k = radii.len();
        out.push(FieldEnergy {
            field,
            field_value: field_value(t, field),
            encircled: c[..k].iter().map(|v| v.to_f64()).collect(),
            ensquared: q[..k].iter().map(|v| v.to_f64()).collect(),
            r50: c[k].to_f64(),
            r80: c[k + 1].to_f64(),
            r90: c[k + 2].to_f64(),
            half_width80: q[k + 1].to_f64(),
            rays_arrived: mom[10 * fi].to_f64() as usize,
        });
    }
    Ok(EncircledEnergyResult {
        radius: radii,
        fields: out,
        wavelengths: g.wavelengths,
        run_info: stamp(run, "encircled_energy", s, warnings(t, s.defocus)),
    })
}

pub fn geometric_mtf_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &GeometricMtfSettings,
    ctx: &JobContext,
) -> MkResult<GeometricMtfResult> {
    crate::energy::validate_mtf(s)?;
    let augmented = default_fields(t, &s.sampling.fields)?;
    let t = augmented.as_ref();
    let (geo, mut p, g, q, mask, moments, fields) = setup(b, t, &s.sampling, s.defocus, ctx)?;
    p.settings = serde_json::to_value(s).unwrap();
    let n = g.hits.rays.count;
    let len = g.pupil_count * g.wavelengths.len();
    p.outputs = vec![q, mask, moments];
    let r = geo.run(p, &ctx.subrange(0.0, 0.3))?;
    let mut run = begin(r.run_info.clone());
    let v = b.read_view(r.outputs[2].view())?;
    let mom = v.as_slice::<Df32>()?;
    let cutoff = 1.0
        / (t.system().wavelengths_um[t.system().primary_wavelength]
            * 1e-3
            * t.paraxial().primary().working_f_number);
    let freqs = linspace(
        0.0,
        s.max_frequency
            .unwrap_or(if cutoff.is_finite() && cutoff > 0.0 {
                cutoff
            } else {
                100.0
            }),
        s.frequency_samples,
    );
    let mut curves: Vec<_> = fields
        .iter()
        .flat_map(|&field| {
            [
                MtfCurve {
                    field,
                    orientation: MtfOrientation::Tangential,
                    modulation: Vec::new(),
                },
                MtfCurve {
                    field,
                    orientation: MtfOrientation::Sagittal,
                    modulation: Vec::new(),
                },
            ]
        })
        .collect();
    for (ki, &freq) in freqs.iter().enumerate() {
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let q = p.import(r.outputs[0].view());
        let mask = p.import(r.outputs[1].view());
        let mut results = Vec::new();
        for axis in 0..2 {
            let frequency = geo.import::<Df32>(&mut p, [fields.len(), 2], |out| {
                for (fi, &field) in fields.iter().enumerate() {
                    let pt = t.field(field).point;
                    let h = pt.hx.hypot(pt.hy);
                    let e = if h > 0.0 {
                        [pt.hx / h, pt.hy / h]
                    } else {
                        [0.0, 1.0]
                    };
                    let e = if axis == 0 { e } else { [e[1], -e[0]] };
                    out[2 * fi] = Df32::from_f64(e[0] * freq);
                    out[2 * fi + 1] = Df32::from_f64(e[1] * freq);
                }
                Ok(())
            })?;
            let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, n]));
            p.operations
                .push(ResidentOp::Geometry(geometry::Request::Characteristic(
                    Characteristic {
                        points: p.view(q)?,
                        mask: p.view(mask)?,
                        segment_len: len,
                        frequency: p.view(frequency)?,
                        output: terms,
                    },
                )));
            let offsets = geo.import::<u32>(&mut p, [fields.len() + 1], |out| {
                for (i, v) in out.iter_mut().enumerate() {
                    *v = (i * len) as u32;
                }
                Ok(())
            })?;
            let sum = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, fields.len()]));
            let status = p.scratch(BufferDesc::new(ElementType::U32, [fields.len()]));
            p.operations.push(ResidentOp::Collectives(
                collectives::Request::SegmentReduce(SegmentReduce {
                    values: p.view(terms)?,
                    offsets: p.view(offsets)?,
                    mask: None,
                    mask_value: 0,
                    weights: None,
                    max_segment_len: len,
                    reference: None,
                    mode: ReduceMode::DoubleFloat,
                    kind: SegmentKind::Sum,
                    output: sum,
                    status,
                }),
            ));
            results.push(sum);
        }
        p.outputs = results;
        let rr = geo.run(
            p,
            &ctx.subrange(
                0.3 + 0.7 * ki as f64 / freqs.len() as f64,
                0.3 + 0.7 * (ki + 1) as f64 / freqs.len() as f64,
            ),
        )?;
        evidence(&mut run, &rr);
        for axis in 0..2 {
            let v = b.read_view(rr.outputs[axis].view())?;
            let v = v.as_slice::<Df32>()?;
            for fi in 0..fields.len() {
                let w = mom[fi * 10 + 1].to_f64();
                curves[2 * fi + axis].modulation.push(if w > 0.0 {
                    v[fi].to_f64().hypot(v[fields.len() + fi].to_f64()) / w
                } else {
                    f64::NAN
                });
            }
        }
    }
    Ok(GeometricMtfResult {
        frequencies: freqs,
        curves,
        diffraction_limit: None,
        cutoff_frequency: cutoff,
        field_values: fields.iter().map(|&f| field_value(t, f)).collect(),
        run_info: stamp(run, "geometric_mtf", s, warnings(t, s.defocus)),
    })
}
