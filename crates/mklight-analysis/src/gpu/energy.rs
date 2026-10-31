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
    p.seed = matches!(
        s.pattern,
        crate::util::PupilPattern::Random | crate::util::PupilPattern::Dithered
    )
    .then_some(s.seed);
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
    encircled_energy_with_uncertainty_job(b, t, s, b.precision().budget.position_mm, ctx)
}
/// A caller-qualified position error bound, including its safety margin, may
/// replace the backend-wide budget for this exact snapshot/sampling domain.
/// The bound and every resulting reference decision are retained in RunInfo.
pub fn encircled_energy_with_uncertainty_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &EncircledEnergySettings,
    position_uncertainty_mm: f64,
    ctx: &JobContext,
) -> MkResult<EncircledEnergyResult> {
    if !position_uncertainty_mm.is_finite() || position_uncertainty_mm < 0.0 {
        return Err(mklight_core::MkError::InvalidParam(
            "position uncertainty must be finite and nonnegative".into(),
        ));
    }
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
    p.settings["position_uncertainty_mm"] = serde_json::json!(position_uncertainty_mm);
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
    let mut ambiguous = Vec::new();
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
        let mut boundaries = Vec::new();
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
            let boundary = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [radii.len()]));
            p.operations
                .push(ResidentOp::Geometry(geometry::Request::Distribution(
                    Distribution {
                        uncertainty: Df32::from_f64(
                            2.0 * std::f64::consts::SQRT_2 * position_uncertainty_mm,
                        ),
                        boundary,
                        keys: p.view(sk)?,
                        cumulative: p.view(cumulative)?,
                        radii: p.view(rid)?,
                        output,
                    },
                )));
            queries.push(output);
            boundaries.push(boundary);
        }
        queries.extend(boundaries);
        p.outputs = queries;
        let rr = geo.run(
            p,
            &ctx.subrange(
                0.5 + 0.4 * fi as f64 / fields.len() as f64,
                0.5 + 0.4 * (fi + 1) as f64 / fields.len() as f64,
            ),
        )?;
        evidence(&mut run, &rr);
        let mut uncertain = false;
        for h in &rr.outputs[2..] {
            let view = b.read_view(h.view())?;
            uncertain |= view.as_slice::<Df32>()?.iter().any(|v| v.to_f64() > 2e-3);
        }
        if uncertain {
            ambiguous.push(field);
        }
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
    if !ambiguous.is_empty() {
        ctx.check()?;
        // The geometry error budget can straddle a discrete CDF step. Keep
        // the reference's exact bins/automatic radius when that step exceeds
        // the energy budget; never silently move or widen a requested bin.
        let mut reference = crate::energy::encircled_energy_job(t, s, &ctx.subrange(0.9, 0.99))?;
        reference.run_info.duration_ms += run.duration_ms;
        reference.run_info.started_unix_ms = run.started_unix_ms;
        reference.run_info.settings["resident_attempt"] = run.settings;
        reference.run_info.settings["adapter_fallbacks"] = serde_json::json!({"stage":"energy_bin_classification","count":1,"trigger_fields":ambiguous,"pupil_rays":n,"requested_backend":b.info().backend_id(),"reason":"geometry uncertainty straddles more than 0.002 of arrived energy"});
        reference.run_info.warnings.push(format!("encircled energy used the f64 reference: geometry uncertainty straddles more than 0.002 of arrived energy in field(s) {ambiguous:?}"));
        ctx.check()?;
        ctx.report(1.0, "encircled energy");
        return Ok(reference);
    }
    ctx.check()?;
    ctx.report(1.0, "encircled energy");
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
    let tile = ((1 << 20) / (2 * n).max(1)).clamp(1, freqs.len());
    for start in (0..freqs.len()).step_by(tile) {
        let end = (start + tile).min(freqs.len());
        let batch = 2 * (end - start);
        let segments = batch * fields.len();
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let q = p.import(r.outputs[0].view());
        let mask = p.import(r.outputs[1].view());
        let frequency = geo.import::<Df32>(&mut p, [batch, fields.len(), 2], |out| {
            for k in 0..batch {
                let axis = k % 2;
                let freq = freqs[start + k / 2];
                for (fi, &field) in fields.iter().enumerate() {
                    let pt = t.field(field).point;
                    let h = pt.hx.hypot(pt.hy);
                    let e = if h > 0.0 {
                        [pt.hx / h, pt.hy / h]
                    } else {
                        [0.0, 1.0]
                    };
                    let e = if axis == 0 { e } else { [e[1], -e[0]] };
                    let i = 2 * (k * fields.len() + fi);
                    out[i] = Df32::from_f64(e[0] * freq);
                    out[i + 1] = Df32::from_f64(e[1] * freq);
                }
            }
            Ok(())
        })?;
        let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, batch * n]));
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
        let offsets = geo.import::<u32>(&mut p, [segments + 1], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i * len) as u32;
            }
            Ok(())
        })?;
        let sum = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, segments]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [segments]));
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
        p.outputs = vec![sum];
        let rr = geo.run(
            p,
            &ctx.subrange(
                0.3 + 0.7 * start as f64 / freqs.len() as f64,
                0.3 + 0.7 * end as f64 / freqs.len() as f64,
            ),
        )?;
        evidence(&mut run, &rr);
        let v = b.read_view(rr.outputs[0].view())?;
        let v = v.as_slice::<Df32>()?;
        for k in 0..batch {
            for fi in 0..fields.len() {
                let i = k * fields.len() + fi;
                let w = mom[fi * 10 + 1].to_f64();
                curves[2 * fi + k % 2].modulation.push(if w > 0.0 {
                    v[i].to_f64().hypot(v[segments + i].to_f64()) / w
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

/// Brent stays on the host; every trial reprojects and reduces resident
/// image states, including the two-pass evaluation near a sharp minimum.
pub fn best_focus_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &BestFocusSettings,
    ctx: &JobContext,
) -> MkResult<BestFocusResult> {
    crate::energy::validate_best_focus(s)?;
    let aug = default_fields(t, &s.sampling.fields)?;
    let t = aug.as_ref();
    let (geo, mut p, g, q, mask, m0, fields) = setup(b, t, &s.sampling, 0.0, ctx)?;
    p.settings = serde_json::to_value(s).unwrap();
    let len = g.pupil_count * g.wavelengths.len();
    let n = g.hits.rays.count;
    let slope = geo.moments_at(&mut p, q, mask, len, true, 2)?;
    let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3, n]));
    p.operations
        .push(ResidentOp::Geometry(geometry::Request::FocusTerms(
            geometry::FocusTerms {
                points: p.view(q)?,
                mask: p.view(mask)?,
                position_moments: p.view(m0)?,
                slope_moments: p.view(slope)?,
                segment_len: len,
                centroid: s.reference == SpotReference::Centroid,
                output: terms,
            },
        )));
    let offsets = geo.import::<u32>(&mut p, [fields.len() + 1], |out| {
        for (i, v) in out.iter_mut().enumerate() {
            *v = (i * len) as u32;
        }
        Ok(())
    })?;
    let quad = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3, fields.len()]));
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
            output: quad,
            status,
        }),
    ));
    p.outputs = vec![
        g.hits.rays.floats,
        g.hits.status,
        g.chiefs.rays.floats,
        g.chiefs.status,
        m0,
        quad,
    ];
    let r = geo.run(p, &ctx.subrange(0.0, 0.2))?;
    let mut run = begin(r.run_info.clone());
    let views: Vec<_> = r.outputs[4..]
        .iter()
        .map(|h| b.read_view(h.view()))
        .collect::<MkResult<_>>()?;
    let initial: Vec<_> = views
        .iter()
        .map(|v| v.as_slice::<Df32>())
        .collect::<MkResult<_>>()?;
    let cstatus = b.read_view(r.outputs[3].view())?;
    let cstatus = cstatus.as_slice::<u32>()?;
    let mut used = Vec::new();
    let mut warnings = t.warnings().to_vec();
    warnings.extend(curved_image_warning(t));
    let sq = |v: &[Df32]| {
        let w = v[1].to_f64();
        if s.reference == SpotReference::Centroid {
            (v[5].to_f64() + v[7].to_f64()) / w
        } else {
            v[4].to_f64() / w
        }
    };
    for (fi, &f) in fields.iter().enumerate() {
        let ci = fi * g.chief_wavelengths.len()
            + g.chief_wavelengths
                .iter()
                .position(|&w| w == t.system().primary_wavelength)
                .unwrap();
        if s.reference == SpotReference::Chief && cstatus[ci] != 0 {
            warnings.push(format!(
                "field {f}: primary chief ray failed; skipped (chief reference)"
            ));
            continue;
        }
        if initial[0][10 * fi + 1].to_f64() <= 0.0 {
            warnings.push(format!("field {f}: no rays arrived"));
            continue;
        }
        let weight_sum = initial[0][10 * fi + 1].to_f64();
        let bb = initial[1][fields.len() + fi].to_f64() / weight_sum;
        let cc = initial[1][2 * fields.len() + fi].to_f64() / weight_sum;
        let weight = t
            .system()
            .system()
            .fields
            .fields
            .get(f)
            .map_or(1.0, |f| f.weight);
        used.push((fi, f, weight, bb, cc));
    }
    let sw: f64 = used.iter().map(|u| u.2).sum();
    if used.is_empty() || sw <= 0.0 {
        return Err(mklight_core::MkError::Numerical(
            "best focus: no field with arrived rays and positive weight".into(),
        ));
    }
    let num: f64 = used.iter().map(|u| u.2 * u.3).sum();
    let den: f64 = used.iter().map(|u| u.2 * u.4).sum();
    let ms = if den > 0.0 { -num / den } else { 0.0 };
    let range = s.range.unwrap_or_else(|| {
        let mut lo = ms;
        let mut hi = ms;
        for u in &used {
            if u.4 > 0.0 && u.2 > 0.0 {
                let x = -u.3 / u.4;
                lo = lo.min(x);
                hi = hi.max(x);
            }
        }
        let pad = (0.05 * (hi - lo)).max(1e-6 * (1.0 + lo.abs().max(hi.abs())));
        [lo - pad, hi + pad]
    });
    let mut calls = 0;
    let mut evaluate = |d: f64| -> MkResult<Vec<f64>> {
        ctx.check()?;
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let mut g = g.clone();
        g.hits.rays.floats = p.import(r.outputs[0].view());
        g.hits.status = p.import(r.outputs[1].view());
        g.chiefs.rays.floats = p.import(r.outputs[2].view());
        g.chiefs.status = p.import(r.outputs[3].view());
        let (q, m) = geo.coordinates(&mut p, &g, d, false)?;
        let m = geo.moments(&mut p, q, m, len, true)?;
        p.outputs = vec![m];
        // A trial's internal progress cannot advance the whole search to 1.
        let fraction = 0.2 + 0.7 * (1.0 - 1.0 / (calls as f64 + 1.0));
        let rr = geo.run(p, &ctx.subrange(fraction, fraction))?;
        evidence(&mut run, &rr);
        calls += 1;
        let v = b.read_view(rr.outputs[0].view())?;
        let v = v.as_slice::<Df32>()?;
        Ok(used
            .iter()
            .map(|u| sq(&v[10 * u.0..]).max(0.0).sqrt())
            .collect())
    };
    let weighted = |v: &[f64]| {
        v.iter()
            .zip(&used)
            .map(|(v, u)| v * u.2)
            .collect::<mklight_core::math::KahanSum>()
            .value()
            / sw
    };
    let br = crate::util::brent_minimize(
        |d| Ok(weighted(&evaluate(d)?)),
        range[0],
        range[1],
        1e-12,
        s.tolerance,
        s.max_evaluations,
    )?;
    let field_rms = evaluate(br.x)?;
    let field_rms_at_zero = evaluate(0.0)?;
    let at_zero = weighted(&field_rms_at_zero);
    let tfd = if s.through_focus_samples > 0 {
        linspace(range[0], range[1], s.through_focus_samples.max(2))
    } else {
        Vec::new()
    };
    let mut tfm = Vec::new();
    for &d in &tfd {
        tfm.push(weighted(&evaluate(d)?));
    }
    if (br.x - range[0]).min(range[1] - br.x) <= 2.0 * s.tolerance {
        warnings.push(format!(
            "best focus is at the edge of the {} search range [{}, {}] mm",
            if s.range.is_some() {
                "given"
            } else {
                "automatic"
            },
            range[0],
            range[1]
        ));
    }
    if !br.converged {
        warnings.push(format!(
            "best focus did not converge within {} merit evaluations",
            s.max_evaluations
        ));
    }
    ctx.report(1.0, "best focus");
    Ok(BestFocusResult {
        field_values: used.iter().map(|u| field_value(t, u.1)).collect(),
        defocus: br.x,
        merit: br.fx,
        merit_at_zero: at_zero,
        fields: used.iter().map(|u| u.1).collect(),
        field_rms,
        field_rms_at_zero,
        mean_square_defocus: ms,
        range,
        evaluations: br.evaluations,
        converged: br.converged,
        through_focus_defocus: tfd,
        through_focus_merit: tfm,
        run_info: stamp(run, "best_focus", s, warnings),
    })
}
