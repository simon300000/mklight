use super::*;
use crate::{
    energy::*,
    spot::SpotReference,
    util::{
        curved_image_warning, default_fields, diffraction_cutoff, field_value, linspace,
        pupil_points, seed_of,
    },
};
use mklight_gpu::{
    ReduceMode,
    collectives::{self, SegmentKind, SegmentReduce},
    geometry::{self, Characteristic},
};
/// Traced grid with chief-relative image coordinates and their moments.
struct Setup<'a> {
    geo: Geometry<'a>,
    plan: ResidentPlan,
    grid: ResidentGrid,
    points: BufferId,
    mask: BufferId,
    moments: BufferId,
    fields: Vec<usize>,
}
fn setup<'a>(
    b: &'a dyn ComputeBackend,
    t: &'a SequentialTracer,
    s: &RaySampling,
    delta: f64,
    route: &Route,
    ctx: &JobContext,
) -> MkResult<Setup<'a>> {
    crate::spot::validate_sampling(s.rings)?;
    crate::util::check_finite("defocus", delta)?;
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let points: Vec<_> = fields.iter().map(|&i| t.field(i).point).collect();
    let geo = Geometry::new(b, t, ctx)?.with_route(route)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.seed = seed_of(s.pattern, s.seed);
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
    geo.append_chiefs(&mut p, &mut g, ctx)?;
    let (q, m) = geo.coordinates(&mut p, &g, delta, false)?;
    let moments = geo.moments(&mut p, q, m, g.pupil_count * wls.len(), true)?;
    Ok(Setup {
        geo,
        plan: p,
        grid: g,
        points: q,
        mask: m,
        moments,
        fields,
    })
}
fn warnings(t: &SequentialTracer, delta: f64) -> Vec<String> {
    let mut w = t.warnings().to_vec();
    if delta != 0.0 {
        w.extend(curved_image_warning(t));
    }
    w
}

pub fn geometric_mtf_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &GeometricMtfSettings,
    route: &Route,
    ctx: &JobContext,
) -> MkResult<GeometricMtfResult> {
    validate_mtf_job(s)?;
    let augmented = default_fields(t, &s.sampling.fields)?;
    Geometry::new(b, augmented.as_ref(), ctx)?
        .with_route(route)?
        .split_large_generations()
        .geometric_mtf_job(s, ctx)
}
fn validate_mtf_job(s: &GeometricMtfSettings) -> MkResult<()> {
    crate::energy::validate_mtf(s)?;
    crate::spot::validate_sampling(s.sampling.rings)
}
impl Geometry<'_> {
    /// [`geometric_mtf_job`] on this adapter: the fields of its tracer (plus
    /// the default axis fields), with its aiming and thresholds.
    pub fn geometric_mtf_job(
        &self,
        s: &GeometricMtfSettings,
        ctx: &JobContext,
    ) -> MkResult<GeometricMtfResult> {
        validate_mtf_job(s)?;
        let geo = self;
        let augmented = default_fields(self.tracer, &s.sampling.fields)?;
        let t = augmented.as_ref();
        let fields = s.sampling.fields.resolve(t.fields().len())?;
        let wls = s
            .sampling
            .wavelengths
            .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
        let pupil = pupil_points(s.sampling.pattern, s.sampling.rings, s.sampling.seed);
        let seed = seed_of(s.sampling.pattern, s.sampling.seed);
        let cutoff = diffraction_cutoff(
            t.system().wavelengths_um[t.system().primary_wavelength],
            t.paraxial().primary().working_f_number,
        );
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
        let points: Vec<_> = fields.iter().map(|&i| t.field(i).point).collect();
        let groups = geo.field_groups(fields.len(), wls.len() * pupil.len());
        let len = pupil.len() * wls.len();
        // Each group's grid plan is submitted once its rays are aimed; its
        // characteristic tiles follow once its points exist, while the host
        // aims the next group.
        let mut run: Option<RunInfo> = None;
        let mut tiles: std::collections::VecDeque<MtfTile> = Default::default();
        let mut previous: Option<(usize, std::ops::Range<usize>, plans::Submitted)> = None;
        // Each group's slice of the progress: its grid, then its tiles.
        let progress = |k: usize, from: f64, to: f64| {
            ctx.subrange(
                k as f64 / groups.len() as f64,
                (k + 1) as f64 / groups.len() as f64,
            )
            .subrange(from, to)
        };
        for (k, range) in groups.iter().enumerate() {
            let sub = progress(k, 0.0, 0.9);
            let mut p = ResidentPlan::new(t.system().snapshot.clone());
            p.seed = seed;
            p.settings = serde_json::to_value(s).unwrap();
            let mut g = geo.append_grid(
                &mut p,
                &points[range.clone()],
                &wls,
                &pupil,
                t.aiming(),
                t.system().image_index(),
                true,
                false,
                &sub,
            )?;
            geo.append_chiefs(&mut p, &mut g, &sub)?;
            let (q, m) = geo.coordinates(&mut p, &g, s.defocus, false)?;
            let moments = geo.moments(&mut p, q, m, len, true)?;
            p.outputs = vec![q, m, moments];
            let submitted = geo.submit(p, &sub)?;
            if let Some((k, range, sub)) = previous.take() {
                let r = geo.finish(sub)?;
                let info = r.run_info.clone();
                plans::merge_run(&mut run, info);
                mtf_tiles(
                    geo,
                    t,
                    &fields,
                    range,
                    &r,
                    &freqs,
                    len,
                    &mut tiles,
                    &mut curves,
                    &mut run,
                    &progress(k, 0.9, 1.0),
                )?;
            }
            previous = Some((k, range.clone(), submitted));
        }
        if let Some((k, range, sub)) = previous.take() {
            let r = geo.finish(sub)?;
            let info = r.run_info.clone();
            plans::merge_run(&mut run, info);
            mtf_tiles(
                geo,
                t,
                &fields,
                range,
                &r,
                &freqs,
                len,
                &mut tiles,
                &mut curves,
                &mut run,
                &progress(k, 0.9, 1.0),
            )?;
        }
        while let Some(tile) = tiles.pop_front() {
            tile.finish(geo, &mut curves, &mut run)?;
        }
        let run = run.unwrap();
        ctx.report(1.0, "geometric MTF");
        Ok(GeometricMtfResult {
            frequencies: freqs,
            curves,
            diffraction_limit: None,
            cutoff_frequency: cutoff,
            field_values: fields.iter().map(|&f| field_value(t, f)).collect(),
            run_info: stamp(run, "geometric_mtf", s, warnings(t, s.defocus)),
        })
    }
}
/// Characteristic-function tiles in flight at most; older tiles are
/// collected before more are submitted.
const MTF_TILES_IN_FLIGHT: usize = 4;
/// A submitted characteristic tile: frequencies `start..end` of the fields
/// `fields` (positions), with their summed weights.
struct MtfTile {
    fields: std::ops::Range<usize>,
    batch: usize,
    weights: Vec<f64>,
    submitted: plans::Submitted,
}
impl MtfTile {
    fn finish(
        self,
        geo: &Geometry<'_>,
        curves: &mut [MtfCurve],
        run: &mut Option<RunInfo>,
    ) -> MkResult<()> {
        let rr = geo.finish(self.submitted)?;
        let v = geo.backend.read_view(rr.outputs[0].view())?;
        let v = v.as_slice::<Df32>()?;
        let nf = self.fields.len();
        let segments = self.batch * nf;
        for k in 0..self.batch {
            for fi in 0..nf {
                let i = k * nf + fi;
                let w = self.weights[fi];
                curves[2 * (self.fields.start + fi) + k % 2]
                    .modulation
                    .push(if w > 0.0 {
                        v[i].to_f64().hypot(v[segments + i].to_f64()) / w
                    } else {
                        f64::NAN
                    });
            }
        }
        plans::merge_run(run, rr.run_info);
        Ok(())
    }
}
/// Submit the characteristic tiles of one field group (positions `range`)
/// whose grid plan `r` produced its points, mask and moments. `ctx` is the
/// tiles' slice of the progress, shared by frequency.
#[allow(clippy::too_many_arguments)]
fn mtf_tiles(
    geo: &Geometry<'_>,
    t: &SequentialTracer,
    fields: &[usize],
    range: std::ops::Range<usize>,
    r: &PlanResult,
    freqs: &[f64],
    len: usize,
    tiles: &mut std::collections::VecDeque<MtfTile>,
    curves: &mut [MtfCurve],
    run: &mut Option<RunInfo>,
    ctx: &JobContext,
) -> MkResult<()> {
    let group = &fields[range.clone()];
    let n = group.len() * len;
    let mom = geo.backend.read_view(r.outputs[2].view())?;
    let mom = mom.as_slice::<Df32>()?;
    let weights: Vec<f64> = (0..group.len())
        .map(|fi| mom[fi * 10 + 1].to_f64())
        .collect();
    // Characteristic terms per tile: 2²⁰, or one frequency pair of every
    // field when that is larger (as for the whole grid at once), so field
    // groups add no tile plans over a single grid plan.
    let budget = (1usize << 20).max(2 * fields.len() * len);
    let tile = (budget / (2 * n).max(1)).clamp(1, freqs.len());
    for start in (0..freqs.len()).step_by(tile) {
        ctx.check()?;
        let end = (start + tile).min(freqs.len());
        let batch = 2 * (end - start);
        let segments = batch * group.len();
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let q = p.import(r.outputs[0].view());
        let mask = p.import(r.outputs[1].view());
        let frequency = geo.import::<Df32>(&mut p, [batch, group.len(), 2], |out| {
            for k in 0..batch {
                let axis = k % 2;
                let freq = freqs[start + k / 2];
                for (fi, &field) in group.iter().enumerate() {
                    let pt = t.field(field).point;
                    let h = pt.hx.hypot(pt.hy);
                    let e = if h > 0.0 {
                        [pt.hx / h, pt.hy / h]
                    } else {
                        [0.0, 1.0]
                    };
                    let e = if axis == 0 { e } else { [e[1], -e[0]] };
                    let i = 2 * (k * group.len() + fi);
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
        if tiles.len() >= MTF_TILES_IN_FLIGHT {
            tiles.pop_front().unwrap().finish(geo, curves, run)?;
        }
        tiles.push_back(MtfTile {
            fields: range.clone(),
            batch,
            weights: weights.clone(),
            submitted: geo.submit(
                p,
                &ctx.subrange(
                    start as f64 / freqs.len() as f64,
                    end as f64 / freqs.len() as f64,
                ),
            )?,
        });
    }
    Ok(())
}

/// Each field's spot mean square is exactly quadratic in defocus, with
/// coefficients from one resident two-pass reduction (`FocusTerms`). Brent
/// and the through-focus curve evaluate that quadratic on the host; the
/// reported optimum is re-reduced two-pass at the chosen plane, where the
/// quadratic form could cancel near a sharp focus.
pub fn best_focus_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &BestFocusSettings,
    route: &Route,
    ctx: &JobContext,
) -> MkResult<BestFocusResult> {
    crate::energy::validate_best_focus(s)?;
    let aug = default_fields(t, &s.sampling.fields)?;
    let t = aug.as_ref();
    let Setup {
        geo,
        plan: mut p,
        grid: g,
        points: q,
        mask,
        moments: m0,
        fields,
    } = setup(b, t, &s.sampling, 0.0, route, ctx)?;
    p.settings = serde_json::to_value(s).unwrap();
    let len = g.pupil_count * g.wavelengths.len();
    let (quad, _) = geo.focus_quadratic(
        &mut p,
        q,
        mask,
        m0,
        len,
        s.reference == SpotReference::Centroid,
    )?;
    let chiefs = g.chiefs()?.clone();
    p.outputs = vec![
        g.hits.rays.floats,
        g.hits.status,
        chiefs.rays.floats,
        chiefs.status,
        m0,
        quad,
    ];
    let r = geo.run(p, &ctx.subrange(0.0, 0.8))?;
    let mut run = r.run_info.clone();
    let m0 = b.read_view(r.outputs[4].view())?;
    let m0 = m0.as_slice::<Df32>()?;
    let quad = b.read_view(r.outputs[5].view())?;
    let quad = quad.as_slice::<Df32>()?;
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
    let nf = fields.len();
    let primary = g
        .chief_wavelengths
        .iter()
        .position(|&w| w == t.system().primary_wavelength)
        .unwrap();
    for (fi, &f) in fields.iter().enumerate() {
        if s.reference == SpotReference::Chief
            && cstatus[fi * g.chief_wavelengths.len() + primary] != 0
        {
            warnings.push(format!(
                "field {f}: primary chief ray failed; skipped (chief reference)"
            ));
            continue;
        }
        let weight_sum = m0[10 * fi + 1].to_f64();
        if weight_sum <= 0.0 {
            warnings.push(format!("field {f}: no rays arrived"));
            continue;
        }
        // Mean square at defocus d: a + 2 b d + c d².
        let [a, bb, cc] = [0, 1, 2].map(|k| quad[k * nf + fi].to_f64() / weight_sum);
        let weight = t
            .system()
            .system()
            .fields
            .fields
            .get(f)
            .map_or(1.0, |f| f.weight);
        used.push((fi, f, weight, bb, cc, a));
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
    let weighted = |v: &[f64]| {
        v.iter()
            .zip(&used)
            .map(|(v, u)| v * u.2)
            .collect::<mklight_core::math::KahanSum>()
            .value()
            / sw
    };
    let quadratic = |d: f64| -> Vec<f64> {
        used.iter()
            .map(|u| (u.5 + d * (2.0 * u.3 + d * u.4)).max(0.0).sqrt())
            .collect()
    };
    ctx.check()?;
    let br = crate::util::brent_minimize(
        |d| Ok(weighted(&quadratic(d))),
        range[0],
        range[1],
        1e-12,
        s.tolerance,
        s.max_evaluations,
    )?;
    ctx.check()?;
    // Two-pass moments at the optimum; δ = 0 reuses the initial reduction.
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    let mut g = g.clone();
    g.hits.rays.floats = p.import(r.outputs[0].view());
    g.hits.status = p.import(r.outputs[1].view());
    let mut c = chiefs;
    c.rays.floats = p.import(r.outputs[2].view());
    c.status = p.import(r.outputs[3].view());
    g.chiefs = Some(c);
    let (q, m) = geo.coordinates(&mut p, &g, br.x, false)?;
    let m = geo.moments(&mut p, q, m, len, true)?;
    p.outputs = vec![m];
    let rr = geo.run(p, &ctx.subrange(0.8, 0.99))?;
    plans::add_run(&mut run, rr.run_info.clone());
    let v = b.read_view(rr.outputs[0].view())?;
    let v = v.as_slice::<Df32>()?;
    let field_rms: Vec<_> = used
        .iter()
        .map(|u| sq(&v[10 * u.0..]).max(0.0).sqrt())
        .collect();
    let field_rms_at_zero: Vec<_> = used
        .iter()
        .map(|u| sq(&m0[10 * u.0..]).max(0.0).sqrt())
        .collect();
    let tfd = if s.through_focus_samples > 0 {
        linspace(range[0], range[1], s.through_focus_samples.max(2))
    } else {
        Vec::new()
    };
    let tfm = tfd.iter().map(|&d| weighted(&quadratic(d))).collect();
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
        merit: weighted(&field_rms),
        merit_at_zero: weighted(&field_rms_at_zero),
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
