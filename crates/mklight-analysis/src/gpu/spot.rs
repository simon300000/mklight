use super::*;
use crate::{
    spot::*,
    util::{curved_image_warning, default_fields, field_value, pupil_points},
};
use division::Division;
/// Complete resident spot pipeline: generation, trace, chief-relative two-pass
/// moments and optional points. No intermediate optical arrays cross the host.
pub fn spot_diagram_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &SpotDiagramSettings,
    ctx: &JobContext,
) -> MkResult<SpotDiagram> {
    Geometry::new(b, t, ctx)?.spot_job(s, ctx, false)
}
/// [`spot_diagram_job`] with the points packed straight from the mapped
/// device views into [`SpotDiagram::packed_points`] (see
/// [`crate::spot::spot_diagram_packed_job`]).
pub fn spot_diagram_packed_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &SpotDiagramSettings,
    ctx: &JobContext,
) -> MkResult<SpotDiagram> {
    Geometry::new(b, t, ctx)?.spot_job(s, ctx, true)
}
/// The device outputs of one field group's spot plan, read in place.
struct Part {
    /// Field positions of the group.
    fields: std::ops::Range<usize>,
    grid: ResidentGrid,
    views: Vec<ReadView>,
}
/// Typed slices of a [`Part`].
struct PartViews<'v> {
    start: usize,
    nf: usize,
    poly: &'v [Df32],
    mono: &'v [Df32],
    chiefs: &'v [f32],
    cstatus: &'v [u32],
    nc: usize,
    quad: &'v [Df32],
    slope: &'v [Df32],
    q: &'v [Df32],
    mask: &'v [u32],
    n: usize,
}
impl Part {
    fn typed(&self, through_focus: bool, points: bool) -> MkResult<PartViews<'_>> {
        let v = &self.views;
        let tf = 4 + 2 * usize::from(through_focus);
        Ok(PartViews {
            start: self.fields.start,
            nf: self.fields.len(),
            poly: v[0].as_slice()?,
            mono: v[1].as_slice()?,
            chiefs: v[2].as_slice()?,
            cstatus: v[3].as_slice()?,
            nc: self.grid.chiefs()?.rays.count,
            quad: if through_focus { v[4].as_slice()? } else { &[] },
            slope: if through_focus { v[5].as_slice()? } else { &[] },
            q: if points { v[tf].as_slice()? } else { &[] },
            mask: if points { v[tf + 1].as_slice()? } else { &[] },
            n: self.grid.hits.rays.count,
        })
    }
}
impl Geometry<'_> {
    /// Reuse immutable systems and chief/aim tables across repeated jobs.
    pub fn spot_diagram_job(
        &self,
        s: &SpotDiagramSettings,
        ctx: &JobContext,
    ) -> MkResult<SpotDiagram> {
        self.spot_job(s, ctx, false)
    }
    /// The spot plan of field points `points`: generation, trace, moments,
    /// optional through-focus terms and points.
    fn spot_plan(
        &self,
        t: &SequentialTracer,
        s: &SpotDiagramSettings,
        points: &[FieldPoint],
        wls: &[usize],
        pupil: &[[f64; 2]],
        ctx: &JobContext,
    ) -> MkResult<(ResidentPlan, ResidentGrid)> {
        let geo = self;
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        p.settings = serde_json::to_value(s).unwrap();
        p.seed = matches!(
            s.pattern,
            crate::util::PupilPattern::Random | crate::util::PupilPattern::Dithered
        )
        .then_some(s.seed);
        let mut g = geo.append_grid(
            &mut p,
            points,
            wls,
            pupil,
            t.aiming(),
            t.system().image_index(),
            true,
            false,
            ctx,
        )?;
        geo.append_chiefs(&mut p, &mut g)?;
        let chief_hits = g.chiefs()?.clone();
        let (q, mask) = geo.coordinates(&mut p, &g, s.defocus, false)?;
        let poly = geo.moments(&mut p, q, mask, wls.len() * pupil.len(), true)?;
        let (qw, mw) = geo.coordinates(&mut p, &g, s.defocus, true)?;
        let mono = geo.moments(&mut p, qw, mw, pupil.len(), false)?;
        p.outputs = vec![poly, mono, chief_hits.rays.floats, chief_hits.status];
        if s.through_focus.is_some() {
            // Spot moments are exactly quadratic in defocus: one reduction
            // serves every through-focus plane.
            let (quad, slope) =
                geo.focus_quadratic(&mut p, q, mask, poly, wls.len() * pupil.len(), true)?;
            p.outputs.extend([quad, slope]);
        }
        if s.include_points {
            p.outputs.extend([q, mask]);
        }
        Ok((p, g))
    }
    fn spot_job(
        &self,
        s: &SpotDiagramSettings,
        ctx: &JobContext,
        packed: bool,
    ) -> MkResult<SpotDiagram> {
        let b = self.backend;
        let t = self.tracer;
        crate::spot::validate_spot_settings(s)?;
        let division = Division::selected()?;
        let augmented = default_fields(t, &s.fields)?;
        let t = augmented.as_ref();
        let fields = s.fields.resolve(t.fields().len())?;
        let wls = s
            .wavelengths
            .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
        let pupil = pupil_points(s.pattern, s.rings, s.seed);
        let points: Vec<_> = fields.iter().map(|&i| t.field(i).point).collect();
        if division == Division::HostReduce {
            return self.spot_host_reduce(t, s, &fields, &wls, &pupil, ctx, packed);
        }
        let groups = division.groups(fields.len(), wls.len() * pupil.len());
        // Each group's rays are aimed while the device works on the groups
        // before it; every plan is collected at the end.
        let mut submitted = Vec::with_capacity(groups.len());
        for (k, range) in groups.iter().enumerate() {
            let sub = ctx.subrange(
                0.95 * k as f64 / groups.len() as f64,
                0.95 * (k + 1) as f64 / groups.len() as f64,
            );
            let (p, g) = self.spot_plan(t, s, &points[range.clone()], &wls, &pupil, &sub)?;
            submitted.push((range.clone(), g, self.submit(p, &sub)?));
        }
        let mut run: Option<RunInfo> = None;
        let mut parts = Vec::with_capacity(submitted.len());
        for (fields, grid, sub) in submitted {
            let r = self.finish(sub)?;
            let views = r
                .outputs
                .iter()
                .map(|h| b.read_view(h.view()))
                .collect::<MkResult<_>>()?;
            division::merge_run(&mut run, r.run_info);
            parts.push(Part {
                fields,
                grid,
                views,
            });
        }
        let mut run = run.unwrap();
        division::mark(&mut run, division, parts.len());
        self.spot_decode(t, s, &fields, &wls, &pupil, &parts, run, ctx, packed)
    }
    /// Spot statistics, chief data and points from the device reductions of
    /// the field groups.
    #[allow(clippy::too_many_arguments)]
    fn spot_decode(
        &self,
        t: &SequentialTracer,
        s: &SpotDiagramSettings,
        fields: &[usize],
        wls: &[usize],
        pupil: &[[f64; 2]],
        parts: &[Part],
        run: RunInfo,
        ctx: &JobContext,
        packed: bool,
    ) -> MkResult<SpotDiagram> {
        let tf = s.through_focus.is_some();
        let views: Vec<PartViews> = parts
            .iter()
            .map(|p| p.typed(tf, s.include_points))
            .collect::<MkResult<_>>()?;
        // The group and local position of field position fi.
        let at = |fi: usize| {
            let v = views
                .iter()
                .find(|v| (v.start..v.start + v.nf).contains(&fi))
                .unwrap();
            (v, fi - v.start)
        };
        let chief_wavelengths = &parts[0].grid.chief_wavelengths;
        let column = |w: usize| chief_wavelengths.iter().position(|&v| v == w).unwrap();
        let np = pupil.len();
        let nf = fields.len();
        let nw = wls.len();
        let planes = s.through_focus.map_or(Vec::new(), |tf| {
            // The automatic range needs the largest centroid RMS over the
            // fields at the base defocus, as the reference computes it.
            let max_rms = (0..nf)
                .map(|fi| {
                    let (v, l) = at(fi);
                    stats(&v.poly[l * 10..l * 10 + 10], np * nw, true, 1.0).rms_radius_centroid
                })
                .filter(|v| v.is_finite())
                .fold(0.0, f64::max);
            tf.planes(ThroughFocus::auto_half_range(
                t.system().wavelengths_um[t.system().primary_wavelength],
                t.paraxial().primary().working_f_number,
                max_rms,
            ))
        });
        // Centroid- and chief-referenced RMS of field fi at absolute defocus d.
        let focus = |fi: usize, d: f64| {
            let (v, l) = at(fi);
            let m = &v.poly[10 * l..];
            let w = m[1].to_f64();
            if !(w > 0.0) {
                return [f64::NAN; 2];
            }
            let e = d - s.defocus;
            let [a, b2, c] = [0, 1, 2].map(|k| v.quad[k * v.nf + l].to_f64());
            let var = ((a + e * (2.0 * b2 + e * c)) / w).max(0.0);
            let centroid: [f64; 2] =
                std::array::from_fn(|k| m[2 + k].to_f64() + e * v.slope[10 * l + 2 + k].to_f64());
            [
                var.sqrt(),
                (var + centroid[0] * centroid[0] + centroid[1] * centroid[1]).sqrt(),
            ]
        };
        // Shared device storage is read in place; points go straight into
        // their final arrays, sized by one pass over the arrival masks.
        let mut packed = packed.then(|| {
            let n = if s.include_points {
                views
                    .iter()
                    .map(|v| v.mask.iter().filter(|&&m| m != 0).count())
                    .sum()
            } else {
                0
            };
            PackedSpotPoints::with_capacity(n, nf)
        });
        let mut out = Vec::new();
        let mut warnings = t.warnings().to_vec();
        for (fi, &field) in fields.iter().enumerate() {
            ctx.check()?;
            let (pv, l) = at(fi);
            let ci = l * chief_wavelengths.len() + column(t.system().primary_wavelength);
            let c = chief(pv.chiefs, pv.cstatus, pv.nc, ci, s.defocus);
            let v = &pv.poly[l * 10..l * 10 + 10];
            let pol = stats(
                v,
                np * nw,
                c[0].is_finite(),
                np as f64
                    * wls
                        .iter()
                        .map(|&w| t.system().wavelength_weights[w])
                        .sum::<f64>(),
            );
            let origin = if c[0].is_finite() { c } else { [0.0; 2] };
            let mean = [v[2].to_f64(), v[3].to_f64()];
            if !c[0].is_finite() {
                warnings.push(format!(
                "field {field}: primary chief ray failed; statistics about the chief are undefined"
            ));
            }
            let centroid = [origin[0] + mean[0], origin[1] + mean[1]];
            let (center, rms, rad) = match s.reference {
                SpotReference::Chief => (
                    if c[0].is_finite() {
                        [0.0; 2]
                    } else {
                        [f64::NAN; 2]
                    },
                    pol.rms_radius_chief,
                    pol.geo_radius_chief,
                ),
                SpotReference::Centroid => (mean, pol.rms_radius_centroid, pol.geo_radius_centroid),
            };
            let per_wavelength = wls
                .iter()
                .enumerate()
                .map(|(wi, &w)| {
                    let ci = l * chief_wavelengths.len() + column(w);
                    // As in the reference, wavelength chiefs are reported
                    // relative to a successful primary chief only.
                    let c = if c[0].is_finite() {
                        chief(pv.chiefs, pv.cstatus, pv.nc, ci, s.defocus)
                    } else {
                        [f64::NAN; 2]
                    };
                    WavelengthSpot {
                        wavelength: w,
                        wavelength_um: t.system().wavelengths_um[w],
                        weight: t.system().wavelength_weights[w],
                        chief: c,
                        stats: stats(
                            &pv.mono[(l * nw + wi) * 10..(l * nw + wi + 1) * 10],
                            np,
                            c[0].is_finite(),
                            np as f64,
                        ),
                    }
                })
                .collect();
            let mut plotted = Vec::new();
            // The reference's slope: zero for the chief (the origin of the
            // slope differences), the weighted mean for the centroid.
            let center_slope = match s.reference {
                SpotReference::Centroid if !planes.is_empty() => {
                    [pv.slope[10 * l + 2].to_f64(), pv.slope[10 * l + 3].to_f64()]
                }
                _ => [0.0; 2],
            };
            let (q, mask, n) = (pv.q, pv.mask, pv.n);
            let rows = l * nw * np..(l + 1) * nw * np;
            if let Some(p) = packed.as_mut() {
                let start = p.len();
                if s.include_points {
                    let with_slope = !planes.is_empty();
                    for chunk in rows.clone().step_by(POINT_CHUNK) {
                        ctx.check()?;
                        for i in chunk..(chunk + POINT_CHUNK).min(rows.end) {
                            if mask[i] != 0 {
                                p.push(
                                    fi,
                                    q[i].to_f64() - center[0],
                                    q[n + i].to_f64() - center[1],
                                    wls[(i / np) % nw],
                                );
                                if with_slope {
                                    p.push_slope(
                                        q[2 * n + i].to_f64() - center_slope[0],
                                        q[3 * n + i].to_f64() - center_slope[1],
                                    );
                                }
                            }
                        }
                    }
                }
                p.close(start);
            } else if s.include_points {
                for i in rows {
                    if mask[i] != 0 {
                        plotted.push(SpotPoint {
                            x: q[i].to_f64() - center[0],
                            y: q[n + i].to_f64() - center[1],
                            wavelength: wls[(i / np) % nw],
                        });
                    }
                }
            }
            let through_focus = planes
                .iter()
                .map(|&d| {
                    let [centroid, chief] = focus(fi, d);
                    FocusSample {
                        defocus: d,
                        rms_radius_centroid: centroid,
                        rms_radius_chief: if c[0].is_finite() { chief } else { f64::NAN },
                    }
                })
                .collect();
            out.push(FieldSpot {
                field,
                field_value: field_value(t, field),
                points: plotted,
                reference: [origin[0] + center[0], origin[1] + center[1]],
                chief: c,
                centroid,
                rms_radius: rms,
                geo_radius: rad,
                polychromatic: pol,
                per_wavelength,
                through_focus,
            });
            ctx.report(0.95 + 0.05 * (fi + 1) as f64 / nf as f64, "spot result");
        }
        if s.defocus != 0.0 || s.through_focus.is_some() {
            warnings.extend(curved_image_warning(t));
        }
        Ok(SpotDiagram {
            fields: out,
            wavelengths_um: t.system().wavelengths_um.clone(),
            wavelengths: wls.to_vec(),
            airy_radius: 1.22
                * t.system().wavelengths_um[t.system().primary_wavelength]
                * 1e-3
                * t.paraxial().primary().working_f_number,
            defocus: s.defocus,
            run_info: stamp(run, "spot_diagram", s, warnings),
            packed_points: packed,
        })
    }
    /// [`Division::HostReduce`]: the device traces the grid and its chief
    /// rays; the host reduces the hits, read in place, with the reference's
    /// own spot reduction.
    #[allow(clippy::too_many_arguments)]
    fn spot_host_reduce(
        &self,
        t: &SequentialTracer,
        s: &SpotDiagramSettings,
        fields: &[usize],
        wls: &[usize],
        pupil: &[[f64; 2]],
        ctx: &JobContext,
        packed: bool,
    ) -> MkResult<SpotDiagram> {
        let seed = matches!(
            s.pattern,
            crate::util::PupilPattern::Random | crate::util::PupilPattern::Dithered
        )
        .then_some(s.seed);
        let traced = self.traced_batch(
            t,
            fields,
            wls,
            pupil,
            Division::HostReduce,
            &serde_json::to_value(s).unwrap(),
            seed,
            &ctx.subrange(0.0, 0.8),
        )?;
        let reduced = crate::spot::spot_reduce(
            mklight_core::job::RunTimer::start(),
            t,
            s,
            &traced.batch,
            &crate::util::BackendInfo::cpu_reference(),
            traced.warnings,
            &ctx.subrange(0.8, 1.0),
            packed,
        )?;
        let mut warnings = t.warnings().to_vec();
        warnings.extend(reduced.run_info.warnings.iter().cloned());
        Ok(SpotDiagram {
            run_info: stamp(traced.run, "spot_diagram", s, warnings),
            ..reduced
        })
    }
}
pub(super) fn stats(v: &[Df32], launched: usize, chief_ok: bool, total: f64) -> SpotStats {
    let x: Vec<_> = v.iter().map(|v| v.to_f64()).collect();
    let w = x[1];
    let nan = |v| if chief_ok { v } else { f64::NAN };
    let root = |v: f64| {
        if w > 0.0 {
            (v / w).max(0.0).sqrt()
        } else {
            f64::NAN
        }
    };
    SpotStats {
        rays_launched: launched,
        rays_arrived: x[0] as usize,
        vignetted_fraction: if total > 0.0 {
            1.0 - w / total
        } else {
            f64::NAN
        },
        centroid: [nan(x[2]), nan(x[3])],
        rms_radius_centroid: root(x[5] + x[7]),
        rms_radius_chief: nan(root(x[4])),
        geo_radius_centroid: if w > 0.0 { x[9].sqrt() } else { f64::NAN },
        geo_radius_chief: nan(if w > 0.0 { x[8].sqrt() } else { f64::NAN }),
        rms_x: root(x[5]),
        rms_y: root(x[7]),
    }
}
