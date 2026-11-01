use super::*;
use crate::{
    spot::*,
    util::{curved_image_warning, default_fields, field_value, linspace, pupil_points},
};
/// Complete resident spot pipeline: generation, trace, chief-relative two-pass
/// moments and optional points. No intermediate optical arrays cross the host.
pub fn spot_diagram_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &SpotDiagramSettings,
    ctx: &JobContext,
) -> MkResult<SpotDiagram> {
    Geometry::new(b, t, ctx)?.spot_diagram_job(s, ctx)
}
impl Geometry<'_> {
    /// Reuse immutable systems and chief/aim tables across repeated jobs.
    pub fn spot_diagram_job(
        &self,
        s: &SpotDiagramSettings,
        ctx: &JobContext,
    ) -> MkResult<SpotDiagram> {
        let b = self.backend;
        let t = self.tracer;
        crate::spot::validate_spot_settings(s)?;
        let augmented = default_fields(t, &s.fields)?;
        let t = augmented.as_ref();
        let fields = s.fields.resolve(t.fields().len())?;
        let wls = s
            .wavelengths
            .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
        let pupil = pupil_points(s.pattern, s.rings, s.seed);
        let points: Vec<_> = fields.iter().map(|&i| t.field(i).point).collect();
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
        let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
        let views: Vec<_> = r
            .outputs
            .iter()
            .map(|h| b.read_view(h.view()))
            .collect::<MkResult<_>>()?;
        let poly = views[0].as_slice::<Df32>()?;
        let mono = views[1].as_slice::<Df32>()?;
        let chiefs = views[2].as_slice::<f32>()?;
        let cstatus = views[3].as_slice::<u32>()?;
        let nc = chief_hits.rays.count;
        let np = pupil.len();
        let nf = fields.len();
        let nw = wls.len();
        let points = 4 + 2 * usize::from(s.through_focus.is_some());
        let planes = s
            .through_focus
            .map_or(Vec::new(), |tf| linspace(tf.min, tf.max, tf.steps));
        let (quad, slope) = if planes.is_empty() {
            (&[][..], &[][..])
        } else {
            (views[4].as_slice::<Df32>()?, views[5].as_slice::<Df32>()?)
        };
        // Centroid- and chief-referenced RMS of field fi at absolute defocus d.
        let focus = |fi: usize, d: f64| {
            let m = &poly[10 * fi..];
            let w = m[1].to_f64();
            if !(w > 0.0) {
                return [f64::NAN; 2];
            }
            let e = d - s.defocus;
            let [a, b2, c] = [0, 1, 2].map(|k| quad[k * nf + fi].to_f64());
            let var = ((a + e * (2.0 * b2 + e * c)) / w).max(0.0);
            let centroid: [f64; 2] =
                std::array::from_fn(|k| m[2 + k].to_f64() + e * slope[10 * fi + 2 + k].to_f64());
            [
                var.sqrt(),
                (var + centroid[0] * centroid[0] + centroid[1] * centroid[1]).sqrt(),
            ]
        };
        let mut out = Vec::new();
        let mut warnings = t.warnings().to_vec();
        for (fi, &field) in fields.iter().enumerate() {
            ctx.check()?;
            let ci = fi * g.chief_wavelengths.len()
                + g.chief_wavelengths
                    .iter()
                    .position(|&w| w == t.system().primary_wavelength)
                    .unwrap();
            let c = chief(chiefs, cstatus, nc, ci, s.defocus);
            let v = &poly[fi * 10..fi * 10 + 10];
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
                    let ci = fi * g.chief_wavelengths.len()
                        + g.chief_wavelengths.iter().position(|&v| v == w).unwrap();
                    // As in the reference, wavelength chiefs are reported
                    // relative to a successful primary chief only.
                    let c = if c[0].is_finite() {
                        chief(chiefs, cstatus, nc, ci, s.defocus)
                    } else {
                        [f64::NAN; 2]
                    };
                    WavelengthSpot {
                        wavelength: w,
                        wavelength_um: t.system().wavelengths_um[w],
                        weight: t.system().wavelength_weights[w],
                        chief: c,
                        stats: stats(
                            &mono[(fi * nw + wi) * 10..(fi * nw + wi + 1) * 10],
                            np,
                            c[0].is_finite(),
                            np as f64,
                        ),
                    }
                })
                .collect();
            let mut plotted = Vec::new();
            if s.include_points {
                let q = views[points].as_slice::<Df32>()?;
                let mask = views[points + 1].as_slice::<u32>()?;
                let n = g.hits.rays.count;
                for i in fi * nw * np..(fi + 1) * nw * np {
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
            wavelengths: wls,
            airy_radius: 1.22
                * t.system().wavelengths_um[t.system().primary_wavelength]
                * 1e-3
                * t.paraxial().primary().working_f_number,
            defocus: s.defocus,
            run_info: stamp(r.run_info, "spot_diagram", s, warnings),
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
