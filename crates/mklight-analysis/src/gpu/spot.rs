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
    crate::spot::validate_spot_settings(s)?;
    let augmented = default_fields(t, &s.fields)?;
    let t = augmented.as_ref();
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let points: Vec<_> = fields.iter().map(|&i| t.field(i).point).collect();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
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
    let (q, mask) = geo.coordinates(&mut p, &g, s.defocus, false)?;
    let poly = geo.moments(&mut p, q, mask, wls.len() * pupil.len(), true)?;
    let (qw, mw) = geo.coordinates(&mut p, &g, s.defocus, true)?;
    let mono = geo.moments(&mut p, qw, mw, pupil.len(), false)?;
    let mut focus = Vec::new();
    if let Some(tf) = s.through_focus {
        for d in linspace(tf.min, tf.max, tf.steps) {
            let (q, m) = geo.coordinates(&mut p, &g, d, false)?;
            let m = geo.moments(&mut p, q, m, wls.len() * pupil.len(), true)?;
            focus.push((d, m));
        }
    }
    p.outputs = vec![poly, mono, g.chiefs.rays.floats, g.chiefs.status];
    if s.include_points {
        p.outputs.extend([q, mask]);
    }
    p.outputs.extend(focus.iter().map(|v| v.1));
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
    let nc = g.chiefs.rays.count;
    let np = pupil.len();
    let nf = fields.len();
    let nw = wls.len();
    let mut out = Vec::new();
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
        let centroid = [c[0] + pol.centroid[0], c[1] + pol.centroid[1]];
        let (center, rms, rad) = match s.reference {
            SpotReference::Chief => ([0.0; 2], pol.rms_radius_chief, pol.geo_radius_chief),
            SpotReference::Centroid => (
                pol.centroid,
                pol.rms_radius_centroid,
                pol.geo_radius_centroid,
            ),
        };
        let per_wavelength = wls
            .iter()
            .enumerate()
            .map(|(wi, &w)| {
                let ci = fi * g.chief_wavelengths.len()
                    + g.chief_wavelengths.iter().position(|&v| v == w).unwrap();
                let c = chief(chiefs, cstatus, nc, ci, s.defocus);
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
            let q = views[4].as_slice::<Df32>()?;
            let mask = views[5].as_slice::<u32>()?;
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
        let through_focus = focus
            .iter()
            .enumerate()
            .map(|(i, (d, _))| {
                let v = views[4 + usize::from(s.include_points) * 2 + i]
                    .as_slice::<Df32>()
                    .unwrap();
                let s = stats(&v[fi * 10..fi * 10 + 10], nw * np, c[0].is_finite(), 1.0);
                FocusSample {
                    defocus: *d,
                    rms_radius_centroid: s.rms_radius_centroid,
                    rms_radius_chief: s.rms_radius_chief,
                }
            })
            .collect();
        out.push(FieldSpot {
            field,
            field_value: field_value(t, field),
            points: plotted,
            reference: [c[0] + center[0], c[1] + center[1]],
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
    let mut warnings = t.warnings().to_vec();
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
