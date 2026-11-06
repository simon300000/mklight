//! Spot diagrams, fans, encircled energy, geometric MTF and best focus
//! against exact results for an ideal lens (where every ray position is a
//! linear function of the pupil point) and against direct traces.

#![allow(clippy::needless_range_loop)]

mod common;

use common::*;
use mklight_analysis::energy::*;
use mklight_analysis::fans::*;
use mklight_analysis::spot::*;
use mklight_analysis::util::{FieldSelection, PupilPattern, WavelengthSelection, pupil_points};
use mklight_core::{SemiDiameter, samples};
use mklight_trace::TraceOptions;

const F: f64 = 100.0;
const EPD: f64 = 20.0;

/// Ideal lens with the image `extra` mm beyond its focus.
fn defocused(extra: f64) -> mklight_trace::SequentialTracer {
    let mut s = ideal_lens(F, EPD, &[0.0, 5.0]);
    s.surfaces[1].thickness = F + extra;
    tracer(s)
}

fn j1(x: f64) -> f64 {
    // (1/2π)∫₀^{2π} cos(τ − x sin τ) dτ, trapezoidal (spectral for periodic).
    let n = 256;
    (0..n)
        .map(|k| {
            let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (t - x * t.sin()).cos()
        })
        .sum::<f64>()
        / n as f64
}

#[test]
fn defocused_ideal_lens_spot_is_the_scaled_pupil() {
    let d = 0.5;
    let t = defocused(d);
    let s = spot_diagram(
        &t,
        &SpotDiagramSettings {
            pattern: PupilPattern::Square,
            rings: 16,
            ..Default::default()
        },
    )
    .unwrap();
    let pts = pupil_points(PupilPattern::Square, 16, 1);
    let r = d * 0.5 * EPD / F; // blur radius
    let ms = pts.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum::<f64>() / pts.len() as f64;
    for fs in &s.fields {
        // Every ray lands at −(d/f)·X relative to the chief (any field).
        assert!(
            (fs.rms_radius - r * ms.sqrt()).abs() < 1e-12,
            "{}",
            fs.rms_radius
        );
        assert!((fs.geo_radius - r).abs() < 1e-12);
        assert!((fs.polychromatic.rms_radius_centroid - r * ms.sqrt()).abs() < 1e-12);
        assert!(
            fs.polychromatic.centroid[0].abs() < 1e-14
                && fs.polychromatic.centroid[1].abs() < 1e-14
        );
        assert_eq!(fs.points.len(), pts.len());
        assert_eq!(fs.polychromatic.vignetted_fraction, 0.0);
    }
    // Airy radius 1.22 λ N_w (paraxial working F/# = f/EPD at infinity).
    assert!((s.airy_radius - 1.22 * 0.5875618e-3 * F / EPD).abs() < 1e-15);
    // Through focus: the spot vanishes at δ = −d.
    let tf = spot_diagram(
        &t,
        &SpotDiagramSettings {
            include_points: false,
            through_focus: Some(ThroughFocus {
                min: Some(-d),
                max: Some(0.0),
                steps: 3,
            }),
            ..Default::default()
        },
    )
    .unwrap();
    let c = &tf.fields[0].through_focus;
    assert!(c[0].rms_radius_chief < 1e-12);
    assert!((c[1].rms_radius_chief - 0.5 * c[2].rms_radius_chief).abs() < 1e-12);
}

#[test]
fn encircled_energy_and_mtf_of_a_uniform_blur() {
    let d = 0.5;
    let t = defocused(d);
    let rings = 40;
    let pts = pupil_points(PupilPattern::Square, rings, 1);
    let r = d * 0.5 * EPD / F;
    let sampling = RaySampling {
        rings,
        fields: FieldSelection::Index { index: 0 },
        wavelengths: WavelengthSelection::Primary,
        ..Default::default()
    };
    let ee = encircled_energy(
        &t,
        &EncircledEnergySettings {
            sampling: sampling.clone(),
            max_radius: Some(r),
            samples: 11,
            ..Default::default()
        },
    )
    .unwrap();
    let f = &ee.fields[0];
    for (i, &rad) in ee.radius.iter().enumerate() {
        // Points within 1e-12 mm of the circle may fall either way.
        let count = |e: f64| {
            pts.iter()
                .filter(|p| r * p[0].hypot(p[1]) <= rad + e)
                .count() as f64
                / pts.len() as f64
        };
        let (lo, hi) = (count(-1e-12), count(1e-12));
        assert!(
            f.encircled[i] >= lo && f.encircled[i] <= hi,
            "{rad}: {} not in [{lo}, {hi}]",
            f.encircled[i]
        );
        // Continuous uniform disk: (r/R)².
        assert!((f.encircled[i] - (rad / r).powi(2)).abs() < 0.03);
        assert!(f.ensquared[i] >= f.encircled[i]);
    }
    assert!((f.r80 - r * 0.8f64.sqrt()).abs() < 0.02 * r);
    assert!((f.encircled[10] - 1.0).abs() < 1e-15);
    // Geometric MTF: 2 J₁(2πνR)/(2πνR).
    let mtf = geometric_mtf(
        &t,
        &GeometricMtfSettings {
            sampling,
            max_frequency: Some(1.5 / r),
            frequency_samples: 16,
            ..Default::default()
        },
    )
    .unwrap();
    for o in [MtfOrientation::Tangential, MtfOrientation::Sagittal] {
        let c = mtf.curve(0, o).unwrap();
        assert_eq!(c.modulation[0], 1.0);
        for (k, &nu) in mtf.frequencies.iter().enumerate().skip(1) {
            let x = 2.0 * std::f64::consts::PI * nu * r;
            let exact = (2.0 * j1(x) / x).abs();
            assert!(
                (c.modulation[k] - exact).abs() < 0.01,
                "{o:?} ν {nu}: {} vs {exact}",
                c.modulation[k]
            );
        }
    }
    assert!(mtf.diffraction_limit.is_none());
}

#[test]
fn best_focus_recovers_a_known_defocus() {
    let t = defocused(0.3);
    let b = best_focus(&t, &BestFocusSettings::default()).unwrap();
    assert!(b.converged);
    assert!((b.defocus + 0.3).abs() < 1e-7, "{}", b.defocus);
    assert!(b.merit < 1e-8 && b.merit_at_zero > 1e-3);
    assert!((b.mean_square_defocus + 0.3).abs() < 1e-12);
}

#[test]
fn best_focus_matches_retraced_spot() {
    let sys = samples::singlet();
    let t = tracer(sys.clone());
    let set = BestFocusSettings {
        sampling: RaySampling {
            fields: FieldSelection::Index { index: 0 },
            ..BestFocusSettings::default().sampling
        },
        through_focus_samples: 5,
        ..Default::default()
    };
    let b = best_focus(&t, &set).unwrap();
    // One field: the minimum of RMS is the closed-form quadratic minimum.
    assert!(
        (b.defocus - b.mean_square_defocus).abs() < 1e-7,
        "{} {}",
        b.defocus,
        b.mean_square_defocus
    );
    assert!(b.through_focus_merit.iter().all(|&m| m >= b.merit));
    // Moving the image surface and retracing gives the same RMS.
    let mut moved = sys;
    let n = moved.surfaces.len();
    moved.surfaces[n - 2].thickness += b.defocus;
    let s = spot_diagram(
        &tracer(moved),
        &SpotDiagramSettings {
            rings: 8,
            fields: FieldSelection::Index { index: 0 },
            include_points: false,
            ..Default::default()
        },
    )
    .unwrap();
    let rms = s.fields[0].polychromatic.rms_radius_centroid;
    assert!(
        (rms - b.merit).abs() < 1e-12 * (1.0 + rms) + 1e-13,
        "{rms} {}",
        b.merit
    );
}

#[test]
fn paraboloid_on_axis_spot_is_a_point() {
    let t = tracer(samples::newtonian_telescope());
    let s = spot_diagram(
        &t,
        &SpotDiagramSettings {
            fields: FieldSelection::Index { index: 0 },
            ..Default::default()
        },
    )
    .unwrap();
    for w in &s.fields[0].per_wavelength {
        assert!(
            w.stats.rms_radius_chief < 1e-9,
            "{}",
            w.stats.rms_radius_chief
        );
        assert!(w.stats.geo_radius_chief < 1e-9);
    }
}

#[test]
fn polychromatic_statistics_are_weighted_per_wavelength_statistics() {
    let mut sys = samples::cooke_triplet();
    sys.wavelengths.wavelengths[0].weight = 0.5;
    sys.wavelengths.wavelengths[2].weight = 2.0;
    let t = tracer(sys);
    let s = spot_diagram(&t, &SpotDiagramSettings::default()).unwrap();
    for f in &s.fields {
        let mut num = [0.0; 2];
        let mut den = 0.0;
        for w in &f.per_wavelength {
            let n = w.stats.rays_arrived as f64 * w.weight;
            for k in 0..2 {
                num[k] += n * (w.chief[k] + w.stats.centroid[k]);
            }
            den += n;
        }
        for k in 0..2 {
            assert!(
                (num[k] / den - f.centroid[k]).abs() < 1e-12,
                "field {}",
                f.field
            );
        }
        // RMS about the centroid never exceeds RMS about the chief.
        assert!(f.polychromatic.rms_radius_centroid <= f.polychromatic.rms_radius_chief + 1e-15);
    }
}

#[test]
fn vignetted_fraction_counts_clipped_rays() {
    let mut s = ideal_lens(F, EPD, &[0.0]);
    s.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 8.0 };
    let t = tracer(s);
    let sp = spot_diagram(
        &t,
        &SpotDiagramSettings {
            pattern: PupilPattern::Square,
            rings: 10,
            ..Default::default()
        },
    )
    .unwrap();
    let pts = pupil_points(PupilPattern::Square, 10, 1);
    let lost = pts.iter().filter(|p| p[0].hypot(p[1]) * 10.0 > 8.0).count();
    let st = &sp.fields[0].polychromatic;
    assert_eq!(st.rays_launched - st.rays_arrived, lost);
    assert!((st.vignetted_fraction - lost as f64 / pts.len() as f64).abs() < 1e-15);
}

#[test]
fn random_patterns_are_reproducible() {
    let t = tracer(samples::double_gauss());
    let set = SpotDiagramSettings {
        pattern: PupilPattern::Random,
        seed: 42,
        ..Default::default()
    };
    let a = spot_diagram(&t, &set).unwrap();
    let b = spot_diagram(&t, &set).unwrap();
    assert_eq!(a.fields, b.fields);
    assert_eq!(a.run_info.seed, Some(42));
    let j = serde_json::to_value(&a).unwrap();
    assert!(j["fields"][0]["rms_radius"].is_number());
}

#[test]
fn fans_match_direct_traces_and_symmetry() {
    let t = tracer(samples::singlet());
    let set = FanSettings {
        samples: 11,
        ..Default::default()
    };
    let rf = ray_fan(&t, &set).unwrap();
    let of = opd_fan(&t, &set).unwrap();
    let o = TraceOptions::default();
    for ff in &rf.fields {
        let pf = t.field(ff.field);
        let c = t.chief(pf, 1);
        for curve in &ff.tangential {
            for (k, &p) in curve.pupil.iter().enumerate() {
                let r = t.trace(pf, curve.wavelength, 0.0, p, &o);
                assert!((curve.values[k] - (r.ray.pos.y - c.ray.pos.y)).abs() < 1e-15);
            }
        }
        // Sagittal fans are odd in Px for εx and even for εy.
        for curve in &ff.sagittal {
            let n = curve.values.len();
            for k in 0..n {
                assert!((curve.values[k] + curve.values[n - 1 - k]).abs() < 1e-12);
                assert!((curve.cross[k] - curve.cross[n - 1 - k]).abs() < 1e-12);
            }
        }
    }
    for ff in &of.fields {
        for curve in ff.tangential.iter().chain(&ff.sagittal) {
            assert!(curve.values[5].abs() < 1e-9); // chief: OPD 0
            let r = t.opd(t.field(ff.field), curve.wavelength, 0.0, curve.pupil[0], &o);
            if ff.tangential.contains(curve) {
                assert_eq!(Some(curve.values[0]), r);
            }
        }
    }
    // An ideal lens at focus has no aberration at all.
    let i = tracer(ideal_lens(F, EPD, &[0.0, 10.0]));
    let rf = ray_fan(&i, &set).unwrap();
    let of = opd_fan(&i, &set).unwrap();
    for ff in rf.fields.iter().chain(&of.fields) {
        for c in ff.tangential.iter().chain(&ff.sagittal) {
            assert!(c.values.iter().all(|v| v.abs() < 1e-9), "{:?}", c.values);
        }
    }
}

#[test]
fn jobs_cancel_and_report_monotone_progress() {
    use mklight_core::{JobContext, MkError};
    let t = tracer(samples::double_gauss());
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(
        spot_diagram_job(&t, &SpotDiagramSettings::default(), &ctx).unwrap_err(),
        MkError::Cancelled
    );
    assert_eq!(
        best_focus_job(&t, &BestFocusSettings::default(), &ctx).unwrap_err(),
        MkError::Cancelled
    );
    assert!(mklight_analysis::layout::layout_job(&t, &Default::default(), &ctx).is_err());
    assert!(
        mklight_analysis::illumination::relative_illumination_job(&t, &Default::default(), &ctx)
            .is_err()
    );
    // Progress reaches 1 and never decreases (the Progress type clamps
    // monotonically; the final value proves the job reported completion).
    let ctx = JobContext::new();
    spot_diagram_job(&t, &SpotDiagramSettings::default(), &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    let ctx = JobContext::new();
    geometric_mtf_job(&t, &GeometricMtfSettings::default(), &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
}

#[test]
fn results_round_trip_through_json_with_non_finite_values() {
    // A vignetted field has NaN statistics: they must serialise (as null)
    // and read back.
    let mut s = ideal_lens(F, EPD, &[0.0, 5.0]);
    // The image aperture passes the on-axis spot only: field 1 is lost.
    s.surfaces[2].semi_diameter = SemiDiameter::Fixed { value: 1.0 };
    let t = tracer(s);
    let sp = spot_diagram(
        &t,
        &SpotDiagramSettings {
            pattern: PupilPattern::Square,
            rings: 2,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(sp.fields[1].polychromatic.vignetted_fraction, 1.0);
    assert!(sp.fields[1].rms_radius.is_nan());
    let text = serde_json::to_string(&sp).unwrap();
    assert!(text.contains("\"rms_radius\":null"));
    let back: SpotDiagram = serde_json::from_str(&text).unwrap();
    assert_eq!(back.fields.len(), sp.fields.len());
    assert_eq!(serde_json::to_string(&back).unwrap(), text);
    let fans = ray_fan(&tracer(samples::cooke_triplet()), &FanSettings::default()).unwrap();
    let text = serde_json::to_string(&fans).unwrap();
    let back: FanResult = serde_json::from_str(&text).unwrap();
    assert_eq!(serde_json::to_string(&back).unwrap(), text);
}

#[test]
fn defocus_on_a_curved_image_surface_warns_everywhere() {
    let mut sys = samples::singlet();
    let n = sys.surfaces.len();
    sys.surfaces[n - 1].shape = mklight_core::SurfaceShape::sphere(-100.0);
    let t = tracer(sys);
    let curved = |w: &[String]| w.iter().any(|s| s.contains("curved image surface"));
    let spot = |defocus: f64, tf: Option<ThroughFocus>| {
        spot_diagram(
            &t,
            &SpotDiagramSettings {
                defocus,
                through_focus: tf,
                ..Default::default()
            },
        )
        .unwrap()
        .run_info
        .warnings
    };
    assert!(!curved(&spot(0.0, None)));
    assert!(curved(&spot(0.5, None)));
    let tf = ThroughFocus::symmetric(0.1, 3);
    assert!(curved(&spot(0.0, Some(tf))));
    let ee = |defocus: f64| {
        encircled_energy(
            &t,
            &EncircledEnergySettings {
                defocus,
                ..Default::default()
            },
        )
        .unwrap()
        .run_info
        .warnings
    };
    assert!(!curved(&ee(0.0)) && curved(&ee(0.5)));
    let mtf = |defocus: f64| {
        geometric_mtf(
            &t,
            &GeometricMtfSettings {
                defocus,
                ..Default::default()
            },
        )
        .unwrap()
        .run_info
        .warnings
    };
    assert!(!curved(&mtf(0.0)) && curved(&mtf(0.5)));
    // A plane image never warns.
    let p = tracer(samples::singlet());
    let w = spot_diagram(
        &p,
        &SpotDiagramSettings {
            defocus: 0.5,
            ..Default::default()
        },
    )
    .unwrap()
    .run_info
    .warnings;
    assert!(!curved(&w));
}

/// Image rays of the given fields and wavelengths, traced directly with the
/// batch tracer, standing in for another backend.
fn external_batch(
    t: &mklight_trace::SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    pupil: &[[f64; 2]],
) -> RayBatch {
    let sys = t.system();
    let weights: Vec<f64> = wls.iter().map(|&w| sys.wavelength_weights[w]).collect();
    let mut out = Vec::new();
    for &f in fields {
        let pf = t.field(f).clone();
        let grid = mklight_trace::Grid {
            fields: std::slice::from_ref(&pf),
            wavelengths: wls,
            pupil,
        };
        let rays: Vec<ImageRay> = mklight_trace::trace_grid(t, &grid, &t.options())
            .iter()
            .map(|r| ImageRay {
                wavelength: r.wavelength as usize,
                hit: r.status.is_ok().then_some(ImageHit {
                    point: [r.point.x, r.point.y],
                    dir: r.dir,
                }),
            })
            .collect();
        let chief = |w: usize| {
            let c = t.chief(&pf, w);
            c.status.is_ok().then_some(ImageHit {
                point: [c.ray.pos.x, c.ray.pos.y],
                dir: c.ray.dir,
            })
        };
        let wc: Vec<_> = wls.iter().map(|&w| chief(w)).collect();
        let mut warn = Vec::new();
        out.push(
            FieldRays::from_image_rays(
                f,
                wls,
                &weights,
                chief(sys.primary_wavelength),
                &wc,
                &rays,
                &mut warn,
            )
            .unwrap(),
        );
        assert!(warn.is_empty());
    }
    RayBatch {
        wavelengths: wls.to_vec(),
        fields: out,
    }
}

#[test]
fn reductions_of_an_external_batch_match_the_traced_analyses() {
    let t = tracer(samples::cooke_triplet());
    let gpu = mklight_analysis::util::BackendInfo {
        name: "test-backend".into(),
        precision: mklight_core::job::Precision::F32Gpu,
    };
    let fields = [0, 1, 2];
    let wls = [0, 1, 2];
    let sp = SpotDiagramSettings {
        rings: 5,
        defocus: 0.02,
        ..Default::default()
    };
    let batch = external_batch(
        &t,
        &fields,
        &wls,
        &pupil_points(sp.pattern, sp.rings, sp.seed),
    );
    let a = spot_diagram(&t, &sp).unwrap();
    let b = spot_from_rays(&t, &sp, &batch, &gpu, Vec::new()).unwrap();
    assert_eq!(a.fields, b.fields);
    assert_eq!(b.run_info.backend, "test-backend");
    assert_eq!(b.run_info.precision, mklight_core::job::Precision::F32Gpu);
    assert_eq!(a.run_info.backend, mklight_analysis::util::BACKEND);

    let sampling = RaySampling {
        pattern: PupilPattern::Square,
        rings: 6,
        ..Default::default()
    };
    let batch = external_batch(&t, &fields, &wls, &pupil_points(sampling.pattern, 6, 1));
    let es = EncircledEnergySettings {
        sampling: sampling.clone(),
        ..Default::default()
    };
    let ea = encircled_energy(&t, &es).unwrap();
    let eb = encircled_energy_from_rays(&t, &es, &batch, &gpu, Vec::new()).unwrap();
    assert_eq!((ea.radius, ea.fields), (eb.radius, eb.fields));
    let ms = GeometricMtfSettings {
        sampling: sampling.clone(),
        ..Default::default()
    };
    let ma = geometric_mtf(&t, &ms).unwrap();
    let mb = geometric_mtf_from_rays(&t, &ms, &batch, &gpu, Vec::new()).unwrap();
    assert_eq!(ma.curves, mb.curves);
    let bs = BestFocusSettings {
        sampling,
        ..Default::default()
    };
    let ba = best_focus(&t, &bs).unwrap();
    let bb = best_focus_from_rays(&t, &bs, &batch, &gpu, Vec::new()).unwrap();
    assert_eq!((ba.defocus, ba.merit), (bb.defocus, bb.merit));

    // Inconsistent batches are rejected.
    let mut bad = batch.clone();
    bad.fields[0].arrived[0] += 1;
    assert!(spot_from_rays(&t, &sp, &bad, &gpu, Vec::new()).is_err());
    let mut bad = batch;
    bad.wavelengths.push(7);
    assert!(best_focus_from_rays(&t, &bs, &bad, &gpu, Vec::new()).is_err());
}

#[test]
fn best_focus_skips_fields_without_a_chief_ray_for_the_chief_reference() {
    // Field 1's primary chief "failed": its positions are relative to the
    // first arrived ray, so an RMS about the chief would be meaningless.
    let t = defocused(0.3);
    let pupil = pupil_points(PupilPattern::Hexapolar, 6, 1);
    let mut batch = external_batch(&t, &[0, 1], &[0], &pupil);
    let lost = {
        let pf = t.field(1).clone();
        let grid = mklight_trace::Grid {
            fields: std::slice::from_ref(&pf),
            wavelengths: &[0],
            pupil: &pupil,
        };
        let rays: Vec<ImageRay> = mklight_trace::trace_grid(&t, &grid, &t.options())
            .iter()
            .map(|r| ImageRay {
                wavelength: 0,
                hit: Some(ImageHit {
                    point: [r.point.x, r.point.y],
                    dir: r.dir,
                }),
            })
            .collect();
        let mut w = Vec::new();
        let f = FieldRays::from_image_rays(1, &[0], &[1.0], None, &[None], &rays, &mut w).unwrap();
        assert!(!f.chief_ok && w.len() == 1);
        f
    };
    batch.fields[1] = lost;
    let set = BestFocusSettings {
        reference: SpotReference::Chief,
        ..Default::default()
    };
    let b = best_focus_from_rays(&t, &set, &batch, &Default::default(), Vec::new()).unwrap();
    assert_eq!(b.fields, vec![0]);
    assert!(b.run_info.warnings.iter().any(|w| w.contains("field 1")));
    assert!((b.defocus + 0.3).abs() < 1e-7);
    // The centroid reference keeps the field.
    let set = BestFocusSettings {
        reference: SpotReference::Centroid,
        ..Default::default()
    };
    let b = best_focus_from_rays(&t, &set, &batch, &Default::default(), Vec::new()).unwrap();
    assert_eq!(b.fields, vec![0, 1]);
}

#[test]
fn best_focus_warns_when_not_converged_or_at_the_range_edge() {
    let t = defocused(0.3);
    let has = |b: &BestFocusResult, s: &str| b.run_info.warnings.iter().any(|w| w.contains(s));
    let b = best_focus(
        &t,
        &BestFocusSettings {
            max_evaluations: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!b.converged && has(&b, "did not converge"));
    let b = best_focus(
        &t,
        &BestFocusSettings {
            range: Some([0.0, 1.0]),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(b.defocus.abs() < 1e-6 && has(&b, "edge of the given search range"));
    let b = best_focus(&t, &BestFocusSettings::default()).unwrap();
    assert!(b.converged && b.run_info.warnings.is_empty());
}

/// The packed slopes place every point on any through-focus plane: the RMS
/// of `q + (d − defocus) · slope` about the reference equals the plane's
/// `rms_radius_*` sample, on the CPU and the GPU, for both references; an
/// automatic range is symmetric and spans more than the depth of focus.
#[test]
fn through_focus_slopes_reproduce_the_plane_statistics() {
    use mklight_analysis::spot::{SpotReference, spot_diagram_packed_job};
    use mklight_core::JobContext;
    let t = mklight_analysis::tracer_for(mklight_core::samples::cooke_triplet()).unwrap();
    let ctx = JobContext::new();
    let gpu = mklight_gpu::CpuF32Backend::new();
    for reference in [SpotReference::Chief, SpotReference::Centroid] {
        for tf in [ThroughFocus::symmetric(0.08, 3), ThroughFocus::default()] {
            let s = SpotDiagramSettings {
                defocus: 0.02,
                reference,
                through_focus: Some(tf),
                ..Default::default()
            };
            let results = [
                spot_diagram_packed_job(&t, &s, &ctx).unwrap(),
                mklight_analysis::gpu::spot_diagram_packed_job(&gpu, &t, &s, &ctx).unwrap(),
            ];
            for r in &results {
                let p = r.packed_points.as_ref().unwrap();
                assert_eq!(p.slope.len(), p.xy.len());
                let planes: Vec<f64> = r.fields[0].through_focus.iter().map(|f| f.defocus).collect();
                assert_eq!(planes.len(), 3);
                if tf.min.is_none() {
                    assert!((planes[0] + planes[2]).abs() < 1e-12 && planes[1] == 0.0);
                    let n = t.paraxial().primary().working_f_number;
                    assert!(planes[2] >= 8.0 * 0.5875618e-3 * n * n);
                }
                for (f, range) in r.fields.iter().zip(&p.ranges) {
                    for sample in &f.through_focus {
                        let e = sample.defocus - s.defocus;
                        let (mut sw, mut sq) = (0.0, 0.0);
                        for i in range[0]..range[0] + range[1] {
                            let w = t.system().wavelength_weights[p.wavelength[i] as usize];
                            let x = p.xy[2 * i] + e * p.slope[2 * i];
                            let y = p.xy[2 * i + 1] + e * p.slope[2 * i + 1];
                            sw += w;
                            sq += w * (x * x + y * y);
                        }
                        let expected = match reference {
                            SpotReference::Chief => sample.rms_radius_chief,
                            SpotReference::Centroid => sample.rms_radius_centroid,
                        };
                        let got = (sq / sw).sqrt();
                        assert!((got - expected).abs() < 1e-6 * expected.max(1e-3), "{got} vs {expected}");
                    }
                }
            }
        }
    }
}
