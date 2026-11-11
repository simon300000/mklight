//! The analysis entry point: energy closure, the class partition and the
//! ghost-solver exclusion, path signatures, Monte-Carlo error calibration,
//! reproducibility across thread counts, and job control.

use crate::common;

use std::sync::Arc;

use common::*;
use mklight_core::{CancelToken, JobContext, MkError, Progress};
use mklight_stray::{
    ClassMask, PathClass, StrayAnalysisResult, StrayAnalysisSettings, StraySource, analyze,
    analyze_system,
};

fn barrelled() -> (
    Arc<mklight_core::resolved::ResolvedSystem>,
    StrayAnalysisSettings,
) {
    let (s, opts) = double_gauss_with_barrels();
    let settings = StrayAnalysisSettings {
        scene: opts.scene,
        detector_margin: opts.detector_margin,
        ..Default::default()
    };
    (Arc::new(resolve(s)), settings)
}

fn sun(angle_y_deg: f64) -> StraySource {
    StraySource::OutOfField {
        angle_x_deg: 0.0,
        angle_y_deg,
        margin: 0.1,
        irradiance: 1.0,
    }
}

fn class_sum(r: &StrayAnalysisResult, mask: ClassMask) -> f64 {
    r.totals
        .iter()
        .filter(|t| mask.contains(t.class))
        .map(|t| t.flux)
        .sum()
}

/// Every unit of emitted flux is detected (in exactly one class), absorbed
/// or escapes; the books close up to the net Monte-Carlo weight adjustment,
/// which is zero within its own standard error.
#[test]
fn energy_closure_and_class_partition() {
    let (rs, base) = barrelled();
    for (name, source) in [
        ("sun 20°", sun(20.0)),
        (
            "field 0",
            StraySource::SystemField {
                index: 0,
                margin: 0.1,
                strength: 1.0,
            },
        ),
    ] {
        let settings = StrayAnalysisSettings {
            source,
            rays: 30_000,
            seed: 9,
            ..base.clone()
        };
        let r = analyze(rs.clone(), &settings, &JobContext::new()).unwrap();
        let e = &r.energy;
        assert!(
            !r.warnings
                .iter()
                .any(|w| w.contains("intersect") || w.contains("edge thickness")),
            "{:?}",
            r.warnings
        );
        assert!(e.is_balanced(), "{}", e.summary());
        let all_classes = class_sum(&r, ClassMask::ALL);
        assert!((all_classes - e.detected).abs() <= 1e-12 * e.detected);
        let not_detected =
            e.absorbed + e.escaped + e.escaped_from_solid + e.truncated + e.sampling_failed;
        let closure = all_classes - (e.emitted - not_detected);
        let zc = closure / e.mc_adjustment_std_error.unwrap();
        println!(
            "{name}: Σ classes {:.6e} W, emitted − absorbed − escaped {:.6e} W, closure {:+.3e} W (z = {zc:+.2}; SE {:.3e} W)\n{}",
            all_classes,
            e.emitted - not_detected,
            closure,
            e.mc_adjustment_std_error.unwrap(),
            e.summary()
        );
        assert!(e.mc_adjustment_std_error.unwrap() > 0.0);
        assert!(zc.abs() < 4.0, "{name}: closure z = {zc}");
        assert!(
            closure.abs()
                < 4.0
                    * e.detected_std_error
                        .unwrap()
                        .max(e.mc_adjustment_std_error.unwrap())
        );
        // The closed barrel leaks nothing out of the solids.
        assert!(e.escaped_from_solid < 1e-4 * e.emitted, "{}", e.summary());
        assert!(e.truncated == 0.0 && e.sampling_failed == 0.0);

        // The stray total is the sum of the included classes, and its map
        // sums to it; each class map sums to its class total.
        assert_eq!(r.classes, ClassMask::ALL_STRAY);
        let stray = class_sum(&r, ClassMask::ALL_STRAY);
        assert!((r.stray_flux - stray).abs() <= 1e-12 * stray.max(1e-300));
        let area = r.detector.half_x * r.detector.half_y * 4.0
            / (r.detector.nx as f64 * r.detector.ny as f64);
        let map_sum = r.total_irradiance.iter().sum::<f64>() * area;
        assert!((map_sum - r.stray_flux).abs() <= 1e-9 * r.stray_flux.max(1e-300));
        assert_eq!(r.maps.len(), 4);
        for m in &r.maps {
            let s = m.irradiance.iter().sum::<f64>() * area;
            let t = r.total_of(m.class).flux;
            assert!((s - t).abs() <= 1e-9 * t.max(1e-300), "{:?}", m.class);
        }
        assert!(r.stray_std_error.unwrap() > 0.0 && r.stray_flux > 0.0);
        assert!(
            r.top_signatures
                .iter()
                .all(|s| s.class != PathClass::Primary)
        );
        // Flux-ordered.
        assert!(r.top_signatures.windows(2).all(|w| w[0].flux >= w[1].flux));

        // With the explicit ghost solver in charge of ghosts, the same run
        // (same random streams) drops exactly the Ghost class from the total.
        let excl = analyze(
            rs.clone(),
            &StrayAnalysisSettings {
                exclude_ghost_class: true,
                ..settings.clone()
            },
            &JobContext::new(),
        )
        .unwrap();
        assert_eq!(excl.classes, ClassMask::STRAY_WITH_GHOST);
        assert_eq!(excl.totals.len(), 5);
        for (a, b) in r.totals.iter().zip(&excl.totals) {
            assert_eq!(a.flux, b.flux);
            assert_eq!(b.included, ClassMask::STRAY_WITH_GHOST.contains(b.class));
        }
        let ghost = r.total_of(PathClass::Ghost).flux;
        assert!(
            (excl.stray_flux + ghost - r.stray_flux).abs() <= 1e-12 * r.stray_flux,
            "{} + {} vs {}",
            excl.stray_flux,
            ghost,
            r.stray_flux
        );
        assert!(excl.maps.iter().all(|m| m.class != PathClass::Ghost));
        assert!(
            excl.top_signatures
                .iter()
                .all(|s| s.class != PathClass::Ghost)
        );
        // Partition of everything detected: sequential + ghost solver + stray.
        let primary = r.total_of(PathClass::Primary).flux;
        assert!((primary + ghost + excl.stray_flux - e.detected).abs() <= 1e-12 * e.detected);
        if name.starts_with("sun") {
            // The sun is outside the field: nothing is imaged as designed,
            // and the barrel and rims scatter light onto the detector.
            assert_eq!(primary, 0.0);
            assert!(r.total_of(PathClass::Scatter).flux > 0.0);
            assert!(
                r.top_signatures
                    .iter()
                    .any(|s| s.label.contains("Ms[barrel 0]")),
                "{:?}",
                r.top_signatures
                    .iter()
                    .map(|s| &s.label)
                    .collect::<Vec<_>>()
            );
        } else {
            assert!(primary > 0.5 * r.source.reference_flux);
        }
    }
}

/// Signature fluxes add up to the class totals.
#[test]
fn signatures_partition_each_class() {
    let (rs, base) = barrelled();
    let sys = mklight_stray::StraySystem::new(rs, &base.system_options()).unwrap();
    let beam = sys.out_of_field_source(0.0, 16.0, 0.1, 1.0).unwrap();
    let set = mklight_stray::StraySettings {
        rays: 20_000,
        seed: 4,
        ..Default::default()
    };
    let r = run(&sys.scene, &beam.spec, &set);
    for c in PathClass::ALL {
        let s: f64 = r
            .signatures
            .iter()
            .filter(|s| s.class == c)
            .map(|s| s.flux)
            .sum();
        let t = r.energy.detected_in(c);
        assert!((s - t).abs() <= 1e-9 * t.max(1e-300), "{c:?}: {s} vs {t}");
    }
    let n: u64 = r.signatures.iter().map(|s| s.hits).sum();
    let hits: u64 = r.energy.detected_hits_by_class.iter().map(|h| h.1).sum();
    assert_eq!(n, hits);
    // Every signature but the designed one has events, and labels match.
    for s in &r.signatures {
        assert_eq!(s.events.len(), s.signature.len());
        assert_eq!(s.label == "designed", s.events.is_empty() && !s.overflow);
    }
}

fn normalised(mut r: StrayAnalysisResult) -> StrayAnalysisResult {
    r.run_info.started_unix_ms = 0;
    r.run_info.duration_ms = 0.0;
    r
}

#[test]
fn analysis_is_reproducible_for_any_thread_count() {
    let (rs, base) = barrelled();
    let settings = StrayAnalysisSettings {
        source: sun(18.0),
        rays: 6000,
        seed: 77,
        transport: mklight_stray::TransportOptions {
            chunk_rays: 256,
            ..Default::default()
        },
        ..base
    };
    let go = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| normalised(analyze(rs.clone(), &settings, &JobContext::new()).unwrap()))
    };
    let a = go(1);
    let b = go(4);
    let c = go(3);
    assert_eq!(a, b);
    assert_eq!(a, c);
    assert!(a.stray_flux > 0.0);
    assert_eq!(a.run_info.seed, Some(77));
    assert_eq!(a.run_info.analysis, "stray_light");
    assert_eq!(a.run_info.snapshot_hash, rs.snapshot.hash().to_hex());
    // Another seed gives another (statistically equal) answer.
    let d = normalised(
        analyze(
            rs.clone(),
            &StrayAnalysisSettings {
                seed: 78,
                ..settings.clone()
            },
            &JobContext::new(),
        )
        .unwrap(),
    );
    assert_ne!(a.stray_flux, d.stray_flux);
    let zz = z(
        a.stray_flux,
        d.stray_flux,
        a.stray_std_error.unwrap().hypot(d.stray_std_error.unwrap()),
    );
    assert!(zz.abs() < 4.0, "z = {zz}");
}

#[test]
fn progress_and_cancellation() {
    let (rs, base) = barrelled();
    let settings = StrayAnalysisSettings {
        source: sun(20.0),
        rays: 3000,
        ..base
    };
    let ctx = JobContext::with(CancelToken::new(), Progress::default());
    ctx.cancel.cancel();
    assert_eq!(
        analyze(rs.clone(), &settings, &ctx).unwrap_err(),
        MkError::Cancelled
    );
    let ctx = JobContext::new();
    let r = analyze(rs.clone(), &settings, &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    assert_eq!(r.energy.rays, 3000);
    // A nested context keeps the analysis inside its slice of the bar.
    let outer = JobContext::new();
    let sub = outer.subrange(0.0, 0.5);
    let sys = mklight_stray::StraySystem::new(rs, &settings.system_options()).unwrap();
    analyze_system(&sys, &settings, &sub).unwrap();
    assert_eq!(outer.progress.fraction(), 0.5);
}

/// The delta-method standard errors of centroid and RMS radius, and the
/// flux standard error, match the scatter of independent runs.
#[test]
fn moment_standard_errors_are_calibrated() {
    let sys = stray_system(
        mklight_core::samples::cooke_triplet(),
        &no_scatter_options(),
    );
    let beam = sys.system_field_source(1, 0.1, 1.0).unwrap();
    let k = 16;
    let mut vals: Vec<[f64; 3]> = Vec::new();
    let mut reported = [0.0; 3];
    for seed in 0..k {
        let set = mklight_stray::StraySettings {
            rays: 4000,
            seed: 500 + seed,
            scatter_reflection: false,
            scatter_transmission: false,
            signatures: false,
            ..Default::default()
        };
        let r = run(&sys.scene, &beam.spec, &set);
        let m = r.detector.moments_of(PathClass::Primary);
        vals.push([
            r.energy.detected_in(PathClass::Primary),
            m.centroid[1],
            m.rms_radius,
        ]);
        let se = [
            r.energy.std_error_in(PathClass::Primary).unwrap(),
            m.centroid_std_error[1].unwrap(),
            m.rms_radius_std_error.unwrap(),
        ];
        for j in 0..3 {
            reported[j] += se[j] / k as f64;
        }
    }
    for (j, what) in ["flux", "centroid y", "RMS radius"].iter().enumerate() {
        let mean = vals.iter().map(|v| v[j]).sum::<f64>() / k as f64;
        let sd = (vals.iter().map(|v| (v[j] - mean).powi(2)).sum::<f64>() / (k - 1) as f64).sqrt();
        let ratio = sd / reported[j];
        println!(
            "{what}: run-to-run SD {sd:.3e}, mean reported SE {:.3e} (ratio {ratio:.2})",
            reported[j]
        );
        // 16 runs: the sample SD is within ±45 % of the truth with high
        // probability.
        assert!((0.55..1.6).contains(&ratio), "{what}: ratio {ratio}");
    }
}

#[test]
fn sources_fill_pupil_and_front_aperture() {
    let (rs, base) = barrelled();
    let sys = mklight_stray::StraySystem::new(rs, &base.system_options()).unwrap();
    let pupil = *sys.tracer().pupil();
    let beam = sys.system_field_source(1, 0.25, 2.0).unwrap();
    match beam.spec {
        mklight_stray::SourceSpec::CollimatedThrough {
            direction,
            center,
            normal,
            radius,
            power,
            ..
        } => {
            assert!((radius - 1.25 * pupil.r_ep).abs() < 1e-12);
            assert_eq!(normal, mklight_core::math::Vec3::Z);
            assert!((center.z - pupil.z_ep).abs() < 1e-12);
            assert!((direction.y / direction.z - 10f64.to_radians().tan()).abs() < 1e-12);
            let want = 2.0 * std::f64::consts::PI * radius * radius * direction.z;
            assert!((power - want).abs() < 1e-9 * want);
            assert!((beam.reference_flux * 1.25 * 1.25 - power).abs() < 1e-9 * power);
        }
        ref s => panic!("{s:?}"),
    }
    let sunb = sys.out_of_field_source(0.0, 30.0, 0.1, 1.0).unwrap();
    match sunb.spec {
        mklight_stray::SourceSpec::Collimated {
            radius,
            center,
            direction,
            start_distance,
            ..
        } => {
            // The disk covers the projection of every primitive's bounding
            // box along the beam, and is no larger than needed (some corner
            // sits on the un-margined rim).
            let mut far = 0.0f64;
            for p in sys.scene.primitives() {
                let b = p.aabb();
                for i in 0..8 {
                    let q = mklight_core::math::Vec3::new(
                        if i & 1 == 0 { b.min.x } else { b.max.x },
                        if i & 2 == 0 { b.min.y } else { b.max.y },
                        if i & 4 == 0 { b.min.z } else { b.max.z },
                    ) - center;
                    let perp = q - direction * q.dot(direction);
                    far = far.max(perp.norm());
                }
            }
            assert!(
                (far - radius / 1.1).abs() < 1e-9 * radius,
                "{far} vs {radius}"
            );
            // It starts in front of everything.
            let (sc, sr) = sys.scene.bounding_sphere();
            let start = center - direction * start_distance;
            assert!((start - sc).dot(direction) < -sr);
        }
        ref s => panic!("{s:?}"),
    }
    assert!(sys.out_of_field_source(0.0, 95.0, 0.1, 1.0).is_err());
    assert!(sys.system_field_source(1, -0.1, 1.0).is_err());
}

#[test]
fn settings_round_trip_and_defaults() {
    let s: StrayAnalysisSettings = serde_json::from_str("{}").unwrap();
    assert_eq!(s, StrayAnalysisSettings::default());
    assert_eq!(s.effective_classes(), ClassMask::ALL_STRAY);
    let t = StrayAnalysisSettings {
        source: sun(25.0),
        exclude_ghost_class: true,
        ..Default::default()
    };
    let back: StrayAnalysisSettings =
        serde_json::from_str(&serde_json::to_string(&t).unwrap()).unwrap();
    assert_eq!(back, t);
    assert_eq!(back.effective_classes(), ClassMask::STRAY_WITH_GHOST);
    assert!(
        StrayAnalysisSettings {
            rays: 0,
            ..Default::default()
        }
        .validate()
        .is_err()
    );
}

/// Settings that would exhaust memory, produce non-finite results or hide a
/// typo are rejected; a zero reference flux gives no fraction (and the
/// result still round-trips through JSON).
#[test]
fn settings_validation() {
    let bad = |s: StrayAnalysisSettings| s.validate().is_err();
    assert!(bad(StrayAnalysisSettings {
        detector_pixels: (100_000, 100_000),
        ..Default::default()
    }));
    assert!(bad(StrayAnalysisSettings {
        detector_pixels: (2048, 1024),
        ..Default::default()
    }));
    assert!(
        StrayAnalysisSettings {
            detector_pixels: (1024, 1024),
            ..Default::default()
        }
        .validate()
        .is_ok()
    );
    for typo in [
        r#"{"ray": 10}"#,
        r#"{"transport": {"mode": {"mode": "analog"}, "max_event": 3}}"#,
        r#"{"scene": {"include_mechanic": false}}"#,
        r#"{"source": {"type": "out_of_field", "angle_y_deg": 20, "margin": 0.1, "irradiance": 1, "angle_x": 0}}"#,
    ] {
        assert!(
            serde_json::from_str::<StrayAnalysisSettings>(typo).is_err(),
            "{typo}"
        );
    }

    let (rs, base) = barrelled();
    let sys = mklight_stray::StraySystem::new(rs, &base.system_options()).unwrap();
    for source in [
        StraySource::OutOfField {
            angle_x_deg: 0.0,
            angle_y_deg: 20.0,
            margin: 0.1,
            irradiance: f64::INFINITY,
        },
        StraySource::SystemField {
            index: 0,
            margin: 0.1,
            strength: f64::INFINITY,
        },
        StraySource::Field {
            hx: 0.0,
            hy: 0.0,
            margin: 0.1,
            strength: f64::NAN,
        },
    ] {
        assert!(source.resolve(&sys).is_err(), "{source:?}");
    }
    let beam = sys.out_of_field_source(0.0, 20.0, 0.1, 1.0).unwrap();
    let custom = |reference_flux| StrayAnalysisSettings {
        source: StraySource::Custom {
            spec: beam.spec.clone(),
            reference_flux,
        },
        rays: 2000,
        ..base.clone()
    };
    assert!(analyze_system(&sys, &custom(f64::NAN), &JobContext::new()).is_err());
    let r = analyze_system(&sys, &custom(0.0), &JobContext::new()).unwrap();
    assert!(r.stray_flux > 0.0);
    assert_eq!(r.stray_fraction, None);
    assert!(r.totals.iter().all(|t| t.fraction.is_none()));
    let back: StrayAnalysisResult =
        serde_json::from_str(&serde_json::to_string(&r).unwrap()).unwrap();
    assert_eq!(back.stray_fraction, None);
    assert_eq!(back.stray_flux, r.stray_flux);
}

/// `analyze_system` keeps the scene the system was built with; settings
/// that ask for another detector get a warning, and `RunInfo` records the
/// options actually used.
#[test]
fn analyze_system_reports_the_scene_it_used() {
    let (rs, base) = barrelled();
    let sys = mklight_stray::StraySystem::new(rs, &base.system_options()).unwrap();
    let settings = StrayAnalysisSettings {
        source: sun(20.0),
        rays: 2000,
        ..base.clone()
    };
    let same = analyze_system(&sys, &settings, &JobContext::new()).unwrap();
    assert!(
        !same.warnings.iter().any(|w| w.contains("differ")),
        "{:?}",
        same.warnings
    );
    let other = StrayAnalysisSettings {
        detector_pixels: (7, 9),
        ..settings
    };
    let r = analyze_system(&sys, &other, &JobContext::new()).unwrap();
    assert_eq!((r.detector.nx, r.detector.ny), (128, 128));
    assert!(
        r.warnings.iter().any(|w| w.contains("differ")),
        "{:?}",
        r.warnings
    );
    let used = &r.run_info.settings["system_options"]["scene"]["detector_pixels"];
    assert_eq!(used, &serde_json::json!([128, 128]));
    assert_eq!(
        r.run_info.settings["settings"]["detector_pixels"],
        serde_json::json!([7, 9])
    );
}

/// Building a system and the sequential reference honour cancellation and
/// report monotone progress up to 1.
#[test]
fn system_build_and_reference_job_control() {
    let (rs, base) = barrelled();
    let opts = base.system_options();
    let ctx = JobContext::with(CancelToken::new(), Progress::default());
    ctx.cancel.cancel();
    assert_eq!(
        mklight_stray::StraySystem::build(rs.clone(), &opts, &ctx).unwrap_err(),
        MkError::Cancelled
    );
    let ctx = JobContext::new();
    let sys = mklight_stray::StraySystem::build(rs, &opts, &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    let beam = sys.system_field_source(0, 0.1, 1.0).unwrap();
    let cancelled = JobContext::with(CancelToken::new(), Progress::default());
    cancelled.cancel.cancel();
    assert_eq!(
        mklight_stray::primary_reference(&sys, &beam.spec, 64, &cancelled).unwrap_err(),
        MkError::Cancelled
    );
    let half = JobContext::new();
    mklight_stray::primary_reference(&sys, &beam.spec, 64, &half.subrange(0.0, 0.5)).unwrap();
    assert_eq!(half.progress.fraction(), 0.5);
    // The retaining ring in front of L1 (inner radius = L1's clear radius,
    // 0.5 mm in front of its vertex) clips the oblique 10° beam, which
    // crosses its plane displaced from where it meets the convex surface.
    // The Monte-Carlo Primary agrees with the obstruction-checked reference,
    // not with the purely sequential flux.
    assert_eq!(sys.obstruction[0], Some(0.0));
    let o = sys.obstruction[1].unwrap();
    assert!(o > 0.005 && o < 0.1, "{o}");
    assert!(sys.warnings.iter().any(|w| w.starts_with("field 1:")));
    let beam = sys.system_field_source(1, 0.1, 1.0).unwrap();
    let reference =
        mklight_stray::primary_reference(&sys, &beam.spec, 300, &JobContext::new()).unwrap();
    let set = mklight_stray::StraySettings {
        rays: 120_000,
        seed: 61,
        ..Default::default()
    };
    let r = run(&sys.scene, &beam.spec, &set);
    let mc = r.energy.detected_in(PathClass::Primary);
    let se = r.energy.std_error_in(PathClass::Primary).unwrap();
    let (zf, zs) = (
        z(mc, reference.flux, se),
        z(mc, reference.sequential_flux, se),
    );
    println!(
        "barrelled double Gauss 10°: obstruction {:.2} %, MC Primary z = {zf:+.2} vs checked, {zs:+.2} vs purely sequential",
        100.0 * reference.obstruction()
    );
    assert!(zf.abs() < 3.0, "z = {zf}");
    assert!(zs < -5.0, "z = {zs}");
}

/// A run of one source ray has no standard errors: they are `None` (JSON
/// `null`), not a number, and the result reads back from its own JSON, as
/// does a run whose errors are all defined.
#[test]
fn undefined_errors_are_null_and_results_round_trip() {
    let (rs, base) = barrelled();
    let round_trip = |r: &StrayAnalysisResult| {
        let json = serde_json::to_string(r).unwrap();
        let back: StrayAnalysisResult = serde_json::from_str(&json).unwrap();
        // The compact signatures are not serialised.
        let mut expected = r.clone();
        for s in &mut expected.top_signatures {
            s.signature = Default::default();
        }
        assert_eq!(back, expected);
        json
    };

    let one = StrayAnalysisSettings {
        source: StraySource::SystemField {
            index: 0,
            margin: 0.1,
            strength: 1.0,
        },
        rays: 1,
        seed: 3,
        classes: ClassMask::ALL,
        detector_pixels: (4, 4),
        ..base.clone()
    };
    let r = analyze(rs.clone(), &one, &JobContext::new()).unwrap();
    assert_eq!(r.energy.rays, 1);
    assert_eq!(r.stray_std_error, None);
    assert_eq!(r.total_irradiance.len(), 16);
    assert_eq!(r.total_irradiance_std_error, None);
    assert_eq!(r.maps.len(), 5);
    assert!(r.maps.iter().all(|m| m.irradiance_std_error.is_none()));
    assert!(r.totals.iter().all(|t| t.std_error.is_none()));
    assert!(r.top_signatures.iter().all(|s| s.std_error.is_none()));
    assert_eq!(r.energy.detected_std_error, None);
    assert_eq!(r.energy.mc_adjustment_std_error, None);
    let json = round_trip(&r);
    assert!(json.contains("\"stray_std_error\":null"));
    assert!(json.contains("\"total_irradiance_std_error\":null"));

    let many = StrayAnalysisSettings { rays: 2000, ..one };
    let r = analyze(rs, &many, &JobContext::new()).unwrap();
    assert!(r.stray_std_error.unwrap() > 0.0);
    assert_eq!(r.total_irradiance_std_error.as_ref().unwrap().len(), 16);
    for m in &r.maps {
        assert_eq!(m.irradiance_std_error.as_ref().unwrap().len(), 16);
    }
    for t in &r.totals {
        assert!(t.std_error.is_some(), "{:?}", t.class);
        // The error of an RMS radius is undefined exactly when it is zero
        // although the class detected something (a single hit).
        assert_eq!(
            t.moments.rms_radius_std_error.is_none(),
            t.moments.flux > 0.0 && t.moments.rms_radius == 0.0,
            "{:?}",
            t.class
        );
    }
    round_trip(&r);
}
