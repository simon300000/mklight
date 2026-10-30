//! The solver partition (CONVENTIONS §10, `docs/STRAY.md` "Partition"):
//! the sequential `Primary`, the explicit ghost solver's ghosts and the
//! stray solver run with the `Ghost` class excluded add up to everything
//! the detector receives, with no path counted twice, because the stray
//! scene clips at exactly the ghost run's apertures.

mod common;

use std::sync::Arc;

use common::*;
use mklight_core::JobContext;
use mklight_core::model::*;
use mklight_ghost::{
    ClipMode, EnumerationSettings, GhostModel, GhostSettings, IrradianceSettings, ghost_irradiance,
};
use mklight_stray::{
    PathClass, SplitMode, StrayAnalysisSettings, StraySource, StraySystem, TransportOptions,
    analyze_system, primary_reference,
};

/// Every refracting surface coated with an ideal 5 % reflector (no
/// absorption, the same `R` for s and p), and a sensor that bounds the
/// detector of both solvers.
/// One wavelength keeps the test fast.
fn coated(s: OpticalSystem, sensor_half: f64) -> OpticalSystem {
    coated_with(s, sensor_half, 0.05)
}

fn coated_with(mut s: OpticalSystem, sensor_half: f64, r: f64) -> OpticalSystem {
    one_wavelength(&mut s);
    s.coatings.insert(
        "R".into(),
        CoatingDef::Ideal {
            reflectance: r,
            transmittance: 1.0 - r,
        },
    );
    let image = s.surfaces.len() - 1;
    for surf in &mut s.surfaces[1..image] {
        surf.coating = Some("R".into());
    }
    let pixels = (2.0 * sensor_half / 0.01).round() as u32;
    s.sensor = Some(Sensor {
        pixels_x: pixels,
        pixels_y: pixels,
        pitch_x: 0.01,
        pitch_y: 0.01,
        max_cra_deg: None,
        spectral_response: Vec::new(),
    });
    s
}

fn settings(field: usize, seed: u64) -> StrayAnalysisSettings {
    let opts = no_scatter_options();
    StrayAnalysisSettings {
        source: StraySource::SystemField {
            index: field,
            margin: 0.1,
            strength: 1.0,
        },
        rays: 20_000,
        seed,
        // Default classes: everything but Primary.
        scene: opts.scene,
        semi_diameters: opts.semi_diameters,
        detector_margin: opts.detector_margin,
        // Every specular branch is followed, so the Ghost class is a smooth
        // function of the source ray and its error is pupil sampling only.
        transport: TransportOptions {
            mode: SplitMode::Split {
                max_splits: 8,
                rare_floor: 0.1,
            },
            roulette_threshold: 1e-6,
            ..Default::default()
        },
        ..Default::default()
    }
}

fn ghost_model(sys: &StraySystem, clip: ClipMode) -> GhostModel {
    GhostModel::new(
        Arc::new(sys.tracer().clone()),
        GhostSettings {
            clip,
            ..Default::default()
        },
    )
    .unwrap()
}

/// Ghost solver totals relative to the entrance-pupil flux: (primary,
/// ghosts of order 2 and 4), on the sensor, with default clipping.
fn ghost_run(sys: &StraySystem, field: usize) -> (f64, f64, mklight_ghost::GhostCoverage) {
    ghost_run_to(sys, field, 4)
}

/// As [`ghost_run`] with ghosts up to `max_reflections`.
fn ghost_run_to(
    sys: &StraySystem,
    field: usize,
    max_reflections: u32,
) -> (f64, f64, mklight_ghost::GhostCoverage) {
    let model = ghost_model(sys, ClipMode::ClearApertures { margin_mm: 0.0 });
    let r = ghost_irradiance(
        &model,
        &IrradianceSettings {
            enumeration: EnumerationSettings {
                max_reflections,
                max_higher_order_paths: 100_000,
                ..Default::default()
            },
            fields: vec![field],
            pupil_samples: 120,
            overfill: 1.3,
            map_bins: [8, 8],
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap();
    assert!(!r.truncated);
    let on_map = |p: &mklight_ghost::PathIrradiance| p.fields[0].flux_on_map;
    let ghosts = r.ghosts.iter().map(on_map).sum::<f64>();
    (on_map(&r.primary), ghosts, r.coverage)
}

fn check(name: &str, base: OpticalSystem, fields: &[usize]) {
    for (i, &k) in fields.iter().enumerate() {
        let set = settings(k, 7 + i as u64);
        let rs = Arc::new(resolve(base.clone()));
        let sys = StraySystem::build(rs, &set.system_options(), &JobContext::new()).unwrap();
        let (p_ghost, g, cov) = ghost_run(&sys, k);
        // The default ghost run and the stray scene clip identically.
        sys.check_ghost_coverage(&cov).unwrap();

        // All classes, and the run combined with the ghost solver.
        let all = analyze_system(&sys, &set, &JobContext::new()).unwrap();
        let excl = analyze_system(
            &sys,
            &StrayAnalysisSettings {
                exclude_ghost_class: true,
                ghost_coverage: Some(cov.clone()),
                ..set.clone()
            },
            &JobContext::new(),
        )
        .unwrap();
        let reference = all.source.reference_flux;
        let frac = |c: PathClass| all.total_of(c).flux / reference;
        let se = |c: PathClass| all.total_of(c).std_error / reference;
        let detected: f64 = PathClass::ALL.iter().map(|c| frac(*c)).sum();
        let stray = excl.stray_flux / reference;
        let stray_with_ghost = all.stray_flux / reference;
        let primary = primary_reference(&sys, &all.source.spec, 300, &JobContext::new())
            .unwrap()
            .flux
            / reference;
        // The exclusion only removes the covered ghosts from the tally
        // (same streams); orders above 4 stay, and are negligible at 5 %.
        let owned = frac(PathClass::Scatter) + frac(PathClass::Mechanical) + frac(PathClass::Sneak);
        let uncovered = excl.energy.detected_uncovered_ghost / reference;
        assert!(
            (stray - owned - uncovered).abs() <= 1e-12 * detected,
            "{name}: {stray} vs {owned} + {uncovered}"
        );
        assert!(
            uncovered < 1e-2 * frac(PathClass::Ghost),
            "{name}: {uncovered}"
        );

        // The Primary term: sequential (obstruction-checked) reference vs
        // the Monte-Carlo Primary class.
        let z_primary = (frac(PathClass::Primary) - primary) / se(PathClass::Primary);
        // Everything else the detector receives must be the ghost solver's
        // ghosts plus the stray total without the Ghost class. The
        // Monte-Carlo error of that remainder is the error of its classes.
        let rest = detected - frac(PathClass::Primary);
        let se_rest = [
            PathClass::Ghost,
            PathClass::Scatter,
            PathClass::Mechanical,
            PathClass::Sneak,
        ]
        .iter()
        .map(|c| se(*c).powi(2))
        .sum::<f64>()
        .sqrt();
        let z_sum = (rest - (g + stray)) / se_rest;
        let z_double = (rest - (g + stray_with_ghost)) / se_rest;
        let z_ghost = (frac(PathClass::Ghost) - g) / se(PathClass::Ghost);
        println!(
            "{name} field {k}: detected {detected:.6} = primary {primary:.6} (MC z = {z_primary:+.2}; \
             ghost-solver primary {p_ghost:.6}) + ghosts {g:.4e} (stray Ghost class {:.4e} ± {:.1e}, \
             z = {z_ghost:+.2}) + stray {stray:.3e}: z = {z_sum:+.2}; with the Ghost class counted \
             twice z = {z_double:+.1}",
            frac(PathClass::Ghost),
            se(PathClass::Ghost)
        );
        assert!(
            (p_ghost / primary - 1.0).abs() < 5e-3,
            "{name}: {p_ghost} vs {primary}"
        );
        assert!(
            z_primary.abs() < 4.0,
            "{name} field {k}: primary z = {z_primary}"
        );
        assert!(z_ghost.abs() < 4.0, "{name} field {k}: ghost z = {z_ghost}");
        assert!(z_sum.abs() < 4.0, "{name} field {k}: partition z = {z_sum}");
        // The check can fail: counting the Ghost class in both solvers.
        assert!(z_double < -20.0, "{name} field {k}: {z_double}");

        // A ghost run with wider apertures (0.5 mm margin) does not
        // partition with this scene and is refused.
        let wide =
            ghost_model(&sys, ClipMode::ClearApertures { margin_mm: 0.5 }).coverage(cov.paths);
        assert!(sys.check_ghost_coverage(&wide).is_err());
        assert!(
            analyze_system(
                &sys,
                &StrayAnalysisSettings {
                    exclude_ghost_class: true,
                    ghost_coverage: Some(wide),
                    ..set.clone()
                },
                &JobContext::new(),
            )
            .is_err()
        );
    }
}

#[test]
fn coated_singlet_partition_does_not_double_count() {
    check(
        "coated singlet",
        coated(mklight_core::samples::singlet(), 12.0),
        &[0, 2],
    );
}

#[test]
fn coated_cooke_triplet_partition_does_not_double_count() {
    check(
        "coated Cooke triplet",
        coated(mklight_core::samples::cooke_triplet(), 22.0),
        &[0, 1],
    );
}

/// Ghost orders the ghost run did not trace stay with the stray solver
/// (CONVENTIONS §10): on the Cooke triplet with 30 % reflectors and a
/// ghost run to 2 reflections, the 4th and higher orders are about a
/// quarter of the ghost light. With the run's coverage the stray total keeps them and the
/// partition closes; dropping the whole `Ghost` class (no coverage given)
/// loses them.
#[test]
fn ghost_orders_outside_the_ghost_run_stay_with_the_stray_solver() {
    let base = coated_with(mklight_core::samples::cooke_triplet(), 22.0, 0.3);
    let set = settings(0, 11);
    let sys = StraySystem::build(
        Arc::new(resolve(base)),
        &set.system_options(),
        &JobContext::new(),
    )
    .unwrap();
    let (_, g2, cov) = ghost_run_to(&sys, 0, 2);
    assert!(cov.paths.iter().all(|p| p.order() == 2), "{:?}", cov.paths);
    let run = |coverage: Option<mklight_ghost::GhostCoverage>| {
        analyze_system(
            &sys,
            &StrayAnalysisSettings {
                exclude_ghost_class: true,
                ghost_coverage: coverage,
                ..set.clone()
            },
            &JobContext::new(),
        )
        .unwrap()
    };
    let all = analyze_system(&sys, &set, &JobContext::new()).unwrap();
    let with = run(Some(cov.clone()));
    let without = run(None);
    let reference = all.source.reference_flux;
    let frac = |c: PathClass| all.total_of(c).flux / reference;
    let se = |c: PathClass| all.total_of(c).std_error / reference;
    let rest = PathClass::ALL.iter().map(|c| frac(*c)).sum::<f64>() - frac(PathClass::Primary);
    let se_rest = [
        PathClass::Ghost,
        PathClass::Scatter,
        PathClass::Mechanical,
        PathClass::Sneak,
    ]
    .iter()
    .map(|c| se(*c).powi(2))
    .sum::<f64>()
    .sqrt();
    let uncovered = with.energy.detected_uncovered_ghost / reference;
    let z_with = (rest - (g2 + with.stray_flux / reference)) / se_rest;
    let z_without = (rest - (g2 + without.stray_flux / reference)) / se_rest;
    // The uncovered part is the Ghost class minus what the ghost run traced.
    let z_uncovered = (uncovered - (frac(PathClass::Ghost) - g2)) / se(PathClass::Ghost);
    println!(
        "Ghost class {:.4e}, ghost run (2 reflections) {g2:.4e}, uncovered {uncovered:.4e}: \
         partition z = {z_with:+.2} with coverage, {z_without:+.1} without",
        frac(PathClass::Ghost)
    );
    assert!(uncovered > 0.1 * frac(PathClass::Ghost), "{uncovered}");
    assert!(z_uncovered.abs() < 4.0, "{z_uncovered}");
    assert!(z_with.abs() < 4.0, "{z_with}");
    assert!(z_without > 20.0, "{z_without}");
    assert_eq!(without.energy.detected_uncovered_ghost, 0.0);
    // The kept ghosts are in the stray maps' total as well as in the flux.
    let map_total: f64 = with.total_irradiance.iter().sum::<f64>() * with.detector.pixel_area();
    let owned_classes: f64 = with
        .maps
        .iter()
        .flat_map(|m| m.irradiance.iter())
        .sum::<f64>()
        * with.detector.pixel_area();
    assert!((map_total - with.stray_flux).abs() < 1e-12 * with.stray_flux);
    assert!((map_total - owned_classes).abs() < 1e-12 * with.stray_flux);
    let ghost_map = with
        .maps
        .iter()
        .find(|m| m.class == PathClass::Ghost)
        .unwrap();
    let ghost_flux = ghost_map.irradiance.iter().sum::<f64>() * with.detector.pixel_area();
    assert!((ghost_flux - with.energy.detected_uncovered_ghost).abs() < 1e-12 * ghost_flux);
    // Ownership never changes transport, and reduction is independent of
    // worker count. Include the partitioned pixel errors in that check.
    let again = rayon::ThreadPoolBuilder::new()
        .num_threads(2)
        .build()
        .unwrap()
        .install(|| run(Some(cov.clone())));
    assert_eq!(with.maps, again.maps);
    assert_eq!(with.total_irradiance, again.total_irradiance);
    assert_eq!(
        with.total_irradiance_std_error,
        again.total_irradiance_std_error
    );
    assert_eq!(with.energy, again.energy);
    assert_eq!(with.run_info.seed, Some(set.seed));
}

/// A stray scene sized with non-default semi-diameter settings partitions
/// with a ghost model that shares its sizing, not with one that sizes on
/// its own at default settings (the apertures are compared exactly).
#[test]
fn ghost_model_shares_the_stray_sizing() {
    use mklight_core::{CancelToken, MkError, Progress};
    let mut opts = no_scatter_options();
    opts.semi_diameters.azimuths = 8;
    let sys = stray_system(mklight_core::samples::cooke_triplet(), &opts);
    let tracer = Arc::new(sys.tracer().clone());
    let none = std::iter::empty();
    let default = ghost_model(&sys, ClipMode::ClearApertures { margin_mm: 0.0 });
    assert!(
        sys.check_ghost_coverage(&default.coverage(none.clone()))
            .is_err()
    );
    let shared = GhostModel::with_semi_diameters(
        tracer.clone(),
        GhostSettings::default(),
        sys.semi_diameters.clone(),
    )
    .unwrap();
    sys.check_ghost_coverage(&shared.coverage(none.clone()))
        .unwrap();
    assert_eq!(shared.semi_diameters(), &sys.semi_diameters);
    // Sizing with the same settings gives the same apertures, and can be
    // cancelled.
    let sized = GhostModel::sized(
        tracer.clone(),
        GhostSettings::default(),
        &opts.semi_diameters,
        &JobContext::new(),
    )
    .unwrap();
    sys.check_ghost_coverage(&sized.coverage(none)).unwrap();
    let cancel = CancelToken::new();
    cancel.cancel();
    let r = GhostModel::sized(
        tracer.clone(),
        GhostSettings::default(),
        &opts.semi_diameters,
        &JobContext::with(cancel, Progress::default()),
    );
    assert!(matches!(r, Err(MkError::Cancelled)));
    // Semi-diameters of another system are refused.
    let mut other = sys.semi_diameters.clone();
    other.values.pop();
    assert!(GhostModel::with_semi_diameters(tracer, GhostSettings::default(), other).is_err());
}

/// Compare order 2 only: at 30 % reflectance higher orders are measurable.
/// Replaying the exact launch rays in the sequence tracer removes all
/// quadrature/sampling differences, isolating transport and its errors.
fn high_reflectance(rays: u64, seeds: &[u64]) {
    let set = settings(0, 7);
    let sys = stray_system(
        coated_with(mklight_core::samples::singlet(), 12.0, 0.3),
        &set.system_options(),
    );
    let model = ghost_model(&sys, ClipMode::ClearApertures { margin_mm: 0.0 });
    let path = mklight_ghost::GhostPath::new(vec![2, 1]);
    let steps = model.check_path(&path).unwrap();
    let frame = mklight_trace::trace::start_frame(sys.resolved());
    for field in [0, 2] {
        let grid = |n| {
            ghost_irradiance(
                &model,
                &IrradianceSettings {
                    paths: Some(vec![path.clone()]),
                    fields: vec![field],
                    pupil_samples: n,
                    overfill: 1.3,
                    map_bins: [8, 8],
                    ..Default::default()
                },
                &JobContext::new(),
            )
            .unwrap()
            .total_ghost_map
            .flux()
        };
        let (coarse, reference) = (grid(120), grid(480));
        // The old centre-only rule aliases the sensor boundary: 1.6 %
        // difference here, even though 120 and 240 gave identical answers.
        assert!(
            (coarse / reference - 1.0).abs() < 0.002,
            "field {field}: {coarse} vs {reference}"
        );
        let (mut delta, mut variance) = (0.0, 0.0);
        for &seed in seeds {
            let r = analyze_system(
                &sys,
                &StrayAnalysisSettings {
                    rays,
                    seed,
                    source: settings(field, seed).source,
                    exclude_ghost_class: true,
                    ghost_coverage: Some(model.coverage([path.clone()])),
                    ..set.clone()
                },
                &JobContext::new(),
            )
            .unwrap();
            let (mut sum, mut sq) = (
                mklight_core::math::KahanSum::default(),
                mklight_core::math::KahanSum::default(),
            );
            for i in 0..rays {
                let mut rng = mklight_core::rng::Philox::new(seed, i);
                // One wavelength: transport consumes no wavelength draw.
                let ray = r.source.spec.sample(&mut rng);
                let start = mklight_trace::Ray::new(
                    frame.to_local_point(ray.origin),
                    frame.to_local_dir(ray.dir),
                );
                let hit = model.trace_steps(&start, 0, &steps, None);
                if hit.status.is_ok() && hit.ray.pos.x.abs() < 12.0 && hit.ray.pos.y.abs() < 12.0 {
                    let w = hit.weight() * r.source.spec.power() / rays as f64;
                    sum.add(w);
                    sq.add(w * w);
                }
            }
            let paired = sum.value();
            let n = rays as f64;
            let paired_se = (n / (n - 1.0) * (sq.value() - paired * paired / n)).sqrt();
            let e = &r.energy;
            assert!((paired / e.detected_covered_ghost - 1.0).abs() < 1e-10);
            assert!((paired_se / e.detected_covered_ghost_std_error - 1.0).abs() < 1e-10);
            let (mc, se) = (
                paired / r.source.reference_flux,
                paired_se / r.source.reference_flux,
            );
            let z = (mc - reference) / se;
            println!(
                "30 % singlet field {field}, seed {seed}, {rays} rays: {mc:.8e} ± {se:.2e}, reference {reference:.8e}, z={z:+.3}"
            );
            assert!(z.abs() < 3.0, "field {field}, seed {seed}: z={z}");
            delta += mc - reference;
            variance += se * se;
        }
        let z_mean = delta / variance.sqrt();
        assert!(z_mean.abs() < 3.0, "field {field}: mean z={z_mean}");
    }
}

#[test]
fn high_reflectance_ghost_flux_has_no_sampling_bias() {
    high_reflectance(60_000, &[7, 8, 9]);
}

#[test]
#[ignore = "large independent-seed ghost flux cross-check"]
fn high_reflectance_ghost_flux_large_run() {
    high_reflectance(2_000_000, &[7, 8, 9, 10]);
}
