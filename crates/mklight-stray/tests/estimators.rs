//! Estimator quality: variance of the default branch selection on a long
//! stack of interfaces, exact standard errors, leak reporting and run
//! metadata.

mod common;

use common::*;
use mklight_core::math::Vec3;
use mklight_core::model::*;
use mklight_stray::{PathClass, SourceSpec, SplitMode, StraySettings};

fn axial_beam(radius: f64) -> SourceSpec {
    SourceSpec::Collimated {
        direction: Vec3::Z,
        center: Vec3::ZERO,
        radius,
        start_distance: 30.0,
        power: 1.0,
    }
}

/// Eight bare windows (16 interfaces), each face with 0.2 % Lambertian scatter.
fn window_stack() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    one_wavelength(&mut s);
    let mut surfs = vec![Surface::new(
        0,
        SurfaceShape::Plane,
        f64::INFINITY,
        MaterialRef::Air,
    )];
    for k in 0..8u64 {
        let mut a = Surface::new(
            2 * k + 1,
            SurfaceShape::Plane,
            3.0,
            MaterialRef::Constant { n: 1.5 },
        );
        let mut b = Surface::new(2 * k + 2, SurfaceShape::Plane, 3.0, MaterialRef::Air);
        for x in [&mut a, &mut b] {
            fixed(x, 10.0);
            x.scatter = Some(ScatterModel::Lambertian { tis: 0.002 });
        }
        surfs.push(a);
        surfs.push(b);
    }
    let mut img = Surface::new(17, SurfaceShape::Plane, 0.0, MaterialRef::Air);
    fixed(&mut img, 15.0);
    surfs.push(img);
    s.surfaces = surfs;
    s.stop = 1;
    s
}

struct Stats {
    /// Mean over seeds and the standard error of that mean.
    mean: f64,
    se_of_mean: f64,
    /// Mean of the per-run reported standard errors.
    reported: f64,
}

fn stats(sc: &mklight_stray::Scene, mode: SplitMode, class: PathClass, seeds: u64) -> Stats {
    let mut reported = 0.0;
    let (mean, se_of_mean) = seeds_mean(seeds, |seed| {
        let r = run(
            sc,
            &axial_beam(3.0),
            &StraySettings {
                rays: 4000,
                seed,
                mode,
                ..Default::default()
            },
        );
        reported += r.energy.std_error_in(class).unwrap() / seeds as f64;
        r.energy.detected_in(class)
    });
    Stats {
        mean,
        se_of_mean,
        reported,
    }
}

/// The default rare-branch forcing must stay close to analog transport on
/// the designed path (no exponential weight inflation over many interfaces)
/// and must beat it on scatter, without bias.
#[test]
fn default_weighting_is_efficient_on_many_interfaces() {
    let sc = scene(window_stack());
    let k = 10;
    let a_p = stats(&sc, SplitMode::Analog, PathClass::Primary, k);
    let d_p = stats(&sc, SplitMode::default(), PathClass::Primary, k);
    let a_s = stats(&sc, SplitMode::Analog, PathClass::Scatter, k);
    let d_s = stats(&sc, SplitMode::default(), PathClass::Scatter, k);
    for (a, d, what) in [(&a_p, &d_p, "primary"), (&a_s, &d_s, "scatter")] {
        let sigma = a.se_of_mean.hypot(d.se_of_mean);
        assert!(
            (a.mean - d.mean).abs() < 5.0 * sigma,
            "{what}: analog {} vs default {} (σ {sigma})",
            a.mean,
            d.mean
        );
    }
    assert!(
        d_p.reported < 2.5 * a_p.reported,
        "primary SE: default {} vs analog {}",
        d_p.reported,
        a_p.reported
    );
    assert!(
        d_s.reported < 0.6 * a_s.reported,
        "scatter SE: default {} vs analog {}",
        d_s.reported,
        a_s.reported
    );
    // The reported per-run standard error agrees with the spread over seeds.
    for (s, what) in [(&a_p, "analog primary"), (&d_s, "default scatter")] {
        let observed = s.se_of_mean * (k as f64).sqrt();
        let ratio = s.reported / observed;
        assert!((0.5..2.0).contains(&ratio), "{what}: ratio {ratio}");
    }
}

/// In analog transport each ray delivers `0` or `w₀ = P/N` to a class or
/// pixel, so the standard error is the binomial one, computed from the hit
/// counts.
#[test]
fn analog_standard_errors_are_binomial() {
    let sc = scene(window(1.5, 5.0, 20.0));
    let n = 5000u64;
    let r = run(
        &sc,
        &axial_beam(2.0),
        &StraySettings {
            rays: n,
            mode: SplitMode::Analog,
            ..Default::default()
        },
    );
    let (nf, w0) = (n as f64, 1.0 / n as f64);
    let binom = |k: f64| (nf / (nf - 1.0) * (k * w0 * w0 - (k * w0).powi(2) / nf)).sqrt();
    let e = &r.energy;
    let hits = e.detected_hits_by_class[PathClass::Primary.index()].1 as f64;
    let se = e.std_error_in(PathClass::Primary).unwrap();
    assert!(
        (se / binom(hits) - 1.0).abs() < 1e-9,
        "{se} {}",
        binom(hits)
    );
    assert!((e.detected_std_error.unwrap() / binom(e.detected / w0) - 1.0).abs() < 1e-9);
    let flux = r.detector.flux_of(PathClass::Primary);
    let ses = &r.detector.flux_std_error.as_ref().unwrap()[PathClass::Primary.index()];
    let mut checked = 0;
    for (f, s) in flux.iter().zip(ses) {
        if *f > 0.0 {
            let k = (f / w0).round();
            assert!((s / binom(k) - 1.0).abs() < 1e-9);
            checked += 1;
        } else {
            assert_eq!(*s, 0.0);
        }
    }
    assert!(checked > 10);
}

/// A ray that leaves the scene while in the glass medium is a leak: it is
/// booked as `escaped_from_solid`, not hidden in bulk absorption, and the
/// run warns about it.
#[test]
fn leaks_from_solids_are_reported() {
    let sc = scene(window(1.5, 5.0, 20.0));
    let glass = 1; // medium after surface 1
    let src = SourceSpec::Collimated {
        direction: -Vec3::Z,
        center: Vec3::new(0.0, 0.0, -50.0),
        radius: 1.0,
        start_distance: 0.0,
        power: 1.0,
    };
    let r = run(
        &sc,
        &src,
        &StraySettings {
            rays: 100,
            start_medium: Some(glass),
            ..Default::default()
        },
    );
    assert!((r.energy.escaped_from_solid - 1.0).abs() < 1e-12);
    assert_eq!(r.energy.escaped, 0.0);
    assert_eq!(r.energy.absorbed, 0.0);
    assert!(r.warnings.iter().any(|w| w.contains("inside a solid")));
}

/// `RunInfo` records the scene options and a fingerprint that changes with
/// any edit made to the scene after it was built.
#[test]
fn run_info_records_scene_options_and_fingerprint() {
    let mut sc = scene(window(1.5, 5.0, 20.0));
    let set = StraySettings {
        rays: 10,
        ..Default::default()
    };
    let info = run(&sc, &axial_beam(1.0), &set).run_info.unwrap();
    let opts = &info.settings["scene_options"];
    assert_eq!(opts["default_edge_margin"], 1.0);
    assert_eq!(opts["polished_outer_zone"], false);
    let fp = info.settings["scene_fingerprint"]
        .as_str()
        .unwrap()
        .to_string();
    assert_eq!(fp.len(), 64);
    assert_eq!(fp, sc.fingerprint());
    sc.media[1].alpha = vec![0.01];
    let info = run(&sc, &axial_beam(1.0), &set).run_info.unwrap();
    assert_ne!(info.settings["scene_fingerprint"].as_str().unwrap(), fp);
}
