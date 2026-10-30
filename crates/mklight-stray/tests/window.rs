//! Known-answer transport tests on a plane-parallel window.

mod common;

use common::*;
use mklight_core::materials::coating::fresnel;
use mklight_core::math::Vec3;
use mklight_core::model::{CoatingDef, ScatterModel};
use mklight_stray::{PathClass, Sink, SourceSpec, SplitMode, StraySettings};

fn beam(theta_deg: f64) -> SourceSpec {
    let th = theta_deg.to_radians();
    SourceSpec::Collimated {
        direction: Vec3::new(0.0, th.sin(), th.cos()),
        center: Vec3::ZERO,
        radius: 2.0,
        start_distance: 30.0,
        power: 1.0,
    }
}

fn split_settings(rays: u64) -> StraySettings {
    StraySettings {
        rays,
        mode: SplitMode::Split {
            max_splits: 64,
            rare_floor: 0.1,
        },
        roulette_threshold: 1e-5,
        ..Default::default()
    }
}

/// Splitting follows every specular branch with weight `w·p`, so the
/// second-order ghost is deterministic: exactly `T²R²` per ray.
#[test]
fn ghost_energy_is_r_squared_with_splitting() {
    let n = 1.5;
    // Image 5 mm behind the window so that the oblique ghost (displaced by
    // 2·t·tan θ' ≈ 3.5 mm) still lands on the detector.
    let sc = scene(window(n, 5.0, 5.0));
    for theta in [0.0f64, 30.0] {
        let r = run(&sc, &beam(theta), &split_settings(2000));
        let ci = theta.to_radians().cos();
        let f1 = fresnel(1.0, n, ci);
        let (rr, tt) = (f1.r_unpolarized(), f1.t_unpolarized());
        let e = &r.energy;
        let ghost2 = e.ghost_by_order[2];
        assert!(
            (ghost2 / (tt * tt * rr * rr) - 1.0).abs() < 1e-10,
            "θ={theta}: {ghost2} vs {}",
            tt * tt * rr * rr
        );
        assert!((e.detected_in(PathClass::Primary) / (tt * tt) - 1.0).abs() < 1e-12);
        // All detected light is either primary or a ghost; the total is the
        // incoherent slab transmittance T²/(1 − R²) (higher orders by roulette).
        let total = tt * tt / (1.0 - rr * rr);
        assert!((e.detected / total - 1.0).abs() < 1e-6, "{}", e.summary());
        assert_eq!(e.detected_in(PathClass::Scatter), 0.0);
        assert_eq!(e.ghost_by_pair[0].0, 2, "first reflection at the back face");
        assert_eq!(e.ghost_by_pair[0].1, 1);
        // Only high-order oblique ghosts walk out to the edge cylinder.
        assert!(e.absorbed < 1e-6 && e.truncated == 0.0, "{}", e.summary());
    }
}

/// With rare-branch forcing, the ghost fraction is an unbiased estimate. The
/// bare reflectance (4 %) is above the default floor, so the test raises the
/// floor to force the reflections.
#[test]
fn ghost_energy_weighted_and_analog_estimates() {
    let n = 1.5;
    let sc = scene(window(n, 5.0, 20.0));
    let f = fresnel(1.0, n, 1.0);
    let want = (f.t_unpolarized() * f.r_unpolarized()).powi(2);
    let (m, se) = seeds_mean(8, |seed| {
        let set = StraySettings {
            rays: 5000,
            seed,
            mode: SplitMode::Weighted { rare_floor: 0.2 },
            ..Default::default()
        };
        run(&sc, &beam(0.0), &set).energy.ghost_by_order[2]
    });
    assert!((m - want).abs() < 5.0 * se + 1e-12, "{m} ± {se} vs {want}");
    assert!(
        se / want < 0.05,
        "weighted mode should resolve the ghost well: {se}"
    );

    let set = StraySettings {
        rays: 200_000,
        seed: 3,
        mode: SplitMode::Analog,
        ..Default::default()
    };
    let r = run(&sc, &beam(0.0), &set);
    let p = r.energy.ghost_by_order[2];
    let sigma = (want * (1.0 - want) / 200_000.0).sqrt();
    assert!(
        (p - want).abs() < 5.0 * sigma,
        "analog {p} vs {want} (σ {sigma})"
    );
    // Analog transport never reweights.
    assert_eq!(r.energy.weighting_gain + r.energy.weighting_loss, 0.0);
}

/// Beer–Lambert: with an ideal (R = 0, T = 1) coating on both faces of an
/// absorbing slab, exactly `exp(−αL)` reaches the detector.
#[test]
fn beer_lambert_in_glass() {
    use mklight_core::materials::coating::ResolvedCoating;
    use mklight_stray::scene::Behavior;
    let mut sys = window(1.5, 10.0, 20.0);
    sys.coatings.insert(
        "ideal".into(),
        CoatingDef::Ideal {
            reflectance: 0.0,
            transmittance: 1.0,
        },
    );
    sys.surfaces[1].coating = Some("ideal".into());
    sys.surfaces[2].coating = Some("ideal".into());
    let rs = resolve(sys);
    let mut sc = mklight_stray::Scene::from_system(&rs, &Default::default()).unwrap();
    // Constant-index glass has no absorption data: give the slab medium α.
    let alpha = 0.02;
    let glass = (0..sc.primitives().len())
        .find_map(|i| match sc.behavior(i) {
            Behavior::Interface(f) if f.surface() == 1 => Some(f.medium_plus),
            _ => None,
        })
        .unwrap();
    sc.media[glass as usize].alpha = vec![alpha];
    assert!(matches!(
        sc.behavior(0),
        Behavior::Interface(f) if matches!(*f.coating, ResolvedCoating::Ideal { .. })
    ));
    let r = run(
        &sc,
        &beam(0.0),
        &StraySettings {
            rays: 1000,
            ..Default::default()
        },
    );
    let tau = (-alpha * 10.0f64).exp();
    // Spawned rays are offset by ~1e-9 mm from the surface, hence 1e-8.
    assert!(
        (r.energy.detected / tau - 1.0).abs() < 1e-8,
        "{} {tau}",
        r.energy.summary()
    );
    assert!((r.energy.absorbed_in(Sink::Bulk) / (1.0 - tau) - 1.0).abs() < 1e-7);
}

/// Lambertian scatter in transmission at the back face: the scattered
/// fraction is `T²·tis`, of which the detector disk (radius 15 at 20 mm)
/// collects the point-to-coaxial-disk configuration factor, averaged over
/// the beam footprint.
#[test]
fn lambertian_scatter_reaches_detector_with_view_factor() {
    let tis = 0.01;
    let mut sys = window(1.5, 5.0, 20.0);
    sys.surfaces[2].scatter = Some(ScatterModel::Lambertian { tis });
    let sc = scene(sys);
    let f = fresnel(1.0, 1.5, 1.0);
    let tt = f.t_unpolarized();
    // Configuration factor element → parallel coaxial disk (radius r at
    // height h, element offset a), averaged over the uniform beam disk.
    let (h, rd) = (20.0f64, 15.0f64);
    let vf = |a: f64| {
        let s = h * h + a * a + rd * rd;
        0.5 * (1.0 - (h * h + a * a - rd * rd) / (s * s - 4.0 * a * a * rd * rd).sqrt())
    };
    let nq = 2000;
    let avg: f64 = (0..nq)
        .map(|k| {
            let a = 2.0 * ((k as f64 + 0.5) / nq as f64).sqrt();
            vf(a)
        })
        .sum::<f64>()
        / nq as f64;
    // Only the transmitted lobe is enabled. Ghost-like paths that scatter after
    // two reflections add O(R²) ≈ 0.2 %, inside the 1 % allowance.
    let want = tt * tt * tis * avg;
    let (m, se) = seeds_mean(6, |seed| {
        let set = StraySettings {
            rays: 20_000,
            seed,
            scatter_reflection: false,
            ..Default::default()
        };
        run(&sc, &beam(0.0), &set)
            .energy
            .detected_in(PathClass::Scatter)
    });
    assert!(
        (m / want - 1.0).abs() < 0.01 + 5.0 * se / want,
        "{m} ± {se} vs {want}"
    );
    assert!(se / want < 0.05);
}
