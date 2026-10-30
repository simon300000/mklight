//! Every sample runs; the Double Gauss gives a ranked ghost list;
//! enumeration filters and budget; determinism and cancellation; the
//! partition coverage.

mod common;

use common::*;
use mklight_core::{JobContext, MkError, samples};
use mklight_ghost::{
    EnumerationSettings, GhostPath, GhostSettings, InterfaceKind, IrradianceSettings, Owner,
    ParaxialGhostSettings, PathClass, PathSignature, classify, enumerate, ghost_irradiance,
    paraxial_ghosts,
};

fn small_irradiance() -> IrradianceSettings {
    IrradianceSettings {
        pupil_samples: 12,
        map_bins: [24, 24],
        ..Default::default()
    }
}

#[test]
fn all_samples_run() {
    for (name, sys) in samples::all() {
        let m = model(sys, GhostSettings::default());
        let rep = paraxial_ghosts(&m, &ParaxialGhostSettings::default())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let n_refr = m
            .interface_kinds()
            .iter()
            .filter(|k| **k == InterfaceKind::Refractive)
            .count();
        assert_eq!(
            rep.ghosts.len(),
            n_refr * n_refr.saturating_sub(1) / 2,
            "{name}"
        );
        for g in &rep.ghosts {
            assert!(
                g.reflectance_product > 0.0 && g.reflectance_product < 1.0,
                "{name}"
            );
            assert!(g.relative_irradiance.is_finite() && g.relative_irradiance >= 0.0);
            assert!(g.spot_diameter.is_finite() && g.vignetting > 0.0 && g.vignetting <= 1.0);
        }
        let res = ghost_irradiance(&m, &small_irradiance(), &JobContext::new())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(res.primary.flux > 0.0, "{name}");
        assert_eq!(res.ghosts.len(), rep.ghosts.len(), "{name}");
        assert!(res.total_flux_ratio.is_finite(), "{name}");
        assert!(
            res.total_ghost_map
                .values
                .iter()
                .all(|v| v.is_finite() && *v >= 0.0)
        );
        // Each ghost loses at least two reflections' worth of flux.
        for g in &res.ghosts {
            assert!(g.flux_ratio < 0.1, "{name} {}: {}", g.label, g.flux_ratio);
        }
        assert_eq!(
            res.run.snapshot_hash,
            m.system().snapshot.hash().to_string()
        );
    }
}

#[test]
fn double_gauss_ranked_list() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let rep = paraxial_ghosts(&m, &ParaxialGhostSettings::default()).unwrap();
    // 10 refracting interfaces (the stop plane is passive), cemented ones included.
    assert_eq!(rep.ghosts.len(), 45);
    assert!(
        rep.ghosts
            .iter()
            .any(|g| g.path == GhostPath::new(vec![8, 7]))
    );
    assert!(
        rep.ghosts
            .windows(2)
            .all(|w| w[0].relative_irradiance >= w[1].relative_irradiance)
    );
    // Primary paraxial focus agrees with the reference first-order data.
    let fo = m.tracer().paraxial().primary();
    let t_last = m.system().surfaces[11].thickness;
    assert!((rep.primary_focus_from_sensor - (fo.image_distance - t_last)).abs() < 1e-9);
    // The strongest ghosts are far stronger than the weakest (the list ranks).
    let first = rep.ghosts[0].relative_irradiance;
    let last = rep.ghosts.last().unwrap().relative_irradiance;
    assert!(first > 100.0 * last, "{first} {last}");
    // Cemented interfaces (small index step) reflect much less than air–glass ones.
    let r = |p: &[u32]| {
        rep.ghosts
            .iter()
            .find(|g| g.path.reflections == p)
            .unwrap()
            .reflectance_product
    };
    assert!(r(&[8, 7]) < 0.05 * r(&[9, 7]));
    // Near-focus flag ⇔ small spot.
    for g in &rep.ghosts {
        assert_eq!(g.near_focus, g.spot_diameter <= 10.0 * rep.signal_diameter);
    }
    let j = serde_json::to_string(&rep).unwrap();
    assert!(j.contains("relative_irradiance"));
}

#[test]
fn irradiance_is_deterministic_and_cancellable() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let s = IrradianceSettings {
        fields: vec![0, 2],
        wavelengths: vec![1],
        pupil_samples: 20,
        map_bins: [32, 32],
        ..Default::default()
    };
    let ctx = JobContext::new();
    let a = ghost_irradiance(&m, &s, &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    let b = ghost_irradiance(&m, &s, &JobContext::new()).unwrap();
    assert_eq!(a.total_ghost_map, b.total_ghost_map);
    assert_eq!(a.ghosts, b.ghosts);
    assert!(
        a.ghosts
            .windows(2)
            .all(|w| w[0].peak_ratio >= w[1].peak_ratio)
    );
    // The total map is the sum of the path maps.
    let s2 = IrradianceSettings {
        keep_path_maps: true,
        ..s.clone()
    };
    let c = ghost_irradiance(&m, &s2, &JobContext::new()).unwrap();
    let sum: f64 = c
        .ghosts
        .iter()
        .map(|g| g.map.as_ref().unwrap().flux())
        .sum();
    assert!((sum - c.total_ghost_map.flux()).abs() < 1e-12 * sum);
    assert!((c.total_flux_ratio - sum / c.primary.flux).abs() < 1e-12);

    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(matches!(
        ghost_irradiance(&m, &s, &ctx),
        Err(MkError::Cancelled)
    ));
}

#[test]
fn enumeration_filters_and_budget() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let all4 = enumerate(
        &m,
        &EnumerationSettings {
            max_reflections: 4,
            max_higher_order_paths: usize::MAX,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!all4.truncated);
    // Count of alternating sequences r0 > r1 < r2 > r3 over 10 reflectors.
    let refl: Vec<u32> = (1..12).filter(|&s| s != 6).collect();
    let mut n4 = 0;
    for &a in &refl {
        for &b in refl.iter().filter(|&&b| b < a) {
            for &c in refl.iter().filter(|&&c| c > b) {
                n4 += refl.iter().filter(|&&d| d < c).count();
            }
        }
    }
    let higher: Vec<_> = all4.paths.iter().filter(|p| p.path.order() == 4).collect();
    assert_eq!(higher.len(), n4);
    assert_eq!(all4.paths.len(), 45 + n4);
    // Budget keeps exactly the strongest higher-order paths.
    let budget = 50;
    let cut = enumerate(
        &m,
        &EnumerationSettings {
            max_reflections: 4,
            max_higher_order_paths: budget,
            ..Default::default()
        },
    )
    .unwrap();
    assert!(cut.truncated);
    let mut ranked = higher.clone();
    ranked.sort_by(|a, b| {
        b.reflectance_product
            .total_cmp(&a.reflectance_product)
            .then_with(|| a.path.cmp(&b.path))
    });
    let mut want: Vec<_> = ranked[..budget].iter().map(|p| p.path.clone()).collect();
    want.sort();
    let mut got: Vec<_> = cut
        .paths
        .iter()
        .filter(|p| p.path.order() == 4)
        .map(|p| p.path.clone())
        .collect();
    got.sort();
    assert_eq!(got, want);
    assert_eq!(cut.coverage.paths.len(), 45 + budget);
    // Reflectance threshold and surface filters.
    let thr = 1e-3;
    let f = enumerate(
        &m,
        &EnumerationSettings {
            min_reflectance_product: thr,
            surface_range: Some([1, 5]),
            exclude_surfaces: vec![2],
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!f.paths.is_empty());
    for p in &f.paths {
        assert!(p.reflectance_product >= thr);
        assert!(
            p.path
                .reflections
                .iter()
                .all(|&s| (1..=5).contains(&s) && s != 2)
        );
    }
    let expected = all4
        .paths
        .iter()
        .filter(|p| p.path.order() == 2)
        .filter(|p| p.reflectance_product >= thr)
        .filter(|p| {
            p.path
                .reflections
                .iter()
                .all(|&s| (1..=5).contains(&s) && s != 2)
        })
        .count();
    assert_eq!(f.paths.len(), expected);
}

#[test]
fn coverage_partitions_enumerated_paths() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let e = enumerate(&m, &EnumerationSettings::default()).unwrap();
    let kinds = m.interface_kinds();
    let prim = PathSignature::for_path(&GhostPath::primary(), kinds).unwrap();
    assert_eq!(classify(&prim, kinds, false, false), PathClass::Primary);
    assert_eq!(e.coverage.owner(&prim, false, false), Owner::Sequential);
    for p in &e.paths {
        let sig = PathSignature::for_path(&p.path, kinds).unwrap();
        // The stop plane (passive) never appears.
        assert!(sig.events.iter().all(|ev| ev.surface != 6));
        assert_eq!(
            classify(&sig, kinds, false, false),
            PathClass::Ghost { reflections: 2 }
        );
        assert_eq!(e.coverage.owner(&sig, false, false), Owner::Ghost);
        assert_eq!(e.coverage.owner(&sig, true, false), Owner::Stray);
    }
    let four = PathSignature::for_path(&GhostPath::new(vec![9, 3, 7, 1]), kinds).unwrap();
    assert_eq!(e.coverage.owner(&four, false, false), Owner::Stray);
}

#[test]
fn irradiance_is_bit_identical_across_thread_counts() {
    let m = model(samples::double_gauss(), GhostSettings::default());
    let s = IrradianceSettings {
        fields: vec![0, 2],
        wavelengths: vec![1],
        pupil_samples: 24,
        map_bins: [16, 16],
        keep_path_maps: true,
        ..Default::default()
    };
    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| ghost_irradiance(&m, &s, &JobContext::new()).unwrap())
    };
    let (a, b) = (run(1), run(5));
    assert_eq!(a.primary, b.primary);
    assert_eq!(a.ghosts, b.ghosts);
    assert_eq!(a.total_ghost_map, b.total_ghost_map);
    assert_eq!(a.total_flux_ratio.to_bits(), b.total_flux_ratio.to_bits());
}
