use mklight_core::materials::{
    GlassCatalog, MaterialLibrary,
    agf::{self, AgfCatalog, GlassStatus},
};
use mklight_core::{MaterialRef, SurfaceId, samples};
use mklight_optim::glass::{CandidateSet, GlassWindow, coordinates, model_variables};
use std::sync::Arc;

#[test]
fn model_snaps_to_real_glass_first() {
    let lib = MaterialLibrary::with_builtin();
    let g = lib.find_glass(None, "N-BK7").unwrap();
    let [nd, vd, dpgf] = coordinates(&g);
    let mut sys = samples::singlet();
    sys.surfaces[1].material = MaterialRef::Model {
        nd: nd + 1e-5,
        vd: vd * 100.0 + 1e-3,
        dpgf,
    };
    let vars = model_variables(&sys);
    assert_eq!(vars.len(), 1);
    let candidates = vars[0]
        .candidates
        .resolve(&sys, SurfaceId(1), &lib)
        .unwrap();
    assert_eq!(candidates[0].name, "N-BK7");
    assert_eq!(candidates[0].catalog, "BUILTIN");
}

#[test]
fn agf_filters_survive_file_and_json_loading() {
    let mut agf = AgfCatalog::from_catalog(GlassCatalog {
        name: "FILTERS".into(),
        ..Default::default()
    });
    for (name, status, tau, cost, climate, stain) in [
        ("CLEAR", 1, 0.99, 2.0, 1.0, 1.0),
        ("ABSORBING", 0, 0.25, 2.0, 1.0, 1.0),
        ("OBSOLETE", 2, 0.99, 2.0, 1.0, 1.0),
        ("SPECIAL", 3, 0.99, 2.0, 1.0, 1.0),
        ("EXPENSIVE", 0, 0.99, 20.0, 1.0, 1.0),
        ("WEATHERED", 0, 0.99, 2.0, 4.0, 1.0),
        ("STAINED", 0, 0.99, 2.0, 1.0, 4.0),
    ] {
        let mut g = agf::model_glass(name, 1.52, 64.0, 0.0).unwrap();
        g.relative_cost = Some(cost);
        g.transmittance
            .push(mklight_core::materials::TransmittancePoint {
                um: 0.48,
                tau,
                thickness_mm: 10.0,
            });
        agf.catalog.glasses.insert(name.into(), g);
        agf.meta.insert(
            name.into(),
            agf::AgfGlassMeta {
                status: GlassStatus::from_code(status),
                ordering: agf::OrderingData {
                    relative_cost: Some(cost),
                    climate_resistance: Some(climate),
                    stain_resistance: Some(stain),
                    ..Default::default()
                },
                ..Default::default()
            },
        );
    }
    let (text, warnings) = agf::write_agf(&agf);
    assert!(warnings.is_empty(), "{warnings:?}");
    let path =
        std::env::temp_dir().join(format!("mklight-glass-filter-{}.agf", std::process::id()));
    std::fs::write(&path, text).unwrap();
    let mut cat = GlassCatalog::from_agf_file(&path).unwrap();
    std::fs::remove_file(path).unwrap();
    cat.name = "FILTERS".into();
    let cat: GlassCatalog = serde_json::from_str(&serde_json::to_string(&cat).unwrap()).unwrap();
    assert_eq!(
        cat.glasses["OBSOLETE"].substitution.status,
        GlassStatus::Obsolete
    );
    let mut lib = MaterialLibrary::with_builtin();
    lib.add_catalog(Arc::new(cat));
    let sys = samples::singlet(); // 5 mm: sqrt(tau_10mm)
    let opts = CandidateSet {
        min_transmission: Some(0.9),
        max_relative_cost: Some(3.0),
        max_climate_class: Some(2.0),
        max_stain_class: Some(2.0),
        ..Default::default()
    };
    let choices = opts.resolve(&sys, SurfaceId(1), &lib).unwrap();
    assert_eq!(
        choices.iter().map(|g| g.name.as_str()).collect::<Vec<_>>(),
        ["CLEAR"]
    );
    let unfiltered = CandidateSet::default()
        .resolve(&sys, SurfaceId(1), &lib)
        .unwrap();
    assert_eq!(unfiltered.len(), 5);
    assert!(unfiltered.iter().all(|g| g.catalog == "FILTERS"));
    // Transmission alone excludes the absorbing entry (5 mm -> 0.5).
    let t = CandidateSet {
        min_transmission: Some(0.9),
        ..Default::default()
    }
    .resolve(&sys, SurfaceId(1), &lib)
    .unwrap();
    assert!(!t.iter().any(|g| g.name == "ABSORBING"));
    // Including an obsolete glass does not defeat status filtering.
    let included = CandidateSet {
        include: vec!["OBSOLETE".into()],
        ..opts.clone()
    }
    .resolve(&sys, SurfaceId(1), &lib)
    .unwrap();
    assert_eq!(included, choices);
    let allowed = CandidateSet {
        statuses: vec![GlassStatus::Obsolete],
        ..Default::default()
    }
    .resolve(&sys, SurfaceId(1), &lib)
    .unwrap();
    assert_eq!(allowed[0].name, "OBSOLETE");
}

#[test]
fn explicit_lists_override_proximity_but_not_exclusions() {
    let lib = MaterialLibrary::with_builtin();
    let sys = samples::singlet();
    let opts = CandidateSet {
        window: Some(GlassWindow { nd: 0.01, vd: 2.0 }),
        nearest: Some(1),
        include: vec!["builtin:F2".into()],
        ..Default::default()
    };
    let choices = opts.resolve(&sys, SurfaceId(1), &lib).unwrap();
    assert_eq!(choices.len(), 2);
    assert_eq!(choices[0].name, "N-BK7");
    assert_eq!(choices[1].name, "F2");
    // Regression: an include that is also among the nearest does not take
    // one of the K places.
    let own = CandidateSet {
        nearest: Some(1),
        include: vec!["N-BK7".into()],
        ..Default::default()
    }
    .resolve(&sys, SurfaceId(1), &lib)
    .unwrap();
    assert_eq!(own.len(), 2);
    assert_eq!(own[0].name, "N-BK7");
    let excluded = CandidateSet {
        exclude: vec!["F2".into()],
        ..opts
    }
    .resolve(&sys, SurfaceId(1), &lib)
    .unwrap();
    assert_eq!(excluded.len(), 1);
    assert!(
        CandidateSet {
            include: vec!["not-loaded".into()],
            ..Default::default()
        }
        .resolve(&sys, SurfaceId(1), &lib)
        .is_err()
    );
}
