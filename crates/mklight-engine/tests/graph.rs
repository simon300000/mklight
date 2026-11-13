//! The analysis graph (`mklight_engine::graph`): shared intermediates give
//! bit-identical results, every kind of edit invalidates what depends on it,
//! concurrent refreshes trace each pupil once, and cancelled or failing
//! builds publish nothing.
use std::time::Duration;

use mklight_core::materials::MaterialLibrary;
use mklight_core::model::CoatingDef;
use mklight_core::{ApertureSpec, JobContext, MaterialRef, OpticalSystem, samples};
use mklight_engine::backend::Backend;
use mklight_engine::buffer::BufferData;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};
use serde_json::{Value, json};

fn engine(graph_bytes: usize) -> Engine {
    Engine::new(EngineConfig {
        worker_threads: 4,
        default_backend: Backend::CpuReference,
        analysis_graph_bytes: graph_bytes,
        ..Default::default()
    })
}

fn session(e: &Engine, sys: &OpticalSystem) -> Session {
    e.session_from_project(Project::new(sys.clone())).unwrap()
}

/// Small wave sampling, so the debug test suite stays fast; the pupil
/// grid stays automatic (the shared sampling decision).
fn cases() -> Vec<(&'static str, Value)> {
    let pupil = json!({"rays": 17});
    vec![
        ("layout", Value::Null),
        ("spot_diagram", Value::Null),
        ("ray_fan", Value::Null),
        ("field_curvature_distortion", Value::Null),
        ("mtf", json!({"pupil": pupil, "frequency_samples": 6})),
        ("psf", json!({"pupil": pupil, "grid_size": 32})),
        ("line_spread", json!({"pupil": pupil, "grid_size": 32})),
        (
            "diffraction_encircled_energy",
            json!({"pupil": pupil, "grid_size": 32, "samples": 8}),
        ),
        (
            "through_focus_mtf",
            json!({"pupil": pupil, "focus_samples": 3, "frequencies": [10.0, 30.0]}),
        ),
        ("wavefront_map", json!({"pupil": pupil})),
        (
            "mtf",
            json!({"pupil": pupil, "frequency_samples": 6, "method": "fft"}),
        ),
    ]
}

fn run(s: &Session, name: &str, settings: &Value) -> AnalysisResult {
    s.run_analysis(
        &AnalysisRequest::new(name, settings.clone()),
        &JobContext::new(),
    )
    .unwrap_or_else(|e| panic!("{name}: {e}"))
}

fn bits(d: &BufferData) -> Vec<u64> {
    match d {
        BufferData::F64(v) => v.iter().map(|x| x.to_bits()).collect(),
        BufferData::F32(v) => v.iter().map(|x| x.to_bits() as u64).collect(),
        BufferData::I32(v) => v.iter().map(|&x| x as u64).collect(),
        BufferData::U32(v) => v.iter().map(|&x| x as u64).collect(),
        BufferData::I64(v) => v.iter().map(|&x| x as u64).collect(),
        BufferData::U64(v) => v.clone(),
        BufferData::U8(v) => v.iter().map(|&x| x as u64).collect(),
    }
}

/// Bit-identical results: JSON text, warnings, backend and every buffer.
fn same(a: &AnalysisResult, b: &AnalysisResult, what: &str) {
    assert_eq!(
        serde_json::to_string(&a.json).unwrap(),
        serde_json::to_string(&b.json).unwrap(),
        "{what}: json"
    );
    assert_eq!(a.run_info.warnings, b.run_info.warnings, "{what}: warnings");
    assert_eq!(a.run_info.snapshot_hash, b.run_info.snapshot_hash, "{what}");
    assert_eq!(a.backend.used, b.backend.used, "{what}");
    assert_eq!(a.buffers.len(), b.buffers.len(), "{what}: buffers");
    for (x, y) in a.buffers.iter().zip(&b.buffers) {
        assert_eq!((&x.name, &x.shape), (&y.name, &y.shape), "{what}");
        assert!(
            bits(x.data()) == bits(y.data()),
            "{what}: buffer {}",
            x.name
        );
    }
}

fn kind(e: &Engine, k: &str) -> mklight_engine::graph::KindStats {
    e.analysis_graph_stats()
        .unwrap()
        .kinds
        .get(k)
        .cloned()
        .unwrap_or_default()
}

#[test]
fn shared_intermediates_are_bit_identical() {
    for sys in [samples::double_gauss(), samples::anamorphic_adapter()] {
        let (on, off) = (engine(512 << 20), engine(0));
        assert!(off.analysis_graph_stats().is_none());
        let (a, b) = (session(&on, &sys), session(&off, &sys));
        for (name, settings) in cases() {
            same(&run(&a, name, &settings), &run(&b, name, &settings), name);
        }
        // Every wave analysis after the first reused the traced lattices,
        // and FFT and direct MTF stay distinct results.
        let lattice = kind(&on, "pupil_lattice");
        assert!(lattice.hits > 0 && lattice.builds > 0, "{lattice:?}");
        let results = kind(&on, "result");
        assert_eq!(results.hits, 0);
        assert_eq!(results.builds as usize, cases().len());
        // Repeated requests are answered from the graph, identically.
        for (name, settings) in cases() {
            same(&run(&a, name, &settings), &run(&b, name, &settings), name);
        }
        assert_eq!(kind(&on, "result").hits as usize, cases().len());
    }
}

type Edit = Box<dyn Fn(&mut OpticalSystem)>;

/// One edit of each kind, applied to the double Gauss.
fn edits() -> Vec<(&'static str, Edit)> {
    vec![
        (
            "radius",
            Box::new(|s: &mut OpticalSystem| {
                let c = s.surfaces[3].shape.curvature();
                mklight_core::param::ParamPath::Curvature {
                    surface: s.surfaces[3].id,
                }
                .set(s, c * 1.001)
                .unwrap();
            }),
        ),
        (
            "thickness",
            Box::new(|s: &mut OpticalSystem| s.surfaces[5].thickness += 0.05),
        ),
        (
            "glass",
            Box::new(|s: &mut OpticalSystem| s.surfaces[7].material = MaterialRef::glass("N-BK7")),
        ),
        (
            "aperture",
            Box::new(|s: &mut OpticalSystem| {
                s.aperture = ApertureSpec::ImageSpaceFNumber { value: 2.2 }
            }),
        ),
        (
            "field",
            Box::new(|s: &mut OpticalSystem| s.fields.fields[2].y = 12.5),
        ),
        (
            "wavelength",
            Box::new(|s: &mut OpticalSystem| s.wavelengths.wavelengths[0].um += 0.01),
        ),
        (
            "environment",
            Box::new(|s: &mut OpticalSystem| s.environment.temperature_c = 35.0),
        ),
        (
            "coating",
            Box::new(|s: &mut OpticalSystem| {
                // Angle dependent, so the pupil amplitudes change shape.
                s.coatings.insert(
                    "test".into(),
                    CoatingDef::Tabulated {
                        reflectance: vec![(0.4, 0.3), (0.8, 0.5)],
                    },
                );
                for i in [1, 2, 10, 11] {
                    s.surfaces[i].coating = Some("test".into());
                }
            }),
        ),
    ]
}

#[test]
fn edits_invalidate_every_dependent_node() {
    let base = samples::double_gauss();
    let on = engine(512 << 20);
    let mut a = session(&on, &base);
    let graphs: Vec<_> = cases().into_iter().take(6).collect();
    let before: Vec<_> = graphs.iter().map(|(n, s)| run(&a, n, s)).collect();
    for (what, edit) in edits() {
        let mut sys = base.clone();
        edit(&mut sys);
        a.set_system(sys.clone()).unwrap();
        let off = engine(0);
        let b = session(&off, &sys);
        let hits = kind(&on, "result").hits;
        let mut changed = 0;
        for (k, (name, settings)) in graphs.iter().enumerate() {
            let r = run(&a, name, settings);
            same(&r, &run(&b, name, settings), &format!("{what} / {name}"));
            changed += (r.json != before[k].json) as usize;
        }
        // The edit changed answers the graph held (coatings change only
        // the wave results, through the pupil amplitudes).
        assert!(changed >= 1, "{what}: no graph changed");
        assert_eq!(
            kind(&on, "result").hits,
            hits,
            "{what}: stale result reused"
        );
        // Undo: the original content is answered from the graph.
        a.set_system(base.clone()).unwrap();
        for (k, (name, settings)) in graphs.iter().enumerate() {
            same(
                &run(&a, name, settings),
                &before[k],
                &format!("undo {what} / {name}"),
            );
        }
        assert_eq!(kind(&on, "result").hits, hits + graphs.len() as u64);
    }
}

#[test]
fn library_changes_invalidate_resolution() {
    let mut sys = samples::double_gauss();
    let on = engine(512 << 20);
    let mut a = session(&on, &sys);
    let mtf = cases()[4].1.clone();
    let first = run(&a, "mtf", &mtf);
    // A user catalog whose N-SK16 differs, selected explicitly.
    let builtin = MaterialLibrary::with_builtin();
    let mut cat = builtin.catalogs()[0].as_ref().clone();
    cat.name = "user".into();
    let bk7 = cat.glasses["N-BK7"].dispersion.clone();
    cat.glasses.get_mut("N-SK16").unwrap().dispersion = bk7;
    for i in [1, 3, 8, 10] {
        sys.surfaces[i].material = MaterialRef::Catalog {
            catalog: Some("user".into()),
            name: "N-SK16".into(),
            offset: Default::default(),
        };
    }
    a.add_catalog(cat.clone(), None);
    a.set_system(sys.clone()).unwrap();
    let off = engine(0);
    let mut b = session(&off, &sys);
    b.add_catalog(cat, None);
    let r = run(&a, "mtf", &mtf);
    same(&r, &run(&b, "mtf", &mtf), "user catalog");
    assert_ne!(r.json, first.json);
}

#[test]
fn queued_refresh_traces_each_pupil_once() {
    let sys = samples::double_gauss();
    let (on, off) = (engine(512 << 20), engine(0));
    let (a, b) = (session(&on, &sys), session(&off, &sys));
    let wave: Vec<_> = cases().into_iter().skip(4).take(5).collect();
    let jobs: Vec<_> = wave
        .iter()
        .map(|(n, s)| {
            a.submit_analysis(&AnalysisRequest::new(*n, s.clone()))
                .unwrap()
        })
        .collect();
    for ((name, settings), id) in wave.iter().zip(jobs) {
        on.wait_job(id, None).unwrap();
        same(&on.take_result(id).unwrap(), &run(&b, name, settings), name);
    }
    // 3 fields × 3 wavelengths, one lattice each, whatever the timing.
    let lattice = kind(&on, "pupil_lattice");
    assert_eq!(lattice.builds, 9, "{lattice:?}");
    assert_eq!(lattice.duplicate_builds, 0);
    assert_eq!(lattice.failed_builds, 0);
}

#[test]
fn cancelled_runs_publish_nothing() {
    let sys = samples::double_gauss();
    let (on, off) = (engine(512 << 20), engine(0));
    let (a, b) = (session(&on, &sys), session(&off, &sys));
    let settings = json!({"pupil": {"rays": 65}, "grid_size": 32});
    // Cancel at increasing delays: some runs stop before, some during and
    // some after their builds.
    for delay in [0u64, 2, 5, 10, 20, 40] {
        on.clear_analysis_graph();
        let ctx = JobContext::new();
        let cancel = ctx.cancel.clone();
        let t = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(delay));
            cancel.cancel();
        });
        let r = a.run_analysis(&AnalysisRequest::new("psf", settings.clone()), &ctx);
        t.join().unwrap();
        if let Err(e) = &r {
            assert!(e.is_cancelled(), "{e}");
        }
        same(
            &run(&a, "psf", &settings),
            &run(&b, "psf", &settings),
            "after cancel",
        );
    }
}

/// A stored result is allocated once: the result returned by the run that
/// built it, the one the graph keeps and every later answer hold the same
/// buffer payloads, and the graph counts their bytes once.
#[test]
fn stored_results_share_their_buffers() {
    use std::sync::Arc;
    let e = engine(512 << 20);
    let s = session(&e, &samples::double_gauss());
    for (name, settings) in [
        ("ray_grid", json!({"pattern": "square", "rings": 24})),
        ("spot_diagram", json!({"include_points": true})),
        ("psf", json!({"pupil": {"rays": 17}, "grid_size": 32})),
    ] {
        let stored_before = kind(&e, "result");
        let built = run(&s, name, &settings);
        assert!(built.buffers.iter().any(|b| !b.data().is_empty()), "{name}");
        let payload_bytes: usize = built
            .buffers
            .iter()
            .map(|b| b.data().len() * b.dtype().size_bytes())
            .sum();
        // Holders of each payload besides the probe: the returned result
        // and the stored one.
        let holders = |r: &AnalysisResult| -> Vec<usize> {
            r.buffers
                .iter()
                .map(|b| Arc::strong_count(&b.share()) - 1)
                .collect()
        };
        assert_eq!(holders(&built), vec![2; built.buffers.len()], "{name}");

        let hit = run(&s, name, &settings);
        let stats = kind(&e, "result");
        assert_eq!(stats.builds, stored_before.builds + 1, "{name}");
        assert_eq!(stats.hits, stored_before.hits + 1, "{name}");
        same(&built, &hit, name);
        for (x, y) in built.buffers.iter().zip(&hit.buffers) {
            assert!(Arc::ptr_eq(&x.share(), &y.share()), "{name}: {}", x.name);
            assert_eq!(x.data().as_ptr(), y.data().as_ptr(), "{name}: {}", x.name);
        }
        assert_eq!(holders(&hit), vec![3; hit.buffers.len()], "{name}");

        // A clone shares, too.
        let clone = hit.clone();
        for (x, y) in clone.buffers.iter().zip(&hit.buffers) {
            assert!(Arc::ptr_eq(&x.share(), &y.share()), "{name}: {}", x.name);
        }
        assert_eq!(clone, hit);
        assert_eq!(holders(&hit), vec![4; hit.buffers.len()], "{name}");

        // The graph accounts for the payload once, however many results
        // share it.
        let added = stats.bytes - stored_before.bytes;
        assert!(
            added >= payload_bytes && added < 2 * payload_bytes + (1 << 20),
            "{name}: {added} bytes stored for {payload_bytes} of payload"
        );
        assert_eq!(kind(&e, "result").bytes, stats.bytes, "{name}");

        // With every returned result gone the graph still answers with the
        // same arrays.
        let pointers: Vec<_> = built.buffers.iter().map(|b| b.data().as_ptr()).collect();
        drop((built, hit, clone));
        let later = run(&s, name, &settings);
        let now: Vec<_> = later.buffers.iter().map(|b| b.data().as_ptr()).collect();
        assert_eq!(now, pointers, "{name}");
        assert_eq!(holders(&later), vec![2; later.buffers.len()], "{name}");
    }
}

#[test]
fn memory_stays_within_budget() {
    let sys = samples::double_gauss();
    let budget = 2 << 20;
    let (on, off) = (engine(budget), engine(0));
    let (mut a, mut b) = (session(&on, &sys), session(&off, &sys));
    for k in 0..3 {
        let mut s = sys.clone();
        s.surfaces[5].thickness += 0.01 * k as f64;
        a.set_system(s.clone()).unwrap();
        b.set_system(s).unwrap();
        for (name, settings) in cases() {
            same(&run(&a, name, &settings), &run(&b, name, &settings), name);
        }
        let st = on.analysis_graph_stats().unwrap();
        assert!(st.bytes <= budget, "{st:?}");
    }
    let st = on.analysis_graph_stats().unwrap();
    assert!(
        st.kinds.values().map(|k| k.evictions).sum::<u64>() > 0,
        "{st:?}"
    );
}

#[test]
fn explicit_grids_share_the_lattice() {
    let sys = samples::double_gauss();
    let on = engine(512 << 20);
    let off = engine(0);
    let (a, b) = (session(&on, &sys), session(&off, &sys));
    for grid in [0, 64, 128] {
        let s = json!({"pupil": {"rays": 17, "grid": grid}, "frequency_samples": 4});
        same(&run(&a, "mtf", &s), &run(&b, "mtf", &s), "mtf grid");
    }
    // One lattice per (field, wavelength), resampled three ways.
    let lattice = kind(&on, "pupil_lattice");
    assert_eq!((lattice.builds, lattice.hits), (9, 18), "{lattice:?}");
    assert!(kind(&on, "resampled_pupil").builds >= 18);
    // Tracers: one per snapshot, shared by all wave analyses.
    assert_eq!(kind(&on, "configured").builds, 1);
}

/// A fresh request (`AnalysisRequest::fresh`) reads no stored node: it
/// rebuilds the result and its intermediates, and what it builds replaces
/// the stored nodes, so later ordinary requests are served the new value.
#[test]
fn fresh_requests_recompute_and_replace_the_stored_result() {
    use std::sync::Arc;
    let e = engine(512 << 20);
    let s = session(&e, &samples::double_gauss());
    let settings = json!({"pupil": {"rays": 17}, "grid_size": 32});
    let req = AnalysisRequest::new("psf", settings);
    let fresh = AnalysisRequest {
        fresh: true,
        ..req.clone()
    };
    let go = |r: &AnalysisRequest| s.run_analysis(r, &JobContext::new()).unwrap();
    let first = go(&req);
    let again = go(&req);
    let r = kind(&e, "result");
    assert_eq!((r.builds, r.hits, r.rebuilds), (1, 1, 0), "{r:?}");
    same(&first, &again, "stored");
    let lattice_hits = kind(&e, "pupil_lattice").hits;

    let recomputed = go(&fresh);
    let r = kind(&e, "result");
    assert_eq!((r.builds, r.hits, r.rebuilds), (2, 1, 1), "{r:?}");
    for k in ["configured", "pupil_lattice"] {
        assert!(kind(&e, k).rebuilds > 0, "{k}: {:?}", kind(&e, k));
    }
    assert_eq!(
        kind(&e, "pupil_lattice").hits,
        lattice_hits,
        "a fresh run reads no lattice"
    );
    same(&first, &recomputed, "recomputed");

    // The fresh result replaced the stored one: the next ordinary request
    // is a hit sharing the fresh run's buffers, not the first run's.
    let after = go(&req);
    let r = kind(&e, "result");
    assert_eq!((r.hits, r.entries), (2, 1), "{r:?}");
    let shares = |a: &AnalysisResult, b: &AnalysisResult| {
        Arc::ptr_eq(&a.buffers[0].share(), &b.buffers[0].share())
    };
    assert!(shares(&after, &recomputed));
    assert!(!shares(&after, &first));
}
