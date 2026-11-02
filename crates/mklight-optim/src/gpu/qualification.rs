//! Offline candidate geometry evidence. This module is compiled only in tests;
//! production candidate evaluation never runs these reference comparisons.
use mklight_core::{JobContext, samples};

#[test]
fn batched_candidate_geometry_preserves_individual_verdicts() {
    check_candidate_geometry(&mklight_gpu::CpuF32Backend::new());
}
fn check_candidate_geometry(backend: &dyn mklight_gpu::ComputeBackend) {
    use mklight_gpu::{
        Radiometry,
        xval::{XvalSettings, qualify, qualify_many},
    };
    let mut systems = vec![
        samples::singlet(),
        samples::cooke_triplet(),
        samples::double_gauss(),
    ];
    let mut changed = systems[1].clone();
    changed.surfaces[1].thickness += 0.25;
    systems.push(changed);
    systems.extend((0..65).map(|_| samples::singlet()));
    let tracers: Vec<_> = systems
        .into_iter()
        .map(|s| mklight_analysis::tracer_for(s).unwrap())
        .collect();
    let refs: Vec<_> = tracers.iter().collect();
    let settings = XvalSettings {
        radiometry: Radiometry::Off,
        pupil: mklight_trace::PupilSampling::Hexapolar { rings: 2 },
        pupil_extra: vec![],
        ..XvalSettings::qualification()
    };
    let batch = qualify_many(backend, &refs, &settings, &JobContext::new()).unwrap();
    assert_eq!(batch.len(), refs.len());
    for (tracer, got) in refs.iter().zip(&batch) {
        let want = qualify(backend, tracer, Some(&settings), &JobContext::new()).unwrap();
        assert_eq!(*got, want);
        assert!(got.passed, "{:?}", got.reason);
    }
    let cancelled = JobContext::new();
    cancelled.cancel.cancel();
    assert!(matches!(
        qualify_many(backend, &refs, &settings, &cancelled),
        Err(mklight_core::MkError::Cancelled)
    ));
}

#[test]
#[ignore = "requires real Metal; batched per-candidate geometry verdicts"]
fn host_batched_candidate_geometry() {
    use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let b = MetalBackend::with_options(&MetalOptions {
            library,
            ..Default::default()
        })
        .unwrap()
        .expect("real Metal required");
        check_candidate_geometry(&b);
    }
}
