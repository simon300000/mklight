#[path = "p1_support.rs"]
mod support;
use mklight_core::{JobContext, RayAiming, samples};
use mklight_gpu::*;
use mklight_trace::PupilSampling;
use support::*;
#[test]
fn reference_all_samples() {
    let cpu = CpuF32Backend::new();
    let mut points = vec![[0., 0.]];
    points.extend((PupilSampling::Grid { n: 7 }).points());
    for (name, model) in testutil::xval_systems().into_iter().take(32) {
        for aiming in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
            let mut model = model.clone();
            model.ray_aiming = aiming;
            let t = testutil::tracer(model);
            for field in 0..t.fields().len() {
                for wl in 0..t.num_wavelengths() {
                    let (p, h) = make_plan(&cpu, &t, field, wl, &points, true, 40, true);
                    let r = cpu
                        .submit_plan(p, &JobContext::new())
                        .unwrap()
                        .wait()
                        .unwrap();
                    let data = read_all(&cpu, &r);
                    let e = check_f64(&t, field, wl, &points, &data, true, true);
                    assert_eq!(r.stats.fallback_count, e.fallback as u64);
                    eprintln!("{name} {aiming:?} f={field} w={wl} {e:?}");
                    cpu.release_systems(h).unwrap();
                }
            }
        }
    }
}

#[test]
fn extended_fixtures_and_failed_iterations() {
    let cpu = CpuF32Backend::new();
    let mut points = vec![[0., 0.]];
    points.extend((PupilSampling::Grid { n: 9 }).points());
    for (name, mut model) in special_systems() {
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        for field in 0..t.fields().len() {
            for wl in 0..t.num_wavelengths() {
                let (p, h) = make_plan(&cpu, &t, field, wl, &points, true, 40, true);
                let r = cpu
                    .submit_plan(p, &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let e = check_f64(&t, field, wl, &points, &read_all(&cpu, &r), true, true);
                eprintln!("fixture {name} f={field} w={wl} {e:?}");
                cpu.release_systems(h).unwrap();
            }
        }
    }
    let mut model = samples::cooke_triplet();
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    let t = testutil::tracer(model);
    let (p, h) = make_plan(&cpu, &t, 2, 0, &points, false, 0, false);
    let r = cpu
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let d = read_all(&cpu, &r);
    let sparse = aiming::unresolved(uints(&d[1]), uints(&d[2]));
    assert!(!sparse.is_empty());
    assert_eq!(sparse.len() as u64, r.stats.fallback_count);
    for (i, id, status) in sparse {
        assert_eq!(id, 0x100000000 + i as u64);
        assert_eq!(status & consts::MKL_RAY_CODE_MASK, consts::MKL_AIM_FAILED);
        assert_eq!(uints(&d[4])[i], status);
    }
    cpu.release_systems(h).unwrap();
}

#[test]
fn projection_sheets_centroid_and_telecentric_known_answers() {
    use aiming::*;
    use resident::*;
    use trace_kernel::KRayIn;
    let dummy = PlanView {
        buffer: BufferId(0),
        offset: 0,
        stride: 1,
        desc: BufferDesc::new(ElementType::DoubleFloat, [0]),
    };
    for sheet in [-1, 1] {
        for telecentric in [false, true] {
            for dz in [-1., 1.] {
                let chief = [0.25, -0.1, 0.0, 0.0, 0.0, dz, 1e6].map(Df32::from_f64);
                let centre = [0.0, 0.0, 0.0].map(Df32::from_f64);
                let r = PupilReference {
                    chief: dummy.clone(),
                    centre: dummy.clone(),
                    radius: Df32::from_f64(1000.0),
                    image_index: Df32::from_f64(1.5),
                    wavelength_um: Df32::from_f64(0.55),
                    sphere_sheet: sheet,
                    telecentric,
                };
                let mut ray = KRayIn {
                    pos: [chief[0].hi, chief[1].hi, 0.0],
                    pos_lo: [chief[0].lo, chief[1].lo, 0.0],
                    dir: [0., 0., dz as f32],
                    opl: chief[6],
                    ..Default::default()
                };
                let out = kernel::project::<Ftz>(&ray, &chief, &centre, &r).unwrap();
                assert_eq!(out[0].to_f64(), 0.0, "chief OPD");
                ray.pos = [2., 1., 0.];
                ray.pos_lo = [0.; 3];
                ray.opl = Df32::from_f64(1e6 + 0.01);
                let expected = if telecentric {
                    -0.01
                } else {
                    -0.01
                        + 1.5
                            * sheet as f64
                            * (5.0 / (1000.0 + (1e6_f64 - 5.0).sqrt())
                                - 0.0725 / (1000.0 + (1e6_f64 - 0.0725).sqrt()))
                };
                let out = kernel::project::<Ftz>(&ray, &chief, &centre, &r).unwrap();
                assert!(
                    (out[0].to_f64() - expected / 0.00055).abs() < 2e-5,
                    "{out:?}"
                );
            }
        }
    }
}
