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
    // Unresolved rays keep their logical IDs and failure reports.
    let (report, ids) = (uints(&d[1]), uints(&d[2]));
    let n = points.len();
    let failed: Vec<_> = (0..n)
        .filter(|&i| !matches!(report[i], 0 | consts::MKL_AIM_OUTSIDE))
        .collect();
    assert!(!failed.is_empty());
    assert_eq!(failed.len() as u64, r.stats.fallback_count);
    for i in failed {
        let id = ids[i] as u64 | ((ids[n + i] as u64) << 32);
        assert_eq!(id, 0x100000000 + i as u64);
        assert_eq!(
            report[i] & consts::MKL_RAY_CODE_MASK,
            consts::MKL_AIM_FAILED
        );
        assert_eq!(uints(&d[4])[i], report[i]);
    }
    cpu.release_systems(h).unwrap();
}

/// Plans with enough rays per field row aim from the interpolated seed table:
/// the same rays as the f64 reference, in about two stop traces per ray
/// (chief-seeded Newton needs four to six on this field).
#[test]
fn seed_table_reaches_reference_rays_in_fewer_iterations() {
    let cpu = CpuF32Backend::new();
    let mut model = samples::cooke_triplet();
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    let t = testutil::tracer(model);
    let points = (PupilSampling::Grid { n: 64 }).points();
    assert!(points.len() >= aiming::TABLE_MIN_RAYS_PER_ROW);
    let (p, h) = make_plan(&cpu, &t, 2, 1, &points, false, 40, true);
    let r = cpu
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let data = read_all(&cpu, &r);
    let e = check_f64(&t, 2, 1, &points, &data, false, true);
    assert_eq!(e.fallback, 0);
    let n = points.len();
    let iterations = uints(&data[1])[n..2 * n].iter().sum::<u32>() as f64 / n as f64;
    assert!(iterations < 1.5, "mean Newton iterations {iterations}");
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

/// Host seed tables solve each row with one batched `generate_many`; every
/// node must be bitwise the scalar `generate` of the same pre-vignetting
/// point (status and aim), on every sample (also vignetted), field and
/// wavelength.
#[test]
fn host_tables_match_scalar_generation_bitwise() {
    let side = 13u32;
    let (s, nodes) = (side as usize, (side * side) as usize);
    let step = 2.25 / (s - 1) as f64;
    let mut solved = 0;
    for (name, mut model) in testutil::xval_systems() {
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let rows: Vec<_> = t
            .fields()
            .iter()
            .flat_map(|f| (0..t.num_wavelengths()).map(move |w| (f, w)))
            .collect();
        let table = aiming::host_tables(&t, &rows, side);
        assert_eq!(table.len(), rows.len() * nodes);
        for (r, &(field, w)) in rows.iter().enumerate() {
            let v = field.point.vignetting;
            for k in 0..nodes {
                let node = &table[r * nodes + k];
                let p = [(k % s) as f64 * step - 1.125, (k / s) as f64 * step - 1.125];
                let expected = (field.is_valid()
                    && v.vcx < 1.
                    && v.vcy < 1.
                    && p[0] * p[0] + p[1] * p[1] <= 1.5625)
                    .then(|| {
                        let px = (p[0] - v.vdx) / (1. - v.vcx);
                        let py = (p[1] - v.vdy) / (1. - v.vcy);
                        t.generate(field, w, px, py).ok()
                    })
                    .flatten();
                match expected {
                    Some(g) => {
                        solved += 1;
                        assert_eq!(node.status, 0, "{name} row {r} node {k}");
                        let aim = g.aim.map(Df32::from_f64);
                        assert_eq!(
                            [aim[0].hi, aim[0].lo, aim[1].hi, aim[1].lo].map(f32::to_bits),
                            [
                                node.aim[0].hi,
                                node.aim[0].lo,
                                node.aim[1].hi,
                                node.aim[1].lo
                            ]
                            .map(f32::to_bits),
                            "{name} row {r} node {k}"
                        );
                    }
                    None => assert_eq!(node.status, 1, "{name} row {r} node {k}"),
                }
            }
        }
    }
    assert!(solved > 10_000, "{solved} solved nodes");
}
