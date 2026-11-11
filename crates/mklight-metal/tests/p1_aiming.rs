#![cfg(target_os = "macos")]
use crate::p1_support as support;
use mklight_core::{JobContext, RayAiming, samples};
use mklight_gpu::*;
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_trace::PupilSampling;
use support::*;
fn backend(library: LibraryChoice) -> MetalBackend {
    let b = MetalBackend::with_options(&MetalOptions { library })
        .unwrap()
        .expect("P1 requires host Metal hardware");
    assert!(b.precision().probe.error_free_transforms());
    eprintln!("device={:?} precision={:?}", b.info(), b.precision());
    b
}
#[test]
fn all_samples_both_libraries() {
    let mut points = vec![[0., 0.]];
    points.extend((PupilSampling::Grid { n: 7 }).points());
    for library in [LibraryChoice::Precompiled, LibraryChoice::RuntimeSource] {
        let gpu = backend(library);
        let cpu = CpuF32Backend::matching(&gpu.precision().probe);
        for (name, model) in testutil::xval_systems().into_iter().take(32) {
            for aiming in [RayAiming::Off, RayAiming::Real { tolerance: 1e-10 }] {
                let mut model = model.clone();
                model.ray_aiming = aiming;
                let t = testutil::tracer(model);
                for field in 0..t.fields().len() {
                    for wl in 0..t.num_wavelengths() {
                        let (p, h) = make_plan(&gpu, &t, field, wl, &points, true, 40, true);
                        let (pc, hc) = make_plan(&cpu, &t, field, wl, &points, true, 40, true);
                        let context = JobContext::new();
                        let r = gpu
                            .submit_plan(p.clone(), &context)
                            .unwrap()
                            .wait()
                            .unwrap();
                        assert_eq!(context.progress.fraction(), 1.0);
                        let data = read_all(&gpu, &r);
                        let rc = cpu
                            .submit_plan(pc, &JobContext::new())
                            .unwrap()
                            .wait()
                            .unwrap();
                        let dc = read_all(&cpu, &rc);
                        bits(&data, &dc);
                        let repeated = gpu
                            .submit_plan(p, &JobContext::new())
                            .unwrap()
                            .wait()
                            .unwrap();
                        bits(&data, &read_all(&gpu, &repeated));
                        let e = check_f64(&t, field, wl, &points, &data, true, true);
                        if name == "newtonian" && field == 0 {
                            for value in &doubles(&data[7])[..points.len()] {
                                if value.hi.is_finite() {
                                    assert!(
                                        value.to_f64().abs() < 1e-5,
                                        "paraboloid phase {value:?}"
                                    );
                                }
                            }
                        }

                        assert_eq!(r.stats.fallback_count, e.fallback as u64);
                        eprintln!(
                            "{name} {aiming:?} f={field} w={wl} {e:?} stats={:?}",
                            r.stats
                        );
                        gpu.release_systems(h).unwrap();
                        cpu.release_systems(hc).unwrap();
                    }
                }
            }
        }
    }
}

/// The seed-table path (plans with many rays per field row) on the device:
/// bitwise equal to the CPU-f32 specification and within the f64 gates.
#[test]
fn seed_table_matches_specification_and_reference() {
    let gpu = backend(LibraryChoice::Precompiled);
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let points = (PupilSampling::Grid { n: 64 }).points();
    assert!(points.len() >= aiming::TABLE_MIN_RAYS_PER_ROW);
    for model in [samples::cooke_triplet(), samples::double_gauss()] {
        let mut model = model;
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        let field = t.fields().len() - 1;
        let (p, h) = make_plan(&gpu, &t, field, 1, &points, false, 40, true);
        let (pc, hc) = make_plan(&cpu, &t, field, 1, &points, false, 40, true);
        let r = gpu
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let rc = cpu
            .submit_plan(pc, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let data = read_all(&gpu, &r);
        bits(&data, &read_all(&cpu, &rc));
        let e = check_f64(&t, field, 1, &points, &data, false, true);
        assert_eq!(e.fallback, 0);
        let n = points.len();
        let iterations = uints(&data[1])[n..2 * n].iter().sum::<u32>() as f64 / n as f64;
        assert!(iterations < 1.5, "mean Newton iterations {iterations}");
        eprintln!("seed table f={field} iterations={iterations} {e:?}");
        gpu.release_systems(h).unwrap();
        cpu.release_systems(hc).unwrap();
    }
}

#[test]
fn fixtures_failures_and_cancellation() {
    use std::time::{Duration, Instant};
    let gpu = backend(LibraryChoice::Precompiled);
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let mut points = vec![[0., 0.]];
    points.extend((PupilSampling::Grid { n: 9 }).points());
    for (name, mut model) in special_systems() {
        model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
        let t = testutil::tracer(model);
        for field in 0..t.fields().len() {
            for wl in 0..t.num_wavelengths() {
                let (p, h) = make_plan(&gpu, &t, field, wl, &points, true, 40, true);
                let (pc, hc) = make_plan(&cpu, &t, field, wl, &points, true, 40, true);
                let r = gpu
                    .submit_plan(p, &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let rc = cpu
                    .submit_plan(pc, &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let data = read_all(&gpu, &r);
                bits(&data, &read_all(&cpu, &rc));
                let e = check_f64(&t, field, wl, &points, &data, true, true);
                if field == 0
                    && (name == "object telecentric" || name == "image telecentric defocus")
                {
                    let hit = floats(&data[3]);
                    let n = points.len();
                    for i in 0..n {
                        if uints(&data[4])[i] == 0 {
                            let expected = if name == "object telecentric" {
                                0.0
                            } else {
                                0.5 * (1.0 - (hit[5 * n + i] as f64 + hit[14 * n + i] as f64))
                                    / (t.system().wavelengths_um[wl] * 1e-3)
                            };
                            assert!(
                                (doubles(&data[7])[i].to_f64() - expected).abs() < 1e-5,
                                "ideal-lens phase"
                            );
                        }
                    }
                }

                eprintln!("fixture {name} f={field} w={wl} {e:?}");
                gpu.release_systems(h).unwrap();
                cpu.release_systems(hc).unwrap();
            }
        }
    }
    let mut model = samples::petzval_asphere();
    model.fields.fields.push(mklight_core::Field::new(0., 500.));
    let t = testutil::tracer(model);
    let fi = t.fields().len() - 1;
    assert!(!t.field(fi).is_valid());
    let (p, h) = make_plan(&gpu, &t, fi, 1, &points, false, 40, false);
    let r = gpu
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let data = read_all(&gpu, &r);
    assert!(
        uints(&data[1])[..points.len()]
            .iter()
            .all(|&s| s == consts::MKL_AIM_FIELD_UNREACHABLE)
    );
    assert_eq!(r.stats.fallback_count, points.len() as u64);
    eprintln!(
        "P1_FAILURES reason=field_unreachable count={} total={}",
        r.stats.fallback_count,
        points.len()
    );
    gpu.release_systems(h).unwrap();
    let mut model = samples::cooke_triplet();
    model.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    let t = testutil::tracer(model);
    let (p, h) = make_plan(&gpu, &t, 0, 0, &points, false, 0, false);
    let r = gpu
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let data = read_all(&gpu, &r);
    let report = uints(&data[1]);
    let unresolved = (0..points.len())
        .filter(|&i| !matches!(report[i], 0 | consts::MKL_AIM_OUTSIDE))
        .count();
    assert!(unresolved > 0 && unresolved < points.len());
    assert_eq!(r.stats.fallback_count, unresolved as u64);
    eprintln!(
        "P1_FAILURES reason=iteration_cap count={} total={} polished={}",
        r.stats.fallback_count,
        points.len(),
        r.stats.polished_rays
    );
    assert_eq!(r.stats.polished_rays, 0);
    gpu.release_systems(h).unwrap();
    // Cancel after a Newton chunk has completed: no replacement CPU job.
    let dense = vec![[0.3, 0.8]; 200_000];
    let (p, h) = make_plan(&gpu, &t, 2, 1, &dense, false, 40, false);
    let ctx = JobContext::new();
    let pending = gpu.submit_plan(p, &ctx).unwrap();
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut last = 0.;
    loop {
        let progress = ctx.progress.fraction();
        assert!(progress >= last);
        last = progress;
        if progress > 0.1 {
            break;
        }
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
    assert!(
        last < 0.5,
        "cancellation must occur during pupil generation"
    );
    let cancel = Instant::now();
    ctx.cancel.cancel();
    assert!(matches!(
        pending.wait(),
        Err(mklight_core::MkError::Cancelled)
    ));
    eprintln!(
        "cancellation_ms={} progress={last}",
        cancel.elapsed().as_secs_f64() * 1000.
    );
    assert!(cancel.elapsed() < Duration::from_secs(5));
    gpu.release_systems(h).unwrap();
    // Concurrent plans share only immutable chief seeds and system leases.
    let (p, h) = make_plan(&gpu, &t, 1, 1, &points, true, 40, true);
    let a = gpu.submit_plan(p.clone(), &JobContext::new()).unwrap();
    let b = gpu.submit_plan(p, &JobContext::new()).unwrap();
    gpu.release_systems(h).unwrap();
    let a = a.wait().unwrap();
    let b = b.wait().unwrap();
    bits(&read_all(&gpu, &a), &read_all(&gpu, &b));
}

#[test]
fn batches_and_reordering_preserve_geometry() {
    use aiming::*;
    use resident::*;
    let gpu = backend(LibraryChoice::Precompiled);
    let t = testutil::tracer(samples::cooke_triplet());
    let points = (PupilSampling::Grid { n: 9 }).points();
    let mut references = vec![];
    for wl in 0..t.num_wavelengths() {
        let (p, h) = make_plan(&gpu, &t, 2, wl, &points, false, 40, false);
        references.push(read_all(
            &gpu,
            &gpu.submit_plan(p, &JobContext::new())
                .unwrap()
                .wait()
                .unwrap(),
        ));
        gpu.release_systems(h).unwrap();
    }
    for batch in [1, 16, 256, 1024] {
        // Each system maps packed wavelength zero to a different physical
        // wavelength. Reading system zero for every ray would change its optics.
        let packed: Vec<_> = (0..batch)
            .map(|k| {
                PackedSystem::pack(
                    t.system(),
                    &PackOptions {
                        wavelengths: Some(
                            (0..t.num_wavelengths())
                                .map(|w| (w + k) % t.num_wavelengths())
                                .collect(),
                        ),
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect();
        let handle = gpu.upload_systems(&packed).unwrap();
        let n = batch * points.len();
        for reverse in [false, true] {
            let (mut p, unused) = make_plan(&gpu, &t, 2, 1, &vec![[0., 0.]; n], false, 40, false);
            gpu.release_systems(unused).unwrap();
            let fields: Vec<_> = (0..batch)
                .map(|k| {
                    FieldSpec::new(t.system(), &t, t.field(2).point, k as u32, 0, t.aiming())
                        .unwrap()
                })
                .collect();
            let fv = upload(&gpu, &mut p, fields.len(), &fields);
            let mut samples: Vec<_> = (0..n)
                .map(|i| {
                    Sample::new(
                        points[i % points.len()],
                        (i / points.len()) as u32,
                        i as u64,
                    )
                })
                .collect();
            if reverse {
                samples.reverse();
            }
            let sv = upload(&gpu, &mut p, n, &samples);
            if let ResidentOp::Aiming(Request::GenerateAim(g)) = &mut p.operations[0] {
                g.systems = handle;
                g.field_wavelength = fv;
                g.samples = sv;
                g.field_key = None;
            }
            if let ResidentOp::Aiming(Request::TraceResident(t)) = &mut p.operations[1] {
                t.systems = handle;
            }
            let now = std::time::Instant::now();
            let result = gpu
                .submit_plan(p, &JobContext::new())
                .unwrap()
                .wait()
                .unwrap();
            let data = read_all(&gpu, &result);
            let ids = uints(&data[2]);
            for i in 0..n {
                let j = ids[i] as usize % points.len();
                let reference = &references[(ids[i] as usize / points.len()) % t.num_wavelengths()];
                for output in [0, 3] {
                    for plane in 0..15 {
                        assert_eq!(
                            floats(&data[output])[plane * n + i].to_bits(),
                            floats(&reference[output])[plane * points.len() + j].to_bits()
                        );
                    }
                }
                for plane in 0..4 {
                    assert_eq!(
                        uints(&data[1])[plane * n + i],
                        uints(&reference[1])[plane * points.len() + j]
                    );
                }
                assert_eq!(uints(&data[4])[i], uints(&reference[4])[j]);
            }
            eprintln!(
                "P1_BATCH batch={batch} rays={n} reverse={reverse} wall={} device={} peak={} fallback={}",
                now.elapsed().as_secs_f64(),
                result.stats.device_seconds,
                result.stats.peak_bytes,
                result.stats.fallback_count
            );
        }
        gpu.release_systems(handle).unwrap();
    }
}

#[test]
fn analytic_projection_on_device() {
    use aiming::*;
    use resident::*;
    let gpu = backend(LibraryChoice::Precompiled);
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    for sheet in [-1, 1] {
        for tele in [false, true] {
            for dz in [-1., 1.] {
                let make = |backend: &dyn QualificationBackend| {
                    let mut p =
                        ResidentPlan::new(mklight_core::Snapshot::new(samples::singlet()).unwrap());
                    let n = 2;
                    let r = rays(&mut p, n);
                    let mut ff = vec![0.; 15 * n];
                    let cp = [0.25, -0.1, 0.0, 0.0, 0.0, dz, 1e6].map(Df32::from_f64);
                    for i in 0..n {
                        let pos = if i == 0 {
                            [0.25, -0.1, 0.]
                        } else {
                            [2., 1., 0.]
                        };
                        for k in 0..3 {
                            let v = Df32::from_f64(pos[k]);
                            ff[k * n + i] = v.hi;
                            ff[(9 + k) * n + i] = v.lo;
                        }
                        ff[5 * n + i] = dz as f32;
                        let opl = Df32::from_f64(1e6 + if i == 0 { 0. } else { 0.01 });
                        ff[6 * n + i] = opl.hi;
                        ff[7 * n + i] = opl.lo;
                    }
                    for (id, data) in [
                        (r.floats, BufferData::F32(ff)),
                        (r.indices, BufferData::U32(vec![0; 2 * n])),
                        (r.logical_ids, BufferData::U32(vec![0; 2 * n])),
                        (r.report, BufferData::U32(vec![0; 4 * n])),
                    ] {
                        let h = backend
                            .upload_buffer(p.desc(id).unwrap().clone(), data)
                            .unwrap();
                        p.buffers[id.0] = PlanBuffer::Import(h.view());
                    }
                    let h = backend
                        .upload_buffer(
                            BufferDesc::new(ElementType::U32, [3, n]),
                            BufferData::U32(vec![0; 3 * n]),
                        )
                        .unwrap();
                    let status = p.import(h.view());
                    let hit = ResidentHits {
                        rays: r,
                        status,
                        records: None,
                        record_low: None,
                    };
                    let chief = df(backend, &mut p, &cp);
                    let centre = df(backend, &mut p, &[Df32::ZERO; 3]);
                    let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [4, n]));
                    let status = p.scratch(BufferDesc::new(ElementType::U32, [n]));
                    p.operations
                        .push(ResidentOp::Aiming(Request::ProjectPupil(ProjectPupil {
                            hits: hit,
                            reference: PupilReference {
                                chief,
                                centre,
                                radius: Df32::from_f64(1000.),
                                image_index: Df32::from_f64(1.5),
                                wavelength_um: Df32::from_f64(0.55),
                                sphere_sheet: sheet,
                                telecentric: tele,
                            },
                            output,
                            status,
                        })));
                    p.outputs = vec![output, status];
                    p
                };
                let a = gpu
                    .submit_plan(make(&gpu), &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let b = cpu
                    .submit_plan(make(&cpu), &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let a = read_all(&gpu, &a);
                bits(&a, &read_all(&cpu, &b));
                let out = doubles(&a[0]);
                assert_eq!(out[0].to_f64(), 0.);
                let expected = if tele {
                    -0.01
                } else {
                    -0.01
                        + 1.5
                            * sheet as f64
                            * (5. / (1000. + (1e6_f64 - 5.).sqrt())
                                - 0.0725 / (1000. + (1e6_f64 - 0.0725).sqrt()))
                };
                assert!((out[1].to_f64() - expected / 0.00055).abs() < 2e-5);
            }
        }
    }
}

#[test]
fn missing_primary_wavelength_is_reported() {
    use aiming::*;
    use resident::*;
    let gpu = backend(LibraryChoice::Precompiled);
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let t = testutil::tracer(samples::petzval_asphere());
    let packed = PackedSystem::pack(
        t.system(),
        &PackOptions {
            wavelengths: Some(vec![0]),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(packed.header.primary_wavelength, consts::MKL_INDEX_NONE);
    let run = |b: &dyn QualificationBackend| {
        let (mut p, old) = make_plan(b, &t, 2, 0, &[[0., 0.], [0., 0.5]], false, 40, false);
        b.release_systems(old).unwrap();
        let handle = b.upload_system(&packed).unwrap();
        for op in &mut p.operations {
            match op {
                ResidentOp::Aiming(Request::GenerateAim(g)) => g.systems = handle,
                ResidentOp::Aiming(Request::TraceResident(t)) => t.systems = handle,
                _ => {}
            }
        }
        let result = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(result.stats.fallback_count, 2);
        eprintln!("P1_FAILURES reason=missing_primary count=2 total=2");
        let data = read_all(b, &result);
        assert_eq!(
            &uints(&data[1])[..2],
            &[consts::MKL_AIM_PRIMARY_UNAVAILABLE; 2]
        );
        b.release_systems(handle).unwrap();
        data
    };
    bits(&run(&gpu), &run(&cpu));
}

#[test]
fn degenerate_finite_launch_is_not_converged() {
    use aiming::*;
    use resident::*;
    let gpu = backend(LibraryChoice::Precompiled);
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let mut model = samples::finite_relay();
    model.ray_aiming = RayAiming::Off;
    let t = testutil::tracer(model);
    let run = |b: &dyn QualificationBackend| {
        let (mut p, h) = make_plan(b, &t, 0, 0, &[[0., 0.], [0.5, 0.]], false, 40, false);
        let mut field =
            FieldSpec::new(t.system(), &t, t.field(0).point, 0, 0, RayAiming::Off).unwrap();
        // An aim plane through the source produces either a zero-length line
        // or a horizontal line. Neither can launch toward the optical train.
        field.pupil[0] = field.origin[2];
        field.source = [Df32::ZERO; 2];
        let fv = upload(b, &mut p, 1, &[field]);
        if let ResidentOp::Aiming(Request::GenerateAim(g)) = &mut p.operations[0] {
            g.field_wavelength = fv;
            g.field_key = None;
        }
        let result = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        assert_eq!(result.stats.fallback_count, 2);
        let data = read_all(b, &result);
        let expected =
            consts::MKL_AIM_FAILED | ((t.system().stop as u32) << consts::MKL_RAY_SURFACE_SHIFT);
        assert_eq!(&uints(&data[1])[..2], &[expected; 2]);
        assert_eq!(&uints(&data[4])[..2], &[expected; 2]);
        assert_eq!(&uints(&data[4])[2..4], &[t.system().stop as u32; 2]);
        b.release_systems(h).unwrap();
        data
    };
    bits(&run(&gpu), &run(&cpu));
}

#[test]
fn resident_path_can_start_backwards() {
    use aiming::*;
    use resident::*;
    let gpu = backend(LibraryChoice::Precompiled);
    let cpu = CpuF32Backend::matching(&gpu.precision().probe);
    let t = testutil::tracer(samples::singlet());
    let packed = PackedSystem::pack_all(t.system()).unwrap();
    let run = |b: &dyn QualificationBackend| {
        let h = b.upload_system(&packed).unwrap();
        let mut p =
            ResidentPlan::new(mklight_core::Snapshot::new(t.system().system().clone()).unwrap());
        let input = rays(&mut p, 1);
        for (id, data) in [
            (
                input.floats,
                BufferData::F32(vec![
                    0.5, 0., 0., 0., 0., -1., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
                ]),
            ),
            (input.indices, BufferData::U32(vec![0, 1])),
            (input.logical_ids, BufferData::U32(vec![7, 0])),
            (input.report, BufferData::U32(vec![0; 4])),
        ] {
            let buffer = b.upload_buffer(p.desc(id).unwrap().clone(), data).unwrap();
            p.buffers[id.0] = PlanBuffer::Import(buffer.view());
        }
        let start = t.system().image_index() as u32;
        let steps: Vec<_> = (0..start)
            .rev()
            .map(|surface| {
                PathStep {
                    surface,
                    mode: StepMode::Transmit,
                    backward: true,
                }
                .encode()
            })
            .collect();
        let buffer = b
            .upload_buffer(
                BufferDesc::new(ElementType::U32, [steps.len()]),
                BufferData::U32(steps),
            )
            .unwrap();
        let si = p.import(buffer.view());
        let hit = hits(&mut p, 1, start as usize + 1);
        p.operations
            .push(ResidentOp::Aiming(Request::TraceResident(TraceResident {
                systems: h,
                input,
                steps: p.view(si).unwrap(),
                start_surface: start,
                apply_apertures: false,
                radiometry: 0,
                newton: Default::default(),
                output: hit.clone(),
            })));
        p.outputs = vec![
            hit.rays.floats,
            hit.status,
            hit.records.unwrap(),
            hit.record_low.unwrap(),
        ];
        let r = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let data = read_all(b, &r);
        assert_eq!(uints(&data[1]), &[0, 0, start]);
        b.release_systems(h).unwrap();
        data
    };
    bits(&run(&gpu), &run(&cpu));
}
