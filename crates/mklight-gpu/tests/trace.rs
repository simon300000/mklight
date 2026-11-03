//! The trace kernel specification (CPU-f32 emulation) against the f64
//! reference: the cross-validation suite over every sample and fixture,
//! path programs against an independent f64 path computation, status
//! classification, radiometry, records, request validation, job mechanics
//! and precision qualification.

use mklight_core::job::{Precision, RunInfo};
use mklight_core::materials::coating::fresnel;
use mklight_core::model::{CoatingDef, Interaction};
use mklight_core::{JobContext, MkError, OpticalSystem, SurfaceShape, Vec3, samples};
use mklight_gpu::batch::RECORD_FLOATS;
use mklight_gpu::consts::*;
use mklight_gpu::testutil::{self, f64_path};
use mklight_gpu::xval::{XvalSettings, cross_validate, qualify, reference_status_word};
use mklight_gpu::*;
use mklight_trace::{
    IntersectSettings, PupilSampling, Ray, TraceOptions, trace_ray, trace_ray_recorded,
};

fn trace(b: &dyn ComputeBackend, req: TraceRequest) -> TraceResult {
    match b
        .run(Job::Trace(Box::new(req)), &JobContext::new())
        .unwrap()
        .output
    {
        JobOutput::Trace(t) => *t,
        o => panic!("{o:?}"),
    }
}

fn batch(rays: &[(Ray, u32)]) -> RayBatch {
    let mut b = RayBatch::with_capacity(rays.len());
    for (r, w) in rays {
        b.push_f64([r.pos.x, r.pos.y, r.pos.z], [r.dir.x, r.dir.y, r.dir.z], *w);
    }
    b
}

fn v3(r: &TraceResult, i: usize) -> (Vec3, Vec3) {
    (
        Vec3::new(r.x[i] as f64, r.y[i] as f64, r.z[i] as f64),
        Vec3::new(r.l[i] as f64, r.m[i] as f64, r.n[i] as f64),
    )
}

/// The full cross-validation suite on the emulation (flush-to-zero, as Apple
/// GPUs): every sample (plain and vignetted) and fixture, all fields and
/// wavelengths, a 64 × 64 pupil grid, unpolarised radiometry. Every
/// quantity must meet the CONVENTIONS §7 budget and survival must agree
/// outside the tolerance bands.
#[test]
fn cross_validation_suite_cpu_f32() {
    let b = CpuF32Backend::with_subnormals(false);
    let s = XvalSettings::default();
    assert_eq!(s.pupil, PupilSampling::Grid { n: 64 });
    for (name, sys) in testutil::xval_systems() {
        let t = testutil::tracer(sys);
        let r = cross_validate(&b, &t, &s, &JobContext::new()).unwrap();
        eprintln!(
            "{name:28} rays {:6} compared {:6} pos {:.2e} dir {:.2e} opd {:.2e} trans {:.1e} band {}",
            r.rays,
            r.compared,
            r.max_position_mm,
            r.max_direction,
            r.max_opd_waves,
            r.max_transmittance_abs,
            r.band_mismatches
        );
        r.check(&s.budget)
            .unwrap_or_else(|e| panic!("{name}: {e} {:?}", r.notes));
        assert!(r.compared > 1000, "{name}: {r:?}");
        assert_eq!(r.opd_compared, r.compared, "{name}");
        assert!(r.max_transmittance_abs < 1e-6, "{name}: {r:?}");
    }
}

/// Ghost-type path programs (two reflections inside a plate, forward and
/// backward steps, straight and tilted/decentred) against the independent
/// f64 path computation: point, direction, optical path and the Fresnel
/// weight `T·R·R·T`.
#[test]
fn plate_two_reflection_path() {
    let b = CpuF32Backend::with_subnormals(false);
    for tilted in [false, true] {
        let t = testutil::tracer(testutil::plate(tilted));
        let sys = t.system();
        let h = b
            .upload_system(&PackedSystem::pack_all(sys).unwrap())
            .unwrap();
        use StepMode::*;
        let prog = vec![
            PathStep::forward(1, Transmit),
            PathStep::forward(2, Reflect),
            PathStep::backward(1, Reflect),
            PathStep::forward(2, Transmit),
            PathStep::forward(3, Transmit),
        ];
        let mut rays = Vec::new();
        for (k, &(x, y, l, m)) in [
            (0.0, 0.0, 0.0, 0.0),
            (1.5, -2.0, 0.08, 0.15),
            (-3.0, 2.5, -0.2, 0.1),
            (4.0, 3.0, 0.05, -0.3),
        ]
        .iter()
        .enumerate()
        {
            let d = Vec3::new(l, m, 1.0).normalized();
            rays.push((Ray::new(Vec3::new(x, y, -5.0), d), (k % 3) as u32));
        }
        let mut req = TraceRequest::path(h, batch(&rays), 0, prog.clone());
        req.record = RecordMode::AllSurfaces;
        let r = trace(&b, req);
        for (i, (ray, w)) in rays.iter().enumerate() {
            let want = f64_path(sys, 0, ray.pos, ray.dir, *w as usize, &prog).unwrap();
            assert!(r.ray_status(i).is_ok(), "{:?}", r.ray_status(i));
            assert_eq!(r.steps[i], 5);
            assert_eq!(r.last_surface[i], 3);
            let (p, d) = v3(&r, i);
            assert!(
                (p - want.pos).norm() < 1e-6,
                "tilted {tilted} ray {i}: {p:?} {:?}",
                want.pos
            );
            assert!((d - want.dir).norm() < 1e-7, "tilted {tilted} ray {i}");
            // The reported OPL is the path to the reported (f32) point.
            let want_opl = want.opl + (p - want.pos).dot(want.dir);
            assert!(
                (r.opl_f64(i) - want_opl).abs() < 1e-9,
                "tilted {tilted} ray {i}: {} vs {want_opl}",
                r.opl_f64(i)
            );
            // Two internal reflections of a bare plate: ≈ 0.04² of the flux.
            assert!(want.weight > 1e-3 && want.weight < 4e-3, "{}", want.weight);
            assert!((r.weight[i] as f64 / want.weight - 1.0).abs() < 1e-5);
            assert_eq!(r.transmittance[i], 1.0);
            // Record slot 3 is the state after the backward reflection at 1.
            let k = (3 * rays.len() + i) * RECORD_FLOATS;
            assert!(
                r.recorded[k + 5] > 0.0,
                "travelling +z again after two reflections"
            );
        }
        // Without radiometry the weight passes unchanged.
        let mut req = TraceRequest::path(h, batch(&rays), 0, prog.clone());
        req.radiometry = Radiometry::Off;
        assert!(trace(&b, req).weight.iter().all(|w| *w == 1.0));
    }
}

/// `T + R = 1` at a bare interface: one Transmit and one Reflect step on
/// the same rays split the flux exactly (to f32 rounding).
#[test]
fn transmit_plus_reflect_conserves_flux() {
    let b = CpuF32Backend::new();
    let t = testutil::tracer(testutil::plate(true));
    let h = b
        .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
        .unwrap();
    let rays: Vec<_> = (0..50)
        .map(|k| {
            let a = k as f64 * 0.02;
            (
                Ray::new(
                    Vec3::new(0.1 * k as f64, 0.0, -1.0),
                    Vec3::new(a.sin(), 0.0, a.cos()),
                ),
                1,
            )
        })
        .collect();
    let tr = trace(
        &b,
        TraceRequest::path(
            h,
            batch(&rays),
            0,
            vec![PathStep::forward(1, StepMode::Transmit)],
        ),
    );
    let rf = trace(
        &b,
        TraceRequest::path(
            h,
            batch(&rays),
            0,
            vec![PathStep::forward(1, StepMode::Reflect)],
        ),
    );
    for i in 0..rays.len() {
        let s = tr.weight[i] as f64 + rf.weight[i] as f64;
        assert!((s - 1.0).abs() < 3e-7, "ray {i}: T + R = {s}");
        assert!(
            rf.weight[i] > 0.04 && rf.weight[i] < 0.25,
            "{}",
            rf.weight[i]
        );
    }
}

/// Sequential radiometry: bare Fresnel transmittance at every refracting
/// surface and bulk absorption, against core Fresnel along the f64 trace's
/// records; ideal coatings scale by their constant `T`.
#[test]
fn sequential_radiometry_matches_reference() {
    let b = CpuF32Backend::new();
    for coated in [false, true] {
        let mut sys = testutil::absorbing_triplet();
        if coated {
            sys.coatings.insert(
                "AR".into(),
                CoatingDef::Ideal {
                    reflectance: 0.004,
                    transmittance: 0.99,
                },
            );
            sys.surfaces[1].coating = Some("AR".into());
            sys.surfaces[6].coating = Some("AR".into());
        }
        let t = testutil::tracer(sys);
        let s = t.system();
        let h = b
            .upload_system(&PackedSystem::pack_all(s).unwrap())
            .unwrap();
        let mut rays = Vec::new();
        for f in 0..3 {
            for p in (PupilSampling::Hexapolar { rings: 3 }).points() {
                let g = t.generate(t.field(f), 0, p[0], p[1]).unwrap();
                rays.push((g.ray, 0u32));
            }
        }
        let r = trace(
            &b,
            TraceRequest::new(h, batch(&rays), 0, s.image_index() as u32),
        );
        for (i, (ray, _)) in rays.iter().enumerate() {
            let (res, rec) = trace_ray_recorded(s, ray, 0, &t.options());
            assert!(res.status.is_ok());
            let mut want = res.transmittance;
            assert!(want < 0.99, "the test glass absorbs");
            for (j, surf) in s.surfaces.iter().enumerate().take(s.image_index()).skip(1) {
                if surf.is_passive() {
                    continue;
                }
                let tj = match surf.coating {
                    mklight_core::materials::coating::ResolvedCoating::Ideal {
                        transmittance,
                        ..
                    } => transmittance,
                    _ => fresnel(s.n_before(j, 0), surf.n_after[0], rec[j].cos_incidence)
                        .t_unpolarized(),
                };
                want *= tj;
            }
            assert!((r.transmittance[i] as f64 - res.transmittance).abs() < 1e-6);
            assert!(
                (r.weight[i] as f64 / want - 1.0).abs() < 2e-6,
                "ray {i}: {} vs {want}",
                r.weight[i]
            );
        }
    }
}

fn one_surface(shape: SurfaceShape) -> OpticalSystem {
    let mut s = testutil::plate(false);
    s.surfaces[1].shape = shape;
    s.surfaces[1].material = mklight_core::MaterialRef::Air;
    s.surfaces[1].thickness = 30.0;
    s.surfaces.remove(2);
    s
}

/// Every failure class is reported with the reference's status and surface:
/// misses, asphere non-convergence at the same step cap, leaving the sag
/// domain, TIR, vignetting and non-finite input.
#[test]
fn status_classification_matches_reference() {
    let b = CpuF32Backend::with_subnormals(false);
    let unit = |x: f64, y: f64, z: f64| Vec3::new(x, y, z).normalized();
    struct Case {
        sys: OpticalSystem,
        ray: Ray,
        cap: u32,
        want: u32,
    }
    let asph = |c: f64, a: Vec<f64>| {
        one_surface(SurfaceShape::EvenAsphere {
            curvature: c,
            conic: 0.0,
            coefficients: a,
        })
    };
    let cases = vec![
        // A ray beside a small sphere.
        Case {
            sys: one_surface(SurfaceShape::sphere(5.0)),
            ray: Ray::new(Vec3::new(0.0, 6.0, -1.0), Vec3::Z),
            cap: 8,
            want: MKL_RAY_MISS | 1 << 8,
        },
        // An asphere the ray never crosses: a miss, not a numerical failure.
        Case {
            sys: asph(0.0, vec![1e-3]),
            ray: Ray::new(Vec3::new(0.0, 9.0, -1.0), unit(0.0, 0.5, 1.0)),
            cap: 40,
            want: MKL_RAY_MISS | 1 << 8,
        },
        // Newton capped at one step while still converging.
        Case {
            sys: asph(1.0 / 30.0, vec![2e-5]),
            ray: Ray::new(Vec3::new(0.0, 8.0, -12.0), unit(0.0, 0.25, 1.0)),
            cap: 1,
            want: MKL_RAY_NOT_CONVERGED | 1 << 8,
        },
        // Beyond a sphere's rim the sag is undefined.
        Case {
            sys: asph(1.0 / 30.0, vec![2e-5]),
            ray: Ray::new(Vec3::new(0.0, 31.0, -1.0), Vec3::Z),
            cap: 8,
            want: u32::MAX,
        },
        // Incidence sin i = 0.8 inside N-BK7 at the block's rear sphere.
        Case {
            sys: testutil::tir_block(),
            ray: Ray::new(Vec3::new(0.0, 8.0, -1.0), Vec3::Z),
            cap: 8,
            want: MKL_RAY_TIR | 2 << 8,
        },
        // Vignetted by the fixed semi-diameter.
        Case {
            sys: testutil::singlet(),
            ray: Ray::new(Vec3::new(0.0, 12.8, -1.0), Vec3::Z),
            cap: 8,
            want: MKL_RAY_VIGNETTED | 1 << 8,
        },
        // Non-finite input is reported at the start surface.
        Case {
            sys: testutil::singlet(),
            ray: Ray::new(Vec3::new(f64::NAN, 0.0, -1.0), Vec3::Z),
            cap: 8,
            want: MKL_RAY_NONFINITE,
        },
    ];
    for (k, c) in cases.into_iter().enumerate() {
        let rs = testutil::resolve(c.sys);
        let opts = TraceOptions {
            intersect: IntersectSettings {
                tolerance: 1e-12,
                max_iterations: c.cap,
            },
            ..Default::default()
        };
        let reference = reference_status_word(&trace_ray(&rs, &c.ray, 1, &opts, None).status);
        if c.want != u32::MAX {
            assert_eq!(reference, c.want, "case {k}: the reference itself");
        }
        let h = b
            .upload_system(&PackedSystem::pack_all(&rs).unwrap())
            .unwrap();
        let mut req = TraceRequest::new(h, batch(&[(c.ray, 1)]), 0, rs.image_index() as u32);
        req.newton.max_iterations = c.cap;
        let r = trace(&b, req);
        assert_eq!(r.status[0], reference, "case {k}: {:?}", r.ray_status(0));
        assert_eq!(r.last_surface[0], reference >> 8, "case {k}");
    }
}

/// Per-surface records against the reference's records; a ray that fails
/// repeats its final state and status in later slots.
#[test]
fn records_follow_the_reference() {
    let b = CpuF32Backend::new();
    let t = testutil::tracer(testutil::vignetted(&samples::double_gauss(), 0.9));
    let s = t.system();
    let h = b
        .upload_system(&PackedSystem::pack_all(s).unwrap())
        .unwrap();
    let mut rays = Vec::new();
    for p in (PupilSampling::Grid { n: 9 }).points() {
        rays.push((t.generate(t.field(2), 1, p[0], p[1]).unwrap().ray, 1u32));
    }
    let n = rays.len();
    let mut req = TraceRequest::new(h, batch(&rays), 0, s.image_index() as u32);
    req.record = RecordMode::AllSurfaces;
    let r = trace(&b, req);
    let slots = s.num_surfaces();
    assert_eq!(r.recorded.len(), slots * n * RECORD_FLOATS);
    let mut failed = 0;
    for (i, (ray, _)) in rays.iter().enumerate() {
        let (res, rec) = trace_ray_recorded(s, ray, 1, &t.options());
        assert_eq!(r.status[i], reference_status_word(&res.status));
        for (j, want) in rec.iter().enumerate().skip(1) {
            let k = (j * n + i) * RECORD_FLOATS;
            let p = Vec3::new(
                r.recorded[k] as f64,
                r.recorded[k + 1] as f64,
                r.recorded[k + 2] as f64,
            );
            assert!((p - want.point).norm() < 2e-6, "ray {i} surface {j}");
            let opl = r.recorded[k + 6] as f64 + r.recorded[k + 7] as f64;
            assert!((opl - want.opl).abs() < 1e-6, "ray {i} surface {j}");
            assert_eq!(r.recorded_status[j * n + i], MKL_RAY_OK);
        }
        if !res.status.is_ok() {
            failed += 1;
            for j in rec.len()..slots {
                assert_eq!(r.recorded_status[j * n + i], r.status[i]);
                let k = (j * n + i) * RECORD_FLOATS;
                assert_eq!(r.recorded[k], r.x[i]);
            }
        }
    }
    assert!(
        failed > 0,
        "the 0.9 semi-diameters vignette part of the field-edge pupil"
    );
}

/// Requests are checked against the stored systems before anything runs.
#[test]
fn trace_requests_are_validated() {
    let b = CpuF32Backend::new();
    let mut sys = samples::newtonian_telescope();
    sys.coatings.insert(
        "HR".into(),
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![],
            mirror_substrate: Some((1.0, 6.0)),
        },
    );
    sys.surfaces[1].coating = Some("HR".into());
    let rs = testutil::resolve(sys);
    let h = b
        .upload_system(&PackedSystem::pack_all(&rs).unwrap())
        .unwrap();
    let ray = [(Ray::new(Vec3::new(0.0, 10.0, -1.0), Vec3::Z), 0u32)];
    let submit = |req: TraceRequest| {
        b.submit(Job::Trace(Box::new(req)), &JobContext::new())
            .map(|_| ())
    };
    let seq = |rays: RayBatch| TraceRequest::new(h, rays, 0, 5);
    // A thin-film coating is not available in GPU radiometry.
    assert!(matches!(
        submit(seq(batch(&ray))),
        Err(MkError::Unsupported(_))
    ));
    let mut ok = seq(batch(&ray));
    ok.radiometry = Radiometry::Off;
    assert!(submit(ok.clone()).is_ok());
    use StepMode::*;
    let bad_programs: Vec<Vec<PathStep>> = vec![
        vec![],
        vec![PathStep::forward(2, Transmit)],
        vec![
            PathStep::forward(1, Sequential),
            PathStep::backward(0, Sequential),
        ],
        // A mirror's reflection keeps the direction through the sequence.
        vec![
            PathStep::forward(1, Reflect),
            PathStep::backward(0, Transmit),
        ],
        // There is no medium behind a mirror.
        vec![PathStep::forward(1, Transmit)],
        // The sensor ghost travels back through the fold flat by its
        // reflection, never through it.
        vec![
            PathStep::forward(1, Sequential),
            PathStep::forward(2, Sequential),
            PathStep::forward(3, Sequential),
            PathStep::forward(4, Sequential),
            PathStep::forward(5, Reflect),
            PathStep::backward(4, Transmit),
            PathStep::backward(3, Transmit),
        ],
        // A ghost reflection at a mirror would continue backward.
        vec![
            PathStep::forward(1, Sequential),
            PathStep::forward(2, Sequential),
            PathStep::forward(3, Reflect),
            PathStep::backward(2, Transmit),
        ],
        // Reflection at a coordinate break.
        vec![
            PathStep::forward(1, Sequential),
            PathStep::forward(2, Reflect),
        ],
        vec![
            PathStep::forward(1, Sequential),
            PathStep::forward(2, Transmit),
            PathStep::forward(3, Sequential),
            PathStep::forward(4, Transmit),
            PathStep::forward(5, Transmit),
            PathStep::forward(6, Transmit),
        ],
    ];
    let good_programs: Vec<Vec<PathStep>> = vec![
        vec![
            PathStep::forward(1, Reflect),
            PathStep::forward(2, Transmit),
        ],
        vec![
            PathStep::forward(1, Sequential),
            PathStep::forward(2, Sequential),
            PathStep::forward(3, Sequential),
            PathStep::forward(4, Sequential),
            PathStep::forward(5, Reflect),
            PathStep::backward(4, Transmit),
            PathStep::backward(3, Reflect),
            PathStep::backward(2, Transmit),
            PathStep::backward(1, Reflect),
        ],
    ];
    for (k, p) in good_programs.into_iter().enumerate() {
        let mut r = TraceRequest::path(h, batch(&ray), 0, p);
        r.radiometry = Radiometry::Off;
        assert!(submit(r).is_ok(), "good program {k}");
    }
    for (k, p) in bad_programs.into_iter().enumerate() {
        let mut r = TraceRequest::path(h, batch(&ray), 0, p);
        r.radiometry = Radiometry::Off;
        assert!(submit(r).is_err(), "program {k}");
    }
    let mut r = ok.clone();
    r.rays.wavelength[0] = 3;
    assert!(submit(r).is_err(), "wavelength out of range");
    let mut r = ok.clone();
    r.rays.system = vec![1];
    assert!(submit(r).is_err(), "system out of range");
    let mut r = ok.clone();
    r.newton.tolerance_rel = f32::NAN;
    assert!(submit(r).is_err());
    let mut r = ok.clone();
    r.rays.l_lo.clear();
    assert!(submit(r).is_err(), "partial residual planes");
    // Encoding round trip and unknown bits.
    for st in [
        PathStep::forward(7, Reflect),
        PathStep::backward(65535, Transmit),
    ] {
        assert_eq!(PathStep::decode(st.encode()), Some(st));
    }
    assert_eq!(PathStep::decode(1 << 25), None);
}

/// Cancellation, monotone progress to 1 and chunking of trace jobs.
#[test]
fn trace_job_mechanics() {
    let b = CpuF32Backend::new();
    let t = testutil::tracer(samples::cooke_triplet());
    let s = t.system();
    let h = b
        .upload_system(&PackedSystem::pack_all(s).unwrap())
        .unwrap();
    let g = t.generate(t.field(1), 1, 0.3, 0.2).unwrap().ray;
    let rays = vec![(g, 1u32); 150_000];
    let req = || TraceRequest::new(h, batch(&rays), 0, s.image_index() as u32);
    let ctx = JobContext::new();
    let r = b.run(Job::Trace(Box::new(req())), &ctx).unwrap();
    assert!(r.stats.command_buffers >= 3);
    assert_eq!(ctx.progress.fraction(), 1.0);
    let JobOutput::Trace(out) = r.output else {
        panic!()
    };
    assert!(
        out.x.iter().all(|x| x.to_bits() == out.x[0].to_bits()),
        "deterministic"
    );
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(
        b.run(Job::Trace(Box::new(req())), &ctx),
        Err(MkError::Cancelled)
    );
    let empty = TraceRequest::new(h, RayBatch::default(), 0, s.image_index() as u32);
    let ctx = JobContext::new();
    let JobOutput::Trace(e) = b.run(Job::Trace(Box::new(empty)), &ctx).unwrap().output else {
        panic!()
    };
    assert!(e.is_empty());
    assert_eq!(ctx.progress.fraction(), 1.0);
}

/// Normal-range data trace identically in the IEEE and flush-to-zero
/// environments.
#[test]
fn ieee_and_ftz_agree_on_normal_data() {
    let t = testutil::tracer(samples::petzval_asphere());
    let s = t.system();
    let p = PackedSystem::pack_all(s).unwrap();
    let mut rays = Vec::new();
    for p in (PupilSampling::Grid { n: 16 }).points() {
        rays.push((t.generate(t.field(2), 2, p[0], p[1]).unwrap().ray, 2u32));
    }
    let run = |b: CpuF32Backend| {
        let h = b.upload_system(&p).unwrap();
        trace(
            &b,
            TraceRequest::new(h, batch(&rays), 0, s.image_index() as u32),
        )
    };
    assert_eq!(
        run(CpuF32Backend::new()),
        run(CpuF32Backend::with_subnormals(false))
    );
}

/// The packed double-float parameters reproduce the model, and ideal-lens
/// vergences are the reference's.
#[test]
fn packed_double_float_parameters() {
    let rs = testutil::resolve(testutil::ideal_lenses());
    let p = PackedSystem::pack_all(&rs).unwrap();
    for (i, s) in rs.surfaces.iter().enumerate() {
        for w in 0..3 {
            let n = p.surfaces[i].n_after[w] as f64 + p.surfaces[i].n_after_lo[w] as f64;
            assert!((n - s.n_after[w]).abs() < 1e-14);
            if matches!(s.shape, SurfaceShape::Paraxial { .. }) {
                let v = mklight_trace::paraxial::ideal_lens_object_vergence(&rs, i, w);
                let ps = &p.surfaces[i];
                assert_eq!(ps.vergence[w], v as f32);
                assert!(
                    (ps.vergence[w] as f64 + ps.vergence_lo[w] as f64 - v).abs() < 1e-14 * v.abs()
                );
                assert!(v != 0.0);
                let SurfaceShape::Paraxial { focal_length } = s.shape else {
                    unreachable!()
                };
                let phi = ps.paraxial_power as f64 + ps.paraxial_power_lo as f64;
                assert!((phi * focal_length - 1.0).abs() < 1e-14);
            }
        }
    }
    let rs = testutil::resolve(testutil::asphere_heavy());
    let p = PackedSystem::pack_all(&rs).unwrap();
    let SurfaceShape::EvenAsphere {
        curvature, conic, ..
    } = rs.surfaces[1].shape
    else {
        panic!()
    };
    let s1 = &p.surfaces[1];
    assert!((s1.curvature as f64 + s1.curvature_lo as f64 - curvature).abs() < 1e-15);
    assert!((s1.conic as f64 + s1.conic_lo as f64 - conic).abs() < 1e-15);
    assert_ne!(s1.curvature_lo, 0.0, "1/3 is not an f32");
    // Asphere coefficients, normalised by R = 2^e: hi + lo = A·R^(2j+4).
    let SurfaceShape::EvenAsphere { coefficients, .. } = &rs.surfaces[2].shape else {
        panic!()
    };
    let s2 = &p.surfaces[2];
    let r2 = 1.0 / s2.asphere_inv_norm2 as f64;
    for (j, a) in coefficients.iter().enumerate() {
        let want = a * r2.powi(j as i32 + 2);
        let got = s2.asphere[j] as f64 + s2.asphere_lo[j] as f64;
        assert!((got - want).abs() <= 1e-14 * want.abs(), "A{}", 2 * j + 4);
    }
    assert!(s2.asphere_lo.iter().any(|&l| l != 0.0));
    // Aperture sizes, with the edge tolerance of `ApertureShape::passes`.
    let rs = testutil::resolve(testutil::apertures());
    let p = PackedSystem::pack_all(&rs).unwrap();
    let mut checked = 0;
    for (i, s) in rs.surfaces.iter().enumerate() {
        let ps = &p.surfaces[i];
        let tolerant = s.aperture.as_ref().map(|a| a.with_edge_tolerance());
        let sd = s
            .fixed_semi_diameter
            .map(|r| r * (1.0 + mklight_core::APERTURE_EDGE_TOLERANCE));
        let sizes: Vec<f64> = match (&tolerant, sd) {
            (
                Some(mklight_core::model::ApertureShape::Annular {
                    inner_radius,
                    outer_radius,
                }),
                _,
            ) => vec![*inner_radius, *outer_radius],
            (Some(mklight_core::model::ApertureShape::Elliptical { semi_x, semi_y }), _) => {
                vec![*semi_x, *semi_y]
            }
            (Some(mklight_core::model::ApertureShape::Rectangular { half_x, half_y }), _) => {
                vec![*half_x, *half_y]
            }
            (None, Some(r)) => vec![r],
            _ => continue,
        };
        for (k, v) in sizes.into_iter().enumerate() {
            let got = ps.aperture_params[k] as f64 + ps.aperture_params_lo[k] as f64;
            assert!((got - v).abs() < 1e-13, "surface {i}");
            checked += 1;
        }
    }
    assert!(checked >= 7);
}

/// Precision qualification passes on a sample, records the backend in
/// `RunInfo`, and fails, with the CPU fallback recorded, when the budget
/// cannot be met or the system cannot run on the backend.
#[test]
fn qualification_and_fallback_recording() {
    let b = CpuF32Backend::with_subnormals(false);
    let t = testutil::tracer(samples::double_gauss());
    let ctx = JobContext::new();
    let q = qualify(&b, &t, None, &ctx).unwrap();
    assert!(q.passed, "{q:?}");
    assert!(q.report.as_ref().unwrap().compared > 500);
    let mut info = RunInfo {
        engine_version: "test".into(),
        snapshot_hash: String::new(),
        analysis: "spot".into(),
        settings: serde_json::json!({ "rays": 100 }),
        backend: String::new(),
        precision: Precision::F64Reference,
        seed: None,
        started_unix_ms: 0,
        duration_ms: 0.0,
        warnings: vec![],
        failed_attempts: vec![],
    };
    q.record(&mut info);
    assert!(info.backend.starts_with("cpu-f32:"));
    assert_eq!(info.precision, Precision::F32Gpu);
    assert_eq!(info.settings["rays"], 100);
    assert_eq!(info.settings["backend_qualification"]["passed"], true);

    // A budget no f32 backend can meet: fails with the measured reason.
    let mut strict = XvalSettings::qualification();
    strict.budget.position_mm = 1e-12;
    let q = qualify(&b, &t, Some(&strict), &ctx).unwrap();
    assert!(!q.passed);
    assert!(
        q.reason.as_ref().unwrap().contains("position error"),
        "{q:?}"
    );
    let mut info2 = info.clone();
    q.record(&mut info2);
    assert_eq!(info2.backend, "cpu-f64");
    assert_eq!(info2.precision, Precision::F64Reference);
    assert!(
        info2
            .warnings
            .iter()
            .any(|w| w.contains("not used") && w.contains("position error"))
    );
    assert_eq!(info2.settings["backend_qualification"]["passed"], false);

    // A system the backend cannot run (thin-film radiometry): fails, says why.
    let mut sys = samples::singlet();
    sys.coatings.insert(
        "AR".into(),
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![],
            mirror_substrate: None,
        },
    );
    sys.surfaces[1].coating = Some("AR".into());
    sys.surfaces[1].interaction = Interaction::Refract;
    let q = qualify(&b, &testutil::tracer(sys), None, &ctx).unwrap();
    assert!(!q.passed);
    assert!(q.reason.unwrap().contains("cannot run"));
}

/// The ideal lens is double-float: direction and phase against the f64
/// reference (`mklight_trace::interact::ideal_lens`) over wide fields, fast
/// lenses and both sphere-term forms (near and far conjugates, both signs).
/// In f32 the phase was off by up to ~3e-6 mm.
#[test]
fn ideal_lens_is_double_float() {
    use mklight_gpu::df::Df32;
    use mklight_gpu::fenv::Ieee;
    use mklight_gpu::fx::D;
    use mklight_gpu::trace_kernel::ideal_lens;
    let d = |x: f64| D::<Ieee>::from_df(Df32::from_f64(x));
    let (mut worst_phase, mut worst_dir) = (0.0f64, 0.0f64);
    for &f in &[50.0, -80.0, 100.0, 1000.0] {
        for &verg in &[0.0, 1.0 / 150.0, -1.0 / 260.0, 1.0 / 3.0, -2.0] {
            for &(n1, n2) in &[(1.0, 1.0), (1.0, 1.5168), (1.62, 1.0)] {
                for k in 0..40 {
                    let a = k as f64 * 0.37;
                    let (x, y) = (25.0 * a.cos() * (k as f64 / 40.0), 18.0 * a.sin());
                    let th = (k as f64 * 1.3) % 0.6 - 0.3;
                    let dir = Vec3::new(th.sin() * a.sin(), th.sin() * a.cos(), th.cos())
                        .normalized()
                        * if k % 5 == 0 { -1.0 } else { 1.0 };
                    let (want_d, want_p) = mklight_trace::interact::ideal_lens(
                        Vec3::new(x, y, 0.0),
                        dir,
                        f,
                        n1,
                        n2,
                        verg,
                    )
                    .unwrap();
                    let (got_d, got_p) = ideal_lens(
                        d(x),
                        d(y),
                        [d(dir.x), d(dir.y), d(dir.z)],
                        d(1.0 / f),
                        d(n1),
                        d(n2),
                        d(verg),
                    )
                    .unwrap();
                    let gd = Vec3::new(
                        got_d[0].0.to_f64(),
                        got_d[1].0.to_f64(),
                        got_d[2].0.to_f64(),
                    );
                    worst_dir = worst_dir.max((gd - want_d).norm());
                    worst_phase = worst_phase.max((got_p.0.to_f64() - want_p).abs());
                }
            }
        }
    }
    assert!(worst_phase < 1e-11, "phase {worst_phase:e} mm");
    assert!(worst_dir < 1e-13, "direction {worst_dir:e}");
}

/// Fast ideal-lens relays at wide field meet the OPD budget (in f32 the
/// phase alone gave 2e-2 waves for this chain), and a single ideal lens is
/// exact in phase to far below it.
#[test]
fn ideal_relay_within_budget() {
    let b = CpuF32Backend::with_subnormals(false);
    let s = XvalSettings::default();
    let r = cross_validate(
        &b,
        &testutil::tracer(testutil::ideal_relay()),
        &s,
        &JobContext::new(),
    )
    .unwrap();
    r.check(&s.budget).unwrap();
    // What remains is the f32 rounding of the reported image point (≤ 1e-6
    // mm at 27 mm) seen through the ray angles (NA 0.25): ≲ 1.5e-3 waves.
    assert!(r.max_opd_waves < 2e-3, "{r:?}");
    let r = cross_validate(
        &b,
        &testutil::tracer(testutil::ideal_lenses()),
        &s,
        &JobContext::new(),
    )
    .unwrap();
    assert!(r.max_opd_waves < 1e-6, "{r:?}");
}

/// A ghost through a catadioptric system: the Newtonian's sensor ghost
/// travels back by reflections at the fold flat and the primary (mirrors
/// keep the direction through the sequence). Known answer: the on-axis
/// paraboloid images perfectly onto the sensor, so the ghost leaves the
/// primary antiparallel to the incoming beam at the mirrored pupil point.
/// Also against the independent f64 path; with radiometry the weight is
/// that path's (0 here: an uncoated air/air sensor plane reflects nothing,
/// the mirrors reflect everything).
#[test]
fn catadioptric_ghost_through_mirrors() {
    let b = CpuF32Backend::with_subnormals(false);
    let t = testutil::tracer(samples::newtonian_telescope());
    let sys = t.system();
    let h = b
        .upload_system(&PackedSystem::pack_all(sys).unwrap())
        .unwrap();
    let prog = testutil::newtonian_sensor_ghost();
    let mut rays = Vec::new();
    for p in (PupilSampling::Grid { n: 15 }).points() {
        rays.push((t.generate(t.field(0), 1, p[0], p[1]).unwrap().ray, 1u32));
    }
    let mut req = TraceRequest::path(h, batch(&rays), 0, prog.clone());
    req.record = RecordMode::AllSurfaces;
    let r = trace(&b, req);
    let n = rays.len();
    let opts = t.options();
    for (i, (ray, w)) in rays.iter().enumerate() {
        assert!(r.ray_status(i).is_ok(), "{:?}", r.ray_status(i));
        assert_eq!((r.last_surface[i], r.steps[i]), (1, 9));
        let (p, d) = v3(&r, i);
        // Known answer from the incoming hit on the primary.
        let (_, rec) = trace_ray_recorded(sys, ray, 1, &opts);
        let hit = rec[1].point;
        let mirrored = Vec3::new(-hit.x, -hit.y, hit.z);
        assert!(
            (p - mirrored).norm() < 2e-5,
            "ray {i}: {p:?} vs {mirrored:?}"
        );
        assert!(
            (d - Vec3::new(0.0, 0.0, -1.0)).norm() < 2e-6,
            "ray {i}: {d:?}"
        );
        // Against the independent f64 path computation.
        let want = f64_path(sys, 0, ray.pos, ray.dir, *w as usize, &prog).unwrap();
        assert!((want.pos - mirrored).norm() < 1e-8, "f64 path ray {i}");
        assert!((p - want.pos).norm() < 4e-6, "ray {i}");
        assert!((d - want.dir).norm() < 1e-7, "ray {i}");
        let want_opl = want.opl + (p - want.pos).dot(want.dir);
        assert!((r.opl_f64(i) - want_opl).abs() < 1e-8, "ray {i}");
        assert_eq!((r.weight[i] as f64, want.weight), (0.0, 0.0));
        // Slot 5: reflected at the sensor, travelling back (−z). Slot 7:
        // after the fold flat's reflection toward the primary, on the same
        // side of the flat as the forward beam after it (slot 3).
        let at = |slot: usize| (slot * n + i) * RECORD_FLOATS + 5;
        assert!(r.recorded[at(5)] < 0.0);
        assert!(r.recorded[at(7)] * r.recorded[at(3)] > 0.0);
    }
}

/// A Reflect step at a mirror-type ideal lens is its own (powered)
/// reflection: identical to the sequential step.
#[test]
fn ideal_mirror_reflect_step_is_sequential() {
    let b = CpuF32Backend::with_subnormals(false);
    let mut s = samples::newtonian_telescope();
    s.surfaces[1].shape = SurfaceShape::Paraxial {
        focal_length: -500.0,
    };
    let t = testutil::tracer(s);
    let sys = t.system();
    let h = b
        .upload_system(&PackedSystem::pack_all(sys).unwrap())
        .unwrap();
    let mut rays = Vec::new();
    for p in (PupilSampling::Grid { n: 9 }).points() {
        rays.push((t.generate(t.field(2), 1, p[0], p[1]).unwrap().ray, 1u32));
    }
    let seq = trace(&b, TraceRequest::new(h, batch(&rays), 0, 5));
    let mut prog = PathStep::sequential(0, 5);
    prog[0].mode = StepMode::Reflect;
    let path = trace(&b, TraceRequest::path(h, batch(&rays), 0, prog));
    assert_eq!(seq, path);
    // And it agrees with the reference's ideal mirror.
    for (i, (ray, _)) in rays.iter().enumerate() {
        let r = trace_ray(sys, ray, 1, &t.options(), None);
        assert!(r.status.is_ok());
        assert!((v3(&seq, i).0 - r.ray.pos).norm() < 2e-5);
    }
}

/// Strong aspheres where the conic seed is a poor start: with the default
/// step cap (the reference's), every ray the reference traces in up to 20
/// Newton steps has the reference's status and position. Longer searches
/// wander between several roots of a line that crosses the surface more
/// than once, and f32 and f64 may end on different ones: such rays stay
/// rare. With a cap below the reference's, a capped search reports
/// NOT_CONVERGED, never MISS.
#[test]
fn strong_aspheres_classify_like_the_reference() {
    let b = CpuF32Backend::with_subnormals(false);
    assert_eq!(
        NewtonSettings::default().max_iterations,
        IntersectSettings::default().max_iterations
    );
    let rays: Vec<_> = testutil::sweep_rays()
        .into_iter()
        .map(|r| (r, 1u32))
        .collect();
    for (k, shape) in testutil::strong_aspheres().into_iter().enumerate() {
        let rs = testutil::resolve(testutil::one_surface(shape));
        let h = b
            .upload_system(&PackedSystem::pack_all(&rs).unwrap())
            .unwrap();
        let opts = TraceOptions::unclipped().to_surface(1);
        let needed = |r: &Ray| {
            (0..=40).find(|&c| {
                let o = TraceOptions {
                    intersect: IntersectSettings {
                        tolerance: 1e-12,
                        max_iterations: c,
                    },
                    ..opts
                };
                trace_ray(&rs, r, 1, &o, None).status.is_ok()
            })
        };
        let mut req = TraceRequest::new(h, batch(&rays), 0, 1);
        req.apply_apertures = false;
        let g = trace(&b, req.clone());
        let (mut slow_ok, mut wandering) = (0, 0);
        for (i, (ray, _)) in rays.iter().enumerate() {
            let r = trace_ray(&rs, ray, 1, &opts, None);
            if !r.status.is_ok() {
                // Both fail (the failure class of a search that never
                // converges may differ); a hit appears only on a long walk.
                if g.ray_status(i).is_ok() {
                    wandering += 1;
                }
                continue;
            }
            let need = needed(ray).unwrap();
            let same = g.status[i] == MKL_RAY_OK && (v3(&g, i).0 - r.ray.pos).norm() < 2e-5;
            if need <= 20 {
                slow_ok += usize::from(need > 8);
                assert!(
                    same,
                    "shape {k} ray {i} ({need} steps): {:?}",
                    g.ray_status(i)
                );
            } else if !same {
                wandering += 1;
            }
        }
        assert!(wandering * 100 < rays.len(), "shape {k}: {wandering}");
        // The cap of 8 used before would have failed these.
        if k < 2 {
            assert!(slow_ok > 50, "shape {k}: {slow_ok}");
        }
        // Capped below the reference: never a miss for a reference hit.
        req.newton.max_iterations = 8;
        let g8 = trace(&b, req);
        let mut capped = 0;
        for (i, (ray, _)) in rays.iter().enumerate() {
            if trace_ray(&rs, ray, 1, &opts, None).status.is_ok() {
                let st = g8.ray_status(i);
                assert!(
                    matches!(st, RayStatus::Ok | RayStatus::NotConverged { .. }),
                    "shape {k} ray {i}: {st:?}"
                );
                capped += usize::from(!st.is_ok());
            }
        }
        if k < 2 {
            assert!(capped > 50, "shape {k}: {capped}");
        }
    }
}

/// Apertures are tested in double-float: at a 300 mm radius half an f32
/// ulp of the size (1.5e-5 mm) exceeds the 1e-5 mm edge band, yet rays
/// 2e-6 mm inside or outside every edge of every shape survive exactly as
/// in the reference, and rays on an edge pass (CONVENTIONS §2).
#[test]
fn large_apertures_clip_in_double_float() {
    use mklight_core::model::ApertureShape;
    let b = CpuF32Backend::with_subnormals(false);
    // Sizes that are not f32 numbers: their f32 rounding is ~1e-5 mm off.
    let (r0, r1) = (260.000_014_9, 300.000_014_9);
    let shapes = [
        ApertureShape::Circular { radius: r1 },
        ApertureShape::Annular {
            inner_radius: r0,
            outer_radius: r1,
        },
        ApertureShape::Rectangular {
            half_x: r0,
            half_y: r1,
        },
        ApertureShape::Elliptical {
            semi_x: r1,
            semi_y: r0,
        },
    ];
    for (k, ap) in shapes.into_iter().enumerate() {
        let mut s = testutil::one_surface(SurfaceShape::Plane);
        s.surfaces[1].aperture = Some(ap.clone());
        let rs = testutil::resolve(s);
        assert!((r1 as f32 as f64 - r1).abs() > 1e-5 && (r0 as f32 as f64 - r0).abs() > 1e-5);
        let h = b
            .upload_system(&PackedSystem::pack_all(&rs).unwrap())
            .unwrap();
        let mut rays = Vec::new();
        let mut on_edge = Vec::new();
        for j in 0..32 {
            let a = j as f64 * std::f64::consts::TAU / 32.0;
            let (c, sn) = (a.cos(), a.sin());
            // Points just inside / on / outside the edges along this direction.
            let edges: Vec<f64> = match ap.clone() {
                ApertureShape::Circular { radius } => vec![radius],
                ApertureShape::Annular {
                    inner_radius,
                    outer_radius,
                } => vec![inner_radius, outer_radius],
                ApertureShape::Rectangular { half_x, half_y } => {
                    vec![(half_x / c.abs()).min(half_y / sn.abs())]
                }
                ApertureShape::Elliptical { semi_x, semi_y } => {
                    vec![1.0 / ((c / semi_x).powi(2) + (sn / semi_y).powi(2)).sqrt()]
                }
            };
            for e in edges {
                for dr in [-2e-6, 0.0, 2e-6] {
                    let p = Vec3::new((e + dr) * c, (e + dr) * sn, -1.0);
                    rays.push((Ray::new(p, Vec3::Z), 1u32));
                    on_edge.push(dr == 0.0);
                }
            }
        }
        let g = trace(&b, TraceRequest::new(h, batch(&rays), 0, 2));
        let mut clipped = 0;
        for (i, (ray, _)) in rays.iter().enumerate() {
            let want = reference_status_word(
                &trace_ray(&rs, ray, 1, &TraceOptions::default(), None).status,
            );
            assert_eq!(g.status[i], want, "shape {k} ray {i}");
            assert!(!on_edge[i] || want == MKL_RAY_OK, "shape {k} ray {i}");
            clipped += usize::from(want != MKL_RAY_OK);
        }
        assert!(
            clipped * 4 > rays.len() && clipped < rays.len(),
            "shape {k}"
        );
    }
}

/// Qualification needs a margin: a small sample can miss the worst rays of
/// a 64 × 64 grid. With an OPD budget the full grid exceeds but the sample
/// meets, checking the sample against the bare budget would pass (silently
/// degraded results); against the default third it is rejected, and the
/// qualification reports how much of the budget it used. The sample's
/// pupil fans find narrow zones a coarse pattern steps over: near-critical
/// refraction in the TIR block, where the hexapolar pattern alone sees 1/50
/// of the worst error.
#[test]
fn qualification_margin_rejects_marginal_systems() {
    let b = CpuF32Backend::with_subnormals(false);
    let ctx = JobContext::new();
    let worst = |t: &mklight_trace::SequentialTracer, s: &XvalSettings| {
        cross_validate(&b, t, s, &ctx).unwrap().max_opd_waves
    };
    let mut lax = XvalSettings::qualification();
    lax.margin = 1.0;
    let tir = testutil::tracer(testutil::tir_block());
    let full = worst(&tir, &XvalSettings::default());
    assert!(
        full < 2.0 * worst(&tir, &lax),
        "fans reach the near-critical zone"
    );
    let mut hex_only = lax.clone();
    hex_only.pupil_extra.clear();
    assert!(full > 10.0 * worst(&tir, &hex_only));

    let t = testutil::tracer(testutil::absorbing_triplet());
    let full = cross_validate(&b, &t, &XvalSettings::default(), &ctx).unwrap();
    let (e_full, e_q) = (full.max_opd_waves, worst(&t, &lax));
    assert!(e_full > 1.2 * e_q, "full {e_full:e} sample {e_q:e}");
    // A budget between the two.
    let opd_budget = (e_q * e_full).sqrt();
    let mut s = XvalSettings::default();
    s.budget.opd_waves = opd_budget;
    assert!(full.check(&s.budget).is_err());
    lax.budget.opd_waves = opd_budget;
    let q = qualify(&b, &t, Some(&lax), &ctx).unwrap();
    assert!(q.passed, "without a margin the sample passes: {q:?}");
    let mut strict = XvalSettings::qualification();
    strict.budget.opd_waves = opd_budget;
    let q = qualify(&b, &t, Some(&strict), &ctx).unwrap();
    assert!(!q.passed);
    assert!((q.margin - 1.0 / 3.0).abs() < 1e-15);
    let frac = q.budget_fraction.unwrap();
    assert!(frac > 1.0 / 3.0 && frac < 1.0, "{frac}");
    assert!(q.reason.unwrap().contains("1/3 of the budget"));
    // At the real budget both qualify, the fast ideal relay included.
    assert!(qualify(&b, &t, None, &ctx).unwrap().passed);
    let relay = testutil::tracer(testutil::ideal_relay());
    let q = qualify(&b, &relay, None, &ctx).unwrap();
    assert!(q.passed && q.budget_fraction.unwrap() < 1.0 / 3.0, "{q:?}");
}

/// Cancellation is not a verdict: `qualify` returns `Cancelled` instead of
/// a failed qualification (which would trigger a CPU fallback).
#[test]
fn qualification_propagates_cancellation() {
    let b = CpuF32Backend::with_subnormals(false);
    let t = testutil::tracer(samples::double_gauss());
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(qualify(&b, &t, None, &ctx), Err(MkError::Cancelled));
    assert_eq!(
        cross_validate(&b, &t, &XvalSettings::qualification(), &ctx),
        Err(MkError::Cancelled)
    );
}

/// Systems with more than 16 wavelengths are validated (and qualified) in
/// packed chunks, like the packer splits them.
#[test]
fn many_wavelengths_are_validated_in_chunks() {
    use mklight_core::model::Wavelength;
    let b = CpuF32Backend::with_subnormals(false);
    let mut sys = samples::cooke_triplet();
    sys.wavelengths.wavelengths = (0..21)
        .map(|k| Wavelength {
            um: 0.45 + 0.01 * k as f64,
            weight: 1.0,
        })
        .collect();
    sys.wavelengths.primary = 10;
    let t = testutil::tracer(sys);
    let ctx = JobContext::new();
    let s = XvalSettings::qualification();
    let r = cross_validate(&b, &t, &s, &ctx).unwrap();
    let per = t.fields().len()
        * (s.pupil.points().len()
            + s.pupil_extra
                .iter()
                .map(|p| p.points().len())
                .sum::<usize>());
    assert_eq!(r.rays, 21 * per);
    assert_eq!(r.compared, r.rays);
    assert_eq!(ctx.progress.fraction(), 1.0);
    let q = qualify(&b, &t, None, &ctx).unwrap();
    assert!(q.passed, "{q:?}");
}

#[test]
fn extended_qualification_cpu_f32() {
    let b = CpuF32Backend::with_subnormals(false);
    for (name, sys) in samples::all()
        .into_iter()
        .filter(|(_, s)| samples::has_extended_surfaces(s))
    {
        let q = qualify(&b, &testutil::tracer(sys), None, &JobContext::new()).unwrap();
        assert!(q.passed, "{name}: {q:?}");
    }
}

/// Steep random rays (±80°, no clipping) on every full-capacity fixture:
/// statuses agree with the reference for more than 99 % of the rays, and
/// every hit both find meets the position, direction and same-point OPL
/// budget. The residual status differences are rays whose reference
/// classification is itself ill-conditioned (docs/GPU.md §5.1.1).
#[test]
fn extended_steep_rays_classify_and_match_f64() {
    let b = CpuF32Backend::with_subnormals(false);
    let rays: Vec<_> = testutil::steep_random_rays()
        .into_iter()
        .map(|r| (r, 1))
        .collect();
    for (name, shape) in testutil::extended_shapes() {
        let rs = testutil::resolve(testutil::one_surface(shape));
        let h = b
            .upload_system(&PackedSystem::pack_all(&rs).unwrap())
            .unwrap();
        let mut req = TraceRequest::new(h, batch(&rays), 0, 1);
        req.apply_apertures = false;
        let got = trace(&b, req);
        let mut pairs = std::collections::BTreeMap::new();
        let (mut matched, mut common) = (0, 0);
        let (mut max_p, mut max_d, mut max_opl) = (0f64, 0f64, 0f64);
        for (i, (ray, _)) in rays.iter().enumerate() {
            let r = trace_ray(&rs, ray, 1, &TraceOptions::unclipped().to_surface(1), None);
            let want = reference_status_word(&r.status);
            if got.status[i] == want {
                matched += 1;
            } else {
                *pairs.entry((want & 255, got.status[i] & 255)).or_insert(0) += 1;
            }
            if want == MKL_RAY_OK && got.status[i] == MKL_RAY_OK {
                common += 1;
                let (p, d) = v3(&got, i);
                // OPLs are compared at the same reported point.
                let eo = (got.opl_f64(i)
                    - r.opl.value()
                    - rs.surfaces[1].n_after[1] * (p - r.ray.pos).dot(r.ray.dir))
                .abs();
                max_p = max_p.max((p - r.ray.pos).norm());
                max_d = max_d.max((d - r.ray.dir).norm());
                max_opl = max_opl.max(eo);
            }
        }
        eprintln!(
            "{name}: matched {matched}/{} common {common} max position {max_p:.2e} direction \
             {max_d:.2e} OPL {max_opl:.2e}; status (f64, f32) {pairs:?}",
            rays.len()
        );
        assert!(common > 1000, "{name}");
        assert!(
            (rays.len() - matched) * 100 < rays.len(),
            "{name}: {pairs:?}"
        );
        assert!(max_p < 2e-5 && max_d < 2e-6, "{name}");
        assert!(max_opl < rs.wavelengths_um[1] * 1e-3 * 5e-3, "{name}");
        b.release_systems(h).unwrap();
    }
}

#[test]
fn doe_orders_opl_and_evanescence() {
    use mklight_core::model::surface::DiffractivePhase;
    let b = CpuF32Backend::with_subnormals(false);
    for interaction in [
        Interaction::Refract,
        Interaction::Reflect,
        Interaction::Dummy,
    ] {
        for order in [-2, 0, 1, 3, 10000] {
            let shape = SurfaceShape::Diffractive {
                base: Box::new(SurfaceShape::Conic {
                    curvature: 0.04,
                    conic: -0.5,
                }),
                phase: DiffractivePhase {
                    norm_radius: 10.0,
                    coefficients: vec![-400.0, 20.0, -2.0],
                    design_order: 2,
                    design_wavelength_um: 0.55,
                    trace_order: order,
                },
            };
            let mut sys = testutil::one_surface(shape);
            sys.surfaces[1].interaction = interaction;
            let rs = testutil::resolve(sys);
            let h = b
                .upload_system(&PackedSystem::pack_all(&rs).unwrap())
                .unwrap();
            let rays: Vec<_> = (0..3)
                .flat_map(|w| {
                    (0..12).map(move |j| {
                        (
                            Ray::new(
                                Vec3::new(0.3 + j as f64 * 0.5, 0.7, -5.0),
                                Vec3::new(0.06, -0.04, 1.0).normalized(),
                            ),
                            w,
                        )
                    })
                })
                .collect();
            let mut req = TraceRequest::new(h, batch(&rays), 0, 1);
            req.record = RecordMode::AllSurfaces;
            req.apply_apertures = false;
            let got = trace(&b, req);
            let mut evanescent = 0;
            for (i, (ray, w)) in rays.iter().enumerate() {
                let r = trace_ray(
                    &rs,
                    ray,
                    *w as usize,
                    &TraceOptions::unclipped().to_surface(1),
                    None,
                );
                assert_eq!(
                    got.status[i],
                    reference_status_word(&r.status),
                    "{interaction:?} {order} ray {i}"
                );
                evanescent += usize::from(matches!(
                    got.ray_status(i),
                    RayStatus::EvanescentOrder { .. }
                ));
                if !r.status.is_ok() {
                    continue;
                }
                let (p, d) = v3(&got, i);
                assert!((p - r.ray.pos).norm() < 2e-5);
                assert!((d - r.ray.dir).norm() < 2e-6);
                let n = if interaction == Interaction::Refract {
                    rs.surfaces[1].n_after[*w as usize]
                } else {
                    rs.surfaces[0].n_after[*w as usize]
                };
                let expected = r.opl.value() + n * (p - r.ray.pos).dot(r.ray.dir);
                assert!(
                    (got.opl_hi[i] as f64 + got.opl_lo[i] as f64 - expected).abs() < 1e-8,
                    "DOE OPL {interaction:?} {order}"
                );
            }
            if order == 10000 && interaction != Interaction::Dummy {
                assert_eq!(evanescent, rays.len());
            }
            b.release_systems(h).unwrap();
        }
    }
}

#[test]
fn passive_doe_has_no_phase_on_forced_reflection() {
    use mklight_core::model::surface::DiffractivePhase;
    let b = CpuF32Backend::with_subnormals(false);
    for at_image in [false, true] {
        let shape = SurfaceShape::Diffractive {
            base: Box::new(SurfaceShape::Plane),
            phase: DiffractivePhase {
                norm_radius: 10.0,
                coefficients: vec![1e8],
                design_order: 1,
                trace_order: 1,
                design_wavelength_um: 0.55,
            },
        };
        let mut sys = testutil::one_surface(SurfaceShape::Plane);
        let si = if at_image { 2 } else { 1 };
        sys.surfaces[si].shape = shape;
        sys.surfaces[si].interaction = Interaction::Dummy;
        let rs = testutil::resolve(sys);
        let h = b
            .upload_system(&PackedSystem::pack_all(&rs).unwrap())
            .unwrap();
        let rays = batch(&[(Ray::new(Vec3::new(1.0, 0.5, -5.0), Vec3::Z), 1)]);
        let steps = if at_image {
            vec![
                PathStep::forward(1, StepMode::Transmit),
                PathStep::forward(2, StepMode::Reflect),
            ]
        } else {
            vec![PathStep::forward(1, StepMode::Reflect)]
        };
        let got = trace(&b, TraceRequest::path(h, rays, 0, steps));
        assert_eq!(got.status[0], MKL_RAY_OK);
        let (p, d) = v3(&got, 0);
        assert_eq!(p, Vec3::new(1.0, 0.5, 0.0));
        assert_eq!(d, -Vec3::Z);
        let expected = 5.0
            + if at_image {
                rs.surfaces[1].thickness * rs.surfaces[1].n_after[1]
            } else {
                0.0
            };
        assert!((got.opl_hi[0] as f64 + got.opl_lo[0] as f64 - expected).abs() < 1e-10);
        b.release_systems(h).unwrap();
    }
}

#[test]
fn odd_cone_normal_survives_small_nonzero_radius() {
    let b = CpuF32Backend::with_subnormals(false);
    let rs = testutil::resolve(testutil::one_surface(SurfaceShape::OddAsphere {
        curvature: 0.0,
        conic: 0.0,
        coefficients: vec![0.01],
    }));
    let h = b
        .upload_system(&PackedSystem::pack_all(&rs).unwrap())
        .unwrap();
    let rays: Vec<_> = [-1e-30, 0.0, 1e-30, 1e-10]
        .into_iter()
        .map(|x| (Ray::new(Vec3::new(x, 0.0, -5.0), Vec3::Z), 1))
        .collect();
    let got = trace(&b, TraceRequest::new(h, batch(&rays), 0, 1));
    for (i, (ray, _)) in rays.iter().enumerate() {
        let r = trace_ray(&rs, ray, 1, &TraceOptions::unclipped().to_surface(1), None);
        assert_eq!(got.status[i], reference_status_word(&r.status));
        assert!(
            (v3(&got, i).1 - r.ray.dir).norm() < 2e-6,
            "radius {}: {:?} vs {:?}",
            ray.pos.x,
            v3(&got, i).1,
            r.ray.dir
        );
    }
}
