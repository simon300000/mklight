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
                assert_eq!(p.surfaces[i].vergence[w], v as f32);
                assert!(v != 0.0);
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
