//! Standalone qualification/benchmark driver. Built by the temporary manifest
//! documented in P2-evidence.md, so P2 adds no solver dependency to the backend.
//! The unchanged wave FFT source is compiled as the CPU reference.
#[allow(dead_code)]
#[path = "../../../mklight-wave/src/fft.rs"]
mod reference;
use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{resident::*, transforms::*, *};
use mklight_metal::{LibraryChoice, MetalBackend, MetalOptions};
use mklight_wave::{PupilFunction, PupilSettings, trace_pupil};
use num_complex::Complex64 as C;
use serde_json::json;
use std::time::Instant;
fn read(b: &dyn ComputeBackend, r: &PlanResult) -> Vec<[f32; 2]> {
    let BufferData::Complex32(v) = b
        .readback(r.outputs[0].view(), &JobContext::new())
        .unwrap()
        .data
    else {
        panic!()
    };
    v
}
fn upload(b: &dyn ComputeBackend, field: &[C], n: usize) -> BufferHandle {
    b.upload_buffer(
        BufferDesc::new(ElementType::Complex32, vec![n, n]),
        BufferData::Complex32(field.iter().map(|z| [z.re as f32, z.im as f32]).collect()),
    )
    .unwrap()
}
fn plan(
    s: Snapshot,
    h: &BufferHandle,
    n: usize,
    m: usize,
    affine: Option<[Df32; 6]>,
    chain: bool,
) -> ResidentPlan {
    let mut p = ResidentPlan::new(s);
    p.settings = json!({"p2_benchmark":"v1","seed":0,"n":n,"m":m});
    let i = p.import(h.view());
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![m, m]));
    p.operations
        .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
            kind: if affine.is_some() {
                TransformKind::AffineMft
            } else {
                TransformKind::Fft
            },
            input: p.view(i).unwrap(),
            output: o,
            dimensions: vec![n, n],
            output_dimensions: vec![m, m],
            sign: -1,
            scale: 1.,
            affine: affine.unwrap_or([Df32::ZERO; 6]),
        })));
    if chain {
        let back = p.scratch(BufferDesc::new(ElementType::Complex32, vec![n, n]));
        p.operations
            .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
                kind: TransformKind::Fft,
                input: p.view(o).unwrap(),
                output: back,
                dimensions: vec![n, n],
                output_dimensions: vec![n, n],
                sign: 1,
                scale: 1. / (n * n) as f32,
                affine: [Df32::ZERO; 6],
            })));
        p.outputs = vec![back];
    } else {
        p.outputs = vec![o];
    }
    p
}
fn error(a: &[[f32; 2]], b: &[C], l1: f64) -> f64 {
    let mut e = 0.;
    let mut norm = 0.;
    let mut max = 0f64;
    for (a, b) in a.iter().zip(b) {
        let z = C::new(a[0] as f64, a[1] as f64) - b;
        e += z.norm_sqr();
        norm += b.norm_sqr();
        max = max.max(z.norm());
    }
    let rel = (e / norm.max(1e-30)).sqrt();
    assert!(
        rel <= 1e-4 && max <= 1e-4 * l1,
        "T gate rel={rel} max={max} L1={l1}"
    );
    rel
}
fn optical(b: &dyn ComputeBackend, s: Snapshot, pupil: &PupilFunction) -> f64 {
    let n = pupil.n();
    let freqs = [0., 5., 17., 53.];
    let ctx = JobContext::new();
    let want = mklight_wave::mtf_direct(pupil, &freqs, &Default::default(), &ctx).unwrap();
    let mut p = ResidentPlan::new(s);
    let mut imp = |ty, shape, data| {
        let h = b.upload_buffer(BufferDesc::new(ty, shape), data).unwrap();
        let i = p.import(h.view());
        p.view(i).unwrap()
    };
    let amplitude = imp(
        ElementType::F32,
        vec![n, n],
        BufferData::F32(pupil.amplitude().iter().map(|v| *v as f32).collect()),
    );
    let coverage = imp(
        ElementType::F32,
        vec![n, n],
        BufferData::F32(pupil.coverage().iter().map(|v| *v as f32).collect()),
    );
    let opd = imp(
        ElementType::DoubleFloat,
        vec![n, n],
        BufferData::DoubleFloat(
            pupil
                .opd_waves()
                .iter()
                .map(|v| Df32::from_f64(if v.is_finite() { *v } else { 0. }))
                .collect(),
        ),
    );
    let frequencies = imp(
        ElementType::F32,
        vec![8, 2],
        BufferData::F32(
            freqs
                .iter()
                .flat_map(|v| [0f32, *v as f32])
                .chain(freqs.iter().flat_map(|v| [*v as f32, 0f32]))
                .collect(),
        ),
    );
    let output = p.scratch(BufferDesc::new(ElementType::Complex32, vec![8]));
    let frame = pupil
        .geometry()
        .chief_frame()
        .map(|v| [v.x, v.y, v.z].map(Df32::from_f64));
    let req = OpticalCorrelation {
        stencil: None,
        amplitude,
        coverage,
        opd,
        frequencies,
        frame,
        sin_max: Df32::from_f64(pupil.geometry().sin_max()),
        wavelength_over_index: Df32::from_f64(
            pupil.wavelength_um() * 1e-3 / pupil.geometry().image_index,
        ),
        output,
    };
    p.operations
        .push(ResidentOp::Transforms(Request::OpticalCorrelation(req)));
    p.outputs = vec![output];
    let r = b.submit_plan(p, &ctx).unwrap().wait().unwrap();
    let got = read(b, &r);
    let mut max = 0f64;
    for (a, b) in got.iter().zip(want.tangential.iter().chain(&want.sagittal)) {
        max = max.max(((a[0] as f64).hypot(a[1] as f64) - b).abs());
    }
    assert!(max <= 2e-3, "optical MTF max={max}");
    max
}
fn bench(
    b: &dyn ComputeBackend,
    s: Snapshot,
    pupil: &PupilFunction,
    name: &str,
    m: usize,
    skew: bool,
) {
    let n = pupil.n();
    let original = pupil.debye_field();
    let field: Vec<C> = original
        .iter()
        .map(|z| C::new(z.re as f32 as f64, z.im as f32 as f64))
        .collect();
    let l1 = field.iter().map(|z| z.norm()).sum();
    std::fs::create_dir_all("/tmp/mklight-p2-pupils").unwrap();
    let stored = format!("/tmp/mklight-p2-pupils/{name}-{n}-{m}-{skew}.complex32");
    let bytes: Vec<u8> = field
        .iter()
        .flat_map(|z| [z.re as f32, z.im as f32])
        .flat_map(f32::to_le_bytes)
        .collect();
    std::fs::write(&stored, bytes).unwrap();
    println!(
        "{}",
        json!({"kind":"input","case":name,"n":n,"m":m,"stored":stored,"snapshot":s.hash().to_hex(),"seed":0,"layout":"little-endian complex32 row-major"})
    );
    let affine = if m != n || skew {
        Some(
            [
                0.37 / n as f64,
                if skew { -0.07 / n as f64 } else { 0. },
                -0.13,
                if skew { -0.07 / n as f64 } else { 0. },
                0.29 / n as f64,
                0.17,
            ]
            .map(Df32::from_f64),
        )
    } else {
        None
    };
    let cpu = || {
        if let Some(a) = affine {
            let a = a.map(Df32::to_f64);
            let nu: Vec<_> = (0..n).map(|i| i as f64).collect();
            let xs: Vec<_> = (0..m).map(|k| k as f64).collect();
            let mut src = field.clone();
            for j in 0..n {
                for i in 0..n {
                    src[j * n + i] *= C::from_polar(
                        1.,
                        -std::f64::consts::TAU * (a[2] * i as f64 + a[5] * j as f64),
                    );
                }
            }
            if skew {
                reference::affine_mft(
                    &src,
                    n,
                    &nu,
                    &xs,
                    &xs,
                    [[-a[0], -a[1]], [-a[3], -a[4]]],
                    &JobContext::new(),
                )
                .unwrap()
            } else {
                let xs: Vec<_> = (0..m).map(|k| -a[0] * k as f64).collect();
                let ys: Vec<_> = (0..m).map(|k| -a[4] * k as f64).collect();
                reference::mft(&src, n, &nu, &nu, &xs, &ys, &JobContext::new()).unwrap()
            }
        } else {
            let mut v = field.clone();
            reference::fft2(&mut v, n, reference::Dir::Forward, n, &JobContext::new()).unwrap();
            v
        }
    };
    let h = upload(b, &field, n);
    let p = plan(s.clone(), &h, n, m, affine, false);
    let cpu_start = Instant::now();
    let want = cpu();
    let cpu_cold = cpu_start.elapsed().as_secs_f64();
    let start = Instant::now();
    let r = b
        .submit_plan(p.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let got = read(b, &r);
    let cold = start.elapsed().as_secs_f64();
    let rel = error(&got, &want, l1);
    println!(
        "{}",
        json!({"case":name,"n":n,"m":m,"skew":skew,"kind":"cold","cpu_s":cpu_cold,"metal_s":cold,"relative_l2":rel,"stats":r.stats,"run_info":r.run_info})
    );
    drop(r);
    // One warm-up is separate from ten measured runs; readback is included.
    let r = b
        .submit_plan(p.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    drop(read(b, &r));
    drop(r);
    drop(cpu());
    for run in 0..10 {
        let t = Instant::now();
        let baseline = cpu();
        let cpu_s = t.elapsed().as_secs_f64();
        std::hint::black_box(baseline);
        let t = Instant::now();
        let r = b
            .submit_plan(p.clone(), &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let got = read(b, &r);
        let wall = t.elapsed().as_secs_f64();
        let rel = error(&got, &want, l1);
        let stats = r.stats.clone();
        drop(r);
        let t = Instant::now();
        let h = upload(b, &field, n);
        let fed = plan(s.clone(), &h, n, m, affine, false);
        let r = b
            .submit_plan(fed, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        drop(read(b, &r));
        let host_fed = t.elapsed().as_secs_f64();
        println!(
            "{}",
            json!({"case":name,"kind":"warm","n":n,"m":m,"skew":skew,"run":run,"cpu_s":cpu_s,"resident_s":wall,"host_fed_s":host_fed,"relative_l2":rel,"stats":stats})
        );
    }
    if affine.is_none() {
        let p = plan(s, &h, n, n, None, true);
        let t = Instant::now();
        let r = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let got = read(b, &r);
        println!(
            "{}",
            json!({"case":name,"kind":"resident_roundtrip","n":n,"wall_s":t.elapsed().as_secs_f64(),"relative_l2":error(&got,&field,l1),"stats":r.stats,"run_info":r.run_info})
        );
    }
}
fn main() {
    let start = Instant::now();
    let b = MetalBackend::with_options(&MetalOptions {
        library: if std::env::var("P2_LIBRARY").as_deref() == Ok("runtime") {
            LibraryChoice::RuntimeSource
        } else {
            LibraryChoice::Precompiled
        },
        ..Default::default()
    })
    .unwrap()
    .expect("host Metal required");
    assert!(b.precision().probe.error_free_transforms());
    println!(
        "{}",
        json!({"device":b.info().backend_id(),"build":b.precision().kernel_build,"probe":format!("{:?}",b.precision().probe),"startup_s":start.elapsed().as_secs_f64()})
    );
    let mode = std::env::args().nth(1).unwrap_or("samples".into());
    for (name, sys) in mklight_core::samples::all() {
        if mode == "skew" && name != "xy_freeform" {
            continue;
        }
        if mode == "dense"
            && ![
                "singlet",
                "cooke_triplet",
                "double_gauss",
                "newtonian",
                "relay",
                "hybrid_doe",
                "anamorphic",
                "xy_freeform",
            ]
            .contains(&name)
        {
            continue;
        }
        let snapshot = Snapshot::new(sys.clone()).unwrap();
        let tracer = mklight_analysis::tracer_for(sys).unwrap();
        if mode == "samples" {
            for f in 0..tracer.fields().len() {
                for w in 0..tracer.num_wavelengths() {
                    let t = Instant::now();
                    let tp = trace_pupil(
                        &tracer,
                        tracer.field(f),
                        w,
                        &PupilSettings::default(),
                        &JobContext::new(),
                    )
                    .unwrap();
                    let pupil = tp.pupil;
                    let n = pupil.n();
                    let prep = t.elapsed().as_secs_f64();
                    let src: Vec<C> = pupil
                        .debye_field()
                        .iter()
                        .map(|z| C::new(z.re as f32 as f64, z.im as f32 as f64))
                        .collect();
                    let mut want = src.clone();
                    reference::fft2(&mut want, n, reference::Dir::Forward, n, &JobContext::new())
                        .unwrap();
                    let h = upload(&b, &src, n);
                    let p = plan(snapshot.clone(), &h, n, n, None, false);
                    let r = b
                        .submit_plan(p, &JobContext::new())
                        .unwrap()
                        .wait()
                        .unwrap();
                    let got = read(&b, &r);
                    let rel = error(&got, &want, src.iter().map(|z| z.norm()).sum());
                    let mtf = optical(&b, snapshot.clone(), &pupil);
                    println!(
                        "{}",
                        json!({"kind":"sample","name":name,"field":f,"wavelength":w,"n":n,"prep_s":prep,"fft_relative_l2":rel,"mtf_max_abs":mtf,"warnings":tp.warnings,"stats":r.stats,"run_info":r.run_info})
                    );
                }
            }
            let tp = trace_pupil(
                &tracer,
                tracer.field(0),
                0,
                &PupilSettings::default(),
                &JobContext::new(),
            )
            .unwrap();
            bench(
                &b,
                snapshot,
                &tp.pupil,
                &format!("{name}:f0:w0"),
                tp.pupil.n(),
                false,
            );
        } else if mode == "skew" {
            let tp = trace_pupil(
                &tracer,
                tracer.field(tracer.fields().len() - 1),
                0,
                &PupilSettings {
                    grid: 1024,
                    ..Default::default()
                },
                &JobContext::new(),
            )
            .unwrap();
            bench(&b, snapshot, &tp.pupil, name, 2048, true);
        } else {
            for n in [64, 256, 1024] {
                let tp = trace_pupil(
                    &tracer,
                    tracer.field(tracer.fields().len() - 1),
                    0,
                    &PupilSettings {
                        grid: n,
                        ..Default::default()
                    },
                    &JobContext::new(),
                )
                .unwrap();
                bench(
                    &b,
                    snapshot.clone(),
                    &tp.pupil,
                    &format!("{name}:f{}:w0", tracer.fields().len() - 1),
                    n,
                    false,
                );
                if n == 1024 {
                    bench(&b, snapshot.clone(), &tp.pupil, name, 2048, false);
                    if name == "xy_freeform" {
                        bench(&b, snapshot.clone(), &tp.pupil, name, 2048, true);
                    }
                }
            }
        }
    }
}
