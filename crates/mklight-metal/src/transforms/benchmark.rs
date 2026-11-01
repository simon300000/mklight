//! Standalone qualification/benchmark driver, built by the temporary manifest
//! described in P2-evidence.md so the backend gains no solver dependency.
//! The unchanged wave FFT source is compiled as the f64 CPU reference.
//!
//! Modes: `samples` (every default sample pupil: FFT and optical MTF vs f64),
//! `dense` (outer-field pupils at N=64/256/1024 and 1024→2048 MFTs vs f64),
//! `sizes` (warm timings: FFT 64²–2048², MFT, optical correlation).
//! `P2_LIBRARY=runtime` selects the runtime-compiled library; `P2_RUNS` the
//! number of measured runs (default 15).
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
fn import(
    b: &dyn ComputeBackend,
    p: &mut ResidentPlan,
    ty: ElementType,
    shape: Vec<usize>,
    data: BufferData,
) -> PlanView {
    let h = b.upload_buffer(BufferDesc::new(ty, shape), data).unwrap();
    let i = p.import(h.view());
    p.view(i).unwrap()
}
fn complex(field: &[C]) -> BufferData {
    BufferData::Complex32(field.iter().map(|z| [z.re as f32, z.im as f32]).collect())
}
/// Forward n×n → m×m transform: FFT, separable ChirpZ or symmetric skew MFT.
fn transform_plan(
    b: &dyn ComputeBackend,
    s: Snapshot,
    field: &[C],
    n: usize,
    m: usize,
    affine: Option<[f64; 6]>,
) -> ResidentPlan {
    let mut p = ResidentPlan::new(s);
    let input = import(
        b,
        &mut p,
        ElementType::Complex32,
        vec![n, n],
        complex(field),
    );
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![m, m]));
    let kind = match affine {
        None => TransformKind::Fft,
        Some(a) if a[1] == 0.0 => TransformKind::ChirpZ,
        Some(_) => TransformKind::AffineMft,
    };
    p.operations
        .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
            kind,
            input,
            output: o,
            dimensions: vec![n, n],
            output_dimensions: vec![m, m],
            sign: -1,
            scale: 1.,
            affine: affine.unwrap_or([0.; 6]).map(Df32::from_f64),
        })));
    p.outputs = vec![o];
    p
}
fn mft_affine(n: usize, skew: bool) -> [f64; 6] {
    let s = if skew { -0.07 / n as f64 } else { 0. };
    [0.37 / n as f64, s, -0.13, s, 0.29 / n as f64, 0.17]
}
/// f64 reference through the unchanged wave routines.
fn reference(field: &[C], n: usize, m: usize, affine: Option<[f64; 6]>) -> Vec<C> {
    let ctx = JobContext::new();
    let Some(a) = affine else {
        let mut v = field.to_vec();
        reference::fft2(&mut v, n, reference::Dir::Forward, n, &ctx).unwrap();
        return v;
    };
    let nu: Vec<_> = (0..n).map(|i| i as f64).collect();
    let mut src = field.to_vec();
    for j in 0..n {
        for i in 0..n {
            src[j * n + i] *= C::from_polar(
                1.,
                -std::f64::consts::TAU * (a[2] * i as f64 + a[5] * j as f64),
            );
        }
    }
    if a[1] != 0.0 {
        let xs: Vec<_> = (0..m).map(|k| k as f64).collect();
        reference::affine_mft(
            &src,
            n,
            &nu,
            &xs,
            &xs,
            [[-a[0], -a[1]], [-a[3], -a[4]]],
            &ctx,
        )
        .unwrap()
    } else {
        let xs: Vec<_> = (0..m).map(|k| -a[0] * k as f64).collect();
        let ys: Vec<_> = (0..m).map(|k| -a[4] * k as f64).collect();
        reference::mft(&src, n, &nu, &nu, &xs, &ys, &ctx).unwrap()
    }
}
/// T gate: relative L2 and scaled maximum; returns relative L2.
fn error(a: &[[f32; 2]], b: &[C], l1: f64) -> f64 {
    let (mut e, mut norm, mut max) = (0., 0., 0f64);
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
fn optical_plan(
    b: &dyn ComputeBackend,
    s: Snapshot,
    pupil: &PupilFunction,
    freqs: &[f64],
) -> ResidentPlan {
    let n = pupil.n();
    let mut p = ResidentPlan::new(s);
    let f32s = |v: &[f64]| BufferData::F32(v.iter().map(|v| *v as f32).collect());
    let amplitude = import(
        b,
        &mut p,
        ElementType::F32,
        vec![n, n],
        f32s(pupil.amplitude()),
    );
    let coverage = import(
        b,
        &mut p,
        ElementType::F32,
        vec![n, n],
        f32s(pupil.coverage()),
    );
    let opd = BufferData::DoubleFloat(
        pupil
            .opd_waves()
            .iter()
            .map(|v| Df32::from_f64(if v.is_finite() { *v } else { 0. }))
            .collect(),
    );
    let opd = import(b, &mut p, ElementType::DoubleFloat, vec![n, n], opd);
    let pairs: Vec<f32> = freqs
        .iter()
        .flat_map(|v| [0f32, *v as f32])
        .chain(freqs.iter().flat_map(|v| [*v as f32, 0f32]))
        .collect();
    let frequencies = import(
        b,
        &mut p,
        ElementType::F32,
        vec![2 * freqs.len(), 2],
        BufferData::F32(pairs),
    );
    let output = p.scratch(BufferDesc::new(
        ElementType::Complex32,
        vec![2 * freqs.len()],
    ));
    let g = pupil.geometry();
    p.operations
        .push(ResidentOp::Transforms(Request::OpticalCorrelation(
            OpticalCorrelation {
                stencil: None,
                amplitude,
                coverage,
                opd,
                frequencies,
                frame: g.chief_frame().map(|v| [v.x, v.y, v.z].map(Df32::from_f64)),
                sin_max: Df32::from_f64(g.sin_max()),
                wavelength_over_index: Df32::from_f64(pupil.wavelength_um() * 1e-3 / g.image_index),
                output,
            },
        )));
    p.outputs = vec![output];
    p
}
/// Optical MTF (tangential then sagittal) vs the f64 direct autocorrelation.
fn mtf_error(got: &[[f32; 2]], want: &mklight_wave::MtfCurve) -> f64 {
    let max = got
        .iter()
        .zip(want.tangential.iter().chain(&want.sagittal))
        .map(|(a, b)| ((a[0] as f64).hypot(a[1] as f64) - b).abs())
        .fold(0., f64::max);
    assert!(max <= 2e-3, "optical MTF max={max}");
    max
}
fn backend() -> MetalBackend {
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
        json!({"device":b.info().backend_id(),"build":b.precision().kernel_build,"startup_s":start.elapsed().as_secs_f64()})
    );
    b
}
fn pupil(name: &str, grid: usize, outer: bool) -> (Snapshot, PupilFunction) {
    let sys = mklight_core::samples::all()
        .into_iter()
        .find(|(n, _)| *n == name)
        .unwrap()
        .1;
    let tracer = mklight_analysis::tracer_for(sys.clone()).unwrap();
    let f = if outer { tracer.fields().len() - 1 } else { 0 };
    let settings = PupilSettings {
        grid,
        ..Default::default()
    };
    let tp = trace_pupil(&tracer, tracer.field(f), 0, &settings, &JobContext::new()).unwrap();
    (Snapshot::new(sys).unwrap(), tp.pupil)
}
fn rounded(field: &[C]) -> Vec<C> {
    field
        .iter()
        .map(|z| C::new(z.re as f32 as f64, z.im as f32 as f64))
        .collect()
}
/// A pupil field centred in an m×m zero grid (PSF padding).
fn embed(field: &[C], n: usize, m: usize) -> Vec<C> {
    let mut out = vec![C::default(); m * m];
    let o = (m - n) / 2;
    for j in 0..n {
        out[(j + o) * m + o..(j + o) * m + o + n].copy_from_slice(&field[j * n..(j + 1) * n]);
    }
    out
}
fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}
const RUNS: usize = 15;
/// Warm medians (one unmeasured warm-up): Metal wall to plan completion and
/// including final readback, Metal device time, and the f64 CPU reference wall.
fn time(case: &str, b: &dyn ComputeBackend, plan: &ResidentPlan, cpu: impl Fn()) -> Vec<[f32; 2]> {
    let run = || {
        let t = Instant::now();
        let r = b
            .submit_plan(plan.clone(), &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let done = t.elapsed().as_secs_f64();
        let v = read(b, &r);
        (t.elapsed().as_secs_f64(), r.stats, v, done)
    };
    let (cold_wall, cold_stats, got, _) = run();
    let cold_cpu = Instant::now();
    cpu();
    let cold_cpu = cold_cpu.elapsed().as_secs_f64();
    let (mut wall, mut device, mut host, mut complete) = (vec![], vec![], vec![], vec![]);
    let mut commands = 0;
    let runs = std::env::var("P2_RUNS").map_or(RUNS, |v| v.parse().unwrap());
    for _ in 0..runs {
        let (w, s, _, c) = run();
        wall.push(w);
        complete.push(c);
        device.push(s.device_seconds);
        commands = s.command_buffers;
        let t = Instant::now();
        cpu();
        host.push(t.elapsed().as_secs_f64());
    }
    let p95 = |v: &[f64]| {
        let mut v = v.to_vec();
        v.sort_by(f64::total_cmp);
        v[(v.len() as f64 * 0.95).ceil() as usize - 1] * 1e3
    };
    let raw = json!({"cpu_seconds":host,"metal_seconds":wall,"device_seconds":device,"complete_seconds":complete});
    let tails = json!({"cpu_p95_ms":p95(&host),"metal_p95_ms":p95(&wall)});
    let ms = |v: Vec<f64>| (median(v) * 1e5).round() / 100.;
    let min = device.iter().copied().fold(f64::INFINITY, f64::min);
    println!(
        "{}",
        json!({"case":case,"raw":raw,"p95":tails,"cold":{"cpu_seconds":cold_cpu,"metal_seconds":cold_wall,"stats":cold_stats},"cpu_f64_ms":ms(host),"metal_wall_ms":ms(wall),"metal_complete_ms":ms(complete),"metal_device_ms":ms(device),"metal_device_min_ms":(min * 1e5).round() / 100.,"commands":commands})
    );
    got
}
fn sizes(b: &dyn ComputeBackend) {
    let (s, p1024) = pupil("singlet", 1024, true);
    let field = rounded(&p1024.debye_field());
    for n in [64, 128, 256, 512, 1024, 2048] {
        let (s, src) = match n {
            1024 => (s.clone(), field.clone()),
            2048 => (s.clone(), embed(&field, 1024, 2048)),
            _ => {
                let (s, p) = pupil("singlet", n, true);
                (s, rounded(&p.debye_field()))
            }
        };
        let plan = transform_plan(b, s, &src, n, n, None);
        let want = reference(&src, n, n, None);
        let got = time(&format!("fft {n}²"), b, &plan, || {
            drop(reference(&src, n, n, None))
        });
        let rel = error(&got, &want, src.iter().map(|z| z.norm()).sum());
        println!("{}", json!({"case":format!("fft {n}²"),"relative_l2":rel}));
    }
    let (s256, p256) = pupil("singlet", 256, true);
    let f256 = rounded(&p256.debye_field());
    for (n, m, skew, src, s) in [
        (256, 512, false, &f256, &s256),
        (1024, 2048, false, &field, &s),
        (256, 512, true, &f256, &s256),
        (1024, 2048, true, &field, &s),
    ] {
        let a = Some(mft_affine(n, skew));
        let case = format!("{} {n}→{m}", if skew { "skew MFT" } else { "MFT" });
        let plan = transform_plan(b, s.clone(), src, n, m, a);
        let want = reference(src, n, m, a);
        let got = time(&case, b, &plan, || drop(reference(src, n, m, a)));
        let rel = error(&got, &want, src.iter().map(|z| z.norm()).sum());
        println!("{}", json!({"case":case,"relative_l2":rel}));
    }
    let freqs: Vec<f64> = (0..32).map(|i| 3.0 * i as f64).collect();
    for (n, s, p) in [(256, &s256, &p256), (1024, &s, &p1024)] {
        let case = format!("optical correlation {n}², {} frequencies", 2 * freqs.len());
        let plan = optical_plan(b, s.clone(), p, &freqs);
        let ctx = JobContext::new();
        let cpu = || mklight_wave::mtf_direct(p, &freqs, &Default::default(), &ctx).unwrap();
        let want = cpu();
        let got = time(&case, b, &plan, || drop(cpu()));
        println!(
            "{}",
            json!({"case":case,"mtf_max_abs":mtf_error(&got, &want)})
        );
    }
}
fn check(
    b: &dyn ComputeBackend,
    s: Snapshot,
    field: &[C],
    n: usize,
    m: usize,
    affine: Option<[f64; 6]>,
) -> f64 {
    let r = b
        .submit_plan(
            transform_plan(b, s, field, n, m, affine),
            &JobContext::new(),
        )
        .unwrap()
        .wait()
        .unwrap();
    error(
        &read(b, &r),
        &reference(field, n, m, affine),
        field.iter().map(|z| z.norm()).sum(),
    )
}
fn main() {
    let b = backend();
    let mode = std::env::args().nth(1).unwrap_or("samples".into());
    if mode == "sizes" {
        return sizes(&b);
    }
    let (mut fft, mut mtf) = (0f64, 0f64);
    for (name, sys) in mklight_core::samples::all() {
        let snapshot = Snapshot::new(sys.clone()).unwrap();
        let tracer = mklight_analysis::tracer_for(sys).unwrap();
        if mode == "samples" {
            for f in 0..tracer.fields().len() {
                for w in 0..tracer.num_wavelengths() {
                    let tp = trace_pupil(
                        &tracer,
                        tracer.field(f),
                        w,
                        &PupilSettings::default(),
                        &JobContext::new(),
                    )
                    .unwrap();
                    let n = tp.pupil.n();
                    let rel = check(
                        &b,
                        snapshot.clone(),
                        &rounded(&tp.pupil.debye_field()),
                        n,
                        n,
                        None,
                    );
                    let freqs = [0., 5., 17., 53.];
                    let r = b
                        .submit_plan(
                            optical_plan(&b, snapshot.clone(), &tp.pupil, &freqs),
                            &JobContext::new(),
                        )
                        .unwrap()
                        .wait()
                        .unwrap();
                    let want = mklight_wave::mtf_direct(
                        &tp.pupil,
                        &freqs,
                        &Default::default(),
                        &JobContext::new(),
                    )
                    .unwrap();
                    let m = mtf_error(&read(&b, &r), &want);
                    (fft, mtf) = (fft.max(rel), mtf.max(m));
                    println!(
                        "{}",
                        json!({"name":name,"field":f,"wavelength":w,"n":n,"fft_relative_l2":rel,"mtf_max_abs":m})
                    );
                }
            }
        } else if [
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
            for n in [64, 256, 1024] {
                let (s, p) = pupil(name, n, true);
                let field = rounded(&p.debye_field());
                let mut cases = vec![(n, None)];
                if n == 1024 {
                    cases.extend([
                        (2048, Some(mft_affine(n, false))),
                        (2048, Some(mft_affine(n, true))),
                    ]);
                }
                for (m, a) in cases {
                    let rel = check(&b, s.clone(), &field, n, m, a);
                    fft = fft.max(rel);
                    println!(
                        "{}",
                        json!({"name":name,"n":n,"m":m,"affine":a,"relative_l2":rel})
                    );
                }
            }
        }
    }
    let summary = if mode == "samples" {
        json!({"mode":mode,"max_relative_l2":fft,"max_mtf_abs":mtf})
    } else {
        json!({"mode":mode,"max_relative_l2":fft})
    };
    println!("{summary}");
}
