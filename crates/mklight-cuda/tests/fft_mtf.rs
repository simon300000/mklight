//! The FFT-MTF operations (`FftMtf`, `kernels/fft_mtf.cu`) on the device
//! against the CPU-f32 specification, and the engine's FFT MTF on
//! `gpu:cuda` against the CPU reference under the GPU.md §9.4 gates.
//!
//! Moments, the normalisation and the domain count have no trigonometry
//! and are asserted bitwise; the sensor field and the OTF samples differ
//! only through the kernels' polynomial cis (and atan2f/hypotf of the
//! centroid) and are bounded far inside the gates.

use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{
    Df32,
    resident::*,
    transforms as tr,
    wave_ops::{self as w, fft_mtf::FftMtf, image_integrals::ImageIntegral},
    *,
};

const PLACE: &str = "cuda:fft-mtf";

fn spec(b: &mklight_webgpu::WebGpuBackend) -> CpuF32Backend {
    CpuF32Backend::matching(&b.precision().probe)
}

/// Upload `inputs`, build a plan over their views and return the words of
/// its outputs and the stage names (or the plan's error).
fn run(
    b: &dyn QualificationBackend,
    inputs: &[(BufferData, Vec<usize>)],
    build: impl FnOnce(&mut ResidentPlan, &[PlanView]) -> Vec<BufferId>,
) -> mklight_core::MkResult<(Vec<Vec<u32>>, Vec<String>)> {
    let handles: Vec<_> = inputs
        .iter()
        .map(|(d, shape)| {
            b.upload_buffer(BufferDesc::new(d.element(), shape.clone()), d.clone())
                .unwrap()
        })
        .collect();
    let mut p = ResidentPlan::new(Snapshot::new(mklight_core::samples::singlet()).unwrap());
    let views: Vec<PlanView> = handles
        .iter()
        .map(|h| {
            let id = p.import(h.view());
            p.view(id).unwrap()
        })
        .collect();
    p.outputs = build(&mut p, &views);
    let ctx = JobContext::new();
    let r = b.submit_plan(p, &ctx)?.wait()?;
    let out = r
        .outputs
        .iter()
        .map(|o| b.readback(o.view(), &ctx).unwrap().data.words().to_vec())
        .collect();
    let names = r.stats.stages.iter().map(|s| s.name.clone()).collect();
    Ok((out, names))
}

/// A pupil of side n: fractional rim coverage, apodised amplitude and a few
/// waves of defocus, astigmatism and coma (`aberrated`), or none.
fn pupil(n: usize, aberrated: bool) -> Vec<(BufferData, Vec<usize>)> {
    let (mut a, mut c, mut o) = (vec![], vec![], vec![]);
    for k in 0..n * n {
        let u = (2 * (k % n) + 1) as f64 / n as f64 - 1.;
        let v = (2 * (k / n) + 1) as f64 / n as f64 - 1.;
        let r2 = u * u + v * v;
        let cov = ((1. - r2.sqrt()) * n as f64 / 2. + 0.5).clamp(0., 1.);
        c.push(cov as f32);
        a.push((1. - 0.3 * r2) as f32);
        let opd = 1.3 * (2. * r2 - 1.) + 0.7 * (u * u - v * v) + 0.9 * (3. * r2 - 2.) * v;
        let opd = if cov > 0. && aberrated { opd } else { 0. };
        o.push(Df32::from_f64(opd));
    }
    vec![
        (BufferData::F32(a), vec![n, n]),
        (BufferData::F32(c), vec![n, n]),
        (BufferData::DoubleFloat(o), vec![n, n]),
    ]
}

/// An orthonormal chief frame tilted by `t` radians about x and `s` about y.
fn frame(t: f64, s: f64) -> [[Df32; 3]; 3] {
    let (st, ct, ss, cs) = (t.sin(), t.cos(), s.sin(), s.cos());
    let e = [
        [cs, 0., -ss],
        [st * ss, ct, st * cs],
        [ct * ss, -st, ct * cs],
    ];
    e.map(|v| v.map(Df32::from_f64))
}

fn floats(words: &[u32]) -> &[f32] {
    bytemuck::cast_slice(words)
}

fn max_abs(a: &[u32], b: &[u32]) -> f64 {
    assert_eq!(a.len(), b.len());
    floats(a)
        .iter()
        .zip(floats(b))
        .map(|(x, y)| {
            assert_eq!(x.is_finite(), y.is_finite(), "finiteness {x} {y}");
            (*x as f64 - *y as f64).abs()
        })
        .fold(0., f64::max)
}

fn relative_l2(a: &[u32], b: &[u32]) -> f64 {
    let (mut d, mut s) = (0., 0.);
    for (x, y) in floats(a).iter().zip(floats(b)) {
        d += (*x as f64 - *y as f64).powi(2);
        s += (*y as f64).powi(2);
    }
    (d / s.max(1e-300)).sqrt()
}

/// Frequencies inside the OTF lattice of side m and spacing d (through the
/// sample's mapping), on and off the axes.
fn frequencies(m: usize, d: f64) -> Vec<[f32; 2]> {
    let reach = (m / 2 - 3) as f64 * d * 0.9;
    (0..97)
        .map(|k| {
            let r = reach * k as f64 / 96.;
            let a = 0.37 * k as f64;
            [(r * a.cos()) as f32, (r * a.sin()) as f32]
        })
        .collect()
}

const MAPPING: [[f64; 2]; 2] = [[1.02, 0.03], [-0.01, 0.97]];

/// The FFT-OTF chain of `ResidentPupil::fft_lattice` for one pupil
/// (Domain, Sensor, Sum, FFT, |·|², FFT, Normalize); returns the field,
/// moments and normalised OTF.
fn lattice(
    p: &mut ResidentPlan,
    v: &[PlanView],
    sine: Df32,
    frame: [[Df32; 3]; 3],
    m: usize,
) -> [BufferId; 3] {
    let n = v[0].desc.shape[0];
    let status = p.scratch(BufferDesc::new(ElementType::U32, [1]));
    p.operations
        .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Domain {
            coverage: v[1].clone(),
            sin_max: sine,
            normal: frame.map(|e| e[2]),
            status,
        })));
    let field = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
    let moments = p.scratch(BufferDesc::new(ElementType::F32, [3, n * n]));
    p.operations
        .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Sensor {
            pupil: w::Pupil {
                amplitude: v[0].clone(),
                coverage: v[1].clone(),
                opd: v[2].clone(),
            },
            sin_max: sine,
            frame,
            field,
            moments,
        })));
    let mv = p.view(moments).unwrap();
    let sums = p.sum(mv, ReduceMode::DoubleFloat);
    let fft = |p: &mut ResidentPlan, input: BufferId, sign: i32| {
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
        let input = p.view(input).unwrap();
        p.operations
            .push(ResidentOp::Transforms(tr::Request::Transform(
                tr::TransformPlan {
                    kind: tr::TransformKind::Fft,
                    input,
                    output,
                    dimensions: vec![m, m],
                    output_dimensions: vec![m, m],
                    sign,
                    scale: 1.,
                    affine: [Df32::ZERO; 6],
                },
            )));
        output
    };
    let spectrum = fft(p, field, 1);
    let power = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
    let input = p.view(spectrum).unwrap();
    p.operations
        .push(ResidentOp::Wave(w::Request::ImageIntegral(
            ImageIntegral::Complex {
                input,
                squared: true,
                output: power,
            },
        )));
    let raw = fft(p, power, -1);
    let otf = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
    let (input, sums) = (p.view(raw).unwrap(), p.view(sums).unwrap());
    p.operations
        .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Normalize {
            input,
            sums,
            output: otf,
        })));
    [field, moments, otf]
}

/// The whole FFT-OTF chain (aberrated and reference pupils, sampled), on
/// and off axis: moments bitwise, fields, OTF lattices and samples within
/// far less than the gates.
#[test]
fn fft_otf_chain_matches_specification() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let (n, m) = (29, 64);
    let d = 0.05;
    let nus = frequencies(m, d);
    let mut inputs = pupil(n, true);
    inputs.extend(pupil(n, false));
    inputs.push((
        BufferData::F32(nus.iter().flatten().copied().collect()),
        vec![nus.len(), 2],
    ));
    for (t, s) in [(0., 0.), (0.21, -0.13)] {
        let build = |p: &mut ResidentPlan, v: &[PlanView]| {
            let sine = Df32::from_f64(0.125);
            let [field, moments, otf] = lattice(p, &v[0..3], sine, frame(t, s), m);
            let [_, _, reference] = lattice(p, &v[3..6], sine, frame(t, s), m);
            let output = p.scratch(BufferDesc::new(ElementType::Complex32, [nus.len()]));
            let (data, reference) = (p.view(otf).unwrap(), p.view(reference).unwrap());
            p.operations
                .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Sample {
                    data,
                    reference,
                    frequencies: v[6].clone(),
                    spacing: Df32::from_f64(d),
                    mapping: MAPPING.map(|r| r.map(Df32::from_f64)),
                    output,
                })));
            vec![field, moments, otf, output]
        };
        let (gpu, names) = run(b, &inputs, build).unwrap();
        let (want, _) = run(&cpu, &inputs, build).unwrap();
        let on = names.iter().filter(|s| *s == PLACE).count();
        assert_eq!(on, 7, "stages {names:?}");
        assert_eq!(gpu[1], want[1], "moments bitwise ({t}, {s})");
        let field = max_abs(&gpu[0], &want[0]);
        let otf = (max_abs(&gpu[2], &want[2]), relative_l2(&gpu[2], &want[2]));
        let sample = max_abs(&gpu[3], &want[3]);
        eprintln!(
            "tilt ({t}, {s}): field max abs {field:.2e}, OTF max abs {:.2e} rel L2 {:.2e}, samples max abs {sample:.2e}",
            otf.0, otf.1
        );
        assert!(field < 1e-5, "field {field}");
        assert!(otf.0 < 1e-5 && otf.1 < 1e-5, "OTF {otf:?}");
        assert!(sample < 1e-5, "samples {sample}");
    }
}

/// Normalize and Sample on host inputs: the normalisation bitwise, the
/// samples within the cis difference.
#[test]
fn normalize_bitwise_and_sample() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let m = 48;
    let d = 0.031;
    let nus = frequencies(m, d);
    let field = |seed: f32| -> Vec<[f32; 2]> {
        (0..m * m)
            .map(|k| {
                let (x, y) = ((k % m) as f32, (k / m) as f32);
                let r = 1. / (1. + 0.01 * (x * x + y * y));
                [
                    r * (seed * x + 0.3 * y).cos(),
                    r * (0.7 * x - seed * y).sin(),
                ]
            })
            .collect()
    };
    let inputs = vec![
        (BufferData::Complex32(field(0.11)), vec![m, m]),
        (BufferData::Complex32(field(0.07)), vec![m, m]),
        (
            BufferData::DoubleFloat([0.3, 0.123456789, 2.].map(Df32::from_f64).to_vec()),
            vec![3],
        ),
        (
            BufferData::F32(nus.iter().flatten().copied().collect()),
            vec![nus.len(), 2],
        ),
    ];
    let build = |p: &mut ResidentPlan, v: &[PlanView]| {
        let otf = p.scratch(BufferDesc::new(ElementType::Complex32, [m, m]));
        p.operations
            .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Normalize {
                input: v[0].clone(),
                sums: v[2].clone(),
                output: otf,
            })));
        let output = p.scratch(BufferDesc::new(ElementType::Complex32, [nus.len()]));
        let data = p.view(otf).unwrap();
        p.operations
            .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Sample {
                data,
                reference: v[1].clone(),
                frequencies: v[3].clone(),
                spacing: Df32::from_f64(d),
                mapping: MAPPING.map(|r| r.map(Df32::from_f64)),
                output,
            })));
        vec![otf, output]
    };
    let (gpu, names) = run(b, &inputs, build).unwrap();
    let (want, _) = run(&cpu, &inputs, build).unwrap();
    assert_eq!(names.iter().filter(|s| *s == PLACE).count(), 2, "{names:?}");
    assert_eq!(gpu[0], want[0], "normalisation bitwise");
    let sample = max_abs(&gpu[1], &want[1]);
    eprintln!("samples max abs {sample:.2e}");
    assert!(sample < 1e-5, "samples {sample}");
}

/// A pupil beyond the image surface's horizon fails as the specification
/// does, with the same count.
#[test]
fn domain_reports_horizon() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let inputs = pupil(33, false);
    let build = |p: &mut ResidentPlan, v: &[PlanView]| {
        let status = p.scratch(BufferDesc::new(ElementType::U32, [1]));
        p.operations
            .push(ResidentOp::Wave(w::Request::FftMtf(FftMtf::Domain {
                coverage: v[1].clone(),
                sin_max: Df32::from_f64(0.9),
                normal: [0.97, 0., 0.2431].map(Df32::from_f64),
                status,
            })));
        vec![status]
    };
    let gpu = run(b, &inputs, build).expect_err("device: horizon error");
    let want = run(&cpu, &inputs, build).expect_err("specification: horizon error");
    eprintln!("{gpu}");
    assert_eq!(gpu.to_string(), want.to_string());
}

/// The engine's FFT MTF on `gpu:cuda` within the GPU.md §9.4 gates of the
/// CPU reference, on and off axis, for a few systems.
#[test]
fn engine_fft_mtf_within_gates() {
    use mklight_bench::comparison;
    use mklight_engine::analysis::AnalysisOutput;
    use mklight_engine::{
        AnalysisRequest, Engine, EngineConfig, backend::Backend, project::Project,
    };
    crate::gpu_or_skip!();
    let engine = Engine::new(EngineConfig {
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    let settings = serde_json::json!({"method": "fft", "pupil": {"grid": 128}});
    for name in ["double_gauss", "cooke_triplet", "petzval_asphere"] {
        let system = mklight_core::samples::all()
            .into_iter()
            .find(|(n, _)| *n == name)
            .unwrap()
            .1;
        let session = engine.session_from_project(Project::new(system)).unwrap();
        let run = |backend: Backend| {
            let mut request = AnalysisRequest::new("mtf", settings.clone());
            request.backend = Some(backend);
            let r = session.run_analysis(&request, &JobContext::new()).unwrap();
            (
                r.backend.used.clone(),
                AnalysisOutput {
                    json: r.json,
                    buffers: r.buffers,
                    ..Default::default()
                },
            )
        };
        let (used, gpu) = run(Backend::Gpu("cuda".into()));
        assert_eq!(used, "gpu:cuda", "{name}");
        let reference = run(Backend::CpuReference).1;
        let c = comparison::compare(&reference, &gpu, "mtf", &JobContext::new()).unwrap();
        eprintln!(
            "{name} fft mtf cuda vs cpu: compared {}, max abs {:.2e} at {:?}, classes {:?}",
            c.compared, c.maximum, c.maximum_path, c.classes
        );
        assert!(c.compared > 0, "{name}");
        assert!(c.failures.is_empty(), "{name}: {:?}", c.failures);
    }
}
