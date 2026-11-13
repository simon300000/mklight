//! Scalar-wave operations on the device (`ops/wave.rs`) against the CPU-f32
//! specification's plans, and the wave analyses on `gpu:webgpu` against the
//! CPU reference and `gpu:metal` under the GPU.md §9.4 gates
//! (`mklight_bench::comparison`).
//!
//! Results without trigonometry (focus phases, phase steps, moments,
//! intensities, line spreads, image metrics, real spectral sums) are
//! asserted bitwise; the others (focus correlations, prepared fields,
//! phased spectral sums, image queries) differ from the specification only
//! through the kernels' polynomial cis and are bounded far inside the gates.

use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{Df32, resident::*, transforms as tr, wave_ops as w, *};

fn spec(b: &mklight_webgpu::WebGpuBackend) -> CpuF32Backend {
    CpuF32Backend::matching(&b.precision().probe)
}

fn snapshot() -> Snapshot {
    Snapshot::new(mklight_core::samples::singlet()).unwrap()
}

/// Upload `inputs`, build a plan over their views and return the words of
/// its outputs and the stage names.
fn run(
    b: &dyn QualificationBackend,
    inputs: &[(BufferData, Vec<usize>)],
    build: impl FnOnce(&mut ResidentPlan, &[PlanView]) -> Vec<BufferId>,
) -> (Vec<Vec<u32>>, Vec<String>) {
    let handles: Vec<_> = inputs
        .iter()
        .map(|(d, shape)| {
            b.upload_buffer(BufferDesc::new(d.element(), shape.clone()), d.clone())
                .unwrap()
        })
        .collect();
    let mut p = ResidentPlan::new(snapshot());
    let views: Vec<PlanView> = handles
        .iter()
        .map(|h| {
            let id = p.import(h.view());
            p.view(id).unwrap()
        })
        .collect();
    p.outputs = build(&mut p, &views);
    let ctx = JobContext::new();
    let r = b.submit_plan(p, &ctx).unwrap().wait().unwrap();
    assert_eq!(ctx.progress.fraction(), 1.);
    let out = r
        .outputs
        .iter()
        .map(|o| b.readback(o.view(), &ctx).unwrap().data.words().to_vec())
        .collect();
    let names = r
        .stats
        .stages
        .iter()
        .map(|s| s.name.clone())
        .filter(|n| n != "metadata")
        .collect();
    (out, names)
}

/// A synthetic pupil of side n: fractional rim coverage, apodised
/// amplitude, defocus, astigmatism and coma of a few waves.
fn pupil(n: usize) -> Vec<(BufferData, Vec<usize>)> {
    let (mut a, mut c, mut o) = (vec![], vec![], vec![]);
    for k in 0..n * n {
        let u = (2 * (k % n) + 1) as f64 / n as f64 - 1.;
        let v = (2 * (k / n) + 1) as f64 / n as f64 - 1.;
        let r2 = u * u + v * v;
        let cov = ((1. - r2.sqrt()) * n as f64 / 2. + 0.5).clamp(0., 1.);
        c.push(cov as f32);
        a.push((1. - 0.3 * r2) as f32);
        let opd = 1.3 * (2. * r2 - 1.) + 0.7 * (u * u - v * v) + 0.9 * (3. * r2 - 2.) * v;
        o.push(Df32::from_f64(if cov > 0. { opd } else { 0. }));
    }
    vec![
        (BufferData::F32(a), vec![n, n]),
        (BufferData::F32(c), vec![n, n]),
        (BufferData::DoubleFloat(o), vec![n, n]),
    ]
}

fn geometry(radius: f64) -> w::Geometry {
    w::Geometry {
        sin_max: Df32::from_f64(0.125),
        wavelength: Df32::from_f64(0.00055),
        radius: Df32::from_f64(radius),
    }
}

fn wave_pupil(v: &[PlanView]) -> w::Pupil {
    w::Pupil {
        amplitude: v[0].clone(),
        coverage: v[1].clone(),
        opd: v[2].clone(),
    }
}

fn df_buffer(values: &[f64]) -> (BufferData, Vec<usize>) {
    (
        BufferData::DoubleFloat(values.iter().map(|&z| Df32::from_f64(z)).collect()),
        vec![values.len()],
    )
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
            if x.is_finite() {
                (*x as f64 - *y as f64).abs()
            } else {
                0.
            }
        })
        .fold(0., f64::max)
}

/// The fraction of identical words.
fn identical(a: &[u32], b: &[u32]) -> f64 {
    a.iter().zip(b).filter(|(x, y)| x == y).count() as f64 / a.len().max(1) as f64
}

fn on_device(names: &[String]) {
    assert!(
        names.iter().all(|n| n.starts_with("webgpu:")),
        "stages {names:?}"
    );
}

/// Focus phases and phase steps for an infinite and a finite reference
/// sphere, both sides of focus: bitwise the specification.
#[test]
fn focus_bitwise() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    for (n, radius) in [(9, 100.), (33, 0.), (64, 42.)] {
        let mut inputs = pupil(n);
        inputs.push(df_buffer(&[-200., -0.37, 0., 0.004, 0.1, 25.]));
        let build = |p: &mut ResidentPlan, v: &[PlanView]| {
            let out = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [6, n, n]));
            let steps = p.scratch(BufferDesc::new(ElementType::F32, [6]));
            p.operations.push(ResidentOp::Wave(w::Request::Focus {
                pupil: wave_pupil(v),
                geometry: geometry(radius),
                shifts: v[3].clone(),
                output: out,
                phase_steps: Some(steps),
            }));
            vec![out, steps]
        };
        let (g, names) = run(b, &inputs, build);
        on_device(&names);
        let (c, _) = run(&cpu, &inputs, build);
        assert_eq!(g[0], c[0], "focus phases n {n} radius {radius}");
        assert_eq!(g[1], c[1], "phase steps n {n} radius {radius}");
    }
}

type Inputs = Vec<(BufferData, Vec<usize>)>;

/// Stencil and focus correlation of one pupil (frame: chief ray `chief`).
fn correlation_plan(
    n: usize,
    freqs: &[[f32; 2]],
    shifts: Option<&[f64]>,
    chief: [f64; 2],
    radius: f64,
) -> (
    Inputs,
    impl Fn(&mut ResidentPlan, &[PlanView]) -> Vec<BufferId> + Copy + use<>,
) {
    let mut inputs = pupil(n);
    inputs.push((
        BufferData::F32(freqs.iter().flatten().copied().collect()),
        vec![freqs.len(), 2],
    ));
    inputs.push(df_buffer(shifts.unwrap_or(&[0.])));
    let planes = shifts.map_or(1, |s| s.len());
    let shifted = shifts.is_some();
    let m = freqs.len();
    let [dx, dy] = chief;
    let dz = (1. - dx * dx - dy * dy).sqrt();
    let q = 1. / (1. + dz);
    let frame = [
        [1. - dx * dx * q, -dx * dy * q, -dx],
        [-dx * dy * q, 1. - dy * dy * q, -dy],
        [dx, dy, dz],
    ]
    .map(|r| r.map(Df32::from_f64));
    let on_axis = dx == 0. && dy == 0.;
    let build = move |p: &mut ResidentPlan, v: &[PlanView]| {
        let stencil = p.scratch(BufferDesc::new(ElementType::F32, [m, n * n, 5]));
        p.operations
            .push(ResidentOp::Transforms(tr::Request::PrepareOpticalStencil(
                tr::OpticalCorrelation {
                    stencil: None,
                    amplitude: v[0].clone(),
                    opd: v[2].clone(),
                    coverage: v[1].clone(),
                    frequencies: v[3].clone(),
                    frame,
                    sin_max: geometry(radius).sin_max,
                    wavelength_over_index: geometry(radius).wavelength,
                    output: stencil,
                },
            )));
        let out = p.scratch(BufferDesc::new(ElementType::Complex32, [planes, m]));
        let steps = p.scratch(BufferDesc::new(ElementType::F32, [planes]));
        p.operations
            .push(ResidentOp::Wave(w::Request::FocusCorrelation {
                on_axis,
                pupil: wave_pupil(v),
                geometry: geometry(radius),
                stencil: p.view(stencil).unwrap(),
                shifts: shifted.then(|| v[4].clone()),
                output: out,
                phase_steps: Some(steps),
            }));
        vec![out, steps]
    };
    (inputs, build)
}

fn frequencies(m: usize, cutoff: f32) -> Vec<[f32; 2]> {
    (0..m)
        .map(|i| {
            let f = cutoff * i as f32 / m as f32;
            if i % 2 == 0 {
                [f, 0.]
            } else {
                [0.3 * f, 0.9 * f]
            }
        })
        .collect()
}

/// Focus correlations (OTF values, |·| ≤ 1) on and off axis, with and
/// without focus planes, against the specification: within 1e-5 absolute
/// (the MTF gate is 2e-3); phase steps bitwise.
#[test]
fn focus_correlation_matches_specification() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let shifts = [-0.05, -0.01, 0., 0.012, 0.04];
    for (n, chief, radius, planes) in [
        (16, [0., 0.], 0., None),
        (32, [0., 0.], 60., Some(&shifts[..])),
        (48, [0.12, -0.05], 75., Some(&shifts[..])),
        (64, [0.2, 0.], 0., None),
    ] {
        let (inputs, build) = correlation_plan(n, &frequencies(40, 440.), planes, chief, radius);
        let (g, names) = run(b, &inputs, build);
        assert_eq!(
            names,
            ["webgpu:optical-stencil", "webgpu:focus-correlation"]
        );
        let (c, _) = run(&cpu, &inputs, build);
        let err = max_abs(&g[0], &c[0]);
        eprintln!(
            "focus correlation n {n} chief {chief:?}: max |Δ| {err:.2e}, identical {:.2}",
            identical(&g[0], &c[0])
        );
        assert!(err <= 1e-5, "n {n}: {err:e}");
        assert_eq!(g[1], c[1], "phase steps n {n}");
    }
}

/// Batches of planes, frequency ranges of one plane and stencil slices give
/// the bits of one batch.
#[test]
fn focus_correlation_batches_identical() {
    use mklight_webgpu::ops::wave::limits;
    use std::sync::atomic::Ordering::Relaxed;
    let b = crate::gpu_or_skip!();
    let shifts = [-0.03, -0.01, 0., 0.01, 0.02, 0.03, 0.05];
    let (inputs, build) =
        correlation_plan(24, &frequencies(37, 400.), Some(&shifts), [0., 0.], 50.);
    let (whole, _) = run(b, &inputs, build);
    let count = 24 * 24;
    for (terms, stencil) in [
        (2 * 37 * count, usize::MAX),
        (5 * count, usize::MAX),
        (37 * count, 11 * count * 20),
        (1, 4 * count * 20),
    ] {
        limits::BATCH_TERMS.store(terms, Relaxed);
        limits::STENCIL_BYTES.store(stencil, Relaxed);
        let (g, names) = run(b, &inputs, build);
        limits::BATCH_TERMS.store(mklight_webgpu::ops::wave::BATCH_TERMS, Relaxed);
        limits::STENCIL_BYTES.store(usize::MAX, Relaxed);
        on_device(&names);
        assert_eq!(g, whole, "batch terms {terms}, stencil bytes {stencil}");
    }
}

/// Prepare (moments bitwise, the field within 1e-6 of its amplitude),
/// intensity from the reduced moments (bitwise; shifted FFT bins and MFT
/// order, peak and unit-energy normalisation) and line spreads (bitwise).
#[test]
fn prepare_intensity_line_spread() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    for n in [17, 64] {
        let side = 2 * n + 6;
        let inputs = pupil(n);
        let normal = [0.1, -0.2, -(1. - 0.05f64).sqrt()].map(Df32::from_f64);
        let build = |p: &mut ResidentPlan, v: &[PlanView]| {
            let field = p.scratch(BufferDesc::new(ElementType::Complex32, [side, side]));
            let moments = p.scratch(BufferDesc::new(ElementType::F32, [3, n * n]));
            p.operations.push(ResidentOp::Wave(w::Request::Prepare {
                pupil: wave_pupil(v),
                geometry: geometry(80.),
                normal,
                field,
                moments,
            }));
            vec![field, moments]
        };
        let (g, names) = run(b, &inputs, build);
        on_device(&names);
        let (c, _) = run(&cpu, &inputs, build);
        assert_eq!(g[1], c[1], "moments n {n}");
        let err = max_abs(&g[0], &c[0]);
        eprintln!("prepare n {n}: field max |Δ| {err:.2e}");
        assert!(err <= 1e-6, "field n {n}: {err:e}");

        // Intensity of the device field with synthetic reduced sums.
        let field = BufferData::Complex32(bytemuck::cast_slice(&g[0]).to_vec());
        let sums = df_buffer(&[37.25, 12.5, 11.75]);
        for (shifted, size, unit) in [
            (true, 2 * n + 1, true),
            (true, n, false),
            (false, side, true),
        ] {
            let inputs = vec![(field.clone(), vec![side, side]), sums.clone()];
            let build = |p: &mut ResidentPlan, v: &[PlanView]| {
                let out = p.scratch(BufferDesc::new(ElementType::F32, [size, size]));
                p.operations.push(ResidentOp::Wave(w::Request::Intensity {
                    field: v[0].clone(),
                    sums: v[1].clone(),
                    geometry: geometry(80.),
                    pupil_side: n,
                    size,
                    shifted,
                    unit_energy: unit,
                    projection: 0.93,
                    output: out,
                }));
                let lsf = p.scratch(BufferDesc::new(ElementType::F32, [2, size]));
                p.operations.push(ResidentOp::Wave(w::Request::LineSpread {
                    image: p.view(out).unwrap(),
                    spacing: 0.0013,
                    output: lsf,
                }));
                vec![out, lsf]
            };
            let (g, names) = run(b, &inputs, build);
            on_device(&names);
            let (c, _) = run(&cpu, &inputs, build);
            assert_eq!(g[0], c[0], "intensity n {n} shifted {shifted}");
            assert_eq!(g[1], c[1], "line spread n {n} shifted {shifted}");
        }
    }
}

/// Spectral sums (real: bitwise; phased complex curves within 1e-6) and
/// image integrals (metrics and complex images bitwise; PSF DFT and
/// circular/square spectral queries within 1e-5 of their scale).
#[test]
fn spectral_and_image_integrals() {
    let b = crate::gpu_or_skip!();
    let cpu = spec(b);
    let n = 48;
    let image: Vec<f32> = (0..n * n)
        .map(|k| {
            let (x, y) = ((k % n) as f32 - 20.3, (k / n) as f32 - 23.1);
            (-(x * x + 1.7 * y * y) / 40.).exp() + 1e-3 * ((k * 7 % 13) as f32)
        })
        .collect();
    let images: Vec<_> = (0..5)
        .map(|j| {
            (
                BufferData::F32(image.iter().map(|v| v * (1. + 0.1 * j as f32)).collect()),
                vec![n, n],
            )
        })
        .collect();
    let weights: Vec<Df32> = (0..5)
        .map(|j| Df32::from_f64(0.1 + 0.07 * j as f64))
        .collect();
    let build = |p: &mut ResidentPlan, v: &[PlanView]| {
        let out = p.scratch(BufferDesc::new(ElementType::F32, [n, n]));
        p.operations.push(ResidentOp::Wave(w::Request::Spectral(
            w::spectral::Spectral {
                inputs: v.to_vec(),
                weights: weights.clone(),
                frequencies: None,
                offsets: vec![[Df32::ZERO; 2]; 5],
                output: out,
            },
        )));
        let metrics = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [5]));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                w::image_integrals::ImageIntegral::Metrics {
                    input: p.view(out).unwrap(),
                    output: metrics,
                },
            )));
        let complex = p.scratch(BufferDesc::new(ElementType::Complex32, [n, n]));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                w::image_integrals::ImageIntegral::Complex {
                    input: p.view(out).unwrap(),
                    squared: false,
                    output: complex,
                },
            )));
        let squared = p.scratch(BufferDesc::new(ElementType::Complex32, [n, n]));
        p.operations
            .push(ResidentOp::Wave(w::Request::ImageIntegral(
                w::image_integrals::ImageIntegral::Complex {
                    input: p.view(complex).unwrap(),
                    squared: true,
                    output: squared,
                },
            )));
        vec![out, metrics, complex, squared]
    };
    let (g, names) = run(b, &images, build);
    on_device(&names);
    let (c, _) = run(&cpu, &images, build);
    assert_eq!(g, c, "spectral sum, metrics, complex images");

    // Phased complex curves.
    let m = 300;
    let mut curves: Vec<_> = (0..3)
        .map(|j| {
            (
                BufferData::Complex32(
                    (0..m)
                        .map(|k| {
                            let t = k as f32 / m as f32;
                            [1. - t * (0.5 + 0.1 * j as f32), 0.2 * t * (j as f32 - 1.)]
                        })
                        .collect(),
                ),
                vec![m],
            )
        })
        .collect();
    curves.push((
        BufferData::F32((0..2 * m).map(|k| (k as f32) * 0.37).collect()),
        vec![m, 2],
    ));
    let build = |p: &mut ResidentPlan, v: &[PlanView]| {
        let out = p.scratch(BufferDesc::new(ElementType::Complex32, [m]));
        p.operations.push(ResidentOp::Wave(w::Request::Spectral(
            w::spectral::Spectral {
                inputs: v[..3].to_vec(),
                weights: weights[..3].to_vec(),
                frequencies: Some(v[3].clone()),
                offsets: (0..3)
                    .map(|j| [0.0013 * j as f64, -0.0021].map(Df32::from_f64))
                    .collect(),
                output: out,
            },
        )));
        vec![out]
    };
    let (g, names) = run(b, &curves, build);
    on_device(&names);
    let (c, _) = run(&cpu, &curves, build);
    let err = max_abs(&g[0], &c[0]);
    eprintln!("phased spectral sum: max |Δ| {err:.2e}");
    assert!(err <= 1e-6, "{err:e}");

    // Queries: PSF DFT (kind 0) on the image, spectral integrals (kinds
    // 1, 2) on its complex spectrum (the complex image stands in for it).
    let queries: Vec<f64> = (0..40)
        .flat_map(|i| [0.9 * i as f64, if i % 3 == 0 { 0. } else { 0.4 * i as f64 }])
        .collect();
    for kind in [0, 1, 2] {
        let input = if kind == 0 {
            (BufferData::F32(image.clone()), vec![n, n])
        } else {
            (
                BufferData::Complex32(image.iter().map(|&v| [v, 0.3 * v]).collect()),
                vec![n, n],
            )
        };
        let inputs = vec![
            input,
            (
                BufferData::DoubleFloat(queries.iter().map(|&q| Df32::from_f64(q)).collect()),
                vec![40, 2],
            ),
        ];
        let build = |p: &mut ResidentPlan, v: &[PlanView]| {
            let out = p.scratch(BufferDesc::new(ElementType::Complex32, [40]));
            p.operations
                .push(ResidentOp::Wave(w::Request::ImageIntegral(
                    w::image_integrals::ImageIntegral::Query {
                        input: v[0].clone(),
                        queries: v[1].clone(),
                        kind,
                        spacing: Df32::from_f64(if kind == 0 { 0.0021 } else { 0.37 }),
                        center: [20.25, 23.5].map(Df32::from_f64),
                        normalization: Df32::from_f64(41.5),
                        output: out,
                    },
                )));
            vec![out]
        };
        let (g, names) = run(b, &inputs, build);
        on_device(&names);
        let (c, _) = run(&cpu, &inputs, build);
        let scale = floats(&c[0])
            .iter()
            .fold(0f64, |m, v| m.max(v.abs() as f64));
        let err = max_abs(&g[0], &c[0]) / scale;
        eprintln!("image query kind {kind}: max |Δ| / max {err:.2e}");
        assert!(err <= 1e-5, "kind {kind}: {err:e}");
    }
}

/// psf, mtf and through_focus_mtf of the Double Gauss on `gpu:webgpu`
/// within the GPU.md §9.4 gates of the CPU reference and of `gpu:metal`.
#[cfg(target_os = "macos")]
#[test]
fn analyses_within_gates() {
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
    let metal = engine
        .backends()
        .iter()
        .any(|b| b.id == "gpu:metal" && b.available);
    let system = mklight_core::samples::all()
        .into_iter()
        .find(|(name, _)| *name == "double_gauss")
        .unwrap()
        .1;
    let session = engine.session_from_project(Project::new(system)).unwrap();
    let run = |analysis: &str, backend: Backend| {
        let mut request = AnalysisRequest::new(analysis, serde_json::Value::Null);
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
    for analysis in ["psf", "mtf", "through_focus_mtf"] {
        let (used, webgpu) = run(analysis, Backend::Gpu("webgpu".into()));
        assert_eq!(used, "gpu:webgpu", "{analysis}");
        let mut against = vec![("cpu", run(analysis, Backend::CpuReference).1)];
        if metal {
            against.push(("metal", run(analysis, Backend::Gpu("metal".into())).1));
        }
        for (name, reference) in against {
            let c = comparison::compare(&reference, &webgpu, analysis, &JobContext::new()).unwrap();
            eprintln!(
                "{analysis} webgpu vs {name}: compared {}, max abs {:.2e} at {:?}, image L1 {:.2e}, energy {:.2e}, classes {:?}",
                c.compared,
                c.maximum,
                c.maximum_path,
                c.image_l1_relative,
                c.energy_relative,
                c.classes
            );
            assert!(c.compared > 0, "{analysis}");
            assert!(
                c.failures.is_empty(),
                "{analysis} vs {name}: {:?}",
                c.failures
            );
        }
    }
}
