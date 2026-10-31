use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{resident::*, transforms::*, *};
pub fn snapshot() -> Snapshot {
    Snapshot::new(mklight_core::samples::singlet()).unwrap()
}
pub fn input(n: usize) -> Vec<[f32; 2]> {
    (0..n)
        .map(|i| {
            [
                ((i * 37 % 101) as f32 - 50.) / 50.,
                ((i * 13 % 97) as f32 - 48.) / 48.,
            ]
        })
        .collect()
}
pub fn upload(b: &dyn ComputeBackend, data: Vec<[f32; 2]>, shape: Vec<usize>) -> BufferHandle {
    b.upload_buffer(
        BufferDesc::new(ElementType::Complex32, shape),
        BufferData::Complex32(data),
    )
    .unwrap()
}
pub fn run(
    b: &dyn ComputeBackend,
    data: &[[f32; 2]],
    dims: &[usize],
    od: &[usize],
    kind: TransformKind,
    sign: i32,
    scale: f32,
    affine: [Df32; 6],
) -> (Vec<[f32; 2]>, PlanResult) {
    let h = upload(b, data.to_vec(), dims.to_vec());
    let mut p = ResidentPlan::new(snapshot());
    let i = p.import(h.view());
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, od.to_vec()));
    p.operations
        .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
            kind,
            input: p.view(i).unwrap(),
            output: o,
            dimensions: dims.to_vec(),
            output_dimensions: od.to_vec(),
            sign,
            scale,
            affine,
        })));
    p.outputs = vec![o];
    let ctx = JobContext::new();
    let r = b.submit_plan(p, &ctx).unwrap().wait().unwrap();
    assert_eq!(ctx.progress.fraction(), 1.);
    assert_eq!(r.stats.readback_bytes, 0);
    assert_eq!(r.stats.fallback_count, 0);
    let BufferData::Complex32(v) = b.readback(r.outputs[0].view(), &ctx).unwrap().data else {
        panic!()
    };
    (v, r)
}
pub fn dft(
    data: &[[f32; 2]],
    dims: &[usize],
    od: &[usize],
    sign: i32,
    scale: f32,
    a: Option<[Df32; 6]>,
) -> Vec<[f64; 2]> {
    let (w, h) = recipe::shape(dims);
    let (ow, oh) = recipe::shape(od);
    let a =
        a.map(|a| a.map(Df32::to_f64))
            .unwrap_or([1. / w as f64, 0., 0., 0., 1. / h as f64, 0.]);
    (0..ow * oh)
        .map(|k| {
            let (x, y) = ((k % ow) as f64, (k / ow) as f64);
            let mut z = [0.; 2];
            for j in 0..w * h {
                let (u, v) = ((j % w) as f64, (j / w) as f64);
                let ph = sign as f64
                    * std::f64::consts::TAU
                    * (u * (a[0] * x + a[1] * y + a[2]) + v * (a[3] * x + a[4] * y + a[5]));
                let (s, c) = ph.sin_cos();
                z[0] += (data[j][0] as f64 * c - data[j][1] as f64 * s) * scale as f64;
                z[1] += (data[j][0] as f64 * s + data[j][1] as f64 * c) * scale as f64;
            }
            z
        })
        .collect()
}
pub fn compare(got: &[[f32; 2]], want: &[[f64; 2]], l1: f64) -> f64 {
    let mut error = 0.;
    let mut norm = 0.;
    let mut max = 0f64;
    for (a, b) in got.iter().zip(want) {
        for i in 0..2 {
            let e = a[i] as f64 - b[i];
            assert!(e.is_finite());
            error += e * e;
            norm += b[i] * b[i];
            max = max.max(e.abs());
        }
    }
    let rel = (error / norm.max(1e-30)).sqrt();
    assert!(rel <= 1e-4, "relative L2={rel}");
    assert!(max <= 1e-4 * l1, "scaled max={max}, L1={l1}");
    rel
}
pub fn suite(b: &dyn ComputeBackend) {
    let mut worst = 0f64;
    for dims in [
        vec![1],
        vec![2],
        vec![3],
        vec![5],
        vec![7],
        vec![11],
        vec![60],
        vec![77],
        vec![3, 5],
        vec![7, 11],
        vec![16, 32],
    ] {
        let n = dims.iter().product();
        let a = input(n);
        for sign in [-1, 1] {
            let (g, _) = run(
                b,
                &a,
                &dims,
                &dims,
                TransformKind::Fft,
                sign,
                0.37,
                [Df32::ZERO; 6],
            );
            let r = dft(&a, &dims, &dims, sign, 0.37, None);
            worst = worst.max(compare(&g, &r, n as f64));
        }
    }
    for (dims, od, affine, kind) in [
        (
            vec![37],
            vec![53],
            [0.013 * 0.21, 0., -0.73, 0., 0., 0.],
            TransformKind::ChirpZ,
        ),
        (
            vec![5, 7],
            vec![9, 11],
            [0.021, 0., -0.17, 0., 0.039, 0.081],
            TransformKind::ChirpZ,
        ),
        (
            vec![7, 9],
            vec![13, 11],
            [0.031, -0.007, 0.23, -0.007, 0.019, -0.14],
            TransformKind::AffineMft,
        ),
        (
            vec![17],
            vec![23],
            [100001.03125, 0., 12345.125, 0., 0., 0.],
            TransformKind::ChirpZ,
        ),
    ] {
        let n = dims.iter().product();
        let a = input(n);
        let affine = affine.map(Df32::from_f64);
        let (g, _) = run(b, &a, &dims, &od, kind, 1, 0.25, affine);
        let r = dft(&a, &dims, &od, 1, 0.25, Some(affine));
        worst = worst.max(compare(&g, &r, n as f64));
    }
    for n in [64, 256] {
        let a = input(n);
        let (g, _) = run(
            b,
            &a,
            &[n],
            &[n],
            TransformKind::Fft,
            -1,
            1.,
            [Df32::ZERO; 6],
        );
        let (back, _) = run(
            b,
            &g,
            &[n],
            &[n],
            TransformKind::Fft,
            1,
            1. / n as f32,
            [Df32::ZERO; 6],
        );
        compare(
            &back,
            &a.iter().map(|v| v.map(f64::from)).collect::<Vec<_>>(),
            n as f64,
        );
        let energy = |v: &[[f32; 2]]| {
            v.iter()
                .map(|v| v[0] as f64 * v[0] as f64 + v[1] as f64 * v[1] as f64)
                .sum::<f64>()
        };
        assert!((energy(&g) / (n as f64 * energy(&a)) - 1.).abs() < 1e-4);
    }
    eprintln!("P2 small DFT/MFT max relative L2={worst:e}");
}
#[test]
fn cpu_f32_independent_dft_mft() {
    suite(&CpuF32Backend::new());
}
#[test]
fn boundary_validation() {
    let b = CpuF32Backend::new();
    let h = upload(&b, input(16), vec![4, 4]);
    let mut p = ResidentPlan::new(snapshot());
    let i = p.import(h.view());
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![4, 4]));
    let mut t = TransformPlan {
        kind: TransformKind::Fft,
        input: p.view(i).unwrap(),
        output: o,
        dimensions: vec![4, 4],
        output_dimensions: vec![4, 4],
        sign: 0,
        scale: 1.,
        affine: [Df32::ZERO; 6],
    };
    p.operations
        .push(ResidentOp::Transforms(Request::Transform(t.clone())));
    p.outputs = vec![o];
    assert!(b.submit_plan(p.clone(), &JobContext::new()).is_err());
    t.sign = -1;
    t.input.stride = 2;
    p.operations[0] = ResidentOp::Transforms(Request::Transform(t));
    assert!(b.submit_plan(p, &JobContext::new()).is_err());
}
pub fn correlation_suite(b: &dyn ComputeBackend) {
    let mut stencil = None;
    for defocus in [0.0, 30.0, -30.0] {
        stencil = Some(correlation_case(b, defocus, stencil.as_ref()));
    }
}
fn correlation_case(
    b: &dyn ComputeBackend,
    defocus: f64,
    previous: Option<&BufferHandle>,
) -> BufferHandle {
    let n = 17;
    let count = n * n;
    let angle = 0.37f64;
    let (s, c) = angle.sin_cos();
    let frame = [[c, 0., -s], [0., 1., 0.], [s, 0., c]];
    let sin_max = 0.17;
    let wave = 0.00055;
    let mut amp = vec![0f32; count];
    let mut cov = vec![0f32; count];
    let mut opd = vec![Df32::ZERO; count];
    for k in 0..count {
        let u = (2 * (k % n) + 1) as f64 / n as f64 - 1.;
        let v = (2 * (k / n) + 1) as f64 / n as f64 - 1.;
        let r = u * u + v * v;
        if r < 0.93 && r > 0.09 {
            cov[k] = if r > 0.7 { 0.43 } else { 1. };
            amp[k] = (0.8 + 0.2 * u) as f32;
            opd[k] = Df32::from_f64(12.1 * u + 3.7 * v + (0.43 + defocus) * r);
        }
    }
    let frequencies = vec![0f32, 0., 13., -7., 55., 27., 300., 0.];
    let uploaded = |element, data| {
        b.upload_buffer(BufferDesc::new(element, vec![n, n]), data)
            .unwrap()
    };
    let ah = uploaded(ElementType::F32, BufferData::F32(amp.clone()));
    let ch = uploaded(ElementType::F32, BufferData::F32(cov.clone()));
    let wh = uploaded(
        ElementType::DoubleFloat,
        BufferData::DoubleFloat(opd.clone()),
    );
    let fh = b
        .upload_buffer(
            BufferDesc::new(ElementType::F32, vec![4, 2]),
            BufferData::F32(frequencies.clone()),
        )
        .unwrap();
    let mut p = ResidentPlan::new(snapshot());
    let a = p.import(ah.view());
    let cc = p.import(ch.view());
    let w = p.import(wh.view());
    let f = p.import(fh.view());
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![4]));
    let mut req = OpticalCorrelation {
        stencil: None,
        amplitude: p.view(a).unwrap(),
        opd: p.view(w).unwrap(),
        coverage: p.view(cc).unwrap(),
        frequencies: p.view(f).unwrap(),
        frame: frame.map(|v| v.map(Df32::from_f64)),
        sin_max: Df32::from_f64(sin_max),
        wavelength_over_index: Df32::from_f64(wave),
        output: o,
    };
    let st = if let Some(previous) = previous {
        p.import(previous.view())
    } else {
        let st = p.scratch(BufferDesc::new(ElementType::F32, vec![4, count, 5]));
        let mut prep = req.clone();
        prep.output = st;
        p.operations
            .push(ResidentOp::Transforms(Request::PrepareOpticalStencil(prep)));
        st
    };
    req.stencil = Some(p.view(st).unwrap());
    p.operations
        .push(ResidentOp::Transforms(Request::OpticalCorrelation(
            req.clone(),
        )));
    p.outputs = if previous.is_some() {
        vec![o]
    } else {
        vec![o, st]
    };
    let r = b
        .submit_plan(p.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let BufferData::Complex32(got) = b
        .readback(r.outputs[0].view(), &JobContext::new())
        .unwrap()
        .data
    else {
        panic!()
    };
    let dot = |x: [f64; 3], y: [f64; 3]| x.iter().zip(y).map(|(x, y)| x * y).sum::<f64>();
    let coords = |k: usize| {
        let a = sin_max * ((2 * (k % n) + 1) as f64 / n as f64 - 1.);
        let v = sin_max * ((2 * (k / n) + 1) as f64 / n as f64 - 1.);
        let g = (1. - a * a - v * v).sqrt();
        let q =
            std::array::from_fn::<_, 3, _>(|i| frame[0][i] * a + frame[1][i] * v - frame[2][i] * g);
        (q, g)
    };
    let mut want = vec![];
    for nu in frequencies.chunks_exact(2) {
        let mut re = 0.;
        let mut im = 0.;
        let mut norm = 0.;
        for k in 0..count {
            if cov[k] == 0. {
                continue;
            }
            let (q, g) = coords(k);
            let e = cov[k] as f64 * (amp[k] as f64).powi(2) / (g * -q[2]);
            norm += e;
            let qx = q[0] + wave * nu[0] as f64;
            let qy = q[1] + wave * nu[1] as f64;
            let z2 = 1. - qx * qx - qy * qy;
            if z2 <= 0. {
                continue;
            }
            let z = -z2.sqrt();
            let sh = [qx, qy, z];
            if dot(sh, frame[2]) >= 0. {
                continue;
            }
            let ix = dot(sh, frame[0]) * 0.5 * n as f64 / sin_max + 0.5 * (n - 1) as f64;
            let iy = dot(sh, frame[1]) * 0.5 * n as f64 / sin_max + 0.5 * (n - 1) as f64;
            let fx = ix.floor();
            let fy = iy.floor();
            let tx = ix - fx;
            let ty = iy - fy;
            let mut a = 0.;
            let mut ww = 0.;
            let mut ph = 0.;
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = fx as i64 + dx;
                    let y = fy as i64 + dy;
                    if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                        continue;
                    }
                    let j = y as usize * n + x as usize;
                    let weight = (if dx == 0 { 1. - tx } else { tx })
                        * (if dy == 0 { 1. - ty } else { ty })
                        * cov[j] as f64;
                    a += weight * amp[j] as f64;
                    ww += weight;
                    ph += weight * opd[j].to_f64();
                }
            }
            if ww > 0. {
                let ph = std::f64::consts::TAU * (ph / ww - opd[k].to_f64());
                let magnitude = cov[k] as f64 * amp[k] as f64 / g * a / -z;
                re += magnitude * ph.cos();
                im += magnitude * ph.sin();
            }
            re += (1. - cov[k] as f64)
                * e
                * (1. - (ix - (k % n) as f64).abs()).max(0.)
                * (1. - (iy - (k / n) as f64).abs()).max(0.);
        }
        want.push([re / norm, im / norm]);
    }
    let mut max = 0f64;
    for (a, b) in got.iter().zip(&want) {
        for i in 0..2 {
            max = max.max((a[i] as f64 - b[i]).abs());
        }
    }
    assert!(max < 2e-3, "optical correlation error {max}");
    assert!((got[0][0] - 1.).abs() < 2e-5);
    eprintln!("P2 off-axis annular optical correlation max_abs={max:e}");
    // Reuse the produced stencil in another plan, then compare with fresh geometry.
    let mut q = ResidentPlan::new(snapshot());
    let a = q.import(ah.view());
    let c = q.import(ch.view());
    let w = q.import(wh.view());
    let f = q.import(fh.view());
    let geometry = previous.cloned().unwrap_or_else(|| r.outputs[1].clone());
    let stencil = q.import(geometry.view());
    let o = q.scratch(BufferDesc::new(ElementType::Complex32, vec![4]));
    req.amplitude = q.view(a).unwrap();
    req.coverage = q.view(c).unwrap();
    req.opd = q.view(w).unwrap();
    req.frequencies = q.view(f).unwrap();
    req.stencil = Some(q.view(stencil).unwrap());
    req.output = o;
    q.operations
        .push(ResidentOp::Transforms(Request::OpticalCorrelation(
            req.clone(),
        )));
    q.outputs = vec![o];
    let rr = b
        .submit_plan(q.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        b.readback(rr.outputs[0].view(), &JobContext::new())
            .unwrap()
            .data
            .words(),
        bytemuck::cast_slice::<_, u32>(&got)
    );
    req.stencil = None;
    q.operations[0] = ResidentOp::Transforms(Request::OpticalCorrelation(req));
    let rr = b
        .submit_plan(q, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        b.readback(rr.outputs[0].view(), &JobContext::new())
            .unwrap()
            .data
            .words(),
        bytemuck::cast_slice::<_, u32>(&got)
    );
    geometry
}
pub fn primitives_suite(b: &dyn ComputeBackend) {
    let dims = vec![7, 11];
    let data = input(77);
    let mut impulse = vec![[0.; 2]; 77];
    impulse[0] = [0.25, 0.];
    impulse[1] = [0.5, 0.];
    impulse[13] = [0.25, 0.];
    let h = upload(b, data.clone(), dims.clone());
    let k = upload(b, impulse, dims.clone());
    let mut p = ResidentPlan::new(snapshot());
    let i = p.import(h.view());
    let j = p.import(k.view());
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, dims.clone()));
    p.operations.push(ResidentOp::Transforms(Request::Convolve {
        left: p.view(i).unwrap(),
        right: p.view(j).unwrap(),
        output: o,
    }));
    p.outputs = vec![o];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let BufferData::Complex32(got) = b
        .readback(r.outputs[0].view(), &JobContext::new())
        .unwrap()
        .data
    else {
        panic!()
    };
    let expected: Vec<[f64; 2]> = (0..77)
        .map(|k| {
            let (x, y) = (k % 11, k / 11);
            std::array::from_fn(|c| {
                0.25 * data[k][c] as f64
                    + 0.5 * data[y * 11 + (x + 10) % 11][c] as f64
                    + 0.25 * data[((y + 6) % 7) * 11 + (x + 9) % 11][c] as f64
            })
        })
        .collect();
    compare(&got, &expected, 77.);
    for c in 0..2 {
        let before = data.iter().map(|v| v[c] as f64).sum::<f64>();
        let after = got.iter().map(|v| v[c] as f64).sum::<f64>();
        assert!((before - after).abs() < 1e-4);
    }

    for mode in 0..3 {
        let n = 64;
        let a: Vec<_> = (0..n)
            .map(|k| match mode {
                0 => {
                    if k == 0 {
                        [1., 0.]
                    } else {
                        [0., 0.]
                    }
                }
                1 => [1., 0.],
                _ => {
                    let ph = std::f64::consts::TAU * 7. * k as f64 / n as f64;
                    [ph.cos() as f32, ph.sin() as f32]
                }
            })
            .collect();
        let (g, _) = run(
            b,
            &a,
            &[n],
            &[n],
            TransformKind::Fft,
            -1,
            1.,
            [Df32::ZERO; 6],
        );
        for (k, z) in g.iter().enumerate() {
            let expected = if mode == 0 {
                1.
            } else if k == if mode == 1 { 0 } else { 7 } {
                64.
            } else {
                0.
            };
            assert!((z[0] - expected).abs() < 1e-4 && z[1].abs() < 1e-4);
        }
    }
}
#[test]
fn cpu_correlation_and_convolution() {
    let b = CpuF32Backend::new();
    correlation_suite(&b);
    primitives_suite(&b);
}
pub fn phase_and_views(b: &dyn ComputeBackend) {
    let data = input(35);
    let mut with_guard = vec![[99., -99.]; 3];
    with_guard.extend(data.clone());
    with_guard.extend([[77., 77.]; 2]);
    let h = upload(b, with_guard, vec![40]);
    let mut p = ResidentPlan::new(snapshot());
    let i = p.import(h.view());
    let input = PlanView {
        buffer: i,
        offset: 3,
        desc: BufferDesc::new(ElementType::Complex32, vec![5, 7]),
        stride: 1,
    };
    let coefficients = [
        100003.03125,
        -0.17,
        20000.123456,
        0.31,
        -0.7,
        17000000.123456,
    ]
    .map(Df32::from_f64);
    let o = p.scratch(input.desc.clone());
    p.operations.push(ResidentOp::Transforms(Request::Phase {
        input,
        coefficients,
        output: o,
    }));
    let t = p.scratch(BufferDesc::new(ElementType::Complex32, vec![7, 5]));
    p.operations
        .push(ResidentOp::Transforms(Request::Transpose {
            input: p.view(o).unwrap(),
            output: t,
        }));
    p.outputs = vec![t];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let BufferData::Complex32(g) = b
        .readback(r.outputs[0].view(), &JobContext::new())
        .unwrap()
        .data
    else {
        panic!()
    };
    let c = coefficients.map(Df32::to_f64);
    let mut want = vec![[0.; 2]; 35];
    for y in 0..5 {
        for x in 0..7 {
            let (u, v) = (x as f64, y as f64);
            let turns = c[0] * u * u + c[1] * u * v + c[2] * v * v + c[3] * u + c[4] * v + c[5];
            let ph = std::f64::consts::TAU * (turns - turns.round());
            let (s, c) = ph.sin_cos();
            let a = data[y * 7 + x];
            want[x * 5 + y] = [
                a[0] as f64 * c - a[1] as f64 * s,
                a[0] as f64 * s + a[1] as f64 * c,
            ];
        }
    }
    compare(&g, &want, 35.);
}
#[test]
fn cpu_large_phase_and_offset_views() {
    phase_and_views(&CpuF32Backend::new());
}
#[test]
fn portable_parameter_layout() {
    use mklight_gpu::transforms::{correlation::Params as C, recipe::Params as T};
    use std::{
        mem::{align_of, offset_of, size_of},
        process::Command,
    };
    let dir = std::env::temp_dir().join(format!("mklight-p2-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = "#include <stdio.h>\n#include <stddef.h>\n#include \"mklight_gpu_types.h\"\nint main(void){printf(\"%zu %zu %zu %zu %zu %zu %zu %zu %zu\",sizeof(struct MklTransformParams),_Alignof(struct MklTransformParams),offsetof(struct MklTransformParams,affine),sizeof(struct MklCorrelationParams),_Alignof(struct MklCorrelationParams),offsetof(struct MklCorrelationParams,mapping),offsetof(struct MklCorrelationParams,frame),offsetof(struct MklCorrelationParams,sin_max),offsetof(struct MklCorrelationParams,reserved_tail));}\n";
    std::fs::write(dir.join("probe.c"), source).unwrap();
    let compile = Command::new("cc")
        .args([
            "-std=c11",
            "-Werror",
            "-Wpadded",
            "-I",
            mklight_gpu::INCLUDE_DIR,
        ])
        .arg(dir.join("probe.c"))
        .arg("-o")
        .arg(dir.join("probe"))
        .output()
        .unwrap();
    assert!(
        compile.status.success(),
        "{}",
        String::from_utf8_lossy(&compile.stderr)
    );
    let result = Command::new(dir.join("probe")).output().unwrap();
    let values: Vec<usize> = String::from_utf8(result.stdout)
        .unwrap()
        .split_whitespace()
        .map(|s| s.parse().unwrap())
        .collect();
    assert_eq!(
        values,
        [
            size_of::<T>(),
            align_of::<T>(),
            offset_of!(T, affine),
            size_of::<C>(),
            align_of::<C>(),
            offset_of!(C, mapping),
            offset_of!(C, frame),
            offset_of!(C, sin_max),
            offset_of!(C, reserved_tail)
        ]
    );
    std::fs::remove_dir_all(dir).unwrap();
}
pub fn complex_correlation_suite(b: &dyn ComputeBackend) {
    let (w, h) = (6, 4);
    let data = input(w * h);
    let cov: Vec<f32> = (0..w * h)
        .map(|k| {
            if k % 5 == 0 {
                0.0
            } else if k % 3 == 0 {
                0.4
            } else {
                1.
            }
        })
        .collect();
    let shifts = [[0., 0.], [1., 0.], [-2., 1.], [20., 0.], [0.25, 0.75]];
    let ih = upload(b, data.clone(), vec![h, w]);
    let ch = b
        .upload_buffer(
            BufferDesc::new(ElementType::F32, vec![h, w]),
            BufferData::F32(cov.clone()),
        )
        .unwrap();
    let fh = b
        .upload_buffer(
            BufferDesc::new(ElementType::F32, vec![12]),
            BufferData::F32([vec![99.], shifts.concat(), vec![77.]].concat()),
        )
        .unwrap();
    let mut p = ResidentPlan::new(snapshot());
    let i = p.import(ih.view());
    let c = p.import(ch.view());
    let f = p.import(fh.view());
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![5]));
    p.operations
        .push(ResidentOp::Transforms(Request::Correlation(
            CorrelationStencil {
                pupil: p.view(i).unwrap(),
                coverage: Some(p.view(c).unwrap()),
                frequencies: PlanView {
                    buffer: f,
                    offset: 1,
                    desc: BufferDesc::new(ElementType::F32, vec![5, 2]),
                    stride: 1,
                },
                mapping: [1., 0., 0., 0., 1., 0.].map(Df32::from_f64),
                output: o,
            },
        )));
    p.outputs = vec![o];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let BufferData::Complex32(got) = b
        .readback(r.outputs[0].view(), &JobContext::new())
        .unwrap()
        .data
    else {
        panic!()
    };
    let mut expected = vec![];
    for [sx, sy] in shifts {
        let mut sum = [0.; 2];
        let mut norm = 0.;
        for k in 0..w * h {
            let a = data[k].map(f64::from);
            let energy = cov[k] as f64 * (a[0] * a[0] + a[1] * a[1]);
            norm += energy;
            let x = (k % w) as f64 + sx as f64;
            let y = (k / w) as f64 + sy as f64;
            let mut shifted = [0.; 2];
            for j in 0..w * h {
                let weight = (1. - (x - (j % w) as f64).abs()).max(0.)
                    * (1. - (y - (j / w) as f64).abs()).max(0.)
                    * cov[j] as f64;
                for c in 0..2 {
                    shifted[c] += weight * data[j][c] as f64;
                }
            }
            sum[0] += cov[k] as f64 * (shifted[0] * a[0] + shifted[1] * a[1])
                + (1. - cov[k] as f64)
                    * energy
                    * (1. - (sx as f64).abs()).max(0.)
                    * (1. - (sy as f64).abs()).max(0.);
            sum[1] += cov[k] as f64 * (shifted[1] * a[0] - shifted[0] * a[1]);
        }
        expected.push(sum.map(|v| v / norm));
    }
    compare(&got, &expected, 1.);
}
#[test]
fn cpu_complex_correlation() {
    complex_correlation_suite(&CpuF32Backend::new());
}
pub fn physical_grid_mft(b: &dyn ComputeBackend) {
    let (w, h, ow, oh) = (7, 5, 11, 9);
    let data = input(w * h);
    let map = [[0.83, -0.17], [-0.17, 0.76]];
    let (dn, dx) = (0.13, 0.027);
    let source = [-0.5, -0.7];
    let image = [-0.13, 0.21];
    let affine = [
        dn * dx * map[0][0],
        dn * dx * map[0][1],
        dn * (map[0][0] * image[0] + map[0][1] * image[1]),
        dn * dx * map[1][0],
        dn * dx * map[1][1],
        dn * (map[1][0] * image[0] + map[1][1] * image[1]),
    ]
    .map(Df32::from_f64);
    let hnd = upload(b, data.clone(), vec![h, w]);
    let mut p = ResidentPlan::new(snapshot());
    let i = p.import(hnd.view());
    let o = p.scratch(BufferDesc::new(ElementType::Complex32, vec![oh, ow]));
    p.operations
        .push(ResidentOp::Transforms(Request::Transform(TransformPlan {
            kind: TransformKind::AffineMft,
            input: p.view(i).unwrap(),
            output: o,
            dimensions: vec![h, w],
            output_dimensions: vec![oh, ow],
            sign: 1,
            scale: 1.,
            affine,
        })));
    let output = p.scratch(BufferDesc::new(ElementType::Complex32, vec![oh, ow]));
    let linear = [
        source[0] * map[0][0] + source[1] * map[1][0],
        source[0] * map[0][1] + source[1] * map[1][1],
    ];
    let coefficients = [
        0.,
        0.,
        0.,
        dx * linear[0],
        dx * linear[1],
        image[0] * linear[0] + image[1] * linear[1],
    ]
    .map(Df32::from_f64);
    p.operations.push(ResidentOp::Transforms(Request::Phase {
        input: p.view(o).unwrap(),
        coefficients,
        output,
    }));
    p.outputs = vec![output];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let BufferData::Complex32(got) = b
        .readback(r.outputs[0].view(), &JobContext::new())
        .unwrap()
        .data
    else {
        panic!()
    };
    let mut want = vec![];
    for l in 0..oh {
        for k in 0..ow {
            let x = image[0] + dx * k as f64;
            let y = image[1] + dx * l as f64;
            let mut sum = [0.; 2];
            for j in 0..h {
                for i in 0..w {
                    let nu = source[0] + dn * i as f64;
                    let mu = source[1] + dn * j as f64;
                    let ph = std::f64::consts::TAU
                        * (nu * (map[0][0] * x + map[0][1] * y)
                            + mu * (map[1][0] * x + map[1][1] * y));
                    let (s, c) = ph.sin_cos();
                    let v = data[j * w + i];
                    sum[0] += v[0] as f64 * c - v[1] as f64 * s;
                    sum[1] += v[0] as f64 * s + v[1] as f64 * c;
                }
            }
            want.push(sum);
        }
    }
    compare(&got, &want, (w * h) as f64);
}
#[test]
fn cpu_physical_grid_mft() {
    physical_grid_mft(&CpuF32Backend::new());
}
