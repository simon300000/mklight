//! Off-axis regressions of the resident wave path: vignetted rims keep the
//! rays that survived the bisection, and elliptical, decentred, tilted and
//! tilted-chief pupils through the GPU transforms follow closed forms and the
//! f64 methods. Frequency-tiled direct correlation reproduces the untiled sums.
#![cfg(target_os = "macos")]
use mklight_core::{JobContext, Snapshot, model::RayAiming, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_wave::gpu::{self, ResidentPupil};
use mklight_wave::*;

const LAM: f64 = 0.55;

/// The host Metal backend; None (the test is skipped) without a device.
fn metal() -> Option<mklight_metal::MetalBackend> {
    let Some(b) = mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions {
        library: mklight_metal::LibraryChoice::Precompiled,
        ..Default::default()
    })
    .unwrap() else {
        eprintln!("no Metal device: skipped");
        return None;
    };
    assert!(b.precision().probe.error_free_transforms());
    Some(b)
}
fn ctx() -> JobContext {
    JobContext::new()
}
fn snapshot() -> Snapshot {
    Snapshot::new(samples::singlet()).unwrap()
}
fn magnitude(v: &[f64; 2]) -> f64 {
    v[0].hypot(v[1])
}
fn worst(a: impl IntoIterator<Item = f64>, b: impl IntoIterator<Item = f64>) -> f64 {
    a.into_iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0., f64::max)
}

/// Vignetted off-axis fields with real ray aiming. The reference keeps the
/// bracketing ray of every rim edge; the resident tracer must keep the same
/// ray from the round that traced it. Aiming the final point again from
/// other seeds converged within tolerance on the clipped side of an
/// aperture edge, dropping a sector of the rim (mask, coverage and clear
/// area) at every vignetted field.
#[test]
fn vignetted_rim_keeps_surviving_rays() {
    let ctx = ctx();
    let Some(b) = metal() else { return };
    let mut sys = samples::double_gauss();
    sys.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    // Clear apertures on the outer elements clip the oblique beams, so the
    // rim is traced against surface edges (not only the compressed disk).
    for (surface, radius) in [(1, 11.), (11, 11.)] {
        sys.surfaces[surface].aperture =
            Some(mklight_core::model::ApertureShape::Circular { radius });
    }
    let t = mklight_analysis::tracer_for(sys).unwrap();
    let service = gpu::PupilTracer::new(&b, &t, &ctx).unwrap();
    let settings = PupilSettings {
        grid: 128,
        ..Default::default()
    };
    let wl = t.system().primary_wavelength;
    for field in 1..t.fields().len() {
        let expected = trace_pupil(&t, t.field(field), wl, &settings, &ctx).unwrap();
        let r = service.trace(t.field(field), wl, &settings, &ctx).unwrap();
        let r = &r.value;
        let (Some(PupilMask::Polygons { loops: a }), Some(PupilMask::Polygons { loops: g })) =
            (&expected.mask, &r.mask)
        else {
            panic!("field {field}: both pupils carry a traced rim");
        };
        assert_eq!(a.len(), g.len(), "field {field} rim loops");
        let mut rim = 0f64;
        for (x, y) in a.iter().zip(g) {
            assert_eq!(x.len(), y.len(), "field {field} rim points");
            for (p, q) in x.iter().zip(y) {
                rim = rim.max((p[0] - q[0]).abs().max((p[1] - q[1]).abs()));
            }
        }
        let area = (r.info.clear_area - expected.info.clear_area).abs();
        let cov = b.read_view(r.pupil.coverage.view()).unwrap();
        let masked = cov
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .zip(expected.pupil.coverage())
            .filter(|(x, y)| (f64::from(**x) - **y).abs() > 1e-9)
            .count();
        eprintln!(
            "P6_RIM field={field} rim_points={} transmitted={} rim={rim:e} area={area:e} coverage_mismatches={masked}",
            r.info.rim_points, r.info.transmitted_fraction
        );
        assert!(r.info.rim_points > 0 && r.info.transmitted_fraction < 0.9);
        assert!(rim < 1e-6, "field {field}: rim point difference {rim:e}");
        assert!(area < 1e-6, "field {field}: clear area difference {area:e}");
        assert_eq!(masked, 0, "field {field}: resampled coverage");
    }
}

/// An elliptical pupil (semi-axes `a`, `b`) decentred by `(cu, cv)` with a
/// tilt of `tilt` waves per unit `v`: uniform amplitude inside, coverage by
/// 8×8 supersampling of each cell.
fn elliptical(
    n: usize,
    g: PupilGeometry,
    a: f64,
    b: f64,
    cu: f64,
    cv: f64,
    tilt: f64,
) -> PupilFunction {
    let h = 2. / n as f64;
    let mut amplitude = vec![0.; n * n];
    let mut coverage = vec![0.; n * n];
    let mut opd = vec![0.; n * n];
    for j in 0..n {
        for i in 0..n {
            let (u, v) = ((i as f64 + 0.5) * h - 1., (j as f64 + 0.5) * h - 1.);
            let mut inside = 0;
            for sj in 0..8 {
                for si in 0..8 {
                    let uu = u + ((si as f64 + 0.5) / 8. - 0.5) * h - cu;
                    let vv = v + ((sj as f64 + 0.5) / 8. - 0.5) * h - cv;
                    inside += usize::from((uu / a).powi(2) + (vv / b).powi(2) <= 1.);
                }
            }
            let k = j * n + i;
            coverage[k] = inside as f64 / 64.;
            amplitude[k] = f64::from(inside > 0);
            opd[k] = tilt * v;
        }
    }
    PupilFunction::from_samples(n, LAM, g, amplitude, coverage, opd).unwrap()
}

/// Closed-form MTF of a uniform elliptical aperture: the circular MTF at the
/// frequency scaled by the semi-axes (independent of decentring and tilt).
fn elliptical_mtf(lf: f64, a: f64, b: f64, f: [f64; 2]) -> f64 {
    diffraction_limited_mtf(
        ((f[0] * lf / a).powi(2) + (f[1] * lf / b).powi(2)).sqrt(),
        0.,
    )
}

/// Elliptical, decentred, tilted pupil (slow F/20 beam so obliquity factors
/// are below 3e-4): GPU FFT and direct MTF follow the closed form and the
/// f64 methods; the GPU PSF centroid follows the tilt `ε = −(λ/NA)·∂W/∂v`.
#[test]
fn elliptical_decentred_tilted_pupil() {
    let ctx = ctx();
    let Some(b) = metal() else { return };
    let g = PupilGeometry::from_working_f_number(20., 100., 1.).unwrap();
    let (a, bb, cu, cv, tilt) = (0.8, 0.5, 0.15, -0.1, 3.);
    let p = elliptical(256, g, a, bb, cu, cv, tilt);
    let lf = p.lambda_f_mm();
    let rp = ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
    let freqs: Vec<f64> = (0..25).map(|i| i as f64 * 0.03 / lf).collect();
    let o = MtfOrientation::default();
    let vectors: Vec<[f64; 2]> = [o.tangential, o.sagittal()]
        .iter()
        .flat_map(|v| freqs.iter().map(move |f| [v[0] * f, v[1] * f]))
        .collect();
    let expected: Vec<f64> = vectors
        .iter()
        .map(|f| elliptical_mtf(lf, a, bb, *f))
        .collect();
    let cpu_fft = mtf_fft(&p, &freqs, &o, &ctx).unwrap();
    let cpu_direct = mtf_direct(&p, &freqs, &o, &ctx).unwrap();
    let fft = rp.fft_otf(&b, snapshot(), &vectors, &ctx).unwrap().value;
    let direct = rp.otf(&b, snapshot(), &vectors, &ctx).unwrap().value;
    let n = freqs.len();
    let cpu = |c: &MtfCurve| {
        c.tangential
            .iter()
            .chain(&c.sagittal)
            .copied()
            .collect::<Vec<_>>()
    };
    let fft_vs_cpu = worst(fft.iter().map(magnitude), cpu(&cpu_fft));
    let direct_vs_cpu = worst(direct.iter().map(magnitude), cpu(&cpu_direct));
    let fft_vs_closed = worst(fft.iter().map(magnitude), expected.iter().copied());
    let direct_vs_closed = worst(direct.iter().map(magnitude), expected.iter().copied());
    eprintln!(
        "P6_ELLIPSE fft/cpu={fft_vs_cpu:e} direct/cpu={direct_vs_cpu:e} fft/closed={fft_vs_closed:e} direct/closed={direct_vs_closed:e}"
    );
    assert!(fft_vs_cpu < 2e-5, "{fft_vs_cpu:e}");
    assert!(direct_vs_cpu < 2e-5, "{direct_vs_cpu:e}");
    assert!(fft_vs_closed < 4e-3, "{fft_vs_closed:e}");
    assert!(direct_vs_closed < 4e-3, "{direct_vs_closed:e}");
    // Anamorphic cutoffs: 0.5 of the circular one along v (the last
    // tangential frequency, 0.72, is beyond it), 0.8 along u (0.9 of the
    // sagittal cutoff still transmits).
    assert!(magnitude(&fft[n - 1]) < 1e-3 && magnitude(&fft[2 * n - 1]) > 0.02);
    // The tilt moves the PSF centroid by −(λ/NA)·tilt along y; the decentred
    // aperture does not move it.
    let settings = PolyPsfSettings {
        grid_size: 512,
        spacing_mm: lf / 4.,
        normalization: PsfNormalization::UnitEnergy,
    };
    let reference = polychromatic_psf(&[(&p, 1.)], &settings, &ctx).unwrap();
    let psf = gpu::polychromatic_psf(&b, snapshot(), &[(&rp, 1.)], &settings, false, &ctx)
        .unwrap()
        .value
        .finish(&b, &ctx)
        .unwrap();
    let shift = -(LAM * 1e-3 / g.numerical_aperture()) * tilt;
    let (c, r) = (psf.centroid(), reference.centroid());
    eprintln!("P6_ELLIPSE centroid gpu={c:?} cpu={r:?} analytic=[0, {shift}]");
    assert!((c[0] - r[0]).abs() < 1e-6 && (c[1] - r[1]).abs() < 1e-6);
    assert!(c[0].abs() < 0.02 * shift.abs() && (c[1] - shift).abs() < 0.02 * shift.abs());
}

/// A circular pupil whose chief ray is tilted 20° about x: the angular
/// spectrum on the image surface keeps the sagittal cutoff and scales the
/// tangential one by cosθ. GPU FFT and direct MTF follow the closed form and
/// the f64 methods through the off-axis (no stencil) correlation path.
#[test]
fn tilted_chief_pupil() {
    let ctx = ctx();
    let Some(b) = metal() else { return };
    let theta = 20f64.to_radians();
    let g = PupilGeometry::from_working_f_number(20., 100., 1.)
        .unwrap()
        .with_chief_ray([0., theta.sin(), theta.cos()])
        .unwrap();
    assert!(!g.is_on_axis());
    let p = PupilFunction::diffraction_limited(256, LAM, g, Aperture::Circular, 8).unwrap();
    let lf = p.lambda_f_mm();
    let rp = ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
    let freqs: Vec<f64> = (0..25).map(|i| i as f64 * 0.03 / lf).collect();
    let o = MtfOrientation::from_field(0., 1.);
    let vectors: Vec<[f64; 2]> = [o.tangential, o.sagittal()]
        .iter()
        .flat_map(|v| freqs.iter().map(move |f| [v[0] * f, v[1] * f]))
        .collect();
    let n = freqs.len();
    let expected: Vec<f64> = vectors
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let scale = if i < n { 1. / theta.cos() } else { 1. };
            diffraction_limited_mtf(magnitude(f) * lf * scale, 0.)
        })
        .collect();
    let cpu_fft = mtf_fft(&p, &freqs, &o, &ctx).unwrap();
    let cpu_direct = mtf_direct(&p, &freqs, &o, &ctx).unwrap();
    let fft = rp.fft_otf(&b, snapshot(), &vectors, &ctx).unwrap().value;
    let direct = rp.otf(&b, snapshot(), &vectors, &ctx).unwrap().value;
    let cpu = |c: &MtfCurve| {
        c.tangential
            .iter()
            .chain(&c.sagittal)
            .copied()
            .collect::<Vec<_>>()
    };
    let fft_vs_cpu = worst(fft.iter().map(magnitude), cpu(&cpu_fft));
    let direct_vs_cpu = worst(direct.iter().map(magnitude), cpu(&cpu_direct));
    let fft_vs_closed = worst(fft.iter().map(magnitude), expected.iter().copied());
    let direct_vs_closed = worst(direct.iter().map(magnitude), expected.iter().copied());
    eprintln!(
        "P6_TILTED fft/cpu={fft_vs_cpu:e} direct/cpu={direct_vs_cpu:e} fft/closed={fft_vs_closed:e} direct/closed={direct_vs_closed:e}"
    );
    assert!(fft_vs_cpu < 2e-5, "{fft_vs_cpu:e}");
    assert!(direct_vs_cpu < 2e-5, "{direct_vs_cpu:e}");
    assert!(fft_vs_closed < 4e-3, "{fft_vs_closed:e}");
    assert!(direct_vs_closed < 4e-3, "{direct_vs_closed:e}");
}

/// Frequency-tiled direct correlation: with more frequencies than one
/// stencil may hold, the tiled plan reproduces the single-frequency plans
/// bit for bit on Metal, the CPU-f32 emulation concatenates the same tiles,
/// and an impossible explicit grid is rejected before any work.
#[test]
fn tiled_direct_correlation_matches_untiled() {
    let ctx = ctx();
    let Some(b) = metal() else { return };
    let n = 512;
    let tile = ResidentPupil::stencil_frequencies(n);
    assert!(tile > 1 && tile < 60, "tile {tile}");
    assert!(ResidentPupil::check_stencil_grid(1024).is_ok());
    assert!(matches!(
        ResidentPupil::check_stencil_grid(4096),
        Err(mklight_core::MkError::Unsupported(_))
    ));
    let g = PupilGeometry::from_working_f_number(4., 100., 1.).unwrap();
    let p = PupilFunction::diffraction_limited(n, LAM, g, Aperture::Circular, 4).unwrap();
    let lf = p.lambda_f_mm();
    let vectors: Vec<[f64; 2]> = (0..60)
        .map(|i| {
            let f = (i as f64 + 0.5) * 0.9 / (60. * lf);
            [f * (i as f64 * 0.1).cos(), f * (i as f64 * 0.1).sin()]
        })
        .collect();
    let rp = ResidentPupil::from_pupil(&b, &p, &ctx).unwrap();
    let tiled = rp.otf(&b, snapshot(), &vectors, &ctx).unwrap();
    assert!(tiled.value.len() == vectors.len());
    let mut singles = vec![];
    for f in &vectors {
        singles.push(rp.otf(&b, snapshot(), &[*f], &ctx).unwrap().value[0]);
    }
    assert_eq!(tiled.value, singles, "tiled sums differ from single plans");
    let closed = worst(
        tiled.value.iter().map(magnitude),
        vectors
            .iter()
            .map(|f| diffraction_limited_mtf(magnitude(f) * lf, 0.)),
    );
    eprintln!("P6_TILED tile={tile} closed={closed:e}");
    assert!(closed < 2e-3, "{closed:e}");
    let cpu = CpuF32Backend::matching(&b.precision().probe);
    let ep = ResidentPupil::from_pupil(&cpu, &p, &ctx).unwrap();
    let emulated = ep.otf(&cpu, snapshot(), &vectors, &ctx).unwrap().value;
    let difference = worst(
        emulated.iter().map(magnitude),
        tiled.value.iter().map(magnitude),
    );
    assert!(difference < 1e-5, "emulation {difference:e}");
}
