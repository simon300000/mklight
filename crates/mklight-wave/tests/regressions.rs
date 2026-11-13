//! Known-answer regression tests for review findings: off-lattice MTF of
//! displaced and aberrated PSFs, clear aperture of ray-lattice pupils,
//! energy on tilted planes, image-surface projection for off-axis chief
//! rays, FFT length rounding, input validation, serialisation, Seidel
//! extraction from high-order sets, polychromatic resampling accuracy and
//! polygonal pupil masks.

use mklight_core::{JobContext, MkError, Vec3, rng::Philox};
use mklight_wave::*;
use std::f64::consts::PI;

const LAM: f64 = 0.55;

fn ctx() -> JobContext {
    JobContext::new()
}

/// F/20, R = 100 mm: Debye factor within 3·10⁻⁴ of one.
fn slow() -> PupilGeometry {
    PupilGeometry::from_working_f_number(20.0, 100.0, 1.0).unwrap()
}

/// F/5, R = 50 mm.
fn fast() -> PupilGeometry {
    PupilGeometry::from_working_f_number(5.0, 50.0, 1.0).unwrap()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

// ------------------------------------------------- MTF between lattice points

/// A pure wavefront tilt only moves the PSF, so its MTF is the diffraction
/// limit at every frequency. Plain bilinear interpolation of the complex
/// OTF lost up to 0.27 here (8 waves of tilt, off-lattice frequencies).
#[test]
fn tilted_wavefront_mtf_is_diffraction_limited_between_lattice_points() {
    let g = slow();
    let n = 64;
    let nc = g.cutoff_frequency(LAM);
    for t in [1.0, 3.0, 8.0] {
        let p = PupilFunction::from_aperture(
            n,
            LAM,
            g,
            Aperture::Circular,
            Apodization::Uniform,
            8,
            |u, v| t * v + 0.3 * t * u,
        )
        .unwrap();
        let dnu = 1.0 / p.alias_free_field_mm();
        let freqs: Vec<f64> = (0..40)
            .map(|k| (k as f64 * 1.53 + 0.37) * dnu)
            .filter(|f| *f < nc)
            .collect();
        let c = mtf_fft(&p, &freqs, &MtfOrientation::default(), &ctx()).unwrap();
        let mut worst: f64 = 0.0;
        for (k, f) in freqs.iter().enumerate() {
            let dl = diffraction_limited_mtf(f / nc, 0.0);
            worst = worst
                .max((c.tangential[k] - dl).abs())
                .max((c.sagittal[k] - dl).abs());
        }
        assert!(worst < 2e-3, "tilt {t} waves: max |MTF − DL| = {worst:.2e}");
    }
}

/// Two waves of Fringe coma at n = 128, at frequencies halfway between the
/// lattice points, against the exact lattice OTF of an n = 512 pupil
/// (whose lattice contains these frequencies). Plain bilinear
/// interpolation was 2·10⁻² off here.
#[test]
fn comatic_mtf_between_lattice_points_matches_fine_reference() {
    let g = slow();
    let z = ZernikeSet::new(
        ZernikeOrdering::Fringe,
        vec![0., 0., 0., 0., 0., 0., 0., 2.0],
    )
    .unwrap();
    let mk = |n| {
        PupilFunction::from_zernike(n, LAM, g, Aperture::Circular, Apodization::Uniform, 8, &z)
            .unwrap()
    };
    let (p, fine) = (mk(128), mk(512));
    let nc = g.cutoff_frequency(LAM);
    let dnu = 1.0 / p.alias_free_field_mm();
    let freqs: Vec<f64> = (0..128)
        .map(|k| (k as f64 + 0.5) * dnu)
        .filter(|f| *f < nc)
        .collect();
    let o = MtfOrientation::default();
    let c = mtf_fft(&p, &freqs, &o, &ctx()).unwrap();
    let r = mtf_fft(&fine, &freqs, &o, &ctx()).unwrap();
    let wt = max_abs_diff(&c.tangential, &r.tangential);
    let ws = max_abs_diff(&c.sagittal, &r.sagittal);
    assert!(wt < 2e-3 && ws < 2e-3, "T {wt:.2e}, S {ws:.2e}");
    // The aberration matters: the MTF is far below the diffraction limit.
    let k = freqs.len() / 3;
    assert!(c.diffraction_limit[k] - c.tangential[k] > 0.1);
    // The direct autocorrelation (an independent discretisation) agrees.
    let d = mtf_direct(&p, &freqs, &o, &ctx()).unwrap();
    assert!(max_abs_diff(&d.tangential, &r.tangential) < 3e-3);
}

// ------------------------------------------------- structured ray lattices

fn lattice_rays(
    k: usize,
    keep: impl Fn(f64, f64) -> bool,
    w: impl Fn(f64, f64) -> f64,
) -> StructuredSamples {
    let mut samples = Vec::with_capacity(k * k);
    for j in 0..k {
        for i in 0..k {
            let (u, v) = (
                2.0 * i as f64 / (k - 1) as f64 - 1.0,
                2.0 * j as f64 / (k - 1) as f64 - 1.0,
            );
            samples.push(keep(u, v).then_some(RaySample {
                u,
                v,
                opd_waves: w(u, v),
                amplitude: 1.0,
            }));
        }
    }
    StructuredSamples {
        nx: k,
        ny: k,
        samples,
    }
}

/// Rays on a 32 × 32 lattice inside the unit circle: the triangles alone
/// end at the outermost rays (area 8 % low), the analytic mask restores
/// the full aperture and the diffraction-limited MTF, and edge cells are
/// filled by exact linear extrapolation.
#[test]
fn structured_lattice_with_mask_keeps_full_aperture() {
    let g = slow();
    let n = 128;
    let lin = |u: f64, v: f64| 0.3 * u - 0.2 * v + 0.1;
    let rays = lattice_rays(32, |u, v| u * u + v * v <= 1.0, lin);
    let area = |p: &PupilFunction| p.coverage().iter().sum::<f64>() * p.spacing().powi(2);

    let (bare, w) = grid_from_structured(&rays, n, LAM, g, None, 8, &ctx()).unwrap();
    assert!(w.is_empty());
    assert!(area(&bare) / PI < 0.95, "{}", area(&bare) / PI);

    let mask = PupilMask::from(Aperture::Circular);
    let (p, w) = grid_from_structured(&rays, n, LAM, g, Some(&mask), 8, &ctx()).unwrap();
    assert!(w.is_empty(), "{w:?}");
    assert!((area(&p) / PI - 1.0).abs() < 1e-3, "{}", area(&p) / PI);
    // Linear OPD everywhere, including full cells beyond the outermost rays.
    let mut beyond = 0;
    for j in 0..n {
        for i in 0..n {
            let q = j * n + i;
            if p.coverage()[q] == 1.0 {
                let (u, v) = (p.coord(i), p.coord(j));
                assert!((p.opd_waves()[q] - lin(u, v)).abs() < 1e-12);
                assert!((p.amplitude()[q] - 1.0).abs() < 1e-12);
                if bare.coverage()[q] == 0.0 {
                    beyond += 1;
                }
            }
        }
    }
    assert!(beyond > 100, "{beyond}");
    // The MTF (tilt and piston do not change it) is diffraction limited.
    let nc = g.cutoff_frequency(LAM);
    let freqs: Vec<f64> = (1..20).map(|k| 0.05 * k as f64 * nc).collect();
    let c = mtf_direct(&p, &freqs, &MtfOrientation::default(), &ctx()).unwrap();
    for (k, f) in freqs.iter().enumerate() {
        let dl = diffraction_limited_mtf(f / nc, 0.0);
        assert!(
            (c.tangential[k] - dl).abs() < 2e-3,
            "{f}: {} vs {dl}",
            c.tangential[k]
        );
        assert!((c.sagittal[k] - dl).abs() < 2e-3);
    }
}

/// Rays only over the lower half of a circular mask: the upper half is
/// extrapolated far beyond the ray spacing and is reported.
#[test]
fn structured_lattice_reports_sparse_rays() {
    let rays = lattice_rays(24, |u, v| u * u + v * v <= 1.0 && v < 0.0, |_, _| 0.0);
    let mask = PupilMask::from(Aperture::Circular);
    let (_, w) = grid_from_structured(&rays, 64, LAM, slow(), Some(&mask), 4, &ctx()).unwrap();
    assert!(
        w.iter()
            .any(|w| matches!(w, WaveWarning::SparseRays { max_gap, .. } if *max_gap > 0.9)),
        "{w:?}"
    );
}

// ------------------------------------------------- energy on tilted planes

/// On a plane tilted by θ the intensity integrates to 1/cosθ times the
/// flux; `total_energy` includes this, so the grid holds the same energy
/// fraction at any tilt and energy measures stay below 1. (Before, the
/// grid fraction was 1.97 at 60° and encircled energy reached 1.97.)
#[test]
fn tilted_plane_energy_is_normalised() {
    let p = PupilFunction::diffraction_limited(40, LAM, fast(), Aperture::Circular, 4).unwrap();
    let lf = p.lambda_f_mm();
    let mut fractions = Vec::new();
    for tilt in [0.0, 30.0, 60.0] {
        for norm in [PsfNormalization::Strehl, PsfNormalization::UnitEnergy] {
            let h = HuygensSettings {
                grid_size: 64,
                spacing_mm: 0.75 * lf,
                plane: ImagePlane::tilted(0.0, tilt, 0.0),
                normalization: norm,
            };
            let f = huygens_psf(&p, &h, &ctx()).unwrap();
            let e = f.energy_in_grid() / f.total_energy;
            assert!(e > 0.97 && e < 1.0, "tilt {tilt}: grid energy {e}");
            fractions.push(e);
            let ee = encircled_energy(&f, &[20.0 * lf], EnergyCenter::Reference, &ctx()).unwrap();
            assert!(ee[0] > 0.95 && ee[0] < 1.0, "tilt {tilt}: EE {}", ee[0]);
            let m = mtf_from_psf(
                &f,
                &[0.02 / lf, 0.1 / lf],
                &MtfOrientation::default(),
                &ctx(),
            )
            .unwrap();
            assert!(
                m.tangential
                    .iter()
                    .chain(&m.sagittal)
                    .all(|&t| t <= 1.0 + 1e-3)
            );
        }
    }
    // The truncated wing fraction barely depends on the tilt.
    let spread = fractions.iter().fold(0.0f64, |a, &b| a.max(b))
        - fractions.iter().fold(1.0f64, |a, &b| a.min(b));
    assert!(spread < 0.015, "{fractions:?}");
}

// ------------------------------------------------- off-axis chief rays

/// A chief ray at θ_c to the image-surface normal stretches the PSF by
/// 1/cosθ_c tangentially: MTF_T(ν) = DL(ν/(cosθ_c·ν_c)), MTF_S unchanged.
#[test]
fn off_axis_mtf_is_projected_onto_the_image_surface() {
    let theta: f64 = 30.0;
    let g = slow().with_chief_angle(theta, 90.0).unwrap();
    let p = PupilFunction::diffraction_limited(96, LAM, g, Aperture::Circular, 8).unwrap();
    let nc = g.cutoff_frequency(LAM);
    let ct = theta.to_radians().cos();
    let freqs: Vec<f64> = (1..12).map(|k| 0.08 * k as f64 * nc).collect();
    for method in [MtfMethod::Fft, MtfMethod::Direct] {
        let c = mtf(&p, &freqs, &MtfOrientation::default(), method, &ctx()).unwrap();
        for (k, f) in freqs.iter().enumerate() {
            let t = diffraction_limited_mtf(f / (ct * nc), 0.0);
            let s = diffraction_limited_mtf(f / nc, 0.0);
            assert!((c.tangential[k] - t).abs() < 2e-3, "{method:?} T {f}");
            assert!((c.sagittal[k] - s).abs() < 2e-3, "{method:?} S {f}");
            assert!((c.diffraction_limit[k] - t).abs() < 2e-3);
        }
    }
    // The PSF's first dark ring along y is at 1.22·λF#/cosθ_c on the surface.
    let lf = p.lambda_f_mm();
    let s = PsfSettings {
        grid_size: 3,
        sampling: ImageSampling::Spacing {
            mm: 1.2197 * lf / ct,
        },
        normalization: PsfNormalization::Strehl,
        transform: PsfTransform::Fft,
    };
    let f = fft_psf(&p, &s, &ctx()).unwrap();
    assert!(f.at(1, 2) < 1e-4, "{:?}", f.data);
    // Sagittally the same distance is 1.2197/cosθ_c λF# on the Airy profile.
    let airy = special::annular_airy(PI * 1.2197 / ct, 0.0);
    assert!((f.at(2, 1) - airy).abs() < 1e-4, "{} vs {airy}", f.at(2, 1));
}

/// `max |FFT − Huygens|` (Strehl units) on the image surface for a chief
/// ray at 25° (azimuth 60°: the non-separable projection) at F/# `f`, over
/// a `grid²` grid at spacing λF#/3, plus the same for the FFT PSF of the
/// pupil without the chief-ray tilt (no projection).
fn off_axis_psf_error(f_number: f64, grid: usize) -> (f64, f64, PupilFunction, PsfSettings) {
    let g = PupilGeometry::from_working_f_number(f_number, 10.0 * f_number, 1.0)
        .unwrap()
        .with_chief_angle(25.0, 60.0)
        .unwrap();
    let z = ZernikeSet::new(
        ZernikeOrdering::Fringe,
        vec![0.0, 0.0, 0.0, 0.05, 0.1, -0.05, 0.1],
    )
    .unwrap();
    let p =
        PupilFunction::from_zernike(40, LAM, g, Aperture::Circular, Apodization::Uniform, 4, &z)
            .unwrap();
    let dx = p.lambda_f_mm() / 3.0;
    let s = PsfSettings {
        grid_size: grid,
        sampling: ImageSampling::Spacing { mm: dx },
        normalization: PsfNormalization::Strehl,
        transform: PsfTransform::Fft,
    };
    let f = fft_psf(&p, &s, &ctx()).unwrap();
    assert!((f.spacing_mm - dx).abs() < 1e-15);
    let h = huygens_psf(
        &p,
        &HuygensSettings {
            grid_size: grid,
            spacing_mm: dx,
            plane: ImagePlane::image_surface(&g),
            normalization: PsfNormalization::Strehl,
        },
        &ctx(),
    )
    .unwrap();
    let on_axis = PupilFunction::from_zernike(
        40,
        LAM,
        PupilGeometry::from_working_f_number(f_number, 10.0 * f_number, 1.0).unwrap(),
        Aperture::Circular,
        Apodization::Uniform,
        4,
        &z,
    )
    .unwrap();
    let unprojected = fft_psf(&on_axis, &s, &ctx()).unwrap();
    (
        max_abs_diff(&f.data, &h.data),
        max_abs_diff(&unprojected.data, &h.data),
        p,
        s,
    )
}

/// The projected FFT PSF is the Huygens PSF on the image surface up to the
/// neglected focus change across the tilted surface, a phase of about
/// `|x|·sinθ_c·NA/4` waves at `|x|` λF# from the chief ray (module docs of
/// `pupil`), which halves from F/5 to F/10. Its energy normalisation
/// accounts for the 1/cosθ_c spread.
#[test]
fn off_axis_psf_matches_huygens_on_image_surface() {
    let (e5, raw5, p, s) = off_axis_psf_error(5.0, 24);
    let (e10, raw10, _, _) = off_axis_psf_error(10.0, 24);
    assert!(e5 < 3e-3, "F/5: {e5:.2e}");
    // The residual is the neglected focus term: proportional to NA.
    assert!(
        e10 / e5 > 0.4 && e10 / e5 < 0.6,
        "F/5 {e5:.2e}, F/10 {e10:.2e}"
    );
    // Without the projection the PSF is wrong by far more.
    assert!(
        raw5 > 20.0 * e5 && raw10 > 20.0 * e10,
        "{raw5:.2e} {raw10:.2e}"
    );
    // Energy: a large grid holds nearly all of it on the surface.
    let lf = p.lambda_f_mm();
    let big = fft_psf(
        &p,
        &PsfSettings {
            grid_size: 96,
            sampling: ImageSampling::Spacing { mm: 0.4 * lf },
            ..s
        },
        &ctx(),
    )
    .unwrap();
    let e = big.energy_in_grid() / big.total_energy;
    assert!(e > 0.97 && e < 1.0, "{e}");
    // `pupil_coords` puts the chief ray at the pupil centre and a marginal
    // ray at radius 1, on the side of the pupil the ray comes from.
    let g = p.geometry();
    let [eu, ev, ew] = g.chief_frame();
    let (u0, v0) = g.pupil_coords(ew);
    assert!(u0.abs() < 1e-15 && v0.abs() < 1e-15);
    let sm = g.sin_max();
    let k = (ew * (1.0 - sm * sm).sqrt() + (eu * 0.6 + ev * 0.8) * sm).normalized();
    let (u1, v1) = g.pupil_coords(k);
    assert!((u1 + 0.6).abs() < 1e-12 && (v1 + 0.8).abs() < 1e-12);
}

// ------------------------------------------------- FFT length rounding

/// A Nyquist request (Q = 2) is never rounded to a coarser spacing.
#[test]
fn nyquist_padding_never_rounds_coarser() {
    for n in [97, 101, 103, 107, 109, 113] {
        let p = PupilFunction::diffraction_limited(n, LAM, slow(), Aperture::Circular, 2).unwrap();
        let s = PsfSettings {
            grid_size: 64,
            sampling: ImageSampling::Padding { factor: 2.0 },
            normalization: PsfNormalization::Strehl,
            transform: PsfTransform::Fft,
        };
        let f = fft_psf(&p, &s, &ctx()).unwrap();
        assert!(
            f.spacing_mm <= 0.5 * p.lambda_f_mm() * (1.0 + 1e-12),
            "n = {n}"
        );
        assert!(
            !f.warnings
                .iter()
                .any(|w| matches!(w, WaveWarning::ImageUndersampled { .. })),
            "n = {n}: {:?}",
            f.warnings
        );
    }
}

// ------------------------------------------------- validation

#[test]
fn huygens_rejects_bad_spacing_and_planes() {
    let p = PupilFunction::diffraction_limited(16, LAM, fast(), Aperture::Circular, 1).unwrap();
    let base = HuygensSettings {
        grid_size: 4,
        spacing_mm: 1e-3,
        plane: ImagePlane::default(),
        normalization: PsfNormalization::Strehl,
    };
    assert!(huygens_psf(&p, &base, &ctx()).is_ok());
    for sp in [0.0, -1e-3, f64::NAN, f64::INFINITY] {
        let s = HuygensSettings {
            spacing_mm: sp,
            ..base
        };
        assert!(matches!(
            huygens_psf(&p, &s, &ctx()),
            Err(MkError::InvalidParam(_))
        ));
    }
    for (x, y) in [
        (Vec3::new(2.0, 0.0, 0.0), Vec3::Y),
        (Vec3::X, Vec3::new(0.6, 0.8, 0.0)),
        (Vec3::X, Vec3::new(f64::NAN, 1.0, 0.0)),
    ] {
        let s = HuygensSettings {
            plane: ImagePlane {
                origin_mm: Vec3::ZERO,
                x_axis: x,
                y_axis: y,
            },
            ..base
        };
        assert!(matches!(
            huygens_psf(&p, &s, &ctx()),
            Err(MkError::InvalidParam(_))
        ));
    }
}

#[test]
fn dark_pupil_is_rejected() {
    let r = PupilFunction::diffraction_limited(
        32,
        LAM,
        slow(),
        Aperture::Annular { obscuration: 1.0 },
        1,
    );
    assert!(matches!(r, Err(MkError::Numerical(_))), "{r:?}");
    let n = 8;
    let r = PupilFunction::from_samples(
        n,
        LAM,
        slow(),
        vec![1.0; n * n],
        vec![0.0; n * n],
        vec![0.0; n * n],
    );
    assert!(matches!(r, Err(MkError::Numerical(_))));
    let r = PupilFunction::from_samples(
        n,
        LAM,
        slow(),
        vec![0.0; n * n],
        vec![1.0; n * n],
        vec![0.0; n * n],
    );
    assert!(matches!(r, Err(MkError::Numerical(_))));
}

#[test]
fn deserialisation_validates() {
    let p = PupilFunction::diffraction_limited(4, LAM, slow(), Aperture::Circular, 2).unwrap();
    let json = serde_json::to_string(&p).unwrap();
    let back: PupilFunction = serde_json::from_str(&json).unwrap();
    assert_eq!(back, p);
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["n"] = 8.into();
    assert!(serde_json::from_value::<PupilFunction>(v).is_err());
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["coverage"][0] = 1.5.into();
    assert!(serde_json::from_value::<PupilFunction>(v).is_err());

    let z = ZernikeSet::new(ZernikeOrdering::Fringe, vec![0.1; 37]).unwrap();
    let json = serde_json::to_string(&z).unwrap();
    assert_eq!(serde_json::from_str::<ZernikeSet>(&json).unwrap(), z);
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["coefficients"].as_array_mut().unwrap().push(0.1.into());
    assert!(serde_json::from_value::<ZernikeSet>(v).is_err());

    let f = fft_psf(&p, &PsfSettings::default(), &ctx()).unwrap();
    let json = serde_json::to_string(&f).unwrap();
    assert_eq!(serde_json::from_str::<Psf>(&json).unwrap(), f);
    let mut v: serde_json::Value = serde_json::from_str(&json).unwrap();
    v["size"] = (f.size + 1).into();
    assert!(serde_json::from_value::<Psf>(v).is_err());
}

// ------------------------------------------------- Seidel from Standard sets

/// Seidel terms need only n ≤ 4; high-order Standard terms are ignored
/// (they used to make the whole conversion to Fringe fail).
#[test]
fn seidel_from_high_order_standard_set() {
    let mut c = vec![0.0; 45];
    c[3] = 0.2; // Noll 4: defocus √3·(2ρ² − 1)
    c[10] = 0.1; // Noll 11: spherical √5·(6ρ⁴ − 6ρ² + 1)
    c[43] = 0.01; // Noll 44: (n, m) = (8, ±8), no Fringe index
    let s = ZernikeSet::new(ZernikeOrdering::Standard, c).unwrap();
    let t = s.seidel().unwrap();
    // Fringe Z4 = 0.2·√3, Z9 = 0.1·√5: focus = 2Z4 − 6Z9, spherical = 6Z9.
    let (z4, z9) = (0.2 * 3f64.sqrt(), 0.1 * 5f64.sqrt());
    assert!((t.focus - (2.0 * z4 - 6.0 * z9)).abs() < 1e-14);
    assert!((t.spherical - 6.0 * z9).abs() < 1e-14);
    assert!(t.coma.abs() < 1e-15 && t.astigmatism.abs() < 1e-15);
    // Equals the Seidel terms of the same set truncated to n ≤ 4.
    let low = ZernikeSet::new(ZernikeOrdering::Standard, s.coefficients()[..15].to_vec()).unwrap();
    assert_eq!(low.seidel().unwrap(), t);
}

// ------------------------------------------------- polygonal pupil masks

/// Overlap of two unit circles whose centres are `d` apart.
fn lens_area(d: f64) -> f64 {
    2.0 * (d / 2.0).acos() - 0.5 * d * (4.0 - d * d).sqrt()
}

/// Cat's-eye (vignetted) pupil: two unit circles, centres at v = ±0.4.
fn cats_eye(u: f64, v: f64) -> bool {
    u * u + (v - 0.4).powi(2) <= 1.0 && u * u + (v + 0.4).powi(2) <= 1.0
}

fn cats_eye_polygon(k: usize) -> PupilMask {
    // Lens boundary: two arcs meeting at (±√(1 − 0.16), 0).
    let a = (0.4f64).acos(); // half-angle of each arc about its centre
    let mut pts = Vec::new();
    for i in 0..k {
        let t = -PI / 2.0 - a + 2.0 * a * i as f64 / k as f64; // lower arc of the upper circle
        pts.push([t.cos(), 0.4 + t.sin()]);
    }
    for i in 0..k {
        let t = PI / 2.0 - a + 2.0 * a * i as f64 / k as f64; // upper arc of the lower circle
        pts.push([t.cos(), -0.4 + t.sin()]);
    }
    PupilMask::polygon(pts)
}

#[test]
fn scattered_rays_on_a_vignetted_cats_eye_pupil() {
    let mask = cats_eye_polygon(2000);
    let mut rng = Philox::new(7, 0);
    let truth = |u: f64, v: f64| 0.2 * (u * u + v * v) - 0.1 * v * (u * u + v * v) + 0.05 * u;
    let mut rays = Vec::new();
    while rays.len() < 800 {
        let (u, v) = (2.0 * rng.uniform() - 1.0, 2.0 * rng.uniform() - 1.0);
        if cats_eye(u, v) {
            rays.push(RaySample {
                u,
                v,
                opd_waves: truth(u, v),
                amplitude: 1.0,
            });
        }
    }
    let basis = ZernikeBasis::standard_to_order(3);
    let n = 96;
    let (p, fit) =
        grid_from_scattered_zernike(&rays, &basis, n, LAM, slow(), &mask, 8, &ctx()).unwrap();
    assert!(fit.residual_rms < 1e-12);
    assert!(
        !fit.warnings
            .iter()
            .any(|w| matches!(w, WaveWarning::SparseRays { .. }))
    );
    let area = p.coverage().iter().sum::<f64>() * p.spacing().powi(2);
    assert!((area / lens_area(0.8) - 1.0).abs() < 2e-3, "{area}");
    for j in 0..n {
        for i in 0..n {
            let (u, v) = (p.coord(i), p.coord(j));
            if p.coverage()[j * n + i] == 1.0 {
                assert!((p.opd_waves()[j * n + i] - truth(u, v)).abs() < 1e-10);
            }
        }
    }
    // Rays over only the lower half of the mask: the fit extrapolates into
    // the upper half, which is reported.
    let half: Vec<RaySample> = rays.iter().copied().filter(|r| r.v < -0.1).collect();
    let (_, fit) =
        grid_from_scattered_zernike(&half, &basis, 48, LAM, slow(), &mask, 2, &ctx()).unwrap();
    assert!(
        fit.warnings
            .iter()
            .any(|w| matches!(w, WaveWarning::SparseRays { .. }))
    );
    // Convex hull of the rays: a rough mask, slightly inside the true rim.
    let hull =
        PupilMask::convex_hull(&rays.iter().map(|r| (r.u, r.v)).collect::<Vec<_>>()).unwrap();
    let inside = rays.iter().filter(|r| hull.contains(r.u, r.v)).count();
    assert!(inside >= rays.len() - 40);
}

#[test]
fn squeezed_lattice_rim_is_not_a_missing_pupil_region() {
    let mut rays = lattice_rays(65, |u, v| u * u + v * v <= 1.0, |u, v| 0.3 * u - 0.2 * v);
    for r in rays.samples.iter_mut().flatten() {
        r.u *= 0.25;
    }
    let mask = PupilMask::polygon(
        (0..512)
            .map(|i| {
                let a = 2.0 * PI * i as f64 / 512.0;
                [0.25 * a.cos(), a.sin()]
            })
            .collect(),
    );
    let (p, warnings) =
        grid_from_structured(&rays, 64, LAM, slow(), Some(&mask), 4, &ctx()).unwrap();
    assert!(warnings.is_empty(), "{warnings:?}");
    let area = p.coverage().iter().sum::<f64>() * p.spacing().powi(2);
    assert!((area / (0.25 * PI) - 1.0).abs() < 0.003);
    for k in 0..p.n() * p.n() {
        if p.coverage()[k] > 0.999 {
            let (u, v) = (p.coord(k % p.n()), p.coord(k / p.n()));
            assert!((p.opd_waves()[k] - (1.2 * u - 0.2 * v)).abs() < 1e-12);
        }
    }
}

/// The lattice interpolation estimator: zero for piston and tilt, `A·h²/4`
/// for a quadratic `A·(u² + v²)` on spacing `h` (the error of linear
/// interpolation inside a cell), independent of uneven ray spacing.
#[test]
fn lattice_interpolation_error_known_answers() {
    let k = 33;
    let h = 2.0 / (k - 1) as f64;
    let disk = |u: f64, v: f64| u * u + v * v <= 1.0;
    let tilt = lattice_rays(k, disk, |u, v| 0.3 + 5.0 * u - 2.0 * v);
    assert!(lattice_interpolation_error(&tilt) < 1e-12);
    assert!(lattice_warning(&tilt).is_none());
    let a = 3.0;
    let quad = lattice_rays(k, disk, |u, v| a * (u * u + v * v));
    let e = lattice_interpolation_error(&quad);
    assert!((e - a * h * h / 4.0).abs() < 1e-12, "{e}");
    // A distorted lattice (u ↦ u + 0.2u³) carrying a wavefront that is
    // linear in the distorted coordinates: no interpolation error.
    let mut warped = lattice_rays(k, disk, |_, _| 0.0);
    for r in warped.samples.iter_mut().flatten() {
        r.u += 0.2 * r.u.powi(3);
        r.opd_waves = 4.0 * r.u + r.v;
    }
    assert!(lattice_interpolation_error(&warped) < 1e-12);
    // Above the limit: reported with a recommendation that brings the
    // estimate below it (the lattice has a rim node on each side).
    let steep = lattice_rays(k, disk, |u, v| 40.0 * (u * u + v * v).powi(3));
    match lattice_warning(&steep) {
        Some(WaveWarning::LatticeUndersampled {
            interpolation_error_waves,
            rays,
            recommended_rays,
        }) => {
            assert!(interpolation_error_waves > LATTICE_INTERPOLATION_LIMIT_WAVES);
            assert_eq!(rays, k - 2);
            assert!(recommended_rays > rays && recommended_rays % 2 == 1);
        }
        w => panic!("{w:?}"),
    }
}
