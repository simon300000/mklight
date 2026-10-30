//! Known-answer tests against closed-form diffraction results.

use mklight_core::{JobContext, MkError, rng::Philox};
use mklight_wave::special::{airy_encircled_energy, annular_airy};
use mklight_wave::*;
use std::f64::consts::PI;

const LAM: f64 = 0.55;

/// Slow beam (F/20): the Debye factor is within 3·10⁻⁴ of one, so the
/// uniform-pupil closed forms apply.
fn slow() -> PupilGeometry {
    PupilGeometry::from_working_f_number(20.0, 100.0, 1.0).unwrap()
}

fn ctx() -> JobContext {
    JobContext::new()
}

fn clear(n: usize) -> PupilFunction {
    PupilFunction::diffraction_limited(n, LAM, slow(), Aperture::Circular, 8).unwrap()
}

fn psf(p: &PupilFunction, grid: usize, q: f64, norm: PsfNormalization) -> Psf {
    let s = PsfSettings {
        grid_size: grid,
        sampling: ImageSampling::Padding { factor: q },
        normalization: norm,
        transform: PsfTransform::Fft,
    };
    fft_psf(p, &s, &ctx()).unwrap()
}

// ---------------------------------------------------------------- Airy

#[test]
fn clear_pupil_psf_is_airy_pattern() {
    let p = clear(128);
    let lf = p.lambda_f_mm();
    let f = psf(&p, 64, 4.0, PsfNormalization::Strehl);
    assert!(f.warnings.is_empty(), "{:?}", f.warnings);
    assert!((f.spacing_mm - lf / 4.0).abs() < 1e-15);
    let c = f.center_index();
    // Peak = 1 at the reference point (diffraction limited ⇒ Strehl 1).
    assert!((f.at(c, c) - 1.0).abs() < 1e-12);
    assert!((f.strehl() - 1.0).abs() < 1e-12);
    // Profiles along x, y and the diagonal follow [2J₁(v)/v]², v = πr/(λF#).
    let mut worst: f64 = 0.0;
    for k in 0..30 {
        for (i, j) in [
            (c + k, c),
            (c, c + k),
            (c - k, c),
            (c + k * 7 / 10, c + k * 7 / 10),
        ] {
            let r = f.x(i).hypot(f.y(j));
            let a = annular_airy(PI * r / lf, 0.0);
            worst = worst.max((f.at(i, j) - a).abs());
        }
    }
    assert!(worst < 2e-4, "max deviation from Airy {worst:.2e}");
    // Unit-energy normalisation: peak = π/(4λ²F#²).
    let u = psf(&p, 64, 4.0, PsfNormalization::UnitEnergy);
    let peak = PI / (4.0 * lf * lf);
    assert!((u.at(c, c) / peak - 1.0).abs() < 1e-3);
    assert!((u.total_energy - 1.0).abs() < 1e-15);
}

#[test]
fn airy_first_zero_at_1_22_lambda_f() {
    let p = clear(128);
    let lf = p.lambda_f_mm();
    // Exact matrix transform on a fine grid: 0.005 λF# spacing.
    let s = PsfSettings {
        grid_size: 600,
        sampling: ImageSampling::Spacing { mm: 0.005 * lf },
        normalization: PsfNormalization::Strehl,
        transform: PsfTransform::Matrix,
    };
    let f = fft_psf(&p, &s, &ctx()).unwrap();
    let c = f.center_index();
    let (mut best, mut at) = (f64::INFINITY, 0.0);
    for i in c..f.size {
        let r = f.x(i) / lf;
        if (1.0..1.5).contains(&r) && f.at(i, c) < best {
            best = f.at(i, c);
            at = r;
        }
    }
    assert!((at - 1.2197).abs() < 0.006, "first zero at {at} λF#");
    assert!(best < 1e-4);
}

#[test]
fn airy_pattern_encircled_energy() {
    let p = clear(128);
    let lf = p.lambda_f_mm();
    let f = psf(&p, 128, 4.0, PsfNormalization::Strehl);
    let radii: Vec<f64> = (1..=24).map(|k| k as f64 * 0.125 * lf).collect();
    let ee = encircled_energy(&f, &radii, EnergyCenter::Reference, &ctx()).unwrap();
    for (r, e) in radii.iter().zip(&ee) {
        let a = airy_encircled_energy(PI * r / lf);
        assert!((e - a).abs() < 5e-4, "EE({}) = {e}, Airy {a}", r / lf);
    }
    // 83.8 % inside the first dark ring.
    let e1 = encircled_energy(&f, &[1.2197 * lf], EnergyCenter::Centroid, &ctx()).unwrap()[0];
    assert!((e1 - 0.8378).abs() < 2e-4, "{e1}");
    // Ensquared energy grows monotonically and the square of half-width h
    // lies between the inscribed and circumscribed circles.
    let h = [0.5 * lf, 1.0 * lf, 2.0 * lf];
    let es = ensquared_energy(&f, &h, EnergyCenter::Reference, &ctx()).unwrap();
    let ein = encircled_energy(&f, &h, EnergyCenter::Reference, &ctx()).unwrap();
    let hd: Vec<f64> = h.iter().map(|x| x * 2f64.sqrt()).collect();
    let eout = encircled_energy(&f, &hd, EnergyCenter::Reference, &ctx()).unwrap();
    for k in 0..3 {
        assert!(ein[k] < es[k] && es[k] < eout[k]);
    }
}

// ---------------------------------------------------------------- MTF

#[test]
fn diffraction_limited_mtf_matches_closed_form() {
    let p = clear(128);
    let nuc = p.geometry().cutoff_frequency(LAM);
    let freqs: Vec<f64> = (0..=50).map(|k| k as f64 / 50.0 * nuc).collect();
    let o = MtfOrientation::default();
    for m in [MtfMethod::Fft, MtfMethod::Direct] {
        let c = mtf(&p, &freqs, &o, m, &ctx()).unwrap();
        assert!((c.cutoff - nuc).abs() < 1e-9);
        for (k, f) in freqs.iter().enumerate() {
            let x = f / nuc;
            let phi = x.min(1.0).acos();
            let a = (2.0 / PI) * (phi - phi.cos() * phi.sin());
            assert!(
                (c.tangential[k] - a).abs() < 1e-3,
                "{m:?} T at {x}: {} vs {a}",
                c.tangential[k]
            );
            assert!((c.sagittal[k] - a).abs() < 1e-3, "{m:?} S at {x}");
            assert!((c.diffraction_limit[k] - a).abs() < 1e-3);
            assert!((diffraction_limited_mtf(x, 0.0) - a).abs() < 1e-14);
        }
    }
    // Also through the PSF: over the full alias-free field (n·Q = 512
    // samples) the DFT of the samples is exact at the lattice frequencies
    // m/(n·λF#).
    let f = psf(&p, 512, 4.0, PsfNormalization::Strehl);
    let lattice: Vec<f64> = (0..=32)
        .map(|m| 4.0 * m as f64 / p.alias_free_field_mm())
        .collect();
    let c = mtf_from_psf(&f, &lattice, &o, &ctx()).unwrap();
    for (k, x) in lattice.iter().enumerate() {
        let a = diffraction_limited_mtf(x / nuc, 0.0);
        assert!(
            (c.tangential[k] - a).abs() < 1e-3,
            "from PSF at {}",
            x / nuc
        );
    }
    assert!(c.warnings.is_empty());
}

#[test]
fn square_aperture_mtf_and_line_spread() {
    // Every cell of a 64² grid lies inside the square: hard, exact edges.
    // F/200 makes the Debye factor 1 within 10⁻⁵, so the PSF is separable.
    let n = 64;
    let g = PupilGeometry::from_working_f_number(200.0, 1000.0, 1.0).unwrap();
    let p = PupilFunction::diffraction_limited(
        n,
        LAM,
        g,
        Aperture::Rectangular {
            half_u: 1.0,
            half_v: 1.0,
        },
        1,
    )
    .unwrap();
    let lf = p.lambda_f_mm();
    let nuc = 1.0 / lf;
    // Along an axis the MTF of a square is the triangle 1 − ν/ν_c.
    let freqs: Vec<f64> = (0..=20).map(|k| k as f64 / 20.0 * nuc).collect();
    let c = mtf_fft(&p, &freqs, &MtfOrientation::default(), &ctx()).unwrap();
    for (k, f) in freqs.iter().enumerate() {
        let a = 1.0 - f / nuc;
        assert!((c.tangential[k] - a).abs() < 3e-4 && (c.sagittal[k] - a).abs() < 3e-4);
    }
    // PSF over the full alias-free field (Q = 4 ⇒ 256 samples): the LSF is
    // the squared Dirichlet kernel D²(x)/(n·L), → sinc²(x/λF#)/λF#.
    let f = psf(&p, 4 * n, 4.0, PsfNormalization::UnitEnergy);
    let lsf = line_spread_function(&f, LsfAxis::X);
    let l = p.alias_free_field_mm();
    let total: f64 = lsf.values.iter().sum::<f64>() * f.spacing_mm;
    assert!((total - 1.0).abs() < 1e-12);
    for (x, v) in lsf.positions_mm.iter().zip(&lsf.values) {
        let t = PI * x / l;
        let d = if t.abs() < 1e-12 {
            n as f64
        } else {
            (n as f64 * t).sin() / t.sin()
        };
        assert!((v - d * d / (n as f64 * l)).abs() < 3e-5 / lf);
        if x.abs() < 5.0 * lf {
            let s = if x.abs() < 1e-15 {
                1.0
            } else {
                (PI * x / lf).sin() / (PI * x / lf)
            };
            assert!((v * lf - s * s).abs() < 2e-3);
        }
    }
    // The LSF along x transforms to the sagittal MTF (exactly, at the
    // lattice frequencies m/L of the full-field grid).
    let tau = 2.0 * PI;
    let lattice: Vec<f64> = (0..48).map(|m| m as f64 / l).collect();
    let c = mtf_fft(&p, &lattice, &MtfOrientation::default(), &ctx()).unwrap();
    for (k, nu) in lattice.iter().enumerate() {
        let (mut re, mut im) = (0.0, 0.0);
        for (x, v) in lsf.positions_mm.iter().zip(&lsf.values) {
            re += v * (tau * nu * x).cos() * f.spacing_mm;
            im -= v * (tau * nu * x).sin() * f.spacing_mm;
        }
        assert!((re.hypot(im) - c.sagittal[k]).abs() < 1e-9);
    }
}

#[test]
fn annular_pupil_known_results() {
    let eps = 0.4;
    let ann = PupilFunction::diffraction_limited(
        128,
        LAM,
        slow(),
        Aperture::Annular { obscuration: eps },
        8,
    )
    .unwrap();
    let lf = ann.lambda_f_mm();
    let f = psf(&ann, 64, 4.0, PsfNormalization::Strehl);
    let c = f.center_index();
    for k in 0..30 {
        let r = f.x(c + k);
        let a = annular_airy(PI * r / lf, eps);
        assert!(
            (f.at(c + k, c) - a).abs() < 1e-3,
            "annular PSF at {} λF#",
            r / lf
        );
    }
    // MTF: overlap of shifted annuli.
    let nuc = 1.0 / lf;
    let freqs: Vec<f64> = (0..=40).map(|k| k as f64 / 40.0 * nuc).collect();
    let m = mtf_fft(&ann, &freqs, &MtfOrientation::default(), &ctx()).unwrap();
    for (k, x) in freqs.iter().enumerate() {
        let a = diffraction_limited_mtf(x / nuc, eps);
        assert!(
            (m.tangential[k] - a).abs() < 1e-3,
            "annular MTF at {}",
            x / nuc
        );
    }
    // Known features: mid-frequency dip below and high-frequency boost
    // above the clear-aperture MTF.
    assert!(diffraction_limited_mtf(0.3, eps) < diffraction_limited_mtf(0.3, 0.0));
    assert!(diffraction_limited_mtf(0.8, eps) > diffraction_limited_mtf(0.8, 0.0));
    // Same outer diameter: the unit-energy peak scales with the open area.
    let full = psf(&clear(128), 16, 4.0, PsfNormalization::UnitEnergy);
    let annu = psf(&ann, 16, 4.0, PsfNormalization::UnitEnergy);
    let ratio = annu.at(8, 8) / full.at(8, 8);
    assert!((ratio - (1.0 - eps * eps)).abs() < 2e-3, "{ratio}");
    // Energy inside the first zero of the clear Airy pattern drops.
    let ea = encircled_energy(&f, &[1.22 * lf], EnergyCenter::Reference, &ctx()).unwrap()[0];
    assert!(ea < 0.8378 - 0.05);
}

// ---------------------------------------------------------------- Strehl

#[test]
fn defocus_strehl_versus_marechal() {
    let p = clear(128);
    for w020 in [0.02, 0.05, 0.1] {
        let d = p.with_defocus_waves(w020);
        // Exact for a uniform disk: |∫ e^{i2πW020ρ²} 2ρdρ|² = sinc²(πW020).
        let x = PI * w020;
        let exact = (x.sin() / x).powi(2);
        let s_ref = d.strehl_at_reference();
        assert!(
            (s_ref - exact).abs() < 2e-5,
            "W020 = {w020}: {s_ref} vs {exact}"
        );
        // RMS of W020·ρ² is W020/√12 (piston removed; tilt removal changes
        // nothing for a centred defocus).
        let st = d.opd_stats(true);
        assert!((st.rms - w020 / 12f64.sqrt()).abs() < 1e-4 * w020.max(0.01));
        // Maréchal agrees for small aberrations.
        let mar = strehl_marechal(st.rms);
        assert!((mar - exact).abs() < 5e-3 * w020 / 0.1, "{mar} vs {exact}");
        assert!(strehl_marechal_classic(st.rms) <= mar);
        // PSF peak at the reference point equals the pupil Strehl.
        let f = psf(&d, 32, 4.0, PsfNormalization::Strehl);
        let c = f.center_index();
        assert!((f.at(c, c) - s_ref).abs() < 1e-10);
        let rep = strehl_report(&d, Some(&f));
        assert!((rep.psf_peak.unwrap() - s_ref).abs() < 1e-10);
    }
}

// ---------------------------------------------------------------- Energy

#[test]
fn parseval_energy_conservation() {
    // Hard mask, aberrated: Σ PSF·Δx² over the full alias-free field equals
    // the total energy exactly.
    let z = ZernikeSet::new(
        ZernikeOrdering::Fringe,
        vec![0.0, 0.1, 0.0, 0.3, 0.2, -0.1, 0.25, 0.0, 0.15],
    )
    .unwrap();
    let p = PupilFunction::from_zernike(
        64,
        LAM,
        slow(),
        Aperture::Circular,
        Apodization::Gaussian {
            edge_intensity: 0.3,
        },
        1,
        &z,
    )
    .unwrap();
    for q in [2.0, 3.0, 4.0] {
        let m = (q * 64.0) as usize;
        for norm in [PsfNormalization::Strehl, PsfNormalization::UnitEnergy] {
            let f = psf(&p, m, q, norm);
            assert!((f.energy_in_grid() / f.total_energy - 1.0).abs() < 1e-12);
        }
    }
    // Grey edges: the grid holds Σ|p|²/E of the energy (the rest is the
    // suppressed far wings).
    let g = PupilFunction::from_zernike(
        64,
        LAM,
        slow(),
        Aperture::Circular,
        Apodization::Uniform,
        8,
        &z,
    )
    .unwrap();
    let f = psf(&g, 256, 4.0, PsfNormalization::UnitEnergy);
    let w = g.field_weights();
    let lattice: f64 = w.iter().map(|x| x * x).sum();
    assert!((f.energy_in_grid() - lattice / g.energy()).abs() < 1e-12);
    assert!(f.energy_in_grid() < 1.0 && f.energy_in_grid() > 0.99);
}

// ---------------------------------------------------------------- Huygens

fn fast_geom() -> PupilGeometry {
    PupilGeometry::from_working_f_number(5.0, 50.0, 1.0).unwrap()
}

#[test]
fn fft_and_huygens_psf_agree() {
    let z = ZernikeSet::new(
        ZernikeOrdering::Fringe,
        vec![0.0, 0.0, 0.0, 0.1, 0.15, 0.0, 0.2, -0.1],
    )
    .unwrap();
    let n = 48;
    let p = PupilFunction::from_zernike(
        n,
        LAM,
        fast_geom(),
        Aperture::Circular,
        Apodization::Uniform,
        4,
        &z,
    )
    .unwrap();
    let lf = p.lambda_f_mm();
    let grid = 24;
    let h = HuygensSettings {
        grid_size: grid,
        spacing_mm: lf / 4.0,
        plane: ImagePlane::default(),
        normalization: PsfNormalization::Strehl,
    };
    let hp = huygens_psf(&p, &h, &ctx()).unwrap();
    for transform in [PsfTransform::Fft, PsfTransform::Matrix] {
        let s = PsfSettings {
            grid_size: grid,
            sampling: ImageSampling::Spacing { mm: lf / 4.0 },
            normalization: PsfNormalization::Strehl,
            transform,
        };
        let fp = fft_psf(&p, &s, &ctx()).unwrap();
        assert!((fp.spacing_mm - hp.spacing_mm).abs() < 1e-15);
        let worst = fp
            .data
            .iter()
            .zip(&hp.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(
            worst < 1e-5,
            "{transform:?}: max |FFT − Huygens| = {worst:.2e}"
        );
    }
    // Defocused plane: Huygens at z = dz ≡ FFT of the refocused pupil.
    let dz = 2.0 * LAM * 1e-3 * 0.75 / 0.01; // W020 ≈ 0.75 wave at NA 0.1
    let hd = huygens_psf(
        &p,
        &HuygensSettings {
            plane: ImagePlane::defocused(dz),
            ..h
        },
        &ctx(),
    )
    .unwrap();
    let s = PsfSettings {
        grid_size: grid,
        sampling: ImageSampling::Spacing { mm: lf / 4.0 },
        normalization: PsfNormalization::Strehl,
        transform: PsfTransform::Matrix,
    };
    let fd = fft_psf(&p.with_focus_shift(dz), &s, &ctx()).unwrap();
    let worst = fd
        .data
        .iter()
        .zip(&hd.data)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    // Residual is the Debye approximation itself: off focus the true field
    // is magnified by ~dz/R (here 1.6·10⁻³) relative to the Debye one.
    assert!(worst < 1e-3, "defocused: {worst:.2e}");
    let peak_change = (fd.peak().0 - fp_peak(&p, &s)).abs();
    assert!(peak_change > 0.1, "defocus must matter");
}

fn fp_peak(p: &PupilFunction, s: &PsfSettings) -> f64 {
    fft_psf(p, s, &JobContext::new()).unwrap().peak().0
}

#[test]
fn huygens_on_tilted_plane_samples_the_3d_psf() {
    let n = 40;
    let p = PupilFunction::diffraction_limited(n, LAM, fast_geom(), Aperture::Circular, 4).unwrap();
    let lf = p.lambda_f_mm();
    let tilt = 60.0f64;
    let h = HuygensSettings {
        grid_size: 17,
        spacing_mm: 0.75 * lf,
        plane: ImagePlane::tilted(0.0, tilt, 0.0),
        normalization: PsfNormalization::Strehl,
    };
    let hp = huygens_psf(&p, &h, &ctx()).unwrap();
    let c = hp.center_index();
    // In-plane x = s maps to (s·cos t, 0, −s·sin t): radius s·cos t in the
    // plane z = −s·sin t, where the rotationally symmetric PSF is that of
    // the refocused pupil.
    for k in [-6i64, -3, 2, 5] {
        let sx = k as f64 * h.spacing_mm;
        let (r, z) = (sx * tilt.to_radians().cos(), -sx * tilt.to_radians().sin());
        // A 3-sample matrix-transform grid with spacing |r| has its sample
        // (2, 1) at x = |r| (the PSF is symmetric).
        let s = PsfSettings {
            grid_size: 3,
            sampling: ImageSampling::Spacing { mm: r.abs() },
            normalization: PsfNormalization::Strehl,
            transform: PsfTransform::Matrix,
        };
        let v = fft_psf(&p.with_focus_shift(z), &s, &ctx())
            .unwrap()
            .at(2, 1);
        let hv = hp.at((c as i64 + k) as usize, c);
        assert!((v - hv).abs() < 1e-4, "k = {k}: {v} vs {hv}");
    }
}

#[test]
fn opd_tilt_moves_image_toward_minus_y() {
    // ε_y = −(λ/NA)·∂W/∂v = −2λF#·t for W = t·v.
    let t = 1.5;
    let p = PupilFunction::from_aperture(
        64,
        LAM,
        fast_geom(),
        Aperture::Circular,
        Apodization::Uniform,
        4,
        |_, v| t * v,
    )
    .unwrap();
    let lf = p.lambda_f_mm();
    let f = psf(&p, 64, 4.0, PsfNormalization::Strehl);
    let (_, i, j) = f.peak();
    assert!((f.x(i)).abs() < 1e-12);
    assert!(
        (f.y(j) + 2.0 * t * lf).abs() < 0.26 * lf,
        "peak at y = {} λF#",
        f.y(j) / lf
    );
    let cy = f.centroid()[1];
    assert!((cy + 2.0 * t * lf).abs() < 0.05 * lf);
    let h = HuygensSettings {
        grid_size: 64,
        spacing_mm: lf / 4.0,
        plane: ImagePlane::default(),
        normalization: PsfNormalization::Strehl,
    };
    let hp = huygens_psf(
        &p.with_added_opd(|_, _| 0.0),
        &HuygensSettings { grid_size: 33, ..h },
        &ctx(),
    )
    .unwrap();
    let (_, _, hj) = hp.peak();
    assert!((hp.y(hj) - f.y(j)).abs() < 1e-12);
}

// ---------------------------------------------------------------- polychromatic

#[test]
fn polychromatic_psf_and_mtf() {
    let g = slow();
    let (l1, l2) = (0.48, 0.65);
    let p1 = PupilFunction::diffraction_limited(96, l1, g, Aperture::Circular, 8).unwrap();
    let mut p2 = PupilFunction::diffraction_limited(96, l2, g, Aperture::Circular, 8).unwrap();
    let dx = 0.25 * g.lambda_f_mm(l1);
    let mk = |t| PolyPsfSettings {
        grid_size: 48,
        spacing_mm: dx,
        normalization: PsfNormalization::UnitEnergy,
        transform: t,
    };
    let pupils = [(&p1, 1.0), (&p2, 3.0)];
    let a = polychromatic_psf(&pupils, &mk(PsfTransform::Matrix), &ctx()).unwrap();
    let b = polychromatic_psf(&pupils, &mk(PsfTransform::Fft), &ctx()).unwrap();
    let pk = a.peak().0;
    let worst = a
        .data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 2e-3 * pk,
        "matrix vs FFT-resampled {:.2e}",
        worst / pk
    );
    // Peak (unit energy) is the weighted sum of the Airy peaks.
    let airy = |l: f64| PI / (4.0 * g.lambda_f_mm(l).powi(2));
    let expect = 0.25 * airy(l1) + 0.75 * airy(l2);
    assert!((a.at(24, 24) / expect - 1.0).abs() < 1e-3);
    assert!((a.diffraction_peak / expect - 1.0).abs() < 1e-3);
    let s = polychromatic_psf(
        &pupils,
        &PolyPsfSettings {
            normalization: PsfNormalization::Strehl,
            ..mk(PsfTransform::Matrix)
        },
        &ctx(),
    )
    .unwrap();
    assert!((s.strehl() - 1.0).abs() < 1e-3);

    // MTF: weighted mean of the monochromatic diffraction limits.
    let freqs: Vec<f64> = (0..10)
        .map(|k| k as f64 * 0.1 * g.cutoff_frequency(l2))
        .collect();
    let o = MtfOrientation::default();
    let m = polychromatic_mtf(&pupils, &freqs, &o, MtfMethod::Fft, &ctx()).unwrap();
    for (k, f) in freqs.iter().enumerate() {
        let e = 0.25 * diffraction_limited_mtf(f / g.cutoff_frequency(l1), 0.0)
            + 0.75 * diffraction_limited_mtf(f / g.cutoff_frequency(l2), 0.0);
        assert!((m.tangential[k] - e).abs() < 1e-3);
    }
    // Lateral colour: shifting one colour by δ along y multiplies its OTF
    // by e^{−2πiνδ} in the tangential direction only.
    let delta = 8.0 * dx; // a whole number of grid rows
    p2.set_image_offset_mm([0.0, delta]);
    let pupils = [(&p1, 1.0), (&p2, 3.0)];
    let m = polychromatic_mtf(&pupils, &freqs, &o, MtfMethod::Fft, &ctx()).unwrap();
    for (k, f) in freqs.iter().enumerate() {
        let (a1, a2) = (
            0.25 * diffraction_limited_mtf(f / g.cutoff_frequency(l1), 0.0),
            0.75 * diffraction_limited_mtf(f / g.cutoff_frequency(l2), 0.0),
        );
        let ph = 2.0 * PI * f * delta;
        let t = (a1 * a1 + a2 * a2 + 2.0 * a1 * a2 * ph.cos()).sqrt();
        assert!((m.tangential[k] - t).abs() < 1e-3, "T {k}");
        assert!((m.sagittal[k] - (a1 + a2)).abs() < 1e-3, "S {k}");
    }
    // The PSF is the weighted sum of the individually shifted colours.
    let shifted = polychromatic_psf(&pupils, &mk(PsfTransform::Matrix), &ctx()).unwrap();
    let one = |p: &PupilFunction| {
        let mut q = p.clone();
        q.set_image_offset_mm([0.0; 2]);
        polychromatic_psf(&[(&q, 1.0)], &mk(PsfTransform::Matrix), &ctx()).unwrap()
    };
    let (a1, a2) = (one(&p1), one(&p2));
    for j in 8..48 {
        for i in 0..48 {
            let e = 0.25 * a1.at(i, j) + 0.75 * a2.at(i, j - 8);
            assert!((shifted.at(i, j) - e).abs() < 1e-12 * pk);
        }
    }
    assert!(shifted.centroid()[1] > 0.5 * delta);
}

// ---------------------------------------------------------------- focus and orientation

#[test]
fn through_focus_is_symmetric_and_peaks_in_focus() {
    let p =
        PupilFunction::diffraction_limited(64, LAM, fast_geom(), Aperture::Circular, 4).unwrap();
    let nu = [0.3 * p.geometry().cutoff_frequency(LAM)];
    let dzs: Vec<Defocus> = [-0.04, -0.02, 0.0, 0.02, 0.04]
        .iter()
        .map(|&mm| Defocus::ShiftMm { mm })
        .collect();
    let curves = through_focus_mtf(
        &p,
        &dzs,
        &nu,
        &MtfOrientation::default(),
        MtfMethod::Fft,
        &ctx(),
    )
    .unwrap();
    let v: Vec<f64> = curves.iter().map(|c| c.tangential[0]).collect();
    // ±dz give conjugate wavefronts (identical MTF) up to the exact
    // reference-sphere term dz²·sin²θ/(2R) ≈ 3·10⁻⁴ wave at the edge.
    assert!(
        (v[0] - v[4]).abs() < 1e-3 && (v[1] - v[3]).abs() < 1e-3,
        "{v:?}"
    );
    assert!(v[2] > v[1] && v[1] > v[0]);
    assert!((v[2] - diffraction_limited_mtf(0.3, 0.0)).abs() < 2e-3);
    // Equivalent defocus in waves gives the same curve.
    let w = p.focus_shift_opd(0.02, 1.0, 0.0);
    let cw = through_focus_mtf(
        &p,
        &[Defocus::Waves { waves: w }],
        &nu,
        &MtfOrientation::default(),
        MtfMethod::Fft,
        &ctx(),
    )
    .unwrap();
    assert!((cw[0].tangential[0] - v[3]).abs() < 2e-3);
}

#[test]
fn astigmatism_swaps_tangential_and_sagittal() {
    let mk = |a: f64| {
        let z = ZernikeSet::new(ZernikeOrdering::Fringe, vec![0.0, 0.0, 0.0, 0.0, a]).unwrap();
        PupilFunction::from_zernike(
            64,
            LAM,
            slow(),
            Aperture::Circular,
            Apodization::Uniform,
            4,
            &z,
        )
        .unwrap()
    };
    let nuc = slow().cutoff_frequency(LAM);
    let freqs: Vec<f64> = (1..8).map(|k| k as f64 * 0.1 * nuc).collect();
    let o = MtfOrientation::default();
    let (pos, neg) = (mk(0.4), mk(-0.4));
    for m in [MtfMethod::Fft, MtfMethod::Direct] {
        let a = mtf(&pos, &freqs, &o, m, &ctx()).unwrap();
        let b = mtf(&neg, &freqs, &o, m, &ctx()).unwrap();
        for k in 0..freqs.len() {
            assert!((a.tangential[k] - b.sagittal[k]).abs() < 1e-12);
            assert!((a.sagittal[k] - b.tangential[k]).abs() < 1e-12);
            assert!(a.tangential[k] < a.diffraction_limit[k]);
        }
    }
    // Field orientation: a field on +x makes x the tangential direction.
    let of = MtfOrientation::from_field(1.0, 0.0);
    assert_eq!(of.tangential, [1.0, 0.0]);
    assert_eq!(of.sagittal(), [0.0, -1.0]);
    let a = mtf(&pos, &freqs, &o, MtfMethod::Fft, &ctx()).unwrap();
    let r = mtf(&pos, &freqs, &of, MtfMethod::Fft, &ctx()).unwrap();
    for k in 0..freqs.len() {
        assert!((a.tangential[k] - r.sagittal[k]).abs() < 1e-12);
    }
}

#[test]
fn direct_and_fft_mtf_agree_with_aberrations() {
    let z = ZernikeSet::new(
        ZernikeOrdering::Fringe,
        vec![0.0, 0.0, 0.0, 0.2, 0.1, 0.0, 0.3, 0.1, -0.15],
    )
    .unwrap();
    let p = PupilFunction::from_zernike(
        96,
        LAM,
        slow(),
        Aperture::Circular,
        Apodization::Uniform,
        4,
        &z,
    )
    .unwrap();
    let dnu = 1.0 / p.alias_free_field_mm();
    // Lattice frequencies: both are the exact discrete autocorrelation.
    let lattice: Vec<f64> = (0..40).map(|k| k as f64 * 4.0 * dnu).collect();
    let o = MtfOrientation::from_field(0.6, 0.8);
    let a = mtf(
        &p,
        &lattice,
        &MtfOrientation::default(),
        MtfMethod::Fft,
        &ctx(),
    )
    .unwrap();
    let b = mtf(
        &p,
        &lattice,
        &MtfOrientation::default(),
        MtfMethod::Direct,
        &ctx(),
    )
    .unwrap();
    for k in 0..lattice.len() {
        assert!((a.tangential[k] - b.tangential[k]).abs() < 1e-10);
        assert!((a.sagittal[k] - b.sagittal[k]).abs() < 1e-10);
    }
    // Off-lattice frequencies and a rotated orientation: interpolation only.
    let freqs: Vec<f64> = (0..30).map(|k| (k as f64 * 3.3 + 0.37) * dnu).collect();
    let a = mtf(&p, &freqs, &o, MtfMethod::Fft, &ctx()).unwrap();
    let b = mtf(&p, &freqs, &o, MtfMethod::Direct, &ctx()).unwrap();
    for k in 0..freqs.len() {
        assert!((a.tangential[k] - b.tangential[k]).abs() < 2e-3);
        assert!((a.sagittal[k] - b.sagittal[k]).abs() < 2e-3);
    }
}

// ---------------------------------------------------------------- sampling diagnostics

#[test]
fn sampling_warnings_trigger() {
    let p = clear(64);
    let has = |f: &Psf, pred: fn(&WaveWarning) -> bool| f.warnings.iter().any(pred);
    let ok = psf(&p, 64, 2.0, PsfNormalization::Strehl);
    assert!(ok.warnings.is_empty(), "{:?}", ok.warnings);
    let under = psf(&p, 64, 1.5, PsfNormalization::Strehl);
    assert!(has(&under, |w| matches!(
        w,
        WaveWarning::ImageUndersampled { .. }
    )));
    // Q = 2 ⇒ the alias-free field holds 128 samples; ask for 200.
    let wrap = psf(&p, 200, 2.0, PsfNormalization::Strehl);
    assert!(has(&wrap, |w| matches!(w, WaveWarning::FieldWraps { .. })));
    // A spacing the FFT cannot hit exactly is rounded and reported.
    let s = PsfSettings {
        grid_size: 32,
        sampling: ImageSampling::Spacing {
            mm: p.lambda_f_mm() / 3.37,
        },
        ..Default::default()
    };
    let adj = fft_psf(&p, &s, &ctx()).unwrap();
    assert!(has(&adj, |w| matches!(
        w,
        WaveWarning::SpacingAdjusted { .. }
    )));
    // Coarser than λF#: falls back to the exact matrix transform.
    let s = PsfSettings {
        grid_size: 16,
        sampling: ImageSampling::Spacing {
            mm: 1.3 * p.lambda_f_mm(),
        },
        ..Default::default()
    };
    let coarse = fft_psf(&p, &s, &ctx()).unwrap();
    assert!((coarse.spacing_mm - 1.3 * p.lambda_f_mm()).abs() < 1e-15);
    assert!(has(&coarse, |w| matches!(
        w,
        WaveWarning::ImageUndersampled { .. }
    )));
    // Strong defocus on a coarse pupil grid.
    let d = p.with_defocus_waves(12.0);
    let f = psf(&d, 64, 4.0, PsfNormalization::Strehl);
    assert!(has(&f, |w| matches!(
        w,
        WaveWarning::PupilUndersampled { .. }
    )));
    // MTF from a PSF beyond its Nyquist frequency.
    let c = mtf_from_psf(
        &ok,
        &[0.9 / ok.spacing_mm],
        &MtfOrientation::default(),
        &ctx(),
    )
    .unwrap();
    assert!(
        c.warnings
            .iter()
            .any(|w| matches!(w, WaveWarning::FrequencyBeyondSampling { .. }))
    );
    // Display text is informative.
    assert!(under.warnings[0].to_string().contains("Nyquist"));
}

// ---------------------------------------------------------------- jobs

#[test]
fn deterministic_across_thread_counts() {
    let z = ZernikeSet::new(
        ZernikeOrdering::Fringe,
        vec![0.0, 0.0, 0.0, 0.3, 0.2, 0.0, 0.25],
    )
    .unwrap();
    let p = PupilFunction::from_zernike(
        64,
        LAM,
        slow(),
        Aperture::Circular,
        Apodization::Uniform,
        4,
        &z,
    )
    .unwrap();
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| {
            let f = psf(&p, 64, 4.0, PsfNormalization::Strehl);
            let ee = encircled_energy(&f, &[0.002, 0.004], EnergyCenter::Centroid, &ctx()).unwrap();
            let m = mtf(
                &p,
                &[50.0, 100.0],
                &MtfOrientation::default(),
                MtfMethod::Direct,
                &ctx(),
            )
            .unwrap();
            (f.data, ee, m.tangential)
        })
    };
    assert_eq!(run(1), run(5));
}

#[test]
fn cancellation_and_progress() {
    let p = clear(64);
    let c = JobContext::new();
    let f = fft_psf(&p, &PsfSettings::default(), &c).unwrap();
    assert!((c.progress.fraction() - 1.0).abs() < 1e-12);
    c.cancel.cancel();
    assert_eq!(
        fft_psf(&p, &PsfSettings::default(), &c).unwrap_err(),
        MkError::Cancelled
    );
    let h = HuygensSettings {
        grid_size: 8,
        spacing_mm: 0.001,
        plane: ImagePlane::default(),
        normalization: PsfNormalization::Strehl,
    };
    assert_eq!(huygens_psf(&p, &h, &c).unwrap_err(), MkError::Cancelled);
    assert_eq!(
        mtf(
            &p,
            &[10.0],
            &MtfOrientation::default(),
            MtfMethod::Direct,
            &c
        )
        .unwrap_err(),
        MkError::Cancelled
    );
    assert_eq!(
        encircled_energy(&f, &[0.001], EnergyCenter::Reference, &c).unwrap_err(),
        MkError::Cancelled
    );
}

// ---------------------------------------------------------------- resampling from rays

#[test]
fn structured_ray_samples_resample_linear_fields_exactly() {
    // A 41² entrance-pupil lattice mapped to the exit pupil with barrel-type
    // pupil distortion; rays outside the unit disk are vignetted.
    let k = 41;
    let lin = |u: f64, v: f64| 0.3 + 0.8 * u - 0.5 * v;
    let mut samples = Vec::new();
    for j in 0..k {
        for i in 0..k {
            let (x, y) = (
                2.0 * i as f64 / (k - 1) as f64 - 1.0,
                2.0 * j as f64 / (k - 1) as f64 - 1.0,
            );
            let r2 = x * x + y * y;
            if r2 > 1.0 {
                samples.push(None);
                continue;
            }
            let s = 1.0 - 0.05 * r2;
            let (u, v) = (x * s, y * s);
            samples.push(Some(RaySample {
                u,
                v,
                opd_waves: lin(u, v),
                amplitude: 1.0,
            }));
        }
    }
    let rays = StructuredSamples {
        nx: k,
        ny: k,
        samples,
    };
    let p = grid_from_structured(&rays, 64, LAM, slow(), 4, &ctx()).unwrap();
    let n = p.n();
    let mut full = 0;
    for j in 0..n {
        for i in 0..n {
            let q = j * n + i;
            if p.coverage()[q] == 1.0 {
                full += 1;
                assert!((p.opd_waves()[q] - lin(p.coord(i), p.coord(j))).abs() < 1e-12);
                assert!((p.amplitude()[q] - 1.0).abs() < 1e-12);
            }
        }
    }
    assert!(full > 2000);
    // Clear area: the lattice polygon inscribed in the distorted disk of
    // radius 0.95 (lattice step 0.05).
    let area: f64 = p.coverage().iter().sum::<f64>() * p.spacing().powi(2);
    assert!(
        area < PI * 0.95f64.powi(2) && area > PI * 0.9f64.powi(2),
        "{area}"
    );
}

#[test]
fn scattered_ray_samples_via_zernike_fit() {
    let truth = ZernikeSet::new(
        ZernikeOrdering::Standard,
        (0..15).map(|j| 0.05 * (j as f64 - 7.0)).collect(),
    )
    .unwrap();
    let mut rng = Philox::new(99, 0);
    let rays: Vec<RaySample> = (0..600)
        .map(|_| {
            let r = rng.uniform().sqrt();
            let t = 2.0 * PI * rng.uniform();
            let (u, v) = (r * t.cos(), r * t.sin());
            RaySample {
                u,
                v,
                opd_waves: truth.eval(u, v),
                amplitude: 1.0,
            }
        })
        .collect();
    let basis = ZernikeBasis::standard_to_order(4);
    let (p, fit) = grid_from_scattered_zernike(
        &rays,
        &basis,
        48,
        LAM,
        slow(),
        Aperture::Circular,
        1,
        &ctx(),
    )
    .unwrap();
    assert!(fit.residual_rms < 1e-12);
    for (a, b) in fit.set.coefficients.iter().zip(&truth.coefficients) {
        assert!((a - b).abs() < 1e-11);
    }
    let n = p.n();
    for j in 0..n {
        for i in 0..n {
            if p.coverage()[j * n + i] > 0.0 {
                let w = truth.eval(p.coord(i), p.coord(j));
                assert!((p.opd_waves()[j * n + i] - w).abs() < 1e-10);
            }
        }
    }
    // The grid's wavefront RMS matches the analytic Zernike RMS.
    let rms = p.opd_stats(false).rms;
    assert!((rms - truth.rms(false)).abs() < 0.02 * truth.rms(false));
}
