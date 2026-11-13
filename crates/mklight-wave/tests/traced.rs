//! The tracer link (`mklight_wave::traced`) and the wave analyses against
//! closed forms, the reference tracer's OPD, Seidel sums, geometric
//! analyses and each other.

use std::f64::consts::PI;

use mklight_analysis::seidel::{SeidelSettings, seidel};
use mklight_analysis::spot::{SpotDiagramSettings, spot_diagram};
use mklight_analysis::tracer_for;
use mklight_analysis::util::{FieldSelection, WavelengthSelection};
use mklight_core::{
    ApertureSpec, Field, FieldKind, FieldSet, Interaction, JobContext, MaterialRef, OpticalSystem,
    SemiDiameter, Surface, SurfaceShape, Wavelength, WavelengthSet, samples,
};
use mklight_trace::SequentialTracer;
use mklight_wave::analysis::*;
use mklight_wave::mtf::diffraction_limited_mtf;
use mklight_wave::*;

const LAM: f64 = 0.55;

fn ctx() -> JobContext {
    JobContext::new()
}

fn mono(um: f64) -> WavelengthSet {
    WavelengthSet {
        wavelengths: vec![Wavelength { um, weight: 1.0 }],
        primary: 0,
    }
}

fn surf(id: u64, shape: SurfaceShape, t: f64) -> Surface {
    Surface::new(id, shape, t, MaterialRef::Air)
}

/// Object at infinity, `body` (ids from 1), image; angle fields.
fn system(epd: f64, body: Vec<Surface>, stop: usize, fields: &[f64]) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: epd };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: fields.iter().map(|&y| Field::new(0.0, y)).collect(),
    };
    s.wavelengths = mono(LAM);
    let n = body.len() as u64;
    s.surfaces = vec![surf(0, SurfaceShape::Plane, f64::INFINITY)];
    s.surfaces.extend(body);
    s.surfaces.push(surf(n + 1, SurfaceShape::Plane, 0.0));
    s.stop = stop;
    s
}

/// Ideal lens (the stop) of focal length `f`; image `dz` beyond focus.
fn ideal(f: f64, epd: f64, dz: f64, fields: &[f64]) -> SequentialTracer {
    tracer_for(system(
        epd,
        vec![surf(1, SurfaceShape::Paraxial { focal_length: f }, f + dz)],
        1,
        fields,
    ))
    .unwrap()
}

/// Stop in the front focal plane of an ideal lens: image-space telecentric.
fn telecentric(f: f64, epd: f64, dz: f64, fields: &[f64]) -> SequentialTracer {
    tracer_for(system(
        epd,
        vec![
            surf(1, SurfaceShape::Plane, f),
            surf(2, SurfaceShape::Paraxial { focal_length: f }, f + dz),
        ],
        1,
        fields,
    ))
    .unwrap()
}

/// Concave paraboloid (the stop) of focal length `f`: the light reaches the
/// image toward −z. Image `dz` beyond focus along the propagation.
fn paraboloid(f: f64, epd: f64, dz: f64, fields: &[f64]) -> SequentialTracer {
    let mut m = surf(
        1,
        SurfaceShape::Conic {
            curvature: -0.5 / f,
            conic: -1.0,
        },
        -(f + dz),
    );
    m.interaction = Interaction::Reflect;
    tracer_for(system(epd, vec![m], 1, fields)).unwrap()
}

fn pupil(t: &SequentialTracer, field: usize, s: &PupilSettings) -> TracedPupil {
    trace_pupil(t, t.field(field), t.system().primary_wavelength, s, &ctx()).unwrap()
}

fn uniform() -> PupilSettings {
    PupilSettings {
        amplitude: AmplitudeModel::Uniform,
        ..Default::default()
    }
}

/// Entrance-pupil coordinates of lattice sample `k` (see `traced`).
fn lattice_point(rays: usize, k: usize) -> [f64; 2] {
    let side = rays + 2;
    let h = 2.0 / (rays - 1) as f64;
    let c = 0.5 * (side - 1) as f64;
    [((k % side) as f64 - c) * h, ((k / side) as f64 - c) * h]
}

// ----------------------------------------------------- exactness of the OPD

/// With the chief-ray reference the OPD of every lattice ray equals the
/// reference tracer's `opd` (the same reference sphere) to round-off, on
/// refracting, reflecting, finite-conjugate and folded systems.
#[test]
fn lattice_opd_equals_reference_tracer_opd() {
    for (name, sys, field) in [
        ("triplet", samples::cooke_triplet(), 2),
        ("newtonian", samples::newtonian_telescope(), 2),
        ("relay", samples::finite_relay(), 1),
        ("petzval", samples::petzval_asphere(), 2),
    ] {
        let t = tracer_for(sys).unwrap();
        let s = PupilSettings {
            rays: 24,
            ..uniform()
        };
        let wl = 0;
        let tp = trace_pupil(&t, t.field(field), wl, &s, &ctx()).unwrap();
        let mut n = 0;
        for (k, r) in tp.samples.samples.iter().enumerate() {
            let Some(r) = r else { continue };
            let p = lattice_point(s.rays, k);
            let o = t.opd(t.field(field), wl, p[0], p[1], &t.options()).unwrap();
            assert!(
                (o - r.opd_waves).abs() < 1e-9 * (1.0 + o.abs()),
                "{name} {p:?}: {o} vs {}",
                r.opd_waves
            );
            n += 1;
        }
        assert!(n > 300, "{name}: {n} rays");
    }
}

// ------------------------------------------------- diffraction-limited systems

fn check_diffraction_limited(t: &SequentialTracer, label: &str, mtf_tol: f64) {
    let tp = pupil(t, 0, &uniform());
    let p = &tp.pupil;
    let st = p.opd_stats(true);
    assert!(
        p.strehl_at_reference() > 0.999,
        "{label}: {}",
        p.strehl_at_reference()
    );
    assert!(st.rms < 0.005, "{label}: rms {}", st.rms);
    // Full circular pupil at the scale: area π up to the inscribed rim.
    assert!((tp.info.clear_area / PI - 1.0).abs() < 5e-4, "{label}");
    assert!((tp.info.transmitted_fraction - 1.0).abs() < 5e-4, "{label}");
    // MTF = analytic incoherent MTF of a circular pupil at the working F/#.
    let n = p.geometry().working_f_number();
    let nu_c = 1.0 / (LAM * 1e-3 * n);
    let freqs: Vec<f64> = (0..=20).map(|k| k as f64 * 0.05 * nu_c).collect();
    let c = mtf_fft(p, &freqs, &MtfOrientation::default(), &ctx()).unwrap();
    for (k, &f) in freqs.iter().enumerate() {
        let a = diffraction_limited_mtf(f / nu_c, 0.0);
        assert!(
            (c.tangential[k] - a).abs() < mtf_tol && (c.sagittal[k] - a).abs() < mtf_tol,
            "{label} ν = {f}: {} / {} vs {a}",
            c.tangential[k],
            c.sagittal[k]
        );
    }
    // Airy first zero at 1.22 λN (FFT PSF, finely sampled, bicubic).
    let lf = LAM * 1e-3 * n;
    let psf = fft_psf(
        p,
        &PsfSettings {
            grid_size: 64,
            sampling: ImageSampling::Spacing { mm: lf / 16.0 },
            normalization: PsfNormalization::Strehl,
            transform: PsfTransform::Matrix,
        },
        &ctx(),
    )
    .unwrap();
    let (mut best, mut rmin) = (f64::INFINITY, 0.0);
    for k in 0..=2000 {
        let r = lf * (1.0 + 0.4 * k as f64 / 2000.0);
        let v = psf.interpolate(r, 0.0) + psf.interpolate(0.0, r);
        if v < best {
            best = v;
            rmin = r;
        }
    }
    assert!(
        (rmin / lf - 1.2197).abs() < 0.005,
        "{label}: first zero at {} λN",
        rmin / lf
    );
    assert!(psf.strehl() > 0.999, "{label}");
}

#[test]
fn diffraction_limited_ideal_lens_paraboloid_and_achromat() {
    check_diffraction_limited(&ideal(100.0, 20.0, 0.0, &[0.0]), "ideal lens", 2e-3);
    check_diffraction_limited(&paraboloid(250.0, 50.0, 0.0, &[0.0]), "paraboloid", 2e-3);
    // The achromat stopped to F/10 and the singlet to F/20 at d light.
    let mut a = samples::achromatic_doublet();
    a.aperture = ApertureSpec::ImageSpaceFNumber { value: 10.0 };
    a.wavelengths = mono(0.5875618);
    let ta = tracer_for(a).unwrap();
    check_diffraction_limited_at(&ta, "achromat F/10");
    let mut s = samples::singlet();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 5.0 };
    s.wavelengths = mono(0.5875618);
    check_diffraction_limited_at(&tracer_for(s).unwrap(), "singlet F/20");
}

/// Real lenses at their own wavelength: Strehl ≈ 1 and MTF within the
/// small aberration of the design.
fn check_diffraction_limited_at(t: &SequentialTracer, label: &str) {
    let tp = pupil(t, 0, &uniform());
    let p = &tp.pupil;
    let lam = p.wavelength_um();
    assert!(p.opd_stats(true).rms < 0.01, "{label}");
    let s = fft_psf(p, &PsfSettings::default(), &ctx()).unwrap();
    assert!(s.strehl() > 0.99, "{label}: Strehl {}", s.strehl());
    let n = p.geometry().working_f_number();
    // The pupil scale agrees with the paraxial working F/#.
    let nw = t.paraxial().primary().working_f_number;
    assert!((n / nw - 1.0).abs() < 0.01, "{label}: {n} vs {nw}");
    let nu_c = 1.0 / (lam * 1e-3 * n);
    let freqs: Vec<f64> = (0..=10).map(|k| k as f64 * 0.1 * nu_c).collect();
    let c = mtf_fft(p, &freqs, &MtfOrientation::default(), &ctx()).unwrap();
    for (k, &f) in freqs.iter().enumerate() {
        let a = diffraction_limited_mtf(f / nu_c, 0.0);
        assert!((c.tangential[k] - a).abs() < 0.01, "{label} {f}");
    }
}

// --------------------------------------------------------------- defocus

/// Exact OPD of a perfect wave converging to a focus `dz` before the
/// reference point, on the sphere of radius `r` about it, at pupil radius
/// `rho` (normalised, scale `s`): the ray angle θ solves
/// `t(θ)·sinθ/R = ρ·s` with `t = −dz cosθ + √(R² − dz² sin²θ)`, and
/// `OPD = n(t − t_c)`, `t_c = R − dz` (mm).
fn defocus_opd_exact(dz: f64, r: f64, s: f64, rho: f64) -> f64 {
    let t = |th: f64| -dz * th.cos() + (r * r - (dz * th.sin()).powi(2)).sqrt();
    let target = rho * s;
    let (mut lo, mut hi) = (0.0, 0.5 * PI);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if t(mid) * mid.sin() / r < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    t(0.5 * (lo + hi)) - (r - dz)
}

#[test]
fn defocus_known_answer_and_zernike() {
    let (f, epd, dz) = (100.0, 20.0, 0.05);
    let t = ideal(f, epd, dz, &[0.0]);
    let s = PupilSettings {
        rays: 48,
        ..uniform()
    };
    let tp = pupil(&t, 0, &s);
    let r = tp.info.reference_radius_mm;
    assert!((r - (f + dz)).abs() < 1e-9, "R = {r}");
    let sm = tp.info.sin_max;
    let lam_mm = LAM * 1e-3;
    // Every lattice ray against the closed form.
    let mut edge: f64 = 0.0;
    for rs in tp.samples.samples.iter().flatten() {
        let rho = rs.u.hypot(rs.v);
        let exact = defocus_opd_exact(dz, r, sm, rho) / lam_mm;
        assert!(
            (rs.opd_waves - exact).abs() < 1e-8,
            "ρ {rho}: {} vs {exact}",
            rs.opd_waves
        );
        edge = edge.max(rs.opd_waves);
    }
    // Image beyond focus: the wave converges more than the reference
    // sphere, so the OPD is positive at the edge (CONVENTIONS §4).
    let w020 = dz * sm * sm / (2.0 * lam_mm);
    assert!(edge > 0.95 * w020 && edge < 1.05 * w020, "{edge} vs {w020}");
    // The grid (triangle interpolation) to second order in the spacing.
    let p = &tp.pupil;
    for k in 0..p.n() * p.n() {
        if p.coverage()[k] == 1.0 {
            let rho = p.coord(k % p.n()).hypot(p.coord(k / p.n()));
            let exact = defocus_opd_exact(dz, r, sm, rho) / lam_mm;
            assert!((p.opd_waves()[k] - exact).abs() < 2e-3, "{rho}");
        }
    }
    // Zernike: W020 = n dz NA²/(2λ) in the focus term, nothing else.
    let res = wavefront_map(
        &t,
        &WavefrontSettings {
            pupil: s,
            ..Default::default()
        },
    )
    .unwrap();
    let m = &res.maps[0];
    assert!(
        (m.seidel.focus / w020 - 1.0).abs() < 0.01,
        "{} vs {w020}",
        m.seidel.focus
    );
    assert!(m.seidel.coma.abs() < 1e-6 && m.seidel.astigmatism.abs() < 1e-6);
    assert!(m.zernike_residual_rms < 1e-4);
    // RMS of pure defocus W020 ρ²: W020/√12 (to the ρ⁴ content).
    assert!((m.rms_waves / (w020 / 12f64.sqrt()) - 1.0).abs() < 0.01);
}

/// Spherical aberration of a stopped-down singlet: the Zernike-derived
/// `W040 = 6·Z9` agrees with the third-order Seidel `W040`, and the image at
/// the paraxial focus has no focus term beyond the higher orders.
#[test]
fn spherical_zernike_matches_seidel_w040() {
    let mut sys = samples::singlet();
    sys.aperture = ApertureSpec::EntrancePupilDiameter { value: 8.0 };
    sys.wavelengths = mono(0.5875618);
    sys.fields.fields.truncate(1);
    // Image at the paraxial focus of this wavelength.
    let t0 = tracer_for(sys.clone()).unwrap();
    let last = sys.surfaces.len() - 2;
    sys.surfaces[last].thickness = t0.paraxial().primary().image_distance;
    let t = tracer_for(sys).unwrap();
    let sd = seidel(&t, &SeidelSettings::default()).unwrap();
    let w040 = sd.wavefront_waves.w040;
    let res = wavefront_map(
        &t,
        &WavefrontSettings {
            pupil: uniform(),
            ..Default::default()
        },
    )
    .unwrap();
    let z = &res.maps[0].seidel;
    assert!(w040 > 0.1, "{w040}");
    assert!(
        (z.spherical / w040 - 1.0).abs() < 0.03,
        "Zernike {} vs Seidel {w040}",
        z.spherical
    );
    assert!(z.focus.abs() < 0.05 * w040, "focus {}", z.focus);
    // The fit reproduces the reference tracer's OPD of the marginal ray at
    // that ray's pupil position.
    let s = uniform();
    let tp = pupil(&t, 0, &s);
    let k = (0..tp.samples.samples.len())
        .find(|&k| {
            let p = lattice_point(s.rays, k);
            p[0].abs() < 1e-12 && (p[1] - 1.0).abs() < 1e-12
        })
        .unwrap();
    let marginal = tp.samples.samples[k].unwrap();
    let opd = t.opd(t.field(0), 0, 0.0, 1.0, &t.options()).unwrap();
    assert!((marginal.opd_waves - opd).abs() < 1e-9);
    let zf = res.maps[0].zernike.eval(marginal.u, marginal.v);
    assert!((zf - opd).abs() < 1e-3 * opd.abs() + 1e-4, "{zf} vs {opd}");
}

// -------------------------------------------------- off-axis FFT vs Huygens

#[test]
fn cooke_triplet_off_axis_mtf_fft_matches_huygens() {
    let mut sys = samples::cooke_triplet();
    sys.wavelengths = mono(0.5875618);
    let t = tracer_for(sys).unwrap();
    let base = MtfSettings {
        fields: FieldSelection::Index { index: 1 },
        pupil: PupilSettings {
            grid: 64,
            rays: 48,
            ..uniform()
        },
        frequency_samples: 13,
        method: MtfAnalysisMethod::Fft,
        huygens: HuygensGrid {
            // 128 samples at λF#/2 span the alias-free field of 64 pupil
            // samples, which holds the whole PSF.
            grid_size: 128,
            padding: 2.0,
        },
        ..Default::default()
    };
    let fft = mtf_analysis(&t, &base).unwrap();
    let cutoff = fft.frequencies.last().copied().unwrap();
    let s = MtfSettings {
        max_frequency: Some(0.5 * cutoff),
        ..base.clone()
    };
    let fft = mtf_analysis(&t, &s).unwrap();
    let hug = mtf_analysis(
        &t,
        &MtfSettings {
            method: MtfAnalysisMethod::Huygens,
            ..s.clone()
        },
    )
    .unwrap();
    let dir = mtf_analysis(
        &t,
        &MtfSettings {
            method: MtfAnalysisMethod::Direct,
            ..s
        },
    )
    .unwrap();
    let (a, b, c) = (&fft.fields[0], &hug.fields[0], &dir.fields[0]);
    let mut spread: f64 = 0.0;
    for k in 0..fft.frequencies.len() {
        let dt = (a.tangential[k] - b.tangential[k]).abs();
        let ds = (a.sagittal[k] - b.sagittal[k]).abs();
        assert!(
            dt < 0.03 && ds < 0.03,
            "ν {}: T {} / {}, S {} / {}",
            fft.frequencies[k],
            a.tangential[k],
            b.tangential[k],
            a.sagittal[k],
            b.sagittal[k]
        );
        // FFT interpolates between its lattice points (Δν = 1/(nλF#)).
        assert!((a.tangential[k] - c.tangential[k]).abs() < 0.02);
        assert!((a.sagittal[k] - c.sagittal[k]).abs() < 0.02);
        spread = spread.max((a.tangential[k] - a.sagittal[k]).abs());
    }
    // Off axis the two orientations differ markedly.
    assert!(spread > 0.05, "{spread}");
    assert_eq!(a.tangential_direction, [0.0, 1.0]);
}

// ----------------------------------------------------------- geometric limit

#[test]
fn large_defocus_mtf_approaches_geometric_mtf() {
    use mklight_analysis::energy::RaySampling;
    use mklight_analysis::energy::{GeometricMtfSettings, MtfOrientation as GO, geometric_mtf};
    // W020 ≈ 10 waves of defocus on an ideal F/5 lens.
    let (f, epd) = (100.0, 20.0);
    let dz = 10.0 * 2.0 * LAM * 1e-3 / 0.01;
    let t = ideal(f, epd, dz, &[0.0]);
    let s = MtfSettings {
        pupil: PupilSettings {
            grid: 160,
            rays: 96,
            ..uniform()
        },
        max_frequency: Some(40.0),
        frequency_samples: 21,
        ..Default::default()
    };
    let w = mtf_analysis(&t, &s).unwrap();
    let g = geometric_mtf(
        &t,
        &GeometricMtfSettings {
            sampling: RaySampling {
                rings: 120,
                ..Default::default()
            },
            max_frequency: Some(40.0),
            frequency_samples: 21,
            ..Default::default()
        },
    )
    .unwrap();
    let gt = g.curve(0, GO::Tangential).unwrap();
    // The blur circle diameter is 2·dz·NA ≈ 0.22 mm: the geometric MTF
    // 2J₁(x)/x has its first zero near 17 cycles/mm.
    for (k, &nu) in w.frequencies.iter().enumerate() {
        let (a, b) = (w.fields[0].tangential[k], gt.modulation[k]);
        assert!((a - b).abs() < 0.05, "ν {nu}: wave {a} vs geometric {b}");
    }
    // And far from the diffraction limit.
    assert!(w.fields[0].tangential[5] < 0.5 * w.fields[0].diffraction_limit[5]);
}

// ------------------------------------------------------------- vignetting

#[test]
fn vignetted_pupil_area_matches_ray_grid_transmission() {
    let mut sys = samples::cooke_triplet();
    sys.wavelengths = mono(0.5875618);
    // A front aperture smaller than the full-field beam footprint.
    sys.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 7.0 };
    sys.surfaces[6].semi_diameter = SemiDiameter::Fixed { value: 8.5 };
    let t = tracer_for(sys).unwrap();
    let s = PupilSettings {
        rays: 48,
        ..uniform()
    };
    let tp = pupil(&t, 2, &s);
    // Dense square grid in the entrance pupil.
    let m = 801;
    let (mut inside, mut pass) = (0usize, 0usize);
    for j in 0..m {
        for i in 0..m {
            let (px, py) = (
                -1.0 + 2.0 * (i as f64 + 0.5) / m as f64,
                -1.0 + 2.0 * (j as f64 + 0.5) / m as f64,
            );
            if px * px + py * py > 1.0 {
                continue;
            }
            inside += 1;
            if t.trace(t.field(2), 0, px, py, &t.options()).status.is_ok() {
                pass += 1;
            }
        }
    }
    let grid = pass as f64 / inside as f64;
    assert!(grid < 0.9 && grid > 0.3, "vignetting {grid}");
    let traced = tp.info.transmitted_fraction;
    assert!((traced - grid).abs() < 2e-3, "rim {traced} vs grid {grid}");
    // The clear area in (u, v) is the mask area, and the resampled pupil
    // covers it.
    let cov: f64 = tp.pupil.coverage().iter().sum::<f64>() * tp.pupil.spacing().powi(2);
    assert!(
        (cov / tp.info.clear_area - 1.0).abs() < 2e-3,
        "{cov} vs {}",
        tp.info.clear_area
    );
    let rim = tp.entrance_rim.as_ref().unwrap();
    assert!(matches!(rim, PupilMask::Polygons { loops } if loops.len() == 1));
    // Without the rim the triangle hull loses up to a ray spacing.
    let hull = pupil(&t, 2, &PupilSettings { rim: false, ..s });
    let hc: f64 = hull.pupil.coverage().iter().sum::<f64>() * hull.pupil.spacing().powi(2);
    assert!(hc < cov * 0.99, "{hc} vs {cov}");
}

// --------------------------------------------------------------- telecentric

#[test]
fn image_space_telecentric_uses_the_plane_wave_limit() {
    let (f, epd, dz) = (50.0, 10.0, 0.02);
    let t = telecentric(f, epd, dz, &[0.0, 5.0]);
    assert!(t.paraxial().primary().image_space_telecentric());
    let tp = pupil(&t, 0, &uniform());
    assert!(tp.info.telecentric && tp.info.reference_radius_mm.is_infinite());
    assert!(tp.pupil.geometry().is_telecentric());
    // Defocus: OPD = n·dz·(1 − cosθ) with sinθ = ρ·sinθ'max exactly.
    let sm = tp.info.sin_max;
    for r in tp.samples.samples.iter().flatten() {
        let s2 = (r.u * r.u + r.v * r.v) * sm * sm;
        let exact = dz * s2 / (1.0 + (1.0 - s2).sqrt()) / (LAM * 1e-3);
        assert!(
            (r.opd_waves - exact).abs() < 1e-8,
            "{} vs {exact}",
            r.opd_waves
        );
    }
    // The reference tracer's OPD takes the same `R → ∞` limit: equal on
    // every lattice ray, on and off axis (positive defocus beyond focus).
    for fi in 0..2 {
        let s = PupilSettings {
            rays: 21,
            ..uniform()
        };
        let tpf = pupil(&t, fi, &s);
        let mut n = 0;
        for (k, r) in tpf.samples.samples.iter().enumerate() {
            let Some(r) = r else { continue };
            let p = lattice_point(s.rays, k);
            let o = t.opd(t.field(fi), 0, p[0], p[1], &t.options()).unwrap();
            assert!(
                (o - r.opd_waves).abs() < 1e-9,
                "field {fi} {p:?}: trace {o} vs wave {}",
                r.opd_waves
            );
            n += 1;
        }
        assert!(n > 300, "{n}");
        let edge = t.opd(t.field(fi), 0, 0.0, 1.0, &t.options()).unwrap();
        assert!(edge > 0.1, "field {fi}: edge OPD {edge}");
    }
    // The pupil's own refocus brings it back to perfect.
    let back = tp.pupil.with_focus_shift(-dz);
    // (to the resampling error: grey edge cells hold the OPD averaged over
    // their covered part, while the refocus term is added at the centre).
    assert!(
        back.opd_stats(false).rms < 1e-3,
        "{}",
        back.opd_stats(false).rms
    );
    assert!(back.strehl_at_reference() > 0.9999);
    // Huygens (plane waves) and FFT PSFs agree on the refocused pupil and
    // on the defocused one.
    for p in [&back, &tp.pupil] {
        let lf = p.lambda_f_mm();
        let fp = fft_psf(
            p,
            &PsfSettings {
                grid_size: 33,
                sampling: ImageSampling::Spacing { mm: lf / 4.0 },
                normalization: PsfNormalization::Strehl,
                transform: PsfTransform::Matrix,
            },
            &ctx(),
        )
        .unwrap();
        let hp = huygens_psf(
            p,
            &HuygensSettings {
                grid_size: 33,
                spacing_mm: lf / 4.0,
                plane: ImagePlane::default(),
                normalization: PsfNormalization::Strehl,
            },
            &ctx(),
        )
        .unwrap();
        let d = fp
            .data
            .iter()
            .zip(&hp.data)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        assert!(d < 1e-9, "FFT vs plane-wave Huygens {d}");
    }
    // Off axis the chief ray stays parallel to the axis.
    let off = pupil(&t, 1, &uniform());
    let d = off.info.chief_direction;
    assert!(d[0].hypot(d[1]) < 1e-9, "{d:?}");
    assert!(off.pupil.geometry().is_on_axis());
    // Without defocus the system is perfect on and off axis.
    let perfect = telecentric(f, epd, 0.0, &[0.0, 5.0]);
    for fi in 0..2 {
        let p = pupil(&perfect, fi, &uniform());
        assert!(p.pupil.opd_stats(true).rms < 1e-6, "field {fi}");
    }
}

// --------------------------------------------------------- mirrors and signs

/// PSF centroid and geometric ray centroid of `field` (both relative to
/// the chief ray, mm), and the PSF spacing `λF#/2`.
fn psf_and_ray_centroids(
    t: &SequentialTracer,
    field: usize,
    amplitude: AmplitudeModel,
    grid: usize,
) -> ([f64; 2], [f64; 2], f64) {
    let ps = psf_analysis(
        t,
        &PsfAnalysisSettings {
            fields: FieldSelection::Index { index: field },
            wavelengths: WavelengthSelection::Primary,
            pupil: PupilSettings {
                amplitude,
                grid,
                ..Default::default()
            },
            // The whole alias-free field of the pupil.
            grid_size: 2 * grid,
            padding: 2.0,
            ..Default::default()
        },
    )
    .unwrap();
    let p = &ps.psfs[0];
    let psf_c = [
        p.reference_mm[0] + p.centroid_mm[0],
        p.reference_mm[1] + p.centroid_mm[1],
    ];
    let sp = spot_diagram(
        t,
        &SpotDiagramSettings {
            fields: FieldSelection::Index { index: field },
            wavelengths: WavelengthSelection::Primary,
            pattern: mklight_analysis::util::PupilPattern::Square,
            rings: 60,
            include_points: false,
            ..Default::default()
        },
    )
    .unwrap();
    let fs = &sp.fields[0];
    let shift = [fs.centroid[0] - fs.chief[0], fs.centroid[1] - fs.chief[1]];
    let got = [psf_c[0] - fs.chief[0], psf_c[1] - fs.chief[1]];
    (got, shift, p.psf.spacing_mm)
}

/// The PSF centroid equals the geometric ray centroid (both relative to
/// the chief ray) for an aberrated field: a check of the pupil orientation
/// (coordinates are sphere positions) and of the mirror frame, in `x` and
/// `y`.
fn centroid_check(t: &SequentialTracer, field: usize, label: &str) {
    let (got, shift, dx) = psf_and_ray_centroids(t, field, AmplitudeModel::Radiometric, 128);
    let lf = dx * 2.0;
    let mag = shift[0].hypot(shift[1]);
    assert!(mag > lf, "{label}: centroid shift {shift:?} too small");
    // The whole-plane PSF centroid equals the ray centroid (moment
    // theorem). The sampled field misses part of the far diffraction wings,
    // whose first moment converges like 1/(field width): about 1.5 % here
    // (3 % on a 64λF# field, 0.6 % on 256λF#). A sign or orientation error
    // would be off by 200 %.
    let tol = 0.03 * mag + 0.02 * lf;
    assert!(
        (got[0] - shift[0]).abs() < tol && (got[1] - shift[1]).abs() < tol,
        "{label}: PSF centroid {got:?} vs ray centroid {shift:?}"
    );
}

#[test]
fn mirror_frames_and_signs() {
    // Single paraboloid: the light reaches the image toward −z.
    let (f, epd) = (250.0, 50.0);
    let t = paraboloid(f, epd, 0.0, &[0.0, 1.5]);
    let tp = pupil(&t, 0, &uniform());
    assert_eq!(tp.info.propagation_sign, -1.0);
    assert!(tp.info.chief_direction[2] < 0.0);
    assert!(tp.pupil.opd_stats(true).rms < 1e-6);
    // The image moved further along the propagation: positive edge OPD,
    // equal to the refracting closed form.
    let dz = 0.2;
    let td = paraboloid(f, epd, dz, &[0.0]);
    let tpd = pupil(
        &td,
        0,
        &PupilSettings {
            rays: 32,
            ..uniform()
        },
    );
    let r = tpd.info.reference_radius_mm;
    assert!((r - (f + dz)).abs() < 1e-6, "R {r}");
    for rs in tpd.samples.samples.iter().flatten() {
        let rho = rs.u.hypot(rs.v);
        let exact = defocus_opd_exact(dz, r, tpd.info.sin_max, rho) / (LAM * 1e-3);
        assert!(
            (rs.opd_waves - exact).abs() < 1e-6,
            "{} vs {exact}",
            rs.opd_waves
        );
    }
    // Off-axis coma of the paraboloid and of the folded Newtonian: the PSF
    // centroid sits where the rays do.
    centroid_check(&t, 1, "paraboloid");
    let mut n = samples::newtonian_telescope();
    n.wavelengths = mono(LAM);
    let tn = tracer_for(n).unwrap();
    let pn = pupil(&tn, 0, &uniform());
    assert!(pn.pupil.opd_stats(true).rms < 1e-6);
    centroid_check(&tn, 2, "newtonian");
    let mut c = samples::cooke_triplet();
    c.wavelengths = mono(0.5875618);
    centroid_check(&tracer_for(c).unwrap(), 1, "triplet");
}

// --------------------------------------------------------------- amplitudes

/// An ideal lens maps entrance-pupil height `h` to `sinθ = h/√(f² + h²)`,
/// so the flux per unit `(u, v)` area grows as `1/cos⁴θ` and the sphere
/// amplitude as `A² ∝ γ/J = 1/cos³θ`. Transmission is 1 (ideal lens in
/// air), and the uniform model is flat.
#[test]
fn radiometric_amplitude_of_an_ideal_lens() {
    let t = ideal(50.0, 50.0, 0.0, &[0.0]);
    let s = PupilSettings {
        amplitude: AmplitudeModel::Radiometric,
        rays: 41,
        ..Default::default()
    };
    let tp = pupil(&t, 0, &s);
    let sm = tp.info.sin_max;
    let mut n = 0;
    for (k, r) in tp.samples.samples.iter().enumerate() {
        let Some(r) = r else { continue };
        let p = lattice_point(s.rays, k);
        // Interior nodes only (central differences).
        if p[0].hypot(p[1]) > 0.9 {
            continue;
        }
        let s2 = (r.u * r.u + r.v * r.v) * sm * sm;
        let cos = (1.0 - s2).sqrt();
        let expect = cos.powi(-3);
        assert!(
            (r.amplitude * r.amplitude / expect - 1.0).abs() < 2e-3,
            "{p:?}: {} vs {expect}",
            r.amplitude * r.amplitude
        );
        n += 1;
    }
    assert!(n > 900);
    let tr = pupil(
        &t,
        0,
        &PupilSettings {
            amplitude: AmplitudeModel::Transmission,
            ..s
        },
    );
    assert!(
        tr.samples
            .samples
            .iter()
            .flatten()
            .all(|r| (r.amplitude - 1.0).abs() < 1e-12)
    );
}

/// A 1:1 ideal-lens relay at finite conjugates (object at 2f) is perfect on
/// axis, with image-space NA equal to the object-space NA. Its ray
/// directions are mirror images (`sinθ' = sinθ`), so the radiometric
/// amplitude `A² ∝ m·γ/J` with the isotropic-source measure
/// `m = |∂(L,M)/∂P|/cosθ`, `J = |∂(u,v)/∂P| ∝ |∂(L,M)/∂P|` and `γ = cosθ` is
/// flat.
#[test]
fn finite_conjugate_ideal_relay() {
    let f = 50.0;
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::ObjectSpaceNa { value: 0.2 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectHeight,
        fields: vec![Field::new(0.0, 0.0), Field::new(0.0, 2.0)],
    };
    s.wavelengths = mono(LAM);
    s.surfaces = vec![
        surf(0, SurfaceShape::Plane, 2.0 * f),
        surf(1, SurfaceShape::Paraxial { focal_length: f }, 2.0 * f),
        surf(2, SurfaceShape::Plane, 0.0),
    ];
    s.stop = 1;
    let t = tracer_for(s).unwrap();
    let tp = pupil(
        &t,
        0,
        &PupilSettings {
            amplitude: AmplitudeModel::Radiometric,
            rays: 41,
            ..Default::default()
        },
    );
    assert!(tp.pupil.opd_stats(true).rms < 1e-6);
    assert!(
        (tp.info.numerical_aperture - 0.2).abs() < 1e-6,
        "{}",
        tp.info.numerical_aperture
    );
    // Symmetric relay: the object-side and image-side direction measures
    // cancel and the amplitude is flat.
    for r in tp.samples.samples.iter().flatten() {
        if r.u.hypot(r.v) < 0.9 {
            assert!((r.amplitude - 1.0).abs() < 2e-3, "{}", r.amplitude);
        }
    }
    let off = pupil(&t, 1, &uniform());
    assert!(
        off.pupil.opd_stats(true).rms < 1e-3,
        "{}",
        off.pupil.opd_stats(true).rms
    );
}

/// MTF vs field at the system field positions equals the MTF analysis of
/// those fields (same pupils, same transform).
#[test]
fn mtf_vs_field_matches_mtf_at_system_fields() {
    let t = tracer_for(samples::singlet()).unwrap();
    let pupil = PupilSettings {
        grid: 32,
        rays: 25,
        ..Default::default()
    };
    let vf = mtf_vs_field(
        &t,
        &MtfVsFieldSettings {
            pupil,
            field_samples: 3,
            frequencies: vec![10.0, 30.0],
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(vf.normalized_field, vec![0.0, 0.5, 1.0]);
    assert_eq!(vf.field_values[2], [0.0, 5.0]);
    let m = mtf_analysis(
        &t,
        &MtfSettings {
            fields: FieldSelection::List {
                indices: vec![0, 2],
            },
            pupil,
            max_frequency: Some(30.0),
            frequency_samples: 4,
            ..Default::default()
        },
    )
    .unwrap();
    for (h, f) in [(0, 0), (2, 1)] {
        for (q, k) in [(0, 1), (1, 3)] {
            assert_eq!(vf.curves[q].tangential[h], m.fields[f].tangential[k]);
            assert_eq!(vf.curves[q].sagittal[h], m.fields[f].sagittal[k]);
        }
    }
}

// ----------------------------------------------------------------- detector

#[test]
fn pixel_mtf_zeros_nyquist_and_product() {
    let p = 0.004;
    let d = DetectorSettings::square(p);
    for k in 1..4 {
        let nu = k as f64 / p;
        assert!(d.mtf([nu, 0.0]) < 1e-12 && d.mtf([0.0, nu]) < 1e-12);
    }
    assert!((d.mtf([0.5 / p, 0.0]) - 2.0 / PI).abs() < 1e-12);
    // Fill factor 1/4: aperture p/2, first zero at 2/p.
    let ff = DetectorSettings {
        fill_factor: 0.25,
        ..d
    };
    assert!(ff.mtf([1.0 / p, 0.0]) > 0.6 && ff.mtf([2.0 / p, 0.0]) < 1e-12);
    // Diffusion: exp(−2π²σ²ν²).
    let g = DetectorSettings {
        diffusion_sigma_mm: 0.002,
        ..d
    };
    let nu = 50.0;
    let expect = d.mtf([nu, 0.0]) * (-2.0 * PI * PI * 0.002f64.powi(2) * nu * nu).exp();
    assert!((g.mtf([nu, 0.0]) - expect).abs() < 1e-15);
    // In the MTF analysis: system = optical × detector, Nyquist marked.
    let t = ideal(50.0, 10.0, 0.0, &[0.0]);
    let r = mtf_analysis(
        &t,
        &MtfSettings {
            pupil: PupilSettings {
                rays: 24,
                grid: 32,
                ..uniform()
            },
            max_frequency: Some(2.0 / p),
            frequency_samples: 9,
            detector: Some(d),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(r.nyquist, Some([125.0, 125.0]));
    let fm = &r.fields[0];
    let det = fm.detector.as_ref().unwrap();
    assert_eq!(det.nyquist_tangential, 125.0);
    let st = fm.system_tangential.as_ref().unwrap();
    for ((s, o), d) in st.iter().zip(&fm.tangential).zip(&det.tangential) {
        assert!((s - o * d).abs() < 1e-15);
    }
    // Frequencies 0, 62.5, …, 500: the sinc zero at 1/p = 250 is index 4.
    assert!(st[4].abs() < 1e-12 && fm.tangential[4] > 0.1);
}

// ----------------------------------------------------------- analyses glue

#[test]
fn registry_runs_every_analysis_from_json() {
    let mut sys = samples::singlet();
    sys.fields.fields.truncate(2);
    let t = tracer_for(sys).unwrap();
    let small = serde_json::json!({
        "pupil": {"grid": 32, "rays": 24},
        "grid_size": 32,
        "frequency_samples": 5,
        "field_samples": 2,
        "focus_samples": 3,
        "samples": 5,
        "huygens": {"grid_size": 16}
    });
    assert_eq!(ANALYSES.len(), 7);
    for a in ANALYSES {
        let d = (a.default_settings)();
        assert!(d.is_object(), "{}", a.name);
        let v = (a.run)(&t, &small, &ctx()).unwrap_or_else(|e| panic!("{}: {e}", a.name));
        assert_eq!(v["run_info"]["analysis"], a.name);
        assert_eq!(v["run_info"]["backend"], "cpu-f64");
        assert!(find(a.name).is_some());
    }
    // Bad settings are an error, not a panic.
    assert!(
        (find("psf").unwrap().run)(&t, &serde_json::json!({"grid_size": 4097}), &ctx()).is_err()
    );
}

#[test]
fn results_round_trip_and_are_deterministic() {
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let s = PsfAnalysisSettings {
        fields: FieldSelection::Index { index: 2 },
        pupil: PupilSettings {
            grid: 32,
            rays: 24,
            ..Default::default()
        },
        grid_size: 48,
        ..Default::default()
    };
    let a = psf_analysis(&t, &s).unwrap();
    let b = psf_analysis(&t, &s).unwrap();
    assert_eq!(a.psfs, b.psfs);
    let json = serde_json::to_string(&a).unwrap();
    let back: PsfResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.psfs, a.psfs);
    let w = wavefront_map(
        &t,
        &WavefrontSettings {
            fields: FieldSelection::Index { index: 2 },
            pupil: PupilSettings {
                grid: 32,
                rays: 24,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    let json = serde_json::to_string(&w).unwrap();
    assert!(json.contains("null"), "masked samples serialise as null");
    let back: WavefrontResult = serde_json::from_str(&json).unwrap();
    assert_eq!(back.maps[0].zernike, w.maps[0].zernike);
}

#[test]
fn cancellation_and_progress() {
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let c = JobContext::new();
    c.cancel.cancel();
    let s = MtfSettings::default();
    assert!(matches!(
        mtf_analysis_job(&t, &s, &c),
        Err(mklight_core::MkError::Cancelled)
    ));
    let c = JobContext::new();
    let s = MtfSettings {
        fields: FieldSelection::Index { index: 1 },
        pupil: PupilSettings {
            grid: 32,
            rays: 24,
            ..Default::default()
        },
        ..Default::default()
    };
    mtf_analysis_job(&t, &s, &c).unwrap();
    assert_eq!(c.progress.fraction(), 1.0);
}

#[test]
fn polychromatic_lateral_colour_and_weights() {
    // The singlet off axis: per-wavelength chief points differ (lateral
    // colour); the polychromatic PSF is the weighted sum of unit-energy
    // PSFs placed at their own chief points.
    let t = tracer_for(samples::singlet()).unwrap();
    let base = PsfAnalysisSettings {
        fields: FieldSelection::Index { index: 2 },
        pupil: PupilSettings {
            grid: 32,
            rays: 24,
            ..uniform()
        },
        grid_size: 64,
        spacing_mm: Some(0.004),
        normalization: PsfNormalization::UnitEnergy,
        ..Default::default()
    };
    let poly = psf_analysis(&t, &base).unwrap();
    let mono = psf_analysis(
        &t,
        &PsfAnalysisSettings {
            polychromatic: false,
            ..base
        },
    )
    .unwrap();
    assert_eq!(mono.psfs.len(), 3);
    let w = &t.system().wavelength_weights;
    let ws: f64 = w.iter().sum();
    let p = &poly.psfs[0].psf;
    for k in 0..p.data.len() {
        let sum: f64 = mono
            .psfs
            .iter()
            .map(|m| w[m.wavelength.unwrap()] / ws * m.psf.data[k])
            .sum();
        assert!((p.data[k] - sum).abs() < 1e-9 * (1.0 + sum.abs()));
    }
}

#[test]
fn configurations_and_through_focus() {
    // A focus configuration moves the image; through-focus MTF of the base
    // system peaks where the configuration's does at zero shift.
    let (f, epd) = (100.0, 20.0);
    let mut sys = system(
        epd,
        vec![surf(
            1,
            SurfaceShape::Paraxial { focal_length: f },
            f + 0.03,
        )],
        1,
        &[0.0],
    );
    sys.configurations = vec![
        mklight_core::Configuration::default(),
        mklight_core::Configuration {
            name: "focused".into(),
            overrides: vec![(
                mklight_core::param::ParamPath::Thickness {
                    surface: mklight_core::SurfaceId(1),
                },
                f,
            )],
        },
    ];
    let lib = std::sync::Arc::new(mklight_core::materials::MaterialLibrary::with_builtin());
    let t0 = tracer_for_configuration(&sys, lib.clone(), 0).unwrap();
    let t1 = tracer_for_configuration(&sys, lib, 1).unwrap();
    let p0 = pupil(&t0, 0, &uniform());
    let p1 = pupil(&t1, 0, &uniform());
    // W020 = dz·NA²/(2λ) ≈ 0.27 waves: σ = W020/√12.
    assert!(p0.pupil.opd_stats(true).rms > 0.07);
    assert!(p1.pupil.opd_stats(true).rms < 1e-6);
    let tf = through_focus_mtf_analysis(
        &t0,
        &ThroughFocusSettings {
            pupil: PupilSettings {
                grid: 48,
                rays: 32,
                ..uniform()
            },
            frequencies: vec![100.0],
            focus_range_mm: Some(0.06),
            focus_samples: 5,
            ..Default::default()
        },
    )
    .unwrap();
    // Shifts −0.06, −0.03, 0, 0.03, 0.06: the best is −0.03 (back to focus).
    let c = &tf.fields[0].curves[0];
    let best = (0..5)
        .max_by(|&a, &b| c.tangential[a].total_cmp(&c.tangential[b]))
        .unwrap();
    assert_eq!(best, 1, "{:?}", c.tangential);
    let dl = diffraction_limited_mtf(
        100.0 * LAM * 1e-3 * p1.pupil.geometry().working_f_number(),
        0.0,
    );
    assert!(
        (c.tangential[1] - dl).abs() < 5e-3,
        "{} vs {dl}",
        c.tangential[1]
    );
    // Symmetric about the focus.
    assert!((c.tangential[0] - c.tangential[2]).abs() < 5e-3);
}

#[test]
fn encircled_energy_and_line_spread_known_answers() {
    // Diffraction-limited ideal lens: EE = 1 − J₀² − J₁² at v = πr/(λN),
    // and the LSFs integrate to one.
    let t = ideal(100.0, 20.0, 0.0, &[0.0]);
    let s = EncircledEnergySettings {
        pupil: PupilSettings {
            grid: 64,
            rays: 48,
            ..uniform()
        },
        grid_size: 256,
        center: EnergyCenter::Reference,
        ..Default::default()
    };
    let e = diffraction_encircled_energy(&t, &s).unwrap();
    let tp = pupil(&t, 0, &uniform());
    let lf = tp.pupil.lambda_f_mm();
    let fe = &e.fields[0];
    for (k, &r) in e.radius_mm.iter().enumerate() {
        if r > 10.0 * lf {
            break;
        }
        let a = mklight_wave::special::airy_encircled_energy(PI * r / lf);
        assert!(
            (fe.encircled[k] - a).abs() < 5e-3,
            "r {}: {} vs {a}",
            r / lf,
            fe.encircled[k]
        );
        assert!((fe.diffraction_limit[k] - fe.encircled[k]).abs() < 1e-3);
    }
    // r50 and r80 of the Airy pattern, by bisection on the closed form.
    let airy_r = |level: f64| {
        let (mut a, mut b) = (0.0, 5.0);
        for _ in 0..100 {
            let m = 0.5 * (a + b);
            if mklight_wave::special::airy_encircled_energy(PI * m) < level {
                a = m;
            } else {
                b = m;
            }
        }
        0.5 * (a + b)
    };
    assert!(
        (fe.r50_mm / lf - airy_r(0.5)).abs() < 5e-3,
        "{}",
        fe.r50_mm / lf
    );
    assert!(
        (fe.r80_mm / lf - airy_r(0.8)).abs() < 2e-2,
        "{}",
        fe.r80_mm / lf
    );
    let ls = line_spread(
        &t,
        &LineSpreadSettings {
            pupil: s.pupil,
            ..Default::default()
        },
    )
    .unwrap();
    let l = &ls.fields[0];
    let dx = l.positions_mm[1] - l.positions_mm[0];
    let ix: f64 = l.x.iter().sum::<f64>() * dx;
    let iy: f64 = l.y.iter().sum::<f64>() * dx;
    assert!(
        (ix - 1.0).abs() < 0.02 && (iy - 1.0).abs() < 0.02,
        "{ix} {iy}"
    );
    assert!(
        l.x.iter()
            .zip(&l.y)
            .all(|(a, b)| (a - b).abs() < 1e-6 * l.x[l.x.len() / 2])
    );
}

// ------------------------------------------------------ limits and robustness

/// Huygens image grids are bounded in every analysis (one JSON value must
/// not start an unbounded job), and so is the Huygens cost.
#[test]
fn huygens_grids_and_cost_are_bounded() {
    let t = ideal(50.0, 10.0, 0.0, &[0.0]);
    let invalid = |name: &str, v: serde_json::Value| {
        let r = (find(name).unwrap().run)(&t, &v, &ctx());
        assert!(
            matches!(r, Err(mklight_core::MkError::InvalidParam(_))),
            "{name} {v}: {r:?}"
        );
    };
    let big = serde_json::json!({"method": "huygens", "huygens": {"grid_size": 100000}});
    invalid("mtf", big.clone());
    invalid("mtf_vs_field", big.clone());
    invalid("through_focus_mtf", big);
    for name in ["psf", "diffraction_encircled_energy", "line_spread"] {
        invalid(
            name,
            serde_json::json!({"method": "huygens", "grid_size": MAX_HUYGENS_GRID + 1}),
        );
    }
    // The FFT route keeps its larger grids.
    let ok = PsfAnalysisSettings {
        pupil: PupilSettings {
            grid: 16,
            rays: 9,
            ..uniform()
        },
        grid_size: 2 * MAX_HUYGENS_GRID,
        ..Default::default()
    };
    psf_analysis(&t, &ok).unwrap();
    // Cost: a 512² pupil (≈ 2·10⁵ samples) on a 1024² Huygens grid.
    let r = psf_analysis(
        &t,
        &PsfAnalysisSettings {
            method: PsfMethod::Huygens,
            grid_size: MAX_HUYGENS_GRID,
            pupil: PupilSettings {
                grid: 512,
                rays: 9,
                supersample: 1,
                ..uniform()
            },
            ..ok
        },
    );
    match r {
        Err(mklight_core::MkError::InvalidParam(m)) => assert!(m.contains("too expensive"), "{m}"),
        other => panic!("{other:?}"),
    }
}

/// A truncated PSF (spot larger than the grid) is flagged; a
/// diffraction-limited one on the same grid is not.
#[test]
fn truncated_psf_is_flagged() {
    let truncated = |dz: f64| {
        let t = ideal(100.0, 20.0, dz, &[0.0]);
        let r = psf_analysis(
            &t,
            &PsfAnalysisSettings {
                pupil: PupilSettings {
                    grid: 64,
                    rays: 33,
                    ..uniform()
                },
                grid_size: 64,
                ..Default::default()
            },
        )
        .unwrap();
        let flagged = r.psfs[0]
            .psf
            .warnings
            .iter()
            .any(|w| matches!(w, mklight_wave::warnings::WaveWarning::PsfTruncated { .. }));
        let e = r.psfs[0].psf.energy_in_grid() / r.psfs[0].psf.total_energy;
        let in_run_info = r.run_info.warnings.iter().any(|w| w.contains("truncated"));
        assert_eq!(flagged, in_run_info);
        (flagged, e)
    };
    let (f0, e0) = truncated(0.0);
    assert!(!f0 && e0 > MIN_ENERGY_IN_GRID, "{e0}");
    // 8 waves of defocus: a spot of ≈ 32 λF# on a 16 λF# grid.
    let (f1, e1) = truncated(8.0 * 2.0 * 0.55e-3 / 0.1f64.powi(2));
    assert!(f1 && e1 < 0.9, "{e1}");
}

/// Cooke triplet with 3 mm front and rear apertures: the full field is
/// blocked. It is skipped with a warning; the usable fields keep their
/// results (and `mtf_vs_field` reports `null` there).
#[test]
fn unusable_fields_are_skipped_with_a_warning() {
    let mut sys = samples::cooke_triplet();
    sys.wavelengths = mono(0.5875618);
    sys.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 3.0 };
    sys.surfaces[6].semi_diameter = SemiDiameter::Fixed { value: 3.0 };
    let t = tracer_for(sys).unwrap();
    let pupil = PupilSettings {
        grid: 32,
        rays: 24,
        ..Default::default()
    };
    let m = mtf_analysis(
        &t,
        &MtfSettings {
            pupil,
            frequency_samples: 5,
            ..Default::default()
        },
    )
    .unwrap();
    let got: Vec<usize> = m.fields.iter().map(|f| f.field).collect();
    assert_eq!(got, vec![0, 1]);
    assert!(
        m.run_info
            .warnings
            .iter()
            .any(|w| w.starts_with("field 2 skipped")),
        "{:?}",
        m.run_info.warnings
    );
    let w = wavefront_map(
        &t,
        &WavefrontSettings {
            pupil,
            ..Default::default()
        },
    )
    .unwrap();
    // Field 1 keeps too few of the 24² lattice rays for a 37-term fit: its
    // map is skipped too, with a warning.
    assert_eq!(w.maps.len(), 1);
    for k in [1, 2] {
        assert!(
            w.run_info
                .warnings
                .iter()
                .any(|x| x.starts_with(&format!("field {k}")) && x.contains("skipped")),
            "{:?}",
            w.run_info.warnings
        );
    }
    // Only the dead field selected: an error.
    let only = mtf_analysis(
        &t,
        &MtfSettings {
            fields: FieldSelection::Index { index: 2 },
            pupil,
            ..Default::default()
        },
    );
    assert!(
        matches!(only, Err(mklight_core::MkError::Trace(_))),
        "{only:?}"
    );
    let vf = mtf_vs_field(
        &t,
        &MtfVsFieldSettings {
            pupil,
            field_samples: 3,
            frequencies: vec![20.0],
            ..Default::default()
        },
    )
    .unwrap();
    let c = &vf.curves[0];
    assert!(c.tangential[0].is_finite() && c.tangential[2].is_nan());
    let json = serde_json::to_value(&vf).unwrap();
    assert!(json["curves"][0]["tangential"][2].is_null());
    let back: MtfVsFieldResult = serde_json::from_value(json).unwrap();
    assert!(back.curves[0].sagittal[2].is_nan());
}

/// The axial sample of `mtf_vs_field` takes the sweep direction as its
/// tangential direction, like every other sample: with non-square pixels
/// the detector factor at `h = 0` is the one along the sweep.
#[test]
fn mtf_vs_field_axial_sample_follows_the_sweep_direction() {
    let t = tracer_for(samples::singlet()).unwrap();
    let pupil = PupilSettings {
        grid: 32,
        rays: 25,
        ..uniform()
    };
    let det = DetectorSettings {
        pitch_mm: [0.002, 0.008],
        ..DetectorSettings::square(0.002)
    };
    let nu = 40.0;
    let vf = mtf_vs_field(
        &t,
        &MtfVsFieldSettings {
            pupil,
            field_samples: 2,
            frequencies: vec![nu],
            direction: [1.0, 0.0],
            detector: Some(det),
            ..Default::default()
        },
    )
    .unwrap();
    let optical = mtf_analysis(
        &t,
        &MtfSettings {
            fields: FieldSelection::Index { index: 0 },
            pupil,
            max_frequency: Some(nu),
            frequency_samples: 2,
            ..Default::default()
        },
    )
    .unwrap();
    // Sweep along x: tangential is the optical x (sagittal of the default
    // axial orientation) times the x pixel factor, and vice versa (to the
    // direct method's ±ν asymmetry; a swap would be off by > 0.005).
    let (ot, os) = (
        optical.fields[0].tangential[1],
        optical.fields[0].sagittal[1],
    );
    let (dx, dy) = (det.mtf([nu, 0.0]), det.mtf([0.0, nu]));
    assert!(dx > dy + 0.1);
    let c = &vf.curves[0];
    assert!(
        (c.tangential[0] - os * dx).abs() < 1e-4,
        "{}",
        c.tangential[0]
    );
    assert!((c.sagittal[0] - ot * dy).abs() < 1e-4, "{}", c.sagittal[0]);
}

/// A configured tracer stamps the session snapshot's hash, the
/// configuration and the solve warnings into the `RunInfo`.
#[test]
fn configured_runs_record_the_configuration() {
    let f = 100.0;
    let mut sys = system(
        20.0,
        vec![surf(1, SurfaceShape::Paraxial { focal_length: f }, f)],
        1,
        &[0.0],
    );
    let thickness = mklight_core::param::ParamPath::Thickness {
        surface: mklight_core::SurfaceId(1),
    };
    sys.solves = vec![mklight_core::Solve {
        target: thickness.clone(),
        kind: mklight_core::SolveKind::MarginalRayHeight { height: 0.0 },
    }];
    sys.configurations = vec![
        mklight_core::Configuration::default(),
        mklight_core::Configuration {
            name: "moved".into(),
            overrides: vec![(thickness, f + 1.0)],
        },
    ];
    let lib = std::sync::Arc::new(mklight_core::materials::MaterialLibrary::with_builtin());
    let cfg = configured_tracer(&sys, lib, 1).unwrap();
    let base = mklight_core::Snapshot::new(sys.clone())
        .unwrap()
        .hash()
        .to_hex();
    assert_eq!(cfg.base_snapshot_hash, base);
    assert_eq!(cfg.configuration, 1);
    assert!(!cfg.solve_warnings.is_empty());
    // The solve wins: the configured system is in focus.
    let tp = pupil(&cfg.tracer, 0, &uniform());
    assert!(tp.pupil.opd_stats(true).rms < 1e-6);
    let small = serde_json::json!({"pupil": {"grid": 16, "rays": 9}});
    let v = find("wavefront_map")
        .unwrap()
        .run_configured(&cfg, &small, &ctx())
        .unwrap();
    let ri: mklight_core::RunInfo = serde_json::from_value(v["run_info"].clone()).unwrap();
    assert_eq!(ri.snapshot_hash, base);
    assert_eq!(ri.settings["configuration"], 1);
    assert_eq!(ri.settings["pupil"]["grid"], 16);
    assert_eq!(ri.warnings[0], cfg.solve_warnings[0]);
    // Unstamped, the run names the configured snapshot.
    let raw = (find("wavefront_map").unwrap().run)(&cfg.tracer, &small, &ctx()).unwrap();
    assert_ne!(raw["run_info"]["snapshot_hash"], base.as_str());
}

// ------------------------------------------------------------- more geometry

/// Stop at `1.5 f` in front of an ideal lens: the exit pupil (at `3f`) lies
/// beyond the image (`s_E = 2f − dz > 0`, `σ = +1`).
fn far_exit_pupil(f: f64, epd: f64, dz: f64) -> SequentialTracer {
    tracer_for(system(
        epd,
        vec![
            surf(1, SurfaceShape::Plane, 1.5 * f),
            surf(2, SurfaceShape::Paraxial { focal_length: f }, f + dz),
        ],
        1,
        &[0.0],
    ))
    .unwrap()
}

/// Virtual exit pupil beyond the image: the reference sphere lies on the
/// far sheet, and a defocus beyond focus is still positive. A perfect wave
/// diverging from the focus `dz` before `I` meets the sphere at
/// `t(θ) = dz cosθ + √(R² − dz² sin²θ)` from the focus, at pupil radius
/// `t sinθ/R`, so `OPD = n(dz + R − t(θ))`.
#[test]
fn exit_pupil_beyond_the_image() {
    let (f, epd, dz) = (50.0, 10.0, 0.05);
    let t = far_exit_pupil(f, epd, dz);
    let tp = pupil(
        &t,
        0,
        &PupilSettings {
            rays: 32,
            ..uniform()
        },
    );
    let r = tp.info.reference_radius_mm;
    assert!((r - (2.0 * f - dz)).abs() < 1e-6, "R {r}");
    assert!(tp.info.exit_pupil_point_mm.unwrap()[2] > tp.info.chief_point_mm[2]);
    let sm = tp.info.sin_max;
    let lam_mm = LAM * 1e-3;
    let exact = |rho: f64| {
        let tt = |th: f64| dz * th.cos() + (r * r - (dz * th.sin()).powi(2)).sqrt();
        let (mut lo, mut hi) = (0.0, 0.5 * PI);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if tt(mid) * mid.sin() / r < rho * sm {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        (dz + r - tt(0.5 * (lo + hi))) / lam_mm
    };
    let mut edge: f64 = 0.0;
    for (k, rs) in tp.samples.samples.iter().enumerate() {
        let Some(rs) = rs else { continue };
        let e = exact(rs.u.hypot(rs.v));
        assert!((rs.opd_waves - e).abs() < 1e-8, "{} vs {e}", rs.opd_waves);
        let p = lattice_point(32, k);
        let o = t.opd(t.field(0), 0, p[0], p[1], &t.options()).unwrap();
        assert!((o - rs.opd_waves).abs() < 1e-9);
        edge = edge.max(rs.opd_waves);
    }
    let w020 = dz * sm * sm / (2.0 * lam_mm);
    assert!(edge > 0.95 * w020 && edge < 1.05 * w020, "{edge} vs {w020}");
    // Refocusing on the sphere brings it back to perfect.
    assert!(tp.pupil.with_focus_shift(-dz).strehl_at_reference() > 0.9999);
}

/// Annular stop (ε = 0.4 in the entrance pupil): two rim loops, 84 %
/// transmission, the clear area of the mapped annulus, and the analytic
/// annular MTF.
#[test]
fn annular_pupil_area_and_mtf() {
    let (f, epd, eps) = (100.0, 20.0, 0.4);
    let mut sys = system(
        epd,
        vec![surf(1, SurfaceShape::Paraxial { focal_length: f }, f)],
        1,
        &[0.0],
    );
    sys.surfaces[1].aperture = Some(mklight_core::ApertureShape::Annular {
        inner_radius: eps * 0.5 * epd,
        outer_radius: 0.5 * epd,
    });
    let t = tracer_for(sys).unwrap();
    let tp = pupil(&t, 0, &uniform());
    let Some(PupilMask::Polygons { loops }) = &tp.mask else {
        panic!("no rim")
    };
    assert_eq!(loops.len(), 2);
    assert!(
        (tp.info.transmitted_fraction - (1.0 - eps * eps)).abs() < 1e-3,
        "{}",
        tp.info.transmitted_fraction
    );
    // The ideal lens maps height h to sinθ = h/√(f² + h²).
    let s = |h: f64| h / (f * f + h * h).sqrt();
    let eps_xp = s(eps * 0.5 * epd) / s(0.5 * epd);
    let area = PI * (1.0 - eps_xp * eps_xp);
    assert!(
        (tp.info.clear_area / area - 1.0).abs() < 2e-3,
        "{} vs {area}",
        tp.info.clear_area
    );
    let p = &tp.pupil;
    let n = p.geometry().working_f_number();
    let nu_c = 1.0 / (LAM * 1e-3 * n);
    let freqs: Vec<f64> = (0..=20).map(|k| k as f64 * 0.05 * nu_c).collect();
    let c = mtf_fft(p, &freqs, &MtfOrientation::default(), &ctx()).unwrap();
    for (k, &nu) in freqs.iter().enumerate() {
        let a = diffraction_limited_mtf(nu / nu_c, eps_xp);
        assert!(
            (c.tangential[k] - a).abs() < 3e-3 && (c.sagittal[k] - a).abs() < 3e-3,
            "ν {nu}: {} vs {a}",
            c.tangential[k]
        );
    }
}

/// Cancellation from another thread while the job runs, and progress that
/// passes through intermediate values and never goes back.
#[test]
fn mid_run_cancellation_and_monotone_progress() {
    let t = tracer_for(samples::cooke_triplet()).unwrap();
    let s = PsfAnalysisSettings {
        method: PsfMethod::Huygens,
        grid_size: 128,
        pupil: PupilSettings {
            grid: 48,
            rays: 24,
            ..Default::default()
        },
        ..Default::default()
    };
    // Progress samples of a complete run.
    let c = JobContext::new();
    let watch = c.clone();
    let done = std::sync::atomic::AtomicBool::new(false);
    let samples_seen = std::thread::scope(|sc| {
        let h = sc.spawn(|| {
            let mut seen = Vec::new();
            while !done.load(std::sync::atomic::Ordering::Relaxed) {
                seen.push(watch.progress.fraction());
                std::thread::sleep(std::time::Duration::from_micros(200));
            }
            seen
        });
        let small = PsfAnalysisSettings {
            grid_size: 48,
            ..s.clone()
        };
        psf_analysis_job(&t, &small, &c).unwrap();
        done.store(true, std::sync::atomic::Ordering::Relaxed);
        h.join().unwrap()
    });
    assert!(samples_seen.windows(2).all(|w| w[1] >= w[0]));
    assert!(
        samples_seen.iter().any(|&f| f > 0.05 && f < 0.95),
        "no intermediate progress"
    );
    assert_eq!(c.progress.fraction(), 1.0);
    // Cancel once the Huygens stage (after 40 % for pupil tracing) runs.
    let c = JobContext::new();
    let r = std::thread::scope(|sc| {
        let watch = c.clone();
        sc.spawn(move || {
            while watch.progress.fraction() < 0.45 {
                std::thread::sleep(std::time::Duration::from_micros(100));
            }
            watch.cancel.cancel();
        });
        psf_analysis_job(&t, &s, &c)
    });
    assert!(matches!(r, Err(mklight_core::MkError::Cancelled)), "{r:?}");
    assert!(c.progress.fraction() < 1.0);
}

/// Centroids for fields along `x` and on a diagonal (a sign error in `u`
/// would go unnoticed with `y` fields alone), on refracting and reflecting
/// systems.
#[test]
fn psf_centroid_follows_x_and_diagonal_fields() {
    let mut c = samples::cooke_triplet();
    c.wavelengths = mono(0.5875618);
    c.fields.fields = vec![
        mklight_core::Field::new(0.0, 0.0),
        mklight_core::Field::new(-14.0, 0.0),
        mklight_core::Field::new(10.0, -10.0),
    ];
    let tc = tracer_for(c).unwrap();
    centroid_check(&tc, 1, "triplet −x");
    centroid_check(&tc, 2, "triplet diagonal");
    let mut s = system(
        50.0,
        vec![{
            let mut m = surf(
                1,
                SurfaceShape::Conic {
                    curvature: -0.5 / 250.0,
                    conic: -1.0,
                },
                -250.0,
            );
            m.interaction = Interaction::Reflect;
            m
        }],
        1,
        &[0.0],
    );
    s.fields.fields = vec![
        mklight_core::Field::new(0.0, 0.0),
        mklight_core::Field::new(1.5, 0.0),
        mklight_core::Field::new(-1.0, 1.0),
    ];
    let tp = tracer_for(s).unwrap();
    centroid_check(&tp, 1, "paraboloid +x");
    centroid_check(&tp, 2, "paraboloid diagonal");
}

/// The `transmission` amplitude leaves out the ray density, so the PSF's
/// energy distribution (and centroid) is biased where the pupil is
/// distorted; `radiometric`, the default, restores the moment theorem.
/// Singlet at full field: the bias is about a third of the centroid shift
/// (WAVE.md §3).
#[test]
fn transmission_amplitude_biases_the_psf_centroid() {
    assert_eq!(PupilSettings::default().amplitude, AmplitudeModel::Radiometric);
    assert_eq!(AmplitudeModel::default(), AmplitudeModel::Radiometric);
    let mut s = samples::singlet();
    s.wavelengths = mono(0.5875618);
    let t = tracer_for(s).unwrap();
    // 256 pupil samples: at 128 the wavefront steps by 0.67 waves per
    // sample and the wrapped flare even flips the centroid's sign (a
    // `pupil_undersampled` warning says so).
    let (gt, shift, _) = psf_and_ray_centroids(&t, 2, AmplitudeModel::Transmission, 256);
    let (gr, _, _) = psf_and_ray_centroids(&t, 2, AmplitudeModel::Radiometric, 256);
    let et = (gt[1] - shift[1]).abs() / shift[1].abs();
    let er = (gr[1] - shift[1]).abs() / shift[1].abs();
    assert!(et > 0.15, "transmission bias {et}");
    assert!(er < 0.1, "radiometric error {er}");
}

/// The default (direct) MTF of a strongly aberrated field is right near
/// zero frequency on the default pupil grid, where the FFT route's lattice
/// interpolation is not (WAVE.md §5): singlet at full field.
#[test]
fn default_mtf_method_is_right_at_low_frequencies() {
    let mut s = samples::singlet();
    s.wavelengths = mono(0.5875618);
    let t = tracer_for(s).unwrap();
    let run = |grid: usize, method: MtfAnalysisMethod| {
        let r = mtf_analysis(
            &t,
            &MtfSettings {
                fields: FieldSelection::Index { index: 2 },
                pupil: PupilSettings {
                    grid,
                    ..Default::default()
                },
                method,
                max_frequency: Some(40.0),
                frequency_samples: 9,
                ..Default::default()
            },
        )
        .unwrap();
        r.fields[0].tangential[1]
    };
    assert_eq!(MtfSettings::default().method, MtfAnalysisMethod::Direct);
    // 5 cycles/mm; the FFT lattice spacing at n = 64 is 5.4 cycles/mm.
    let reference = run(256, MtfAnalysisMethod::Fft);
    let direct = run(64, MtfAnalysisMethod::Direct);
    let fft = run(64, MtfAnalysisMethod::Fft);
    assert!((direct - reference).abs() < 0.01, "{direct} vs {reference}");
    assert!((fft - reference).abs() > 0.02, "{fft} vs {reference}");
}

/// A Standard fit takes at most [`MAX_ZERNIKE_TERMS`] terms; more are
/// rejected before any pupil is traced.
#[test]
fn wavefront_fit_terms_are_bounded() {
    let t = tracer_for(samples::singlet()).unwrap();
    let settings = |terms| WavefrontSettings {
        fields: FieldSelection::Index { index: 0 },
        pupil: PupilSettings {
            rays: 33,
            ..Default::default()
        },
        zernike_ordering: ZernikeOrdering::Standard,
        zernike_terms: terms,
        ..Default::default()
    };
    let full = wavefront_map(&t, &settings(MAX_ZERNIKE_TERMS)).unwrap();
    assert_eq!(full.maps[0].zernike.coefficients().len(), MAX_ZERNIKE_TERMS);
    assert!(matches!(
        wavefront_map(&t, &settings(MAX_ZERNIKE_TERMS + 1)),
        Err(mklight_core::MkError::InvalidParam(_))
    ));
}

// ------------------------------------------- lattice interpolation accuracy

/// F/1.29 aspheric singlet (n = 1.7848, R = 20 mm with r², r⁴, r⁶ terms,
/// EPD 20 mm, 0.587 µm): about 2 waves of secondary spherical with a steep
/// rim. Found by an independent cross-check: with the default 65-ray
/// lattice the linearly interpolated pupil gave a Strehl ratio 20 % low
/// (0.0267 against 0.0330 at 257 rays, which an independent FFT-PSF code
/// also gives at 256 rays) without any warning. The lattice estimator now
/// reports it, and the recommended sampling is clean and converged.
#[test]
fn coarse_lattice_on_a_fast_asphere_is_reported() {
    let mut s1 = Surface::new(
        1,
        SurfaceShape::OddAsphere {
            curvature: 1.0 / 20.0,
            conic: 0.0,
            coefficients: vec![0.0, -2.248851e-4, 0.0, -4.690412e-6, 0.0, -6.404376e-8],
        },
        7.0,
        MaterialRef::Constant {
            n: 1.7848059679000448,
        },
    );
    s1.semi_diameter = SemiDiameter::Auto;
    let mut sys = system(
        20.0,
        vec![s1, surf(2, SurfaceShape::Plane, 21.56201105)],
        1,
        &[0.0],
    );
    sys.wavelengths = mono(0.587);
    let t = tracer_for(sys).unwrap();
    let run = |rays: usize| {
        let res = wavefront_map(
            &t,
            &WavefrontSettings {
                pupil: PupilSettings { rays, ..uniform() },
                ..Default::default()
            },
        )
        .unwrap();
        let warned = res
            .run_info
            .warnings
            .iter()
            .any(|w| w.contains("under-resolves the wavefront"));
        (res.maps[0].strehl_at_reference, warned)
    };
    let (coarse, coarse_warned) = run(65);
    let (fine, fine_warned) = run(257);
    assert!(coarse_warned, "65 rays must be reported");
    assert!(!fine_warned, "257 rays must be clean");
    assert!(coarse < 0.85 * fine, "coarse {coarse} vs fine {fine}");
    assert!((fine - 0.0330).abs() < 0.001, "fine {fine}");
    // The estimator itself on the traced lattice.
    let tp = pupil(&t, 0, &uniform());
    let e = lattice_interpolation_error(&tp.samples);
    assert!(e > 2.0 * LATTICE_INTERPOLATION_LIMIT_WAVES, "{e}");
    match lattice_warning(&tp.samples) {
        Some(WaveWarning::LatticeUndersampled {
            rays,
            recommended_rays,
            ..
        }) => {
            assert_eq!(rays, 65);
            assert!(
                recommended_rays > 65 && recommended_rays < 257,
                "{recommended_rays}"
            );
            assert!(
                !run(recommended_rays).1,
                "{recommended_rays} rays still reported"
            );
        }
        w => panic!("{w:?}"),
    }
}

/// A confocal paraboloid beam compressor (f 500 / f −100, EPD 50) is
/// afocal and aberration-free on axis. The reference sphere about the
/// chief ray's image point used to report about 23 waves RMS and a Strehl
/// of 1e-5 with no warning; the afocal image space is now reported (and
/// measured from a plane reference, `afocal.rs`). A
/// focusing paraboloid is not.
#[test]
fn afocal_image_space_is_reported() {
    let mirror = |id, c: f64, t| {
        let mut m = surf(
            id,
            SurfaceShape::Conic {
                curvature: c,
                conic: -1.0,
            },
            t,
        );
        m.interaction = Interaction::Reflect;
        m
    };
    let sys = system(
        50.0,
        vec![
            mirror(1, -1.0 / 1000.0, -400.0),
            mirror(2, -1.0 / 200.0, 200.0),
        ],
        1,
        &[0.0],
    );
    let t = tracer_for(sys).unwrap();
    assert!(t.paraxial().primary().image_na < 1e-9);
    let afocal = |t: &SequentialTracer| {
        let res = wavefront_map(t, &WavefrontSettings::default()).unwrap();
        res.run_info
            .warnings
            .iter()
            .any(|w| w.contains("image space is afocal"))
    };
    assert!(afocal(&t));
    assert!(!afocal(&paraboloid(100.0, 20.0, 0.0, &[0.0])));
}
