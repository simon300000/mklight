//! Afocal image spaces: the wave analyses on a plane reference normal to
//! the chief ray, in angular units (`mklight_wave::pupil`, "Afocal image
//! spaces"): positions in mrad, frequencies in cycles/mrad, focus as
//! vergence in millidioptres.

use mklight_analysis::tracer_for;
use mklight_analysis::util::FieldSelection;
use mklight_core::{
    ApertureSpec, Field, FieldKind, FieldSet, Interaction, JobContext, MaterialRef, MkError,
    OpticalSystem, Surface, SurfaceShape, Wavelength, WavelengthSet,
};
use mklight_trace::SequentialTracer;
use mklight_wave::analysis::*;
use mklight_wave::mtf::diffraction_limited_mtf;

const LAM: f64 = 0.55;

fn surf(id: u64, shape: SurfaceShape, t: f64, m: MaterialRef) -> Surface {
    Surface::new(id, shape, t, m)
}

/// Object at infinity, `body` (ids from 1), image; angle fields.
fn system(epd: f64, body: Vec<Surface>, fields: &[f64]) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: epd };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: fields.iter().map(|&y| Field::new(0.0, y)).collect(),
    };
    s.wavelengths = WavelengthSet {
        wavelengths: vec![Wavelength {
            um: LAM,
            weight: 1.0,
        }],
        primary: 0,
    };
    let n = body.len() as u64;
    s.surfaces = vec![surf(
        0,
        SurfaceShape::Plane,
        f64::INFINITY,
        MaterialRef::Air,
    )];
    s.surfaces.extend(body);
    s.surfaces
        .push(surf(n + 1, SurfaceShape::Plane, 0.0, MaterialRef::Air));
    s.stop = 1;
    s
}

fn mirror(id: u64, c: f64, t: f64) -> Surface {
    let mut m = surf(
        id,
        SurfaceShape::Conic {
            curvature: c,
            conic: -1.0,
        },
        t,
        MaterialRef::Air,
    );
    m.interaction = Interaction::Reflect;
    m
}

/// Confocal paraboloids (f 500 and 100 mm), EPD 50 mm: a 5× compressor
/// with a 10 mm output beam. Its exit pupil (the primary imaged by the
/// secondary) lies 400/3 mm behind the secondary.
fn compressor() -> SequentialTracer {
    tracer_for(system(
        50.0,
        vec![
            mirror(1, -1.0 / 1000.0, -400.0),
            mirror(2, -1.0 / 200.0, 200.0),
        ],
        &[0.0],
    ))
    .unwrap()
}

/// Two ideal lenses separated by `f1 + f2`; image 100 mm behind.
fn telescope(f1: f64, f2: f64, epd: f64, fields: &[f64]) -> SequentialTracer {
    tracer_for(system(
        epd,
        vec![
            surf(
                1,
                SurfaceShape::Paraxial { focal_length: f1 },
                f1 + f2,
                MaterialRef::Air,
            ),
            surf(
                2,
                SurfaceShape::Paraxial { focal_length: f2 },
                100.0,
                MaterialRef::Air,
            ),
        ],
        fields,
    ))
    .unwrap()
}

fn ctx() -> JobContext {
    JobContext::new()
}

/// Aberration-free afocal telescopes: zero wavefront error, Strehl 1, the
/// diffraction-limited MTF with cutoff `D/λ` cycles/rad (`D` the output
/// beam diameter) in cycles/mrad, an Airy pattern in mrad, units reported,
/// and image-space field angles. Previously the reference sphere about the
/// chief ray's image point gave tens of waves of defocus.
#[test]
fn aberration_free_telescopes_are_diffraction_limited_in_angle() {
    // (label, tracer, output beam diameter mm, image-space tan of the 1° field)
    let one = 1f64.to_radians().tan();
    let cases = [
        ("compressor", compressor(), 10.0, None),
        (
            "keplerian 10×",
            telescope(20.0, 200.0, 5.0, &[0.0, 1.0]),
            50.0,
            Some(-one / 10.0),
        ),
        (
            "galilean 5×",
            telescope(-20.0, 100.0, 4.0, &[0.0, 1.0]),
            20.0,
            Some(one / 5.0),
        ),
    ];
    for (label, t, d, tan_out) in cases {
        assert!(t.afocal_image_space(), "{label}");
        let w = wavefront_map(&t, &WavefrontSettings::default()).unwrap();
        assert!(w.units.afocal && w.units.position == "mrad", "{label}");
        assert!(
            w.run_info
                .warnings
                .iter()
                .any(|m| m.contains("image space is afocal")),
            "{label}: {:?}",
            w.run_info.warnings
        );
        for m in &w.maps {
            let e = &m.exit_pupil;
            assert!(e.afocal && !e.telecentric, "{label}");
            let a = e.beam_radius_mm.unwrap();
            assert!((a - 0.5 * d).abs() < 1e-6 * d, "{label}: beam radius {a}");
            assert!(m.rms_waves < 1e-6, "{label}: RMS {}", m.rms_waves);
            assert!(m.strehl_at_reference > 1.0 - 1e-9, "{label}");
            let angle = e.field_angle_deg.unwrap();
            assert!(angle[0].abs() < 1e-12, "{label}");
            let expect = if m.field == 0 {
                0.0
            } else {
                tan_out.unwrap().atan().to_degrees()
            };
            assert!(
                (angle[1] - expect).abs() < 1e-9,
                "{label} field {}: {} vs {expect}",
                m.field,
                angle[1]
            );
        }

        let psf = psf_analysis(&t, &PsfAnalysisSettings::default()).unwrap();
        assert_eq!(psf.units.position, "mrad");
        for p in &psf.psfs {
            assert!(
                (p.strehl - 1.0).abs() < 1e-6,
                "{label}: Strehl {}",
                p.strehl
            );
            // The PSF sits at the beam direction: 1000 × its direction sine.
            let expect = tan_out.map_or(0.0, |t| {
                if p.field == 0 {
                    0.0
                } else {
                    1000.0 * t.atan().sin()
                }
            });
            assert!(
                (p.reference_mm[1] - expect).abs() < 1e-9,
                "{label}: reference {:?} vs {expect}",
                p.reference_mm
            );
        }

        let mtf = mtf_analysis(&t, &MtfSettings::default()).unwrap();
        assert_eq!(mtf.units.frequency, "cycles/mrad");
        let cutoff = d / (LAM * 1e-3) * 1e-3;
        for f in &mtf.fields {
            assert!(
                (f.cutoff - cutoff).abs() < 1e-6 * cutoff,
                "{label}: cutoff {} vs {cutoff}",
                f.cutoff
            );
            for (k, &nu) in mtf.frequencies.iter().enumerate() {
                let exact = diffraction_limited_mtf(nu / cutoff, 0.0);
                for v in [f.tangential[k], f.sagittal[k], f.diffraction_limit[k]] {
                    assert!(
                        (v - exact).abs() < 2e-3,
                        "{label} {nu} cycles/mrad: {v} vs {exact}"
                    );
                }
            }
        }

        // Airy pattern: 83.8 % of the energy inside the first dark ring at
        // 1.22 λ/D rad.
        let ee = diffraction_encircled_energy(
            &t,
            &EncircledEnergySettings {
                fields: FieldSelection::Index { index: 0 },
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(ee.units.position, "mrad");
        let r1 = 1.22 * LAM * 1e-3 / d * 1e3;
        let f = &ee.fields[0];
        let k = ee.radius_mm.iter().position(|&r| r >= r1).unwrap();
        let (r0, r1k) = (ee.radius_mm[k - 1], ee.radius_mm[k]);
        let s = (r1 - r0) / (r1k - r0);
        let at = f.encircled[k - 1] + s * (f.encircled[k] - f.encircled[k - 1]);
        assert!(
            (at - 0.838).abs() < 0.01,
            "{label}: EE at the dark ring {at}"
        );
    }
}

/// The Huygens routes sum plane waves (the far field) and agree with the
/// FFT; through-focus is in millidioptres of vergence and peaks at zero for
/// a collimated beam: a vergence `V` is `W = V a²/2` of defocus.
#[test]
fn huygens_and_through_focus_in_angular_units() {
    let t = compressor();
    let huygens = psf_analysis(
        &t,
        &PsfAnalysisSettings {
            method: PsfMethod::Huygens,
            grid_size: 33,
            ..Default::default()
        },
    )
    .unwrap();
    let fft = psf_analysis(
        &t,
        &PsfAnalysisSettings {
            grid_size: 33,
            spacing_mm: Some(huygens.psfs[0].psf.spacing_mm),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((huygens.psfs[0].strehl - 1.0).abs() < 1e-6);
    let (h, f) = (&huygens.psfs[0].psf.data, &fft.psfs[0].psf.data);
    let worst = h
        .iter()
        .zip(f)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-6, "Huygens vs FFT PSF: {worst}");
    let mh = mtf_analysis(
        &t,
        &MtfSettings {
            method: MtfAnalysisMethod::Huygens,
            frequency_samples: 6,
            ..Default::default()
        },
    )
    .unwrap();
    let cutoff = 10.0 / (LAM * 1e-3) * 1e-3;
    for (k, &nu) in mh.frequencies.iter().enumerate() {
        let exact = diffraction_limited_mtf(nu / cutoff, 0.0);
        assert!(
            (mh.fields[0].tangential[k] - exact).abs() < 5e-3,
            "Huygens MTF {nu}: {} vs {exact}",
            mh.fields[0].tangential[k]
        );
    }

    // ±V mD of vergence on a 5 mm beam radius: W = V·1e-6/mm · a²/2.
    let v = 10.0;
    let tf = through_focus_mtf_analysis(
        &t,
        &ThroughFocusSettings {
            frequencies: vec![0.25 * cutoff],
            focus_range_mm: Some(v),
            focus_samples: 3,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(tf.units.focus, "mD");
    let c = &tf.fields[0].curves[0];
    assert!((c.tangential[1] - diffraction_limited_mtf(0.25, 0.0)).abs() < 2e-3);
    assert!((c.tangential[0] - c.tangential[2]).abs() < 1e-9);
    assert!(c.tangential[0] < c.tangential[1]);
    // The same defocus added to the pupil directly.
    let w020 = v * 1e-6 * 25.0 / 2.0 / (LAM * 1e-3);
    let tp = mklight_wave::traced::trace_pupil(
        &t,
        t.field(0),
        0,
        &mklight_wave::traced::PupilSettings::default(),
        &ctx(),
    )
    .unwrap();
    let refocused = tp.pupil.with_focus_shift(v);
    let defocused = tp.pupil.with_defocus_waves(w020);
    let worst = refocused
        .opd_waves()
        .iter()
        .zip(defocused.opd_waves())
        .filter(|(a, _)| a.is_finite())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(
        worst < 1e-3 * w020,
        "vergence vs defocus: {worst} of {w020}"
    );
}

/// A thin plate with an `A·r⁴` face at the exit pupil of a 10× Keplerian
/// telescope (ideal lenses; the exit pupil is `f2(f1 + f2)/f1` = 2200 mm
/// behind the second lens) adds the closed-form wavefront `(n − 1)·A·r⁴`
/// (positive: less glass at the edge) while the image space stays afocal.
#[test]
fn phase_plate_at_the_exit_pupil_gives_the_closed_form_wavefront() {
    let (n, plate) = (1.5, 1.0);
    let a: f64 = 25.0;
    let edge_waves = 1.0;
    let coef = edge_waves * LAM * 1e-3 / ((n - 1.0) * a.powi(4));
    let (f1, f2) = (20.0, 200.0);
    let z_xp = f2 * (f1 + f2) / f1;
    let glass = surf(
        3,
        SurfaceShape::OddAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: vec![0.0, 0.0, 0.0, coef],
        },
        plate,
        MaterialRef::Constant { n },
    );
    let t = tracer_for(system(
        5.0,
        vec![
            surf(
                1,
                SurfaceShape::Paraxial { focal_length: f1 },
                f1 + f2,
                MaterialRef::Air,
            ),
            surf(
                2,
                SurfaceShape::Paraxial { focal_length: f2 },
                z_xp,
                MaterialRef::Air,
            ),
            glass,
            surf(4, SurfaceShape::Plane, 100.0, MaterialRef::Air),
        ],
        &[0.0],
    ))
    .unwrap();
    assert!(t.afocal_image_space());
    let w = wavefront_map(&t, &WavefrontSettings::default()).unwrap();
    let m = &w.maps[0];
    let beam = m.exit_pupil.beam_radius_mm.unwrap();
    assert!((beam - a).abs() < 1e-6, "beam radius {beam}");
    let g = m.grid;
    let mut checked = 0;
    for j in 0..g {
        for i in 0..g {
            let o = m.opd_waves[j * g + i];
            if !o.is_finite() {
                continue;
            }
            let rho2 = m.coordinates[i].powi(2) + m.coordinates[j].powi(2);
            // Cells crossing the rim are partly extrapolated by the resampling.
            if rho2 > 0.95 {
                continue;
            }
            let exact = (n - 1.0) * coef * (beam * beam * rho2).powi(2) / (LAM * 1e-3);
            assert!(
                (o - exact).abs() < 5e-3,
                "({}, {}): {o} vs {exact}",
                m.coordinates[i],
                m.coordinates[j]
            );
            checked += 1;
        }
    }
    assert!(checked > 1000);
    // Fringe Z9 (6ρ⁴ − 6ρ² + 1) carries ρ⁴/6.
    let z9 = m.zernike.coefficients()[8];
    assert!((z9 - edge_waves / 6.0).abs() < 2e-3, "Z9 {z9}");
}

/// Image simulation needs image-surface positions and rejects afocal
/// image spaces.
#[test]
fn image_simulation_rejects_afocal_image_spaces() {
    let t = compressor();
    let Err(e) = mklight_wave::image_simulation::build_psf_grid(
        &t,
        &mklight_wave::image_simulation::ImageGridSettings::default(),
        &ctx(),
    ) else {
        panic!("afocal image simulation ran");
    };
    assert!(
        matches!(&e, MkError::InvalidModel(m) if m.contains("afocal")),
        "{e}"
    );
}
