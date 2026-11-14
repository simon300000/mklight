//! Polarisation ray tracing and radiometry: analytic, conservation and
//! cross-check tests (CONVENTIONS §5, §9; docs/POLARIZATION.md).

#![allow(clippy::needless_range_loop)]

use crate::common;

use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::Arc;

use common::{D_LINE, angle_fields, infinite_system, surf};
use mklight_core::materials::coating::{
    Complex64 as C, InterfaceAmplitudes, ResolvedCoating, fresnel, fresnel_amplitudes,
    tmm_amplitudes,
};
use mklight_core::materials::{GlassCatalog, MaterialLibrary, TransmittancePoint};
use mklight_core::math::Mat3;
use mklight_core::rng::Philox;
use mklight_core::{
    CoatingDef, Interaction, JobContext, Layer, MaterialRef, MkError, OpticalSystem, Placement,
    ResolvedSystem, Snapshot, SurfaceShape, Vec3, samples,
};
use mklight_trace::polarization::{
    StepKind, double_pole_basis, interface_prt, perpendicular, s_vector,
};
use mklight_trace::radiometry::{PolarizationMapSettings, PupilSamples};
use mklight_trace::{
    FieldPoint, JonesMatrix, JonesVector, PolarizationState, PolarizedRay, RadiometryModel,
    SequentialTracer, Stokes, TraceOptions, TransmissionSettings, polarization_pupil_map,
    system_transmission, vector_pupil,
};

// ---------------------------------------------------------------------------
// Fixtures.

fn tracer_lib(sys: OpticalSystem, lib: MaterialLibrary) -> SequentialTracer {
    let snap = Snapshot::new(sys).expect("valid system");
    let r = ResolvedSystem::new(&snap, Arc::new(lib)).expect("resolve");
    SequentialTracer::new(Arc::new(r)).expect("tracer")
}

fn tracer(sys: OpticalSystem) -> SequentialTracer {
    tracer_lib(sys, MaterialLibrary::with_builtin())
}

fn unpol() -> RadiometryModel {
    RadiometryModel::Polarized {
        state: PolarizationState::Unpolarized,
    }
}

fn jones_state(e: JonesVector) -> PolarizationState {
    PolarizationState::Jones { e }
}

fn qw_layer(n: f64) -> Layer {
    Layer {
        material: MaterialRef::Constant { n },
        thickness_um: 0.0,
        quarter_waves: Some(1.0),
        extinction: 0.0,
    }
}

fn thin_film(layers: Vec<Layer>, mirror: Option<(f64, f64)>) -> CoatingDef {
    CoatingDef::ThinFilm {
        reference_wavelength_um: D_LINE,
        layers,
        mirror_substrate: mirror,
        substrate: Default::default(),
    }
}

/// N-BK7 singlet at infinity, monochromatic d-line, F/10.
fn singlet(material: MaterialRef, coating: Option<CoatingDef>) -> OpticalSystem {
    let mut s = infinite_system(
        10.0,
        vec![
            surf(1, SurfaceShape::sphere(60.0), 5.0, material),
            surf(2, SurfaceShape::sphere(-350.0), 96.7, MaterialRef::Air),
        ],
        3,
    );
    s.fields = angle_fields(&[0.0, 5.0]);
    if let Some(c) = coating {
        s.coatings.insert("C".into(), c);
        s.surfaces[1].coating = Some("C".into());
        s.surfaces[2].coating = Some("C".into());
    }
    s
}

/// Plane-parallel plate tilted by `tilt_deg` about x (plane of incidence
/// y–z: s = x̂, p = ŷ for the on-axis ray).
fn tilted_plate(
    material: MaterialRef,
    tilt_deg: f64,
    coating: Option<CoatingDef>,
) -> OpticalSystem {
    let mut front = surf(1, SurfaceShape::Plane, 5.0, material);
    front.placement = Some(Placement {
        tilt_x: tilt_deg,
        ..Default::default()
    });
    let mut s = infinite_system(
        4.0,
        vec![front, surf(2, SurfaceShape::Plane, 10.0, MaterialRef::Air)],
        3,
    );
    s.fields = angle_fields(&[0.0]);
    if let Some(c) = coating {
        s.coatings.insert("C".into(), c);
        s.surfaces[1].coating = Some("C".into());
        s.surfaces[2].coating = Some("C".into());
    }
    s
}

/// A 45° fold mirror (placement tilt, no coordinate breaks): after it the
/// light travels toward −z of the image frame.
fn fold_mirror(coating: Option<CoatingDef>) -> OpticalSystem {
    let mut m = surf(2, SurfaceShape::Plane, -20.0, MaterialRef::Air);
    m.interaction = Interaction::Reflect;
    m.placement = Some(Placement {
        tilt_x: 45.0,
        ..Default::default()
    });
    let mut s = infinite_system(
        4.0,
        vec![surf(1, SurfaceShape::Plane, 10.0, MaterialRef::Air), m],
        3,
    );
    s.surfaces[3].placement = Some(Placement {
        tilt_x: 45.0,
        ..Default::default()
    });
    s.fields = angle_fields(&[0.0]);
    if let Some(c) = coating {
        s.coatings.insert("M".into(), c);
        s.surfaces[2].coating = Some("M".into());
    }
    s
}

fn on_axis(t: &SequentialTracer, px: f64, py: f64) -> PolarizedRay {
    let r = t.trace_polarized(t.field(0), 0, px, py, &TraceOptions::unclipped());
    assert!(r.is_ok(), "{:?}", r.trace.status);
    r
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(1e-300)
}

/// Largest elementwise difference; NaN if any element is NaN (a plain
/// `f64::max` fold would drop it).
fn jones_diff(a: &JonesMatrix, b: &JonesMatrix) -> f64 {
    let mut d = 0.0f64;
    for i in 0..2 {
        for j in 0..2 {
            let x = (a.m[i][j] - b.m[i][j]).norm();
            if x.is_nan() {
                return f64::NAN;
            }
            d = d.max(x);
        }
    }
    d
}

/// Largest component difference of two power sets; NaN-propagating.
fn rt_diff(
    a: &mklight_core::materials::coating::InterfaceRT,
    b: &mklight_core::materials::coating::InterfaceRT,
) -> f64 {
    let mut d = 0.0f64;
    for (x, y) in [(a.rs, b.rs), (a.rp, b.rp), (a.ts, b.ts), (a.tp, b.tp)] {
        let e = (x - y).abs();
        if e.is_nan() {
            return f64::NAN;
        }
        d = d.max(e);
    }
    d
}

#[test]
fn comparison_helpers_see_nan() {
    use mklight_core::materials::coating::InterfaceRT;
    let mut a = JonesMatrix::IDENTITY;
    a.m[0][0] = C::new(f64::NAN, 0.0);
    assert!(jones_diff(&a, &JonesMatrix::IDENTITY).is_nan());
    let x = InterfaceRT {
        rp: f64::NAN,
        ..InterfaceRT::TRANSPARENT
    };
    assert!(rt_diff(&x, &InterfaceRT::TRANSPARENT).is_nan());
    let mut p = mklight_trace::PrtMatrix::IDENTITY;
    p.m[1][2] = C::new(0.0, f64::NAN);
    assert!(p.max_abs_diff(&mklight_trace::PrtMatrix::IDENTITY).is_nan());
}

fn rot2(a: f64) -> JonesMatrix {
    let (s, c) = a.sin_cos();
    JonesMatrix {
        m: [
            [C::new(c, 0.0), C::new(-s, 0.0)],
            [C::new(s, 0.0), C::new(c, 0.0)],
        ],
    }
}

fn unit(r: &mut Philox) -> Vec3 {
    let z = 2.0 * r.uniform() - 1.0;
    let (s, c) = (2.0 * PI * r.uniform()).sin_cos();
    let q = ((1.0 - z) * (1.0 + z)).sqrt();
    Vec3::new(q * c, q * s, z)
}

fn refracted(k: Vec3, n: Vec3, n1: f64, n2: f64) -> Vec3 {
    mklight_trace::interact::refract(k, n, n1, n2).unwrap().0
}

// ---------------------------------------------------------------------------
// Jones / Stokes algebra.

#[test]
fn jones_properties_of_known_elements() {
    // J = R(θ) diag(a e^{iδ/2}, b e^{−iδ/2}) R(−θ): homogeneous elliptic-free
    // element with D = (a² − b²)/(a² + b²), retardance δ, both axes along
    // the linear direction θ (Stokes (cos 2θ, sin 2θ, 0)); x leads, so it is
    // the fast axis.
    let mut r = Philox::new(1, 0);
    for _ in 0..2000 {
        let th = PI * r.uniform();
        let (a, b) = (0.2 + 0.8 * r.uniform(), 0.1 + 0.5 * r.uniform());
        let delta = PI * r.uniform();
        let d = JonesMatrix {
            m: [
                [C::from_polar(a, 0.5 * delta), C::new(0.0, 0.0)],
                [C::new(0.0, 0.0), C::from_polar(b, -0.5 * delta)],
            ],
        };
        let j = rot2(th).mul(&d).mul(&rot2(-th)).scale(1.7);
        let p = j.properties();
        let (a2, b2) = (a * a * 2.89, b * b * 2.89);
        assert!(rel(p.t_max, a2.max(b2)) < 1e-12 && rel(p.t_min, a2.min(b2)) < 1e-12);
        assert!((p.diattenuation - (a2 - b2).abs() / (a2 + b2)).abs() < 1e-13);
        assert!(
            (p.retardance - delta).abs() < 1e-11,
            "{} {delta}",
            p.retardance
        );
        let ax = [(2.0 * th).cos(), (2.0 * th).sin(), 0.0];
        let sgn = if a >= b { 1.0 } else { -1.0 };
        for k in 0..3 {
            assert!(
                (p.diattenuation_axis[k] - sgn * ax[k]).abs() < 1e-9,
                "{p:?}"
            );
            assert!((p.fast_axis[k] - ax[k]).abs() < 1e-9, "{p:?} {ax:?}");
        }
        assert!((p.t_unpolarized - 0.5 * (a2 + b2)).abs() < 1e-13);
    }
    // Quarter-wave plate diag(1, −i): y lags, x is fast.
    let qwp = JonesMatrix {
        m: [
            [C::new(1.0, 0.0), C::new(0.0, 0.0)],
            [C::new(0.0, 0.0), C::new(0.0, -1.0)],
        ],
    };
    let p = qwp.properties();
    assert!((p.retardance - FRAC_PI_2).abs() < 1e-15 && p.diattenuation < 1e-16);
    assert!((p.fast_axis[0] - 1.0).abs() < 1e-15);
    // A rotator by ψ is a circular retarder of 2ψ; it turns x toward y
    // (counter-clockwise looking at the source), which the S₃ < 0 state leads.
    let p = rot2(0.3).properties();
    assert!((p.retardance - 0.6).abs() < 1e-14);
    assert!((p.fast_axis[2] + 1.0).abs() < 1e-14, "{p:?}");
    // Small retardance keeps full relative accuracy.
    let tiny = JonesMatrix {
        m: [
            [C::from_polar(1.0, 1e-9), C::new(0.0, 0.0)],
            [C::new(0.0, 0.0), C::new(1.0, 0.0)],
        ],
    };
    assert!(rel(tiny.properties().retardance, 1e-9) < 1e-6);
    // Perfect polariser: D = 1, retardance reported as 0.
    let pol = JonesMatrix {
        m: [
            [C::new(1.0, 0.0), C::new(0.0, 0.0)],
            [C::new(0.0, 0.0), C::new(0.0, 0.0)],
        ],
    };
    let p = pol.properties();
    assert_eq!((p.diattenuation, p.retardance, p.t_min), (1.0, 0.0, 0.0));
}

#[test]
fn stokes_intensity_and_unpolarized_average() {
    let mut r = Philox::new(2, 0);
    let mut c = || C::new(2.0 * r.uniform() - 1.0, 2.0 * r.uniform() - 1.0);
    for _ in 0..2000 {
        let j = JonesMatrix {
            m: [[c(), c()], [c(), c()]],
        };
        let e = JonesVector::new(c(), c());
        // Pure state: Stokes formula equals |J e|².
        let direct = j.apply(&e).intensity();
        assert!(rel(j.intensity(&e.stokes()), direct) < 1e-12);
        assert!((e.stokes().degree_of_polarization() - 1.0).abs() < 1e-12);
        // Unpolarised = average of any two orthogonal states.
        let u = j.intensity_of(&PolarizationState::Unpolarized);
        let s2 = std::f64::consts::FRAC_1_SQRT_2;
        let pairs = [
            (JonesVector::linear(0.0), JonesVector::linear(FRAC_PI_2)),
            (
                JonesVector::linear(0.25 * PI),
                JonesVector::linear(-0.25 * PI),
            ),
            (
                JonesVector::new(C::new(s2, 0.0), C::new(0.0, s2)),
                JonesVector::new(C::new(s2, 0.0), C::new(0.0, -s2)),
            ),
        ];
        for (a, b) in pairs {
            let avg = 0.5 * (j.apply(&a).intensity() + j.apply(&b).intensity());
            assert!(rel(u, avg) < 1e-12);
        }
        // Partially polarised = incoherent mixture.
        let s = Stokes {
            s0: 1.0,
            s1: 0.3,
            s2: -0.2,
            s3: 0.1,
        };
        let pol = (0.3f64.hypot(0.2)).hypot(0.1);
        let pure = Stokes {
            s0: pol,
            s1: 0.3,
            s2: -0.2,
            s3: 0.1,
        };
        let mixed = j.intensity(&pure) + (1.0 - pol) * u;
        assert!(rel(j.intensity(&s), mixed) < 1e-12);
    }
}

// ---------------------------------------------------------------------------
// Single interfaces.

#[test]
fn interface_prt_maps_directions_powers_and_bases() {
    let mut r = Philox::new(3, 0);
    for _ in 0..3000 {
        let n = unit(&mut r);
        let mut k = unit(&mut r);
        if k.dot(n) < 0.0 {
            k = -k;
        }
        let (n1, n2) = (1.0 + r.uniform(), 1.0 + r.uniform());
        let ci = k.dot(n);
        let Some((ko, _)) = mklight_trace::interact::refract(k, n, n1, n2) else {
            continue;
        };
        let a = fresnel_amplitudes(n1, n2, ci);
        let s = s_vector(k, n);
        let p = interface_prt(k, ko, s, a.ts_flux(), a.tp_flux());
        // P k = k'.
        let pk = p.apply(k);
        for (x, y) in pk.iter().zip([ko.x, ko.y, ko.z]) {
            assert!((x - C::new(y, 0.0)).norm() < 1e-14);
        }
        // |P s|² = T_s, |P p|² = T_p, no cross talk.
        let rt = fresnel(n1, n2, ci);
        let ps = p.apply(s);
        let pp = p.apply(k.cross(s));
        let n2c = |v: &[C; 3]| v.iter().map(|c| c.norm_sqr()).sum::<f64>();
        assert!((n2c(&ps) - rt.ts).abs() < 1e-12 && (n2c(&pp) - rt.tp).abs() < 1e-12);
        let cross: C = ps.iter().zip(pp.iter()).map(|(a, b)| a.conj() * b).sum();
        assert!(cross.norm() < 1e-14);
        // Reflection: geometric matrix of a perfect conductor is orthogonal
        // with determinant −1 (mirror parity).
        let (kr, _) = mklight_trace::interact::reflect(k, n);
        let q = interface_prt(k, kr, s, C::new(-1.0, 0.0), C::new(1.0, 0.0)).re();
        let qtq = q.transpose().mul_mat(&q);
        for i in 0..3 {
            for j in 0..3 {
                let e = if i == j { 1.0 } else { 0.0 };
                assert!((qtq.m[i][j] - e).abs() < 1e-14);
            }
        }
        let m = &q.m;
        let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
        assert!((det + 1.0).abs() < 1e-13);
    }
}

#[test]
fn normal_incidence_basis_is_irrelevant_and_continuous() {
    // At normal incidence any s ⟂ k gives the same P (transmission, metal
    // reflection, absorbing multilayer).
    let k = Vec3::new(0.3, -0.2, 0.9).normalized();
    let n = -k; // either orientation of the normal
    let stack = [(C::new(2.1, 0.0), 0.07), (C::new(1.6, -0.05), 0.12)];
    let cases: Vec<(InterfaceAmplitudes, bool)> = vec![
        (fresnel_amplitudes(1.0, 1.75, 1.0), false),
        (
            tmm_amplitudes(C::new(1.0, 0.0), &stack, C::new(1.52, 0.0), 1.0, 0.55),
            false,
        ),
        (
            tmm_amplitudes(C::new(1.0, 0.0), &stack, C::new(0.96, -6.69), 1.0, 0.55),
            true,
        ),
    ];
    let s0 = perpendicular(k);
    for (a, mirror) in cases {
        let (ko, cs, cp) = if mirror {
            (-k, a.rs, a.rp)
        } else {
            (k, a.ts_flux(), a.tp_flux())
        };
        let reference = interface_prt(k, ko, s0, cs, cp);
        // Closed form: a(I − kkᵀ) + k'kᵀ with a = t̃ (transmission) or r_s
        // (reflection, where p' = −p and r_p = −r_s).
        let kv = [k.x, k.y, k.z];
        let kov = [ko.x, ko.y, ko.z];
        for i in 0..3 {
            for j in 0..3 {
                let id = if i == j { 1.0 } else { 0.0 };
                let want = cs * (id - kv[i] * kv[j]) + C::new(kov[i] * kv[j], 0.0);
                assert!((reference.m[i][j] - want).norm() < 1e-15, "{i}{j}");
            }
        }
        for j in 0..24 {
            let psi = 0.3 * j as f64;
            let s = s0 * psi.cos() + k.cross(s0) * psi.sin();
            let p = interface_prt(k, ko, s, cs, cp);
            assert!(p.max_abs_diff(&reference) < 2e-15, "ψ={psi}");
        }
        assert_eq!(s_vector(k, n), s0);
    }
    // Continuity: incidence angles from 1e-2 down to below the 1e-12
    // threshold, at many azimuths. ‖P(θ) − P(0)‖ = O(θ) with no jump at
    // the fallback, for a refracting and a reflecting interface.
    let nz = Vec3::Z;
    let (n1, n2) = (1.0, 1.8);
    let p0 = {
        let a = fresnel_amplitudes(n1, n2, 1.0);
        interface_prt(nz, nz, s_vector(nz, nz), a.ts_flux(), a.tp_flux())
    };
    let metal = |ci: f64| tmm_amplitudes(C::new(1.0, 0.0), &[], C::new(0.96, -6.69), ci, 0.55);
    let m0 = {
        let a = metal(1.0);
        interface_prt(nz, -nz, s_vector(nz, nz), a.rs, a.rp)
    };
    for e in 2..=15 {
        let th = 10f64.powi(-e);
        for j in 0..12 {
            let phi = 0.5 + j as f64 * PI / 6.0;
            let (st, ct) = th.sin_cos();
            let k = Vec3::new(st * phi.cos(), st * phi.sin(), ct);
            let ko = refracted(k, nz, n1, n2);
            let ci = k.dot(nz);
            let a = fresnel_amplitudes(n1, n2, ci);
            let p = interface_prt(k, ko, s_vector(k, nz), a.ts_flux(), a.tp_flux());
            assert!(
                p.max_abs_diff(&p0) <= 2.0 * th + 1e-15,
                "θ={th}: {}",
                p.max_abs_diff(&p0)
            );
            let kr = mklight_trace::interact::reflect(k, nz).0;
            let a = metal(ci);
            let m = interface_prt(k, kr, s_vector(k, nz), a.rs, a.rp);
            assert!(m.max_abs_diff(&m0) <= 3.0 * th + 1e-15, "θ={th}");
        }
    }
}

#[test]
fn malus_law_with_ideal_polarizer_interface() {
    // An ideal p-transmitting interface (t̃_s = 0, t̃_p = 1) at 30° incidence;
    // linear input at angle θ from p transmits cos²θ, and two such
    // polarisers crossed by the input rotation give the full Malus curve.
    let n = Vec3::Z;
    let th = 30f64.to_radians();
    let k = Vec3::new(0.0, th.sin(), th.cos());
    let ko = refracted(k, n, 1.0, 1.5);
    let s = s_vector(k, n);
    let pol = interface_prt(k, ko, s, C::new(0.0, 0.0), C::new(1.0, 0.0));
    let p_in = k.cross(s);
    let p_out = ko.cross(s);
    let j = pol.jones((s, p_out), (s, p_in));
    for i in 0..=36 {
        let a = i as f64 * PI / 36.0;
        // Angle a from p (= second basis vector).
        let e = JonesVector::new(C::new(a.sin(), 0.0), C::new(a.cos(), 0.0));
        let out = j.intensity(&e.stokes());
        assert!((out - a.cos().powi(2)).abs() < 1e-15, "{a}: {out}");
    }
    let p = j.properties();
    assert!((p.diattenuation - 1.0).abs() < 1e-15 && (p.diattenuation_axis[0] + 1.0).abs() < 1e-15);
}

// ---------------------------------------------------------------------------
// Bases.

#[test]
fn double_pole_basis_properties() {
    let mut r = Philox::new(4, 0);
    let frame = Mat3::rot_x(0.4).mul_mat(&Mat3::rot_z(-1.1));
    for _ in 0..5000 {
        let k = unit(&mut r);
        let (x, y) = double_pole_basis(k, &frame);
        assert!((x.norm() - 1.0).abs() < 1e-14 && (y.norm() - 1.0).abs() < 1e-14);
        assert!(x.dot(y).abs() < 1e-14 && x.dot(k).abs() < 1e-14 && y.dot(k).abs() < 1e-14);
        assert!(x.cross(y).max_abs_diff(k) < 1e-14);
    }
    // At the pole it is the frame's own (x̂, ±ŷ).
    let ez = frame.mul_vec(Vec3::Z);
    let (x, y) = double_pole_basis(ez, &frame);
    assert!(x.max_abs_diff(frame.mul_vec(Vec3::X)) < 1e-15);
    assert!(y.max_abs_diff(frame.mul_vec(Vec3::Y)) < 1e-15);
    let (x, y) = double_pole_basis(-ez, &frame);
    assert!(x.max_abs_diff(frame.mul_vec(Vec3::X)) < 1e-15);
    assert!(y.max_abs_diff(-frame.mul_vec(Vec3::Y)) < 1e-15);
    // Covariance about the pole: rotating k by φ about z gives the rotated
    // basis turned back by φ in its own transverse plane (the basis keeps
    // following the fixed x̂, ŷ), B(R_z k) = R_z B(k) R(−φ).
    let k = Vec3::new(0.3, 0.1, 0.9).normalized();
    let (bx, by) = double_pole_basis(k, &Mat3::IDENTITY);
    for j in 0..10 {
        let phi = 0.6 * j as f64;
        let rz = Mat3::rot_z(phi);
        let x0 = bx * phi.cos() - by * phi.sin();
        let (x1, _) = double_pole_basis(rz.mul_vec(k), &Mat3::IDENTITY);
        let d = x1.max_abs_diff(rz.mul_vec(x0));
        assert!(d < 1e-15, "{d:e}");
    }
}

// ---------------------------------------------------------------------------
// Radiometry of traced systems.

/// N-BK7 with a constant internal transmittance τ(10 mm) = 0.9.
fn absorbing_library() -> MaterialLibrary {
    let mut lib = MaterialLibrary::with_builtin();
    let bk7 = lib.find_glass(None, "N-BK7").unwrap();
    let mut g = (*bk7).clone();
    g.name = "ABS-BK7".into();
    g.transmittance = vec![
        TransmittancePoint {
            um: 0.3,
            tau: 0.9,
            thickness_mm: 10.0,
        },
        TransmittancePoint {
            um: 2.5,
            tau: 0.9,
            thickness_mm: 10.0,
        },
    ];
    let mut cat = GlassCatalog {
        name: "TEST".into(),
        ..Default::default()
    };
    cat.glasses.insert(g.name.clone(), g);
    lib.add_catalog(Arc::new(cat));
    lib
}

#[test]
fn uncoated_singlet_normal_incidence_is_one_minus_r_squared() {
    let t = tracer_lib(
        singlet(MaterialRef::glass("ABS-BK7"), None),
        absorbing_library(),
    );
    let n = t.system().surfaces[1].n_after[0];
    let r = ((n - 1.0) / (n + 1.0)).powi(2);
    let tau = (0.9f64.ln() * 5.0 / 10.0).exp();
    let ray = on_axis(&t, 0.0, 0.0);
    assert!(rel(ray.internal_transmittance, tau) < 1e-13);
    let expect = (1.0 - r) * (1.0 - r) * tau;
    let models = [
        RadiometryModel::Unpolarized,
        unpol(),
        RadiometryModel::Polarized {
            state: jones_state(JonesVector::linear(0.7)),
        },
        RadiometryModel::Polarized {
            state: PolarizationState::Stokes {
                s: Stokes {
                    s0: 2.0,
                    s1: 0.5,
                    s2: 0.0,
                    s3: -1.0,
                },
            },
        },
    ];
    for m in &models {
        let th = mklight_trace::RayThroughput::of(&ray, m);
        assert!(
            rel(th.total, expect) < 1e-13,
            "{m:?}: {} vs {expect}",
            th.total
        );
    }
    // Normal incidence is isotropic: J is a multiple of the identity in
    // the double-pole bases.
    let j = ray.jones();
    let a = ((1.0 - r) * (1.0 - r)).sqrt();
    assert!(
        jones_diff(&j, &JonesMatrix::IDENTITY.scale(a)) < 1e-14,
        "{j:?}"
    );
    // Pupil-averaged transmission of the slow (F/10) lens is close to the
    // normal-incidence value and below it (Fresnel losses grow with angle).
    let settings = TransmissionSettings {
        fields: vec![FieldPoint::normalized(0.0, 0.0)],
        pupil: PupilSamples::Grid { n: 15 },
        ..Default::default()
    };
    let clear = tracer(singlet(MaterialRef::glass("N-BK7"), None));
    let res = system_transmission(&clear, &settings, &JobContext::new()).unwrap();
    let p = res.at(0, 0);
    let expect = (1.0 - r) * (1.0 - r);
    assert_eq!((p.failed, p.samples, p.unvignetted_fraction), (0, 149, 1.0));
    assert!(
        p.transmission <= expect && p.transmission > expect - 2e-4,
        "{p:?}"
    );
    // With absorption the thinner lens edge transmits more than the centre.
    let res = system_transmission(&t, &settings, &JobContext::new()).unwrap();
    assert!(res.at(0, 0).transmission > (1.0 - r) * (1.0 - r) * tau);
}

#[test]
fn quarter_wave_ar_reduces_losses_as_predicted() {
    // MgF2-like single layer, quarter wave at the d-line.
    let n1 = 1.38;
    let bare = tracer(singlet(MaterialRef::glass("N-BK7"), None));
    let ar = tracer(singlet(
        MaterialRef::glass("N-BK7"),
        Some(thin_film(vec![qw_layer(n1)], None)),
    ));
    let ns = bare.system().surfaces[1].n_after[0];
    let r_bare = ((ns - 1.0) / (ns + 1.0)).powi(2);
    let r_ar = ((ns - n1 * n1) / (ns + n1 * n1)).powi(2);
    for (t, r) in [(&bare, r_bare), (&ar, r_ar)] {
        let ray = on_axis(t, 0.0, 0.0);
        for m in [RadiometryModel::Unpolarized, unpol()] {
            let got = mklight_trace::RayThroughput::of(&ray, &m).total;
            assert!(rel(got, (1.0 - r) * (1.0 - r)) < 1e-12, "{got} {r}");
        }
        // Both faces: reflected power R and R(1 − R) reach the budget.
        let b = ray.energy_budget(&PolarizationState::Unpolarized);
        assert!((b.surfaces[0].diverted - r).abs() < 1e-14);
        assert!((b.surfaces[1].diverted - r * (1.0 - r)).abs() < 1e-14);
    }
    // Predicted improvement: ≈ 8.4 % loss → ≈ 2.5 % loss.
    let loss = |t: &SequentialTracer| {
        1.0 - mklight_trace::RayThroughput::of(&on_axis(t, 0.0, 0.0), &unpol()).total
    };
    let ratio = loss(&ar) / loss(&bare);
    let predicted = (1.0 - (1.0 - r_ar).powi(2)) / (1.0 - (1.0 - r_bare).powi(2));
    assert!(rel(ratio, predicted) < 1e-10 && ratio < 0.35, "{ratio}");
}

#[test]
fn meridional_unpolarized_exact_vs_incoherent_model() {
    // Off-axis meridional ray of a coaxial lens: every s vector is x̂, so the
    // exact unpolarised throughput is ½(ΠT_s + ΠT_p) and the incoherent model
    // Π½(T_s + T_p); they differ by exactly ¼(T_s1 − T_p1)(T_s2 − T_p2).
    let t = tracer(singlet(MaterialRef::glass("N-BK7"), None));
    let f = t.field(1).clone();
    for py in [-1.0, -0.4, 0.3, 1.0] {
        let ray = t.trace_polarized(&f, 0, 0.0, py, &TraceOptions::unclipped());
        assert!(ray.is_ok());
        assert_eq!(ray.steps.len(), 2);
        for st in &ray.steps {
            assert!(st.s.max_abs_diff(Vec3::X).min(st.s.max_abs_diff(-Vec3::X)) < 1e-15);
        }
        let (a, b) = (ray.steps[0].power, ray.steps[1].power);
        let exact = mklight_trace::RayThroughput::of(&ray, &unpol()).interfaces;
        let model =
            mklight_trace::RayThroughput::of(&ray, &RadiometryModel::Unpolarized).interfaces;
        assert!(rel(exact, 0.5 * (a.ts * b.ts + a.tp * b.tp)) < 1e-14);
        assert!(rel(model, 0.25 * (a.ts + a.tp) * (b.ts + b.tp)) < 1e-14);
        let d = 0.25 * (a.ts - a.tp) * (b.ts - b.tp);
        assert!(
            (exact - model - d).abs() < 1e-15 && d > 0.0,
            "{exact} {model} {d:e} {:e}",
            exact - model - d
        );
        // Jones matrix is diagonal in (x, y) = (s, p): no cross-polarisation.
        let j = ray.jones();
        assert!(j.m[0][1].norm() < 1e-15 && j.m[1][0].norm() < 1e-15);
    }
}

fn coated_triplet() -> OpticalSystem {
    let mut s = samples::cooke_triplet();
    s.coatings.insert(
        "V".into(),
        thin_film(vec![qw_layer(2.05), qw_layer(1.38)], None),
    );
    s.coatings.insert(
        "ABS".into(),
        thin_film(
            vec![
                Layer {
                    material: MaterialRef::Constant { n: 2.0 },
                    thickness_um: 0.03,
                    quarter_waves: None,
                    extinction: 0.4,
                },
                qw_layer(1.38),
            ],
            None,
        ),
    );
    s.coatings.insert(
        "ID".into(),
        CoatingDef::Ideal {
            reflectance: 0.01,
            transmittance: 0.97,
        },
    );
    s.coatings.insert(
        "TAB".into(),
        CoatingDef::Tabulated {
            reflectance: vec![(0.4, 0.01), (0.55, 0.004), (0.7, 0.012)],
        },
    );
    s.surfaces[1].coating = Some("V".into());
    s.surfaces[2].coating = Some("ABS".into());
    s.surfaces[3].coating = Some("ID".into());
    s.surfaces[5].coating = Some("TAB".into());
    s
}

#[test]
fn energy_is_conserved_along_real_rays() {
    // For every interface of real rays (all fields, wavelengths, pupil
    // points) T + R + A = 1 per polarisation, and the polarised budget
    // input = transmitted + Σ diverted + Σ absorbed + bulk closes.
    let mut systems = vec![
        (coated_triplet(), MaterialLibrary::with_builtin()),
        (samples::double_gauss(), MaterialLibrary::with_builtin()),
        (
            singlet(
                MaterialRef::glass("ABS-BK7"),
                Some(thin_film(vec![qw_layer(1.38)], None)),
            ),
            absorbing_library(),
        ),
    ];
    let mut newt = samples::newtonian_telescope();
    newt.coatings.insert(
        "AL".into(),
        thin_film(vec![qw_layer(1.46)], Some((0.96, 6.69))),
    );
    newt.surfaces[1].coating = Some("AL".into());
    newt.surfaces[3].coating = Some("AL".into());
    systems.push((newt, MaterialLibrary::with_builtin()));
    let states = [
        PolarizationState::Unpolarized,
        jones_state(JonesVector::new(C::new(0.6, 0.1), C::new(-0.3, 0.7))),
        PolarizationState::Stokes {
            s: Stokes {
                s0: 1.0,
                s1: -0.4,
                s2: 0.3,
                s3: 0.5,
            },
        },
    ];
    let pts = mklight_trace::pupil::hexapolar(3);
    let mut checked = 0;
    for (sys, lib) in systems {
        let t = tracer_lib(sys, lib);
        for f in t.fields() {
            for wl in 0..t.num_wavelengths() {
                for p in &pts {
                    let ray = t.trace_polarized(f, wl, p[0], p[1], &TraceOptions::unclipped());
                    if !ray.is_ok() {
                        continue;
                    }
                    for st in &ray.steps {
                        let (ps, pp) = st.path_sp();
                        let (ds, dp) = st.diverted_sp();
                        let (a_s, a_p) = (1.0 - ps - ds, 1.0 - pp - dp);
                        assert!(a_s > -1e-13 && a_p > -1e-13, "{st:?}");
                        // Same numbers as the power API.
                        let s = &t.system().surfaces[st.surface];
                        let rt = s.coating.rt(
                            t.system().wavelengths_um[wl],
                            t.system().n_before(st.surface, wl),
                            s.n_after[wl],
                            st.cos_incidence,
                            s.is_mirror(),
                            mklight_core::materials::coating::Arrival::Before,
                        );
                        assert!(rt_diff(&rt, &st.power) < 1e-12, "{rt:?} {:?}", st.power);
                        // Amplitudes reproduce the shared power coefficients…
                        let ap = st.amplitudes.power();
                        assert!(rt_diff(&ap, &rt) < 1e-12, "{ap:?} {rt:?}");
                        // …and the step's PRT matrix carries exactly the path
                        // fraction of pure s and p light.
                        let n2 = |v: [C; 3]| v.iter().map(|c| c.norm_sqr()).sum::<f64>();
                        let p_in = st.k_in.cross(st.s);
                        assert!((n2(st.prt.apply(st.s)) - ps).abs() < 1e-12);
                        assert!((n2(st.prt.apply(p_in)) - pp).abs() < 1e-12);
                        if let ResolvedCoating::ThinFilm { layers, .. } = &s.coating
                            && !s.is_mirror()
                        {
                            // Lossless films conserve exactly; absorbing ones absorb.
                            let lossless = layers.iter().all(|l| l.extinction == 0.0);
                            assert_eq!(a_s.abs() < 1e-12, lossless, "surface {}", st.surface);
                        }
                    }
                    for state in &states {
                        // `path` comes from the propagated PRT coherency
                        // matrix, `diverted`/`absorbed` from the s/p powers
                        // and `rt`, so none of the checks below holds by
                        // construction.
                        let b = ray.energy_budget(state);
                        assert!(b.residual().abs() < 1e-12, "{b:?}");
                        // `intensity` is a fraction of the input power S₀.
                        let th = ray.intensity(state);
                        assert!((b.transmitted - th * state.stokes().s0).abs() < 1e-13);
                        let (mut prev_path, mut prev_tau) = (b.input, ray.records[0].transmittance);
                        for (l, st) in b.surfaces.iter().zip(&ray.steps) {
                            assert_eq!(l.surface, st.surface);
                            assert!(l.absorbed > -1e-13 && l.diverted >= 0.0 && l.path >= 0.0);
                            // Transverse field: the s and p powers add up.
                            let ws_wp = l.incident_s + l.incident_p;
                            assert!((ws_wp - l.incident).abs() < 1e-13, "{l:?}");
                            // T + R + A = 1 per polarisation, applied to the
                            // incident s / p powers, closes at every surface.
                            let close = l.incident - l.path - l.diverted - l.absorbed;
                            assert!(close.abs() < 1e-12, "{l:?}");
                            // Only bulk absorption (Beer–Lambert from the
                            // scalar trace) lies between interacting surfaces.
                            let tau = ray.records[st.surface].transmittance;
                            assert!(l.incident <= prev_path + 1e-15);
                            assert!((l.incident - prev_path * tau / prev_tau).abs() < 1e-13);
                            (prev_path, prev_tau) = (l.path, tau);
                        }
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 500, "{checked}");
}

#[test]
fn lossless_plate_closes_with_all_reflected_orders() {
    // Tilted, AR-coated lossless plate. The primary path transmits T₁T₂; the
    // incoherent sum over all multiply reflected orders gives
    // T_tot = T₁T₂/(1 − R₂R₁') and R_tot = R₁ + T₁T₁'R₂/(1 − R₂R₁'), with the
    // primed coefficients for light inside the plate, from the coating code.
    // R_tot + T_tot = 1 per polarisation.
    let coat = thin_film(vec![qw_layer(2.0), qw_layer(1.45)], None);
    let t = tracer(tilted_plate(
        MaterialRef::glass("N-BK7"),
        35.0,
        Some(coat.clone()),
    ));
    let ray = on_axis(&t, 0.0, 0.0);
    assert_eq!(ray.steps.len(), 2);
    let sys = t.system();
    let ng = sys.surfaces[1].n_after[0];
    let c = &sys.surfaces[1].coating;
    let (st1, st2) = (&ray.steps[0], &ray.steps[1]);
    // Snell inside the plate.
    let sin_in = (1.0 - st2.cos_incidence.powi(2)).sqrt();
    assert!((ng * sin_in - 35f64.to_radians().sin()).abs() < 1e-14);
    // Surface 1's film seen from inside the glass, as on surface 2.
    let back = c.rt(
        D_LINE,
        ng,
        1.0,
        st2.cos_incidence,
        false,
        mklight_core::materials::coating::Arrival::After,
    );
    for (pol, x) in [
        ("s", JonesVector::linear(0.0)),
        ("p", JonesVector::linear(FRAC_PI_2)),
    ] {
        let pick = |rt: mklight_core::materials::coating::InterfaceRT| {
            if pol == "s" {
                (rt.rs, rt.ts)
            } else {
                (rt.rp, rt.tp)
            }
        };
        let (r1, t1) = pick(st1.power);
        let (r2, t2) = pick(st2.power);
        let (r1b, t1b) = pick(back);
        // Primary path from polarisation tracing equals T₁T₂.
        let th = ray.intensity(&jones_state(x));
        assert!(rel(th, t1 * t2) < 1e-13, "{pol}");
        // Reciprocity of the front-face stack: light inside the plate sees
        // the same lossless coating from the glass side (reversed layer
        // order, other incident medium) and must have T₁' = T₁, R₁' = R₁.
        assert!(
            (t1b - t1).abs() < 1e-14 && (r1b - r1).abs() < 1e-14,
            "{pol}"
        );
        assert!((r1 + t1 - 1.0).abs() < 1e-14 && (r2 + t2 - 1.0).abs() < 1e-14);
        // Incoherent multiple reflections, summed order by order: the k-th
        // transmitted order is T₁(R₂R₁')ᵏT₂, the k-th reflected one
        // (k ≥ 1) T₁R₂(R₁'R₂)^(k−1)T₁'.
        let (mut t_sum, mut r_sum) = (0.0, r1);
        let mut w = t1; // power entering the plate in the current order
        for _ in 0..200 {
            t_sum += w * t2;
            r_sum += w * r2 * t1b;
            w *= r2 * r1b;
        }
        let geom = 1.0 / (1.0 - r2 * r1b);
        let t_tot = t1 * t2 * geom;
        let r_tot = r1 + t1 * t1b * r2 * geom;
        assert!(
            rel(t_sum, t_tot) < 1e-14 && rel(r_sum, r_tot) < 1e-13,
            "{pol}"
        );
        assert!(
            (t_sum + r_sum - 1.0).abs() < 1e-14,
            "{pol}: {t_sum} + {r_sum}"
        );
        // The primary path misses exactly the light in the higher orders.
        assert!(th < t_tot && t_tot - th < 2.0 * r1 * r2);
    }
    let p = ray.properties();
    let (ts, tp) = (st1.power.ts * st2.power.ts, st1.power.tp * st2.power.tp);
    assert!((p.diattenuation - (ts - tp).abs() / (ts + tp)).abs() < 1e-14);
}

#[test]
fn brewster_plate_malus_type_law() {
    // A plate of n = 1.5 at Brewster's angle transmits p completely at both
    // faces. Linear input at angle θ from p gives
    // I(θ) = T_p cos²θ + T_s sin²θ with T_p = 1, T_s = (1 − r_s²)².
    let n: f64 = 1.5;
    let tb = n.atan().to_degrees();
    let t = tracer(tilted_plate(MaterialRef::Constant { n }, tb, None));
    let ray = on_axis(&t, 0.0, 0.0);
    for st in &ray.steps {
        assert!((st.power.tp - 1.0).abs() < 1e-14 && st.power.rp < 1e-28);
    }
    let rs = fresnel(1.0, n, tb.to_radians().cos()).rs;
    let ts = (1.0 - rs) * (1.0 - rs);
    for i in 0..=24 {
        let th = i as f64 * PI / 24.0;
        // p = ŷ in the input double-pole basis.
        let e = JonesVector::linear(FRAC_PI_2 - th);
        let got = ray.intensity(&jones_state(e));
        let want = th.cos().powi(2) + ts * th.sin().powi(2);
        assert!((got - want).abs() < 1e-14, "{th}: {got} vs {want}");
    }
    let p = ray.properties();
    assert!((p.diattenuation - (1.0 - ts) / (1.0 + ts)).abs() < 1e-14);
    assert!(p.retardance < 1e-14, "{p:?}");
    // Best-transmitted state is p = ŷ: Stokes S₁ = −1.
    assert!((p.diattenuation_axis[0] + 1.0).abs() < 1e-14);
}

#[test]
fn metal_fold_mirror_retardance() {
    // 45° fold on aluminium (N = 0.96 − 6.69i). The proper Jones matrix is
    // diag(−r_s, r_p) in (s, p): retardance |arg(r_p/(−r_s))|, diattenuation
    // (|r_s|² − |r_p|²)/(|r_s|² + |r_p|²), from the closed-form complex
    // Fresnel equations.
    let al = thin_film(vec![], Some((0.96, 6.69)));
    let t = tracer(fold_mirror(Some(al)));
    let ray = on_axis(&t, 0.0, 0.0);
    // The stop (air → air) is a lossless refracting step; the mirror the other.
    assert_eq!(ray.steps.len(), 2);
    assert_eq!(ray.steps[1].kind, StepKind::Reflect);
    assert!((ray.steps[1].cos_incidence - 45f64.to_radians().cos()).abs() < 1e-15);
    assert!(
        jones_diff(
            &ray.steps[0]
                .prt
                .jones((Vec3::X, Vec3::Y), (Vec3::X, Vec3::Y)),
            &JonesMatrix::IDENTITY
        ) < 1e-15
    );
    // Image space travels toward −z of the image frame.
    assert!(ray.trace.ray.dir.z < -0.999_999);
    let nm = C::new(0.96, -6.69);
    let (s, c) = 45f64.to_radians().sin_cos();
    let mut nc = (nm * nm - C::new(s * s, 0.0)).sqrt();
    if nc.im > 0.0 {
        nc = -nc;
    }
    let rs = (c - nc) / (c + nc);
    let rp = (nm * c - nc / nm) / (nm * c + nc / nm);
    let ret = (rp / (-rs)).arg();
    let (rs2, rp2) = (rs.norm_sqr(), rp.norm_sqr());
    let p = ray.properties();
    assert!(
        (p.retardance - ret.abs()).abs() < 1e-12,
        "{} vs {ret}",
        p.retardance
    );
    assert!(ret.to_degrees() < -5.0, "{}", ret.to_degrees());
    assert!((p.diattenuation - (rs2 - rp2) / (rs2 + rp2)).abs() < 1e-13);
    assert!((p.t_unpolarized - 0.5 * (rs2 + rp2)).abs() < 1e-13);
    // p lags (negative phase), so s = x̂ (Stokes S₁ = +1) is the fast axis
    // and the better-reflected state.
    assert!((p.fast_axis[0] - 1.0).abs() < 1e-12, "{p:?}");
    assert!((p.diattenuation_axis[0] - 1.0).abs() < 1e-12);
    // Jones matrix in the double-pole bases (odd-mirror image pole −z):
    // still diagonal for this meridional fold, with the same moduli.
    let j = ray.jones();
    assert!(j.m[0][1].norm() < 1e-14 && j.m[1][0].norm() < 1e-14);
    assert!((j.m[0][0].norm() - rs.norm()).abs() < 1e-14);
    assert!((j.m[1][1].norm() - rp.norm()).abs() < 1e-14);
    // A bare (perfect) mirror is non-polarising: proper Jones = identity.
    let t = tracer(fold_mirror(None));
    let ray = on_axis(&t, 0.0, 0.0);
    assert!(jones_diff(&ray.proper_jones(), &JonesMatrix::IDENTITY) < 1e-15);
    assert!((ray.jones().det().norm() - 1.0).abs() < 1e-15);
}

#[test]
fn ideal_lens_is_lossless_and_non_polarizing() {
    let mut s = infinite_system(
        20.0,
        vec![
            surf(1, SurfaceShape::Plane, 5.0, MaterialRef::Air),
            surf(
                2,
                SurfaceShape::Paraxial { focal_length: 50.0 },
                50.0,
                MaterialRef::Air,
            ),
        ],
        3,
    );
    s.fields = angle_fields(&[0.0, 10.0]);
    let t = tracer(s);
    for f in t.fields() {
        for p in mklight_trace::pupil::hexapolar(3) {
            let ray = t.trace_polarized(f, 0, p[0], p[1], &TraceOptions::unclipped());
            assert!(ray.is_ok());
            assert!(jones_diff(&ray.proper_jones(), &JonesMatrix::IDENTITY) < 1e-14);
            assert!((ray.intensity(&PolarizationState::Unpolarized) - 1.0).abs() < 1e-14);
            assert!(
                ray.prt
                    .apply(ray.k_in)
                    .iter()
                    .zip([ray.k_out.x, ray.k_out.y, ray.k_out.z])
                    .all(|(a, b)| (a - C::new(b, 0.0)).norm() < 1e-14)
            );
        }
    }
}

#[test]
fn jones_pupil_is_rotationally_covariant() {
    // On-axis rotationally symmetric lens, object at infinity: the
    // double-pole bases turn rigidly with the pupil azimuth, so
    // J(r, φ) = R(φ) J(r, 0) R(−φ), and the uncoated lens transmits the
    // radial (p) state best: diattenuation axis (cos 2φ, sin 2φ, 0).
    let t = tracer(singlet(
        MaterialRef::glass("N-BK7"),
        Some(thin_film(vec![qw_layer(2.1), qw_layer(1.38)], None)),
    ));
    let t_bare = tracer(singlet(MaterialRef::glass("N-BK7"), None));
    for rr in [0.3, 0.7, 1.0] {
        let j0 = on_axis(&t, rr, 0.0).jones();
        for k in 1..12 {
            let phi = k as f64 * PI / 6.0 + 0.1;
            let ray = on_axis(&t, rr * phi.cos(), rr * phi.sin());
            let want = rot2(phi).mul(&j0).mul(&rot2(-phi));
            assert!(jones_diff(&ray.jones(), &want) < 1e-13, "r={rr} φ={phi}");
            let b = on_axis(&t_bare, rr * phi.cos(), rr * phi.sin()).properties();
            let ax = [(2.0 * phi).cos(), (2.0 * phi).sin(), 0.0];
            for i in 0..3 {
                assert!((b.diattenuation_axis[i] - ax[i]).abs() < 1e-9, "{b:?}");
            }
        }
    }
    // Continuity through the pupil centre (normal incidence at both faces).
    let j0 = on_axis(&t, 0.0, 0.0).jones();
    for e in 3..=12 {
        let eps = 10f64.powi(-e);
        for phi in [0.0, 1.0, 2.5, 4.0] {
            let j = on_axis(&t, eps * f64::cos(phi), eps * f64::sin(phi)).jones();
            assert!(jones_diff(&j, &j0) < 1e-2 * eps + 1e-15, "ε={eps}");
        }
    }
}

#[test]
fn vector_pupil_fields_are_transverse_and_consistent() {
    let t = tracer_lib(
        singlet(
            MaterialRef::glass("ABS-BK7"),
            Some(thin_film(vec![qw_layer(1.38)], None)),
        ),
        absorbing_library(),
    );
    let pts = mklight_trace::pupil::grid(9);
    let f = t.field(1).clone();
    let ctx = JobContext::new();
    let vp = vector_pupil(&t, &f, 0, &pts, &TraceOptions::unclipped(), &ctx).unwrap();
    assert_eq!(vp.len(), pts.len());
    assert_eq!(ctx.progress.fraction(), 1.0);
    let e = JonesVector::new(C::new(0.3, -0.2), C::new(0.5, 0.6));
    for s in &vp {
        assert!(s.status.is_ok());
        let k = s.dir;
        let field = s.field(&e);
        let kd: C = field[0] * k.x + field[1] * k.y + field[2] * k.z;
        assert!(kd.norm() < 1e-14);
        // |E|² equals the polarised intensity (bulk absorption included).
        let ray = t.trace_polarized(&f, 0, s.pupil[0], s.pupil[1], &TraceOptions::unclipped());
        let i = mklight_trace::polarization::field_intensity(&field);
        // `intensity` is normalised by the input power |e|² = 0.74.
        assert!(rel(i, ray.intensity(&jones_state(e)) * e.intensity()) < 1e-13);
        assert!(rel(s.jones.apply(&e).intensity(), i) < 1e-13);
    }
    // Cancellation.
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(
        vector_pupil(&t, &f, 0, &pts, &TraceOptions::unclipped(), &ctx).unwrap_err(),
        MkError::Cancelled
    );
}

#[test]
fn system_transmission_is_deterministic_and_consistent() {
    let mut sys = coated_triplet();
    sys.wavelengths = mklight_core::WavelengthSet::default();
    let t = tracer(sys);
    let base = TransmissionSettings {
        pupil: PupilSamples::Random {
            count: 400,
            seed: 0x5eed,
        },
        ..Default::default()
    };
    let ctx = JobContext::new();
    let a = system_transmission(&t, &base, &ctx).unwrap();
    assert_eq!(ctx.progress.fraction(), 1.0);
    let b = system_transmission(&t, &base, &JobContext::new()).unwrap();
    assert_eq!(a.points, b.points);
    assert_eq!(a.run_info.seed, Some(0x5eed));
    assert_eq!(a.run_info.analysis, "system_transmission");
    assert_eq!(a.points.len(), t.fields().len() * t.num_wavelengths());
    // Seeded points are uniform on the disk and reproducible.
    let pts = base.pupil.points();
    assert_eq!(pts, base.pupil.points());
    assert!(pts.iter().all(|p| p[0].hypot(p[1]) < 1.0));
    let mean_r2 = pts.iter().map(|p| p[0] * p[0] + p[1] * p[1]).sum::<f64>() / pts.len() as f64;
    assert!((mean_r2 - 0.5).abs() < 0.05);
    // The exact unpolarised mean differs from the incoherent model only at
    // second order; the unvignetted mean ≥ the vignetted transmission.
    let exact = system_transmission(
        &t,
        &TransmissionSettings {
            model: unpol(),
            ..base.clone()
        },
        &JobContext::new(),
    )
    .unwrap();
    for (x, y) in a.points.iter().zip(&exact.points) {
        assert!(
            (x.transmission - y.transmission).abs() < 2e-4,
            "{x:?} {y:?}"
        );
        assert!(x.mean_unvignetted >= x.transmission);
        assert!(x.transmission > 0.5 && x.transmission < 1.0);
    }
    // Polychromatic value is the spectral-weight average.
    let w = &t.system().wavelength_weights;
    let num: f64 = (0..3).map(|k| w[k] * a.at(0, k).transmission).sum();
    assert!((a.polychromatic[0] - num / w.iter().sum::<f64>()).abs() < 1e-15);
    // Unclipped on-axis transmission with one sample equals the chief ray.
    let one = system_transmission(
        &t,
        &TransmissionSettings {
            fields: vec![FieldPoint::normalized(0.0, 0.0)],
            wavelengths: vec![1],
            pupil: PupilSamples::Grid { n: 1 },
            clip: false,
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap();
    let f0 = t.prepare_field(FieldPoint::normalized(0.0, 0.0)).unwrap();
    let chief = t.throughput(
        &f0,
        1,
        0.0,
        0.0,
        &TraceOptions::unclipped(),
        &RadiometryModel::Unpolarized,
    );
    assert_eq!(one.points[0].transmission, chief.total);
    // Errors and cancellation.
    let bad = TransmissionSettings {
        wavelengths: vec![7],
        ..base.clone()
    };
    assert!(system_transmission(&t, &bad, &JobContext::new()).is_err());
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert_eq!(
        system_transmission(&t, &base, &ctx).unwrap_err(),
        MkError::Cancelled
    );
}

#[test]
fn polarization_pupil_map_matches_transmission() {
    let t = tracer(coated_triplet());
    let state = jones_state(JonesVector::linear(0.4));
    let settings = PolarizationMapSettings {
        field: FieldPoint::normalized(0.0, 0.7),
        wavelength: 1,
        grid: 11,
        state,
        clip: true,
    };
    let map = polarization_pupil_map(&t, &settings, &JobContext::new()).unwrap();
    assert_eq!(map.run_info.analysis, "polarization_pupil_map");
    let tr = system_transmission(
        &t,
        &TransmissionSettings {
            fields: vec![settings.field],
            wavelengths: vec![1],
            pupil: PupilSamples::Grid { n: 11 },
            model: RadiometryModel::Polarized { state },
            clip: true,
        },
        &JobContext::new(),
    )
    .unwrap();
    assert!(rel(map.mean_intensity, tr.points[0].mean_unvignetted) < 1e-13);
    assert!(map.max_diattenuation > 1e-4 && map.max_diattenuation < 0.2);
    for s in map.samples.iter().filter(|s| s.status.is_ok()) {
        // Map Jones / intensity / unpolarised transmittance agree.
        assert!(rel(s.jones.intensity_of(&state), s.intensity) < 1e-13);
        assert!(rel(s.properties.t_unpolarized, s.jones.t_unpolarized()) < 1e-12);
        // Proper and pupil-basis Jones share the diattenuation.
        assert!((s.properties.diattenuation - s.jones.properties().diattenuation).abs() < 1e-12);
        assert!(s.properties.retardance <= PI && s.retardance_jones <= PI);
    }
    // The thin films retard (s and p phases differ off normal incidence).
    assert!(map.max_retardance > 1e-6);
}

#[test]
fn serde_roundtrip_of_polarization_types() {
    let j = JonesMatrix {
        m: [
            [C::new(0.1, -0.2), C::new(0.3, 0.4)],
            [C::new(-0.5, 0.6), C::new(0.7, -0.8)],
        ],
    };
    let s = serde_json::to_string(&j).unwrap();
    assert_eq!(s, "[[[0.1,-0.2],[0.3,0.4]],[[-0.5,0.6],[0.7,-0.8]]]");
    assert_eq!(serde_json::from_str::<JonesMatrix>(&s).unwrap(), j);
    let st = PolarizationState::Jones {
        e: JonesVector::linear(0.25),
    };
    let s = serde_json::to_string(&st).unwrap();
    assert_eq!(serde_json::from_str::<PolarizationState>(&s).unwrap(), st);
    let m = RadiometryModel::Polarized {
        state: PolarizationState::Unpolarized,
    };
    let s = serde_json::to_string(&m).unwrap();
    assert_eq!(serde_json::from_str::<RadiometryModel>(&s).unwrap(), m);
}

// ---------------------------------------------------------------------------
// Review regressions.

#[test]
fn near_polarizer_retardance_accuracy_and_undefined_retarder() {
    // J = R(θ) diag(1, ε e^{0.7i}) R(−θ): retardance 0.7 for every ε > 0.
    // The retarder is ill-conditioned as ε → 0: rounding of J (≈ 1e-16)
    // moves δ by about 1e-16/ε (PolarizationProperties::retardance docs).
    let delta = 0.7;
    for th in [0.0, 0.3, 1.1] {
        for k in 1..=11 {
            let eps = 10f64.powi(-k);
            let d = JonesMatrix {
                m: [
                    [C::new(1.0, 0.0), C::new(0.0, 0.0)],
                    [C::new(0.0, 0.0), C::from_polar(eps, delta)],
                ],
            };
            let p = rot2(th).mul(&d).mul(&rot2(-th)).properties();
            assert!(p.retarder_defined, "θ={th} ε={eps:e}");
            let bound = 1e-16 / eps * 50.0 + 1e-14;
            assert!(
                (p.retardance - delta).abs() < bound,
                "θ={th} ε={eps:e}: {} (bound {bound:e})",
                p.retardance
            );
        }
        // Beyond T_min/T_max = 1e-24 the retarder is flagged undefined
        // (not silently reported as a zero retardance of a valid retarder).
        let d = JonesMatrix {
            m: [
                [C::new(1.0, 0.0), C::new(0.0, 0.0)],
                [C::new(0.0, 0.0), C::from_polar(1e-13, delta)],
            ],
        };
        let p = rot2(th).mul(&d).mul(&rot2(-th)).properties();
        assert!(!p.retarder_defined && p.retardance == 0.0, "{p:?}");
        assert!((p.diattenuation - 1.0).abs() < 1e-15);
    }
    // A zero (blocked) Jones matrix has no retarder either.
    let z = JonesMatrix::IDENTITY.scale(0.0).properties();
    assert!(!z.retarder_defined && z.t_max == 0.0);
    // Old serialised properties without the flag read as defined.
    let mut v = serde_json::to_value(JonesMatrix::IDENTITY.properties()).unwrap();
    v.as_object_mut().unwrap().remove("retarder_defined");
    let p: mklight_trace::PolarizationProperties = serde_json::from_value(v).unwrap();
    assert!(p.retarder_defined);
}

/// Cooke triplet whose stop-side surface 3 is clipped to 3 mm.
fn vignetted_triplet() -> OpticalSystem {
    let mut s = samples::cooke_triplet();
    s.surfaces[3].semi_diameter = mklight_core::SemiDiameter::Fixed { value: 3.0 };
    s
}

#[test]
fn vignetted_vector_pupil_samples_carry_no_light() {
    let t = tracer(vignetted_triplet());
    let opts = t.options();
    assert!(opts.clip);
    let f = t.field(0).clone();
    let pts = [[0.0, 0.0], [0.0, 0.9], [0.0, -0.95], [0.9, 0.0], [0.2, 0.1]];
    let vp = vector_pupil(&t, &f, 1, &pts, &opts, &JobContext::new()).unwrap();
    let mut blocked = 0;
    for s in &vp {
        if s.status.is_ok() {
            assert!(s.jones.t_unpolarized() > 0.5, "{s:?}");
            assert!((s.dir.norm() - 1.0).abs() < 1e-14);
            continue;
        }
        blocked += 1;
        assert!(
            matches!(s.status, mklight_trace::TraceStatus::Clipped { .. }),
            "{:?}",
            s.status
        );
        assert_eq!(s.jones, JonesMatrix::IDENTITY.scale(0.0));
        assert_eq!(s.field_x, [[0.0; 2]; 3]);
        assert_eq!(s.field_y, [[0.0; 2]; 3]);
        assert_eq!(s.dir, Vec3::ZERO);
        let e = s.field(&JonesVector::linear(0.3));
        assert_eq!(mklight_trace::polarization::field_intensity(&e), 0.0);
    }
    assert!(blocked >= 2, "{vp:?}");
    // The pupil map agrees: the same samples are blocked and zero.
    let map = polarization_pupil_map(
        &t,
        &PolarizationMapSettings {
            field: FieldPoint::normalized(0.0, 0.0),
            wavelength: 1,
            grid: 9,
            state: PolarizationState::Unpolarized,
            clip: true,
        },
        &JobContext::new(),
    )
    .unwrap();
    let grid = mklight_trace::pupil::grid(9);
    let vp = vector_pupil(&t, &f, 1, &grid, &opts, &JobContext::new()).unwrap();
    assert!(map.samples.iter().any(|s| !s.status.is_ok()));
    for (m, v) in map.samples.iter().zip(&vp) {
        assert_eq!(m.status, v.status);
        if !m.status.is_ok() {
            assert_eq!(m.jones, v.jones);
            assert_eq!(m.intensity, 0.0);
        } else {
            assert!(jones_diff(&m.jones, &v.jones) == 0.0);
        }
    }
}

#[test]
fn intensity_is_normalized_and_states_are_validated() {
    let t = tracer(coated_triplet());
    let f = t.field(1).clone();
    let opts = TraceOptions::unclipped();
    let ray = t.trace_polarized(&f, 1, 0.3, -0.5, &opts);
    assert!(ray.is_ok());
    let unit = Stokes {
        s0: 1.0,
        s1: 0.3,
        s2: -0.5,
        s3: 0.4,
    };
    let twice = Stokes {
        s0: 2.0,
        s1: 0.6,
        s2: -1.0,
        s3: 0.8,
    };
    let (a, b) = (
        PolarizationState::Stokes { s: unit },
        PolarizationState::Stokes { s: twice },
    );
    // A power fraction: independent of the input power, and equal to the
    // radiometry API's throughput.
    let (ia, ib) = (ray.intensity(&a), ray.intensity(&b));
    assert!(rel(ia, ib) < 1e-15 && ia > 0.5 && ia < 1.0);
    let model = RadiometryModel::Polarized { state: b };
    let th = mklight_trace::RayThroughput::of(&ray, &model);
    assert!(rel(th.total, ib) < 1e-15);
    let e = JonesVector::new(C::new(2.0, 1.0), C::new(-1.0, 0.5));
    let en = JonesVector::new(e.x / e.intensity().sqrt(), e.y / e.intensity().sqrt());
    assert!(
        rel(
            ray.intensity(&jones_state(e)),
            ray.intensity(&jones_state(en))
        ) < 1e-14
    );

    // Unphysical states are rejected by both jobs.
    let bad = [
        PolarizationState::Stokes {
            s: Stokes {
                s0: 1.0,
                s1: 2.0,
                s2: 0.0,
                s3: 0.0,
            },
        },
        PolarizationState::Stokes {
            s: Stokes {
                s0: f64::NAN,
                ..Stokes::UNPOLARIZED
            },
        },
        PolarizationState::Stokes {
            s: Stokes {
                s0: 0.0,
                ..Stokes::UNPOLARIZED
            },
        },
        PolarizationState::Stokes {
            s: Stokes {
                s3: f64::INFINITY,
                ..Stokes::UNPOLARIZED
            },
        },
        jones_state(JonesVector::new(C::new(0.0, 0.0), C::new(0.0, 0.0))),
        jones_state(JonesVector::new(C::new(f64::NAN, 0.0), C::new(1.0, 0.0))),
    ];
    for state in bad {
        assert!(state.validate().is_err(), "{state:?}");
        let r = system_transmission(
            &t,
            &TransmissionSettings {
                pupil: PupilSamples::Grid { n: 3 },
                model: RadiometryModel::Polarized { state },
                ..Default::default()
            },
            &JobContext::new(),
        );
        assert!(matches!(r, Err(MkError::InvalidModel(_))), "{state:?}");
        let r = polarization_pupil_map(
            &t,
            &PolarizationMapSettings {
                field: FieldPoint::normalized(0.0, 0.0),
                wavelength: 0,
                grid: 3,
                state,
                clip: true,
            },
            &JobContext::new(),
        );
        assert!(matches!(r, Err(MkError::InvalidModel(_))), "{state:?}");
    }
    // Fully polarised states (DOP = 1 up to rounding) are accepted.
    for state in [
        PolarizationState::Unpolarized,
        a,
        jones_state(e),
        PolarizationState::Stokes { s: e.stokes() },
    ] {
        state.validate().unwrap();
    }
}

#[test]
fn empty_pupil_samplings_are_rejected() {
    let t = tracer(singlet(MaterialRef::glass("N-BK7"), None));
    assert!(mklight_trace::pupil::grid(2).is_empty());
    for pupil in [
        PupilSamples::Grid { n: 0 },
        PupilSamples::Grid { n: 2 },
        PupilSamples::Random { count: 0, seed: 1 },
    ] {
        let r = system_transmission(
            &t,
            &TransmissionSettings {
                pupil,
                ..Default::default()
            },
            &JobContext::new(),
        );
        assert!(matches!(r, Err(MkError::InvalidModel(_))), "{pupil:?}");
    }
    for grid in [0, 2] {
        let r = polarization_pupil_map(
            &t,
            &PolarizationMapSettings {
                field: FieldPoint::normalized(0.0, 0.0),
                wavelength: 0,
                grid,
                state: PolarizationState::Unpolarized,
                clip: true,
            },
            &JobContext::new(),
        );
        assert!(matches!(r, Err(MkError::InvalidModel(_))), "{grid}");
    }
    // n = 1 (the pupil centre) and n = 3 are valid.
    for n in [1, 3] {
        let r = system_transmission(
            &t,
            &TransmissionSettings {
                pupil: PupilSamples::Grid { n },
                ..Default::default()
            },
            &JobContext::new(),
        )
        .unwrap();
        assert!(
            r.points
                .iter()
                .all(|p| p.samples > 0 && p.transmission > 0.9)
        );
    }
}

#[test]
fn unpolarized_fast_path_equals_prt_path() {
    // The scalar model is evaluated from the surface records without PRT
    // data; it must equal the value derived from the full polarised ray
    // bit for bit (same factors, same order), for refracting surfaces,
    // coated/metal mirrors, ideal lenses and blocked rays.
    let mut newt = samples::newtonian_telescope();
    newt.coatings.insert(
        "AL".into(),
        thin_film(vec![qw_layer(1.46)], Some((0.96, 6.69))),
    );
    newt.surfaces[1].coating = Some("AL".into());
    let mut ideal = infinite_system(
        20.0,
        vec![
            surf(1, SurfaceShape::Plane, 5.0, MaterialRef::Air),
            surf(
                2,
                SurfaceShape::Paraxial { focal_length: 50.0 },
                50.0,
                MaterialRef::Air,
            ),
        ],
        3,
    );
    ideal.fields = angle_fields(&[0.0, 10.0]);
    let systems = [
        (coated_triplet(), MaterialLibrary::with_builtin()),
        (vignetted_triplet(), MaterialLibrary::with_builtin()),
        (newt, MaterialLibrary::with_builtin()),
        (ideal, MaterialLibrary::with_builtin()),
        (
            singlet(MaterialRef::glass("ABS-BK7"), None),
            absorbing_library(),
        ),
    ];
    let (mut checked, mut blocked) = (0, 0);
    for (sys, lib) in systems {
        let t = tracer_lib(sys, lib);
        let opts = t.options();
        for f in t.fields() {
            for wl in 0..t.num_wavelengths() {
                for p in mklight_trace::pupil::hexapolar(4) {
                    let fast =
                        t.throughput(f, wl, p[0], p[1], &opts, &RadiometryModel::Unpolarized);
                    let ray = t.trace_polarized(f, wl, p[0], p[1], &opts);
                    let full =
                        mklight_trace::RayThroughput::of(&ray, &RadiometryModel::Unpolarized);
                    assert_eq!(fast, full);
                    if ray.is_ok() {
                        let prod: f64 = ray.steps.iter().map(|s| s.path_unpolarized()).product();
                        assert_eq!(fast.interfaces, prod);
                        assert!(fast.total > 0.0 && fast.total <= 1.0);
                    } else {
                        blocked += 1;
                    }
                    checked += 1;
                }
            }
        }
    }
    assert!(checked > 300 && blocked > 0, "{checked} {blocked}");
}
