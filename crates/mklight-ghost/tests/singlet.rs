//! Singlet ghost: paraxial focus against a hand-derived vergence
//! calculation (the inner reflections as mirrors in glass), and the real
//! ghost focus converging to it as the aperture shrinks.

use crate::common;

use common::*;
use mklight_core::samples;
use mklight_ghost::paraxial::trace_steps;
use mklight_ghost::{
    GhostPath, GhostSettings, ParaxialGhostSettings, paraxial_ghosts, real_ghost_focus,
};

/// Reduced vergence `V = n/s` after propagating `d` (along the light) in
/// index `n`: `s → s − d`.
fn transfer(v: f64, d: f64, n: f64) -> f64 {
    v / (1.0 - d * v / n)
}

#[test]
fn singlet_ghost_focus_matches_mirror_equivalent() {
    let sys = samples::singlet();
    let (c1, c2) = (1.0 / 60.0, -1.0 / 350.0);
    let (t, bfl) = (sys.surfaces[1].thickness, sys.surfaces[2].thickness);
    let m = model(sys, GhostSettings::default());
    let rs = m.system();
    let wl = rs.primary_wavelength;
    let n = rs.surfaces[1].n_after[wl];
    let r_ep = m.tracer().pupil().r_ep;

    // Path [2, 1]: refract into the glass at S1, reflect at S2 (a mirror of
    // curvature c2 met travelling +z inside glass: power −2n·c2), back to
    // S1, reflect there travelling −z (the curvature seen along the light
    // is −c1: power +2n·c1), forward again and out through S2. Powers of
    // refraction: (n₂ − n₁)·c along the light. Heights follow
    // y' = y(1 − dV/n).
    let mut v = (n - 1.0) * c1;
    let mut y = r_ep;
    for (d, power) in [(t, -2.0 * n * c2), (t, 2.0 * n * c1), (t, (1.0 - n) * c2)] {
        y *= 1.0 - d * v / n;
        v = transfer(v, d, n) + power;
    }
    let s_img = 1.0 / v; // image distance after S2, in air
    let z_expect = s_img - bfl;
    let y_sensor = y * (1.0 - bfl * v);

    let rep = paraxial_ghosts(
        &m,
        &ParaxialGhostSettings {
            paths: Some(vec![GhostPath::new(vec![2, 1])]),
            ..Default::default()
        },
    )
    .unwrap();
    let g = &rep.ghosts[0];
    assert!(
        (g.focus_from_sensor - z_expect).abs() < 1e-9 * z_expect.abs().max(1.0),
        "{} vs {z_expect}",
        g.focus_from_sensor
    );
    assert!(
        (g.spot_diameter - 2.0 * y_sensor.abs()).abs() < 1e-10,
        "{} vs {}",
        g.spot_diameter,
        2.0 * y_sensor.abs()
    );
    // Bare N-BK7: R² at normal incidence, from inside and outside alike.
    let r = r_normal(1.0, n);
    assert!((g.reflectance_product - r * r).abs() < 1e-15);
    assert!((g.transmittance - (1.0 - r).powi(2)).abs() < 1e-14);
    assert!(g.relative_irradiance > 0.0 && g.relative_irradiance < r * r);

    // Real ghost focus → paraxial as the aperture shrinks, with the
    // quadratic convergence of spherical aberration.
    let p = GhostPath::new(vec![2, 1]);
    let e1 = real_ghost_focus(&m, &p, wl, 0.04).unwrap() - g.focus_from_sensor;
    let e2 = real_ghost_focus(&m, &p, wl, 0.02).unwrap() - g.focus_from_sensor;
    assert!(e2.abs() < 1e-3 * g.focus_from_sensor.abs(), "{e2}");
    assert!(((e1 / e2) - 4.0).abs() < 0.05, "{e1} {e2}");
}

#[test]
fn primary_step_trace_reproduces_reference_first_order() {
    for (name, sys) in samples::all()
        .into_iter()
        .filter(|(_, s)| !samples::has_extended_surfaces(s))
    {
        let m = model(sys, GhostSettings::default());
        let rs = m.system();
        let steps = m.check_path(&GhostPath::primary()).unwrap();
        for wl in 0..rs.wavelengths_um.len() {
            let fo = m.tracer().paraxial().at(wl);
            for ray in [&fo.marginal, &fo.chief] {
                let tr = trace_steps(&m, &steps, wl, ray.y[1], ray.nu[0]);
                for (k, st) in steps.iter().enumerate() {
                    let i = st.surface as usize;
                    let scale = ray.y[i].abs().max(1.0);
                    assert!((tr.y[k] - ray.y[i]).abs() < 1e-12 * scale, "{name} y[{i}]");
                    assert!(
                        (tr.nu[k] - ray.nu[i]).abs() < 1e-12 * ray.nu[i].abs().max(1e-3),
                        "{name} nu[{i}]"
                    );
                }
            }
        }
    }
}
