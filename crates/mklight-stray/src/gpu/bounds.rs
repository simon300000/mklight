//! Conservative coefficient bounds, never sampled extrema. The deliberately
//! loose enclosure is used only by the caster; the sag still comes from P1.
use super::*;
use mklight_core::model::surface::ZernikeOrdering;
fn even(a: &[f64], r: f64) -> f64 {
    a.iter()
        .enumerate()
        .map(|(i, a)| a.abs() * r.powi((2 * i + 4) as i32))
        .sum()
}
fn bound(s: &SurfaceShape, r: f64) -> f64 {
    let base =
        |c: f64, k: f64| c.abs() * r * r / (1. + (1. - (1. + k) * c * c * r * r).max(0.).sqrt());
    match s {
        SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::CoordinateBreak => 0.,
        SurfaceShape::Conic { curvature, conic } => base(*curvature, *conic),
        SurfaceShape::EvenAsphere {
            curvature,
            conic,
            coefficients,
        } => base(*curvature, *conic) + even(coefficients, r),
        SurfaceShape::OddAsphere {
            curvature,
            conic,
            coefficients,
        } => {
            base(*curvature, *conic)
                + coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, a)| a.abs() * r.powi((i + 1) as i32))
                    .sum::<f64>()
        }
        SurfaceShape::QCon {
            curvature,
            conic,
            coefficients,
            ..
        } => {
            let mut binomial = 1.;
            let mut sum = 0.;
            for (i, a) in coefficients.iter().enumerate() {
                if i > 0 {
                    binomial *= (i + 4) as f64 / i as f64;
                }
                sum += a.abs() * binomial;
            }
            base(*curvature, *conic) + sum
        }
        SurfaceShape::QBfs {
            curvature,
            norm_radius,
            coefficients,
        } => {
            let (f, g, h) = mklight_core::surface_math::qbfs_recurrence();
            let mut q = [0.; 20];
            let (mut pm, mut pp) = (0., 2.);
            let mut sum = 0.;
            for (i, a) in coefficients.iter().enumerate() {
                let p = match i {
                    0 => 2.,
                    1 => 6.,
                    _ => 2. * pp + pm,
                };
                q[i] =
                    (p + if i >= 1 {
                        g[i - 1].abs() * q[i - 1]
                    } else {
                        0.
                    } + if i >= 2 {
                        h[i - 2].abs() * q[i - 2]
                    } else {
                        0.
                    }) / f[i];
                sum += a.abs() * q[i];
                pm = pp;
                pp = p;
            }
            let radius = r.min(*norm_radius);
            let w = (1. - curvature * curvature * radius * radius).sqrt();
            base(*curvature, 0.) + if sum == 0. { 0. } else { 0.25 * sum / w }
        }
        SurfaceShape::Toroid {
            curvature_x,
            curvature_y,
            conic_y,
            y_asphere,
        } => {
            let y = base(*curvature_y, *conic_y) + even(y_asphere, r);
            let c = curvature_x.abs();
            if c == 0. {
                y
            } else if c * y < 1. {
                y + c * r * r / (1. - c * y)
            } else {
                2. * y + 1. / c
            }
        }
        SurfaceShape::Biconic {
            curvature_x,
            curvature_y,
            x_asphere,
            y_asphere,
            ..
        } => {
            (curvature_x.abs() + curvature_y.abs()) * r * r
                + even(x_asphere, r)
                + even(y_asphere, r)
        }
        SurfaceShape::XYPolynomial {
            curvature,
            conic,
            coefficients,
            ..
        } => base(*curvature, *conic) + coefficients.iter().map(|a| a.abs()).sum::<f64>(),
        SurfaceShape::ZernikeSag {
            curvature,
            conic,
            coefficients,
            ordering,
            ..
        } => {
            base(*curvature, *conic)
                + coefficients
                    .iter()
                    .enumerate()
                    .map(|(j, a)| {
                        let (n, m) =
                            mklight_core::surface_math::zernike_term(*ordering, j + 1).unwrap();
                        a.abs()
                            * if *ordering == ZernikeOrdering::Fringe {
                                1.
                            } else {
                                ((n + 1) as f64 * if m == 0 { 1. } else { 2. }).sqrt()
                            }
                    })
                    .sum::<f64>()
        }
        SurfaceShape::Diffractive { base, .. } => bound(base, r),
        SurfaceShape::SagOverlay {
            base,
            norm_radius,
            test_wavelength_um,
            astigmatism,
            spherical,
            ..
        } => {
            let t = (r / norm_radius).powi(2);
            bound(base, r)
                + test_wavelength_um
                    * 1e-3
                    * (astigmatism.abs() * t + 4. * spherical.abs() * (t + t * t))
        }
    }
}
pub(super) fn local(p: &Primitive) -> MkResult<Aabb> {
    let mut b = p.shape.local_bounds();
    if let Shape::Sag(s) = &p.shape
        && !matches!(
            s.shape(),
            SurfaceShape::Plane | SurfaceShape::Paraxial { .. } | SurfaceShape::Conic { .. }
        )
    {
        let z = bound(s.shape(), s.r_outer());
        if !z.is_finite() {
            return Err(unsupported("sag bounds are unbounded at the domain rim"));
        }
        let pad = 1e-12 * (1. + z);
        b.min.z = -z - pad;
        b.max.z = z + pad;
    }
    Ok(b)
}
