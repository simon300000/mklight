//! Clear-aperture manufacturing metrics, using the core sag/phase derivatives.
use std::f64::consts::TAU;

use mklight_core::math::KahanSum;
use mklight_core::model::{ApertureShape, SemiDiameter, Surface};
use mklight_core::surface_math::{self, SagDerivatives};
use mklight_core::{OpticalSystem, SurfaceId, SurfaceShape};

use super::operand::OperandKind;
use super::{OperandError, OperandValue};

/// The surface of a manufacturing metric operand (`None` for other kinds).
fn metric_surface(k: &OperandKind) -> Option<SurfaceId> {
    use OperandKind::*;
    match k {
        DoeMinZonePeriod { surface }
        | DoeMaxPhaseSlope { surface }
        | DoeTotalPhase { surface }
        | QMaxDeparture { surface }
        | QMaxDepartureSlope { surface }
        | FreeformMaxDeparture { surface } => Some(*surface),
        _ => None,
    }
}

pub(super) fn is_kind(k: &OperandKind) -> bool {
    metric_surface(k).is_some()
}

fn invalid(why: &str) -> OperandError {
    OperandError::Invalid(why.into())
}

/// Radial bounds along one azimuth. An explicit aperture overrides the SD,
/// just as it does in the sequential tracer.
fn radial_bounds(sf: &Surface, cos: f64, sin: f64) -> Result<(f64, f64), OperandError> {
    Ok(match sf.aperture {
        Some(ApertureShape::Circular { radius }) => (0.0, radius),
        Some(ApertureShape::Annular {
            inner_radius,
            outer_radius,
        }) => (inner_radius, outer_radius),
        Some(ApertureShape::Rectangular { half_x, half_y }) => {
            (0.0, (half_x / cos.abs()).min(half_y / sin.abs()))
        }
        Some(ApertureShape::Elliptical { semi_x, semi_y }) => {
            (0.0, 1.0 / (cos / semi_x).hypot(sin / semi_y))
        }
        None => match sf.semi_diameter {
            SemiDiameter::Fixed { value } => (0.0, value),
            SemiDiameter::Auto => {
                return Err(invalid("surface metric needs a clear semi-diameter"));
            }
        },
    })
}

struct Node {
    r2: f64,
    weight: f64,
    sag: SagDerivatives,
    x: f64,
    y: f64,
}

fn nodes(sf: &Surface) -> Result<Vec<Node>, OperandError> {
    let mut azimuths: Vec<_> = (0..64).map(|a| (TAU * a as f64 / 64.0, 1.0)).collect();
    // Rectangular corners can carry the extremum of even a quadratic XY
    // term. Include them explicitly, without changing the fit quadrature.
    if let Some(ApertureShape::Rectangular { half_x, half_y }) = sf.aperture {
        let theta = half_y.atan2(half_x);
        azimuths.extend(
            [
                theta,
                -theta,
                std::f64::consts::PI - theta,
                std::f64::consts::PI + theta,
            ]
            .map(|a| (a, 0.0)),
        );
    }
    let mut nodes = Vec::with_capacity(65 * azimuths.len());
    for (theta, angular_weight) in azimuths {
        let (sin, cos) = theta.sin_cos();
        let (lo, hi) = radial_bounds(sf, cos, sin)?;
        let span = (hi - lo) * (hi + lo);
        for j in 0..=64 {
            let r2 = lo * lo + span * j as f64 / 64.0;
            let r = r2.sqrt();
            let (x, y) = (r * cos, r * sin);
            let sag = surface_math::evaluate(&sf.shape, x, y)
                .ok_or_else(|| invalid("clear aperture extends outside the sag domain"))?;
            nodes.push(Node {
                r2,
                x,
                y,
                sag,
                weight: angular_weight * span * if j == 0 || j == 64 { 0.5 } else { 1.0 },
            });
        }
    }
    Ok(nodes)
}

fn sphere(c: f64, r2: f64) -> (f64, f64) {
    let w = (1.0 - c * c * r2).sqrt();
    (c * r2 / (1.0 + w), c / w)
}

/// Area-weighted least-squares sphere, including an axial piston. Subtract
/// means before squaring, so a large base sag does not obscure a small fit.
fn best_fit_sphere(nodes: &[Node]) -> (f64, f64) {
    let weight: f64 = nodes.iter().map(|n| n.weight).sum();
    let fit = |c: f64| {
        let mut sum = KahanSum::default();
        for n in nodes {
            sum.add(n.weight * (n.sag.sag - sphere(c, n.r2).0));
        }
        let piston = sum.value() / weight;
        let mut squares = KahanSum::default();
        for n in nodes {
            let d = n.sag.sag - sphere(c, n.r2).0 - piston;
            squares.add(n.weight * d * d);
        }
        (squares.value(), piston)
    };
    let max_r2 = nodes.iter().map(|n| n.r2).fold(0.0, f64::max);
    let limit = (1.0 - 1e-12) / max_r2.sqrt();
    let (mut lo, mut hi) = (-limit, limit);
    let ratio = (5.0_f64.sqrt() - 1.0) / 2.0;
    let (mut a, mut b) = (hi - ratio * (hi - lo), lo + ratio * (hi - lo));
    let (mut fa, mut fb) = (fit(a).0, fit(b).0);
    for _ in 0..72 {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - ratio * (hi - lo);
            fa = fit(a).0;
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + ratio * (hi - lo);
            fb = fit(b).0;
        }
    }
    let c = (lo + hi) / 2.0;
    (c, fit(c).1)
}

pub(super) fn value(s: &OpticalSystem, k: &OperandKind) -> OperandValue {
    use OperandKind::*;
    let surface = metric_surface(k).ok_or(OperandError::Unsupported)?;
    let sf = s
        .surface_index(surface)
        .map(|i| &s.surfaces[i])
        .ok_or_else(|| invalid("surface metric: unknown surface"))?;
    let nodes = nodes(sf)?;
    match k {
        DoeMinZonePeriod { .. } | DoeMaxPhaseSlope { .. } | DoeTotalPhase { .. } => {
            let SurfaceShape::Diffractive { phase, .. } = &sf.shape else {
                return Err(invalid("DOE operand needs a diffractive surface"));
            };
            let (mut slope, mut low, mut high) = (0.0_f64, f64::INFINITY, f64::NEG_INFINITY);
            for n in &nodes {
                let p = surface_math::phase(phase, n.x, n.y)
                    .ok_or_else(|| invalid("clear aperture extends outside the phase domain"))?;
                // |∇surface Φ|² = |∇xy Φ|² − (∇z·∇xy Φ)²/(1+|∇z|²).
                // Evaluate by an orthogonal tangent basis to avoid cancellation.
                let [zx, zy] = n.sag.gradient;
                let [px, py] = p.gradient;
                let h = 1.0_f64.hypot(zx);
                let g1 = px / h;
                let g2 = ((1.0 + zx * zx) * py - zx * zy * px) / (h * h.hypot(zy));
                slope = slope.max(g1.hypot(g2));
                low = low.min(p.sag);
                high = high.max(p.sag);
            }
            Ok(match k {
                DoeMinZonePeriod { .. } => TAU / slope,
                DoeMaxPhaseSlope { .. } => slope,
                _ => high - low,
            })
        }
        QMaxDeparture { .. } | QMaxDepartureSlope { .. } => {
            if !matches!(
                sf.shape,
                SurfaceShape::QCon { .. } | SurfaceShape::QBfs { .. }
            ) {
                return Err(invalid("Q departure needs a Q-type surface"));
            }
            let (c, piston) = best_fit_sphere(&nodes);
            Ok(nodes
                .iter()
                .map(|n| {
                    let (z, slope) = sphere(c, n.r2);
                    if matches!(k, QMaxDeparture { .. }) {
                        (n.sag.sag - z - piston).abs()
                    } else {
                        (n.sag.gradient[0] - slope * n.x).hypot(n.sag.gradient[1] - slope * n.y)
                    }
                })
                .fold(0.0, f64::max))
        }
        FreeformMaxDeparture { .. } => {
            let (SurfaceShape::XYPolynomial {
                curvature, conic, ..
            }
            | SurfaceShape::ZernikeSag {
                curvature, conic, ..
            }) = sf.shape
            else {
                return Err(invalid(
                    "freeform departure needs an XY or Zernike sag surface",
                ));
            };
            let base = SurfaceShape::Conic { curvature, conic };
            Ok(nodes
                .iter()
                .map(|n| (n.sag.sag - base.sag(n.r2).unwrap()).abs())
                .fold(0.0, f64::max))
        }
        _ => unreachable!(),
    }
}
