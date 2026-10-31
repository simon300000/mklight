//! Portable algebra, mirrored operation for operation by geometry.metal.
use crate::{
    Df32, FpEnv,
    fx::{D, F},
};
/// Hi+lo geometry, without the legacy rounded-point OPL correction.
pub fn coordinate<E: FpEnv>(p: [Df32; 4], c: [Df32; 4], delta: Df32) -> [Df32; 4] {
    let p = p.map(D::<E>::from_df);
    let c = c.map(D::<E>::from_df);
    let d = D::<E>::from_df(delta);
    let dt = [p[2] - c[2], p[3] - c[3]];
    [
        (p[0] - c[0]) + d * dt[0],
        (p[1] - c[1]) + d * dt[1],
        dt[0],
        dt[1],
    ]
    .map(|v| v.0)
}
pub fn radial<E: FpEnv>(p: [Df32; 2], centre: [Df32; 2]) -> [Df32; 2] {
    let x = D::<E>::from_df(p[0]) - D::from_df(centre[0]);
    let y = D::<E>::from_df(p[1]) - D::from_df(centre[1]);
    let ax = if x.hi() < 0.0 { -x } else { x };
    let ay = if y.hi() < 0.0 { -y } else { y };
    [
        (x * x + y * y).sqrt().0,
        if (ax - ay).hi() >= 0.0 { ax.0 } else { ay.0 },
    ]
}
/// Reduce turns in double-float before transcendental evaluation. CPU and
/// Metal libm need only meet the characteristic-function tolerance gate.
pub fn characteristic<E: FpEnv>(p: [Df32; 2], frequency: [Df32; 2], weight: Df32) -> [Df32; 2] {
    let a = D::<E>::from_df(p[0]) * D::from_df(frequency[0])
        + D::from_df(p[1]) * D::from_df(frequency[1]);
    let a = a - D::new(a.hi().v().round(), 0.0);
    let phase = (a * D::from_df(Df32::from_f64(std::f64::consts::TAU)))
        .hi()
        .v();
    let w = D::<E>::from_df(weight);
    [(w * F::new(phase.cos())).0, (w * F::new(phase.sin())).0]
}
