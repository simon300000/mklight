//! Building a [`PupilFunction`] grid from traced ray samples.
//!
//! The tracer knows the OPD only at the exit-pupil points where its rays
//! land, which (with pupil aberrations and vignetting) are not the grid
//! points. Two methods are provided:
//!
//! * [`grid_from_structured`] — for rays traced on a logical `nx × ny`
//!   lattice (the usual case: a square grid in the entrance pupil). Each
//!   logical cell whose corners all survived is split along its shorter
//!   diagonal into two triangles (three surviving corners give one
//!   triangle). The triangles, placed at the rays' *exit-pupil* coordinates,
//!   are rasterised onto `supersample²` sub-points per grid cell; OPD and
//!   amplitude are interpolated linearly (barycentric) at each covered
//!   sub-point and averaged per cell, and the covered fraction becomes the
//!   cell's coverage. Linear fields are reproduced exactly, the clear
//!   aperture is the union of the triangles (vignetting and obscurations
//!   included), and no smoothing is imposed.
//! * [`grid_from_scattered_zernike`] — for unstructured points: a weighted
//!   least-squares Zernike fit of the OPD, evaluated on the grid inside a
//!   given aperture. Smooth and robust to noise, but limited to the fitted
//!   order (it cannot represent sharp local features).

use mklight_core::{JobContext, MkError, MkResult};
use serde::{Deserialize, Serialize};

use crate::pupil::{Aperture, PupilFunction, PupilGeometry};
use crate::zernike::{ZernikeBasis, ZernikeFit, ZernikeFitSettings, fit_zernike};

/// One traced ray at the exit pupil.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RaySample {
    /// Normalised exit-pupil coordinates (direction-cosine scaled, see
    /// [`crate::pupil`]).
    pub u: f64,
    pub v: f64,
    pub opd_waves: f64,
    /// Relative amplitude (√ of transmitted intensity × ray-density factor).
    pub amplitude: f64,
}

/// Rays traced on a logical lattice; `None` marks a vignetted or failed ray.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StructuredSamples {
    pub nx: usize,
    pub ny: usize,
    /// Row-major `samples[j·nx + i]`.
    pub samples: Vec<Option<RaySample>>,
}

/// Triangulated barycentric resampling of structured ray samples onto an
/// `n × n` pupil grid (see the module docs).
pub fn grid_from_structured(
    rays: &StructuredSamples,
    n: usize,
    wavelength_um: f64,
    geometry: PupilGeometry,
    supersample: usize,
    ctx: &JobContext,
) -> MkResult<PupilFunction> {
    let (nx, ny) = (rays.nx, rays.ny);
    if rays.samples.len() != nx * ny || nx < 2 || ny < 2 {
        return Err(MkError::InvalidParam(
            "structured samples: bad lattice size".into(),
        ));
    }
    let s = supersample.max(1);
    let ns = n * s; // sub-grid side
    let d = 2.0 / ns as f64; // sub-grid spacing
    let mut taken = vec![false; ns * ns];
    let mut cnt = vec![0u32; n * n];
    let mut sum_w = vec![0.0; n * n];
    let mut sum_a = vec![0.0; n * n];
    let get = |i: usize, j: usize| rays.samples[j * nx + i];
    let mut raster = |t: [RaySample; 3]| {
        let (x0, y0) = (t[0].u, t[0].v);
        let (e1x, e1y) = (t[1].u - x0, t[1].v - y0);
        let (e2x, e2y) = (t[2].u - x0, t[2].v - y0);
        let det = e1x * e2y - e1y * e2x;
        if det.abs() < 1e-300 {
            return;
        }
        let umin = t.iter().map(|r| r.u).fold(f64::INFINITY, f64::min);
        let umax = t.iter().map(|r| r.u).fold(f64::NEG_INFINITY, f64::max);
        let vmin = t.iter().map(|r| r.v).fold(f64::INFINITY, f64::min);
        let vmax = t.iter().map(|r| r.v).fold(f64::NEG_INFINITY, f64::max);
        // Sub-point a sits at −1 + (a + ½)·d.
        let fi = |x: f64| (x + 1.0) / d - 0.5;
        let a0 = fi(umin).ceil().max(0.0) as usize;
        let a1 = fi(umax).floor().min(ns as f64 - 1.0);
        let b0 = fi(vmin).ceil().max(0.0) as usize;
        let b1 = fi(vmax).floor().min(ns as f64 - 1.0);
        if a1 < a0 as f64 || b1 < b0 as f64 {
            return;
        }
        let tol = 1e-12;
        for b in b0..=b1 as usize {
            let v = -1.0 + (b as f64 + 0.5) * d;
            for a in a0..=a1 as usize {
                if taken[b * ns + a] {
                    continue;
                }
                let u = -1.0 + (a as f64 + 0.5) * d;
                let (px, py) = (u - x0, v - y0);
                let l1 = (px * e2y - py * e2x) / det;
                let l2 = (e1x * py - e1y * px) / det;
                let l0 = 1.0 - l1 - l2;
                if l0 < -tol || l1 < -tol || l2 < -tol {
                    continue;
                }
                taken[b * ns + a] = true;
                let c = (b / s) * n + a / s;
                cnt[c] += 1;
                sum_w[c] += l0 * t[0].opd_waves + l1 * t[1].opd_waves + l2 * t[2].opd_waves;
                sum_a[c] += l0 * t[0].amplitude + l1 * t[1].amplitude + l2 * t[2].amplitude;
            }
        }
    };
    for j in 0..ny - 1 {
        if j % 16 == 0 {
            ctx.check()?;
            ctx.report(j as f64 / (ny - 1) as f64, "pupil resampling");
        }
        for i in 0..nx - 1 {
            let c = [get(i, j), get(i + 1, j), get(i + 1, j + 1), get(i, j + 1)];
            let valid: Vec<RaySample> = c.iter().flatten().copied().collect();
            match valid.len() {
                4 => {
                    let [p0, p1, p2, p3] =
                        [c[0].unwrap(), c[1].unwrap(), c[2].unwrap(), c[3].unwrap()];
                    let d02 = (p0.u - p2.u).hypot(p0.v - p2.v);
                    let d13 = (p1.u - p3.u).hypot(p1.v - p3.v);
                    if d02 <= d13 {
                        raster([p0, p1, p2]);
                        raster([p0, p2, p3]);
                    } else {
                        raster([p0, p1, p3]);
                        raster([p1, p2, p3]);
                    }
                }
                3 => raster([valid[0], valid[1], valid[2]]),
                _ => {}
            }
        }
    }
    let ss = (s * s) as f64;
    let mut cov = vec![0.0; n * n];
    let mut amp = vec![0.0; n * n];
    let mut opd = vec![0.0; n * n];
    for k in 0..n * n {
        if cnt[k] > 0 {
            let c = cnt[k] as f64;
            cov[k] = c / ss;
            amp[k] = sum_a[k] / c;
            opd[k] = sum_w[k] / c;
        }
    }
    PupilFunction::from_samples(n, wavelength_um, geometry, amp, cov, opd)
}

/// Zernike-fit resampling of unstructured ray samples: the OPD is fitted
/// with `basis` (weights = amplitude²) and the amplitude with the Standard
/// terms up to n = 4 (clamped at 0), then both are evaluated on the grid
/// inside `aperture`. Returns the pupil and the OPD fit (residual and
/// conditioning).
#[allow(clippy::too_many_arguments)]
pub fn grid_from_scattered_zernike(
    rays: &[RaySample],
    basis: &ZernikeBasis,
    n: usize,
    wavelength_um: f64,
    geometry: PupilGeometry,
    aperture: Aperture,
    supersample: usize,
    ctx: &JobContext,
) -> MkResult<(PupilFunction, ZernikeFit)> {
    let pts: Vec<(f64, f64)> = rays.iter().map(|r| (r.u, r.v)).collect();
    let opd: Vec<f64> = rays.iter().map(|r| r.opd_waves).collect();
    let amp: Vec<f64> = rays.iter().map(|r| r.amplitude).collect();
    let w: Vec<f64> = amp.iter().map(|a| a * a).collect();
    let settings = ZernikeFitSettings::default();
    let fit = fit_zernike(
        &pts,
        &opd,
        Some(&w),
        basis,
        &settings,
        &ctx.subrange(0.0, 0.5),
    )?;
    let abasis = ZernikeBasis::standard_to_order(4);
    let afit = fit_zernike(
        &pts,
        &amp,
        None,
        &abasis,
        &settings,
        &ctx.subrange(0.5, 0.8),
    )?;
    let (oc, ac) = (fit.set.coefficients.clone(), afit.set.coefficients.clone());
    let pupil = PupilFunction::from_fn(
        n,
        wavelength_um,
        geometry,
        supersample,
        |u, v| aperture.contains(u, v),
        |u, v| (abasis.eval(u, v, &ac).max(0.0), basis.eval(u, v, &oc)),
    )?;
    Ok((pupil, fit))
}
