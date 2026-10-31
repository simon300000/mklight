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
//!   cell's coverage. Linear fields are reproduced exactly and no smoothing
//!   is imposed.
//!
//!   **The clear aperture.** The union of the triangles ends at the
//!   outermost surviving rays, so on its own it shrinks the true aperture
//!   by up to one ray spacing `h` (for a circular pupil on a 32 × 32 ray
//!   lattice the area is about 8 % low, which lowers the MTF near cutoff
//!   markedly). The tracer should therefore pass the true boundary as a
//!   [`PupilMask`]: the analytic stop shape, or a polygon of rays traced
//!   along the vignetting rim. The mask then sets the coverage, triangles
//!   are clipped to it, and mask sub-points outside every triangle take the
//!   linear extrapolation of the nearest triangle (by sub-grid distance), an
//!   `O(h²)` error for a smooth wavefront. Extrapolation further than two
//!   ray spacings is reported as [`WaveWarning::SparseRays`].
//! * [`grid_from_scattered_zernike`] — for unstructured points: a weighted
//!   least-squares Zernike fit of the OPD, evaluated on the grid inside a
//!   given [`PupilMask`] (analytic or polygonal, e.g. a cat's-eye
//!   vignetted pupil). Smooth and robust to noise, but limited to the fitted
//!   order (it cannot represent sharp local features). Mask regions far
//!   from every ray are reported as [`WaveWarning::SparseRays`].

use std::collections::VecDeque;

use mklight_core::{JobContext, MkError, MkResult};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::pupil::{PupilFunction, PupilGeometry, PupilMask};
use crate::warnings::WaveWarning;
use crate::zernike::{ZernikeBasis, ZernikeFit, ZernikeFitSettings, fit_zernike};

/// One traced ray at the exit pupil.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct RaySample {
    /// Normalised exit-pupil coordinates in the chief-ray frame (the ray's
    /// position on the reference sphere, see [`crate::pupil`],
    /// [`PupilGeometry::sphere_coords`] and [`PupilGeometry::pupil_coords`]).
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

/// Linear interpolant over a triangle of rays (barycentric; outside the
/// triangle it extrapolates the same plane).
#[derive(Clone, Copy)]
struct Tri {
    v: [RaySample; 3],
    e1: (f64, f64),
    e2: (f64, f64),
    det: f64,
}

impl Tri {
    fn new(v: [RaySample; 3]) -> Option<Self> {
        let e1 = (v[1].u - v[0].u, v[1].v - v[0].v);
        let e2 = (v[2].u - v[0].u, v[2].v - v[0].v);
        let det = e1.0 * e2.1 - e1.1 * e2.0;
        (det.abs() > 1e-300).then_some(Tri { v, e1, e2, det })
    }
    fn bary(&self, u: f64, v: f64) -> [f64; 3] {
        let (px, py) = (u - self.v[0].u, v - self.v[0].v);
        let l1 = (px * self.e2.1 - py * self.e2.0) / self.det;
        let l2 = (self.e1.0 * py - self.e1.1 * px) / self.det;
        [1.0 - l1 - l2, l1, l2]
    }
    /// `(OPD, amplitude ≥ 0)` of the plane through the three rays.
    fn eval(&self, l: [f64; 3]) -> (f64, f64) {
        let w =
            l[0] * self.v[0].opd_waves + l[1] * self.v[1].opd_waves + l[2] * self.v[2].opd_waves;
        let a =
            l[0] * self.v[0].amplitude + l[1] * self.v[1].amplitude + l[2] * self.v[2].amplitude;
        (w, a.max(0.0))
    }
}

/// Triangulated barycentric resampling of structured ray samples onto an
/// `n × n` pupil grid (see the module docs). With `mask` the clear aperture
/// is taken from the mask and edge cells are filled by extrapolation;
/// without it, the aperture is the union of the ray triangles. Returns the
/// pupil and any [`WaveWarning::SparseRays`] diagnostic.
pub fn grid_from_structured(
    rays: &StructuredSamples,
    n: usize,
    wavelength_um: f64,
    geometry: PupilGeometry,
    mask: Option<&PupilMask>,
    supersample: usize,
    ctx: &JobContext,
) -> MkResult<(PupilFunction, Vec<WaveWarning>)> {
    let (nx, ny) = (rays.nx, rays.ny);
    if rays.samples.len() != nx * ny || nx < 2 || ny < 2 || n < 2 {
        return Err(MkError::InvalidParam(
            "structured samples: bad lattice size".into(),
        ));
    }
    let finite = |r: &RaySample| {
        r.u.is_finite() && r.v.is_finite() && r.opd_waves.is_finite() && r.amplitude.is_finite()
    };
    if rays.samples.iter().flatten().any(|r| !finite(r)) {
        return Err(MkError::InvalidParam("non-finite ray sample".into()));
    }
    if let Some(m) = mask {
        m.validate()?;
    }
    // Triangulation of the logical lattice.
    let get = |i: usize, j: usize| rays.samples[j * nx + i];
    let mut tris: Vec<Tri> = Vec::new();
    for j in 0..ny - 1 {
        for i in 0..nx - 1 {
            let c = [get(i, j), get(i + 1, j), get(i + 1, j + 1), get(i, j + 1)];
            let valid: Vec<RaySample> = c.iter().flatten().copied().collect();
            match valid.len() {
                4 => {
                    let [p0, p1, p2, p3] = [valid[0], valid[1], valid[2], valid[3]];
                    let d02 = (p0.u - p2.u).hypot(p0.v - p2.v);
                    let d13 = (p1.u - p3.u).hypot(p1.v - p3.v);
                    let pair = if d02 <= d13 {
                        [[p0, p1, p2], [p0, p2, p3]]
                    } else {
                        [[p0, p1, p3], [p1, p2, p3]]
                    };
                    tris.extend(pair.into_iter().filter_map(Tri::new));
                }
                3 => tris.extend(Tri::new([valid[0], valid[1], valid[2]])),
                _ => {}
            }
        }
    }
    ctx.check()?;
    // Rasterise: owner[sub-point] = first triangle containing it.
    let s = supersample.max(1);
    let ns = n * s; // sub-grid side
    let d = 2.0 / ns as f64; // sub-grid spacing
    let sub_coord = |a: usize| -1.0 + (a as f64 + 0.5) * d;
    const NONE: u32 = u32::MAX;
    let mut owner = vec![NONE; ns * ns];
    for (t_id, t) in tris.iter().enumerate() {
        if t_id % 4096 == 0 {
            ctx.check()?;
            ctx.report(0.5 * t_id as f64 / tris.len() as f64, "pupil resampling");
        }
        let us = t.v.map(|r| r.u);
        let vs = t.v.map(|r| r.v);
        let fi = |x: f64| (x + 1.0) / d - 0.5;
        let a0 = fi(us.iter().copied().fold(f64::INFINITY, f64::min))
            .ceil()
            .max(0.0);
        let a1 = fi(us.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .floor()
            .min(ns as f64 - 1.0);
        let b0 = fi(vs.iter().copied().fold(f64::INFINITY, f64::min))
            .ceil()
            .max(0.0);
        let b1 = fi(vs.iter().copied().fold(f64::NEG_INFINITY, f64::max))
            .floor()
            .min(ns as f64 - 1.0);
        if a1 < a0 || b1 < b0 {
            continue;
        }
        for j in b0 as usize / s..=b1 as usize / s {
            for i in a0 as usize / s..=a1 as usize / s {
                let (aa, bb) = (i * s, j * s);
                let full = aa >= a0 as usize
                    && aa + s - 1 <= a1 as usize
                    && bb >= b0 as usize
                    && bb + s - 1 <= b1 as usize
                    && [
                        (aa, bb),
                        (aa + s - 1, bb),
                        (aa, bb + s - 1),
                        (aa + s - 1, bb + s - 1),
                    ]
                    .iter()
                    .all(|&(x, y)| {
                        t.bary(sub_coord(x), sub_coord(y))
                            .iter()
                            .all(|&l| l >= -1e-12)
                    });
                for b in bb.max(b0 as usize)..(bb + s).min(b1 as usize + 1) {
                    for a in aa.max(a0 as usize)..(aa + s).min(a1 as usize + 1) {
                        let k = b * ns + a;
                        if owner[k] == NONE
                            && (full
                                || t.bary(sub_coord(a), sub_coord(b))
                                    .iter()
                                    .all(|&l| l >= -1e-12))
                        {
                            owner[k] = t_id as u32;
                        }
                    }
                }
            }
        }
    }
    let mut warnings = Vec::new();
    // Which sub-points belong to the aperture, and the triangle that
    // supplies their value (its interior or its extrapolation).
    let source: Vec<u32> = match mask {
        None => owner,
        Some(mask) => {
            let inside = mask.contains_lattice(&(0..ns).map(sub_coord).collect::<Vec<_>>());
            ctx.check()?;
            // Multi-source BFS (8-neighbour) from every rasterised sub-point:
            // each reached sub-point inherits the triangle of its source.
            let missing: Vec<bool> = owner.iter().map(|&id| id == NONE).collect();
            let mut from = owner;
            let mut src_pt = vec![NONE; ns * ns];
            let mut seeds = Vec::new();
            let mut queue = VecDeque::new();
            for k in 0..ns * ns {
                if !missing[k] {
                    // Interior seeds cannot reach an unowned point before a
                    // boundary seed. Preserve row order for identical ties.
                    let (a, b) = (k % ns, k / ns);
                    if (b.saturating_sub(1)..=(b + 1).min(ns - 1)).any(|y| {
                        (a.saturating_sub(1)..=(a + 1).min(ns - 1)).any(|x| missing[y * ns + x])
                    }) {
                        src_pt[k] = seeds.len() as u32;
                        seeds.push(k);
                        queue.push_back(k);
                    }
                }
            }
            let mut remaining = inside
                .iter()
                .zip(&missing)
                .filter(|(i, m)| **i && **m)
                .count();
            while remaining > 0
                && let Some(k) = queue.pop_front()
            {
                if k % ns == 0 {
                    ctx.check()?;
                }
                let (a, b) = ((k % ns) as i64, (k / ns) as i64);
                for (da, db) in [
                    (1, 0),
                    (-1, 0),
                    (0, 1),
                    (0, -1),
                    (1, 1),
                    (1, -1),
                    (-1, 1),
                    (-1, -1),
                ] {
                    let (aa, bb) = (a + da, b + db);
                    if aa < 0 || bb < 0 || aa >= ns as i64 || bb >= ns as i64 {
                        continue;
                    }
                    let q = bb as usize * ns + aa as usize;
                    if from[q] == NONE {
                        from[q] = from[k];
                        src_pt[q] = src_pt[k];
                        remaining -= usize::from(inside[q]);
                        queue.push_back(q);
                    }
                }
            }
            // Compare each gap with its source triangle's longer lattice
            // edge, not sqrt(area): anamorphic pupils have unequal spacings.
            let mut worst = (0.0, 1.0);
            for k in 0..ns * ns {
                if inside[k] && missing[k] && src_pt[k] != NONE {
                    let q = seeds[src_pt[k] as usize];
                    let gap = (sub_coord(k % ns) - sub_coord(q % ns))
                        .hypot(sub_coord(k / ns) - sub_coord(q / ns));
                    let t = &tris[from[k] as usize];
                    let mut edges = [0, 1, 2].map(|i| {
                        let (a, b) = (t.v[i], t.v[(i + 1) % 3]);
                        (a.u - b.u).hypot(a.v - b.v)
                    });
                    edges.sort_by(f64::total_cmp);
                    let h = edges[1]; // excludes the diagonal
                    if gap / h > worst.0 / worst.1 {
                        worst = (gap, h);
                    }
                }
            }
            if worst.0 > 2.0 * worst.1 {
                warnings.push(WaveWarning::SparseRays {
                    max_gap: worst.0,
                    typical_spacing: worst.1,
                });
            }
            from.iter()
                .zip(&inside)
                .map(|(&f, &i)| if i { f } else { NONE })
                .collect()
        }
    };
    ctx.report(0.8, "pupil resampling");
    let ss = (s * s) as f64;
    let mut cov = vec![0.0; n * n];
    let mut amp = vec![0.0; n * n];
    let mut opd = vec![0.0; n * n];
    cov.par_chunks_mut(n)
        .zip(amp.par_chunks_mut(n))
        .zip(opd.par_chunks_mut(n))
        .enumerate()
        .try_for_each(|(j, ((cov, amp), opd))| -> MkResult<()> {
            ctx.check()?;
            for i in 0..n {
                let (a, b) = (i * s, j * s);
                let id = source[b * ns + a];
                let uniform =
                    id != NONE && (b..b + s).all(|y| (a..a + s).all(|x| source[y * ns + x] == id));
                if uniform {
                    let t = &tris[id as usize];
                    // Both fields are affine inside a triangle. The mean
                    // of a symmetric subgrid is its centre value. Retain
                    // subpoint integration if extrapolated amplitude clips.
                    let positive = [
                        (a, b),
                        (a + s - 1, b),
                        (a, b + s - 1),
                        (a + s - 1, b + s - 1),
                    ]
                    .iter()
                    .all(|&(x, y)| {
                        let l = t.bary(sub_coord(x), sub_coord(y));
                        l.iter()
                            .zip(&t.v)
                            .map(|(l, r)| l * r.amplitude)
                            .sum::<f64>()
                            >= 0.0
                    });
                    if positive {
                        let u = -1.0 + (i as f64 + 0.5) * 2.0 / n as f64;
                        let v = -1.0 + (j as f64 + 0.5) * 2.0 / n as f64;
                        let (w, amplitude) = t.eval(t.bary(u, v));
                        cov[i] = 1.0;
                        amp[i] = amplitude;
                        opd[i] = w;
                        continue;
                    }
                }
                let (mut count, mut w, mut amplitude) = (0, 0.0, 0.0);
                for y in b..b + s {
                    for x in a..a + s {
                        let id = source[y * ns + x];
                        if id != NONE {
                            let t = &tris[id as usize];
                            let (ww, aa) = t.eval(t.bary(sub_coord(x), sub_coord(y)));
                            count += 1;
                            w += ww;
                            amplitude += aa;
                        }
                    }
                }
                if count > 0 {
                    cov[i] = count as f64 / ss;
                    amp[i] = amplitude / count as f64;
                    opd[i] = w / count as f64;
                }
            }
            Ok(())
        })?;
    let pupil = PupilFunction::from_samples(n, wavelength_um, geometry, amp, cov, opd)?;
    ctx.report(1.0, "pupil resampling");
    Ok((pupil, warnings))
}

/// Zernike-fit resampling of unstructured ray samples: the OPD is fitted
/// with `basis` (weights = amplitude²) and the amplitude with the Standard
/// terms up to n = 4 (clamped at 0), then both are evaluated on the grid
/// inside `mask`. Returns the pupil and the OPD fit (residual,
/// conditioning, and a [`WaveWarning::SparseRays`] warning when parts of
/// the mask lie more than three typical ray spacings from every ray, where
/// the fit extrapolates).
#[allow(clippy::too_many_arguments)]
pub fn grid_from_scattered_zernike(
    rays: &[RaySample],
    basis: &ZernikeBasis,
    n: usize,
    wavelength_um: f64,
    geometry: PupilGeometry,
    mask: &PupilMask,
    supersample: usize,
    ctx: &JobContext,
) -> MkResult<(PupilFunction, ZernikeFit)> {
    mask.validate()?;
    let pts: Vec<(f64, f64)> = rays.iter().map(|r| (r.u, r.v)).collect();
    let opd: Vec<f64> = rays.iter().map(|r| r.opd_waves).collect();
    let amp: Vec<f64> = rays.iter().map(|r| r.amplitude).collect();
    let w: Vec<f64> = amp.iter().map(|a| a * a).collect();
    let settings = ZernikeFitSettings::default();
    let mut fit = fit_zernike(
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
    let (oc, ac) = (
        fit.set.coefficients().to_vec(),
        afit.set.coefficients().to_vec(),
    );
    let pupil = PupilFunction::from_fn(
        n,
        wavelength_um,
        geometry,
        supersample,
        |u, v| mask.contains(u, v),
        |u, v| (abasis.eval(u, v, &ac).max(0.0), basis.eval(u, v, &oc)),
    )?;
    ctx.check()?;
    // Coverage check: distance from each aperture cell to the nearest ray.
    let used: Vec<(f64, f64)> = rays
        .iter()
        .filter(|r| r.amplitude > 0.0 && r.u.is_finite() && r.v.is_finite())
        .map(|r| (r.u, r.v))
        .collect();
    let area: f64 = pupil.coverage().iter().sum::<f64>() * pupil.spacing().powi(2);
    let n_in = used.iter().filter(|&&(u, v)| mask.contains(u, v)).count();
    if n_in > 0 {
        let h = (area / n_in as f64).sqrt();
        let near = NearestPoint::new(&used);
        let max_gap = (0..n * n)
            .into_par_iter()
            .filter(|&k| pupil.coverage()[k] > 0.0)
            .map(|k| near.distance(pupil.coord(k % n), pupil.coord(k / n)))
            .reduce(|| 0.0, f64::max);
        if max_gap > 3.0 * h {
            fit.warnings.push(WaveWarning::SparseRays {
                max_gap,
                typical_spacing: h,
            });
        }
    } else {
        fit.warnings.push(WaveWarning::SparseRays {
            max_gap: f64::INFINITY,
            typical_spacing: f64::INFINITY,
        });
    }
    Ok((pupil, fit))
}

/// Bucketed nearest-point search over a point set in the pupil plane.
struct NearestPoint<'a> {
    pts: &'a [(f64, f64)],
    lo: (f64, f64),
    cell: f64,
    g: usize,
    buckets: Vec<Vec<usize>>,
}

impl<'a> NearestPoint<'a> {
    fn new(pts: &'a [(f64, f64)]) -> Self {
        let lo = pts.iter().fold((f64::INFINITY, f64::INFINITY), |a, p| {
            (a.0.min(p.0), a.1.min(p.1))
        });
        let hi = pts
            .iter()
            .fold((f64::NEG_INFINITY, f64::NEG_INFINITY), |a, p| {
                (a.0.max(p.0), a.1.max(p.1))
            });
        let g = ((pts.len() as f64).sqrt().ceil() as usize).clamp(1, 1024);
        let cell = ((hi.0 - lo.0).max(hi.1 - lo.1) / g as f64).max(1e-12);
        let mut buckets = vec![Vec::new(); g * g];
        for (k, p) in pts.iter().enumerate() {
            let (i, j) = Self::bucket_of(lo, cell, g, *p);
            buckets[j * g + i].push(k);
        }
        NearestPoint {
            pts,
            lo,
            cell,
            g,
            buckets,
        }
    }

    fn bucket_of(lo: (f64, f64), cell: f64, g: usize, p: (f64, f64)) -> (usize, usize) {
        let f = |x: f64, l: f64| (((x - l) / cell).floor().max(0.0) as usize).min(g - 1);
        (f(p.0, lo.0), f(p.1, lo.1))
    }

    /// Euclidean distance from `(u, v)` to the nearest point: rings of
    /// buckets are searched outward until the ring's inner distance
    /// exceeds the best distance found.
    fn distance(&self, u: f64, v: f64) -> f64 {
        let g = self.g as i64;
        let fi = |x: f64, l: f64| ((x - l) / self.cell).floor() as i64;
        let (ci, cj) = (fi(u, self.lo.0), fi(v, self.lo.1));
        let mut best = f64::INFINITY;
        let visit = |i: i64, j: i64, best: &mut f64| {
            if i >= 0 && j >= 0 && i < g && j < g {
                for &k in &self.buckets[(j * g + i) as usize] {
                    let p = self.pts[k];
                    *best = best.min((p.0 - u).hypot(p.1 - v));
                }
            }
        };
        // Rings beyond this one lie entirely outside the bucket grid.
        let last = (ci
            .abs()
            .max(cj.abs())
            .max((ci - g).abs())
            .max((cj - g).abs()))
            + 1;
        for ring in 0..=last {
            // Points in ring `ring` are at least (ring − 1)·cell away.
            if (ring - 1) as f64 * self.cell > best {
                break;
            }
            if ring == 0 {
                visit(ci, cj, &mut best);
                continue;
            }
            // Only the ring's perimeter, clipped to the grid.
            let (i0, i1) = ((ci - ring).max(0), (ci + ring).min(g - 1));
            let (j0, j1) = ((cj - ring + 1).max(0), (cj + ring - 1).min(g - 1));
            for i in i0..=i1 {
                visit(i, cj - ring, &mut best);
                visit(i, cj + ring, &mut best);
            }
            for j in j0..=j1 {
                visit(ci - ring, j, &mut best);
                visit(ci + ring, j, &mut best);
            }
        }
        best
    }
}
