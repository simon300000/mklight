//! Deterministic sequential reference for the `Primary` class.
//!
//! [`primary_reference`] integrates the designed path over a source's own
//! sampling domain with the sequential reference tracer, so it computes
//! exactly the expectation that the Monte Carlo's `Primary` class
//! estimates:
//!
//! * the source's unit disk (beam cross-section, pupil-plane disk, or the
//!   equal-area map of a point source's cone, `1 − cos θ = s²(1 − cos α)`)
//!   is covered by the centres of an `n × n` square grid; each centre
//!   inside the disk carries the exact weight `P·(2/n)²/π`, so the beam's
//!   outer edge (which the stop always blocks) adds no quadrature error and
//!   only the clipping boundary inside the beam does (≈ `n^{-4/3}`
//!   relative);
//! * each sample is traced through [`StraySystem::clipped_system`] (the
//!   scene's clear apertures as hard clips) at every system wavelength and
//!   counted when it reaches the detector ([`StraySystem::detector_shape`]);
//! * its flux is `w · τ · Π T_i`, with the internal transmittance `τ` of the
//!   trace and, per interacting surface, the unpolarised (s/p-averaged)
//!   `T = ResolvedCoating::rt(λ, n₁, n₂, cos i).t_unpolarized()` (`R` for a
//!   mirror): the same coating code and polarisation average the Monte
//!   Carlo uses at every interface;
//! * wavelengths are mixed with the normalised system weights, as the Monte
//!   Carlo draws them.
//!
//! The result gives the transmitted flux and the flux-weighted centroid and
//! RMS radius on the image surface, plus the unweighted geometric spot of
//! the same rays.

use std::f64::consts::PI;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use mklight_core::error::{MkError, MkResult};
use mklight_core::job::JobContext;
use mklight_core::math::{KahanSum, Vec3};
use mklight_core::model::{Interaction, SurfaceShape};
use mklight_core::resolved::ResolvedSystem;
use mklight_trace::{Ray, SurfaceRecord, TraceOptions, TraceStatus, trace_ray};

use crate::geom::onb;
use crate::source::SourceSpec;
use crate::system::StraySystem;

/// Sequential reference of the `Primary` class (module docs). Positions are
/// in the image surface's local frame (mm).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PrimaryReference {
    /// Emitted power of the source (W).
    pub emitted: f64,
    /// Transmitted flux on the detector (W), wavelength-weighted.
    pub flux: f64,
    /// `flux` per wavelength, each as if the source emitted all its power
    /// at that wavelength.
    pub flux_by_wavelength: Vec<f64>,
    /// Flux-weighted centroid and RMS radius about it.
    pub centroid: [f64; 2],
    pub rms_radius: f64,
    /// Unweighted (geometric) spot of the rays that reach the detector, over
    /// all wavelengths with equal weight.
    pub geometric_centroid: [f64; 2],
    pub geometric_rms_radius: f64,
    /// Grid samples inside the source disk, and samples (× wavelengths)
    /// that reached the detector.
    pub samples: usize,
    pub passed: usize,
}

/// Unit-disk point `(u, v)` → starting ray (global frame) of `src`.
fn sample_ray(src: &SourceSpec, u: f64, v: f64) -> MkResult<(Vec3, Vec3)> {
    Ok(match *src {
        SourceSpec::Collimated {
            direction,
            center,
            radius,
            start_distance,
            ..
        } => {
            let d = direction.normalized();
            let (t1, t2) = onb(d);
            (
                center + t1 * (radius * u) + t2 * (radius * v) - d * start_distance,
                d,
            )
        }
        SourceSpec::CollimatedThrough {
            direction,
            center,
            normal,
            radius,
            start_distance,
            ..
        } => {
            let d = direction.normalized();
            let (t1, t2) = onb(normal.normalized());
            (
                center + t1 * (radius * u) + t2 * (radius * v) - d * start_distance,
                d,
            )
        }
        SourceSpec::Point {
            position,
            axis,
            half_angle_deg,
            ..
        } => {
            let a = axis.normalized();
            let (t1, t2) = onb(a);
            let h = 2.0 * (0.5 * half_angle_deg.to_radians()).sin().powi(2);
            let s2 = u * u + v * v;
            let one_minus_cos = s2 * h;
            let cos_t = 1.0 - one_minus_cos;
            let sin_t = (one_minus_cos * (2.0 - one_minus_cos)).max(0.0).sqrt();
            let phi = v.atan2(u);
            let (s, c) = phi.sin_cos();
            (
                position,
                (t1 * (sin_t * c) + t2 * (sin_t * s) + a * cos_t).normalized(),
            )
        }
        SourceSpec::LambertianDisk { .. } => {
            return Err(MkError::InvalidModel(
                "the sequential reference supports collimated and point sources".into(),
            ));
        }
    })
}

/// Flux factor of a traced designed path: internal transmittance times the
/// unpolarised coating transmittance (reflectance for mirrors) of every
/// interacting surface.
pub fn path_transmittance(
    rs: &ResolvedSystem,
    wl: usize,
    rec: &[SurfaceRecord],
    internal: f64,
) -> f64 {
    let image = rs.image_index();
    let lambda = rs.wavelengths_um[wl];
    let mut t = internal;
    for (i, r) in rec.iter().enumerate().take(image).skip(1) {
        let s = &rs.surfaces[i];
        if s.is_passive() || matches!(s.shape, SurfaceShape::CoordinateBreak) {
            continue;
        }
        let n1 = rs.n_before(i, wl);
        let mirror = s.interaction == Interaction::Reflect;
        let rt = s
            .coating
            .rt(lambda, n1, s.n_after[wl], r.cos_incidence, mirror);
        t *= if mirror {
            rt.r_unpolarized()
        } else {
            rt.t_unpolarized()
        };
    }
    t
}

#[derive(Clone, Copy, Default)]
struct Acc {
    f: KahanSum,
    sx: KahanSum,
    sy: KahanSum,
    sr: KahanSum,
}

impl Acc {
    fn add(&mut self, w: f64, x: f64, y: f64) {
        self.f.add(w);
        self.sx.add(w * x);
        self.sy.add(w * y);
        self.sr.add(w * (x * x + y * y));
    }
    fn merge(&mut self, o: &Acc) {
        self.f.add(o.f.value());
        self.sx.add(o.sx.value());
        self.sy.add(o.sy.value());
        self.sr.add(o.sr.value());
    }
    fn moments(&self) -> ([f64; 2], f64) {
        let f = self.f.value();
        if f <= 0.0 {
            return ([0.0; 2], 0.0);
        }
        let (cx, cy) = (self.sx.value() / f, self.sy.value() / f);
        let var = self.sr.value() / f - cx * cx - cy * cy;
        ([cx, cy], var.max(0.0).sqrt())
    }
}

/// One grid row: per-wavelength flux accumulators, the geometric
/// accumulator, and sample and pass counts.
struct Row {
    acc: Vec<Acc>,
    geo: Acc,
    samples: usize,
    passed: usize,
}

/// The deterministic `Primary` reference of `source` on `sys` with an
/// `n × n` grid (module docs). Rows run in parallel and are folded in
/// order, so the result does not depend on the thread count.
pub fn primary_reference(
    sys: &StraySystem,
    source: &SourceSpec,
    n: usize,
    ctx: &JobContext,
) -> MkResult<PrimaryReference> {
    source.validate()?;
    if n == 0 {
        return Err(MkError::InvalidModel(
            "reference grid must be non-empty".into(),
        ));
    }
    let rs = sys.clipped_system();
    let nw = rs.wavelengths_um.len();
    let total_w: f64 = rs.wavelength_weights.iter().sum();
    let wfrac: Vec<f64> = rs.wavelength_weights.iter().map(|w| w / total_w).collect();
    let start = mklight_trace::trace::start_frame(&rs);
    let det = sys.detector_shape;
    let opts = TraceOptions {
        intersect: sys.tracer().settings().intersect,
        ..TraceOptions::default()
    };
    let power = source.power();
    let cell = 4.0 / (n * n) as f64 / PI * power;
    let h = 2.0 / n as f64;
    let rows: Vec<MkResult<Row>> = (0..n)
        .into_par_iter()
        .map(|iy| {
            ctx.check()?;
            let v = -1.0 + h * (iy as f64 + 0.5);
            let mut acc = vec![Acc::default(); nw];
            let mut geo = Acc::default();
            let (mut samples, mut passed) = (0usize, 0usize);
            let mut rec = Vec::with_capacity(rs.surfaces.len());
            for ix in 0..n {
                let u = -1.0 + h * (ix as f64 + 0.5);
                if u * u + v * v > 1.0 {
                    continue;
                }
                samples += 1;
                let (o, d) = sample_ray(source, u, v)?;
                let ray = Ray::new(start.to_local_point(o), start.to_local_dir(d));
                for (wl, a) in acc.iter_mut().enumerate() {
                    let r = trace_ray(&rs, &ray, wl, &opts, Some(&mut rec));
                    if r.status != TraceStatus::Ok {
                        continue;
                    }
                    let (x, y) = (r.ray.pos.x, r.ray.pos.y);
                    if !det.contains(x, y) {
                        continue;
                    }
                    passed += 1;
                    let t = path_transmittance(&rs, wl, &rec, r.transmittance);
                    a.add(cell * t, x, y);
                    geo.add(1.0, x, y);
                }
            }
            Ok(Row {
                acc,
                geo,
                samples,
                passed,
            })
        })
        .collect();
    let mut acc = vec![Acc::default(); nw];
    let mut geo = Acc::default();
    let (mut samples, mut passed) = (0, 0);
    for row in rows {
        let Row {
            acc: a,
            geo: g,
            samples: s,
            passed: p,
        } = row?;
        for (x, y) in acc.iter_mut().zip(&a) {
            x.merge(y);
        }
        geo.merge(&g);
        samples += s;
        passed += p;
    }
    let mut mix = Acc::default();
    for (a, w) in acc.iter().zip(&wfrac) {
        mix.f.add(w * a.f.value());
        mix.sx.add(w * a.sx.value());
        mix.sy.add(w * a.sy.value());
        mix.sr.add(w * a.sr.value());
    }
    let (centroid, rms_radius) = mix.moments();
    let (geometric_centroid, geometric_rms_radius) = geo.moments();
    ctx.report(1.0, "");
    Ok(PrimaryReference {
        emitted: power,
        flux: mix.f.value(),
        flux_by_wavelength: acc.iter().map(|a| a.f.value()).collect(),
        centroid,
        rms_radius,
        geometric_centroid,
        geometric_rms_radius,
        samples,
        passed,
    })
}
