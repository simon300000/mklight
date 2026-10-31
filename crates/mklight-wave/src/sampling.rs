//! System-derived defaults, bounded independently of user-requested sampling.
use crate::{PupilFunction, StructuredSamples};

pub(crate) const MAX_AUTO_PUPIL: usize = 1024;
pub(crate) const MAX_AUTO_IMAGE: usize = 2048;

/// Largest Cartesian phase gradient of the lattice's linear triangles.
/// Unlike entrance-pupil differences this retains anamorphic pupil scaling.
pub(crate) fn pupil_grid(rays: &StructuredSamples) -> usize {
    let mut gradient: f64 = 0.0;
    for j in 0..rays.ny - 1 {
        for i in 0..rays.nx - 1 {
            let at = |x, y| rays.samples[y * rays.nx + x];
            let cell = [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)];
            for ids in [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]] {
                let [Some(a), Some(b), Some(c)] = ids.map(|i| cell[i]) else {
                    continue;
                };
                let (ux, uy) = (b.u - a.u, b.v - a.v);
                let (vx, vy) = (c.u - a.u, c.v - a.v);
                let det = ux * vy - uy * vx;
                if det.abs() > 1e-20 {
                    let (dw, ew) = (b.opd_waves - a.opd_waves, c.opd_waves - a.opd_waves);
                    gradient = gradient
                        .max(((dw * vy - ew * uy) / det).abs())
                        .max(((ew * ux - dw * vx) / det).abs());
                }
            }
        }
    }
    // At most 0.35 waves per sample: margin below the 0.5-wave diagnostic
    // for edge-cell averaging and refocusing. PSF windows enlarge this
    // further through `FieldPupils::ensure_window`.
    ((2.0 * gradient / 0.35)
        .clamp(64.0, MAX_AUTO_PUPIL as f64)
        .ceil() as usize)
        .next_power_of_two()
}

/// Image-space geometrical displacement from phase slope (ε = −λ/NA ∇W),
/// maximum about the reference and RMS about the mean. Projection carries
/// the chief-normal plane onto the image surface.
pub(crate) fn blur(p: &PupilFunction) -> (f64, f64) {
    let n = p.n();
    let b = p.geometry().surface_to_normal();
    let det = b[0][0] * b[1][1] - b[0][1] * b[1][0];
    let mut points = Vec::new();
    let scale = -2.0 * p.lambda_f_mm() / p.spacing();
    let off = p.image_offset_mm();
    let mut extent: f64 = 0.0;
    for j in 0..n - 1 {
        for i in 0..n - 1 {
            let k = j * n + i;
            if [k, k + 1, k + n].iter().all(|&q| p.coverage()[q] > 0.0) {
                let x = scale * (p.opd_waves()[k + 1] - p.opd_waves()[k]);
                let y = scale * (p.opd_waves()[k + n] - p.opd_waves()[k]);
                let q = [
                    (b[1][1] * x - b[0][1] * y) / det + off[0],
                    (b[0][0] * y - b[1][0] * x) / det + off[1],
                ];
                extent = extent.max(q[0].abs()).max(q[1].abs());
                points.push((q, p.coverage()[k] * p.amplitude()[k].powi(2)));
            }
        }
    }
    let rms = mklight_analysis::util::Moments::of(points.iter().copied()).rms_centroid;
    (extent, rms)
}

pub(crate) fn image_side(pupils: &[(&PupilFunction, f64)]) -> f64 {
    pupils
        .iter()
        .map(|(p, _)| {
            // Geometric flare plus a 16 λN diffraction halo on each side.
            2.0 * (blur(p).0 + 16.0 * p.lambda_f_mm())
        })
        .fold(0.0, f64::max)
}

pub(crate) fn image_grid(pupils: &[(&PupilFunction, f64)], spacing: f64) -> usize {
    ((image_side(pupils) / spacing).ceil() as usize).clamp(64, MAX_AUTO_IMAGE)
}

pub(crate) fn useful_frequency(pupils: &[(&PupilFunction, f64)]) -> f64 {
    pupils
        .iter()
        .map(|(p, _)| (0.25 / p.lambda_f_mm()).min(0.5 / blur(p).1).min(100.0))
        .fold(f64::INFINITY, f64::min)
}

pub(crate) fn round_frequency(f: f64) -> f64 {
    let decade = 10.0f64.powf(f.log10().floor());
    let v = f / decade;
    decade
        * if v >= 5.0 {
            5.0
        } else if v >= 2.0 {
            2.0
        } else {
            1.0
        }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{PupilGeometry, RaySample};

    #[test]
    fn defocus_sets_phase_sampling_and_geometric_blur() {
        // W = 10 rho²: max |dW/du| = 20; the image is a uniform
        // geometrical disk of radius 4 lambda N * 10.
        let m = 65;
        let samples = StructuredSamples {
            nx: m,
            ny: m,
            samples: (0..m * m)
                .map(|k| {
                    let u = -1.0 + 2.0 * (k % m) as f64 / (m - 1) as f64;
                    let v = -1.0 + 2.0 * (k / m) as f64 / (m - 1) as f64;
                    (u * u + v * v <= 1.0).then_some(RaySample {
                        u,
                        v,
                        opd_waves: 10.0 * (u * u + v * v),
                        amplitude: 1.0,
                    })
                })
                .collect(),
        };
        let n = pupil_grid(&samples);
        assert_eq!(n, 128);
        let geom = PupilGeometry::new(5.0, 100.0, 1.0).unwrap();
        let p = PupilFunction::from_fn(
            n,
            0.5,
            geom,
            4,
            |u, v| u * u + v * v <= 1.0,
            |u, v| (1.0, 10.0 * (u * u + v * v)),
        )
        .unwrap();
        assert!(p.max_phase_step() < 0.35);
        let (extent, rms) = blur(&p);
        let radius = 40.0 * p.lambda_f_mm();
        assert!((extent / radius - 1.0).abs() < 0.04);
        assert!((rms / (radius / 2.0f64.sqrt()) - 1.0).abs() < 0.04);
        assert!((useful_frequency(&[(&p, 1.0)]) * 2.0 * rms - 1.0).abs() < 1e-12);
        assert!(image_side(&[(&p, 1.0)]) > 2.0 * radius);
    }

    #[test]
    fn frequency_rounding_spans_decades() {
        for (f, want) in [(87.0, 50.0), (39.0, 20.0), (14.0, 10.0), (0.76, 0.5)] {
            assert!((round_frequency(f) - want).abs() < 1e-12);
        }
    }
}
