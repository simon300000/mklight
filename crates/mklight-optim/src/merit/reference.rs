//! References of per-ray operands over a [`RayGroup`](super::operand::RayGroup):
//! the spot centroid and the piston/tilt of the wavefront.
//!
//! Inputs are the group's traced rays: pupil coordinates, quadrature weights
//! (for a polychromatic group, `w_node · w_λ`) and the per-ray values. Failed
//! rays are simply left out; weights need not sum to 1.
//!
//! **Bilateral (half-pupil) groups.** With
//! [`PupilSymmetry::Bilateral`] only `px ≥ 0` is traced and the half pupil
//! stands for the full pupil mirrored about the meridional plane. Under that
//! symmetry `εx(−px, py) = −εx(px, py)` and `εy`, OPD are even in `px`, so
//! over the full pupil the x centroid and the x tilt are exactly zero, while
//! the y centroid, piston and y tilt equal their half-pupil weighted values
//! (each mirrored ray duplicates an equation with the same weight). Taking
//! the plain half-pupil mean of `εx` instead would give a spurious, non-zero
//! x centroid — the helpers below never do that.

use super::quadrature::PupilSymmetry;
use crate::linalg::{Matrix, Qr};

/// Weighted spot centroid `(cx, cy)` (mm) of transverse aberrations
/// `(εx, εy)`. `None` when no ray has positive weight.
pub fn spot_centroid(
    weights: &[f64],
    ex: &[f64],
    ey: &[f64],
    symmetry: PupilSymmetry,
) -> Option<[f64; 2]> {
    let mut sw = mklight_core::math::KahanSum::default();
    let mut sx = mklight_core::math::KahanSum::default();
    let mut sy = mklight_core::math::KahanSum::default();
    for ((w, x), y) in weights.iter().zip(ex).zip(ey) {
        sw.add(*w);
        sx.add(w * x);
        sy.add(w * y);
    }
    let w = sw.value();
    if !(w > 0.0) {
        return None;
    }
    let cx = match symmetry {
        PupilSymmetry::None => sx.value() / w,
        PupilSymmetry::Bilateral => 0.0,
    };
    Some([cx, sy.value() / w])
}

/// Weighted least-squares piston and tilt `[a, bx, by]` of
/// `OPD ≈ a + bx·px + by·py` (waves; tilts per unit normalised pupil
/// coordinate). `bx = 0` for a bilateral group. `None` when the rays do not
/// determine the fit (fewer independent rays than coefficients).
pub fn piston_tilt(
    px: &[f64],
    py: &[f64],
    weights: &[f64],
    opd: &[f64],
    symmetry: PupilSymmetry,
) -> Option<[f64; 3]> {
    let rows: Vec<usize> = (0..opd.len()).filter(|&k| weights[k] > 0.0).collect();
    let bilateral = symmetry == PupilSymmetry::Bilateral;
    let cols = if bilateral { 2 } else { 3 };
    if rows.len() < cols {
        return None;
    }
    // Rows scaled by √w; QR, never the normal equations.
    let mut a = Matrix::zeros(rows.len(), cols);
    let mut b = vec![0.0; rows.len()];
    for (r, &k) in rows.iter().enumerate() {
        let s = weights[k].sqrt();
        a[(r, 0)] = s;
        if bilateral {
            a[(r, 1)] = s * py[k];
        } else {
            a[(r, 1)] = s * px[k];
            a[(r, 2)] = s * py[k];
        }
        b[r] = s * opd[k];
    }
    let qr = Qr::new(a);
    if !qr.is_full_rank() {
        return None;
    }
    let c = qr.solve_ls(&b)?;
    Some(if bilateral {
        [c[0], 0.0, c[1]]
    } else {
        [c[0], c[1], c[2]]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merit::quadrature::{PupilNode, PupilSampling};

    /// Bilateral `arms` arms ≡ full pupil with `2·arms` arms rotated so the
    /// nodes are exactly the mirrored half pupil.
    fn pair(rings: usize, arms: usize) -> (Vec<PupilNode>, Vec<PupilNode>) {
        let half = PupilSampling::bilateral(rings, arms).nodes();
        let full = PupilSampling {
            rings,
            arms: 2 * arms,
            arm_offset_deg: (-90.0 + 90.0 / arms as f64),
            ..Default::default()
        }
        .nodes();
        (half, full)
    }

    // A meridionally symmetric aberration pattern: εx odd, εy and W even in px.
    fn ex(px: f64, py: f64) -> f64 {
        px * (0.3 + 0.8 * (px * px + py * py)) + 0.5 * px * py
    }
    fn ey(px: f64, py: f64) -> f64 {
        0.2 + 0.4 * py + 0.9 * py * py + 0.6 * px * px + 0.1 * py.powi(3)
    }
    fn w(px: f64, py: f64) -> f64 {
        0.25 + 0.15 * py + 0.5 * (px * px + py * py) + 0.7 * px * px * py
    }

    #[test]
    fn bilateral_reference_equals_mirrored_full_pupil() {
        let (half, full) = pair(4, 5);
        let ev = |nodes: &[PupilNode], f: fn(f64, f64) -> f64| -> Vec<f64> {
            nodes.iter().map(|n| f(n.px, n.py)).collect()
        };
        let wt = |nodes: &[PupilNode]| nodes.iter().map(|n| n.weight).collect::<Vec<_>>();
        let c_full = spot_centroid(
            &wt(&full),
            &ev(&full, ex),
            &ev(&full, ey),
            PupilSymmetry::None,
        )
        .unwrap();
        let c_half = spot_centroid(
            &wt(&half),
            &ev(&half, ex),
            &ev(&half, ey),
            PupilSymmetry::Bilateral,
        )
        .unwrap();
        assert!(c_full[0].abs() < 1e-14, "{c_full:?}");
        assert!((c_full[1] - c_half[1]).abs() < 1e-14 && c_half[0] == 0.0);
        // The naive half-pupil mean of εx is far from the true centroid.
        let naive = spot_centroid(
            &wt(&half),
            &ev(&half, ex),
            &ev(&half, ey),
            PupilSymmetry::None,
        )
        .unwrap();
        assert!(naive[0] > 0.1, "{naive:?}");

        let coords = |nodes: &[PupilNode]| -> (Vec<f64>, Vec<f64>) {
            (
                nodes.iter().map(|n| n.px).collect(),
                nodes.iter().map(|n| n.py).collect(),
            )
        };
        let (fx, fy) = coords(&full);
        let (hx, hy) = coords(&half);
        let pt_full =
            piston_tilt(&fx, &fy, &wt(&full), &ev(&full, w), PupilSymmetry::None).unwrap();
        let pt_half = piston_tilt(
            &hx,
            &hy,
            &wt(&half),
            &ev(&half, w),
            PupilSymmetry::Bilateral,
        )
        .unwrap();
        for k in 0..3 {
            assert!(
                (pt_full[k] - pt_half[k]).abs() < 1e-13,
                "{pt_full:?} {pt_half:?}"
            );
        }
        assert!(pt_full[1].abs() < 1e-13);
    }

    #[test]
    fn piston_tilt_recovers_exact_plane_and_known_moments() {
        let nodes = PupilSampling {
            rings: 3,
            arms: 7,
            ..Default::default()
        }
        .nodes();
        let px: Vec<f64> = nodes.iter().map(|n| n.px).collect();
        let py: Vec<f64> = nodes.iter().map(|n| n.py).collect();
        let wt: Vec<f64> = nodes.iter().map(|n| n.weight).collect();
        let plane: Vec<f64> = nodes
            .iter()
            .map(|n| 0.3 - 0.2 * n.px + 0.7 * n.py)
            .collect();
        let c = piston_tilt(&px, &py, &wt, &plane, PupilSymmetry::None).unwrap();
        assert!((c[0] - 0.3).abs() < 1e-14 && (c[1] + 0.2).abs() < 1e-14);
        assert!((c[2] - 0.7).abs() < 1e-14);
        // Defocus ρ² over the unit disk: piston = mean ρ² = 1/2, no tilt.
        // Coma-like ρ²·py: tilt = <ρ² py²>/<py²> = (1/6)/(1/4) = 2/3.
        let w2: Vec<f64> = nodes
            .iter()
            .map(|n| (n.px * n.px + n.py * n.py) * (1.0 + n.py))
            .collect();
        let c = piston_tilt(&px, &py, &wt, &w2, PupilSymmetry::None).unwrap();
        assert!((c[0] - 0.5).abs() < 1e-14 && c[1].abs() < 1e-14);
        assert!((c[2] - 2.0 / 3.0).abs() < 1e-14, "{c:?}");
        // Under-determined → None.
        assert!(
            piston_tilt(
                &px[..2],
                &py[..2],
                &wt[..2],
                &plane[..2],
                PupilSymmetry::None
            )
            .is_none()
        );
    }
}
