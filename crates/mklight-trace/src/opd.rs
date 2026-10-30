//! Optical path difference (CONVENTIONS §4).
//!
//! `OPD(P) = OPL_chief − OPL_ray`, each measured from the object point (or,
//! at infinity, from the field's common plane wavefront through the global
//! origin, see [`crate::tracer`]) to the **reference sphere**:
//!
//! * centre `I`: the chief ray's intersection with the image surface;
//! * the sphere passes through the chief ray's exit-pupil point `E`. `E` is
//!   the point on the image-space chief ray whose local `z` equals the
//!   paraxial exit-pupil position (at the ray's wavelength):
//!   `E = I + s_E d_c` with `s_E = (z_XP − I_z)/d_{c,z}`. The radius is
//!   `R = |s_E|`; `sgn(s_E)` selects the sheet on the exit-pupil side.
//!
//! For a ray reaching the image surface at `Q` with unit direction `d`, write
//! `w = Q − I`, `b = d·w` and `p² = |w × d|²` (squared distance of the ray
//! line from `I`). The ray meets the sphere at `s = −b + sgn(s_E)√(R² − p²)`,
//! so
//!
//! `OPD = (OPL_c + n s_E) − (OPL_r + n s)
//!      = (OPL_c − OPL_r) + n (b + sgn(s_E) · p² / (R + √(R² − p²)))`,
//!
//! which is free of cancellation (no `√(R² − p²) − R`). `OPL_c − OPL_r` is
//! formed from the compensated path sums before rounding. An image-space
//! telecentric system (exit pupil at infinity) takes the limit `R → ∞` of
//! the same expression: the correction term vanishes and
//! `OPD = (OPL_c − OPL_r) + n (d·w)`, the phase of each ray's plane wave at
//! `I` (a defocused image beyond focus has positive OPD, as for finite `R`).
//! (An earlier version referred to the plane through `I` normal to the
//! chief ray, `n (w·d_c)/(d·d_c)`, which is not that limit and gave the
//! defocus with the opposite sign.)
//!
//! Results are reported in waves at the ray's wavelength (indices are
//! relative to air, so the air wavelength is the reference).

use mklight_core::Vec3;
use serde::{Deserialize, Serialize};

use crate::numeric::Compensated;
use crate::trace::{TraceOptions, TraceResult, TraceStatus};
use crate::tracer::{PreparedField, SequentialTracer};

/// The reference sphere of one (field, wavelength).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpdReference {
    pub chief_opl: Compensated,
    /// Chief-ray image-surface point (image local frame): the sphere centre.
    pub center: Vec3,
    pub chief_dir: Vec3,
    /// Signed distance `s_E` from the centre to the exit-pupil point along the
    /// chief ray (negative when the exit pupil precedes the image).
    /// Infinite for an image-space telecentric system.
    pub s_exit_pupil: f64,
    /// Index of image space.
    pub n_image: f64,
    pub wavelength_um: f64,
}

impl OpdReference {
    /// OPD in mm of a ray traced to the image surface; `None` unless the
    /// trace succeeded and the ray meets the reference sphere.
    pub fn opd_mm(&self, ray: &TraceResult) -> Option<f64> {
        if !ray.status.is_ok() {
            return None;
        }
        let d_opl = self.chief_opl.diff(&ray.opl);
        let w = ray.ray.pos - self.center;
        let d = ray.ray.dir;
        let n = self.n_image;
        if !self.s_exit_pupil.is_finite() {
            // R → ∞ limit of the sphere formula (module docs).
            return Some(n.mul_add(d.dot(w), d_opl));
        }
        let r = self.s_exit_pupil.abs();
        let sg = self.s_exit_pupil.signum();
        let b = d.dot(w);
        let p2 = w.cross(d).norm2();
        let rr = r * r;
        if p2 > rr {
            return None;
        }
        let corr = sg * p2 / (r + (rr - p2).sqrt());
        Some(n.mul_add(b + corr, d_opl))
    }

    /// OPD in waves at this reference's wavelength.
    pub fn opd_waves(&self, ray: &TraceResult) -> Option<f64> {
        self.opd_mm(ray).map(|o| o / (self.wavelength_um * 1e-3))
    }
}

impl SequentialTracer {
    /// Build the reference sphere of a field at wavelength `wl` from its
    /// (unclipped) chief ray. Fails with the chief ray's status.
    pub fn opd_reference(
        &self,
        field: &PreparedField,
        wl: usize,
    ) -> Result<OpdReference, TraceStatus> {
        let c = self.chief(field, wl);
        if !c.status.is_ok() {
            return Err(c.status);
        }
        let sys = self.system();
        let img = sys.image_index();
        let z_xp = self.paraxial().at(wl).exit_pupil_position;
        let s_e = if z_xp.is_finite() {
            if c.ray.dir.z == 0.0 {
                return Err(TraceStatus::Failed {
                    surface: img,
                    reason: crate::trace::FailReason::Degenerate,
                });
            }
            (z_xp - c.ray.pos.z) / c.ray.dir.z
        } else {
            f64::INFINITY
        };
        Ok(OpdReference {
            chief_opl: c.opl,
            center: c.ray.pos,
            chief_dir: c.ray.dir,
            s_exit_pupil: s_e,
            n_image: sys.n_before(img, wl),
            wavelength_um: sys.wavelengths_um[wl],
        })
    }

    /// OPD (waves) of one pupil point; `None` when the chief or the ray fails.
    pub fn opd(
        &self,
        field: &PreparedField,
        wl: usize,
        px: f64,
        py: f64,
        opts: &TraceOptions,
    ) -> Option<f64> {
        let r = self.opd_reference(field, wl).ok()?;
        r.opd_waves(&self.trace(field, wl, px, py, opts))
    }
}
