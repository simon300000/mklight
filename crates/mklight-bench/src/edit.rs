//! The unique tiny edit of a session: the curvature `c` of the first curved
//! surface changed by an amount that follows a serial number, so every edit
//! has a content hash of its own and nothing of an earlier call is reused.
//!
//! The harnesses perturb at different sizes, and a size is part of what a
//! harness measures (its records and digests are compared between builds),
//! so each perturbation has its own function, named by its size and sign
//! rule. Every function sets `session` to `base` with that one curvature
//! changed and returns `true`; when no surface of `base` after the object is
//! curved it returns `false` and leaves the session as it is.

use mklight_core::{OpticalSystem, param::ParamPath};
use mklight_engine::Session;

fn apply(session: &mut Session, base: &OpticalSystem, change: impl FnOnce(f64) -> f64) -> bool {
    let mut system = base.clone();
    let Some(surface) = system
        .surfaces
        .iter()
        .skip(1)
        .find(|s| s.shape.curvature() != 0.)
    else {
        return false;
    };
    let param = ParamPath::Curvature {
        surface: surface.id,
    };
    let c = surface.shape.curvature();
    param.set(&mut system, change(c)).unwrap();
    session.set_system(system).unwrap();
    true
}

/// The radius changed by parts per million: `c / (1 ± (1 + serial·1e-6)·1e-6)`,
/// the sign alternating with `serial` (`cost_calibration`, `project_bench`).
pub fn radius_ppm(session: &mut Session, base: &OpticalSystem, serial: usize) -> bool {
    let relative = (if serial.is_multiple_of(2) { 1. } else { -1. })
        * (1. + serial as f64 / 1_000_000.)
        * 1e-6;
    apply(session, base, |c| c / (1. + relative))
}

fn curvature_ppb(session: &mut Session, base: &OpticalSystem, serial: u64, sign: f64) -> bool {
    apply(session, base, |c| {
        c * (1. + sign * (1. + serial as f64 * 1e-3) * 1e-9)
    })
}

/// The curvature changed by parts per billion:
/// `c · (1 ± (1 + serial·1e-3)·1e-9)`, the sign alternating with `serial`
/// (`fieldmaps_profile`).
pub fn curvature_ppb_alternating(session: &mut Session, base: &OpticalSystem, serial: u64) -> bool {
    let sign = if serial.is_multiple_of(2) { 1. } else { -1. };
    curvature_ppb(session, base, serial, sign)
}

/// [`curvature_ppb_alternating`] without the alternation: the curvature
/// always grows, `c · (1 + (1 + serial·1e-3)·1e-9)` (`solver_profile`).
pub fn curvature_ppb_growing(session: &mut Session, base: &OpticalSystem, serial: u64) -> bool {
    curvature_ppb(session, base, serial, 1.)
}

/// The curvature changed by one to two parts per billion, for serials that
/// are unique between processes: `serial` is `pid << 20 | call counter`
/// (`backend_bench`).
///
/// Only the counter (and a small residue of the process id, for uniqueness
/// between processes) sets the size, so every edit stays between 1e-9 and
/// about 2e-9 relative for the first 10,000 calls. The sign comes from a
/// hash of the counter, so it does not follow the column order of an even
/// number of columns.
pub fn curvature_ppb_per_process(session: &mut Session, base: &OpticalSystem, serial: u64) -> bool {
    let counter = serial & 0xf_ffff;
    let process = (serial >> 20) % 1000;
    let sign = if counter.wrapping_mul(0x9e37_79b9_7f4a_7c15) >> 63 == 0 {
        1.
    } else {
        -1.
    };
    let size = (1. + counter as f64 * 1e-4 + process as f64 * 1e-6) * 1e-9;
    apply(session, base, |c| c * (1. + sign * size))
}

/// The curvature grown in steps of a tenth of a part per billion:
/// `c · (1 + step·1e-10)` (`stray_ghost_profile`).
pub fn curvature_tenth_ppb(session: &mut Session, base: &OpticalSystem, step: usize) -> bool {
    apply(session, base, |c| c * (1. + step as f64 * 1e-10))
}
