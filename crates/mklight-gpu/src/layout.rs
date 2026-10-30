//! The packed f32 system layout: Rust mirror of `include/mklight_gpu_types.h`
//! and the packer that turns a `ResolvedSystem` into it.
//!
//! The C header is authoritative; every struct here is `#[repr(C)]`, `Pod`,
//! and its size, alignment and field offsets are asserted equal to the header
//! (compile-time size asserts below, offset tests against a C compiler in
//! `tests/c_layout.rs`, and against the Metal compiler in mklight-metal).
//!
//! Packing rounds the f64 model once to f32. Values whose rounding would lose
//! more than the error budget allows are carried with an f32 residual
//! (`to_next_origin_lo`). Anything the layout cannot represent (too many
//! wavelengths, asphere terms beyond A22, a model parameter whose f32 is
//! infinite, subnormal or a spurious zero) is an error: the GPU never silently
//! computes a different system. Subnormals are excluded because GPUs flush
//! them to zero (`fenv::Ftz`). The only values written as an exact zero
//! although the f64 value is not are rotation entries and origin residuals
//! below `f32::MIN_POSITIVE` (≈ 1.2e-38), whose effect is far below one ulp
//! of the quantities they modify.

use bytemuck::{Pod, Zeroable};
use mklight_core::materials::coating::ResolvedCoating;
use mklight_core::math::Mat3;
use mklight_core::model::{ApertureShape, Interaction, SurfaceShape};
use mklight_core::snapshot::SnapshotHash;
use mklight_core::{MkError, MkResult, ResolvedSystem};

use crate::consts::*;
use crate::df::Df32;

pub const MAX_WAVELENGTHS: usize = MKL_MAX_WAVELENGTHS as usize;
pub const MAX_ASPHERE: usize = MKL_MAX_ASPHERE as usize;

/// Mirror of `struct MklPackedHeader`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PackedHeader {
    pub magic: u32,
    pub layout_version: u32,
    pub num_surfaces: u32,
    pub num_wavelengths: u32,
    pub surface_offset: u32,
    pub stop: u32,
    pub primary_wavelength: u32,
    pub flags: u32,
    pub wavelength_um: [f32; MAX_WAVELENGTHS],
    pub wavelength_source: [u32; MAX_WAVELENGTHS],
    pub reserved: [u32; 4],
}

/// Mirror of `struct MklPackedSurface`. See the header for units and the
/// sag/transform formulas.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Pod, Zeroable)]
pub struct PackedSurface {
    pub shape: u32,
    pub interaction: u32,
    pub aperture: u32,
    pub flags: u32,
    pub curvature: f32,
    pub conic: f32,
    pub paraxial_power: f32,
    pub asphere_inv_norm2: f32,
    pub num_asphere: u32,
    pub source_index: u32,
    pub coating: u32,
    pub curvature_lo: f32,
    pub asphere: [f32; MAX_ASPHERE],
    pub aperture_params: [f32; 4],
    pub to_next_rot: [f32; 9],
    pub to_next_origin: [f32; 3],
    pub to_next_origin_lo: [f32; 3],
    pub coating_r: f32,
    pub coating_t: f32,
    pub conic_lo: f32,
    pub n_after: [f32; MAX_WAVELENGTHS],
    pub alpha_after: [f32; MAX_WAVELENGTHS],
    pub vergence: [f32; MAX_WAVELENGTHS],
    pub to_next_rot_lo: [f32; 9],
    pub to_next_shift: [f32; 3],
    pub to_next_shift_lo: [f32; 3],
    pub paraxial_power_lo: f32,
    pub n_after_lo: [f32; MAX_WAVELENGTHS],
    pub vergence_lo: [f32; MAX_WAVELENGTHS],
    pub aperture_params_lo: [f32; 4],
    pub asphere_lo: [f32; MAX_ASPHERE],
    pub coefficient_offset: u32,
    pub coefficient_count: u32,
}

/// Mirror of `struct MklPhiloxParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct PhiloxParams {
    pub key: [u32; 2],
    pub stream: [u32; 2],
    pub first_block: [u32; 2],
    pub num_blocks: u32,
    pub reserved: u32,
}

/// Mirror of `struct MklReduceParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ReduceParams {
    pub count: u32,
    pub channel_stride: u32,
    pub num_groups: u32,
    pub first_group: u32,
}

/// Mirror of `struct MklTraceParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct TraceParams {
    pub num_rays: u32,
    pub stride: u32,
    pub num_steps: u32,
    pub start_surface: u32,
    pub flags: u32,
    pub max_iterations: u32,
    pub tolerance_abs: f32,
    pub tolerance_rel: f32,
}

/// Mirror of `struct MklSagParams`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct SagParams {
    pub surface: u32,
    pub count: u32,
    pub reserved: [u32; 2],
}

const _: () = {
    assert!(std::mem::size_of::<PackedHeader>() == MKL_PACKED_HEADER_SIZE as usize);
    assert!(std::mem::size_of::<PackedSurface>() == MKL_PACKED_SURFACE_SIZE as usize);
    assert!(std::mem::align_of::<PackedHeader>() == 4);
    assert!(std::mem::align_of::<PackedSurface>() == 4);
    assert!(std::mem::size_of::<PhiloxParams>() == 32);
    assert!(std::mem::size_of::<ReduceParams>() == 16);
    assert!(std::mem::size_of::<SagParams>() == 16);
    assert!(std::mem::size_of::<TraceParams>() == 32);
    assert!(MKL_PACKED_HEADER_SIZE.is_multiple_of(16));
    assert!(MKL_PACKED_SURFACE_SIZE.is_multiple_of(16));
};

/// Field offsets of the Rust mirrors, by C field name. Tests compare these
/// with the C and Metal compilers' `offsetof`.
pub fn header_offsets() -> Vec<(&'static str, usize)> {
    use std::mem::offset_of as o;
    type H = PackedHeader;
    vec![
        ("magic", o!(H, magic)),
        ("layout_version", o!(H, layout_version)),
        ("num_surfaces", o!(H, num_surfaces)),
        ("num_wavelengths", o!(H, num_wavelengths)),
        ("surface_offset", o!(H, surface_offset)),
        ("stop", o!(H, stop)),
        ("primary_wavelength", o!(H, primary_wavelength)),
        ("flags", o!(H, flags)),
        ("wavelength_um", o!(H, wavelength_um)),
        ("wavelength_source", o!(H, wavelength_source)),
        ("reserved", o!(H, reserved)),
    ]
}

/// See [`header_offsets`].
pub fn surface_offsets() -> Vec<(&'static str, usize)> {
    use std::mem::offset_of as o;
    type S = PackedSurface;
    vec![
        ("shape", o!(S, shape)),
        ("interaction", o!(S, interaction)),
        ("aperture", o!(S, aperture)),
        ("flags", o!(S, flags)),
        ("curvature", o!(S, curvature)),
        ("conic", o!(S, conic)),
        ("paraxial_power", o!(S, paraxial_power)),
        ("asphere_inv_norm2", o!(S, asphere_inv_norm2)),
        ("num_asphere", o!(S, num_asphere)),
        ("source_index", o!(S, source_index)),
        ("coating", o!(S, coating)),
        ("curvature_lo", o!(S, curvature_lo)),
        ("asphere", o!(S, asphere)),
        ("aperture_params", o!(S, aperture_params)),
        ("to_next_rot", o!(S, to_next_rot)),
        ("to_next_origin", o!(S, to_next_origin)),
        ("to_next_origin_lo", o!(S, to_next_origin_lo)),
        ("coating_r", o!(S, coating_r)),
        ("coating_t", o!(S, coating_t)),
        ("conic_lo", o!(S, conic_lo)),
        ("n_after", o!(S, n_after)),
        ("alpha_after", o!(S, alpha_after)),
        ("vergence", o!(S, vergence)),
        ("to_next_rot_lo", o!(S, to_next_rot_lo)),
        ("to_next_shift", o!(S, to_next_shift)),
        ("to_next_shift_lo", o!(S, to_next_shift_lo)),
        ("paraxial_power_lo", o!(S, paraxial_power_lo)),
        ("n_after_lo", o!(S, n_after_lo)),
        ("vergence_lo", o!(S, vergence_lo)),
        ("aperture_params_lo", o!(S, aperture_params_lo)),
        ("asphere_lo", o!(S, asphere_lo)),
        ("coefficient_offset", o!(S, coefficient_offset)),
        ("coefficient_count", o!(S, coefficient_count)),
    ]
}

/// Options for [`PackedSystem::pack`].
#[derive(Clone, Debug, Default)]
pub struct PackOptions {
    /// Model wavelength indices to pack, in order. `None` packs all of them
    /// (an error above [`MAX_WAVELENGTHS`]; use [`PackedSystem::pack_chunks`]).
    pub wavelengths: Option<Vec<usize>>,
    /// Per-surface radius hints (mm), e.g. the tracer's automatic
    /// semi-diameters. Only used to choose the asphere normalisation radius.
    pub radius_hints: Option<Vec<f64>>,
}

/// One system packed for one wavelength set.
#[derive(Clone, Debug)]
pub struct PackedSystem {
    pub header: PackedHeader,
    pub surfaces: Vec<PackedSurface>,
    /// Shared hi/lo pool, indexed in pairs by each surface.
    pub coefficients: Vec<Df32>,
    /// Content hash of the snapshot this was packed from (cache key and
    /// `RunInfo` provenance).
    pub snapshot_hash: SnapshotHash,
    /// Model wavelength index of each packed wavelength.
    pub wavelengths: Vec<usize>,
}

/// `x` rounded to f32, which must be a normal number or an exact zero that
/// stands for an exact zero (`v == 0 ⇔ x == 0`, see the sentinel rule in the
/// header).
fn f32_checked(x: f64, what: &str) -> MkResult<f32> {
    let v = x as f32;
    if !v.is_finite() || v.is_subnormal() || ((v == 0.0) != (x == 0.0)) {
        return Err(MkError::Unsupported(format!(
            "{what} = {x:e} is not representable as a normal f32"
        )));
    }
    Ok(v)
}

/// `x` as a double-float `(hi, lo)`: `hi = f32_checked(x)`, `lo` its
/// residual (zero when subnormal).
fn split_checked(x: f64, what: &str) -> MkResult<(f32, f32)> {
    let hi = f32_checked(x, what)?;
    Ok((hi, normal_or_zero((x - hi as f64) as f32)))
}

/// `v`, or a zero of the same sign if `v` is subnormal (what a flush-to-zero
/// device would read anyway); used only where the module docs allow it.
fn normal_or_zero(v: f32) -> f32 {
    if v.is_subnormal() {
        0f32.copysign(v)
    } else {
        v
    }
}

/// Choose `e` so that every power-series coefficient `c_j·2^(e(first + step·j))`
/// is a normal, finite f32 (or exactly zero): the normalised coefficients of
/// the radius `2^e`. Searches outward from the hint's exponent.
fn norm_exponent(series: &[&[f64]], hint_radius: f64, first: i32, step: i32) -> Option<i32> {
    let e0 = if hint_radius.is_finite() && hint_radius > 0.0 {
        hint_radius.log2().round() as i32
    } else {
        0
    };
    let ok = |e: i32| {
        series.iter().all(|c| {
            c.iter().enumerate().all(|(j, &a)| {
                a == 0.0 || ((a * 2f64.powi(e * (first + step * j as i32))) as f32).is_normal()
            })
        })
    };
    (0..=40)
        .flat_map(|d| [e0 + d, e0 - d])
        .find(|&e| (-60..=60).contains(&e) && ok(e))
}

/// `c_j·2^(e(first + step·j))` (exact: powers of two).
fn normalise(c: &[f64], e: i32, first: i32, step: i32) -> Vec<f64> {
    c.iter()
        .enumerate()
        .map(|(j, a)| a * 2f64.powi(e * (first + step * j as i32)))
        .collect()
}

/// `c` without trailing zeros (they add nothing but per-ray work).
fn trimmed(c: &[f64]) -> &[f64] {
    &c[..c.iter().rposition(|&a| a != 0.0).map_or(0, |k| k + 1)]
}

/// The record fields of a plane, conic, even asphere, ideal lens or
/// coordinate break (also a DOE's base).
fn pack_legacy(
    p: &mut PackedSurface,
    shape: &SurfaceShape,
    bounding: Option<f64>,
    what: &dyn Fn(&str) -> String,
) -> MkResult<()> {
    match shape {
        SurfaceShape::Plane => p.shape = MKL_SHAPE_PLANE,
        SurfaceShape::CoordinateBreak => p.shape = MKL_SHAPE_COORD_BREAK,
        SurfaceShape::Paraxial { focal_length } => {
            p.shape = MKL_SHAPE_PARAXIAL;
            if *focal_length == 0.0 || focal_length.is_nan() {
                return Err(MkError::InvalidModel(what("has zero focal length")));
            }
            (p.paraxial_power, p.paraxial_power_lo) =
                split_checked(1.0 / focal_length, &what("power"))?;
        }
        SurfaceShape::Conic { curvature, conic } => {
            p.shape = MKL_SHAPE_CONIC;
            p.curvature = f32_checked(*curvature, &what("curvature"))?;
            p.conic = f32_checked(*conic, &what("conic"))?;
            p.curvature_lo = normal_or_zero((curvature - p.curvature as f64) as f32);
            p.conic_lo = normal_or_zero((conic - p.conic as f64) as f32);
        }
        SurfaceShape::EvenAsphere {
            curvature,
            conic,
            coefficients,
        } => {
            p.shape = MKL_SHAPE_EVEN_ASPHERE;
            p.curvature = f32_checked(*curvature, &what("curvature"))?;
            p.conic = f32_checked(*conic, &what("conic"))?;
            p.curvature_lo = normal_or_zero((curvature - p.curvature as f64) as f32);
            p.conic_lo = normal_or_zero((conic - p.conic as f64) as f32);
            let coeffs = trimmed(coefficients);
            let used = coeffs.len();
            if used > MAX_ASPHERE {
                return Err(MkError::Unsupported(what(&format!(
                    "asphere term r^{} exceeds the GPU layout's A{}",
                    2 * used + 2,
                    2 * MAX_ASPHERE + 2
                ))));
            }
            if coeffs.iter().any(|a| !a.is_finite()) {
                return Err(MkError::InvalidModel(what("asphere coefficient")));
            }
            if used > 0 {
                let e =
                    norm_exponent(&[coeffs], bounding.unwrap_or(1.0), 4, 2).ok_or_else(|| {
                        MkError::Unsupported(what(
                            "asphere coefficients span too many decades for f32",
                        ))
                    })?;
                p.num_asphere = used as u32;
                p.asphere_inv_norm2 = 2f32.powi(-2 * e);
                for (j, v) in normalise(coeffs, e, 4, 2).into_iter().enumerate() {
                    p.asphere[j] = v as f32;
                    p.asphere_lo[j] = normal_or_zero((v - p.asphere[j] as f64) as f32);
                }
            }
        }
        _ => unreachable!("extended shape"),
    }
    Ok(())
}

/// Packs `shape`; extended shapes append their `MKL_COEFF_*` prefix and
/// coefficients to `pool` (the record does not grow).
fn pack_shape(
    p: &mut PackedSurface,
    shape: &SurfaceShape,
    bounding: Option<f64>,
    what: &dyn Fn(&str) -> String,
    wavelengths_um: &[f64],
    pool: &mut Vec<Df32>,
) -> MkResult<()> {
    use SurfaceShape::*;
    use mklight_core::model::surface::ZernikeOrdering;
    if shape.is_legacy() {
        return pack_legacy(p, shape, bounding, what);
    }
    let mut data = vec![0.0; MKL_COEFF_PREFIX as usize];
    data[MKL_COEFF_EXACT_CONIC as usize] =
        f64::from(mklight_core::surface_math::exact_conic(shape).is_some());
    (p.curvature, p.curvature_lo) = split_checked(shape.curvature(), &what("curvature"))?;
    (p.conic, p.conic_lo) = split_checked(shape.conic(), &what("conic"))?;
    let too_wide = || MkError::Unsupported(what("coefficients span too many decades for f32"));
    let (a, b) = match shape {
        OddAsphere { coefficients, .. } => {
            p.shape = MKL_SHAPE_ODD_ASPHERE;
            let c = trimmed(coefficients);
            let e = norm_exponent(&[c], bounding.unwrap_or(1.0), 1, 1).ok_or_else(too_wide)?;
            data[MKL_COEFF_RADIUS as usize] = 2f64.powi(e);
            (normalise(c, e, 1, 1), vec![])
        }
        QCon {
            norm_radius,
            coefficients,
            ..
        }
        | XYPolynomial {
            norm_radius,
            coefficients,
            ..
        }
        | ZernikeSag {
            norm_radius,
            coefficients,
            ..
        } => {
            p.shape = match shape {
                QCon { .. } => MKL_SHAPE_Q_CON,
                XYPolynomial { .. } => MKL_SHAPE_XY,
                _ => MKL_SHAPE_ZERNIKE,
            };
            data[MKL_COEFF_RADIUS as usize] = *norm_radius;
            if let ZernikeSag { ordering, .. } = shape {
                data[MKL_COEFF_KIND as usize] = f64::from(*ordering == ZernikeOrdering::Standard);
            }
            (trimmed(coefficients).to_vec(), vec![])
        }
        QBfs {
            norm_radius,
            coefficients,
            ..
        } => {
            p.shape = MKL_SHAPE_Q_BFS;
            data[MKL_COEFF_RADIUS as usize] = *norm_radius;
            // Forbes' triangular conversion to the auxiliary basis (SURFACES
            // §2) is independent of the ray: done once here, in f64.
            let a = trimmed(coefficients);
            let (mut f, mut g, mut h) = ([0.0f64; 20], [0.0f64; 20], [0.0f64; 20]);
            f[0] = 2.0;
            f[1] = 19.0f64.sqrt() / 2.0;
            g[0] = -0.5;
            for m in 2..20 {
                h[m - 2] = -((m * (m - 1)) as f64) / (2.0 * f[m - 2]);
                g[m - 1] = -(1.0 + g[m - 2] * h[m - 2]) / f[m - 1];
                f[m] =
                    ((m * (m + 1) + 3) as f64 - g[m - 1] * g[m - 1] - h[m - 2] * h[m - 2]).sqrt();
            }
            let mut b = [0.0; 22];
            for m in (0..a.len()).rev() {
                b[m] = (a[m] - g[m] * b[m + 1] - h[m] * b[m + 2]) / f[m];
            }
            (b[..a.len()].to_vec(), vec![])
        }
        Toroid {
            curvature_x,
            y_asphere,
            ..
        } => {
            p.shape = MKL_SHAPE_TOROID;
            data[MKL_COEFF_CX as usize] = *curvature_x;
            let y = trimmed(y_asphere);
            let e = norm_exponent(&[y], bounding.unwrap_or(1.0), 4, 2).ok_or_else(too_wide)?;
            data[MKL_COEFF_RADIUS as usize] = 2f64.powi(e);
            (vec![], normalise(y, e, 4, 2))
        }
        Biconic {
            curvature_x,
            conic_x,
            x_asphere,
            y_asphere,
            ..
        } => {
            p.shape = MKL_SHAPE_BICONIC;
            data[MKL_COEFF_CX as usize] = *curvature_x;
            data[MKL_COEFF_KX as usize] = *conic_x;
            let (x, y) = (trimmed(x_asphere), trimmed(y_asphere));
            let e = norm_exponent(&[x, y], bounding.unwrap_or(1.0), 4, 2).ok_or_else(too_wide)?;
            data[MKL_COEFF_RADIUS as usize] = 2f64.powi(e);
            (normalise(x, e, 4, 2), normalise(y, e, 4, 2))
        }
        Diffractive { base, phase } => {
            pack_legacy(p, base, bounding, what)?;
            data[MKL_COEFF_KIND as usize] = p.shape as f64;
            p.shape = MKL_SHAPE_DIFFRACTIVE;
            data[MKL_COEFF_RADIUS as usize] = phase.norm_radius;
            data[MKL_COEFF_TRACE_ORDER as usize] = phase.trace_order as f64;
            data[MKL_COEFF_DESIGN_ORDER as usize] = phase.design_order as f64;
            data[MKL_COEFF_DESIGN_WAVELENGTH as usize] = phase.design_wavelength_um;
            // m·λ/(2π) per packed wavelength, λ in mm (SURFACES §3).
            data.extend((0..MAX_WAVELENGTHS).map(|w| {
                wavelengths_um.get(w).map_or(0.0, |lambda| {
                    phase.trace_order as f64 * lambda * 1e-3 / std::f64::consts::TAU
                })
            }));
            (trimmed(&phase.coefficients).to_vec(), vec![])
        }
        Plane | Conic { .. } | EvenAsphere { .. } | Paraxial { .. } | CoordinateBreak => {
            unreachable!("legacy shape")
        }
    };
    data[MKL_COEFF_NA as usize] = a.len() as f64;
    data[MKL_COEFF_NB as usize] = b.len() as f64;
    data.extend(a);
    data.extend(b);
    p.coefficient_offset = u32::try_from(pool.len())
        .map_err(|_| MkError::Unsupported(what("coefficient pool too large")))?;
    p.coefficient_count = data.len() as u32;
    for value in data {
        let (hi, lo) = split_checked(value, &what("coefficient"))?;
        pool.push(Df32::new(hi, lo));
    }
    Ok(())
}

/// The pool entries of an extended surface, checked before any kernel
/// indexes them without bounds checks: the range, the prefix, every value
/// finite and not subnormal, a positive radius, and integral counts within
/// the shape's capacity that add up to `coefficient_count`. (Design order
/// and wavelength are metadata no kernel reads.)
fn validate_coefficients(s: &PackedSurface, pool: &[Df32]) -> Result<(), &'static str> {
    if s.shape <= MKL_SHAPE_COORD_BREAK {
        return match s.coefficient_count {
            0 => Ok(()),
            _ => Err("coefficients on a legacy shape"),
        };
    }
    let start = s.coefficient_offset as usize;
    let d = pool
        .get(start..start + s.coefficient_count as usize)
        .filter(|d| d.len() >= MKL_COEFF_PREFIX as usize)
        .ok_or("coefficient range")?;
    if d.iter().any(|v| {
        !(v.hi.is_finite() && v.lo.is_finite()) || v.hi.is_subnormal() || v.lo.is_subnormal()
    }) {
        return Err("coefficient value");
    }
    if d[MKL_COEFF_RADIUS as usize].hi <= 0.0 {
        return Err("normalisation radius");
    }
    let integer = |i: u32, max: u32| -> Result<usize, &'static str> {
        let v = d[i as usize];
        if v.lo == 0.0 && v.hi >= 0.0 && v.hi <= max as f32 && v.hi.fract() == 0.0 {
            Ok(v.hi as usize)
        } else {
            Err("coefficient metadata")
        }
    };
    let (max_a, max_b, extra) = match s.shape {
        MKL_SHAPE_ODD_ASPHERE | MKL_SHAPE_Q_CON | MKL_SHAPE_Q_BFS => (20, 0, 0),
        MKL_SHAPE_XY => (65, 0, 0),
        MKL_SHAPE_ZERNIKE => ([37, 45][integer(MKL_COEFF_KIND, 1)?], 0, 0),
        MKL_SHAPE_TOROID => (0, 10, 0),
        MKL_SHAPE_BICONIC => (10, 10, 0),
        _ => {
            // DOE: a plane, conic or even-asphere base and the wavelength
            // factors before the phase coefficients.
            integer(MKL_COEFF_KIND, MKL_SHAPE_EVEN_ASPHERE)?;
            (10, 0, MAX_WAVELENGTHS)
        }
    };
    let n = integer(MKL_COEFF_NA, max_a)? + integer(MKL_COEFF_NB, max_b)?;
    if d.len() != MKL_COEFF_PREFIX as usize + extra + n {
        return Err("coefficient count");
    }
    Ok(())
}

impl PackedSystem {
    /// Pack every wavelength of `rs` (at most [`MAX_WAVELENGTHS`]).
    pub fn pack_all(rs: &ResolvedSystem) -> MkResult<Self> {
        Self::pack(rs, &PackOptions::default())
    }

    /// Pack all wavelengths in chunks of at most [`MAX_WAVELENGTHS`].
    pub fn pack_chunks(rs: &ResolvedSystem, radius_hints: Option<Vec<f64>>) -> MkResult<Vec<Self>> {
        let n = rs.wavelengths_um.len();
        (0..n)
            .step_by(MAX_WAVELENGTHS)
            .map(|start| {
                let wl = (start..(start + MAX_WAVELENGTHS).min(n)).collect();
                Self::pack(
                    rs,
                    &PackOptions {
                        wavelengths: Some(wl),
                        radius_hints: radius_hints.clone(),
                    },
                )
            })
            .collect()
    }

    pub fn pack(rs: &ResolvedSystem, opts: &PackOptions) -> MkResult<Self> {
        let nw_model = rs.wavelengths_um.len();
        let wls: Vec<usize> = opts
            .wavelengths
            .clone()
            .unwrap_or_else(|| (0..nw_model).collect());
        if wls.is_empty() || wls.len() > MAX_WAVELENGTHS {
            return Err(MkError::Unsupported(format!(
                "a packed system holds 1..={MAX_WAVELENGTHS} wavelengths (got {}); pack in chunks",
                wls.len()
            )));
        }
        if let Some(&w) = wls.iter().find(|&&w| w >= nw_model) {
            return Err(MkError::InvalidModel(format!("no wavelength {w}")));
        }
        let n = rs.num_surfaces();
        if let Some(h) = &opts.radius_hints
            && h.len() != n
        {
            return Err(MkError::InvalidModel(format!(
                "radius_hints has {} entries for {n} surfaces",
                h.len()
            )));
        }

        let mut header = PackedHeader::zeroed();
        header.magic = MKL_PACKED_MAGIC;
        header.layout_version = MKL_LAYOUT_VERSION;
        header.num_surfaces = n as u32;
        header.num_wavelengths = wls.len() as u32;
        header.stop = rs.stop as u32;
        header.primary_wavelength = wls
            .iter()
            .position(|&w| w == rs.primary_wavelength)
            .map_or(MKL_INDEX_NONE, |p| p as u32);
        if rs.object_at_infinity {
            header.flags |= MKL_SYS_OBJECT_AT_INFINITY;
        }
        for (k, &w) in wls.iter().enumerate() {
            header.wavelength_um[k] = f32_checked(rs.wavelengths_um[w], "wavelength")?;
            header.wavelength_source[k] = w as u32;
        }

        let mut surfaces = Vec::with_capacity(n);
        let mut coefficients = Vec::new();
        let wavelengths_um: Vec<f64> = wls.iter().map(|&w| rs.wavelengths_um[w]).collect();
        for (i, s) in rs.surfaces.iter().enumerate() {
            let what = |f: &str| format!("surface {i} {f}");
            let mut p = PackedSurface::zeroed();
            p.source_index = i as u32;
            p.asphere_inv_norm2 = 1.0;
            p.interaction = match s.interaction {
                Interaction::Refract => MKL_INTERACT_REFRACT,
                Interaction::Reflect => MKL_INTERACT_REFLECT,
                Interaction::Dummy => MKL_INTERACT_DUMMY,
            };

            // Aperture: an explicit shape overrides a fixed semi-diameter.
            let hint = opts.radius_hints.as_ref().map(|h| h[i]);
            let mut bounding = hint;
            match (&s.aperture, s.fixed_semi_diameter) {
                (Some(a), _) => {
                    let (t, prm) = match *a {
                        ApertureShape::Circular { radius } => {
                            (MKL_APERTURE_CIRCULAR, [radius, 0.0])
                        }
                        ApertureShape::Annular {
                            inner_radius,
                            outer_radius,
                        } => (MKL_APERTURE_ANNULAR, [inner_radius, outer_radius]),
                        ApertureShape::Rectangular { half_x, half_y } => {
                            (MKL_APERTURE_RECTANGULAR, [half_x, half_y])
                        }
                        ApertureShape::Elliptical { semi_x, semi_y } => {
                            (MKL_APERTURE_ELLIPTICAL, [semi_x, semi_y])
                        }
                    };
                    p.aperture = t;
                    for (k, &v) in prm.iter().enumerate() {
                        (p.aperture_params[k], p.aperture_params_lo[k]) =
                            split_checked(v, &what("aperture"))?;
                    }
                    bounding = bounding.or(Some(a.bounding_radius()));
                }
                (None, Some(r)) => {
                    p.aperture = MKL_APERTURE_CIRCULAR;
                    p.flags |= MKL_SURF_APERTURE_FROM_SD;
                    (p.aperture_params[0], p.aperture_params_lo[0]) =
                        split_checked(r, &what("semi-diameter"))?;
                    bounding = bounding.or(Some(r));
                }
                (None, None) => p.aperture = MKL_APERTURE_NONE,
            }

            pack_shape(
                &mut p,
                &s.shape,
                bounding,
                &what,
                &wavelengths_um,
                &mut coefficients,
            )?;

            // The transform into the next frame. For an object at infinity,
            // surface 0 carries the entrance frame -> surface 1 map (header
            // docs), so a trace's starting frame is always surface 0's.
            let to_next = match (&s.to_next, i) {
                (Some(f), _) => Some((*f, s.to_next_is_axial)),
                (None, 0) if rs.object_at_infinity => {
                    let f = rs.surfaces[1].frame;
                    let axial = f.rot.is_identity() && f.origin.x == 0.0 && f.origin.y == 0.0;
                    Some((f, axial))
                }
                (None, _) => None,
            };
            if let Some((f, axial)) = to_next {
                p.flags |= MKL_SURF_HAS_TO_NEXT;
                if axial {
                    p.flags |= MKL_SURF_AXIAL;
                }
                let rt: Mat3 = f.rot.transpose();
                for r in 0..3 {
                    for c in 0..3 {
                        let v = rt.m[r][c];
                        let hi = normal_or_zero(v as f32);
                        p.to_next_rot[3 * r + c] = hi;
                        p.to_next_rot_lo[3 * r + c] = normal_or_zero((v - hi as f64) as f32);
                    }
                }
                let origin = [f.origin.x, f.origin.y, f.origin.z];
                for (k, &o) in origin.iter().enumerate() {
                    let hi = f32_checked(o, &what("position"))?;
                    p.to_next_origin[k] = hi;
                    p.to_next_origin_lo[k] = normal_or_zero((o - hi as f64) as f32);
                }
                // shift = rot · origin, in f64 (compensated dot product).
                for r in 0..3 {
                    let v = mklight_core::math::KahanSum::from_iter(
                        (0..3).map(|c| rt.m[r][c] * origin[c]),
                    )
                    .value();
                    let hi = normal_or_zero(v as f32);
                    if !hi.is_finite() {
                        return Err(MkError::Unsupported(what("position overflows f32")));
                    }
                    p.to_next_shift[r] = hi;
                    p.to_next_shift_lo[r] = normal_or_zero((v - hi as f64) as f32);
                }
                if !axial {
                    header.flags |= MKL_SYS_HAS_TILTS;
                }
            } else {
                p.to_next_rot = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
            }
            if i == rs.stop {
                p.flags |= MKL_SURF_IS_STOP;
            }
            if i == 0 {
                p.flags |= MKL_SURF_OBJECT;
            }
            if i == n - 1 {
                p.flags |= MKL_SURF_IMAGE;
            }
            match &s.coating {
                ResolvedCoating::Bare => p.coating = MKL_COATING_BARE,
                ResolvedCoating::Ideal {
                    reflectance,
                    transmittance,
                } => {
                    p.flags |= MKL_SURF_COATED;
                    p.coating = MKL_COATING_IDEAL;
                    p.coating_r = normal_or_zero(*reflectance as f32);
                    p.coating_t = normal_or_zero(*transmittance as f32);
                }
                ResolvedCoating::ThinFilm { .. } | ResolvedCoating::Tabulated { .. } => {
                    p.flags |= MKL_SURF_COATED;
                    p.coating = MKL_COATING_TABLE;
                }
            }
            for (k, &w) in wls.iter().enumerate() {
                p.n_after[k] = f32_checked(s.n_after[w], &what("index"))?;
                p.n_after_lo[k] = normal_or_zero((s.n_after[w] - p.n_after[k] as f64) as f32);
                p.alpha_after[k] = f32_checked(s.alpha_after[w], &what("absorption"))?;
                if p.shape == MKL_SHAPE_PARAXIAL {
                    let v = mklight_trace::paraxial::ideal_lens_object_vergence(rs, i, w);
                    (p.vergence[k], p.vergence_lo[k]) =
                        split_checked(v, &what("ideal-lens object vergence"))?;
                }
            }
            surfaces.push(p);
        }
        let out = PackedSystem {
            header,
            surfaces,
            coefficients,
            snapshot_hash: rs.snapshot.hash(),
            wavelengths: wls,
        };
        out.validate()?;
        Ok(out)
    }

    /// Checks a backend runs before uploading. Every field a kernel reads is
    /// checked (the fields are public, so a record need not come from the
    /// packer): magic and version, counts, enum codes, flag bits,
    /// `num_asphere ≤ MAX_ASPHERE`, finiteness of every used float, and
    /// physical ranges (`n_after > 0`, `alpha_after ≥ 0`, wavelengths > 0).
    pub fn validate(&self) -> MkResult<()> {
        let h = &self.header;
        let bad = |m: String| Err(MkError::Backend(format!("invalid packed system: {m}")));
        if h.magic != MKL_PACKED_MAGIC {
            return bad(format!("magic {:#x}", h.magic));
        }
        if h.layout_version != MKL_LAYOUT_VERSION {
            return bad(format!(
                "layout version {} (expected {MKL_LAYOUT_VERSION})",
                h.layout_version
            ));
        }
        if h.num_surfaces as usize != self.surfaces.len() || self.surfaces.len() < 3 {
            return bad(format!("{} surfaces", self.surfaces.len()));
        }
        if h.num_wavelengths == 0 || h.num_wavelengths > MKL_MAX_WAVELENGTHS {
            return bad(format!("{} wavelengths", h.num_wavelengths));
        }
        if h.stop == 0 || h.stop + 1 >= h.num_surfaces {
            return bad(format!("stop {}", h.stop));
        }
        let nw = h.num_wavelengths as usize;
        if h.primary_wavelength != MKL_INDEX_NONE && h.primary_wavelength >= h.num_wavelengths {
            return bad(format!("primary wavelength {}", h.primary_wavelength));
        }
        if h.flags & !(MKL_SYS_OBJECT_AT_INFINITY | MKL_SYS_HAS_TILTS) != 0 {
            return bad(format!("header flags {:#x}", h.flags));
        }
        if let Some(w) = h.wavelength_um[..nw]
            .iter()
            .find(|w| !(w.is_normal() && **w > 0.0))
        {
            return bad(format!("wavelength {w} µm"));
        }
        const SURF_FLAGS: u32 = MKL_SURF_HAS_TO_NEXT
            | MKL_SURF_AXIAL
            | MKL_SURF_IS_STOP
            | MKL_SURF_COATED
            | MKL_SURF_APERTURE_FROM_SD
            | MKL_SURF_OBJECT
            | MKL_SURF_IMAGE;
        for (i, s) in self.surfaces.iter().enumerate() {
            let bad = |m: &str| bad(format!("surface {i}: {m}"));
            if s.shape > MKL_SHAPE_ZERNIKE {
                return bad(&format!("shape code {}", s.shape));
            }
            if let Err(m) = validate_coefficients(s, &self.coefficients) {
                return bad(m);
            }
            if s.interaction > MKL_INTERACT_DUMMY {
                return bad(&format!("interaction code {}", s.interaction));
            }
            if s.aperture > MKL_APERTURE_ELLIPTICAL {
                return bad(&format!("aperture code {}", s.aperture));
            }
            if s.flags & !SURF_FLAGS != 0 {
                return bad(&format!("flags {:#x}", s.flags));
            }
            if s.coating > MKL_COATING_TABLE {
                return bad(&format!("coating code {}", s.coating));
            }
            if !(0.0..=1.0).contains(&s.coating_r)
                || !(0.0..=1.0).contains(&s.coating_t)
                || s.coating_r + s.coating_t > 1.0 + 1e-6
            {
                return bad("ideal coating needs R, T in [0, 1] and R + T <= 1");
            }
            if s.num_asphere as usize > MAX_ASPHERE {
                return bad(&format!("num_asphere {}", s.num_asphere));
            }
            let floats = [
                s.curvature,
                s.curvature_lo,
                s.conic,
                s.conic_lo,
                s.paraxial_power,
                s.paraxial_power_lo,
            ]
            .into_iter()
            .chain(s.asphere)
            .chain(s.aperture_params)
            .chain(s.to_next_rot)
            .chain(s.to_next_origin)
            .chain(s.to_next_origin_lo)
            .chain(s.to_next_rot_lo)
            .chain(s.to_next_shift)
            .chain(s.to_next_shift_lo)
            .chain(s.n_after_lo)
            .chain(s.vergence_lo)
            .chain(s.aperture_params_lo)
            .chain(s.asphere_lo);
            if floats.into_iter().any(|v| !v.is_finite()) {
                return bad("non-finite parameter");
            }
            if !(s.asphere_inv_norm2.is_normal() && s.asphere_inv_norm2 > 0.0) {
                return bad("asphere_inv_norm2");
            }
            if s.aperture_params.iter().any(|&a| a < 0.0) {
                return bad("negative aperture size");
            }
            if s.n_after[..nw].iter().any(|n| !(n.is_finite() && *n > 0.0)) {
                return bad("refractive index not finite and positive");
            }
            if s.alpha_after[..nw]
                .iter()
                .any(|a| !(a.is_finite() && *a >= 0.0))
            {
                return bad("absorption not finite and non-negative");
            }
            if s.vergence[..nw].iter().any(|v| !v.is_finite()) {
                return bad("ideal-lens vergence not finite");
            }
            if s.shape == MKL_SHAPE_PARAXIAL && s.paraxial_power == 0.0 {
                return bad("ideal lens with zero power");
            }
        }
        Ok(())
    }

    pub fn num_surfaces(&self) -> usize {
        self.surfaces.len()
    }
}

/// Several packed systems sharing one surface array (optimiser candidates,
/// tolerance samples, configurations). Header `k` points at its surfaces via
/// `surface_offset`; rays select a system by index.
#[derive(Clone, Debug, Default)]
pub struct PackedSystemSet {
    pub headers: Vec<PackedHeader>,
    pub surfaces: Vec<PackedSurface>,
    /// Shared hi/lo pool, indexed in pairs by each surface.
    pub coefficients: Vec<Df32>,
    pub hashes: Vec<SnapshotHash>,
}

impl PackedSystemSet {
    pub fn new(systems: &[PackedSystem]) -> MkResult<Self> {
        if systems.is_empty() {
            return Err(MkError::Backend("empty system set".into()));
        }
        let mut set = PackedSystemSet::default();
        for s in systems {
            s.validate()?;
            let mut h = s.header;
            h.surface_offset = u32::try_from(set.surfaces.len())
                .map_err(|_| MkError::Backend("too many surfaces".into()))?;
            set.headers.push(h);
            let offset = u32::try_from(set.coefficients.len())
                .map_err(|_| MkError::Backend("coefficient pool too large".into()))?;
            for surface in &s.surfaces {
                let mut surface = *surface;
                surface.coefficient_offset = surface
                    .coefficient_offset
                    .checked_add(offset)
                    .ok_or_else(|| MkError::Backend("coefficient pool too large".into()))?;
                set.surfaces.push(surface);
            }
            set.coefficients.extend_from_slice(&s.coefficients);
            set.hashes.push(s.snapshot_hash);
        }
        Ok(set)
    }
    pub fn len(&self) -> usize {
        self.headers.len()
    }
    pub fn is_empty(&self) -> bool {
        self.headers.is_empty()
    }
    pub fn header_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.headers)
    }
    pub fn surface_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.surfaces)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fmath::packed_sag;
    use mklight_core::Wavelength;
    use mklight_core::model::Placement;

    use crate::testutil::{resolve, singlet};

    #[test]
    fn packs_singlet() {
        let rs = resolve(singlet());
        let p = PackedSystem::pack_all(&rs).unwrap();
        assert_eq!(p.header.num_surfaces, 4);
        assert_eq!(p.header.num_wavelengths, 3);
        assert_eq!(p.header.primary_wavelength, 1);
        assert_eq!(
            p.header.flags & MKL_SYS_OBJECT_AT_INFINITY,
            MKL_SYS_OBJECT_AT_INFINITY
        );
        assert_eq!(p.header.flags & MKL_SYS_HAS_TILTS, 0);
        // Object at infinity: surface 0 carries the entrance frame -> surface
        // 1 map (here the identity, an axial zero shift).
        let s0 = &p.surfaces[0];
        assert_ne!(s0.flags & MKL_SURF_HAS_TO_NEXT, 0);
        assert_ne!(s0.flags & MKL_SURF_AXIAL, 0);
        assert_eq!(s0.to_next_origin, [0.0; 3]);
        assert_ne!(s0.flags & MKL_SURF_OBJECT, 0);
        let s1 = &p.surfaces[1];
        assert_eq!(s1.shape, MKL_SHAPE_CONIC);
        assert_eq!(s1.curvature, 0.02);
        assert_eq!(s1.aperture, MKL_APERTURE_CIRCULAR);
        assert_ne!(s1.flags & MKL_SURF_APERTURE_FROM_SD, 0);
        assert_ne!(s1.flags & MKL_SURF_IS_STOP, 0);
        assert_ne!(s1.flags & MKL_SURF_AXIAL, 0);
        assert_eq!(s1.to_next_origin, [0.0, 0.0, 5.0]);
        assert_eq!(s1.to_next_origin_lo, [0.0; 3]);
        assert!((s1.n_after[1] - 1.5168).abs() < 3e-5);
        let s2 = &p.surfaces[2];
        assert_eq!(s2.shape, MKL_SHAPE_EVEN_ASPHERE);
        assert_eq!(s2.aperture, MKL_APERTURE_RECTANGULAR);
        assert_eq!(s2.aperture_params[..2], [10.0, 8.0]);
        // Trailing zero coefficient dropped, interior zero kept.
        assert_eq!(s2.num_asphere, 4);
        // Thickness 45.123456789 is not an f32: hi + lo carries it to ~1e-14 mm.
        let t = s2.to_next_origin[2] as f64 + s2.to_next_origin_lo[2] as f64;
        assert!((t - 45.123456789).abs() < 1e-12);
        assert_ne!(s2.to_next_origin_lo[2], 0.0);
        assert_eq!(p.surfaces[3].flags & MKL_SURF_HAS_TO_NEXT, 0);
        assert_ne!(p.surfaces[3].flags & MKL_SURF_IMAGE, 0);
    }

    /// Normalisation is exact: un-normalising the packed coefficients gives
    /// the f32-rounded model coefficients, and the f32 sag agrees with the
    /// core's f64 definition to f32 accuracy.
    #[test]
    fn asphere_normalisation_exact_and_sag_matches() {
        let rs = resolve(singlet());
        let p = PackedSystem::pack_all(&rs).unwrap();
        let s2 = &p.surfaces[2];
        // Bounding radius hypot(10, 8) = 12.8 -> R = 16 = 2^4.
        assert_eq!(s2.asphere_inv_norm2, 2f32.powi(-8));
        let SurfaceShape::EvenAsphere { coefficients, .. } = &rs.surfaces[2].shape else {
            unreachable!()
        };
        for (j, &a) in coefficients.iter().take(4).enumerate() {
            let back = s2.asphere[j] as f64 * 16f64.powi(-(2 * j as i32 + 4));
            assert_eq!(back, a as f32 as f64, "coefficient {j}");
        }
        for k in 0..=50 {
            let h = 12.0 * k as f64 / 50.0;
            let exact = rs.surfaces[2].shape.sag(h * h).unwrap();
            let got = packed_sag(s2, (h * h) as f32) as f64;
            assert!((got - exact).abs() < 2e-6 * exact.abs().max(1e-3), "h={h}");
        }
    }

    /// A22-scale coefficients that would underflow f32 unnormalised survive.
    #[test]
    fn tiny_high_order_coefficients() {
        let mut s = singlet();
        let mut c = vec![0.0; 10];
        c[0] = 1e-6;
        c[9] = 3e-39; // A22: subnormal as a raw f32
        s.surfaces[2].shape = SurfaceShape::EvenAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: c,
        };
        let rs = resolve(s);
        let p = PackedSystem::pack_all(&rs).unwrap();
        let s2 = &p.surfaces[2];
        assert_eq!(s2.num_asphere, 10);
        assert!(s2.asphere.iter().all(|a| *a == 0.0 || a.is_normal()));
        let h2 = 144.0f64;
        let exact = rs.surfaces[2].shape.sag(h2).unwrap();
        let got = packed_sag(s2, h2 as f32) as f64;
        assert!((got - exact).abs() < 1e-6 * exact.abs());
    }

    #[test]
    fn rejects_unrepresentable() {
        let mut s = singlet();
        s.surfaces[2].shape = SurfaceShape::EvenAsphere {
            curvature: 0.0,
            conic: 0.0,
            coefficients: vec![0.0; 10].into_iter().chain([1e-40]).collect(),
        };
        let rs = resolve(s);
        // The message names the surface (lost once when packing moved into
        // a per-shape helper).
        assert!(matches!(
            PackedSystem::pack_all(&rs),
            Err(MkError::Unsupported(m)) if m.starts_with("surface 2 asphere term r^24")
        ));

        let mut s = singlet();
        s.wavelengths.wavelengths = (0..20)
            .map(|i| Wavelength {
                um: 0.45 + 0.01 * i as f64,
                weight: 1.0,
            })
            .collect();
        let rs = resolve(s);
        assert!(PackedSystem::pack_all(&rs).is_err());
        let chunks = PackedSystem::pack_chunks(&rs, None).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].header.num_wavelengths, 16);
        assert_eq!(chunks[1].header.num_wavelengths, 4);
        assert_eq!(chunks[1].wavelengths, vec![16, 17, 18, 19]);
        assert_eq!(chunks[1].header.wavelength_source[0], 16);
        assert_eq!(chunks[1].header.primary_wavelength, MKL_INDEX_NONE);
    }

    /// The packed transform reproduces the core's f64 frame mapping:
    /// `p_next = rot · (p − origin)` equals `to_next.to_local_point(p)`.
    #[test]
    fn tilted_transform_matches_core() {
        let mut s = singlet();
        s.surfaces[2].placement = Some(Placement {
            decenter_x: 0.3,
            decenter_y: -0.2,
            tilt_x: 2.0,
            tilt_y: -1.5,
            tilt_z: 10.0,
            ..Default::default()
        });
        let rs = resolve(s);
        let p = PackedSystem::pack_all(&rs).unwrap();
        assert_ne!(p.header.flags & MKL_SYS_HAS_TILTS, 0);
        for i in [1, 2] {
            let f = rs.surfaces[i].to_next.unwrap();
            let ps = &p.surfaces[i];
            assert_eq!(
                ps.flags & MKL_SURF_AXIAL != 0,
                rs.surfaces[i].to_next_is_axial
            );
            let pt = mklight_core::Vec3::new(1.5, -2.0, 4.0);
            let want = f.to_local_point(pt);
            let o: Vec<f64> = (0..3)
                .map(|k| ps.to_next_origin[k] as f64 + ps.to_next_origin_lo[k] as f64)
                .collect();
            let d = [pt.x - o[0], pt.y - o[1], pt.z - o[2]];
            for r in 0..3 {
                let got: f64 = (0..3)
                    .map(|c| ps.to_next_rot[3 * r + c] as f64 * d[c])
                    .sum();
                let w = [want.x, want.y, want.z][r];
                assert!((got - w).abs() < 1e-6, "surface {i} row {r}: {got} vs {w}");
            }
        }
    }

    #[test]
    fn system_set_offsets() {
        let rs = resolve(singlet());
        let a = PackedSystem::pack_all(&rs).unwrap();
        let set = PackedSystemSet::new(&[a.clone(), a]).unwrap();
        assert_eq!(set.headers[0].surface_offset, 0);
        assert_eq!(set.headers[1].surface_offset, 4);
        assert_eq!(
            set.surface_bytes().len(),
            8 * MKL_PACKED_SURFACE_SIZE as usize
        );
        assert_eq!(
            set.header_bytes().len(),
            2 * MKL_PACKED_HEADER_SIZE as usize
        );
    }

    /// Model values whose f32 would be subnormal or a spurious zero are
    /// rejected, not silently changed.
    #[test]
    fn rejects_values_that_vanish_in_f32() {
        for c in [1e-46, 1e-40, -3e-39] {
            let mut s = singlet();
            s.surfaces[1].shape = SurfaceShape::Conic {
                curvature: c,
                conic: 0.0,
            };
            let rs = resolve(s);
            assert!(
                matches!(PackedSystem::pack_all(&rs), Err(MkError::Unsupported(_))),
                "curvature {c}"
            );
        }
        // The smallest normal f32 and exact zero are fine.
        let mut s = singlet();
        s.surfaces[1].shape = SurfaceShape::Conic {
            curvature: f32::MIN_POSITIVE as f64,
            conic: 0.0,
        };
        let p = PackedSystem::pack_all(&resolve(s)).unwrap();
        assert_eq!(p.surfaces[1].curvature, f32::MIN_POSITIVE);
        assert_eq!(p.surfaces[1].conic, 0.0);
    }

    /// `validate` checks every record a kernel reads, so a hand-modified
    /// system cannot reach a backend.
    #[test]
    fn validate_checks_every_record() {
        let good = PackedSystem::pack_all(&resolve(singlet())).unwrap();
        assert!(good.validate().is_ok());
        type Tamper = fn(&mut PackedSystem);
        let cases: [(&str, Tamper); 12] = [
            ("num_asphere", |p| p.surfaces[2].num_asphere = 11),
            ("shape", |p| p.surfaces[1].shape = 99),
            ("interaction", |p| p.surfaces[1].interaction = 3),
            ("aperture", |p| p.surfaces[1].aperture = 5),
            ("flags", |p| p.surfaces[1].flags |= 1 << 20),
            ("curvature", |p| p.surfaces[1].curvature = f32::NAN),
            ("rot", |p| p.surfaces[1].to_next_rot[4] = f32::INFINITY),
            ("inv_norm2", |p| p.surfaces[2].asphere_inv_norm2 = 0.0),
            ("index", |p| p.surfaces[1].n_after[2] = 0.0),
            ("absorption", |p| p.surfaces[1].alpha_after[0] = -1.0),
            ("primary", |p| p.header.primary_wavelength = 3),
            ("wavelength", |p| p.header.wavelength_um[1] = f32::NAN),
        ];
        for (what, f) in cases {
            let mut p = good.clone();
            f(&mut p);
            assert!(p.validate().is_err(), "{what}");
            assert!(PackedSystemSet::new(&[p]).is_err(), "{what}");
        }
        // Unused wavelength slots are not checked (they are zero).
        let mut p = good.clone();
        p.surfaces[1].n_after[15] = 0.0;
        assert!(p.validate().is_ok());
    }

    #[test]
    fn coefficient_pool_rebases_systems_and_rejects_bad_ranges() {
        use crate::{surface_math::data, testutil};
        let systems: Vec<_> = testutil::extended_shapes()
            .into_iter()
            .map(|(_, shape)| {
                PackedSystem::pack_all(&testutil::resolve(testutil::one_surface(shape))).unwrap()
            })
            .collect();
        let set = PackedSystemSet::new(&systems).unwrap();
        for (h, p) in set.headers.iter().zip(&systems) {
            let s = &set.surfaces[h.surface_offset as usize + 1];
            assert_eq!(
                data(s, &set.coefficients),
                data(&p.surfaces[1], &p.coefficients)
            );
        }
        type Tamper = fn(&mut PackedSystem);
        let cases: [(&str, Tamper); 6] = [
            ("offset", |p| p.surfaces[1].coefficient_offset = u32::MAX),
            ("count", |p| p.surfaces[1].coefficient_count = u32::MAX),
            ("radius", |p| p.coefficients[0].hi = 0.0),
            ("capacity", |p| {
                p.coefficients[MKL_COEFF_NA as usize].hi = 21.0
            }),
            ("fraction", |p| {
                p.coefficients[MKL_COEFF_NA as usize].hi = 1.5
            }),
            ("NaN", |p| p.coefficients.last_mut().unwrap().lo = f32::NAN),
        ];
        for (what, f) in cases {
            let mut p = systems[0].clone();
            f(&mut p);
            assert!(p.validate().is_err(), "{what}");
        }
    }
}
