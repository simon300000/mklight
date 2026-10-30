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
    pub reserved0: u32,
    pub asphere: [f32; MAX_ASPHERE],
    pub aperture_params: [f32; 4],
    pub to_next_rot: [f32; 9],
    pub to_next_origin: [f32; 3],
    pub to_next_origin_lo: [f32; 3],
    pub coating_r: f32,
    pub coating_t: f32,
    pub reserved1: u32,
    pub n_after: [f32; MAX_WAVELENGTHS],
    pub alpha_after: [f32; MAX_WAVELENGTHS],
    pub vergence: [f32; MAX_WAVELENGTHS],
    pub to_next_rot_lo: [f32; 9],
    pub to_next_shift: [f32; 3],
    pub to_next_shift_lo: [f32; 3],
    pub reserved2: u32,
    pub n_after_lo: [f32; MAX_WAVELENGTHS],
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
        ("reserved0", o!(S, reserved0)),
        ("asphere", o!(S, asphere)),
        ("aperture_params", o!(S, aperture_params)),
        ("to_next_rot", o!(S, to_next_rot)),
        ("to_next_origin", o!(S, to_next_origin)),
        ("to_next_origin_lo", o!(S, to_next_origin_lo)),
        ("coating_r", o!(S, coating_r)),
        ("coating_t", o!(S, coating_t)),
        ("reserved1", o!(S, reserved1)),
        ("n_after", o!(S, n_after)),
        ("alpha_after", o!(S, alpha_after)),
        ("vergence", o!(S, vergence)),
        ("to_next_rot_lo", o!(S, to_next_rot_lo)),
        ("to_next_shift", o!(S, to_next_shift)),
        ("to_next_shift_lo", o!(S, to_next_shift_lo)),
        ("reserved2", o!(S, reserved2)),
        ("n_after_lo", o!(S, n_after_lo)),
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

/// `v`, or a zero of the same sign if `v` is subnormal (what a flush-to-zero
/// device would read anyway); used only where the module docs allow it.
fn normal_or_zero(v: f32) -> f32 {
    if v.is_subnormal() {
        0f32.copysign(v)
    } else {
        v
    }
}

/// Choose `e` so that every coefficient `A_(2j+4)·2^(e(2j+4))` is a normal,
/// finite f32 (or exactly zero). Searches outward from the hint's exponent.
fn asphere_norm_exponent(coeffs: &[f64], hint_radius: f64) -> Option<i32> {
    let e0 = if hint_radius.is_finite() && hint_radius > 0.0 {
        hint_radius.log2().round() as i32
    } else {
        0
    };
    let ok = |e: i32| {
        coeffs.iter().enumerate().all(|(j, &a)| {
            if a == 0.0 {
                return true;
            }
            let v = (a * 2f64.powi(e * (2 * j as i32 + 4))) as f32;
            v.is_normal()
        })
    };
    (0..=40)
        .flat_map(|d| [e0 + d, e0 - d])
        .find(|&e| (-60..=60).contains(&e) && ok(e))
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
                    p.aperture_params[0] = f32_checked(prm[0], &what("aperture"))?;
                    p.aperture_params[1] = f32_checked(prm[1], &what("aperture"))?;
                    bounding = bounding.or(Some(a.bounding_radius()));
                }
                (None, Some(r)) => {
                    p.aperture = MKL_APERTURE_CIRCULAR;
                    p.flags |= MKL_SURF_APERTURE_FROM_SD;
                    p.aperture_params[0] = f32_checked(r, &what("semi-diameter"))?;
                    bounding = bounding.or(Some(r));
                }
                (None, None) => p.aperture = MKL_APERTURE_NONE,
            }

            match &s.shape {
                SurfaceShape::Plane => p.shape = MKL_SHAPE_PLANE,
                SurfaceShape::CoordinateBreak => p.shape = MKL_SHAPE_COORD_BREAK,
                SurfaceShape::Paraxial { focal_length } => {
                    p.shape = MKL_SHAPE_PARAXIAL;
                    if *focal_length == 0.0 || focal_length.is_nan() {
                        return Err(MkError::InvalidModel(what("has zero focal length")));
                    }
                    p.paraxial_power = f32_checked(1.0 / focal_length, &what("power"))?;
                }
                SurfaceShape::Conic { curvature, conic } => {
                    p.shape = MKL_SHAPE_CONIC;
                    p.curvature = f32_checked(*curvature, &what("curvature"))?;
                    p.conic = f32_checked(*conic, &what("conic"))?;
                }
                SurfaceShape::EvenAsphere {
                    curvature,
                    conic,
                    coefficients,
                } => {
                    p.shape = MKL_SHAPE_EVEN_ASPHERE;
                    p.curvature = f32_checked(*curvature, &what("curvature"))?;
                    p.conic = f32_checked(*conic, &what("conic"))?;
                    let used = coefficients
                        .iter()
                        .rposition(|&a| a != 0.0)
                        .map_or(0, |k| k + 1);
                    if used > MAX_ASPHERE {
                        return Err(MkError::Unsupported(format!(
                            "surface {i}: asphere term r^{} exceeds the GPU layout's A{}",
                            2 * used + 2,
                            2 * MAX_ASPHERE + 2
                        )));
                    }
                    let coeffs = &coefficients[..used];
                    if coeffs.iter().any(|a| !a.is_finite()) {
                        return Err(MkError::InvalidModel(what("asphere coefficient")));
                    }
                    if used > 0 {
                        let e = asphere_norm_exponent(coeffs, bounding.unwrap_or(1.0)).ok_or_else(
                            || {
                                MkError::Unsupported(format!(
                                    "surface {i}: asphere coefficients span too many \
                                     decades for f32"
                                ))
                            },
                        )?;
                        p.num_asphere = used as u32;
                        p.asphere_inv_norm2 = 2f32.powi(-2 * e);
                        for (j, &a) in coeffs.iter().enumerate() {
                            p.asphere[j] = (a * 2f64.powi(e * (2 * j as i32 + 4))) as f32;
                        }
                    }
                }
            }

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
                    p.vergence[k] = f32_checked(v, &what("ideal-lens object vergence"))?;
                }
            }
            surfaces.push(p);
        }
        let out = PackedSystem {
            header,
            surfaces,
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
            if s.shape > MKL_SHAPE_COORD_BREAK {
                return bad(&format!("shape code {}", s.shape));
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
            let floats = [s.curvature, s.conic, s.paraxial_power]
                .into_iter()
                .chain(s.asphere)
                .chain(s.aperture_params)
                .chain(s.to_next_rot)
                .chain(s.to_next_origin)
                .chain(s.to_next_origin_lo)
                .chain(s.to_next_rot_lo)
                .chain(s.to_next_shift)
                .chain(s.to_next_shift_lo)
                .chain(s.n_after_lo);
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
            set.surfaces.extend_from_slice(&s.surfaces);
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
        assert!(matches!(
            PackedSystem::pack_all(&rs),
            Err(MkError::Unsupported(_))
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
}
