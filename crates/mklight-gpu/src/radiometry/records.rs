//! Immutable material/wavelength boundary. Layer indices are resolved once per
//! material state; angle evaluation and matrix products run on the backend.
use crate::{Df32, consts::*, resident::bad};
use bytemuck::{Pod, Zeroable};
use mklight_core::{MkResult, materials::coating::ResolvedCoating};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CoatingSpec {
    pub kind: u32,
    pub offset: u32,
    pub count: u32,
    pub mirror: u32,
    pub wavelength: Df32,
    pub n_in: Df32,
    pub n_out: Df32,
    pub substrate_n: Df32,
    pub substrate_k: Df32,
    pub reflectance: Df32,
    pub transmittance: Df32,
    pub reserved: Df32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CoatingLayer {
    pub n: Df32,
    pub k: Df32,
    pub thickness: Df32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CoatingQuery {
    pub coating: u32,
    pub reserved: u32,
    pub cosine: Df32,
}
impl CoatingQuery {
    pub fn new(coating: u32, cosine: f64) -> MkResult<Self> {
        if !cosine.is_finite() || !(0.0..=1.0).contains(&cosine) {
            return Err(bad("coating incidence cosine"));
        }
        Ok(Self {
            coating,
            reserved: 0,
            cosine: Df32::from_f64(cosine),
        })
    }
}
/// rs, rp, flux-normalised ts, tp (real/imaginary pairs), then
/// Rs, Rp, Ts, Tp, As, Ap. Absorption in films is a layer Poynting-flux
/// difference, not 1-R-T. A mirror includes power entering its substrate.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct CoatingValue {
    pub amplitudes: [Df32; 8],
    pub powers: [Df32; 6],
    pub status: u32,
    pub reserved: u32,
}
impl CoatingSpec {
    pub fn resolve(
        c: &ResolvedCoating,
        wavelength: f64,
        n_in: f64,
        n_out: f64,
        mirror: bool,
        layers: &mut Vec<CoatingLayer>,
    ) -> MkResult<Self> {
        let offset = layers.len() as u32;
        Self::resolve_with(c, wavelength, n_in, n_out, mirror, offset, |l| {
            layers.push(l)
        })
    }
    pub(super) fn resolve_with(
        c: &ResolvedCoating,
        wavelength: f64,
        n_in: f64,
        n_out: f64,
        mirror: bool,
        offset: u32,
        mut layer: impl FnMut(CoatingLayer),
    ) -> MkResult<Self> {
        if ![wavelength, n_in, n_out]
            .iter()
            .all(|v| v.is_finite() && *v > 0.0)
        {
            return Err(bad("coating wavelength or medium"));
        }
        let mut s = Self::zeroed();
        s.wavelength = Df32::from_f64(wavelength);
        s.n_in = Df32::from_f64(n_in);
        s.n_out = Df32::from_f64(n_out);
        s.mirror = mirror as u32;
        s.substrate_n = s.n_out;
        match c {
            ResolvedCoating::Bare => s.kind = MKL_RAD_BARE,
            ResolvedCoating::Ideal {
                reflectance,
                transmittance,
            } => {
                s.kind = MKL_RAD_IDEAL;
                s.reflectance = Df32::from_f64(*reflectance);
                s.transmittance = Df32::from_f64(*transmittance);
            }
            ResolvedCoating::Tabulated { reflectance } => {
                s.kind = MKL_RAD_TABLE;
                s.reflectance = Df32::from_f64(
                    mklight_core::materials::dispersion::monotone_cubic(reflectance, wavelength)
                        .clamp(0.0, 1.0),
                );
            }
            ResolvedCoating::ThinFilm {
                layers: ls,
                mirror_substrate,
                ..
            } => {
                s.kind = MKL_RAD_FILM;
                if ls.len() > MKL_RAD_MAX_LAYERS as usize {
                    return crate::resident::unsupported("coating stack exceeds 32 layers");
                }
                s.offset = offset;
                s.count = ls.len() as u32;
                if mirror {
                    let (n, k) = mirror_substrate.unwrap_or((1e-9, 1e9));
                    s.substrate_n = Df32::from_f64(n);
                    s.substrate_k = Df32::from_f64(k);
                }
                for j in 0..ls.len() {
                    let i = if mirror || n_out >= n_in {
                        ls.len() - 1 - j
                    } else {
                        j
                    };
                    let l = &ls[i];
                    layer(CoatingLayer {
                        n: Df32::from_f64(l.material.index(wavelength)),
                        k: Df32::from_f64(l.extinction),
                        thickness: Df32::from_f64(l.thickness_um),
                    });
                }
            }
        }
        Ok(s)
    }
}

/// Surface-major, wavelength-minor primary-path metadata. kind: 0 passive,
/// 1 refract, 2 reflect, 3 ideal lens, 4 ideal mirror. The transfer maps the
/// preceding local hit into this frame; rotations map directions globally.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SurfaceSpec {
    pub coating: CoatingSpec,
    pub rotation: [Df32; 9],
    pub transfer: [Df32; 9],
    pub translation: [Df32; 3],
    pub alpha: Df32,
    pub kind: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PolarizationValue {
    /// Pupil and proper Jones matrices, row major complex pairs, including bulk.
    pub jones: [Df32; 8],
    pub proper: [Df32; 8],
    /// Image-local vector pupil, input x then y, three complex components each.
    pub fields: [Df32; 12],
    pub direction: [Df32; 3],
    /// Scalar product of averages; average of separate s/p products; polarised
    /// intensity; exact unpolarised intensity; diverted; coating loss; bulk loss.
    pub power: [Df32; 7],
    /// Proper D, retardance, D axis (3), fast axis (3), pupil retardance.
    pub properties: [Df32; 9],
    pub status: u32,
    /// Bit 0 proper retarder defined, bit 1 pupil retarder defined.
    pub defined: u32,
}
// Byte sizes of the matching MSL structs in mklight_radiometry_types.h
// (float2 members, 8-byte aligned).
const _: () = {
    use std::mem::size_of;
    assert!(size_of::<CoatingSpec>() == 80 && size_of::<CoatingLayer>() == 24);
    assert!(size_of::<CoatingQuery>() == 16 && size_of::<CoatingValue>() == 120);
    assert!(size_of::<SurfaceSpec>() == 264 && size_of::<PolarizationValue>() == 384);
};
pub fn surface_specs(
    sys: &mklight_core::ResolvedSystem,
    layers: &mut Vec<CoatingLayer>,
) -> MkResult<Vec<SurfaceSpec>> {
    let mut out = vec![];
    visit_surface_specs(
        sys,
        layers.len() as u32,
        |v| out.push(v),
        |v| layers.push(v),
    )?;
    Ok(out)
}
pub(super) fn layer_count(c: &ResolvedCoating) -> usize {
    match c {
        ResolvedCoating::ThinFilm { layers, .. } => layers.len(),
        _ => 0,
    }
}
pub(super) fn visit_surface_specs(
    sys: &mklight_core::ResolvedSystem,
    mut offset: u32,
    mut surface: impl FnMut(SurfaceSpec),
    mut layer: impl FnMut(CoatingLayer),
) -> MkResult<()> {
    use mklight_trace::polarization::StepKind;
    for i in 0..sys.surfaces.len() {
        let s = &sys.surfaces[i];
        let frame = if i == 0 {
            mklight_trace::trace::start_frame(sys)
        } else {
            s.frame
        };
        let transfer = if i == 0 {
            mklight_core::math::Frame::IDENTITY
        } else if i == 1 {
            mklight_trace::trace::start_frame(sys).relative_to(&frame)
        } else {
            {
                let f = sys.surfaces[i - 1].to_next.unwrap();
                let rot = f.rot.transpose();
                mklight_core::math::Frame {
                    rot,
                    origin: rot.mul_vec(-f.origin),
                }
            }
        };
        for w in 0..sys.wavelengths_um.len() {
            let coating = CoatingSpec::resolve_with(
                &s.coating,
                sys.wavelengths_um[w],
                sys.n_before(i, w),
                s.n_after[w],
                s.is_mirror(),
                offset,
                &mut layer,
            )?;
            offset += coating.count;
            let kind = if i == 0 || i == sys.image_index() {
                0
            } else {
                match StepKind::of(s) {
                    None => 0,
                    Some(StepKind::Refract) => 1,
                    Some(StepKind::Reflect) => 2,
                    Some(StepKind::IdealLens) => 3,
                    Some(StepKind::IdealMirror) => 4,
                }
            };
            surface(SurfaceSpec {
                coating,
                rotation: std::array::from_fn(|k| Df32::from_f64(frame.rot.m[k / 3][k % 3])),
                transfer: std::array::from_fn(|k| Df32::from_f64(transfer.rot.m[k / 3][k % 3])),
                translation: [transfer.origin.x, transfer.origin.y, transfer.origin.z]
                    .map(Df32::from_f64),
                alpha: Df32::from_f64(if i == 0 || (i == 1 && sys.object_at_infinity) {
                    0.0
                } else {
                    sys.surfaces[i - 1].alpha_after[w]
                }),
                kind,
                reserved: 0,
            });
        }
    }
    Ok(())
}
