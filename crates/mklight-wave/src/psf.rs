//! Point spread functions: FFT (Debye/Fraunhofer), exact matrix Fourier
//! transform, direct Huygens–Fresnel integration, and polychromatic sums.
//!
//! # Image-plane coordinates
//!
//! A [`Psf`] is an `M × M` row-major grid (`data[j·M + i]`, `i` → image x,
//! sagittal; `j` → image y, tangential) with spacing `Δx` mm on the
//! **image surface**. Sample `(c, c)`, `c = M/2` (integer division), sits
//! at `center_mm`, the PSF's reference point (the chief-ray intersection;
//! for monochromatic results the pupil's `image_offset_mm`). Sample `(i, j)`
//! is at `center + ((i − c)·Δx, (j − c)·Δx)`.
//!
//! For an off-axis chief ray the transform is evaluated at the projected
//! points `B·x_s` (see the chief-ray frame in [`crate::pupil`]), always by
//! the matrix transform: separable when `B` is diagonal (chief ray in the
//! x–z or y–z plane), otherwise summed directly at `O(n²M²)` cost.
//! [`huygens_psf`] is the exact alternative for tilted planes.
//!
//! # FFT relations
//!
//! With `n` pupil samples zero-padded to an `M`-point transform, the image
//! spacing is `Δx = λ·F#·n/M = λ·F#/Q` (`Q = M/n` is the padding factor) and
//! the PSF is periodic with the alias-free field `L = n·λ·F#`. The intensity
//! PSF is band-limited to `1/(λF#)`, so Nyquist sampling needs
//! `Δx ≤ λF#/2` (`Q ≥ 2`); coarser sampling is computed but flagged. `M` is
//! rounded *up* to a 7-smooth length, so the spacing is never coarser than
//! requested (any change is reported as `SpacingAdjusted`).
//! The field at image point `x` is
//! `U(x) = Σ_j p_j · exp(+i2π·(NA/λ)·u_j·x)` with the Debye-weighted pupil
//! field `p` (see [`crate::pupil`]); a positive OPD tilt `W = t·v` then moves
//! the image toward −y, as transverse aberration `ε = −(λ/NA)·∂W/∂v` requires.
//!
//! # Normalisation
//!
//! * [`PsfNormalization::Strehl`]: the aberration-free PSF of the same
//!   aperture and amplitude peaks at 1, so the peak value is the Strehl
//!   ratio.
//! * [`PsfNormalization::UnitEnergy`]: `∫ PSF dA = 1` over the whole plane;
//!   values are fractions of energy per mm².
//!
//! Either way `total_energy` (`∫ PSF dA` over the whole plane, from
//! Parseval) and `diffraction_peak` record the conversion. The PSF is the
//! field intensity `|U|²`; on a plane inclined to the beam (an off-axis
//! image surface, a tilted Huygens plane) its integral exceeds the flux by
//! the obliquity factor (`1/cosθ` for a beam at `θ` to the normal), and
//! `total_energy` includes that factor, so energy fractions and MTFs
//! normalised by it remain correct.

use mklight_core::{JobContext, MkError, MkResult, Vec3, math::Mat3};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::fft::{affine_mft,Dir, fft2, mft, next_fast_len};
use crate::pupil::PupilFunction;
use crate::warnings::{WaveWarning, push_unique};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsfNormalization {
    /// Diffraction-limited peak = 1 (peak value = Strehl ratio).
    #[default]
    Strehl,
    /// Total energy = 1 (values in mm⁻²).
    UnitEnergy,
}

/// How the image-plane spacing is chosen.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ImageSampling {
    /// `Δx = λF#/factor` (factor 2 is Nyquist).
    Padding { factor: f64 },
    /// A requested spacing in mm.
    Spacing { mm: f64 },
}

/// Transform used for a PSF.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PsfTransform {
    /// Zero-padded FFT. The spacing is rounded so that the transform length
    /// is 7-smooth (reported as `SpacingAdjusted` when a spacing was
    /// requested).
    #[default]
    Fft,
    /// Exact matrix Fourier transform on the requested grid (no rounding;
    /// `O(n²M + nM²)`).
    Matrix,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PsfSettings {
    /// Output samples per side `M`.
    pub grid_size: usize,
    pub sampling: ImageSampling,
    pub normalization: PsfNormalization,
    pub transform: PsfTransform,
}

impl Default for PsfSettings {
    fn default() -> Self {
        PsfSettings {
            grid_size: 128,
            sampling: ImageSampling::Padding { factor: 4.0 },
            normalization: PsfNormalization::Strehl,
            transform: PsfTransform::Fft,
        }
    }
}

/// A sampled intensity PSF (see the module docs for coordinates and
/// normalisation). Deserialisation checks that the grid is consistent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "PsfData")]
pub struct Psf {
    pub size: usize,
    pub spacing_mm: f64,
    pub center_mm: [f64; 2],
    pub data: Vec<f64>,
    pub normalization: PsfNormalization,
    /// `∫ PSF dA` over the whole (infinite) plane in these units.
    pub total_energy: f64,
    /// Peak of the aberration-free PSF in these units.
    pub diffraction_peak: f64,
    /// Wavelength (µm); `None` for polychromatic results.
    pub wavelength_um: Option<f64>,
    pub warnings: Vec<WaveWarning>,
}

#[derive(Deserialize)]
struct PsfData {
    size: usize,
    spacing_mm: f64,
    center_mm: [f64; 2],
    data: Vec<f64>,
    normalization: PsfNormalization,
    total_energy: f64,
    diffraction_peak: f64,
    wavelength_um: Option<f64>,
    #[serde(default)]
    warnings: Vec<WaveWarning>,
}

impl TryFrom<PsfData> for Psf {
    type Error = MkError;
    fn try_from(d: PsfData) -> MkResult<Self> {
        let ok = d.size > 0
            && d.data.len() == d.size * d.size
            && d.spacing_mm > 0.0
            && d.spacing_mm.is_finite()
            && d.center_mm.iter().all(|c| c.is_finite())
            && d.data.iter().all(|v| v.is_finite());
        if !ok {
            return Err(MkError::InvalidParam("inconsistent PSF grid".into()));
        }
        Ok(Psf {
            size: d.size,
            spacing_mm: d.spacing_mm,
            center_mm: d.center_mm,
            data: d.data,
            normalization: d.normalization,
            total_energy: d.total_energy,
            diffraction_peak: d.diffraction_peak,
            wavelength_um: d.wavelength_um,
            warnings: d.warnings,
        })
    }
}

impl Psf {
    /// Index of the reference sample along either axis.
    pub fn center_index(&self) -> usize {
        self.size / 2
    }
    pub fn x(&self, i: usize) -> f64 {
        self.center_mm[0] + (i as f64 - self.center_index() as f64) * self.spacing_mm
    }
    pub fn y(&self, j: usize) -> f64 {
        self.center_mm[1] + (j as f64 - self.center_index() as f64) * self.spacing_mm
    }
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.size + i]
    }
    /// `(value, i, j)` of the largest sample.
    pub fn peak(&self) -> (f64, usize, usize) {
        let mut best = (f64::NEG_INFINITY, 0, 0);
        for (k, &v) in self.data.iter().enumerate() {
            if v > best.0 {
                best = (v, k % self.size, k / self.size);
            }
        }
        best
    }
    /// Strehl ratio from the PSF peak (sample maximum).
    pub fn strehl(&self) -> f64 {
        self.peak().0 / self.diffraction_peak
    }
    /// `Σ PSF·Δx²` over the grid (compensated).
    pub fn energy_in_grid(&self) -> f64 {
        let s: mklight_core::math::KahanSum = self.data.iter().copied().collect();
        s.value() * self.spacing_mm * self.spacing_mm
    }
    /// Intensity-weighted centroid (mm).
    pub fn centroid(&self) -> [f64; 2] {
        let (mut sx, mut sy, mut s) = (0.0, 0.0, 0.0);
        for j in 0..self.size {
            for i in 0..self.size {
                let v = self.at(i, j);
                sx += v * self.x(i);
                sy += v * self.y(j);
                s += v;
            }
        }
        [sx / s, sy / s]
    }

    /// Bicubic (Keys, a = −½) interpolation at image point `(x, y)` mm;
    /// zero outside the grid. The error is `O(Δx³)`: about 0.3 % of the
    /// peak at `Δx = λF#/4` and below 0.1 % at `λF#/8`.
    pub fn interpolate(&self, x: f64, y: f64) -> f64 {
        let fx = (x - self.center_mm[0]) / self.spacing_mm + self.center_index() as f64;
        let fy = (y - self.center_mm[1]) / self.spacing_mm + self.center_index() as f64;
        let (ix, iy) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - ix, fy - iy);
        let (ix, iy) = (ix as i64, iy as i64);
        let wx = keys_weights(tx);
        let wy = keys_weights(ty);
        let m = self.size as i64;
        let mut s = 0.0;
        for (b, wyb) in wy.iter().enumerate() {
            let j = iy + b as i64 - 1;
            if j < 0 || j >= m {
                continue;
            }
            for (a, wxa) in wx.iter().enumerate() {
                let i = ix + a as i64 - 1;
                if i < 0 || i >= m {
                    continue;
                }
                s += wxa * wyb * self.data[(j * m + i) as usize];
            }
        }
        s
    }

    /// Resampled onto a new grid by bicubic interpolation (normalisation
    /// metadata unchanged).
    pub fn resampled(&self, spacing_mm: f64, size: usize, center_mm: [f64; 2]) -> Psf {
        let c = (size / 2) as f64;
        let data = (0..size * size)
            .into_par_iter()
            .map(|k| {
                let x = center_mm[0] + ((k % size) as f64 - c) * spacing_mm;
                let y = center_mm[1] + ((k / size) as f64 - c) * spacing_mm;
                self.interpolate(x, y)
            })
            .collect();
        Psf {
            size,
            spacing_mm,
            center_mm,
            data,
            ..self.clone()
        }
    }
}

fn keys_weights(t: f64) -> [f64; 4] {
    let a = -0.5;
    let k = |x: f64| {
        let x = x.abs();
        if x <= 1.0 {
            ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
        } else {
            0.0
        }
    };
    [k(1.0 + t), k(t), k(1.0 - t), k(2.0 - t)]
}

/// Raw (unnormalised) `|U|²` on an `size²` grid with spacing `dx`, centred
/// on the pupil's reference point, from an `m`-point zero-padded FFT.
fn fft_intensity(
    pupil: &PupilFunction,
    m: usize,
    size: usize,
    ctx: &JobContext,
) -> MkResult<Vec<f64>> {
    let n = pupil.n();
    let field = pupil.debye_field();
    let mut buf = vec![C64::default(); m * m];
    for j in 0..n {
        buf[j * m..j * m + n].copy_from_slice(&field[j * n..(j + 1) * n]);
    }
    fft2(&mut buf, m, Dir::Inverse, n, ctx)?;
    let c = (size / 2) as i64;
    let mi = m as i64;
    Ok((0..size * size)
        .map(|k| {
            let i = ((k % size) as i64 - c).rem_euclid(mi) as usize;
            let j = ((k / size) as i64 - c).rem_euclid(mi) as usize;
            buf[j * m + i].norm_sqr()
        })
        .collect())
}

/// A diagonal surface-to-normal map: the matrix transform is separable.
fn separable(b: &[[f64; 2]; 2]) -> bool {
    b[0][1].abs() < 1e-15 && b[1][0].abs() < 1e-15
}

/// Raw `|U|²` at image-surface points
/// `x_s = origin + ((k − c)·dx, (l − c)·dx)` relative to the pupil's
/// reference point, evaluated at their normal-plane feet `B·x_s` (module
/// docs). A diagonal `B` (on axis, or a chief ray in the x–z or y–z plane)
/// uses the separable matrix transform; otherwise the sum
/// `Σ_j r_y^j Σ_i r_x^i p_ij` (`r = e^{i2πΔν·ξ}`, a unit-modulus phase
/// factor dropped) is evaluated by Horner's rule along both pupil axes,
/// `O(n²M²)`.
fn surface_intensity(
    pupil: &PupilFunction,
    dx: f64,
    size: usize,
    origin: [f64; 2],
    ctx: &JobContext,
) -> MkResult<Vec<f64>> {
    let n = pupil.n();
    let s = pupil.frequency_scale();
    let nu: Vec<f64> = (0..n).map(|i| pupil.coord(i) * s).collect();
    let c = (size / 2) as f64;
    let xs: Vec<f64> = (0..size).map(|k| origin[0] + (k as f64 - c) * dx).collect();
    let ys: Vec<f64> = (0..size).map(|k| origin[1] + (k as f64 - c) * dx).collect();
    let b = pupil.geometry().surface_to_normal();
    let field = pupil.debye_field();
    if separable(&b) {
        let xn: Vec<f64> = xs.iter().map(|x| b[0][0] * x).collect();
        let yn: Vec<f64> = ys.iter().map(|y| b[1][1] * y).collect();
        let u = mft(&field, n, &nu, &nu, &xn, &yn, ctx)?;
        return Ok(u.iter().map(|z| z.norm_sqr()).collect());
    }
    // Bounding box of the non-zero samples keeps the Horner loops short.
    let (mut lo, mut hi) = ((n, n), (0, 0));
    for (k, z) in field.iter().enumerate() {
        if z.re != 0.0 || z.im != 0.0 {
            lo = (lo.0.min(k % n), lo.1.min(k / n));
            hi = (hi.0.max(k % n), hi.1.max(k / n));
        }
    }
    let tau = 2.0 * std::f64::consts::PI;
    let dnu = 2.0 * s / n as f64;
    let done = AtomicUsize::new(0);
    let rows: Vec<Vec<f64>> = (0..size)
        .into_par_iter()
        .map(|l| -> MkResult<Vec<f64>> {
            ctx.check()?;
            let row = xs
                .iter()
                .map(|&x| {
                    let xi = b[0][0] * x + b[0][1] * ys[l];
                    let eta = b[1][0] * x + b[1][1] * ys[l];
                    let rx = C64::from_polar(1.0, tau * dnu * xi);
                    let ry = C64::from_polar(1.0, tau * dnu * eta);
                    let mut acc = C64::default();
                    for j in (lo.1..=hi.1).rev() {
                        let frow = &field[j * n..(j + 1) * n];
                        let mut inner = C64::default();
                        for i in (lo.0..=hi.0).rev() {
                            inner = inner * rx + frow[i];
                        }
                        acc = acc * ry + inner;
                    }
                    acc.norm_sqr()
                })
                .collect();
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            ctx.report(d as f64 / size as f64, "");
            Ok(row)
        })
        .collect::<MkResult<_>>()?;
    Ok(rows.into_iter().flatten().collect())
}

/// `(Σ|p|)²` and `L²·E/cosθ_c`: the raw diffraction-limited peak and the
/// raw whole-plane integral of the PSF on the image surface
/// (`E` = [`PupilFunction::energy`]; `cosθ_c` the projection Jacobian).
fn raw_constants(pupil: &PupilFunction) -> (f64, f64) {
    let s1: f64 = pupil.field_weights().iter().sum();
    let l = pupil.alias_free_field_mm();
    let g = pupil.geometry();
    (s1 * s1, l * l * pupil.energy() / g.projection_jacobian())
}

fn normalized(
    mut raw: Vec<f64>,
    raw_peak: f64,
    raw_energy: f64,
    norm: PsfNormalization,
) -> (Vec<f64>, f64, f64) {
    let (div, total, peak) = match norm {
        PsfNormalization::Strehl => (raw_peak, raw_energy / raw_peak, 1.0),
        PsfNormalization::UnitEnergy => (raw_energy, 1.0, raw_peak / raw_energy),
    };
    raw.iter_mut().for_each(|v| *v /= div);
    (raw, total, peak)
}

fn image_checks(pupil: &PupilFunction, spacing: f64, size: usize, warnings: &mut Vec<WaveWarning>) {
    let lf = pupil.lambda_f_mm();
    if spacing > 0.5 * lf * (1.0 + 1e-9) {
        push_unique(
            warnings,
            WaveWarning::ImageUndersampled {
                spacing_mm: spacing,
                nyquist_mm: 0.5 * lf,
            },
        );
    }
    let l = pupil.alias_free_field_mm();
    if size as f64 * spacing > l * (1.0 + 1e-9) {
        push_unique(
            warnings,
            WaveWarning::FieldWraps {
                grid_mm: size as f64 * spacing,
                alias_free_mm: l,
            },
        );
    }
}

fn check_grid(size: usize) -> MkResult<()> {
    if size == 0 {
        return Err(MkError::InvalidParam(
            "PSF grid size must be positive".into(),
        ));
    }
    Ok(())
}

/// Largest FFT length [`fft_psf`] uses before switching to the matrix
/// transform.
pub const MAX_FFT_LEN: usize = 8192;

/// Monochromatic PSF on the image surface by Fourier transform of the
/// pupil (Debye integral).
///
/// With [`PsfTransform::Fft`] the pupil is zero-padded to the smallest
/// 7-smooth length `M ≥ n·λF#/Δx` (at least `n`); the transform is cropped
/// (or, if the grid exceeds the alias-free field, periodically extended,
/// with a warning) to `grid_size²` samples centred on the reference point.
/// The exact matrix transform is used instead when `M` would be below `n`
/// (spacing coarser than `λF#`) or above `MAX_FFT_LEN`, and for an
/// off-axis chief ray (the projection makes the x and y spacings differ on
/// the normal plane).
pub fn fft_psf(pupil: &PupilFunction, settings: &PsfSettings, ctx: &JobContext) -> MkResult<Psf> {
    check_grid(settings.grid_size)?;
    ctx.report(0.0, "PSF");
    let n = pupil.n();
    let lf = pupil.lambda_f_mm();
    let l = pupil.alias_free_field_mm();
    let mut warnings = pupil.sampling_warnings();
    let requested = match settings.sampling {
        ImageSampling::Padding { factor } => lf / factor,
        ImageSampling::Spacing { mm } => mm,
    };
    if !(requested > 0.0 && requested.is_finite()) {
        return Err(MkError::InvalidParam(format!("PSF spacing {requested} mm")));
    }
    let size = settings.grid_size;
    // The FFT needs M ≥ n; very fine spacings would need huge transforms,
    // for which the matrix transform (cost set by the output grid) is used.
    // M is rounded up, never down, so the spacing never becomes coarser
    // than requested (in particular never coarser than a Nyquist request).
    let ratio = l / requested;
    let m = (ratio >= n as f64 * (1.0 - 1e-12) && ratio <= MAX_FFT_LEN as f64)
        .then(|| next_fast_len(((ratio * (1.0 - 1e-12)).ceil() as usize).max(n)));
    let use_fft = settings.transform == PsfTransform::Fft
        && pupil.geometry().is_on_axis()
        && m.is_some_and(|m| m <= MAX_FFT_LEN);
    let (spacing, raw) = match m {
        Some(m) if use_fft => {
            let spacing = l / m as f64;
            if (spacing / requested - 1.0).abs() > 1e-6 {
                warnings.push(WaveWarning::SpacingAdjusted {
                    requested_mm: requested,
                    actual_mm: spacing,
                });
            }
            (spacing, fft_intensity(pupil, m, size, ctx)?)
        }
        _ => (
            requested,
            surface_intensity(pupil, requested, size, [0.0; 2], ctx)?,
        ),
    };
    image_checks(pupil, spacing, size, &mut warnings);
    let (rp, re) = raw_constants(pupil);
    let (data, total_energy, diffraction_peak) = normalized(raw, rp, re, settings.normalization);
    ctx.report(1.0, "PSF");
    Ok(Psf {
        size,
        spacing_mm: spacing,
        center_mm: pupil.image_offset_mm(),
        data,
        normalization: settings.normalization,
        total_energy,
        diffraction_peak,
        wavelength_um: Some(pupil.wavelength_um()),
        warnings,
    })
}

/// An observation plane in image space, relative to the pupil's reference
/// point (the centre of the reference sphere), in the chief-ray frame
/// `(ê_u, ê_v, ê_w)` of [`crate::pupil`] (+z along the chief ray, away from
/// the exit pupil; on axis this is the image frame). The axes must be
/// orthonormal.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImagePlane {
    pub origin_mm: Vec3,
    /// Unit vector of the grid's i (x) direction.
    pub x_axis: Vec3,
    /// Unit vector of the grid's j (y) direction.
    pub y_axis: Vec3,
}

impl Default for ImagePlane {
    fn default() -> Self {
        ImagePlane {
            origin_mm: Vec3::ZERO,
            x_axis: Vec3::X,
            y_axis: Vec3::Y,
        }
    }
}

impl ImagePlane {
    /// Plane normal to the axis at `z = dz`.
    pub fn defocused(dz_mm: f64) -> Self {
        ImagePlane {
            origin_mm: Vec3::new(0.0, 0.0, dz_mm),
            ..Default::default()
        }
    }
    /// Plane through `(0, 0, dz)` rotated by `tilt_x` then `tilt_y`
    /// (degrees, right-hand rule, CONVENTIONS §2).
    pub fn tilted(tilt_x_deg: f64, tilt_y_deg: f64, dz_mm: f64) -> Self {
        let r = Mat3::rot_x(tilt_x_deg.to_radians()).mul_mat(&Mat3::rot_y(tilt_y_deg.to_radians()));
        ImagePlane {
            origin_mm: Vec3::new(0.0, 0.0, dz_mm),
            x_axis: r.mul_vec(Vec3::X),
            y_axis: r.mul_vec(Vec3::Y),
        }
    }

    /// The image surface (tangent plane at the chief-ray point) of a pupil
    /// with this geometry: its x and y axes expressed in the chief-ray
    /// frame. Huygens on this plane is the exact counterpart of the
    /// projected [`fft_psf`] grid.
    pub fn image_surface(geometry: &crate::pupil::PupilGeometry) -> Self {
        let [eu, ev, ew] = geometry.chief_frame();
        ImagePlane {
            origin_mm: Vec3::ZERO,
            x_axis: Vec3::new(eu.x, ev.x, ew.x),
            y_axis: Vec3::new(eu.y, ev.y, ew.y),
        }
    }

    /// Unit normal `x_axis × y_axis`.
    pub fn normal(&self) -> Vec3 {
        self.x_axis.cross(self.y_axis)
    }

    pub fn validate(&self) -> MkResult<()> {
        let unit = |v: Vec3| v.is_finite() && (v.norm() - 1.0).abs() < 1e-9;
        if unit(self.x_axis)
            && unit(self.y_axis)
            && self.x_axis.dot(self.y_axis).abs() < 1e-9
            && self.origin_mm.is_finite()
        {
            Ok(())
        } else {
            Err(MkError::InvalidParam(format!(
                "image plane axes must be orthonormal: {self:?}"
            )))
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct HuygensSettings {
    pub grid_size: usize,
    pub spacing_mm: f64,
    pub plane: ImagePlane,
    pub normalization: PsfNormalization,
}

/// PSF by direct Huygens–Fresnel (Rayleigh–Sommerfeld) integration from
/// the reference sphere to each image point:
/// `U(x) ∝ Σ_j P_j · e^{−ik'(r_j − R)} · cosχ_j / r_j · dS_j`,
/// with pupil points `p_j = R·(α, β, −γ)` on the sphere, `dS = R²·dα·dβ/γ`,
/// `k' = 2πn'/λ`, and `χ` the angle between the inward sphere normal and
/// `x − p_j`. (The kernel is the complex conjugate of the `e^{−iωt}`
/// propagator, matching the FFT phase convention; intensities are
/// unaffected.) `r − R` is formed as `(|x|² − 2p·x)/(r + R)` to avoid
/// cancellation. No paraxial or Debye approximation, so it validates the
/// FFT PSF and handles tilted and defocused planes. Cost `O(n²·M²)`,
/// parallel over image rows. For a telecentric beam (`R = ∞`) the sum is
/// its limit, a superposition of plane waves
/// `U(x) ∝ Σ_j (c·A/γ)_j·e^{i2πW_j}·e^{−ik'·k̂_j·x}`, still exact on tilted
/// and defocused planes.
///
/// On a plane with normal `N̂` the plane integral of `|U|²` is
/// `Σ_j E_j·γ_j/|k̂_j·N̂| / Σ_j E_j` times the flux (`k̂_j = (−α, −β, γ)` the
/// direction of pupil sample `j`, `E_j` its energy): a plane wave crossing
/// a plane at angle `θ` covers `1/cosθ` more area. `total_energy` (and
/// the unit-energy scaling) include this factor.
pub fn huygens_psf(
    pupil: &PupilFunction,
    settings: &HuygensSettings,
    ctx: &JobContext,
) -> MkResult<Psf> {
    check_grid(settings.grid_size)?;
    if !(settings.spacing_mm > 0.0 && settings.spacing_mm.is_finite()) {
        return Err(MkError::InvalidParam(format!(
            "PSF spacing {} mm",
            settings.spacing_mm
        )));
    }
    settings.plane.validate()?;
    let normal = settings.plane.normal();
    let n = pupil.n();
    let g = pupil.geometry();
    let telecentric = g.is_telecentric();
    let r = g.reference_radius_mm;
    let sm = g.sin_max();
    let k = 2.0 * std::f64::consts::PI * g.image_index / (pupil.wavelength_um() * 1e-3);
    let d_alpha = sm * pupil.spacing();
    let tau = 2.0 * std::f64::consts::PI;
    // (point on the sphere, P·dS) for covered samples.
    let mut pts: Vec<(Vec3, C64)> = Vec::new();
    let mut u_dl0 = 0.0;
    // Σ E_j·γ_j/|k̂_j·N̂| and Σ E_j for the plane's obliquity factor.
    let (mut e_plane, mut e_sum) = (0.0, 0.0);
    for j in 0..n {
        for i in 0..n {
            let q = j * n + i;
            let c = pupil.coverage()[q];
            if c <= 0.0 {
                continue;
            }
            let (a, b) = (sm * pupil.coord(i), sm * pupil.coord(j));
            let s2 = a * a + b * b;
            let gamma = ((1.0 - s2).max(0.0)).sqrt();
            // Telecentric beams: the R → ∞ limit, scaled by 1/R (which
            // cancels in the normalisation), leaves the plane-wave weight
            // c·A·dα²/γ.
            let ds = if telecentric {
                d_alpha * d_alpha / gamma
            } else {
                r * r * d_alpha * d_alpha / gamma
            };
            let amp = c * pupil.amplitude()[q] * ds;
            u_dl0 += if telecentric { amp } else { amp / r };
            let cos_in = Vec3::new(-a, -b, gamma).dot(normal).abs();
            if !(cos_in > 1e-9) {
                return Err(MkError::InvalidParam(
                    "image plane is parallel to rays of the beam".into(),
                ));
            }
            let e = c * (pupil.amplitude()[q] / gamma).powi(2);
            e_plane += e * gamma / cos_in;
            e_sum += e;
            // Telecentric beams store the unit sphere point −k̂ instead.
            let rr = if telecentric { 1.0 } else { r };
            pts.push((
                Vec3::new(rr * a, rr * b, -rr * gamma),
                C64::from_polar(amp, tau * pupil.opd_waves()[q]),
            ));
        }
    }
    let size = settings.grid_size;
    let cidx = (size / 2) as f64;
    let pl = settings.plane;
    let done = AtomicUsize::new(0);
    ctx.report(0.0, "Huygens PSF");
    let rows: Vec<Vec<f64>> = (0..size)
        .into_par_iter()
        .map(|jy| -> MkResult<Vec<f64>> {
            ctx.check()?;
            let row = (0..size)
                .map(|ix| {
                    let x = pl.origin_mm
                        + pl.x_axis * ((ix as f64 - cidx) * settings.spacing_mm)
                        + pl.y_axis * ((jy as f64 - cidx) * settings.spacing_mm);
                    let x2 = x.norm2();
                    let (mut re, mut im) = (0.0, 0.0);
                    if telecentric {
                        // Plane waves `e^{−ik'·k̂·x}` with `k̂ = −p`.
                        for (p, w) in &pts {
                            let (s, c) = (k * p.dot(x)).sin_cos();
                            re += w.re * c - w.im * s;
                            im += w.re * s + w.im * c;
                        }
                        return (re * re + im * im) / (u_dl0 * u_dl0);
                    }
                    for (p, w) in &pts {
                        let px = p.dot(x);
                        let dist = (x - *p).norm2().sqrt();
                        let dr = (x2 - 2.0 * px) / (dist + r);
                        let cos_chi = (r * r - px) / (r * dist);
                        let f = cos_chi / dist;
                        let (s, c) = (-k * dr).sin_cos();
                        // w · f · e^{iφ}
                        re += f * (w.re * c - w.im * s);
                        im += f * (w.re * s + w.im * c);
                    }
                    (re * re + im * im) / (u_dl0 * u_dl0)
                })
                .collect();
            let d = done.fetch_add(1, Ordering::Relaxed) + 1;
            ctx.report(d as f64 / size as f64, "");
            Ok(row)
        })
        .collect::<MkResult<_>>()?;
    let strehl_data: Vec<f64> = rows.into_iter().flatten().collect();
    // Unit-energy peak on this plane: the normal-plane value over the
    // obliquity factor.
    let dl_unit = pupil.diffraction_peak_unit_energy() * e_sum / e_plane;
    let (data, total_energy, diffraction_peak) = match settings.normalization {
        PsfNormalization::Strehl => (strehl_data, 1.0 / dl_unit, 1.0),
        PsfNormalization::UnitEnergy => (
            strehl_data.iter().map(|v| v * dl_unit).collect(),
            1.0,
            dl_unit,
        ),
    };
    let mut warnings = pupil.sampling_warnings();
    image_checks(pupil, settings.spacing_mm, size, &mut warnings);
    let off = pupil.image_offset_mm();
    Ok(Psf {
        size,
        spacing_mm: settings.spacing_mm,
        center_mm: [off[0] + pl.origin_mm.x, off[1] + pl.origin_mm.y],
        data,
        normalization: settings.normalization,
        total_energy,
        diffraction_peak,
        wavelength_um: Some(pupil.wavelength_um()),
        warnings,
    })
}

/// Settings for a polychromatic PSF on a common grid centred on the common
/// image reference point (each pupil's `image_offset_mm` places its own
/// PSF).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolyPsfSettings {
    pub grid_size: usize,
    pub spacing_mm: f64,
    pub normalization: PsfNormalization,
    /// `Matrix` (the default from [`PolyPsfSettings::new`]) evaluates every
    /// wavelength exactly on the common grid (band-limited resampling).
    /// `Fft` keeps that separable transform where `B` is diagonal (on axis,
    /// or a chief ray in the x–z or y–z plane), which is also the cheaper
    /// one there. A skew pupil, whose exact transform costs `n²M²`, is
    /// transformed by FFT on its chief-normal plane at a native spacing
    /// `≤ min(Δx, λF#/8)` and projected bicubically onto the image surface,
    /// with errors below about 10⁻³ of the peak.
    pub transform: PsfTransform,
}

impl PolyPsfSettings {
    /// Exact (matrix-transform) settings, Strehl-normalised.
    pub fn new(grid_size: usize, spacing_mm: f64) -> Self {
        PolyPsfSettings {
            grid_size,
            spacing_mm,
            normalization: PsfNormalization::Strehl,
            transform: PsfTransform::Matrix,
        }
    }
}

/// Polychromatic PSF `Σ ŵ_λ · PSF_λ` with each `PSF_λ` normalised to unit
/// energy and `ŵ = w/Σw` (spectral weights are relative energies). With
/// `Strehl` normalisation the result is divided by `Σ ŵ_λ·peak_DL,λ`, the
/// polychromatic diffraction-limited peak.
pub fn polychromatic_psf(
    pupils: &[(&PupilFunction, f64)],
    settings: &PolyPsfSettings,
    ctx: &JobContext,
) -> MkResult<Psf> {
    check_grid(settings.grid_size)?;
    let wsum: f64 = pupils.iter().map(|(_, w)| *w).sum();
    if pupils.is_empty() || !(wsum > 0.0) || pupils.iter().any(|(_, w)| *w < 0.0) {
        return Err(MkError::InvalidParam(
            "polychromatic PSF needs positive weights".into(),
        ));
    }
    let size = settings.grid_size;
    let dx = settings.spacing_mm;
    if !(dx > 0.0 && dx.is_finite()) {
        return Err(MkError::InvalidParam(format!("PSF spacing {dx} mm")));
    }
    let mut acc = vec![0.0; size * size];
    let mut dl_peak = 0.0;
    let mut warnings = Vec::new();
    for (q, (pupil, w)) in pupils.iter().enumerate() {
        let sub = ctx.subrange(
            q as f64 / pupils.len() as f64,
            (q + 1) as f64 / pupils.len() as f64,
        );
        let wn = w / wsum;
        let off = pupil.image_offset_mm();
        let (_, raw_energy) = raw_constants(pupil);
        let b = pupil.geometry().surface_to_normal();
        let raw = if settings.transform == PsfTransform::Matrix || separable(&b) {
            surface_intensity(pupil,dx,size,[-off[0],-off[1]],&sub)?
        } else {
            let n=pupil.n();
            let nu: Vec<_>=(0..n).map(|i| pupil.coord(i)*pupil.frequency_scale()).collect();
            let xs: Vec<_>=(0..size).map(|i| (i as f64-(size/2) as f64)*dx-off[0]).collect();
            let ys: Vec<_>=(0..size).map(|i| (i as f64-(size/2) as f64)*dx-off[1]).collect();
            affine_mft(&pupil.debye_field(),n,&nu,&xs,&ys,b,&sub)?.into_iter().map(|z| z.norm_sqr()).collect()
        };
        let unit: Vec<_> = raw.into_iter().map(|v| v/raw_energy).collect();
        for (a, v) in acc.iter_mut().zip(&unit) {
            *a += wn * v;
        }
        // Unit-energy peak on the image surface (the projection spreads the
        // energy over 1/cosθ_c more area).
        dl_peak +=
            wn * pupil.diffraction_peak_unit_energy() * pupil.geometry().projection_jacobian();
        for wv in pupil.sampling_warnings() {
            push_unique(&mut warnings, wv);
        }
        image_checks(pupil, dx, size, &mut warnings);
    }
    let (data, total_energy, diffraction_peak) = match settings.normalization {
        PsfNormalization::UnitEnergy => (acc, 1.0, dl_peak),
        PsfNormalization::Strehl => (
            acc.iter().map(|v| v / dl_peak).collect(),
            1.0 / dl_peak,
            1.0,
        ),
    };
    Ok(Psf {
        size,
        spacing_mm: dx,
        center_mm: [0.0, 0.0],
        data,
        normalization: settings.normalization,
        total_energy,
        diffraction_peak,
        wavelength_um: None,
        warnings,
    })
}
