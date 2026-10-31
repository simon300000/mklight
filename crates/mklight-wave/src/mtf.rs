//! Optical and modulation transfer functions.
//!
//! The OTF is the normalised Fourier transform of the intensity PSF,
//! `OTF(ν) = ∫ PSF(x)·e^{−2πiν·x} dA / ∫ PSF dA`, equivalently the
//! normalised autocorrelation of the Debye-weighted pupil field,
//! `OTF(ν) = Σ_u p(u + s)·p*(u) / Σ|p|²` with the pupil shift
//! `s = 2ν·λF#` (normalised units; `s = 2` at the cutoff `ν_c = 1/(λF#)`).
//! MTF = |OTF|, 1 at zero frequency. Frequencies are in cycles/mm on the
//! image surface. Off axis the OTF is the autocorrelation of the angular
//! spectrum over the image-surface direction cosines (`SensorMap`), which
//! keeps the longitudinal phase across the PSF; the tangential cutoff on the
//! surface is `cosθ_c/(λF#)`.
//!
//! **Orientation** (CONVENTIONS §4): *tangential* frequencies run along the
//! meridional direction of the field point (image +y on axis or for a field
//! on +y), *sagittal* along the perpendicular direction (image x). For a
//! field at `(Hx, Hy)` the tangential unit vector is `(Hx, Hy)/|H|`.
//!
//! Three evaluation routes:
//! * [`otf_fft`] / [`mtf_fft`]: the pupil is zero-padded to `≥ 2n`, Fourier
//!   transformed to the PSF, squared, and transformed back. This yields the
//!   exact discrete autocorrelation on the frequency lattice
//!   `Δν = 1/(n·λF#)` (normal plane; off axis the angular spectrum is first
//!   resampled, see `sensor_spectrum`); other frequencies are interpolated
//!   (see [`Otf::at`]).
//! * [`mtf_direct`]: the autocorrelation evaluated directly for each
//!   requested frequency, shifting the pupil by `s` with bilinear
//!   interpolation of amplitude and OPD (no frequency lattice). An
//!   independent check of the FFT route between lattice points.
//! * [`mtf_from_psf`]: a direct DFT of a sampled PSF (any PSF: Huygens,
//!   polychromatic, tilted planes).

use mklight_core::{JobContext, MkError, MkResult, Vec3};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::fft::{Dir, fft2, next_fast_len};
use crate::psf::Psf;
use crate::pupil::PupilFunction;
use crate::warnings::{WaveWarning, push_unique};

/// Tangential/sagittal directions in the image plane.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MtfOrientation {
    /// Unit vector of the tangential direction `(x, y)`.
    pub tangential: [f64; 2],
}

impl Default for MtfOrientation {
    fn default() -> Self {
        MtfOrientation {
            tangential: [0.0, 1.0],
        }
    }
}

impl MtfOrientation {
    /// Orientation for normalised field `(hx, hy)`; on axis the tangential
    /// direction is +y.
    pub fn from_field(hx: f64, hy: f64) -> Self {
        let r = hx.hypot(hy);
        if r < 1e-12 {
            Self::default()
        } else {
            MtfOrientation {
                tangential: [hx / r, hy / r],
            }
        }
    }
    /// Sagittal unit vector: the tangential one rotated by −90° (`+x` for a
    /// `+y` tangential direction).
    pub fn sagittal(&self) -> [f64; 2] {
        [self.tangential[1], -self.tangential[0]]
    }
}

/// MTF slices at a list of frequencies.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MtfCurve {
    /// Cycles/mm.
    pub frequencies: Vec<f64>,
    pub tangential: Vec<f64>,
    pub sagittal: Vec<f64>,
    /// MTF of the same aperture and amplitude without aberrations (the
    /// diffraction limit), along the tangential direction. Empty when not
    /// known ([`mtf_from_psf`]).
    pub diffraction_limit: Vec<f64>,
    /// Incoherent cutoff `1/(λF#)` (largest over wavelengths) in the plane
    /// normal to the chief ray, cycles/mm (the tangential cutoff on the
    /// image surface is `cosθ_c` times this); NaN when not known
    /// ([`mtf_from_psf`]).
    pub cutoff: f64,
    pub warnings: Vec<WaveWarning>,
}

/// Analytic diffraction-limited MTF of a clear circular (`obscuration = 0`)
/// or annular pupil with uniform amplitude at normalised frequency
/// `nu = ν/ν_c`. Computed as the overlap area of the shifted annuli
/// (shift `2·nu` in units of the outer radius) over the annulus area; for
/// the clear disk this is `(2/π)(φ − cosφ·sinφ)`, `φ = acos(nu)`.
pub fn diffraction_limited_mtf(nu: f64, obscuration: f64) -> f64 {
    let nu = nu.abs();
    if nu >= 1.0 {
        return 0.0;
    }
    let e = obscuration.clamp(0.0, 1.0);
    let d = 2.0 * nu;
    let a = circle_overlap(1.0, 1.0, d) - 2.0 * circle_overlap(1.0, e, d) + circle_overlap(e, e, d);
    let area = std::f64::consts::PI * (1.0 - e * e);
    (a / area).max(0.0)
}

/// Area of intersection of disks of radii `r1`, `r2` with centres `d` apart.
pub fn circle_overlap(r1: f64, r2: f64, d: f64) -> f64 {
    use std::f64::consts::PI;
    if r1 <= 0.0 || r2 <= 0.0 || d >= r1 + r2 {
        return 0.0;
    }
    if d <= (r1 - r2).abs() {
        let r = r1.min(r2);
        return PI * r * r;
    }
    let c1 = ((d * d + r1 * r1 - r2 * r2) / (2.0 * d * r1)).clamp(-1.0, 1.0);
    let c2 = ((d * d + r2 * r2 - r1 * r1) / (2.0 * d * r2)).clamp(-1.0, 1.0);
    let k = ((-d + r1 + r2) * (d + r1 - r2) * (d - r1 + r2) * (d + r1 + r2)).max(0.0);
    r1 * r1 * c1.acos() + r2 * r2 * c2.acos() - 0.5 * k.sqrt()
}

/// Full 2-D OTF on the frequency lattice of the pupil sampling, with
/// interpolation to arbitrary image-surface frequencies.
///
/// # Interpolation between lattice points
///
/// The lattice spacing `Δν = 1/L` is set by the alias-free field `L`, so a
/// PSF of extent `w` makes the OTF vary on the scale `1/w` and plain
/// interpolation fails once `w` is a sizeable part of `L`. Two parts of
/// that variation are removed before interpolating:
///
/// 1. A displaced PSF (tilt, coma flare) gives the OTF a linear phase
///    `e^{−2πiν·x̄}`. The centroid `x̄` is estimated from the first lattice
///    harmonic, `x̄ = −arg OTF(Δν, 0)/(2πΔν)` (exact for a pure tilt), and
///    the ramp is divided out and restored after interpolation.
/// 2. The OTF has a cone-shaped kink at `ν = 0` and a `(Δ)^{3/2}` edge at
///    the support boundary that come from the aperture edge, not from the
///    aberrations. The lattice OTF `D` of the same pupil without
///    aberrations has the same kink (the `|ν|` term depends only on `|p|`
///    along the rim), so the difference `OTF·ramp − D` is smooth there and is
///    interpolated with Keys bicubic weights, while `D` itself is
///    interpolated bilinearly (exact at the kink, `O(Δν²)` elsewhere).
///
/// For a pupil without aberrations this is bilinear interpolation; on
/// lattice points it returns the lattice values exactly.
#[derive(Clone, Debug, PartialEq)]
pub struct Otf {
    /// Grid side `M ≥ 2n`; bin `(i, j)` is the normal-plane frequency
    /// `((i − M/2)·Δν, (j − M/2)·Δν)`.
    pub size: usize,
    /// `Δν = 1/(n·λF#)` cycles/mm.
    pub spacing: f64,
    /// Normalised OTF (`1` at zero frequency), row-major, centred, without
    /// the image-offset phase.
    pub data: Vec<C64>,
    /// Lattice OTF of the same pupil without aberrations; `None` when the
    /// pupil has none (it then equals `data`).
    pub reference: Option<Vec<C64>>,
    /// PSF centroid estimate (normal plane, mm, relative to the reference
    /// point) whose linear phase is removed before interpolation.
    pub centroid_mm: [f64; 2],
    /// Image offset of the PSF reference point (mm); applied as the phase
    /// `e^{−2πiν·offset}` by [`Otf::at`].
    pub offset_mm: [f64; 2],
    /// `B^{−T}`: image-surface frequency → normal-plane frequency.
    pub surface_to_normal: [[f64; 2]; 2],
    /// Cutoff `1/(λF#)` (normal plane).
    pub cutoff: f64,
}

/// Keys cubic convolution weights (a = −½) for the four samples around a
/// point at fraction `t` past the second one.
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

impl Otf {
    fn lattice(grid: &[C64], m: usize, i: i64, j: i64) -> C64 {
        let mi = m as i64;
        if i < 0 || j < 0 || i >= mi || j >= mi {
            C64::default()
        } else {
            grid[(j * mi + i) as usize]
        }
    }

    fn bilinear(grid: &[C64], m: usize, fx: f64, fy: f64) -> C64 {
        let (ix, iy) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - ix, fy - iy);
        let (i, j) = (ix as i64, iy as i64);
        Self::lattice(grid, m, i, j) * ((1.0 - tx) * (1.0 - ty))
            + Self::lattice(grid, m, i + 1, j) * (tx * (1.0 - ty))
            + Self::lattice(grid, m, i, j + 1) * ((1.0 - tx) * ty)
            + Self::lattice(grid, m, i + 1, j + 1) * (tx * ty)
    }

    /// Lattice coordinates of a normal-plane frequency.
    fn index_of(&self, nx: f64, ny: f64) -> (f64, f64) {
        let c = (self.size / 2) as f64;
        (nx / self.spacing + c, ny / self.spacing + c)
    }

    /// `e^{+2πiν·x̄}` at lattice bin `(i, j)`.
    fn ramp(&self, i: i64, j: i64) -> C64 {
        let c = (self.size / 2) as i64;
        let ph = 2.0
            * std::f64::consts::PI
            * self.spacing
            * ((i - c) as f64 * self.centroid_mm[0] + (j - c) as f64 * self.centroid_mm[1]);
        C64::from_polar(1.0, ph)
    }

    /// OTF at a normal-plane frequency, without the image-offset phase (see
    /// the type docs for the interpolation); zero beyond the grid.
    pub fn at_normal(&self, nx: f64, ny: f64) -> C64 {
        let (fx, fy) = self.index_of(nx, ny);
        let back = C64::from_polar(
            1.0,
            -2.0 * std::f64::consts::PI * (nx * self.centroid_mm[0] + ny * self.centroid_mm[1]),
        );
        let m = self.size;
        let (ix, iy) = (fx.floor(), fy.floor());
        match &self.reference {
            None => {
                let (tx, ty) = (fx - ix, fy - iy);
                let (i, j) = (ix as i64, iy as i64);
                let v = Self::lattice(&self.data, m, i, j)
                    * self.ramp(i, j)
                    * ((1.0 - tx) * (1.0 - ty))
                    + Self::lattice(&self.data, m, i + 1, j)
                        * self.ramp(i + 1, j)
                        * (tx * (1.0 - ty))
                    + Self::lattice(&self.data, m, i, j + 1)
                        * self.ramp(i, j + 1)
                        * ((1.0 - tx) * ty)
                    + Self::lattice(&self.data, m, i + 1, j + 1)
                        * self.ramp(i + 1, j + 1)
                        * (tx * ty);
                v * back
            }
            Some(dl) => {
                let wx = keys_weights(fx - ix);
                let wy = keys_weights(fy - iy);
                let mut d = C64::default();
                for (b, wyb) in wy.iter().enumerate() {
                    let j = iy as i64 + b as i64 - 1;
                    for (a, wxa) in wx.iter().enumerate() {
                        let i = ix as i64 + a as i64 - 1;
                        let v = Self::lattice(&self.data, m, i, j) * self.ramp(i, j)
                            - Self::lattice(dl, m, i, j);
                        d += v * (wxa * wyb);
                    }
                }
                (d + Self::bilinear(dl, m, fx, fy)) * back
            }
        }
    }

    /// OTF at an image-surface frequency `(νx, νy)` (cycles/mm), including
    /// the image-offset phase.
    pub fn at(&self, nx: f64, ny: f64) -> C64 {
        let t = self.surface_to_normal;
        let v = self.at_normal(t[0][0] * nx + t[0][1] * ny, t[1][0] * nx + t[1][1] * ny);
        let ph = -2.0 * std::f64::consts::PI * (nx * self.offset_mm[0] + ny * self.offset_mm[1]);
        v * C64::from_polar(1.0, ph)
    }

    /// OTF of the same pupil without aberrations or image offset (the
    /// diffraction limit) at an image-surface frequency, interpolated
    /// bilinearly.
    pub fn reference_at(&self, nx: f64, ny: f64) -> C64 {
        let t = self.surface_to_normal;
        let (fx, fy) = self.index_of(t[0][0] * nx + t[0][1] * ny, t[1][0] * nx + t[1][1] * ny);
        Self::bilinear(
            self.reference.as_ref().unwrap_or(&self.data),
            self.size,
            fx,
            fy,
        )
    }
}

/// Normalised lattice OTF of `pupil` (`M = next_fast_len(2n)`, centred).
fn lattice_otf(pupil: &PupilFunction, ctx: &JobContext) -> MkResult<(usize, Vec<C64>)> {
    let n = pupil.n();
    let m = next_fast_len(2 * n);
    let (field, energy) = if pupil.geometry().is_on_axis() {
        (pupil.debye_field(), pupil.energy())
    } else {
        sensor_spectrum(pupil)?
    };
    let mut buf = vec![C64::default(); m * m];
    for j in 0..n {
        buf[j * m..j * m + n].copy_from_slice(&field[j * n..(j + 1) * n]);
    }
    fft2(&mut buf, m, Dir::Inverse, n, &ctx.subrange(0.0, 0.5))?;
    buf.iter_mut()
        .for_each(|z| *z = C64::new(z.norm_sqr(), 0.0));
    fft2(&mut buf, m, Dir::Forward, m, &ctx.subrange(0.5, 1.0))?;
    // Σ_k I_k = M²·Σ|p|²; normalise by the area-weighted energy instead
    // (see `PupilFunction::energy`) and set the zero lag to exactly 1.
    let o0 = (m * m) as f64 * energy;
    if !(o0 > 0.0) {
        return Err(MkError::Numerical("pupil transmits no light".into()));
    }
    let c = (m / 2) as i64;
    let mi = m as i64;
    let mut data: Vec<C64> = (0..m * m)
        .map(|k| {
            let i = ((k % m) as i64 - c).rem_euclid(mi) as usize;
            let j = ((k / m) as i64 - c).rem_euclid(mi) as usize;
            buf[j * m + i] / o0
        })
        .collect();
    data[(m / 2) * m + m / 2] = C64::new(1.0, 0.0);
    Ok((m, data))
}

/// The angular spectrum (`SensorMap`) of an off-axis pupil, sampled at the
/// directions with transverse components `α·ê_u + β·ê_v − γ_rim·ê_w`, a
/// linearisation of the lattice through the aperture rim. Its lattice
/// transform is the PSF at normal-plane coordinates `B·x` as on axis; only
/// the remainder `(γ_rim − γ)·ê_w` (a fraction of a cell for moderate
/// apertures, zero on the rim so the hard edge is not blurred) is
/// interpolated. Returns the samples and their hard-edged energy (see
/// [`PupilFunction::energy`]).
fn sensor_spectrum(pupil: &PupilFunction) -> MkResult<(Vec<C64>, f64)> {
    let n = pupil.n();
    let map = SensorMap::new(pupil);
    map.covered(pupil)?;
    let [eu, ev, ew] = map.frame;
    let g = (1.0 - map.sin_max * map.sin_max).sqrt();
    let tau = 2.0 * std::f64::consts::PI;
    let mut field = vec![C64::default(); n * n];
    let mut energy = mklight_core::math::KahanSum::default();
    for (k, z) in field.iter_mut().enumerate() {
        let (a, b) = (
            map.sin_max * pupil.coord(k % n),
            map.sin_max * pupil.coord(k / n),
        );
        let (x, y) = (
            a * eu.x + b * ev.x - g * ew.x,
            a * eu.y + b * ev.y - g * ew.y,
        );
        let Some((ix, iy, qz)) = map.lattice(x, y) else {
            continue;
        };
        let (s, c, w) = map.sample(ix, iy);
        if c > 0.0 {
            // `s = c·A`: the hard edge counts `c·A²`.
            energy.add(s * s / (c * qz * qz));
            *z = C64::from_polar(s / qz, tau * w);
        }
    }
    Ok((field, energy.value()))
}

/// 2-D OTF by FFT (pupil → PSF → OTF), exact on its frequency lattice. For
/// an aberrated pupil the lattice OTF without aberrations is computed too
/// (for the interpolation, see [`Otf`]).
pub fn otf_fft(pupil: &PupilFunction, ctx: &JobContext) -> MkResult<Otf> {
    let aberrated = pupil.opd_waves().iter().any(|&w| w != 0.0);
    let (m, data) = lattice_otf(pupil, &ctx.subrange(0.0, if aberrated { 0.5 } else { 1.0 }))?;
    let reference = if aberrated {
        Some(lattice_otf(&pupil.without_aberrations(), &ctx.subrange(0.5, 1.0))?.1)
    } else {
        None
    };
    let spacing = 1.0 / pupil.alias_free_field_mm();
    let c = m / 2;
    let centroid = |z: C64| {
        if z.norm() > 1e-9 {
            -z.arg() / (2.0 * std::f64::consts::PI * spacing)
        } else {
            0.0
        }
    };
    Ok(Otf {
        size: m,
        spacing,
        centroid_mm: [
            centroid(data[c * m + c + 1]),
            centroid(data[(c + 1) * m + c]),
        ],
        data,
        reference,
        offset_mm: pupil.image_offset_mm(),
        surface_to_normal: pupil.geometry().surface_to_normal_frequency(),
        cutoff: pupil.geometry().cutoff_frequency(pupil.wavelength_um()),
    })
}

/// How an OTF is evaluated.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtfMethod {
    #[default]
    Fft,
    Direct,
}

fn freq_vectors(freqs: &[f64], o: &MtfOrientation) -> Vec<[f64; 2]> {
    let t = o.tangential;
    let s = o.sagittal();
    freqs
        .iter()
        .map(|&f| [f * t[0], f * t[1]])
        .chain(freqs.iter().map(|&f| [f * s[0], f * s[1]]))
        .collect()
}

/// Complex OTF of `pupil` at image-surface frequency vectors `nus`
/// (cycles/mm), including the image-offset phase.
pub fn otf_at(
    pupil: &PupilFunction,
    nus: &[[f64; 2]],
    method: MtfMethod,
    ctx: &JobContext,
) -> MkResult<Vec<C64>> {
    Ok(otf_and_reference(pupil, nus, method, false, ctx)?.0)
}

/// The OTF at surface frequencies `nus` and, with `with_reference`, the
/// OTF of the same pupil without aberrations or image offset.
fn otf_and_reference(
    pupil: &PupilFunction,
    nus: &[[f64; 2]],
    method: MtfMethod,
    with_reference: bool,
    ctx: &JobContext,
) -> MkResult<(Vec<C64>, Vec<C64>)> {
    match method {
        MtfMethod::Fft => {
            let otf = otf_fft(pupil, ctx)?;
            let o = nus.iter().map(|v| otf.at(v[0], v[1])).collect();
            let r = if with_reference {
                nus.iter().map(|v| otf.reference_at(v[0], v[1])).collect()
            } else {
                Vec::new()
            };
            Ok((o, r))
        }
        MtfMethod::Direct => {
            let (o, r) = if pupil.geometry().is_on_axis() {
                let sub = if with_reference { 0.5 } else { 1.0 };
                let o = otf_direct(pupil, nus, &ctx.subrange(0.0, sub))?;
                let r = if with_reference {
                    otf_direct(&pupil.without_aberrations(), nus, &ctx.subrange(0.5, 1.0))?
                } else {
                    Vec::new()
                };
                (o, r)
            } else {
                let (o, r) = otf_surface(pupil, nus, ctx)?;
                (o, if with_reference { r } else { Vec::new() })
            };
            let off = pupil.image_offset_mm();
            let tau = 2.0 * std::f64::consts::PI;
            let o = o
                .into_iter()
                .zip(nus)
                .map(|(z, v)| z * C64::from_polar(1.0, -tau * (v[0] * off[0] + v[1] * off[1])))
                .collect();
            Ok((o, r))
        }
    }
}

/// The map between the pupil lattice and the unit directions
/// `q = α·ê_u + β·ê_v − γ·ê_w` of the reference sphere in image-surface
/// coordinates. The sensor field is `∫ S·e^{ik'(q_x x + q_y y)} dq_x dq_y`
/// with the angular spectrum `S = c·A·e^{i2πW}/|q_z|` (the Jacobian
/// `|∂(q_x, q_y)/∂(α, β)| = |q_z|/γ` turns the Debye `1/γ` into `1/|q_z|`).
/// Keeping `−γ·ê_w` in `(q_x, q_y)` keeps the longitudinal phase across an
/// off-axis PSF that the chief-normal projection `B` drops.
struct SensorMap {
    frame: [Vec3; 3],
    sin_max: f64,
    n: usize,
    /// `[c·A, c, c·W]` per sample.
    cells: Vec<[f64; 3]>,
}

impl SensorMap {
    fn new(pupil: &PupilFunction) -> Self {
        let (amp, cov, opd) = (pupil.amplitude(), pupil.coverage(), pupil.opd_waves());
        SensorMap {
            frame: pupil.geometry().chief_frame(),
            sin_max: pupil.geometry().sin_max(),
            n: pupil.n(),
            cells: (0..cov.len())
                .map(|k| [cov[k] * amp[k], cov[k], cov[k] * opd[k]])
                .collect(),
        }
    }

    /// `(k, q, γ)` of every covered sample.
    fn covered(&self, pupil: &PupilFunction) -> MkResult<Vec<(usize, Vec3, f64)>> {
        let [eu, ev, ew] = self.frame;
        let n = self.n;
        (0..n * n)
            .filter(|&k| self.cells[k][1] > 0.0)
            .map(|k| {
                let (a, b) = (
                    self.sin_max * pupil.coord(k % n),
                    self.sin_max * pupil.coord(k / n),
                );
                let gamma = (1.0 - a * a - b * b).sqrt();
                let q = eu * a + ev * b - ew * gamma;
                if q.z >= 0.0 {
                    // The map to (q_x, q_y) is no longer one-to-one.
                    return Err(MkError::InvalidParam(
                        "pupil crosses the image-surface horizon".into(),
                    ));
                }
                Ok((k, q, gamma))
            })
            .collect()
    }

    /// Lattice coordinates (in samples) and `|q_z|` of the direction with
    /// transverse components `(x, y)`; `None` off the unit sphere or behind
    /// the pupil's hemisphere.
    fn lattice(&self, x: f64, y: f64) -> Option<(f64, f64, f64)> {
        let z2 = 1.0 - x * x - y * y;
        if z2 <= 0.0 {
            return None;
        }
        let q = Vec3::new(x, y, -z2.sqrt());
        let [eu, ev, ew] = self.frame;
        if q.dot(ew) >= 0.0 {
            return None;
        }
        let (s, o) = (
            0.5 * self.n as f64 / self.sin_max,
            0.5 * (self.n - 1) as f64,
        );
        Some((q.dot(eu) * s + o, q.dot(ev) * s + o, -q.z))
    }

    /// Bilinear `c·A` and `c`, and the coverage-weighted OPD (as in
    /// [`otf_direct`]), at lattice coordinates `(x, y)`.
    fn sample(&self, x: f64, y: f64) -> (f64, f64, f64) {
        let n = self.n as i64;
        let (fx, fy) = (x.floor(), y.floor());
        let (tx, ty) = (x - fx, y - fy);
        let (i, j) = (fx as i64, fy as i64);
        let (mut a, mut c, mut w) = (0.0, 0.0, 0.0);
        for (dx, dy, weight) in [
            (0, 0, (1.0 - tx) * (1.0 - ty)),
            (1, 0, tx * (1.0 - ty)),
            (0, 1, (1.0 - tx) * ty),
            (1, 1, tx * ty),
        ] {
            let (x, y) = (i + dx, j + dy);
            if x < 0 || y < 0 || x >= n || y >= n {
                continue;
            }
            let [ca, cc, cw] = self.cells[(y * n + x) as usize];
            a += weight * ca;
            c += weight * cc;
            w += weight * cw;
        }
        (a, c, if c > 0.0 { w / c } else { 0.0 })
    }
}

/// Direct autocorrelation of the angular spectrum (`SensorMap`) of an
/// off-axis pupil at image-surface frequencies, and the same without
/// aberrations: `(q_x, q_y)` shifts by `λν/n'`, integrated over the pupil
/// lattice with weight `γ/|q_z|` and normalised by `Σ c·A²/(γ|q_z|)`.
fn otf_surface(
    pupil: &PupilFunction,
    nus: &[[f64; 2]],
    ctx: &JobContext,
) -> MkResult<(Vec<C64>, Vec<C64>)> {
    let n = pupil.n();
    let map = SensorMap::new(pupil);
    let scale = pupil.wavelength_um() * 1e-3 / pupil.geometry().image_index;
    let tau = 2.0 * std::f64::consts::PI;
    let (amp, cov, opd) = (pupil.amplitude(), pupil.coverage(), pupil.opd_waves());
    let mut norm = mklight_core::math::KahanSum::default();
    let points: Vec<_> = map
        .covered(pupil)?
        .into_iter()
        .map(|(k, q, gamma)| {
            let e = cov[k] * amp[k] * amp[k] / (gamma * -q.z);
            norm.add(e);
            (k, q, cov[k] * amp[k] / gamma, (1.0 - cov[k]) * e)
        })
        .collect();
    let norm = norm.value();
    if !(norm > 0.0) {
        return Err(MkError::Numerical("pupil transmits no light".into()));
    }
    let done = std::sync::atomic::AtomicUsize::new(0);
    let values: Vec<(C64, C64)> = nus
        .par_iter()
        .map(|nu| {
            ctx.check()?;
            let [mut re, mut im, mut dl] = [(); 3].map(|_| mklight_core::math::KahanSum::default());
            for &(k, q, weight, deficit) in &points {
                let Some((ix, iy, qz)) = map.lattice(q.x + scale * nu[0], q.y + scale * nu[1])
                else {
                    continue;
                };
                let (a, _, w) = map.sample(ix, iy);
                if a <= 0.0 {
                    continue;
                }
                // Zero-lag deficit of partly covered cells, as in
                // `otf_direct`, at this sample's own shift.
                let d = if deficit > 0.0 {
                    let (i, j) = ((k % n) as f64, (k / n) as f64);
                    deficit * (1.0 - (ix - i).abs()).max(0.0) * (1.0 - (iy - j).abs()).max(0.0)
                } else {
                    0.0
                };
                let m = weight * a / qz;
                let (s, c) = (tau * (w - opd[k])).sin_cos();
                re.add(m * c + d);
                im.add(m * s);
                dl.add(m + d);
            }
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            ctx.report(d as f64 / nus.len() as f64, "");
            Ok((
                C64::new(re.value(), im.value()) / norm,
                C64::new(dl.value() / norm, 0.0),
            ))
        })
        .collect::<MkResult<_>>()?;
    Ok(values.into_iter().unzip())
}

/// Direct pupil autocorrelation `Σ_u p(u + s)·p*(u) / E` for each
/// normal-plane frequency, `s = ν/(ν-per-unit-u)` (no image-offset phase). The shifted pupil is interpolated
/// bilinearly: `|p|` directly, and the OPD as a coverage-weighted bilinear
/// mean of the covered neighbours (so the phase is never mixed with empty
/// samples). Parallel over frequencies; each sum is sequential and
/// compensated, so results are deterministic.
fn otf_direct(pupil: &PupilFunction, nus: &[[f64; 2]], ctx: &JobContext) -> MkResult<Vec<C64>> {
    let n = pupil.n();
    let w = pupil.field_weights();
    let opd = pupil.opd_waves();
    let cov = pupil.coverage();
    let tau = 2.0 * std::f64::consts::PI;
    let field = pupil.debye_field();
    let norm = pupil.energy();
    // Zero-lag deficit of the lattice sum (see `PupilFunction::energy`).
    let lag0_fix = norm - w.iter().map(|x| x * x).sum::<f64>();
    if !(norm > 0.0) {
        return Err(MkError::Numerical("pupil transmits no light".into()));
    }
    let active: Vec<usize> = (0..n * n).filter(|&k| w[k] > 0.0).collect();
    // Samples per cycle/mm: shift in sample units = ν·n·λF#.
    let per_nu = pupil.alias_free_field_mm();
    let done = std::sync::atomic::AtomicUsize::new(0);
    nus.par_iter()
        .map(|nu| -> MkResult<C64> {
            ctx.check()?;
            let (sx, sy) = (nu[0] * per_nu, nu[1] * per_nu);
            let (fx0, fy0) = (sx.floor(), sy.floor());
            let (tx, ty) = (sx - fx0, sy - fy0);
            let (dx, dy) = (fx0 as i64, fy0 as i64);
            let wts = [
                (1.0 - tx) * (1.0 - ty),
                tx * (1.0 - ty),
                (1.0 - tx) * ty,
                tx * ty,
            ];
            let offs = [(0, 0), (1, 0), (0, 1), (1, 1)];
            let (mut re, mut im) = (
                mklight_core::math::KahanSum::default(),
                mklight_core::math::KahanSum::default(),
            );
            for &k in &active {
                let (i, j) = ((k % n) as i64, (k / n) as i64);
                let (mut a, mut ws, mut wo) = (0.0, 0.0, 0.0);
                for (bw, (oi, oj)) in wts.iter().zip(offs) {
                    if *bw == 0.0 {
                        continue;
                    }
                    let (ii, jj) = (i + dx + oi, j + dy + oj);
                    if ii < 0 || jj < 0 || ii >= n as i64 || jj >= n as i64 {
                        continue;
                    }
                    let q = jj as usize * n + ii as usize;
                    if w[q] <= 0.0 {
                        continue;
                    }
                    a += bw * w[q];
                    ws += bw * cov[q];
                    wo += bw * cov[q] * opd[q];
                }
                if a <= 0.0 {
                    continue;
                }
                let shifted = C64::from_polar(a, tau * wo / ws);
                let z = shifted * field[k].conj();
                re.add(z.re);
                im.add(z.im);
            }
            for (bw, (oi, oj)) in wts.iter().zip(offs) {
                if dx + oi == 0 && dy + oj == 0 {
                    re.add(bw * lag0_fix);
                }
            }
            let d = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            ctx.report(d as f64 / nus.len() as f64, "");
            Ok(C64::new(re.value(), im.value()) / norm)
        })
        .collect()
}

fn curve_from_otf(
    freqs: &[f64],
    otf: &[C64],
    dl: &[C64],
    cutoff: f64,
    warnings: Vec<WaveWarning>,
) -> MtfCurve {
    let k = freqs.len();
    MtfCurve {
        frequencies: freqs.to_vec(),
        tangential: otf[..k].iter().map(|z| z.norm()).collect(),
        sagittal: otf[k..].iter().map(|z| z.norm()).collect(),
        diffraction_limit: dl[..k].iter().map(|z| z.norm()).collect(),
        cutoff,
        warnings,
    }
}

fn check_freqs(freqs: &[f64]) -> MkResult<()> {
    if freqs.iter().any(|f| !f.is_finite() || *f < 0.0) {
        return Err(MkError::InvalidParam(
            "MTF frequencies must be finite and ≥ 0".into(),
        ));
    }
    Ok(())
}

/// Monochromatic MTF slices (see the module docs for the methods).
pub fn mtf(
    pupil: &PupilFunction,
    freqs: &[f64],
    orientation: &MtfOrientation,
    method: MtfMethod,
    ctx: &JobContext,
) -> MkResult<MtfCurve> {
    polychromatic_mtf(&[(pupil, 1.0)], freqs, orientation, method, ctx)
}

/// MTF by FFT. Shorthand for [`mtf`] with [`MtfMethod::Fft`].
pub fn mtf_fft(
    pupil: &PupilFunction,
    freqs: &[f64],
    orientation: &MtfOrientation,
    ctx: &JobContext,
) -> MkResult<MtfCurve> {
    mtf(pupil, freqs, orientation, MtfMethod::Fft, ctx)
}

/// MTF by direct pupil autocorrelation. Shorthand for [`mtf`] with
/// [`MtfMethod::Direct`].
pub fn mtf_direct(
    pupil: &PupilFunction,
    freqs: &[f64],
    orientation: &MtfOrientation,
    ctx: &JobContext,
) -> MkResult<MtfCurve> {
    mtf(pupil, freqs, orientation, MtfMethod::Direct, ctx)
}

/// Polychromatic MTF: `|Σ ŵ_λ·OTF_λ(ν)|` with `ŵ = w/Σw`. Each `OTF_λ` is
/// normalised to 1 at zero frequency (so the weights act on energy) and
/// carries its image-offset phase, so lateral colour lowers the MTF as it
/// should. The diffraction limit is the same sum for unaberrated pupils
/// without offsets.
pub fn polychromatic_mtf(
    pupils: &[(&PupilFunction, f64)],
    freqs: &[f64],
    orientation: &MtfOrientation,
    method: MtfMethod,
    ctx: &JobContext,
) -> MkResult<MtfCurve> {
    check_freqs(freqs)?;
    let wsum: f64 = pupils.iter().map(|(_, w)| *w).sum();
    if pupils.is_empty() || !(wsum > 0.0) || pupils.iter().any(|(_, w)| *w < 0.0) {
        return Err(MkError::InvalidParam(
            "MTF needs positive spectral weights".into(),
        ));
    }
    let nus = freq_vectors(freqs, orientation);
    let mut acc = vec![C64::default(); nus.len()];
    let mut dl = vec![C64::default(); nus.len()];
    let mut warnings = Vec::new();
    let mut cutoff: f64 = 0.0;
    let np = pupils.len() as f64;
    for (q, (p, w)) in pupils.iter().enumerate() {
        let sub = ctx.subrange(q as f64 / np, (q + 1) as f64 / np);
        let wn = w / wsum;
        let (o, d) = otf_and_reference(p, &nus, method, true, &sub)?;
        for k in 0..nus.len() {
            acc[k] += o[k] * wn;
            dl[k] += d[k] * wn;
        }
        for wv in p.sampling_warnings() {
            push_unique(&mut warnings, wv);
        }
        cutoff = cutoff.max(p.geometry().cutoff_frequency(p.wavelength_um()));
    }
    Ok(curve_from_otf(freqs, &acc, &dl, cutoff, warnings))
}

/// One covered sample of a direct autocorrelation at a fixed frequency:
/// the coverage-weighted bilinear stencil of the shifted OPD and the
/// magnitude `|p(u + s)·p*(u)|` (as in [`otf_direct`] and [`otf_surface`]).
struct FocusTerm {
    source: usize,
    neighbours: [usize; 4],
    coefficients: [f64; 4],
    magnitude: f64,
}

/// The terms of one frequency, plus the zero-lag deficit, the
/// normalisation and the image-offset phasor.
struct FocusCorrelation {
    terms: Vec<FocusTerm>,
    deficit: f64,
    norm: f64,
    offset: C64,
}

/// Defocus changes only the OPD, so the stencils are prepared once.
fn focus_correlations(
    p: &PupilFunction,
    nus: &[[f64; 2]],
    ctx: &JobContext,
) -> MkResult<Vec<FocusCorrelation>> {
    let n = p.n();
    let map = SensorMap::new(p);
    let axis = p.geometry().is_on_axis();
    let points = map.covered(p)?;
    let field_weights = p.field_weights();
    let (amp, cov) = (p.amplitude(), p.coverage());
    let norm = if axis {
        p.energy()
    } else {
        points
            .iter()
            .map(|&(k, q, gamma)| cov[k] * amp[k] * amp[k] / (gamma * -q.z))
            .collect::<mklight_core::math::KahanSum>()
            .value()
    };
    if !(norm > 0.0) {
        return Err(MkError::Numerical("pupil transmits no light".into()));
    }
    let lag0 = norm - field_weights.iter().map(|w| w * w).sum::<f64>();
    let scale = p.wavelength_um() * 1e-3 / p.geometry().image_index;
    let per_nu = p.alias_free_field_mm();
    let off = p.image_offset_mm();
    nus.par_iter()
        .map(|nu| {
            ctx.check()?;
            let mut terms = Vec::with_capacity(points.len());
            let mut deficit = mklight_core::math::KahanSum::default();
            for &(k, q, gamma) in &points {
                let (ix, iy, qz) = if axis {
                    (
                        (k % n) as f64 + nu[0] * per_nu,
                        (k / n) as f64 + nu[1] * per_nu,
                        1.0,
                    )
                } else {
                    let Some(x) = map.lattice(q.x + scale * nu[0], q.y + scale * nu[1]) else {
                        continue;
                    };
                    x
                };
                let (i, j) = (ix.floor() as i64, iy.floor() as i64);
                let (tx, ty) = (ix - ix.floor(), iy - iy.floor());
                let weights = [
                    (1.0 - tx) * (1.0 - ty),
                    tx * (1.0 - ty),
                    (1.0 - tx) * ty,
                    tx * ty,
                ];
                let mut neighbours = [k; 4];
                let mut coefficients = [0.0; 4];
                let (mut a, mut c) = (0.0, 0.0);
                for (s, (dx, dy)) in [(0, 0), (1, 0), (0, 1), (1, 1)].into_iter().enumerate() {
                    let (x, y) = (i + dx, j + dy);
                    if x < 0 || y < 0 || x >= n as i64 || y >= n as i64 {
                        continue;
                    }
                    let index = y as usize * n + x as usize;
                    if axis && field_weights[index] <= 0.0 {
                        continue;
                    }
                    neighbours[s] = index;
                    coefficients[s] = weights[s] * cov[index];
                    c += coefficients[s];
                    a += weights[s]
                        * if axis {
                            field_weights[index]
                        } else {
                            cov[index] * amp[index]
                        };
                }
                if a <= 0.0 {
                    continue;
                }
                coefficients.iter_mut().for_each(|x| *x /= c);
                let magnitude = if axis {
                    field_weights[k] * a
                } else {
                    let e = cov[k] * amp[k] * amp[k] / (gamma * -q.z);
                    deficit.add(
                        (1.0 - cov[k])
                            * e
                            * (1.0 - (ix - (k % n) as f64).abs()).max(0.0)
                            * (1.0 - (iy - (k / n) as f64).abs()).max(0.0),
                    );
                    cov[k] * amp[k] / gamma * a / qz
                };
                terms.push(FocusTerm {
                    source: k,
                    neighbours,
                    coefficients,
                    magnitude,
                });
            }
            if axis {
                let (sx, sy) = (nu[0] * per_nu, nu[1] * per_nu);
                deficit.add(lag0 * (1.0 - sx.abs()).max(0.0) * (1.0 - sy.abs()).max(0.0));
            }
            Ok(FocusCorrelation {
                terms,
                deficit: deficit.value(),
                norm,
                offset: C64::from_polar(
                    1.0,
                    -2.0 * std::f64::consts::PI * (nu[0] * off[0] + nu[1] * off[1]),
                ),
            })
        })
        .collect()
}

/// [`polychromatic_mtf`] by [`MtfMethod::Direct`] at each image-plane
/// shift in `focus` (mm, as in [`through_focus_mtf`]). Each pupil's
/// autocorrelation stencils are prepared once and reused for every plane.
pub(crate) fn polychromatic_focus_mtf(
    pupils: &[(&PupilFunction, f64)],
    focus: &[f64],
    freqs: &[f64],
    orientation: &MtfOrientation,
    ctx: &JobContext,
) -> MkResult<Vec<MtfCurve>> {
    let nus = freq_vectors(freqs, orientation);
    let wsum: f64 = pupils.iter().map(|(_, w)| w).sum();
    let tau = 2.0 * std::f64::consts::PI;
    let mut acc = vec![vec![C64::default(); nus.len()]; focus.len()];
    let mut dl = vec![C64::default(); nus.len()];
    let mut warnings = vec![Vec::new(); focus.len()];
    let mut cutoff: f64 = 0.0;
    let np = pupils.len() as f64;
    for (q, &(p, w)) in pupils.iter().enumerate() {
        let sub = ctx.subrange(q as f64 / np, (q + 1) as f64 / np);
        let wn = w / wsum;
        let prepared = focus_correlations(p, &nus, &sub.subrange(0.0, 0.2))?;
        for (d, c) in dl.iter_mut().zip(&prepared) {
            let sum: mklight_core::math::KahanSum = c.terms.iter().map(|t| t.magnitude).collect();
            *d += C64::new((sum.value() + c.deficit) / c.norm, 0.0) * wn;
        }
        let planes: Vec<(Vec<C64>, Vec<WaveWarning>)> = focus
            .par_iter()
            .map(|&dz| {
                sub.check()?;
                let shifted = p.with_focus_shift(dz / p.geometry().chief_ray[2]);
                let opd = shifted.opd_waves();
                let otf = prepared
                    .iter()
                    .map(|c| {
                        let mut re = mklight_core::math::KahanSum::default();
                        let mut im = mklight_core::math::KahanSum::default();
                        for t in &c.terms {
                            let w: f64 = t
                                .neighbours
                                .iter()
                                .zip(&t.coefficients)
                                .map(|(&k, c)| c * opd[k])
                                .sum();
                            let (s, cs) = (tau * (w - opd[t.source])).sin_cos();
                            re.add(t.magnitude * cs);
                            im.add(t.magnitude * s);
                        }
                        C64::new(re.value() + c.deficit, im.value()) / c.norm * c.offset
                    })
                    .collect();
                Ok((otf, shifted.sampling_warnings()))
            })
            .collect::<MkResult<_>>()?;
        for ((acc, warnings), (otf, plane)) in acc.iter_mut().zip(&mut warnings).zip(planes) {
            for (a, z) in acc.iter_mut().zip(otf) {
                *a += z * wn;
            }
            for x in plane {
                push_unique(warnings, x);
            }
        }
        cutoff = cutoff.max(p.geometry().cutoff_frequency(p.wavelength_um()));
        sub.report(1.0, "through-focus MTF");
    }
    Ok(acc
        .iter()
        .zip(warnings)
        .map(|(a, w)| curve_from_otf(freqs, a, &dl, cutoff, w))
        .collect())
}

/// A focus change for [`through_focus_mtf`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Defocus {
    /// Move the image plane by this many mm along the axis (exact
    /// reference-sphere shift, [`PupilFunction::with_focus_shift`]).
    ShiftMm { mm: f64 },
    /// Add `waves·ρ²` of defocus.
    Waves { waves: f64 },
}

/// MTF at each focus setting: one curve per entry of `focus`.
pub fn through_focus_mtf(
    pupil: &PupilFunction,
    focus: &[Defocus],
    freqs: &[f64],
    orientation: &MtfOrientation,
    method: MtfMethod,
    ctx: &JobContext,
) -> MkResult<Vec<MtfCurve>> {
    let nf = focus.len().max(1) as f64;
    focus
        .iter()
        .enumerate()
        .map(|(q, d)| {
            let p = match *d {
                Defocus::ShiftMm { mm } => pupil.with_focus_shift(mm),
                Defocus::Waves { waves } => pupil.with_defocus_waves(waves),
            };
            mtf(
                &p,
                freqs,
                orientation,
                method,
                &ctx.subrange(q as f64 / nf, (q + 1) as f64 / nf),
            )
        })
        .collect()
}

/// MTF of a sampled PSF by direct DFT,
/// `OTF(ν) = Δx²·Σ PSF_k·e^{−2πiν·(x_k − x_c)} / total_energy` (and exactly
/// 1 at ν = 0).
///
/// Normalising by the PSF's whole-plane energy rather than by the energy
/// inside the grid keeps the result right when the grid truncates the far
/// wings: those only carry frequencies below `~1/(grid width)`, which is the
/// only range a truncated grid gets wrong. Frequencies above `1/(2Δx)` are
/// flagged.
pub fn mtf_from_psf(
    psf: &Psf,
    freqs: &[f64],
    orientation: &MtfOrientation,
    ctx: &JobContext,
) -> MkResult<MtfCurve> {
    check_freqs(freqs)?;
    let nus = freq_vectors(freqs, orientation);
    let m = psf.size;
    let c = psf.center_index() as f64;
    let dx = psf.spacing_mm;
    let total = psf.total_energy / (dx * dx);
    if !(total > 0.0) {
        return Err(MkError::Numerical("PSF has no energy".into()));
    }
    let tau = 2.0 * std::f64::consts::PI;
    let otf: Vec<C64> = nus
        .par_iter()
        .map(|nu| -> MkResult<C64> {
            ctx.check()?;
            // Separable phase: e^{−2πi(νx·x + νy·y)}.
            let ex: Vec<C64> = (0..m)
                .map(|i| C64::from_polar(1.0, -tau * nu[0] * (i as f64 - c) * dx))
                .collect();
            let mut acc = C64::default();
            for j in 0..m {
                let ey = C64::from_polar(1.0, -tau * nu[1] * (j as f64 - c) * dx);
                let row = &psf.data[j * m..(j + 1) * m];
                let s: C64 = row.iter().zip(&ex).map(|(v, e)| e * *v).sum();
                acc += s * ey;
            }
            if nu[0] == 0.0 && nu[1] == 0.0 {
                return Ok(C64::new(1.0, 0.0));
            }
            Ok(acc / total)
        })
        .collect::<MkResult<_>>()?;
    let mut warnings = psf.warnings.clone();
    let nyq = 0.5 / dx;
    if let Some(&f) = freqs.iter().find(|&&f| f > nyq * (1.0 + 1e-12)) {
        push_unique(
            &mut warnings,
            WaveWarning::FrequencyBeyondSampling {
                frequency: f,
                limit: nyq,
            },
        );
    }
    let mut curve = curve_from_otf(freqs, &otf, &otf, f64::NAN, warnings);
    // A sampled PSF carries no aperture description.
    curve.diffraction_limit.clear();
    Ok(curve)
}

#[cfg(test)]
mod focus_tests {
    use super::*;

    #[test]
    fn cached_focus_matches_independent_pupil_autocorrelations() {
        for system in [
            mklight_core::samples::double_gauss(),
            mklight_core::samples::anamorphic_adapter(),
        ] {
            let t = mklight_analysis::tracer_for(system).unwrap();
            for field in [0, t.fields().len() - 1] {
                let pupils: Vec<_> = (0..2)
                    .map(|wl| {
                        crate::trace_pupil(
                            &t,
                            t.field(field),
                            wl,
                            &crate::PupilSettings {
                                grid: 64,
                                rays: 33,
                                ..Default::default()
                            },
                            &JobContext::new(),
                        )
                        .unwrap()
                    })
                    .collect();
                let weighted: Vec<_> = pupils.iter().map(|p| (&p.pupil, 1.0)).collect();
                let focus = [-0.037, 0.0, 0.042];
                let frequencies = [0.0, 7.3, 31.0, 85.0];
                let o = MtfOrientation::from_field(0.3, 0.7);
                let cached = polychromatic_focus_mtf(
                    &weighted,
                    &focus,
                    &frequencies,
                    &o,
                    &JobContext::new(),
                )
                .unwrap();
                for (i, dz) in focus.into_iter().enumerate() {
                    let shifted: Vec<_> = pupils
                        .iter()
                        .map(|p| {
                            p.pupil
                                .with_focus_shift(dz / p.pupil.geometry().chief_ray[2])
                        })
                        .collect();
                    let weighted: Vec<_> = shifted.iter().map(|p| (p, 1.0)).collect();
                    let want = polychromatic_mtf(
                        &weighted,
                        &frequencies,
                        &o,
                        MtfMethod::Direct,
                        &JobContext::new(),
                    )
                    .unwrap();
                    for (a, b) in cached[i]
                        .tangential
                        .iter()
                        .chain(&cached[i].sagittal)
                        .chain(&cached[i].diffraction_limit)
                        .zip(
                            want.tangential
                                .iter()
                                .chain(&want.sagittal)
                                .chain(&want.diffraction_limit),
                        )
                    {
                        assert!((a - b).abs() < 2e-12, "field={field}, dz={dz}: {a} vs {b}");
                    }
                }
            }
        }
    }
}
