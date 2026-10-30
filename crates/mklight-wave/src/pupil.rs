//! The sampled exit-pupil function.
//!
//! # Sampling convention
//!
//! A [`PupilFunction`] is an `n × n` grid over the square `[−1, 1]²` of
//! normalised exit-pupil coordinates `(u, v)`; the unit disk is the full
//! aperture. Samples are **cell centred**: sample `i` sits at
//! `u_i = (2i + 1 − n)/n`, spacing `δ = 2/n`, so the grid is symmetric about
//! the axis and each sample stands for a `δ × δ` cell. Storage is row-major,
//! `index = j·n + i` with `i` along `u` (sagittal, image x) and `j` along `v`
//! (tangential, image y).
//!
//! `(u, v)` are proportional to the direction cosines of the converging
//! wave in image space, `(α, β) = sinθ'max · (u, v)` with
//! `sinθ'max = a/R` (`a` the exit-pupil radius on the reference sphere, `R`
//! the reference-sphere radius). A grid uniform in `(u, v)` is therefore
//! uniform in direction cosines, which makes the image field a Fourier
//! transform of the pupil field (Debye integral) and lets the FFT be exact
//! for it.
//!
//! # Field
//!
//! Each sample stores the amplitude `A ≥ 0`, the coverage `c ∈ [0, 1]` (the
//! fraction of the cell inside the clear aperture, which carries
//! obscurations and vignetting), and the OPD `W` in waves (CONVENTIONS §4
//! sign). The complex pupil function is `P = c·A·exp(i2πW)`.
//!
//! `A` is the field amplitude on the reference sphere, i.e. the square root
//! of the irradiance per unit *sphere area* (transmission × apodization ×
//! ray-density factor). For an aplanatic system with a uniformly filled
//! entrance pupil this is `A ∝ √γ`; at low NA it is simply the transmission
//! amplitude.
//!
//! Transforms use the Debye-weighted field `p = P/γ`,
//! `γ = √(1 − sin²θ'max·(u² + v²))`, because the area element on the
//! reference sphere is `R²·dα·dβ/γ` (scalar Richards–Wolf/Debye integral).
//! `1/γ` is within 1 % of one for `F/# ≳ 3.5`.
//!
//! The phase convention is `exp(+i2πW)` with image kernel
//! `exp(+i2π·ν·x)` (see [`crate::psf`]); this is the complex conjugate of
//! the physical `e^{−iωt}` field, which leaves every intensity unchanged.

use mklight_core::{MkError, MkResult};
use num_complex::Complex64 as C64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::warnings::WaveWarning;
use crate::zernike::ZernikeSet;

/// Image-space geometry needed to scale image-plane coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PupilGeometry {
    /// Transverse radius `a` of the exit pupil on the reference sphere (mm).
    pub exit_pupil_radius_mm: f64,
    /// Reference-sphere radius `R`: exit pupil → reference image point (mm,
    /// positive).
    pub reference_radius_mm: f64,
    /// Image-space refractive index `n'` (relative to air).
    pub image_index: f64,
}

impl PupilGeometry {
    pub fn new(
        exit_pupil_radius_mm: f64,
        reference_radius_mm: f64,
        image_index: f64,
    ) -> MkResult<Self> {
        let g = PupilGeometry {
            exit_pupil_radius_mm,
            reference_radius_mm,
            image_index,
        };
        g.validate()?;
        Ok(g)
    }

    /// Geometry with working F/# `1/(2 n' sinθ')` and reference radius `R`.
    pub fn from_working_f_number(
        f_number: f64,
        reference_radius_mm: f64,
        image_index: f64,
    ) -> MkResult<Self> {
        let sin_max = 1.0 / (2.0 * f_number * image_index);
        Self::new(
            sin_max * reference_radius_mm,
            reference_radius_mm,
            image_index,
        )
    }

    pub fn validate(&self) -> MkResult<()> {
        let ok = self.exit_pupil_radius_mm > 0.0
            && self.reference_radius_mm > self.exit_pupil_radius_mm
            && self.image_index > 0.0
            && self.reference_radius_mm.is_finite();
        if ok {
            Ok(())
        } else {
            Err(MkError::InvalidModel(format!(
                "invalid pupil geometry {self:?}"
            )))
        }
    }

    /// `sinθ'max = a/R`.
    pub fn sin_max(&self) -> f64 {
        self.exit_pupil_radius_mm / self.reference_radius_mm
    }
    /// Image-space numerical aperture `n'·a/R`.
    pub fn numerical_aperture(&self) -> f64 {
        self.image_index * self.sin_max()
    }
    /// Working F/# `1/(2·NA)`.
    pub fn working_f_number(&self) -> f64 {
        0.5 / self.numerical_aperture()
    }
    /// `λ·F#` in mm for a wavelength in µm: the natural image-plane length
    /// unit (Airy first zero at 1.22·λF#).
    pub fn lambda_f_mm(&self, wavelength_um: f64) -> f64 {
        wavelength_um * 1e-3 * self.working_f_number()
    }
    /// Incoherent cutoff frequency `1/(λ·F#) = 2NA/λ` (cycles/mm).
    pub fn cutoff_frequency(&self, wavelength_um: f64) -> f64 {
        1.0 / self.lambda_f_mm(wavelength_um)
    }
}

/// Clear-aperture shape in normalised pupil coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Aperture {
    /// Unit disk.
    Circular,
    /// Unit disk with a central obscuration of radius `obscuration` (0…1).
    Annular { obscuration: f64 },
    /// `|u| ≤ half_u`, `|v| ≤ half_v`.
    Rectangular { half_u: f64, half_v: f64 },
}

impl Aperture {
    pub fn contains(&self, u: f64, v: f64) -> bool {
        let r2 = u * u + v * v;
        match *self {
            Aperture::Circular => r2 <= 1.0,
            Aperture::Annular { obscuration } => r2 <= 1.0 && r2 >= obscuration * obscuration,
            Aperture::Rectangular { half_u, half_v } => u.abs() <= half_u && v.abs() <= half_v,
        }
    }
}

/// Amplitude apodization across the pupil.
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Apodization {
    #[default]
    Uniform,
    /// Gaussian amplitude `edge_intensity^(ρ²/2)`: the intensity falls to
    /// `edge_intensity` at `ρ = 1`.
    Gaussian { edge_intensity: f64 },
}

impl Apodization {
    pub fn amplitude(&self, u: f64, v: f64) -> f64 {
        match *self {
            Apodization::Uniform => 1.0,
            Apodization::Gaussian { edge_intensity } => {
                (0.5 * (u * u + v * v) * edge_intensity.ln()).exp()
            }
        }
    }
}

/// Sampled complex exit-pupil function at one wavelength (see the module
/// docs for the sampling convention).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PupilFunction {
    n: usize,
    wavelength_um: f64,
    geometry: PupilGeometry,
    /// Position of this pupil's reference point (chief ray on the image
    /// surface) relative to the common image reference, mm. Non-zero for
    /// lateral colour in polychromatic results.
    image_offset_mm: [f64; 2],
    amplitude: Vec<f64>,
    coverage: Vec<f64>,
    opd_waves: Vec<f64>,
}

/// Fewest samples across the pupil for which results are trusted without a
/// warning.
pub const MIN_PUPIL_SAMPLES: usize = 32;

impl PupilFunction {
    /// Builds from raw sample arrays (each `n²`, row-major). Amplitude and
    /// OPD are ignored (set to 0) where the coverage is 0.
    pub fn from_samples(
        n: usize,
        wavelength_um: f64,
        geometry: PupilGeometry,
        amplitude: Vec<f64>,
        coverage: Vec<f64>,
        opd_waves: Vec<f64>,
    ) -> MkResult<Self> {
        geometry.validate()?;
        if n < 2 || amplitude.len() != n * n || coverage.len() != n * n || opd_waves.len() != n * n
        {
            return Err(MkError::InvalidParam(format!(
                "pupil arrays must hold {n}×{n} samples"
            )));
        }
        if !(wavelength_um > 0.0) {
            return Err(MkError::InvalidParam(format!(
                "wavelength {wavelength_um} µm"
            )));
        }
        let s2max = geometry.sin_max().powi(2);
        let mut amplitude = amplitude;
        let mut opd_waves = opd_waves;
        for k in 0..n * n {
            let c = coverage[k];
            if !(0.0..=1.0).contains(&c) {
                return Err(MkError::InvalidParam(format!(
                    "pupil coverage {c} at sample {k}"
                )));
            }
            if c == 0.0 {
                amplitude[k] = 0.0;
                opd_waves[k] = 0.0;
                continue;
            }
            if !(amplitude[k] >= 0.0 && amplitude[k].is_finite() && opd_waves[k].is_finite()) {
                return Err(MkError::InvalidParam(format!("bad pupil sample {k}")));
            }
            if Self::rho2_at(n, k) * s2max >= 1.0 {
                return Err(MkError::InvalidModel(
                    "pupil samples beyond the evanescent limit (sinθ ≥ 1)".into(),
                ));
            }
        }
        Ok(PupilFunction {
            n,
            wavelength_um,
            geometry,
            image_offset_mm: [0.0; 2],
            amplitude,
            coverage,
            opd_waves,
        })
    }

    fn rho2_at(n: usize, k: usize) -> f64 {
        let u = Self::coord_of(n, k % n);
        let v = Self::coord_of(n, k / n);
        u * u + v * v
    }

    #[inline]
    fn coord_of(n: usize, i: usize) -> f64 {
        (2.0 * i as f64 + 1.0 - n as f64) / n as f64
    }

    /// Samples an aperture and a field function on the grid.
    ///
    /// Coverage is the fraction of `supersample × supersample` sub-points of
    /// each cell for which `inside(u, v)` holds (`supersample = 1` gives a
    /// hard mask tested at the cell centre). `field(u, v) → (A, W_waves)` is
    /// evaluated once per covered cell, at the centroid of its covered
    /// sub-points, so edge cells never query the field far outside the
    /// aperture.
    pub fn from_fn(
        n: usize,
        wavelength_um: f64,
        geometry: PupilGeometry,
        supersample: usize,
        inside: impl Fn(f64, f64) -> bool + Sync,
        field: impl Fn(f64, f64) -> (f64, f64) + Sync,
    ) -> MkResult<Self> {
        let s = supersample.max(1);
        let d = 2.0 / n as f64;
        let rows: Vec<Vec<(f64, f64, f64)>> = (0..n)
            .into_par_iter()
            .map(|j| {
                let v0 = Self::coord_of(n, j);
                (0..n)
                    .map(|i| {
                        let u0 = Self::coord_of(n, i);
                        let (mut cnt, mut su, mut sv) = (0usize, 0.0, 0.0);
                        for b in 0..s {
                            let v = v0 + d * ((b as f64 + 0.5) / s as f64 - 0.5);
                            for a in 0..s {
                                let u = u0 + d * ((a as f64 + 0.5) / s as f64 - 0.5);
                                if inside(u, v) {
                                    cnt += 1;
                                    su += u;
                                    sv += v;
                                }
                            }
                        }
                        if cnt == 0 {
                            return (0.0, 0.0, 0.0);
                        }
                        let (amp, w) = field(su / cnt as f64, sv / cnt as f64);
                        (cnt as f64 / (s * s) as f64, amp, w)
                    })
                    .collect()
            })
            .collect();
        let mut cov = Vec::with_capacity(n * n);
        let mut amp = Vec::with_capacity(n * n);
        let mut opd = Vec::with_capacity(n * n);
        for (c, a, w) in rows.into_iter().flatten() {
            cov.push(c);
            amp.push(a);
            opd.push(w);
        }
        Self::from_samples(n, wavelength_um, geometry, amp, cov, opd)
    }

    /// An aperture with apodization and an OPD function (waves).
    pub fn from_aperture(
        n: usize,
        wavelength_um: f64,
        geometry: PupilGeometry,
        aperture: Aperture,
        apodization: Apodization,
        supersample: usize,
        opd_waves: impl Fn(f64, f64) -> f64 + Sync,
    ) -> MkResult<Self> {
        Self::from_fn(
            n,
            wavelength_um,
            geometry,
            supersample,
            |u, v| aperture.contains(u, v),
            |u, v| (apodization.amplitude(u, v), opd_waves(u, v)),
        )
    }

    /// Aberration-free pupil.
    pub fn diffraction_limited(
        n: usize,
        wavelength_um: f64,
        geometry: PupilGeometry,
        aperture: Aperture,
        supersample: usize,
    ) -> MkResult<Self> {
        Self::from_aperture(
            n,
            wavelength_um,
            geometry,
            aperture,
            Apodization::Uniform,
            supersample,
            |_, _| 0.0,
        )
    }

    /// Wavefront given by Zernike coefficients (waves).
    pub fn from_zernike(
        n: usize,
        wavelength_um: f64,
        geometry: PupilGeometry,
        aperture: Aperture,
        apodization: Apodization,
        supersample: usize,
        zernike: &ZernikeSet,
    ) -> MkResult<Self> {
        let basis = zernike.basis();
        let c = &zernike.coefficients;
        Self::from_aperture(
            n,
            wavelength_um,
            geometry,
            aperture,
            apodization,
            supersample,
            |u, v| basis.eval(u, v, c),
        )
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn wavelength_um(&self) -> f64 {
        self.wavelength_um
    }
    pub fn geometry(&self) -> &PupilGeometry {
        &self.geometry
    }
    pub fn image_offset_mm(&self) -> [f64; 2] {
        self.image_offset_mm
    }
    pub fn set_image_offset_mm(&mut self, offset: [f64; 2]) {
        self.image_offset_mm = offset;
    }
    /// Normalised coordinate of sample index `i` (either axis).
    pub fn coord(&self, i: usize) -> f64 {
        Self::coord_of(self.n, i)
    }
    /// Sample spacing `δ = 2/n` in normalised units.
    pub fn spacing(&self) -> f64 {
        2.0 / self.n as f64
    }
    pub fn amplitude(&self) -> &[f64] {
        &self.amplitude
    }
    pub fn coverage(&self) -> &[f64] {
        &self.coverage
    }
    pub fn opd_waves(&self) -> &[f64] {
        &self.opd_waves
    }
    /// `λ·F#` (mm).
    pub fn lambda_f_mm(&self) -> f64 {
        self.geometry.lambda_f_mm(self.wavelength_um)
    }
    /// Width of the alias-free image field `L = n·λ·F#` (mm): the period of
    /// the PSF implied by the pupil sampling. Image-plane spacing of a
    /// transform padded to `M` samples is `L/M`.
    pub fn alias_free_field_mm(&self) -> f64 {
        self.n as f64 * self.lambda_f_mm()
    }
    /// Spatial frequency per unit normalised pupil coordinate:
    /// `ν = u·NA/λ = u/(2λF#)` (cycles/mm).
    pub fn frequency_scale(&self) -> f64 {
        0.5 / self.lambda_f_mm()
    }

    /// Debye factor `1/γ` at `(u, v)`.
    pub fn debye_factor(&self, u: f64, v: f64) -> f64 {
        let s2 = self.geometry.sin_max().powi(2) * (u * u + v * v);
        1.0 / ((1.0 - s2).max(1e-300)).sqrt()
    }

    /// Debye-weighted field magnitude `|p| = c·A/γ` per sample.
    pub fn field_weights(&self) -> Vec<f64> {
        let n = self.n;
        (0..n * n)
            .map(|k| {
                let c = self.coverage[k];
                if c <= 0.0 {
                    0.0
                } else {
                    c * self.amplitude[k] * self.debye_factor(self.coord(k % n), self.coord(k / n))
                }
            })
            .collect()
    }

    /// Debye-weighted complex field `p = c·A·exp(i2πW)/γ` per sample.
    pub fn debye_field(&self) -> Vec<C64> {
        let tau = 2.0 * std::f64::consts::PI;
        self.field_weights()
            .iter()
            .zip(&self.opd_waves)
            .map(|(&w, &opd)| {
                if w == 0.0 {
                    C64::default()
                } else {
                    C64::from_polar(w, tau * opd)
                }
            })
            .collect()
    }

    /// A copy with `extra(u, v)` waves added to the OPD of covered samples.
    pub fn with_added_opd(&self, extra: impl Fn(f64, f64) -> f64) -> Self {
        let mut p = self.clone();
        let n = self.n;
        for k in 0..n * n {
            if p.coverage[k] > 0.0 {
                p.opd_waves[k] += extra(self.coord(k % n), self.coord(k / n));
            }
        }
        p
    }

    /// A copy with the same aperture and amplitude and zero OPD.
    pub fn without_aberrations(&self) -> Self {
        let mut p = self.clone();
        p.opd_waves.iter_mut().for_each(|w| *w = 0.0);
        p
    }

    /// OPD (waves) that moves the reference point by `dz` mm along the axis
    /// (`dz > 0` away from the pupil). Exact on the reference sphere:
    /// `W = (n'/λ)·[r(1) − r(γ)]` with `r(γ) = √(R² + 2Rγ·dz + dz²)`, so a
    /// positive shift adds positive OPD at the pupil edge (CONVENTIONS §4:
    /// the wave then converges more than the shifted reference sphere).
    /// Paraxially `W ≈ dz·NA²·ρ²/(2n'λ)`.
    pub fn focus_shift_opd(&self, dz_mm: f64, u: f64, v: f64) -> f64 {
        let g = &self.geometry;
        let r = g.reference_radius_mm;
        // r(γ) − R in cancellation-free form.
        let dr = |gamma: f64| {
            (2.0 * r * gamma * dz_mm + dz_mm * dz_mm)
                / ((r * r + 2.0 * r * gamma * dz_mm + dz_mm * dz_mm).sqrt() + r)
        };
        let s = g.sin_max() * (u * u + v * v).sqrt();
        let gamma = ((1.0 - s) * (1.0 + s)).max(0.0).sqrt();
        g.image_index / (self.wavelength_um * 1e-3) * (dr(1.0) - dr(gamma))
    }

    /// A copy refocused by `dz` mm (see [`Self::focus_shift_opd`]).
    pub fn with_focus_shift(&self, dz_mm: f64) -> Self {
        self.with_added_opd(|u, v| self.focus_shift_opd(dz_mm, u, v))
    }

    /// A copy with `w020·ρ²` waves of defocus added.
    pub fn with_defocus_waves(&self, w020: f64) -> Self {
        self.with_added_opd(|u, v| w020 * (u * u + v * v))
    }

    /// Largest OPD change (waves) between covered neighbouring samples.
    pub fn max_phase_step(&self) -> f64 {
        let n = self.n;
        let mut m: f64 = 0.0;
        for j in 0..n {
            for i in 0..n {
                let k = j * n + i;
                if self.coverage[k] <= 0.0 {
                    continue;
                }
                if i + 1 < n && self.coverage[k + 1] > 0.0 {
                    m = m.max((self.opd_waves[k + 1] - self.opd_waves[k]).abs());
                }
                if j + 1 < n && self.coverage[k + n] > 0.0 {
                    m = m.max((self.opd_waves[k + n] - self.opd_waves[k]).abs());
                }
            }
        }
        m
    }

    /// Samples across the clear aperture's widest extent.
    pub fn samples_across(&self) -> usize {
        let n = self.n;
        let mut lo = (n, n);
        let mut hi = (0, 0);
        for j in 0..n {
            for i in 0..n {
                if self.coverage[j * n + i] > 0.0 {
                    lo = (lo.0.min(i), lo.1.min(j));
                    hi = (hi.0.max(i), hi.1.max(j));
                }
            }
        }
        if hi.0 < lo.0 {
            0
        } else {
            (hi.0 - lo.0 + 1).max(hi.1 - lo.1 + 1)
        }
    }

    /// Pupil-side sampling diagnostics.
    pub fn sampling_warnings(&self) -> Vec<WaveWarning> {
        let mut w = Vec::new();
        let across = self.samples_across();
        if across < MIN_PUPIL_SAMPLES {
            w.push(WaveWarning::FewPupilSamples {
                samples: across,
                recommended: MIN_PUPIL_SAMPLES,
            });
        }
        let step = self.max_phase_step();
        if step > 0.5 {
            w.push(WaveWarning::PupilUndersampled {
                max_step_waves: step,
            });
        }
        w
    }

    /// Weighted OPD statistics (waves) with weights `|p| = c·A/γ`, the
    /// weights under which `Strehl ≈ 1 − (2π·σ)²`. Piston is always
    /// removed; with `remove_tilt` the best-fit tilt is removed as well
    /// (the RMS that governs the PSF *peak* rather than the value at the
    /// reference point).
    #[allow(clippy::needless_range_loop)]
    pub fn opd_stats(&self, remove_tilt: bool) -> OpdStats {
        let w = self.field_weights();
        let n = self.n;
        // Weighted least squares on {1, u, v} (or {1}) via 3×3 normal equations.
        let mut g = [[0.0f64; 3]; 3];
        let mut b = [0.0f64; 3];
        for k in 0..n * n {
            if w[k] <= 0.0 {
                continue;
            }
            let f = [1.0, self.coord(k % n), self.coord(k / n)];
            for p in 0..3 {
                b[p] += w[k] * f[p] * self.opd_waves[k];
                for q in 0..3 {
                    g[p][q] += w[k] * f[p] * f[q];
                }
            }
        }
        let coef = if remove_tilt {
            solve3(g, b).unwrap_or([b[0] / g[0][0], 0.0, 0.0])
        } else {
            [b[0] / g[0][0], 0.0, 0.0]
        };
        let (mut s2, mut sw) = (mklight_core::math::KahanSum::default(), 0.0);
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for k in 0..n * n {
            if w[k] <= 0.0 {
                continue;
            }
            let r = self.opd_waves[k]
                - coef[0]
                - coef[1] * self.coord(k % n)
                - coef[2] * self.coord(k / n);
            s2.add(w[k] * r * r);
            sw += w[k];
            lo = lo.min(r);
            hi = hi.max(r);
        }
        OpdStats {
            mean: coef[0],
            tilt: [coef[1], coef[2]],
            rms: if sw > 0.0 {
                (s2.value() / sw).sqrt()
            } else {
                0.0
            },
            pv: if hi >= lo { hi - lo } else { 0.0 },
        }
    }

    /// Strehl ratio at the reference point, exact for the sampled pupil:
    /// `|Σ p|² / (Σ |p|)²`.
    pub fn strehl_at_reference(&self) -> f64 {
        let f = self.debye_field();
        let (mut re, mut im, mut a) = (0.0, 0.0, 0.0);
        for z in &f {
            re += z.re;
            im += z.im;
            a += z.norm();
        }
        if a == 0.0 {
            0.0
        } else {
            (re * re + im * im) / (a * a)
        }
    }

    /// Pupil energy `E = Σ c·(A/γ)²` with partially covered cells counted
    /// by area.
    ///
    /// This is the energy of the *hard-edged* pupil the coverage describes.
    /// The lattice sum `Σ|p|² = Σ c²·(A/γ)²` is smaller by `Σ c(1−c)(A/γ)²`:
    /// grey edge cells suppress the PSF's far wings (beyond the alias-free
    /// field, where the hard edge would alias anyway) but leave the core
    /// intact. Normalising energies and the OTF by `E` rather than by
    /// `Σ|p|²` removes the resulting `O(1/n)` bias; the lattice OTF value at
    /// zero lag (`Σ|p|²/E`) is the only one replaced (by its exact value 1).
    pub fn energy(&self) -> f64 {
        let n = self.n;
        (0..n * n)
            .filter(|&k| self.coverage[k] > 0.0)
            .map(|k| {
                let a = self.amplitude[k] * self.debye_factor(self.coord(k % n), self.coord(k / n));
                self.coverage[k] * a * a
            })
            .sum()
    }

    /// Peak of the aberration-free PSF of this aperture when the PSF is
    /// normalised to unit total energy (mm⁻²): `(Σ|p|)² / (L²·E)`, with
    /// `L` the alias-free field and `E` [`Self::energy`]. For a clear
    /// circular pupil this is `π/(4λ²F#²)`.
    pub fn diffraction_peak_unit_energy(&self) -> f64 {
        let s1: f64 = self.field_weights().iter().sum();
        let l = self.alias_free_field_mm();
        s1 * s1 / (l * l * self.energy())
    }
}

/// Weighted wavefront statistics (waves).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OpdStats {
    pub mean: f64,
    /// Removed tilt coefficients `(t_u, t_v)` in waves per unit `u`, `v`.
    pub tilt: [f64; 2],
    pub rms: f64,
    pub pv: f64,
}

fn solve3(a: [[f64; 3]; 3], b: [f64; 3]) -> Option<[f64; 3]> {
    let det = |m: &[[f64; 3]; 3]| {
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    };
    let d = det(&a);
    if d.abs() < 1e-300 {
        return None;
    }
    let mut x = [0.0; 3];
    for (c, xc) in x.iter_mut().enumerate() {
        let mut m = a;
        for r in 0..3 {
            m[r][c] = b[r];
        }
        *xc = det(&m) / d;
    }
    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom() -> PupilGeometry {
        PupilGeometry::from_working_f_number(5.0, 50.0, 1.0).unwrap()
    }

    #[test]
    fn geometry_relations() {
        let g = geom();
        assert!((g.working_f_number() - 5.0).abs() < 1e-14);
        assert!((g.numerical_aperture() - 0.1).abs() < 1e-15);
        assert!((g.lambda_f_mm(0.5) - 0.0025).abs() < 1e-15);
        assert!((g.cutoff_frequency(0.5) - 400.0).abs() < 1e-9);
        assert!(PupilGeometry::new(10.0, 5.0, 1.0).is_err());
    }

    #[test]
    fn cell_centred_grid_and_coverage() {
        let p =
            PupilFunction::diffraction_limited(64, 0.55, geom(), Aperture::Circular, 16).unwrap();
        assert!((p.coord(0) + 1.0 - 1.0 / 64.0).abs() < 1e-15);
        assert!((p.coord(63) - 1.0 + 1.0 / 64.0).abs() < 1e-15);
        // Area of the unit disk from the coverage: π within the
        // supersampling resolution.
        let area: f64 = p.coverage().iter().sum::<f64>() * p.spacing().powi(2);
        assert!((area - std::f64::consts::PI).abs() < 2e-3, "{area}");
        assert_eq!(p.samples_across(), 64);
        assert!(p.sampling_warnings().is_empty());
    }

    #[test]
    fn opd_stats_of_tilt_and_defocus() {
        // Pure tilt: removed exactly with remove_tilt.
        let p = PupilFunction::from_aperture(
            64,
            0.55,
            geom(),
            Aperture::Circular,
            Apodization::Uniform,
            1,
            |u, v| 0.3 * u - 0.1 * v + 0.05,
        )
        .unwrap();
        let s = p.opd_stats(true);
        assert!(
            s.rms < 1e-12 && (s.tilt[0] - 0.3).abs() < 1e-12 && (s.tilt[1] + 0.1).abs() < 1e-12
        );
        // Defocus ρ² on a uniform disk: σ = 1/√12.
        let p = PupilFunction::from_aperture(
            256,
            0.55,
            geom(),
            Aperture::Circular,
            Apodization::Uniform,
            1,
            |u, v| u * u + v * v,
        )
        .unwrap();
        // NA 0.1 → Debye weights vary by 0.5 %, which shifts σ slightly.
        assert!((p.opd_stats(false).rms - 1.0 / 12f64.sqrt()).abs() < 2e-3);
    }

    #[test]
    fn focus_shift_matches_paraxial_defocus() {
        let p = PupilFunction::diffraction_limited(32, 0.5, geom(), Aperture::Circular, 1).unwrap();
        let dz = 0.01;
        // W020 = dz·NA²/(2n'λ) = 0.01 · 0.01 / (2 · 0.0005) = 0.1 wave.
        let w = p.focus_shift_opd(dz, 1.0, 0.0);
        assert!((w - 0.1).abs() < 0.1 * 0.01, "{w}");
        assert!(p.focus_shift_opd(dz, 0.0, 0.0).abs() < 1e-15);
        assert!(p.focus_shift_opd(-dz, 1.0, 0.0) < 0.0);
    }

    #[test]
    fn phase_step_warning() {
        let p = PupilFunction::from_aperture(
            64,
            0.55,
            geom(),
            Aperture::Circular,
            Apodization::Uniform,
            1,
            |u, v| 20.0 * (u * u + v * v),
        )
        .unwrap();
        // d(20ρ²)/dρ · δ = 40 · (2/64) ≈ 1.2 waves at the edge.
        assert!(
            p.sampling_warnings()
                .iter()
                .any(|w| matches!(w, WaveWarning::PupilUndersampled { .. }))
        );
        let p =
            PupilFunction::diffraction_limited(16, 0.55, geom(), Aperture::Circular, 1).unwrap();
        assert!(
            p.sampling_warnings()
                .iter()
                .any(|w| matches!(w, WaveWarning::FewPupilSamples { .. }))
        );
    }
}
