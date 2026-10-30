//! Optical and modulation transfer functions.
//!
//! The OTF is the normalised Fourier transform of the intensity PSF,
//! `OTF(ν) = ∫ PSF(x)·e^{−2πiν·x} dA / ∫ PSF dA`, equivalently the
//! normalised autocorrelation of the Debye-weighted pupil field,
//! `OTF(ν) = Σ_u p(u + s)·p*(u) / Σ|p|²` with the pupil shift
//! `s = 2ν·λF#` (normalised units; `s = 2` at the cutoff `ν_c = 1/(λF#)`).
//! MTF = |OTF|, 1 at zero frequency. Frequencies are in cycles/mm on the
//! image surface.
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
//!   `Δν = 1/(n·λF#)`; other frequencies are bilinearly interpolated.
//! * [`mtf_direct`]: the autocorrelation evaluated directly for each
//!   requested frequency, shifting the pupil by `s` with bilinear
//!   interpolation of amplitude and OPD (no frequency lattice).
//! * [`mtf_from_psf`]: a direct DFT of a sampled PSF (any PSF: Huygens,
//!   polychromatic, tilted planes).

use mklight_core::{JobContext, MkError, MkResult};
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
    /// Incoherent cutoff `1/(λF#)` (largest over wavelengths), cycles/mm;
    /// NaN when not known ([`mtf_from_psf`]).
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

/// Full 2-D OTF on the frequency lattice of the pupil sampling.
#[derive(Clone, Debug, PartialEq)]
pub struct Otf {
    /// Grid side `M ≥ 2n`; bin `(i, j)` is frequency
    /// `((i − M/2)·Δν, (j − M/2)·Δν)`.
    pub size: usize,
    /// `Δν = 1/(n·λF#)` cycles/mm.
    pub spacing: f64,
    /// Normalised OTF (`1` at zero frequency), row-major, centred.
    pub data: Vec<C64>,
    /// Image offset of the PSF reference point (mm); applied as the phase
    /// `e^{−2πiν·offset}` by [`Otf::at`].
    pub offset_mm: [f64; 2],
    /// Cutoff `1/(λF#)`.
    pub cutoff: f64,
}

impl Otf {
    /// Bilinear interpolation of the OTF at `(νx, νy)`; zero beyond the grid.
    pub fn at(&self, nx: f64, ny: f64) -> C64 {
        let c = (self.size / 2) as f64;
        let fx = nx / self.spacing + c;
        let fy = ny / self.spacing + c;
        let (ix, iy) = (fx.floor(), fy.floor());
        let (tx, ty) = (fx - ix, fy - iy);
        let m = self.size as i64;
        let get = |i: i64, j: i64| -> C64 {
            if i < 0 || j < 0 || i >= m || j >= m {
                C64::default()
            } else {
                self.data[(j * m + i) as usize]
            }
        };
        let (i, j) = (ix as i64, iy as i64);
        let v = get(i, j) * ((1.0 - tx) * (1.0 - ty))
            + get(i + 1, j) * (tx * (1.0 - ty))
            + get(i, j + 1) * ((1.0 - tx) * ty)
            + get(i + 1, j + 1) * (tx * ty);
        let ph = -2.0 * std::f64::consts::PI * (nx * self.offset_mm[0] + ny * self.offset_mm[1]);
        v * C64::from_polar(1.0, ph)
    }
}

/// 2-D OTF by FFT (pupil → PSF → OTF), exact on its frequency lattice.
pub fn otf_fft(pupil: &PupilFunction, ctx: &JobContext) -> MkResult<Otf> {
    let n = pupil.n();
    let m = next_fast_len(2 * n);
    let field = pupil.debye_field();
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
    let o0 = (m * m) as f64 * pupil.energy();
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
    Ok(Otf {
        size: m,
        spacing: 1.0 / pupil.alias_free_field_mm(),
        data,
        offset_mm: pupil.image_offset_mm(),
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

/// Complex OTF of `pupil` at frequency vectors `nus` (cycles/mm), including
/// the image-offset phase.
pub fn otf_at(
    pupil: &PupilFunction,
    nus: &[[f64; 2]],
    method: MtfMethod,
    ctx: &JobContext,
) -> MkResult<Vec<C64>> {
    match method {
        MtfMethod::Fft => {
            let otf = otf_fft(pupil, ctx)?;
            Ok(nus.iter().map(|v| otf.at(v[0], v[1])).collect())
        }
        MtfMethod::Direct => otf_direct(pupil, nus, ctx),
    }
}

/// Direct pupil autocorrelation `Σ_u p(u + s)·p*(u) / E` for each
/// frequency, `s = ν/(ν-per-unit-u)`. The shifted pupil is interpolated
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
    let off = pupil.image_offset_mm();
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
            let ph = -tau * (nu[0] * off[0] + nu[1] * off[1]);
            Ok(C64::new(re.value(), im.value()) / norm * C64::from_polar(1.0, ph))
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
        let o = otf_at(p, &nus, method, &sub.subrange(0.0, 0.5))?;
        let mut clean = p.without_aberrations();
        clean.set_image_offset_mm([0.0; 2]);
        let d = otf_at(&clean, &nus, method, &sub.subrange(0.5, 1.0))?;
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
