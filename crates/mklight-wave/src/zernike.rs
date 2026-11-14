//! Zernike polynomials on the unit disk (CONVENTIONS §4).
//!
//! * Coordinates: `x = ρ cos θ`, `y = ρ sin θ` in normalised exit-pupil
//!   coordinates; θ is measured from +x toward +y.
//! * A term is `(n, m)`: `m > 0` multiplies `cos(mθ)`, `m < 0` multiplies
//!   `sin(|m|θ)`, `m = 0` is rotationally symmetric.
//! * **Fringe** (University of Arizona) ordering: terms are un-normalised
//!   (`R_n^m(1) = 1`), grouped by `k = (n + |m|)/2`, inside a group `|m|`
//!   decreases with cos before sin and the `m = 0` term last; Z37 is the
//!   12th-order spherical term. Only Z1–Z37 exist.
//! * **Standard** ordering is Noll's: `j` increases with `n`, then `|m|`; even
//!   `j` carry cos, odd `j` sin; terms are normalised to unit RMS over the
//!   disk (`√(n+1)` for `m = 0`, `√(2(n+1))` otherwise), so the wavefront
//!   RMS is the root-sum-square of the coefficients excluding piston.
//! * Coefficients are in waves.
//!
//! Radial polynomials are evaluated as `R_n^m(ρ) = ρ^m · P_k^{(0,m)}(2ρ² − 1)`,
//! `k = (n − m)/2`, with the three-term Jacobi recurrence, which is stable on
//! [−1, 1] even for high orders (the explicit factorial sum is not). The
//! angular factor `ρ^m e^{imθ}` is `(x + iy)^m`, so no `atan2` is needed and the
//! origin is regular.

use mklight_core::{JobContext, MkError, MkResult};
use serde::{Deserialize, Serialize};

use crate::warnings::WaveWarning;
use mklight_core::linalg::lstsq_equilibrated;

/// Index ordering and normalisation of a coefficient vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ZernikeOrdering {
    /// Fringe (University of Arizona), un-normalised, Z1–Z37.
    #[default]
    Fringe,
    /// Noll-ordered, RMS-normalised ("standard").
    Standard,
}

/// One Zernike term `(n, m)`; see the module docs for the sign of `m`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ZernikeTerm {
    pub n: u32,
    pub m: i32,
}

/// Number of terms in the Fringe set.
pub const FRINGE_TERMS: usize = 37;

impl ZernikeTerm {
    pub fn new(n: u32, m: i32) -> MkResult<Self> {
        let am = m.unsigned_abs();
        if am > n || !(n - am).is_multiple_of(2) {
            return Err(MkError::InvalidParam(format!("Zernike term ({n}, {m})")));
        }
        Ok(ZernikeTerm { n, m })
    }

    /// Term for 1-based Fringe index `j` (1…37).
    pub fn fringe(j: usize) -> Option<Self> {
        match j {
            0 => None,
            37 => Some(ZernikeTerm { n: 12, m: 0 }),
            38.. => None,
            _ => {
                let k = ((j - 1) as f64).sqrt().floor() as usize;
                let off = j - 1 - k * k;
                let (am, sin) = if off == 2 * k {
                    (0, false)
                } else {
                    (k - off / 2, off % 2 == 1)
                };
                let n = (2 * k - am) as u32;
                let m = if sin { -(am as i32) } else { am as i32 };
                Some(ZernikeTerm { n, m })
            }
        }
    }

    /// 1-based Fringe index, if the term belongs to the Fringe set.
    pub fn fringe_index(&self) -> Option<usize> {
        let am = self.m.unsigned_abs() as usize;
        let k = (self.n as usize + am) / 2;
        if self.n == 12 && self.m == 0 {
            return Some(37);
        }
        if k > 5 {
            return None;
        }
        let base = k * k + 1;
        Some(if am == 0 {
            base + 2 * k
        } else {
            base + 2 * (k - am) + usize::from(self.m < 0)
        })
    }

    /// Term for 1-based Noll index `j`.
    pub fn noll(j: usize) -> Option<Self> {
        if j == 0 {
            return None;
        }
        let mut n = (((8 * j - 7) as f64).sqrt() as usize - 1) / 2;
        // Guard against rounding in the square root.
        while (n + 1) * (n + 2) / 2 < j {
            n += 1;
        }
        while n > 0 && n * (n + 1) / 2 >= j {
            n -= 1;
        }
        let k = j - n * (n + 1) / 2 - 1;
        let am = if n.is_multiple_of(2) {
            2 * k.div_ceil(2)
        } else {
            2 * (k / 2) + 1
        };
        let m = if am == 0 {
            0
        } else if j.is_multiple_of(2) {
            am as i32
        } else {
            -(am as i32)
        };
        Some(ZernikeTerm { n: n as u32, m })
    }

    /// 1-based Noll index.
    pub fn noll_index(&self) -> usize {
        let n = self.n as usize;
        let am = self.m.unsigned_abs() as usize;
        let row = n * (n + 1) / 2;
        if am == 0 {
            return row + 1;
        }
        let j0 = row + am; // k0 = |m| − 1 ⇒ j0 = row + k0 + 1
        let want_even = self.m > 0;
        if j0.is_multiple_of(2) == want_even {
            j0
        } else {
            j0 + 1
        }
    }

    /// RMS normalisation factor of the Standard term relative to the
    /// un-normalised one.
    pub fn noll_norm(&self) -> f64 {
        let n1 = (self.n + 1) as f64;
        if self.m == 0 {
            n1.sqrt()
        } else {
            (2.0 * n1).sqrt()
        }
    }

    /// Un-normalised radial polynomial `R_n^{|m|}(ρ)`.
    pub fn radial(&self, rho: f64) -> f64 {
        let m = self.m.unsigned_abs();
        let k = (self.n - m) / 2;
        let p = jacobi_all(k as usize, m, 2.0 * rho * rho - 1.0);
        rho.powi(m as i32) * p[k as usize]
    }

    /// Un-normalised term value at `(x, y)`.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let m = self.m.unsigned_abs();
        let k = (self.n - m) / 2;
        let rho2 = x * x + y * y;
        let p = jacobi_all(k as usize, m, 2.0 * rho2 - 1.0);
        let (mut re, mut im) = (1.0, 0.0);
        for _ in 0..m {
            (re, im) = (re * x - im * y, re * y + im * x);
        }
        let ang = if self.m >= 0 { re } else { im };
        ang * p[k as usize]
    }
}

/// `P_0^{(0,m)}(t) … P_kmax^{(0,m)}(t)` by the three-term recurrence.
fn jacobi_all(kmax: usize, m: u32, t: f64) -> Vec<f64> {
    let mut p = Vec::with_capacity(kmax + 1);
    jacobi_into(kmax, m, t, &mut p);
    p
}

fn jacobi_into(kmax: usize, m: u32, t: f64, p: &mut Vec<f64>) {
    p.clear();
    let b = m as f64;
    p.push(1.0);
    if kmax == 0 {
        return;
    }
    p.push(1.0 + (b + 2.0) * (t - 1.0) / 2.0);
    for k in 1..kmax {
        let kf = k as f64;
        let s = 2.0 * kf + b; // 2k + α + β with α = 0
        let a1 = 2.0 * (kf + 1.0) * (kf + b + 1.0) * s;
        let a2 = (s + 1.0) * ((s + 2.0) * s * t - b * b);
        let a3 = 2.0 * kf * (kf + b) * (s + 2.0);
        let next = (a2 * p[k] - a3 * p[k - 1]) / a1;
        p.push(next);
    }
}

/// An ordered list of Zernike terms with fast evaluation of all of them at
/// once.
#[derive(Clone, Debug, PartialEq)]
pub struct ZernikeBasis {
    ordering: ZernikeOrdering,
    terms: Vec<ZernikeTerm>,
    scale: Vec<f64>,
    max_m: u32,
    /// For each |m|: largest k = (n − |m|)/2 used.
    kmax: Vec<Option<usize>>,
}

impl ZernikeBasis {
    /// The first `count` terms of `ordering` (Fringe allows at most 37).
    pub fn new(ordering: ZernikeOrdering, count: usize) -> MkResult<Self> {
        let terms: Option<Vec<ZernikeTerm>> = (1..=count)
            .map(|j| match ordering {
                ZernikeOrdering::Fringe => ZernikeTerm::fringe(j),
                ZernikeOrdering::Standard => ZernikeTerm::noll(j),
            })
            .collect();
        let terms = terms.ok_or_else(|| {
            MkError::InvalidParam(format!(
                "the Fringe set has {FRINGE_TERMS} terms; {count} requested"
            ))
        })?;
        let scale = terms
            .iter()
            .map(|t| match ordering {
                ZernikeOrdering::Fringe => 1.0,
                ZernikeOrdering::Standard => t.noll_norm(),
            })
            .collect();
        let max_m = terms.iter().map(|t| t.m.unsigned_abs()).max().unwrap_or(0);
        let mut kmax = vec![None; max_m as usize + 1];
        for t in &terms {
            let am = t.m.unsigned_abs() as usize;
            let k = (t.n as usize - am) / 2;
            kmax[am] = Some(kmax[am].map_or(k, |v: usize| v.max(k)));
        }
        Ok(ZernikeBasis {
            ordering,
            terms,
            scale,
            max_m,
            kmax,
        })
    }

    /// All Standard (Noll) terms with radial order `n ≤ n_max`.
    pub fn standard_to_order(n_max: u32) -> Self {
        let n = n_max as usize;
        Self::new(ZernikeOrdering::Standard, (n + 1) * (n + 2) / 2).expect("standard basis")
    }

    pub fn ordering(&self) -> ZernikeOrdering {
        self.ordering
    }
    pub fn terms(&self) -> &[ZernikeTerm] {
        &self.terms
    }
    pub fn len(&self) -> usize {
        self.terms.len()
    }
    pub fn is_empty(&self) -> bool {
        self.terms.is_empty()
    }

    /// Writes every basis function at `(x, y)` into `out` (length `len()`).
    pub fn eval_into(&self, x: f64, y: f64, out: &mut [f64]) {
        let t = 2.0 * (x * x + y * y) - 1.0;
        // (x + iy)^m for m = 0..=max_m.
        let mut pw = Vec::with_capacity(self.max_m as usize + 1);
        let (mut re, mut im) = (1.0, 0.0);
        pw.push((re, im));
        for _ in 0..self.max_m {
            (re, im) = (re * x - im * y, re * y + im * x);
            pw.push((re, im));
        }
        let mut jac: Vec<Vec<f64>> = Vec::with_capacity(self.kmax.len());
        for (m, km) in self.kmax.iter().enumerate() {
            let mut v = Vec::new();
            if let Some(k) = km {
                jacobi_into(*k, m as u32, t, &mut v);
            }
            jac.push(v);
        }
        for ((o, term), s) in out.iter_mut().zip(&self.terms).zip(&self.scale) {
            let am = term.m.unsigned_abs() as usize;
            let k = (term.n as usize - am) / 2;
            let ang = if term.m >= 0 { pw[am].0 } else { pw[am].1 };
            *o = s * ang * jac[am][k];
        }
    }

    /// `Σ c_j Z_j(x, y)`.
    pub fn eval(&self, x: f64, y: f64, coefficients: &[f64]) -> f64 {
        let mut z = vec![0.0; self.len()];
        self.eval_into(x, y, &mut z);
        z.iter().zip(coefficients).map(|(a, b)| a * b).sum()
    }
}

/// A wavefront as Zernike coefficients (waves). `coefficients()[j − 1]` is
/// the coefficient of term `j` in `ordering()`.
///
/// The fields are private so every set is valid (at most 37 Fringe terms,
/// finite coefficients); deserialisation runs the same checks as
/// [`ZernikeSet::new`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "ZernikeSetData")]
pub struct ZernikeSet {
    ordering: ZernikeOrdering,
    coefficients: Vec<f64>,
}

#[derive(Deserialize)]
struct ZernikeSetData {
    ordering: ZernikeOrdering,
    coefficients: Vec<f64>,
}

impl TryFrom<ZernikeSetData> for ZernikeSet {
    type Error = MkError;
    fn try_from(d: ZernikeSetData) -> MkResult<Self> {
        ZernikeSet::new(d.ordering, d.coefficients)
    }
}

/// Seidel-equivalent third-order aberrations (waves at the pupil edge)
/// extracted from Fringe Z1–Z9, following the usual relations
/// (e.g. Wyant & Creath, *Applied Optics and Optical Engineering* XI, 1992):
///
/// | term | magnitude | angle |
/// |---|---|---|
/// | tilt | `√((Z2 − 2Z7)² + (Z3 − 2Z8)²)` | `atan2(Z3 − 2Z8, Z2 − 2Z7)` |
/// | focus | `2Z4 − 6Z9 − √(Z5² + Z6²)` | |
/// | astigmatism | `2√(Z5² + Z6²)` | `½·atan2(Z6, Z5)` |
/// | coma | `3√(Z7² + Z8²)` | `atan2(Z8, Z7)` |
/// | spherical | `6Z9` | |
///
/// so that, up to piston,
/// `W = tilt·ρcos(θ−θt) + focus·ρ² + astig·ρ²cos²(θ−θa) + coma·ρ³cos(θ−θc) + spherical·ρ⁴`.
/// The astigmatism sign is fixed positive; its `ρ²` part is moved into
/// focus. Angles in degrees, measured from +x toward +y.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SeidelTerms {
    pub piston: f64,
    pub tilt: f64,
    pub tilt_angle_deg: f64,
    pub focus: f64,
    pub astigmatism: f64,
    pub astigmatism_angle_deg: f64,
    pub coma: f64,
    pub coma_angle_deg: f64,
    pub spherical: f64,
}

impl SeidelTerms {
    /// Wavefront value (waves) of these terms at `(x, y)`.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        let r2 = x * x + y * y;
        let r = r2.sqrt();
        let th = y.atan2(x);
        let (ta, aa, ca) = (
            self.tilt_angle_deg.to_radians(),
            self.astigmatism_angle_deg.to_radians(),
            self.coma_angle_deg.to_radians(),
        );
        self.piston
            + self.tilt * r * (th - ta).cos()
            + self.focus * r2
            + self.astigmatism * r2 * (th - aa).cos().powi(2)
            + self.coma * r2 * r * (th - ca).cos()
            + self.spherical * r2 * r2
    }
}

impl ZernikeSet {
    pub fn new(ordering: ZernikeOrdering, coefficients: Vec<f64>) -> MkResult<Self> {
        if ordering == ZernikeOrdering::Fringe && coefficients.len() > FRINGE_TERMS {
            return Err(MkError::InvalidParam(format!(
                "the Fringe set has {FRINGE_TERMS} terms; {} given",
                coefficients.len()
            )));
        }
        if let Some(c) = coefficients.iter().find(|c| !c.is_finite()) {
            return Err(MkError::InvalidParam(format!(
                "non-finite Zernike coefficient {c}"
            )));
        }
        Ok(ZernikeSet {
            ordering,
            coefficients,
        })
    }

    pub fn ordering(&self) -> ZernikeOrdering {
        self.ordering
    }

    /// Coefficients (waves); index `j − 1` holds term `j`.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    pub fn basis(&self) -> ZernikeBasis {
        // Infallible: `new` (the only constructor) bounds Fringe sets to 37.
        ZernikeBasis::new(self.ordering, self.coefficients.len()).expect("valid set")
    }

    pub fn term(&self, index: usize) -> Option<ZernikeTerm> {
        match self.ordering {
            ZernikeOrdering::Fringe => ZernikeTerm::fringe(index + 1),
            ZernikeOrdering::Standard => ZernikeTerm::noll(index + 1),
        }
    }

    /// Wavefront (waves) at `(x, y)`.
    pub fn eval(&self, x: f64, y: f64) -> f64 {
        self.basis().eval(x, y, &self.coefficients)
    }

    /// `(term, RMS-normalised coefficient)` pairs.
    fn normalized(&self) -> impl Iterator<Item = (ZernikeTerm, f64)> + '_ {
        self.coefficients.iter().enumerate().map(|(i, &c)| {
            let t = self.term(i).expect("valid set");
            let a = match self.ordering {
                ZernikeOrdering::Fringe => c / t.noll_norm(),
                ZernikeOrdering::Standard => c,
            };
            (t, a)
        })
    }

    /// RMS over the full unit disk (waves), piston removed; with
    /// `remove_tilt` the n = 1 terms are excluded as well. Exact, from the
    /// orthogonality of the terms.
    pub fn rms(&self, remove_tilt: bool) -> f64 {
        self.normalized()
            .filter(|(t, _)| t.n > 1 || (t.n == 1 && !remove_tilt))
            .map(|(_, a)| a * a)
            .sum::<f64>()
            .sqrt()
    }

    /// Peak-to-valley over the unit disk (waves), from a polar sampling
    /// with `rings` radial and `8·rings` azimuthal samples (plus the
    /// centre).
    pub fn pv(&self, rings: usize) -> f64 {
        let basis = self.basis();
        let mut z = vec![0.0; basis.len()];
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        let mut visit = |x: f64, y: f64| {
            basis.eval_into(x, y, &mut z);
            let w: f64 = z.iter().zip(&self.coefficients).map(|(a, b)| a * b).sum();
            lo = lo.min(w);
            hi = hi.max(w);
        };
        visit(0.0, 0.0);
        let rings = rings.max(1);
        for i in 1..=rings {
            let r = i as f64 / rings as f64;
            let na = 8 * rings.max(i);
            for a in 0..na {
                let th = 2.0 * std::f64::consts::PI * a as f64 / na as f64;
                visit(r * th.cos(), r * th.sin());
            }
        }
        hi - lo
    }

    /// Re-express in another ordering. Converting to Fringe fails if a
    /// non-zero term has no Fringe index.
    pub fn to_ordering(&self, target: ZernikeOrdering) -> MkResult<ZernikeSet> {
        if target == self.ordering {
            return Ok(self.clone());
        }
        let mut out: Vec<f64> = Vec::new();
        for (t, a) in self.normalized() {
            let (idx, c) = match target {
                ZernikeOrdering::Standard => (t.noll_index(), a),
                ZernikeOrdering::Fringe => match t.fringe_index() {
                    Some(j) => (j, a * t.noll_norm()),
                    None if a == 0.0 => continue,
                    None => {
                        return Err(MkError::Unsupported(format!(
                            "term (n={}, m={}) has no Fringe index",
                            t.n, t.m
                        )));
                    }
                },
            };
            if out.len() < idx {
                out.resize(idx, 0.0);
            }
            out[idx - 1] = c;
        }
        ZernikeSet::new(target, out)
    }

    /// Seidel-equivalent terms from the Fringe Z1–Z9 content of this set.
    /// Only the terms with radial order `n ≤ 4` that are Fringe Z1–Z9 are
    /// read, so any Standard set (of any order) works; higher-order terms
    /// are ignored.
    pub fn seidel(&self) -> MkResult<SeidelTerms> {
        let mut f = [0.0; 9];
        for (t, a) in self.normalized() {
            if let Some(j @ 1..=9) = t.fringe_index() {
                f[j - 1] = a * t.noll_norm();
            }
        }
        let z = |j: usize| f[j - 1];
        let tx = z(2) - 2.0 * z(7);
        let ty = z(3) - 2.0 * z(8);
        let a = z(5).hypot(z(6));
        Ok(SeidelTerms {
            piston: z(1) - z(4) + z(9),
            tilt: tx.hypot(ty),
            tilt_angle_deg: ty.atan2(tx).to_degrees(),
            focus: 2.0 * z(4) - 6.0 * z(9) - a,
            astigmatism: 2.0 * a,
            astigmatism_angle_deg: 0.5 * z(6).atan2(z(5)).to_degrees(),
            coma: 3.0 * z(7).hypot(z(8)),
            coma_angle_deg: z(8).atan2(z(7)).to_degrees(),
            spherical: 6.0 * z(9),
        })
    }
}

/// Settings for a least-squares Zernike fit.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZernikeFitSettings {
    /// Condition number of the equilibrated design matrix above which the
    /// fit is reported as ill-conditioned (a warning, not an error).
    pub warn_condition: f64,
}

impl Default for ZernikeFitSettings {
    fn default() -> Self {
        ZernikeFitSettings {
            warn_condition: 1e6,
        }
    }
}

/// Result of [`fit_zernike`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ZernikeFit {
    pub set: ZernikeSet,
    /// Weighted RMS of the fit residual (waves).
    pub residual_rms: f64,
    /// 2-norm condition number of the column-equilibrated design matrix.
    pub condition: f64,
    pub warnings: Vec<WaveWarning>,
}

/// Weighted least-squares fit of `values` (waves) at scattered normalised
/// pupil points `(x, y)` to `basis`. Weights default to 1. Fails when the
/// problem is rank deficient (too few or degenerate points); warns when the
/// condition number exceeds `settings.warn_condition`.
pub fn fit_zernike(
    points: &[(f64, f64)],
    values: &[f64],
    weights: Option<&[f64]>,
    basis: &ZernikeBasis,
    settings: &ZernikeFitSettings,
    ctx: &JobContext,
) -> MkResult<ZernikeFit> {
    if points.len() != values.len() || weights.is_some_and(|w| w.len() != points.len()) {
        return Err(MkError::InvalidParam(
            "fit_zernike: points, values and weights differ in length".into(),
        ));
    }
    let k = basis.len();
    let mut cols = vec![Vec::with_capacity(points.len()); k];
    let mut b = Vec::with_capacity(points.len());
    let mut z = vec![0.0; k];
    let mut wsum = 0.0;
    for (i, (&(x, y), &v)) in points.iter().zip(values).enumerate() {
        let w = weights.map_or(1.0, |w| w[i]);
        if !(w > 0.0) || !v.is_finite() {
            continue;
        }
        let sw = w.sqrt();
        wsum += w;
        basis.eval_into(x, y, &mut z);
        for (c, zj) in cols.iter_mut().zip(&z) {
            c.push(sw * zj);
        }
        b.push(sw * v);
    }
    ctx.check()?;
    let sol = lstsq_equilibrated(cols, b, ctx)?;
    let mut warnings = Vec::new();
    if sol.condition > settings.warn_condition {
        warnings.push(WaveWarning::IllConditioned {
            condition: sol.condition,
            limit: settings.warn_condition,
        });
    }
    Ok(ZernikeFit {
        set: ZernikeSet::new(basis.ordering(), sol.x)?,
        residual_rms: if wsum > 0.0 {
            sol.residual_norm / wsum.sqrt()
        } else {
            0.0
        },
        condition: sol.condition,
        warnings,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::special::gauss_legendre;
    use mklight_core::rng::Philox;

    fn factorial(n: u32) -> f64 {
        (1..=n).map(|k| k as f64).product()
    }

    /// Explicit factorial form, fine for low orders.
    fn radial_explicit(n: u32, m: u32, rho: f64) -> f64 {
        let mut s = 0.0;
        for k in 0..=(n - m) / 2 {
            let num = if k % 2 == 0 { 1.0 } else { -1.0 } * factorial(n - k);
            let den = factorial(k) * factorial((n + m) / 2 - k) * factorial((n - m) / 2 - k);
            s += num / den * rho.powi((n - 2 * k) as i32);
        }
        s
    }

    /// Core's Zernike sag shares this module's ordering, normalisation and
    /// angle convention without depending on it.
    #[test]
    fn core_zernike_sag_matches_wave_basis() {
        use mklight_core::model::surface::ZernikeOrdering as SagOrdering;
        for (ordering, sag_ordering, count) in [
            (ZernikeOrdering::Fringe, SagOrdering::Fringe, 37),
            (ZernikeOrdering::Standard, SagOrdering::Standard, 45),
        ] {
            let basis = ZernikeBasis::new(ordering, count).unwrap();
            for j in 0..count {
                let mut coefficients = vec![0.0; count];
                coefficients[j] = 1.0;
                let shape = mklight_core::SurfaceShape::ZernikeSag {
                    curvature: 0.0,
                    conic: 0.0,
                    norm_radius: 2.0,
                    ordering: sag_ordering,
                    coefficients: coefficients.clone(),
                };
                for (x, y) in [(0.0, 0.0), (0.3, -0.5), (-0.7, 0.2), (0.6, 0.8)] {
                    let sag = mklight_core::surface_math::evaluate(&shape, 2.0 * x, 2.0 * y)
                        .unwrap()
                        .sag;
                    let wave = basis.eval(x, y, &coefficients);
                    assert!((sag - wave).abs() < 1e-12, "{ordering:?} j={}", j + 1);
                }
            }
        }
    }

    #[test]
    fn fringe_table() {
        let expect = [
            (1, 0, 0),
            (2, 1, 1),
            (3, 1, -1),
            (4, 2, 0),
            (5, 2, 2),
            (6, 2, -2),
            (7, 3, 1),
            (8, 3, -1),
            (9, 4, 0),
            (10, 3, 3),
            (11, 3, -3),
            (16, 6, 0),
            (17, 4, 4),
            (25, 8, 0),
            (26, 5, 5),
            (36, 10, 0),
            (37, 12, 0),
        ];
        for (j, n, m) in expect {
            let t = ZernikeTerm::fringe(j).unwrap();
            assert_eq!((t.n, t.m), (n, m), "Fringe Z{j}");
            assert_eq!(t.fringe_index(), Some(j));
        }
        assert!(ZernikeTerm::fringe(38).is_none());
        for j in 1..=37 {
            assert_eq!(ZernikeTerm::fringe(j).unwrap().fringe_index(), Some(j));
        }
    }

    #[test]
    fn noll_table_and_roundtrip() {
        let expect = [
            (1, 0, 0),
            (2, 1, 1),
            (3, 1, -1),
            (4, 2, 0),
            (5, 2, -2),
            (6, 2, 2),
            (7, 3, -1),
            (8, 3, 1),
            (9, 3, -3),
            (10, 3, 3),
            (11, 4, 0),
            (12, 4, 2),
            (13, 4, -2),
            (22, 6, 0),
        ];
        for (j, n, m) in expect {
            let t = ZernikeTerm::noll(j).unwrap();
            assert_eq!((t.n, t.m), (n, m), "Noll Z{j}");
        }
        for j in 1..2000 {
            assert_eq!(ZernikeTerm::noll(j).unwrap().noll_index(), j);
        }
    }

    #[test]
    fn radial_recurrence_matches_explicit_and_is_bounded() {
        for n in 0..=16u32 {
            for m in (n % 2..=n).step_by(2) {
                let t = ZernikeTerm::new(n, m as i32).unwrap();
                for i in 0..=50 {
                    let r = i as f64 / 50.0;
                    assert!((t.radial(r) - radial_explicit(n, m, r)).abs() < 1e-10);
                }
            }
        }
        // High order: |R| ≤ 1 and R(1) = 1, where the explicit sum would
        // lose all digits to cancellation.
        for (n, m) in [(60u32, 0u32), (61, 1), (80, 20)] {
            let t = ZernikeTerm::new(n, m as i32).unwrap();
            assert!((t.radial(1.0) - 1.0).abs() < 1e-11);
            for i in 0..=400 {
                assert!(t.radial(i as f64 / 400.0).abs() <= 1.0 + 1e-11);
            }
        }
    }

    /// ⟨Z_i Z_j⟩ over the unit disk = δ_ij for Standard terms, by Gauss–
    /// Legendre in ρ² and an exact uniform rule in θ.
    #[test]
    fn standard_terms_are_orthonormal() {
        let basis = ZernikeBasis::standard_to_order(10); // 66 terms
        let k = basis.len();
        let (xg, wg) = gauss_legendre(16);
        let na = 48;
        let mut gram = vec![0.0; k * k];
        let mut z = vec![0.0; k];
        for (xi, wi) in xg.iter().zip(&wg) {
            // s = ρ² ∈ [0, 1]; dA = ½ ds dθ.
            let s = 0.5 * (xi + 1.0);
            let rho = s.sqrt();
            for a in 0..na {
                let th = 2.0 * std::f64::consts::PI * a as f64 / na as f64;
                basis.eval_into(rho * th.cos(), rho * th.sin(), &mut z);
                let w = wi * 0.5 * 0.5 * (2.0 * std::f64::consts::PI / na as f64)
                    / std::f64::consts::PI;
                for p in 0..k {
                    for q in 0..k {
                        gram[p * k + q] += w * z[p] * z[q];
                    }
                }
            }
        }
        for p in 0..k {
            for q in 0..k {
                let e = if p == q { 1.0 } else { 0.0 };
                assert!((gram[p * k + q] - e).abs() < 1e-12, "({p},{q})");
            }
        }
    }

    #[test]
    fn exact_fit_of_synthetic_wavefront() {
        let mut rng = Philox::new(7, 1);
        let truth: Vec<f64> = (0..37).map(|_| rng.uniform() - 0.5).collect();
        let set = ZernikeSet::new(ZernikeOrdering::Fringe, truth.clone()).unwrap();
        let pts: Vec<(f64, f64)> = (0..2000)
            .map(|_| {
                let r = rng.uniform().sqrt();
                let t = 2.0 * std::f64::consts::PI * rng.uniform();
                (r * t.cos(), r * t.sin())
            })
            .collect();
        let vals: Vec<f64> = pts.iter().map(|&(x, y)| set.eval(x, y)).collect();
        let basis = ZernikeBasis::new(ZernikeOrdering::Fringe, 37).unwrap();
        let fit = fit_zernike(
            &pts,
            &vals,
            None,
            &basis,
            &Default::default(),
            &JobContext::new(),
        )
        .unwrap();
        for (a, b) in fit.set.coefficients.iter().zip(&truth) {
            assert!((a - b).abs() < 1e-10);
        }
        assert!(fit.residual_rms < 1e-11);
        assert!(fit.condition < 100.0);
        assert!(fit.warnings.is_empty());
    }

    #[test]
    fn fit_detects_degenerate_sampling() {
        let basis = ZernikeBasis::new(ZernikeOrdering::Fringe, 9).unwrap();
        let ctx = JobContext::new();
        // All points on the x axis: sin terms vanish → rank deficient.
        let pts: Vec<(f64, f64)> = (0..50)
            .map(|i| (i as f64 / 49.0 * 2.0 - 1.0, 0.0))
            .collect();
        let vals = vec![0.0; pts.len()];
        assert!(fit_zernike(&pts, &vals, None, &basis, &Default::default(), &ctx).is_err());
        // Too few points.
        assert!(
            fit_zernike(
                &pts[..5],
                &vals[..5],
                None,
                &basis,
                &Default::default(),
                &ctx
            )
            .is_err()
        );
        // Points crowded near the centre: valid but badly conditioned.
        let mut rng = Philox::new(3, 0);
        let pts: Vec<(f64, f64)> = (0..400)
            .map(|_| (0.02 * (rng.uniform() - 0.5), 0.02 * (rng.uniform() - 0.5)))
            .collect();
        let vals = vec![0.0; pts.len()];
        let fit = fit_zernike(&pts, &vals, None, &basis, &Default::default(), &ctx).unwrap();
        assert!(fit.condition > 1e6);
        assert!(matches!(
            fit.warnings[0],
            WaveWarning::IllConditioned { .. }
        ));
    }

    #[test]
    fn ordering_conversion_and_rms() {
        let mut fr = vec![0.0; 37];
        fr[3] = 0.4; // Z4 defocus
        fr[8] = -0.2; // Z9 spherical
        fr[6] = 0.1; // Z7 coma x
        let f = ZernikeSet::new(ZernikeOrdering::Fringe, fr).unwrap();
        let s = f.to_ordering(ZernikeOrdering::Standard).unwrap();
        assert!((s.coefficients[3] - 0.4 / 3f64.sqrt()).abs() < 1e-15); // Noll 4
        assert!((s.coefficients[10] + 0.2 / 5f64.sqrt()).abs() < 1e-15); // Noll 11
        assert!((s.coefficients[7] - 0.1 / 8f64.sqrt()).abs() < 1e-15); // Noll 8
        let back = s.to_ordering(ZernikeOrdering::Fringe).unwrap();
        for (a, b) in back.coefficients.iter().zip(&f.coefficients) {
            assert!((a - b).abs() < 1e-15);
        }
        // Same wavefront in both orderings.
        for &(x, y) in &[(0.3, -0.2), (0.0, 0.9), (-0.7, 0.1)] {
            assert!((f.eval(x, y) - s.eval(x, y)).abs() < 1e-13);
        }
        // RMS: defocus a·(2ρ²−1) has RMS a/√3.
        let d = ZernikeSet::new(ZernikeOrdering::Fringe, vec![0.0, 0.0, 0.0, 0.3]).unwrap();
        assert!((d.rms(false) - 0.3 / 3f64.sqrt()).abs() < 1e-15);
        assert!((d.pv(64) - 0.6).abs() < 1e-12);
        // A Noll term without a Fringe index cannot be converted.
        let mut hi = vec![0.0; 45];
        hi[43] = 1.0; // Noll 44 = (8, 8)
        let h = ZernikeSet::new(ZernikeOrdering::Standard, hi).unwrap();
        assert!(h.to_ordering(ZernikeOrdering::Fringe).is_err());
    }

    #[test]
    fn seidel_extraction_reproduces_wavefront() {
        let truth = SeidelTerms {
            piston: 0.0,
            tilt: 0.3,
            tilt_angle_deg: 20.0,
            focus: -0.15,
            astigmatism: 0.25,
            astigmatism_angle_deg: 35.0,
            coma: 0.4,
            coma_angle_deg: -60.0,
            spherical: 0.12,
        };
        let mut rng = Philox::new(11, 0);
        let pts: Vec<(f64, f64)> = (0..500)
            .map(|_| {
                let r = rng.uniform().sqrt();
                let t = 2.0 * std::f64::consts::PI * rng.uniform();
                (r * t.cos(), r * t.sin())
            })
            .collect();
        let vals: Vec<f64> = pts.iter().map(|&(x, y)| truth.eval(x, y)).collect();
        let basis = ZernikeBasis::new(ZernikeOrdering::Fringe, 9).unwrap();
        let fit = fit_zernike(
            &pts,
            &vals,
            None,
            &basis,
            &Default::default(),
            &JobContext::new(),
        )
        .unwrap();
        let s = fit.set.seidel().unwrap();
        assert!((s.tilt - truth.tilt).abs() < 1e-12);
        assert!((s.tilt_angle_deg - truth.tilt_angle_deg).abs() < 1e-9);
        assert!((s.focus - truth.focus).abs() < 1e-12);
        assert!((s.astigmatism - truth.astigmatism).abs() < 1e-12);
        assert!((s.astigmatism_angle_deg - truth.astigmatism_angle_deg).abs() < 1e-9);
        assert!((s.coma - truth.coma).abs() < 1e-12);
        assert!((s.coma_angle_deg - truth.coma_angle_deg).abs() < 1e-9);
        assert!((s.spherical - truth.spherical).abs() < 1e-12);
        // And the extracted terms reproduce the wavefront up to piston.
        let p0 = s.eval(0.0, 0.0) - truth.eval(0.0, 0.0);
        for &(x, y) in pts.iter().take(50) {
            assert!((s.eval(x, y) - truth.eval(x, y) - p0).abs() < 1e-11);
        }
    }
}
