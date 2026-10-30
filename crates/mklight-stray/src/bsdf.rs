//! Surface scatter (BSDF) for every [`ScatterModel`].
//!
//! # Conventions
//!
//! All models are written in **direction-cosine space** (Harvey–Shack). For
//! a unit direction `k` leaving a surface with outward unit normal `n` and
//! tangent basis `(t₁, t₂)`, `β = (k·t₁, k·t₂)` lies in the unit disk. The
//! projected solid angle is `cos θ dω = dβ₁ dβ₂`, so a BSDF `f` in sr⁻¹ (per
//! unit projected solid angle, as `ScatterModel` specifies) gives the
//! scattered fraction from an area integral over the disk:
//!
//! ```text
//! TIS(β₀) = ∫_{|β|<1} f(β; β₀) d²β
//! ```
//!
//! `β₀` is the specular direction's projection. Lambertian scatter is flat,
//! `f = tis/π`. ABg and the Gaussian microfacet model are shift-invariant
//! lobes `f = F(|β − β₀|)` centred on the specular direction:
//!
//! * ABg: `F(ρ) = A / (B + ρ^g)`.
//! * Gaussian: `F(ρ) = K·exp(−ρ²/(2s²))` with `s = 2·σ_slope`. A facet tilted
//!   by `α` deviates the reflected beam by `2α`, so `|Δβ| ≈ 2α` for small
//!   slopes. `K` makes the integral over the unit disk at normal incidence
//!   equal `tis`. **Convention**: `σ_slope` is the RMS slope *per axis* (the
//!   profile RMS slope a profilometer measures), so each component of `Δβ`
//!   has standard deviation `2σ_slope`. If a value is the total 2-D RMS slope
//!   `m = √(⟨s_x² + s_y²⟩)` (the microfacet "RMS slope"), pass `m/√2`. The
//!   shared model (`ScatterModel::Gaussian`) does not yet state either
//!   convention; this one is the stray solver's until it does.
//!
//! The part of a lobe outside the unit disk is evanescent and is not
//! scattered, so `TIS` falls toward grazing incidence. The same lobe is used
//! about the reflected and about the refracted specular direction (the
//! customary Harvey-space approximation for BTDFs).
//!
//! # Energy conservation
//!
//! `TIS(β₀) ≤ 1` for every incidence. Lambertian and Gaussian `tis` are
//! clamped to `[0, 1]`. ABg is multiplied by a constant `scale ≤ 1` so that
//! the largest TIS is at most 1. For `g > 0` the lobe is radially decreasing,
//! so by Anderson's inequality (the integral of a symmetric unimodal function
//! over a disk is largest when the disk is centred on the peak) the maximum
//! is at normal incidence and `scale = min(1, 1/TIS(0))` is exact. For `g ≤ 0`
//! the maximum of the TIS table is used. A constant scale keeps the lobe
//! shape and reciprocity.
//!
//! The path tracer splits flux into specular `(1 − TIS)` and scattered `TIS`
//! parts with the *tabulated* TIS ([`Bsdf::tis`]), then samples the scattered
//! direction exactly ∝ `f` ([`Bsdf::sample`]). Each event therefore conserves
//! energy exactly. The only approximation is the TIS table's interpolation
//! error (relative ≲ 1e-4 for lobes wider than ~1e-3 in β, worse only for
//! extremely narrow lobes near grazing incidence).
//!
//! # Reciprocity
//!
//! The BRDFs depend only on `|β − β₀|`. With `β₀ = β_in` (tangential
//! components of the incident propagation direction are preserved on
//! reflection), swapping and reversing the directions maps `β_in → −β_out` and
//! `β_out → −β_in`, so `f(in → out) = f(−out → −in)` holds exactly (tested).
//! The BTDF approximation does not satisfy the generalised `n²` reciprocity
//! of transmission.
//!
//! # Sampling
//!
//! Sampling is exact (the returned `β` is distributed ∝ `f` on the unit disk):
//! Lambertian uses a uniform disk sample. The Gaussian uses an inverted,
//! truncated Rayleigh radius about `β₀`. ABg uses rejection from a piecewise
//! power-law envelope `h(ρ) = min(1/B, ρ^−g)`, with acceptance
//! `1/((B + ρ^g)·h) ∈ [½, 1]`. Proposals outside the unit disk are rejected. The
//! exact pdf per projected solid angle is `f/TIS(β₀)`.

use std::f64::consts::{FRAC_PI_2, PI};

use mklight_core::error::{MkError, MkResult};
use mklight_core::math::Vec3;
use mklight_core::model::ScatterModel;
use mklight_core::rng::Philox;

use crate::integrate::{gauss_kronrod, gk15, periodic_trapezoid};

/// Number of TIS table nodes, uniform in incidence angle on `[0, π/2]`.
pub const TIS_TABLE_NODES: usize = 257;
/// Rejection-sampling attempts before [`Bsdf::sample`] gives up.
pub const MAX_REJECTIONS: u32 = 4096;

/// Radial lobe profile and its cumulative radial integral.
#[derive(Clone, Debug)]
enum Lobe {
    /// `f = value` everywhere on the disk (not centred on β₀).
    Lambertian { value: f64 },
    /// `F(ρ) = a / (b + ρ^g)`.
    Abg {
        a: f64,
        b: f64,
        g: f64,
        radial: RadialTable,
    },
    /// `F(ρ) = k·exp(−ρ²/(2s²))`.
    Gaussian { k: f64, s: f64 },
}

/// Cumulative `G(ρ) = ∫₀^ρ r/(b + r^g) dr` for ABg (unit `a`). Nodes are
/// geometric from `ρ₀ = 10⁻⁴·ρ_c` (`ρ_c = b^{1/g}`, where `r^g` reaches `b`)
/// to 2, so each panel is smooth and one GK15 panel is accurate to rounding.
#[derive(Clone, Debug)]
struct RadialTable {
    b: f64,
    g: f64,
    nodes: Vec<f64>,
    cum: Vec<f64>,
}

impl RadialTable {
    fn integrand(&self, r: f64) -> f64 {
        r / (self.b + r.powf(self.g))
    }
    fn new(b: f64, g: f64) -> Self {
        let mut t = RadialTable {
            b,
            g,
            nodes: Vec::new(),
            cum: Vec::new(),
        };
        if b == 0.0 {
            return t; // closed form
        }
        let rho0 = if g > 0.0 {
            let ln_rc = b.ln() / g;
            (ln_rc + (1e-4f64).ln()).exp().clamp(1e-200, 1e-3)
        } else {
            1e-12
        };
        let mut r = rho0;
        let mut c = t.head(rho0);
        t.nodes.push(r);
        t.cum.push(c);
        while r < 2.0 {
            let next = (r * 1.25).min(2.0);
            let q = gauss_kronrod(|x| t.integrand(x), r, next, 0.0, 1e-15, 64);
            c += q.value;
            r = next;
            t.nodes.push(r);
            t.cum.push(c);
        }
        t
    }
    /// Integral over `[0, ρ]` for `ρ` below the first node, where `b`
    /// dominates: the series `ρ²/(2b)·(1 − 2x/(g+2) + x²/(g+1) − …)`, with
    /// `x = ρ^g/b ≤ 10⁻⁴ᵍ`.
    fn head(&self, rho: f64) -> f64 {
        let g = self.g;
        let x = rho.powf(g) / self.b;
        if g > 0.0 && x < 0.5 {
            rho * rho / (2.0 * self.b) * (1.0 - 2.0 * x / (g + 2.0) + x * x / (g + 1.0))
        } else {
            rho * rho / (2.0 * (self.b + rho.powf(g)))
        }
    }
    fn eval(&self, rho: f64) -> f64 {
        if rho <= 0.0 {
            return 0.0;
        }
        if self.b == 0.0 {
            let e = 2.0 - self.g;
            return rho.powf(e) / e;
        }
        if rho <= self.nodes[0] {
            return self.head(rho);
        }
        let rho = rho.min(2.0);
        let k = match self.nodes.binary_search_by(|x| x.total_cmp(&rho)) {
            Ok(k) => return self.cum[k],
            Err(k) => k - 1,
        };
        self.cum[k] + gk15(&|x| self.integrand(x), self.nodes[k], rho).0
    }
}

/// Distance from `β₀` (magnitude `b`, along +x) to the unit circle in the
/// direction at angle `φ`, without cancellation near grazing incidence.
#[inline]
fn rho_edge(b: f64, phi: f64) -> f64 {
    let (s, c) = phi.sin_cos();
    let root = ((1.0 - b * s) * (1.0 + b * s)).max(0.0).sqrt();
    if c > 0.0 {
        ((1.0 - b) * (1.0 + b)).max(0.0) / (root + b * c)
    } else {
        root - b * c
    }
}

/// Scaled lobe parameters ([`Bsdf::lobe_params`]).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LobeParams {
    Lambertian { value: f64 },
    Abg { a: f64, b: f64, g: f64 },
    Gaussian { k: f64, s: f64 },
}

/// A ready-to-use BSDF for one [`ScatterModel`] (validated, normalised, with
/// its TIS table).
#[derive(Clone, Debug)]
pub struct Bsdf {
    model: ScatterModel,
    lobe: Lobe,
    scale: f64,
    /// TIS at incidence angles `θ_j = j·(π/2)/(N−1)`; empty when constant.
    tis_table: Vec<f64>,
    tis_const: f64,
    clamped: bool,
}

impl Bsdf {
    /// Validates the model, normalises it so that `TIS ≤ 1`, and builds the
    /// TIS table.
    pub fn new(model: &ScatterModel) -> MkResult<Bsdf> {
        let bad = |m: &str| Err(MkError::InvalidModel(format!("scatter model: {m}")));
        let mut clamped = false;
        let mut clamp_tis = |tis: f64| -> MkResult<f64> {
            if !(tis.is_finite() && tis >= 0.0) {
                return Err(MkError::InvalidModel(
                    "scatter model: tis must be finite and ≥ 0".into(),
                ));
            }
            if tis > 1.0 {
                clamped = true;
            }
            Ok(tis.min(1.0))
        };
        let (lobe, scale) = match *model {
            ScatterModel::Lambertian { tis } => {
                let tis = clamp_tis(tis)?;
                (Lobe::Lambertian { value: tis / PI }, 1.0)
            }
            ScatterModel::Gaussian { tis, sigma_slope } => {
                let tis = clamp_tis(tis)?;
                if !(sigma_slope.is_finite() && sigma_slope > 0.0) {
                    return bad("sigma_slope must be finite and > 0");
                }
                let s = 2.0 * sigma_slope;
                // ∫_disk exp(−ρ²/2s²) d²β = 2π s² (1 − exp(−1/(2s²))).
                let norm = 2.0 * PI * s * s * (-(-0.5 / (s * s)).exp_m1());
                (Lobe::Gaussian { k: tis / norm, s }, 1.0)
            }
            ScatterModel::Abg { a, b, g } => {
                if !(a.is_finite() && b.is_finite() && g.is_finite()) {
                    return bad("A, B and g must be finite");
                }
                if a < 0.0 || b < 0.0 {
                    return bad("A and B must be ≥ 0");
                }
                if b == 0.0 && g >= 2.0 {
                    return bad("B = 0 needs g < 2 (otherwise the TIS diverges)");
                }
                (
                    Lobe::Abg {
                        a,
                        b,
                        g,
                        radial: RadialTable::new(b, g),
                    },
                    1.0,
                )
            }
        };
        let mut out = Bsdf {
            model: model.clone(),
            lobe,
            scale,
            tis_table: Vec::new(),
            tis_const: 0.0,
            clamped,
        };
        match out.lobe {
            Lobe::Lambertian { value } => out.tis_const = value * PI,
            _ => {
                out.tis_table = (0..TIS_TABLE_NODES)
                    .map(|j| {
                        let th = FRAC_PI_2 * j as f64 / (TIS_TABLE_NODES - 1) as f64;
                        out.tis_exact(th.sin())
                    })
                    .collect();
                let max = out.tis_table.iter().copied().fold(0.0, f64::max);
                let max = match out.lobe {
                    Lobe::Abg { g, .. } if g > 0.0 => out.tis_table[0],
                    _ => max,
                };
                if max > 1.0 {
                    out.clamped = true;
                    out.scale = 1.0 / max;
                    for t in &mut out.tis_table {
                        *t = (*t * out.scale).min(1.0);
                    }
                }
            }
        }
        Ok(out)
    }

    pub fn model(&self) -> &ScatterModel {
        &self.model
    }
    /// Constant factor applied to the model's lobe for energy conservation
    /// (`1` when the model already had `TIS ≤ 1`).
    pub fn scale(&self) -> f64 {
        self.scale
    }
    /// True when the model's parameters were clamped or rescaled.
    pub fn was_normalised(&self) -> bool {
        self.clamped
    }

    /// The TIS table a GPU backend packs: `TIS_TABLE_NODES` values at
    /// incidence angles `θ_j = j·(π/2)/(N−1)`, interpolated as in
    /// [`Bsdf::tis`]. Empty when the TIS does not depend on incidence (then
    /// [`Bsdf::tis`] returns the constant).
    pub fn tis_table(&self) -> &[f64] {
        &self.tis_table
    }

    /// Lobe parameters for a GPU backend, with the energy scale applied:
    /// `Lambertian { value }` (`f = value`), `Abg { a, b, g }`
    /// (`f = a/(b + ρ^g)`) or `Gaussian { k, s }` (`f = k·exp(−ρ²/2s²)`),
    /// where `ρ = |β − β₀|`.
    pub fn lobe_params(&self) -> LobeParams {
        match self.lobe {
            Lobe::Lambertian { value } => LobeParams::Lambertian {
                value: self.scale * value,
            },
            Lobe::Abg { a, b, g, .. } => LobeParams::Abg {
                a: self.scale * a,
                b,
                g,
            },
            Lobe::Gaussian { k, s } => LobeParams::Gaussian {
                k: self.scale * k,
                s,
            },
        }
    }

    /// Unscaled radial profile `F(ρ)`.
    fn profile(&self, rho: f64) -> f64 {
        match self.lobe {
            Lobe::Lambertian { value } => value,
            Lobe::Abg { a, b, g, .. } => a / (b + rho.powf(g)),
            Lobe::Gaussian { k, s } => k * (-0.5 * (rho / s) * (rho / s)).exp(),
        }
    }

    /// Unscaled cumulative `∫₀^ρ F(r) r dr` for lobes centred on β₀.
    fn cumulative(&self, rho: f64) -> f64 {
        match &self.lobe {
            Lobe::Lambertian { value } => value * rho * rho / 2.0,
            Lobe::Abg { a, radial, .. } => a * radial.eval(rho),
            Lobe::Gaussian { k, s } => k * s * s * (-(-0.5 * (rho / s) * (rho / s)).exp_m1()),
        }
    }

    /// BSDF value (sr⁻¹ per projected solid angle) at direction cosines `beta`
    /// for specular direction cosines `beta0`. Zero outside the unit disk.
    pub fn eval(&self, beta0: [f64; 2], beta: [f64; 2]) -> f64 {
        if beta[0] * beta[0] + beta[1] * beta[1] > 1.0 {
            return 0.0;
        }
        let rho = (beta[0] - beta0[0]).hypot(beta[1] - beta0[1]);
        self.scale * self.profile(rho)
    }

    /// BRDF for directions: `k_in` is the incident propagation direction,
    /// `k_out` the scattered one, and `n` the unit normal on the reflected side
    /// (`k_in·n < 0`, `k_out·n > 0`).
    pub fn eval_brdf(&self, k_in: Vec3, k_out: Vec3, n: Vec3) -> f64 {
        let (t1, t2) = crate::geom::onb(n);
        // The specular direction's tangential part equals the incident one's.
        let b0 = [k_in.dot(t1), k_in.dot(t2)];
        self.eval(b0, [k_out.dot(t1), k_out.dot(t2)])
    }

    /// Tabulated TIS for a specular direction with `|β₀| = beta0_mag`: cubic
    /// (Catmull–Rom) interpolation in incidence angle, clamped to `[0, 1]`.
    /// This is the value the tracer uses.
    pub fn tis(&self, beta0_mag: f64) -> f64 {
        let t = &self.tis_table;
        if t.is_empty() {
            return self.tis_const;
        }
        let n = TIS_TABLE_NODES;
        let th = beta0_mag.clamp(0.0, 1.0).asin();
        let x = th / FRAC_PI_2 * (n - 1) as f64;
        let j = (x.floor() as usize).min(n - 2);
        let f = x - j as f64;
        let (p1, p2) = (t[j], t[j + 1]);
        // TIS is even in θ, so the node below 0 mirrors node 1.
        let p0 = if j == 0 { t[1] } else { t[j - 1] };
        let p3 = if j + 2 < n { t[j + 2] } else { 2.0 * p2 - p1 };
        let (m1, m2) = (0.5 * (p2 - p0), 0.5 * (p3 - p1));
        let (f2, f3) = (f * f, f * f * f);
        let v = (2.0 * f3 - 3.0 * f2 + 1.0) * p1
            + (f3 - 2.0 * f2 + f) * m1
            + (-2.0 * f3 + 3.0 * f2) * p2
            + (f3 - f2) * m2;
        v.clamp(0.0, 1.0)
    }

    /// TIS by adaptive quadrature: `∫_0^{2π} G(ρ_edge(φ)) dφ`, with the radial
    /// integral `G` exact (closed form or tabulated Gauss–Kronrod) and a
    /// doubling periodic trapezoid rule over `φ`.
    pub fn tis_exact(&self, beta0_mag: f64) -> f64 {
        if let Lobe::Lambertian { value } = self.lobe {
            return value * PI;
        }
        let b = beta0_mag.clamp(0.0, 1.0);
        let q = periodic_trapezoid(
            |phi| self.cumulative(rho_edge(b, phi)),
            2.0 * PI,
            1e-12,
            1e-300,
            18,
        );
        self.scale * q.value
    }

    /// Largest tabulated TIS over all incidence angles (≤ 1).
    pub fn max_tis(&self) -> f64 {
        if self.tis_table.is_empty() {
            self.tis_const
        } else {
            self.tis_table.iter().copied().fold(0.0, f64::max)
        }
    }

    /// Sampling pdf per projected solid angle, using the tabulated TIS.
    pub fn pdf(&self, beta0: [f64; 2], beta: [f64; 2]) -> f64 {
        let t = self.tis(beta0[0].hypot(beta0[1]));
        if t > 0.0 {
            self.eval(beta0, beta) / t
        } else {
            0.0
        }
    }
    /// Exact sampling pdf, using the adaptive TIS.
    pub fn pdf_exact(&self, beta0: [f64; 2], beta: [f64; 2]) -> f64 {
        let t = self.tis_exact(beta0[0].hypot(beta0[1]));
        if t > 0.0 {
            self.eval(beta0, beta) / t
        } else {
            0.0
        }
    }

    /// Probability mass of the polar cell `ρ ∈ [ρ1, ρ2], φ ∈ [φ1, φ2]` around
    /// `β₀` (with `φ` measured from `β₀`'s direction), clipped to the unit disk.
    /// Used by sampling tests and by detector-side importance estimates.
    pub fn cell_mass(&self, beta0_mag: f64, rho: [f64; 2], phi: [f64; 2]) -> f64 {
        let b = beta0_mag.clamp(0.0, 1.0);
        let t = self.tis_exact(b);
        if t <= 0.0 {
            return 0.0;
        }
        if let Lobe::Lambertian { value } = self.lobe {
            // Flat lobe: area of the clipped cell times the value.
            let q = gauss_kronrod(
                |p| {
                    let e = rho_edge(b, p);
                    let (r1, r2) = (rho[0].min(e), rho[1].min(e));
                    0.5 * (r2 * r2 - r1 * r1)
                },
                phi[0],
                phi[1],
                1e-14,
                1e-11,
                2000,
            );
            return value * q.value / t;
        }
        let q = gauss_kronrod(
            |p| {
                let e = rho_edge(b, p);
                self.cumulative(rho[1].min(e)) - self.cumulative(rho[0].min(e))
            },
            phi[0],
            phi[1],
            1e-14,
            1e-11,
            2000,
        );
        self.scale * q.value / t
    }

    /// Draws scattered direction cosines distributed exactly ∝ `f(·; β₀)` on the
    /// unit disk. Returns `None` only if [`MAX_REJECTIONS`] proposals were all
    /// rejected (the caller books that flux as a sampling failure).
    pub fn sample(&self, beta0: [f64; 2], rng: &mut Philox) -> Option<[f64; 2]> {
        let b = beta0[0].hypot(beta0[1]);
        let rho_max = 1.0 + b;
        let around = |rho: f64, u: f64| {
            let (s, c) = (2.0 * PI * u).sin_cos();
            [beta0[0] + rho * c, beta0[1] + rho * s]
        };
        let in_disk = |p: [f64; 2]| p[0] * p[0] + p[1] * p[1] < 1.0;
        match self.lobe {
            Lobe::Lambertian { .. } => {
                let r = rng.uniform().sqrt();
                let (s, c) = (2.0 * PI * rng.uniform()).sin_cos();
                Some([r * c, r * s])
            }
            Lobe::Gaussian { s, .. } => {
                let m = -(-0.5 * (rho_max / s) * (rho_max / s)).exp_m1();
                for _ in 0..MAX_REJECTIONS {
                    let u = rng.uniform();
                    let rho = s * (-2.0 * (-u * m).ln_1p()).sqrt();
                    let p = around(rho, rng.uniform());
                    if in_disk(p) {
                        return Some(p);
                    }
                }
                None
            }
            Lobe::Abg { b: bb, g, .. } => {
                for _ in 0..MAX_REJECTIONS {
                    let (rho, accept) = abg_proposal(bb, g, rho_max, rng);
                    let p = around(rho, rng.uniform());
                    if rng.uniform() < accept && in_disk(p) {
                        return Some(p);
                    }
                }
                None
            }
        }
    }
}

/// One radius from the ABg envelope on `[0, ρ_max]` and its acceptance
/// probability `F(ρ)/(a·h(ρ))`.
fn abg_proposal(b: f64, g: f64, rho_max: f64, rng: &mut Philox) -> (f64, f64) {
    let u0 = rng.uniform();
    if g.abs() < 1e-12 {
        // Constant lobe: uniform disk of radius ρ_max.
        return (rho_max * u0.sqrt(), 1.0);
    }
    if g < 0.0 {
        // Increasing lobe: uniform proposal, bounded by its value at ρ_max.
        let rho = rho_max * u0.sqrt();
        return (rho, (b + rho_max.powf(g)) / (b + rho.powf(g)));
    }
    let e = 2.0 - g;
    // Breakpoint ρ_c where ρ^g = b (0 when b = 0).
    let rc = if b == 0.0 { 0.0 } else { (b.ln() / g).exp() };
    if rc >= rho_max {
        // Region 1 only: q ∝ ρ/b.
        let rho = rho_max * u0.sqrt();
        return (rho, b / (b + rho.powf(g)));
    }
    let l = if rc > 0.0 {
        (rho_max / rc).ln()
    } else {
        f64::INFINITY
    };
    // Masses relative to rc^e: region 1 → ½, region 2 → (e^{eL} − 1)/e.
    let p1 = if rc == 0.0 {
        0.0
    } else {
        let w2 = if e.abs() < 1e-9 {
            l
        } else {
            (e * l).exp_m1() / e
        };
        if w2.is_finite() {
            0.5 / (0.5 + w2)
        } else {
            0.0
        }
    };
    if u0 < p1 {
        let rho = rc * (u0 / p1).sqrt();
        (rho, b / (b + rho.powf(g)))
    } else {
        let u = if p1 > 0.0 { (u0 - p1) / (1.0 - p1) } else { u0 };
        let ln_rho = if e > 1e-9 {
            // ρ^e = rc^e + u (ρmax^e − rc^e), written relative to ρmax.
            let ratio = if rc > 0.0 { (-e * l).exp() } else { 0.0 };
            rho_max.ln() + (u + (1.0 - u) * ratio).ln() / e
        } else if e < -1e-9 {
            rc.ln() + (u * (e * l).exp_m1()).ln_1p() / e
        } else {
            rc.ln() + u * l
        };
        let rho = ln_rho.exp();
        let rg = rho.powf(g);
        (rho, rg / (b + rg))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lam(t: f64) -> Bsdf {
        Bsdf::new(&ScatterModel::Lambertian { tis: t }).unwrap()
    }
    fn abg(a: f64, b: f64, g: f64) -> Bsdf {
        Bsdf::new(&ScatterModel::Abg { a, b, g }).unwrap()
    }
    fn gauss(t: f64, s: f64) -> Bsdf {
        Bsdf::new(&ScatterModel::Gaussian {
            tis: t,
            sigma_slope: s,
        })
        .unwrap()
    }

    /// Independent 2-D integral of `eval` over the unit disk in polar
    /// coordinates about the *origin* (nested adaptive GK).
    fn brute_tis(f: &Bsdf, b0: [f64; 2]) -> f64 {
        gauss_kronrod(
            |r| {
                r * gauss_kronrod(
                    |p| f.eval(b0, [r * p.cos(), r * p.sin()]),
                    0.0,
                    2.0 * PI,
                    1e-13,
                    1e-10,
                    400,
                )
                .value
            },
            0.0,
            1.0,
            1e-12,
            1e-9,
            400,
        )
        .value
    }

    /// The exported lobe parameters (for GPU backends) reproduce `eval`,
    /// including the energy scale of an over-unity ABg model.
    #[test]
    fn lobe_params_reproduce_eval() {
        let (b0, beta): ([f64; 2], [f64; 2]) = ([0.1, -0.2], [0.25, 0.05]);
        let rho = (beta[0] - b0[0]).hypot(beta[1] - b0[1]);
        for f in [lam(0.3), abg(1.0, 1e-3, 2.0), gauss(0.02, 0.01)] {
            let v = match f.lobe_params() {
                LobeParams::Lambertian { value } => value,
                LobeParams::Abg { a, b, g } => a / (b + rho.powf(g)),
                LobeParams::Gaussian { k, s } => k * (-0.5 * (rho / s).powi(2)).exp(),
            };
            assert!((v / f.eval(b0, beta) - 1.0).abs() < 1e-14);
        }
        assert!(abg(1.0, 1e-3, 2.0).was_normalised());
        assert_eq!(lam(0.3).tis_table().len(), 0);
        assert_eq!(gauss(0.02, 0.01).tis_table().len(), TIS_TABLE_NODES);
    }

    #[test]
    fn lambertian_tis_equals_parameter() {
        let f = lam(0.037);
        for b in [0.0, 0.3, 0.9, 1.0] {
            assert_eq!(f.tis(b), 0.037);
            assert!((f.tis_exact(b) - 0.037).abs() < 1e-15);
        }
        assert!((brute_tis(&f, [0.4, 0.2]) - 0.037).abs() < 1e-10);
        // Clamped above 1.
        let f = lam(1.7);
        assert!(f.was_normalised() && f.tis(0.2) == 1.0);
    }

    #[test]
    fn abg_g2_closed_form() {
        // g = 2, normal incidence: TIS = π A ln(1 + 1/B).
        let (a, b) = (1e-3, 1e-4);
        let f = abg(a, b, 2.0);
        let exact = PI * a * (1.0 / b).ln_1p();
        assert!((f.tis_exact(0.0) / exact - 1.0).abs() < 1e-10);
        assert!((f.tis(0.0) / exact - 1.0).abs() < 1e-10);
        // Radial table against the closed form ½ ln(1 + ρ²/B).
        if let Lobe::Abg { radial, .. } = &f.lobe {
            for rho in [1e-5, 3e-3, 0.1, 0.77, 1.9] {
                let g = 0.5 * (rho * rho / b).ln_1p();
                assert!((radial.eval(rho) / g - 1.0).abs() < 1e-11, "{rho}");
            }
        }
    }

    #[test]
    fn abg_general_g_matches_brute_force() {
        let f = abg(2e-3, 1e-2, 1.5);
        for b0 in [[0.0f64, 0.0], [0.5, 0.1], [0.0, -0.95]] {
            let exact = f.tis_exact(b0[0].hypot(b0[1]));
            let brute = brute_tis(&f, b0);
            assert!(
                (exact / brute - 1.0).abs() < 1e-6,
                "{b0:?}: {exact} {brute}"
            );
            let tab = f.tis(b0[0].hypot(b0[1]));
            assert!((tab / exact - 1.0).abs() < 1e-4, "{tab} {exact}");
        }
        // Anderson: TIS is largest at normal incidence and falls to grazing.
        assert!(f.tis(0.0) > f.tis(0.5) && f.tis(0.5) > f.tis(0.99));
    }

    #[test]
    fn abg_b_zero_closed_form() {
        // B = 0, g = 1: F = A/ρ, G = ρ: TIS(0) = 2π A.
        let f = abg(0.01, 0.0, 1.0);
        assert!((f.tis_exact(0.0) - 2.0 * PI * 0.01).abs() < 1e-14);
        assert!(
            Bsdf::new(&ScatterModel::Abg {
                a: 1.0,
                b: 0.0,
                g: 2.0
            })
            .is_err()
        );
    }

    #[test]
    fn energy_normalisation_caps_tis() {
        let f = abg(10.0, 1e-4, 2.0);
        assert!(f.was_normalised() && f.scale() < 1.0);
        assert!((f.tis_exact(0.0) - 1.0).abs() < 1e-9);
        for j in 0..=20 {
            let b = j as f64 / 20.0;
            assert!(f.tis(b) <= 1.0 + 1e-12 && f.tis_exact(b) <= 1.0 + 1e-9);
        }
    }

    #[test]
    fn gaussian_tis_at_normal_incidence() {
        for s in [0.002, 0.05, 0.3] {
            let f = gauss(0.02, s);
            assert!((f.tis_exact(0.0) - 0.02).abs() < 1e-13, "{s}");
            // Never above the normal-incidence value; strictly below once the
            // lobe reaches the edge of the disk.
            assert!(f.tis(0.95) <= 0.02 * (1.0 + 1e-9));
            if s >= 0.05 {
                assert!(f.tis(0.95) < 0.019);
            }
        }
        let f = gauss(0.02, 0.1);
        assert!((brute_tis(&f, [0.3, -0.4]) / f.tis_exact(0.5) - 1.0).abs() < 1e-7);
    }

    #[test]
    fn brdf_reciprocity() {
        let n = Vec3::new(0.1, -0.2, 1.0).normalized();
        let k_in = Vec3::new(0.3, 0.2, -1.0).normalized();
        let k_out = Vec3::new(-0.5, 0.4, 0.7).normalized();
        for f in [lam(0.1), abg(1e-3, 1e-3, 1.8), gauss(0.05, 0.2)] {
            let a = f.eval_brdf(k_in, k_out, n);
            let b = f.eval_brdf(-k_out, -k_in, n);
            assert!(a > 0.0 && (a - b).abs() <= 1e-14 * a, "{a} {b}");
        }
    }

    /// Chi-square test of `sample` against `cell_mass` on a polar grid about β₀.
    fn chi_square(f: &Bsdf, b0: [f64; 2], rho_edges: &[f64], n: usize, seed: u64) {
        let b = b0[0].hypot(b0[1]);
        let dir = b0[1].atan2(b0[0]);
        let n_phi = 8;
        let mut counts = vec![0u64; rho_edges.len() * n_phi];
        let mut rng = Philox::new(seed, 0);
        for _ in 0..n {
            let p = f.sample(b0, &mut rng).expect("sampling failed");
            assert!(p[0] * p[0] + p[1] * p[1] < 1.0);
            let (dx, dy) = (p[0] - b0[0], p[1] - b0[1]);
            let rho = dx.hypot(dy);
            let phi = (dy.atan2(dx) - dir).rem_euclid(2.0 * PI);
            let ri = rho_edges
                .iter()
                .position(|&e| rho < e)
                .unwrap_or(rho_edges.len() - 1);
            let pi = ((phi / (2.0 * PI) * n_phi as f64) as usize).min(n_phi - 1);
            counts[ri * n_phi + pi] += 1;
        }
        let (mut x2, mut dof, mut total) = (0.0, 0usize, 0.0);
        for (ri, &hi) in rho_edges.iter().enumerate() {
            let lo = if ri == 0 { 0.0 } else { rho_edges[ri - 1] };
            let hi = if ri + 1 == rho_edges.len() { 3.0 } else { hi };
            for pi in 0..n_phi {
                let ph = [
                    2.0 * PI * pi as f64 / n_phi as f64,
                    2.0 * PI * (pi + 1) as f64 / n_phi as f64,
                ];
                let m = f.cell_mass(b, [lo, hi], ph);
                total += m;
                let e = m * n as f64;
                let o = counts[ri * n_phi + pi] as f64;
                if e > 5.0 {
                    x2 += (o - e) * (o - e) / e;
                    dof += 1;
                } else {
                    assert!(o < 5.0 * e + 20.0, "cell {ri},{pi}: {o} vs {e}");
                }
            }
        }
        assert!((total - 1.0).abs() < 1e-7, "pdf mass {total}");
        let dof = (dof - 1) as f64;
        // ≈ 5σ upper bound of a χ² variable.
        assert!(
            x2 < dof + 5.0 * (2.0 * dof).sqrt(),
            "χ² = {x2} with {dof} dof"
        );
    }

    #[test]
    fn sampling_matches_pdf_chi_square() {
        let edges = [0.01, 0.03, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0];
        chi_square(&abg(1e-3, 1e-3, 1.8), [0.0, 0.0], &edges, 100_000, 1);
        chi_square(&abg(1e-3, 1e-3, 1.8), [0.6, 0.5], &edges, 100_000, 2);
        chi_square(&abg(1e-3, 1e-2, 2.6), [0.0, 0.9], &edges, 100_000, 3);
        chi_square(&abg(1e-3, 0.0, 1.2), [0.2, 0.0], &edges, 100_000, 4);
        chi_square(&abg(1e-3, 0.5, -0.5), [0.2, 0.0], &edges, 50_000, 5);
        chi_square(
            &gauss(0.1, 0.05),
            [0.3, 0.3],
            &[0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5],
            100_000,
            6,
        );
        chi_square(&lam(0.1), [0.7, 0.0], &[0.2, 0.5, 0.8, 1.2, 1.7], 50_000, 7);
    }

    #[test]
    fn lambertian_moments() {
        // Cosine-weighted hemisphere: E[β] = 0, E[|β|²] = ½.
        let f = lam(0.5);
        let mut rng = Philox::new(9, 0);
        let n = 200_000;
        let (mut mx, mut m2) = (0.0, 0.0);
        for _ in 0..n {
            let p = f.sample([0.3, 0.0], &mut rng).unwrap();
            mx += p[0];
            m2 += p[0] * p[0] + p[1] * p[1];
        }
        let (mx, m2) = (mx / n as f64, m2 / n as f64);
        // σ(β_x) = ½, σ(|β|²) = 1/√12.
        assert!(mx.abs() < 5.0 * 0.5 / (n as f64).sqrt(), "{mx}");
        assert!((m2 - 0.5).abs() < 5.0 * 0.2887 / (n as f64).sqrt(), "{m2}");
    }

    #[test]
    fn narrow_lobes() {
        // Very specular finishes: lobes far narrower than the TIS table spacing.
        let t0 = std::time::Instant::now();
        let f = abg(1e-6, 1e-8, 2.0);
        let exact = PI * 1e-6 * (1e8f64).ln_1p();
        assert!((f.tis(0.0) / exact - 1.0).abs() < 1e-9);
        // The 1/ρ² wings are cut by the disk at any oblique incidence.
        for b in [0.5, 0.9, 0.999] {
            let (t, e) = (f.tis(b), f.tis_exact(b));
            assert!(t < exact && (t / e - 1.0).abs() < 1e-4, "{b}: {t} {e}");
        }
        let g = gauss(0.01, 1e-4);
        assert!((g.tis(0.0) - 0.01).abs() < 1e-14 && (g.tis(0.99) - 0.01).abs() < 1e-12);
        // Near grazing the lobe is cut by the disk edge: about half survives.
        assert!(
            (g.tis_exact(1.0) / 0.01 - 0.5).abs() < 0.01,
            "{}",
            g.tis_exact(1.0)
        );
        assert!(t0.elapsed().as_secs_f64() < 20.0);
    }
}
