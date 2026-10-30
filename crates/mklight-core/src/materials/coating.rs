//! Interface physics: bare Fresnel coefficients and thin-film multilayers
//! (characteristic-matrix method, Macleod's convention `N = n − i k`).
//!
//! All results are *power* coefficients with `R + T + A = 1` for each
//! polarisation (CONVENTIONS §5). Every solver uses this module, so primary
//! transmission, ghost reflection and stray-light splitting all agree.

use num_complex::Complex64 as C;
use std::f64::consts::PI;

use crate::error::{MkError, MkResult};
use crate::materials::{MaterialLibrary, ResolvedMaterial};
use crate::model::{CoatingDef, Environment};

/// Power reflectance and transmittance for s and p polarisation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterfaceRT {
    pub rs: f64,
    pub rp: f64,
    pub ts: f64,
    pub tp: f64,
}

impl InterfaceRT {
    pub const PERFECT_MIRROR: InterfaceRT = InterfaceRT {
        rs: 1.0,
        rp: 1.0,
        ts: 0.0,
        tp: 0.0,
    };
    pub const TRANSPARENT: InterfaceRT = InterfaceRT {
        rs: 0.0,
        rp: 0.0,
        ts: 1.0,
        tp: 1.0,
    };

    pub fn r_unpolarized(&self) -> f64 {
        0.5 * (self.rs + self.rp)
    }
    pub fn t_unpolarized(&self) -> f64 {
        0.5 * (self.ts + self.tp)
    }
    pub fn a_unpolarized(&self) -> f64 {
        (1.0 - self.r_unpolarized() - self.t_unpolarized()).max(0.0)
    }
}

/// Bare dielectric interface. `cos_i` is the cosine of the incidence angle,
/// in [0, 1], in medium `n1`. Total internal reflection gives `R = 1`.
pub fn fresnel(n1: f64, n2: f64, cos_i: f64) -> InterfaceRT {
    let ci = cos_i.clamp(0.0, 1.0);
    let sin2_i = (1.0 - ci) * (1.0 + ci);
    let sin2_t = (n1 / n2) * (n1 / n2) * sin2_i;
    if sin2_t >= 1.0 {
        return InterfaceRT::PERFECT_MIRROR;
    }
    let ct = ((1.0 - sin2_t).max(0.0)).sqrt();
    let (a, b) = (n1 * ci, n2 * ct);
    let rs = if a + b == 0.0 { 1.0 } else { (a - b) / (a + b) };
    let (c, d) = (n2 * ci, n1 * ct);
    let rp = if c + d == 0.0 { 1.0 } else { (c - d) / (c + d) };
    let (rs2, rp2) = (rs * rs, rp * rp);
    InterfaceRT {
        rs: rs2,
        rp: rp2,
        ts: 1.0 - rs2,
        tp: 1.0 - rp2,
    }
}

/// A layer resolved to a complex index function and a physical thickness.
#[derive(Clone, Debug)]
pub struct ResolvedLayer {
    pub material: ResolvedMaterial,
    pub extinction: f64,
    pub thickness_um: f64,
}

#[derive(Clone, Debug)]
pub enum ResolvedCoating {
    /// Uncoated interface: Fresnel for refracting surfaces, a perfect mirror
    /// (`R = 1`) for reflecting ones.
    Bare,
    ThinFilm {
        layers: Vec<ResolvedLayer>,
        reference_wavelength_um: f64,
        mirror_substrate: Option<(f64, f64)>,
    },
    Ideal {
        reflectance: f64,
        transmittance: f64,
    },
    Tabulated {
        reflectance: Vec<(f64, f64)>,
    },
}

impl ResolvedCoating {
    pub fn resolve(
        def: Option<&CoatingDef>,
        lib: &MaterialLibrary,
        env: &Environment,
    ) -> MkResult<Self> {
        let Some(def) = def else {
            return Ok(ResolvedCoating::Bare);
        };
        Ok(match def {
            CoatingDef::Ideal {
                reflectance,
                transmittance,
            } => {
                if *reflectance < 0.0
                    || *transmittance < 0.0
                    || reflectance + transmittance > 1.0 + 1e-12
                {
                    return Err(MkError::InvalidModel(
                        "ideal coating needs R, T ≥ 0 and R + T ≤ 1".into(),
                    ));
                }
                ResolvedCoating::Ideal {
                    reflectance: *reflectance,
                    transmittance: *transmittance,
                }
            }
            CoatingDef::Tabulated { reflectance } => {
                if reflectance.is_empty()
                    || reflectance.iter().any(|(_, r)| !(0.0..=1.0).contains(r))
                {
                    return Err(MkError::InvalidModel(
                        "tabulated coating needs 0 ≤ R ≤ 1 samples".into(),
                    ));
                }
                ResolvedCoating::Tabulated {
                    reflectance: reflectance.clone(),
                }
            }
            CoatingDef::ThinFilm {
                reference_wavelength_um,
                layers,
                mirror_substrate,
            } => {
                let mut out = Vec::with_capacity(layers.len());
                for l in layers {
                    let m = lib.resolve(&l.material, env)?;
                    let t = match l.quarter_waves {
                        Some(q) => {
                            q * reference_wavelength_um / (4.0 * m.index(*reference_wavelength_um))
                        }
                        None => l.thickness_um,
                    };
                    if !(t >= 0.0 && t.is_finite()) {
                        return Err(MkError::InvalidModel(
                            "coating layer thickness must be ≥ 0".into(),
                        ));
                    }
                    out.push(ResolvedLayer {
                        material: m,
                        extinction: l.extinction,
                        thickness_um: t,
                    });
                }
                ResolvedCoating::ThinFilm {
                    layers: out,
                    reference_wavelength_um: *reference_wavelength_um,
                    mirror_substrate: *mirror_substrate,
                }
            }
        })
    }

    /// Power coefficients for light arriving from medium `n_in` at an
    /// interface with medium `n_out` on the other side. `cos_i` lies in
    /// [0, 1]. For a reflecting surface, `n_out` is ignored, and the
    /// transmitted power counts as absorbed (`T = 0`).
    pub fn rt(
        &self,
        lambda_um: f64,
        n_in: f64,
        n_out: f64,
        cos_i: f64,
        mirror: bool,
    ) -> InterfaceRT {
        match self {
            ResolvedCoating::Bare => {
                if mirror {
                    InterfaceRT::PERFECT_MIRROR
                } else {
                    fresnel(n_in, n_out, cos_i)
                }
            }
            ResolvedCoating::Ideal {
                reflectance,
                transmittance,
            } => {
                let t = if mirror { 0.0 } else { *transmittance };
                InterfaceRT {
                    rs: *reflectance,
                    rp: *reflectance,
                    ts: t,
                    tp: t,
                }
            }
            ResolvedCoating::Tabulated { reflectance } => {
                let r0 = crate::materials::dispersion::monotone_cubic(reflectance, lambda_um)
                    .clamp(0.0, 1.0);
                if mirror {
                    return InterfaceRT {
                        rs: r0,
                        rp: r0,
                        ts: 0.0,
                        tp: 0.0,
                    };
                }
                // Scale the bare-Fresnel angular shape so that normal incidence gives r0.
                let bare = fresnel(n_in, n_out, cos_i);
                let bare0 = fresnel(n_in, n_out, 1.0).r_unpolarized();
                let s = if bare0 > 0.0 { r0 / bare0 } else { 0.0 };
                // Total internal reflection stays total.
                let scale = |r: f64| if r >= 1.0 { 1.0 } else { (r * s).min(1.0) };
                let (rs, rp) = (scale(bare.rs), scale(bare.rp));
                InterfaceRT {
                    rs,
                    rp,
                    ts: 1.0 - rs,
                    tp: 1.0 - rp,
                }
            }
            ResolvedCoating::ThinFilm {
                layers,
                mirror_substrate,
                ..
            } => {
                let ni = C::new(n_in, 0.0);
                if mirror {
                    let (n, k) = mirror_substrate.unwrap_or((1e-9, 1e9)); // ≈ perfect conductor
                    let sub = C::new(n, -k);
                    // The light arrives on the coated side, so the stack runs
                    // outermost layer first, down to the substrate.
                    let seq: Vec<(C, f64)> =
                        layers.iter().rev().map(|l| layer_n(l, lambda_um)).collect();
                    let mut r = tmm(ni, &seq, sub, cos_i, lambda_um);
                    r.ts = 0.0;
                    r.tp = 0.0;
                    return r;
                }
                let no = C::new(n_out, 0.0);
                let seq: Vec<(C, f64)> = if n_out >= n_in {
                    // The substrate is on the far side: outermost layer first.
                    layers.iter().rev().map(|l| layer_n(l, lambda_um)).collect()
                } else {
                    // Arriving from inside the substrate: layer 0 first.
                    layers.iter().map(|l| layer_n(l, lambda_um)).collect()
                };
                tmm(ni, &seq, no, cos_i, lambda_um)
            }
        }
    }
}

fn layer_n(l: &ResolvedLayer, lambda_um: f64) -> (C, f64) {
    (
        C::new(l.material.index(lambda_um), -l.extinction),
        l.thickness_um,
    )
}

/// `N cos θ` in medium `n`, taking the branch with non-positive imaginary part
/// (Macleod `N = n − ik`, so fields decay into absorbing media). `s0` is the
/// conserved invariant `n0 sin θ0`.
fn n_cos(n: C, s0: f64) -> C {
    let v = (n * n - C::new(s0 * s0, 0.0)).sqrt();
    // Choose the root in the fourth quadrant (Re ≥ 0, Im ≤ 0), which makes
    // evanescent waves decay.
    if v.im > 0.0 || (v.im == 0.0 && v.re < 0.0) {
        -v
    } else {
        v
    }
}

/// Characteristic-matrix calculation for a stack in order from the incident
/// medium `n0` to the exit medium `ns`. Returns power coefficients.
pub fn tmm(n0: C, layers: &[(C, f64)], ns: C, cos_i: f64, lambda_um: f64) -> InterfaceRT {
    let ci = cos_i.clamp(0.0, 1.0);
    let s0 = n0.re * ((1.0 - ci) * (1.0 + ci)).sqrt();
    let y0c = n_cos(n0, s0); // n0 cosθ0
    let ysc = n_cos(ns, s0);
    let calc = |pol_p: bool| -> (f64, f64) {
        // Tilted admittances: s → N cosθ, p → N / cosθ = N² / (N cosθ).
        let adm = |n: C, nc: C| if pol_p { n * n / nc } else { nc };
        let eta0 = adm(n0, y0c);
        let etas = if ysc.norm() == 0.0 {
            C::new(0.0, 0.0)
        } else {
            adm(ns, ysc)
        };
        let mut m = [
            [C::new(1.0, 0.0), C::new(0.0, 0.0)],
            [C::new(0.0, 0.0), C::new(1.0, 0.0)],
        ];
        for &(n, d) in layers {
            let nc = n_cos(n, s0);
            let delta = 2.0 * PI * nc * d / lambda_um;
            let eta = adm(n, nc);
            let (cd, sd) = (delta.cos(), delta.sin());
            let i = C::new(0.0, 1.0);
            let l = [[cd, i * sd / eta], [i * eta * sd, cd]];
            m = [
                [
                    m[0][0] * l[0][0] + m[0][1] * l[1][0],
                    m[0][0] * l[0][1] + m[0][1] * l[1][1],
                ],
                [
                    m[1][0] * l[0][0] + m[1][1] * l[1][0],
                    m[1][0] * l[0][1] + m[1][1] * l[1][1],
                ],
            ];
        }
        let b = m[0][0] + m[0][1] * etas;
        let c = m[1][0] + m[1][1] * etas;
        let den = eta0 * b + c;
        if den.norm() == 0.0 {
            return (1.0, 0.0);
        }
        let r = (eta0 * b - c) / den;
        let rr = r.norm_sqr().min(1.0);
        let tt = (4.0 * eta0.re * etas.re / den.norm_sqr()).max(0.0);
        (rr, tt.min(1.0 - rr))
    };
    let (rs, ts) = calc(false);
    let (rp, tp) = calc(true);
    InterfaceRT { rs, rp, ts, tp }
}

// ---------------------------------------------------------------------------
// Complex amplitude coefficients (CONVENTIONS §8).
//
// The power functions above stay the reference for `R`, `T` and `A`; the
// amplitude functions below reproduce them (`|r|²`, `|t|²·flux`) and add the
// phases that polarisation ray tracing and vector wave optics need.

/// Complex number type of the amplitude API (re-exported so that dependent
/// crates use the same type without their own `num-complex` dependency).
pub use num_complex::Complex64;

/// Complex amplitude reflection and transmission coefficients for s and p
/// polarisation (CONVENTIONS §8): time dependence `exp(i(ωt − k·r))`,
/// `N = n − ik`, s along `k × normal`, p along `k × s` before and `k' × s`
/// after the interface, so at normal incidence `r_p = −r_s` and `t_p = t_s`.
///
/// `t_s`, `t_p` are ratios of the full field amplitudes. The transmitted
/// power fraction is `T = |t|²·flux`, where `flux` is the ratio of the
/// normal energy-flux densities per unit squared amplitude,
/// `Re(N₂ cos θ₂)/(n₁ cos θ₁)` for s and
/// `Re(N₂ / cos θ₂)/(n₁ / cos θ₁) · |cos θ₂ / cos θ₁|²` for p; both equal
/// `n₂ cos θ₂ / (n₁ cos θ₁)` for a non-absorbing exit medium. On a mirror,
/// and for evanescent (totally reflected) transmission, `flux = 0`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterfaceAmplitudes {
    pub rs: Complex64,
    pub rp: Complex64,
    pub ts: Complex64,
    pub tp: Complex64,
    /// `T_s = |t_s|²·flux_s`.
    pub flux_s: f64,
    /// `T_p = |t_p|²·flux_p`.
    pub flux_p: f64,
}

const CZERO: C = C::new(0.0, 0.0);

impl InterfaceAmplitudes {
    /// A perfect conductor (`E_tangential = 0`): `r_s = −1`, `r_p = +1`.
    /// This is the phase given to every ideal reflector (bare mirrors,
    /// ideal coatings): a non-polarising mirror without retardance.
    pub const PERFECT_CONDUCTOR: InterfaceAmplitudes = InterfaceAmplitudes {
        rs: C::new(-1.0, 0.0),
        rp: C::new(1.0, 0.0),
        ts: CZERO,
        tp: CZERO,
        flux_s: 0.0,
        flux_p: 0.0,
    };
    /// A lossless, phase-free, non-reflecting interface (`t = 1`, `flux = 1`).
    pub const TRANSPARENT: InterfaceAmplitudes = InterfaceAmplitudes {
        rs: CZERO,
        rp: CZERO,
        ts: C::new(1.0, 0.0),
        tp: C::new(1.0, 0.0),
        flux_s: 1.0,
        flux_p: 1.0,
    };

    /// Power coefficients: `R = |r|²`, `T = |t|²·flux`.
    pub fn power(&self) -> InterfaceRT {
        InterfaceRT {
            rs: self.rs.norm_sqr(),
            rp: self.rp.norm_sqr(),
            ts: self.ts.norm_sqr() * self.flux_s,
            tp: self.tp.norm_sqr() * self.flux_p,
        }
    }
    /// Flux-normalised s transmission `t_s·√flux_s`: same phase as `t_s`,
    /// squared modulus `T_s`.
    pub fn ts_flux(&self) -> Complex64 {
        self.ts * self.flux_s.sqrt()
    }
    /// Flux-normalised p transmission `t_p·√flux_p`.
    pub fn tp_flux(&self) -> Complex64 {
        self.tp * self.flux_p.sqrt()
    }
    /// The same amplitudes with the transmission removed (`t = 0`,
    /// `flux = 0`): a reflecting surface counts transmitted power as absorbed.
    fn opaque(mut self) -> Self {
        self.ts = CZERO;
        self.tp = CZERO;
        self.flux_s = 0.0;
        self.flux_p = 0.0;
        self
    }
}

/// Complex Fresnel amplitudes of a bare interface between non-absorbing
/// media (`cos_i` in medium `n1`, clamped to [0, 1]), consistent with
/// [`fresnel`]:
///
/// * `r_s = (n₁c₁ − n₂c₂)/(n₁c₁ + n₂c₂)`, `t_s = 2n₁c₁/(n₁c₁ + n₂c₂)`,
/// * `r_p = (n₂c₁ − n₁c₂)/(n₂c₁ + n₁c₂)`, `t_p = 2n₁c₁/(n₂c₁ + n₁c₂)`.
///
/// The numerators are evaluated without cancellation (Snell's law gives
/// `n₁²c₁² − n₂²c₂² = (n₁ − n₂)(n₁ + n₂)` and
/// `n₂²c₁² − n₁²c₂² = (n₂² − n₁²)(n₂c₁ − n₁s₁)(n₂c₁ + n₁s₁)/n₂²`), so
/// low-contrast (cemented) interfaces keep full relative accuracy.
///
/// Beyond the critical angle `n₂c₂ = −i·g`, `g = √(n₁²s₁² − n₂²)` (the
/// decaying root for `N = n − ik`), so `r_s = (n₁c₁ + ig)/(n₁c₁ − ig)` and
/// `r_p = (n₂c₁ + i g n₁/n₂)/(n₂c₁ − i g n₁/n₂)` have unit modulus and the
/// phases `φ_s = 2 atan(g/(n₁c₁))`, `φ_p = 2 atan(n₁ g/(n₂² c₁))`. The
/// evanescent `t` is returned with `flux = 0`.
pub fn fresnel_amplitudes(n1: f64, n2: f64, cos_i: f64) -> InterfaceAmplitudes {
    let ci = cos_i.clamp(0.0, 1.0);
    let sin2_i = (1.0 - ci) * (1.0 + ci);
    let mu = n1 / n2;
    let sin2_t = mu * mu * sin2_i;
    let a = n1 * ci; // n₁c₁
    if sin2_t >= 1.0 {
        let si = sin2_i.sqrt();
        let g = ((n1 * si - n2) * (n1 * si + n2)).max(0.0).sqrt();
        let b = n2 * ci; // n₂c₁
        let h = g * mu; // |n₁c₂|
        let rs = C::new(a, g) / C::new(a, -g);
        let rp = C::new(b, h) / C::new(b, -h);
        let ts = C::new(2.0 * a, 0.0) / C::new(a, -g);
        let tp = C::new(2.0 * a, 0.0) / C::new(b, -h);
        return InterfaceAmplitudes {
            rs,
            rp,
            ts,
            tp,
            flux_s: 0.0,
            flux_p: 0.0,
        };
    }
    let ct = (-mu * mu).mul_add(sin2_i, 1.0).max(0.0).sqrt();
    let b = n2 * ct; // n₂c₂
    let c = n2 * ci; // n₂c₁
    let d = n1 * ct; // n₁c₂
    let ss = a + b;
    let sp = c + d;
    let rs = (n1 - n2) * (n1 + n2) / (ss * ss);
    let si = sin2_i.sqrt();
    let rp = ((n2 - n1) * (n2 + n1) * ((c - n1 * si) * (c + n1 * si)) / (n2 * n2)) / (sp * sp);
    let (ts, tp, flux) = if a > 0.0 {
        (2.0 * a / ss, 2.0 * a / sp, b / a)
    } else {
        (0.0, 0.0, 0.0) // grazing incidence: nothing enters
    };
    InterfaceAmplitudes {
        rs: C::new(rs, 0.0),
        rp: C::new(rp, 0.0),
        ts: C::new(ts, 0.0),
        tp: C::new(tp, 0.0),
        flux_s: flux,
        flux_p: flux,
    }
}

/// Characteristic-matrix amplitudes for a stack ordered from the incident
/// medium `n0` to the exit medium `ns`, the amplitude counterpart of
/// [`tmm`] (same branch choices and degenerate cases, so
/// `tmm_amplitudes(..).power()` equals `tmm(..)` to rounding).
///
/// Macleod's tilted admittances (`η_s = N cos θ`, `η_p = N / cos θ`) give
/// the tangential-field coefficients `ρ = (η₀B − C)/(η₀B + C)` and
/// `τ = 2η₀/(η₀B + C)` with `[B, C]ᵀ = M·[1, η_sub]ᵀ` and the layer
/// matrices `[[cos δ, i sin δ/η], [iη sin δ, cos δ]]`, `δ = 2πN d cos θ/λ`.
/// In the basis of CONVENTIONS §8 the field coefficients are
/// `r_s = ρ_s`, `t_s = τ_s`, `r_p = −ρ_p` (the reflected p vector `k' × s`
/// has the opposite tangential component) and
/// `t_p = τ_p · cos θ₀ / cos θ_sub` (the p field is `E_tan / cos θ`), with
/// `cos θ = (N cos θ)/N` complex in absorbing media.
pub fn tmm_amplitudes(
    n0: C,
    layers: &[(C, f64)],
    ns: C,
    cos_i: f64,
    lambda_um: f64,
) -> InterfaceAmplitudes {
    let ci = cos_i.clamp(0.0, 1.0);
    let s0 = n0.re * ((1.0 - ci) * (1.0 + ci)).sqrt();
    let y0c = n_cos(n0, s0);
    let ysc = n_cos(ns, s0);
    let one = C::new(1.0, 0.0);
    // Returns (ρ, τ, Re η₀·Re η_s/|η₀|²) for the tangential fields.
    let calc = |pol_p: bool| -> (C, C, f64) {
        let adm = |n: C, nc: C| if pol_p { n * n / nc } else { nc };
        let eta0 = adm(n0, y0c);
        let etas = if ysc.norm() == 0.0 {
            CZERO
        } else {
            adm(ns, ysc)
        };
        let mut m = [[one, CZERO], [CZERO, one]];
        for &(n, d) in layers {
            let nc = n_cos(n, s0);
            let delta = 2.0 * PI * nc * d / lambda_um;
            let eta = adm(n, nc);
            let (cd, sd) = (delta.cos(), delta.sin());
            let i = C::new(0.0, 1.0);
            let l = [[cd, i * sd / eta], [i * eta * sd, cd]];
            m = [
                [
                    m[0][0] * l[0][0] + m[0][1] * l[1][0],
                    m[0][0] * l[0][1] + m[0][1] * l[1][1],
                ],
                [
                    m[1][0] * l[0][0] + m[1][1] * l[1][0],
                    m[1][0] * l[0][1] + m[1][1] * l[1][1],
                ],
            ];
        }
        let b = m[0][0] + m[0][1] * etas;
        let c = m[1][0] + m[1][1] * etas;
        let den = eta0 * b + c;
        if den.norm() == 0.0 {
            return (one, CZERO, 0.0);
        }
        let rho = (eta0 * b - c) / den;
        let tau = 2.0 * eta0 / den;
        let e2 = eta0.norm_sqr();
        let flux = if e2 > 0.0 {
            (eta0.re * etas.re / e2).max(0.0)
        } else {
            0.0
        };
        (rho, tau, flux)
    };
    let (rho_s, tau_s, flux_s) = calc(false);
    let (rho_p, tau_p, flux_tan_p) = calc(true);
    // cos θ₀ / cos θ_sub for the p field amplitude.
    let (tp, flux_p) = if ysc.norm() == 0.0 || y0c.norm() == 0.0 {
        (CZERO, 0.0)
    } else {
        let ratio = (y0c / n0) / (ysc / ns);
        (tau_p * ratio, flux_tan_p / ratio.norm_sqr())
    };
    InterfaceAmplitudes {
        rs: rho_s,
        rp: -rho_p,
        ts: tau_s,
        tp,
        flux_s,
        flux_p,
    }
}

/// Field-amplitude transmission `t` with `|t|²·flux = T` for a coefficient
/// that has only a power value: the bare-interface flux factor
/// `n₂c₂/(n₁c₁)` (1 when it is undefined), so the amplitudes stay
/// consistent with CONVENTIONS §8 while carrying no phase.
fn phase_free_transmission(n_in: f64, n_out: f64, cos_i: f64, t_power: f64) -> (C, f64) {
    let ci = cos_i.clamp(0.0, 1.0);
    let mu = n_in / n_out;
    let disc = (-mu * mu).mul_add((1.0 - ci) * (1.0 + ci), 1.0);
    let flux = if disc > 0.0 && ci > 0.0 {
        n_out * disc.sqrt() / (n_in * ci)
    } else {
        1.0
    };
    (C::new((t_power / flux).sqrt(), 0.0), flux)
}

/// `amp` rescaled to modulus `√power` (keeping its phase; a zero `amp`
/// becomes the positive real `√power`).
fn with_power(amp: C, power: f64) -> C {
    let m = amp.norm();
    if m > 0.0 {
        amp * (power.max(0.0).sqrt() / m)
    } else {
        C::new(power.max(0.0).sqrt(), 0.0)
    }
}

impl ResolvedCoating {
    /// Complex amplitude coefficients, the counterpart of [`Self::rt`] with
    /// the same arguments: `self.amplitudes(..).power()` equals
    /// `self.rt(..)` to rounding for every coating kind.
    ///
    /// Phases (CONVENTIONS §8):
    /// * `Bare`: Fresnel ([`fresnel_amplitudes`]); a bare mirror is a
    ///   perfect conductor (`r_s = −1`, `r_p = +1`).
    /// * `Ideal`: phase-free and non-polarising. Reflection
    ///   `r_s = −√R`, `r_p = +√R` (a scaled perfect conductor, zero
    ///   retardance), transmission `t_s = t_p` real positive with
    ///   `|t|²·flux = T` (bare-interface flux factor).
    /// * `Tabulated`: the bare-Fresnel phases with moduli rescaled to the
    ///   tabulated powers (the table holds no phase information); on a
    ///   mirror a scaled perfect conductor.
    /// * `ThinFilm`: [`tmm_amplitudes`].
    pub fn amplitudes(
        &self,
        lambda_um: f64,
        n_in: f64,
        n_out: f64,
        cos_i: f64,
        mirror: bool,
    ) -> InterfaceAmplitudes {
        match self {
            ResolvedCoating::Bare => {
                if mirror {
                    InterfaceAmplitudes::PERFECT_CONDUCTOR
                } else {
                    fresnel_amplitudes(n_in, n_out, cos_i)
                }
            }
            ResolvedCoating::Ideal {
                reflectance,
                transmittance,
            } => {
                let r = reflectance.sqrt();
                let (t, flux) = if mirror {
                    (CZERO, 0.0)
                } else {
                    phase_free_transmission(n_in, n_out, cos_i, *transmittance)
                };
                InterfaceAmplitudes {
                    rs: C::new(-r, 0.0),
                    rp: C::new(r, 0.0),
                    ts: t,
                    tp: t,
                    flux_s: flux,
                    flux_p: flux,
                }
            }
            ResolvedCoating::Tabulated { .. } => {
                let p = self.rt(lambda_um, n_in, n_out, cos_i, mirror);
                if mirror {
                    let r = p.rs.sqrt();
                    return InterfaceAmplitudes {
                        rs: C::new(-r, 0.0),
                        rp: C::new(r, 0.0),
                        ..InterfaceAmplitudes::PERFECT_CONDUCTOR
                    };
                }
                let bare = fresnel_amplitudes(n_in, n_out, cos_i);
                if bare.flux_s == 0.0 && bare.flux_p == 0.0 {
                    // Total internal reflection stays total (and keeps its phase).
                    return bare;
                }
                let scale_t = |t: C, flux: f64, power: f64| {
                    if flux > 0.0 {
                        with_power(t, power / flux)
                    } else {
                        CZERO
                    }
                };
                InterfaceAmplitudes {
                    rs: with_power(bare.rs, p.rs),
                    rp: with_power(bare.rp, p.rp),
                    ts: scale_t(bare.ts, bare.flux_s, p.ts),
                    tp: scale_t(bare.tp, bare.flux_p, p.tp),
                    ..bare
                }
            }
            ResolvedCoating::ThinFilm {
                layers,
                mirror_substrate,
                ..
            } => {
                let ni = C::new(n_in, 0.0);
                if mirror {
                    let (n, k) = mirror_substrate.unwrap_or((1e-9, 1e9));
                    let sub = C::new(n, -k);
                    let seq: Vec<(C, f64)> =
                        layers.iter().rev().map(|l| layer_n(l, lambda_um)).collect();
                    return tmm_amplitudes(ni, &seq, sub, cos_i, lambda_um).opaque();
                }
                let no = C::new(n_out, 0.0);
                let seq: Vec<(C, f64)> = if n_out >= n_in {
                    layers.iter().rev().map(|l| layer_n(l, lambda_um)).collect()
                } else {
                    layers.iter().map(|l| layer_n(l, lambda_um)).collect()
                };
                tmm_amplitudes(ni, &seq, no, cos_i, lambda_um)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Layer, MaterialRef};

    #[test]
    fn fresnel_normal_incidence() {
        let r = fresnel(1.0, 1.5, 1.0);
        assert!((r.rs - 0.04).abs() < 1e-12 && (r.rp - 0.04).abs() < 1e-12);
        assert!((r.r_unpolarized() + r.t_unpolarized() - 1.0).abs() < 1e-15);
    }

    #[test]
    fn brewster_angle() {
        let tb = (1.5f64).atan();
        let r = fresnel(1.0, 1.5, tb.cos());
        assert!(r.rp < 1e-12);
    }

    #[test]
    fn tir() {
        let r = fresnel(1.5, 1.0, (60f64.to_radians()).cos());
        assert_eq!(r, InterfaceRT::PERFECT_MIRROR);
    }

    #[test]
    fn tmm_no_layers_equals_fresnel() {
        for deg in [0.0, 20.0, 45.0, 70.0, 85.0] {
            let ci = f64::cos(f64::to_radians(deg));
            let a = fresnel(1.0, 1.7, ci);
            let b = tmm(C::new(1.0, 0.0), &[], C::new(1.7, 0.0), ci, 0.55);
            assert!(
                (a.rs - b.rs).abs() < 1e-12 && (a.rp - b.rp).abs() < 1e-12,
                "{deg}: {a:?} {b:?}"
            );
            assert!((b.rs + b.ts - 1.0).abs() < 1e-12 && (b.rp + b.tp - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn quarter_wave_ar_is_ideal_at_design() {
        // Single quarter-wave layer with n = √(1.0·1.69) on n = 1.69 gives R = 0 at λ0.
        let lib = MaterialLibrary::with_builtin();
        let env = Environment::default();
        let def = CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![Layer {
                material: MaterialRef::Constant { n: 1.3 },
                thickness_um: 0.0,
                quarter_waves: Some(1.0),
                extinction: 0.0,
            }],
            mirror_substrate: None,
        };
        let c = ResolvedCoating::resolve(Some(&def), &lib, &env).unwrap();
        let r = c.rt(0.55, 1.0, 1.69, 1.0, false);
        assert!(r.r_unpolarized() < 1e-12, "{r:?}");
        // Same coating seen from inside the glass gives the same R (reciprocity).
        let r2 = c.rt(0.55, 1.69, 1.0, 1.0, false);
        assert!(r2.r_unpolarized() < 1e-12);
        // Energy conservation at oblique incidence and off-design wavelength.
        let r3 = c.rt(0.45, 1.0, 1.69, 0.6, false);
        assert!((r3.rs + r3.ts - 1.0).abs() < 1e-12 && (r3.rp + r3.tp - 1.0).abs() < 1e-12);
    }

    #[test]
    fn absorbing_layer_conserves_energy() {
        let r = tmm(
            C::new(1.0, 0.0),
            &[(C::new(2.0, -0.3), 0.05)],
            C::new(1.5, 0.0),
            0.8,
            0.6,
        );
        let a_s = 1.0 - r.rs - r.ts;
        let a_p = 1.0 - r.rp - r.tp;
        assert!(a_s > 0.0 && a_p > 0.0 && a_s < 1.0 && a_p < 1.0);
    }

    #[test]
    fn metal_mirror_reflects() {
        let r = tmm(C::new(1.0, 0.0), &[], C::new(0.96, -6.69), 1.0, 0.55); // aluminium
        assert!(r.r_unpolarized() > 0.9 && r.r_unpolarized() < 0.93, "{r:?}");
    }
}

#[cfg(test)]
mod amplitude_tests {
    use super::*;
    use crate::model::{Layer, MaterialRef};
    use crate::rng::Philox;

    fn close(a: C, b: C, tol: f64) -> bool {
        (a - b).norm() <= tol
    }

    fn assert_power_eq(a: InterfaceRT, b: InterfaceRT, tol: f64, what: &str) {
        let d = (a.rs - b.rs)
            .abs()
            .max((a.rp - b.rp).abs())
            .max((a.ts - b.ts).abs())
            .max((a.tp - b.tp).abs());
        assert!(d <= tol, "{what}: {a:?} vs {b:?} (diff {d:e})");
    }

    /// A random stack: 0–6 layers, some absorbing.
    fn random_stack(r: &mut Philox, absorbing: bool) -> Vec<(C, f64)> {
        let n = (r.uniform() * 7.0) as usize;
        (0..n)
            .map(|_| {
                let k = if absorbing && r.uniform() < 0.4 {
                    r.uniform() * 0.8
                } else {
                    0.0
                };
                (C::new(1.3 + 1.1 * r.uniform(), -k), 0.3 * r.uniform())
            })
            .collect()
    }

    #[test]
    fn fresnel_amplitudes_match_power() {
        let mut r = Philox::new(11, 0);
        for _ in 0..20_000 {
            let n1 = 1.0 + r.uniform();
            let n2 = 1.0 + r.uniform();
            let ci = r.uniform();
            let a = fresnel_amplitudes(n1, n2, ci);
            let p = fresnel(n1, n2, ci);
            assert_power_eq(a.power(), p, 1e-12, &format!("n=({n1},{n2}) c={ci}"));
        }
        // Cemented interface at near-normal incidence keeps relative accuracy.
        let (n1, n2) = (1.5168, 1.5168 + 1e-9);
        let a = fresnel_amplitudes(n1, n2, 1.0);
        let exact = (n1 - n2) / (n1 + n2);
        assert!((a.rs.re - exact).abs() <= 1e-14 * exact.abs());
        assert!((a.rp.re + exact).abs() <= 1e-14 * exact.abs());
    }

    #[test]
    fn tmm_amplitudes_match_power_for_random_stacks() {
        let mut r = Philox::new(7, 1);
        for case in 0..5_000 {
            let stack = random_stack(&mut r, true);
            let n0 = C::new(1.0 + 0.8 * r.uniform(), 0.0);
            let ns = C::new(1.0 + 0.8 * r.uniform(), 0.0);
            let ci = 0.02 + 0.98 * r.uniform();
            let lam = 0.4 + 0.4 * r.uniform();
            let a = tmm_amplitudes(n0, &stack, ns, ci, lam);
            let p = tmm(n0, &stack, ns, ci, lam);
            assert_power_eq(a.power(), p, 1e-12, &format!("case {case}"));
            // Metal substrate (mirror back side) too.
            let sub = C::new(0.2 + r.uniform(), -(2.0 + 6.0 * r.uniform()));
            let a = tmm_amplitudes(n0, &stack, sub, ci, lam);
            let p = tmm(n0, &stack, sub, ci, lam);
            assert_power_eq(a.power(), p, 1e-12, &format!("metal case {case}"));
        }
    }

    #[test]
    fn empty_stack_equals_bare_fresnel_including_tir() {
        let mut r = Philox::new(3, 2);
        for _ in 0..5_000 {
            let n1 = 1.0 + r.uniform();
            let n2 = 1.0 + r.uniform();
            let ci = 0.01 + 0.99 * r.uniform();
            let a = fresnel_amplitudes(n1, n2, ci);
            let b = tmm_amplitudes(C::new(n1, 0.0), &[], C::new(n2, 0.0), ci, 0.55);
            for (x, y) in [(a.rs, b.rs), (a.rp, b.rp), (a.ts, b.ts), (a.tp, b.tp)] {
                assert!(close(x, y, 1e-12), "n=({n1},{n2}) c={ci}: {a:?} {b:?}");
            }
        }
    }

    #[test]
    fn normal_incidence_is_isotropic() {
        // The basis convention makes r_p = −r_s and t_p = t_s at normal
        // incidence for every stack (no preferred direction exists).
        let mut r = Philox::new(5, 3);
        for _ in 0..2_000 {
            let stack = random_stack(&mut r, true);
            let a = tmm_amplitudes(C::new(1.0, 0.0), &stack, C::new(1.52, 0.0), 1.0, 0.55);
            assert!(
                close(a.rp, -a.rs, 1e-12) && close(a.tp, a.ts, 1e-12),
                "{a:?}"
            );
            assert!((a.flux_s - a.flux_p).abs() < 1e-12);
        }
    }

    #[test]
    fn stokes_relations_and_reciprocity() {
        // Forward: n0 → stack → ns at angle θ0; reverse: ns → reversed stack
        // → n0 at the Snell angle θs. Flux-normalised transmission is
        // reciprocal (t̃ = t̃'), and for lossless stacks the scattering matrix
        // is unitary: |r|² + |t̃|² = 1 and r' = −r*·t̃/t̃*.
        let mut r = Philox::new(9, 4);
        let mut checked = 0;
        for case in 0..3_000 {
            let absorbing = case % 2 == 1;
            let stack = random_stack(&mut r, absorbing);
            let (n0, ns) = (1.0 + 0.6 * r.uniform(), 1.0 + 0.6 * r.uniform());
            let ci = 0.05 + 0.95 * r.uniform();
            let s0 = n0 * ((1.0 - ci) * (1.0 + ci)).sqrt();
            if s0 >= ns * 0.999 {
                continue; // no propagating transmitted wave
            }
            checked += 1;
            let cs = (1.0 - (s0 / ns) * (s0 / ns)).sqrt();
            let rev: Vec<_> = stack.iter().rev().copied().collect();
            let lam = 0.55;
            let f = tmm_amplitudes(C::new(n0, 0.0), &stack, C::new(ns, 0.0), ci, lam);
            let b = tmm_amplitudes(C::new(ns, 0.0), &rev, C::new(n0, 0.0), cs, lam);
            for (pol, (tf, tb, rf, rb)) in [
                ("s", (f.ts_flux(), b.ts_flux(), f.rs, b.rs)),
                ("p", (f.tp_flux(), b.tp_flux(), f.rp, b.rp)),
            ] {
                assert!(close(tf, tb, 1e-11), "{case} {pol}: t {tf} vs {tb}");
                if !absorbing {
                    assert!((rf.norm_sqr() + tf.norm_sqr() - 1.0).abs() < 1e-12);
                    let expect = -rf.conj() * tf / tf.conj();
                    assert!(
                        close(rb, expect, 1e-11),
                        "{case} {pol}: r' {rb} vs {expect}"
                    );
                }
            }
        }
        assert!(checked > 1000);
    }

    #[test]
    fn tir_phase_matches_closed_form() {
        // Glass → air beyond the critical angle: |r| = 1 with
        // tan(φs/2) = √(sin²θ − n²)/cos θ and tan(φp/2) = √(sin²θ − n²)/(n² cos θ),
        // n = n₂/n₁.
        let (n1, n2) = (1.5, 1.0);
        let n = n2 / n1;
        for deg in [41.9f64, 45.0, 50.0, 60.0, 75.0, 89.0] {
            let th = deg.to_radians();
            let (s, c) = th.sin_cos();
            let g = (s * s - n * n).sqrt();
            let phi_s = 2.0 * (g / c).atan();
            let phi_p = 2.0 * (g / (n * n * c)).atan();
            for a in [
                fresnel_amplitudes(n1, n2, c),
                tmm_amplitudes(C::new(n1, 0.0), &[], C::new(n2, 0.0), c, 0.55),
            ] {
                assert!((a.rs.norm() - 1.0).abs() < 1e-14 && (a.rp.norm() - 1.0).abs() < 1e-14);
                assert!(
                    (a.rs.arg() - phi_s).abs() < 1e-12,
                    "{deg}: {} {phi_s}",
                    a.rs.arg()
                );
                assert!(
                    (a.rp.arg() - phi_p).abs() < 1e-12,
                    "{deg}: {} {phi_p}",
                    a.rp.arg()
                );
                assert_eq!(a.power().ts, 0.0);
                assert_eq!(a.power().tp, 0.0);
            }
        }
        // Fresnel rhomb: n = 1.51 at 54.6° gives a 45° p–s phase difference.
        let th = 54.6f64.to_radians();
        let a = fresnel_amplitudes(1.51, 1.0, th.cos());
        let d = (a.rp / a.rs).arg().to_degrees();
        assert!((d - 45.0).abs() < 0.3, "{d}");
    }

    #[test]
    fn brewster_angle_amplitudes() {
        for (n1, n2) in [(1.0, 1.5), (1.0, 1.9), (1.6, 1.0), (1.52, 1.62)] {
            let tb = f64::atan(n2 / n1);
            let a = fresnel_amplitudes(n1, n2, tb.cos());
            assert!(a.rp.norm() < 1e-15, "{a:?}");
            // Known answer: t_p = n₁/n₂ at Brewster's angle, and T_p = 1.
            assert!((a.tp.re - n1 / n2).abs() < 1e-14);
            assert!((a.power().tp - 1.0).abs() < 1e-14);
            let b = tmm_amplitudes(C::new(n1, 0.0), &[], C::new(n2, 0.0), tb.cos(), 0.55);
            assert!(b.rp.norm() < 1e-14);
        }
    }

    #[test]
    fn metal_mirror_phase_matches_closed_form() {
        // Aluminium at 550 nm, N = 0.96 − 6.69i, from air.
        let nm = C::new(0.96, -6.69);
        let one = C::new(1.0, 0.0);
        let a = tmm_amplitudes(one, &[], nm, 1.0, 0.55);
        let r0 = (one - nm) / (one + nm);
        assert!(close(a.rs, r0, 1e-14) && close(a.rp, -r0, 1e-14), "{a:?}");
        // Closed-form complex Fresnel at 45°: N₂cos θ₂ = √(N² − sin²θ) on the
        // decaying branch, cos θ₂ = (N₂ cos θ₂)/N₂.
        let th = 45f64.to_radians();
        let (s, c) = th.sin_cos();
        let mut nc = (nm * nm - C::new(s * s, 0.0)).sqrt();
        if nc.im > 0.0 {
            nc = -nc;
        }
        let c2 = nc / nm;
        let rs = (c - nc) / (c + nc);
        let rp = (nm * c - c2) / (nm * c + c2);
        let a = tmm_amplitudes(one, &[], nm, c, 0.55);
        assert!(
            close(a.rs, rs, 1e-13) && close(a.rp, rp, 1e-13),
            "{a:?} {rs} {rp}"
        );
        // Reflection retardance relative to a perfect mirror, arg(r_p/(−r_s)),
        // agrees with the good-conductor expansion
        // r_p/(−r_s) ≈ 1 − 2 sin θ tan θ / N, i.e. a phase ≈ −2 sin θ tan θ·Im(1/N)
        // (≈ −11.9° for aluminium at 45°; the expansion is first order in 1/|N|).
        let ret = (a.rp / (-a.rs)).arg();
        let approx = -2.0 * s * (s / c) * (one / nm).im;
        assert!(
            ret < 0.0 && (ret - approx).abs() < 0.05 * approx.abs(),
            "{ret} {approx}"
        );
        // The default thin-film mirror substrate approximates a perfect conductor.
        let pc = tmm_amplitudes(one, &[], C::new(1e-9, -1e9), 0.7, 0.55);
        let ideal = InterfaceAmplitudes::PERFECT_CONDUCTOR;
        assert!(
            close(pc.rs, ideal.rs, 1e-8) && close(pc.rp, ideal.rp, 1e-8),
            "{pc:?}"
        );
    }

    #[test]
    fn resolved_coating_amplitudes_match_rt() {
        let lib = MaterialLibrary::with_builtin();
        let env = Environment::default();
        let layer = |n: f64, q: f64, k: f64| Layer {
            material: MaterialRef::Constant { n },
            thickness_um: 0.0,
            quarter_waves: Some(q),
            extinction: k,
        };
        let defs = [
            None,
            Some(CoatingDef::Ideal {
                reflectance: 0.3,
                transmittance: 0.6,
            }),
            Some(CoatingDef::Tabulated {
                reflectance: vec![(0.4, 0.02), (0.55, 0.005), (0.7, 0.03)],
            }),
            Some(CoatingDef::ThinFilm {
                reference_wavelength_um: 0.55,
                layers: vec![layer(2.1, 1.0, 0.0), layer(1.38, 1.0, 0.0)],
                mirror_substrate: None,
            }),
            Some(CoatingDef::ThinFilm {
                reference_wavelength_um: 0.55,
                layers: vec![layer(1.6, 1.0, 0.02), layer(1.38, 0.7, 0.0)],
                mirror_substrate: Some((0.96, 6.69)),
            }),
        ];
        for def in &defs {
            let c = ResolvedCoating::resolve(def.as_ref(), &lib, &env).unwrap();
            for (n_in, n_out) in [(1.0, 1.52), (1.52, 1.0), (1.62, 1.52)] {
                for deg in [0.0f64, 10.0, 30.0, 40.0, 50.0, 70.0, 85.0] {
                    let ci = deg.to_radians().cos();
                    for mirror in [false, true] {
                        for lam in [0.45, 0.55, 0.65] {
                            let a = c.amplitudes(lam, n_in, n_out, ci, mirror);
                            let p = c.rt(lam, n_in, n_out, ci, mirror);
                            assert_power_eq(
                                a.power(),
                                p,
                                1e-12,
                                &format!("{def:?} n=({n_in},{n_out}) {deg}° m={mirror}"),
                            );
                        }
                    }
                }
            }
        }
        // Ideal coatings are non-polarising: r_p = −r_s, t_p = t_s at any angle.
        let c = ResolvedCoating::resolve(defs[1].as_ref(), &lib, &env).unwrap();
        let a = c.amplitudes(0.55, 1.0, 1.5, 0.6, false);
        assert!(a.rp == -a.rs && a.tp == a.ts);
        assert!(a.rs.re < 0.0 && a.ts.im == 0.0 && a.ts.re > 0.0);
    }
}
