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
