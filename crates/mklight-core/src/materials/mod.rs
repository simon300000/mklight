//! Optical materials: glass catalogs, dispersion, thermal index changes and
//! internal absorption, plus interface physics (Fresnel equations and
//! thin-film coatings) in `coating`.

pub mod agf;
pub mod builtin;
pub mod coating;
pub mod diffraction;
pub mod dispersion;

#[cfg(test)]
mod offset_tests;

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::{Arc, OnceLock};

pub use dispersion::{Dispersion, ThermalCoefficients, air_index};

use crate::error::{MkError, MkResult};
use crate::model::{Environment, MaterialRef};

/// Default catalog reference conditions. A glass whose thermal data name
/// another reference temperature (`ThermalCoefficients::t_ref_c`) uses that one.
pub const CATALOG_T_C: f64 = 20.0;
pub const CATALOG_P_ATM: f64 = 1.0;

/// Changes `Δn_d`, `ΔV_d` to a catalogue glass (MATERIALS.md): the
/// dispersion is scaled about d, keeping partial-dispersion ratios.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MaterialOffset {
    pub dn_d: f64,
    pub dv_d: f64,
}

impl MaterialOffset {
    /// Structural bounds on `|Δn_d|` and `|ΔV_d|` (CONVENTIONS §13).
    pub const MAX_DN_D: f64 = 0.05;
    pub const MAX_DV_D: f64 = 10.0;

    pub fn is_zero(&self) -> bool {
        self.dn_d == 0.0 && self.dv_d == 0.0
    }
}

/// Wavelengths of the standard spectral lines (µm).
pub mod lines {
    pub const D: f64 = 0.5875618; // He d
    pub const F: f64 = 0.4861327; // H F
    pub const C: f64 = 0.6562725; // H C
    pub const G: f64 = 0.4358343; // Hg g
    pub const E: f64 = 0.5460740; // Hg e
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransmittancePoint {
    pub um: f64,
    /// Internal transmittance (excluding Fresnel losses) of a sample of `thickness_mm`.
    pub tau: f64,
    pub thickness_mm: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Glass {
    pub name: String,
    pub dispersion: Dispersion,
    /// Catalog `n_d` and `V_d`, used for display and to check consistency
    /// against `dispersion`.
    pub nd: f64,
    pub vd: f64,
    /// Valid wavelength range of the dispersion data (µm).
    pub min_um: f64,
    pub max_um: f64,
    #[serde(default)]
    pub thermal: Option<ThermalCoefficients>,
    /// Coefficient of thermal expansion (1/K) for −30…+70 °C.
    #[serde(default)]
    pub cte: Option<f64>,
    /// Density (g/cm³).
    #[serde(default)]
    pub density: Option<f64>,
    #[serde(default)]
    pub transmittance: Vec<TransmittancePoint>,
    /// Relative cost or availability code, used by the glass-substitution optimiser.
    #[serde(default)]
    pub relative_cost: Option<f64>,
    /// Availability and resistance data retained when loading an AGF catalog.
    #[serde(default, skip_serializing_if = "GlassSubstitution::is_default")]
    pub substitution: GlassSubstitution,
    #[serde(default)]
    pub comment: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct GlassSubstitution {
    pub status: agf::GlassStatus,
    pub exclude_substitution: bool,
    pub climate_resistance: Option<f64>,
    pub stain_resistance: Option<f64>,
}

impl GlassSubstitution {
    fn is_default(&self) -> bool {
        *self == Self::default()
    }
}

impl Glass {
    /// Index relative to air at catalog conditions.
    pub fn catalog_index(&self, lambda_um: f64) -> f64 {
        self.dispersion.index(lambda_um)
    }

    /// Conditions at which `catalog_index` holds: `T_ref` and `CATALOG_P_ATM`.
    pub fn catalog_environment(&self) -> Environment {
        Environment {
            temperature_c: self.thermal.map_or(CATALOG_T_C, |t| t.t_ref_c),
            pressure_atm: CATALOG_P_ATM,
        }
    }

    /// Internal absorption coefficient α (1/mm), so that `τ(L) = exp(−α L)`.
    /// Returns 0 when no transmittance data exists.
    pub fn absorption_per_mm(&self, lambda_um: f64) -> f64 {
        let t = &self.transmittance;
        if t.is_empty() {
            return 0.0;
        }
        let alpha = |p: &TransmittancePoint| -(p.tau.max(1e-300)).ln() / p.thickness_mm;
        if lambda_um <= t[0].um {
            return alpha(&t[0]);
        }
        for w in t.windows(2) {
            if lambda_um <= w[1].um {
                let f = (lambda_um - w[0].um) / (w[1].um - w[0].um);
                return alpha(&w[0]) * (1.0 - f) + alpha(&w[1]) * f;
            }
        }
        alpha(&t[t.len() - 1])
    }
}

#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct GlassCatalog {
    pub name: String,
    pub glasses: BTreeMap<String, Glass>,
}

impl GlassCatalog {
    /// Loads an AGF catalog file in any common encoding (UTF-8, Latin-1,
    /// UTF-16 LE/BE), named after the file stem, with the default
    /// skip-or-keep policy. Availability and climate/stain resistance are
    /// retained in `Glass::substitution`; use `agf::read_agf_file` to also
    /// keep warnings and the complete AGF metadata.
    pub fn from_agf_file(path: impl AsRef<std::path::Path>) -> MkResult<Self> {
        Ok(agf::read_agf_file(path, &agf::AgfOptions::default())?.catalog)
    }
}

/// Every catalog loaded into a session. Lookups are case-insensitive.
#[derive(Clone, Debug, Default)]
pub struct MaterialLibrary {
    catalogs: Vec<Arc<GlassCatalog>>,
}

/// A material resolved for one environment, so it can be evaluated quickly
/// at any wavelength.
#[derive(Clone, Debug)]
pub enum ResolvedMaterial {
    Air,
    Vacuum {
        env: Environment,
    },
    Glass {
        glass: Arc<Glass>,
        env: Environment,
    },
    Model {
        cauchy: [f64; 3],
    },
    Constant {
        n: f64,
    },
    /// A catalogue glass with an offset: `n_d + Δn_d + scale·(n_base(λ) − n_d)`.
    Offset {
        base: Box<ResolvedMaterial>,
        nd: f64,
        dn_d: f64,
        scale: f64,
    },
}

impl ResolvedMaterial {
    /// Index relative to air at the system environment (CONVENTIONS §2).
    pub fn index(&self, lambda_um: f64) -> f64 {
        match self {
            ResolvedMaterial::Air => 1.0,
            ResolvedMaterial::Vacuum { env } => {
                1.0 / air_index(lambda_um, env.temperature_c, env.pressure_atm)
            }
            ResolvedMaterial::Constant { n } => *n,
            ResolvedMaterial::Offset {
                base,
                nd,
                dn_d,
                scale,
            } => nd + dn_d + scale * (base.index(lambda_um) - nd),
            ResolvedMaterial::Model { cauchy } => {
                cauchy[0] + cauchy[1] / (lambda_um * lambda_um) + cauchy[2] / lambda_um.powi(4)
            }
            ResolvedMaterial::Glass { glass, env } => {
                let n_cat = glass.catalog_index(lambda_um);
                // The catalog index is relative to air at the glass's own
                // reference conditions: T_ref from its thermal data (else
                // CATALOG_T_C) and CATALOG_P_ATM. There ΔT = 0, so it is returned as is.
                let t_ref = glass.thermal.map_or(CATALOG_T_C, |th| th.t_ref_c);
                if env.temperature_c == t_ref && env.pressure_atm == CATALOG_P_ATM {
                    return n_cat;
                }
                let n_air_ref = air_index(lambda_um, t_ref, CATALOG_P_ATM);
                let n_abs_ref = n_cat * n_air_ref;
                let dn = glass.thermal.map_or(0.0, |th| {
                    th.delta_n_abs(n_abs_ref, lambda_um, env.temperature_c)
                });
                (n_abs_ref + dn) / air_index(lambda_um, env.temperature_c, env.pressure_atm)
            }
        }
    }

    pub fn absorption_per_mm(&self, lambda_um: f64) -> f64 {
        match self {
            ResolvedMaterial::Glass { glass, .. } => glass.absorption_per_mm(lambda_um),
            ResolvedMaterial::Offset { base, .. } => base.absorption_per_mm(lambda_um),
            _ => 0.0,
        }
    }

    /// Valid wavelength range, if the material has one.
    pub fn range_um(&self) -> Option<(f64, f64)> {
        match self {
            ResolvedMaterial::Glass { glass, .. } => Some((glass.min_um, glass.max_um)),
            ResolvedMaterial::Offset { base, .. } => base.range_um(),
            _ => None,
        }
    }
}

/// Normal-glass line for the relative partial dispersion `P_g,F` against `V_d`
/// (the line through the K7 and F2 reference glasses).
pub fn normal_pgf(vd: f64) -> f64 {
    0.6438 - 0.001682 * vd
}

/// Fits a Cauchy formula `n = A + B/λ² + C/λ⁴` to a model glass so that it
/// reproduces `n_d`, `n_F − n_C = (n_d − 1)/V_d`, and
/// `n_g − n_F = (P_gF,normal + ΔP_gF)(n_F − n_C)`.
pub fn model_glass_cauchy(nd: f64, vd: f64, dpgf: f64) -> MkResult<[f64; 3]> {
    if !(nd > 1.0 && vd > 0.0) {
        return Err(MkError::InvalidModel(format!(
            "model glass needs nd > 1 and vd > 0 (got {nd}, {vd})"
        )));
    }
    let dfc = (nd - 1.0) / vd;
    let pgf = normal_pgf(vd) + dpgf;
    let u = |l: f64| 1.0 / (l * l);
    let w = |l: f64| 1.0 / l.powi(4);
    // Unknowns A, B, C:
    //   A + B u_d + C w_d               = nd
    //       B(u_F−u_C) + C(w_F−w_C)     = dfc
    //       B(u_g−u_F) + C(w_g−w_F)     = pgf·dfc
    let (a11, a12, b1) = (u(lines::F) - u(lines::C), w(lines::F) - w(lines::C), dfc);
    let (a21, a22, b2) = (
        u(lines::G) - u(lines::F),
        w(lines::G) - w(lines::F),
        pgf * dfc,
    );
    let det = a11 * a22 - a12 * a21;
    let b = (b1 * a22 - a12 * b2) / det;
    let c = (a11 * b2 - a21 * b1) / det;
    let a = nd - b * u(lines::D) - c * w(lines::D);
    Ok([a, b, c])
}

impl MaterialLibrary {
    pub fn new() -> Self {
        Self::default()
    }

    /// The library with only the built-in catalog loaded.
    pub fn with_builtin() -> Self {
        static BUILTIN: OnceLock<Arc<GlassCatalog>> = OnceLock::new();
        let mut lib = Self::new();
        lib.add_catalog(BUILTIN.get_or_init(|| Arc::new(builtin::catalog())).clone());
        lib
    }

    pub fn add_catalog(&mut self, cat: Arc<GlassCatalog>) {
        self.catalogs
            .retain(|c| !c.name.eq_ignore_ascii_case(&cat.name));
        self.catalogs.push(cat);
    }

    pub fn catalogs(&self) -> &[Arc<GlassCatalog>] {
        &self.catalogs
    }

    pub fn find_glass(&self, catalog: Option<&str>, name: &str) -> MkResult<Arc<Glass>> {
        for cat in &self.catalogs {
            if let Some(c) = catalog
                && !cat.name.eq_ignore_ascii_case(c)
            {
                continue;
            }
            if let Some(g) = cat.glasses.get(name) {
                return Ok(Arc::new(g.clone()));
            }
            if let Some((_, g)) = cat
                .glasses
                .iter()
                .find(|(k, _)| k.eq_ignore_ascii_case(name))
            {
                return Ok(Arc::new(g.clone()));
            }
        }
        Err(MkError::UnknownMaterial(match catalog {
            Some(c) => format!("{c}:{name}"),
            None => name.to_string(),
        }))
    }

    pub fn resolve(&self, m: &MaterialRef, env: &Environment) -> MkResult<ResolvedMaterial> {
        Ok(match m {
            MaterialRef::Air => ResolvedMaterial::Air,
            MaterialRef::Vacuum => ResolvedMaterial::Vacuum { env: *env },
            MaterialRef::Constant { n } => {
                if !(*n > 0.0 && n.is_finite()) {
                    return Err(MkError::InvalidModel(format!(
                        "constant index must be positive (got {n})"
                    )));
                }
                ResolvedMaterial::Constant { n: *n }
            }
            MaterialRef::Model { nd, vd, dpgf } => ResolvedMaterial::Model {
                cauchy: model_glass_cauchy(*nd, *vd, *dpgf)?,
            },
            MaterialRef::Catalog {
                catalog,
                name,
                offset,
            } => {
                let base = ResolvedMaterial::Glass {
                    glass: self.find_glass(catalog.as_deref(), name)?,
                    env: *env,
                };
                if offset.is_zero() {
                    return Ok(base);
                }
                // Anchors at the requested environment (MATERIALS.md).
                let nd = base.index(lines::D);
                let vd = (nd - 1.0) / (base.index(lines::F) - base.index(lines::C));
                let (nd_new, vd_new) = (nd + offset.dn_d, vd + offset.dv_d);
                if !(nd > 1.0 && nd_new > 1.0 && vd > 0.0 && vd_new > 0.0 && vd.is_finite()) {
                    return Err(MkError::InvalidModel(format!(
                        "offset glass {name} needs n_d > 1 and finite V_d > 0 \
                         (got {nd}, {vd} → {nd_new}, {vd_new})"
                    )));
                }
                ResolvedMaterial::Offset {
                    base: Box::new(base),
                    nd,
                    dn_d: offset.dn_d,
                    scale: (nd_new - 1.0) / (nd - 1.0) * vd / vd_new,
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_glass_reproduces_nd_vd() {
        let c = model_glass_cauchy(1.6, 40.0, 0.0).unwrap();
        let m = ResolvedMaterial::Model { cauchy: c };
        assert!((m.index(lines::D) - 1.6).abs() < 1e-12);
        let vd = (m.index(lines::D) - 1.0) / (m.index(lines::F) - m.index(lines::C));
        assert!((vd - 40.0).abs() < 1e-9);
        let pgf = (m.index(lines::G) - m.index(lines::F)) / (m.index(lines::F) - m.index(lines::C));
        assert!((pgf - normal_pgf(40.0)).abs() < 1e-9);
    }

    #[test]
    fn vacuum_below_air() {
        let lib = MaterialLibrary::with_builtin();
        let v = lib
            .resolve(&MaterialRef::Vacuum, &Environment::default())
            .unwrap();
        assert!(v.index(0.55) < 1.0 && v.index(0.55) > 0.9996);
    }

    /// Catalog data stated at T_ref = 25 °C with zero dn/dT: the index at
    /// 25 °C, 1 atm is the catalog index exactly, and at 20 °C it differs only
    /// by the change of the air index, n_cat·n_air(25 °C)/n_air(20 °C).
    #[test]
    fn thermal_reference_temperature_is_honoured() {
        let mut g = builtin::catalog().glasses["N-BK7"].clone();
        g.thermal = Some(ThermalCoefficients {
            d0: 0.0,
            d1: 0.0,
            d2: 0.0,
            e0: 0.0,
            e1: 0.0,
            lambda_tk_um: 0.17,
            t_ref_c: 25.0,
        });
        let glass = Arc::new(g);
        let at = |t: f64| ResolvedMaterial::Glass {
            glass: glass.clone(),
            env: Environment {
                temperature_c: t,
                ..Environment::default()
            },
        };
        let l = lines::D;
        let n_cat = glass.catalog_index(l);
        assert_eq!(at(25.0).index(l), n_cat);
        let expect = n_cat * air_index(l, 25.0, 1.0) / air_index(l, 20.0, 1.0);
        let n20 = at(20.0).index(l);
        assert!((n20 - expect).abs() < 1e-15, "{n20} vs {expect}");
        // Air at 25 °C is less refractive than at 20 °C, by about 4.6·10⁻⁶.
        assert!(n_cat - n20 > 5e-6 && n_cat - n20 < 1e-5);
    }

    #[test]
    fn thermal_identity_at_reference() {
        let lib = MaterialLibrary::with_builtin();
        let g = lib
            .resolve(&MaterialRef::glass("N-BK7"), &Environment::default())
            .unwrap();
        let n = g.index(lines::D);
        assert!((n - 1.5168).abs() < 2e-5, "N-BK7 nd = {n}");
    }
}
