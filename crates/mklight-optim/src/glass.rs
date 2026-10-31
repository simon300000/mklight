//! Catalogue candidates for discrete material variables (glass substitution).
//!
//! Glasses are compared at `(n_d, V_d / 100, ΔP_g,F)`, computed from each
//! glass's dispersion formula at catalogue conditions (not its rounded
//! catalogue n_d / V_d), with the squared distance
//! `d² = Δn_d² + (ΔV_d / 100)² + 100 · Δ(ΔP_g,F)²` ([`DISTANCE_WEIGHTS`]):
//! a change of 0.01 in index, 1 in Abbe number or 0.001 in partial-dispersion
//! anomaly counts the same. Candidate sets and the search neighbourhoods
//! both use this metric.

use mklight_core::materials::{
    Glass, MaterialLibrary, ResolvedMaterial, agf::GlassStatus, builtin::BUILTIN_CATALOG_NAME,
    lines, normal_pgf,
};
use mklight_core::{MaterialRef, MkError, MkResult, OpticalSystem, SurfaceId};
use serde::{Deserialize, Serialize};

/// Squared-distance weights on `(n_d, V_d / 100, ΔP_g,F)`.
pub const DISTANCE_WEIGHTS: [f64; 3] = [1.0, 1.0, 100.0];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlassWindow {
    /// Maximum absolute difference in d-line refractive index.
    pub nd: f64,
    /// Maximum absolute difference in Abbe number (unscaled).
    pub vd: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct CandidateSet {
    /// Empty selects all user catalogues, or BUILTIN when none are loaded.
    pub catalogs: Vec<String>,
    pub statuses: Vec<GlassStatus>,
    pub window: Option<GlassWindow>,
    /// Keep the K nearest after the window and quality filters (includes do
    /// not count); null means no limit.
    pub nearest: Option<usize>,
    /// Names or `catalog:name`. Includes bypass the window and `nearest` only.
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub min_transmission: Option<f64>,
    pub max_relative_cost: Option<f64>,
    pub max_climate_class: Option<f64>,
    pub max_stain_class: Option<f64>,
}

impl Default for CandidateSet {
    fn default() -> Self {
        Self {
            catalogs: Vec::new(),
            statuses: vec![
                GlassStatus::Standard,
                GlassStatus::Preferred,
                GlassStatus::Melt,
            ],
            window: None,
            nearest: Some(10),
            include: Vec::new(),
            exclude: Vec::new(),
            min_transmission: None,
            max_relative_cost: None,
            max_climate_class: None,
            max_stain_class: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MaterialVariable {
    /// Stable id of the surface preceding the glass gap.
    pub surface: SurfaceId,
    #[serde(default)]
    pub candidates: CandidateSet,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlassChoice {
    pub catalog: String,
    pub name: String,
    pub nd: f64,
    pub vd: f64,
    pub dpgf: f64,
    /// Beer–Lambert thickness limit, when a transmission filter has data.
    pub max_thickness_mm: Option<f64>,
}

impl GlassChoice {
    /// A replacement names the nominal catalogue glass: melt offsets are cleared.
    pub fn material(&self) -> MaterialRef {
        MaterialRef::Catalog {
            offset: Default::default(),
            catalog: Some(self.catalog.clone()),
            name: self.name.clone(),
        }
    }
    pub fn coordinates(&self) -> Vec<f64> {
        vec![self.nd, self.vd / 100.0, self.dpgf]
    }
}

/// `(n_d, V_d / 100, ΔP_g,F)` of a glass from its dispersion formula.
pub fn coordinates(glass: &Glass) -> [f64; 3] {
    let n = |l| glass.catalog_index(l);
    let dfc = n(lines::F) - n(lines::C);
    let nd = n(lines::D);
    let vd = (nd - 1.0) / dfc;
    [
        nd,
        vd / 100.0,
        (n(lines::G) - n(lines::F)) / dfc - normal_pgf(vd),
    ]
}

fn matches(names: &[String], catalog: &str, name: &str) -> bool {
    names.iter().any(|n| match n.split_once(':') {
        Some((c, n)) => c.eq_ignore_ascii_case(catalog) && n.eq_ignore_ascii_case(name),
        None => n.eq_ignore_ascii_case(name),
    })
}

fn below(value: Option<f64>, limit: Option<f64>) -> bool {
    value.zip(limit).is_none_or(|(v, l)| v <= l)
}

impl CandidateSet {
    /// Resolve a candidate set in nearest-first order. Missing optional quality
    /// data does not exclude a glass. Explicit exclusions and quality/status
    /// filters take precedence over includes. A glass must cover every system
    /// wavelength and have finite visible-line substitution coordinates.
    pub fn resolve(
        &self,
        system: &OpticalSystem,
        surface: SurfaceId,
        library: &MaterialLibrary,
    ) -> MkResult<Vec<GlassChoice>> {
        let index = system.surface_index(surface).ok_or_else(|| {
            MkError::InvalidParam(format!("unknown material surface {}", surface.0))
        })?;
        if index == 0 || index + 1 == system.surfaces.len() {
            return Err(MkError::InvalidParam(
                "material variables must name an optical gap".into(),
            ));
        }
        let sf = &system.surfaces[index];
        let origin = match &sf.material {
            MaterialRef::Model { nd, vd, dpgf } => [*nd, *vd / 100.0, *dpgf],
            MaterialRef::Catalog { catalog, name, .. } => {
                let glass = library.find_glass(catalog.as_deref(), name)?;
                let env = mklight_core::model::Environment {
                    temperature_c: glass
                        .thermal
                        .map_or(mklight_core::materials::CATALOG_T_C, |t| t.t_ref_c),
                    ..Default::default()
                };
                let m = library.resolve(&sf.material, &env)?;
                resolved_coordinates(&m)
            }
            _ => {
                return Err(MkError::InvalidParam(
                    "material substitution needs a catalogue or model glass".into(),
                ));
            }
        };
        let catalogs = library.catalogs();
        for name in &self.catalogs {
            if !catalogs.iter().any(|c| c.name.eq_ignore_ascii_case(name)) {
                return Err(MkError::UnknownMaterial(format!("catalogue {name}")));
            }
        }
        let has_user = catalogs
            .iter()
            .any(|c| !c.name.eq_ignore_ascii_case(BUILTIN_CATALOG_NAME));
        let shortest = system
            .wavelengths
            .wavelengths
            .iter()
            .map(|w| w.um)
            .fold(f64::INFINITY, f64::min);
        let longest = system
            .wavelengths
            .wavelengths
            .iter()
            .map(|w| w.um)
            .fold(0.0, f64::max);
        let mut choices = Vec::new();
        let mut found_includes = vec![false; self.include.len()];
        for cat in catalogs {
            let selected = if self.catalogs.is_empty() {
                !has_user || !cat.name.eq_ignore_ascii_case(BUILTIN_CATALOG_NAME)
            } else {
                self.catalogs
                    .iter()
                    .any(|n| n.eq_ignore_ascii_case(&cat.name))
            };
            if !selected {
                continue;
            }
            for g in cat.glasses.values() {
                for (n, found) in self.include.iter().zip(&mut found_includes) {
                    *found |= matches(std::slice::from_ref(n), &cat.name, &g.name);
                }
                let meta = g.substitution;
                if meta.exclude_substitution
                    || !self.statuses.contains(&meta.status)
                    || matches(&self.exclude, &cat.name, &g.name)
                    || !below(g.relative_cost, self.max_relative_cost)
                    || !below(meta.climate_resistance, self.max_climate_class)
                    || !below(meta.stain_resistance, self.max_stain_class)
                    || shortest < g.min_um
                    || longest > g.max_um
                {
                    continue;
                }
                let max_thickness_mm = self
                    .min_transmission
                    .filter(|&t| t > 0.0)
                    .filter(|_| !g.transmittance.is_empty())
                    .and_then(|t| {
                        let alpha = g.absorption_per_mm(shortest);
                        (alpha > 0.0).then(|| -t.ln() / alpha)
                    });
                if max_thickness_mm.is_some_and(|t| sf.thickness.abs() > t) {
                    continue;
                }
                let coords = coordinates(g);
                if coords.iter().any(|v| !v.is_finite()) || coords[0] <= 1.0 || coords[1] <= 0.0 {
                    continue;
                }
                let include = matches(&self.include, &cat.name, &g.name);
                if !include
                    && self.window.as_ref().is_some_and(|w| {
                        (coords[0] - origin[0]).abs() > w.nd
                            || (coords[1] - origin[1]).abs() * 100.0 > w.vd
                    })
                {
                    continue;
                }
                let distance: f64 = coords
                    .iter()
                    .zip(origin)
                    .zip(DISTANCE_WEIGHTS)
                    .map(|((a, b), w)| w * (a - b).powi(2))
                    .sum();
                choices.push((
                    distance,
                    include,
                    GlassChoice {
                        catalog: cat.name.clone(),
                        name: g.name.clone(),
                        nd: coords[0],
                        vd: coords[1] * 100.0,
                        dpgf: coords[2],
                        max_thickness_mm,
                    },
                ));
            }
        }
        if let Some(i) = found_includes.iter().position(|f| !f) {
            return Err(MkError::UnknownMaterial(self.include[i].clone()));
        }
        choices.sort_by(|a, b| {
            a.0.total_cmp(&b.0)
                .then(a.2.catalog.cmp(&b.2.catalog))
                .then(a.2.name.cmp(&b.2.name))
        });
        // The K nearest, plus the includes wherever they fall.
        let mut nearest = 0;
        let result: Vec<_> = choices
            .into_iter()
            .filter(|(_, include, _)| {
                *include || {
                    nearest += 1;
                    self.nearest.is_none_or(|k| nearest <= k)
                }
            })
            .map(|(_, _, g)| g)
            .collect();
        if result.is_empty() {
            return Err(MkError::InvalidParam(format!(
                "no candidate glasses for surface {}",
                surface.0
            )));
        }
        Ok(result)
    }
}

fn resolved_coordinates(m: &ResolvedMaterial) -> [f64; 3] {
    let [d, f, c, g] = [lines::D, lines::F, lines::C, lines::G].map(|l| m.index(l));
    let vd = (d - 1.0) / (f - c);
    [d, vd / 100.0, (g - f) / (f - c) - normal_pgf(vd)]
}

/// Empty material-variable settings snap every model-glass optical gap.
pub fn model_variables(system: &OpticalSystem) -> Vec<MaterialVariable> {
    system
        .surfaces
        .iter()
        .skip(1)
        .take(system.surfaces.len().saturating_sub(2))
        .filter(|s| matches!(s.material, MaterialRef::Model { .. }))
        .map(|s| MaterialVariable {
            surface: s.id,
            candidates: CandidateSet::default(),
        })
        .collect()
}
