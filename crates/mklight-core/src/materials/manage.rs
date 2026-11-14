//! Managing glass catalogs: availability overrides, reload reports, glasses
//! made from a specification (user glasses), and the table rows and details
//! a catalog browser shows. Everything here works on in-memory catalogs and
//! JSON-friendly types, so every front end (native or web) uses the same
//! logic.

use serde::{Deserialize, Serialize};

use crate::error::{MkError, MkResult};
use crate::materials::agf::{
    self, AgfCatalog, AgfWarning, GlassStatus, computed_nd_vd, dispersion_from_agf,
    dispersion_to_agf,
};
use crate::materials::{
    Dispersion, Glass, GlassCatalog, ThermalCoefficients, TransmittancePoint, lines,
    model_glass_cauchy, normal_pgf,
};

// ---------------------------------------------------------------------------
// Availability overrides
// ---------------------------------------------------------------------------

/// A user's availability setting for one catalog glass: its status
/// (preferred, standard, obsolete, …) and whether glass substitution may use
/// it. A `None` field keeps the catalog's value. Applied to the material
/// library (`Glass::substitution`), so the glass map and glass substitution
/// both see it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GlassOverride {
    pub catalog: String,
    pub name: String,
    #[serde(default)]
    pub status: Option<GlassStatus>,
    #[serde(default)]
    pub exclude_substitution: Option<bool>,
}

/// `cat` with the overrides that name it applied (catalog and glass names
/// are case-insensitive), or `None` when none applies.
pub fn apply_overrides(cat: &GlassCatalog, overrides: &[GlassOverride]) -> Option<GlassCatalog> {
    let mine: Vec<&GlassOverride> = overrides
        .iter()
        .filter(|o| o.catalog.eq_ignore_ascii_case(&cat.name))
        .collect();
    if mine.is_empty() {
        return None;
    }
    let mut out = cat.clone();
    let mut changed = false;
    for o in mine {
        let key = out
            .glasses
            .keys()
            .find(|k| k.eq_ignore_ascii_case(&o.name))
            .cloned();
        let Some(g) = key.and_then(|k| out.glasses.get_mut(&k)) else {
            continue;
        };
        let before = g.substitution;
        if let Some(s) = o.status {
            g.substitution.status = s;
        }
        if let Some(e) = o.exclude_substitution {
            g.substitution.exclude_substitution = e;
        }
        changed |= g.substitution != before;
    }
    changed.then_some(out)
}

// ---------------------------------------------------------------------------
// Reload report
// ---------------------------------------------------------------------------

/// One glass whose data differ between two versions of a catalog.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlassChange {
    pub name: String,
    pub nd_before: f64,
    pub nd_after: f64,
    pub vd_before: f64,
    pub vd_after: f64,
    /// What changed: any of `nd`, `vd`, `dispersion`, `range`, `thermal`,
    /// `transmittance`, `availability`, `cost`, `density`, `expansion`,
    /// `comment`, `other`.
    pub fields: Vec<String>,
}

/// What reloading a catalog changed (glass names in catalog order).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
pub struct CatalogChanges {
    pub catalog: String,
    pub added: Vec<String>,
    pub removed: Vec<String>,
    pub changed: Vec<GlassChange>,
    pub unchanged: usize,
}

impl CatalogChanges {
    pub fn is_empty(&self) -> bool {
        self.added.is_empty() && self.removed.is_empty() && self.changed.is_empty()
    }
}

/// The differences between two versions of a catalog, by glass name.
pub fn diff_catalogs(before: &GlassCatalog, after: &GlassCatalog) -> CatalogChanges {
    let mut out = CatalogChanges {
        catalog: after.name.clone(),
        ..Default::default()
    };
    for (k, a) in &after.glasses {
        match before.glasses.get(k) {
            None => out.added.push(a.name.clone()),
            Some(b) if b == a => out.unchanged += 1,
            Some(b) => {
                let mut fields = Vec::new();
                let mut note = |cond: bool, what: &str| {
                    if cond {
                        fields.push(what.to_string());
                    }
                };
                note(b.nd.to_bits() != a.nd.to_bits(), "nd");
                note(b.vd.to_bits() != a.vd.to_bits(), "vd");
                note(b.dispersion != a.dispersion, "dispersion");
                note(b.min_um != a.min_um || b.max_um != a.max_um, "range");
                note(b.thermal != a.thermal, "thermal");
                note(b.transmittance != a.transmittance, "transmittance");
                note(b.substitution != a.substitution, "availability");
                note(b.relative_cost != a.relative_cost, "cost");
                note(b.density != a.density, "density");
                note(b.cte != a.cte, "expansion");
                note(b.comment != a.comment, "comment");
                if fields.is_empty() {
                    fields.push("other".into());
                }
                out.changed.push(GlassChange {
                    name: a.name.clone(),
                    nd_before: b.nd,
                    nd_after: a.nd,
                    vd_before: b.vd,
                    vd_after: a.vd,
                    fields,
                });
            }
        }
    }
    out.removed = before
        .glasses
        .iter()
        .filter(|(k, _)| !after.glasses.contains_key(*k))
        .map(|(_, g)| g.name.clone())
        .collect();
    out
}

// ---------------------------------------------------------------------------
// Glasses from a specification
// ---------------------------------------------------------------------------

/// A model glass: `n_d`, `V_d` and the deviation `ΔP_g,F` from the normal line.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelGlassSpec {
    pub nd: f64,
    pub vd: f64,
    #[serde(default)]
    pub dpgf: f64,
}

/// A user glass, as an editor describes it. The dispersion is either a model
/// glass or an AGF formula (number 1–13 and its `CD` coefficients). `base`
/// is the glass being edited or duplicated: its other data (thermal
/// coefficients, transmittance, AGF records, resistance classes) are kept.
/// The optional fields here are set exactly as given (`None` clears them).
#[derive(Clone, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct GlassSpec {
    pub name: String,
    pub model: Option<ModelGlassSpec>,
    pub formula: Option<u32>,
    pub coefficients: Vec<f64>,
    /// Valid wavelength range (µm). Defaults: the base glass's, else
    /// 0.35–1.1 µm for a model glass and 0.3–2.5 µm for a formula.
    pub min_um: Option<f64>,
    pub max_um: Option<f64>,
    /// g/cm³.
    pub density: Option<f64>,
    /// Thermal expansion −30…+70 °C, in 10⁻⁶/K (as catalogs list it).
    pub tce: Option<f64>,
    pub relative_cost: Option<f64>,
    pub status: GlassStatus,
    pub exclude_substitution: bool,
    pub comment: String,
    pub base: Option<Glass>,
}

/// Builds a catalog glass from `spec`; `n_d` and `V_d` are computed from the
/// dispersion (zero when the d line lies outside the glass's range, as
/// catalogs list IR materials).
pub fn glass_from_spec(spec: &GlassSpec) -> MkResult<Glass> {
    let name = spec.name.trim();
    if name.is_empty() {
        return Err(MkError::InvalidParam("a glass needs a name".into()));
    }
    if name.chars().any(char::is_whitespace) {
        return Err(MkError::InvalidParam(format!(
            "glass name '{name}' contains whitespace (AGF names are single words)"
        )));
    }
    let (dispersion, default_range) = match (&spec.model, spec.formula) {
        (Some(m), None) => {
            let [a, b, c] = model_glass_cauchy(m.nd, m.vd, m.dpgf)?;
            (Dispersion::Cauchy { a, b, c }, (0.35, 1.1))
        }
        (None, Some(f)) => {
            if spec.coefficients.iter().any(|x| !x.is_finite()) {
                return Err(MkError::InvalidParam(
                    "dispersion coefficients must be finite".into(),
                ));
            }
            (dispersion_from_agf(f, &spec.coefficients)?, (0.3, 2.5))
        }
        (None, None) => match &spec.base {
            Some(b) => (b.dispersion.clone(), (b.min_um, b.max_um)),
            None => {
                return Err(MkError::InvalidParam(
                    "a glass needs a model (n_d, V_d) or a dispersion formula".into(),
                ));
            }
        },
        (Some(_), Some(_)) => {
            return Err(MkError::InvalidParam(
                "give either a model glass or a dispersion formula, not both".into(),
            ));
        }
    };
    let base_range = spec.base.as_ref().map(|b| (b.min_um, b.max_um));
    let (lo, hi) = (
        spec.min_um
            .unwrap_or(base_range.map_or(default_range.0, |r| r.0)),
        spec.max_um
            .unwrap_or(base_range.map_or(default_range.1, |r| r.1)),
    );
    if !(lo.is_finite() && hi.is_finite() && lo >= 0.0 && hi > lo) {
        return Err(MkError::InvalidParam(format!(
            "wavelength range {lo}…{hi} µm needs 0 ≤ min < max"
        )));
    }
    let (mut nd, mut vd) = (0.0, 0.0);
    if (lo..=hi).contains(&lines::D) {
        let (n, v) = computed_nd_vd(&dispersion);
        if !(n.is_finite() && v.is_finite() && n > 1.0 && v > 0.0) {
            return Err(MkError::InvalidModel(format!(
                "the dispersion gives n_d = {n}, V_d = {v} (needs n_d > 1, V_d > 0)"
            )));
        }
        (nd, vd) = (n, v);
    }
    for (what, v) in [
        ("density", spec.density),
        ("thermal expansion", spec.tce),
        ("relative cost", spec.relative_cost),
    ] {
        if v.is_some_and(|x| !x.is_finite()) {
            return Err(MkError::InvalidParam(format!("{what} must be finite")));
        }
    }
    let mut g = spec.base.clone().unwrap_or_else(|| Glass {
        name: String::new(),
        dispersion: dispersion.clone(),
        nd,
        vd,
        min_um: lo,
        max_um: hi,
        thermal: None,
        cte: None,
        density: None,
        transmittance: Vec::new(),
        relative_cost: None,
        substitution: Default::default(),
        comment: String::new(),
        agf: None,
    });
    g.name = name.to_string();
    g.dispersion = dispersion;
    (g.nd, g.vd) = (nd, vd);
    (g.min_um, g.max_um) = (lo, hi);
    g.density = spec.density.filter(|d| *d > 0.0);
    g.cte = spec.tce.map(|t| t / 1e6);
    g.relative_cost = spec.relative_cost.filter(|c| *c > 0.0);
    g.substitution.status = spec.status;
    g.substitution.exclude_substitution = spec.exclude_substitution;
    g.comment = spec.comment.clone();
    if let Some(m) = g.agf.as_deref_mut() {
        // The formula number follows the dispersion; a stale ΔP_g,F of the
        // base glass would contradict an edited dispersion.
        m.formula = dispersion_to_agf(&g.dispersion).map_or(0, |(f, _)| f);
        if spec.model.is_some() || spec.formula.is_some() {
            m.dpgf = None;
        }
        m.ignore_thermal_expansion &= g.cte.is_none();
        m.tce_m30_70 = spec.tce;
    }
    Ok(g)
}

/// The catalog as AGF text, and the writer's warnings (glasses it could not
/// express). Built on [`agf::write_agf`].
pub fn catalog_to_agf(cat: &GlassCatalog) -> (String, Vec<AgfWarning>) {
    agf::write_agf(&AgfCatalog::from_catalog(cat.clone()))
}

// ---------------------------------------------------------------------------
// Browser rows and details
// ---------------------------------------------------------------------------

/// One row of a glass table: the catalog values with what a browser sorts
/// and filters by. Optional values are `None` when the catalog has none.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlassRow {
    pub catalog: String,
    pub name: String,
    /// Catalog `n_d`, `V_d` (display values).
    pub nd: f64,
    pub vd: f64,
    /// `P_g,F = (n_g − n_F)/(n_F − n_C)` and its deviation from the normal
    /// line, from the dispersion formula; `None` outside the glass's range.
    pub pgf: Option<f64>,
    pub dpgf: Option<f64>,
    pub min_um: f64,
    pub max_um: f64,
    /// The status in effect (after the user's overrides).
    pub status: GlassStatus,
    pub exclude_substitution: bool,
    /// Whether the user's overrides changed status or exclusion.
    pub overridden: bool,
    pub density: Option<f64>,
    pub relative_cost: Option<f64>,
    /// Thermal expansion −30…+70 °C, 10⁻⁶/K.
    pub tce: Option<f64>,
    pub climate_resistance: Option<f64>,
    pub stain_resistance: Option<f64>,
    /// The wavelengths the internal-transmittance data span (µm).
    pub transmission_min_um: Option<f64>,
    pub transmission_max_um: Option<f64>,
    /// The shortest wavelength at which 10 mm of the glass transmits at
    /// least 80 % internally, from the transmittance data.
    pub lambda80_um: Option<f64>,
}

fn finite(x: f64) -> Option<f64> {
    x.is_finite().then_some(x)
}

fn partial_dispersion(g: &Glass) -> (Option<f64>, Option<f64>) {
    if !(g.min_um <= lines::G && g.max_um >= lines::C) {
        return (None, None);
    }
    let n = |l| g.catalog_index(l);
    let dfc = n(lines::F) - n(lines::C);
    let pgf = (n(lines::G) - n(lines::F)) / dfc;
    let vd = (n(lines::D) - 1.0) / dfc;
    (finite(pgf), finite(pgf - normal_pgf(vd)))
}

fn lambda80(g: &Glass) -> Option<f64> {
    let t = &g.transmittance;
    let tau10 = |p: &TransmittancePoint| p.tau.max(1e-300).powf(10.0 / p.thickness_mm);
    let first = t.iter().position(|p| tau10(p) >= 0.8)?;
    if first == 0 {
        return Some(t[0].um);
    }
    // Interpolate between the last point below and the first point above.
    let (a, b) = (&t[first - 1], &t[first]);
    let (ta, tb) = (tau10(a), tau10(b));
    let f = if tb > ta { (0.8 - ta) / (tb - ta) } else { 1.0 };
    Some(a.um + f * (b.um - a.um))
}

/// The table row of `glass` (with overrides applied) from `catalog`;
/// `original` is the glass as the catalog has it, before overrides.
pub fn glass_row(catalog: &str, glass: &Glass, original: Option<&Glass>) -> GlassRow {
    let (pgf, dpgf) = partial_dispersion(glass);
    let t = &glass.transmittance;
    GlassRow {
        catalog: catalog.to_string(),
        name: glass.name.clone(),
        nd: glass.nd,
        vd: glass.vd,
        pgf,
        dpgf,
        min_um: glass.min_um,
        max_um: glass.max_um,
        status: glass.substitution.status,
        exclude_substitution: glass.substitution.exclude_substitution,
        overridden: original.is_some_and(|o| o.substitution != glass.substitution),
        density: glass.density,
        relative_cost: glass.relative_cost,
        tce: glass
            .cte
            .map(|c| c * 1e6)
            .or_else(|| glass.agf.as_ref().and_then(|m| m.tce_m30_70)),
        climate_resistance: glass.substitution.climate_resistance,
        stain_resistance: glass.substitution.stain_resistance,
        transmission_min_um: t.first().map(|p| p.um),
        transmission_max_um: t.last().map(|p| p.um),
        lambda80_um: lambda80(glass),
    }
}

/// The index at one named spectral line.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LineIndex {
    pub line: String,
    pub um: f64,
    pub n: f64,
}

/// Everything a browser shows about one glass.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GlassDetails {
    #[serde(flatten)]
    pub row: GlassRow,
    /// The catalog's own status and exclusion (before the user's overrides).
    pub catalog_status: GlassStatus,
    pub catalog_exclude_substitution: bool,
    /// AGF formula number and `CD` coefficients, when the dispersion has one.
    pub formula: Option<u32>,
    pub formula_name: String,
    pub coefficients: Vec<f64>,
    /// `n_d`, `V_d` computed from the dispersion formula.
    pub computed_nd: Option<f64>,
    pub computed_vd: Option<f64>,
    /// A model glass's parameters, when the dispersion is a model glass's fit.
    pub model: Option<ModelGlassSpec>,
    /// ΔP_g,F as the catalog lists it (AGF `ED`).
    pub catalog_dpgf: Option<f64>,
    pub tce_100_300: Option<f64>,
    pub mil_code: Option<String>,
    pub melt_frequency: Option<i32>,
    pub acid_resistance: Option<f64>,
    pub alkali_resistance: Option<f64>,
    pub phosphate_resistance: Option<f64>,
    pub thermal: Option<ThermalCoefficients>,
    pub transmittance: Vec<TransmittancePoint>,
    pub comment: String,
    /// Indices at the standard lines inside the glass's range.
    pub lines: Vec<LineIndex>,
    /// `[λ µm, n]` samples across the glass's range (clipped to 0.25–2.5 µm).
    pub index_curve: Vec<[f64; 2]>,
    /// The glass itself as the catalog has it (before overrides), for
    /// duplicating or editing it as a user glass.
    pub glass: Glass,
}

/// AGF name of formula `f`.
pub fn formula_name(f: u32) -> &'static str {
    match f {
        1 => "Schott",
        2 => "Sellmeier 1",
        3 => "Herzberger",
        4 => "Sellmeier 2",
        5 => "Conrady",
        6 => "Sellmeier 3",
        7 => "Handbook of Optics 1",
        8 => "Handbook of Optics 2",
        9 => "Sellmeier 4",
        10 => "Extended",
        11 => "Sellmeier 5",
        12 => "Extended 2",
        13 => "Extended 3",
        _ => "unknown",
    }
}

/// The model-glass parameters of a Cauchy dispersion: `n_d`, `V_d` and
/// `ΔP_g,F` at the lines the model fit is anchored at.
fn model_of(d: &Dispersion) -> Option<ModelGlassSpec> {
    let Dispersion::Cauchy { .. } = d else {
        return None;
    };
    let n = |l| d.index(l);
    let dfc = n(lines::F) - n(lines::C);
    let nd = n(lines::D);
    let vd = (nd - 1.0) / dfc;
    let dpgf = (n(lines::G) - n(lines::F)) / dfc - normal_pgf(vd);
    Some(ModelGlassSpec { nd, vd, dpgf })
}

/// The details of `glass` (overrides applied) from `catalog`; `original` is
/// the catalog's own glass.
pub fn glass_details(catalog: &str, glass: &Glass, original: &Glass) -> GlassDetails {
    let row = glass_row(catalog, glass, Some(original));
    let (formula, coefficients, formula_name) = match &glass.dispersion {
        Dispersion::Tabulated { .. } => (None, Vec::new(), "Tabulated".to_string()),
        Dispersion::Cauchy { a, b, c } => (None, vec![*a, *b, *c], "Cauchy (model glass)".into()),
        d => match dispersion_to_agf(d) {
            Ok((f, cd)) => (Some(f), cd, self::formula_name(f).to_string()),
            Err(_) => (None, Vec::new(), "Sellmeier".into()),
        },
    };
    let inside = |l: f64| glass.min_um <= l && l <= glass.max_um;
    let (computed_nd, computed_vd) = if inside(lines::D) && inside(lines::F) && inside(lines::C) {
        let (n, v) = computed_nd_vd(&glass.dispersion);
        (finite(n), finite(v))
    } else {
        (None, None)
    };
    let named = [
        ("i", 0.365_015),
        ("h", 0.404_656),
        ("g", lines::G),
        ("F′", 0.479_991),
        ("F", lines::F),
        ("e", lines::E),
        ("d", lines::D),
        ("C′", 0.643_847),
        ("C", lines::C),
        ("r", 0.706_519),
        ("t", 1.013_98),
    ];
    let lines = named
        .iter()
        .filter(|(_, l)| inside(*l))
        .filter_map(|(name, l)| {
            finite(glass.catalog_index(*l)).map(|n| LineIndex {
                line: name.to_string(),
                um: *l,
                n,
            })
        })
        .collect();
    let lo = glass.min_um.max(0.25);
    let hi = glass.max_um.min(2.5);
    let index_curve = if hi > lo {
        (0..=80)
            .filter_map(|k| {
                let l = lo + (hi - lo) * k as f64 / 80.0;
                finite(glass.catalog_index(l)).map(|n| [l, n])
            })
            .collect()
    } else {
        Vec::new()
    };
    let m = glass.agf.as_deref();
    GlassDetails {
        row,
        catalog_status: original.substitution.status,
        catalog_exclude_substitution: original.substitution.exclude_substitution,
        formula,
        formula_name,
        coefficients,
        computed_nd,
        computed_vd,
        model: model_of(&glass.dispersion),
        catalog_dpgf: m.and_then(|m| m.dpgf),
        tce_100_300: m.and_then(|m| m.tce_100_300),
        mil_code: m.and_then(|m| m.mil_code.clone()),
        melt_frequency: m.and_then(|m| m.melt_frequency),
        acid_resistance: m.and_then(|m| m.ordering.acid_resistance),
        alkali_resistance: m.and_then(|m| m.ordering.alkali_resistance),
        phosphate_resistance: m.and_then(|m| m.ordering.phosphate_resistance),
        thermal: glass.thermal,
        transmittance: glass.transmittance.clone(),
        comment: glass.comment.clone(),
        lines,
        index_curve,
        glass: original.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::materials::agf::{AgfOptions, parse_agf_with};

    /// A synthetic two-glass catalog (made-up names and values).
    const MINI: &str = "CC synthetic test catalog\r\n\
NM TESTCROWN 2 517642 1.5168 64.17 0 1 0\r\n\
GC a crown\r\n\
ED 7.1 8.3 2.51 -0.0009 0\r\n\
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653 0 0 0 0\r\n\
TD 1.86e-6 1.31e-8 -1.37e-11 4.34e-7 6.27e-10 0.17 20\r\n\
OD 1.0 1 0 1 2 1.3\r\n\
LD 0.3 2.5\r\n\
IT 0.31 0.2 10\r\n\
IT 0.35 0.7 10\r\n\
IT 0.40 0.95 10\r\n\
NM TESTFLINT 2 620364 1.62004 36.37 0 2 0\r\n\
CD 1.34533359 0.00997743871 0.209073176 0.0470450767 0.937357162 111.886764 0 0 0 0\r\n\
LD 0.32 2.5\r\n";

    fn mini() -> GlassCatalog {
        parse_agf_with("MINI", MINI, &AgfOptions::default())
            .unwrap()
            .catalog
    }

    #[test]
    fn overrides_apply_case_insensitively_and_only_when_they_change_something() {
        let cat = mini();
        let o = |status, excl| GlassOverride {
            catalog: "mini".into(),
            name: "testflint".into(),
            status,
            exclude_substitution: excl,
        };
        let out = apply_overrides(&cat, &[o(Some(GlassStatus::Preferred), Some(true))]).unwrap();
        let g = &out.glasses["TESTFLINT"].substitution;
        assert_eq!(g.status, GlassStatus::Preferred);
        assert!(g.exclude_substitution);
        assert_eq!(out.glasses["TESTCROWN"], cat.glasses["TESTCROWN"]);
        // The catalog's own value, another catalog, an unknown glass: nothing.
        assert!(apply_overrides(&cat, &[o(Some(GlassStatus::Obsolete), None)]).is_none());
        let other = GlassOverride {
            catalog: "OTHER".into(),
            ..o(Some(GlassStatus::Preferred), None)
        };
        assert!(apply_overrides(&cat, &[other]).is_none());
    }

    #[test]
    fn diff_reports_added_removed_and_changed_glasses() {
        let before = mini();
        let mut after = before.clone();
        after.glasses.remove("TESTFLINT");
        let mut g = after.glasses["TESTCROWN"].clone();
        g.nd = 1.5170;
        g.relative_cost = Some(2.0);
        after.glasses.insert("TESTCROWN".into(), g.clone());
        g.name = "NEWGLASS".into();
        after.glasses.insert("NEWGLASS".into(), g);
        let d = diff_catalogs(&before, &after);
        assert_eq!(d.added, ["NEWGLASS"]);
        assert_eq!(d.removed, ["TESTFLINT"]);
        assert_eq!(d.changed.len(), 1);
        let c = &d.changed[0];
        assert_eq!((c.nd_before, c.nd_after), (1.5168, 1.5170));
        assert_eq!(c.fields, ["nd", "cost"]);
        assert_eq!(d.unchanged, 0);
        assert!(diff_catalogs(&before, &before).is_empty());
        assert_eq!(diff_catalogs(&before, &before).unchanged, 2);
    }

    #[test]
    fn model_and_formula_specs_compute_nd_vd() {
        let g = glass_from_spec(&GlassSpec {
            name: "MYMODEL".into(),
            model: Some(ModelGlassSpec {
                nd: 1.6,
                vd: 40.0,
                dpgf: 0.002,
            }),
            density: Some(3.1),
            ..Default::default()
        })
        .unwrap();
        assert!((g.nd - 1.6).abs() < 1e-12 && (g.vd - 40.0).abs() < 1e-9);
        assert_eq!((g.min_um, g.max_um), (0.35, 1.1));
        let d = glass_details("USER", &g, &g);
        let m = d.model.unwrap();
        assert!((m.dpgf - 0.002).abs() < 1e-9, "{m:?}");
        assert!((d.row.dpgf.unwrap() - 0.002).abs() < 1e-9);

        let crown = &mini().glasses["TESTCROWN"];
        let (f, cd) = dispersion_to_agf(&crown.dispersion).unwrap();
        let g = glass_from_spec(&GlassSpec {
            name: "COPY".into(),
            formula: Some(f),
            coefficients: cd,
            tce: Some(7.1),
            base: Some(crown.clone()),
            ..Default::default()
        })
        .unwrap();
        assert!((g.nd - 1.5168).abs() < 2e-5);
        // The base's other data stay.
        assert_eq!(g.thermal, crown.thermal);
        assert_eq!(g.transmittance, crown.transmittance);
        assert_eq!((g.min_um, g.max_um), (0.3, 2.5));
        assert_eq!(g.cte, Some(7.1 / 1e6));

        for bad in [
            GlassSpec::default(),
            GlassSpec {
                name: "two words".into(),
                ..Default::default()
            },
            GlassSpec {
                name: "X".into(),
                formula: Some(14),
                ..Default::default()
            },
            GlassSpec {
                name: "X".into(),
                model: Some(ModelGlassSpec {
                    nd: 0.9,
                    vd: 40.0,
                    dpgf: 0.0,
                }),
                ..Default::default()
            },
        ] {
            assert!(glass_from_spec(&bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn rows_and_details_carry_the_agf_data() {
        let cat = mini();
        let g = &cat.glasses["TESTCROWN"];
        let mut shown = g.clone();
        shown.substitution.status = GlassStatus::Obsolete;
        let d = glass_details("MINI", &shown, g);
        assert_eq!(d.row.status, GlassStatus::Obsolete);
        assert!(d.row.overridden);
        assert_eq!(d.catalog_status, GlassStatus::Preferred);
        assert_eq!(d.formula, Some(2));
        assert_eq!(d.formula_name, "Sellmeier 1");
        assert_eq!(d.catalog_dpgf, Some(-0.0009));
        assert_eq!(d.tce_100_300, Some(8.3));
        assert_eq!(d.mil_code.as_deref(), Some("517642"));
        assert_eq!(d.alkali_resistance, Some(2.0));
        assert_eq!(d.row.tce, Some(7.1));
        assert_eq!(d.row.density, Some(2.51));
        assert_eq!(d.row.transmission_min_um, Some(0.31));
        assert_eq!(d.row.transmission_max_um, Some(0.40));
        // τ10 = 0.7 at 0.35 µm and 0.95 at 0.40 µm: 0.8 is reached at 0.37 µm.
        assert!((d.row.lambda80_um.unwrap() - 0.37).abs() < 1e-12);
        assert!(d.lines.iter().any(|l| l.line == "d"));
        assert_eq!(d.index_curve.len(), 81);
        assert!((d.computed_nd.unwrap() - 1.5168).abs() < 2e-5);
        assert_eq!(d.glass, *g);
    }

    #[test]
    fn agf_write_read_round_trip_keeps_user_and_agf_data() {
        let mut cat = mini();
        let user = glass_from_spec(&GlassSpec {
            name: "MYMODEL".into(),
            model: Some(ModelGlassSpec {
                nd: 1.7,
                vd: 30.0,
                dpgf: 0.0,
            }),
            relative_cost: Some(3.5),
            status: GlassStatus::Preferred,
            exclude_substitution: true,
            comment: "user glass".into(),
            ..Default::default()
        })
        .unwrap();
        cat.glasses.insert(user.name.clone(), user);
        // An overridden status is what gets written (and read back into
        // the AGF data too).
        let flint = cat.glasses.get_mut("TESTFLINT").unwrap();
        flint.substitution.status = GlassStatus::Standard;
        flint.agf.as_mut().unwrap().status = GlassStatus::Standard;
        let (text, warnings) = catalog_to_agf(&cat);
        assert!(warnings.is_empty(), "{warnings:?}");
        let back = parse_agf_with("MINI", &text, &AgfOptions::default()).unwrap();
        assert!(back.warnings.is_empty(), "{:?}", back.warnings);
        let back = back.catalog;
        assert_eq!(back.glasses.len(), 3);
        for k in ["TESTCROWN", "TESTFLINT"] {
            assert_eq!(back.glasses[k], cat.glasses[k], "{k}");
        }
        let (a, b) = (&back.glasses["MYMODEL"], &cat.glasses["MYMODEL"]);
        for l in [lines::G, lines::F, lines::D, lines::C] {
            assert!((a.catalog_index(l) - b.catalog_index(l)).abs() < 1e-12);
        }
        assert_eq!(a.substitution.status, GlassStatus::Preferred);
        assert!(a.substitution.exclude_substitution);
        assert_eq!(a.relative_cost, Some(3.5));
        assert_eq!(a.comment, "user glass");
        // And the JSON form keeps everything exactly.
        let json = serde_json::to_string(&cat).unwrap();
        assert_eq!(serde_json::from_str::<GlassCatalog>(&json).unwrap(), cat);
    }
}
