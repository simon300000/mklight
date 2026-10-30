use serde::{Deserialize, Serialize};

/// Reference to an optical medium. Resolution into numbers happens in
/// `materials::MaterialLibrary`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    remote = "Self",
    tag = "type",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum MaterialRef {
    /// Air at the system temperature and pressure (index 1 by convention, CONVENTIONS §2).
    Air,
    /// Vacuum (index `1/n_air`).
    Vacuum,
    /// A glass from a named catalog. `catalog: None` searches all loaded catalogs
    /// in load order.
    Catalog {
        #[serde(default)]
        catalog: Option<String>,
        name: String,
    },
    /// Model glass: `n_d`, Abbe `V_d` and an optional partial-dispersion
    /// deviation `ΔP_g,F`. Continuous, so it can be a variable during optimisation.
    Model {
        nd: f64,
        vd: f64,
        #[serde(default)]
        dpgf: f64,
    },
    /// Constant index at every wavelength (ideal/test media).
    Constant { n: f64 },
}

crate::serde_util::strict_unit_variants!(MaterialRef, "type", ["air", "vacuum"]);

impl MaterialRef {
    pub fn glass(name: &str) -> Self {
        MaterialRef::Catalog {
            catalog: None,
            name: name.to_string(),
        }
    }
    pub fn is_air(&self) -> bool {
        matches!(self, MaterialRef::Air)
    }
    pub fn display_name(&self) -> String {
        match self {
            MaterialRef::Air => "AIR".into(),
            MaterialRef::Vacuum => "VACUUM".into(),
            MaterialRef::Catalog { name, .. } => name.clone(),
            MaterialRef::Model { nd, vd, .. } => format!("{nd:.5}/{vd:.2}"),
            MaterialRef::Constant { n } => format!("n={n}"),
        }
    }
}
