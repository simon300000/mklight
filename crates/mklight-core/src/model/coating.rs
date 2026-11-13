use serde::{Deserialize, Serialize};

use crate::model::material::MaterialRef;

/// One thin-film layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Layer {
    pub material: MaterialRef,
    /// Physical thickness in µm. When `quarter_waves` is set, the thickness is
    /// derived from it at `CoatingDef::reference_wavelength_um` instead.
    #[serde(default)]
    pub thickness_um: f64,
    /// Optical thickness in quarter waves at the reference wavelength and
    /// normal incidence.
    #[serde(default)]
    pub quarter_waves: Option<f64>,
    /// Extinction coefficient `k` (constant). Absorbing films have `k > 0`.
    #[serde(default)]
    pub extinction: f64,
}

/// Which medium of a refracting surface carries a thin-film stack.
///
/// The choice is made once per surface, so the stack keeps its order at every
/// wavelength and angle and for light arriving from either side.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CoatingSubstrate {
    /// The medium with the higher refractive index at the coating's
    /// `reference_wavelength_um` (the medium after the surface when the two
    /// are equal there). Right for ordinary coatings on glass in air.
    #[default]
    HigherIndex,
    /// The medium before the surface in sequence order.
    Before,
    /// The medium after the surface in sequence order.
    After,
}

impl CoatingSubstrate {
    pub fn is_default(&self) -> bool {
        *self == CoatingSubstrate::HigherIndex
    }
}

/// A named coating, used by surfaces through `Surface::coating`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum CoatingDef {
    /// Multilayer stack. Layers are listed **outward from the substrate**
    /// (`layers[0]` touches the substrate). On a refracting surface the
    /// substrate is chosen by `substrate`, once per surface (by default the
    /// medium with the higher index at `reference_wavelength_um`); the
    /// choice holds at every wavelength, even where the two media's
    /// dispersion curves cross. On a mirror it is the medium behind the
    /// mirror, given by `mirror_substrate`. The TMM code orders the stack by
    /// the side the light arrives from, so one coating definition works on
    /// both faces of a lens.
    ThinFilm {
        reference_wavelength_um: f64,
        layers: Vec<Layer>,
        /// Complex index `(n, k)` of the reflector behind a coated mirror
        /// (for example aluminium). Ignored on refracting surfaces.
        #[serde(default)]
        mirror_substrate: Option<(f64, f64)>,
        /// Which medium of a refracting surface the stack sits on. Optional
        /// (default `higher_index`), so older projects load unchanged.
        #[serde(default, skip_serializing_if = "CoatingSubstrate::is_default")]
        substrate: CoatingSubstrate,
    },
    /// Ideal coating with fixed intensity reflectance and transmittance,
    /// independent of angle and wavelength. Absorption is `1 − R − T`.
    Ideal {
        reflectance: f64,
        transmittance: f64,
    },
    /// Tabulated unpolarised reflectance against wavelength (µm) at normal
    /// incidence, applied as a scale on the bare-Fresnel angular dependence.
    /// Energy-conserving, with `T = 1 − R` (no absorption).
    Tabulated { reflectance: Vec<(f64, f64)> },
}
