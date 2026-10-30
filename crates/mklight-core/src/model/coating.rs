use serde::{Deserialize, Serialize};

use crate::model::material::MaterialRef;

/// One thin-film layer.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
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

/// A named coating, used by surfaces through `Surface::coating`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum CoatingDef {
    /// Multilayer stack. Layers are listed **outward from the substrate**
    /// (`layers[0]` touches the substrate). The substrate is the side with the
    /// higher index at the reference wavelength. On a mirror it is the medium
    /// behind the mirror, given by `mirror_substrate`. The TMM code orders the
    /// stack by the side the light arrives from, so one coating definition
    /// works on both faces of a lens.
    ThinFilm {
        reference_wavelength_um: f64,
        layers: Vec<Layer>,
        /// Complex index `(n, k)` of the reflector behind a coated mirror
        /// (for example aluminium). Ignored on refracting surfaces.
        #[serde(default)]
        mirror_substrate: Option<(f64, f64)>,
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
