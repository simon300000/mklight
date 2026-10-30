use serde::{Deserialize, Serialize};

use crate::model::surface::SurfaceId;

/// Surface scatter models (BSDF). Values are per unit projected solid angle,
/// in sr⁻¹, unless stated otherwise.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum ScatterModel {
    /// Lambertian scatter with total integrated scatter `tis` (fraction of the
    /// incident flux). BRDF = `tis/π`.
    Lambertian { tis: f64 },
    /// Harvey–Shack ABg: `BSDF(β−β₀) = A / (B + |β−β₀|^g)`.
    Abg { a: f64, b: f64, g: f64 },
    /// Gaussian microfacet roughness with RMS slope `sigma_slope` (radians) and
    /// total integrated scatter `tis`.
    Gaussian { tis: f64, sigma_slope: f64 },
}

/// Non-optical geometry, in global coordinates, for the stray-light solver.
/// Barrels, spacers and baffles are rotationally symmetric about a surface's
/// local z axis.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum MechanicalPart {
    /// Cylindrical tube between two surfaces' vertex planes. Its axis is the
    /// `from` surface's z.
    Barrel {
        from: SurfaceId,
        to: SurfaceId,
        inner_radius: f64,
        finish: ScatterModel,
    },
    /// Flat annular baffle or aperture plate in a surface's local frame, at
    /// local `z`.
    AnnularBaffle {
        surface: SurfaceId,
        z: f64,
        inner_radius: f64,
        outer_radius: f64,
        finish: ScatterModel,
    },
}

/// Image sensor at the image surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sensor {
    pub pixels_x: u32,
    pub pixels_y: u32,
    /// Pixel pitch (mm).
    pub pitch_x: f64,
    pub pitch_y: f64,
    /// Largest chief-ray angle the microlenses accept (degrees). Used by
    /// design checks, not by clipping.
    #[serde(default)]
    pub max_cra_deg: Option<f64>,
    /// Relative spectral response as `(λ µm, response)` pairs, interpolated
    /// linearly. Empty means a flat response.
    #[serde(default)]
    pub spectral_response: Vec<(f64, f64)>,
}

impl Sensor {
    pub fn half_width(&self) -> f64 {
        0.5 * self.pixels_x as f64 * self.pitch_x
    }
    pub fn half_height(&self) -> f64 {
        0.5 * self.pixels_y as f64 * self.pitch_y
    }
    pub fn response(&self, wavelength_um: f64) -> f64 {
        let r = &self.spectral_response;
        if r.is_empty() {
            return 1.0;
        }
        if wavelength_um <= r[0].0 {
            return r[0].1;
        }
        for w in r.windows(2) {
            let ((x0, y0), (x1, y1)) = (w[0], w[1]);
            if wavelength_um <= x1 {
                let t = (wavelength_um - x0) / (x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }
        r[r.len() - 1].1
    }
}
