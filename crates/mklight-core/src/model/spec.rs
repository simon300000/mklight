use serde::{Deserialize, Serialize};

/// How the system aperture is specified.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(
    remote = "Self",
    tag = "type",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum ApertureSpec {
    /// Entrance pupil diameter (mm).
    EntrancePupilDiameter { value: f64 },
    /// Paraxial image-space F/# for an object at infinity (`f'/EPD`).
    ImageSpaceFNumber { value: f64 },
    /// Object-space numerical aperture `n·sinθ` of the marginal ray (finite objects).
    ObjectSpaceNa { value: f64 },
    /// Paraxial working F/# (`1/(2 n' sin θ')`).
    ParaxialWorkingFNumber { value: f64 },
    /// The stop's fixed semi-diameter sets the aperture.
    FloatByStopSize,
}

crate::serde_util::strict_unit_variants!(ApertureSpec, "type", ["float_by_stop_size"]);

impl Default for ApertureSpec {
    fn default() -> Self {
        ApertureSpec::EntrancePupilDiameter { value: 10.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FieldKind {
    /// Field angle in degrees in object space (object at infinity or finite).
    #[default]
    ObjectAngle,
    /// Object height in mm (finite object only).
    ObjectHeight,
    /// Paraxial image height in mm.
    ParaxialImageHeight,
    /// Real image height in mm (solved by real ray tracing).
    RealImageHeight,
}

/// Vignetting factors (CONVENTIONS §3).
#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Vignetting {
    #[serde(default)]
    pub vdx: f64,
    #[serde(default)]
    pub vdy: f64,
    #[serde(default)]
    pub vcx: f64,
    #[serde(default)]
    pub vcy: f64,
}

impl Vignetting {
    pub fn is_none(&self) -> bool {
        *self == Vignetting::default()
    }
    /// Apply to normalised pupil coordinates.
    pub fn apply(&self, px: f64, py: f64) -> (f64, f64) {
        (
            self.vdx + (1.0 - self.vcx) * px,
            self.vdy + (1.0 - self.vcy) * py,
        )
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Field {
    pub x: f64,
    pub y: f64,
    #[serde(default = "one")]
    pub weight: f64,
    #[serde(default)]
    pub vignetting: Vignetting,
}

fn one() -> f64 {
    1.0
}

impl Field {
    pub fn new(x: f64, y: f64) -> Self {
        Field {
            x,
            y,
            weight: 1.0,
            vignetting: Vignetting::default(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FieldSet {
    pub kind: FieldKind,
    pub fields: Vec<Field>,
}

impl Default for FieldSet {
    fn default() -> Self {
        FieldSet {
            kind: FieldKind::ObjectAngle,
            fields: vec![Field::new(0.0, 0.0)],
        }
    }
}

impl FieldSet {
    /// Maximum radial field. It is the normaliser for `(Hx, Hy)`, and is `1.0`
    /// when every field is on axis, which keeps normalisation defined.
    pub fn max_radial(&self) -> f64 {
        let m = self
            .fields
            .iter()
            .map(|f| f.x.hypot(f.y))
            .fold(0.0, f64::max);
        if m > 0.0 { m } else { 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Wavelength {
    /// Micrometres.
    pub um: f64,
    #[serde(default = "one")]
    pub weight: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WavelengthSet {
    pub wavelengths: Vec<Wavelength>,
    /// Index into `wavelengths`.
    pub primary: usize,
}

impl Default for WavelengthSet {
    /// F, d, C visible set with d as the primary.
    fn default() -> Self {
        WavelengthSet {
            wavelengths: vec![
                Wavelength {
                    um: 0.4861327,
                    weight: 1.0,
                },
                Wavelength {
                    um: 0.5875618,
                    weight: 1.0,
                },
                Wavelength {
                    um: 0.6562725,
                    weight: 1.0,
                },
            ],
            primary: 1,
        }
    }
}

impl WavelengthSet {
    pub fn primary_um(&self) -> f64 {
        self.wavelengths[self.primary].um
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Default, Serialize, Deserialize)]
#[serde(
    remote = "Self",
    tag = "mode",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum RayAiming {
    /// Pupil coordinates map onto the paraxial entrance pupil.
    #[default]
    Off,
    /// Iteratively aim real rays so that `(Px, Py)` maps onto the stop.
    Real {
        /// Convergence tolerance on the stop, as a fraction of the stop
        /// radius. The model default ([`RayAiming::DEFAULT_TOLERANCE`]) lets
        /// the run's aim-stop policy derive the stop; any other value is an
        /// explicit request and is honoured exactly (CONVENTIONS §7).
        #[serde(default = "aim_tol")]
        tolerance: f64,
    },
}

crate::serde_util::strict_unit_variants!(RayAiming, "mode", ["off"]);

fn aim_tol() -> f64 {
    RayAiming::DEFAULT_TOLERANCE
}

impl RayAiming {
    /// The model default of `Real { tolerance }`; serialised files state it,
    /// so it reads the same whether written by hand or by default.
    pub const DEFAULT_TOLERANCE: f64 = 1e-10;

    /// The tolerance of real aiming when it is an explicit request: any
    /// value other than the model default. `None` for the default (which
    /// an aim-stop policy may derive per field, never tighter) and for
    /// aiming off.
    pub fn explicit_tolerance(&self) -> Option<f64> {
        match *self {
            RayAiming::Real { tolerance } if tolerance != Self::DEFAULT_TOLERANCE => {
                Some(tolerance)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Environment {
    pub temperature_c: f64,
    pub pressure_atm: f64,
}

impl Default for Environment {
    fn default() -> Self {
        Environment {
            temperature_c: 20.0,
            pressure_atm: 1.0,
        }
    }
}
