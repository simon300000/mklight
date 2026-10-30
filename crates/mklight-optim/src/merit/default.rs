//! Default merit-function generator: RMS spot or RMS wavefront over a
//! Gaussian rings/arms pupil, for every (configuration, field, wavelength),
//! plus optional glass/air thickness boundaries.
//!
//! The spot criterion emits one `RayAberration` operand per pupil node and
//! axis with weight `W · w_field · w_λ · w_node / (Σw_field · Σw_λ · n_cfg)`,
//! so that `Σ w ε²` is `W` times the weighted mean of the squared RMS spot
//! radius (the quadrature weights average over the pupil, see
//! [`quadrature`](super::quadrature)). The wavefront criterion does the same
//! with `WavefrontAberration` (waves), giving the mean squared RMS wavefront
//! error. One residual per ray keeps the least-squares structure that LM
//! exploits.

use serde::{Deserialize, Serialize};

use mklight_core::model::{Interaction, MaterialRef, SemiDiameter};
use mklight_core::{MkError, MkResult, OpticalSystem};

use super::operand::{Axis, OperandKind, SpotReference, WavefrontReference};
use super::quadrature::{PupilSampling, PupilSymmetry};
use super::{MeritFunctionSpec, Operand};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DefaultCriterion {
    RmsSpot {
        #[serde(default)]
        reference: SpotReference,
    },
    Wavefront {
        #[serde(default)]
        reference: WavefrontReference,
    },
}

impl Default for DefaultCriterion {
    fn default() -> Self {
        DefaultCriterion::RmsSpot {
            reference: SpotReference::Centroid,
        }
    }
}

/// Thickness boundary values (mm). `None` disables a boundary.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThicknessBoundaries {
    pub glass_min_centre: Option<f64>,
    pub glass_max_centre: Option<f64>,
    pub glass_min_edge: Option<f64>,
    pub air_min_centre: Option<f64>,
    pub air_max_centre: Option<f64>,
    pub air_min_edge: Option<f64>,
    /// Emit explicit constraints (default) instead of one-sided residuals.
    pub as_constraints: bool,
    /// Weight when emitted as residuals.
    pub weight: f64,
}

impl Default for ThicknessBoundaries {
    fn default() -> Self {
        ThicknessBoundaries {
            glass_min_centre: Some(1.0),
            glass_max_centre: Some(30.0),
            glass_min_edge: Some(1.0),
            air_min_centre: Some(0.1),
            air_max_centre: None,
            air_min_edge: Some(0.1),
            as_constraints: true,
            weight: 1.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DefaultMeritSpec {
    #[serde(default)]
    pub criterion: DefaultCriterion,
    #[serde(default)]
    pub sampling: PupilSampling,
    /// Switch to a half pupil when the system is symmetric about the
    /// meridional plane.
    #[serde(default = "yes")]
    pub auto_symmetry: bool,
    /// Field indices (`None` = all with non-zero weight).
    #[serde(default)]
    pub fields: Option<Vec<usize>>,
    #[serde(default)]
    pub wavelengths: Option<Vec<usize>>,
    /// Configurations (empty = `[0]`).
    #[serde(default)]
    pub configs: Vec<usize>,
    #[serde(default)]
    pub thickness: Option<ThicknessBoundaries>,
    #[serde(default = "one")]
    pub weight: f64,
}

fn yes() -> bool {
    true
}
fn one() -> f64 {
    1.0
}

impl Default for DefaultMeritSpec {
    fn default() -> Self {
        DefaultMeritSpec {
            criterion: DefaultCriterion::default(),
            sampling: PupilSampling::default(),
            auto_symmetry: true,
            fields: None,
            wavelengths: None,
            configs: Vec::new(),
            thickness: Some(ThicknessBoundaries::default()),
            weight: 1.0,
        }
    }
}

/// Is the system mirror-symmetric about the y–z plane (all fields on the y
/// axis, no x decenters, no tilts about y or z, no x vignetting decenter)?
pub fn is_meridionally_symmetric(s: &OpticalSystem) -> bool {
    s.fields
        .fields
        .iter()
        .all(|f| f.x == 0.0 && f.vignetting.vdx == 0.0)
        && s.surfaces.iter().all(|sf| {
            sf.placement
                .is_none_or(|p| p.decenter_x == 0.0 && p.tilt_y == 0.0 && p.tilt_z == 0.0)
        })
}

impl DefaultMeritSpec {
    pub fn generate(&self, s: &OpticalSystem) -> MkResult<MeritFunctionSpec> {
        let fields: Vec<usize> = match &self.fields {
            Some(v) => v.clone(),
            None => (0..s.fields.fields.len())
                .filter(|&i| s.fields.fields[i].weight > 0.0)
                .collect(),
        };
        let wls: Vec<usize> = match &self.wavelengths {
            Some(v) => v.clone(),
            None => (0..s.wavelengths.wavelengths.len())
                .filter(|&i| s.wavelengths.wavelengths[i].weight > 0.0)
                .collect(),
        };
        if fields.iter().any(|&f| f >= s.fields.fields.len())
            || wls.iter().any(|&w| w >= s.wavelengths.wavelengths.len())
        {
            return Err(MkError::InvalidParam(
                "default merit: field or wavelength index out of range".into(),
            ));
        }
        let configs = if self.configs.is_empty() {
            vec![0]
        } else {
            self.configs.clone()
        };
        let mut sampling = self.sampling;
        if self.auto_symmetry && is_meridionally_symmetric(s) {
            sampling.symmetry = PupilSymmetry::Bilateral;
        }
        let nodes = sampling.nodes();
        let fw: f64 = fields.iter().map(|&f| s.fields.fields[f].weight).sum();
        let ww: f64 = wls
            .iter()
            .map(|&w| s.wavelengths.wavelengths[w].weight)
            .sum();
        let norm = fw * ww * configs.len() as f64;
        if !(norm > 0.0) {
            return Err(MkError::InvalidParam(
                "default merit: no field/wavelength with positive weight".into(),
            ));
        }
        let mut ops = Vec::new();
        for &c in &configs {
            for &f in &fields {
                for &w in &wls {
                    let base = self.weight
                        * s.fields.fields[f].weight
                        * s.wavelengths.wavelengths[w].weight
                        / norm;
                    for nd in &nodes {
                        let kinds: Vec<OperandKind> = match self.criterion {
                            DefaultCriterion::RmsSpot { reference } => [Axis::X, Axis::Y]
                                .into_iter()
                                .map(|axis| OperandKind::RayAberration {
                                    axis,
                                    px: nd.px,
                                    py: nd.py,
                                    reference,
                                })
                                .collect(),
                            DefaultCriterion::Wavefront { reference } => {
                                vec![OperandKind::WavefrontAberration {
                                    px: nd.px,
                                    py: nd.py,
                                    reference,
                                }]
                            }
                        };
                        for k in kinds {
                            ops.push(
                                Operand::new(k)
                                    .weight(base * nd.weight)
                                    .at_field(f)
                                    .at_wavelength(w)
                                    .in_config(c),
                            );
                        }
                    }
                }
            }
        }
        if let Some(tb) = &self.thickness {
            ops.extend(thickness_operands(s, tb, &configs));
        }
        Ok(MeritFunctionSpec {
            name: "default".into(),
            operands: ops,
            ..Default::default()
        })
    }
}

/// Boundary operands for every gap between the first optical surface and
/// the image. Edge boundaries are emitted only where a bounding surface has a
/// fixed semi-diameter (automatic ones need the tracer — integration point).
pub fn thickness_operands(
    s: &OpticalSystem,
    tb: &ThicknessBoundaries,
    configs: &[usize],
) -> Vec<Operand> {
    let n = s.surfaces.len();
    let mut ops = Vec::new();
    let mut medium = s.surfaces[0].material.clone();
    let mk = |kind: OperandKind, lo: Option<f64>, hi: Option<f64>, c: usize| {
        let mut v = Vec::new();
        for (bound, is_lo) in [(lo, true), (hi, false)] {
            if let Some(b) = bound {
                let mut op = Operand::new(kind.clone()).in_config(c);
                op = if is_lo {
                    op.greater_than(b)
                } else {
                    op.less_than(b)
                };
                op = if tb.as_constraints {
                    op.as_constraint()
                } else {
                    op.weight(tb.weight)
                };
                v.push(op);
            }
        }
        v
    };
    for i in 1..n.saturating_sub(1) {
        let sf = &s.surfaces[i];
        if sf.interaction == Interaction::Refract && !sf.is_coordinate_break() {
            medium = sf.material.clone();
        }
        if sf.is_coordinate_break() || i + 1 == n - 1 {
            // The last gap (to the image) is the back focal distance, not a
            // manufacturable thickness.
            continue;
        }
        let air = matches!(medium, MaterialRef::Air | MaterialRef::Vacuum);
        let (cmin, cmax, emin) = if air {
            (tb.air_min_centre, tb.air_max_centre, tb.air_min_edge)
        } else {
            (tb.glass_min_centre, tb.glass_max_centre, tb.glass_min_edge)
        };
        let fixed = |k: usize| matches!(s.surfaces[k].semi_diameter, SemiDiameter::Fixed { .. });
        for &c in configs {
            ops.extend(mk(
                OperandKind::CentreThickness { surface: sf.id },
                cmin,
                cmax,
                c,
            ));
            if fixed(i) || fixed(i + 1) {
                ops.extend(mk(
                    OperandKind::EdgeThickness {
                        surface: sf.id,
                        height: None,
                    },
                    emin,
                    None,
                    c,
                ));
            }
        }
    }
    ops
}
