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
//!
//! Every ray operand carries the [`RayGroup`] (the full sampling, symmetry
//! included) it belongs to. With a `Centroid` reference the group defines the
//! reference, so a tracer-backed evaluator computes the centroid /
//! piston-tilt over the right — mirrored, for a half pupil — ray set (see
//! [`super::reference`]); with either reference it tells the evaluator that
//! the operand is part of a half pupil.
//!
//! **Half pupil.** With `auto_symmetry` the bilateral half pupil is used only
//! when the system is mirror-symmetric about the meridional plane in *every*
//! emitted configuration and no variable can break that symmetry (x
//! decenters, tilts about y or z, x field coordinates, XY or Zernike
//! coefficients); a system that is not symmetric always gets the full pupil.
//! Anamorphic and freeform systems sample a 3×3 x/y field grid
//! (`docs/OPTIMIZATION.md`). Use
//! [`DefaultMeritSpec::generate_for`] to pass the variables, and
//! [`DefaultMeritSpec::generate_for_tolerancing`] for a tolerancing
//! criterion (whose perturbations are the "variables"). Should a half-pupil
//! merit still meet an asymmetric candidate, the tracer-backed evaluator
//! traces the mirrored nodes too and reports pair values that keep `Σ w r²`
//! equal to the full-pupil merit (see [`RayGroup`]), at twice the ray
//! count.
//!
//! Thickness boundaries are placed on the physical gaps between optically
//! active surfaces ([`optical_gaps`]), never on gaps bounded by a
//! stop, dummy or coordinate break, and never on the back-focus gap.

use serde::{Deserialize, Serialize};

use mklight_core::model::SemiDiameter;
use mklight_core::param::{CoefficientSet, ParamPath};
use mklight_core::surface_math::{Symmetry, system_symmetry};
use mklight_core::{MkError, MkResult, OpticalSystem};

use super::geometric::optical_gaps;
use super::operand::{
    Axis, CentroidScope, OperandKind, RayGroup, SpotReference, WavefrontReference,
};
use super::quadrature::{PupilSampling, PupilSymmetry};
use super::{FieldSelector, MeritFunctionSpec, Operand, VariableSpec};
use crate::tolerance::ToleranceSpec;

pub use super::mtf_default::MtfCriterion;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DefaultCriterion {
    Mtf(MtfCriterion),
    RmsSpot {
        #[serde(default)]
        reference: SpotReference,
        /// Centroid per wavelength or of the white-light spot.
        #[serde(default)]
        scope: CentroidScope,
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
            scope: CentroidScope::Monochromatic,
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
    /// Also emit edge boundaries on gaps bounded only by surfaces with
    /// automatic semi-diameters. Their height is the clear semi-diameter
    /// computed by the reference tracer for each candidate, so the spec then
    /// needs a tracer-backed evaluator
    /// ([`TraceEvaluator`](super::trace_eval::TraceEvaluator)); the
    /// model-only evaluator reports such operands as invalid.
    #[serde(default)]
    pub auto_semi_diameters: bool,
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
            auto_semi_diameters: false,
        }
    }
}

impl ThicknessBoundaries {
    /// The default boundaries with edge thicknesses on every gap, sized by
    /// the tracer's automatic semi-diameters where none are fixed.
    pub fn with_auto_semi_diameters() -> Self {
        ThicknessBoundaries {
            auto_semi_diameters: true,
            ..Default::default()
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
    /// Optional signed EFL_y/EFL_x equality target at the primary wavelength.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub squeeze_target: Option<f64>,
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
            squeeze_target: None,
        }
    }
}

/// Is the system mirror-symmetric about the y–z plane (core
/// [`system_symmetry`] not `None`, all fields on the y axis, no x vignetting
/// decenter)?
pub fn is_meridionally_symmetric(s: &OpticalSystem) -> bool {
    system_symmetry(s) != Symmetry::None
        && s.fields
            .fields
            .iter()
            .all(|f| f.x == 0.0 && f.vignetting.vdx == 0.0)
}

/// Can a variable on this parameter break meridional symmetry?
pub fn breaks_meridional_symmetry(p: &ParamPath) -> bool {
    matches!(
        p,
        ParamPath::ElementPlacement { .. }
            | ParamPath::DecenterX { .. }
            | ParamPath::TiltY { .. }
            | ParamPath::TiltZ { .. }
            | ParamPath::FieldX { .. }
            | ParamPath::ShapeCoefficient {
                set: CoefficientSet::Xy | CoefficientSet::Zernike,
                ..
            }
    )
}

impl DefaultMeritSpec {
    /// Sample the 3×3 x/y field grid instead of the system fields? Yes for
    /// anamorphic and freeform shape types, whatever their coefficients.
    fn uses_field_grid(&self, s: &OpticalSystem) -> bool {
        use mklight_core::SurfaceShape::*;
        self.fields.is_none()
            && s.surfaces.iter().any(|sf| {
                matches!(
                    sf.shape,
                    Toroid { .. } | Biconic { .. } | XYPolynomial { .. } | ZernikeSag { .. }
                )
            })
    }

    /// The configurations this spec emits operands for.
    fn configs(&self) -> Vec<usize> {
        if self.configs.is_empty() {
            vec![0]
        } else {
            self.configs.clone()
        }
    }

    /// Are the system and the sampled fields mirror-symmetric about the
    /// meridional plane in every emitted configuration?
    fn meridional(&self, s: &OpticalSystem) -> MkResult<bool> {
        if self.uses_field_grid(s) {
            return Ok(false);
        }
        for c in self.configs() {
            if !is_meridionally_symmetric(&s.with_configuration(c)?) {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// May the half pupil be used for `s` with these variables?
    pub fn half_pupil_allowed(
        &self,
        s: &OpticalSystem,
        variables: &[VariableSpec],
    ) -> MkResult<bool> {
        Ok(self.auto_symmetry
            && !variables
                .iter()
                .any(|v| breaks_meridional_symmetry(&v.param))
            && self.meridional(s)?)
    }

    /// Generate for a system without optimisation variables.
    pub fn generate(&self, s: &OpticalSystem) -> MkResult<MeritFunctionSpec> {
        self.generate_for(s, &[])
    }

    /// Generate a tolerancing criterion: the toleranced parameters and the
    /// compensators count as variables, so a tolerance that can break the
    /// meridional symmetry (x decenter, y or z tilt) selects the full pupil.
    pub fn generate_for_tolerancing(
        &self,
        s: &OpticalSystem,
        tolerances: &ToleranceSpec,
    ) -> MkResult<MeritFunctionSpec> {
        let vars: Vec<VariableSpec> = tolerances
            .operands
            .iter()
            .map(|o| &o.param)
            .chain(tolerances.compensators.iter().map(|c| &c.param))
            .map(|p| VariableSpec::new(p.clone()))
            .collect();
        self.generate_for(s, &vars)
    }

    /// Generate for a system that will be optimised over `variables` (which
    /// decide, with the configurations, whether the half pupil is valid).
    pub fn generate_for(
        &self,
        s: &OpticalSystem,
        variables: &[VariableSpec],
    ) -> MkResult<MeritFunctionSpec> {
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
        let configs = self.configs();
        let mut sampling = self.sampling;
        if self.half_pupil_allowed(s, variables)? {
            sampling.symmetry = PupilSymmetry::Bilateral;
        } else if !self.meridional(s)? {
            sampling.symmetry = PupilSymmetry::None;
        }
        let nodes = sampling.nodes();
        let field_points: Vec<(FieldSelector, f64)> = if self.uses_field_grid(s) {
            let h = std::f64::consts::FRAC_1_SQRT_2;
            [-h, 0.0, h]
                .into_iter()
                .flat_map(|hx| {
                    [-h, 0.0, h]
                        .into_iter()
                        .map(move |hy| (FieldSelector::Normalized { hx, hy }, 1.0))
                })
                .collect()
        } else {
            fields
                .iter()
                .map(|&f| (FieldSelector::Index { index: f }, s.fields.fields[f].weight))
                .collect()
        };
        let fw: f64 = field_points.iter().map(|(_, w)| w).sum();
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
        if let DefaultCriterion::Mtf(mtf) = &self.criterion {
            let ops = mtf.operands(&field_points, &wls, &configs, self.weight)?;
            return self.finish(s, &configs, ops);
        }
        let mut ops = Vec::new();
        for &c in &configs {
            for &(field, field_weight) in &field_points {
                for &w in &wls {
                    let base =
                        self.weight * field_weight * s.wavelengths.wavelengths[w].weight / norm;
                    for nd in &nodes {
                        let kinds: Vec<OperandKind> = match &self.criterion {
                            DefaultCriterion::RmsSpot { reference, scope } => {
                                let group = Some(RayGroup {
                                    sampling,
                                    scope: *scope,
                                });
                                [Axis::X, Axis::Y]
                                    .into_iter()
                                    .map(|axis| OperandKind::RayAberration {
                                        axis,
                                        px: nd.px,
                                        py: nd.py,
                                        reference: *reference,
                                        group,
                                    })
                                    .collect()
                            }
                            DefaultCriterion::Wavefront { reference } => {
                                let group = Some(RayGroup::new(sampling));
                                vec![OperandKind::WavefrontAberration {
                                    px: nd.px,
                                    py: nd.py,
                                    reference: *reference,
                                    group,
                                }]
                            }
                            DefaultCriterion::Mtf(_) => unreachable!(),
                        };
                        for k in kinds {
                            let mut op = Operand::new(k)
                                .weight(base * nd.weight)
                                .at_wavelength(w)
                                .in_config(c);
                            op.field = field;
                            ops.push(op);
                        }
                    }
                }
            }
        }
        self.finish(s, &configs, ops)
    }

    fn finish(
        &self,
        s: &OpticalSystem,
        configs: &[usize],
        mut ops: Vec<Operand>,
    ) -> MkResult<MeritFunctionSpec> {
        if let Some(target) = self.squeeze_target {
            for &c in configs {
                ops.push(
                    Operand::new(OperandKind::AnamorphicSqueeze)
                        .target(target)
                        .as_constraint()
                        .in_config(c),
                );
            }
        }
        if let Some(tb) = &self.thickness {
            ops.extend(thickness_operands(s, tb, configs));
        }
        Ok(MeritFunctionSpec {
            name: "default".into(),
            operands: ops,
            ..Default::default()
        })
    }
}

/// Boundary operands for every physical gap between optically active
/// surfaces except the back-focus gap to the image
/// ([`optical_gaps`]). Edge boundaries are emitted where a bounding
/// surface has a fixed semi-diameter, and on every gap with
/// [`ThicknessBoundaries::auto_semi_diameters`] (the height then comes from
/// the tracer's clear semi-diameters).
pub fn thickness_operands(
    s: &OpticalSystem,
    tb: &ThicknessBoundaries,
    configs: &[usize],
) -> Vec<Operand> {
    let n = s.surfaces.len();
    let mut ops = Vec::new();
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
    let fixed = |k: usize| matches!(s.surfaces[k].semi_diameter, SemiDiameter::Fixed { .. });
    for g in optical_gaps(s) {
        if g.to == n - 1 {
            // The back focal distance, not a manufacturable thickness.
            continue;
        }
        let (cmin, cmax, emin) = if g.is_air() {
            (tb.air_min_centre, tb.air_max_centre, tb.air_min_edge)
        } else {
            (tb.glass_min_centre, tb.glass_max_centre, tb.glass_min_edge)
        };
        let surface = s.surfaces[g.from].id;
        let to = (g.to != g.from + 1).then_some(s.surfaces[g.to].id);
        for &c in configs {
            ops.extend(mk(
                OperandKind::CentreThickness { surface, to },
                cmin,
                cmax,
                c,
            ));
            if tb.auto_semi_diameters || fixed(g.from) || fixed(g.to) {
                ops.extend(mk(
                    OperandKind::EdgeThickness {
                        surface,
                        to,
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
