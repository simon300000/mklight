//! Explicit ghost and Monte-Carlo stray merit operands. The stray seed is
//! a setting, never a candidate hash: neighbouring candidates use common
//! random numbers.

use super::mtf_eval::{fields, operand_result, wavelengths};
use super::operand::{
    Capability, GhostFocusMetric, GhostMetric, GhostPair, OperandKind, StrayPathClass,
};
use super::{FieldSelector, Operand, OperandEvaluator, OperandValue, TraceEvaluator};
use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, MkError, MkResult, OpticalSystem};
use mklight_ghost::irradiance::{IrradianceSettings, ghost_irradiance};
use mklight_ghost::paraxial::{ParaxialGhostSettings, paraxial_ghosts};
use mklight_ghost::path::EnumerationSettings;
use mklight_ghost::{GhostModel, GhostPath};
use mklight_stray::analysis::{StrayAnalysisSettings, StraySource, analyze};
use mklight_stray::path::{ClassMask, PathClass};
use mklight_trace::{FieldPoint, SequentialTracer};
use std::sync::Arc;

#[derive(Clone)]
pub struct SolverEvaluator {
    trace: TraceEvaluator,
    pub ghost: IrradianceSettings,
    pub stray: StrayAnalysisSettings,
}

impl SolverEvaluator {
    pub const CAPABILITIES: [Capability; 2] = [Capability::Ghost, Capability::StrayLight];
    pub fn new(library: Arc<MaterialLibrary>) -> Self {
        Self {
            trace: TraceEvaluator::new(library),
            ghost: IrradianceSettings {
                pupil_samples: 32,
                ..Default::default()
            },
            stray: StrayAnalysisSettings {
                detector_pixels: (1, 1),
                top_signatures: 0,
                ..Default::default()
            },
        }
    }
    pub fn builtin() -> Self {
        Self::new(Arc::new(MaterialLibrary::with_builtin()))
    }

    fn candidate(
        &self,
        s: &OpticalSystem,
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<OperandValue>> {
        let tr = match self.trace.tracer(s) {
            Ok(t) => Arc::new(t),
            Err(e) => return Ok(vec![Err(e); ops.len()]),
        };
        // One ghost model per candidate, built on first use.
        let mut model = None;
        let n = ops.len() as f64;
        ops.iter()
            .enumerate()
            .map(|(i, op)| {
                ctx.check()?;
                let sub = ctx.subrange(i as f64 / n, (i + 1) as f64 / n);
                let v = match op.kind.capability() {
                    Capability::StrayLight => self.stray(&tr, s, op, &sub),
                    _ => ghost_model(&mut model, &tr).and_then(|m| self.ghost(m, s, op, &sub)),
                };
                sub.report(1.0, "ghost / stray merit");
                operand_result(v)
            })
            .collect()
    }

    fn ghost(
        &self,
        model: &GhostModel,
        s: &OpticalSystem,
        op: &Operand,
        ctx: &JobContext,
    ) -> MkResult<f64> {
        let wls = wavelengths(model.tracer(), op.wavelength, &[])?;
        match &op.kind {
            OperandKind::GhostFocus {
                pair,
                path,
                worst,
                metric,
                max_reflections,
            } => {
                let paths = explicit_paths(s, pair, path)?;
                // Distance averages over the spectrum; peak ranks at the
                // primary wavelength and integrates the selected spectrum.
                let ranking = match metric {
                    GhostFocusMetric::Distance => wls.clone(),
                    GhostFocusMetric::Peak => vec![(s.wavelengths.primary, 1.0)],
                };
                let mut value = 0.0;
                for (wl, weight) in ranking {
                    let report = paraxial_ghosts(
                        model,
                        &ParaxialGhostSettings {
                            paths: paths.clone(),
                            wavelength: Some(wl),
                            enumeration: EnumerationSettings {
                                max_reflections: *max_reflections,
                                ..Default::default()
                            },
                            ..Default::default()
                        },
                    )?;
                    let take = if paths.is_some() { usize::MAX } else { *worst };
                    let selected: Vec<_> = report.ghosts.iter().take(take).collect();
                    if selected.is_empty() {
                        return Err(MkError::Numerical("no ghost paths selected".into()));
                    }
                    value += weight
                        * match metric {
                            GhostFocusMetric::Distance => selected
                                .iter()
                                .map(|g| g.focus_from_sensor.abs())
                                .fold(f64::INFINITY, f64::min),
                            GhostFocusMetric::Peak => {
                                let settings = IrradianceSettings {
                                    paths: Some(selected.iter().map(|g| g.path.clone()).collect()),
                                    wavelengths: wls.iter().map(|w| w.0).collect(),
                                    ..self.ghost.clone()
                                };
                                ghost_irradiance(model, &settings, ctx)?
                                    .ghosts
                                    .iter()
                                    .map(|g| g.peak_ratio)
                                    .fold(0.0, f64::max)
                            }
                        };
                }
                Ok(value)
            }
            OperandKind::GhostIrradiance {
                pair,
                path,
                metric,
                orders,
                max_reflections,
            } => {
                let mut settings = self.ghost.clone();
                settings.enumeration.max_reflections = *max_reflections;
                settings.paths = explicit_paths(s, pair, path)?;
                if !orders.is_empty() && settings.paths.is_none() {
                    let all = mklight_ghost::path::enumerate(model, &settings.enumeration)?;
                    settings.paths = Some(
                        all.paths
                            .into_iter()
                            .filter(|p| orders.contains(&(p.path.order() as u32)))
                            .map(|p| p.path)
                            .collect(),
                    );
                }
                settings.wavelengths = wls.iter().map(|w| w.0).collect();
                match op.field {
                    FieldSelector::All => {}
                    FieldSelector::Index { index } => settings.fields = vec![index],
                    FieldSelector::Normalized { hx, hy } => {
                        settings.points = vec![FieldPoint::normalized(hx, hy)];
                    }
                }
                let r = ghost_irradiance(model, &settings, ctx)?;
                Ok(match metric {
                    GhostMetric::Peak => r.total_peak_ratio,
                    GhostMetric::Mean | GhostMetric::Power => r.total_flux_ratio,
                })
            }
            _ => unreachable!("routed by capability"),
        }
    }

    /// Field-weighted MC stray fraction with the operand's fixed seed.
    fn stray(
        &self,
        tr: &SequentialTracer,
        s: &OpticalSystem,
        op: &Operand,
        ctx: &JobContext,
    ) -> MkResult<f64> {
        let OperandKind::StrayLightPower { class, rays, seed } = op.kind else {
            unreachable!("routed by capability")
        };
        // Apply the spectral selection at the source.
        let mut rs = tr.system().clone();
        rs.wavelength_weights.fill(0.0);
        for (wl, w) in wavelengths(tr, op.wavelength, &[])? {
            rs.wavelength_weights[wl] = w;
        }
        let rs = Arc::new(rs);
        let classes = match class {
            StrayPathClass::Ghost => ClassMask(PathClass::Ghost.bit()),
            StrayPathClass::Scatter => ClassMask(PathClass::Scatter.bit()),
            StrayPathClass::Mechanical => ClassMask(PathClass::Mechanical.bit()),
            StrayPathClass::Sneak => ClassMask(PathClass::Sneak.bit()),
            StrayPathClass::AllNonPrimary => ClassMask::ALL_STRAY,
        };
        let (mut value, mut total) = (0.0, 0.0);
        for (field, weight) in fields(s, op.field) {
            let source = match field {
                FieldSelector::Index { index } => StraySource::SystemField {
                    index,
                    margin: 0.1,
                    strength: 1.0,
                },
                FieldSelector::Normalized { hx, hy } => StraySource::Field {
                    hx,
                    hy,
                    margin: 0.1,
                    strength: 1.0,
                },
                FieldSelector::All => unreachable!("expanded by `fields`"),
            };
            let settings = StrayAnalysisSettings {
                rays,
                seed,
                classes,
                source,
                ..self.stray.clone()
            };
            let fraction = analyze(rs.clone(), &settings, ctx)?
                .stray_fraction
                .ok_or_else(|| MkError::Numerical("zero primary reference flux".into()))?;
            value += weight * fraction;
            total += weight;
        }
        Ok(value / total)
    }
}

fn ghost_model<'m>(
    slot: &'m mut Option<GhostModel>,
    tr: &Arc<SequentialTracer>,
) -> MkResult<&'m GhostModel> {
    if slot.is_none() {
        *slot = Some(GhostModel::new(tr.clone(), Default::default())?);
    }
    Ok(slot.as_ref().unwrap())
}

/// An explicit `path`, or the two-reflection path of a surface `pair`
/// (reflection at `second`, then `first`).
fn explicit_paths(
    s: &OpticalSystem,
    pair: &Option<GhostPair>,
    path: &Option<GhostPath>,
) -> MkResult<Option<Vec<GhostPath>>> {
    match (pair, path) {
        (Some(_), Some(_)) => Err(MkError::InvalidParam("choose ghost pair or path".into())),
        (None, Some(p)) => Ok(Some(vec![p.clone()])),
        (None, None) => Ok(None),
        (Some(p), None) => {
            let index = |id| {
                s.surface_index(id).map(|i| i as u32).ok_or_else(|| {
                    MkError::InvalidParam(format!("ghost surface {id:?} does not exist"))
                })
            };
            Ok(Some(vec![GhostPath::new(vec![
                index(p.second)?,
                index(p.first)?,
            ])]))
        }
    }
}

impl OperandEvaluator for SolverEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        Self::CAPABILITIES.contains(&kind.capability())
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        ops: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        systems
            .iter()
            .map(|s| self.candidate(s, ops, ctx))
            .collect()
    }
}
