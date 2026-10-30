//! Explicit ghost and Monte-Carlo stray merit operands. The stray seed is
//! a setting, never a candidate hash: neighbouring candidates use CRN.

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
use mklight_ghost::{GhostModel, GhostPath, GhostSettings};
use mklight_stray::analysis::{StrayAnalysisSettings, StraySource, analyze};
use mklight_stray::path::{ClassMask, PathClass};
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

    fn path(
        s: &OpticalSystem,
        pair: &Option<GhostPair>,
        path: &Option<GhostPath>,
    ) -> MkResult<Option<Vec<GhostPath>>> {
        if pair.is_some() && path.is_some() {
            return Err(MkError::InvalidParam("choose ghost pair or path".into()));
        }
        if let Some(p) = path {
            return Ok(Some(vec![p.clone()]));
        }
        pair.map(|p| {
            let first = s.surface_index(p.first).ok_or_else(|| {
                MkError::InvalidParam("ghost first surface does not exist".into())
            })?;
            let second = s.surface_index(p.second).ok_or_else(|| {
                MkError::InvalidParam("ghost second surface does not exist".into())
            })?;
            Ok(vec![GhostPath::new(vec![second as u32, first as u32])])
        })
        .transpose()
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
        let mut model = None;
        let mut out = Vec::new();
        for (i, op) in ops.iter().enumerate() {
            ctx.check()?;
            let sub = ctx.subrange(
                i as f64 / ops.len() as f64,
                (i + 1) as f64 / ops.len() as f64,
            );
            let value = (|| {
                let wls = wavelengths(&tr, op.wavelength, &[])?;
                match &op.kind {
                    OperandKind::GhostFocus {
                        pair,
                        path,
                        worst,
                        metric,
                        max_reflections,
                    } => {
                        if *worst == 0 {
                            return Err(MkError::InvalidParam(
                                "worst ghost count must be positive".into(),
                            ));
                        }
                        if model.is_none() {
                            model = Some(GhostModel::sized(
                                tr.clone(),
                                GhostSettings::default(),
                                &Default::default(),
                                &sub.subrange(0.0, 0.1),
                            )?);
                        }
                        let model = model.as_ref().unwrap();
                        let paths = Self::path(s, pair, path)?;
                        let ranking_wls = match metric {
                            GhostFocusMetric::Distance => wls.clone(),
                            GhostFocusMetric::Peak => vec![(tr.system().primary_wavelength, 1.0)],
                        };
                        let mut value = 0.0;
                        for (j, &(wl, weight)) in ranking_wls.iter().enumerate() {
                            sub.check()?;
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
                            let selected: Vec<_> = report
                                .ghosts
                                .iter()
                                .take(if paths.is_some() { usize::MAX } else { *worst })
                                .collect();
                            if selected.is_empty() {
                                return Err(MkError::Numerical("no ghost paths selected".into()));
                            }
                            let v = match metric {
                                GhostFocusMetric::Distance => selected
                                    .iter()
                                    .map(|g| g.focus_from_sensor.abs())
                                    .fold(f64::INFINITY, f64::min),
                                GhostFocusMetric::Peak => {
                                    let mut settings = self.ghost.clone();
                                    settings.paths =
                                        Some(selected.iter().map(|g| g.path.clone()).collect());
                                    settings.wavelengths = wls.iter().map(|w| w.0).collect();
                                    let r = ghost_irradiance(
                                        model,
                                        &settings,
                                        &sub.subrange(
                                            0.1 + 0.9 * j as f64 / ranking_wls.len() as f64,
                                            0.1 + 0.9 * (j + 1) as f64 / ranking_wls.len() as f64,
                                        ),
                                    )?;
                                    r.ghosts.iter().map(|g| g.peak_ratio).fold(0.0, f64::max)
                                }
                            };
                            value += weight * v;
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
                        if model.is_none() {
                            model = Some(GhostModel::sized(
                                tr.clone(),
                                GhostSettings::default(),
                                &Default::default(),
                                &sub.subrange(0.0, 0.1),
                            )?);
                        }
                        let mut settings = self.ghost.clone();
                        settings.paths = Self::path(s, pair, path)?;
                        settings.enumeration.max_reflections = *max_reflections;
                        if !orders.is_empty() && settings.paths.is_none() {
                            settings.paths = Some(
                                mklight_ghost::path::enumerate(
                                    model.as_ref().unwrap(),
                                    &settings.enumeration,
                                )?
                                .paths
                                .into_iter()
                                .filter(|p| orders.contains(&(p.path.order() as u32)))
                                .map(|p| p.path)
                                .collect(),
                            );
                        }
                        settings.wavelengths = wls.iter().map(|w| w.0).collect();
                        match op.field {
                            FieldSelector::All => {
                                settings.fields = (0..s.fields.fields.len()).collect();
                                settings.points.clear();
                            }
                            FieldSelector::Index { index } => {
                                settings.fields = vec![index];
                                settings.points.clear();
                            }
                            FieldSelector::Normalized { hx, hy } => {
                                settings.fields.clear();
                                settings.points =
                                    vec![mklight_trace::FieldPoint::normalized(hx, hy)];
                            }
                        }
                        let r = ghost_irradiance(
                            model.as_ref().unwrap(),
                            &settings,
                            &sub.subrange(0.1, 1.0),
                        )?;
                        Ok(match metric {
                            GhostMetric::Peak => r.total_peak_ratio,
                            GhostMetric::Mean | GhostMetric::Power => r.total_flux_ratio,
                        })
                    }
                    OperandKind::StrayLightPower { class, rays, seed } => {
                        // Apply the spectral selection at the source boundary.
                        let mut rs = tr.system().clone();
                        rs.wavelength_weights.fill(0.0);
                        for &(wl, w) in &wls {
                            rs.wavelength_weights[wl] = w;
                        }
                        let rs = Arc::new(rs);
                        let fs = fields(s, op.field);
                        let mut value = 0.0;
                        let mut total = 0.0;
                        for (j, (field, weight)) in fs.iter().enumerate() {
                            let mut settings = self.stray.clone();
                            settings.rays = *rays;
                            settings.seed = *seed;
                            settings.classes = match class {
                                StrayPathClass::Ghost => ClassMask(PathClass::Ghost.bit()),
                                StrayPathClass::Scatter => ClassMask(PathClass::Scatter.bit()),
                                StrayPathClass::Mechanical => {
                                    ClassMask(PathClass::Mechanical.bit())
                                }
                                StrayPathClass::Sneak => ClassMask(PathClass::Sneak.bit()),
                                StrayPathClass::AllNonPrimary => ClassMask::ALL_STRAY,
                            };
                            settings.source = match field {
                                FieldSelector::Index { index } => StraySource::SystemField {
                                    index: *index,
                                    margin: 0.1,
                                    strength: 1.0,
                                },
                                FieldSelector::Normalized { hx, hy } => StraySource::Field {
                                    hx: *hx,
                                    hy: *hy,
                                    margin: 0.1,
                                    strength: 1.0,
                                },
                                FieldSelector::All => unreachable!(),
                            };
                            let r = analyze(
                                rs.clone(),
                                &settings,
                                &sub.subrange(
                                    j as f64 / fs.len() as f64,
                                    (j + 1) as f64 / fs.len() as f64,
                                ),
                            )?;
                            value += weight
                                * r.stray_fraction.ok_or_else(|| {
                                    MkError::Numerical("zero primary reference flux".into())
                                })?;
                            total += weight;
                        }
                        if total <= 0.0 {
                            return Err(MkError::InvalidParam("no positive field weights".into()));
                        }
                        Ok(value / total)
                    }
                    _ => Err(MkError::InvalidParam(
                        "expected ghost or stray operand".into(),
                    )),
                }
            })();
            out.push(operand_result(value)?);
            sub.report(1.0, "ghost / stray merit");
        }
        Ok(out)
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
            .enumerate()
            .map(|(i, s)| {
                self.candidate(
                    s,
                    ops,
                    &ctx.subrange(
                        i as f64 / systems.len() as f64,
                        (i + 1) as f64 / systems.len() as f64,
                    ),
                )
            })
            .collect()
    }
}
