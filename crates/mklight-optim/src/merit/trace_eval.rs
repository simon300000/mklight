//! [`TraceEvaluator`]: operands computed with the CPU double-precision
//! reference tracer (`mklight-trace`) — first-order (paraxial), real-ray and
//! wavefront operands, plus the model-only operands with edge thicknesses
//! sized by the tracer's automatic semi-diameters.
//!
//! # Per candidate
//!
//! Each candidate system (configuration already applied) is frozen into a
//! `Snapshot`, resolved against the evaluator's shared
//! `Arc<MaterialLibrary>` into a `ResolvedSystem`, and wrapped in a
//! `SequentialTracer` (paraxial data, pupils, prepared fields). The operand
//! list is then planned: every (field, wavelength) whose chief ray or OPD
//! reference sphere is needed, and every pupil sampling ([`RayGroup`] or
//! aggregate sampling) at a (field, wavelength), is traced **once** into a
//! per-candidate cache, in parallel, before the operands are read from it.
//! Candidates of a batch are evaluated in parallel too (rayon). Every value is
//! a pure function of the candidate and the operand, assembled in a fixed
//! order, so results do not depend on the thread count.
//!
//! # Conventions
//!
//! * Lengths in mm, angles in degrees, OPD in waves at the ray's wavelength
//!   (CONVENTIONS §1, §4); real-ray positions and directions in the surface's
//!   local frame unless `global`.
//! * `Chief` spot references are the **primary-wavelength** chief ray of the
//!   field (so lateral colour stays in the merit); `Chief` wavefront
//!   references are the CONVENTIONS §4 sphere at the operand's wavelength.
//! * `Centroid` references follow the [`RayGroup`] contract: every node of the
//!   group is traced at the operand's field and wavelength (or every
//!   positive-weight wavelength for `Polychromatic`, weights
//!   `w_node · w_λ`), and the reference is formed with
//!   [`spot_centroid`] / [`piston_tilt`], honouring the half-pupil symmetry.
//!   Failed rays are left out.
//! * Aggregate operands (`RmsSpot`, `WavefrontRms`) with `FieldSelector::All`
//!   or `WavelengthSelector::All` average the *squared* RMS over fields and
//!   wavelengths with their weights (RMS spot at `All` wavelengths is the
//!   polychromatic spot about its own reference); first-order operands at
//!   `All` wavelengths are the weighted mean.
//! * Rays are clipped by fixed apertures when [`TraceEvalSettings::clip`]
//!   is set (default). A ray that fails gives `OperandError::RayFailure`, which
//!   the spec's `FailurePolicy` turns into a penalty or a failed candidate;
//!   aggregates leave failed rays out and fail only when none survive.
//!
//! MTF, ghost, stray-light and plug-in operands are **not** computed here:
//! combine this evaluator with the ones providing them through
//! [`CompositeEvaluator`](super::CompositeEvaluator).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use mklight_core::materials::MaterialLibrary;
use mklight_core::job::{Precision, RunTimer};
use mklight_core::{
    JobContext, MkResult, OpticalSystem, ResolvedSystem, RunInfo, SemiDiameter, Snapshot,
    SurfaceId, Vec3,
};
use mklight_trace::{
    ObjectSource, OpdReference, ParaxialModel, PreparedField, SemiDiameterSettings,
    SequentialTracer, SurfaceRecord, TraceOptions, TraceResult, TracerSettings, fix_semi_diameters,
    trace::to_global,
};

use super::geometric::{is_model_kind, model_value, needs_semi_diameters};
use super::operand::{
    Axis, CentroidScope, DistortionMapping, FNumberKind, OperandKind, ParaxialQuantity,
    ParaxialRayKind, RayGroup, RayQuantity, SpotReference, WavefrontReference,
};
use super::quadrature::{PupilNode, PupilSampling, PupilSymmetry};
use super::reference::{piston_tilt, spot_centroid};
use super::{
    FieldSelector, Operand, OperandError, OperandEvaluator, OperandValue, WavelengthSelector,
};

/// Settings of a [`TraceEvaluator`].
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct TraceEvalSettings {
    /// Ray aiming override and intersection settings of every tracer.
    #[serde(default)]
    pub tracer: TracerSettings,
    /// Let fixed semi-diameters and apertures clip (vignette) rays.
    pub clip: bool,
    /// Pupil-edge azimuths for automatic semi-diameters (edge thickness).
    pub semi_diameter_azimuths: usize,
    /// Polar pupil grid of the relative-illumination operand.
    pub illumination_rings: usize,
    pub illumination_arms: usize,
}

impl Default for TraceEvalSettings {
    fn default() -> Self {
        TraceEvalSettings {
            tracer: TracerSettings::default(),
            clip: true,
            semi_diameter_azimuths: 16,
            illumination_rings: 12,
            illumination_arms: 24,
        }
    }
}

/// Operand evaluator backed by the CPU f64 reference tracer (module docs).
#[derive(Clone, Debug)]
pub struct TraceEvaluator {
    library: Arc<MaterialLibrary>,
    settings: TraceEvalSettings,
}

impl TraceEvaluator {
    pub fn new(library: Arc<MaterialLibrary>) -> Self {
        Self::with_settings(library, TraceEvalSettings::default())
    }
    pub fn with_settings(library: Arc<MaterialLibrary>, settings: TraceEvalSettings) -> Self {
        TraceEvaluator { library, settings }
    }
    /// With the built-in glass catalog.
    pub fn builtin() -> Self {
        Self::new(Arc::new(MaterialLibrary::with_builtin()))
    }
    pub fn library(&self) -> &Arc<MaterialLibrary> {
        &self.library
    }
    pub fn settings(&self) -> &TraceEvalSettings {
        &self.settings
    }

    /// Backend name reported in [`RunInfo`]s of runs using this evaluator.
    pub const BACKEND: &'static str = "cpu-f64-reference";

    /// `RunInfo` of an optimisation or tolerancing run on `start` (the
    /// system the run started from) that used this evaluator: snapshot hash,
    /// the run's settings, seed, timing and the tracer's warnings.
    pub fn run_info(
        &self,
        timer: RunTimer,
        start: &OpticalSystem,
        analysis: &str,
        settings: &impl Serialize,
        seed: Option<u64>,
    ) -> MkResult<RunInfo> {
        let snap = Snapshot::new(start.clone())?;
        let warnings = match self.tracer(start) {
            Ok(t) => t.warnings().to_vec(),
            Err(e) => vec![format!("{e:?}")],
        };
        Ok(timer.finish(
            &snap,
            analysis,
            settings,
            Self::BACKEND,
            Precision::F64Reference,
            seed,
            warnings,
        ))
    }

    /// Snapshot → `ResolvedSystem` → `SequentialTracer` for one system.
    /// Invalid systems give `Invalid`, paraxial failures `RayFailure`.
    pub fn tracer(&self, s: &OpticalSystem) -> Result<SequentialTracer, OperandError> {
        let snap = Snapshot::new(s.clone()).map_err(|e| OperandError::Invalid(e.to_string()))?;
        let rs = ResolvedSystem::new(&snap, self.library.clone())
            .map_err(|e| OperandError::Invalid(e.to_string()))?;
        SequentialTracer::with_settings(Arc::new(rs), self.settings.tracer)
            .map_err(|e| OperandError::RayFailure(e.to_string()))
    }

    /// Values of `operands` (all supported) for one system.
    pub fn evaluate_system(
        &self,
        s: &OpticalSystem,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<OperandValue>> {
        ctx.check()?;
        let needs_tracer = operands
            .iter()
            .any(|o| !is_model_kind(&o.kind) || needs_semi_diameters(&o.kind));
        let tracer = if needs_tracer {
            Some(self.tracer(s))
        } else {
            None
        };
        let cand = match &tracer {
            Some(Ok(tr)) => Some(Candidate::build(self, tr, operands, ctx)?),
            _ => None,
        };
        let mut fixed_sd: Option<OpticalSystem> = None;
        let mut out = Vec::with_capacity(operands.len());
        for op in operands {
            let v = if is_model_kind(&op.kind) {
                if needs_semi_diameters(&op.kind) && has_auto_semi_diameters(s) {
                    match &tracer {
                        Some(Ok(tr)) => {
                            let sys = fixed_sd.get_or_insert_with(|| {
                                let sd = tr.semi_diameters(&SemiDiameterSettings {
                                    azimuths: self.settings.semi_diameter_azimuths.max(4),
                                    clip: self.settings.clip,
                                });
                                fix_semi_diameters(s, &sd, 0.0)
                            });
                            model_value(sys, op)
                        }
                        Some(Err(e)) => Err(e.clone()),
                        None => model_value(s, op),
                    }
                } else {
                    model_value(s, op)
                }
            } else if let OperandKind::ToleranceSensitivity {
                param,
                delta,
                criterion,
            } = &op.kind
            {
                self.sensitivity(s, op, param, *delta, criterion, ctx)?
            } else {
                match (&tracer, &cand) {
                    (_, Some(c)) => c.value(op),
                    (Some(Err(e)), _) => Err(e.clone()),
                    _ => Err(OperandError::Unsupported),
                }
            };
            out.push(v);
        }
        Ok(out)
    }

    /// `ToleranceSensitivity`: the criterion at nominal and at `param ± δ`.
    fn sensitivity(
        &self,
        s: &OpticalSystem,
        op: &Operand,
        param: &mklight_core::param::ParamPath,
        delta: f64,
        criterion: &OperandKind,
        ctx: &JobContext,
    ) -> MkResult<OperandValue> {
        let crit = Operand {
            kind: criterion.clone(),
            role: super::OperandRole::Residual,
            ..op.clone()
        };
        let Ok(v0) = param.get(s) else {
            return Ok(Err(OperandError::Invalid("unknown parameter".into())));
        };
        let mut vals = [0.0; 3];
        for (k, d) in [0.0, delta, -delta].into_iter().enumerate() {
            let mut sk = s.clone();
            if let Err(e) = param.set(&mut sk, v0 + d) {
                return Ok(Err(OperandError::Invalid(e.to_string())));
            }
            match self.evaluate_system(&sk, &[&crit], ctx)?.remove(0) {
                Ok(v) => vals[k] = v,
                Err(e) => return Ok(Err(e)),
            }
        }
        let (dp, dm) = (vals[1] - vals[0], vals[2] - vals[0]);
        Ok(Ok(((dp * dp + dm * dm) / 2.0).sqrt()))
    }
}

fn has_auto_semi_diameters(s: &OpticalSystem) -> bool {
    s.surfaces
        .iter()
        .any(|sf| matches!(sf.semi_diameter, SemiDiameter::Auto))
}

/// Kinds a [`TraceEvaluator`] computes: model, paraxial, real-ray and
/// wavefront operands, and tolerance sensitivities of those.
pub fn trace_supports(kind: &OperandKind) -> bool {
    use super::operand::Capability::*;
    match kind {
        OperandKind::ToleranceSensitivity { criterion, .. } => trace_supports(criterion),
        k => matches!(k.capability(), Model | Paraxial | RealRay | Wavefront),
    }
}

impl OperandEvaluator for TraceEvaluator {
    fn supports(&self, kind: &OperandKind) -> bool {
        trace_supports(kind)
    }

    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        systems
            .par_iter()
            .map(|s| self.evaluate_system(s, operands, ctx))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Per-candidate cache
// ---------------------------------------------------------------------------

/// A field: a system field by index or an arbitrary normalised point (bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum FieldKey {
    Index(usize),
    Point(u64, u64),
}

const ON_AXIS: FieldKey = FieldKey::Point(0, 0);

/// Bit-exact identity of a pupil sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct SamplingKey {
    rings: usize,
    arms: usize,
    bilateral: bool,
    obscuration: u64,
    offset: u64,
}

impl SamplingKey {
    fn of(s: &PupilSampling) -> Self {
        SamplingKey {
            rings: s.rings,
            arms: s.arms,
            bilateral: s.symmetry == PupilSymmetry::Bilateral,
            obscuration: s.obscuration.to_bits(),
            offset: s.arm_offset_deg.to_bits(),
        }
    }
}

type GroupKey = (FieldKey, usize, SamplingKey);

/// Chief ray and OPD reference sphere of one (field, wavelength).
struct ChiefData {
    chief: Result<TraceResult, OperandError>,
    reference: Result<OpdReference, OperandError>,
}

/// One traced pupil node: image-surface `(x, y)` (local frame) and OPD
/// (waves), `None` where the ray (or its OPD) failed.
#[derive(Clone, Copy)]
struct NodeRay {
    pos: Option<[f64; 2]>,
    opd: Option<f64>,
}

struct GroupRays {
    nodes: Vec<PupilNode>,
    rays: Vec<NodeRay>,
}

/// What an operand needs traced before evaluation.
#[derive(Default)]
struct Plan {
    points: BTreeSet<FieldKey>,
    chiefs: BTreeSet<(FieldKey, usize)>,
    groups: BTreeMap<GroupKey, PupilSampling>,
}

fn status_error(r: &TraceResult) -> OperandError {
    OperandError::RayFailure(format!("{:?}", r.status))
}

fn axis_value(p: [f64; 2], axis: Axis) -> f64 {
    match axis {
        Axis::X => p[0],
        Axis::Y => p[1],
    }
}

/// Weighted mean of `f(item)` over `(item, weight)` pairs.
fn weighted_mean<T>(
    items: &[(T, f64)],
    mut f: impl FnMut(&T) -> Result<f64, OperandError>,
) -> OperandValue {
    let mut s = mklight_core::math::KahanSum::default();
    let mut w = 0.0;
    for (it, wt) in items {
        s.add(wt * f(it)?);
        w += wt;
    }
    if !(w > 0.0) {
        return Err(OperandError::Invalid("no positive weight".into()));
    }
    Ok(s.value() / w)
}

struct Candidate<'a> {
    ev: &'a TraceEvaluator,
    tr: &'a SequentialTracer,
    opts: TraceOptions,
    points: BTreeMap<FieldKey, Result<PreparedField, OperandError>>,
    chiefs: BTreeMap<(FieldKey, usize), ChiefData>,
    groups: BTreeMap<GroupKey, GroupRays>,
}

impl<'a> Candidate<'a> {
    fn build(
        ev: &'a TraceEvaluator,
        tr: &'a SequentialTracer,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Self> {
        let opts = TraceOptions {
            clip: ev.settings.clip,
            ..tr.options()
        };
        let mut c = Candidate {
            ev,
            tr,
            opts,
            points: BTreeMap::new(),
            chiefs: BTreeMap::new(),
            groups: BTreeMap::new(),
        };
        let mut plan = Plan::default();
        for op in operands {
            c.plan(op, &mut plan);
        }
        // Group (field, λ) pairs need their chief ray / reference sphere.
        for &(f, w, _) in plan.groups.keys() {
            plan.chiefs.insert((f, w));
        }
        for &(f, _) in &plan.chiefs {
            if let FieldKey::Point(..) = f {
                plan.points.insert(f);
            }
        }
        // Stage 1: arbitrary field points.
        c.points = plan
            .points
            .par_iter()
            .map(|&k| {
                let FieldKey::Point(hx, hy) = k else {
                    unreachable!()
                };
                let fp = mklight_trace::FieldPoint::normalized(f64::from_bits(hx), f64::from_bits(hy));
                let f = tr
                    .prepare_field(fp)
                    .map_err(|e| OperandError::RayFailure(e.to_string()));
                (k, f)
            })
            .collect();
        ctx.check()?;
        // Stage 2: chief rays and reference spheres.
        c.chiefs = plan
            .chiefs
            .par_iter()
            .map(|&(f, w)| {
                let d = match c.field(f) {
                    Ok(field) => {
                        let chief = tr.chief(field, w);
                        let reference = tr.opd_reference(field, w).map_err(|s| {
                            OperandError::RayFailure(format!("chief ray: {s:?}"))
                        });
                        ChiefData {
                            chief: if chief.status.is_ok() {
                                Ok(chief)
                            } else {
                                Err(status_error(&chief))
                            },
                            reference,
                        }
                    }
                    Err(e) => ChiefData {
                        chief: Err(e.clone()),
                        reference: Err(e),
                    },
                };
                ((f, w), d)
            })
            .collect();
        ctx.check()?;
        // Stage 3: pupil samplings.
        let groups: MkResult<Vec<(GroupKey, GroupRays)>> = plan
            .groups
            .par_iter()
            .map(|(&key, sampling)| {
                ctx.check()?;
                Ok((key, c.trace_group(key, sampling)))
            })
            .collect();
        c.groups = groups?.into_iter().collect();
        Ok(c)
    }

    fn trace_group(&self, (f, w, _): GroupKey, sampling: &PupilSampling) -> GroupRays {
        let nodes = sampling.nodes();
        let reference = self.chiefs[&(f, w)].reference.as_ref().ok();
        let rays = match self.field(f) {
            Ok(field) => nodes
                .iter()
                .map(|n| {
                    let r = self.tr.trace(field, w, n.px, n.py, &self.opts);
                    if r.status.is_ok() {
                        NodeRay {
                            pos: Some([r.ray.pos.x, r.ray.pos.y]),
                            opd: reference.and_then(|rf| rf.opd_waves(&r)),
                        }
                    } else {
                        NodeRay {
                            pos: None,
                            opd: None,
                        }
                    }
                })
                .collect(),
            Err(_) => vec![
                NodeRay {
                    pos: None,
                    opd: None
                };
                nodes.len()
            ],
        };
        GroupRays { nodes, rays }
    }

    // ----- selectors -----------------------------------------------------

    fn system(&self) -> &OpticalSystem {
        self.tr.system().system()
    }

    fn primary(&self) -> usize {
        self.tr.system().primary_wavelength
    }

    /// `(wavelength, weight)` of a selector.
    fn wavelengths(&self, sel: WavelengthSelector) -> Result<Vec<(usize, f64)>, OperandError> {
        let nw = self.tr.num_wavelengths();
        match sel {
            WavelengthSelector::Primary => Ok(vec![(self.primary(), 1.0)]),
            WavelengthSelector::Index { index } if index < nw => Ok(vec![(index, 1.0)]),
            WavelengthSelector::Index { index } => Err(OperandError::Invalid(format!(
                "wavelength {index} does not exist"
            ))),
            WavelengthSelector::All => {
                let w = &self.tr.system().wavelength_weights;
                let v: Vec<(usize, f64)> = (0..nw)
                    .filter(|&i| w[i] > 0.0)
                    .map(|i| (i, w[i]))
                    .collect();
                if v.is_empty() {
                    Err(OperandError::Invalid("no wavelength with positive weight".into()))
                } else {
                    Ok(v)
                }
            }
        }
    }

    fn single_wavelength(&self, sel: WavelengthSelector) -> Result<usize, OperandError> {
        match sel {
            WavelengthSelector::All => Err(OperandError::Invalid(
                "operand needs a single wavelength".into(),
            )),
            s => Ok(self.wavelengths(s)?[0].0),
        }
    }

    /// `(field, weight)` of a selector.
    fn fields(&self, sel: FieldSelector) -> Result<Vec<(FieldKey, f64)>, OperandError> {
        let fs = &self.system().fields.fields;
        match sel {
            FieldSelector::Index { index } if index < fs.len() => {
                Ok(vec![(FieldKey::Index(index), 1.0)])
            }
            FieldSelector::Index { index } => Err(OperandError::Invalid(format!(
                "field {index} does not exist"
            ))),
            FieldSelector::Normalized { hx, hy } => Ok(vec![(
                FieldKey::Point((hx + 0.0).to_bits(), (hy + 0.0).to_bits()),
                1.0,
            )]),
            FieldSelector::All => {
                let v: Vec<(FieldKey, f64)> = fs
                    .iter()
                    .enumerate()
                    .filter(|(_, f)| f.weight > 0.0)
                    .map(|(i, f)| (FieldKey::Index(i), f.weight))
                    .collect();
                if v.is_empty() {
                    Err(OperandError::Invalid("no field with positive weight".into()))
                } else {
                    Ok(v)
                }
            }
        }
    }

    fn single_field(&self, sel: FieldSelector) -> Result<FieldKey, OperandError> {
        match sel {
            FieldSelector::All => Err(OperandError::Invalid("operand needs a single field".into())),
            s => Ok(self.fields(s)?[0].0),
        }
    }

    fn field(&self, k: FieldKey) -> Result<&PreparedField, OperandError> {
        match k {
            FieldKey::Index(i) => self
                .tr
                .fields()
                .get(i)
                .ok_or_else(|| OperandError::Invalid(format!("field {i} does not exist"))),
            FieldKey::Point(..) => match self.points.get(&k) {
                Some(Ok(f)) => Ok(f),
                Some(Err(e)) => Err(e.clone()),
                None => Err(OperandError::Invalid("field point was not prepared".into())),
            },
        }
    }

    /// Wavelengths of a ray group for an operand at wavelength `wl`.
    fn group_wavelengths(&self, g: &RayGroup, wl: usize) -> Vec<(usize, f64)> {
        match g.scope {
            CentroidScope::Monochromatic => vec![(wl, 1.0)],
            CentroidScope::Polychromatic => self
                .wavelengths(WavelengthSelector::All)
                .unwrap_or_default(),
        }
    }

    // ----- planning ------------------------------------------------------

    fn plan(&self, op: &Operand, p: &mut Plan) {
        use OperandKind::*;
        let fields = self.fields(op.field).unwrap_or_default();
        let wls = self.wavelengths(op.wavelength).unwrap_or_default();
        let primary = self.primary();
        let mut group = |f: FieldKey, w: usize, s: &PupilSampling| {
            p.groups.insert((f, w, SamplingKey::of(s)), *s);
        };
        match &op.kind {
            RayAberration {
                reference, group: g, ..
            } => {
                for &(f, _) in &fields {
                    if let Some(g) = g {
                        for &(w, _) in &wls {
                            for (gw, _) in self.group_wavelengths(g, w) {
                                group(f, gw, &g.sampling);
                            }
                        }
                    }
                    if *reference == SpotReference::Chief {
                        p.chiefs.insert((f, primary));
                    }
                    p.points.insert(f);
                }
            }
            WavefrontAberration { group: g, .. } => {
                for &(f, _) in &fields {
                    for &(w, _) in &wls {
                        if let Some(g) = g {
                            for (gw, _) in self.group_wavelengths(g, w) {
                                group(f, gw, &g.sampling);
                            }
                        }
                        p.chiefs.insert((f, w));
                    }
                }
            }
            RmsSpot {
                reference,
                sampling,
                ..
            } => {
                for &(f, _) in &fields {
                    for &(w, _) in &wls {
                        group(f, w, sampling);
                    }
                    if *reference == SpotReference::Chief {
                        p.chiefs.insert((f, primary));
                    }
                }
            }
            SpotCentroid { .. } => {
                for &(f, _) in &fields {
                    for &(w, _) in &wls {
                        group(f, w, &PupilSampling::default());
                    }
                    p.chiefs.insert((f, primary));
                }
            }
            WavefrontRms { sampling, .. } => {
                for &(f, _) in &fields {
                    for &(w, _) in &wls {
                        group(f, w, sampling);
                    }
                }
            }
            Distortion { .. } => {
                for &(f, _) in &fields {
                    for &(w, _) in &wls {
                        p.chiefs.insert((f, w));
                    }
                }
            }
            LateralColour { short, long } => {
                for &(f, _) in &fields {
                    p.chiefs.insert((f, *short));
                    p.chiefs.insert((f, *long));
                }
            }
            RealRay { .. } | ChiefRayAngle { .. } => {
                for &(f, _) in &fields {
                    p.points.insert(f);
                }
            }
            RelativeIllumination => {
                for &(f, _) in &fields {
                    p.points.insert(f);
                }
                p.points.insert(ON_AXIS);
            }
            FNumber {
                kind: FNumberKind::RealWorking,
            } => {
                p.points.insert(ON_AXIS);
            }
            _ => {}
        }
        // Only arbitrary points need preparing.
        p.points.retain(|k| matches!(k, FieldKey::Point(..)));
    }

    // ----- rays ------------------------------------------------------------

    fn chief(&self, f: FieldKey, w: usize) -> Result<&TraceResult, OperandError> {
        match self.chiefs.get(&(f, w)) {
            Some(d) => d.chief.as_ref().map_err(|e| e.clone()),
            None => Err(OperandError::Invalid("chief ray was not planned".into())),
        }
    }

    fn reference(&self, f: FieldKey, w: usize) -> Result<&OpdReference, OperandError> {
        match self.chiefs.get(&(f, w)) {
            Some(d) => d.reference.as_ref().map_err(|e| e.clone()),
            None => Err(OperandError::Invalid("reference was not planned".into())),
        }
    }

    fn group(&self, f: FieldKey, w: usize, s: &PupilSampling) -> Result<&GroupRays, OperandError> {
        self.groups
            .get(&(f, w, SamplingKey::of(s)))
            .ok_or_else(|| OperandError::Invalid("pupil sampling was not planned".into()))
    }

    /// One real ray (clipping per settings), optionally recording surfaces.
    fn ray(
        &self,
        f: FieldKey,
        w: usize,
        px: f64,
        py: f64,
        opts: &TraceOptions,
        record: Option<&mut Vec<SurfaceRecord>>,
    ) -> Result<TraceResult, OperandError> {
        let field = self.field(f)?;
        let r = self.tr.trace_with(field, w, px, py, opts, record);
        if r.status.is_ok() {
            Ok(r)
        } else {
            Err(status_error(&r))
        }
    }

    /// Node `(px, py)` of a traced group, by bit-exact coordinates.
    fn node_in<'g>(g: &'g GroupRays, px: f64, py: f64) -> Option<&'g NodeRay> {
        g.nodes
            .iter()
            .position(|n| n.px.to_bits() == px.to_bits() && n.py.to_bits() == py.to_bits())
            .map(|i| &g.rays[i])
    }

    /// Image `(x, y)` of pupil point `(px, py)`, reusing a group trace.
    fn image_point(
        &self,
        f: FieldKey,
        w: usize,
        px: f64,
        py: f64,
        group: Option<&RayGroup>,
    ) -> Result<[f64; 2], OperandError> {
        if let Some(g) = group
            && let Ok(gr) = self.group(f, w, &g.sampling)
            && let Some(n) = Self::node_in(gr, px, py)
        {
            return n
                .pos
                .ok_or_else(|| OperandError::RayFailure("ray failed".into()));
        }
        let r = self.ray(f, w, px, py, &self.opts, None)?;
        Ok([r.ray.pos.x, r.ray.pos.y])
    }

    /// OPD (waves) of pupil point `(px, py)`, reusing a group trace.
    fn opd(
        &self,
        f: FieldKey,
        w: usize,
        px: f64,
        py: f64,
        group: Option<&RayGroup>,
    ) -> Result<f64, OperandError> {
        let reference = self.reference(f, w)?;
        if let Some(g) = group
            && let Ok(gr) = self.group(f, w, &g.sampling)
            && let Some(n) = Self::node_in(gr, px, py)
        {
            return n
                .opd
                .ok_or_else(|| OperandError::RayFailure("ray or OPD failed".into()));
        }
        let r = self.ray(f, w, px, py, &self.opts, None)?;
        reference
            .opd_waves(&r)
            .ok_or_else(|| OperandError::RayFailure("ray misses the reference sphere".into()))
    }

    /// `(weights, x, y)` of the surviving rays of `sampling` over `wls`.
    fn spot_rays(
        &self,
        f: FieldKey,
        wls: &[(usize, f64)],
        s: &PupilSampling,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), OperandError> {
        let (mut wt, mut xs, mut ys) = (Vec::new(), Vec::new(), Vec::new());
        for &(w, ww) in wls {
            let g = self.group(f, w, s)?;
            for (n, r) in g.nodes.iter().zip(&g.rays) {
                if let Some(p) = r.pos {
                    wt.push(n.weight * ww);
                    xs.push(p[0]);
                    ys.push(p[1]);
                }
            }
        }
        if wt.is_empty() {
            return Err(OperandError::RayFailure(
                "no ray of the pupil sampling reached the image".into(),
            ));
        }
        Ok((wt, xs, ys))
    }

    /// Spot centroid of a group over `wls`.
    fn centroid(
        &self,
        f: FieldKey,
        wls: &[(usize, f64)],
        s: &PupilSampling,
    ) -> Result<[f64; 2], OperandError> {
        let (wt, xs, ys) = self.spot_rays(f, wls, s)?;
        spot_centroid(&wt, &xs, &ys, s.symmetry)
            .ok_or_else(|| OperandError::RayFailure("spot centroid undefined".into()))
    }

    /// Piston/tilt `[a, bx, by]` of a group's OPD over `wls`.
    fn piston_tilt(
        &self,
        f: FieldKey,
        wls: &[(usize, f64)],
        s: &PupilSampling,
    ) -> Result<[f64; 3], OperandError> {
        let (mut px, mut py, mut wt, mut opd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &(w, ww) in wls {
            let g = self.group(f, w, s)?;
            for (n, r) in g.nodes.iter().zip(&g.rays) {
                if let Some(o) = r.opd {
                    px.push(n.px);
                    py.push(n.py);
                    wt.push(n.weight * ww);
                    opd.push(o);
                }
            }
        }
        piston_tilt(&px, &py, &wt, &opd, s.symmetry)
            .ok_or_else(|| OperandError::RayFailure("too few rays for piston and tilt".into()))
    }

    /// Unit vector along the field direction on the image (`+y` on axis).
    fn field_direction(&self, f: FieldKey) -> Result<([f64; 2], f64, bool), OperandError> {
        let field = self.field(f)?;
        let (t, at_inf) = match field.source {
            ObjectSource::Direction { dir } => ([dir.x / dir.z, dir.y / dir.z], true),
            ObjectSource::Point { point } => ([point.x, point.y], false),
        };
        let r = t[0].hypot(t[1]);
        if r > 0.0 {
            Ok(([t[0] / r, t[1] / r], r, at_inf))
        } else {
            Ok(([0.0, 1.0], 0.0, at_inf))
        }
    }

    fn surface_index(&self, id: Option<SurfaceId>) -> Result<usize, OperandError> {
        match id {
            None => Ok(self.tr.system().image_index()),
            Some(id) => self
                .system()
                .surface_index(id)
                .ok_or_else(|| OperandError::Invalid(format!("no surface with id {}", id.0))),
        }
    }

    // ----- values ----------------------------------------------------------

    fn value(&self, op: &Operand) -> OperandValue {
        use OperandKind::*;
        let px = self.tr.paraxial();
        let fo_mean = |f: &dyn Fn(&mklight_trace::FirstOrder) -> f64| -> OperandValue {
            let wls = self.wavelengths(op.wavelength)?;
            weighted_mean(&wls, |&w| Ok(f(px.at(w))))
        };
        let finite = |v: f64, what: &str| -> OperandValue {
            if v.is_finite() {
                Ok(v)
            } else {
                Err(OperandError::Invalid(format!("{what} is at infinity")))
            }
        };
        match &op.kind {
            Efl => fo_mean(&|fo| fo.efl),
            Bfl => fo_mean(&|fo| fo.bfl),
            FNumber {
                kind: FNumberKind::ParaxialImageSpace,
            } => fo_mean(&|fo| fo.image_f_number),
            FNumber {
                kind: FNumberKind::ParaxialWorking,
            } => fo_mean(&|fo| fo.working_f_number),
            FNumber {
                kind: FNumberKind::RealWorking,
            } => {
                let wls = self.wavelengths(op.wavelength)?;
                weighted_mean(&wls, |&w| self.real_working_f_number(w))
            }
            Magnification => fo_mean(&|fo| fo.magnification),
            EntrancePupilDiameter => fo_mean(&|fo| fo.entrance_pupil_diameter),
            ExitPupilPosition => finite(fo_mean(&|fo| fo.exit_pupil_position)?, "exit pupil"),
            EntrancePupilPosition => finite(
                fo_mean(&|fo| fo.entrance_pupil_position)?,
                "entrance pupil",
            ),
            ParaxialRay {
                ray,
                quantity,
                surface,
            } => {
                let j = self.surface_index(*surface)?;
                fo_mean(&|fo| {
                    let t = match ray {
                        ParaxialRayKind::Marginal => &fo.marginal,
                        ParaxialRayKind::Chief => &fo.chief,
                    };
                    match quantity {
                        ParaxialQuantity::Height => t.y[j],
                        ParaxialQuantity::Slope => t.u[j],
                        ParaxialQuantity::Angle => t.u[j].atan().to_degrees(),
                    }
                })
            }
            PetzvalSum => {
                let wls = self.wavelengths(op.wavelength)?;
                weighted_mean(&wls, |&w| {
                    let m = ParaxialModel::new(self.tr.system(), w);
                    let nn = &m.reduced_index;
                    let mut s = mklight_core::math::KahanSum::default();
                    for j in 1..nn.len() {
                        if m.power[j] != 0.0 {
                            s.add(m.power[j] / (nn[j] * nn[j - 1]));
                        }
                    }
                    Ok(s.value())
                })
            }
            AxialColour { short, long } => {
                let nw = self.tr.num_wavelengths();
                if *short >= nw || *long >= nw {
                    return Err(OperandError::Invalid("wavelength index out of range".into()));
                }
                Ok(px.at(*short).image_distance - px.at(*long).image_distance)
            }
            RealRay {
                quantity,
                surface,
                px: ppx,
                py: ppy,
                global,
            } => self.real_ray(op, *quantity, *surface, *ppx, *ppy, *global),
            RayAberration {
                axis,
                px: ppx,
                py: ppy,
                reference,
                group,
            } => {
                let f = self.single_field(op.field)?;
                let w = self.single_wavelength(op.wavelength)?;
                let p = self.image_point(f, w, *ppx, *ppy, group.as_ref())?;
                let r = match reference {
                    SpotReference::Chief => {
                        let c = self.chief(f, self.primary())?;
                        [c.ray.pos.x, c.ray.pos.y]
                    }
                    SpotReference::Centroid => {
                        let g = group.as_ref().ok_or_else(|| {
                            OperandError::Invalid("centroid reference needs a ray group".into())
                        })?;
                        self.centroid(f, &self.group_wavelengths(g, w), &g.sampling)?
                    }
                };
                Ok(axis_value(p, *axis) - axis_value(r, *axis))
            }
            WavefrontAberration {
                px: ppx,
                py: ppy,
                reference,
                group,
            } => {
                let f = self.single_field(op.field)?;
                let w = self.single_wavelength(op.wavelength)?;
                let v = self.opd(f, w, *ppx, *ppy, group.as_ref())?;
                match reference {
                    WavefrontReference::Chief => Ok(v),
                    WavefrontReference::Centroid => {
                        let g = group.as_ref().ok_or_else(|| {
                            OperandError::Invalid("centroid reference needs a ray group".into())
                        })?;
                        let c = self.piston_tilt(f, &self.group_wavelengths(g, w), &g.sampling)?;
                        Ok(v - (c[0] + c[1] * ppx + c[2] * ppy))
                    }
                }
            }
            ChiefRayAngle { surface } => {
                let f = self.single_field(op.field)?;
                let w = self.single_wavelength(op.wavelength)?;
                let j = self.surface_index(*surface)?;
                let mut rec = Vec::new();
                self.ray(
                    f,
                    w,
                    0.0,
                    0.0,
                    &TraceOptions {
                        clip: false,
                        ..self.opts
                    },
                    Some(&mut rec),
                )?;
                let r = rec
                    .get(j)
                    .ok_or_else(|| OperandError::Invalid("surface not reached".into()))?;
                Ok(r.cos_incidence.clamp(0.0, 1.0).acos().to_degrees())
            }
            RmsSpot {
                reference,
                component,
                sampling,
            } => {
                let fields = self.fields(op.field)?;
                let wls = self.wavelengths(op.wavelength)?;
                let ms = weighted_mean(&fields, |&f| {
                    let (wt, xs, ys) = self.spot_rays(f, &wls, sampling)?;
                    let c = match reference {
                        SpotReference::Centroid => spot_centroid(&wt, &xs, &ys, sampling.symmetry)
                            .ok_or_else(|| {
                                OperandError::RayFailure("spot centroid undefined".into())
                            })?,
                        SpotReference::Chief => {
                            let c = self.chief(f, self.primary())?;
                            [c.ray.pos.x, c.ray.pos.y]
                        }
                    };
                    let mut s = mklight_core::math::KahanSum::default();
                    let mut sw = 0.0;
                    for k in 0..wt.len() {
                        let (dx, dy) = (xs[k] - c[0], ys[k] - c[1]);
                        let e2 = match component {
                            None => dx * dx + dy * dy,
                            Some(Axis::X) => dx * dx,
                            Some(Axis::Y) => dy * dy,
                        };
                        s.add(wt[k] * e2);
                        sw += wt[k];
                    }
                    Ok(s.value() / sw)
                })?;
                Ok(ms.sqrt())
            }
            SpotCentroid { axis } => {
                let f = self.single_field(op.field)?;
                let wls = self.wavelengths(op.wavelength)?;
                let s = PupilSampling::default();
                let c = self.centroid(f, &wls, &s)?;
                let ch = self.chief(f, self.primary())?;
                Ok(axis_value(c, *axis) - axis_value([ch.ray.pos.x, ch.ray.pos.y], *axis))
            }
            WavefrontRms {
                reference,
                sampling,
            } => {
                let fields = self.fields(op.field)?;
                let wls = self.wavelengths(op.wavelength)?;
                let pairs: Vec<((FieldKey, usize), f64)> = fields
                    .iter()
                    .flat_map(|&(f, wf)| wls.iter().map(move |&(w, ww)| ((f, w), wf * ww)))
                    .collect();
                let ms = weighted_mean(&pairs, |&(f, w)| {
                    self.wavefront_ms(f, w, *reference, sampling)
                })?;
                Ok(ms.sqrt())
            }
            Distortion { mapping } => {
                let f = self.single_field(op.field)?;
                let w = self.single_wavelength(op.wavelength)?;
                let (u, r, at_inf) = self.field_direction(f)?;
                if r == 0.0 {
                    return Ok(0.0);
                }
                let fo = px.at(w);
                let reference = match (mapping, at_inf) {
                    (DistortionMapping::FTanTheta, true) => fo.image_height_per_tan.map(|h| h * r),
                    (DistortionMapping::FTheta, true) => {
                        fo.image_height_per_tan.map(|h| h * r.atan())
                    }
                    (DistortionMapping::FTanTheta, false) => {
                        fo.image_height_per_object_height.map(|h| h * r)
                    }
                    (DistortionMapping::FTheta, false) => None,
                }
                .filter(|h| *h != 0.0 && h.is_finite())
                .ok_or_else(|| {
                    OperandError::Invalid("no paraxial reference height for this mapping".into())
                })?;
                let c = self.chief(f, w)?;
                let real = c.ray.pos.x * u[0] + c.ray.pos.y * u[1];
                Ok(100.0 * (real - reference) / reference)
            }
            LateralColour { short, long } => {
                let f = self.single_field(op.field)?;
                let (u, _, _) = self.field_direction(f)?;
                let a = self.chief(f, *short)?.ray.pos;
                let b = self.chief(f, *long)?.ray.pos;
                Ok((a.x - b.x) * u[0] + (a.y - b.y) * u[1])
            }
            RelativeIllumination => {
                let f = self.single_field(op.field)?;
                let wls = self.wavelengths(op.wavelength)?;
                weighted_mean(&wls, |&w| {
                    let axis = self.projected_solid_angle(ON_AXIS, w)?;
                    if !(axis > 0.0) {
                        return Err(OperandError::RayFailure(
                            "no unvignetted on-axis pupil".into(),
                        ));
                    }
                    Ok(self.projected_solid_angle(f, w)? / axis)
                })
            }
            _ => Err(OperandError::Unsupported),
        }
    }

    fn wavefront_ms(
        &self,
        f: FieldKey,
        w: usize,
        reference: WavefrontReference,
        s: &PupilSampling,
    ) -> Result<f64, OperandError> {
        let g = self.group(f, w, s)?;
        let fit = match reference {
            WavefrontReference::Centroid => self.piston_tilt(f, &[(w, 1.0)], s)?,
            WavefrontReference::Chief => {
                // Piston only: the weighted mean OPD.
                let mut sw = 0.0;
                let mut so = mklight_core::math::KahanSum::default();
                for (n, r) in g.nodes.iter().zip(&g.rays) {
                    if let Some(o) = r.opd {
                        sw += n.weight;
                        so.add(n.weight * o);
                    }
                }
                if !(sw > 0.0) {
                    return Err(OperandError::RayFailure("no OPD in the pupil".into()));
                }
                [so.value() / sw, 0.0, 0.0]
            }
        };
        let mut sw = 0.0;
        let mut s2 = mklight_core::math::KahanSum::default();
        for (n, r) in g.nodes.iter().zip(&g.rays) {
            if let Some(o) = r.opd {
                let d = o - (fit[0] + fit[1] * n.px + fit[2] * n.py);
                sw += n.weight;
                s2.add(n.weight * d * d);
            }
        }
        if !(sw > 0.0) {
            return Err(OperandError::RayFailure("no OPD in the pupil".into()));
        }
        Ok(s2.value() / sw)
    }

    fn real_ray(
        &self,
        op: &Operand,
        quantity: RayQuantity,
        surface: Option<SurfaceId>,
        ppx: f64,
        ppy: f64,
        global: bool,
    ) -> OperandValue {
        let f = self.single_field(op.field)?;
        let w = self.single_wavelength(op.wavelength)?;
        let j = self.surface_index(surface)?;
        let mut rec = Vec::new();
        let opts = TraceOptions {
            last_surface: Some(j),
            ..self.opts
        };
        self.ray(f, w, ppx, ppy, &opts, Some(&mut rec))?;
        let r = rec
            .get(j)
            .ok_or_else(|| OperandError::Invalid("surface not reached".into()))?;
        let (p, d) = if global {
            let g = to_global(
                self.tr.system(),
                j,
                &mklight_trace::Ray::new(r.point, r.dir),
            );
            (g.pos, g.dir)
        } else {
            (r.point, r.dir)
        };
        Ok(match quantity {
            RayQuantity::X => p.x,
            RayQuantity::Y => p.y,
            RayQuantity::Z => p.z,
            RayQuantity::L => d.x,
            RayQuantity::M => d.y,
            RayQuantity::N => d.z,
            RayQuantity::AngleX => d.x.atan2(d.z.abs()).to_degrees(),
            RayQuantity::AngleY => d.y.atan2(d.z.abs()).to_degrees(),
            RayQuantity::IncidenceAngle => r.cos_incidence.clamp(0.0, 1.0).acos().to_degrees(),
            RayQuantity::OpticalPath => r.opl,
        })
    }

    /// `1 / (2 n' sin θ')` from the on-axis real marginal ray `(0, 1)` and
    /// the chief ray, at wavelength `w`.
    fn real_working_f_number(&self, w: usize) -> Result<f64, OperandError> {
        let m = self.ray(ON_AXIS, w, 0.0, 1.0, &self.opts, None)?;
        let c = self.ray(ON_AXIS, w, 0.0, 0.0, &self.opts, None)?;
        let sin = m.ray.dir.cross(c.ray.dir).norm();
        let n = self.tr.system().n_before(self.tr.system().image_index(), w);
        if !(sin > 0.0) {
            return Err(OperandError::Invalid("afocal image space".into()));
        }
        Ok(1.0 / (2.0 * n * sin))
    }

    /// Projected solid angle of the image-space ray bundle of field `f`
    /// (area of the bundle in direction-cosine space), from a polar pupil
    /// grid whose cells count only when all corners reach the image.
    fn projected_solid_angle(&self, f: FieldKey, w: usize) -> Result<f64, OperandError> {
        let field = self.field(f)?;
        let nr = self.ev.settings.illumination_rings.max(2);
        let na = self.ev.settings.illumination_arms.max(6);
        let dir = |px: f64, py: f64| -> Option<Vec3> {
            let r = self.tr.trace(field, w, px, py, &self.opts);
            r.status.is_ok().then_some(r.ray.dir)
        };
        let centre = dir(0.0, 0.0);
        // rings[i][j] for ρ = (i + 1)/nr.
        let rings: Vec<Vec<Option<Vec3>>> = (1..=nr)
            .map(|i| {
                let rho = i as f64 / nr as f64;
                (0..na)
                    .map(|j| {
                        let t = std::f64::consts::TAU * j as f64 / na as f64;
                        dir(rho * t.cos(), rho * t.sin())
                    })
                    .collect()
            })
            .collect();
        let area = |pts: &[Option<Vec3>]| -> Option<f64> {
            let mut a = 0.0;
            for k in 0..pts.len() {
                let (p, q) = (pts[k]?, pts[(k + 1) % pts.len()]?);
                a += p.x * q.y - q.x * p.y;
            }
            Some(0.5 * a.abs())
        };
        let mut s = mklight_core::math::KahanSum::default();
        for j in 0..na {
            let j1 = (j + 1) % na;
            if let Some(a) = area(&[centre, rings[0][j], rings[0][j1]]) {
                s.add(a);
            }
            for i in 0..nr - 1 {
                if let Some(a) = area(&[rings[i][j], rings[i + 1][j], rings[i + 1][j1], rings[i][j1]])
                {
                    s.add(a);
                }
            }
        }
        Ok(s.value())
    }
}
