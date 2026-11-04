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
//! * A half-pupil ([`PupilSymmetry::Bilateral`]) sampling is traced as such
//!   only when the candidate is mirror-symmetric about the meridional plane
//!   (`is_meridionally_symmetric`) and the field point is on the y axis.
//!   Otherwise its mirrored nodes are traced too (half weights each):
//!   references and aggregates are then full-pupil values, and per-ray
//!   operands of the group report the [`RayGroup`] pair value, so a
//!   half-pupil merit still equals the full-pupil merit.
//! * Relative illumination comes from energy conservation (étendue of the
//!   transmitted beam over the image area per unit field), see
//!   `Candidate::irradiance`.
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

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::f64::consts::{PI, TAU};
use std::sync::{Arc, Mutex, OnceLock};

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use mklight_core::job::{Precision, RunTimer};
use mklight_core::materials::MaterialLibrary;
use mklight_core::{
    JobContext, MkResult, OpticalSystem, ResolvedSystem, RunInfo, SemiDiameter, Snapshot,
    SurfaceId, SurfaceShape, Vec3,
};
use mklight_trace::{
    AimSeeds, ObjectSource, OpdReference, ParaxialModel, ParaxialXY, PreparedField,
    SemiDiameterSettings, SequentialTracer, SurfaceRecord, TraceOptions, TraceResult,
    TracerSettings, fix_semi_diameters, trace::to_global,
};

use super::default::is_meridionally_symmetric;
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
use crate::deriv::FdScheme;

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
    /// Radial samples per azimuth of the relative-illumination pupil
    /// integration (vignetting boundaries between samples are then located
    /// by bisection).
    pub illumination_rings: usize,
    /// Azimuths of the relative-illumination pupil integration.
    pub illumination_arms: usize,
}

impl Default for TraceEvalSettings {
    fn default() -> Self {
        TraceEvalSettings {
            tracer: TracerSettings::default(),
            clip: true,
            semi_diameter_azimuths: 16,
            illumination_rings: 12,
            illumination_arms: 32,
        }
    }
}

/// The result of an optimisation or tolerancing run with its [`RunInfo`]
/// (from [`TraceEvaluator::run`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunResult<T> {
    pub result: T,
    pub info: RunInfo,
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

    /// Run `f` — an optimisation or tolerancing driver (`least_squares`,
    /// `hammer`, `sensitivity`, `monte_carlo`, …) on a problem using this
    /// evaluator, started from `start` — and return its result with the
    /// [`RunInfo`] attached: the timer brackets `f`, and the snapshot hash,
    /// `settings`, `seed`, backend and tracer warnings are recorded as in
    /// [`Self::run_info`]. Prefer this to calling `run_info` by hand.
    pub fn run<T>(
        &self,
        start: &OpticalSystem,
        analysis: &str,
        settings: &impl Serialize,
        seed: Option<u64>,
        f: impl FnOnce() -> MkResult<T>,
    ) -> MkResult<RunResult<T>> {
        let timer = RunTimer::start();
        let result = f()?;
        let info = self.run_info(timer, start, analysis, settings, seed)?;
        Ok(RunResult { result, info })
    }

    /// Snapshot → `ResolvedSystem` → `SequentialTracer` for one system.
    /// Invalid systems give `Invalid`, paraxial failures `RayFailure`.
    pub fn tracer(&self, s: &OpticalSystem) -> Result<SequentialTracer, OperandError> {
        self.seeded_tracer(s, &Seeding::None)
    }

    /// [`Self::tracer`], seeded from or recording into an anchor's solves.
    fn seeded_tracer(
        &self,
        s: &OpticalSystem,
        seeding: &Seeding,
    ) -> Result<SequentialTracer, OperandError> {
        let snap = Snapshot::new(s.clone()).map_err(|e| OperandError::Invalid(e.to_string()))?;
        let rs = Arc::new(
            ResolvedSystem::new(&snap, self.library.clone())
                .map_err(|e| OperandError::Invalid(e.to_string()))?,
        );
        match seeding {
            Seeding::Read(seeds) => {
                SequentialTracer::with_seeds(rs, self.settings.tracer, seeds.clone())
            }
            _ => SequentialTracer::with_settings(rs, self.settings.tracer),
        }
        .map_err(|e| OperandError::RayFailure(e.to_string()))
    }

    /// Values of `operands` (all supported) for one system.
    pub fn evaluate_system(
        &self,
        s: &OpticalSystem,
        operands: &[&Operand],
        ctx: &JobContext,
    ) -> MkResult<Vec<OperandValue>> {
        self.evaluate_one(s, operands, ctx, true)
    }

    /// [`evaluate_system`](Self::evaluate_system), tracing the candidate's
    /// ray groups in parallel or not (batches parallelise over candidates).
    pub(super) fn evaluate_one(
        &self,
        s: &OpticalSystem,
        operands: &[&Operand],
        ctx: &JobContext,
        parallel: bool,
    ) -> MkResult<Vec<OperandValue>> {
        Ok(self
            .evaluate_seeded(s, operands, ctx, parallel, &Seeding::None)?
            .0)
    }

    /// [`Self::evaluate_one`] with the tracer seeded by `seeding`; an
    /// anchor (`Seeding::Record`) also returns its recorded solves.
    fn evaluate_seeded(
        &self,
        s: &OpticalSystem,
        operands: &[&Operand],
        ctx: &JobContext,
        parallel: bool,
        seeding: &Seeding,
    ) -> MkResult<(Vec<OperandValue>, Option<Arc<AimSeeds>>)> {
        ctx.check()?;
        let auto_sd = has_auto_semi_diameters(s);
        let needs_tracer = operands.iter().any(|o| {
            (!is_model_kind(&o.kind) && !matches!(o.kind, OperandKind::ToleranceSensitivity { .. }))
                || (auto_sd && needs_semi_diameters(&o.kind))
        });
        let mut recorded = None;
        let tracer = if needs_tracer {
            let mut t = self.seeded_tracer(s, seeding);
            if let (Seeding::Record, Ok(t)) = (seeding, &mut t) {
                recorded = Some(t.record_seeds());
            }
            Some(t)
        } else {
            None
        };
        let cand = match &tracer {
            Some(Ok(tr)) => Some(Candidate::build(self, tr, operands, ctx, parallel)?),
            _ => None,
        };
        let mut fixed_sd: Option<OpticalSystem> = None;
        let mut memo = GroupMemo::default();
        let mut out = Vec::with_capacity(operands.len());
        for op in operands {
            let v = if is_model_kind(&op.kind) {
                if auto_sd && needs_semi_diameters(&op.kind) {
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
                    (_, Some(c)) => c.value_with(op, &mut memo),
                    (Some(Err(e)), _) => Err(e.clone()),
                    _ => Err(OperandError::Unsupported),
                }
            };
            out.push(v);
        }
        Ok((out, recorded))
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
        // Candidates also parallelise inside (field preparation, chiefs,
        // groups): work stealing then balances the tail of a batch that
        // is not a multiple of the pool (49 FD columns on 12 workers).
        let inner = true;
        systems
            .par_iter()
            .map(|s| self.evaluate_one(s, operands, ctx, inner))
            .collect()
    }

    /// Real aiming: aims are seeded from the anchor (`evaluate_batch_near`),
    /// so a finite difference must take its base value from the same batch.
    fn anchored_differences(&self, base: &OpticalSystem) -> bool {
        matches!(
            self.settings.tracer.aiming.unwrap_or(base.ray_aiming),
            mklight_core::model::RayAiming::Real { .. }
        )
    }

    /// Candidates near `anchor` (finite-difference neighbours): the anchor
    /// is traced once and records its solves ([`AimSeeds`]), which seed the
    /// field, chief and pupil-ray aims of every candidate. A seeded solve
    /// resolves its change from the anchor to 1e-6 (`mklight_trace`
    /// `seeded_tolerance`), far below the aiming tolerance, so values agree
    /// with [`Self::evaluate_batch`] to within that tolerance and the probes
    /// of one Jacobian, the re-evaluated base point included, share one
    /// consistent solve path.
    fn evaluate_batch_near(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        anchor: Option<&OpticalSystem>,
        ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        let seeds = match anchor {
            Some(a) if systems.len() >= MIN_SEEDED_BATCH && self.anchored_differences(a) => {
                self.evaluate_seeded(a, operands, ctx, true, &Seeding::Record)?
                    .1
            }
            _ => None,
        };
        let Some(seeds) = seeds else {
            return self.evaluate_batch(systems, operands, ctx);
        };
        let seeding = Seeding::Read(seeds);
        systems
            .par_iter()
            .map(|s| Ok(self.evaluate_seeded(s, operands, ctx, true, &seeding)?.0))
            .collect()
    }

    /// Wavefront operands: path lengths of ~100 mm carry an absolute
    /// rounding noise of ~1e-14 mm (~2e-11 waves), which the `√ε` / `∛ε`
    /// defaults, tuned to relative noise, difference into ~1e-4 relative
    /// Jacobian errors; [`WAVEFRONT_FD_RELATIVE_STEP_FORWARD`] /
    /// [`WAVEFRONT_FD_RELATIVE_STEP_CENTRAL`] balance it against truncation.
    fn fd_relative_step(&self, kind: &OperandKind, scheme: FdScheme) -> Option<f64> {
        match kind {
            OperandKind::ToleranceSensitivity { criterion, .. } => {
                self.fd_relative_step(criterion, scheme)
            }
            OperandKind::WavefrontAberration { .. } | OperandKind::WavefrontRms { .. } => {
                Some(match scheme {
                    FdScheme::Forward => WAVEFRONT_FD_RELATIVE_STEP_FORWARD,
                    FdScheme::Central => WAVEFRONT_FD_RELATIVE_STEP_CENTRAL,
                })
            }
            _ => None,
        }
    }
}

/// How a candidate's tracer uses an anchor's solves ([`AimSeeds`]).
enum Seeding {
    None,
    /// The anchor: record its solves.
    Record,
    /// A candidate near the anchor.
    Read(Arc<AimSeeds>),
}

/// Smallest batch whose candidates are seeded from their anchor: tracing
/// the anchor costs one extra candidate.
const MIN_SEEDED_BATCH: usize = 4;

/// Relative forward-difference step for wavefront operands on the f64
/// reference (see [`TraceEvaluator`]'s `fd_relative_step`).
pub const WAVEFRONT_FD_RELATIVE_STEP_FORWARD: f64 = 1e-6;
/// Central-difference counterpart of [`WAVEFRONT_FD_RELATIVE_STEP_FORWARD`].
pub const WAVEFRONT_FD_RELATIVE_STEP_CENTRAL: f64 = 1e-4;

// ---------------------------------------------------------------------------
// Per-candidate cache
// ---------------------------------------------------------------------------

/// A field: a system field by index or an arbitrary normalised point (bits).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum FieldKey {
    Index(usize),
    Point(u64, u64),
}

const ON_AXIS: FieldKey = FieldKey::Point(0, 0);

/// Bit-exact identity of a pupil sampling.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) struct SamplingKey {
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

pub(super) type GroupKey = (FieldKey, usize, SamplingKey);

/// One pupil ray to the image: (field, λ, bits of `px`, bits of `py`).
pub(super) type SingleKey = (FieldKey, usize, u64, u64);

/// One real ray to surface `j`: (field, λ, bits of `px`, bits of `py`, `j`).
pub(super) type RealKey = (FieldKey, usize, u64, u64, usize);

/// Chief ray and OPD reference sphere of one (field, wavelength).
pub(super) struct ChiefData {
    pub(super) chief: Result<TraceResult, OperandError>,
    pub(super) reference: Result<OpdReference, OperandError>,
}

/// One traced pupil node: image-surface `(x, y)` (local frame) and OPD
/// (waves), `None` where the ray (or its OPD) failed.
#[derive(Clone, Copy)]
pub(super) struct NodeRay {
    pub(super) pos: Option<[f64; 2]>,
    pub(super) opd: Option<f64>,
}

impl NodeRay {
    pub(super) const FAILED: NodeRay = NodeRay {
        pos: None,
        opd: None,
    };
}

/// A real ray's state on surface `j` (local frame), as in `SurfaceRecord`:
/// the point, the direction after the interaction and the optical path from
/// the start of the trace.
#[derive(Clone, Copy, Debug)]
pub(super) struct RealHit {
    pub(super) point: Vec3,
    pub(super) dir: Vec3,
    pub(super) opl: f64,
}

pub(super) struct GroupRays {
    /// The sampling's nodes, followed by their mirrors `(−px, py)` when a
    /// half pupil is traced in full (asymmetric candidate); weights then
    /// halved.
    pub(super) nodes: Vec<PupilNode>,
    /// Symmetry of the traced node set (`None` once mirrored).
    pub(super) symmetry: PupilSymmetry,
    pub(super) rays: Vec<NodeRay>,
    /// Node index by the bits of `(px, py)` (only for more than 64 nodes).
    index: HashMap<(u64, u64), usize>,
}

/// What an operand needs traced before evaluation. `singles` (per-ray
/// operands without a ray group) and `real` (real rays recorded on a
/// surface) are traced on demand by the CPU path; an external tracer (the
/// GPU evaluator) traces them up front.
#[derive(Default)]
pub(super) struct Plan {
    pub(super) points: BTreeSet<FieldKey>,
    pub(super) chiefs: BTreeSet<(FieldKey, usize)>,
    pub(super) groups: BTreeMap<GroupKey, PupilSampling>,
    pub(super) singles: BTreeSet<SingleKey>,
    pub(super) real: BTreeSet<RealKey>,
}

pub(super) fn status_error(r: &TraceResult) -> OperandError {
    OperandError::RayFailure(format!("{:?}", r.status))
}

fn axis_value(p: [f64; 2], axis: Axis) -> f64 {
    match axis {
        Axis::X => p[0],
        Axis::Y => p[1],
    }
}

/// Transmitted radial intervals of one pupil azimuth and the surfaces that
/// bound them (relative illumination).
#[derive(Default)]
struct Arm {
    intervals: Vec<(f64, f64)>,
    bounds: Vec<Option<usize>>,
}

/// Identity of what a per-ray operand of a ray group plans (its traces do not
/// depend on the pupil point): kind, reference, selectors and group.
type GroupPlanKey = (u8, u8, (u8, u64, u64), (u8, u64), SamplingKey, bool);

fn group_plan_key(op: &Operand) -> Option<GroupPlanKey> {
    let field = match op.field {
        FieldSelector::All => (0, 0, 0),
        FieldSelector::Index { index } => (1, index as u64, 0),
        FieldSelector::Normalized { hx, hy } => (2, hx.to_bits(), hy.to_bits()),
    };
    let wl = match op.wavelength {
        WavelengthSelector::Primary => (0, 0),
        WavelengthSelector::All => (1, 0),
        WavelengthSelector::Index { index } => (2, index as u64),
    };
    let (kind, reference, g) = match &op.kind {
        OperandKind::RayAberration {
            reference,
            group: Some(g),
            ..
        } => (0, (*reference == SpotReference::Chief) as u8, g),
        OperandKind::WavefrontAberration { group: Some(g), .. } => (1, 0, g),
        _ => return None,
    };
    Some((
        kind,
        reference,
        field,
        wl,
        SamplingKey::of(&g.sampling),
        g.scope == CentroidScope::Polychromatic,
    ))
}

/// The last ray group a per-ray operand was read from
/// ([`Candidate::value_with`]).
#[derive(Default)]
pub(super) struct GroupMemo<'c> {
    ctx: Option<GroupCtx<'c>>,
}

struct GroupCtx<'c> {
    field: FieldSelector,
    wavelength: WavelengthSelector,
    group: RayGroup,
    wavefront: bool,
    chief_ref: bool,
    rays: &'c GroupRays,
    /// Spot reference `[x, y, 0]` or piston/tilt, or the error of every
    /// operand of the group.
    reference: Result<[f64; 3], OperandError>,
    /// The OPD reference sphere exists (wavefront operands).
    sphere: Result<(), OperandError>,
    mirrored: bool,
    /// Index after the last node read.
    next: usize,
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

/// Half-pupil pair value of a per-ray operand on an asymmetric candidate
/// ([`RayGroup`] contract): `s·√((a² + b²)/2)` for the values `a` at the node
/// and `b` at its mirror, `s` the sign of the mirror-even part (`a − b` for a
/// quantity odd in `px`, else `a + b`).
fn pair_value(a: f64, b: f64, odd: bool) -> f64 {
    let even = if odd { a - b } else { a + b };
    let m = (0.5 * (a * a + b * b)).sqrt();
    if even < 0.0 { -m } else { m }
}

/// `items.map(f)` in order, on the rayon pool when `parallel`.
fn pmap<T: Sync, R: Send>(
    items: &[T],
    parallel: bool,
    f: impl Fn(&T) -> R + Sync + Send,
) -> Vec<R> {
    if parallel && items.len() > 1 {
        items.par_iter().map(f).collect()
    } else {
        items.iter().map(f).collect()
    }
}

/// Weights, image `x` and image `y` of the surviving rays of a spot, and the
/// symmetry of the traced node set.
type SpotRays = (Vec<f64>, Vec<f64>, Vec<f64>, PupilSymmetry);

/// Memo key of a group reference: (field, operand λ, sampling,
/// polychromatic scope, wavefront (piston/tilt) rather than spot centroid).
type RefKey = (FieldKey, usize, SamplingKey, bool, bool);

pub(super) struct Candidate<'a> {
    /// Completed P3 moments; absent on the unchanged reference evaluator.
    pub(super) resident_wavefits: BTreeMap<GroupKey, [[f64; 4]; 2]>,
    pub(super) resident_moments: BTreeMap<GroupKey, [f64; 10]>,
    /// Initial RI radial scans; adaptive boundaries retain the same reference method.
    pub(super) resident_visibility: BTreeMap<SingleKey, Option<usize>>,
    pub(super) visibility_lookups: [std::sync::atomic::AtomicU64; 2],
    pub(super) cancel: Option<mklight_core::CancelToken>,
    ev: &'a TraceEvaluator,
    pub(super) tr: &'a SequentialTracer,
    opts: TraceOptions,
    points: BTreeMap<FieldKey, Result<PreparedField, OperandError>>,
    pub(super) chiefs: BTreeMap<(FieldKey, usize), ChiefData>,
    pub(super) groups: BTreeMap<GroupKey, GroupRays>,
    /// Rays traced up front by an external tracer, consulted before tracing
    /// on the CPU (empty on the CPU path).
    pub(super) singles: BTreeMap<SingleKey, NodeRay>,
    pub(super) real_rays: BTreeMap<RealKey, Result<RealHit, OperandError>>,
    /// Centroids `[cx, cy, 0]` and piston/tilts of ray groups, computed on
    /// first use (every ray operand of a group shares its reference).
    refs: Mutex<BTreeMap<RefKey, Result<[f64; 3], OperandError>>>,
    /// Is the candidate mirror-symmetric about the meridional plane?
    symmetric: bool,
    /// The x and y first-order sweeps, computed on first use.
    xy: OnceLock<Result<ParaxialXY, OperandError>>,
}

impl<'a> Candidate<'a> {
    fn build(
        ev: &'a TraceEvaluator,
        tr: &'a SequentialTracer,
        operands: &[&Operand],
        ctx: &JobContext,
        parallel: bool,
    ) -> MkResult<Self> {
        let (mut c, plan) = Self::planned(ev, tr, operands);
        // Parallel tracing inside one candidate only pays for a few hundred
        // rays or more (measured on the Double Gauss, docs/OPTIMIZATION.md).
        let rays: usize = plan.groups.values().map(|s| s.rings * s.arms).sum();
        let parallel = parallel && rays >= 256;
        c.prepare_points(&plan, parallel);
        ctx.check()?;
        c.trace_cpu(&plan, ctx, parallel)?;
        Ok(c)
    }

    /// An empty candidate and the plan of what `operands` need traced.
    pub(super) fn planned(
        ev: &'a TraceEvaluator,
        tr: &'a SequentialTracer,
        operands: &[&Operand],
    ) -> (Self, Plan) {
        let opts = TraceOptions {
            clip: ev.settings.clip,
            ..tr.options()
        };
        let c = Candidate {
            resident_visibility: BTreeMap::new(),
            visibility_lookups: Default::default(),
            cancel: None,
            resident_moments: BTreeMap::new(),
            resident_wavefits: BTreeMap::new(),
            ev,
            tr,
            opts,
            points: BTreeMap::new(),
            chiefs: BTreeMap::new(),
            groups: BTreeMap::new(),
            singles: BTreeMap::new(),
            real_rays: BTreeMap::new(),
            refs: Mutex::new(BTreeMap::new()),
            symmetric: is_meridionally_symmetric(tr.system().system()),
            xy: OnceLock::new(),
        };
        let mut plan = Plan::default();
        // The per-ray operands of one ray group (hundreds in a default
        // merit) all plan the same traces: plan each group once.
        let mut seen = std::collections::HashSet::new();
        for op in operands {
            if let Some(k) = group_plan_key(op)
                && !seen.insert(k)
            {
                continue;
            }
            c.plan(op, &mut plan);
        }
        // Group (field, λ) pairs need their chief ray / reference sphere.
        for &(f, w, _) in plan.groups.keys() {
            plan.chiefs.insert((f, w));
        }
        let fields: Vec<FieldKey> = (plan.chiefs.iter().map(|k| k.0))
            .chain(plan.singles.iter().map(|k| k.0))
            .chain(plan.real.iter().map(|k| k.0))
            .collect();
        for f in fields {
            if let FieldKey::Point(..) = f {
                plan.points.insert(f);
            }
        }
        (c, plan)
    }

    /// Stage 1: prepare the arbitrary field points of `plan`.
    pub(super) fn prepare_points(&mut self, plan: &Plan, parallel: bool) {
        let tr = self.tr;
        let points: Vec<FieldKey> = plan.points.iter().copied().collect();
        self.points = pmap(&points, parallel, |&k| {
            let FieldKey::Point(hx, hy) = k else {
                unreachable!()
            };
            let fp = mklight_trace::FieldPoint::normalized(f64::from_bits(hx), f64::from_bits(hy));
            let f = tr
                .prepare_field(fp)
                .map_err(|e| OperandError::RayFailure(e.to_string()));
            (k, f)
        })
        .into_iter()
        .collect();
    }

    /// Stages 2 and 3 on the CPU: chief rays, reference spheres and pupil
    /// samplings of `plan`.
    fn trace_cpu(&mut self, plan: &Plan, ctx: &JobContext, parallel: bool) -> MkResult<()> {
        let tr = self.tr;
        let c = &*self;
        // Stage 2: chief rays and reference spheres.
        let chiefs: Vec<(FieldKey, usize)> = plan.chiefs.iter().copied().collect();
        let chiefs = pmap(&chiefs, parallel, |&(f, w)| {
            let d = match c.field(f) {
                Ok(field) => {
                    let chief = tr.chief(field, w);
                    let reference = tr
                        .opd_reference(field, w)
                        .map_err(|s| OperandError::RayFailure(format!("chief ray: {s:?}")));
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
        .into_iter()
        .collect();
        self.chiefs = chiefs;
        ctx.check()?;
        // Stage 3: pupil samplings.
        let c = &*self;
        let groups: Vec<(GroupKey, PupilSampling)> =
            plan.groups.iter().map(|(k, s)| (*k, *s)).collect();
        let traced = pmap(&groups, parallel, |&(key, sampling)| {
            ctx.check()?;
            Ok((key, c.trace_group(key, &sampling)))
        });
        self.groups = traced.into_iter().collect::<MkResult<_>>()?;
        Ok(())
    }

    /// Must the half-pupil `sampling` be traced in full at field `f`? (The
    /// candidate, or the field point, is not mirror-symmetric.)
    fn mirrored(&self, f: FieldKey, sampling: &PupilSampling) -> bool {
        sampling.symmetry == PupilSymmetry::Bilateral
            && (!self.symmetric || matches!(f, FieldKey::Point(hx, _) if f64::from_bits(hx) != 0.0))
    }

    /// The nodes traced for `sampling` at field `f` and the symmetry of the
    /// node set: the sampling's nodes, followed by their mirrors (half
    /// weights each) when a half pupil must be traced in full.
    pub(super) fn group_nodes(
        &self,
        f: FieldKey,
        sampling: &PupilSampling,
    ) -> (Vec<PupilNode>, PupilSymmetry) {
        self.group_nodes_from(f, sampling, sampling.nodes())
    }
    pub(super) fn group_nodes_from(
        &self,
        f: FieldKey,
        sampling: &PupilSampling,
        mut nodes: Vec<PupilNode>,
    ) -> (Vec<PupilNode>, PupilSymmetry) {
        let mut symmetry = sampling.symmetry;
        if self.mirrored(f, sampling) {
            let mirrors: Vec<PupilNode> = nodes
                .iter_mut()
                .map(|n| {
                    n.weight *= 0.5;
                    PupilNode { px: -n.px, ..*n }
                })
                .collect();
            nodes.extend(mirrors);
            symmetry = PupilSymmetry::None;
        }
        (nodes, symmetry)
    }

    /// A traced group from its nodes and their rays (same order).
    pub(super) fn group_rays(
        nodes: Vec<PupilNode>,
        symmetry: PupilSymmetry,
        rays: Vec<NodeRay>,
    ) -> GroupRays {
        // Small groups are scanned (`node_in`). Resident aggregate-only
        // groups carry sampling metadata but do not materialise host rays.
        let index = if nodes.len() > 64 && !rays.is_empty() {
            nodes
                .iter()
                .enumerate()
                .map(|(i, n)| ((n.px.to_bits(), n.py.to_bits()), i))
                .collect()
        } else {
            HashMap::new()
        };
        GroupRays {
            nodes,
            symmetry,
            rays,
            index,
        }
    }

    fn trace_group(&self, (f, w, _): GroupKey, sampling: &PupilSampling) -> GroupRays {
        let (nodes, symmetry) = self.group_nodes(f, sampling);
        let reference = self.chiefs[&(f, w)].reference.as_ref().ok();
        let rays = match self.field(f) {
            Ok(field) => {
                // Aimed and traced together (bitwise equal to one by one).
                let requests: Vec<_> = nodes.iter().map(|n| (field, w, [n.px, n.py])).collect();
                self.tr
                    .trace_many(&requests, &self.opts)
                    .into_iter()
                    .map(|r| {
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
                    .collect()
            }
            Err(_) => vec![NodeRay::FAILED; nodes.len()],
        };
        Self::group_rays(nodes, symmetry, rays)
    }

    // ----- selectors -----------------------------------------------------

    fn system(&self) -> &OpticalSystem {
        self.tr.system().system()
    }

    fn primary(&self) -> usize {
        self.tr.system().primary_wavelength
    }

    /// `(wavelength, weight)` of a selector.
    pub(super) fn wavelengths(
        &self,
        sel: WavelengthSelector,
    ) -> Result<Vec<(usize, f64)>, OperandError> {
        let nw = self.tr.num_wavelengths();
        match sel {
            WavelengthSelector::Primary => Ok(vec![(self.primary(), 1.0)]),
            WavelengthSelector::Index { index } if index < nw => Ok(vec![(index, 1.0)]),
            WavelengthSelector::Index { index } => Err(OperandError::Invalid(format!(
                "wavelength {index} does not exist"
            ))),
            WavelengthSelector::All => {
                let w = &self.tr.system().wavelength_weights;
                let v: Vec<(usize, f64)> =
                    (0..nw).filter(|&i| w[i] > 0.0).map(|i| (i, w[i])).collect();
                if v.is_empty() {
                    Err(OperandError::Invalid(
                        "no wavelength with positive weight".into(),
                    ))
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
            WavelengthSelector::Primary => Ok(self.primary()),
            WavelengthSelector::Index { index } if index < self.tr.num_wavelengths() => Ok(index),
            s => Ok(self.wavelengths(s)?[0].0),
        }
    }

    /// `(field, weight)` of a selector.
    pub(super) fn fields(&self, sel: FieldSelector) -> Result<Vec<(FieldKey, f64)>, OperandError> {
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
                    Err(OperandError::Invalid(
                        "no field with positive weight".into(),
                    ))
                } else {
                    Ok(v)
                }
            }
        }
    }

    fn single_field(&self, sel: FieldSelector) -> Result<FieldKey, OperandError> {
        match sel {
            FieldSelector::All => Err(OperandError::Invalid("operand needs a single field".into())),
            FieldSelector::Index { index } if index < self.system().fields.fields.len() => {
                Ok(FieldKey::Index(index))
            }
            s => Ok(self.fields(s)?[0].0),
        }
    }

    pub(super) fn field(&self, k: FieldKey) -> Result<&PreparedField, OperandError> {
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
                reference,
                group: g,
                px,
                py,
                ..
            } => {
                for &(f, _) in &fields {
                    for &(w, _) in &wls {
                        match g {
                            Some(g) => {
                                for (gw, _) in self.group_wavelengths(g, w) {
                                    group(f, gw, &g.sampling);
                                }
                            }
                            None => {
                                p.singles.insert((f, w, px.to_bits(), py.to_bits()));
                            }
                        }
                    }
                    if *reference == SpotReference::Chief {
                        p.chiefs.insert((f, primary));
                    }
                    p.points.insert(f);
                }
            }
            WavefrontAberration {
                group: g, px, py, ..
            } => {
                for &(f, _) in &fields {
                    for &(w, _) in &wls {
                        match g {
                            Some(g) => {
                                for (gw, _) in self.group_wavelengths(g, w) {
                                    group(f, gw, &g.sampling);
                                }
                            }
                            None => {
                                p.singles.insert((f, w, px.to_bits(), py.to_bits()));
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
                // Out-of-range wavelengths are reported by `value`.
                if *short < self.tr.num_wavelengths() && *long < self.tr.num_wavelengths() {
                    for &(f, _) in &fields {
                        p.chiefs.insert((f, *short));
                        p.chiefs.insert((f, *long));
                    }
                }
            }
            RealRay {
                quantity,
                surface,
                px,
                py,
                ..
            } => {
                for &(f, _) in &fields {
                    p.points.insert(f);
                    // The incidence angle needs the surface normal: always
                    // traced on demand.
                    if *quantity != RayQuantity::IncidenceAngle
                        && let Ok(j) = self.surface_index(*surface)
                    {
                        for &(w, _) in &wls {
                            p.real.insert((f, w, px.to_bits(), py.to_bits(), j));
                        }
                    }
                }
            }
            ChiefRayAngle { .. } => {
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
    fn node_in(g: &GroupRays, px: f64, py: f64) -> Option<&NodeRay> {
        Self::node_index(g, px, py).map(|i| &g.rays[i])
    }

    /// Index of node `(px, py)` in a traced group (bit-exact).
    fn node_index(g: &GroupRays, px: f64, py: f64) -> Option<usize> {
        let (bx, by) = (px.to_bits(), py.to_bits());
        // A scan beats hashing for the usual few dozen nodes.
        if g.nodes.len() <= 64 {
            return g
                .nodes
                .iter()
                .position(|n| n.px.to_bits() == bx && n.py.to_bits() == by);
        }
        g.index.get(&(bx, by)).copied()
    }

    /// [`Self::value`], reading the per-ray operands of a ray group through
    /// `memo`: consecutive operands of one group (the layout of a default
    /// merit) resolve the group, its reference and the node position once.
    /// Values are identical to `value`'s.
    pub(super) fn value_with<'c>(&'c self, op: &Operand, memo: &mut GroupMemo<'c>) -> OperandValue {
        use OperandKind::*;
        let (wavefront, chief_ref, g, px, py, axis) = match &op.kind {
            RayAberration {
                axis,
                px,
                py,
                reference,
                group: Some(g),
            } => (
                false,
                *reference == SpotReference::Chief,
                g,
                *px,
                *py,
                *axis,
            ),
            WavefrontAberration {
                px,
                py,
                reference,
                group: Some(g),
            } => (
                true,
                *reference == WavefrontReference::Chief,
                g,
                *px,
                *py,
                Axis::Y,
            ),
            _ => return self.value(op),
        };
        let same = memo.ctx.as_ref().is_some_and(|c| {
            c.wavefront == wavefront
                && c.chief_ref == chief_ref
                && c.field == op.field
                && c.wavelength == op.wavelength
                && c.group == *g
        });
        if !same {
            memo.ctx = self.group_ctx(op, g, wavefront, chief_ref);
        }
        let Some(c) = memo.ctx.as_mut() else {
            return self.value(op);
        };
        let r = match &c.reference {
            Ok(r) => *r,
            Err(e) => return Err(e.clone()),
        };
        if let Err(e) = &c.sphere {
            return Err(e.clone());
        }
        // The node, by position after the previous one (the merit's order),
        // else by search; not in the group: the general path.
        let (bx, by) = (px.to_bits(), py.to_bits());
        let nodes = &c.rays.nodes;
        let hint = [c.next, c.next.wrapping_sub(1)].into_iter().find(|&i| {
            i < nodes.len() && nodes[i].px.to_bits() == bx && nodes[i].py.to_bits() == by
        });
        let Some(i) = hint.or_else(|| Self::node_index(c.rays, px, py)) else {
            return self.value(op);
        };
        c.next = i + 1;
        let e = |n: &NodeRay, x: f64| -> OperandValue {
            if wavefront {
                let v = n
                    .opd
                    .ok_or_else(|| OperandError::RayFailure("ray or OPD failed".into()))?;
                Ok(v - (r[0] + r[1] * x + r[2] * py))
            } else {
                let p = n
                    .pos
                    .ok_or_else(|| OperandError::RayFailure("ray failed".into()))?;
                Ok(axis_value(p, axis) - axis_value([r[0], r[1]], axis))
            }
        };
        let a = e(&c.rays.rays[i], px)?;
        if !c.mirrored {
            return Ok(a);
        }
        let Some(m) = Self::node_index(c.rays, -px, py) else {
            return self.value(op);
        };
        let odd = !wavefront && axis == Axis::X;
        Ok(pair_value(a, e(&c.rays.rays[m], -px)?, odd))
    }

    /// The memo entry of a group operand (`None`: not answerable from a
    /// traced group, use the general path).
    fn group_ctx(
        &self,
        op: &Operand,
        g: &RayGroup,
        wavefront: bool,
        chief_ref: bool,
    ) -> Option<GroupCtx<'_>> {
        let f = self.single_field(op.field).ok()?;
        let w = self.single_wavelength(op.wavelength).ok()?;
        let rays = self.group(f, w, &g.sampling).ok()?;
        let reference = match (wavefront, chief_ref) {
            (true, true) => Ok([0.0; 3]),
            (true, false) => self.group_reference(f, w, g, true),
            (false, true) => self
                .chief(f, self.primary())
                .map(|c| [c.ray.pos.x, c.ray.pos.y, 0.0]),
            (false, false) => self.group_reference(f, w, g, false),
        };
        let sphere = if wavefront {
            self.reference(f, w).map(|_| ())
        } else {
            Ok(())
        };
        Some(GroupCtx {
            field: op.field,
            wavelength: op.wavelength,
            group: *g,
            wavefront,
            chief_ref,
            rays,
            reference,
            sphere,
            mirrored: self.mirrored(f, &g.sampling),
            next: 0,
        })
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
        if let Some(n) = self.singles.get(&(f, w, px.to_bits(), py.to_bits())) {
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
        if let Some(n) = self.singles.get(&(f, w, px.to_bits(), py.to_bits())) {
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
    ) -> Result<SpotRays, OperandError> {
        let (mut wt, mut xs, mut ys) = (Vec::new(), Vec::new(), Vec::new());
        let mut symmetry = s.symmetry;
        for &(w, ww) in wls {
            let g = self.group(f, w, s)?;
            symmetry = g.symmetry;
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
        Ok((wt, xs, ys, symmetry))
    }

    /// Memoised reference of ray group `g` for an operand at wavelength
    /// `w`: the spot centroid `[cx, cy, 0]`, or the piston/tilt.
    fn group_reference(
        &self,
        f: FieldKey,
        w: usize,
        g: &RayGroup,
        wavefront: bool,
    ) -> Result<[f64; 3], OperandError> {
        let key = (
            f,
            w,
            SamplingKey::of(&g.sampling),
            g.scope == CentroidScope::Polychromatic,
            wavefront,
        );
        if let Some(r) = self.refs.lock().ok().and_then(|m| m.get(&key).cloned()) {
            return r;
        }
        let wls = self.group_wavelengths(g, w);
        let r = if wavefront {
            self.piston_tilt(f, &wls, &g.sampling)
        } else {
            self.centroid(f, &wls, &g.sampling)
                .map(|c| [c[0], c[1], 0.0])
        };
        if let Ok(mut m) = self.refs.lock() {
            m.insert(key, r.clone());
        }
        r
    }

    /// Spot centroid of a group over `wls`.
    fn centroid(
        &self,
        f: FieldKey,
        wls: &[(usize, f64)],
        s: &PupilSampling,
    ) -> Result<[f64; 2], OperandError> {
        if wls.iter().all(|&(w, _)| {
            self.resident_moments
                .contains_key(&(f, w, SamplingKey::of(s)))
        }) {
            let mut total = 0.;
            let mut mean = [0.; 2];
            for &(w, ww) in wls {
                let m = self.resident_moments[&(f, w, SamplingKey::of(s))];
                let weight = m[1] * ww;
                total += weight;
                mean[0] += weight * m[2];
                mean[1] += weight * m[3];
            }
            return if total > 0. {
                Ok([mean[0] / total, mean[1] / total])
            } else {
                Err(OperandError::RayFailure("spot centroid undefined".into()))
            };
        }
        let (wt, xs, ys, symmetry) = self.spot_rays(f, wls, s)?;
        spot_centroid(&wt, &xs, &ys, symmetry)
            .ok_or_else(|| OperandError::RayFailure("spot centroid undefined".into()))
    }

    /// Piston/tilt `[a, bx, by]` of a group's OPD over `wls`.
    fn piston_tilt(
        &self,
        f: FieldKey,
        wls: &[(usize, f64)],
        s: &PupilSampling,
    ) -> Result<[f64; 3], OperandError> {
        if wls.len() == 1
            && let Some(fits) = self
                .resident_wavefits
                .get(&(f, wls[0].0, SamplingKey::of(s)))
        {
            let fit = fits[1];
            return if fit[0].is_finite() {
                Ok([fit[0], fit[1], fit[2]])
            } else {
                Err(OperandError::RayFailure(
                    "too few rays for piston and tilt".into(),
                ))
            };
        }
        let (mut px, mut py, mut wt, mut opd) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut symmetry = s.symmetry;
        for &(w, ww) in wls {
            let g = self.group(f, w, s)?;
            symmetry = g.symmetry;
            for (n, r) in g.nodes.iter().zip(&g.rays) {
                if let Some(o) = r.opd {
                    px.push(n.px);
                    py.push(n.py);
                    wt.push(n.weight * ww);
                    opd.push(o);
                }
            }
        }
        piston_tilt(&px, &py, &wt, &opd, symmetry)
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

    /// Both wavelengths of a colour operand exist.
    fn colour_pair(&self, short: usize, long: usize) -> Result<(), OperandError> {
        let nw = self.tr.num_wavelengths();
        if short < nw && long < nw {
            Ok(())
        } else {
            Err(OperandError::Invalid(format!(
                "colour wavelength index out of range ({short}, {long}; {nw} wavelengths)"
            )))
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

    pub(super) fn value(&self, op: &Operand) -> OperandValue {
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
            EflX | EflY | BflX | BflY | AnamorphicSqueeze => {
                let xy = self
                    .xy
                    .get_or_init(|| {
                        ParaxialXY::new(self.tr.system())
                            .map_err(|e| OperandError::RayFailure(e.to_string()))
                    })
                    .as_ref()
                    .map_err(Clone::clone)?;
                let wls = self.wavelengths(op.wavelength)?;
                weighted_mean(&wls, |&w| {
                    Ok(match op.kind {
                        EflX => xy.x[w].efl,
                        EflY => xy.y[w].efl,
                        BflX => xy.x[w].bfl,
                        BflY => xy.y[w].bfl,
                        _ => xy.squeeze[w],
                    })
                })
            }
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
            EntrancePupilPosition => {
                finite(fo_mean(&|fo| fo.entrance_pupil_position)?, "entrance pupil")
            }
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
                    let rs = self.tr.system();
                    let m = ParaxialModel::new(rs, w);
                    let nn = &m.reduced_index;
                    let mut s = mklight_core::math::KahanSum::default();
                    for j in 1..nn.len() {
                        // An ideal lens images a plane onto a plane without
                        // aberration (a thin lens of infinite index, φ/n → 0),
                        // so it adds no field curvature.
                        let ideal = matches!(rs.surfaces[j].shape, SurfaceShape::Paraxial { .. });
                        if m.power[j] != 0.0 && !ideal {
                            s.add(m.power[j] / (nn[j] * nn[j - 1]));
                        }
                    }
                    Ok(s.value())
                })
            }
            AxialColour { short, long } => {
                self.colour_pair(*short, *long)?;
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
                let r = match reference {
                    SpotReference::Chief => {
                        let c = self.chief(f, self.primary())?;
                        [c.ray.pos.x, c.ray.pos.y]
                    }
                    SpotReference::Centroid => {
                        let g = group.as_ref().ok_or_else(|| {
                            OperandError::Invalid("centroid reference needs a ray group".into())
                        })?;
                        let c = self.group_reference(f, w, g, false)?;
                        [c[0], c[1]]
                    }
                };
                let e = |px: f64| -> OperandValue {
                    let p = self.image_point(f, w, px, *ppy, group.as_ref())?;
                    Ok(axis_value(p, *axis) - axis_value(r, *axis))
                };
                let a = e(*ppx)?;
                match group {
                    Some(g) if self.mirrored(f, &g.sampling) => {
                        let odd = *axis == Axis::X;
                        Ok(pair_value(a, e(-*ppx)?, odd))
                    }
                    _ => Ok(a),
                }
            }
            WavefrontAberration {
                px: ppx,
                py: ppy,
                reference,
                group,
            } => {
                let f = self.single_field(op.field)?;
                let w = self.single_wavelength(op.wavelength)?;
                let fit = match reference {
                    WavefrontReference::Chief => [0.0; 3],
                    WavefrontReference::Centroid => {
                        let g = group.as_ref().ok_or_else(|| {
                            OperandError::Invalid("centroid reference needs a ray group".into())
                        })?;
                        self.group_reference(f, w, g, true)?
                    }
                };
                let e = |px: f64| -> OperandValue {
                    let v = self.opd(f, w, px, *ppy, group.as_ref())?;
                    Ok(v - (fit[0] + fit[1] * px + fit[2] * ppy))
                };
                let a = e(*ppx)?;
                match group {
                    Some(g) if self.mirrored(f, &g.sampling) => Ok(pair_value(a, e(-*ppx)?, false)),
                    _ => Ok(a),
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
                    if wls.iter().all(|&(w, _)| {
                        self.resident_moments
                            .contains_key(&(f, w, SamplingKey::of(sampling)))
                    }) {
                        let centre = match reference {
                            SpotReference::Centroid => self.centroid(f, &wls, sampling)?,
                            SpotReference::Chief => {
                                let r = self.chief(f, self.primary())?;
                                [r.ray.pos.x, r.ray.pos.y]
                            }
                        };
                        let mut sum = 0.;
                        let mut weight = 0.;
                        for &(w, ww) in &wls {
                            let m = self.resident_moments[&(f, w, SamplingKey::of(sampling))];
                            let x = m[5] + m[1] * (m[2] - centre[0]).powi(2);
                            let y = m[7] + m[1] * (m[3] - centre[1]).powi(2);
                            sum += ww
                                * match component {
                                    Some(Axis::X) => x,
                                    Some(Axis::Y) => y,
                                    None => x + y,
                                };
                            weight += ww * m[1];
                        }
                        return if weight > 0. {
                            Ok(sum / weight)
                        } else {
                            Err(OperandError::RayFailure(
                                "no pupil ray reached image".into(),
                            ))
                        };
                    }
                    let (wt, xs, ys, symmetry) = self.spot_rays(f, &wls, sampling)?;
                    let c = match reference {
                        SpotReference::Centroid => spot_centroid(&wt, &xs, &ys, symmetry)
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
                self.colour_pair(*short, *long)?;
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
                    let axis = self.irradiance(ON_AXIS, w)?;
                    if !(axis > 0.0) {
                        return Err(OperandError::RayFailure(
                            "no on-axis ray reaches the image".into(),
                        ));
                    }
                    Ok(self.irradiance(f, w)? / axis)
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
        if let Some(fits) = self.resident_wavefits.get(&(f, w, SamplingKey::of(s))) {
            let v = fits[usize::from(reference == WavefrontReference::Centroid)][3];
            return if v.is_finite() {
                Ok(v)
            } else {
                Err(OperandError::RayFailure("no fitted OPD in pupil".into()))
            };
        }
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
        let hit = match self.real_rays.get(&(f, w, ppx.to_bits(), ppy.to_bits(), j)) {
            Some(h) if quantity != RayQuantity::IncidenceAngle => (h.clone()?, None),
            _ => {
                let mut rec = Vec::new();
                let opts = TraceOptions {
                    last_surface: Some(j),
                    ..self.opts
                };
                self.ray(f, w, ppx, ppy, &opts, Some(&mut rec))?;
                let r = rec
                    .get(j)
                    .ok_or_else(|| OperandError::Invalid("surface not reached".into()))?;
                (
                    RealHit {
                        point: r.point,
                        dir: r.dir,
                        opl: r.opl,
                    },
                    Some(r.cos_incidence),
                )
            }
        };
        let (r, cos_incidence) = hit;
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
            RayQuantity::IncidenceAngle => cos_incidence
                .ok_or_else(|| OperandError::Invalid("incidence angle not traced".into()))?
                .clamp(0.0, 1.0)
                .acos()
                .to_degrees(),
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

    /// Irradiance (arbitrary but field-independent units) at the image of
    /// field `f` for a uniform-radiance Lambertian object, from energy
    /// conservation: `E = G · |dμ/dh| / |dA'/dh|`.
    ///
    /// * `G`: étendue of the transmitted beam per unit object measure — for
    ///   an object at infinity the beam's cross-section perpendicular to its
    ///   direction (mm²), for a finite object the projected solid angle of
    ///   the transmitted cone about the object normal (the area of its
    ///   direction cosines `(L, M)`) — integrated over the pupil region whose
    ///   rays reach the image ([`Self::transmitted_etendue`]).
    /// * `dμ/dh`: object measure (solid angle at infinity, object area
    ///   otherwise) per unit normalised field area, and `dA'/dh` the image
    ///   area per unit normalised field area of the chief ray's image point —
    ///   both by central differences in `(hx, hy)` (step 1e-4).
    ///
    /// This holds for any (non-aplanatic, distorting, vignetting) lens; the
    /// ratio to the on-axis value is the relative illumination.
    fn irradiance(&self, f: FieldKey, w: usize) -> Result<f64, OperandError> {
        let field = self.field(f)?;
        if !field.is_valid() {
            return Err(OperandError::RayFailure(
                field.error.clone().unwrap_or_default(),
            ));
        }
        const DH: f64 = 1e-4;
        let p = field.point;
        let shifted = |dx: f64, dy: f64| -> Result<(ObjectSource, [f64; 2]), OperandError> {
            let pf = self
                .tr
                .prepare_field(mklight_trace::FieldPoint {
                    hx: p.hx + dx,
                    hy: p.hy + dy,
                    vignetting: p.vignetting,
                })
                .map_err(|e| OperandError::RayFailure(e.to_string()))?;
            let c = self.tr.chief(&pf, w);
            if !c.status.is_ok() {
                return Err(status_error(&c));
            }
            Ok((pf.source, [c.ray.pos.x, c.ray.pos.y]))
        };
        let (sxp, ixp) = shifted(DH, 0.0)?;
        let (sxm, ixm) = shifted(-DH, 0.0)?;
        let (syp, iyp) = shifted(0.0, DH)?;
        let (sym, iym) = shifted(0.0, -DH)?;
        let d = |a: [f64; 2], b: [f64; 2]| [(a[0] - b[0]) / (2.0 * DH), (a[1] - b[1]) / (2.0 * DH)];
        let (ix, iy) = (d(ixp, ixm), d(iyp, iym));
        let image = (ix[0] * iy[1] - ix[1] * iy[0]).abs();
        let vec = |s: &ObjectSource| match *s {
            ObjectSource::Direction { dir } => dir,
            ObjectSource::Point { point } => point,
        };
        let dv = |a: &ObjectSource, b: &ObjectSource| (vec(a) - vec(b)) * (0.5 / DH);
        let (ox, oy) = (dv(&sxp, &sxm), dv(&syp, &sym));
        let object = match field.source {
            // Solid angle swept by the unit direction.
            ObjectSource::Direction { dir } => dir.dot(ox.cross(oy)).abs(),
            // Object area (projected on the object plane).
            ObjectSource::Point { .. } => ox.cross(oy).z.abs(),
        };
        if !(image > 0.0 && image.is_finite() && object.is_finite()) {
            return Err(OperandError::Invalid(
                "field maps to no image area (afocal image space?)".into(),
            ));
        }
        Ok(self.transmitted_etendue(f, field, w)? * object / image)
    }

    /// Étendue `G` of the rays of `field` that reach the image (see
    /// [`Self::irradiance`]): `∬ |∂Q/∂(px, py)| dpx dpy` over the transmitted
    /// part of the unit pupil, where `Q(px, py)` is the generated ray's
    /// position across the beam (object at infinity) or its direction cosines
    /// `(L, M)` (finite object).
    ///
    /// Polar integration. Along an azimuth the transmitted radial intervals
    /// are found from `illumination_rings + 1` equally spaced samples, each
    /// pass/block transition refined by bisection to 1e-7 of the pupil
    /// radius, and `|∂Q/∂p| ρ` (central differences of `Q`) is integrated
    /// over each interval by 6-point Gauss–Legendre. In azimuth,
    /// `illumination_arms` equally spaced azimuths are scanned first. If the
    /// surfaces bounding the intervals are the same on all of them, the
    /// integrand is smooth and periodic and their trapezoidal sum is used
    /// (spectrally accurate). Otherwise the azimuths where the bounding
    /// surfaces change — corners of a vignetted pupil, tangents of a pupil
    /// region that excludes the centre — are located by bisection, and each
    /// smooth arc between them is integrated by Gauss–Legendre in a
    /// cosine-mapped variable (which also absorbs the square-root behaviour at
    /// a tangent). Vignetted boundaries are therefore located, not
    /// approximated by grid cells; what can still be missed are transmitted
    /// (or blocked) features narrower than the radial sample spacing, and
    /// corners formed by one surface's own non-circular aperture.
    fn transmitted_etendue(
        &self,
        f: FieldKey,
        field: &PreparedField,
        w: usize,
    ) -> Result<f64, OperandError> {
        let nr = self.ev.settings.illumination_rings.max(2);
        let na = self.ev.settings.illumination_arms.max(8);
        let (e1, e2) = match field.source {
            ObjectSource::Direction { dir } => {
                let a = if dir.x.abs() < 0.9 { Vec3::X } else { Vec3::Y };
                let e1 = (a - dir * a.dot(dir)).normalized();
                (e1, dir.cross(e1))
            }
            ObjectSource::Point { .. } => (Vec3::X, Vec3::Y),
        };
        let at_inf = matches!(field.source, ObjectSource::Direction { .. });
        let q = |px: f64, py: f64| -> Result<[f64; 2], OperandError> {
            if self.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                return Err(OperandError::Invalid("RI completion cancelled".into()));
            }
            let g = self
                .tr
                .generate(field, w, px, py)
                .map_err(|_| OperandError::RayFailure("pupil ray cannot be aimed".into()))?;
            let v = if at_inf { g.ray.pos } else { g.ray.dir };
            Ok([v.dot(e1), v.dot(e2)])
        };
        const DP: f64 = 1e-5;
        let jac = |px: f64, py: f64| -> Result<f64, OperandError> {
            let (a, b) = (q(px + DP, py)?, q(px - DP, py)?);
            let (c, d) = (q(px, py + DP)?, q(px, py - DP)?);
            let qx = [a[0] - b[0], a[1] - b[1]];
            let qy = [c[0] - d[0], c[1] - d[1]];
            Ok((qx[0] * qy[1] - qx[1] * qy[0]).abs() / (4.0 * DP * DP))
        };
        let (gx, gw) = super::quadrature::gauss_legendre(6);
        // `None` when the ray reaches the image, else the surface stopping it.
        let blocked = |px: f64, py: f64| {
            if self.cancel.as_ref().is_some_and(|c| c.is_cancelled()) {
                return None;
            }
            if let Some(&by) = self
                .resident_visibility
                .get(&(f, w, px.to_bits(), py.to_bits()))
            {
                self.visibility_lookups[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                self.visibility_lookups[1].fetch_add(
                    u64::from(by.is_some()),
                    std::sync::atomic::Ordering::Relaxed,
                );
                by
            } else {
                self.tr.trace(field, w, px, py, &self.opts).status.surface()
            }
        };
        let centre = blocked(0.0, 0.0);
        // Transmitted radial intervals along azimuth θ, and the surfaces
        // bounding them (a change of these is a kink of the integrand in θ).
        let scan = |theta: f64| -> Arm {
            let (sn, cs) = theta.sin_cos();
            let at = |r: f64| blocked(r * cs, r * sn);
            // Boundary between a passing radius and a blocked one; returns
            // the radius and the blocking surface.
            let edge = |mut good: f64, mut bad: f64, mut by: Option<usize>| {
                while (bad - good).abs() > 1e-7 {
                    let m = 0.5 * (good + bad);
                    match at(m) {
                        None => good = m,
                        b => {
                            bad = m;
                            by = b;
                        }
                    }
                }
                (0.5 * (good + bad), by)
            };
            let mut arm = Arm::default();
            let mut open = centre.is_none().then_some(0.0);
            let mut prev = (0.0, centre);
            for k in 1..=nr {
                let r = k as f64 / nr as f64;
                let b = at(r);
                match (prev.1, b) {
                    (Some(by), None) => {
                        let (e, by) = edge(r, prev.0, Some(by));
                        open = Some(e);
                        arm.bounds.push(by);
                    }
                    (None, Some(by)) => {
                        let (e, by) = edge(prev.0, r, Some(by));
                        if let Some(a) = open.take() {
                            arm.intervals.push((a, e));
                        }
                        arm.bounds.push(by);
                    }
                    _ => {}
                }
                prev = (r, b);
            }
            if let Some(a) = open {
                arm.intervals.push((a, 1.0));
            }
            arm
        };
        // ∫ |∂Q/∂p| ρ dρ over the transmitted intervals of azimuth θ.
        let radial = |theta: f64, arm: &Arm| -> Result<f64, OperandError> {
            let (sn, cs) = theta.sin_cos();
            let mut sum = mklight_core::math::KahanSum::default();
            for &(a, b) in &arm.intervals {
                let (h, m) = (0.5 * (b - a), 0.5 * (b + a));
                for (t, wt) in gx.iter().zip(&gw) {
                    let r = m + h * t;
                    sum.add(h * wt * r * jac(r * cs, r * sn)?);
                }
            }
            Ok(sum.value())
        };
        let dt = TAU / na as f64;
        let coarse: Vec<Arm> = (0..na).map(|j| scan(dt * j as f64)).collect();
        // Kinks: azimuths where the bounding surfaces change (corners of a
        // vignetted pupil, tangents of a pupil not containing its centre),
        // located by bisection to 1e-6 rad.
        let mut kinks = Vec::new();
        for j in 0..na {
            let mut stack = vec![(
                dt * j as f64,
                coarse[j].bounds.clone(),
                dt * (j + 1) as f64,
                coarse[(j + 1) % na].bounds.clone(),
            )];
            while let Some((a, sa, b, sb)) = stack.pop() {
                if sa == sb {
                    continue;
                }
                let m = 0.5 * (a + b);
                if b - a < 1e-6 {
                    kinks.push(m);
                    continue;
                }
                let sm = scan(m).bounds;
                stack.push((m, sm.clone(), b, sb));
                stack.push((a, sa, m, sm));
            }
        }
        kinks.sort_by(f64::total_cmp);
        let mut total = mklight_core::math::KahanSum::default();
        if kinks.is_empty() {
            // Smooth and periodic in θ: the trapezoidal rule converges
            // spectrally.
            for (j, arm) in coarse.iter().enumerate() {
                total.add(dt * radial(dt * j as f64, arm)?);
            }
            return Ok(total.value());
        }
        // Piecewise smooth: Gauss–Legendre on each arc between kinks, in
        // `u` with θ = a + L(1 − cos πu)/2, which also smooths the
        // square-root behaviour at a tangent.
        for (i, &a) in kinks.iter().enumerate() {
            let b = kinks.get(i + 1).copied().unwrap_or(kinks[0] + TAU);
            let len = b - a;
            let n = (len / dt).ceil() as usize + 4;
            let (ux, uw) = super::quadrature::gauss_legendre(n);
            for (x, wt) in ux.iter().zip(&uw) {
                let u = 0.5 * (x + 1.0);
                let theta = a + 0.5 * len * (1.0 - (PI * u).cos());
                let jw = 0.25 * wt * len * PI * (PI * u).sin();
                total.add(jw * radial(theta, &scan(theta))?);
            }
        }
        Ok(total.value())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::merit::default::{DefaultCriterion, DefaultMeritSpec};
    use mklight_core::model::Placement;
    use mklight_core::samples;

    /// The memoised group path returns exactly what `value` returns, in the
    /// merit's order and in any other, on symmetric and asymmetric
    /// (mirrored half pupil) candidates, for every reference.
    #[test]
    fn memoised_group_values_equal_value() {
        let ev = TraceEvaluator::builtin();
        let base = samples::cooke_triplet();
        let mut decentered = base.clone();
        decentered.surfaces[3].placement = Some(Placement {
            decenter_x: 0.05,
            ..Default::default()
        });
        let mut ops = Vec::new();
        for criterion in [
            DefaultCriterion::default(),
            DefaultCriterion::RmsSpot {
                reference: SpotReference::Chief,
                scope: CentroidScope::Polychromatic,
            },
            DefaultCriterion::Wavefront {
                reference: WavefrontReference::Centroid,
            },
            DefaultCriterion::Wavefront {
                reference: WavefrontReference::Chief,
            },
        ] {
            let spec = DefaultMeritSpec {
                criterion,
                thickness: None,
                ..Default::default()
            };
            ops.extend(spec.generate(&base).unwrap().operands);
        }
        // A pupil point outside the group: the general path.
        let mut off = ops[0].clone();
        if let OperandKind::RayAberration { px, .. } = &mut off.kind {
            *px = 0.123;
        }
        ops.push(off);
        let refs: Vec<&Operand> = ops.iter().collect();
        let ctx = JobContext::new();
        let mut mirrored = 0;
        for sys in [&base, &decentered] {
            let tr = ev.tracer(sys).unwrap();
            let c = Candidate::build(&ev, &tr, &refs, &ctx, false).unwrap();
            let slow: Vec<OperandValue> = refs.iter().map(|op| c.value(op)).collect();
            let mut memo = GroupMemo::default();
            let fast: Vec<OperandValue> =
                refs.iter().map(|op| c.value_with(op, &mut memo)).collect();
            let mut memo = GroupMemo::default();
            let mut rev: Vec<OperandValue> = refs
                .iter()
                .rev()
                .map(|op| c.value_with(op, &mut memo))
                .collect();
            rev.reverse();
            for ((s, f), r) in slow.iter().zip(&fast).zip(&rev) {
                let bits =
                    |v: &OperandValue| v.as_ref().map(|x| x.to_bits()).map_err(|e| e.clone());
                assert_eq!(bits(s), bits(f));
                assert_eq!(bits(s), bits(r));
                assert!(s.is_ok(), "{s:?}");
            }
            mirrored += c
                .groups
                .values()
                .filter(|g| g.symmetry == PupilSymmetry::None)
                .count();
        }
        assert!(
            mirrored > 0,
            "the decentered candidate traces mirrored half pupils"
        );
    }
}
