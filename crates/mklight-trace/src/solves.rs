//! Solves and pickups (`docs/SOLVES.md`): compute every solve-controlled
//! parameter of an [`OpticalSystem`] and return a **new** system with those
//! values set. The input is never modified; the engine builds the snapshot
//! of the solved system.
//!
//! # Method
//!
//! Each solve has a *condition* `measured(x) = goal` on its target `x`
//! (module [`mklight_core::model::solve`] and `docs/SOLVES.md` define them).
//! Paraxial conditions use the primary wavelength's [`FirstOrder`] rays.
//!
//! 1. **Closed forms.** With the ray scaling frozen, every condition is
//!    linear in a natural variable of its target, so one update solves it:
//!    height `y_{j+1} = y_j + t_j u_j` ⇒ `t_j = (h − y_j)/u_j`; slope after a
//!    refraction `ω_k = ω_{k−1} − y_k φ_k` ⇒ `φ_k = (ω_{k−1} − N_k u)/y_k` with
//!    `φ = c (N_k − N_{k−1})` (or `|N_k|/f` for an ideal lens); F/# through
//!    the system power `ω'_A` of the unit axial ray, which is affine in `φ_k`
//!    with slope `−y_{A,k} ∂ω'_A/∂ω_k` (a tangent y–nu trace); position,
//!    centre-of-curvature and edge-thickness solves are sums of thicknesses
//!    and sags. The update is repeated as a fixed-point iteration when the
//!    ray scaling itself depends on the target (an F/#-sized aperture, a
//!    stop-size aperture behind the target, real-height fields, automatic
//!    semi-diameters).
//! 2. **Root finding.** When there is no closed form, or the fixed point
//!    stalls or diverges, the residual is solved by bracketed Brent iteration
//!    ([`brent`]) in a variable in which it is smooth (`t`, `c`, or `1/f` for
//!    ideal lenses), starting from an expanding bracket around the current
//!    value. A root that cannot be bracketed, or a bracket that collapses on
//!    a discontinuity, is reported as an error naming the solve.
//! 3. **Coupling.** Solves run in [`OpticalSystem::solve_order`] (explicit
//!    dependencies first, then upstream targets first). Whole sweeps are
//!    repeated until no solved parameter moves by more than the tolerance
//!    (a fixed point of the coupled set); the report says how many sweeps
//!    were needed.
//!
//! A condition holds when `|measured − goal| ≤ tolerance · scale`. The scale
//! of a ray condition is the ray's largest height or slope, **capped** by the
//! input system's size ([`length_scale`], [`RAY_CAP`]): a ray whose scaling
//! blows up (e.g. a chief ray rescaled to an image height that its unit ray
//! almost cannot reach) must not loosen its own tolerance. Every condition is
//! re-checked on the returned system.
//!
//! Everything is deterministic: fixed order, no randomness, no threads.
//! Cancellation is checked before every evaluation of a candidate system.
//! Multi-configuration systems are solved per configuration *after* its
//! overrides ([`apply_solves_configuration`]); an override of a
//! solve-controlled parameter is overwritten and reported as a warning, and
//! the returned system has no configurations left to re-apply.

use std::cell::OnceCell;
use std::sync::Arc;

use mklight_core::job::{Precision, RunTimer};
use mklight_core::materials::MaterialLibrary;
use mklight_core::model::Solve;
use mklight_core::param::ParamPath;
use mklight_core::{
    ApertureSpec, FieldKind, JobContext, MkError, MkResult, OpticalSystem, ResolvedSystem, RunInfo,
    SemiDiameter, Snapshot, SolveKind,
};
use serde::{Deserialize, Serialize};

use crate::numeric::Compensated;
use crate::paraxial::{FirstOrder, ParaxialModel, ParaxialRayTable, first_order};
use crate::semidia::SemiDiameterSettings;
use crate::tracer::SequentialTracer;

/// Solver settings (serialised into the report's `RunInfo`).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveSettings {
    /// Relative tolerance: on each condition (against its natural scale, e.g.
    /// the largest marginal-ray height) and on the change of every solved
    /// parameter between two sweeps.
    pub tolerance: f64,
    /// Upper bound on sweeps over all solves.
    pub max_sweeps: usize,
    /// Upper bound on evaluations per solve and sweep (fixed point plus
    /// root finding).
    pub max_iterations: usize,
    /// Use closed-form updates. `false` finds every solve by bracketed root
    /// finding alone (a cross-check of the closed forms).
    pub closed_form: bool,
}

impl Default for SolveSettings {
    fn default() -> Self {
        SolveSettings {
            tolerance: 1e-12,
            max_sweeps: 50,
            max_iterations: 200,
            closed_form: true,
        }
    }
}

/// How a solve's value was found (the strongest method any sweep needed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolveMethod {
    /// The condition already held.
    Unchanged,
    /// One closed-form update.
    ClosedForm,
    /// Repeated closed-form updates (the ray scaling depends on the target).
    FixedPoint,
    /// Bracketed Brent iteration.
    RootFinding,
}

/// Result of one solve, in declaration order.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveOutcome {
    /// Index into `OpticalSystem::solves`.
    pub index: usize,
    pub kind: String,
    pub target: ParamPath,
    /// Value in the input system (`None` when it had none, e.g. an automatic
    /// semi-diameter).
    pub previous: Option<f64>,
    pub value: f64,
    /// `measured − goal` of the condition in the solved system, in the
    /// condition's units (mm, slope, F/#, or dimensionless `c·D − 1`).
    pub residual: f64,
    pub method: SolveMethod,
    /// System evaluations spent on this solve over all sweeps.
    pub evaluations: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SolveReport {
    pub outcomes: Vec<SolveOutcome>,
    /// Evaluation order (indices into `OpticalSystem::solves`).
    pub order: Vec<usize>,
    /// Sweeps until the coupled set reached its fixed point (the last one
    /// only confirms it).
    pub sweeps: usize,
    /// Largest relative parameter change in the last sweep.
    pub max_change: f64,
    /// Parameters controlled by solves (exclude them from optimisation).
    pub solved_params: Vec<ParamPath>,
    pub warnings: Vec<String>,
    /// The configuration that was solved ([`apply_solves_configuration`]);
    /// `None` for the system as given ([`apply_solves_with`]).
    #[serde(default)]
    pub configuration: Option<usize>,
    pub run_info: RunInfo,
}

/// Settings as recorded in `RunInfo` (with the configuration, if any).
#[derive(Serialize)]
struct Recorded<'a> {
    #[serde(flatten)]
    settings: &'a SolveSettings,
    #[serde(skip_serializing_if = "Option::is_none")]
    configuration: Option<usize>,
}

/// Ray heights beyond `RAY_CAP ×` the system's [`length_scale`], and ray
/// slopes beyond `RAY_CAP`, no longer widen a ray condition's tolerance.
pub const RAY_CAP: f64 = 1e3;

/// The final check of every condition on the returned system allows this
/// factor on the tolerance: a later solve of the last sweep may move an
/// earlier one's parameters by up to `tolerance` (relative).
const FINAL_SLACK: f64 = 10.0;

/// Physical length scale of a system (mm), independent of any paraxial ray:
/// the largest of 1 mm, the sum of the finite |thicknesses| (including a
/// finite object distance), an entrance-pupil diameter aperture, the largest
/// height-type field, and the largest fixed, mechanical or aperture radius.
pub fn length_scale(sys: &OpticalSystem) -> f64 {
    let track = sum(sys
        .surfaces
        .iter()
        .map(|s| s.thickness.abs())
        .filter(|t| t.is_finite()));
    let aperture = match sys.aperture {
        ApertureSpec::EntrancePupilDiameter { value } => value,
        _ => 0.0,
    };
    let field = match sys.fields.kind {
        FieldKind::ObjectAngle => 0.0,
        _ => sys.fields.max_radial(),
    };
    let radius = sys.surfaces.iter().fold(0.0f64, |r, s| {
        let fixed = match s.semi_diameter {
            SemiDiameter::Fixed { value } => value,
            SemiDiameter::Auto => 0.0,
        };
        let mech = s.edge.mechanical_semi_diameter.unwrap_or(0.0);
        let ap = s.aperture.as_ref().map_or(0.0, |a| a.bounding_radius());
        r.max(fixed).max(mech).max(ap)
    });
    [track, aperture, field, radius]
        .into_iter()
        .filter(|v| v.is_finite())
        .fold(1.0, f64::max)
}

/// Parameters controlled by solves, sorted (= `OpticalSystem::solved_params`).
pub fn solved_params(sys: &OpticalSystem) -> Vec<ParamPath> {
    sys.solved_params()
}

/// Solve `sys` with default settings (module docs).
pub fn apply_solves(
    sys: &OpticalSystem,
    lib: &MaterialLibrary,
) -> MkResult<(OpticalSystem, SolveReport)> {
    apply_solves_with(sys, lib, &SolveSettings::default(), &JobContext::new())
}

/// Solve configuration `k` (its overrides first, then the solves). The
/// returned system *is* the solved configuration: its base values are those
/// of configuration `k` and its `configurations` list is empty, because
/// re-applying any configuration's overrides would overwrite solved values
/// (or change what they depend on). `report.configuration` is `Some(k)`.
pub fn apply_solves_configuration(
    sys: &OpticalSystem,
    lib: &MaterialLibrary,
    k: usize,
    settings: &SolveSettings,
    ctx: &JobContext,
) -> MkResult<(OpticalSystem, SolveReport)> {
    let mut cfg = sys.with_configuration(k)?;
    cfg.configurations.clear();
    let mut warnings = Vec::new();
    if let Some(c) = sys.configurations.get(k) {
        let solved = sys.solved_params();
        for (p, _) in &c.overrides {
            if solved.contains(p) {
                warnings.push(format!(
                    "configuration {k} overrides {p:?}, which a solve controls: the solve wins"
                ));
            }
        }
    }
    let (out, mut report) = solve_system(&cfg, lib, settings, ctx, Some(k))?;
    warnings.append(&mut report.warnings);
    report.run_info.warnings = warnings.clone();
    report.warnings = warnings;
    Ok((out, report))
}

/// Solve every configuration (just the base system when there are none).
pub fn apply_solves_all(
    sys: &OpticalSystem,
    lib: &MaterialLibrary,
    settings: &SolveSettings,
    ctx: &JobContext,
) -> MkResult<Vec<(OpticalSystem, SolveReport)>> {
    let n = sys.configurations.len().max(1);
    (0..n)
        .map(|k| {
            let sub = ctx.subrange(k as f64 / n as f64, (k + 1) as f64 / n as f64);
            apply_solves_configuration(sys, lib, k, settings, &sub)
        })
        .collect()
}

/// Solve `sys` (module docs). Cancellable before every evaluation; progress
/// is monotone and reaches 1 on success.
///
/// The returned system keeps `sys.configurations` unchanged (so it can be
/// stored back losslessly), but it is solved for the base values only:
/// `with_configuration(k)` on it is **not** a solved system. Solve a
/// configuration with [`apply_solves_configuration`].
pub fn apply_solves_with(
    sys: &OpticalSystem,
    lib: &MaterialLibrary,
    settings: &SolveSettings,
    ctx: &JobContext,
) -> MkResult<(OpticalSystem, SolveReport)> {
    solve_system(sys, lib, settings, ctx, None)
}

fn solve_system(
    sys: &OpticalSystem,
    lib: &MaterialLibrary,
    settings: &SolveSettings,
    ctx: &JobContext,
    configuration: Option<usize>,
) -> MkResult<(OpticalSystem, SolveReport)> {
    let timer = RunTimer::start();
    let input = Snapshot::new(sys.clone())?;
    if !(settings.tolerance > 0.0 && settings.tolerance < 1.0) || settings.max_sweeps == 0 {
        return Err(MkError::InvalidModel(
            "solve settings: tolerance must be in (0, 1) and max_sweeps ≥ 1".into(),
        ));
    }
    let env = Env {
        lib: Arc::new(lib.clone()),
        length: length_scale(sys),
        job: ctx,
    };
    let order = sys.solve_order()?;
    let mut warnings = Vec::new();
    if sys.solves.iter().any(|s| s.kind.is_paraxial())
        && sys.surfaces.iter().any(|s| s.is_displaced())
    {
        warnings.push(
            "system has tilts or decenters: paraxial solves use the unfolded, rotationally \
             symmetric equivalent"
                .into(),
        );
    }
    for (k, s) in sys.solves.iter().enumerate() {
        let range = match s.kind {
            SolveKind::Position { from, to, .. } => {
                sys.surface_index(from).zip(sys.surface_index(to))
            }
            SolveKind::CenterOfCurvature { surface } => s
                .target
                .surface()
                .and_then(|id| sys.surface_index(id))
                .zip(sys.surface_index(surface))
                .map(|(a, b)| (a.min(b), a.max(b))),
            _ => None,
        };
        if let Some((lo, hi)) = range
            && placed_between(sys, lo, hi)
        {
            warnings.push(format!(
                "solve {k} ({} on {:?}): tilts or decenters lie between surfaces {lo} and {hi}; \
                 the axial distance is that of the unfolded, rotationally symmetric equivalent",
                s.kind.name(),
                s.target
            ));
        }
    }

    let deps: Vec<Vec<usize>> = (0..sys.solves.len())
        .map(|k| dependents(sys, &order, k))
        .collect();
    let mut cur = sys.clone();
    let mut stats: Vec<(SolveMethod, usize)> = vec![(SolveMethod::Unchanged, 0); sys.solves.len()];
    let mut sweeps = 0;
    let mut max_change = f64::INFINITY;
    while sweeps < settings.max_sweeps {
        sweeps += 1;
        max_change = 0.0;
        for (done, &k) in order.iter().enumerate() {
            ctx.check()?;
            let solve = &sys.solves[k];
            let before = solve.target.get(&cur).ok();
            let run = run_solve(k, &sys.solves, &deps[k], &mut cur, &env, settings);
            // A cancelled sizing trace surfaces as a failed evaluation.
            ctx.check()?;
            let (method, evals) = run?;
            let after = solve.target.get(&cur)?;
            stats[k].0 = stats[k].0.max(method);
            stats[k].1 += evals;
            let change = match before {
                Some(b) => {
                    let floor = change_floor(&solve.target);
                    (after - b).abs() / b.abs().max(after.abs()).max(floor)
                }
                None => f64::INFINITY,
            };
            max_change = max_change.max(change);
            ctx.report(sweep_progress(sweeps, done + 1, order.len()), "solves");
        }
        if max_change <= settings.tolerance {
            break;
        }
    }
    if max_change > settings.tolerance {
        return Err(MkError::Numerical(format!(
            "coupled solves did not converge in {sweeps} sweeps (last relative change {max_change:.3e})"
        )));
    }
    cur.validate()?;

    // Final residuals, all measured (and checked) on the solved system.
    let eval = Eval::new(&cur, &env);
    let mut outcomes = Vec::with_capacity(sys.solves.len());
    for (k, s) in sys.solves.iter().enumerate() {
        let m = measure(s, &eval);
        ctx.check()?;
        let m = m.map_err(|e| solve_error(k, s, &e))?;
        if !m.converged(FINAL_SLACK * settings.tolerance) {
            return Err(solve_error(
                k,
                s,
                &format!(
                    "the condition does not hold in the solved system (residual {:.3e}, \
                     tolerance {:.3e}){}",
                    m.residual(),
                    FINAL_SLACK * settings.tolerance * m.scale,
                    m.degenerate_note(env.length)
                ),
            ));
        }
        outcomes.push(SolveOutcome {
            index: k,
            kind: s.kind.name().into(),
            target: s.target.clone(),
            previous: s.target.get(sys).ok(),
            value: s.target.get(&cur)?,
            residual: m.residual(),
            method: stats[k].0,
            evaluations: stats[k].1,
        });
    }
    ctx.report(1.0, "solves converged");
    let run_info = timer.finish(
        &input,
        "solves",
        &Recorded {
            settings,
            configuration,
        },
        "cpu-f64",
        Precision::F64Reference,
        None,
        warnings.clone(),
    );
    Ok((
        cur,
        SolveReport {
            outcomes,
            order,
            sweeps,
            max_change,
            solved_params: sys.solved_params(),
            warnings,
            configuration,
            run_info,
        },
    ))
}

fn solve_error(k: usize, s: &Solve, why: &str) -> MkError {
    MkError::Numerical(format!(
        "solve {k} ({} on {:?}): {why}",
        s.kind.name(),
        s.target
    ))
}

/// Progress after `done` of `n` solves of sweep `sweep` (1-based): sweep `s`
/// covers `[1 − 1/s, 1 − 1/(s+1)]`, so the sequence is strictly increasing
/// and stays below 1 however many sweeps are needed.
fn sweep_progress(sweep: usize, done: usize, n: usize) -> f64 {
    let (lo, hi) = (1.0 - 1.0 / sweep as f64, 1.0 - 1.0 / (sweep + 1) as f64);
    lo + (hi - lo) * done as f64 / n.max(1) as f64
}

/// Do tilts or decenters change the geometry between surfaces `lo` and `hi`?
/// A placement applies before its surface, so the one on `lo` moves both
/// ends rigidly and does not count.
fn placed_between(sys: &OpticalSystem, lo: usize, hi: usize) -> bool {
    sys.surfaces.iter().enumerate().any(|(i, s)| {
        (lo < i && i <= hi && s.placement.is_some_and(|p| !p.is_identity()))
            || s.element_placement.as_ref().is_some_and(|e| {
                let last = sys.surface_index(e.last).unwrap_or(i);
                !e.motion.is_identity() && (lo < i && i <= hi || lo <= last && last < hi)
            })
    })
}

/// Scale below which a parameter change is measured absolutely
/// (relative above it).
fn change_floor(p: &ParamPath) -> f64 {
    match p {
        ParamPath::Curvature { .. } => 1e-6,
        ParamPath::Asphere { .. } => f64::MIN_POSITIVE,
        ParamPath::TiltX { .. } | ParamPath::TiltY { .. } | ParamPath::TiltZ { .. } => 1e-6,
        ParamPath::FieldWeight { .. } | ParamPath::WavelengthWeight { .. } => 1e-6,
        _ => 1.0,
    }
}

// ---------------------------------------------------------------------------
// Evaluation of one candidate system.

struct Para {
    resolved: Arc<ResolvedSystem>,
    model: ParaxialModel,
    fo: FirstOrder,
}

/// Fixed for one solve of a whole system.
struct Env<'a> {
    lib: Arc<MaterialLibrary>,
    /// [`length_scale`] of the input system (mm).
    length: f64,
    job: &'a JobContext,
}

/// Lazily evaluated paraxial data and semi-diameters of one system.
struct Eval<'a> {
    sys: &'a OpticalSystem,
    env: &'a Env<'a>,
    para: OnceCell<Result<Para, String>>,
    sd: OnceCell<Result<Vec<f64>, String>>,
}

impl<'a> Eval<'a> {
    fn new(sys: &'a OpticalSystem, env: &'a Env<'a>) -> Self {
        Eval {
            sys,
            env,
            para: OnceCell::new(),
            sd: OnceCell::new(),
        }
    }
    fn para(&self) -> Result<&Para, String> {
        self.para
            .get_or_init(|| {
                let snap = Snapshot::new(self.sys.clone()).map_err(|e| e.to_string())?;
                let resolved =
                    ResolvedSystem::new(&snap, self.env.lib.clone()).map_err(|e| e.to_string())?;
                let wl = resolved.primary_wavelength;
                let fo = first_order(&resolved, wl).map_err(|e| e.to_string())?;
                let model = ParaxialModel::new(&resolved, wl);
                Ok(Para {
                    resolved: Arc::new(resolved),
                    model,
                    fo,
                })
            })
            .as_ref()
            .map_err(|e| e.clone())
    }
    /// Automatic (traced) clear semi-diameters of every surface.
    fn semi_diameters(&self) -> Result<&[f64], String> {
        self.sd
            .get_or_init(|| {
                let p = self.para()?;
                let tracer =
                    SequentialTracer::new(p.resolved.clone()).map_err(|e| e.to_string())?;
                // Cancellation only: the solve owns the progress bar.
                let cancel_only = JobContext::with(self.env.job.cancel.clone(), Default::default());
                tracer
                    .semi_diameters_with(&SemiDiameterSettings::default(), &cancel_only)
                    .map(|sd| sd.values)
                    .map_err(|e| e.to_string())
            })
            .as_ref()
            .map(|v| v.as_slice())
            .map_err(|e| e.clone())
    }
}

/// A condition evaluated on one system: `measured = goal` within
/// `tolerance · scale`, and the closed-form value of the target (or why
/// there is none).
struct Measure {
    measured: f64,
    goal: f64,
    /// Convergence scale (capped, see the module docs).
    scale: f64,
    /// The ray's own (uncapped) height or slope scale; 0 for conditions
    /// without a ray.
    ray_scale: f64,
    update: Result<f64, String>,
}

impl Measure {
    fn residual(&self) -> f64 {
        self.measured - self.goal
    }
    fn converged(&self, tol: f64) -> bool {
        let r = self.residual();
        r.is_finite() && r.abs() <= tol * self.scale
    }
    /// Explanation appended to a failure when the ray scaling is degenerate.
    fn degenerate_note(&self, length: f64) -> String {
        if self.ray_scale > self.scale {
            format!(
                "; the paraxial ray is scaled to {:.3e}, far beyond the system's size \
                 ({length:.3e} mm): the condition is met, if at all, only by a degenerate ray \
                 scaling",
                self.ray_scale
            )
        } else {
            String::new()
        }
    }
}

/// A paraxial ray table must be finite to be measured.
fn finite_ray(ray: &ParaxialRayTable) -> Result<(), String> {
    if ray.y.iter().chain(&ray.u).all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err("the paraxial ray is not finite".into())
    }
}

fn idx(sys: &OpticalSystem, p: &ParamPath) -> Result<usize, String> {
    p.surface()
        .and_then(|id| sys.surface_index(id))
        .ok_or_else(|| format!("{p:?} does not name a surface"))
}

fn max_abs(v: &[f64]) -> f64 {
    v.iter().fold(0.0f64, |m, x| m.max(x.abs()))
}

/// Exact sum of `v` (compensated).
fn sum(v: impl IntoIterator<Item = f64>) -> f64 {
    let mut s = Compensated::ZERO;
    for x in v {
        s.add(x);
    }
    s.value()
}

/// Convert a target surface power `φ` into the target parameter.
fn power_to_param(
    target: &ParamPath,
    model: &ParaxialModel,
    k: usize,
    phi: f64,
) -> Result<f64, String> {
    let nn = &model.reduced_index;
    match target {
        ParamPath::Curvature { .. } => {
            let dn = nn[k] - nn[k - 1];
            if dn == 0.0 {
                return Err(format!(
                    "surface {k} separates equal indices: its curvature has no paraxial power"
                ));
            }
            Ok(phi / dn)
        }
        ParamPath::ParaxialFocalLength { .. } => {
            if phi == 0.0 {
                return Err("the solution is an ideal lens of zero power".into());
            }
            Ok(nn[k].abs() / phi)
        }
        _ => Err("the target is not a power-type parameter".into()),
    }
}

/// Reduced slope `ω` after the last surface (n−2) of the ray that has
/// height `y` at surface `k` and reduced slope `w` after it.
fn propagate_w(model: &ParaxialModel, k: usize, mut y: f64, mut w: f64) -> f64 {
    let n = model.power.len();
    for m in k + 1..n - 1 {
        y = (model.thickness[m - 1] / model.reduced_index[m - 1]).mul_add(w, y);
        w = (-y).mul_add(model.power[m], w);
    }
    w
}

fn measure(s: &Solve, ev: &Eval) -> Result<Measure, String> {
    let sys = ev.sys;
    let tgt = &s.target;
    match &s.kind {
        SolveKind::MarginalRayHeight { height } | SolveKind::ChiefRayHeight { height } => {
            let p = ev.para()?;
            let ray: &ParaxialRayTable = if matches!(s.kind, SolveKind::MarginalRayHeight { .. }) {
                &p.fo.marginal
            } else {
                &p.fo.chief
            };
            finite_ray(ray)?;
            let j = idx(sys, tgt)?;
            let u = ray.u[j];
            let ray_scale = max_abs(&ray.y[1..]);
            Ok(Measure {
                measured: ray.y[j + 1],
                goal: *height,
                scale: height.abs().max(ray_scale.min(RAY_CAP * ev.env.length)),
                ray_scale,
                update: if u != 0.0 {
                    Ok((height - ray.y[j]) / u)
                } else {
                    Err(format!(
                        "the ray is parallel to the axis after surface {j}, so this thickness \
                         cannot change its height"
                    ))
                },
            })
        }
        SolveKind::MarginalRayAngle { slope } | SolveKind::ChiefRayAngle { slope } => {
            let p = ev.para()?;
            let ray = if matches!(s.kind, SolveKind::MarginalRayAngle { .. }) {
                &p.fo.marginal
            } else {
                &p.fo.chief
            };
            finite_ray(ray)?;
            let k = idx(sys, tgt)?;
            let nn = &p.model.reduced_index;
            let y = ray.y[k];
            let update = if y != 0.0 {
                // ω_k = ω_{k−1} − y_k φ_k = N_k · slope.
                let phi = (-nn[k]).mul_add(*slope, ray.nu[k - 1]) / y;
                power_to_param(tgt, &p.model, k, phi)
            } else {
                Err(format!(
                    "the ray has zero height on surface {k}, so its power cannot bend it"
                ))
            };
            let ray_scale = max_abs(&ray.u);
            Ok(Measure {
                measured: ray.u[k],
                goal: *slope,
                scale: slope.abs().max(ray_scale.min(RAY_CAP)),
                ray_scale,
                update,
            })
        }
        SolveKind::FNumber { value } => {
            let p = ev.para()?;
            let k = idx(sys, tgt)?;
            let model = &p.model;
            let n = model.power.len();
            let last = n - 2;
            let epd = p.fo.entrance_pupil_diameter;
            let n_img = model.reduced_index[last].abs();
            let update = (|| {
                if !(epd.is_finite() && epd > 0.0) {
                    return Err("the entrance pupil is not finite".to_string());
                }
                // Unit axial ray A: ω'_A = −1/EFL is affine in φ_k.
                let ra = model.trace(1.0, 0.0);
                let w_a = ra.nu[last];
                let efl = p.fo.efl;
                let sign = if efl.is_finite() && efl < 0.0 {
                    -1.0
                } else {
                    1.0
                };
                // f'/EPD = F  ⇒  |EFL| = F·EPD/|N'|.
                let w_goal = -n_img / (sign * value * epd);
                let dw_dphi = -ra.y[k] * propagate_w(model, k, 0.0, 1.0);
                if dw_dphi == 0.0 {
                    return Err(format!(
                        "the power of surface {k} does not change the system power \
                         (the axial ray has zero height there)"
                    ));
                }
                let phi = model.power[k] + (w_goal - w_a) / dw_dphi;
                power_to_param(tgt, model, k, phi)
            })();
            Ok(Measure {
                measured: p.fo.image_f_number,
                goal: *value,
                scale: *value,
                ray_scale: 0.0,
                update,
            })
        }
        SolveKind::Pickup {
            source,
            scale,
            offset,
        } => {
            let src = source.get(sys).map_err(|e| e.to_string())?;
            let goal = scale.mul_add(src, *offset);
            Ok(Measure {
                measured: tgt.get(sys).unwrap_or(f64::NAN),
                goal,
                scale: goal.abs().max(change_floor(tgt)),
                ray_scale: 0.0,
                update: Ok(goal),
            })
        }
        SolveKind::Position { from, to, distance } => {
            let a = sys.surface_index(*from).ok_or("unknown `from` surface")?;
            let b = sys.surface_index(*to).ok_or("unknown `to` surface")?;
            let j = idx(sys, tgt)?;
            let t = |i: usize| sys.surfaces[i].thickness;
            let others = sum((a..b).filter(|&i| i != j).map(t));
            Ok(Measure {
                measured: sum((a..b).map(t)),
                goal: *distance,
                scale: distance
                    .abs()
                    .max(sum((a..b).map(|i| t(i).abs())).min(RAY_CAP * ev.env.length))
                    .max(1.0),
                ray_scale: 0.0,
                update: Ok(distance - others),
            })
        }
        SolveKind::CenterOfCurvature { surface } => {
            let i = idx(sys, tgt)?;
            let r = sys.surface_index(*surface).ok_or("unknown surface")?;
            let t = |m: usize| sys.surfaces[m].thickness;
            // Axial position of vertex r relative to vertex i.
            let d = if r > i {
                sum((i..r).map(t))
            } else {
                -sum((r..i).map(t))
            };
            let c = tgt.get(sys).map_err(|e| e.to_string())?;
            Ok(Measure {
                measured: c * d,
                goal: 1.0,
                scale: 1.0,
                ray_scale: 0.0,
                update: if d != 0.0 && d.is_finite() {
                    Ok(1.0 / d)
                } else {
                    Err(format!("the vertices of surfaces {i} and {r} coincide"))
                },
            })
        }
        SolveKind::EdgeThickness { value, radius } => {
            let j = idx(sys, tgt)?;
            let r = match radius {
                Some(r) => *r,
                None => edge_radius(ev, j)?,
            };
            let sag = |i: usize| {
                sys.surfaces[i].shape.sag(r * r).ok_or_else(|| {
                    format!("radius {r} mm lies outside the conic domain of surface {i}")
                })
            };
            let (s0, s1) = (sag(j)?, sag(j + 1)?);
            let t = sys.surfaces[j].thickness;
            Ok(Measure {
                measured: sum([t, s1, -s0]),
                goal: *value,
                scale: value
                    .abs()
                    .max(t.abs().min(RAY_CAP * ev.env.length))
                    .max(1.0),
                ray_scale: 0.0,
                update: Ok(sum([*value, -s1, s0])),
            })
        }
    }
}

/// Default edge radius of the element between surfaces `j` and `j+1`: the
/// larger of the two semi-diameters, each the mechanical semi-diameter if
/// given, else the explicit aperture's bounding radius, else a fixed
/// semi-diameter, else the automatic (traced) one.
fn edge_radius(ev: &Eval, j: usize) -> Result<f64, String> {
    let mut r = 0.0f64;
    for i in [j, j + 1] {
        let s = &ev.sys.surfaces[i];
        let v = match (
            s.edge.mechanical_semi_diameter,
            &s.aperture,
            s.semi_diameter,
        ) {
            (Some(m), _, _) => m,
            (None, Some(a), _) => a.bounding_radius(),
            (None, None, SemiDiameter::Fixed { value }) => value,
            (None, None, SemiDiameter::Auto) => ev.semi_diameters()?[i],
        };
        r = r.max(v);
    }
    if r > 0.0 && r.is_finite() {
        Ok(r)
    } else {
        Err(format!("surfaces {j} and {} have no semi-diameter", j + 1))
    }
}

// ---------------------------------------------------------------------------
// One solve.

/// The variable in which root finding runs: the parameter itself, or `1/f`
/// for an ideal lens (smooth through zero power).
#[derive(Clone, Copy)]
enum Var {
    Direct,
    Reciprocal,
}

impl Var {
    fn of(p: &ParamPath) -> Var {
        match p {
            ParamPath::ParaxialFocalLength { .. } => Var::Reciprocal,
            _ => Var::Direct,
        }
    }
    fn to_param(self, v: f64) -> f64 {
        match self {
            Var::Direct => v,
            Var::Reciprocal => 1.0 / v,
        }
    }
    fn of_param(self, x: f64) -> f64 {
        self.to_param(x)
    }
    fn initial_step(self, p: &ParamPath, v: f64) -> f64 {
        let floor = match (self, p) {
            (Var::Reciprocal, _) => 1e-6,
            (_, ParamPath::Curvature { .. }) => 1e-5,
            (_, ParamPath::Thickness { .. }) => 1e-2,
            _ => 1e-6,
        };
        (1e-2 * v.abs()).max(floor)
    }
}

fn with_param(sys: &OpticalSystem, p: &ParamPath, x: f64) -> Option<OpticalSystem> {
    let mut s = sys.clone();
    p.set(&mut s, x).ok()?;
    Some(s)
}

/// Re-apply the direct solves `deps` (pickups, position, centre-of-curvature
/// and edge-thickness solves that read a parameter just changed) in order.
fn apply_dependents(
    sys: &mut OpticalSystem,
    solves: &[Solve],
    deps: &[usize],
    env: &Env,
) -> Result<(), String> {
    for &j in deps {
        let x = measure(&solves[j], &Eval::new(sys, env))?.update?;
        solves[j].target.set(sys, x).map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Solves that read `solves[k]`'s target, directly or through each other,
/// in evaluation order. Only direct (non-paraxial) solves read parameters
/// explicitly, so each has a closed-form update.
fn dependents(sys: &OpticalSystem, order: &[usize], k: usize) -> Vec<usize> {
    let mut changed = vec![sys.solves[k].target.clone()];
    let mut deps = Vec::new();
    for &j in order {
        if j != k && changed.iter().any(|p| sys.solves[j].reads(sys, p)) {
            deps.push(j);
            changed.push(sys.solves[j].target.clone());
        }
    }
    deps
}

/// Solve `solves[k]` on `cur` in place, keeping its dependents `deps`
/// consistent with every trial value (so a solve and the pickups of its
/// target are solved as one 1-D problem). Returns the method and the number
/// of evaluations. Cancellation is checked before every evaluation.
fn run_solve(
    k: usize,
    solves: &[Solve],
    deps: &[usize],
    cur: &mut OpticalSystem,
    env: &Env,
    settings: &SolveSettings,
) -> MkResult<(SolveMethod, usize)> {
    let s = &solves[k];
    let tol = settings.tolerance;
    let job = env.job;
    // A failure caused by cancellation is reported as `Cancelled`.
    let err = |why: &str| match job.check() {
        Err(e) => e,
        Ok(()) => solve_error(k, s, why),
    };
    let max_fixed_point = settings.max_iterations.min(60);
    let mut evals = 0usize;
    let mut updates = 0usize;
    let mut why_not = String::from("closed forms disabled");
    apply_dependents(cur, solves, deps, env).map_err(|e| err(&e))?;

    if settings.closed_form {
        why_not = "fixed-point iteration did not converge".into();
        let mut last_res = f64::INFINITY;
        let mut slow = 0;
        // The system before the latest update, to fall back on when the
        // update produced a system that cannot be evaluated.
        let mut before: Option<OpticalSystem> = None;
        while evals < max_fixed_point {
            job.check()?;
            evals += 1;
            let m = match (measure(s, &Eval::new(cur, env)), before.take()) {
                (Ok(m), _) => m,
                (Err(e), None) => return Err(err(&e)),
                (Err(e), Some(prev)) => {
                    *cur = prev;
                    why_not = format!("a closed-form step gave an invalid system ({e})");
                    break;
                }
            };
            if m.converged(tol) {
                let method = match updates {
                    0 => SolveMethod::Unchanged,
                    1 => SolveMethod::ClosedForm,
                    _ => SolveMethod::FixedPoint,
                };
                return Ok((method, evals));
            }
            // A contracting fixed point shrinks the residual every step;
            // otherwise (coupling through dependents or the ray scaling)
            // hand over to root finding.
            let r = m.residual().abs();
            if r.is_nan() || r > 0.9 * last_res {
                slow += 1;
                if slow >= 3 {
                    why_not = format!(
                        "fixed-point iteration stalled at residual {r:.3e}{}",
                        m.degenerate_note(env.length)
                    );
                    break;
                }
            }
            last_res = last_res.min(r);
            let x = match m.update {
                Ok(x) if x.is_finite() => x,
                Ok(_) => {
                    why_not = "the closed form gives a non-finite value".into();
                    break;
                }
                Err(e) => {
                    why_not = e;
                    break;
                }
            };
            before = Some(cur.clone());
            s.target.set(cur, x).map_err(|e| err(&e.to_string()))?;
            apply_dependents(cur, solves, deps, env).map_err(|e| err(&e))?;
            updates += 1;
        }
    }

    // Bracketed root finding in the smooth variable.
    let var = Var::of(&s.target);
    let x0 = s.target.get(cur).map_err(|e| err(&e.to_string()))?;
    let v0 = var.of_param(x0);
    let base = cur.clone();
    let mut evals_rf = 0usize;
    let trial = |v: f64| -> Option<OpticalSystem> {
        let x = var.to_param(v);
        if !x.is_finite() {
            return None;
        }
        let mut sys = with_param(&base, &s.target, x)?;
        apply_dependents(&mut sys, solves, deps, env).ok()?;
        Some(sys)
    };
    // `None` for an invalid system or after cancellation (`err` then turns
    // the failure into `Cancelled`).
    let mut eval = |v: f64| -> Option<(f64, f64)> {
        if job.cancel.is_cancelled() {
            return None;
        }
        evals_rf += 1;
        let sys = trial(v)?;
        let m = measure(s, &Eval::new(&sys, env)).ok()?;
        let r = m.residual();
        r.is_finite().then_some((r, tol * m.scale))
    };
    let Some((f0, ftol0)) = eval(v0) else {
        return Err(err(&format!(
            "{why_not}; and the condition cannot be evaluated at the current value"
        )));
    };
    if f0.abs() <= ftol0 {
        return Ok((SolveMethod::Unchanged, evals + evals_rf));
    }
    // Expanding bracket around the current value, nearest sign change first.
    let h0 = var.initial_step(&s.target, v0);
    let mut bracket = None;
    let (mut prev_p, mut prev_m) = ((v0, f0), (v0, f0));
    'expand: for i in 0..80 {
        let h = h0 * 2f64.powi(i);
        for (dir, prev) in [(1.0f64, &mut prev_p), (-1.0, &mut prev_m)] {
            let v = dir.mul_add(h, v0);
            if let Some((f, _)) = eval(v) {
                if (f > 0.0) != (prev.1 > 0.0) || f == 0.0 {
                    bracket = Some((prev.0, prev.1, v, f));
                    break 'expand;
                }
                *prev = (v, f);
            }
        }
        job.check()?;
    }
    let Some((a, fa, b, fb)) = bracket else {
        return Err(err(&format!(
            "{why_not}; and no sign change of the residual was found"
        )));
    };
    // The residual tolerance scales with the condition at the latest point.
    let ftol = std::cell::Cell::new(ftol0);
    let res = brent(
        |v| {
            eval(v).map(|(f, t)| {
                ftol.set(t);
                f
            })
        },
        (a, fa),
        (b, fb),
        |_| ftol.get(),
        settings.max_iterations,
    );
    let (v, fv) = res.map_err(|e| err(&format!("{why_not}; root finding: {e}")))?;
    let sys_v = trial(v).ok_or_else(|| err("root finding produced an invalid value"))?;
    let m = measure(s, &Eval::new(&sys_v, env)).map_err(|e| err(&e))?;
    if !m.converged(tol) {
        return Err(err(&format!(
            "{why_not}; root finding ended on a discontinuity (residual {fv:.3e}){}",
            m.degenerate_note(env.length)
        )));
    }
    *cur = sys_v;
    Ok((SolveMethod::RootFinding, evals + evals_rf + 1))
}

/// Brent's method (Brent 1973, *Algorithms for Minimization without
/// Derivatives*, ch. 4) on the bracket `[a, b]` with `f(a)·f(b) ≤ 0`:
/// inverse quadratic interpolation and secant steps, safeguarded by
/// bisection. Stops when `|f| ≤ ftol(x)` or the bracket has shrunk to a few
/// ulps. `f` returning `None` (an invalid system) is an error. Returns
/// `(x, f(x))`.
pub fn brent(
    mut f: impl FnMut(f64) -> Option<f64>,
    (mut a, mut fa): (f64, f64),
    (mut b, mut fb): (f64, f64),
    mut ftol: impl FnMut(f64) -> f64,
    max_iter: usize,
) -> Result<(f64, f64), String> {
    if (fa > 0.0) == (fb > 0.0) && fa != 0.0 && fb != 0.0 {
        return Err("the interval does not bracket a root".into());
    }
    if fa == 0.0 {
        return Ok((a, fa));
    }
    let (mut c, mut fc) = (a, fa);
    let mut d = b - a;
    let mut e = d;
    for _ in 0..max_iter {
        if (fb > 0.0) == (fc > 0.0) {
            (c, fc) = (a, fa);
            d = b - a;
            e = d;
        }
        if fc.abs() < fb.abs() {
            (a, fa) = (b, fb);
            (b, fb) = (c, fc);
            (c, fc) = (a, fa);
        }
        let tol1 = 2.0 * f64::EPSILON * b.abs() + f64::MIN_POSITIVE;
        let xm = 0.5 * (c - b);
        if fb == 0.0 || fb.abs() <= ftol(b) || xm.abs() <= tol1 {
            return Ok((b, fb));
        }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q);
            if a == c {
                p = 2.0 * xm * s;
                q = 1.0 - s;
            } else {
                let q0 = fa / fc;
                let r = fb / fc;
                p = s * (2.0 * xm * q0 * (q0 - r) - (b - a) * (r - 1.0));
                q = (q0 - 1.0) * (r - 1.0) * (s - 1.0);
            }
            if p > 0.0 {
                q = -q;
            } else {
                p = -p;
            }
            if 2.0 * p < (3.0 * xm * q - (tol1 * q).abs()).min((e * q).abs()) {
                e = d;
                d = p / q;
            } else {
                d = xm;
                e = d;
            }
        } else {
            d = xm;
            e = d;
        }
        (a, fa) = (b, fb);
        b += if d.abs() > tol1 { d } else { tol1.copysign(xm) };
        fb = f(b).ok_or_else(|| format!("the condition cannot be evaluated at {b}"))?;
    }
    Err(format!("no convergence in {max_iter} iterations"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::{CancelToken, Progress, samples};

    #[test]
    fn progress_is_strictly_increasing_and_below_one() {
        for n in [1, 2, 7] {
            let mut last = 0.0;
            for sweep in 1..=60 {
                for done in 1..=n {
                    let p = sweep_progress(sweep, done, n);
                    assert!(p > last && p < 1.0, "{sweep} {done} {n}: {p} after {last}");
                    last = p;
                }
            }
        }
    }

    /// Cancellation is seen inside one solve (both in the fixed point and in
    /// root finding), not only between solves.
    #[test]
    fn cancellation_inside_a_solve() {
        let mut sys = samples::double_gauss();
        let last = sys.surfaces[sys.image_index() - 1].id;
        sys.solves = vec![Solve::marginal_ray_height(last, 0.0)];
        let j = sys.image_index() - 1;
        sys.surfaces[j].thickness += 3.0;
        let cancel = CancelToken::new();
        cancel.cancel();
        let job = JobContext::with(cancel, Progress::default());
        let env = Env {
            lib: Arc::new(MaterialLibrary::with_builtin()),
            length: length_scale(&sys),
            job: &job,
        };
        for closed_form in [true, false] {
            let settings = SolveSettings {
                closed_form,
                ..SolveSettings::default()
            };
            let mut cur = sys.clone();
            let r = run_solve(0, &sys.solves, &[], &mut cur, &env, &settings);
            assert!(matches!(r, Err(MkError::Cancelled)), "{r:?}");
        }
        // The same solve completes without cancellation.
        let job = JobContext::new();
        let env = Env { job: &job, ..env };
        let mut cur = sys.clone();
        let (method, _) = run_solve(
            0,
            &sys.solves,
            &[],
            &mut cur,
            &env,
            &SolveSettings {
                closed_form: false,
                ..SolveSettings::default()
            },
        )
        .unwrap();
        assert_eq!(method, SolveMethod::RootFinding);
    }

    #[test]
    fn length_scale_ignores_rays_and_infinities() {
        let sys = samples::double_gauss();
        let track: f64 = sys.surfaces[1..].iter().map(|s| s.thickness.abs()).sum();
        assert!((length_scale(&sys) - track).abs() < 1e-12);
        let mut tiny = OpticalSystem::new_empty();
        tiny.surfaces.clear();
        tiny.aperture = ApertureSpec::ImageSpaceFNumber { value: 2.0 };
        assert_eq!(length_scale(&tiny), 1.0);
        tiny.aperture = ApertureSpec::EntrancePupilDiameter { value: 25.0 };
        assert_eq!(length_scale(&tiny), 25.0);
    }

    #[test]
    fn brent_known_roots() {
        // cos x = x has its root at the Dottie number.
        let dottie = 0.739_085_133_215_160_6;
        let (x, _) = brent(
            |x| Some(x.cos() - x),
            (0.0, 1.0),
            (1.0, 1f64.cos() - 1.0),
            |_| 0.0,
            100,
        )
        .unwrap();
        assert!((x - dottie).abs() < 4e-16);
        // x³ − 2: a root of odd multiplicity at 2^(1/3).
        let (x, _) = brent(
            |x| Some(x * x * x - 2.0),
            (0.0, -2.0),
            (5.0, 123.0),
            |_| 0.0,
            200,
        )
        .unwrap();
        assert!((x - 2f64.cbrt()).abs() < 1e-15);
        // Not a bracket.
        assert!(brent(|x| Some(x * x + 1.0), (-1.0, 2.0), (1.0, 2.0), |_| 0.0, 50).is_err());
        // A pole is "found" as a sign change but its residual is huge: the
        // caller's residual check rejects it.
        let (_, fx) = brent(|x| Some(1.0 / x), (-1.0, -1.0), (2.0, 0.5), |_| 0.0, 5000).unwrap();
        assert!(fx.abs() > 1e10);
        // An invalid evaluation inside the bracket is an error.
        assert!(brent(|_| None, (-1.0, -1.0), (1.0, 1.0), |_| 0.0, 50).is_err());
    }
}
