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
//! Everything is deterministic: fixed order, no randomness, no threads.
//! Multi-configuration systems are solved per configuration *after* its
//! overrides ([`apply_solves_configuration`]); an override of a
//! solve-controlled parameter is overwritten and reported as a warning.

use std::cell::OnceCell;
use std::sync::Arc;

use mklight_core::job::{Precision, RunTimer};
use mklight_core::materials::MaterialLibrary;
use mklight_core::model::Solve;
use mklight_core::param::ParamPath;
use mklight_core::{
    JobContext, MkError, MkResult, OpticalSystem, ResolvedSystem, RunInfo, SemiDiameter, Snapshot,
    SolveKind,
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
    pub run_info: RunInfo,
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

/// Solve configuration `k` (its overrides first, then the solves).
pub fn apply_solves_configuration(
    sys: &OpticalSystem,
    lib: &MaterialLibrary,
    k: usize,
    settings: &SolveSettings,
    ctx: &JobContext,
) -> MkResult<(OpticalSystem, SolveReport)> {
    let cfg = sys.with_configuration(k)?;
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
    let (out, mut report) = apply_solves_with(&cfg, lib, settings, ctx)?;
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

/// Solve `sys` (module docs). Cancellable between solves; progress is
/// monotone and reaches 1 on success.
pub fn apply_solves_with(
    sys: &OpticalSystem,
    lib: &MaterialLibrary,
    settings: &SolveSettings,
    ctx: &JobContext,
) -> MkResult<(OpticalSystem, SolveReport)> {
    let timer = RunTimer::start();
    let input = Snapshot::new(sys.clone())?;
    if !(settings.tolerance > 0.0 && settings.tolerance < 1.0) || settings.max_sweeps == 0 {
        return Err(MkError::InvalidModel(
            "solve settings: tolerance must be in (0, 1) and max_sweeps ≥ 1".into(),
        ));
    }
    let lib = Arc::new(lib.clone());
    let order = sys.solve_order()?;
    let mut warnings = Vec::new();
    if sys.solves.iter().any(|s| s.kind.is_paraxial())
        && sys
            .surfaces
            .iter()
            .any(|s| s.placement.is_some_and(|p| !p.is_identity()))
    {
        warnings.push(
            "system has tilts or decenters: paraxial solves use the unfolded, rotationally \
             symmetric equivalent"
                .into(),
        );
    }

    let deps: Vec<Vec<usize>> = (0..sys.solves.len())
        .map(|k| dependents(sys, &order, k))
        .collect();
    let mut cur = sys.clone();
    let mut stats: Vec<(SolveMethod, usize)> = vec![(SolveMethod::Unchanged, 0); sys.solves.len()];
    let mut sweeps = 0;
    let mut max_change = f64::INFINITY;
    let m = order.len().max(1) as f64;
    while sweeps < settings.max_sweeps {
        sweeps += 1;
        max_change = 0.0;
        let (lo, hi) = (1.0 - 1.0 / sweeps as f64, 1.0 - 1.0 / (sweeps + 1) as f64);
        for (done, &k) in order.iter().enumerate() {
            ctx.check()?;
            let solve = &sys.solves[k];
            let before = solve.target.get(&cur).ok();
            let (method, evals) = run_solve(k, &sys.solves, &deps[k], &mut cur, &lib, settings)?;
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
            ctx.report(lo + (hi - lo) * (done + 1) as f64 / m, "solves");
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

    // Final residuals, all measured on the solved system.
    let eval = Eval::new(&cur, &lib);
    let mut outcomes = Vec::with_capacity(sys.solves.len());
    for (k, s) in sys.solves.iter().enumerate() {
        let m = measure(s, &eval).map_err(|e| solve_error(k, s, &e))?;
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
        settings,
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

/// Lazily evaluated paraxial data and semi-diameters of one system.
struct Eval<'a> {
    sys: &'a OpticalSystem,
    lib: &'a Arc<MaterialLibrary>,
    para: OnceCell<Result<Para, String>>,
    sd: OnceCell<Result<Vec<f64>, String>>,
}

impl<'a> Eval<'a> {
    fn new(sys: &'a OpticalSystem, lib: &'a Arc<MaterialLibrary>) -> Self {
        Eval {
            sys,
            lib,
            para: OnceCell::new(),
            sd: OnceCell::new(),
        }
    }
    fn para(&self) -> Result<&Para, String> {
        self.para
            .get_or_init(|| {
                let snap = Snapshot::new(self.sys.clone()).map_err(|e| e.to_string())?;
                let resolved =
                    ResolvedSystem::new(&snap, self.lib.clone()).map_err(|e| e.to_string())?;
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
                Ok(tracer
                    .semi_diameters(&SemiDiameterSettings::default())
                    .values)
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
    scale: f64,
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
            let j = idx(sys, tgt)?;
            let u = ray.u[j];
            Ok(Measure {
                measured: ray.y[j + 1],
                goal: *height,
                scale: height.abs().max(max_abs(&ray.y[1..])),
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
            Ok(Measure {
                measured: ray.u[k],
                goal: *slope,
                scale: slope.abs().max(max_abs(&ray.u)),
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
                scale: distance.abs().max(sum((a..b).map(|i| t(i).abs()))).max(1.0),
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
                scale: value.abs().max(t.abs()).max(1.0),
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
    lib: &Arc<MaterialLibrary>,
) -> Result<(), String> {
    for &j in deps {
        let x = measure(&solves[j], &Eval::new(sys, lib))?.update?;
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
/// of evaluations.
fn run_solve(
    k: usize,
    solves: &[Solve],
    deps: &[usize],
    cur: &mut OpticalSystem,
    lib: &Arc<MaterialLibrary>,
    settings: &SolveSettings,
) -> MkResult<(SolveMethod, usize)> {
    let s = &solves[k];
    let tol = settings.tolerance;
    let err = |why: &str| solve_error(k, s, why);
    let max_fixed_point = settings.max_iterations.min(60);
    let mut evals = 0usize;
    let mut updates = 0usize;
    let mut why_not = String::from("closed forms disabled");
    apply_dependents(cur, solves, deps, lib).map_err(|e| err(&e))?;

    if settings.closed_form {
        why_not = "fixed-point iteration did not converge".into();
        let mut last_res = f64::INFINITY;
        let mut slow = 0;
        while evals < max_fixed_point {
            let m = measure(s, &Eval::new(cur, lib)).map_err(|e| err(&e))?;
            evals += 1;
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
                    why_not = format!("fixed-point iteration stalled at residual {r:.3e}");
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
            s.target.set(cur, x).map_err(|e| err(&e.to_string()))?;
            apply_dependents(cur, solves, deps, lib).map_err(|e| err(&e))?;
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
        apply_dependents(&mut sys, solves, deps, lib).ok()?;
        Some(sys)
    };
    let mut eval = |v: f64| -> Option<(f64, f64)> {
        evals_rf += 1;
        let sys = trial(v)?;
        let m = measure(s, &Eval::new(&sys, lib)).ok()?;
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
    let m = measure(s, &Eval::new(&sys_v, lib)).map_err(|e| err(&e))?;
    if !m.converged(tol) {
        return Err(err(&format!(
            "{why_not}; root finding ended on a discontinuity (residual {fv:.3e})"
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
