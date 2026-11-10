//! Discrete optimisation: variables that take one of a finite set of values
//! (glass substitution, catalogue lens or stock-part selection, choice of
//! surface type…).
//!
//! The search is a **tabu search** over a pluggable [`Neighbourhood`]: at each
//! iteration every move the neighbourhood proposes is evaluated in **one
//! batch**, the best non-tabu move is taken (even if it is worse, which lets
//! the search walk out of local minima), and the value a variable left is
//! tabu for `tabu_tenure` iterations. A tabu move is still allowed if it
//! beats the best assignment found (aspiration). After `stall_iterations`
//! without a new best, the search restarts from the best assignment with
//! `perturbation` random changes. Evaluations are cached, so revisiting an
//! assignment costs nothing.
//!
//! Glass substitution typically wants a neighbourhood of *optically similar*
//! glasses: [`NearestChoices::from_coordinates`] builds it from per-choice
//! coordinates such as `(n_d, V_d / 100, ΔP_g,F)`. When each discrete
//! assignment must be followed by a continuous re-optimisation (re-bending
//! lenses after a glass change), wrap the continuous problem in
//! [`LocallyRefined`].

use std::collections::HashMap;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use mklight_core::{JobContext, MkError, MkResult};

use crate::lsq::{LsqSettings, least_squares};
use crate::monitor::{DEFAULT_FEASIBILITY_TOL, IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{Problem, ScalarEval};
use crate::rand::{Rng, streams};

/// A problem over assignments `a[v] ∈ 0..num_choices()[v]`.
pub trait DiscreteProblem: Sync {
    fn num_choices(&self) -> &[usize];
    fn evaluate_batch(
        &self,
        assignments: &[Vec<usize>],
        ctx: &JobContext,
    ) -> MkResult<Vec<ScalarEval>>;
}

/// A move: set variable `.0` to choice `.1` for every listed pair.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Move(pub Vec<(usize, usize)>);

/// Generates candidate moves from the current assignment.
pub trait Neighbourhood: Sync {
    fn moves(&self, current: &[usize], choices: &[usize], rng: &mut Rng) -> Vec<Move>;
}

/// Every single-variable substitution (randomly subsampled to `max_moves`).
#[derive(Clone, Debug, Default)]
pub struct SingleChange {
    pub max_moves: Option<usize>,
}

impl Neighbourhood for SingleChange {
    fn moves(&self, cur: &[usize], choices: &[usize], rng: &mut Rng) -> Vec<Move> {
        let mut out = Vec::new();
        for (v, (&c, &n)) in cur.iter().zip(choices).enumerate() {
            for k in 0..n {
                if k != c {
                    out.push(Move(vec![(v, k)]));
                }
            }
        }
        subsample(out, self.max_moves, rng)
    }
}

fn subsample(mut v: Vec<Move>, max: Option<usize>, rng: &mut Rng) -> Vec<Move> {
    if let Some(m) = max
        && v.len() > m
    {
        for i in 0..m {
            let j = i + rng.index(v.len() - i);
            v.swap(i, j);
        }
        v.truncate(m);
    }
    v
}

/// Per variable and per choice, an explicit list of neighbouring choices.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NearestChoices {
    /// `neighbours[v][c]` = choices reachable from choice `c` of variable `v`.
    pub neighbours: Vec<Vec<Vec<usize>>>,
    /// Also propose pairs of simultaneous changes on two variables (sampled).
    #[serde(default)]
    pub pair_moves: usize,
}

impl NearestChoices {
    /// `k` nearest choices in a weighted Euclidean metric over per-choice
    /// coordinates `coords[v][c]` (e.g. `[n_d, V_d/100]` of each glass).
    pub fn from_coordinates(coords: &[Vec<Vec<f64>>], weights: &[f64], k: usize) -> Self {
        let neighbours = coords
            .iter()
            .map(|pts| {
                (0..pts.len())
                    .map(|c| {
                        let mut d: Vec<(f64, usize)> = (0..pts.len())
                            .filter(|&o| o != c)
                            .map(|o| {
                                let dist: f64 = pts[c]
                                    .iter()
                                    .zip(&pts[o])
                                    .enumerate()
                                    .map(|(i, (a, b))| {
                                        weights.get(i).copied().unwrap_or(1.0) * (a - b).powi(2)
                                    })
                                    .sum();
                                (dist, o)
                            })
                            .collect();
                        d.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                        d.into_iter().take(k).map(|(_, o)| o).collect()
                    })
                    .collect()
            })
            .collect();
        NearestChoices {
            neighbours,
            pair_moves: 0,
        }
    }
}

impl Neighbourhood for NearestChoices {
    fn moves(&self, cur: &[usize], _choices: &[usize], rng: &mut Rng) -> Vec<Move> {
        let mut singles = Vec::new();
        for (v, &c) in cur.iter().enumerate() {
            if let Some(list) = self.neighbours.get(v).and_then(|l| l.get(c)) {
                for &k in list {
                    singles.push((v, k));
                }
            }
        }
        let mut out: Vec<Move> = singles.iter().map(|&m| Move(vec![m])).collect();
        if singles.len() >= 2 {
            for _ in 0..self.pair_moves {
                let i = rng.index(singles.len());
                let j = rng.index(singles.len());
                if singles[i].0 != singles[j].0 {
                    out.push(Move(vec![singles[i], singles[j]]));
                }
            }
        }
        out
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscreteSettings {
    pub max_iterations: usize,
    pub max_evaluations: usize,
    pub tabu_tenure: usize,
    pub stall_iterations: usize,
    pub restarts: usize,
    /// Random changes applied to the best assignment at a restart.
    pub perturbation: usize,
    pub seed: u64,
    pub target_merit: f64,
}

impl Default for DiscreteSettings {
    fn default() -> Self {
        DiscreteSettings {
            max_iterations: 500,
            max_evaluations: 100_000,
            tabu_tenure: 5,
            stall_iterations: 20,
            restarts: 5,
            perturbation: 2,
            seed: 0,
            target_merit: f64::NEG_INFINITY,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DiscreteResult {
    pub assignment: Vec<usize>,
    pub merit: f64,
    pub violation: f64,
    pub termination: Termination,
    pub iterations: usize,
    /// Distinct assignments evaluated.
    pub evaluations: usize,
    pub history: Vec<IterationRecord>,
}

struct Cache<'a, D: DiscreteProblem + ?Sized> {
    p: &'a D,
    mon: &'a Monitor,
    map: HashMap<Vec<usize>, ScalarEval>,
}

impl<D: DiscreteProblem + ?Sized> Cache<'_, D> {
    fn eval(&mut self, xs: &[Vec<usize>]) -> MkResult<Vec<ScalarEval>> {
        let mut todo: Vec<Vec<usize>> = Vec::new();
        for x in xs {
            if !self.map.contains_key(x) && !todo.contains(x) {
                todo.push(x.clone());
            }
        }
        if !todo.is_empty() {
            let r = self.p.evaluate_batch(&todo, &self.mon.job)?;
            if r.len() != todo.len() {
                return Err(MkError::Numerical(
                    "discrete evaluation returned the wrong count".into(),
                ));
            }
            for (x, e) in todo.into_iter().zip(r) {
                self.mon
                    .best_assignment
                    .offer(&x, e, self.mon.feasibility_tol);
                self.map.insert(x, e);
            }
        }
        Ok(xs.iter().map(|x| self.map[x]).collect())
    }
}

/// The proposed moves that stay inside the choice ranges and change `cur`.
pub(crate) fn legal_moves(moves: Vec<Move>, cur: &[usize], choices: &[usize]) -> Vec<Move> {
    moves
        .into_iter()
        .filter(|m| {
            m.0.iter()
                .all(|&(v, c)| v < choices.len() && c < choices[v])
                && m.0.iter().any(|&(v, c)| cur[v] != c)
        })
        .collect()
}

/// Is `a` one choice per variable, each inside its range?
pub(crate) fn check_assignment(a: &[usize], choices: &[usize]) -> MkResult<()> {
    if a.len() != choices.len() || a.iter().zip(choices).any(|(a, n)| a >= n) {
        return Err(MkError::InvalidParam(
            "initial assignment out of range".into(),
        ));
    }
    Ok(())
}

fn apply(cur: &[usize], m: &Move) -> Vec<usize> {
    let mut a = cur.to_vec();
    for &(v, c) in &m.0 {
        a[v] = c;
    }
    a
}

/// Tabu search from `initial`.
pub fn tabu_search<D: DiscreteProblem + ?Sized, N: Neighbourhood + ?Sized>(
    p: &D,
    neighbourhood: &N,
    initial: &[usize],
    s: &DiscreteSettings,
    mon: &Monitor,
) -> MkResult<DiscreteResult> {
    let choices = p.num_choices().to_vec();
    check_assignment(initial, &choices)?;
    mon.begin_run();
    let tol = mon.feasibility_tol;
    let mut cache = Cache {
        p,
        mon,
        map: HashMap::new(),
    };
    let mut history = Vec::new();
    let mut cur = initial.to_vec();
    let mut fcur = match cache.eval(std::slice::from_ref(&cur)) {
        Ok(v) => v[0],
        Err(MkError::Cancelled) => ScalarEval::FAILED,
        Err(e) => return Err(e),
    };
    let mut best = (cur.clone(), fcur);
    let mut tabu: HashMap<(usize, usize), usize> = HashMap::new();
    let mut stall = 0;
    let mut restarts_left = s.restarts;
    let mut termination = Termination::MaxIterations;
    let mut it = 0;
    while it < s.max_iterations {
        if mon.cancelled() {
            termination = Termination::Cancelled;
            break;
        }
        if best.1.merit <= s.target_merit && best.1.violation <= tol {
            termination = Termination::TargetReached;
            break;
        }
        if cache.map.len() >= s.max_evaluations {
            termination = Termination::MaxEvaluations;
            break;
        }
        it += 1;
        let mut rng = Rng::new(s.seed, streams::DISCRETE | it as u64);
        let mut moves = legal_moves(
            neighbourhood.moves(&cur, &choices, &mut rng),
            &cur,
            &choices,
        );
        let mut cands: Vec<Vec<usize>> = moves.iter().map(|m| apply(&cur, m)).collect();
        // Keep the batch within the evaluation budget (cached ones are free).
        let room = s.max_evaluations.saturating_sub(cache.map.len());
        let mut fresh: Vec<&Vec<usize>> = Vec::new();
        let mut keep = cands.len();
        for (q, c) in cands.iter().enumerate() {
            if !cache.map.contains_key(c) && !fresh.contains(&c) {
                if fresh.len() == room {
                    keep = q;
                    break;
                }
                fresh.push(c);
            }
        }
        cands.truncate(keep);
        moves.truncate(keep);
        let fits = match cache.eval(&cands) {
            Ok(f) => f,
            Err(MkError::Cancelled) => {
                termination = Termination::Cancelled;
                break;
            }
            Err(e) => return Err(e),
        };
        let mut pick: Option<usize> = None;
        for (k, m) in moves.iter().enumerate() {
            let is_tabu =
                m.0.iter()
                    .any(|&(v, c)| tabu.get(&(v, c)).is_some_and(|&t| t >= it));
            let aspires = fits[k].better_than(&best.1, tol);
            if is_tabu && !aspires {
                continue;
            }
            if pick.is_none_or(|q| fits[k].better_than(&fits[q], tol)) {
                pick = Some(k);
            }
        }
        let mut improved = false;
        if let Some(k) = pick {
            for &(v, _) in &moves[k].0 {
                tabu.insert((v, cur[v]), it + s.tabu_tenure);
            }
            cur = cands[k].clone();
            fcur = fits[k];
            if fcur.better_than(&best.1, tol) {
                best = (cur.clone(), fcur);
                improved = true;
            }
        }
        stall = if improved { 0 } else { stall + 1 };
        let mut rec = IterationRecord::new("tabu", it, cache.map.len());
        rec.merit = fcur.merit;
        rec.violation = fcur.violation;
        rec.best_merit = best.1.merit;
        mon.record(&mut history, rec);
        mon.progress(it as f64 / s.max_iterations as f64, "discrete search");
        if pick.is_none() || stall >= s.stall_iterations {
            if restarts_left == 0 {
                termination = Termination::Stalled;
                break;
            }
            if cache.map.len() >= s.max_evaluations {
                termination = Termination::MaxEvaluations;
                break;
            }
            restarts_left -= 1;
            stall = 0;
            tabu.clear();
            cur = best.0.clone();
            for _ in 0..s.perturbation {
                let v = rng.index(cur.len());
                if choices[v] > 1 {
                    cur[v] = rng.index(choices[v]);
                }
            }
            fcur = match cache.eval(std::slice::from_ref(&cur)) {
                Ok(f) => f[0],
                Err(MkError::Cancelled) => {
                    termination = Termination::Cancelled;
                    break;
                }
                Err(e) => return Err(e),
            };
            if fcur.better_than(&best.1, tol) {
                best = (cur.clone(), fcur);
            }
        }
    }
    Ok(DiscreteResult {
        assignment: best.0,
        merit: best.1.merit,
        violation: best.1.violation,
        termination,
        iterations: it,
        evaluations: cache.map.len(),
        history,
    })
}

type DiscreteFn = dyn Fn(&[usize]) -> ScalarEval + Send + Sync;

/// A [`DiscreteProblem`] from a closure, evaluated in parallel.
pub struct FnDiscrete {
    choices: Vec<usize>,
    f: Box<DiscreteFn>,
}

impl FnDiscrete {
    pub fn new(
        choices: Vec<usize>,
        f: impl Fn(&[usize]) -> ScalarEval + Send + Sync + 'static,
    ) -> Self {
        FnDiscrete {
            choices,
            f: Box::new(f),
        }
    }
}

impl DiscreteProblem for FnDiscrete {
    fn num_choices(&self) -> &[usize] {
        &self.choices
    }
    fn evaluate_batch(&self, a: &[Vec<usize>], ctx: &JobContext) -> MkResult<Vec<ScalarEval>> {
        a.par_iter()
            .map(|x| {
                ctx.check()?;
                Ok((self.f)(x))
            })
            .collect()
    }
}

/// Continuous problem builder used by [`LocallyRefined`]: for an assignment,
/// return the continuous problem and its start point.
pub type BuildFn = dyn Fn(&[usize]) -> MkResult<(Box<dyn Problem + Send>, Vec<f64>)> + Send + Sync;

/// Mixed discrete/continuous problems: every assignment is scored by the
/// merit *after* a local least-squares solve of the continuous variables
/// (e.g. re-bending all lenses after a glass substitution).
pub struct LocallyRefined {
    pub choices: Vec<usize>,
    pub build: Box<BuildFn>,
    pub settings: LsqSettings,
}

impl LocallyRefined {
    /// Refine each assignment in parallel, from its built start point or from
    /// `starts[i]`. Results are in input order. Each answer is the best point
    /// the solve saw under the feasibility rule, so a solve interrupted, or
    /// ending on its budget short of feasibility, never loses a better start.
    pub fn refine_batch(
        &self,
        assignments: &[Vec<usize>],
        starts: Option<&[Vec<f64>]>,
        ctx: &JobContext,
        feasibility_tol: f64,
    ) -> Vec<MkResult<OptimResult>> {
        assignments
            .par_iter()
            .enumerate()
            .map(|(i, a)| {
                ctx.check()?;
                let (p, x0) = (self.build)(a)?;
                refine_one(
                    &*p,
                    starts.map_or(&x0, |s| &s[i]),
                    &self.settings,
                    ctx,
                    feasibility_tol,
                )
            })
            .collect()
    }
}

/// One local solve from `x0`, answering with the best point it saw under
/// the feasibility rule (re-scored by the problem's reference evaluation
/// when that is not the solver's own end point).
fn refine_one<P: Problem + ?Sized>(
    p: &P,
    x0: &[f64],
    settings: &LsqSettings,
    ctx: &JobContext,
    feasibility_tol: f64,
) -> MkResult<OptimResult> {
    let mut mon = Monitor::new(JobContext::with(ctx.cancel.clone(), Default::default()));
    mon.keep_history = false;
    mon.feasibility_tol = feasibility_tol;
    let mut r = least_squares(p, x0, settings, &mon)?;
    if let Some(best) = mon.best.get()
        && best.x != r.x
        && best.scalar().better_than(&r.scalar(), feasibility_tol)
    {
        (r.x, r.merit, r.violation) = (best.x, best.merit, best.violation);
        // Only a restored point needs another independent final score.
        if r.termination != Termination::Cancelled
            && let Some(reference) = p.reference_evaluation(&r.x, ctx)
        {
            match reference {
                Ok(e) => {
                    r.merit = crate::problem::merit(p.weights(), &e.residuals, e.status);
                    r.violation =
                        crate::problem::total_violation(p.constraints(), &e.constraints, e.status);
                }
                Err(MkError::Cancelled) => r.termination = Termination::Cancelled,
                Err(e) => return Err(e),
            }
        }
    }
    Ok(r)
}

impl DiscreteProblem for LocallyRefined {
    fn num_choices(&self) -> &[usize] {
        &self.choices
    }
    fn evaluate_batch(&self, a: &[Vec<usize>], ctx: &JobContext) -> MkResult<Vec<ScalarEval>> {
        self.refine_batch(a, None, ctx, DEFAULT_FEASIBILITY_TOL)
            .into_iter()
            .map(|r| {
                let r = r?;
                if r.termination == Termination::Cancelled {
                    return Err(MkError::Cancelled);
                }
                Ok(r.scalar())
            })
            .collect()
    }
}
