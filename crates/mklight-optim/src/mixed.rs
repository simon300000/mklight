//! Retained local refinements for tabu search and mixed glass/shape basin
//! hopping. The archive is assembled in input order, never completion order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use mklight_core::{JobContext, MkError, MkResult};
use serde::{Deserialize, Serialize};

use crate::discrete::{DiscreteProblem, LocallyRefined, Neighbourhood};
use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{EvalStatus, ScalarEval, project};
use crate::rand::{Rng, streams};

pub fn score(r: &OptimResult) -> ScalarEval {
    if r.merit.is_finite() && r.violation.is_finite() {
        ScalarEval {
            merit: r.merit,
            violation: r.violation,
            status: EvalStatus::Ok,
        }
    } else {
        ScalarEval::FAILED
    }
}

#[derive(Clone, Debug, Default)]
pub struct RefinementArchive {
    pub assignments: BTreeSet<Vec<usize>>,
    /// Best continuous answer for each distinct assignment.
    pub results: BTreeMap<Vec<usize>, OptimResult>,
    pub local_runs: usize,
    pub evaluations: usize,
    pub cache_hits: usize,
    pub failed: usize,
    pub history: Vec<IterationRecord>,
}

type StartKey = (Vec<usize>, Vec<u64>);

#[derive(Default)]
struct State {
    archive: RefinementArchive,
    cache: BTreeMap<StartKey, OptimResult>,
}

/// Tabu's scalar interface with a retained full prescription per assignment.
/// Cache keys include the continuous start: revisiting a glass combination
/// from a new basin must still run LM.
pub struct RefinedSearch {
    pub local: LocallyRefined,
    pub monitor: Monitor,
    state: Mutex<State>,
}

impl RefinedSearch {
    pub fn new(local: LocallyRefined, monitor: Monitor) -> Self {
        Self {
            local,
            monitor,
            state: Mutex::default(),
        }
    }

    pub fn archive(&self) -> RefinementArchive {
        self.state.lock().unwrap().archive.clone()
    }

    pub fn local_runs(&self) -> usize {
        self.state.lock().unwrap().archive.local_runs
    }

    pub fn refine(
        &self,
        assignments: &[Vec<usize>],
        starts: Option<&[Vec<f64>]>,
        ctx: &JobContext,
    ) -> MkResult<Vec<OptimResult>> {
        let keys: Vec<StartKey> = assignments
            .iter()
            .enumerate()
            .map(|(i, a)| {
                (
                    a.clone(),
                    starts.map_or_else(Vec::new, |s| s[i].iter().map(|v| v.to_bits()).collect()),
                )
            })
            .collect();
        let mut todo = Vec::new();
        {
            let mut state = self.state.lock().unwrap();
            for (i, key) in keys.iter().enumerate() {
                if state.cache.contains_key(key) || todo.iter().any(|&j| keys[j] == *key) {
                    state.archive.cache_hits += 1;
                } else {
                    todo.push(i);
                }
            }
        }
        let a: Vec<_> = todo.iter().map(|&i| assignments[i].clone()).collect();
        let x: Option<Vec<_>> = starts.map(|s| todo.iter().map(|&i| s[i].clone()).collect());
        let runs = self
            .local
            .refine_batch(&a, x.as_deref(), ctx, self.monitor.feasibility_tol);
        let mut error = None;
        let mut state = self.state.lock().unwrap();
        for (i, run) in todo.into_iter().zip(runs) {
            let r = match run {
                Ok(r) => r,
                Err(e) => {
                    error = Some(e);
                    continue;
                }
            };
            let fit = score(&r);
            let archive = &mut state.archive;
            archive.local_runs += 1;
            archive.assignments.insert(assignments[i].clone());
            archive.evaluations += r.evaluations;
            if fit.status.is_usable() {
                let assignment = &assignments[i];
                if archive
                    .results
                    .get(assignment)
                    .is_none_or(|b| fit.better_than(&score(b), self.monitor.feasibility_tol))
                {
                    archive.results.insert(assignment.clone(), r.clone());
                }
                self.monitor
                    .best_assignment
                    .offer(assignment, fit, self.monitor.feasibility_tol);
            } else {
                archive.failed += 1;
            }
            let mut rec =
                IterationRecord::new("glass_refinement", archive.local_runs, archive.evaluations);
            rec.merit = r.merit;
            rec.violation = r.violation;
            rec.best_merit = self
                .monitor
                .best_assignment
                .get()
                .map_or(f64::NAN, |b| b.merit);
            self.monitor.record(&mut archive.history, rec);
            if r.termination == Termination::Cancelled {
                error.get_or_insert(MkError::Cancelled);
            }
            state.cache.insert(keys[i].clone(), r);
        }
        if let Some(e) = error {
            return Err(e);
        }
        Ok(keys.iter().map(|key| state.cache[key].clone()).collect())
    }
}

impl DiscreteProblem for RefinedSearch {
    fn num_choices(&self) -> &[usize] {
        &self.local.choices
    }
    fn evaluate_batch(&self, a: &[Vec<usize>], ctx: &JobContext) -> MkResult<Vec<ScalarEval>> {
        Ok(self.refine(a, None, ctx)?.iter().map(score).collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MixedHammerSettings {
    pub rounds: usize,
    pub batch: usize,
    /// Gaussian perturbation in continuous variable scales.
    pub step: f64,
    pub swap_probability: f64,
    /// Relative Metropolis temperature (0 accepts only improvements).
    pub temperature: f64,
    pub stall_rounds: usize,
}

impl Default for MixedHammerSettings {
    fn default() -> Self {
        Self {
            rounds: 100,
            batch: 8,
            step: 0.5,
            swap_probability: 0.5,
            temperature: 0.05,
            stall_rounds: 20,
        }
    }
}

/// Basin hopping in both material assignment and continuous shape. Every
/// proposed move is locally refined; unlike continuous hammer, there is no
/// pre-refinement screen that could discard a promising re-bent glass pair.
/// `max_combinations` caps local runs, including repeated assignments with
/// different starting shapes. Exact repeated starts are cached.
pub fn mixed_hammer<N: Neighbourhood + ?Sized>(
    search: &RefinedSearch,
    neighbourhood: &N,
    initial: &[usize],
    settings: &MixedHammerSettings,
    max_combinations: usize,
    seed: u64,
    ctx: &JobContext,
) -> MkResult<Termination> {
    let s = settings;
    if s.rounds == 0
        || s.batch == 0
        || !s.step.is_finite()
        || s.step <= 0.0
        || !s.temperature.is_finite()
        || s.temperature < 0.0
        || !(0.0..=1.0).contains(&s.swap_probability)
        || s.stall_rounds == 0
        || max_combinations == 0
    {
        return Err(MkError::InvalidParam(
            "invalid mixed hammer budget or perturbation".into(),
        ));
    }
    let first = search.refine(&[initial.to_vec()], None, ctx)?;
    let mut current = (initial.to_vec(), first[0].clone());
    let mut best = score(&current.1);
    let mut step = s.step;
    let mut stall = 0;
    for round in 0..s.rounds {
        ctx.check()?;
        let room = max_combinations.saturating_sub(search.local_runs());
        if room == 0 {
            return Ok(Termination::MaxEvaluations);
        }
        let (problem, _) = (search.local.build)(&current.0)?;
        let vars = problem.variables();
        let mut rng = Rng::new(seed, streams::HAMMER | round as u64);
        let moves = neighbourhood.moves(&current.0, &search.local.choices, &mut rng);
        let mut assignments = Vec::new();
        let mut starts = Vec::new();
        for _ in 0..s.batch.min(room) {
            let mut a = current.0.clone();
            if !moves.is_empty() && rng.uniform() < s.swap_probability {
                for &(v, c) in &moves[rng.index(moves.len())].0 {
                    a[v] = c;
                }
            }
            let mut x: Vec<_> = current
                .1
                .x
                .iter()
                .zip(vars)
                .map(|(&x, v)| x + step * v.safe_scale() * rng.normal())
                .collect();
            project(vars, &mut x);
            assignments.push(a);
            starts.push(x);
        }
        let runs = search.refine(&assignments, Some(&starts), ctx)?;
        let pick = (0..runs.len())
            .min_by(|&a, &b| {
                compare(
                    score(&runs[a]),
                    score(&runs[b]),
                    search.monitor.feasibility_tol,
                )
                .then(a.cmp(&b))
            })
            .unwrap();
        let next = score(&runs[pick]);
        if next.better_than(&best, search.monitor.feasibility_tol) {
            best = next;
            stall = 0;
        } else {
            stall += 1;
        }
        let cur = score(&current.1);
        let accepted = next.better_than(&cur, search.monitor.feasibility_tol)
            || (s.temperature > 0.0
                && next.status.is_usable()
                && next.violation <= search.monitor.feasibility_tol
                && cur.violation <= search.monitor.feasibility_tol
                && rng.uniform()
                    < (-(next.merit - cur.merit)
                        / (s.temperature * cur.merit.abs().max(f64::MIN_POSITIVE)))
                    .exp());
        if accepted {
            current = (assignments[pick].clone(), runs[pick].clone());
        }
        step = (if accepted { step / 0.9 } else { step * 0.9 }).clamp(s.step * 1e-3, s.step * 10.0);
        if stall >= s.stall_rounds {
            return Ok(Termination::Stalled);
        }
    }
    Ok(Termination::MaxIterations)
}

pub fn compare(a: ScalarEval, b: ScalarEval, tolerance: f64) -> std::cmp::Ordering {
    if a.better_than(&b, tolerance) {
        std::cmp::Ordering::Less
    } else if b.better_than(&a, tolerance) {
        std::cmp::Ordering::Greater
    } else {
        std::cmp::Ordering::Equal
    }
}
