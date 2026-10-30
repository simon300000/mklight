//! Discrete searches whose every assignment is re-optimised by
//! [`LocallyRefined`], keeping the full refined answer per assignment: tabu
//! search through [`RefinedSearch`]'s [`DiscreteProblem`] interface, and
//! [`mixed_hammer`], basin hopping over assignment and continuous shape
//! together. Archives are assembled in input order, never completion order.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

use mklight_core::{JobContext, MkError, MkResult};

use crate::discrete::{DiscreteProblem, LocallyRefined, Neighbourhood};
use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{ScalarEval, project};
use crate::rand::{Rng, streams};

#[derive(Clone, Debug, Default)]
pub struct RefinementArchive {
    /// Every assignment refined, including failed refinements.
    pub assignments: BTreeSet<Vec<usize>>,
    /// Best usable continuous answer for each assignment.
    pub results: BTreeMap<Vec<usize>, OptimResult>,
    pub local_runs: usize,
    pub evaluations: usize,
    pub failed: usize,
    pub history: Vec<IterationRecord>,
}

/// Runs and archives local refinements. As a [`DiscreteProblem`] every
/// assignment starts from the builder's start point, so its score does not
/// depend on the path the search took; tabu search caches it by assignment.
pub struct RefinedSearch {
    pub local: LocallyRefined,
    pub monitor: Monitor,
    archive: Mutex<RefinementArchive>,
}

impl RefinedSearch {
    pub fn new(local: LocallyRefined, monitor: Monitor) -> Self {
        Self {
            local,
            monitor,
            archive: Mutex::default(),
        }
    }

    pub fn archive(&self) -> RefinementArchive {
        self.archive.lock().unwrap().clone()
    }

    fn local_runs(&self) -> usize {
        self.archive.lock().unwrap().local_runs
    }

    /// Refine `assignments` in parallel (from `starts` when given) and
    /// archive the answers. Interrupted solves are archived before
    /// `Cancelled` is returned, so Stop keeps them.
    pub fn refine(
        &self,
        assignments: &[Vec<usize>],
        starts: Option<&[Vec<f64>]>,
        ctx: &JobContext,
    ) -> MkResult<Vec<OptimResult>> {
        let tol = self.monitor.feasibility_tol;
        let runs = self.local.refine_batch(assignments, starts, ctx, tol);
        let mut archive = self.archive.lock().unwrap();
        let mut out = Vec::with_capacity(runs.len());
        let mut error = None;
        for (a, run) in assignments.iter().zip(runs) {
            let r = match run {
                Ok(r) => r,
                Err(e) => {
                    error.get_or_insert(e);
                    continue;
                }
            };
            let fit = r.scalar();
            archive.local_runs += 1;
            archive.evaluations += r.evaluations;
            archive.assignments.insert(a.clone());
            if fit.status.is_usable() {
                if archive
                    .results
                    .get(a)
                    .is_none_or(|b| fit.better_than(&b.scalar(), tol))
                {
                    archive.results.insert(a.clone(), r.clone());
                }
                self.monitor.best_assignment.offer(a, fit, tol);
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
            out.push(r);
        }
        match error {
            Some(e) => Err(e),
            None => Ok(out),
        }
    }
}

impl DiscreteProblem for RefinedSearch {
    fn num_choices(&self) -> &[usize] {
        &self.local.choices
    }
    fn evaluate_batch(&self, a: &[Vec<usize>], ctx: &JobContext) -> MkResult<Vec<ScalarEval>> {
        Ok(self
            .refine(a, None, ctx)?
            .iter()
            .map(OptimResult::scalar)
            .collect())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MixedHammerSettings {
    /// Moves proposed and refined in parallel per round.
    pub batch: usize,
    /// Gaussian shape perturbation, in variable scales.
    pub step: f64,
    /// Probability that a move also swaps glasses.
    pub swap_probability: f64,
    /// Relative Metropolis temperature (0 accepts only improvements).
    pub temperature: f64,
    /// Rounds without a new best before stopping.
    pub stall_rounds: usize,
}

impl Default for MixedHammerSettings {
    fn default() -> Self {
        Self {
            batch: 8,
            step: 0.5,
            swap_probability: 0.5,
            temperature: 0.05,
            stall_rounds: 20,
        }
    }
}

/// Basin hopping over assignment and continuous shape. Each round proposes
/// `batch` moves (a Gaussian shape perturbation of the current point, plus a
/// neighbourhood swap with `swap_probability`), refines every one, and takes
/// the best by the acceptance rule of [`crate::global::hammer`]. Unlike the
/// continuous hammer there is no pre-refinement screen, which could discard
/// a glass pair that is only good once re-bent. Stops after
/// `max_combinations` local solves (the same glasses with another shape count
/// again) or `stall_rounds` rounds without a new best.
pub fn mixed_hammer<N: Neighbourhood + ?Sized>(
    search: &RefinedSearch,
    neighbourhood: &N,
    initial: &[usize],
    s: &MixedHammerSettings,
    max_combinations: usize,
    seed: u64,
    ctx: &JobContext,
) -> MkResult<Termination> {
    let tol = search.monitor.feasibility_tol;
    let first = search.refine(&[initial.to_vec()], None, ctx)?;
    let vars = (search.local.build)(initial)?.0.variables().to_vec();
    let mut current = (initial.to_vec(), first[0].clone());
    let mut best = current.1.scalar();
    let mut step = s.step;
    let mut stall = 0;
    let mut round = 0;
    loop {
        round += 1;
        let room = max_combinations.saturating_sub(search.local_runs());
        if room == 0 {
            return Ok(Termination::MaxEvaluations);
        }
        let mut rng = Rng::new(seed, streams::HAMMER | round);
        let moves = neighbourhood.moves(&current.0, &search.local.choices, &mut rng);
        let (assignments, starts): (Vec<_>, Vec<_>) = (0..s.batch.min(room))
            .map(|_| {
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
                    .zip(&vars)
                    .map(|(&x, v)| x + step * v.safe_scale() * rng.normal())
                    .collect();
                project(&vars, &mut x);
                (a, x)
            })
            .unzip();
        let runs = search.refine(&assignments, Some(&starts), ctx)?;
        let pick = (0..runs.len())
            .min_by(|&a, &b| runs[a].scalar().compare(&runs[b].scalar(), tol))
            .unwrap();
        let next = runs[pick].scalar();
        if next.better_than(&best, tol) {
            best = next;
            stall = 0;
        } else {
            stall += 1;
        }
        let cur = current.1.scalar();
        let accepted = next.better_than(&cur, tol)
            || (s.temperature > 0.0
                && next.violation <= tol
                && cur.violation <= tol
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
}
