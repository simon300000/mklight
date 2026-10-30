//! Differential evolution (Storn & Price 1997).
//!
//! Each generation builds one trial vector per individual by mutation
//! (`rand/1`, `best/1` or `current-to-best/1`) and binomial crossover, then
//! evaluates the **whole generation as one batch** and keeps a trial when it
//! is at least as good as its parent under Deb's feasibility rule. The
//! mutation factor is dithered per individual in `[F_min, F_max]`, which is
//! more robust than a fixed `F`. Mutant components that leave the box are
//! "bounced back" half-way between the base vector and the violated bound,
//! which keeps the population's distribution near the boundary instead of
//! piling individuals onto it (as clipping would).
//!
//! Random numbers: generation `g` draws from Philox stream `DE | g`, in a
//! fixed order, so a run is reproducible for a given seed whatever the
//! evaluation backend or thread count.

use serde::{Deserialize, Serialize};

use mklight_core::{MkError, MkResult};

use super::{Evaluator, require_domain, sample_coord, spread};
use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::Problem;
use crate::rand::{Rng, streams};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeStrategy {
    /// `v = x_r1 + F (x_r2 − x_r3)`: most explorative.
    #[default]
    Rand1Bin,
    /// `v = x_best + F (x_r1 − x_r2)`: fast, greedier.
    Best1Bin,
    /// `v = x_i + F (x_best − x_i) + F (x_r1 − x_r2)`.
    CurrentToBest1Bin,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeSettings {
    pub strategy: DeStrategy,
    /// Population size; 0 = `max(15, 10 n)`.
    pub population: usize,
    pub f_min: f64,
    pub f_max: f64,
    /// Crossover probability.
    pub cr: f64,
    pub max_generations: usize,
    pub max_evaluations: usize,
    pub seed: u64,
    /// Stop when the merit range of the population is ≤ `ftol·(1 + |best|)`
    /// and its spread (in variable scales) ≤ `xtol`.
    pub ftol: f64,
    pub xtol: f64,
    pub target_merit: f64,
    /// For unbounded variables: initial spread around `x0`, in scales.
    pub init_spread: f64,
}

impl Default for DeSettings {
    fn default() -> Self {
        DeSettings {
            strategy: DeStrategy::Rand1Bin,
            population: 0,
            f_min: 0.5,
            f_max: 1.0,
            cr: 0.9,
            max_generations: 1000,
            max_evaluations: 1_000_000,
            seed: 0,
            ftol: 1e-12,
            xtol: 1e-9,
            target_merit: 0.0,
            init_spread: 1.0,
        }
    }
}

/// Index of the best individual (ties → lowest index).
fn best_index<P: Problem + ?Sized>(fit: &[crate::problem::ScalarEval], ev: &Evaluator<P>) -> usize {
    (0..fit.len())
        .min_by(|&a, &b| ev.cmp(&fit[a], &fit[b]).then(a.cmp(&b)))
        .unwrap_or(0)
}

/// Minimise with differential evolution. `x0` (if given) is inserted into
/// the initial population; it is required when a variable is unbounded.
pub fn differential_evolution<P: Problem + ?Sized>(
    p: &P,
    x0: Option<&[f64]>,
    s: &DeSettings,
    mon: &Monitor,
) -> MkResult<OptimResult> {
    let vars = p.variables();
    require_domain(vars, x0)?;
    let n = vars.len();
    let np = if s.population == 0 {
        (10 * n).max(15)
    } else {
        s.population
    };
    if np < 4 {
        return Err(MkError::InvalidParam(
            "differential evolution needs a population of at least 4".into(),
        ));
    }
    let mut ev = Evaluator::new(p, mon, s.max_evaluations);

    // Initial population: x0 plus samples (uniform in the box, or around x0).
    let mut rng = Rng::new(s.seed, streams::DE | (u32::MAX as u64));
    let mut pop: Vec<Vec<f64>> = Vec::with_capacity(np);
    if let Some(x) = x0 {
        let mut x = x.to_vec();
        crate::problem::project(vars, &mut x);
        pop.push(x);
    }
    while pop.len() < np {
        let centre = x0;
        pop.push(
            vars.iter()
                .enumerate()
                .map(|(j, v)| {
                    let c = if v.is_bounded() {
                        None
                    } else {
                        centre.map(|x| x[j])
                    };
                    sample_coord(v, c, s.init_spread, &mut rng)
                })
                .collect(),
        );
    }
    if mon.cancelled() {
        let f = crate::problem::ScalarEval::FAILED;
        return Ok(ev.finish((&pop[0], f), Termination::Cancelled, 0));
    }
    // The initial population must fit the budget as a whole; otherwise only
    // its first members are evaluated and the run ends there.
    if !ev.budget_left(np) {
        pop.truncate(ev.remaining());
        if pop.is_empty() {
            let f = crate::problem::ScalarEval::FAILED;
            let x = x0.map_or_else(
                || vars.iter().map(|v| v.clamp(0.0)).collect(),
                |x| x.to_vec(),
            );
            return Ok(ev.finish((&x, f), Termination::MaxEvaluations, 0));
        }
        return match ev.eval(&pop) {
            Ok(fit) => {
                let b = best_index(&fit, &ev);
                Ok(ev.finish((&pop[b].clone(), fit[b]), Termination::MaxEvaluations, 0))
            }
            Err(MkError::Cancelled) => {
                let f = crate::problem::ScalarEval::FAILED;
                Ok(ev.finish((&pop[0].clone(), f), Termination::Cancelled, 0))
            }
            Err(e) => Err(e),
        };
    }
    let mut fit = match ev.eval(&pop) {
        Ok(f) => f,
        Err(MkError::Cancelled) => {
            let f = crate::problem::ScalarEval::FAILED;
            return Ok(ev.finish((&pop[0], f), Termination::Cancelled, 0));
        }
        Err(e) => return Err(e),
    };
    let mut best = best_index(&fit, &ev);
    let mut termination = Termination::MaxIterations;
    let mut gen_done = 0;
    for g in 0..s.max_generations {
        if mon.cancelled() {
            termination = Termination::Cancelled;
            break;
        }
        if fit[best].merit <= s.target_merit && fit[best].violation <= ev.tol() {
            termination = Termination::TargetReached;
            break;
        }
        if !ev.budget_left(np) {
            termination = Termination::MaxEvaluations;
            break;
        }
        let mut rng = Rng::new(s.seed, streams::DE | g as u64);
        let mut trials = Vec::with_capacity(np);
        for i in 0..np {
            let f = rng.range(s.f_min, s.f_max);
            let r = rng.distinct(np, 3, &[i]);
            let (base, mutant): (&[f64], Vec<f64>) = match s.strategy {
                DeStrategy::Rand1Bin => (
                    &pop[r[0]],
                    (0..n)
                        .map(|j| pop[r[0]][j] + f * (pop[r[1]][j] - pop[r[2]][j]))
                        .collect(),
                ),
                DeStrategy::Best1Bin => (
                    &pop[best],
                    (0..n)
                        .map(|j| pop[best][j] + f * (pop[r[0]][j] - pop[r[1]][j]))
                        .collect(),
                ),
                DeStrategy::CurrentToBest1Bin => (
                    &pop[i],
                    (0..n)
                        .map(|j| {
                            pop[i][j]
                                + f * (pop[best][j] - pop[i][j])
                                + f * (pop[r[0]][j] - pop[r[1]][j])
                        })
                        .collect(),
                ),
            };
            let jrand = rng.index(n);
            let mut t = pop[i].clone();
            for j in 0..n {
                let u = rng.uniform();
                if j == jrand || u < s.cr {
                    let mut v = mutant[j];
                    let var = &vars[j];
                    if v < var.lo() {
                        v = 0.5 * (base[j] + var.lo());
                    } else if v > var.hi() {
                        v = 0.5 * (base[j] + var.hi());
                    }
                    t[j] = var.clamp(v);
                }
            }
            trials.push(t);
        }
        let tf = match ev.eval(&trials) {
            Ok(f) => f,
            Err(MkError::Cancelled) => {
                termination = Termination::Cancelled;
                break;
            }
            Err(e) => return Err(e),
        };
        for i in 0..np {
            if !fit[i].better_than(&tf[i], ev.tol()) {
                pop[i] = std::mem::take(&mut trials[i]);
                fit[i] = tf[i];
            }
        }
        best = best_index(&fit, &ev);
        gen_done = g + 1;
        let sp = spread(&pop, vars);
        let finite: Vec<f64> = fit
            .iter()
            .filter(|f| f.violation <= ev.tol())
            .map(|f| f.merit)
            .collect();
        let frange = if finite.len() == np {
            finite.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - finite.iter().cloned().fold(f64::INFINITY, f64::min)
        } else {
            f64::INFINITY
        };
        let mut rec = IterationRecord::new("de", gen_done, ev.evaluations);
        rec.merit = fit[best].merit;
        rec.violation = fit[best].violation;
        rec.best_merit = fit[best].merit;
        rec.spread = Some(sp);
        ev.record(rec);
        mon.progress(
            gen_done as f64 / s.max_generations as f64,
            "differential evolution",
        );
        if frange <= s.ftol * (1.0 + fit[best].merit.abs()) && sp <= s.xtol {
            termination = Termination::Converged;
            break;
        }
    }
    let (bx, bf) = (pop[best].clone(), fit[best]);
    Ok(ev.finish((&bx, bf), termination, gen_done))
}
