//! CMA-ES: covariance-matrix-adaptation evolution strategy (Hansen 2016,
//! "The CMA Evolution Strategy: A Tutorial"), `(μ/μ_w, λ)` with rank-one and
//! rank-μ updates, cumulative step-size adaptation and optional IPOP restarts
//! (population doubled at each restart, Auger & Hansen 2005).
//!
//! **Why CMA-ES rather than particle swarm.** Lens merit landscapes are
//! dominated by long, curved, strongly coupled valleys (bending of a lens,
//! power trades between neighbouring elements, curvature vs. air space). CMA-ES
//! learns that coupling as a covariance matrix and is invariant to rotations
//! and rescalings of the variables, so its behaviour does not depend on how a
//! designer parameterised the system; its defaults are derived from theory and
//! rarely need tuning. PSO is neither rotation invariant nor robust to
//! ill-conditioning and needs problem-specific inertia/acceleration tuning.
//! Both are equally batch friendly (one generation = one batch of `λ`).
//!
//! The strategy runs in variable-scale units `x_j / scale_j`.
//!
//! **Bounds:** sampled coordinates that leave the box are reflected back into
//! it (and clipped if they are still outside), and the *repaired* point is
//! used both for evaluation and for the distribution update ("Lamarckian"
//! repair), so the update is consistent with what was evaluated. Ranking uses
//! Deb's feasibility rule.

use serde::{Deserialize, Serialize};

use mklight_core::{MkError, MkResult};

use super::{Evaluator, require_domain};
use crate::linalg::{Matrix, sym_eigen};
use crate::monitor::{IterationRecord, Monitor, OptimResult, Termination};
use crate::problem::{Problem, ScalarEval, Variable};
use crate::rand::{Rng, streams};

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CmaEsSettings {
    /// Initial step size in variable scales.
    pub sigma0: f64,
    /// Population `λ`; 0 = `4 + ⌊3 ln n⌋`.
    pub population: usize,
    pub max_generations: usize,
    pub max_evaluations: usize,
    pub seed: u64,
    /// Stop when `σ · √(max eigenvalue of C)` (in scales) is below this.
    pub xtol: f64,
    /// Stop when the best merits of the last `10 + 30n/λ` generations and the
    /// current population span less than this (absolute).
    pub ftol: f64,
    pub target_merit: f64,
    /// IPOP restarts (each doubles `λ`).
    pub restarts: usize,
}

impl Default for CmaEsSettings {
    fn default() -> Self {
        CmaEsSettings {
            sigma0: 0.3,
            population: 0,
            max_generations: 10_000,
            max_evaluations: 1_000_000,
            seed: 0,
            xtol: 1e-11,
            ftol: 1e-15,
            target_merit: 0.0,
            restarts: 0,
        }
    }
}

fn reflect(v: &Variable, x: f64) -> f64 {
    let (lo, hi) = (v.lo(), v.hi());
    let mut y = x;
    if y < lo {
        y = lo + (lo - y);
    }
    if y > hi {
        y = hi - (y - hi);
    }
    v.clamp(y)
}

/// Minimise with CMA-ES. The initial mean is `x0` or, if absent, the centre
/// of the (then required) box.
pub fn cma_es<P: Problem + ?Sized>(
    p: &P,
    x0: Option<&[f64]>,
    s: &CmaEsSettings,
    mon: &Monitor,
) -> MkResult<OptimResult> {
    let vars = p.variables();
    require_domain(vars, x0)?;
    let n = vars.len();
    let nf = n as f64;
    let scale: Vec<f64> = vars.iter().map(|v| v.safe_scale()).collect();
    let mut m0: Vec<f64> = match x0 {
        Some(x) => x.to_vec(),
        None => vars.iter().map(|v| 0.5 * (v.lo() + v.hi())).collect(),
    };
    crate::problem::project(vars, &mut m0);
    let mut ev = Evaluator::new(p, mon, s.max_evaluations);
    let mut termination = Termination::MaxIterations;
    let mut generations = 0usize;
    let mut best: (Vec<f64>, ScalarEval) = (m0.clone(), ScalarEval::FAILED);
    let lambda0 = if s.population == 0 {
        4 + (3.0 * nf.ln()).floor() as usize
    } else {
        s.population.max(4)
    };

    'restarts: for restart in 0..=s.restarts {
        let lambda = lambda0 << restart.min(20);
        let mu = lambda / 2;
        let wraw: Vec<f64> = (1..=mu)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - (i as f64).ln())
            .collect();
        let wsum: f64 = wraw.iter().sum();
        let w: Vec<f64> = wraw.iter().map(|v| v / wsum).collect();
        let mueff = 1.0 / w.iter().map(|v| v * v).sum::<f64>();
        let cs = (mueff + 2.0) / (nf + mueff + 5.0);
        let ds = 1.0 + 2.0 * (((mueff - 1.0) / (nf + 1.0)).sqrt() - 1.0).max(0.0) + cs;
        let cc = (4.0 + mueff / nf) / (nf + 4.0 + 2.0 * mueff / nf);
        let c1 = 2.0 / ((nf + 1.3).powi(2) + mueff);
        let cmu = (1.0 - c1).min(2.0 * (mueff - 2.0 + 1.0 / mueff) / ((nf + 2.0).powi(2) + mueff));
        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        // State in scaled coordinates.
        let mut mean: Vec<f64> = (0..n).map(|j| m0[j] / scale[j]).collect();
        let mut sigma = s.sigma0;
        let mut ps = vec![0.0; n];
        let mut pc = vec![0.0; n];
        let mut c = Matrix::identity(n);
        let mut bm = Matrix::identity(n);
        let mut dd = vec![1.0; n];
        let hist_len = 10 + (30.0 * nf / lambda as f64).ceil() as usize;
        let mut recent: Vec<f64> = Vec::new();

        for g in 0..s.max_generations {
            if mon.cancelled() {
                termination = Termination::Cancelled;
                break 'restarts;
            }
            if !ev.budget_left(lambda) {
                termination = Termination::MaxEvaluations;
                break 'restarts;
            }
            let mut rng = Rng::new(s.seed, streams::CMAES | ((restart as u64) << 32) | g as u64);
            let mut ys: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            let mut xs: Vec<Vec<f64>> = Vec::with_capacity(lambda);
            for _ in 0..lambda {
                let z: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
                let dz: Vec<f64> = (0..n).map(|k| dd[k] * z[k]).collect();
                let y = bm.mul_vec(&dz);
                let x: Vec<f64> = (0..n)
                    .map(|j| reflect(&vars[j], (mean[j] + sigma * y[j]) * scale[j]))
                    .collect();
                // Lamarckian repair: the update uses the step actually taken.
                let y: Vec<f64> = (0..n)
                    .map(|j| (x[j] / scale[j] - mean[j]) / sigma)
                    .collect();
                ys.push(y);
                xs.push(x);
            }
            let fit = match ev.eval(&xs) {
                Ok(f) => f,
                Err(MkError::Cancelled) => {
                    termination = Termination::Cancelled;
                    break 'restarts;
                }
                Err(e) => return Err(e),
            };
            generations += 1;
            let mut order: Vec<usize> = (0..lambda).collect();
            order.sort_by(|&a, &b| ev.cmp(&fit[a], &fit[b]).then(a.cmp(&b)));
            let b0 = order[0];
            if fit[b0].better_than(&best.1, ev.tol()) {
                best = (xs[b0].clone(), fit[b0]);
            }
            // Mean update.
            let mut yw = vec![0.0; n];
            for (i, &k) in order[..mu].iter().enumerate() {
                for j in 0..n {
                    yw[j] += w[i] * ys[k][j];
                }
            }
            for j in 0..n {
                mean[j] += sigma * yw[j];
            }
            // C^{-1/2} y_w = B D⁻¹ Bᵀ y_w.
            let bty = bm.tr_mul_vec(&yw);
            let dinv: Vec<f64> = (0..n).map(|k| bty[k] / dd[k]).collect();
            let cinv_y = bm.mul_vec(&dinv);
            let a = (cs * (2.0 - cs) * mueff).sqrt();
            for j in 0..n {
                ps[j] = (1.0 - cs) * ps[j] + a * cinv_y[j];
            }
            let psn = crate::linalg::norm2(&ps);
            let hs = psn / (1.0 - (1.0 - cs).powi(2 * (g as i32 + 1))).sqrt()
                < (1.4 + 2.0 / (nf + 1.0)) * chi_n;
            let hsf = if hs { 1.0 } else { 0.0 };
            let ac = (cc * (2.0 - cc) * mueff).sqrt();
            for j in 0..n {
                pc[j] = (1.0 - cc) * pc[j] + hsf * ac * yw[j];
            }
            let delta_h = (1.0 - hsf) * cc * (2.0 - cc);
            for r in 0..n {
                for q in 0..=r {
                    let mut rank_mu = 0.0;
                    for (i, &k) in order[..mu].iter().enumerate() {
                        rank_mu += w[i] * ys[k][r] * ys[k][q];
                    }
                    let v = (1.0 - c1 - cmu) * c[(r, q)]
                        + c1 * (pc[r] * pc[q] + delta_h * c[(r, q)])
                        + cmu * rank_mu;
                    c[(r, q)] = v;
                    c[(q, r)] = v;
                }
            }
            sigma *= ((cs / ds) * (psn / chi_n - 1.0)).min(1.0).exp();
            let (ev_vals, vecs) = sym_eigen(&c);
            let maxev = ev_vals.iter().cloned().fold(0.0_f64, f64::max).max(1e-300);
            for k in 0..n {
                dd[k] = ev_vals[k].max(maxev * 1e-20).sqrt();
            }
            bm = vecs;

            let best_now = fit[b0];
            recent.push(best_now.merit);
            if recent.len() > hist_len {
                recent.remove(0);
            }
            let mut rec = IterationRecord::new("cma_es", generations, ev.evaluations);
            rec.stage = format!("restart {restart}");
            rec.merit = best_now.merit;
            rec.violation = best_now.violation;
            rec.best_merit = best.1.merit;
            rec.trust_radius = Some(sigma);
            rec.spread = Some(sigma * dd.iter().cloned().fold(0.0, f64::max));
            ev.record(rec);
            mon.progress(
                (restart as f64 + g as f64 / s.max_generations as f64) / (s.restarts + 1) as f64,
                "cma-es",
            );

            if best.1.merit <= s.target_merit && best.1.violation <= ev.tol() {
                termination = Termination::TargetReached;
                break 'restarts;
            }
            let worst_mu = fit[order[mu - 1]].merit;
            let frange = recent
                .iter()
                .chain(std::iter::once(&worst_mu))
                .fold(f64::NEG_INFINITY, |m, v| m.max(*v))
                - recent.iter().fold(f64::INFINITY, |m, v| m.min(*v));
            let xspread = sigma * dd.iter().cloned().fold(0.0, f64::max);
            let cond = maxev
                / ev_vals
                    .iter()
                    .cloned()
                    .fold(f64::INFINITY, f64::min)
                    .max(1e-300);
            if xspread <= s.xtol
                || (recent.len() >= hist_len && frange <= s.ftol)
                || cond > 1e14
                || !sigma.is_finite()
            {
                termination = Termination::Converged;
                break;
            }
            if g + 1 == s.max_generations {
                termination = Termination::MaxIterations;
            }
        }
        // Restart from a fresh mean: box-uniform when bounded, else x0.
        if restart < s.restarts {
            let mut rng = Rng::new(s.seed, streams::CMAES | (0xFFFF_u64 << 40) | restart as u64);
            m0 = (0..n)
                .map(|j| {
                    let v = &vars[j];
                    if v.is_bounded() {
                        rng.range(v.lo(), v.hi())
                    } else {
                        x0.map_or(0.0, |x| x[j])
                    }
                })
                .collect();
        }
    }
    let (bx, bf) = best;
    ev.finish((&bx, bf), termination, generations)
}
