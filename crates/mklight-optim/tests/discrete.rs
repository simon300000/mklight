//! Discrete optimisation: toy problems checked against brute force.

mod common;

use common::vars;
use mklight_core::JobContext;
use mklight_core::rng::Philox;
use mklight_optim::discrete::{
    DiscreteProblem, DiscreteSettings, FnDiscrete, LocallyRefined, Move, NearestChoices,
    Neighbourhood, SingleChange, tabu_search,
};
use mklight_optim::lsq::LsqSettings;
use mklight_optim::mixed::{MixedHammerSettings, RefinedSearch, mixed_hammer};
use mklight_optim::problem::{EvalStatus, ScalarEval};
use mklight_optim::rand::Rng;
use mklight_optim::{FnProblem, Monitor, Problem, Termination};

fn ok(m: f64) -> ScalarEval {
    ScalarEval {
        merit: m,
        violation: 0.0,
        status: EvalStatus::Ok,
    }
}

/// Six variables with five choices each; unary + chained pairwise costs from
/// a fixed Philox table (a rugged landscape with many local minima).
fn toy() -> FnDiscrete {
    let mut r = Philox::new(1234, 0);
    let unary: Vec<Vec<f64>> = (0..6)
        .map(|_| (0..5).map(|_| r.uniform()).collect())
        .collect();
    let pair: Vec<Vec<f64>> = (0..5)
        .map(|_| (0..5).map(|_| 2.0 * r.uniform()).collect())
        .collect();
    FnDiscrete::new(vec![5; 6], move |a| {
        let mut f = 0.0;
        for i in 0..6 {
            f += unary[i][a[i]];
            if i + 1 < 6 {
                f += pair[a[i]][a[i + 1]];
            }
        }
        ok(f)
    })
}

fn brute_force<D: DiscreteProblem>(p: &D) -> (Vec<usize>, f64) {
    let ch = p.num_choices().to_vec();
    let total: usize = ch.iter().product();
    let all: Vec<Vec<usize>> = (0..total)
        .map(|mut k| {
            ch.iter()
                .map(|&n| {
                    let v = k % n;
                    k /= n;
                    v
                })
                .collect()
        })
        .collect();
    let f = p.evaluate_batch(&all, &JobContext::new()).unwrap();
    let i = (0..total)
        .min_by(|&a, &b| f[a].merit.total_cmp(&f[b].merit))
        .unwrap();
    (all[i].clone(), f[i].merit)
}

#[test]
fn tabu_finds_brute_force_optimum() {
    let p = toy();
    let (xa, fa) = brute_force(&p);
    let s = DiscreteSettings {
        seed: 3,
        ..Default::default()
    };
    let r = tabu_search(
        &p,
        &SingleChange::default(),
        &[0; 6],
        &s,
        &Monitor::default(),
    )
    .unwrap();
    assert_eq!(r.assignment, xa);
    assert!((r.merit - fa).abs() < 1e-15);
    assert!(
        r.evaluations < 15625 / 2,
        "used {} evaluations",
        r.evaluations
    );
    // Deterministic.
    let r2 = tabu_search(
        &p,
        &SingleChange::default(),
        &[0; 6],
        &s,
        &Monitor::default(),
    )
    .unwrap();
    assert_eq!(r.history, r2.history);
}

/// Glass-substitution flavoured mixed problem: choose two glasses (V numbers)
/// for a thin achromatic doublet; for each choice the element powers are
/// solved by least squares for unit total power and zero axial colour, with a
/// small penalty on large powers. Best = largest Abbe-number difference.
#[test]
fn glass_choice_with_local_refinement() {
    // (n_d, V_d) of eight model glasses.
    let glasses: Vec<(f64, f64)> = vec![
        (1.5168, 64.2),
        (1.6200, 36.4),
        (1.5891, 61.3),
        (1.7552, 27.6),
        (1.4875, 70.4),
        (1.6727, 32.2),
        (1.5481, 45.8),
        (1.8467, 23.8),
    ];
    let g2 = glasses.clone();
    let problem = LocallyRefined {
        choices: vec![glasses.len(); 2],
        build: Box::new(move |a: &[usize]| {
            let (v1, v2) = (g2[a[0]].1, g2[a[1]].1);
            let p = FnProblem::residuals(vars(2), 4, move |x, r| {
                r[0] = 10.0 * (x[0] + x[1] - 1.0);
                r[1] = 100.0 * (x[0] / v1 + x[1] / v2);
                r[2] = 0.1 * x[0];
                r[3] = 0.1 * x[1];
            });
            Ok((Box::new(p) as Box<dyn Problem + Send>, vec![1.0, 0.0]))
        }),
        settings: LsqSettings::default(),
    };
    let (xa, fa) = brute_force(&problem);
    let coords: Vec<Vec<Vec<f64>>> =
        vec![glasses.iter().map(|g| vec![g.0, g.1 / 100.0]).collect(); 2];
    let mut nb = NearestChoices::from_coordinates(&coords, &[1.0, 1.0], 3);
    nb.pair_moves = 4;
    let s = DiscreteSettings {
        seed: 1,
        ..Default::default()
    };
    let r = tabu_search(&problem, &nb, &[0, 2], &s, &Monitor::default()).unwrap();
    assert!(
        (r.merit - fa).abs() < 1e-12,
        "{:?} {} vs {xa:?} {fa}",
        r.assignment,
        r.merit
    );
    // The optimum pairs the highest- and lowest-V glasses.
    let mut v = [glasses[r.assignment[0]].1, glasses[r.assignment[1]].1];
    v.sort_by(f64::total_cmp);
    assert_eq!(v, [23.8, 70.4]);
}

#[test]
fn nearest_choices_metric() {
    let coords = vec![vec![vec![0.0], vec![1.0], vec![3.0], vec![10.0]]];
    let nb = NearestChoices::from_coordinates(&coords, &[1.0], 2);
    assert_eq!(nb.neighbours[0][0], vec![1, 2]);
    assert_eq!(nb.neighbours[0][3], vec![2, 1]);
}

/// A neighbourhood without moves: the search advances by restarts only.
struct NoMoves;

impl Neighbourhood for NoMoves {
    fn moves(&self, _: &[usize], _: &[usize], _: &mut Rng) -> Vec<Move> {
        Vec::new()
    }
}

#[test]
fn a_better_restart_point_becomes_the_answer() {
    // Every change away from the all-zero start improves the merit.
    let p = FnDiscrete::new(vec![4; 3], |a| ok(a.iter().map(|&c| (3 - c) as f64).sum()));
    let s = DiscreteSettings {
        seed: 2,
        ..Default::default()
    };
    let mon = Monitor::default();
    let r = tabu_search(&p, &NoMoves, &[0; 3], &s, &mon).unwrap();
    let best = mon.best_assignment.get().unwrap();
    assert!(best.merit < 9.0, "{best:?}");
    assert_eq!((r.assignment, r.merit), (best.assignment, best.merit));
}

#[test]
fn mixed_hammer_checks_its_settings_and_moves() {
    let local = || LocallyRefined {
        choices: vec![3; 2],
        build: Box::new(|a: &[usize]| {
            let shift = (a[0] + 2 * a[1]) as f64;
            let p = FnProblem::residuals(vars(1), 2, move |x, r| {
                r[0] = x[0] - 1.0;
                r[1] = 0.1 * shift;
            });
            Ok((Box::new(p) as Box<dyn Problem + Send>, vec![0.0]))
        }),
        settings: LsqSettings::default(),
    };
    /// Proposes a variable, then a choice, that does not exist.
    struct OutOfRange;
    impl Neighbourhood for OutOfRange {
        fn moves(&self, _: &[usize], _: &[usize], _: &mut Rng) -> Vec<Move> {
            vec![Move(vec![(5, 0)]), Move(vec![(0, 7)])]
        }
    }
    let ctx = JobContext::new();
    let run = |settings: MixedHammerSettings, initial: &[usize]| {
        let search = RefinedSearch::new(local(), Monitor::default());
        let t = mixed_hammer(&search, &OutOfRange, initial, &settings, 12, 3, &ctx);
        (t, search.archive())
    };
    // A batch of zero still refines one move per round; illegal moves are
    // dropped, so only the start assignment is ever refined.
    let always_swap = MixedHammerSettings {
        batch: 0,
        swap_probability: 1.0,
        ..Default::default()
    };
    let (t, archive) = run(always_swap, &[1, 1]);
    assert_eq!(t.unwrap(), Termination::MaxEvaluations);
    assert_eq!(archive.local_runs, 12);
    assert_eq!(
        archive.assignments.into_iter().collect::<Vec<_>>(),
        [[1, 1]]
    );
    let negative = MixedHammerSettings {
        step: -0.5,
        ..Default::default()
    };
    for (settings, initial) in [(negative, &[1, 1][..]), (always_swap, &[1, 3][..])] {
        let (t, _) = run(settings, initial);
        assert!(
            matches!(t, Err(mklight_core::MkError::InvalidParam(_))),
            "{t:?}"
        );
    }
}
