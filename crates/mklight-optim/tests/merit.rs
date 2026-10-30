//! Merit-function data model, pupil quadrature and the optical-problem bridge.

use mklight_core::model::{Field, FieldKind, MaterialRef, SemiDiameter, Surface, SurfaceShape};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, OpticalSystem, SurfaceId};
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::merit::default::{DefaultCriterion, DefaultMeritSpec};
use mklight_optim::merit::geometric::ModelEvaluator;
use mklight_optim::merit::operand::{Axis, MediumFilter, ThicknessMeasure};
use mklight_optim::merit::{
    CompositeEvaluator, FailurePolicy, MeritFunctionSpec, Operand, OperandError, OperandEvaluator,
    OperandKind, OperandRegistry, OperandValue, OpticalProblem, PupilSampling, PupilSymmetry,
    VariableSpec,
};
use mklight_optim::problem::evaluate_one;
use mklight_optim::{EvalStatus, Monitor, Problem, Termination};

// ---------- quadrature ----------

fn fact(n: u32) -> f64 {
    (1..=n).map(|k| k as f64).product()
}
/// Γ(k + ½) for integer k ≥ 0.
fn gamma_half(k: u32) -> f64 {
    fact(2 * k) * std::f64::consts::PI.sqrt() / (4f64.powi(k as i32) * fact(k))
}
/// ∬ over the annulus ε ≤ ρ ≤ 1 of xᵃ yᵇ, divided by its area.
fn disk_mean(a: u32, b: u32, eps: f64) -> f64 {
    if a % 2 == 1 || b % 2 == 1 {
        return 0.0;
    }
    let d = (a + b) as f64;
    let ang = 2.0 * gamma_half(a / 2) * gamma_half(b / 2) / fact((a + b) / 2);
    let rad = (1.0 - eps.powf(d + 2.0)) / (d + 2.0);
    ang * rad / (std::f64::consts::PI * (1.0 - eps * eps))
}

fn quad(s: &PupilSampling, a: u32, b: u32) -> f64 {
    s.nodes()
        .iter()
        .map(|n| n.weight * n.px.powi(a as i32) * n.py.powi(b as i32))
        .sum()
}

#[test]
fn rings_arms_integrate_polynomials_exactly() {
    for (rings, arms, eps) in [
        (1, 3, 0.0),
        (2, 7, 0.0),
        (3, 11, 0.0),
        (4, 15, 0.3),
        (6, 23, 0.5),
    ] {
        let s = PupilSampling {
            rings,
            arms,
            obscuration: eps,
            arm_offset_deg: 13.0,
            ..Default::default()
        };
        let deg = s.exact_degree() as u32;
        assert_eq!(deg, (4 * rings - 2).min(arms - 1) as u32);
        assert_eq!(s.nodes().len(), rings * arms);
        assert!((s.nodes().iter().map(|n| n.weight).sum::<f64>() - 1.0).abs() < 1e-14);
        for d in 0..=deg {
            for a in 0..=d {
                let b = d - a;
                let q = quad(&s, a, b);
                let e = disk_mean(a, b, eps);
                assert!(
                    (q - e).abs() < 1e-13,
                    "rings {rings} arms {arms} ε {eps}: x^{a} y^{b}: {q} vs {e}"
                );
            }
        }
        // Beyond the limits the rule is *not* exact (so the checks above can
        // fail). Radial: ρ^(4N) = (px² + py²)^(2N) is rotationally symmetric
        // but one degree too high for N-point Gauss–Legendre in u = ρ².
        let mean_rho = |p: i32| -> f64 {
            s.nodes()
                .iter()
                .map(|n| n.weight * (n.px * n.px + n.py * n.py).powi(p / 2))
                .sum()
        };
        let exact_rho = |p: i32| {
            let k = p as f64 / 2.0 + 1.0;
            (1.0 - eps.powf(2.0 * k)) / (k * (1.0 - eps * eps))
        };
        let lim = 4 * rings as i32 - 2;
        assert!((mean_rho(lim) - exact_rho(lim)).abs() < 1e-13);
        assert!((mean_rho(lim + 2) - exact_rho(lim + 2)).abs() > 1e-12);
        // Angular: Re (px + i py)^M = ρ^M cos Mθ has zero mean, but M arms
        // alias it to ρ^M cos(M θ₀) ≠ 0.
        let m = arms as i32;
        let alias: f64 = s
            .nodes()
            .iter()
            .map(|n| n.weight * n.px.hypot(n.py).powi(m) * (m as f64 * n.py.atan2(n.px)).cos())
            .sum();
        assert!(alias.abs() > 1e-3, "{alias}");
    }
}

#[test]
fn bilateral_half_pupil_is_exact_for_even_functions() {
    let s = PupilSampling::bilateral(3, 5);
    assert_eq!(s.symmetry, PupilSymmetry::Bilateral);
    assert!(s.nodes().iter().all(|n| n.px >= 0.0));
    let deg = s.exact_degree() as u32; // min(10, 9) = 9
    for d in 0..=deg {
        for a in (0..=d).step_by(2) {
            let b = d - a;
            let q = quad(&s, a, b);
            assert!((q - disk_mean(a, b, 0.0)).abs() < 1e-13, "x^{a} y^{b}");
        }
    }
}

// ---------- optical problem bridge ----------

/// A cemented doublet plus an air gap: OBJ | 1 | 2 | 3 | STOP | IMG.
fn doublet() -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    let glass = MaterialRef::Model {
        nd: 1.5168,
        vd: 64.2,
        dpgf: 0.0,
    };
    let flint = MaterialRef::Model {
        nd: 1.6200,
        vd: 36.4,
        dpgf: 0.0,
    };
    let mut s1 = Surface::new(10, SurfaceShape::sphere(60.0), 6.0, glass);
    s1.semi_diameter = SemiDiameter::Fixed { value: 12.0 };
    let mut s2 = Surface::new(11, SurfaceShape::sphere(-40.0), 2.5, flint);
    s2.semi_diameter = SemiDiameter::Fixed { value: 12.0 };
    let mut s3 = Surface::new(12, SurfaceShape::sphere(-150.0), 5.0, MaterialRef::Air);
    s3.semi_diameter = SemiDiameter::Fixed { value: 12.0 };
    s.surfaces.insert(1, s1);
    s.surfaces.insert(2, s2);
    s.surfaces.insert(3, s3);
    s.stop = 4;
    s.surfaces[4].thickness = 90.0;
    s.fields.kind = FieldKind::ObjectAngle;
    s.fields.fields = vec![
        Field::new(0.0, 0.0),
        Field::new(0.0, 5.0),
        Field::new(0.0, 7.0),
    ];
    s.validate().unwrap();
    s
}

#[test]
fn model_evaluator_values() {
    let s = doublet();
    let ev = ModelEvaluator;
    let ops = [
        Operand::new(OperandKind::TotalTrack {
            from: None,
            to: None,
        }),
        Operand::new(OperandKind::CentreThickness {
            surface: SurfaceId(10),
        }),
        Operand::new(OperandKind::EdgeThickness {
            surface: SurfaceId(10),
            height: None,
        }),
        Operand::new(OperandKind::MinThickness {
            measure: ThicknessMeasure::Centre,
            medium: MediumFilter::Glass,
            from: None,
            to: None,
        }),
        Operand::new(OperandKind::ParamValue {
            param: ParamPath::Curvature {
                surface: SurfaceId(11),
            },
        }),
        Operand::new(OperandKind::Efl),
    ];
    let refs: Vec<&Operand> = ops.iter().collect();
    let v = ev.evaluate_batch(&[s], &refs, &JobContext::new()).unwrap();
    let v = &v[0];
    assert!((v[0].clone().unwrap() - (6.0 + 2.5 + 5.0 + 90.0)).abs() < 1e-12);
    assert_eq!(v[1].clone().unwrap(), 6.0);
    // Edge at h = 12: 6 − sag(R=60) + sag(R=−40).
    let sag = |r: f64, h: f64| r.signum() * (r.abs() - (r * r - h * h).sqrt());
    let edge = 6.0 - sag(60.0, 12.0) + sag(-40.0, 12.0);
    assert!(
        (v[2].clone().unwrap() - edge).abs() < 1e-12,
        "{:?} vs {edge}",
        v[2]
    );
    assert_eq!(v[3].clone().unwrap(), 2.5);
    assert!((v[4].clone().unwrap() + 1.0 / 40.0).abs() < 1e-15);
    assert_eq!(v[5], Err(OperandError::Unsupported));
}

#[test]
fn optical_problem_solves_geometric_targets() {
    // Variables: the three thicknesses after surfaces 10, 11 and the stop.
    // Targets: total track 100 mm, glass min centre ≥ 3 (constraint), and
    // edge thickness of the crown element = 2.5 mm.
    let s = doublet();
    let vars = vec![
        VariableSpec::bounded(
            ParamPath::Thickness {
                surface: SurfaceId(10),
            },
            0.5,
            20.0,
        ),
        VariableSpec::bounded(
            ParamPath::Thickness {
                surface: SurfaceId(11),
            },
            0.5,
            20.0,
        ),
        VariableSpec::bounded(
            ParamPath::Thickness {
                surface: SurfaceId(1),
            },
            10.0,
            200.0,
        ),
    ];
    let spec = MeritFunctionSpec::new(vec![
        Operand::new(OperandKind::TotalTrack {
            from: None,
            to: None,
        })
        .target(100.0)
        .weight(1.0),
        Operand::new(OperandKind::EdgeThickness {
            surface: SurfaceId(10),
            height: None,
        })
        .target(2.5)
        .weight(4.0),
        Operand::new(OperandKind::MinThickness {
            measure: ThicknessMeasure::Centre,
            medium: MediumFilter::Glass,
            from: None,
            to: None,
        })
        .greater_than(3.0)
        .as_constraint(),
        // Keep the flint as thin as allowed (pull it down; the constraint holds it).
        Operand::new(OperandKind::CentreThickness {
            surface: SurfaceId(11),
        })
        .target(0.0)
        .weight(1e-4),
    ]);
    let reg = OperandRegistry::builtin();
    let p = OpticalProblem::new(s, vars, spec, ModelEvaluator, &reg).unwrap();
    assert_eq!(p.num_residuals(), 3);
    assert_eq!(p.num_constraints(), 1);
    let x0 = p.start_point().unwrap();
    assert_eq!(x0, vec![6.0, 2.5, 90.0]);
    let r = least_squares(&p, &x0, &LsqSettings::default(), &Monitor::default()).unwrap();
    assert!(r.termination.is_converged(), "{:?}", r.termination);
    let sys = p.apply(&r.x).unwrap();
    let sag = |rr: f64, h: f64| rr.signum() * (rr.abs() - (rr * rr - h * h).sqrt());
    let t1 = 2.5 + sag(60.0, 12.0) - sag(-40.0, 12.0);
    assert!((r.x[0] - t1).abs() < 1e-6, "{:?} vs {t1}", r.x);
    assert!((r.x[1] - 3.0).abs() < 1e-5, "{:?}", r.x);
    let track: f64 = sys.surfaces[1..5].iter().map(|s| s.thickness).sum();
    assert!((track - 100.0).abs() < 1e-6);
}

/// An evaluator that fails `Efl` for some candidates, to exercise the
/// failure policies.
struct Flaky;
impl OperandEvaluator for Flaky {
    fn supports(&self, k: &OperandKind) -> bool {
        matches!(k, OperandKind::Efl)
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        _ctx: &JobContext,
    ) -> mklight_core::MkResult<Vec<Vec<OperandValue>>> {
        Ok(systems
            .iter()
            .map(|s| {
                operands
                    .iter()
                    .map(|_| {
                        let t = s.surfaces[1].thickness;
                        if t > 8.0 {
                            Err(OperandError::RayFailure("missed".into()))
                        } else {
                            Ok(10.0 * t)
                        }
                    })
                    .collect()
            })
            .collect())
    }
}

#[test]
fn failure_policies_and_composition() {
    let s = doublet();
    let var = vec![VariableSpec::new(ParamPath::Thickness {
        surface: SurfaceId(10),
    })];
    let ops = vec![
        Operand::new(OperandKind::Efl).target(50.0),
        Operand::new(OperandKind::CentreThickness {
            surface: SurfaceId(11),
        }),
        Operand::new(OperandKind::Combine {
            op: mklight_optim::merit::operand::CombineOp::Quotient,
            args: vec![0, 1],
        })
        .target(0.0)
        .weight(0.0),
    ];
    let ev = || CompositeEvaluator::new().with(ModelEvaluator).with(Flaky);
    let reg = OperandRegistry::builtin();
    let mut spec = MeritFunctionSpec::new(ops);
    spec.on_failure = FailurePolicy::Penalty { value: 7.0 };
    let p = OpticalProblem::new(s.clone(), var.clone(), spec.clone(), ev(), &reg).unwrap();
    let ctx = JobContext::new();
    let e = evaluate_one(&p, &[5.0], &ctx).unwrap();
    assert_eq!(e.status, EvalStatus::Ok);
    assert_eq!(e.residuals, vec![0.0, 2.5, 50.0 / 2.5]);
    let e = evaluate_one(&p, &[9.0], &ctx).unwrap();
    assert_eq!(e.status, EvalStatus::Degraded { failures: 2 });
    assert_eq!(e.residuals, vec![7.0, 2.5, 7.0]);
    spec.on_failure = FailurePolicy::FailCandidate;
    let p = OpticalProblem::new(s.clone(), var.clone(), spec, ev(), &reg).unwrap();
    assert_eq!(
        evaluate_one(&p, &[9.0], &ctx).unwrap().status,
        EvalStatus::Failed
    );
    // Invalid parameter value (negative semi-diameter) → Failed, not a panic.
    let pv = vec![VariableSpec::new(ParamPath::SemiDiameter {
        surface: SurfaceId(10),
    })];
    let p = OpticalProblem::new(
        s.clone(),
        pv,
        MeritFunctionSpec::new(vec![Operand::new(OperandKind::TotalTrack {
            from: None,
            to: None,
        })]),
        ModelEvaluator,
        &reg,
    )
    .unwrap();
    assert_eq!(
        evaluate_one(&p, &[-1.0], &ctx).unwrap().status,
        EvalStatus::Failed
    );
    // An operand nobody evaluates is rejected up front.
    assert!(
        OpticalProblem::new(
            s,
            var,
            MeritFunctionSpec::new(vec![Operand::new(OperandKind::Bfl)]),
            ModelEvaluator,
            &reg
        )
        .is_err()
    );
}

#[test]
fn default_merit_generator() {
    let s = doublet();
    let gen_ = DefaultMeritSpec {
        sampling: PupilSampling {
            rings: 3,
            arms: 6,
            ..Default::default()
        },
        ..Default::default()
    };
    let spec = gen_.generate(&s).unwrap();
    // Symmetric system → half pupil: 3 fields × 3 λ × 18 nodes × 2 axes.
    let rays: Vec<&Operand> = spec
        .operands
        .iter()
        .filter(|o| matches!(o.kind, OperandKind::RayAberration { .. }))
        .collect();
    assert_eq!(rays.len(), 3 * 3 * 18 * 2);
    for o in &rays {
        if let OperandKind::RayAberration { px, .. } = o.kind {
            assert!(px >= 0.0);
        }
    }
    // Weights per axis sum to the overall weight.
    let wy: f64 = rays
        .iter()
        .filter(|o| matches!(o.kind, OperandKind::RayAberration { axis: Axis::Y, .. }))
        .map(|o| o.weight)
        .sum();
    assert!((wy - 1.0).abs() < 1e-12);
    // Thickness boundaries: gaps after 10 (glass), 11 (glass), 12 (air);
    // the stop→image gap is excluded. Fixed SDs → edge boundaries too.
    let bounds: Vec<&Operand> = spec
        .operands
        .iter()
        .filter(|o| o.role == mklight_optim::merit::OperandRole::Constraint)
        .collect();
    // glass: min+max centre + min edge (×2), air: min centre + min edge.
    assert_eq!(bounds.len(), 3 + 3 + 2);
    // Wavefront criterion and an off-axis x field → full pupil.
    let mut s2 = s.clone();
    s2.fields.fields.push(Field::new(3.0, 0.0));
    let w = DefaultMeritSpec {
        criterion: DefaultCriterion::Wavefront {
            reference: Default::default(),
        },
        thickness: None,
        ..Default::default()
    }
    .generate(&s2)
    .unwrap();
    assert_eq!(w.operands.len(), 4 * 3 * 18);
    // JSON round trip of a generated spec.
    let back = MeritFunctionSpec::from_json(&spec.to_json().unwrap()).unwrap();
    assert_eq!(back, spec);
}

#[test]
fn cancellation_inside_optical_problem() {
    let s = doublet();
    let p = OpticalProblem::new(
        s,
        vec![VariableSpec::new(ParamPath::Thickness {
            surface: SurfaceId(10),
        })],
        MeritFunctionSpec::new(vec![
            Operand::new(OperandKind::TotalTrack {
                from: None,
                to: None,
            })
            .target(1.0),
        ]),
        ModelEvaluator,
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    let r = least_squares(&p, &[6.0], &LsqSettings::default(), &Monitor::new(ctx)).unwrap();
    assert_eq!(r.termination, Termination::Cancelled);
}
