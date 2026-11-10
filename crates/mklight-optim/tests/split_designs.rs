//! The hybrid host/device division of the resident evaluator (in one plan
//! per batch and in pipelined sub-batches) against the CPU f64 reference:
//! agreement within the CONVENTIONS §7 budget on the samples (CPU-f32
//! emulation everywhere, Metal when present), rayless operands equal to the
//! reference, host completion of unresolved device rays and learned host
//! routing, equal results with and without pipelining, finite differences,
//! and least squares with speculative Jacobians.

use std::sync::{Arc, OnceLock};

use mklight_core::model::{OpticalSystem, Placement};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, RayAiming, Snapshot, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_optim::deriv::{FdScheme, JacobianSettings, jacobian};
use mklight_optim::gpu::{DEVICE_AIM_FROM, GpuContext, ResidentEvaluator};
use mklight_optim::linalg::Matrix;
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::merit::default::{DefaultCriterion, DefaultMeritSpec};
use mklight_optim::merit::operand::{
    Axis, OperandKind, RayQuantity, SpotReference, WavefrontReference,
};
use mklight_optim::merit::{
    BuiltinEvaluator, Capability, Operand, OperandEvaluator, OperandRegistry, OperandValue,
    OpticalProblem, VariableSpec, WavelengthSelector,
};
use mklight_optim::problem::{CandidateBatch, Problem};
use mklight_optim::{Monitor, Termination};

fn metal() -> Option<Arc<dyn ComputeBackend>> {
    static B: OnceLock<Option<Arc<dyn ComputeBackend>>> = OnceLock::new();
    let b = B
        .get_or_init(|| {
            #[cfg(target_os = "macos")]
            {
                mklight_metal::MetalBackend::new()
                    .ok()
                    .flatten()
                    .map(|b| Arc::new(b) as Arc<dyn ComputeBackend>)
            }
            #[cfg(not(target_os = "macos"))]
            {
                None
            }
        })
        .clone();
    if b.is_none() {
        eprintln!("SKIP: no Metal device");
    }
    b
}

fn backends() -> Vec<(&'static str, Arc<dyn ComputeBackend>)> {
    let mut b: Vec<(&'static str, Arc<dyn ComputeBackend>)> =
        vec![("emulation", Arc::new(CpuF32Backend::new()))];
    if let Some(m) = metal() {
        b.push(("metal", m));
    }
    b
}

/// The hybrid evaluator, every batch in pipelined sub-batches or (`false`)
/// every batch in one plan.
fn evaluator(
    backend: &Arc<dyn ComputeBackend>,
    system: &OpticalSystem,
    pipelined: bool,
) -> ResidentEvaluator {
    ResidentEvaluator::new(
        GpuContext::new(backend.clone(), Snapshot::new(system.clone()).unwrap()),
        BuiltinEvaluator::builtin(),
    )
    .with_pipelining_from(if pipelined { 1 } else { usize::MAX })
}

/// The default spot and wavefront merits with their thickness boundaries
/// (model operands on automatic semi-diameters), first-order operands,
/// aggregates, single rays and real rays on an inner surface.
fn operands(sys: &OpticalSystem) -> Vec<Operand> {
    let last = sys.fields.fields.len() - 1;
    let mut ops = DefaultMeritSpec::default().generate(sys).unwrap().operands;
    ops.extend(
        DefaultMeritSpec {
            thickness: None,
            criterion: DefaultCriterion::Wavefront {
                reference: WavefrontReference::Centroid,
            },
            ..Default::default()
        }
        .generate(sys)
        .unwrap()
        .operands,
    );
    let mut all = Operand::new(OperandKind::RmsSpot {
        reference: SpotReference::Centroid,
        component: None,
        sampling: Default::default(),
    });
    all.wavelength = WavelengthSelector::All;
    ops.push(all);
    ops.push(Operand::new(OperandKind::WavefrontRms {
        reference: WavefrontReference::Centroid,
        sampling: Default::default(),
    }));
    ops.push(
        Operand::new(OperandKind::RayAberration {
            axis: Axis::Y,
            px: 0.3,
            py: -0.9,
            reference: SpotReference::Chief,
            group: None,
        })
        .at_field(last),
    );
    for quantity in [RayQuantity::Y, RayQuantity::M, RayQuantity::OpticalPath] {
        ops.push(
            Operand::new(OperandKind::RealRay {
                quantity,
                surface: Some(sys.surfaces[2].id),
                px: 0.5,
                py: 0.7,
                global: false,
            })
            .at_field(last),
        );
    }
    ops.push(Operand::new(OperandKind::Efl));
    ops.push(Operand::new(OperandKind::Bfl));
    ops.push(Operand::new(OperandKind::PetzvalSum));
    ops
}

/// The CONVENTIONS §7 budget of a ray operand's value.
fn budget(op: &Operand, cpu: f64) -> f64 {
    use OperandKind::*;
    const POS: f64 = 2e-5;
    match &op.kind {
        RayAberration { .. } => 2.0 * POS,
        RealRay { quantity, .. } => match quantity {
            RayQuantity::L | RayQuantity::M | RayQuantity::N => 2e-6,
            _ => POS,
        },
        RmsSpot { .. } => (5e-3 * cpu.abs()).max(POS),
        WavefrontAberration { .. } | WavefrontRms { .. } => 5e-3,
        k => panic!("unexpected operand {}", k.name()),
    }
}

fn rayless(op: &Operand) -> bool {
    matches!(
        op.kind.capability(),
        Capability::Model | Capability::Paraxial
    )
}

/// Largest error as a fraction of the budget; rayless operands must equal
/// the reference exactly (they are computed by the same f64 code).
fn compare(dev: &[Vec<OperandValue>], cpu: &[Vec<OperandValue>], ops: &[Operand]) -> f64 {
    let mut worst = 0.0f64;
    for (d, c) in dev.iter().zip(cpu) {
        for ((d, c), op) in d.iter().zip(c).zip(ops) {
            match (d, c) {
                (Ok(d), Ok(c)) if rayless(op) => {
                    assert_eq!(d.to_bits(), c.to_bits(), "{}: {d} vs {c}", op.kind.name())
                }
                (Ok(d), Ok(c)) => worst = worst.max((d - c).abs() / budget(op, *c)),
                (Err(_), Err(_)) => {}
                _ => panic!("{}: device {d:?}, CPU {c:?}", op.kind.name()),
            }
        }
    }
    worst
}

fn values(
    ev: &dyn OperandEvaluator,
    systems: &[OpticalSystem],
    ops: &[Operand],
) -> Vec<Vec<OperandValue>> {
    let refs: Vec<&Operand> = ops.iter().collect();
    ev.evaluate_batch(systems, &refs, &JobContext::new())
        .unwrap()
}

/// A sample, an x-decentered copy (half pupils traced in full) and bent
/// copies.
fn candidates(sys: &OpticalSystem) -> Vec<OpticalSystem> {
    let mut out = vec![sys.clone()];
    let mut decentered = sys.clone();
    decentered.surfaces[1].placement = Some(Placement {
        decenter_x: 0.02,
        ..Default::default()
    });
    out.push(decentered);
    for f in [1.01, 0.98] {
        let mut bent = sys.clone();
        let c = ParamPath::Curvature {
            surface: sys.surfaces[1].id,
        };
        let v = c.get(sys).unwrap();
        c.set(&mut bent, v * f).unwrap();
        out.push(bent);
    }
    out
}

fn real_aimed(mut sys: OpticalSystem) -> OpticalSystem {
    sys.ray_aiming = RayAiming::Real { tolerance: 1e-10 };
    sys
}

#[test]
fn hybrid_designs_agree_with_the_reference() {
    for (backend_name, backend) in backends() {
        for (name, sys) in samples::all()
            .into_iter()
            .filter(|(_, s)| !samples::has_extended_surfaces(s))
            .chain([("cooke real aiming", real_aimed(samples::cooke_triplet()))])
        {
            let ops = operands(&sys);
            let cands = candidates(&sys);
            let reference = values(&BuiltinEvaluator::builtin(), &cands, &ops);
            // Device aiming with host completion (the `aiming` key's
            // device designs), and host aiming with device tracing.
            for (pipelined, host_aim) in [(false, false), (true, false), (false, true)] {
                let ev = evaluator(&backend, &sys, pipelined).with_host_aim(host_aim);
                let v = values(&ev, &cands, &ops);
                let worst = compare(&v, &reference, &ops);
                eprintln!(
                    "{backend_name} {name} pipelined {pipelined} host aim {host_aim}: worst \
                     budget fraction {worst:.3}"
                );
                assert!(
                    worst <= 1.0,
                    "{backend_name} {name} {pipelined} {host_aim}: {worst}"
                );
                let st = ev.stats();
                assert!(st.device_operands > 0, "{name}: {st:?}");
                assert!(st.fallbacks.is_empty(), "{name}: {st:?}");
            }
        }
    }
}

#[test]
fn pipelining_gives_the_same_values() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(CpuF32Backend::new());
    let sys = real_aimed(samples::double_gauss());
    let ops = operands(&sys);
    let mut cands = candidates(&sys);
    cands.extend(candidates(&samples::double_gauss()));
    let h = values(&evaluator(&backend, &sys, false), &cands, &ops);
    let p = values(&evaluator(&backend, &sys, true), &cands, &ops);
    for (a, b) in h.iter().zip(&p) {
        for (a, b) in a.iter().zip(b) {
            match (a, b) {
                (Ok(a), Ok(b)) => assert_eq!(a.to_bits(), b.to_bits()),
                (Err(_), Err(_)) => {}
                _ => panic!("{a:?} vs {b:?}"),
            }
        }
    }
}

/// The default aiming rule: a small batch aims on the host and a large one
/// of an aspheric system on the device, a spherical system's never; a pinned
/// aiming holds for every batch.
#[test]
fn aiming_rule_by_batch() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(CpuF32Backend::new());
    let rule = evaluator(&backend, &samples::cooke_triplet(), false);
    assert_eq!(rule.device_aim_from(), Some(DEVICE_AIM_FROM));
    for (sys, aspheric) in [
        (real_aimed(samples::petzval_asphere()), true),
        (real_aimed(samples::cooke_triplet()), false),
    ] {
        let ops = operands(&sys);
        let refs: Vec<&Operand> = ops.iter().collect();
        let large = 1 << 24;
        assert!(!rule.aims_on_device(&sys, 1, &refs));
        assert_eq!(rule.aims_on_device(&sys, large, &refs), aspheric);
        let host = evaluator(&backend, &sys, false).with_host_aim(true);
        let device = evaluator(&backend, &sys, false).with_host_aim(false);
        assert!(!host.aims_on_device(&sys, large, &refs));
        assert!(device.aims_on_device(&sys, 1, &refs));
        // The decision reaches the plan: one pinned device batch, one rule
        // batch of a single candidate.
        values(&device, std::slice::from_ref(&sys), &ops);
        let rule = evaluator(&backend, &sys, false);
        values(&rule, std::slice::from_ref(&sys), &ops);
        assert_eq!(device.stats().device_aimed_batches, 1);
        assert_eq!(rule.stats().device_aimed_batches, 0);
    }
}

/// Low device schedule caps leave real-aimed pupil rays unresolved on the
/// device: the host completes each one in f64, later batches route the
/// rays learned as slow to the host before the plan, and when the device
/// leaves most rays to the host the evaluator hands the run to the
/// reference. Every outcome agrees with the reference.
#[test]
fn unresolved_rays_are_completed_and_routed_on_the_host() {
    for (backend_name, backend) in backends() {
        let sys = real_aimed(samples::cooke_triplet());
        let ops = operands(&sys);
        let cands = candidates(&sys);
        let reference = values(&BuiltinEvaluator::builtin(), &cands, &ops);
        for pipelined in [false, true] {
            let mut outcomes = Vec::new();
            for cap in [1, 4, 5, 6, 8] {
                let ev = evaluator(&backend, &sys, pipelined)
                    .with_host_aim(false)
                    .with_device_aim_cap(cap);
                let first = values(&ev, &cands, &ops);
                let second = values(&ev, &cands, &ops);
                let host_rays = ev.context.stats().polished_rays;
                let host_only = ev.stats().fallbacks.keys().any(|k| k.contains("most rays"));
                let worst =
                    compare(&first, &reference, &ops).max(compare(&second, &reference, &ops));
                eprintln!(
                    "{backend_name} pipelined {pipelined} cap {cap}: host rays {host_rays}, host only \
                     {host_only}, worst {worst:.3}"
                );
                assert!(
                    worst <= 1.0,
                    "{backend_name} pipelined {pipelined} cap {cap}: {worst}"
                );
                outcomes.push((cap, host_rays, host_only));
            }
            // One iteration resolves almost nothing: completed, then handed over.
            assert!(outcomes[0].1 > 0 && outcomes[0].2, "{outcomes:?}");
            // The full cap keeps the run on the device.
            assert!(!outcomes[4].2, "{outcomes:?}");
            // Some cap in between routes rays to the host without handing over.
            assert!(
                outcomes.iter().any(|&(_, rays, only)| rays > 0 && !only),
                "{outcomes:?}"
            );
        }
    }
}

fn triplet_problem<E: OperandEvaluator>(sys: &OpticalSystem, ev: E) -> OpticalProblem<E> {
    let spec = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
    .generate(sys)
    .unwrap();
    let vars = [1, 3, 6]
        .iter()
        .map(|&s| {
            VariableSpec::new(ParamPath::Curvature {
                surface: sys.surfaces[s].id,
            })
        })
        .chain([VariableSpec::new(ParamPath::Thickness {
            surface: sys.surfaces[sys.surfaces.len() - 2].id,
        })])
        .collect();
    OpticalProblem::new(sys.clone(), vars, spec, ev, &OperandRegistry::builtin()).unwrap()
}

fn column_error(j: &Matrix, r: &Matrix) -> f64 {
    (0..j.cols)
        .map(|c| {
            let (mut num, mut den) = (0.0f64, 0.0f64);
            for i in 0..j.rows {
                num += (j[(i, c)] - r[(i, c)]).powi(2);
                den += r[(i, c)].powi(2);
            }
            (num / den.max(1e-300)).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Finite differences of the hybrid evaluator (anchored batches, host
/// completion with a low device cap included) at its automatic step stay at
/// the reference's level against f64 central differences (the f32-era
/// `1e-3` step measured 1.8e-4 here).
#[test]
fn hybrid_finite_differences() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(CpuF32Backend::new());
    let sys = real_aimed(samples::cooke_triplet());
    let reference = triplet_problem(&sys, BuiltinEvaluator::builtin());
    let x = reference.start_point().unwrap();
    let ctx = JobContext::new();
    let central = JacobianSettings {
        scheme: FdScheme::Central,
        relative_step: Some(1e-5),
        use_analytic: false,
    };
    let f0 = mklight_optim::problem::evaluate_one(&reference, &x, &ctx).unwrap();
    let r = jacobian(&reference, &x, &f0, &central, &ctx)
        .unwrap()
        .residuals;
    for (cap, host_aim) in [(None, false), (Some(2), false), (None, true)] {
        let mut ev = evaluator(&backend, &sys, true).with_host_aim(host_aim);
        if let Some(cap) = cap {
            ev = ev.with_device_aim_cap(cap);
        }
        let p = triplet_problem(&sys, ev);
        assert!(p.anchored_differences());
        let f0 = mklight_optim::problem::evaluate_one(&p, &x, &ctx).unwrap();
        let j = jacobian(&p, &x, &f0, &JacobianSettings::default(), &ctx)
            .unwrap()
            .residuals;
        let e = column_error(&j, &r);
        eprintln!("cap {cap:?} host aim {host_aim}: largest relative column error {e:.2e}");
        assert!(e < 1e-5, "cap {cap:?} host aim {host_aim}: {e}");
    }
}

/// Least squares on the hybrid evaluator, with and without speculative
/// Jacobians, reaches the reference's merit; on the reference itself a
/// speculative run follows the same iterates in fewer batches.
#[test]
fn least_squares_with_speculative_jacobians() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(CpuF32Backend::new());
    let sys = samples::cooke_triplet();
    let ctx = JobContext::new();
    let settings = |speculative| LsqSettings {
        max_iterations: 8,
        speculative_jacobian: speculative,
        ..Default::default()
    };
    let reference = triplet_problem(&sys, BuiltinEvaluator::builtin());
    let x0 = reference.start_point().unwrap();
    let plain = least_squares(
        &reference,
        &x0,
        &settings(false),
        &Monitor::new(ctx.clone()),
    )
    .unwrap();
    let spec = least_squares(&reference, &x0, &settings(true), &Monitor::new(ctx.clone())).unwrap();
    assert_eq!(plain.x, spec.x, "speculation changes no iterate");
    assert_eq!(plain.merit.to_bits(), spec.merit.to_bits());
    assert_eq!(plain.iterations, spec.iterations);
    eprintln!(
        "reference: {} evaluations plain, {} speculative",
        plain.evaluations, spec.evaluations
    );
    for speculative in [false, true] {
        let p = triplet_problem(&sys, evaluator(&backend, &sys, true));
        let r = least_squares(&p, &x0, &settings(speculative), &Monitor::new(ctx.clone())).unwrap();
        assert_ne!(r.termination, Termination::Cancelled);
        let rel = (r.merit - plain.merit).abs() / plain.merit;
        eprintln!(
            "hybrid speculative={speculative}: merit {:.6e} ({rel:.1e} from the reference)",
            r.merit
        );
        assert!(rel < 0.05, "speculative={speculative}: {rel}");
    }
}

#[test]
fn speculative_batches_hold_the_trial_point_and_the_next_probes() {
    // A batch-counting reference problem: one batch per accepted iteration
    // with speculation, two without.
    use std::sync::atomic::{AtomicUsize, Ordering};
    struct Counting<P>(P, AtomicUsize);
    impl<P: Problem> Problem for Counting<P> {
        fn variables(&self) -> &[mklight_optim::Variable] {
            self.0.variables()
        }
        fn num_residuals(&self) -> usize {
            self.0.num_residuals()
        }
        fn weights(&self) -> &[f64] {
            self.0.weights()
        }
        fn constraints(&self) -> &[mklight_optim::Constraint] {
            self.0.constraints()
        }
        fn evaluate_batch(
            &self,
            x: &CandidateBatch,
            ctx: &JobContext,
        ) -> mklight_core::MkResult<mklight_optim::problem::BatchEvaluation> {
            self.1.fetch_add(1, Ordering::Relaxed);
            self.0.evaluate_batch(x, ctx)
        }
        fn anchored_differences(&self) -> bool {
            self.0.anchored_differences()
        }
        fn fd_relative_step(&self, s: FdScheme) -> Option<f64> {
            self.0.fd_relative_step(s)
        }
    }
    let sys = real_aimed(samples::cooke_triplet());
    let ctx = JobContext::new();
    let mut batches = Vec::new();
    for speculative in [false, true] {
        let p = Counting(
            triplet_problem(&sys, BuiltinEvaluator::builtin()),
            AtomicUsize::new(0),
        );
        let x0 = p.0.start_point().unwrap();
        let s = LsqSettings {
            max_iterations: 6,
            speculative_jacobian: speculative,
            ..Default::default()
        };
        let r = least_squares(&p, &x0, &s, &Monitor::new(ctx.clone())).unwrap();
        batches.push((p.1.load(Ordering::Relaxed), r.iterations, r.merit));
    }
    eprintln!("(batches, iterations, merit) plain / speculative: {batches:?}");
    let (plain, spec) = (batches[0], batches[1]);
    assert!(spec.0 < plain.0, "{batches:?}");
    assert!((spec.2 - plain.2).abs() <= 1e-3 * plain.2, "{batches:?}");
}
