//! The GPU-batched evaluator against the CPU f64 reference evaluator:
//! agreement within the CONVENTIONS §7 budget on sample systems (with the
//! CPU-f32 emulation backend, which runs everywhere, and on the Metal
//! device, bit-identical to the emulation, skipped without one), batch
//! independence and determinism, CPU fallback, finite-difference Jacobians
//! and population methods on GPU evaluations.

use std::sync::{Arc, OnceLock};

use mklight_core::job::{Precision, RunTimer};
use mklight_core::model::{OpticalSystem, Placement};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkResult, SurfaceId, samples};
use mklight_gpu::{
    ComputeBackend, CpuF32Backend, DeviceInfo, Job, JobKind, JobResult, PackedSystem, Pending,
    PrecisionDescriptor, SystemHandle,
};
use mklight_optim::deriv::{FdScheme, JacobianSettings, jacobian};
use mklight_optim::global::cmaes::{CmaEsSettings, cma_es};
use mklight_optim::global::de::{DeSettings, differential_evolution};
use mklight_optim::linalg::Matrix;
use mklight_optim::lsq::{LsqSettings, least_squares};
use mklight_optim::merit::default::{DefaultCriterion, DefaultMeritSpec};
use mklight_optim::merit::gpu_eval::gpu_supports;
use mklight_optim::merit::operand::{
    Axis, DistortionMapping, OperandKind, RayQuantity, SpotReference, WavefrontReference,
};
use mklight_optim::merit::{
    GpuEvalSettings, GpuTraceEvaluator, Operand, OperandEvaluator, OperandRegistry, OperandValue,
    OpticalProblem, TraceEvaluator, VariableSpec, WavelengthSelector,
};
use mklight_optim::problem::{Problem, evaluate_one};
use mklight_optim::{Monitor, Termination};
use serde_json::json;

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

fn emulation() -> Arc<dyn ComputeBackend> {
    Arc::new(CpuF32Backend::new())
}

fn gpu_eval(backend: Arc<dyn ComputeBackend>, sys: &OpticalSystem) -> GpuTraceEvaluator {
    GpuTraceEvaluator::new(
        backend,
        TraceEvaluator::builtin(),
        sys,
        GpuEvalSettings::default(),
        &JobContext::new(),
    )
    .unwrap()
}

fn curv(id: u64) -> VariableSpec {
    VariableSpec::new(ParamPath::Curvature {
        surface: SurfaceId(id),
    })
}
fn thick(id: u64) -> VariableSpec {
    VariableSpec::new(ParamPath::Thickness {
        surface: SurfaceId(id),
    })
}

/// The default spot and wavefront merits plus every other accelerated kind:
/// aggregates, single rays without a group, real rays on the image and on
/// an inner surface (local and global), distortion and lateral colour.
fn operands(sys: &OpticalSystem) -> Vec<Operand> {
    let nf = sys.fields.fields.len();
    let last = nf - 1;
    let mut ops = DefaultMeritSpec {
        thickness: None,
        ..Default::default()
    }
    .generate(sys)
    .unwrap()
    .operands;
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
    let rms = |reference, component| OperandKind::RmsSpot {
        reference,
        component,
        sampling: Default::default(),
    };
    let mut all = Operand::new(rms(SpotReference::Centroid, None));
    all.wavelength = WavelengthSelector::All;
    ops.push(all);
    ops.push(Operand::new(rms(SpotReference::Chief, Some(Axis::Y))));
    ops.push(Operand::new(OperandKind::SpotCentroid { axis: Axis::Y }).at_field(last));
    ops.push(Operand::new(OperandKind::WavefrontRms {
        reference: WavefrontReference::Centroid,
        sampling: Default::default(),
    }));
    ops.push(
        Operand::new(OperandKind::WavefrontAberration {
            px: 0.0,
            py: 0.7,
            reference: WavefrontReference::Chief,
            group: None,
        })
        .at_field(last),
    );
    ops.push(
        Operand::new(OperandKind::RayAberration {
            axis: Axis::Y,
            px: 0.3,
            py: -0.9,
            reference: SpotReference::Chief,
            group: None,
        })
        .at_field(last)
        .at_wavelength(0),
    );
    let inner = sys.surfaces[2].id;
    for (surface, global) in [(None, false), (Some(inner), false), (Some(inner), true)] {
        for quantity in [
            RayQuantity::X,
            RayQuantity::Y,
            RayQuantity::Z,
            RayQuantity::L,
            RayQuantity::M,
            RayQuantity::N,
            RayQuantity::AngleX,
            RayQuantity::AngleY,
            RayQuantity::OpticalPath,
        ] {
            ops.push(
                Operand::new(OperandKind::RealRay {
                    quantity,
                    surface,
                    px: 0.5,
                    py: 0.7,
                    global,
                })
                .at_field(last),
            );
        }
    }
    ops.push(
        Operand::new(OperandKind::Distortion {
            mapping: DistortionMapping::FTanTheta,
        })
        .at_field(last),
    );
    ops.push(Operand::new(OperandKind::LateralColour { short: 0, long: 2 }).at_field(last));
    ops
}

/// The CONVENTIONS §7 budget for one operand's value.
fn tolerance(op: &Operand, cpu: f64) -> f64 {
    use OperandKind::*;
    const POS: f64 = 2e-5;
    const DIR: f64 = 2e-6;
    match &op.kind {
        RayAberration { .. } | SpotCentroid { .. } | LateralColour { .. } => 2.0 * POS,
        RealRay { quantity, .. } => match quantity {
            RayQuantity::L | RayQuantity::M | RayQuantity::N => DIR,
            RayQuantity::AngleX | RayQuantity::AngleY => 2.0 * DIR.to_degrees(),
            _ => POS,
        },
        RmsSpot { .. } => (5e-3 * cpu.abs()).max(POS),
        WavefrontAberration { .. } | WavefrontRms { .. } => 5e-3,
        // 100·δh/h of a ≥ 1 mm image height.
        Distortion { .. } => 100.0 * 2.0 * POS,
        k => panic!("unexpected operand {}", k.name()),
    }
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

/// Largest error as a fraction of the operand's budget, per kind.
fn compare(gpu: &[Vec<OperandValue>], cpu: &[Vec<OperandValue>], ops: &[Operand]) -> f64 {
    let mut worst = 0.0f64;
    let mut per_kind = std::collections::BTreeMap::<&str, f64>::new();
    for (g, c) in gpu.iter().zip(cpu) {
        for ((g, c), op) in g.iter().zip(c).zip(ops) {
            match (g, c) {
                (Ok(g), Ok(c)) => {
                    let e = (g - c).abs();
                    let f = e / tolerance(op, *c);
                    worst = worst.max(f);
                    let k = per_kind.entry(op.kind.name()).or_default();
                    *k = k.max(e);
                }
                (Err(_), Err(_)) => {}
                _ => panic!("{}: GPU {g:?}, CPU {c:?}", op.kind.name()),
            }
        }
    }
    eprintln!("   max |GPU − CPU| per kind: {per_kind:?}; worst budget fraction {worst:.3}");
    worst
}

/// Sample systems and perturbed copies: an x-decentered element (asymmetric
/// candidate: half pupils traced in full) and a bent one.
fn candidates(sys: &OpticalSystem) -> Vec<OpticalSystem> {
    let mut decentered = sys.clone();
    decentered.surfaces[1].placement = Some(Placement {
        decenter_x: 0.02,
        ..Default::default()
    });
    decentered.surfaces[2].placement = Some(Placement {
        decenter_x: 0.02,
        ..Default::default()
    });
    let mut bent = sys.clone();
    let c = ParamPath::Curvature {
        surface: sys.surfaces[1].id,
    };
    let v = c.get(sys).unwrap();
    c.set(&mut bent, v * 1.01).unwrap();
    vec![sys.clone(), decentered, bent]
}

fn agreement(backend: Arc<dyn ComputeBackend>) {
    for (name, sys) in [
        ("singlet", samples::singlet()),
        ("cooke_triplet", samples::cooke_triplet()),
        ("double_gauss", samples::double_gauss()),
        ("petzval_asphere", samples::petzval_asphere()),
    ] {
        let gpu = gpu_eval(backend.clone(), &sys);
        assert!(gpu.is_active(), "{name}: {:?}", gpu.qualification().reason);
        let ops = operands(&sys);
        assert!(ops.iter().all(|o| gpu_supports(&o.kind, true)));
        let cands = candidates(&sys);
        let g = values(&gpu, &cands, &ops);
        let c = values(&TraceEvaluator::builtin(), &cands, &ops);
        eprintln!(
            "{name} ({} operands × {} candidates)",
            ops.len(),
            cands.len()
        );
        let worst = compare(&g, &c, &ops);
        assert!(worst <= 1.0, "{name}: {worst} of the budget");
        let st = gpu.stats();
        assert_eq!(st.gpu_candidates, cands.len() as u64);
        assert_eq!(st.cpu_candidates, 0);
        assert!(st.rays > 0 && st.dispatches >= 2);
    }
}

#[test]
fn emulated_gpu_agrees_with_cpu_reference() {
    agreement(emulation());
}

#[test]
fn metal_gpu_agrees_with_cpu_reference_and_emulation() {
    let Some(gpu) = metal() else { return };
    agreement(gpu.clone());
    // Bit-identical to the emulation in the device's f32 environment.
    let emu: Arc<dyn ComputeBackend> = Arc::new(CpuF32Backend::matching(&gpu.precision().probe));
    let sys = samples::double_gauss();
    let ops = operands(&sys);
    let cands = candidates(&sys);
    let a = values(&gpu_eval(gpu, &sys), &cands, &ops);
    let b = values(&gpu_eval(emu, &sys), &cands, &ops);
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        match (x, y) {
            (Ok(x), Ok(y)) => assert_eq!(x.to_bits(), y.to_bits()),
            (x, y) => assert_eq!(x, y),
        }
    }
}

/// A candidate's values do not depend on the batch around it, and two runs
/// agree bit for bit.
#[test]
fn gpu_values_are_deterministic_and_batch_independent() {
    let sys = samples::cooke_triplet();
    let gpu = gpu_eval(emulation(), &sys);
    let ops = operands(&sys);
    let cands = candidates(&sys);
    let alone = values(&gpu, &cands[1..2], &ops);
    let batch = values(&gpu, &cands, &ops);
    let again = values(&gpu, &cands, &ops);
    let bits = |v: &[OperandValue]| -> Vec<Option<u64>> {
        v.iter()
            .map(|x| x.as_ref().ok().map(|x| x.to_bits()))
            .collect()
    };
    assert_eq!(bits(&alone[0]), bits(&batch[1]));
    for (a, b) in batch.iter().zip(&again) {
        assert_eq!(bits(a), bits(b));
    }
    // Split into several dispatches: the same numbers.
    let small = GpuTraceEvaluator::new(
        emulation(),
        TraceEvaluator::builtin(),
        &sys,
        GpuEvalSettings {
            max_batch: 2,
            ..Default::default()
        },
        &JobContext::new(),
    )
    .unwrap();
    let split = values(&small, &cands, &ops);
    for (a, b) in batch.iter().zip(&split) {
        assert_eq!(bits(a), bits(b));
    }
}

/// Triplet problem: 3 curvatures, an air space and the back focus.
fn triplet_problem<E: OperandEvaluator>(criterion: DefaultCriterion, ev: E) -> OpticalProblem<E> {
    let sys = samples::cooke_triplet();
    let vars = vec![curv(1), curv(3), curv(6), thick(2), thick(6)];
    let spec = DefaultMeritSpec {
        thickness: None,
        criterion,
        ..Default::default()
    }
    .generate_for(&sys, &vars)
    .unwrap();
    OpticalProblem::new(sys, vars, spec, ev, &OperandRegistry::builtin()).unwrap()
}

/// Largest relative column error `‖J[:,j] − R[:,j]‖ / ‖R[:,j]‖`.
fn column_error(j: &Matrix, r: &Matrix) -> f64 {
    (0..r.cols)
        .map(|c| {
            let (mut d, mut n) = (0.0, 0.0);
            for i in 0..r.rows {
                d += (j[(i, c)] - r[(i, c)]).powi(2);
                n += r[(i, c)].powi(2);
            }
            (d / n).sqrt()
        })
        .fold(0.0, f64::max)
}

/// Central differences on the f64 reference at `η` (accurate to ~1e-8
/// relative for these operands, see `docs/OPTIMIZATION.md`).
fn reference_jacobian<P: Problem>(p: &P, x: &[f64], eta: f64) -> Matrix {
    let ctx = JobContext::new();
    let f0 = evaluate_one(p, x, &ctx).unwrap();
    let s = JacobianSettings {
        scheme: FdScheme::Central,
        relative_step: Some(eta),
        ..Default::default()
    };
    jacobian(p, x, &f0, &s, &ctx).unwrap().residuals
}

/// Column error of `p`'s finite-difference Jacobian at `x` against `r`.
fn fd_error<P: Problem>(p: &P, x: &[f64], scheme: FdScheme, step: Option<f64>, r: &Matrix) -> f64 {
    let ctx = JobContext::new();
    let f0 = evaluate_one(p, x, &ctx).unwrap();
    let s = JacobianSettings {
        scheme,
        relative_step: step,
        ..Default::default()
    };
    column_error(&jacobian(p, x, &f0, &s, &ctx).unwrap().residuals, r)
}

const SPOT: DefaultCriterion = DefaultCriterion::RmsSpot {
    reference: SpotReference::Centroid,
    scope: mklight_optim::merit::CentroidScope::Monochromatic,
};
const WAVEFRONT: DefaultCriterion = DefaultCriterion::Wavefront {
    reference: WavefrontReference::Centroid,
};

/// Jacobian columns from GPU evaluations: the evaluator's steps keep the
/// f32 noise out of them; the f64 defaults would difference it.
#[test]
fn gpu_finite_differences_use_noise_aware_steps() {
    for (criterion, forward, central) in [(SPOT, 1e-3, 2e-4), (WAVEFRONT, 3e-4, 3e-5)] {
        let cpu = triplet_problem(criterion, TraceEvaluator::builtin());
        let x = cpu.start_point().unwrap();
        let r = reference_jacobian(&cpu, &x, 1e-4);
        let gpu = triplet_problem(criterion, gpu_eval(emulation(), cpu.base()).composite());
        let f = fd_error(&gpu, &x, FdScheme::Forward, None, &r);
        let c = fd_error(&gpu, &x, FdScheme::Central, None, &r);
        let naive = fd_error(&gpu, &x, FdScheme::Forward, Some(f64::EPSILON.sqrt()), &r);
        eprintln!(
            "{criterion:?}: GPU column error forward {f:.2e}, central {c:.2e}, √ε {naive:.2e}"
        );
        assert!(f < forward, "forward {f}");
        assert!(c < central, "central {c}");
        assert!(
            naive > 1e-2,
            "√ε steps should difference the f32 noise: {naive}"
        );
    }
}

/// Wavefront operands on the f64 reference: OPD carries absolute rounding
/// noise, so the problem picks `η = 1e-6` (forward) / `1e-4` (central)
/// instead of `√ε` / `∛ε` when the settings leave the step open.
#[test]
fn wavefront_fd_steps_are_chosen_automatically() {
    let spot = triplet_problem(SPOT, TraceEvaluator::builtin());
    assert_eq!(spot.fd_relative_step(FdScheme::Forward), None);
    let p = triplet_problem(WAVEFRONT, TraceEvaluator::builtin());
    assert_eq!(p.fd_relative_step(FdScheme::Forward), Some(1e-6));
    assert_eq!(p.fd_relative_step(FdScheme::Central), Some(1e-4));
    let x = p.start_point().unwrap();
    let r = reference_jacobian(&p, &x, 3e-5);
    let auto = fd_error(&p, &x, FdScheme::Forward, None, &r);
    let old = fd_error(&p, &x, FdScheme::Forward, Some(f64::EPSILON.sqrt()), &r);
    let central = fd_error(&p, &x, FdScheme::Central, None, &r);
    eprintln!(
        "wavefront FD column error: forward auto {auto:.2e} (√ε {old:.2e}), central {central:.2e}"
    );
    assert!(auto < 5e-6, "{auto}");
    assert!(old > 1e-5, "{old}");
    assert!(central < 1e-6, "{central}");
}

/// The emulation backend without its trace kernel.
struct NoTrace(CpuF32Backend);

impl ComputeBackend for NoTrace {
    fn info(&self) -> &DeviceInfo {
        self.0.info()
    }
    fn precision(&self) -> &PrecisionDescriptor {
        self.0.precision()
    }
    fn supports(&self, kind: JobKind) -> bool {
        kind != JobKind::Trace && self.0.supports(kind)
    }
    fn upload_systems(&self, systems: &[PackedSystem]) -> MkResult<SystemHandle> {
        self.0.upload_systems(systems)
    }
    fn release_systems(&self, handle: SystemHandle) -> MkResult<()> {
        self.0.release_systems(handle)
    }
    fn submit(&self, job: Job, ctx: &JobContext) -> MkResult<Pending<JobResult>> {
        self.0.submit(job, ctx)
    }
}

fn assert_bits_eq(a: &[Vec<OperandValue>], b: &[Vec<OperandValue>]) {
    for (x, y) in a.iter().flatten().zip(b.iter().flatten()) {
        match (x, y) {
            (Ok(x), Ok(y)) => assert_eq!(x.to_bits(), y.to_bits()),
            (x, y) => assert_eq!(x, y),
        }
    }
}

/// A backend that fails qualification is not used: the CPU reference
/// computes every value, and the `RunInfo` says so.
#[test]
fn unqualified_backend_falls_back_to_the_cpu_reference() {
    let sys = samples::cooke_triplet();
    let gpu = gpu_eval(Arc::new(NoTrace(CpuF32Backend::new())), &sys);
    assert!(!gpu.is_active());
    assert_eq!(gpu.precision(), Precision::F64Reference);
    let why = gpu.qualification().reason.unwrap();
    assert!(why.contains("trace kernel"), "{why}");
    let ops = operands(&sys);
    let cands = candidates(&sys);
    assert_bits_eq(
        &values(&gpu, &cands, &ops),
        &values(&TraceEvaluator::builtin(), &cands, &ops),
    );
    assert_eq!(gpu.stats().cpu_candidates, cands.len() as u64);
    assert_eq!(gpu.stats().gpu_candidates, 0);
    // The CPU rule for FD steps applies.
    let spot = &ops[0].kind;
    assert_eq!(gpu.fd_relative_step(spot, FdScheme::Forward), None);
    let mut info = TraceEvaluator::builtin()
        .run_info(RunTimer::start(), &sys, "optimize", &json!({}), None)
        .unwrap();
    gpu.record(&mut info);
    assert_eq!(info.backend, "cpu-f64");
    assert_eq!(info.precision, Precision::F64Reference);
    assert!(
        info.warnings.iter().any(|w| w.contains("not used")),
        "{:?}",
        info.warnings
    );
    assert_eq!(
        info.settings["backend_qualification"]["passed"],
        json!(false)
    );
    assert_eq!(info.settings["gpu_evaluation"]["gpu_candidates"], json!(0));
}

/// A candidate the f32 packer cannot represent (here a conic of 1e-300,
/// which rounds to zero) is evaluated on the CPU; the rest of the batch on
/// the GPU.
#[test]
fn unpackable_candidates_are_evaluated_on_the_cpu() {
    let sys = samples::cooke_triplet();
    let gpu = gpu_eval(emulation(), &sys);
    assert!(gpu.is_active());
    let mut bad = sys.clone();
    ParamPath::Conic {
        surface: sys.surfaces[1].id,
    }
    .set(&mut bad, 1e-300)
    .unwrap();
    let ops = operands(&sys);
    let cands = vec![sys.clone(), bad];
    let g = values(&gpu, &cands, &ops);
    let c = values(&TraceEvaluator::builtin(), &cands, &ops);
    assert_bits_eq(&g[1..], &c[1..]);
    assert!(compare(&g[..1], &c[..1], &ops) <= 1.0);
    let st = gpu.stats();
    assert_eq!((st.gpu_candidates, st.cpu_candidates), (1, 1));
    let mut info = TraceEvaluator::builtin()
        .run_info(RunTimer::start(), &sys, "optimize", &json!({}), None)
        .unwrap();
    gpu.record(&mut info);
    assert_eq!(info.precision, Precision::F32Gpu);
    assert_eq!(info.backend, gpu.backend_id());
}

/// Least squares (GPU Jacobian columns), differential evolution and CMA-ES
/// (GPU populations): seeded runs are reproducible, improve the start, and
/// their results re-evaluate on the f64 reference to the GPU merit.
#[test]
fn optimisers_run_on_gpu_evaluations() {
    let cpu = triplet_problem(SPOT, TraceEvaluator::builtin());
    let gpu = triplet_problem(SPOT, gpu_eval(emulation(), cpu.base()).composite());
    let x0 = cpu.start_point().unwrap();
    let ctx = JobContext::new();
    let merit_of = |p: &OpticalProblem<_>, x: &[f64]| {
        let e = evaluate_one(p, x, &ctx).unwrap();
        mklight_optim::problem::merit(p.weights(), &e.residuals, e.status)
    };
    let m0 = merit_of(&cpu, &x0);
    let check = |name: &str, r: &mklight_optim::OptimResult| {
        let m_cpu = merit_of(&cpu, &r.x);
        eprintln!(
            "{name}: {m0:.4e} -> {:.4e} (f64 reference {m_cpu:.4e}), {} evaluations, {:?}",
            r.merit, r.evaluations, r.termination
        );
        assert!(r.merit <= m0, "{name}: {} vs {m0}", r.merit);
        assert!(
            ((m_cpu - r.merit) / m_cpu).abs() < 1e-3,
            "{name}: {m_cpu} vs {}",
            r.merit
        );
    };

    let lm = LsqSettings {
        max_iterations: 8,
        ..Default::default()
    };
    let a = least_squares(&gpu, &x0, &lm, &Monitor::default()).unwrap();
    let b = least_squares(&gpu, &x0, &lm, &Monitor::default()).unwrap();
    assert_eq!(a.x, b.x);
    check("LM", &a);
    // As good as the same run on the reference.
    let c = least_squares(&cpu, &x0, &lm, &Monitor::default()).unwrap();
    assert!(
        a.merit < 1.05 * c.merit,
        "GPU LM {} vs CPU LM {}",
        a.merit,
        c.merit
    );

    let de = DeSettings {
        population: 12,
        max_generations: 4,
        seed: 7,
        init_spread: 0.05,
        ..Default::default()
    };
    let a = differential_evolution(&gpu, Some(&x0), &de, &Monitor::default()).unwrap();
    let b = differential_evolution(&gpu, Some(&x0), &de, &Monitor::default()).unwrap();
    assert_eq!(a.x, b.x);
    assert_eq!(a.merit.to_bits(), b.merit.to_bits());
    check("DE", &a);
    assert_ne!(a.termination, Termination::Cancelled);

    let cs = CmaEsSettings {
        population: 8,
        max_generations: 6,
        sigma0: 0.05,
        seed: 3,
        ..Default::default()
    };
    let a = cma_es(&gpu, Some(&x0), &cs, &Monitor::default()).unwrap();
    let b = cma_es(&gpu, Some(&x0), &cs, &Monitor::default()).unwrap();
    assert_eq!(a.x, b.x);
    check("CMA-ES", &a);
}
