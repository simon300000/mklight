//! The GPU-batched evaluator against the CPU f64 reference evaluator:
//! agreement within the CONVENTIONS §7 budget on sample systems (with the
//! CPU-f32 emulation backend, which runs everywhere, and on the Metal
//! device, bit-identical to the emulation, skipped without one), batch
//! independence and determinism, CPU fallback, finite-difference Jacobians
//! and population methods on GPU evaluations.

use std::sync::{Arc, OnceLock};

use mklight_core::model::{OpticalSystem, Placement};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkResult, SurfaceId, samples};
use mklight_gpu::{ComputeBackend, CpuF32Backend};
use mklight_optim::deriv::{CheckSettings, FdScheme, JacobianSettings, check_jacobian, jacobian};
use mklight_optim::global::de::{DeSettings, differential_evolution};
use mklight_optim::linalg::Matrix;
use mklight_optim::merit::default::{DefaultCriterion, DefaultMeritSpec};
use mklight_optim::merit::gpu_eval::gpu_supports;
use mklight_optim::merit::operand::{
    Axis, DistortionMapping, OperandKind, RayQuantity, SpotReference, WavefrontReference,
};
use mklight_optim::merit::{
    GpuEvalSettings, GpuTraceEvaluator, Operand, OperandEvaluator, OperandRegistry, OperandValue,
    OpticalProblem, TraceEvaluator, VariableSpec, WavelengthSelector,
};
use mklight_optim::problem::{
    BatchEvaluation, CandidateBatch, Constraint, Problem, Variable, evaluate_one,
};
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
