#![cfg(target_os = "macos")]
use mklight_core::{JobContext, Snapshot, param::ParamPath};
use mklight_gpu::ComputeBackend;
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::{
    gpu::{GpuContext, ResidentEvaluator},
    merit::*,
    problem::{CandidateBatch, Problem},
};
use std::sync::Arc;
#[test]
fn host_metal_perturbed_samples() {
    let backend: Arc<dyn ComputeBackend> = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    );
    for (name, system) in mklight_core::samples::all() {
        let mut spec = DefaultMeritSpec {
            thickness: None,
            ..Default::default()
        }
        .generate(&system)
        .unwrap();
        spec.operands.extend(
            DefaultMeritSpec {
                thickness: None,
                criterion: mklight_optim::merit::default::DefaultCriterion::Wavefront {
                    reference: mklight_optim::merit::operand::WavefrontReference::Centroid,
                },
                ..Default::default()
            }
            .generate(&system)
            .unwrap()
            .operands,
        );
        let index = system.surfaces.len() - 2;
        let param = ParamPath::Thickness {
            surface: system.surfaces[index].id,
        };
        let v = param.get(&system).unwrap();
        let evaluator = ResidentEvaluator::new(
            GpuContext::new(backend.clone(), Snapshot::new(system.clone()).unwrap()),
            BuiltinEvaluator::builtin(),
        );
        let p = OpticalProblem::new(
            system,
            vec![VariableSpec::new(param)],
            spec,
            evaluator.clone(),
            &OperandRegistry::builtin(),
        )
        .unwrap();
        let b =
            CandidateBatch::from_points(1, [[v], [v + 0.001], [v - 0.001]].iter().map(|x| &x[..]));
        let e = p.evaluate_batch(&b, &JobContext::new()).unwrap();
        assert_eq!(e.len(), 3);
        let stats = evaluator.stats();
        eprintln!(
            "P8 sample={name} {}",
            serde_json::to_string(&stats).unwrap()
        );
        assert!(stats.fallbacks.is_empty(), "{name}: {stats:?}");
        assert!(
            stats.device_operands > 0,
            "{name}: no actual device operands: {stats:?}"
        );
        assert!(e.residuals.iter().all(|v| v.is_finite()));
        let again = p.evaluate_batch(&b, &JobContext::new()).unwrap();
        assert_eq!(e, again, "{name} replay");
    }
}

#[test]
fn asymmetric_half_pupils_and_cached_cancellation(){
    use mklight_core::MkError;
    let backend:Arc<dyn ComputeBackend>=Arc::new(mklight_metal::MetalBackend::new().unwrap().expect("P8 requires host Metal"));
    let system=mklight_core::samples::cooke_triplet();let surface=system.surfaces[1].id;
    let spec=DefaultMeritSpec{thickness:None,..Default::default()}.generate(&system).unwrap();
    let e=ResidentEvaluator::new(GpuContext::new(backend,Snapshot::new(system.clone()).unwrap()),BuiltinEvaluator::builtin());
    let problem=OpticalProblem::new(system.clone(),vec![VariableSpec::new(ParamPath::DecenterX{surface}),VariableSpec::new(ParamPath::TiltY{surface})],spec,e.clone(),&OperandRegistry::builtin()).unwrap();
    let points=CandidateBatch::from_points(2,[[0.,0.],[0.01,0.],[-0.01,0.02]].iter().map(|v|&v[..]));
    problem.evaluate_batch(&points,&JobContext::new()).unwrap();assert_eq!(e.stats().fallback_operands,0);assert!(e.stats().device_operands>0);
    let ctx=JobContext::new();ctx.cancel.cancel();assert!(matches!(problem.evaluate_batch(&points,&ctx),Err(MkError::Cancelled)));
}
