#![cfg(target_os = "macos")]
use mklight_core::{JobContext, Snapshot, param::ParamPath, samples};
use mklight_optim::{deriv::*, gpu::*, merit::operand::*, merit::*, problem::*};
use std::sync::Arc;
// Diagnostic view deliberately bypasses the production reference-FD policy.
struct Raw<'a>(&'a OpticalProblem<ResidentEvaluator>);
impl Problem for Raw<'_> {
    fn variables(&self) -> &[Variable] {
        self.0.variables()
    }
    fn num_residuals(&self) -> usize {
        self.0.num_residuals()
    }
    fn weights(&self) -> &[f64] {
        self.0.weights()
    }
    fn gpu_context(&self) -> Option<&GpuContext> {
        self.0.gpu_context()
    }
    fn evaluate_batch(
        &self,
        x: &CandidateBatch,
        ctx: &JobContext,
    ) -> mklight_core::MkResult<BatchEvaluation> {
        self.0.evaluate_batch(x, ctx)
    }
}
#[test]
#[ignore = "host MTF finite-difference step qualification"]
fn mtf_steps() {
    let b = Arc::new(
        mklight_metal::MetalBackend::new()
            .unwrap()
            .expect("P8 requires host Metal"),
    );
    let system = samples::cooke_triplet();
    let ctx = JobContext::new();
    let reference = BuiltinEvaluator::builtin();
    let gpu = ResidentEvaluator::new(
        GpuContext::new(b, Snapshot::new(system.clone()).unwrap()),
        reference.clone(),
    );
    let spec = MeritFunctionSpec::new(
        [0, 2]
            .into_iter()
            .flat_map(|field| {
                [10., 20., 40.].into_iter().map(move |frequency| {
                    Operand::new(OperandKind::Mtf {
                        frequency,
                        orientation: MtfOrientation::Tangential,
                        method: MtfMethod::Diffraction,
                        defocus_mm: 0.,
                        detector: None,
                        wavelengths: vec![],
                    })
                    .at_field(field)
                })
            })
            .collect(),
    );
    let vars = vec![
        VariableSpec::new(ParamPath::Curvature {
            surface: system.surfaces[1].id,
        }),
        VariableSpec::new(ParamPath::Thickness {
            surface: system.surfaces[system.surfaces.len() - 2].id,
        }),
    ];
    let cpu = OpticalProblem::new(
        system.clone(),
        vars.clone(),
        spec.clone(),
        reference,
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let p =
        OpticalProblem::new(system, vars, spec, gpu.clone(), &OperandRegistry::builtin()).unwrap();
    let x = cpu.start_point().unwrap();
    let f = evaluate_one(&cpu, &x, &ctx).unwrap();
    let fg = evaluate_one(&p, &x, &ctx).unwrap();
    let config = |scheme, step| JacobianSettings {
        scheme,
        relative_step: Some(step),
        use_analytic: false,
    };
    let a = fd_jacobian(&cpu, &x, &f, &config(FdScheme::Central, 3e-5), &ctx).unwrap();
    let b = fd_jacobian(&cpu, &x, &f, &config(FdScheme::Central, 1.5e-5), &ctx).unwrap();
    let rich: Vec<_> = a
        .residuals
        .data
        .iter()
        .zip(&b.residuals.data)
        .map(|(a, b)| (4. * b - a) / 3.)
        .collect();
    for scheme in [FdScheme::Forward, FdScheme::Central] {
        for step in [1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 1e-2] {
            let d = fd_jacobian(&Raw(&p), &x, &fg, &config(scheme, step), &ctx).unwrap();
            let errors: Vec<_> = (0..x.len())
                .map(|j| {
                    let mut err = 0.;
                    let mut norm = 0.;
                    for i in 0..p.num_residuals() {
                        let r = rich[i * x.len() + j];
                        err += (d.residuals[(i, j)] - r).powi(2);
                        norm += r * r;
                    }
                    (err / norm.max(1e-16)).sqrt()
                })
                .collect();
            if scheme == FdScheme::Central && step == 1e-5 {
                assert!(errors.iter().all(|&e| e < 1e-3), "{errors:?}");
            }
            eprintln!(
                "P8_STEP mtf=true scheme={scheme:?} step={step:e} column_errors={errors:?} stats={:?}",
                gpu.stats()
            );
        }
    }
    let got = fd_jacobian(&p, &x, &fg, &config(FdScheme::Forward, 1e-6), &ctx).unwrap();
    let expected = fd_jacobian(&cpu, &x, &f, &config(FdScheme::Forward, 1e-6), &ctx).unwrap();
    for (a, b) in got.residuals.data.iter().zip(&expected.residuals.data) {
        assert!((a - b).abs() < 1e-6 * b.abs().max(1e-8));
    }
    assert!(gpu.context.stats().fallback_count >= 3);
}
