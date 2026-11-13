#![cfg(target_os = "macos")]
//! The resident evaluator on the host's Metal device against the reference:
//! perturbed candidates of every sample, replay, half pupils of asymmetric
//! candidates, independence of batch neighbours and cancellation.

use crate::common;

use mklight_core::{JobContext, Snapshot, param::ParamPath};
use mklight_optim::merit::default::DefaultMeritSpec;
use mklight_optim::{
    gpu::{GpuContext, ResidentEvaluator},
    merit::*,
    problem::{CandidateBatch, Problem},
};
#[test]
fn host_metal_perturbed_samples() {
    let Some(backend) = common::metal() else {
        return;
    };
    for (name, system) in mklight_core::samples::all() {
        let mut spec = DefaultMeritSpec {
            thickness: None,
            ..Default::default()
        }
        .generate(&system)
        .unwrap();
        let spot_operands = spec.operands.len();
        // Both wavefront references: piston/tilt and the device's piston-only
        // fit (chief) removed from the per-ray OPD.
        for reference in [
            mklight_optim::merit::operand::WavefrontReference::Centroid,
            mklight_optim::merit::operand::WavefrontReference::Chief,
        ] {
            spec.operands.extend(
                DefaultMeritSpec {
                    thickness: None,
                    criterion: mklight_optim::merit::default::DefaultCriterion::Wavefront {
                        reference,
                    },
                    ..Default::default()
                }
                .generate(&system)
                .unwrap()
                .operands,
            );
        }
        let index = system.surfaces.len() - 2;
        let param = ParamPath::Thickness {
            surface: system.surfaces[index].id,
        };
        let v = param.get(&system).unwrap();
        let evaluator = ResidentEvaluator::new(
            GpuContext::new(backend.clone(), Snapshot::new(system.clone()).unwrap()),
            BuiltinEvaluator::builtin(),
        );
        let reference = OpticalProblem::new(
            system.clone(),
            vec![VariableSpec::new(param.clone())],
            spec.clone(),
            BuiltinEvaluator::builtin(),
            &OperandRegistry::builtin(),
        )
        .unwrap();
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
        let expected = reference.evaluate_batch(&b, &JobContext::new()).unwrap();
        let mut spot_max = 0_f64;
        let mut wave_max = 0_f64;
        assert_eq!(e.residuals.len(), expected.residuals.len());
        for row in 0..e.len() {
            for (i, (a, b)) in e
                .residuals(row)
                .iter()
                .zip(expected.residuals(row))
                .enumerate()
            {
                if i < spot_operands {
                    spot_max = spot_max.max((a - b).abs());
                } else {
                    wave_max = wave_max.max((a - b).abs());
                }
            }
        }
        // Qualification publishes nothing beyond 2e-5 mm / 5e-3 waves; the
        // measured maxima are 8.9e-9 mm and 2.7e-8 waves, so gate near them.
        assert!(spot_max < 1e-7, "{name}: {spot_max:e}");
        assert!(wave_max < 3e-7, "{name}: {wave_max:e}");
        eprintln!(
            "accuracy sample={name} spot_max_mm={spot_max:e} wavefront_max_waves={wave_max:e}"
        );
        let stats = evaluator.stats();
        eprintln!("sample={name} {}", serde_json::to_string(&stats).unwrap());
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
fn asymmetric_half_pupils_and_cached_cancellation() {
    use mklight_core::MkError;
    let Some(backend) = common::metal() else {
        return;
    };
    let mut system = mklight_core::samples::cooke_triplet();
    system.ray_aiming = mklight_core::RayAiming::Real { tolerance: 1e-10 };
    let surface = system.surfaces[1].id;
    let spec = DefaultMeritSpec {
        thickness: None,
        sampling: PupilSampling::bilateral(6, 24),
        ..Default::default()
    }
    .generate(&system)
    .unwrap();
    let e = ResidentEvaluator::new(
        GpuContext::new(backend, Snapshot::new(system.clone()).unwrap()),
        BuiltinEvaluator::builtin(),
    );
    let problem = OpticalProblem::new(
        system.clone(),
        vec![
            VariableSpec::new(ParamPath::DecenterX { surface }),
            VariableSpec::new(ParamPath::TiltY { surface }),
        ],
        spec,
        e.clone(),
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let points = CandidateBatch::from_points(
        2,
        [[0., 0.], [0.01, 0.], [-0.01, 0.02]].iter().map(|v| &v[..]),
    );
    let together = problem.evaluate_batch(&points, &JobContext::new()).unwrap();
    for row in 0..points.len() {
        let solo = CandidateBatch::from_points(2, [points.point(row)]);
        let solo = problem.evaluate_batch(&solo, &JobContext::new()).unwrap();
        assert_eq!(
            together.residuals(row),
            solo.residuals(0),
            "aim-table choice must not depend on batch neighbours"
        );
    }
    assert_eq!(e.stats().fallback_operands, 0);
    assert!(e.stats().device_operands > 0);
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(matches!(
        problem.evaluate_batch(&points, &ctx),
        Err(MkError::Cancelled)
    ));
}
