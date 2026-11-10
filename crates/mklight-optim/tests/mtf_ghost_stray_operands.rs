//! MTF, ghost and stray-light merit operands against the analyses and
//! solvers they wrap, their finite differences and an MTF optimisation.

use mklight_analysis::util::{FieldSelection, WavelengthSelection};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkResult, SurfaceId, samples};
use mklight_optim::deriv::{CheckSettings, FdScheme, JacobianSettings, check_jacobian, jacobian};
use mklight_optim::linalg::Matrix;
use mklight_optim::merit::default::{DefaultCriterion, DefaultMeritSpec, MtfCriterion};
use mklight_optim::merit::operand::{
    GhostFocusMetric, GhostMetric, GhostPair, MtfMethod, MtfOrientation, StrayPathClass,
};
use mklight_optim::merit::*;
use mklight_optim::problem::{
    BatchEvaluation, CandidateBatch, Constraint, Problem, Variable, evaluate_one,
};
use mklight_wave::analysis::{MtfAnalysisMethod, MtfSettings, mtf_analysis_job};
use std::sync::Arc;

fn mtf(f: f64, method: MtfMethod, orientation: MtfOrientation) -> Operand {
    let mut o = Operand::new(OperandKind::Mtf {
        frequency: f,
        orientation,
        method,
        defocus_mm: 0.0,
        detector: None,
        wavelengths: Vec::new(),
    })
    .at_field(1);
    o.wavelength = WavelengthSelector::All;
    o
}

#[test]
fn mtf_matches_wave_and_geometric_analyses() {
    let s = samples::cooke_triplet();
    let tr = TraceEvaluator::builtin().tracer(&s).unwrap();
    let ev = BuiltinEvaluator::builtin();
    let ctx = JobContext::new();
    let settings = MtfSettings {
        fields: FieldSelection::Index { index: 1 },
        wavelengths: WavelengthSelection::All,
        pupil: ev.mtf.pupil,
        method: MtfAnalysisMethod::Fft,
        max_frequency: Some(30.0),
        frequency_samples: 4,
        ..Default::default()
    };
    let wave = mtf_analysis_job(&tr, &settings, &ctx).unwrap();
    let mut ops = Vec::new();
    for f in [0.0, 10.0, 20.0, 30.0] {
        ops.push(mtf(f, MtfMethod::Diffraction, MtfOrientation::Tangential));
        ops.push(mtf(f, MtfMethod::Diffraction, MtfOrientation::Sagittal));
    }
    let vals = ev
        .evaluate_system(&s, &ops.iter().collect::<Vec<_>>(), &ctx)
        .unwrap();
    for i in 0..4 {
        assert!((vals[2 * i].as_ref().unwrap() - wave.fields[0].tangential[i]).abs() < 1e-12);
        assert!((vals[2 * i + 1].as_ref().unwrap() - wave.fields[0].sagittal[i]).abs() < 1e-12);
    }
    assert!(wave.fields[0].tangential[3] > 0.01 && wave.fields[0].tangential[3] < 0.99);
    let mut g = mklight_analysis::energy::GeometricMtfSettings {
        max_frequency: Some(30.0),
        frequency_samples: 2,
        defocus: 0.05,
        ..Default::default()
    };
    g.sampling.fields = FieldSelection::Index { index: 1 };
    let geo = mklight_analysis::energy::geometric_mtf_job(&tr, &g, &ctx).unwrap();
    let mut op = mtf(30.0, MtfMethod::Geometric, MtfOrientation::Average);
    if let OperandKind::Mtf { defocus_mm, .. } = &mut op.kind {
        *defocus_mm = 0.05;
    }
    let v = ev.evaluate_system(&s, &[&op], &ctx).unwrap()[0]
        .clone()
        .unwrap();
    assert!((v - 0.5 * (geo.curves[0].modulation[1] + geo.curves[1].modulation[1])).abs() < 1e-12);

    // Cross-check image-plane defocus and detector against the analysis,
    // including its oblique chief-ray reference-sphere conversion.
    let d = mklight_wave::detector::DetectorSettings::square(0.01);
    let focus = mklight_wave::analysis::through_focus_mtf_analysis_job(
        &tr,
        &mklight_wave::analysis::ThroughFocusSettings {
            fields: FieldSelection::Index { index: 1 },
            wavelengths: WavelengthSelection::Index { index: 1 },
            pupil: ev.mtf.pupil,
            method: MtfAnalysisMethod::Fft,
            frequencies: vec![30.0],
            focus_range_mm: Some(0.1),
            focus_samples: 3,
            detector: Some(d),
            ..Default::default()
        },
        &ctx,
    )
    .unwrap();
    let mut op = mtf(30.0, MtfMethod::Diffraction, MtfOrientation::Tangential).at_wavelength(1);
    if let OperandKind::Mtf {
        defocus_mm,
        detector,
        ..
    } = &mut op.kind
    {
        *defocus_mm = 0.1;
        *detector = Some(d);
    }
    let v = ev.evaluate_system(&s, &[&op], &ctx).unwrap()[0]
        .clone()
        .unwrap();
    assert!((v - focus.fields[0].curves[0].tangential[2]).abs() < 1e-12);
}

struct Fd<'a, P>(&'a P, FdScheme);
impl<P: Problem> Problem for Fd<'_, P> {
    fn variables(&self) -> &[Variable] {
        self.0.variables()
    }
    fn num_residuals(&self) -> usize {
        self.0.num_residuals()
    }
    fn weights(&self) -> &[f64] {
        self.0.weights()
    }
    fn constraints(&self) -> &[Constraint] {
        self.0.constraints()
    }
    fn evaluate_batch(&self, b: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        self.0.evaluate_batch(b, ctx)
    }
    fn jacobian(&self, x: &[f64], ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
        Some((|| {
            let f = evaluate_one(self.0, x, ctx)?;
            let j = jacobian(
                self.0,
                x,
                &f,
                &JacobianSettings {
                    scheme: self.1,
                    ..Default::default()
                },
                ctx,
            )?;
            Ok((j.residuals, j.constraints))
        })())
    }
}

#[test]
fn mtf_finite_differences_pass_richardson_check() {
    let s = samples::cooke_triplet();
    let ops = [MtfOrientation::Tangential, MtfOrientation::Sagittal]
        .into_iter()
        .map(|o| mtf(30.0, MtfMethod::Diffraction, o).greater_than(0.95))
        .collect();
    let p = OpticalProblem::new(
        s,
        vec![
            VariableSpec::new(ParamPath::Curvature {
                surface: SurfaceId(3),
            }),
            VariableSpec::new(ParamPath::Thickness {
                surface: SurfaceId(6),
            }),
        ],
        MeritFunctionSpec::new(ops),
        BuiltinEvaluator::builtin(),
        &OperandRegistry::builtin(),
    )
    .unwrap();
    let x = p.start_point().unwrap();
    for scheme in [FdScheme::Forward, FdScheme::Central] {
        let settings = CheckSettings {
            relative_step: 3e-5,
            rel_tol: if scheme == FdScheme::Central {
                2e-4
            } else {
                2e-3
            },
            abs_floor: 1e-5,
        };
        let r = check_jacobian(&Fd(&p, scheme), &x, &settings, &JobContext::new()).unwrap();
        assert!(r.passed, "{scheme:?}: {:?}", r.mismatches);
    }
}

/// Focus of the Cooke triplet, started 0.1 mm (≈ 1.3 waves of defocus) behind the
/// nominal image, against T/S lower bounds of 0.8 at 30 cycles/mm. The
/// smallest of the six field/direction MTFs (14° field) peaks at ≈ 0.17 near
/// the nominal focus; the field-weighted bounds instead trade it for the
/// axis and settle where it is ≈ 0.06. Worst-field must find the former.
#[test]
fn cooke_mtf_worst_field_optimisation_raises_the_minimum() {
    let run = |worst_field| {
        let mut s = samples::cooke_triplet();
        s.surfaces[6].thickness += 0.1;
        let t = s.surfaces[6].thickness;
        let vars = vec![VariableSpec::bounded(
            ParamPath::Thickness {
                surface: SurfaceId(6),
            },
            t - 0.5,
            t + 0.5,
        )];
        let spec = DefaultMeritSpec {
            criterion: DefaultCriterion::Mtf(MtfCriterion {
                frequencies: vec![30.0],
                target: 0.8,
                method: MtfMethod::Diffraction,
                worst_field,
            }),
            thickness: None,
            ..Default::default()
        }
        .generate_for(&s, &vars)
        .unwrap();
        let p = OpticalProblem::new(
            s,
            vars,
            spec,
            BuiltinEvaluator::builtin(),
            &OperandRegistry::builtin(),
        )
        .unwrap();
        let min = |x: &[f64]| {
            p.operand_values(&CandidateBatch::from_points(1, [x]), &JobContext::new())
                .unwrap()[0]
                .as_ref()
                .unwrap()
                .iter()
                .zip(&p.spec().operands)
                .filter(|(_, o)| matches!(o.kind, OperandKind::Mtf { .. }))
                .map(|(v, _)| *v.as_ref().unwrap())
                .fold(f64::INFINITY, f64::min)
        };
        let x = p.start_point().unwrap();
        let r = mklight_optim::lsq::least_squares(
            &p,
            &x,
            &mklight_optim::lsq::LsqSettings {
                max_iterations: 20,
                ..Default::default()
            },
            &mklight_optim::Monitor::default(),
        )
        .unwrap();
        (min(&x), min(&r.x))
    };
    let (before, worst) = run(true);
    let (_, average) = run(false);
    assert!(
        (before - 0.109).abs() < 0.005 && worst > 0.17 && average < 0.1,
        "minimum MTF {before} -> {worst} (worst field), {average} (field average)"
    );
}

#[test]
fn ghost_operands_match_explicit_solvers() {
    let s = samples::singlet();
    let ev = BuiltinEvaluator::builtin();
    let tr = Arc::new(ev.trace.tracer(&s).unwrap());
    let ctx = JobContext::new();
    let model = mklight_ghost::GhostModel::new(tr, Default::default()).unwrap();
    let path = mklight_ghost::GhostPath::new(vec![2, 1]);
    let focus = mklight_ghost::paraxial::paraxial_ghosts(
        &model,
        &mklight_ghost::paraxial::ParaxialGhostSettings {
            paths: Some(vec![path.clone()]),
            ..Default::default()
        },
    )
    .unwrap();
    let op = Operand::new(OperandKind::GhostFocus {
        pair: Some(GhostPair {
            first: SurfaceId(1),
            second: SurfaceId(2),
        }),
        path: None,
        worst: 1,
        metric: GhostFocusMetric::Distance,
        max_reflections: 2,
    });
    let v = ev.evaluate_system(&s, &[&op], &ctx).unwrap()[0]
        .clone()
        .unwrap();
    assert!((v - focus.ghosts[0].focus_from_sensor.abs()).abs() < 1e-12);
    let mut settings = ev.solvers.ghost.clone();
    settings.paths = Some(vec![path.clone()]);
    settings.wavelengths = vec![s.wavelengths.primary];
    let irr = mklight_ghost::irradiance::ghost_irradiance(&model, &settings, &ctx).unwrap();
    assert!(irr.total_flux_ratio > 0.0);
    for (metric, reference) in [
        (GhostMetric::Power, irr.total_flux_ratio),
        (GhostMetric::Mean, irr.total_flux_ratio),
        (GhostMetric::Peak, irr.total_peak_ratio),
    ] {
        let op = Operand::new(OperandKind::GhostIrradiance {
            pair: None,
            path: Some(path.clone()),
            metric,
            orders: Vec::new(),
            max_reflections: 2,
        });
        let v = ev.evaluate_system(&s, &[&op], &ctx).unwrap()[0]
            .clone()
            .unwrap();
        assert!(
            (v - reference).abs() < 1e-12,
            "{metric:?}: {v} vs {reference}"
        );
    }
}

#[test]
fn worst_ghosts_and_selected_orders_match_solver() {
    let s = samples::cooke_triplet();
    let ev = BuiltinEvaluator::builtin();
    let model =
        mklight_ghost::GhostModel::new(Arc::new(ev.trace.tracer(&s).unwrap()), Default::default())
            .unwrap();
    let report = mklight_ghost::paraxial::paraxial_ghosts(&model, &Default::default()).unwrap();
    assert!(report.ghosts.len() > 2);
    let expected = report
        .ghosts
        .iter()
        .take(2)
        .map(|g| g.focus_from_sensor.abs())
        .fold(f64::INFINITY, f64::min);
    let op = Operand::new(OperandKind::GhostFocus {
        pair: None,
        path: None,
        worst: 2,
        metric: GhostFocusMetric::Distance,
        max_reflections: 2,
    });
    let actual = ev.evaluate_system(&s, &[&op], &JobContext::new()).unwrap()[0]
        .clone()
        .unwrap();
    assert!((actual - expected).abs() < 1e-12);

    let s = samples::singlet();
    let model =
        mklight_ghost::GhostModel::new(Arc::new(ev.trace.tracer(&s).unwrap()), Default::default())
            .unwrap();
    let mut settings = ev.solvers.ghost.clone();
    settings.enumeration.max_reflections = 4;
    let paths = mklight_ghost::path::enumerate(&model, &settings.enumeration)
        .unwrap()
        .paths
        .into_iter()
        .filter(|p| p.path.order() == 4)
        .map(|p| p.path)
        .collect::<Vec<_>>();
    assert!(!paths.is_empty());
    settings.paths = Some(paths);
    settings.wavelengths = vec![s.wavelengths.primary];
    let reference =
        mklight_ghost::irradiance::ghost_irradiance(&model, &settings, &JobContext::new()).unwrap();
    assert!(reference.total_flux_ratio > 0.0);
    let op = Operand::new(OperandKind::GhostIrradiance {
        pair: None,
        path: None,
        metric: GhostMetric::Power,
        orders: vec![4],
        max_reflections: 4,
    });
    let actual = ev.evaluate_system(&s, &[&op], &JobContext::new()).unwrap()[0]
        .clone()
        .unwrap();
    assert!((actual - reference.total_flux_ratio).abs() < 1e-12);
    // Peak focus mode uses real irradiance, rather than the paraxial estimate.
    let mut settings = ev.solvers.ghost.clone();
    settings.paths = Some(vec![mklight_ghost::GhostPath::new(vec![2, 1])]);
    settings.wavelengths = (0..s.wavelengths.wavelengths.len()).collect();
    let reference =
        mklight_ghost::irradiance::ghost_irradiance(&model, &settings, &JobContext::new()).unwrap();
    let mut op = Operand::new(OperandKind::GhostFocus {
        pair: None,
        path: settings.paths.unwrap().pop(),
        worst: 1,
        metric: GhostFocusMetric::Peak,
        max_reflections: 2,
    });
    op.wavelength = WavelengthSelector::All;
    let actual = ev.evaluate_system(&s, &[&op], &JobContext::new()).unwrap()[0]
        .clone()
        .unwrap();
    assert!((actual - reference.ghosts[0].peak_ratio).abs() < 1e-12);
}

#[test]
fn stray_agrees_with_solver_and_uses_common_random_numbers() {
    let s = samples::singlet();
    let mut ev = BuiltinEvaluator::builtin();
    ev.solvers.stray.transport.mode = mklight_stray::SplitMode::Split {
        max_splits: 8,
        rare_floor: 0.1,
    };
    ev.solvers.stray.transport.roulette_threshold = 1e-6;
    let tr = ev.trace.tracer(&s).unwrap();
    let mut rs = tr.system().clone();
    rs.wavelength_weights.fill(0.0);
    rs.wavelength_weights[rs.primary_wavelength] = 1.0;
    let ctx = JobContext::new();
    let mut settings = ev.solvers.stray.clone();
    settings.rays = 2048;
    settings.seed = 42;
    let reference = mklight_stray::analysis::analyze(Arc::new(rs), &settings, &ctx).unwrap();
    let op = Operand::new(OperandKind::StrayLightPower {
        class: StrayPathClass::AllNonPrimary,
        rays: settings.rays,
        seed: settings.seed,
    })
    .at_field(0);
    let values = ev
        .evaluate_batch(&[s.clone(), s.clone()], &[&op], &ctx)
        .unwrap();
    let a = values[0][0].as_ref().unwrap();
    assert!(*a > 0.0);
    assert_eq!(a.to_bits(), values[1][0].as_ref().unwrap().to_bits());
    assert_eq!(a.to_bits(), reference.stray_fraction.unwrap().to_bits());
    let mut changed = s.clone();
    changed.surfaces[1].thickness += 0.01;
    let mut reseeded = op.clone();
    if let OperandKind::StrayLightPower { seed, .. } = &mut reseeded.kind {
        *seed += 1;
    }
    let b = ev
        .evaluate_batch(&[changed, s.clone()], &[&op], &ctx)
        .unwrap();
    assert_eq!(a.to_bits(), b[1][0].as_ref().unwrap().to_bits());
    // Common random numbers: a small prescription change moves the value
    // far less than the Monte-Carlo noise of another seed.
    let step = (b[0][0].as_ref().unwrap() - a).abs();
    let noise = (ev.evaluate_system(&s, &[&reseeded], &ctx).unwrap()[0]
        .as_ref()
        .unwrap()
        - a)
        .abs();
    assert!(step < 0.1 * noise, "CRN step {step}, seed noise {noise}");
}
