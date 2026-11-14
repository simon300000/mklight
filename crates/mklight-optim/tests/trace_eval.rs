//! Tracer-backed operands: known answers (ideal lenses, thick-lens
//! formulas), cross-checks against `mklight-trace`, ray-group references,
//! merit consistency, failures, determinism and derivative validation.

use mklight_core::model::{
    ApertureSpec, Field, FieldKind, FieldSet, MaterialRef, OpticalSystem, SemiDiameter, Surface,
    SurfaceShape,
};
use mklight_core::param::ParamPath;
use mklight_core::{JobContext, MkResult, SurfaceId, samples};
use mklight_optim::deriv::{CheckSettings, FdScheme, JacobianSettings, check_jacobian, jacobian};
use mklight_optim::linalg::Matrix;
use mklight_optim::merit::default::{DefaultMeritSpec, ThicknessBoundaries};
use mklight_optim::merit::geometric::ModelEvaluator;
use mklight_optim::merit::operand::*;
use mklight_optim::merit::{
    CompositeEvaluator, FailurePolicy, MeritFunctionSpec, Operand, OperandError, OperandEvaluator,
    OperandRegistry, OperandValue, OpticalProblem, PupilSampling, PupilSymmetry, TraceEvaluator,
    VariableSpec,
};
use mklight_optim::problem::{
    BatchEvaluation, CandidateBatch, Constraint, EvalStatus, Problem, Variable, evaluate_one, merit,
};
use mklight_trace::SemiDiameterSettings;

fn eval(sys: &OpticalSystem, ops: &[Operand]) -> Vec<OperandValue> {
    let refs: Vec<&Operand> = ops.iter().collect();
    TraceEvaluator::builtin()
        .evaluate_batch(std::slice::from_ref(sys), &refs, &JobContext::new())
        .unwrap()
        .remove(0)
}

fn ok(v: &OperandValue) -> f64 {
    v.clone()
        .unwrap_or_else(|e| panic!("operand failed: {e:?}"))
}

/// Ideal (paraxial) lens of focal length `f` at the stop, object at
/// infinity, image `f + dz` behind it, angle fields.
fn ideal_lens(f: f64, epd: f64, dz: f64, fields: &[f64]) -> OpticalSystem {
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: epd };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: fields.iter().map(|&y| Field::new(0.0, y)).collect(),
    };
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        Surface::new(
            1,
            SurfaceShape::Paraxial { focal_length: f },
            f + dz,
            MaterialRef::Air,
        ),
        Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s.validate().unwrap();
    s
}

fn at(kind: OperandKind, field: usize, wl: usize) -> Operand {
    Operand::new(kind).at_field(field).at_wavelength(wl)
}

#[test]
fn paraxial_operands_match_thick_lens_formulas_and_first_order() {
    let sys = samples::singlet();
    let ev = TraceEvaluator::builtin();
    let tr = ev.tracer(&sys).unwrap();
    let n = tr.system().surfaces[1].n_after[1];
    let (c1, c2, t) = (1.0 / 60.0, -1.0 / 350.0, 5.0);
    // Thick-lens power and Petzval sum of a singlet in air.
    let phi = (n - 1.0) * (c1 - c2) + (n - 1.0).powi(2) * t * c1 * c2 / n;
    let petzval = (n - 1.0) * (c1 - c2) / n;
    let fo = tr.paraxial().primary();
    let ops = vec![
        Operand::new(OperandKind::Efl),
        Operand::new(OperandKind::Bfl),
        Operand::new(OperandKind::PetzvalSum),
        Operand::new(OperandKind::EntrancePupilDiameter),
        Operand::new(OperandKind::EntrancePupilPosition),
        Operand::new(OperandKind::ExitPupilPosition),
        Operand::new(OperandKind::FNumber {
            kind: FNumberKind::ParaxialImageSpace,
        }),
        Operand::new(OperandKind::ParaxialRay {
            ray: ParaxialRayKind::Marginal,
            quantity: ParaxialQuantity::Height,
            surface: Some(SurfaceId(1)),
        }),
        Operand::new(OperandKind::ParaxialRay {
            ray: ParaxialRayKind::Chief,
            quantity: ParaxialQuantity::Angle,
            surface: Some(SurfaceId(0)),
        }),
        Operand::new(OperandKind::ParaxialRay {
            ray: ParaxialRayKind::Marginal,
            quantity: ParaxialQuantity::Slope,
            surface: None,
        }),
        Operand::new(OperandKind::AxialColour { short: 0, long: 2 }),
        Operand::new(OperandKind::Efl).at_wavelength(0),
        {
            let mut o = Operand::new(OperandKind::Efl);
            o.wavelength = mklight_optim::merit::WavelengthSelector::All;
            o
        },
        Operand::new(OperandKind::Magnification),
    ];
    let v = eval(&sys, &ops);
    assert!(
        (ok(&v[0]) - 1.0 / phi).abs() < 1e-9,
        "{:?} {}",
        v[0],
        1.0 / phi
    );
    assert!((ok(&v[1]) - fo.bfl).abs() < 1e-12);
    assert!((ok(&v[2]) - petzval).abs() < 1e-15);
    assert!((ok(&v[3]) - 20.0).abs() < 1e-12);
    assert!(ok(&v[4]).abs() < 1e-12, "stop at the first surface");
    assert_eq!(ok(&v[5]), fo.exit_pupil_position);
    assert!((ok(&v[6]) - (1.0 / phi) / 20.0).abs() < 1e-9);
    assert!((ok(&v[7]) - 10.0).abs() < 1e-12);
    assert!((ok(&v[8]) - 5.0).abs() < 1e-12);
    assert_eq!(ok(&v[9]), *fo.marginal.u.last().unwrap());
    assert!(
        ok(&v[9]) < -0.09 && ok(&v[9]) > -0.11,
        "≈ −EPD/2f: {:?}",
        v[9]
    );
    let p = tr.paraxial();
    assert_eq!(ok(&v[10]), p.at(0).image_distance - p.at(2).image_distance);
    // A singlet focuses blue light shorter.
    assert!(ok(&v[10]) < -1.0);
    assert_eq!(ok(&v[11]), p.at(0).efl);
    let mean = (p.at(0).efl + p.at(1).efl + p.at(2).efl) / 3.0;
    assert!((ok(&v[12]) - mean).abs() < 1e-12);
    assert_eq!(ok(&v[13]), 0.0);
}

#[test]
fn ideal_lens_known_answers() {
    let (f, epd) = (100.0, 5.0);
    let theta = 20f64.to_radians();
    let sys = ideal_lens(f, epd, 0.0, &[0.0, 10.0, 20.0]);
    let ops = vec![
        at(
            OperandKind::RealRay {
                quantity: RayQuantity::Y,
                surface: None,
                px: 0.0,
                py: 0.0,
                global: false,
            },
            2,
            1,
        ),
        at(
            OperandKind::Distortion {
                mapping: DistortionMapping::FTanTheta,
            },
            2,
            1,
        ),
        at(
            OperandKind::Distortion {
                mapping: DistortionMapping::FTheta,
            },
            2,
            1,
        ),
        at(OperandKind::ChiefRayAngle { surface: None }, 2, 1),
        Operand::new(OperandKind::RelativeIllumination).at_field(2),
        Operand::new(OperandKind::FNumber {
            kind: FNumberKind::RealWorking,
        }),
        Operand::new(OperandKind::RmsSpot {
            reference: SpotReference::Centroid,
            component: None,
            sampling: PupilSampling::default(),
        }),
        Operand::new(OperandKind::WavefrontRms {
            reference: WavefrontReference::Centroid,
            sampling: PupilSampling::default(),
        }),
        at(
            OperandKind::RealRay {
                quantity: RayQuantity::AngleY,
                surface: None,
                px: 0.0,
                py: 0.0,
                global: false,
            },
            2,
            1,
        ),
        Operand::new(OperandKind::RelativeIllumination).at_field(0),
    ];
    let v = eval(&sys, &ops);
    assert!((ok(&v[0]) - f * theta.tan()).abs() < 1e-10, "{:?}", v[0]);
    assert!(
        ok(&v[1]).abs() < 1e-9,
        "ideal lens is free of f·tanθ distortion"
    );
    let ftheta = 100.0 * (theta.tan() / theta - 1.0);
    assert!((ok(&v[2]) - ftheta).abs() < 1e-8, "{:?} {ftheta}", v[2]);
    // Stop at the lens: the chief ray is undeviated, CRA = field angle.
    assert!((ok(&v[3]) - 20.0).abs() < 1e-9);
    assert!((ok(&v[8]) - 20.0).abs() < 1e-9);
    // Natural vignetting: cos⁴θ for an aperture at an ideal lens.
    let cos4 = theta.cos().powi(4);
    assert!((ok(&v[4]) - cos4).abs() < 1e-9 * cos4, "{:?} {cos4}", v[4]);
    assert!((ok(&v[9]) - 1.0).abs() < 1e-12);
    let r = epd / 2.0;
    let fw = (f * f + r * r).sqrt() / (2.0 * r);
    assert!((ok(&v[5]) - fw).abs() < 1e-9 * fw, "{:?} {fw}", v[5]);
    assert!(ok(&v[6]) < 1e-12, "{:?}", v[6]);
    assert!(ok(&v[7]) < 1e-9, "{:?}", v[7]);
}

#[test]
fn defocus_spot_and_wavefront_known_answers() {
    let (f, epd, dz) = (100.0, 5.0, 0.5);
    let sys = ideal_lens(f, epd, dz, &[0.0]);
    let r = epd / 2.0;
    let lambda_mm = 0.587_561_8e-3;
    let rays = |reference| OperandKind::RmsSpot {
        reference,
        component: None,
        sampling: PupilSampling::default(),
    };
    let ops = vec![
        Operand::new(rays(SpotReference::Centroid)),
        Operand::new(rays(SpotReference::Chief)),
        Operand::new(OperandKind::RmsSpot {
            reference: SpotReference::Centroid,
            component: Some(Axis::Y),
            sampling: PupilSampling::default(),
        }),
        Operand::new(OperandKind::WavefrontRms {
            reference: WavefrontReference::Centroid,
            sampling: PupilSampling::default(),
        }),
        at(
            OperandKind::WavefrontAberration {
                px: 0.0,
                py: 1.0,
                reference: WavefrontReference::Chief,
                group: None,
            },
            0,
            1,
        ),
        at(
            OperandKind::RayAberration {
                axis: Axis::Y,
                px: 0.0,
                py: 1.0,
                reference: SpotReference::Chief,
                group: None,
            },
            0,
            1,
        ),
    ];
    let v = eval(&sys, &ops);
    // ε = ρ r δz / f exactly for an ideal lens; RMS over the disk = r δz/(f√2).
    let rms = r * dz / (f * 2f64.sqrt());
    assert!((ok(&v[0]) - rms).abs() < 1e-12 * rms, "{:?} {rms}", v[0]);
    assert!((ok(&v[1]) - rms).abs() < 1e-12 * rms);
    assert!((ok(&v[2]) - rms / 2f64.sqrt()).abs() < 1e-12 * rms);
    assert!((ok(&v[5]).abs() - r * dz / f).abs() < 1e-12);
    // Defocus W(h) ≈ h² δz / (2 f (f + δz)); RMS with piston removed = W(r)/√12.
    let w_edge = r * r * dz / (2.0 * f * (f + dz)) / lambda_mm;
    assert!(
        (ok(&v[4]).abs() - w_edge).abs() < 2e-3 * w_edge,
        "{:?} {w_edge}",
        v[4]
    );
    let w_rms = w_edge / 12f64.sqrt();
    assert!(
        (ok(&v[3]) - w_rms).abs() < 2e-3 * w_rms,
        "{:?} {w_rms}",
        v[3]
    );
}

#[test]
fn image_space_telecentric_lens_has_zero_chief_ray_angle() {
    // Stop at the front focal plane.
    let mut s = ideal_lens(100.0, 5.0, 0.0, &[0.0, 15.0]);
    s.surfaces[1] = Surface::new(1, SurfaceShape::Plane, 100.0, MaterialRef::Air);
    s.surfaces.insert(
        2,
        Surface::new(
            5,
            SurfaceShape::Paraxial {
                focal_length: 100.0,
            },
            100.0,
            MaterialRef::Air,
        ),
    );
    s.validate().unwrap();
    let v = eval(
        &s,
        &[
            at(OperandKind::ChiefRayAngle { surface: None }, 1, 1),
            Operand::new(OperandKind::ExitPupilPosition),
            Operand::new(OperandKind::EntrancePupilPosition),
        ],
    );
    assert!(ok(&v[0]) < 1e-9, "{:?}", v[0]);
    assert!(matches!(v[1], Err(OperandError::Invalid(_))), "{:?}", v[1]);
    assert!((ok(&v[2]) - 0.0).abs() < 1e-9);
}

#[test]
fn real_ray_operands_match_the_tracer() {
    let sys = samples::newtonian_telescope();
    let ev = TraceEvaluator::builtin();
    let tr = ev.tracer(&sys).unwrap();
    let field = tr.field(2).clone();
    let (mut rec, opts) = (Vec::new(), tr.options());
    let r = tr.trace_with(&field, 1, 0.3, -0.7, &opts, Some(&mut rec));
    assert!(r.status.is_ok());
    let q = |quantity, surface: u64, global| {
        at(
            OperandKind::RealRay {
                quantity,
                surface: Some(SurfaceId(surface)),
                px: 0.3,
                py: -0.7,
                global,
            },
            2,
            1,
        )
    };
    use RayQuantity::*;
    let ops = vec![
        q(X, 3, false),
        q(Y, 3, false),
        q(M, 3, false),
        q(Y, 5, true),
        q(Z, 5, true),
        q(OpticalPath, 5, false),
        q(IncidenceAngle, 1, false),
    ];
    let v = eval(&sys, &ops);
    assert_eq!(ok(&v[0]), rec[3].point.x);
    assert_eq!(ok(&v[1]), rec[3].point.y);
    assert_eq!(ok(&v[2]), rec[3].dir.y);
    let g = mklight_trace::trace::to_global(
        tr.system(),
        5,
        &mklight_trace::Ray::new(rec[5].point, rec[5].dir),
    );
    assert_eq!(ok(&v[3]), g.pos.y);
    assert_eq!(ok(&v[4]), g.pos.z);
    // The fold sends the beam along −y: the image is ~400 mm down the tube
    // and 100 mm below the axis.
    assert!(
        (g.pos.y + 100.0).abs() < 1.0 && (g.pos.z + 400.0).abs() < 5.0,
        "{g:?}"
    );
    assert_eq!(ok(&v[5]), rec[5].opl);
    assert_eq!(ok(&v[6]), rec[1].cos_incidence.acos().to_degrees());

    // Lateral colour and spot centroid against direct chief rays.
    let sys = samples::cooke_triplet();
    let tr = ev.tracer(&sys).unwrap();
    let f2 = tr.field(2).clone();
    let (cf, cc) = (tr.chief(&f2, 0), tr.chief(&f2, 2));
    let v = eval(
        &sys,
        &[
            Operand::new(OperandKind::LateralColour { short: 0, long: 2 }).at_field(2),
            Operand::new(OperandKind::SpotCentroid { axis: Axis::X }).at_field(2),
            Operand::new(OperandKind::LateralColour { short: 0, long: 2 }).at_field(0),
        ],
    );
    assert_eq!(ok(&v[0]), cf.ray.pos.y - cc.ray.pos.y);
    assert!(ok(&v[0]).abs() > 1e-4);
    assert!(ok(&v[1]).abs() < 1e-14, "symmetric spot: {:?}", v[1]);
    assert_eq!(ok(&v[2]), 0.0);

    // An arbitrary normalised field point at the edge field's coordinates
    // reproduces the system field (no vignetting factors on either).
    let ray = |field| {
        let mut o = Operand::new(OperandKind::RealRay {
            quantity: RayQuantity::Y,
            surface: None,
            px: 0.2,
            py: 0.9,
            global: false,
        });
        o.field = field;
        o
    };
    use mklight_optim::merit::FieldSelector;
    let v = eval(
        &sys,
        &[
            ray(FieldSelector::Index { index: 2 }),
            ray(FieldSelector::Normalized { hx: 0.0, hy: 1.0 }),
            ray(FieldSelector::Normalized { hx: 0.0, hy: 0.5 }),
        ],
    );
    assert_eq!(ok(&v[0]).to_bits(), ok(&v[1]).to_bits());
    assert!(ok(&v[2]) > 0.0 && ok(&v[2]) < ok(&v[0]));
}

/// Full-pupil sampling whose nodes are the mirrored bilateral half pupil.
fn mirrored(half: &PupilSampling) -> PupilSampling {
    PupilSampling {
        rings: half.rings,
        arms: 2 * half.arms,
        symmetry: PupilSymmetry::None,
        obscuration: half.obscuration,
        arm_offset_deg: -90.0 + 90.0 / half.arms as f64,
    }
}

#[test]
fn ray_groups_bilateral_equals_mirrored_full_pupil() {
    let sys = samples::cooke_triplet();
    let half = PupilSampling::bilateral(4, 5);
    let full = mirrored(&half);
    for scope in [CentroidScope::Monochromatic, CentroidScope::Polychromatic] {
        let spot = |s: PupilSampling| {
            let mut o = Operand::new(OperandKind::RmsSpot {
                reference: SpotReference::Centroid,
                component: None,
                sampling: s,
            })
            .at_field(2);
            if scope == CentroidScope::Polychromatic {
                o.wavelength = mklight_optim::merit::WavelengthSelector::All;
            }
            o
        };
        let ab = |s: PupilSampling, px: f64, py: f64, axis| {
            at(
                OperandKind::RayAberration {
                    axis,
                    px,
                    py,
                    reference: SpotReference::Centroid,
                    group: Some(RayGroup { sampling: s, scope }),
                },
                2,
                0,
            )
        };
        let n = half.nodes()[3];
        let v = eval(
            &sys,
            &[
                spot(half),
                spot(full),
                ab(half, n.px, n.py, Axis::Y),
                ab(full, n.px, n.py, Axis::Y),
                ab(half, n.px, n.py, Axis::X),
                ab(full, n.px, n.py, Axis::X),
            ],
        );
        for k in [0, 2, 4] {
            let (a, b) = (ok(&v[k]), ok(&v[k + 1]));
            assert!(
                (a - b).abs() < 1e-12 * (1.0 + a.abs()),
                "{scope:?} {k}: {a} {b}"
            );
        }
        assert!(ok(&v[0]) > 1e-3, "the edge field has a visible spot");
    }
    // Wavefront: the bilateral piston/tilt equals the mirrored full pupil.
    let wf = |s: PupilSampling| {
        at(
            OperandKind::WavefrontAberration {
                px: half.nodes()[5].px,
                py: half.nodes()[5].py,
                reference: WavefrontReference::Centroid,
                group: Some(RayGroup::new(s)),
            },
            2,
            1,
        )
    };
    let v = eval(&sys, &[wf(half), wf(full)]);
    assert!((ok(&v[0]) - ok(&v[1])).abs() < 1e-10, "{v:?}");
}

/// `Σ w r²` of the generated default merit and the mean squared RMS of the
/// aggregate operand over the same (field, λ) set agree: for the spot about
/// each wavelength's centroid and about the white-light centroid (the
/// default), for the wavefront about the centroid (piston and tilt removed)
/// and about the chief ray (piston removed).
#[test]
fn default_merit_equals_mean_squared_rms() {
    use mklight_optim::merit::default::DefaultCriterion;
    let sys = samples::cooke_triplet();
    let reg = OperandRegistry::builtin();
    assert_eq!(
        DefaultCriterion::default(),
        DefaultCriterion::RmsSpot {
            reference: SpotReference::Centroid,
            scope: CentroidScope::Polychromatic
        }
    );
    let mut merits = Vec::new();
    for (spot, scope, wf_ref) in [
        (true, CentroidScope::Monochromatic, WavefrontReference::Centroid),
        (true, CentroidScope::Polychromatic, WavefrontReference::Centroid),
        (false, CentroidScope::Monochromatic, WavefrontReference::Centroid),
        (false, CentroidScope::Monochromatic, WavefrontReference::Chief),
    ] {
        let gen_ = DefaultMeritSpec {
            criterion: if spot {
                DefaultCriterion::RmsSpot {
                    reference: SpotReference::Centroid,
                    scope,
                }
            } else {
                mklight_optim::merit::default::DefaultCriterion::Wavefront { reference: wf_ref }
            },
            thickness: None,
            ..Default::default()
        };
        let spec = gen_.generate(&sys).unwrap();
        let p = OpticalProblem::new(
            sys.clone(),
            vec![VariableSpec::new(ParamPath::Curvature {
                surface: SurfaceId(1),
            })],
            spec,
            TraceEvaluator::builtin(),
            &reg,
        )
        .unwrap();
        let x0 = p.start_point().unwrap();
        let e = evaluate_one(&p, &x0, &JobContext::new()).unwrap();
        assert_eq!(e.status, EvalStatus::Ok);
        let phi = merit(p.weights(), &e.residuals, e.status);
        merits.push(phi);
        let sampling = PupilSampling {
            symmetry: PupilSymmetry::Bilateral,
            ..Default::default()
        };
        let mut ops = Vec::new();
        for f in 0..3 {
            for w in 0..3 {
                ops.push(at(
                    if spot {
                        OperandKind::RmsSpot {
                            reference: SpotReference::Centroid,
                            component: None,
                            sampling,
                        }
                    } else {
                        OperandKind::WavefrontRms {
                            reference: wf_ref,
                            sampling,
                        }
                    },
                    f,
                    w,
                ));
            }
        }
        if scope == CentroidScope::Polychromatic {
            // About the white-light centroid: the field mean of the squared
            // polychromatic RMS spot radius (equal spectral weights).
            let poly: Vec<Operand> = (0..3)
                .map(|f| {
                    let mut o = ops[3 * f].clone();
                    o.wavelength = mklight_optim::merit::WavelengthSelector::All;
                    o
                })
                .collect();
            let v = eval(&sys, &poly);
            let mean_sq = v.iter().map(|x| ok(x).powi(2)).sum::<f64>() / 3.0;
            assert!(
                (phi - mean_sq).abs() < 1e-12 * mean_sq,
                "white light: {phi} vs {mean_sq}"
            );
            continue;
        }
        let v = eval(&sys, &ops);
        let mean_sq = v.iter().map(|x| ok(x).powi(2)).sum::<f64>() / 9.0;
        assert!(
            (phi - mean_sq).abs() < 1e-12 * mean_sq,
            "spot={spot} {wf_ref:?}: {phi} vs {mean_sq}"
        );
        // All fields and wavelengths at once: the same mean square.
        let mut all = ops[0].clone();
        all.field = mklight_optim::merit::FieldSelector::All;
        if spot {
            // `All` wavelengths is the polychromatic spot, not the mean of
            // monochromatic ones; compare the field average per λ instead.
            all.wavelength = mklight_optim::merit::WavelengthSelector::Index { index: 1 };
            let per_f: f64 = (0..3).map(|f| ok(&v[3 * f + 1]).powi(2)).sum::<f64>() / 3.0;
            let a = ok(&eval(&sys, &[all])[0]);
            assert!((a * a - per_f).abs() < 1e-12 * per_f);
        } else {
            all.wavelength = mklight_optim::merit::WavelengthSelector::All;
            let a = ok(&eval(&sys, &[all])[0]);
            assert!((a * a - mean_sq).abs() < 1e-12 * mean_sq);
        }
    }
    // The white-light centroid adds the lateral colour (each wavelength's
    // centroid offset from the white-light one) to the per-wavelength spots.
    assert!(merits[1] > merits[0], "{merits:?}");
}

#[test]
fn auto_semi_diameter_edge_thickness() {
    let sys = samples::cooke_triplet();
    let ev = TraceEvaluator::builtin();
    let spec = DefaultMeritSpec {
        thickness: Some(ThicknessBoundaries::with_auto_semi_diameters()),
        ..Default::default()
    }
    .generate(&sys)
    .unwrap();
    let edges: Vec<Operand> = spec
        .operands
        .iter()
        .filter(|o| matches!(o.kind, OperandKind::EdgeThickness { .. }))
        .cloned()
        .collect();
    // Three glass elements and two air gaps, one minimum-edge bound each.
    assert_eq!(edges.len(), 5);
    let mut ops = edges.clone();
    ops.push(Operand::new(OperandKind::MinThickness {
        measure: ThicknessMeasure::Edge,
        medium: MediumFilter::Glass,
        from: None,
        to: None,
    }));
    let v = eval(&sys, &ops);
    // Reference: freeze the tracer's semi-diameters, then the model formula.
    let tr = ev.tracer(&sys).unwrap();
    let sd = tr.semi_diameters(&SemiDiameterSettings {
        azimuths: ev.settings().semi_diameter_azimuths,
        clip: true,
    });
    let fixed = mklight_trace::fix_semi_diameters(&sys, &sd, 0.0);
    assert!(
        fixed
            .surfaces
            .iter()
            .skip(1)
            .take(6)
            .all(|s| matches!(s.semi_diameter, SemiDiameter::Fixed { .. }))
    );
    let refs: Vec<&Operand> = ops.iter().collect();
    let model = ModelEvaluator
        .evaluate_batch(std::slice::from_ref(&fixed), &refs, &JobContext::new())
        .unwrap()
        .remove(0);
    assert_eq!(v, model);
    // L2 is a negative lens: its edge is thicker than its centre.
    let l2 = ok(&v[2]);
    assert!(l2 > 1.0, "{l2}");
    let min_glass = ok(&v[5]);
    assert!(min_glass > 0.0 && min_glass < l2);
    // Without the tracer the model evaluator cannot size the edges.
    let raw = ModelEvaluator
        .evaluate_batch(std::slice::from_ref(&sys), &refs, &JobContext::new())
        .unwrap()
        .remove(0);
    assert!(
        raw.iter()
            .all(|r| matches!(r, Err(OperandError::Invalid(_))))
    );
}

#[test]
fn failures_follow_the_policy() {
    // A small fixed aperture on the first surface clips the outer pupil.
    let mut sys = samples::singlet();
    sys.surfaces[1].semi_diameter = SemiDiameter::Fixed { value: 7.0 };
    let edge = |py| {
        at(
            OperandKind::RayAberration {
                axis: Axis::Y,
                px: 0.0,
                py,
                reference: SpotReference::Chief,
                group: None,
            },
            0,
            1,
        )
    };
    let spot = Operand::new(OperandKind::RmsSpot {
        reference: SpotReference::Centroid,
        component: None,
        sampling: PupilSampling::default(),
    });
    let v = eval(&sys, &[edge(1.0), edge(0.5), spot.clone()]);
    assert!(
        matches!(v[0], Err(OperandError::RayFailure(_))),
        "{:?}",
        v[0]
    );
    assert!(v[1].is_ok());
    assert!(ok(&v[2]) > 0.0, "the aggregate keeps the surviving rays");
    let reg = OperandRegistry::builtin();
    let mk = |policy| {
        let mut spec = MeritFunctionSpec::new(vec![edge(1.0), edge(0.5), spot.clone()]);
        spec.on_failure = policy;
        OpticalProblem::new(
            sys.clone(),
            vec![VariableSpec::new(ParamPath::Curvature {
                surface: SurfaceId(1),
            })],
            spec,
            TraceEvaluator::builtin(),
            &reg,
        )
        .unwrap()
    };
    let p = mk(FailurePolicy::Penalty { value: 5.0 });
    let x = p.start_point().unwrap();
    let e = evaluate_one(&p, &x, &JobContext::new()).unwrap();
    assert_eq!(e.status, EvalStatus::Degraded { failures: 1 });
    assert_eq!(e.residuals[0], 5.0);
    let p = mk(FailurePolicy::FailCandidate);
    let e = evaluate_one(&p, &x, &JobContext::new()).unwrap();
    assert_eq!(e.status, EvalStatus::Failed);
    // A system that does not resolve (unknown glass) fails every traced
    // operand but still evaluates model operands.
    let mut bad = samples::singlet();
    bad.surfaces[1].material = MaterialRef::glass("NO-SUCH-GLASS");
    let v = eval(
        &bad,
        &[
            Operand::new(OperandKind::Efl),
            Operand::new(OperandKind::CentreThickness {
                surface: SurfaceId(1),
                to: None,
            }),
        ],
    );
    assert!(matches!(v[0], Err(OperandError::Invalid(_))), "{:?}", v[0]);
    assert_eq!(v[1], Ok(5.0));
}

#[test]
fn batches_are_deterministic_and_match_single_evaluations() {
    let base = samples::cooke_triplet();
    let spec = DefaultMeritSpec {
        thickness: Some(ThicknessBoundaries::with_auto_semi_diameters()),
        ..Default::default()
    }
    .generate(&base)
    .unwrap();
    let mut ops: Vec<Operand> = spec.operands.clone();
    ops.push(Operand::new(OperandKind::Efl));
    ops.push(Operand::new(OperandKind::RelativeIllumination).at_field(2));
    let refs: Vec<&Operand> = ops.iter().collect();
    let systems: Vec<OpticalSystem> = (0..6)
        .map(|k| {
            let mut s = base.clone();
            s.surfaces[3].shape = SurfaceShape::sphere(-22.21328 + 0.05 * k as f64);
            s
        })
        .collect();
    let ev = TraceEvaluator::builtin();
    let ctx = JobContext::new();
    let a = ev.evaluate_batch(&systems, &refs, &ctx).unwrap();
    let b = ev.evaluate_batch(&systems, &refs, &ctx).unwrap();
    assert_eq!(a, b);
    for (k, s) in systems.iter().enumerate() {
        let one = ev.evaluate_system(s, &refs, &ctx).unwrap();
        for (x, y) in one.iter().zip(&a[k]) {
            match (x, y) {
                (Ok(x), Ok(y)) => assert_eq!(x.to_bits(), y.to_bits()),
                _ => assert_eq!(x, y),
            }
        }
    }
    // Different candidates give different values.
    assert_ne!(a[0], a[5]);
    // Cancellation is reported as an error.
    let c = JobContext::new();
    c.cancel.cancel();
    assert!(ev.evaluate_batch(&systems, &refs, &c).is_err());
}

#[test]
fn tolerance_sensitivity_operand() {
    let sys = samples::cooke_triplet();
    let param = ParamPath::Thickness {
        surface: SurfaceId(2),
    };
    let crit = OperandKind::Efl;
    let v = eval(
        &sys,
        &[Operand::new(OperandKind::ToleranceSensitivity {
            param: param.clone(),
            delta: 0.1,
            criterion: Box::new(crit.clone()),
        })],
    );
    let efl = |dt: f64| {
        let mut s = sys.clone();
        s.surfaces[2].thickness += dt;
        ok(&eval(&s, &[Operand::new(crit.clone())])[0])
    };
    let (e0, ep, em) = (efl(0.0), efl(0.1), efl(-0.1));
    let want = (((ep - e0).powi(2) + (em - e0).powi(2)) / 2.0).sqrt();
    assert!((ok(&v[0]) - want).abs() < 1e-12, "{:?} {want}", v[0]);
    assert!(want > 1e-3);
}

/// A stand-in for an MTF evaluator living in another crate.
struct ConstantMtf;
impl OperandEvaluator for ConstantMtf {
    fn supports(&self, kind: &OperandKind) -> bool {
        matches!(kind, OperandKind::Mtf { .. })
    }
    fn evaluate_batch(
        &self,
        systems: &[OpticalSystem],
        operands: &[&Operand],
        _ctx: &JobContext,
    ) -> MkResult<Vec<Vec<OperandValue>>> {
        Ok(vec![vec![Ok(0.5); operands.len()]; systems.len()])
    }
}

#[test]
fn plug_in_seam_for_mtf_ghost_and_stray_operands() {
    let ev = TraceEvaluator::builtin();
    let mtf = OperandKind::Mtf {
        frequency: 30.0,
        orientation: MtfOrientation::Tangential,
        method: MtfMethod::Diffraction,
        defocus_mm: 0.0,
        detector: None,
        wavelengths: Vec::new(),
    };
    let reg = OperandRegistry::builtin();
    for d in &reg.kinds {
        let expect = !matches!(
            d.capability,
            Capability::Mtf
                | Capability::Ghost
                | Capability::StrayLight
                | Capability::Composite
                | Capability::Plugin
        );
        assert_eq!(ev.supports(&d.template), expect, "{}", d.name);
    }
    let comp = CompositeEvaluator::new()
        .with(TraceEvaluator::builtin())
        .with(ConstantMtf);
    let sys = samples::cooke_triplet();
    let spec = MeritFunctionSpec::new(vec![
        Operand::new(OperandKind::Efl).target(50.0),
        Operand::new(mtf).at_field(1).greater_than(0.6),
    ]);
    let p = OpticalProblem::new(
        sys,
        vec![VariableSpec::new(ParamPath::Curvature {
            surface: SurfaceId(1),
        })],
        spec,
        comp,
        &reg,
    )
    .unwrap();
    let e = evaluate_one(&p, &p.start_point().unwrap(), &JobContext::new()).unwrap();
    assert_eq!(e.status, EvalStatus::Ok);
    assert!((e.residuals[1] + 0.1).abs() < 1e-15);
}

/// Exposes the optimiser's finite-difference Jacobian through the analytic
/// hook, so [`check_jacobian`] validates it against Richardson-extrapolated
/// central differences.
struct FdAsAnalytic<'a, P: Problem>(&'a P, FdScheme, Option<f64>);

impl<P: Problem> Problem for FdAsAnalytic<'_, P> {
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
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        self.0.evaluate_batch(x, ctx)
    }
    fn jacobian(&self, x: &[f64], ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
        Some((|| {
            let f0 = evaluate_one(self.0, x, ctx)?;
            let s = JacobianSettings {
                scheme: self.1,
                relative_step: self.2,
                ..Default::default()
            };
            let j = jacobian(self.0, x, &f0, &s, ctx)?;
            Ok((j.residuals, j.constraints))
        })())
    }
}

/// Analytic `∂EFL/∂c₁`, `∂EFL/∂c₂` of a thick singlet in air.
struct SingletEfl<P: Problem> {
    inner: P,
    n: f64,
    t: f64,
}

impl<P: Problem> Problem for SingletEfl<P> {
    fn variables(&self) -> &[Variable] {
        self.inner.variables()
    }
    fn num_residuals(&self) -> usize {
        self.inner.num_residuals()
    }
    fn weights(&self) -> &[f64] {
        self.inner.weights()
    }
    fn evaluate_batch(&self, x: &CandidateBatch, ctx: &JobContext) -> MkResult<BatchEvaluation> {
        self.inner.evaluate_batch(x, ctx)
    }
    fn jacobian(&self, x: &[f64], _ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
        let (n, t) = (self.n, self.t);
        let (c1, c2) = (x[0], x[1]);
        let k = (n - 1.0).powi(2) * t / n;
        let phi = (n - 1.0) * (c1 - c2) + k * c1 * c2;
        let d1 = -((n - 1.0) + k * c2) / (phi * phi);
        let d2 = -(-(n - 1.0) + k * c1) / (phi * phi);
        Some(Ok((
            Matrix::from_row_major(1, 2, vec![d1, d2]),
            Matrix::zeros(0, 2),
        )))
    }
}

#[test]
fn derivatives_of_real_optical_operands_validate() {
    let reg = OperandRegistry::builtin();
    let ctx = JobContext::new();
    // Analytic thick-lens EFL derivatives vs the traced EFL operand.
    let sys = samples::singlet();
    let n = TraceEvaluator::builtin()
        .tracer(&sys)
        .unwrap()
        .system()
        .surfaces[1]
        .n_after[1];
    let curv = |id| {
        VariableSpec::new(ParamPath::Curvature {
            surface: SurfaceId(id),
        })
    };
    let p = OpticalProblem::new(
        sys.clone(),
        vec![curv(1), curv(2)],
        MeritFunctionSpec::new(vec![Operand::new(OperandKind::Efl).target(100.0)]),
        TraceEvaluator::builtin(),
        &reg,
    )
    .unwrap();
    let x = p.start_point().unwrap();
    let chk = check_jacobian(
        &SingletEfl {
            inner: p,
            n,
            t: 5.0,
        },
        &x,
        &CheckSettings::default(),
        &ctx,
    )
    .unwrap();
    assert!(chk.passed, "{chk:?}");

    // FD Jacobians of real-ray and wavefront operands on the Cooke triplet:
    // curvatures, an air space and the back focus, with an EFL constraint.
    let sys = samples::cooke_triplet();
    let gen_ = DefaultMeritSpec {
        thickness: None,
        fields: Some(vec![1, 2]),
        ..Default::default()
    };
    let vars = vec![
        curv(1),
        curv(3),
        curv(6),
        VariableSpec::new(ParamPath::Thickness {
            surface: SurfaceId(2),
        }),
        VariableSpec::new(ParamPath::Thickness {
            surface: SurfaceId(6),
        }),
    ];
    // OPD carries an absolute rounding noise of ~1e-14 mm (path lengths of
    // ~100 mm), i.e. ~2e-11 waves, so wavefront operands need larger steps
    // than the √ε / ∛ε defaults (docs/OPTIMIZATION.md).
    for (criterion, central_step, forward_step) in [
        (
            mklight_optim::merit::default::DefaultCriterion::default(),
            None,
            None,
        ),
        (
            mklight_optim::merit::default::DefaultCriterion::Wavefront {
                reference: WavefrontReference::Centroid,
            },
            Some(1e-4),
            Some(1e-6),
        ),
    ] {
        let mut spec = DefaultMeritSpec {
            criterion,
            ..gen_.clone()
        }
        .generate_for(&sys, &vars)
        .unwrap();
        spec.operands.extend([
            Operand::new(OperandKind::Efl).target(50.0).as_constraint(),
            Operand::new(OperandKind::Distortion {
                mapping: DistortionMapping::FTanTheta,
            })
            .at_field(2),
            Operand::new(OperandKind::ChiefRayAngle { surface: None }).at_field(2),
        ]);
        let p = OpticalProblem::new(
            sys.clone(),
            vars.clone(),
            spec,
            TraceEvaluator::builtin(),
            &reg,
        )
        .unwrap();
        let x = p.start_point().unwrap();
        let settings = CheckSettings {
            rel_tol: 1e-5,
            abs_floor: 1e-6,
            ..Default::default()
        };
        let chk = check_jacobian(
            &FdAsAnalytic(&p, FdScheme::Central, central_step),
            &x,
            &settings,
            &ctx,
        )
        .unwrap();
        assert!(
            chk.passed,
            "central: {:?}",
            &chk.mismatches[..3.min(chk.mismatches.len())]
        );
        let loose = CheckSettings {
            rel_tol: 1e-3,
            abs_floor: 1e-5,
            ..Default::default()
        };
        let chk = check_jacobian(
            &FdAsAnalytic(&p, FdScheme::Forward, forward_step),
            &x,
            &loose,
            &ctx,
        )
        .unwrap();
        assert!(
            chk.passed,
            "forward: {:?}",
            &chk.mismatches[..3.min(chk.mismatches.len())]
        );
        // A wrong Jacobian is caught.
        struct Wrong<'a, P: Problem>(&'a P);
        impl<P: Problem> Problem for Wrong<'_, P> {
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
            fn evaluate_batch(
                &self,
                x: &CandidateBatch,
                ctx: &JobContext,
            ) -> MkResult<BatchEvaluation> {
                self.0.evaluate_batch(x, ctx)
            }
            fn jacobian(&self, x: &[f64], ctx: &JobContext) -> Option<MkResult<(Matrix, Matrix)>> {
                let (mut a, b) = FdAsAnalytic(self.0, FdScheme::Central, None)
                    .jacobian(x, ctx)?
                    .ok()?;
                let r = a.row_mut(0);
                r[0] += 1e-3 * (1.0 + r[0].abs());
                Some(Ok((a, b)))
            }
        }
        let chk = check_jacobian(&Wrong(&p), &x, &settings, &ctx).unwrap();
        assert!(!chk.passed);
    }
}
