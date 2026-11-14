#![allow(dead_code)]
use mklight_core::{
    JobContext, Snapshot,
    materials::{MaterialLibrary, coating::ResolvedCoating},
    model::{CoatingDef, Environment, Layer, MaterialRef},
    samples,
};
use mklight_gpu::{radiometry::*, resident::*, *};
/// Regression bounds for the double-float path, far inside the GPU.md §9.4
/// budget (Jones 1e-4 relative, intensity/D 1e-4, retardance 1e-3 rad, ledger
/// 1e-5): observed maxima on every sample and fixture are about 1e-11.
const FIELD_TOL: f64 = 1e-9;
/// Retardance leaves double-float through an f32 atan2 (about 1e-7 relative).
const RETARDANCE_TOL: f64 = 1e-6;
const LEDGER_TOL: f64 = 1e-10;
pub fn upload<T: bytemuck::Pod>(b: &dyn ComputeBackend, p: &mut ResidentPlan, v: &[T]) -> PlanView {
    let h = b
        .fill_buffer(record_desc::<T>(v.len()), &mut |mut f| {
            bytemuck::cast_slice_mut::<u32, T>(f.as_mut_slice::<u32>()?).copy_from_slice(v);
            Ok(())
        })
        .unwrap();
    let id = p.import(h.view());
    p.view(id).unwrap()
}
pub fn plan(
    b: &dyn ComputeBackend,
    specs: &[CoatingSpec],
    layers: &[CoatingLayer],
    queries: &[CoatingQuery],
    tile: usize,
) -> ResidentPlan {
    let mut p = ResidentPlan::new(Snapshot::new(samples::singlet()).unwrap());
    p.settings = serde_json::json!({"package":"P7","version":consts::MKL_RAD_VERSION,"evaluation":"direct-double-float"});
    let specs = upload(b, &mut p, specs);
    let layers = upload(b, &mut p, layers);
    let queries = upload(b, &mut p, queries);
    let output = p.scratch(record_desc::<CoatingValue>(queries.desc.shape[0]));
    p.operations
        .push(ResidentOp::Radiometry(Request::Coating(CoatingRequest {
            specs,
            layers,
            queries,
            output,
            tile,
        })));
    p.outputs = vec![output];
    p
}
pub fn run(b: &dyn ComputeBackend, p: ResidentPlan) -> (Vec<CoatingValue>, PlanResult) {
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let view = b.read_view(r.outputs[0].view()).unwrap();
    let v = bytemuck::cast_slice::<u32, CoatingValue>(view.words()).to_vec();
    (v, r)
}
pub fn coatings() -> Vec<ResolvedCoating> {
    let layer = |n, k, q| Layer {
        material: MaterialRef::Constant { n },
        extinction: k,
        thickness_um: 0.0,
        quarter_waves: Some(q),
    };
    let defs = [
        None,
        Some(CoatingDef::Ideal {
            reflectance: 0.2,
            transmittance: 0.7,
        }),
        Some(CoatingDef::Tabulated {
            reflectance: vec![(0.4, 0.01), (0.55, 0.03), (0.7, 0.02)],
        }),
        Some(CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![layer(1.3, 0., 1.)],
            mirror_substrate: None,
            substrate: Default::default(),
        }),
        Some(CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![layer(2.1, 0.2, 0.6), layer(1.38, 0., 1.)],
            mirror_substrate: Some((0.96, 6.69)),
            substrate: Default::default(),
        }),
    ];
    defs.iter()
        .map(|d| {
            ResolvedCoating::resolve(
                d.as_ref(),
                &MaterialLibrary::with_builtin(),
                &Environment::default(),
            )
            .unwrap()
        })
        .collect()
}
pub fn check_coatings(b: &dyn ComputeBackend) {
    let cs = coatings();
    let mut layers = vec![];
    let mut specs = vec![];
    let mut queries = vec![];
    let mut refs = vec![];
    for c in &cs {
        for (n1, n2) in [(1., 1.69), (1.52, 1.), (1.52, 1.520000001)] {
            for mirror in [false, true] {
                for wl in [0.4, 0.55, 0.7] {
                    let id = specs.len() as u32;
                    // Standalone coatings keep the substrate behind the surface, so
                    // light from the denser side arrives from it.
                    let side = mklight_core::materials::coating::Arrival::forward(n2 >= n1);
                    specs.push(
                        CoatingSpec::resolve(c, wl, n1, n2, mirror, side, &mut layers).unwrap(),
                    );
                    let mut angles = vec![
                        0.0,
                        1e-12,
                        1e-9,
                        1e-7,
                        1e-5,
                        0.001,
                        0.1,
                        0.5,
                        0.9,
                        1.0,
                        (n2 / n1).atan().cos(),
                    ];
                    if n1 > n2 {
                        let critical = (1.0 - (n2 / n1) * (n2 / n1)).sqrt();
                        angles.extend([
                            critical - 1e-7,
                            critical + 1e-7,
                            critical,
                            critical - 1e-12,
                            critical + 1e-12,
                        ]);
                    }
                    for ci in angles {
                        queries.push(CoatingQuery::new(id, ci).unwrap());
                        refs.push((
                            c.amplitudes(wl, n1, n2, ci, mirror, side),
                            c.rt(wl, n1, n2, ci, mirror, side),
                        ));
                    }
                }
            }
        }
    }
    let (values, r) = run(b, plan(b, &specs, &layers, &queries, 1024));
    let mut max_power = 0.0f64;
    let mut max_budget_fraction = 0.0f64;
    let mut max_amp = 0.0f64;
    let mut max_ledger = 0.0f64;
    for (i, (v, (a, p))) in values.iter().zip(refs).enumerate() {
        if v.status == consts::MKL_RAD_REFERENCE {
            let q = queries[i];
            let s = specs[q.coating as usize];
            let mu = s.n_in.to_f64() / s.n_out.to_f64();
            let ci = q.cosine.to_f64();
            assert!(ci < 1e-4 || (mu * mu * (1. - ci) * (1. + ci) - 1.).abs() < 1e-8);
            continue;
        }
        assert_eq!(v.status, 0, "query {i}");
        for (k, z) in [a.rs, a.rp, a.ts_flux(), a.tp_flux()].iter().enumerate() {
            let dr = v.amplitudes[2 * k].to_f64() - z.re;
            let di = v.amplitudes[2 * k + 1].to_f64() - z.im;
            let err = dr.hypot(di);
            max_amp = max_amp.max(err);
            assert!(
                err <= 1e-4 * z.norm() + 1e-7,
                "amplitude {i}:{k} err={err:e} {v:?} ref={a:?}"
            );
        }
        for (k, refp) in [p.rs, p.rp, p.ts, p.tp].into_iter().enumerate() {
            let err = (v.powers[k].to_f64() - refp).abs();
            max_power = max_power.max(err);
            max_budget_fraction = max_budget_fraction.max(err / (1e-5 * refp.abs() + 1e-10));
            assert!(
                err <= 1e-5 * refp.abs() + 1e-10,
                "power {i}:{k} err={err:e} ref={refp:e} {v:?}"
            );
        }
        for pol in 0..2 {
            let err =
                (v.powers[pol].to_f64() + v.powers[pol + 2].to_f64() + v.powers[pol + 4].to_f64()
                    - 1.)
                    .abs();
            max_ledger = max_ledger.max(err);
            assert!(err <= 1e-6, "ledger {i}:{pol} {err:e}");
        }
    }
    assert!(
        max_budget_fraction <= 1.0 / 3.0,
        "coating sampled safety margin {max_budget_fraction}"
    );
    assert_eq!(
        r.stats.fallback_count,
        values
            .iter()
            .filter(|v| v.status == consts::MKL_RAD_REFERENCE)
            .count() as u64
    );
    assert_eq!(r.stats.copied_bytes, 0);
    eprintln!(
        "P7 coating count={} max_power={max_power:e} max_amplitude={max_amp:e} max_ledger={max_ledger:e} max_budget_fraction={max_budget_fraction:e} stats={:?}",
        values.len(),
        r.stats
    );
}
use crate::p1_support as p1;
pub fn transport_plan(
    b: &dyn QualificationBackend,
    t: &mklight_trace::SequentialTracer,
    field: usize,
    wl: usize,
    points: &[[f64; 2]],
    tile: usize,
) -> (ResidentPlan, SystemHandle) {
    let (mut p, h) = p1::make_plan(b, t, field, wl, points, true, 40, false);
    let hit = p
        .operations
        .iter()
        .find_map(|op| {
            if let ResidentOp::Aiming(aiming::Request::TraceResident(t)) = op {
                Some(t.output.clone())
            } else {
                None
            }
        })
        .unwrap();
    let mut ls = vec![];
    let ss = surface_specs(t.system(), &mut ls).unwrap();
    let surfaces = upload(b, &mut p, &ss);
    let layers = upload(b, &mut p, &ls);
    let output = p.scratch(record_desc::<PolarizationValue>(points.len()));
    p.operations
        .push(ResidentOp::Radiometry(Request::Polarization(
            PolarizationRequest {
                scalar: false,
                surfaces,
                layers,
                records: p.view(hit.records.unwrap()).unwrap(),
                record_low: p.view(hit.record_low.unwrap()).unwrap(),
                indices: p.view(hit.rays.indices).unwrap(),
                trace_status: p.view(hit.status).unwrap(),
                wavelengths: t.num_wavelengths(),
                stokes: [1.0, 0.2, 0.3, 0.4].map(Df32::from_f64),
                output,
                tile,
            }
            .into(),
        )));
    p.outputs = vec![output, hit.status];
    p.settings = serde_json::json!({"package":"P7","field":field,"wavelength":wl,"tile":tile,"version":consts::MKL_RAD_VERSION});
    (p, h)
}
pub fn check_samples(b: &dyn QualificationBackend) {
    let state = mklight_trace::PolarizationState::Stokes {
        s: mklight_trace::Stokes {
            s0: 1.,
            s1: 0.2,
            s2: 0.3,
            s3: 0.4,
        },
    };
    let points = (mklight_trace::PupilSampling::Grid {
        n: std::env::var("P7_GRID")
            .ok()
            .map(|v| v.parse().unwrap())
            .unwrap_or(7),
    })
    .points();
    let mut totals = [0f64; 4];
    let mut count = 0;
    let mut blocked = 0;
    for (name, model) in samples::all() {
        let mut maxima = [0f64; 4];
        let t = testutil::tracer(model);
        for f in 0..t.fields().len() {
            for w in 0..t.num_wavelengths() {
                let (p, h) = transport_plan(b, &t, f, w, &points, 1024);
                let r = b
                    .submit_plan(p, &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let guard = b.read_view(r.outputs[0].view()).unwrap();
                let values: &[PolarizationValue] = bytemuck::cast_slice(guard.words());
                for (i, (v, pt)) in values.iter().zip(&points).enumerate() {
                    let refp = t.trace_polarized(t.field(f), w, pt[0], pt[1], &Default::default());
                    if !refp.is_ok() {
                        assert_eq!(
                            v.status,
                            consts::MKL_RAD_BLOCKED,
                            "{name} f={f} w={w} i={i}"
                        );
                        assert!(v.jones.iter().all(|x| x.to_f64() == 0.));
                        blocked += 1;
                        continue;
                    }
                    assert_eq!(v.status, 0, "{name} f={f} w={w} i={i} {v:?}");
                    count += 1;
                    for (array, j) in [
                        (&v.jones, refp.jones_amplitude()),
                        (
                            &v.proper,
                            refp.proper_jones()
                                .scale(refp.internal_transmittance.sqrt()),
                        ),
                    ] {
                        for k in 0..4 {
                            let z = j.m[k / 2][k % 2];
                            let err = (array[2 * k].to_f64() - z.re)
                                .hypot(array[2 * k + 1].to_f64() - z.im);
                            maxima[0] = maxima[0].max(err);
                            assert!(
                                err < FIELD_TOL,
                                "Jones {name} f={f} w={w} i={i} k={k}: {err:e} {array:?} ref={j:?}"
                            );
                        }
                    }
                    let intensity = refp.intensity(&state);
                    let err = (v.power[2].to_f64() - intensity).abs();
                    maxima[1] = maxima[1].max(err);
                    assert!(err < FIELD_TOL, "intensity {name}: {err:e}");
                    let props = refp.properties();
                    let err = (v.properties[1].to_f64() - props.retardance).abs();
                    maxima[2] = maxima[2].max(err);
                    assert!(err < RETARDANCE_TOL, "retardance {name}: {err:e}");
                    assert!((v.properties[0].to_f64() - props.diattenuation).abs() < FIELD_TOL);
                    let ledger = v.power[2].to_f64()
                        + v.power[4].to_f64()
                        + v.power[5].to_f64()
                        + v.power[6].to_f64();
                    maxima[3] = maxima[3].max((ledger - 1.).abs());
                    assert!(
                        (ledger - 1.).abs() < LEDGER_TOL,
                        "ledger {name}: {ledger:e}"
                    );
                    let fields = refp.image_fields();
                    for (k, z) in fields.iter().flatten().enumerate() {
                        let err = (v.fields[2 * k].to_f64() - z.re)
                            .hypot(v.fields[2 * k + 1].to_f64() - z.im);
                        assert!(err < FIELD_TOL, "vector {name}: {err:e}");
                    }
                    let scalar = mklight_trace::polarization::unpolarized_path_product(
                        t.system(),
                        w,
                        &refp.records,
                    ) * refp.internal_transmittance;
                    assert!(
                        (v.power[0].to_f64() - scalar).abs() <= FIELD_TOL * scalar.max(1.),
                        "scalar {name}"
                    );
                }
                assert_eq!(r.stats.fallback_count, 0, "{name}");
                assert_eq!(r.stats.copied_bytes, 0);
                b.release_systems(h).unwrap();
            }
        }
        eprintln!("sample {name} maxima={maxima:?}");
        for (t, m) in totals.iter_mut().zip(maxima) {
            *t = t.max(m);
        }
    }
    eprintln!(
        "P7 sample totals survivors={count} blocked={blocked} max Jones/intensity/retardance/ledger={totals:?}"
    );
}
pub fn fixtures(b: &dyn ComputeBackend) {
    use mklight_trace::{PolarizationState, PupilSamples, RadiometryModel, TransmissionSettings};
    let defs = [
        CoatingDef::Ideal {
            reflectance: 0.2,
            transmittance: 0.7,
        },
        CoatingDef::Tabulated {
            reflectance: vec![(0.4, 0.05), (0.7, 0.1)],
        },
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![Layer {
                material: MaterialRef::Constant { n: 1.7 },
                extinction: 0.03,
                thickness_um: 0.1,
                quarter_waves: None,
            }],
            mirror_substrate: Some((0.96, 6.69)),
            substrate: Default::default(),
        },
        CoatingDef::Ideal {
            reflectance: 0.,
            transmittance: 0.,
        },
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![Layer {
                material: MaterialRef::Constant { n: 1.7 },
                extinction: 0.,
                thickness_um: 1e6,
                quarter_waves: None,
            }],
            mirror_substrate: None,
            substrate: Default::default(),
        },
    ];
    for def in defs {
        let mut model = samples::cooke_triplet();
        model.coatings.insert("P7".into(), def.clone());
        for s in &mut model.surfaces[1..] {
            s.coating = Some("P7".into());
        }
        let t = testutil::tracer(model);
        let points = vec![[0., 0.], [0.3, 0.4], [0., 0.8], [1.4, 1.4]];
        let state = PolarizationState::Jones {
            e: mklight_trace::JonesVector::linear(0.3),
        };
        let ctx = JobContext::new();
        let r = pupil(b, &t, t.field(2).point, 1, &points, &state, true, &ctx).unwrap();
        assert_eq!(ctx.progress.fraction(), 1.);
        for (v, pt) in r.values.iter().zip(points) {
            let reference = t.trace_polarized(t.field(2), 1, pt[0], pt[1], &Default::default());
            if reference.is_ok() {
                let err = (v.power[2].to_f64() - reference.intensity(&state)).abs();
                assert!(err < FIELD_TOL, "fixture {def:?} {err:e}");
                assert_eq!(v.defined & 1 != 0, reference.properties().retarder_defined);
                let ledger = v.power[2].to_f64()
                    + v.power[4].to_f64()
                    + v.power[5].to_f64()
                    + v.power[6].to_f64();
                assert!((ledger - 1.).abs() < LEDGER_TOL, "fixture ledger {ledger}");
            } else {
                assert_eq!(v.status, consts::MKL_RAD_BLOCKED);
            }
        }
        if matches!(def,CoatingDef::ThinFilm {ref layers,..} if layers[0].thickness_um>1e5) {
            assert!(r.stats.fallback_count > 0);
            assert!(
                r.run_info
                    .warnings
                    .iter()
                    .any(|w| w.contains("reference evaluated"))
            );
        }
    }
    for model in [samples::singlet(), samples::hybrid_diffractive_achromat()] {
        let t = testutil::tracer(model);
        for model in [
            RadiometryModel::Unpolarized,
            RadiometryModel::Polarized {
                state: PolarizationState::Unpolarized,
            },
        ] {
            let settings = TransmissionSettings {
                fields: vec![t.field(0).point, t.field(t.fields().len() - 1).point],
                wavelengths: vec![0],
                pupil: PupilSamples::Grid { n: 7 },
                model,
                clip: true,
            };
            let a = system_transmission(b, &t, &settings, &JobContext::new()).unwrap();
            let c = mklight_trace::system_transmission(&t, &settings, &JobContext::new()).unwrap();
            for (a, c) in a.points.iter().zip(c.points) {
                assert_eq!(a.samples, c.samples);
                assert_eq!(a.failed, c.failed);
                assert!(
                    (a.transmission - c.transmission).abs() < FIELD_TOL,
                    "transmission {:?} {a:?} {c:?}",
                    settings.model
                );
            }
        }
    }
}
pub fn known_answers(b: &dyn ComputeBackend) {
    let tracer = testutil::tracer(samples::singlet());
    for n in [0, 2] {
        let settings = mklight_trace::TransmissionSettings {
            pupil: mklight_trace::PupilSamples::Grid { n },
            ..Default::default()
        };
        assert!(matches!(
            system_transmission(b, &tracer, &settings, &JobContext::new()),
            Err(mklight_core::MkError::InvalidModel(_))
        ));
    }

    for scale in [1e-200, 1., 1e200] {
        let state = mklight_trace::PolarizationState::Stokes {
            s: mklight_trace::Stokes {
                s0: scale,
                s1: 0.2 * scale,
                s2: 0.3 * scale,
                s3: 0.4 * scale,
            },
        };
        let r = pupil(
            b,
            &tracer,
            tracer.field(0).point,
            0,
            &[[0., 0.5]],
            &state,
            true,
            &JobContext::new(),
        )
        .unwrap();
        let reference = tracer.trace_polarized(tracer.field(0), 0, 0., 0.5, &Default::default());
        assert!((r.values[0].power[2].to_f64() - reference.intensity(&state)).abs() < 1e-12);
    }
    let cs = coatings();
    let domains = vec![
        CoatingDomain {
            coating: cs[0].clone(),
            wavelength: 0.55,
            n_in: 1.,
            n_out: 1.5,
            mirror: false,
            arrival: mklight_core::materials::coating::Arrival::Before,
        },
        CoatingDomain {
            coating: cs[0].clone(),
            wavelength: 0.55,
            n_in: 1.51,
            n_out: 1.,
            mirror: false,
            arrival: mklight_core::materials::coating::Arrival::After,
        },
        CoatingDomain {
            coating: cs[3].clone(),
            wavelength: 0.55,
            n_in: 1.,
            n_out: 1.69,
            mirror: false,
            arrival: mklight_core::materials::coating::Arrival::Before,
        },
        CoatingDomain {
            coating: cs[3].clone(),
            wavelength: 0.55,
            n_in: 1.69,
            n_out: 1.,
            mirror: false,
            arrival: mklight_core::materials::coating::Arrival::After,
        },
        CoatingDomain {
            coating: cs[4].clone(),
            wavelength: 0.55,
            n_in: 1.,
            n_out: 1.5,
            mirror: true,
            arrival: mklight_core::materials::coating::Arrival::Before,
        },
    ];
    let prepared = PreparedCoatings::new(b, &domains, &JobContext::new()).unwrap();
    let queries = [
        CoatingQuery::new(0, 1.).unwrap(),
        CoatingQuery::new(0, 1.5f64.atan().cos()).unwrap(),
        CoatingQuery::new(1, 54.6f64.to_radians().cos()).unwrap(),
        CoatingQuery::new(2, 1.).unwrap(),
        CoatingQuery::new(3, 1.).unwrap(),
        CoatingQuery::new(4, 1.).unwrap(),
        CoatingQuery::new(2, 1e-7).unwrap(),
    ];
    let r = prepared
        .evaluate(
            b,
            Snapshot::new(samples::singlet()).unwrap(),
            &queries,
            &JobContext::new(),
        )
        .unwrap();
    let v = &r.values;
    assert!((v[0].powers[0].to_f64() - 0.04).abs() < 1e-12);
    assert!(v[1].powers[1].to_f64() < 1e-24);
    assert!((v[1].powers[3].to_f64() - 1.).abs() < 1e-12);
    let phase = |a: &CoatingValue, k: usize| {
        a.amplitudes[2 * k + 1]
            .to_f64()
            .atan2(a.amplitudes[2 * k].to_f64())
    };
    let rhomb = (phase(&v[2], 1) - phase(&v[2], 0)).to_degrees();
    assert!((rhomb - 45.).abs() < 0.3);
    for p in 0..2 {
        assert!((v[2].powers[p].to_f64() - 1.).abs() < 1e-12);
        assert_eq!(v[2].powers[p + 2].to_f64(), 0.);
    }
    for i in [3, 4] {
        assert!(v[i].powers[0].to_f64() < 1e-24);
        assert!((v[i].powers[2].to_f64() - 1.).abs() < 1e-12);
    }
    for k in 0..8 {
        assert!((v[3].amplitudes[k].to_f64() - v[4].amplitudes[k].to_f64()).abs() < 1e-12);
    }
    assert!((v[5].amplitudes[0].to_f64() + v[5].amplitudes[2].to_f64()).abs() < 1e-12);
    assert!((v[5].amplitudes[1].to_f64() + v[5].amplitudes[3].to_f64()).abs() < 1e-12);
    assert_eq!(r.stats.fallback_count, 1);
    assert_eq!(r.run_info.warnings.len(), 1);
    let refa = domains[2].coating.amplitudes(
        0.55,
        1.,
        1.69,
        1e-7,
        false,
        mklight_core::materials::coating::Arrival::Before,
    );
    assert!((v[6].powers[2].to_f64() - refa.power().ts).abs() < 1e-12);
    // A changed wavelength/medium produces a distinct immutable table, while
    // the old lease still returns its original answer.
    let mut changed = domains.clone();
    changed[2].wavelength = 0.45;
    changed[2].n_out = 1.8;
    let newer = PreparedCoatings::new(b, &changed, &JobContext::new()).unwrap();
    let q = [queries[3]];
    let old = prepared
        .evaluate(
            b,
            Snapshot::new(samples::singlet()).unwrap(),
            &q,
            &JobContext::new(),
        )
        .unwrap();
    let new = newer
        .evaluate(
            b,
            Snapshot::new(samples::singlet()).unwrap(),
            &q,
            &JobContext::new(),
        )
        .unwrap();
    assert!(old.values[0].powers[0].to_f64() < 1e-24);
    assert!(new.values[0].powers[0].to_f64() > 1e-4);
}
pub fn malus_and_channels(b: &dyn ComputeBackend) {
    use mklight_trace::{JonesVector, PolarizationState};
    let t = testutil::tracer(samples::singlet());
    let points = [[0., 0.9]];
    let mut intensities = vec![];
    for angle in [0., std::f64::consts::FRAC_PI_2, 0.35] {
        let state = PolarizationState::Jones {
            e: JonesVector::linear(angle),
        };
        let r = pupil(
            b,
            &t,
            t.field(0).point,
            1,
            &points,
            &state,
            true,
            &JobContext::new(),
        )
        .unwrap();
        assert!(r.values.is_resident() && r.trace_status.is_resident());
        let v = r.values[0];
        intensities.push(v.power[2].to_f64());
        let k = v.direction.map(Df32::to_f64);
        for a in 0..2 {
            let dot = (0..3)
                .map(|i| k[i] * v.fields[6 * a + 2 * i].to_f64())
                .sum::<f64>();
            assert!(dot.abs() < 1e-12);
        }
        assert!((v.power[1].to_f64() - v.power[3].to_f64()).abs() < 1e-12);
        assert!(v.power[1].to_f64() > v.power[0].to_f64());
    }
    let expected = intensities[0] * 0.35f64.cos().powi(2) + intensities[1] * 0.35f64.sin().powi(2);
    assert!((intensities[2] - expected).abs() < 1e-12);
}
pub fn geometric_and_dim_fixtures(b: &dyn ComputeBackend) {
    use mklight_trace::PolarizationState;
    let mut absorbing = samples::singlet();
    absorbing.surfaces[1].material = MaterialRef::glass("ABSORB");
    // Bulk path lengths through tilted/decentred elements exercise the local
    // frame transfers, not only the axial case.
    let mut tilted_absorbing = testutil::tilted_triplet();
    tilted_absorbing.surfaces[3].material = MaterialRef::glass("ABSORB");
    tilted_absorbing.surfaces[5].material = MaterialRef::glass("ABSORB");
    let mut long = samples::singlet();
    long.surfaces[2].thickness = 900.;
    let mut mirror = samples::newtonian_telescope();
    mirror.coatings.insert(
        "film".into(),
        CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: vec![],
            mirror_substrate: Some((0.96, 6.69)),
            substrate: Default::default(),
        },
    );
    for s in &mut mirror.surfaces {
        if s.interaction == mklight_core::Interaction::Reflect {
            s.coating = Some("film".into());
        }
    }
    let mut dim = samples::singlet();
    dim.coatings.insert(
        "dim".into(),
        CoatingDef::Ideal {
            reflectance: 0.,
            transmittance: 1e-12,
        },
    );
    dim.surfaces[1].coating = Some("dim".into());
    dim.surfaces[2].coating = Some("dim".into());
    let systems = [
        ("absorbing", absorbing),
        ("long", long),
        ("tilted", testutil::tilted_triplet()),
        ("tilted absorbing", tilted_absorbing),
        (
            "vignetted",
            testutil::vignetted(&samples::cooke_triplet(), 0.5),
        ),
        ("metal", mirror),
        ("dim", dim),
        ("ideal lenses", testutil::ideal_lenses()),
        ("ideal relay", testutil::ideal_relay()),
        ("TIR", testutil::tir_block()),
    ];
    let pts = (mklight_trace::PupilSampling::Grid { n: 17 }).points();
    let mut blocked = 0;
    let tele = p1::special_systems()
        .into_iter()
        .filter(|(name, _)| name.contains("telecentric"));
    for (name, model) in systems
        .into_iter()
        .map(|(n, s)| (n.to_string(), s))
        .chain(tele)
    {
        let t = testutil::tracer(model);
        let f = t.fields().len() - 1;
        let state = PolarizationState::Unpolarized;
        let r = pupil(
            b,
            &t,
            t.field(f).point,
            0,
            &pts,
            &state,
            true,
            &JobContext::new(),
        )
        .unwrap();
        let mut bulk = 0f64;
        for (v, pt) in r.values.iter().zip(&pts) {
            let reference = t.trace_polarized(t.field(f), 0, pt[0], pt[1], &Default::default());
            if !reference.is_ok() {
                assert_eq!(v.status, consts::MKL_RAD_BLOCKED);
                blocked += 1;
                continue;
            }
            assert_eq!(v.status, 0);
            assert_eq!(
                v.defined & 1 != 0,
                reference.properties().retarder_defined,
                "{name}"
            );
            assert!(
                (v.power[2].to_f64() - reference.intensity(&state)).abs() < FIELD_TOL,
                "{name} point={pt:?} power={:?} reference={} tau={} records={:?}",
                v.power,
                reference.intensity(&state),
                reference.internal_transmittance,
                reference.records
            );
            // The f64 reference keeps a unit longitudinal PRT term, whose
            // rounding is relative to 1, not to the transmitted amplitude: the
            // dim pupil (|J|² ≈ 1e-24) carries ~1e-6 rad of reference noise.
            let props = reference.properties();
            let tolerance = RETARDANCE_TOL.max(1e-15 / props.t_unpolarized.sqrt());
            assert!(
                (v.properties[1].to_f64() - props.retardance).abs() < tolerance,
                "{name} retardance {} reference {:?}",
                v.properties[1].to_f64(),
                props
            );
            let budget = reference.energy_budget(&state);
            assert!(
                (v.power[6].to_f64() - budget.bulk_absorbed).abs() < FIELD_TOL,
                "{name}"
            );
            bulk = bulk.max(v.power[6].to_f64());
            let ledger = v.power[2].to_f64()
                + v.power[4].to_f64()
                + v.power[5].to_f64()
                + v.power[6].to_f64();
            assert!((ledger - 1.).abs() < LEDGER_TOL, "{name} ledger {ledger}");
        }
        assert_eq!(
            name.contains("absorbing"),
            bulk > 0.01,
            "{name} bulk {bulk}"
        );
        eprintln!("P7 geometric fixture {name} max_bulk={bulk}");
    }
    assert!(blocked > 0);
    eprintln!("P7 geometric fixtures blocked={blocked}");
}
#[path = "../../mklight-trace/tests/common/mod.rs"]
mod optical_fixture;
pub fn singular_retarder(b: &dyn ComputeBackend) {
    use mklight_core::{Interaction, Placement, SurfaceShape};
    use mklight_trace::PolarizationState;
    for offset in [0., -1e-10, 1e-10, 1e-6] {
        let angle = 1.5f64.atan().to_degrees() + offset;
        let mut mirror = optical_fixture::surf(2, SurfaceShape::Plane, -20., MaterialRef::Air);
        mirror.interaction = Interaction::Reflect;
        mirror.placement = Some(Placement {
            tilt_x: angle,
            ..Default::default()
        });
        mirror.coating = Some("dielectric".into());
        let mut s = optical_fixture::infinite_system(
            4.,
            vec![
                optical_fixture::surf(1, SurfaceShape::Plane, 10., MaterialRef::Air),
                mirror,
            ],
            3,
        );
        s.surfaces[3].placement = Some(Placement {
            tilt_x: angle,
            ..Default::default()
        });
        s.fields = optical_fixture::angle_fields(&[0.]);
        s.coatings.insert(
            "dielectric".into(),
            CoatingDef::ThinFilm {
                reference_wavelength_um: 0.55,
                layers: vec![],
                mirror_substrate: Some((1.5, 0.)),
                substrate: Default::default(),
            },
        );
        let t = testutil::tracer(s);
        let r = pupil(
            b,
            &t,
            t.field(0).point,
            0,
            &[[0., 0.]],
            &PolarizationState::Unpolarized,
            false,
            &JobContext::new(),
        )
        .unwrap();
        let reference = t.trace_polarized(
            t.field(0),
            0,
            0.,
            0.,
            &mklight_trace::TraceOptions::unclipped(),
        );
        assert!(reference.is_ok());
        let props = reference.properties();
        let v = r.values[0];
        assert_eq!(
            v.defined & 1 != 0,
            props.retarder_defined,
            "offset {offset}"
        );
        assert!(
            (v.properties[1].to_f64() - props.retardance).abs() < RETARDANCE_TOL,
            "offset {offset}"
        );
        if offset.abs() < 1e-9 {
            assert!(r.stats.fallback_count > 0);
        }
    }
}
pub fn scalar_reference(b: &dyn QualificationBackend) {
    for (name, model) in samples::all() {
        let t = testutil::tracer(model);
        let pts = (mklight_trace::PupilSampling::Grid { n: 17 }).points();
        let f = t.fields().len() - 1;
        let w = t.system().primary_wavelength;
        let (mut p, h) = transport_plan(b, &t, f, w, &pts, 4096);
        for op in &mut p.operations {
            if let ResidentOp::Radiometry(Request::Polarization(r)) = op {
                r.scalar = true;
            }
        }
        let result = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let view = b.read_view(result.outputs[0].view()).unwrap();
        let values: &[PolarizationValue] = bytemuck::cast_slice(view.words());
        for (v, pt) in values.iter().zip(pts) {
            let r = t.trace_polarized(t.field(f), w, pt[0], pt[1], &Default::default());
            if !r.is_ok() {
                assert_eq!(v.status, consts::MKL_RAD_BLOCKED);
                continue;
            }
            let expected =
                mklight_trace::polarization::unpolarized_path_product(t.system(), w, &r.records)
                    * r.internal_transmittance;
            let error = (v.power[0].to_f64() - expected).abs();
            assert!(error <= FIELD_TOL * expected.max(1.), "{name}: {error:e}");
        }
        b.release_systems(h).unwrap();
    }
}
