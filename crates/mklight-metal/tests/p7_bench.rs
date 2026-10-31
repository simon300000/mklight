#![cfg(target_os = "macos")]
#[path = "../../mklight-gpu/tests/p7_support.rs"]
mod support;
use mklight_core::{JobContext, samples};
use mklight_gpu::{radiometry::*, *};
use mklight_metal::MetalBackend;
use mklight_trace::{PolarizationState, PupilSampling};
use rayon::prelude::*;
use std::time::Instant;
fn percentiles(mut x: Vec<f64>) -> (f64, f64) {
    x.sort_by(f64::total_cmp);
    (x[x.len() / 2], x[(x.len() * 95).div_ceil(100) - 1])
}
#[test]
#[ignore = "quiet host, real Metal release measurement"]
fn grids() {
    let initialization = Instant::now();
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 benchmark requires Metal");
    eprintln!(
        "P7 backend_init_ms={} hardware={:?} build={:?}",
        initialization.elapsed().as_secs_f64() * 1e3,
        b.info(),
        b.precision()
    );
    let dense = std::env::var_os("P7_DENSE").is_some();
    let mut models = samples::all();
    for (name, count, metal) in [
        ("cooke_film1", 1, false),
        ("cooke_film4", 4, false),
        ("newtonian_metal", 1, true),
    ] {
        use mklight_core::{CoatingDef, Layer, MaterialRef};
        let mut model = if metal {
            samples::newtonian_telescope()
        } else {
            samples::cooke_triplet()
        };
        model.coatings.insert(
            "P7 bench".into(),
            CoatingDef::ThinFilm {
                reference_wavelength_um: 0.55,
                layers: (0..count)
                    .map(|i| Layer {
                        material: MaterialRef::Constant {
                            n: if i % 2 == 0 { 1.38 } else { 2.1 },
                        },
                        extinction: if i == 0 { 0.02 } else { 0. },
                        thickness_um: 0.07,
                        quarter_waves: None,
                    })
                    .collect(),
                mirror_substrate: if metal { Some((0.96, 6.69)) } else { None },
            },
        );
        for surface in &mut model.surfaces {
            if !metal || surface.interaction == mklight_core::Interaction::Reflect {
                surface.coating = Some("P7 bench".into());
            }
        }
        models.push((name, model));
    }
    for (name, model) in models {
        let sizes = if dense
            && [
                "singlet",
                "cooke_triplet",
                "double_gauss",
                "newtonian",
                "relay",
                "hybrid_doe",
                "anamorphic",
                "cooke_film1",
                "cooke_film4",
                "newtonian_metal",
            ]
            .contains(&name)
        {
            vec![65, 256, 1024]
        } else {
            vec![65]
        };
        let t = testutil::tracer(model);
        let field = t.fields().len() - 1;
        let wl = t.system().primary_wavelength;
        for side in sizes {
            let pts = (PupilSampling::Grid { n: side }).points();
            let state = PolarizationState::Unpolarized;
            let setup = Instant::now();
            let prepared = PreparedRadiometry::new(&b, t.system(), &JobContext::new()).unwrap();
            let h = b
                .upload_system(&PackedSystem::pack_all(t.system()).unwrap())
                .unwrap();
            let p = pupil_plan(
                &b,
                &t,
                &prepared,
                h,
                t.field(field),
                wl,
                &pts,
                &state,
                true,
                &JobContext::new(),
            )
            .unwrap();
            let setup_ms = setup.elapsed().as_secs_f64() * 1e3;
            let gpu = || {
                let start = Instant::now();
                let r = b
                    .submit_plan(p.clone(), &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let view = b.read_view(r.outputs[0].view()).unwrap();
                let sums = b.read_view(r.outputs[2].view()).unwrap();
                std::hint::black_box((view.words().len(), sums.words()));
                (start.elapsed().as_secs_f64() * 1e3, r)
            };
            let cpu = || {
                let start = Instant::now();
                let outputs: Vec<_> = pts
                    .par_iter()
                    .map(|pt| {
                        let r = t.trace_polarized(
                            t.field(field),
                            wl,
                            pt[0],
                            pt[1],
                            &Default::default(),
                        );
                        let j = r.jones_amplitude();
                        let budget = r.energy_budget(&state);
                        let losses = budget.surfaces.iter().fold([0., 0.], |mut v, s| {
                            v[0] += s.diverted;
                            v[1] += s.absorbed;
                            v
                        });
                        let products = r.steps.iter().fold([1., 1., 1.], |v, s| {
                            let (a, b) = s.path_sp();
                            [v[0] * 0.5 * (a + b), v[1] * a, v[2] * b]
                        });
                        (
                            j,
                            r.image_fields(),
                            r.properties(),
                            j.properties().retardance,
                            products,
                            [losses[0], losses[1], budget.bulk_absorbed],
                            r.k_out,
                        )
                    })
                    .collect();
                std::hint::black_box(&outputs);
                start.elapsed().as_secs_f64() * 1e3
            };
            let (cold, first) = gpu();
            let qualification = Instant::now();
            let guard = b.read_view(first.outputs[0].view()).unwrap();
            let values: &[PolarizationValue] = bytemuck::cast_slice(guard.words());
            let mut max_jones = 0f64;
            for i in (0..pts.len()).step_by((pts.len() / 128).max(1)) {
                let pt = pts[i];
                let r = t.trace_polarized(t.field(field), wl, pt[0], pt[1], &Default::default());
                let v = values[i];
                if !r.is_ok() {
                    assert_eq!(v.status, consts::MKL_RAD_BLOCKED);
                    continue;
                }
                assert_eq!(v.status, 0);
                let j = r.jones_amplitude();
                for k in 0..4 {
                    let z = j.m[k / 2][k % 2];
                    let err =
                        (v.jones[2 * k].to_f64() - z.re).hypot(v.jones[2 * k + 1].to_f64() - z.im);
                    max_jones = max_jones.max(err);
                    assert!(err < 1e-4 * z.norm() + 1e-7, "{name} {side} {err:e}");
                }
                let ledger = v.power[2].to_f64()
                    + v.power[4].to_f64()
                    + v.power[5].to_f64()
                    + v.power[6].to_f64();
                assert!((ledger - 1.).abs() < 1e-5);
            }
            drop(guard);
            let qualification_ms = qualification.elapsed().as_secs_f64() * 1e3;
            let cpu_cold = cpu();
            let mut walls = vec![];
            let mut devices = vec![];
            let mut cpus = vec![];
            let mut last = first;
            for _ in 0..10 {
                cpus.push(cpu());
                let (w, r) = gpu();
                walls.push(w);
                devices.push(r.stats.device_seconds * 1e3);
                last = r;
            }
            let (gm, gp) = percentiles(walls.clone());
            let (dm, dp) = percentiles(devices.clone());
            let (cm, cp) = percentiles(cpus.clone());
            eprintln!(
                "{}",
                serde_json::json!({"qualification_ms":qualification_ms,"max_jones":max_jones,"raw_gpu_ms":walls,"raw_device_ms":devices,"raw_cpu_ms":cpus,"sample":name,"grid":side,"rays":pts.len(),"field":field,"wavelength":wl,"seed":null,"setup_ms":setup_ms,"cold_gpu_ms":cold,"cold_cpu_ms":cpu_cold,"gpu_median_ms":gm,"gpu_p95_ms":gp,"device_median_ms":dm,"device_p95_ms":dp,"cpu_median_ms":cm,"cpu_p95_ms":cp,"speedup":cm/gm,"stats":last.stats,"table_bytes":prepared.surfaces.desc().byte_len().unwrap()+prepared.layers.desc().byte_len().unwrap(),"old_metal":"no PRT path"})
            );
            b.release_systems(h).unwrap();
        }
    }
}

#[test]
#[ignore = "quiet host coating table/stack sweep"]
fn coating_stacks() {
    use mklight_core::{
        materials::{MaterialLibrary, coating::ResolvedCoating},
        model::{CoatingDef, Environment, Layer, MaterialRef},
    };
    let initialization = Instant::now();
    let b = MetalBackend::new().unwrap().expect("P7 requires Metal");
    eprintln!(
        "P7 backend_init_ms={} hardware={:?} build={:?}",
        initialization.elapsed().as_secs_f64() * 1e3,
        b.info(),
        b.precision()
    );
    for count in [0, 1, 4, 16, 32] {
        let def = CoatingDef::ThinFilm {
            reference_wavelength_um: 0.55,
            layers: (0..count)
                .map(|i| Layer {
                    material: MaterialRef::Constant {
                        n: if i % 2 == 0 { 1.38 } else { 2.1 },
                    },
                    extinction: if i % 4 == 0 { 0.001 } else { 0. },
                    thickness_um: 0.04,
                    quarter_waves: None,
                })
                .collect(),
            mirror_substrate: None,
        };
        let coating = ResolvedCoating::resolve(
            if count == 0 { None } else { Some(&def) },
            &MaterialLibrary::with_builtin(),
            &Environment::default(),
        )
        .unwrap();
        let mut layers = vec![];
        let specs: Vec<_> = [0.45, 0.55, 0.65]
            .into_iter()
            .flat_map(|w| [(w, 1., 1.52), (w, 1.52, 1.)])
            .map(|(w, a, z)| CoatingSpec::resolve(&coating, w, a, z, false, &mut layers).unwrap())
            .collect();
        for n in [100, 10_000, 1_000_000] {
            let q: Vec<_> = (0..n)
                .map(|i| {
                    CoatingQuery::new((i % 6) as u32, 0.8 + 0.2 * (i % 1000) as f64 / 1000.)
                        .unwrap()
                })
                .collect();
            let setup = Instant::now();
            let p = support::plan(&b, &specs, &layers, &q, 65536);
            let setup_ms = setup.elapsed().as_secs_f64() * 1e3;
            let gpu = || {
                let start = Instant::now();
                let r = b
                    .submit_plan(p.clone(), &JobContext::new())
                    .unwrap()
                    .wait()
                    .unwrap();
                let view = b.read_view(r.outputs[0].view()).unwrap();
                std::hint::black_box(view.words());
                (start.elapsed().as_secs_f64() * 1e3, r)
            };
            let cpu = || {
                let start = Instant::now();
                let a: Vec<_> = q
                    .par_iter()
                    .map(|q| {
                        let s = specs[q.coating as usize];
                        let a = coating.amplitudes(
                            s.wavelength.to_f64(),
                            s.n_in.to_f64(),
                            s.n_out.to_f64(),
                            q.cosine.to_f64(),
                            false,
                        );
                        let p = a.power();
                        (
                            [a.rs, a.rp, a.ts_flux(), a.tp_flux()],
                            [p.rs, p.rp, p.ts, p.tp, 1. - p.rs - p.ts, 1. - p.rp - p.tp],
                        )
                    })
                    .collect();
                std::hint::black_box(a);
                start.elapsed().as_secs_f64() * 1e3
            };
            let (cold, mut last) = gpu();
            let guard = b.read_view(last.outputs[0].view()).unwrap();
            let values: &[CoatingValue] = bytemuck::cast_slice(guard.words());
            let mut maxima = [0f64; 3];
            for (i, v) in values.iter().take(3000).enumerate() {
                let q = q[i];
                let spec = specs[q.coating as usize];
                let a = coating.amplitudes(
                    spec.wavelength.to_f64(),
                    spec.n_in.to_f64(),
                    spec.n_out.to_f64(),
                    q.cosine.to_f64(),
                    false,
                );
                let rt = a.power();
                assert_eq!(v.status, 0);
                for (k, z) in [a.rs, a.rp, a.ts_flux(), a.tp_flux()].iter().enumerate() {
                    let err = (v.amplitudes[2 * k].to_f64() - z.re)
                        .hypot(v.amplitudes[2 * k + 1].to_f64() - z.im);
                    maxima[0] = maxima[0].max(err);
                    assert!(err <= 1e-4 * z.norm() + 1e-7);
                }
                for (k, z) in [rt.rs, rt.rp, rt.ts, rt.tp].into_iter().enumerate() {
                    let err = (v.powers[k].to_f64() - z).abs();
                    maxima[1] = maxima[1].max(err);
                    assert!(
                        err <= 1e-5 * z.abs() + 1e-10,
                        "stack {count} query {i} power {k}: {err:e}"
                    );
                }
                for pol in 0..2 {
                    let err = (v.powers[pol].to_f64()
                        + v.powers[pol + 2].to_f64()
                        + v.powers[pol + 4].to_f64()
                        - 1.)
                        .abs();
                    maxima[2] = maxima[2].max(err);
                    assert!(err <= 1e-6);
                }
            }
            drop(guard);
            let cc = cpu();
            let mut walls = vec![];
            let mut cpus = vec![];
            let mut devices = vec![];
            for _ in 0..10 {
                cpus.push(cpu());
                let (w, r) = gpu();
                walls.push(w);
                devices.push(r.stats.device_seconds * 1e3);
                last = r;
            }
            let (cm, cp) = percentiles(cpus.clone());
            let (gm, gp) = percentiles(walls.clone());
            let (dm, dp) = percentiles(devices.clone());
            eprintln!(
                "{}",
                serde_json::json!({"accuracy_amplitude_power_ledger":maxima,"layers":count,"queries":n,"setup_ms":setup_ms,"cold_gpu_ms":cold,"cold_cpu_ms":cc,"cpu_median_ms":cm,"cpu_p95_ms":cp,"gpu_median_ms":gm,"gpu_p95_ms":gp,"device_median_ms":dm,"device_p95_ms":dp,"raw_gpu_ms":walls,"raw_cpu_ms":cpus,"raw_device_ms":devices,"stats":last.stats,"table_bytes":specs.len()*size_of::<CoatingSpec>()+layers.len()*size_of::<CoatingLayer>(),"sampling":"cosine 0.8..1, three wavelengths, both sides","speedup":cm/gm})
            );
        }
    }
}

#[test]
#[ignore = "quiet host end-to-end transmission jobs"]
fn transmission_jobs() {
    use mklight_trace::{PupilSamples, RadiometryModel, TransmissionSettings};
    let initialization = Instant::now();
    let b = MetalBackend::new().unwrap().expect("P7 requires Metal");
    eprintln!(
        "P7 backend_init_ms={} hardware={:?} build={:?}",
        initialization.elapsed().as_secs_f64() * 1e3,
        b.info(),
        b.precision()
    );
    for (name, model) in [
        ("singlet", samples::singlet()),
        ("cooke", samples::cooke_triplet()),
        ("double_gauss", samples::double_gauss()),
    ] {
        let t = testutil::tracer(model);
        for n in [65, 256, 1024] {
            let settings = TransmissionSettings {
                fields: vec![t.field(t.fields().len() - 1).point],
                wavelengths: vec![t.system().primary_wavelength],
                pupil: PupilSamples::Grid { n },
                model: RadiometryModel::Unpolarized,
                clip: true,
            };
            let gpu = || {
                let start = Instant::now();
                let r = system_transmission(&b, &t, &settings, &JobContext::new()).unwrap();
                (start.elapsed().as_secs_f64() * 1e3, r)
            };
            let cpu = || {
                let start = Instant::now();
                let r =
                    mklight_trace::system_transmission(&t, &settings, &JobContext::new()).unwrap();
                (start.elapsed().as_secs_f64() * 1e3, r)
            };
            let (cold, mut result) = gpu();
            let (cc, reference) = cpu();
            assert!(
                (result.points[0].transmission - reference.points[0].transmission).abs() < 1e-5
            );
            let mut walls = vec![];
            let mut cpus = vec![];
            for _ in 0..10 {
                cpus.push(cpu().0);
                let (w, r) = gpu();
                walls.push(w);
                result = r;
            }
            let (cm, cp) = percentiles(cpus.clone());
            let (gm, gp) = percentiles(walls.clone());
            eprintln!(
                "{}",
                serde_json::json!({"job":"scalar_transmission","sample":name,"grid":n,"cold_gpu_ms":cold,"cold_cpu_ms":cc,"gpu_median_ms":gm,"gpu_p95_ms":gp,"cpu_median_ms":cm,"cpu_p95_ms":cp,"raw_gpu_ms":walls,"raw_cpu_ms":cpus,"speedup":cm/gm,"metadata":result.run_info.settings})
            );
        }
    }
}
