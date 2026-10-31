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
    let b = MetalBackend::new()
        .unwrap()
        .expect("P7 benchmark requires Metal");
    eprintln!("P7 hardware={:?} build={:?}", b.info(), b.precision());
    let dense = std::env::var_os("P7_DENSE").is_some();
    for (name, model) in samples::all() {
        let sizes = if dense
            && [
                "singlet",
                "cooke_triplet",
                "double_gauss",
                "newtonian",
                "relay",
                "hybrid_doe",
                "anamorphic",
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
                        (
                            r.jones_amplitude(),
                            r.image_fields(),
                            r.properties(),
                            r.energy_budget(&state),
                        )
                    })
                    .collect();
                std::hint::black_box(&outputs);
                start.elapsed().as_secs_f64() * 1e3
            };
            let (cold, first) = gpu();
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
                serde_json::json!({"raw_gpu_ms":walls,"raw_device_ms":devices,"raw_cpu_ms":cpus,"sample":name,"grid":side,"rays":pts.len(),"field":field,"wavelength":wl,"seed":null,"setup_ms":setup_ms,"cold_gpu_ms":cold,"cold_cpu_ms":cpu_cold,"gpu_median_ms":gm,"gpu_p95_ms":gp,"device_median_ms":dm,"device_p95_ms":dp,"cpu_median_ms":cm,"cpu_p95_ms":cp,"speedup":cm/gm,"stats":last.stats,"table_bytes":prepared.surfaces.desc().byte_len().unwrap()+prepared.layers.desc().byte_len().unwrap(),"old_metal":"no PRT path"})
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
    let b = MetalBackend::new().unwrap().expect("P7 requires Metal");
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
            Some(&def),
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
                        coating.amplitudes(
                            s.wavelength.to_f64(),
                            s.n_in.to_f64(),
                            s.n_out.to_f64(),
                            q.cosine.to_f64(),
                            false,
                        )
                    })
                    .collect();
                std::hint::black_box(a);
                start.elapsed().as_secs_f64() * 1e3
            };
            let (cold, mut last) = gpu();
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
                serde_json::json!({"layers":count,"queries":n,"setup_ms":setup_ms,"cold_gpu_ms":cold,"cold_cpu_ms":cc,"cpu_median_ms":cm,"cpu_p95_ms":cp,"gpu_median_ms":gm,"gpu_p95_ms":gp,"device_median_ms":dm,"device_p95_ms":dp,"raw_gpu_ms":walls,"raw_cpu_ms":cpus,"raw_device_ms":devices,"stats":last.stats,"table_bytes":specs.len()*size_of::<CoatingSpec>()+layers.len()*size_of::<CoatingLayer>(),"sampling":"cosine 0.8..1, three wavelengths, both sides","speedup":cm/gm})
            );
        }
    }
}
