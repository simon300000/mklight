//! Kernel designs of the stray transport (work-split study key
//! `stray_kernel`): warm device transport time per scene type and design,
//! the designs alternated within each repetition.
//!
//! ```text
//! stray_kernels [RAYS] [REPS] [SCENES] [LENS.mklight ...]
//! ```
//!
//! Every device transport runs to completion (`gpu::Design::CpuTally`), so
//! the plan device time is the transport kernel's. `SCENES` is a
//! comma-separated filter of scene names (`all` for every scene); each lens
//! file adds a scene with the default (Lambertian-edge) options.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use mklight_core::model::{CoatingDef, Layer, MaterialRef, OpticalSystem, ScatterModel};
use mklight_core::{JobContext, ResolvedSystem, Snapshot, split};
use mklight_engine::{Engine, EngineConfig};
use mklight_stray::{SplitMode, StraySettings, StraySystem, SystemOptions, gpu};

fn resolve(s: OpticalSystem) -> Arc<ResolvedSystem> {
    let snapshot = Snapshot::new(s).unwrap();
    Arc::new(
        ResolvedSystem::new(
            &snapshot,
            Arc::new(mklight_core::materials::MaterialLibrary::with_builtin()),
        )
        .unwrap(),
    )
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let rays: u64 = a.first().map(|s| s.parse().unwrap()).unwrap_or(20_000);
    let reps: usize = a.get(1).map(|s| s.parse().unwrap()).unwrap_or(3);
    let filter: Vec<String> = a
        .get(2)
        .map(|s| s.split(',').map(String::from).collect())
        .unwrap_or_else(|| vec!["all".into()]);
    let engine = Engine::new(EngineConfig::default());
    let plain = {
        let mut o = SystemOptions::default();
        o.scene.default_edge_finish = ScatterModel::Lambertian { tis: 0.0 };
        o.scene.include_mechanics = false;
        o
    };
    let with_finish = |finish| {
        let mut o = SystemOptions::default();
        o.scene.default_edge_finish = finish;
        o
    };
    let weighted = SplitMode::Weighted { rare_floor: 0.02 };
    let filmed = {
        let mut s = mklight_core::samples::double_gauss();
        s.coatings.insert(
            "mgf2".into(),
            CoatingDef::ThinFilm {
                reference_wavelength_um: 0.55,
                layers: vec![Layer {
                    material: MaterialRef::Constant { n: 1.38 },
                    extinction: 0.0,
                    thickness_um: 0.0,
                    quarter_waves: Some(1.0),
                }],
                mirror_substrate: None,
            },
        );
        for surface in &mut s.surfaces[1..] {
            surface.coating = Some("mgf2".into());
        }
        s
    };
    type Scene = (String, Arc<ResolvedSystem>, SystemOptions, SplitMode);
    let mut scenes: Vec<Scene> = vec![
        (
            "dgauss plain".into(),
            resolve(mklight_core::samples::double_gauss()),
            plain.clone(),
            weighted,
        ),
        (
            "dgauss Lambertian".into(),
            resolve(mklight_core::samples::double_gauss()),
            SystemOptions::default(),
            weighted,
        ),
        (
            "dgauss Gaussian".into(),
            resolve(mklight_core::samples::double_gauss()),
            with_finish(ScatterModel::Gaussian {
                tis: 0.05,
                sigma_slope: 0.2,
            }),
            weighted,
        ),
        (
            "dgauss ABg".into(),
            resolve(mklight_core::samples::double_gauss()),
            with_finish(ScatterModel::Abg {
                a: 0.01,
                b: 0.001,
                g: 1.5,
            }),
            weighted,
        ),
        (
            "dgauss films".into(),
            resolve(filmed),
            SystemOptions::default(),
            weighted,
        ),
        (
            "petzval asphere".into(),
            resolve(mklight_core::samples::petzval_asphere()),
            SystemOptions::default(),
            weighted,
        ),
        (
            "hybrid DOE".into(),
            resolve(mklight_core::samples::hybrid_diffractive_achromat()),
            SystemOptions::default(),
            weighted,
        ),
        (
            "dgauss analog".into(),
            resolve(mklight_core::samples::double_gauss()),
            SystemOptions::default(),
            SplitMode::Analog,
        ),
        (
            "dgauss split".into(),
            resolve(mklight_core::samples::double_gauss()),
            SystemOptions::default(),
            SplitMode::Split {
                max_splits: 3,
                rare_floor: 0.02,
            },
        ),
    ];
    for path in a.iter().skip(3) {
        let s = engine.open_session(Path::new(path)).unwrap();
        let name = Path::new(path)
            .file_stem()
            .unwrap()
            .to_string_lossy()
            .chars()
            .take(12)
            .collect::<String>();
        scenes.push((
            format!("{name} Lambertian"),
            s.resolved().unwrap(),
            SystemOptions::default(),
            weighted,
        ));
    }
    let metal = mklight_metal::MetalBackend::new().unwrap().unwrap();
    let ctx = JobContext::default();
    let designs = ["specialized", "partial", "uber"];
    println!(
        "# {rays} rays, median of {reps} warm runs; wall ms / device ms per design, device Mevents/s"
    );
    for (name, rs, opts, mode) in &scenes {
        if !filter
            .iter()
            .any(|f| f == "all" || name.contains(f.as_str()))
        {
            continue;
        }
        let sys = StraySystem::new(rs.clone(), opts).unwrap();
        let field = (rs.system().fields.fields.len() > 1) as usize;
        let source = sys.system_field_source(field, 0.1, 1.).unwrap().spec;
        let settings = StraySettings {
            rays,
            mode: *mode,
            ..Default::default()
        };
        let mut wall = vec![Vec::new(); designs.len()];
        let mut device = vec![Vec::new(); designs.len()];
        let mut events = 0;
        for rep in 0..=reps {
            for (k, d) in designs.iter().enumerate() {
                split::set("stray_kernel", Some(d));
                let t = Instant::now();
                let r = gpu::run_design(
                    &metal,
                    rs.snapshot.clone(),
                    &sys.scene,
                    &source,
                    &settings,
                    gpu::TransportOptions::default(),
                    gpu::Design::CpuTally,
                    &ctx,
                )
                .unwrap();
                let ms = t.elapsed().as_secs_f64() * 1e3;
                split::reset();
                let info = &r.run_info.as_ref().unwrap().settings;
                assert_eq!(info["fallback_count"], 0, "{name}: {:?}", r.warnings);
                let dev: f64 = info["plans"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|p| p["device_seconds"].as_f64().unwrap())
                    .sum::<f64>()
                    * 1e3;
                events = r.energy.events;
                if rep > 0 {
                    wall[k].push(ms);
                    device[k].push(dev);
                }
            }
        }
        let cells = designs
            .iter()
            .enumerate()
            .map(|(k, d)| {
                let dev = median(&mut device[k]);
                format!(
                    "{d} {:.1}/{dev:.1} ({:.2})",
                    median(&mut wall[k]),
                    events as f64 / dev / 1e3
                )
            })
            .collect::<Vec<_>>();
        println!("{name:<22} {events:>9} events | {}", cells.join(" | "));
    }
}
