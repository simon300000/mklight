//! Afocal image spaces on every backend: the wave analyses report angular
//! units and the aberration-free beam compressor is diffraction limited;
//! device OPD work (the OPD fan, the solvers' wavefront operands) is
//! completed by the f64 reference, which measures OPD from the plane
//! reference (docs/GPU.md §9.2). Runs on the CPU reference, the CPU-f32
//! emulation and, when this machine has them, Metal and WebGPU.

use serde_json::{Value, json};
use std::sync::Arc;

use mklight_core::{
    ApertureSpec, Field, FieldKind, FieldSet, Interaction, JobContext, MaterialRef, OpticalSystem,
    Surface, SurfaceShape, Wavelength, WavelengthSet,
};
use mklight_engine::backend::Backend;
use mklight_engine::buffer::BufferData;
use mklight_engine::gpu::GpuProvider;
use mklight_engine::project::Project;
use mklight_engine::{AnalysisRequest, AnalysisResult, Engine, EngineConfig, Session};

fn compressor() -> OpticalSystem {
    let mirror = |id, c: f64, t| {
        let mut m = Surface::new(
            id,
            SurfaceShape::Conic {
                curvature: c,
                conic: -1.0,
            },
            t,
            MaterialRef::Air,
        );
        m.interaction = Interaction::Reflect;
        m
    };
    let mut s = OpticalSystem::new_empty();
    s.aperture = ApertureSpec::EntrancePupilDiameter { value: 50.0 };
    s.fields = FieldSet {
        kind: FieldKind::ObjectAngle,
        fields: vec![Field::new(0.0, 0.0)],
    };
    s.wavelengths = WavelengthSet {
        wavelengths: vec![Wavelength {
            um: 0.55,
            weight: 1.0,
        }],
        primary: 0,
    };
    s.surfaces = vec![
        Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
        mirror(1, -1.0 / 1000.0, -400.0),
        mirror(2, -1.0 / 200.0, 200.0),
        Surface::new(3, SurfaceShape::Plane, 0.0, MaterialRef::Air),
    ];
    s.stop = 1;
    s
}

fn engine() -> Engine {
    let e = Engine::new(EngineConfig {
        worker_threads: 2,
        default_backend: Backend::CpuReference,
        ..Default::default()
    });
    e.register_backend(Arc::new(GpuProvider::emulation()));
    e
}

fn run_on(s: &Session, name: &str, settings: Value, backend: &Backend) -> AnalysisResult {
    let mut req = AnalysisRequest::new(name, settings);
    req.backend = Some(backend.clone());
    s.run_analysis(&req, &JobContext::new())
        .unwrap_or_else(|e| panic!("{name} on {}: {e}", backend.id()))
}

fn f64s(r: &AnalysisResult, name: &str) -> Vec<f64> {
    match r.buffers.iter().find(|b| b.name == name).unwrap().data() {
        BufferData::F64(v) => v.clone(),
        BufferData::F32(v) => v.iter().map(|&x| x as f64).collect(),
        _ => panic!(),
    }
}

#[test]
fn afocal_compressor_on_every_backend() {
    let e = engine();
    let s = e.session_from_project(Project::new(compressor())).unwrap();
    let mut backends = vec![Backend::CpuReference, Backend::Gpu("cpu_f32".into())];
    for name in ["metal", "webgpu"] {
        let id = format!("gpu:{name}");
        match e.backends().into_iter().find(|b| b.id == id) {
            Some(b) if b.available => backends.push(Backend::Gpu(name.into())),
            _ => eprintln!("skipping {id}: unavailable"),
        }
    }
    // Output beam 10 mm: cutoff D/λ cycles/rad.
    let cutoff = 10.0 / 0.55e-3 * 1e-3;
    for b in &backends {
        let label = b.id();
        let psf = run_on(&s, "psf", json!({"grid_size": 64}), b);
        assert_eq!(psf.json["units"]["position"], "mrad", "{label}");
        let strehl = psf.json["psfs"][0]["strehl"].as_f64().unwrap();
        assert!((strehl - 1.0).abs() < 1e-3, "{label}: Strehl {strehl}");
        let mtf = run_on(&s, "mtf", json!({"frequency_samples": 9}), b);
        assert_eq!(mtf.json["units"]["frequency"], "cycles/mrad", "{label}");
        let c = mtf.json["fields"][0]["cutoff"].as_f64().unwrap();
        assert!((c - cutoff).abs() < 1e-4 * cutoff, "{label}: cutoff {c}");
        let nu = f64s(&mtf, mtf.json["frequencies"].as_str().unwrap());
        let t = f64s(&mtf, mtf.json["fields"][0]["tangential"].as_str().unwrap());
        for (n, v) in nu.iter().zip(&t) {
            let exact = mklight_wave::diffraction_limited_mtf(n / cutoff, 0.0);
            assert!((v - exact).abs() < 3e-3, "{label} {n}: {v} vs {exact}");
        }
        let tf = run_on(&s, "through_focus_mtf", json!({"focus_samples": 3}), b);
        assert_eq!(tf.json["units"]["focus"], "mD", "{label}");
        if label == "gpu:webgpu" {
            // WebGPU has no native OPD fan or solver path.
            continue;
        }

        let fan = run_on(&s, "opd_fan", json!({"samples": 11}), b);
        let worst = fan
            .buffers
            .iter()
            .filter(|x| x.name.starts_with("values"))
            .flat_map(|x| match x.data() {
                BufferData::F64(v) => v.clone(),
                _ => vec![],
            })
            .fold(0.0, |m: f64, v| m.max(v.abs()));
        assert!(worst < 1e-6, "{label}: OPD fan {worst} waves");

        let spec = json!({"operands": [
            {"kind": {"type": "wavefront_rms"}, "target": 0.0, "weight": 1.0}
        ]});
        let merit = run_on(
            &s,
            "merit_evaluation",
            json!({"merit": spec, "hold_efl": false}),
            b,
        );
        assert_eq!(
            merit.json["operands"].as_array().unwrap().len(),
            1,
            "{label}"
        );
        let m = merit.json["merit"].as_f64().unwrap();
        assert!(m < 1e-10, "{label}: wavefront merit {m}");
    }
}
