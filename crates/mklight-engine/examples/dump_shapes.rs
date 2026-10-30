use mklight_core::materials::MaterialLibrary;
use mklight_core::{JobContext, ResolvedSystem, Snapshot, samples};
use std::sync::Arc;
fn main() {
    let snap = Snapshot::new(samples::double_gauss()).unwrap();
    let rs = Arc::new(ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap());
    let model = mklight_ghost::GhostModel::from_system(rs.clone(), Default::default()).unwrap();
    println!("=== ghost model settings\n{}", serde_json::to_string(model.settings()).unwrap());
    let ps = mklight_ghost::ParaxialGhostSettings::default();
    println!("=== paraxial ghost settings\n{}", serde_json::to_string(&ps).unwrap());
    let r = mklight_ghost::paraxial_ghosts(&model, &ps).unwrap();
    let mut v = serde_json::to_value(&r).unwrap();
    trunc(&mut v);
    println!("--- result\n{}", v);
    let is = mklight_ghost::IrradianceSettings {
        pupil_samples: 8,
        map_bins: [8, 8],
        ..Default::default()
    };
    println!("=== irradiance settings\n{}", serde_json::to_string(&mklight_ghost::IrradianceSettings::default()).unwrap());
    let r = mklight_ghost::ghost_irradiance(&model, &is, &JobContext::new()).unwrap();
    let mut v = serde_json::to_value(&r).unwrap();
    trunc(&mut v);
    println!("--- result\n{}", v);
    let ss = mklight_stray::analysis::StrayAnalysisSettings { rays: 200, ..Default::default() };
    println!("=== stray settings\n{}", serde_json::to_string(&mklight_stray::analysis::StrayAnalysisSettings::default()).unwrap());
    let r = mklight_stray::analysis::analyze(rs.clone(), &ss, &JobContext::new()).unwrap();
    let mut v = serde_json::to_value(&r).unwrap();
    trunc(&mut v);
    println!("--- result\n{}", v);
    let t = mklight_trace::SequentialTracer::new(rs.clone()).unwrap();
    let ts = mklight_trace::TransmissionSettings::default();
    println!("=== transmission settings\n{}", serde_json::to_string(&ts).unwrap());
    let r = mklight_trace::system_transmission(&t, &ts, &JobContext::new()).unwrap();
    let mut v = serde_json::to_value(&r).unwrap();
    trunc(&mut v);
    println!("--- result\n{}", v);
    let pm = mklight_trace::PolarizationMapSettings {
        field: t.fields()[0].point,
        wavelength: 1,
        grid: 5,
        state: mklight_trace::PolarizationState::Unpolarized,
        clip: true,
    };
    println!("=== pol settings\n{}", serde_json::to_string(&pm).unwrap());
    let r = mklight_trace::polarization_pupil_map(&t, &pm, &JobContext::new()).unwrap();
    let mut v = serde_json::to_value(&r).unwrap();
    trunc(&mut v);
    println!("--- result\n{}", v);
    let (_, rep) = mklight_trace::apply_solves(&samples::double_gauss(), &MaterialLibrary::with_builtin()).unwrap();
    let mut v = serde_json::to_value(&rep).unwrap();
    trunc(&mut v);
    println!("=== solves\n{}", v);
}
fn trunc(v: &mut serde_json::Value) {
    match v {
        serde_json::Value::Array(a) => {
            a.truncate(2);
            for x in a {
                trunc(x)
            }
        }
        serde_json::Value::Object(o) => {
            o.remove("run_info");
            o.remove("run");
            for (_, x) in o {
                trunc(x)
            }
        }
        _ => {}
    }
}
