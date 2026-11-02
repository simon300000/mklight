//! Automatic backend cost rules (GPU.md §9.3). Each analysis measures its
//! workload from the normalised settings and the system; at or below the
//! rule's threshold the CPU f64 reference is faster, above it the qualified
//! Metal route runs. Thresholds come from the P11 quiet-host measurements on
//! an Apple M2 Pro (GPU.md §9.2) and apply to Apple GPUs. Other devices and
//! analyses without a rule prefer qualified Metal; an explicit backend
//! bypasses this table.
use mklight_core::OpticalSystem;
use mklight_gpu::{BackendKind, DeviceInfo};
use serde_json::{Value, json};

pub(crate) const VERSION: u32 = 3;
const EVIDENCE: &str = "P11 quiet host, Apple M2 Pro, medians of 10 paired warm runs (GPU.md §9.2)";

/// One factor of a workload measure; the measure is their product.
#[derive(Clone, Copy)]
enum Factor {
    /// Selected fields (`fields` selector; null or an empty list is all).
    Fields,
    /// Selected wavelengths (`wavelengths` selector).
    Wavelengths,
    /// Pupil points of the `pattern` and `rings` settings.
    Pupil,
    /// The number at a settings pointer.
    Num(&'static str),
    /// The square of the number at a pointer.
    Square(&'static str),
    /// The product of the array at a pointer.
    Product(&'static str),
    /// Wave pupil samples of the pupil object at a pointer: `grid²`, or
    /// `rays²` for an automatic grid (its minimum).
    WavePupil(&'static str),
    /// Surfaces of the system.
    Surfaces,
}
use Factor::*;

/// Why a threshold has its value.
#[derive(Clone, Copy)]
enum Basis {
    /// Interpolated crossover of the dense series, between the largest CPU
    /// win and the smallest Metal win.
    Crossover,
    /// CPU won every measured shape; the threshold is the `ray_grid`
    /// crossover of the same trace kernels.
    TraceCrossover,
    /// CPU won, or minimised the total over the designs, at every measured
    /// shape; not extrapolated beyond the largest one.
    Measured,
}

struct Rule {
    analysis: &'static str,
    measure: &'static [Factor],
    /// CPU at or below this workload; `None`: always CPU.
    cpu_max: Option<f64>,
    basis: Basis,
}
use Basis::*;

const fn rule(
    analysis: &'static str,
    measure: &'static [Factor],
    cpu_max: f64,
    basis: Basis,
) -> Rule {
    Rule {
        analysis,
        measure,
        cpu_max: Some(cpu_max),
        basis,
    }
}
const PUPIL_RAYS: &[Factor] = &[Fields, Wavelengths, Pupil];
const FAN: &[Factor] = &[Fields, Wavelengths, Num("/samples")];
const WAVE: &[Factor] = &[Fields, Wavelengths, WavePupil("/pupil")];
/// The `ray_grid` crossover, shared by the trace-only analyses.
const TRACE: f64 = 32768.;

/// Thresholds and their measured basis: GPU.md §9.2. `Measured` thresholds
/// are the default-settings workload of the 3-field, 3-wavelength samples.
#[rustfmt::skip]
const RULES: &[Rule] = &[
    rule("spot_diagram", PUPIL_RAYS, 32768., Crossover),
    rule("ray_grid", PUPIL_RAYS, 32768., Crossover),
    rule("encircled_energy", PUPIL_RAYS, 32768., Crossover),
    rule("best_focus", PUPIL_RAYS, 16384., Crossover),
    rule("geometric_mtf", PUPIL_RAYS, 2048., Crossover),
    rule("ray_fan", FAN, TRACE, TraceCrossover),
    rule("opd_fan", FAN, TRACE, TraceCrossover),
    rule("longitudinal_aberration", &[Wavelengths, Num("/pupil_samples")], TRACE, TraceCrossover),
    rule("lateral_colour", &[Num("/field_samples")], TRACE, TraceCrossover),
    rule("field_curvature_distortion", &[Wavelengths, Num("/field_samples")], TRACE, TraceCrossover),
    rule("grid_distortion", &[Square("/grid_size")], TRACE, TraceCrossover),
    rule("chief_ray_angle", &[Product("/grid")], TRACE, TraceCrossover),
    rule("pupil_aberration", &[Product("/grid")], TRACE, TraceCrossover),
    rule("layout", &[Fields, Wavelengths, Num("/rays_per_field")], TRACE, TraceCrossover),
    rule("footprint", PUPIL_RAYS, 2043., Measured),
    rule("relative_illumination", &[Num("/field_samples"), Num("/radial_samples"), Num("/azimuth_samples")], 6336., Measured),
    rule("relative_illumination_map", &[Product("/grid"), Num("/radial_samples"), Num("/azimuth_samples")], 14400., Measured),
    rule("system_transmission", &[Fields, Wavelengths, Square("/pupil/n")], 9216., Measured),
    rule("polarization_pupil_map", &[Square("/grid")], 1089., Measured),
    rule("ghost_paths", &[Surfaces], 13., Measured),
    rule("ghost_irradiance", &[Surfaces, Square("/pupil_samples")], 29952., Measured),
    rule("stray_light", &[Num("/rays")], 20000., Measured),
    rule("wavefront_map", WAVE, 12675., Measured),
    rule("psf", WAVE, 38025., Measured),
    rule("mtf", WAVE, 38025., Measured),
    rule("line_spread", WAVE, 38025., Measured),
    rule("diffraction_encircled_energy", WAVE, 38025., Measured),
    rule("through_focus_mtf", &[Fields, Wavelengths, Num("/focus_samples"), WavePupil("/pupil")], 798525., Measured),
    rule("mtf_vs_field", &[Num("/field_samples"), Wavelengths, WavePupil("/pupil")], 139425., Measured),
    rule("image_simulation", &[Square("/optics/field_grid"), WavePupil("/optics/pupil")], 38025., Measured),
];

fn selected(v: &Value, all: usize) -> usize {
    match v {
        Value::Array(a) if !a.is_empty() => a.len(),
        Value::Object(o) => match o.get("type").and_then(Value::as_str) {
            Some("index" | "primary") => 1,
            Some("list") => o["indices"].as_array().map_or(all, Vec::len),
            _ => all,
        },
        _ => all,
    }
}

fn workload(measure: &[Factor], s: &Value, system: &OpticalSystem) -> f64 {
    let num = |p: &str| s.pointer(p).and_then(Value::as_f64).unwrap_or(1.);
    measure
        .iter()
        .map(|f| match *f {
            Fields => selected(&s["fields"], system.fields.fields.len()) as f64,
            Wavelengths => selected(&s["wavelengths"], system.wavelengths.wavelengths.len()) as f64,
            Pupil => {
                let r = num("/rings").max(1.);
                match s["pattern"].as_str() {
                    // Square grids of 2r + 1 points per side, clipped to the disk.
                    Some("square" | "dithered") => std::f64::consts::PI * (r + 0.5).powi(2),
                    _ => 1. + 3. * r * (r + 1.),
                }
            }
            Num(p) => num(p),
            Square(p) => num(p).powi(2),
            Product(p) => s
                .pointer(p)
                .and_then(Value::as_array)
                .map_or(1., |a| a.iter().filter_map(Value::as_f64).product()),
            WavePupil(p) => {
                let grid = num(&format!("{p}/grid"));
                if grid > 0. {
                    grid
                } else {
                    num(&format!("{p}/rays"))
                }
                .powi(2)
            }
            Surfaces => system.surfaces.len() as f64,
        })
        .product()
}

/// The automatic route for `analysis`: `"cpu_reference"` or
/// `"qualified_metal"`, with the rule, workload and threshold that chose it.
pub(crate) fn decision(
    analysis: &str,
    settings: &Value,
    system: &OpticalSystem,
    device: &DeviceInfo,
) -> Value {
    if device.kind != BackendKind::Metal || device.vendor != "Apple" {
        return json!({"decision":"qualified_metal","reason":"no cost evidence for this device","version":VERSION});
    }
    if matches!(
        analysis,
        "optimize" | "merit_evaluation" | "tolerance" | "glass_substitution"
    ) {
        return json!({"decision":"qualified_metal","reason":"per-candidate f64 checks removed; re-measure pending","version":VERSION});
    }
    let Some(rule) = RULES.iter().find(|r| r.analysis == analysis) else {
        return json!({"decision":"qualified_metal","reason":"no cost rule for this analysis","version":VERSION});
    };
    let workload = workload(rule.measure, settings, system).round();
    let cpu = rule.cpu_max.is_none_or(|max| workload <= max);
    json!({
        "decision": if cpu { "cpu_reference" } else { "qualified_metal" },
        "rule": match rule.basis {
            Crossover => "dense crossover",
            TraceCrossover => "ray_grid trace crossover",
            Measured => "measured sizes",
        },
        "workload": workload,
        "threshold": rule.cpu_max,
        "evidence": EVIDENCE,
        "version": VERSION,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::samples;
    use mklight_gpu::ComputeBackend;

    fn measured(analysis: &str, settings: Value) -> f64 {
        let rule = RULES.iter().find(|r| r.analysis == analysis).unwrap();
        workload(rule.measure, &settings, &samples::cooke_triplet())
    }

    #[test]
    fn workloads_follow_settings_and_system() {
        let spot = json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"hexapolar","rings":6});
        assert_eq!(measured("spot_diagram", spot), 9. * 127.);
        let dense = json!({"fields":{"type":"index","index":2},"wavelengths":{"type":"primary"},"pattern":"square","rings":512});
        assert!((measured("spot_diagram", dense) - 825_159.).abs() < 1.);
        let wave = json!({"fields":{"type":"list","indices":[0,1]},"wavelengths":{"type":"all"},"pupil":{"grid":0,"rays":65}});
        assert_eq!(measured("psf", wave.clone()), 6. * 65. * 65.);
        let mut explicit = wave;
        explicit["pupil"]["grid"] = json!(256);
        assert_eq!(measured("psf", explicit), 6. * 256. * 256.);
        assert_eq!(measured("ghost_paths", json!({})), 8.);
        assert_eq!(measured("chief_ray_angle", json!({"grid":[5,7]})), 35.);
        // The default sizes of every CPU-preferred rule stay within it.
        assert!(measured("footprint", json!({"pattern":"square","rings":8})) <= 2043.);
    }

    #[test]
    fn decisions_need_apple_evidence() {
        let mut apple = mklight_gpu::CpuF32Backend::new().info().clone();
        let emulation = apple.clone();
        apple.kind = BackendKind::Metal;
        apple.vendor = "Apple".into();
        let singlet = samples::singlet();
        let spot = |rings: u32| json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"square","rings":rings});
        let d = decision("spot_diagram", &spot(6), &singlet, &apple);
        assert_eq!(d["decision"], "cpu_reference", "{d}");
        assert_eq!(d["threshold"], 32768.);
        assert_eq!(
            decision("spot_diagram", &spot(64), &singlet, &apple)["decision"],
            "qualified_metal"
        );
        assert_eq!(
            decision("spot_diagram", &spot(6), &singlet, &emulation)["decision"],
            "qualified_metal"
        );
        assert_eq!(
            decision("optimize", &json!({}), &singlet, &apple)["decision"],
            "qualified_metal"
        );
        assert_eq!(
            decision("seidel", &json!({}), &singlet, &apple)["decision"],
            "qualified_metal"
        );
    }

    #[test]
    fn rules_cover_the_resident_catalogue() {
        for name in crate::analysis::gpu::GPU_ANALYSES {
            assert!(
                RULES.iter().any(|r| r.analysis == *name)
                    || matches!(
                        *name,
                        "optimize" | "merit_evaluation" | "tolerance" | "glass_substitution"
                    ),
                "{name}"
            );
        }
    }
}
