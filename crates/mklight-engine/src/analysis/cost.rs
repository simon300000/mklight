//! Automatic backend cost rules (GPU.md §9.2–9.3). `decide` is the single
//! entry point of the router, so a calibrated stage-level cost model can
//! replace this table without touching routing.
//!
//! Every GPU-capable analysis has one rule:
//!
//! * `KnownFailure`: the native output failed its f64 comparison on the
//!   measured evidence. `auto` runs the CPU reference at every size and on
//!   every device (the kernels and their precision do not depend on it).
//! * `Candidates`: candidate solvers; the reference was faster at every
//!   measured size.
//! * `Threshold`: a workload measured from the normalised settings and the
//!   system; at or below `cpu_max` the CPU reference, above it qualified
//!   Metal.
//!
//! Cost rules (`Candidates`, `Threshold`) carry Apple GPU evidence (M2 Pro);
//! other devices use qualified Metal. A decision that needs the device
//! before it is open is `Undecided`. An explicit backend bypasses the table.
use mklight_core::OpticalSystem;
use mklight_gpu::{BackendKind, DeviceInfo};
use serde_json::{Value, json};

pub(crate) const VERSION: u32 = 4;
const EVIDENCE: &str = "Apple M2 Pro: P11 quiet-host medians of 10 paired warm runs over 16 samples, and a complex 29-surface lens (6 fields, 5 wavelengths, real aiming) with every analysis on CPU/Metal (GPU.md §9.2)";

/// The device a decision is for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Device {
    /// An Apple GPU: the measured family.
    Apple,
    /// A device without cost evidence.
    Other,
    /// Not open yet, and its family is not known in advance.
    Unknown,
}

impl Device {
    pub(crate) fn of(info: &DeviceInfo) -> Self {
        if is_apple(info) {
            Device::Apple
        } else {
            Device::Other
        }
    }

    pub(crate) fn hinted(apple: Option<bool>) -> Self {
        match apple {
            Some(true) => Device::Apple,
            Some(false) => Device::Other,
            None => Device::Unknown,
        }
    }
}

pub(crate) fn is_apple(info: &DeviceInfo) -> bool {
    info.kind == BackendKind::Metal && info.vendor == "Apple"
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Route {
    Cpu,
    Metal,
    /// The rule needs the opened device.
    Undecided,
}

pub(crate) struct Decision {
    pub route: Route,
    /// The workload-sized threshold rules; they read the settings and the
    /// system's field and wavelength counts.
    pub sized: bool,
    /// Recorded as `BackendReport::cost_decision`.
    pub json: Value,
}

impl Decision {
    /// The reported reason for a CPU route.
    pub(crate) fn note(&self) -> &'static str {
        if self.json["rule"] == "known native failure" {
            "auto: the native path failed its f64 comparison on the measured evidence for this analysis"
        } else {
            "auto: the CPU reference is faster for this workload"
        }
    }
}

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
    /// Geometric mean, rounded down to a power of two, of the largest
    /// measured CPU-win and the smallest larger Metal-win workload over the
    /// P11 dense series and the complex lens.
    Crossover,
    /// Fans reached parity on the complex lens at 30,750 samples
    /// (CPU 45.5 / Metal 49.5 ms); trace-only analyses without any measured
    /// Metal win share that aimed-trace threshold.
    FanParity,
    /// CPU won at every measured size; the threshold is the largest such
    /// workload. Larger jobs are unmeasured and use qualified Metal.
    Measured,
}

enum Policy {
    KnownFailure(&'static str),
    Candidates,
    Threshold {
        measure: &'static [Factor],
        cpu_max: f64,
        basis: Basis,
    },
}
use Policy::*;

struct Rule {
    analysis: &'static str,
    policy: Policy,
}

const fn sized(
    analysis: &'static str,
    measure: &'static [Factor],
    cpu_max: f64,
    basis: Basis,
) -> Rule {
    Rule {
        analysis,
        policy: Threshold {
            measure,
            cpu_max,
            basis,
        },
    }
}
const fn failing(analysis: &'static str, evidence: &'static str) -> Rule {
    Rule {
        analysis,
        policy: KnownFailure(evidence),
    }
}
const fn candidates(analysis: &'static str) -> Rule {
    Rule {
        analysis,
        policy: Candidates,
    }
}

const PUPIL_RAYS: &[Factor] = &[Fields, Wavelengths, Pupil];
const FAN: &[Factor] = &[Fields, Wavelengths, Num("/samples")];
const WAVE: &[Factor] = &[Fields, Wavelengths, WavePupil("/pupil")];
/// The fan parity, shared by the trace-only analyses.
const TRACE: f64 = 32768.;
/// Dense geometric series: the complex lens kept the CPU ahead at 374,430
/// rays (ray grid 2.3×, best focus 2.0×); Metal led every P11 design at
/// 825,159 (spot 1.04–1.25×, ray grid 0.88–2.4×, best focus 3.8–4.4×) and the
/// complex lens' dense spot (23.6M, 2.1×).
const DENSE: f64 = 524_288.;

/// Thresholds and their evidence: GPU.md §9.2. Workloads in comments are
/// the measured CPU / Metal-DF warm medians.
#[rustfmt::skip]
const RULES: &[Rule] = &[
    sized("spot_diagram", PUPIL_RAYS, DENSE, Basis::Crossover),
    sized("ray_grid", PUPIL_RAYS, DENSE, Basis::Crossover),
    sized("best_focus", PUPIL_RAYS, DENSE, Basis::Crossover),
    // Complex lens: CPU 4.4× at 99,549 (140 / 619 ms); Metal 6.5–9.6× at
    // 825,159 on every P11 design and 5.1× at 24.75M on the complex lens.
    sized("geometric_mtf", PUPIL_RAYS, 262_144., Basis::Crossover),
    // Complex lens: CPU 1.7× at 6,809; P5 Cooke 823,473 rays Metal 2.0×.
    sized("footprint", PUPIL_RAYS, 65_536., Basis::Crossover),
    sized("ray_fan", FAN, TRACE, Basis::FanParity),
    sized("opd_fan", FAN, TRACE, Basis::FanParity),
    sized("longitudinal_aberration", &[Wavelengths, Num("/pupil_samples")], TRACE, Basis::FanParity),
    sized("lateral_colour", &[Num("/field_samples")], TRACE, Basis::FanParity),
    sized("layout", &[Fields, Wavelengths, Num("/rays_per_field")], TRACE, Basis::FanParity),
    // P11 defaults 7/8 mixed; complex lens CPU 22× at 6,336 (56 / 1267 ms).
    sized("relative_illumination", &[Num("/field_samples"), Num("/radial_samples"), Num("/azimuth_samples")], 6336., Basis::Measured),
    // P11 16/0; complex lens CPU 29× at 30,720 (21 / 603 ms).
    sized("system_transmission", &[Fields, Wavelengths, Square("/pupil/n")], 30_720., Basis::Measured),
    // Static capability (routing.rs) keeps every run on the reference.
    sized("encircled_energy", PUPIL_RAYS, DENSE, Basis::Crossover),
    // Static capability (routing.rs) keeps every run on the reference.
    sized("polarization_pupil_map", &[Square("/grid")], 1089., Basis::Measured),
    // Complex lens: 29 surfaces CPU 1.4× (3.9 / 5.3 ms).
    sized("ghost_paths", &[Surfaces], 29., Basis::Measured),
    // Complex lens: CPU 1.3× at 66,816 (17.9 / 24.0 s).
    sized("ghost_irradiance", &[Surfaces, Square("/pupil_samples")], 66_816., Basis::Measured),
    // Static capability (routing.rs) keeps every run on the reference.
    sized("wavefront_map", WAVE, 12675., Basis::Measured),
    // P11 9/5; the complex lens' native route failed (unusable chief).
    sized("image_simulation", &[Square("/optics/field_grid"), WavePupil("/optics/pupil")], 38025., Basis::Measured),
    failing("psf", "complex lens: Metal DF image 5.6e2× its f64 budget (and 8× slower)"),
    failing("mtf", "complex lens: Metal DF MTF 1.5e6× its f64 budget (and 8.6× slower)"),
    failing("line_spread", "complex lens: Metal DF LSF 1.5e2× its f64 budget (and 8.1× slower)"),
    failing("diffraction_encircled_energy", "complex lens: Metal DF 7.3e2× its f64 budget (and 2.6× slower)"),
    failing("through_focus_mtf", "complex lens: Metal DF 9.8e2× its f64 budget (and 2.2× slower)"),
    failing("mtf_vs_field", "complex lens: unresolved in-pupil aims and driver interactivity failures fell back to f64 (12× slower)"),
    failing("chief_ray_angle", "complex lens: 80 mask/discrete mismatches against f64 (and 54× slower)"),
    failing("pupil_aberration", "complex lens: 56 mask/discrete mismatches against f64 (and 53× slower)"),
    failing("relative_illumination_map", "complex lens: 20 mask/discrete mismatches against f64 (and 24× slower)"),
    failing("grid_distortion", "complex lens: 200 mask/discrete mismatches against f64 (and 76× slower)"),
    failing("field_curvature_distortion", "complex lens: close-ray focus 4.4× its f64 budget; P11 petzval_asphere residual gap"),
    failing("stray_light", "complex lens: sink energies 4.8e4× their f64 budget; 6/16 P11 samples outside the gates"),
    // Complex lens CPU 4–52× (merit 23 / 96 ms, optimise 286 / 5819 ms,
    // tolerance 78 / 3401 ms, glass 763 / 31715 ms); P11 16/0 each.
    candidates("optimize"),
    candidates("merit_evaluation"),
    candidates("tolerance"),
    candidates("glass_substitution"),
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

/// The automatic route for `analysis` with normalised `settings` on
/// `system`, and the record of the rule that chose it.
pub(crate) fn decide(
    analysis: &str,
    settings: &Value,
    system: &OpticalSystem,
    device: Device,
) -> Decision {
    let decision = |route, sized, json| Decision { route, sized, json };
    let Some(rule) = RULES.iter().find(|r| r.analysis == analysis) else {
        return decision(
            Route::Metal,
            false,
            json!({"decision":"qualified_metal","reason":"no cost rule for this analysis","version":VERSION}),
        );
    };
    if let KnownFailure(evidence) = rule.policy {
        return decision(
            Route::Cpu,
            false,
            json!({"decision":"cpu_reference","rule":"known native failure","evidence":evidence,"version":VERSION}),
        );
    }
    match device {
        Device::Unknown => return decision(Route::Undecided, false, Value::Null),
        Device::Other => {
            return decision(
                Route::Metal,
                false,
                json!({"decision":"qualified_metal","reason":"no cost evidence for this device","version":VERSION}),
            );
        }
        Device::Apple => (),
    }
    let Threshold {
        measure,
        cpu_max,
        basis,
    } = rule.policy
    else {
        return decision(
            Route::Cpu,
            false,
            json!({"decision":"cpu_reference","rule":"candidate solvers","evidence":EVIDENCE,"version":VERSION}),
        );
    };
    let workload = workload(measure, settings, system).round();
    let cpu = workload <= cpu_max;
    decision(
        if cpu { Route::Cpu } else { Route::Metal },
        true,
        json!({
            "decision": if cpu { "cpu_reference" } else { "qualified_metal" },
            "rule": match basis {
                Basis::Crossover => "dense crossover",
                Basis::FanParity => "fan parity",
                Basis::Measured => "measured sizes",
            },
            "workload": workload,
            "threshold": cpu_max,
            "evidence": EVIDENCE,
            "version": VERSION,
        }),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::samples;
    use mklight_gpu::ComputeBackend;

    fn measured(analysis: &str, settings: Value) -> f64 {
        let rule = RULES.iter().find(|r| r.analysis == analysis).unwrap();
        let Threshold { measure, .. } = rule.policy else {
            panic!("{analysis} has no workload")
        };
        workload(measure, &settings, &samples::cooke_triplet())
    }

    #[test]
    fn workloads_follow_settings_and_system() {
        let spot = json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"hexapolar","rings":6});
        assert_eq!(measured("spot_diagram", spot), 9. * 127.);
        let dense = json!({"fields":{"type":"index","index":2},"wavelengths":{"type":"primary"},"pattern":"square","rings":512});
        assert!((measured("spot_diagram", dense) - 825_159.).abs() < 1.);
        let wave = json!({"fields":{"type":"list","indices":[0,1]},"wavelengths":{"type":"all"},"pupil":{"grid":0,"rays":65}});
        assert_eq!(measured("wavefront_map", wave.clone()), 6. * 65. * 65.);
        let mut explicit = wave;
        explicit["pupil"]["grid"] = json!(256);
        assert_eq!(measured("wavefront_map", explicit), 6. * 256. * 256.);
        assert_eq!(measured("ghost_paths", json!({})), 8.);
        // The default sizes of every CPU-preferred rule stay within it.
        assert!(measured("footprint", json!({"pattern":"square","rings":8})) <= 65_536.);
    }

    #[test]
    fn decisions_follow_device_evidence() {
        let mut apple = mklight_gpu::CpuF32Backend::new().info().clone();
        let emulation = Device::of(&apple);
        apple.kind = BackendKind::Metal;
        apple.vendor = "Apple".into();
        let apple = Device::of(&apple);
        assert_eq!((apple, emulation), (Device::Apple, Device::Other));
        let singlet = samples::singlet();
        let spot = |rings: u32| json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"square","rings":rings});
        let route = |a: &str, s: &Value, d| decide(a, s, &singlet, d).route;
        let d = decide("spot_diagram", &spot(6), &singlet, apple);
        assert_eq!((d.route, d.sized), (Route::Cpu, true));
        assert_eq!(d.json["threshold"], DENSE);
        assert_eq!(route("spot_diagram", &spot(512), apple), Route::Metal);
        assert_eq!(route("spot_diagram", &spot(6), emulation), Route::Metal);
        assert_eq!(
            route("spot_diagram", &spot(6), Device::Unknown),
            Route::Undecided
        );
        // Candidate solvers: the reference on the measured family only.
        assert_eq!(route("optimize", &json!({}), apple), Route::Cpu);
        assert_eq!(route("optimize", &json!({}), emulation), Route::Metal);
        assert_eq!(
            route("optimize", &json!({}), Device::Unknown),
            Route::Undecided
        );
        // Known native failures: the reference at every size, on any device,
        // even before the device is known.
        for d in [apple, emulation, Device::Unknown] {
            let r = decide("mtf", &json!({"pupil":{"grid":2048}}), &singlet, d);
            assert_eq!((r.route, r.sized), (Route::Cpu, false));
            assert_eq!(r.json["rule"], "known native failure");
            assert!(r.note().contains("f64 comparison"));
        }
        assert_eq!(route("seidel", &json!({}), apple), Route::Metal);
    }

    #[test]
    fn rules_cover_the_resident_catalogue() {
        for name in crate::analysis::gpu::GPU_ANALYSES {
            assert!(RULES.iter().any(|r| r.analysis == *name), "{name}");
        }
        assert_eq!(
            RULES.len(),
            crate::analysis::gpu::GPU_ANALYSES.len(),
            "one rule per analysis"
        );
    }
}
