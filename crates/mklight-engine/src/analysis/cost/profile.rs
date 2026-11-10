//! Stage work of one analysis request, from its normalised settings and the
//! system alone (no tracing): the features of the cost model (`model`).
//!
//! Counts are proportional measures, not exact operation counts: each
//! analysis has its own fitted coefficients, so only the scaling with
//! settings and design matters (a constant factor is absorbed by the fit).
use mklight_core::{OpticalSystem, model::RayAiming};
use serde_json::Value;

use super::model::Features;

/// Device seed-table nodes and the generation size that builds one
/// (`mklight_gpu::aiming`).
const TABLE_NODES: f64 = mklight_gpu::aiming::TABLE_NODES as f64;
const TABLE_MIN_RAYS_PER_ROW: f64 = mklight_gpu::aiming::TABLE_MIN_RAYS_PER_ROW as f64;

/// Stage work of one request.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Profile {
    /// Field–wavelength rows prepared on the host in f64 (sources, chiefs
    /// and their Jacobians); shared through the `tracer` graph node.
    pub rows: f64,
    /// Rays aimed (with real aiming) and traced through the system.
    pub rays: f64,
    /// Transform/reduction work: samples × log₂ samples for transforms,
    /// samples × frequencies for direct sums.
    pub transform: f64,
    /// Bulk output bytes (typed buffers).
    pub output: f64,
    /// The ray work is a shared CPU pupil lattice (`pupil_lattice` node).
    pub lattice: bool,
}

/// What the system contributes to ray work.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SystemShape {
    pub surfaces: f64,
    /// Surfaces traced per aim evaluation (object to stop).
    pub stop: f64,
    /// Real ray aiming.
    pub aimed: bool,
}

impl SystemShape {
    pub fn of(system: &OpticalSystem) -> Self {
        SystemShape {
            surfaces: system.surfaces.len().max(2) as f64,
            stop: (system.stop.max(1)) as f64,
            aimed: matches!(system.ray_aiming, RayAiming::Real { .. }),
        }
    }
}

impl Profile {
    /// Model features with `traces` stop traces per aimed ray (the aiming
    /// difficulty; ignored without real aiming).
    pub fn features(&self, shape: &SystemShape, traces: f64) -> Features {
        let aim = if shape.aimed { traces * shape.stop } else { 0. };
        // Aimed geometric generations of at least `TABLE_MIN_RAYS_PER_ROW`
        // rays per row build a device seed table (wave pupils use host f64
        // tables instead).
        let per_row = self.rays / self.rows.max(1.);
        let table = if shape.aimed && !self.lattice && per_row >= TABLE_MIN_RAYS_PER_ROW {
            self.rows * TABLE_NODES * aim
        } else {
            0.
        };
        [
            1.,
            self.rows * shape.surfaces,
            self.rays * (shape.surfaces + aim),
            self.transform,
            self.output,
            table,
        ]
    }
}

fn selected(v: &Value, all: usize) -> f64 {
    (match v {
        Value::Array(a) if !a.is_empty() => a.len(),
        Value::Object(o) => match o.get("type").and_then(Value::as_str) {
            Some("index" | "primary") => 1,
            Some("list") => o["indices"].as_array().map_or(all, Vec::len),
            _ => all,
        },
        _ => all,
    })
    .max(1) as f64
}

/// Pupil points of a geometric pattern.
fn pattern(s: &Value, rings: f64) -> f64 {
    let r = rings.max(1.);
    match s["pattern"].as_str() {
        Some("square" | "dithered") => std::f64::consts::PI * (r + 0.5).powi(2),
        _ => 1. + 3. * r * (r + 1.),
    }
}

fn lg(n: f64) -> f64 {
    n.max(2.).log2()
}

/// Stage work of `analysis` with normalised `settings` on `system`; `None`
/// for analyses without a GPU path.
pub fn profile(analysis: &str, s: &Value, system: &OpticalSystem) -> Option<Profile> {
    profile_with(analysis, s, system, None)
}

/// [`profile`] with the automatic wave pupil grid an earlier call decided
/// (`None`: twice the lattice, a power of two in 64..1024).
pub fn profile_with(
    analysis: &str,
    s: &Value,
    system: &OpticalSystem,
    decided_grid: Option<f64>,
) -> Option<Profile> {
    let all_f = system.fields.fields.len().max(1);
    let all_w = system.wavelengths.wavelengths.len().max(1);
    let num = |p: &str, d: f64| s.pointer(p).and_then(Value::as_f64).unwrap_or(d);
    // A singular `field` / `wavelength` setting selects one.
    let f = if s.get("field").is_some() {
        1.
    } else {
        selected(&s["fields"], all_f)
    };
    let w = if s.get("wavelength").is_some() {
        1.
    } else {
        selected(&s["wavelengths"], all_w)
    };
    let fw = f * w;
    let surfaces = system.surfaces.len().max(2) as f64;
    // Wave pupils: lattice rays (a quarter of the guard-ringed square is
    // culled) and the resampled grid (automatic grids are at least twice
    // the lattice, a power of two, at most 1024).
    let pupil = |p: &str| {
        let rays = num(&format!("{p}/rays"), 65.);
        let grid = num(&format!("{p}/grid"), 0.);
        let n = if grid > 0. {
            grid
        } else {
            decided_grid.unwrap_or_else(|| (2. * rays).log2().ceil().exp2().clamp(64., 1024.))
        };
        (0.8 * rays * rays, n)
    };
    let mut p = Profile::default();
    match analysis {
        "spot_diagram" | "ray_grid" | "encircled_energy" | "footprint" | "geometric_mtf"
        | "best_focus" => {
            p.rows = fw;
            p.rays = fw * pattern(s, num("/rings", 6.));
            match analysis {
                "spot_diagram" if s["include_points"].as_bool() == Some(true) => {
                    p.output = p.rays * 24.
                }
                "ray_grid" => p.output = p.rays * 56.,
                "footprint" => p.output = p.rays * 16.,
                "geometric_mtf" => p.transform = p.rays * num("/frequency_samples", 41.) * 2.,
                "encircled_energy" => p.transform = p.rays * lg(p.rays),
                "best_focus" => p.transform = p.rays * num("/max_evaluations", 200.).min(64.),
                _ => (),
            }
        }
        "ray_fan" | "opd_fan" => {
            p.rows = fw;
            p.rays = fw * num("/samples", 41.) * 2.;
        }
        "layout" => {
            p.rows = fw;
            p.rays = fw * num("/rays_per_field", 7.);
            p.output = surfaces * num("/profile_samples", 41.) * 16.
                + surfaces * num("/rings", 3.) * num("/ring_samples", 64.) * 24.;
        }
        "field_curvature_distortion" | "lateral_colour" => {
            p.rows = num("/field_samples", 21.) * w;
            p.rays = p.rows;
        }
        "longitudinal_aberration" => {
            p.rows = w;
            p.rays = w * num("/pupil_samples", 21.) * 2.;
        }
        "grid_distortion" => {
            let n = num("/grid_size", 11.);
            p.rows = n * n;
            p.rays = n * n * w;
            p.output = n * n * w * 32.;
        }
        "relative_illumination" => {
            p.rows = num("/field_samples", 11.) * w;
            p.rays = p.rows * num("/radial_samples", 12.) * num("/azimuth_samples", 48.);
        }
        "pupil_aberration" | "relative_illumination_map" | "chief_ray_angle" => {
            // The map's nodes: the grid, or one of its axes for a sweep.
            let side = |axis: usize| s["grid"][axis].as_f64().unwrap_or(5.);
            let g = match s["mode"].as_str() {
                Some("x") => side(0),
                Some("y") => side(1),
                _ => side(0) * side(1),
            };
            p.rows = g * w;
            p.rays = match analysis {
                "relative_illumination_map" => {
                    p.rows * num("/radial_samples", 12.) * num("/azimuth_samples", 48.)
                }
                "pupil_aberration" => p.rows * 16.,
                _ => p.rows,
            };
            p.output = p.rows * 64.;
        }
        "system_transmission" => {
            // In nodes of the pupil grid's square (n², the unit of the
            // fit): the disk keeps π/4 of them; a random pupil traces
            // `count` rays.
            let nodes = match s["pupil"]["type"].as_str() {
                Some("random") => num("/pupil/count", 1024.) / std::f64::consts::FRAC_PI_4,
                _ => num("/pupil/n", 32.).powi(2),
            };
            p.rows = fw;
            p.rays = fw * nodes;
        }
        "polarization_pupil_map" => {
            let n = num("/grid", 33.);
            p.rows = fw;
            p.rays = fw * n * n;
            p.output = p.rays * 64.;
        }
        "wavefront_map"
        | "psf"
        | "mtf"
        | "line_spread"
        | "diffraction_encircled_energy"
        | "through_focus_mtf"
        | "mtf_vs_field" => {
            let (lattice, n) = pupil("/pupil");
            let pupils = if analysis == "mtf_vs_field" {
                num("/field_samples", 11.) * w
            } else {
                fw
            };
            p.rows = pupils;
            p.rays = pupils * lattice;
            p.lattice = true;
            let freqs = s["frequencies"]
                .as_array()
                .filter(|a| !a.is_empty())
                .map_or(num("/frequency_samples", 41.), |a| a.len() as f64);
            let image = num("/grid_size", 0.).max(n);
            p.transform = pupils
                * match analysis {
                    "wavefront_map" => n * n,
                    "mtf" if s["method"].as_str() == Some("fft") => n * n * lg(n) * 4.,
                    "mtf" | "mtf_vs_field" => n * n * freqs * 2.,
                    "through_focus_mtf" => n * n * freqs * num("/focus_samples", 21.),
                    _ => image * image * lg(image) * 4.,
                };
            p.output = match analysis {
                "psf" => f * image * image * 8.,
                "wavefront_map" => pupils * n * n * 8.,
                _ => 0.,
            };
        }
        "image_simulation" => {
            let g = num("/optics/field_grid", 3.);
            let (lattice, n) = pupil("/optics/pupil");
            p.rows = g * g * w;
            p.rays = p.rows * lattice;
            p.lattice = true;
            p.transform = p.rows * 4. * n * n * lg(n);
        }
        "ghost_paths" => {
            p.rows = w;
            p.rays = surfaces * surfaces * w;
        }
        "ghost_irradiance" => {
            let n = num("/pupil_samples", 48.);
            p.rows = fw;
            p.rays = fw * surfaces * surfaces * 0.5 * n * n;
            p.output = surfaces * surfaces * 0.5 * 8.;
        }
        "stray_light" => {
            p.rows = w;
            p.rays = num("/rays", 2e4).max(1.) * 4.;
        }
        "merit_evaluation" | "optimize" | "tolerance" | "glass_substitution" => {
            let vars = s["variables"].as_array().map_or(0., |v| v.len() as f64);
            let evaluations = match analysis {
                "optimize" => num("/max_iterations", 50.).min(1000.) * (vars + 1.),
                "tolerance" => num("/trials", 100.) + 2. * surfaces,
                "glass_substitution" => {
                    num("/max_combinations", 20.) * num("/local_iterations", 5.) * (vars + 1.)
                }
                _ => 1.,
            };
            p.rows = fw * evaluations;
            // Default merit rays per field and wavelength (3 rings × 6 arms).
            p.rays = p.rows * 19.;
        }
        _ => return None,
    }
    Some(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::samples;
    use serde_json::json;

    #[test]
    fn work_follows_settings_and_system() {
        let cooke = samples::cooke_triplet();
        let spot = |rings: u32| {
            profile("spot_diagram", &json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"hexapolar","rings":rings}), &cooke).unwrap()
        };
        assert_eq!(spot(6).rays, 9. * 127.);
        assert_eq!(spot(6).rows, 9.);
        assert!(spot(24).rays > 14. * spot(6).rays);
        let dense = profile("spot_diagram", &json!({"fields":{"type":"index","index":2},"wavelengths":{"type":"primary"},"pattern":"square","rings":512}), &cooke).unwrap();
        assert!((dense.rays - 825_159.).abs() < 1.);
        let wave = |grid: u32| {
            profile("psf", &json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pupil":{"grid":grid,"rays":65}}), &cooke).unwrap()
        };
        assert!(wave(0).lattice && wave(512).transform > 4. * wave(0).transform);
        assert_eq!(wave(0).rays, wave(512).rays);
        let shape = SystemShape::of(&cooke);
        assert!(!shape.aimed);
        let f = spot(6).features(&shape, 4.);
        assert_eq!(f[2], 9. * 127. * shape.surfaces);
        assert!(profile("seidel", &json!({}), &cooke).is_none());
        // A random transmission pupil counts its rays, in the grid's unit.
        let transmission = |pupil: Value| {
            profile(
                "system_transmission",
                &json!({"fields":[],"wavelengths":[],"pupil":pupil}),
                &cooke,
            )
            .unwrap()
            .rays
        };
        let grid = transmission(json!({"type":"grid","n":32}));
        assert_eq!(grid, 9. * 1024.);
        let random = transmission(json!({"type":"random","count":1_000_000,"seed":1}));
        assert!((random / grid - 1e6 / (1024. * std::f64::consts::FRAC_PI_4)).abs() < 1e-9);
        // A field map sweeps one axis in the x and y modes.
        let map = |mode: &str| {
            profile(
                "chief_ray_angle",
                &json!({"grid":[21,5],"mode":mode,"wavelength":0}),
                &cooke,
            )
            .unwrap()
            .rows
        };
        assert_eq!((map("grid"), map("x"), map("y")), (105., 21., 5.));
    }
}
