//! Offline accuracy contract (GPU §9.4 "Contracts"): every result field of
//! every compared analysis is mapped to a quantity class in [`FIELDS`], and
//! each class carries its own tolerance, derived from the CONVENTIONS §7 /
//! GPU §9.4 budgets. Offline qualification gates at one third of the budget
//! (the sampled margin); the statistical class gates at its stated
//! confidence. A numeric field the table does not map fails loudly.
//!
//! Data that live on a sampling axis or grid ([`Rule::axis`]) are compared at
//! the same physical coordinates: index by index where both sides sampled the
//! same coordinates (within the axis class), otherwise after resampling the
//! denser result onto the sparser one's coordinates (linear in 1-D, bilinear
//! on regular 2-D grids). Outlines and loops are compared by their Hausdorff
//! distance, not by vertex count.
#![allow(dead_code)]
use mklight_core::JobContext;
use mklight_engine::{analysis::AnalysisOutput, buffer::BufferData, error::EngineResult};
use serde_json::Value;
use std::collections::BTreeMap;

/// Quantity classes. `budget` gives the full tolerance; the deterministic
/// classes gate at a third of it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Class {
    /// Ray, centroid, focus and image positions; lengths (mm). §7: 2e-5 mm.
    Position,
    /// The difference of two positions (geometric spot radii). §9.4 R: 4e-5 mm.
    PositionDifference,
    /// Direction cosines, numerical aperture, `sin_max`. §7: 2e-6.
    Direction,
    /// Spatial frequencies (cycles/mm): `n sinθ / λ`, so the §7 direction
    /// budget gives a relative 2e-6 / (n sinθ) ≥ 2e-6; 2e-6 relative.
    Frequency,
    /// Working F-numbers, `1 / (2 n sinθ)`: relative 2e-6, as `Frequency`.
    FNumber,
    /// OPD, Zernike and Seidel terms, wavefront RMS/PV (waves). §7: 5e-3.
    Waves,
    /// MTF values, Strehl ratios, encircled/ensquared fractions. §7, §9.4 R/E: 2e-3.
    Modulation,
    /// RMS spot radii (mm). §7: max(0.5 % of the reference, 2e-5 mm).
    Rms,
    /// Energy-curve radii r50/r80/r90: §7 position, at most one radial sample.
    Quantile,
    /// Angles in degrees (chief-ray angles, incidence). §9.4 D: 2e-4°.
    Angle,
    /// Orientation of a wavefront term (degrees, periodic): wrapped
    /// difference ≤ 2e-4°; undefined (not compared) below 1e-3 waves.
    Orientation(f64),
    /// Pupil magnification and its variation. §9.4 D: 1e-3 relative or 1e-5.
    Magnification,
    /// Projected solid angles and integrated PSF energy: 1e-3 relative (§9.4 E).
    Relative1e3,
    /// Relative illumination. §9.4 D: 1e-3 absolute.
    Illumination,
    /// Fluxes, transmissions, weights: §9.4 E / GHOST §6, 1e-5 relative + 1e-10.
    Radiometric,
    /// Stray-light energy ledger entries: 1e-5 of the emitted power + 1e-10.
    Ledger,
    /// Diattenuation, polarised intensity, principal transmissions: §9.4 E 1e-4.
    Polarimetric,
    /// Retardance (rad). §9.4 E: 1e-3 rad.
    Retardance,
    /// Fast axis of a retarder (unit Stokes vector): compared where the
    /// reference retarder is defined and its retardance ≥ the 1e-3 rad
    /// budget, at 2e-5 per component.
    FastAxis,
    /// Dimensionless geometry without a derived budget (distortion percent,
    /// clear-area and vignetting fractions, normalised pupil coordinates):
    /// the generic 2e-5 absolute, unchanged.
    Generic,
    /// Fit condition numbers: a diagnostic of the least-squares problem, not
    /// an optical output; 1e-3 relative.
    Diagnostic,
    /// Solver merits. §9.4 D: 1 % or 1e-8 (violations keep the generic
    /// budget; feasibility, the §9.4 D criterion, is a label compared exactly).
    Merit,
    /// Counts, indices, statuses and fractions that follow deterministically
    /// from survival, and echoed inputs: equal.
    Exact,
    /// Monte-Carlo counts under Russian roulette: |a − b| ≤ 5 combined
    /// Poisson standard errors, 5·√(a + b) (§9.4 E stray contract).
    Statistical,
    /// Monte-Carlo class fluxes and image moments: |a − b| ≤ 5 combined
    /// standard errors, 5·√(σa² + σb²), with σ the result's own standard
    /// error at the named path (same indices) (§9.4 E stray contract).
    MonteCarlo(&'static str),
    /// Fractions of ray counts (vignetted, unvignetted): survival exact. One
    /// ray changes the fraction by at least 1/N ≥ 1e-9, so 1e-12 absolute
    /// tells any count change from the rounding of a different division.
    CountRatio,
    /// `[offset, count]` of a convex outline in a vertex buffer, compared by
    /// the Hausdorff distance of the outlines ≤ the §7 position budget.
    Outline(&'static str),
    /// A closed loop buffer `[k, d]` compared by Hausdorff distance at the
    /// `Generic` budget (normalised pupil coordinates).
    Loop,
    /// Vertices referenced by an `Outline`; compared through the outlines.
    OutlineVertices,
    /// Images and profiles: normalised L1 ≤ 2e-3, integrated energy ≤ 1e-3 (§9.4 E).
    Image,
    /// Jones matrices `[N, 2, 2, 2]`: Frobenius error ≤ 1e-4‖J‖ + 1e-7/√2 (§9.4 E).
    Jones,
    /// A backend or request sampling choice (pupil and image grid sizes,
    /// spacings): recorded, not gated; the data it samples are compared at
    /// common physical coordinates.
    Sampling,
    /// The value of a first-order table entry `{key, unit, value}`, classed
    /// by its key and unit ([`entry_class`]).
    Entry,
}

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::Orientation(_) => "orientation",
            Class::Outline(_) => "outline",
            Class::MonteCarlo(_) => "monte_carlo",
            Class::Position => "position",
            Class::PositionDifference => "position_difference",
            Class::Direction => "direction",
            Class::Frequency => "frequency",
            Class::FNumber => "f_number",
            Class::Waves => "waves",
            Class::Modulation => "modulation",
            Class::Rms => "rms",
            Class::Quantile => "quantile",
            Class::Angle => "angle",
            Class::Magnification => "magnification",
            Class::Relative1e3 => "relative_1e-3",
            Class::Illumination => "illumination",
            Class::Radiometric => "radiometric",
            Class::Ledger => "ledger",
            Class::Polarimetric => "polarimetric",
            Class::Retardance => "retardance",
            Class::FastAxis => "fast_axis",
            Class::Generic => "generic",
            Class::Diagnostic => "diagnostic",
            Class::Merit => "merit",
            Class::Exact => "exact",
            Class::Statistical => "statistical",
            Class::CountRatio => "count_ratio",
            Class::Loop => "loop",
            Class::OutlineVertices => "outline_vertices",
            Class::Image => "image",
            Class::Jones => "jones",
            Class::Sampling => "sampling",
            Class::Entry => "entry",
        }
    }

    /// Full tolerance for reference value `r` (and test value `t`).
    fn budget(self, r: f64, t: f64, ctx: &Scale) -> f64 {
        match self {
            Class::Position | Class::FastAxis | Class::Generic | Class::Loop => 2e-5,
            Class::Outline(_) => 2e-5,
            Class::PositionDifference => 4e-5,
            Class::Direction => 2e-6,
            Class::Frequency | Class::FNumber => 2e-6 * r.abs(),
            Class::Waves => 5e-3,
            Class::Modulation => 2e-3,
            Class::Rms => (0.005 * r.abs()).max(2e-5),
            Class::Quantile => ctx.interval.map_or(2e-5, |step| step.min(2e-5)),
            Class::Angle | Class::Orientation(_) => 2e-4,
            Class::Magnification => (1e-3 * r.abs()).max(1e-5),
            Class::Relative1e3 => 1e-3 * r.abs(),
            Class::Illumination => 1e-3,
            Class::Radiometric => 1e-5 * r.abs() + 1e-10,
            Class::Ledger => 1e-5 * ctx.emitted.unwrap_or(r.abs()) + 1e-10,
            Class::Polarimetric => 1e-4,
            Class::Retardance => 1e-3,
            Class::Diagnostic => 1e-3 * r.abs(),
            Class::Merit => (0.01 * r.abs()).max(1e-8),
            Class::Statistical => 5. * (r.abs() + t.abs()).sqrt(),
            // The standard errors arrive through `Scale::sigma`.
            Class::MonteCarlo(_) => 5. * ctx.sigma + 1e-12,
            Class::CountRatio => 3e-12,
            Class::Exact
            | Class::OutlineVertices
            | Class::Image
            | Class::Jones
            | Class::Sampling
            | Class::Entry => 0.,
        }
    }

    /// The gate: a third of the budget (the sampled margin), except for the
    /// statistical classes, whose budgets are already confidence bounds.
    fn gate(self, r: f64, t: f64, ctx: &Scale) -> f64 {
        let b = self.budget(r, t, ctx);
        if matches!(self, Class::Statistical | Class::MonteCarlo(_)) {
            b
        } else {
            b / 3.
        }
    }
}

/// Scales a class tolerance may depend on.
#[derive(Default, Clone, Copy)]
struct Scale {
    /// Emitted power of a stray-light run (the ledger scale).
    emitted: Option<f64>,
    /// Smallest radial sample interval of an energy curve.
    interval: Option<f64>,
    /// Combined standard error √(σa² + σb²) of a Monte-Carlo value.
    sigma: f64,
}

/// One table entry. `axis` names the sampling coordinate of the data (a
/// JSON array in the same path form with the same indices, or a buffer
/// `@name` with `*` replaced by the data buffer's own index), or
/// `PSF_GRID` / `@coordinates_*` for the 2-D grids of PSFs and pupil maps.
#[derive(Clone, Copy)]
pub struct Rule {
    pub path: &'static str,
    pub class: Class,
    pub axis: Option<&'static str>,
}
const fn r(path: &'static str, class: Class) -> Rule {
    Rule {
        path,
        class,
        axis: None,
    }
}
const fn on(path: &'static str, class: Class, axis: &'static str) -> Rule {
    Rule {
        path,
        class,
        axis: Some(axis),
    }
}
/// The regular image grid of a PSF (`center_mm`, `spacing_mm`, `size` and
/// `reference_mm` of the object that names the buffer).
pub const PSF_GRID: &str = "psf-grid";

use Class::*;

/// Result keys that are execution telemetry, never compared.
pub const TELEMETRY: &[&str] = &[
    "run_info",
    "run",
    "warnings",
    "evaluations",
    "iterations",
    "duration_ms",
];

/// Every numeric result field and every buffer of every compared analysis
/// (GPU path, solvers excluded). Paths drop array indices (`fields[].cutoff`,
/// array elements end in `[]`); buffers are `@name` with `*` for indices;
/// columns of a `values` table are `@values.<column>`. Strings, booleans
/// and nulls are labels and compared exactly; integers are exact unless
/// their class is statistical, an outline or a sampling choice.
pub const FIELDS: &[(&str, &[Rule])] = &[
    (
        "best_focus",
        &[
            r("defocus", Position),
            r("field_rms[]", Rms),
            r("field_rms_at_zero[]", Rms),
            r("field_values[][]", Exact),
            r("fields[]", Exact),
            r("mean_square_defocus", Generic),
            r("merit", Rms),
            r("merit_at_zero", Rms),
            r("range[]", Position),
            r("through_focus_defocus[]", Position),
            on("through_focus_merit[]", Rms, "through_focus_defocus[]"),
        ],
    ),
    (
        "chief_ray_angle",
        &[
            r("shape[]", Exact),
            r("wavelength", Exact),
            r("@field_coordinates", Exact),
            r("@image_mm", Position),
            r("@values.incidence", Angle),
            r("@values.angle_x", Angle),
            r("@values.angle_y", Angle),
            r("@values.limit", Angle),
            r("@values.margin", Angle),
        ],
    ),
    (
        "diffraction_encircled_energy",
        &[
            r("fields[].center_mm[]", Position),
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            r("fields[].grid_size", Sampling),
            r("fields[].pupil_sampling[].grid", Sampling),
            r("fields[].pupil_sampling[].wavelength", Exact),
            r("fields[].r50_mm", Quantile),
            r("fields[].r80_mm", Quantile),
            r("fields[].r90_mm", Quantile),
            r("fields[].spacing_mm", Sampling),
            on("@diffraction_limit_*", Modulation, "@radius_mm"),
            on("@encircled_*", Modulation, "@radius_mm"),
            on("@ensquared_*", Modulation, "@radius_mm"),
            r("@radius_mm", Position),
        ],
    ),
    (
        "encircled_energy",
        &[
            on("fields[].encircled[]", Modulation, "radius[]"),
            on("fields[].ensquared[]", Modulation, "radius[]"),
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            r("fields[].half_width80", Quantile),
            r("fields[].r50", Quantile),
            r("fields[].r80", Quantile),
            r("fields[].r90", Quantile),
            r("fields[].rays_arrived", Exact),
            r("radius[]", Position),
            r("wavelengths[]", Exact),
        ],
    ),
    (
        "field_curvature_distortion",
        &[
            on(
                "curves[].distortion_percent[]",
                Generic,
                "normalized_field[]",
            ),
            on(
                "curves[].f_tan_theta_distortion_percent[]",
                Generic,
                "normalized_field[]",
            ),
            on(
                "curves[].f_theta_distortion_percent[]",
                Generic,
                "normalized_field[]",
            ),
            on("curves[].paraxial_height[]", Position, "normalized_field[]"),
            on("curves[].real_height[]", Position, "normalized_field[]"),
            on("curves[].sagittal[]", Position, "normalized_field[]"),
            on(
                "curves[].sagittal_close_rays[]",
                Position,
                "normalized_field[]",
            ),
            on("curves[].tangential[]", Position, "normalized_field[]"),
            on(
                "curves[].tangential_close_rays[]",
                Position,
                "normalized_field[]",
            ),
            r("curves[].wavelength", Exact),
            r("curves[].wavelength_um", Exact),
            r("field_coordinates[][]", Exact),
            r("field_values[]", Exact),
            r("normalized_field[]", Exact),
            r("wavelengths_um[]", Exact),
        ],
    ),
    (
        "footprint",
        &[
            r("extent[]", Position),
            r("field_values[][]", Exact),
            r("max_radius", Position),
            r("overlays[].field", Exact),
            r("overlays[].outline[]", Outline("@global_outlines")),
            r("overlays[].points[]", Exact),
            r("overlays[].surface", Exact),
            r("overlays[].wavelength", Exact),
            r("semi_diameter", Position),
            r("sets[].clipped[]", Exact),
            r("sets[].field", Exact),
            r("sets[].launched", Exact),
            r("sets[].lost_before", Exact),
            r("sets[].points[]", Exact),
            r("sets[].wavelength", Exact),
            r("surface", Exact),
            r("@clipped", Position),
            r("@global_outlines", OutlineVertices),
            r("@global_points", Position),
            r("@points", Position),
        ],
    ),
    (
        "geometric_mtf",
        &[
            r("curves[].field", Exact),
            on("curves[].modulation[]", Modulation, "frequencies[]"),
            r("cutoff_frequency", Frequency),
            on("diffraction_limit[]", Modulation, "frequencies[]"),
            r("field_values[][]", Exact),
            r("frequencies[]", Frequency),
        ],
    ),
    (
        "ghost_irradiance",
        &[
            r("coverage.apertures[].radius", Position),
            r("coverage.paths[][]", Exact),
            r("ghosts[].fields[].centroid[]", Position),
            r("ghosts[].fields[].field.index", Exact),
            r("ghosts[].fields[].flux", Radiometric),
            r("ghosts[].fields[].flux_on_map", Radiometric),
            r("ghosts[].fields[].peak", Radiometric),
            r("ghosts[].fields[].rays_detected", Exact),
            r("ghosts[].fields[].rays_traced", Exact),
            r("ghosts[].fields[].rms_radius", Rms),
            r("ghosts[].flux", Radiometric),
            r("ghosts[].flux_ratio", Radiometric),
            r("ghosts[].map_index", Exact),
            r("ghosts[].path[]", Exact),
            r("ghosts[].peak_ratio", Radiometric),
            r("map.bins[]", Exact),
            r("map.half_size[]", Position),
            r("primary.fields[].centroid[]", Position),
            r("primary.fields[].field.index", Exact),
            r("primary.fields[].flux", Radiometric),
            r("primary.fields[].flux_on_map", Radiometric),
            r("primary.fields[].peak", Radiometric),
            r("primary.fields[].rays_detected", Exact),
            r("primary.fields[].rays_traced", Exact),
            r("primary.fields[].rms_radius", Rms),
            r("primary.flux", Radiometric),
            r("primary.flux_ratio", Radiometric),
            r("primary.map_index", Exact),
            r("primary.path[]", Exact),
            r("primary.peak_ratio", Radiometric),
            r("total_flux_ratio", Radiometric),
            r("total_peak_ratio", Radiometric),
            r("@primary_map", Image),
            r("@total_ghost_map", Image),
            r("@path_maps", Image),
        ],
    ),
    (
        "ghost_paths",
        &[
            r("coverage.apertures[].radius", Position),
            r("coverage.paths[][]", Exact),
            r("ghosts[].focus_from_sensor", Position),
            r("ghosts[].image_height", Position),
            r("ghosts[].path[]", Exact),
            r("ghosts[].reflectance_product", Radiometric),
            r("ghosts[].relative_irradiance", Radiometric),
            r("ghosts[].spot_diameter", Position),
            r("ghosts[].transmittance", Radiometric),
            r("ghosts[].vignetting", Generic),
            r("primary_focus_from_sensor", Position),
            r("primary_image_height", Position),
            r("primary_transmittance", Radiometric),
            r("signal_diameter", Position),
            r("wavelength", Exact),
        ],
    ),
    (
        "grid_distortion",
        &[
            r("grid_size", Exact),
            r("max_abs_distortion_percent", Generic),
            r("wavelength", Exact),
            r("@distortion_percent", Generic),
            r("@field", Exact),
            r("@paraxial", Position),
            r("@real", Position),
        ],
    ),
    (
        "image_simulation",
        &[
            r("channels", Exact),
            r("field_grid", Exact),
            r("field_mapping[].field[]", Exact),
            r("field_mapping[].ideal_mm[]", Position),
            r("field_mapping[].pupil_sampling[].grid", Sampling),
            r("field_mapping[].pupil_sampling[].wavelength", Exact),
            r("field_mapping[].real_mm[]", Position),
            r("field_mapping[].relative_illumination", Illumination),
            r("origin_mm[]", Position),
            r("pixel_size_mm[]", Position),
            r("psf_size", Sampling),
            r("psf_spacing_mm", Sampling),
            r("size[]", Exact),
            r("@image", Image),
        ],
    ),
    (
        "lateral_colour",
        &[
            r("airy_radius", Position),
            on("curves[].offset[]", Position, "normalized_field[]"),
            r("curves[].wavelength", Exact),
            r("curves[].wavelength_um", Exact),
            r("field_coordinates[][]", Exact),
            r("field_values[]", Exact),
            on("lateral_colour[]", Position, "normalized_field[]"),
            r("long_wavelength", Exact),
            r("normalized_field[]", Exact),
            r("short_wavelength", Exact),
        ],
    ),
    (
        "layout",
        &[
            r("bounds.max[]", Position),
            r("bounds.min[]", Position),
            r("elements[].first_surface", Exact),
            r("elements[].last_surface", Exact),
            r("elements[].mechanical_semi_diameter", Position),
            r("elements[].outline_polyline", Exact),
            r("field_count", Exact),
            r("field_values[][]", Exact),
            r("mesh_triangle_count", Exact),
            r("origin[]", Position),
            r("polyline_count", Exact),
            r("surfaces[].axis[]", Direction),
            r("surfaces[].id", Exact),
            r("surfaces[].index", Exact),
            r("surfaces[].mechanical_semi_diameter", Position),
            r("surfaces[].semi_diameter", Position),
            r("surfaces[].vertex[]", Position),
            r("vertex_count", Exact),
            r("wavelengths_um[]", Exact),
            r("@fields", Exact),
            r("@flags", Exact),
            r("@kinds", Exact),
            r("@mesh_surfaces", Exact),
            r("@offsets", Exact),
            r("@surface_mesh", Position),
            r("@surfaces", Exact),
            r("@vertices", Position),
            r("@vertices_f64", Position),
            r("@wavelengths", Exact),
        ],
    ),
    (
        "line_spread",
        &[
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            r("fields[].pupil_sampling[].grid", Sampling),
            r("fields[].pupil_sampling[].wavelength", Exact),
            r("@positions_mm_*", Position),
            on("@x_*", Image, "@positions_mm_*"),
            on("@y_*", Image, "@positions_mm_*"),
        ],
    ),
    (
        "longitudinal_aberration",
        &[
            on("curves[].aberration[]", Position, "pupil[]"),
            on("curves[].focus[]", Position, "pupil[]"),
            r("curves[].paraxial_focus", Position),
            r("curves[].wavelength", Exact),
            r("curves[].wavelength_um", Exact),
            r("primary_paraxial_focus", Position),
            r("pupil[]", Exact),
        ],
    ),
    (
        "mtf",
        &[
            r("fields[].cutoff", Frequency),
            r("fields[].detector.nyquist_sagittal", Frequency),
            r("fields[].detector.nyquist_tangential", Frequency),
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            r("fields[].pupil_sampling[].grid", Sampling),
            r("fields[].pupil_sampling[].wavelength", Exact),
            r("fields[].tangential_direction[]", Direction),
            r("nyquist[]", Frequency),
            on("@diffraction_limit_*", Modulation, "@frequencies"),
            on("@sagittal_*", Modulation, "@frequencies"),
            on("@tangential_*", Modulation, "@frequencies"),
            on("@detector_sagittal_*", Modulation, "@frequencies"),
            on("@detector_tangential_*", Modulation, "@frequencies"),
            on("@system_sagittal_*", Modulation, "@frequencies"),
            on("@system_tangential_*", Modulation, "@frequencies"),
            r("@frequencies", Frequency),
        ],
    ),
    (
        "mtf_vs_field",
        &[
            r("curves[].frequency", Frequency),
            r("pupil_sampling[][].grid", Sampling),
            r("pupil_sampling[][].wavelength", Exact),
            r("@field_values", Exact),
            r("@frequencies", Frequency),
            r("@normalized_field", Exact),
            on("@sagittal_*", Modulation, "@normalized_field"),
            on("@tangential_*", Modulation, "@normalized_field"),
        ],
    ),
    (
        "opd_fan",
        &[
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            r("fields[].sagittal[].pupil[]", Exact),
            on(
                "fields[].sagittal[].values[]",
                Waves,
                "fields[].sagittal[].pupil[]",
            ),
            r("fields[].sagittal[].wavelength", Exact),
            r("fields[].tangential[].pupil[]", Exact),
            on(
                "fields[].tangential[].values[]",
                Waves,
                "fields[].tangential[].pupil[]",
            ),
            r("fields[].tangential[].wavelength", Exact),
            r("wavelengths_um[]", Exact),
        ],
    ),
    (
        "polarization_pupil_map",
        &[
            r("field.hx", Exact),
            r("field.hy", Exact),
            r("field.vignetting.vcx", Exact),
            r("field.vignetting.vcy", Exact),
            r("field.vignetting.vdx", Exact),
            r("field.vignetting.vdy", Exact),
            r("grid", Exact),
            r("max_diattenuation", Polarimetric),
            r("max_retardance", Retardance),
            r("mean_intensity", Polarimetric),
            r("rms_retardance", Retardance),
            r("sample_count", Exact),
            r("undefined_retarder", Exact),
            r("wavelength", Exact),
            r("wavelength_um", Exact),
            r("@diattenuation", Polarimetric),
            r("@diattenuation_axis", Generic),
            r("@fast_axis", FastAxis),
            r("@intensity", Polarimetric),
            r("@jones", Jones),
            r("@pupil", Exact),
            r("@reached", Exact),
            r("@retardance", Retardance),
            r("@retardance_jones", Retardance),
            r("@retarder_defined", Exact),
            r("@t_max", Polarimetric),
            r("@t_min", Polarimetric),
        ],
    ),
    (
        "psf",
        &[
            r("psfs[].centroid_mm[]", Position),
            r("psfs[].field", Exact),
            r("psfs[].field_value[]", Exact),
            r("psfs[].peak_mm[]", Position),
            r("psfs[].psf.center_mm[]", Sampling),
            r("psfs[].psf.diffraction_peak", Relative1e3),
            r("psfs[].psf.size", Sampling),
            r("psfs[].psf.spacing_mm", Sampling),
            r("psfs[].psf.total_energy", Relative1e3),
            r("psfs[].psf.wavelength_um", Exact),
            r("psfs[].pupil_sampling[].grid", Sampling),
            r("psfs[].pupil_sampling[].wavelength", Exact),
            r("psfs[].reference_mm[]", Position),
            r("psfs[].strehl", Modulation),
            r("psfs[].wavelength", Exact),
            on("@data_*", Image, PSF_GRID),
        ],
    ),
    (
        "pupil_aberration",
        &[
            r("shape[]", Exact),
            r("wavelength", Exact),
            r("@field_coordinates", Exact),
            r("@image_mm", Position),
            r("@values.stop_error_x", Position),
            r("@values.stop_error_y", Position),
            r("@values.magnification_x", Magnification),
            r("@values.magnification_y", Magnification),
            r("@values.variation_x", Magnification),
            r("@values.variation_y", Magnification),
        ],
    ),
    (
        "ray_fan",
        &[
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            on(
                "fields[].sagittal[].cross[]",
                Position,
                "fields[].sagittal[].pupil[]",
            ),
            r("fields[].sagittal[].pupil[]", Exact),
            on(
                "fields[].sagittal[].values[]",
                Position,
                "fields[].sagittal[].pupil[]",
            ),
            r("fields[].sagittal[].wavelength", Exact),
            on(
                "fields[].tangential[].cross[]",
                Position,
                "fields[].tangential[].pupil[]",
            ),
            r("fields[].tangential[].pupil[]", Exact),
            on(
                "fields[].tangential[].values[]",
                Position,
                "fields[].tangential[].pupil[]",
            ),
            r("fields[].tangential[].wavelength", Exact),
            r("wavelengths_um[]", Exact),
        ],
    ),
    (
        "ray_grid",
        &[
            r("arrived", Exact),
            r("field_values[][]", Exact),
            r("fields[]", Exact),
            r("launched", Exact),
            r("pupil_count", Exact),
            r("wavelengths[]", Exact),
            r("wavelengths_um[]", Exact),
            r("@chief", Position),
            r("@directions", Direction),
            r("@points", Position),
            r("@pupil", Exact),
        ],
    ),
    (
        "relative_illumination",
        &[
            on("chief_ray_angle_deg[]", Angle, "normalized_field[]"),
            on("cos4_object[]", Generic, "normalized_field[]"),
            r("field_coordinates[][]", Exact),
            r("field_values[]", Exact),
            r("normalized_field[]", Exact),
            on("projected_solid_angle[]", Relative1e3, "normalized_field[]"),
            on(
                "relative_illumination[]",
                Illumination,
                "normalized_field[]",
            ),
            on("vignetting_fraction[]", Generic, "normalized_field[]"),
            r("wavelength", Exact),
        ],
    ),
    (
        "relative_illumination_map",
        &[
            r("shape[]", Exact),
            r("wavelength", Exact),
            r("@field_coordinates", Exact),
            r("@image_mm", Position),
            r("@values.relative_illumination", Illumination),
            r("@values.projected_solid_angle", Relative1e3),
            r("@values.vignetting_fraction", Generic),
        ],
    ),
    (
        "spot_diagram",
        &[
            r("airy_radius", Position),
            r("defocus", Position),
            r("fields[].centroid[]", Position),
            r("fields[].chief[]", Position),
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            r("fields[].geo_radius", PositionDifference),
            r("fields[].per_wavelength[].chief[]", Position),
            r("fields[].per_wavelength[].stats.centroid[]", Position),
            r(
                "fields[].per_wavelength[].stats.geo_radius_centroid",
                PositionDifference,
            ),
            r(
                "fields[].per_wavelength[].stats.geo_radius_chief",
                PositionDifference,
            ),
            r("fields[].per_wavelength[].stats.rays_arrived", Exact),
            r("fields[].per_wavelength[].stats.rays_launched", Exact),
            r("fields[].per_wavelength[].stats.rms_radius_centroid", Rms),
            r("fields[].per_wavelength[].stats.rms_radius_chief", Rms),
            r("fields[].per_wavelength[].stats.rms_x", Rms),
            r("fields[].per_wavelength[].stats.rms_y", Rms),
            r(
                "fields[].per_wavelength[].stats.vignetted_fraction",
                CountRatio,
            ),
            r("fields[].per_wavelength[].wavelength", Exact),
            r("fields[].per_wavelength[].wavelength_um", Exact),
            r("fields[].per_wavelength[].weight", Exact),
            r("fields[].point_range[]", Exact),
            r("fields[].polychromatic.centroid[]", Position),
            r(
                "fields[].polychromatic.geo_radius_centroid",
                PositionDifference,
            ),
            r(
                "fields[].polychromatic.geo_radius_chief",
                PositionDifference,
            ),
            r("fields[].polychromatic.rays_arrived", Exact),
            r("fields[].polychromatic.rays_launched", Exact),
            r("fields[].polychromatic.rms_radius_centroid", Rms),
            r("fields[].polychromatic.rms_radius_chief", Rms),
            r("fields[].polychromatic.rms_x", Rms),
            r("fields[].polychromatic.rms_y", Rms),
            r("fields[].polychromatic.vignetted_fraction", CountRatio),
            r("fields[].reference[]", Position),
            r("fields[].rms_radius", Rms),
            r("fields[].through_focus[].defocus", Position),
            r("fields[].through_focus[].rms_radius_centroid", Rms),
            r("fields[].through_focus[].rms_radius_chief", Rms),
            r("wavelengths[]", Exact),
            r("wavelengths_um[]", Exact),
            r("@point_field", Exact),
            r("@point_wavelength", Exact),
            r("@points", Position),
            // Slopes are direction ratios of the image-space rays.
            r("@point_slope", Direction),
        ],
    ),
    (
        "stray_light",
        &[
            r("classes", Exact),
            r("clear_semi_diameters[]", Position),
            r("detector.half_x", Position),
            r("detector.half_y", Position),
            r("detector.nx", Exact),
            r("detector.ny", Exact),
            r("detector.surface", Exact),
            r("emitted", Ledger),
            r("energy.absorbed", Ledger),
            r("energy.absorbed_by_sink[][]", Ledger),
            r("energy.balance_residual", Ledger),
            r("energy.detected", Ledger),
            r("energy.detected_by_class[][]", Ledger),
            r("energy.detected_by_wavelength[][]", Ledger),
            r("energy.detected_covered_ghost", Ledger),
            r("energy.detected_covered_ghost_std_error", Ledger),
            r("energy.detected_hits_by_class[][]", Statistical),
            r("energy.detected_owned", Ledger),
            r("energy.detected_owned_std_error", Ledger),
            r("energy.detected_std_error", Ledger),
            r("energy.detected_std_error_by_class[][]", Ledger),
            r("energy.detected_uncovered_ghost", Ledger),
            r("energy.detected_uncovered_ghost_std_error", Ledger),
            r("energy.emitted", Ledger),
            r("energy.emitted_by_wavelength[]", Ledger),
            r("energy.escaped", Ledger),
            r("energy.escaped_from_solid", Ledger),
            r("energy.events", Statistical),
            r("energy.ghost_by_order[]", Ledger),
            r("energy.ghost_by_pair[][]", Ledger),
            r("energy.mc_adjustment", Ledger),
            r("energy.mc_adjustment_std_error", Ledger),
            r("energy.owned_classes", Exact),
            r("energy.rays", Exact),
            r("energy.roulette_gain", Ledger),
            r("energy.roulette_loss", Ledger),
            r("energy.sampling_failed", Ledger),
            r("energy.truncated", Ledger),
            r("energy.weighting_gain", Ledger),
            r("energy.weighting_loss", Ledger),
            r("partition.covered_ghost_flux_mc", Ledger),
            r("partition.covered_ghost_std_error", Ledger),
            r("partition.ghost_class_flux_mc", Ledger),
            r("partition.ghost_class_std_error", Ledger),
            r("partition.ghost_paths", Exact),
            r("partition.uncovered_ghost_flux", Ledger),
            r("partition.uncovered_ghost_signatures", Statistical),
            r("partition.uncovered_ghost_std_error", Ledger),
            r("primary_obstruction", Generic),
            r("source.reference_flux", Radiometric),
            r("source.spec.axis.x", Exact),
            r("source.spec.axis.y", Exact),
            r("source.spec.axis.z", Exact),
            r("source.spec.center.x", Exact),
            r("source.spec.center.y", Exact),
            r("source.spec.center.z", Exact),
            r("source.spec.direction.x", Exact),
            r("source.spec.direction.y", Exact),
            r("source.spec.direction.z", Exact),
            r("source.spec.half_angle_deg", Exact),
            r("source.spec.intensity", Exact),
            r("source.spec.irradiance", Exact),
            r("source.spec.normal.x", Exact),
            r("source.spec.normal.y", Exact),
            r("source.spec.normal.z", Exact),
            r("source.spec.position.x", Exact),
            r("source.spec.position.y", Exact),
            r("source.spec.position.z", Exact),
            r("source.spec.power", Exact),
            r("source.spec.radius", Exact),
            r("source.spec.start_distance", Exact),
            r("stray_flux", Ledger),
            r("stray_fraction", Generic),
            r("stray_std_error", Ledger),
            r("top_signatures[].events[][]", Exact),
            r("top_signatures[].flux", Ledger),
            r("top_signatures[].hits", Statistical),
            r("top_signatures[].std_error", Ledger),
            r("totals[].flux", MonteCarlo("totals[].std_error")),
            r("totals[].fraction", Generic),
            r("totals[].hits", Statistical),
            r(
                "totals[].moments.centroid[]",
                MonteCarlo("totals[].moments.centroid_std_error[]"),
            ),
            r(
                "totals[].moments.centroid_std_error[]",
                MonteCarlo("totals[].moments.centroid_std_error[]"),
            ),
            r("totals[].moments.flux", MonteCarlo("totals[].std_error")),
            r(
                "totals[].moments.rms_radius",
                MonteCarlo("totals[].moments.rms_radius_std_error"),
            ),
            r(
                "totals[].moments.rms_radius_std_error",
                MonteCarlo("totals[].moments.rms_radius_std_error"),
            ),
            r("totals[].std_error", MonteCarlo("totals[].std_error")),
            r("@class_irradiance", Image),
            r("@class_irradiance_std_error", Image),
            r("@total_irradiance", Image),
            r("@total_irradiance_std_error", Image),
        ],
    ),
    (
        "system_transmission",
        &[
            r("fields[].hx", Exact),
            r("fields[].hy", Exact),
            r("fields[].vignetting.vcx", Exact),
            r("fields[].vignetting.vcy", Exact),
            r("fields[].vignetting.vdx", Exact),
            r("fields[].vignetting.vdy", Exact),
            r("points[].failed", Exact),
            r("points[].field", Exact),
            r("points[].mean_unvignetted", Radiometric),
            r("points[].samples", Exact),
            r("points[].transmission", Radiometric),
            r("points[].unvignetted_fraction", CountRatio),
            r("points[].wavelength", Exact),
            r("points[].wavelength_um", Exact),
            r("polychromatic[]", Radiometric),
            r("wavelengths[]", Exact),
        ],
    ),
    (
        "through_focus_mtf",
        &[
            r("fields[].curves[].frequency", Frequency),
            r("fields[].field", Exact),
            r("fields[].field_value[]", Exact),
            r("fields[].pupil_sampling[].grid", Sampling),
            r("fields[].pupil_sampling[].wavelength", Exact),
            r("@focus_mm", Position),
            r("@frequencies", Frequency),
            on("@sagittal_*", Modulation, "@focus_mm"),
            on("@tangential_*", Modulation, "@focus_mm"),
        ],
    ),
    (
        "wavefront_map",
        &[
            r("maps[].exit_pupil.chief_direction[]", Direction),
            r("maps[].exit_pupil.chief_point_mm[]", Position),
            r("maps[].exit_pupil.clear_area", Generic),
            r("maps[].exit_pupil.effective_f_number", FNumber),
            r("maps[].exit_pupil.exit_pupil_point_mm[]", Position),
            r("maps[].exit_pupil.image_index", Exact),
            r("maps[].exit_pupil.lattice_rays", Exact),
            r("maps[].exit_pupil.numerical_aperture", Direction),
            r("maps[].exit_pupil.paraxial_position_mm", Position),
            r("maps[].exit_pupil.paraxial_radius_mm", Position),
            r("maps[].exit_pupil.propagation_sign", Exact),
            r("maps[].exit_pupil.reference_point_mm[]", Position),
            r("maps[].exit_pupil.reference_radius_mm", Position),
            r("maps[].exit_pupil.rim_points", Exact),
            r("maps[].exit_pupil.sin_max", Direction),
            r("maps[].exit_pupil.surviving_rays", Exact),
            r("maps[].exit_pupil.transmitted_fraction", Generic),
            r("maps[].exit_pupil.wavelength_um", Exact),
            r("maps[].field", Exact),
            r("maps[].field_value[]", Exact),
            r("maps[].grid", Sampling),
            r("maps[].pv_waves", Waves),
            r("maps[].pv_waves_tilt_removed", Waves),
            r("maps[].rms_waves", Waves),
            r("maps[].rms_waves_tilt_removed", Waves),
            r("maps[].seidel.astigmatism", Waves),
            r("maps[].seidel.astigmatism_angle_deg", Orientation(180.)),
            r("maps[].seidel.coma", Waves),
            r("maps[].seidel.coma_angle_deg", Orientation(360.)),
            r("maps[].seidel.focus", Waves),
            r("maps[].seidel.piston", Waves),
            r("maps[].seidel.spherical", Waves),
            r("maps[].seidel.tilt", Waves),
            r("maps[].seidel.tilt_angle_deg", Orientation(360.)),
            r("maps[].strehl_at_reference", Modulation),
            r("maps[].strehl_marechal", Modulation),
            r("maps[].wavelength", Exact),
            r("maps[].wavelength_um", Exact),
            r("maps[].zernike_condition", Diagnostic),
            r("maps[].zernike_residual_rms", Waves),
            on("@amplitude_*", Generic, "@coordinates_*"),
            r("@coefficients_*", Waves),
            r("@coordinates_*", Generic),
            on("@opd_waves_*", Waves, "@coordinates_*"),
            r("@rim_*", Loop),
        ],
    ),
];

/// The solvers' physical score reports as `project_bench` compares them
/// (merit, constraint violation, status, operand values, tolerance criteria
/// and finalists; execution telemetry and returned designs are not compared).
/// Operand values, constraint violations and tolerance criteria are in their
/// operands' own units, so they keep the generic budget.
pub const SOLVER_SCORES: &[(&str, &[Rule])] = &[
    (
        "merit_evaluation",
        &[
            r("merit", Merit),
            r("violation", Generic),
            r("status.*", Exact),
            r("operands[].config", Exact),
            r("operands[].contribution", Merit),
            r("operands[].contribution_fraction", Generic),
            r("operands[].field*", Exact),
            r("operands[].index", Exact),
            r("operands[].residual", Generic),
            r("operands[].start", Generic),
            r("operands[].target", Exact),
            r("operands[].value", Generic),
            r("operands[].wavelength*", Exact),
            r("operands[].weight", Exact),
        ],
    ),
    (
        "optimize",
        &[
            r("start.merit", Merit),
            r("start.violation", Generic),
            r("start.status.*", Exact),
            r("final.merit", Merit),
            r("final.violation", Generic),
            r("final.status.*", Exact),
        ],
    ),
    (
        "tolerance",
        &[
            r("sensitivity.nominal.compensators[]", Generic),
            r("sensitivity.nominal.criterion", Generic),
            r("sensitivity.operands[].change_minus", Generic),
            r("sensitivity.operands[].change_plus", Generic),
            r("sensitivity.operands[].compensators_minus[]", Generic),
            r("sensitivity.operands[].compensators_plus[]", Generic),
            r("sensitivity.operands[].criterion_minus", Generic),
            r("sensitivity.operands[].criterion_plus", Generic),
            r("sensitivity.operands[].index", Exact),
            r("sensitivity.operands[].minus", Exact),
            r("sensitivity.operands[].param*", Exact),
            r("sensitivity.operands[].plus", Exact),
            r("sensitivity.operands[].worst", Generic),
            r("sensitivity.predicted_criterion", Generic),
            r("sensitivity.ranking[]", Exact),
            r("sensitivity.rss_change", Generic),
            r("monte_carlo.compensator_mean[]", Generic),
            r("monte_carlo.compensator_std[]", Generic),
            r("monte_carlo.criteria[]", Generic),
            r("monte_carlo.failures", Exact),
            r("monte_carlo.max", Generic),
            r("monte_carlo.mean", Generic),
            r("monte_carlo.min", Generic),
            r("monte_carlo.nominal.compensators[]", Generic),
            r("monte_carlo.nominal.criterion", Generic),
            r("monte_carlo.percentiles[].percentile", Exact),
            r("monte_carlo.percentiles[].value", Generic),
            r("monte_carlo.records[].compensators[]", Generic),
            r("monte_carlo.records[].criterion", Generic),
            r("monte_carlo.records[].perturbations[]", Exact),
            r("monte_carlo.records[].trial", Exact),
            r("monte_carlo.seed", Exact),
            r("monte_carlo.std_dev", Generic),
            r("monte_carlo.threshold", Exact),
            r("monte_carlo.trials", Exact),
            r("monte_carlo.yield_fraction", Generic),
        ],
    ),
    (
        "glass_substitution",
        &[
            r("finalists[].merit", Merit),
            r("finalists[].violation", Generic),
            r("finalists[].glasses[].dpgf", Exact),
            r("finalists[].glasses[].nd", Exact),
            r("finalists[].glasses[].vd", Exact),
            r("finalists[].glasses[].surface", Exact),
        ],
    ),
];

/// Analyses without a GPU path: every backend request runs the CPU f64
/// reference, so a comparison checks that the routed result is the
/// reference's. Lengths, wavefront terms, angles, F-numbers and apertures
/// keep their physical classes; echoed inputs, catalogue data and the host
/// f64 material evaluation (indices, Abbe numbers, partial dispersions,
/// which no backend computes on a device) are exact.
pub const REFERENCE_ONLY: &[(&str, &[Rule])] = &[
    (
        "chromatic_focal_shift",
        &[
            r("focal_shift[]", Position),
            r("focal_shift_x[]", Position),
            r("paraxial_focus[]", Position),
            r("paraxial_focus_x[]", Position),
            r("range", PositionDifference),
            r("primary_wavelength_um", Exact),
            r("wavelengths_um[]", Exact),
        ],
    ),
    (
        "first_order",
        &[
            r("entries[].value", Entry),
            r("tables[].entries[].value", Entry),
            r("tables[].wavelength*", Exact),
            r("wavelength_um", Exact),
            r("paraxial_xy.wavelengths[].bfl_*", Position),
            r("paraxial_xy.wavelengths[].efl_*", Position),
            r("paraxial_xy.wavelengths[].squeeze", Generic),
            r("paraxial_xy.*", Exact),
            r("system.*", Exact),
        ],
    ),
    ("glass_info", &[r("*", Exact)]),
    (
        "glass_map",
        &[r("glasses[].*", Exact), r("@coordinates", Exact)],
    ),
    (
        "rng_stream",
        &[
            // Philox draws are bit-identical on every backend (CONVENTIONS §6).
            r("count", Exact),
            r("first[]", Exact),
            r("min", Exact),
            r("max", Exact),
            r("@uniform", Exact),
            // Reductions over the draws.
            r("mean", Generic),
            r("variance", Generic),
        ],
    ),
    (
        "seidel",
        &[
            r("image_marginal_slope", Direction),
            r("lagrange_invariant", Position),
            r("petzval_radius", Position),
            r("petzval_sum", Generic),
            // Seidel and chromatic coefficients in mm of wavefront; the
            // same terms in waves carry the OPD budget.
            r("sums.*", Position),
            r("surfaces[].surface", Exact),
            r("surfaces[].*", Position),
            r("wavefront_mm.*", Position),
            r("wavefront_waves.*", Waves),
            r("*wavelength", Exact),
            r("wavelength_um", Exact),
        ],
    ),
    (
        "solves",
        &[r("sweeps", Exact), r("max_change", Generic), r("*", Exact)],
    ),
    (
        "system_summary",
        &[
            r("paraxial_xy.wavelengths[].bfl_*", Position),
            r("paraxial_xy.wavelengths[].efl_*", Position),
            r("paraxial_xy.wavelengths[].squeeze", Generic),
            r("paraxial_xy.*", Exact),
            r("first_to_image_distance", Position),
            r("surfaces[].frame.origin[]", Position),
            r("surfaces[].frame.rot[][]", Direction),
            r("surfaces[].radius", Position),
            r("@vertex_global", Position),
            r("@n_after", Exact),
            // The echoed system and its host f64 material indices.
            r("*", Exact),
        ],
    ),
    ("validate", &[r("*", Exact)]),
];

/// The class of a first-order table entry `{key, unit, value}` (`Entry`):
/// echoed inputs exact, F-numbers, numerical apertures and magnification by
/// their classes, lengths as positions.
fn entry_class(entry: &serde_json::Map<String, Value>) -> Class {
    let key = entry.get("key").and_then(Value::as_str).unwrap_or("");
    let unit = entry.get("unit").and_then(Value::as_str).unwrap_or("");
    match key {
        "object_distance" | "max_field" => Exact,
        k if k.ends_with("f_number") => FNumber,
        k if k.ends_with("_na") => Direction,
        "magnification" => Magnification,
        _ if unit == "mm" => Position,
        _ => Generic,
    }
}

/// The rule of a normalised path of `analysis`, if the table maps it.
pub fn rule(analysis: &str, path: &str) -> Option<Rule> {
    FIELDS
        .iter()
        .chain(SOLVER_SCORES)
        .chain(REFERENCE_ONLY)
        .find(|(a, _)| *a == analysis)?
        .1
        .iter()
        .find(|r| glob(r.path, path))
        .copied()
}

/// `*` matches any (possibly empty) run of characters.
fn glob(pattern: &str, s: &str) -> bool {
    match pattern.split_once('*') {
        None => pattern == s,
        Some((head, tail)) => {
            let Some(rest) = s.strip_prefix(head) else {
                return false;
            };
            (0..=rest.len()).any(|i| rest.is_char_boundary(i) && glob(tail, &rest[i..]))
        }
    }
}

/// The `*` captures of `name` under `pattern`, substituted into `template`.
fn substitute(pattern: &str, name: &str, template: &str) -> String {
    let (head, tail) = pattern.split_once('*').unwrap_or((pattern, ""));
    let capture = name
        .strip_prefix(head)
        .and_then(|s| s.strip_suffix(tail))
        .unwrap_or("");
    template.replacen('*', capture, 1)
}

/// Normalised form of a concrete JSON path: indices dropped.
fn normalise(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    let mut skip = false;
    for c in path.chars() {
        match c {
            '[' => {
                out.push('[');
                skip = true;
            }
            ']' => {
                out.push(']');
                skip = false;
            }
            _ if skip => (),
            _ => out.push(c),
        }
    }
    out
}

/// Indices of a concrete JSON path, in order.
fn indices(path: &str) -> Vec<usize> {
    path.split('[')
        .skip(1)
        .filter_map(|s| s.split(']').next()?.parse().ok())
        .collect()
}

/// A normalised template with the first indices of a concrete path.
fn concrete(template: &str, idx: &[usize]) -> String {
    let mut out = String::new();
    let mut it = idx.iter();
    let mut parts = template.split("[]").peekable();
    while let Some(p) = parts.next() {
        out.push_str(p);
        if parts.peek().is_some() {
            match it.next() {
                Some(i) => out.push_str(&format!("[{i}]")),
                None => out.push_str("[]"),
            }
        }
    }
    out
}

fn get<'a>(root: &'a Value, path: &str) -> Option<&'a Value> {
    let mut v = root;
    for part in path.split('.').filter(|s| !s.is_empty()) {
        let (key, rest) = part.split_once('[').unwrap_or((part, ""));
        if !key.is_empty() {
            v = v.get(key)?;
        }
        for i in rest.split('[').filter(|s| !s.is_empty()) {
            v = v.get(i.trim_end_matches(']').parse::<usize>().ok()?)?;
        }
    }
    Some(v)
}

fn floats(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?
        .iter()
        .map(|x| x.as_f64().or(x.is_null().then_some(f64::NAN)))
        .collect()
}

fn buffer_floats(d: &BufferData) -> Option<Vec<f64>> {
    match d {
        BufferData::F64(v) => Some(v.clone()),
        BufferData::F32(v) => Some(v.iter().map(|x| *x as f64).collect()),
        _ => None,
    }
}

/// Per-class summary: compared values, worst error as a fraction of the
/// gate, and where.
#[derive(Default, Clone, Debug, serde::Serialize)]
pub struct ClassStat {
    pub compared: usize,
    pub worst_gate_fraction: f64,
    pub worst_path: Option<String>,
    pub worst_error: f64,
}

#[derive(Default)]
pub struct Comparison {
    pub compared: usize,
    pub maximum: f64,
    pub maximum_path: Option<String>,
    pub image_l1_relative: f64,
    pub energy_relative: f64,
    /// The first gate failure (kept for the harness output format).
    pub reason: Option<String>,
    /// Every gate failure, one per path.
    pub failures: Vec<String>,
    pub classes: BTreeMap<&'static str, ClassStat>,
    /// Curves and maps compared after resampling to common coordinates.
    pub resampled: usize,
    /// Sampling choices that differ between the two results (not gated).
    pub sampling_differences: Vec<String>,
    /// Values not compared because they are undefined (orientation of a
    /// vanishing term, axis of an undefined retarder) or fall outside the
    /// other result's sampled domain after resampling.
    pub undefined: usize,
    analysis: String,
    scale: Scale,
    outlines: Vec<(String, &'static str, [f64; 2], [f64; 2])>,
}

impl Comparison {
    fn fail(&mut self, path: &str, reason: &str) {
        let line = format!("chain qualification at {path}: {reason}");
        if self.reason.is_none() {
            self.reason = Some(line.clone());
        }
        if self.failures.len() < 64
            && !self
                .failures
                .iter()
                .any(|f| f.starts_with(&format!("chain qualification at {path}:")))
        {
            self.failures.push(line);
        }
    }

    fn stat(&mut self, class: Class, error: f64, gate: f64, path: &str) {
        let s = self.classes.entry(class.name()).or_default();
        s.compared += 1;
        let fraction = if gate > 0. {
            error / gate
        } else if error > 0. {
            f64::INFINITY
        } else {
            0.
        };
        if fraction > s.worst_gate_fraction {
            s.worst_gate_fraction = fraction;
            s.worst_path = Some(path.into());
            s.worst_error = error;
        }
        if error > self.maximum {
            self.maximum = error;
            self.maximum_path = Some(path.into());
        }
    }

    /// Compare one number under its class.
    fn number(&mut self, class: Class, a: f64, b: f64, path: &str) {
        self.compared += 1;
        if a.is_nan() && b.is_nan() || a == b {
            self.stat(class, 0., 1., path);
            return;
        }
        if class == Sampling {
            if (a - b).abs() > 1e-9 * a.abs().max(b.abs()) {
                self.sampling_differences
                    .push(format!("{path}: {a} vs {b}"));
            }
            return;
        }
        if !a.is_finite() || !b.is_finite() {
            self.fail(path, "undefined/survival masks differ");
            return;
        }
        let error = (a - b).abs();
        if matches!(class, Exact | OutlineVertices) {
            self.stat(class, error, 0., path);
            self.fail(path, &format!("discrete values differ ({a} vs {b})"));
            return;
        }
        let gate = class.gate(a, b, &self.scale);
        self.stat(class, error, gate, path);
        if error > gate {
            self.fail(
                path,
                &format!("{} error {error:e} exceeds the gate {gate:e}", class.name()),
            );
        }
    }

    fn unmapped(&mut self, path: &str) {
        self.fail(
            path,
            &format!(
                "unmapped result field `{}` of {} (add it to comparison::FIELDS)",
                normalise(path).trim_start_matches("result."),
                self.analysis
            ),
        );
    }

    fn rule(&self, path: &str) -> Option<Rule> {
        rule(&self.analysis, path)
    }

    /// JSON walk over `a` (the reference) against `b`.
    fn json(&mut self, ra: &Value, rb: &Value, a: &Value, b: &Value, path: &str) {
        let key = normalise(path);
        let key = key.strip_prefix(".").unwrap_or(&key).to_string();
        match (a, b) {
            (Value::Object(oa), Value::Object(ob)) => {
                for (k, va) in oa {
                    if TELEMETRY.contains(&k.as_str()) {
                        continue;
                    }
                    let sub = format!("{path}.{k}");
                    if self.is_axis(&format!("{}[]", &normalise(&sub)[1..])) {
                        continue;
                    }
                    let Some(vb) = ob.get(k) else {
                        self.fail(&format!("result{path}"), &format!("missing result key {k}"));
                        continue;
                    };
                    if let Some(Rule {
                        class: Orientation(period),
                        ..
                    }) = self.rule(&normalise(&sub)[1..])
                    {
                        let magnitude = k
                            .strip_suffix("_angle_deg")
                            .and_then(|m| oa.get(m))
                            .and_then(Value::as_f64);
                        self.orientation(period, magnitude, va, vb, &format!("result{sub}"));
                        continue;
                    }
                    if let (Some(Rule { class: Entry, .. }), Some(x), Some(y)) =
                        (self.rule(&normalise(&sub)[1..]), va.as_f64(), vb.as_f64())
                    {
                        self.number(entry_class(oa), x, y, &format!("result{sub}"));
                        continue;
                    }
                    self.json(ra, rb, va, vb, &sub);
                }
            }
            (Value::Array(xa), Value::Array(xb)) => {
                let element = format!("{key}[]");
                match self.rule(&element) {
                    Some(Rule {
                        class: Outline(buffer),
                        ..
                    }) => {
                        let range = |v: &[Value]| {
                            [
                                v.first().and_then(Value::as_f64).unwrap_or(0.),
                                v.get(1).and_then(Value::as_f64).unwrap_or(0.),
                            ]
                        };
                        self.outlines
                            .push((format!("result{path}"), buffer, range(xa), range(xb)));
                        return;
                    }
                    Some(Rule {
                        class,
                        axis: Some(axis),
                        ..
                    }) => {
                        let idx = indices(path);
                        let axis_path = concrete(axis.trim_end_matches("[]"), &idx);
                        let ax = get(ra, &axis_path).and_then(floats);
                        let bx = get(rb, &axis_path).and_then(floats);
                        let (ya, yb) = (floats(a), floats(b));
                        if let (Some(ax), Some(bx), Some(ya), Some(yb)) = (ax, bx, ya, yb)
                            && ax.len() == ya.len()
                            && bx.len() == yb.len()
                        {
                            let axis_class = self.rule(axis).map_or(Exact, |r| r.class);
                            self.curve(
                                class,
                                axis_class,
                                &ax,
                                &ya,
                                &bx,
                                &yb,
                                &format!("result{path}"),
                            );
                            return;
                        }
                    }
                    _ => (),
                }
                if xa.len() != xb.len() {
                    self.fail(&format!("result{path}"), "result lengths differ");
                    return;
                }
                for (i, (va, vb)) in xa.iter().zip(xb).enumerate() {
                    self.json(ra, rb, va, vb, &format!("{path}[{i}]"));
                }
            }
            (Value::Number(na), Value::Number(nb)) => {
                let full = format!("result{path}");
                let Some(rule) = self.rule(&key) else {
                    self.unmapped(&full);
                    return;
                };
                let integers = !na.is_f64() && !nb.is_f64();
                let class = match rule.class {
                    Statistical | Sampling | Outline(_) => rule.class,
                    _ if integers => Exact,
                    c => c,
                };
                if let MonteCarlo(se) = class {
                    let at = concrete(se, &indices(path));
                    let sigma = |root: &Value| get(root, &at).and_then(Value::as_f64).unwrap_or(0.);
                    self.scale.sigma = sigma(ra).hypot(sigma(rb));
                }
                self.number(class, na.as_f64().unwrap(), nb.as_f64().unwrap(), &full);
            }
            _ if a == b => (),
            (Value::Number(_), _) | (_, Value::Number(_)) | (Value::Null, _) | (_, Value::Null) => {
                if !a.is_number() || self.rule(&key).is_some() {
                    self.fail(&format!("result{path}"), "undefined/survival masks differ");
                } else {
                    self.unmapped(&format!("result{path}"));
                }
            }
            _ => self.fail(&format!("result{path}"), "labels differ"),
        }
    }

    /// Periodic orientation of a wavefront term: undefined below 1e-3 waves.
    fn orientation(
        &mut self,
        period: f64,
        magnitude: Option<f64>,
        a: &Value,
        b: &Value,
        path: &str,
    ) {
        let (Some(x), Some(y)) = (a.as_f64(), b.as_f64()) else {
            if a != b {
                self.fail(path, "undefined/survival masks differ");
            }
            return;
        };
        if magnitude.is_some_and(|m| m.abs() < 1e-3) {
            self.undefined += 1;
            return;
        }
        let d = (y - x).rem_euclid(period);
        let d = if d > 0.5 * period { d - period } else { d };
        self.number(Orientation(period), x, x + d, path);
    }

    /// A curve `y(x)`: index by index where both results sampled the same
    /// coordinates (within the axis class), else the denser curve linearly
    /// resampled onto the sparser one's coordinates; coordinates outside
    /// the other curve's range are not compared.
    #[allow(clippy::too_many_arguments)]
    fn curve(
        &mut self,
        class: Class,
        axis_class: Class,
        ax: &[f64],
        ya: &[f64],
        bx: &[f64],
        yb: &[f64],
        path: &str,
    ) {
        let same = ax.len() == bx.len()
            && ax
                .iter()
                .zip(bx)
                .all(|(p, q)| p == q || (p - q).abs() <= axis_class.gate(*p, *q, &self.scale));
        if same {
            for (i, (p, q)) in ax.iter().zip(bx).enumerate() {
                if axis_class != Sampling {
                    self.number(axis_class, *p, *q, &format!("{path}.axis[{i}]"));
                }
            }
            if class == Image {
                self.profile(ya, yb, path);
            } else {
                for (i, (p, q)) in ya.iter().zip(yb).enumerate() {
                    self.number(class, *p, *q, &format!("{path}[{i}]"));
                }
            }
            return;
        }
        self.resampled += 1;
        self.sampling_differences
            .push(format!("{path}: axis resampled"));
        // Interpolate the denser curve onto the sparser one's coordinates,
        // so the comparison carries the smaller interpolation error.
        let density = |x: &[f64]| {
            let span = (x[x.len() - 1] - x[0]).abs();
            if span > 0. {
                (x.len() - 1) as f64 / span
            } else {
                0.
            }
        };
        let flip = ax.len() > 1 && bx.len() > 1 && density(ax) > density(bx);
        let (tx, ty, sx, sy) = if flip {
            (bx, yb, ax, ya)
        } else {
            (ax, ya, bx, yb)
        };
        let resampled: Vec<f64> = tx.iter().map(|x| interpolate(sx, sy, *x)).collect();
        let (pa, pb): (Vec<f64>, Vec<f64>) = if flip {
            (resampled, ty.to_vec())
        } else {
            (ty.to_vec(), resampled)
        };
        let mut pairs = Vec::new();
        for (i, (p, q)) in pa.iter().zip(&pb).enumerate() {
            if p.is_nan() != q.is_nan() {
                self.undefined += 1;
            } else {
                pairs.push((tx[i], *p, *q));
            }
        }
        if class == Image {
            let (p, q): (Vec<f64>, Vec<f64>) = pairs.iter().map(|(_, p, q)| (*p, *q)).unzip();
            self.profile(&p, &q, path);
            return;
        }
        for (x, p, q) in pairs {
            self.number(class, p, q, &format!("{path}@{x}"));
        }
    }

    /// Images and profiles: L1 and integrated-energy gates (§9.4 E).
    fn profile(&mut self, a: &[f64], b: &[f64], path: &str) {
        let mut norm = 0.;
        let mut delta = 0.;
        let mut energy = 0.;
        for (&a, &b) in a.iter().zip(b) {
            self.compared += 1;
            if !a.is_finite() || !b.is_finite() {
                if !(a.is_nan() && b.is_nan()) && a != b {
                    self.fail(path, "image masks differ");
                }
                continue;
            }
            norm += a.abs();
            delta += (a - b).abs();
            energy += a - b;
            if (a - b).abs() > self.maximum {
                self.maximum = (a - b).abs();
                self.maximum_path = Some(path.into());
            }
        }
        let l1 = if norm > 0. { delta / norm } else { 0. };
        let e = if norm > 0. { energy.abs() / norm } else { 0. };
        let s = self.classes.entry("image").or_default();
        s.compared += a.len();
        let fraction = (l1 / (2e-3 / 3.)).max(e / (1e-3 / 3.));
        if fraction > s.worst_gate_fraction {
            s.worst_gate_fraction = fraction;
            s.worst_path = Some(path.into());
            s.worst_error = l1;
        }
        if delta > norm * 2e-3 / 3. || energy.abs() > norm * 1e-3 / 3. {
            self.fail(path, "image L1 or integrated-energy gate failed");
        }
        self.image_l1_relative = self.image_l1_relative.max(l1);
        self.energy_relative = self.energy_relative.max(e);
    }

    fn jones(&mut self, a: &[f64], b: &[f64], path: &str) {
        for (i, (a, b)) in a.chunks(8).zip(b.chunks(8)).enumerate() {
            let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
            let error = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            self.compared += a.len();
            if a.iter().zip(b).any(|(a, b)| a.is_nan() != b.is_nan()) {
                self.fail(path, "matrix masks differ");
                continue;
            }
            let gate = (1e-4 * norm + 1e-7 / std::f64::consts::SQRT_2) / 3.;
            self.stat(Jones, error, gate, &format!("{path}[{i}]"));
            if error > gate {
                self.fail(
                    path,
                    &format!("Jones norm error {error:e} exceeds the gate {gate:e}"),
                );
            }
        }
    }

    fn hausdorff(&mut self, class: Class, a: &[[f64; 3]], b: &[[f64; 3]], path: &str) {
        if a.is_empty() || b.is_empty() {
            if a.len() != b.len() {
                self.fail(path, "outline present on one side only");
            }
            return;
        }
        let d = directed(a, b).max(directed(b, a));
        let gate = class.gate(0., 0., &self.scale);
        self.compared += 1;
        self.stat(class, d, gate, path);
        if d > gate {
            self.fail(
                path,
                &format!("outline Hausdorff distance {d:e} exceeds the gate {gate:e}"),
            );
        }
    }
}

/// Linear interpolation of `y(x)` at `t` (monotone `x`); NaN outside.
fn interpolate(x: &[f64], y: &[f64], t: f64) -> f64 {
    if x.len() == 1 {
        return if x[0] == t { y[0] } else { f64::NAN };
    }
    let rising = x.last() >= x.first();
    let tol = 1e-12 * (x[0].abs() + x[x.len() - 1].abs() + 1.);
    for i in 0..x.len().saturating_sub(1) {
        let (x0, x1) = (x[i], x[i + 1]);
        let (lo, hi) = if rising { (x0, x1) } else { (x1, x0) };
        if t >= lo - tol && t <= hi + tol {
            if x1 == x0 {
                return y[i];
            }
            let w = ((t - x0) / (x1 - x0)).clamp(0., 1.);
            return y[i] + w * (y[i + 1] - y[i]);
        }
    }
    f64::NAN
}

/// Bilinear interpolation on a regular grid with axes `gx` (fast) and `gy`.
fn bilinear(gx: &[f64], gy: &[f64], data: &[f64], x: f64, y: f64) -> f64 {
    let cell = |g: &[f64], t: f64| -> Option<(usize, f64)> {
        if g.len() < 2 {
            return (g.first() == Some(&t)).then_some((0, 0.));
        }
        let step = (g[g.len() - 1] - g[0]) / (g.len() - 1) as f64;
        let u = (t - g[0]) / step;
        let tol = 1e-9;
        if u < -tol || u > (g.len() - 1) as f64 + tol {
            return None;
        }
        let i = (u.floor().max(0.) as usize).min(g.len() - 2);
        Some((i, (u - i as f64).clamp(0., 1.)))
    };
    let (Some((i, wx)), Some((j, wy))) = (cell(gx, x), cell(gy, y)) else {
        return f64::NAN;
    };
    let n = gx.len();
    let at = |ii: usize, jj: usize| data[jj.min(gy.len() - 1) * n + ii.min(n - 1)];
    let v0 = at(i, j) * (1. - wx) + at(i + 1, j) * wx;
    let v1 = at(i, j + 1) * (1. - wx) + at(i + 1, j + 1) * wx;
    v0 * (1. - wy) + v1 * wy
}

/// Largest distance from points along polygon `a` (closed, each edge
/// sampled at 16 points) to the closed polygon `b`.
fn directed(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    let segment = |p: [f64; 3], s: [f64; 3], e: [f64; 3]| {
        let d = [e[0] - s[0], e[1] - s[1], e[2] - s[2]];
        let w = [p[0] - s[0], p[1] - s[1], p[2] - s[2]];
        let dd = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
        let t = if dd > 0. {
            ((w[0] * d[0] + w[1] * d[1] + w[2] * d[2]) / dd).clamp(0., 1.)
        } else {
            0.
        };
        let q = [w[0] - t * d[0], w[1] - t * d[1], w[2] - t * d[2]];
        (q[0] * q[0] + q[1] * q[1] + q[2] * q[2]).sqrt()
    };
    let to_b = |p: [f64; 3]| {
        (0..b.len())
            .map(|j| segment(p, b[j], b[(j + 1) % b.len()]))
            .fold(f64::INFINITY, f64::min)
    };
    let mut worst: f64 = 0.;
    for i in 0..a.len() {
        let (s, e) = (a[i], a[(i + 1) % a.len()]);
        for k in 0..16 {
            let t = k as f64 / 16.;
            worst = worst.max(to_b([
                s[0] + t * (e[0] - s[0]),
                s[1] + t * (e[1] - s[1]),
                s[2] + t * (e[2] - s[2]),
            ]));
        }
    }
    worst
}

fn points(data: &[f64], dim: usize) -> Vec<[f64; 3]> {
    data.chunks(dim.max(1))
        .map(|c| {
            [
                c[0],
                c.get(1).copied().unwrap_or(0.),
                c.get(2).copied().unwrap_or(0.),
            ]
        })
        .collect()
}

/// Ghost paths of equal irradiance (symmetric relays) rank in either order,
/// and stray signatures of near-equal flux likewise: match them by identity.
fn ranked(analysis: &str, json: &Value) -> Value {
    let mut json = json.clone();
    if matches!(analysis, "ghost_paths" | "ghost_irradiance")
        && let Some(ghosts) = json["ghosts"].as_array_mut()
    {
        ghosts.sort_by_cached_key(|g| (g["label"].to_string(), g["path"].to_string()));
    }
    json
}

/// The strongest stray-light signatures are a Monte-Carlo ranking: match
/// them by identity (class, label, events); a signature listed on one side
/// only must lie within 5 combined standard errors of the other side's cut
/// (its weakest listed flux), else it is a failure.
fn align_signatures(c: &mut Comparison, a: &mut Value, b: &mut Value) {
    let (Some(la), Some(lb)) = (
        a["top_signatures"].as_array().cloned(),
        b["top_signatures"].as_array().cloned(),
    ) else {
        return;
    };
    let id = |g: &Value| format!("{}|{}|{}", g["class"], g["label"], g["events"]);
    let flux = |g: &Value| g["flux"].as_f64().unwrap_or(0.);
    let se = |g: &Value| g["std_error"].as_f64().unwrap_or(0.);
    let mut pa = Vec::new();
    let mut pb = Vec::new();
    for g in &la {
        if let Some(h) = lb.iter().find(|h| id(h) == id(g)) {
            pa.push(g.clone());
            pb.push(h.clone());
        }
    }
    for (only, other, side) in [(&la, &lb, "reference"), (&lb, &la, "result")] {
        let cut = other
            .iter()
            .min_by(|x, y| flux(x).total_cmp(&flux(y)))
            .map_or((0., 0.), |w| (flux(w), se(w)));
        for g in only
            .iter()
            .filter(|g| !other.iter().any(|h| id(h) == id(g)))
        {
            let band = 5. * se(g).hypot(cut.1);
            if flux(g) <= cut.0 + band {
                c.undefined += 1;
                c.sampling_differences.push(format!(
                    "top_signatures {} only in the {side}: flux {:e} within 5σ of the cut {:e}",
                    g["label"],
                    flux(g),
                    cut.0
                ));
            } else {
                c.fail(
                    "result.top_signatures",
                    &format!(
                        "signature {} only in the {side}: flux {:e} above the cut {:e} + 5σ",
                        g["label"],
                        flux(g),
                        cut.0
                    ),
                );
            }
        }
    }
    a["top_signatures"] = Value::Array(pa);
    b["top_signatures"] = Value::Array(pb);
}

/// The image grid of a PSF buffer: x and y relative to the PSF's reference
/// point (its position is compared separately, as a position).
fn psf_grid(json: &Value, buffer: &str) -> Option<(Vec<f64>, Vec<f64>)> {
    let psfs = json["psfs"].as_array()?;
    let p = psfs.iter().find(|p| p["psf"]["data"] == buffer)?;
    let size = p["psf"]["size"].as_u64()? as usize;
    let spacing = p["psf"]["spacing_mm"].as_f64()?;
    let c = floats(&p["psf"]["center_mm"])?;
    let axis = |k: usize| {
        (0..size)
            .map(|i| c[k] + (i as f64 - (size / 2) as f64) * spacing)
            .collect::<Vec<_>>()
    };
    Some((axis(0), axis(1)))
}

pub fn compare(
    cpu: &AnalysisOutput,
    gpu: &AnalysisOutput,
    analysis: &str,
    ctx: &JobContext,
) -> EngineResult<Comparison> {
    let radius: Vec<f64> = cpu
        .buffers
        .iter()
        .find(|b| b.name == "radius_mm")
        .and_then(|b| buffer_floats(&b.data))
        .unwrap_or_else(|| floats(&cpu.json["radius"]).unwrap_or_default());
    let mut c = Comparison {
        analysis: analysis.into(),
        scale: Scale {
            emitted: cpu.json["energy"]["emitted"].as_f64(),
            interval: radius
                .windows(2)
                .map(|v| (v[1] - v[0]).abs())
                .filter(|v| *v > 0.)
                .reduce(f64::min),
            sigma: 0.,
        },
        ..Default::default()
    };
    if !FIELDS
        .iter()
        .chain(SOLVER_SCORES)
        .chain(REFERENCE_ONLY)
        .any(|(a, _)| *a == analysis)
    {
        c.fail(
            "result",
            &format!("analysis {analysis} has no contract table"),
        );
        return Ok(c);
    }
    let (mut ja, mut jb) = (ranked(analysis, &cpu.json), ranked(analysis, &gpu.json));
    if analysis == "stray_light" {
        align_signatures(&mut c, &mut ja, &mut jb);
    }
    c.json(&ja, &jb, &ja, &jb, "");
    let find = |o: &AnalysisOutput, name: &str| o.buffers.iter().find(|b| b.name == name).cloned();
    for (path, buffer, ra, rb) in std::mem::take(&mut c.outlines) {
        let name = buffer.trim_start_matches('@');
        let (Some(va), Some(vb)) = (find(cpu, name), find(gpu, name)) else {
            c.fail(&path, "outline buffer missing");
            continue;
        };
        let dim = va.shape.last().copied().unwrap_or(3);
        let (Some(fa), Some(fb)) = (buffer_floats(&va.data), buffer_floats(&vb.data)) else {
            continue;
        };
        let pick = |f: &[f64], r: [f64; 2]| {
            let (o, n) = (r[0] as usize * dim, r[1] as usize * dim);
            points(f.get(o..o + n).unwrap_or(&[]), dim)
        };
        c.hausdorff(Outline(buffer), &pick(&fa, ra), &pick(&fb, rb), &path);
    }
    let columns: Vec<String> = cpu.json["columns"]
        .as_array()
        .map(|cs| {
            cs.iter()
                .filter_map(|c| c["name"].as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    for (bi, a) in cpu.buffers.iter().enumerate() {
        if bi % 16 == 0 {
            ctx.check()?;
        }
        let path = format!("buffer.{}", a.name);
        let key = format!("@{}", a.name);
        if c.is_axis(&key) {
            continue;
        }
        let Some(b) = gpu.buffers.iter().find(|b| b.name == a.name) else {
            c.fail(&path, "buffer missing");
            continue;
        };
        let rule = if a.name == "values" && !columns.is_empty() {
            None
        } else {
            match c.rule(&key) {
                Some(r) => Some(r),
                None => {
                    c.unmapped(&path);
                    continue;
                }
            }
        };
        let (Some(fa), Some(fb)) = (buffer_floats(&a.data), buffer_floats(&b.data)) else {
            if a.data != b.data {
                c.fail(&path, "typed buffer values differ");
            }
            continue;
        };
        if a.name == "values" && !columns.is_empty() {
            if a.shape != b.shape {
                c.fail(&path, "buffer shape differs");
                continue;
            }
            for (i, (x, y)) in fa.iter().zip(&fb).enumerate() {
                let name = &columns[i % columns.len()];
                let key = format!("@values.{name}");
                let Some(r) = c.rule(&key) else {
                    c.unmapped(&format!("{path}.{name}"));
                    continue;
                };
                c.number(r.class, *x, *y, &format!("{path}.{name}"));
            }
            continue;
        }
        let rule = rule.unwrap();
        match (rule.class, rule.axis) {
            (OutlineVertices, _) => continue,
            (Loop, _) => {
                let dim = a.shape.last().copied().unwrap_or(2);
                c.hausdorff(Loop, &points(&fa, dim), &points(&fb, dim), &path);
                continue;
            }
            (Jones, _) => {
                if a.shape != b.shape {
                    c.fail(&path, "buffer shape differs");
                } else {
                    c.jones(&fa, &fb, &path);
                }
                continue;
            }
            (FastAxis, _) => {
                let defined = |n: &str| {
                    cpu.buffers
                        .iter()
                        .find(|b| b.name == n)
                        .map(|b| b.data.clone())
                };
                let (Some(BufferData::U8(def)), Some(BufferData::F64(ret))) =
                    (defined("retarder_defined"), defined("retardance"))
                else {
                    c.fail(&path, "fast axis without retarder_defined/retardance");
                    continue;
                };
                for (i, (p, q)) in fa.chunks(3).zip(fb.chunks(3)).enumerate() {
                    if def[i] == 0 || ret[i] < 1e-3 {
                        c.undefined += 1;
                        continue;
                    }
                    for (x, y) in p.iter().zip(q) {
                        c.number(FastAxis, *x, *y, &format!("{path}[{i}]"));
                    }
                }
                continue;
            }
            (class, Some(PSF_GRID)) => {
                let (Some((ax, ay)), Some((bx, by))) =
                    (psf_grid(&cpu.json, &a.name), psf_grid(&gpu.json, &a.name))
                else {
                    c.fail(&path, "PSF grid not found");
                    continue;
                };
                c.map2(class, Position, &ax, &ay, &fa, &bx, &by, &fb, &path);
                continue;
            }
            (class, Some(axis)) if axis.starts_with('@') => {
                let axis_name = substitute(rule.path, &key, axis);
                let (Some(xa), Some(xb)) = (find(cpu, &axis_name[1..]), find(gpu, &axis_name[1..]))
                else {
                    c.fail(&path, &format!("axis {axis_name} missing"));
                    continue;
                };
                let (Some(xa), Some(xb)) = (buffer_floats(&xa.data), buffer_floats(&xb.data))
                else {
                    continue;
                };
                let axis_class = c.rule(&axis_name).map_or(Exact, |r| r.class);
                if a.shape.len() == 2 && b.shape.len() == 2 {
                    // A square map over the same coordinates along both axes.
                    c.map2(class, axis_class, &xa, &xa, &fa, &xb, &xb, &fb, &path);
                } else if fa.len() == xa.len() && fb.len() == xb.len() {
                    c.curve(class, axis_class, &xa, &fa, &xb, &fb, &path);
                } else {
                    c.fail(&path, "curve and axis lengths differ");
                }
                continue;
            }
            _ => (),
        }
        if a.shape != b.shape {
            c.fail(&path, "buffer identity/shape differs");
            continue;
        }
        if rule.class == Image {
            c.profile(&fa, &fb, &path);
            continue;
        }
        for (index, (x, y)) in fa.iter().zip(&fb).enumerate() {
            if index % 4096 == 0 {
                ctx.check()?;
            }
            c.number(rule.class, *x, *y, &path);
        }
    }
    for b in &gpu.buffers {
        if !cpu.buffers.iter().any(|a| a.name == b.name) {
            c.fail(&format!("buffer.{}", b.name), "extra buffer");
        }
    }
    Ok(c)
}

impl Comparison {
    /// A 2-D map on regular grids `(ax, ay)` and `(bx, by)` (x fastest):
    /// index by index on grids equal within the `axis` class, else the
    /// denser map bilinearly resampled onto the sparser one's coordinates
    /// (samples outside the other grid or next to its undefined cells are
    /// not compared).
    #[allow(clippy::too_many_arguments)]
    fn map2(
        &mut self,
        class: Class,
        axis: Class,
        ax: &[f64],
        ay: &[f64],
        fa: &[f64],
        bx: &[f64],
        by: &[f64],
        fb: &[f64],
        path: &str,
    ) {
        // Grids that agree within the axis class (a pupil-coordinate or image
        // position gate) sample the same coordinates.
        let scale = self.scale;
        let equal = |p: &[f64], q: &[f64]| {
            p.len() == q.len()
                && p.iter()
                    .zip(q)
                    .all(|(u, v)| u == v || (u - v).abs() <= axis.gate(*u, *v, &scale))
        };
        let same = equal(ax, bx) && equal(ay, by) && fa.len() == fb.len();
        // Resample the denser grid onto the sparser one's sample points.
        let step = |g: &[f64]| {
            if g.len() > 1 {
                (g[g.len() - 1] - g[0]).abs() / (g.len() - 1) as f64
            } else {
                0.
            }
        };
        let flip = !same && step(bx) > step(ax);
        let (fa, resampled): (Vec<f64>, Vec<f64>) = if same {
            (fa.to_vec(), fb.to_vec())
        } else {
            self.resampled += 1;
            self.sampling_differences
                .push(format!("{path}: grid resampled"));
            let (tx, ty, sx, sy, sf, tf) = if flip {
                (bx, by, ax, ay, fa, fb)
            } else {
                (ax, ay, bx, by, fb, fa)
            };
            let n = tx.len().max(1);
            let r: Vec<f64> = (0..tf.len())
                .map(|k| bilinear(sx, sy, sf, tx[k % n], ty[k / n]))
                .collect();
            if flip {
                (r, tf.to_vec())
            } else {
                (tf.to_vec(), r)
            }
        };
        let (pa, pb): (Vec<f64>, Vec<f64>) = fa
            .iter()
            .zip(&resampled)
            .filter(|(p, q)| {
                let keep = same || p.is_nan() == q.is_nan();
                if !keep {
                    self.undefined += 1;
                }
                keep
            })
            .map(|(p, q)| (*p, *q))
            .unzip();
        if class == Image {
            self.profile(&pa, &pb, path);
            return;
        }
        for (p, q) in pa.iter().zip(&pb) {
            self.number(class, *p, *q, path);
        }
    }

    /// Whether a normalised path is the sampling axis of other data; axes
    /// are compared with the data they sample.
    fn is_axis(&self, path: &str) -> bool {
        FIELDS
            .iter()
            .find(|(a, _)| *a == self.analysis)
            .is_some_and(|(_, rules)| {
                rules
                    .iter()
                    .any(|r| r.axis.is_some_and(|ax| ax != PSF_GRID && glob(ax, path)))
            })
    }
}

/// Every numeric leaf and buffer a result schema declares, in table form
/// (`fields[].cutoff`, `@tangential_*`), for the coverage test.
pub fn schema_leaves(schema: &Value) -> Vec<String> {
    fn numeric(s: &Value) -> bool {
        let t = |x: &Value| {
            matches!(
                x.get("type").and_then(Value::as_str),
                Some("number" | "integer")
            ) || x
                .get("type")
                .and_then(Value::as_array)
                .is_some_and(|ts| ts.iter().any(|t| t == "number" || t == "integer"))
        };
        t(s) || s
            .get("anyOf")
            .and_then(Value::as_array)
            .is_some_and(|a| a.iter().any(t))
    }
    fn walk(s: &Value, path: &str, out: &mut Vec<String>) {
        let inner = s
            .get("anyOf")
            .and_then(Value::as_array)
            .and_then(|a| {
                a.iter()
                    .find(|x| x.get("properties").is_some() || x.get("items").is_some())
            })
            .unwrap_or(s);
        if let Some(p) = inner.get("properties").and_then(Value::as_object) {
            for (k, v) in p {
                if TELEMETRY.contains(&k.as_str()) {
                    continue;
                }
                let sub = if path.is_empty() {
                    k.clone()
                } else {
                    format!("{path}.{k}")
                };
                walk(v, &sub, out);
            }
        } else if let Some(items) = inner.get("items") {
            walk(items, &format!("{path}[]"), out);
        } else if numeric(s) {
            out.push(path.to_string());
        }
    }
    let mut out = Vec::new();
    walk(schema, "", &mut out);
    for b in schema["x-buffers"].as_array().into_iter().flatten() {
        if let Some(n) = b["name"].as_str()
            && n != "values"
        {
            out.push(format!("@{n}"));
        }
    }
    out
}

/// Gated classes and their budgets, for the documentation test.
pub fn class_budget(class: Class, reference: f64) -> f64 {
    class.budget(reference, reference, &Scale::default())
}

/// The physical score view of a solver report (`SOLVER_SCORES`): execution
/// telemetry and returned designs legitimately differ between backends.
pub fn solver_scores(analysis: &str, v: &Value) -> Value {
    use serde_json::json;
    match analysis {
        "merit_evaluation" => {
            json!({"merit":v["merit"],"violation":v["violation"],"status":v["status"],"operands":v["operands"]})
        }
        "optimize" => json!({"start":v["start"],"final":v["final"]}),
        "tolerance" => {
            let mut mc = v["monte_carlo"].clone();
            if let Some(object) = mc.as_object_mut() {
                object.remove("seconds_per_trial");
            }
            json!({"sensitivity":v["sensitivity"],"monte_carlo":mc})
        }
        _ => {
            json!({"finalists":v["results"].as_array().map(|rs| rs.iter().map(|r| json!({"merit":r["merit"],"violation":r["violation"],"feasible":r["feasible"],"glasses":r["glasses"]})).collect::<Vec<_>>())})
        }
    }
}

/// Native attempts that did not produce the published result, as
/// `backend outcome: reason` lines. A failed, skipped (a cached
/// deterministic failure), infeasible or reference-completed native run is a
/// native failure even when the reference fallback succeeded; `downgraded`
/// (the static capability list ran an explicit request on the reference) is
/// a deliberate route, not a failure.
pub fn native_failures(info: &mklight_core::RunInfo) -> Vec<String> {
    info.failed_attempts
        .iter()
        .filter(|a| a.outcome != "downgraded")
        .map(|a| format!("{} {}: {}", a.backend, a.outcome, a.reason))
        .collect()
}
