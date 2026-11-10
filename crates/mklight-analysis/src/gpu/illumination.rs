use super::*;
use crate::{
    illumination::{RelativeIlluminationResult, RelativeIlluminationSettings},
    util::{check_count, field_scale, field_sweep, field_unit, wavelength_or_primary},
};
/// Matched RI rows `[solid angle, surviving area share, CRA, cos4]`, plus
/// on-axis normalisation, image coordinates and reproducibility evidence.
pub type IlluminationRows = (Vec<[f64; 4]>, f64, Vec<[f64; 2]>, RunInfo);
/// Resident RI rows for field coordinates (see [`IlluminationRows`]): the
/// CPU solves the image points, the GPU traces the pencils
/// (`illumination_split`).
pub fn illumination_rows(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    coordinates: &[[f64; 2]],
    wl: usize,
    radial: usize,
    azimuth: usize,
    ctx: &JobContext,
) -> MkResult<IlluminationRows> {
    check_count("radial samples", radial, 1, 512)?;
    check_count("azimuth samples", azimuth, 4, 4096)?;
    super::illumination_split::rows(b, t, coordinates, wl, radial, azimuth, ctx)
}

pub fn relative_illumination_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &RelativeIlluminationSettings,
    ctx: &JobContext,
) -> MkResult<RelativeIlluminationResult> {
    check_count("field samples", s.field_samples, 2, 10_001)?;
    let wl = wavelength_or_primary(t, s.wavelength)?;
    let sweep = field_sweep(t, s.field_samples, s.field_axes);
    let (rows, omega0, _, run) = illumination_rows(
        b,
        t,
        &sweep.coordinates,
        wl,
        s.radial_samples,
        s.azimuth_samples,
        &ctx.subrange(0.0, 0.95),
    )?;
    ctx.check()?;
    ctx.report(1.0, "relative illumination");
    Ok(RelativeIlluminationResult {
        field_values: sweep.heights.iter().map(|h| h * field_scale(t)).collect(),
        field_unit: field_unit(t.system().system().fields.kind).into(),
        normalized_field: sweep.heights,
        field_coordinates: sweep.coordinates,
        wavelength: wl,
        relative_illumination: rows.iter().map(|r| r[0] / omega0).collect(),
        projected_solid_angle: rows.iter().map(|r| r[0]).collect(),
        vignetting_fraction: rows.iter().map(|r| 1.0 - r[1]).collect(),
        chief_ray_angle_deg: rows.iter().map(|r| r[2]).collect(),
        cos4_object: rows.iter().map(|r| r[3]).collect(),
        run_info: stamp(run, "relative_illumination", s, t.warnings().to_vec()),
    })
}
