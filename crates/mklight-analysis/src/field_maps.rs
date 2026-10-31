//! Signed Cartesian field maps (or x / y sweeps), with no symmetry shortcuts.
//!
//! * Pupil aberration: the chief ray launched without aiming, at the centre
//!   of the paraxial entrance pupil, misses the stop centre by the stop-height
//!   error (mm, stop frame). The pupil magnifications `m_x`, `m_y` are the
//!   column norms of `∂(stop x, y)/∂(entrance-pupil x, y)` about that ray
//!   (central differences); their variation is relative to the on-axis
//!   value. An aberration-free pupil has zero error and no variation.
//! * Relative illumination: the radial analysis's projected-solid-angle
//!   integral (`illumination` module docs), on a 2-D grid of fields.
//! * Chief-ray angle: the real chief ray (through the stop centre, whatever
//!   the design's aiming) against the image-surface normal, its x / y
//!   direction angles, and an optional sensor CRA limit with the margin.
use crate::illumination::{illumination_rows, real_aimed};
use crate::util::{check_count, run_info, wavelength_or_primary};
use mklight_core::job::RunTimer;
use mklight_core::model::RayAiming;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::{FieldPoint, PreparedField, SequentialTracer, TraceOptions, TracerSettings};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MapMode {
    #[default]
    Grid,
    X,
    Y,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct FieldMapSettings {
    pub grid: [usize; 2],
    pub extent: [f64; 2],
    pub mode: MapMode,
    pub wavelength: Option<usize>,
    pub radial_samples: usize,
    pub azimuth_samples: usize,
    /// [image radius mm, maximum incidence degrees], linear interpolation,
    /// held constant outside the table. Empty = no sensor constraint.
    pub sensor_limit: Vec<[f64; 2]>,
}
impl Default for FieldMapSettings {
    fn default() -> Self {
        Self {
            grid: [5, 5],
            extent: [1.0, 1.0],
            mode: MapMode::Grid,
            wavelength: None,
            radial_samples: 12,
            azimuth_samples: 48,
            sensor_limit: Vec::new(),
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Column {
    pub name: String,
    pub unit: String,
    pub definition: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldMapResult {
    pub shape: [usize; 2],
    pub wavelength: usize,
    pub columns: Vec<Column>,
    pub field_coordinates: Vec<[f64; 2]>,
    #[serde(with = "crate::util::nf::vec2")]
    pub image_mm: Vec<[f64; 2]>,
    #[serde(with = "crate::util::nf::vec")]
    pub values: Vec<f64>,
    pub run_info: RunInfo,
}
#[derive(Clone, Copy)]
pub enum FieldAnalysis {
    PupilAberration,
    RelativeIllumination,
    ChiefRayAngle,
}
impl FieldAnalysis {
    pub fn name(self) -> &'static str {
        match self {
            Self::PupilAberration => "pupil_aberration",
            Self::RelativeIllumination => "relative_illumination_map",
            Self::ChiefRayAngle => "chief_ray_angle",
        }
    }
    fn columns(self) -> Vec<Column> {
        let rows: &[(&str, &str, &str)] = match self {
            Self::PupilAberration => &[
                (
                    "stop_error_x",
                    "mm",
                    "Un-aimed chief x at stop minus stop centre",
                ),
                (
                    "stop_error_y",
                    "mm",
                    "Un-aimed chief y at stop minus stop centre",
                ),
                ("magnification_x", "1", "Norm of d(stop x,y)/d(entrance x)"),
                ("magnification_y", "1", "Norm of d(stop x,y)/d(entrance y)"),
                (
                    "variation_x",
                    "1",
                    "Magnification x divided by on-axis value minus one",
                ),
                (
                    "variation_y",
                    "1",
                    "Magnification y divided by on-axis value minus one",
                ),
            ],
            Self::RelativeIllumination => &[
                (
                    "relative_illumination",
                    "1",
                    "Projected solid angle divided by on-axis value",
                ),
                (
                    "projected_solid_angle",
                    "sr",
                    "Projected exit-pupil solid angle at the image point",
                ),
                ("vignetting_fraction", "1", "Clipped fraction of stop area"),
            ],
            Self::ChiefRayAngle => &[
                (
                    "incidence",
                    "deg",
                    "Chief incidence against actual image-surface normal",
                ),
                ("angle_x", "deg", "atan2(dx, abs(dz)) in image vertex frame"),
                ("angle_y", "deg", "atan2(dy, abs(dz)) in image vertex frame"),
                (
                    "limit",
                    "deg",
                    "Interpolated sensor CRA limit at image radius; NaN without limit",
                ),
                (
                    "margin",
                    "deg",
                    "Limit minus incidence; negative violates sensor limit",
                ),
            ],
        };
        rows.iter()
            .map(|&(n, u, d)| Column {
                name: n.into(),
                unit: u.into(),
                definition: d.into(),
            })
            .collect()
    }
}
/// Un-aimed chief at the stop `[x, y]` and the stop/entrance-pupil
/// magnifications `[m_x, m_y]` (module docs), from central differences.
fn pupil(t: &SequentialTracer, f: &PreparedField, wl: usize) -> Option<[f64; 4]> {
    let opts = TraceOptions::unclipped().to_surface(t.system().system().stop);
    let hit = |x, y| {
        let r = t.trace(f, wl, x, y, &opts);
        r.status.is_ok().then_some(r.ray.pos)
    };
    let c = hit(0.0, 0.0)?;
    let h = 1e-4;
    let (xa, xb, ya, yb) = (hit(h, 0.0)?, hit(-h, 0.0)?, hit(0.0, h)?, hit(0.0, -h)?);
    let scale = 2.0 * h * t.pupil().r_ep;
    Some([
        c.x,
        c.y,
        (xa.x - xb.x).hypot(xa.y - xb.y) / scale,
        (ya.x - yb.x).hypot(ya.y - yb.y) / scale,
    ])
}

pub fn field_map_job(
    t: &SequentialTracer,
    kind: FieldAnalysis,
    s: &FieldMapSettings,
    ctx: &JobContext,
) -> MkResult<FieldMapResult> {
    let timer = RunTimer::start();
    for &n in &s.grid {
        check_count("field grid", n, 1, 101)?;
    }
    if s.extent.iter().any(|x| !(*x > 0.0 && x.is_finite())) {
        return Err(MkError::InvalidParam(
            "field extent must be positive".into(),
        ));
    }
    if s.sensor_limit
        .iter()
        .any(|p| !(p[0] >= 0.0 && p[0].is_finite() && (0.0..=90.0).contains(&p[1])))
        || s.sensor_limit.windows(2).any(|w| w[0][0] >= w[1][0])
    {
        return Err(MkError::InvalidParam(
            "sensor limit needs increasing nonnegative heights and angles in [0,90]".into(),
        ));
    }
    let wl = wavelength_or_primary(t, s.wavelength)?;
    let shape = match s.mode {
        MapMode::Grid => s.grid,
        MapMode::X => [s.grid[0], 1],
        MapMode::Y => [1, s.grid[1]],
    };
    let coordinate = |i, n, e| {
        if n == 1 {
            0.0
        } else {
            e * (2.0 * i as f64 / (n - 1) as f64 - 1.0)
        }
    };
    let coords: Vec<_> = (0..shape[1])
        .flat_map(|y| {
            (0..shape[0]).map(move |x| {
                [
                    coordinate(x, shape[0], s.extent[0]),
                    coordinate(y, shape[1], s.extent[1]),
                ]
            })
        })
        .collect();
    let mut warnings = Vec::new();
    // The chief ray passes the stop centre (real aiming), as in the radial
    // relative-illumination analysis, whatever the design's aiming.
    let rt = real_aimed(t)?;
    let mut images = Vec::with_capacity(coords.len());
    let mut chiefs = Vec::with_capacity(coords.len());
    for &p in &coords {
        ctx.check()?;
        if matches!(kind, FieldAnalysis::RelativeIllumination) && p[0].hypot(p[1]) > 1.0 + 1e-12 {
            images.push([f64::NAN; 2]);
            chiefs.push(None);
            continue;
        }
        let f = rt.prepare_field(FieldPoint::normalized(p[0], p[1]))?;
        let mut records = Vec::new();
        let c = rt.trace_with(
            &f,
            wl,
            0.0,
            0.0,
            &TraceOptions::unclipped(),
            Some(&mut records),
        );
        if c.status.is_ok() {
            images.push([c.ray.pos.x, c.ray.pos.y]);
            chiefs.push(Some((c.ray, records.last().unwrap().cos_incidence)));
        } else {
            warnings.push(format!("field ({},{}): chief ray failed", p[0], p[1]));
            images.push([f64::NAN; 2]);
            chiefs.push(None);
        }
    }
    let values = match kind {
        FieldAnalysis::RelativeIllumination => {
            let inside: Vec<_> = coords
                .iter()
                .copied()
                .filter(|p| p[0].hypot(p[1]) <= 1.0 + 1e-12)
                .collect();
            let (rows, omega0) = illumination_rows(
                t,
                &inside,
                wl,
                s.radial_samples,
                s.azimuth_samples,
                &mut warnings,
                ctx,
            )?;
            let mut rows = rows.iter();
            coords
                .iter()
                .flat_map(|p| {
                    if p[0].hypot(p[1]) > 1.0 + 1e-12 {
                        [f64::NAN; 3]
                    } else {
                        let r = rows.next().unwrap();
                        [r[0] / omega0, r[0], 1.0 - r[1]]
                    }
                })
                .collect()
        }
        FieldAnalysis::PupilAberration => {
            let raw = SequentialTracer::with_settings(
                t.system_arc(),
                TracerSettings {
                    aiming: Some(RayAiming::Off),
                    ..*t.settings()
                },
            )?;
            let probe = |p: [f64; 2]| -> MkResult<_> {
                let f = raw.prepare_field(FieldPoint::normalized(p[0], p[1]))?;
                Ok(pupil(&raw, &f, wl))
            };
            let axial = probe([0.0; 2])?;
            let mut values = Vec::with_capacity(6 * coords.len());
            for &p in &coords {
                ctx.check()?;
                values.extend(match (probe(p)?, axial) {
                    (Some(v), Some(a)) => {
                        [v[0], v[1], v[2], v[3], v[2] / a[2] - 1.0, v[3] / a[3] - 1.0]
                    }
                    _ => [f64::NAN; 6],
                });
            }
            values
        }
        FieldAnalysis::ChiefRayAngle => chiefs
            .iter()
            .zip(&images)
            .flat_map(|(chief, p)| match chief {
                Some((ray, cos_incidence)) => {
                    let angle = cos_incidence.clamp(0.0, 1.0).acos().to_degrees();
                    let limit = interpolate_limit(&s.sensor_limit, p[0].hypot(p[1]));
                    let d = ray.dir;
                    [
                        angle,
                        d.x.atan2(d.z.abs()).to_degrees(),
                        d.y.atan2(d.z.abs()).to_degrees(),
                        limit,
                        limit - angle,
                    ]
                }
                None => [f64::NAN; 5],
            })
            .collect(),
    };
    ctx.report(1.0, kind.name());
    Ok(FieldMapResult {
        shape,
        wavelength: wl,
        columns: kind.columns(),
        field_coordinates: coords,
        image_mm: images,
        values,
        run_info: run_info(timer, t, kind.name(), s, None, warnings),
    })
}
/// Linear interpolation, held constant outside the table; NaN when empty.
fn interpolate_limit(table: &[[f64; 2]], r: f64) -> f64 {
    let (Some(first), Some(last)) = (table.first(), table.last()) else {
        return f64::NAN;
    };
    if r <= first[0] {
        return first[1];
    }
    table
        .windows(2)
        .find(|p| r <= p[1][0])
        .map_or(last[1], |p| {
            p[0][1] + (p[1][1] - p[0][1]) * (r - p[0][0]) / (p[1][0] - p[0][0])
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::{ApertureSpec, Field, MaterialRef, OpticalSystem, Surface, SurfaceShape};
    fn ideal() -> SequentialTracer {
        let mut s = OpticalSystem::new_empty();
        s.aperture = ApertureSpec::EntrancePupilDiameter { value: 10.0 };
        s.fields.fields = vec![Field::new(0.0, 0.0), Field::new(0.0, 20.0)];
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(
                1,
                SurfaceShape::Paraxial {
                    focal_length: 100.0,
                },
                100.0,
                MaterialRef::Air,
            ),
            Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        crate::tracer_for(s).unwrap()
    }
    #[test]
    fn pupil_is_undistorted_at_the_stop_of_an_ideal_lens() {
        let t = ideal();
        let r = field_map_job(
            &t,
            FieldAnalysis::PupilAberration,
            &Default::default(),
            &JobContext::new(),
        )
        .unwrap();
        for row in r.values.chunks(6) {
            assert!(row[0].abs() < 1e-12 && row[1].abs() < 1e-12);
            assert!((row[2] - 1.0).abs() < 1e-10 && (row[3] - 1.0).abs() < 1e-10);
            assert!(row[4].abs() < 1e-10 && row[5].abs() < 1e-10);
        }
    }
    #[test]
    fn chief_angles_and_sensor_margin_match_geometry() {
        let t = ideal();
        let s = FieldMapSettings {
            grid: [3, 3],
            sensor_limit: vec![[0.0, 10.0], [100.0, 30.0]],
            ..Default::default()
        };
        let r = field_map_job(&t, FieldAnalysis::ChiefRayAngle, &s, &JobContext::new()).unwrap();
        for (i, row) in r.values.chunks(5).enumerate() {
            let p = r.image_mm[i];
            let exact = (p[0].hypot(p[1]) / 100.0).atan().to_degrees();
            assert!((row[0] - exact).abs() < 1e-10);
            assert!((row[3] - (10.0 + 0.2 * p[0].hypot(p[1]))).abs() < 1e-10);
            assert!((row[4] - (row[3] - row[0])).abs() < 1e-12);
        }
    }
    #[test]
    fn illumination_matches_disk_view_factor_in_both_axes() {
        let t = ideal();
        let s = FieldMapSettings {
            grid: [3, 3],
            ..Default::default()
        };
        let r = field_map_job(
            &t,
            FieldAnalysis::RelativeIllumination,
            &s,
            &JobContext::new(),
        )
        .unwrap();
        let disk = |h: f64| {
            let q = 10000.0 + h * h;
            std::f64::consts::PI
                * 0.5
                * (1.0 - (q - 25.0) / ((q + 25.0).powi(2) - 100.0 * h * h).sqrt())
        };
        for (i, row) in r.values.chunks(3).enumerate() {
            if r.field_coordinates[i][0].hypot(r.field_coordinates[i][1]) > 1.0 + 1e-12 {
                assert!(row.iter().all(|v| v.is_nan()));
                continue;
            }
            let h = r.image_mm[i][0].hypot(r.image_mm[i][1]);
            let expected = disk(h);
            assert!((row[1] / expected - 1.0).abs() < 1e-8);
            assert!((row[0] - expected / disk(0.0)).abs() < 1e-8);
        }
    }
    #[test]
    fn maps_honour_cancellation() {
        let ctx = JobContext::new();
        ctx.cancel.cancel();
        assert!(matches!(
            field_map_job(
                &ideal(),
                FieldAnalysis::ChiefRayAngle,
                &Default::default(),
                &ctx
            ),
            Err(MkError::Cancelled)
        ));
    }
    /// The map and the radial curve share one quadrature, and the CRA is the
    /// real (stop-centred) chief ray even when the design does not aim rays:
    /// the Cooke triplet's un-aimed chief misses the stop centre by 0.23 mm
    /// at full field, which used to tilt the reported CRA by 0.33°.
    #[test]
    fn maps_match_the_radial_curve_and_real_chief() {
        let t = crate::tracer_for(mklight_core::samples::cooke_triplet()).unwrap();
        assert!(matches!(t.aiming(), RayAiming::Off));
        let radial = crate::illumination::relative_illumination(
            &t,
            &crate::illumination::RelativeIlluminationSettings {
                field_samples: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let s = FieldMapSettings {
            grid: [5, 5],
            ..Default::default()
        };
        let ri = field_map_job(
            &t,
            FieldAnalysis::RelativeIllumination,
            &s,
            &JobContext::new(),
        )
        .unwrap();
        let cra = field_map_job(&t, FieldAnalysis::ChiefRayAngle, &s, &JobContext::new()).unwrap();
        let node = |p: [f64; 2]| ri.field_coordinates.iter().position(|&q| q == p).unwrap();
        for (k, &[_, h]) in radial.field_coordinates.iter().enumerate() {
            for i in [node([0.0, h]), node([h, 0.0]), node([0.0, -h])] {
                assert!((ri.values[3 * i] - radial.relative_illumination[k]).abs() < 1e-12);
                assert!((cra.values[5 * i] - radial.chief_ray_angle_deg[k]).abs() < 1e-9);
            }
        }
        assert!(radial.chief_ray_angle_deg[2] > 19.3);
    }
}
