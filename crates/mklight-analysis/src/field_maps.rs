//! Full signed Cartesian field maps, with no symmetry shortcuts.
//! Pupil aberration measures the paraxially launched chief's real stop-height
//! error. Its Jacobian is d(stop x,y)/d(entrance-pupil x,y), evaluated about
//! that chief; column norms give the two pupil magnifications.
use crate::util::{check_count, gauss_legendre_01, run_info, wavelength_or_primary};
use mklight_core::job::RunTimer;
use mklight_core::model::RayAiming;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::{
    FieldPoint, ObjectSource, PreparedField, SequentialTracer, TraceOptions, TracerSettings,
};
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
            radial_samples: 4,
            azimuth_samples: 16,
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
/// Spectrally monochromatic irradiance probe; this reuses the exact same
/// projected-solid-angle quadrature as the radial analysis.
pub fn illumination_samples(
    t: &SequentialTracer,
    coordinates: &[[f64; 2]],
    wl: usize,
    radial: usize,
    azimuth: usize,
    ctx: &JobContext,
) -> MkResult<Vec<[f64; 3]>> {
    check_count("radial samples", radial, 1, 512)?;
    check_count("azimuth samples", azimuth, 4, 4096)?;
    let rt = SequentialTracer::with_settings(
        t.system_arc(),
        TracerSettings {
            aiming: Some(RayAiming::Real { tolerance: 1e-12 }),
            ..*t.settings()
        },
    )?;
    let (r, w) = gauss_legendre_01(radial);
    let dp = std::f64::consts::TAU / azimuth as f64;
    let nodes: Vec<_> = r
        .iter()
        .zip(&w)
        .flat_map(|(&r, &w)| {
            (0..azimuth).map(move |i| {
                let (s, c) = (dp * i as f64).sin_cos();
                (r * c, r * s, w * r * dp)
            })
        })
        .collect();
    let f = rt.prepare_field(FieldPoint::normalized(0.0, 1.0))?;
    let scale = match f.source {
        ObjectSource::Direction { dir } => dir.x.hypot(dir.y) / dir.z.abs(),
        ObjectSource::Point { point } => point.x.hypot(point.y),
    }
    .max(1e-3);
    let probe = |p: [f64; 2]| -> MkResult<[f64; 2]> {
        let f = rt.prepare_field(FieldPoint::normalized(p[0], p[1]))?;
        let (mut omega, mut area, mut total) = (0.0, 0.0, 0.0);
        for chunk in nodes.chunks(64) {
            ctx.check()?;
            let Some((r, _)) = crate::illumination::integrate_field(&rt, &f, wl, chunk, scale)
            else {
                return Ok([f64::NAN; 2]);
            };
            let weight: f64 = chunk.iter().map(|p| p.2).sum();
            omega += r[0];
            area += r[1] * weight;
            total += weight;
        }
        Ok([omega, 1.0 - area / total])
    };
    let axial = probe([0.0, 0.0])?[0];
    if !(axial > 0.0) {
        return Err(MkError::Numerical(
            "on-axis beam has no projected solid angle".into(),
        ));
    }
    coordinates
        .iter()
        .enumerate()
        .map(|(i, &p)| {
            ctx.check()?;
            let r = probe(p)?;
            ctx.report(
                (i + 1) as f64 / coordinates.len() as f64,
                "illumination map",
            );
            Ok([r[0] / axial, r[0], r[1]])
        })
        .collect()
}
fn pupil(t: &SequentialTracer, f: &PreparedField, wl: usize) -> Option<[f64; 4]> {
    let stop = t.system().system().stop;
    let opts = TraceOptions::unclipped().to_surface(stop);
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
    ctx.check()?;
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
        .any(|p| !(p[0] >= 0.0 && p[0].is_finite() && p[1] >= 0.0 && p[1] <= 90.0))
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
    let raw = SequentialTracer::with_settings(
        t.system_arc(),
        TracerSettings {
            aiming: Some(RayAiming::Off),
            ..*t.settings()
        },
    )?;
    let axial = pupil(
        &raw,
        &raw.prepare_field(FieldPoint::normalized(0.0, 0.0))?,
        wl,
    );
    let ri = if matches!(kind, FieldAnalysis::RelativeIllumination) {
        illumination_samples(
            t,
            &coords,
            wl,
            s.radial_samples,
            s.azimuth_samples,
            &ctx.subrange(0.0, 0.9),
        )?
    } else {
        Vec::new()
    };
    let mut images = Vec::new();
    let mut values = Vec::new();
    let mut warnings = Vec::new();
    for (i, &p) in coords.iter().enumerate() {
        ctx.check()?;
        let f = t.prepare_field(FieldPoint::normalized(p[0], p[1]))?;
        let mut records = Vec::new();
        let c = t.trace_with(
            &f,
            wl,
            0.0,
            0.0,
            &TraceOptions::unclipped(),
            Some(&mut records),
        );
        let ok = c.status.is_ok();
        images.push(if ok {
            [c.ray.pos.x, c.ray.pos.y]
        } else {
            [f64::NAN; 2]
        });
        if !ok {
            warnings.push(format!("field ({},{}): chief ray failed", p[0], p[1]));
        }
        match kind {
            FieldAnalysis::RelativeIllumination => values.extend(ri[i]),
            FieldAnalysis::PupilAberration => {
                let f = raw.prepare_field(FieldPoint::normalized(p[0], p[1]))?;
                match (pupil(&raw, &f, wl), axial) {
                    (Some(v), Some(a)) => values.extend([
                        v[0],
                        v[1],
                        v[2],
                        v[3],
                        v[2] / a[2] - 1.0,
                        v[3] / a[3] - 1.0,
                    ]),
                    _ => values.extend([f64::NAN; 6]),
                }
            }
            FieldAnalysis::ChiefRayAngle => {
                if ok {
                    let d = c.ray.dir;
                    let angle = records
                        .last()
                        .unwrap()
                        .cos_incidence
                        .clamp(0.0, 1.0)
                        .acos()
                        .to_degrees();
                    let radius = c.ray.pos.x.hypot(c.ray.pos.y);
                    let limit = interpolate_limit(&s.sensor_limit, radius);
                    values.extend([
                        angle,
                        d.x.atan2(d.z.abs()).to_degrees(),
                        d.y.atan2(d.z.abs()).to_degrees(),
                        limit,
                        limit - angle,
                    ]);
                } else {
                    values.extend([f64::NAN; 5]);
                }
            }
        }
        ctx.report(
            0.9 + 0.1 * (i + 1) as f64 / coords.len() as f64,
            kind.name(),
        );
    }
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
fn interpolate_limit(table: &[[f64; 2]], r: f64) -> f64 {
    if table.is_empty() {
        return f64::NAN;
    }
    if r <= table[0][0] {
        return table[0][1];
    }
    for p in table.windows(2) {
        if r <= p[1][0] {
            return p[0][1] + (p[1][1] - p[0][1]) * (r - p[0][0]) / (p[1][0] - p[0][0]);
        }
    }
    table.last().unwrap()[1]
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
            radial_samples: 12,
            azimuth_samples: 48,
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
}
