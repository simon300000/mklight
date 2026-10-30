//! Global-frame footprint clouds and convex outlines on every surface.
//! Hulls are computed in the surface's local x/y plane, retaining each
//! vertex's traced sag; global conversion includes placements and mirrors.
use crate::illumination::FootprintSettings;
use crate::util::{default_fields, pupil_points, run_info};
use mklight_core::job::RunTimer;
use mklight_core::{JobContext, MkResult, RunInfo, Vec3};
use mklight_trace::SequentialTracer;
use serde::{Deserialize, Serialize};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OverlaySet {
    pub surface: usize,
    pub field: usize,
    pub wavelength: usize,
    pub points: [usize; 2],
    pub outline: [usize; 2],
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FootprintOverlay {
    pub sets: Vec<OverlaySet>,
    pub points: Vec<[f64; 3]>,
    pub outlines: Vec<[f64; 3]>,
    pub run_info: RunInfo,
}
/// Monotone-chain hull, counterclockwise and unclosed. Collinear interior
/// points are removed. Fewer than three hits yield a point or a segment.
pub fn convex_hull(points: &[Vec3]) -> Vec<Vec3> {
    let mut p = points.to_vec();
    p.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    p.dedup_by(|a, b| a.x == b.x && a.y == b.y);
    if p.len() < 3 {
        return p;
    }
    let cross = |a: Vec3, b: Vec3, c: Vec3| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    for &q in &p {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], q) <= 0.0 {
            lower.pop();
        }
        lower.push(q);
    }
    for &q in p.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], q) <= 0.0 {
            upper.pop();
        }
        upper.push(q);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}
pub fn footprint_overlay_job(
    t: &SequentialTracer,
    s: &FootprintSettings,
    ctx: &JobContext,
) -> MkResult<FootprintOverlay> {
    let timer = RunTimer::start();
    ctx.check()?;
    let tracer = default_fields(t, &s.fields)?;
    let t = tracer.as_ref();
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let ns = t.system().num_surfaces();
    let mut sets = Vec::new();
    let mut points = Vec::new();
    let mut outlines = Vec::new();
    for (fi, &f) in fields.iter().enumerate() {
        for (wi, &w) in wls.iter().enumerate() {
            let mut hits = vec![Vec::new(); ns];
            for p in &pupil {
                ctx.check()?;
                let mut records = Vec::new();
                t.trace_with(t.field(f), w, p[0], p[1], &t.options(), Some(&mut records));
                for (i, r) in records.iter().enumerate().skip(1) {
                    hits[i].push(r.point);
                }
            }
            for (i, hit) in hits.iter().enumerate().skip(1) {
                ctx.check()?;
                let frame = &t.system().surfaces[i].frame;
                let global = |p: Vec3| {
                    let q = frame.to_global_point(p);
                    [q.x, q.y, q.z]
                };
                let hull = convex_hull(hit);
                sets.push(OverlaySet {
                    surface: i,
                    field: f,
                    wavelength: w,
                    points: [points.len(), hit.len()],
                    outline: [outlines.len(), hull.len()],
                });
                points.extend(hit.iter().copied().map(global));
                outlines.extend(hull.into_iter().map(global));
            }
            ctx.report(
                (fi * wls.len() + wi + 1) as f64 / (fields.len() * wls.len()) as f64,
                "global footprints",
            );
        }
    }
    Ok(FootprintOverlay {
        sets,
        points,
        outlines,
        run_info: run_info(timer, t, "footprint", s, None, Vec::new()),
    })
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hull_omits_interior_and_collinear_hits() {
        let p = [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
            (0.5, 0.5),
            (0.5, 0.0),
        ];
        let hull = convex_hull(&p.map(|(x, y)| Vec3::new(x, y, 0.0)));
        assert_eq!(hull.len(), 4);
        let area: f64 = (0..4)
            .map(|i| hull[i].x * hull[(i + 1) % 4].y - hull[i].y * hull[(i + 1) % 4].x)
            .sum();
        assert!((area - 2.0).abs() < 1e-14);
    }
}
#[cfg(test)]
mod global_tests {
    use super::*;
    #[test]
    fn tilted_surface_cloud_matches_recorded_trace() {
        let mut sys = mklight_core::samples::cooke_triplet();
        sys.surfaces[1].placement = Some(mklight_core::Placement {
            tilt_y: 3.0,
            ..Default::default()
        });
        let t = crate::tracer_for(sys).unwrap();
        let s = FootprintSettings {
            rings: 2,
            fields: crate::util::FieldSelection::Index { index: 0 },
            wavelengths: crate::util::WavelengthSelection::Primary,
            ..Default::default()
        };
        let r = footprint_overlay_job(&t, &s, &JobContext::new()).unwrap();
        let pupil = pupil_points(s.pattern, s.rings, s.seed);
        let mut records = Vec::new();
        t.trace_with(
            t.field(0),
            t.system().primary_wavelength,
            pupil[0][0],
            pupil[0][1],
            &t.options(),
            Some(&mut records),
        );
        let set = &r.sets[0];
        assert_eq!(set.surface, 1);
        let p = t.system().surfaces[1]
            .frame
            .to_global_point(records[1].point);
        assert_eq!(r.points[set.points[0]], [p.x, p.y, p.z]);
        assert!(r.sets.iter().all(|s| s.surface > 0));
        assert!(r.sets.iter().any(|s| s.surface == t.system().image_index()));
    }
}
