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
/// Relative distance (of the neighbours' chord) below which a hull vertex
/// counts as collinear with its neighbours: far above f64 and double-float
/// rounding of local hit coordinates, far below any optical tolerance
/// (1e-9 of a 10 mm chord is 1e-8 mm).
const COLLINEAR: f64 = 1e-9;
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FootprintOverlay {
    pub sets: Vec<OverlaySet>,
    pub points: Vec<[f64; 3]>,
    pub outlines: Vec<[f64; 3]>,
    pub run_info: RunInfo,
}
/// Monotone-chain hull, counterclockwise and unclosed. Collinear interior
/// points are removed, including vertices collinear with their neighbours
/// only to rounding (within [`COLLINEAR`] of their chord), so the outline
/// of a pupil grid row does not depend on the last bits of its hits: the
/// hull keeps each vertex's sag, and an interior row point kept by rounding
/// would bend the outline by the sag's curvature along the row. The exact
/// hull comes first, so rounding never reorders points along a line. Fewer
/// than three hits yield a point or a segment.
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
    // Vertices within rounding of their neighbours' chord (on a convex hull
    // they project inside it), until none is left.
    let mut hull = lower;
    let mut i = 0;
    let mut kept = 0;
    while hull.len() > 3 && kept < hull.len() {
        let n = hull.len();
        let (a, b, c) = (hull[(i + n - 1) % n], hull[i % n], hull[(i + 1) % n]);
        let chord = (c.x - a.x).hypot(c.y - a.y);
        if cross(a, b, c) <= COLLINEAR * chord * chord {
            hull.remove(i % n);
            kept = 0;
        } else {
            i += 1;
            kept += 1;
        }
        i %= hull.len();
    }
    hull
}
/// Every passing hit (records stop before a clipping surface) of the
/// footprint's pupil pattern on every surface after the object, per field
/// and wavelength, in the global frame, with its convex outline.
pub fn footprint_overlay_job(
    t: &SequentialTracer,
    s: &FootprintSettings,
    ctx: &JobContext,
) -> MkResult<FootprintOverlay> {
    let timer = RunTimer::start();
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
    /// A grid row collinear only to rounding (as hits recomputed through
    /// another frame are) gives the same outline as the exact row: the
    /// outline does not depend on the last bits of the hits.
    #[test]
    fn hull_drops_points_collinear_to_rounding() {
        let row = |noise: f64| {
            let mut p: Vec<Vec3> = (0..=8)
                .flat_map(|i| {
                    let x = -4.0 + i as f64;
                    // Odd hits a few ulps above or below the row.
                    let e = if i % 2 == 1 {
                        noise * (i as f64 - 4.0)
                    } else {
                        0.0
                    };
                    [Vec3::new(x, -3.0 + e, x * x), Vec3::new(x, 3.0 - e, x * x)]
                })
                .collect();
            p.push(Vec3::new(0.0, 0.0, 0.0));
            convex_hull(&p)
        };
        let exact = row(0.0);
        assert_eq!(exact.len(), 4);
        for noise in [4e-16, -4e-16, 3e-15, -3e-15] {
            let rounded = row(noise);
            assert_eq!(rounded.len(), exact.len(), "noise {noise}");
            for (a, b) in rounded.iter().zip(&exact) {
                assert!((a.x - b.x).abs() < 1e-12 && (a.y - b.y).abs() < 1e-12);
                assert_eq!(a.z, b.z);
            }
        }
        // A real corner well above rounding is kept.
        let mut p = row(0.0);
        p.push(Vec3::new(0.0, -3.0 - 1e-6, 0.0));
        assert_eq!(convex_hull(&p).len(), 5);
    }
    /// Rounding of a grid column's x scrambles the sort order along the
    /// column; the extreme hits of each column must survive.
    #[test]
    fn hull_keeps_column_extremes_whose_order_rounding_scrambled() {
        let mut p = Vec::new();
        for i in 0..9 {
            for j in 0..9 {
                // Deterministic jitter of a few ulps, unrelated to j's order.
                let e =
                    [3.0, -2.0, 1.0, -4.0, 0.0, 2.0, -1.0, 4.0, -3.0][(i * 5 + j * 7) % 9] * 2e-16;
                let (x, y) = (i as f64 - 4.0 + e, j as f64 - 4.0);
                p.push(Vec3::new(x, y, 0.1 * (x * x + y * y)));
            }
        }
        let hull = convex_hull(&p);
        assert_eq!(hull.len(), 4, "{hull:?}");
        for v in &hull {
            assert!((v.x.abs() - 4.0).abs() < 1e-12 && (v.y.abs() - 4.0).abs() < 1e-12);
        }
    }
    /// On a centred system the global hits are the local footprint's points
    /// lifted onto the surface sag at the vertex's global position.
    #[test]
    fn global_hits_are_the_local_footprint_on_the_sag() {
        let t = crate::tracer_for(mklight_core::samples::cooke_triplet()).unwrap();
        let s = FootprintSettings {
            surface: 4,
            rings: 3,
            ..Default::default()
        };
        let ctx = JobContext::new();
        let local = crate::illumination::footprint_job(&t, &s, &ctx).unwrap();
        let o = footprint_overlay_job(&t, &s, &ctx).unwrap();
        let surface = &t.system().surfaces[4];
        let vertex = surface.frame.to_global_point(Vec3::ZERO);
        let sets: Vec<_> = o.sets.iter().filter(|x| x.surface == 4).collect();
        assert_eq!(sets.len(), local.rays.len());
        for (set, rays) in sets.iter().zip(&local.rays) {
            let hits = &o.points[set.points[0]..set.points[0] + set.points[1]];
            assert_eq!(hits.len(), rays.points.len());
            for (g, l) in hits.iter().zip(&rays.points) {
                let sag = surface.shape.sag(l[0] * l[0] + l[1] * l[1]).unwrap();
                assert!((g[0] - l[0]).abs() < 1e-12 && (g[1] - l[1]).abs() < 1e-12);
                assert!((g[2] - vertex.z - sag).abs() < 1e-9);
            }
        }
    }
}
