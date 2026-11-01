//! Resident source generation and analytic closest-hit queries. These are
//! building blocks, not a completed stray-light solver. Reference transport
//! remains the only complete implementation until split-tree qualification.
use crate::{
    Scene, SourceSpec,
    geom::{Aabb, Mask, Primitive, Shape, onb},
};
use mklight_core::{
    JobContext, MkError, MkResult, Snapshot,
    math::Vec3,
    model::{ApertureShape, SurfaceShape},
};
use mklight_gpu::{ComputeBackend, Df32, resident::*, stray_ops as ops};

fn df(v: Vec3) -> [Df32; 3] {
    [v.x, v.y, v.z].map(Df32::from_f64)
}
fn unsupported(message: &str) -> MkError {
    MkError::Unsupported(format!("stray GPU: {message}"))
}
/// Small scene topology is built on the host. Unsupported sags are rejected
/// before uploading anything; sampled bounds never stand in for a proof that
/// a freeform is enclosed.
pub fn prepare_geometry(
    b: &dyn ComputeBackend,
    primitives: &[Primitive],
    ctx: &JobContext,
) -> MkResult<ops::Geometry> {
    ctx.check()?;
    if primitives.len() > mklight_gpu::consts::MKL_STRAY_MAX_PRIMITIVES as usize {
        return Err(unsupported("large scene device build is not implemented"));
    }
    let mut packed = Vec::with_capacity(primitives.len());
    for (i, p) in primitives.iter().enumerate() {
        if i % 64 == 0 {
            ctx.check()?;
        }
        let bounds = p.shape.local_bounds();
        let (kind, c, k, ri, ro, mask) = match &p.shape {
            Shape::Cylinder(c) => (2, 0., 0., 0., c.radius, None),
            Shape::Sag(s) => {
                let (kind, c, k) = match s.shape() {
                    SurfaceShape::Plane | SurfaceShape::Paraxial { .. } => (0, 0., 0.),
                    SurfaceShape::Conic { curvature, conic } => {
                        (if *curvature == 0. { 0 } else { 1 }, *curvature, *conic)
                    }
                    _ => {
                        return Err(unsupported(
                            "extended sag closest-hit bracketing is not implemented",
                        ));
                    }
                };
                (kind, c, k, s.r_inner(), s.r_outer(), s.mask())
            }
        };
        let (mask, aperture) = match mask {
            None => (0, [0., 0.]),
            Some(Mask::InsideRect { half_x, half_y }) => (3, [*half_x, *half_y]),
            Some(Mask::Inside(a)) => aperture_record(a, false),
            Some(Mask::Outside(a)) => aperture_record(a, true),
        };
        packed.push(ops::Primitive {
            kind,
            mask,
            origin: df(p.frame.origin),
            rotation: std::array::from_fn(|i| Df32::from_f64(p.frame.rot.m[i / 3][i % 3])),
            lower: df(bounds.min),
            upper: df(bounds.max),
            curvature: Df32::from_f64(c),
            conic: Df32::from_f64(k),
            inner_radius: Df32::from_f64(ri),
            outer_radius: Df32::from_f64(ro),
            aperture: aperture.map(Df32::from_f64),
        });
    }
    fn build(prims: &[Primitive], ids: &mut [usize], nodes: &mut Vec<ops::Node>) {
        let bounds = ids
            .iter()
            .fold(Aabb::EMPTY, |a, &i| a.union(&prims[i].aabb()));
        let index = nodes.len();
        nodes.push(ops::Node {
            lower: df(bounds.min),
            upper: df(bounds.max),
            escape: 0,
            primitive: u32::MAX,
        });
        if ids.len() == 1 {
            nodes[index].primitive = ids[0] as u32;
        } else {
            let e = bounds.extent();
            let axis = if e.x >= e.y && e.x >= e.z {
                0
            } else if e.y >= e.z {
                1
            } else {
                2
            };
            ids.sort_by(|&a, &b| {
                let ca = prims[a].aabb().centroid();
                let cb = prims[b].aabb().centroid();
                [ca.x, ca.y, ca.z][axis]
                    .total_cmp(&[cb.x, cb.y, cb.z][axis])
                    .then(a.cmp(&b))
            });
            let mid = ids.len() / 2;
            let (a, b) = ids.split_at_mut(mid);
            build(prims, a, nodes);
            build(prims, b, nodes);
        }
        nodes[index].escape = nodes.len() as u32;
    }
    let mut nodes = Vec::with_capacity(primitives.len() * 2);
    if !primitives.is_empty() {
        build(
            primitives,
            &mut (0..primitives.len()).collect::<Vec<_>>(),
            &mut nodes,
        );
    }
    ctx.check()?;
    ops::Geometry::new(b, &packed, &nodes, ctx)
}
fn aperture_record(a: &ApertureShape, outside: bool) -> (u32, [f64; 2]) {
    let (kind, v) = match a.with_edge_tolerance() {
        ApertureShape::Circular { radius } => (1, [radius, 0.]),
        ApertureShape::Annular {
            inner_radius,
            outer_radius,
        } => (2, [inner_radius, outer_radius]),
        ApertureShape::Rectangular { half_x, half_y } => (3, [half_x, half_y]),
        ApertureShape::Elliptical { semi_x, semi_y } => (4, [semi_x, semi_y]),
    };
    (kind | if outside { 256 } else { 0 }, v)
}
/// Append a source operation. Each logical source ID retains its reference
/// Philox stream and draw position so a future transport continuation can
/// consume the next draw without changing the depth-first stream.
pub fn generate(
    plan: &mut ResidentPlan,
    source: &SourceSpec,
    weights: &[f64],
    seed: u64,
    first_ray: u64,
    count: usize,
    tile: usize,
) -> MkResult<BufferId> {
    source.validate()?;
    let (kind, center, axis, normal, radius, inner, cone, distance) = match *source {
        SourceSpec::Collimated {
            direction,
            center,
            radius,
            start_distance,
            ..
        } => (
            0,
            center,
            direction,
            direction,
            radius,
            0.,
            0.,
            start_distance,
        ),
        SourceSpec::CollimatedThrough {
            direction,
            center,
            normal,
            radius,
            start_distance,
            ..
        } => (1, center, direction, normal, radius, 0., 0., start_distance),
        SourceSpec::Point {
            position,
            axis,
            half_angle_deg,
            ..
        } => (
            2,
            position,
            axis,
            axis,
            0.,
            0.,
            2. * (half_angle_deg.to_radians() * 0.5).sin().powi(2),
            0.,
        ),
        SourceSpec::LambertianDisk {
            center,
            normal,
            radius,
            inner_radius,
            max_angle_deg,
            ..
        } => (
            3,
            center,
            normal,
            normal,
            radius,
            inner_radius,
            max_angle_deg.to_radians().sin(),
            0.,
        ),
    };
    let total: f64 = weights.iter().sum();
    if weights.is_empty()
        || !total.is_finite()
        || total <= 0.
        || weights.iter().any(|v| !v.is_finite() || *v < 0.)
    {
        return Err(MkError::InvalidModel("stray source spectrum".into()));
    }
    let (t1, t2) = onb(normal.normalized());
    let mut acc = 0.;
    let mut cdf: Vec<_> = weights
        .iter()
        .map(|v| {
            acc += v / total;
            Df32::from_f64(acc)
        })
        .collect();
    *cdf.last_mut().unwrap() = Df32::from_f32(1.);
    let source = ops::Source {
        kind,
        wavelength_count: weights.len() as u32,
        center: df(center),
        axis: df(axis.normalized()),
        tangent: [df(t1), df(t2)],
        radius: Df32::from_f64(radius),
        inner_radius: Df32::from_f64(inner),
        cone: Df32::from_f64(cone),
        distance: Df32::from_f64(distance),
    };
    let output = plan.scratch(ops::record_desc::<ops::Ray>(count));
    plan.operations
        .push(ResidentOp::Stray(ops::Request::Generate(ops::Generate {
            source,
            cdf,
            seed,
            first_ray,
            ray_count: count,
            output,
            tile,
        })));
    Ok(output)
}
/// Complete source → first-hit job. RunInfo names this diagnostic pipeline,
/// not `stray_light`; no class flux or transport qualification is implied.
pub fn first_hits(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &Scene,
    source: &SourceSpec,
    seed: u64,
    first_ray: u64,
    count: usize,
    ctx: &JobContext,
) -> MkResult<PlanResult> {
    let geometry = prepare_geometry(b, scene.primitives(), ctx)?;
    let mut plan = ResidentPlan::new(snapshot);
    plan.seed = Some(seed);
    plan.settings = serde_json::json!({"pipeline":"stray_first_hits","version":mklight_gpu::consts::MKL_STRAY_VERSION,"source":source,"first_ray":first_ray,"rays":count,"scene_fingerprint":scene.fingerprint()});
    let rays = generate(
        &mut plan,
        source,
        &scene.wavelength_weights,
        seed,
        first_ray,
        count,
        65536,
    )?;
    let input = plan.view(rays)?;
    let hits = geometry.append(&mut plan, input, 65536);
    plan.outputs = vec![rays, hits];
    b.submit_plan(plan, ctx)?.wait()
}
