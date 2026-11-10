//! Resident stray-light transport and analytic compute BVH; the CPU forms
//! the per-source detector estimators from the device records. Capability
//! flags and routing belong to the engine.
use crate::{
    Scene, SourceSpec,
    geom::{Aabb, Mask, Primitive, Shape, onb},
};
use mklight_core::{
    JobContext, MkError, MkResult, Snapshot,
    math::Vec3,
    model::{ApertureShape, SurfaceShape},
};
mod bounds;
use mklight_gpu::{ComputeBackend, Df32, resident::*, stray_ops as ops};

fn df(v: Vec3) -> [Df32; 3] {
    [v.x, v.y, v.z].map(Df32::from_f64)
}
fn unsupported(message: &str) -> MkError {
    MkError::Unsupported(format!("stray GPU: {message}"))
}
/// Small scene topology is built on the host. Unsupported sags are rejected
/// before publishing geometry; sampled bounds never stand in for a proof that
/// a freeform is enclosed.
pub fn prepare_geometry(
    b: &dyn ComputeBackend,
    primitives: &[Primitive],
    ctx: &JobContext,
) -> MkResult<ops::Geometry> {
    ctx.check()?;
    if primitives.len() > mklight_gpu::consts::MKL_STRAY_MAX_PRIMITIVES as usize {
        return Err(unsupported("scene exceeds the resident primitive capacity"));
    }
    let pack = |i: usize| {
        let p = &primitives[i];
        let bounds = bounds::local(p)?;
        let (kind, c, k, ri, ro, mask) = match &p.shape {
            Shape::Cylinder(c) => (2, 0., 0., 0., c.radius, None),
            Shape::Sag(s) => {
                let (kind, c, k) = s
                    .conic()
                    .map_or((3, s.shape().curvature(), s.shape().conic()), |(c, k)| {
                        (if c == 0. { 0 } else { 1 }, c, k)
                    });
                (kind, c, k, s.r_inner(), s.r_outer(), s.mask())
            }
        };
        let (mask, aperture) = match mask {
            None => (0, [0., 0.]),
            Some(Mask::InsideRect { half_x, half_y }) => (3, [*half_x, *half_y]),
            Some(Mask::Inside(a)) => aperture_record(a, false),
            Some(Mask::Outside(a)) => aperture_record(a, true),
        };
        Ok(ops::Primitive {
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
        })
    };
    fn build(prims: &[Aabb], ids: &mut [usize], nodes: &mut Vec<ops::Node>) {
        let bounds = ids.iter().fold(Aabb::EMPTY, |a, &i| a.union(&prims[i]));
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
                let ca = prims[a].centroid();
                let cb = prims[b].centroid();
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
    let boxes = if primitives.len() < 512 {
        primitives
            .iter()
            .map(|p| bounds::local(p).map(|b| b.transformed(&p.frame)))
            .collect::<MkResult<Vec<_>>>()?
    } else {
        Vec::new()
    };
    let mut nodes = Vec::with_capacity(primitives.len() * 2);
    if !primitives.is_empty() && primitives.len() < 512 {
        build(
            &boxes,
            &mut (0..primitives.len()).collect::<Vec<_>>(),
            &mut nodes,
        );
    }
    ctx.check()?;
    let geometry = ops::Geometry::from_fn(b, primitives.len(), &nodes, ctx, pack)?;
    // Only iteratively intersected sags need their surface records.
    if !primitives
        .iter()
        .any(|p| matches!(&p.shape, Shape::Sag(s) if s.conic().is_none()))
    {
        return Ok(geometry);
    }
    // Reuse the sequential tracer's authoritative surface packing, without
    // duplicating any surface basis.
    use mklight_core::model::{MaterialRef, OpticalSystem, SemiDiameter, Surface};
    let mut model = OpticalSystem::new_empty();
    model.surfaces = vec![Surface::new(
        0,
        SurfaceShape::Plane,
        f64::INFINITY,
        MaterialRef::Air,
    )];
    for (i, p) in primitives.iter().enumerate() {
        let shape = match &p.shape {
            Shape::Sag(s) => s.shape().clone(),
            Shape::Cylinder(_) => SurfaceShape::Plane,
        };
        let mut s = Surface::new(i as u64 + 1, shape, 1., MaterialRef::Air);
        s.semi_diameter = SemiDiameter::Fixed {
            value: match &p.shape {
                Shape::Sag(s) => s.r_outer(),
                Shape::Cylinder(c) => c.radius,
            },
        };
        model.surfaces.push(s);
    }
    model.surfaces.push(Surface::new(
        primitives.len() as u64 + 1,
        SurfaceShape::Plane,
        0.,
        MaterialRef::Air,
    ));
    let snapshot = Snapshot::new(model)?;
    let resolved = mklight_core::ResolvedSystem::new(
        &snapshot,
        std::sync::Arc::new(mklight_core::materials::MaterialLibrary::with_builtin()),
    )?;
    let packed = mklight_gpu::PackedSystem::pack_all(&resolved)?;
    geometry.with_shapes(b, ctx, |i| {
        let mut out = ops::SurfaceData::default();
        let s = packed.surfaces[i + 1];
        let data =
            &packed.coefficients[s.coefficient_offset as usize..][..s.coefficient_count as usize];
        if data.len() > out.coefficients.len() {
            return Err(unsupported("sag coefficient capacity"));
        }
        out.surface = s;
        out.surface.coefficient_offset = 0;
        out.coefficients[..data.len()].copy_from_slice(data);
        use mklight_gpu::consts::*;
        if s.shape <= MKL_SHAPE_EVEN_ASPHERE && s.flags & MKL_SURF_HAS_SAG_OVERLAY == 0 {
            out.surface.flags |= MKL_SURF_HAS_SAG_OVERLAY;
            out.surface.coefficient_count = MKL_OVERLAY_COUNT;
            out.coefficients[MKL_OVERLAY_INV_RADIUS as usize] = Df32::from_f32(1.);
        }
        Ok(out)
    })
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
/// The device record of `source` and the cumulative distribution of the
/// wavelength `weights`.
fn source_record(source: &SourceSpec, weights: &[f64]) -> MkResult<(ops::Source, Vec<Df32>)> {
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
    Ok((source, cdf))
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
    let (source, cdf) = source_record(source, weights)?;
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

pub mod hybrid;
mod transport;
pub use hybrid::Design;
pub(crate) use transport::run_design_with;
pub use transport::{TransportOptions, TransportScene, run, run_design};

/// Engine-facing analysis adapter, including coverage/aperture validation,
/// obstruction reference, source measures, map rows and complete RunInfo.
fn analyze_impl(
    b: &dyn ComputeBackend,
    rs: std::sync::Arc<mklight_core::ResolvedSystem>,
    settings: &crate::StrayAnalysisSettings,
    options: TransportOptions,
    design: Design,
    all_signatures: bool,
    ctx: &JobContext,
) -> MkResult<(crate::StrayAnalysisResult, Vec<crate::tally::SignatureFlux>)> {
    settings.validate()?;
    let timer = mklight_core::job::RunTimer::start();
    let snapshot = rs.snapshot.clone();
    // The obstruction checks of the system fields run on the CPU while the
    // device transports.
    let sys =
        crate::StraySystem::build_scene(rs, &settings.system_options(), &ctx.subrange(0., 0.1))?;
    let (mut result, signatures) = crate::analysis::analyze_system_impl(
        &sys,
        settings,
        all_signatures,
        Some((b, options, design)),
        true,
        &ctx.subrange(0.1, 0.99),
    )?;
    result.run_info = timer.finish(
        &snapshot,
        "stray_light",
        &result.run_info.settings,
        &result.run_info.backend,
        result.run_info.precision,
        Some(settings.seed),
        result.warnings.clone(),
    );
    ctx.check()?;
    ctx.report(1., "stray light: done");
    Ok((result, signatures))
}
/// Whether a device analysis ran as the CPU f64 reference by the
/// transport's own division: the device took no source of this run
/// (`hybrid::default_share`), which is not a fallback from the device.
pub fn reference_by_division(run: &mklight_core::RunInfo) -> bool {
    run.precision == mklight_core::job::Precision::F64Reference
        && run.settings["resident"]["settings"]["fallback_count"] == 0
}
pub fn analyze_system(
    b: &dyn ComputeBackend,
    sys: &crate::StraySystem,
    settings: &crate::StrayAnalysisSettings,
    options: TransportOptions,
    ctx: &JobContext,
) -> MkResult<crate::StrayAnalysisResult> {
    crate::analysis::analyze_system_impl(
        sys,
        settings,
        false,
        Some((b, options, Design::selected()?)),
        false,
        ctx,
    )
    .map(|r| r.0)
}
pub fn analyze(
    b: &dyn ComputeBackend,
    rs: std::sync::Arc<mklight_core::ResolvedSystem>,
    settings: &crate::StrayAnalysisSettings,
    options: TransportOptions,
    ctx: &JobContext,
) -> MkResult<crate::StrayAnalysisResult> {
    analyze_impl(b, rs, settings, options, Design::selected()?, false, ctx).map(|r| r.0)
}

pub fn analyze_with_signatures(
    b: &dyn ComputeBackend,
    rs: std::sync::Arc<mklight_core::ResolvedSystem>,
    settings: &crate::StrayAnalysisSettings,
    options: TransportOptions,
    ctx: &JobContext,
) -> MkResult<(crate::StrayAnalysisResult, Vec<crate::tally::SignatureFlux>)> {
    analyze_impl(b, rs, settings, options, Design::selected()?, true, ctx)
}

/// [`analyze`] with an explicit division of the transport instead of the
/// selected one.
pub fn analyze_design(
    b: &dyn ComputeBackend,
    rs: std::sync::Arc<mklight_core::ResolvedSystem>,
    settings: &crate::StrayAnalysisSettings,
    options: TransportOptions,
    design: Design,
    ctx: &JobContext,
) -> MkResult<crate::StrayAnalysisResult> {
    analyze_impl(b, rs, settings, options, design, false, ctx).map(|r| r.0)
}
