use super::*;
use crate::{
    footprint_overlay::{FootprintOverlay, OverlaySet, convex_hull},
    illumination::FootprintSettings,
    util::{default_fields, pupil_points},
};
use mklight_core::{SurfaceShape, Vec3};
use mklight_gpu::{
    ReduceMode,
    collectives::{self, SegmentKind, SegmentReduce},
    geometry::{self, Records},
};
use mklight_trace::{SemiDiameterSettings, SemiDiameterSource, SemiDiameters};
impl Geometry<'_> {
    pub fn global_records(
        &self,
        p: &mut ResidentPlan,
        hits: &ResidentHits,
        slots: usize,
        origin: [f64; 3],
        terminal: bool,
    ) -> MkResult<(BufferId, BufferId, BufferId, BufferId, BufferId)> {
        let frames = self.import::<Df32>(p, [slots, 12], |out| {
            for (i, s) in self.tracer.system().surfaces.iter().take(slots).enumerate() {
                for k in 0..9 {
                    out[12 * i + k] = Df32::from_f64(s.frame.rot.m[k / 3][k % 3]);
                }
                for (k, v) in [s.frame.origin.x, s.frame.origin.y, s.frame.origin.z]
                    .into_iter()
                    .enumerate()
                {
                    out[12 * i + 9 + k] = Df32::from_f64(v);
                }
            }
            Ok(())
        })?;
        let n = hits.rays.count;
        let launch = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n, 3]));
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [slots, n, 3]));
        let radii = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, slots * n]));
        let mask = p.scratch(BufferDesc::new(ElementType::U32, [slots * n]));
        let failures = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n]));
        p.operations
            .push(ResidentOp::Geometry(geometry::Request::Records(Records {
                terminal,
                launch,
                hits: hits.clone(),
                slots,
                frames: p.view(frames)?,
                origin: origin.map(Df32::from_f64),
                output,
                radii,
                mask,
                failures,
            })));
        Ok((output, radii, mask, failures, launch))
    }
}
pub fn semi_diameters_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &SemiDiameterSettings,
    ctx: &JobContext,
) -> MkResult<(SemiDiameters, RunInfo)> {
    Geometry::new(b, t, ctx)?.semi_diameters(s, ctx)
}
impl Geometry<'_> {
    pub fn semi_diameters(
        &self,
        s: &SemiDiameterSettings,
        ctx: &JobContext,
    ) -> MkResult<(SemiDiameters, RunInfo)> {
        let (geo, b, t) = (self, self.backend, self.tracer);
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        p.settings = serde_json::to_value(s).unwrap();
        let mut pupil = vec![[0.0; 2]];
        pupil.extend(mklight_trace::pupil::ring(s.azimuths.max(4), 1.0));
        let fields: Vec<_> = t.fields().iter().map(|f| f.point).collect();
        let wls: Vec<_> = (0..t.num_wavelengths()).collect();
        let slots = t.system().num_surfaces();
        let g = geo.append_grid(
            &mut p,
            &fields,
            &wls,
            &pupil,
            t.aiming(),
            slots - 1,
            s.clip,
            true,
            ctx,
        )?;
        let (_, radii, mask, fail, _) =
            geo.global_records(&mut p, &g.hits, slots, [0.0; 3], false)?;
        let n = g.hits.rays.count;
        let offsets = geo.import::<u32>(&mut p, [slots + 1], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i * n) as u32;
            }
            Ok(())
        })?;
        let summary = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, slots, 4]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [slots]));
        p.operations.push(ResidentOp::Collectives(
            collectives::Request::SegmentReduce(SegmentReduce {
                values: p.view(radii)?,
                offsets: p.view(offsets)?,
                mask: Some(p.view(mask)?),
                mask_value: 1,
                weights: None,
                max_segment_len: n,
                reference: None,
                mode: ReduceMode::DoubleFloat,
                kind: SegmentKind::Summary,
                output: summary,
                status,
            }),
        ));
        let offsets = geo.import::<u32>(&mut p, [2], |out| {
            out[0] = 0;
            out[1] = n as u32;
            Ok(())
        })?;
        let sum = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 1]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [1]));
        let mut v = p.view(fail)?;
        v.desc.shape = vec![1, n];
        p.operations.push(ResidentOp::Collectives(
            collectives::Request::SegmentReduce(SegmentReduce {
                values: v,
                offsets: p.view(offsets)?,
                mask: None,
                mask_value: 0,
                weights: None,
                max_segment_len: n,
                reference: None,
                mode: ReduceMode::DoubleFloat,
                kind: SegmentKind::Sum,
                output: sum,
                status,
            }),
        ));
        p.outputs = vec![summary, sum];
        let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
        let v = b.read_view(r.outputs[0].view())?;
        let v = v.as_slice::<Df32>()?;
        let traced: Vec<_> = (0..slots).map(|j| v[j * 4 + 3].to_f64().max(0.0)).collect();
        let mut values = Vec::new();
        let mut sources = Vec::new();
        for (j, surf) in t.system().surfaces.iter().enumerate() {
            let fixed = surf
                .aperture
                .as_ref()
                .map(|a| a.bounding_radius())
                .or(surf.fixed_semi_diameter);
            let (v, source) = if let Some(v) = fixed {
                (v, SemiDiameterSource::Fixed)
            } else if matches!(surf.shape, SurfaceShape::CoordinateBreak) {
                (0.0, SemiDiameterSource::None)
            } else if j == t.system().stop {
                (t.pupil().r_stop, SemiDiameterSource::Stop)
            } else if j == 0 && t.system().object_at_infinity {
                (0.0, SemiDiameterSource::None)
            } else {
                (traced[j], SemiDiameterSource::Traced)
            };
            values.push(v);
            sources.push(source);
        }
        let failed = b.read_view(r.outputs[1].view())?.as_slice::<Df32>()?[0].to_f64() as usize;
        ctx.report(1.0, "semi-diameters");
        Ok((
            SemiDiameters {
                values,
                traced,
                sources,
                failed_rays: failed,
            },
            stamp(r.run_info, "semi_diameters", s, t.warnings().to_vec()),
        ))
    }
}

pub fn footprint_overlay_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &FootprintSettings,
    ctx: &JobContext,
) -> MkResult<FootprintOverlay> {
    crate::spot::validate_sampling(s.pattern, s.rings)?;
    let aug = default_fields(t, &s.fields)?;
    let t = aug.as_ref();
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let pts: Vec<_> = fields.iter().map(|&f| t.field(f).point).collect();
    let slots = t.system().num_surfaces();
    let geo = Geometry::new(b, t, ctx)?;
    // Bound all-surface scratch independently of field/wavelength multiplicity.
    let tile = ((1usize << 20) / (slots * pts.len() * wls.len())).max(1);
    let count = pupil.len().div_ceil(tile);
    let mut clouds = vec![Vec::new(); fields.len() * wls.len() * slots];
    let mut evidence: Option<RunInfo> = None;
    for (part, pupil) in pupil.chunks(tile).enumerate() {
        let sub = ctx.subrange(
            0.9 * part as f64 / count as f64,
            0.9 * (part + 1) as f64 / count as f64,
        );
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        p.settings = serde_json::to_value(s).unwrap();
        p.seed = matches!(
            s.pattern,
            crate::util::PupilPattern::Random | crate::util::PupilPattern::Dithered
        )
        .then_some(s.seed);
        let g = geo.append_grid(
            &mut p,
            &pts,
            &wls,
            pupil,
            t.aiming(),
            slots - 1,
            true,
            true,
            &sub,
        )?;
        let (points, _, mask, _, _) =
            geo.global_records(&mut p, &g.hits, slots, [0.0; 3], false)?;
        p.outputs = vec![points, mask];
        let r = geo.run(p, &sub)?;
        let q = b.read_view(r.outputs[0].view())?;
        let q = q.as_slice::<Df32>()?;
        let m = b.read_view(r.outputs[1].view())?;
        let m = m.as_slice::<u32>()?;
        let n = g.hits.rays.count;
        for group in 0..fields.len() * wls.len() {
            for surface in 1..slots {
                let start = surface * n + group * pupil.len();
                clouds[group * slots + surface].extend(
                    (start..start + pupil.len())
                        .filter(|&i| m[i] != 0)
                        .map(|i| {
                            Vec3::new(
                                q[3 * i].to_f64(),
                                q[3 * i + 1].to_f64(),
                                q[3 * i + 2].to_f64(),
                            )
                        }),
                );
            }
        }
        if let Some(run) = &mut evidence {
            run.duration_ms += r.run_info.duration_ms;
            run.settings["extra_plans"]
                .as_array_mut()
                .unwrap()
                .push(r.run_info.settings);
        } else {
            let mut run = r.run_info;
            run.settings["extra_plans"] = serde_json::json!([]);
            evidence = Some(run);
        }
    }
    let mut sets = Vec::with_capacity(clouds.len());
    // Exactly sized: the clouds are released one by one as they are copied.
    let mut points = Vec::with_capacity(clouds.iter().map(Vec::len).sum());
    let mut outlines = Vec::new();
    for (fi, &f) in fields.iter().enumerate() {
        for (wi, &w) in wls.iter().enumerate() {
            for surface in 1..slots {
                ctx.check()?;
                let hit = std::mem::take(&mut clouds[(fi * wls.len() + wi) * slots + surface]);
                // Hull topology is small host control; order in local xy, while retaining
                // the device-computed global points including the actual sag.
                let frame = t.system().surfaces[surface].frame;
                let local: Vec<_> = hit.iter().map(|&p| frame.to_local_point(p)).collect();
                let hull = convex_hull(&local);
                let global = |p: Vec3| {
                    let p = frame.to_global_point(p);
                    [p.x, p.y, p.z]
                };
                sets.push(OverlaySet {
                    surface,
                    field: f,
                    wavelength: w,
                    points: [points.len(), hit.len()],
                    outline: [outlines.len(), hull.len()],
                });
                points.extend(hit.iter().map(|p| [p.x, p.y, p.z]));
                outlines.extend(hull.into_iter().map(global));
            }
        }
    }
    ctx.report(1.0, "global footprints");
    Ok(FootprintOverlay {
        sets,
        points,
        outlines,
        run_info: stamp(evidence.unwrap(), "footprint", s, t.warnings().to_vec()),
    })
}

/// Final image hits of the `fields × wavelengths × pupil` grid, widened from
/// the shared device storage straight into the public buffer layout.
#[derive(Clone, Debug)]
pub struct GridResult {
    pub fields: Vec<usize>,
    pub wavelengths: Vec<usize>,
    pub pupil: Vec<[f64; 2]>,
    /// Hit `[x, y]` per ray (image local frame), NaN where it did not arrive.
    pub points: Vec<f64>,
    /// Unit direction `[x, y, z]` per ray, NaN where it did not arrive.
    pub directions: Vec<f64>,
    pub run_info: RunInfo,
}
impl GridResult {
    /// Number of rays.
    pub fn len(&self) -> usize {
        self.points.len() / 2
    }
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
    /// Ray `i`, `None` when it did not arrive.
    pub fn hit(&self, i: usize) -> Option<crate::spot::ImageHit> {
        let d = &self.directions[3 * i..3 * i + 3];
        (!self.points[2 * i].is_nan()).then(|| crate::spot::ImageHit {
            point: [self.points[2 * i], self.points[2 * i + 1]],
            dir: Vec3::new(d[0], d[1], d[2]),
        })
    }
}
pub fn ray_grid_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &crate::energy::RaySampling,
    ctx: &JobContext,
) -> MkResult<GridResult> {
    crate::spot::validate_sampling(s.pattern, s.rings)?;
    let aug = default_fields(t, &s.fields)?;
    let t = aug.as_ref();
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let pts: Vec<_> = fields.iter().map(|&f| t.field(f).point).collect();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.seed = matches!(
        s.pattern,
        crate::util::PupilPattern::Random | crate::util::PupilPattern::Dithered
    )
    .then_some(s.seed);
    p.settings = serde_json::to_value(s).unwrap();
    let g = geo.append_grid(
        &mut p,
        &pts,
        &wls,
        &pupil,
        t.aiming(),
        t.system().image_index(),
        true,
        false,
        ctx,
    )?;
    p.outputs = vec![g.hits.rays.floats, g.hits.status];
    let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
    let f = b.read_view(r.outputs[0].view())?;
    let f = f.as_slice::<f32>()?;
    let status = b.read_view(r.outputs[1].view())?;
    let status = status.as_slice::<u32>()?;
    let n = g.hits.rays.count;
    let mut points = vec![f64::NAN; 2 * n];
    let mut directions = vec![f64::NAN; 3 * n];
    for (i, &code) in status[..n].iter().enumerate() {
        if i % 4096 == 0 {
            ctx.check()?;
        }
        if code == 0 {
            points[2 * i..2 * i + 2].copy_from_slice(&[value(f, n, i, 0), value(f, n, i, 1)]);
            directions[3 * i..3 * i + 3].copy_from_slice(&[
                value(f, n, i, 3),
                value(f, n, i, 4),
                value(f, n, i, 5),
            ]);
        }
    }
    ctx.report(1.0, "ray grid");
    Ok(GridResult {
        fields,
        wavelengths: wls,
        pupil,
        points,
        directions,
        run_info: stamp(r.run_info, "ray_grid", s, t.warnings().to_vec()),
    })
}
pub fn footprint_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &FootprintSettings,
    ctx: &JobContext,
) -> MkResult<crate::illumination::FootprintResult> {
    use crate::illumination::{FootprintRays, FootprintResult};
    use mklight_gpu::consts::*;
    crate::spot::validate_sampling(s.pattern, s.rings)?;
    let aug = default_fields(t, &s.fields)?;
    let t = aug.as_ref();
    if s.surface == 0 || s.surface > t.system().image_index() {
        return Err(mklight_core::MkError::InvalidModel(format!(
            "footprint surface {} must be in 1…{}",
            s.surface,
            t.system().image_index()
        )));
    }
    let fields = s.fields.resolve(t.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(t.num_wavelengths(), t.system().primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let pts: Vec<_> = fields.iter().map(|&f| t.field(f).point).collect();
    let geo = Geometry::new(b, t, ctx)?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.settings = serde_json::to_value(s).unwrap();
    p.seed = matches!(
        s.pattern,
        crate::util::PupilPattern::Random | crate::util::PupilPattern::Dithered
    )
    .then_some(s.seed);
    let g = geo.append_grid(
        &mut p,
        &pts,
        &wls,
        &pupil,
        t.aiming(),
        s.surface,
        true,
        false,
        ctx,
    )?;
    p.outputs = vec![g.hits.rays.floats, g.hits.status];
    let r = geo.run(p, &ctx.subrange(0.0, 0.7))?;
    let f = b.read_view(r.outputs[0].view())?;
    let f = f.as_slice::<f32>()?;
    let st = b.read_view(r.outputs[1].view())?;
    let st = st.as_slice::<u32>()?;
    let n = g.hits.rays.count;
    let mut extent = [
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut max_r = f64::NAN;
    let mut rays = Vec::new();
    for (fi, &field) in fields.iter().enumerate() {
        for (wi, &wavelength) in wls.iter().enumerate() {
            ctx.check()?;
            let mut fr = FootprintRays {
                field,
                wavelength,
                points: Vec::new(),
                clipped: Vec::new(),
                launched: pupil.len(),
                lost_before: 0,
            };
            let start = (fi * wls.len() + wi) * pupil.len();
            for (i, &code) in st.iter().enumerate().skip(start).take(pupil.len()) {
                let q = [value(f, n, i, 0), value(f, n, i, 1)];
                if code == 0 {
                    fr.points.push(q);
                    extent = [
                        extent[0].min(q[0]),
                        extent[1].max(q[0]),
                        extent[2].min(q[1]),
                        extent[3].max(q[1]),
                    ];
                    max_r = max_r.max(q[0].hypot(q[1]));
                } else if code & MKL_RAY_CODE_MASK == MKL_RAY_VIGNETTED
                    && (code >> MKL_RAY_SURFACE_SHIFT) as usize == s.surface
                {
                    fr.clipped.push(q);
                } else {
                    fr.lost_before += 1;
                }
            }
            rays.push(fr);
        }
    }
    let (sd, sd_run) = geo.semi_diameters(&Default::default(), &ctx.subrange(0.7, 0.95))?;
    let mut run = r.run_info;
    run.duration_ms += sd_run.duration_ms;
    run.settings["sizing"] = sd_run.settings;
    let extent = if extent[0].is_finite() {
        extent.to_vec()
    } else {
        vec![f64::NAN; 4]
    };
    ctx.report(1.0, "footprint");
    Ok(FootprintResult {
        field_values: (0..t.fields().len())
            .map(|i| crate::util::field_value(t, i))
            .collect(),
        surface: s.surface,
        semi_diameter: sd.values[s.surface],
        aperture: t.system().surfaces[s.surface].aperture.clone(),
        rays,
        extent,
        max_radius: max_r,
        run_info: stamp(run, "footprint", s, t.warnings().to_vec()),
    })
}
