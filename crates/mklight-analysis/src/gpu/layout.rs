use super::*;
use crate::layout::{
    self, LayoutProvider, LayoutRay, LayoutScene, LayoutSettings, LayoutVertex, Polyline,
    PolylineKind,
};
use mklight_core::{SurfaceShape, Vec3};
use mklight_gpu::geometry::{self, SurfaceSample, SurfaceSamples, VertexProjection};
use std::cell::RefCell;
struct Provider<'a> {
    backend: &'a dyn ComputeBackend,
    evidence: RefCell<Vec<RunInfo>>,
}
impl LayoutProvider for Provider<'_> {
    fn semi_diameters(
        &self,
        t: &SequentialTracer,
        ctx: &JobContext,
    ) -> MkResult<mklight_trace::SemiDiameters> {
        let (sd, run) =
            super::records::semi_diameters_job(self.backend, t, &Default::default(), ctx)?;
        self.evidence.borrow_mut().push(run);
        Ok(sd)
    }
    fn vertices(
        &self,
        t: &SequentialTracer,
        samples: &[LayoutVertex],
        ctx: &JobContext,
    ) -> MkResult<Vec<[f64; 3]>> {
        let geo = Geometry::new(self.backend, t, ctx)?;
        let slots = t.system().num_surfaces();
        let mut points = Vec::with_capacity(samples.len());
        // Surface meshes can contain millions of repeated topology vertices;
        // tile evaluation, preserving their stable topology indices.
        for start in (0..samples.len()).step_by(1 << 18) {
            let end = (start + (1 << 18)).min(samples.len());
            let src = &samples[start..end];
            let n = src.len();
            let mut p = ResidentPlan::new(t.system().snapshot.clone());
            let inputs = geo.records::<SurfaceSample>(&mut p, n, |i| {
                Ok(SurfaceSample {
                    system: 0,
                    surface: src[i].surface as u32,
                    reserved: [0; 2],
                    xy: src[i].sag_xy.map(Df32::from_f64),
                })
            })?;
            let xy = geo.import::<Df32>(&mut p, [n, 2], |out| {
                for (i, s) in src.iter().enumerate() {
                    out[2 * i] = Df32::from_f64(s.xy[0]);
                    out[2 * i + 1] = Df32::from_f64(s.xy[1]);
                }
                Ok(())
            })?;
            let frames = geo.import::<Df32>(&mut p, [slots, 12], |out| {
                for (i, s) in t.system().surfaces.iter().enumerate() {
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
            // A local display origin removes the common global translation on
            // device before f32 rounding. Public scene coordinates remain f64.
            let origin = t.system().surfaces[slots / 2].frame.origin;
            let origin = [origin.x, origin.y, origin.z];
            let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [n, 3]));
            let status = p.scratch(BufferDesc::new(ElementType::U32, [n]));
            p.operations
                .push(ResidentOp::Geometry(geometry::Request::SurfaceSamples(
                    SurfaceSamples {
                        systems: geo.systems,
                        samples: p.view(inputs)?,
                        projection: Some(VertexProjection {
                            xy: p.view(xy)?,
                            frames: p.view(frames)?,
                            origin: origin.map(Df32::from_f64),
                        }),
                        output,
                        status,
                    },
                )));
            p.outputs = vec![output];
            let r = geo.run(
                p,
                &ctx.subrange(
                    start as f64 / samples.len() as f64,
                    end as f64 / samples.len() as f64,
                ),
            )?;
            let v = self.backend.read_view(r.outputs[0].view())?;
            for p in v.as_slice::<Df32>()?.as_chunks::<3>().0 {
                points.push(std::array::from_fn(|k| p[k].to_f64() + origin[k]));
            }
            self.evidence.borrow_mut().push(r.run_info);
        }
        Ok(points)
    }
    fn rays(
        &self,
        t: &SequentialTracer,
        jobs: &[LayoutRay],
        lead: f64,
        ctx: &JobContext,
    ) -> MkResult<Vec<Polyline>> {
        let geo = Geometry::new(self.backend, t, ctx)?;
        let slots = t.system().num_surfaces();
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        // The selected layout fan is small; each job keeps its own field
        // row, so arbitrary field/wavelength choices retain the original order.
        let steps = geo.import::<u32>(&mut p, [slots - 1], |out| {
            for (i, v) in out.iter_mut().enumerate() {
                *v = (i + 1) as u32;
            }
            Ok(())
        })?;
        let rows = geo.aim_rows(
            &mut p,
            &jobs
                .iter()
                .map(|j| (t.field(j.0).point, j.1))
                .collect::<Vec<_>>(),
            t.aiming(),
        )?;
        let spec = |i: usize| aim::RaySpec {
            row: i as u32,
            pupil: [jobs[i].2, jobs[i].3],
            weight: 1.0,
            id: i as u64,
        };
        let input = geo.aim_rays(&mut p, &rows, jobs.len(), &spec, ctx)?;
        let hits = geo.trace_rays(&mut p, input, steps, true, true)?;
        let (points, _, mask, _, launch) =
            geo.global_records(&mut p, &hits, slots, [0.0; 3], true)?;
        p.outputs = vec![points, mask, launch, hits.status];
        let r = geo.run(p, &ctx.subrange(0.0, 0.95))?;
        let views: Vec<_> = r
            .outputs
            .iter()
            .map(|h| self.backend.read_view(h.view()))
            .collect::<MkResult<_>>()?;
        let q = views[0].as_slice::<Df32>()?;
        let mask = views[1].as_slice::<u32>()?;
        let launch = views[2].as_slice::<Df32>()?;
        let status = views[3].as_slice::<u32>()?;
        let n = jobs.len();
        let mut out = Vec::new();
        for (i, &(field, w, _, _, in_section)) in jobs.iter().enumerate() {
            ctx.check()?;
            let mut points = Vec::new();
            for slot in 0..slots {
                if slot == 0 && t.system().object_at_infinity
                    || matches!(
                        t.system().surfaces[slot].shape,
                        SurfaceShape::CoordinateBreak
                    )
                {
                    continue;
                }
                let j = slot * n + i;
                if mask[j] != 0 {
                    points.push([
                        q[3 * j].to_f64(),
                        q[3 * j + 1].to_f64(),
                        q[3 * j + 2].to_f64(),
                    ]);
                }
            }
            if t.system().object_at_infinity && mask[i] != 0 {
                let d = Vec3::new(
                    launch[3 * i].to_f64(),
                    launch[3 * i + 1].to_f64(),
                    launch[3 * i + 2].to_f64(),
                );
                let first = if let Some(p) = points.first() {
                    Vec3::new(p[0], p[1], p[2]) - d * lead
                } else {
                    Vec3::new(
                        q[3 * i].to_f64(),
                        q[3 * i + 1].to_f64(),
                        q[3 * i + 2].to_f64(),
                    )
                };
                points.insert(0, [first.x, first.y, first.z]);
                if points.len() == 1 {
                    let p = first + d * (2.0 * lead);
                    points.push([p.x, p.y, p.z]);
                }
            }
            out.push(Polyline {
                kind: PolylineKind::Ray,
                points,
                in_section,
                surface: None,
                field: Some(field),
                wavelength: Some(w),
                vignetted: status[i] != 0,
            });
        }
        self.evidence.borrow_mut().push(r.run_info);
        Ok(out)
    }
}
pub fn layout_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &LayoutSettings,
    ctx: &JobContext,
) -> MkResult<LayoutScene> {
    let provider = Provider {
        backend: b,
        evidence: RefCell::new(Vec::new()),
    };
    let mut scene = layout::layout_with(t, s, ctx, &provider)?;
    let mut runs = provider.evidence.into_inner().into_iter();
    let mut run = runs.next().unwrap();
    for r in runs {
        plans::add_run(&mut run, r);
    }
    scene.run_info = stamp(run, "layout", s, t.warnings().to_vec());
    Ok(scene)
}
