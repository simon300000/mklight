//! `system_transmission` rows on the device. Each (field, wavelength) row is
//! a pupil average of per-ray throughputs. The CPU aims every ray (the
//! reference's f64 packet aiming, all cores) and writes the launch rays
//! straight into shared ray planes; the GPU traces them and evaluates the
//! radiometry and the per-ray summary channels; the host sums each row in
//! f64 (reading the shared buffer in place) and completes the rays the
//! device could not decide with the reference. Rows run in blocks: the GPU
//! traces block k while the CPU aims block k+1.
use super::jobs::{reference_status, reference_value};
use super::*;
use crate::{ComputeBackend, Df32, PackedSystem, aiming};
use mklight_core::{JobContext, math::KahanSum};
use mklight_trace::{
    FieldPoint, PolarizationState, PreparedField, RadiometryModel, SequentialTracer, TraceOptions,
};
use rayon::prelude::*;

/// Prepared fields, radiometry tables and system upload of a request.
pub(super) struct Prep {
    prepared: PreparedRadiometry,
    handle: crate::SystemHandle,
    fields: Vec<(usize, PreparedField)>,
}
impl Prep {
    fn field(&self, f: usize) -> &PreparedField {
        &self.fields.iter().find(|p| p.0 == f).unwrap().1
    }
    fn release(self, b: &dyn ComputeBackend) -> MkResult<()> {
        b.release_systems(self.handle)
    }
}

/// A U32 plan import of `shape`, filled in place.
fn import_u32(
    b: &dyn ComputeBackend,
    p: &mut ResidentPlan,
    shape: [usize; 2],
    write: &mut dyn FnMut(&mut [u32]),
) -> MkResult<BufferId> {
    let h = b.fill_buffer(BufferDesc::new(ElementType::U32, shape), &mut |mut f| {
        write(f.as_mut_slice::<u32>()?);
        Ok(())
    })?;
    Ok(p.import(h.view()))
}

/// One row's pupil average inputs.
pub(super) struct RowSums {
    /// Sum of the throughputs (DOE efficiency included).
    pub sum: f64,
    pub usable: usize,
    pub ok: usize,
    pub failed: usize,
    pub warnings: Vec<String>,
    /// Execution evidence of the row's plan or reference work.
    pub run: serde_json::Value,
}

/// The rows of one transmission request, field-major.
pub(super) struct TransmissionRows<'a> {
    pub t: &'a SequentialTracer,
    pub fields: &'a [FieldPoint],
    pub wavelengths: &'a [usize],
    pub samples: &'a [[f64; 2]],
    pub state: &'a PolarizationState,
    pub model: &'a RadiometryModel,
    pub clip: bool,
    /// Summary channel of the model's throughput (0 scalar, 1 polarised).
    pub channel: usize,
}

fn aim_failed(s: u32) -> bool {
    (s & crate::consts::MKL_RAY_CODE_MASK) == crate::consts::MKL_AIM_FAILED
}

impl TransmissionRows<'_> {
    pub fn count(&self) -> usize {
        self.fields.len() * self.wavelengths.len()
    }
    fn scalar(&self) -> bool {
        matches!(self.model, RadiometryModel::Unpolarized)
    }
    fn row(&self, r: usize) -> (usize, usize) {
        (
            r / self.wavelengths.len(),
            self.wavelengths[r % self.wavelengths.len()],
        )
    }

    /// Prepared fields of `rows`, radiometry tables and the system upload
    /// (released by the caller with [`Prep::release`]).
    fn prep(
        &self,
        b: &dyn ComputeBackend,
        rows: std::ops::Range<usize>,
        ctx: &JobContext,
    ) -> MkResult<Prep> {
        ctx.check()?;
        let t = self.t;
        let mut field_ids: Vec<usize> = rows.map(|r| self.row(r).0).collect();
        field_ids.dedup();
        let fields: Vec<(usize, PreparedField)> = field_ids
            .par_iter()
            .map(|&f| t.prepare_field(self.fields[f]).map(|p| (f, p)))
            .collect::<MkResult<_>>()?;
        let prepared = PreparedRadiometry::new(b, t.system(), &ctx.subrange(0.0, 0.05))?;
        let packed = PackedSystem::pack_all(t.system())?;
        let cache_key = SystemCacheKey {
            snapshot: t.system().snapshot.hash().to_hex(),
            library: prepared.identity.clone(),
            configuration: "primary".into(),
            spectrum: "all".into(),
            layout: crate::consts::MKL_LAYOUT_VERSION,
            kernel_build: b.precision().kernel_build.clone(),
            operation_settings: "radiometry geometry v1".into(),
        };
        let handle = b.upload_systems_cached(&cache_key, std::slice::from_ref(&packed))?;
        Ok(Prep {
            prepared,
            handle,
            fields,
        })
    }

    /// The launch rays of `rows` aimed on the CPU (the reference's packet
    /// aiming, f64, all cores), written straight into shared ray planes.
    fn host_rays(
        &self,
        b: &dyn ComputeBackend,
        prep: &Prep,
        p: &mut ResidentPlan,
        rows: std::ops::Range<usize>,
        ctx: &JobContext,
    ) -> MkResult<aiming::ResidentRays> {
        let np = self.samples.len();
        let n = rows.len() * np;
        let first = rows.start;
        // Packets of 64 rays, aimed together.
        let generated: Vec<_> = (0..n.div_ceil(64))
            .into_par_iter()
            .flat_map_iter(|c| {
                let range = c * 64..((c + 1) * 64).min(n);
                let requests: Vec<mklight_trace::RayRequest> = range
                    .map(|i| {
                        let (f, wl) = self.row(first + i / np);
                        (prep.field(f), wl, self.samples[i % np])
                    })
                    .collect();
                self.t.generate_many(&requests)
            })
            .collect();
        ctx.check()?;
        let floats = {
            let h = b.fill_buffer(BufferDesc::new(ElementType::F32, [15, n]), &mut |mut f| {
                let planes = f.as_mut_slice::<f32>()?;
                for (i, g) in generated.iter().enumerate() {
                    let ray = g.as_ref().map(|g| g.ray).unwrap_or(mklight_trace::Ray::new(
                        mklight_core::Vec3::ZERO,
                        mklight_core::Vec3::Z,
                    ));
                    let split = |v: f64| {
                        let hi = v as f32;
                        (hi, (v - hi as f64) as f32)
                    };
                    let pos = [ray.pos.x, ray.pos.y, ray.pos.z].map(split);
                    let dir = [ray.dir.x, ray.dir.y, ray.dir.z].map(split);
                    let values = [
                        pos[0].0, pos[1].0, pos[2].0, dir[0].0, dir[1].0, dir[2].0, 0.0, 0.0, 1.0,
                        pos[0].1, pos[1].1, pos[2].1, dir[0].1, dir[1].1, dir[2].1,
                    ];
                    for (k, v) in values.into_iter().enumerate() {
                        planes[k * n + i] = v;
                    }
                }
                Ok(())
            })?;
            p.import(h.view())
        };
        let indices = import_u32(b, p, [2, n], &mut |w| {
            for i in 0..n {
                w[i] = 0;
                w[n + i] = self.row(first + i / np).1 as u32;
            }
        })?;
        let logical_ids = import_u32(b, p, [2, n], &mut |w| {
            for i in 0..n {
                w[i] = i as u32;
                w[n + i] = 0;
            }
        })?;
        let stop = (self.t.system().stop as u32) << crate::consts::MKL_RAY_SURFACE_SHIFT;
        let report = import_u32(b, p, [4, n], &mut |w| {
            for (i, g) in generated.iter().enumerate() {
                let (status, iterations) = match g {
                    Ok(g) => (0, g.report.iterations),
                    Err(r) => (crate::consts::MKL_AIM_FAILED | stop, r.iterations),
                };
                w[i] = status;
                w[n + i] = iterations;
                w[2 * n + i] = 0;
                w[3 * n + i] = 0;
            }
        })?;
        Ok(aiming::ResidentRays {
            floats,
            indices,
            count: n,
            logical_ids,
            report,
        })
    }

    /// One plan for `rows`: host-aimed rays, trace, radiometry and the
    /// per-ray summary channels; submitted, not waited for.
    fn submit_rows(
        &self,
        b: &dyn ComputeBackend,
        prep: &Prep,
        rows: std::ops::Range<usize>,
        ctx: &JobContext,
    ) -> MkResult<crate::pending::Pending<PlanResult>> {
        let t = self.t;
        let np = self.samples.len();
        let nr = rows.len();
        let n = nr * np;
        if n > u32::MAX as usize {
            return Err(mklight_core::MkError::InvalidModel(
                "transmission batch too large".into(),
            ));
        }
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        p.identity.library_hash = prep.prepared.identity.clone();
        p.settings = serde_json::json!({"operation": "radiometry_transmission_rows",
            "version": crate::consts::MKL_RAD_VERSION, "rows": nr, "point_count": np,
            "state": self.state, "clip": self.clip});
        let rays = self.host_rays(b, prep, &mut p, rows.clone(), ctx)?;
        let steps = t.system().image_index();
        let program = b.fill_buffer(BufferDesc::new(ElementType::U32, [steps]), &mut |mut f| {
            for (i, v) in f.as_mut_slice::<u32>()?.iter_mut().enumerate() {
                *v = (i + 1) as u32;
            }
            Ok(())
        })?;
        let program = p.import(program.view());
        let hits = aiming::hits(&mut p, n, steps + 1);
        p.operations
            .push(ResidentOp::Aiming(aiming::Request::TraceResident(
                aiming::TraceResident {
                    systems: SystemSet::Uploaded(prep.handle),
                    input: rays,
                    steps: p.view(program)?,
                    start_surface: 0,
                    apply_apertures: self.clip,
                    radiometry: 0,
                    newton: Default::default(),
                    output: hits.clone(),
                },
            )));
        let values = prep.prepared.append(&mut p, &hits, self.state)?;
        if self.scalar() {
            for op in &mut p.operations {
                if let ResidentOp::Radiometry(Request::Polarization(r)) = op {
                    r.scalar = true;
                }
            }
            p.settings["scalar"] = serde_json::json!(true);
        }
        let channels = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [10, n]));
        p.operations.push(ResidentOp::Radiometry(Request::Channels {
            values: p.view(values)?,
            trace_status: p.view(hits.status)?,
            output: channels,
        }));
        p.outputs = vec![values, hits.status, channels];
        b.submit_plan(p, ctx)
    }

    /// Row sums of a completed plan for `rows`, in f64 in pupil order, the
    /// rays the device could not decide completed by the reference.
    fn sum_rows(
        &self,
        b: &dyn ComputeBackend,
        prep: &Prep,
        rows: std::ops::Range<usize>,
        result: PlanResult,
        evidence: serde_json::Value,
        ctx: &JobContext,
    ) -> MkResult<Vec<RowSums>> {
        ctx.check()?;
        let t = self.t;
        let np = self.samples.len();
        let n = rows.len() * np;
        let clock = std::time::Instant::now();
        let values = b.read_view(result.outputs[0].view())?;
        let values: &[PolarizationValue] = bytemuck::cast_slice(values.words());
        let status = b.read_view(result.outputs[1].view())?;
        let status = &status.as_slice::<u32>()?[..n];
        let channels = b.read_view(result.outputs[2].view())?;
        let channels = channels.as_slice::<Df32>()?;
        let opts = TraceOptions {
            clip: self.clip,
            ..t.options()
        };
        // Reference completion (as `pupil_mode`) of the rays the device
        // could not decide, all in parallel.
        let reference: Vec<usize> = (0..n)
            .filter(|&i| {
                values[i].status == crate::consts::MKL_RAD_REFERENCE || aim_failed(status[i])
            })
            .collect();
        let completed: Vec<[Df32; 10]> = reference
            .par_iter()
            .map(|&i| {
                let (f, wl) = self.row(rows.start + i / np);
                let p = self.samples[i % np];
                let ray = t.trace_polarized(prep.field(f), wl, p[0], p[1], &opts);
                let value = reference_value(&ray, self.state);
                super::summary_channels(value, reference_status(&ray.trace.status))
            })
            .collect();
        ctx.check()?;
        let fallback = reference.len();
        let mut next = reference.iter().zip(&completed).peekable();
        let mut out = Vec::with_capacity(rows.len());
        for (k, r) in rows.clone().enumerate() {
            let (_, wl) = self.row(r);
            let mut sums = [KahanSum::default(); 10];
            for j in 0..np {
                let i = k * np + j;
                match next.peek() {
                    Some(&(&ri, ch)) if ri == i => {
                        for (s, v) in sums.iter_mut().zip(ch) {
                            s.add(v.to_f64());
                        }
                        next.next();
                    }
                    _ => {
                        for (c, s) in sums.iter_mut().enumerate() {
                            s.add(channels[c * n + i].to_f64());
                        }
                    }
                }
            }
            out.push(RowSums {
                sum: sums[self.channel].value()
                    * mklight_trace::radiometry::diffraction_product(t.system(), wl),
                usable: sums[5].value() as usize,
                ok: sums[6].value() as usize,
                failed: sums[9].value() as usize,
                warnings: Vec::new(),
                run: serde_json::Value::Null,
            });
        }
        let host_ms = clock.elapsed().as_secs_f64() * 1e3;
        if let Some(first) = out.first_mut() {
            let mut run = result.run_info;
            run.settings["rows"] = serde_json::json!({"rows": rows.len(), "rays": n,
                "host_sum_ms": host_ms, "reference_rays": fallback, "stages": evidence,
                "stats": result.stats});
            if fallback > 0 {
                first.warnings.push(format!(
                    "P7 reference evaluated {fallback} pupil rays (coating/retarder conditioning or unresolved aiming)"
                ));
            }
            first.run = serde_json::json!({"backend": run.backend, "settings": run.settings});
        }
        Ok(out)
    }

    /// Every row, in blocks: the GPU traces block k while the CPU aims block
    /// k+1.
    pub fn run(&self, b: &dyn ComputeBackend, ctx: &JobContext) -> MkResult<Vec<RowSums>> {
        let total = self.count();
        let prep = self.prep(b, 0..total, ctx)?;
        // About four blocks, at least 4096 rays each.
        let per = (total.div_ceil(4))
            .max(4096usize.div_ceil(self.samples.len()))
            .max(1);
        let blocks: Vec<_> = (0..total)
            .step_by(per)
            .map(|s| s..(s + per).min(total))
            .collect();
        let run = || -> MkResult<Vec<RowSums>> {
            let mut out = Vec::with_capacity(total);
            type Submitted = crate::pending::Pending<PlanResult>;
            let mut pending: Option<(std::ops::Range<usize>, Submitted, f64)> = None;
            for block in blocks
                .iter()
                .cloned()
                .map(Some)
                .chain(std::iter::once(None))
            {
                let next = match block {
                    Some(rows) => {
                        let clock = std::time::Instant::now();
                        let submitted = self.submit_rows(b, &prep, rows.clone(), ctx)?;
                        Some((rows, submitted, clock.elapsed().as_secs_f64() * 1e3))
                    }
                    None => None,
                };
                if let Some((rows, submitted, aim_ms)) = pending.take() {
                    let clock = std::time::Instant::now();
                    let result = submitted.wait()?;
                    let wait_ms = clock.elapsed().as_secs_f64() * 1e3;
                    out.extend(self.sum_rows(
                        b,
                        &prep,
                        rows,
                        result,
                        serde_json::json!({"host_aim_ms": aim_ms, "device_wait_ms": wait_ms, "blocks": blocks.len()}),
                        ctx,
                    )?);
                }
                pending = next;
            }
            Ok(out)
        };
        let out = run();
        prep.release(b)?;
        out
    }
}
