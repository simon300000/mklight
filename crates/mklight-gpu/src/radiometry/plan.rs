//! Backend-neutral composition helpers. Tables are immutable leases: reusing a
//! prepared object reuses its material state, and changed material/wavelength
//! values create new leases. No global table identity can alias stale content.
use super::*;
use crate::{
    ComputeBackend, Df32,
    aiming::ResidentHits,
    collectives::{self, SegmentKind, SegmentReduce},
};
use mklight_core::{JobContext, ResolvedSystem};

pub struct PreparedRadiometry {
    pub surfaces: BufferHandle,
    pub layers: BufferHandle,
    pub wavelengths: usize,
    pub identity: String,
}
fn fill<T: bytemuck::Pod>(
    b: &dyn ComputeBackend,
    values: &[T],
    ctx: &JobContext,
) -> MkResult<BufferHandle> {
    b.fill_buffer(record_desc::<T>(values.len()), &mut |mut f| {
        let output: &mut [T] = bytemuck::cast_slice_mut(f.as_mut_slice::<u32>()?);
        for (a, z) in values.chunks(1024).zip(output.chunks_mut(1024)) {
            ctx.check()?;
            z.copy_from_slice(a);
        }
        Ok(())
    })
}
impl PreparedRadiometry {
    pub fn new(b: &dyn ComputeBackend, sys: &ResolvedSystem, ctx: &JobContext) -> MkResult<Self> {
        ctx.check()?;
        let mut layers = vec![];
        let surfaces = surface_specs(sys, &mut layers)?;
        let mut hash = blake3::Hasher::new();
        hash.update(bytemuck::cast_slice(&surfaces));
        hash.update(bytemuck::cast_slice(&layers));
        Ok(Self {
            identity: hash.finalize().to_hex().to_string(),
            surfaces: fill(b, &surfaces, ctx)?,
            layers: fill(b, &layers, ctx)?,
            wavelengths: sys.wavelengths_um.len(),
        })
    }
    /// Primary-path records must have slots 0..image in order, radiometry off,
    /// and belong to this material state. P4 callers may instead construct a
    /// PolarizationRequest with device-generated SurfaceSpec rows.
    pub fn append(
        &self,
        p: &mut ResidentPlan,
        hits: &ResidentHits,
        state: &mklight_trace::PolarizationState,
    ) -> MkResult<BufferId> {
        state.validate()?;
        let st = state.stokes();
        let surfaces = p.import(self.surfaces.view());
        let layers = p.import(self.layers.view());
        let records = hits
            .records
            .ok_or_else(|| bad("polarization needs P1 surface records"))?;
        let low = hits
            .record_low
            .ok_or_else(|| bad("polarization needs P1 residual records"))?;
        let output = p.scratch(record_desc::<PolarizationValue>(hits.rays.count));
        p.operations
            .push(ResidentOp::Radiometry(Request::Polarization(
                PolarizationRequest {
                    surfaces: p.view(surfaces)?,
                    layers: p.view(layers)?,
                    records: p.view(records)?,
                    record_low: p.view(low)?,
                    indices: p.view(hits.rays.indices)?,
                    trace_status: p.view(hits.status)?,
                    wavelengths: self.wavelengths,
                    stokes: [st.s0, st.s1, st.s2, st.s3].map(Df32::from_f64),
                    output,
                    tile: 65536,
                },
            )));
        Ok(output)
    }
}
/// Sum/count/min/max for seven channels: scalar, polarised, exact unpolarised,
/// diattenuation, retardance², usable count, unvignetted count. Failed numerical
/// traces and reference-required records are excluded, physical clipping counts
/// as zero. Reference rows must be resolved before presenting a total.
/// Separate masks/count channels preserve the undefined-retarder rule.
pub fn append_summary(
    b: &dyn ComputeBackend,
    p: &mut ResidentPlan,
    values: BufferId,
    status: BufferId,
    ctx: &JobContext,
) -> MkResult<BufferId> {
    let n = p.desc(values)?.shape[0];
    let channels = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [10, n]));
    p.operations.push(ResidentOp::Radiometry(Request::Channels {
        values: p.view(values)?,
        trace_status: p.view(status)?,
        output: channels,
    }));
    let offsets = b.fill_buffer(BufferDesc::new(ElementType::U32, [2]), &mut |mut f| {
        ctx.check()?;
        f.as_mut_slice::<u32>()?.copy_from_slice(&[0, n as u32]);
        Ok(())
    })?;
    let offsets = p.import(offsets.view());
    let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [10, 1, 4]));
    let flag = p.scratch(BufferDesc::new(ElementType::U32, [1]));
    p.operations.push(ResidentOp::Collectives(
        collectives::Request::SegmentReduce(SegmentReduce {
            values: p.view(channels)?,
            offsets: p.view(offsets)?,
            mask: None,
            mask_value: 0,
            weights: None,
            max_segment_len: n,
            reference: None,
            mode: crate::ReduceMode::DoubleFloat,
            kind: SegmentKind::Summary,
            output,
            status: flag,
        }),
    ));
    Ok(output)
}

#[derive(Clone)]
pub struct CoatingDomain {
    pub coating: mklight_core::materials::coating::ResolvedCoating,
    pub wavelength: f64,
    pub n_in: f64,
    pub n_out: f64,
    pub mirror: bool,
}
/// Persistent material/side/wavelength records; independent of angle sampling.
pub struct PreparedCoatings {
    pub specs: BufferHandle,
    pub layers: BufferHandle,
    domains: Vec<CoatingDomain>,
}
pub struct CoatingResult {
    pub values: Vec<CoatingValue>,
    pub run_info: mklight_core::job::RunInfo,
    pub stats: PlanStats,
}
impl PreparedCoatings {
    pub fn new(
        b: &dyn ComputeBackend,
        domains: &[CoatingDomain],
        ctx: &JobContext,
    ) -> MkResult<Self> {
        let mut layers = vec![];
        let mut specs = vec![];
        for domain in domains {
            ctx.check()?;
            specs.push(CoatingSpec::resolve(
                &domain.coating,
                domain.wavelength,
                domain.n_in,
                domain.n_out,
                domain.mirror,
                &mut layers,
            )?);
        }
        Ok(Self {
            specs: fill(b, &specs, ctx)?,
            layers: fill(b, &layers, ctx)?,
            domains: domains.to_vec(),
        })
    }
    /// Return powers and flux-normalised complex amplitudes. Explicitly
    /// complete only flagged samples with the original f64 model.
    pub fn evaluate(
        &self,
        b: &dyn ComputeBackend,
        snapshot: mklight_core::Snapshot,
        queries: &[CoatingQuery],
        ctx: &JobContext,
    ) -> MkResult<CoatingResult> {
        ctx.check()?;
        for q in queries {
            if q.coating as usize >= self.domains.len()
                || !q.cosine.is_finite()
                || !(0.0..=1.0).contains(&q.cosine.to_f64())
            {
                return Err(bad("coating query index or cosine"));
            }
        }
        let mut p = ResidentPlan::new(snapshot);
        p.settings = serde_json::json!({"operation":"coating","version":crate::consts::MKL_RAD_VERSION,"specs_hash":blake3::hash(bytemuck::cast_slice(b.read_view(self.specs.view())?.words())).to_hex().to_string(),"layers_hash":blake3::hash(bytemuck::cast_slice(b.read_view(self.layers.view())?.words())).to_hex().to_string(),"query_hash":blake3::hash(bytemuck::cast_slice(queries)).to_hex().to_string()});
        let specs = p.import(self.specs.view());
        let layers = p.import(self.layers.view());
        let query = fill(b, queries, ctx)?;
        let query = p.import(query.view());
        let output = p.scratch(record_desc::<CoatingValue>(queries.len()));
        p.operations
            .push(ResidentOp::Radiometry(Request::Coating(CoatingRequest {
                specs: p.view(specs)?,
                layers: p.view(layers)?,
                queries: p.view(query)?,
                output,
                tile: 16384,
            })));
        p.outputs = vec![output];
        let mut r = b.submit_plan(p, &ctx.subrange(0., 0.9))?.wait()?;
        let guard = b.read_view(r.outputs[0].view())?;
        let mut values = Vec::with_capacity(queries.len());
        for (i, v) in bytemuck::cast_slice::<u32, CoatingValue>(guard.words())
            .iter()
            .enumerate()
        {
            if i % 256 == 0 {
                ctx.check()?;
            }
            let mut v = *v;
            if v.status == crate::consts::MKL_RAD_REFERENCE {
                let q = queries[i];
                let d = &self.domains[q.coating as usize];
                let a = d.coating.amplitudes(
                    d.wavelength,
                    d.n_in,
                    d.n_out,
                    q.cosine.to_f64(),
                    d.mirror,
                );
                let rt = d
                    .coating
                    .rt(d.wavelength, d.n_in, d.n_out, q.cosine.to_f64(), d.mirror);
                for (k, z) in [a.rs, a.rp, a.ts_flux(), a.tp_flux()].iter().enumerate() {
                    v.amplitudes[2 * k] = Df32::from_f64(z.re);
                    v.amplitudes[2 * k + 1] = Df32::from_f64(z.im);
                }
                v.powers = [
                    rt.rs,
                    rt.rp,
                    rt.ts,
                    rt.tp,
                    1. - rt.rs - rt.ts,
                    1. - rt.rp - rt.tp,
                ]
                .map(Df32::from_f64);
                v.status = 0;
            }
            values.push(v);
        }
        if r.stats.fallback_count > 0 {
            r.run_info.warnings.push(format!("P7 coating: {} direct evaluations completed with CPU f64 (grazing film or phase range)",r.stats.fallback_count));
        }
        ctx.check()?;
        ctx.report(1., "coating complete");
        Ok(CoatingResult {
            values,
            run_info: r.run_info,
            stats: r.stats,
        })
    }
}
