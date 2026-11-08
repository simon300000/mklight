//! Device-traced ray batches for the host-reduce division
//! ([`division::Division::HostReduce`]): the device traces the aimed grid
//! and its chief rays, and the host reads the hits in place into the
//! reference's [`RayBatch`], which the reference's own reductions
//! (`*_from_rays`) consume.
use super::*;
use crate::spot::{FieldRays, ImageHit, RayBatch};
use division::Division;

/// The image hit of ray `i` of resident hit planes (`f` [15, n], `status`
/// [3, n]), `None` unless it arrived.
pub(super) fn image_hit(f: &[f32], status: &[u32], n: usize, i: usize) -> Option<ImageHit> {
    (status[i] == 0).then(|| ImageHit {
        point: [value(f, n, i, 0), value(f, n, i, 1)],
        dir: mklight_core::Vec3::new(value(f, n, i, 3), value(f, n, i, 4), value(f, n, i, 5)),
    })
}

/// The reference's field rays from device-traced grid and chief hits, read
/// in place (`floats`/`status`: grid, then chiefs) for the fields of `g`.
pub(super) fn host_fields(
    t: &SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    g: &ResidentGrid,
    floats: [&[f32]; 2],
    status: [&[u32]; 2],
    warnings: &mut Vec<String>,
) -> MkResult<Vec<FieldRays>> {
    let sys = t.system();
    let weights: Vec<f64> = wls.iter().map(|&w| sys.wavelength_weights[w]).collect();
    let n = g.hits.rays.count;
    let np = g.pupil_count;
    let nc = g.chiefs()?.rays.count;
    let ncw = g.chief_wavelengths.len();
    let column = |w: usize| g.chief_wavelengths.iter().position(|&v| v == w).unwrap();
    let nw = wls.len();
    fields
        .iter()
        .enumerate()
        .map(|(fi, &f)| {
            let chief = |w: usize| image_hit(floats[1], status[1], nc, fi * ncw + column(w));
            let pc = chief(sys.primary_wavelength);
            let wl_chiefs: Vec<_> = wls.iter().map(|&w| chief(w)).collect();
            let rays = (fi * nw * np..(fi + 1) * nw * np)
                .map(|i| (wls[(i / np) % nw], image_hit(floats[0], status[0], n, i)));
            FieldRays::from_hits(f, wls, &weights, pc, &wl_chiefs, rays, warnings)
        })
        .collect()
}

/// A device-traced batch and the evidence of its plans.
pub(crate) struct TracedBatch {
    pub batch: RayBatch,
    pub warnings: Vec<String>,
    pub run: RunInfo,
}

impl Geometry<'_> {
    /// Trace `fields` (positions in the tracer's field list) × `wls` ×
    /// `pupil` and the chief rays on the device and read them into a
    /// [`RayBatch`]. Pipelined, each field group is its own plan, aimed
    /// while the device traces the groups before it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn traced_batch(
        &self,
        t: &SequentialTracer,
        fields: &[usize],
        wls: &[usize],
        pupil: &[[f64; 2]],
        division: Division,
        settings: &serde_json::Value,
        seed: Option<u64>,
        ctx: &JobContext,
    ) -> MkResult<TracedBatch> {
        let b = self.backend;
        let points: Vec<_> = fields.iter().map(|&i| t.field(i).point).collect();
        let groups = division.groups(fields.len(), wls.len() * pupil.len());
        let mut submitted = Vec::with_capacity(groups.len());
        for (k, range) in groups.iter().enumerate() {
            let sub = ctx.subrange(
                0.9 * k as f64 / groups.len() as f64,
                0.9 * (k + 1) as f64 / groups.len() as f64,
            );
            let mut p = ResidentPlan::new(t.system().snapshot.clone());
            p.settings = settings.clone();
            p.seed = seed;
            let mut g = self.append_grid(
                &mut p,
                &points[range.clone()],
                wls,
                pupil,
                t.aiming(),
                t.system().image_index(),
                true,
                false,
                &sub,
            )?;
            self.append_chiefs(&mut p, &mut g)?;
            let chiefs = g.chiefs()?.clone();
            p.outputs = vec![
                g.hits.rays.floats,
                g.hits.status,
                chiefs.rays.floats,
                chiefs.status,
            ];
            submitted.push((range.clone(), g, self.submit(p, &sub)?));
        }
        let mut run: Option<RunInfo> = None;
        let mut warnings = Vec::new();
        let mut out = Vec::with_capacity(fields.len());
        for (range, g, sub) in submitted {
            let r = self.finish(sub)?;
            ctx.check()?;
            let views: Vec<_> = r
                .outputs
                .iter()
                .map(|h| b.read_view(h.view()))
                .collect::<MkResult<_>>()?;
            out.extend(host_fields(
                t,
                &fields[range],
                wls,
                &g,
                [views[0].as_slice()?, views[2].as_slice()?],
                [views[1].as_slice()?, views[3].as_slice()?],
                &mut warnings,
            )?);
            division::merge_run(&mut run, r.run_info);
        }
        let mut run = run.unwrap();
        division::mark(&mut run, division, groups.len());
        Ok(TracedBatch {
            batch: RayBatch {
                wavelengths: wls.to_vec(),
                fields: out,
            },
            warnings,
            run,
        })
    }
}
