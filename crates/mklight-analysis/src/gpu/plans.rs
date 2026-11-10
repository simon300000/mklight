//! Plans of the geometric analyses. The fields of a spot diagram, geometric
//! MTF, footprint or ray grid are split into groups of at least
//! [`GROUP_MIN_RAYS`] rays ([`Geometry::field_groups`]), each group a plan
//! submitted as soon as its rays are aimed ([`super::aim`]), so the host
//! aims group k + 1 while the device traces and reduces group k (several
//! plans in flight, results collected at the end). Smaller grids form one
//! group.

use super::*;

/// Rays per field group at least: below this a plan's fixed cost
/// (submission, command buffers, completion) is comparable to its work.
pub const GROUP_MIN_RAYS: usize = 1 << 15;

/// Consecutive groups of `fields` field positions with `rays_per_field`
/// rays each, of at least `min_rays` rays (the last group takes the
/// remainder).
pub fn field_groups(
    fields: usize,
    rays_per_field: usize,
    min_rays: usize,
) -> Vec<std::ops::Range<usize>> {
    let min = min_rays.max(1);
    let per = min.div_ceil(rays_per_field.max(1)).max(1);
    let mut out: Vec<std::ops::Range<usize>> = (0..fields)
        .step_by(per)
        .map(|s| s..(s + per).min(fields))
        .collect();
    // Merge a short tail into its predecessor.
    if out.len() > 1 && out.last().is_some_and(|r| r.len() * rays_per_field < min) {
        let tail = out.pop().unwrap();
        out.last_mut().unwrap().end = tail.end;
    }
    if out.is_empty() {
        out.push(0..fields);
    }
    out
}

/// A plan in flight, with the aiming evidence gathered while it was built.
pub(crate) struct Submitted {
    pending: mklight_gpu::Pending<PlanResult>,
    evidence: aim::AimEvidence,
}

impl Geometry<'_> {
    /// The field groups ([`field_groups`]) of this adapter's
    /// [`Tuning::group_min_rays`].
    pub(crate) fn field_groups(
        &self,
        fields: usize,
        rays_per_field: usize,
    ) -> Vec<std::ops::Range<usize>> {
        field_groups(fields, rays_per_field, self.tuning.group_min_rays)
    }
    /// Submit a plan without waiting for it.
    pub(crate) fn submit(&self, p: ResidentPlan, ctx: &JobContext) -> MkResult<Submitted> {
        let evidence = self.take_aim_evidence();
        Ok(Submitted {
            pending: self.backend.submit_plan(p, ctx)?,
            evidence,
        })
    }
    /// Wait for a submitted plan; its run evidence names the aiming work.
    pub(crate) fn finish(&self, s: Submitted) -> MkResult<PlanResult> {
        let mut r = s.pending.wait()?;
        r.run_info.settings["aiming"] = serde_json::to_value(s.evidence).unwrap();
        Ok(r)
    }
}

/// Accumulate the evidence of several plans into one run: the first plan's
/// run, the others under `extra_plans`, durations summed.
pub(crate) fn merge_run(run: &mut Option<RunInfo>, r: RunInfo) {
    match run {
        Some(run) => add_run(run, r),
        None => *run = Some(r),
    }
}

/// Add a further plan's evidence to the first plan's `run` ([`merge_run`]).
pub(crate) fn add_run(run: &mut RunInfo, r: RunInfo) {
    run.duration_ms += r.duration_ms;
    if !run.settings["extra_plans"].is_array() {
        run.settings["extra_plans"] = serde_json::json!([]);
    }
    run.settings["extra_plans"]
        .as_array_mut()
        .unwrap()
        .push(r.settings);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_cover_fields_with_large_plans() {
        let field_groups = |fields, rays| field_groups(fields, rays, GROUP_MIN_RAYS);
        assert_eq!(field_groups(6, 635), vec![0..6]);
        assert_eq!(
            field_groups(6, 1 << 15),
            (0..6).map(|i| i..i + 1).collect::<Vec<_>>()
        );
        let g = field_groups(7, 20_000);
        assert_eq!(g.first().unwrap().start, 0);
        assert_eq!(g.last().unwrap().end, 7);
        for w in g.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert!(g.iter().all(|r| r.len() * 20_000 >= GROUP_MIN_RAYS));
        assert_eq!(field_groups(0, 10), vec![0..0]);
    }
}
