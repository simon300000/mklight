//! Division of the geometric analyses' stages between host and device
//! (study key `geometric`, [`mklight_core::split`]). Rays are aimed by the
//! adapter's aiming design ([`super::aim`]) in every division:
//!
//! * [`Division::Fused`] (default): one plan per analysis; the device traces
//!   the aimed grid and reduces it (coordinates, moments, characteristic
//!   sums) without the host seeing a ray.
//! * [`Division::Pipelined`]: the fields are split into groups of at least
//!   [`PIPELINE_MIN_RAYS`] rays, each group a plan submitted as soon as its
//!   rays are aimed, so the host aims group k + 1 while the device traces
//!   and reduces group k (several plans in flight, results collected at the
//!   end). Smaller generations form one group, as in the fused division.
//! * [`Division::HostReduce`]: the device traces; the host reduces the hits
//!   in place from the shared planes with the reference's own reduction.

use super::*;
use mklight_core::split;

/// The study key of the division.
pub const KEY: &str = "geometric";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Division {
    #[default]
    Fused,
    Pipelined,
    HostReduce,
}
impl Division {
    /// The division selected for this run: `geometric` = `fused`,
    /// `pipelined` or `host-reduce`; the default without a selection.
    pub fn selected() -> MkResult<Self> {
        Ok(match split::variant(KEY).as_deref() {
            None | Some("fused") => Self::default(),
            Some("pipelined") => Self::Pipelined,
            Some("host-reduce") => Self::HostReduce,
            Some(other) => {
                return Err(mklight_core::MkError::InvalidParam(format!(
                    "unknown geometric division '{other}' (fused, pipelined, host-reduce)"
                )));
            }
        })
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Fused => "fused",
            Self::Pipelined => "pipelined",
            Self::HostReduce => "host-reduce",
        }
    }
    /// Field groups (ranges of field positions) of a generation of
    /// `rays_per_field` rays per field: one group unless pipelined.
    pub fn groups(self, fields: usize, rays_per_field: usize) -> Vec<std::ops::Range<usize>> {
        match self {
            Self::Pipelined => pipeline_groups(fields, rays_per_field),
            _ => std::iter::once(0..fields).collect(),
        }
    }
}

/// Rays per pipelined plan at least: below this a plan's fixed cost
/// (submission, command buffers, completion) is comparable to its work.
pub const PIPELINE_MIN_RAYS: usize = 1 << 15;
static MIN_RAYS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(PIPELINE_MIN_RAYS);
/// Override [`PIPELINE_MIN_RAYS`] (tests exercise several groups on small
/// grids); `None` restores it.
#[doc(hidden)]
pub fn set_pipeline_min_rays(n: Option<usize>) {
    MIN_RAYS.store(
        n.unwrap_or(PIPELINE_MIN_RAYS).max(1),
        std::sync::atomic::Ordering::Relaxed,
    );
}

/// Consecutive field groups of at least [`PIPELINE_MIN_RAYS`] rays each
/// (the last group takes the remainder).
pub fn pipeline_groups(fields: usize, rays_per_field: usize) -> Vec<std::ops::Range<usize>> {
    let min = MIN_RAYS.load(std::sync::atomic::Ordering::Relaxed);
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
    /// Submit a plan without waiting for it (the pipelined division).
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
        if trace_evidence() >= 2 || trace_evidence() == 1 && s.evidence.generations > 0 {
            eprintln!(
                "mklight aiming: {} plan {:.2} ms device {:.2} ms",
                serde_json::to_string(&s.evidence).unwrap(),
                r.stats.wall_seconds * 1e3,
                r.stats.device_seconds * 1e3,
            );
        }
        r.run_info.settings["aiming"] = serde_json::to_value(s.evidence).unwrap();
        Ok(r)
    }
}

/// `MKLIGHT_TRACE_AIM_EVIDENCE=1` prints the aiming evidence of each plan
/// that aimed rays to stderr, `=2` of every plan, `=3` also each device
/// generation (development measurements).
pub(crate) fn trace_evidence() -> u32 {
    static ON: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("MKLIGHT_TRACE_AIM_EVIDENCE")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

/// Accumulate the evidence of several plans into one run: the first plan's
/// run, the others under `extra_plans`, durations summed.
pub(crate) fn merge_run(run: &mut Option<RunInfo>, r: RunInfo) {
    match run {
        Some(run) => {
            run.duration_ms += r.duration_ms;
            if !run.settings["extra_plans"].is_array() {
                run.settings["extra_plans"] = serde_json::json!([]);
            }
            run.settings["extra_plans"]
                .as_array_mut()
                .unwrap()
                .push(r.settings);
        }
        None => *run = Some(r),
    }
}
/// Record the division (and plan count) in a run's evidence.
pub(crate) fn mark(run: &mut RunInfo, division: Division, plans: usize) {
    run.settings["geometric"] = serde_json::json!({
        "division": division.name(),
        "plans": plans,
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn groups_cover_fields_with_large_plans() {
        assert_eq!(pipeline_groups(6, 635), vec![0..6]);
        assert_eq!(
            pipeline_groups(6, 1 << 15),
            (0..6).map(|i| i..i + 1).collect::<Vec<_>>()
        );
        let g = pipeline_groups(7, 20_000);
        assert_eq!(g.first().unwrap().start, 0);
        assert_eq!(g.last().unwrap().end, 7);
        for w in g.windows(2) {
            assert_eq!(w[0].end, w[1].start);
        }
        assert!(g.iter().all(|r| r.len() * 20_000 >= PIPELINE_MIN_RAYS));
        assert_eq!(pipeline_groups(0, 10), vec![0..0]);
    }
}
