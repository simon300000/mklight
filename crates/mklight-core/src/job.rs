//! Job framework: cooperative cancellation, progress reporting and run
//! metadata for reproducibility (ARCHITECTURE §State model).

use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::error::{MkError, MkResult};

#[derive(Clone, Default, Debug)]
pub struct CancelToken(Arc<AtomicBool>);

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn cancel(&self) {
        self.0.store(true, Ordering::Relaxed);
    }
    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::Relaxed)
    }
    /// `Err(Cancelled)` once the token has been cancelled.
    pub fn check(&self) -> MkResult<()> {
        if self.is_cancelled() {
            Err(MkError::Cancelled)
        } else {
            Ok(())
        }
    }
}

#[derive(Default, Debug)]
struct ProgressInner {
    fraction_bits: AtomicU64,
    stage: Mutex<String>,
}

/// Shared progress state, written by the job and read by the UI.
#[derive(Clone, Default, Debug)]
pub struct Progress(Arc<ProgressInner>);

impl Progress {
    pub fn fraction(&self) -> f64 {
        f64::from_bits(self.0.fraction_bits.load(Ordering::Relaxed))
    }
    pub fn stage(&self) -> String {
        self.0.stage.lock().map(|s| s.clone()).unwrap_or_default()
    }
    fn set_fraction(&self, f: f64) {
        let f = if f.is_finite() {
            f.clamp(0.0, 1.0)
        } else {
            0.0
        };
        // Monotone: progress never goes backwards.
        let mut cur = self.0.fraction_bits.load(Ordering::Relaxed);
        while f > f64::from_bits(cur) {
            match self.0.fraction_bits.compare_exchange_weak(
                cur,
                f.to_bits(),
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(c) => cur = c,
            }
        }
    }
    fn set_stage(&self, s: &str) {
        if let Ok(mut g) = self.0.stage.lock()
            && *g != s
        {
            s.clone_into(&mut g);
        }
    }
}

/// Handed to every long-running computation. `subrange` gives a nested stage
/// its own slice of the 0…1 progress bar.
#[derive(Clone, Debug)]
pub struct JobContext {
    pub cancel: CancelToken,
    pub progress: Progress,
    range: (f64, f64),
}

impl Default for JobContext {
    fn default() -> Self {
        Self::new()
    }
}

impl JobContext {
    pub fn new() -> Self {
        JobContext {
            cancel: CancelToken::new(),
            progress: Progress::default(),
            range: (0.0, 1.0),
        }
    }
    pub fn with(cancel: CancelToken, progress: Progress) -> Self {
        JobContext {
            cancel,
            progress,
            range: (0.0, 1.0),
        }
    }
    #[inline]
    pub fn check(&self) -> MkResult<()> {
        self.cancel.check()
    }
    /// Report local progress `f` in 0…1, mapped into this context's range.
    pub fn report(&self, f: f64, stage: &str) {
        let (a, b) = self.range;
        self.progress.set_fraction(a + f.clamp(0.0, 1.0) * (b - a));
        if !stage.is_empty() {
            self.progress.set_stage(stage);
        }
    }
    /// A child context that covers `[from, to]` of this context's range.
    pub fn subrange(&self, from: f64, to: f64) -> JobContext {
        let (a, b) = self.range;
        JobContext {
            cancel: self.cancel.clone(),
            progress: self.progress.clone(),
            range: (a + from * (b - a), a + to * (b - a)),
        }
    }
}

/// Floating-point model used for a computation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// CPU double precision (the reference).
    #[default]
    F64Reference,
    /// GPU or emulated single precision with the CONVENTIONS §7 techniques.
    F32Gpu,
}

/// Accuracy tier of a run (CONVENTIONS §7, GPU §9.4). Every result is
/// labelled with it; nothing below design grade is ever silent.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tier {
    /// Design-grade budgets and sampling (the default).
    #[default]
    Design,
    /// Opt-in preview: cheaper sampling and aiming within the preview
    /// budgets (position ≤ 0.05 λN, MTF ≤ 0.01, OPD RMS ≤ 0.02 waves);
    /// not for final decisions.
    Preview,
}

/// How the real-aim solves of a run stopped (CONVENTIONS §7).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AimStopKind {
    /// At the system's requested aiming tolerance.
    #[default]
    Requested,
    /// At the residual derived per field and wavelength from the propagated
    /// error (never looser than the tier's allocation, never tighter than
    /// the request).
    Conditioned,
}

/// Metadata attached to every result, so it can be reproduced and checked
/// for staleness.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RunInfo {
    pub engine_version: String,
    pub snapshot_hash: String,
    pub analysis: String,
    pub settings: serde_json::Value,
    pub backend: String,
    pub precision: Precision,
    pub seed: Option<u64>,
    pub started_unix_ms: u64,
    pub duration_ms: f64,
    /// Accuracy tier the result was computed at.
    #[serde(default)]
    pub tier: Tier,
    /// Aim-stop policy the run's real-aim solves used.
    #[serde(default)]
    pub aim_stop: AimStopKind,
    #[serde(default)]
    pub warnings: Vec<String>,
    /// Backend attempts that did not compute this result, in order: native
    /// runs that failed or were skipped/rejected before running, and explicit
    /// requests the capability policy downgraded. Their time is part of
    /// `duration_ms`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub failed_attempts: Vec<BackendAttempt>,
}

/// One backend attempt that did not produce the published result
/// (`RunInfo::failed_attempts`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BackendAttempt {
    /// Backend id (`gpu:metal`).
    pub backend: String,
    /// `failed` (ran and failed), `skipped` (the same input already failed
    /// deterministically on this device), `infeasible` (rejected by its
    /// capability/resource check before preparation), `downgraded` (an
    /// explicit request the static capability policy ran on the reference),
    /// or `reference_completion` (the native adapter completed the whole
    /// result on the reference).
    pub outcome: String,
    pub reason: String,
    /// Wall time spent in the attempt; 0 when it never started.
    pub elapsed_ms: f64,
}

/// Helper to build a `RunInfo` around a computation.
pub struct RunTimer {
    started_unix_ms: u64,
    t0: Instant,
}

impl RunTimer {
    pub fn start() -> Self {
        let started_unix_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        RunTimer {
            started_unix_ms,
            t0: Instant::now(),
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn finish(
        self,
        snapshot: &crate::Snapshot,
        analysis: &str,
        settings: &impl Serialize,
        backend: &str,
        precision: Precision,
        seed: Option<u64>,
        warnings: Vec<String>,
    ) -> RunInfo {
        RunInfo {
            engine_version: crate::ENGINE_VERSION.to_string(),
            snapshot_hash: snapshot.hash().to_hex(),
            analysis: analysis.to_string(),
            settings: serde_json::to_value(settings).unwrap_or(serde_json::Value::Null),
            backend: backend.to_string(),
            precision,
            seed,
            started_unix_ms: self.started_unix_ms,
            duration_ms: self.t0.elapsed().as_secs_f64() * 1e3,
            tier: Tier::Design,
            aim_stop: AimStopKind::Requested,
            warnings,
            failed_attempts: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subrange_maps_progress() {
        let ctx = JobContext::new();
        let sub = ctx.subrange(0.5, 1.0);
        sub.report(0.5, "half of second half");
        assert!((ctx.progress.fraction() - 0.75).abs() < 1e-12);
        assert_eq!(ctx.progress.stage(), "half of second half");
        sub.cancel.cancel();
        assert_eq!(ctx.check(), Err(MkError::Cancelled));
    }
}
