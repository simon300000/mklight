//! Native result adapters. Engine routing and qualification policy remain in P11.
//!
//! Every job reads its work-split design once ([`Design::selected`],
//! `gpu::split`) and names it in `RunInfo.settings.work_split`. The design
//! decides where pupils are traced and resampled ([`Source`]), whether the
//! sampling decisions that read them (image windows, refinements, automatic
//! frequencies and focus ranges) run on the host with the reference's own
//! rules or on the device, and whether per-field host preparation overlaps
//! the device transforms of other fields ([`split::schedule`]).
use super::split::{self, Design};
use super::traced::accumulate;
use super::*;
use crate::analysis::{self as a, PupilSampling};
use crate::{PupilSettings, ReferencePoint, ZernikeBasis};
use mklight_core::job::RunTimer;
use mklight_core::memo::Memo;
use mklight_trace::{PreparedField, SequentialTracer};
use rayon::prelude::*;
use serde::Serialize;
pub(super) struct Field {
    pub(super) index: usize,
    pub(super) value: [f64; 2],
    pub(super) reference: [f64; 2],
    pub(super) pupils: Vec<(usize, f64, ResidentTracedPupil)>,
}
impl Field {
    pub(super) fn weighted(&self) -> Vec<(&ResidentPupil, f64)> {
        self.pupils.iter().map(|(_, w, p)| (&p.pupil, *w)).collect()
    }
    pub(super) fn sampling(&self) -> Vec<PupilSampling> {
        self.pupils
            .iter()
            .map(|(wl, _, p)| PupilSampling {
                wavelength: *wl,
                grid: p.pupil.n,
            })
            .collect()
    }
    /// The reference pupils with their weights, when the host resampled
    /// every pupil of the field.
    pub(super) fn host_weighted(&self) -> Option<Vec<(&PupilFunction, f64)>> {
        self.pupils
            .iter()
            .map(|(_, w, p)| p.host_pupil().map(|h| (h, *w)))
            .collect()
    }
}
/// A field to trace: its index (system field or sweep position), field
/// value and prepared field.
pub(super) type Request<'f> = (usize, [f64; 2], &'f PreparedField);
/// A per-field chain's input: a traced field, or (streaming) a request its
/// host stage traces.
pub(super) enum Item<'f> {
    Traced(Field),
    Request(Request<'f>),
}
/// A per-field chain's outcome: a value with its statistics and warnings,
/// or an unusable field (Trace/Numerical failure of one of its pupils).
// Moved once per field between its stages; boxing would gain nothing.
#[allow(clippy::large_enum_variant)]
pub(super) enum Staged<T> {
    Done {
        value: T,
        stats: PlanStats,
        warnings: Vec<String>,
    },
    Skipped {
        index: usize,
        error: MkError,
    },
}
impl<T> Staged<T> {
    /// The device stage `f` of a prepared field, carrying its statistics and
    /// warnings; skipped fields pass through.
    pub(super) fn then<U>(
        self,
        f: impl FnOnce(T, &mut PlanStats, &mut Vec<String>) -> MkResult<U>,
    ) -> MkResult<Staged<U>> {
        Ok(match self {
            Staged::Done {
                value,
                mut stats,
                mut warnings,
            } => Staged::Done {
                value: f(value, &mut stats, &mut warnings)?,
                stats,
                warnings,
            },
            Staged::Skipped { index, error } => Staged::Skipped { index, error },
        })
    }
}
/// Merge per-field outcomes in order: statistics and warnings, a warning
/// for each skipped field, and the first skip's error when no field is
/// usable.
pub(super) fn gather<T>(
    results: Vec<MkResult<Staged<T>>>,
    stats: &mut PlanStats,
    warnings: &mut Vec<String>,
) -> MkResult<Vec<T>> {
    let mut out = vec![];
    let mut first = None;
    for r in results {
        match r? {
            Staged::Done {
                value,
                stats: s,
                warnings: w,
            } => {
                accumulate(stats, &s);
                warnings.extend(w);
                out.push(value);
            }
            Staged::Skipped { index, error } => {
                warnings.push(format!("field {index} skipped: {error}"));
                first.get_or_insert(error);
            }
        }
    }
    match first {
        Some(e) if out.is_empty() => Err(e),
        _ => Ok(out),
    }
}
/// The host pupils of a field transformed by the reference's code (designs
/// with host pupils only).
fn host_pupils(f: &Field) -> MkResult<Vec<(&PupilFunction, f64)>> {
    f.host_weighted()
        .ok_or_else(|| MkError::Unsupported("host transform without host pupils".into()))
}
/// The PSF result of one field (and wavelength, unless polychromatic).
fn field_psf(f: &Field, wl: Option<usize>, psf: Psf) -> a::FieldPsf {
    let (_, i, j) = psf.peak();
    a::FieldPsf {
        field: f.index,
        field_value: f.value,
        wavelength: wl,
        pupil_sampling: f
            .sampling()
            .into_iter()
            .filter(|s| wl.is_none_or(|w| s.wavelength == w))
            .collect(),
        reference_mm: f.reference,
        strehl: psf.strehl(),
        centroid_mm: psf.centroid(),
        peak_mm: [psf.x(i), psf.y(j)],
        psf,
    }
}
/// The MTF result of one field, with the detector product when supplied.
fn field_mtf(
    f: &Field,
    o: crate::MtfOrientation,
    c: crate::MtfCurve,
    detector: Option<crate::detector::DetectorCurve>,
) -> a::FieldMtf {
    let product =
        |d: &[f64], c: &[f64]| -> Vec<f64> { d.iter().zip(c).map(|(a, b)| a * b).collect() };
    a::FieldMtf {
        field: f.index,
        field_value: f.value,
        pupil_sampling: f.sampling(),
        tangential_direction: o.tangential,
        system_tangential: detector
            .as_ref()
            .map(|d| product(&d.tangential, &c.tangential)),
        system_sagittal: detector.as_ref().map(|d| product(&d.sagittal, &c.sagittal)),
        tangential: c.tangential,
        sagittal: c.sagittal,
        diffraction_limit: c.diffraction_limit,
        cutoff: c.cutoff,
        detector,
    }
}
/// Work of the matrix-transform PSFs of a field on a `grid²` image: the
/// separable chirp transform's row and column passes, `Σ n·grid·(n + grid)`
/// over its pupils.
fn transform_work(f: &Field, grid: usize) -> f64 {
    f.pupils
        .iter()
        .map(|(_, _, p)| {
            let n = p.pupil.n as f64;
            n * grid as f64 * (n + grid as f64)
        })
        .sum()
}
/// Work of direct-autocorrelation OTFs of a field at `points` frequency
/// vectors (planes × frequencies): `points · Σ n²` over its pupils.
fn correlation_work(f: &Field, points: usize) -> f64 {
    points as f64
        * f.pupils
            .iter()
            .map(|(_, _, p)| (p.pupil.n * p.pupil.n) as f64)
            .sum::<f64>()
}
/// Host-transform rules (`split::HostRule`): the `by-size` work limit below
/// which the reference's host code is predicted faster than the device
/// chain, and the host/device time ratio `co-op` weighs. Measured per field
/// on the M2 Pro (default and dense settings, three lenses): matrix-transform
/// PSFs of 5e8–2e10 took 60–220 ms on the host against 10–85 ms on the
/// device, so only tiny problems stay on the host; direct OTFs of 2e7–3e8
/// took 2–4× the device time; focus sweeps took about the same on both.
const PSF_HOST: split::HostRule = split::HostRule {
    limit: 1e8,
    ratio: 4.,
};
const OTF_HOST: split::HostRule = split::HostRule {
    limit: 1e6,
    ratio: 2.5,
};
const FOCUS_HOST: split::HostRule = split::HostRule {
    limit: 1e6,
    ratio: 1.3,
};
/// Trace and Numerical errors make one pupil (and its field) unusable;
/// every other error fails the whole run.
fn per_pupil(e: MkError) -> MkResult<MkError> {
    match e {
        MkError::Trace(_) | MkError::Numerical(_) => Ok(e),
        e => Err(e),
    }
}

/// Where a run's pupils are traced and resampled, after its design.
pub(super) struct Source<'a> {
    pub(super) b: &'a dyn ComputeBackend,
    pub(super) t: &'a SequentialTracer,
    pub(super) design: Design,
    pub(super) memo: Option<&'a dyn Memo>,
    device: Option<PupilTracer<'a>>,
    /// Where each scheduled item's transform ran ([`Self::record`]).
    ran: std::sync::Mutex<Vec<Option<split::Ran>>>,
}
impl<'a> Source<'a> {
    pub(super) fn new(
        b: &'a dyn ComputeBackend,
        t: &'a SequentialTracer,
        memo: Option<&'a dyn Memo>,
        ctx: &JobContext,
    ) -> MkResult<Self> {
        let design = Design::selected();
        let device = if design.host_lattice() {
            None
        } else {
            Some(PupilTracer::new(b, t, ctx)?.with_host_aim(design == Design::CpuAim))
        };
        Ok(Self {
            b,
            t,
            design,
            memo,
            device,
            ran: Default::default(),
        })
    }
    /// Keep a schedule's placements for the run's evidence.
    pub(super) fn record(&self, ran: Vec<Option<split::Ran>>) {
        self.ran
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .extend(ran);
    }
    /// The host transform of a schedule, for the designs that place
    /// transforms on the host (host pupils only).
    pub(super) fn reference<'r, P, R>(
        &self,
        transform: &'r (dyn Fn(usize, P) -> MkResult<R> + Sync),
        work: &'r (dyn Fn(&P) -> f64 + Sync),
        rule: split::HostRule,
    ) -> Option<split::Reference<'r, P, R>> {
        self.design.host_pupil().then_some(split::Reference {
            transform,
            work,
            limit: rule.limit * split::by_size_scale(),
            ratio: rule.ratio,
        })
    }
    pub(super) fn snapshot(&self) -> Snapshot {
        self.t.system().snapshot.clone()
    }
    /// Trace every request at `wls`: a field whose pupil fails with a
    /// Trace/Numerical error (first in wavelength order) is returned as that
    /// error; any other error fails the call.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn trace(
        &self,
        requests: &[Request<'_>],
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
        stats: &mut PlanStats,
        warnings: &mut Vec<String>,
    ) -> MkResult<Vec<MkResult<Field>>> {
        match &self.device {
            Some(service) => trace_fields(
                service, self.t, requests, wls, settings, ctx, stats, warnings,
            ),
            None => self.host_trace(requests, wls, settings, ctx, stats, warnings),
        }
    }
    /// [`Self::trace`] with the reference's lattices: every (field,
    /// wavelength) lattice is traced on the host at once (shared through
    /// the memo), then written into shared allocations and resampled on
    /// the host or the device. Reference points follow the reference
    /// (`crate::analysis`): the primary chief ray, or the weighted centroid.
    fn host_trace(
        &self,
        requests: &[Request<'_>],
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
        stats: &mut PlanStats,
        warnings: &mut Vec<String>,
    ) -> MkResult<Vec<MkResult<Field>>> {
        settings.validate()?;
        let total = (requests.len() * wls.len()).max(1) as f64;
        let trace_ctx = ctx.subrange(0., 0.8);
        let jobs: Vec<_> = requests
            .iter()
            .flat_map(|&(_, _, field)| wls.iter().map(move |&(wl, _)| (field, wl)))
            .enumerate()
            .map(|(k, (field, wl))| crate::shared::PupilJob {
                field,
                wl,
                ctx: trace_ctx.subrange(k as f64 / total, (k + 1) as f64 / total),
            })
            .collect();
        let lattices =
            crate::shared::traced_lattices(self.t, &jobs, settings, self.memo, &trace_ctx);
        let snapshot = self.snapshot();
        let host = self.design.host_pupil();
        let place_ctx = ctx.subrange(0.8, 1.);
        let place = |k: usize, r: MkResult<(mklight_core::memo::NodeKey, std::sync::Arc<_>)>| {
            let sub = place_ctx.subrange(k as f64 / total, (k + 1) as f64 / total);
            match r.and_then(|(key, lattice)| {
                split::place(
                    self.b, &snapshot, key, lattice, settings, host, self.memo, &sub,
                )
            }) {
                Ok(p) => Ok(Ok(p)),
                Err(e) => per_pupil(e).map(Err),
            }
        };
        // Host resampling is CPU work; device resampling chains wait on
        // the device.
        let placed: Vec<MkResult<MkResult<split::Placed>>> = if host {
            lattices
                .into_par_iter()
                .enumerate()
                .map(|(k, r)| place(k, r))
                .collect()
        } else {
            let lattices: Vec<_> = lattices
                .into_iter()
                .map(|r| std::sync::Mutex::new(Some(r)))
                .collect();
            concurrent(&lattices, |k, r| {
                place(k, r.lock().unwrap().take().expect("one placement"))
            })
        };
        let mut placed = placed.into_iter();
        let primary = self.t.system().primary_wavelength;
        let mut out = vec![];
        for &(index, value, field) in requests {
            let mut pupils = vec![];
            let mut error = None;
            for (&(wl, weight), r) in wls.iter().zip(placed.by_ref().take(wls.len())) {
                match r? {
                    Ok(p) => {
                        accumulate(stats, &p.stats);
                        warnings.extend(p.warnings);
                        pupils.push((wl, weight, p.value));
                    }
                    Err(e) => {
                        error.get_or_insert(e);
                    }
                }
            }
            if let Some(e) = error {
                out.push(Err(e));
                continue;
            }
            let first = pupils[0].2.info.reference_point_mm;
            let reference = match settings.reference {
                ReferencePoint::Chief => {
                    let c = self.t.chief(field, primary);
                    if c.status.is_ok() {
                        [c.ray.pos.x, c.ray.pos.y]
                    } else {
                        [first[0], first[1]]
                    }
                }
                ReferencePoint::Centroid => {
                    let mut v = [0.; 2];
                    for (_, w, p) in &pupils {
                        for (j, x) in v.iter_mut().enumerate() {
                            *x += w * p.info.reference_point_mm[j];
                        }
                    }
                    v
                }
            };
            for (_, _, p) in &mut pupils {
                let r = p.info.reference_point_mm;
                p.set_image_offset([r[0] - reference[0], r[1] - reference[1]]);
            }
            out.push(Ok(Field {
                index,
                value,
                reference,
                pupils,
            }));
        }
        Ok(out)
    }
    /// The usable fields of `requests`, skipping (with a warning) fields
    /// whose pupils fail; fails when none is usable.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn fields(
        &self,
        requests: &[Request<'_>],
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
        stats: &mut PlanStats,
        warnings: &mut Vec<String>,
    ) -> MkResult<Vec<Field>> {
        let traced = self.trace(requests, wls, settings, ctx, stats, warnings)?;
        let mut out = vec![];
        let mut first_error = None;
        for (&(index, _, _), f) in requests.iter().zip(traced) {
            match f {
                Ok(f) => out.push(f),
                Err(e) => {
                    warnings.push(format!("field {index} skipped: {e}"));
                    first_error.get_or_insert(e);
                }
            }
        }
        if out.is_empty()
            && let Some(e) = first_error
        {
            return Err(e);
        }
        Ok(out)
    }
    /// The per-field chain inputs of `requests`: requests the host stage
    /// traces one field at a time (`stream`), or the usable fields traced
    /// together.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn items<'f>(
        &self,
        requests: Vec<Request<'f>>,
        stream: bool,
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
        stats: &mut PlanStats,
        warnings: &mut Vec<String>,
    ) -> MkResult<Vec<Item<'f>>> {
        if stream {
            settings.validate()?;
            return Ok(requests.into_iter().map(Item::Request).collect());
        }
        Ok(self
            .fields(&requests, wls, settings, ctx, stats, warnings)?
            .into_iter()
            .map(Item::Traced)
            .collect())
    }
    /// The field of a chain input (tracing a request), or its index and
    /// error when it is unusable.
    pub(super) fn ready(
        &self,
        item: Item<'_>,
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
        stats: &mut PlanStats,
        warnings: &mut Vec<String>,
    ) -> MkResult<Result<Field, (usize, MkError)>> {
        match item {
            Item::Traced(f) => Ok(Ok(f)),
            Item::Request(r) => Ok(self
                .trace(&[r], wls, settings, ctx, stats, warnings)?
                .pop()
                .expect("one field")
                .map_err(|e| (r.0, e))),
        }
    }
    /// Blur estimates `(extent, rms)` of every pupil of `f`: the
    /// reference's rule on host pupils, the device's otherwise.
    pub(super) fn blurs(
        &self,
        f: &Field,
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<Vec<(f64, f64)>> {
        f.pupils
            .iter()
            .map(|(_, _, p)| match p.host_pupil() {
                Some(h) => Ok(crate::sampling::blur(h)),
                None => {
                    let r = p.pupil.blur(self.b, self.snapshot(), ctx)?;
                    accumulate(stats, &r.stats);
                    Ok((r.value.extent, r.value.rms))
                }
            })
            .collect()
    }
    /// The automatic image grid of `field` at `spacing` (or `requested`).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn image_grid(
        &self,
        field: &Field,
        requested: usize,
        spacing: f64,
        method: a::PsfMethod,
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<usize> {
        if requested != 0 {
            return Ok(requested);
        }
        if let Some(host) = field.host_weighted() {
            return Ok(a::psf::image_grid(0, method, &host, spacing));
        }
        let mut side = 0f64;
        for ((_, _, p), (extent, _)) in field.pupils.iter().zip(self.blurs(field, ctx, stats)?) {
            side = side.max(2. * (extent + 16. * p.pupil.lambda_f()));
        }
        Ok(((side / spacing).ceil() as usize).clamp(
            64,
            if method == a::PsfMethod::Huygens {
                1024
            } else {
                2048
            },
        ))
    }
    /// Refine automatic pupils so their alias-free field holds `window`
    /// (explicit grids are kept); host pupils refine in parallel on the
    /// host. Returns the resampling warnings.
    pub(super) fn ensure_window(
        &self,
        field: &mut Field,
        window: f64,
        s: &PupilSettings,
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<Vec<String>> {
        if s.grid != 0 {
            return Ok(vec![]);
        }
        let grids: Vec<usize> = field
            .pupils
            .iter()
            .map(|(_, _, p)| {
                ((window / p.pupil.lambda_f())
                    .max(p.pupil.n as f64)
                    .min(1024.)
                    .ceil() as usize)
                    .next_power_of_two()
            })
            .collect();
        self.refine(field, &grids, s.supersample, ctx, stats)
    }
    /// Resample every pupil of `field` to its grid of `grids`.
    pub(super) fn refine(
        &self,
        field: &mut Field,
        grids: &[usize],
        supersample: usize,
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<Vec<String>> {
        let snapshot = self.snapshot();
        let host = field.host_weighted().is_some();
        let refined: Vec<MkResult<(PlanStats, Vec<String>)>> = if host {
            field
                .pupils
                .par_iter_mut()
                .zip(grids)
                .map(|((_, _, p), &n)| {
                    p.refine(self.b, snapshot.clone(), n, supersample, self.memo, ctx)
                })
                .collect()
        } else {
            field
                .pupils
                .iter_mut()
                .zip(grids)
                .map(|((_, _, p), &n)| {
                    p.refine(self.b, snapshot.clone(), n, supersample, self.memo, ctx)
                })
                .collect()
        };
        let mut warnings = vec![];
        for r in refined {
            let (s, w) = r?;
            accumulate(stats, &s);
            warnings.extend(w);
        }
        Ok(warnings)
    }
    /// The phase-contrast frequency limit of a field's pupils.
    pub(super) fn useful(
        &self,
        field: &Field,
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<f64> {
        if let Some(host) = field.host_weighted() {
            return Ok(crate::sampling::useful_frequency(&host));
        }
        let r = useful_frequency(self.b, self.snapshot(), &field.weighted(), ctx)?;
        accumulate(stats, &r.stats);
        Ok(r.value)
    }
    /// The smallest [`Self::useful`] of `fields`, evaluated side by side.
    pub(super) fn useful_limit(
        &self,
        fields: &[&Field],
        ctx: &JobContext,
        stats: &mut PlanStats,
    ) -> MkResult<f64> {
        let mut limit = f64::INFINITY;
        for r in concurrent(fields, |_, f| {
            let mut s = PlanStats::default();
            self.useful(f, ctx, &mut s).map(|v| (v, s))
        }) {
            let (v, s) = r?;
            accumulate(stats, &s);
            limit = limit.min(v);
        }
        Ok(limit)
    }
    pub(super) fn info(
        &self,
        timer: RunTimer,
        name: &str,
        settings: &impl Serialize,
        stats: &PlanStats,
        warnings: Vec<String>,
    ) -> RunInfo {
        let mut r = info(self.b, self.t, timer, name, settings, stats, warnings);
        r.settings["work_split"] = self.design.evidence();
        let ran = self.ran.lock().unwrap_or_else(|e| e.into_inner());
        if !ran.is_empty() {
            r.settings["work_split"]["items"] = serde_json::json!(*ran);
        }
        r
    }
}

pub(super) fn info(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    timer: RunTimer,
    name: &str,
    settings: &impl Serialize,
    stats: &PlanStats,
    warnings: Vec<String>,
) -> RunInfo {
    let mut r = timer.finish(
        &t.system().snapshot,
        name,
        settings,
        &b.info().backend_id(),
        b.precision().model,
        None,
        warnings,
    );
    r.settings["resident"] = serde_json::json!({"version":1,"device":b.info(),"precision":b.precision(),"plan_stats":stats});
    r
}
/// The system fields `indices` as trace requests.
fn system_requests<'t>(t: &'t SequentialTracer, indices: &[usize]) -> Vec<Request<'t>> {
    indices
        .iter()
        .map(|&index| {
            let f = &t.system().system().fields.fields[index];
            (index, [f.x, f.y], t.field(index))
        })
        .collect()
}
/// Trace every requested field at `wls` in shared device batches. A field
/// whose pupil fails with a Trace/Numerical error (first in wavelength
/// order) is returned as that error; any other error fails the call.
#[allow(clippy::too_many_arguments)]
pub(super) fn trace_fields(
    service: &PupilTracer<'_>,
    t: &SequentialTracer,
    requests: &[Request<'_>],
    wls: &[(usize, f64)],
    settings: &PupilSettings,
    ctx: &JobContext,
    stats: &mut PlanStats,
    warnings: &mut Vec<String>,
) -> MkResult<Vec<MkResult<Field>>> {
    let pupils: Vec<_> = requests
        .iter()
        .flat_map(|&(_, _, f)| wls.iter().map(move |&(wl, _)| (f, wl)))
        .collect();
    let (traced, shared) = service.trace_many(&pupils, settings, ctx)?;
    accumulate(stats, &shared);
    let mut traced = traced.into_iter();
    let mut out = vec![];
    for &(index, value, _) in requests {
        let mut pupils = vec![];
        let mut error = None;
        for (&(wl, weight), r) in wls.iter().zip(traced.by_ref().take(wls.len())) {
            match r {
                Ok(r) => {
                    accumulate(stats, &r.stats);
                    warnings.extend(r.run_info.warnings);
                    pupils.push((wl, weight, r.value));
                }
                Err(e) => {
                    error.get_or_insert(e);
                }
            }
        }
        out.push(match error {
            Some(e) => Err(e),
            None => Ok(Field {
                index,
                value,
                reference: [0.; 2],
                pupils,
            }),
        });
    }
    // The chief reference is the primary wavelength's chief ray; when the
    // primary is not traced (or its chief fails) the first pupil's reference.
    let primary = t.system().primary_wavelength;
    let missing: Vec<_> = requests
        .iter()
        .zip(&out)
        .filter(|(_, f)| {
            settings.reference == ReferencePoint::Chief
                && f.as_ref()
                    .is_ok_and(|f| f.pupils.iter().all(|(wl, _, _)| *wl != primary))
        })
        .map(|(r, _)| r.2)
        .collect();
    let mut chiefs = if missing.is_empty() {
        vec![]
    } else {
        service.chief_points(&missing, &ctx.subrange(1., 1.), stats)?
    }
    .into_iter();
    for f in out.iter_mut().flatten() {
        let first = f.pupils[0].2.info.reference_point_mm;
        f.reference = match settings.reference {
            ReferencePoint::Chief => match f.pupils.iter().find(|(wl, _, _)| *wl == primary) {
                Some((_, _, p)) => [p.info.chief_point_mm[0], p.info.chief_point_mm[1]],
                None => chiefs.next().unwrap().unwrap_or([first[0], first[1]]),
            },
            ReferencePoint::Centroid => {
                let mut v = [0.; 2];
                for (_, w, p) in &f.pupils {
                    for (j, x) in v.iter_mut().enumerate() {
                        *x += w * p.info.reference_point_mm[j];
                    }
                }
                v
            }
        };
        for (_, _, p) in &mut f.pupils {
            p.pupil.image_offset_mm = [
                p.info.reference_point_mm[0] - f.reference[0],
                p.info.reference_point_mm[1] - f.reference[1],
            ];
        }
    }
    Ok(out)
}

/// One wavefront map's statistics and arrays.
struct MapData {
    stats: WavefrontStatistics,
    opd: Vec<f64>,
    amplitude: Vec<f64>,
}
/// Wavefront statistics and arrays of a host pupil, by the reference.
fn host_map(p: &PupilFunction) -> MapData {
    let s0 = p.opd_stats(false);
    let s1 = p.opd_stats(true);
    MapData {
        stats: WavefrontStatistics {
            rms: s0.rms,
            pv: s0.pv,
            rms_tilt_removed: s1.rms,
            pv_tilt_removed: s1.pv,
            strehl: p.strehl_at_reference(),
            tilt_condition_inf: f64::NAN,
        },
        opd: p
            .opd_waves()
            .iter()
            .zip(p.coverage())
            .map(|(&w, &c)| if c > 0. { w } else { f64::NAN })
            .collect(),
        amplitude: p
            .amplitude()
            .iter()
            .zip(p.coverage())
            .map(|(&a, &c)| if c > 0. { a } else { 0. })
            .collect(),
    }
}
/// Wavefront statistics (device) and arrays (read in place) of a resident pupil.
fn device_map(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    rp: &ResidentPupil,
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<MapData> {
    let st = rp.statistics(b, t.system().snapshot.clone(), ctx)?;
    accumulate(stats, &st.stats);
    let opd = b.read_view(rp.opd.clone())?;
    let opd = opd.as_slice::<Df32>()?;
    let c = b.read_view(rp.coverage.view())?;
    let c = c.as_slice::<f32>()?;
    let amp = b.read_view(rp.amplitude.view())?;
    let amp = amp.as_slice::<f32>()?;
    let mut waves = Vec::with_capacity(rp.n * rp.n);
    let mut amplitudes = Vec::with_capacity(rp.n * rp.n);
    for k in 0..rp.n * rp.n {
        if k % 4096 == 0 {
            ctx.check()?;
        }
        waves.push(if c[k] > 0. { opd[k].to_f64() } else { f64::NAN });
        amplitudes.push(if c[k] > 0. { amp[k] as f64 } else { 0. });
    }
    Ok(MapData {
        stats: st.value,
        opd: waves,
        amplitude: amplitudes,
    })
}

/// A wavefront map's field, wavelength and pupil.
type MapItem<'f> = (&'f Field, usize, &'f ResidentTracedPupil);
/// A lattice fit `(set, residual, condition)` with its statistics and
/// warnings, or the Numerical error that skips the pupil.
type Fitted = MkResult<((crate::ZernikeSet, f64, f64), PlanStats, Vec<String>)>;
/// One wavefront map from its lattice fit `(set, residual, condition)` and
/// its statistics and arrays.
fn wavefront(
    (f, wl, p): MapItem<'_>,
    (set, residual, condition): (crate::ZernikeSet, f64, f64),
    data: MapData,
) -> MkResult<a::WavefrontMap> {
    let rp = &p.pupil;
    Ok(a::WavefrontMap {
        field: f.index,
        field_value: f.value,
        wavelength: wl,
        wavelength_um: rp.wavelength_um,
        exit_pupil: p.info.clone(),
        grid: rp.n,
        coordinates: (0..rp.n)
            .map(|i| (2 * i + 1) as f64 / rp.n as f64 - 1.)
            .collect(),
        opd_waves: data.opd,
        amplitude: data.amplitude,
        rms_waves: data.stats.rms,
        pv_waves: data.stats.pv,
        rms_waves_tilt_removed: data.stats.rms_tilt_removed,
        pv_waves_tilt_removed: data.stats.pv_tilt_removed,
        strehl_at_reference: data.stats.strehl,
        strehl_marechal: crate::energy::strehl_marechal(data.stats.rms_tilt_removed),
        seidel: set.seidel()?,
        zernike: set,
        zernike_residual_rms: residual,
        zernike_condition: condition,
        rim: match &p.mask {
            Some(crate::PupilMask::Polygons { loops }) => loops.clone(),
            _ => vec![],
        },
    })
}
/// Host rule of a wavefront map's statistics (work: pupil cells). The
/// reference's statistics of a host pupil took 1–25 ms against 2–47 ms for
/// the device reductions and readback, up to 1024² cells.
const MAP_HOST: split::HostRule = split::HostRule {
    limit: 1e7,
    ratio: 0.5,
};

/// Complete wavefront maps. Device-traced pupils use the device lattice QR
/// fit; host-traced pupils use the reference's fit of the same lattice
/// (bit-identical Zernike sets and Seidel terms); statistics and arrays are
/// device reductions of the resident pupil, or the reference's own where
/// the design places transforms on the host.
pub fn wavefront_map_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::WavefrontSettings,
    ctx: &JobContext,
) -> MkResult<a::WavefrontResult> {
    wavefront_map_shared(b, t, s, None, ctx)
}
/// [`wavefront_map_job`] sharing host lattices through `memo`.
pub fn wavefront_map_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::WavefrontSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<a::WavefrontResult> {
    let timer = RunTimer::start();
    let basis = ZernikeBasis::new(s.zernike_ordering, s.zernike_terms)?;
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo, ctx)?;
    let fields = src.fields(
        &system_requests(t, &idx),
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.7),
        &mut stats,
        &mut warnings,
    )?;
    let items: Vec<_> = fields
        .iter()
        .flat_map(|f| f.pupils.iter().map(move |(wl, _, p)| (f, *wl, p)))
        .collect();
    let total = items.len().max(1);
    let fit_ctx = ctx.subrange(0.7, 1.);
    let range = |k: usize| fit_ctx.subrange(k as f64 / total as f64, (k + 1) as f64 / total as f64);
    // First the lattice fit (the reference's on host lattices, the device QR
    // otherwise), then the statistics and arrays (device reductions, or the
    // reference's on host pupils where the design places them). A pupil
    // too sparse for the fit is skipped like an unusable field.
    let (results, ran) = split::schedule_with(
        src.design,
        items,
        |k, item: (&Field, usize, &ResidentTracedPupil)| {
            let (f, wl, p) = item;
            let sub = range(k);
            let mut stats = PlanStats::default();
            let mut warnings = vec![];
            let fit = match &p.host {
                Some(h) => {
                    crate::analysis::wavefront::fit_samples(&h.lattice.samples, &basis, &sub).map(
                        |fit| {
                            warnings.extend(fit.warnings.iter().map(ToString::to_string));
                            (fit.set, fit.residual_rms, fit.condition)
                        },
                    )
                }
                None => fit_resident_zernike(
                    b,
                    t.system().snapshot.clone(),
                    p.nodes.view(),
                    &basis,
                    &sub.subrange(0., 0.5),
                )
                .map(|fit| {
                    accumulate(&mut stats, &fit.stats);
                    warnings.extend(fit.run_info.warnings);
                    let v = fit.value;
                    (v.set, v.residual_rms, v.condition_inf)
                }),
            };
            let warnings: Vec<_> = warnings
                .into_iter()
                .chain(p.pupil.warnings.iter().map(ToString::to_string))
                .map(|w| format!("field {}, wavelength {wl}: {w}", f.index))
                .collect();
            Ok(match fit {
                Ok(fit) => (item, Ok((fit, stats, warnings))),
                Err(e @ MkError::Numerical(_)) => (item, Err(e)),
                Err(e) => return Err(e),
            })
        },
        |k, (item, fit)| {
            Ok(match fit {
                Ok((fit, mut stats, warnings)) => {
                    let data =
                        device_map(b, t, &item.2.pupil, &range(k).subrange(0.5, 1.), &mut stats)?;
                    Ok((wavefront(item, fit, data)?, stats, warnings))
                }
                Err(e) => Err((item.0.index, item.1, e)),
            })
        },
        src.reference(
            &|_, (item, fit): (MapItem<'_>, Fitted)| {
                Ok(match fit {
                    Ok((fit, stats, warnings)) => {
                        let pupil = item.2.host_pupil().ok_or_else(|| {
                            MkError::Unsupported("host statistics without a host pupil".into())
                        })?;
                        Ok((wavefront(item, fit, host_map(pupil))?, stats, warnings))
                    }
                    Err(e) => Err((item.0.index, item.1, e)),
                })
            },
            &|(item, _)| (item.2.pupil.n * item.2.pupil.n) as f64,
            MAP_HOST,
        ),
    );
    src.record(ran);
    let mut maps = vec![];
    let mut first_error = None;
    for r in results {
        match r? {
            Ok((map, s, w)) => {
                accumulate(&mut stats, &s);
                warnings.extend(w);
                maps.push(map);
            }
            Err((field, wl, e)) => {
                warnings.push(format!("field {field}, wavelength {wl} skipped: {e}"));
                first_error.get_or_insert(e);
            }
        }
    }
    if !src.design.host_lattice() {
        warnings.push("Zernike condition is the equilibrated R infinity-norm estimate".into());
    }
    if maps.is_empty()
        && let Some(e) = first_error
    {
        return Err(e);
    }
    ctx.report(1., "wavefront map");
    Ok(a::WavefrontResult {
        maps,
        run_info: src.info(timer, "wavefront_map", s, &stats, warnings),
    })
}
/// Automatic-grid native FFT/MFT or Huygens PSFs with resident spectral sums.
pub fn psf_analysis_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::PsfAnalysisSettings,
    ctx: &JobContext,
) -> MkResult<a::PsfResult> {
    psf_analysis_shared(b, t, s, None, ctx)
}
/// [`psf_analysis_job`] sharing host lattices and pupils through `memo`.
pub fn psf_analysis_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::PsfAnalysisSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<a::PsfResult> {
    let timer = RunTimer::start();
    if s.grid_size > 4096 {
        return Err(MkError::InvalidParam("PSF grid size".into()));
    }
    if s.method == a::PsfMethod::Huygens && s.grid_size != 0 {
        a::psf::check_huygens_grid(s.grid_size)?;
    }
    a::check_pos("padding", s.padding)?;
    if let Some(dx) = s.spacing_mm {
        a::check_pos("PSF spacing", dx)?;
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo, ctx)?;
    // Every decision of a PSF is per field: pipelined runs trace field by field.
    let stream = src.design.pipelined();
    let items = src.items(
        system_requests(t, &idx),
        stream,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.4),
        &mut stats,
        &mut warnings,
    )?;
    let total = items.len().max(1);
    let range = |index: usize| {
        if stream {
            ctx.subrange(
                index as f64 / total as f64,
                (index + 1) as f64 / total as f64,
            )
        } else {
            ctx.subrange(
                0.4 + 0.6 * index as f64 / total as f64,
                0.4 + 0.6 * (index + 1) as f64 / total as f64,
            )
        }
    };
    let (results, ran) = split::schedule_with(
        src.design,
        items,
        |index, item| {
            let sub = range(index);
            let mut stats = PlanStats::default();
            let mut warnings = vec![];
            let mut f = match src.ready(
                item,
                &wls,
                &s.pupil,
                &sub.subrange(0., 0.2),
                &mut stats,
                &mut warnings,
            )? {
                Ok(f) => f,
                Err((index, error)) => return Ok(Staged::Skipped { index, error }),
            };
            let spacing = s.spacing_mm.unwrap_or_else(|| {
                f.pupils
                    .iter()
                    .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
                    .fold(f64::INFINITY, f64::min)
            });
            let grid = src.image_grid(
                &f,
                s.grid_size,
                spacing,
                s.method,
                &sub.subrange(0.2, 0.25),
                &mut stats,
            )?;
            warnings.extend(src.ensure_window(
                &mut f,
                grid as f64 * spacing,
                &s.pupil,
                &sub.subrange(0.25, 0.35),
                &mut stats,
            )?);
            Ok(Staged::Done {
                value: (f, grid, spacing),
                stats,
                warnings,
            })
        },
        |index, staged| {
            let sub = range(index);
            staged.then(|(f, grid, spacing), stats, warnings| {
                let mut psfs = vec![];
                let groups: Vec<_> = if s.polychromatic {
                    vec![((f.pupils.len() == 1).then(|| f.pupils[0].0), f.weighted())]
                } else {
                    f.pupils
                        .iter()
                        .map(|(wl, _, p)| (Some(*wl), vec![(&p.pupil, 1.)]))
                        .collect()
                };
                for (j, (wl, group)) in groups.iter().enumerate() {
                    let r = polychromatic_psf(
                        b,
                        t.system().snapshot.clone(),
                        group,
                        &crate::PolyPsfSettings {
                            grid_size: grid,
                            spacing_mm: spacing,
                            normalization: s.normalization,
                            transform: PsfTransform::Matrix,
                        },
                        s.method == a::PsfMethod::Huygens,
                        &sub.subrange(
                            0.35 + 0.55 * j as f64 / groups.len() as f64,
                            0.35 + 0.55 * (j + 1) as f64 / groups.len() as f64,
                        ),
                    )?;
                    accumulate(stats, &r.stats);
                    let mut psf = r.value.finish(b, &sub.subrange(0.9, 1.))?;
                    if group.len() == 1 {
                        psf.wavelength_um = Some(group[0].0.wavelength_um);
                    }
                    warnings.extend(psf.warnings.iter().map(ToString::to_string));
                    psfs.push(field_psf(&f, *wl, psf));
                }
                Ok(psfs)
            })
        },
        src.reference(
            &|index, staged: Staged<(Field, usize, f64)>| {
                let sub = range(index);
                staged.then(|(f, grid, spacing), _, warnings| {
                    let host = host_pupils(&f)?;
                    let groups: Vec<_> = if s.polychromatic {
                        vec![((f.pupils.len() == 1).then(|| f.pupils[0].0), host)]
                    } else {
                        f.pupils
                            .iter()
                            .zip(host)
                            .map(|((wl, _, _), (p, _))| (Some(*wl), vec![(p, 1.)]))
                            .collect()
                    };
                    let mut psfs = vec![];
                    for (j, (wl, group)) in groups.iter().enumerate() {
                        let psf = a::psf::compute_psf(
                            group,
                            s.method,
                            grid,
                            spacing,
                            s.normalization,
                            &sub.subrange(
                                0.35 + 0.65 * j as f64 / groups.len() as f64,
                                0.35 + 0.65 * (j + 1) as f64 / groups.len() as f64,
                            ),
                        )?;
                        warnings.extend(psf.warnings.iter().map(ToString::to_string));
                        psfs.push(field_psf(&f, *wl, psf));
                    }
                    Ok(psfs)
                })
            },
            &|staged| match staged {
                Staged::Done {
                    value: (f, grid, _),
                    ..
                } => transform_work(f, *grid),
                Staged::Skipped { .. } => 0.,
            },
            PSF_HOST,
        ),
    );
    src.record(ran);
    let psfs = gather(results, &mut stats, &mut warnings)?
        .into_iter()
        .flatten()
        .collect();
    ctx.report(1., "PSF");
    Ok(a::PsfResult {
        psfs,
        run_info: src.info(timer, "psf", s, &stats, warnings),
    })
}

#[allow(clippy::too_many_arguments)]
fn curve(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    pupils: &[(&ResidentPupil, f64)],
    freqs: &[f64],
    orientation: crate::MtfOrientation,
    method: a::MtfAnalysisMethod,
    huygens: a::HuygensGrid,
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<crate::MtfCurve> {
    let vectors: Vec<_> = [orientation.tangential, orientation.sagittal()]
        .iter()
        .flat_map(|v| freqs.iter().map(move |f| [v[0] * f, v[1] * f]))
        .collect();
    let mut curve_warnings: Vec<_> = pupils
        .iter()
        .flat_map(|(p, _)| p.warnings.clone())
        .collect();
    let optical = if method == a::MtfAnalysisMethod::Huygens {
        let dx = pupils
            .iter()
            .map(|(p, _)| p.lambda_f() / huygens.padding)
            .fold(f64::INFINITY, f64::min);
        let r = polychromatic_psf(
            b,
            t.system().snapshot.clone(),
            pupils,
            &crate::PolyPsfSettings {
                grid_size: huygens.grid_size,
                spacing_mm: dx,
                normalization: PsfNormalization::UnitEnergy,
                transform: PsfTransform::Matrix,
            },
            true,
            &ctx.subrange(0., 0.7),
        )?;
        accumulate(stats, &r.stats);
        curve_warnings = r.value.warnings.clone();
        let nyquist = 0.5 / dx;
        if let Some(&f) = freqs.iter().find(|&&f| f > nyquist * (1. + 1e-12)) {
            curve_warnings.push(WaveWarning::FrequencyBeyondSampling {
                frequency: f,
                limit: nyquist,
            });
        }
        r.value.otf(
            b,
            t.system().snapshot.clone(),
            &vectors,
            &ctx.subrange(0.7, 0.8),
        )?
    } else {
        polychromatic_otf_method(
            b,
            t.system().snapshot.clone(),
            pupils,
            &vectors,
            method == a::MtfAnalysisMethod::Fft,
            &ctx.subrange(0., 0.8),
        )?
    };
    accumulate(stats, &optical.stats);
    let ideal: Vec<_> = pupils
        .iter()
        .map(|(p, w)| {
            let mut ideal = p.without_aberrations(b, ctx)?;
            ideal.image_offset_mm = [0.; 2];
            Ok((ideal, *w))
        })
        .collect::<MkResult<_>>()?;
    let ideal_refs: Vec<_> = ideal.iter().map(|(p, w)| (p, *w)).collect();
    let dl = polychromatic_otf_method(
        b,
        t.system().snapshot.clone(),
        &ideal_refs,
        &vectors,
        method != a::MtfAnalysisMethod::Direct,
        &ctx.subrange(0.8, 1.),
    )?;
    accumulate(stats, &dl.stats);
    let n = freqs.len();
    let magnitude = |v: &[f64; 2]| v[0].hypot(v[1]);
    Ok(crate::MtfCurve {
        frequencies: freqs.to_vec(),
        tangential: optical.value[..n].iter().map(magnitude).collect(),
        sagittal: optical.value[n..].iter().map(magnitude).collect(),
        diffraction_limit: dl.value[..n].iter().map(magnitude).collect(),
        cutoff: pupils
            .iter()
            .map(|(p, _)| 1. / p.lambda_f())
            .fold(0., f64::max),
        warnings: curve_warnings,
    })
}
/// Complete native direct/FFT/Huygens MTF with the unchanged automatic frequency rule.
pub fn mtf_analysis_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::MtfSettings,
    ctx: &JobContext,
) -> MkResult<a::WaveMtfResult> {
    mtf_analysis_shared(b, t, s, None, ctx)
}
/// [`mtf_analysis_job`] sharing host lattices and pupils through `memo`.
pub fn mtf_analysis_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::MtfSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<a::WaveMtfResult> {
    let timer = RunTimer::start();
    s.huygens.validate()?;
    if !(2..=100001).contains(&s.frequency_samples) {
        return Err(MkError::InvalidParam("MTF frequency count".into()));
    }
    if let Some(d) = s.detector {
        d.validate()?;
    }
    if let Some(f) = s.max_frequency {
        a::check_pos("maximum frequency", f)?;
    }
    if s.method == a::MtfAnalysisMethod::Direct {
        ResidentPupil::check_stencil_grid(s.pupil.grid)?;
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo, ctx)?;
    // The automatic frequency limit is the smallest over all fields; with an
    // explicit limit, pipelined runs trace field by field.
    let stream = src.design.pipelined() && s.max_frequency.is_some();
    let items = src.items(
        system_requests(t, &idx),
        stream,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.5),
        &mut stats,
        &mut warnings,
    )?;
    let maximum = if let Some(f) = s.max_frequency {
        f
    } else {
        let fields: Vec<_> = items
            .iter()
            .filter_map(|i| match i {
                Item::Traced(f) => Some(f),
                Item::Request(_) => None,
            })
            .collect();
        src.useful_limit(&fields, &ctx.subrange(0.5, 0.55), &mut stats)?
    };
    let frequencies = a::linspace(0., maximum, s.frequency_samples);
    let total = items.len().max(1);
    let range = |i: usize| {
        ctx.subrange(
            0.55 + 0.45 * i as f64 / total as f64,
            0.55 + 0.45 * (i + 1) as f64 / total as f64,
        )
    };
    let orientation = |f: &Field| {
        let point = t.field(f.index).point;
        crate::MtfOrientation::from_field(point.hx, point.hy)
    };
    let (results, ran) = split::schedule_with(
        src.design,
        items,
        |i, item| {
            let mut stats = PlanStats::default();
            let mut warnings = vec![];
            Ok(
                match src.ready(item, &wls, &s.pupil, &range(i), &mut stats, &mut warnings)? {
                    Ok(f) => Staged::Done {
                        value: f,
                        stats,
                        warnings,
                    },
                    Err((index, error)) => Staged::Skipped { index, error },
                },
            )
        },
        |i, staged| {
            let sub = range(i);
            staged.then(|f, stats, warnings| {
                let o = orientation(&f);
                let c = curve(
                    b,
                    t,
                    &f.weighted(),
                    &frequencies,
                    o,
                    s.method,
                    s.huygens,
                    &sub.subrange(0., 0.95),
                    stats,
                )?;
                warnings.extend(c.warnings.iter().map(ToString::to_string));
                let detector = if let Some(d) = s.detector {
                    let vectors: Vec<_> = [o.tangential, o.sagittal()]
                        .iter()
                        .flat_map(|v| frequencies.iter().map(move |f| [v[0] * f, v[1] * f]))
                        .collect();
                    let r = detector_mtf(
                        b,
                        t.system().snapshot.clone(),
                        &d,
                        &vectors,
                        &sub.subrange(0.95, 1.),
                    )?;
                    accumulate(stats, &r.stats);
                    Some(crate::detector::DetectorCurve {
                        tangential: r.value[..frequencies.len()].to_vec(),
                        sagittal: r.value[frequencies.len()..].to_vec(),
                        nyquist_tangential: d.nyquist_along(o.tangential),
                        nyquist_sagittal: d.nyquist_along(o.sagittal()),
                    })
                } else {
                    None
                };
                Ok(field_mtf(&f, o, c, detector))
            })
        },
        src.reference(
            &|i, staged: Staged<Field>| {
                let sub = range(i);
                staged.then(|f, _, warnings| {
                    let o = orientation(&f);
                    let c = a::mtf::field_mtf(
                        &o,
                        &host_pupils(&f)?,
                        &frequencies,
                        s.method,
                        &s.huygens,
                        &sub,
                    )?;
                    warnings.extend(c.warnings.iter().map(ToString::to_string));
                    let detector = s.detector.map(|d| d.curve(&frequencies, &o));
                    Ok(field_mtf(&f, o, c, detector))
                })
            },
            &|staged| match staged {
                Staged::Done { value, .. } => correlation_work(value, 2 * frequencies.len()),
                Staged::Skipped { .. } => 0.,
            },
            OTF_HOST,
        ),
    );
    src.record(ran);
    let out = gather(results, &mut stats, &mut warnings)?;
    ctx.report(1., "MTF");
    Ok(a::WaveMtfResult {
        frequencies,
        fields: out,
        nyquist: s.detector.map(|d| d.nyquist()),
        run_info: src.info(timer, "mtf", s, &stats, warnings),
    })
}
fn fixed_frequencies(
    src: &Source<'_>,
    fields: &[&Field],
    list: &[f64],
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<Vec<f64>> {
    if !list.is_empty() {
        if list.iter().any(|f| !f.is_finite() || *f < 0.) {
            return Err(MkError::InvalidParam(
                "frequencies must be finite and nonnegative".into(),
            ));
        }
        return Ok(list.to_vec());
    }
    let limit = src.useful_limit(fields, ctx, stats)?;
    let base = crate::sampling::round_frequency(limit / 4.);
    Ok(vec![base, 2. * base, 4. * base])
}
/// Native field sweep; unusable samples retain their positions as NaN.
pub fn mtf_vs_field_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::MtfVsFieldSettings,
    ctx: &JobContext,
) -> MkResult<a::MtfVsFieldResult> {
    mtf_vs_field_shared(b, t, s, None, ctx)
}
/// [`mtf_vs_field_job`] sharing host lattices and pupils through `memo`.
pub fn mtf_vs_field_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::MtfVsFieldSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<a::MtfVsFieldResult> {
    let timer = RunTimer::start();
    s.huygens.validate()?;
    if let Some(d) = s.detector {
        d.validate()?;
    }
    if !(1..=1001).contains(&s.field_samples) {
        return Err(MkError::InvalidParam("field sample count".into()));
    }
    let len = s.direction[0].hypot(s.direction[1]);
    a::check_pos("field direction", len)?;
    if s.method == a::MtfAnalysisMethod::Direct {
        ResidentPupil::check_stencil_grid(s.pupil.grid)?;
    }
    let o = crate::MtfOrientation {
        tangential: s.direction.map(|v| v / len),
    };
    let hs = a::linspace(0., 1., s.field_samples);
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let src = Source::new(b, t, memo, ctx)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let mut values = vec![];
    let mut prepared = vec![];
    for &h in &hs {
        let point = mklight_trace::FieldPoint::normalized(h * o.tangential[0], h * o.tangential[1]);
        values.push(a::field_value(t, &point));
        prepared.push(t.prepare_field(point));
    }
    let requests: Vec<_> = prepared
        .iter()
        .enumerate()
        .filter_map(|(i, f)| f.as_ref().ok().map(|f| (i, values[i], f)))
        .collect();
    let mut traced = src
        .trace(
            &requests,
            &wls,
            &s.pupil,
            &ctx.subrange(0., 0.6),
            &mut stats,
            &mut warnings,
        )?
        .into_iter();
    let mut per = vec![];
    let mut error = None;
    for (&h, f) in hs.iter().zip(prepared) {
        match f.and_then(|_| traced.next().unwrap()) {
            Ok(f) => per.push(Some(f)),
            Err(e @ (MkError::Trace(_) | MkError::Numerical(_))) => {
                warnings.push(format!("field {h:.3} skipped: {e}"));
                error.get_or_insert(e);
                per.push(None);
            }
            Err(e) => return Err(e),
        }
    }
    if per.iter().all(Option::is_none) {
        return Err(error.unwrap_or_else(|| MkError::Trace("no usable field".into())));
    }
    let usable: Vec<_> = per.iter().flatten().collect();
    let freqs = fixed_frequencies(
        &src,
        &usable,
        &s.frequencies,
        &ctx.subrange(0.6, 0.65),
        &mut stats,
    )?;
    let mut curves: Vec<_> = freqs
        .iter()
        .map(|&frequency| a::MtfFieldCurve {
            frequency,
            tangential: vec![f64::NAN; hs.len()],
            sagittal: vec![f64::NAN; hs.len()],
        })
        .collect();
    let detector = s.detector.map(|d| d.curve(&freqs, &o));
    // Field samples are independent chains, run side by side.
    let sampling: Vec<_> = per
        .iter()
        .map(|f| f.as_ref().map_or_else(Vec::new, Field::sampling))
        .collect();
    let range = |i: usize| {
        ctx.subrange(
            0.65 + 0.35 * i as f64 / hs.len() as f64,
            0.65 + 0.35 * (i + 1) as f64 / hs.len() as f64,
        )
    };
    let (computed, ran) = split::schedule_with(
        src.design,
        per,
        |_, f| Ok(f),
        |i, f| {
            let Some(f) = f else { return Ok(None) };
            let mut stats = PlanStats::default();
            let c = curve(
                b,
                t,
                &f.weighted(),
                &freqs,
                o,
                s.method,
                s.huygens,
                &range(i),
                &mut stats,
            )?;
            Ok(Some((c, stats)))
        },
        src.reference(
            &|i, f: Option<Field>| {
                let Some(f) = f else { return Ok(None) };
                let c = a::mtf::field_mtf(
                    &o,
                    &host_pupils(&f)?,
                    &freqs,
                    s.method,
                    &s.huygens,
                    &range(i),
                )?;
                Ok(Some((c, PlanStats::default())))
            },
            &|f| {
                f.as_ref()
                    .map_or(0., |f| correlation_work(f, 2 * freqs.len()))
            },
            OTF_HOST,
        ),
    );
    src.record(ran);
    for (i, r) in computed.into_iter().enumerate() {
        if let Some((c, s)) = r? {
            accumulate(&mut stats, &s);
            for (j, row) in curves.iter_mut().enumerate() {
                row.tangential[i] =
                    c.tangential[j] * detector.as_ref().map_or(1., |d| d.tangential[j]);
                row.sagittal[i] = c.sagittal[j] * detector.as_ref().map_or(1., |d| d.sagittal[j]);
            }
            warnings.extend(c.warnings.iter().map(ToString::to_string));
        }
    }
    ctx.report(1., "MTF vs field");
    Ok(a::MtfVsFieldResult {
        pupil_sampling: sampling,
        normalized_field: hs,
        field_values: values,
        frequencies: freqs,
        curves,
        run_info: src.info(timer, "mtf_vs_field", s, &stats, warnings),
    })
}
/// Native line-spread profiles from the resident polychromatic PSF.
pub fn line_spread_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::LineSpreadSettings,
    ctx: &JobContext,
) -> MkResult<a::LineSpreadResult> {
    line_spread_shared(b, t, s, None, ctx)
}
/// [`line_spread_job`] sharing host lattices and pupils through `memo`.
pub fn line_spread_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::LineSpreadSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<a::LineSpreadResult> {
    let timer = RunTimer::start();
    a::energy::check_grid(s.grid_size, s.padding, s.method)?;
    a::check_pos("padding", s.padding)?;
    if s.grid_size > 4096 {
        return Err(MkError::InvalidParam("PSF grid".into()));
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo, ctx)?;
    // Every decision is per field: pipelined runs trace field by field.
    let stream = src.design.pipelined();
    let items = src.items(
        system_requests(t, &idx),
        stream,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.5),
        &mut stats,
        &mut warnings,
    )?;
    let total = items.len().max(1);
    let range = |i: usize| {
        if stream {
            ctx.subrange(i as f64 / total as f64, (i + 1) as f64 / total as f64)
        } else {
            ctx.subrange(
                0.5 + 0.5 * i as f64 / total as f64,
                0.5 + 0.5 * (i + 1) as f64 / total as f64,
            )
        }
    };
    let (results, ran) = split::schedule_with(
        src.design,
        items,
        |i, item| {
            let sub = range(i);
            let mut stats = PlanStats::default();
            let mut warnings = vec![];
            let mut f = match src.ready(
                item,
                &wls,
                &s.pupil,
                &sub.subrange(0., 0.2),
                &mut stats,
                &mut warnings,
            )? {
                Ok(f) => f,
                Err((index, error)) => return Ok(Staged::Skipped { index, error }),
            };
            let dx = f
                .pupils
                .iter()
                .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
                .fold(f64::INFINITY, f64::min);
            let grid = src.image_grid(
                &f,
                s.grid_size,
                dx,
                s.method,
                &sub.subrange(0.2, 0.25),
                &mut stats,
            )?;
            src.ensure_window(
                &mut f,
                grid as f64 * dx,
                &s.pupil,
                &sub.subrange(0.25, 0.3),
                &mut stats,
            )?;
            Ok(Staged::Done {
                value: (f, grid, dx),
                stats,
                warnings,
            })
        },
        |i, staged| {
            let sub = range(i);
            staged.then(|(f, grid, dx), stats, warnings| {
                let r = polychromatic_psf(
                    b,
                    t.system().snapshot.clone(),
                    &f.weighted(),
                    &crate::PolyPsfSettings {
                        grid_size: grid,
                        spacing_mm: dx,
                        normalization: PsfNormalization::UnitEnergy,
                        transform: PsfTransform::Matrix,
                    },
                    s.method == a::PsfMethod::Huygens,
                    &sub.subrange(0.3, 0.9),
                )?;
                accumulate(stats, &r.stats);
                warnings.extend(r.value.warnings.iter().map(ToString::to_string));
                let r =
                    r.value
                        .line_spread(b, t.system().snapshot.clone(), &sub.subrange(0.9, 1.))?;
                accumulate(stats, &r.stats);
                let [x, y] = r.value;
                Ok(a::FieldLineSpread {
                    field: f.index,
                    field_value: f.value,
                    pupil_sampling: f.sampling(),
                    positions_mm: (0..grid)
                        .map(|i| (i as f64 - (grid / 2) as f64) * dx)
                        .collect(),
                    x,
                    y,
                })
            })
        },
        src.reference(
            &|i, staged: Staged<(Field, usize, f64)>| {
                let sub = range(i);
                staged.then(|(f, grid, dx), _, warnings| {
                    let psf = a::psf::compute_psf(
                        &host_pupils(&f)?,
                        s.method,
                        grid,
                        dx,
                        PsfNormalization::UnitEnergy,
                        &sub.subrange(0.3, 1.),
                    )?;
                    warnings.extend(psf.warnings.iter().map(ToString::to_string));
                    let lx = crate::energy::line_spread_function(&psf, crate::energy::LsfAxis::X);
                    let ly = crate::energy::line_spread_function(&psf, crate::energy::LsfAxis::Y);
                    Ok(a::FieldLineSpread {
                        field: f.index,
                        field_value: f.value,
                        pupil_sampling: f.sampling(),
                        positions_mm: lx.positions_mm,
                        x: lx.values,
                        y: ly.values,
                    })
                })
            },
            &|staged| match staged {
                Staged::Done {
                    value: (f, grid, _),
                    ..
                } => transform_work(f, *grid),
                Staged::Skipped { .. } => 0.,
            },
            PSF_HOST,
        ),
    );
    src.record(ran);
    let out = gather(results, &mut stats, &mut warnings)?;
    ctx.report(1., "line spread");
    Ok(a::LineSpreadResult {
        fields: out,
        run_info: src.info(timer, "line_spread", s, &stats, warnings),
    })
}
/// Native band-limited diffraction energy, preserving dense quantile sampling.
pub fn diffraction_encircled_energy_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::EncircledEnergySettings,
    ctx: &JobContext,
) -> MkResult<a::WaveEncircledEnergyResult> {
    diffraction_encircled_energy_shared(b, t, s, None, ctx)
}
/// [`diffraction_encircled_energy_job`] sharing host lattices and pupils
/// through `memo`.
pub fn diffraction_encircled_energy_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::EncircledEnergySettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<a::WaveEncircledEnergyResult> {
    let timer = RunTimer::start();
    a::energy::check_grid(s.grid_size, s.padding, s.method)?;
    a::check_pos("padding", s.padding)?;
    if !(2..=10001).contains(&s.samples) || s.grid_size > 4096 {
        return Err(MkError::InvalidParam("energy sampling".into()));
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo, ctx)?;
    let per = src.fields(
        &system_requests(t, &idx),
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.3),
        &mut stats,
        &mut warnings,
    )?;
    let spacing = per
        .iter()
        .flat_map(|f| &f.pupils)
        .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
        .fold(f64::INFINITY, f64::min);
    let mut grid = s.grid_size;
    for r in concurrent(&per, |_, f| {
        let mut st = PlanStats::default();
        src.image_grid(
            f,
            s.grid_size,
            spacing,
            s.method,
            &ctx.subrange(0.3, 0.35),
            &mut st,
        )
        .map(|g| (g, st))
    }) {
        let (g, st) = r?;
        accumulate(&mut stats, &st);
        grid = grid.max(g);
    }
    let max = s.max_radius_mm.unwrap_or(0.5 * grid as f64 * spacing);
    a::check_pos("maximum radius", max)?;
    let radius = a::linspace(0., max, s.samples);
    let dense = a::linspace(0., max, 8 * s.samples);
    let mut all = radius.clone();
    all.extend_from_slice(&dense);
    let total = per.len().max(1);
    let range = |i: usize| {
        ctx.subrange(
            0.35 + 0.45 * i as f64 / total as f64,
            0.35 + 0.45 * (i + 1) as f64 / total as f64,
        )
    };
    let settings = crate::PolyPsfSettings {
        grid_size: grid,
        spacing_mm: spacing,
        normalization: PsfNormalization::UnitEnergy,
        transform: PsfTransform::Matrix,
    };
    let (results, ran) = split::schedule(
        src.design,
        per,
        |i, mut f| {
            let mut stats = PlanStats::default();
            let warnings = src.ensure_window(
                &mut f,
                grid as f64 * spacing,
                &s.pupil,
                &range(i).subrange(0., 0.05),
                &mut stats,
            )?;
            Ok(Staged::Done {
                value: f,
                stats,
                warnings,
            })
        },
        |i, staged| {
            let sub = range(i);
            staged.then(|f, stats, warnings| {
                let r = polychromatic_psf(
                    b,
                    t.system().snapshot.clone(),
                    &f.weighted(),
                    &settings,
                    s.method == a::PsfMethod::Huygens,
                    &sub.subrange(0.05, 0.3),
                )?;
                accumulate(stats, &r.stats);
                let psf = r.value;
                let metrics =
                    psf.metrics(b, t.system().snapshot.clone(), &sub.subrange(0.3, 0.32))?;
                accumulate(stats, &metrics.stats);
                let center_mm = match s.center {
                    crate::EnergyCenter::Reference => [0.; 2],
                    crate::EnergyCenter::Centroid => metrics.value.centroid_mm,
                    crate::EnergyCenter::Peak => [
                        ((metrics.value.peak_index % grid) as f64 - (grid / 2) as f64) * spacing,
                        ((metrics.value.peak_index / grid) as f64 - (grid / 2) as f64) * spacing,
                    ],
                    crate::EnergyCenter::Point { x, y } => [x, y],
                };
                let sq = psf.energy(
                    b,
                    t.system().snapshot.clone(),
                    &radius,
                    crate::EnergyCenter::Point {
                        x: center_mm[0],
                        y: center_mm[1],
                    },
                    true,
                    &sub.subrange(0.32, 0.42),
                )?;
                accumulate(stats, &sq.stats);
                let clean: Vec<_> = f
                    .pupils
                    .iter()
                    .map(|(_, w, p)| Ok((p.pupil.without_aberrations(b, &sub)?, *w)))
                    .collect::<MkResult<_>>()?;
                let clean: Vec<_> = clean.iter().map(|(p, w)| (p, *w)).collect();
                let dl = polychromatic_psf(
                    b,
                    t.system().snapshot.clone(),
                    &clean,
                    &settings,
                    false,
                    &sub.subrange(0.7, 0.85),
                )?;
                accumulate(stats, &dl.stats);
                if metrics.value.energy_in_grid < a::MIN_ENERGY_IN_GRID {
                    warnings.push(
                        WaveWarning::PsfTruncated {
                            energy_in_grid: metrics.value.energy_in_grid,
                        }
                        .to_string(),
                    );
                }
                warnings.extend(psf.warnings.iter().map(ToString::to_string));
                Ok((
                    f.index,
                    f.value,
                    f.sampling(),
                    center_mm,
                    sq.value,
                    psf,
                    dl.value,
                ))
            })
        },
    );
    src.record(ran);
    let pending = gather(results, &mut stats, &mut warnings)?;
    let psfs: Vec<_> = pending
        .iter()
        .flat_map(|(_, _, _, center, _, psf, dl)| {
            [
                (
                    psf,
                    crate::EnergyCenter::Point {
                        x: center[0],
                        y: center[1],
                    },
                ),
                (dl, crate::EnergyCenter::Centroid),
            ]
        })
        .collect();
    let curves = encircled_energy_batch(
        b,
        t.system().snapshot.clone(),
        &psfs,
        &all,
        &ctx.subrange(0.8, 1.),
    )?;
    accumulate(&mut stats, &curves.stats);
    let out = pending
        .into_iter()
        .zip(curves.value.chunks(2))
        .map(
            |((field, field_value, pupil_sampling, center_mm, ensquared, _, _), energy)| {
                let (enc, fine) = energy[0].split_at(radius.len());
                a::FieldEnergy {
                    field,
                    field_value,
                    pupil_sampling,
                    grid_size: grid,
                    spacing_mm: spacing,
                    center_mm,
                    encircled: enc.to_vec(),
                    ensquared,
                    diffraction_limit: energy[1][..radius.len()].to_vec(),
                    r50_mm: a::energy::quantile(&dense, fine, 0.5),
                    r80_mm: a::energy::quantile(&dense, fine, 0.8),
                    r90_mm: a::energy::quantile(&dense, fine, 0.9),
                }
            },
        )
        .collect();
    ctx.report(1., "encircled energy");
    Ok(a::WaveEncircledEnergyResult {
        radius_mm: radius,
        fields: out,
        run_info: src.info(timer, "diffraction_encircled_energy", s, &stats, warnings),
    })
}
/// Native focus sweeps with shared correlation stencils and bounded FFT/Huygens batches.
pub fn through_focus_mtf_analysis_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::ThroughFocusSettings,
    ctx: &JobContext,
) -> MkResult<a::ThroughFocusResult> {
    through_focus_mtf_analysis_shared(b, t, s, None, ctx)
}
/// [`through_focus_mtf_analysis_job`] sharing host lattices and pupils
/// through `memo`.
pub fn through_focus_mtf_analysis_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &a::ThroughFocusSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<a::ThroughFocusResult> {
    let timer = RunTimer::start();
    s.huygens.validate()?;
    if let Some(d) = s.detector {
        d.validate()?;
    }
    if !(1..=1001).contains(&s.focus_samples) {
        return Err(MkError::InvalidParam("focus samples".into()));
    }
    if let Some(r) = s.focus_range_mm {
        a::check_pos("focus range", r)?;
    }
    if s.method == a::MtfAnalysisMethod::Direct {
        ResidentPupil::check_stencil_grid(s.pupil.grid)?;
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo, ctx)?;
    let per = src.fields(
        &system_requests(t, &idx),
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.3),
        &mut stats,
        &mut warnings,
    )?;
    let frequencies = fixed_frequencies(
        &src,
        &per.iter().collect::<Vec<_>>(),
        &s.frequencies,
        &ctx.subrange(0.3, 0.32),
        &mut stats,
    )?;
    let half = if let Some(r) = s.focus_range_mm {
        r
    } else {
        // max over pupils of max(8 λ N², 4 N · RMS blur)
        let mut half = 0f64;
        let blurs = concurrent(&per, |_, f| {
            let mut st = PlanStats::default();
            src.blurs(f, &ctx.subrange(0.32, 0.35), &mut st)
                .map(|b| (b, st))
        });
        for (f, r) in per.iter().zip(blurs) {
            let (blurs, st) = r?;
            accumulate(&mut stats, &st);
            for ((_, _, p), (_, rms)) in f.pupils.iter().zip(blurs) {
                let n = p.pupil.geometry.working_f_number();
                half = half.max((8. * p.pupil.wavelength_um * 1e-3 * n * n).max(4. * n * rms));
            }
        }
        half
    };
    let focus = if s.focus_samples == 1 {
        vec![0.]
    } else {
        a::linspace(-half, half, s.focus_samples)
    };
    let count = per.len().max(1);
    let range = |i: usize| {
        ctx.subrange(
            0.35 + 0.65 * i as f64 / count as f64,
            0.35 + 0.65 * (i + 1) as f64 / count as f64,
        )
    };
    let (results, ran) = split::schedule_with(
        src.design,
        per,
        |i, mut field| {
            let sub = range(i);
            let mut stats = PlanStats::default();
            let mut warnings = vec![];
            if s.pupil.grid == 0 {
                warnings.extend(refine_for_focus(
                    &src,
                    &mut field,
                    half,
                    s.pupil.supersample,
                    &sub.subrange(0., 0.2),
                    &mut stats,
                )?);
            }
            Ok(Staged::Done {
                value: field,
                stats,
                warnings,
            })
        },
        |i, staged| {
            let sub = range(i);
            staged.then(|field, stats, warnings| {
                let point = t.field(field.index).point;
                let o = crate::MtfOrientation::from_field(point.hx, point.hy);
                let vectors: Vec<_> = [o.tangential, o.sagittal()]
                    .iter()
                    .flat_map(|v| frequencies.iter().map(move |f| [v[0] * f, v[1] * f]))
                    .collect();
                let r = focus_otf(
                    b,
                    t.system().snapshot.clone(),
                    &field.weighted(),
                    &focus,
                    &vectors,
                    s.method,
                    s.huygens,
                    &sub.subrange(0.2, 0.98),
                )?;
                accumulate(stats, &r.stats);
                warnings.extend(r.run_info.warnings);
                let detector = s.detector.map(|d| d.curve(&frequencies, &o));
                let nf = frequencies.len();
                let curves = frequencies
                    .iter()
                    .enumerate()
                    .map(|(q, &frequency)| a::ThroughFocusCurve {
                        frequency,
                        tangential: (0..focus.len())
                            .map(|z| {
                                let v = r.value[z * 2 * nf + q];
                                v[0].hypot(v[1]) * detector.as_ref().map_or(1., |d| d.tangential[q])
                            })
                            .collect(),
                        sagittal: (0..focus.len())
                            .map(|z| {
                                let v = r.value[z * 2 * nf + nf + q];
                                v[0].hypot(v[1]) * detector.as_ref().map_or(1., |d| d.sagittal[q])
                            })
                            .collect(),
                    })
                    .collect();
                sub.report(1., "focus result");
                Ok(a::ThroughFocusField {
                    field: field.index,
                    field_value: field.value,
                    pupil_sampling: field.sampling(),
                    curves,
                })
            })
        },
        src.reference(
            &|i, staged: Staged<Field>| {
                let sub = range(i).subrange(0.2, 1.);
                staged.then(|field, _, warnings| {
                    let point = t.field(field.index).point;
                    let o = crate::MtfOrientation::from_field(point.hx, point.hy);
                    let host = host_pupils(&field)?;
                    // The reference's planes (`crate::analysis::mtf`).
                    let planes = if s.method == a::MtfAnalysisMethod::Direct {
                        crate::mtf::polychromatic_focus_mtf(&host, &focus, &frequencies, &o, &sub)?
                    } else {
                        let nz = focus.len() as f64;
                        focus
                            .iter()
                            .enumerate()
                            .map(|(z, &dz)| {
                                let shifted: Vec<PupilFunction> = host
                                    .iter()
                                    .map(|(p, _)| {
                                        p.with_focus_shift(dz / p.geometry().chief_ray[2])
                                    })
                                    .collect();
                                let weighted: Vec<_> = shifted
                                    .iter()
                                    .zip(&host)
                                    .map(|(p, (_, w))| (p, *w))
                                    .collect();
                                a::mtf::field_mtf(
                                    &o,
                                    &weighted,
                                    &frequencies,
                                    s.method,
                                    &s.huygens,
                                    &sub.subrange(z as f64 / nz, (z + 1) as f64 / nz),
                                )
                            })
                            .collect::<MkResult<Vec<_>>>()?
                    };
                    for c in &planes {
                        for w in &c.warnings {
                            let w = w.to_string();
                            if !warnings.contains(&w) {
                                warnings.push(w);
                            }
                        }
                    }
                    let detector = s.detector.map(|d| d.curve(&frequencies, &o));
                    let curves = frequencies
                        .iter()
                        .enumerate()
                        .map(|(q, &frequency)| a::ThroughFocusCurve {
                            frequency,
                            tangential: planes
                                .iter()
                                .map(|c| {
                                    c.tangential[q]
                                        * detector.as_ref().map_or(1., |d| d.tangential[q])
                                })
                                .collect(),
                            sagittal: planes
                                .iter()
                                .map(|c| {
                                    c.sagittal[q] * detector.as_ref().map_or(1., |d| d.sagittal[q])
                                })
                                .collect(),
                        })
                        .collect();
                    Ok(a::ThroughFocusField {
                        field: field.index,
                        field_value: field.value,
                        pupil_sampling: field.sampling(),
                        curves,
                    })
                })
            },
            &|staged| match staged {
                Staged::Done { value, .. } => {
                    correlation_work(value, 2 * frequencies.len() * focus.len())
                }
                Staged::Skipped { .. } => 0.,
            },
            FOCUS_HOST,
        ),
    );
    src.record(ran);
    let out = gather(results, &mut stats, &mut warnings)?;
    ctx.report(1., "through-focus MTF");
    Ok(a::ThroughFocusResult {
        focus_mm: focus,
        frequencies,
        fields: out,
        run_info: src.info(timer, "through_focus_mtf", s, &stats, warnings),
    })
}
/// Refine automatic pupils against the phase steps at both focus ends
/// (`±half` along the image normal): the reference's rule on host pupils,
/// device focus batches otherwise.
fn refine_for_focus(
    src: &Source<'_>,
    field: &mut Field,
    half: f64,
    supersample: usize,
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<Vec<String>> {
    let mut grids = vec![];
    for (_, _, p) in &field.pupils {
        let step = match p.host_pupil() {
            Some(h) => [-half, half]
                .map(|dz| {
                    h.with_focus_shift(dz / h.geometry().chief_ray[2])
                        .max_phase_step()
                })
                .into_iter()
                .fold(0., f64::max),
            None => {
                let shifts = [
                    -half / p.pupil.geometry.chief_ray[2],
                    half / p.pupil.geometry.chief_ray[2],
                ];
                let r =
                    p.pupil
                        .focus_batch(src.b, src.snapshot(), &shifts, &ctx.subrange(0., 0.2))?;
                accumulate(stats, &r.stats);
                r.value
                    .iter()
                    .map(|p| p.max_phase_step)
                    .fold(0f64, f64::max)
            }
        };
        grids.push(
            ((p.pupil.n as f64 * step / 0.35)
                .max(p.pupil.n as f64)
                .min(1024.)
                .ceil() as usize)
                .next_power_of_two(),
        );
    }
    src.refine(field, &grids, supersample, &ctx.subrange(0.2, 1.), stats)
}
