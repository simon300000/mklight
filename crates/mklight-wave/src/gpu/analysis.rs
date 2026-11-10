//! Native result adapters. Routing and qualification policy belong to the engine.
//!
//! The host traces and resamples every pupil with the reference's code
//! ([`Source`]) and makes the sampling decisions that read them (image
//! windows, refinements, automatic frequencies and focus ranges) with the
//! reference's own rules; the device runs the transforms. Per-field host
//! preparation overlaps the device transforms of other fields
//! ([`split::schedule`]).
use super::split;
use super::traced::accumulate;
use super::*;
use crate::analysis::{self as a, ItemProgress, PupilSampling};
use crate::{PupilSettings, ReferencePoint};
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
    /// The reference pupils with their weights (every pupil is resampled
    /// on the host).
    pub(super) fn host_weighted(&self) -> Vec<(&PupilFunction, f64)> {
        self.pupils
            .iter()
            .map(|(_, w, p)| {
                let h = p.host_pupil().expect("host-resampled pupil");
                (h, *w)
            })
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
    /// A field prepared on the host, with its warnings.
    pub(super) fn prepared(value: T, warnings: Vec<String>) -> Self {
        Staged::Done {
            value,
            stats: PlanStats::default(),
            warnings,
        }
    }
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
/// Trace and Numerical errors make one pupil (and its field) unusable;
/// every other error fails the whole run.
fn per_pupil(e: MkError) -> MkResult<MkError> {
    match e {
        MkError::Trace(_) | MkError::Numerical(_) => Ok(e),
        e => Err(e),
    }
}

/// A run's pupils, traced and resampled on the host.
pub(super) struct Source<'a> {
    pub(super) b: &'a dyn ComputeBackend,
    pub(super) t: &'a SequentialTracer,
    pub(super) memo: Option<&'a dyn Memo>,
}
impl<'a> Source<'a> {
    pub(super) fn new(
        b: &'a dyn ComputeBackend,
        t: &'a SequentialTracer,
        memo: Option<&'a dyn Memo>,
    ) -> Self {
        Self { b, t, memo }
    }
    /// Trace every request at `wls` with the reference's lattices: every
    /// (field, wavelength) lattice is traced on the host at once (shared
    /// through the memo), then resampled on the host and written into shared
    /// allocations. Reference points follow the reference
    /// (`crate::analysis`): the primary chief ray, or the weighted centroid.
    /// A field whose pupil fails with a Trace/Numerical error (first in
    /// wavelength order) is returned as that error; any other error fails
    /// the call.
    pub(super) fn trace(
        &self,
        requests: &[Request<'_>],
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
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
        let place_ctx = ctx.subrange(0.8, 1.);
        let place = |k: usize, r: MkResult<(mklight_core::memo::NodeKey, std::sync::Arc<_>)>| {
            let sub = place_ctx.subrange(k as f64 / total, (k + 1) as f64 / total);
            match r.and_then(|(key, lattice)| {
                split::place(self.b, key, lattice, settings, self.memo, &sub)
            }) {
                Ok(p) => Ok(Ok(p)),
                Err(e) => per_pupil(e).map(Err),
            }
        };
        let placed: Vec<MkResult<MkResult<split::Placed>>> = lattices
            .into_par_iter()
            .enumerate()
            .map(|(k, r)| place(k, r))
            .collect();
        let mut placed = placed.into_iter();
        let primary = self.t.system().primary_wavelength;
        let mut out = vec![];
        for &(index, value, field) in requests {
            let mut pupils = vec![];
            let mut error = None;
            for (&(wl, weight), r) in wls.iter().zip(placed.by_ref().take(wls.len())) {
                match r? {
                    Ok((pupil, notes)) => {
                        warnings.extend(notes);
                        pupils.push((wl, weight, pupil));
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
    pub(super) fn fields(
        &self,
        requests: &[Request<'_>],
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
        warnings: &mut Vec<String>,
    ) -> MkResult<Vec<Field>> {
        let traced = self.trace(requests, wls, settings, ctx, warnings)?;
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
    pub(super) fn items<'f>(
        &self,
        requests: Vec<Request<'f>>,
        stream: bool,
        wls: &[(usize, f64)],
        settings: &PupilSettings,
        ctx: &JobContext,
        warnings: &mut Vec<String>,
    ) -> MkResult<Vec<Item<'f>>> {
        if stream {
            settings.validate()?;
            return Ok(requests.into_iter().map(Item::Request).collect());
        }
        Ok(self
            .fields(&requests, wls, settings, ctx, warnings)?
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
        warnings: &mut Vec<String>,
    ) -> MkResult<Result<Field, (usize, MkError)>> {
        match item {
            Item::Traced(f) => Ok(Ok(f)),
            Item::Request(r) => Ok(self
                .trace(&[r], wls, settings, ctx, warnings)?
                .pop()
                .expect("one field")
                .map_err(|e| (r.0, e))),
        }
    }
    /// Blur estimates `(extent, rms)` of every pupil of `f` (the
    /// reference's rule).
    pub(super) fn blurs(&self, f: &Field) -> Vec<(f64, f64)> {
        f.host_weighted()
            .into_iter()
            .map(|(h, _)| crate::sampling::blur(h))
            .collect()
    }
    /// The automatic image grid of `field` at `spacing` (or `requested`),
    /// by the reference's rule.
    pub(super) fn image_grid(
        &self,
        field: &Field,
        requested: usize,
        spacing: f64,
        method: a::PsfMethod,
    ) -> usize {
        if requested != 0 {
            return requested;
        }
        a::psf::image_grid(0, method, &field.host_weighted(), spacing)
    }
    /// Refine automatic pupils so their alias-free field holds `window`
    /// (explicit grids are kept), in parallel on the host. Returns the
    /// resampling warnings.
    pub(super) fn ensure_window(
        &self,
        field: &mut Field,
        window: f64,
        s: &PupilSettings,
        ctx: &JobContext,
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
        self.refine(field, &grids, s.supersample, ctx)
    }
    /// Resample every pupil of `field` to its grid of `grids`, in parallel
    /// on the host.
    pub(super) fn refine(
        &self,
        field: &mut Field,
        grids: &[usize],
        supersample: usize,
        ctx: &JobContext,
    ) -> MkResult<Vec<String>> {
        let refined: Vec<MkResult<Vec<String>>> = field
            .pupils
            .par_iter_mut()
            .zip(grids)
            .map(|((_, _, p), &n)| p.refine(self.b, n, supersample, self.memo, ctx))
            .collect();
        let mut warnings = vec![];
        for r in refined {
            warnings.extend(r?);
        }
        Ok(warnings)
    }
    /// The phase-contrast frequency limit of a field's pupils (the
    /// reference's rule).
    pub(super) fn useful(&self, field: &Field) -> f64 {
        crate::sampling::useful_frequency(&field.host_weighted())
    }
    /// The smallest [`Self::useful`] of `fields`, evaluated side by side.
    pub(super) fn useful_limit(&self, fields: &[&Field]) -> f64 {
        let each: Vec<f64> = fields.par_iter().map(|f| self.useful(f)).collect();
        each.into_iter().fold(f64::INFINITY, f64::min)
    }
    pub(super) fn info(
        &self,
        timer: RunTimer,
        name: &str,
        settings: &impl Serialize,
        stats: &PlanStats,
        warnings: Vec<String>,
    ) -> RunInfo {
        info(self.b, self.t, timer, name, settings, stats, warnings)
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
/// One wavefront map's statistics and arrays.
struct MapData {
    stats: WavefrontStatistics,
    opd: Vec<f64>,
    amplitude: Vec<f64>,
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
/// Complete wavefront maps. The Zernike fit is the reference's fit of the
/// host lattice (bit-identical Zernike sets and Seidel terms); statistics
/// and arrays are device reductions of the resident pupil.
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
    let basis = a::wavefront::fit_basis(s)?;
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo);
    let fields = src.fields(
        &system_requests(t, &idx),
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.7),
        &mut warnings,
    )?;
    let items: Vec<_> = fields
        .iter()
        .flat_map(|f| f.pupils.iter().map(move |(wl, _, p)| (f, *wl, p)))
        .collect();
    let progress = ItemProgress::new(ctx.subrange(0.7, 1.), items.len());
    // First the lattice fit (the reference's, on the host lattice), then the
    // statistics and arrays (device reductions). A pupil too sparse for the
    // fit is skipped like an unusable field.
    let results = split::schedule(
        items,
        |_, item: (&Field, usize, &ResidentTracedPupil)| {
            let (f, wl, p) = item;
            let sub = progress.item();
            let mut warnings = vec![];
            let h = p.host.as_ref().expect("host-traced pupil");
            let fit = crate::analysis::wavefront::fit_samples(&h.lattice.samples, &basis, &sub)
                .map(|fit| {
                    warnings.extend(fit.warnings.iter().map(ToString::to_string));
                    (fit.set, fit.residual_rms, fit.condition)
                });
            let warnings: Vec<_> = warnings
                .into_iter()
                .chain(p.pupil.warnings.iter().map(ToString::to_string))
                .map(|w| format!("field {}, wavelength {wl}: {w}", f.index))
                .collect();
            Ok(match fit {
                Ok(fit) => (item, Ok((fit, warnings))),
                Err(e @ MkError::Numerical(_)) => (item, Err(e)),
                Err(e) => return Err(e),
            })
        },
        |_, (item, fit)| {
            let map = match fit {
                Ok((fit, warnings)) => {
                    let mut stats = PlanStats::default();
                    let sub = progress.item().subrange(0.5, 1.);
                    let data = device_map(b, t, &item.2.pupil, &sub, &mut stats)?;
                    Ok((wavefront(item, fit, data)?, stats, warnings))
                }
                Err(e) => Err((item.0.index, item.1, e)),
            };
            progress.finish();
            Ok(map)
        },
    );
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
    let src = Source::new(b, t, memo);
    // Every decision of a PSF is per field: trace field by field.
    let items = src.items(
        system_requests(t, &idx),
        true,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.4),
        &mut warnings,
    )?;
    let progress = ItemProgress::new(ctx.clone(), items.len());
    let results = split::schedule(
        items,
        |_, item| {
            let sub = progress.item();
            let mut warnings = vec![];
            let mut f =
                match src.ready(item, &wls, &s.pupil, &sub.subrange(0., 0.2), &mut warnings)? {
                    Ok(f) => f,
                    Err((index, error)) => return Ok(Staged::Skipped { index, error }),
                };
            let spacing = s.spacing_mm.unwrap_or_else(|| {
                f.pupils
                    .iter()
                    .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
                    .fold(f64::INFINITY, f64::min)
            });
            let grid = src.image_grid(&f, s.grid_size, spacing, s.method);
            warnings.extend(src.ensure_window(
                &mut f,
                grid as f64 * spacing,
                &s.pupil,
                &sub.subrange(0.25, 0.35),
            )?);
            Ok(Staged::prepared((f, grid, spacing), warnings))
        },
        |_, staged| {
            let sub = progress.item();
            let psfs = staged.then(|(f, grid, spacing), stats, warnings| {
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
            });
            progress.finished(psfs)
        },
    );
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
    let src = Source::new(b, t, memo);
    // The automatic frequency limit is the smallest over all fields; with an
    // explicit limit, trace field by field.
    let stream = s.max_frequency.is_some();
    let items = src.items(
        system_requests(t, &idx),
        stream,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.5),
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
        let limit = src.useful_limit(&fields);
        a::check_pos("maximum frequency", limit)?;
        limit
    };
    let frequencies = a::linspace(0., maximum, s.frequency_samples);
    let progress = ItemProgress::new(ctx.subrange(0.55, 1.), items.len());
    let orientation = |f: &Field| {
        let point = t.field(f.index).point;
        crate::MtfOrientation::from_field(point.hx, point.hy)
    };
    let results = split::schedule(
        items,
        |_, item| {
            let mut warnings = vec![];
            Ok(
                match src.ready(item, &wls, &s.pupil, &progress.item(), &mut warnings)? {
                    Ok(f) => Staged::prepared(f, warnings),
                    Err((index, error)) => Staged::Skipped { index, error },
                },
            )
        },
        |_, staged| {
            let sub = progress.item();
            let field = staged.then(|f, stats, warnings| {
                let o = orientation(&f);
                let c = curve(
                    b,
                    t,
                    &f.weighted(),
                    &frequencies,
                    o,
                    s.method,
                    s.huygens,
                    &sub,
                    stats,
                )?;
                warnings.extend(c.warnings.iter().map(ToString::to_string));
                let detector = s.detector.map(|d| d.curve(&frequencies, &o));
                Ok(field_mtf(&f, o, c, detector))
            });
            progress.finished(field)
        },
    );
    let out = gather(results, &mut stats, &mut warnings)?;
    ctx.report(1., "MTF");
    Ok(a::WaveMtfResult {
        frequencies,
        fields: out,
        nyquist: s.detector.map(|d| d.nyquist()),
        run_info: src.info(timer, "mtf", s, &stats, warnings),
    })
}
fn fixed_frequencies(src: &Source<'_>, fields: &[&Field], list: &[f64]) -> MkResult<Vec<f64>> {
    if !list.is_empty() {
        if list.iter().any(|f| !f.is_finite() || *f < 0.) {
            return Err(MkError::InvalidParam(
                "frequencies must be finite and nonnegative".into(),
            ));
        }
        return Ok(list.to_vec());
    }
    let limit = src.useful_limit(fields);
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
    let src = Source::new(b, t, memo);
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
    let freqs = fixed_frequencies(&src, &usable, &s.frequencies)?;
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
    let progress = ItemProgress::new(ctx.subrange(0.65, 1.), hs.len());
    let computed = split::schedule(
        per,
        |_, f| Ok(f),
        |_, f| {
            let Some(f) = f else {
                progress.finish();
                return Ok(None);
            };
            let mut stats = PlanStats::default();
            let c = curve(
                b,
                t,
                &f.weighted(),
                &freqs,
                o,
                s.method,
                s.huygens,
                &progress.item(),
                &mut stats,
            )?;
            progress.finish();
            Ok(Some((c, stats)))
        },
    );
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
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo);
    // Every decision is per field: trace field by field.
    let items = src.items(
        system_requests(t, &idx),
        true,
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.5),
        &mut warnings,
    )?;
    let progress = ItemProgress::new(ctx.clone(), items.len());
    let results = split::schedule(
        items,
        |_, item| {
            let sub = progress.item();
            let mut warnings = vec![];
            let mut f =
                match src.ready(item, &wls, &s.pupil, &sub.subrange(0., 0.2), &mut warnings)? {
                    Ok(f) => f,
                    Err((index, error)) => return Ok(Staged::Skipped { index, error }),
                };
            let dx = f
                .pupils
                .iter()
                .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
                .fold(f64::INFINITY, f64::min);
            let grid = src.image_grid(&f, s.grid_size, dx, s.method);
            warnings.extend(src.ensure_window(
                &mut f,
                grid as f64 * dx,
                &s.pupil,
                &sub.subrange(0.25, 0.3),
            )?);
            Ok(Staged::prepared((f, grid, dx), warnings))
        },
        |_, staged| {
            let sub = progress.item();
            let field = staged.then(|(f, grid, dx), stats, warnings| {
                let r = polychromatic_psf(
                    b,
                    t.system().snapshot.clone(),
                    &f.weighted(),
                    &crate::PolyPsfSettings {
                        grid_size: grid,
                        spacing_mm: dx,
                        normalization: PsfNormalization::UnitEnergy,
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
            });
            progress.finished(field)
        },
    );
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
    if !(2..=10001).contains(&s.samples) {
        return Err(MkError::InvalidParam("energy sampling".into()));
    }
    let idx = a::fields(t, &s.fields)?;
    let wls = a::wavelengths(t, &s.wavelengths)?;
    let mut stats = PlanStats::default();
    let mut warnings = t.warnings().to_vec();
    let src = Source::new(b, t, memo);
    let per = src.fields(
        &system_requests(t, &idx),
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.3),
        &mut warnings,
    )?;
    let spacing = per
        .iter()
        .flat_map(|f| &f.pupils)
        .map(|(_, _, p)| p.pupil.lambda_f() / s.padding)
        .fold(f64::INFINITY, f64::min);
    let grid = per
        .par_iter()
        .map(|f| src.image_grid(f, s.grid_size, spacing, s.method))
        .max()
        .unwrap_or(s.grid_size);
    let max = s.max_radius_mm.unwrap_or(0.5 * grid as f64 * spacing);
    a::check_pos("maximum radius", max)?;
    let radius = a::linspace(0., max, s.samples);
    let dense = a::linspace(0., max, 8 * s.samples);
    let mut all = radius.clone();
    all.extend_from_slice(&dense);
    let progress = ItemProgress::new(ctx.subrange(0.35, 0.8), per.len());
    let settings = crate::PolyPsfSettings {
        grid_size: grid,
        spacing_mm: spacing,
        normalization: PsfNormalization::UnitEnergy,
    };
    let results = split::schedule(
        per,
        |_, mut f| {
            let warnings = src.ensure_window(
                &mut f,
                grid as f64 * spacing,
                &s.pupil,
                &progress.item().subrange(0., 0.05),
            )?;
            Ok(Staged::prepared(f, warnings))
        },
        |_, staged| {
            let sub = progress.item();
            let field = staged.then(|f, stats, warnings| {
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
                let sq = psf.ensquared(
                    b,
                    t.system().snapshot.clone(),
                    &radius,
                    center_mm,
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
            });
            progress.finished(field)
        },
    );
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
    let src = Source::new(b, t, memo);
    let per = src.fields(
        &system_requests(t, &idx),
        &wls,
        &s.pupil,
        &ctx.subrange(0., 0.3),
        &mut warnings,
    )?;
    let frequencies = fixed_frequencies(&src, &per.iter().collect::<Vec<_>>(), &s.frequencies)?;
    let half = if let Some(r) = s.focus_range_mm {
        r
    } else {
        // max over pupils of max(8 λ N², 4 N · RMS blur)
        let mut half = 0f64;
        let blurs: Vec<_> = per.par_iter().map(|f| src.blurs(f)).collect();
        for (f, blurs) in per.iter().zip(blurs) {
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
    let progress = ItemProgress::new(ctx.subrange(0.35, 1.), per.len());
    let results = split::schedule(
        per,
        |_, mut field| {
            let sub = progress.item();
            let mut warnings = vec![];
            if s.pupil.grid == 0 {
                warnings.extend(refine_for_focus(
                    &src,
                    &mut field,
                    half,
                    s.pupil.supersample,
                    &sub.subrange(0., 0.2),
                )?);
            }
            Ok(Staged::prepared(field, warnings))
        },
        |_, staged| {
            let sub = progress.item();
            let field = staged.then(|field, stats, warnings| {
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
            });
            progress.finished(field)
        },
    );
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
/// (`±half` along the image normal), by the reference's rule.
fn refine_for_focus(
    src: &Source<'_>,
    field: &mut Field,
    half: f64,
    supersample: usize,
    ctx: &JobContext,
) -> MkResult<Vec<String>> {
    let mut grids = vec![];
    for (_, _, p) in &field.pupils {
        let h = p.host_pupil().expect("host-resampled pupil");
        let step = [-half, half]
            .map(|dz| {
                h.with_focus_shift(dz / h.geometry().chief_ray[2])
                    .max_phase_step()
            })
            .into_iter()
            .fold(0., f64::max);
        grids.push(
            ((p.pupil.n as f64 * step / 0.35)
                .max(p.pupil.n as f64)
                .min(1024.)
                .ceil() as usize)
                .next_power_of_two(),
        );
    }
    src.refine(field, &grids, supersample, &ctx.subrange(0.2, 1.))
}
