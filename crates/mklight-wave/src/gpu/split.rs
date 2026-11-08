//! Work-split designs of the native wave analyses: which stages run on the
//! host (the CPU f64 reference code, in parallel) and which on the device,
//! selected once per run through `mklight_core::split` (study key [`KEY`])
//! and named in the run's `RunInfo` (`settings.work_split`).
//!
//! | design | host | device |
//! |---|---|---|
//! | `resident` | seed tables, chief rows | aiming, tracing, rim, placement, resampling, sampling decisions, transforms |
//! | `cpu-lattice` | lattice and rim aiming, tracing and placement | resampling and refinements, sampling decisions, transforms |
//! | `cpu-pupil` | lattice, resampling, refinements, sampling decisions | transforms |
//! | `pipelined` | as `cpu-pupil`, overlapped (dataflow): the device transforms each field as soon as the host has prepared it | transforms |
//! | `co-op` (default) | as `pipelined`, and one host worker also transforms fields with the reference's own code, taking them from the other end of the queue when that pays | transforms of the remaining fields |
//! | `by-size` | as `pipelined`, and transforms of small fields (a measured work rule per analysis) with the reference's own code | transforms of large fields |
//! | `cpu-aim` | aiming of lattice, chief, saddle and rim probes | tracing, rim survival, placement, resampling, decisions, transforms |
//!
//! Host lattices are the reference's own ([`crate::traced::trace_lattice`])
//! and are shared with the CPU analyses through the engine memo
//! (`crate::shared`); host-resampled pupils are the reference's
//! [`PupilFunction`]s, written straight into shared allocations.
use super::*;
use crate::traced::{PupilLattice, PupilSettings};
use mklight_core::memo::{Memo, NodeKey};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

/// The `mklight_core::split` study key of the wave analyses.
pub const KEY: &str = "wave";

/// One division of the native wave analyses between host and device
/// (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Design {
    Resident,
    CpuLattice,
    CpuPupil,
    Pipelined,
    Cooperative,
    BySize,
    CpuAim,
}

impl Design {
    /// The design that runs without a selection.
    pub const DEFAULT: Design = Design::Cooperative;
    pub const ALL: [Design; 7] = [
        Design::Resident,
        Design::CpuLattice,
        Design::CpuPupil,
        Design::Pipelined,
        Design::Cooperative,
        Design::BySize,
        Design::CpuAim,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Design::Resident => "resident",
            Design::CpuLattice => "cpu-lattice",
            Design::CpuPupil => "cpu-pupil",
            Design::Pipelined => "pipelined",
            Design::Cooperative => "co-op",
            Design::BySize => "by-size",
            Design::CpuAim => "cpu-aim",
        }
    }
    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|d| d.name() == name)
    }
    /// The selected design (an unknown name selects the default; the
    /// requested name is kept in [`Self::evidence`]).
    pub fn selected() -> Self {
        mklight_core::split::variant(KEY)
            .and_then(|v| Self::from_name(&v))
            .unwrap_or(Self::DEFAULT)
    }
    /// The host traces the pupil lattices with the reference tracer.
    pub fn host_lattice(self) -> bool {
        !matches!(self, Design::Resident | Design::CpuAim)
    }
    /// The host resamples pupils and makes the sampling decisions that
    /// read them (image windows, refinements, automatic frequencies).
    pub fn host_pupil(self) -> bool {
        self.host_lattice() && self != Design::CpuLattice
    }
    /// Host preparation of fields overlaps the device transforms of others
    /// (dataflow, [`schedule`]); per-field analyses trace field by field.
    pub fn pipelined(self) -> bool {
        matches!(
            self,
            Design::Pipelined | Design::Cooperative | Design::BySize
        )
    }
    /// `RunInfo.settings.work_split`.
    pub fn evidence(self) -> serde_json::Value {
        serde_json::json!({
            "key": KEY,
            "design": self.name(),
            "requested": mklight_core::split::variant(KEY),
        })
    }
}

/// Where one scheduled item's transform ran (`"device"` or `"host"`), and
/// its host-preparation and transform times (evidence for the run's
/// `work_split`).
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub(crate) struct Ran {
    pub on: &'static str,
    /// The analysis' work measure of the item ([`Reference::work`]).
    pub work: f64,
    pub prepare_ms: f64,
    pub transform_ms: f64,
}

/// The host (reference) part of a pupil the host traced.
#[derive(Clone, Debug)]
pub struct HostPupil {
    /// The reference lattice the resident nodes were written from.
    pub lattice: Arc<PupilLattice>,
    /// Its `pupil_lattice` node key (`crate::shared`, f64 family).
    pub key: NodeKey,
    /// The reference pupil the resident pupil was written from, at the
    /// resident grid (host-resampling designs).
    pub pupil: Option<Arc<PupilFunction>>,
}

/// Resident copies of a host lattice's nodes (DF `[ny, nx, 4]`: u, v,
/// unwrapped OPD, amplitude) and survival (u32 `[ny, nx]`), written in
/// place; the layout of [`from_structured`].
pub(crate) fn lattice_nodes(
    b: &dyn ComputeBackend,
    lattice: &PupilLattice,
) -> MkResult<(BufferHandle, BufferHandle)> {
    let rays = &lattice.samples;
    let nodes = b.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [rays.ny, rays.nx, 4]),
        &mut |mut a| {
            for (row, s) in a.as_mut_slice::<Df32>()?.chunks_mut(4).zip(&rays.samples) {
                let v = match s {
                    Some(r) => [r.u, r.v, r.opd_waves, r.amplitude],
                    None => [0.; 4],
                };
                for (d, v) in row.iter_mut().zip(v) {
                    *d = Df32::from_f64(v);
                    if !d.is_finite() {
                        return Err(MkError::InvalidParam("non-finite pupil sample".into()));
                    }
                }
            }
            Ok(())
        },
    )?;
    let valid = b.fill_buffer(
        BufferDesc::new(ElementType::U32, [rays.ny, rays.nx]),
        &mut |mut a| {
            for (v, s) in a.as_mut_slice::<u32>()?.iter_mut().zip(&rays.samples) {
                *v = s.is_some() as u32;
            }
            Ok(())
        },
    )?;
    Ok((nodes, valid))
}

/// A host pupil written straight into shared allocations (amplitude and
/// coverage f32, OPD double-float), as [`ResidentPupil::from_pupil`] but
/// identified by `identity` instead of a content digest, and converted in
/// parallel.
pub(crate) fn upload_pupil(
    b: &dyn ComputeBackend,
    p: &PupilFunction,
    identity: String,
    ctx: &JobContext,
) -> MkResult<ResidentPupil> {
    ctx.check()?;
    const CHUNK: usize = 1 << 14;
    let n = p.n();
    let f32s = |src: &[f64]| {
        b.fill_buffer(BufferDesc::new(ElementType::F32, [n, n]), &mut |mut a| {
            a.as_mut_slice::<f32>()?
                .par_chunks_mut(CHUNK)
                .zip(src.par_chunks(CHUNK))
                .for_each(|(d, s)| {
                    for (d, s) in d.iter_mut().zip(s) {
                        *d = *s as f32;
                    }
                });
            Ok(())
        })
    };
    let amplitude = f32s(p.amplitude())?;
    let coverage = f32s(p.coverage())?;
    let opd = b.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [n, n]),
        &mut |mut a| {
            a.as_mut_slice::<Df32>()?
                .par_chunks_mut(CHUNK)
                .zip(p.opd_waves().par_chunks(CHUNK))
                .for_each(|(d, s)| {
                    for (d, s) in d.iter_mut().zip(s) {
                        *d = Df32::from_f64(*s);
                    }
                });
            Ok(())
        },
    )?;
    Ok(ResidentPupil {
        input_hash: identity,
        covered_samples: p.coverage().iter().filter(|&&c| c > 0.).count(),
        clear_area: p.coverage().iter().sum::<f64>() * p.spacing().powi(2),
        max_phase_step: p.max_phase_step(),
        amplitude,
        coverage,
        opd: opd.view(),
        n,
        wavelength_um: p.wavelength_um(),
        geometry: *p.geometry(),
        image_offset_mm: p.image_offset_mm(),
        warnings: p.sampling_warnings(),
    })
}

fn identity(key: &NodeKey, n: usize, supersample: usize) -> String {
    let mut s: String = key.iter().map(|b| format!("{b:02x}")).collect();
    s += &format!(":{n}:{supersample}");
    s
}

/// A pupil traced on the host as a resident traced pupil.
pub(crate) struct Placed {
    pub value: ResidentTracedPupil,
    pub stats: PlanStats,
    pub warnings: Vec<String>,
}

/// A resident pupil with the host pupil it was written from (host
/// resampling), the device statistics and the resampling warnings.
type Resampled = (
    ResidentPupil,
    Option<Arc<PupilFunction>>,
    PlanStats,
    Vec<String>,
);

/// The resident pupil of a host `lattice` (key `key`) at grid `n`:
/// resampled on the host through `memo` (`host_resample`) and written into
/// shared allocations, or resampled on the device from its resident nodes.
#[allow(clippy::too_many_arguments)]
fn resident_at(
    b: &dyn ComputeBackend,
    snapshot: &Snapshot,
    lattice: &Arc<PupilLattice>,
    key: &NodeKey,
    nodes: &BufferHandle,
    valid: &BufferHandle,
    n: usize,
    supersample: usize,
    host_resample: bool,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<Resampled> {
    let id = identity(key, n, supersample);
    if host_resample {
        let r = crate::shared::resampled(lattice, key, n, supersample, memo, ctx)?;
        let pupil = Arc::new(r.0.clone());
        let resident = upload_pupil(b, &pupil, id, ctx)?;
        return Ok((resident, Some(pupil), PlanStats::default(), r.1.clone()));
    }
    let r = from_resident_lattice(
        b,
        snapshot.clone(),
        nodes.view(),
        valid.view(),
        id,
        n,
        lattice.info.wavelength_um,
        lattice.geometry,
        lattice.mask.as_ref(),
        supersample,
        ctx,
    )?;
    Ok((r.value, None, r.stats, r.run_info.warnings))
}

/// Place a host-traced lattice: its nodes become resident, and its pupil
/// at the lattice's grid ([`PupilLattice::grid_for`]) is resampled on the
/// host or on the device.
#[allow(clippy::too_many_arguments)]
pub(crate) fn place(
    b: &dyn ComputeBackend,
    snapshot: &Snapshot,
    key: NodeKey,
    lattice: Arc<PupilLattice>,
    settings: &PupilSettings,
    host_resample: bool,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<Placed> {
    let (nodes, valid) = lattice_nodes(b, &lattice)?;
    let n = lattice.grid_for(settings);
    let (pupil, host, stats, resample_warnings) = resident_at(
        b,
        snapshot,
        &lattice,
        &key,
        &nodes,
        &valid,
        n,
        settings.supersample,
        host_resample,
        memo,
        ctx,
    )?;
    let mut warnings = lattice.notes();
    warnings.extend(resample_warnings);
    // Without a rim mask the clear area is measured on the resampled grid
    // (`PupilLattice::assemble`).
    let mut info = lattice.info.clone();
    if lattice.mask.is_none() {
        info.clear_area = pupil.clear_area;
        let sin_e = info.sin_max * (info.clear_area / std::f64::consts::PI).sqrt();
        info.effective_f_number = 0.5 / (info.image_index * sin_e);
    }
    Ok(Placed {
        value: ResidentTracedPupil {
            pupil,
            info,
            nodes,
            valid,
            mask: lattice.mask.clone(),
            entrance_rim: lattice.entrance_rim.clone(),
            key,
            host: Some(HostPupil {
                lattice,
                key,
                pupil: host,
            }),
        },
        stats,
        warnings,
    })
}

impl ResidentTracedPupil {
    /// The pupil at grid `n` (keeping its image offset): resampled on the
    /// host through `memo` when the host resampled it before, on the device
    /// otherwise. Returns the device statistics and resampling warnings.
    pub(crate) fn refine(
        &mut self,
        b: &dyn ComputeBackend,
        snapshot: Snapshot,
        n: usize,
        supersample: usize,
        memo: Option<&dyn Memo>,
        ctx: &JobContext,
    ) -> MkResult<(PlanStats, Vec<String>)> {
        if n == self.pupil.n {
            return Ok((PlanStats::default(), vec![]));
        }
        match &mut self.host {
            Some(h) if h.pupil.is_some() => {
                let offset = self.pupil.image_offset_mm;
                let (pupil, host, stats, warnings) = resident_at(
                    b,
                    &snapshot,
                    &h.lattice,
                    &h.key,
                    &self.nodes,
                    &self.valid,
                    n,
                    supersample,
                    true,
                    memo,
                    ctx,
                )?;
                self.pupil = pupil;
                h.pupil = host;
                self.set_image_offset(offset);
                Ok((stats, warnings))
            }
            _ => {
                let r = self.resample(b, snapshot, n, supersample, ctx)?;
                Ok((r.stats, r.run_info.warnings))
            }
        }
    }
    /// The host f64 pupil the resident pupil was written from.
    pub fn host_pupil(&self) -> Option<&PupilFunction> {
        self.host.as_ref()?.pupil.as_deref()
    }
    /// Place the pupil (resident and host copies) at `offset` from the
    /// field's common reference point: the reference's sampling rules read
    /// the offset (lateral colour widens the image window).
    pub(crate) fn set_image_offset(&mut self, offset: [f64; 2]) {
        self.pupil.image_offset_mm = offset;
        if let Some(p) = self.host.as_mut().and_then(|h| h.pupil.as_mut()) {
            Arc::make_mut(p).set_image_offset_mm(offset);
        }
    }
}

/// The reference's own transform of a prepared item and the `by-size`
/// rule choosing it ([`schedule_with`]).
pub(crate) struct Reference<'s, P, R> {
    pub transform: &'s (dyn Fn(usize, P) -> MkResult<R> + Sync),
    /// The item's transform work (the analysis' own measure, recorded in
    /// [`Ran`]); `by-size` transforms items below `limit` on the host.
    pub work: &'s (dyn Fn(&P) -> f64 + Sync),
    pub limit: f64,
    /// Measured host/device transform-time ratio of one item: `co-op`'s
    /// host worker takes an item only when the devices would need at least
    /// as long to drain the rest of the queue.
    pub ratio: f64,
}

/// An analysis' measured host-transform rule ([`Reference::limit`],
/// [`Reference::ratio`]).
#[derive(Clone, Copy, Debug)]
pub(crate) struct HostRule {
    pub limit: f64,
    pub ratio: f64,
}

/// Scale of every `by-size` work limit (`MKLIGHT_WAVE_BY_SIZE`, default 1;
/// for calibration: 0 places every transform on the device, `inf` on the
/// host).
pub(crate) fn by_size_scale() -> f64 {
    static S: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
    *S.get_or_init(|| {
        std::env::var("MKLIGHT_WAVE_BY_SIZE")
            .ok()
            .and_then(|v| v.trim().parse().ok())
            .filter(|v: &f64| *v >= 0.)
            .unwrap_or(1.)
    })
}

/// [`schedule_with`] without host transforms.
pub(crate) fn schedule<T: Send, P: Send, R: Send>(
    design: Design,
    items: Vec<T>,
    host: impl Fn(usize, T) -> MkResult<P> + Sync,
    device: impl Fn(usize, P) -> MkResult<R> + Sync,
) -> (Vec<MkResult<R>>, Vec<Option<Ran>>) {
    schedule_with(design, items, host, device, None)
}

/// `host(i, item)` then a transform of the prepared item for every item,
/// results (and where each transform ran) in item order.
///
/// Chain designs run each item's host and device stages as one chain, up to
/// [`concurrency`] chains at once ([`concurrent`]), which bounds the
/// prepared items held at once. Pipelined designs (dataflow) start every
/// item's host stage at once on the host's thread pool (each parallel
/// inside) and queue each prepared item as soon as it is ready; up to
/// [`concurrency`] device workers take items from the front of the queue,
/// so the device transforms item k while the host still prepares others.
/// With a `reference`, `co-op` adds one host worker that transforms items
/// from the back of the queue with the reference's code when the queue is
/// long enough for it to pay ([`Reference::ratio`]; the two ends meet
/// wherever the faster side has taken more), and `by-size` transforms the
/// items its rule calls small with the reference's code.
pub(crate) fn schedule_with<T: Send, P: Send, R: Send>(
    design: Design,
    items: Vec<T>,
    host: impl Fn(usize, T) -> MkResult<P> + Sync,
    device: impl Fn(usize, P) -> MkResult<R> + Sync,
    reference: Option<Reference<'_, P, R>>,
) -> (Vec<MkResult<R>>, Vec<Option<Ran>>) {
    let count = items.len();
    let ran: Vec<Mutex<Option<Ran>>> = (0..count).map(|_| Mutex::new(None)).collect();
    let transform = |i: usize, p: P, prepare_ms: f64, on_host: bool| {
        let start = Instant::now();
        let work = reference.as_ref().map_or(f64::NAN, |c| (c.work)(&p));
        let (on, r) = match &reference {
            Some(c) if on_host || (design == Design::BySize && work < c.limit) => {
                ("host", (c.transform)(i, p))
            }
            _ => ("device", device(i, p)),
        };
        *ran[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(Ran {
            on,
            work,
            prepare_ms,
            transform_ms: start.elapsed().as_secs_f64() * 1e3,
        });
        r
    };
    let timed = |i: usize, t: T| {
        let start = Instant::now();
        let p = host(i, t);
        (p, start.elapsed().as_secs_f64() * 1e3)
    };
    let results = if !design.pipelined() {
        let items: Vec<_> = items.into_iter().map(|t| Mutex::new(Some(t))).collect();
        concurrent(&items, |i, t| {
            let t = t
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .expect("one run per item");
            let (p, ms) = timed(i, t);
            p.and_then(|p| transform(i, p, ms, false))
        })
    } else {
        let slots: Vec<Mutex<Option<MkResult<R>>>> = (0..count).map(|_| Mutex::new(None)).collect();
        let devices = concurrency().min(count);
        // Prepared items, whether every host stage has finished, and the
        // device workers free to take an item.
        let queue = (
            Mutex::new((VecDeque::<(usize, MkResult<P>, f64)>::new(), false, devices)),
            Condvar::new(),
        );
        // The host worker takes an item only when more are queued than free
        // device workers will take (it never keeps a free device waiting),
        // and only when the devices would need at least as long for the rest
        // of the queue as it needs for the item: with host/device ratio r,
        // at least (r − 1) items per device remain.
        let ratio = reference.as_ref().map_or(1., |c| c.ratio);
        let worth = |queued: usize| (queued - 1) as f64 >= (ratio - 1.).max(0.) * devices as f64;
        let worker = |back: bool| loop {
            let next = {
                let mut q = queue.0.lock().unwrap_or_else(|e| e.into_inner());
                loop {
                    let item = if !back {
                        q.0.pop_front()
                    } else if q.0.len() > q.2 && worth(q.0.len()) {
                        q.0.pop_back()
                    } else {
                        None
                    };
                    if item.is_some() {
                        q.2 -= usize::from(!back);
                        break item;
                    }
                    if q.1 && q.0.is_empty() {
                        break None;
                    }
                    q = queue.1.wait(q).unwrap_or_else(|e| e.into_inner());
                }
            };
            let Some((i, p, ms)) = next else { break };
            let r = p.and_then(|p| transform(i, p, ms, back));
            *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
            queue.0.lock().unwrap_or_else(|e| e.into_inner()).2 += usize::from(!back);
            queue.1.notify_all();
        };
        std::thread::scope(|s| {
            for _ in 0..devices {
                s.spawn(|| worker(false));
            }
            if design == Design::Cooperative && reference.is_some() {
                s.spawn(|| worker(true));
            }
            items.into_par_iter().enumerate().for_each(|(i, t)| {
                let (p, ms) = timed(i, t);
                queue
                    .0
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .0
                    .push_back((i, p, ms));
                queue.1.notify_all();
            });
            queue.0.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
            queue.1.notify_all();
        });
        slots
            .into_iter()
            .map(|m| {
                m.into_inner()
                    .unwrap_or_else(|e| e.into_inner())
                    .expect("every item ran")
            })
            .collect()
    };
    let ran = ran
        .into_iter()
        .map(|m| m.into_inner().unwrap_or_else(|e| e.into_inner()))
        .collect();
    (results, ran)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn design_names_round_trip() {
        for d in Design::ALL {
            assert_eq!(Design::from_name(d.name()), Some(d));
            assert!(!d.host_pupil() || d.host_lattice());
        }
        assert_eq!(Design::from_name("nonexistent"), None);
        assert!(Design::DEFAULT.host_lattice());
    }

    /// Every design yields every item's result in item order, with its
    /// errors, whichever worker transformed it.
    #[test]
    fn schedules_keep_item_order_and_errors() {
        for design in Design::ALL {
            let items: Vec<usize> = (0..23).collect();
            let host = |i: usize, t: usize| {
                assert_eq!(i, t);
                if t == 7 {
                    Err(MkError::Numerical("prepare".into()))
                } else {
                    Ok(10 * t)
                }
            };
            let device = |i: usize, p: usize| {
                std::thread::sleep(std::time::Duration::from_micros(200));
                if i == 11 {
                    Err(MkError::Numerical("device".into()))
                } else {
                    Ok(p + 1)
                }
            };
            let transform = |_: usize, p: usize| Ok(p + 1);
            let work = |p: &usize| *p as f64;
            let (results, ran) = schedule_with(
                design,
                items,
                host,
                device,
                Some(Reference {
                    transform: &transform,
                    work: &work,
                    limit: 100.,
                    ratio: 1.,
                }),
            );
            assert_eq!(results.len(), 23);
            for (i, (r, ran)) in results.iter().zip(&ran).enumerate() {
                if i == 7 {
                    assert!(r.is_err() && ran.is_none(), "{design:?}");
                    continue;
                }
                let ran = ran.expect("transformed");
                if i == 11 && ran.on == "device" {
                    assert!(r.is_err());
                } else {
                    assert_eq!(*r.as_ref().unwrap(), 10 * i + 1, "{design:?}");
                }
                // by-size: work below the limit runs on the host.
                if design == Design::BySize {
                    assert_eq!(ran.on == "host", 10 * i < 100);
                }
                if !matches!(design, Design::BySize | Design::Cooperative) {
                    assert_eq!(ran.on, "device");
                }
            }
        }
    }

    /// The co-op host worker never takes an item an idle device worker
    /// would take: a single item goes to the device.
    #[test]
    fn cooperative_host_worker_leaves_single_items_to_the_device() {
        let transform = |_: usize, p: usize| Ok(p);
        let work = |_: &usize| 0.;
        let (results, ran) = schedule_with(
            Design::Cooperative,
            vec![1usize],
            |_, t| Ok(t),
            |_, p| Ok(p),
            Some(Reference {
                transform: &transform,
                work: &work,
                limit: 0.,
                ratio: 1.,
            }),
        );
        assert_eq!(*results[0].as_ref().unwrap(), 1);
        assert_eq!(ran[0].unwrap().on, "device");
    }

    /// A host far slower than the device (ratio beyond any queue) never
    /// takes an item.
    #[test]
    fn cooperative_host_worker_weighs_the_host_device_ratio() {
        let transform = |_: usize, p: usize| Ok(p);
        let work = |_: &usize| 0.;
        let (_, ran) = schedule_with(
            Design::Cooperative,
            (0..64).collect(),
            |_, t: usize| Ok(t),
            |_, p| {
                std::thread::sleep(std::time::Duration::from_micros(500));
                Ok(p)
            },
            Some(Reference {
                transform: &transform,
                work: &work,
                limit: 0.,
                ratio: 1e9,
            }),
        );
        assert!(ran.iter().all(|r| r.unwrap().on == "device"));
    }
}
