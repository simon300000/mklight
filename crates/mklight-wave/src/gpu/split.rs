//! Host/device division of the native wave analyses. The host traces the
//! pupil lattices with the reference tracer
//! ([`crate::traced::trace_lattice`], shared with the CPU analyses through
//! the engine memo, `crate::shared`), resamples them into the reference's
//! [`PupilFunction`]s, written straight into shared allocations, and makes
//! the sampling decisions that read them (image windows, refinements,
//! automatic frequencies and focus ranges) with the reference's own rules;
//! the device runs the transforms. Host preparation and device transforms
//! overlap ([`schedule`]): every item's host stage starts at once, and each
//! prepared item goes to the first free of [`CONCURRENT`] device workers.
use super::*;
use crate::traced::{PupilLattice, PupilSettings};
use mklight_core::memo::{Memo, NodeKey};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

/// The host (reference) part of a pupil the host traced.
#[derive(Clone, Debug)]
pub struct HostPupil {
    /// The reference lattice the resident pupil was resampled from.
    pub lattice: Arc<PupilLattice>,
    /// Its `pupil_lattice` node key (`crate::shared`, f64 family).
    pub key: NodeKey,
    /// The reference pupil the resident pupil was written from, at the
    /// resident grid.
    pub pupil: Option<Arc<PupilFunction>>,
}

/// A host pupil written straight into shared allocations (amplitude and
/// coverage f32, OPD double-float), as [`ResidentPupil::from_pupil`] but
/// converted in parallel.
pub(crate) fn upload_pupil(
    b: &dyn ComputeBackend,
    p: &PupilFunction,
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
    let (covered_samples, covered) = p
        .coverage()
        .iter()
        .fold((0, 0.), |(k, sum), &c| (k + usize::from(c > 0.), sum + c));
    Ok(ResidentPupil {
        covered_samples,
        clear_area: covered * p.spacing().powi(2),
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

/// The resident pupil of a host `lattice` (key `key`) at grid `n`,
/// resampled on the host through `memo` and written into shared
/// allocations, with the host pupil it was written from and the resampling
/// warnings.
fn resident_at(
    b: &dyn ComputeBackend,
    lattice: &Arc<PupilLattice>,
    key: &NodeKey,
    n: usize,
    supersample: usize,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<(ResidentPupil, Arc<PupilFunction>, Vec<String>)> {
    let r = crate::shared::resampled(lattice, key, n, supersample, memo, ctx)?;
    let pupil = Arc::new(r.0.clone());
    let resident = upload_pupil(b, &pupil, ctx)?;
    Ok((resident, pupil, r.1.clone()))
}

/// A pupil traced on the host as a resident traced pupil, and its warnings.
pub(crate) type Placed = (ResidentTracedPupil, Vec<String>);

/// Place a host-traced lattice: its pupil at the lattice's grid
/// ([`PupilLattice::grid_for`]) is resampled on the host and written into
/// shared allocations.
pub(crate) fn place(
    b: &dyn ComputeBackend,
    key: NodeKey,
    lattice: Arc<PupilLattice>,
    settings: &PupilSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<Placed> {
    let n = lattice.grid_for(settings);
    let (pupil, host, resample_warnings) =
        resident_at(b, &lattice, &key, n, settings.supersample, memo, ctx)?;
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
    let placed = ResidentTracedPupil {
        pupil,
        info,
        nodes: None,
        valid: None,
        mask: lattice.mask.clone(),
        entrance_rim: lattice.entrance_rim.clone(),
        key,
        host: Some(HostPupil {
            lattice,
            key,
            pupil: Some(host),
        }),
    };
    Ok((placed, warnings))
}

impl ResidentTracedPupil {
    /// The pupil at grid `n` (keeping its image offset), resampled on the
    /// host through `memo`. Returns the resampling warnings.
    pub(crate) fn refine(
        &mut self,
        b: &dyn ComputeBackend,
        n: usize,
        supersample: usize,
        memo: Option<&dyn Memo>,
        ctx: &JobContext,
    ) -> MkResult<Vec<String>> {
        if n == self.pupil.n {
            return Ok(vec![]);
        }
        let Some(h) = &mut self.host else {
            return Err(MkError::Unsupported(
                "host refinement of a device-traced pupil".into(),
            ));
        };
        let offset = self.pupil.image_offset_mm;
        let (pupil, host, warnings) =
            resident_at(b, &h.lattice, &h.key, n, supersample, memo, ctx)?;
        self.pupil = pupil;
        h.pupil = Some(host);
        self.set_image_offset(offset);
        Ok(warnings)
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

/// `host(i, item)` then `device(i, prepared)` for every item, results in
/// item order. Every item's host stage starts at once on the host's thread
/// pool (each parallel inside), and each prepared item is queued as soon as
/// it is ready; up to [`CONCURRENT`] device workers take items from the
/// front of the queue, so the device transforms item k while the host
/// still prepares others.
pub(crate) fn schedule<T: Send, P: Send, R: Send>(
    items: Vec<T>,
    host: impl Fn(usize, T) -> MkResult<P> + Sync,
    device: impl Fn(usize, P) -> MkResult<R> + Sync,
) -> Vec<MkResult<R>> {
    let count = items.len();
    let devices = CONCURRENT.min(count);
    let slots: Vec<Mutex<Option<MkResult<R>>>> = (0..count).map(|_| Mutex::new(None)).collect();
    // Prepared items and whether every host stage has finished.
    let queue = (
        Mutex::new((VecDeque::<(usize, MkResult<P>)>::new(), false)),
        Condvar::new(),
    );
    let worker = || loop {
        let next = {
            let mut q = queue.0.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(item) = q.0.pop_front() {
                    break Some(item);
                }
                if q.1 {
                    break None;
                }
                q = queue.1.wait(q).unwrap_or_else(|e| e.into_inner());
            }
        };
        let Some((i, p)) = next else { break };
        let r = p.and_then(|p| device(i, p));
        *slots[i].lock().unwrap_or_else(|e| e.into_inner()) = Some(r);
    };
    // Marks the host stages finished when dropped, so the device workers
    // leave the queue even when a host stage panics.
    struct Finish<'q, P>(&'q (Mutex<(VecDeque<P>, bool)>, Condvar));
    impl<P> Drop for Finish<'_, P> {
        fn drop(&mut self) {
            self.0.0.lock().unwrap_or_else(|e| e.into_inner()).1 = true;
            self.0.1.notify_all();
        }
    }
    std::thread::scope(|s| {
        for _ in 0..devices {
            s.spawn(worker);
        }
        let _finish = Finish(&queue);
        items.into_par_iter().enumerate().for_each(|(i, t)| {
            let p = host(i, t);
            queue
                .0
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .0
                .push_back((i, p));
            queue.1.notify_all();
        });
    });
    slots
        .into_iter()
        .map(|m| {
            m.into_inner()
                .unwrap_or_else(|e| e.into_inner())
                .expect("every item ran")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every item's result comes back in item order, with its errors.
    #[test]
    fn schedule_keeps_item_order_and_errors() {
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
        let results = schedule(items, host, device);
        assert_eq!(results.len(), 23);
        for (i, r) in results.iter().enumerate() {
            if i == 7 || i == 11 {
                assert!(r.is_err());
            } else {
                assert_eq!(*r.as_ref().unwrap(), 10 * i + 1);
            }
        }
    }

    /// A panicking host stage unwinds out of `schedule`; the device workers
    /// do not wait for items that never come.
    #[test]
    fn schedule_propagates_a_host_panic() {
        let outcome = std::panic::catch_unwind(|| {
            schedule(
                (0..9).collect::<Vec<usize>>(),
                |_, t| {
                    assert_ne!(t, 4, "host stage");
                    Ok(t)
                },
                |_, p: usize| Ok(p),
            )
        });
        assert!(outcome.is_err());
    }
}
