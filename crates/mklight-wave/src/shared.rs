//! Traced pupils shared between analyses through a
//! [`Memo`](mklight_core::memo::Memo) (the engine's analysis graph).
//!
//! Two node families, both CPU f64:
//!
//! | node | key | value |
//! |---|---|---|
//! | `pupil_lattice` | tracer content, prepared field, wavelength, lattice settings | [`PupilLattice`]: lattice and rim rays, reference sphere, masks, automatic grid |
//! | `resampled_pupil` | lattice key, grid, supersampling | the [`PupilFunction`] on that grid and its warnings |
//!
//! The tracer content is the hash of its (configured, solved) snapshot and
//! its settings; the prepared field enters bit for bit (source and chief
//! aims), so an analysis-local field point is its own node. Grid and
//! supersampling are not part of the lattice key: an explicit grid, the
//! automatic grid and every window refinement resample the same lattice.
//! The material library is not visible here; an engine memo scopes keys
//! by it. Without a memo every node is built, which is exactly
//! [`trace_pupil`](crate::traced::trace_pupil).

use std::sync::Arc;

use mklight_core::memo::{Built, KeyHasher, Memo, NodeKey, downcast};
use mklight_core::{JobContext, MkResult};
use mklight_trace::{ObjectSource, PreparedField, SequentialTracer};
use rayon::prelude::*;

use crate::pupil::PupilFunction;
use crate::traced::{PupilLattice, PupilSettings, TracedPupil, trace_lattice};

/// Node families and their algorithm versions (bump on any change of
/// what a node computes).
pub const LATTICE_NODE: &str = "pupil_lattice";
pub const RESAMPLED_NODE: &str = "resampled_pupil";
const LATTICE_VERSION: &str = "mklight-wave/pupil_lattice/cpu-f64/1";
const RESAMPLED_VERSION: &str = "mklight-wave/resampled_pupil/cpu-f64/1";

/// A resampled pupil with offset zero and its resampling warnings.
pub type Resampled = (PupilFunction, Vec<String>);

fn field_key(h: &mut KeyHasher, f: &PreparedField) {
    let p = &f.point;
    let v = &p.vignetting;
    h.f64s(&[p.hx, p.hy, v.vdx, v.vdy, v.vcx, v.vcy]);
    match f.source {
        ObjectSource::Direction { dir } => h.str("direction").f64s(&[dir.x, dir.y, dir.z]),
        ObjectSource::Point { point } => h.str("point").f64s(&[point.x, point.y, point.z]),
    };
    h.usize(f.aims.len());
    for a in &f.aims {
        h.f64s(&a.chief).bool(a.converged).f64(a.residual);
        match a.jacobian {
            Some(j) => h.bool(true).f64s(&[j[0][0], j[0][1], j[1][0], j[1][1]]),
            None => h.bool(false),
        };
    }
    match &f.error {
        Some(e) => h.bool(true).str(e),
        None => h.bool(false),
    };
}

/// Key of the lattice of `field` at wavelength `wl` (module docs).
pub fn lattice_key(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    settings: &PupilSettings,
) -> NodeKey {
    let mut h = KeyHasher::new(LATTICE_VERSION);
    h.bytes(&tracer.system().snapshot.hash().0)
        .json(tracer.settings())
        .json(&tracer.aiming())
        .usize(wl);
    field_key(&mut h, field);
    h.usize(settings.rays)
        .json(&settings.reference)
        .json(&settings.amplitude)
        .bool(settings.rim)
        .f64(settings.rim_tolerance);
    h.finish()
}

/// Key of `lattice` resampled to `n × n` with `supersample`.
pub fn resampled_key(lattice: &NodeKey, n: usize, supersample: usize) -> NodeKey {
    KeyHasher::new(RESAMPLED_VERSION)
        .key(lattice)
        .usize(n)
        .usize(supersample)
        .finish()
}

fn pupil_bytes(p: &PupilFunction) -> usize {
    std::mem::size_of::<PupilFunction>() + 3 * 8 * p.n() * p.n()
}

/// Nodes of `keys` from `memo`, or all built in parallel without one.
fn nodes<T: Send + Sync + 'static>(
    memo: Option<&dyn Memo>,
    kind: &'static str,
    keys: &[NodeKey],
    ctx: &JobContext,
    build: &(dyn Fn(usize) -> MkResult<(T, usize)> + Sync),
) -> Vec<MkResult<Arc<T>>> {
    match memo {
        Some(m) => m
            .nodes(kind, keys, ctx, &|i| {
                build(i).map(|(v, bytes)| Built::new(v, bytes))
            })
            .into_iter()
            .map(|r| r.map(downcast::<T>))
            .collect(),
        None => (0..keys.len())
            .into_par_iter()
            .map(|i| build(i).map(|(v, _)| Arc::new(v)))
            .collect(),
    }
}

/// One traced pupil to build: field, wavelength index and its progress
/// context.
pub(crate) struct PupilJob<'a> {
    pub field: &'a PreparedField,
    pub wl: usize,
    pub ctx: JobContext,
}

/// The traced pupils of `jobs` at `settings`, in order, each with its
/// lattice key ([`crate::traced::trace_pupil`] per job, sharing lattices and
/// resampled grids through `memo`).
pub(crate) fn traced_pupils(
    tracer: &SequentialTracer,
    jobs: &[PupilJob],
    settings: &PupilSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> Vec<MkResult<(NodeKey, TracedPupil)>> {
    if let Err(e) = settings.validate() {
        return jobs.iter().map(|_| Err(e.clone())).collect();
    }
    let keys: Vec<NodeKey> = jobs
        .iter()
        .map(|j| lattice_key(tracer, j.field, j.wl, settings))
        .collect();
    let lattices: Vec<MkResult<Arc<PupilLattice>>> = nodes(memo, LATTICE_NODE, &keys, ctx, &|i| {
        let j = &jobs[i];
        trace_lattice(tracer, j.field, j.wl, settings, &j.ctx).map(|l| {
            let b = l.bytes();
            (l, b)
        })
    });
    // Resample every lattice that was traced, at its decided grid.
    let ok: Vec<usize> = (0..jobs.len()).filter(|&i| lattices[i].is_ok()).collect();
    let lattice = |i: usize| lattices[i].as_ref().expect("traced lattice");
    let rkeys: Vec<NodeKey> = ok
        .iter()
        .map(|&i| {
            resampled_key(
                &keys[i],
                lattice(i).grid_for(settings),
                settings.supersample,
            )
        })
        .collect();
    let resampled: Vec<MkResult<Arc<Resampled>>> = nodes(memo, RESAMPLED_NODE, &rkeys, ctx, &|k| {
        let i = ok[k];
        let l = lattice(i);
        l.resample(l.grid_for(settings), settings.supersample, &jobs[i].ctx)
            .map(|r| {
                let b = pupil_bytes(&r.0) + r.1.iter().map(String::len).sum::<usize>();
                (r, b)
            })
    });
    let mut resampled = resampled.into_iter();
    lattices
        .into_iter()
        .zip(keys)
        .map(|(l, key)| {
            let l = l?;
            let r = resampled.next().expect("one resampled pupil per lattice")?;
            Ok((key, l.assemble(r.0.clone(), &r.1)))
        })
        .collect()
}

/// The traced lattices of `jobs` at `settings`, in order, each with its
/// key: the first stage of [`traced_pupils`] alone, for callers that
/// resample elsewhere (the native device analyses, `crate::gpu`).
/// Shared through `memo` exactly as [`traced_pupils`] shares them.
pub(crate) fn traced_lattices(
    tracer: &SequentialTracer,
    jobs: &[PupilJob],
    settings: &PupilSettings,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> Vec<MkResult<(NodeKey, Arc<PupilLattice>)>> {
    if let Err(e) = settings.validate() {
        return jobs.iter().map(|_| Err(e.clone())).collect();
    }
    let keys: Vec<NodeKey> = jobs
        .iter()
        .map(|j| lattice_key(tracer, j.field, j.wl, settings))
        .collect();
    nodes(memo, LATTICE_NODE, &keys, ctx, &|i| {
        let j = &jobs[i];
        trace_lattice(tracer, j.field, j.wl, settings, &j.ctx).map(|l| {
            let b = l.bytes();
            (l, b)
        })
    })
    .into_iter()
    .zip(keys)
    .map(|(l, k)| l.map(|l| (k, l)))
    .collect()
}

/// `lattice` (key `key`) resampled to `n × n` through `memo` (offset zero):
/// the resampled-pupil node [`traced_pupils`] and [`resample`] share.
pub(crate) fn resampled(
    lattice: &PupilLattice,
    key: &NodeKey,
    n: usize,
    supersample: usize,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<Arc<Resampled>> {
    let rkey = resampled_key(key, n, supersample);
    nodes(memo, RESAMPLED_NODE, &[rkey], ctx, &|_| {
        lattice.resample(n, supersample, ctx).map(|r| {
            let b = pupil_bytes(&r.0) + r.1.iter().map(String::len).sum::<usize>();
            (r, b)
        })
    })
    .pop()
    .expect("one node")
}

/// `tp` (from the lattice `key`) resampled to `n × n`, keeping its image
/// offset ([`TracedPupil::resample`] through `memo`).
pub(crate) fn resample(
    tp: &mut TracedPupil,
    key: &NodeKey,
    n: usize,
    supersample: usize,
    memo: Option<&dyn Memo>,
    ctx: &JobContext,
) -> MkResult<Vec<String>> {
    let Some(m) = memo else {
        return tp.resample(n, supersample, ctx);
    };
    if n == tp.pupil.n() {
        return Ok(Vec::new());
    }
    let rkey = resampled_key(key, n, supersample);
    let built = {
        let src: &TracedPupil = tp;
        nodes(Some(m), RESAMPLED_NODE, &[rkey], ctx, &|_| {
            // The lattice's samples, mask and geometry are the traced
            // pupil's (`TracedPupil::resample`).
            let p = &src.pupil;
            let (mut refined, warnings) = crate::grid_from_structured(
                &src.samples,
                n,
                p.wavelength_um(),
                *p.geometry(),
                src.mask.as_ref(),
                supersample,
                ctx,
            )?;
            refined.set_image_offset_mm([0.0, 0.0]);
            let w: Vec<String> = warnings.iter().map(ToString::to_string).collect();
            let b = pupil_bytes(&refined) + w.iter().map(String::len).sum::<usize>();
            Ok(((refined, w), b))
        })
        .pop()
        .expect("one node")?
    };
    let offset = tp.pupil.image_offset_mm();
    tp.pupil = built.0.clone();
    tp.pupil.set_image_offset_mm(offset);
    Ok(built.1.clone())
}
