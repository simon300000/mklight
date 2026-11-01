//! Resident non-sequential sources and analytic BVH queries. Full split-tree
//! transport remains explicitly unsupported until its independent qualification.
mod kernel;
mod math;
mod records;
use crate::{ComputeBackend, Df32, FpEnv, pending::JobControl, resident::*};
use mklight_core::{JobContext, MkResult};
pub use records::*;

#[derive(Clone, Debug)]
pub struct Geometry {
    primitives: BufferHandle,
    nodes: BufferHandle,
}
impl Geometry {
    /// Numeric upload boundary. The builder uses preorder nodes with forward
    /// escape links, so traversal cannot cycle or index beyond an allocation.
    pub fn new(
        b: &dyn ComputeBackend,
        primitives: &[Primitive],
        nodes: &[Node],
        ctx: &JobContext,
    ) -> MkResult<Self> {
        Self::from_fn(b, primitives.len(), nodes, ctx, |i| Ok(primitives[i]))
    }
    /// Pack directly into shared storage. Only the small BVH topology needs
    /// a host collection; no packed-primitive staging array is required.
    pub fn from_fn(
        b: &dyn ComputeBackend,
        count: usize,
        nodes: &[Node],
        ctx: &JobContext,
        mut primitive: impl FnMut(usize) -> MkResult<Primitive>,
    ) -> MkResult<Self> {
        if count > crate::consts::MKL_STRAY_MAX_PRIMITIVES as usize
            || nodes.len() > 2 * count
            || (count == 0) != nodes.is_empty()
        {
            return unsupported("stray BVH capacity");
        }
        let bounds_ok = |lower: [Df32; 3], upper: [Df32; 3]| {
            (0..3).all(|i| {
                lower[i].to_f64().is_finite()
                    && upper[i].to_f64().is_finite()
                    && lower[i].to_f64() <= upper[i].to_f64()
            })
        };
        for (i, n) in nodes.iter().enumerate() {
            if n.escape as usize <= i
                || n.escape as usize > nodes.len()
                || (n.primitive != u32::MAX
                    && (n.primitive as usize >= count || n.escape as usize != i + 1))
            {
                return Err(bad("stray BVH topology"));
            }
            if !bounds_ok(n.lower, n.upper) {
                return Err(bad("stray BVH bounds"));
            }
        }
        let primitives = try_fill_records(b, count, ctx, |i| {
            let p = primitive(i)?;
            if p.origin
                .iter()
                .chain(p.rotation.iter())
                .chain([&p.curvature, &p.conic, &p.inner_radius, &p.outer_radius])
                .chain(p.aperture.iter())
                .any(|v| !v.hi.is_finite() || !v.lo.is_finite())
                || p.outer_radius.to_f64() <= 0.
                || p.inner_radius.to_f64() < 0.
                || p.inner_radius.to_f64() >= p.outer_radius.to_f64()
                || !bounds_ok(p.lower, p.upper)
            {
                return Err(bad("stray primitive numeric record"));
            }
            if p.kind > 2 || (p.mask & 255) > 4 || p.mask & !511 != 0 {
                return Err(bad("stray primitive kind/mask"));
            }
            Ok(p)
        })?;
        Ok(Self {
            primitives,
            nodes: fill_records(b, nodes.len(), ctx, |i| nodes[i])?,
        })
    }
    pub fn append(&self, plan: &mut ResidentPlan, rays: PlanView, tile: usize) -> BufferId {
        let primitives = plan.import(self.primitives.view());
        let nodes = plan.import(self.nodes.view());
        let output = plan.scratch(record_desc::<Hit>(rays.desc.shape[0]));
        plan.operations.push(ResidentOp::Stray(Request::Cast(Cast {
            primitives: plan.view(primitives).unwrap(),
            nodes: plan.view(nodes).unwrap(),
            rays,
            output,
            tile,
        })));
        output
    }
}
#[derive(Clone, Debug)]
pub struct Cast {
    primitives: PlanView,
    nodes: PlanView,
    rays: PlanView,
    output: BufferId,
    tile: usize,
}
impl Cast {
    pub fn primitives(&self) -> &PlanView {
        &self.primitives
    }
    pub fn nodes(&self) -> &PlanView {
        &self.nodes
    }
    pub fn rays(&self) -> &PlanView {
        &self.rays
    }
    pub fn output(&self) -> BufferId {
        self.output
    }
    pub fn tile(&self) -> usize {
        self.tile
    }
}
#[derive(Clone, Debug)]
pub struct Generate {
    pub source: Source,
    pub cdf: Vec<Df32>,
    pub seed: u64,
    pub first_ray: u64,
    pub ray_count: usize,
    pub output: BufferId,
    pub tile: usize,
}
/// Retained frozen transport seam. The operation does not publish partial
/// transport as a completed stray result.
#[derive(Clone, Debug)]
pub struct Transport {
    pub systems: SystemSet,
    pub coverage: crate::ghost_ops::Coverage,
    pub scene: PlanView,
    pub sources: PlanView,
    pub settings: PlanView,
    pub seed: u64,
    pub first_ray: u64,
    pub ray_count: usize,
    pub output: BufferId,
    pub ledger: BufferId,
}
#[derive(Clone, Debug)]
pub enum Request {
    Generate(Generate),
    Cast(Cast),
    Transport(Box<Transport>),
}
fn check<T: bytemuck::Pod>(v: &PlanView, p: &ResidentPlan) -> MkResult<usize> {
    let n = v.desc.shape.first().copied().unwrap_or(0);
    let base = match p.buffers.get(v.buffer.0) {
        Some(PlanBuffer::Import(v)) => v.offset,
        Some(_) => 0,
        None => return Err(bad("stray buffer ID")),
    };
    if v.desc != record_desc::<T>(n) || v.stride != 1 || !(base + v.offset).is_multiple_of(2) {
        return Err(bad("stray record layout"));
    }
    Ok(n)
}
pub fn validate(r: &Request, p: &ResidentPlan) -> MkResult<()> {
    let (n, tile) = match r {
        Request::Generate(r) => {
            let s = r.source;
            if s.kind > 3
                || s.wavelength_count == 0
                || s.wavelength_count as usize != r.cdf.len()
                || r.cdf.last().unwrap().to_f64() != 1.
                || r.cdf
                    .iter()
                    .any(|v| !v.to_f64().is_finite() || v.to_f64() < 0. || v.to_f64() > 1.)
                || r.cdf.windows(2).any(|w| w[0].to_f64() > w[1].to_f64())
                || r.first_ray.checked_add(r.ray_count as u64).is_none()
            {
                return Err(bad("stray source spectrum or logical IDs"));
            }
            if s.center
                .iter()
                .chain(s.axis.iter())
                .chain(s.tangent.iter().flatten())
                .chain([&s.radius, &s.inner_radius, &s.cone, &s.distance])
                .any(|v| !v.to_f64().is_finite())
                || s.radius.to_f64() < 0.
                || s.inner_radius.to_f64() < 0.
                || s.inner_radius.to_f64() > s.radius.to_f64()
                || s.cone.to_f64() < 0.
                || s.cone.to_f64() > if s.kind == 2 { 2. } else { 1. }
            {
                return Err(bad("stray source numeric record"));
            }
            if p.desc(r.output)? != &record_desc::<Ray>(r.ray_count) {
                return Err(bad("stray generated ray output"));
            }
            (r.ray_count, r.tile)
        }
        Request::Cast(r) => {
            check::<Primitive>(&r.primitives, p)?;
            check::<Node>(&r.nodes, p)?;
            let n = check::<Ray>(&r.rays, p)?;
            if p.desc(r.output)? != &record_desc::<Hit>(n) {
                return Err(bad("stray hit output"));
            }
            (n, r.tile)
        }
        Request::Transport(_) => return unsupported("stray split-tree transport is not qualified"),
    };
    if n > u32::MAX as usize || tile == 0 || tile > crate::consts::MKL_STRAY_MAX_TILE as usize {
        return Err(bad("stray ray count or command tile"));
    }
    Ok(())
}
pub fn accesses(r: &Request) -> Access {
    match r {
        Request::Generate(r) => Access {
            writes: vec![r.output],
            ..Default::default()
        },
        Request::Cast(r) => Access {
            reads: vec![r.primitives.clone(), r.nodes.clone(), r.rays.clone()],
            writes: vec![r.output],
            ..Default::default()
        },
        Request::Transport(_) => Access::default(),
    }
}
pub fn execute(r: &Request, x: &mut cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    if x.flush_to_zero {
        execute_in::<crate::Ftz>(r, x, ctl)
    } else {
        execute_in::<crate::Ieee>(r, x, ctl)
    }
}
fn rows<'a, T: bytemuck::Pod>(x: &'a cpu::Execution, v: &PlanView) -> &'a [T] {
    bytemuck::cast_slice(&x.words(v.buffer)[v.offset..v.offset + v.desc.len().unwrap()])
}
fn execute_in<E: FpEnv>(r: &Request, x: &mut cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    let n = match r {
        Request::Generate(r) => r.ray_count,
        Request::Cast(r) => r.rays.desc.shape[0],
        Request::Transport(_) => return unsupported("stray transport"),
    };
    for i in 0..n {
        if i % 256 == 0 {
            ctl.check()?;
            ctl.report(i as f64 / n as f64, "stray source/BVH");
        }
        match r {
            Request::Generate(r) => {
                let ray = kernel::generate::<E>(r.source, &r.cdf, r.seed, r.first_ray + i as u64);
                bytemuck::cast_slice_mut::<u32, Ray>(x.words_mut(r.output))[i] = ray;
            }
            Request::Cast(r) => {
                let out = kernel::closest::<E>(
                    rows(x, &r.primitives),
                    rows(x, &r.nodes),
                    rows::<Ray>(x, &r.rays)[i],
                );
                bytemuck::cast_slice_mut::<u32, Hit>(x.words_mut(r.output))[i] = out;
            }
            Request::Transport(_) => unreachable!(),
        }
    }
    Ok(())
}
