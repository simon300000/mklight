//! Numeric transport contract. Source-local stacks preserve depth-first draws.
use super::*;
use crate::radiometry::CoatingSpec;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Material {
    pub coatings: [CoatingSpec; 2],
    pub index: [Df32; 2],
    pub alpha: Df32,
    pub specular: Df32,
    pub focal: Df32,
    pub bsdf: [Df32; 3],
    pub phase: [Df32; 10],
    pub norm_radius: Df32,
    pub blaze: [Df32; 3],
    /// 0 opaque, 1 refract, 2 mirror, 3 ideal lens, 4 detector.
    pub kind: u32,
    pub media: [u32; 2],
    pub sink: u32,
    pub pass_start: u32,
    pub pass_count: u32,
    /// 0 absent, 1 Lambertian, 2 Gaussian, 3 ABg.
    pub bsdf_kind: u32,
    pub tis_start: u32,
    pub tis_count: u32,
    /// 0 absent, 1 ideal, 2 kinoform, 3 multilevel.
    pub doe: u32,
    pub selected: i32,
    pub levels: u32,
    pub phase_count: u32,
    /// Reference ABg proposal branches resolved from the original f64 g.
    pub bsdf_flags: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Pass {
    pub surface: u32,
    pub side: u32,
    pub mask: u32,
    pub reserved: u32,
    pub aperture: [Df32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Medium {
    pub alpha: Df32,
    pub ambient: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct History {
    pub flags: u32,
    pub reflections: u32,
    pub next_seq: u32,
    pub next_specular: i32,
    pub first: [u32; 2],
    pub scatter_events: u32,
    pub length: u32,
    pub overflow: u32,
    pub order_count: u32,
    pub events: [u32; 6],
    /// surface | reflected << 16 | selected << 17, signed order bits.
    pub orders: [[u32; 2]; 32],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct State {
    pub ray: Ray,
    pub weight: Df32,
    pub history: History,
    pub medium: u32,
    pub events: u32,
    pub splits: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Detection {
    pub history: History,
    pub weight: Df32,
    pub x: Df32,
    pub y: Df32,
    pub class: u32,
    pub wavelength: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Ledger {
    /// Absorption sinks 0..12; escape, leak, truncation, sampling failure;
    /// roulette gain/loss, weighting gain/loss; emitted.
    pub sums: [Df32; 21],
    pub draws: u64,
    pub events: u32,
    pub hits: u32,
    pub top: u32,
    /// 0 complete, 1 continuation, 2 stack capacity, 3 hit capacity,
    /// 4 reference-required coating/order arithmetic.
    pub status: u32,
    pub wavelength: u32,
    pub reserved: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Settings {
    pub weight: Df32,
    pub roulette: Df32,
    pub ceiling: Df32,
    pub rare_floor: Df32,
    pub mode: u32,
    pub max_splits: u32,
    pub max_events: u32,
    pub scatter: u32,
    pub start_medium: u32,
    pub detector_accept: u32,
    pub wavelengths: u32,
    pub sequence_count: u32,
}
#[derive(Clone, Debug)]
pub struct Transport {
    pub primitives: PlanView,
    pub nodes: PlanView,
    pub shapes: PlanView,
    pub materials: PlanView,
    pub layers: PlanView,
    pub passes: PlanView,
    pub media: PlanView,
    pub tis: PlanView,
    pub sequence: PlanView,
    pub source: Source,
    pub cdf: Vec<Df32>,
    pub settings: Settings,
    pub seed: u64,
    pub first_ray: u64,
    pub ray_count: usize,
    /// Fixed source tile, independent of command/continuation boundaries.
    pub stack_capacity: usize,
    pub hit_capacity: usize,
    pub event_budget: usize,
    /// `hit_capacity` detections per source; rows past `Ledger::hits` are
    /// unspecified.
    pub output: BufferId,
    pub ledger: BufferId,
    /// Handoff mode: the per-source continuation stacks are an output of
    /// `stack_capacity` states per source. A source whose stack would
    /// overflow (`Ledger::status` 2), whose detection journal is full (3) or
    /// that reached `head_events` (5) stops on the device with a consistent
    /// stack, ledger and draw cursor, and the host continues it; a source
    /// that needs reference arithmetic (4) stops and is retraced by the host.
    /// Nothing grows and nothing fails the plan. `None`: the stack is
    /// internal, grows on device, and capacity or reference status fails it.
    pub stack_output: Option<BufferId>,
    /// Handoff mode only: events per source on the device before status 5
    /// (0: no limit).
    pub head_events: u32,
    /// Handoff mode only: run the transport as a wavefront (one event of
    /// every live source per bounce, in separate pop, closest-hit and
    /// interaction kernels) instead of persistent lanes. Scheduling only:
    /// the records are identical.
    pub wavefront: bool,
    /// Extended-sag intersection on the device: 0 the double-float cell
    /// scan; 1 a plain-f32 pre-scan that skips cells the scan would find
    /// empty (the decided roots are those of mode 0); 2 (handoff mode only)
    /// mode 0, but a scan longer than `MKL_STRAY_SAG_HANDOFF_CELLS` cells
    /// stops the source before that event (`Ledger::status` 6) for the host
    /// to continue; 3 modes 1 and 2. The CPU specification always scans as
    /// mode 0.
    pub sag_mode: u32,
}
impl Transport {
    pub fn reads(&self) -> Vec<PlanView> {
        vec![
            self.primitives.clone(),
            self.nodes.clone(),
            self.shapes.clone(),
            self.materials.clone(),
            self.layers.clone(),
            self.passes.clone(),
            self.media.clone(),
            self.tis.clone(),
            self.sequence.clone(),
        ]
    }
    pub fn validate(&self, p: &ResidentPlan) -> MkResult<()> {
        validate_source(self.source, &self.cdf, self.first_ray, self.ray_count)?;
        if !self.settings.weight.to_f64().is_finite() || self.settings.weight.hi < f32::MIN_POSITIVE
        {
            return unsupported("stray source weight outside the normal f32 range");
        }
        let primitives = check::<Primitive>(&self.primitives, p)?;
        check::<Node>(&self.nodes, p)?;
        check::<crate::radiometry::CoatingLayer>(&self.layers, p)?;
        check::<Pass>(&self.passes, p)?;
        check::<Df32>(&self.tis, p)?;
        let media = check::<Medium>(&self.media, p)?;
        let seq = check::<u32>(&self.sequence, p)?;
        let nw = self.settings.wavelengths as usize;
        if nw != self.source.wavelength_count as usize
            || seq != self.settings.sequence_count as usize
            || check::<Material>(&self.materials, p)? != primitives * nw
            || check::<SurfaceData>(&self.shapes, p)? != primitives
            || !media.is_multiple_of(nw)
            || self.settings.start_medium as usize >= media / nw
        {
            return Err(bad("stray transport scene extents"));
        }
        if self.ray_count == 0
            || self.ray_count > 65536
            || self.stack_capacity == 0
            || self.stack_capacity > 8192
            || self.hit_capacity == 0
            || self.hit_capacity > 1024
            || self.event_budget == 0
            || self.event_budget > 128
            || self.first_ray.checked_add(self.ray_count as u64).is_none()
            || p.desc(self.output)? != &record_desc::<Detection>(self.ray_count * self.hit_capacity)
            || p.desc(self.ledger)? != &record_desc::<Ledger>(self.ray_count)
            || ((self.head_events != 0 || self.wavefront || self.sag_mode >= 2)
                && self.stack_output.is_none())
            || self.sag_mode > 3
        {
            return Err(bad("stray transport capacity/output"));
        }
        if let Some(stack) = self.stack_output
            && p.desc(stack)? != &record_desc::<State>(self.ray_count * self.stack_capacity)
        {
            return Err(bad("stray transport stack output"));
        }
        Ok(())
    }
}

impl Default for Material {
    fn default() -> Self {
        Self::zeroed()
    }
}
