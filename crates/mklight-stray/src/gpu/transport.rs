use super::*;
use crate::{
    bsdf::LobeParams,
    path::{EventKind, History, Signature},
    scene::{Accept, Behavior, InterfaceKind, Side},
    tally::Sink,
    tracer::{SplitMode, StrayResult, StraySettings},
};
use mklight_core::materials::{coating::ResolvedCoating, diffraction::DiffractionEfficiency};
use mklight_gpu::radiometry::{CoatingLayer, CoatingSpec};

fn numeric(value: f64, name: &str) -> MkResult<Df32> {
    let out = Df32::from_f64(value);
    if !out.hi.is_finite() || !out.lo.is_finite() || (value != 0. && !out.hi.is_normal()) {
        return Err(unsupported(&format!(
            "{name} is outside the normal f32 parameter range"
        )));
    }
    Ok(out)
}
/// Memory/command choices do not alter source IDs or their random streams;
/// options that move work between the device and the CPU change results
/// only by the rounding of the part that moved.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TransportOptions {
    /// Maximum sources per resident plan; the plan memory budget may lower it.
    pub source_tile: usize,
    /// Stack records per source on the device (at least 4); a source whose
    /// stack would overflow continues on the CPU.
    pub stack_capacity: usize,
    /// Detection journal per source on the device (at most 8 are used); a
    /// source whose journal is full continues on the CPU.
    pub detections_per_source: usize,
    /// Events per source on the device before the CPU continues it (0: no
    /// limit).
    pub head_events: u32,
    /// Share of the source chunks transported by the device
    /// (`hybrid::Design::CoSplit`); the CPU reference traces the others.
    /// `None`: the route key `stray_share`, else a fixed rule of the scene
    /// (`hybrid::default_share`).
    pub gpu_share: Option<f64>,
}
impl Default for TransportOptions {
    fn default() -> Self {
        Self {
            source_tile: 65536,
            stack_capacity: 16,
            detections_per_source: 32,
            head_events: 64,
            gpu_share: None,
        }
    }
}
pub struct TransportScene {
    geometry: ops::Geometry,
    materials: BufferHandle,
    layers: BufferHandle,
    passes: BufferHandle,
    media: BufferHandle,
    tis: BufferHandle,
    sequence: BufferHandle,
}
impl TransportScene {
    pub fn new(b: &dyn ComputeBackend, scene: &Scene, ctx: &JobContext) -> MkResult<Self> {
        let geometry = prepare_geometry(b, scene.primitives(), ctx)?;
        let nw = scene.num_wavelengths();
        let mut layers = Vec::<CoatingLayer>::new();
        let mut passes = Vec::<ops::Pass>::new();
        let mut tis = Vec::<Df32>::new();
        let materials = ops::try_fill_records(b, scene.primitives().len() * nw, ctx, |row| {
            let (pi, wl) = (row / nw, row % nw);
            let mut m = ops::Material {
                coatings: [CoatingSpec::resolve(
                    &ResolvedCoating::Bare,
                    scene.wavelengths_um[wl],
                    1.,
                    1.,
                    false,
                    &mut layers,
                )?; 2],
                ..Default::default()
            };
            let bsdf = match scene.behavior(pi) {
                Behavior::Detector => {
                    m.kind = 4;
                    None
                }
                Behavior::Opaque(o) => {
                    m.sink = Sink::of(o.kind) as u32;
                    m.specular = numeric(o.specular_reflectance, "opaque reflectance")?;
                    o.finish.as_deref()
                }
                Behavior::Interface(i) => {
                    m.kind = match i.kind {
                        InterfaceKind::Refract => 1,
                        InterfaceKind::Mirror => 2,
                        InterfaceKind::IdealLens { focal_length } => {
                            m.focal = numeric(focal_length, "ideal-lens focal length")?;
                            3
                        }
                    };
                    m.media = [i.medium_minus as u32, i.medium_plus as u32];
                    m.index = m
                        .media
                        .map(|id| Df32::from_f64(scene.media[id as usize].n[wl]));
                    for side in 0..2 {
                        m.coatings[side] = CoatingSpec::resolve(
                            &i.coating,
                            scene.wavelengths_um[wl],
                            m.index[side].to_f64(),
                            m.index[1 - side].to_f64(),
                            m.kind == 2,
                            &mut layers,
                        )?;
                    }
                    m.pass_start = passes.len() as u32;
                    m.pass_count = i.passes.len() as u32;
                    for p in &i.passes {
                        let (mask, a) = p
                            .clip
                            .as_ref()
                            .map_or((0, [0.; 2]), |a| aperture_record(a, false));
                        passes.push(ops::Pass {
                            surface: p.surface as u32,
                            side: (p.incident_side == Side::Plus) as u32,
                            mask,
                            reserved: 0,
                            aperture: a.map(Df32::from_f64),
                        });
                    }
                    if let Some(d) = i.diffraction.as_deref() {
                        m.doe = match d.efficiency {
                            DiffractionEfficiency::Ideal => 1,
                            DiffractionEfficiency::Kinoform => 2,
                            DiffractionEfficiency::Multilevel { levels } => {
                                m.levels = levels.get() as u32;
                                3
                            }
                        };
                        m.selected = d.phase.trace_order;
                        m.norm_radius = numeric(d.phase.norm_radius, "DOE normalization radius")?;
                        numeric(scene.wavelengths_um[wl], "DOE wavelength")?;
                        m.phase_count = d.phase.coefficients.len() as u32;
                        for (to, from) in m.phase.iter_mut().zip(&d.phase.coefficients) {
                            *to = numeric(*from, "DOE coefficient")?;
                        }
                        let blazes = [
                            d.transmitted_blaze[wl],
                            d.reflected_blaze_minus[wl],
                            d.reflected_blaze_plus[wl],
                        ];
                        if m.doe > 1 {
                            for (out, v) in m.blaze.iter_mut().zip(blazes) {
                                *out = numeric(v, "DOE blaze")?;
                            }
                        }
                        // The bounded Philox uniform gives |Cauchy proposal| <
                        // 2^24 - 2^20. Keep every sampled integer exactly f32.
                        if m.doe > 1 && m.blaze.iter().any(|v| v.to_f64().abs() > 1048576.) {
                            return Err(unsupported(
                                "non-ideal DOE blaze exceeds the exact-integer sampling envelope",
                            ));
                        }
                    }
                    i.scatter.as_deref()
                }
            };
            if let Some(bsdf) = bsdf {
                let parameters = match bsdf.lobe_params() {
                    LobeParams::Lambertian { .. } => {
                        m.bsdf_kind = 1;
                        [bsdf.tis(0.), 0., 0.]
                    }
                    LobeParams::Gaussian { k, s } => {
                        m.bsdf_kind = 2;
                        [k, s, 0.]
                    }
                    LobeParams::Abg { a, b, g } => {
                        m.bsdf_flags = (g.abs() < 1e-12) as u32
                            | (((g < 0.) as u32) << 1)
                            | ((((2. - g).abs() < 1e-9) as u32) << 2)
                            | ((((2. - g) > 1e-9) as u32) << 3);
                        m.bsdf_kind = 3;
                        [a, b, g]
                    }
                };
                for (out, value) in m.bsdf.iter_mut().zip(parameters) {
                    *out = numeric(value, "BSDF parameter")?;
                }
                m.tis_start = tis.len() as u32;
                m.tis_count = bsdf.tis_table().len() as u32;
                for &v in bsdf.tis_table() {
                    tis.push(numeric(v, "BSDF TIS")?);
                }
            }
            Ok(m)
        })?;
        Ok(Self {
            geometry,
            materials,
            layers: ops::fill_records(b, layers.len(), ctx, |i| layers[i])?,
            passes: ops::fill_records(b, passes.len(), ctx, |i| passes[i])?,
            tis: ops::fill_records(b, tis.len(), ctx, |i| tis[i])?,
            media: ops::fill_records(b, scene.media.len() * nw, ctx, |i| ops::Medium {
                alpha: Df32::from_f64(scene.media[i / nw].alpha[i % nw]),
                ambient: (scene.media[i / nw] == scene.media[scene.ambient as usize]) as u32,
                reserved: 0,
            })?,
            sequence: ops::fill_records(b, scene.sequence.len(), ctx, |i| {
                scene.sequence[i] as u32
            })?,
        })
    }
    /// Appends the transport of sources `first..first + count` to `plan`:
    /// their detections, ledgers and continuation stacks (`ops::Transport`).
    /// A source stops with a consistent device stack after
    /// `options.head_events` events, or when its stack or detection journal
    /// is full, for the host to continue; `sag_mode` selects the
    /// extended-sag intersection (`ops::Transport::sag_mode`).
    #[allow(clippy::too_many_arguments)]
    pub fn append(
        &self,
        plan: &mut ResidentPlan,
        scene: &Scene,
        source: &SourceSpec,
        settings: &StraySettings,
        first: u64,
        count: usize,
        options: TransportOptions,
        sag_mode: u32,
    ) -> MkResult<(BufferId, BufferId, BufferId)> {
        // Reuse the validated source descriptor and CDF, then fuse its device
        // generation into the first transport command.
        generate(
            plan,
            source,
            &scene.wavelength_weights,
            settings.seed,
            first,
            count,
            65536,
        )?;
        let ResidentOp::Stray(ops::Request::Generate(g)) = plan.operations.pop().unwrap() else {
            unreachable!()
        };
        plan.buffers.pop();
        let (primitives, nodes, shapes) = self.geometry.import(plan);
        let mut import = |b: &BufferHandle| {
            let id = plan.import(b.view());
            plan.view(id).unwrap()
        };
        let materials = import(&self.materials);
        let layers = import(&self.layers);
        let passes = import(&self.passes);
        let media = import(&self.media);
        let tis = import(&self.tis);
        let sequence = import(&self.sequence);
        let (mode, max_splits, rare) = match settings.mode {
            SplitMode::Analog => (0, 0, 0.),
            SplitMode::Weighted { rare_floor } => (1, 0, rare_floor),
            SplitMode::Split {
                max_splits,
                rare_floor,
            } => (2, max_splits, rare_floor),
        };
        let output = plan.scratch(ops::record_desc::<ops::Detection>(
            count * options.detections_per_source,
        ));
        let ledger = plan.scratch(ops::record_desc::<ops::Ledger>(count));
        let stack = plan.scratch(ops::record_desc::<ops::State>(
            count * options.stack_capacity,
        ));
        plan.operations
            .push(ResidentOp::Stray(ops::Request::Transport(Box::new(
                ops::Transport {
                    primitives,
                    nodes,
                    shapes,
                    materials,
                    layers,
                    passes,
                    media,
                    tis,
                    sequence,
                    source: g.source,
                    cdf: g.cdf,
                    settings: ops::Settings {
                        weight: Df32::from_f64(source.power() / settings.rays as f64),
                        roulette: Df32::from_f64(settings.roulette_threshold),
                        ceiling: Df32::from_f64(settings.split_ceiling.min(f32::MAX as f64)),
                        rare_floor: Df32::from_f64(rare),
                        mode,
                        max_splits,
                        max_events: settings.max_events,
                        scatter: settings.scatter_reflection as u32
                            + 2 * settings.scatter_transmission as u32,
                        start_medium: settings.start_medium.unwrap_or(scene.ambient) as u32,
                        detector_accept: match scene.detector.accept {
                            Accept::Minus => 0,
                            Accept::Plus => 1,
                            Accept::Both => 2,
                        },
                        wavelengths: scene.num_wavelengths() as u32,
                        sequence_count: scene.sequence.len() as u32,
                    },
                    seed: settings.seed,
                    first_ray: first,
                    ray_count: count,
                    stack_capacity: options.stack_capacity,
                    hit_capacity: options.detections_per_source,
                    output,
                    ledger,
                    stack,
                    head_events: options.head_events,
                    sag_mode,
                },
            ))));
        Ok((output, ledger, stack))
    }
}
pub(super) fn history(h: &ops::History) -> History {
    let mut signature = Signature::default();
    for e in &h.events[..h.length as usize] {
        let k = match e >> 28 {
            1 => EventKind::Reflect,
            2 => EventKind::Tir,
            3 => EventKind::ScatterReflect,
            4 => EventKind::ScatterTransmit,
            5 => EventKind::MechanicalSpecular,
            6 => EventKind::MechanicalScatter,
            7 => EventKind::OutsideClip,
            _ => unreachable!(),
        };
        signature.push(k, (e & 0xfffffff) as usize);
    }
    for &[tag, order] in &h.orders[..h.order_count as usize] {
        let order = order as i32;
        signature.diffract(
            (tag & 65535) as usize,
            order,
            if tag & (1 << 17) != 0 {
                order
            } else {
                order.wrapping_add(1)
            },
            tag & (1 << 16) != 0,
        );
    }
    if h.overflow != 0 {
        signature.mark_overflow();
    }
    let mut out = History::default();
    out.flags = h.flags as u8;
    out.stray_reflections = h.reflections as u8;
    out.next_seq = h.next_seq as u16;
    out.first_reflections = h.first.map(|v| v as u16);
    out.scatter_events = h.scatter_events as u16;
    out.signature = signature;
    out.set_next_specular(h.next_specular);
    out
}
/// Explicit resident adapter; unsupported capacities return an error for the
/// engine's reported fallback policy. Cancellation never invokes reference work.
pub fn run(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &Scene,
    source: &SourceSpec,
    settings: &StraySettings,
    options: TransportOptions,
    ctx: &JobContext,
) -> MkResult<StrayResult> {
    run_design_with(
        b,
        snapshot,
        scene,
        source,
        settings,
        options,
        super::hybrid::Design::selected()?,
        &mut || Ok(()),
        ctx,
    )
}
/// [`run`] with an explicit division of the work instead of the selected
/// one (`hybrid::Design`).
#[allow(clippy::too_many_arguments)]
pub fn run_design(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &Scene,
    source: &SourceSpec,
    settings: &StraySettings,
    options: TransportOptions,
    design: super::hybrid::Design,
    ctx: &JobContext,
) -> MkResult<StrayResult> {
    run_design_with(
        b,
        snapshot,
        scene,
        source,
        settings,
        options,
        design,
        &mut || Ok(()),
        ctx,
    )
}
/// [`run_design`] with CPU work of the caller, `side`, that runs once while
/// the device transports the first plans (before the CPU fallback, which
/// has no overlap).
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_design_with(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &Scene,
    source: &SourceSpec,
    settings: &StraySettings,
    options: TransportOptions,
    design: super::hybrid::Design,
    side: &mut dyn FnMut() -> MkResult<()>,
    ctx: &JobContext,
) -> MkResult<StrayResult> {
    let timer = mklight_core::job::RunTimer::start();
    match super::hybrid::run(
        b,
        snapshot.clone(),
        scene,
        source,
        settings,
        options,
        design,
        side,
        ctx,
    ) {
        Err(MkError::Unsupported(reason)) => {
            ctx.check()?;
            side()?;
            let mut result = crate::tracer::run(
                scene,
                source,
                settings,
                &ctx.subrange(ctx.progress.fraction(), 0.99),
            )?;
            let warning = format!(
                "{} stray transport fell back to CPU reference: {reason}",
                b.info().backend_id()
            );
            result.warnings.push(warning);
            result.run_info=Some(timer.finish(&snapshot,"stray_light",&serde_json::json!({"settings":settings,"source":source,"resident_options":options,"requested_backend":b.info().backend_id(),"fallback_count":1,"fallback_reason":reason}),"cpu-f64",mklight_core::job::Precision::F64Reference,Some(settings.seed),result.warnings.clone()));
            ctx.check()?;
            ctx.report(1., "stray reference replacement complete");
            Ok(result)
        }
        other => other,
    }
}
