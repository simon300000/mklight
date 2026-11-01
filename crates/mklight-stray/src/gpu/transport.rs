use super::*;
use crate::{
    bsdf::LobeParams,
    path::{EventKind, History, Signature},
    scene::{Accept, Behavior, InterfaceKind, Side},
    tally::{Sink, StrayTally},
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
/// Memory/command choices do not alter source IDs or their random streams.
#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct TransportOptions {
    /// Maximum sources per resident plan; the plan memory budget may lower it.
    pub source_tile: usize,
    /// Initial stack records per source; grows on device when exceeded.
    pub stack_capacity: usize,
    /// A source exceeding this falls back to the reference.
    pub detections_per_source: usize,
    /// Maximum events per lane per transport round.
    pub event_budget: usize,
}
/// Resident scratch budget per plan (two plans overlap): rays, ledgers,
/// queues, initial stacks, detections and their tags.
const PLAN_BYTES: usize = 160 << 20;
impl TransportOptions {
    /// Sources per plan, a multiple of the 256-source estimator leaves.
    fn plan_sources(&self) -> usize {
        use std::mem::size_of;
        let per_source = size_of::<ops::Ray>()
            + size_of::<ops::Ledger>()
            + 8
            + self.stack_capacity.max(4) * size_of::<ops::State>()
            + self.detections_per_source * (size_of::<ops::Detection>() + 16);
        (PLAN_BYTES / per_source).min(self.source_tile.div_ceil(256) * 256) / 256 * 256
    }
}
impl Default for TransportOptions {
    fn default() -> Self {
        Self {
            source_tile: 65536,
            stack_capacity: 16,
            detections_per_source: 32,
            event_budget: 32,
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
    ) -> MkResult<(BufferId, BufferId)> {
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
                    event_budget: options.event_budget,
                    output,
                    ledger,
                },
            ))));
        Ok((output, ledger))
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
    let timer = mklight_core::job::RunTimer::start();
    match run_resident(b, snapshot.clone(), scene, source, settings, options, ctx) {
        Err(MkError::Unsupported(reason)) => {
            ctx.check()?;
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
fn run_resident(
    b: &dyn ComputeBackend,
    snapshot: Snapshot,
    scene: &Scene,
    source: &SourceSpec,
    settings: &StraySettings,
    options: TransportOptions,
    ctx: &JobContext,
) -> MkResult<StrayResult> {
    crate::tracer::validate_run(scene, source, settings)?;
    if options.source_tile == 0
        || options.source_tile > 65536
        || options.plan_sources() == 0
        || options.detections_per_source == 0
        || options.detections_per_source > 1024
        || options.stack_capacity == 0
        || options.stack_capacity > 8192
        || settings
            .start_medium
            .is_some_and(|m| m as usize >= scene.media.len())
    {
        return Err(MkError::InvalidModel("stray transport settings".into()));
    }
    // P3 forms per-source second products and sums at most 256 sources.
    // Reject absolute power scales that cannot represent those products;
    // the reference result keeps the requested physical units unchanged.
    let w = source.power() / settings.rays as f64;
    let floor = match settings.mode {
        SplitMode::Analog => 0.,
        SplitMode::Weighted { rare_floor } | SplitMode::Split { rare_floor, .. } => rare_floor,
    };
    let growth = (1. + 3. * floor)
        .powf(settings.max_events as f64)
        .min(settings.split_ceiling);
    let radius = scene
        .primitives()
        .iter()
        .enumerate()
        .filter(|(i, _)| matches!(scene.behavior(*i), Behavior::Detector))
        .map(|(_, p)| match &p.shape {
            Shape::Sag(s) => s.r_outer(),
            Shape::Cylinder(c) => c.radius,
        })
        .fold(0f64, f64::max);
    let term = w * options.detections_per_source as f64 * growth * radius.powi(2).max(1.);
    if w < (f32::MIN_POSITIVE as f64).sqrt()
        || !term.is_finite()
        || term * term * 256. > f32::MAX as f64 / 16.
    {
        return Err(unsupported(
            "source power is outside the f32 estimator-product range",
        ));
    }
    let timer = mklight_core::job::RunTimer::start();
    let prep_started = std::time::Instant::now();
    let prepared = TransportScene::new(b, scene, &ctx.subrange(0., 0.05))?;
    let preparation_seconds = prep_started.elapsed().as_secs_f64();
    let mut transport_seconds = 0.;
    let mut tally_seconds = 0.;
    let mut tally = StrayTally::new(
        scene.num_wavelengths(),
        (scene.detector.nx * scene.detector.ny) as usize,
    );
    let mut plans = Vec::new();
    let coverage = super::tally::coverage(b, settings, ctx)?;
    let submit = |first: u64| {
        let count = (settings.rays - first).min(options.plan_sources() as u64) as usize;
        let mut plan = ResidentPlan::new(snapshot.clone());
        plan.seed = Some(settings.seed);
        plan.settings = serde_json::json!({"stray_version":mklight_gpu::consts::MKL_STRAY_VERSION,"scene":scene.fingerprint(),"first":first});
        let (hits, ledger) =
            prepared.append(&mut plan, scene, source, settings, first, count, options)?;
        let tags = super::tally::tags(
            &mut plan,
            hits,
            ledger,
            &coverage,
            super::tally::params(scene, settings, count, options.detections_per_source),
        )?;
        plan.outputs = vec![hits, ledger, tags];
        let sub = ctx.subrange(
            0.05 + first as f64 / settings.rays as f64 * 0.9,
            0.05 + (first + count as u64) as f64 / settings.rays as f64 * 0.9,
        );
        MkResult::Ok((first, count, b.submit_plan(plan, &sub)?, sub))
    };
    // The next plan's transport overlaps this plan's host key enumeration.
    let mut next = Some(submit(0)?);
    while let Some((first, count, pending, sub)) = next.take() {
        ctx.check()?;
        let started = std::time::Instant::now();
        let result = pending.wait()?;
        transport_seconds += started.elapsed().as_secs_f64();
        if first + (count as u64) < settings.rays {
            next = Some(submit(first + count as u64)?);
        }
        let started = std::time::Instant::now();
        let views = std::array::from_fn(|i| result.outputs[i].view());
        for chunk in super::tally::collect(
            b,
            &snapshot,
            scene,
            settings,
            views,
            count,
            options.detections_per_source,
            &sub,
            &mut plans,
        )? {
            tally.add(&chunk);
        }
        tally_seconds += started.elapsed().as_secs_f64();
        plans.push(result.stats);
    }
    let pack_started = std::time::Instant::now();
    let (detector, energy, mut signatures) = tally.finish(&scene.detector, settings.owned_classes);
    for s in &mut signatures {
        s.label = s.signature.describe(|p| scene.label(p).to_string());
    }
    let mut warnings = scene.warnings.clone();
    if energy.sampling_failed > 0. {
        warnings.push(format!(
            "{:.3e} W lost to BSDF rejection-sampling failures",
            energy.sampling_failed
        ));
    }
    if energy.escaped_from_solid > 1e-12 * energy.emitted {
        warnings.push(format!(
            "{:.3e} W left the scene from inside a solid",
            energy.escaped_from_solid
        ));
    }
    if energy.truncated > 1e-6 * energy.emitted {
        warnings.push(format!(
            "{:.3e} W truncated at {} events per path",
            energy.truncated, settings.max_events
        ));
    }
    let packing_seconds = pack_started.elapsed().as_secs_f64();
    let info=timer.finish(&snapshot,"stray_light",&serde_json::json!({"settings":settings,"source":source,"resident_options":options,"stray_version":mklight_gpu::consts::MKL_STRAY_VERSION,"plans":plans,"device":b.info(),"precision":b.precision(),"fallback_count":0,"scene_fingerprint":scene.fingerprint(),"timings":{"preparation_seconds":preparation_seconds,"transport_seconds":transport_seconds,"tally_seconds":tally_seconds,"packing_seconds":packing_seconds}}),&b.info().backend_id(),mklight_core::job::Precision::F32Gpu,Some(settings.seed),warnings.clone());
    ctx.check()?;
    ctx.report(1., "stray complete");
    Ok(StrayResult {
        detector,
        energy,
        signatures,
        warnings,
        run_info: Some(info),
    })
}
