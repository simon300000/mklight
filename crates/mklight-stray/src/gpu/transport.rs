use super::*;
use crate::{
    bsdf::LobeParams,
    path::{EventKind, History, PathClass, Signature},
    scene::{Accept, Behavior, InterfaceKind, Side},
    tally::{ChunkTally, Sink, StrayTally},
    tracer::{SplitMode, StrayResult, StraySettings},
};
use mklight_core::materials::{coating::ResolvedCoating, diffraction::DiffractionEfficiency};
use mklight_gpu::radiometry::{CoatingLayer, CoatingSpec};

/// Memory/command choices do not alter source IDs or their random streams.
#[derive(Clone, Copy, Debug)]
pub struct TransportOptions {
    pub source_tile: usize,
    pub stack_capacity: usize,
    pub detections_per_source: usize,
    pub event_budget: usize,
}
impl Default for TransportOptions {
    fn default() -> Self {
        Self {
            source_tile: 256,
            stack_capacity: 128,
            detections_per_source: 128,
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
            let mut m = ops::Material::default();
            m.coatings = [CoatingSpec::resolve(
                &ResolvedCoating::Bare,
                scene.wavelengths_um[wl],
                1.,
                1.,
                false,
                &mut layers,
            )?; 2];
            let bsdf = match scene.behavior(pi) {
                Behavior::Detector => {
                    m.kind = 4;
                    None
                }
                Behavior::Opaque(o) => {
                    m.sink = Sink::of(o.kind) as u32;
                    m.specular = Df32::from_f64(o.specular_reflectance);
                    o.finish.as_deref()
                }
                Behavior::Interface(i) => {
                    m.kind = match i.kind {
                        InterfaceKind::Refract => 1,
                        InterfaceKind::Mirror => 2,
                        InterfaceKind::IdealLens { focal_length } => {
                            m.focal = Df32::from_f64(focal_length);
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
                        m.norm_radius = Df32::from_f64(d.phase.norm_radius);
                        m.phase_count = d.phase.coefficients.len() as u32;
                        for (to, from) in m.phase.iter_mut().zip(&d.phase.coefficients) {
                            *to = Df32::from_f64(*from);
                        }
                        m.blaze = [
                            d.transmitted_blaze[wl],
                            d.reflected_blaze_minus[wl],
                            d.reflected_blaze_plus[wl],
                        ]
                        .map(Df32::from_f64);
                    }
                    i.scatter.as_deref()
                }
            };
            if let Some(bsdf) = bsdf {
                m.bsdf = match bsdf.lobe_params() {
                    LobeParams::Lambertian { .. } => {
                        m.bsdf_kind = 1;
                        [bsdf.tis(0.), 0., 0.]
                    }
                    LobeParams::Gaussian { k, s } => {
                        m.bsdf_kind = 2;
                        [k, s, 0.]
                    }
                    LobeParams::Abg { a, b, g } => {
                        m.bsdf_kind = 3;
                        [a, b, g]
                    }
                }
                .map(Df32::from_f64);
                m.tis_start = tis.len() as u32;
                m.tis_count = bsdf.tis_table().len() as u32;
                tis.extend(bsdf.tis_table().iter().copied().map(Df32::from_f64));
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
fn history(h: &ops::History) -> History {
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
    settings.validate()?;
    source.validate()?;
    if options.source_tile == 0
        || options.source_tile > 65536
        || settings
            .start_medium
            .is_some_and(|m| m as usize >= scene.media.len())
    {
        return Err(MkError::InvalidModel("stray transport settings".into()));
    }
    let timer = mklight_core::job::RunTimer::start();
    let prepared = TransportScene::new(b, scene, ctx)?;
    let mut tally = StrayTally::new(
        scene.num_wavelengths(),
        (scene.detector.nx * scene.detector.ny) as usize,
    );
    let mut plans = Vec::new();
    let mut first = 0;
    while first < settings.rays {
        ctx.check()?;
        let count = (settings.rays - first).min(options.source_tile as u64) as usize;
        let mut plan = ResidentPlan::new(snapshot.clone());
        plan.seed = Some(settings.seed);
        plan.settings = serde_json::json!({"stray_version":mklight_gpu::consts::MKL_STRAY_VERSION,"scene":scene.fingerprint(),"first":first});
        let (hits, ledger) =
            prepared.append(&mut plan, scene, source, settings, first, count, options)?;
        plan.outputs = vec![hits, ledger];
        let sub = ctx.subrange(
            first as f64 / settings.rays as f64 * 0.95,
            (first + count as u64) as f64 / settings.rays as f64 * 0.95,
        );
        let result = b.submit_plan(plan, &sub)?.wait()?;
        let hv = b.read_view(result.outputs[0].view())?;
        let lv = b.read_view(result.outputs[1].view())?;
        let hits = ops::read_records::<ops::Detection>(&hv)?;
        let ledgers = ops::read_records::<ops::Ledger>(&lv)?;
        let mut chunk = ChunkTally::new(scene.num_wavelengths(), settings.owned_classes);
        chunk.track_signatures = settings.signatures;
        for (i, l) in ledgers.iter().enumerate() {
            chunk.begin_ray();
            chunk.rays += 1;
            chunk.events += l.events as u64;
            let v = l.sums.map(Df32::to_f64);
            chunk.emitted += v[20];
            chunk.emitted_by_wavelength[l.wavelength as usize] += v[20];
            for k in 0..12 {
                chunk.absorbed[k] += v[k];
            }
            chunk.escaped += v[12];
            chunk.escaped_from_solid += v[13];
            chunk.truncated += v[14];
            chunk.sampling_failed += v[15];
            chunk.roulette(v[16]);
            chunk.roulette(-v[17]);
            chunk.reweight(v[18]);
            chunk.reweight(-v[19]);
            for h in &hits[i * options.detections_per_source..][..l.hits as usize] {
                let hist = history(&h.history);
                let c = PathClass::ALL[h.class as usize];
                let (x, y) = (h.x.to_f64(), h.y.to_f64());
                chunk.detect(
                    c,
                    settings.owns(c, &hist.signature),
                    scene.detector.pixel(x, y),
                    h.weight.to_f64(),
                    h.wavelength as usize,
                    &hist,
                    x,
                    y,
                );
            }
            chunk.end_ray();
        }
        tally.add(&chunk);
        plans.push(result.stats);
        first += count as u64;
    }
    let (detector, energy, mut signatures) = tally.finish(&scene.detector, settings.owned_classes);
    for s in &mut signatures {
        s.label = s.signature.describe(|p| scene.label(p).to_string());
    }
    let warnings = scene.warnings.clone();
    let info=timer.finish(&snapshot,"stray_light",&serde_json::json!({"settings":settings,"source":source,"stray_version":mklight_gpu::consts::MKL_STRAY_VERSION,"plans":plans}),&b.info().backend_id(),mklight_core::job::Precision::F32Gpu,Some(settings.seed),warnings.clone());
    ctx.report(1., "stray complete");
    Ok(StrayResult {
        detector,
        energy,
        signatures,
        warnings,
        run_info: Some(info),
    })
}
