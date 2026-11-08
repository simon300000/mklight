use super::*;
use mklight_core::MkError;
use mklight_gpu::resident::unsupported;

/// Lanes per transport round. Measured throughput saturates near 8–16k lanes
/// on an M2 Pro; each lane keeps claiming sources until its event budget ends.
const PERSISTENT_LANES: usize = 16384;
const STACK_BYTES: usize = 256 << 20;
/// Rounds per command and the lane event budget adapt to this device time.
const TARGET_COMMAND_SECONDS: f64 = 0.015;

/// Work-split study key of the transport kernel designs.
pub const KERNEL_STUDY_KEY: &str = "stray_kernel";
/// Feature bits whose state is large (split children, thin-film matrices,
/// iterative sag intersection) and that `partial` keeps specialised.
const HEAVY_FEATURES: u32 = 2 | 4 | 8;
/// Every feature bit of `MKL_STRAY_FEATURES_CONSTANT`.
const ALL_FEATURES: u32 = 0xff;

/// A design of the transport kernel: which scene features are function
/// constants (one pipeline per combination) and which are uniform branches
/// on the run's feature word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelDesign {
    /// Every feature a constant: up to 108 pipelines.
    Specialized,
    /// Split mode, thin films and extended sags constants (8 pipelines);
    /// analog/weighted, DOE presence and order kind, scatter presence and
    /// BSDF kind are branches.
    Partial,
    /// One pipeline; every feature is a branch.
    Uber,
}

impl KernelDesign {
    /// The design selected for this run.
    pub fn selected() -> MkResult<KernelDesign> {
        Ok(
            match mklight_core::split::variant(KERNEL_STUDY_KEY).as_deref() {
                None | Some("specialized") => KernelDesign::Specialized,
                Some("partial") => KernelDesign::Partial,
                Some("uber") => KernelDesign::Uber,
                Some(other) => {
                    return Err(MkError::InvalidModel(format!(
                        "unknown stray kernel design '{other}' (specialized, partial, uber)"
                    )));
                }
            },
        )
    }
    pub fn name(self) -> &'static str {
        match self {
            KernelDesign::Specialized => "specialized",
            KernelDesign::Partial => "partial",
            KernelDesign::Uber => "uber",
        }
    }
    /// Feature bits read at run time.
    pub fn dynamic(self) -> u32 {
        match self {
            KernelDesign::Specialized => 0,
            KernelDesign::Partial => ALL_FEATURES & !HEAVY_FEATURES,
            KernelDesign::Uber => ALL_FEATURES,
        }
    }
    /// The pipeline for a run with scene `features`; the constant holds only
    /// the specialised bits, so runs that differ in dynamic bits share it.
    pub fn pipeline(self, features: u32) -> PipelineKey {
        self.pipeline_of("mkl_stray_transport", features)
    }
    /// As [`Self::pipeline`] for another kernel that reads the features.
    pub fn pipeline_of(self, function: &'static str, features: u32) -> PipelineKey {
        let constant = features & !self.dynamic();
        let mut key = PipelineKey::with_u32(
            function,
            mklight_gpu::consts::MKL_STRAY_FEATURES_CONSTANT as u16,
            constant,
        );
        if self != KernelDesign::Specialized {
            key.constants.push(crate::context::FunctionConstant::U32 {
                index: mklight_gpu::consts::MKL_STRAY_DYNAMIC_CONSTANT as u16,
                value: self.dynamic(),
            });
        }
        key
    }
}

pub(super) fn execute(r: &Transport, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let source_pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_generate"))?;
    // Immutable topology/material features specialize away unused large local
    // split, thin-film, sag and scatter state without changing the algebra.
    let (mb, mo) = x.buffer(&r.materials);
    let materials: &[Material] =
        bytemuck::cast_slice(&mb.as_slice()[mo..mo + r.materials.desc.len()?]);
    let (pb, po) = x.buffer(&r.primitives);
    let prims: &[Primitive] =
        bytemuck::cast_slice(&pb.as_slice()[po..po + r.primitives.desc.len()?]);
    let features = r.settings.mode
        | (materials
            .iter()
            .any(|m| m.coatings.iter().any(|c| c.kind == 3)) as u32)
            << 2
        | (prims.iter().any(|p| p.kind == 3) as u32) << 3
        | (materials.iter().any(|m| m.doe != 0) as u32) << 4
        | (materials.iter().any(|m| m.bsdf_kind != 0) as u32) << 5
        | (materials.iter().any(|m| m.bsdf_kind >= 2) as u32) << 6
        | (materials.iter().any(|m| m.doe >= 2) as u32) << 7;
    let design = KernelDesign::selected()?;
    if r.wavefront {
        return execute_wavefront(r, x, ctl, features, design, start);
    }
    let transport = x.gpu.pipeline(&design.pipeline(features))?;
    let spill = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_spill"))?;
    let reset = x
        .gpu
        .pipeline(&PipelineKey::plain("mkl_stray_queue_reset"))?;
    let n = r.ray_count;
    let threads = n.min(PERSISTENT_LANES);
    let mut capacity = r.stack_capacity.max(4);
    let words = |size: usize, count: usize| size / 4 * count;
    let mut rays = x.allocate(words(std::mem::size_of::<Ray>(), n))?;
    let stack_words = words(std::mem::size_of::<State>(), n * capacity);
    if stack_words * 4 > STACK_BYTES {
        return unsupported("stray initial stack exceeds 256 MiB");
    }
    // Handoff mode: the stack is a plan output of fixed capacity.
    let handoff = r.stack_output.is_some();
    let mut stack = match r.stack_output {
        Some(id) => x.take_output(id),
        None => x.allocate(stack_words)?,
    };
    let mut cdf = x.allocate(2 * r.cdf.len())?;
    cdf.as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(&r.cdf));
    let mut queue = x.allocate(8)?;
    queue.as_mut_slice().fill(0);
    let mut lists = x.allocate(2 * n)?;
    let mut hits = x.take_output(r.output);
    let mut ledger = x.take_output(r.ledger);
    let mut stats = StageStats {
        name: format!("stray transport ({} kernel)", design.name()),
        ..Default::default()
    };
    let (mut parity, mut rounds, mut commands) = (0u32, 1usize, 0u64);
    let mut budget = r.event_budget.min(4);
    loop {
        ctl.check()?;
        let wall = Instant::now();
        // SAFETY: all records are boundary-packed, source-local stack slices
        // are disjoint, queue claims are unique and no lease is published
        // until every source is done.
        let cs = unsafe {
            x.gpu.run("stray transport", |e| {
                if commands == 0 {
                    let gp = [
                        n as u32,
                        0,
                        r.seed as u32,
                        (r.seed >> 32) as u32,
                        r.first_ray as u32,
                        (r.first_ray >> 32) as u32,
                        0,
                        0,
                    ];
                    e.set_pipeline(&source_pso);
                    e.set_bytes(0, &r.source);
                    e.set_buffer(1, &cdf, 0);
                    e.set_buffer_mut(2, &mut rays, 0);
                    e.set_bytes(3, &gp);
                    e.dispatch_groups([n.div_ceil(64), 1, 1], 64);
                }
                for round in 0..rounds {
                    let par = parity ^ (round as u32 & 1);
                    e.set_pipeline(&reset);
                    e.set_buffer_mut(0, &mut queue, 0);
                    e.set_bytes(1, &par);
                    e.dispatch_groups([1, 1, 1], 1);
                    e.set_pipeline(&transport);
                    for (i, v) in [
                        &r.primitives,
                        &r.nodes,
                        &r.materials,
                        &r.layers,
                        &r.passes,
                        &r.media,
                        &r.tis,
                        &r.sequence,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        let (b, o) = x.buffer(v);
                        e.set_buffer(i, b, o);
                    }
                    e.set_buffer(8, &rays, 0);
                    e.set_buffer_mut(9, &mut stack, 0);
                    e.set_buffer_mut(10, &mut hits, 0);
                    e.set_buffer_mut(11, &mut ledger, 0);
                    e.set_bytes(12, &r.settings);
                    e.set_bytes(
                        13,
                        &[
                            n as u32,
                            r.nodes.desc.shape[0] as u32,
                            capacity as u32,
                            r.hit_capacity as u32,
                            budget as u32,
                            par,
                            r.seed as u32,
                            (r.seed >> 32) as u32,
                            threads as u32,
                            r.head_events,
                            handoff as u32,
                            features | r.sag_mode << 8,
                        ],
                    );
                    let (b, o) = x.buffer(&r.shapes);
                    e.set_buffer(14, b, o);
                    e.set_buffer_mut(15, &mut queue, 0);
                    e.set_buffer_mut(16, &mut lists, 0);
                    e.dispatch_groups([threads.div_ceil(32), 1, 1], 32);
                }
                Ok(())
            })?
        };
        let seconds = wall.elapsed().as_secs_f64();
        parity ^= rounds as u32 & 1;
        commands += 1;
        x.stats.parameter_bytes += (rounds * (std::mem::size_of::<Settings>() + 52)) as u64;
        stats.device_seconds += cs.gpu_seconds;
        stats.command_buffers += 1;
        stats.max_command_seconds = stats.max_command_seconds.max(seconds);
        ctl.check()?;
        let q = queue.as_slice();
        let (fresh, resumed, status) = (q[0] as usize, q[1 + parity as usize] as usize, q[4]);
        // In handoff mode every stopped source carries its own status.
        if status & 16 != 0 && !handoff {
            return unsupported("stray coating/transport arithmetic requires reference");
        }
        if status & 8 != 0 && !handoff {
            return unsupported("stray detection capacity; no partial result published");
        }
        if fresh >= n && resumed == 0 {
            break;
        }
        if status & 4 != 0 && !handoff {
            let next_capacity = capacity * 2;
            let size = words(std::mem::size_of::<State>(), n * next_capacity);
            if size * 4 > STACK_BYTES {
                return Err(MkError::Unsupported(
                    "stray continuation stack exceeds 256 MiB; no paths discarded".into(),
                ));
            }
            let mut next = x.allocate(size)?;
            let spill_wall = Instant::now();
            let offset = parity as usize * n;
            let cs = unsafe {
                x.gpu.run("stray stack spill", |e| {
                    e.set_pipeline(&spill);
                    e.set_buffer(0, &stack, 0);
                    e.set_buffer_mut(1, &mut next, 0);
                    e.set_buffer(2, &ledger, 0);
                    e.set_bytes(
                        3,
                        &[resumed as u32, capacity as u32, next_capacity as u32, 0],
                    );
                    e.set_buffer(4, &lists, offset);
                    e.dispatch_groups([resumed.div_ceil(64), 1, 1], 64);
                    Ok(())
                })?
            };
            let wall = spill_wall.elapsed().as_secs_f64();
            x.stats.device_seconds += cs.gpu_seconds;
            x.stats.command_buffers += 1;
            x.stats.parameter_bytes += 16;
            x.stats.stages.push(StageStats {
                name: "stray stack spill".into(),
                wall_seconds: wall,
                device_seconds: cs.gpu_seconds,
                command_buffers: 1,
                max_command_seconds: wall,
                ..Default::default()
            });
            x.recycle(stack);
            stack = next;
            capacity = next_capacity;
            queue.as_mut_slice()[4] = 0;
        }
        // Keep each command near the latency target: shorter rounds when one
        // overruns it, otherwise more rounds per command.
        let round = cs.gpu_seconds.max(1e-5) / rounds as f64;
        if round > TARGET_COMMAND_SECONDS && budget > 1 {
            budget /= 2;
            rounds = 1;
        } else {
            rounds = (TARGET_COMMAND_SECONDS / round).clamp(1., 64.) as usize;
        }
        // Claimed sources are reported once complete; reserve a tail for
        // long split trees instead of guessing their final path count.
        let done = fresh.min(n) - resumed;
        ctl.report(
            (0.9 * done as f64 + 0.09 * n as f64 * (1. - 1. / (commands + 1) as f64)) / n as f64,
            "stray continuation",
        );
    }
    ctl.report(1., "stray transport");
    match r.stack_output {
        Some(id) => x.put_output(id, stack),
        None => x.recycle(stack),
    }
    for b in [rays, cdf, queue, lists] {
        x.recycle(b);
    }
    x.put_output(r.output, hits);
    x.put_output(r.ledger, ledger);
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}

/// Wavefront transport (handoff mode): per bounce, every live source pops
/// its next path (`mkl_stray_wave_pop`), casts it (`mkl_stray_wave_cast`)
/// and interacts (`mkl_stray_wave_interact`); live-source lists are
/// compacted on the device and each kernel is dispatched indirectly over
/// them. Several bounces share a command; only the live count is read back.
fn execute_wavefront(
    r: &Transport,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    features: u32,
    design: KernelDesign,
    start: Instant,
) -> MkResult<()> {
    let Some(stack_id) = r.stack_output else {
        return unsupported("stray wavefront needs handoff mode");
    };
    let source_pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_generate"))?;
    let init = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_wave_init"))?;
    let args_pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_wave_args"))?;
    let pop = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_wave_pop"))?;
    let cast = x
        .gpu
        .pipeline(&design.pipeline_of("mkl_stray_wave_cast", features))?;
    let interact = x
        .gpu
        .pipeline(&design.pipeline_of("mkl_stray_wave_interact", features))?;
    let widths = [&pop, &cast, &interact].map(|p| crate::context::Encoder::width(p) as u32);
    let n = r.ray_count;
    let words = |size: usize, count: usize| size / 4 * count;
    let mut rays = x.allocate(words(std::mem::size_of::<Ray>(), n))?;
    let mut current = x.allocate(words(std::mem::size_of::<State>(), n))?;
    let mut found = x.allocate(words(std::mem::size_of::<Hit>(), n))?;
    let mut cdf = x.allocate(2 * r.cdf.len())?;
    cdf.as_mut_slice()
        .copy_from_slice(bytemuck::cast_slice(&r.cdf));
    let mut lists = x.allocate(2 * n)?;
    let mut counters = x.allocate(4)?;
    counters.as_mut_slice().fill(0);
    let mut args = x.allocate(12)?;
    args.as_mut_slice().fill(0);
    let mut stack = x.take_output(stack_id);
    let mut hits = x.take_output(r.output);
    let mut ledger = x.take_output(r.ledger);
    let mut stats = StageStats {
        name: format!("stray wavefront ({} kernel)", design.name()),
        ..Default::default()
    };
    let params = |parity: u32| {
        [
            n as u32,
            r.nodes.desc.shape[0] as u32,
            r.stack_capacity as u32,
            r.hit_capacity as u32,
            0,
            parity,
            r.seed as u32,
            (r.seed >> 32) as u32,
            0,
            r.head_events,
            1,
            features | r.sag_mode << 8,
        ]
    };
    // Encodes the sources' generation (`first`) and then `bounces` bounces
    // from list `parity`, each of the phases in `phases` (bits: pop, cast,
    // interact; all three except when profiling the kernels).
    let gpu = x.gpu;
    let reads = |v: &mklight_gpu::resident::PlanView| x.buffer(v);
    let mut encode = |e: &crate::context::Encoder<'_>,
                      counters: &mut crate::resident::Lease,
                      first: bool,
                      bounces: usize,
                      parity: u32,
                      phases: u32| {
        if first {
            let gp = [
                n as u32,
                0,
                r.seed as u32,
                (r.seed >> 32) as u32,
                r.first_ray as u32,
                (r.first_ray >> 32) as u32,
                0,
                0,
            ];
            e.set_pipeline(&source_pso);
            e.set_bytes(0, &r.source);
            e.set_buffer(1, &cdf, 0);
            e.set_buffer_mut(2, &mut rays, 0);
            e.set_bytes(3, &gp);
            e.dispatch_groups([n.div_ceil(64), 1, 1], 64);
            e.set_pipeline(&init);
            e.set_buffer(0, &rays, 0);
            e.set_buffer_mut(1, &mut stack, 0);
            e.set_buffer_mut(2, &mut ledger, 0);
            e.set_bytes(3, &r.settings);
            e.set_bytes(4, &params(0));
            e.set_buffer_mut(5, &mut lists, 0);
            e.set_buffer_mut(6, &mut *counters, 0);
            e.dispatch_groups([n.div_ceil(64), 1, 1], 64);
        }
        for b in 0..bounces {
            let par = parity ^ (b as u32 & 1);
            let p = params(par);
            if phases & 1 != 0 {
                // Pop: list `par` into list `1 - par` (cleared first).
                e.set_pipeline(&args_pso);
                e.set_buffer_mut(0, &mut *counters, 0);
                e.set_buffer_mut(1, &mut args, 0);
                e.set_bytes(2, &[par, widths[0], 0, 1 - par]);
                e.dispatch_groups([1, 1, 1], 1);
                e.set_pipeline(&pop);
                e.set_buffer_mut(0, &mut stack, 0);
                e.set_buffer_mut(1, &mut ledger, 0);
                e.set_buffer_mut(2, &mut current, 0);
                e.set_bytes(3, &r.settings);
                e.set_bytes(4, &p);
                e.set_buffer_mut(5, &mut lists, 0);
                e.set_buffer_mut(6, &mut *counters, 0);
                e.dispatch_indirect(&pop, &args, 0);
                // Arguments of the cast and interaction over list `1 - par`.
                e.set_pipeline(&args_pso);
                e.set_buffer_mut(0, &mut *counters, 0);
                e.set_buffer_mut(1, &mut args, 0);
                e.set_bytes(2, &[1 - par, widths[1], 3, 2]);
                e.dispatch_groups([1, 1, 1], 1);
                e.set_bytes(2, &[1 - par, widths[2], 6, 2]);
                e.dispatch_groups([1, 1, 1], 1);
            }
            if phases & 2 != 0 {
                e.set_pipeline(&cast);
                for (i, v) in [&r.primitives, &r.nodes, &r.shapes].into_iter().enumerate() {
                    let (b, o) = reads(v);
                    e.set_buffer(i, b, o);
                }
                e.set_buffer(3, &current, 0);
                e.set_buffer_mut(4, &mut found, 0);
                e.set_bytes(5, &p);
                e.set_buffer(6, &lists, 0);
                e.set_buffer(7, &*counters, 0);
                e.dispatch_indirect(&cast, &args, 3);
            }
            if phases & 4 != 0 {
                e.set_pipeline(&interact);
                for (i, v) in [
                    &r.primitives,
                    &r.materials,
                    &r.layers,
                    &r.passes,
                    &r.media,
                    &r.tis,
                    &r.sequence,
                ]
                .into_iter()
                .enumerate()
                {
                    let (b, o) = reads(v);
                    e.set_buffer(i, b, o);
                }
                e.set_buffer_mut(7, &mut stack, 0);
                e.set_buffer_mut(8, &mut hits, 0);
                e.set_buffer_mut(9, &mut ledger, 0);
                e.set_bytes(10, &r.settings);
                e.set_bytes(11, &p);
                e.set_buffer(12, &current, 0);
                e.set_buffer(13, &found, 0);
                e.set_buffer(14, &lists, 0);
                e.set_buffer(15, &*counters, 0);
                e.dispatch_indirect(&interact, &args, 6);
            }
        }
        Ok(())
    };
    // Diagnostics: MKLIGHT_STRAY_TRACE prints each command;
    // MKLIGHT_STRAY_TRACE=kernels also runs each kernel phase in its own
    // command and accumulates its device time.
    let trace = std::env::var("MKLIGHT_STRAY_TRACE").ok();
    let profile = trace.as_deref() == Some("kernels");
    let mut phase_seconds = [0f64; 3];
    let (mut parity, mut bounces, mut commands, mut parameter_bytes) = (0u32, 4usize, 0u64, 0u64);
    loop {
        ctl.check()?;
        let wall = Instant::now();
        // SAFETY: records are boundary-packed; every kernel bounds its
        // thread index by the device count of its list, list entries are
        // distinct sources, and source-local stack and journal slices are
        // disjoint.
        let gpu_seconds = if profile {
            let mut total = 0.;
            for (k, mask) in [1u32, 2, 4].into_iter().enumerate() {
                let cs = unsafe {
                    gpu.run("stray wavefront phase", |e| {
                        encode(e, &mut counters, commands == 0 && k == 0, 1, parity, mask)
                    })?
                };
                phase_seconds[k] += cs.gpu_seconds;
                total += cs.gpu_seconds;
            }
            bounces = 1;
            total
        } else {
            unsafe {
                gpu.run("stray wavefront", |e| {
                    encode(e, &mut counters, commands == 0, bounces, parity, 7)
                })?
            }
            .gpu_seconds
        };
        let seconds = wall.elapsed().as_secs_f64();
        parity ^= bounces as u32 & 1;
        commands += 1;
        parameter_bytes += (bounces * (2 * std::mem::size_of::<Settings>() + 180)) as u64;
        stats.device_seconds += gpu_seconds;
        stats.command_buffers += if profile { 3 } else { 1 };
        stats.max_command_seconds = stats.max_command_seconds.max(seconds);
        let live = counters.as_slice()[parity as usize] as usize;
        if trace.is_some() {
            eprintln!(
                "stray wave cmd {commands}: n {n} bounces {bounces} live {live} gpu {:.2} ms \
                 (phases pop/cast/interact {:.1}/{:.1}/{:.1} ms)",
                gpu_seconds * 1e3,
                phase_seconds[0] * 1e3,
                phase_seconds[1] * 1e3,
                phase_seconds[2] * 1e3
            );
        }
        if live == 0 {
            break;
        }
        // About TARGET_COMMAND_SECONDS of device time per command.
        let per = gpu_seconds.max(1e-5) / bounces as f64;
        bounces = (TARGET_COMMAND_SECONDS / per).clamp(1., 256.) as usize;
        ctl.report(1. - live as f64 / n as f64, "stray wavefront");
    }
    drop(encode);
    x.stats.parameter_bytes += parameter_bytes;
    ctl.report(1., "stray transport");
    x.put_output(stack_id, stack);
    for b in [rays, current, found, cdf, lists, counters, args] {
        x.recycle(b);
    }
    x.put_output(r.output, hits);
    x.put_output(r.ledger, ledger);
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}

pub(super) fn build(r: &Build, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_stray_build"))?;
    let n = r.primitives.desc.shape[0];
    let size = n.next_power_of_two();
    let depth = size.ilog2();
    let mut out = x.take_output(r.output);
    let start = Instant::now();
    let mut stats = StageStats {
        name: "stray BVH build".into(),
        ..Default::default()
    };
    for level in (0..=depth).rev() {
        ctl.check()?;
        let wall = Instant::now();
        let cs = unsafe {
            x.gpu.run("stray BVH level", |e| {
                e.set_pipeline(&pso);
                let (b, o) = x.buffer(&r.primitives);
                e.set_buffer(0, b, o);
                e.set_buffer_mut(1, &mut out, 0);
                e.set_bytes(2, &[n as u32, size as u32, level, 0]);
                e.dispatch_groups([(1usize << level).div_ceil(64), 1, 1], 64);
                Ok(())
            })?
        };
        stats.device_seconds += cs.gpu_seconds;
        stats.command_buffers += 1;
        stats.max_command_seconds = stats.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.check()?;
        ctl.report(
            (depth - level + 1) as f64 / (depth + 1) as f64,
            "stray BVH build",
        );
    }
    x.put_output(r.output, out);
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{MetalContext, MetalOptions};
    use objc2_metal::MTLComputePipelineState;

    /// Occupancy evidence of the kernel designs (ignored; prints a table):
    /// maximum threads per threadgroup (register pressure), SIMD width and
    /// the pipeline build time, for the stray transport and the ghost trace.
    #[test]
    #[ignore]
    fn kernel_design_occupancy() {
        let Some(ctx) = MetalContext::new(&MetalOptions::default()).unwrap() else {
            return;
        };
        let report = |label: String, key: PipelineKey| {
            let t = Instant::now();
            let pso = ctx.pipeline(&key).unwrap();
            println!(
                "{label:<52} max threads {:>4}  width {}  build {:6.2} s",
                pso.maxTotalThreadsPerThreadgroup(),
                pso.threadExecutionWidth(),
                t.elapsed().as_secs_f64()
            );
        };
        let scenes = [
            ("weighted plain", 1),
            ("weighted, Lambertian scatter", 1 | 32),
            ("weighted, Gaussian/ABg scatter", 1 | 32 | 64),
            ("weighted, scatter, films", 1 | 32 | 4),
            ("weighted, scatter, aspheres", 1 | 32 | 8),
            ("weighted, scatter, DOE", 1 | 32 | 16),
            ("weighted, scatter, kinoform DOE", 1 | 32 | 16 | 128),
            ("split, scatter", 2 | 32),
            ("analog, scatter", 32),
            ("everything (split)", ALL_FEATURES & !1),
        ];
        for design in [
            KernelDesign::Specialized,
            KernelDesign::Partial,
            KernelDesign::Uber,
        ] {
            for (name, features) in scenes {
                report(
                    format!("stray {} / {name}", design.name()),
                    design.pipeline(features),
                );
            }
        }
        use crate::ghost_ops::KernelDesign as G;
        use mklight_gpu::consts::*;
        for design in [G::Specialized, G::Partial, G::Uber] {
            for (name, features) in [
                ("geometry only", 0),
                ("coatings", MKL_GHOST_FEATURE_COATINGS),
                (
                    "coatings, films",
                    MKL_GHOST_FEATURE_COATINGS | MKL_GHOST_FEATURE_FILMS,
                ),
            ] {
                report(
                    format!("ghost {design:?} / {name}"),
                    design.pipeline(features),
                );
            }
        }
    }
}
