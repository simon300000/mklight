//! Resident Metal collectives. Dispatch boundaries only partition a fixed tree.
use crate::{
    context::{Encoder, PipelineKey},
    resident::{Execution, Lease},
};
use mklight_core::MkResult;
use mklight_gpu::{
    collectives::*,
    pending::JobControl,
    reduce,
    resident::{ElementType as Ty, PlanView, StageStats},
};
use std::time::Instant;
type Params = [u32; 16];
const CHUNK: usize = 65536;

// Each operation contributes its measured substage; the runtime also appends
// its enclosing domain stage (wall time, with no duplicate command counts).
fn launch(
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
    name: &str,
    p: &Params,
    encode: impl FnOnce(&Encoder<'_>, &Execution<'_>) -> MkResult<()>,
) -> MkResult<()> {
    ctl.check()?;
    let start = Instant::now();
    // SAFETY: request validation established types/extents and single writers.
    // Each dispatch below bounds its flattened indices to those extents.
    let cs = unsafe { x.gpu.run(name, |e| encode(e, x))? };
    stats.device_seconds += cs.gpu_seconds;
    stats.command_buffers += 1;
    stats.max_command_seconds = stats.max_command_seconds.max(start.elapsed().as_secs_f64());
    x.stats.parameter_bytes += std::mem::size_of_val(p) as u64;
    ctl.check()
}
fn bind(e: &Encoder<'_>, slot: usize, x: &Execution<'_>, v: &PlanView) {
    let (b, o) = x.buffer(v);
    e.set_buffer(slot, b, o);
}
fn kind(t: Ty) -> u32 {
    match t {
        Ty::U32 => 0,
        Ty::F32 => 1,
        _ => 2,
    }
}
fn ranges(n: usize) -> impl Iterator<Item = (usize, usize)> {
    (0..n).step_by(CHUNK).map(move |i| (i, CHUNK.min(n - i)))
}
fn initialise(
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
    src: &Lease,
    dst: &mut Lease,
    n: usize,
    op: u32,
    words: usize,
    inclusive: bool,
) -> MkResult<()> {
    let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_c_init"))?;
    for (i, len) in ranges(n) {
        let p = [
            n as u32,
            i as u32,
            len as u32,
            op,
            words as u32,
            u32::from(inclusive),
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        launch(x, ctl, stats, "collective initialise", &p, |e, _| {
            e.set_bytes(2, &p);
            e.set_buffer(0, src, 0);
            e.set_buffer_mut(1, dst, 0);
            e.dispatch_1d(&pso, len);
            Ok(())
        })?;
    }
    Ok(())
}
#[derive(Clone, Copy)]
enum Input<'a> {
    View(&'a PlanView),
    Buffer(&'a crate::buffer::SharedBuffer<u32>, usize),
}
fn scan(
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
    input: Input<'_>,
    n: usize,
    ty: u32,
    out: &mut Lease,
) -> MkResult<()> {
    let words = if ty == 0 { 1 } else { 2 };
    let blocks = n.div_ceil(256);
    let mut totals = x.allocate(blocks * words)?;
    let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_c_scan"))?;
    for (first, count) in (0..blocks)
        .step_by(CHUNK / 256)
        .map(|i| (i, (CHUNK / 256).min(blocks - i)))
    {
        let p = [
            n as u32,
            first as u32,
            ty,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        launch(x, ctl, stats, "collective block scan", &p, |e, x| {
            e.set_pipeline(&pso);
            match input {
                Input::View(v) => bind(e, 0, x, v),
                Input::Buffer(b, o) => e.set_buffer(0, b, o),
            }
            e.set_buffer_mut(1, out, 0);
            e.set_buffer_mut(2, &mut totals, 0);
            e.set_bytes(3, &p);
            e.dispatch_groups([count, 1, 1], 256);
            Ok(())
        })?;
        ctl.report(
            0.5 * (first + count) as f64 / blocks.max(1) as f64,
            "scan blocks",
        );
    }
    if blocks > 1 {
        let mut prefixes = x.allocate(blocks * words)?;
        scan(
            x,
            &ctl.subrange(0.5, 0.75),
            stats,
            Input::Buffer(&totals, 0),
            blocks,
            if ty == 0 { 0 } else { 2 },
            &mut prefixes,
        )?;
        let pso = x.gpu.pipeline(&PipelineKey::plain("mkl_c_scan_add"))?;
        for (i, len) in ranges(n) {
            let p = [
                n as u32,
                i as u32,
                len as u32,
                u32::from(ty == 0),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ];
            launch(x, ctl, stats, "collective scan prefixes", &p, |e, _| {
                e.set_buffer_mut(0, out, 0);
                e.set_buffer(1, &prefixes, 0);
                e.set_bytes(2, &p);
                e.dispatch_1d(&pso, len);
                Ok(())
            })?;
            ctl.report(0.75 + 0.25 * (i + len) as f64 / n as f64, "scan prefixes");
        }
        x.recycle(prefixes);
    }
    x.recycle(totals);
    Ok(())
}
pub(crate) fn execute(r: &Request, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let start = Instant::now();
    let mut stats = StageStats {
        operation: x.stats.stages.last().map(|s| s.operation + 1).unwrap_or(0),
        name: "collectives".into(),
        ..Default::default()
    };
    match r {
        Request::Scan(r) => {
            let n = r.values.desc.shape[0];
            let words = if r.values.desc.element == Ty::U32 {
                1
            } else {
                2
            };
            let mut out = x.take_output(r.output);
            let mut tmp = x.allocate(n * words)?;
            scan(
                x,
                &ctl.subrange(0.0, 0.9),
                &mut stats,
                Input::View(&r.values),
                n,
                kind(r.values.desc.element),
                &mut tmp,
            )?;
            initialise(x, ctl, &mut stats, &tmp, &mut out, n, 2, words, r.inclusive)?;
            x.recycle(tmp);
            x.put_output(r.output, out);
        }
        Request::StableCompact(r) => compact(r, x, ctl, &mut stats)?,
        Request::StableSort(r) => sort(r, x, ctl, &mut stats)?,
        Request::SegmentReduce(r) => segments(r, x, ctl, &mut stats)?,
        Request::QrFit(r) => qr(r, x, ctl, &mut stats)?,
    }
    stats.wall_seconds = start.elapsed().as_secs_f64();
    x.stats.device_seconds += stats.device_seconds;
    x.stats.command_buffers += stats.command_buffers;
    x.stats.stages.push(stats);
    ctl.check()?;
    ctl.report(1.0, "collectives");
    Ok(())
}
fn compact(
    r: &StableCompact,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
) -> MkResult<()> {
    let n = r.ids.desc.shape[0];
    let width = if n == 0 {
        0
    } else {
        r.values.desc.len()? * r.values.desc.element.words() / n
    };
    let mut out = x.take_output(r.output);
    let mut ids = x.take_output(r.output_ids);
    let mut count = x.take_output(r.count);
    let mut mask = x.allocate(n)?;
    let mut prefix = x.allocate(n)?;
    let init = x.gpu.pipeline(&PipelineKey::plain("mkl_c_init"))?;
    for (i, len) in ranges(n) {
        let p = [
            n as u32, i as u32, len as u32, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ];
        launch(x, ctl, stats, "compact mask", &p, |e, x| {
            bind(e, 0, x, &r.mask);
            e.set_buffer_mut(1, &mut mask, 0);
            e.set_bytes(2, &p);
            e.dispatch_1d(&init, len);
            Ok(())
        })?;
    }
    scan(
        x,
        &ctl.subrange(0.0, 0.4),
        stats,
        Input::Buffer(&mask, 0),
        n,
        0,
        &mut prefix,
    )?;
    let clear = x.gpu.pipeline(&PipelineKey::plain("mkl_c_compact_clear"))?;
    let scatter = x.gpu.pipeline(&PipelineKey::plain("mkl_c_compact"))?;
    // Empty output still publishes a defined count, through a device write.
    if n == 0 {
        let p = [1, 0, 1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        launch(x, ctl, stats, "empty compact", &p, |e, _| {
            e.set_buffer(0, &prefix, 0);
            e.set_buffer_mut(1, &mut count, 0);
            e.set_bytes(2, &p);
            e.dispatch_1d(&init, 1);
            Ok(())
        })?;
    }
    for (i, len) in ranges(n) {
        let p = [
            n as u32,
            i as u32,
            len as u32,
            width as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        launch(x, ctl, stats, "compact tail", &p, |e, _| {
            e.set_buffer_mut(0, &mut out, 0);
            e.set_buffer_mut(1, &mut ids, 0);
            e.set_buffer_mut(2, &mut count, 0);
            e.set_buffer(3, &prefix, 0);
            e.set_bytes(4, &p);
            e.dispatch_1d(&clear, len);
            Ok(())
        })?;
        ctl.report(0.4 + 0.1 * (i + len) as f64 / n as f64, "compact prefix");
    }
    for (i, len) in ranges(n) {
        let p = [
            n as u32,
            i as u32,
            len as u32,
            width as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        launch(x, ctl, stats, "compact scatter", &p, |e, x| {
            bind(e, 0, x, &r.values);
            bind(e, 1, x, &r.ids);
            bind(e, 2, x, &r.mask);
            e.set_buffer(3, &prefix, 0);
            e.set_buffer_mut(4, &mut out, 0);
            e.set_buffer_mut(5, &mut ids, 0);
            e.set_bytes(6, &p);
            e.dispatch_1d(&scatter, len);
            Ok(())
        })?;
        ctl.report(0.5 + 0.5 * (i + len) as f64 / n as f64, "compact scatter");
    }
    x.recycle(mask);
    x.recycle(prefix);
    x.put_output(r.output, out);
    x.put_output(r.output_ids, ids);
    x.put_output(r.count, count);
    Ok(())
}
fn sort(
    r: &StableSort,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
) -> MkResult<()> {
    let n = r.ids.desc.shape[0];
    let mut a = x.allocate(n)?;
    let mut b = x.allocate(n)?;
    initialise(x, ctl, stats, &b, &mut a, n, 1, 1, true)?;
    let merge = x.gpu.pipeline(&PipelineKey::plain("mkl_c_merge"))?;
    let mut width = 1;
    let levels = n.max(1).next_power_of_two().ilog2().max(1) as f64;
    let mut level = 0;
    while width < n {
        for (i, len) in ranges(n) {
            let p = [
                n as u32,
                i as u32,
                len as u32,
                width as u32,
                kind(r.keys.desc.element),
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ];
            launch(x, ctl, stats, "stable merge rank", &p, |e, x| {
                bind(e, 0, x, &r.keys);
                bind(e, 1, x, &r.ids);
                e.set_buffer(2, &a, 0);
                e.set_buffer_mut(3, &mut b, 0);
                e.set_bytes(4, &p);
                e.dispatch_1d(&merge, len);
                Ok(())
            })?;
            ctl.report(
                (level as f64 + (i + len) as f64 / n as f64) / (levels + 1.0),
                "stable sort",
            );
        }
        std::mem::swap(&mut a, &mut b);
        width *= 2;
        level += 1;
    }
    let mut keys = x.take_output(r.output_keys);
    let mut ids = x.take_output(r.output_ids);
    let mut vals = if let Some(o) = r.output_values {
        x.take_output(o)
    } else {
        x.allocate(0)?
    };
    let vw = r
        .values
        .as_ref()
        .map(|v| {
            if n == 0 {
                0
            } else {
                v.desc.len().unwrap() * v.desc.element.words() / n
            }
        })
        .unwrap_or(0);
    let gather = x.gpu.pipeline(&PipelineKey::plain("mkl_c_gather"))?;
    for (i, len) in ranges(n) {
        let p = [
            n as u32,
            i as u32,
            len as u32,
            r.keys.desc.element.words() as u32,
            vw as u32,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        launch(x, ctl, stats, "stable sort gather", &p, |e, x| {
            bind(e, 0, x, &r.keys);
            bind(e, 1, x, &r.ids);
            if let Some(v) = &r.values {
                bind(e, 2, x, v);
            } else {
                e.set_buffer(2, &b, 0);
            }
            e.set_buffer(3, &a, 0);
            e.set_buffer_mut(4, &mut keys, 0);
            e.set_buffer_mut(5, &mut ids, 0);
            e.set_buffer_mut(6, &mut vals, 0);
            e.set_bytes(7, &p);
            e.dispatch_1d(&gather, len);
            Ok(())
        })?;
    }
    x.recycle(a);
    x.recycle(b);
    x.put_output(r.output_keys, keys);
    x.put_output(r.output_ids, ids);
    if let Some(o) = r.output_values {
        x.put_output(o, vals);
    } else {
        x.recycle(vals);
    }
    Ok(())
}
fn segments(
    r: &SegmentReduce,
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
) -> MkResult<()> {
    let [c, n] = r.values.desc.shape[..] else {
        unreachable!()
    };
    let s = r.offsets.desc.shape[0] - 1;
    if s == 0 {
        return Ok(());
    }
    let maxg = reduce::num_groups(r.max_segment_len);
    let mut out = x.take_output(r.output);
    let mut status = x.take_output(r.status);
    let p1 = x.gpu.pipeline(&PipelineKey::plain("mkl_c_seg1"))?;
    let p2 = x.gpu.pipeline(&PipelineKey::plain("mkl_c_seg2"))?;
    let finish = x.gpu.pipeline(&PipelineKey::plain("mkl_c_seg_finish"))?;
    let phases = if r.kind == SegmentKind::Moments2 {
        2
    } else {
        1
    };
    for phase in 0..phases {
        let terms = match r.kind {
            SegmentKind::Sum | SegmentKind::SourceProducts => c,
            SegmentKind::Summary => c * 4,
            SegmentKind::Moments2 => {
                if phase == 0 {
                    4
                } else {
                    6
                }
            }
        };
        // At most 16 MiB of partials, including detector maps with many bins.
        let tile = (16 * 1024 * 1024 / (terms * maxg * 8)).clamp(1, (64 / maxg.min(64)).max(1));
        let mut partials = x.allocate(tile * terms * maxg * 2)?;
        for start in (0..s).step_by(tile) {
            let ns = tile.min(s - start);
            let mut p = [
                n as u32,
                c as u32,
                s as u32,
                start as u32,
                ns as u32,
                maxg as u32,
                terms as u32,
                phase as u32,
                r.kind as u32,
                r.mode as u32,
                r.values.desc.element.words() as u32,
                r.reference
                    .as_ref()
                    .map(|v| v.desc.element.words() as u32)
                    .unwrap_or(0),
                r.max_segment_len as u32,
                0,
                0,
                0,
            ];
            p[15] = r.mask_value;
            let options = [
                u32::from(r.mask.is_some()),
                r.weights
                    .as_ref()
                    .map(|v| v.desc.element.words() as u32)
                    .unwrap_or(0),
            ];
            // Partition group ranges without changing their logical indices.
            for term in 0..terms {
                p[14] = term as u32;
                for g in (0..maxg).step_by(64) {
                    p[13] = g as u32;
                    let ng = 64.min(maxg - g);
                    x.stats.parameter_bytes += 8;
                    launch(x, ctl, stats, "segment tree pass1", &p, |e, x| {
                        e.set_pipeline(&p1);
                        bind(e, 0, x, &r.values);
                        bind(e, 1, x, &r.offsets);
                        if let Some(v) = &r.reference {
                            bind(e, 2, x, v);
                        } else {
                            e.set_buffer(2, &status, 0);
                        }
                        e.set_buffer(3, &out, 0);
                        e.set_buffer_mut(4, &mut partials, 0);
                        e.set_bytes(5, &p);
                        if let Some(v) = &r.mask {
                            bind(e, 6, x, v);
                        } else {
                            e.set_buffer(6, &status, 0);
                        }
                        if let Some(v) = &r.weights {
                            bind(e, 7, x, v);
                        } else {
                            e.set_buffer(7, &status, 0);
                        }
                        e.set_bytes(8, &options);
                        e.dispatch_groups([ng, 1, ns], 256);
                        Ok(())
                    })?;
                }
            }
            p[13] = 0;
            p[14] = 0;
            launch(x, ctl, stats, "segment tree pass2", &p, |e, x| {
                e.set_pipeline(&p2);
                e.set_buffer(0, &partials, 0);
                bind(e, 1, x, &r.offsets);
                e.set_buffer_mut(2, &mut out, 0);
                e.set_buffer_mut(3, &mut status, 0);
                e.set_bytes(4, &p);
                e.dispatch_groups([terms, ns, 1], 256);
                Ok(())
            })?;
            if phase == 0
                && (r.kind == SegmentKind::Moments2 || r.kind == SegmentKind::SourceProducts)
            {
                launch(x, ctl, stats, "segment finalise", &p, |e, _| {
                    e.set_buffer_mut(0, &mut out, 0);
                    e.set_bytes(1, &p);
                    e.dispatch_1d(&finish, ns);
                    Ok(())
                })?;
            }
            ctl.report(
                (phase as f64 + (start + ns) as f64 / s as f64) / phases as f64,
                "segmented reduction",
            );
        }
        x.recycle(partials);
    }
    x.put_output(r.output, out);
    x.put_output(r.status, status);
    Ok(())
}
fn qr_reduce(
    x: &mut Execution<'_>,
    ctl: &JobControl,
    stats: &mut StageStats,
    work: &Lease,
    factors: &Lease,
    aux: &mut Lease,
    partial: &mut Lease,
    p: &mut Params,
    first: usize,
    cols: usize,
    stage: u32,
) -> MkResult<()> {
    p[4] = stage;
    let pass1 = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_reduce1"))?;
    let pass2 = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_reduce2"))?;
    // Fixed batch tiles retain at most 16 MiB of partials. Each committed
    // launch covers <=64 reduction groups for bounded cancellation latency.
    let n = p[1] as usize;
    let groups = p[5] as usize;
    let batch = p[2] as usize;
    let tile = (16 * 1024 * 1024 / ((n + 1) * groups * 8)).clamp(1, 64);
    for b in (0..batch).step_by(tile) {
        let nb = tile.min(batch - b);
        p[8] = b as u32;
        let col_step = (64 / nb).max(1);
        for col in (first..first + cols).step_by(col_step) {
            let nc = col_step.min(first + cols - col);
            p[7] = col as u32;
            let group_step = (64 / (nb * nc)).max(1);
            for g in (0..groups).step_by(group_step) {
                p[6] = g as u32;
                let ng = group_step.min(groups - g);
                launch(x, ctl, stats, "QR reduction pass1", p, |e, _| {
                    e.set_pipeline(&pass1);
                    e.set_buffer(0, work, 0);
                    e.set_buffer(1, factors, 0);
                    e.set_buffer(2, aux, 0);
                    e.set_buffer_mut(3, partial, 0);
                    e.set_bytes(4, p);
                    e.dispatch_groups([ng, nc, nb], 256);
                    Ok(())
                })?;
            }
            launch(x, ctl, stats, "QR reduction pass2", p, |e, _| {
                e.set_pipeline(&pass2);
                e.set_buffer(0, partial, 0);
                e.set_buffer_mut(1, aux, 0);
                e.set_bytes(2, p);
                e.dispatch_groups([nc, nb, 1], 256);
                Ok(())
            })?;
        }
    }
    Ok(())
}
fn qr(r: &QrFit, x: &mut Execution<'_>, ctl: &JobControl, stats: &mut StageStats) -> MkResult<()> {
    let [b, m, n] = r.matrix.desc.shape[..] else {
        unreachable!()
    };
    if b == 0 {
        return Ok(());
    }
    let groups = reduce::num_groups(m);
    let mut factors = x.take_output(r.factors);
    let mut solution = x.take_output(r.solution);
    let mut diag = x.take_output(r.rank_condition);
    let mut work = x.allocate(b * (n + 1) * m * 2)?;
    let mut aux = x.allocate(b * (3 * (n + 1) + 4) * 2)?;
    let tile = (16 * 1024 * 1024 / ((n + 1) * groups * 8)).clamp(1, 64);
    let mut partial = x.allocate(tile * (n + 1) * groups * 2)?;
    let mut p = [
        m as u32,
        n as u32,
        b as u32,
        0,
        0,
        groups as u32,
        0,
        0,
        0,
        0,
        0,
        0,
        r.matrix.desc.element.words() as u32,
        0,
        r.rank_tolerance.to_bits(),
        (3 * (n + 1) + 4) as u32,
    ];
    let init = x.gpu.pipeline(&PipelineKey::plain("mkl_c_init"))?;
    for (i, len) in ranges(b * (m + n + 2) * n * 2) {
        let pp = [
            (b * (m + n + 2) * n * 2) as u32,
            i as u32,
            len as u32,
            3,
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ];
        launch(x, ctl, stats, "QR clear factors", &pp, |e, _| {
            e.set_buffer(0, &work, 0);
            e.set_buffer_mut(1, &mut factors, 0);
            e.set_bytes(2, &pp);
            e.dispatch_1d(&init, len);
            Ok(())
        })?;
    }
    let init = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_init"))?;
    for (i, len) in ranges(b * (n + 1) * m) {
        p[10] = i as u32;
        p[11] = len as u32;
        launch(x, ctl, stats, "QR initialise", &p, |e, x| {
            bind(e, 0, x, &r.matrix);
            bind(e, 1, x, &r.rhs);
            e.set_buffer_mut(2, &mut work, 0);
            e.set_buffer_mut(3, &mut factors, 0);
            e.set_buffer_mut(4, &mut diag, 0);
            e.set_buffer_mut(5, &mut aux, 0);
            e.set_bytes(6, &p);
            e.dispatch_1d(&init, len);
            Ok(())
        })?;
    }
    qr_reduce(
        x,
        ctl,
        stats,
        &work,
        &factors,
        &mut aux,
        &mut partial,
        &mut p,
        0,
        n,
        0,
    )?;
    qr_reduce(
        x,
        ctl,
        stats,
        &work,
        &factors,
        &mut aux,
        &mut partial,
        &mut p,
        0,
        n,
        1,
    )?;
    let scale = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_scale"))?;
    for (i, len) in ranges(b * n * m) {
        p[10] = i as u32;
        p[11] = len as u32;
        launch(x, ctl, stats, "QR equilibrate", &p, |e, _| {
            e.set_buffer_mut(0, &mut work, 0);
            e.set_buffer_mut(1, &mut factors, 0);
            e.set_buffer(2, &aux, 0);
            e.set_bytes(3, &p);
            e.dispatch_1d(&scale, len);
            Ok(())
        })?;
    }
    let pivot = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_pivot"))?;
    let swap = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_swap"))?;
    let update = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_update"))?;
    for k in 0..n {
        p[3] = k as u32;
        qr_reduce(
            x,
            ctl,
            stats,
            &work,
            &factors,
            &mut aux,
            &mut partial,
            &mut p,
            k,
            n - k,
            0,
        )?;
        qr_reduce(
            x,
            ctl,
            stats,
            &work,
            &factors,
            &mut aux,
            &mut partial,
            &mut p,
            k,
            n - k,
            1,
        )?;
        for (i, len) in ranges(b) {
            p[8] = i as u32;
            p[13] = len as u32;
            launch(x, ctl, stats, "QR pivot", &p, |e, _| {
                e.set_buffer_mut(0, &mut factors, 0);
                e.set_buffer_mut(1, &mut diag, 0);
                e.set_buffer_mut(2, &mut aux, 0);
                e.set_bytes(3, &p);
                e.dispatch_1d(&pivot, len);
                Ok(())
            })?;
        }
        for (i, len) in ranges(b * m) {
            p[10] = i as u32;
            p[11] = len as u32;
            launch(x, ctl, stats, "QR normalise column", &p, |e, _| {
                e.set_buffer_mut(0, &mut work, 0);
                e.set_buffer_mut(1, &mut factors, 0);
                e.set_buffer(2, &diag, 0);
                e.set_buffer(3, &aux, 0);
                e.set_bytes(4, &p);
                e.dispatch_1d(&swap, len);
                Ok(())
            })?;
        }
        for _ in 0..2 {
            qr_reduce(
                x,
                ctl,
                stats,
                &work,
                &factors,
                &mut aux,
                &mut partial,
                &mut p,
                k + 1,
                n - k,
                2,
            )?;
            for (i, len) in ranges(b * (n - k) * m) {
                p[10] = i as u32;
                p[11] = len as u32;
                launch(x, ctl, stats, "QR reorthogonalise", &p, |e, _| {
                    e.set_buffer_mut(0, &mut work, 0);
                    e.set_buffer_mut(1, &mut factors, 0);
                    e.set_buffer(2, &diag, 0);
                    e.set_buffer_mut(3, &mut aux, 0);
                    e.set_bytes(4, &p);
                    e.dispatch_1d(&update, len);
                    Ok(())
                })?;
            }
        }
        ctl.report((k + 1) as f64 / (n + 1) as f64, "QR orthogonalisation");
    }
    qr_reduce(
        x,
        ctl,
        stats,
        &work,
        &factors,
        &mut aux,
        &mut partial,
        &mut p,
        n,
        1,
        0,
    )?;
    qr_reduce(
        x,
        ctl,
        stats,
        &work,
        &factors,
        &mut aux,
        &mut partial,
        &mut p,
        n,
        1,
        1,
    )?;
    let finish = x.gpu.pipeline(&PipelineKey::plain("mkl_c_qr_finish"))?;
    // Condition estimation is cubic only in this bounded small column count.
    let fit_step = (CHUNK / (n * n * n)).max(1);
    for i in (0..b).step_by(fit_step) {
        let len = fit_step.min(b - i);
        p[8] = i as u32;
        p[13] = len as u32;
        launch(x, ctl, stats, "QR solve and condition", &p, |e, _| {
            e.set_buffer(0, &factors, 0);
            e.set_buffer_mut(1, &mut diag, 0);
            e.set_buffer_mut(2, &mut aux, 0);
            e.set_buffer_mut(3, &mut solution, 0);
            e.set_bytes(4, &p);
            e.dispatch_1d(&finish, len);
            Ok(())
        })?;
    }
    x.recycle(work);
    x.recycle(aux);
    x.recycle(partial);
    x.put_output(r.factors, factors);
    x.put_output(r.solution, solution);
    x.put_output(r.rank_condition, diag);
    Ok(())
}
