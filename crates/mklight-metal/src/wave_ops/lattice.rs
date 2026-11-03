use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::lattice::*};
use std::time::Instant;
/// Threads per dispatch. Per-thread work is a few dozen operations (or one
/// word copy), so every dispatch of an operation shares one command.
const TILE: usize = 1 << 18;
/// One dispatch range: input channel (split only), output, parameters, threads.
struct Part {
    input: usize,
    output: usize,
    params: [u32; 8],
    threads: usize,
}
pub fn execute(r: &Lattice, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let a = access(r);
    let part = |params: [u32; 8], threads: usize| Part {
        input: 0,
        output: 0,
        params,
        threads,
    };
    let (name, parts) = match r {
        Lattice::Wavelength { indices, table, .. } => {
            let n = indices.desc.shape[1];
            let p = [table.desc.shape[0] as u32, n as u32, 0, 0, 0, 0, 0, 0];
            ("mkl_wave_wavelength", vec![part(p, n)])
        }
        Lattice::Generate { side, copies, .. } => {
            let p = [*side as u32, 0, 0, 0, 0, 0, 0, 0];
            ("mkl_wave_lattice", vec![part(p, copies * side * side)])
        }
        Lattice::Survival { samples, .. } => (
            "mkl_wave_survival",
            vec![part([0; 8], samples.desc.shape[0])],
        ),
        Lattice::Rim {
            edges,
            previous,
            tolerance,
            levels,
            final_point,
            ..
        } => {
            let p = [
                *levels,
                0,
                u32::from(previous.is_some()),
                u32::from(*final_point),
                tolerance.hi.to_bits(),
                tolerance.lo.to_bits(),
                0,
                0,
            ];
            ("mkl_wave_rim", vec![part(p, edges.desc.shape[0])])
        }
        Lattice::Split {
            channels, ranges, ..
        } => {
            let mut parts = vec![];
            for (c, (v, rows)) in channels.iter().enumerate() {
                for (k, &range) in ranges.iter().enumerate() {
                    let (offset, stride, planes, len) = split_part(v, *rows, range);
                    parts.push(Part {
                        input: c,
                        output: c * ranges.len() + k,
                        params: [offset as u32, stride as u32, 0, len as u32, 0, 0, 0, 0],
                        threads: planes * len,
                    });
                }
            }
            ("mkl_wave_split", parts)
        }
        Lattice::Keep {
            channels,
            previous,
            source,
            ..
        } => {
            let n = source.desc.shape[0];
            let parts = channels
                .iter()
                .enumerate()
                .map(|(c, (v, rows))| {
                    let (big, planes, w) = keep_part(v, *rows);
                    Part {
                        input: c,
                        output: c,
                        params: [
                            big as u32,
                            n as u32,
                            w as u32,
                            u32::from(previous.is_some()),
                            0,
                            0,
                            0,
                            0,
                        ],
                        threads: n * planes.max(w),
                    }
                })
                .collect();
            ("mkl_wave_keep", parts)
        }
        Lattice::Concat { parts, .. } => {
            let words = parts[0].desc.element.words();
            let planes = concat_part(&parts[0]).0;
            let total: usize = parts.iter().map(|v| concat_part(v).1).sum();
            let mut start = 0;
            let parts = parts
                .iter()
                .enumerate()
                .map(|(c, v)| {
                    let m = concat_part(v).1;
                    let part = Part {
                        input: c,
                        output: 0,
                        params: [
                            (start * words) as u32,
                            (total * words) as u32,
                            0,
                            (m * words) as u32,
                            0,
                            0,
                            0,
                            0,
                        ],
                        threads: planes * m * words,
                    };
                    start += m;
                    part
                })
                .collect();
            ("mkl_wave_concat", parts)
        }
    };
    let pipe = x.gpu.pipeline(&PipelineKey::plain(name))?;
    let mut outs: Vec<_> = a.writes.iter().map(|&id| x.take_output(id)).collect();
    let start = Instant::now();
    ctl.check()?;
    let xr = &*x;
    // Rim reads [edges, fields, previous?, status?] at 0, 6, 1, 2; Metal still
    // requires bindings for the statically declared optional inputs.
    let slots: Vec<usize> = match r {
        Lattice::Rim { .. } => vec![0, 6, 1, 2],
        _ => (0..a.reads.len()).collect(),
    };
    // SAFETY: all views and outputs have validated row counts and layouts;
    // split ranges lie inside their validated channels and outputs.
    unsafe {
        xr.run("wave lattice", |e| {
            if let Some(v) = a.reads.first() {
                let (b, o) = xr.buffer(v);
                for i in [1, 2] {
                    e.set_buffer(i, b, o);
                }
            }
            let gather = matches!(
                r,
                Lattice::Split { .. } | Lattice::Keep { .. } | Lattice::Concat { .. }
            );
            if !gather {
                for (&i, v) in slots.iter().zip(&a.reads) {
                    let (b, o) = xr.buffer(v);
                    e.set_buffer(i, b, o);
                }
            }
            if let (Lattice::Rim { .. }, [_, samples, source]) = (r, &mut outs[..]) {
                e.set_buffer_mut(4, samples, 0);
                e.set_buffer_mut(7, source, 0);
            }
            if matches!(r, Lattice::Keep { .. }) {
                // Keep binds probe channel c at 0, previous c (or c again
                // without one) at 1 and the source indices at 2.
                let (b, o) = xr.buffer(a.reads.last().unwrap());
                e.set_buffer(2, b, o);
            }
            for p in &parts {
                if gather {
                    let (b, o) = xr.buffer(&a.reads[p.input]);
                    e.set_buffer(0, b, o);
                }
                if let Lattice::Keep { channels, .. } = r {
                    let prev = if p.params[3] != 0 {
                        &a.reads[channels.len() + p.input]
                    } else {
                        &a.reads[p.input]
                    };
                    let (b, o) = xr.buffer(prev);
                    e.set_buffer(1, b, o);
                }
                e.set_buffer_mut(3, &mut outs[p.output], 0);
                for first in (0..p.threads).step_by(TILE) {
                    let mut q = p.params;
                    q[6] = first as u32;
                    q[7] = (p.threads - first).min(TILE) as u32;
                    e.set_bytes(5, &q);
                    e.dispatch_1d(&pipe, q[7] as usize);
                }
            }
            Ok(())
        })?
    };
    ctl.check()?;
    ctl.report(1., "wave lattice");
    for (&id, b) in a.writes.iter().zip(outs) {
        x.put_output(id, b);
    }
    let wall = start.elapsed().as_secs_f64();
    x.stats.parameter_bytes += parts.len() as u64 * 32;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "wave_lattice".into(),
        wall_seconds: wall,
        device_seconds: 0.,
        command_buffers: 0,
        max_command_seconds: wall,
    });
    Ok(())
}
