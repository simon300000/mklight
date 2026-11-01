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
            final_point,
            ..
        } => {
            let p = [
                0,
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
    let cs = unsafe {
        xr.gpu.run("wave lattice", |e| {
            if let Some(v) = a.reads.first() {
                let (b, o) = xr.buffer(v);
                for i in [1, 2] {
                    e.set_buffer(i, b, o);
                }
            }
            if !matches!(r, Lattice::Split { .. }) {
                for (&i, v) in slots.iter().zip(&a.reads) {
                    let (b, o) = xr.buffer(v);
                    e.set_buffer(i, b, o);
                }
            }
            if let (Lattice::Rim { .. }, [_, samples]) = (r, &mut outs[..]) {
                e.set_buffer_mut(4, samples, 0);
            }
            for p in &parts {
                if matches!(r, Lattice::Split { .. }) {
                    let (b, o) = xr.buffer(&a.reads[p.input]);
                    e.set_buffer(0, b, o);
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
    x.stats.command_buffers += 1;
    x.stats.device_seconds += cs.gpu_seconds;
    x.stats.parameter_bytes += parts.len() as u64 * 32;
    x.stats.stages.push(StageStats {
        operation: x.stats.stages.len(),
        name: "wave_lattice".into(),
        wall_seconds: wall,
        device_seconds: cs.gpu_seconds,
        command_buffers: 1,
        max_command_seconds: wall,
    });
    Ok(())
}
