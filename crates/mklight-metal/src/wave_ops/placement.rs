use crate::{context::PipelineKey, resident::Execution};
use mklight_core::MkResult;
use mklight_gpu::{pending::JobControl, resident::StageStats, wave_ops::placement::*};
use std::time::Instant;
pub fn execute(r: &Placement, x: &mut Execution<'_>, ctl: &JobControl) -> MkResult<()> {
    let a = access(r);
    let mut p = params(r);
    let n = p.n as usize;
    let blocks = n.div_ceil(256);
    // Kernels in order; the norm and maxima are small fixed-order block
    // reductions that finish in the same command.
    let names: &[&str] = match r {
        Placement::Centroid { .. } => &["mkl_wave_centroid"],
        Placement::Project { .. } => &[
            "mkl_wave_place",
            "mkl_wave_max_blocks",
            "mkl_wave_max_finish",
        ],
        Placement::Nodes { .. } => &[
            "mkl_wave_node_amplitude",
            "mkl_wave_node_norm_blocks",
            "mkl_wave_node_norm_finish",
            "mkl_wave_nodes",
        ],
        Placement::Slopes { .. } => &[
            "mkl_wave_slopes",
            "mkl_wave_max_blocks",
            "mkl_wave_max_finish",
        ],
    };
    let pipes = names
        .iter()
        .map(|k| x.gpu.pipeline(&PipelineKey::plain(k)))
        .collect::<MkResult<Vec<_>>>()?;
    let nodes = matches!(r, Placement::Nodes { .. });
    let reduced = matches!(r, Placement::Project { .. } | Placement::Slopes { .. });
    let mut outs: Vec<_> = a.writes.iter().map(|&id| x.take_output(id)).collect();
    // Per-node amplitudes or reduced values (6), norm (7) and block partials (9).
    let mut values = if nodes || reduced {
        Some(x.allocate(n * 2)?)
    } else {
        None
    };
    let mut norm = if nodes { Some(x.allocate(4)?) } else { None };
    let mut partials = if nodes || reduced {
        Some(x.allocate(blocks * 6)?)
    } else {
        None
    };
    let start = Instant::now();
    let mut stage = StageStats {
        operation: x.stats.stages.len(),
        name: "wave_placement".into(),
        ..Default::default()
    };
    let mut dispatches = vec![];
    for (step, name) in names.iter().enumerate() {
        let count = if name.ends_with("_blocks") {
            blocks
        } else if name.ends_with("_finish") {
            1
        } else {
            n
        };
        for first in (0..count).step_by(1 << 18) {
            let len = (count - first).min(1 << 18);
            dispatches.push(((step, first as u32, len as u32), len));
        }
    }
    let groups = super::commands(dispatches);
    for (done, group) in groups.iter().enumerate() {
        ctl.check()?;
        let xr = &*x;
        let wall = Instant::now();
        // SAFETY: validated contiguous inputs; all mutable storage is plan-local.
        unsafe {
            xr.run("wave placement", |e| {
                for (i, v) in a.reads.iter().enumerate() {
                    let (b, o) = xr.buffer(v);
                    e.set_buffer(i, b, o);
                }
                // Statically declared optional power inputs are never read.
                let (b, o) = xr.buffer(&a.reads[0]);
                match r {
                    Placement::Centroid { power: None, .. } => e.set_buffer(1, b, o),
                    Placement::Project { power: None, .. } => e.set_buffer(4, b, o),
                    _ => {}
                }
                match &mut outs[..] {
                    [out, radius] => {
                        e.set_buffer_mut(5, out, 0);
                        e.set_buffer_mut(7, radius, 0);
                    }
                    [out] if reduced => e.set_buffer_mut(7, out, 0),
                    [out] => e.set_buffer_mut(5, out, 0),
                    _ => unreachable!("placement writes one or two outputs"),
                }
                if let Some(v) = values.as_mut() {
                    e.set_buffer_mut(6, v, 0);
                }
                if let Some(v) = norm.as_mut() {
                    e.set_buffer_mut(7, v, 0);
                }
                if let Some(v) = partials.as_mut() {
                    e.set_buffer_mut(9, v, 0);
                }
                for &(step, first, count) in group {
                    p.first = first;
                    p.count = count;
                    e.set_bytes(8, &p);
                    e.dispatch_1d(&pipes[step], count as usize);
                }
                Ok(())
            })?
        };

        stage.max_command_seconds = stage.max_command_seconds.max(wall.elapsed().as_secs_f64());
        ctl.report((done + 1) as f64 / groups.len() as f64, "wave placement");
    }
    ctl.check()?;
    if let Some(v) = norm.take() {
        x.check_buffer(v, |v, _| {
            let a: &[mklight_gpu::Df32] = bytemuck::cast_slice(v.as_slice());
            if !a[1].is_finite() || a[1].hi <= 0. {
                return Err(mklight_core::MkError::Numerical(
                    "pupil transmits no finite light".into(),
                ));
            }
            Ok(())
        });
    }
    for (&id, b) in a.writes.iter().zip(outs) {
        x.put_output(id, b);
    }
    for b in [values, norm, partials].into_iter().flatten() {
        x.recycle(b);
    }
    stage.wall_seconds = start.elapsed().as_secs_f64();

    x.stats.parameter_bytes +=
        groups.iter().map(Vec::len).sum::<usize>() as u64 * size_of::<Params>() as u64;
    x.stats.stages.push(stage);
    Ok(())
}
