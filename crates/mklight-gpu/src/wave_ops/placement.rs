//! Reference-sphere samples and scalar radiometric amplitudes.
use super::*;
#[derive(Clone, Debug)]
pub enum Placement {
    Centroid {
        hits: PlanView,
        power: Option<PlanView>,
        chief: [Df32; 2],
        count: usize,
        output: BufferId,
    },
    /// Output DF [N,8]: alpha,beta,OPD,gamma,T,object x/y,start direction z.
    Project {
        projection: PlanView,
        status: PlanView,
        starts: PlanView,
        hits: PlanView,
        power: Option<PlanView>,
        frame: [[Df32; 3]; 3],
        object_axes: [[Df32; 3]; 2],
        infinite: bool,
        propagation: f32,
        sheet_factor: f32,
        output: BufferId,
        radii: BufferId,
    },
    /// Original lattice only; normalise amplitudes at its chief (or maximum).
    Nodes {
        placed: PlanView,
        valid: PlanView,
        sin_max: Df32,
        side: usize,
        model: u32,
        infinite: bool,
        output: BufferId,
    },
    /// All four possible cell triangles participate in the automatic-grid rule.
    Slopes {
        nodes: PlanView,
        valid: PlanView,
        output: BufferId,
    },
}
pub fn validate(r: &Placement, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        Placement::Centroid {
            hits,
            power,
            chief,
            count,
            output,
        } => {
            let n = hits.desc.shape.get(1).copied().unwrap_or(0);
            view(hits, F32, &[15, n])?;
            if let Some(v) = power {
                view(v, U32, &[n, 96])?;
            }
            if chief.iter().any(|v| !v.is_finite()) {
                return Err(bad("centroid chief"));
            }
            if *count > n {
                return Err(bad("centroid extent"));
            }
            out(p, *output, DoubleFloat, &[3, *count])
        }
        Placement::Project {
            projection,
            status,
            starts,
            hits,
            power,
            frame,
            object_axes,
            propagation,
            sheet_factor,
            output,
            radii,
            ..
        } => {
            let n = projection.desc.shape.get(1).copied().unwrap_or(0);
            view(projection, DoubleFloat, &[4, n])?;
            view(status, U32, &[n])?;
            view(starts, F32, &[15, n])?;
            view(hits, F32, &[15, n])?;
            if let Some(v) = power {
                view(v, U32, &[n, 96])?;
            }
            if frame
                .iter()
                .flatten()
                .chain(object_axes.iter().flatten())
                .any(|v| !v.is_finite())
                || ![-1., 1.].contains(propagation)
                || ![-1., 1.].contains(sheet_factor)
            {
                return Err(bad("wave placement frame"));
            }
            out(p, *output, DoubleFloat, &[n, 8])?;
            out(p, *radii, DoubleFloat, &[n])
        }
        Placement::Nodes {
            placed,
            valid,
            sin_max,
            side,
            model,
            output,
            ..
        } => {
            if *side < 2 || *model > 2 || !sin_max.is_finite() || sin_max.hi <= 0. {
                return Err(bad("wave node settings"));
            }
            view(placed, DoubleFloat, &[side * side, 8])?;
            view(valid, U32, &[*side, *side])?;
            out(p, *output, DoubleFloat, &[*side, *side, 4])
        }
        Placement::Slopes {
            nodes,
            valid,
            output,
        } => {
            let side = nodes.desc.shape[0];
            if side < 2 {
                return Err(bad("wave slope side"));
            }
            view(nodes, DoubleFloat, &[side, side, 4])?;
            view(valid, U32, &[side, side])?;
            out(p, *output, DoubleFloat, &[(side - 1) * (side - 1)])
        }
    }
}
pub fn access(r: &Placement) -> Access {
    let (reads, writes) = match r {
        Placement::Centroid {
            hits,
            power,
            output,
            ..
        } => {
            let mut v = vec![hits.clone()];
            v.extend(power.clone());
            (v, vec![*output])
        }
        Placement::Project {
            projection,
            status,
            starts,
            hits,
            power,
            output,
            radii,
            ..
        } => {
            let mut v = vec![
                projection.clone(),
                status.clone(),
                starts.clone(),
                hits.clone(),
            ];
            v.extend(power.clone());
            (v, vec![*output, *radii])
        }
        Placement::Nodes {
            placed,
            valid,
            output,
            ..
        } => (vec![placed.clone(), valid.clone()], vec![*output]),
        Placement::Slopes {
            nodes,
            valid,
            output,
        } => (vec![nodes.clone(), valid.clone()], vec![*output]),
    };
    Access {
        reads,
        writes,
        systems: vec![],
    }
}
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Params {
    pub n: u32,
    pub side: u32,
    pub model: u32,
    pub infinite: u32,
    pub power: u32,
    pub first: u32,
    pub count: u32,
    pub reserved: u32,
    pub sin_max: Df32,
    pub propagation: f32,
    pub sheet_factor: f32,
    pub frame: [[Df32; 3]; 3],
    pub object_axes: [[Df32; 3]; 2],
}
pub fn params(r: &Placement) -> Params {
    use bytemuck::Zeroable;
    let mut p = Params::zeroed();
    match r {
        Placement::Centroid {
            hits,
            power,
            chief,
            count,
            ..
        } => {
            p.n = *count as u32;
            p.side = hits.desc.shape[1] as u32;
            p.power = power.is_some() as u32;
            p.object_axes[0][0] = chief[0];
            p.object_axes[0][1] = chief[1];
        }
        Placement::Project {
            projection,
            frame,
            object_axes,
            infinite,
            power,
            propagation,
            sheet_factor,
            ..
        } => {
            p.n = projection.desc.shape[1] as u32;
            p.frame = *frame;
            p.object_axes = *object_axes;
            p.infinite = *infinite as u32;
            p.power = power.is_some() as u32;
            p.propagation = *propagation;
            p.sheet_factor = *sheet_factor;
        }
        Placement::Nodes {
            side,
            sin_max,
            model,
            infinite,
            ..
        } => {
            p.n = (side * side) as u32;
            p.side = *side as u32;
            p.sin_max = *sin_max;
            p.model = *model;
            p.infinite = *infinite as u32;
        }
        Placement::Slopes { nodes, .. } => {
            p.side = nodes.desc.shape[0] as u32;
            p.n = (p.side - 1) * (p.side - 1);
        }
    }
    p
}
pub fn cpu(
    r: &Placement,
    x: &mut crate::resident::cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    fn vals<'a, T: bytemuck::Pod>(x: &'a crate::resident::cpu::Execution, v: &PlanView) -> &'a [T] {
        let a = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..v.offset + v.desc.len().unwrap()]
    }
    match r {
        Placement::Centroid {
            hits,
            power,
            chief,
            count,
            output,
        } => {
            let n = hits.desc.shape[1];
            let h = vals::<f32>(x, hits);
            let power = power.as_ref().map(|v| {
                bytemuck::cast_slice::<u32, crate::radiometry::PolarizationValue>(vals(x, v))
            });
            let mut a = vec![Df32::ZERO; 3 * count];
            for k in 0..*count {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                for j in 0..2 {
                    a[j * count + k] = Df32::new(h[j * n + k], h[(j + 9) * n + k]) - chief[j];
                }
                a[2 * count + k] = power.map_or(Df32::from_f32(1.), |v| v[k].power[0]);
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
        Placement::Project {
            projection,
            status,
            starts,
            hits,
            power,
            frame,
            object_axes,
            infinite,
            propagation,
            sheet_factor,
            output,
            radii,
        } => {
            let n = projection.desc.shape[1];
            let q = vals::<Df32>(x, projection);
            let status = vals::<u32>(x, status);
            let start = vals::<f32>(x, starts);
            let hit = vals::<f32>(x, hits);
            let power = power.as_ref().map(|v| {
                bytemuck::cast_slice::<u32, crate::radiometry::PolarizationValue>(vals(x, v))
            });
            let mut rows = vec![[Df32::ZERO; 8]; n];
            let mut radius = vec![Df32::ZERO; n];
            let get = |v: &[f32], k, j| Df32::new(v[j * n + k], v[(j + 9) * n + k]);
            for k in 0..n {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                if status[k] == 0 {
                    continue;
                }
                let sphere = std::array::from_fn(|j| q[(j + 1) * n + k].mul_f32(*sheet_factor));
                let d = std::array::from_fn(|j| {
                    get(hit, k, j + 3).mul_f32(if j == 2 { *propagation } else { 1. })
                });
                let source =
                    std::array::from_fn(|j| get(start, k, j + if *infinite { 0 } else { 3 }));
                let dot = spec::dot;
                let alpha = dot(sphere, frame[0]);
                let beta = dot(sphere, frame[1]);
                let gamma = dot(d, frame[2]);
                let t = power.map_or(Df32::from_f32(1.), |v| v[k].power[0]);
                rows[k] = [
                    alpha,
                    beta,
                    q[k],
                    if gamma.hi < 0. { -gamma } else { gamma },
                    t,
                    dot(source, object_axes[0]),
                    dot(source, object_axes[1]),
                    get(start, k, 5),
                ];
                radius[k] = (alpha * alpha + beta * beta).sqrt();
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&rows));
            x.words_mut(*radii)
                .copy_from_slice(bytemuck::cast_slice(&radius));
        }
        Placement::Nodes {
            placed,
            valid,
            sin_max,
            side,
            model,
            infinite,
            output,
        } => {
            let raw = vals::<Df32>(x, placed).as_chunks::<8>().0;
            let mask = vals::<u32>(x, valid);
            let n = side * side;
            let mut amps = vec![None; n];
            let deriv = |k: usize, axis: usize, col: usize| -> Option<[Df32; 2]> {
                let i = k % side;
                let j = k / side;
                let d = if axis == 0 { 1 } else { *side };
                let pos = if axis == 0 { i } else { j };
                let a = (pos + 1 < *side && mask[k + d] != 0).then_some(k + d);
                let b = (pos > 0 && mask[k - d] != 0).then_some(k.wrapping_sub(d));
                let (lo, hi, f) = match (a, b) {
                    (Some(a), Some(b)) => (b, a, 0.5),
                    (Some(a), None) => (k, a, 1.),
                    (None, Some(b)) => (b, k, 1.),
                    _ => return None,
                };
                Some([
                    (raw[hi][col] - raw[lo][col]).mul_f32(f),
                    (raw[hi][col + 1] - raw[lo][col + 1]).mul_f32(f),
                ])
            };
            for k in 0..n {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                if mask[k] == 0 {
                    continue;
                }
                amps[k] = match model {
                    0 => Some(Df32::from_f32(1.)),
                    1 => Some(raw[k][4].sqrt()),
                    _ => {
                        let calc = || {
                            let ux = deriv(k, 0, 0)?;
                            let uy = deriv(k, 1, 0)?;
                            let ox = deriv(k, 0, 5)?;
                            let oy = deriv(k, 1, 5)?;
                            let abs = |v: Df32| if v.hi < 0. { -v } else { v };
                            let jac = abs(ux[0] * uy[1] - ux[1] * uy[0]) / (*sin_max * *sin_max);
                            let mut m = abs(ox[0] * oy[1] - ox[1] * oy[0]);
                            if !*infinite {
                                m = m / abs(raw[k][7]);
                            }
                            if jac.hi <= 0. {
                                None
                            } else {
                                Some((raw[k][4] * m * raw[k][3] / jac).sqrt())
                            }
                        };
                        calc()
                    }
                };
            }
            let mut sum = Df32::ZERO;
            let mut count = 0;
            let mut max = Df32::ZERO;
            for block in amps.chunks(256) {
                let mut partial = Df32::ZERO;
                for v in block.iter().flatten() {
                    partial = partial + *v;
                    count += 1;
                    if v.to_f64() > max.to_f64() {
                        max = *v;
                    }
                }
                sum = sum + partial;
            }
            let mean = if count > 0 {
                sum / Df32::from_f32(count as f32)
            } else {
                Df32::from_f32(1.)
            };
            for k in 0..n {
                if mask[k] != 0 && amps[k].is_none() {
                    amps[k] = Some(mean);
                    if mean.to_f64() > max.to_f64() {
                        max = mean;
                    }
                }
            }
            let center = (side / 2) * side + side / 2;
            let norm = if side % 2 == 1 && mask[center] != 0 && amps[center].unwrap().hi > 0. {
                amps[center].unwrap()
            } else {
                max
            };
            if !norm.is_finite() || norm.hi <= 0. {
                return Err(mklight_core::MkError::Numerical(
                    "pupil transmits no finite light".into(),
                ));
            }
            let mut rows = vec![[Df32::ZERO; 4]; n];
            for k in 0..n {
                if mask[k] != 0 {
                    rows[k] = [
                        raw[k][0] / *sin_max,
                        raw[k][1] / *sin_max,
                        raw[k][2],
                        amps[k].unwrap() / norm,
                    ];
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&rows));
        }
        Placement::Slopes {
            nodes,
            valid,
            output,
        } => {
            let a = vals::<Df32>(x, nodes).as_chunks::<4>().0;
            let mask = vals::<u32>(x, valid);
            let side = nodes.desc.shape[0];
            let mut out = vec![Df32::ZERO; (side - 1) * (side - 1)];
            for (k, g) in out.iter_mut().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let i = k % (side - 1);
                let j = k / (side - 1);
                let ids = [
                    j * side + i,
                    j * side + i + 1,
                    (j + 1) * side + i + 1,
                    (j + 1) * side + i,
                ];
                for tri in [[0, 1, 2], [0, 2, 3], [0, 1, 3], [1, 2, 3]] {
                    let [ia, ib, ic] = tri.map(|i| ids[i]);
                    if mask[ia] == 0 || mask[ib] == 0 || mask[ic] == 0 {
                        continue;
                    }
                    let (a, b, c) = (a[ia], a[ib], a[ic]);
                    let (ux, uy, vx, vy) = (b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1]);
                    let det = ux * vy - uy * vx;
                    if det.to_f64().abs() > 1e-20 {
                        let (dw, ew) = (b[2] - a[2], c[2] - a[2]);
                        for v in [(dw * vy - ew * uy) / det, (ew * ux - dw * vx) / det] {
                            let v = if v.hi < 0. { -v } else { v };
                            if v.to_f64() > g.to_f64() {
                                *g = v;
                            }
                        }
                    }
                }
            }
            x.words_mut(*output)
                .copy_from_slice(bytemuck::cast_slice(&out));
        }
    }
    Ok(())
}
