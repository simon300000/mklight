//! Device pupil probes, disk masking and batched boundary bisection.
use super::*;
use crate::aiming::Sample;
#[derive(Clone, Debug)]
pub enum Lattice {
    Wavelength {
        count: usize,
        wavelength: u32,
        output: BufferId,
    },
    Generate {
        side: usize,
        centres: bool,
        samples: BufferId,
    },
    Survival {
        samples: PlanView,
        status: PlanView,
        output: BufferId,
    },
    /// Edges are [inside x/y, outside x/y]. State stores a,b,lo,hi,trial,done.
    Rim {
        edges: PlanView,
        previous: Option<PlanView>,
        status: Option<PlanView>,
        tolerance: Df32,
        final_point: bool,
        state: BufferId,
        samples: BufferId,
    },
}
pub fn validate(r: &Lattice, p: &ResidentPlan) -> MkResult<()> {
    use ElementType::*;
    match r {
        Lattice::Wavelength { count, output, .. } => out(p, *output, U32, &[2, *count]),
        Lattice::Generate {
            side,
            centres,
            samples,
        } => {
            if !(10..=2050).contains(side) {
                return Err(bad("wave lattice side"));
            }
            let n = side * side + if *centres { (side - 1) * (side - 1) } else { 0 };
            out(p, *samples, U32, &[n, 8])
        }
        Lattice::Survival {
            samples,
            status,
            output,
        } => {
            let n = samples.desc.shape[0];
            view(samples, U32, &[n, 8])?;
            view(status, U32, &[3, n])?;
            out(p, *output, U32, &[n])
        }
        Lattice::Rim {
            edges,
            previous,
            status,
            tolerance,
            state,
            samples,
            final_point,
        } => {
            let n = edges.desc.shape[0];
            view(edges, DoubleFloat, &[n, 4])?;
            if previous.is_some() != status.is_some()
                || !tolerance.is_finite()
                || tolerance.hi <= 0.
            {
                return Err(bad("rim state/tolerance"));
            }
            if let Some(v) = previous {
                view(v, DoubleFloat, &[n, 8])?;
            }
            if let Some(v) = status {
                view(v, U32, &[3, 16 * n])?;
            }
            out(p, *state, DoubleFloat, &[n, 8])?;
            out(
                p,
                *samples,
                U32,
                &[if *final_point { n } else { 16 * n }, 8],
            )
        }
    }
}
pub fn access(r: &Lattice) -> Access {
    let (reads, writes) = match r {
        Lattice::Wavelength { output, .. } => (vec![], vec![*output]),
        Lattice::Generate { samples, .. } => (vec![], vec![*samples]),
        Lattice::Survival {
            samples,
            status,
            output,
        } => (vec![samples.clone(), status.clone()], vec![*output]),
        Lattice::Rim {
            edges,
            previous,
            status,
            state,
            samples,
            ..
        } => {
            let mut reads = vec![edges.clone()];
            reads.extend(previous.clone());
            reads.extend(status.clone());
            (reads, vec![*state, *samples])
        }
    };
    Access {
        reads,
        writes,
        systems: vec![],
    }
}
pub fn cpu(r: &Lattice, x: &mut crate::resident::cpu::Execution, ctl: &JobControl) -> MkResult<()> {
    fn vals<'a, T: bytemuck::Pod>(x: &'a crate::resident::cpu::Execution, v: &PlanView) -> &'a [T] {
        let a = bytemuck::cast_slice(x.words(v.buffer));
        &a[v.offset..v.offset + v.desc.len().unwrap()]
    }
    let one = Df32::from_f32(1.);
    match r {
        Lattice::Wavelength {
            count,
            wavelength,
            output,
        } => {
            let out = x.words_mut(*output);
            out[..*count].fill(0);
            out[*count..].fill(*wavelength);
        }
        Lattice::Generate {
            side,
            centres,
            samples,
        } => {
            let n = side * side + if *centres { (side - 1) * (side - 1) } else { 0 };
            let mut a = vec![Sample::new([0.; 2], 0, 0); n];
            let h = Df32::from_f32(2.) / Df32::from_f32((side - 3) as f32);
            let c = Df32::from_f32((side - 1) as f32 * 0.5);
            for (k, s) in a.iter_mut().enumerate() {
                if k % 4096 == 0 {
                    ctl.check()?;
                }
                let (i, j, offset) = if k < side * side {
                    (k % side, k / side, 0.)
                } else {
                    let t = k - side * side;
                    (t % (side - 1), t / (side - 1), 0.5)
                };
                s.pupil = [
                    (Df32::from_f32(i as f32 + offset) - c) * h,
                    (Df32::from_f32(j as f32 + offset) - c) * h,
                ];
                s.id = [k as u32, 0];
            }
            x.words_mut(*samples)
                .copy_from_slice(bytemuck::cast_slice(&a));
        }
        Lattice::Survival {
            samples,
            status,
            output,
        } => {
            let s: &[Sample] = bytemuck::cast_slice(vals::<u32>(x, samples));
            let t = vals::<u32>(x, status);
            let a: Vec<u32> = s
                .iter()
                .enumerate()
                .map(|(k, s)| {
                    u32::from(
                        t[k] == 0
                            && (s.pupil[0] * s.pupil[0] + s.pupil[1] * s.pupil[1]).to_f64()
                                <= 1. + 1e-12,
                    )
                })
                .collect();
            x.words_mut(*output).copy_from_slice(&a);
        }
        Lattice::Rim {
            edges,
            previous,
            status,
            tolerance,
            final_point,
            state,
            samples,
        } => {
            let e = vals::<Df32>(x, edges).as_chunks::<4>().0;
            let prev = previous
                .as_ref()
                .map(|v| vals::<Df32>(x, v).as_chunks::<8>().0);
            let st = status.as_ref().map(|v| vals::<u32>(x, v));
            let mut a = vec![[Df32::ZERO; 8]; e.len()];
            let width = if *final_point { 1 } else { 16 };
            let mut probes = vec![Sample::new([0.; 2], 0, 0); e.len() * width];
            for (k, (a, e)) in a.iter_mut().zip(e).enumerate() {
                if k % 256 == 0 {
                    ctl.check()?;
                }
                a[..4].copy_from_slice(e);
                let dx = e[2] - e[0];
                let dy = e[3] - e[1];
                let len = (dx * dx + dy * dy).sqrt();
                if let Some(prev) = prev {
                    *a = prev[k];
                    if a[7].hi == 0. {
                        if st.unwrap()[16 * k] == 0 {
                            a[4] = a[5];
                        } else {
                            let (lo, hi) = (a[4], a[5]);
                            let span = hi - lo;
                            let (mut l, mut h) = (0, 16);
                            for _ in 0..4 {
                                let m = (l + h) / 2;
                                if st.unwrap()[16 * k + m] == 0 {
                                    l = m;
                                } else {
                                    h = m;
                                }
                            }
                            a[4] = lo + span.mul_f32(l as f32 / 16.);
                            a[5] = lo + span.mul_f32(h as f32 / 16.);
                        }
                        if ((a[5] - a[4]) * len).to_f64() <= tolerance.to_f64() {
                            a[7] = one;
                        }
                    }
                } else {
                    a[5] = one;
                    let outside = e[2] * e[2] + e[3] * e[3];
                    if outside.to_f64() > 1. {
                        let qa = dx * dx + dy * dy;
                        let qb = e[0] * dx + e[1] * dy;
                        let qc = e[0] * e[0] + e[1] * e[1] - one;
                        let disc = qb * qb - qa * qc;
                        let root = if disc.hi < 0. {
                            Df32::ZERO
                        } else {
                            disc.sqrt()
                        };
                        let t = if qb.hi <= 0. {
                            (-qb + root) / qa
                        } else {
                            -qc / (qb + root)
                        };
                        a[5] = if t.to_f64() < 0. {
                            Df32::ZERO
                        } else if t.to_f64() > 1. {
                            one
                        } else {
                            t
                        };
                    }
                }
                for j in 0..width {
                    let t = if *final_point || a[7].hi > 0. {
                        a[4]
                    } else if j == 0 {
                        a[5]
                    } else {
                        a[4] + (a[5] - a[4]).mul_f32(j as f32 / 16.)
                    };
                    let s = &mut probes[k * width + j];
                    s.pupil = [e[0] + dx * t, e[1] + dy * t];
                    s.id = [(k * width + j) as u32, 0];
                    if prev.is_none() {
                        let rad = (s.pupil[0] * s.pupil[0] + s.pupil[1] * s.pupil[1]).sqrt();
                        if rad.to_f64() > 1. {
                            s.pupil = s.pupil.map(|v| v / rad);
                        }
                    }
                }
            }
            x.words_mut(*state)
                .copy_from_slice(bytemuck::cast_slice(&a));
            x.words_mut(*samples)
                .copy_from_slice(bytemuck::cast_slice(&probes));
        }
    }
    Ok(())
}
