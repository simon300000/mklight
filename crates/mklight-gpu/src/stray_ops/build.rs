use super::*;
#[derive(Clone, Debug)]
pub struct Build {
    pub primitives: PlanView,
    pub output: BufferId,
}
pub fn node_count(n: usize) -> usize {
    2 * n.next_power_of_two() - 1
}
pub(super) fn execute<E: FpEnv>(
    r: &Build,
    x: &mut cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    let n = r.primitives.desc.shape[0];
    let size = n.next_power_of_two();
    let depth = size.ilog2();
    for level in (0..=depth).rev() {
        let range = size >> level;
        for j in 0..1usize << level {
            if j % 1024 == 0 {
                ctl.check()?;
            }
            let a = j * range;
            let id = 2 * a + level as usize - a.count_ones() as usize;
            let mut out = Node {
                lower: [Df32::ZERO; 3],
                upper: [Df32::ZERO; 3],
                escape: (id + 2 * range - 1) as u32,
                primitive: u32::MAX,
            };
            if level == depth && a < n {
                use super::math::*;
                let p = rows::<Primitive>(x, &r.primitives)[a];
                let mut lo = [D::<E>::f(f32::MAX); 3];
                let mut hi = [D::<E>::f(-f32::MAX); 3];
                for k in 0..8 {
                    let q: V<E> = std::array::from_fn(|i| {
                        D::from(if k & (1 << i) == 0 {
                            p.lower[i]
                        } else {
                            p.upper[i]
                        })
                    });
                    let w: V<E> = std::array::from_fn(|i| {
                        D::from(p.origin[i])
                            + D::from(p.rotation[3 * i]) * q[0]
                            + D::from(p.rotation[3 * i + 1]) * q[1]
                            + D::from(p.rotation[3 * i + 2]) * q[2]
                    });
                    for i in 0..3 {
                        if (w[i] - lo[i]).v() < 0. {
                            lo[i] = w[i];
                        }
                        if (w[i] - hi[i]).v() > 0. {
                            hi[i] = w[i];
                        }
                    }
                }
                out.lower = lo.map(|v| v.0);
                out.upper = hi.map(|v| v.0);
                out.primitive = a as u32;
            } else if level < depth {
                let nodes: &[Node] = bytemuck::cast_slice(x.words(r.output));
                let (l, h) = (nodes[id + 1], nodes[id + range]);
                for k in 0..3 {
                    out.lower[k] = if (l.lower[k] - h.lower[k]).hi < 0. {
                        l.lower[k]
                    } else {
                        h.lower[k]
                    };
                    out.upper[k] = if (l.upper[k] - h.upper[k]).hi > 0. {
                        l.upper[k]
                    } else {
                        h.upper[k]
                    };
                }
            }
            bytemuck::cast_slice_mut::<u32, Node>(x.words_mut(r.output))[id] = out;
        }
    }
    Ok(())
}
