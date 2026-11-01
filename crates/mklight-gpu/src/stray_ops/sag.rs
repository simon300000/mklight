use super::{math::*, records::*};
use crate::{FpEnv, fx, surface_math};
fn eval<E: FpEnv>(s: &SurfaceData, p: V<E>, r: D<E>) -> Option<[D<E>; 3]> {
    let radius = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let k = if (radius - r).v() > 0. {
        r / radius
    } else {
        D::f(1.)
    };
    let j = surface_math::evaluate(
        &s.surface,
        &s.coefficients,
        fx::D::<E>::from_df((p[0] * k).0),
        fx::D::<E>::from_df((p[1] * k).0),
    )?;
    Some([D::from(j.v.0), D::from(j.x.0), D::from(j.y.0)])
}
pub(super) fn intersect<E: FpEnv>(
    p: Primitive,
    s: &SurfaceData,
    ray: Ray,
    limit: D<E>,
) -> Option<Hit> {
    let o = rotate(p.rotation, sub(vec(ray.origin), vec(p.origin)));
    let d = rotate(p.rotation, vec(ray.direction));
    let (mut ta, mut tb) = (D::f(0.), limit);
    for k in 0..3 {
        if d[k].v() == 0. {
            if (o[k] - D::from(p.lower[k])).v() < 0. || (o[k] - D::from(p.upper[k])).v() > 0. {
                return None;
            }
        } else {
            let mut a = (D::from(p.lower[k]) - o[k]) / d[k];
            let mut b = (D::from(p.upper[k]) - o[k]) / d[k];
            if (a - b).v() > 0. {
                std::mem::swap(&mut a, &mut b);
            }
            if (a - ta).v() > 0. {
                ta = a;
            }
            if (b - tb).v() < 0. {
                tb = b;
            }
        }
    }
    if (ta - tb).v() > 0. {
        return None;
    }
    // Clip to the cylindrical domain before scanning the z enclosure.
    let r = D::from(p.outer_radius);
    let aa = d[0] * d[0] + d[1] * d[1];
    let bb = o[0] * d[0] + o[1] * d[1];
    let cc = o[0] * o[0] + o[1] * o[1] - r * r;
    if aa.v() > 0. {
        let disc = bb * bb - aa * cc;
        if disc.v() < 0. {
            return None;
        }
        let root = disc.sqrt();
        let lo = (-bb - root) / aa;
        let hi = (-bb + root) / aa;
        if (lo - ta).v() > 0. {
            ta = lo;
        }
        if (hi - tb).v() < 0. {
            tb = hi;
        }
    } else if cc.v() > 0. {
        return None;
    }
    if (ta - tb).v() > 0. {
        return None;
    }
    let oo = add(o, scale(d, ta));
    let len = tb - ta;
    let at = |t: D<E>| -> Option<(D<E>, D<E>)> {
        let q = add(oo, scale(d, t));
        let z = eval(s, q, r)?;
        Some((q[2] - z[0], d[2] - z[1] * d[0] - z[2] * d[1]))
    };
    let n = ((len.v() * 64. / r.v()).ceil() as u32).clamp(8, 512);
    let step = len / D::f(n as f32);
    let tol = D::f(2e-13) * (D::f(1.) + len + dot(oo, oo).sqrt());
    let refine = |mut lo: D<E>, mut hi: D<E>, fa: D<E>| -> Option<D<E>> {
        let mut t = (lo + hi) * D::f(0.5);
        for _ in 0..64 {
            let (ft, dt) = at(t)?;
            if ft.v() == 0. {
                return Some(t);
            }
            if (ft * fa).v() < 0. {
                hi = t;
            } else {
                lo = t;
            }
            if (hi - lo - tol).v() <= 0. {
                break;
            }
            let next = t - ft / dt;
            t = if (next - lo).v() > 0. && (next - hi).v() < 0. {
                next
            } else {
                (lo + hi) * D::f(0.5)
            };
        }
        Some((lo + hi) * D::f(0.5))
    };
    let mut a = D::f(0.);
    let mut fa = at(a)?.0;
    for i in 1..=n {
        let b = if i == n { len } else { step * D::f(i as f32) };
        let fb = at(b)?.0;
        let mut roots = [None; 2];
        if fa.v() == 0. {
            roots[0] = Some(a);
        } else if (fa * fb).v() < 0. {
            roots[0] = refine(a, b, fa);
        } else if fb.v() == 0. {
            roots[0] = Some(b);
        } else {
            let da = at(a)?.1;
            let db = at(b)?.1;
            if (da * db).v() < 0. {
                let (mut lo, mut hi) = (a, b);
                for _ in 0..64 {
                    let mid = (lo + hi) * D::f(0.5);
                    if (at(mid)?.1 * da).v() > 0. {
                        lo = mid;
                    } else {
                        hi = mid;
                    }
                    if (hi - lo - tol).v() <= 0. {
                        break;
                    }
                }
                let t = (lo + hi) * D::f(0.5);
                let ft = at(t)?.0;
                if ft.v() == 0. {
                    roots[0] = Some(t);
                } else if (ft * fa).v() < 0. {
                    roots = [refine(a, t, fa), refine(t, b, ft)];
                }
            }
        }
        for t in roots.into_iter().flatten() {
            let tt = ta + t;
            if tt.v() <= 0. || (tt - limit).v() > 0. {
                continue;
            }
            let q = add(oo, scale(d, t));
            let rr = q[0] * q[0] + q[1] * q[1];
            let ri = D::from(p.inner_radius);
            if (rr - ri * ri).v() < 0. || (rr - r * r).v() > 0. || !super::kernel::mask(p, q) {
                continue;
            }
            let z = eval(s, q, r)?;
            let normal = unit([-z[1], -z[2], D::f(1.)]);
            return Some(Hit {
                point: q.map(|v| v.0),
                normal: normal.map(|v| v.0),
                distance: tt.0,
                primitive: 0,
                reserved: 0,
            });
        }
        a = b;
        fa = fb;
    }
    None
}
