//! Extended-sag intersection of the stray caster (`sgintersect` in
//! stray_ops.metal): a double-float scan over the clipped ray interval, each
//! bracketed root refined by safeguarded Newton/bisection. Even aspheres in
//! the plain radial form are evaluated as `S(u)`, `u = x² + y²`, and their
//! roots located by f32 Newton and polished by two double-float Newton
//! steps, as on the device; every other extended sag uses the jets of
//! `surface_math`. The device's f32 pre-scan only skips cells without a
//! root or tangency, so the plain double-float scan here finds the same
//! roots.
use super::{math::*, records::*};
use crate::{FpEnv, consts::*, fx, fx::F, surface_math, trace_kernel};

/// `sradial`: an even asphere whose sag is the plain radial form (its
/// overlay is the zero one the stray packing adds).
fn radial<E: FpEnv>(s: &SurfaceData) -> bool {
    let f = &s.surface;
    if f.shape != MKL_SHAPE_EVEN_ASPHERE || f.flags & MKL_SURF_HAS_SAG_OVERLAY == 0 {
        return false;
    }
    let Some(base) = (f.coefficient_count as usize).checked_sub(MKL_OVERLAY_COUNT as usize) else {
        return false;
    };
    (MKL_OVERLAY_COS..MKL_OVERLAY_COUNT).all(|k| {
        let c = s.coefficients[base + k as usize];
        E::eq(c.hi, 0.) && E::eq(c.lo, 0.)
    })
}
/// `sgeval_radial`: `S(u) = c·u/(1 + w) + P(u)`, `w = √(1 − (1+k)c²u)`, and
/// the gradient `(x, y)·(c/w + 2P'(u))`, in double-float.
fn eval_radial<E: FpEnv>(s: &SurfaceData, x: D<E>, y: D<E>) -> Option<[D<E>; 3]> {
    let f = &s.surface;
    let u = x * x + y * y;
    let c = D::from(crate::Df32::new(f.curvature, f.curvature_lo));
    let kap = D::from(crate::Df32::new(f.conic, f.conic_lo)) + D::f(1.);
    let arg = D::f(1.) - kap * c * c * u;
    if arg.v() < 0. {
        return None;
    }
    let w = arg.sqrt();
    let (p, dp) = trace_kernel::asphere_poly_dp_df(f, fx::D::<E>::from_df(u.0));
    let g = c / w + D::f(2.) * D::from(dp.0);
    let z = c * u / (D::f(1.) + w) + D::from(p.0);
    Some([z, x * g, y * g])
}
/// `fgeval_radial`: the sag and `2S'(u)` in plain f32.
fn eval_radial_f32<E: FpEnv>(s: &SurfaceData, x: F<E>, y: F<E>) -> Option<(F<E>, F<E>)> {
    let f = &s.surface;
    let u = x * x + y * y;
    let c = F::new(f.curvature);
    let arg = 1.0 - (1.0 + F::new(f.conic)) * c * c * u;
    if arg < 0.0 {
        return None;
    }
    let w = arg.sqrt();
    let rho = u * f.asphere_inv_norm2;
    let (mut p, mut dp) = (F::ZERO, F::ZERO);
    for j in (0..(f.num_asphere as usize).min(crate::layout::MAX_ASPHERE)).rev() {
        dp = dp * rho + F::new(f.asphere[j]) * ((j + 2) as f32);
        p = p * rho + f.asphere[j];
    }
    let z = c * u / (1.0 + w) + p * rho * rho;
    let g = c / w + 2.0 * dp * rho * f.asphere_inv_norm2;
    Some((z, g))
}
/// `sgeval`: the sag and its gradient at `p`, clamped to the radius `r`.
fn eval<E: FpEnv>(s: &SurfaceData, p: V<E>, r: D<E>, rad: bool) -> Option<[D<E>; 3]> {
    let radius = (p[0] * p[0] + p[1] * p[1]).sqrt();
    let k = if (radius - r).v() > 0. {
        r / radius
    } else {
        D::f(1.)
    };
    if rad {
        return eval_radial(s, p[0] * k, p[1] * k);
    }
    let j = surface_math::evaluate(
        &s.surface,
        &s.coefficients,
        fx::D::<E>::from_df((p[0] * k).0),
        fx::D::<E>::from_df((p[1] * k).0),
    )?;
    Some([D::from(j.v.0), D::from(j.x.0), D::from(j.y.0)])
}
/// `fgat` of the radial evaluation: the plain-f32 residual and slope at ray
/// parameter `t` from `o` along `d`.
fn residual_f32<E: FpEnv>(
    s: &SurfaceData,
    o: [F<E>; 3],
    d: [F<E>; 3],
    radius: F<E>,
    t: F<E>,
) -> Option<(F<E>, F<E>)> {
    let q: [F<E>; 3] = std::array::from_fn(|i| o[i] + d[i] * t);
    let r = (q[0] * q[0] + q[1] * q[1]).sqrt();
    let k = if r > radius { radius / r } else { F::ONE };
    let (z, g) = eval_radial_f32(s, q[0] * k, q[1] * k)?;
    let (jx, jy) = (q[0] * k * g, q[1] * k * g);
    let f = q[2] - z;
    let df = d[2] - jx * d[0] - jy * d[1];
    (f.is_finite() && df.is_finite()).then_some((f, df))
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
    let rad = radial::<E>(s);
    let at = |t: D<E>| -> Option<(D<E>, D<E>)> {
        let q = add(oo, scale(d, t));
        let z = eval(s, q, r, rad)?;
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
    // `sgnewton`: the root of a bracket with a sign change (f(lo) of the sign
    // of `fa`) by safeguarded f32 Newton, polished by two double-float Newton
    // steps; `refine` when the polished root leaves the bracket.
    let (fo, fd) = (oo.map(|v| F::new(v.0.hi)), d.map(|v| F::new(v.0.hi)));
    let newton = |lo: D<E>, hi: D<E>, fa: D<E>| -> Option<D<E>> {
        let (mut a, mut b) = (F::<E>::new(lo.v()), F::<E>::new(hi.v()));
        let mut t = 0.5 * (a + b);
        let negative = F::<E>::new(fa.v()) < 0.0;
        for _ in 0..16 {
            let Some((f, df)) = residual_f32(s, fo, fd, F::new(r.v()), t) else {
                break;
            };
            if (f < 0.0) == negative {
                a = t;
            } else {
                b = t;
            }
            let next = t - f / df;
            let next = if next > a && next < b {
                next
            } else {
                0.5 * (a + b)
            };
            let done = (next - t).abs() <= 1e-6 * (1.0 + t.abs());
            t = next;
            if done {
                break;
            }
        }
        let mut tb = D::f(t.v());
        for _ in 0..2 {
            let Some((ft, dt)) = at(tb) else {
                return refine(lo, hi, fa);
            };
            let step = ft / dt;
            if step.v().is_finite() {
                tb = tb - step;
            }
        }
        if (tb - lo).v() < 0. || (tb - hi).v() > 0. {
            return refine(lo, hi, fa);
        }
        Some(tb)
    };
    let root = |lo, hi, fa| {
        if rad {
            newton(lo, hi, fa)
        } else {
            refine(lo, hi, fa)
        }
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
            roots[0] = root(a, b, fa);
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
                    roots = [root(a, t, fa), root(t, b, ft)];
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
            let z = eval(s, q, r, rad)?;
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
