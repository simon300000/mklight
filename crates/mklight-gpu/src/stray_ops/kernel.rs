//! Portable source and analytic compute-BVH specification. No optical
//! transport or approximate replacement of unsupported sag families.
use super::{math::*, records::*};
use crate::{Df32, FpEnv};
use bytemuck::Zeroable;
use mklight_core::rng::philox4x32_10;

pub fn generate<E: FpEnv>(s: Source, cdf: &[Df32], seed: u64, id: u64) -> Ray {
    let mut draws = 0u32;
    let mut uniform = || {
        let w = philox4x32_10(
            [draws / 4, 0, id as u32, (id >> 32) as u32],
            [seed as u32, (seed >> 32) as u32],
        )[(draws % 4) as usize];
        draws += 1;
        // Exactly the f64 reference mapping, including the lower half ulp.
        (D::<E>::f((w >> 8) as f32) + D::f(0.5)) * D::f(1. / 16777216.)
    };
    let mut wl = 0;
    if s.wavelength_count > 1 {
        let u = uniform();
        wl = cdf
            .iter()
            .position(|&v| (u - D::from(v)).v() < 0.)
            .unwrap_or(cdf.len() - 1) as u32;
    }
    let (u, v) = (uniform(), uniform());
    let tau = D::from(Df32::new(std::f32::consts::TAU, -1.7484555e-7));
    let (sn, cs) = sincos(tau * v);
    let axis = vec(s.axis);
    let (t1, t2) = (vec(s.tangent[0]), vec(s.tangent[1]));
    let radius = D::from(s.radius);
    let mut origin = vec(s.center);
    let direction;
    if s.kind == 2 {
        let h = u * D::from(s.cone);
        let st = (h * (D::f(2.) - h)).sqrt();
        direction = unit(add(
            add(scale(t1, st * cs), scale(t2, st * sn)),
            scale(axis, D::f(1.) - h),
        ));
    } else {
        let ri = D::from(s.inner_radius);
        let r = if s.kind == 3 {
            (ri * ri + u * (radius * radius - ri * ri)).sqrt()
        } else {
            radius * u.sqrt()
        };
        origin = add(add(origin, scale(t1, r * cs)), scale(t2, r * sn));
        if s.kind == 3 {
            let b = D::from(s.cone) * uniform().sqrt();
            let (s2, c2) = sincos(tau * uniform());
            let z = ((D::f(1.) - b) * (D::f(1.) + b)).sqrt();
            direction = unit(add(
                add(scale(t1, b * c2), scale(t2, b * s2)),
                scale(axis, z),
            ));
        } else {
            origin = sub(origin, scale(axis, D::from(s.distance)));
            direction = axis;
        }
    }
    Ray {
        origin: origin.map(|v| v.0),
        direction: direction.map(|v| v.0),
        id: [id as u32, (id >> 32) as u32],
        draws,
        wavelength: wl,
    }
}

fn slab<E: FpEnv>(lo: [Df32; 3], hi: [Df32; 3], o: V<E>, d: V<E>, limit: D<E>) -> Option<D<E>> {
    let (mut a, mut b) = (D::f(0.), limit);
    for i in 0..3 {
        if d[i].v() == 0. {
            if (o[i] - D::from(lo[i])).v() < 0. || (o[i] - D::from(hi[i])).v() > 0. {
                return None;
            }
        } else {
            let mut x = (D::from(lo[i]) - o[i]) / d[i];
            let mut y = (D::from(hi[i]) - o[i]) / d[i];
            if (x - y).v() > 0. {
                std::mem::swap(&mut x, &mut y);
            }
            if (x - a).v() > 0. {
                a = x;
            }
            if (y - b).v() < 0. {
                b = y;
            }
            if (a - b).v() > 0. {
                return None;
            }
        }
    }
    Some(a)
}
pub(super) fn mask<E: FpEnv>(p: Primitive, q: V<E>) -> bool {
    let x = q[0];
    let y = q[1];
    let r = x * x + y * y;
    let a = D::from(p.aperture[0]);
    let b = D::from(p.aperture[1]);
    let pass = match p.mask & 255 {
        0 => true,
        1 => (r - a * a).v() <= 0.,
        2 => (r - a * a).v() >= 0. && (r - b * b).v() <= 0.,
        3 => (x - a).v() <= 0. && (x + a).v() >= 0. && (y - b).v() <= 0. && (y + b).v() >= 0.,
        4 => ((x / a) * (x / a) + (y / b) * (y / b) - D::f(1.)).v() <= 0.,
        _ => unreachable!(),
    };
    pass ^ (p.mask & 256 != 0)
}
fn intersect<E: FpEnv>(p: Primitive, ray: Ray, limit: D<E>) -> Option<Hit> {
    let o = rotate(p.rotation, sub(vec(ray.origin), vec(p.origin)));
    let d = rotate(p.rotation, vec(ray.direction));
    let shift = slab(p.lower, p.upper, o, d, limit)?;
    let oo = add(o, scale(d, shift));
    let (mut roots, mut count) = ([D::f(0.); 2], 1);
    let c = D::from(p.curvature);
    let k = D::f(1.) + D::from(p.conic);
    if p.kind == 0 {
        if d[2].v() == 0. {
            return None;
        }
        roots[0] = -oo[2] / d[2];
    } else {
        let (a, b, cc) = if p.kind == 2 {
            let r = D::from(p.outer_radius);
            (
                d[0] * d[0] + d[1] * d[1],
                oo[0] * d[0] + oo[1] * d[1],
                oo[0] * oo[0] + oo[1] * oo[1] - r * r,
            )
        } else {
            (
                c * (d[0] * d[0] + d[1] * d[1] + k * d[2] * d[2]),
                c * (oo[0] * d[0] + oo[1] * d[1] + k * oo[2] * d[2]) - d[2],
                c * (oo[0] * oo[0] + oo[1] * oo[1] + k * oo[2] * oo[2]) - D::f(2.) * oo[2],
            )
        };
        if a.v() == 0. {
            if b.v() == 0. {
                return None;
            }
            roots[0] = -cc / (D::f(2.) * b);
        } else {
            let disc = b * b - a * cc;
            if disc.v() < 0. {
                return None;
            }
            let sq = disc.sqrt();
            let q = -(b + if b.v() < 0. { -sq } else { sq });
            if q.v() == 0. {
                roots[0] = -b / a;
            } else {
                roots = [q / a, cc / q];
                count = 2;
                if (roots[0] - roots[1]).v() > 0. {
                    roots.swap(0, 1);
                }
            }
        }
    }
    for root in &roots[..count] {
        let t = shift + *root;
        if t.v() <= 0. || (t - limit).v() > 0. {
            continue;
        }
        let q = add(oo, scale(d, *root));
        let normal = if p.kind == 2 {
            if (q[2] - D::from(p.lower[2])).v() < 0. || (q[2] - D::from(p.upper[2])).v() > 0. {
                continue;
            }
            unit([q[0], q[1], D::f(0.)])
        } else {
            if (k * c * q[2] - D::f(1.)).v() > 1e-9 {
                continue;
            }
            let r = q[0] * q[0] + q[1] * q[1];
            let ri = D::from(p.inner_radius);
            let ro = D::from(p.outer_radius);
            if (r - ri * ri).v() < 0. || (r - ro * ro).v() > 0. || !mask(p, q) {
                continue;
            }
            unit([-c * q[0], -c * q[1], D::f(1.) - c * k * q[2]])
        };
        return Some(Hit {
            point: q.map(|x| x.0),
            normal: normal.map(|x| x.0),
            distance: t.0,
            primitive: 0,
            reserved: 0,
        });
    }
    None
}
/// Stackless preorder BVH. Equal distance selects the smaller primitive ID,
/// independent of node order. Scene topology is validated at preparation.
pub fn closest_shapes<E: FpEnv>(
    prims: &[Primitive],
    nodes: &[Node],
    shapes: &[SurfaceData],
    ray: Ray,
) -> Hit {
    let mut out = Hit {
        primitive: u32::MAX,
        ..Hit::zeroed()
    };
    let mut limit = D::<E>::f(f32::MAX);
    let mut i = 0;
    while i < nodes.len() {
        let node = nodes[i];
        if slab(
            node.lower,
            node.upper,
            vec(ray.origin),
            vec(ray.direction),
            limit,
        )
        .is_none()
        {
            i = node.escape as usize;
            continue;
        }
        if node.primitive != u32::MAX {
            let p = prims[node.primitive as usize];
            let hit = if p.kind == 3 {
                super::sag::intersect(p, &shapes[node.primitive as usize], ray, limit)
            } else {
                intersect(p, ray, limit)
            };
            if let Some(mut h) = hit {
                let delta = (D::from(h.distance) - limit).v();
                if out.primitive == u32::MAX
                    || delta < 0.
                    || (delta == 0. && node.primitive < out.primitive)
                {
                    h.primitive = node.primitive;
                    limit = D::from(h.distance);
                    out = h;
                }
            }
        }
        i += 1;
    }
    out
}
