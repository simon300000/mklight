//! CPU-f32 transport specification. Only library transcendentals are compared
//! by tolerance; source IDs, history, stack order and counter draws are exact.
use super::service::{coating, math::exp};
use super::*;
use crate::{Df32 as D, FpEnv};
use bytemuck::Zeroable;
type V = [D; 3];
fn f(x: f32) -> D {
    D::from_f32(x)
}
fn v(x: D) -> f32 {
    x.hi + x.lo
}
fn min(a: D, b: D) -> D {
    if v(a - b) < 0. { a } else { b }
}
fn max(a: D, b: D) -> D {
    if v(a - b) > 0. { a } else { b }
}
fn dot(a: V, b: V) -> D {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn scale(a: V, b: D) -> V {
    a.map(|a| a * b)
}
fn add(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] + b[i])
}
fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn unit(a: V) -> V {
    scale(a, f(1.) / dot(a, a).sqrt())
}
fn rotate(m: [D; 9], a: V) -> V {
    std::array::from_fn(|i| m[i] * a[0] + m[3 + i] * a[1] + m[6 + i] * a[2])
}
fn world(m: [D; 9], a: V) -> V {
    std::array::from_fn(|i| m[3 * i] * a[0] + m[3 * i + 1] * a[1] + m[3 * i + 2] * a[2])
}
fn onb(n: V) -> (V, V) {
    let s = f(if v(n[2]) < 0. { -1. } else { 1. });
    let a = -f(1.) / (s + n[2]);
    let b = n[0] * n[1] * a;
    (
        [f(1.) + s * n[0] * n[0] * a, s * b, -s * n[0]],
        [b, s + n[1] * n[1] * a, -n[1]],
    )
}
struct Rng {
    seed: u64,
    id: [u32; 2],
    draws: u64,
}
impl Rng {
    fn uniform(&mut self) -> D {
        let w = mklight_core::rng::philox4x32_10(
            [
                (self.draws / 4) as u32,
                (self.draws >> 34) as u32,
                self.id[0],
                self.id[1],
            ],
            [self.seed as u32, (self.seed >> 32) as u32],
        )[(self.draws % 4) as usize];
        self.draws += 1;
        (f((w >> 8) as f32) + f(0.5)) * f(1. / 16777216.)
    }
}
fn sincos(x: D) -> (D, D) {
    let (s, c) = super::math::sincos::<crate::Ieee>(super::math::D::from(x));
    (s.0, c.0)
}
fn tau() -> D {
    D::new(std::f32::consts::TAU, -1.7484555e-7)
}
fn book(l: &mut Ledger, s: usize, w: D) {
    l.sums[s] = l.sums[s] + w;
}
fn change(l: &mut Ledger, gain: usize, d: D) {
    book(
        l,
        gain + usize::from(v(d) < 0.),
        if v(d) < 0. { -d } else { d },
    );
}
fn push(h: &mut History, k: u32, id: u32) {
    if h.length < 6 {
        h.events[h.length as usize] = (k << 28) | id.min(0xfffffff);
        h.length += 1;
    } else {
        h.overflow = 1;
    }
}
fn spec(h: &mut History, seq: &[u32], surface: u32, designed: bool, extra: bool) {
    if seq.get(h.next_specular as usize) != Some(&surface) {
        h.flags |= 32;
    }
    if extra {
        h.flags ^= 64;
    }
    h.next_specular += if h.flags & 64 != 0 { -1 } else { 1 };
    if h.flags & 4 == 0 && designed && seq.get(h.next_seq as usize) == Some(&surface) {
        h.next_seq += 1;
    } else {
        h.flags |= 4;
    }
    if extra {
        if h.reflections < 2 {
            h.first[h.reflections as usize] = surface.min(65534);
        }
        h.reflections = (h.reflections + 1).min(255);
    }
}
fn class(h: History, count: usize) -> u32 {
    if h.flags & 1 != 0 {
        2
    } else if h.flags & 2 != 0 {
        3
    } else if h.flags & 8 != 0 {
        4
    } else if (h.reflections >= 2 || h.flags & 128 != 0)
        && h.flags & 96 == 0
        && h.next_specular == count as i32 - 1
    {
        1
    } else if h.reflections == 0
        && h.flags & 132 == 0
        && (count == 0 || h.next_seq as usize + 1 == count)
    {
        0
    } else {
        4
    }
}
fn mask(mask: u32, ap: [D; 2], p: V) -> bool {
    let [x, y, _] = p;
    let [a, b] = ap;
    let r = x * x + y * y;
    let pass = match mask & 255 {
        0 => true,
        1 => v(r - a * a) <= 0.,
        2 => v(r - a * a) >= 0. && v(r - b * b) <= 0.,
        3 => v(x - a) <= 0. && v(x + a) >= 0. && v(y - b) <= 0. && v(y + b) >= 0.,
        4 => v((x / a) * (x / a) + (y / b) * (y / b) - f(1.)) <= 0.,
        _ => unreachable!(),
    };
    pass != (mask & 256 != 0)
}
fn tis(m: Material, t: &[D], spec: V, n: V) -> D {
    if m.bsdf_kind == 0 {
        return f(0.);
    }
    if m.tis_count == 0 {
        return m.bsdf[0];
    }
    let c = min(f(1.), max(f(-1.), dot(spec, n)));
    let beta = v(max(f(0.), (f(1.) - c) * (f(1.) + c)).sqrt());
    let x = beta.clamp(0., 1.).asin() * (2. / std::f32::consts::PI) * (m.tis_count - 1) as f32;
    let j = (x.floor() as usize).min(m.tis_count as usize - 2);
    let t = &t[m.tis_start as usize..];
    let ff = f(x - j as f32);
    let f2 = ff * ff;
    let f3 = f2 * ff;
    let (p1, p2) = (t[j], t[j + 1]);
    let p0 = t[if j == 0 { 1 } else { j - 1 }];
    let p3 = if j + 2 < m.tis_count as usize {
        t[j + 2]
    } else {
        f(2.) * p2 - p1
    };
    let z = (f(2.) * f3 - f(3.) * f2 + f(1.)) * p1
        + (f3 - f(2.) * f2 + ff) * (p2 - p0) * f(0.5)
        + (-f(2.) * f3 + f(3.) * f2) * p2
        + (f3 - f2) * (p3 - p1) * f(0.5);
    min(f(1.), max(f(0.), z))
}
fn expm1(x: f32) -> f32 {
    if x.abs() > 0.125 {
        return x.exp() - 1.;
    }
    let (mut t, mut s) = (x, x);
    for j in 2..=9 {
        t = t * x / j as f32;
        s += t;
    }
    s
}
fn log1p(x: f32) -> f32 {
    let y = 1. + x;
    if y == 1. { x } else { y.ln() * (x / (y - 1.)) }
}
fn abg(b: f32, g: f32, rm: f32, u: f32) -> (f32, f32) {
    if g.abs() < 1e-12 {
        return (rm * u.sqrt(), 1.);
    }
    if g < 0. {
        let r = rm * u.sqrt();
        return (r, (b + rm.powf(g)) / (b + r.powf(g)));
    }
    let e = 2. - g;
    let rc = if b == 0. { 0. } else { (b.ln() / g).exp() };
    if rc >= rm {
        let r = rm * u.sqrt();
        return (r, b / (b + r.powf(g)));
    }
    let l = if rc > 0. {
        (rm / rc).ln()
    } else {
        f32::INFINITY
    };
    let w2 = if e.abs() < 1e-9 { l } else { expm1(e * l) / e };
    let p1 = if rc == 0. || !w2.is_finite() {
        0.
    } else {
        0.5 / (0.5 + w2)
    };
    if u < p1 {
        let r = rc * (u / p1).sqrt();
        return (r, b / (b + r.powf(g)));
    }
    let u = if p1 > 0. { (u - p1) / (1. - p1) } else { u };
    let lr = if e > 1e-9 {
        rm.ln() + (u + (1. - u) * if rc > 0. { (-e * l).exp() } else { 0. }).ln() / e
    } else if e < -1e-9 {
        rc.ln() + log1p(u * expm1(e * l)) / e
    } else {
        rc.ln() + u * l
    };
    let r = lr.exp();
    let rg = r.powf(g);
    (r, rg / (b + rg))
}
fn scatter(m: Material, spec: V, n: V, rng: &mut Rng) -> Option<V> {
    let (a, b) = onb(n);
    let (bx, by) = (dot(spec, a), dot(spec, b));
    let rm = 1. + v((bx * bx + by * by).sqrt());
    let sigma = v(m.bsdf[1]);
    for _ in 0..4096 {
        let u = rng.uniform();
        let (r, accept) = match m.bsdf_kind {
            1 => (u.sqrt(), 1.),
            2 => (
                f(sigma
                    * (-2. * log1p(-v(u) * (-expm1(-0.5 * (rm / sigma) * (rm / sigma))))).sqrt()),
                1.,
            ),
            _ => (
                {
                    let (r, _) = abg(v(m.bsdf[1]), v(m.bsdf[2]), rm, v(u));
                    f(r)
                },
                abg(v(m.bsdf[1]), v(m.bsdf[2]), rm, v(u)).1,
            ),
        };
        let (sn, cs) = sincos(tau() * rng.uniform());
        let (mut x, mut y) = (r * cs, r * sn);
        if m.bsdf_kind != 1 {
            x = x + bx;
            y = y + by;
        }
        let ok = m.bsdf_kind != 3 || v(rng.uniform()) < accept;
        let rad = x * x + y * y;
        if ok && (m.bsdf_kind == 1 || v(rad) < 1.) {
            return Some(unit(add(
                add(scale(a, x), scale(b, y)),
                scale(n, max(f(0.), f(1.) - rad).sqrt()),
            )));
        }
    }
    None
}
fn kino(blaze: f32, rng: &mut Rng) -> i32 {
    if blaze == blaze.round() {
        return blaze as i32;
    }
    let center = blaze.round() as i32;
    loop {
        let k = (std::f32::consts::PI * (v(rng.uniform()) - 0.5))
            .tan()
            .round();
        if k + center as f32 > 2147483520. || k + (center as f32) < -2147483520. {
            continue;
        }
        let m = center + k as i32;
        let q = if k == 0. {
            2. * 0.5f32.atan() / std::f32::consts::PI
        } else {
            1f32.atan2((k - 0.5) * (k + 0.5) + 1.) / std::f32::consts::PI
        };
        let z = m as f32 - blaze;
        let rem = z - z.round();
        let s = if z == 0. {
            1.
        } else {
            (std::f32::consts::PI * rem).sin() / (std::f32::consts::PI * z)
        };
        if v(rng.uniform()) * 4. * q < s * s {
            return m;
        }
    }
}
fn order(m: Material, blaze: f32, rng: &mut Rng) -> i32 {
    if m.doe == 1 {
        return m.selected;
    }
    if m.doe == 2 {
        return kino(blaze, rng);
    }
    loop {
        let n = m.levels as i32;
        let a = kino(blaze, rng);
        let r = ((a % n) + n) % n;
        let k = kino(-(r as f32) / n as f32, rng);
        if (k as f32).abs() * n as f32 + (r as f32) < 2147483520. {
            return k * n + r;
        }
    }
}
struct Event {
    point: V,
    normal: V,
    refl: V,
    trans: V,
    mi: u32,
    mo: u32,
    prim: u32,
    surface: u32,
    dt: bool,
    dr: bool,
    stray: bool,
    mechanical: bool,
    tir: bool,
    side: usize,
}
fn apply(
    s: &mut State,
    w: D,
    br: usize,
    e: &Event,
    m: Material,
    p: Primitive,
    seq: &[u32],
    rng: &mut Rng,
) -> Result<(), usize> {
    let reflected = br < 2;
    let mut dir = if reflected { e.refl } else { e.trans };
    if m.doe != 0 {
        let o = order(m, v(m.blaze[if reflected { 1 + e.side } else { 0 }]), rng);
        let local = rotate(p.rotation, sub(e.point, p.origin));
        let inv = f(1.) / m.norm_radius;
        let r2 = (local[0] * local[0] + local[1] * local[1]) * inv * inv;
        let mut grad = f(0.);
        for j in (0..m.phase_count as usize).rev() {
            grad = grad * r2 + f((j + 1) as f32) * m.phase[j];
        }
        let factor =
            f(o as f32) * m.coatings[e.side].wavelength * D::new(0.001, -4.749745e-11) / tau();
        let g = world(
            p.rotation,
            [
                f(2.) * local[0] * inv * inv * grad * factor,
                f(2.) * local[1] * inv * inv * grad * factor,
                f(0.),
            ],
        );
        let d = s.ray.direction;
        let ni = m.index[e.side];
        let no = m.index[if reflected { e.side } else { 1 - e.side }];
        let t = scale(
            add(
                scale(sub(d, scale(e.normal, dot(d, e.normal))), ni),
                sub(g, scale(e.normal, dot(g, e.normal))),
            ),
            f(1.) / no,
        );
        let disc = f(1.) - dot(t, t);
        if v(disc) < 0. {
            return Err(11);
        }
        dir = unit(add(
            t,
            scale(e.normal, f(if reflected { 1. } else { -1. }) * disc.sqrt()),
        ));
        let h = &mut s.history;
        if h.order_count < 32 {
            h.orders[h.order_count as usize] = [
                (e.surface & 65535)
                    | ((reflected as u32) << 16)
                    | (((o == m.selected) as u32) << 17),
                o as u32,
            ];
            h.order_count += 1;
        } else {
            h.overflow = 1;
        }
        if o != m.selected {
            h.flags |= 128;
        }
    }
    if br & 1 != 0 {
        dir = scatter(
            m,
            dir,
            scale(e.normal, f(if reflected { 1. } else { -1. })),
            rng,
        )
        .ok_or(15usize)?;
    }
    let h = &mut s.history;
    if br & 1 != 0 {
        h.flags |= 5;
        if e.mechanical {
            h.flags |= 2;
        }
        h.scatter_events = (h.scatter_events + 1).min(65535);
        push(
            h,
            if e.mechanical {
                6
            } else if reflected {
                3
            } else {
                4
            },
            if e.mechanical { e.prim } else { e.surface },
        );
    } else if e.mechanical {
        h.flags |= 6;
        push(h, 5, e.prim);
    } else {
        spec(
            h,
            seq,
            e.surface,
            if reflected { e.dr } else { e.dt },
            reflected && e.stray,
        );
        if reflected && e.tir {
            h.flags |= 16;
        }
        if reflected && e.stray {
            push(h, if e.tir { 2 } else { 1 }, e.surface);
        }
    }
    let eps = D::new(1e-9, 2.8281932e-17) * (f(1.) + dot(e.point, e.point).sqrt());
    s.ray.origin = add(e.point, scale(e.normal, if reflected { eps } else { -eps }));
    s.ray.direction = dir;
    s.weight = w;
    s.medium = if reflected { e.mi } else { e.mo };
    Ok(())
}
fn roulette(s: &mut State, rr: D, rng: &mut Rng, l: &mut Ledger) -> bool {
    let w = s.weight;
    if v(w) <= 0. {
        return false;
    }
    if v(w - rr) >= 0. {
        return true;
    }
    if v(rng.uniform() * rr - w) < 0. {
        change(l, 16, rr - w);
        s.weight = rr;
        true
    } else {
        change(l, 16, -w);
        s.weight = f(0.);
        false
    }
}
pub(super) fn execute<E: FpEnv>(
    r: &Transport,
    x: &mut cpu::Execution,
    ctl: &JobControl,
) -> MkResult<()> {
    for i in 0..r.ray_count {
        ctl.check()?;
        let ray = kernel::generate::<E>(r.source, &r.cdf, r.seed, r.first_ray + i as u64);
        let mut rng = Rng {
            seed: r.seed,
            id: ray.id,
            draws: ray.draws as u64,
        };
        let set = r.settings;
        let mut l = Ledger::zeroed();
        l.wavelength = ray.wavelength;
        l.sums[20] = set.weight;
        let mut initial = State::zeroed();
        initial.ray = ray;
        initial.weight = set.weight;
        initial.medium = set.start_medium;
        initial.history.first = [65535; 2];
        let mut stack = vec![initial];
        let mut hits = Vec::new();
        let wl = ray.wavelength as usize;
        let nw = set.wavelengths as usize;
        let rr = set.roulette * set.weight;
        let ceiling = set.ceiling * set.weight;
        let prims = rows::<Primitive>(x, &r.primitives);
        let nodes = rows::<Node>(x, &r.nodes);
        let materials = rows::<Material>(x, &r.materials);
        let passes = rows::<Pass>(x, &r.passes);
        let media = rows::<Medium>(x, &r.media);
        let table = rows::<D>(x, &r.tis);
        let seq = rows::<u32>(x, &r.sequence);
        let layers = rows::<crate::radiometry::CoatingLayer>(x, &r.layers);
        while let Some(mut s) = stack.pop() {
            if l.events % r.event_budget as u32 == 0 {
                ctl.check()?;
            }
            let mut w = s.weight;
            if v(w) <= 0. {
                continue;
            }
            if s.events >= set.max_events {
                book(&mut l, 14, w);
                continue;
            }
            s.events += 1;
            l.events += 1;
            let hit = kernel::closest_shapes::<E>(prims, nodes, rows(x, &r.shapes), s.ray);
            let med = media[s.medium as usize * nw + wl];
            if hit.primitive == u32::MAX {
                book(&mut l, if med.ambient != 0 { 12 } else { 13 }, w);
                continue;
            }
            if v(med.alpha) > 0. {
                let t = exp(-med.alpha * hit.distance);
                book(&mut l, 0, w * (f(1.) - t));
                w = w * t;
                s.weight = w;
            }
            let p = prims[hit.primitive as usize];
            let m = materials[hit.primitive as usize * nw + wl];
            let point = add(world(p.rotation, hit.point), p.origin);
            let n = world(p.rotation, hit.normal);
            let d = s.ray.direction;
            let side = usize::from(v(dot(d, n)) < 0.);
            let ni = scale(n, f(if side == 1 { 1. } else { -1. }));
            let mut e = Event {
                point,
                normal: ni,
                refl: sub(d, scale(n, f(2.) * dot(d, n))),
                trans: d,
                mi: s.medium,
                mo: s.medium,
                prim: hit.primitive,
                surface: 0,
                dt: false,
                dr: false,
                stray: false,
                mechanical: m.kind == 0,
                tir: false,
                side,
            };
            if m.kind == 4 {
                if set.detector_accept == 2 || set.detector_accept == side as u32 {
                    if hits.len() == r.hit_capacity {
                        return unsupported("stray detection capacity");
                    }
                    hits.push(Detection {
                        history: s.history,
                        weight: w,
                        x: hit.point[0],
                        y: hit.point[1],
                        class: class(s.history, seq.len()),
                        wavelength: wl as u32,
                    });
                } else {
                    book(&mut l, 3, w);
                }
                continue;
            }
            let (probs, absorb, sink) = if m.kind == 0 {
                let t = tis(m, table, e.refl, ni);
                let rr = min(max(m.specular, f(0.)), f(1.) - t);
                (
                    [rr, t, f(0.), f(0.)],
                    max(f(0.), f(1.) - rr - t),
                    m.sink as usize,
                )
            } else {
                let mirror = m.kind == 2;
                let ps = &passes[m.pass_start as usize..][..m.pass_count as usize];
                let expected = if s.history.flags & 35 == 0 {
                    seq.get(s.history.next_specular as usize).copied()
                } else {
                    None
                };
                let rev = if !mirror && s.history.flags & 64 != 0 {
                    ps.iter()
                        .find(|p| Some(p.surface) == expected && p.side != side as u32)
                } else {
                    None
                };
                let (pass, designed) = if let Some(p) = rev {
                    (p, false)
                } else if let Some(p) = ps
                    .iter()
                    .find(|p| p.side == side as u32 && Some(p.surface) == expected)
                {
                    (p, true)
                } else if let Some(p) = ps.iter().find(|p| p.side == side as u32) {
                    (p, true)
                } else {
                    (&ps[0], false)
                };
                if mirror && !designed {
                    book(&mut l, 2, w);
                    continue;
                }
                if !mask(pass.mask, pass.aperture, hit.point) {
                    if s.history.flags & 8 == 0 {
                        push(&mut s.history, 7, pass.surface);
                    }
                    s.history.flags |= 8;
                }
                e.surface = pass.surface;
                e.dt = designed && !mirror;
                e.dr = designed && mirror;
                e.stray = !mirror;
                e.mi = m.media[side];
                e.mo = m.media[1 - side];
                let ci = min(f(1.), max(f(0.), -dot(d, ni)));
                let spec = m.coatings[side];
                let cv = coating::evaluate(
                    spec,
                    &layers[spec.offset as usize..][..spec.count as usize],
                    ci,
                );
                if cv.status != 0 {
                    return unsupported("stray coating requires reference");
                }
                let mut rr = (cv.powers[0] + cv.powers[1]) * f(0.5);
                let mut t = if mirror {
                    f(0.)
                } else {
                    (cv.powers[2] + cv.powers[3]) * f(0.5)
                };
                let mut trans = !mirror;
                if !mirror && m.doe == 0 {
                    let ratio = m.index[side] / m.index[1 - side];
                    let disc = f(1.) - ratio * ratio * (f(1.) - ci) * (f(1.) + ci);
                    if v(disc) < 0. {
                        rr = rr + t;
                        t = f(0.);
                        e.tir = true;
                        trans = false;
                    } else {
                        e.trans = unit(add(scale(d, ratio), scale(ni, ratio * ci - disc.sqrt())));
                        if m.kind == 3 {
                            let dl = rotate(p.rotation, e.trans);
                            let az = if v(dl[2]) < 0. { -dl[2] } else { dl[2] };
                            if v(az) != 0. {
                                e.trans = world(
                                    p.rotation,
                                    unit([
                                        dl[0] / az - hit.point[0] / m.focal,
                                        dl[1] / az - hit.point[1] / m.focal,
                                        f(if v(dl[2]) < 0. { -1. } else { 1. }),
                                    ]),
                                );
                            }
                        }
                    }
                }
                let sum = rr + t;
                if v(sum) > 1. {
                    rr = rr / sum;
                    t = t / sum;
                }
                let sr = if set.scatter & 1 != 0 && v(rr) > 0. {
                    tis(m, table, e.refl, ni)
                } else {
                    f(0.)
                };
                let st = if set.scatter & 2 != 0 && trans && v(t) > 0. {
                    tis(m, table, e.trans, scale(ni, f(-1.)))
                } else {
                    f(0.)
                };
                (
                    [rr * (f(1.) - sr), rr * sr, t * (f(1.) - st), t * st],
                    max(f(0.), f(1.) - rr - t),
                    1,
                )
            };
            let branches = probs.iter().filter(|&&p| v(p) > 0.).count();
            let sum = probs.into_iter().fold(f(0.), |a, b| a + b);
            if branches == 0 {
                book(&mut l, sink, w);
                continue;
            }
            if set.mode == 2 && s.splits < set.max_splits && branches >= 2 {
                book(&mut l, sink, w * absorb);
                let mut children = Vec::new();
                for (j, pj) in probs.iter().enumerate() {
                    if v(*pj) <= 0. {
                        continue;
                    }
                    let mut c = s;
                    c.splits += 1;
                    let cw = w * *pj;
                    match apply(&mut c, cw, j, &e, m, p, seq, &mut rng) {
                        Ok(()) => children.push(c),
                        Err(lost) => book(&mut l, lost, cw),
                    }
                }
                if let Some(first) = children.first_mut() {
                    roulette(first, rr, &mut rng, &mut l);
                }
                stack.extend(children.into_iter().rev());
                continue;
            }
            let (mut pick, mut next) = (0, w);
            if set.mode == 0 {
                let u = rng.uniform();
                if v(u - absorb) < 0. {
                    book(&mut l, sink, w);
                    continue;
                }
                let mut acc = absorb;
                for (j, pj) in probs.iter().enumerate() {
                    if v(*pj) > 0. {
                        acc = acc + *pj;
                        pick = j;
                        if v(u - acc) < 0. {
                            break;
                        }
                    }
                }
            } else {
                book(&mut l, sink, w * absorb);
                let raw = probs.map(|p| {
                    if v(p) > 0. {
                        max(p / sum, set.rare_floor)
                    } else {
                        f(0.)
                    }
                });
                let z = raw.into_iter().fold(f(0.), |a, b| a + b);
                let u = rng.uniform();
                let mut acc = f(0.);
                for (j, pj) in probs.iter().enumerate() {
                    if v(*pj) > 0. {
                        acc = acc + raw[j] / z;
                        pick = j;
                        if v(u - acc) < 0. {
                            break;
                        }
                    }
                }
                next = w * probs[pick] / (raw[pick] / z);
                change(&mut l, 18, next - w * sum);
            }
            if let Err(lost) = apply(&mut s, next, pick, &e, m, p, seq, &mut rng) {
                book(&mut l, lost, next);
                continue;
            }
            if !roulette(&mut s, rr, &mut rng, &mut l) {
                continue;
            }
            let copies = if v(s.weight - ceiling) > 0. {
                v(s.weight / set.weight).ceil().min(64.) as usize
            } else {
                1
            };
            s.weight = s.weight / f(copies as f32);
            stack.extend(std::iter::repeat_n(s, copies));
        }
        l.draws = rng.draws;
        l.hits = hits.len() as u32;
        let out = bytemuck::cast_slice_mut::<u32, Detection>(x.words_mut(r.output));
        let row = &mut out[i * r.hit_capacity..][..r.hit_capacity];
        row.fill(Detection::zeroed());
        row[..hits.len()].copy_from_slice(&hits);
        bytemuck::cast_slice_mut::<u32, Ledger>(x.words_mut(r.ledger))[i] = l;
        ctl.report((i + 1) as f64 / r.ray_count as f64, "stray transport");
    }
    Ok(())
}
