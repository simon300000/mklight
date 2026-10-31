use super::{coating, math::*, *};
use crate::{Df32 as D, consts::*};
use bytemuck::Zeroable;
type V = [D; 3];
type E = [C; 3];
fn dot(a: V, b: V) -> D {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V, b: V) -> V {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn norm(a: V) -> V {
    let n = dot(a, a).sqrt();
    a.map(|v| v / n)
}
fn sub(a: V, b: V) -> V {
    std::array::from_fn(|i| a[i] - b[i])
}
fn scale(a: V, s: D) -> V {
    a.map(|v| v * s)
}
fn rot(a: [D; 9], v: V) -> V {
    std::array::from_fn(|i| dot([a[3 * i], a[3 * i + 1], a[3 * i + 2]], v))
}
fn tr(a: [D; 9]) -> [D; 9] {
    std::array::from_fn(|i| a[i % 3 * 3 + i / 3])
}
fn cdot(a: V, b: E) -> C {
    C::real(a[0]) * b[0] + C::real(a[1]) * b[1] + C::real(a[2]) * b[2]
}
fn perp(k: V) -> V {
    let a = k.map(|x| value(x).abs());
    let i = if a[0] <= a[1] && a[0] <= a[2] {
        0
    } else if a[1] <= a[2] {
        1
    } else {
        2
    };
    let mut axis = [D::ZERO; 3];
    axis[i] = d(1.0);
    norm(cross(k, axis))
}
fn sv(k: V, n: V) -> V {
    let c = cross(k, n);
    let c = sub(c, scale(k, dot(c, k)));
    if value(dot(c, c)) > 1e-24 {
        norm(c)
    } else {
        perp(k)
    }
}
fn ortho(s: V, k: V) -> V {
    let a = sub(s, scale(k, dot(s, k)));
    if value(dot(a, a)) > 0.25 {
        norm(a)
    } else {
        perp(k)
    }
}
fn basis(k: V, r: [D; 9]) -> [V; 2] {
    let v = rot(tr(r), k);
    let sg = if value(v[2]) < 0.0 { d(-1.0) } else { d(1.0) };
    let (x, y, z) = (v[0], sg * v[1], sg * v[2]);
    let den = d(1.0) + z;
    [
        rot(r, [d(1.0) - x * x / den, sg * (-x * y / den), sg * (-x)]),
        rot(r, [-x * y / den, sg * (d(1.0) - y * y / den), sg * (-y)]),
    ]
}
fn apply(e: E, ki: V, ko: V, s: V, as_: C, ap: C) -> E {
    let pi = cross(ki, s);
    let so = ortho(s, ko);
    let po = cross(ko, so);
    let es = cdot(s, e) * as_;
    let ep = cdot(pi, e) * ap;
    let ek = cdot(ki, e);
    std::array::from_fn(|i| C::real(so[i]) * es + C::real(po[i]) * ep + C::real(ko[i]) * ek)
}
fn intensity(a: C, b: C, st: [D; 4]) -> D {
    let aa = a.norm2();
    let bb = b.norm2();
    let c = a.conj() * b;
    ((aa + bb) * d(0.5) * st[0] + (aa - bb) * d(0.5) * st[1] + c.re * st[2] - c.im * st[3]) / st[0]
}
fn energy(e: [E; 2], st: [D; 4]) -> D {
    (0..3).fold(D::ZERO, |s, i| s + intensity(e[0][i], e[1][i], st))
}
fn properties(j: Matrix) -> ([D; 8], bool, bool) {
    let aa = j[0].norm2() + j[2].norm2();
    let bb = j[1].norm2() + j[3].norm2();
    let b = j[0].conj() * j[1] + j[2].conj() * j[3];
    let t = aa + bb;
    let m = [(aa - bb) * d(0.5), b.re, -b.im];
    let mn = dot(m, m).sqrt();
    let top = t * d(0.5) + mn;
    let det = (j[0] * j[3] - j[1] * j[2]).norm();
    let mut out = [D::ZERO; 8];
    if value(t) > 0.0 {
        out[0] = min(d(1.0), d(2.0) * mn / t);
    }
    if value(mn) > 0.0 {
        out[2..5].copy_from_slice(&scale(m, d(1.0) / mn));
    }
    if value(top) == 0.0 {
        return (out, false, false);
    }
    let contrast = det * det / (top * top);
    // The reference's physical undefined threshold stays 1e-24. Below this
    // numerical envelope request f64, including the undefined-mask decision.
    if value(contrast) < 1e-18 {
        return (out, false, true);
    }
    let hscale = d(1.0) / (t + d(2.0) * det).sqrt();
    let h00 = (aa + det) * hscale;
    let h11 = (bb + det) * hscale;
    let h01 = b * C::real(hscale);
    let mut u = product(j, [C::real(h11), -h01, -h01.conj(), C::real(h00)]);
    for a in &mut u {
        *a = *a / C::real(det);
    }
    let phase = (u[0] * u[3] - u[1] * u[2]).sqrt();
    let phase = phase / C::real(phase.norm());
    for a in &mut u {
        *a = *a / phase;
    }
    let mut c = (u[0] + u[3]).re * d(0.5);
    let mut v = [
        -(u[0] - u[3]).im * d(0.5),
        -(u[1] + u[2]).im * d(0.5),
        (u[2] - u[1]).re * d(0.5),
    ];
    if value(c) < 0.0 {
        c = -c;
        v = scale(v, d(-1.0));
    }
    let vn = dot(v, v).sqrt();
    out[1] = d(2.0 * value(vn).atan2(value(c)));
    if value(vn) > 0.0 {
        out[5..8].copy_from_slice(&scale(v, -d(1.0) / vn));
    }
    (out, true, false)
}
fn record(hi: &[f32], lo: &[f32], slot: usize, n: usize, i: usize, k: usize) -> D {
    let a = (slot * n + i) * 13 + k;
    D::new(hi[a], lo[a])
}
#[allow(clippy::too_many_arguments)]
pub fn evaluate(
    surfaces: &[SurfaceSpec],
    layers: &[CoatingLayer],
    hi: &[f32],
    lo: &[f32],
    n: usize,
    i: usize,
    wl: usize,
    nw: usize,
    st: [D; 4],
) -> PolarizationValue {
    let slots = surfaces.len() / nw;
    let spec = |j: usize| surfaces[j * nw + wl];
    let vec = |j, k| std::array::from_fn(|a| record(hi, lo, j, n, i, k + a));
    let mut ki = norm(rot(spec(0).rotation, vec(0, 3)));
    let basis_in = basis(ki, spec(0).rotation);
    let mut e = basis_in.map(|v| v.map(C::real));
    let mut g = e;
    let mut scalar = d(1.0);
    let mut sp = [d(1.0); 2];
    let mut loss = [D::ZERO; 3];
    let mut out = PolarizationValue::zeroed();
    for j in 1..slots {
        let s = spec(j);
        if s.coating.offset as usize + s.coating.count as usize > layers.len()
            || s.coating.count > MKL_RAD_MAX_LAYERS
        {
            out.status = MKL_RAD_REFERENCE;
            return out;
        }
        let ko = norm(rot(s.rotation, vec(j, 3)));
        let normal = norm(rot(s.rotation, vec(j, 8)));
        let pp = rot(s.transfer, vec(j - 1, 0));
        let delta = sub(vec(j, 0), std::array::from_fn(|a| pp[a] + s.translation[a]));
        let tau = if value(s.alpha) > 0.0 {
            exp(-s.alpha * dot(delta, delta).sqrt())
        } else {
            d(1.0)
        };
        loss[2] = loss[2] + energy(e, st) * (d(1.0) - tau);
        let at = C::real(tau.sqrt());
        for col in &mut e {
            for z in col {
                *z = *z * at;
            }
        }
        scalar = scalar * tau;
        sp = sp.map(|v| v * tau);
        if s.kind != 0 {
            let mut axis = sv(ki, normal);
            let mut mid = ko;
            let mut coeff = [C::real(d(1.0)); 2];
            let mut geom = [C::real(d(1.0)); 2];
            let mut power = [d(1.0); 2];
            if s.kind == 3 {
                axis = sv(ki, ko);
            } else {
                let ci = dot(ki, normal);
                let ci = min(d(1.0), if value(ci) < 0.0 { -ci } else { ci });
                let a = coating::evaluate(
                    s.coating,
                    &layers[s.coating.offset as usize..][..s.coating.count as usize],
                    ci,
                );
                if a.status != 0 {
                    out.status = MKL_RAD_REFERENCE;
                    return out;
                }
                let mirror = s.kind == 2 || s.kind == 4;
                let offset = if mirror { 0 } else { 2 };
                for pol in 0..2 {
                    coeff[pol] = C::new(
                        a.amplitudes[2 * (offset + pol)],
                        a.amplitudes[2 * (offset + pol) + 1],
                    );
                    power[pol] = a.powers[offset + pol];
                }
                if mirror {
                    geom[0] = C::real(d(-1.0));
                }
                if s.kind == 4 {
                    mid = norm(sub(ki, scale(normal, d(2.0) * dot(ki, normal))));
                }
                let ws = intensity(cdot(axis, e[0]), cdot(axis, e[1]), st);
                let pi = cross(ki, axis);
                let wp = intensity(cdot(pi, e[0]), cdot(pi, e[1]), st);
                if !mirror {
                    loss[0] = loss[0] + ws * a.powers[0] + wp * a.powers[1];
                }
                loss[1] = loss[1] + ws * a.powers[4] + wp * a.powers[5];
            }
            scalar = scalar * (power[0] + power[1]) * d(0.5);
            for p in 0..2 {
                sp[p] = sp[p] * power[p];
            }
            for p in 0..2 {
                e[p] = apply(e[p], ki, mid, axis, coeff[0], coeff[1]);
                g[p] = apply(g[p], ki, mid, axis, geom[0], geom[1]);
            }
            if s.kind == 4 {
                let axis = sv(mid, ko);
                for p in 0..2 {
                    e[p] = apply(e[p], mid, ko, axis, C::real(d(1.0)), C::real(d(1.0)));
                    g[p] = apply(g[p], mid, ko, axis, C::real(d(1.0)), C::real(d(1.0)));
                }
            }
        }
        ki = ko;
    }
    let last = spec(slots - 1);
    let bout = basis(ki, last.rotation);
    let mut jones = [C::default(); 4];
    let mut proper = jones;
    for a in 0..2 {
        for b in 0..2 {
            jones[2 * a + b] = cdot(bout[a], e[b]);
            proper[2 * a + b] = cdot(g[a].map(|z| z.re), e[b]);
        }
    }
    for k in 0..4 {
        out.jones[2 * k] = jones[k].re;
        out.jones[2 * k + 1] = jones[k].im;
        out.proper[2 * k] = proper[k].re;
        out.proper[2 * k + 1] = proper[k].im;
    }
    let rt = tr(last.rotation);
    for a in 0..2 {
        for k in 0..3 {
            let z = cdot([rt[3 * k], rt[3 * k + 1], rt[3 * k + 2]], e[a]);
            out.fields[6 * a + 2 * k] = z.re;
            out.fields[6 * a + 2 * k + 1] = z.im;
        }
    }
    out.direction = rot(rt, ki);
    out.power = [
        scalar,
        (sp[0] + sp[1]) * d(0.5),
        energy(e, st),
        energy(e, [d(1.0), D::ZERO, D::ZERO, D::ZERO]),
        loss[0],
        loss[1],
        loss[2],
    ];
    let (props, defined, ref1) = properties(proper);
    let (jp, jdefined, ref2) = properties(jones);
    out.properties[..8].copy_from_slice(&props);
    out.properties[8] = jp[1];
    out.defined = defined as u32 + 2 * jdefined as u32;
    if ref1 || ref2 {
        out.status = MKL_RAD_REFERENCE;
    }
    out
}
