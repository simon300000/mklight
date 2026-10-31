//! Double-float Coddington pencils and image-frame focal-line labels.
use super::RECORD_FLOATS;
use crate::{
    Df32, FpEnv,
    consts::*,
    fx::D,
    layout::{PackedHeader, PackedSurface},
    trace_kernel as tr,
};
type V<E> = [D<E>; 3];
type M<E> = [[D<E>; 2]; 2];
type B<E> = [V<E>; 2];
fn c<E: FpEnv>(v: f32) -> D<E> {
    D::new(v, 0.0)
}
fn add<E: FpEnv>(a: V<E>, b: V<E>) -> V<E> {
    std::array::from_fn(|k| a[k] + b[k])
}
fn sub<E: FpEnv>(a: V<E>, b: V<E>) -> V<E> {
    std::array::from_fn(|k| a[k] - b[k])
}
fn scale<E: FpEnv>(a: V<E>, s: D<E>) -> V<E> {
    a.map(|v| v * s)
}
fn cross<E: FpEnv>(a: V<E>, b: V<E>) -> V<E> {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn unit<E: FpEnv>(a: V<E>) -> V<E> {
    scale(a, c(1.0) / tr::ddot(a, a).sqrt())
}
fn basis<E: FpEnv>(k: V<E>) -> B<E> {
    let mut a = sub([c(1.0), c(0.0), c(0.0)], scale(k, k[0]));
    if tr::ddot(a, a).hi() < 1e-12 {
        a = sub([c(0.0), c(1.0), c(0.0)], scale(k, k[1]));
    }
    let a = unit(a);
    [a, cross(k, a)]
}
fn mm<E: FpEnv>(a: M<E>, b: M<E>) -> M<E> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j]))
}
fn ma<E: FpEnv>(a: M<E>, b: M<E>) -> M<E> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][j] + b[i][j]))
}
fn ms<E: FpEnv>(a: M<E>, s: D<E>) -> M<E> {
    a.map(|r| r.map(|v| v * s))
}
fn mt<E: FpEnv>(a: M<E>) -> M<E> {
    [[a[0][0], a[1][0]], [a[0][1], a[1][1]]]
}
fn inv<E: FpEnv>(a: M<E>) -> Option<M<E>> {
    let d = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !d.hi().is_finite() || d.hi() == 0.0 {
        return None;
    }
    Some([[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]])
}
fn proj<E: FpEnv>(u: B<E>, e: B<E>) -> M<E> {
    std::array::from_fn(|i| std::array::from_fn(|j| tr::ddot(u[i], e[j])))
}
fn reflect<E: FpEnv>(u: B<E>, n: V<E>) -> B<E> {
    u.map(|v| sub(v, scale(n, c(2.0) * tr::ddot(v, n))))
}
fn carry<E: FpEnv>(u: B<E>, k: V<E>, k2: V<E>) -> B<E> {
    let co = tr::ddot(k, k2);
    if co.hi() > -0.9 {
        let a = cross(k, k2);
        let rot = |v| {
            add(
                add(scale(v, co), cross(a, v)),
                scale(a, tr::ddot(a, v) / (c(1.0) + co)),
            )
        };
        let u0 = unit(rot(u[0]));
        let v = rot(u[1]);
        [u0, unit(sub(v, scale(u0, tr::ddot(v, u0))))]
    } else {
        let u0 = unit(sub(u[0], scale(k2, tr::ddot(u[0], k2))));
        [u0, cross(k2, u0)]
    }
}
fn tj<E: FpEnv>(k: V<E>, u: B<E>) -> Option<M<E>> {
    let az = if k[2].hi() < 0.0 { -k[2] } else { k[2] };
    if az.hi() == 0.0 {
        return None;
    }
    let sign = c(if k[2].hi() < 0.0 { -1.0 } else { 1.0 });
    let b = [
        [c(1.0) / az, c(0.0), -sign * k[0] / (az * az)],
        [c(0.0), c(1.0) / az, -sign * k[1] / (az * az)],
    ];
    Some(proj(b, u))
}
fn form<E: FpEnv>(h: [D<E>; 3], e: B<E>) -> M<E> {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            e[i][0] * (h[0] * e[j][0] + h[1] * e[j][1])
                + e[i][1] * (h[1] * e[j][0] + h[2] * e[j][1])
        })
    })
}
fn foci<E: FpEnv>(x: M<E>, t: M<E>, u: B<E>, k: V<E>) -> Option<[D<E>; 3]> {
    let (mut a, curvature) = if let Some(ti) = inv(t) {
        (ms(mm(x, ti), c(-1.0)), false)
    } else {
        (ms(mm(t, inv(x)?), c(-1.0)), true)
    };
    a = ms(ma(a, mt(a)), c(0.5));
    let mean = (a[0][0] + a[1][1]) * c(0.5);
    let diff = (a[0][0] - a[1][1]) * c(0.5);
    let off = a[0][1];
    let r = (diff * diff + off * off).sqrt();
    let angle = 0.5 * off.hi().v().atan2(diff.hi().v());
    let e = unit(sub([c(0.0), c(1.0), c(0.0)], scale(k, k[1])));
    let w = [tr::ddot(u[0], e), tr::ddot(u[1], e)];
    let dot = c::<E>(angle.cos()) * w[0] + c::<E>(angle.sin()) * w[1];
    let plus = (dot * dot).hi() >= 0.5;
    let mut v = [mean + r, mean - r];
    if curvature {
        for q in &mut v {
            *q = if q.hi() == 0.0 {
                c(f32::INFINITY)
            } else {
                c(1.0) / *q
            };
        }
    }
    if !plus {
        v.swap(0, 1);
    }
    Some([v[0], v[1], k[2]])
}
pub fn focus<E: FpEnv>(
    h: &PackedHeader,
    s: &[PackedSurface],
    pool: &[Df32],
    hi: &[f32],
    lo: &[f32],
    n: usize,
    i: usize,
    wl: usize,
) -> Option<[Df32; 3]> {
    let rec = |slot: usize, k: usize| {
        D::<E>::new(
            hi[(slot * n + i) * RECORD_FLOATS + k],
            lo[(slot * n + i) * RECORD_FLOATS + k],
        )
    };
    let vec = |slot: usize, k: usize| std::array::from_fn(|j| rec(slot, k + j));
    let identity = [[c::<E>(1.0), c(0.0)], [c(0.0), c(1.0)]];
    let (mut x, mut t) = (identity, ms(identity, c(0.0)));
    let mut u = [[c(1.0), c(0.0), c(0.0)], [c(0.0), c(1.0), c(0.0)]];
    for j in 1..h.num_surfaces as usize {
        let mut ip = vec(j - 1, 0);
        let mut k = vec(j - 1, 3);
        tr::xf_forward(&s[j - 1], &mut ip, &mut k);
        if j == 1 {
            u = basis(k);
        } else {
            for v in &mut u {
                let mut zero = [D::ZERO; 3];
                tr::xf_forward(&s[j - 1], &mut zero, v);
            }
        }
        let d = tr::ddot(sub(vec(j, 0), ip), k);
        if j == 1 && h.flags & MKL_SYS_OBJECT_AT_INFINITY == 0 {
            x = ms(identity, d);
            t = identity;
        } else {
            x = ma(x, ms(t, d));
        }
        if j + 1 == h.num_surfaces as usize {
            return foci(x, t, u, k).map(|v| v.map(|v| v.0));
        }
        let sf = &s[j];
        if sf.interaction == MKL_INTERACT_DUMMY || sf.shape == MKL_SHAPE_COORD_BREAK {
            continue;
        }
        let n1 = rec(j - 1, 11);
        let reflect = sf.interaction == MKL_INTERACT_REFLECT;
        let n2 = if reflect {
            n1
        } else {
            D::new(sf.n_after[wl], sf.n_after_lo[wl])
        };
        let kout = vec(j, 3);
        let nrm = unit(vec(j, 8));
        if sf.shape == MKL_SHAPE_PARAXIAL {
            let z = [c(0.0), c(0.0), c(1.0)];
            let (kin, uin) = if reflect {
                (sub(k, scale(z, c(2.0) * k[2])), self::reflect(u, z))
            } else {
                (k, u)
            };
            let e = [[c(1.0), c(0.0), c(0.0)], [c(0.0), c(1.0), c(0.0)]];
            let uout = carry(uin, kin, kout);
            let ds = mm(inv(proj(uin, e))?, x);
            let power = D::new(sf.paraxial_power, sf.paraxial_power_lo);
            let dt = ma(ms(mm(tj(kin, uin)?, t), n1 / n2), ms(ds, -power));
            t = mm(inv(tj(kout, uout)?)?, dt);
            x = mm(proj(uout, e), ds);
            u = uout;
        } else {
            let p = vec(j, 0);
            let v = super::sag::evaluate(sf, pool, p[0], p[1])?;
            if v[3..].iter().any(|v| !v.hi().is_finite()) {
                return None;
            }
            let e0 = unit([c(1.0), c(0.0), v[1]]);
            let e = [e0, cross(nrm, e0)];
            let km = ms(form([v[3], v[4], v[5]], e), nrm[2]);
            let uout = if reflect {
                self::reflect(u, nrm)
            } else {
                carry(u, k, kout)
            };
            let pin = proj(u, e);
            let pout = proj(uout, e);
            let ds = mm(inv(pin)?, x);
            let g = n2 * tr::ddot(kout, nrm) - n1 * tr::ddot(k, nrm);
            let mut power = ms(km, -g);
            if sf.shape == MKL_SHAPE_DIFFRACTIVE {
                let ph = super::sag::phase(sf, pool, p[0], p[1]);
                let gn = ph[0] * nrm[0] + ph[1] * nrm[1];
                let data = crate::surface_math::data(sf, pool);
                let q = D::from_df(data[MKL_COEFF_PREFIX as usize + wl]);
                power = ma(power, ms(ma(form([ph[2], ph[3], ph[4]], e), ms(km, gn)), q));
            }
            let rhs = ma(ms(mm(mt(pin), t), n1), mm(power, ds));
            t = ms(mm(mt(inv(pout)?), rhs), c(1.0) / n2);
            x = mm(pout, ds);
            u = uout;
        }
    }
    None
}

pub fn close<E: FpEnv>(f: &[f32], status: &[u32], n: usize, field: usize) -> Option<[Df32; 3]> {
    let start = field * 5;
    if status[start..start + 5].iter().any(|&v| v != 0) {
        return None;
    }
    let read = |i: usize, k: usize| {
        std::array::from_fn(|j| D::<E>::new(f[(k + j) * n + i], f[(k + j + 9) * n + i]))
    };
    let ip = read(start, 0);
    let cd = read(start, 3);
    let et = unit(sub([c(0.0), c(1.0), c(0.0)], scale(cd, cd[1])));
    let u = [et, cross(cd, et)];
    let project = |i| {
        let p = read(i, 0);
        let d = read(i, 3);
        let dc = tr::ddot(d, cd);
        if dc.hi() <= 0.0 {
            return None;
        }
        let q = add(p, scale(d, tr::ddot(sub(ip, p), cd) / dc));
        Some((q, sub(scale(d, c(1.0) / dc), cd)))
    };
    let (ya, yb, xa, xb) = (
        project(start + 1)?,
        project(start + 2)?,
        project(start + 3)?,
        project(start + 4)?,
    );
    let xy = sub(ya.0, yb.0);
    let xx = sub(xa.0, xb.0);
    let ty = sub(ya.1, yb.1);
    let tx = sub(xa.1, xb.1);
    let x = [
        [tr::ddot(xy, u[0]), tr::ddot(xx, u[0])],
        [tr::ddot(xy, u[1]), tr::ddot(xx, u[1])],
    ];
    let t = [
        [tr::ddot(ty, u[0]), tr::ddot(tx, u[0])],
        [tr::ddot(ty, u[1]), tr::ddot(tx, u[1])],
    ];
    foci(x, t, u, cd).map(|v| v.map(|v| v.0))
}
