//! Nested image-point solve and implicit direction Jacobian, in double-float.
use super::RECORD_FLOATS;
use crate::{
    Df32, FpEnv,
    aiming::FieldSpec,
    consts::*,
    fx::D,
    layout::{PackedHeader, PackedSurface},
    trace_kernel::{self as tr, KRayIn, ResidentState},
};
type V<E> = [D<E>; 3];
type P<E> = [D<E>; 2];
type M<E> = [[D<E>; 2]; 2];
fn c<E: FpEnv>(v: f32) -> D<E> {
    D::new(v, 0.0)
}
fn dot<E: FpEnv>(a: V<E>, b: V<E>) -> D<E> {
    tr::ddot(a, b)
}
fn unit<E: FpEnv>(a: V<E>) -> V<E> {
    let inv = c(1.0) / dot(a, a).sqrt();
    a.map(|v| v * inv)
}
fn cross<E: FpEnv>(a: V<E>, b: V<E>) -> V<E> {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
fn inverse<E: FpEnv>(a: M<E>) -> Option<M<E>> {
    let d = a[0][0] * a[1][1] - a[0][1] * a[1][0];
    if !d.hi().is_finite() || d.hi() == 0.0 {
        return None;
    }
    Some([[a[1][1] / d, -a[0][1] / d], [-a[1][0] / d, a[0][0] / d]])
}
fn apply<E: FpEnv>(a: M<E>, v: P<E>) -> P<E> {
    std::array::from_fn(|k| a[k][0] * v[0] + a[k][1] * v[1])
}
fn mul<E: FpEnv>(a: M<E>, b: M<E>) -> M<E> {
    std::array::from_fn(|i| std::array::from_fn(|j| a[i][0] * b[0][j] + a[i][1] * b[1][j]))
}
struct Probe<'a, E: FpEnv> {
    h: &'a PackedHeader,
    s: &'a [PackedSurface],
    pool: &'a [Df32],
    f: &'a FieldSpec,
    seed: P<E>,
    aim: M<E>,
    q0: P<E>,
    x0: P<E>,
    basis: [V<E>; 2],
    hq: D<E>,
    check: &'a dyn Fn() -> bool,
}
impl<E: FpEnv> Probe<'_, E> {
    fn local(&self, p: V<E>, point: bool) -> V<E> {
        let a = std::array::from_fn(|k| {
            if point {
                p[k] - D::from_df(self.f.origin[k])
            } else {
                p[k]
            }
        });
        std::array::from_fn(|k| {
            dot(
                std::array::from_fn(|j| D::from_df(self.f.rotation[3 * j + k])),
                a,
            )
        })
    }
    fn initial(&self, q: P<E>, a: P<E>) -> Option<KRayIn> {
        let finite = self.f.flags & MKL_AIM_FINITE != 0;
        let tele = self.f.flags & MKL_AIM_TELECENTRIC != 0;
        let z = D::<E>::from_df(self.f.pupil[0]);
        let (pos, dir) = if !finite {
            let d = unit([q[0], q[1], c(1.0)]);
            let ap = [a[0], a[1], z];
            let t = dot(ap, d);
            (std::array::from_fn(|k| ap[k] - d[k] * t), d)
        } else {
            let p = [
                q[0],
                q[1],
                super::sag::evaluate(&self.s[0], self.pool, q[0], q[1])?[0],
            ];
            if tele {
                (p, self.local(unit([a[0], a[1], c(1.0)]), false))
            } else {
                let ap = self.local([a[0], a[1], z], true);
                let mut d = unit(std::array::from_fn(|k| ap[k] - p[k]));
                let rz = dot(
                    std::array::from_fn(|k| D::from_df(self.f.rotation[6 + k])),
                    d,
                );
                if rz.hi() == 0.0 {
                    return None;
                }
                if rz.hi() < 0.0 {
                    d = d.map(|v| -v);
                }
                (p, d)
            }
        };
        Some(KRayIn {
            pos: pos.map(|v| v.hi().v()),
            pos_lo: pos.map(|v| v.lo().v()),
            dir: dir.map(|v| v.hi().v()),
            dir_lo: dir.map(|v| v.lo().v()),
            wavelength: self.f.wavelength,
            weight: 1.0,
            ..Default::default()
        })
    }
    fn trace(&self, q: P<E>, a: P<E>, last: u32, clip: bool) -> Option<ResidentState> {
        if !(self.check)() {
            return None;
        }
        let ray = self.initial(q, a)?;
        let mut params = crate::aiming::kernel::newton_params();
        params.num_steps = last;
        params.flags = if clip { MKL_TRACE_CLIP } else { 0 };
        let steps: Vec<_> = (1..=last).collect();
        let out =
            tr::trace_resident::<E>(self.h, self.s, self.pool, &steps, &params, &ray, |_, _| {});
        (out.status == 0).then_some(out)
    }
    fn aim_inverse(&self, q: P<E>, a: P<E>) -> Option<M<E>> {
        let step = c(1e-6) * D::from_df(self.f.pupil[1]);
        let mut j = [[D::ZERO; 2]; 2];
        for k in 0..2 {
            let mut aa = a;
            let mut bb = a;
            aa[k] = aa[k] + step;
            bb[k] = bb[k] - step;
            let ha = self.trace(q, aa, self.h.stop, false)?;
            let hb = self.trace(q, bb, self.h.stop, false)?;
            for (i, row) in j.iter_mut().enumerate() {
                row[k] = (D::new(ha.ray.pos[i], ha.ray.pos_lo[i])
                    - D::new(hb.ray.pos[i], hb.ray.pos_lo[i]))
                    / (c(2.0) * step);
            }
        }
        inverse(j)
    }
    fn aim_solve(&self, q: P<E>, target: P<E>, mut a: P<E>) -> Option<P<E>> {
        let mut inverse = self.aim;
        let mut previous = c(f32::INFINITY);
        let tol = c(1e-12) * D::from_df(self.f.pupil[2]);
        let mut hit = self.trace(q, a, self.h.stop, false)?;
        for _ in 0..60 {
            let r =
                std::array::from_fn(|k| D::<E>::new(hit.ray.pos[k], hit.ray.pos_lo[k]) - target[k]);
            let rn = (r[0] * r[0] + r[1] * r[1]).sqrt();
            if (rn - tol).hi() <= 0.0 {
                return Some(a);
            }
            if (rn - c(0.5) * previous).hi() > 0.0 {
                inverse = self.aim_inverse(q, a)?;
            }
            previous = rn;
            let da = apply(inverse, r);
            let mut alpha = c(1.0);
            let mut accepted = false;
            for _ in 0..12 {
                let next = std::array::from_fn(|k| a[k] - da[k] * alpha);
                if let Some(candidate) = self.trace(q, next, self.h.stop, false) {
                    let e: [D<E>; 2] = std::array::from_fn(|k| {
                        D::new(candidate.ray.pos[k], candidate.ray.pos_lo[k]) - target[k]
                    });
                    if ((e[0] * e[0] + e[1] * e[1]).sqrt() - rn).hi() < 0.0 {
                        a = next;
                        hit = candidate;
                        accepted = true;
                        break;
                    }
                }
                alpha = alpha * c(0.5);
            }
            if !accepted {
                return None;
            }
        }
        None
    }
    fn ray(&self, q: P<E>, p: P<E>, clip: bool) -> Option<[D<E>; 4]> {
        let target: P<E> = std::array::from_fn(|k| {
            (D::<E>::from_df(self.f.vignette[k])
                + (c(1.0) - D::from_df(self.f.vignette[k + 2])) * p[k])
                * D::from_df(self.f.pupil[2])
        });
        let da = apply(self.aim, target);
        let predicted = std::array::from_fn(|k| self.seed[k] + da[k]);
        let a = match self.aim_solve(q, target, predicted) {
            Some(a) => a,
            None => {
                let mut a = self.seed;
                for fraction in [0.0, 0.25, 0.5, 0.75, 1.0] {
                    a = self.aim_solve(q, target.map(|v| v * c(fraction)), a)?;
                }
                a
            }
        };
        let hit = self.trace(q, a, self.h.num_surfaces - 1, clip)?;
        let dir = std::array::from_fn(|k| D::new(hit.ray.dir[k], hit.ray.dir_lo[k]));
        Some([
            D::new(hit.ray.pos[0], hit.ray.pos_lo[0]),
            D::new(hit.ray.pos[1], hit.ray.pos_lo[1]),
            dot(dir, self.basis[0]),
            dot(dir, self.basis[1]),
        ])
    }
    fn jac(&self, q: P<E>, p: P<E>, columns: usize) -> Option<[[D<E>; 4]; 4]> {
        let mut j = [[D::ZERO; 4]; 4];
        for k in 0..columns {
            let h = if k < 2 {
                self.hq
            } else {
                D::from_df(Df32::from_f64(1e-4))
            };
            let (mut qa, mut qb, mut pa, mut pb) = (q, q, p, p);
            if k < 2 {
                qa[k] = qa[k] + h;
                qb[k] = qb[k] - h;
            } else {
                pa[k - 2] = pa[k - 2] + h;
                pb[k - 2] = pb[k - 2] - h;
            }
            let (a, b) = (self.ray(qa, pa, false)?, self.ray(qb, pb, false)?);
            for i in 0..4 {
                j[i][k] = (a[i] - b[i]) / (c(2.0) * h);
            }
        }
        Some(j)
    }
    fn node(&self, p: P<E>) -> Option<(D<E>, bool)> {
        let j = self.jac(self.q0, [D::ZERO; 2], 2)?;
        let mut ji = inverse([[j[0][0], j[0][1]], [j[1][0], j[1][1]]])?;
        let mut q = self.q0;
        let mut prev = c(f32::INFINITY);
        let tol = c(1e-10) * (c(1.0) + (self.x0[0] * self.x0[0] + self.x0[1] * self.x0[1]).sqrt());
        let mut converged = false;
        for _ in 0..40 {
            let x = self.ray(q, p, false)?;
            let r = [x[0] - self.x0[0], x[1] - self.x0[1]];
            let rn = (r[0] * r[0] + r[1] * r[1]).sqrt();
            if (rn - tol).hi() <= 0.0 {
                converged = true;
                break;
            }
            if (rn - c(0.5) * prev).hi() > 0.0 {
                let j = self.jac(q, p, 2)?;
                ji = inverse([[j[0][0], j[0][1]], [j[1][0], j[1][1]]])?;
            }
            prev = rn;
            let dq = apply(ji, r);
            q = [q[0] - dq[0], q[1] - dq[1]];
        }
        if !converged {
            return None;
        }
        if self.ray(q, p, true).is_none() {
            return Some((D::ZERO, false));
        }
        let j = self.jac(q, p, 4)?;
        let xq = [[j[0][0], j[0][1]], [j[1][0], j[1][1]]];
        let xp = [[j[0][2], j[0][3]], [j[1][2], j[1][3]]];
        let lq = [[j[2][0], j[2][1]], [j[3][0], j[3][1]]];
        let lp = [[j[2][2], j[2][3]], [j[3][2], j[3][3]]];
        let corr = mul(lq, mul(inverse(xq)?, xp));
        let d: M<E> = std::array::from_fn(|i| std::array::from_fn(|k| lp[i][k] - corr[i][k]));
        let det = d[0][0] * d[1][1] - d[0][1] * d[1][0];
        Some((if det.hi() < 0.0 { -det } else { det }, true))
    }
}
#[allow(clippy::too_many_arguments)]
pub fn node<E: FpEnv>(
    h: &PackedHeader,
    s: &[PackedSurface],
    pool: &[Df32],
    f: &FieldSpec,
    hi: &[f32],
    lo: &[f32],
    nr: usize,
    field: usize,
    pupil: [Df32; 2],
    check: &dyn Fn() -> bool,
) -> Option<[Df32; 4]> {
    let read = |i: usize, slot: usize, k: usize| {
        std::array::from_fn(|j| {
            D::<E>::new(
                hi[(slot * nr + i) * RECORD_FLOATS + k + j],
                lo[(slot * nr + i) * RECORD_FLOATS + k + j],
            )
        })
    };
    let pos = read(field, 0, 0);
    let dir = read(field, 0, 3);
    let finite = f.flags & MKL_AIM_FINITE != 0;
    let q0 = if finite {
        [pos[0], pos[1]]
    } else {
        [dir[0] / dir[2], dir[1] / dir[2]]
    };
    let global = |p: V<E>, point: bool| {
        std::array::from_fn(|k| {
            dot(
                std::array::from_fn(|j| D::<E>::from_df(f.rotation[3 * k + j])),
                p,
            ) + if point {
                D::from_df(f.origin[k])
            } else {
                D::ZERO
            }
        })
    };
    let dp = if finite { global(dir, false) } else { dir };
    let pp = if finite { global(pos, true) } else { pos };
    let seed = if f.flags & MKL_AIM_TELECENTRIC != 0 {
        [dp[0] / dp[2], dp[1] / dp[2]]
    } else {
        let t = (D::from_df(f.pupil[0]) - pp[2]) / dp[2];
        [pp[0] + t * dp[0], pp[1] + t * dp[1]]
    };
    let last = h.num_surfaces as usize - 1;
    let normal = unit(read(field, last, 8));
    let mut t1 = std::array::from_fn(|k| {
        if k == 0 {
            c(1.0) - normal[k] * normal[0]
        } else {
            -normal[k] * normal[0]
        }
    });
    if dot(t1, t1).hi() < 1e-12 {
        t1 = std::array::from_fn(|k| {
            if k == 1 {
                c(1.0) - normal[k] * normal[1]
            } else {
                -normal[k] * normal[1]
            }
        });
    }
    t1 = unit(t1);
    let basis = [t1, cross(normal, t1)];
    let cp = read(field, last, 0);
    let scale_pos = read(nr - 1, 0, 0);
    let scale_dir = read(nr - 1, 0, 3);
    let qs = if finite {
        [scale_pos[0], scale_pos[1]]
    } else {
        [scale_dir[0] / scale_dir[2], scale_dir[1] / scale_dir[2]]
    };
    let norm = (qs[0] * qs[0] + qs[1] * qs[1]).sqrt();
    let scale = if norm.hi() > 1e-3 { norm } else { c(1e-3) };
    let mut probe = Probe {
        h,
        s,
        pool,
        f,
        seed,
        aim: [[c(1.0), c(0.0)], [c(0.0), c(1.0)]],
        q0,
        x0: [cp[0], cp[1]],
        basis,
        hq: D::from_df(Df32::from_f64(1e-5)) * scale,
        check,
    };
    let cz = dp[2] * dp[2];
    let cos4 = (cz * cz).0;
    let Some(aim) = probe.aim_inverse(probe.q0, probe.seed) else {
        return Some([Df32::ZERO, Df32::ZERO, Df32::from_f32(1.0), cos4]);
    };
    probe.aim = aim;
    match probe.node(pupil.map(D::from_df)) {
        Some((j, pass)) => Some([
            j.0,
            if pass {
                Df32::from_f32(1.0)
            } else {
                Df32::ZERO
            },
            Df32::ZERO,
            cos4,
        ]),
        None => Some([Df32::ZERO, Df32::ZERO, Df32::from_f32(1.0), cos4]),
    }
}
