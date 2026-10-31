//! Double-float Newton–Broyden specification. The Metal twin is aiming.metal.
use super::*;
use crate::{
    FpEnv,
    fx::{D, F, f},
    layout::{PackedHeader, PackedSurface, TraceParams},
    trace_kernel::{self as tr, KRayIn, ResidentState},
};
type V2<E> = [D<E>; 2];
type V3<E> = [D<E>; 3];
type Mat<E> = [[D<E>; 2]; 2];
fn c<E: FpEnv>(x: f32) -> D<E> {
    D::from_f(f(x))
}
fn norm2<E: FpEnv>(x: V2<E>) -> D<E> {
    (x[0] * x[0] + x[1] * x[1]).sqrt()
}
fn unit<E: FpEnv>(v: V3<E>) -> V3<E> {
    let n = tr::dnorm(v);
    v.map(|x| x / n)
}
fn add<E: FpEnv>(a: V2<E>, b: V2<E>) -> V2<E> {
    [a[0] + b[0], a[1] + b[1]]
}
fn sub<E: FpEnv>(a: V2<E>, b: V2<E>) -> V2<E> {
    [a[0] - b[0], a[1] - b[1]]
}
fn scale<E: FpEnv>(a: V2<E>, b: D<E>) -> V2<E> {
    a.map(|x| x * b)
}
fn finite<E: FpEnv>(a: V2<E>) -> bool {
    a.iter().all(|v| v.hi().is_finite())
}
fn solve2<E: FpEnv>(j: Mat<E>, r: V2<E>) -> Option<V2<E>> {
    let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
    let mut s = F::<E>::ZERO;
    for row in j {
        for v in row {
            s = s.max(v.hi().abs());
        }
    }
    if !det.hi().is_finite() || det.hi().abs() <= 1e-14 * s * s {
        return None;
    }
    let x = [
        (r[0] * j[1][1] - j[0][1] * r[1]) / det,
        (j[0][0] * r[1] - r[0] * j[1][0]) / det,
    ];
    finite(x).then_some(x)
}
fn jacobian<E: FpEnv>(
    fun: &impl Fn(V2<E>) -> Option<V2<E>>,
    x: V2<E>,
    fx: V2<E>,
    h: D<E>,
) -> Option<Mat<E>> {
    let mut j = [[D::ZERO; 2]; 2];
    for k in 0..2 {
        let mut xp = x;
        let mut xm = x;
        xp[k] = xp[k] + h;
        xm[k] = xm[k] - h;
        let col = match (fun(xp), fun(xm)) {
            (Some(a), Some(b)) => scale(sub(a, b), c::<E>(1.0) / (c::<E>(2.0) * h)),
            (Some(a), None) => scale(sub(a, fx), c::<E>(1.0) / h),
            (None, Some(b)) => scale(sub(fx, b), c::<E>(1.0) / h),
            _ => return None,
        };
        j[0][k] = col[0];
        j[1][k] = col[1];
    }
    Some(j)
}
#[derive(Clone, Copy)]
struct Solve<E: FpEnv> {
    x: V2<E>,
    j: Option<Mat<E>>,
    residual: D<E>,
    iterations: u32,
    converged: bool,
}
fn newton<E: FpEnv>(
    fun: impl Fn(V2<E>) -> Option<V2<E>>,
    x0: V2<E>,
    j0: Option<Mat<E>>,
    h: D<E>,
    tol: D<E>,
    cap: u32,
) -> Solve<E> {
    let mut out = Solve {
        x: x0,
        j: j0,
        residual: c(f32::INFINITY),
        iterations: 0,
        converged: false,
    };
    let Some(mut r) = fun(x0) else {
        return out;
    };
    let mut x = x0;
    let mut rn = norm2(r);
    let (mut j, mut fresh) = match j0 {
        Some(j) => (j, false),
        None => match jacobian(&fun, x, r, h) {
            Some(j) => (j, true),
            None => {
                out.residual = rn;
                return out;
            }
        },
    };
    let mut it = 0;
    while it < cap {
        if (rn - tol).hi() <= 0.0 {
            break;
        }
        it += 1;
        let Some(dx) = solve2(j, scale(r, c(-1.0))) else {
            if fresh {
                break;
            }
            match jacobian(&fun, x, r, h) {
                Some(jn) => {
                    j = jn;
                    fresh = true;
                    continue;
                }
                None => break,
            }
        };
        let mut lam = c(1.0);
        let mut accepted = None;
        for _ in 0..16 {
            let xn = add(x, scale(dx, lam));
            if let Some(rr) = fun(xn) {
                let nn = norm2(rr);
                if nn.hi().is_finite() && ((nn - rn).hi() < 0.0 || (nn - tol).hi() <= 0.0) {
                    accepted = Some((xn, rr, nn));
                    break;
                }
            }
            lam = lam * c(0.5);
        }
        match accepted {
            Some((xn, rr, nn)) => {
                let s = sub(xn, x);
                let ss = s[0] * s[0] + s[1] * s[1];
                if ss.hi() > 0.0 {
                    let y = sub(rr, r);
                    let js = [
                        j[0][0] * s[0] + j[0][1] * s[1],
                        j[1][0] * s[0] + j[1][1] * s[1],
                    ];
                    for row in 0..2 {
                        for col in 0..2 {
                            j[row][col] = j[row][col] + (y[row] - js[row]) * s[col] / ss;
                        }
                    }
                }
                let slow = (nn - c(0.5) * rn).hi() > 0.0;
                x = xn;
                r = rr;
                rn = nn;
                fresh = false;
                if slow && (rn - tol).hi() > 0.0 {
                    if let Some(jn) = jacobian(&fun, x, r, h) {
                        j = jn;
                        fresh = true;
                    }
                }
            }
            None => {
                if fresh {
                    break;
                }
                match jacobian(&fun, x, r, h) {
                    Some(jn) => {
                        j = jn;
                        fresh = true;
                    }
                    None => break,
                }
            }
        }
    }
    Solve {
        x,
        j: Some(j),
        residual: rn,
        iterations: it,
        converged: (rn - tol).hi() <= 0.0,
    }
}

/// Immutable optical data for one field/wavelength. Evaluations trace local
/// double-float states directly, without rounded legacy record correction.
pub struct FieldKernel<'a, E: FpEnv> {
    pub h: &'a PackedHeader,
    pub s: &'a [PackedSurface],
    pub pool: &'a [Df32],
    pub field: &'a FieldSpec,
    pub tolerance: f32,
    pub cap: u32,
    pub check: &'a dyn Fn() -> bool,
    pub env: std::marker::PhantomData<E>,
}
impl<E: FpEnv> FieldKernel<'_, E> {
    fn finite(&self) -> bool {
        self.field.flags & MKL_AIM_FINITE != 0
    }
    fn real(&self) -> bool {
        self.field.flags & MKL_AIM_REAL != 0
    }
    fn tele(&self) -> bool {
        self.field.flags & MKL_AIM_TELECENTRIC != 0
    }
    fn pupil(&self, k: usize) -> D<E> {
        D::from_df(self.field.pupil[k])
    }
    fn step(&self) -> D<E> {
        c(1e-6) * c(self.pupil(1).hi().v().max(1e-6))
    }
    fn tol(&self) -> D<E> {
        c(self.tolerance) * self.pupil(2)
    }
    fn params(&self, last: u32) -> TraceParams {
        TraceParams {
            num_steps: last,
            max_iterations: tr::DEFAULT_MAX_ITERATIONS,
            tolerance_abs: tr::DEFAULT_TOL_ABS,
            tolerance_rel: tr::DEFAULT_TOL_REL,
            ..Default::default()
        }
    }
    fn local(&self, p: V3<E>, point: bool) -> V3<E> {
        let a = std::array::from_fn(|k| {
            if point {
                p[k] - D::from_df(self.field.origin[k])
            } else {
                p[k]
            }
        });
        std::array::from_fn(|k| {
            let r = std::array::from_fn(|j| D::from_df(self.field.rotation[j * 3 + k]));
            tr::ddot(r, a)
        })
    }
    fn object(&self, p: V2<E>) -> Option<V3<E>> {
        // The reference's radial object-height convention evaluates extended
        // object sags on +y, then retains the requested local x/y.
        let at = if self.s[0].shape > MKL_SHAPE_COORD_BREAK {
            [D::ZERO, norm2(p), D::ZERO]
        } else {
            [p[0], p[1], D::ZERO]
        };
        let hit = tr::intersect(
            &self.s[0],
            self.pool,
            at,
            [D::ZERO, D::ZERO, c(1.0)],
            &self.params(0),
        )
        .ok()?;
        Some([p[0], p[1], hit.x[2]])
    }
    fn source(&self, p: V2<E>) -> Option<V3<E>> {
        if self.finite() {
            self.object(p)
        } else {
            Some(unit([p[0], p[1], c(1.0)]))
        }
    }
    fn initial_source(&self) -> Option<V3<E>> {
        let p = self.field.source.map(D::from_df);
        if self.finite() && self.field.flags & MKL_AIM_ANGLE != 0 {
            if self.tele() {
                return None;
            }
            let dir = self.local(unit([p[0], p[1], c(1.0)]), false);
            let point = self.local([D::ZERO, D::ZERO, self.pupil(0)], true);
            Some(
                tr::intersect(&self.s[0], self.pool, point, dir, &self.params(0))
                    .ok()?
                    .x,
            )
        } else {
            self.source(p)
        }
    }
    fn start(&self, src: V3<E>, a: V2<E>) -> Option<KRayIn> {
        let (pos, dir) = if !self.finite() {
            let ap = [a[0], a[1], self.pupil(0)];
            let t = tr::ddot(ap, src);
            (std::array::from_fn(|k| ap[k] - src[k] * t), src)
        } else if self.tele() {
            (src, self.local(unit([a[0], a[1], c(1.0)]), false))
        } else {
            let ap = self.local([a[0], a[1], self.pupil(0)], true);
            let mut d = unit(std::array::from_fn(|k| ap[k] - src[k]));
            let z = tr::ddot(
                std::array::from_fn(|k| D::from_df(self.field.rotation[6 + k])),
                d,
            );
            if z.hi() == 0.0 || !z.hi().is_finite() || d.iter().any(|v| !v.hi().is_finite()) {
                return None;
            }
            if z.hi() < 0.0 {
                d = d.map(|x| -x);
            }
            (src, d)
        };
        Some(KRayIn {
            pos: pos.map(|x| x.hi().v()),
            pos_lo: pos.map(|x| x.lo().v()),
            dir: dir.map(|x| x.hi().v()),
            dir_lo: dir.map(|x| x.lo().v()),
            weight: 1.0,
            wavelength: self.field.wavelength,
            ..Default::default()
        })
    }
    fn trace(&self, src: V3<E>, a: V2<E>, last: u32, wl: u32) -> Option<ResidentState> {
        if !(self.check)() {
            return None;
        }
        let mut ray = self.start(src, a)?;
        ray.wavelength = wl;
        let steps: Vec<_> = (1..=last).collect();
        let r = tr::trace_resident::<E>(
            self.h,
            self.s,
            self.pool,
            &steps,
            &self.params(last),
            &ray,
            |_, _| {},
        );
        (r.status == 0).then_some(r)
    }
    fn solve(&self, src: V3<E>, wl: u32, target: V2<E>, x: V2<E>, j: Option<Mat<E>>) -> Solve<E> {
        newton(
            |a| {
                let r = self.trace(src, a, self.h.stop, wl)?;
                Some(std::array::from_fn(|k| {
                    D::new(r.ray.pos[k], r.ray.pos_lo[k]) - target[k]
                }))
            },
            x,
            j,
            self.step(),
            self.tol(),
            self.cap,
        )
    }
    fn chief(&self, src: V3<E>, wl: u32) -> Solve<E> {
        if !self.real() {
            return Solve {
                x: [D::ZERO; 2],
                j: None,
                residual: D::ZERO,
                iterations: 0,
                converged: true,
            };
        }
        let mut s = self.solve(src, wl, [D::ZERO; 2], [D::ZERO; 2], None);
        if !s.converged {
            let n = if self.finite() { 16 } else { 64 };
            let mut x = [D::ZERO; 2];
            let mut j = None;
            let mut entrance = None;
            for k in 0..=n {
                let frac = c(k as f32) / c(n as f32);
                let step = if k == n {
                    Some(src)
                } else if self.finite() {
                    self.source([src[0] * frac, src[1] * frac])
                } else {
                    Some(unit([
                        src[0] * frac,
                        src[1] * frac,
                        c(1.0) - frac + src[2] * frac,
                    ]))
                };
                let Some(step) = step else {
                    break;
                };
                if let Some(e) = entrance {
                    let e: V3<E> = e;
                    let t = (self.pupil(0) - e[2]) / step[2];
                    x = [e[0] + t * step[0], e[1] + t * step[1]];
                }
                let sn = self.solve(step, wl, [D::ZERO; 2], x, j);
                if !sn.converged {
                    break;
                }
                x = sn.x;
                j = sn.j;
                if k == n {
                    s = sn;
                } else if !self.finite() {
                    if let Some(r) = self.trace(step, x, 1, wl) {
                        let mut p = std::array::from_fn(|k| D::new(r.ray.pos[k], r.ray.pos_lo[k]));
                        let mut d = [D::ZERO; 3];
                        tr::xf_backward(&self.s[0], &mut p, &mut d);
                        entrance = Some(p);
                    }
                }
            }
        }
        let fun = |a| {
            let r = self.trace(src, a, self.h.stop, wl)?;
            Some(std::array::from_fn(|k| {
                D::new(r.ray.pos[k], r.ray.pos_lo[k])
            }))
        };
        s.j = fun(s.x)
            .and_then(|v| jacobian(&fun, s.x, v, self.step()))
            .or(s.j);
        s
    }
    pub fn prepare(&self) -> ChiefSeed {
        if self.field.flags & MKL_AIM_IMAGE_HEIGHT != 0
            && self.h.primary_wavelength >= self.h.num_wavelengths
        {
            return ChiefSeed {
                status: MKL_AIM_PRIMARY_UNAVAILABLE,
                ..bytemuck::Zeroable::zeroed()
            };
        }
        let fail = ChiefSeed {
            status: MKL_AIM_FIELD_UNREACHABLE,
            ..bytemuck::Zeroable::zeroed()
        };
        let Some(mut src) = self.initial_source() else {
            return fail;
        };
        let target = self.field.target.map(D::from_df);
        if self.field.flags & MKL_AIM_IMAGE_HEIGHT != 0
            && (target[0].hi() != 0.0 || target[1].hi() != 0.0)
        {
            let p = if self.finite() {
                [src[0], src[1]]
            } else {
                [src[0] / src[2], src[1] / src[2]]
            };
            let h = c(1e-6) * c(norm2(p).hi().v().max(1e-6));
            let tol = c(1e-12) * c(norm2(target).hi().v().max(1e-3));
            let result = newton(
                |p| {
                    let src = self.source(p)?;
                    let aim = self.chief(src, self.h.primary_wavelength);
                    if !aim.converged {
                        return None;
                    }
                    let r = self.trace(
                        src,
                        aim.x,
                        self.h.num_surfaces - 1,
                        self.h.primary_wavelength,
                    )?;
                    Some(std::array::from_fn(|k| {
                        D::new(r.ray.pos[k], r.ray.pos_lo[k]) - target[k]
                    }))
                },
                p,
                None,
                h,
                tol,
                60,
            );
            if !result.converged {
                return fail;
            }
            let Some(s) = self.source(result.x) else {
                return fail;
            };
            src = s;
        }
        let s = self.chief(src, self.field.wavelength);
        ChiefSeed {
            source: src.map(|x| x.0),
            chief: s.x.map(|x| x.0),
            jacobian: std::array::from_fn(|k| s.j.unwrap_or([[D::ZERO; 2]; 2])[k / 2][k % 2].0),
            residual: s.residual.0,
            status: if s.converged {
                0
            } else {
                MKL_AIM_FAILED | (self.h.stop << MKL_RAY_SURFACE_SHIFT)
            },
            iterations: s.iterations,
            has_jacobian: u32::from(s.j.is_some()),
            reserved: 0,
        }
    }
    pub fn generate(&self, seed: &ChiefSeed, sample: &Sample) -> (KRayIn, [u32; 4]) {
        let src = seed.source.map(D::from_df);
        let chief = seed.chief.map(D::from_df);
        let p = std::array::from_fn(|k| {
            D::from_df(self.field.vignette[k])
                + (c(1.0) - D::from_df(self.field.vignette[k + 2])) * D::from_df(sample.pupil[k])
        });
        let j = (seed.has_jacobian != 0).then(|| {
            std::array::from_fn(|row| {
                std::array::from_fn(|col| D::from_df(seed.jacobian[row * 2 + col]))
            })
        });
        let mut s = Solve {
            x: scale(p, self.pupil(1)),
            j: None,
            residual: D::ZERO,
            iterations: 0,
            converged: true,
        };
        if matches!(
            seed.status,
            MKL_AIM_FIELD_UNREACHABLE | MKL_AIM_PRIMARY_UNAVAILABLE
        ) {
            return (
                KRayIn {
                    wavelength: self.field.wavelength,
                    ..Default::default()
                },
                [seed.status, 0, 0, 0],
            );
        }
        if self.real() {
            let target = scale(p, self.pupil(2));
            let x = add(
                chief,
                j.and_then(|j| solve2(j, target))
                    .unwrap_or(scale(p, self.pupil(1))),
            );
            s = self.solve(src, self.field.wavelength, target, x, j);
            let mut it = s.iterations;
            if !s.converged {
                let mut x = chief;
                let mut j = j;
                let mut prev = [D::ZERO; 2];
                for frac in [0.25, 0.5, 0.75, 1.0] {
                    let t = scale(target, c(frac));
                    let seed = j
                        .and_then(|j| solve2(j, sub(t, prev)))
                        .map_or(x, |d| add(x, d));
                    s = self.solve(src, self.field.wavelength, t, seed, j);
                    it += s.iterations;
                    if !s.converged {
                        break;
                    }
                    x = s.x;
                    j = s.j;
                    prev = t;
                }
            }
            s.iterations = it;
        }
        let Some(mut ray) = self.start(src, s.x) else {
            return (
                KRayIn {
                    wavelength: self.field.wavelength,
                    ..Default::default()
                },
                [
                    MKL_AIM_FAILED | (self.h.stop << MKL_RAY_SURFACE_SHIFT),
                    s.iterations,
                    s.residual.hi().v().to_bits(),
                    s.residual.lo().v().to_bits(),
                ],
            );
        };
        ray.weight = sample.weight;
        let residual = if self.real() {
            s.residual / self.pupil(2)
        } else {
            D::ZERO
        };
        (
            ray,
            [
                if s.converged {
                    0
                } else {
                    MKL_AIM_FAILED | (self.h.stop << MKL_RAY_SURFACE_SHIFT)
                },
                s.iterations,
                residual.hi().v().to_bits(),
                residual.lo().v().to_bits(),
            ],
        )
    }
}

/// Chief-relative reference sphere, including displaced-centre references.
/// Returns OPD (waves) and the three unit-radius sphere coordinates in the
/// image frame (z mirrored for negative propagation). P6 selects its basis.
pub fn project<E: FpEnv>(
    ray: &KRayIn,
    chief: &[Df32],
    centre: &[Df32],
    reference: &PupilReference,
) -> Option<[Df32; 4]> {
    let center: V3<E> = std::array::from_fn(|k| D::from_df(centre[k]));
    let pos: V3<E> = std::array::from_fn(|k| D::new(ray.pos[k], ray.pos_lo[k]));
    let dir: V3<E> = std::array::from_fn(|k| D::new(ray.dir[k], ray.dir_lo[k]));
    let cp: V3<E> = std::array::from_fn(|k| D::from_df(chief[k]));
    let cd: V3<E> = std::array::from_fn(|k| D::from_df(chief[3 + k]));
    let r = D::from_df(reference.radius);
    let sign = c(reference.sphere_sheet as f32);
    let correction = |p: V3<E>, d: V3<E>| {
        let w: V3<E> = std::array::from_fn(|k| p[k] - center[k]);
        let b = tr::ddot(d, w);
        if reference.telecentric {
            return Some((b, d.map(|x| -x)));
        }
        let cross = [
            w[1] * d[2] - w[2] * d[1],
            w[2] * d[0] - w[0] * d[2],
            w[0] * d[1] - w[1] * d[0],
        ];
        let p2 = tr::ddot(cross, cross);
        let delta = r * r - p2;
        if delta.hi() < 0.0 {
            return None;
        }
        let root = delta.sqrt();
        let corr = b + sign * p2 / (r + root);
        let t = -b + sign * root;
        Some((corr, std::array::from_fn(|k| (w[k] + d[k] * t) / r)))
    };
    let (corr, mut pupil) = correction(pos, dir)?;
    let (cc, _) = correction(cp, cd)?;
    let opd = (D::from_df(chief[6]) - D::from_df(ray.opl))
        + D::from_df(reference.image_index) * (corr - cc);
    if cd[2].hi() < 0.0 {
        pupil[2] = -pupil[2];
    }
    Some(
        [
            opd / (D::from_df(reference.wavelength_um) * D::from_df(Df32::from_f64(1e-3))),
            pupil[0],
            pupil[1],
            pupil[2],
        ]
        .map(|x| x.0),
    )
}

/// Procedural lattice/fan mapping, with the same double-float algebra in MSL.
pub fn sample_at<E: FpEnv>(base: Sample, sampling: Sampling, i: usize, count: usize) -> Sample {
    let mut out = base;
    let (kind, side, extent) = sampling.parameters();
    if kind == 0 {
        return out;
    }
    let coordinate = |k: usize, n: usize| {
        if n <= 1 {
            D::ZERO
        } else {
            (c::<E>(-1.0) + c::<E>(2.0) * c::<E>(k as f32) / c::<E>((n - 1) as f32))
                * D::from_df(extent)
        }
    };
    out.pupil = if kind == 1 {
        [
            coordinate(i % side as usize, side as usize).0,
            coordinate(i / side as usize, side as usize).0,
        ]
    } else {
        let mut p = [Df32::ZERO; 2];
        p[side as usize] = coordinate(i, count).0;
        p
    };
    let id = (base.id[0] as u64 | ((base.id[1] as u64) << 32)).wrapping_add(i as u64);
    out.id = [id as u32, (id >> 32) as u32];
    out
}
