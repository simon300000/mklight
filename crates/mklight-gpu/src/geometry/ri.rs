//! Image-point illumination pencils, in double-float: the specification of
//! `mkl_geometry_ri_rays` (one thread per aimed ray).
//!
//! Field-level data (chief aim and stop-Jacobian inverse, the image basis
//! and tolerance) comes from the host's f64 field preparation ([`RiField`]),
//! so no thread repeats per-field work. Each thread runs a state machine
//! whose every step is one trace from a single call site: the threads of a
//! SIMD group trace together whatever stage of their solve they are in, and
//! the kernel holds one copy of the trace.
//!
//! An aimed ray `(q, P)` solves the stop aim by chord Newton from the chief
//! (or a given seed) with finite-difference refreshes and backtracking,
//! falls back to continuation in the pupil radius, then traces to the image.
//! The host solves each node's image point `X(q, P) = X₀` for `q` and
//! writes its pencil ([`PENCIL`] rays): the clipped ray and the central
//! differences of `(X, L, M)` in `q` and `P` (module docs of
//! `mklight_analysis::illumination`). The pencil rays are seeded with the
//! aim of the node's converged ray, so their stop solves start within one
//! Newton step.
use super::sag;
use crate::{
    Df32, FpEnv,
    aiming::FieldSpec,
    consts::*,
    fx::D,
    layout::{PackedHeader, PackedSurface, TraceParams},
    trace_kernel::{self as tr, KRayIn, ResidentState},
};
use bytemuck::{Pod, Zeroable};

/// Mirror of `MklRiField` (header docs).
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct RiField {
    pub aim: FieldSpec,
    pub chief: [Df32; 2],
    pub aim_inverse: [Df32; 4],
    pub basis: [Df32; 6],
    pub tolerance: Df32,
    pub status: u32,
    /// Stall acceptance of the stop aims, a fraction of the stop radius.
    pub aim_floor: f32,
}
/// Mirror of `MklRiTask`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RiTask {
    pub field: u32,
    pub flags: u32,
    pub q: [Df32; 2],
    pub pupil: [Df32; 2],
    pub seed: [Df32; 2],
}
/// Mirror of `MklRiRay`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Pod, Zeroable)]
pub struct RiRay {
    pub status: u32,
    pub iterations: u32,
    pub x: [Df32; 2],
    pub lm: [Df32; 2],
    pub aim: [Df32; 2],
}
const _: () = assert!(size_of::<RiField>() == MKL_RI_FIELD_WORDS as usize * 4);
const _: () = assert!(size_of::<RiTask>() == MKL_RI_TASK_WORDS as usize * 4);
const _: () = assert!(size_of::<RiRay>() == MKL_RI_RAY_WORDS as usize * 4);

type P<E> = [D<E>; 2];
type M<E> = [[D<E>; 2]; 2];
fn c<E: FpEnv>(v: f32) -> D<E> {
    D::new(v, 0.0)
}
fn dfs<E: FpEnv, const N: usize>(a: [Df32; N]) -> [D<E>; N] {
    a.map(D::from_df)
}
fn m<E: FpEnv>(a: [Df32; 4]) -> M<E> {
    [
        [D::from_df(a[0]), D::from_df(a[1])],
        [D::from_df(a[2]), D::from_df(a[3])],
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
fn norm<E: FpEnv>(r: P<E>) -> D<E> {
    (r[0] * r[0] + r[1] * r[1]).sqrt()
}

/// Shared read-only inputs of one thread.
pub struct System<'a> {
    pub h: &'a PackedHeader,
    pub s: &'a [PackedSurface],
    pub pool: &'a [Df32],
    /// `1..=num_surfaces-1`, the sequential program.
    pub steps: &'a [u32],
}
/// The starting ray of object parameters `q` and aim point `a`
/// (`mklight_trace::SequentialTracer::start_ray`).
fn initial<E: FpEnv>(sys: &System, f: &FieldSpec, q: P<E>, a: P<E>) -> Option<KRayIn> {
    let finite = f.flags & MKL_AIM_FINITE != 0;
    let tele = f.flags & MKL_AIM_TELECENTRIC != 0;
    let z = D::<E>::from_df(f.pupil[0]);
    let unit = |v: [D<E>; 3]| {
        let inv = c::<E>(1.0) / tr::ddot(v, v).sqrt();
        v.map(|x| x * inv)
    };
    let local = |p: [D<E>; 3], point: bool| -> [D<E>; 3] {
        let a: [D<E>; 3] = std::array::from_fn(|k| {
            if point {
                p[k] - D::from_df(f.origin[k])
            } else {
                p[k]
            }
        });
        std::array::from_fn(|k| {
            tr::ddot(
                std::array::from_fn(|j| D::from_df(f.rotation[3 * j + k])),
                a,
            )
        })
    };
    let (pos, dir) = if !finite {
        let d = unit([q[0], q[1], c(1.0)]);
        let ap = [a[0], a[1], z];
        let t = tr::ddot(ap, d);
        (std::array::from_fn(|k| ap[k] - d[k] * t), d)
    } else {
        let p = [
            q[0],
            q[1],
            sag::evaluate(&sys.s[0], sys.pool, q[0], q[1])?[0],
        ];
        if tele {
            (p, local(unit([a[0], a[1], c(1.0)]), false))
        } else {
            let ap = local([a[0], a[1], z], true);
            let mut d = unit(std::array::from_fn(|k| ap[k] - p[k]));
            let rz = tr::ddot(std::array::from_fn(|k| D::from_df(f.rotation[6 + k])), d);
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
        wavelength: f.wavelength,
        weight: 1.0,
        ..Default::default()
    })
}
/// One trace of the machines: to the stop (`last = stop`) or the image.
/// `Err` carries the failure status (`MKL_AIM_INVALID` without a starting
/// ray).
fn trace<E: FpEnv>(
    sys: &System,
    f: &FieldSpec,
    q: P<E>,
    a: P<E>,
    last: u32,
    clip: bool,
) -> Result<ResidentState, u32> {
    let ray = initial(sys, f, q, a).ok_or(MKL_AIM_INVALID)?;
    let params = TraceParams {
        num_steps: last,
        flags: if clip { MKL_TRACE_CLIP } else { 0 },
        ..crate::aiming::kernel::newton_params()
    };
    let out = tr::trace_resident::<E>(
        sys.h,
        sys.s,
        sys.pool,
        &sys.steps[..last as usize],
        &params,
        &ray,
        |_, _| {},
    );
    if out.status == 0 {
        Ok(out)
    } else {
        Err(out.status)
    }
}

/// Stop target of normalised pupil point `p` (vignetting applied).
fn target<E: FpEnv>(f: &FieldSpec, p: P<E>) -> P<E> {
    std::array::from_fn(|k| {
        (D::<E>::from_df(f.vignette[k]) + (c(1.0) - D::from_df(f.vignette[k + 2])) * p[k])
            * D::from_df(f.pupil[2])
    })
}

const PHASE_EVAL: u32 = 0;
const PHASE_FD: u32 = 1;
const PHASE_BACK: u32 = 2;
const PHASE_FINAL: u32 = 3;
const PHASE_DONE: u32 = 4;
/// Newton iterations of one stop solve, backtracking halvings per step.
const AIM_ITERATIONS: u32 = 60;
const AIM_HALVINGS: u32 = 12;
/// Residual (fraction of the stop radius) below which a stuck stop solve
/// meets the double-float trace's noise rather than a convergence basin.
const AIM_NOISE: f32 = 1e-6;
/// Trace budget of one aimed ray: the device fails what does not finish
/// within it (the reference completes those), so no thread's chain of
/// dependent traces runs long. Converging rays take 3–10 traces.
pub const RAY_TRACES: u32 = 48;
/// Budget of a flat-pass ray seeded with a converged neighbouring aim (a
/// pencil ray; 3–6 traces when it converges).
pub const RAY_TRACES_SEEDED: u32 = 16;
/// A chord step that does not shrink the residual below this fraction
/// refreshes the Jacobian: slow linear convergence costs more traces than a
/// four-ray refresh.
const REFRESH: f32 = 0.25;

/// Resumable aimed ray `(q, P)`: stop solve, then the image trace.
#[derive(Clone, Copy)]
pub struct AimMachine<E: FpEnv> {
    pub q: P<E>,
    pub clip: bool,
    target: P<E>,
    goal: P<E>,
    pub a: P<E>,
    r: P<E>,
    rn: D<E>,
    inverse: M<E>,
    prev: D<E>,
    da: P<E>,
    alpha: D<E>,
    fd_plus: P<E>,
    j: M<E>,
    it: u32,
    sub: u32,
    /// -1 direct solve; 0..=4 continuation stages.
    stage: i32,
    pub phase: u32,
    pub iterations: u32,
    /// Traces so far.
    pub traces: u32,
    /// Trace budget of the stop solve.
    pub budget: u32,
    /// Ok: image x/y and direction cosines on the basis.
    pub result: Option<[D<E>; 4]>,
    /// When done: 0, `MKL_AIM_FAILED`, or the image trace's failure status.
    pub status: u32,
}
/// What a machine traces next.
#[derive(Clone, Copy)]
pub struct Request<E: FpEnv> {
    pub q: P<E>,
    pub a: P<E>,
    pub last: u32,
    pub clip: bool,
}
impl<E: FpEnv> AimMachine<E> {
    /// A ray from `seed` (or the chief prediction); `aimed`: the seed is a
    /// converged aim of this very `(q, P)`, traced to the image directly.
    pub fn new(f: &RiField, q: P<E>, p: P<E>, clip: bool, seed: Option<P<E>>, aimed: bool) -> Self {
        let target = target(&f.aim, p);
        let a = match seed {
            Some(s) => s,
            None => {
                let da = apply(m(f.aim_inverse), target);
                std::array::from_fn(|k| D::from_df(f.chief[k]) + da[k])
            }
        };
        AimMachine {
            q,
            clip,
            target,
            goal: target,
            a,
            r: [D::ZERO; 2],
            rn: D::ZERO,
            inverse: m(f.aim_inverse),
            prev: c(f32::INFINITY),
            da: [D::ZERO; 2],
            alpha: c(1.0),
            fd_plus: [D::ZERO; 2],
            j: [[D::ZERO; 2]; 2],
            it: 0,
            sub: 0,
            stage: -1,
            phase: if aimed { PHASE_FINAL } else { PHASE_EVAL },
            iterations: 0,
            traces: 0,
            budget: RAY_TRACES,
            result: None,
            status: MKL_AIM_FAILED,
        }
    }
    fn step(f: &RiField) -> D<E> {
        c::<E>(1e-6) * D::from_df(f.aim.pupil[1])
    }
    pub fn request(&self, f: &RiField, h: &PackedHeader) -> Request<E> {
        let mut a = self.a;
        let mut last = h.stop;
        let mut clip = false;
        match self.phase {
            PHASE_FD => {
                let s = Self::step(f);
                let k = (self.sub / 2) as usize;
                a[k] = if self.sub.is_multiple_of(2) {
                    a[k] + s
                } else {
                    a[k] - s
                };
            }
            PHASE_BACK => {
                a = std::array::from_fn(|k| self.a[k] - self.da[k] * self.alpha);
            }
            PHASE_FINAL => {
                last = h.num_surfaces - 1;
                clip = self.clip;
            }
            _ => {}
        }
        Request {
            q: self.q,
            a,
            last,
            clip,
        }
    }
    /// Whether the residual is within the stall acceptance.
    fn floored(&self, f: &RiField) -> bool {
        (self.rn - c::<E>(f.aim_floor) * D::from_df(f.aim.pupil[2])).hi() <= 0.0
    }
    /// The stop solve from the current iterate, whose residual is set.
    fn check(&mut self, f: &RiField) {
        if self.it >= AIM_ITERATIONS {
            return self.solve_ended(f);
        }
        let tol = c::<E>(1e-12) * D::from_df(f.aim.pupil[2]);
        if (self.rn - tol).hi() <= 0.0 {
            return self.solve_succeeded(f);
        }
        if (self.rn - c::<E>(REFRESH) * self.prev).hi() > 0.0 {
            // Stalled: at the trace's noise floor, accept; else refresh.
            if self.floored(f) {
                return self.solve_succeeded(f);
            }
            self.phase = PHASE_FD;
            self.sub = 0;
            return;
        }
        self.newton_step();
    }
    /// A solve that cannot step further: accepted within the floor.
    /// A solve stuck below `AIM_NOISE` of the stop radius has met the
    /// trace's noise, which no continuation removes: the ray fails at once.
    fn solve_ended(&mut self, f: &RiField) {
        if self.floored(f) {
            self.solve_succeeded(f)
        } else if (self.rn - c::<E>(AIM_NOISE) * D::from_df(f.aim.pupil[2])).hi() <= 0.0 {
            self.phase = PHASE_DONE;
        } else {
            self.solve_failed(f)
        }
    }
    fn newton_step(&mut self) {
        self.prev = self.rn;
        self.da = apply(self.inverse, self.r);
        self.alpha = c(1.0);
        self.sub = 0;
        self.phase = PHASE_BACK;
    }
    fn restart(&mut self, f: &RiField) {
        self.inverse = m(f.aim_inverse);
        self.prev = c(f32::INFINITY);
        self.it = 0;
        self.phase = PHASE_EVAL;
    }
    fn solve_failed(&mut self, f: &RiField) {
        if self.stage < 0 {
            self.stage = 0;
            self.a = dfs(f.chief);
            self.goal = self.target.map(|v| v * c(0.0));
            self.restart(f);
        } else {
            self.phase = PHASE_DONE;
        }
    }
    fn solve_succeeded(&mut self, f: &RiField) {
        if self.stage < 0 || self.stage == 4 {
            self.phase = PHASE_FINAL;
        } else {
            self.stage += 1;
            let fraction = c::<E>(0.25 * self.stage as f32);
            self.goal = self.target.map(|v| v * fraction);
            self.restart(f);
        }
    }
    fn residual(&mut self, hit: &ResidentState) {
        self.r = std::array::from_fn(|k| D::new(hit.ray.pos[k], hit.ray.pos_lo[k]) - self.goal[k]);
        self.rn = norm(self.r);
    }
    /// Advance with the outcome of [`Self::request`]'s trace. A ray whose
    /// stop solve has not finished within its budget fails.
    pub fn feed(&mut self, f: &RiField, traced: Result<&ResidentState, u32>) {
        self.advance(f, traced);
        self.traces += 1;
        if self.phase != PHASE_DONE && self.phase != PHASE_FINAL && self.traces >= self.budget {
            self.phase = PHASE_DONE;
        }
    }
    fn advance(&mut self, f: &RiField, traced: Result<&ResidentState, u32>) {
        let hit = traced.ok();
        match self.phase {
            PHASE_EVAL => match hit {
                None => self.solve_failed(f),
                Some(hit) => {
                    self.residual(hit);
                    self.check(f);
                }
            },
            PHASE_FD => {
                let Some(hit) = hit else {
                    return self.solve_failed(f);
                };
                let pos = [
                    D::new(hit.ray.pos[0], hit.ray.pos_lo[0]),
                    D::new(hit.ray.pos[1], hit.ray.pos_lo[1]),
                ];
                if self.sub.is_multiple_of(2) {
                    self.fd_plus = pos;
                } else {
                    let k = (self.sub / 2) as usize;
                    let s = Self::step(f);
                    for (i, row) in self.j.iter_mut().enumerate() {
                        row[k] = (self.fd_plus[i] - pos[i]) / (c(2.0) * s);
                    }
                }
                self.sub += 1;
                if self.sub == 4 {
                    match inverse(self.j) {
                        Some(j) => {
                            self.inverse = j;
                            self.newton_step();
                        }
                        None => self.solve_failed(f),
                    }
                }
            }
            PHASE_BACK => {
                let accepted = hit.and_then(|hit| {
                    let e: P<E> = std::array::from_fn(|k| {
                        D::new(hit.ray.pos[k], hit.ray.pos_lo[k]) - self.goal[k]
                    });
                    let en = norm(e);
                    ((en - self.rn).hi() < 0.0).then_some((e, en))
                });
                match accepted {
                    Some((e, en)) => {
                        self.a = std::array::from_fn(|k| self.a[k] - self.da[k] * self.alpha);
                        self.r = e;
                        self.rn = en;
                        self.it += 1;
                        self.iterations += 1;
                        self.check(f);
                    }
                    None => {
                        self.alpha = self.alpha * c(0.5);
                        self.sub += 1;
                        if self.sub == AIM_HALVINGS {
                            self.solve_ended(f);
                        }
                    }
                }
            }
            PHASE_FINAL => {
                self.phase = PHASE_DONE;
                self.status = traced.err().unwrap_or(0);
                self.result = hit.map(|hit| {
                    let dir: [D<E>; 3] =
                        std::array::from_fn(|k| D::new(hit.ray.dir[k], hit.ray.dir_lo[k]));
                    let b0 = [f.basis[0], f.basis[1], f.basis[2]].map(D::from_df);
                    let b1 = [f.basis[3], f.basis[4], f.basis[5]].map(D::from_df);
                    [
                        D::new(hit.ray.pos[0], hit.ray.pos_lo[0]),
                        D::new(hit.ray.pos[1], hit.ray.pos_lo[1]),
                        tr::ddot(dir, b0),
                        tr::ddot(dir, b1),
                    ]
                });
            }
            _ => {}
        }
    }
}

/// Runs a machine's requests to completion, one trace each.
fn run<E: FpEnv>(sys: &System, f: &RiField, m: &mut AimMachine<E>, check: &dyn Fn() -> bool) {
    while m.phase != PHASE_DONE {
        if !check() {
            m.phase = PHASE_DONE;
            m.result = None;
            return;
        }
        let r = m.request(f, sys.h);
        let hit = trace(sys, &f.aim, r.q, r.a, r.last, r.clip);
        m.feed(f, hit.as_ref().map_err(|e| *e));
    }
}

/// `mkl_geometry_ri_rays`: one aimed ray.
pub fn ray<E: FpEnv>(sys: &System, f: &RiField, task: &RiTask, check: &dyn Fn() -> bool) -> RiRay {
    if task.flags & MKL_RI_SKIP != 0 {
        return RiRay {
            status: MKL_AIM_OUTSIDE,
            ..Default::default()
        };
    }
    if f.status != 0 {
        return RiRay {
            status: MKL_AIM_FIELD_UNREACHABLE,
            ..Default::default()
        };
    }
    let mut machine = AimMachine::<E>::new(
        f,
        dfs(task.q),
        dfs(task.pupil),
        task.flags & MKL_RI_CLIP != 0,
        (task.flags & MKL_RI_SEEDED != 0).then(|| dfs(task.seed)),
        task.flags & MKL_RI_AIMED != 0,
    );
    if task.flags & MKL_RI_SEEDED != 0 {
        machine.budget = RAY_TRACES_SEEDED;
    }
    run(sys, f, &mut machine, check);
    let mut out = RiRay {
        status: machine.status,
        iterations: machine.iterations,
        aim: machine.a.map(|v| v.0),
        ..Default::default()
    };
    if let Some(v) = machine.result {
        out.x = [v[0].0, v[1].0];
        out.lm = [v[2].0, v[3].0];
    }
    out
}

/// Rays of a node's pencil: the clipped ray, then the central differences
/// `q₀ ±, q₁ ±, P₀ ±, P₁ ±` (stencil index `s` = 2·column + minus).
pub const PENCIL: usize = 9;
/// Object and pupil step of the pupil columns.
pub fn pupil_step() -> Df32 {
    Df32::from_f64(1e-4)
}
/// The thread entry of `mkl_geometry_ri_rays` for the CPU emulation.
pub fn run_ray<E: FpEnv>(
    sys: &System,
    fields: &[RiField],
    task: &RiTask,
    check: &dyn Fn() -> bool,
) -> RiRay {
    match fields.get(task.field as usize) {
        Some(f) => ray::<E>(sys, f, task, check),
        None => RiRay {
            status: MKL_AIM_INVALID,
            ..Default::default()
        },
    }
}
