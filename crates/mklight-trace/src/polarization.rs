//! Polarisation ray tracing (PRT) along sequential rays.
//!
//! Conventions are CONVENTIONS §9 (time dependence `exp(i(ωt − k·r))`,
//! `N = n − ik`, s/p basis with `r_p = −r_s` at normal incidence, Stokes
//! signs, double-pole pupil basis). Design notes: `docs/POLARIZATION.md`.
//!
//! # PRT matrices
//!
//! Every interacting surface `i` of a traced ray gets a 3×3 complex matrix
//! `P_i` in **global** coordinates that maps the incident field vector to
//! the outgoing one and the incident direction to the outgoing direction:
//!
//! `P_i = a_s · s'sᵀ + a_p · p'pᵀ + k'kᵀ`,
//!
//! with `s ⟂ k` (the s vector, see [`s_vector`]), `p = k × s`,
//! `s' = s` re-orthogonalised against `k'`, `p' = k' × s'`, and
//! `(a_s, a_p)` the flux-normalised transmission `(t̃_s, t̃_p)` of a
//! refracting surface or the reflection `(r_s, r_p)` of a mirror, from
//! [`mklight_core::materials::coating::ResolvedCoating::amplitudes`]. With
//! flux-normalised amplitudes `|P E|²` is the power the ray carries on, so
//! the polarised throughput is read directly off the field.
//!
//! Only rotations are applied to directions (surface frames are rotated by
//! their exact global rotation matrices), so the global frame costs no
//! precision. The cumulative matrix is `P = P_N ⋯ P_1`.
//!
//! The **geometric transformation** `Q` is the same product for an ideal,
//! non-polarising system: `a_s = a_p = 1` at refracting surfaces and a
//! perfect conductor (`r_s = −1`, `r_p = +1`) at mirrors. `Q` is real
//! orthogonal, and `Qᵀ P` restricted to the input transverse plane is the
//! *proper* Jones matrix: polarisation change with the geometric rotation
//! of the s/p planes and the mirror parity removed. Its retardance and
//! diattenuation are independent of any pupil basis.
//!
//! # Normal incidence
//!
//! `s = k × n̂` is undefined at normal incidence. Below `|k × n̂| = 1e-12`
//! [`s_vector`] uses a deterministic fallback (`k × e`, `e` the coordinate
//! axis least aligned with `k`). Results do not depend on the choice: with
//! the CONVENTIONS §9 basis every coating has `a_p = a_s` (transmission) or
//! `r_p = −r_s` (reflection, where `p' = −p`) at normal incidence, so
//! `P_i = a(I − kkᵀ) + k'kᵀ` there for *any* `s ⟂ k`. Near normal incidence
//! the anisotropic part `a_s − a_p` (or `r_s + r_p`) is `O(θ²)`, so the
//! error of an arbitrary `s` is `O(θ²)` and an `s` with angular error `ε/θ`
//! (rounding in the cross product) changes `P_i` by `O(θ ε)`: `P_i` is
//! continuous through normal incidence (tested).
//!
//! # Special surfaces
//!
//! Dummy surfaces, coordinate breaks and the image surface do not interact
//! (`P_i = I`). An ideal (`Paraxial`) lens is lossless and non-polarising:
//! `P_i` is the minimal rotation taking `k` to `k'`; a reflecting ideal lens
//! applies its coating's mirror amplitudes first (at the incidence angle on
//! its plane), then that rotation.

// Small fixed-size matrix code reads best with explicit indices.
#![allow(clippy::needless_range_loop)]

use mklight_core::materials::coating::{Complex64, InterfaceAmplitudes, InterfaceRT};
use mklight_core::math::Mat3;
use mklight_core::resolved::ResolvedSurface;
use mklight_core::{
    Interaction, JobContext, MkError, MkResult, ResolvedSystem, SurfaceShape, Vec3,
};
use serde::{Deserialize, Serialize};

use crate::batch::run_blocks;
use crate::interact::reflect;
use crate::trace::{Ray, SurfaceRecord, TraceOptions, TraceResult, TraceStatus, start_frame};
use crate::tracer::{PreparedField, SequentialTracer};

/// Complex scalar of the polarisation code.
pub type C = Complex64;

const ZERO: C = C::new(0.0, 0.0);
const ONE: C = C::new(1.0, 0.0);

/// Below this `|k × n̂|` the plane of incidence is treated as undefined.
const NORMAL_INCIDENCE_SIN: f64 = 1e-12;

// ---------------------------------------------------------------------------
// Small complex linear algebra.

/// A complex 3-vector (a field vector in some frame).
pub type CVec3 = [C; 3];

fn cv(v: Vec3) -> CVec3 {
    [C::new(v.x, 0.0), C::new(v.y, 0.0), C::new(v.z, 0.0)]
}

/// `Σ a_i b_i` of a real and a complex vector (no conjugation).
fn rdot(a: Vec3, b: &CVec3) -> C {
    b[0] * a.x + b[1] * a.y + b[2] * a.z
}

fn cnorm2(v: &CVec3) -> f64 {
    v[0].norm_sqr() + v[1].norm_sqr() + v[2].norm_sqr()
}

/// A 3×3 complex polarisation ray-tracing matrix (row-major).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PrtMatrix {
    pub m: [[C; 3]; 3],
}

impl PrtMatrix {
    pub const IDENTITY: PrtMatrix = PrtMatrix {
        m: [[ONE, ZERO, ZERO], [ZERO, ONE, ZERO], [ZERO, ZERO, ONE]],
    };
    pub const ZERO: PrtMatrix = PrtMatrix { m: [[ZERO; 3]; 3] };

    pub fn from_real(r: &Mat3) -> Self {
        let mut m = [[ZERO; 3]; 3];
        for (i, row) in m.iter_mut().enumerate() {
            for (j, c) in row.iter_mut().enumerate() {
                *c = C::new(r.m[i][j], 0.0);
            }
        }
        PrtMatrix { m }
    }

    /// `self += c · u vᵀ`.
    fn add_dyad(&mut self, c: C, u: Vec3, v: Vec3) {
        let (u, v) = ([u.x, u.y, u.z], [v.x, v.y, v.z]);
        for i in 0..3 {
            for j in 0..3 {
                self.m[i][j] += c * (u[i] * v[j]);
            }
        }
    }

    pub fn mul(&self, o: &PrtMatrix) -> PrtMatrix {
        let mut r = PrtMatrix::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                r.m[i][j] =
                    self.m[i][0] * o.m[0][j] + self.m[i][1] * o.m[1][j] + self.m[i][2] * o.m[2][j];
            }
        }
        r
    }

    pub fn mul_vec(&self, v: &CVec3) -> CVec3 {
        let mut r = [ZERO; 3];
        for (i, ri) in r.iter_mut().enumerate() {
            *ri = self.m[i][0] * v[0] + self.m[i][1] * v[1] + self.m[i][2] * v[2];
        }
        r
    }

    /// `self · v` for a real vector.
    pub fn apply(&self, v: Vec3) -> CVec3 {
        self.mul_vec(&cv(v))
    }

    /// Real part (the matrix of a real PRT such as `Q`).
    pub fn re(&self) -> Mat3 {
        let mut m = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = self.m[i][j].re;
            }
        }
        Mat3 { m }
    }

    /// Largest elementwise modulus of `self − o`; NaN if any element
    /// difference is NaN (unlike a plain `f64::max` fold, which drops NaN).
    pub fn max_abs_diff(&self, o: &PrtMatrix) -> f64 {
        let mut d = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                let x = (self.m[i][j] - o.m[i][j]).norm();
                if x.is_nan() {
                    return f64::NAN;
                }
                d = d.max(x);
            }
        }
        d
    }

    /// The 2×2 Jones matrix `J_ij = out_i · (self · in_j)` in real
    /// orthonormal transverse bases `in` (of the input direction) and `out`
    /// (of the output direction).
    pub fn jones(&self, out: (Vec3, Vec3), inp: (Vec3, Vec3)) -> JonesMatrix {
        let (cx, cy) = (self.apply(inp.0), self.apply(inp.1));
        JonesMatrix {
            m: [
                [rdot(out.0, &cx), rdot(out.0, &cy)],
                [rdot(out.1, &cx), rdot(out.1, &cy)],
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Jones and Stokes algebra.

type CPair = [[f64; 2]; 2];
type CQuad = [[[f64; 2]; 2]; 2];

/// A Jones vector `(E_x, E_y)` in a transverse basis (CONVENTIONS §9).
/// Serialised as `[[re, im], [re, im]]`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "CPair", into = "CPair")]
pub struct JonesVector {
    pub x: C,
    pub y: C,
}

impl From<CPair> for JonesVector {
    fn from(a: CPair) -> Self {
        JonesVector {
            x: C::new(a[0][0], a[0][1]),
            y: C::new(a[1][0], a[1][1]),
        }
    }
}
impl From<JonesVector> for CPair {
    fn from(v: JonesVector) -> Self {
        [[v.x.re, v.x.im], [v.y.re, v.y.im]]
    }
}

impl JonesVector {
    pub fn new(x: C, y: C) -> Self {
        JonesVector { x, y }
    }
    /// Unit linear polarisation at angle `a` (radians) from `x̂` toward `ŷ`.
    pub fn linear(a: f64) -> Self {
        let (s, c) = a.sin_cos();
        JonesVector::new(C::new(c, 0.0), C::new(s, 0.0))
    }
    pub fn intensity(&self) -> f64 {
        self.x.norm_sqr() + self.y.norm_sqr()
    }
    pub fn stokes(&self) -> Stokes {
        let (ax, ay) = (self.x.norm_sqr(), self.y.norm_sqr());
        let c = self.x.conj() * self.y;
        Stokes {
            s0: ax + ay,
            s1: ax - ay,
            s2: 2.0 * c.re,
            s3: 2.0 * c.im,
        }
    }
}

/// Stokes parameters (CONVENTIONS §9).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Stokes {
    pub s0: f64,
    pub s1: f64,
    pub s2: f64,
    pub s3: f64,
}

impl Stokes {
    pub const UNPOLARIZED: Stokes = Stokes {
        s0: 1.0,
        s1: 0.0,
        s2: 0.0,
        s3: 0.0,
    };
    /// Degree of polarisation `√(S₁² + S₂² + S₃²)/S₀`.
    pub fn degree_of_polarization(&self) -> f64 {
        if self.s0 > 0.0 {
            (self.s1.hypot(self.s2)).hypot(self.s3) / self.s0
        } else {
            0.0
        }
    }
    /// Coherency matrix `Φ = ½(S₀ I + S₁σ_z + S₂σ_x + S₃σ_y)`, so that
    /// `Φ = E E†` for a pure state `E`.
    pub fn coherency(&self) -> [[C; 2]; 2] {
        [
            [
                C::new(0.5 * (self.s0 + self.s1), 0.0),
                C::new(0.5 * self.s2, -0.5 * self.s3),
            ],
            [
                C::new(0.5 * self.s2, 0.5 * self.s3),
                C::new(0.5 * (self.s0 - self.s1), 0.0),
            ],
        ]
    }
}

/// Input polarisation of a radiometric calculation, in the object-space
/// double-pole basis of each ray ([`double_pole_basis`]).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PolarizationState {
    /// Natural light: the average of any two orthogonal pure states.
    Unpolarized,
    /// A pure state.
    Jones { e: JonesVector },
    /// Any (partially polarised) state; `S₀` is the input power.
    Stokes { s: Stokes },
}

impl PolarizationState {
    pub fn stokes(&self) -> Stokes {
        match self {
            PolarizationState::Unpolarized => Stokes::UNPOLARIZED,
            PolarizationState::Jones { e } => e.stokes(),
            PolarizationState::Stokes { s } => *s,
        }
    }

    /// Check that the state is physical: finite, `S₀ > 0` and degree of
    /// polarisation `≤ 1` (to `1e-12` relative, for rounding in
    /// user-supplied Stokes vectors). A Jones vector must be finite and
    /// non-zero.
    pub fn validate(&self) -> MkResult<()> {
        let s = self.stokes();
        let finite = [s.s0, s.s1, s.s2, s.s3].iter().all(|x| x.is_finite());
        if !finite || s.s0.is_nan() || s.s0 <= 0.0 {
            return Err(MkError::InvalidModel(format!(
                "polarisation state must be finite with S0 > 0: {self:?}"
            )));
        }
        let pol = (s.s1.hypot(s.s2)).hypot(s.s3);
        if pol > s.s0 * (1.0 + 1e-12) {
            return Err(MkError::InvalidModel(format!(
                "polarisation state has degree of polarisation {} > 1: {self:?}",
                pol / s.s0
            )));
        }
        Ok(())
    }
}

/// A 2×2 Jones matrix. Serialised as `[[[re, im], …], …]` (row-major).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(from = "CQuad", into = "CQuad")]
pub struct JonesMatrix {
    pub m: [[C; 2]; 2],
}

impl From<CQuad> for JonesMatrix {
    fn from(a: CQuad) -> Self {
        let c = |p: [f64; 2]| C::new(p[0], p[1]);
        JonesMatrix {
            m: [[c(a[0][0]), c(a[0][1])], [c(a[1][0]), c(a[1][1])]],
        }
    }
}
impl From<JonesMatrix> for CQuad {
    fn from(j: JonesMatrix) -> Self {
        let p = |c: C| [c.re, c.im];
        [[p(j.m[0][0]), p(j.m[0][1])], [p(j.m[1][0]), p(j.m[1][1])]]
    }
}

impl JonesMatrix {
    pub const IDENTITY: JonesMatrix = JonesMatrix {
        m: [[ONE, ZERO], [ZERO, ONE]],
    };
    pub fn apply(&self, e: &JonesVector) -> JonesVector {
        JonesVector::new(
            self.m[0][0] * e.x + self.m[0][1] * e.y,
            self.m[1][0] * e.x + self.m[1][1] * e.y,
        )
    }
    pub fn mul(&self, o: &JonesMatrix) -> JonesMatrix {
        let a = &self.m;
        let b = &o.m;
        JonesMatrix {
            m: [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                ],
            ],
        }
    }
    pub fn scale(&self, s: f64) -> JonesMatrix {
        let mut j = *self;
        for row in &mut j.m {
            for c in row {
                *c *= s;
            }
        }
        j
    }
    pub fn det(&self) -> C {
        self.m[0][0] * self.m[1][1] - self.m[0][1] * self.m[1][0]
    }
    /// `J†J = (t/2) I + m₁σ_z + m₂σ_x + m₃σ_y`: returns `(t, [m₁, m₂, m₃])`.
    /// `t/2` is the unpolarised transmittance and `m` the diattenuation
    /// vector in Stokes space.
    fn gram(&self) -> (f64, [f64; 3]) {
        let j = &self.m;
        let a = j[0][0].norm_sqr() + j[1][0].norm_sqr();
        let d = j[0][1].norm_sqr() + j[1][1].norm_sqr();
        let b = j[0][0].conj() * j[0][1] + j[1][0].conj() * j[1][1];
        (a + d, [0.5 * (a - d), b.re, -b.im])
    }
    /// Output power for an input state (Stokes `S`):
    /// `I = ½ S₀ tr(J†J) + m·(S₁, S₂, S₃)`.
    pub fn intensity(&self, s: &Stokes) -> f64 {
        let (t, m) = self.gram();
        (0.5 * s.s0).mul_add(t, m[0] * s.s1 + m[1] * s.s2 + m[2] * s.s3)
    }
    pub fn intensity_of(&self, state: &PolarizationState) -> f64 {
        self.intensity(&state.stokes())
    }
    /// Transmittance for unpolarised light, `½ tr(J†J)`.
    pub fn t_unpolarized(&self) -> f64 {
        0.5 * self.gram().0
    }
    /// Diattenuation and retardance (module docs, CONVENTIONS §9).
    pub fn properties(&self) -> PolarizationProperties {
        PolarizationProperties::from_jones(self)
    }
}

/// Diattenuation and retardance of a Jones matrix.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct PolarizationProperties {
    /// `½ tr(J†J)`.
    pub t_unpolarized: f64,
    /// Largest and smallest power transmittance over input states
    /// (squared singular values).
    pub t_max: f64,
    pub t_min: f64,
    /// `(T_max − T_min)/(T_max + T_min)`.
    pub diattenuation: f64,
    /// Unit Stokes vector `(S₁, S₂, S₃)` of the best-transmitted state
    /// (zero when `D = 0`).
    pub diattenuation_axis: [f64; 3],
    /// Retardance `δ ∈ [0, π]` of the unitary polar factor (radians).
    ///
    /// The retarder of a nearly singular `J` is ill-conditioned: a relative
    /// perturbation `ε` of `J` (rounding, `ε ≈ 1e-16`, plus any upstream
    /// error) changes `δ` by up to about `ε·√(T_max/T_min)`. For
    /// `T_min < 1e-24·T_max` (amplification above `1e12`) the retarder is
    /// treated as undefined: `retarder_defined = false` and `δ = 0`.
    pub retardance: f64,
    /// Unit Stokes vector of the fast (leading-phase) eigenpolarisation of
    /// the retarder (zero when `δ = 0`).
    pub fast_axis: [f64; 3],
    /// `false` when `J` is (numerically) singular, e.g. a perfect polariser
    /// or a blocked ray: `retardance` and `fast_axis` are then
    /// placeholders (zero) and must not enter retardance statistics.
    #[serde(default = "default_true")]
    pub retarder_defined: bool,
}

fn default_true() -> bool {
    true
}

/// Below this `T_min/T_max` the retarder of a Jones matrix is undefined
/// ([`PolarizationProperties::retardance`]).
pub const RETARDER_MIN_CONTRAST: f64 = 1e-24;

impl PolarizationProperties {
    /// From the polar decomposition `J = U H`, `H = √(J†J)`.
    ///
    /// The singular values come from `J†J = (t/2)I + m·σ`:
    /// `T_max = t/2 + |m|`, `T_min = |det J|²/T_max` (no cancellation), and
    /// `D = 2|m|/t`. For 2×2 matrices `H = (J†J + |det J| I)/(σ₁ + σ₂)`, with
    /// `σ₁ + σ₂ = √(t + 2|det J|)` and `det H = |det J|`, so
    /// `U = J adj(H)/|det J|`. Writing `U = e^{iα}(cos(δ/2) I − i sin(δ/2) a·σ)`
    /// (`e^{2iα} = det U`, sign chosen so that `cos(δ/2) ≥ 0`) gives
    /// `δ = 2 atan2(sin(δ/2), cos(δ/2))`, accurate for small `δ`; the
    /// eigenstate with Stokes vector `+a` lags, so the fast axis is `−a`.
    pub fn from_jones(j: &JonesMatrix) -> Self {
        let (t, m) = j.gram();
        let mn = (m[0].hypot(m[1])).hypot(m[2]);
        let t_max = 0.5 * t + mn;
        let dabs = j.det().norm();
        let t_min = if t_max > 0.0 {
            dabs * dabs / t_max
        } else {
            0.0
        };
        let diattenuation = if t > 0.0 {
            (2.0 * mn / t).min(1.0)
        } else {
            0.0
        };
        let diattenuation_axis = if mn > 0.0 {
            [m[0] / mn, m[1] / mn, m[2] / mn]
        } else {
            [0.0; 3]
        };
        let retarder_defined = t_max > 0.0 && t_min > RETARDER_MIN_CONTRAST * t_max;
        let (retardance, fast_axis) = if retarder_defined {
            retarder(j, t, m, dabs)
        } else {
            (0.0, [0.0; 3])
        };
        PolarizationProperties {
            t_unpolarized: 0.5 * t,
            t_max,
            t_min,
            diattenuation,
            diattenuation_axis,
            retardance,
            fast_axis,
            retarder_defined,
        }
    }
}

fn retarder(j: &JonesMatrix, t: f64, m: [f64; 3], dabs: f64) -> (f64, [f64; 3]) {
    // J†J = [[t/2 + m₁, m₂ − i m₃], [m₂ + i m₃, t/2 − m₁]].
    let h_scale = 1.0 / (t + 2.0 * dabs).sqrt();
    let h00 = (0.5 * t + m[0] + dabs) * h_scale;
    let h11 = (0.5 * t - m[0] + dabs) * h_scale;
    let h01 = C::new(m[1], -m[2]) * h_scale;
    let h10 = h01.conj();
    // adj(H) = [[h11, −h01], [−h10, h00]]
    let adj = JonesMatrix {
        m: [[C::new(h11, 0.0), -h01], [-h10, C::new(h00, 0.0)]],
    };
    let u = j.mul(&adj).scale(1.0 / dabs);
    let phase = u.det().sqrt();
    let n = phase.norm();
    if n == 0.0 {
        return (0.0, [0.0; 3]);
    }
    let ph = phase / n;
    let up = u.m.map(|row| row.map(|c| c / ph));
    let mut c = 0.5 * (up[0][0] + up[1][1]).re;
    let mut v = [
        -0.5 * (up[0][0] - up[1][1]).im,
        -0.5 * (up[0][1] + up[1][0]).im,
        0.5 * (up[1][0] - up[0][1]).re,
    ];
    if c < 0.0 {
        c = -c;
        v = v.map(|x| -x);
    }
    let vn = (v[0].hypot(v[1])).hypot(v[2]);
    let delta = 2.0 * vn.atan2(c);
    if vn == 0.0 {
        return (delta, [0.0; 3]);
    }
    (delta, [-v[0] / vn, -v[1] / vn, -v[2] / vn])
}

// ---------------------------------------------------------------------------
// Bases.

/// The s vector of a ray with unit direction `k` at a surface with unit
/// normal `n` (either orientation): `k × n` made exactly orthogonal to `k`
/// and normalised. Below `|k × n| = 1e-12` (normal incidence) a
/// deterministic unit vector normal to `k` is returned (module docs).
pub fn s_vector(k: Vec3, n: Vec3) -> Vec3 {
    let c = k.cross(n);
    let c = c - k * c.dot(k);
    let len = c.norm();
    if len > NORMAL_INCIDENCE_SIN {
        return c / len;
    }
    perpendicular(k)
}

/// A deterministic unit vector normal to unit `k`: `k × e` for the
/// coordinate axis `e` least aligned with `k` (first on ties).
pub fn perpendicular(k: Vec3) -> Vec3 {
    let (ax, ay, az) = (k.x.abs(), k.y.abs(), k.z.abs());
    let e = if ax <= ay && ax <= az {
        Vec3::X
    } else if ay <= az {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let c = k.cross(e);
    let c = c - k * c.dot(k);
    c.normalized()
}

/// `v` made orthogonal to unit `k` and normalised.
fn orthonormalize(v: Vec3, k: Vec3) -> Vec3 {
    let w = v - k * v.dot(k);
    let len = w.norm();
    if len > 0.5 { w / len } else { perpendicular(k) }
}

/// Double-pole transverse basis `(x̂', ŷ')` of a ray with unit direction
/// `k` (global), relative to a frame with rotation `frame` (columns = the
/// frame's axes in global coordinates).
///
/// The pole is the frame's `+ẑ`, or `−ẑ` when `k` points into the frame's
/// negative-z half space; the reference axes are then `(x̂, −ŷ)` so that
/// `(x̂, ±ŷ, ±ẑ)` stays right-handed. `(x̂', ŷ')` are the reference axes
/// carried by the minimal rotation taking the pole to `k`. With
/// `(k₁, k₂, k₃)` the components of `k` on the reference triad,
/// `x̂' = (1 − k₁²/(1 + k₃), −k₁k₂/(1 + k₃), −k₁)` and
/// `ŷ' = (−k₁k₂/(1 + k₃), 1 − k₂²/(1 + k₃), −k₂)`; `k₃ ≥ 0`, so nothing
/// cancels. `x̂' × ŷ' = k`. The basis is smooth over the half space and
/// follows the fixed `x̂`, `ŷ`: turning `k` by `φ` about the pole gives
/// `B(R_z k) = R_z B(k) R(−φ)`, so the Jones pupil of a rotationally
/// symmetric system obeys `J(r, φ) = R(φ) J(r, 0) R(−φ)`.
pub fn double_pole_basis(k: Vec3, frame: &Mat3) -> (Vec3, Vec3) {
    let l = frame.transpose().mul_vec(k);
    let sg = if l.z < 0.0 { -1.0 } else { 1.0 };
    let (k1, k2, k3) = (l.x, sg * l.y, sg * l.z);
    let d = 1.0 + k3;
    let x = Vec3::new(1.0 - k1 * k1 / d, -k1 * k2 / d, -k1);
    let y = Vec3::new(-k1 * k2 / d, 1.0 - k2 * k2 / d, -k2);
    let to_global = |v: Vec3| frame.mul_vec(Vec3::new(v.x, sg * v.y, sg * v.z));
    (to_global(x), to_global(y))
}

// ---------------------------------------------------------------------------
// Interface matrices.

/// PRT matrix of one interface: incident direction `k_in`, outgoing
/// direction `k_out` (unit, global), s vector `s` (from [`s_vector`],
/// orthogonal to `k_in`) and the s/p coefficients `(a_s, a_p)` (module docs).
pub fn interface_prt(k_in: Vec3, k_out: Vec3, s: Vec3, a_s: C, a_p: C) -> PrtMatrix {
    let p_in = k_in.cross(s);
    let s_out = orthonormalize(s, k_out);
    let p_out = k_out.cross(s_out);
    let mut m = PrtMatrix::ZERO;
    m.add_dyad(a_s, s_out, s);
    m.add_dyad(a_p, p_out, p_in);
    m.add_dyad(ONE, k_out, k_in);
    m
}

/// The polarisation-free rotation taking `k_in` to `k_out` (minimal rotation
/// about `k_in × k_out`).
pub fn rotation_prt(k_in: Vec3, k_out: Vec3) -> PrtMatrix {
    interface_prt(k_in, k_out, s_vector(k_in, k_out), ONE, ONE)
}

/// What a surface does to the primary path.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKind {
    /// Coated or bare refracting interface (transmission).
    Refract,
    /// Mirror (reflection).
    Reflect,
    /// Ideal lens: lossless, non-polarising rotation.
    IdealLens,
    /// Reflecting ideal lens: mirror coefficients, then the lens rotation.
    IdealMirror,
}

/// One interacting surface along a polarisation-traced ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InterfaceStep {
    pub surface: usize,
    pub kind: StepKind,
    pub cos_incidence: f64,
    /// Global incident and outgoing unit directions.
    pub k_in: Vec3,
    pub k_out: Vec3,
    /// Global s vector (orthogonal to `k_in`).
    pub s: Vec3,
    /// Complex coefficients of the surface's coating (identity
    /// coefficients for an ideal lens).
    pub amplitudes: InterfaceAmplitudes,
    /// Power coefficients from `ResolvedCoating::rt`, the code every solver
    /// shares (equal to `|r|²`, `|t̃|²` of `amplitudes` to rounding); ideal
    /// lenses are `InterfaceRT::TRANSPARENT`.
    pub power: InterfaceRT,
    /// Internal transmittance of the medium segment before this surface.
    pub segment_transmittance: f64,
    pub prt: PrtMatrix,
    /// Geometric (non-polarising) counterpart of `prt`.
    pub geometric: PrtMatrix,
}

impl StepKind {
    /// The kind of an interacting surface; `None` for passive surfaces
    /// (dummies, coordinate breaks), which do not interact.
    pub fn of(s: &ResolvedSurface) -> Option<StepKind> {
        if s.is_passive() {
            return None;
        }
        let ideal = matches!(s.shape, SurfaceShape::Paraxial { .. });
        let mirror = s.interaction == Interaction::Reflect;
        Some(match (ideal, mirror) {
            (true, false) => StepKind::IdealLens,
            (true, true) => StepKind::IdealMirror,
            (false, true) => StepKind::Reflect,
            (false, false) => StepKind::Refract,
        })
    }

    /// Fraction of s / p power that continues along the sequential path
    /// for power coefficients `power`: `T` at a refracting surface, `R` at
    /// a mirror, 1 at an ideal lens.
    pub fn path_sp(self, power: &InterfaceRT) -> (f64, f64) {
        match self {
            StepKind::Refract => (power.ts, power.tp),
            StepKind::Reflect | StepKind::IdealMirror => (power.rs, power.rp),
            StepKind::IdealLens => (1.0, 1.0),
        }
    }
}

/// Power coefficients of interacting surface `i` (kind `kind`) for a ray
/// arriving with incidence cosine `cos_i`: `ResolvedCoating::rt`, the code
/// every solver shares, or `InterfaceRT::TRANSPARENT` for an ideal lens.
fn step_power(
    sys: &ResolvedSystem,
    wl: usize,
    i: usize,
    kind: StepKind,
    cos_i: f64,
) -> InterfaceRT {
    if kind == StepKind::IdealLens {
        return InterfaceRT::TRANSPARENT;
    }
    let s = &sys.surfaces[i];
    s.coating.rt(
        sys.wavelengths_um[wl],
        sys.n_before(i, wl),
        s.n_after[wl],
        cos_i,
        s.interaction == Interaction::Reflect,
    )
}

/// The scalar `Unpolarized` radiometry factor of a recorded ray: the
/// product over interacting surfaces (in path order) of `½(X_s + X_p)`.
/// Equal (bit for bit) to the product of
/// [`InterfaceStep::path_unpolarized`] over [`interface_steps`], without
/// building any amplitude or PRT data.
pub fn unpolarized_path_product(sys: &ResolvedSystem, wl: usize, records: &[SurfaceRecord]) -> f64 {
    let image = sys.image_index();
    (1..records.len().min(image))
        .filter_map(|i| StepKind::of(&sys.surfaces[i]).map(|k| (i, k)))
        .map(|(i, k)| {
            let (s, p) = k.path_sp(&step_power(sys, wl, i, k, records[i].cos_incidence));
            0.5 * (s + p)
        })
        .product()
}

impl InterfaceStep {
    /// Fraction of the power of s / p light that continues along the
    /// sequential path: `T` at a refracting surface, `R` at a mirror,
    /// 1 at an ideal lens.
    pub fn path_sp(&self) -> (f64, f64) {
        self.kind.path_sp(&self.power)
    }
    /// Fraction of s / p power that leaves the sequential path specularly
    /// (the reflected beam at a refracting surface; zero at mirrors, whose
    /// transmitted power counts as absorbed).
    pub fn diverted_sp(&self) -> (f64, f64) {
        match self.kind {
            StepKind::Refract => (self.power.rs, self.power.rp),
            _ => (0.0, 0.0),
        }
    }
    /// The incoherent (unpolarised) path factor `½(X_s + X_p)`.
    pub fn path_unpolarized(&self) -> f64 {
        let (s, p) = self.path_sp();
        0.5 * (s + p)
    }
}

fn global_rot(sys: &ResolvedSystem, i: usize) -> Mat3 {
    if i == 0 {
        start_frame(sys).rot
    } else {
        sys.surfaces[i].frame.rot
    }
}

/// The interacting surfaces of a recorded ray (`records[i]` = surface `i`,
/// as filled by [`crate::trace_ray`]), at system wavelength `wl`.
pub fn interface_steps(
    sys: &ResolvedSystem,
    wl: usize,
    records: &[SurfaceRecord],
) -> Vec<InterfaceStep> {
    let image = sys.image_index();
    let lambda = sys.wavelengths_um[wl];
    let mut out = Vec::new();
    for i in 1..records.len().min(image) {
        let s = &sys.surfaces[i];
        let Some(step_kind) = StepKind::of(s) else {
            continue;
        };
        let (prev, rec) = (&records[i - 1], &records[i]);
        let rot = global_rot(sys, i);
        let k_in = global_rot(sys, i - 1).mul_vec(prev.dir);
        let k_out = rot.mul_vec(rec.dir);
        let normal = rot.mul_vec(rec.normal);
        let n1 = sys.n_before(i, wl);
        let n2 = s.n_after[wl];
        let mirror = s.interaction == Interaction::Reflect;
        let ci = rec.cos_incidence;
        let segment_transmittance = if prev.transmittance > 0.0 {
            (rec.transmittance / prev.transmittance).min(1.0)
        } else {
            0.0
        };
        let ideal = matches!(step_kind, StepKind::IdealLens | StepKind::IdealMirror);
        let (kind, amplitudes, sv, prt, geometric) = if step_kind == StepKind::IdealLens {
            let sv = s_vector(k_in, k_out);
            let g = rotation_prt(k_in, k_out);
            (
                StepKind::IdealLens,
                InterfaceAmplitudes::TRANSPARENT,
                sv,
                g,
                g,
            )
        } else {
            let a = s.coating.amplitudes(lambda, n1, n2, ci, mirror);
            let sv = s_vector(k_in, normal);
            if ideal {
                // Reflect on the lens plane, then the ideal-lens rotation.
                let k_r = reflect(k_in, normal).0;
                let pc = InterfaceAmplitudes::PERFECT_CONDUCTOR;
                let rot = rotation_prt(k_r, k_out);
                let p = rot.mul(&interface_prt(k_in, k_r, sv, a.rs, a.rp));
                let g = rot.mul(&interface_prt(k_in, k_r, sv, pc.rs, pc.rp));
                (StepKind::IdealMirror, a, sv, p, g)
            } else if mirror {
                let pc = InterfaceAmplitudes::PERFECT_CONDUCTOR;
                let p = interface_prt(k_in, k_out, sv, a.rs, a.rp);
                let g = interface_prt(k_in, k_out, sv, pc.rs, pc.rp);
                (StepKind::Reflect, a, sv, p, g)
            } else {
                let p = interface_prt(k_in, k_out, sv, a.ts_flux(), a.tp_flux());
                let g = interface_prt(k_in, k_out, sv, ONE, ONE);
                (StepKind::Refract, a, sv, p, g)
            }
        };
        out.push(InterfaceStep {
            surface: i,
            kind,
            cos_incidence: ci,
            k_in,
            k_out,
            s: sv,
            amplitudes,
            power: step_power(sys, wl, i, kind, ci),
            segment_transmittance,
            prt,
            geometric,
        });
    }
    out
}

// ---------------------------------------------------------------------------
// Polarised rays.

/// A ray traced with polarisation: the scalar trace, its per-surface
/// records and the PRT data. Everything is meaningful up to the surface
/// where the trace stopped; Jones pupils and throughputs need
/// `trace.status.is_ok()`.
#[derive(Clone, Debug, PartialEq)]
pub struct PolarizedRay {
    pub trace: TraceResult,
    pub records: Vec<SurfaceRecord>,
    pub steps: Vec<InterfaceStep>,
    /// Cumulative PRT matrix `P_N ⋯ P_1` (global).
    pub prt: PrtMatrix,
    /// Cumulative geometric transformation `Q` (real orthogonal).
    pub geometric: Mat3,
    /// Global directions in object space and after the last surface.
    pub k_in: Vec3,
    pub k_out: Vec3,
    /// Object-space (starting-frame) and image-frame double-pole bases
    /// (global coordinates).
    pub basis_in: (Vec3, Vec3),
    pub basis_out: (Vec3, Vec3),
    /// Rotation of the frame the ray ended in (image frame on success).
    pub end_rot: Mat3,
    /// Internal (bulk) transmittance of the whole path.
    pub internal_transmittance: f64,
}

impl PolarizedRay {
    /// Build the PRT data of a recorded trace (records as filled by
    /// [`crate::trace_ray`]; empty records give an identity ray).
    pub fn from_trace(
        sys: &ResolvedSystem,
        wl: usize,
        trace: TraceResult,
        records: Vec<SurfaceRecord>,
    ) -> Self {
        let steps = interface_steps(sys, wl, &records);
        let mut prt = PrtMatrix::IDENTITY;
        let mut q = PrtMatrix::IDENTITY;
        for st in &steps {
            prt = st.prt.mul(&prt);
            q = st.geometric.mul(&q);
        }
        let r0 = global_rot(sys, 0);
        let last = records.len().saturating_sub(1);
        let end_rot = global_rot(sys, last);
        let (k_in, k_out) = match (records.first(), records.last()) {
            (Some(a), Some(b)) => (r0.mul_vec(a.dir), end_rot.mul_vec(b.dir)),
            _ => (Vec3::Z, Vec3::Z),
        };
        PolarizedRay {
            internal_transmittance: records
                .last()
                .map_or(trace.transmittance, |r| r.transmittance),
            trace,
            records,
            steps,
            prt,
            geometric: q.re(),
            k_in,
            k_out,
            basis_in: double_pole_basis(k_in, &r0),
            basis_out: double_pole_basis(k_out, &end_rot),
            end_rot,
        }
    }

    pub fn is_ok(&self) -> bool {
        self.trace.status.is_ok()
    }

    /// Jones matrix of the interfaces in the double-pole bases (object
    /// space → image space), flux-normalised: `|J e|²` is the transmitted
    /// power fraction before bulk absorption.
    pub fn jones(&self) -> JonesMatrix {
        self.prt.jones(self.basis_out, self.basis_in)
    }

    /// Jones matrix in caller-chosen real orthonormal transverse bases.
    pub fn jones_in(&self, out: (Vec3, Vec3), inp: (Vec3, Vec3)) -> JonesMatrix {
        self.prt.jones(out, inp)
    }

    /// Amplitude Jones matrix `√τ_internal · J` (the pupil apodisation and
    /// polarisation of the ray, without its optical path).
    pub fn jones_amplitude(&self) -> JonesMatrix {
        self.jones()
            .scale(self.internal_transmittance.max(0.0).sqrt())
    }

    /// Proper Jones matrix `[x y]ᵀ Qᵀ P [x y]` in the input double-pole
    /// basis: the polarisation change with the geometric transformation
    /// removed. Ideal (non-polarising) systems give the identity.
    pub fn proper_jones(&self) -> JonesMatrix {
        let q = &self.geometric;
        let (x, y) = self.basis_in;
        self.prt.jones((q.mul_vec(x), q.mul_vec(y)), (x, y))
    }

    /// Diattenuation and (proper) retardance of the ray.
    pub fn properties(&self) -> PolarizationProperties {
        self.proper_jones().properties()
    }

    /// Transmitted power fraction for an input state (interfaces and bulk
    /// absorption): output power divided by the input power `S₀`, the same
    /// normalisation as the radiometry API. Zero for `S₀ ≤ 0` (see
    /// [`PolarizationState::validate`]).
    pub fn intensity(&self, state: &PolarizationState) -> f64 {
        let s0 = state.stokes().s0;
        if s0 > 0.0 {
            self.jones().intensity_of(state) / s0 * self.internal_transmittance
        } else {
            0.0
        }
    }

    /// Image-local 3-D field vectors (complex amplitudes, including
    /// `√τ_internal`) for unit input states along the object-space basis
    /// `x̂` and `ŷ`: columns of the 3×2 vector pupil matrix.
    pub fn image_fields(&self) -> [CVec3; 2] {
        let a = self.internal_transmittance.max(0.0).sqrt();
        let rt = self.end_rot.transpose();
        let local = |v: CVec3| {
            let mut r = [ZERO; 3];
            for (i, ri) in r.iter_mut().enumerate() {
                *ri = (v[0] * rt.m[i][0] + v[1] * rt.m[i][1] + v[2] * rt.m[i][2]) * a;
            }
            r
        };
        [
            local(self.prt.apply(self.basis_in.0)),
            local(self.prt.apply(self.basis_in.1)),
        ]
    }

    /// Power bookkeeping of the ray for an input state, surface by surface
    /// (see [`EnergyBudget`]). Propagates the 3×3 coherency matrix
    /// `Φ → τ_seg P_i Φ P_i†`. At each interface `path` is the trace of the
    /// propagated `Φ` (the PRT amplitudes), while `diverted` and `absorbed`
    /// come from the incident s and p powers `sᵀΦs`, `pᵀΦp` and the shared
    /// power coefficients (`R`, `A = 1 − R − T`). The budget therefore
    /// closes ([`EnergyBudget::residual`]) only if the PRT matrices, the
    /// power coefficients and the bulk factors agree.
    pub fn energy_budget(&self, state: &PolarizationState) -> EnergyBudget {
        let st = state.stokes();
        let c2 = st.coherency();
        let (x, y) = self.basis_in;
        let e = [x, y];
        // Φ = E Φ₂ Eᵀ (E = [x y], real).
        let mut phi = PrtMatrix::ZERO;
        for a in 0..2 {
            for b in 0..2 {
                phi.add_dyad(c2[a][b], e[a], e[b]);
            }
        }
        let trace = |p: &PrtMatrix| p.m[0][0].re + p.m[1][1].re + p.m[2][2].re;
        let quad = |p: &PrtMatrix, v: Vec3| {
            let pv = p.apply(v);
            rdot(v, &pv).re
        };
        let mut bulk = 0.0;
        let mut surfaces = Vec::with_capacity(self.steps.len());
        let mut prev_t = 1.0;
        let mut step_iter = self.steps.iter().peekable();
        for (i, rec) in self.records.iter().enumerate().skip(1) {
            let tau = if prev_t > 0.0 {
                (rec.transmittance / prev_t).min(1.0)
            } else {
                0.0
            };
            prev_t = rec.transmittance;
            let w = trace(&phi);
            bulk += w * (1.0 - tau);
            phi = scale_prt(&phi, tau);
            let Some(stp) = step_iter.next_if(|s| s.surface == i) else {
                continue;
            };
            let incident = trace(&phi);
            let ws = quad(&phi, stp.s);
            let wp = quad(&phi, stp.k_in.cross(stp.s));
            let (ds, dp) = stp.diverted_sp();
            let p = &stp.prt;
            let mut adj = PrtMatrix::ZERO;
            for a in 0..3 {
                for b in 0..3 {
                    adj.m[a][b] = p.m[b][a].conj();
                }
            }
            phi = p.mul(&phi).mul(&adj);
            let path = trace(&phi);
            let (ps, pp) = stp.path_sp();
            surfaces.push(SurfaceLoss {
                surface: i,
                incident,
                incident_s: ws,
                incident_p: wp,
                path,
                diverted: ds * ws + dp * wp,
                absorbed: (1.0 - ps - ds) * ws + (1.0 - pp - dp) * wp,
            });
        }
        EnergyBudget {
            input: st.s0,
            transmitted: trace(&phi),
            surfaces,
            bulk_absorbed: bulk,
        }
    }
}

fn scale_prt(p: &PrtMatrix, s: f64) -> PrtMatrix {
    let mut r = *p;
    for row in &mut r.m {
        for c in row {
            *c *= s;
        }
    }
    r
}

/// Power lost or kept at one interacting surface.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SurfaceLoss {
    pub surface: usize,
    /// Power arriving at the surface (`tr Φ`).
    pub incident: f64,
    /// Incident power in the s and p components (`sᵀΦs`, `pᵀΦp`); their
    /// sum equals `incident` for a transverse field.
    pub incident_s: f64,
    pub incident_p: f64,
    /// Power continuing along the sequential path (from the PRT matrix).
    pub path: f64,
    /// Power leaving the path specularly (reflection at a refracting
    /// surface: the ghost solver's share).
    pub diverted: f64,
    /// Power absorbed by the coating (and, at a mirror, transmitted
    /// through it): `A_s·incident_s + A_p·incident_p` with
    /// `A = 1 − R − T` from the power coefficients. Conservation means
    /// `incident = path + diverted + absorbed`.
    pub absorbed: f64,
}

/// Power bookkeeping of one polarised ray:
/// `input = transmitted + Σ diverted + Σ absorbed + bulk_absorbed`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EnergyBudget {
    pub input: f64,
    pub transmitted: f64,
    pub surfaces: Vec<SurfaceLoss>,
    pub bulk_absorbed: f64,
}

impl EnergyBudget {
    /// `input − (transmitted + losses)`. Zero up to rounding when the PRT
    /// amplitudes, the power coefficients and the bulk factors are
    /// mutually consistent (it is not zero by construction:
    /// [`PolarizedRay::energy_budget`]).
    pub fn residual(&self) -> f64 {
        let mut s = mklight_core::math::KahanSum::default();
        s.add(self.input);
        s.add(-self.transmitted);
        s.add(-self.bulk_absorbed);
        for l in &self.surfaces {
            s.add(-l.diverted);
            s.add(-l.absorbed);
        }
        s.value()
    }
}

/// Trace one starting ray (starting frame, see [`crate::trace`]) with
/// polarisation.
pub fn trace_ray_polarized(
    sys: &ResolvedSystem,
    start: &Ray,
    wl: usize,
    opts: &TraceOptions,
) -> PolarizedRay {
    let mut rec = Vec::with_capacity(sys.surfaces.len());
    let r = crate::trace::trace_ray(sys, start, wl, opts, Some(&mut rec));
    PolarizedRay::from_trace(sys, wl, r, rec)
}

impl SequentialTracer {
    /// Generate and trace pupil point `(px, py)` with polarisation. A ray
    /// that cannot be generated has empty records and an identity PRT.
    pub fn trace_polarized(
        &self,
        field: &PreparedField,
        wl: usize,
        px: f64,
        py: f64,
        opts: &TraceOptions,
    ) -> PolarizedRay {
        let mut rec = Vec::with_capacity(self.system().surfaces.len());
        let r = self.trace_with(field, wl, px, py, opts, Some(&mut rec));
        PolarizedRay::from_trace(self.system(), wl, r, rec)
    }
}

// ---------------------------------------------------------------------------
// Vector pupil (the interface for polarised wave optics).

/// Polarisation data of one pupil sample for vector diffraction.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorPupilSample {
    /// Requested (pre-vignetting) pupil coordinates.
    pub pupil: [f64; 2],
    /// Ray status. When it is not OK (vignetted, missed, TIR, failed) the
    /// sample carries no light: `jones`, `field_x`, `field_y` are zero and
    /// `dir` is the zero vector.
    pub status: TraceStatus,
    /// Image-local unit direction of the ray at the image surface.
    pub dir: Vec3,
    /// Amplitude Jones matrix `√τ·J` in the double-pole bases
    /// (object space → image space, CONVENTIONS §9).
    pub jones: JonesMatrix,
    /// Image-local field vectors (`[x, y, z]` per input state, as
    /// `[[re, im]; 3]`) for unit input along the object-space `x̂` and `ŷ`
    /// of the double-pole basis, including `√τ`.
    pub field_x: [[f64; 2]; 3],
    pub field_y: [[f64; 2]; 3],
}

impl VectorPupilSample {
    fn from_ray(pupil: [f64; 2], r: &PolarizedRay) -> Self {
        if !r.is_ok() {
            return VectorPupilSample {
                pupil,
                status: r.trace.status,
                dir: Vec3::ZERO,
                jones: JonesMatrix { m: [[ZERO; 2]; 2] },
                field_x: [[0.0; 2]; 3],
                field_y: [[0.0; 2]; 3],
            };
        }
        let [fx, fy] = r.image_fields();
        let pack = |v: CVec3| v.map(|c| [c.re, c.im]);
        VectorPupilSample {
            pupil,
            status: r.trace.status,
            dir: r.trace.ray.dir,
            jones: r.jones_amplitude(),
            field_x: pack(fx),
            field_y: pack(fy),
        }
    }
    /// The image-local field for an input Jones vector (object-space
    /// double-pole basis).
    pub fn field(&self, e: &JonesVector) -> CVec3 {
        let c = |p: [f64; 2]| C::new(p[0], p[1]);
        let mut r = [ZERO; 3];
        for (i, ri) in r.iter_mut().enumerate() {
            *ri = c(self.field_x[i]) * e.x + c(self.field_y[i]) * e.y;
        }
        r
    }
}

/// Vector pupil of one (field, wavelength): one [`VectorPupilSample`] per
/// pupil point, in input order. Cancellable, with monotone progress.
/// Combine with the OPD of the same points (`exp(+i2π·OPD)` in waves for
/// CONVENTIONS §9 time dependence, see `docs/POLARIZATION.md`).
pub fn vector_pupil(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
    pupil: &[[f64; 2]],
    opts: &TraceOptions,
    ctx: &JobContext,
) -> MkResult<Vec<VectorPupilSample>> {
    run_blocks(pupil.len(), 4096, ctx, "vector pupil", |i| {
        let p = pupil[i];
        VectorPupilSample::from_ray(p, &tracer.trace_polarized(field, wl, p[0], p[1], opts))
    })
}

/// Intensity of a complex field vector.
pub fn field_intensity(v: &CVec3) -> f64 {
    cnorm2(v)
}
