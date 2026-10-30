//! Shared pieces of every analysis: field/wavelength selections, the
//! non-finite-safe serde helpers, weighted spot statistics, Philox pupil
//! patterns and the `RunInfo` builder.

use std::f64::consts::PI;

use mklight_core::job::{Precision, RunTimer};
use mklight_core::math::KahanSum;
use mklight_core::rng::Philox;
use mklight_core::{FieldKind, MkError, MkResult, RunInfo};
use mklight_trace::{ObjectSource, PreparedField, SequentialTracer};
use serde::{Deserialize, Serialize};

/// Backend identifier recorded in `RunInfo` by every analysis of this crate
/// (the CPU f64 reference tracer).
pub const BACKEND: &str = "cpu-f64";

/// Build the `RunInfo` of a finished analysis. The tracer's warnings
/// (resolution, paraxial, aiming) are prepended to the analysis' own.
pub(crate) fn run_info(
    timer: RunTimer,
    tracer: &SequentialTracer,
    analysis: &str,
    settings: &impl Serialize,
    seed: Option<u64>,
    warnings: Vec<String>,
) -> RunInfo {
    let mut w = tracer.warnings().to_vec();
    w.extend(warnings);
    timer.finish(
        &tracer.system().snapshot,
        analysis,
        settings,
        BACKEND,
        Precision::F64Reference,
        seed,
        w,
    )
}

// ----- Selections ---------------------------------------------------------------

/// Which system fields an analysis covers. JSON: `{"type": "all"}`,
/// `{"type": "index", "index": 2}` or `{"type": "list", "indices": [0, 2]}`.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldSelection {
    #[default]
    All,
    Index {
        index: usize,
    },
    List {
        indices: Vec<usize>,
    },
}

impl FieldSelection {
    /// The selected field indices of a system with `count` fields, in
    /// ascending order without duplicates.
    pub fn resolve(&self, count: usize) -> MkResult<Vec<usize>> {
        resolve_list(
            match self {
                FieldSelection::All => return Ok((0..count).collect()),
                FieldSelection::Index { index } => std::slice::from_ref(index),
                FieldSelection::List { indices } => indices,
            },
            count,
            "field",
        )
    }
}

/// Which system wavelengths an analysis covers. JSON: `{"type": "all"}`,
/// `{"type": "primary"}`, `{"type": "index", "index": 0}` or
/// `{"type": "list", "indices": [...]}`.
#[derive(Clone, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum WavelengthSelection {
    #[default]
    All,
    Primary,
    Index {
        index: usize,
    },
    List {
        indices: Vec<usize>,
    },
}

impl WavelengthSelection {
    /// The selected wavelength indices (ascending, without duplicates).
    pub fn resolve(&self, count: usize, primary: usize) -> MkResult<Vec<usize>> {
        resolve_list(
            match self {
                WavelengthSelection::All => return Ok((0..count).collect()),
                WavelengthSelection::Primary => std::slice::from_ref(&primary),
                WavelengthSelection::Index { index } => std::slice::from_ref(index),
                WavelengthSelection::List { indices } => indices,
            },
            count,
            "wavelength",
        )
    }
}

fn resolve_list(list: &[usize], count: usize, what: &str) -> MkResult<Vec<usize>> {
    let mut v = list.to_vec();
    v.sort_unstable();
    v.dedup();
    if v.is_empty() {
        return Err(MkError::InvalidModel(format!("no {what} selected")));
    }
    if let Some(&bad) = v.iter().find(|&&i| i >= count) {
        return Err(MkError::InvalidModel(format!(
            "{what} {bad} does not exist ({count} defined)"
        )));
    }
    Ok(v)
}

/// A wavelength index, or the primary wavelength when `None`.
pub fn wavelength_or_primary(tracer: &SequentialTracer, w: Option<usize>) -> MkResult<usize> {
    let sys = tracer.system();
    let w = w.unwrap_or(sys.primary_wavelength);
    if w >= sys.wavelengths_um.len() {
        return Err(MkError::InvalidModel(format!(
            "wavelength {w} does not exist"
        )));
    }
    Ok(w)
}

/// Display unit of field values: `°` for angles, `mm` for heights.
pub fn field_unit(kind: FieldKind) -> &'static str {
    match kind {
        FieldKind::ObjectAngle => "°",
        _ => "mm",
    }
}

/// Field value `(x, y)` of system field `i`, in the system's field units.
pub(crate) fn field_value(tracer: &SequentialTracer, i: usize) -> [f64; 2] {
    let f = &tracer.system().system().fields.fields[i];
    [f.x, f.y]
}

/// Physical field value (system field units) of normalised field `h`.
pub(crate) fn field_scale(tracer: &SequentialTracer) -> f64 {
    tracer.system().system().fields.max_radial()
}

/// The paraxial image point `(x, y)` (image local frame) of a prepared field
/// at wavelength `wl`: the paraxial chief-ray height per unit `tan θ` (object
/// at infinity) or per unit object height (finite object), applied to the
/// field's object source. `None` when the paraxial chief ray is undefined
/// (e.g. an object-space telecentric system with an angle source).
pub(crate) fn paraxial_image_point(
    tracer: &SequentialTracer,
    field: &PreparedField,
    wl: usize,
) -> Option<[f64; 2]> {
    let fo = tracer.paraxial().at(wl);
    match field.source {
        ObjectSource::Direction { dir } => {
            let k = fo.image_height_per_tan?;
            (dir.z != 0.0).then(|| [k * dir.x / dir.z, k * dir.y / dir.z])
        }
        ObjectSource::Point { point } => {
            let k = fo.image_height_per_object_height?;
            Some([k * point.x, k * point.y])
        }
    }
}

pub(crate) fn check_count(name: &str, v: usize, lo: usize, hi: usize) -> MkResult<()> {
    if v < lo || v > hi {
        return Err(MkError::InvalidModel(format!(
            "{name} must be in {lo}…{hi} (is {v})"
        )));
    }
    Ok(())
}

pub(crate) fn check_finite(name: &str, v: f64) -> MkResult<()> {
    if !v.is_finite() {
        return Err(MkError::InvalidModel(format!("{name} must be finite")));
    }
    Ok(())
}

/// `n` evenly spaced values from `a` to `b` inclusive (`[a]` for `n = 1`).
pub(crate) fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
    match n {
        0 => vec![],
        1 => vec![a],
        _ => (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                // Exact end points; (1 − t)a + tb is monotone and exact at both ends.
                if i == n - 1 {
                    b
                } else {
                    (1.0 - t).mul_add(a, t * b)
                }
            })
            .collect(),
    }
}

/// Gauss–Legendre nodes and weights on `[0, 1]` (`n ≥ 1`), by Newton
/// iteration on `P_n` from the Chebyshev-like initial guesses; exact for
/// polynomials of degree `2n − 1`.
pub fn gauss_legendre_01(n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut x = vec![0.0; n];
    let mut w = vec![0.0; n];
    let nf = n as f64;
    for i in 0..n.div_ceil(2) {
        let mut z = (PI * (i as f64 + 0.75) / (nf + 0.5)).cos();
        let mut dp = 1.0;
        for _ in 0..100 {
            let (mut p0, mut p1) = (1.0f64, z);
            for k in 2..=n {
                let kf = k as f64;
                let p2 = ((2.0 * kf - 1.0) * z * p1 - (kf - 1.0) * p0) / kf;
                p0 = p1;
                p1 = p2;
            }
            dp = nf * (z * p1 - p0) / (z * z - 1.0);
            let dz = p1 / dp;
            z -= dz;
            if dz.abs() < 1e-16 {
                break;
            }
        }
        let wi = 2.0 / ((1.0 - z * z) * dp * dp);
        // Map [−1, 1] → [0, 1].
        x[i] = 0.5 * (1.0 - z);
        x[n - 1 - i] = 0.5 * (1.0 + z);
        w[i] = 0.5 * wi;
        w[n - 1 - i] = 0.5 * wi;
    }
    (x, w)
}

// ----- Pupil patterns -----------------------------------------------------------

/// Pupil sampling of the spot-based analyses. Each pattern's density is set
/// by `rings` (see the variants); every pattern is deterministic.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PupilPattern {
    /// Centre plus `rings` rings of `6k` points: `1 + 3·rings·(rings+1)` rays.
    #[default]
    Hexapolar,
    /// Square grid of pitch `1/rings` over `[−1, 1]²`, clipped to the unit
    /// disk (`2·rings + 1` points per side).
    Square,
    /// `1 + 3·rings·(rings+1)` points uniform over the disk (`ρ = √u₁`,
    /// `φ = 2πu₂`), Philox stream 0 of the seed (CONVENTIONS §6).
    Random,
    /// Jittered square grid: one uniform point inside each cell of the
    /// `Square` grid (Philox stream 1 of the seed), clipped to the disk.
    Dithered,
}

/// The pupil points of `pattern` (CONVENTIONS §3 normalised coordinates).
/// Random patterns draw `u32_to_unit_f64` uniforms from `Philox(seed, stream)`
/// at fixed draw indices, so the set depends only on `(pattern, rings, seed)`.
pub fn pupil_points(pattern: PupilPattern, rings: usize, seed: u64) -> Vec<[f64; 2]> {
    let r = rings.max(1);
    match pattern {
        PupilPattern::Hexapolar => mklight_trace::pupil::hexapolar(r),
        PupilPattern::Square => mklight_trace::pupil::grid(2 * r + 1),
        PupilPattern::Random => {
            let n = 1 + 3 * r * (r + 1);
            let mut g = Philox::new(seed, 0);
            (0..n)
                .map(|_| {
                    let rho = g.uniform().sqrt();
                    let (s, c) = (2.0 * PI * g.uniform()).sin_cos();
                    [rho * c, rho * s]
                })
                .collect()
        }
        PupilPattern::Dithered => {
            let n = 2 * r + 1;
            let cell = 2.0 / n as f64;
            let mut g = Philox::new(seed, 1);
            let mut out = Vec::with_capacity(n * n);
            for j in 0..n {
                for i in 0..n {
                    // Two draws per cell whether or not the point is kept.
                    let (ux, uy) = (g.uniform(), g.uniform());
                    let x = (i as f64 + ux).mul_add(cell, -1.0);
                    let y = (j as f64 + uy).mul_add(cell, -1.0);
                    if x.mul_add(x, y * y) <= 1.0 {
                        out.push([x, y]);
                    }
                }
            }
            out
        }
    }
}

// ----- Weighted point statistics ---------------------------------------------------

/// Weighted 2-D point statistics. Points are given **relative to an origin**
/// (normally the chief ray), so the large common image height is removed
/// before any sum; centroid and second moments are compensated sums, and the
/// RMS about the centroid is a two-pass `Σw|q − μ|²`, never `E[q²] − μ²`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Moments {
    pub weight: f64,
    pub count: usize,
    /// Centroid relative to the origin.
    pub centroid: [f64; 2],
    pub rms_origin: f64,
    pub rms_centroid: f64,
    pub geo_origin: f64,
    pub geo_centroid: f64,
    /// RMS of the x and y components about the centroid.
    pub rms_x: f64,
    pub rms_y: f64,
}

impl Moments {
    /// `pts`: `(q, w)` with `q` relative to the origin. NaN statistics when
    /// the total weight is zero.
    pub fn of(pts: impl Iterator<Item = ([f64; 2], f64)> + Clone) -> Moments {
        let mut sw = KahanSum::default();
        let mut sx = KahanSum::default();
        let mut sy = KahanSum::default();
        let mut count = 0;
        for (q, w) in pts.clone() {
            sw.add(w);
            sx.add(w * q[0]);
            sy.add(w * q[1]);
            count += 1;
        }
        let wt = sw.value();
        if !(wt > 0.0) {
            return Moments {
                weight: 0.0,
                count,
                centroid: [f64::NAN; 2],
                rms_origin: f64::NAN,
                rms_centroid: f64::NAN,
                geo_origin: f64::NAN,
                geo_centroid: f64::NAN,
                rms_x: f64::NAN,
                rms_y: f64::NAN,
            };
        }
        let mu = [sx.value() / wt, sy.value() / wt];
        let mut s0 = KahanSum::default();
        let mut cx = KahanSum::default();
        let mut cy = KahanSum::default();
        let (mut g0, mut gc) = (0.0f64, 0.0f64);
        for (q, w) in pts {
            let (dx, dy) = (q[0] - mu[0], q[1] - mu[1]);
            s0.add(w * q[0].mul_add(q[0], q[1] * q[1]));
            cx.add(w * dx * dx);
            cy.add(w * dy * dy);
            if w > 0.0 {
                g0 = g0.max(q[0].hypot(q[1]));
                gc = gc.max(dx.hypot(dy));
            }
        }
        let (vx, vy) = (cx.value() / wt, cy.value() / wt);
        Moments {
            weight: wt,
            count,
            centroid: mu,
            rms_origin: (s0.value() / wt).sqrt(),
            rms_centroid: (vx + vy).sqrt(),
            geo_origin: g0,
            geo_centroid: gc,
            rms_x: vx.sqrt(),
            rms_y: vy.sqrt(),
        }
    }
}

// ----- Brent minimisation -----------------------------------------------------------

/// Result of [`brent_minimize`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BrentResult {
    pub x: f64,
    pub fx: f64,
    pub evaluations: usize,
    pub converged: bool,
}

/// Brent's method (golden section with parabolic interpolation) for the
/// minimum of `f` on `[a, b]`. Terminates when the bracket is below
/// `2·(tol_rel·|x| + tol_abs)`; `max_eval` bounds the evaluations. For a
/// unimodal `f` it converges superlinearly to the minimum; otherwise to a
/// local minimum inside the bracket.
pub fn brent_minimize(
    mut f: impl FnMut(f64) -> MkResult<f64>,
    a: f64,
    b: f64,
    tol_rel: f64,
    tol_abs: f64,
    max_eval: usize,
) -> MkResult<BrentResult> {
    const CGOLD: f64 = 0.381_966_011_250_105_1; // (3 − √5)/2
    let (mut a, mut b) = if a <= b { (a, b) } else { (b, a) };
    let mut x = a + CGOLD * (b - a);
    let (mut w, mut v) = (x, x);
    let mut fx = f(x)?;
    let (mut fw, mut fv) = (fx, fx);
    let mut evals = 1;
    let (mut d, mut e) = (0.0f64, 0.0f64);
    while evals < max_eval {
        let xm = 0.5 * (a + b);
        let tol1 = tol_rel.mul_add(x.abs(), tol_abs);
        let tol2 = 2.0 * tol1;
        if (x - xm).abs() <= tol2 - 0.5 * (b - a) {
            return Ok(BrentResult {
                x,
                fx,
                evaluations: evals,
                converged: true,
            });
        }
        let mut golden = true;
        if e.abs() > tol1 {
            // Parabola through (v, fv), (w, fw), (x, fx).
            let r = (x - w) * (fx - fv);
            let mut q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if q > 0.0 {
                p = -p;
            }
            q = q.abs();
            let etemp = e;
            e = d;
            if p.abs() < (0.5 * q * etemp).abs() && p > q * (a - x) && p < q * (b - x) {
                d = p / q;
                let u = x + d;
                if u - a < tol2 || b - u < tol2 {
                    d = tol1.copysign(xm - x);
                }
                golden = false;
            }
        }
        if golden {
            e = if x >= xm { a - x } else { b - x };
            d = CGOLD * e;
        }
        let u = if d.abs() >= tol1 {
            x + d
        } else {
            x + tol1.copysign(d)
        };
        let fu = f(u)?;
        evals += 1;
        if fu <= fx {
            if u >= x {
                a = x;
            } else {
                b = x;
            }
            (v, fv) = (w, fw);
            (w, fw) = (x, fx);
            (x, fx) = (u, fu);
        } else {
            if u < x {
                a = u;
            } else {
                b = u;
            }
            if fu <= fw || w == x {
                (v, fv) = (w, fw);
                (w, fw) = (u, fu);
            } else if fu <= fv || v == x || v == w {
                (v, fv) = (u, fu);
            }
        }
    }
    Ok(BrentResult {
        x,
        fx,
        evaluations: evals,
        converged: false,
    })
}

// ----- Non-finite-safe serde ---------------------------------------------------------

/// Serde for result numbers that may be non-finite (core `serde_util`
/// convention, CONVENTIONS §2): `±∞` are written as the strings `"inf"` /
/// `"-inf"` and NaN (undefined, e.g. a vignetted ray) as `null`. Reading
/// accepts numbers, `null` (NaN), `"inf"`, `"-inf"` and `"nan"`.
pub mod nf {
    use serde::de::Error as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    /// One possibly non-finite `f64` as a serde value.
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct Num(pub f64);

    impl Serialize for Num {
        fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let v = self.0;
            if v.is_nan() {
                s.serialize_none()
            } else if v.is_infinite() {
                s.serialize_str(if v > 0.0 { "inf" } else { "-inf" })
            } else {
                s.serialize_f64(v)
            }
        }
    }

    impl<'de> Deserialize<'de> for Num {
        fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            #[derive(Deserialize)]
            #[serde(untagged)]
            enum Raw {
                F(f64),
                S(String),
                N(()),
            }
            match Raw::deserialize(d)? {
                Raw::F(f) => Ok(Num(f)),
                Raw::N(_) => Ok(Num(f64::NAN)),
                Raw::S(s) => match s.trim().to_ascii_lowercase().as_str() {
                    "inf" | "+inf" | "infinity" | "+infinity" => Ok(Num(f64::INFINITY)),
                    "-inf" | "-infinity" => Ok(Num(f64::NEG_INFINITY)),
                    "nan" => Ok(Num(f64::NAN)),
                    o => o.parse::<f64>().map(Num).map_err(D::Error::custom),
                },
            }
        }
    }

    pub fn serialize<S: Serializer>(v: &f64, s: S) -> Result<S::Ok, S::Error> {
        Num(*v).serialize(s)
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
        Num::deserialize(d).map(|n| n.0)
    }

    /// `Vec<f64>`.
    pub mod vec {
        use super::*;
        pub fn serialize<S: Serializer>(v: &[f64], s: S) -> Result<S::Ok, S::Error> {
            s.collect_seq(v.iter().map(|x| Num(*x)))
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<f64>, D::Error> {
            Ok(Vec::<Num>::deserialize(d)?
                .into_iter()
                .map(|n| n.0)
                .collect())
        }
    }

    /// `[f64; 2]` as `[x, y]`.
    pub mod arr2 {
        use super::*;
        pub fn serialize<S: Serializer>(v: &[f64; 2], s: S) -> Result<S::Ok, S::Error> {
            [Num(v[0]), Num(v[1])].serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f64; 2], D::Error> {
            let a = <[Num; 2]>::deserialize(d)?;
            Ok([a[0].0, a[1].0])
        }
    }

    /// `[f64; 3]` as `[x, y, z]`.
    pub mod arr3 {
        use super::*;
        pub fn serialize<S: Serializer>(v: &[f64; 3], s: S) -> Result<S::Ok, S::Error> {
            [Num(v[0]), Num(v[1]), Num(v[2])].serialize(s)
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[f64; 3], D::Error> {
            let a = <[Num; 3]>::deserialize(d)?;
            Ok([a[0].0, a[1].0, a[2].0])
        }
    }

    /// `Vec<[f64; 2]>`.
    pub mod vec2 {
        use super::*;
        pub fn serialize<S: Serializer>(v: &[[f64; 2]], s: S) -> Result<S::Ok, S::Error> {
            s.collect_seq(v.iter().map(|p| [Num(p[0]), Num(p[1])]))
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[f64; 2]>, D::Error> {
            Ok(Vec::<[Num; 2]>::deserialize(d)?
                .into_iter()
                .map(|a| [a[0].0, a[1].0])
                .collect())
        }
    }

    /// `Vec<[f64; 3]>`.
    pub mod vec3 {
        use super::*;
        pub fn serialize<S: Serializer>(v: &[[f64; 3]], s: S) -> Result<S::Ok, S::Error> {
            s.collect_seq(v.iter().map(|p| [Num(p[0]), Num(p[1]), Num(p[2])]))
        }
        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<[f64; 3]>, D::Error> {
            Ok(Vec::<[Num; 3]>::deserialize(d)?
                .into_iter()
                .map(|a| [a[0].0, a[1].0, a[2].0])
                .collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Serialize, Deserialize, Debug)]
    struct T {
        #[serde(with = "nf")]
        a: f64,
        #[serde(with = "nf::vec")]
        v: Vec<f64>,
        #[serde(with = "nf::vec2")]
        p: Vec<[f64; 2]>,
    }

    #[test]
    fn non_finite_roundtrip() {
        let t = T {
            a: f64::NEG_INFINITY,
            v: vec![1.5, f64::NAN, f64::INFINITY],
            p: vec![[f64::NAN, 2.0]],
        };
        let s = serde_json::to_string(&t).unwrap();
        assert_eq!(s, r#"{"a":"-inf","v":[1.5,null,"inf"],"p":[[null,2.0]]}"#);
        let u: T = serde_json::from_str(&s).unwrap();
        assert_eq!(u.a, f64::NEG_INFINITY);
        assert!(u.v[1].is_nan() && u.v[2] == f64::INFINITY && u.p[0][0].is_nan());
        let w: T = serde_json::from_str(r#"{"a":"nan","v":[],"p":[]}"#).unwrap();
        assert!(w.a.is_nan());
    }

    #[test]
    fn gauss_legendre_integrates_polynomials_exactly() {
        for n in 1..12 {
            let (x, w) = gauss_legendre_01(n);
            assert!(x.windows(2).all(|p| p[0] < p[1]));
            for deg in 0..2 * n {
                let q: f64 = x.iter().zip(&w).map(|(x, w)| w * x.powi(deg as i32)).sum();
                assert!(
                    (q - 1.0 / (deg + 1) as f64).abs() < 1e-14,
                    "n {n} deg {deg}"
                );
            }
        }
    }

    #[test]
    fn selections() {
        let s: FieldSelection = serde_json::from_str(r#"{"type":"index","index":2}"#).unwrap();
        assert_eq!(s.resolve(3).unwrap(), vec![2]);
        assert!(s.resolve(2).is_err());
        let w: WavelengthSelection = serde_json::from_str(r#"{"type":"primary"}"#).unwrap();
        assert_eq!(w.resolve(3, 1).unwrap(), vec![1]);
        assert_eq!(
            serde_json::to_string(&FieldSelection::All).unwrap(),
            r#"{"type":"all"}"#
        );
    }

    #[test]
    fn brent_finds_minimum() {
        // Asymmetric smooth function with minimum at x = ln 2.
        let r = brent_minimize(|x| Ok((x).exp() - 2.0 * x), -3.0, 5.0, 1e-12, 1e-14, 200).unwrap();
        assert!(r.converged);
        assert!((r.x - 2f64.ln()).abs() < 1e-7, "{}", r.x);
        // |x − 0.3| is not smooth: golden-section steps still converge.
        let r = brent_minimize(|x| Ok((x - 0.3).abs()), -1.0, 1.0, 0.0, 1e-10, 500).unwrap();
        assert!((r.x - 0.3).abs() < 1e-9);
    }

    #[test]
    fn moments_two_pass() {
        // Large common offset: the naive E[x²] − μ² would lose every digit.
        let (a, b) = (1.0e4, 1.0e4 + 2.0e-6);
        let half = 0.5 * (b - a); // exact (Sterbenz)
        let pts: Vec<([f64; 2], f64)> = (0..100)
            .map(|i| ([if i % 2 == 0 { a } else { b }, 0.0], 1.0))
            .collect();
        let m = Moments::of(pts.iter().copied());
        assert!((m.rms_centroid - half).abs() < 1e-14, "{}", m.rms_centroid);
        assert!((m.geo_centroid - half).abs() < 1e-11);
        let naive = (pts.iter().map(|p| p.0[0] * p.0[0]).sum::<f64>() / 100.0
            - m.centroid[0] * m.centroid[0])
            .max(0.0)
            .sqrt();
        assert!(
            (naive - half).abs() > 1e-8,
            "naive form unexpectedly accurate"
        );
    }

    #[test]
    fn pupil_patterns_deterministic_and_inside() {
        for p in [
            PupilPattern::Hexapolar,
            PupilPattern::Square,
            PupilPattern::Random,
            PupilPattern::Dithered,
        ] {
            let a = pupil_points(p, 5, 7);
            assert_eq!(a, pupil_points(p, 5, 7));
            assert!(a.iter().all(|q| q[0].hypot(q[1]) <= 1.0 + 1e-12));
            assert!(a.len() > 50, "{p:?}");
        }
        assert_ne!(
            pupil_points(PupilPattern::Random, 5, 7),
            pupil_points(PupilPattern::Random, 5, 8)
        );
        // Uniform random sampling: mean ρ² → 1/2.
        let r = pupil_points(PupilPattern::Random, 60, 3);
        let m = r.iter().map(|q| q[0] * q[0] + q[1] * q[1]).sum::<f64>() / r.len() as f64;
        assert!((m - 0.5).abs() < 0.01, "{m}");
    }
}
