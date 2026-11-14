//! Operation-for-operation twin of ghost_ops.metal. Geometry uses the trace
//! kernel's intersection and transfer; coating powers come from the
//! radiometry coating evaluation. No optical path is needed for incoherent
//! maps.
use super::*;
use crate::{
    FpEnv,
    consts::*,
    fx::{D, F, f},
    layout::{PackedHeader, PackedSurface, TraceParams},
    trace_kernel as tr,
};
type V<E> = [D<E>; 3];
fn hi<E: FpEnv>(v: V<E>) -> [F<E>; 3] {
    v.map(|v| v.hi())
}
fn norm<E: FpEnv>(v: V<E>) -> V<E> {
    let n = tr::dnorm(v);
    v.map(|x| x / n)
}
fn renorm<E: FpEnv>(d: V<E>, g: V<E>) -> V<E> {
    let m = tr::ddot(d, d) + f(-1.);
    let h = tr::dot(hi(d), hi(g));
    if h == 0. {
        return d;
    }
    let b = m.hi() / (2. * h);
    std::array::from_fn(|k| d[k] - g[k] * b)
}
fn diffract<E: FpEnv>(
    k: V<E>,
    g: V<E>,
    n1: D<E>,
    n2: D<E>,
    grad: [D<E>; 2],
    reflected: bool,
) -> Option<V<E>> {
    let normal = norm(g);
    let kn = tr::ddot(k, normal);
    let grad = [grad[0], grad[1], D::ZERO];
    let gn = tr::ddot(grad, normal);
    let inv = D::from_f(F::ONE) / n2;
    let tangent =
        std::array::from_fn(|i| ((k[i] - normal[i] * kn) * n1 + grad[i] - normal[i] * gn) * inv);
    let disc = D::from_f(F::ONE) - tr::ddot(tangent, tangent);
    if disc.hi() < 0. {
        return None;
    }
    let root = disc.sqrt() * if (kn.hi() < 0.) != reflected { -1. } else { 1. };
    Some(std::array::from_fn(|i| tangent[i] + normal[i] * root))
}
/// A clipping test whose margin lies within the f64/double-float ambiguity
/// of the hit: the reference decides it (`MKL_GHOST_APERTURE_TIE_BITS`).
/// Clear apertures sized by the rim rays of the edge field put those rays
/// exactly on the edge, where the two arithmetics may disagree.
fn aperture_tie<E: FpEnv>(s: &PackedSurface, x: D<E>, y: D<E>) -> bool {
    let tie = f32::from_bits(MKL_GHOST_APERTURE_TIE_BITS);
    let a0 = D::<E>::new(s.aperture_params[0], s.aperture_params_lo[0]);
    let a1 = D::<E>::new(s.aperture_params[1], s.aperture_params_lo[1]);
    let near = |m: D<E>, scale: f32| m.hi().v().abs() <= tie * scale;
    match s.aperture {
        MKL_APERTURE_CIRCULAR => {
            let q = a0 * a0;
            near(x * x + y * y - q, q.hi().v())
        }
        MKL_APERTURE_ANNULAR => {
            let r2 = x * x + y * y;
            let (q0, q1) = (a0 * a0, a1 * a1);
            near(r2 - q0, q0.hi().v()) || near(r2 - q1, q1.hi().v())
        }
        MKL_APERTURE_RECTANGULAR => {
            near(tr::dabs(x) - a0, a0.hi().v()) || near(tr::dabs(y) - a1, a1.hi().v())
        }
        MKL_APERTURE_ELLIPTICAL => {
            let (u, v) = (x / a0, y / a1);
            near(u * u + v * v - F::ONE, 1.)
        }
        _ => false,
    }
}
/// Coating records of the fused weight fold (`TraceCoatings`).
pub struct Coatings<'a> {
    pub specs: &'a [CoatingSpec],
    pub layers: &'a [CoatingLayer],
    pub sensor: Df32,
}
impl Coatings<'_> {
    /// Coating value of one step, with the coating kernel's record checks.
    fn evaluate(&self, coating: u32, cosine: Df32) -> CoatingValue {
        match self.specs.get(coating as usize) {
            Some(s)
                if s.count <= MKL_RAD_MAX_LAYERS
                    && s.offset as usize + s.count as usize <= self.layers.len() =>
            {
                crate::radiometry::coating::evaluate(
                    *s,
                    &self.layers[s.offset as usize..][..s.count as usize],
                    cosine,
                )
            }
            _ => CoatingValue {
                status: MKL_RAD_REFERENCE,
                ..CoatingValue::zeroed()
            },
        }
    }
}
/// One probe's whole path. With coatings, every completed step folds its
/// channel powers and efficiency into the s/p weights and its absorption
/// exponent in step order, as soon as it completes; a coating that needs the
/// reference ends the probe with `MKL_GHOST_REFERENCE`.
#[allow(clippy::too_many_arguments)]
pub fn trace<E: FpEnv>(
    h: &PackedHeader,
    s: &[PackedSurface],
    pool: &[Df32],
    steps: &[Step],
    q: Probe,
    reject_virtual: bool,
    sensor: [Df32; 2],
    newton: &crate::NewtonSettings,
    coatings: Option<Coatings>,
) -> Value {
    let prm = TraceParams {
        max_iterations: newton.max_iterations,
        tolerance_abs: newton.tolerance_abs,
        tolerance_rel: newton.tolerance_rel,
        ..Default::default()
    };
    let mut pos = q.position.map(D::from_df);
    let mut dir = norm(q.direction.map(D::from_df));
    let mut since = D::<E>::ZERO;
    let mut counting = h.flags & MKL_SYS_OBJECT_AT_INFINITY == 0;
    let mut v = Value {
        weight_s: Df32::from_f32(1.),
        weight_p: Df32::from_f32(1.),
        internal: Df32::from_f32(1.),
        ..Value::zeroed()
    };
    let one = D::<E>::from_f(F::ONE);
    let (mut ws, mut wp, mut absorb) = (one, one, D::<E>::ZERO);
    for st in steps {
        let si = (st.word & MKL_STEP_SURFACE_MASK) as usize;
        let back = st.word & MKL_STEP_BACKWARD != 0;
        let surf = &s[si];
        let fail = |c| c | ((si as u32) << MKL_RAY_SURFACE_SHIFT);
        if back {
            tr::xf_backward(&s[si], &mut pos, &mut dir)
        } else {
            tr::xf_forward(&s[si - 1], &mut pos, &mut dir)
        }
        let root = if !back && si == 1 && h.flags & MKL_SYS_OBJECT_AT_INFINITY != 0 {
            MKL_ROOT_NEAREST
        } else {
            MKL_ROOT_FROM_START
        };
        let hit = match tr::intersect(surf, pool, pos, dir, root, &prm) {
            Ok(h) => h,
            Err(c) => {
                v.status = fail(c);
                break;
            }
        };
        let t = hit.ta + hit.tb;
        if t.hi().abs() > MKL_GHOST_MAX_SEGMENT_MM as f32 {
            v.status = fail(MKL_GHOST_REFERENCE);
            break;
        }
        if counting {
            since = since + t;
        }
        if st.physical != 0 {
            if counting && reject_virtual && since.hi() < -1e-9 {
                v.status = fail(MKL_GHOST_VIRTUAL);
                break;
            }
            since = D::ZERO;
            counting = true;
        }
        pos = hit.x;
        if aperture_tie(surf, pos[0], pos[1]) {
            v.status = fail(MKL_GHOST_REFERENCE);
            break;
        }
        if !tr::passes_aperture(surf, pos[0], pos[1]) {
            v.status = fail(MKL_RAY_VIGNETTED);
            break;
        }
        if st.channel == 4
            && (pos[0].hi().abs() > D::<E>::from_df(sensor[0]).hi()
                || pos[1].hi().abs() > D::<E>::from_df(sensor[1]).hi())
        {
            v.status = fail(MKL_RAY_VIGNETTED);
            break;
        }
        let n1s = &s[if back { si } else { si - 1 }];
        let n2s = &s[if back { si - 1 } else { si }];
        let w = q.wavelength as usize;
        let n1 = D::<E>::new(n1s.n_after[w], n1s.n_after_lo[w]);
        let n2 = D::<E>::new(n2s.n_after[w], n2s.n_after_lo[w]);
        let n2 = if st.channel == 2 || st.channel == 4 {
            n1
        } else {
            n2
        };
        let cosine = tr::ddot(dir, hit.g) / tr::dnorm(hit.g);
        let cosine = if cosine.hi() < 0. { -cosine } else { cosine };
        let mut next = dir;
        let refl = st.channel == 2 || st.channel == 4;
        if st.channel == 1 || refl {
            if surf.shape == MKL_SHAPE_DIFFRACTIVE && st.channel != 4 {
                let p = crate::surface_math::phase(surf, pool, pos[0], pos[1]);
                let scale = D::from_df(st.scale);
                match diffract(dir, hit.g, n1, n2, [p.x * scale, p.y * scale], refl) {
                    Some(d) => next = d,
                    None => {
                        v.status = fail(MKL_RAY_EVANESCENT);
                        break;
                    }
                }
            } else if surf.shape == MKL_SHAPE_PARAXIAL && st.channel != 4 {
                let mut kk = dir;
                if refl {
                    kk[2] = -kk[2];
                }
                match tr::ideal_lens(pos[0], pos[1], kk, D::from_df(st.power), n1, n2, D::ZERO) {
                    Some((d, _)) => next = d,
                    None => {
                        v.status = fail(MKL_RAY_DEGENERATE);
                        break;
                    }
                }
            } else if refl {
                let gh = hi(hit.g);
                let b = -2. * tr::dot(hi(dir), gh) / tr::dot(gh, gh);
                next = renorm(std::array::from_fn(|j| dir[j] + hit.g[j] * b), hit.g);
            } else {
                let mu = n1 / n2;
                let one = D::<E>::from_f(F::ONE);
                let disc = one - (mu * mu) * (one - cosine * cosine);
                // The f32 TIR branch and grazing exit geometry are not
                // qualified inside this critical-angle envelope.
                if disc.hi().abs() <= f32::from_bits(MKL_GHOST_CRITICAL_BAND_BITS) {
                    v.status = fail(MKL_GHOST_REFERENCE);
                    break;
                }
                let Some((_, _, gamma)) = tr::refract(hi(dir), hit.normal, n1.hi(), n2.hi()) else {
                    v.status = fail(MKL_RAY_TIR);
                    break;
                };
                let b = gamma / tr::dnorm(hit.g).hi();
                let d = std::array::from_fn(|j| dir[j] * mu + hit.g[j] * b);
                next = renorm(renorm(d, hit.g), hit.g);
            }
        }
        if let Some(c) = &coatings {
            let sensor = D::<E>::from_df(c.sensor);
            let (a, b) = match st.channel {
                1 | 2 => {
                    let value = c.evaluate(st.coating, cosine.0);
                    if value.status != MKL_RAD_OK {
                        v.status = MKL_GHOST_REFERENCE;
                        break;
                    }
                    let p = value.powers.map(D::<E>::from_df);
                    if st.channel == 1 {
                        (p[2], p[3])
                    } else {
                        (p[0], p[1])
                    }
                }
                3 => (one - sensor, one - sensor),
                4 => (sensor, sensor),
                _ => (one, one),
            };
            let eta = D::from_df(st.efficiency);
            ws = ws * (a * eta);
            wp = wp * (b * eta);
            absorb = absorb + D::<E>::from_df(st.alpha) * t;
        }
        dir = next;
        v.done += 1;
    }
    if coatings.is_some() {
        v.weight_s = ws.0;
        v.weight_p = wp.0;
        v.internal = D::from_f(f::<E>(crate::fmath::exp_in::<E>((-absorb).hi().v()))).0;
    }
    v.position = pos.map(|v| v.0);
    v.direction = dir.map(|v| v.0);
    v
}

pub(super) fn launch<E: FpEnv>(
    f: &crate::aiming::FieldSpec,
    q: LaunchSample,
    pos: [Df32; 3],
    dir: [Df32; 3],
) -> Probe {
    let mut pos = pos.map(D::<E>::from_df);
    let mut dir = dir.map(D::<E>::from_df);
    let shift = q.pupil.map(|x| D::from_df(x) * D::from_df(f.pupil[1]));
    let rotation = f.rotation.map(D::<E>::from_df);
    let local = |v: V<E>| -> V<E> {
        std::array::from_fn(|k| {
            rotation[k] * v[0] + rotation[3 + k] * v[1] + rotation[6 + k] * v[2]
        })
    };
    let global = |v: V<E>| -> V<E> {
        std::array::from_fn(|k| {
            rotation[k * 3] * v[0] + rotation[k * 3 + 1] * v[1] + rotation[k * 3 + 2] * v[2]
        })
    };
    if f.flags & MKL_AIM_FINITE == 0 {
        let delta = [shift[0], shift[1], D::ZERO];
        let along = tr::ddot(delta, dir);
        pos = std::array::from_fn(|k| pos[k] + delta[k] - dir[k] * along);
    } else if f.flags & MKL_AIM_TELECENTRIC != 0 {
        let d = global(dir);
        let az = tr::dabs(d[2]);
        dir = norm(local([
            d[0] / az + shift[0],
            d[1] / az + shift[1],
            D::from_f(F::ONE),
        ]));
    } else {
        let origin = f.origin.map(D::from_df);
        let pg = global(pos);
        let pg: V<E> = std::array::from_fn(|k| pg[k] + origin[k]);
        let dg = global(dir);
        let t = (D::from_df(f.pupil[0]) - pg[2]) / dg[2];
        let ap = [
            pg[0] + dg[0] * t + shift[0],
            pg[1] + dg[1] * t + shift[1],
            D::from_df(f.pupil[0]),
        ];
        dir = norm(local(std::array::from_fn(|k| ap[k] - pg[k])));
        if tr::ddot([rotation[6], rotation[7], rotation[8]], dir).hi() < 0. {
            dir = dir.map(|v| -v);
        }
    }
    Probe {
        position: pos.map(|v| v.0),
        direction: dir.map(|v| v.0),
        path: q.path,
        wavelength: q.wavelength,
    }
}
pub(super) fn pixel<E: FpEnv>(xy: [Df32; 2], bins: [u32; 2], half: [Df32; 2]) -> u32 {
    let f = std::array::from_fn::<_, 2, _>(|k| {
        let v = (D::<E>::from_df(xy[k]) + D::from_df(half[k])) / (D::from_df(half[k]) * 2.)
            * bins[k] as f32;
        // Floor of hi+lo: an integral hi with a negative lo lies just below it.
        let (hi, lo) = (v.hi().v(), v.lo().v());
        if hi.floor() == hi && lo < 0. {
            hi - 1.
        } else {
            hi.floor()
        }
    });
    if (0..2).any(|k| !(f[k] >= 0. && f[k] < bins[k] as f32)) {
        return bins[0] * bins[1];
    }
    f[1] as u32 * bins[0] + f[0] as u32
}
pub(super) fn leaf_weight<E: FpEnv>(v: Value, area: Df32) -> Df32 {
    (((D::<E>::from_df(v.weight_s) + D::from_df(v.weight_p)) * 0.5)
        * D::from_df(v.internal)
        * D::from_df(area))
    .0
}
