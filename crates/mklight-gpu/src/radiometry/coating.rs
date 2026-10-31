use super::{math::*, *};
use crate::{Df32 as D, consts::*};
use bytemuck::Zeroable;
fn scaled(a: C, p: D) -> C {
    let n = a.norm();
    if value(n) > 0.0 {
        a * C::real(positive(p).sqrt() / n)
    } else {
        C::real(positive(p).sqrt())
    }
}
fn adm(n: C, c: C, p: bool) -> C {
    if p { n * n / c } else { c }
}
struct LayerTerms {
    n: C,
    c: C,
    cs: C,
    isn: C,
    phase: C,
}
impl LayerTerms {
    fn new(l: CoatingLayer, invariant: D, wavelength: D) -> Self {
        let n = C::new(l.n, -l.k);
        let c = nc(n, invariant);
        let phase =
            c * C::real(D::new(std::f32::consts::TAU, -1.7484555e-7) * l.thickness / wavelength);
        let (sn, cs) = phase.sincos();
        Self {
            n,
            c,
            cs,
            isn: C::new(-sn.im, sn.re),
            phase,
        }
    }
    fn matrix(&self, p: bool) -> Matrix {
        let eta = adm(self.n, self.c, p);
        [self.cs, self.isn / eta, self.isn * eta, self.cs]
    }
}
/// Independent absorption from the normal Poynting flux at each layer's
/// two faces. The two polarisations share propagation phases, not powers.
fn film(s: CoatingSpec, ls: &[CoatingLayer], ci: D) -> ([C; 4], [D; 2], bool) {
    let ni = C::real(s.n_in);
    let ns = C::new(s.substrate_n, -s.substrate_k);
    let inv = s.n_in * positive((d(1.0) - ci) * (d(1.0) + ci)).sqrt();
    let c0 = C::real(s.n_in * ci);
    let cs = nc(ns, inv);
    if value(ci) == 0.0 {
        return (
            [
                C::real(d(-1.0)),
                C::real(d(-1.0)),
                C::default(),
                C::default(),
            ],
            [D::ZERO; 2],
            true,
        );
    }
    let e0 = [adm(ni, c0, false), adm(ni, c0, true)];
    let es = if value(cs.norm2()) == 0.0 {
        [C::default(); 2]
    } else {
        [adm(ns, cs, false), adm(ns, cs, true)]
    };
    let mut m = [identity(); 2];
    let mut a = [C::default(); 4];
    let mut absorption = [D::ZERO; 2];
    for &l in ls {
        let term = LayerTerms::new(l, inv, s.wavelength);
        if value(term.phase.im).abs() > 30.0 || value(term.phase.re).abs() > 1e6 {
            return (a, absorption, false);
        }
        for (p, m) in m.iter_mut().enumerate() {
            *m = product(*m, term.matrix(p == 1));
        }
    }
    let mut e = [C::default(); 2];
    let mut h = e;
    for p in 0..2 {
        let b = m[p][0] + m[p][1] * es[p];
        let c = m[p][2] + m[p][3] * es[p];
        let den = e0[p] * b + c;
        let rho = (e0[p] * b - c) / den;
        let tau = C::real(d(2.0)) * e0[p] / den;
        a[p] = if p == 1 { -rho } else { rho };
        let flux = positive(e0[p].re * es[p].re / e0[p].norm2());
        a[p + 2] = tau * C::real(flux.sqrt());
        if p == 1 && value(cs.norm2()) > 0.0 {
            let ratio = (c0 / ni) / (cs / ns);
            a[p + 2] = a[p + 2] * ratio / C::real(ratio.norm());
        }
        e[p] = tau;
        h[p] = es[p] * tau;
        if s.mirror != 0 {
            absorption[p] = (e[p] * h[p].conj()).re / e0[p].re;
            a[p + 2] = C::default();
        }
    }
    if ls.iter().any(|l| value(l.k) > 0.0) {
        for &l in ls.iter().rev() {
            let term = LayerTerms::new(l, inv, s.wavelength);
            for p in 0..2 {
                let before = (e[p] * h[p].conj()).re;
                let lm = term.matrix(p == 1);
                let ee = lm[0] * e[p] + lm[1] * h[p];
                let hh = lm[2] * e[p] + lm[3] * h[p];
                e[p] = ee;
                h[p] = hh;
                if value(l.k) > 0.0 {
                    absorption[p] = absorption[p] + ((e[p] * h[p].conj()).re - before) / e0[p].re;
                }
            }
        }
    }
    (a, absorption.map(positive), true)
}
pub fn evaluate(s: CoatingSpec, ls: &[CoatingLayer], ci: D) -> CoatingValue {
    let mut out = CoatingValue::zeroed();
    let mu = s.n_in / s.n_out;
    let critical = mu * mu * (d(1.0) - ci) * (d(1.0) + ci) - d(1.0);
    if s.mirror == 0 && s.kind != MKL_RAD_IDEAL && value(critical).abs() < 1e-8 {
        out.status = MKL_RAD_REFERENCE;
        return out;
    }
    let mut a = [C::default(); 4];
    let mut absorption = [D::ZERO; 2];
    if s.kind == MKL_RAD_FILM {
        if value(ci) > 0.0 && value(ci) < 1e-4 {
            out.status = MKL_RAD_REFERENCE;
            return out;
        }
        let (aa, ab, ok) = film(s, ls, ci);
        a = aa;
        absorption = ab;
        if !ok {
            out.status = MKL_RAD_REFERENCE;
            return out;
        }
    } else if s.kind == MKL_RAD_IDEAL || s.mirror != 0 {
        let r = if s.kind == MKL_RAD_BARE {
            d(1.0)
        } else {
            s.reflectance
        };
        let t = if s.mirror != 0 {
            D::ZERO
        } else {
            s.transmittance
        };
        a = [
            C::real(-r.sqrt()),
            C::real(r.sqrt()),
            C::real(t.sqrt()),
            C::real(t.sqrt()),
        ];
        absorption = [positive(d(1.0) - r - t); 2];
    } else {
        let mu = s.n_in / s.n_out;
        let sin2 = (d(1.0) - ci) * (d(1.0) + ci);
        let st = mu * mu * sin2;
        let x = s.n_in * ci;
        if value(st - d(1.0)) >= 0.0 {
            let g = positive(s.n_in * s.n_in * sin2 - s.n_out * s.n_out).sqrt();
            let b = s.n_out * ci;
            let h = g * mu;
            a[0] = C::new(x, g) / C::new(x, -g);
            a[1] = C::new(b, h) / C::new(b, -h);
        } else {
            let ct = positive(d(1.0) - st).sqrt();
            let b = s.n_out * ct;
            let c = s.n_out * ci;
            let dd = s.n_in * ct;
            let ss = x + b;
            let sp = c + dd;
            a[0] = C::real((s.n_in - s.n_out) * (s.n_in + s.n_out) / (ss * ss));
            let si = sin2.sqrt();
            a[1] = C::real(
                (s.n_out - s.n_in) * (s.n_out + s.n_in) * (c - s.n_in * si) * (c + s.n_in * si)
                    / (s.n_out * s.n_out)
                    / (sp * sp),
            );
            if value(x) > 0.0 {
                let f = (b / x).sqrt();
                a[2] = C::real(d(2.0) * x / ss * f);
                a[3] = C::real(d(2.0) * x / sp * f);
            }
            if s.kind == MKL_RAD_TABLE && value(x) > 0.0 {
                let r = (s.n_in - s.n_out) / (s.n_in + s.n_out);
                let r0 = r * r;
                let scale = if value(r0) > 0.0 {
                    s.reflectance / r0
                } else {
                    D::ZERO
                };
                for p in 0..2 {
                    let rr = min(d(1.0), a[p].norm2() * scale);
                    a[p] = scaled(a[p], rr);
                    a[p + 2] = scaled(a[p + 2], d(1.0) - rr);
                }
            }
        }
    }
    for (i, a) in a.iter().enumerate() {
        out.amplitudes[2 * i] = a.re;
        out.amplitudes[2 * i + 1] = a.im;
        out.powers[i] = a.norm2();
    }
    out.powers[4] = absorption[0];
    out.powers[5] = absorption[1];
    if out
        .amplitudes
        .iter()
        .chain(&out.powers)
        .any(|v| !v.is_finite())
    {
        out = CoatingValue::zeroed();
        out.status = MKL_RAD_REFERENCE;
    }
    out
}
