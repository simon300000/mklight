//! Material algebra compiled once from the authoritative library.
use super::{algebra::E, compile::Compiler};
use mklight_core::{
    MkError, MkResult, SurfaceId,
    materials::{Dispersion, ResolvedMaterial, lines},
    model::MaterialRef,
    param::ParamPath,
};

pub(crate) fn air(l: E, t: E, p: E) -> E {
    let l2 = l.clone() * l;
    let r = (l2.c(6432.8)
        + l2.clone() * 2949810.0 / (l2.clone() * 146.0 - 1.0)
        + l2.clone() * 25540.0 / (l2 * 41.0 - 1.0))
        * 1e-8;
    r.clone().c(1.0) + r * p / ((t - 15.0) * 3.4785e-3 + 1.0)
}
fn horner(c: &[f64], x: E) -> E {
    c.iter().rev().fold(x.c(0.0), |a, &v| (a + v) * x.clone())
}
pub(crate) fn dispersion(d: &Dispersion, l: E) -> MkResult<E> {
    // Constant wavelengths are immutable numeric material tables, resolved once.
    if let Some(v) = l.value() {
        return Ok(l.c(d.index(v)));
    }
    let l2 = l.clone() * l.clone();
    let inv = l.c(1.0) / l2.clone();
    let resonance = |b: f64, c: f64| l2.clone() * b / (l2.clone() - c);
    Ok(match d {
        Dispersion::Sellmeier { b, c } => b
            .iter()
            .zip(c)
            .fold(l.c(1.0), |s, (&b, &c)| s + resonance(b, c))
            .sqrt(),
        Dispersion::Schott { a } => (l.c(a[0]) + l2 * a[1] + horner(&a[2..], inv)).sqrt(),
        Dispersion::Extended { a } => (l.c(a[0]) + l2 * a[1] + horner(&a[2..], inv)).sqrt(),
        Dispersion::Extended2 { a } => (l.c(a[0])
            + l2.clone() * a[1]
            + horner(&a[2..6], inv)
            + l2.clone() * l2.clone() * (l.c(a[6]) + l2 * a[7]))
            .sqrt(),
        Dispersion::Extended3 { a } => {
            (l.c(a[0]) + horner(&a[1..3], l2) + horner(&a[3..], inv)).sqrt()
        }
        Dispersion::Sellmeier2 { a, b, lambda_um: w } => (l.c(1.0 + *a)
            + l2 * b[0] / ((l.clone() - w[0]) * (l.clone() + w[0]))
            + l.c(b[1]) / ((l.clone() - w[1]) * (l.clone() + w[1])))
            .sqrt(),
        Dispersion::Sellmeier4 { a, b, c, d, e } => {
            (l.c(*a) + resonance(*b, *c) + resonance(*d, *e)).sqrt()
        }
        Dispersion::Handbook1 { a, b, c, d } => {
            (l.c(*a) + l.c(*b) / (l2.clone() - *c) - l2 * *d).sqrt()
        }
        Dispersion::Handbook2 { a, b, c, d } => (l.c(*a) + resonance(*b, *c) - l2 * *d).sqrt(),
        Dispersion::Herzberger { a } => {
            l.c(a[0]) + horner(&a[1..3], l.c(1.0) / (l2.clone() - 0.028)) + horner(&a[3..], l2)
        }
        Dispersion::Conrady { n0, a, b } => {
            l.c(*n0) + l.c(*a) / l.clone() + l.c(*b) / (l2 * l.clone() * l.sqrt())
        }
        Dispersion::Cauchy { a, b, c } => l.c(*a) + inv.clone() * (l.c(*b) + inv * *c),
        Dispersion::Tabulated { .. } => {
            return Err(MkError::Unsupported(
                "P4: variable-wavelength monotone cubic material table".into(),
            ));
        }
    })
}
fn glass_index(g: &mklight_core::materials::Glass, l: E, t: E, p: E) -> MkResult<E> {
    let n = dispersion(&g.dispersion, l.clone())?;
    let tr = g.thermal.map_or(20.0, |x| x.t_ref_c);
    if t.value() == Some(tr) && p.value() == Some(1.0) {
        return Ok(n);
    }
    let abs = n * air(l.clone(), l.c(tr), l.c(1.0));
    let dn = if let Some(th) = g.thermal {
        let dt = t.clone() - tr;
        let dt2 = dt.clone() * dt.clone();
        let den = l.clone() * l.clone() - th.lambda_tk_um * th.lambda_tk_um;
        let et = den.clone().abs() - 1e-12;
        let et = et.select(l.c(0.0), (dt.clone() * th.e0 + dt2.clone() * th.e1) / den);
        (abs.clone() * abs.clone() - 1.0) / (abs.clone() * 2.0)
            * (dt.clone() * th.d0 + dt2.clone() * th.d1 + dt2 * dt * th.d2 + et)
    } else {
        l.c(0.0)
    };
    Ok((abs + dn) / air(l, t, p))
}
impl Compiler {
    pub(super) fn material(&mut self, id: SurfaceId, m: &MaterialRef, l: E) -> MkResult<(E, E)> {
        let t = self.param(&ParamPath::TemperatureC)?;
        let p = self.param(&ParamPath::PressureAtm)?;
        let (n, alpha) = match m {
            MaterialRef::Air => (l.c(1.0), l.c(0.0)),
            MaterialRef::Vacuum => (l.c(1.0) / air(l.clone(), t, p), l.c(0.0)),
            MaterialRef::Constant { n } => (l.c(*n), l.c(0.0)),
            MaterialRef::Model { nd, vd, dpgf } => {
                let nd = if self.choice_active {
                    l.c(*nd)
                } else {
                    self.param(&ParamPath::ModelNd { surface: id })?
                };
                let vd = if self.choice_active {
                    l.c(*vd)
                } else {
                    self.param(&ParamPath::ModelVd { surface: id })?
                };
                let pg = if self.choice_active {
                    l.c(*dpgf)
                } else {
                    self.param(&ParamPath::ModelDpgf { surface: id })?
                };
                let pg = pg + 0.6438 - vd.clone() * 0.001682;
                self.positive(nd.clone() - 1.0);
                self.positive(vd.clone());
                let dfc = (nd.clone() - 1.0) / vd;
                let u = |x: f64| 1.0 / (x * x);
                let w = |x: f64| u(x) * u(x);
                let (a11, a12, a21, a22) = (
                    u(lines::F) - u(lines::C),
                    w(lines::F) - w(lines::C),
                    u(lines::G) - u(lines::F),
                    w(lines::G) - w(lines::F),
                );
                let det = a11 * a22 - a12 * a21;
                let b = (dfc.clone() * a22 - pg.clone() * dfc.clone() * a12) / det;
                let c = (pg * dfc.clone() * a11 - dfc * a21) / det;
                let a = nd - b.clone() * u(lines::D) - c.clone() * w(lines::D);
                let il2 = l.c(1.0) / (l.clone() * l.clone());
                (a + b * il2.clone() + c * il2.clone() * il2, l.c(0.0))
            }
            MaterialRef::Catalog {
                catalog,
                name,
                offset,
            } => {
                let nominal = MaterialRef::Catalog {
                    catalog: catalog.clone(),
                    name: name.clone(),
                    offset: Default::default(),
                };
                let ResolvedMaterial::Glass { glass, .. } =
                    self.library.resolve(&nominal, &self.model.environment)?
                else {
                    unreachable!()
                };
                let n = glass_index(&glass, l.clone(), t.clone(), p.clone())?;
                let dn = if self.choice_active {
                    l.c(offset.dn_d)
                } else {
                    self.param(&ParamPath::MaterialDnD { surface: id })?
                };
                let dv = if self.choice_active {
                    l.c(offset.dv_d)
                } else {
                    self.param(&ParamPath::MaterialDvD { surface: id })?
                };
                self.nonnegative(l.c(0.05) - dn.clone().abs());
                self.nonnegative(l.c(10.0) - dv.clone().abs());
                let n = if dn.value() == Some(0.0) && dv.value() == Some(0.0) {
                    n
                } else {
                    let nd = glass_index(&glass, l.c(lines::D), t.clone(), p.clone())?;
                    let nf = glass_index(&glass, l.c(lines::F), t.clone(), p.clone())?;
                    let nc = glass_index(&glass, l.c(lines::C), t, p)?;
                    let vd = (nd.clone() - 1.0) / (nf - nc);
                    self.positive(nd.clone() + dn.clone() - 1.0);
                    self.positive(vd.clone() + dv.clone());
                    let scale = (nd.clone() + dn.clone() - 1.0) / (nd.clone() - 1.0) * vd.clone()
                        / (vd + dv);
                    nd.clone() + dn + scale * (n - nd)
                };
                let points = &glass.transmittance;
                let alpha = if let Some(v) = l.value() {
                    l.c(glass.absorption_per_mm(v))
                } else if points.is_empty() {
                    l.c(0.0)
                } else {
                    let a = |i: usize| -(points[i].tau.max(1e-300)).ln() / points[i].thickness_mm;
                    let mut out = l.c(a(points.len() - 1));
                    for i in (0..points.len() - 1).rev() {
                        let f = (l.clone() - points[i].um) / (points[i + 1].um - points[i].um);
                        let y = l.c(a(i)) * (l.c(1.0) - f.clone()) + f * a(i + 1);
                        out = (l.clone() - points[i + 1].um).select(y, out);
                    }
                    (l.clone() - points[0].um).select(l.c(a(0)), out)
                };
                (n, alpha)
            }
        };
        self.positive(n.clone());
        Ok((n, alpha))
    }
}
