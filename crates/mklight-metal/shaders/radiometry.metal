#include "mklight_common.h"

// Double-float scalar and complex operators; every operation has a Rust twin
// in radiometry/{math,coating}.rs. No implicit fma contraction.
struct RD {
  float2 v;
  RD() : v(0) {}
  RD(float x) : v(x, 0) {}
  RD(float2 x) : v(mkl_df_input(x)) {}
};
static RD operator+(RD a, RD b) { return RD(mkl_df_add(a.v, b.v)); }
static RD operator-(RD a, RD b) { return RD(mkl_df_sub(a.v, b.v)); }
static RD operator-(RD a) { return RD(-a.v); }
static RD operator*(RD a, RD b) { return RD(mkl_df_mul(a.v, b.v)); }
static RD operator/(RD a, RD b) { return RD(mkl_df_div(a.v, b.v)); }
static float rv(RD a) { return a.v.x + a.v.y; }
static RD rsqrt(RD a) { return RD(mkl_df_sqrt(a.v)); }
static RD rpos(RD a) { return rv(a) > 0 ? a : RD(); }
static RD rmin(RD a, RD b) { return rv(a - b) < 0 ? a : b; }
static RD rexp(RD x) {
  RD ln2(float2(0.6931472f, -1.9046542e-9f));
  float k = round(rv(x / ln2));
  RD r = x - ln2 * RD(k), t(1), s(1);
  for (uint i = 1; i <= 14; i++) {
    t = t * r / RD(float(i));
    s = s + t;
  }
  return s * RD(ldexp(1.0f, int(k)));
}
static void rsincos(RD x, thread RD &s, thread RD &c) {
  RD tau(float2(6.2831855f, -1.7484555e-7f));
  RD r = x - tau * RD(round(rv(x / tau))), rr = -(r * r), st = r, ct(1);
  s = st;
  c = ct;
  for (uint i = 1; i <= 13; i++) {
    st = st * rr / RD(float(2 * i * (2 * i + 1)));
    s = s + st;
    ct = ct * rr / RD(float((2 * i - 1) * 2 * i));
    c = c + ct;
  }
}
struct RC {
  RD re, im;
  RC() : re(), im() {}
  RC(RD r) : re(r), im() {}
  RC(RD r, RD i) : re(r), im(i) {}
};
static RC operator+(RC a, RC b) { return RC(a.re + b.re, a.im + b.im); }
static RC operator-(RC a, RC b) { return RC(a.re - b.re, a.im - b.im); }
static RC operator-(RC a) { return RC(-a.re, -a.im); }
static RC operator*(RC a, RC b) {
  return RC(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
static RC rconj(RC a) { return RC(a.re, -a.im); }
static RD rnorm2(RC a) { return a.re * a.re + a.im * a.im; }
static RD rnorm(RC a) { return rsqrt(rnorm2(a)); }
static RC operator/(RC a, RC b) {
  RD q = rnorm2(b);
  RC z = a * rconj(b);
  return RC(z.re / q, z.im / q);
}
static RC rcsqrt(RC a) {
  RD r = rnorm(a);
  if (rv(r) == 0)
    return RC();
  RD t = rsqrt((r + (rv(a.re) >= 0 ? a.re : -a.re)) * RD(0.5f));
  if (rv(a.re) >= 0)
    return RC(t, a.im / (RD(2) * t));
  RD v = rv(a.im) < 0 ? -t : t;
  return RC(a.im / (RD(2) * v), v);
}
static void rcsincos(RC a, thread RC &sn, thread RC &cs) {
  RD s, c;
  rsincos(a.re, s, c);
  RD e = rexp(a.im), inv = RD(1) / e, ch = (e + inv) * RD(0.5f),
     sh = (e - inv) * RD(0.5f);
  sn = RC(s * ch, c * sh);
  cs = RC(c * ch, -s * sh);
}
static RC rnc(RC n, RD inv) {
  RC z = rcsqrt(n * n - RC(inv * inv));
  if (rv(z.im) > 0 || (rv(z.im) == 0 && rv(z.re) < 0))
    z = -z;
  return z;
}
struct RM {
  RC a, b, c, d;
};
static RM rid() { return {RC(RD(1)), RC(), RC(), RC(RD(1))}; }
static RM rprod(RM a, RM b) {
  return {a.a * b.a + a.b * b.c, a.a * b.b + a.b * b.d, a.c * b.a + a.d * b.c,
          a.c * b.b + a.d * b.d};
}
static RC radm(RC n, RC c, bool p) { return p ? n * n / c : c; }
struct RLayer {
  RC n, c, cs, isn, phase;
};
static RLayer rlayer(MklCoatingLayer l, RD inv, RD wavelength) {
  RLayer t;
  t.n = RC(RD(l.n), -RD(l.k));
  t.c = rnc(t.n, inv);
  t.phase = t.c * RC(RD(float2(6.2831855f, -1.7484555e-7f)) * RD(l.thickness) /
                     wavelength);
  RC sn;
  rcsincos(t.phase, sn, t.cs);
  t.isn = RC(-sn.im, sn.re);
  return t;
}
static RM rlayer_matrix(RLayer t, bool p) {
  RC eta = radm(t.n, t.c, p);
  return {t.cs, t.isn / eta, t.isn * eta, t.cs};
}
static bool rfilm(MklCoatingSpec s, device const MklCoatingLayer *ls, RD ci,
                  thread RC *a, thread RD *absorption) {
  RC ni{RD(s.n_in)}, ns{RD(s.substrate_n), -RD(s.substrate_k)};
  RD inv = RD(s.n_in) * rsqrt(rpos((RD(1) - ci) * (RD(1) + ci)));
  RC c0{RD(s.n_in) * ci}, cs = rnc(ns, inv);
  if (rv(ci) == 0) {
    a[0] = a[1] = RC(RD(-1));
    return true;
  }
  RC e0[2] = {radm(ni, c0, false), radm(ni, c0, true)}, es[2];
  if (rv(rnorm2(cs)) != 0) {
    es[0] = radm(ns, cs, false);
    es[1] = radm(ns, cs, true);
  }
  RM m[2] = {rid(), rid()};
  bool absorbing = false;
  for (uint j = 0; j < s.count; j++) {
    MklCoatingLayer l = ls[s.offset + j];
    RLayer t = rlayer(l, inv, RD(s.wavelength));
    if (abs(rv(t.phase.im)) > 30 || abs(rv(t.phase.re)) > 1e6f)
      return false;
    for (uint p = 0; p < 2; p++)
      m[p] = rprod(m[p], rlayer_matrix(t, p == 1));
    absorbing = absorbing || rv(RD(l.k)) > 0;
  }
  RC e[2], h[2];
  for (uint p = 0; p < 2; p++) {
    RC b = m[p].a + m[p].b * es[p], c = m[p].c + m[p].d * es[p],
       den = e0[p] * b + c, rho = (e0[p] * b - c) / den,
       tau = RC(RD(2)) * e0[p] / den;
    a[p] = p == 1 ? -rho : rho;
    RD flux = rpos(e0[p].re * es[p].re / rnorm2(e0[p]));
    a[p + 2] = tau * RC(rsqrt(flux));
    if (p == 1 && rv(rnorm2(cs)) > 0) {
      RC ratio = (c0 / ni) / (cs / ns);
      a[p + 2] = a[p + 2] * ratio / RC(rnorm(ratio));
    }
    e[p] = tau;
    h[p] = es[p] * tau;
    if (s.mirror) {
      absorption[p] = (e[p] * rconj(h[p])).re / e0[p].re;
      a[p + 2] = RC();
    }
  }
  if (absorbing)
    for (uint j = s.count; j > 0; j--) {
      MklCoatingLayer l = ls[s.offset + j - 1];
      RLayer t = rlayer(l, inv, RD(s.wavelength));
      for (uint p = 0; p < 2; p++) {
        RD before = (e[p] * rconj(h[p])).re;
        RM lm = rlayer_matrix(t, p == 1);
        RC ee = lm.a * e[p] + lm.b * h[p], hh = lm.c * e[p] + lm.d * h[p];
        e[p] = ee;
        h[p] = hh;
        if (rv(RD(l.k)) > 0)
          absorption[p] =
              absorption[p] + ((e[p] * rconj(h[p])).re - before) / e0[p].re;
      }
    }
  absorption[0] = rpos(absorption[0]);
  absorption[1] = rpos(absorption[1]);
  return true;
}
static RC rscaled(RC a, RD p) {
  RD n = rnorm(a);
  return rv(n) > 0 ? a * RC(rsqrt(rpos(p)) / n) : RC(rsqrt(rpos(p)));
}
static MklCoatingValue revaluate(MklCoatingSpec s,
                                 device const MklCoatingLayer *ls, RD ci) {
  MklCoatingValue out = {};
  RD mu = RD(s.n_in) / RD(s.n_out),
     critical = mu * mu * (RD(1) - ci) * (RD(1) + ci) - RD(1);
  if (!s.mirror && s.kind != MKL_RAD_IDEAL && abs(rv(critical)) < 1e-8f) {
    out.status = MKL_RAD_REFERENCE;
    return out;
  }
  RC a[4];
  RD absorption[2];
  if (s.kind == MKL_RAD_FILM) {
    if (rv(ci) > 0 && rv(ci) < 1e-4f) {
      out.status = MKL_RAD_REFERENCE;
      return out;
    }
    if (!rfilm(s, ls, ci, a, absorption)) {
      out.status = MKL_RAD_REFERENCE;
      return out;
    }
  } else if (s.kind == MKL_RAD_IDEAL || s.mirror) {
    RD r = s.kind == MKL_RAD_BARE ? RD(1) : RD(s.reflectance),
       t = s.mirror ? RD() : RD(s.transmittance);
    a[0] = RC(-rsqrt(r));
    a[1] = RC(rsqrt(r));
    a[2] = a[3] = RC(rsqrt(t));
    absorption[0] = absorption[1] = rpos(RD(1) - r - t);
  } else {
    RD n1(s.n_in), n2(s.n_out),
        mu = n1 / n2, sin2 = (RD(1) - ci) * (RD(1) + ci), st = mu * mu * sin2,
        x = n1 * ci;
    if (rv(st - RD(1)) >= 0) {
      RD g = rsqrt(rpos(n1 * n1 * sin2 - n2 * n2)), b = n2 * ci, h = g * mu;
      a[0] = RC(x, g) / RC(x, -g);
      a[1] = RC(b, h) / RC(b, -h);
    } else {
      RD ct = rsqrt(rpos(RD(1) - st)), b = n2 * ct, c = n2 * ci, dd = n1 * ct,
         ss = x + b, sp = c + dd;
      a[0] = RC((n1 - n2) * (n1 + n2) / (ss * ss));
      RD si = rsqrt(sin2);
      a[1] = RC((n2 - n1) * (n2 + n1) * (c - n1 * si) * (c + n1 * si) /
                (n2 * n2) / (sp * sp));
      if (rv(x) > 0) {
        RD f = rsqrt(b / x);
        a[2] = RC(RD(2) * x / ss * f);
        a[3] = RC(RD(2) * x / sp * f);
      }
      if (s.kind == MKL_RAD_TABLE && rv(x) > 0) {
        RD r = (n1 - n2) / (n1 + n2), r0 = r * r,
           scale = rv(r0) > 0 ? RD(s.reflectance) / r0 : RD();
        for (uint p = 0; p < 2; p++) {
          RD rr = rmin(RD(1), rnorm2(a[p]) * scale);
          a[p] = rscaled(a[p], rr);
          a[p + 2] = rscaled(a[p + 2], RD(1) - rr);
        }
      }
    }
  }
  for (uint i = 0; i < 4; i++) {
    out.amplitudes[2 * i] = a[i].re.v;
    out.amplitudes[2 * i + 1] = a[i].im.v;
    out.powers[i] = rnorm2(a[i]).v;
  }
  out.powers[4] = absorption[0].v;
  out.powers[5] = absorption[1].v;
  for (uint i = 0; i < 8; i++)
    if (!all(isfinite(out.amplitudes[i]))) {
      out = {};
      out.status = MKL_RAD_REFERENCE;
      return out;
    }
  for (uint i = 0; i < 6; i++)
    if (!all(isfinite(out.powers[i]))) {
      out = {};
      out.status = MKL_RAD_REFERENCE;
      return out;
    }
  return out;
}
// Everything above is the coating evaluation that ghost_ops.metal and
// stray_ops.metal include into their own namespaces; they define
// MKL_RADIOMETRY_COATING_ONLY so that the kernels below exist only once.
#ifndef MKL_RADIOMETRY_COATING_ONLY
kernel void
mkl_radiometry_coating(device const MklCoatingSpec *specs [[buffer(0)]],
                       device const MklCoatingLayer *layers [[buffer(1)]],
                       device const MklCoatingQuery *queries [[buffer(2)]],
                       device MklCoatingValue *output [[buffer(3)]],
                       constant MklRadParams &p [[buffer(4)]],
                       uint tid [[thread_position_in_grid]]) {
  uint i = p.offset + tid;
  if (i >= p.end)
    return;
  MklCoatingQuery q = queries[i];
  if (q.coating >= p.specs) {
    output[i] = {};
    output[i].status = MKL_RAD_REFERENCE;
    return;
  }
  MklCoatingSpec s = specs[q.coating];
  if (s.offset > p.layers || s.count > p.layers - s.offset ||
      s.count > MKL_RAD_MAX_LAYERS) {
    output[i] = {};
    output[i].status = MKL_RAD_REFERENCE;
    return;
  }
  output[i] = revaluate(s, layers, RD(q.cosine));
}

struct RV {
  RD a[3];
};
struct RE {
  RC a[3];
};
static RD rdot(RV a, RV b) {
  return a.a[0] * b.a[0] + a.a[1] * b.a[1] + a.a[2] * b.a[2];
}
static RV rcross(RV a, RV b) {
  return {{a.a[1] * b.a[2] - a.a[2] * b.a[1], a.a[2] * b.a[0] - a.a[0] * b.a[2],
           a.a[0] * b.a[1] - a.a[1] * b.a[0]}};
}
static RV rscale(RV a, RD s) {
  for (uint i = 0; i < 3; i++)
    a.a[i] = a.a[i] * s;
  return a;
}
static RV rsub(RV a, RV b) {
  for (uint i = 0; i < 3; i++)
    a.a[i] = a.a[i] - b.a[i];
  return a;
}
static RV runit(RV a) { return rscale(a, RD(1) / rsqrt(rdot(a, a))); }
static RV rrot(thread const float2 *m, RV v, bool transpose = false) {
  RV out;
  for (uint i = 0; i < 3; i++) {
    RV row;
    for (uint j = 0; j < 3; j++)
      row.a[j] = RD(m[transpose ? j * 3 + i : i * 3 + j]);
    out.a[i] = rdot(row, v);
  }
  return out;
}
static RC rdotc(RV a, RE b) {
  return RC(a.a[0]) * b.a[0] + RC(a.a[1]) * b.a[1] + RC(a.a[2]) * b.a[2];
}
static RV rperp(RV k) {
  float x = abs(rv(k.a[0])), y = abs(rv(k.a[1])), z = abs(rv(k.a[2]));
  uint i = x <= y && x <= z ? 0 : y <= z ? 1 : 2;
  RV axis;
  axis.a[i] = RD(1);
  return runit(rcross(k, axis));
}
static RV rsv(RV k, RV n) {
  RV c = rcross(k, n);
  c = rsub(c, rscale(k, rdot(c, k)));
  return rv(rdot(c, c)) > 1e-24f ? runit(c) : rperp(k);
}
static RV rortho(RV s, RV k) {
  RV a = rsub(s, rscale(k, rdot(s, k)));
  return rv(rdot(a, a)) > 0.25f ? runit(a) : rperp(k);
}
static void rbasis(RV k, thread const float2 *r, thread RV *out) {
  RV v = rrot(r, k, true);
  RD sg(rv(v.a[2]) < 0 ? -1 : 1), x = v.a[0], y = sg * v.a[1], z = sg * v.a[2],
                                  den = RD(1) + z;
  out[0] = rrot(r, {{RD(1) - x * x / den, sg * (-x * y / den), sg * (-x)}});
  out[1] = rrot(r, {{-x * y / den, sg * (RD(1) - y * y / den), sg * (-y)}});
}
static RE rapply(RE e, RV ki, RV ko, RV s, RC as_, RC ap) {
  RV pi = rcross(ki, s), so = rortho(s, ko), po = rcross(ko, so);
  RC es = rdotc(s, e) * as_, ep = rdotc(pi, e) * ap, ek = rdotc(ki, e);
  RE out;
  for (uint i = 0; i < 3; i++)
    out.a[i] = RC(so.a[i]) * es + RC(po.a[i]) * ep + RC(ko.a[i]) * ek;
  return out;
}
static RD rintensity(RC a, RC b, constant MklPolarizationParams &p) {
  RD aa = rnorm2(a), bb = rnorm2(b);
  RC c = rconj(a) * b;
  return ((aa + bb) * RD(0.5f) * RD(p.stokes[0]) +
          (aa - bb) * RD(0.5f) * RD(p.stokes[1]) + c.re * RD(p.stokes[2]) -
          c.im * RD(p.stokes[3])) /
         RD(p.stokes[0]);
}
static RD renergy(thread RE *e, constant MklPolarizationParams &p) {
  RD s;
  for (uint i = 0; i < 3; i++)
    s = s + rintensity(e[0].a[i], e[1].a[i], p);
  return s;
}
static void rproperties(RM j, thread float2 *out, thread bool &defined,
                        thread bool &fallback) {
  float scale = max(abs(rv(j.a.re)), abs(rv(j.a.im)));
  scale = max(scale, max(abs(rv(j.b.re)), abs(rv(j.b.im))));
  scale = max(scale, max(abs(rv(j.c.re)), abs(rv(j.c.im))));
  scale = max(scale, max(abs(rv(j.d.re)), abs(rv(j.d.im))));
  if (scale < 0x1p-126f) {
    for (uint i = 0; i < 8; i++)
      out[i] = float2(0);
    defined = false;
    fallback = true;
    return;
  }
  RD scale_df(scale);
  j = {RC(j.a.re / scale_df, j.a.im / scale_df),
       RC(j.b.re / scale_df, j.b.im / scale_df),
       RC(j.c.re / scale_df, j.c.im / scale_df),
       RC(j.d.re / scale_df, j.d.im / scale_df)};

  RD aa = rnorm2(j.a) + rnorm2(j.c), bb = rnorm2(j.b) + rnorm2(j.d);
  RC b = rconj(j.a) * j.b + rconj(j.c) * j.d;
  RD t = aa + bb;
  RV m{{(aa - bb) * RD(0.5f), b.re, -b.im}};
  RD mn = rsqrt(rdot(m, m)), top = t * RD(0.5f) + mn,
     det = rnorm(j.a * j.d - j.b * j.c);
  defined = false;
  fallback = false;
  for (uint i = 0; i < 8; i++)
    out[i] = float2(0);
  if (rv(t) > 0)
    out[0] = rmin(RD(1), RD(2) * mn / t).v;
  if (rv(mn) > 0)
    for (uint i = 0; i < 3; i++)
      out[2 + i] = (m.a[i] / mn).v;
  if (rv(top) == 0)
    return;
  RD contrast = det * det / (top * top);
  if (rv(contrast) < 1e-18f) {
    fallback = true;
    return;
  }
  RD hs = RD(1) / rsqrt(t + RD(2) * det), h00 = (aa + det) * hs,
     h11 = (bb + det) * hs;
  RC h01 = b * RC(hs);
  RM u = rprod(j, {RC(h11), -h01, -rconj(h01), RC(h00)});
  u = {u.a / RC(det), u.b / RC(det), u.c / RC(det), u.d / RC(det)};
  RC phase = rcsqrt(u.a * u.d - u.b * u.c);
  phase = phase / RC(rnorm(phase));
  u = {u.a / phase, u.b / phase, u.c / phase, u.d / phase};
  RD c = (u.a + u.d).re * RD(0.5f);
  RV v{{-(u.a - u.d).im * RD(0.5f), -(u.b + u.c).im * RD(0.5f),
        (u.c - u.b).re * RD(0.5f)}};
  if (rv(c) < 0) {
    c = -c;
    v = rscale(v, RD(-1));
  }
  RD vn = rsqrt(rdot(v, v));
  out[1] = float2(2 * atan2(rv(vn), rv(c)), 0);
  if (rv(vn) > 0)
    for (uint i = 0; i < 3; i++)
      out[5 + i] = (-v.a[i] / vn).v;
  defined = true;
}
static RV rrecord(device const float *hi, device const float *lo, uint slot,
                  uint n, uint i, uint k) {
  RV v;
  for (uint a = 0; a < 3; a++) {
    uint ix = (slot * n + i) * 13 + k + a;
    v.a[a] = RD(float2(hi[ix], lo[ix]));
  }
  return v;
}
kernel void
mkl_radiometry_polarization(device const MklRadSurface *surfaces [[buffer(0)]],
                            device const MklCoatingLayer *layers [[buffer(1)]],
                            device const float *hi [[buffer(2)]],
                            device const float *lo [[buffer(3)]],
                            device const uint *indices [[buffer(4)]],
                            device const uint *statuses [[buffer(5)]],
                            device MklPolarizationValue *output [[buffer(6)]],
                            constant MklPolarizationParams &p [[buffer(7)]],
                            uint tid [[thread_position_in_grid]]) {
  uint i = p.offset + tid, n = p.count;
  if (i >= p.end)
    return;
  MklPolarizationValue out = {};
  if (statuses[i] != 0) {
    out.status = MKL_RAD_BLOCKED;
    output[i] = out;
    return;
  }
  uint wl = indices[n + i];
  if (wl >= p.wavelengths) {
    out.status = MKL_RAD_REFERENCE;
    output[i] = out;
    return;
  }
  MklRadSurface first = surfaces[wl];
  RV ki = runit(rrot(first.rotation, rrecord(hi, lo, 0, n, i, 3))), bi[2];
  rbasis(ki, first.rotation, bi);
  RE e[2], g[2];
  for (uint a = 0; a < 2; a++)
    for (uint k = 0; k < 3; k++) {
      e[a].a[k] = RC(bi[a].a[k]);
      g[a].a[k] = e[a].a[k];
    }
  RD scalar(1), sp[2] = {RD(1), RD(1)}, loss[3];
  for (uint j = 1; j < p.slots; j++) {
    MklRadSurface s = surfaces[j * p.wavelengths + wl];
    if (s.coating.offset > p.layers ||
        s.coating.count > p.layers - s.coating.offset ||
        s.coating.count > MKL_RAD_MAX_LAYERS) {
      out.status = MKL_RAD_REFERENCE;
      output[i] = out;
      return;
    }
    RV ko = runit(rrot(s.rotation, rrecord(hi, lo, j, n, i, 3))),
       normal = runit(rrot(s.rotation, rrecord(hi, lo, j, n, i, 8)));
    RV pp = rrot(s.transfer, rrecord(hi, lo, j - 1, n, i, 0));
    for (uint a = 0; a < 3; a++)
      pp.a[a] = pp.a[a] + RD(s.translation[a]);
    RV delta = rsub(rrecord(hi, lo, j, n, i, 0), pp);
    RD tau = rv(RD(s.alpha)) > 0
                 ? rexp(-RD(s.alpha) *
                        rdot(delta,
                             rrot(s.transfer, rrecord(hi, lo, j - 1, n, i, 3))))
                 : RD(1);
    loss[2] = loss[2] + renergy(e, p) * (RD(1) - tau);
    RC at(rsqrt(tau));
    for (uint a = 0; a < 2; a++)
      for (uint k = 0; k < 3; k++)
        e[a].a[k] = e[a].a[k] * at;
    scalar = scalar * tau;
    sp[0] = sp[0] * tau;
    sp[1] = sp[1] * tau;
    if (s.kind != 0) {
      RV axis = rsv(ki, normal), mid = ko;
      RC coeff[2] = {RC(RD(1)), RC(RD(1))}, geom[2] = {RC(RD(1)), RC(RD(1))};
      RD power[2] = {RD(1), RD(1)};
      if (s.kind == 3)
        axis = rsv(ki, ko);
      else {
        RD ci = rdot(ki, normal);
        ci = rmin(RD(1), rv(ci) < 0 ? -ci : ci);
        MklCoatingValue a = revaluate(s.coating, layers, ci);
        if (a.status) {
          out.status = MKL_RAD_REFERENCE;
          output[i] = out;
          return;
        }
        bool mirror = s.kind == 2 || s.kind == 4;
        uint offset = mirror ? 0 : 2;
        for (uint pol = 0; pol < 2; pol++) {
          coeff[pol] = RC(RD(a.amplitudes[2 * (offset + pol)]),
                          RD(a.amplitudes[2 * (offset + pol) + 1]));
          power[pol] = RD(a.powers[offset + pol]);
        }
        if (mirror)
          geom[0] = RC(RD(-1));
        if (s.kind == 4)
          mid = runit(rsub(ki, rscale(normal, RD(2) * rdot(ki, normal))));
        RD ws = rintensity(rdotc(axis, e[0]), rdotc(axis, e[1]), p);
        RV pi = rcross(ki, axis);
        RD wp = rintensity(rdotc(pi, e[0]), rdotc(pi, e[1]), p);
        if (!mirror)
          loss[0] = loss[0] + ws * RD(a.powers[0]) + wp * RD(a.powers[1]);
        loss[1] = loss[1] + ws * RD(a.powers[4]) + wp * RD(a.powers[5]);
      }
      scalar = scalar * (power[0] + power[1]) * RD(0.5f);
      for (uint a = 0; a < 2; a++) {
        sp[a] = sp[a] * power[a];
        e[a] = rapply(e[a], ki, mid, axis, coeff[0], coeff[1]);
        g[a] = rapply(g[a], ki, mid, axis, geom[0], geom[1]);
      }
      if (s.kind == 4) {
        RV axis = rsv(mid, ko);
        for (uint a = 0; a < 2; a++) {
          e[a] = rapply(e[a], mid, ko, axis, RC(RD(1)), RC(RD(1)));
          g[a] = rapply(g[a], mid, ko, axis, RC(RD(1)), RC(RD(1)));
        }
      }
    }
    ki = ko;
  }
  MklRadSurface last = surfaces[(p.slots - 1) * p.wavelengths + wl];
  RV bo[2];
  rbasis(ki, last.rotation, bo);
  RC jones[4], proper[4];
  for (uint a = 0; a < 2; a++) {
    RV gg{{g[a].a[0].re, g[a].a[1].re, g[a].a[2].re}};
    for (uint b = 0; b < 2; b++) {
      jones[2 * a + b] = rdotc(bo[a], e[b]);
      proper[2 * a + b] = rdotc(gg, e[b]);
    }
  }
  for (uint k = 0; k < 4; k++) {
    out.jones[2 * k] = jones[k].re.v;
    out.jones[2 * k + 1] = jones[k].im.v;
    out.proper[2 * k] = proper[k].re.v;
    out.proper[2 * k + 1] = proper[k].im.v;
  }
  for (uint a = 0; a < 2; a++)
    for (uint k = 0; k < 3; k++) {
      RV row{{RD(last.rotation[k]), RD(last.rotation[3 + k]),
              RD(last.rotation[6 + k])}};
      RC z = rdotc(row, e[a]);
      out.fields[6 * a + 2 * k] = z.re.v;
      out.fields[6 * a + 2 * k + 1] = z.im.v;
    }
  RV dir = rrot(last.rotation, ki, true);
  for (uint k = 0; k < 3; k++)
    out.direction[k] = dir.a[k].v;
  out.power[0] = scalar.v;
  out.power[1] = ((sp[0] + sp[1]) * RD(0.5f)).v;
  out.power[2] = renergy(e, p).v;
  RD unpol;
  for (uint k = 0; k < 3; k++)
    unpol = unpol + (rnorm2(e[0].a[k]) + rnorm2(e[1].a[k])) * RD(0.5f);
  out.power[3] = unpol.v;
  for (uint k = 0; k < 3; k++)
    out.power[k + 4] = loss[k].v;
  bool defined, jdefined, ref1, ref2;
  rproperties({proper[0], proper[1], proper[2], proper[3]}, out.properties,
              defined, ref1);
  float2 jp[8];
  rproperties({jones[0], jones[1], jones[2], jones[3]}, jp, jdefined, ref2);
  out.properties[8] = jp[1];
  out.defined = uint(defined) + 2 * uint(jdefined);
  if (ref1 || ref2)
    out.status = MKL_RAD_REFERENCE;
  output[i] = out;
}
kernel void mkl_radiometry_channels(device const MklPolarizationValue *values
                                    [[buffer(0)]],
                                    device const uint *status [[buffer(1)]],
                                    device float2 *output [[buffer(2)]],
                                    constant uint2 &p [[buffer(3)]],
                                    uint tid [[thread_position_in_grid]]) {
  uint i = p.y + tid, n = p.x;
  if (i >= n)
    return;
  MklPolarizationValue v = values[i];
  bool failed = (status[i] & MKL_RAY_CODE_MASK) >= MKL_RAY_NONFINITE,
       fallback = v.status == MKL_RAD_REFERENCE,
       ok = v.status == MKL_RAD_OK && !failed, defined = ok && (v.defined & 1);
  output[i] = ok ? v.power[0] : float2(0);
  output[n + i] = ok ? v.power[2] : float2(0);
  output[2 * n + i] = ok ? v.power[3] : float2(0);
  output[3 * n + i] = ok ? v.properties[0] : float2(0);
  output[4 * n + i] =
      defined ? mkl_df_mul(v.properties[1], v.properties[1]) : float2(0);
  output[5 * n + i] = float2(float(!failed && !fallback), 0);
  output[6 * n + i] = float2(float(ok), 0);
  output[7 * n + i] = float2(float(defined), 0);
  output[8 * n + i] = float2(float(fallback), 0);
  output[9 * n + i] = float2(float(failed), 0);
}

kernel void mkl_radiometry_count(device const uint *values [[buffer(0)]],
                                 device atomic_uint *count [[buffer(1)]],
                                 constant uint4 &p [[buffer(2)]],
                                 uint tid [[thread_position_in_grid]]) {
  uint i = p.y + tid;
  if (i < p.x && values[i * p.z + p.w] == MKL_RAD_REFERENCE)
    atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
}
kernel void
mkl_radiometry_scalar(device const MklRadSurface *surfaces [[buffer(0)]],
                      device const MklCoatingLayer *layers [[buffer(1)]],
                      device const float *hi [[buffer(2)]],
                      device const float *lo [[buffer(3)]],
                      device const uint *indices [[buffer(4)]],
                      device const uint *statuses [[buffer(5)]],
                      device MklPolarizationValue *output [[buffer(6)]],
                      constant MklPolarizationParams &p [[buffer(7)]],
                      uint tid [[thread_position_in_grid]]) {
  uint i = p.offset + tid, n = p.count;
  if (i >= p.end)
    return;
  MklPolarizationValue out = {};
  if (statuses[i]) {
    out.status = MKL_RAD_BLOCKED;
    output[i] = out;
    return;
  }
  uint wl = indices[n + i];
  if (wl >= p.wavelengths) {
    out.status = MKL_RAD_REFERENCE;
    output[i] = out;
    return;
  }
  MklRadSurface first = surfaces[wl];
  RV ki = runit(rrot(first.rotation, rrecord(hi, lo, 0, n, i, 3)));
  RD product(1), sp[2] = {RD(1), RD(1)};
  for (uint j = 1; j < p.slots; j++) {
    MklRadSurface s = surfaces[j * p.wavelengths + wl];
    if (s.coating.offset > p.layers ||
        s.coating.count > p.layers - s.coating.offset ||
        s.coating.count > MKL_RAD_MAX_LAYERS) {
      out.status = MKL_RAD_REFERENCE;
      output[i] = out;
      return;
    }
    if (rv(RD(s.alpha)) > 0) {
      RV pp = rrot(s.transfer, rrecord(hi, lo, j - 1, n, i, 0));
      for (uint a = 0; a < 3; a++)
        pp.a[a] = pp.a[a] + RD(s.translation[a]);
      RV delta = rsub(rrecord(hi, lo, j, n, i, 0), pp);
      RD tau =
          rexp(-RD(s.alpha) *
               rdot(delta, rrot(s.transfer, rrecord(hi, lo, j - 1, n, i, 3))));
      product = product * tau;
      sp[0] = sp[0] * tau;
      sp[1] = sp[1] * tau;
    }
    if (s.kind != 0 && s.kind != 3) {
      RV normal = runit(rrot(s.rotation, rrecord(hi, lo, j, n, i, 8)));
      RD ci = rdot(ki, normal);
      ci = rmin(RD(1), rv(ci) < 0 ? -ci : ci);
      MklCoatingValue a = revaluate(s.coating, layers, ci);
      if (a.status) {
        out.status = a.status;
        output[i] = out;
        return;
      }
      uint offset = (s.kind == 2 || s.kind == 4) ? 0 : 2;
      RD ps(a.powers[offset]), pp(a.powers[offset + 1]);
      product = product * (ps + pp) * RD(0.5f);
      sp[0] = sp[0] * ps;
      sp[1] = sp[1] * pp;
    }
    ki = runit(rrot(s.rotation, rrecord(hi, lo, j, n, i, 3)));
  }
  out.power[0] = product.v;
  out.power[1] = ((sp[0] + sp[1]) * RD(0.5f)).v;
  output[i] = out;
}
#endif // MKL_RADIOMETRY_COATING_ONLY
