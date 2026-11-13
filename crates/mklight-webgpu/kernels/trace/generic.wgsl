fn mkl_jconst@(v: R_) -> J_ {
    return J_(v, c@(0.0), c@(0.0));
}
fn mkl_jvar@(v: R_, is_x: bool) -> J_ {
    if (is_x) {
        return J_(v, c@(1.0), c@(0.0));
    }
    return J_(v, c@(0.0), c@(1.0));
}
fn mkl_jchain@(a: J_, v: R_, d: R_) -> J_ {
    return J_(v, mul@(d, a.x), mul@(d, a.y));
}
fn mkl_jscale@(a: J_, b: R_) -> J_ {
    return J_(mul@(a.v, b), mul@(a.x, b), mul@(a.y, b));
}
fn mkl_jadd@(a: J_, b: R_) -> J_ {
    return J_(add@(a.v, b), a.x, a.y);
}
fn mkl_jsqrt@(a: J_) -> J_ {
    let v = sqrt@(a.v);
    return mkl_jchain@(a, v, div@(c@(0.5), v));
}
fn mkl_jrecip@(a: J_) -> J_ {
    let r = div@(c@(1.0), a.v);
    return mkl_jchain@(a, r, mul@(neg@(r), r));
}
fn mkl_jsum@(a: J_, b: J_) -> J_ {
    return J_(add@(a.v, b.v), add@(a.x, b.x), add@(a.y, b.y));
}
fn mkl_jneg@(a: J_) -> J_ {
    return mkl_jscale@(a, c@(-1.0));
}
fn mkl_jdiff@(a: J_, b: J_) -> J_ {
    return mkl_jsum@(a, mkl_jneg@(b));
}
fn mkl_jmul@(a: J_, b: J_) -> J_ {
    return J_(
        mul@(a.v, b.v),
        add@(mul@(a.x, b.v), mul@(a.v, b.x)),
        add@(mul@(a.y, b.v), mul@(a.v, b.y)),
    );
}
fn mkl_jdiv@(a: J_, b: J_) -> J_ {
    return mkl_jmul@(a, mkl_jrecip@(b));
}
fn mkl_le_value@(a: R_, b: R_) -> bool {
    return le@(sub@(a, b), 0.0);
}
fn mkl_sag_fail@() -> S_ {
    return S_(false, mkl_jconst@(c@(0.0)));
}
fn mkl_poly@(pool: u32, start: u32, n: u32, t: J_) -> J_ {
    var p = mkl_jconst@(c@(0.0));
    for (var j = i32(n) - 1; j >= 0; j--) {
        p = mkl_jadd@(mkl_jmul@(p, t), load@(pool, start + u32(j)));
    }
    return p;
}
fn mkl_even@(pool: u32, start: u32, n: u32, t: J_) -> J_ {
    return mkl_jmul@(mkl_jmul@(mkl_poly@(pool, start, n, t), t), t);
}
fn mkl_conic_sag_jet@(c: R_, k: R_, t: J_) -> S_ {
    let arg = sub@(c@(1.0), mul@(mul@(mul@(add@(c@(1.0), k), c), c), t.v));
    if (!mkl_le_value@(c@(0.0), arg)) {
        return mkl_sag_fail@();
    }
    let w = sqrt@(arg);
    return S_(true, mkl_jchain@(t, div@(mul@(c, t.v), add@(c@(1.0), w)), div@(c, mul@(c@(2.0), w))));
}
fn mkl_scaled_jet@(x: J_, y: J_, radius: R_) -> U_ {
    let inv = div@(c@(1.0), radius);
    let u = mkl_jscale@(x, inv);
    let v = mkl_jscale@(y, inv);
    return U_(u, v, mkl_jsum@(mkl_jmul@(u, u), mkl_jmul@(v, v)));
}
// mkl_normalized_jet's test on a scaled jet.
fn mkl_normalized@(q: U_) -> bool {
    return mkl_le_value@(q.s.v, parts@(1.0, 2e-12));
}
fn mkl_jacobi@(n: u32, b: R_, t: J_) -> J_ {
    var p0 = mkl_jconst@(c@(1.0));
    if (n == 0u) {
        return p0;
    }
    var p1 = mkl_jadd@(mkl_jscale@(mkl_jadd@(t, c@(-1.0)), div@(add@(b, c@(2.0)), c@(2.0))), c@(1.0));
    for (var ki = 1u; ki < n; ki++) {
        let k = c@(f32(ki));
        let s = add@(mul@(c@(2.0), k), b);
        let a1 = mul@(mul@(mul@(c@(2.0), add@(k, c@(1.0))), add@(add@(k, b), c@(1.0))), s);
        let a2 = mkl_jscale@(mkl_jadd@(mkl_jscale@(t, mul@(add@(s, c@(2.0)), s)), neg@(mul@(b, b))), add@(s, c@(1.0)));
        let a3 = mul@(mul@(mul@(c@(2.0), k), add@(k, b)), add@(s, c@(2.0)));
        let next = mkl_jscale@(mkl_jdiff@(mkl_jmul@(a2, p1), mkl_jscale@(p0, a3)), div@(c@(1.0), a1));
        p0 = p1;
        p1 = next;
    }
    return p1;
}
// Q-con recurrence: v = 2k+4, d = 2(k+1)(k+5)v.
fn mkl_qcon_d@(ki: u32, v: R_) -> R_ {
    let k = c@(f32(ki));
    return mul@(mul@(mul@(c@(2.0), add@(k, c@(1.0))), add@(k, c@(5.0))), v);
}
fn mkl_qcon_a@(ki: u32, s: J_) -> J_ {
    let v = add@(mul@(c@(2.0), c@(f32(ki))), c@(4.0));
    let d = mkl_qcon_d@(ki, v);
    return mkl_jscale@(
        mkl_jadd@(mkl_jscale@(mkl_jadd@(mkl_jscale@(s, c@(2.0)), c@(-1.0)), mul@(add@(v, c@(2.0)), v)), c@(-16.0)),
        div@(add@(v, c@(1.0)), d),
    );
}
fn mkl_qcon_b@(ki: u32) -> R_ {
    let k = c@(f32(ki));
    let v = add@(mul@(c@(2.0), k), c@(4.0));
    return div@(mul@(mul@(mul@(c@(2.0), k), add@(k, c@(4.0))), add@(v, c@(2.0))), mkl_qcon_d@(ki, v));
}
fn mkl_qcon@(pool: u32, n: u32, s: J_) -> J_ {
    var b1 = mkl_jconst@(c@(0.0));
    var b2 = b1;
    for (var k = i32(n) - 1; k >= 0; k--) {
        let b = mkl_jadd@(
            mkl_jdiff@(mkl_jmul@(mkl_qcon_a@(u32(k), s), b1), mkl_jscale@(b2, mkl_qcon_b@(u32(k + 1)))),
            load@(pool, u32(k)),
        );
        b2 = b1;
        b1 = b;
    }
    return mkl_jmul@(mkl_jmul@(b1, s), s);
}
fn mkl_qbfs@(pool: u32, n: u32, s: J_) -> J_ {
    var a1 = mkl_jconst@(c@(0.0));
    var a2 = a1;
    for (var m = i32(n) - 1; m >= 0; m--) {
        let a = mkl_jadd@(
            mkl_jdiff@(mkl_jmul@(mkl_jadd@(mkl_jscale@(s, c@(-4.0)), c@(2.0)), a1), a2),
            load@(pool, u32(m)),
        );
        a2 = a1;
        a1 = a;
    }
    return mkl_jscale@(mkl_jsum@(a1, a2), c@(2.0));
}
fn mkl_zernike@(pool: u32, n: u32, standard: bool, x: J_, y: J_, s: J_) -> J_ {
    var z = mkl_jconst@(c@(0.0));
    for (var j = 1u; j <= n; j++) {
        let a = load@(pool, j - 1u);
        if (zero@(a)) {
            continue;
        }
        var degree: u32;
        var m: i32;
        if (standard) {
            degree = 0u;
            while ((degree + 1u) * (degree + 2u) / 2u < j) {
                degree++;
            }
            let k = j - degree * (degree + 1u) / 2u - 1u;
            var am: i32;
            if (degree % 2u == 0u) {
                am = i32(2u * ((k + 1u) / 2u));
            } else {
                am = i32(2u * (k / 2u) + 1u);
            }
            m = select(-am, am, j % 2u == 0u);
        } else if (j == 37u) {
            degree = 12u;
            m = 0;
        } else {
            var k = 0u;
            while ((k + 1u) * (k + 1u) < j) {
                k++;
            }
            let off = j - 1u - k * k;
            let am = i32(k - off / 2u);
            degree = 2u * k - u32(am);
            m = select(am, -am, off % 2u == 1u);
        }
        let am = u32(abs(m));
        var re = mkl_jconst@(c@(1.0));
        var im = mkl_jconst@(c@(0.0));
        for (var i = 0u; i < am; i++) {
            let r = mkl_jdiff@(mkl_jmul@(re, x), mkl_jmul@(im, y));
            im = mkl_jsum@(mkl_jmul@(re, y), mkl_jmul@(im, x));
            re = r;
        }
        var scale = c@(1.0);
        if (standard) {
            scale = sqrt@(c@(f32((degree + 1u) * select(2u, 1u, m == 0))));
        }
        var part = im;
        if (m >= 0) {
            part = re;
        }
        let radial = mkl_jacobi@((degree - am) / 2u, c@(f32(am)), mkl_jadd@(mkl_jscale@(s, c@(2.0)), c@(-1.0)));
        z = mkl_jsum@(z, mkl_jscale@(mkl_jmul@(radial, part), mul@(a, scale)));
    }
    return z;
}
fn mkl_surface_domain@(s: u32, x: R_, y: R_) -> bool {
    let data = mkl_surface_data(s);
    return mkl_normalized@(mkl_scaled_jet@(mkl_jvar@(x, true), mkl_jvar@(y, false), load@(data, 0u)));
}
fn mkl_phase@(s: u32, x: R_, y: R_) -> J_ {
    let data = mkl_surface_data(s);
    let q = mkl_scaled_jet@(mkl_jvar@(x, true), mkl_jvar@(y, false), load@(data, 0u));
    let n = u32(mkl_pool[data + 2u * MKL_COEFF_NA]);
    return mkl_jmul@(mkl_poly@(data, MKL_COEFF_PREFIX + MKL_MAX_WAVELENGTHS, n, q.s), q.s);
}
fn mkl_extended_sag@(s: u32, xv: R_, yv: R_) -> S_ {
    let data = mkl_surface_data(s);
    let radius = load@(data, 0u);
    let na = u32(mkl_pool[data + 2u * MKL_COEFF_NA]);
    let nb = u32(mkl_pool[data + 2u * MKL_COEFF_NB]);
    let coeff = data + 2u * MKL_COEFF_PREFIX;
    let x = mkl_jvar@(xv, true);
    let y = mkl_jvar@(yv, false);
    let r2 = mkl_jsum@(mkl_jmul@(x, x), mkl_jmul@(y, y));
    let c = parts@(mkl_surfaces[s].curvature, mkl_surfaces[s].curvature_lo);
    let k = parts@(mkl_surfaces[s].conic, mkl_surfaces[s].conic_lo);
    let shape = mkl_surfaces[s].shape;
    if (shape == MKL_SHAPE_ODD_ASPHERE) {
        let base = mkl_conic_sag_jet@(c, k, r2);
        if (!base.ok) {
            return base;
        }
        if (zero@(x.v) && zero@(y.v)) {
            return base;
        }
        var radial: J_;
        if (zero@(r2.v)) {
            let ax = select(x.v, neg@(x.v), le@(x.v, 0.0));
            let ay = select(y.v, neg@(y.v), le@(y.v, 0.0));
            let scale = select(ax, ay, mkl_le_value@(ax, ay));
            let u = div@(x.v, scale);
            let v = div@(y.v, scale);
            let w = sqrt@(add@(mul@(u, u), mul@(v, v)));
            radial = J_(mul@(scale, w), div@(u, w), div@(v, w));
        } else {
            radial = mkl_jsqrt@(r2);
        }
        let r = mkl_jscale@(radial, div@(c@(1.0), radius));
        return S_(true, mkl_jsum@(base.j, mkl_jmul@(mkl_poly@(coeff, 0u, na, r), r)));
    }
    if (shape == MKL_SHAPE_TOROID || shape == MKL_SHAPE_BICONIC) {
        let cx = load@(data, MKL_COEFF_CX);
        let inv = div@(c@(1.0), radius);
        let xx = mkl_jscale@(x, inv);
        let yy = mkl_jscale@(y, inv);
        if (shape == MKL_SHAPE_TOROID) {
            let base = mkl_conic_sag_jet@(c, k, mkl_jmul@(y, y));
            if (!base.ok) {
                return base;
            }
            let zy = mkl_jsum@(base.j, mkl_even@(coeff, na, nb, mkl_jmul@(yy, yy)));
            let a = mkl_jadd@(mkl_jscale@(mkl_jneg@(zy), cx), c@(1.0));
            let q = mkl_jdiff@(mkl_jmul@(a, a), mkl_jscale@(mkl_jmul@(x, x), mul@(cx, cx)));
            if (le@(a.v, 0.0) || !mkl_le_value@(c@(0.0), q.v)) {
                return mkl_sag_fail@();
            }
            return S_(true, mkl_jsum@(zy, mkl_jdiv@(mkl_jscale@(mkl_jmul@(x, x), cx), mkl_jsum@(a, mkl_jsqrt@(q)))));
        }
        let kx = load@(data, MKL_COEFF_KX);
        let arg = mkl_jadd@(
            mkl_jsum@(
                mkl_jscale@(mkl_jmul@(x, x), mul@(mul@(neg@(add@(c@(1.0), kx)), cx), cx)),
                mkl_jscale@(mkl_jmul@(y, y), mul@(mul@(neg@(add@(c@(1.0), k)), c), c)),
            ),
            c@(1.0),
        );
        if (!mkl_le_value@(c@(0.0), arg.v)) {
            return mkl_sag_fail@();
        }
        let num = mkl_jsum@(mkl_jscale@(mkl_jmul@(x, x), cx), mkl_jscale@(mkl_jmul@(y, y), c));
        let z = mkl_jsum@(
            mkl_jsum@(mkl_jdiv@(num, mkl_jadd@(mkl_jsqrt@(arg), c@(1.0))), mkl_even@(coeff, 0u, na, mkl_jmul@(xx, xx))),
            mkl_even@(coeff, na, nb, mkl_jmul@(yy, yy)),
        );
        return S_(true, z);
    }
    let q = mkl_scaled_jet@(x, y, radius);
    if (!mkl_normalized@(q)) {
        return mkl_sag_fail@();
    }
    let base = mkl_conic_sag_jet@(c, k, r2);
    if (!base.ok) {
        return base;
    }
    let u = q.u;
    let v = q.v;
    let t = q.s;
    if (shape == MKL_SHAPE_Q_CON) {
        return S_(true, mkl_jsum@(base.j, mkl_qcon@(coeff, na, t)));
    }
    if (shape == MKL_SHAPE_Q_BFS) {
        if (na == 0u) {
            return base;
        }
        let w = mkl_jsqrt@(mkl_jadd@(mkl_jscale@(r2, mul@(neg@(c), c)), c@(1.0)));
        if (zero@(w.v)) {
            return mkl_sag_fail@();
        }
        let tt = mkl_jmul@(mkl_jmul@(t, mkl_jadd@(mkl_jneg@(t), c@(1.0))), mkl_qbfs@(coeff, na, t));
        return S_(true, mkl_jsum@(base.j, mkl_jdiv@(tt, w)));
    }
    if (shape == MKL_SHAPE_XY) {
        var start = 0u;
        var degree = 1u;
        var z = base.j;
        while (start < na) {
            var row = mkl_jconst@(c@(0.0));
            var vn = mkl_jconst@(c@(1.0));
            for (var n = 0u; n <= degree; n++) {
                var a = c@(0.0);
                if (start + n < na) {
                    a = load@(coeff, start + n);
                }
                row = mkl_jsum@(mkl_jmul@(row, u), mkl_jscale@(vn, a));
                vn = mkl_jmul@(vn, v);
            }
            z = mkl_jsum@(z, row);
            start += degree + 1u;
            degree++;
        }
        return S_(true, z);
    }
    let standard = mkl_pool[data + 2u * MKL_COEFF_KIND] != 0.0;
    return S_(true, mkl_jsum@(base.j, mkl_zernike@(coeff, na, standard, u, v, t)));
}
// The base of an overlaid surface may be legacy (or a DOE's legacy base).
fn mkl_legacy_sag@(s: u32, xv: R_, yv: R_) -> S_ {
    let x = mkl_jvar@(xv, true);
    let y = mkl_jvar@(yv, false);
    let r2 = mkl_jsum@(mkl_jmul@(x, x), mkl_jmul@(y, y));
    let sf = &mkl_surfaces[s];
    let base = mkl_conic_sag_jet@(parts@((*sf).curvature, (*sf).curvature_lo), parts@((*sf).conic, (*sf).conic_lo), r2);
    if (!base.ok) {
        return base;
    }
    let t = mkl_jscale@(r2, c@((*sf).asphere_inv_norm2));
    var p = mkl_jconst@(c@(0.0));
    for (var j = i32((*sf).num_asphere) - 1; j >= 0; j--) {
        p = mkl_jadd@(mkl_jmul@(p, t), parts@((*sf).asphere[j], (*sf).asphere_lo[j]));
    }
    return S_(true, mkl_jsum@(base.j, mkl_jmul@(mkl_jmul@(p, t), t)));
}
// Sag jet of an extended surface or, with MKL_OVERLAYS, of any surface with
// MKL_SURF_HAS_SAG_OVERLAY. Code without MKL_OVERLAYS never sees the overlay.
fn mkl_surface_eval@(s: u32, xv: R_, yv: R_) -> S_ {
    if (!MKL_OVERLAYS || (mkl_surfaces[s].flags & MKL_SURF_HAS_SAG_OVERLAY) == 0u) {
        return mkl_extended_sag@(s, xv, yv);
    }
    let data = mkl_surface_data(s);
    var shape = mkl_surfaces[s].shape;
    if (shape == MKL_SHAPE_DIFFRACTIVE) {
        shape = u32(mkl_pool[data + 2u * MKL_COEFF_KIND]);
    }
    var base: S_;
    if (shape <= MKL_SHAPE_EVEN_ASPHERE) {
        base = mkl_legacy_sag@(s, xv, yv);
    } else {
        base = mkl_extended_sag@(s, xv, yv);
    }
    if (!base.ok) {
        return base;
    }
    let d = data + 2u * (mkl_surfaces[s].coefficient_count - MKL_OVERLAY_COUNT);
    let inv = load@(d, MKL_OVERLAY_INV_RADIUS);
    let c = load@(d, MKL_OVERLAY_COS);
    let sn = load@(d, MKL_OVERLAY_SIN);
    let s4 = load@(d, MKL_OVERLAY_S4);
    let two = c@(2.0);
    let u = mul@(xv, inv);
    let v = mul@(yv, inv);
    let t = add@(mul@(u, u), mul@(v, v));
    let q = add@(load@(d, MKL_OVERLAY_S2), mul@(t, s4));
    let w = add@(q, mul@(t, s4));
    let o = J_(
        add@(add@(mul@(c, mul@(sub@(u, v), add@(u, v))), mul@(mul@(two, sn), mul@(u, v))), mul@(t, q)),
        mul@(mul@(two, inv), add@(mul@(u, add@(c, w)), mul@(sn, v))),
        mul@(mul@(two, inv), add@(mul@(v, sub@(w, c)), mul@(sn, u))),
    );
    return S_(true, mkl_jsum@(base.j, o));
}
