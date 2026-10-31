//! Portable CPU-f32 specification of the collective kernels.
use crate::{Df32 as D, consts::*, fenv::FpEnv, pending::JobControl};
use mklight_core::MkResult;
use std::cmp::Ordering;

pub fn scan_u32(x: &[u32], ctl: &JobControl) -> MkResult<Vec<u32>> {
    scan(x, 0, |a, b| a.wrapping_add(b), ctl)
}
pub fn scan_df<E: FpEnv>(x: &[D], ctl: &JobControl) -> MkResult<Vec<D>> {
    scan(x, D::ZERO, |a, b| a.add_in::<E>(b), ctl)
}
fn scan<T: Copy>(
    x: &[T],
    zero: T,
    add: impl Fn(T, T) -> T + Copy,
    ctl: &JobControl,
) -> MkResult<Vec<T>> {
    let mut out = vec![zero; x.len()];
    let mut totals = Vec::new();
    for (b, block) in x.chunks(256).enumerate() {
        ctl.check()?;
        let mut a = [zero; 256];
        a[..block.len()].copy_from_slice(block);
        for step in [1, 2, 4, 8, 16, 32, 64, 128] {
            let prev = a;
            for i in step..256 {
                a[i] = add(prev[i - step], prev[i]);
            }
        }
        out[b * 256..b * 256 + block.len()].copy_from_slice(&a[..block.len()]);
        totals.push(a[255]);
        ctl.report(
            0.5 * (b + 1) as f64 / x.len().div_ceil(256).max(1) as f64,
            "scan blocks",
        );
    }
    if totals.len() > 1 {
        let prefixes = scan(&totals, zero, add, &ctl.subrange(0.5, 0.75))?;
        for (b, block) in out.chunks_mut(256).enumerate().skip(1) {
            ctl.check()?;
            let block_len = block.len();
            for v in block {
                *v = add(prefixes[b - 1], *v);
            }
            ctl.report(
                0.75 + 0.25 * (b * 256 + block_len) as f64 / x.len() as f64,
                "scan prefixes",
            );
        }
    }
    Ok(out)
}
/// Numeric total order, including a deterministic NaN tail. Signed zero ties.
pub fn float_cmp(a: f32, b: f32) -> Ordering {
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        (false, false) => a.partial_cmp(&b).unwrap(),
    }
}
pub fn df_cmp(a: D, b: D) -> Ordering {
    let an = a.hi.is_nan() || a.lo.is_nan();
    let bn = b.hi.is_nan() || b.lo.is_nan();
    if an || bn {
        return an.cmp(&bn);
    }
    float_cmp(a.hi, b.hi).then_with(|| float_cmp(a.lo, b.lo))
}
pub fn permutation(
    n: usize,
    cmp: impl Fn(usize, usize) -> Ordering,
    ctl: &JobControl,
) -> MkResult<Vec<usize>> {
    let mut a: Vec<_> = (0..n).collect();
    let mut b = vec![0; n];
    let mut width = 1;
    while width < n {
        for base in (0..n).step_by(2 * width) {
            ctl.check()?;
            let mid = (base + width).min(n);
            let end = (base + 2 * width).min(n);
            let (mut l, mut r) = (base, mid);
            for (index, dest) in b[base..end].iter_mut().enumerate() {
                if index % 4096 == 0 {
                    ctl.check()?;
                }
                if l < mid && (r == end || cmp(a[l], a[r]) != Ordering::Greater) {
                    *dest = a[l];
                    l += 1;
                } else {
                    *dest = a[r];
                    r += 1;
                }
            }
        }
        std::mem::swap(&mut a, &mut b);
        width *= 2;
    }
    Ok(a)
}

pub struct QrResult {
    pub factors: Vec<D>,
    pub solution: Vec<D>,
    pub diagnostics: [D; 4],
}
/// Threads per QR column group for `rows` rows.
pub fn qr_threads(rows: usize) -> usize {
    if rows >= MKL_COLLECT_QR_WIDE_ROWS as usize {
        MKL_COLLECT_QR_WIDE_THREADS as usize
    } else {
        MKL_COLLECT_QR_THREADS as usize
    }
}
/// Strides 16..1 of the binary tree inside each 32-lane SIMD group.
fn simd_levels<E: FpEnv>(v: &mut [D]) {
    for s in [16, 8, 4, 2, 1] {
        for l in 0..s {
            v[l] = v[l].add_in::<E>(v[l + s]);
        }
    }
}
/// QR group sum: thread t adds rows t, t+T, ...; SIMD levels in each SIMD
/// group, then the same levels over the zero-padded SIMD partials.
fn qr_sum<E: FpEnv>(m: usize, f: impl Fn(usize) -> D) -> D {
    let threads = qr_threads(m);
    let mut t = vec![D::ZERO; threads];
    for i in 0..m {
        t[i % threads] = t[i % threads].add_in::<E>(f(i));
    }
    let mut partials = [D::ZERO; 32];
    for (s, lanes) in t.chunks_mut(32).enumerate() {
        simd_levels::<E>(lanes);
        partials[s] = lanes[0];
    }
    simd_levels::<E>(&mut partials);
    partials[0]
}
fn max_nan(a: f32, b: f32) -> f32 {
    if a.is_nan() || b.is_nan() { f32::NAN } else { a.max(b) }
}
/// x 2^e through two normal powers of two.
fn ldexp<E: FpEnv>(x: D, e: i32) -> D {
    let pow2 = |k: i32| f32::from_bits(((127 + k) as u32) << 23);
    let (a, b) = (pow2(e / 2), pow2(e - e / 2));
    D::new(E::mul(E::mul(x.hi, a), b), E::mul(E::mul(x.lo, a), b))
}
/// 2-norm scaled by the exponent of the largest |hi|; NaN if any entry is
/// non-finite.
fn norm<E: FpEnv>(v: &[D]) -> D {
    let mx = v.iter().fold(0f32, |m, x| max_nan(m, x.hi.abs()));
    if !mx.is_finite() {
        return D::from_f32(f32::NAN);
    }
    if mx == 0.0 {
        return D::ZERO;
    }
    let e = ((mx.to_bits() >> 23) & 255) as i32 - 126;
    let s = qr_sum::<E>(v.len(), |i| {
        let x = ldexp::<E>(v[i], -e);
        x.mul_in::<E>(x)
    });
    ldexp::<E>(s.sqrt_in::<E>(), e)
}
fn abs(x: D) -> D {
    if x.hi < 0.0 { -x } else { x }
}
fn ext_max(a: D, b: D) -> D {
    if a.hi.is_nan() || (!b.hi.is_nan() && df_cmp(a, b) != Ordering::Less) { a } else { b }
}
/// One fit, step for step as `mkl_c_qr_{prepare,pivot,project,finish}`.
pub fn qr<E: FpEnv>(
    a: &[D],
    rhs: &[D],
    m: usize,
    n: usize,
    tolerance: f32,
    ctl: &JobControl,
) -> MkResult<QrResult> {
    let at = |i: usize, j: usize| i * n + j;
    let mut f = vec![D::ZERO; (m + n + 2) * n];
    let mut v: Vec<Vec<D>> = (0..=n)
        .map(|j| (0..m).map(|i| if j == n { rhs[i] } else { a[at(i, j)] }).collect())
        .collect();
    let mut norms = vec![D::ZERO; n + 1];
    let mut y = vec![D::ZERO; n];
    for j in 0..=n {
        let scale = norm::<E>(&v[j]);
        if j == n {
            norms[n] = scale;
            continue;
        }
        f[at(m + n, j)] = scale;
        f[at(m + n + 1, j)] = D::from_f64(j as f64);
        if scale.hi != 0.0 {
            v[j].iter_mut().for_each(|x| *x = x.div_in::<E>(scale));
        }
        norms[j] = norm::<E>(&v[j]);
    }
    let (mut rank, mut status, mut first) = (0usize, 0f32, D::ZERO);
    let mut qs: Vec<Vec<D>> = Vec::with_capacity(n);
    for k in 0..n {
        ctl.check()?;
        let pivot = |f: &[D], j: usize| f[at(m + n + 1, j)].hi;
        let mut best = k;
        for j in k + 1..n {
            let c = df_cmp(norms[j], norms[best]);
            if c == Ordering::Greater || (c == Ordering::Equal && pivot(&f, j) < pivot(&f, best)) {
                best = j;
            }
        }
        let d = norms[best];
        if k == 0 {
            first = d;
        }
        if !d.is_finite() || (k == 0 && (0..n).any(|j| !f[at(m + n, j)].is_finite())) {
            status = 2.0;
            break;
        }
        if d.hi <= first.mul_f32_in::<E>(tolerance).hi {
            status = 1.0;
            break;
        }
        rank = k + 1;
        for r in (m..m + k).chain([m + n + 1]) {
            f.swap(at(r, k), at(r, best));
        }
        v.swap(k, best);
        // Classical Gram-Schmidt, twice, against every earlier q.
        for _ in 0..2 * usize::from(k > 0) {
            let c: Vec<D> = (0..k)
                .map(|i| qr_sum::<E>(m, |r| qs[i][r].mul_in::<E>(v[k][r])))
                .collect();
            for (r, x) in v[k].iter_mut().enumerate() {
                *x = (0..k).fold(*x, |x, i| x.add_in::<E>(-(qs[i][r].mul_in::<E>(c[i]))));
            }
            for (i, c) in c.iter().enumerate() {
                f[at(m + i, k)] = f[at(m + i, k)].add_in::<E>(*c);
            }
        }
        let d = norm::<E>(&v[k]);
        f[at(m + k, k)] = d;
        let q: Vec<D> = v[k].iter().map(|x| x.div_in::<E>(d)).collect();
        for (i, &q) in q.iter().enumerate() {
            f[at(i, k)] = q;
        }
        for j in k + 1..=n {
            let c = &mut v[j];
            let dot1 = qr_sum::<E>(m, |i| q[i].mul_in::<E>(c[i]));
            let once = |i: usize| c[i].add_in::<E>(-(q[i].mul_in::<E>(dot1)));
            let dot2 = qr_sum::<E>(m, |i| q[i].mul_in::<E>(once(i)));
            let projected: Vec<D> = (0..m)
                .map(|i| once(i).add_in::<E>(-(q[i].mul_in::<E>(dot2))))
                .collect();
            *c = projected;
            let r = dot1.add_in::<E>(dot2);
            if j < n {
                f[at(m + k, j)] = r;
            } else {
                y[k] = r;
            }
            norms[j] = norm::<E>(c);
        }
        qs.push(q);
        ctl.report((k + 1) as f64 / (n + 1) as f64, "QR orthogonalisation");
    }
    let r = |i: usize, j: usize| f[at(m + i, j)];
    let residual = norms[n];
    let mut ok = residual.is_finite() && status == 0.0;
    let mut x = vec![D::ZERO; n];
    let mut solution = vec![D::from_f32(f32::NAN); n];
    if ok {
        for i in (0..n).rev() {
            let mut t = y[i];
            for j in i + 1..n {
                t = t.add_in::<E>(-(r(i, j).mul_in::<E>(x[j])));
            }
            x[i] = t.div_in::<E>(r(i, i));
        }
    }
    for (i, &xi) in x.iter().enumerate() {
        if !ok {
            break;
        }
        let j = f[at(m + n + 1, i)].hi as usize;
        let c = xi.div_in::<E>(f[at(m + n, j)]);
        solution[j] = c;
        ok = c.is_finite();
    }
    if !residual.is_finite() || (!ok && status == 0.0) {
        status = 2.0;
    }
    let mut condition = D::from_f32(f32::INFINITY);
    if ok {
        let (mut normr, mut norminv) = (D::ZERO, D::ZERO);
        let mut z = vec![D::ZERO; n];
        for i in 0..n {
            ctl.check()?;
            let row = (i..n).fold(D::ZERO, |s, j| s.add_in::<E>(abs(r(i, j))));
            z[i] = D::from_f32(1.0).div_in::<E>(r(i, i));
            let mut sum = abs(z[i]);
            for j in i + 1..n {
                let t = (i..j).fold(D::ZERO, |t, l| t.add_in::<E>(z[l].mul_in::<E>(r(l, j))));
                z[j] = (-t).div_in::<E>(r(j, j));
                sum = sum.add_in::<E>(abs(z[j]));
            }
            normr = ext_max(normr, row);
            norminv = ext_max(norminv, sum);
        }
        condition = normr.mul_in::<E>(norminv);
    } else {
        solution.fill(D::from_f32(f32::NAN));
    }
    Ok(QrResult {
        factors: f,
        solution,
        diagnostics: [D::from_f32(rank as f32), condition, residual, D::from_f32(status)],
    })
}
