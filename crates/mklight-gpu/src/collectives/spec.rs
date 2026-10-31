//! Portable CPU-f32 specification. These routines perform no backend dispatch.
use crate::{Df32 as D, ReduceMode, fenv::FpEnv, pending::JobControl, reduce};
use mklight_core::MkResult;
use std::cmp::Ordering;

pub fn sum<E: FpEnv>(n: usize, f: impl Fn(usize) -> D, ctl: &JobControl) -> MkResult<D> {
    reduce::mapped_in::<E>(ReduceMode::DoubleFloat, n, f, ctl)
}
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
/// Independent columns are scaled before orthogonalisation. Norms use a
/// maximum scale before squaring, retaining finite very small/large columns.
fn norm<E: FpEnv>(x: &[D], ctl: &JobControl) -> MkResult<D> {
    let mut scale = 0f32;
    for (i, v) in x.iter().enumerate() {
        if i % 4096 == 0 {
            ctl.check()?;
        }
        if !v.is_finite() {
            return Ok(D::from_f32(f32::NAN));
        }
        scale = scale.max(v.hi.abs());
    }
    if E::eq(scale, 0.0) {
        return Ok(D::ZERO);
    }
    Ok(sum::<E>(
        x.len(),
        |i| {
            let v = x[i].div_in::<E>(D::from_f32(scale));
            v.mul_in::<E>(v)
        },
        ctl,
    )?
    .sqrt_in::<E>()
    .mul_f32_in::<E>(scale))
}
pub fn qr<E: FpEnv>(
    a: &[D],
    rhs: &[D],
    m: usize,
    n: usize,
    tolerance: f32,
    ctl: &JobControl,
) -> MkResult<QrResult> {
    let mut v: Vec<Vec<D>> = (0..n)
        .map(|j| (0..m).map(|i| a[i * n + j]).collect())
        .collect();
    let mut b = rhs.to_vec();
    let mut scales = vec![D::ZERO; n];
    let mut pivot: Vec<_> = (0..n).collect();
    let mut q = vec![vec![D::ZERO; m]; n];
    let mut r = vec![vec![D::ZERO; n]; n];
    let mut y = vec![D::ZERO; n];
    let mut rank = 0usize;
    let mut status = 0f32;
    for j in 0..n {
        scales[j] = norm::<E>(&v[j], ctl)?;
        if !scales[j].is_finite() {
            status = 2.0;
        }
        if scales[j].hi != 0.0 {
            for (index, x) in v[j].iter_mut().enumerate() {
                if index % 4096 == 0 {
                    ctl.check()?;
                }
                *x = x.div_in::<E>(scales[j]);
            }
        }
    }
    let mut first = D::ZERO;
    for k in 0..n {
        ctl.check()?;
        let mut norms = vec![D::ZERO; n];
        for j in k..n {
            norms[j] = norm::<E>(&v[j], ctl)?;
        }
        let mut best = k;
        for j in k + 1..n {
            if df_cmp(norms[j], norms[best]) == Ordering::Greater
                || (df_cmp(norms[j], norms[best]) == Ordering::Equal && pivot[j] < pivot[best])
            {
                best = j;
            }
        }
        let d = norms[best];
        if k == 0 {
            first = d;
        }
        if !d.is_finite() || status == 2.0 {
            status = 2.0;
            break;
        }
        if d.hi <= first.mul_f32_in::<E>(tolerance).hi {
            status = 1.0;
            break;
        }
        v.swap(k, best);
        pivot.swap(k, best);
        for row in r.iter_mut().take(k) {
            row.swap(k, best);
        }
        rank += 1;
        r[k][k] = d;
        for i in 0..m {
            if i % 4096 == 0 {
                ctl.check()?;
            }
            q[k][i] = v[k][i].div_in::<E>(d);
        }
        // Two passes prevent loss of orthogonality on clipped/conditioned pupils.
        for _ in 0..2 {
            for j in k + 1..n {
                let dot = sum::<E>(m, |i| q[k][i].mul_in::<E>(v[j][i]), ctl)?;
                r[k][j] = r[k][j].add_in::<E>(dot);
                for i in 0..m {
                    if i % 4096 == 0 {
                        ctl.check()?;
                    }
                    v[j][i] = v[j][i].add_in::<E>(-(q[k][i].mul_in::<E>(dot)));
                }
            }
            let dot = sum::<E>(m, |i| q[k][i].mul_in::<E>(b[i]), ctl)?;
            y[k] = y[k].add_in::<E>(dot);
            for i in 0..m {
                if i % 4096 == 0 {
                    ctl.check()?;
                }
                b[i] = b[i].add_in::<E>(-(q[k][i].mul_in::<E>(dot)));
            }
        }
        ctl.report((k + 1) as f64 / (n + 1) as f64, "QR orthogonalisation");
    }
    let nan = D::from_f32(f32::NAN);
    let mut solution = vec![nan; n];
    let mut condition = D::from_f32(f32::INFINITY);
    if rank == n && status == 0.0 {
        let mut x = vec![D::ZERO; n];
        for i in (0..n).rev() {
            let mut t = y[i];
            for j in i + 1..n {
                t = t.add_in::<E>(-(r[i][j].mul_in::<E>(x[j])));
            }
            x[i] = t.div_in::<E>(r[i][i]);
            solution[pivot[i]] = x[i].div_in::<E>(scales[pivot[i]]);
        }
        let mut rows = vec![D::ZERO; n];
        let mut normr = D::ZERO;
        for i in 0..n {
            let mut row = D::ZERO;
            for j in i..n {
                row = row.add_in::<E>(abs(r[i][j]));
            }
            if df_cmp(row, normr) == Ordering::Greater {
                normr = row;
            }
        }
        for col in 0..n {
            ctl.check()?;
            x.fill(D::ZERO);
            for i in (0..=col).rev() {
                let mut t = D::from_f32(if i == col { 1.0 } else { 0.0 });
                for j in i + 1..=col {
                    t = t.add_in::<E>(-(r[i][j].mul_in::<E>(x[j])));
                }
                x[i] = t.div_in::<E>(r[i][i]);
                rows[i] = rows[i].add_in::<E>(abs(x[i]));
            }
        }
        let norminv = rows.into_iter().max_by(|a, b| df_cmp(*a, *b)).unwrap();
        condition = normr.mul_in::<E>(norminv);
    }
    let mut factors = vec![D::ZERO; (m + n + 2) * n];
    for i in 0..m {
        for j in 0..n {
            factors[i * n + j] = q[j][i];
        }
    }
    for i in 0..n {
        for j in i..n {
            factors[(m + i) * n + j] = r[i][j];
        }
    }
    for j in 0..n {
        factors[(m + n) * n + j] = scales[j];
        factors[(m + n + 1) * n + j] = D::from_f64(pivot[j] as f64);
    }
    let residual = norm::<E>(&b, ctl)?;
    if !residual.is_finite() || status == 0.0 && solution.iter().any(|x| !x.is_finite()) {
        status = 2.0;
        condition = D::from_f32(f32::INFINITY);
        solution.fill(nan);
    }
    Ok(QrResult {
        factors,
        solution,
        diagnostics: [
            D::from_f32(rank as f32),
            condition,
            residual,
            D::from_f32(status),
        ],
    })
}
fn abs(x: D) -> D {
    if x.hi < 0.0 { -x } else { x }
}
