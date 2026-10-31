//! Deterministic compensated parallel reduction: the shared specification
//! of the GPU kernels `mkl_reduce_pass1` / `mkl_reduce_pass2` and their exact
//! CPU emulation (CONVENTIONS §7.5).
//!
//! The summation order is fixed by the data size alone, never by timing or by
//! the device, so results are reproducible run to run and bit-identical
//! between the CPU emulation and any GPU that passes the numerics probe:
//!
//! * `T = MKL_REDUCE_THREADS` threads per group, `G = clamp(⌈n / (T·16)⌉, 1, 256)`
//!   groups per channel, `P = G·T` threads.
//! * Pass 1: thread `g` accumulates `x[g], x[g+P], x[g+2P], …` in that order
//!   (plain f32, Neumaier f32 or double-float), then each group combines its
//!   `T` thread values by a binary tree (`stride = T/2 … 1`,
//!   `v[i] ← v[i] ⊕ v[i+stride]`) into one double-float partial.
//! * Pass 2: one group per channel; thread `t` folds partials `t, t+T, …`
//!   (at most one, since `G ≤ T`) into zero, then the same tree.
//!
//! No atomics are used, so the order never depends on scheduling.

use std::ops::Range;

use rayon::prelude::*;

use crate::backend::ReduceMode;
use crate::consts::*;
use crate::df::{Df32, NeumaierF32};
use crate::fenv::{FpEnv, Ieee};

pub const THREADS: usize = MKL_REDUCE_THREADS as usize;
pub const MAX_GROUPS: usize = MKL_REDUCE_MAX_GROUPS as usize;
pub const ITEMS_PER_THREAD: usize = MKL_REDUCE_ITEMS_PER_THREAD as usize;

const _: () = assert!(MAX_GROUPS <= THREADS && THREADS.is_power_of_two());

/// Pass-1 groups per channel for `count` values.
pub fn num_groups(count: usize) -> usize {
    count
        .div_ceil(THREADS * ITEMS_PER_THREAD)
        .clamp(1, MAX_GROUPS)
}

/// `a ⊕ b` for the tree stages.
#[inline]
pub fn combine(mode: ReduceMode, a: Df32, b: Df32) -> Df32 {
    combine_in::<Ieee>(mode, a, b)
}

/// [`combine`] in environment `E`.
#[inline]
pub fn combine_in<E: FpEnv>(mode: ReduceMode, a: Df32, b: Df32) -> Df32 {
    match mode {
        ReduceMode::Plain => Df32::from_f32(E::add(a.hi, b.hi)),
        ReduceMode::Kahan | ReduceMode::DoubleFloat => a.add_in::<E>(b),
    }
}

/// One thread's grid-stride accumulation.
#[inline]
fn thread_sum<E: FpEnv>(mode: ReduceMode, x: &[f32], g: usize, p: usize) -> Df32 {
    let it = x.iter().skip(g).step_by(p).copied();
    match mode {
        ReduceMode::Plain => {
            let mut s = 0f32;
            for v in it {
                s = E::add(s, v);
            }
            Df32::from_f32(s)
        }
        ReduceMode::Kahan => {
            let mut k = NeumaierF32::default();
            for v in it {
                k.add_in::<E>(v);
            }
            k.value_in::<E>()
        }
        ReduceMode::DoubleFloat => {
            let mut s = Df32::ZERO;
            for v in it {
                s = s.add_f32_in::<E>(v);
            }
            s
        }
    }
}

/// Pass 1 for one channel: the `num_groups(x.len())` group partials.
pub fn pass1(mode: ReduceMode, x: &[f32]) -> Vec<Df32> {
    pass1_in::<Ieee>(mode, x, 0..num_groups(x.len()))
}

/// The partials of pass-1 groups `groups` (a sub-range of
/// `0..num_groups(x.len())`) in environment `E`. Each group's result does not
/// depend on which other groups run with it, so a backend may split pass 1
/// into several launches (for cancellation) without changing any bit.
pub fn pass1_in<E: FpEnv>(mode: ReduceMode, x: &[f32], groups: Range<usize>) -> Vec<Df32> {
    let p = num_groups(x.len()) * THREADS;
    groups
        .into_par_iter()
        .map(|group| {
            let mut v: Vec<Df32> = (0..THREADS)
                .map(|t| thread_sum::<E>(mode, x, group * THREADS + t, p))
                .collect();
            tree_acc::<E>(Accumulator::Sum(mode), &mut v)
        })
        .collect()
}

/// Pass 2: fold the partials of one channel.
pub fn pass2(mode: ReduceMode, partials: &[Df32]) -> Df32 {
    pass2_in::<Ieee>(mode, partials)
}

/// [`pass2`] in environment `E`.
pub fn pass2_in<E: FpEnv>(mode: ReduceMode, partials: &[Df32]) -> Df32 {
    let mut v: Vec<Df32> = (0..THREADS)
        .map(|t| {
            partials
                .iter()
                .skip(t)
                .step_by(THREADS)
                .fold(Df32::ZERO, |acc, &q| combine_in::<E>(mode, acc, q))
        })
        .collect();
    tree_acc::<E>(Accumulator::Sum(mode), &mut v)
}

/// The complete reduction of one channel, bit-identical to the GPU.
pub fn reduce_channel(mode: ReduceMode, x: &[f32]) -> Df32 {
    reduce_channel_in::<Ieee>(mode, x)
}

/// [`reduce_channel`] in environment `E`.
pub fn reduce_channel_in<E: FpEnv>(mode: ReduceMode, x: &[f32]) -> Df32 {
    pass2_in::<E>(mode, &pass1_in::<E>(mode, x, 0..num_groups(x.len())))
}

/// Accumulator of a fixed tree: a summation mode or a double-float extremum
/// (ties keep the earlier operand, the first NaN wins).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Accumulator {
    Sum(ReduceMode),
    Min,
    Max,
}
fn extremum(a: Df32, b: Df32, max: bool) -> Df32 {
    use std::cmp::Ordering::*;
    if a.hi.is_nan() {
        return a;
    }
    if b.hi.is_nan() {
        return b;
    }
    let c = crate::collectives::spec::df_cmp(a, b);
    if (max && c == Less) || (!max && c == Greater) {
        b
    } else {
        a
    }
}
fn combine_acc<E: FpEnv>(op: Accumulator, a: Df32, b: Df32) -> Df32 {
    match op {
        Accumulator::Sum(mode) => combine_in::<E>(mode, a, b),
        Accumulator::Min => extremum(a, b, false),
        Accumulator::Max => extremum(a, b, true),
    }
}

/// The fixed two-pass tree of this module over generated double-float
/// values, starting every thread and pass-2 slot from `identity`. For f32
/// values (lo = 0) and sums from zero it reproduces [`reduce_channel_in`].
pub fn tree_in<E: FpEnv>(
    op: Accumulator,
    identity: Df32,
    count: usize,
    value: impl Fn(usize) -> Df32,
    ctl: &crate::pending::JobControl,
) -> mklight_core::MkResult<Df32> {
    let groups = num_groups(count);
    let mut partials = Vec::with_capacity(groups);
    for g in 0..groups {
        ctl.check()?;
        let mut v: Vec<Df32> = (0..THREADS)
            .map(|t| {
                let items = (g * THREADS + t..count)
                    .step_by(groups * THREADS)
                    .map(&value);
                match op {
                    Accumulator::Sum(ReduceMode::Plain) => Df32::from_f32(
                        items.fold(identity.hi, |s, q| E::add(s, E::add(q.hi, q.lo))),
                    ),
                    Accumulator::Sum(ReduceMode::Kahan) => {
                        let mut k = NeumaierF32::default();
                        for q in items {
                            k.add_in::<E>(q.hi);
                            if q.lo != 0.0 {
                                k.add_in::<E>(q.lo);
                            }
                        }
                        k.value_in::<E>()
                    }
                    _ => items.fold(identity, |a, q| combine_acc::<E>(op, a, q)),
                }
            })
            .collect();
        partials.push(tree_acc::<E>(op, &mut v));
    }
    let mut v: Vec<Df32> = (0..THREADS)
        .map(|t| {
            partials
                .iter()
                .skip(t)
                .step_by(THREADS)
                .fold(identity, |a, &q| combine_acc::<E>(op, a, q))
        })
        .collect();
    Ok(tree_acc::<E>(op, &mut v))
}
/// The in-group binary tree over exactly `THREADS` values.
fn tree_acc<E: FpEnv>(op: Accumulator, v: &mut [Df32]) -> Df32 {
    let mut stride = THREADS / 2;
    while stride > 0 {
        for i in 0..stride {
            v[i] = combine_acc::<E>(op, v[i], v[i + stride]);
        }
        stride /= 2;
    }
    v[0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::math::KahanSum;
    use mklight_core::rng::Philox;

    fn exact(x: &[f32]) -> f64 {
        // f64 Neumaier over f32 inputs: error ≲ 2^-53 relative, far below
        // the double-float result's ~2^-46.
        x.iter().map(|&v| v as f64).collect::<KahanSum>().value()
    }

    #[test]
    fn geometry() {
        assert_eq!(num_groups(0), 1);
        assert_eq!(num_groups(1), 1);
        assert_eq!(num_groups(4096), 1);
        assert_eq!(num_groups(4097), 2);
        assert_eq!(num_groups(1 << 30), MAX_GROUPS);
    }

    #[test]
    fn small_inputs_exact() {
        for n in [0usize, 1, 2, 3, 255, 256, 257, 5000] {
            let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
            let want = (n * n.saturating_sub(1) / 2) as f64;
            for mode in [
                ReduceMode::Plain,
                ReduceMode::Kahan,
                ReduceMode::DoubleFloat,
            ] {
                assert_eq!(reduce_channel(mode, &x).to_f64(), want, "{mode:?} n={n}");
            }
        }
    }

    #[test]
    fn compensated_modes_are_accurate() {
        let mut r = Philox::new(9, 1);
        let n = 1_000_000;
        let x: Vec<f32> = (0..n)
            .map(|i| {
                let s = if i % 2 == 0 { 1.0 } else { -1.0 };
                (s * 1e3 + r.uniform()) as f32
            })
            .collect();
        let want = exact(&x);
        let l1: f64 = x.iter().map(|v| v.abs() as f64).sum();
        let df = reduce_channel(ReduceMode::DoubleFloat, &x).to_f64();
        let kh = reduce_channel(ReduceMode::Kahan, &x).to_f64();
        let pl = reduce_channel(ReduceMode::Plain, &x).to_f64();
        assert!(
            (df - want).abs() <= 1e-14 * l1,
            "df {}",
            (df - want).abs() / l1
        );
        assert!(
            (kh - want).abs() <= 1e-12 * l1,
            "kahan {}",
            (kh - want).abs() / l1
        );
        assert!(((df - want) / want).abs() < 1e-11);
        assert!(((pl - want) / want).abs() > 1e-7);
    }

    #[test]
    fn deterministic() {
        let x: Vec<f32> = (0..100_000)
            .map(|i| ((i * 7919) % 1000) as f32 * 1e-3)
            .collect();
        let a = reduce_channel(ReduceMode::DoubleFloat, &x);
        let b = reduce_channel(ReduceMode::DoubleFloat, &x);
        assert_eq!(a, b);
    }

    /// Splitting pass 1 into group ranges changes no bit.
    #[test]
    fn split_pass1_is_identical() {
        let x: Vec<f32> = (0..300_001usize)
            .map(|i| ((i * 7919) % 1013) as f32 * 1e-2)
            .collect();
        let g = num_groups(x.len());
        assert!(g > 10);
        for mode in [
            ReduceMode::Plain,
            ReduceMode::Kahan,
            ReduceMode::DoubleFloat,
        ] {
            let whole = pass1(mode, &x);
            let mut parts = pass1_in::<Ieee>(mode, &x, 0..3);
            parts.extend(pass1_in::<Ieee>(mode, &x, 3..g - 1));
            parts.extend(pass1_in::<Ieee>(mode, &x, g - 1..g));
            assert_eq!(whole, parts, "{mode:?}");
        }
    }

    /// Under flush-to-zero, subnormal inputs contribute nothing.
    #[test]
    fn ftz_flushes_subnormal_inputs() {
        use crate::fenv::Ftz;
        let x = vec![1e-39f32; 4096];
        for mode in [
            ReduceMode::Plain,
            ReduceMode::Kahan,
            ReduceMode::DoubleFloat,
        ] {
            assert_eq!(reduce_channel_in::<Ftz>(mode, &x), Df32::ZERO, "{mode:?}");
            assert!(reduce_channel(mode, &x).to_f64() > 4e-36);
        }
    }
}
