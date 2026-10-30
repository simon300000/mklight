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

use rayon::prelude::*;

use crate::backend::ReduceMode;
use crate::consts::*;
use crate::df::{Df32, NeumaierF32};

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
    match mode {
        ReduceMode::Plain => Df32::from_f32(a.hi + b.hi),
        ReduceMode::Kahan | ReduceMode::DoubleFloat => a + b,
    }
}

/// One thread's grid-stride accumulation.
#[inline]
fn thread_sum(mode: ReduceMode, x: &[f32], g: usize, p: usize) -> Df32 {
    let it = x.iter().skip(g).step_by(p).copied();
    match mode {
        ReduceMode::Plain => {
            let mut s = 0f32;
            for v in it {
                s += v;
            }
            Df32::from_f32(s)
        }
        ReduceMode::Kahan => {
            let mut k = NeumaierF32::default();
            for v in it {
                k.add(v);
            }
            k.value()
        }
        ReduceMode::DoubleFloat => {
            let mut s = Df32::ZERO;
            for v in it {
                s = s.add_f32(v);
            }
            s
        }
    }
}

/// The in-group binary tree over exactly `THREADS` values.
fn tree(mode: ReduceMode, v: &mut [Df32]) -> Df32 {
    debug_assert_eq!(v.len(), THREADS);
    let mut stride = THREADS / 2;
    while stride > 0 {
        for i in 0..stride {
            v[i] = combine(mode, v[i], v[i + stride]);
        }
        stride /= 2;
    }
    v[0]
}

/// Pass 1 for one channel: the `num_groups(x.len())` group partials.
pub fn pass1(mode: ReduceMode, x: &[f32]) -> Vec<Df32> {
    let g = num_groups(x.len());
    let p = g * THREADS;
    (0..g)
        .into_par_iter()
        .map(|group| {
            let mut v: Vec<Df32> = (0..THREADS)
                .map(|t| thread_sum(mode, x, group * THREADS + t, p))
                .collect();
            tree(mode, &mut v)
        })
        .collect()
}

/// Pass 2: fold the partials of one channel.
pub fn pass2(mode: ReduceMode, partials: &[Df32]) -> Df32 {
    let mut v: Vec<Df32> = (0..THREADS)
        .map(|t| {
            partials
                .iter()
                .skip(t)
                .step_by(THREADS)
                .fold(Df32::ZERO, |acc, &q| combine(mode, acc, q))
        })
        .collect();
    tree(mode, &mut v)
}

/// The complete reduction of one channel, bit-identical to the GPU.
pub fn reduce_channel(mode: ReduceMode, x: &[f32]) -> Df32 {
    pass2(mode, &pass1(mode, x))
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
}
