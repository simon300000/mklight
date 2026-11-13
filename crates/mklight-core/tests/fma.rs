//! `platform::fma` (the web build's fused multiply-add) against the
//! hardware's: bit for bit on every kind of input, the fast path's range
//! and every fallback. Natively `f64::mul_add` is the instruction (aarch64
//! `fmadd`, x86_64 with FMA), so it is the oracle; on a target without it,
//! it is the same correctly rounded integer algorithm as `fma_exact`.

use mklight_core::platform::{fma, fma_exact};

/// SplitMix64: a reproducible stream of 64-bit values.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }
    /// Random sign and mantissa with a biased exponent in `lo..=hi`.
    fn with_exp(&mut self, lo: u64, hi: u64) -> f64 {
        let e = lo + self.next() % (hi - lo + 1);
        let m = self.next() & ((1 << 52) - 1);
        let s = self.next() & (1 << 63);
        f64::from_bits(s | (e << 52) | m)
    }
    /// A value of the size the tracer works with (1e-12 … 1e6, either sign).
    fn tracer(&mut self) -> f64 {
        let mag = 10f64.powf(-12.0 + 18.0 * self.unit());
        if self.next() & 1 == 0 { mag } else { -mag }
    }
}

/// `n` cases, times `MKLIGHT_FMA_STRESS` when set (a longer run on demand:
/// `MKLIGHT_FMA_STRESS=100 cargo test --release -p mklight-core --test tests fma::`).
fn cases(n: usize) -> usize {
    n * std::env::var("MKLIGHT_FMA_STRESS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1)
}

fn check(a: f64, b: f64, c: f64) {
    let want = a.mul_add(b, c);
    let got = fma(a, b, c);
    let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
    assert!(
        same,
        "fma({a:e}, {b:e}, {c:e}) = {got:e} ({:#x}), want {want:e} ({:#x})",
        got.to_bits(),
        want.to_bits()
    );
    let exact = fma_exact(a, b, c);
    assert!(
        exact.to_bits() == want.to_bits() || (exact.is_nan() && want.is_nan()),
        "fma_exact({a:e}, {b:e}, {c:e})"
    );
}

#[test]
fn special_values_and_signed_zeros() {
    let specials = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        f64::EPSILON,
        1.0 + f64::EPSILON,
        1e300,
        -1e-300,
        3.0,
        0.1,
    ];
    for &a in &specials {
        for &b in &specials {
            for &c in &specials {
                check(a, b, c);
            }
        }
    }
}

#[test]
fn random_bit_patterns_over_every_exponent() {
    let mut r = Rng(1);
    for _ in 0..cases(2_000_000) {
        let (a, b, c) = (
            f64::from_bits(r.next()),
            f64::from_bits(r.next()),
            f64::from_bits(r.next()),
        );
        check(a, b, c);
    }
}

#[test]
fn the_fast_path_range_and_its_edges() {
    let mut r = Rng(2);
    for _ in 0..cases(3_000_000) {
        // Exponents around the fast path's limits (523..=1523, a sum of
        // 1146, an addend up to 2023) and well inside them.
        let a = r.with_exp(500, 1546);
        let b = r.with_exp(500, 1546);
        let c = match r.next() % 4 {
            0 => r.with_exp(0, 2046),
            1 => r.with_exp(2000, 2046),
            2 => r.with_exp(0, 140),
            _ => r.with_exp(900, 1150),
        };
        check(a, b, c);
    }
}

#[test]
fn tracer_sized_values() {
    let mut r = Rng(3);
    for _ in 0..cases(3_000_000) {
        check(r.tracer(), r.tracer(), r.tracer());
    }
}

#[test]
fn cancellation_and_near_ties() {
    let mut r = Rng(4);
    for _ in 0..cases(2_000_000) {
        let a = r.tracer();
        let b = r.tracer();
        let p = a * b;
        // c ≈ -a·b: the result is the product's rounding error and below,
        // down to exact zero and the subnormal range.
        let k = r.next() % 6;
        let c = match k {
            0 => -p,
            1 => -p + p * f64::EPSILON * (r.unit() - 0.5),
            2 => f64::from_bits((-p).to_bits().wrapping_add(r.next() % 5).wrapping_sub(2)),
            // A product that is a half-way case after adding c: the exact
            // error of a·b decides the rounding.
            3 => {
                let ulp = f64::from_bits(p.abs().to_bits() + 1) - p.abs();
                p.signum() * ulp * 0.5 * (1 + (r.next() % 3)) as f64
            }
            4 => -p * 0.5,
            _ => p * (r.unit() * 1e-17),
        };
        check(a, b, c);
        // The same in the subnormal and huge ranges.
        let s = 2f64.powi(-1000);
        check(a * s, b, c * s);
        check(a * 2f64.powi(900), b, c * 2f64.powi(900));
    }
}

#[test]
fn products_at_the_underflow_and_overflow_boundaries() {
    let mut r = Rng(5);
    for _ in 0..cases(1_000_000) {
        // Products near 2^-1022, the subnormal range and 2^1024.
        let ea = r.next() % 2047;
        let lo = 1023u64
            .saturating_sub(ea.saturating_sub(1023))
            .saturating_sub(60);
        let eb_target = (1023 + 1023 + (r.next() % 120)).saturating_sub(ea + 60);
        let a = r.with_exp(ea.max(1), ea.max(1));
        let b = r.with_exp(eb_target.clamp(1, 2046), eb_target.clamp(1, 2046));
        let _ = lo;
        let c = match r.next() % 3 {
            0 => 0.0,
            1 => r.with_exp(0, 60),
            _ => -(a * b),
        };
        check(a, b, c);
        // Overflow side.
        let a2 = r.with_exp(1500, 2046);
        let b2 = r.with_exp(1000, 1100);
        check(a2, b2, -(a2 * b2));
        check(a2, b2, r.with_exp(2040, 2046));
    }
}
