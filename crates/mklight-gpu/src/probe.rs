//! The floating-point behaviour probe every GPU backend runs at start-up
//! (GPU.md §3.3): deterministic inputs, the IEEE binary32 results its kernel
//! must reproduce, and the verdict. Backends only dispatch the kernel
//! (`mkl_numerics_probe` in MSL and WGSL) and hand its output here.

use crate::backend::NumericsProbe;
use crate::consts::MKL_PROBE_WORDS;
use mklight_core::rng::Philox;

/// Words the probe kernel writes per input triple.
pub const WORDS: usize = MKL_PROBE_WORDS as usize;

/// Deterministic inputs for the numerics probe: random magnitudes and
/// signs, triples with `c = −fl(a·b)` (where fused and unfused results
/// differ), and operands whose products land in the subnormal range.
pub fn inputs() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut r = Philox::new(0x0005_EED0_FA11, 0);
    let mut rnd = |emin: i32, emax: i32| {
        let m = 1.0 + r.uniform();
        let s = if r.uniform() < 0.5 { -1.0 } else { 1.0 };
        let e = emin + ((emax - emin) as f64 * r.uniform()) as i32;
        (s * m * 2f64.powi(e)) as f32
    };
    let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
    for i in 0..4096 {
        let x = rnd(-20, 20);
        let y = rnd(-20, 20);
        let z = match i % 3 {
            0 => -(x * y),
            1 => rnd(-20, 20),
            _ => x * y * rnd(-3, 0),
        };
        // Every 8th triple: |a| in [2^-9, 2^13) so (a·2^-70)·2^-70 is subnormal.
        let x = if i % 8 == 7 { rnd(-9, 12) } else { x };
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

/// The IEEE binary32 results the probe kernel must reproduce, in the
/// kernel's word order:
/// `a·b + c` (not contracted), `fma(a, b, c)`, `a / b`, `sqrt(|a|)`,
/// `a + b`, `a · b`, `(a 2⁻⁷⁰) 2⁻⁷⁰` (subnormal), `(a + b) − b` (not
/// reassociated).
pub fn expected(x: f32, y: f32, z: f32) -> [f32; WORDS] {
    let p70 = 2f32.powi(-70);
    [
        x * y + z,
        x.mul_add(y, z),
        x / y,
        x.abs().sqrt(),
        x + y,
        x * y,
        (x * p70) * p70,
        (x + y) - y,
    ]
}

/// Bitwise equality, any NaN equal to any NaN.
pub fn same(a: f32, b: f32) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

/// The verdict on the kernel's output `got` (`WORDS` per triple of
/// [`inputs`]).
pub fn evaluate(a: &[f32], b: &[f32], c: &[f32], got: &[f32]) -> NumericsProbe {
    let mut ok = [true; WORDS];
    for i in 0..a.len() {
        let want = expected(a[i], b[i], c[i]);
        for w in 0..WORDS {
            ok[w] &= same(got[i * WORDS + w], want[w]);
        }
    }
    NumericsProbe {
        no_contraction: ok[0],
        true_fma: ok[1],
        correctly_rounded_div: ok[2],
        correctly_rounded_sqrt: ok[3],
        ieee_add_mul: ok[4] && ok[5],
        subnormals: ok[6],
        no_reassociation: ok[7],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_cpu_passes_its_own_probe() {
        let (a, b, c) = inputs();
        let got: Vec<f32> = (0..a.len())
            .flat_map(|i| expected(a[i], b[i], c[i]))
            .collect();
        assert_eq!(evaluate(&a, &b, &c, &got), NumericsProbe::IEEE);
    }
}
