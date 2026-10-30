//! Philox4x32-10 counter-based RNG (Salmon et al., SC'11). Stateless and
//! keyed by `(seed, stream, counter)`, so CPU and GPU kernels produce identical
//! streams (CONVENTIONS §6). GPU kernels must implement exactly this
//! function. The known-answer test below pins it.

const M0: u32 = 0xD251_1F53;
const M1: u32 = 0xCD9E_8D57;
const W0: u32 = 0x9E37_79B9;
const W1: u32 = 0xBB67_AE85;

#[inline]
fn mulhilo(a: u32, b: u32) -> (u32, u32) {
    let p = (a as u64) * (b as u64);
    ((p >> 32) as u32, p as u32)
}

/// One Philox4x32-10 block: counter `ctr`, key `key`.
pub fn philox4x32_10(ctr: [u32; 4], key: [u32; 2]) -> [u32; 4] {
    let mut c = ctr;
    let mut k = key;
    for round in 0..10 {
        if round > 0 {
            k[0] = k[0].wrapping_add(W0);
            k[1] = k[1].wrapping_add(W1);
        }
        let (hi0, lo0) = mulhilo(M0, c[0]);
        let (hi1, lo1) = mulhilo(M1, c[2]);
        c = [hi1 ^ c[1] ^ k[0], lo1, hi0 ^ c[3] ^ k[1], lo0];
    }
    c
}

/// Converts a u32 to a float in the open interval (0, 1) from its top 24
/// bits `k`: `(k + ½)·2⁻²⁴`. This is the f64 reference mapping.
///
/// For `k ≥ 2²³` this value needs 25 significand bits, so it is not an f32
/// (and rounding it to nearest could give 1.0). GPU kernels therefore use
/// [`u32_to_unit_f32`], which differs from this by half an f32 ulp for the
/// upper half of the range. CPU and GPU streams are identical as raw words and
/// as f32 uniforms; a consumer that needs the f64 uniform applies this
/// function to the raw words (CONVENTIONS §6).
#[inline]
pub fn u32_to_unit_f64(x: u32) -> f64 {
    ((x >> 8) as f64 + 0.5) * (1.0 / 16_777_216.0)
}

/// The canonical f32 uniform in (0, 1), computed identically by every GPU
/// backend: `(k + ½)·2⁻²⁴` rounded toward zero to f32, with `k = x >> 8`.
/// Equal to [`u32_to_unit_f64`] for `k < 2²³` and `2⁻²⁵` below it otherwise.
/// Every operation is exact (`k < 2²⁴`, power-of-two scale, and the half-ulp
/// term is added only where it fits), so the result does not depend on
/// rounding modes, FMA or flush-to-zero.
#[inline]
pub fn u32_to_unit_f32(x: u32) -> f32 {
    const TWO_M24: f32 = 1.0 / 16_777_216.0;
    const TWO_M25: f32 = 1.0 / 33_554_432.0;
    let k = x >> 8;
    let half = if k < 0x80_0000 { TWO_M25 } else { 0.0 };
    (k as f32).mul_add(TWO_M24, half)
}

/// Stream of uniform numbers for `(seed, stream)`. Draw `i` comes from block
/// `i / 4`, lane `i % 4`, with counter = `(block_lo, block_hi, stream_lo, stream_hi)`.
#[derive(Clone, Debug)]
pub struct Philox {
    key: [u32; 2],
    stream: u64,
    block: u64,
    buf: [u32; 4],
    lane: usize,
}

impl Philox {
    pub fn new(seed: u64, stream: u64) -> Self {
        Philox {
            key: [seed as u32, (seed >> 32) as u32],
            stream,
            block: 0,
            buf: [0; 4],
            lane: 4,
        }
    }
    /// Position the generator at draw index `i`: the next draw is draw `i`.
    pub fn seek(&mut self, i: u64) {
        self.block = i / 4;
        self.refill();
        // `block` always names the next block to generate.
        self.block = self.block.wrapping_add(1);
        self.lane = (i % 4) as usize;
    }
    fn refill(&mut self) {
        let ctr = [
            self.block as u32,
            (self.block >> 32) as u32,
            self.stream as u32,
            (self.stream >> 32) as u32,
        ];
        self.buf = philox4x32_10(ctr, self.key);
    }
    pub fn next_u32(&mut self) -> u32 {
        if self.lane >= 4 {
            self.refill();
            self.block = self.block.wrapping_add(1);
            self.lane = 0;
        }
        let v = self.buf[self.lane];
        self.lane += 1;
        v
    }
    /// Uniform in (0, 1).
    pub fn uniform(&mut self) -> f64 {
        u32_to_unit_f64(self.next_u32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-answer test from the Random123 distribution (kat_vectors).
    #[test]
    fn known_answers() {
        assert_eq!(
            philox4x32_10([0, 0, 0, 0], [0, 0]),
            [0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]
        );
        assert_eq!(
            philox4x32_10([0xffffffff; 4], [0xffffffff; 2]),
            [0x408f276d, 0x41c83b0e, 0xa20bc7c6, 0x6d5451fd]
        );
        assert_eq!(
            philox4x32_10(
                [0x243f6a88, 0x85a308d3, 0x13198a2e, 0x03707344],
                [0xa4093822, 0x299f31d0]
            ),
            [0xd16cfe09, 0x94fdcceb, 0x5001e420, 0x24126ea1]
        );
    }

    /// `seek(k)` followed by draws equals the sequential stream from `k`,
    /// including across block boundaries and for `k` not a multiple of 4.
    #[test]
    fn seek_matches_sequential() {
        let mut seq = Philox::new(0xABCD_EF01_2345_6789, 3);
        let all: Vec<u32> = (0..64).map(|_| seq.next_u32()).collect();
        for k in 0..40u64 {
            let mut r = Philox::new(0xABCD_EF01_2345_6789, 3);
            r.next_u32(); // seeking discards any state
            r.seek(k);
            for (j, &w) in all[k as usize..k as usize + 20].iter().enumerate() {
                assert_eq!(r.next_u32(), w, "seek {k}, draw {j}");
            }
        }
    }

    #[test]
    fn unit_mappings() {
        for x in [
            0u32,
            1,
            255,
            256,
            0x7FFF_FFFF,
            0x8000_0000,
            0xFFFF_FF00,
            u32::MAX,
        ]
        .into_iter()
        .chain((0..100_000u32).map(|i| i.wrapping_mul(0x9E37_79B9)))
        {
            let (f, d) = (u32_to_unit_f32(x), u32_to_unit_f64(x));
            assert!(f > 0.0 && f < 1.0 && d > 0.0 && d < 1.0, "{x:#x}");
            if (x >> 8) < 0x80_0000 {
                assert_eq!(f as f64, d);
            } else {
                assert_eq!(f as f64, d - 2f64.powi(-25));
            }
        }
    }

    #[test]
    fn uniform_in_open_interval() {
        let mut r = Philox::new(42, 7);
        for _ in 0..10000 {
            let u = r.uniform();
            assert!(u > 0.0 && u < 1.0);
        }
    }
}
