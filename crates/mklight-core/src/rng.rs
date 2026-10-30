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

/// Converts a u32 to a float in the open interval (0, 1), using the top 24
/// bits. This is the same mapping the GPU uses in f32, so both sides draw
/// identical values.
#[inline]
pub fn u32_to_unit_f64(x: u32) -> f64 {
    ((x >> 8) as f64 + 0.5) * (1.0 / 16_777_216.0)
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
    /// Position the generator at draw index `i`.
    pub fn seek(&mut self, i: u64) {
        self.block = i / 4;
        self.refill();
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
            self.block += 1;
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

    #[test]
    fn uniform_in_open_interval() {
        let mut r = Philox::new(42, 7);
        for _ in 0..10000 {
            let u = r.uniform();
            assert!(u > 0.0 && u < 1.0);
        }
    }
}
