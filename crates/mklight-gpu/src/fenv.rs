//! The f32 floating-point environment of an emulated device.
//!
//! Every f32 algorithm in this crate (`df`, `fmath`, `reduce`) is written
//! against [`FpEnv`], so the CPU emulation can reproduce a GPU bit for bit in
//! both of the environments real devices have:
//!
//! * [`Ieee`]: IEEE-754 binary32 with gradual underflow (subnormals kept).
//!   Every method is the plain Rust operation and compiles to it.
//! * [`Ftz`]: flush-to-zero, as measured on Apple GPUs (the Metal backend's
//!   numerics probe reports it as `subnormals: false`). Arithmetic
//!   (`+ − × fma / sqrt`) reads a subnormal operand as a zero of the same sign
//!   and writes a subnormal result as a zero of the same sign. `fma` rounds the
//!   exact `a·b + c` of the flushed operands once, then flushes. Comparisons
//!   also read subnormal operands as zero. Bit operations (`abs`, negation,
//!   `max`/`min`, selection, loads and stores) move bits unchanged, so a raw
//!   subnormal can pass through them.
//!
//! Normal-range operands with normal-range results behave identically in both
//! environments.

/// Arithmetic and comparisons of one f32 environment (see the module docs).
pub trait FpEnv: Copy + Send + Sync + 'static {
    /// Subnormals are flushed to zero.
    const FTZ: bool;

    /// An operand or result as the environment sees it.
    #[inline(always)]
    fn d(x: f32) -> f32 {
        if Self::FTZ && x.is_subnormal() {
            0f32.copysign(x)
        } else {
            x
        }
    }
    #[inline(always)]
    fn add(a: f32, b: f32) -> f32 {
        Self::d(Self::d(a) + Self::d(b))
    }
    #[inline(always)]
    fn sub(a: f32, b: f32) -> f32 {
        Self::d(Self::d(a) - Self::d(b))
    }
    #[inline(always)]
    fn mul(a: f32, b: f32) -> f32 {
        Self::d(Self::d(a) * Self::d(b))
    }
    /// `a·b + c` with a single rounding.
    #[inline(always)]
    fn fma(a: f32, b: f32, c: f32) -> f32 {
        Self::d(Self::d(a).mul_add(Self::d(b), Self::d(c)))
    }
    /// Correctly rounded division (bitwise agreement with a device needs the
    /// probe's `correctly_rounded_div`).
    #[inline(always)]
    fn div(a: f32, b: f32) -> f32 {
        Self::d(Self::d(a) / Self::d(b))
    }
    /// Correctly rounded square root (see [`FpEnv::div`]).
    #[inline(always)]
    fn sqrt(a: f32) -> f32 {
        Self::d(Self::d(a).sqrt())
    }
    #[inline(always)]
    fn lt(a: f32, b: f32) -> bool {
        Self::d(a) < Self::d(b)
    }
    #[inline(always)]
    fn le(a: f32, b: f32) -> bool {
        Self::d(a) <= Self::d(b)
    }
    #[inline(always)]
    fn gt(a: f32, b: f32) -> bool {
        Self::d(a) > Self::d(b)
    }
    #[inline(always)]
    fn ge(a: f32, b: f32) -> bool {
        Self::d(a) >= Self::d(b)
    }
    #[inline(always)]
    fn eq(a: f32, b: f32) -> bool {
        Self::d(a) == Self::d(b)
    }
}

/// IEEE-754 binary32 with subnormals (the CPU's own environment).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ieee;
impl FpEnv for Ieee {
    const FTZ: bool = false;
}

/// Flush-to-zero environment (Apple GPUs).
#[derive(Clone, Copy, Debug, Default)]
pub struct Ftz;
impl FpEnv for Ftz {
    const FTZ: bool = true;
}

#[cfg(test)]
mod tests {
    use super::*;

    const SUB: f32 = 1e-39;

    #[test]
    fn ieee_is_plain_arithmetic() {
        assert_eq!(Ieee::mul(SUB, 1.0), SUB);
        assert_eq!(Ieee::add(1.5e-38, -1.4e-38), 1.5e-38 - 1.4e-38);
        assert!(Ieee::lt(0.0, SUB));
    }

    /// The cases measured on an Apple M2 Pro (see the Metal backend tests).
    #[test]
    fn ftz_matches_measured_gpu() {
        assert_eq!(Ftz::mul(SUB, 1.0).to_bits(), 0);
        assert_eq!(Ftz::mul(-SUB, 1.0).to_bits(), (-0f32).to_bits());
        assert_eq!(Ftz::add(1.5e-38, -1.4e-38).to_bits(), 0);
        assert_eq!(Ftz::add(-1.5e-38, 1.4e-38).to_bits(), (-0f32).to_bits());
        assert_eq!(Ftz::fma(-1e-20, 1e-20, 0.0).to_bits(), (-0f32).to_bits());
        // 1e-38 is subnormal: the unfused product flushes, the fused sum too.
        assert_eq!(
            Ftz::add(Ftz::mul(1e-19, 1e-19), -1.175_494_4e-38),
            -1.175_494_4e-38
        );
        assert_eq!(
            Ftz::fma(1e-19, 1e-19, -1.175_494_4e-38).to_bits(),
            (-0f32).to_bits()
        );
        assert!(Ftz::div(1.0, SUB).is_infinite());
        assert!(Ftz::div(SUB, 0.0).is_nan());
        assert_eq!(Ftz::sqrt(SUB), 0.0);
        assert!(Ftz::le(SUB, 0.0) && Ftz::eq(-SUB, 0.0) && !Ftz::lt(0.0, SUB));
        // Normal operands with normal results are unaffected.
        assert_eq!(Ftz::add(1.0, 2f32.powi(-24)), 1.0 + 2f32.powi(-24));
        assert_eq!(Ftz::mul(f32::MIN_POSITIVE, 1.0), f32::MIN_POSITIVE);
    }
}
