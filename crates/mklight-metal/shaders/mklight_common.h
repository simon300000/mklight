// mklight_common.h — numerics shared by every MKLight Metal kernel.
//
// Each function here is the MSL twin of a Rust function in crates/mklight-gpu
// (df.rs, fmath.rs, mklight_core::rng). They are written operation for
// operation alike, so the CPU-f32 emulation reproduces the GPU bit for bit.
// Change both sides together.
//
// Numerics rules (docs/CONVENTIONS.md §7):
//   * Compiled with fast-math OFF and FP contraction OFF (build.rs flags,
//     runtime MTLCompileOptions, and the pragmas below on MSL >= 3.2).
//     Fast-math reassociation folds the error term of two_sum, (s - a) - b,
//     to zero; implicit contraction changes a*b + c into fma(a, b, c) and
//     breaks bitwise agreement with the CPU. Fusion is always written
//     explicitly with fma().
//   * The backend runs mkl_numerics_probe at start-up and refuses double-float
//     work if these assumptions do not hold on the device.
//   * Exact float equality only against the sentinels listed in
//     mklight_gpu_types.h (packed exact zeros).
//   * Apple GPUs flush subnormals to zero (operands and results of arithmetic,
//     operands of comparisons). mklight_gpu::fenv::Ftz models exactly that, so
//     the CPU emulation stays bit-identical; never rely on gradual underflow.
#ifndef MKLIGHT_COMMON_H
#define MKLIGHT_COMMON_H

#include <metal_stdlib>
#include "mklight_gpu_types.h"

using namespace metal;

#if defined(__METAL_VERSION__) && __METAL_VERSION__ >= 320
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#endif

// Exact powers of two from their bit patterns: 2^e = (e + 127) << 23.
#define MKL_POW2(e) as_type<float>((uint)((e) + 127) << 23)

static inline float mkl_qnan() { return as_type<float>(0x7FC00000u); }

// ---------------------------------------------------------------------------
// Error-free transforms (double-float building blocks)
// ---------------------------------------------------------------------------

// MKLIGHT_PURE_F32 (opt-in measurement build, docs/GPU.md §3.6): every
// double-float operation collapses to plain f32 with a zero lo part, and
// compensated sums become plain sums. The packed ABI and the algorithms are
// unchanged. mkl_df_input / mkl_df_low drop lo parts where kernels load or
// read them directly (only for hi+lo values: float2 is also complex/xy data).
#ifdef MKLIGHT_PURE_F32
static inline float2 mkl_df_input(float2 x) { return float2(x.x, 0); }
static inline float mkl_df_low(float x) { (void)x; return 0; }
static inline float2 mkl_two_sum(float a, float b) { return float2(a + b, 0); }
static inline float2 mkl_fast_two_sum(float a, float b) { return float2(a + b, 0); }
static inline float2 mkl_two_prod(float a, float b) { return float2(a * b, 0); }
static inline float2 mkl_df_add_f32(float2 x, float y) { return float2(x.x + y, 0); }
static inline float2 mkl_df_add(float2 x, float2 y) { return float2(x.x + y.x, 0); }
static inline float2 mkl_df_neg(float2 x) { return float2(-x.x, 0); }
static inline float2 mkl_df_sub(float2 x, float2 y) { return float2(x.x - y.x, 0); }
static inline float2 mkl_df_mul_f32(float2 x, float y) { return float2(x.x * y, 0); }
static inline float2 mkl_df_mul(float2 x, float2 y) { return float2(x.x * y.x, 0); }
static inline float2 mkl_df_div(float2 x, float2 y) { return float2(x.x / y.x, 0); }
static inline float2 mkl_df_sqrt(float2 x) { return float2(sqrt(x.x), 0); }
static inline void mkl_neumaier_add(thread float& s, thread float& c, float x) { s += x; c = 0; }
#else
static inline float2 mkl_df_input(float2 x) { return x; }
static inline float mkl_df_low(float x) { return x; }

// s + t == a + b exactly (Knuth 2Sum, no precondition).
static inline float2 mkl_two_sum(float a, float b) {
    float s = a + b;
    float a1 = s - b;
    float b1 = s - a1;
    float da = a - a1;
    float db = b - b1;
    return float2(s, da + db);
}

// s + t == a + b exactly when |a| >= |b| or a == 0 (Dekker Fast2Sum).
static inline float2 mkl_fast_two_sum(float a, float b) {
    float s = a + b;
    float z = s - a;
    return float2(s, b - z);
}

// p + e == a * b exactly (barring underflow), using the hardware FMA.
static inline float2 mkl_two_prod(float a, float b) {
    float p = a * b;
    return float2(p, fma(a, b, -p));
}

// ---------------------------------------------------------------------------
// Double-float arithmetic: x = (hi, lo) = x.x + x.y, |lo| <= ulp(hi)/2.
// Algorithms: Joldes, Muller, Popescu, ACM TOMS 44(2), 2017 (see df.rs for
// the error bounds).
// ---------------------------------------------------------------------------

// DWPlusFP (Alg. 4), relative error <= 2u^2.
static inline float2 mkl_df_add_f32(float2 x, float y) {
    float2 s = mkl_two_sum(x.x, y);
    float v = x.y + s.y;
    return mkl_fast_two_sum(s.x, v);
}

// AccurateDWPlusDW (Alg. 6), relative error <= 3u^2 + 13u^3.
static inline float2 mkl_df_add(float2 x, float2 y) {
    float2 s = mkl_two_sum(x.x, y.x);
    float2 t = mkl_two_sum(x.y, y.y);
    float c = s.y + t.x;
    float2 v = mkl_fast_two_sum(s.x, c);
    float w = t.y + v.y;
    return mkl_fast_two_sum(v.x, w);
}

static inline float2 mkl_df_neg(float2 x) { return float2(-x.x, -x.y); }
static inline float2 mkl_df_sub(float2 x, float2 y) { return mkl_df_add(x, mkl_df_neg(y)); }

// DWTimesFP3 (Alg. 9), relative error <= 2u^2.
static inline float2 mkl_df_mul_f32(float2 x, float y) {
    float2 c = mkl_two_prod(x.x, y);
    float cl3 = fma(x.y, y, c.y);
    return mkl_fast_two_sum(c.x, cl3);
}

// DWTimesDW3 (Alg. 12), relative error <= 4u^2.
static inline float2 mkl_df_mul(float2 x, float2 y) {
    float2 c = mkl_two_prod(x.x, y.x);
    float tl0 = x.y * y.y;
    float tl1 = fma(x.x, y.y, tl0);
    float cl2 = fma(x.y, y.x, tl1);
    float cl3 = c.y + cl2;
    return mkl_fast_two_sum(c.x, cl3);
}

// DWDivDW2 (Alg. 17), relative error <= 15u^2 + 56u^3. The first quotient
// only needs to be within a few ulp; x.hi - r.hi is exact (Sterbenz).
static inline float2 mkl_df_div(float2 x, float2 y) {
    float th = x.x / y.x;
    float2 r = mkl_df_mul_f32(y, th);
    float pi_h = x.x - r.x;
    float dl = x.y - r.y;
    float d = pi_h + dl;
    float tl = d / y.x;
    return mkl_fast_two_sum(th, tl);
}

// sqrt with one Newton (Karp) correction from an exact residual.
static inline float2 mkl_df_sqrt(float2 x) {
    if (x.x <= 0.0f) {
        return x.x == 0.0f ? float2(0.0f) : float2(mkl_qnan());
    }
    float s = sqrt(x.x);
    float2 p = mkl_two_prod(s, s);
    float r = ((x.x - p.x) - p.y) + x.y;
    float e = r / (2.0f * s);
    return mkl_fast_two_sum(s, e);
}

// ---------------------------------------------------------------------------
// Compensated summation (CONVENTIONS §7.5)
// ---------------------------------------------------------------------------

// Neumaier's improved Kahan step; the result is two_sum(s, c).
static inline void mkl_neumaier_add(thread float& s, thread float& c, float x) {
    float t = s + x;
    if (fabs(s) >= fabs(x)) {
        c += (s - t) + x;
    } else {
        c += (x - t) + s;
    }
    s = t;
}

#endif // MKLIGHT_PURE_F32

// ---------------------------------------------------------------------------
// Philox4x32-10 (CONVENTIONS §6): exactly mklight_core::rng::philox4x32_10.
// ---------------------------------------------------------------------------
static inline uint4 mkl_philox4x32_10(uint4 c, uint2 k) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            k.x += 0x9E3779B9u;
            k.y += 0xBB67AE85u;
        }
        uint hi0 = mulhi(0xD2511F53u, c.x);
        uint lo0 = 0xD2511F53u * c.x;
        uint hi1 = mulhi(0xCD9E8D57u, c.z);
        uint lo1 = 0xCD9E8D57u * c.z;
        c = uint4(hi1 ^ c.y ^ k.x, lo1, hi0 ^ c.w ^ k.y, lo0);
    }
    return c;
}

// Uniform in (0, 1) from the top 24 bits: (k + 1/2) 2^-24 rounded toward zero
// to f32 (mklight_gpu::fmath::u32_to_unit_f32). Every operation is exact.
static inline float mkl_u32_to_unit(uint x) {
    uint k = x >> 8;
    float half_ulp = k < 0x800000u ? MKL_POW2(-25) : 0.0f;
    return fma(float(k), MKL_POW2(-24), half_ulp);
}

// ---------------------------------------------------------------------------
// Utility math (CONVENTIONS §7.3, §7.6)
// ---------------------------------------------------------------------------

// Same nesting as mklight_core::Vec3::dot.
static inline float mkl_dot3(float3 a, float3 b) {
    return fma(a.x, b.x, fma(a.y, b.y, a.z * b.z));
}

// Power-of-two scaling into [2^-60, 2^60] so squares cannot overflow or
// underflow; the scaling itself is exact.
static inline float2 mkl_hypot_scale(float m) {
    if (m > MKL_POW2(60)) return float2(MKL_POW2(-80), MKL_POW2(80));
    if (m < MKL_POW2(-60)) return float2(MKL_POW2(80), MKL_POW2(-80));
    return float2(1.0f, 1.0f);
}

// IEEE hypot special values: any infinite component gives +inf (even with a
// NaN), otherwise any NaN gives NaN (max() drops NaN, hence the explicit
// test), and all zeros give +0. Constants are returned, never a raw input, so
// flush-to-zero cannot pass a subnormal through. Returns -1 when m is regular.
static inline float mkl_hypot_special(float m, bool any_nan) {
    if (m == INFINITY) return INFINITY;
    if (any_nan) return mkl_qnan();
    if (m == 0.0f) return 0.0f;
    return -1.0f;
}

static inline float mkl_hypot2(float x, float y) {
    float m = max(fabs(x), fabs(y));
    float sp = mkl_hypot_special(m, isnan(x) || isnan(y));
    if (!(sp < 0.0f)) return sp;
    float2 sc = mkl_hypot_scale(m);
    float xs = x * sc.x, ys = y * sc.x;
    return sqrt(fma(xs, xs, ys * ys)) * sc.y;
}

static inline float mkl_hypot3(float x, float y, float z) {
    float m = max(max(fabs(x), fabs(y)), fabs(z));
    float sp = mkl_hypot_special(m, isnan(x) || isnan(y) || isnan(z));
    if (!(sp < 0.0f)) return sp;
    float2 sc = mkl_hypot_scale(m);
    float3 v = float3(x * sc.x, y * sc.x, z * sc.x);
    return sqrt(mkl_dot3(v, v)) * sc.y;
}

// e^x in f32 (mklight_gpu::fmath::exp): k = rint(x log2 e), Cody-Waite
// reduction with fma, degree-7 Taylor polynomial, two exact power-of-two
// factors. Platform exp() is not used: its results differ between devices.
static inline float mkl_exp(float x) {
    if (isnan(x)) return x;
    if (x > 88.72284f) return INFINITY;
    if (x < -103.97208f) return 0.0f;
    float kf = rint(x * as_type<float>(0x3FB8AA3Bu));
    int k = int(kf);
    float r = fma(-kf, as_type<float>(0x3F317200u), x);
    r = fma(-kf, as_type<float>(0x35BFBE8Eu), r);
    float p = as_type<float>(0x39500D01u);
    p = fma(p, r, as_type<float>(0x3AB60B61u));
    p = fma(p, r, as_type<float>(0x3C088889u));
    p = fma(p, r, as_type<float>(0x3D2AAAABu));
    p = fma(p, r, as_type<float>(0x3E2AAAABu));
    p = fma(p, r, 0.5f);
    p = fma(p, r, 1.0f);
    p = fma(p, r, 1.0f);
    int k1 = k / 2;
    return (p * MKL_POW2(k1)) * MKL_POW2(k - k1);
}

// ---------------------------------------------------------------------------
// Surface sag (packed layout, see mklight_gpu_types.h)
// ---------------------------------------------------------------------------

// Cancellation-free conic sag c r^2 / (1 + sqrt(1 - (1+k) c^2 r^2)); NaN
// outside the conic's domain.
static inline float mkl_conic_sag(float c, float k, float r2) {
    float cr2 = c * r2;
    float ck = (1.0f + k) * c;
    float arg = fma(-ck, cr2, 1.0f);
    if (isnan(arg) || arg < 0.0f) return mkl_qnan();
    return cr2 / (1.0f + sqrt(arg));
}

static inline float mkl_packed_sag(MklPackedSurface s, float r2) {
    if (s.shape == MKL_SHAPE_CONIC) {
        return mkl_conic_sag(s.curvature, s.conic, r2);
    }
    if (s.shape == MKL_SHAPE_EVEN_ASPHERE) {
        float base = mkl_conic_sag(s.curvature, s.conic, r2);
        uint n = s.num_asphere;
        if (n == 0u) return base;
        float rho2 = r2 * s.asphere_inv_norm2;
        float acc = s.asphere[n - 1u];
        for (uint j = n - 1u; j > 0u; --j) {
            acc = fma(acc, rho2, s.asphere[j - 1u]);
        }
        float rho4 = rho2 * rho2;
        return fma(acc, rho4, base);
    }
    return 0.0f;
}

#endif // MKLIGHT_COMMON_H
