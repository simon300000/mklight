// Numerics self-tests (numerics.wgsl's interface, numerics.metal's
// arithmetic): the floating-point behaviour probe and elementwise
// double-float operations, compared with the CPU-f32 emulation.
#include "mkl_cuda.h"

struct MklCountParams {
    uint n, op, reserved0, reserved1;
};

// Per triple MKL_PROBE_WORDS results of plain expressions at out[i * W],
// then the same operations as explicit IEEE intrinsics at out[(n + i) * W]
// (mklight_gpu::probe::expected's word order).
MKL_KERNEL mkl_numerics_probe(const MklCountParams* pc, const float* pa, const float* pb,
                              const float* pcc, float* pout) {
    const uint i = mkl_gid_x();
    const uint n = pc->n;
    if (i >= n) return;
    const float x = pa[i], y = pb[i], z = pcc[i];
    const uint w = MKL_PROBE_WORDS;
    uint o = i * w;
    pout[o + 0u] = x * y + z;
    pout[o + 1u] = fma(x, y, z);
    pout[o + 2u] = x / y;
    pout[o + 3u] = sqrt(fabs(x));
    pout[o + 4u] = x + y;
    pout[o + 5u] = x * y;
    pout[o + 6u] = (x * 8.470329472543003e-22f) * 8.470329472543003e-22f;
    pout[o + 7u] = (x + y) - y;
    o = (n + i) * w;
    const float p70 = MKL_POW2(-70);
    pout[o + 0u] = __fadd_rn(__fmul_rn(x, y), z);
    pout[o + 1u] = __fmaf_rn(x, y, z);
    pout[o + 2u] = __fdiv_rn(x, y);
    pout[o + 3u] = __fsqrt_rn(fabs(x));
    pout[o + 4u] = __fadd_rn(x, y);
    pout[o + 5u] = __fmul_rn(x, y);
    pout[o + 6u] = __fmul_rn(__fmul_rn(x, p70), p70);
    pout[o + 7u] = __fsub_rn(__fadd_rn(x, y), y);
}

// Elementwise double-float operation MKL_DFOP_* (pc.op) on (a[i], b[i]).
MKL_KERNEL mkl_df_ops(const MklCountParams* pc, const float2* da, const float2* db, float2* dout) {
    const uint i = mkl_gid_x();
    if (i >= pc->n) return;
    const float2 x = da[i], y = db[i];
    float2 r;
    switch (pc->op) {
    case MKL_DFOP_TWO_SUM: r = mkl_two_sum(x.x, y.x); break;
    case MKL_DFOP_FAST_TWO_SUM: r = mkl_fast_two_sum(x.x, y.x); break;
    case MKL_DFOP_TWO_PROD: r = mkl_two_prod(x.x, y.x); break;
    case MKL_DFOP_ADD: r = mkl_df_add(x, y); break;
    case MKL_DFOP_MUL: r = mkl_df_mul(x, y); break;
    case MKL_DFOP_DIV: r = mkl_df_div(x, y); break;
    case MKL_DFOP_SQRT: r = mkl_df_sqrt(x); break;
    case MKL_DFOP_ADD_F32: r = mkl_df_add_f32(x, y.x); break;
    case MKL_DFOP_MUL_F32: r = mkl_df_mul_f32(x, y.x); break;
    case MKL_DFOP_HYPOT2: r = float2(mkl_hypot2(x.x, y.x), 0.0f); break;
    default: r = float2(mkl_qnan()); break;
    }
    dout[i] = r;
}
