// Numerics self-tests and diagnostics: double-float operations, the
// floating-point behaviour probe, the packed-layout probe and the packed-sag
// probe. Results are compared with the CPU-f32 emulation and the f64
// reference by the crate's tests.
#include "mklight_common.h"

// Operation selector (MKL_DFOP_*), specialised per pipeline.
constant uint mkl_fc_df_op [[function_constant(0)]];

kernel void mkl_df_ops(device const float2* a [[buffer(0)]],
                       device const float2* b [[buffer(1)]],
                       device float2* out [[buffer(2)]],
                       constant uint& n [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float2 x = a[i];
    float2 y = b[i];
    float2 r;
    switch (mkl_fc_df_op) {
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
    out[i] = r;
}

// Floating-point behaviour probe. For each input triple (a, b, c) it writes
// MKL_PROBE_WORDS results that the host compares bit for bit with IEEE-754
// binary32 computed on the CPU:
//   0: a*b + c          (must NOT be contracted into an fma)
//   1: fma(a, b, c)     (must be a true single-rounding fma)
//   2: a / b            (correctly rounded division?)
//   3: sqrt(|a|)        (correctly rounded sqrt?)
//   4: a + b            (round-to-nearest-even add)
//   5: a * b            (round-to-nearest-even multiply)
//   6: (a 2^-70) 2^-70  (subnormal results preserved or flushed?)
//   7: (a + b) - b      (must NOT be reassociated to a)
kernel void mkl_numerics_probe(device const float* a [[buffer(0)]],
                               device const float* b [[buffer(1)]],
                               device const float* c [[buffer(2)]],
                               device float* out [[buffer(3)]],
                               constant uint& n [[buffer(4)]],
                               uint i [[thread_position_in_grid]]) {
    if (i >= n) return;
    float x = a[i], y = b[i], z = c[i];
    device float* o = out + i * MKL_PROBE_WORDS;
    o[0] = x * y + z;
    o[1] = fma(x, y, z);
    o[2] = x / y;
    o[3] = sqrt(fabs(x));
    o[4] = x + y;
    o[5] = x * y;
    o[6] = (x * MKL_POW2(-70)) * MKL_POW2(-70);
    o[7] = (x + y) - y;
}

// Sizes, alignments and field offsets of the shared structs as the Metal
// compiler lays them out, in the order of mklight_gpu::layout::{header_offsets,
// surface_offsets}.
#define MKL_OFF(T, f) (uint)__builtin_offsetof(T, f)
kernel void mkl_layout_probe(device uint* out [[buffer(0)]],
                             uint i [[thread_position_in_grid]]) {
    if (i != 0) return;
    uint k = 0;
    out[k++] = (uint)sizeof(MklPackedHeader);
    out[k++] = (uint)sizeof(MklPackedSurface);
    out[k++] = (uint)alignof(MklPackedHeader);
    out[k++] = (uint)alignof(MklPackedSurface);
    out[k++] = MKL_OFF(MklPackedHeader, magic);
    out[k++] = MKL_OFF(MklPackedHeader, layout_version);
    out[k++] = MKL_OFF(MklPackedHeader, num_surfaces);
    out[k++] = MKL_OFF(MklPackedHeader, num_wavelengths);
    out[k++] = MKL_OFF(MklPackedHeader, surface_offset);
    out[k++] = MKL_OFF(MklPackedHeader, stop);
    out[k++] = MKL_OFF(MklPackedHeader, primary_wavelength);
    out[k++] = MKL_OFF(MklPackedHeader, flags);
    out[k++] = MKL_OFF(MklPackedHeader, wavelength_um);
    out[k++] = MKL_OFF(MklPackedHeader, wavelength_source);
    out[k++] = MKL_OFF(MklPackedHeader, reserved);
    out[k++] = MKL_OFF(MklPackedSurface, shape);
    out[k++] = MKL_OFF(MklPackedSurface, interaction);
    out[k++] = MKL_OFF(MklPackedSurface, aperture);
    out[k++] = MKL_OFF(MklPackedSurface, flags);
    out[k++] = MKL_OFF(MklPackedSurface, curvature);
    out[k++] = MKL_OFF(MklPackedSurface, conic);
    out[k++] = MKL_OFF(MklPackedSurface, paraxial_power);
    out[k++] = MKL_OFF(MklPackedSurface, asphere_inv_norm2);
    out[k++] = MKL_OFF(MklPackedSurface, num_asphere);
    out[k++] = MKL_OFF(MklPackedSurface, source_index);
    out[k++] = MKL_OFF(MklPackedSurface, coating);
    out[k++] = MKL_OFF(MklPackedSurface, curvature_lo);
    out[k++] = MKL_OFF(MklPackedSurface, asphere);
    out[k++] = MKL_OFF(MklPackedSurface, aperture_params);
    out[k++] = MKL_OFF(MklPackedSurface, to_next_rot);
    out[k++] = MKL_OFF(MklPackedSurface, to_next_origin);
    out[k++] = MKL_OFF(MklPackedSurface, to_next_origin_lo);
    out[k++] = MKL_OFF(MklPackedSurface, coating_r);
    out[k++] = MKL_OFF(MklPackedSurface, coating_t);
    out[k++] = MKL_OFF(MklPackedSurface, conic_lo);
    out[k++] = MKL_OFF(MklPackedSurface, n_after);
    out[k++] = MKL_OFF(MklPackedSurface, alpha_after);
    out[k++] = MKL_OFF(MklPackedSurface, vergence);
    out[k++] = MKL_OFF(MklPackedSurface, to_next_rot_lo);
    out[k++] = MKL_OFF(MklPackedSurface, to_next_shift);
    out[k++] = MKL_OFF(MklPackedSurface, to_next_shift_lo);
    out[k++] = MKL_OFF(MklPackedSurface, reserved2);
    out[k++] = MKL_OFF(MklPackedSurface, n_after_lo);
}

// Sag of one packed surface (absolute index into the surface array) at the
// given r^2 values. Proves the uploaded layout is read correctly on the GPU.
kernel void mkl_sag_probe(constant MklPackedSurface* surfaces [[buffer(0)]],
                          constant MklSagParams& p [[buffer(1)]],
                          device const float* r2 [[buffer(2)]],
                          device float* out [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    out[i] = mkl_packed_sag(surfaces[p.surface], r2[i]);
}
