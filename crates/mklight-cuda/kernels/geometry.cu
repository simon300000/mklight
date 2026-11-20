// Resident geometric algebra (geometry::Request::{Coordinates, FocusTerms,
// Characteristic}): geometry.wgsl's interface and arithmetic, operation for
// operation (mklight_gpu::geometry::{cpu, spec}). Divisions are IEEE here,
// so mkl_df_div is geometry.wgsl's mkl_df_div_rn; the characteristic
// function's cos/sin are geometry.wgsl's double-float evaluation.
//
// Arguments: parameters (binding 1), in0..in4 (u32 words), out0..out1
// (u32 words). Thread `tid` of a pass handles element `offset + tid`.
#include "mkl_cuda.h"

struct MklWgslGeometryParams {
    uint count, offset;
    uint rays, chiefs;
    uint segment, centroid;
    uint reserved0, reserved1;
    float defocus[2];
};

#define GEO_ARGS                                                                      \
    const MklWgslGeometryParams *u, const uint *g_in0, const uint *g_in1,              \
        const uint *g_in2, const uint *g_in3, const uint *g_in4, uint *g_out0, uint *g_out1

__device__ inline float geo_f(const uint* w, uint i) { return __uint_as_float(w[i]); }

__device__ inline void geo_put(uint* out, uint i, float2 v) {
    out[2u * i] = __float_as_uint(v.x);
    out[2u * i + 1u] = __float_as_uint(v.y);
}

// The double-float element i of a DF word array.
__device__ inline float2 geo_df(const uint* w, uint i) {
    return float2(geo_f(w, 2u * i), geo_f(w, 2u * i + 1u));
}

// ---------------------------------------------------------------------------
// Coordinates: in0 hit floats [15, rays], in1 hit status, in2 chief floats
// [15, chiefs], in3 chief status, in4 chief index; out0 DF [5, rays], out1
// mask.
// ---------------------------------------------------------------------------

// Hit float plane k and k + 9 (hi, lo) of ray i of n.
__device__ inline float2 geo_hit(const uint* f, uint n, uint i, uint k) {
    return float2(geo_f(f, k * n + i), geo_f(f, (k + 9u) * n + i));
}

MKL_KERNEL mkl_geometry_coordinates(GEO_ARGS) {
    const MklWgslGeometryParams p = *u;
    const uint gid = mkl_gid_x();
    if (gid >= p.count) return;
    const uint i = p.offset + gid;
    const uint n = p.rays;
    const uint nc = p.chiefs;
    const bool ok = g_in1[i] == 0u;
    const uint j = g_in4[i];
    const bool cok = j < nc && g_in3[j] == 0u;
    g_out1[i] = (uint)ok;
    for (uint k = 0u; k < 5u; k++) geo_put(g_out0, k * n + i, float2(0.0f));
    if (!ok) return;
    const float2 dz = geo_hit(g_in0, n, i, 5u);
    float2 c[4] = {float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f)};
    if (cok) {
        const float2 cdz = geo_hit(g_in2, nc, j, 5u);
        c[0] = geo_hit(g_in2, nc, j, 0u);
        c[1] = geo_hit(g_in2, nc, j, 1u);
        c[2] = mkl_df_div(geo_hit(g_in2, nc, j, 3u), cdz);
        c[3] = mkl_df_div(geo_hit(g_in2, nc, j, 4u), cdz);
    }
    const float2 defocus = float2(p.defocus[0], p.defocus[1]);
    for (uint k = 0u; k < 2u; k++) {
        const float2 t = mkl_df_div(geo_hit(g_in0, n, i, k + 3u), dz);
        const float2 dt = mkl_df_sub(t, c[k + 2u]);
        const float2 q = mkl_df_add(mkl_df_sub(geo_hit(g_in0, n, i, k), c[k]), mkl_df_mul(defocus, dt));
        geo_put(g_out0, k * n + i, q);
        geo_put(g_out0, (k + 2u) * n + i, dt);
    }
    geo_put(g_out0, 4u * n + i, float2(geo_f(g_in0, 8u * n + i), 0.0f));
}

// ---------------------------------------------------------------------------
// FocusTerms: in0 points DF [5, rays], in1 mask, in2 position moments DF
// [segments, 10], in3 slope moments; out0 DF [3, rays].
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_geometry_focus_terms(GEO_ARGS) {
    const MklWgslGeometryParams p = *u;
    const uint gid = mkl_gid_x();
    if (gid >= p.count) return;
    const uint i = p.offset + gid;
    const uint n = p.rays;
    for (uint k = 0u; k < 3u; k++) geo_put(g_out0, k * n + i, float2(0.0f));
    if (g_in1[i] == 0u) return;
    const uint row = (i / p.segment) * 10u;
    float2 q[2], t[2];
    for (uint k = 0u; k < 2u; k++) {
        float2 qc = float2(0.0f), tc = float2(0.0f);
        if (p.centroid != 0u) {
            qc = geo_df(g_in2, row + 2u + k);
            tc = geo_df(g_in3, row + 2u + k);
        }
        q[k] = mkl_df_sub(geo_df(g_in0, k * n + i), qc);
        t[k] = mkl_df_sub(geo_df(g_in0, (k + 2u) * n + i), tc);
    }
    const float2 w = geo_df(g_in0, 4u * n + i);
    geo_put(g_out0, i, mkl_df_mul(mkl_df_add(mkl_df_mul(q[0], q[0]), mkl_df_mul(q[1], q[1])), w));
    geo_put(g_out0, n + i, mkl_df_mul(mkl_df_add(mkl_df_mul(q[0], t[0]), mkl_df_mul(q[1], t[1])), w));
    geo_put(g_out0, 2u * n + i, mkl_df_mul(mkl_df_add(mkl_df_mul(t[0], t[0]), mkl_df_mul(t[1], t[1])), w));
}

// ---------------------------------------------------------------------------
// Characteristic: in0 points DF [5, rays], in1 mask, in2 frequencies DF
// [frequencies, segments, 2]; out0 DF [2, frequencies · rays] (the second
// plane starts at element p.chiefs = frequencies · rays).
// ---------------------------------------------------------------------------

// f32 nearest to x, ties away from zero (Rust f32::round).
__device__ inline float geo_round(float x) {
    const float t = truncf(x);
    if (fabsf(x - t) >= 0.5f) return t + (x > 0.0f ? 1.0f : -1.0f);
    return t;
}

// cos and sin of |x| <= 4 in double-float: r = x - k pi/2 with a three-part
// pi/2 (k·P exact), Taylor polynomials through r^18 / r^19 on |r| <= pi/4 in
// Horner form, then the quadrant's signs and swap; the high words are the
// rounded results.
__constant__ const float GEO_COS[10][2] = {
    {0x1p+0f, 0.0f},
    {-0x1p-1f, 0.0f},
    {0x1.555556p-5f, -0x1.555556p-30f},
    {-0x1.6c16c2p-10f, 0x1.27d27ep-35f},
    {0x1.a01a02p-16f, -0x1.7f97fap-42f},
    {-0x1.27e4fcp-22f, 0x1.10ec14p-47f},
    {0x1.1eed8ep-29f, 0x1.ff1b12p-54f},
    {-0x1.93974ap-37f, -0x1.180f94p-62f},
    {0x1.ae7f3ep-45f, 0x1.ccee08p-71f},
    {-0x1.682786p-53f, -0x1.dcbeccp-80f},
};
__constant__ const float GEO_SIN[10][2] = {
    {0x1p+0f, 0.0f},
    {-0x1.555556p-3f, 0x1.555556p-28f},
    {0x1.111112p-7f, -0x1.dddddep-32f},
    {-0x1.a01a02p-13f, 0x1.7f97fap-39f},
    {0x1.71de3ap-19f, 0x1.55b1ccp-45f},
    {-0x1.ae6456p-26f, -0x1.fd5138p-52f},
    {0x1.612462p-33f, -0x1.8af25ep-58f},
    {-0x1.ae7f3ep-41f, -0x1.ccee08p-67f},
    {0x1.952c78p-49f, -0x1.f9ea56p-74f},
    {-0x1.2f49b4p-57f, -0x1.a05056p-83f},
};

__device__ inline float2 geo_cos_sin(float x) {
    const float k = geo_round(x * __uint_as_float(0x3f22f983u));
    float2 r = mkl_two_sum(x, -(k * __uint_as_float(0x3fc90fdbu)));
    r = mkl_df_add_f32(r, -(k * __uint_as_float(0xb33bbd2eu)));
    r = mkl_df_add_f32(r, -(k * __uint_as_float(0xa6f72cedu)));
    const float2 r2 = mkl_df_mul(r, r);
    float2 c = float2(GEO_COS[9][0], GEO_COS[9][1]);
    float2 s = float2(GEO_SIN[9][0], GEO_SIN[9][1]);
    for (int j = 8; j >= 0; j--) {
        c = mkl_df_add(float2(GEO_COS[j][0], GEO_COS[j][1]), mkl_df_mul(r2, c));
        s = mkl_df_add(float2(GEO_SIN[j][0], GEO_SIN[j][1]), mkl_df_mul(r2, s));
    }
    s = mkl_df_mul(r, s);
    const int q = ((int)k % 4 + 4) % 4;
    if (q == 1) return float2(-s.x, c.x);
    if (q == 2) return float2(-c.x, -s.x);
    if (q == 3) return float2(s.x, -c.x);
    return float2(c.x, s.x);
}

MKL_KERNEL mkl_geometry_characteristic(GEO_ARGS) {
    const MklWgslGeometryParams p = *u;
    const uint gid = mkl_gid_x();
    if (gid >= p.count) return;
    const uint i = p.offset + gid;
    const uint nr = p.rays;
    const uint total = p.chiefs;
    const uint ray = i % nr;
    const uint frequency = i / nr;
    geo_put(g_out0, i, float2(0.0f));
    geo_put(g_out0, total + i, float2(0.0f));
    if (g_in1[ray] == 0u) return;
    const uint j = 2u * (frequency * (nr / p.segment) + ray / p.segment);
    float2 a = mkl_df_add(mkl_df_mul(geo_df(g_in0, ray), geo_df(g_in2, j)),
                          mkl_df_mul(geo_df(g_in0, nr + ray), geo_df(g_in2, j + 1u)));
    a = mkl_df_sub(a, float2(geo_round(a.x), 0.0f));
    const float2 tau = float2(__uint_as_float(0x40c90fdbu), __uint_as_float(0xb43bbd2eu));
    const float phase = mkl_df_mul(a, tau).x;
    const float2 w = geo_df(g_in0, 4u * nr + ray);
    const float2 cs = geo_cos_sin(phase);
    geo_put(g_out0, i, mkl_df_mul_f32(w, cs.x));
    geo_put(g_out0, total + i, mkl_df_mul_f32(w, cs.y));
}
