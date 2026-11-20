// Regular transforms: FFT rows and stages, chirp-z, transpose, elementwise
// chirp/phase/spectral steps and the correlation stencils (transforms.wgsl's
// interface and arithmetic, operation for operation; the steps of
// mklight_gpu::transforms::recipe and the CPU-f32 specification in
// mklight-gpu/src/transforms/cpu.rs, correlation.rs).
//
// No pass has a uniform block (`u` is null): the MklTransformParams /
// MklCorrelationParams are the first storage binding (binding 2), the other
// buffers follow in the host's binding order.
//
// Numerics: as transforms.wgsl, every complex product, sum and scale is one
// IEEE operation in the specification's order (common.wgsl's protected
// operations are plain IEEE operations here, mkl_cuda.h). Twiddles and fused
// chirp factors are f64-derived host tables; phases formed on the device are
// double-float turns evaluated by p2_cis's own polynomial: no device sin/cos.
#include "mkl_cuda.h"

typedef MklTransformParams P2Params;
typedef MklCorrelationParams P2CParams;

#define P2_KERNEL MKL_KERNEL __launch_bounds__(256)

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

__device__ inline bool p2_is_finite(float x) {
    return (__float_as_uint(x) & 0x7F800000u) != 0x7F800000u;
}

__device__ inline float2 p2_cmul(float2 a, float2 b) {
    return float2(mkl_sub(mkl_mul(a.x, b.x), mkl_mul(a.y, b.y)),
                  mkl_add(mkl_mul(a.x, b.y), mkl_mul(a.y, b.x)));
}
__device__ inline float2 p2_cadd(float2 a, float2 b) {
    return float2(mkl_add(a.x, b.x), mkl_add(a.y, b.y));
}
__device__ inline float2 p2_csub(float2 a, float2 b) {
    return float2(mkl_sub(a.x, b.x), mkl_sub(a.y, b.y));
}
__device__ inline float2 p2_cscale(float2 a, float s) {
    return float2(mkl_mul(a.x, s), mkl_mul(a.y, s));
}
// v·(s·i), exact for s = ±1.
__device__ inline float2 p2_rot(float2 v, float s) {
    return float2(-s * v.y, s * v.x);
}

// exp(2πi·a) for double-float turns a: reduced modulo a turn (twice, ties to
// even), then to an octant r = f - q/4, |r| ≤ 1/8, and evaluated by Taylor
// polynomials of degree 9 (sin) and 10 (cos) in x = 2πr, rotated by q
// quarter turns.
__device__ inline float2 p2_cis(float2 a_in) {
    float2 a = mkl_df_add_f32(a_in, -rintf(a_in.x));
    a = mkl_df_add_f32(a, -rintf(a.x));
    const float q = rintf(mkl_mul(a.x, 4.0f));
    // a.x - q/4 is exact (Sterbenz) for |a.x| ≤ 1/2.
    const float r = mkl_add(mkl_sub(a.x, mkl_mul(q, 0.25f)), a.y);
    const float x = mkl_mul(r, 6.2831853071795864769f);
    const float x2 = mkl_mul(x, x);
    float s = fmaf(x2, 2.7557319e-6f, -1.9841270e-4f);
    s = fmaf(s, x2, 8.3333333e-3f);
    s = fmaf(s, x2, -1.6666667e-1f);
    s = fmaf(mkl_mul(s, x2), x, x);
    float c = fmaf(x2, -2.7557319e-7f, 2.4801587e-5f);
    c = fmaf(c, x2, -1.3888889e-3f);
    c = fmaf(c, x2, 4.1666668e-2f);
    c = fmaf(c, x2, -0.5f);
    c = fmaf(c, x2, 1.0f);
    const int k = ((int)q % 4 + 4) % 4;
    switch (k) {
    case 1: return float2(-s, c);
    case 2: return float2(-c, -s);
    case 3: return float2(s, -c);
    default: return float2(c, s);
    }
}

// ---------------------------------------------------------------------------
// Elementwise steps (MKL_TRANSFORM_KERNEL/PRE/POST/MULTIPLY/PHASE) and the
// table-driven chirp factors of long chirp rows. Storage: params, a, b, out.
// ---------------------------------------------------------------------------

__device__ inline float2 p2_aff(const P2Params& p, uint i) {
    return float2(p.affine[i][0], p.affine[i][1]);
}
__device__ inline float p2_scale(const P2Params& p) { return __uint_as_float(p.scale_bits); }
__device__ inline float p2_sign(const P2Params& p) { return (float)(int)p.sign_bits; }

// 0.5·(a0·x² + 2·a1·x·y + a4·y²) in double-float turns (cpu::quadratic).
__device__ inline float2 p2_q(const P2Params& p, float x, float y) {
    const float2 t0 = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(p, 0u), x), x);
    const float2 t1 = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(p, 1u), mkl_mul(2.0f, x)), y);
    const float2 t2 = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(p, 4u), y), y);
    return mkl_df_mul_f32(mkl_df_add(mkl_df_add(t0, t1), t2), 0.5f);
}
// cpu::pre: q + a2·x + a5·y, left to right.
__device__ inline float2 p2_pre(const P2Params& p, float x, float y) {
    const float2 t = mkl_df_add(p2_q(p, x, y), mkl_df_mul_f32(p2_aff(p, 2u), x));
    return mkl_df_add(t, mkl_df_mul_f32(p2_aff(p, 5u), y));
}

// The thread's element of a 1-D grid launched as (x, y) blocks of 256.
__device__ inline uint p2_linear() {
    return (blockIdx.y * gridDim.x + blockIdx.x) * 256u + threadIdx.x;
}

P2_KERNEL mkl_transform(const void*, const P2Params* pp, const float2* e_a, const float2* e_b,
                        float2* e_out) {
    const P2Params& p = *pp;
    const uint id = p2_linear();
    if (id >= p.count) return;
    const uint w = p.width;
    const uint h = p.height;
    const float scale = p2_scale(p);
    const float sign = p2_sign(p);
    float2 z = float2(0.0f, 0.0f);
    switch (p.mode) {
    case MKL_TRANSFORM_KERNEL: {
        const uint ix = id % w;
        const uint iy = id / w;
        if ((ix < p.ox || ix > w - p.nx) && (iy < p.oy || iy > h - p.ny)) {
            float x = (float)ix;
            if (ix >= p.ox) x = mkl_sub((float)ix, (float)w);
            float y = (float)iy;
            if (iy >= p.oy) y = mkl_sub((float)iy, (float)h);
            z = p2_cis(mkl_df_mul_f32(p2_q(p, x, y), -sign));
        }
        break;
    }
    case MKL_TRANSFORM_PRE: {
        const uint x = id % w;
        const uint y = id / w;
        if (x < p.nx && y < p.ny)
            z = p2_cmul(e_a[y * p.nx + x], p2_cis(mkl_df_mul_f32(p2_pre(p, (float)x, (float)y), sign)));
        break;
    }
    case MKL_TRANSFORM_POST: {
        const uint x = id % p.ox;
        const uint y = id / p.ox;
        const float2 v = p2_cmul(e_a[y * w + x], p2_cis(mkl_df_mul_f32(p2_q(p, (float)x, (float)y), sign)));
        z = p2_cscale(v, scale);
        break;
    }
    case MKL_TRANSFORM_MULTIPLY: {
        uint j = id;
        if ((p.flags & 2u) != 0u) j = id % w;
        z = p2_cscale(p2_cmul(e_a[id], e_b[j]), scale);
        break;
    }
    case MKL_TRANSFORM_PHASE: {
        const float x = (float)(id % w);
        const float y = (float)(id / w);
        float2 ph = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(p, 0u), x), x);
        ph = mkl_df_add(ph, mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(p, 1u), x), y));
        ph = mkl_df_add(ph, mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(p, 2u), y), y));
        ph = mkl_df_add(ph, mkl_df_mul_f32(p2_aff(p, 3u), x));
        ph = mkl_df_add(ph, mkl_df_mul_f32(p2_aff(p, 4u), y));
        ph = mkl_df_add(ph, p2_aff(p, 5u));
        z = p2_cmul(e_a[id], p2_cis(ph));
        break;
    }
    default: break;
    }
    e_out[id] = z;
}

// Pre-chirp of long chirp rows (padded to width, count = width·rows):
// a[row·nx + i]·chirp[i] for i < nx, else 0. b = recipe::chirp_factors.
P2_KERNEL mkl_chirp_pre(const void*, const P2Params* pp, const float2* e_a, const float2* e_b,
                        float2* e_out) {
    const P2Params& p = *pp;
    const uint id = p2_linear();
    if (id >= p.count) return;
    const uint i = id % p.width;
    const uint row = id / p.width;
    float2 z = float2(0.0f, 0.0f);
    if (i < p.nx) z = p2_cmul(e_a[row * p.nx + i], e_b[i]);
    e_out[id] = z;
}

// Post-chirp of long chirp rows (count = ox·rows):
// a[row·width + i]·chirp[nx + i]·scale.
P2_KERNEL mkl_chirp_post(const void*, const P2Params* pp, const float2* e_a, const float2* e_b,
                         float2* e_out) {
    const P2Params& p = *pp;
    const uint id = p2_linear();
    if (id >= p.count) return;
    const uint i = id % p.ox;
    const uint row = id / p.ox;
    e_out[id] = p2_cscale(p2_cmul(e_a[row * p.width + i], e_b[p.nx + i]), p2_scale(p));
}

// ---------------------------------------------------------------------------
// Transpose: 32×32 tiles, 256 threads; the padded column avoids bank
// conflicts. Storage: params, a, out.
// ---------------------------------------------------------------------------

P2_KERNEL mkl_transform_transpose(const void*, const P2Params* pp, const float2* t_a, float2* t_out) {
    __shared__ float2 tile[32][33];
    const P2Params& p = *pp;
    const uint w = p.width;
    const uint h = p.height;
    const uint tid = threadIdx.x;
    const uint tx = tid % 32u;
    const uint ty = tid / 32u;
    const uint bx = blockIdx.x * 32u;
    const uint by = blockIdx.y * 32u;
    const float scale = p2_scale(p);
    for (uint j = ty; j < 32u; j += 8u)
        if (bx + tx < w && by + j < h) tile[j][tx] = t_a[(by + j) * w + bx + tx];
    __syncthreads();
    for (uint j = ty; j < 32u; j += 8u)
        if (bx + j < w && by + tx < h) t_out[(bx + j) * h + by + tx] = p2_cscale(tile[tx][j], scale);
}

// ---------------------------------------------------------------------------
// Stockham FFT. Twiddles are f64-derived host tables (recipe::twiddles):
// stage span p, radix R, entry p-1+k(R-1)+t-1 = exp(-2πi·t·k/(pR)); sign +1
// conjugates. Storage: params, twiddles, a, out (, spectrum, chirp).
// ---------------------------------------------------------------------------

// cos/sin(2π·q/R) for R = 3, 5, 7 (rows), from f64 rounded to f32 as the
// specification's generic DFT.
__constant__ float P2_COS[24] = {
    1.0f, -0.5f, -0.5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.309017003f, -0.809017003f, -0.809017003f, 0.309017003f, 0.0f, 0.0f, 0.0f,
    1.0f, 0.623489797f, -0.222520933f, -0.90096885f, -0.90096885f, -0.222520933f, 0.623489797f, 0.0f,
};
__constant__ float P2_SIN[24] = {
    0.0f, 0.866025388f, -0.866025388f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.95105654f, 0.587785244f, -0.587785244f, -0.95105654f, 0.0f, 0.0f, 0.0f,
    0.0f, 0.781831503f, 0.974927902f, 0.433883727f, -0.433883727f, -0.974927902f, -0.781831503f, 0.0f,
};
#define P2_SQRT1_2 0.707106769f

__device__ inline void p2_dft4(float2* u, uint i0, uint i1, uint i2, uint i3, float s) {
    const float2 a0 = p2_cadd(u[i0], u[i2]);
    const float2 a1 = p2_csub(u[i0], u[i2]);
    const float2 a2 = p2_cadd(u[i1], u[i3]);
    const float2 a3 = p2_rot(p2_csub(u[i1], u[i3]), s);
    u[i0] = p2_cadd(a0, a2);
    u[i2] = p2_csub(a0, a2);
    u[i1] = p2_cadd(a1, a3);
    u[i3] = p2_csub(a1, a3);
}

// Unnormalised DFT of r values in u[0..8), exp(s·2πi·m·t/r) (cpu::dft).
__device__ inline void p2_dft(float2* u, uint r, float s) {
    switch (r) {
    case 2u: {
        const float2 a = u[0];
        u[0] = p2_cadd(a, u[1]);
        u[1] = p2_csub(a, u[1]);
        break;
    }
    case 4u: p2_dft4(u, 0u, 1u, 2u, 3u, s); break;
    case 8u: {
        const float c = P2_SQRT1_2;
        p2_dft4(u, 0u, 2u, 4u, 6u, s);
        p2_dft4(u, 1u, 3u, 5u, 7u, s);
        const float2 u3 = u[3];
        const float2 u7 = u[7];
        const float2 o0 = u[1];
        const float2 o1 = float2(mkl_mul(c, mkl_sub(u3.x, s * u3.y)), mkl_mul(c, mkl_add(u3.y, s * u3.x)));
        const float2 o2 = p2_rot(u[5], s);
        const float2 o3 = float2(mkl_mul(c, mkl_sub(-u7.x, s * u7.y)), mkl_mul(c, mkl_sub(s * u7.x, u7.y)));
        const float2 e0 = u[0], e1 = u[2], e2 = u[4], e3 = u[6];
        u[0] = p2_cadd(e0, o0);
        u[4] = p2_csub(e0, o0);
        u[1] = p2_cadd(e1, o1);
        u[5] = p2_csub(e1, o1);
        u[2] = p2_cadd(e2, o2);
        u[6] = p2_csub(e2, o2);
        u[3] = p2_cadd(e3, o3);
        u[7] = p2_csub(e3, o3);
        break;
    }
    default: {
        const uint row = r == 3u ? 0u : (r == 5u ? 1u : 2u);
        float2 o[8];
        for (uint m = 0u; m < r; m++) {
            float2 acc = u[0];
            for (uint t = 1u; t < r; t++) {
                const uint q = row * 8u + (m * t) % r;
                acc = p2_cadd(acc, p2_cmul(u[t], float2(P2_COS[q], s * P2_SIN[q])));
            }
            o[m] = acc;
        }
        for (uint m = 0u; m < r; m++) u[m] = o[m];
        break;
    }
    }
}

// Twiddle and butterfly b (of n/r) at span p; returns the output base j.
__device__ inline uint p2_butterfly(const float2* tw, float2* v, uint r, uint b, uint span, float s) {
    const uint k = b % span;
    const uint w0 = span - 1u + k * (r - 1u);
    for (uint t = 1u; t < r; t++) {
        const float2 c = tw[w0 + t - 1u];
        v[t] = p2_cmul(v[t], float2(c.x, -s * c.y));
    }
    p2_dft(v, r, s);
    return (b - k) * r + k;
}

__device__ inline uint p2_radix(const P2Params& p, uint i) {
    return (p.radices[i / 8u] >> (4u * (i % 8u))) & 15u;
}

// Shared memory of the row kernels: 2048 complex values (the 16 KiB WebGPU
// limit), E = 8 elements per thread, T = ⌈n/8⌉ threads per row, G =
// 2048/(8T) rows per block of 256 threads (recipe::row_geometry).
#define P2_SH 2048u
#define P2_E 8u

struct P2Lane {
    uint base, lane, threads;
    // Threads past G·T hold no row but join every barrier.
    bool holds;
};

// One in-place Stockham stage in shared memory. T lanes cover n/r
// butterflies in K = ⌈8/r⌉ rounds; all reads precede the barrier and writes.
__device__ inline void p2_stage(float2* sh, const float2* tw, P2Lane l, uint n, uint r, uint span, float s) {
    const uint m = n / r;
    const uint rounds = (P2_E + r - 1u) / r;
    float2 v[16];
    for (uint q = 0u; q < rounds; q++) {
        const uint b = l.lane + q * l.threads;
        if (l.holds && b < m)
            for (uint t = 0u; t < r; t++) v[q * r + t] = sh[l.base + b + t * m];
    }
    __syncthreads();
    for (uint q = 0u; q < rounds; q++) {
        const uint b = l.lane + q * l.threads;
        if (l.holds && b < m) {
            float2 u[8];
            for (uint t = 0u; t < r; t++) u[t] = v[q * r + t];
            const uint j = p2_butterfly(tw, u, r, b, span, s);
            for (uint t = 0u; t < r; t++) sh[l.base + j + t * span] = u[t];
        }
    }
    __syncthreads();
}

__device__ inline void p2_fft(float2* sh, const float2* tw, const P2Params& p, P2Lane l, uint n,
                              uint stages, float s) {
    uint span = 1u;
    for (uint i = 0u; i < stages; i++) {
        const uint r = p2_radix(p, i);
        p2_stage(sh, tw, l, n, r, span, s);
        span *= r;
    }
}

__device__ inline P2Lane p2_lane(uint tid, uint n) {
    const uint threads = (n + P2_E - 1u) / P2_E;
    const uint rows = P2_SH / (P2_E * threads);
    const uint slot = tid / threads;
    P2Lane l;
    l.base = slot * n;
    l.lane = tid % threads;
    l.threads = threads;
    l.holds = slot < rows;
    return l;
}

__device__ inline uint p2_row(uint tid, uint group, uint n) {
    const uint threads = (n + P2_E - 1u) / P2_E;
    return group * (P2_SH / (P2_E * threads)) + tid / threads;
}

// `height` independent rows of `width` points, whole rows in shared memory.
P2_KERNEL mkl_fft_rows(const void*, const P2Params* pp, const float2* f_tw, const float2* f_a,
                       float2* f_out) {
    __shared__ float2 sh[P2_SH];
    const P2Params& p = *pp;
    const uint tid = threadIdx.x;
    const uint n = p.width;
    const P2Lane l = p2_lane(tid, n);
    const uint row = p2_row(tid, blockIdx.x, n);
    const bool live = l.holds && row < p.height;
    if (live)
        for (uint i = l.lane; i < n; i += l.threads) sh[l.base + i] = f_a[row * n + i];
    __syncthreads();
    p2_fft(sh, f_tw, p, l, n, p.stages, p2_sign(p));
    const float scale = p2_scale(p);
    if (live)
        for (uint i = l.lane; i < n; i += l.threads) f_out[row * n + i] = p2_cscale(sh[l.base + i], scale);
}

// Exact chirp-z of a row, fused: pre-chirp, FFT, kernel spectrum, inverse
// FFT, post-chirp. f_chirp holds nx pre- then ox post-factors.
P2_KERNEL mkl_chirp_rows(const void*, const P2Params* pp, const float2* f_tw, const float2* f_a,
                         float2* f_out, const float2* f_spectrum, const float2* f_chirp) {
    __shared__ float2 sh[P2_SH];
    const P2Params& p = *pp;
    const uint tid = threadIdx.x;
    const uint n = p.width;
    const uint nx = p.nx;
    const P2Lane l = p2_lane(tid, n);
    const uint row = p2_row(tid, blockIdx.x, n);
    const bool live = l.holds && row < p.height;
    if (l.holds) {
        for (uint i = l.lane; i < n; i += l.threads) {
            float2 z = float2(0.0f, 0.0f);
            if (live && i < nx) z = p2_cmul(f_a[row * nx + i], f_chirp[i]);
            sh[l.base + i] = z;
        }
    }
    __syncthreads();
    p2_fft(sh, f_tw, p, l, n, p.stages, -1.0f);
    if (l.holds)
        for (uint i = l.lane; i < n; i += l.threads) sh[l.base + i] = p2_cmul(sh[l.base + i], f_spectrum[i]);
    __syncthreads();
    p2_fft(sh, f_tw, p, l, n, p.stages, 1.0f);
    const float scale = p2_scale(p);
    if (live)
        for (uint i = l.lane; i < p.ox; i += l.threads)
            f_out[row * p.ox + i] = p2_cscale(p2_cmul(sh[l.base + i], f_chirp[nx + i]), scale);
}

// One global Stockham stage (radix radices[0], span) for rows longer than
// the shared memory holds: grid (butterfly, row) in blocks of 256.
P2_KERNEL mkl_fft_stage(const void*, const P2Params* pp, const float2* f_tw, const float2* f_a,
                        float2* f_out) {
    const P2Params& p = *pp;
    const uint n = p.width;
    const uint r = p.radices[0];
    const uint m = n / r;
    const uint b = mkl_gid_x();
    const uint row = mkl_gid_y();
    if (b >= m || row >= p.height) return;
    const uint base = row * n;
    const float s = p2_sign(p);
    const float scale = p2_scale(p);
    float2 v[8];
    for (uint t = 0u; t < r; t++) v[t] = f_a[base + b + t * m];
    const uint j = p2_butterfly(f_tw, v, r, b, p.span, s);
    for (uint t = 0u; t < r; t++) f_out[base + j + t * p.span] = p2_cscale(v[t], scale);
}

// ---------------------------------------------------------------------------
// Correlation over a tile of frequencies (correlation.rs): grid (sample k,
// tile frequency f), the tile's frequencies first..first+frequencies. The
// stencil and terms planes are the tile's own.
// ---------------------------------------------------------------------------

__device__ inline float2 p2_cdf(const float v[2]) { return float2(v[0], v[1]); }
__device__ inline float2 p2_frame(const P2CParams& c, uint i, uint j) {
    return float2(c.frame[i][j][0], c.frame[i][j][1]);
}
// (a0·b0 + a1·b1) + a2·b2 with frame row i.
__device__ inline float2 p2_dot_frame(const P2CParams& c, const float2 a[3], uint i) {
    return mkl_df_add(mkl_df_add(mkl_df_mul(a[0], p2_frame(c, i, 0u)), mkl_df_mul(a[1], p2_frame(c, i, 1u))),
                      mkl_df_mul(a[2], p2_frame(c, i, 2u)));
}

struct P2Geometry {
    float x, y, z, gamma, source_z;
};

__device__ inline P2Geometry p2_geometry(const P2CParams& c, uint k, float2 nu) {
    const uint n = c.n;
    const float2 sin_max = p2_cdf(c.sin_max);
    const float2 a = mkl_df_mul_f32(sin_max, mkl_sub(mkl_div((float)(2u * (k % n) + 1u), (float)n), 1.0f));
    const float2 b = mkl_df_mul_f32(sin_max, mkl_sub(mkl_div((float)(2u * (k / n) + 1u), (float)n), 1.0f));
    const float2 one = float2(1.0f, 0.0f);
    const float2 gamma = mkl_df_sqrt(mkl_df_sub(mkl_df_sub(one, mkl_df_mul(a, a)), mkl_df_mul(b, b)));
    float2 q[3];
    for (uint i = 0u; i < 3u; i++)
        q[i] = mkl_df_sub(mkl_df_add(mkl_df_mul(p2_frame(c, 0u, i), a), mkl_df_mul(p2_frame(c, 1u, i), b)),
                          mkl_df_mul(p2_frame(c, 2u, i), gamma));
    const float2 wl = p2_cdf(c.wavelength);
    const float2 sx = mkl_df_add(q[0], mkl_df_mul_f32(wl, nu.x));
    const float2 sy = mkl_df_add(q[1], mkl_df_mul_f32(wl, nu.y));
    const float2 z2 = mkl_df_sub(mkl_df_sub(one, mkl_df_mul(sx, sx)), mkl_df_mul(sy, sy));
    P2Geometry g = {mkl_qnan(), mkl_qnan(), 0.0f, gamma.x, -q[2].x};
    if (z2.x <= 0.0f || q[2].x >= 0.0f || !p2_is_finite(gamma.x)) return g;
    const float2 sz = mkl_df_neg(mkl_df_sqrt(z2));
    const float2 shifted[3] = {sx, sy, sz};
    if (p2_dot_frame(c, shifted, 2u).x >= 0.0f) return g;
    const float2 scale = mkl_df_div(float2(mkl_mul(0.5f, (float)n), 0.0f), sin_max);
    const float offset = mkl_mul(0.5f, (float)(n - 1u));
    const float2 ix = mkl_df_add_f32(mkl_df_mul(p2_dot_frame(c, shifted, 0u), scale), offset);
    const float2 iy = mkl_df_add_f32(mkl_df_mul(p2_dot_frame(c, shifted, 1u), scale), offset);
    g.x = mkl_add(ix.x, ix.y);
    g.y = mkl_add(iy.x, iy.y);
    g.z = -sz.x;
    return g;
}

// Storage: params, frequencies (whole), stencil tile (rw).
P2_KERNEL mkl_optical_stencil(const void*, const P2CParams* pc, const float* s_freq, float* s_dst) {
    const P2CParams& c = *pc;
    const uint k = mkl_gid_x();
    const uint fy = mkl_gid_y();
    const uint count = c.n * c.height;
    if (k >= count || fy >= c.frequencies) return;
    const uint f = c.first + fy;
    const P2Geometry g = p2_geometry(c, k, float2(s_freq[2u * f], s_freq[2u * f + 1u]));
    const uint o = (fy * count + k) * 5u;
    s_dst[o] = g.x;
    s_dst[o + 1u] = g.y;
    s_dst[o + 2u] = g.z;
    s_dst[o + 3u] = g.gamma;
    s_dst[o + 4u] = g.source_z;
}

// Storage: params, field, opd, coverage, frequencies, stencil tile, terms.
// Writes channels (real, imaginary, energy) of tile frequency f at
// terms[(3f + c)·count + k].
P2_KERNEL mkl_correlation_terms(const void*, const P2CParams* pc, const float* c_field, const float2* c_opd,
                                const float* c_cov, const float* c_freq, const float* c_stencil,
                                float* c_terms) {
    const P2CParams& pr = *pc;
    const uint k = mkl_gid_x();
    const uint fy = mkl_gid_y();
    const uint count = pr.n * pr.height;
    if (k >= count || fy >= pr.frequencies) return;
    const uint f = pr.first + fy;
    const uint base = 3u * fy * count;
    c_terms[base + k] = 0.0f;
    c_terms[base + count + k] = 0.0f;
    c_terms[base + 2u * count + k] = 0.0f;
    const bool covered = pr.covered != 0u;
    const float c = covered ? c_cov[k] : 1.0f;
    if (c <= 0.0f) return;
    const float2 nu = float2(c_freq[2u * f], c_freq[2u * f + 1u]);
    const uint n = pr.n;
    const bool optical = pr.optical != 0u;
    P2Geometry g;
    if (optical) {
        if (pr.prepared != 0u) {
            const uint o = (fy * count + k) * 5u;
            g = {c_stencil[o], c_stencil[o + 1u], c_stencil[o + 2u], c_stencil[o + 3u], c_stencil[o + 4u]};
        } else {
            g = p2_geometry(pr, k, nu);
        }
    } else {
        const float2 sx = mkl_df_add(
            mkl_df_add(mkl_df_mul_f32(p2_cdf(pr.mapping[0]), nu.x), mkl_df_mul_f32(p2_cdf(pr.mapping[1]), nu.y)),
            p2_cdf(pr.mapping[2]));
        const float2 sy = mkl_df_add(
            mkl_df_add(mkl_df_mul_f32(p2_cdf(pr.mapping[3]), nu.x), mkl_df_mul_f32(p2_cdf(pr.mapping[4]), nu.y)),
            p2_cdf(pr.mapping[5]));
        g = {mkl_add(mkl_add((float)(k % n), sx.x), sx.y), mkl_add(mkl_add((float)(k / n), sy.x), sy.y), 1.0f,
             1.0f, 1.0f};
    }
    float power;
    if (optical)
        power = mkl_mul(c_field[k], c_field[k]);
    else
        power = mkl_add(mkl_mul(c_field[2u * k], c_field[2u * k]), mkl_mul(c_field[2u * k + 1u], c_field[2u * k + 1u]));
    const float energy = mkl_div(mkl_mul(c, power), mkl_mul(g.gamma, g.source_z));
    c_terms[base + 2u * count + k] = energy;
    if (!p2_is_finite(g.x) || !p2_is_finite(g.y)) return;
    const float fx = floorf(g.x);
    const float fyf = floorf(g.y);
    const float tx = mkl_sub(g.x, fx);
    const float ty = mkl_sub(g.y, fyf);
    float ar = 0.0f, ai = 0.0f, coverage = 0.0f;
    float2 phase = float2(0.0f, 0.0f);
    for (int corner = 0; corner < 4; corner++) {
        const int dx = corner % 2;
        const int dy = corner / 2;
        const int xi = (int)fx + dx;
        const int yi = (int)fyf + dy;
        if (xi < 0 || yi < 0 || xi >= (int)n || yi >= (int)pr.height) continue;
        const uint j = (uint)yi * n + (uint)xi;
        const float wx = dx == 1 ? tx : mkl_sub(1.0f, tx);
        const float wy = dy == 1 ? ty : mkl_sub(1.0f, ty);
        const float w = mkl_mul(mkl_mul(wx, wy), covered ? c_cov[j] : 1.0f);
        if (w == 0.0f) continue;
        if (optical) {
            ar = mkl_add(ar, mkl_mul(w, c_field[j]));
            coverage = mkl_add(coverage, w);
            phase = mkl_df_add(phase, mkl_df_mul_f32(c_opd[j], w));
        } else {
            ar = mkl_add(ar, mkl_mul(w, c_field[2u * j]));
            ai = mkl_add(ai, mkl_mul(w, c_field[2u * j + 1u]));
        }
    }
    const float ex = fmaxf(0.0f, mkl_sub(1.0f, fabsf(mkl_sub(g.x, (float)(k % n)))));
    const float ey = fmaxf(0.0f, mkl_sub(1.0f, fabsf(mkl_sub(g.y, (float)(k / n)))));
    const float deficit = mkl_mul(mkl_mul(mkl_mul(mkl_sub(1.0f, c), energy), ex), ey);
    float2 v = float2(0.0f, 0.0f);
    if (optical) {
        if (coverage > 0.0f) {
            const float2 z = p2_cis(mkl_df_sub(mkl_df_div(phase, float2(coverage, 0.0f)), c_opd[k]));
            const float m = mkl_div(mkl_mul(mkl_div(mkl_mul(c, c_field[k]), g.gamma), ar), g.z);
            v = p2_cscale(z, m);
        }
    } else {
        const float fr = c_field[2u * k];
        const float fi = c_field[2u * k + 1u];
        v = float2(mkl_mul(c, mkl_add(mkl_mul(ar, fr), mkl_mul(ai, fi))),
                   mkl_mul(c, mkl_sub(mkl_mul(ai, fr), mkl_mul(ar, fi))));
    }
    c_terms[base + k] = mkl_add(v.x, deficit);
    c_terms[base + count + k] = v.y;
}

// Storage: params, the tile's double-float channel sums, out (the tile's
// normalised complex values).
MKL_KERNEL __launch_bounds__(64)
mkl_correlation_normalize(const void*, const P2CParams* pc, const float2* n_sums, float2* n_out) {
    const P2CParams& c = *pc;
    const uint f = mkl_gid_x();
    if (f >= c.frequencies) return;
    const float2 a = mkl_df_div(n_sums[3u * f], n_sums[3u * f + 2u]);
    const float2 b = mkl_df_div(n_sums[3u * f + 1u], n_sums[3u * f + 2u]);
    n_out[f] = float2(mkl_add(a.x, a.y), mkl_add(b.x, b.y));
}
