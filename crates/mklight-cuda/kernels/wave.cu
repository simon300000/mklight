// Scalar-wave operations: wave.wgsl's interface and arithmetic, operation for
// operation (the CPU-f32 specification in mklight-gpu/src/wave_ops/{spec,
// cpu,spectral,image_integrals}.rs).
//
// Every kernel takes its parameter struct (MklWaveParams,
// MklWaveSpectralParams, MklWaveIntegralParams) as the first storage
// argument (binding 2; the passes have no uniform block, so `u` is null),
// the other storage buffers follow (bindings 3..).
//
// Numerics: division and square root are IEEE here (wave.wgsl's rounded
// mkl_div_rn, mkl_sqrt_rn, mkl_df_div_rn, mkl_df_sqrt_rn), so everything but
// the trigonometry is bitwise the specification. The specification's f32
// sin/cos are wave.wgsl's w_cis (double-float turns, reduced, evaluated by
// polynomials: no device sin/cos).
//
// Sums keep the specification's fixed order: W_BLOCK (256) consecutive
// samples per thread in sample order, then the blocks in block order.
#include "mkl_cuda.h"

#define W_BLOCK 256u

__device__ inline float2 w_df(const float* v) { return float2(v[0], v[1]); }
// spec::scalar: hi + lo.
__device__ inline float w_scalar(float2 v) { return v.x + v.y; }
__device__ inline float2 w_lift(float x) { return float2(x, 0.0f); }

// The thread's element of a 1-D grid launched as (x, y) blocks of 256.
__device__ inline uint w_linear() { return (blockIdx.y * gridDim.x + blockIdx.x) * 256u + threadIdx.x; }

// max() that ignores a NaN operand (common.wgsl's mkl_max_num).
__device__ inline float w_max_num(float a, float b) {
    if (isnan(a)) return b;
    if (isnan(b)) return a;
    return a >= b ? a : b;
}

// exp(2πi·a) for double-float turns a: reduced modulo a turn twice, ties to
// even (as spec::cis), then to an octant and evaluated by Taylor polynomials
// of degree 9/10 (truncation < 2e-9).
__device__ float2 w_cis(float2 a_in) {
    float2 a = mkl_df_add_f32(a_in, -rintf(a_in.x));
    a = mkl_df_add_f32(a, -rintf(a.x));
    const float q = rintf(a.x * 4.0f);
    const float r = (a.x - q * 0.25f) + a.y;
    const float x = r * 6.2831853071795864769f;
    const float x2 = x * x;
    float s = fmaf(x2, 2.7557319e-6f, -1.9841270e-4f);
    s = fmaf(s, x2, 8.3333333e-3f);
    s = fmaf(s, x2, -1.6666667e-1f);
    s = fmaf(s * x2, x, x);
    float c = fmaf(x2, -2.7557319e-7f, 2.4801587e-5f);
    c = fmaf(c, x2, -1.3888889e-3f);
    c = fmaf(c, x2, 4.1666668e-2f);
    c = fmaf(c, x2, -0.5f);
    c = fmaf(c, x2, 1.0f);
    switch ((((int)q) % 4 + 4) % 4) {
    case 1: return float2(-s, c);
    case 2: return float2(-c, -s);
    case 3: return float2(s, -c);
    default: return float2(c, s);
    }
}

// (cos x, sin x) of an f32 angle in radians: x/2π in double-float turns
// (1/2π as hi + lo), then w_cis.
__device__ inline float2 w_cos_sin(float x) {
    const float2 inv_tau = float2(__uint_as_float(0x3E22F983u), __uint_as_float(0x31DC9C88u));
    return w_cis(mkl_df_mul_f32(inv_tau, x));
}

// ---------------------------------------------------------------------------
// Pupil geometry (spec::direction, spec::focus) from MklWaveParams.
// ---------------------------------------------------------------------------

struct WDir {
    float2 a, b, z;
};

// (2i+1)/n - 1 in double-float.
__device__ inline float2 w_uv(uint i, uint n) {
    return mkl_df_sub(mkl_df_div(w_lift((float)(2u * i + 1u)), w_lift((float)n)), w_lift(1.0f));
}

__device__ inline WDir w_dir(const MklWaveParams& wp, uint k) {
    const uint n = wp.n;
    const float2 sm = w_df(wp.sin_max);
    const float2 a = mkl_df_mul(sm, w_uv(k % n, n));
    const float2 b = mkl_df_mul(sm, w_uv(k / n, n));
    const float2 g = mkl_df_sqrt(mkl_df_sub(mkl_df_sub(w_lift(1.0f), mkl_df_mul(a, a)), mkl_df_mul(b, b)));
    return WDir{a, b, mkl_df_neg(g)};
}

// Exact refocus OPD (waves) of pupil sample k for a chief-axis shift dz.
__device__ float2 w_focus(const MklWaveParams& wp, uint k, float2 dz) {
    if (dz.x == 0.0f && dz.y == 0.0f) return float2(0.0f);
    const WDir q = w_dir(wp, k);
    const float2 gamma = mkl_df_neg(q.z);
    const float2 r = w_df(wp.radius);
    const float2 s2 = mkl_df_add(mkl_df_mul(q.a, q.a), mkl_df_mul(q.b, q.b));
    const float2 one_gamma = mkl_df_add(w_lift(1.0f), gamma);
    float2 dr;
    if (r.x == 0.0f) {
        dr = mkl_df_div(mkl_df_mul(dz, s2), one_gamma);
    } else {
        const float2 rd = mkl_df_mul_f32(mkl_df_mul(r, dz), 2.0f);
        const float2 delta = mkl_df_add(mkl_df_mul(rd, gamma), mkl_df_mul(dz, dz));
        const float2 path = mkl_df_sqrt(mkl_df_add(mkl_df_mul(r, r), delta));
        float2 axial = mkl_df_add(r, dz);
        if (axial.x < 0.0f) axial = mkl_df_neg(axial);
        if (q.a.x == 0.0f && q.b.x == 0.0f) return float2(0.0f);
        dr = mkl_df_div(mkl_df_mul(rd, s2), mkl_df_mul(one_gamma, mkl_df_add(axial, path)));
    }
    return mkl_df_div(dr, w_df(wp.wavelength));
}

// ---------------------------------------------------------------------------
// Focus: phases of planes first..first+count, [count, n²] double-float: opd +
// focus where covered, opd elsewhere. Storage: params, coverage, opd, shifts
// (whole), out.
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_wave_focus(const void*, const MklWaveParams* pw, const float* f_cov, const float2* f_opd,
                          const float2* f_shifts, float2* f_out) {
    const MklWaveParams& wp = *pw;
    const uint t = w_linear();
    const uint count = wp.n * wp.n;
    if (t >= wp.count * count) return;
    const uint k = t % count;
    float2 v = f_opd[k];
    if (f_cov[k] > 0.0f) v = mkl_df_add(v, w_focus(wp, k, f_shifts[wp.first + t / count]));
    f_out[t] = v;
}

// The on-axis amplitude divisor -scalar(direction(j).z) of every sample.
// Storage: params, out.
MKL_KERNEL mkl_wave_gamma(const void*, const MklWaveParams* pw, float* g_out) {
    const MklWaveParams& wp = *pw;
    const uint k = w_linear();
    if (k >= wp.n * wp.n) return;
    g_out[k] = -w_scalar(w_dir(wp, k).z);
}

// ---------------------------------------------------------------------------
// Phase steps (cpu::phase_step): the largest |phase difference| between
// covered neighbours of each plane. Blocks of 256 samples, then the blocks'
// maximum (order-free). Storage: params, coverage, phases [count, n²], block
// maxima [count, blocks], out [count].
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_wave_step_blocks(const void*, const MklWaveParams* pw, const float* p_cov,
                                const float2* p_phase, float* p_blocks, float* p_out) {
    const MklWaveParams& wp = *pw;
    const uint t = w_linear();
    const uint n = wp.n;
    const uint count = n * n;
    const uint blocks = (count + W_BLOCK - 1u) / W_BLOCK;
    if (t >= wp.count * blocks) return;
    const uint base = (t / blocks) * count;
    const uint first = (t % blocks) * W_BLOCK;
    const uint end = min(first + W_BLOCK, count);
    float step = 0.0f;
    for (uint k = first; k < end; k++) {
        if (p_cov[k] <= 0.0f) continue;
        const float2 v = p_phase[base + k];
        if (k % n + 1u < n && p_cov[k + 1u] > 0.0f)
            step = w_max_num(step, fabsf(w_scalar(mkl_df_sub(p_phase[base + k + 1u], v))));
        if (k / n + 1u < n && p_cov[k + n] > 0.0f)
            step = w_max_num(step, fabsf(w_scalar(mkl_df_sub(p_phase[base + k + n], v))));
    }
    p_blocks[t] = step;
}

MKL_KERNEL mkl_wave_step_finish(const void*, const MklWaveParams* pw, const float* p_cov,
                                const float2* p_phase, float* p_blocks, float* p_out) {
    const MklWaveParams& wp = *pw;
    const uint plane = mkl_gid_x();
    if (plane >= wp.count) return;
    const uint blocks = (wp.n * wp.n + W_BLOCK - 1u) / W_BLOCK;
    float step = 0.0f;
    for (uint b = 0u; b < blocks; b++) step = w_max_num(step, p_blocks[plane * blocks + b]);
    p_out[plane] = step;
}

// ---------------------------------------------------------------------------
// Focus correlation (spec::correlation_term over focused phases, dz = 0).
// Outputs o = plane·F + f (F = field_side frequencies) of the batch's planes;
// a thread sums block b of 256 samples of output first + t/blocks into
// (real, imaginary, energy) double-floats; the finish kernel adds the blocks
// in order and normalises. The batch's phases start at plane reserved[0], its
// outputs at output reserved[1], and the bound stencil slice at frequency
// sample_count; sample_first is the block count. Storage: params, amplitude,
// coverage, phases [planes, n²], stencil [frequencies, n², 5], on-axis
// divisors, block sums.
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_wave_focus_correlation(const void*, const MklWaveParams* pw, const float* c_amp,
                                      const float* c_cov, const float2* c_phase, const float* c_stencil,
                                      const float* c_gamma, float2* c_acc) {
    const MklWaveParams& wp = *pw;
    const uint t = w_linear();
    const uint blocks = wp.sample_first;
    if (t >= wp.count * blocks) return;
    const uint n = wp.n;
    const uint count = n * n;
    const uint o = wp.first + t / blocks;
    const uint frequency = o % wp.field_side;
    const uint base = (o / wp.field_side - wp.reserved[0]) * count;
    const uint first = (t % blocks) * W_BLOCK;
    const uint end = min(first + W_BLOCK, count);
    const bool on_axis = wp.shifted != 0u;
    float2 re = float2(0.0f), im = float2(0.0f), sum = float2(0.0f);
    for (uint k = first; k < end; k++) {
        const float c = c_cov[k];
        if (c <= 0.0f) {
            re = mkl_df_add_f32(re, 0.0f);
            im = mkl_df_add_f32(im, 0.0f);
            sum = mkl_df_add_f32(sum, 0.0f);
            continue;
        }
        const uint off = ((frequency - wp.sample_count) * count + k) * 5u;
        const float ix = c_stencil[off];
        const float iy = c_stencil[off + 1u];
        const float qz = c_stencil[off + 2u];
        const float gamma = c_stencil[off + 3u];
        const float srcz = c_stencil[off + 4u];
        const float ak = c_amp[k];
        const float energy = ((c * ak) * ak) / (gamma * srcz);
        float t0 = 0.0f, t1 = 0.0f;
        if (isfinite(ix) && isfinite(iy)) {
            const float fx = floorf(ix);
            const float fy = floorf(iy);
            const float tx = ix - fx;
            const float ty = iy - fy;
            float ar = 0.0f, cov = 0.0f;
            float2 phase = float2(0.0f);
            // Corners outside [-1, n] cannot touch the grid (and keep the
            // integer conversions in range).
            if (fx >= -1.0f && fy >= -1.0f && fx <= (float)n && fy <= (float)n) {
                for (int corner = 0; corner < 4; corner++) {
                    const int dx = corner % 2;
                    const int dy = corner / 2;
                    const int xi = (int)fx + dx;
                    const int yi = (int)fy + dy;
                    if (xi < 0 || yi < 0 || xi >= (int)n || yi >= (int)n) continue;
                    const uint j = (uint)yi * n + (uint)xi;
                    const float wx = dx == 1 ? tx : 1.0f - tx;
                    const float wy = dy == 1 ? ty : 1.0f - ty;
                    const float weight = (wx * wy) * c_cov[j];
                    if (weight == 0.0f) continue;
                    float amp = weight * c_amp[j];
                    if (on_axis) amp = amp / c_gamma[j];
                    ar = ar + amp;
                    cov = cov + weight;
                    phase = mkl_df_add(phase, mkl_df_mul_f32(c_phase[base + j], weight));
                }
            }
            const float ex = fmaxf(1.0f - fabsf(ix - (float)(k % n)), 0.0f);
            const float ey = fmaxf(1.0f - fabsf(iy - (float)(k / n)), 0.0f);
            const float deficit = (((1.0f - c) * energy) * ex) * ey;
            t0 = deficit;
            if (cov > 0.0f) {
                const float2 z = w_cis(mkl_df_sub(mkl_df_div(phase, w_lift(cov)), c_phase[base + k]));
                float m = ((c * ak) / gamma) * ar;
                if (!on_axis) m = m / qz;
                t0 = m * z.x + deficit;
                t1 = m * z.y;
            }
        }
        re = mkl_df_add_f32(re, t0);
        im = mkl_df_add_f32(im, t1);
        sum = mkl_df_add_f32(sum, energy);
    }
    c_acc[3u * t] = re;
    c_acc[3u * t + 1u] = im;
    c_acc[3u * t + 2u] = sum;
}

// Storage: params, block sums, out (the batch's outputs, complex).
MKL_KERNEL mkl_wave_focus_correlation_finish(const void*, const MklWaveParams* pw, const float2* cf_acc,
                                             float2* cf_out) {
    const MklWaveParams& wp = *pw;
    const uint id = mkl_gid_x();
    if (id >= wp.count) return;
    const uint blocks = wp.sample_first;
    float2 re = float2(0.0f), im = float2(0.0f), sum = float2(0.0f);
    for (uint b = 0u; b < blocks; b++) {
        const uint j = 3u * (id * blocks + b);
        re = mkl_df_add(re, cf_acc[j]);
        im = mkl_df_add(im, cf_acc[j + 1u]);
        sum = mkl_df_add(sum, cf_acc[j + 2u]);
    }
    // The batch's outputs from output reserved[1].
    cf_out[wp.first - wp.reserved[1] + id] = float2(w_scalar(mkl_df_div(re, sum)), w_scalar(mkl_df_div(im, sum)));
}

// ---------------------------------------------------------------------------
// Prepare (spec::prepare): the Debye field padded at the low corner of a
// field_side² grid and the three moment planes. Storage: params, amplitude,
// coverage, opd, field, moments.
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_wave_prepare(const void*, const MklWaveParams* pw, const float* r_amp, const float* r_cov,
                            const float2* r_opd, float2* r_field, float* r_mom) {
    const MklWaveParams& wp = *pw;
    const uint k = w_linear();
    const uint side = wp.field_side;
    if (k >= side * side) return;
    const uint n = wp.n;
    const uint x = k % side;
    const uint y = k / side;
    float2 z = float2(0.0f);
    if (x < n && y < n) {
        const uint j = y * n + x;
        const uint count = n * n;
        const float c = r_cov[j];
        float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f;
        if (c > 0.0f) {
            const WDir q = w_dir(wp, j);
            const float gamma = -w_scalar(q.z);
            const float a = r_amp[j];
            const float2 d = mkl_df_add(
                mkl_df_add(mkl_df_mul(q.a, w_df(wp.normal[0])), mkl_df_mul(q.b, w_df(wp.normal[1]))),
                mkl_df_mul(q.z, w_df(wp.normal[2])));
            const float nz = fabsf(w_scalar(d));
            const float weight = (c * a) / gamma;
            const float ag = a / gamma;
            const float e = (c * ag) * ag;
            float third = mkl_qnan();
            if (nz > 1e-9f) third = (e * gamma) / nz;
            m0 = weight;
            m1 = e;
            m2 = third;
            const float2 ph = w_cis(r_opd[j]);
            z = float2(weight * ph.x, weight * ph.y);
        }
        r_mom[j] = m0;
        r_mom[count + j] = m1;
        r_mom[2u * count + j] = m2;
    }
    r_field[k] = z;
}

// ---------------------------------------------------------------------------
// Intensity (spec::intensity). Storage: params, field, sums, out.
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_wave_intensity(const void*, const MklWaveParams* pw, const float2* i_field,
                              const float2* i_sums, float* i_out) {
    const MklWaveParams& wp = *pw;
    const uint k = w_linear();
    const uint m = wp.size;
    if (k >= m * m) return;
    uint j = k;
    if (wp.shifted != 0u) {
        const int side = (int)wp.field_side;
        const int ix = (((int)(k % m) - (int)(m / 2u)) % side + side) % side;
        const int iy = (((int)(k / m) - (int)(m / 2u)) % side + side) % side;
        j = (uint)iy * wp.field_side + (uint)ix;
    }
    const float2 z = i_field[j];
    const float2 l = mkl_df_div(mkl_df_mul(w_lift((float)wp.n), w_df(wp.wavelength)),
                                mkl_df_mul_f32(w_df(wp.sin_max), 2.0f));
    float2 div;
    if (wp.unit_energy != 0u) div = mkl_df_div(mkl_df_mul(mkl_df_mul(l, l), i_sums[1]), w_lift(wp.projection));
    else div = mkl_df_mul(i_sums[0], i_sums[0]);
    const float power = z.x * z.x + z.y * z.y;
    i_out[k] = w_scalar(mkl_df_div(w_lift(power), div));
}

// ---------------------------------------------------------------------------
// Line spread: rows (x first) and columns of an n² image, double-float sums
// in sample order times the spacing. Storage: params, image, out.
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_wave_lsf(const void*, const MklWaveParams* pw, const float* l_image, float* l_out) {
    const MklWaveParams& wp = *pw;
    const uint k = mkl_gid_x();
    const uint n = wp.n;
    if (k >= 2u * n) return;
    float2 s = float2(0.0f);
    for (uint j = 0u; j < n; j++) {
        const uint at = k < n ? j * n + k : (k - n) * n + j;
        s = mkl_df_add_f32(s, l_image[at]);
    }
    l_out[k] = w_scalar(mkl_df_mul(s, w_df(wp.spacing)));
}

// ---------------------------------------------------------------------------
// Spectral accumulation (spectral::cpu): the weighted double-float sum of up
// to 16 equally shaped inputs, optionally phase-shifted complex curves.
// Storage: params, inputs [inputs, count·width] (concatenated), frequencies
// (or any buffer when unused), out.
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_wave_spectral(const void*, const MklWaveSpectralParams* ps, const float* s_in,
                             const float* s_freq, float* s_out) {
    const MklWaveSpectralParams& sp = *ps;
    const uint k = w_linear();
    if (k >= sp.count) return;
    const bool complex = sp.complex != 0u;
    const uint width = complex ? 2u : 1u;
    const uint stride = sp.count * width;
    float2 s0 = float2(0.0f), s1 = float2(0.0f);
    for (uint j = 0u; j < sp.inputs; j++) {
        float2 z = float2(s_in[j * stride + k * width], 0.0f);
        if (complex) z.y = s_in[j * stride + k * width + 1u];
        if (sp.phase != 0u) {
            const float2 o0 = w_df(sp.offsets[j][0]);
            const float2 o1 = w_df(sp.offsets[j][1]);
            const float2 ph = mkl_df_neg(
                mkl_df_add(mkl_df_mul_f32(o0, s_freq[2u * k]), mkl_df_mul_f32(o1, s_freq[2u * k + 1u])));
            const float2 q = w_cis(ph);
            z = float2(z.x * q.x - z.y * q.y, z.x * q.y + z.y * q.x);
        }
        const float2 w = w_df(sp.weights[j]);
        s0 = mkl_df_add(s0, mkl_df_mul_f32(w, z.x));
        if (complex) s1 = mkl_df_add(s1, mkl_df_mul_f32(w, z.y));
    }
    s_out[k * width] = w_scalar(s0);
    if (complex) s_out[k * width + 1u] = w_scalar(s1);
}

// ---------------------------------------------------------------------------
// Image integrals (image_integrals::cpu), from MklWaveIntegralParams.
// ---------------------------------------------------------------------------

// Metrics: per block of 256 pixels (sum, Σv·x, Σv·y, maximum, its index),
// then the blocks in order. Storage: params, image, blocks [blocks, 5]
// double-float, out [5] double-float.
MKL_KERNEL mkl_wave_metrics(const void*, const MklWaveIntegralParams* pi, const float* m_image,
                            float2* m_blocks, float2* m_out) {
    const MklWaveIntegralParams& ip = *pi;
    const uint b = w_linear();
    const uint n = ip.n;
    const uint count = n * n;
    if (b * W_BLOCK >= count) return;
    float2 row[5] = {float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f), float2(0.0f)};
    const uint end = min((b + 1u) * W_BLOCK, count);
    for (uint k = b * W_BLOCK; k < end; k++) {
        const float v = m_image[k];
        row[0] = mkl_df_add_f32(row[0], v);
        row[1] = mkl_df_add(row[1], mkl_df_mul_f32(w_lift(v), (float)(k % n)));
        row[2] = mkl_df_add(row[2], mkl_df_mul_f32(w_lift(v), (float)(k / n)));
        if (k % W_BLOCK == 0u || v > row[3].x) {
            row[3] = w_lift(v);
            row[4] = w_lift((float)k);
        }
    }
    for (uint j = 0u; j < 5u; j++) m_blocks[5u * b + j] = row[j];
}

MKL_KERNEL mkl_wave_metrics_finish(const void*, const MklWaveIntegralParams* pi, const float* m_image,
                                   float2* m_blocks, float2* m_out) {
    const MklWaveIntegralParams& ip = *pi;
    if (mkl_gid_x() != 0u) return;
    const uint blocks = (ip.n * ip.n + W_BLOCK - 1u) / W_BLOCK;
    float2 sum[5];
    for (uint j = 0u; j < 5u; j++) sum[j] = m_blocks[j];
    for (uint b = 1u; b < blocks; b++) {
        for (uint j = 0u; j < 3u; j++) sum[j] = mkl_df_add(sum[j], m_blocks[5u * b + j]);
        if (m_blocks[5u * b + 3u].x > sum[3].x) {
            sum[3] = m_blocks[5u * b + 3u];
            sum[4] = m_blocks[5u * b + 4u];
        }
    }
    for (uint j = 0u; j < 5u; j++) m_out[j] = sum[j];
}

// Complex: the image (or |z|² of a complex one) as complex values.
// Storage: params, input, out.
MKL_KERNEL mkl_wave_image_complex(const void*, const MklWaveIntegralParams* pi, const float* x_in,
                                  float2* x_out) {
    const MklWaveIntegralParams& ip = *pi;
    const uint k = w_linear();
    if (k >= ip.n * ip.n) return;
    if (ip.kind == 4u) {
        const float re = x_in[2u * k];
        const float im = x_in[2u * k + 1u];
        x_out[k] = float2(re * re + im * im, 0.0f);
    } else {
        x_out[k] = float2(x_in[k], 0.0f);
    }
}

// image_integrals::j1: the specification's polynomials in its nesting; the
// cosine through w_cos_sin.
__device__ float w_j1(float x) {
    const float ax = fabsf(x);
    if (ax <= 3.0f) {
        const float x3 = x / 3.0f;
        const float t = x3 * x3;
        float p = -0.00031761f + t * 0.00001109f;
        p = 0.00443319f + t * p;
        p = -0.03954289f + t * p;
        p = 0.21093573f + t * p;
        p = -0.5624998f + t * p;
        p = 0.5f + t * p;
        return x * p;
    }
    const float y = 3.0f / ax;
    float f = 0.00113653f - y * 0.00020033f;
    f = -0.00249511f + y * f;
    f = 0.00017105f + y * f;
    f = 0.01659667f + y * f;
    f = 0.00000156f + y * f;
    f = 0.7978846f + y * f;
    float h = 0.00079824f - y * 0.00029166f;
    h = 0.00074348f + y * h;
    h = -0.00637879f + y * h;
    h = 0.00005650f + y * h;
    h = 0.12499612f + y * h;
    const float th = (ax - 2.3561945f) + y * h;
    const float r = (f * w_cos_sin(th).x) / sqrtf(ax);
    return x < 0.0f ? -r : r;
}

// image_integrals::term of pixel k for query (q0, q1) and value v.
__device__ float2 w_integral_term(const MklWaveIntegralParams& ip, uint k, float2 q0, float2 q1, float2 v) {
    const uint n = ip.n;
    const float2 center0 = w_df(ip.center[0]);
    const float2 center1 = w_df(ip.center[1]);
    const float2 spacing = w_df(ip.spacing);
    if (ip.kind == 0u) {
        const float2 x = w_lift((float)(k % n));
        const float2 y = w_lift((float)(k / n));
        const float2 ph = mkl_df_mul(
            mkl_df_neg(mkl_df_add(mkl_df_mul(mkl_df_sub(x, center0), q0), mkl_df_mul(mkl_df_sub(y, center1), q1))),
            spacing);
        const float2 z = w_cis(ph);
        return float2(v.x * z.x, v.x * z.y);
    }
    int ix = (int)(k % n);
    if (k % n > (n - 1u) / 2u) ix = ix - (int)n;
    int iy = (int)(k / n);
    if (k / n > (n - 1u) / 2u) iy = iy - (int)n;
    const float2 nn = w_lift((float)n);
    const float2 inv = mkl_df_div(w_lift(1.0f), mkl_df_mul(nn, spacing));
    const float2 fx = mkl_df_mul_f32(inv, (float)ix);
    const float2 fy = mkl_df_mul_f32(inv, (float)iy);
    const float2 ph =
        mkl_df_div(mkl_df_add(mkl_df_mul_f32(center0, (float)ix), mkl_df_mul_f32(center1, (float)iy)), nn);
    const float2 z = w_cis(ph);
    const float real = v.x * z.x - v.y * z.y;
    const float radius = w_scalar(q0);
    const float pi = 3.14159265358979323846f;
    const float tau = 6.28318530717958647692f;
    float kernel = 0.0f;
    if (radius <= 0.0f) {
        kernel = 0.0f;
    } else if (ip.kind == 1u) {
        const float rho = w_scalar(mkl_df_sqrt(mkl_df_add(mkl_df_mul(fx, fx), mkl_df_mul(fy, fy))));
        if (rho * radius < 1e-12f) kernel = (pi * radius) * radius;
        else kernel = (radius * w_j1((tau * rho) * radius)) / rho;
    } else {
        const float f[2] = {w_scalar(fx), w_scalar(fy)};
        float sinc[2] = {0.0f, 0.0f};
        for (int i = 0; i < 2; i++) {
            if (f[i] == 0.0f) {
                sinc[i] = 2.0f * radius;
            } else {
                const float s = w_cos_sin((tau * f[i]) * radius).y;
                sinc[i] = s / (pi * f[i]);
            }
        }
        kernel = sinc[0] * sinc[1];
    }
    return float2(real * kernel, 0.0f);
}

// Query: thread t sums block t % blocks of query first + t / blocks.
// Storage: params, input, queries [queries, 2] double-float, block sums.
MKL_KERNEL mkl_wave_image_query(const void*, const MklWaveIntegralParams* pi, const float* q_in,
                                const float2* q_queries, float2* q_acc) {
    const MklWaveIntegralParams& ip = *pi;
    const uint t = w_linear();
    const uint count = ip.n * ip.n;
    const uint blocks = (count + W_BLOCK - 1u) / W_BLOCK;
    if (t >= ip.count * blocks) return;
    const uint q = ip.first + t / blocks;
    const float2 q0 = q_queries[2u * q];
    const float2 q1 = q_queries[2u * q + 1u];
    const uint first = (t % blocks) * W_BLOCK;
    const uint end = min(first + W_BLOCK, count);
    float2 s0 = float2(0.0f), s1 = float2(0.0f);
    for (uint k = first; k < end; k++) {
        float2 v = float2(q_in[k], 0.0f);
        if (ip.kind != 0u) v = float2(q_in[2u * k], q_in[2u * k + 1u]);
        const float2 z = w_integral_term(ip, k, q0, q1, v);
        s0 = mkl_df_add_f32(s0, z.x);
        s1 = mkl_df_add_f32(s1, z.y);
    }
    q_acc[2u * t] = s0;
    q_acc[2u * t + 1u] = s1;
}

// Storage: params, queries, block sums, out (complex, all queries).
MKL_KERNEL mkl_wave_image_query_finish(const void*, const MklWaveIntegralParams* pi, const float2* qf_queries,
                                       const float2* qf_acc, float2* qf_out) {
    const MklWaveIntegralParams& ip = *pi;
    const uint id = mkl_gid_x();
    if (id >= ip.count) return;
    const uint q = ip.first + id;
    const uint blocks = (ip.n * ip.n + W_BLOCK - 1u) / W_BLOCK;
    float2 s0 = float2(0.0f), s1 = float2(0.0f);
    for (uint b = 0u; b < blocks; b++) {
        s0 = mkl_df_add(s0, qf_acc[2u * (id * blocks + b)]);
        s1 = mkl_df_add(s1, qf_acc[2u * (id * blocks + b) + 1u]);
    }
    if (ip.kind == 0u && qf_queries[2u * q].x == 0.0f && qf_queries[2u * q + 1u].x == 0.0f) {
        qf_out[q] = float2(1.0f, 0.0f);
        return;
    }
    const float2 norm = w_df(ip.normalization);
    qf_out[q] = float2(w_scalar(mkl_df_div(s0, norm)), w_scalar(mkl_df_div(s1, norm)));
}
