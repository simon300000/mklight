// FFT-MTF wave operations (`FftMtf`): the sensor-domain check, the sensor
// angular-spectrum field, the OTF normalisation and the OTF interpolation of
// the CPU-f32 specification (mklight-gpu/src/wave_ops/fft_mtf.rs), operation
// for operation.
//
// CUDA only: there is no WGSL twin, and WebGPU devices run the specification
// (`Gpu::ports("fft_mtf")`). The module is compiled after wave.cu in one
// unit (kernels.rs) and uses its helpers (w_df, w_scalar, w_lift, w_linear,
// w_cis). Block size 256 (kernels::CUDA_ENTRIES), grids from grid_1d.
//
// Interface as wave.cu's: `u` is null (no uniform block), the parameter
// struct MklWaveFftMtfParams (fft_mtf::Params, its `side` and `count` set by
// the host) is the first storage binding, the other buffers follow:
//
//   mkl_fft_mtf_domain     params, coverage f32[n²], status u32[1] (zeroed)
//   mkl_fft_mtf_sensor     params, amplitude f32[n²], coverage f32[n²],
//                          opd df[n²], field c32[side²], moments f32[3, n²]
//   mkl_fft_mtf_normalize  params, input c32[n²], sums df[3], output c32[n²]
//   mkl_fft_mtf_sample     params, data c32[n²], reference c32[n²],
//                          frequencies f32[count, 2], output c32[count]
//
// Numerics: IEEE division and square root, no contraction, so everything but
// the trigonometry is bitwise the specification. Its f32 sin/cos are w_cis;
// its atan2 and hypot (the OTF centroid) are CUDA's atan2f and hypotf.

struct FDir {
    float2 x, y, z;
};

// spec::dot.
__device__ inline float2 f_dot(FDir a, const float (*b)[2]) {
    return mkl_df_add(mkl_df_add(mkl_df_mul(a.x, w_df(b[0])), mkl_df_mul(a.y, w_df(b[1]))),
                      mkl_df_mul(a.z, w_df(b[2])));
}

// (2i+1)/n - 1 in double-float (the specification's `coord`).
__device__ inline float2 f_coord(uint i, uint n) {
    return mkl_df_sub(mkl_df_div(w_lift((float)(2u * i + 1u)), w_lift((float)n)), w_lift(1.0f));
}

// `f as i64` of Rust: NaN to 0, saturating.
__device__ inline long long f_i64(float v) {
    if (isnan(v)) return 0;
    if (v >= 9.2233720368547758e18f) return 9223372036854775807LL;
    if (v <= -9.2233720368547758e18f) return -9223372036854775807LL - 1;
    return (long long)v;
}

__device__ inline float2 f_cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// ---------------------------------------------------------------------------
// Domain: covered cells whose sensor direction is at or beyond the image
// surface's horizon (normal = p.frame[0]) are counted into status[0].
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_fft_mtf_domain(const void*, const MklWaveFftMtfParams* pp, const float* coverage, uint* status) {
    const MklWaveFftMtfParams& p = *pp;
    const uint k = w_linear();
    if (k >= p.count || coverage[k] <= 0.0f) return;
    const uint n = p.n;
    const float2 sine = w_df(p.sine);
    const float2 a = mkl_df_mul(sine, f_coord(k % n, n));
    const float2 b = mkl_df_mul(sine, f_coord(k / n, n));
    const float2 gamma = mkl_df_sqrt(mkl_df_sub(mkl_df_sub(w_lift(1.0f), mkl_df_mul(a, a)), mkl_df_mul(b, b)));
    const float2 z = mkl_df_sub(mkl_df_add(mkl_df_mul(w_df(p.frame[0][0]), a), mkl_df_mul(w_df(p.frame[0][1]), b)),
                                mkl_df_mul(w_df(p.frame[0][2]), gamma));
    if (!isfinite(z.x) || !isfinite(z.y) || z.x >= 0.0f) atomicAdd(status, 1u);
}

// ---------------------------------------------------------------------------
// Sensor: thread k of side² writes field[k] (zero outside the n×n pupil
// lattice and where the specification's `sensor` is zero) and, inside it,
// the moments of node j·n + i (channels 0 and 2 zero, channel 1 the energy).
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_fft_mtf_sensor(const void*, const MklWaveFftMtfParams* pp, const float* amp, const float* cov,
                              const float2* opd, float2* field, float* moments) {
    const MklWaveFftMtfParams& p = *pp;
    const uint k = w_linear();
    if (k >= p.count) return;
    field[k] = float2(0.0f);
    const uint n = p.n, i = k % p.side, j = k / p.side;
    if (i >= n || j >= n) return;
    const uint node = j * n + i, nn = n * n;
    moments[node] = 0.0f;
    moments[nn + node] = 0.0f;
    moments[2u * nn + node] = 0.0f;

    const float2 one = w_lift(1.0f);
    const float2 sine = w_df(p.sine);
    const float2 al = mkl_df_mul(sine, f_coord(i, n));
    const float2 be = mkl_df_mul(sine, f_coord(j, n));
    const float2 g = mkl_df_sqrt(mkl_df_sub(one, mkl_df_mul(sine, sine)));
    const float2 x = mkl_df_sub(mkl_df_add(mkl_df_mul(w_df(p.frame[0][0]), al), mkl_df_mul(w_df(p.frame[1][0]), be)),
                                mkl_df_mul(w_df(p.frame[2][0]), g));
    const float2 y = mkl_df_sub(mkl_df_add(mkl_df_mul(w_df(p.frame[0][1]), al), mkl_df_mul(w_df(p.frame[1][1]), be)),
                                mkl_df_mul(w_df(p.frame[2][1]), g));
    const float2 z2 = mkl_df_sub(mkl_df_sub(one, mkl_df_mul(x, x)), mkl_df_mul(y, y));
    if (z2.x <= 0.0f) return;
    const FDir q = {x, y, mkl_df_neg(mkl_df_sqrt(z2))};
    if (f_dot(q, p.frame[2]).x >= 0.0f) return;
    const float2 scale = mkl_df_div(w_lift((float)n * 0.5f), sine);
    const float half = (float)(n - 1u) * 0.5f;
    const float ix = w_scalar(mkl_df_add_f32(mkl_df_mul(f_dot(q, p.frame[0]), scale), half));
    const float iy = w_scalar(mkl_df_add_f32(mkl_df_mul(f_dot(q, p.frame[1]), scale), half));
    const float fx = floorf(ix), fy = floorf(iy);
    const float tx = ix - fx, ty = iy - fy;
    const long long x0 = f_i64(fx), y0 = f_i64(fy);
    float a = 0.0f, c = 0.0f;
    float2 phase = float2(0.0f);
    for (uint t = 0; t < 4; ++t) {
        const uint dx = t & 1u, dy = t >> 1;
        const float wt = (dx ? tx : 1.0f - tx) * (dy ? ty : 1.0f - ty);
        const long long xx = x0 + dx, yy = y0 + dy;
        if (xx < 0 || yy < 0 || xx >= (long long)n || yy >= (long long)n) continue;
        const uint at = (uint)yy * n + (uint)xx;
        const float weight = wt * cov[at];
        a += weight * amp[at];
        c += weight;
        phase = mkl_df_add(phase, mkl_df_mul_f32(opd[at], weight));
    }
    if (c <= 0.0f) return;
    const float qz = -w_scalar(q.z);
    const float2 z = w_cis(mkl_df_div(phase, w_lift(c)));
    const float s = a / qz;
    field[k] = float2(s * z.x, s * z.y);
    moments[nn + node] = a * a / (c * qz * qz);
}

// ---------------------------------------------------------------------------
// Normalize: the raw OTF (n×n, DC at 0) shifted to DC at (n/2, n/2) and
// divided by n²·sums[1]; the DC value is exactly 1.
// ---------------------------------------------------------------------------
MKL_KERNEL mkl_fft_mtf_normalize(const void*, const MklWaveFftMtfParams* pp, const float2* in, const float2* sums,
                                 float2* out) {
    const MklWaveFftMtfParams& p = *pp;
    const uint k = w_linear();
    const uint n = p.n;
    if (k >= p.count) return;
    if (k == (n / 2u) * n + n / 2u) {
        out[k] = float2(1.0f, 0.0f);
        return;
    }
    const uint x = (k % n + n - n / 2u) % n, y = (k / n + n - n / 2u) % n;
    const float2 z = in[y * n + x];
    const float2 e = mkl_df_mul_f32(sums[1], (float)(n * n));
    out[k] = float2(w_scalar(mkl_df_div(w_lift(z.x), e)), w_scalar(mkl_df_div(w_lift(z.y), e)));
}

// ---------------------------------------------------------------------------
// Sample: the OTF at frequency k, Keys-cubic interpolation of the
// centroid-corrected difference from the reference plus bilinear
// interpolation of the reference, then the centroid phase restored.
// ---------------------------------------------------------------------------
__device__ inline float f_keys(float x) {
    x = fabsf(x);
    if (x <= 1.0f) return (1.5f * x - 2.5f) * x * x + 1.0f;
    if (x < 2.0f) return ((-0.5f * x + 2.5f) * x - 4.0f) * x + 2.0f;
    return 0.0f;
}

__device__ inline float2 f_at(const float2* v, uint n, long long i, long long j) {
    if (i < 0 || j < 0 || i >= (long long)n || j >= (long long)n) return float2(0.0f);
    return v[(size_t)j * n + (size_t)i];
}

__device__ inline float f_centroid(float2 z, float d) {
    return hypotf(z.x, z.y) > 1e-9f ? -atan2f(z.y, z.x) / (6.28318530717958647692f * d) : 0.0f;
}

MKL_KERNEL mkl_fft_mtf_sample(const void*, const MklWaveFftMtfParams* pp, const float2* data, const float2* dl,
                              const float* freq, float2* out) {
    const MklWaveFftMtfParams& p = *pp;
    const uint k = w_linear();
    if (k >= p.count) return;
    const uint n = p.n, center = n / 2u;
    const float d = w_scalar(w_df(p.spacing));
    const float cx = f_centroid(data[center * n + center + 1u], d);
    const float cy = f_centroid(data[(center + 1u) * n + center], d);
    const float nu0 = freq[2u * k], nu1 = freq[2u * k + 1u];
    const float nx =
        w_scalar(mkl_df_add(mkl_df_mul_f32(w_df(p.mapping[0][0]), nu0), mkl_df_mul_f32(w_df(p.mapping[0][1]), nu1)));
    const float ny =
        w_scalar(mkl_df_add(mkl_df_mul_f32(w_df(p.mapping[1][0]), nu0), mkl_df_mul_f32(w_df(p.mapping[1][1]), nu1)));
    const float x = nx / d + (float)center;
    const float y = ny / d + (float)center;
    const float ix = floorf(x), iy = floorf(y);
    const float tx = x - ix, ty = y - iy;
    const long long i0 = f_i64(ix), j0 = f_i64(iy);
    float2 sum = float2(0.0f);
    for (int by = -1; by <= 2; ++by) {
        for (int bx = -1; bx <= 2; ++bx) {
            const long long i = i0 + bx, j = j0 + by;
            const float2 phase = mkl_df_mul(
                w_lift(d), mkl_df_add(mkl_df_mul_f32(w_lift((float)(i - (long long)center)), cx),
                                      mkl_df_mul_f32(w_lift((float)(j - (long long)center)), cy)));
            const float2 z = f_cmul(f_at(data, n, i, j), w_cis(phase));
            const float2 r = f_at(dl, n, i, j);
            const float wt = f_keys(tx - (float)bx) * f_keys(ty - (float)by);
            sum.x += (z.x - r.x) * wt;
            sum.y += (z.y - r.y) * wt;
        }
    }
    for (uint t = 0; t < 4; ++t) {
        const uint dx = t & 1u, dy = t >> 1;
        const float wt = (dx ? tx : 1.0f - tx) * (dy ? ty : 1.0f - ty);
        const float2 z = f_at(dl, n, i0 + dx, j0 + dy);
        sum.x += z.x * wt;
        sum.y += z.y * wt;
    }
    out[k] = f_cmul(sum, w_cis(mkl_df_neg(mkl_df_add(mkl_df_mul_f32(w_lift(nx), cx), mkl_df_mul_f32(w_lift(ny), cy)))));
}
