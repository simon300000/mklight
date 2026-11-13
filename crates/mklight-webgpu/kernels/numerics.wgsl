// Numerics self-tests: the floating-point behaviour probe and elementwise
// double-float operations (numerics.metal's twins), compared with the
// CPU-f32 emulation by the crate's tests.

struct MklCountParams {
    n: u32,
    op: u32,
    reserved: vec2<u32>,
}
@group(0) @binding(1) var<uniform> pc: MklCountParams;

// Probe: per triple, MKL_PROBE_WORDS raw results (what the compiler makes
// of plain expressions) at out[i * W], then the same operations written with
// the protected primitives every kernel uses, at out[(n + i) * W]. Word
// order: a*b + c, fma(a, b, c), a / b, sqrt(|a|), a + b, a * b,
// (a 2^-70) 2^-70, (a + b) - b (mklight_gpu::probe::expected).
@group(0) @binding(2) var<storage, read> pa: array<f32>;
@group(0) @binding(3) var<storage, read> pb: array<f32>;
@group(0) @binding(4) var<storage, read> pcc: array<f32>;
@group(0) @binding(5) var<storage, read_write> pout: array<f32>;

@compute @workgroup_size(64)
fn mkl_numerics_probe(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    let i = gid.x;
    if (i >= pc.n) {
        return;
    }
    let x = pa[i];
    let y = pb[i];
    let z = pcc[i];
    let w = MKL_PROBE_WORDS;
    var o = i * w;
    pout[o + 0u] = x * y + z;
    pout[o + 1u] = fma(x, y, z);
    pout[o + 2u] = x / y;
    pout[o + 3u] = sqrt(abs(x));
    pout[o + 4u] = x + y;
    pout[o + 5u] = x * y;
    pout[o + 6u] = (x * 8.470329472543003e-22) * 8.470329472543003e-22;
    pout[o + 7u] = (x + y) - y;
    o = (pc.n + i) * w;
    let p70 = mkl_pow2(-70);
    pout[o + 0u] = mkl_add(mkl_mul(x, y), z);
    pout[o + 1u] = mkl_fma(x, y, z);
    pout[o + 2u] = mkl_div(x, y);
    pout[o + 3u] = mkl_sqrt(abs(x));
    pout[o + 4u] = mkl_add(x, y);
    pout[o + 5u] = mkl_mul(x, y);
    pout[o + 6u] = mkl_mul(mkl_mul(x, p70), p70);
    pout[o + 7u] = mkl_sub(mkl_add(x, y), y);
}

// Elementwise double-float operation MKL_DFOP_* (pc.op) on (a[i], b[i]).
@group(0) @binding(2) var<storage, read> da: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read> db: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> dout: array<vec2<f32>>;

@compute @workgroup_size(64)
fn mkl_df_ops(@builtin(global_invocation_id) gid: vec3<u32>) {
    mkl_enter();
    let i = gid.x;
    if (i >= pc.n) {
        return;
    }
    let x = da[i];
    let y = db[i];
    var r: vec2<f32>;
    switch (pc.op) {
        case MKL_DFOP_TWO_SUM: { r = mkl_two_sum(x.x, y.x); }
        case MKL_DFOP_FAST_TWO_SUM: { r = mkl_fast_two_sum(x.x, y.x); }
        case MKL_DFOP_TWO_PROD: { r = mkl_two_prod(x.x, y.x); }
        case MKL_DFOP_ADD: { r = mkl_df_add(x, y); }
        case MKL_DFOP_MUL: { r = mkl_df_mul(x, y); }
        case MKL_DFOP_DIV: { r = mkl_df_div(x, y); }
        case MKL_DFOP_SQRT: { r = mkl_df_sqrt(x); }
        case MKL_DFOP_ADD_F32: { r = mkl_df_add_f32(x, y.x); }
        case MKL_DFOP_MUL_F32: { r = mkl_df_mul_f32(x, y.x); }
        case MKL_DFOP_HYPOT2: { r = vec2<f32>(mkl_hypot2(x.x, y.x), 0.0); }
        default: { r = vec2<f32>(mkl_qnan(), mkl_qnan()); }
    }
    dout[i] = r;
}
