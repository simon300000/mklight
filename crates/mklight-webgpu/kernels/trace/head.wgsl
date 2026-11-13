// The sequential / path ray-trace kernel: the WGSL twin, operation for
// operation, of crates/mklight-metal/shaders/{surface_math.h,
// trace_shared.h, trace.metal} and of the resident trace (mkl_trace_resident)
// of aiming.metal. The Rust specification is mklight-gpu's trace_kernel.rs,
// surface_math.rs and fmath.rs; change all of them together.
//
// src/kernels.rs compiles this file twice, prefixed with
// `const MKL_OVERLAYS: bool = false;` (module "trace") or `true` (module
// "trace_overlays", for systems with sag overlays), as Metal compiles
// trace_overlays.metal: the overlay code costs systems without overlays
// nothing.
//
// Numerics (common.wgsl): every floating-point operation is a protected
// mkl_* operation (one IEEE rounding, never fused or reassociated); plain
// operators only negate, take absolute values, compare and select. Tests of
// values that may be NaN use the bit tests (mkl_is_nan / mkl_is_finite) or
// test NaN explicitly before a negated comparison, since a fast-math
// compiler may rewrite !(a <= b) as a > b. Hardware `/` and `sqrt` are not
// correctly rounded on every WebGPU compiler, so the kernel divides and takes
// square roots with mkl_div_rn / mkl_sqrt_rn (and the double-float
// operations built on them): every output and record is bitwise the CPU-f32
// emulation's (NaN payloads aside).
//
// Templates: surface_math.h is generic in the scalar R (float or the
// double-float MklDf). Below it appears twice, with the suffix _f for
// R = f32 and _d for R = vec2<f32> (hi, lo); the two families are the same
// text and must stay so.
//
// Bindings (group 0): 0 mkl_env (common.wgsl), 1 the parameters, then
// 2 headers, 3 surfaces, 4 coefficient pool, 5 path program (step words),
// 6 input planes, 7 output planes, 8 records, every plane `stride` long and
// starting at the launch's first ray (the host packs each launch's rays):
//   mkl_trace           in:  MKL_TRACE_IN_FLOATS float planes (bits), then
//                            MKL_TRACE_IN_UINTS planes (wavelength, system)
//                       out: MKL_TRACE_OUT_FLOATS float planes, then
//                            MKL_TRACE_OUT_UINTS planes
//                       rec: [slot][ray][MKL_TRACE_RECORD_FLOATS] floats,
//                            then [slot][ray] status words
//   mkl_trace_resident  in:  15 float planes (aiming ray layout), system,
//                            wavelength, 2 id planes, 4 report planes
//   (+ _records)        out: the same 23 planes, then 3 status planes
//                            (status, last surface, steps)
//                       rec: [slot][ray][MKL_AIM_RECORD_FLOATS] hi, then lo

struct MklWgslTraceParams {
    trace: MklTraceParams,
    // Systems in the bound set (resident rays carry device indices).
    systems: u32,
    reserved0: u32,
    reserved1: u32,
    reserved2: u32,
}

@group(0) @binding(1) var<uniform> prm: MklWgslTraceParams;
@group(0) @binding(2) var<storage, read> mkl_headers: array<MklPackedHeader>;
@group(0) @binding(3) var<storage, read> mkl_surfaces: array<MklPackedSurface>;
@group(0) @binding(4) var<storage, read> mkl_pool: array<f32>;
@group(0) @binding(5) var<storage, read> mkl_steps: array<u32>;
@group(0) @binding(6) var<storage, read> mkl_in: array<u32>;
@group(0) @binding(7) var<storage, read_write> mkl_out: array<u32>;
@group(0) @binding(8) var<storage, read_write> mkl_records: array<u32>;

// The ray of this invocation and the plane stride.
var<private> g_ray: u32;
var<private> g_stride: u32;

// 2^-20: vertex-sheet tolerance (trace_kernel::SHEET_TOL).
fn mkl_sheet_tol() -> f32 {
    return bitcast<f32>(0x35800000u);
}
fn mkl_flt_min() -> f32 {
    return bitcast<f32>(0x00800000u);
}

// min() that ignores a NaN operand like MSL fmin / Rust f32::min.
fn mkl_min_num(a: f32, b: f32) -> f32 {
    if (mkl_is_nan(a)) {
        return b;
    }
    if (mkl_is_nan(b)) {
        return a;
    }
    return select(b, a, a <= b);
}

// a <= b, false when either is NaN.
fn mkl_le_num(a: f32, b: f32) -> bool {
    return !mkl_is_nan(a) && !mkl_is_nan(b) && a <= b;
}

// ---------------------------------------------------------------------------
// Scalar primitives of the two R types (surface_math.h's MklDf operators,
// mkl_c, mkl_parts, mkl_sqrt, mkl_le, mkl_zero, mkl_load).
// ---------------------------------------------------------------------------
fn add_f(a: f32, b: f32) -> f32 {
    return mkl_add(a, b);
}
fn sub_f(a: f32, b: f32) -> f32 {
    return mkl_sub(a, b);
}
fn mul_f(a: f32, b: f32) -> f32 {
    return mkl_mul(a, b);
}
fn div_f(a: f32, b: f32) -> f32 {
    return mkl_div(a, b);
}
fn neg_f(a: f32) -> f32 {
    return -a;
}
fn sqrt_f(a: f32) -> f32 {
    return mkl_sqrt(a);
}
fn c_f(v: f32) -> f32 {
    return v;
}
fn parts_f(hi: f32, lo: f32) -> f32 {
    return hi;
}
fn le_f(a: f32, b: f32) -> bool {
    return mkl_le_num(a, b);
}
fn zero_f(a: f32) -> bool {
    return a == 0.0;
}
// Element i of the double-float array at float index `base` of the pool.
fn load_f(base: u32, i: u32) -> f32 {
    return mkl_pool[base + 2u * i];
}

fn add_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_add(a, b);
}
fn sub_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_sub(a, b);
}
fn mul_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_mul(a, b);
}
fn div_d(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return mkl_df_div(a, b);
}
fn neg_d(a: vec2<f32>) -> vec2<f32> {
    return mkl_df_neg(a);
}
fn sqrt_d(a: vec2<f32>) -> vec2<f32> {
    return mkl_df_sqrt(a);
}
fn c_d(v: f32) -> vec2<f32> {
    return vec2<f32>(v, 0.0);
}
fn parts_d(hi: f32, lo: f32) -> vec2<f32> {
    return vec2<f32>(hi, lo);
}
fn le_d(a: vec2<f32>, b: f32) -> bool {
    return mkl_le_num(mkl_df_add_f32(a, -b).x, 0.0);
}
fn zero_d(a: vec2<f32>) -> bool {
    return a.x == 0.0;
}
fn load_d(base: u32, i: u32) -> vec2<f32> {
    return vec2<f32>(mkl_pool[base + 2u * i], mkl_pool[base + 2u * i + 1u]);
}

// First-order jets (value, d/dx, d/dy), the sag of a surface with its
// slopes, and the scaled pupil coordinates u, v and s = u^2 + v^2.
struct MklJetF {
    v: f32,
    x: f32,
    y: f32,
}
struct MklJetD {
    v: vec2<f32>,
    x: vec2<f32>,
    y: vec2<f32>,
}
struct MklSagF {
    ok: bool,
    j: MklJetF,
}
struct MklSagD {
    ok: bool,
    j: MklJetD,
}
struct MklUvsF {
    u: MklJetF,
    v: MklJetF,
    s: MklJetF,
}
struct MklUvsD {
    u: MklJetD,
    v: MklJetD,
    s: MklJetD,
}

// Float index of a surface's coefficient data in the pool.
fn mkl_surface_data(s: u32) -> u32 {
    return 2u * mkl_surfaces[s].coefficient_offset;
}
