// Regular transforms: FFT rows and stages, chirp-z, transpose, elementwise
// chirp/phase/spectral steps and the correlation stencils (transforms.metal's
// twins; the steps of mklight_gpu::transforms::recipe and the CPU-f32
// specification in mklight-gpu/src/transforms/cpu.rs, correlation.rs).
//
// Every kernel reads its MklTransformParams / MklCorrelationParams from a
// read-only storage binding (binding 2): the structs hold 4-byte-stride
// arrays, which uniform blocks cannot. The other storage buffers follow in
// the host's binding order (bindings 3..).
//
// Numerics: every complex product, sum and scale of the butterflies and the
// elementwise steps is written with the protected operations of common.wgsl,
// in the specification's order, so no fast-math compiler contracts or
// reassociates them. Twiddles (recipe::twiddles) and fused chirp factors
// (recipe::chirp_factors) are f64-derived host tables; the radix-3/5/7 DFT
// constants are the specification's f64 cos/sin rounded to f32. Phases formed
// on the device (chirp kernels, Phase, the optical correlation) are
// double-float turns, reduced modulo a turn and evaluated by p2_cis's own
// polynomial: no device sin/cos.

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn p2_cmul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(
        mkl_sub(mkl_mul(a.x, b.x), mkl_mul(a.y, b.y)),
        mkl_add(mkl_mul(a.x, b.y), mkl_mul(a.y, b.x)),
    );
}
fn p2_cadd(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(mkl_add(a.x, b.x), mkl_add(a.y, b.y));
}
fn p2_csub(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(mkl_sub(a.x, b.x), mkl_sub(a.y, b.y));
}
fn p2_cscale(a: vec2<f32>, s: f32) -> vec2<f32> {
    return vec2<f32>(mkl_mul(a.x, s), mkl_mul(a.y, s));
}
// v·(s·i), exact for s = ±1.
fn p2_rot(v: vec2<f32>, s: f32) -> vec2<f32> {
    return vec2<f32>(-s * v.y, s * v.x);
}

// exp(2πi·a) for double-float turns a: reduced modulo a turn (twice, ties to
// even, as cpu::cis), then to an octant r = f - q/4, |r| ≤ 1/8, and
// evaluated by Taylor polynomials of degree 9 (sin) and 10 (cos) in
// x = 2πr, |x| ≤ π/4 (truncation < 2e-9), rotated by q quarter turns.
fn p2_cis(a_in: vec2<f32>) -> vec2<f32> {
    var a = mkl_df_add_f32(a_in, -round(a_in.x));
    a = mkl_df_add_f32(a, -round(a.x));
    let q = round(mkl_mul(a.x, 4.0));
    // a.x - q/4 is exact (Sterbenz) for |a.x| ≤ 1/2.
    let r = mkl_add(mkl_sub(a.x, mkl_mul(q, 0.25)), a.y);
    let x = mkl_mul(r, 6.2831853071795864769);
    let x2 = mkl_mul(x, x);
    var s = mkl_fma(x2, 2.7557319e-6, -1.9841270e-4);
    s = mkl_fma(s, x2, 8.3333333e-3);
    s = mkl_fma(s, x2, -1.6666667e-1);
    s = mkl_fma(mkl_mul(s, x2), x, x);
    var c = mkl_fma(x2, -2.7557319e-7, 2.4801587e-5);
    c = mkl_fma(c, x2, -1.3888889e-3);
    c = mkl_fma(c, x2, 4.1666668e-2);
    c = mkl_fma(c, x2, -0.5);
    c = mkl_fma(c, x2, 1.0);
    let k = (i32(q) % 4 + 4) % 4;
    switch (k) {
        case 1: { return vec2<f32>(-s, c); }
        case 2: { return vec2<f32>(-c, -s); }
        case 3: { return vec2<f32>(s, -c); }
        default: { return vec2<f32>(c, s); }
    }
}

// ---------------------------------------------------------------------------
// Elementwise steps (MKL_TRANSFORM_KERNEL/PRE/POST/MULTIPLY/PHASE) and the
// table-driven chirp factors of long chirp rows. Storage: params, a, b, out.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var<storage, read> p2p: MklTransformParams;
@group(0) @binding(3) var<storage, read> e_a: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> e_b: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> e_out: array<vec2<f32>>;

fn p2_aff(i: u32) -> vec2<f32> {
    return vec2<f32>(p2p.affine[i][0], p2p.affine[i][1]);
}
fn p2_scale() -> f32 {
    return bitcast<f32>(p2p.scale_bits);
}
fn p2_sign() -> f32 {
    return f32(bitcast<i32>(p2p.sign_bits));
}
// 0.5·(a0·x² + 2·a1·x·y + a4·y²) in double-float turns (cpu::quadratic).
fn p2_q(x: f32, y: f32) -> vec2<f32> {
    let t0 = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(0u), x), x);
    let t1 = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(1u), mkl_mul(2.0, x)), y);
    let t2 = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(4u), y), y);
    return mkl_df_mul_f32(mkl_df_add(mkl_df_add(t0, t1), t2), 0.5);
}
// cpu::pre: q + a2·x + a5·y, left to right.
fn p2_pre(x: f32, y: f32) -> vec2<f32> {
    let t = mkl_df_add(p2_q(x, y), mkl_df_mul_f32(p2_aff(2u), x));
    return mkl_df_add(t, mkl_df_mul_f32(p2_aff(5u), y));
}

// The invocation's element of a 1-D grid launched as (x, y) groups of 256.
fn p2_linear(lid: vec3<u32>, wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return (wg.y * nwg.x + wg.x) * 256u + lid.x;
}

@compute @workgroup_size(256)
fn mkl_transform(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let id = p2_linear(lid, wg, nwg);
    if (id >= p2p.count) {
        return;
    }
    let w = p2p.width;
    let h = p2p.height;
    let scale = p2_scale();
    let sign = p2_sign();
    var z = vec2<f32>(0.0, 0.0);
    switch (p2p.mode) {
        case MKL_TRANSFORM_KERNEL: {
            let ix = id % w;
            let iy = id / w;
            if ((ix < p2p.ox || ix > w - p2p.nx) && (iy < p2p.oy || iy > h - p2p.ny)) {
                var x = f32(ix);
                if (ix >= p2p.ox) {
                    x = mkl_sub(f32(ix), f32(w));
                }
                var y = f32(iy);
                if (iy >= p2p.oy) {
                    y = mkl_sub(f32(iy), f32(h));
                }
                z = p2_cis(mkl_df_mul_f32(p2_q(x, y), -sign));
            }
        }
        case MKL_TRANSFORM_PRE: {
            let x = id % w;
            let y = id / w;
            if (x < p2p.nx && y < p2p.ny) {
                z = p2_cmul(e_a[y * p2p.nx + x], p2_cis(mkl_df_mul_f32(p2_pre(f32(x), f32(y)), sign)));
            }
        }
        case MKL_TRANSFORM_POST: {
            let x = id % p2p.ox;
            let y = id / p2p.ox;
            let v = p2_cmul(e_a[y * w + x], p2_cis(mkl_df_mul_f32(p2_q(f32(x), f32(y)), sign)));
            z = p2_cscale(v, scale);
        }
        case MKL_TRANSFORM_MULTIPLY: {
            var j = id;
            if ((p2p.flags & 2u) != 0u) {
                j = id % w;
            }
            z = p2_cscale(p2_cmul(e_a[id], e_b[j]), scale);
        }
        case MKL_TRANSFORM_PHASE: {
            let x = f32(id % w);
            let y = f32(id / w);
            var ph = mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(0u), x), x);
            ph = mkl_df_add(ph, mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(1u), x), y));
            ph = mkl_df_add(ph, mkl_df_mul_f32(mkl_df_mul_f32(p2_aff(2u), y), y));
            ph = mkl_df_add(ph, mkl_df_mul_f32(p2_aff(3u), x));
            ph = mkl_df_add(ph, mkl_df_mul_f32(p2_aff(4u), y));
            ph = mkl_df_add(ph, p2_aff(5u));
            z = p2_cmul(e_a[id], p2_cis(ph));
        }
        default: {}
    }
    e_out[id] = z;
}

// Pre-chirp of long chirp rows (padded to width, count = width·rows):
// a[row·nx + i]·chirp[i] for i < nx, else 0. b = recipe::chirp_factors.
@compute @workgroup_size(256)
fn mkl_chirp_pre(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let id = p2_linear(lid, wg, nwg);
    if (id >= p2p.count) {
        return;
    }
    let i = id % p2p.width;
    let row = id / p2p.width;
    var z = vec2<f32>(0.0, 0.0);
    if (i < p2p.nx) {
        z = p2_cmul(e_a[row * p2p.nx + i], e_b[i]);
    }
    e_out[id] = z;
}

// Post-chirp of long chirp rows (count = ox·rows):
// a[row·width + i]·chirp[nx + i]·scale.
@compute @workgroup_size(256)
fn mkl_chirp_post(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let id = p2_linear(lid, wg, nwg);
    if (id >= p2p.count) {
        return;
    }
    let i = id % p2p.ox;
    let row = id / p2p.ox;
    e_out[id] = p2_cscale(p2_cmul(e_a[row * p2p.width + i], e_b[p2p.nx + i]), p2_scale());
}

// ---------------------------------------------------------------------------
// Transpose: 32×32 tiles, 256 threads; the padded column avoids bank
// conflicts. Storage: params, a, out.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> t_a: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> t_out: array<vec2<f32>>;
var<workgroup> p2_tile: array<array<vec2<f32>, 33>, 32>;

@compute @workgroup_size(256)
fn mkl_transform_transpose(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let w = p2p.width;
    let h = p2p.height;
    let tx = tid % 32u;
    let ty = tid / 32u;
    let bx = group.x * 32u;
    let by = group.y * 32u;
    let scale = p2_scale();
    for (var j = ty; j < 32u; j += 8u) {
        if (bx + tx < w && by + j < h) {
            p2_tile[j][tx] = t_a[(by + j) * w + bx + tx];
        }
    }
    workgroupBarrier();
    for (var j = ty; j < 32u; j += 8u) {
        if (bx + j < w && by + tx < h) {
            t_out[(bx + j) * h + by + tx] = p2_cscale(p2_tile[tx][j], scale);
        }
    }
}

// ---------------------------------------------------------------------------
// Stockham FFT. Twiddles are f64-derived host tables (recipe::twiddles):
// stage span p, radix R, entry p-1+k(R-1)+t-1 = exp(-2πi·t·k/(pR)); sign +1
// conjugates. Storage: params, twiddles, a, out (, spectrum, chirp).
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> f_tw: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> f_a: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> f_out: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> f_spectrum: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> f_chirp: array<vec2<f32>>;

// cos/sin(2π·q/R) for R = 3, 5, 7 (rows), from f64 rounded to f32 as the
// specification's generic DFT.
const P2_COS = array<f32, 24>(
    1.0, -0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0,
    1.0, 0.309017003, -0.809017003, -0.809017003, 0.309017003, 0.0, 0.0, 0.0,
    1.0, 0.623489797, -0.222520933, -0.90096885, -0.90096885, -0.222520933, 0.623489797, 0.0,
);
const P2_SIN = array<f32, 24>(
    0.0, 0.866025388, -0.866025388, 0.0, 0.0, 0.0, 0.0, 0.0,
    0.0, 0.95105654, 0.587785244, -0.587785244, -0.95105654, 0.0, 0.0, 0.0,
    0.0, 0.781831503, 0.974927902, 0.433883727, -0.433883727, -0.974927902, -0.781831503, 0.0,
);
const P2_SQRT1_2: f32 = 0.707106769;

alias P2Row = array<vec2<f32>, 8>;

fn p2_dft4(u: ptr<function, P2Row>, i0: u32, i1: u32, i2: u32, i3: u32, s: f32) {
    let a0 = p2_cadd((*u)[i0], (*u)[i2]);
    let a1 = p2_csub((*u)[i0], (*u)[i2]);
    let a2 = p2_cadd((*u)[i1], (*u)[i3]);
    let a3 = p2_rot(p2_csub((*u)[i1], (*u)[i3]), s);
    (*u)[i0] = p2_cadd(a0, a2);
    (*u)[i2] = p2_csub(a0, a2);
    (*u)[i1] = p2_cadd(a1, a3);
    (*u)[i3] = p2_csub(a1, a3);
}

// Unnormalised DFT of r values, exp(s·2πi·m·t/r) (cpu::dft).
fn p2_dft(u: ptr<function, P2Row>, r: u32, s: f32) {
    switch (r) {
        case 2u: {
            let a = (*u)[0];
            (*u)[0] = p2_cadd(a, (*u)[1]);
            (*u)[1] = p2_csub(a, (*u)[1]);
        }
        case 4u: {
            p2_dft4(u, 0u, 1u, 2u, 3u, s);
        }
        case 8u: {
            let c = P2_SQRT1_2;
            p2_dft4(u, 0u, 2u, 4u, 6u, s);
            p2_dft4(u, 1u, 3u, 5u, 7u, s);
            let u3 = (*u)[3];
            let u7 = (*u)[7];
            let o0 = (*u)[1];
            let o1 = vec2<f32>(mkl_mul(c, mkl_sub(u3.x, s * u3.y)), mkl_mul(c, mkl_add(u3.y, s * u3.x)));
            let o2 = p2_rot((*u)[5], s);
            let o3 = vec2<f32>(mkl_mul(c, mkl_sub(-u7.x, s * u7.y)), mkl_mul(c, mkl_sub(s * u7.x, u7.y)));
            let e0 = (*u)[0];
            let e1 = (*u)[2];
            let e2 = (*u)[4];
            let e3 = (*u)[6];
            (*u)[0] = p2_cadd(e0, o0);
            (*u)[4] = p2_csub(e0, o0);
            (*u)[1] = p2_cadd(e1, o1);
            (*u)[5] = p2_csub(e1, o1);
            (*u)[2] = p2_cadd(e2, o2);
            (*u)[6] = p2_csub(e2, o2);
            (*u)[3] = p2_cadd(e3, o3);
            (*u)[7] = p2_csub(e3, o3);
        }
        default: {
            var row = 2u;
            if (r == 3u) {
                row = 0u;
            } else if (r == 5u) {
                row = 1u;
            }
            var o: P2Row;
            for (var m = 0u; m < r; m++) {
                var acc = (*u)[0];
                for (var t = 1u; t < r; t++) {
                    let q = row * 8u + (m * t) % r;
                    acc = p2_cadd(acc, p2_cmul((*u)[t], vec2<f32>(P2_COS[q], s * P2_SIN[q])));
                }
                o[m] = acc;
            }
            for (var m = 0u; m < r; m++) {
                (*u)[m] = o[m];
            }
        }
    }
}

// Twiddle and butterfly b (of n/r) at span p; returns the output base j.
fn p2_butterfly(v: ptr<function, P2Row>, r: u32, b: u32, span: u32, s: f32) -> u32 {
    let k = b % span;
    let w0 = span - 1u + k * (r - 1u);
    for (var t = 1u; t < r; t++) {
        let c = f_tw[w0 + t - 1u];
        (*v)[t] = p2_cmul((*v)[t], vec2<f32>(c.x, -s * c.y));
    }
    p2_dft(v, r, s);
    return (b - k) * r + k;
}

fn p2_radix(i: u32) -> u32 {
    return (p2p.radices[i / 8u] >> (4u * (i % 8u))) & 15u;
}

// Shared memory of the row kernels: 2048 complex values (the 16 KiB WebGPU
// limit), E = 8 elements per thread, T = ⌈n/8⌉ threads per row, G =
// 2048/(8T) rows per group of 256 threads (recipe::row_geometry).
const P2_SH: u32 = 2048u;
const P2_E: u32 = 8u;
var<workgroup> p2_sh: array<vec2<f32>, 2048>;

struct P2Lane {
    base: u32,
    lane: u32,
    threads: u32,
    // Threads past G·T hold no row but join every barrier.
    holds: bool,
}

// One in-place Stockham stage in shared memory. T lanes cover n/r
// butterflies in K = ⌈8/r⌉ rounds; all reads precede the barrier and writes.
fn p2_stage(l: P2Lane, n: u32, r: u32, span: u32, s: f32) {
    let m = n / r;
    let rounds = (P2_E + r - 1u) / r;
    var v: array<vec2<f32>, 16>;
    for (var q = 0u; q < rounds; q++) {
        let b = l.lane + q * l.threads;
        if (l.holds && b < m) {
            for (var t = 0u; t < r; t++) {
                v[q * r + t] = p2_sh[l.base + b + t * m];
            }
        }
    }
    workgroupBarrier();
    for (var q = 0u; q < rounds; q++) {
        let b = l.lane + q * l.threads;
        if (l.holds && b < m) {
            var u: P2Row;
            for (var t = 0u; t < r; t++) {
                u[t] = v[q * r + t];
            }
            let j = p2_butterfly(&u, r, b, span, s);
            for (var t = 0u; t < r; t++) {
                p2_sh[l.base + j + t * span] = u[t];
            }
        }
    }
    workgroupBarrier();
}

fn p2_fft(l: P2Lane, n: u32, stages: u32, s: f32) {
    var span = 1u;
    for (var i = 0u; i < stages; i++) {
        let r = p2_radix(i);
        p2_stage(l, n, r, span, s);
        span *= r;
    }
}

fn p2_lane(tid: u32, n: u32) -> P2Lane {
    let threads = (n + P2_E - 1u) / P2_E;
    let rows = P2_SH / (P2_E * threads);
    let slot = tid / threads;
    return P2Lane(slot * n, tid % threads, threads, slot < rows);
}

fn p2_row(tid: u32, group: u32, n: u32) -> u32 {
    let threads = (n + P2_E - 1u) / P2_E;
    return group * (P2_SH / (P2_E * threads)) + tid / threads;
}

// `height` independent rows of `width` points, whole rows in shared memory.
@compute @workgroup_size(256)
fn mkl_fft_rows(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let n = p2p.width;
    let stages = p2p.stages;
    let l = p2_lane(tid, n);
    let row = p2_row(tid, group.x, n);
    let live = l.holds && row < p2p.height;
    if (live) {
        for (var i = l.lane; i < n; i += l.threads) {
            p2_sh[l.base + i] = f_a[row * n + i];
        }
    }
    workgroupBarrier();
    p2_fft(l, n, stages, p2_sign());
    let scale = p2_scale();
    if (live) {
        for (var i = l.lane; i < n; i += l.threads) {
            f_out[row * n + i] = p2_cscale(p2_sh[l.base + i], scale);
        }
    }
}

// Exact chirp-z of a row, fused: pre-chirp, FFT, kernel spectrum, inverse
// FFT, post-chirp. f_chirp holds nx pre- then ox post-factors.
@compute @workgroup_size(256)
fn mkl_chirp_rows(
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) group: vec3<u32>,
) {
    mkl_enter();
    let n = p2p.width;
    let nx = p2p.nx;
    let stages = p2p.stages;
    let l = p2_lane(tid, n);
    let row = p2_row(tid, group.x, n);
    let live = l.holds && row < p2p.height;
    if (l.holds) {
        for (var i = l.lane; i < n; i += l.threads) {
            var z = vec2<f32>(0.0, 0.0);
            if (live && i < nx) {
                z = p2_cmul(f_a[row * nx + i], f_chirp[i]);
            }
            p2_sh[l.base + i] = z;
        }
    }
    workgroupBarrier();
    p2_fft(l, n, stages, -1.0);
    if (l.holds) {
        for (var i = l.lane; i < n; i += l.threads) {
            p2_sh[l.base + i] = p2_cmul(p2_sh[l.base + i], f_spectrum[i]);
        }
    }
    workgroupBarrier();
    p2_fft(l, n, stages, 1.0);
    let scale = p2_scale();
    if (live) {
        for (var i = l.lane; i < p2p.ox; i += l.threads) {
            f_out[row * p2p.ox + i] = p2_cscale(p2_cmul(p2_sh[l.base + i], f_chirp[nx + i]), scale);
        }
    }
}

// One global Stockham stage (radix radices[0], span) for rows longer than
// the shared memory holds: grid (butterfly, row) in groups of 256.
@compute @workgroup_size(256)
fn mkl_fft_stage(
    @builtin(global_invocation_id) id: vec3<u32>,
) {
    mkl_enter();
    let n = p2p.width;
    let r = p2p.radices[0];
    let m = n / r;
    let b = id.x;
    if (b >= m || id.y >= p2p.height) {
        return;
    }
    let base = id.y * n;
    let s = p2_sign();
    let scale = p2_scale();
    var v: P2Row;
    for (var t = 0u; t < r; t++) {
        v[t] = f_a[base + b + t * m];
    }
    let j = p2_butterfly(&v, r, b, p2p.span, s);
    for (var t = 0u; t < r; t++) {
        f_out[base + j + t * p2p.span] = p2_cscale(v[t], scale);
    }
}

// ---------------------------------------------------------------------------
// Correlation over a tile of frequencies (correlation.rs): grid (sample k,
// tile frequency f), the tile's frequencies first..first+frequencies. The
// stencil and terms planes are the tile's own.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var<storage, read> p2c: MklCorrelationParams;

fn p2_cdf(v: array<f32, 2>) -> vec2<f32> {
    return vec2<f32>(v[0], v[1]);
}
fn p2_frame(i: u32, j: u32) -> vec2<f32> {
    return vec2<f32>(p2c.frame[i][j][0], p2c.frame[i][j][1]);
}
// (a0·b0 + a1·b1) + a2·b2 with frame row i.
fn p2_dot_frame(a: array<vec2<f32>, 3>, i: u32) -> vec2<f32> {
    return mkl_df_add(
        mkl_df_add(mkl_df_mul(a[0], p2_frame(i, 0u)), mkl_df_mul(a[1], p2_frame(i, 1u))),
        mkl_df_mul(a[2], p2_frame(i, 2u)),
    );
}

struct P2Geometry {
    x: f32,
    y: f32,
    z: f32,
    gamma: f32,
    source_z: f32,
}

fn p2_geometry(k: u32, nu: vec2<f32>) -> P2Geometry {
    let n = p2c.n;
    let sin_max = p2_cdf(p2c.sin_max);
    let a = mkl_df_mul_f32(sin_max, mkl_sub(mkl_div(f32(2u * (k % n) + 1u), f32(n)), 1.0));
    let b = mkl_df_mul_f32(sin_max, mkl_sub(mkl_div(f32(2u * (k / n) + 1u), f32(n)), 1.0));
    let one = vec2<f32>(1.0, 0.0);
    let gamma = mkl_df_sqrt(mkl_df_sub(mkl_df_sub(one, mkl_df_mul(a, a)), mkl_df_mul(b, b)));
    var q: array<vec2<f32>, 3>;
    for (var i = 0u; i < 3u; i++) {
        q[i] = mkl_df_sub(
            mkl_df_add(mkl_df_mul(p2_frame(0u, i), a), mkl_df_mul(p2_frame(1u, i), b)),
            mkl_df_mul(p2_frame(2u, i), gamma),
        );
    }
    let wl = p2_cdf(p2c.wavelength);
    let sx = mkl_df_add(q[0], mkl_df_mul_f32(wl, nu.x));
    let sy = mkl_df_add(q[1], mkl_df_mul_f32(wl, nu.y));
    let z2 = mkl_df_sub(mkl_df_sub(one, mkl_df_mul(sx, sx)), mkl_df_mul(sy, sy));
    var g = P2Geometry(mkl_qnan(), mkl_qnan(), 0.0, gamma.x, -q[2].x);
    if (z2.x <= 0.0 || q[2].x >= 0.0 || !mkl_is_finite(gamma.x)) {
        return g;
    }
    let sz = mkl_df_neg(mkl_df_sqrt(z2));
    let shifted = array<vec2<f32>, 3>(sx, sy, sz);
    if (p2_dot_frame(shifted, 2u).x >= 0.0) {
        return g;
    }
    let scale = mkl_df_div(vec2<f32>(mkl_mul(0.5, f32(n)), 0.0), sin_max);
    let offset = mkl_mul(0.5, f32(n - 1u));
    let ix = mkl_df_add_f32(mkl_df_mul(p2_dot_frame(shifted, 0u), scale), offset);
    let iy = mkl_df_add_f32(mkl_df_mul(p2_dot_frame(shifted, 1u), scale), offset);
    g.x = mkl_add(ix.x, ix.y);
    g.y = mkl_add(iy.x, iy.y);
    g.z = -sz.x;
    return g;
}

// Storage: params, frequencies (whole), stencil tile (rw).
@group(0) @binding(3) var<storage, read> s_freq: array<f32>;
@group(0) @binding(4) var<storage, read_write> s_dst: array<f32>;

@compute @workgroup_size(256)
fn mkl_optical_stencil(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    let k = id.x;
    let count = p2c.n * p2c.height;
    if (k >= count || id.y >= p2c.frequencies) {
        return;
    }
    let f = p2c.first + id.y;
    let g = p2_geometry(k, vec2<f32>(s_freq[2u * f], s_freq[2u * f + 1u]));
    let o = (id.y * count + k) * 5u;
    s_dst[o] = g.x;
    s_dst[o + 1u] = g.y;
    s_dst[o + 2u] = g.z;
    s_dst[o + 3u] = g.gamma;
    s_dst[o + 4u] = g.source_z;
}

// Storage: params, field, opd, coverage, frequencies, stencil tile, terms.
// Writes channels (real, imaginary, energy) of tile frequency f at
// terms[(3f + c)·count + k].
@group(0) @binding(3) var<storage, read> c_field: array<f32>;
@group(0) @binding(4) var<storage, read> c_opd: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> c_cov: array<f32>;
@group(0) @binding(6) var<storage, read> c_freq: array<f32>;
@group(0) @binding(7) var<storage, read> c_stencil: array<f32>;
@group(0) @binding(8) var<storage, read_write> c_terms: array<f32>;

fn p2_coverage(j: u32) -> f32 {
    if (p2c.covered != 0u) {
        return c_cov[j];
    }
    return 1.0;
}

@compute @workgroup_size(256)
fn mkl_correlation_terms(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    let k = id.x;
    let count = p2c.n * p2c.height;
    if (k >= count || id.y >= p2c.frequencies) {
        return;
    }
    let f = p2c.first + id.y;
    let base = 3u * id.y * count;
    c_terms[base + k] = 0.0;
    c_terms[base + count + k] = 0.0;
    c_terms[base + 2u * count + k] = 0.0;
    let c = p2_coverage(k);
    if (c <= 0.0) {
        return;
    }
    let nu = vec2<f32>(c_freq[2u * f], c_freq[2u * f + 1u]);
    let n = p2c.n;
    let optical = p2c.optical != 0u;
    var g: P2Geometry;
    if (optical) {
        if (p2c.prepared != 0u) {
            let o = (id.y * count + k) * 5u;
            g = P2Geometry(c_stencil[o], c_stencil[o + 1u], c_stencil[o + 2u], c_stencil[o + 3u], c_stencil[o + 4u]);
        } else {
            g = p2_geometry(k, nu);
        }
    } else {
        let m = p2c.mapping;
        let sx = mkl_df_add(
            mkl_df_add(mkl_df_mul_f32(p2_cdf(m[0]), nu.x), mkl_df_mul_f32(p2_cdf(m[1]), nu.y)),
            p2_cdf(m[2]),
        );
        let sy = mkl_df_add(
            mkl_df_add(mkl_df_mul_f32(p2_cdf(m[3]), nu.x), mkl_df_mul_f32(p2_cdf(m[4]), nu.y)),
            p2_cdf(m[5]),
        );
        g = P2Geometry(
            mkl_add(mkl_add(f32(k % n), sx.x), sx.y),
            mkl_add(mkl_add(f32(k / n), sy.x), sy.y),
            1.0, 1.0, 1.0,
        );
    }
    var power: f32;
    if (optical) {
        power = mkl_mul(c_field[k], c_field[k]);
    } else {
        power = mkl_add(mkl_mul(c_field[2u * k], c_field[2u * k]), mkl_mul(c_field[2u * k + 1u], c_field[2u * k + 1u]));
    }
    let energy = mkl_div(mkl_mul(c, power), mkl_mul(g.gamma, g.source_z));
    c_terms[base + 2u * count + k] = energy;
    if (!mkl_is_finite(g.x) || !mkl_is_finite(g.y)) {
        return;
    }
    let fx = floor(g.x);
    let fy = floor(g.y);
    let tx = mkl_sub(g.x, fx);
    let ty = mkl_sub(g.y, fy);
    var ar = 0.0;
    var ai = 0.0;
    var coverage = 0.0;
    var phase = vec2<f32>(0.0, 0.0);
    for (var corner = 0; corner < 4; corner++) {
        let dx = corner % 2;
        let dy = corner / 2;
        let xi = i32(fx) + dx;
        let yi = i32(fy) + dy;
        if (xi < 0 || yi < 0 || xi >= i32(n) || yi >= i32(p2c.height)) {
            continue;
        }
        let j = u32(yi) * n + u32(xi);
        var wx = mkl_sub(1.0, tx);
        if (dx == 1) {
            wx = tx;
        }
        var wy = mkl_sub(1.0, ty);
        if (dy == 1) {
            wy = ty;
        }
        let w = mkl_mul(mkl_mul(wx, wy), p2_coverage(j));
        if (w == 0.0) {
            continue;
        }
        if (optical) {
            ar = mkl_add(ar, mkl_mul(w, c_field[j]));
            coverage = mkl_add(coverage, w);
            phase = mkl_df_add(phase, mkl_df_mul_f32(c_opd[j], w));
        } else {
            ar = mkl_add(ar, mkl_mul(w, c_field[2u * j]));
            ai = mkl_add(ai, mkl_mul(w, c_field[2u * j + 1u]));
        }
    }
    let ex = max(0.0, mkl_sub(1.0, abs(mkl_sub(g.x, f32(k % n)))));
    let ey = max(0.0, mkl_sub(1.0, abs(mkl_sub(g.y, f32(k / n)))));
    let deficit = mkl_mul(mkl_mul(mkl_mul(mkl_sub(1.0, c), energy), ex), ey);
    var v = vec2<f32>(0.0, 0.0);
    if (optical) {
        if (coverage > 0.0) {
            let z = p2_cis(mkl_df_sub(mkl_df_div(phase, vec2<f32>(coverage, 0.0)), c_opd[k]));
            let m = mkl_div(mkl_mul(mkl_div(mkl_mul(c, c_field[k]), g.gamma), ar), g.z);
            v = p2_cscale(z, m);
        }
    } else {
        let fr = c_field[2u * k];
        let fi = c_field[2u * k + 1u];
        v = vec2<f32>(
            mkl_mul(c, mkl_add(mkl_mul(ar, fr), mkl_mul(ai, fi))),
            mkl_mul(c, mkl_sub(mkl_mul(ai, fr), mkl_mul(ar, fi))),
        );
    }
    c_terms[base + k] = mkl_add(v.x, deficit);
    c_terms[base + count + k] = v.y;
}

// Storage: params, the tile's double-float channel sums, out (the tile's
// normalised complex values).
@group(0) @binding(3) var<storage, read> n_sums: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> n_out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn mkl_correlation_normalize(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    let f = id.x;
    if (f >= p2c.frequencies) {
        return;
    }
    let a = mkl_df_div(n_sums[3u * f], n_sums[3u * f + 2u]);
    let b = mkl_df_div(n_sums[3u * f + 1u], n_sums[3u * f + 2u]);
    n_out[f] = vec2<f32>(mkl_add(a.x, a.y), mkl_add(b.x, b.y));
}
