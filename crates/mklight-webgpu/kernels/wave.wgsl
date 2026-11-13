// Scalar-wave operations (wave_ops.metal's twins; the CPU-f32 specification
// in mklight-gpu/src/wave_ops/{spec,cpu,spectral,image_integrals}.rs).
//
// Every kernel reads its parameter struct (MklWaveParams,
// MklWaveSpectralParams, MklWaveIntegralParams) from a read-only storage
// binding (binding 2): the structs hold 4-byte-stride arrays, which uniform
// blocks cannot. The other storage buffers follow (bindings 3..).
//
// Numerics: every operation is the specification's, in its order, through
// the protected operations of common.wgsl; divisions and square roots are
// rounded to nearest (rn.wgsl, prepended to this module), so everything but
// the trigonometry is bitwise the specification: focus phases, phase steps,
// moments, intensities, line spreads and image metrics. The specification's
// f32 sin/cos are replaced by w_cis (double-float turns, reduced, evaluated
// by polynomials: no device sin/cos), within a few ulp of them.
//
// Sums keep the specification's fixed order: SUM_BLOCK (256) consecutive
// samples per thread in sample order, then the blocks in block order.

const W_BLOCK: u32 = 256u;

fn w_df(v: array<f32, 2>) -> vec2<f32> {
    return vec2<f32>(v[0], v[1]);
}
// spec::scalar: hi + lo.
fn w_scalar(v: vec2<f32>) -> f32 {
    return mkl_add(v.x, v.y);
}
fn w_lift(x: f32) -> vec2<f32> {
    return vec2<f32>(x, 0.0);
}

// The invocation's element of a 1-D grid launched as (x, y) groups of 256.
fn w_linear(lid: vec3<u32>, wg: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return (wg.y * nwg.x + wg.x) * 256u + lid.x;
}

// exp(2πi·a) for double-float turns a (transforms.wgsl's p2_cis): reduced
// modulo a turn twice, ties to even (as spec::cis), then to an octant and
// evaluated by Taylor polynomials of degree 9/10 (truncation < 2e-9).
fn w_cis(a_in: vec2<f32>) -> vec2<f32> {
    var a = mkl_df_add_f32(a_in, -round(a_in.x));
    a = mkl_df_add_f32(a, -round(a.x));
    let q = round(mkl_mul(a.x, 4.0));
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

// (cos x, sin x) of an f32 angle in radians: x/2π in double-float turns
// (1/2π as hi + lo), then w_cis.
fn w_cos_sin(x: f32) -> vec2<f32> {
    let inv_tau = vec2<f32>(bitcast<f32>(0x3E22F983u), bitcast<f32>(0x31DC9C88u));
    return w_cis(mkl_df_mul_f32(inv_tau, x));
}

// ---------------------------------------------------------------------------
// Pupil geometry (spec::direction, spec::focus) from MklWaveParams.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var<storage, read> wp: MklWaveParams;

struct WDir {
    a: vec2<f32>,
    b: vec2<f32>,
    z: vec2<f32>,
}

// (2i+1)/n - 1 in double-float.
fn w_uv(i: u32, n: u32) -> vec2<f32> {
    return mkl_df_sub(mkl_df_div_rn(w_lift(f32(2u * i + 1u)), w_lift(f32(n))), w_lift(1.0));
}

fn w_dir(k: u32) -> WDir {
    let n = wp.n;
    let sm = w_df(wp.sin_max);
    let a = mkl_df_mul(sm, w_uv(k % n, n));
    let b = mkl_df_mul(sm, w_uv(k / n, n));
    let g = mkl_df_sqrt_rn(mkl_df_sub(mkl_df_sub(w_lift(1.0), mkl_df_mul(a, a)), mkl_df_mul(b, b)));
    return WDir(a, b, mkl_df_neg(g));
}

// Exact refocus OPD (waves) of pupil sample k for a chief-axis shift dz.
fn w_focus(k: u32, dz: vec2<f32>) -> vec2<f32> {
    if (dz.x == 0.0 && dz.y == 0.0) {
        return vec2<f32>(0.0, 0.0);
    }
    let q = w_dir(k);
    let gamma = mkl_df_neg(q.z);
    let r = w_df(wp.radius);
    let s2 = mkl_df_add(mkl_df_mul(q.a, q.a), mkl_df_mul(q.b, q.b));
    let one_gamma = mkl_df_add(w_lift(1.0), gamma);
    var dr: vec2<f32>;
    if (r.x == 0.0) {
        dr = mkl_df_div_rn(mkl_df_mul(dz, s2), one_gamma);
    } else {
        let rd = mkl_df_mul_f32(mkl_df_mul(r, dz), 2.0);
        let delta = mkl_df_add(mkl_df_mul(rd, gamma), mkl_df_mul(dz, dz));
        let path = mkl_df_sqrt_rn(mkl_df_add(mkl_df_mul(r, r), delta));
        var axial = mkl_df_add(r, dz);
        if (axial.x < 0.0) {
            axial = mkl_df_neg(axial);
        }
        if (q.a.x == 0.0 && q.b.x == 0.0) {
            return vec2<f32>(0.0, 0.0);
        }
        dr = mkl_df_div_rn(mkl_df_mul(rd, s2), mkl_df_mul(one_gamma, mkl_df_add(axial, path)));
    }
    return mkl_df_div_rn(dr, w_df(wp.wavelength));
}

// ---------------------------------------------------------------------------
// Focus: phases of planes first..first+count, [count, n²] double-float:
// opd + focus where covered, opd elsewhere. Storage: params, coverage, opd,
// shifts (whole), out.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> f_cov: array<f32>;
@group(0) @binding(4) var<storage, read> f_opd: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read> f_shifts: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> f_out: array<vec2<f32>>;

@compute @workgroup_size(256)
fn mkl_wave_focus(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let t = w_linear(lid, wg, nwg);
    let count = wp.n * wp.n;
    if (t >= wp.count * count) {
        return;
    }
    let k = t % count;
    var v = f_opd[k];
    if (f_cov[k] > 0.0) {
        v = mkl_df_add(v, w_focus(k, f_shifts[wp.first + t / count]));
    }
    f_out[t] = v;
}

// The on-axis amplitude divisor -scalar(direction(j).z) of every sample.
// Storage: params, out.
@group(0) @binding(3) var<storage, read_write> g_out: array<f32>;

@compute @workgroup_size(256)
fn mkl_wave_gamma(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let k = w_linear(lid, wg, nwg);
    if (k >= wp.n * wp.n) {
        return;
    }
    g_out[k] = -w_scalar(w_dir(k).z);
}

// ---------------------------------------------------------------------------
// Phase steps (cpu::phase_step): the largest |phase difference| between
// covered neighbours of each plane. Blocks of 256 samples, then the blocks'
// maximum (order-free). Storage: params, coverage, phases [count, n²],
// block maxima [count, blocks], out [count].
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> p_cov: array<f32>;
@group(0) @binding(4) var<storage, read> p_phase: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> p_blocks: array<f32>;
@group(0) @binding(6) var<storage, read_write> p_out: array<f32>;

@compute @workgroup_size(256)
fn mkl_wave_step_blocks(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let t = w_linear(lid, wg, nwg);
    let n = wp.n;
    let count = n * n;
    let blocks = (count + W_BLOCK - 1u) / W_BLOCK;
    if (t >= wp.count * blocks) {
        return;
    }
    let base = (t / blocks) * count;
    let first = (t % blocks) * W_BLOCK;
    var step = 0.0;
    for (var k = first; k < min(first + W_BLOCK, count); k++) {
        if (p_cov[k] <= 0.0) {
            continue;
        }
        let v = p_phase[base + k];
        if (k % n + 1u < n && p_cov[k + 1u] > 0.0) {
            step = mkl_max_num(step, abs(w_scalar(mkl_df_sub(p_phase[base + k + 1u], v))));
        }
        if (k / n + 1u < n && p_cov[k + n] > 0.0) {
            step = mkl_max_num(step, abs(w_scalar(mkl_df_sub(p_phase[base + k + n], v))));
        }
    }
    p_blocks[t] = step;
}

@compute @workgroup_size(64)
fn mkl_wave_step_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    let plane = id.x;
    if (plane >= wp.count) {
        return;
    }
    let blocks = (wp.n * wp.n + W_BLOCK - 1u) / W_BLOCK;
    var step = 0.0;
    for (var b = 0u; b < blocks; b++) {
        step = mkl_max_num(step, p_blocks[plane * blocks + b]);
    }
    p_out[plane] = step;
}

// ---------------------------------------------------------------------------
// Focus correlation (spec::correlation_term over focused phases, dz = 0).
// Outputs o = plane·F + f (F = field_side frequencies) of the batch's
// planes; a thread sums block b of 256 samples of output first + t/blocks
// into (real, imaginary, energy) double-floats; the finish kernel adds the
// blocks in order and normalises. The batch's phases start at plane
// reserved[0], its outputs at output reserved[1], and the bound stencil
// slice at frequency sample_count. Storage: params, amplitude, coverage,
// phases [planes, n²], stencil [frequencies, n², 5], on-axis divisors,
// block sums.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> c_amp: array<f32>;
@group(0) @binding(4) var<storage, read> c_cov: array<f32>;
@group(0) @binding(5) var<storage, read> c_phase: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> c_stencil: array<f32>;
@group(0) @binding(7) var<storage, read> c_gamma: array<f32>;
@group(0) @binding(8) var<storage, read_write> c_acc: array<vec2<f32>>;

@compute @workgroup_size(256)
fn mkl_wave_focus_correlation(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let t = w_linear(lid, wg, nwg);
    let blocks = wp.sample_first;
    if (t >= wp.count * blocks) {
        return;
    }
    let n = wp.n;
    let count = n * n;
    let o = wp.first + t / blocks;
    let frequency = o % wp.field_side;
    // Phases of the batch's planes from plane reserved[0].
    let base = (o / wp.field_side - wp.reserved[0]) * count;
    let first = (t % blocks) * W_BLOCK;
    let on_axis = wp.shifted != 0u;
    var re = vec2<f32>(0.0, 0.0);
    var im = vec2<f32>(0.0, 0.0);
    var sum = vec2<f32>(0.0, 0.0);
    for (var k = first; k < min(first + W_BLOCK, count); k++) {
        let c = c_cov[k];
        if (c <= 0.0) {
            re = mkl_df_add_f32(re, 0.0);
            im = mkl_df_add_f32(im, 0.0);
            sum = mkl_df_add_f32(sum, 0.0);
            continue;
        }
        let off = ((frequency - wp.sample_count) * count + k) * 5u;
        let ix = c_stencil[off];
        let iy = c_stencil[off + 1u];
        let qz = c_stencil[off + 2u];
        let gamma = c_stencil[off + 3u];
        let srcz = c_stencil[off + 4u];
        let ak = c_amp[k];
        let energy = mkl_div_rn(mkl_mul(mkl_mul(c, ak), ak), mkl_mul(gamma, srcz));
        var t0 = 0.0;
        var t1 = 0.0;
        if (mkl_is_finite(ix) && mkl_is_finite(iy)) {
            let fx = floor(ix);
            let fy = floor(iy);
            let tx = mkl_sub(ix, fx);
            let ty = mkl_sub(iy, fy);
            var ar = 0.0;
            var cov = 0.0;
            var phase = vec2<f32>(0.0, 0.0);
            // Corners outside [-1, n] cannot touch the grid (and keep the
            // integer conversions in range).
            if (fx >= -1.0 && fy >= -1.0 && fx <= f32(n) && fy <= f32(n)) {
                for (var corner = 0; corner < 4; corner++) {
                    let dx = corner % 2;
                    let dy = corner / 2;
                    let xi = i32(fx) + dx;
                    let yi = i32(fy) + dy;
                    if (xi < 0 || yi < 0 || xi >= i32(n) || yi >= i32(n)) {
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
                    let weight = mkl_mul(mkl_mul(wx, wy), c_cov[j]);
                    if (weight == 0.0) {
                        continue;
                    }
                    var amp = mkl_mul(weight, c_amp[j]);
                    if (on_axis) {
                        amp = mkl_div_rn(amp, c_gamma[j]);
                    }
                    ar = mkl_add(ar, amp);
                    cov = mkl_add(cov, weight);
                    phase = mkl_df_add(phase, mkl_df_mul_f32(c_phase[base + j], weight));
                }
            }
            let ex = max(mkl_sub(1.0, abs(mkl_sub(ix, f32(k % n)))), 0.0);
            let ey = max(mkl_sub(1.0, abs(mkl_sub(iy, f32(k / n)))), 0.0);
            let deficit = mkl_mul(mkl_mul(mkl_mul(mkl_sub(1.0, c), energy), ex), ey);
            t0 = deficit;
            if (cov > 0.0) {
                let z = w_cis(mkl_df_sub(mkl_df_div_rn(phase, w_lift(cov)), c_phase[base + k]));
                var m = mkl_mul(mkl_div_rn(mkl_mul(c, ak), gamma), ar);
                if (!on_axis) {
                    m = mkl_div_rn(m, qz);
                }
                t0 = mkl_add(mkl_mul(m, z.x), deficit);
                t1 = mkl_mul(m, z.y);
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
@group(0) @binding(3) var<storage, read> cf_acc: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read_write> cf_out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn mkl_wave_focus_correlation_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    if (id.x >= wp.count) {
        return;
    }
    let blocks = wp.sample_first;
    var re = vec2<f32>(0.0, 0.0);
    var im = vec2<f32>(0.0, 0.0);
    var sum = vec2<f32>(0.0, 0.0);
    for (var b = 0u; b < blocks; b++) {
        let j = 3u * (id.x * blocks + b);
        re = mkl_df_add(re, cf_acc[j]);
        im = mkl_df_add(im, cf_acc[j + 1u]);
        sum = mkl_df_add(sum, cf_acc[j + 2u]);
    }
    // The batch's outputs from output reserved[1].
    cf_out[wp.first - wp.reserved[1] + id.x] = vec2<f32>(
        w_scalar(mkl_df_div_rn(re, sum)),
        w_scalar(mkl_df_div_rn(im, sum)),
    );
}

// ---------------------------------------------------------------------------
// Prepare (spec::prepare): the Debye field padded at the low corner of a
// field_side² grid and the three moment planes. Storage: params, amplitude,
// coverage, opd, field, moments.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> r_amp: array<f32>;
@group(0) @binding(4) var<storage, read> r_cov: array<f32>;
@group(0) @binding(5) var<storage, read> r_opd: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read_write> r_field: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read_write> r_mom: array<f32>;

@compute @workgroup_size(256)
fn mkl_wave_prepare(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let k = w_linear(lid, wg, nwg);
    let side = wp.field_side;
    if (k >= side * side) {
        return;
    }
    let n = wp.n;
    let x = k % side;
    let y = k / side;
    var z = vec2<f32>(0.0, 0.0);
    if (x < n && y < n) {
        let j = y * n + x;
        let count = n * n;
        let c = r_cov[j];
        var m = vec3<f32>(0.0, 0.0, 0.0);
        if (c > 0.0) {
            let q = w_dir(j);
            let gamma = -w_scalar(q.z);
            let a = r_amp[j];
            let d = mkl_df_add(
                mkl_df_add(mkl_df_mul(q.a, w_df(wp.normal[0])), mkl_df_mul(q.b, w_df(wp.normal[1]))),
                mkl_df_mul(q.z, w_df(wp.normal[2])),
            );
            let nz = abs(w_scalar(d));
            let weight = mkl_div_rn(mkl_mul(c, a), gamma);
            let ag = mkl_div_rn(a, gamma);
            let e = mkl_mul(mkl_mul(c, ag), ag);
            var third = mkl_qnan();
            if (nz > 1e-9) {
                third = mkl_div_rn(mkl_mul(e, gamma), nz);
            }
            m = vec3<f32>(weight, e, third);
            let ph = w_cis(r_opd[j]);
            z = vec2<f32>(mkl_mul(weight, ph.x), mkl_mul(weight, ph.y));
        }
        r_mom[j] = m.x;
        r_mom[count + j] = m.y;
        r_mom[2u * count + j] = m.z;
    }
    r_field[k] = z;
}

// ---------------------------------------------------------------------------
// Intensity (spec::intensity). Storage: params, field, sums, out.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> i_field: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> i_sums: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> i_out: array<f32>;

@compute @workgroup_size(256)
fn mkl_wave_intensity(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let k = w_linear(lid, wg, nwg);
    let m = wp.size;
    if (k >= m * m) {
        return;
    }
    var j = k;
    if (wp.shifted != 0u) {
        let side = i32(wp.field_side);
        let ix = ((i32(k % m) - i32(m / 2u)) % side + side) % side;
        let iy = ((i32(k / m) - i32(m / 2u)) % side + side) % side;
        j = u32(iy) * wp.field_side + u32(ix);
    }
    let z = i_field[j];
    let l = mkl_df_div_rn(
        mkl_df_mul(w_lift(f32(wp.n)), w_df(wp.wavelength)),
        mkl_df_mul_f32(w_df(wp.sin_max), 2.0),
    );
    var div: vec2<f32>;
    if (wp.unit_energy != 0u) {
        div = mkl_df_div_rn(mkl_df_mul(mkl_df_mul(l, l), i_sums[1]), w_lift(wp.projection));
    } else {
        div = mkl_df_mul(i_sums[0], i_sums[0]);
    }
    let power = mkl_add(mkl_mul(z.x, z.x), mkl_mul(z.y, z.y));
    i_out[k] = w_scalar(mkl_df_div_rn(w_lift(power), div));
}

// ---------------------------------------------------------------------------
// Line spread: rows (x first) and columns of an n² image, double-float sums
// in sample order times the spacing. Storage: params, image, out.
// ---------------------------------------------------------------------------

@group(0) @binding(3) var<storage, read> l_image: array<f32>;
@group(0) @binding(4) var<storage, read_write> l_out: array<f32>;

@compute @workgroup_size(64)
fn mkl_wave_lsf(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    let k = id.x;
    let n = wp.n;
    if (k >= 2u * n) {
        return;
    }
    var s = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < n; j++) {
        var at = (k - n) * n + j;
        if (k < n) {
            at = j * n + k;
        }
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

@group(0) @binding(2) var<storage, read> sp: MklWaveSpectralParams;
@group(0) @binding(3) var<storage, read> s_in: array<f32>;
@group(0) @binding(4) var<storage, read> s_freq: array<f32>;
@group(0) @binding(5) var<storage, read_write> s_out: array<f32>;

@compute @workgroup_size(256)
fn mkl_wave_spectral(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let k = w_linear(lid, wg, nwg);
    if (k >= sp.count) {
        return;
    }
    let complex = sp.complex != 0u;
    var width = 1u;
    if (complex) {
        width = 2u;
    }
    let stride = sp.count * width;
    var s0 = vec2<f32>(0.0, 0.0);
    var s1 = vec2<f32>(0.0, 0.0);
    for (var j = 0u; j < sp.inputs; j++) {
        var z = vec2<f32>(s_in[j * stride + k * width], 0.0);
        if (complex) {
            z.y = s_in[j * stride + k * width + 1u];
        }
        if (sp.phase != 0u) {
            let o0 = w_df(sp.offsets[j][0]);
            let o1 = w_df(sp.offsets[j][1]);
            let ph = mkl_df_neg(mkl_df_add(
                mkl_df_mul_f32(o0, s_freq[2u * k]),
                mkl_df_mul_f32(o1, s_freq[2u * k + 1u]),
            ));
            let q = w_cis(ph);
            z = vec2<f32>(
                mkl_sub(mkl_mul(z.x, q.x), mkl_mul(z.y, q.y)),
                mkl_add(mkl_mul(z.x, q.y), mkl_mul(z.y, q.x)),
            );
        }
        let w = w_df(sp.weights[j]);
        s0 = mkl_df_add(s0, mkl_df_mul_f32(w, z.x));
        if (complex) {
            s1 = mkl_df_add(s1, mkl_df_mul_f32(w, z.y));
        }
    }
    s_out[k * width] = w_scalar(s0);
    if (complex) {
        s_out[k * width + 1u] = w_scalar(s1);
    }
}

// ---------------------------------------------------------------------------
// Image integrals (image_integrals::cpu), from MklWaveIntegralParams.
// ---------------------------------------------------------------------------

@group(0) @binding(2) var<storage, read> ip: MklWaveIntegralParams;

// Metrics: per block of 256 pixels (sum, Σv·x, Σv·y, maximum, its index),
// then the blocks in order. Storage: params, image, blocks [blocks, 5]
// double-float, out [5] double-float.
@group(0) @binding(3) var<storage, read> m_image: array<f32>;
@group(0) @binding(4) var<storage, read_write> m_blocks: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> m_out: array<vec2<f32>>;

@compute @workgroup_size(256)
fn mkl_wave_metrics(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let b = w_linear(lid, wg, nwg);
    let n = ip.n;
    let count = n * n;
    if (b * W_BLOCK >= count) {
        return;
    }
    var row = array<vec2<f32>, 5>();
    for (var k = b * W_BLOCK; k < min((b + 1u) * W_BLOCK, count); k++) {
        let v = m_image[k];
        row[0] = mkl_df_add_f32(row[0], v);
        row[1] = mkl_df_add(row[1], mkl_df_mul_f32(w_lift(v), f32(k % n)));
        row[2] = mkl_df_add(row[2], mkl_df_mul_f32(w_lift(v), f32(k / n)));
        if (k % W_BLOCK == 0u || v > row[3].x) {
            row[3] = w_lift(v);
            row[4] = w_lift(f32(k));
        }
    }
    for (var j = 0u; j < 5u; j++) {
        m_blocks[5u * b + j] = row[j];
    }
}

@compute @workgroup_size(64)
fn mkl_wave_metrics_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    if (id.x != 0u) {
        return;
    }
    let blocks = (ip.n * ip.n + W_BLOCK - 1u) / W_BLOCK;
    var sum = array<vec2<f32>, 5>();
    for (var j = 0u; j < 5u; j++) {
        sum[j] = m_blocks[j];
    }
    for (var b = 1u; b < blocks; b++) {
        for (var j = 0u; j < 3u; j++) {
            sum[j] = mkl_df_add(sum[j], m_blocks[5u * b + j]);
        }
        if (m_blocks[5u * b + 3u].x > sum[3].x) {
            sum[3] = m_blocks[5u * b + 3u];
            sum[4] = m_blocks[5u * b + 4u];
        }
    }
    for (var j = 0u; j < 5u; j++) {
        m_out[j] = sum[j];
    }
}

// Complex: the image (or |z|² of a complex one) as complex values.
// Storage: params, input, out.
@group(0) @binding(3) var<storage, read> x_in: array<f32>;
@group(0) @binding(4) var<storage, read_write> x_out: array<vec2<f32>>;

@compute @workgroup_size(256)
fn mkl_wave_image_complex(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let k = w_linear(lid, wg, nwg);
    if (k >= ip.n * ip.n) {
        return;
    }
    if (ip.kind == 4u) {
        let re = x_in[2u * k];
        let im = x_in[2u * k + 1u];
        x_out[k] = vec2<f32>(mkl_add(mkl_mul(re, re), mkl_mul(im, im)), 0.0);
    } else {
        x_out[k] = vec2<f32>(x_in[k], 0.0);
    }
}

// image_integrals::j1: the specification's polynomials in its nesting; the
// cosine through w_cos_sin.
fn w_j1(x: f32) -> f32 {
    let ax = abs(x);
    if (ax <= 3.0) {
        let x3 = mkl_div_rn(x, 3.0);
        let t = mkl_mul(x3, x3);
        var p = mkl_add(-0.00031761, mkl_mul(t, 0.00001109));
        p = mkl_add(0.00443319, mkl_mul(t, p));
        p = mkl_add(-0.03954289, mkl_mul(t, p));
        p = mkl_add(0.21093573, mkl_mul(t, p));
        p = mkl_add(-0.5624998, mkl_mul(t, p));
        p = mkl_add(0.5, mkl_mul(t, p));
        return mkl_mul(x, p);
    }
    let y = mkl_div_rn(3.0, ax);
    var f = mkl_sub(0.00113653, mkl_mul(y, 0.00020033));
    f = mkl_add(-0.00249511, mkl_mul(y, f));
    f = mkl_add(0.00017105, mkl_mul(y, f));
    f = mkl_add(0.01659667, mkl_mul(y, f));
    f = mkl_add(0.00000156, mkl_mul(y, f));
    f = mkl_add(0.7978846, mkl_mul(y, f));
    var h = mkl_sub(0.00079824, mkl_mul(y, 0.00029166));
    h = mkl_add(0.00074348, mkl_mul(y, h));
    h = mkl_add(-0.00637879, mkl_mul(y, h));
    h = mkl_add(0.00005650, mkl_mul(y, h));
    h = mkl_add(0.12499612, mkl_mul(y, h));
    let th = mkl_add(mkl_sub(ax, 2.3561945), mkl_mul(y, h));
    let r = mkl_div_rn(mkl_mul(f, w_cos_sin(th).x), mkl_sqrt_rn(ax));
    if (x < 0.0) {
        return -r;
    }
    return r;
}

// image_integrals::term of pixel k for query (q0, q1) and value v.
fn w_integral_term(k: u32, q0: vec2<f32>, q1: vec2<f32>, v: vec2<f32>) -> vec2<f32> {
    let n = ip.n;
    let center0 = w_df(ip.center[0]);
    let center1 = w_df(ip.center[1]);
    let spacing = w_df(ip.spacing);
    if (ip.kind == 0u) {
        let x = w_lift(f32(k % n));
        let y = w_lift(f32(k / n));
        let ph = mkl_df_mul(
            mkl_df_neg(mkl_df_add(
                mkl_df_mul(mkl_df_sub(x, center0), q0),
                mkl_df_mul(mkl_df_sub(y, center1), q1),
            )),
            spacing,
        );
        let z = w_cis(ph);
        return vec2<f32>(mkl_mul(v.x, z.x), mkl_mul(v.x, z.y));
    }
    var ix = i32(k % n);
    if (k % n > (n - 1u) / 2u) {
        ix = ix - i32(n);
    }
    var iy = i32(k / n);
    if (k / n > (n - 1u) / 2u) {
        iy = iy - i32(n);
    }
    let nn = w_lift(f32(n));
    let inv = mkl_df_div_rn(w_lift(1.0), mkl_df_mul(nn, spacing));
    let fx = mkl_df_mul_f32(inv, f32(ix));
    let fy = mkl_df_mul_f32(inv, f32(iy));
    let ph = mkl_df_div_rn(
        mkl_df_add(mkl_df_mul_f32(center0, f32(ix)), mkl_df_mul_f32(center1, f32(iy))),
        nn,
    );
    let z = w_cis(ph);
    let real = mkl_sub(mkl_mul(v.x, z.x), mkl_mul(v.y, z.y));
    let radius = w_scalar(q0);
    let pi = 3.14159265358979323846;
    let tau = 6.28318530717958647692;
    var kernel = 0.0;
    if (radius <= 0.0) {
        kernel = 0.0;
    } else if (ip.kind == 1u) {
        let rho = w_scalar(mkl_df_sqrt_rn(mkl_df_add(mkl_df_mul(fx, fx), mkl_df_mul(fy, fy))));
        if (mkl_mul(rho, radius) < 1e-12) {
            kernel = mkl_mul(mkl_mul(pi, radius), radius);
        } else {
            kernel = mkl_div_rn(mkl_mul(radius, w_j1(mkl_mul(mkl_mul(tau, rho), radius))), rho);
        }
    } else {
        let f = vec2<f32>(w_scalar(fx), w_scalar(fy));
        var sinc = vec2<f32>(0.0, 0.0);
        for (var i = 0; i < 2; i++) {
            if (f[i] == 0.0) {
                sinc[i] = mkl_mul(2.0, radius);
            } else {
                let s = w_cos_sin(mkl_mul(mkl_mul(tau, f[i]), radius)).y;
                sinc[i] = mkl_div_rn(s, mkl_mul(pi, f[i]));
            }
        }
        kernel = mkl_mul(sinc.x, sinc.y);
    }
    return vec2<f32>(mkl_mul(real, kernel), 0.0);
}

// Query: thread t sums block t % blocks of query first + t / blocks.
// Storage: params, input, queries [queries, 2] double-float, block sums.
@group(0) @binding(3) var<storage, read> q_in: array<f32>;
@group(0) @binding(4) var<storage, read> q_queries: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> q_acc: array<vec2<f32>>;

@compute @workgroup_size(256)
fn mkl_wave_image_query(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    mkl_enter();
    let t = w_linear(lid, wg, nwg);
    let count = ip.n * ip.n;
    let blocks = (count + W_BLOCK - 1u) / W_BLOCK;
    if (t >= ip.count * blocks) {
        return;
    }
    let q = ip.first + t / blocks;
    let q0 = q_queries[2u * q];
    let q1 = q_queries[2u * q + 1u];
    let first = (t % blocks) * W_BLOCK;
    var s0 = vec2<f32>(0.0, 0.0);
    var s1 = vec2<f32>(0.0, 0.0);
    for (var k = first; k < min(first + W_BLOCK, count); k++) {
        var v = vec2<f32>(q_in[k], 0.0);
        if (ip.kind != 0u) {
            v = vec2<f32>(q_in[2u * k], q_in[2u * k + 1u]);
        }
        let z = w_integral_term(k, q0, q1, v);
        s0 = mkl_df_add_f32(s0, z.x);
        s1 = mkl_df_add_f32(s1, z.y);
    }
    q_acc[2u * t] = s0;
    q_acc[2u * t + 1u] = s1;
}

// Storage: params, queries, block sums, out (complex, all queries).
@group(0) @binding(3) var<storage, read> qf_queries: array<vec2<f32>>;
@group(0) @binding(4) var<storage, read> qf_acc: array<vec2<f32>>;
@group(0) @binding(5) var<storage, read_write> qf_out: array<vec2<f32>>;

@compute @workgroup_size(64)
fn mkl_wave_image_query_finish(@builtin(global_invocation_id) id: vec3<u32>) {
    mkl_enter();
    if (id.x >= ip.count) {
        return;
    }
    let q = ip.first + id.x;
    let blocks = (ip.n * ip.n + W_BLOCK - 1u) / W_BLOCK;
    var s0 = vec2<f32>(0.0, 0.0);
    var s1 = vec2<f32>(0.0, 0.0);
    for (var b = 0u; b < blocks; b++) {
        s0 = mkl_df_add(s0, qf_acc[2u * (id.x * blocks + b)]);
        s1 = mkl_df_add(s1, qf_acc[2u * (id.x * blocks + b) + 1u]);
    }
    if (ip.kind == 0u && qf_queries[2u * q].x == 0.0 && qf_queries[2u * q + 1u].x == 0.0) {
        qf_out[q] = vec2<f32>(1.0, 0.0);
        return;
    }
    let norm = w_df(ip.normalization);
    qf_out[q] = vec2<f32>(w_scalar(mkl_df_div_rn(s0, norm)), w_scalar(mkl_df_div_rn(s1, norm)));
}
