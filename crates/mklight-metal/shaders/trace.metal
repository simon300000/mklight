// Sequential / path ray-trace kernel (mkl_trace).
//
// The loop below is the legacy twin of trace_kernel::trace_resident and of
// mkl_trace_raw (trace_shared.h); change all three together. It is kept
// separate so that the resident state, normal and record hooks cannot shift
// the compiler's whole-kernel decisions for mkl_trace: its IR is identical
// to the single-file kernel it replaced.
//
// trace_overlays.metal compiles this file again as mkl_trace_overlays, with
// the sag-overlay code, in its own translation unit: that code costs
// mkl_trace nothing (a few percent when it shares a unit with it).
#include "trace_shared.h"
#ifndef MKLIGHT_TRACE_BODY
#define MKLIGHT_TRACE_BODY
// One thread per ray. Planes are `stride` long and bound at the launch's
// first ray (see mklight_gpu_types.h). The host has validated the program,
// the systems and the wavelengths (PreparedTrace::new). OVERLAYS: whether
// the systems have sag overlays.
template<bool OVERLAYS> static inline void mkl_trace_body(constant MklPackedHeader* headers,constant MklPackedSurface* surfaces,constant uint* steps,
                                     constant MklTraceParams& prm,device const float* in_f,device const uint* in_u,
                                     device float* out_f,device uint* out_u,device float* rec_f,device uint* rec_u,
                                     constant float* coefficients,uint i) {
    if (i >= prm.num_rays) return;
    const uint S = prm.stride;
    const uint w = in_u[mkl_plane(MKL_TRACE_INU_WAVELENGTH, S, i)];
    constant MklPackedHeader& h = headers[in_u[mkl_plane(MKL_TRACE_INU_SYSTEM, S, i)]];
    constant MklPackedSurface* s = surfaces + h.surface_offset;
    const bool clip = (prm.flags & MKL_TRACE_CLIP) != 0u;
    const bool radiometry = (prm.flags & MKL_TRACE_RADIOMETRY) != 0u;
    const bool record = (prm.flags & MKL_TRACE_RECORD) != 0u;
    const bool at_infinity = (h.flags & MKL_SYS_OBJECT_AT_INFINITY) != 0u;
    const uint num_steps = prm.num_steps;

    MklDv pos, dir;
    pos.v[0] = float2(in_f[mkl_plane(MKL_TRACE_IN_X, S, i)], in_f[mkl_plane(MKL_TRACE_IN_X_LO, S, i)]);
    pos.v[1] = float2(in_f[mkl_plane(MKL_TRACE_IN_Y, S, i)], in_f[mkl_plane(MKL_TRACE_IN_Y_LO, S, i)]);
    pos.v[2] = float2(in_f[mkl_plane(MKL_TRACE_IN_Z, S, i)], in_f[mkl_plane(MKL_TRACE_IN_Z_LO, S, i)]);
    dir.v[0] = float2(in_f[mkl_plane(MKL_TRACE_IN_L, S, i)], in_f[mkl_plane(MKL_TRACE_IN_L_LO, S, i)]);
    dir.v[1] = float2(in_f[mkl_plane(MKL_TRACE_IN_M, S, i)], in_f[mkl_plane(MKL_TRACE_IN_M_LO, S, i)]);
    dir.v[2] = float2(in_f[mkl_plane(MKL_TRACE_IN_N, S, i)], in_f[mkl_plane(MKL_TRACE_IN_N_LO, S, i)]);
    float2 opl = float2(in_f[mkl_plane(MKL_TRACE_IN_OPL_HI, S, i)], in_f[mkl_plane(MKL_TRACE_IN_OPL_LO, S, i)]);
    const float w_in = in_f[mkl_plane(MKL_TRACE_IN_WEIGHT, S, i)];
    float tw = 1.0f;
    float absorb = 0.0f;
    const uint start = prm.start_surface;
    float2 n_cur = mkl_index(s[start], w);
    uint last = start;
    uint status = MKL_RAY_OK;
    uint done = 0u;

    if (record) mkl_record(rec_f, rec_u, 0u, S, i, mkl_report(pos, dir, opl, n_cur, w_in, MKL_RAY_OK));
    if (mkl_finite3(mkl_hi3(pos)) && mkl_finite3(mkl_hi3(dir)) && mkl_finite3(mkl_lo3(pos))
        && mkl_finite3(mkl_lo3(dir))) {
        dir = mkl_renorm(dir);
    } else {
        status = MKL_RAY_NONFINITE | (start << MKL_RAY_SURFACE_SHIFT);
    }

    for (uint k = 0u; status == MKL_RAY_OK && k < num_steps; ++k) {
        const uint word = steps[k];
        const uint si = word & MKL_STEP_SURFACE_MASK;
        const uint mode = (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK;
        const bool back = (word & MKL_STEP_BACKWARD) != 0u;
        const uint gap = back ? si : si - 1u;
        const float2 n1 = mkl_index(s[gap], w);
        const float alpha = (gap == 0u && at_infinity) ? 0.0f : s[gap].alpha_after[w];
        const float2 n_other = back ? (si >= 1u ? mkl_index(s[si - 1u], w) : n1) : mkl_index(s[si], w);
        // 1. Transfer into the surface's frame.
        if (back) {
            mkl_xf_backward(s[si], pos, dir);
        } else {
            mkl_xf_forward(s[si - 1u], pos, dir);
        }
        constant MklPackedSurface& surf = s[si];
        last = si;
        n_cur = n1;
        const uint fail_s = si << MKL_RAY_SURFACE_SHIFT;
        // 2. Intersect; optical path and absorption of the segment.
        MklHit hit;
        uint code = mkl_intersect<OVERLAYS>(surf, coefficients, pos, dir, prm, hit);
        if (code != MKL_RAY_OK) {
            status = code | fail_s;
            break;
        }
        opl = mkl_df_add(opl, mkl_df_mul(hit.ta, n1));
        opl = mkl_df_add(opl, mkl_df_mul(hit.tb, n1));
        if (alpha != 0.0f) absorb = fma(alpha, hit.ta.x + hit.tb.x, absorb);
        pos = hit.x;
        if (!mkl_finite3(mkl_hi3(pos))) {
            status = MKL_RAY_NONFINITE | fail_s;
            break;
        }
        // 3. Clip.
        const bool is_cb = surf.shape == MKL_SHAPE_COORD_BREAK;
        if (clip && !is_cb && !mkl_passes_aperture(surf, pos.v[0], pos.v[1])) {
            status = MKL_RAY_VIGNETTED | fail_s;
            break;
        }
        // 4. Interact (0 none, 1 refract, 2 reflect, 3 lens, 4 reflecting lens).
        // A DOE's selected nonzero order replaces 1 and 2 (trace_kernel's
        // Act::Diffract), tested first: faster than a separate act code.
        const bool is_image = (surf.flags & MKL_SURF_IMAGE) != 0u;
        const bool paraxial = surf.shape == MKL_SHAPE_PARAXIAL;
        const bool mirror = surf.interaction == MKL_INTERACT_REFLECT;
        const bool dummy = surf.interaction == MKL_INTERACT_DUMMY;
        uint act = 0u;
        if (!is_cb) {
            if (mode == MKL_STEP_SEQUENTIAL) {
                act = (is_image || dummy) ? 0u : paraxial ? (mirror ? 4u : 3u) : mirror ? 2u : 1u;
            } else if (mode == MKL_STEP_TRANSMIT) {
                act = (is_image || dummy) ? 0u : paraxial ? 3u : 1u;
            } else {
                act = (paraxial && mirror) ? 4u : 2u;
            }
        }
        const uint coating = surf.coating;
        const float3 dfv = mkl_hi3(dir);
        if ((act == 1u || act == 2u) && surf.shape == MKL_SHAPE_DIFFRACTIVE && !dummy && !is_image &&
            mkl_surface_data(surf, coefficients)[2 * MKL_COEFF_TRACE_ORDER] != 0.0f) {
            // m lambda/(2 pi) scales grad Phi to momentum and Phi to optical path.
            const MklDf scale = mkl_load<MklDf>(mkl_surface_data(surf, coefficients), MKL_COEFF_PREFIX + w);
            const MklJet<MklDf> v = mkl_phase(surf, coefficients, MklDf{pos.v[0]}, MklDf{pos.v[1]});
            const bool refl = act == 2u;
            const float2 n2 = refl ? n1 : n_other;
            MklDv d2;
            if (!mkl_diffract(dir, hit.g, n1, n2, v.x * scale, v.y * scale, refl, d2)) {
                status = MKL_RAY_EVANESCENT | fail_s;
                break;
            }
            dir = d2;
            n_cur = n2;
            opl = mkl_df_add(opl, (scale * v.v).v);
            if (radiometry) {
                float cos_i;
                mkl_reflect(dfv, hit.normal, cos_i);
                const float2 rt = mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i);
                tw = tw * (refl ? rt.x : rt.y);
            }
        } else if (act == 1u) {
            float cos_i, gamma;
            if (!mkl_refract(dfv, hit.normal, n1.x, n_other.x, cos_i, gamma)) {
                status = MKL_RAY_TIR | fail_s;
                break;
            }
            float2 mu = mkl_df_div(n1, n_other);
            float b = gamma / mkl_norm3(mkl_hi3(hit.g));
            MklDv d2;
            for (int c = 0; c < 3; ++c) {
                d2.v[c] = mkl_df_add(mkl_df_mul(dir.v[c], mu), mkl_df_mul_f32(hit.g.v[c], b));
            }
            dir = mkl_renorm_along(mkl_renorm_along(d2, hit.g), hit.g);
            n_cur = n_other;
            if (radiometry) tw = tw * mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i).y;
        } else if (act == 2u) {
            float cos_i;
            mkl_reflect(dfv, hit.normal, cos_i);
            float3 gh = mkl_hi3(hit.g);
            float b = -2.0f * mkl_dot3(dfv, gh) / mkl_dot3(gh, gh);
            MklDv d2;
            for (int c = 0; c < 3; ++c) d2.v[c] = mkl_df_add(dir.v[c], mkl_df_mul_f32(hit.g.v[c], b));
            dir = mkl_renorm_along(d2, hit.g);
            if (radiometry) tw = tw * mkl_surface_rt(surf, mirror, n1.x, n_other.x, cos_i).x;
        } else if (act >= 3u) {
            const bool refl = act == 4u;
            MklDv kk = dir;
            if (refl) kk.v[2] = mkl_df_neg(kk.v[2]);
            const float2 n2 = refl ? n1 : n_other;
            const float2 verg = back ? float2(0.0f) : float2(surf.vergence[w], surf.vergence_lo[w]);
            const float2 phi = float2(surf.paraxial_power, surf.paraxial_power_lo);
            MklDv d2;
            float2 delta;
            if (!mkl_ideal_lens(pos.v[0], pos.v[1], kk, phi, n1, n2, verg, d2, delta)) {
                status = MKL_RAY_DEGENERATE | fail_s;
                break;
            }
            dir = d2;
            n_cur = n2;
            opl = mkl_df_add(opl, delta);
            if (radiometry && coating == MKL_COATING_IDEAL) {
                tw = tw * (refl ? surf.coating_r : surf.coating_t);
            }
        }
        done = k + 1u;
        if (record) {
            float wnow = radiometry ? w_in * tw * mkl_exp(-absorb) : w_in;
            mkl_record(rec_f, rec_u, done, S, i, mkl_report(pos, dir, opl, n_cur, wnow, MKL_RAY_OK));
        }
    }

    const float transmittance = mkl_exp(-absorb);
    const float weight = radiometry ? w_in * tw * transmittance : w_in;
    MklState st = mkl_report(pos, dir, opl, n_cur, weight, status);
    if (record) {
        for (uint slot = done + 1u; slot <= num_steps; ++slot) mkl_record(rec_f, rec_u, slot, S, i, st);
    }
    out_f[mkl_plane(MKL_TRACE_OUT_X, S, i)] = st.pos.x;
    out_f[mkl_plane(MKL_TRACE_OUT_Y, S, i)] = st.pos.y;
    out_f[mkl_plane(MKL_TRACE_OUT_Z, S, i)] = st.pos.z;
    out_f[mkl_plane(MKL_TRACE_OUT_L, S, i)] = st.dir.x;
    out_f[mkl_plane(MKL_TRACE_OUT_M, S, i)] = st.dir.y;
    out_f[mkl_plane(MKL_TRACE_OUT_N, S, i)] = st.dir.z;
    out_f[mkl_plane(MKL_TRACE_OUT_OPL_HI, S, i)] = st.opl.x;
    out_f[mkl_plane(MKL_TRACE_OUT_OPL_LO, S, i)] = st.opl.y;
    out_f[mkl_plane(MKL_TRACE_OUT_WEIGHT, S, i)] = st.weight;
    out_f[mkl_plane(MKL_TRACE_OUT_TRANSMITTANCE, S, i)] = transmittance;
    out_u[mkl_plane(MKL_TRACE_OUTU_STATUS, S, i)] = st.status;
    out_u[mkl_plane(MKL_TRACE_OUTU_LAST_SURFACE, S, i)] = last;
    out_u[mkl_plane(MKL_TRACE_OUTU_STEPS, S, i)] = done;
}
#endif
#ifndef MKL_TRACE_OVERLAYS
#define MKL_TRACE_OVERLAYS false
#define MKL_TRACE_KERNEL mkl_trace
#endif
kernel void MKL_TRACE_KERNEL(constant MklPackedHeader* headers [[buffer(0)]],
                             constant MklPackedSurface* surfaces [[buffer(1)]],
                             constant uint* steps [[buffer(2)]],
                             constant MklTraceParams& prm [[buffer(3)]],
                             device const float* in_f [[buffer(4)]],
                             device const uint* in_u [[buffer(5)]],
                             device float* out_f [[buffer(6)]],
                             device uint* out_u [[buffer(7)]],
                             device float* rec_f [[buffer(8)]],
                             device uint* rec_u [[buffer(9)]],
                             constant float* coefficients [[buffer(10)]],
                             uint i [[thread_position_in_grid]]) {
    mkl_trace_body<MKL_TRACE_OVERLAYS>(headers,surfaces,steps,prm,in_f,in_u,out_f,out_u,rec_f,rec_u,coefficients,i);
}
#undef MKL_TRACE_OVERLAYS
#undef MKL_TRACE_KERNEL
