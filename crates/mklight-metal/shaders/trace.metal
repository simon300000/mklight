#include "trace_shared.h"
#ifndef MKLIGHT_TRACE_BODY
#define MKLIGHT_TRACE_BODY
// One thread per ray. Planes are `stride` long and bound at the launch's
// first ray (see mklight_gpu_types.h). The host has validated the program,
// the systems and the wavelengths (PreparedTrace::new). OVERLAYS: whether
// the systems have sag overlays.
template<bool OVERLAYS> static inline void mkl_trace_body(constant MklPackedHeader* headers,constant MklPackedSurface* surfaces,constant uint* steps,
                                     constant MklTraceParams& params,device const float* in_f,device const uint* in_u,
                                     device float* out_f,device uint* out_u,device float* rec_f,device uint* rec_u,
                                     constant float* coefficients,uint i) {
    const MklTraceParams prm = params;
    if (i >= prm.num_rays) return;
    const uint S = prm.stride;
    const uint w = in_u[mkl_plane(MKL_TRACE_INU_WAVELENGTH, S, i)];
    constant MklPackedHeader& h = headers[in_u[mkl_plane(MKL_TRACE_INU_SYSTEM, S, i)]];
    constant MklPackedSurface* s = surfaces + h.surface_offset;
    MklDv pos, dir;
    pos.v[0] = float2(in_f[mkl_plane(MKL_TRACE_IN_X, S, i)], in_f[mkl_plane(MKL_TRACE_IN_X_LO, S, i)]);
    pos.v[1] = float2(in_f[mkl_plane(MKL_TRACE_IN_Y, S, i)], in_f[mkl_plane(MKL_TRACE_IN_Y_LO, S, i)]);
    pos.v[2] = float2(in_f[mkl_plane(MKL_TRACE_IN_Z, S, i)], in_f[mkl_plane(MKL_TRACE_IN_Z_LO, S, i)]);
    dir.v[0] = float2(in_f[mkl_plane(MKL_TRACE_IN_L, S, i)], in_f[mkl_plane(MKL_TRACE_IN_L_LO, S, i)]);
    dir.v[1] = float2(in_f[mkl_plane(MKL_TRACE_IN_M, S, i)], in_f[mkl_plane(MKL_TRACE_IN_M_LO, S, i)]);
    dir.v[2] = float2(in_f[mkl_plane(MKL_TRACE_IN_N, S, i)], in_f[mkl_plane(MKL_TRACE_IN_N_LO, S, i)]);
    float2 opl = float2(in_f[mkl_plane(MKL_TRACE_IN_OPL_HI, S, i)], in_f[mkl_plane(MKL_TRACE_IN_OPL_LO, S, i)]);
    const float w_in = in_f[mkl_plane(MKL_TRACE_IN_WEIGHT, S, i)];
    MklRawState input = {};
    input.pos = pos; input.dir = dir; input.opl = opl; input.weight = w_in;
    MklLegacyRecords rec = {rec_f,rec_u,S,i,(prm.flags & MKL_TRACE_RECORD) != 0u};
    MklRawState raw = mkl_trace_raw<OVERLAYS>(h,s,coefficients,steps,prm,w,input,rec);
    MklState st = mkl_report(raw.pos,raw.dir,raw.opl,raw.index,raw.weight,raw.status);
    out_f[mkl_plane(MKL_TRACE_OUT_X, S, i)] = st.pos.x;
    out_f[mkl_plane(MKL_TRACE_OUT_Y, S, i)] = st.pos.y;
    out_f[mkl_plane(MKL_TRACE_OUT_Z, S, i)] = st.pos.z;
    out_f[mkl_plane(MKL_TRACE_OUT_L, S, i)] = st.dir.x;
    out_f[mkl_plane(MKL_TRACE_OUT_M, S, i)] = st.dir.y;
    out_f[mkl_plane(MKL_TRACE_OUT_N, S, i)] = st.dir.z;
    out_f[mkl_plane(MKL_TRACE_OUT_OPL_HI, S, i)] = st.opl.x;
    out_f[mkl_plane(MKL_TRACE_OUT_OPL_LO, S, i)] = st.opl.y;
    out_f[mkl_plane(MKL_TRACE_OUT_WEIGHT, S, i)] = st.weight;
    out_f[mkl_plane(MKL_TRACE_OUT_TRANSMITTANCE, S, i)] = raw.transmittance;
    out_u[mkl_plane(MKL_TRACE_OUTU_STATUS, S, i)] = st.status;
    out_u[mkl_plane(MKL_TRACE_OUTU_LAST_SURFACE, S, i)] = raw.last;
    out_u[mkl_plane(MKL_TRACE_OUTU_STEPS, S, i)] = raw.done;
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
