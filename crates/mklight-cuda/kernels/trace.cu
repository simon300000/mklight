// The sequential / path ray-trace kernel (kernels/trace/*.wgsl's interface,
// the Metal trace's arithmetic): mkl_trace, mkl_trace_resident and
// mkl_trace_resident_records.
//
// kernels.rs compiles this file twice, prefixed with `#define MKL_OVERLAYS
// false` (module "trace") or `true` (module "trace_overlays"), like the WGSL
// modules of the same names.
//
// The trace itself is mkl_trace_raw of trace_shared.h (the twin of
// trace_kernel::trace_resident and of the WGSL mkl_trace_core); the entry
// points below only adapt the WGSL bindings and planes to it:
//
//   binding 1 MklWgslTraceParams, 2 headers, 3 surfaces, 4 coefficient pool,
//   5 path program, 6 input planes, 7 output planes, 8 records, every plane
//   `stride` words long and starting at the launch's first ray.
//   mkl_trace           in:  MKL_TRACE_IN_FLOATS float planes, then
//                            MKL_TRACE_IN_UINTS planes (wavelength, system)
//                       out: MKL_TRACE_OUT_FLOATS float planes, then
//                            MKL_TRACE_OUT_UINTS planes
//                       rec: [slot][ray][MKL_TRACE_RECORD_FLOATS] floats,
//                            then [slot][ray] status words
//   mkl_trace_resident  in:  15 float planes (aiming ray layout), system,
//   (+ _records)             wavelength, 2 id planes, 4 report planes
//                       out: the same 23 planes, then 3 status planes
//                            (status, last surface, steps)
//                       rec: [slot][ray][MKL_AIM_RECORD_FLOATS] hi, then lo
#include "mkl_cuda.h"

// Compiled per entry point (MKL_ONE_ENTRY, kernels::unit): each inlines the
// whole trace.
#include "trace_bindings.h"

#ifndef MKL_OVERLAYS
#error "trace.cu is compiled with MKL_OVERLAYS defined (kernels.rs)"
#endif

// mkl_trace's records: the rounded state (mkl_report) and its status.
struct MklTraceRecords {
    float* f;
    uint* status;
    uint n, i;
    void write(uint slot, MklRawState s) {
        const MklState st = mkl_report(s.pos, s.dir, s.opl, s.index, s.weight, s.status);
        mkl_record(f, status, slot, n, i, st);
    }
};

// The resident trace's paired hi / lo records (aiming.metal's MklAimRecords,
// always enabled).
struct MklResidentRecords {
    float* hi;
    float* lo;
    uint n, i;
    void write(uint slot, MklRawState s) {
        const ulong a = mkl_plane(slot, n, i) * ulong(MKL_AIM_RECORD_FLOATS);
        for (uint k = 0; k < 3; ++k) {
            hi[a + k] = s.pos.v[k].x;
            lo[a + k] = s.pos.v[k].y;
            hi[a + 3 + k] = s.dir.v[k].x;
            lo[a + 3 + k] = s.dir.v[k].y;
            hi[a + 8 + k] = s.normal.v[k].x;
            lo[a + 8 + k] = s.normal.v[k].y;
        }
        hi[a + 6] = s.opl.x;
        lo[a + 6] = s.opl.y;
        hi[a + 7] = s.weight;
        lo[a + 7] = 0;
        hi[a + 11] = s.index.x;
        lo[a + 11] = s.index.y;
        hi[a + 12] = as_type<float>(s.status);
        lo[a + 12] = 0;
    }
};

#if !defined(MKL_ONE_ENTRY) || defined(MKL_ENTRY_mkl_trace)
// One thread per ray. The host has validated the program, the systems and
// the wavelengths (PreparedTrace::new).
MKL_KERNEL mkl_trace(MKL_TRACE_ARGS) {
    const uint i = mkl_gid_x();
    const MklTraceParams& prm = u->trace;
    if (i >= prm.num_rays) return;
    const uint S = prm.stride;
    const float* in_f = reinterpret_cast<const float*>(in);
    const uint* in_u = in + mkl_plane(MKL_TRACE_IN_FLOATS, S, 0);
    float* out_f = reinterpret_cast<float*>(out);
    uint* out_u = out + mkl_plane(MKL_TRACE_OUT_FLOATS, S, 0);
    const uint w = in_u[mkl_plane(MKL_TRACE_INU_WAVELENGTH, S, i)];
    const MklPackedHeader& h = headers[in_u[mkl_plane(MKL_TRACE_INU_SYSTEM, S, i)]];
    const MklPackedSurface* s = surfaces + h.surface_offset;

    MklRawState ray = {};
    ray.pos.v[0] = float2(in_f[mkl_plane(MKL_TRACE_IN_X, S, i)], in_f[mkl_plane(MKL_TRACE_IN_X_LO, S, i)]);
    ray.pos.v[1] = float2(in_f[mkl_plane(MKL_TRACE_IN_Y, S, i)], in_f[mkl_plane(MKL_TRACE_IN_Y_LO, S, i)]);
    ray.pos.v[2] = float2(in_f[mkl_plane(MKL_TRACE_IN_Z, S, i)], in_f[mkl_plane(MKL_TRACE_IN_Z_LO, S, i)]);
    ray.dir.v[0] = float2(in_f[mkl_plane(MKL_TRACE_IN_L, S, i)], in_f[mkl_plane(MKL_TRACE_IN_L_LO, S, i)]);
    ray.dir.v[1] = float2(in_f[mkl_plane(MKL_TRACE_IN_M, S, i)], in_f[mkl_plane(MKL_TRACE_IN_M_LO, S, i)]);
    ray.dir.v[2] = float2(in_f[mkl_plane(MKL_TRACE_IN_N, S, i)], in_f[mkl_plane(MKL_TRACE_IN_N_LO, S, i)]);
    ray.opl = float2(in_f[mkl_plane(MKL_TRACE_IN_OPL_HI, S, i)], in_f[mkl_plane(MKL_TRACE_IN_OPL_LO, S, i)]);
    ray.weight = in_f[mkl_plane(MKL_TRACE_IN_WEIGHT, S, i)];

    MklRawState st;
    if ((prm.flags & MKL_TRACE_RECORD) != 0u) {
        const ulong slots = prm.num_steps + 1u;
        float* rec_f = reinterpret_cast<float*>(records);
        uint* rec_u = records + slots * S * ulong(MKL_TRACE_RECORD_FLOATS);
        MklTraceRecords rec = {rec_f, rec_u, S, i};
        st = mkl_trace_raw<MKL_OVERLAYS>(h, s, pool, steps, prm, prm.num_steps, w, ray, rec);
    } else {
        MklNoRecords none;
        st = mkl_trace_raw<MKL_OVERLAYS>(h, s, pool, steps, prm, prm.num_steps, w, ray, none);
    }
    const MklState r = mkl_report(st.pos, st.dir, st.opl, st.index, st.weight, st.status);
    out_f[mkl_plane(MKL_TRACE_OUT_X, S, i)] = r.pos.x;
    out_f[mkl_plane(MKL_TRACE_OUT_Y, S, i)] = r.pos.y;
    out_f[mkl_plane(MKL_TRACE_OUT_Z, S, i)] = r.pos.z;
    out_f[mkl_plane(MKL_TRACE_OUT_L, S, i)] = r.dir.x;
    out_f[mkl_plane(MKL_TRACE_OUT_M, S, i)] = r.dir.y;
    out_f[mkl_plane(MKL_TRACE_OUT_N, S, i)] = r.dir.z;
    out_f[mkl_plane(MKL_TRACE_OUT_OPL_HI, S, i)] = r.opl.x;
    out_f[mkl_plane(MKL_TRACE_OUT_OPL_LO, S, i)] = r.opl.y;
    out_f[mkl_plane(MKL_TRACE_OUT_WEIGHT, S, i)] = r.weight;
    out_f[mkl_plane(MKL_TRACE_OUT_TRANSMITTANCE, S, i)] = st.transmittance;
    out_u[mkl_plane(MKL_TRACE_OUTU_STATUS, S, i)] = st.status;
    out_u[mkl_plane(MKL_TRACE_OUTU_LAST_SURFACE, S, i)] = st.last;
    out_u[mkl_plane(MKL_TRACE_OUTU_STEPS, S, i)] = st.done;
}
#endif

// Planes of the resident ray layout (aiming/mod.rs): 15 floats (position,
// direction, OPL hi and lo, weight, then the position and direction lo
// parts), system, wavelength, 2 logical ids, 4 report words; outputs add 3
// status planes.
#define MKL_RES_SYSTEM 15u
#define MKL_RES_WAVELENGTH 16u
#define MKL_RES_IDS 17u
#define MKL_RES_REPORT 19u
#define MKL_RES_STATUS 23u

// Resident rays carry device-computed system/wavelength indices and may run
// a device-built path program, so both are checked before they index data
// (aiming.metal's mkl_trace_resident).
template <typename Recorder>
static inline void mkl_trace_resident_ray(MKL_TRACE_ARGS, uint i, thread Recorder& rec) {
    const MklTraceParams& prm = u->trace;
    const uint n = prm.stride;
    const float* in_f = reinterpret_cast<const float*>(in);
    float* out_f = reinterpret_cast<float*>(out);
    const uint system = in[mkl_plane(MKL_RES_SYSTEM, n, i)];
    const uint wl = in[mkl_plane(MKL_RES_WAVELENGTH, n, i)];
    uint4 report;
    for (uint k = 0; k < 4; ++k) report[k] = in[mkl_plane(MKL_RES_REPORT + k, n, i)];
    MklRawState ray = {};
    for (uint k = 0; k < 3; ++k) {
        ray.pos.v[k] = float2(in_f[mkl_plane(k, n, i)], in_f[mkl_plane(9 + k, n, i)]);
        ray.dir.v[k] = float2(in_f[mkl_plane(3 + k, n, i)], in_f[mkl_plane(12 + k, n, i)]);
    }
    ray.opl = float2(in_f[mkl_plane(6, n, i)], in_f[mkl_plane(7, n, i)]);
    ray.weight = in_f[mkl_plane(8, n, i)];
    ray.status = report.x;
    ray.last = report.x >> MKL_RAY_SURFACE_SHIFT;
    if (ray.status == 0) {
        bool valid = system < u->systems;
        if (valid) {
            const MklPackedHeader& h = headers[system];
            valid = wl < h.num_wavelengths && prm.start_surface < h.num_surfaces;
            uint current = prm.start_surface;
            bool backwards = false;
            for (uint k = 0; valid && k < prm.num_steps; ++k) {
                uint word = steps[k], si = word & MKL_STEP_SURFACE_MASK,
                     mode = (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK;
                bool back = (word & MKL_STEP_BACKWARD) != 0;
                valid = (word & ~(MKL_STEP_SURFACE_MASK | (MKL_STEP_MODE_MASK << MKL_STEP_MODE_SHIFT) |
                                  MKL_STEP_BACKWARD)) == 0 &&
                        si < h.num_surfaces && mode <= MKL_STEP_REFLECT &&
                        (back ? current > 0 && si + 1 == current : si == current + 1) &&
                        (k == 0 || back == backwards);
                if (valid) {
                    const MklPackedSurface& s = surfaces[h.surface_offset + si];
                    const bool cb = s.shape == MKL_SHAPE_COORD_BREAK;
                    const bool mirror = !cb && s.interaction == MKL_INTERACT_REFLECT;
                    valid = !(back && mode == MKL_STEP_SEQUENTIAL) && !(mode == MKL_STEP_TRANSMIT && mirror) &&
                            !(mode == MKL_STEP_REFLECT && cb);
                    backwards = back;
                    if (mode == MKL_STEP_REFLECT && !mirror) backwards = !backwards;
                }
                current = si;
            }
            if (valid)
                ray = mkl_trace_raw<MKL_OVERLAYS>(h, surfaces + h.surface_offset, pool, steps, prm,
                                                  prm.num_steps, wl, ray, rec);
        }
        if (!valid) ray.status = MKL_AIM_INVALID;
    }
    if (report.x != 0 || ray.status == MKL_AIM_INVALID)
        for (uint k = 0; k <= prm.num_steps; ++k) rec.write(k, ray);
    for (uint k = 0; k < 3; ++k) {
        out_f[mkl_plane(k, n, i)] = ray.pos.v[k].x;
        out_f[mkl_plane(3 + k, n, i)] = ray.dir.v[k].x;
        out_f[mkl_plane(9 + k, n, i)] = ray.pos.v[k].y;
        out_f[mkl_plane(12 + k, n, i)] = ray.dir.v[k].y;
    }
    out_f[mkl_plane(6, n, i)] = ray.opl.x;
    out_f[mkl_plane(7, n, i)] = ray.opl.y;
    out_f[mkl_plane(8, n, i)] = ray.weight;
    out[mkl_plane(MKL_RES_SYSTEM, n, i)] = system;
    out[mkl_plane(MKL_RES_WAVELENGTH, n, i)] = wl;
    out[mkl_plane(MKL_RES_IDS, n, i)] = in[mkl_plane(MKL_RES_IDS, n, i)];
    out[mkl_plane(MKL_RES_IDS + 1, n, i)] = in[mkl_plane(MKL_RES_IDS + 1, n, i)];
    for (uint k = 0; k < 4; ++k) out[mkl_plane(MKL_RES_REPORT + k, n, i)] = report[k];
    out[mkl_plane(MKL_RES_STATUS, n, i)] = ray.status;
    out[mkl_plane(MKL_RES_STATUS + 1, n, i)] = ray.last;
    out[mkl_plane(MKL_RES_STATUS + 2, n, i)] = ray.done;
}

#if !defined(MKL_ONE_ENTRY) || defined(MKL_ENTRY_mkl_trace_resident)
MKL_KERNEL mkl_trace_resident(MKL_TRACE_ARGS) {
    const uint i = mkl_gid_x();
    if (i >= u->trace.num_rays) return;
    MklNoRecords none;
    mkl_trace_resident_ray(u, headers, surfaces, pool, steps, in, out, records, i, none);
}
#endif

#if !defined(MKL_ONE_ENTRY) || defined(MKL_ENTRY_mkl_trace_resident_records)
// Records cost the plain resident trace time: a separate entry point.
MKL_KERNEL mkl_trace_resident_records(MKL_TRACE_ARGS) {
    const uint i = mkl_gid_x();
    if (i >= u->trace.num_rays) return;
    const ulong half = ulong(u->trace.num_steps + 1u) * u->trace.stride * MKL_AIM_RECORD_FLOATS;
    float* hi = reinterpret_cast<float*>(records);
    MklResidentRecords rec = {hi, hi + half, u->trace.stride, i};
    mkl_trace_resident_ray(u, headers, surfaces, pool, steps, in, out, records, i, rec);
}
#endif
