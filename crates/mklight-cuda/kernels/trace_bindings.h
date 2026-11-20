// The bindings of the trace modules' entry points (kernels/trace/*.wgsl and
// aiming.wgsl share them): binding 1 MklWgslTraceParams, 2 headers,
// 3 surfaces, 4 coefficient pool, 5 path program, 6 input words, 7 output
// words, 8 records.
#ifndef MKLIGHT_CUDA_TRACE_BINDINGS_H
#define MKLIGHT_CUDA_TRACE_BINDINGS_H

#include "trace_shared.h"

struct MklWgslTraceParams {
    MklTraceParams trace;
    // Systems in the bound set (resident rays carry device indices).
    uint systems, reserved0, reserved1, reserved2;
};

#define MKL_TRACE_ARGS                                                                           \
    const MklWgslTraceParams *u, const MklPackedHeader *headers,                                 \
        const MklPackedSurface *surfaces, const float *pool, const uint *steps, const uint *in, \
        uint *out, uint *records

#endif
