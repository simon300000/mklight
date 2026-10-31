// mkl_trace_overlays: mkl_trace for systems with sag overlays
// (MKL_SURF_HAS_SAG_OVERLAY); see trace.metal.
#define MKL_TRACE_OVERLAYS true
#define MKL_TRACE_KERNEL mkl_trace_overlays
#include "trace.metal"
