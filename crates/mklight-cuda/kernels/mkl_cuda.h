// Common header of the CUDA kernels.
//
// The kernels implement the WebGPU backend's kernel interface (entry names,
// parameter blocks and storage bindings of crates/mklight-webgpu/kernels,
// GPU.md §11.1), so the WebGPU host operations drive them unchanged:
//
//   extern "C" __global__ void <entry>(const Params* u, <binding 2>, <binding 3>, ...)
//
// `u` is the pass's uniform block in device memory (null when the pass has
// none) and the storage bindings follow in order. Block size = the WGSL
// entry's @workgroup_size, grid = the pass's workgroups.
//
// The arithmetic is the Metal kernels' (crates/mklight-metal/shaders),
// included directly through the MSL subset of <metal_stdlib> (metal_stdlib.h).
#ifndef MKLIGHT_CUDA_H
#define MKLIGHT_CUDA_H

#include "mklight_common.h"

#define MKL_KERNEL extern "C" __global__ void

// WGSL builtins.
__device__ inline uint mkl_gid_x() { return blockIdx.x * blockDim.x + threadIdx.x; }
__device__ inline uint mkl_gid_y() { return blockIdx.y * blockDim.y + threadIdx.y; }
__device__ inline uint mkl_gid_z() { return blockIdx.z * blockDim.z + threadIdx.z; }

// The protected operations of common.wgsl: on CUDA every operation is
// already one IEEE rounding (no contraction, no reassociation).
__device__ inline float mkl_add(float a, float b) { return __fadd_rn(a, b); }
__device__ inline float mkl_sub(float a, float b) { return __fsub_rn(a, b); }
__device__ inline float mkl_mul(float a, float b) { return __fmul_rn(a, b); }
__device__ inline float mkl_div(float a, float b) { return __fdiv_rn(a, b); }
__device__ inline float mkl_sqrt_rn(float a) { return __fsqrt_rn(a); }

#endif
