#ifndef MKLIGHT_OPTIM_OPS_TYPES_H
#define MKLIGHT_OPTIM_OPS_TYPES_H
#define MKL_OPTIM_VERSION 1
#ifdef __METAL_VERSION__
struct MklOptimParams {
    uint kind, count, offset, batches, rows, columns, inner, flag;
    float2 damping;
};
#endif
#endif
