/* transforms numeric records, owned by that package. Included by mklight_gpu_types.h. */
#ifndef MKLIGHT_TRANSFORMS_TYPES_H
#define MKLIGHT_TRANSFORMS_TYPES_H
#define MKL_TRANSFORM_FFT_ROWS 0u
#define MKL_TRANSFORM_FFT_STAGE 1u
#define MKL_TRANSFORM_TRANSPOSE 2u
#define MKL_TRANSFORM_CHIRP_ROWS 3u
#define MKL_TRANSFORM_KERNEL 4u
#define MKL_TRANSFORM_PRE 5u
#define MKL_TRANSFORM_POST 6u
#define MKL_TRANSFORM_MULTIPLY 7u
#define MKL_TRANSFORM_PHASE 8u
/* Offsets count complex elements. Rust twin: transforms::recipe::Params. */
struct MklTransformParams {
    MKL_U32 mode, count, width, height, nx, ny, ox, oy;
    MKL_U32 radices[2], stages, span, sign_bits, scale_bits, flags, reserved;
    MKL_F32 affine[6][2];
};
/* Rust twin: transforms::correlation::Params. */
struct MklCorrelationParams {
    MKL_U32 n, height, frequencies, first, optical, covered, prepared, reserved;
    MKL_F32 mapping[6][2], frame[3][3][2], sin_max[2], wavelength[2];
    MKL_U32 reserved_tail[2];
};
#define MKL_TRANSFORM_PARAMS_SIZE 112u
#define MKL_CORRELATION_PARAMS_SIZE 176u
MKL_STATIC_ASSERT(sizeof(struct MklTransformParams)==MKL_TRANSFORM_PARAMS_SIZE,"transform parameters");
MKL_STATIC_ASSERT(sizeof(struct MklCorrelationParams)==MKL_CORRELATION_PARAMS_SIZE,"correlation parameters");
#endif
