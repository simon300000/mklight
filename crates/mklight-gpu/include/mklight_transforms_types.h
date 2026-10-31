#ifndef MKLIGHT_TRANSFORMS_TYPES_H
#define MKLIGHT_TRANSFORMS_TYPES_H
#define MKL_TRANSFORM_COPY 0u
#define MKL_TRANSFORM_TRANSPOSE 1u
#define MKL_TRANSFORM_FFT 2u
#define MKL_TRANSFORM_TWIDDLE 3u
#define MKL_TRANSFORM_PRE 4u
#define MKL_TRANSFORM_KERNEL 5u
#define MKL_TRANSFORM_POST 6u
#define MKL_TRANSFORM_MULTIPLY 7u
#define MKL_TRANSFORM_PHASE 8u
/* All offsets count complex elements. Parameters are 16 u32 words followed
   by six (hi,lo) coefficients; Rust recipe::Params is the host twin. */
struct MklTransformParams {
    MKL_U32 mode, count, width, height, nx, ny, ox, oy;
    MKL_U32 radix, span, sign_bits, scale_bits, start, conjugate, reserved0, reserved1;
    MKL_F32 affine[6][2];
};
struct MklCorrelationParams {
    MKL_U32 n, height, frequency, start, optical, covered, prepared, reserved;
    MKL_F32 mapping[6][2], frame[3][3][2], sin_max[2], wavelength[2];
    MKL_U32 reserved_tail[2];
};
#define MKL_TRANSFORM_PARAMS_SIZE 112u
#define MKL_CORRELATION_PARAMS_SIZE 176u
MKL_STATIC_ASSERT(sizeof(struct MklTransformParams)==MKL_TRANSFORM_PARAMS_SIZE,"transform parameters");
MKL_STATIC_ASSERT(sizeof(struct MklCorrelationParams)==MKL_CORRELATION_PARAMS_SIZE,"correlation parameters");
#endif
