/* Collectives. Offsets and lengths count elements. Included by mklight_gpu_types.h. */
#ifndef MKLIGHT_COLLECTIVES_TYPES_H
#define MKLIGHT_COLLECTIVES_TYPES_H
#define MKL_COLLECT_SUM 0u
#define MKL_COLLECT_SUMMARY 1u
#define MKL_COLLECT_MOMENTS2 2u
#define MKL_COLLECT_SOURCE_PRODUCTS 3u
/* Accumulators: MKL_REDUCE_PLAIN/KAHAN/DF, then extrema. */
#define MKL_COLLECT_MIN 3u
#define MKL_COLLECT_MAX 4u
/* Pivoted QR threads per column group, fixed by the row count (and with it
   the reduction tree): wide groups from MKL_COLLECT_QR_WIDE_ROWS rows. */
#define MKL_COLLECT_QR_THREADS 256u
#define MKL_COLLECT_QR_WIDE_THREADS 1024u
#define MKL_COLLECT_QR_WIDE_ROWS 8192u
/* Radix sort: 8-bit digits, 256 threads x 4 records per block. */
#define MKL_COLLECT_SORT_BLOCK 1024u
/* Rust twin: mklight_metal collectives::SegmentParams. */
struct MklSegmentParams {
    MKL_U32 count, channels, segments, first, end, max_len, max_groups, kind;
    MKL_U32 mode, phase, value_words, reference_words, weight_words, masked, mask_value, terms;
};
/* Rust twin: mklight_metal collectives::QrParams. */
struct MklQrParams {
    MKL_U32 rows, cols, batch, step, input_words, tolerance_bits, aux_stride, threads;
};
/* Scan, compaction and sort steps. Rust twin: mklight_metal collectives::StepParams. */
struct MklCollectParams {
    MKL_U32 n, kind, flag, words, shift, blocks, value_words, reserved;
};
MKL_STATIC_ASSERT(sizeof(struct MklSegmentParams) == 64, "segment parameters");
MKL_STATIC_ASSERT(sizeof(struct MklQrParams) == 32, "QR parameters");
MKL_STATIC_ASSERT(sizeof(struct MklCollectParams) == 32, "collective parameters");
#endif
