/* Portable collective numeric contract. Offsets and lengths count elements. */
#ifndef MKLIGHT_COLLECTIVES_TYPES_H
#define MKLIGHT_COLLECTIVES_TYPES_H
#define MKL_COLLECT_SUM 0u
#define MKL_COLLECT_SUMMARY 1u
#define MKL_COLLECT_MOMENTS2 2u
#define MKL_COLLECT_SOURCE_PRODUCTS 3u
#define MKL_COLLECT_THREADS 256u
#define MKL_COLLECT_QR_MAX_COLUMNS 64u
#define MKL_COLLECT_QR_FIELDS 4u
/* Main parameter blocks use sixteen u32 words, not Rust request structs.
   Segmented mask/weight options use an additional two u32 words.
   The encoder documents the words per kernel. Numeric arrays use scalar
   loads so typed views need no additional float2 alignment. */
#endif
