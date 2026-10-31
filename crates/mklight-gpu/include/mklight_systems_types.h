/* P4 sidecar v1. All offsets are words except DF coefficient offsets.
 * Candidate-major, then wavelength chunks of at most MKL_MAX_WAVELENGTHS.
 * No change to the packed trace layout. */
#ifndef MKLIGHT_SYSTEMS_TYPES_H
#define MKLIGHT_SYSTEMS_TYPES_H
#define MKL_SYSTEMS_VERSION 1
#define MKL_SYSTEMS_READY 0
#define MKL_SYSTEMS_OUTSIDE_BOUNDS 1
#define MKL_SYSTEMS_INVALID_NUMERIC 2
#define MKL_SYSTEMS_DIRTY_ALL 7
struct MklSystemsDispatch {
    uint32_t first, count, parameters, instructions;
    uint32_t patches, guards, header_words, surface_words;
    uint32_t coefficient_words, first_order_words, chunks, surfaces;
    uint32_t model_words, first_instruction, instruction_count, reserved;
};
/* Instruction uint4: opcode, a, b, c. Constant: a/b are hi/lo bits;
 * parameter: a is its column; arithmetic operands are earlier node indices.
 * Opcodes: constant=0, parameter=1, add=2, sub=3, mul=4, div=5,
 * sqrt=6, select(a<=0,b,c)=7. Validated host compilation is authoritative.
 * Patches uint4: destination (header/surface/coefficient/first-order/model),
 * hi word offset, lo word offset (UINT_MAX means absent), node index.
 * Positive finite guards follow the patches as uint node indices.
 * Status uint2: preparation status, dirty aim/aperture/spectrum bits.
 * READY is a numeric preparation status, never optical qualification. */
#endif
