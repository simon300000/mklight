/* wave_ops numeric sidecar. All offsets are typed element indices.
 * Descriptor words: input, output, count, flags. Domain records are added by
 * the owning package without changing the packed system layout. */
#ifndef MKLIGHT_WAVE_OPS_TYPES_H
#define MKLIGHT_WAVE_OPS_TYPES_H
struct MklWaveOpsOp {
    MKL_U32 input;
    MKL_U32 output;
    MKL_U32 count;
    MKL_U32 flags;
};
MKL_STATIC_ASSERT(sizeof(struct MklWaveOpsOp) == 16, "wave_ops descriptor size");
#endif
