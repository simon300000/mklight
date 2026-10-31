/* geometry numeric sidecar. All offsets are typed element indices.
 * Descriptor words: input, output, count, flags. Domain records are added by
 * the owning package without changing the packed system layout. */
#ifndef MKLIGHT_GEOMETRY_TYPES_H
#define MKLIGHT_GEOMETRY_TYPES_H
struct MklGeometryOp {
    MKL_U32 input;
    MKL_U32 output;
    MKL_U32 count;
    MKL_U32 flags;
};
MKL_STATIC_ASSERT(sizeof(struct MklGeometryOp) == 16, "geometry descriptor size");
#endif
