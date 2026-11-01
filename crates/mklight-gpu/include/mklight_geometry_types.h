#ifndef MKLIGHT_GEOMETRY_TYPES_H
#define MKLIGHT_GEOMETRY_TYPES_H
#define MKL_GEOMETRY_SAMPLE_WORDS 8u
#define MKL_GEOMETRY_POINT_CHANNELS 5u
struct MklGeometrySample {
    MKL_U32 system, surface, reserved[2];
    float xy[4];
};
/* Rust twin: mklight_metal geometry::Params. All offsets count elements. */
struct MklGeometryParams {
    MKL_U32 count, offset, stride, chiefs;
    MKL_U32 segment, centroid, systems, reserved;
    float defocus[2], origin[6];
};
MKL_STATIC_ASSERT(sizeof(struct MklGeometrySample) == 32, "geometry sample");
MKL_STATIC_ASSERT(sizeof(struct MklGeometryParams) == 64, "geometry parameters");
#endif
