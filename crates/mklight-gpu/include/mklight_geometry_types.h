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
/* Image-point illumination solve (relative illumination). One field row per
 * image point, host-prepared in f64 and stored as hi/lo pairs: the aiming
 * metadata, the chief aim point and stop-Jacobian inverse that seed every
 * pupil aim, the image point X0 and its tangent basis, the chief object
 * parameters q0 with the chord inverse (dX/dq)^-1 at (q0, P = 0), the object
 * difference step and the image-point tolerance. Stop aims converge at
 * 1e-12 of the stop radius; a solve that stalls (or exhausts its steps)
 * within aim_floor of the stop radius (the reference's own aim tolerance,
 * at least the device trace's noise floor) is accepted there. Status
 * nonzero: every node of the field fails. */
#define MKL_RI_FIELD_WORDS 96u
struct MklRiField {
    struct MklAimField aim;
    float chief[4], aim_inverse[8], x0[4], basis[12], q0[4], q_inverse[8];
    float hq[2], tolerance[2];
    MKL_U32 status;
    float aim_floor;
};
/* One aimed ray (q, P) of the flat illumination stage: aimed at the stop
 * from `seed` (flag MKL_RI_SEEDED) or from the chief prediction, then traced
 * to the image, clipped with MKL_RI_CLIP. MKL_RI_AIMED: the seed is the
 * converged aim of this (q, P), traced to the image directly. MKL_RI_SKIP:
 * no work (status MKL_AIM_OUTSIDE). A node's pencil is 9 tasks: the clipped
 * ray, then q0+-, q1+-, P0+-, P1+- central differences. */
#define MKL_RI_CLIP 1u
#define MKL_RI_SEEDED 2u
#define MKL_RI_AIMED 4u
#define MKL_RI_SKIP 8u
#define MKL_RI_PENCIL 9u
#define MKL_RI_TASK_WORDS 14u
struct MklRiTask {
    MKL_U32 field, flags;
    float q[4], pupil[4], seed[4];
};
/* Result of one task: status 0 or a ray/aim failure code; Newton iterations
 * of the stop aims; image x/y; direction cosines on the image basis; the
 * converged aim point. */
#define MKL_RI_RAY_WORDS 14u
struct MklRiRay {
    MKL_U32 status, iterations;
    float x[4], lm[4], aim[4];
};
MKL_STATIC_ASSERT(sizeof(struct MklRiField) == 4 * MKL_RI_FIELD_WORDS, "illumination field");
MKL_STATIC_ASSERT(sizeof(struct MklRiTask) == 4 * MKL_RI_TASK_WORDS, "illumination task");
MKL_STATIC_ASSERT(sizeof(struct MklRiRay) == 4 * MKL_RI_RAY_WORDS, "illumination ray");
#endif
