/* Resident aiming records. All floating-point quantities are hi/lo pairs.
 * Fields are small immutable metadata; samples retain exact two-word IDs.
 * Source stores direction tangents (infinite/angle fields), otherwise local
 * object x/y. Real-image fields store the paraxial source guess plus target.
 * Rotation maps the object-local frame to the entrance/global frame. */
#ifndef MKLIGHT_AIMING_TYPES_H
#define MKLIGHT_AIMING_TYPES_H
#define MKL_AIM_FINITE 1u
#define MKL_AIM_TELECENTRIC 2u
#define MKL_AIM_REAL 4u
#define MKL_AIM_IMAGE_HEIGHT 8u
#define MKL_AIM_ANGLE 16u
#define MKL_AIM_FIELD_WORDS 50u
#define MKL_AIM_SAMPLE_WORDS 8u
#define MKL_AIM_CACHE_WORDS 24u
#define MKL_AIM_REPORT_PLANES 4u
#define MKL_AIM_FAILED 9u
#define MKL_AIM_FIELD_UNREACHABLE 10u
#define MKL_AIM_INVALID 11u
#define MKL_PUPIL_MISS 12u
#define MKL_AIM_PRIMARY_UNAVAILABLE 13u
/* A sample the caller excluded from aiming: outside the unit pupil disk when
 * culling, or carrying MKL_AIM_SKIP in its field word. Never aimed, traced or
 * counted as an aim. */
#define MKL_AIM_OUTSIDE 14u
/* A pending ray whose aim has not finished its iteration budget yet; only
 * ever observed between continuation rounds. */
#define MKL_AIM_PENDING 15u
#define MKL_AIM_SKIP 0x80000000u
#define MKL_AIM_RECORD_FLOATS 13u
/* Seed table: a side² lattice of aim solves per real-aim field row over
 * vignetted pupil coordinates [-1.125, 1.125]²; nodes beyond pupil radius
 * 1.25 are not solved. MKL_AIM_TABLE_SIDE is the default side (node
 * spacing 1/16); hosts may supply tables of another odd side. */
#define MKL_AIM_TABLE_SIDE 37u
#define MKL_AIM_NODE_WORDS 6u
/* Resumable pupil-aim state between bounded iteration units (words):
 * x[4] r[4] j[8] rn[2] xc[4] prev[4] jc[8] phase it total flags. */
#define MKL_AIM_STATE_WORDS 38u
/* Numeric layouts use scalar arrays: no vector ABI padding. */
struct MklAimField {
    unsigned int system, wavelength, flags, reserved;
    float source[4], target[4], pupil[6], vignette[8], origin[6], rotation[18];
};
struct MklAimSample {
    float pupil[4];
    unsigned int field, id_lo, id_hi;
    float weight;
};
struct MklAimNode {
    float aim[4];
    unsigned int status, reserved;
};
struct MklAimCache {
    float source[6], chief[4], jacobian[8], residual[2];
    unsigned int status, iterations, has_jacobian, reserved;
};
#endif
