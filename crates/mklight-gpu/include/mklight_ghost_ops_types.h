#ifndef MKLIGHT_GHOST_OPS_TYPES_H
#define MKLIGHT_GHOST_OPS_TYPES_H
#define MKL_GHOST_VERSION 2
/* Function constant of mkl_ghost_trace: bit 0 folds coating powers, bit 1
   keeps the thin-film matrices (otherwise only bare/ideal/table kinds). */
#define MKL_GHOST_FEATURES_CONSTANT 61
#define MKL_GHOST_FEATURE_COATINGS 1u
#define MKL_GHOST_FEATURE_FILMS 2u
#define MKL_GHOST_AXIAL_NONE 0u
#define MKL_GHOST_AXIAL_TRANSMIT 1u
#define MKL_GHOST_AXIAL_REFLECT 2u
#define MKL_GHOST_AXIAL_MIRROR 3u
#define MKL_GHOST_AXIAL_DETECT 4u
#define MKL_GHOST_AXIAL_SENSOR 5u
#define MKL_GHOST_CRITICAL_BAND_BITS 0x358637bdu
#define MKL_GHOST_MAX_SEGMENT_MM 1000u
/* 2^-28: relative aperture margin (of r^2, |x| or the ellipse form) inside
   which the f64 reference decides a clipping test. Clear apertures sized by
   the rim rays of the edge field put those rays exactly on the edge. */
#define MKL_GHOST_APERTURE_TIE_BITS 0x31800000u
#define MKL_GHOST_FIRSTORDER_SLOPE_BITS 0x358637bdu
#define MKL_GHOST_VIRTUAL 16
#define MKL_GHOST_REFERENCE 17
#ifdef __METAL_VERSION__
struct MklGhostStep {
  uint word, physical, coating, channel;
  float2 scale, efficiency, alpha, power;
};
struct MklGhostProbe {
  float2 position[3], direction[3];
  uint path, wavelength;
};
struct MklGhostValue {
  float2 position[3], direction[3], weight_s, weight_p, internal;
  uint status, done;
};
struct MklGhostLaunchSample{float2 pupil[2];int2 lattice;uint field,path,wavelength,reserved;};
struct MklGhostLeaf{uint row,destination;float2 weight;uint batch,map,pixel,reserved;};
struct MklGhostFirstOrderStep{float2 power,index,distance,efficiency,absorption;uint coating,channel;};
struct MklGhostTraceParams {
  MklTraceParams trace;
  uint end, offset, paths, wavelengths;
  uint reject_virtual, specs, layers, reserved;
  float2 sensor[2], sensor_reflectance;
};
#endif
#endif
