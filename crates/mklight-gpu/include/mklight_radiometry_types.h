/* radiometry numeric records, owned by that package. Included by mklight_gpu_types.h. */
#ifndef MKLIGHT_RADIOMETRY_TYPES_H
#define MKLIGHT_RADIOMETRY_TYPES_H
#define MKL_RAD_VERSION 1
#define MKL_RAD_BARE 0
#define MKL_RAD_IDEAL 1
#define MKL_RAD_TABLE 2
#define MKL_RAD_FILM 3
#define MKL_RAD_OK 0
#define MKL_RAD_REFERENCE 1
#define MKL_RAD_BLOCKED 2
#define MKL_RAD_MAX_LAYERS 32
#ifdef __METAL_VERSION__
struct MklCoatingSpec {
  uint kind, offset, count, mirror;
  float2 wavelength, n_in, n_out, substrate_n, substrate_k, reflectance,
      transmittance, reserved;
};
struct MklCoatingLayer {
  float2 n, k, thickness;
};
struct MklCoatingQuery {
  uint coating, reserved;
  float2 cosine;
};
struct MklCoatingValue {
  float2 amplitudes[8];
  float2 powers[6];
  uint status, reserved;
};
struct MklRadSurface {
  MklCoatingSpec coating;
  float2 rotation[9], transfer[9], translation[3], alpha;
  uint kind, reserved;
};
struct MklPolarizationValue {
  float2 jones[8], proper[8], fields[12], direction[3], power[7], properties[9];
  uint status, defined;
};
/* Rows [offset, end) of count; records are [slots][count][13]. */
struct MklPolarizationParams {
  uint count, offset, slots, wavelengths;
  float2 stokes[4];
  uint layers, end;
};
/* Queries [offset, end). */
struct MklRadParams {
  uint end, offset, specs, layers;
};
#endif
#endif
