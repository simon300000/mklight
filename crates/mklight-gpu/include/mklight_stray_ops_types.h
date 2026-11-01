#ifndef MKLIGHT_STRAY_OPS_TYPES_H
#define MKLIGHT_STRAY_OPS_TYPES_H
#define MKL_STRAY_VERSION 1
#define MKL_STRAY_MAX_PRIMITIVES 4096
#define MKL_STRAY_MAX_TILE 65536
#ifdef __METAL_VERSION__
struct MklStraySource {
  uint kind, wavelength_count;
  float2 center[3], axis[3], tangent[2][3];
  float2 radius, inner_radius, cone, distance;
};
struct MklStrayRay {
  float2 origin[3], direction[3];
  uint id[2], draws, wavelength;
};
struct MklStrayPrimitive {
  uint kind, mask;
  float2 origin[3], rotation[9], lower[3], upper[3];
  float2 curvature, conic, inner_radius, outer_radius, aperture[2];
};
struct MklStrayNode {
  float2 lower[3], upper[3];
  uint escape, primitive;
};
struct MklStrayHit {
  float2 point[3], normal[3], distance;
  uint primitive, reserved;
};
struct MklStrayParams {
  uint end, offset, seed[2], first[2], nodes, primitives;
};
#endif
#endif
