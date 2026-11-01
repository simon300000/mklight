#ifndef MKLIGHT_GHOST_OPS_TYPES_H
#define MKLIGHT_GHOST_OPS_TYPES_H
#define MKL_GHOST_VERSION 1
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
struct MklGhostFirstOrderStep{float2 power,index,distance;};
struct MklGhostTraceParams {
  MklTraceParams trace;
  uint end, offset, paths, wavelengths;
  uint reject_virtual, reserved;
  float2 sensor[2];
};
#endif
#endif
