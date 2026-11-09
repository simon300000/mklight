#ifndef MKLIGHT_STRAY_OPS_TYPES_H
#define MKLIGHT_STRAY_OPS_TYPES_H
#define MKL_STRAY_VERSION 6
#define MKL_STRAY_FEATURES_CONSTANT 60
/* Grazing handoff (Transport::sag_mode bit 0): an extended-sag scan of more
   cells stops the source before the event for the host to continue. */
#define MKL_STRAY_SAG_HANDOFF_CELLS 32
#define MKL_STRAY_MAX_PRIMITIVES 1048576
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
struct MklStraySurfaceData { MklPackedSurface surface; float2 coefficients[256]; };
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

struct MklStrayMaterial {
  MklCoatingSpec coatings[2];
  float2 index[2], alpha, specular, focal, bsdf[3], phase[10], norm_radius, blaze[3];
  uint kind, media[2], sink, pass_start, pass_count, bsdf_kind, tis_start, tis_count, doe;
  int selected;
  uint levels, phase_count, bsdf_flags;
};
struct MklStrayPass { uint surface, side, mask, reserved; float2 aperture[2]; };
struct MklStrayMedium { float2 alpha; uint ambient, reserved; };
struct MklStrayHistory {
  uint flags, reflections, next_seq; int next_specular;
  uint first[2], scatter_events, length, overflow, order_count, events[6], orders[32][2];
};
struct MklStrayState {
  MklStrayRay ray; float2 weight; MklStrayHistory history;
  uint medium, events, splits, reserved;
};
struct MklStrayDetection {
  MklStrayHistory history; float2 weight, x, y; uint path_class, wavelength;
};
struct MklStrayLedger {
  float2 sums[21]; ulong draws; uint events, hits, top, status, wavelength, reserved;
};
struct MklStraySettings {
  float2 weight, roulette, ceiling, rare_floor;
  uint mode, max_splits, max_events, scatter, start_medium, detector_accept, wavelengths, sequence_count;
};
/* Wavefront transport: `tile_count` list entries from `tile_first` on (0:
   the whole list), `head_events` events per source (0: no limit),
   `sag_mode` as Transport::sag_mode. */
struct MklStrayTransportParams {
  uint count, nodes, stack_capacity, hit_capacity, tile_count, parity, seed[2], tile_first, head_events, sag_mode, reserved;
};
#endif
#endif
