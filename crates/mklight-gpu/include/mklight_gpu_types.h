/*
 * mklight_gpu_types.h — THE single definition of every data structure that
 * crosses the host/GPU boundary in MKLight.
 *
 * Consumers:
 *   - C / C++ / Objective-C (host tools, layout probes),
 *   - Metal Shading Language (crates/mklight-metal/shaders),
 *   - CUDA and GLSL/HLSL backends (later).
 * The Rust mirror lives in crates/mklight-gpu/src/layout.rs. Its size, alignment
 * and every field offset are asserted equal to this file by tests, and the
 * integer `#define`s below are parsed by mklight-gpu's build script, so this
 * header is the only place a layout constant is written down.
 *
 * Layout rules (they make the layout identical in C, MSL, CUDA and GLSL std430):
 *   - only 4-byte scalars (MKL_U32, MKL_F32) and fixed arrays of them;
 *     no vector types (MSL float3 is 16 bytes, GLSL vec3 aligns to 16);
 *   - every struct size is a multiple of 16 bytes (explicit `reserved` padding);
 *   - no implicit padding anywhere: Rust derives bytemuck::Pod on the mirror.
 * Changing anything here requires bumping MKL_LAYOUT_VERSION.
 *
 * Units follow docs/CONVENTIONS.md: lengths in mm, wavelengths in µm,
 * curvatures in 1/mm, absorption in 1/mm, indices relative to air.
 *
 * Exact float comparison sentinels (CONVENTIONS §7.6): kernels may compare a
 * packed parameter with exactly 0.0f (curvature, conic, asphere coefficients,
 * paraxial power, to_next_origin components). The packer writes an exact 0.0f
 * whenever the model value is exactly zero. No other float is ever compared for
 * equality.
 */
#ifndef MKLIGHT_GPU_TYPES_H
#define MKLIGHT_GPU_TYPES_H

/* ------------------------------------------------------------------------ */
/* Scalar types per language                                                 */
/* ------------------------------------------------------------------------ */
#if defined(__METAL_VERSION__)
#define MKL_U32 uint
#define MKL_F32 float
#define MKL_STATIC_ASSERT(c, m) static_assert(c, m)
#elif defined(GL_core_profile) || defined(GL_ES)
#define MKL_U32 uint
#define MKL_F32 float
#define MKL_STATIC_ASSERT(c, m)
#else
#include <stdint.h>
#define MKL_U32 uint32_t
#define MKL_F32 float
#if defined(__cplusplus)
#define MKL_STATIC_ASSERT(c, m) static_assert(c, m)
#else
#define MKL_STATIC_ASSERT(c, m) _Static_assert(c, m)
#endif
#endif

/* ------------------------------------------------------------------------ */
/* Versioning and fixed capacities                                           */
/* ------------------------------------------------------------------------ */
/* Layout version: bump on any change to a struct or enum below. */
#define MKL_LAYOUT_VERSION 2u
/* 'MKLP' little-endian; first word of every packed system header. */
#define MKL_PACKED_MAGIC 0x504C4B4Du

/* Maximum wavelengths in one packed system. Systems with more wavelengths are
 * packed in several chunks (one PackedSystem per chunk). */
#define MKL_MAX_WAVELENGTHS 16u
/* Maximum even-asphere coefficients: A4, A6, ..., A22 (orders r^4..r^22).
 * Packing a surface with a non-zero coefficient beyond A22 is an error, never
 * a silent truncation. */
#define MKL_MAX_ASPHERE 10u

/* "No index" sentinel for MKL_U32 index fields. */
#define MKL_INDEX_NONE 0xFFFFFFFFu

/* ------------------------------------------------------------------------ */
/* Enumerations (stored as MKL_U32)                                          */
/* ------------------------------------------------------------------------ */
/* MklPackedSurface.shape */
#define MKL_SHAPE_PLANE 0u
#define MKL_SHAPE_CONIC 1u
#define MKL_SHAPE_EVEN_ASPHERE 2u
#define MKL_SHAPE_PARAXIAL 3u
#define MKL_SHAPE_COORD_BREAK 4u

/* MklPackedSurface.interaction */
#define MKL_INTERACT_REFRACT 0u
#define MKL_INTERACT_REFLECT 1u
#define MKL_INTERACT_DUMMY 2u

/* MklPackedSurface.aperture and the meaning of aperture_params[0..3] (mm):
 *   NONE         -
 *   CIRCULAR     [0] = radius
 *   ANNULAR      [0] = inner radius, [1] = outer radius
 *   RECTANGULAR  [0] = half width x, [1] = half width y
 *   ELLIPTICAL   [0] = semi axis x,  [1] = semi axis y
 * Unused entries are 0. A fixed semi-diameter is packed as CIRCULAR with the
 * MKL_SURF_APERTURE_FROM_SD flag; an explicit aperture always wins. */
#define MKL_APERTURE_NONE 0u
#define MKL_APERTURE_CIRCULAR 1u
#define MKL_APERTURE_ANNULAR 2u
#define MKL_APERTURE_RECTANGULAR 3u
#define MKL_APERTURE_ELLIPTICAL 4u

/* MklPackedSurface.flags */
#define MKL_SURF_HAS_TO_NEXT (1u << 0)     /* to_next_* fields are valid */
#define MKL_SURF_AXIAL (1u << 1)           /* to_next is a pure shift: rot = I, origin = (0, 0, thickness) */
#define MKL_SURF_IS_STOP (1u << 2)         /* aperture stop */
#define MKL_SURF_COATED (1u << 3)          /* has a coating (tables are not on the GPU in layout v1) */
#define MKL_SURF_APERTURE_FROM_SD (1u << 4) /* CIRCULAR aperture comes from a fixed semi-diameter */
#define MKL_SURF_OBJECT (1u << 5)          /* surface 0 */
#define MKL_SURF_IMAGE (1u << 6)           /* surface N-1 */

/* MklPackedHeader.flags */
#define MKL_SYS_OBJECT_AT_INFINITY (1u << 0)
#define MKL_SYS_HAS_TILTS (1u << 1) /* at least one to_next is not axial */

/* ------------------------------------------------------------------------ */
/* Per-ray status word written by trace kernels                              */
/*   status = code | (surface index << MKL_RAY_SURFACE_SHIFT)                */
/* The surface is where the ray stopped (meaningless for MKL_RAY_OK).        */
/* ------------------------------------------------------------------------ */
#define MKL_RAY_OK 0u
#define MKL_RAY_MISS 1u          /* no real intersection with the surface */
#define MKL_RAY_TIR 2u           /* total internal reflection on a refracting surface */
#define MKL_RAY_VIGNETTED 3u     /* blocked by an aperture */
#define MKL_RAY_NONFINITE 4u     /* NaN/Inf appeared (always a bug or invalid input) */
#define MKL_RAY_NOT_CONVERGED 5u /* asphere Newton iteration failed its sag tolerance */
#define MKL_RAY_CODE_MASK 0xFFu
#define MKL_RAY_SURFACE_SHIFT 8u

/* Values recorded per ray per surface when a trace records every surface:
 * x, y, z, l, m, n, opl_hi, opl_lo, weight (in that order). */
#define MKL_TRACE_RECORD_FLOATS 9u

/* ------------------------------------------------------------------------ */
/* Packed system header (one per system; several systems may share one       */
/* surface array, e.g. optimiser candidates or tolerance samples).           */
/* ------------------------------------------------------------------------ */
struct MklPackedHeader {
    MKL_U32 magic;              /*   0  MKL_PACKED_MAGIC */
    MKL_U32 layout_version;     /*   4  MKL_LAYOUT_VERSION */
    MKL_U32 num_surfaces;       /*   8  N, including object (0) and image (N-1) */
    MKL_U32 num_wavelengths;    /*  12  1..MKL_MAX_WAVELENGTHS */
    MKL_U32 surface_offset;     /*  16  index of this system's surface 0 in the surface array */
    MKL_U32 stop;               /*  20  aperture-stop surface index (1..N-2) */
    MKL_U32 primary_wavelength; /*  24  packed index of the primary wavelength, or MKL_INDEX_NONE */
    MKL_U32 flags;              /*  28  MKL_SYS_* */
    MKL_F32 wavelength_um[MKL_MAX_WAVELENGTHS];     /*  32  µm, vacuum-referenced system wavelengths */
    MKL_U32 wavelength_source[MKL_MAX_WAVELENGTHS]; /*  96  index into the model's wavelength list */
    MKL_U32 reserved[4];                            /* 160  zero */
};                                                  /* 176 bytes */

/* ------------------------------------------------------------------------ */
/* Packed surface record                                                     */
/* ------------------------------------------------------------------------ */
/*
 * Sag in the local frame (CONVENTIONS §2), with r² = x² + y²:
 *   z = c r² / (1 + sqrt(1 - (1+k) c² r²))  +  ρ⁴ Σ_j asphere[j] ρ^(2j),
 *   ρ² = r² · asphere_inv_norm2.
 * The asphere coefficients are stored normalised: asphere[j] = A_(2j+4) · R^(2j+4)
 * (mm), with R = 1/sqrt(asphere_inv_norm2) an exact power of two chosen by the
 * packer near the surface's semi-diameter. Scaling by a power of two is exact,
 * so normalisation adds no rounding; it only keeps high-order coefficients
 * (A22 ~ 1e-35 mm^-21) away from f32 underflow/overflow and keeps Horner
 * intermediates O(sag).
 *
 * Transform to the next surface (CONVENTIONS §7.1). to_next_origin is the
 * next surface's vertex expressed in THIS surface's local frame, and
 * to_next_rot maps this frame's axes onto the next frame's axes, so
 *   p_next = to_next_rot · (p − to_next_origin),   d_next = to_next_rot · d
 * (row-major: rot[3*i+j] = M_ij). Subtracting the origin before rotating keeps
 * the operand small near the next vertex and avoids the cancellation of the
 * rot·p + t form. to_next_origin_lo carries the f32 rounding residual of the
 * origin (hi + lo equals the f64 value to ~2^-48 relative), so long air gaps
 * can be applied in double-float where the error budget requires it.
 * In terms of mklight_core::ResolvedSurface::to_next (a Frame giving the pose
 * of the next surface in this one): origin = to_next.origin and
 * rot = to_next.rot transposed.
 */
struct MklPackedSurface {
    MKL_U32 shape;             /*   0  MKL_SHAPE_* */
    MKL_U32 interaction;       /*   4  MKL_INTERACT_* */
    MKL_U32 aperture;          /*   8  MKL_APERTURE_* */
    MKL_U32 flags;             /*  12  MKL_SURF_* */
    MKL_F32 curvature;         /*  16  c = 1/R (1/mm); 0 for plane/paraxial/coordinate break */
    MKL_F32 conic;             /*  20  k (dimensionless) */
    MKL_F32 paraxial_power;    /*  24  1/f (1/mm) for MKL_SHAPE_PARAXIAL, else 0 */
    MKL_F32 asphere_inv_norm2; /*  28  1/R² (1/mm²), exact power of two; 1 when unused */
    MKL_U32 num_asphere;       /*  32  coefficients in use (0..MKL_MAX_ASPHERE) */
    MKL_U32 source_index;      /*  36  surface index in the model */
    MKL_U32 reserved0;         /*  40  zero */
    MKL_U32 reserved1;         /*  44  zero */
    MKL_F32 asphere[MKL_MAX_ASPHERE]; /*  48  normalised coefficients (mm), see above */
    MKL_F32 aperture_params[4];       /*  88  mm, meaning per MKL_APERTURE_* */
    MKL_F32 to_next_rot[9];           /* 104  row-major rotation into the next local frame */
    MKL_F32 to_next_origin[3];        /* 140  next vertex in this local frame (mm), f32 hi part */
    MKL_F32 to_next_origin_lo[3];     /* 152  f32 residual of to_next_origin (mm) */
    MKL_U32 reserved2[3];             /* 164  zero */
    MKL_F32 n_after[MKL_MAX_WAVELENGTHS];     /* 176  index after the surface per packed wavelength */
    MKL_F32 alpha_after[MKL_MAX_WAVELENGTHS]; /* 240  internal absorption after the surface (1/mm) */
};                                            /* 304 bytes */

/* Size checks (sizes are also asserted in Rust). */
#define MKL_PACKED_HEADER_SIZE 176u
#define MKL_PACKED_SURFACE_SIZE 304u
MKL_STATIC_ASSERT(sizeof(struct MklPackedHeader) == MKL_PACKED_HEADER_SIZE, "MklPackedHeader size");
MKL_STATIC_ASSERT(sizeof(struct MklPackedSurface) == MKL_PACKED_SURFACE_SIZE, "MklPackedSurface size");

/* ------------------------------------------------------------------------ */
/* Kernel parameter blocks (passed by value with setBytes / push constants)  */
/* ------------------------------------------------------------------------ */

/* Philox stream kernel: draws [first_block*4, (first_block+num_blocks)*4) of
 * the stream (seed, stream), identical to mklight_core::rng::Philox. */
struct MklPhiloxParams {
    MKL_U32 key[2];        /*  0  seed lo, seed hi */
    MKL_U32 stream[2];     /*  8  stream lo, stream hi -> counter words 2, 3 */
    MKL_U32 first_block[2];/* 16  block lo, hi -> counter words 0, 1 */
    MKL_U32 num_blocks;    /* 24 */
    MKL_U32 reserved;      /* 28 */
};                         /* 32 bytes */

/* Compensated reduction kernel. Values are channel-major: channel c occupies
 * [c*channel_stride, c*channel_stride + count). Pass 1 may be split into
 * several launches over group ranges: a launch of k threadgroups computes
 * groups first_group .. first_group + k - 1 (each group's partial does not
 * depend on the split). */
struct MklReduceParams {
    MKL_U32 count;          /*  0  values per channel */
    MKL_U32 channel_stride; /*  4  elements between channels */
    MKL_U32 num_groups;     /*  8  threadgroups per channel in pass 1 (all launches) */
    MKL_U32 first_group;    /* 12  pass 1: index of this launch's first group; pass 2: 0 */
};                          /* 16 bytes */

/* Packed-sag probe kernel: evaluate the sag of one packed surface. */
struct MklSagParams {
    MKL_U32 surface;  /*  0  absolute index into the surface array */
    MKL_U32 count;    /*  4  number of r² samples */
    MKL_U32 reserved[2];
};                    /* 16 bytes */

MKL_STATIC_ASSERT(sizeof(struct MklPhiloxParams) == 32, "MklPhiloxParams size");
MKL_STATIC_ASSERT(sizeof(struct MklReduceParams) == 16, "MklReduceParams size");
MKL_STATIC_ASSERT(sizeof(struct MklSagParams) == 16, "MklSagParams size");

/* Fixed reduction geometry. Results depend on the summation order, so these
 * are part of the numerical contract shared with the CPU-f32 emulation. */
#define MKL_REDUCE_THREADS 256u     /* threads per threadgroup */
#define MKL_REDUCE_MAX_GROUPS 256u  /* pass-1 threadgroups per channel (cap) */
#define MKL_REDUCE_ITEMS_PER_THREAD 16u /* target items per thread when choosing num_groups */

/* Reduction modes (function constant 0 of the reduction kernels). */
#define MKL_REDUCE_PLAIN 0u /* plain f32 (for comparison only) */
#define MKL_REDUCE_KAHAN 1u /* per-thread Neumaier f32, pairs combined in double-float */
#define MKL_REDUCE_DF 2u    /* double-float accumulation throughout */

/* Double-float self-test operations (function constant 0 of mkl_df_ops). */
#define MKL_DFOP_TWO_SUM 0u      /* (a.hi, b.hi) -> two_sum */
#define MKL_DFOP_FAST_TWO_SUM 1u /* (a.hi, b.hi) -> fast_two_sum, |a| >= |b| */
#define MKL_DFOP_TWO_PROD 2u     /* (a.hi, b.hi) -> two_prod (fma) */
#define MKL_DFOP_ADD 3u          /* df + df */
#define MKL_DFOP_MUL 4u          /* df * df */
#define MKL_DFOP_DIV 5u          /* df / df */
#define MKL_DFOP_SQRT 6u         /* sqrt(df a) */
#define MKL_DFOP_ADD_F32 7u      /* df + a f32 (b.hi) */
#define MKL_DFOP_MUL_F32 8u      /* df * a f32 (b.hi) */
#define MKL_DFOP_HYPOT2 9u       /* hypot(a.hi, b.hi) in f32 (stable, scaled) */
#define MKL_DFOP_COUNT 10u

/* Numerics probe: number of result words written by mkl_numerics_probe. */
#define MKL_PROBE_WORDS 8u

#endif /* MKLIGHT_GPU_TYPES_H */
