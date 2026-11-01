#ifndef MKLIGHT_WAVE_OPS_TYPES_H
#define MKLIGHT_WAVE_OPS_TYPES_H
/* Larger extrapolation gaps request the reference rather than monopolize a command. */
#define MKL_WAVE_MAX_EXTRAPOLATION 128
/* f32 pairs below are unevaluated hi+lo values. */
typedef struct {
    unsigned int n, size, field_side, mode;
    unsigned int first, count, sample_first, sample_count;
    unsigned int unit_energy, shifted, reserved[2];
    float sin_max[2], wavelength[2], radius[2], spacing[2];
    float origin[3][2], x_axis[3][2], y_axis[3][2], normal[3][2];
    float aperture[2], sigma, projection;
} MklWaveParams;
typedef struct {
    unsigned int nx,ny,n,supersample,mask,edges,first,count;
    float aperture[2]; unsigned int reserved[2];
} MklWaveResampleParams;
typedef struct {
    unsigned int n,side,model,infinite,power,first,count,reserved;
    float sin_max[2],propagation,sheet_factor;
    float frame[3][3][2],object_axes[2][3][2];
} MklWavePlacementParams;
typedef struct { unsigned int count,inputs,complex,phase; float weights[16][2],offsets[16][2][2]; } MklWaveSpectralParams;
typedef struct { unsigned int n,step,side,count; float scale[2],offset[2][2],inverse[2][2][2]; } MklWaveBlurParams;
typedef struct { unsigned int n,queries,kind,first,count,finish,reserved[2]; float spacing[2],center[2][2],normalization[2]; } MklWaveIntegralParams;
#endif
