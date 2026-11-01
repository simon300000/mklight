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
#endif
