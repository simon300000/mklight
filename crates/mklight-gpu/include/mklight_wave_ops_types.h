#ifndef MKLIGHT_WAVE_OPS_TYPES_H
#define MKLIGHT_WAVE_OPS_TYPES_H
/* f32 pairs below are unevaluated hi+lo values. */
typedef struct {
    unsigned int n, size, field_side, mode;
    unsigned int first, count, sample_first, sample_count;
    unsigned int unit_energy, shifted, reserved[2];
    float sin_max[2], wavelength[2], radius[2], spacing[2];
    float origin[3][2], x_axis[3][2], y_axis[3][2], normal[3][2];
    float aperture[2], sigma, projection;
} MklWaveParams;
#endif
