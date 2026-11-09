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
typedef struct { unsigned int n,queries,kind,first,count,finish,reserved[2]; float spacing[2],center[2][2],normalization[2]; } MklWaveIntegralParams;
typedef struct { unsigned int n,side,count,kind; float sine[2],spacing[2],frame[3][3][2],mapping[2][2][2]; } MklWaveFftMtfParams;
typedef struct {
    unsigned int mode,width,height,channels,source_width,source_height,source_channels,grid;
    unsigned int radius,canvas,node,channel,tile[4],pattern,quadrature,previous,first;
    unsigned int count,seed[2],reserved;
    float origin[2][2],pitch[2][2],aperture[2][2],spacing[2],normalization[2],well,sigma;
} MklWaveSimulationParams;
typedef struct { unsigned int size,length,mode,first,count,fields,queries,query_first,query_count,finish,reserved[2]; float center[2][2],normalization[2],frequency_step[2]; } MklWaveShellParams;
#endif
