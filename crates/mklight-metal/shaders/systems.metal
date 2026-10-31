#include "mklight_common.h"

static inline bool mkl_p4_le(float2 a, float2 b) {
    return a.x < b.x || (a.x == b.x && a.y <= b.y);
}
kernel void mkl_systems_prepare(
    device const float2 *parameters [[buffer(0)]],
    device const uint *slots [[buffer(1)]],
    device const float2 *bounds [[buffer(2)]],
    device const uint *program [[buffer(3)]],
    device const uint *base [[buffer(4)]],
    device float2 *scratch [[buffer(5)]],
    device uint *headers [[buffer(6)]],
    device uint *surfaces [[buffer(7)]],
    device uint *coefficients [[buffer(8)]],
    device uint *first_order [[buffer(9)]],
    device uint *candidate_ids [[buffer(10)]],
    device uint *configuration_ids [[buffer(11)]],
    device uint *status_out [[buffer(12)]],
    device uint *model_data [[buffer(13)]],
    constant MklSystemsDispatch &p [[buffer(14)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    uint row = p.first + tid;
    // Instruction-major scratch coalesces each SIMD group's node accesses.
    device float2 *v = scratch;
    device const float2 *params = parameters + row * p.parameters;
    uint status = slots[row*4+3];
    for (uint j=0; j<p.parameters; ++j) {
        float2 a = params[j];
        if (!all(isfinite(a)) || !mkl_p4_le(bounds[2*j],a) || !mkl_p4_le(a,bounds[2*j+1]))
            status = MKL_SYSTEMS_OUTSIDE_BOUNDS;
    }
    for (uint j=p.first_instruction; j<p.first_instruction+p.instruction_count; ++j) {
        uint op=program[4*j], ia=program[4*j+1], ib=program[4*j+2], ic=program[4*j+3];
        if (op==0) { v[j*p.count+tid]=float2(as_type<float>(ia),as_type<float>(ib)); continue; }
        if (op==1) { v[j*p.count+tid]=params[ia]; continue; }
        float2 a=v[ia*p.count+tid], b=v[ib*p.count+tid], c=v[ic*p.count+tid], value;
        switch(op) {
            case 2: value=mkl_df_add(a,b); break;
            case 3: value=mkl_df_sub(a,b); break;
            case 4: value=mkl_df_mul(a,b); break;
            case 5: value=b.x==0.0f ? float2(a.x/b.x,0.0f) : mkl_df_div(a,b); break;
            case 6: value=mkl_df_sqrt(a); break;
            default: value=mkl_p4_le(a,float2(0.0f)) ? b : c; break;
        }
        v[j*p.count+tid]=value;
    }
    if (p.first_instruction+p.instruction_count<p.instructions) return;
    uint guard_start=4*(p.instructions+p.patches);
    for (uint j=0; j<p.guards; ++j) {
        float2 value=v[program[guard_start+j]*p.count+tid];
        if (status==MKL_SYSTEMS_READY && (!all(isfinite(value)) || mkl_p4_le(value,float2(0.0f))))
            status=MKL_SYSTEMS_INVALID_NUMERIC;
    }
    device uint *h=headers+row*p.header_words;
    device uint *s=surfaces+row*p.surface_words;
    device uint *c=coefficients+row*p.coefficient_words;
    device uint *f=first_order+row*p.first_order_words;
    device uint *m=model_data+row*p.model_words;
    for(uint j=0;j<p.header_words;++j) h[j]=base[j];
    for(uint j=0;j<p.surface_words;++j) s[j]=base[p.header_words+j];
    for(uint j=0;j<p.coefficient_words;++j) c[j]=base[p.header_words+p.surface_words+j];
    for(uint j=0;j<p.patches;++j) {
        uint off=4*(p.instructions+j), dest=program[off];
        uint hi=program[off+1],lo=program[off+2];float2 value=v[program[off+3]*p.count+tid];
        if(dest<3 && status==MKL_SYSTEMS_READY && !all(isfinite(value))) status=MKL_SYSTEMS_INVALID_NUMERIC;
        device uint *out=dest==0?h:dest==1?s:dest==2?c:dest==3?f:m;
        out[hi]=as_type<uint>(value.x);
        if(lo!=0xffffffffu) out[lo]=as_type<uint>(value.y);
    }
    for(uint j=0;j<p.chunks;++j)
        h[j*sizeof(MklPackedHeader)/4+__builtin_offsetof(MklPackedHeader,surface_offset)/4]+=row*p.chunks*p.surfaces;
    for(uint j=0;j<p.chunks*p.surfaces;++j)
        s[j*sizeof(MklPackedSurface)/4+__builtin_offsetof(MklPackedSurface,coefficient_offset)/4]+=row*p.coefficient_words/2;
    candidate_ids[row*2]=slots[row*4];candidate_ids[row*2+1]=slots[row*4+1];
    configuration_ids[row]=slots[row*4+2];
    status_out[row*2]=status;status_out[row*2+1]=MKL_SYSTEMS_DIRTY_ALL;
}
