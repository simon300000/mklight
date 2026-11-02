#include "mklight_common.h"

static inline bool mkl_p4_le(float2 a, float2 b) {
    return a.x < b.x || (a.x == b.x && mkl_df_low(a.y) <= mkl_df_low(b.y));
}
// One thread per candidate evaluates the straight-line program; scratch is
// instruction-major so a SIMD group's node accesses coalesce. The last
// instruction tile also writes the candidate map.
kernel void mkl_systems_prepare(
    device const float2 *parameters [[buffer(0)]],
    device const uint *slots [[buffer(1)]],
    device const float2 *bounds [[buffer(2)]],
    device const uint *program [[buffer(3)]],
    device float2 *v [[buffer(5)]],
    device uint *candidate_ids [[buffer(10)]],
    device uint *configuration_ids [[buffer(11)]],
    device uint *status_out [[buffer(12)]],
    constant MklSystemsDispatch &p [[buffer(14)]],
    uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    uint row = p.first + tid;
    device const float2 *params = parameters + row * p.parameters;
    for (uint j=p.first_instruction; j<p.first_instruction+p.instruction_count; ++j) {
        uint op=program[4*j], ia=program[4*j+1], ib=program[4*j+2], ic=program[4*j+3];
        if (op==0) { v[j*p.count+tid]=mkl_df_input(float2(as_type<float>(ia),as_type<float>(ib))); continue; }
        if (op==1) { v[j*p.count+tid]=mkl_df_input(params[ia]); continue; }
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
    uint status = slots[row*4+3];
    for (uint j=0; j<p.parameters; ++j) {
        float2 a = params[j];
        if (!all(isfinite(a)) || !mkl_p4_le(bounds[2*j],a) || !mkl_p4_le(a,bounds[2*j+1]))
            status = MKL_SYSTEMS_OUTSIDE_BOUNDS;
    }
    if (status==MKL_SYSTEMS_READY) {
        uint guard_start=4*(p.instructions+p.patches);
        for (uint j=0; j<p.guards; ++j) {
            float2 value=v[program[guard_start+j]*p.count+tid];
            if (!all(isfinite(value)) || mkl_p4_le(value,float2(0.0f))) status=MKL_SYSTEMS_INVALID_NUMERIC;
        }
        for (uint j=0; j<p.patches; ++j) {
            uint off=4*(p.instructions+j);
            if (program[off]<3 && !all(isfinite(v[program[off+3]*p.count+tid]))) status=MKL_SYSTEMS_INVALID_NUMERIC;
        }
    }
    candidate_ids[row*2]=slots[row*4];candidate_ids[row*2+1]=slots[row*4+1];
    configuration_ids[row]=slots[row*4+2];
    status_out[row*2]=status;status_out[row*2+1]=MKL_SYSTEMS_DIRTY_ALL;
}
// One thread per template word: copy and rebase the packed offsets.
kernel void mkl_systems_copy(
    device const uint *base [[buffer(4)]],
    device uint *headers [[buffer(6)]],
    device uint *surfaces [[buffer(7)]],
    device uint *coefficients [[buffer(8)]],
    constant MklSystemsDispatch &p [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
    uint words=p.header_words+p.surface_words+p.coefficient_words;
    if (gid >= p.count*words) return;
    uint row=p.first+gid/words, j=gid%words, value=base[j];
    if (j<p.header_words) {
        if (j%(sizeof(MklPackedHeader)/4)==__builtin_offsetof(MklPackedHeader,surface_offset)/4)
            value+=row*p.chunks*p.surfaces;
        headers[row*p.header_words+j]=value;
    } else if ((j-=p.header_words)<p.surface_words) {
        if (j%(sizeof(MklPackedSurface)/4)==__builtin_offsetof(MklPackedSurface,coefficient_offset)/4)
            value+=row*p.coefficient_words/2;
        surfaces[row*p.surface_words+j]=value;
    } else {
        j-=p.surface_words;
        coefficients[row*p.coefficient_words+j]=value;
    }
}
// One thread per (candidate, patch); destinations are distinct words.
kernel void mkl_systems_patch(
    device const uint *program [[buffer(3)]],
    device const float2 *v [[buffer(5)]],
    device uint *headers [[buffer(6)]],
    device uint *surfaces [[buffer(7)]],
    device uint *coefficients [[buffer(8)]],
    device uint *first_order [[buffer(9)]],
    device uint *model_data [[buffer(13)]],
    constant MklSystemsDispatch &p [[buffer(14)]],
    uint gid [[thread_position_in_grid]]) {
    if (gid >= p.count*p.patches) return;
    uint tid=gid%p.count, row=p.first+tid, off=4*(p.instructions+gid/p.count);
    uint dest=program[off], hi=program[off+1], lo=program[off+2];
    float2 value=v[program[off+3]*p.count+tid];
    device uint *out=dest==0?headers+row*p.header_words:dest==1?surfaces+row*p.surface_words:
        dest==2?coefficients+row*p.coefficient_words:dest==3?first_order+row*p.first_order_words:
        model_data+row*p.model_words;
    out[hi]=as_type<uint>(value.x);
    if(lo!=0xffffffffu) out[lo]=as_type<uint>(value.y);
}
