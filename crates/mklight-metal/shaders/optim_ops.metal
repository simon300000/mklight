#include "mklight_common.h"
static inline float2 mkl_o_load(device const uint *v,uint i) {return as_type<float2>(uint2(v[2*i],v[2*i+1]));}
static inline bool mkl_o_negative(float2 v) {return v.x<0.0f || (v.x==0.0f && v.y<0.0f);}
kernel void mkl_optim_algebra(device const uint *a [[buffer(0)]],device const uint *b [[buffer(1)]],
    device const uint *c [[buffer(2)]],device const uint *d [[buffer(3)]],device float2 *out [[buffer(4)]],
    device uint *extra [[buffer(5)]],constant MklOptimParams &p [[buffer(6)]],uint tid [[thread_position_in_grid]]) {
    if(tid>=p.count)return;
    uint i=tid+p.offset,m=p.rows,n=p.columns,k=p.inner;
    float2 v=float2(0.0f);
    if(p.kind==0u) {
        uint col=i%m,cmp=c[col];float2 raw=mkl_o_load(a,i),z=mkl_df_sub(raw,mkl_o_load(b,col));
        v=cmp==3u?raw:(((cmp==1u && mkl_o_negative(z))||(cmp==2u && mkl_o_negative(-z)))?float2(0.0f):z);
    } else if(p.kind==1u) {
        uint col=i%n,row=i/n;uint3 ids=uint3(b[3*col],b[3*col+1],b[3*col+2]);
        bool bad=any(ids>=p.batches);if(row==0u)extra[col]=bad?1u:0u;
        if(bad) v=float2(mkl_qnan(),0.0f);
        else {float2 base=mkl_o_load(a,ids.x*m+row);
            v=mkl_df_add(mkl_df_mul(mkl_df_sub(mkl_o_load(a,ids.y*m+row),base),mkl_o_load(c,2*col)),
                mkl_df_mul(mkl_df_sub(mkl_o_load(a,ids.z*m+row),base),mkl_o_load(c,2*col+1)));}
    } else if(p.kind==2u) {
        uint batch=i/(m*n),row=(i/n)%m,col=i%n;
        for(uint j=0;j<k;++j) {uint ai=batch*m*k+(p.flag==1u?j*m+row:row*k+j);
            v=mkl_df_add(v,mkl_df_mul(mkl_o_load(a,ai),mkl_o_load(b,(batch*k+j)*n+col)));}
    } else if(p.kind==3u) {v=mkl_df_add(mkl_o_load(a,p.flag==1u?i%m:i),mkl_o_load(b,i));}
    else if(p.kind==4u) {
        uint batch=i/((m+n)*n),row=(i/n)%(m+n),col=i%n;
        float2 rhs=float2(0.0f);
        if(row<m) {float2 w=mkl_df_sqrt(mkl_o_load(c,row));v=mkl_df_mul(w,mkl_o_load(a,(batch*m+row)*n+col));rhs=-mkl_df_mul(w,mkl_o_load(b,batch*m+row));}
        else if(row-m==col) v=mkl_df_mul(mkl_df_sqrt(p.damping),mkl_o_load(d,batch*n+col));
        if(col==0u) {uint at=2*(batch*(m+n)+row);extra[at]=as_type<uint>(rhs.x);extra[at+1]=as_type<uint>(rhs.y);}
    }
    out[i]=v;
}
