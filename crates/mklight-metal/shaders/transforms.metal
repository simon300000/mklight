#include "mklight_common.h"
static float2 p2_df(constant float *v){return float2(v[0],v[1]);}
static float2 p2_cmul(float2 a,float2 b) {return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
static float2 p2_cis(float2 a) {
    a=mkl_df_add_f32(a,-rint(a.x));
    a=mkl_df_add_f32(a,-rint(a.x));
    float t=(a.x+a.y)*6.2831853071795864769f;
    return float2(cos(t),sin(t));
}
static float2 p2_q(constant MklTransformParams &p,float x,float y) {
    return mkl_df_mul_f32(mkl_df_add(mkl_df_add(mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[0]),x),x),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[1]),2*x),y)),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[4]),y),y)),0.5f);
}
kernel void mkl_transform(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],device float2 *out [[buffer(2)]],constant MklTransformParams &p [[buffer(3)]],uint tid [[thread_position_in_grid]]) {
    uint id=tid+p.start;if(id>=p.count)return;
    uint w=p.width,h=p.height;float scale=as_type<float>(p.scale_bits);float sign=float(as_type<int>(p.sign_bits));float2 z=0;
    switch(p.mode) {
    case MKL_TRANSFORM_COPY:z=a[id]*scale;break;
    case MKL_TRANSFORM_TRANSPOSE:z=a[(id%h)*w+id/h];break;
    case MKL_TRANSFORM_TWIDDLE:z=p2_cis(mkl_df_div(float2(sign*float(id),0),float2(float(w),0)));break;
    case MKL_TRANSFORM_FFT:{
        uint k=id%w,row=id/w,r=p.radix,s=p.span,i=k%s,j=(k/s)%r,g=k/(s*r);float2 re=0,im=0;
        for(uint t=0;t<r;++t){float2 v=p2_cmul(a[row*w+g*s+i+t*w/r],b[(t*(i+j*s)*(w/(r*s)))%w]);re=mkl_df_add_f32(re,v.x);im=mkl_df_add_f32(im,v.y);}
        z=float2(re.x+re.y,im.x+im.y);break;
    }
    case MKL_TRANSFORM_KERNEL:{
        uint ix=id%w,iy=id/w;
        if((ix<p.ox||ix>w-p.nx)&&(iy<p.oy||iy>h-p.ny)){
            float x=ix<p.ox?float(ix):float(ix)-float(w),y=iy<p.oy?float(iy):float(iy)-float(h);
            z=p2_cis(mkl_df_mul_f32(p2_q(p,x,y),-sign));
        }break;
    }
    case MKL_TRANSFORM_PRE:{uint x=id%w,y=id/w;if(x<p.nx&&y<p.ny){
        float2 ph=mkl_df_add(p2_q(p,float(x),float(y)),mkl_df_add(mkl_df_mul_f32(p2_df(p.affine[2]),float(x)),mkl_df_mul_f32(p2_df(p.affine[5]),float(y))));
        z=p2_cmul(a[y*p.nx+x],p2_cis(mkl_df_mul_f32(ph,sign)));}break;}
    case MKL_TRANSFORM_POST:{uint x=id%p.ox,y=id/p.ox;z=p2_cmul(a[y*w+x],p2_cis(mkl_df_mul_f32(p2_q(p,float(x),float(y)),sign)))*scale;break;}
    case MKL_TRANSFORM_MULTIPLY:{float2 v=b[p.reserved0?id%w:id];if(p.conjugate)v.y=-v.y;z=p2_cmul(a[id],v)*scale;break;}
    case MKL_TRANSFORM_PHASE:{float x=float(id%w),y=float(id/w);float2 ph=mkl_df_add(mkl_df_add(mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[0]),x),x),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[1]),x),y)),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[2]),y),y));
        ph=mkl_df_add(ph,mkl_df_add(mkl_df_add(mkl_df_mul_f32(p2_df(p.affine[3]),x),mkl_df_mul_f32(p2_df(p.affine[4]),y)),p2_df(p.affine[5])));z=p2_cmul(a[id],p2_cis(ph));break;}
    }
    out[id]=z;
}
static float2 p2_dot(thread float2 *a,constant float b[3][2]){return mkl_df_add(mkl_df_add(mkl_df_mul(a[0],p2_df(b[0])),mkl_df_mul(a[1],p2_df(b[1]))),mkl_df_mul(a[2],p2_df(b[2])));}
struct P2Geometry {float x,y,z,gamma,source_z;};
static float2 p2_gamma(constant MklCorrelationParams&p,uint k){
    float2 a=mkl_df_mul_f32(p2_df(p.sin_max),float(2*(k%p.n)+1)/float(p.n)-1.0f);
    float2 b=mkl_df_mul_f32(p2_df(p.sin_max),float(2*(k/p.n)+1)/float(p.n)-1.0f);
    return mkl_df_sqrt(mkl_df_sub(mkl_df_sub(float2(1,0),mkl_df_mul(a,a)),mkl_df_mul(b,b)));
}
static P2Geometry p2_geometry(constant MklCorrelationParams&p,uint k,float2 nu){
    float2 a=mkl_df_mul_f32(p2_df(p.sin_max),float(2*(k%p.n)+1)/float(p.n)-1.0f);
    float2 b=mkl_df_mul_f32(p2_df(p.sin_max),float(2*(k/p.n)+1)/float(p.n)-1.0f);
    float2 gamma=p2_gamma(p,k),q[3];
    for(uint i=0;i<3;i++)q[i]=mkl_df_sub(mkl_df_add(mkl_df_mul(p2_df(p.frame[0][i]),a),mkl_df_mul(p2_df(p.frame[1][i]),b)),mkl_df_mul(p2_df(p.frame[2][i]),gamma));
    float2 sx=mkl_df_add(q[0],mkl_df_mul_f32(p2_df(p.wavelength),nu.x)),sy=mkl_df_add(q[1],mkl_df_mul_f32(p2_df(p.wavelength),nu.y));
    float2 z2=mkl_df_sub(mkl_df_sub(float2(1,0),mkl_df_mul(sx,sx)),mkl_df_mul(sy,sy));
    P2Geometry g={mkl_qnan(),mkl_qnan(),0,gamma.x,-q[2].x};
    if(z2.x<=0||q[2].x>=0||!isfinite(gamma.x))return g;
    float2 sz=-mkl_df_sqrt(z2),shifted[3]={sx,sy,sz};
    if(p2_dot(shifted,p.frame[2]).x>=0)return g;
    float2 scale=mkl_df_div(float2(0.5f*float(p.n),0),p2_df(p.sin_max));float offset=0.5f*float(p.n-1);
    float2 ix=mkl_df_add_f32(mkl_df_mul(p2_dot(shifted,p.frame[0]),scale),offset),iy=mkl_df_add_f32(mkl_df_mul(p2_dot(shifted,p.frame[1]),scale),offset);
    g.x=ix.x+ix.y;g.y=iy.x+iy.y;g.z=-sz.x;return g;
}
kernel void mkl_optical_stencil(device const float*freq[[buffer(0)]],device float*dst[[buffer(1)]],constant MklCorrelationParams&p[[buffer(2)]],uint tid[[thread_position_in_grid]]){
    uint k=tid+p.start,count=p.n*p.height;if(k>=count)return;
    P2Geometry g=p2_geometry(p,k,float2(freq[2*p.frequency],freq[2*p.frequency+1]));uint o=(p.frequency*count+k)*5;
    dst[o]=g.x;dst[o+1]=g.y;dst[o+2]=g.z;dst[o+3]=g.gamma;dst[o+4]=g.source_z;
}
kernel void mkl_correlation_terms(device const float*field[[buffer(0)]],device const float2*opd[[buffer(1)]],device const float*cov[[buffer(2)]],device const float*freq[[buffer(3)]],device const float*stencil[[buffer(4)]],device float*out[[buffer(5)]],constant MklCorrelationParams&p[[buffer(6)]],uint tid[[thread_position_in_grid]]){
    uint k=tid+p.start,count=p.n*p.height;if(k>=count)return;out[k]=0;out[count+k]=0;out[2*count+k]=0;
    float c=p.covered?cov[k]:1.0f;if(c<=0)return;float2 nu=float2(freq[2*p.frequency],freq[2*p.frequency+1]);P2Geometry g;
    if(p.optical){if(p.prepared){uint o=(p.frequency*count+k)*5;g={stencil[o],stencil[o+1],stencil[o+2],stencil[o+3],stencil[o+4]};}else g=p2_geometry(p,k,nu);}
    else{
        float2 sx=mkl_df_add(mkl_df_add(mkl_df_mul_f32(p2_df(p.mapping[0]),nu.x),mkl_df_mul_f32(p2_df(p.mapping[1]),nu.y)),p2_df(p.mapping[2]));
        float2 sy=mkl_df_add(mkl_df_add(mkl_df_mul_f32(p2_df(p.mapping[3]),nu.x),mkl_df_mul_f32(p2_df(p.mapping[4]),nu.y)),p2_df(p.mapping[5]));
        g={float(k%p.n)+sx.x+sx.y,float(k/p.n)+sy.x+sy.y,1,1,1};
    }
    float power=p.optical?field[k]*field[k]:field[2*k]*field[2*k]+field[2*k+1]*field[2*k+1];float energy=c*power/(g.gamma*g.source_z);out[2*count+k]=energy;
    if(!isfinite(g.x)||!isfinite(g.y))return;
    float fx=floor(g.x),fy=floor(g.y),tx=g.x-fx,ty=g.y-fy;
    float ar=0,ai=0,coverage=0;float2 phase=0;
    for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){
        int xi=int(fx)+dx,yi=int(fy)+dy;if(xi<0||yi<0||xi>=int(p.n)||yi>=int(p.height))continue;
        uint j=uint(yi)*p.n+uint(xi);float w=(dx?tx:1-tx)*(dy?ty:1-ty)*(p.covered?cov[j]:1.0f);if(w==0)continue;
        if(p.optical){ar+=w*field[j];coverage+=w;phase=mkl_df_add(phase,mkl_df_mul_f32(opd[j],w));}
        else{ar+=w*field[2*j];ai+=w*field[2*j+1];}
    }
    float deficit=(1-c)*energy*max(0.0f,1-abs(g.x-float(k%p.n)))*max(0.0f,1-abs(g.y-float(k/p.n)));float2 v=0;
    if(p.optical){if(coverage>0){float2 z=p2_cis(mkl_df_sub(mkl_df_div(phase,float2(coverage,0)),opd[k]));v=(c*field[k]/g.gamma*ar/g.z)*z;}}
    else v=c*float2(ar*field[2*k]+ai*field[2*k+1],ai*field[2*k]-ar*field[2*k+1]);
    out[k]=v.x+deficit;out[count+k]=v.y;
}
kernel void mkl_correlation_normalize(device const float2*sums[[buffer(0)]],device float2*out[[buffer(1)]],uint tid[[thread_position_in_grid]]){
    if(tid)return;float2 a=mkl_df_div(sums[0],sums[2]),b=mkl_df_div(sums[1],sums[2]);out[0]=float2(a.x+a.y,b.x+b.y);
}

// 16x16 tiles, padded shared rows avoid bank conflicts during the swap.
kernel void mkl_transform_transpose(device const float2*a[[buffer(0)]],device float2*out[[buffer(2)]],constant MklTransformParams&p[[buffer(3)]],uint tid[[thread_index_in_threadgroup]],uint group[[threadgroup_position_in_grid]]){
    threadgroup float2 tile[16][17];uint nx=(p.width+15)/16,t=group+p.start;
    uint bx=(t%nx)*16,by=(t/nx)*16,tx=tid%16,ty=tid/16;
    if(bx+tx<p.width&&by+ty<p.height)tile[ty][tx]=a[(by+ty)*p.width+bx+tx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(bx+ty<p.width&&by+tx<p.height)out[(bx+ty)*p.height+by+tx]=tile[tx][ty];
}
// One independent row per group. A complete stage is read into registers
// before writing shared storage; the barriers preserve the Stockham ordering.
static void p2_local_rows(device const float2*a,device const float2*tw,device float2*out,constant MklTransformParams&p,threadgroup float2*data,uint tid,uint threads,uint group){
    uint w=p.width,row=group+p.start;
    for(uint k=tid;k<w;k+=threads)data[k]=a[row*w+k];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint span=1,remaining=w;const uint radices[4]={2,3,5,7};
    for(uint ri=0;ri<4;ri++){
        uint r=radices[ri];
        while(remaining%r==0){
            float2 values[16];
            for(uint k=tid,q=0;k<w;k+=threads,q++){
                uint i=k%span,j=(k/span)%r,g=k/(span*r);
                float2 sum=data[g*span+i];
                for(uint t=1;t<r;t++)sum+=p2_cmul(data[g*span+i+t*w/r],tw[(t*(i+j*span)*(w/(r*span)))%w]);
                values[q]=sum;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for(uint k=tid,q=0;k<w;k+=threads,q++)data[k]=values[q];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            span*=r;remaining/=r;
        }
    }
    for(uint k=tid;k<w;k+=threads)out[row*w+k]=data[k];
}
kernel void mkl_transform_rows_small(device const float2*a[[buffer(0)]],device const float2*tw[[buffer(1)]],device float2*out[[buffer(2)]],constant MklTransformParams&p[[buffer(3)]],uint tid[[thread_index_in_threadgroup]],uint size[[threads_per_threadgroup]],uint group[[threadgroup_position_in_grid]]){
    threadgroup float2 data[1024];p2_local_rows(a,tw,out,p,data,tid,size,group);
}
kernel void mkl_transform_rows_large(device const float2*a[[buffer(0)]],device const float2*tw[[buffer(1)]],device float2*out[[buffer(2)]],constant MklTransformParams&p[[buffer(3)]],uint tid[[thread_index_in_threadgroup]],uint size[[threads_per_threadgroup]],uint group[[threadgroup_position_in_grid]]){
    threadgroup float2 data[4096];p2_local_rows(a,tw,out,p,data,tid,size,group);
}
