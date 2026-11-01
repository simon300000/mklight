#include "mklight_common.h"
static float2 wdf(constant float *v) { return float2(v[0],v[1]); }
static float wf(float2 v) { return v.x+v.y; }
static float2 wcis(float2 v) {
    v=mkl_df_add_f32(v,-rint(v.x)); v=mkl_df_add_f32(v,-rint(v.x));
    float t=wf(v)*6.2831853071795864769f; return float2(cos(t),sin(t));
}
struct WVec {float2 x,y,z;};
static WVec wdir(constant MklWaveParams &p,uint k) {
    float2 u=mkl_df_add_f32(mkl_df_div(float2(float(2*(k%p.n)+1),0),float2(float(p.n),0)),-1);
    float2 v=mkl_df_add_f32(mkl_df_div(float2(float(2*(k/p.n)+1),0),float2(float(p.n),0)),-1);
    float2 a=mkl_df_mul(wdf(p.sin_max),u),b=mkl_df_mul(wdf(p.sin_max),v);
    float2 g=mkl_df_sqrt(mkl_df_sub(mkl_df_sub(float2(1,0),mkl_df_mul(a,a)),mkl_df_mul(b,b)));
    return {a,b,-g};
}
static float2 wdot(WVec a,WVec b) {return mkl_df_add(mkl_df_add(mkl_df_mul(a.x,b.x),mkl_df_mul(a.y,b.y)),mkl_df_mul(a.z,b.z));}
static float2 wfocus(constant MklWaveParams &p,uint k,float2 dz) {
    WVec q=wdir(p,k);float2 gamma=-q.z,r=wdf(p.radius);
    float2 s2=mkl_df_add(mkl_df_mul(q.x,q.x),mkl_df_mul(q.y,q.y));
    float2 dr;
    if(r.x==0) dr=mkl_df_div(mkl_df_mul(dz,s2),mkl_df_add_f32(gamma,1));
    else {
        float2 rd=mkl_df_mul_f32(mkl_df_mul(r,dz),2);
        float2 delta=mkl_df_add(mkl_df_mul(rd,gamma),mkl_df_mul(dz,dz));
        float2 path=mkl_df_sqrt(mkl_df_add(mkl_df_mul(r,r),delta));
        dr=mkl_df_div(mkl_df_mul(rd,s2),mkl_df_mul(mkl_df_add_f32(gamma,1),mkl_df_add(mkl_df_add(r,dz),path)));
    }
    return mkl_df_div(dr,wdf(p.wavelength));
}
kernel void mkl_wave_prepare(device const float *amp [[buffer(0)]], device const float *cov [[buffer(1)]], device const float2 *opd [[buffer(2)]], device float2 *field [[buffer(4)]], device float *mom [[buffer(5)]], constant MklWaveParams &p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=id+p.first,x=k%p.field_side,y=k/p.field_side;
    float2 z=0;
    if(x<p.n&&y<p.n) {
        uint j=y*p.n+x,N=p.n*p.n;float c=cov[j];float3 m=0;
        if(c>0) {
            WVec q=wdir(p,j);float gamma=-wf(q.z),a=amp[j];
            WVec normal={wdf(p.normal[0]),wdf(p.normal[1]),wdf(p.normal[2])};float nz=abs(wf(wdot(q,normal)));
            float w=c*a/gamma,e=c*(a/gamma)*(a/gamma);z=w*wcis(opd[j]);m=float3(w,e,nz>1e-9f?e*gamma/nz:mkl_qnan());
        }
        mom[j]=m.x;mom[N+j]=m.y;mom[2*N+j]=m.z;
    }
    field[k]=z;
}
kernel void mkl_wave_intensity(device const float2 *field [[buffer(0)]],device const float2 *s [[buffer(3)]],device float *out [[buffer(4)]],constant MklWaveParams &p [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=p.first+id,j=k;
    if(p.shifted) {int m=int(p.field_side),ix=(int(k%p.size)-int(p.size/2))%m,iy=(int(k/p.size)-int(p.size/2))%m;j=uint((iy+m)%m)*p.field_side+uint((ix+m)%m);}
    float2 z=field[j];float2 l=mkl_df_div(mkl_df_mul_f32(wdf(p.wavelength),float(p.n)),mkl_df_mul_f32(wdf(p.sin_max),2));
    float2 div=p.unit_energy?mkl_df_div(mkl_df_mul(mkl_df_mul(l,l),s[1]),float2(p.projection,0)):mkl_df_mul(s[0],s[0]);
    out[k]=wf(mkl_df_div(float2(z.x*z.x+z.y*z.y,0),div));
}
kernel void mkl_wave_focus(device const float2 *opd [[buffer(2)]],device const float2 *shifts [[buffer(3)]],device float2 *out [[buffer(4)]],constant MklWaveParams &p [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=id+p.first,N=p.n*p.n;out[k]=mkl_df_add(opd[k%N],wfocus(p,k%N,shifts[k/N]));
}
static float wsinc(float x) {float t=3.14159265358979323846f*x;return x==0?1:sin(t)/t;}
kernel void mkl_wave_detector(device const float2 *freq [[buffer(0)]],device float *out [[buffer(4)]],constant MklWaveParams &p [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=id+p.first;float2 v=freq[k];
    out[k]=abs(wsinc(p.aperture[0]*v.x)*wsinc(p.aperture[1]*v.y))*exp(-2*3.14159265358979323846f*3.14159265358979323846f*p.sigma*p.sigma*(v.x*v.x+v.y*v.y));
}
kernel void mkl_wave_lsf(device const float *image [[buffer(0)]],device float *out [[buffer(4)]],constant MklWaveParams &p [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=id+p.first;float2 s=0;
    for(uint j=0;j<p.n;++j)s=mkl_df_add_f32(s,image[k<p.n?j*p.n+k:(k-p.n)*p.n+j]);
    out[k]=wf(mkl_df_mul(s,wdf(p.spacing)));
}
// Each thread owns one image pixel. Sample tiles preserve the same addition
// order across command boundaries, without atomics or a pair matrix.
kernel void mkl_wave_huygens(device const float *amp [[buffer(0)]],device const float *cov [[buffer(1)]],device const float2 *opd [[buffer(2)]],device const float2 *s [[buffer(3)]],device float *out [[buffer(4)]],device float4 *acc [[buffer(6)]],constant MklWaveParams &p [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint pixel=p.first+id;
    float2 x=mkl_df_mul_f32(wdf(p.spacing),float(pixel%p.size)-float(p.size/2));
    float2 y=mkl_df_mul_f32(wdf(p.spacing),float(pixel/p.size)-float(p.size/2));
    WVec point;
    point.x=mkl_df_add(wdf(p.origin[0]),mkl_df_add(mkl_df_mul(wdf(p.x_axis[0]),x),mkl_df_mul(wdf(p.y_axis[0]),y)));
    point.y=mkl_df_add(wdf(p.origin[1]),mkl_df_add(mkl_df_mul(wdf(p.x_axis[1]),x),mkl_df_mul(wdf(p.y_axis[1]),y)));
    point.z=mkl_df_add(wdf(p.origin[2]),mkl_df_add(mkl_df_mul(wdf(p.x_axis[2]),x),mkl_df_mul(wdf(p.y_axis[2]),y)));
    float2 r=wdf(p.radius),xx=wdot(point,point);float4 z=p.sample_first?acc[id]:float4(0);
    for(uint k=p.sample_first;k<p.sample_first+p.sample_count;++k) {
        float c=cov[k];if(c<=0)continue;
        WVec q=wdir(p,k);float2 qx=wdot(q,point),phase,factor;
        if(r.x==0) {phase=mkl_df_div(qx,wdf(p.wavelength));factor=float2(1,0);}
        else {
            float2 rq=mkl_df_mul(r,qx),rr=mkl_df_mul(r,r),delta=mkl_df_sub(xx,mkl_df_mul_f32(rq,2));
            float2 dist=mkl_df_sqrt(mkl_df_add(rr,delta));
            phase=-mkl_df_div(mkl_df_div(delta,mkl_df_add(dist,r)),wdf(p.wavelength));
            factor=mkl_df_div(mkl_df_sub(rr,rq),mkl_df_mul(dist,dist));
        }
        float weight=c*amp[k]/(-wf(q.z))*wf(factor);float2 term=weight*wcis(mkl_df_add(opd[k],phase));
        z.xy=mkl_df_add_f32(z.xy,term.x);z.zw=mkl_df_add_f32(z.zw,term.y);
    }
    acc[id]=z;
    if(p.sample_first+p.sample_count==p.n*p.n) {
        float2 l=mkl_df_div(mkl_df_mul_f32(wdf(p.wavelength),float(p.n)),mkl_df_mul_f32(wdf(p.sin_max),2));
        float2 div=p.unit_energy?mkl_df_mul(mkl_df_mul(l,l),s[2]):mkl_df_mul(s[0],s[0]);
        out[pixel]=wf(mkl_df_div(mkl_df_add(mkl_df_mul(z.xy,z.xy),mkl_df_mul(z.zw,z.zw)),div));
    }
}
