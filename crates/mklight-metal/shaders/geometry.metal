// Resident geometric algebra. Strict math and the shared trace/sag hooks.
#include "trace_shared.h"
struct MklGeometryParams {
    uint count, offset, stride, chiefs;
    uint segment, centroid, systems, reserved;
    float2 defocus;
};
static inline MklDf geo_d(device const float2* a, ulong i) { return {a[i]}; }
static inline MklDf geo_c(float x) { return mkl_c<MklDf>(x); }
static inline MklDf geo_hit(device const float* a, uint n, uint i, uint k) {
    return mkl_parts<MklDf>(a[ulong(k)*n+i],a[ulong(k+9)*n+i]);
}
kernel void mkl_geometry_coordinates(device const float* rays [[buffer(0)]], device const uint* status [[buffer(1)]],
    device const float* chiefs [[buffer(2)]], device const uint* chief_status [[buffer(3)]], device const uint* index [[buffer(4)]],
    device float2* out [[buffer(5)]], device uint* mask [[buffer(6)]], constant MklGeometryParams& p [[buffer(7)]], uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid; if(tid>=p.count) return;
    uint j=index[i]; bool ok=status[i]==0, cok=j<p.chiefs && chief_status[j]==0;
    mask[i]=uint(ok);
    MklDf dz=geo_hit(rays,p.stride,i,5), cdz=cok?geo_hit(chiefs,p.chiefs,j,5):geo_c(1);
    for(uint k=0;k<2;++k) {
        MklDf c=cok?geo_hit(chiefs,p.chiefs,j,k):geo_c(0);
        MklDf ct=cok?geo_hit(chiefs,p.chiefs,j,k+3)/cdz:geo_c(0);
        MklDf dt=geo_hit(rays,p.stride,i,k+3)/dz-ct;
        out[ulong(k)*p.stride+i]=ok?((geo_hit(rays,p.stride,i,k)-c)+MklDf{p.defocus}*dt).v:float2(0);
        out[ulong(k+2)*p.stride+i]=ok?dt.v:float2(0);
    }
    out[4ul*p.stride+i]=float2(ok?rays[8ul*p.stride+i]:0,0);
}
kernel void mkl_geometry_radial(device const float2* points [[buffer(0)]], device const uint* mask [[buffer(1)]],
    device const float2* moments [[buffer(2)]], device float2* radius [[buffer(3)]], device float2* square [[buffer(4)]],
    device float2* weight [[buffer(5)]], device uint* ids [[buffer(6)]], constant MklGeometryParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid; if(tid>=p.count) return; ids[i]=i;
    radius[i]=float2(INFINITY,0); square[i]=float2(INFINITY,0);weight[i]=float2(0);
    if(mask[i]==0) return;
    ulong j=ulong(i/p.segment)*10;
    MklDf x=geo_d(points,i)-(p.centroid?geo_d(moments,j+2):geo_c(0));
    MklDf y=geo_d(points,ulong(p.stride)+i)-(p.centroid?geo_d(moments,j+3):geo_c(0));
    radius[i]=mkl_sqrt(x*x+y*y).v;
    MklDf ax=x.v.x<0?-x:x,ay=y.v.x<0?-y:y;
    square[i]=mkl_le(ax-ay,0)?ay.v:ax.v;
    weight[i]=points[4ul*p.stride+i];
}
kernel void mkl_geometry_characteristic(device const float2* points [[buffer(0)]], device const uint* mask [[buffer(1)]],
    device const float2* frequency [[buffer(2)]], device float2* out [[buffer(3)]],constant MklGeometryParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid; if(tid>=p.count) return;
    out[i]=float2(0);out[ulong(p.stride)+i]=float2(0);if(mask[i]==0) return;
    ulong j=2ul*(i/p.segment);
    MklDf a=geo_d(points,i)*geo_d(frequency,j)+geo_d(points,ulong(p.stride)+i)*geo_d(frequency,j+1);
    a=a-geo_c(round(a.v.x));
    float phase=(a*mkl_parts<MklDf>(6.2831854820251465f,-1.7484555314695172e-7f)).v.x;
    MklDf w=geo_d(points,4ul*p.stride+i);
    out[i]=mkl_df_mul_f32(w.v,cos(phase));out[ulong(p.stride)+i]=mkl_df_mul_f32(w.v,sin(phase));
}
kernel void mkl_geometry_surfaces(constant MklPackedHeader* headers [[buffer(0)]], constant MklPackedSurface* surfaces [[buffer(1)]],
    constant float* pool [[buffer(2)]],device const uint* samples [[buffer(3)]], device float2* out [[buffer(4)]],device uint* status [[buffer(5)]],
    constant MklGeometryParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid;if(tid>=p.count) return;
    ulong j=8ul*i;uint sys=samples[j],surf=samples[j+1];status[i]=0;
    for(uint k=0;k<3;++k) out[3ul*i+k]=float2(NAN,0);
    if(sys>=p.systems || surf>=headers[sys].num_surfaces){status[i]=2;return;}
    constant MklPackedSurface& s=surfaces[headers[sys].surface_offset+surf];
    MklDf x=mkl_parts<MklDf>(as_type<float>(samples[j+4]),as_type<float>(samples[j+5]));
    MklDf y=mkl_parts<MklDf>(as_type<float>(samples[j+6]),as_type<float>(samples[j+7]));
    uint shape=s.shape==MKL_SHAPE_DIFFRACTIVE?uint(mkl_surface_data(s,pool)[2*MKL_COEFF_KIND]):s.shape;
    MklJet<MklDf> v;
    bool ok=(s.flags&MKL_SURF_HAS_SAG_OVERLAY)!=0?mkl_surface_eval<true>(s,pool,x,y,v):
            (shape<=MKL_SHAPE_COORD_BREAK?mkl_legacy_sag(s,x,y,v):mkl_surface_eval<false>(s,pool,x,y,v));
    if(!ok){status[i]=1;return;}
    out[3ul*i]=v.v.v;out[3ul*i+1]=v.x.v;out[3ul*i+2]=v.y.v;
}
kernel void mkl_geometry_distribution(device const float2* keys [[buffer(0)]],device const float2* cum [[buffer(1)]],
    device const float2* radii [[buffer(2)]],device float2* out [[buffer(3)]],constant MklGeometryParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid;if(tid>=p.count) return;
    uint n=p.segment,k=p.chiefs;MklDf total=n>0?geo_d(cum,n-1):geo_c(0);
    if(i==k+3){out[i]=total.v;return;}
    uint a=0,b=n;
    if(i<k){MklDf radius=geo_d(radii,i);while(a<b){uint mid=(a+b)/2;if(mkl_le(geo_d(keys,mid)-radius,0))a=mid+1;else b=mid;}
        out[i]=a>0 && total.v.x>0?(geo_d(cum,a-1)/total).v:float2(0);
    }else{MklDf fraction=i==k?mkl_parts<MklDf>(0.5f,-4.996003610813204e-16f):i==k+1?mkl_parts<MklDf>(0.8f,-1.192092980328851e-8f):mkl_parts<MklDf>(0.9f,2.38418573772492e-8f);
        MklDf target=total*fraction;while(a<b){uint mid=(a+b)/2;if((geo_d(cum,mid)-target).v.x<0)a=mid+1;else b=mid;}
        out[i]=a<n && total.v.x>0?keys[a]:float2(NAN,0);
    }
}
