// Resident geometric algebra. Strict math and the shared trace/sag hooks.
#include "trace_shared.h"
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
        out[ulong(k)*p.stride+i]=ok?((geo_hit(rays,p.stride,i,k)-c)+mkl_parts<MklDf>(p.defocus[0],p.defocus[1])*dt).v:float2(0);
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
    uint ray=i%p.stride,frequency_id=i/p.stride;
    out[i]=float2(0);out[ulong(p.systems)+i]=float2(0);if(mask[ray]==0) return;
    ulong j=2ul*(ulong(frequency_id)*(p.stride/p.segment)+ray/p.segment);
    MklDf a=geo_d(points,ray)*geo_d(frequency,j)+geo_d(points,ulong(p.stride)+ray)*geo_d(frequency,j+1);
    a=a-geo_c(round(a.v.x));
    float phase=(a*mkl_parts<MklDf>(6.2831854820251465f,-1.7484555314695172e-7f)).v.x;
    MklDf w=geo_d(points,4ul*p.stride+ray);
    out[i]=mkl_df_mul_f32(w.v,cos(phase));out[ulong(p.systems)+i]=mkl_df_mul_f32(w.v,sin(phase));
}
kernel void mkl_geometry_surfaces(constant MklPackedHeader* headers [[buffer(0)]], constant MklPackedSurface* surfaces [[buffer(1)]],
    constant float* pool [[buffer(2)]],device const uint* samples [[buffer(3)]], device float2* out [[buffer(4)]],device uint* status [[buffer(5)]],
    constant MklGeometryParams& p [[buffer(7)]],device const float2* frames [[buffer(6)]],device const float2* xy [[buffer(8)]],uint tid [[thread_position_in_grid]]) {
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
    if(p.centroid!=0){uint frame=headers[sys].surface_offset+surf;if(frame>=p.segment){status[i]=2;for(uint k=0;k<3;++k)out[3ul*i+k]=float2(NAN,0);return;}
        MklDv point;point.v[0]=xy[2ul*i];point.v[1]=xy[2ul*i+1];point.v[2]=v.v.v;
        for(uint k=0;k<3;++k){MklDv rot;for(uint a=0;a<3;++a)rot.v[a]=frames[12ul*frame+3*k+a];out[3ul*i+k]=((geo_d(frames,12ul*frame+9+k)-mkl_parts<MklDf>(p.origin[2*k],p.origin[2*k+1]))+MklDf{mkl_ddot(rot,point)}).v;}
    }

}
kernel void mkl_geometry_distribution(device const float2* keys [[buffer(0)]],device const float2* cum [[buffer(1)]],
    device const float2* radii [[buffer(2)]],device float2* out [[buffer(3)]],device float2* boundary [[buffer(4)]],constant MklGeometryParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid;if(tid>=p.count) return;
    uint n=p.segment,k=p.chiefs;MklDf total=n>0?geo_d(cum,n-1):geo_c(0);
    if(i==k+3){out[i]=total.v;return;}
    uint a=0,b=n;
    if(i<k){MklDf radius=geo_d(radii,i);while(a<b){uint mid=(a+b)/2;if(mkl_le(geo_d(keys,mid)-radius,0))a=mid+1;else b=mid;}
        boundary[i]=float2(0);
        if(total.v.x>0){MklDf radius=geo_d(radii,i),error=mkl_parts<MklDf>(p.defocus[0],p.defocus[1]),low=radius-error,high=radius+error;uint la=0,lb=n,ha=0,hb=n;
            while(la<lb){uint mid=(la+lb)/2;if((geo_d(keys,mid)-low).v.x<0)la=mid+1;else lb=mid;}
            while(ha<hb){uint mid=(ha+hb)/2;if((geo_d(keys,mid)-high).v.x<=0)ha=mid+1;else hb=mid;}
            MklDf lower=la>0?geo_d(cum,la-1):geo_c(0),upper=ha>0?geo_d(cum,ha-1):geo_c(0);boundary[i]=((upper-lower)/total).v;
        }
        out[i]=a>0 && total.v.x>0?(geo_d(cum,a-1)/total).v:float2(0);
    }else{MklDf fraction=i==k?mkl_parts<MklDf>(0.5f,-4.996003610813204e-16f):i==k+1?mkl_parts<MklDf>(0.8f,-1.192092980328851e-8f):mkl_parts<MklDf>(0.9f,2.38418573772492e-8f);
        MklDf target=total*fraction;while(a<b){uint mid=(a+b)/2;if((geo_d(cum,mid)-target).v.x<0)a=mid+1;else b=mid;}
        out[i]=a<n && total.v.x>0?keys[a]:float2(NAN,0);
    }
}
kernel void mkl_geometry_fields(device const float* f [[buffer(0)]],device const uint* status [[buffer(1)]],
    device const float* hi [[buffer(2)]],device const float* lo [[buffer(3)]],device float2* out [[buffer(4)]],
    constant MklGeometryParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid;if(tid>=p.count)return;uint n=p.chiefs;
    for(uint k=0;k<5;++k)out[5ul*i+k]=float2(NAN,0);
    if(p.centroid==2){ulong a=ulong(MKL_AIM_RECORD_FLOATS)*i;
        if(status[i]==0){out[5ul*i]=geo_hit(f,n,i,0).v;out[5ul*i+1]=geo_hit(f,n,i,1).v;}if(as_type<uint>(hi[a+12])!=0)return;
        for(uint k=0;k<2;++k){uint j=p.defocus[0]==0?3+k:k;MklDf v=mkl_parts<MklDf>(hi[a+j],lo[a+j]);out[5ul*i+2+k]=p.defocus[0]==0?(v/mkl_parts<MklDf>(hi[a+5],lo[a+5])).v:v.v;}out[5ul*i+4]=float2(0);
    }else if(p.centroid==0){if(status[i]!=0)return;
        ulong a=(ulong(p.segment)*n+i)*MKL_AIM_RECORD_FLOATS;MklDv normal,d;
        for(uint k=0;k<3;++k){normal.v[k]=float2(hi[a+8+k],lo[a+8+k]);d.v[k]=geo_hit(f,n,i,k+3).v;}
        MklDf dot={mkl_ddot(d,normal)};MklDv cross;cross.v[0]=(MklDf{d.v[1]}*MklDf{normal.v[2]}-MklDf{d.v[2]}*MklDf{normal.v[1]}).v;cross.v[1]=(MklDf{d.v[2]}*MklDf{normal.v[0]}-MklDf{d.v[0]}*MklDf{normal.v[2]}).v;cross.v[2]=(MklDf{d.v[0]}*MklDf{normal.v[1]}-MklDf{d.v[1]}*MklDf{normal.v[0]}).v;float sn=mkl_sqrt(MklDf{mkl_ddot(cross,cross)}).v.x;
        out[5ul*i]=geo_hit(f,n,i,0).v;out[5ul*i+1]=geo_hit(f,n,i,1).v;
        out[5ul*i+2]=float2(atan2(sn,fabs(dot.v.x))*(180.0f/M_PI_F),0);
        out[5ul*i+3]=float2(atan2(d.v[0].x,fabs(d.v[2].x))*(180.0f/M_PI_F),0);
        out[5ul*i+4]=float2(atan2(d.v[1].x,fabs(d.v[2].x))*(180.0f/M_PI_F),0);
    }else{uint a=5*i;for(uint k=0;k<5;++k)if(status[a+k]!=0)return;
        out[5ul*i]=geo_hit(f,n,a,0).v;out[5ul*i+1]=geo_hit(f,n,a,1).v;
        for(uint k=0;k<2;++k){uint u=a+1+2*k;MklDf dx=geo_hit(f,n,u,0)-geo_hit(f,n,u+1,0),dy=geo_hit(f,n,u,1)-geo_hit(f,n,u+1,1);out[5ul*i+2+k]=isinf(p.defocus[0])?float2(0):(mkl_sqrt(dx*dx+dy*dy)/mkl_parts<MklDf>(p.defocus[0],p.defocus[1])).v;}
        out[5ul*i+4]=float2(0);
    }
}

kernel void mkl_geometry_records(device const float* hi [[buffer(0)]],device const float* lo [[buffer(1)]],device const float2* frames [[buffer(2)]],
    device float2* out [[buffer(3)]],device float2* radii [[buffer(4)]],device uint* mask [[buffer(5)]],device float2* failures [[buffer(6)]],
    constant MklGeometryParams& p [[buffer(7)]],device const uint* status [[buffer(8)]],device float2* launch [[buffer(9)]],device const float* floats [[buffer(10)]],uint tid [[thread_position_in_grid]]) {
    uint i=p.offset+tid;if(tid>=p.count)return;uint n=p.chiefs;ulong a=ulong(MKL_AIM_RECORD_FLOATS)*i;
    if(i<n){failures[i]=float2(status[i]!=0?1.0f:0.0f,0);for(uint k=0;k<3;++k)launch[3ul*i+k]=float2(hi[ulong(MKL_AIM_RECORD_FLOATS)*i+3+k],lo[ulong(MKL_AIM_RECORD_FLOATS)*i+3+k]);}
    mask[i]=0;radii[i]=float2(0);for(uint k=0;k<3;++k)out[3ul*i+k]=float2(NAN,0);
    uint ray=i%n,code=status[ray]&MKL_RAY_CODE_MASK;bool terminal=p.centroid!=0&&status[ulong(n)+ray]==i/n&&(code==MKL_RAY_VIGNETTED||code==MKL_RAY_TIR);
    if(as_type<uint>(hi[a+12])!=0&&!terminal)return;mask[i]=terminal?2:1;
    MklDv v;for(uint k=0;k<3;++k)v.v[k]=terminal?geo_hit(floats,n,ray,k).v:float2(hi[a+k],lo[a+k]);
    radii[i]=mkl_sqrt(MklDf{v.v[0]}*MklDf{v.v[0]}+MklDf{v.v[1]}*MklDf{v.v[1]}).v;
    ulong f=12ul*(i/n);for(uint k=0;k<3;++k){MklDv rot;for(uint j=0;j<3;++j)rot.v[j]=frames[f+3*k+j];
        out[3ul*i+k]=((geo_d(frames,f+9+k)-mkl_parts<MklDf>(p.origin[2*k],p.origin[2*k+1]))+MklDf{mkl_ddot(rot,v)}).v;}
}

// A dual scalar differentiates the shared gradient recurrence in Cartesian x/y.
struct MklGeoDual {MklDf v;MklDf g[2];};
static inline MklGeoDual gd_constant(MklDf v){return {v,{geo_c(0),geo_c(0)}};}
static inline MklGeoDual operator+(MklGeoDual a,MklGeoDual b){return {a.v+b.v,{a.g[0]+b.g[0],a.g[1]+b.g[1]}};}
static inline MklGeoDual operator-(MklGeoDual a){return {-a.v,{-a.g[0],-a.g[1]}};}
static inline MklGeoDual operator-(MklGeoDual a,MklGeoDual b){return a+-b;}
static inline MklGeoDual operator*(MklGeoDual a,MklGeoDual b){return {a.v*b.v,{a.g[0]*b.v+a.v*b.g[0],a.g[1]*b.v+a.v*b.g[1]}};}
static inline MklGeoDual operator/(MklGeoDual a,MklGeoDual b){MklDf r=geo_c(1)/b.v;return a*MklGeoDual{r,{-r*r*b.g[0],-r*r*b.g[1]}};}
template<> inline MklGeoDual mkl_parts<MklGeoDual>(float hi,float lo){return gd_constant(mkl_parts<MklDf>(hi,lo));}
static inline MklGeoDual mkl_sqrt(MklGeoDual a){MklDf v=mkl_sqrt(a.v),d=geo_c(0.5)/v;return {v,{d*a.g[0],d*a.g[1]}};}
static inline bool mkl_le(MklGeoDual a,float b){return mkl_le(a.v,b);}
static inline bool mkl_zero(MklGeoDual a){return mkl_zero(a.v);}
static inline bool geo_sag(constant MklPackedSurface& s,constant float* pool,MklDf x,MklDf y,thread MklDf* v){
    MklGeoDual xd={x,{geo_c(1),geo_c(0)}},yd={y,{geo_c(0),geo_c(1)}};MklJet<MklGeoDual> j;
    uint shape=s.shape==MKL_SHAPE_DIFFRACTIVE?uint(mkl_surface_data(s,pool)[2*MKL_COEFF_KIND]):s.shape;
    bool ok=(s.flags&MKL_SURF_HAS_SAG_OVERLAY)!=0?mkl_surface_eval<true>(s,pool,xd,yd,j):
        (shape<=MKL_SHAPE_COORD_BREAK?mkl_legacy_sag(s,xd,yd,j):mkl_surface_eval<false>(s,pool,xd,yd,j));
    if(!ok)return false;
    v[0]=j.v.v;v[1]=j.x.v;v[2]=j.y.v;v[3]=j.x.g[0];v[4]=j.x.g[1];v[5]=j.y.g[1];
    if(s.shape==MKL_SHAPE_ODD_ASPHERE && mkl_zero(x) && mkl_zero(y)){
        constant float* d=mkl_surface_data(s,pool);uint na=uint(d[2*MKL_COEFF_NA]);
        if(na>0&&d[2*MKL_COEFF_PREFIX]!=0){v[3]=geo_c(NAN);v[4]=geo_c(NAN);v[5]=geo_c(NAN);}
        else if(na>1){MklDf radius=mkl_load<MklDf>(d,0),q=geo_c(2)*mkl_load<MklDf>(d,MKL_COEFF_PREFIX+1)/(radius*radius);v[3]=v[3]+q;v[5]=v[5]+q;}
    }return true;
}
struct GeoV{MklDf v[3];};struct GeoM{MklDf v[2][2];};struct GeoB{GeoV v[2];};
static inline GeoV gv(MklDf x,MklDf y,MklDf z){return {{x,y,z}};}
static inline GeoV va(GeoV a,GeoV b){GeoV o;for(uint k=0;k<3;++k)o.v[k]=a.v[k]+b.v[k];return o;}
static inline GeoV vs(GeoV a,MklDf s){GeoV o;for(uint k=0;k<3;++k)o.v[k]=a.v[k]*s;return o;}
static inline GeoV vb(GeoV a,GeoV b){return va(a,vs(b,geo_c(-1)));}
static inline MklDf vd(GeoV a,GeoV b){MklDf s=a.v[2]*b.v[2];s=s+a.v[1]*b.v[1];return s+a.v[0]*b.v[0];}
static inline GeoV vc(GeoV a,GeoV b){return gv(a.v[1]*b.v[2]-a.v[2]*b.v[1],a.v[2]*b.v[0]-a.v[0]*b.v[2],a.v[0]*b.v[1]-a.v[1]*b.v[0]);}
static inline GeoV vu(GeoV a){return vs(a,geo_c(1)/mkl_sqrt(vd(a,a)));}
static inline GeoB gb(GeoV k){GeoV a=vb(gv(geo_c(1),geo_c(0),geo_c(0)),vs(k,k.v[0]));if(vd(a,a).v.x<1e-12f)a=vb(gv(geo_c(0),geo_c(1),geo_c(0)),vs(k,k.v[1]));a=vu(a);return {{a,vc(k,a)}};}
static inline GeoM mm(GeoM a,GeoM b){GeoM o;for(uint i=0;i<2;++i)for(uint j=0;j<2;++j)o.v[i][j]=a.v[i][0]*b.v[0][j]+a.v[i][1]*b.v[1][j];return o;}
static inline GeoM ma(GeoM a,GeoM b){GeoM o;for(uint i=0;i<2;++i)for(uint j=0;j<2;++j)o.v[i][j]=a.v[i][j]+b.v[i][j];return o;}
static inline GeoM ms(GeoM a,MklDf s){GeoM o;for(uint i=0;i<2;++i)for(uint j=0;j<2;++j)o.v[i][j]=a.v[i][j]*s;return o;}
static inline GeoM mt(GeoM a){return {{{a.v[0][0],a.v[1][0]},{a.v[0][1],a.v[1][1]}}};}
static inline bool mi(GeoM a,thread GeoM& out){MklDf d=a.v[0][0]*a.v[1][1]-a.v[0][1]*a.v[1][0];if(!isfinite(d.v.x)||mkl_zero(d))return false;out={{{a.v[1][1]/d,-a.v[0][1]/d},{-a.v[1][0]/d,a.v[0][0]/d}}};return true;}
static inline GeoM mp(GeoB u,GeoB e){GeoM o;for(uint i=0;i<2;++i)for(uint j=0;j<2;++j)o.v[i][j]=vd(u.v[i],e.v[j]);return o;}
static inline GeoB br(GeoB u,GeoV n){GeoB o;for(uint k=0;k<2;++k)o.v[k]=vb(u.v[k],vs(n,geo_c(2)*vd(u.v[k],n)));return o;}
static inline GeoV vr(GeoV v,GeoV a,MklDf co){return va(va(vs(v,co),vc(a,v)),vs(a,vd(a,v)/(geo_c(1)+co)));}
static inline GeoB bc(GeoB u,GeoV k,GeoV k2){MklDf co=vd(k,k2);if(co.v.x>-0.9f){GeoV a=vc(k,k2),u0=vu(vr(u.v[0],a,co)),v=vr(u.v[1],a,co);return {{u0,vu(vb(v,vs(u0,vd(v,u0))))}};}GeoV u0=vu(vb(u.v[0],vs(k2,vd(u.v[0],k2))));return {{u0,vc(k2,u0)}};}
static inline bool tj(GeoV k,GeoB u,thread GeoM& out){MklDf az=k.v[2].v.x<0?-k.v[2]:k.v[2];if(mkl_zero(az))return false;MklDf sign=geo_c(k.v[2].v.x<0?-1:1);GeoB b={{gv(geo_c(1)/az,geo_c(0),-sign*k.v[0]/(az*az)),gv(geo_c(0),geo_c(1)/az,-sign*k.v[1]/(az*az))}};out=mp(b,u);return true;}
static inline GeoM mf(MklDf xx,MklDf xy,MklDf yy,GeoB e){GeoM o;for(uint i=0;i<2;++i)for(uint j=0;j<2;++j)o.v[i][j]=e.v[i].v[0]*(xx*e.v[j].v[0]+xy*e.v[j].v[1])+e.v[i].v[1]*(xy*e.v[j].v[0]+yy*e.v[j].v[1]);return o;}
static inline bool gf(GeoM x,GeoM t,GeoB u,GeoV k,thread MklDf* out){GeoM ti,a;bool curvature=!mi(t,ti);if(curvature){if(!mi(x,ti))return false;a=ms(mm(t,ti),geo_c(-1));}else a=ms(mm(x,ti),geo_c(-1));a=ms(ma(a,mt(a)),geo_c(0.5));MklDf mean=(a.v[0][0]+a.v[1][1])*geo_c(0.5),diff=(a.v[0][0]-a.v[1][1])*geo_c(0.5),off=a.v[0][1],r=mkl_sqrt(diff*diff+off*off);float angle=0.5f*atan2(off.v.x,diff.v.x);GeoV e=vu(vb(gv(geo_c(0),geo_c(1),geo_c(0)),vs(k,k.v[1])));MklDf dot=geo_c(cos(angle))*vd(u.v[0],e)+geo_c(sin(angle))*vd(u.v[1],e);bool plus=(dot*dot).v.x>=0.5f;MklDf v[2]={mean+r,mean-r};if(curvature)for(uint j=0;j<2;++j)v[j]=mkl_zero(v[j])?geo_c(INFINITY):geo_c(1)/v[j];out[0]=v[plus?0:1];out[1]=v[plus?1:0];out[2]=k.v[2];return true;}
static inline GeoV gr(device const float* hi,device const float* lo,uint n,uint i,uint slot,uint k){GeoV o;ulong a=(ulong(slot)*n+i)*MKL_AIM_RECORD_FLOATS;for(uint j=0;j<3;++j)o.v[j]=mkl_parts<MklDf>(hi[a+k+j],lo[a+k+j]);return o;}
static inline void gx(constant MklPackedSurface& s,thread GeoV& p,thread GeoV& d){MklDv a,b;for(uint k=0;k<3;++k){a.v[k]=p.v[k].v;b.v[k]=d.v[k].v;}mkl_xf_forward(s,a,b);for(uint k=0;k<3;++k){p.v[k]=MklDf{a.v[k]};d.v[k]=MklDf{b.v[k]};}}
static inline bool geo_coddington(constant MklPackedHeader& h,constant MklPackedSurface* s,constant float* pool,
    device const float* hi,device const float* lo,uint n,uint i,uint wl,thread MklDf* out){
    GeoM identity={{{geo_c(1),geo_c(0)},{geo_c(0),geo_c(1)}}},x=identity,t=ms(identity,geo_c(0));GeoB u={{gv(geo_c(1),geo_c(0),geo_c(0)),gv(geo_c(0),geo_c(1),geo_c(0))}};
    for(uint j=1;j<h.num_surfaces;++j){GeoV ip=gr(hi,lo,n,i,j-1,0),k=gr(hi,lo,n,i,j-1,3);gx(s[j-1],ip,k);if(j==1)u=gb(k);else for(uint v=0;v<2;++v){GeoV zero=gv(geo_c(0),geo_c(0),geo_c(0));gx(s[j-1],zero,u.v[v]);}
        MklDf d=vd(vb(gr(hi,lo,n,i,j,0),ip),k);if(j==1&&(h.flags&MKL_SYS_OBJECT_AT_INFINITY)==0){x=ms(identity,d);t=identity;}else x=ma(x,ms(t,d));
        if(j+1==h.num_surfaces)return gf(x,t,u,k,out);
        constant auto& sf=s[j];if(sf.interaction==MKL_INTERACT_DUMMY||sf.shape==MKL_SHAPE_COORD_BREAK)continue;
        ulong a=(ulong(j-1)*n+i)*MKL_AIM_RECORD_FLOATS+11;MklDf n1=mkl_parts<MklDf>(hi[a],lo[a]);bool reflect=sf.interaction==MKL_INTERACT_REFLECT;MklDf n2=reflect?n1:mkl_parts<MklDf>(sf.n_after[wl],sf.n_after_lo[wl]);GeoV kout=gr(hi,lo,n,i,j,3),nrm=vu(gr(hi,lo,n,i,j,8));
        if(sf.shape==MKL_SHAPE_PARAXIAL){GeoV z=gv(geo_c(0),geo_c(0),geo_c(1)),kin=reflect?vb(k,vs(z,geo_c(2)*k.v[2])):k;GeoB uin=reflect?br(u,z):u,e={{gv(geo_c(1),geo_c(0),geo_c(0)),gv(geo_c(0),geo_c(1),geo_c(0))}},uo=bc(uin,kin,kout);GeoM pinv,bin,bout,bi;
            if(!mi(mp(uin,e),pinv)||!tj(kin,uin,bin)||!tj(kout,uo,bout)||!mi(bout,bi))return false;
            GeoM ds=mm(pinv,x),dt=ma(ms(mm(bin,t),n1/n2),ms(ds,-mkl_parts<MklDf>(sf.paraxial_power,sf.paraxial_power_lo)));t=mm(bi,dt);x=mm(mp(uo,e),ds);u=uo;
        }else{GeoV p=gr(hi,lo,n,i,j,0);MklDf v[6];if(!geo_sag(sf,pool,p.v[0],p.v[1],v))return false;for(uint q=3;q<6;++q)if(!isfinite(v[q].v.x))return false;
            GeoV e0=vu(gv(geo_c(1),geo_c(0),v[1]));GeoB e={{e0,vc(nrm,e0)}},uo=reflect?br(u,nrm):bc(u,k,kout);GeoM km=ms(mf(v[3],v[4],v[5],e),nrm.v[2]),pin=mp(u,e),pout=mp(uo,e),pinv,pouti;if(!mi(pin,pinv)||!mi(pout,pouti))return false;
            GeoM ds=mm(pinv,x);MklDf g=n2*vd(kout,nrm)-n1*vd(k,nrm);GeoM power=ms(km,-g);
            if(sf.shape==MKL_SHAPE_DIFFRACTIVE){MklGeoDual xd={p.v[0],{geo_c(1),geo_c(0)}},yd={p.v[1],{geo_c(0),geo_c(1)}};auto ph=mkl_phase(sf,pool,xd,yd);MklDf gn=ph.x.v*nrm.v[0]+ph.y.v*nrm.v[1],q=mkl_load<MklDf>(mkl_surface_data(sf,pool),MKL_COEFF_PREFIX+wl);power=ma(power,ms(ma(mf(ph.x.g[0],ph.x.g[1],ph.y.g[1],e),ms(km,gn)),q));}
            GeoM rhs=ma(ms(mm(mt(pin),t),n1),mm(power,ds));t=ms(mm(mt(pouti),rhs),geo_c(1)/n2);x=mm(pout,ds);u=uo;
        }
    }return false;
}
kernel void mkl_geometry_coddington(constant MklPackedHeader* headers [[buffer(0)]],constant MklPackedSurface* surfaces [[buffer(1)]],constant float* pool [[buffer(2)]],
    device const float* hi [[buffer(3)]],device const float* lo [[buffer(4)]],device const uint* indices [[buffer(5)]],device const uint* hit_status [[buffer(6)]],
    constant MklGeometryParams& p [[buffer(7)]],device float2* out [[buffer(8)]],device uint* status [[buffer(9)]],uint tid [[thread_position_in_grid]]){
    uint i=p.offset+tid;if(tid>=p.count)return;uint n=p.stride;status[i]=1;for(uint k=0;k<3;++k)out[3ul*i+k]=float2(NAN,0);if(hit_status[i]!=0)return;
    uint system=indices[i],wl=indices[ulong(n)+i];if(system>=p.systems)return;constant auto& h=headers[system];if(wl>=h.num_wavelengths||h.num_surfaces!=p.segment)return;
    MklDf v[3];if(!geo_coddington(h,surfaces+h.surface_offset,pool,hi,lo,n,i,wl,v))return;for(uint k=0;k<3;++k)out[3ul*i+k]=v[k].v;status[i]=0;
}
static inline GeoV gread(device const float* f,uint n,uint i,uint k){GeoV o;for(uint j=0;j<3;++j)o.v[j]=geo_hit(f,n,i,k+j);return o;}
static inline bool gproject(device const float* f,uint n,uint i,GeoV ip,GeoV cd,thread GeoV& q,thread GeoV& slope){GeoV point=gread(f,n,i,0),d=gread(f,n,i,3);MklDf dc=vd(d,cd);if(dc.v.x<=0)return false;q=va(point,vs(d,vd(vb(ip,point),cd)/dc));slope=vb(vs(d,geo_c(1)/dc),cd);return true;}
kernel void mkl_geometry_close(device const float* f [[buffer(0)]],device const uint* hit_status [[buffer(1)]],device float2* out [[buffer(2)]],device uint* status [[buffer(3)]],constant MklGeometryParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]){
    uint i=p.offset+tid;if(tid>=p.count)return;uint n=p.chiefs,a=5*i;status[i]=1;for(uint j=0;j<3;++j)out[3ul*i+j]=float2(NAN,0);for(uint j=0;j<5;++j)if(hit_status[a+j]!=0)return;
    GeoV ip=gread(f,n,a,0),cd=gread(f,n,a,3),et=vu(vb(gv(geo_c(0),geo_c(1),geo_c(0)),vs(cd,cd.v[1])));GeoB u={{et,vc(cd,et)}};GeoV q[4],d[4];for(uint k=0;k<4;++k)if(!gproject(f,n,a+1+k,ip,cd,q[k],d[k]))return;
    GeoV xy=vb(q[0],q[1]),xx=vb(q[2],q[3]),ty=vb(d[0],d[1]),tx=vb(d[2],d[3]);GeoM x={{{vd(xy,u.v[0]),vd(xx,u.v[0])},{vd(xy,u.v[1]),vd(xx,u.v[1])}}},t={{{vd(ty,u.v[0]),vd(tx,u.v[0])},{vd(ty,u.v[1]),vd(tx,u.v[1])}}};MklDf v[3];if(!gf(x,t,u,cd,v))return;for(uint k=0;k<3;++k)out[3ul*i+k]=v[k].v;status[i]=0;
}
struct GeoRiProbe {
    constant MklPackedHeader* h;constant MklPackedSurface* s;constant float* pool;constant MklAimField* f;constant MklTraceParams* newton;
    MklDf seed[2],q0[2],x0[2],hq;GeoB basis;GeoM aim;
    MklDf load(constant float* p,uint k)const{return mkl_parts<MklDf>(p[2*k],p[2*k+1]);}
    GeoV local(GeoV p,bool point)const{GeoV a=p,o;if(point)for(uint k=0;k<3;++k)a.v[k]=p.v[k]-load(f->origin,k);for(uint k=0;k<3;++k){GeoV r;for(uint j=0;j<3;++j)r.v[j]=load(f->rotation,3*j+k);o.v[k]=vd(r,a);}return o;}
    bool initial(thread const MklDf* q,thread const MklDf* a,thread MklRawState& out)const{
        bool finite=(f->flags&MKL_AIM_FINITE)!=0,tele=(f->flags&MKL_AIM_TELECENTRIC)!=0;MklDf z=load(f->pupil,0);GeoV pos,dir;
        if(!finite){dir=vu(gv(q[0],q[1],geo_c(1)));GeoV ap=gv(a[0],a[1],z);pos=vb(ap,vs(dir,vd(ap,dir)));}
        else{MklJet<MklDf> v;uint shape=s[0].shape==MKL_SHAPE_DIFFRACTIVE?uint(mkl_surface_data(s[0],pool)[2*MKL_COEFF_KIND]):s[0].shape;
            bool ok=(s[0].flags&MKL_SURF_HAS_SAG_OVERLAY)!=0?mkl_surface_eval<true>(s[0],pool,q[0],q[1],v):(shape<=MKL_SHAPE_COORD_BREAK?mkl_legacy_sag(s[0],q[0],q[1],v):mkl_surface_eval<false>(s[0],pool,q[0],q[1],v));if(!ok)return false;
            pos=gv(q[0],q[1],v.v);if(tele)dir=local(vu(gv(a[0],a[1],geo_c(1))),false);else{GeoV ap=local(gv(a[0],a[1],z),true);dir=vu(vb(ap,pos));GeoV r=gv(load(f->rotation,6),load(f->rotation,7),load(f->rotation,8));MklDf rz=vd(r,dir);if(mkl_zero(rz))return false;if(rz.v.x<0)dir=vs(dir,geo_c(-1));}}
        out={};out.weight=1;for(uint k=0;k<3;++k){out.pos.v[k]=pos.v[k].v;out.dir.v[k]=dir.v[k].v;}return true;
    }
    __attribute__((noinline)) bool trace(thread const MklDf* q,thread const MklDf* a,uint last,bool clip,thread MklRawState& out)const{
        MklRawState ray;if(!initial(q,a,ray))return false;MklNoRecords rec;out=mkl_trace_raw<true>(*h,s,pool,nullptr,newton[clip?1:0],last,f->wavelength,ray,rec);return out.status==0;
    }
    __attribute__((noinline)) bool aim_inverse(thread const MklDf* q,thread const MklDf* a,thread GeoM& out)const{
        MklDf step=geo_c(1e-6f)*load(f->pupil,1);GeoM j;for(uint k=0;k<2;++k){MklDf aa[2]={a[0],a[1]},bb[2]={a[0],a[1]};aa[k]=aa[k]+step;bb[k]=bb[k]-step;MklRawState ha,hb;if(!trace(q,aa,h->stop,false,ha)||!trace(q,bb,h->stop,false,hb))return false;for(uint v=0;v<2;++v)j.v[v][k]=(MklDf{ha.pos.v[v]}-MklDf{hb.pos.v[v]})/(geo_c(2)*step);}return mi(j,out);
    }
    __attribute__((noinline)) bool aim_solve(thread const MklDf* q,thread const MklDf* target,thread const MklDf* initial,thread MklDf* a)const{
        a[0]=initial[0];a[1]=initial[1];GeoM inverse=aim;MklDf previous=geo_c(INFINITY),tol=geo_c(1e-12f)*load(f->pupil,2);MklRawState hit;if(!trace(q,a,h->stop,false,hit))return false;
        for(uint it=0;it<60;++it){MklDf r[2]={MklDf{hit.pos.v[0]}-target[0],MklDf{hit.pos.v[1]}-target[1]},rn=mkl_sqrt(r[0]*r[0]+r[1]*r[1]);if((rn-tol).v.x<=0)return true;
            if((rn-geo_c(0.5)*previous).v.x>0)if(!aim_inverse(q,a,inverse))return false;previous=rn;MklDf da[2];for(uint k=0;k<2;++k)da[k]=inverse.v[k][0]*r[0]+inverse.v[k][1]*r[1];MklDf alpha=geo_c(1);bool accepted=false;
            for(uint back=0;back<12;++back){MklDf next[2]={a[0]-da[0]*alpha,a[1]-da[1]*alpha};MklRawState candidate;if(trace(q,next,h->stop,false,candidate)){MklDf e[2]={MklDf{candidate.pos.v[0]}-target[0],MklDf{candidate.pos.v[1]}-target[1]};if((mkl_sqrt(e[0]*e[0]+e[1]*e[1])-rn).v.x<0){a[0]=next[0];a[1]=next[1];hit=candidate;accepted=true;break;}}alpha=alpha*geo_c(0.5);}if(!accepted)return false;
        }return false;
    }
    __attribute__((noinline)) bool ray(thread const MklDf* q,thread const MklDf* p,bool clip,thread MklDf* result)const{
        MklDf target[2];for(uint k=0;k<2;++k)target[k]=(load(f->vignette,k)+(geo_c(1)-load(f->vignette,k+2))*p[k])*load(f->pupil,2);
        MklDf predicted[2];for(uint k=0;k<2;++k)predicted[k]=seed[k]+(aim.v[k][0]*target[0]+aim.v[k][1]*target[1]);MklDf a[2];
        if(!aim_solve(q,target,predicted,a)){MklDf current[2]={seed[0],seed[1]};for(uint step=0;step<5;++step){MklDf fraction=geo_c(0.25f*float(step)),t[2]={target[0]*fraction,target[1]*fraction};if(!aim_solve(q,t,current,a))return false;current[0]=a[0];current[1]=a[1];}}
        MklRawState hit;if(!trace(q,a,h->num_surfaces-1,clip,hit))return false;GeoV d;for(uint k=0;k<3;++k)d.v[k]=MklDf{hit.dir.v[k]};result[0]=MklDf{hit.pos.v[0]};result[1]=MklDf{hit.pos.v[1]};result[2]=vd(d,basis.v[0]);result[3]=vd(d,basis.v[1]);return true;
    }
    __attribute__((noinline)) bool jac(thread const MklDf* q,thread const MklDf* p,thread MklDf j[4][4],uint columns)const{
        for(uint k=0;k<columns;++k){MklDf step=k<2?hq:mkl_parts<MklDf>(1e-4f,2.526212488436659e-12f),qa[2]={q[0],q[1]},qb[2]={q[0],q[1]},pa[2]={p[0],p[1]},pb[2]={p[0],p[1]};if(k<2){qa[k]=qa[k]+step;qb[k]=qb[k]-step;}else{pa[k-2]=pa[k-2]+step;pb[k-2]=pb[k-2]-step;}MklDf a[4],b[4];if(!ray(qa,pa,false,a)||!ray(qb,pb,false,b))return false;for(uint i=0;i<4;++i)j[i][k]=(a[i]-b[i])/(geo_c(2)*step);}return true;
    }
    __attribute__((noinline)) bool node(thread const MklDf* p,thread MklDf& density,thread bool& pass)const{
        MklDf zero[2]={geo_c(0),geo_c(0)},j[4][4];if(!jac(q0,zero,j,2))return false;GeoM j0={{{j[0][0],j[0][1]},{j[1][0],j[1][1]}}},ji;if(!mi(j0,ji))return false;
        MklDf q[2]={q0[0],q0[1]},prev=geo_c(INFINITY),tol=geo_c(1e-10f)*(geo_c(1)+mkl_sqrt(x0[0]*x0[0]+x0[1]*x0[1]));bool converged=false;
        for(uint it=0;it<40;++it){MklDf x[4];if(!ray(q,p,false,x))return false;MklDf r[2]={x[0]-x0[0],x[1]-x0[1]},rn=mkl_sqrt(r[0]*r[0]+r[1]*r[1]);if((rn-tol).v.x<=0){converged=true;break;}if((rn-geo_c(0.5)*prev).v.x>0){if(!jac(q,p,j,2))return false;GeoM xq={{{j[0][0],j[0][1]},{j[1][0],j[1][1]}}};if(!mi(xq,ji))return false;}prev=rn;MklDf dq[2];for(uint k=0;k<2;++k)dq[k]=ji.v[k][0]*r[0]+ji.v[k][1]*r[1];for(uint k=0;k<2;++k)q[k]=q[k]-dq[k];}
        if(!converged)return false;MklDf clipped[4];if(!ray(q,p,true,clipped)){density=geo_c(0);pass=false;return true;}if(!jac(q,p,j,4))return false;
        GeoM xq={{{j[0][0],j[0][1]},{j[1][0],j[1][1]}}},xp={{{j[0][2],j[0][3]},{j[1][2],j[1][3]}}},lq={{{j[2][0],j[2][1]},{j[3][0],j[3][1]}}},lp={{{j[2][2],j[2][3]},{j[3][2],j[3][3]}}},xi;
        if(!mi(xq,xi))return false;GeoM d=ma(lp,ms(mm(lq,mm(xi,xp)),geo_c(-1)));MklDf det=d.v[0][0]*d.v[1][1]-d.v[0][1]*d.v[1][0];density=det.v.x<0?-det:det;pass=true;return true;
    }
};
static inline MklDf ri_load(constant float* p,uint k){return mkl_parts<MklDf>(p[2*k],p[2*k+1]);}
static inline GeoV ri_global(GeoV v,bool point,constant MklAimField& f){GeoV out;for(uint k=0;k<3;++k){GeoV row;for(uint j=0;j<3;++j)row.v[j]=ri_load(f.rotation,3*k+j);out.v[k]=vd(row,v)+(point?ri_load(f.origin,k):geo_c(0));}return out;}
kernel void mkl_geometry_illumination(constant MklPackedHeader* headers [[buffer(0)]],constant MklPackedSurface* surfaces [[buffer(1)]],constant float* pool [[buffer(2)]],
    device const float* hi [[buffer(3)]],device const float* lo [[buffer(4)]],constant MklAimField* fields [[buffer(5)]],constant MklAimSample* nodes [[buffer(6)]],
    constant MklGeometryParams& p [[buffer(7)]],device float2* out [[buffer(8)]],device uint* status [[buffer(9)]],constant MklTraceParams* newton [[buffer(10)]],device const uint* chief_status [[buffer(11)]],device uint* ids [[buffer(12)]],uint tid [[thread_position_in_grid]]){
    uint i=p.offset+tid;if(tid>=p.count)return;uint nr=p.chiefs,field=nodes[i].field;ids[i]=i;status[i]=1;for(uint k=0;k<4;++k)out[ulong(k)*p.stride+i]=float2(0);out[2ul*p.stride+i]=float2(1,0);
    if(field>=nr||chief_status[field]!=0)return;constant auto& f=fields[field];if(f.system>=p.systems)return;constant auto& h=headers[f.system];if(h.num_surfaces!=p.segment||f.wavelength>=h.num_wavelengths)return;constant auto* s=surfaces+h.surface_offset;
    GeoV pos=gr(hi,lo,nr,field,0,0),dir=gr(hi,lo,nr,field,0,3);bool finite=(f.flags&MKL_AIM_FINITE)!=0;GeoV dp=finite?ri_global(dir,false,f):dir,pp=finite?ri_global(pos,true,f):pos;
    GeoRiProbe probe;probe.h=&h;probe.s=s;probe.pool=pool;probe.f=&f;probe.newton=newton;
    for(uint k=0;k<2;++k)probe.q0[k]=finite?pos.v[k]:dir.v[k]/dir.v[2];
    if((f.flags&MKL_AIM_TELECENTRIC)!=0){probe.seed[0]=dp.v[0]/dp.v[2];probe.seed[1]=dp.v[1]/dp.v[2];}else{MklDf t=(ri_load(f.pupil,0)-pp.v[2])/dp.v[2];for(uint k=0;k<2;++k)probe.seed[k]=pp.v[k]+t*dp.v[k];}
    GeoV normal=vu(gr(hi,lo,nr,field,h.num_surfaces-1,8)),t1=vb(gv(geo_c(1),geo_c(0),geo_c(0)),vs(normal,normal.v[0]));if(vd(t1,t1).v.x<1e-12f)t1=vb(gv(geo_c(0),geo_c(1),geo_c(0)),vs(normal,normal.v[1]));t1=vu(t1);probe.basis={{t1,vc(normal,t1)}};GeoV cp=gr(hi,lo,nr,field,h.num_surfaces-1,0);probe.x0[0]=cp.v[0];probe.x0[1]=cp.v[1];
    GeoV sp=gr(hi,lo,nr,nr-1,0,0),sd=gr(hi,lo,nr,nr-1,0,3);MklDf qs[2]={finite?sp.v[0]:sd.v[0]/sd.v[2],finite?sp.v[1]:sd.v[1]/sd.v[2]},norm=mkl_sqrt(qs[0]*qs[0]+qs[1]*qs[1]);MklDf scale=norm.v.x>1e-3f?norm:geo_c(1e-3f);probe.hq=mkl_parts<MklDf>(1e-5f,2.526212491824791e-13f)*scale;
    MklDf cz=dp.v[2]*dp.v[2];out[3ul*p.stride+i]=mkl_df_mul_f32((cz*cz).v,nodes[i].weight);
    if(!probe.aim_inverse(probe.q0,probe.seed,probe.aim))return;
    MklDf pupil[2]={ri_load(nodes[i].pupil,0),ri_load(nodes[i].pupil,1)},density;bool pass;if(!probe.node(pupil,density,pass))return;
    float w=nodes[i].weight;out[i]=mkl_df_mul_f32(density.v,w);out[ulong(p.stride)+i]=float2(pass?w:0,0);out[2ul*p.stride+i]=float2(0);out[3ul*p.stride+i]=mkl_df_mul_f32((cz*cz).v,w);status[i]=0;
}
kernel void mkl_geometry_opd(device const float* f [[buffer(0)]],device const uint* hit_status [[buffer(1)]],device const float* cf [[buffer(2)]],device const uint* chief_status [[buffer(3)]],device const uint* index [[buffer(4)]],device const float2* meta [[buffer(5)]],device float2* out [[buffer(6)]],constant MklGeometryParams& p [[buffer(7)]],device uint* status [[buffer(8)]],uint tid [[thread_position_in_grid]]){
    uint i=p.offset+tid;if(tid>=p.count)return;uint n=p.stride,nc=p.chiefs,j=index[i];out[i]=float2(NAN,0);status[i]=1;if(j>=nc||hit_status[i]!=0||chief_status[j]!=0)return;
    GeoV pos=gread(f,n,i,0),dir=gread(f,n,i,3),cp=gread(cf,nc,j,0),cd=gread(cf,nc,j,3),w=vb(pos,cp);
    if(p.centroid!=0){MklDf dc=vd(dir,cd);GeoV q=vb(va(pos,vs(dir,vd(vb(cp,pos),cd)/dc)),cp),t=vb(vs(dir,geo_c(1)/dc),cd);MklDf t2=vd(t,t),distance=t2.v.x>0?-vd(q,t)/t2:geo_c(INFINITY);out[i]=t2.v.x>0?(cp.v[2]+distance*cd.v[2]).v:float2(mkl_zero(cd.v[2])?NAN:copysign(INFINITY,cd.v[2].v.x),0);status[i]=0;return;}
MklDf b=vd(dir,w),zxp=geo_d(meta,3ul*j),corr=b;
    if(isfinite(zxp.v.x)){MklDf se=(zxp-cp.v[2])/cd.v[2],r=se.v.x<0?-se:se,sign=geo_c(se.v.x<0?-1:1);GeoV cr=vc(w,dir);MklDf p2=vd(cr,cr),delta=r*r-p2;if(delta.v.x<0)return;corr=b+sign*p2/(r+mkl_sqrt(delta));}
    MklDf chief_opl=mkl_parts<MklDf>(cf[6ul*nc+j],cf[7ul*nc+j]),opl=mkl_parts<MklDf>(f[6ul*n+i],f[7ul*n+i]);out[i]=((chief_opl-opl)+geo_d(meta,3ul*j+1)*corr).v;
    out[i]=mkl_df_div(out[i],(geo_d(meta,3ul*j+2)*mkl_parts<MklDf>(1e-3f,-4.7497450417210985e-11f)).v);status[i]=0;
}
kernel void mkl_geometry_focus_terms(device const float2* p [[buffer(0)]],device const uint* mask [[buffer(1)]],device const float2* qm [[buffer(2)]],device const float2* tm [[buffer(3)]],device float2* out [[buffer(4)]],constant MklGeometryParams& params [[buffer(7)]],uint tid [[thread_position_in_grid]]){
    uint i=params.offset+tid;if(tid>=params.count)return;uint n=params.stride;for(uint k=0;k<3;++k)out[ulong(k)*n+i]=float2(0);if(mask[i]==0)return;ulong row=ulong(i/params.segment)*10;MklDf q[2],t[2];for(uint k=0;k<2;++k){q[k]=geo_d(p,ulong(k)*n+i)-(params.centroid?geo_d(qm,row+2+k):geo_c(0));t[k]=geo_d(p,ulong(k+2)*n+i)-(params.centroid?geo_d(tm,row+2+k):geo_c(0));}MklDf w=geo_d(p,4ul*n+i);out[i]=((q[0]*q[0]+q[1]*q[1])*w).v;out[ulong(n)+i]=((q[0]*t[0]+q[1]*t[1])*w).v;out[2ul*n+i]=((t[0]*t[0]+t[1]*t[1])*w).v;
}

// ---- Restructured image-point illumination (mklight_gpu::geometry::ri) ----
// Field data comes from the host (MklRiField). Each thread runs a state
// machine whose every step is one trace from a single call site, so the
// threads of a SIMD group trace together whatever stage they are in.
static inline MklDf ri_ld(device const float* p, uint k) { return mkl_parts<MklDf>(p[2 * k], p[2 * k + 1]); }
struct RiM { MklDf v[2][2]; };
static inline RiM ri_m(device const float* p) { return {{{ri_ld(p, 0), ri_ld(p, 1)}, {ri_ld(p, 2), ri_ld(p, 3)}}}; }
static inline bool ri_inverse(RiM a, thread RiM& out) {
    MklDf d = a.v[0][0] * a.v[1][1] - a.v[0][1] * a.v[1][0];
    if (!isfinite(d.v.x) || d.v.x == 0.0f) return false;
    out = {{{a.v[1][1] / d, -a.v[0][1] / d}, {-a.v[1][0] / d, a.v[0][0] / d}}};
    return true;
}
static inline RiM ri_mul(RiM a, RiM b) {
    RiM o;
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j) o.v[i][j] = a.v[i][0] * b.v[0][j] + a.v[i][1] * b.v[1][j];
    return o;
}
static inline MklDf ri_norm(MklDf a, MklDf b) { return mkl_sqrt(a * a + b * b); }
// Stop target of pupil point (p0, p1), vignetting applied.
static inline void ri_target(device const MklAimField& f, MklDf p0, MklDf p1, thread MklDf& t0, thread MklDf& t1) {
    t0 = (ri_ld(f.vignette, 0) + (geo_c(1) - ri_ld(f.vignette, 2)) * p0) * ri_ld(f.pupil, 2);
    t1 = (ri_ld(f.vignette, 1) + (geo_c(1) - ri_ld(f.vignette, 3)) * p1) * ri_ld(f.pupil, 2);
}
// The starting ray of object parameters q and aim point a (GeoRiProbe::initial).
static inline bool ri_initial(constant MklPackedSurface* s, constant float* pool, device const MklAimField& f,
    MklDf q0, MklDf q1, MklDf a0, MklDf a1, thread MklRawState& out) {
    bool finite = (f.flags & MKL_AIM_FINITE) != 0, tele = (f.flags & MKL_AIM_TELECENTRIC) != 0;
    MklDf z = ri_ld(f.pupil, 0);
    GeoV pos, dir;
    if (!finite) {
        dir = vu(gv(q0, q1, geo_c(1)));
        GeoV ap = gv(a0, a1, z);
        pos = vb(ap, vs(dir, vd(ap, dir)));
    } else {
        MklJet<MklDf> v;
        uint shape = s[0].shape == MKL_SHAPE_DIFFRACTIVE ? uint(mkl_surface_data(s[0], pool)[2 * MKL_COEFF_KIND]) : s[0].shape;
        bool ok = (s[0].flags & MKL_SURF_HAS_SAG_OVERLAY) != 0 ? mkl_surface_eval<true>(s[0], pool, q0, q1, v)
            : (shape <= MKL_SHAPE_COORD_BREAK ? mkl_legacy_sag(s[0], q0, q1, v) : mkl_surface_eval<false>(s[0], pool, q0, q1, v));
        if (!ok) return false;
        pos = gv(q0, q1, v.v);
        if (tele) {
            GeoV d = vu(gv(a0, a1, geo_c(1))), o;
            for (uint k = 0; k < 3; ++k) { GeoV r; for (uint j = 0; j < 3; ++j) r.v[j] = ri_ld(f.rotation, 3 * j + k); o.v[k] = vd(r, d); }
            dir = o;
        } else {
            GeoV ap = gv(a0, a1, z), a, o;
            for (uint k = 0; k < 3; ++k) a.v[k] = ap.v[k] - ri_ld(f.origin, k);
            for (uint k = 0; k < 3; ++k) { GeoV r; for (uint j = 0; j < 3; ++j) r.v[j] = ri_ld(f.rotation, 3 * j + k); o.v[k] = vd(r, a); }
            dir = vu(vb(o, pos));
            GeoV r = gv(ri_ld(f.rotation, 6), ri_ld(f.rotation, 7), ri_ld(f.rotation, 8));
            MklDf rz = vd(r, dir);
            if (mkl_zero(rz)) return false;
            if (rz.v.x < 0) dir = vs(dir, geo_c(-1));
        }
    }
    out = {};
    out.weight = 1;
    for (uint k = 0; k < 3; ++k) { out.pos.v[k] = pos.v[k].v; out.dir.v[k] = dir.v[k].v; }
    return true;
}
#define RI_EVAL 0u
#define RI_FD 1u
#define RI_BACK 2u
#define RI_FINAL 3u
#define RI_DONE 4u
// Resumable aimed ray (q, P): stop solve, then the image trace (ri::AimMachine).
struct RiAim {
    MklDf q[2], target[2], goal[2], a[2], r[2], rn, prev, da[2], alpha, plus[2];
    RiM inverse, j;
    MklDf x[4];
    uint it, sub, phase, iterations, status, traces;
    int stage;
    bool clip, ok;
};
static inline RiAim ri_aim(device const MklRiField& f, MklDf q0, MklDf q1, MklDf p0, MklDf p1, bool clip,
    bool seeded, MklDf s0, MklDf s1, bool aimed) {
    RiAim m;
    m.q[0] = q0; m.q[1] = q1; m.clip = clip;
    ri_target(f.aim, p0, p1, m.target[0], m.target[1]);
    m.goal[0] = m.target[0]; m.goal[1] = m.target[1];
    m.inverse = ri_m(f.aim_inverse);
    if (seeded) { m.a[0] = s0; m.a[1] = s1; }
    else {
        for (uint k = 0; k < 2; ++k) m.a[k] = ri_ld(f.chief, k) + (m.inverse.v[k][0] * m.target[0] + m.inverse.v[k][1] * m.target[1]);
    }
    m.r[0] = geo_c(0); m.r[1] = geo_c(0); m.rn = geo_c(0);
    m.prev = geo_c(INFINITY);
    m.da[0] = geo_c(0); m.da[1] = geo_c(0); m.alpha = geo_c(1);
    m.plus[0] = geo_c(0); m.plus[1] = geo_c(0);
    m.j = {{{geo_c(0), geo_c(0)}, {geo_c(0), geo_c(0)}}};
    for (uint k = 0; k < 4; ++k) m.x[k] = geo_c(0);
    m.it = 0; m.sub = 0; m.stage = -1; m.phase = aimed ? RI_FINAL : RI_EVAL; m.iterations = 0; m.traces = 0;
    m.status = MKL_AIM_FAILED; m.ok = false;
    return m;
}
static inline MklDf ri_step(device const MklRiField& f) { return geo_c(1e-6f) * ri_ld(f.aim.pupil, 1); }
// What the machine traces next: aim point, last surface, clipping.
static inline void ri_request(thread const RiAim& m, device const MklRiField& f, constant MklPackedHeader& h,
    thread MklDf& a0, thread MklDf& a1, thread uint& last, thread bool& clip) {
    a0 = m.a[0]; a1 = m.a[1]; last = h.stop; clip = false;
    if (m.phase == RI_FD) {
        MklDf s = ri_step(f);
        bool plus = m.sub % 2 == 0;
        if (m.sub < 2) a0 = plus ? m.a[0] + s : m.a[0] - s;
        else a1 = plus ? m.a[1] + s : m.a[1] - s;
    } else if (m.phase == RI_BACK) {
        a0 = m.a[0] - m.da[0] * m.alpha;
        a1 = m.a[1] - m.da[1] * m.alpha;
    } else if (m.phase == RI_FINAL) {
        last = h.num_surfaces - 1;
        clip = m.clip;
    }
}
static inline void ri_restart(thread RiAim& m, device const MklRiField& f) {
    m.inverse = ri_m(f.aim_inverse); m.prev = geo_c(INFINITY); m.it = 0; m.phase = RI_EVAL;
}
static inline void ri_failed(thread RiAim& m, device const MklRiField& f) {
    if (m.stage < 0) {
        m.stage = 0;
        m.a[0] = ri_ld(f.chief, 0); m.a[1] = ri_ld(f.chief, 1);
        m.goal[0] = m.target[0] * geo_c(0); m.goal[1] = m.target[1] * geo_c(0);
        ri_restart(m, f);
    } else {
        m.phase = RI_DONE;
    }
}
static inline void ri_succeeded(thread RiAim& m, device const MklRiField& f) {
    if (m.stage < 0 || m.stage == 4) { m.phase = RI_FINAL; return; }
    m.stage += 1;
    MklDf fraction = geo_c(0.25f * float(m.stage));
    m.goal[0] = m.target[0] * fraction; m.goal[1] = m.target[1] * fraction;
    ri_restart(m, f);
}
static inline void ri_newton(thread RiAim& m) {
    m.prev = m.rn;
    for (uint k = 0; k < 2; ++k) m.da[k] = m.inverse.v[k][0] * m.r[0] + m.inverse.v[k][1] * m.r[1];
    m.alpha = geo_c(1); m.sub = 0; m.phase = RI_BACK;
}
static inline bool ri_floored(thread const RiAim& m, device const MklRiField& f) {
    return (m.rn - geo_c(f.aim_floor) * ri_ld(f.aim.pupil, 2)).v.x <= 0;
}
// A solve that cannot step further: accepted within the floor.
// Stuck below 1e-6 of the stop radius (ri::AIM_NOISE): trace noise, the ray fails at once.
static inline void ri_ended(thread RiAim& m, device const MklRiField& f) {
    if (ri_floored(m, f)) ri_succeeded(m, f);
    else if ((m.rn - geo_c(1e-6f) * ri_ld(f.aim.pupil, 2)).v.x <= 0) m.phase = RI_DONE;
    else ri_failed(m, f);
}
static inline void ri_check(thread RiAim& m, device const MklRiField& f) {
    if (m.it >= 60u) { ri_ended(m, f); return; }
    MklDf tol = geo_c(1e-12f) * ri_ld(f.aim.pupil, 2);
    if ((m.rn - tol).v.x <= 0) { ri_succeeded(m, f); return; }
    if ((m.rn - geo_c(0.25f) * m.prev).v.x > 0) {
        // Stalled: at the trace's noise floor, accept; else refresh.
        if (ri_floored(m, f)) { ri_succeeded(m, f); return; }
        m.phase = RI_FD; m.sub = 0; return;
    }
    ri_newton(m);
}
// Advance with the outcome (status code, final state) of the request's trace.
static inline void ri_advance(thread RiAim& m, device const MklRiField& f, uint code, thread const MklRawState& hit) {
    bool ok = code == 0u;
    MklDf pos0 = MklDf{hit.pos.v[0]}, pos1 = MklDf{hit.pos.v[1]};
    if (m.phase == RI_EVAL) {
        if (!ok) { ri_failed(m, f); return; }
        m.r[0] = pos0 - m.goal[0]; m.r[1] = pos1 - m.goal[1];
        m.rn = ri_norm(m.r[0], m.r[1]);
        ri_check(m, f);
    } else if (m.phase == RI_FD) {
        if (!ok) { ri_failed(m, f); return; }
        if (m.sub % 2 == 0) { m.plus[0] = pos0; m.plus[1] = pos1; }
        else {
            MklDf s = ri_step(f);
            MklDf c0 = (m.plus[0] - pos0) / (geo_c(2) * s), c1 = (m.plus[1] - pos1) / (geo_c(2) * s);
            if (m.sub < 2) { m.j.v[0][0] = c0; m.j.v[1][0] = c1; } else { m.j.v[0][1] = c0; m.j.v[1][1] = c1; }
        }
        m.sub += 1;
        if (m.sub == 4) {
            RiM inv;
            if (ri_inverse(m.j, inv)) { m.inverse = inv; ri_newton(m); }
            else ri_failed(m, f);
        }
    } else if (m.phase == RI_BACK) {
        bool accepted = false;
        MklDf e0 = geo_c(0), e1 = geo_c(0), en = geo_c(0);
        if (ok) {
            e0 = pos0 - m.goal[0]; e1 = pos1 - m.goal[1];
            en = ri_norm(e0, e1);
            accepted = (en - m.rn).v.x < 0;
        }
        if (accepted) {
            m.a[0] = m.a[0] - m.da[0] * m.alpha; m.a[1] = m.a[1] - m.da[1] * m.alpha;
            m.r[0] = e0; m.r[1] = e1; m.rn = en;
            m.it += 1; m.iterations += 1;
            ri_check(m, f);
        } else {
            m.alpha = m.alpha * geo_c(0.5f);
            m.sub += 1;
            if (m.sub == 12u) ri_ended(m, f);
        }
    } else if (m.phase == RI_FINAL) {
        m.phase = RI_DONE;
        m.status = code;
        m.ok = ok;
        if (ok) {
            GeoV d;
            for (uint k = 0; k < 3; ++k) d.v[k] = MklDf{hit.dir.v[k]};
            GeoV b0 = gv(ri_ld(f.basis, 0), ri_ld(f.basis, 1), ri_ld(f.basis, 2));
            GeoV b1 = gv(ri_ld(f.basis, 3), ri_ld(f.basis, 4), ri_ld(f.basis, 5));
            m.x[0] = pos0; m.x[1] = pos1; m.x[2] = vd(d, b0); m.x[3] = vd(d, b1);
        }
    }
}
// ri_advance, then the ray's trace budget (ri::RAY_TRACES = 48): an unfinished stop solve fails.
static inline void ri_feed(thread RiAim& m, device const MklRiField& f, uint code, thread const MklRawState& hit) {
    ri_advance(m, f, code, hit);
    m.traces += 1;
    if (m.phase != RI_DONE && m.phase != RI_FINAL && m.traces >= 48u) m.phase = RI_DONE;
}
// The one trace of a machine step; returns its status (MKL_AIM_INVALID without a starting ray).
static inline uint ri_trace(constant MklPackedHeader& h, constant MklPackedSurface* s, constant float* pool,
    constant MklTraceParams* newton, device const MklRiField& f, MklDf q0, MklDf q1, MklDf a0, MklDf a1, uint last,
    bool clip, thread MklRawState& out) {
    MklRawState ray;
    if (!ri_initial(s, pool, f.aim, q0, q1, a0, a1, ray)) { out = {}; return MKL_AIM_INVALID; }
    MklNoRecords rec;
    out = mkl_trace_raw<true>(h, s, pool, nullptr, newton[clip ? 1 : 0], last, f.aim.wavelength, ray, rec);
    return out.status;
}
// Field row checks shared by both kernels; MKL_RAY_OK when usable.
static inline uint ri_field_check(constant MklPackedHeader* headers, device const MklRiField* fields, uint field,
    constant MklGeometryParams& p) {
    if (field >= p.chiefs) return MKL_AIM_INVALID;
    device const MklRiField& f = fields[field];
    if (f.aim.system >= p.systems) return MKL_AIM_INVALID;
    constant MklPackedHeader& h = headers[f.aim.system];
    if (h.num_surfaces != p.segment || f.aim.wavelength >= h.num_wavelengths) return MKL_AIM_INVALID;
    return MKL_RAY_OK;
}
static inline void ri_write_df(thread float* o, MklDf v) { o[0] = v.v.x; o[1] = v.v.y; }
static inline void ri_write_df(device float* o, MklDf v) { o[0] = v.v.x; o[1] = v.v.y; }
kernel void mkl_geometry_ri_rays(constant MklPackedHeader* headers [[buffer(0)]], constant MklPackedSurface* surfaces [[buffer(1)]],
    constant float* pool [[buffer(2)]], device const MklRiField* fields [[buffer(3)]], device const MklRiTask* tasks [[buffer(4)]],
    device MklRiRay* out [[buffer(5)]], constant MklGeometryParams& p [[buffer(7)]], constant MklTraceParams* newton [[buffer(8)]],
    uint tid [[thread_position_in_grid]]) {
    uint i = p.offset + tid;
    if (tid >= p.count) return;
    MklRiTask task = tasks[i];
    MklRiRay o = {};
    o.status = ri_field_check(headers, fields, task.field, p);
    if (o.status != MKL_RAY_OK) { out[i] = o; return; }
    if ((task.flags & MKL_RI_SKIP) != 0) { o.status = MKL_AIM_OUTSIDE; out[i] = o; return; }
    device const MklRiField& f = fields[task.field];
    if (f.status != 0) { o.status = MKL_AIM_FIELD_UNREACHABLE; out[i] = o; return; }
    constant MklPackedHeader& h = headers[f.aim.system];
    constant MklPackedSurface* s = surfaces + h.surface_offset;
    RiAim m = ri_aim(f, mkl_parts<MklDf>(task.q[0], task.q[1]), mkl_parts<MklDf>(task.q[2], task.q[3]),
        mkl_parts<MklDf>(task.pupil[0], task.pupil[1]), mkl_parts<MklDf>(task.pupil[2], task.pupil[3]),
        (task.flags & MKL_RI_CLIP) != 0, (task.flags & MKL_RI_SEEDED) != 0,
        mkl_parts<MklDf>(task.seed[0], task.seed[1]), mkl_parts<MklDf>(task.seed[2], task.seed[3]),
        (task.flags & MKL_RI_AIMED) != 0);
    while (m.phase != RI_DONE) {
        MklDf a0, a1; uint last; bool clip;
        ri_request(m, f, h, a0, a1, last, clip);
        MklRawState st;
        uint code = ri_trace(h, s, pool, newton, f, m.q[0], m.q[1], a0, a1, last, clip, st);
        ri_feed(m, f, code, st);
    }
    o.status = m.status;
    o.iterations = m.iterations;
    ri_write_df(o.aim, m.a[0]); ri_write_df(o.aim + 2, m.a[1]);
    if (m.ok) {
        ri_write_df(o.x, m.x[0]); ri_write_df(o.x + 2, m.x[1]);
        ri_write_df(o.lm, m.x[2]); ri_write_df(o.lm + 2, m.x[3]);
    }
    out[i] = o;
}
#define RI_NODE_NEWTON 0u
#define RI_NODE_REFRESH 1u
#define RI_NODE_CLIP 2u
#define RI_NODE_JACOBIAN 3u
#define RI_NODE_DONE 4u
// Stencil ray s about (q, P) with converged aim (ri::stencil): q0 q1 p0 p1 seed0 seed1.
static inline void ri_stencil(device const MklRiField& f, MklDf p0, MklDf p1, MklDf q0, MklDf q1, MklDf a0, MklDf a1,
    uint s, thread MklDf* out) {
    uint k = s / 2;
    bool plus = s % 2 == 0;
    out[0] = q0; out[1] = q1; out[2] = p0; out[3] = p1; out[4] = a0; out[5] = a1;
    if (k < 2) {
        MklDf hq = ri_ld(f.hq, 0);
        if (k == 0) out[0] = plus ? q0 + hq : q0 - hq;
        else out[1] = plus ? q1 + hq : q1 - hq;
    } else {
        MklDf hp = mkl_parts<MklDf>(1e-4f, 2.526212488436659e-12f);
        if (k == 2) out[2] = plus ? p0 + hp : p0 - hp;
        else out[3] = plus ? p1 + hp : p1 - hp;
        MklDf t0, t1, u0, u1;
        ri_target(f.aim, p0, p1, t0, t1);
        ri_target(f.aim, out[2], out[3], u0, u1);
        RiM inv = ri_m(f.aim_inverse);
        MklDf d0 = u0 - t0, d1 = u1 - t1;
        out[4] = a0 + (inv.v[0][0] * d0 + inv.v[0][1] * d1);
        out[5] = a1 + (inv.v[1][0] * d0 + inv.v[1][1] * d1);
    }
}
static inline void ri_task(device MklRiTask& t, uint field, uint flags, MklDf q0, MklDf q1, MklDf p0, MklDf p1,
    MklDf s0, MklDf s1) {
    t.field = field; t.flags = flags;
    ri_write_df(t.q, q0); ri_write_df(t.q + 2, q1);
    ri_write_df(t.pupil, p0); ri_write_df(t.pupil + 2, p1);
    ri_write_df(t.seed, s0); ri_write_df(t.seed + 2, s1);
}
// One pupil node (ri::node); p.centroid != 0: Newton only, writing the node's pencil tasks.
kernel void mkl_geometry_ri_nodes(constant MklPackedHeader* headers [[buffer(0)]], constant MklPackedSurface* surfaces [[buffer(1)]],
    constant float* pool [[buffer(2)]], device const MklRiField* fields [[buffer(3)]], device const MklAimSample* nodes [[buffer(4)]],
    device float2* density [[buffer(5)]], device uint* status [[buffer(6)]], constant MklGeometryParams& p [[buffer(7)]],
    constant MklTraceParams* newton [[buffer(8)]], device MklRiTask* pencils [[buffer(9)]], uint tid [[thread_position_in_grid]]) {
    uint i = p.offset + tid;
    if (tid >= p.count) return;
    bool full = p.centroid == 0;
    uint field = nodes[i].field;
    MklDf p0 = mkl_parts<MklDf>(nodes[i].pupil[0], nodes[i].pupil[1]), p1 = mkl_parts<MklDf>(nodes[i].pupil[2], nodes[i].pupil[3]);
    density[i] = float2(0);
    status[i] = 1u;
    bool solved = false;
    MklDf q[2] = {geo_c(0), geo_c(0)}, aim[2] = {geo_c(0), geo_c(0)};
    if (ri_field_check(headers, fields, field, p) == MKL_RAY_OK && fields[field].status == 0) {
        device const MklRiField& f = fields[field];
        constant MklPackedHeader& h = headers[f.aim.system];
        constant MklPackedSurface* s = surfaces + h.surface_offset;
        MklDf x0 = ri_ld(f.x0, 0), x1 = ri_ld(f.x0, 1), tol = ri_ld(f.tolerance, 0), hq = ri_ld(f.hq, 0);
        MklDf hp = mkl_parts<MklDf>(1e-4f, 2.526212488436659e-12f);
        q[0] = ri_ld(f.q0, 0); q[1] = ri_ld(f.q0, 1);
        RiM ji = ri_m(f.q_inverse);
        MklDf prev = geo_c(INFINITY), r[2] = {geo_c(0), geo_c(0)}, rn = geo_c(0), column[4];
        MklDf j[4][4];
        for (uint a = 0; a < 4; ++a) { column[a] = geo_c(0); for (uint b = 0; b < 4; ++b) j[a][b] = geo_c(0); }
        uint it = 0, sub = 0, phase = RI_NODE_NEWTON, outcome = 1u, traces = 0;
        bool has_aim = false;
        RiM aim_inv = ri_m(f.aim_inverse);  // carried from ray to ray (ri::node)
        RiAim m = ri_aim(f, q[0], q[1], p0, p1, false, false, geo_c(0), geo_c(0), false);
        while (phase != RI_NODE_DONE) {
            if (m.phase != RI_DONE) {
                if (traces >= 192u) break;  // ri::NODE_TRACES
                MklDf a0, a1; uint last; bool clip;
                ri_request(m, f, h, a0, a1, last, clip);
                MklRawState st;
                uint code = ri_trace(h, s, pool, newton, f, m.q[0], m.q[1], a0, a1, last, clip, st);
                ri_feed(m, f, code, st);
                traces += 1;
                continue;
            }
            if (phase == RI_NODE_NEWTON) {
                if (!m.ok) break;
                aim[0] = m.a[0]; aim[1] = m.a[1]; has_aim = true;
                r[0] = m.x[0] - x0; r[1] = m.x[1] - x1;
                rn = ri_norm(r[0], r[1]);
                it += 1;
                // Stall acceptance within 1000 tolerances (ri::NODE_FLOOR).
                bool stalled = (rn - geo_c(0.25f) * prev).v.x > 0;
                bool floored = (rn - geo_c(1000.0f) * tol).v.x <= 0;
                if ((rn - tol).v.x <= 0 || (floored && (stalled || it >= 40u))) {
                    if (!full) { outcome = 0u; solved = true; break; }
                    phase = RI_NODE_CLIP;
                } else if (it >= 40u) {
                    break;
                } else if (stalled) {
                    phase = RI_NODE_REFRESH; sub = 0;
                } else {
                    prev = rn;
                    MklDf d0 = ji.v[0][0] * r[0] + ji.v[0][1] * r[1], d1 = ji.v[1][0] * r[0] + ji.v[1][1] * r[1];
                    q[0] = q[0] - d0; q[1] = q[1] - d1;
                }
            } else if (phase == RI_NODE_REFRESH) {
                if (!m.ok) break;
                if (sub % 2 == 0) { for (uint a = 0; a < 4; ++a) column[a] = m.x[a]; }
                else {
                    MklDf c0 = (column[0] - m.x[0]) / (geo_c(2) * hq), c1 = (column[1] - m.x[1]) / (geo_c(2) * hq);
                    if (sub < 2) { j[0][0] = c0; j[1][0] = c1; } else { j[0][1] = c0; j[1][1] = c1; }
                }
                sub += 1;
                if (sub == 4) {
                    RiM inv;
                    if (!ri_inverse(RiM{{{j[0][0], j[0][1]}, {j[1][0], j[1][1]}}}, inv)) break;
                    ji = inv;
                    prev = rn;
                    MklDf d0 = ji.v[0][0] * r[0] + ji.v[0][1] * r[1], d1 = ji.v[1][0] * r[0] + ji.v[1][1] * r[1];
                    q[0] = q[0] - d0; q[1] = q[1] - d1;
                    phase = RI_NODE_NEWTON; sub = 0;
                }
            } else if (phase == RI_NODE_CLIP) {
                if (!m.ok) { outcome = 2u; break; }
                phase = RI_NODE_JACOBIAN; sub = 0;
            } else {
                if (!m.ok) break;
                uint k = sub / 2;
                if (sub % 2 == 0) { for (uint a = 0; a < 4; ++a) column[a] = m.x[a]; }
                else {
                    MklDf hh = k < 2 ? hq : hp;
                    for (uint a = 0; a < 4; ++a) {
                        MklDf v = (column[a] - m.x[a]) / (geo_c(2) * hh);
                        if (k == 0) j[a][0] = v; else if (k == 1) j[a][1] = v; else if (k == 2) j[a][2] = v; else j[a][3] = v;
                    }
                }
                sub += 1;
                if (sub == 8) {
                    RiM xq = {{{j[0][0], j[0][1]}, {j[1][0], j[1][1]}}}, xp = {{{j[0][2], j[0][3]}, {j[1][2], j[1][3]}}};
                    RiM lq = {{{j[2][0], j[2][1]}, {j[3][0], j[3][1]}}}, lp = {{{j[2][2], j[2][3]}, {j[3][2], j[3][3]}}}, xi;
                    if (ri_inverse(xq, xi)) {
                        RiM corr = ri_mul(lq, ri_mul(xi, xp)), d;
                        for (uint a = 0; a < 2; ++a) for (uint b = 0; b < 2; ++b) d.v[a][b] = lp.v[a][b] - corr.v[a][b];
                        MklDf det = d.v[0][0] * d.v[1][1] - d.v[0][1] * d.v[1][0];
                        density[i] = (det.v.x < 0 ? -det : det).v;
                        outcome = 0u;
                    }
                    break;
                }
            }
            if (m.ok) aim_inv = m.inverse;
            if (phase == RI_NODE_NEWTON) {
                m = ri_aim(f, q[0], q[1], p0, p1, false, has_aim, aim[0], aim[1], false);
            } else if (phase == RI_NODE_REFRESH) {
                MklDf qa = q[0], qb = q[1];
                bool plus = sub % 2 == 0;
                if (sub < 2) qa = plus ? q[0] + hq : q[0] - hq; else qb = plus ? q[1] + hq : q[1] - hq;
                m = ri_aim(f, qa, qb, p0, p1, false, has_aim, aim[0], aim[1], false);
            } else if (phase == RI_NODE_CLIP) {
                m = ri_aim(f, q[0], q[1], p0, p1, true, has_aim, aim[0], aim[1], true);
            } else {
                MklDf st[6];
                ri_stencil(f, p0, p1, q[0], q[1], aim[0], aim[1], sub, st);
                m = ri_aim(f, st[0], st[1], st[2], st[3], false, true, st[4], st[5], false);
            }
            m.inverse = aim_inv;
        }
        status[i] = outcome;
    }
    if (!full) {
        device MklRiTask* t = pencils + ulong(i) * MKL_RI_PENCIL;
        if (solved) {
            device const MklRiField& f = fields[field];
            ri_task(t[0], field, MKL_RI_CLIP | MKL_RI_SEEDED | MKL_RI_AIMED, q[0], q[1], p0, p1, aim[0], aim[1]);
            for (uint s = 0; s < 8; ++s) {
                MklDf st[6];
                ri_stencil(f, p0, p1, q[0], q[1], aim[0], aim[1], s, st);
                ri_task(t[1 + s], field, MKL_RI_SEEDED, st[0], st[1], st[2], st[3], st[4], st[5]);
            }
        } else {
            for (uint s = 0; s < MKL_RI_PENCIL; ++s) {
                ri_task(t[s], field, MKL_RI_SKIP, geo_c(0), geo_c(0), geo_c(0), geo_c(0), geo_c(0), geo_c(0));
            }
        }
    }
}
