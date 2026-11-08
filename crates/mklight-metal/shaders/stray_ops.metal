#include "mklight_common.h"
#include "surface_math.h"
#include "trace_shared.h"
constant uint mkl_stray_features [[function_constant(MKL_STRAY_FEATURES_CONSTANT)]];
// Feature bits taken from the run's parameters instead of the constant: 0
// (every feature specialised) unless the pipeline sets it. The kernel folds
// both into one mask, so a specialised bit is a compile-time constant and a
// dynamic bit a uniform branch.
constant uint mkl_stray_dynamic_value [[function_constant(MKL_STRAY_DYNAMIC_CONSTANT)]];
constant uint mkl_stray_dynamic = is_function_constant_defined(mkl_stray_dynamic_value) ? mkl_stray_dynamic_value : 0u;
static uint sfeatures(uint runtime){return (mkl_stray_features&~mkl_stray_dynamic)|(runtime&mkl_stray_dynamic);}
// Same expression order as stray_ops/{math,kernel}.rs; no float atomics.
struct SD {
  float2 x;
  SD():x(0) {}
  SD(float v):x(v,0) {}
  SD(float2 v):x(mkl_df_input(v)) {}
};
static SD operator+(SD a,SD b){return SD(mkl_df_add(a.x,b.x));}
static SD operator-(SD a,SD b){return SD(mkl_df_sub(a.x,b.x));}
static SD operator-(SD a){return SD(-a.x);}
static SD operator*(SD a,SD b){return SD(mkl_df_mul(a.x,b.x));}
static SD operator/(SD a,SD b){return SD(mkl_df_div(a.x,b.x));}
static float sv(SD a){return a.x.x+a.x.y;}
static SD ssqrt(SD a){return SD(mkl_df_sqrt(a.x));}
struct SV {SD a[3];};
template<typename P> static SV svec(P v){return {{SD(v[0]),SD(v[1]),SD(v[2])}};}
static SV sadd(SV a,SV b){for(uint i=0;i<3;i++)a.a[i]=a.a[i]+b.a[i];return a;}
static SV ssub(SV a,SV b){for(uint i=0;i<3;i++)a.a[i]=a.a[i]-b.a[i];return a;}
static SV sscale(SV a,SD b){for(uint i=0;i<3;i++)a.a[i]=a.a[i]*b;return a;}
static SD sdot(SV a,SV b){return a.a[0]*b.a[0]+a.a[1]*b.a[1]+a.a[2]*b.a[2];}
static SV sunit(SV a){return sscale(a,SD(1)/ssqrt(sdot(a,a)));}
template<typename P> static SV srotate(P m,SV a){
  SV v; for(uint i=0;i<3;i++)v.a[i]=SD(m[i])*a.a[0]+SD(m[3+i])*a.a[1]+SD(m[6+i])*a.a[2];return v;
}
static void ssincos(SD x,thread SD &s,thread SD &c){
  SD tau(float2(6.2831855f,-1.7484555e-7f));
  SD r=x-tau*SD(round(sv(x/tau))),rr=-(r*r),st=r,ct(1);s=st;c=ct;
  for(uint i=1;i<=13;i++){
    st=st*rr/SD(float(2*i*(2*i+1)));s=s+st;
    ct=ct*rr/SD(float((2*i-1)*2*i));c=c+ct;
  }
}
static SD suniform(uint2 seed,uint2 id,thread uint &draws){
  uint4 w=mkl_philox4x32_10(uint4(draws/4,0,id.x,id.y),seed);
  uint v=w[draws%4];draws++;
  return (SD(float(v>>8))+SD(0.5f))*SD(1.0f/16777216.0f);
}
static SD suniform(uint2 seed,uint2 id,thread ulong &draws){
  uint4 w=mkl_philox4x32_10(uint4(uint(draws>>2),uint(draws>>34),id.x,id.y),seed);
  uint v=w[uint(draws&3)];draws++;
  return (SD(float(v>>8))+SD(0.5f))*SD(1.0f/16777216.0f);
}
kernel void mkl_stray_generate(constant MklStraySource &src [[buffer(0)]],
 device const float2 *cdf [[buffer(1)]],device MklStrayRay *out [[buffer(2)]],
 constant MklStrayParams &p [[buffer(3)]],uint tid [[thread_position_in_grid]]){
  uint i=p.offset+tid;if(i>=p.end)return;
  MklStraySource s=src;
  uint lo=p.first[0]+i;uint2 id(lo,p.first[1]+uint(lo<p.first[0]));
  uint2 seed(p.seed[0],p.seed[1]);uint draws=0,wl=0;
  if(s.wavelength_count>1){
    SD u=suniform(seed,id,draws);wl=s.wavelength_count-1;
    for(uint j=0;j<s.wavelength_count;j++)if(sv(u-SD(cdf[j]))<0){wl=j;break;}
  }
  SD u=suniform(seed,id,draws),v=suniform(seed,id,draws),tau(float2(6.2831855f,-1.7484555e-7f));
  SD sn,cs;ssincos(tau*v,sn,cs);
  SV axis=svec(s.axis),t1=svec(s.tangent[0]),t2=svec(s.tangent[1]),origin=svec(s.center),direction;
  SD radius(s.radius);
  if(s.kind==2){
    SD h=u*SD(s.cone),st=ssqrt(h*(SD(2)-h));
    direction=sunit(sadd(sadd(sscale(t1,st*cs),sscale(t2,st*sn)),sscale(axis,SD(1)-h)));
  } else {
    SD ri(s.inner_radius),r=s.kind==3?ssqrt(ri*ri+u*(radius*radius-ri*ri)):radius*ssqrt(u);
    origin=sadd(sadd(origin,sscale(t1,r*cs)),sscale(t2,r*sn));
    if(s.kind==3){
      SD b=SD(s.cone)*ssqrt(suniform(seed,id,draws)),s2,c2;
      ssincos(tau*suniform(seed,id,draws),s2,c2);
      SD z=ssqrt((SD(1)-b)*(SD(1)+b));
      direction=sunit(sadd(sadd(sscale(t1,b*c2),sscale(t2,b*s2)),sscale(axis,z)));
    } else {origin=ssub(origin,sscale(axis,SD(s.distance)));direction=axis;}
  }
  MklStrayRay r;for(uint j=0;j<3;j++){r.origin[j]=origin.a[j].x;r.direction[j]=direction.a[j].x;}
  r.id[0]=id.x;r.id[1]=id.y;r.draws=draws;r.wavelength=wl;out[i]=r;
}
template<typename P> static bool sslab(P lo,P hi,SV o,SV d,SD limit,thread SD &entry){
  SD a(0),b=limit;
  for(uint i=0;i<3;i++){
    if(sv(d.a[i])==0){if(sv(o.a[i]-SD(lo[i]))<0||sv(o.a[i]-SD(hi[i]))>0)return false;}
    else {
      SD x=(SD(lo[i])-o.a[i])/d.a[i],y=(SD(hi[i])-o.a[i])/d.a[i];
      if(sv(x-y)>0){SD t=x;x=y;y=t;}
      if(sv(x-a)>0)a=x;if(sv(y-b)<0)b=y;if(sv(a-b)>0)return false;
    }
  }
  entry=a;return true;
}
static bool smask(uint mask,float2 a0,float2 a1,SV q){
  SD x=q.a[0],y=q.a[1],r=x*x+y*y,a(a0),b(a1);bool pass=false;
  switch(mask&255){
  case 0:pass=true;break;
  case 1:pass=sv(r-a*a)<=0;break;
  case 2:pass=sv(r-a*a)>=0&&sv(r-b*b)<=0;break;
  case 3:pass=sv(x-a)<=0&&sv(x+a)>=0&&sv(y-b)<=0&&sv(y+b)>=0;break;
  case 4:pass=sv((x/a)*(x/a)+(y/b)*(y/b)-SD(1))<=0;break;
  }
  return pass != ((mask&256)!=0);
}
// Asphere modes (stray_asphere; reserved[2] bits 10-11): 0 the sag jets of
// surface_math.h; on an even asphere whose sag is the plain radial form (its
// overlay is the zero one the stray packing adds), 1 the f32 pre-scan in
// the radial form S(u), u = x² + y², gradient 2S'(u)·(x, y) (the records of
// mode 0), 2 also the double-float evaluations in that form, with P and P'
// from the trace's mkl_asphere_poly_dp_df (trace_shared.h), 3 mode 2 with
// each bracketed root located in f32 and polished by two double-float
// Newton steps, as the trace refines an asphere hit (mkl_intersect_legacy).
static bool sradial(constant MklStraySurfaceData &s){
  constant MklPackedSurface &f=s.surface;
  if(f.shape!=MKL_SHAPE_EVEN_ASPHERE||(f.flags&MKL_SURF_HAS_SAG_OVERLAY)==0u)return false;
  constant float *d=reinterpret_cast<constant float*>(s.coefficients)+2ul*(f.coefficient_count-MKL_OVERLAY_COUNT);
  for(uint k=MKL_OVERLAY_COS;k<MKL_OVERLAY_COUNT;k++)if(d[2*k]!=0.0f||d[2*k+1]!=0.0f)return false;
  return true;
}
// S(u) = c·u/(1 + w) + P(u), w = √(1 − (1+k)c²u); 2S'(u) = c/w + 2P'(u).
static bool sgeval_radial(constant MklPackedSurface &f,SD x,SD y,thread SD &z,thread SD &gx,thread SD &gy){
  SD u=x*x+y*y,c(float2(f.curvature,f.curvature_lo)),kap=SD(float2(f.conic,f.conic_lo))+SD(1);
  SD arg=SD(1)-kap*c*c*u;if(sv(arg)<0)return false;
  SD w=ssqrt(arg);float2 p,dp;mkl_asphere_poly_dp_df(f,u.x,p,dp);
  SD g=c/w+SD(2)*SD(dp);z=c*u/(SD(1)+w)+SD(p);gx=x*g;gy=y*g;return true;
}
static bool fgeval_radial(constant MklPackedSurface &f,float x,float y,thread float &z,thread float &g){
  float u=x*x+y*y,c=f.curvature,arg=1.0f-(1.0f+f.conic)*c*c*u;if(arg<0)return false;
  float w=sqrt(arg),rho=u*f.asphere_inv_norm2,p=0,dp=0;
  for(int j=int(min(f.num_asphere,MKL_MAX_ASPHERE))-1;j>=0;--j){dp=dp*rho+f.asphere[j]*float(j+2);p=p*rho+f.asphere[j];}
  z=c*u/(1.0f+w)+p*rho*rho;g=c/w+2.0f*dp*rho*f.asphere_inv_norm2;return true;
}
static bool sgeval(constant MklStraySurfaceData &s,SV q,SD radius,thread SD &z,thread SD &gx,thread SD &gy,bool rad=false){
  SD r=ssqrt(q.a[0]*q.a[0]+q.a[1]*q.a[1]),k=sv(r-radius)>0?radius/r:SD(1);
  if(rad)return sgeval_radial(s.surface,q.a[0]*k,q.a[1]*k,z,gx,gy);
  MklJet<MklDf> j;
  if(!mkl_surface_eval<true>(s.surface,reinterpret_cast<constant float*>(s.coefficients),MklDf{(q.a[0]*k).x},MklDf{(q.a[1]*k).x},j))return false;
  z=SD(j.v.v);gx=SD(j.x.v);gy=SD(j.y.v);return true;
}
static bool sgat(constant MklStraySurfaceData &s,SV o,SV d,SD r,SD t,thread SD &f,thread SD &df,bool rad){
  SV q=sadd(o,sscale(d,t));SD z,gx,gy;if(!sgeval(s,q,r,z,gx,gy,rad))return false;
  f=q.a[2]-z;df=d.a[2]-gx*d.a[0]-gy*d.a[1];return true;
}
static SD sgroot(constant MklStraySurfaceData &s,SV o,SV d,SD r,SD lo,SD hi,SD fa,SD tol,bool rad){
  SD t=(lo+hi)*SD(0.5f);
  for(uint i=0;i<64;i++){SD ft,dt;if(!sgat(s,o,d,r,t,ft,dt,rad))return SD(NAN);if(sv(ft)==0)return t;
    if(sv(ft*fa)<0)hi=t;else lo=t;if(sv(hi-lo-tol)<=0)break;
    SD next=t-ft/dt;t=sv(next-lo)>0&&sv(next-hi)<0?next:(lo+hi)*SD(0.5f);
  }return (lo+hi)*SD(0.5f);
}
// Plain-f32 residual, slope and residual tolerance of the sag at ray
// parameter t (the pre-scan of sag mode 1 only; never decides a root).
static bool fgat(constant MklStraySurfaceData &s,float3 o,float3 d,float radius,float t,thread float &f,thread float &df,thread float &tf,thread float &td,bool rad){
  float3 q=o+d*t;float r=sqrt(q.x*q.x+q.y*q.y),k=r>radius?radius/r:1.0f;
  MklJet<float> j;
  if(rad){float g;if(!fgeval_radial(s.surface,q.x*k,q.y*k,j.v,g))return false;j.x=q.x*k*g;j.y=q.y*k*g;}
  else if(!mkl_surface_eval<true>(s.surface,reinterpret_cast<constant float*>(s.coefficients),q.x*k,q.y*k,j))return false;
  f=q.z-j.v;df=d.z-j.x*d.x-j.y*d.y;
  // About 100 times the f32 rounding of each quantity.
  tf=2e-5f*(1+fabs(q.x)+fabs(q.y)+fabs(q.z)+fabs(j.v));td=1e-4f*(1+fabs(j.x)+fabs(j.y));
  return isfinite(f)&&isfinite(df);
}
// Asphere mode 3: the root of a bracket [lo, hi] with a sign change
// (f(lo) of the sign of fa), located by safeguarded f32 Newton and then
// polished by two double-float Newton steps; the double-float bracketing of
// sgroot if the polished root leaves the bracket.
static SD sgnewton(constant MklStraySurfaceData &s,SV o,SV d,SD r,SD lo,SD hi,SD fa,SD tol){
  float3 fo(o.a[0].x.x,o.a[1].x.x,o.a[2].x.x),fd(d.a[0].x.x,d.a[1].x.x,d.a[2].x.x);
  float a=sv(lo),b=sv(hi),t=0.5f*(a+b);bool neg=sv(fa)<0;
  for(uint i=0;i<16;i++){
    float f,df,tf,td;if(!fgat(s,fo,fd,sv(r),t,f,df,tf,td,true))break;
    if((f<0)==neg)a=t;else b=t;
    float next=t-f/df;next=next>a&&next<b?next:0.5f*(a+b);
    bool done=fabs(next-t)<=1e-6f*(1.0f+fabs(t));t=next;if(done)break;
  }
  SD tb(t);
  for(uint it=0;it<2;it++){SD ft,dt;if(!sgat(s,o,d,r,tb,ft,dt,true))return sgroot(s,o,d,r,lo,hi,fa,tol,true);
    SD step=ft/dt;if(isfinite(sv(step)))tb=tb-step;}
  if(sv(tb-lo)<0||sv(tb-hi)>0)return sgroot(s,o,d,r,lo,hi,fa,tol,true);
  return tb;
}
// Sag modes (MklStrayTransportParams.reserved[2] bits 8-9, asphere modes in
// bits 10-11 above): 0 the
// double-float cell scan; 1 a plain-f32 pre-scan that skips cells whose
// residual and slope are clearly of one sign at both ends (no root and no
// tangency the scan would find), every other cell exactly as mode 0; 2 mode 0,
// but a scan longer than MKL_STRAY_SAG_HANDOFF_CELLS cells hands the ray to
// the host (`handoff`) before any root is decided; 3 modes 1 and 2.
static bool sgintersect(device const MklStrayPrimitive &p,constant MklStraySurfaceData &s,MklStrayRay ray,SD limit,thread MklStrayHit &hit,uint sag,thread bool &handoff){
  SV o=srotate(p.rotation,ssub(svec(ray.origin),svec(p.origin))),d=srotate(p.rotation,svec(ray.direction));
  SD ta(0),tb=limit;
  for(uint k=0;k<3;k++){
    if(sv(d.a[k])==0){if(sv(o.a[k]-SD(p.lower[k]))<0||sv(o.a[k]-SD(p.upper[k]))>0)return false;}
    else{SD a=(SD(p.lower[k])-o.a[k])/d.a[k],b=(SD(p.upper[k])-o.a[k])/d.a[k];if(sv(a-b)>0){SD v=a;a=b;b=v;}if(sv(a-ta)>0)ta=a;if(sv(b-tb)<0)tb=b;}
  }
  if(sv(ta-tb)>0)return false;
  SD r(p.outer_radius),aa=d.a[0]*d.a[0]+d.a[1]*d.a[1],bb=o.a[0]*d.a[0]+o.a[1]*d.a[1],cc=o.a[0]*o.a[0]+o.a[1]*o.a[1]-r*r;
  if(sv(aa)>0){SD disc=bb*bb-aa*cc;if(sv(disc)<0)return false;SD root=ssqrt(disc),lo=(-bb-root)/aa,hi=(-bb+root)/aa;if(sv(lo-ta)>0)ta=lo;if(sv(hi-tb)<0)tb=hi;}
  else if(sv(cc)>0)return false;if(sv(ta-tb)>0)return false;
  SV oo=sadd(o,sscale(d,ta));SD len=tb-ta,step;uint n=clamp(uint(ceil(sv(len)*64/sv(r))),8u,512u);step=len/SD(float(n));
  uint mode=sag&3,am=(sag>>2)&3;
  if(mode>=2&&n>MKL_STRAY_SAG_HANDOFF_CELLS){handoff=true;return false;}
  SD tol=SD(2e-13f)*(SD(1)+len+ssqrt(sdot(oo,oo))),a(0),fa,da;
  bool rad=am!=0&&sradial(s),fast=mode==1||mode==3,have=!fast,rf=rad,rd=rad&&am>=2,newton=rad&&am==3;
  float3 fo(oo.a[0].x.x,oo.a[1].x.x,oo.a[2].x.x),fdir(d.a[0].x.x,d.a[1].x.x,d.a[2].x.x);
  float fa32=0,da32=0,tfa=0,tda=0;bool oka=fast&&fgat(s,fo,fdir,sv(r),0.0f,fa32,da32,tfa,tda,rf);
  if(!fast&&!sgat(s,oo,d,r,a,fa,da,rd))return false;
  for(uint i=1;i<=n;i++){
    SD b=i==n?len:step*SD(float(i)),fb,db;
    float fb32=0,db32=0,tfb=0,tdb=0;
    SD roots[2]={SD(NAN),SD(NAN)};
    bool clear=false;
    if(fast){
      bool okb=fgat(s,fo,fdir,sv(r),sv(b),fb32,db32,tfb,tdb,rf);
      bool sure=oka&&okb&&fabs(fa32)>tfa&&fabs(fb32)>tfb;
      if(sure&&(fa32>0)==(fb32>0)&&fabs(da32)>tda&&fabs(db32)>tdb&&(da32>0)==(db32>0)){
        a=b;fa32=fb32;da32=db32;tfa=tfb;tda=tdb;have=false;continue;
      }
      oka=okb;
      // Asphere mode 3: a sign change clear in f32 is one the double-float
      // scan finds too; locate its root without the cell's end points.
      clear=newton&&sure&&(fa32>0)!=(fb32>0);
      if(clear)roots[0]=sgnewton(s,oo,d,r,a,b,SD(fa32),tol);
      else if(!have&&!sgat(s,oo,d,r,a,fa,da,rd))return false;
    }
    if(!clear){
      if(!sgat(s,oo,d,r,b,fb,db,rd))return false;
      if(sv(fa)==0)roots[0]=a;else if(sv(fa*fb)<0)roots[0]=newton?sgnewton(s,oo,d,r,a,b,fa,tol):sgroot(s,oo,d,r,a,b,fa,tol,rd);else if(sv(fb)==0)roots[0]=b;
      else if(sv(da*db)<0){SD lo=a,hi=b;
        for(uint k=0;k<64;k++){SD mid=(lo+hi)*SD(0.5f),fm,dm;if(!sgat(s,oo,d,r,mid,fm,dm,rd))return false;if(sv(dm*da)>0)lo=mid;else hi=mid;if(sv(hi-lo-tol)<=0)break;}
        SD t=(lo+hi)*SD(0.5f),ft,dt;if(!sgat(s,oo,d,r,t,ft,dt,rd))return false;
        if(sv(ft)==0)roots[0]=t;else if(sv(ft*fa)<0){
          roots[0]=newton?sgnewton(s,oo,d,r,a,t,fa,tol):sgroot(s,oo,d,r,a,t,fa,tol,rd);
          roots[1]=newton?sgnewton(s,oo,d,r,t,b,ft,tol):sgroot(s,oo,d,r,t,b,ft,tol,rd);
        }
      }
    }
    for(uint j=0;j<2;j++){
      SD t=roots[j];if(isnan(sv(t)))continue;SD tt=ta+t;if(sv(tt)<=0||sv(tt-limit)>0)continue;
      SV q=sadd(oo,sscale(d,t));SD rr=q.a[0]*q.a[0]+q.a[1]*q.a[1],ri(p.inner_radius);
      if(sv(rr-ri*ri)<0||sv(rr-r*r)>0||!smask(p.mask,p.aperture[0],p.aperture[1],q))continue;
      SD z,gx,gy;if(!sgeval(s,q,r,z,gx,gy,rd))continue;SV normal=sunit({{-gx,-gy,SD(1)}});
      for(uint k=0;k<3;k++){hit.point[k]=q.a[k].x;hit.normal[k]=normal.a[k].x;}hit.distance=tt.x;hit.primitive=0;hit.reserved=0;return true;
    }
    a=b;fa=fb;da=db;have=!clear;fa32=fb32;da32=db32;tfa=tfb;tda=tdb;
  }return false;
}
static bool sintersect(device const MklStrayPrimitive &p,constant MklStraySurfaceData &shape,MklStrayRay ray,SD limit,thread MklStrayHit &out,bool extended,uint sag,thread bool &handoff){
  if(extended&&p.kind==3)return sgintersect(p,shape,ray,limit,out,sag,handoff);
  SV o=srotate(p.rotation,ssub(svec(ray.origin),svec(p.origin))),d=srotate(p.rotation,svec(ray.direction));
  SD shift;if(!sslab(p.lower,p.upper,o,d,limit,shift))return false;
  SV oo=sadd(o,sscale(d,shift));SD roots[2];uint count=1;
  SD c(p.curvature),k=SD(1)+SD(p.conic);
  if(p.kind==0){if(sv(d.a[2])==0)return false;roots[0]=-oo.a[2]/d.a[2];}
  else {
    SD a,b,cc;
    if(p.kind==2){
      SD r(p.outer_radius);a=d.a[0]*d.a[0]+d.a[1]*d.a[1];
      b=oo.a[0]*d.a[0]+oo.a[1]*d.a[1];cc=oo.a[0]*oo.a[0]+oo.a[1]*oo.a[1]-r*r;
    } else {
      a=c*(d.a[0]*d.a[0]+d.a[1]*d.a[1]+k*d.a[2]*d.a[2]);
      b=c*(oo.a[0]*d.a[0]+oo.a[1]*d.a[1]+k*oo.a[2]*d.a[2])-d.a[2];
      cc=c*(oo.a[0]*oo.a[0]+oo.a[1]*oo.a[1]+k*oo.a[2]*oo.a[2])-SD(2)*oo.a[2];
    }
    if(sv(a)==0){if(sv(b)==0)return false;roots[0]=-cc/(SD(2)*b);}
    else {
      SD disc=b*b-a*cc;if(sv(disc)<0)return false;
      SD sq=ssqrt(disc),q=-(b+(sv(b)<0?-sq:sq));
      if(sv(q)==0)roots[0]=-b/a;
      else {roots[0]=q/a;roots[1]=cc/q;count=2;if(sv(roots[0]-roots[1])>0){SD t=roots[0];roots[0]=roots[1];roots[1]=t;}}
    }
  }
  for(uint j=0;j<count;j++){
    SD t=shift+roots[j];if(sv(t)<=0||sv(t-limit)>0)continue;
    SV q=sadd(oo,sscale(d,roots[j])),normal;
    if(p.kind==2){
      if(sv(q.a[2]-SD(p.lower[2]))<0||sv(q.a[2]-SD(p.upper[2]))>0)continue;
      normal=sunit({{q.a[0],q.a[1],SD(0)}});
    } else {
      if(sv(k*c*q.a[2]-SD(1))>1e-9f)continue;
      SD r=q.a[0]*q.a[0]+q.a[1]*q.a[1],ri(p.inner_radius),ro(p.outer_radius);
      if(sv(r-ri*ri)<0||sv(r-ro*ro)>0||!smask(p.mask,p.aperture[0],p.aperture[1],q))continue;
      normal=sunit({{-c*q.a[0],-c*q.a[1],SD(1)-c*k*q.a[2]}});
    }
    for(uint i=0;i<3;i++){out.point[i]=q.a[i].x;out.normal[i]=normal.a[i].x;}
    out.distance=t.x;out.primitive=0;out.reserved=0;return true;
  }
  return false;
}
kernel void mkl_stray_cast(device const MklStrayPrimitive *prims [[buffer(0)]],
 device const MklStrayNode *nodes [[buffer(1)]],device const MklStrayRay *rays [[buffer(2)]],
 device MklStrayHit *output [[buffer(3)]],constant MklStrayParams &p [[buffer(4)]],constant MklStraySurfaceData *shapes [[buffer(5)]],uint tid [[thread_position_in_grid]]){
  uint index=p.offset+tid;if(index>=p.end)return;MklStrayRay ray=rays[index];
  MklStrayHit out={};out.primitive=0xffffffff;SD limit(FLT_MAX);uint i=0;
  while(i<p.nodes){
    MklStrayNode node=nodes[i];SD entry;
    if(!sslab(node.lower,node.upper,svec(ray.origin),svec(ray.direction),limit,entry)){i=node.escape;continue;}
    if(node.primitive!=0xffffffff){
      MklStrayHit h;
      bool handoff=false;
      if(sintersect(prims[node.primitive],shapes[node.primitive],ray,limit,h,true,0,handoff)){
        float delta=sv(SD(h.distance)-limit);
        if(out.primitive==0xffffffff||delta<0||(delta==0&&node.primitive<out.primitive)){
          h.primitive=node.primitive;limit=SD(h.distance);out=h;
        }
      }
    }
    i++;
  }
  output[index]=out;
}

// Reuse the P7 implementation verbatim. Its namespace keeps the runtime-source
// concatenation and offline translation units equivalent without another TMM.
namespace stray_coating {
#include "radiometry.metal"
}
static MklCoatingValue scoating(MklCoatingSpec s,device const MklCoatingLayer *layers,SD ci,uint features){
  if(features&4)return stray_coating::revaluate(s,layers,stray_coating::RD(ci.x));
  // Constant kinds let the compiler remove the film matrices in this variant.
  switch(s.kind){
    case MKL_RAD_BARE:s.kind=MKL_RAD_BARE;return stray_coating::revaluate(s,layers,stray_coating::RD(ci.x));
    case MKL_RAD_IDEAL:s.kind=MKL_RAD_IDEAL;return stray_coating::revaluate(s,layers,stray_coating::RD(ci.x));
    case MKL_RAD_TABLE:s.kind=MKL_RAD_TABLE;return stray_coating::revaluate(s,layers,stray_coating::RD(ci.x));
    default:MklCoatingValue v={};v.status=MKL_RAD_REFERENCE;return v;
  }
}
static SD smin(SD a,SD b){return sv(a-b)<0?a:b;}
static SD smax(SD a,SD b){return sv(a-b)>0?a:b;}
template<typename P> static SV sworld(P m,SV a){
  SV v;for(uint i=0;i<3;i++)v.a[i]=SD(m[3*i])*a.a[0]+SD(m[3*i+1])*a.a[1]+SD(m[3*i+2])*a.a[2];return v;
}
static void sonb(SV n,thread SV &a,thread SV &b){
  SD sign(sv(n.a[2])<0?-1:1),inv=-SD(1)/(sign+n.a[2]),xy=n.a[0]*n.a[1]*inv;
  a={{SD(1)+sign*n.a[0]*n.a[0]*inv,sign*xy,-sign*n.a[0]}};
  b={{xy,sign+n.a[1]*n.a[1]*inv,-n.a[1]}};
}
// Plain-f32 pre-test of a node box, a superset of sslab's double-float
// decision: the box grows by 1e-5 of the coordinate scale (20x the f32
// rounding of the slab distances) and the limit by 1e-6, so it only skips
// boxes that sslab rejects too, and sslab still decides every other box.
template<typename P> static bool fslab(P lo,P hi,float3 o,float3 d,float3 inv,float limit){
  float a=0,b=limit*(1+1e-6f)+1e-6f;
  for(uint i=0;i<3;i++){
    float m=1e-5f*(1+fabs(lo[i].x)+fabs(hi[i].x)+fabs(o[i])),l=lo[i].x-m,h=hi[i].x+m;
    if(d[i]==0){if(o[i]<l||o[i]>h)return false;}
    else{float x=(l-o[i])*inv[i],y=(h-o[i])*inv[i];a=max(a,min(x,y));b=min(b,max(x,y));if(a>b)return false;}
  }
  return true;
}
// Closest hit; primitive 0xfffffffe: the ray is handed to the host (sag mode 2).
static MklStrayHit sclosest(device const MklStrayPrimitive *prims,device const MklStrayNode *nodes,constant MklStraySurfaceData *shapes,uint count,MklStrayRay ray,bool extended,uint sag=0){
  MklStrayHit out={};out.primitive=0xffffffff;SD limit(FLT_MAX);uint i=0;bool handoff=false;
  float3 fo(ray.origin[0].x,ray.origin[1].x,ray.origin[2].x),fd(ray.direction[0].x,ray.direction[1].x,ray.direction[2].x),finv=1/fd;
  while(i<count){MklStrayNode node=nodes[i];SD entry;
    if(!fslab(node.lower,node.upper,fo,fd,finv,sv(limit))||!sslab(node.lower,node.upper,svec(ray.origin),svec(ray.direction),limit,entry)){i=node.escape;continue;}
    if(node.primitive!=0xffffffff){MklStrayHit h;
      if(sintersect(prims[node.primitive],shapes[node.primitive],ray,limit,h,extended,sag,handoff)){
        float delta=sv(SD(h.distance)-limit);
        if(out.primitive==0xffffffff||delta<0||(delta==0&&node.primitive<out.primitive)){h.primitive=node.primitive;limit=SD(h.distance);out=h;}
      }
      if(handoff){out.primitive=0xfffffffe;return out;}
    }i++;
  }return out;
}
// Constant-index selects keep the ledger and history in registers; a dynamic
// index would place the whole private record in slow stack memory.
static void sbook(thread MklStrayLedger &l,uint sink,SD w){
  float2 v=0;for(uint k=0;k<21;k++)v=k==sink?l.sums[k]:v;v=(SD(v)+w).x;
  for(uint k=0;k<21;k++)if(k==sink)l.sums[k]=v;if(!all(isfinite(v)))l.status=4;
}
static void schange(thread MklStrayLedger &l,uint gain,SD delta){sbook(l,gain+uint(sv(delta)<0),sv(delta)<0?-delta:delta);}
static void spush(thread MklStrayHistory &h,uint kind,uint id){
  uint e=(kind<<28)|min(id,0xfffffffU);for(uint j=0;j<6;j++)if(j==h.length)h.events[j]=e;
  if(h.length<6)h.length++;else h.overflow=1;
}
static void sspec(thread MklStrayHistory &h,device const uint *seq,uint count,uint surface,bool designed,bool extra){
  if(h.next_specular<0||uint(h.next_specular)>=count||seq[h.next_specular]!=surface)h.flags|=32;
  if(extra)h.flags^=64;h.next_specular+=(h.flags&64)?-1:1;
  if(!(h.flags&4)&&designed&&h.next_seq<count&&seq[h.next_seq]==surface)h.next_seq++;else h.flags|=4;
  if(extra){if(h.reflections==0)h.first[0]=min(surface,65534U);if(h.reflections==1)h.first[1]=min(surface,65534U);h.reflections=min(h.reflections+1,255U);}
}
static uint sclass(MklStrayHistory h,uint count){
  if(h.flags&1)return 2;if(h.flags&2)return 3;if(h.flags&8)return 4;
  if((h.reflections>=2||(h.flags&128))&&!(h.flags&96)&&h.next_specular==int(count)-1)return 1;
  if(h.reflections==0&&!(h.flags&132)&&(count==0||h.next_seq+1==count))return 0;
  return 4;
}
static SD stis(device const MklStrayMaterial &m,device const float2 *table,SV spec,SV normal){
  if(m.bsdf_kind==0)return SD(0);
  if(m.tis_count==0)return SD(m.bsdf[0]);
  SD c=smin(SD(1),smax(SD(-1),sdot(spec,normal)));
  SD beta=ssqrt(smax(SD(0),(SD(1)-c)*(SD(1)+c))),co=sv(c)<0?-c:c;
  SD angle(atan2(sv(beta),sv(co))),sn,cs;ssincos(angle,sn,cs);
  SD theta=angle+(beta*cs-co*sn)/(beta*sn+co*cs);
  SD x=theta/SD(float2(1.5707964f,-4.371139e-8f))*SD(float(m.tis_count-1));
  uint j=min(uint(floor(sv(x))),m.tis_count-2);SD f=x-SD(float(j)),f2=f*f,f3=f2*f;
  SD p1(table[m.tis_start+j]),p2(table[m.tis_start+j+1]);
  SD p0(table[m.tis_start+(j==0?1:j-1)]),p3=j+2<m.tis_count?SD(table[m.tis_start+j+2]):SD(2)*p2-p1;
  SD v=(SD(2)*f3-SD(3)*f2+SD(1))*p1+(f3-SD(2)*f2+f)*(p2-p0)*SD(0.5f)
      +(-SD(2)*f3+SD(3)*f2)*p2+(f3-f2)*(p3-p1)*SD(0.5f);
  return smin(SD(1),smax(SD(0),v));
}
static SD sexp_d(SD x){return sv(x)<-87?SD(0):SD(stray_coating::rexp(stray_coating::RD(x.x)).v);}
static SD slog_d(SD x){
  if(sv(x)<=0)return SD(NAN);int e=int((as_type<uint>(x.x.x)>>23)&255)-127;
  SD m=x*SD(ldexp(1.0f,-e)),a(log(sv(m))),ea=sexp_d(a),delta=(m-ea)/ea;
  return a+delta-delta*delta*SD(0.5f)+SD(float(e))*SD(float2(0.6931472f,-1.9046542e-9f));
}
static SD slog1p_d(SD x){
  if(abs(sv(x))>0.125f)return slog_d(SD(1)+x);
  SD t=x,s=x;for(uint j=2;j<=20;j++){t=t*(-x);s=s+t/SD(float(j));}return s;
}
static SD sexpm1_d(SD x){
  if(abs(sv(x))>0.125f)return sexp_d(x)-SD(1);
  SD t=x,s=x;for(uint j=2;j<=14;j++){t=t*x/SD(float(j));s=s+t;}return s;
}
static SD slogadd(SD a,SD b){SD hi=smax(a,b);return hi+slog1p_d(sexp_d(smin(a,b)-hi));}
static SD spower_log(SD g,SD r){float t=sv(g)*sv(r);return t>128?SD(128):(t<-128?SD(-128):g*r);}
static SD slogistic(SD x){if(sv(x)>40)return SD(0);if(sv(x)<-40)return SD(1);return SD(1)/(SD(1)+sexp_d(x));}
static SD sabg(SD b,SD g,SD rm,SD u,uint flags,thread SD &accept){
  if(flags&1){accept=SD(1);return rm*ssqrt(u);}SD lm=slog_d(rm);
  if(flags&2){SD r=rm*ssqrt(u),lr=slog_d(r);
    if(sv(b)==0)accept=sexp_d(smin(SD(0),spower_log(g,lm-lr)));
    else{SD lb=slog_d(b);accept=sexp_d(smin(SD(0),slogadd(lb,spower_log(g,lm))-slogadd(lb,spower_log(g,lr))));}return r;
  }
  SD e=SD(2)-g;
  if(sv(b)==0){SD lr=lm+slog_d(u)/e;accept=sv(g*lr)<-700?SD(NAN):SD(1);return sexp_d(lr);}
  SD lb=slog_d(b),lc=lb/g;
  if(sv(lc-lm)>=0){SD r=rm*ssqrt(u);accept=slogistic(spower_log(g,slog_d(r))-lb);return r;}
  SD l=lm-lc,el=e*l,p1(0);
  if(sv(el)<=80){SD w2=(flags&4)?l:sexpm1_d(el)/e;p1=SD(0.5f)/(SD(0.5f)+w2);}
  if(sv(u-p1)<0){SD r=sexp_d(lc)*ssqrt(u/p1);accept=slogistic(spower_log(g,slog_d(r))-lb);return r;}
  u=sv(p1)>0?(u-p1)/(SD(1)-p1):u;SD lr;
  if(flags&8)lr=lm+(sv(el)>80?slog_d(u)/e:slog1p_d((SD(1)-u)*sexpm1_d(-el))/e);
  else if(flags&4)lr=lc+u*l;else lr=lc+slog1p_d(u*sexpm1_d(el))/e;
  accept=slogistic(lb-spower_log(g,lr));return sexp_d(lr);
}
static bool sscatter(device const MklStrayMaterial &m,uint kind,SV spec,SV normal,uint2 seed,uint2 id,thread ulong &draws,thread SV &out,thread bool &reference){
  SV a,b;sonb(normal,a,b);SD bx=sdot(spec,a),by=sdot(spec,b);
  SD rm=SD(1)+ssqrt(bx*bx+by*by),sigma(m.bsdf[1]);
  SD tau(float2(6.2831855f,-1.7484555e-7f));
  for(uint k=0;k<4096;k++){
    SD u=suniform(seed,id,draws),r,accept(1);
    if(kind==1)r=ssqrt(u);
    else if(kind==2){SD ratio=rm/sigma,mass=sv(ratio)>10?SD(1):-sexpm1_d(-SD(0.5f)*ratio*ratio);r=sigma*ssqrt(-SD(2)*slog1p_d(-u*mass));}
    else r=sabg(SD(m.bsdf[1]),SD(m.bsdf[2]),rm,u,m.bsdf_flags,accept);
    if(!all(isfinite(r.x))||!all(isfinite(accept.x))){reference=true;return false;}
    SD sn,cs;ssincos(tau*suniform(seed,id,draws),sn,cs);
    SD x=r*cs,y=r*sn;if(kind!=1){x=x+bx;y=y+by;}
    bool ok=kind!=3||sv(suniform(seed,id,draws)-accept)<0;
    SD radius=x*x+y*y;
    if(ok&&(kind==1||sv(radius)<1)){
      out=sunit(sadd(sadd(sscale(a,x),sscale(b,y)),sscale(normal,ssqrt(smax(SD(0),SD(1)-radius)))));return true;
    }
  }return false;
}
static SD sinteger(int x){return SD(float(x>>16))*SD(65536)+SD(float(uint(x)&65535));}
static float snearest(SD x){
  float c=round(sv(x));SD d=x-SD(c);
  if(sv(d-SD(0.5f))>0)c+=1;else if(sv(d+SD(0.5f))<0)c-=1;return c;
}
static int skino(SD blaze,uint2 seed,uint2 id,thread ulong &draws){
  float center=snearest(blaze);if(sv(blaze-SD(center))==0)return int(center);
  SD pi(float2(3.1415927f,-8.742278e-8f));
  for(;;){
    SD sn,cs;ssincos(pi*(suniform(seed,id,draws)-SD(0.5f)),sn,cs);float k=snearest(sn/cs);
    long wide=long(center)+long(k);if(wide>2147483647L||wide<(-2147483647L-1))continue;
    int m=int(wide);float q=k==0?2*atan(0.5f)/M_PI_F:atan2(1.0f,(k-0.5f)*(k+0.5f)+1)/M_PI_F;
    SD v=sinteger(m)-blaze,rem=v-SD(snearest(v)),s(1);
    if(sv(v)!=0){SD c;ssincos(pi*rem,s,c);s=s/(pi*v);}
    if(sv(suniform(seed,id,draws)*SD(4)*SD(q)-s*s)<0)return m;
  }
}
static int sorder(device const MklStrayMaterial &m,SD blaze,uint2 seed,uint2 id,thread ulong &draws,uint features){
  if((features&128)==0||m.doe==1)return m.selected;
  if(m.doe==2)return skino(blaze,seed,id,draws);
  for(;;){int n=int(m.levels),a=skino(blaze,seed,id,draws),r=((a%n)+n)%n;
    int k=skino(-SD(float(r))/SD(float(n)),seed,id,draws);
    long v=long(k)*long(n)+long(r);if(v<=2147483647L&&v>=(-2147483647L-1))return int(v);
  }
}
struct SEvt {SV point,normal,refl,trans;uint mi,mo,prim,surface;bool dt,dr,stray,mechanical,tir;uint side;};
static uint sapply(thread MklStrayState &s,SD weight,uint branch,thread const SEvt &ev,device const MklStrayMaterial &m,device const MklStrayPrimitive &prim,
 device const uint *seq,uint seq_count,uint2 seed,thread ulong &draws,uint features){
  bool reflected=branch<2;SV spec=reflected?ev.refl:ev.trans;
  uint2 id(s.ray.id[0],s.ray.id[1]);
  if((features&16)&&m.doe){
    int order=sorder(m,SD(m.blaze[reflected?1+ev.side:0]),seed,id,draws,features);
    SV local=srotate(prim.rotation,ssub(ev.point,svec(prim.origin)));
    SD inv=SD(1)/SD(m.norm_radius),r2=(local.a[0]*local.a[0]+local.a[1]*local.a[1])*inv*inv,grad(0);
    for(int j=int(m.phase_count)-1;j>=0;j--)grad=grad*r2+SD(float(j+1))*SD(m.phase[j]);
    SD scale=sinteger(order)*SD(m.coatings[ev.side].wavelength)*SD(float2(0.001f,-4.749745e-11f))/SD(float2(6.2831855f,-1.7484555e-7f));
    SV g=sworld(prim.rotation,{{SD(2)*local.a[0]*inv*inv*grad*scale,SD(2)*local.a[1]*inv*inv*grad*scale,SD(0)}});
    SV d=svec(s.ray.direction);SD ni(m.index[ev.side]),no(m.index[reflected?ev.side:1-ev.side]);
    SV tangent=sadd(sscale(ssub(d,sscale(ev.normal,sdot(d,ev.normal))),ni),ssub(g,sscale(ev.normal,sdot(g,ev.normal))));
    tangent=sscale(tangent,SD(1)/no);SD disc=SD(1)-sdot(tangent,tangent);if(sv(disc)<0)return 11;
    spec=sunit(sadd(tangent,sscale(ev.normal,(reflected?SD(1):SD(-1))*ssqrt(disc))));
    uint k=s.history.order_count;
    if(k<32){s.history.orders[k][0]=(ev.surface&65535)|(uint(reflected)<<16)|(uint(order==m.selected)<<17);s.history.orders[k][1]=as_type<uint>(order);s.history.order_count++;}
    else s.history.overflow=1;
    if(order!=m.selected)s.history.flags|=128;
  }
  for(uint k=0;k<3;k++)if(!all(isfinite(spec.a[k].x)))return 0xfffffffe;
  SV dir=spec;bool reference=false;
  uint bsdf=(features&64)?m.bsdf_kind:1;
  if((features&32)&&(branch&1)&&!sscatter(m,bsdf,spec,sscale(ev.normal,reflected?SD(1):SD(-1)),seed,id,draws,dir,reference))return reference?0xfffffffe:15;
  if(branch&1){s.history.flags|=5;if(ev.mechanical)s.history.flags|=2;s.history.scatter_events=min(65535U,s.history.scatter_events+1);spush(s.history,ev.mechanical?6:(reflected?3:4),ev.mechanical?ev.prim:ev.surface);}
  else if(ev.mechanical){s.history.flags|=6;spush(s.history,5,ev.prim);}
  else {
    sspec(s.history,seq,seq_count,ev.surface,reflected?ev.dr:ev.dt,reflected&&ev.stray);
    if(reflected&&ev.tir)s.history.flags|=16;
    if(reflected&&ev.stray)spush(s.history,ev.tir?2:1,ev.surface);
  }
  SD eps=SD(float2(1e-9f,2.8281932e-17f))*(SD(1)+ssqrt(sdot(ev.point,ev.point)));
  SV o=sadd(ev.point,sscale(ev.normal,reflected?eps:-eps));
  for(uint k=0;k<3;k++){s.ray.origin[k]=o.a[k].x;s.ray.direction[k]=dir.a[k].x;}
  s.weight=weight.x;s.medium=reflected?ev.mi:ev.mo;return 0xffffffff;
}
static bool sroulette(thread MklStrayState &s,SD threshold,uint2 seed,thread ulong &draws,thread MklStrayLedger &l){
  SD w(s.weight);if(sv(w)<=0)return false;if(sv(w-threshold)>=0)return true;
  if(sv(suniform(seed,uint2(s.ray.id[0],s.ray.id[1]),draws)*threshold-w)<0){schange(l,16,threshold-w);s.weight=threshold.x;return true;}
  schange(l,16,-w);s.weight=float2(0);return false;
}
// One event of a source after its closest hit `hit`: bulk absorption,
// detection, the interaction's branches, the new rays and their pushes onto
// the source's stack. Returns early with l.status 4 when the source needs
// reference arithmetic. Shared by the persistent and wavefront kernels.
static void sinteract(thread MklStrayState &s,MklStrayHit hit,thread MklStrayLedger &l,uint base,uint hbase,
 device const MklStrayPrimitive *prims,device const MklStrayMaterial *materials,device const MklCoatingLayer *layers,
 device const MklStrayPass *passes,device const MklStrayMedium *media,device const float2 *tis,device const uint *seq,
 device MklStrayState *stack,device MklStrayDetection *hits,constant MklStraySettings &set,uint2 seed,uint features,SD rr,SD ceiling){
      uint wl=l.wavelength;SD w(s.weight);
      MklStrayMedium med=media[s.medium*set.wavelengths+wl];
      if(hit.primitive==0xffffffff){sbook(l,med.ambient?12:13,w);return;}
      if(sv(SD(med.alpha))>0){SD tau(stray_coating::rexp(stray_coating::RD((-SD(med.alpha)*SD(hit.distance)).x)).v);sbook(l,0,w*(SD(1)-tau));w=w*tau;s.weight=w.x;}
      device const MklStrayPrimitive &prim=prims[hit.primitive];device const MklStrayMaterial &m=materials[hit.primitive*set.wavelengths+wl];
      SV point=sadd(sworld(prim.rotation,svec(hit.point)),svec(prim.origin)),normal=sworld(prim.rotation,svec(hit.normal)),d=svec(s.ray.direction);
      uint side=sv(sdot(d,normal))<0?1:0;SV ni=sscale(normal,side?SD(1):SD(-1));
      SEvt ev;ev.point=point;ev.normal=ni;ev.refl=ssub(d,sscale(normal,SD(2)*sdot(d,normal)));ev.trans=d;ev.mi=s.medium;ev.mo=s.medium;ev.prim=hit.primitive;ev.surface=0;ev.dt=false;ev.dr=false;ev.stray=false;ev.mechanical=m.kind==0;ev.tir=false;ev.side=side;
      if(m.kind==4){
        if(set.detector_accept==2||set.detector_accept==side){MklStrayDetection h;h.history=s.history;h.weight=w.x;h.x=hit.point[0];h.y=hit.point[1];h.path_class=sclass(s.history,set.sequence_count);h.wavelength=wl;hits[hbase+l.hits++]=h;}
        else sbook(l,3,w);return;
      }
      SD probs[4],absorb;uint sink;
      if(m.kind==0){SD t=stis(m,tis,ev.refl,ni),r=smin(smax(SD(m.specular),SD(0)),SD(1)-t);probs[0]=r;probs[1]=t;probs[2]=SD(0);probs[3]=SD(0);absorb=smax(SD(0),SD(1)-r-t);sink=m.sink;}
      else {
        bool mirror=m.kind==2,designed=false;uint chosen=m.pass_start;
        int expected=(!(s.history.flags&35)&&s.history.next_specular>=0&&uint(s.history.next_specular)<set.sequence_count)?int(seq[s.history.next_specular]):-1;
        bool found=false;
        if(!mirror&&(s.history.flags&64))for(uint k=0;k<m.pass_count;k++){MklStrayPass q=passes[m.pass_start+k];if(int(q.surface)==expected&&q.side!=side){chosen=m.pass_start+k;found=true;break;}}
        if(!found){
          for(uint k=0;k<m.pass_count;k++){MklStrayPass q=passes[m.pass_start+k];if(q.side==side&&int(q.surface)==expected){chosen=m.pass_start+k;designed=true;found=true;break;}}
          if(!found)for(uint k=0;k<m.pass_count;k++)if(passes[m.pass_start+k].side==side){chosen=m.pass_start+k;designed=true;break;}
        }
        if(mirror&&!designed){sbook(l,2,w);return;}
        MklStrayPass pass=passes[chosen];
        if(!smask(pass.mask,pass.aperture[0],pass.aperture[1],svec(hit.point))){if(!(s.history.flags&8))spush(s.history,7,pass.surface);s.history.flags|=8;}
        ev.surface=pass.surface;ev.dt=designed&&!mirror;ev.dr=designed&&mirror;ev.stray=!mirror;ev.mi=m.media[side];ev.mo=m.media[1-side];
        SD ci=smin(SD(1),smax(SD(0),-sdot(d,ni)));
        MklCoatingValue cv=scoating(m.coatings[side],layers,ci,features);
        if(cv.status){l.status=4;return;}
        SD r=(SD(cv.powers[0])+SD(cv.powers[1]))*SD(0.5f),t=mirror?SD(0):(SD(cv.powers[2])+SD(cv.powers[3]))*SD(0.5f);
        bool trans=!mirror;
        if(!mirror&&!m.doe){SD ratio=SD(m.index[side])/SD(m.index[1-side]),disc=SD(1)-ratio*ratio*(SD(1)-ci)*(SD(1)+ci);
          if(sv(disc)<0){r=r+t;t=SD(0);ev.tir=true;trans=false;}
          else {ev.trans=sunit(sadd(sscale(d,ratio),sscale(ni,ratio*ci-ssqrt(disc))));
            if(m.kind==3){SV dl=srotate(prim.rotation,ev.trans);SD az=sv(dl.a[2])<0?-dl.a[2]:dl.a[2];if(sv(az)!=0){dl=sunit({{dl.a[0]/az-SD(hit.point[0])/SD(m.focal),dl.a[1]/az-SD(hit.point[1])/SD(m.focal),SD(sv(dl.a[2])<0?-1:1)}});ev.trans=sworld(prim.rotation,dl);}}
          }
        }
        SD sum=r+t;if(sv(sum)>1){r=r/sum;t=t/sum;}
        SD sr=(set.scatter&1)&&sv(r)>0?stis(m,tis,ev.refl,ni):SD(0);
        SD st=(set.scatter&2)&&trans&&sv(t)>0?stis(m,tis,ev.trans,sscale(ni,SD(-1))):SD(0);
        probs[0]=r*(SD(1)-sr);probs[1]=r*sr;probs[2]=t*(SD(1)-st);probs[3]=t*st;absorb=smax(SD(0),SD(1)-r-t);sink=1;
      }
      uint branches=0;SD sum(0);for(uint j=0;j<4;j++){branches+=uint(sv(probs[j])>0);sum=sum+probs[j];}
      if(branches==0){sbook(l,sink,w);return;}
      uint2 id(s.ray.id[0],s.ray.id[1]);
      if((features&3)==2&&s.splits<set.max_splits&&branches>=2){
        sbook(l,sink,w*absorb);MklStrayState children[4];uint count=0;
        for(uint j=0;j<4;j++)if(sv(probs[j])>0){MklStrayState c=s;c.splits++;SD cw=w*probs[j];uint lost=sapply(c,cw,j,ev,m,prim,seq,set.sequence_count,seed,l.draws,features);
          if(lost==0xffffffff)children[count++]=c;else if(lost==0xfffffffe){l.status=4;break;}else sbook(l,lost,cw);
        }
        if(count){sroulette(children[0],rr,seed,l.draws,l);for(uint j=count;j>0;j--)stack[base+l.top++]=children[j-1];}
        return;
      }
      uint pick=0;SD next=w;
      if((features&3)==0){SD u=suniform(seed,id,l.draws);if(sv(u-absorb)<0){sbook(l,sink,w);return;}SD acc=absorb;
        for(uint j=0;j<4;j++)if(sv(probs[j])>0){acc=acc+probs[j];pick=j;if(sv(u-acc)<0)break;}
      }else {
        sbook(l,sink,w*absorb);SD raw[4],z(0);for(uint j=0;j<4;j++){raw[j]=sv(probs[j])>0?smax(probs[j]/sum,SD(set.rare_floor)):SD(0);z=z+raw[j];}
        SD u=suniform(seed,id,l.draws),acc(0);for(uint j=0;j<4;j++)if(sv(probs[j])>0){acc=acc+raw[j]/z;pick=j;if(sv(u-acc)<0)break;}
        next=w*probs[pick]/(raw[pick]/z);schange(l,18,next-w*sum);
      }
      uint lost=sapply(s,next,pick,ev,m,prim,seq,set.sequence_count,seed,l.draws,features);
      if(lost==0xfffffffe){l.status=4;return;}
      if(lost!=0xffffffff){sbook(l,lost,next);return;}
      if(!sroulette(s,rr,seed,l.draws,l))return;
      uint copies=1;if(sv(SD(s.weight)-ceiling)>0){copies=uint(min(ceil(sv(SD(s.weight)/SD(set.weight))),64.0f));s.weight=(SD(s.weight)/SD(float(copies))).x;}
      for(uint j=0;j<copies;j++)stack[base+l.top++]=s;
}
// Persistent lanes claim fresh sources and resumed sources from device queues,
// so SIMD groups stay full while long split trees finish. Every source keeps
// its own stack, ledger, detections and Philox cursor: scheduling never
// changes a source's depth-first draws or records.
// queue: 0 fresh cursor, 1/2 resumed counts, 3 read cursor, 4 status bits.
kernel void mkl_stray_queue_reset(device uint *queue [[buffer(0)]],constant uint &parity [[buffer(1)]]){
  queue[3]=0;queue[2-parity]=0;
}
kernel void mkl_stray_transport(device const MklStrayPrimitive *prims [[buffer(0)]],device const MklStrayNode *nodes [[buffer(1)]],
 device const MklStrayMaterial *materials [[buffer(2)]],device const MklCoatingLayer *layers [[buffer(3)]],
 device const MklStrayPass *passes [[buffer(4)]],device const MklStrayMedium *media [[buffer(5)]],device const float2 *tis [[buffer(6)]],device const uint *seq [[buffer(7)]],
 device const MklStrayRay *rays [[buffer(8)]],device MklStrayState *stack [[buffer(9)]],device MklStrayDetection *hits [[buffer(10)]],device MklStrayLedger *ledgers [[buffer(11)]],
 constant MklStraySettings &set [[buffer(12)]],constant MklStrayTransportParams &p [[buffer(13)]],constant MklStraySurfaceData *shapes [[buffer(14)]],
 device atomic_uint *queue [[buffer(15)]],device uint *lists [[buffer(16)]],uint tid [[thread_position_in_grid]]){
  if(tid>=p.threads)return;uint2 seed(p.seed[0],p.seed[1]);uint budget=p.budget,par=p.parity,resumed=atomic_load_explicit(&queue[1+par],memory_order_relaxed);
  // Handoff mode (reserved[1]): a full stack or journal, or the head event
  // budget (reserved[0]), ends the source here for the host to continue.
  uint head=p.reserved[0],features=sfeatures(p.reserved[2]);bool handoff=p.reserved[1]!=0;
  SD rr=SD(set.roulette)*SD(set.weight),ceiling=SD(set.ceiling)*SD(set.weight);
  // Weighted forcing has Z <= 1 + 3*floor. Reserve its bounded window
  // fanout (and all four physical branches) before consuming any draws.
  uint reserve=sv(SD(set.ceiling))==FLT_MAX?3u:uint(clamp(ceil((1+3*sv(SD(set.rare_floor)))*sv(SD(set.ceiling))),3.0f,63.0f));
  while(budget){
    uint i,r=atomic_fetch_add_explicit(&queue[3],1,memory_order_relaxed);bool fresh=r>=resumed;
    if(!fresh)i=lists[par*p.count+r];
    else {i=atomic_fetch_add_explicit(&queue[0],1,memory_order_relaxed);if(i>=p.count)break;}
    uint base=i*p.stack_capacity,hbase=i*p.hit_capacity;MklStrayLedger l;
    if(fresh){l={};MklStrayState s={};s.ray=rays[i];s.weight=set.weight;s.medium=set.start_medium;s.history.first[0]=65535;s.history.first[1]=65535;
      stack[base]=s;l.top=1;l.draws=s.ray.draws;l.wavelength=s.ray.wavelength;l.sums[20]=set.weight;
    }else l=ledgers[i];
    l.status=1;
    for(;budget&&l.top;budget--){
      if(l.status==4)break;
      if(head&&l.events>=head){l.status=5;break;}
      if(l.top+reserve>p.stack_capacity){l.status=2;break;}
      if(l.hits>=p.hit_capacity){l.status=3;break;}
      MklStrayState s=stack[base+--l.top];SD w(s.weight);if(sv(w)<=0)continue;
      if(s.events>=set.max_events){sbook(l,14,w);continue;}
      s.events++;l.events++;MklStrayHit hit=sclosest(prims,nodes,shapes,p.nodes,s.ray,(features&8)!=0,(p.reserved[2]>>8)&15);
      // A ray handed to the host goes back on the stack unchanged.
      if(hit.primitive==0xfffffffe){s.events--;l.events--;stack[base+l.top++]=s;l.status=6;break;}
      sinteract(s,hit,l,base,hbase,prims,materials,layers,passes,media,tis,seq,stack,hits,set,seed,features,rr,ceiling);
    }
    if(l.top==0&&l.status==1)l.status=0;
    if(l.status==1||(l.status==2&&!handoff))lists[(1-par)*p.count+atomic_fetch_add_explicit(&queue[2-par],1,memory_order_relaxed)]=i;
    if(l.status>1)atomic_fetch_or_explicit(&queue[4],1u<<l.status,memory_order_relaxed);
    ledgers[i]=l;
  }
}
// Wavefront transport (handoff mode): each bounce advances every live
// source by one event in three kernels (pop, cast, interact), so the closest
// hit runs in its own small kernel instead of beside the interaction state.
// The per-source arithmetic and draws are those of mkl_stray_transport.
// lists: two source-id lists of p.count; counters: their lengths. A bounce
// pops from list p.parity into list 1 - p.parity.
kernel void mkl_stray_wave_init(device const MklStrayRay *rays [[buffer(0)]],device MklStrayState *stack [[buffer(1)]],device MklStrayLedger *ledgers [[buffer(2)]],
 constant MklStraySettings &set [[buffer(3)]],constant MklStrayTransportParams &p [[buffer(4)]],device uint *lists [[buffer(5)]],device uint *counters [[buffer(6)]],uint i [[thread_position_in_grid]]){
  if(i==0){counters[0]=p.count;counters[1]=0;}
  if(i>=p.count)return;
  MklStrayLedger l={};MklStrayState s={};s.ray=rays[i];s.weight=set.weight;s.medium=set.start_medium;s.history.first[0]=65535;s.history.first[1]=65535;
  stack[i*p.stack_capacity]=s;l.top=1;l.draws=s.ray.draws;l.wavelength=s.ray.wavelength;l.sums[20]=set.weight;l.status=1;ledgers[i]=l;lists[i]=i;
}
// Indirect arguments for the next wavefront kernel: q.x the counter, q.y the
// kernel's threadgroup width, q.z the word offset, q.w a counter to clear (2: none).
kernel void mkl_stray_wave_args(device uint *counters [[buffer(0)]],device uint *args [[buffer(1)]],constant uint4 &q [[buffer(2)]]){
  args[q.z]=(counters[q.x]+q.y-1)/q.y;args[q.z+1]=1;args[q.z+2]=1;if(q.w<2)counters[q.w]=0;
}
kernel void mkl_stray_wave_pop(device MklStrayState *stack [[buffer(0)]],device MklStrayLedger *ledgers [[buffer(1)]],device MklStrayState *current [[buffer(2)]],
 constant MklStraySettings &set [[buffer(3)]],constant MklStrayTransportParams &p [[buffer(4)]],device uint *lists [[buffer(5)]],device atomic_uint *counters [[buffer(6)]],uint tid [[thread_position_in_grid]]){
  uint par=p.parity;if(tid>=atomic_load_explicit(&counters[par],memory_order_relaxed))return;
  uint i=lists[par*p.count+tid],base=i*p.stack_capacity,head=p.reserved[0];MklStrayLedger l=ledgers[i];
  uint reserve=sv(SD(set.ceiling))==FLT_MAX?3u:uint(clamp(ceil((1+3*sv(SD(set.rare_floor)))*sv(SD(set.ceiling))),3.0f,63.0f));
  for(;;){
    if(l.top==0){if(l.status==1)l.status=0;break;}
    if(l.status!=1)break;
    if(head&&l.events>=head){l.status=5;break;}
    if(l.top+reserve>p.stack_capacity){l.status=2;break;}
    if(l.hits>=p.hit_capacity){l.status=3;break;}
    MklStrayState s=stack[base+--l.top];SD w(s.weight);if(sv(w)<=0)continue;
    if(s.events>=set.max_events){sbook(l,14,w);continue;}
    s.events++;l.events++;current[i]=s;
    lists[(1-par)*p.count+atomic_fetch_add_explicit(&counters[1-par],1,memory_order_relaxed)]=i;break;
  }
  ledgers[i]=l;
}
kernel void mkl_stray_wave_cast(device const MklStrayPrimitive *prims [[buffer(0)]],device const MklStrayNode *nodes [[buffer(1)]],constant MklStraySurfaceData *shapes [[buffer(2)]],
 device const MklStrayState *current [[buffer(3)]],device MklStrayHit *found [[buffer(4)]],constant MklStrayTransportParams &p [[buffer(5)]],device const uint *lists [[buffer(6)]],
 device const uint *counters [[buffer(7)]],uint tid [[thread_position_in_grid]]){
  // A tile (p.budget threads, 0: all) of the list from p.threads on.
  uint par=p.parity,t=tid+p.threads;if(t>=counters[1-par]||(p.budget!=0&&tid>=p.budget))return;uint i=lists[(1-par)*p.count+t];
  found[i]=sclosest(prims,nodes,shapes,p.nodes,current[i].ray,(sfeatures(p.reserved[2])&8)!=0,(p.reserved[2]>>8)&15);
}
kernel void mkl_stray_wave_interact(device const MklStrayPrimitive *prims [[buffer(0)]],device const MklStrayMaterial *materials [[buffer(1)]],device const MklCoatingLayer *layers [[buffer(2)]],
 device const MklStrayPass *passes [[buffer(3)]],device const MklStrayMedium *media [[buffer(4)]],device const float2 *tis [[buffer(5)]],device const uint *seq [[buffer(6)]],
 device MklStrayState *stack [[buffer(7)]],device MklStrayDetection *hits [[buffer(8)]],device MklStrayLedger *ledgers [[buffer(9)]],
 constant MklStraySettings &set [[buffer(10)]],constant MklStrayTransportParams &p [[buffer(11)]],device const MklStrayState *current [[buffer(12)]],device const MklStrayHit *found [[buffer(13)]],
 device const uint *lists [[buffer(14)]],device const uint *counters [[buffer(15)]],uint tid [[thread_position_in_grid]]){
  // A tile (p.budget threads, 0: all) of the list from p.threads on.
  uint par=p.parity,t=tid+p.threads;if(t>=counters[1-par]||(p.budget!=0&&tid>=p.budget))return;uint i=lists[(1-par)*p.count+t];
  MklStrayLedger l=ledgers[i];MklStrayState s=current[i];
  // A ray handed to the host goes back on the stack unchanged.
  if(found[i].primitive==0xfffffffe){s.events--;l.events--;stack[i*p.stack_capacity+l.top++]=s;l.status=6;ledgers[i]=l;return;}
  SD rr=SD(set.roulette)*SD(set.weight),ceiling=SD(set.ceiling)*SD(set.weight);
  sinteract(s,found[i],l,i*p.stack_capacity,i*p.hit_capacity,prims,materials,layers,passes,media,tis,seq,stack,hits,set,uint2(p.seed[0],p.seed[1]),sfeatures(p.reserved[2]),rr,ceiling);
  ledgers[i]=l;
}
// Only resumed sources own live stack entries; p: count, old/new capacity.
kernel void mkl_stray_spill(device const MklStrayState *old [[buffer(0)]],device MklStrayState *next [[buffer(1)]],device const MklStrayLedger *ledger [[buffer(2)]],
 constant uint4 &p [[buffer(3)]],device const uint *list [[buffer(4)]],uint t [[thread_position_in_grid]]){
  if(t>=p.x)return;uint i=list[t];for(uint j=0;j<ledger[i].top;j++)next[i*p.z+j]=old[i*p.y+j];
}
kernel void mkl_stray_build(device const MklStrayPrimitive *prims [[buffer(0)]],device MklStrayNode *nodes [[buffer(1)]],constant uint4 &p [[buffer(2)]],uint j [[thread_position_in_grid]]){
  uint count=1u<<p.z;if(j>=count)return;uint range=p.y>>p.z,a=j*range,id=2*a+p.z-popcount(a);
  MklStrayNode out={};out.escape=id+2*range-1;out.primitive=0xffffffff;
  if(range==1&&a<p.x){MklStrayPrimitive q=prims[a];SV lo={{SD(FLT_MAX),SD(FLT_MAX),SD(FLT_MAX)}},hi=sscale(lo,SD(-1));
    for(uint k=0;k<8;k++){SV v;for(uint i=0;i<3;i++)v.a[i]=SD((k&(1u<<i))?q.upper[i]:q.lower[i]);
      SV w;for(uint i=0;i<3;i++)w.a[i]=SD(q.origin[i])+SD(q.rotation[3*i])*v.a[0]+SD(q.rotation[3*i+1])*v.a[1]+SD(q.rotation[3*i+2])*v.a[2];
      for(uint i=0;i<3;i++){lo.a[i]=smin(lo.a[i],w.a[i]);hi.a[i]=smax(hi.a[i],w.a[i]);}}
    for(uint i=0;i<3;i++){out.lower[i]=lo.a[i].x;out.upper[i]=hi.a[i].x;}out.primitive=a;
  }else if(range>1){MklStrayNode l=nodes[id+1],h=nodes[id+range];for(uint i=0;i<3;i++){out.lower[i]=smin(SD(l.lower[i]),SD(h.lower[i])).x;out.upper[i]=smax(SD(l.upper[i]),SD(h.upper[i])).x;}}
  nodes[id]=out;
}
static bool ssame(MklStrayHistory a,MklStrayHistory b){
  if(a.length!=b.length||a.overflow!=b.overflow||a.order_count!=b.order_count)return false;
  for(uint j=0;j<6;j++)if(a.events[j]!=b.events[j])return false;
  for(uint j=0;j<32;j++)if(a.orders[j][0]!=b.orders[j][0]||a.orders[j][1]!=b.orders[j][1])return false;
  return true;
}
static bool scovered(MklStrayHistory h,device const MklStrayHistory *coverage,uint n){
  if(h.overflow)return false;
  uint refs[6],nr=0,orders[32][2],nd=0;
  for(uint j=0;j<h.length;j++){uint e=h.events[j],kind=e>>28;if(kind==1||kind==2)refs[nr++]=e&0xfffffff;}
  for(uint j=0;j<h.order_count;j++){uint tag=h.orders[j][0];if(tag&(1<<17))continue;uint surface=tag&65535,visit=0;
    for(uint k=0;k<j;k++)visit+=uint((h.orders[k][0]&65535)==surface);
    orders[nd][0]=surface|(visit<<16);orders[nd++][1]=h.orders[j][1];
  }
  for(uint k=0;k<n;k++){MklStrayHistory c=coverage[k];if(c.length!=nr||c.order_count!=nd)continue;bool same=true;
    for(uint j=0;j<nr;j++)same=same&&refs[j]==c.events[j];
    for(uint j=0;j<nd;j++){bool found=false;for(uint q=0;q<nd;q++)found=found||(orders[j][0]==c.orders[q][0]&&orders[j][1]==c.orders[q][1]);same=same&&found;}
    if(same)return true;
  }return false;
}
static uint spixel(MklStrayDetection h,constant MklStrayTallyParams &p){
  SD x=(SD(h.x)+SD(p.half_size[0]))/(SD(p.half_size[0])*SD(2)),y=(SD(h.y)+SD(p.half_size[1]))/(SD(p.half_size[1])*SD(2));
  if(sv(x)<0||sv(x-SD(1))>0||sv(y)<0||sv(y-SD(1))>0)return 0xffffffff;
  SD xx=x*SD(float(p.nx)),yy=y*SD(float(p.ny));int ix=int(floor(xx.x.x)),iy=int(floor(yy.x.x));
  if(sv(xx-SD(float(ix)))<0)ix--;if(sv(yy-SD(float(iy)))<0)iy--;
  return min(uint(iy),p.ny-1)*p.nx+min(uint(ix),p.nx-1);
}
kernel void mkl_stray_tags(device const MklStrayDetection *hits [[buffer(0)]],device const MklStrayLedger *ledger [[buffer(1)]],device const MklStrayHistory *coverage [[buffer(2)]],
 device uint4 *out [[buffer(3)]],constant MklStrayTallyParams &p [[buffer(4)]],constant uint &nc [[buffer(5)]],uint i [[thread_position_in_grid]]){
  if(i>=p.count)return;MklStrayLedger l=ledger[i/p.capacity];uint4 tag(0xffffffff,0,0,0);
  if(i%p.capacity<l.hits){MklStrayDetection h=hits[i];bool owned=(p.owned&(1<<h.path_class))!=0;
    bool uncovered=!owned&&h.path_class==1&&p.covered&&!scovered(h.history,coverage,nc);
    tag=uint4(spixel(h,p),h.path_class,uint(owned||uncovered),uint(uncovered));}
  out[i]=tag;
}
static bool sselected(uint slot,uint c,uint4 tag){if(slot<5)return c==slot;if(slot==5)return tag.z!=0;if(slot==6)return tag.w!=0;if(slot==7)return true;if(slot==8)return c==1&&tag.w==0;return false;}
static void sterms(device const MklStrayDetection *hits,device const MklStrayLedger *ledger,device const uint4 *tags,device const MklStrayQuery *queries,
 device float2 *out,constant MklStrayTallyParams &p,uint i,uint width){
  uint n=p.sources,len=n*p.queries;if(i>=len)return;device const MklStrayQuery &q=queries[i/n];uint source=i%n;device const MklStrayLedger &l=ledger[source];SD sum[4]={SD(0),SD(0),SD(0),SD(0)};
  for(uint j=0;j<(q.kind==3?1:l.hits);j++){uint k=source*p.capacity+j;SD values[4]={SD(0),SD(0),SD(0),SD(0)};
  if(q.kind==3&&j==0){if(q.key<21)values[0]=SD(l.sums[q.key]);else if(q.key==21)values[0]=SD(l.sums[16])-SD(l.sums[17])+SD(l.sums[18])-SD(l.sums[19]);else if(q.key==22)values[0]=SD(float(l.events));else if(q.key==23&&l.wavelength==q.other)values[0]=SD(l.sums[20]);}
  else if(j<l.hits){device const MklStrayDetection &h=hits[k];uint4 tag=tags[k];bool keep=false;
    switch(q.kind){case 0:keep=sselected(q.slot,h.path_class,tag);break;case 1:keep=tag.x==q.key&&sselected(q.slot,h.path_class,tag);break;
    case 2:keep=h.path_class==q.slot&&ssame(h.history,q.history);break;case 4:keep=h.path_class==1&&min(h.history.reflections,8u)==q.key;break;
    case 5:keep=h.path_class==1&&h.history.first[0]==q.key&&h.history.first[1]==q.other;break;case 6:keep=h.path_class==q.slot&&h.wavelength==q.key;break;
    case 7:keep=h.path_class==q.slot;break;}
    if(keep){values[0]=q.kind==7?SD(1):SD(h.weight);if(q.kind==0){values[1]=SD(h.weight)*SD(h.x);values[2]=SD(h.weight)*SD(h.y);values[3]=SD(h.weight)*(SD(h.x)*SD(h.x)+SD(h.y)*SD(h.y));}if(q.kind==2)values[1]=SD(1);}
  }
  for(uint c=0;c<4;c++)sum[c]=sum[c]+values[c];}
  // width 2 keeps (sum, sum²); 14 keeps all sums and upper-triangle products.
  uint row=i/n;
  if(width==2){out[(2*row)*n+source]=sum[0].x;out[(2*row+1)*n+source]=(sum[0]*sum[0]).x;}
  else if(width==14){
    for(uint c=0;c<4;c++)out[(14*row+c)*n+source]=sum[c].x;
    uint c=4;for(uint a=0;a<4;a++)for(uint b=a;b<4;b++)out[(14*row+c++)*n+source]=(sum[a]*sum[b]).x;
  }else for(uint c=0;c<4;c++)out[c*len+i]=sum[c].x;
}
kernel void mkl_stray_terms(device const MklStrayDetection *hits [[buffer(0)]],device const MklStrayLedger *ledger [[buffer(1)]],device const uint4 *tags [[buffer(2)]],device const MklStrayQuery *queries [[buffer(3)]],device float2 *out [[buffer(4)]],constant MklStrayTallyParams &p [[buffer(5)]],uint i [[thread_position_in_grid]]){sterms(hits,ledger,tags,queries,out,p,i,0);}
kernel void mkl_stray_moments(device const MklStrayDetection *hits [[buffer(0)]],device const MklStrayLedger *ledger [[buffer(1)]],device const uint4 *tags [[buffer(2)]],device const MklStrayQuery *queries [[buffer(3)]],device float2 *out [[buffer(4)]],constant MklStrayTallyParams &p [[buffer(5)]],
 constant uint &width [[buffer(6)]],uint i [[thread_position_in_grid]]){sterms(hits,ledger,tags,queries,out,p,i,width);}
kernel void mkl_stray_tally_transpose(device const float2 *input [[buffer(0)]],device float2 *out [[buffer(1)]],constant uint4 &p [[buffer(2)]],uint i [[thread_position_in_grid]]){
  uint n=p.x,q=p.y,w=p.z;if(i>=w*n*q)return;uint c=i/n,j=i%n;out[i]=input[((c/w)*n+j)*w+c%w];
}

kernel void mkl_stray_gather(device const MklStrayDetection *hits [[buffer(0)]],device const uint *indices [[buffer(1)]],device float2 *out [[buffer(2)]],constant uint &n [[buffer(3)]],uint i [[thread_position_in_grid]]){
  if(i>=n)return;out[i]=hits[indices[i]].weight;out[n+i]=float2(1,0);
}
