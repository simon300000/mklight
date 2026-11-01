#include "mklight_common.h"
#include "surface_math.h"
// Same expression order as stray_ops/{math,kernel}.rs; no float atomics.
struct SD {
  float2 x;
  SD():x(0) {}
  SD(float v):x(v,0) {}
  SD(float2 v):x(v) {}
};
static SD operator+(SD a,SD b){return SD(mkl_df_add(a.x,b.x));}
static SD operator-(SD a,SD b){return SD(mkl_df_sub(a.x,b.x));}
static SD operator-(SD a){return SD(-a.x);}
static SD operator*(SD a,SD b){return SD(mkl_df_mul(a.x,b.x));}
static SD operator/(SD a,SD b){return SD(mkl_df_div(a.x,b.x));}
static float sv(SD a){return a.x.x+a.x.y;}
static SD ssqrt(SD a){return SD(mkl_df_sqrt(a.x));}
struct SV {SD a[3];};
static SV svec(thread const float2 *v){return {{SD(v[0]),SD(v[1]),SD(v[2])}};}
static SV sadd(SV a,SV b){for(uint i=0;i<3;i++)a.a[i]=a.a[i]+b.a[i];return a;}
static SV ssub(SV a,SV b){for(uint i=0;i<3;i++)a.a[i]=a.a[i]-b.a[i];return a;}
static SV sscale(SV a,SD b){for(uint i=0;i<3;i++)a.a[i]=a.a[i]*b;return a;}
static SD sdot(SV a,SV b){return a.a[0]*b.a[0]+a.a[1]*b.a[1]+a.a[2]*b.a[2];}
static SV sunit(SV a){return sscale(a,SD(1)/ssqrt(sdot(a,a)));}
static SV srotate(thread const float2 *m,SV a){
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
static bool sslab(thread const float2 *lo,thread const float2 *hi,SV o,SV d,SD limit,thread SD &entry){
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
static bool smask(MklStrayPrimitive p,SV q){
  SD x=q.a[0],y=q.a[1],r=x*x+y*y,a(p.aperture[0]),b(p.aperture[1]);bool pass=false;
  switch(p.mask&255){
  case 0:pass=true;break;
  case 1:pass=sv(r-a*a)<=0;break;
  case 2:pass=sv(r-a*a)>=0&&sv(r-b*b)<=0;break;
  case 3:pass=sv(x-a)<=0&&sv(x+a)>=0&&sv(y-b)<=0&&sv(y+b)>=0;break;
  case 4:pass=sv((x/a)*(x/a)+(y/b)*(y/b)-SD(1))<=0;break;
  }
  return pass != ((p.mask&256)!=0);
}
static bool sgeval(constant MklStraySurfaceData &s,SV q,SD radius,thread SD &z,thread SD &gx,thread SD &gy){
  SD r=ssqrt(q.a[0]*q.a[0]+q.a[1]*q.a[1]),k=sv(r-radius)>0?radius/r:SD(1);
  MklJet<MklDf> j;
  if(!mkl_surface_eval<true>(s.surface,reinterpret_cast<constant float*>(s.coefficients),MklDf{(q.a[0]*k).x},MklDf{(q.a[1]*k).x},j))return false;
  z=SD(j.v.v);gx=SD(j.x.v);gy=SD(j.y.v);return true;
}
static bool sgat(constant MklStraySurfaceData &s,SV o,SV d,SD r,SD t,thread SD &f,thread SD &df){
  SV q=sadd(o,sscale(d,t));SD z,gx,gy;if(!sgeval(s,q,r,z,gx,gy))return false;
  f=q.a[2]-z;df=d.a[2]-gx*d.a[0]-gy*d.a[1];return true;
}
static SD sgroot(constant MklStraySurfaceData &s,SV o,SV d,SD r,SD lo,SD hi,SD fa,SD tol){
  SD t=(lo+hi)*SD(0.5f);
  for(uint i=0;i<64;i++){SD ft,dt;if(!sgat(s,o,d,r,t,ft,dt))return SD(NAN);if(sv(ft)==0)return t;
    if(sv(ft*fa)<0)hi=t;else lo=t;if(sv(hi-lo-tol)<=0)break;
    SD next=t-ft/dt;t=sv(next-lo)>0&&sv(next-hi)<0?next:(lo+hi)*SD(0.5f);
  }return (lo+hi)*SD(0.5f);
}
static bool sgintersect(MklStrayPrimitive p,constant MklStraySurfaceData &s,MklStrayRay ray,SD limit,thread MklStrayHit &hit){
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
  SD tol=SD(2e-13f)*(SD(1)+len+ssqrt(sdot(oo,oo))),a(0),fa,da;
  if(!sgat(s,oo,d,r,a,fa,da))return false;
  for(uint i=1;i<=n;i++){
    SD b=i==n?len:step*SD(float(i)),fb,db;if(!sgat(s,oo,d,r,b,fb,db))return false;
    SD roots[2]={SD(NAN),SD(NAN)};
    if(sv(fa)==0)roots[0]=a;else if(sv(fa*fb)<0)roots[0]=sgroot(s,oo,d,r,a,b,fa,tol);else if(sv(fb)==0)roots[0]=b;
    else if(sv(da*db)<0){SD lo=a,hi=b;
      for(uint k=0;k<64;k++){SD mid=(lo+hi)*SD(0.5f),fm,dm;if(!sgat(s,oo,d,r,mid,fm,dm))return false;if(sv(dm*da)>0)lo=mid;else hi=mid;if(sv(hi-lo-tol)<=0)break;}
      SD t=(lo+hi)*SD(0.5f),ft,dt;if(!sgat(s,oo,d,r,t,ft,dt))return false;
      if(sv(ft)==0)roots[0]=t;else if(sv(ft*fa)<0){roots[0]=sgroot(s,oo,d,r,a,t,fa,tol);roots[1]=sgroot(s,oo,d,r,t,b,ft,tol);}
    }
    for(uint j=0;j<2;j++){
      SD t=roots[j];if(isnan(sv(t)))continue;SD tt=ta+t;if(sv(tt)<=0||sv(tt-limit)>0)continue;
      SV q=sadd(oo,sscale(d,t));SD rr=q.a[0]*q.a[0]+q.a[1]*q.a[1],ri(p.inner_radius);
      if(sv(rr-ri*ri)<0||sv(rr-r*r)>0||!smask(p,q))continue;
      SD z,gx,gy;if(!sgeval(s,q,r,z,gx,gy))continue;SV normal=sunit({{-gx,-gy,SD(1)}});
      for(uint k=0;k<3;k++){hit.point[k]=q.a[k].x;hit.normal[k]=normal.a[k].x;}hit.distance=tt.x;hit.primitive=0;hit.reserved=0;return true;
    }
    a=b;fa=fb;da=db;
  }return false;
}
static bool sintersect(MklStrayPrimitive p,constant MklStraySurfaceData &shape,MklStrayRay ray,SD limit,thread MklStrayHit &out){
  if(p.kind==3)return sgintersect(p,shape,ray,limit,out);
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
      if(sv(r-ri*ri)<0||sv(r-ro*ro)>0||!smask(p,q))continue;
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
      if(sintersect(prims[node.primitive],shapes[node.primitive],ray,limit,h)){
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
static SD smin(SD a,SD b){return sv(a-b)<0?a:b;}
static SD smax(SD a,SD b){return sv(a-b)>0?a:b;}
static SV sworld(thread const float2 *m,SV a){
  SV v;for(uint i=0;i<3;i++)v.a[i]=SD(m[3*i])*a.a[0]+SD(m[3*i+1])*a.a[1]+SD(m[3*i+2])*a.a[2];return v;
}
static void sonb(SV n,thread SV &a,thread SV &b){
  SD sign(sv(n.a[2])<0?-1:1),inv=-SD(1)/(sign+n.a[2]),xy=n.a[0]*n.a[1]*inv;
  a={{SD(1)+sign*n.a[0]*n.a[0]*inv,sign*xy,-sign*n.a[0]}};
  b={{xy,sign+n.a[1]*n.a[1]*inv,-n.a[1]}};
}
static MklStrayHit sclosest(device const MklStrayPrimitive *prims,device const MklStrayNode *nodes,constant MklStraySurfaceData *shapes,uint count,MklStrayRay ray){
  MklStrayHit out={};out.primitive=0xffffffff;SD limit(FLT_MAX);uint i=0;
  while(i<count){MklStrayNode node=nodes[i];SD entry;
    if(!sslab(node.lower,node.upper,svec(ray.origin),svec(ray.direction),limit,entry)){i=node.escape;continue;}
    if(node.primitive!=0xffffffff){MklStrayHit h;
      if(sintersect(prims[node.primitive],shapes[node.primitive],ray,limit,h)){
        float delta=sv(SD(h.distance)-limit);
        if(out.primitive==0xffffffff||delta<0||(delta==0&&node.primitive<out.primitive)){h.primitive=node.primitive;limit=SD(h.distance);out=h;}
      }
    }i++;
  }return out;
}
static void sbook(thread MklStrayLedger &l,uint sink,SD w){l.sums[sink]=(SD(l.sums[sink])+w).x;}
static void schange(thread MklStrayLedger &l,uint gain,SD delta){sbook(l,gain+uint(sv(delta)<0),sv(delta)<0?-delta:delta);}
static void spush(thread MklStrayHistory &h,uint kind,uint id){
  if(h.length<6)h.events[h.length++]=(kind<<28)|min(id,0xfffffffU);else h.overflow=1;
}
static void sspec(thread MklStrayHistory &h,device const uint *seq,uint count,uint surface,bool designed,bool extra){
  if(h.next_specular<0||uint(h.next_specular)>=count||seq[h.next_specular]!=surface)h.flags|=32;
  if(extra)h.flags^=64;h.next_specular+=(h.flags&64)?-1:1;
  if(!(h.flags&4)&&designed&&h.next_seq<count&&seq[h.next_seq]==surface)h.next_seq++;else h.flags|=4;
  if(extra){if(h.reflections<2)h.first[h.reflections]=min(surface,65534U);h.reflections=min(h.reflections+1,255U);}
}
static uint sclass(MklStrayHistory h,uint count){
  if(h.flags&1)return 2;if(h.flags&2)return 3;if(h.flags&8)return 4;
  if((h.reflections>=2||(h.flags&128))&&!(h.flags&96)&&h.next_specular==int(count)-1)return 1;
  if(h.reflections==0&&!(h.flags&132)&&(count==0||h.next_seq+1==count))return 0;
  return 4;
}
static SD stis(MklStrayMaterial m,device const float2 *table,SV spec,SV normal){
  if(m.bsdf_kind==0)return SD(0);
  if(m.tis_count==0)return SD(m.bsdf[0]);
  SD c=smin(SD(1),smax(SD(-1),sdot(spec,normal)));
  float beta=sv(ssqrt(smax(SD(0),(SD(1)-c)*(SD(1)+c))));
  float x=asin(clamp(beta,0.0f,1.0f))*(2.0f/M_PI_F)*float(m.tis_count-1);
  uint j=min(uint(floor(x)),m.tis_count-2);SD f(x-float(j)),f2=f*f,f3=f2*f;
  SD p1(table[m.tis_start+j]),p2(table[m.tis_start+j+1]);
  SD p0(table[m.tis_start+(j==0?1:j-1)]),p3=j+2<m.tis_count?SD(table[m.tis_start+j+2]):SD(2)*p2-p1;
  SD v=(SD(2)*f3-SD(3)*f2+SD(1))*p1+(f3-SD(2)*f2+f)*(p2-p0)*SD(0.5f)
      +(-SD(2)*f3+SD(3)*f2)*p2+(f3-f2)*(p3-p1)*SD(0.5f);
  return smin(SD(1),smax(SD(0),v));
}
static float sexpm1(float x){
  if(abs(x)>0.125f)return exp(x)-1.0f;
  float t=x,s=x;for(uint j=2;j<=9;j++){t=t*x/float(j);s=s+t;}return s;
}
static float slog1p(float x){
  float y=1.0f+x;return y==1.0f?x:log(y)*(x/(y-1.0f));
}
static float sabg(float b,float g,float rm,float u,thread float &accept){
  if(abs(g)<1e-12f){accept=1;return rm*sqrt(u);}
  if(g<0){float r=rm*sqrt(u);accept=(b+pow(rm,g))/(b+pow(r,g));return r;}
  float e=2-g,rc=b==0?0:exp(log(b)/g);
  if(rc>=rm){float r=rm*sqrt(u);accept=b/(b+pow(r,g));return r;}
  float l=rc>0?log(rm/rc):INFINITY;
  float w2=abs(e)<1e-9f?l:sexpm1(e*l)/e,p1=rc==0||!isfinite(w2)?0:0.5f/(0.5f+w2);
  if(u<p1){float r=rc*sqrt(u/p1);accept=b/(b+pow(r,g));return r;}
  float v=p1>0?(u-p1)/(1-p1):u;
  float lr=e>1e-9f?log(rm)+log(v+(1-v)*(rc>0?exp(-e*l):0))/e:
      (e < -1e-9f?log(rc)+slog1p(v*sexpm1(e*l))/e:log(rc)+v*l);
  float r=exp(lr),rg=pow(r,g);accept=rg/(b+rg);return r;
}
static bool sscatter(MklStrayMaterial m,SV spec,SV normal,uint2 seed,uint2 id,thread uint &draws,thread SV &out){
  SV a,b;sonb(normal,a,b);SD bx=sdot(spec,a),by=sdot(spec,b);
  float rm=1+sv(ssqrt(bx*bx+by*by)),sigma=sv(SD(m.bsdf[1]));
  SD tau(float2(6.2831855f,-1.7484555e-7f));
  for(uint k=0;k<4096;k++){
    SD u=suniform(seed,id,draws),r;float accept=1;
    if(m.bsdf_kind==1)r=ssqrt(u);
    else if(m.bsdf_kind==2)r=SD(sigma*sqrt(-2*slog1p(-sv(u)*(-sexpm1(-0.5f*(rm/sigma)*(rm/sigma))))));
    else r=SD(sabg(sv(SD(m.bsdf[1])),sv(SD(m.bsdf[2])),rm,sv(u),accept));
    SD sn,cs;ssincos(tau*suniform(seed,id,draws),sn,cs);
    SD x=r*cs,y=r*sn;if(m.bsdf_kind!=1){x=x+bx;y=y+by;}
    bool ok=m.bsdf_kind!=3||sv(suniform(seed,id,draws))<accept;
    SD radius=x*x+y*y;
    if(ok&&(m.bsdf_kind==1||sv(radius)<1)){
      out=sunit(sadd(sadd(sscale(a,x),sscale(b,y)),sscale(normal,ssqrt(smax(SD(0),SD(1)-radius)))));return true;
    }
  }return false;
}
static int skino(float blaze,uint2 seed,uint2 id,thread uint &draws){
  if(blaze==round(blaze))return int(blaze);
  int center=int(round(blaze));
  for(;;){
    float k=round(tan(M_PI_F*(sv(suniform(seed,id,draws))-0.5f)));
    if(k+float(center)>2147483520.0f||k+float(center)<-2147483520.0f)continue;
    int m=center+int(k);float q=k==0?2*atan(0.5f)/M_PI_F:atan2(1.0f,(k-0.5f)*(k+0.5f)+1)/M_PI_F;
    float v=float(m)-blaze,rem=v-round(v),s=v==0?1:sin(M_PI_F*rem)/(M_PI_F*v);
    if(sv(suniform(seed,id,draws))*4*q<s*s)return m;
  }
}
static int sorder(MklStrayMaterial m,float blaze,uint2 seed,uint2 id,thread uint &draws){
  if(m.doe==1)return m.selected;
  if(m.doe==2)return skino(blaze,seed,id,draws);
  for(;;){int n=int(m.levels),a=skino(blaze,seed,id,draws),r=((a%n)+n)%n;
    int k=skino(-float(r)/float(n),seed,id,draws);
    if(abs(float(k))*float(n)+float(r)<2147483520.0f)return k*n+r;
  }
}
struct SEvt {SV point,normal,refl,trans;uint mi,mo,prim,surface;bool dt,dr,stray,mechanical,tir;uint side;};
static uint sapply(thread MklStrayState &s,SD weight,uint branch,SEvt ev,MklStrayMaterial m,MklStrayPrimitive prim,
 device const uint *seq,uint seq_count,uint2 seed,thread uint &draws){
  bool reflected=branch<2;SV spec=reflected?ev.refl:ev.trans;
  uint2 id(s.ray.id[0],s.ray.id[1]);
  if(m.doe){
    int order=sorder(m,sv(SD(m.blaze[reflected?1+ev.side:0])),seed,id,draws);
    SV local=srotate(prim.rotation,ssub(ev.point,svec(prim.origin)));
    SD inv=SD(1)/SD(m.norm_radius),r2=(local.a[0]*local.a[0]+local.a[1]*local.a[1])*inv*inv,grad(0);
    for(int j=int(m.phase_count)-1;j>=0;j--)grad=grad*r2+SD(float(j+1))*SD(m.phase[j]);
    SD scale=SD(float(order))*SD(m.coatings[ev.side].wavelength)*SD(float2(0.001f,-4.749745e-11f))/SD(float2(6.2831855f,-1.7484555e-7f));
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
  SV dir=spec;
  if((branch&1)&&!sscatter(m,spec,sscale(ev.normal,reflected?SD(1):SD(-1)),seed,id,draws,dir))return 15;
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
static bool sroulette(thread MklStrayState &s,SD threshold,uint2 seed,thread uint &draws,thread MklStrayLedger &l){
  SD w(s.weight);if(sv(w)<=0)return false;if(sv(w-threshold)>=0)return true;
  if(sv(suniform(seed,uint2(s.ray.id[0],s.ray.id[1]),draws)*threshold-w)<0){schange(l,16,threshold-w);s.weight=threshold.x;return true;}
  schange(l,16,-w);s.weight=float2(0);return false;
}
kernel void mkl_stray_transport(device const MklStrayPrimitive *prims [[buffer(0)]],device const MklStrayNode *nodes [[buffer(1)]],
 device const MklStrayMaterial *materials [[buffer(2)]],device const MklCoatingLayer *layers [[buffer(3)]],
 device const MklStrayPass *passes [[buffer(4)]],device const MklStrayMedium *media [[buffer(5)]],device const float2 *tis [[buffer(6)]],device const uint *seq [[buffer(7)]],
 device const MklStrayRay *rays [[buffer(8)]],device MklStrayState *stack [[buffer(9)]],device MklStrayDetection *hits [[buffer(10)]],device MklStrayLedger *ledgers [[buffer(11)]],
 constant MklStraySettings &set [[buffer(12)]],constant MklStrayTransportParams &p [[buffer(13)]],constant MklStraySurfaceData *shapes [[buffer(14)]],uint i [[thread_position_in_grid]]){
  if(i>=p.count)return;uint base=i*p.stack_capacity,hbase=i*p.hit_capacity;uint2 seed(p.seed[0],p.seed[1]);
  MklStrayLedger l;
  if(p.initialize){l={};MklStrayState s={};s.ray=rays[i];s.weight=set.weight;s.medium=set.start_medium;s.history.first[0]=65535;s.history.first[1]=65535;
    MklStrayDetection empty={};for(uint j=0;j<p.hit_capacity;j++)hits[hbase+j]=empty;
    stack[base]=s;l.top=1;l.draws=s.ray.draws;l.wavelength=s.ray.wavelength;l.sums[20]=set.weight;
  }else l=ledgers[i];
  l.status=1;uint wl=l.wavelength;SD rr=SD(set.roulette)*SD(set.weight),ceiling=SD(set.ceiling)*SD(set.weight);
  for(uint step=0;step<p.budget&&l.top;step++){
    // Reserve the reference's largest weight-window fanout before consuming
    // any random words; a spill can resume this exact event.
    if(l.top+63>p.stack_capacity){l.status=2;break;}
    if(l.hits>=p.hit_capacity){l.status=3;break;}
    MklStrayState s=stack[base+--l.top];SD w(s.weight);if(sv(w)<=0)continue;
    if(s.events>=set.max_events){sbook(l,14,w);continue;}
    s.events++;l.events++;MklStrayHit hit=sclosest(prims,nodes,shapes,p.nodes,s.ray);
    MklStrayMedium med=media[s.medium*set.wavelengths+wl];
    if(hit.primitive==0xffffffff){sbook(l,med.ambient?12:13,w);continue;}
    if(sv(SD(med.alpha))>0){SD tau(stray_coating::rexp(stray_coating::RD((-SD(med.alpha)*SD(hit.distance)).x)).v);sbook(l,0,w*(SD(1)-tau));w=w*tau;s.weight=w.x;}
    MklStrayPrimitive prim=prims[hit.primitive];MklStrayMaterial m=materials[hit.primitive*set.wavelengths+wl];
    SV point=sadd(sworld(prim.rotation,svec(hit.point)),svec(prim.origin)),normal=sworld(prim.rotation,svec(hit.normal)),d=svec(s.ray.direction);
    uint side=sv(sdot(d,normal))<0?1:0;SV ni=sscale(normal,side?SD(1):SD(-1));
    SEvt ev;ev.point=point;ev.normal=ni;ev.refl=ssub(d,sscale(normal,SD(2)*sdot(d,normal)));ev.trans=d;ev.mi=s.medium;ev.mo=s.medium;ev.prim=hit.primitive;ev.surface=0;ev.dt=false;ev.dr=false;ev.stray=false;ev.mechanical=m.kind==0;ev.tir=false;ev.side=side;
    if(m.kind==4){
      if(set.detector_accept==2||set.detector_accept==side){MklStrayDetection h;h.history=s.history;h.weight=w.x;h.x=hit.point[0];h.y=hit.point[1];h.path_class=sclass(s.history,set.sequence_count);h.wavelength=wl;hits[hbase+l.hits++]=h;}
      else sbook(l,3,w);continue;
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
      if(mirror&&!designed){sbook(l,2,w);continue;}
      MklStrayPass pass=passes[chosen];MklStrayPrimitive clip=prim;clip.mask=pass.mask;clip.aperture[0]=pass.aperture[0];clip.aperture[1]=pass.aperture[1];
      if(!smask(clip,svec(hit.point))){if(!(s.history.flags&8))spush(s.history,7,pass.surface);s.history.flags|=8;}
      ev.surface=pass.surface;ev.dt=designed&&!mirror;ev.dr=designed&&mirror;ev.stray=!mirror;ev.mi=m.media[side];ev.mo=m.media[1-side];
      SD ci=smin(SD(1),smax(SD(0),-sdot(d,ni)));
      MklCoatingValue cv=stray_coating::revaluate(m.coatings[side],layers,stray_coating::RD(ci.x));
      if(cv.status){l.status=4;break;}
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
    if(branches==0){sbook(l,sink,w);continue;}
    uint2 id(s.ray.id[0],s.ray.id[1]);
    if(set.mode==2&&s.splits<set.max_splits&&branches>=2){
      sbook(l,sink,w*absorb);MklStrayState children[4];uint count=0;
      for(uint j=0;j<4;j++)if(sv(probs[j])>0){MklStrayState c=s;c.splits++;SD cw=w*probs[j];uint lost=sapply(c,cw,j,ev,m,prim,seq,set.sequence_count,seed,l.draws);
        if(lost==0xffffffff)children[count++]=c;else sbook(l,lost,cw);
      }
      if(count){sroulette(children[0],rr,seed,l.draws,l);for(uint j=count;j>0;j--)stack[base+l.top++]=children[j-1];}
      continue;
    }
    uint pick=0;SD next=w;
    if(set.mode==0){SD u=suniform(seed,id,l.draws);if(sv(u-absorb)<0){sbook(l,sink,w);continue;}SD acc=absorb;
      for(uint j=0;j<4;j++)if(sv(probs[j])>0){acc=acc+probs[j];pick=j;if(sv(u-acc)<0)break;}
    }else {
      sbook(l,sink,w*absorb);SD raw[4],z(0);for(uint j=0;j<4;j++){raw[j]=sv(probs[j])>0?smax(probs[j]/sum,SD(set.rare_floor)):SD(0);z=z+raw[j];}
      SD u=suniform(seed,id,l.draws),acc(0);for(uint j=0;j<4;j++)if(sv(probs[j])>0){acc=acc+raw[j]/z;pick=j;if(sv(u-acc)<0)break;}
      next=w*probs[pick]/(raw[pick]/z);schange(l,18,next-w*sum);
    }
    uint lost=sapply(s,next,pick,ev,m,prim,seq,set.sequence_count,seed,l.draws);
    if(lost!=0xffffffff){sbook(l,lost,next);continue;}
    if(!sroulette(s,rr,seed,l.draws,l))continue;
    uint copies=1;if(sv(SD(s.weight)-ceiling)>0){copies=uint(min(ceil(sv(SD(s.weight)/SD(set.weight))),64.0f));s.weight=(SD(s.weight)/SD(float(copies))).x;}
    for(uint j=0;j<copies;j++)stack[base+l.top++]=s;
  }
  if(l.top==0&&l.status==1)l.status=0;ledgers[i]=l;
}

kernel void mkl_stray_spill(device const MklStrayState *old [[buffer(0)]],device MklStrayState *next [[buffer(1)]],device const MklStrayLedger *ledger [[buffer(2)]],constant uint4 &p [[buffer(3)]],uint i [[thread_position_in_grid]]){
  if(i>=p.x)return;for(uint j=0;j<ledger[i].top;j++)next[i*p.z+j]=old[i*p.y+j];
}
