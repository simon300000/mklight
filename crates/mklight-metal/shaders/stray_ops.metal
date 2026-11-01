#include "mklight_common.h"
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
static bool sintersect(MklStrayPrimitive p,MklStrayRay ray,SD limit,thread MklStrayHit &out){
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
 device MklStrayHit *output [[buffer(3)]],constant MklStrayParams &p [[buffer(4)]],uint tid [[thread_position_in_grid]]){
  uint index=p.offset+tid;if(index>=p.end)return;MklStrayRay ray=rays[index];
  MklStrayHit out={};out.primitive=0xffffffff;SD limit(FLT_MAX);uint i=0;
  while(i<p.nodes){
    MklStrayNode node=nodes[i];SD entry;
    if(!sslab(node.lower,node.upper,svec(ray.origin),svec(ray.direction),limit,entry)){i=node.escape;continue;}
    if(node.primitive!=0xffffffff){
      MklStrayHit h;
      if(sintersect(prims[node.primitive],ray,limit,h)){
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
