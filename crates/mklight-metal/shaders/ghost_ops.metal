// Ghost-specific sequence rules, with P1 local-frame geometry. P7 evaluates
// all step incidences in one batch before the separate s/p fold below.
#include "trace_shared.h"
kernel void mkl_ghost_trace(constant MklPackedHeader* headers [[buffer(0)]],
 constant MklPackedSurface* s [[buffer(1)]], constant float* pool [[buffer(2)]],
 device const MklGhostProbe* probes [[buffer(3)]], constant uint* offsets [[buffer(4)]],
 constant MklGhostStep* steps [[buffer(5)]], device MklGhostValue* output [[buffer(6)]],
 device MklCoatingQuery* queries [[buffer(7)]], device float2* factors [[buffer(8)]],
 constant MklGhostTraceParams& p [[buffer(9)]], device const uint* rows [[buffer(10)]], uint tid [[thread_position_in_grid]]) {
 uint i=p.offset+tid;if(i>=p.end)return;
 MklGhostProbe q=probes[i];MklGhostValue v={};v.weight_s=float2(1,0);v.weight_p=float2(1,0);v.internal=float2(1,0);
 uint plane=offsets[p.paths],a=offsets[q.path],r0=rows[i],span=rows[i+1]-r0,len=min(offsets[q.path+1]-a,span);
 MklDv pos,dir;for(uint j=0;j<3;j++){pos.v[j]=q.position[j];dir.v[j]=q.direction[j];}
 float2 dn=mkl_dnorm(dir);for(uint j=0;j<3;j++)dir.v[j]=mkl_df_div(dir.v[j],dn);
 float2 since=float2(0);bool counting=(headers[0].flags&MKL_SYS_OBJECT_AT_INFINITY)==0;
 for(uint k=0;k<len;k++){
  MklGhostStep st=steps[q.wavelength*plane+a+k];uint si=st.word&MKL_STEP_SURFACE_MASK;bool back=(st.word&MKL_STEP_BACKWARD)!=0;constant MklPackedSurface& surf=s[si];uint fail=si<<MKL_RAY_SURFACE_SHIFT;
  if(back)mkl_xf_backward(s[si],pos,dir);else mkl_xf_forward(s[si-1],pos,dir);
  MklHit hit;uint code=mkl_intersect<true>(surf,pool,pos,dir,p.trace,hit);if(code!=0){v.status=fail|code;break;}
  float2 t=mkl_df_add(hit.ta,hit.tb);if(fabs(t.x)>float(MKL_GHOST_MAX_SEGMENT_MM)){v.status=fail|MKL_GHOST_REFERENCE;break;}if(counting)since=mkl_df_add(since,t);
  if(st.physical!=0){if(counting&&p.reject_virtual!=0&&since.x< -1e-9f){v.status=fail|MKL_GHOST_VIRTUAL;break;}since=float2(0);counting=true;}
  pos=hit.x;
  if(!mkl_passes_aperture(surf,pos.v[0],pos.v[1])){v.status=fail|MKL_RAY_VIGNETTED;break;}
  if(st.channel==4&&(fabs(pos.v[0].x)>p.sensor[0].x||fabs(pos.v[1].x)>p.sensor[1].x)){v.status=fail|MKL_RAY_VIGNETTED;break;}
  float2 n1=mkl_index(s[back?si:si-1],q.wavelength),n2=mkl_index(s[back?si-1:si],q.wavelength);
  bool refl=st.channel==2||st.channel==4;if(refl)n2=n1;
  float2 cosine=mkl_df_div(mkl_ddot(dir,hit.g),mkl_dnorm(hit.g));if(cosine.x<0)cosine=mkl_df_neg(cosine);
  MklDv next=dir;
  if(st.channel==1||refl){
   if(surf.shape==MKL_SHAPE_DIFFRACTIVE&&st.channel!=4){
    MklJet<MklDf> phase=mkl_phase(surf,pool,MklDf{pos.v[0]},MklDf{pos.v[1]});
    if(!mkl_diffract(dir,hit.g,n1,n2,phase.x*MklDf{st.scale},phase.y*MklDf{st.scale},refl,next)){v.status=fail|MKL_RAY_EVANESCENT;break;}
   }else if(surf.shape==MKL_SHAPE_PARAXIAL&&st.channel!=4){
    MklDv kk=dir;if(refl)kk.v[2]=mkl_df_neg(kk.v[2]);float2 delta;
    if(!mkl_ideal_lens(pos.v[0],pos.v[1],kk,st.power,n1,n2,float2(0),next,delta)){v.status=fail|MKL_RAY_DEGENERATE;break;}
   }else if(refl){
    float3 gh=mkl_hi3(hit.g);float b= -2.0f*mkl_dot3(mkl_hi3(dir),gh)/mkl_dot3(gh,gh);
    for(uint j=0;j<3;j++)next.v[j]=mkl_df_add(dir.v[j],mkl_df_mul_f32(hit.g.v[j],b));next=mkl_renorm_along(next,hit.g);
   }else{
    float2 mu=mkl_df_div(n1,n2);float2 critical=mkl_df_sub(float2(1,0),mkl_df_mul(mkl_df_mul(mu,mu),mkl_df_sub(float2(1,0),mkl_df_mul(cosine,cosine))));
    if(fabs(critical.x)<=as_type<float>(MKL_GHOST_CRITICAL_BAND_BITS)){v.status=fail|MKL_GHOST_REFERENCE;break;}
    float ci,gamma;if(!mkl_refract(mkl_hi3(dir),hit.normal,n1.x,n2.x,ci,gamma)){v.status=fail|MKL_RAY_TIR;break;}
    float b=gamma/mkl_dnorm(hit.g).x;
    for(uint j=0;j<3;j++)next.v[j]=mkl_df_add(mkl_df_mul(dir.v[j],mu),mkl_df_mul_f32(hit.g.v[j],b));next=mkl_renorm_along(mkl_renorm_along(next,hit.g),hit.g);
   }
  }
  uint row=r0+k;queries[row]={st.coating,0,cosine};factors[2*row]=mkl_df_mul(st.alpha,t);factors[2*row+1]=st.efficiency;dir=next;v.done++;
 }
 for(uint k=v.done;k<span;k++){uint row=r0+k;queries[row]={};factors[2*row]=float2(0);factors[2*row+1]=float2(0);}
 for(uint j=0;j<3;j++){v.position[j]=pos.v[j];v.direction[j]=dir.v[j];}output[i]=v;
}
struct MklGhostWeightParams{uint end,offset,paths,reserved;float2 sensor;};
kernel void mkl_ghost_weights(device const MklGhostValue* values [[buffer(0)]],device const MklCoatingValue* coatings [[buffer(1)]],device const float2* factors [[buffer(2)]],constant uint* offsets [[buffer(3)]],constant MklGhostStep* steps [[buffer(4)]],device const MklGhostProbe* probes [[buffer(5)]],device MklGhostValue* output [[buffer(6)]],constant MklGhostWeightParams& p [[buffer(7)]],device const uint* rows [[buffer(8)]],uint tid [[thread_position_in_grid]]){
 uint i=p.offset+tid;if(i>=p.end)return;MklGhostValue v=values[i];MklGhostProbe q=probes[i];uint a=q.wavelength*offsets[p.paths]+offsets[q.path];float2 ws=float2(1,0),wp=ws,absorb=float2(0);
 for(uint k=0;k<v.done;k++){
  uint row=rows[i]+k,c=steps[a+k].channel;MklCoatingValue coat=coatings[row];if((c==1||c==2)&&coat.status!=MKL_RAD_OK){v.status=MKL_GHOST_REFERENCE;break;}
  float2 one=float2(1,0),x=one,y=one;
  if(c==1){x=coat.powers[2];y=coat.powers[3];}else if(c==2){x=coat.powers[0];y=coat.powers[1];}else if(c==3){x=mkl_df_sub(one,p.sensor);y=x;}else if(c==4){x=p.sensor;y=x;}
  float2 eta=factors[2*row+1];ws=mkl_df_mul(ws,mkl_df_mul(x,eta));wp=mkl_df_mul(wp,mkl_df_mul(y,eta));absorb=mkl_df_add(absorb,factors[2*row]);
 }
 v.weight_s=ws;v.weight_p=wp;v.internal=float2(mkl_exp(-absorb.x),0);output[i]=v;
}

static MklDv ghost_rotate(constant float* m,MklDv v,bool transpose){
 MklDv out;for(uint k=0;k<3;k++){float2 sum=float2(0);for(uint j=0;j<3;j++){uint c=transpose?j*3+k:k*3+j;float2 term=mkl_df_mul(float2(m[2*c],m[2*c+1]),v.v[j]);sum=j==0?term:mkl_df_add(sum,term);}out.v[k]=sum;}return out;
}
kernel void mkl_ghost_launch(device const float* chiefs [[buffer(0)]],constant MklAimField* fields [[buffer(1)]],device const MklGhostLaunchSample* samples [[buffer(2)]],device MklGhostProbe* output [[buffer(3)]],constant uint4& p [[buffer(4)]],uint tid [[thread_position_in_grid]]){
 uint i=p.y+tid;if(i>=p.x)return;MklGhostLaunchSample q=samples[i];constant MklAimField& f=fields[q.field];uint n=p.z,j=q.field;
 MklDv pos,dir;for(uint k=0;k<3;k++){pos.v[k]=float2(chiefs[k*n+j],chiefs[(9+k)*n+j]);dir.v[k]=float2(chiefs[(3+k)*n+j],chiefs[(12+k)*n+j]);}
 float2 scale=float2(f.pupil[2],f.pupil[3]);float2 sx=mkl_df_mul(q.pupil[0],scale),sy=mkl_df_mul(q.pupil[1],scale);
 if((f.flags&MKL_AIM_FINITE)==0){MklDv delta={{sx,sy,float2(0)}};float2 along=mkl_ddot(delta,dir);for(uint k=0;k<3;k++)pos.v[k]=mkl_df_sub(mkl_df_add(pos.v[k],delta.v[k]),mkl_df_mul(dir.v[k],along));}
 else if((f.flags&MKL_AIM_TELECENTRIC)!=0){MklDv d=ghost_rotate(f.rotation,dir,false);float2 az=mkl_dabs(d.v[2]);MklDv t={{mkl_df_add(mkl_df_div(d.v[0],az),sx),mkl_df_add(mkl_df_div(d.v[1],az),sy),float2(1,0)}};dir=ghost_rotate(f.rotation,t,true);float2 length=mkl_dnorm(dir);for(uint k=0;k<3;k++)dir.v[k]=mkl_df_div(dir.v[k],length);}
 else{MklDv pg=ghost_rotate(f.rotation,pos,false),dg=ghost_rotate(f.rotation,dir,false);for(uint k=0;k<3;k++)pg.v[k]=mkl_df_add(pg.v[k],float2(f.origin[2*k],f.origin[2*k+1]));float2 z=float2(f.pupil[0],f.pupil[1]);float2 t=mkl_df_div(mkl_df_sub(z,pg.v[2]),dg.v[2]);MklDv delta={{mkl_df_sub(mkl_df_add(mkl_df_add(pg.v[0],mkl_df_mul(dg.v[0],t)),sx),pg.v[0]),mkl_df_sub(mkl_df_add(mkl_df_add(pg.v[1],mkl_df_mul(dg.v[1],t)),sy),pg.v[1]),mkl_df_sub(z,pg.v[2])}};dir=ghost_rotate(f.rotation,delta,true);float2 length=mkl_dnorm(dir);for(uint k=0;k<3;k++)dir.v[k]=mkl_df_div(dir.v[k],length);MklDv axis;for(uint k=0;k<3;k++)axis.v[k]=float2(f.rotation[12+2*k],f.rotation[13+2*k]);if(mkl_ddot(axis,dir).x<0)for(uint k=0;k<3;k++)dir.v[k]=mkl_df_neg(dir.v[k]);}
 MklGhostProbe out;for(uint k=0;k<3;k++){out.position[k]=pos.v[k];out.direction[k]=dir.v[k];}out.path=q.path;out.wavelength=q.wavelength;output[i]=out;
}
struct MklGhostPixelParams{uint end,offset,nx,ny,maps,reserved;float2 extent[2];};
// Floor of hi+lo: an integral hi with a negative lo lies just below it.
static float ghost_floor(float2 v){float f=floor(v.x);return f==v.x&&v.y<0?f-1:f;}
kernel void mkl_ghost_pixel_keys(device const float2* hits [[buffer(0)]],device uint* keys [[buffer(1)]],device uint* ids [[buffer(2)]],device float2* weights [[buffer(3)]],constant MklGhostPixelParams& p [[buffer(4)]],uint tid [[thread_position_in_grid]]){
 uint i=p.offset+tid;if(i>=p.end)return;float x=ghost_floor(mkl_df_mul_f32(mkl_df_div(mkl_df_add(hits[5*i],p.extent[0]),mkl_df_mul_f32(p.extent[0],2)),float(p.nx)));float y=ghost_floor(mkl_df_mul_f32(mkl_df_div(mkl_df_add(hits[5*i+1],p.extent[1]),mkl_df_mul_f32(p.extent[1],2)),float(p.ny)));uint pixel=hits[5*i+4].x>0?uint(hits[5*i+4].x)-1:((x>=0&&y>=0&&x<float(p.nx)&&y<float(p.ny))?uint(y)*p.nx+uint(x):p.nx*p.ny);keys[i]=pixel==p.nx*p.ny?p.maps*p.nx*p.ny:uint(hits[5*i+3].x)*p.nx*p.ny+pixel;ids[i]=i;weights[i]=hits[5*i+2];
}


kernel void mkl_ghost_gather(device const MklGhostValue* values [[buffer(0)]],device const MklGhostLeaf* leaves [[buffer(1)]],device float2* output [[buffer(2)]],constant uint2& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){
 if(tid>=p.x)return;MklGhostLeaf l=leaves[p.y+tid];MklGhostValue v=values[l.row];uint d=5*l.destination;output[d+3]=float2(float(l.map),0);output[d+4]=float2(float(l.pixel),0);output[d]=v.position[0];output[d+1]=v.position[1];output[d+2]=mkl_df_mul(mkl_df_mul(mkl_df_mul_f32(mkl_df_add(v.weight_s,v.weight_p),0.5f),v.internal),l.weight);
}

struct MklGhostFirstOrderParams{uint end,offset;float2 sensor;};
kernel void mkl_ghost_first_order(device const MklGhostFirstOrderStep* steps [[buffer(0)]],device const uint* offsets [[buffer(1)]],device const float2* initial [[buffer(2)]],device float2* output [[buffer(3)]],device const MklCoatingValue* coatings [[buffer(4)]],device float2* throughput [[buffer(5)]],device uint* status [[buffer(6)]],constant MklGhostFirstOrderParams& p [[buffer(7)]],uint tid [[thread_position_in_grid]]){
 uint i=p.offset+tid;if(i>=p.end)return;float2 v[4];for(uint j=0;j<4;j++)v[j]=initial[4*i+j];float2 r=float2(1,0),t=r,absorption=float2(0);bool reference=false;
 for(uint k=offsets[i];k<offsets[i+1];k++){MklGhostFirstOrderStep s=steps[k];t=mkl_df_mul(t,s.efficiency);absorption=mkl_df_add(absorption,s.absorption);
  if(s.channel==MKL_GHOST_AXIAL_SENSOR)r=mkl_df_mul(r,p.sensor);else if(s.channel==MKL_GHOST_AXIAL_DETECT)t=mkl_df_mul(t,mkl_df_sub(float2(1,0),p.sensor));else if(s.channel!=MKL_GHOST_AXIAL_NONE){MklCoatingValue c=coatings[s.coating];if(c.status!=MKL_RAD_OK)reference=true;else{float2 cr=mkl_df_mul_f32(mkl_df_add(c.powers[0],c.powers[1]),0.5f),ct=mkl_df_mul_f32(mkl_df_add(c.powers[2],c.powers[3]),0.5f);if(s.channel==MKL_GHOST_AXIAL_REFLECT)r=mkl_df_mul(r,cr);else t=mkl_df_mul(t,s.channel==MKL_GHOST_AXIAL_MIRROR?cr:ct);}}
  v[1]=mkl_df_sub(v[1],mkl_df_mul(v[0],s.power));v[3]=mkl_df_sub(v[3],mkl_df_mul(v[2],s.power));for(uint j=0;j<4;j++)output[4*k+j]=v[j];float2 distance=mkl_df_div(s.distance,s.index);v[0]=mkl_df_add(v[0],mkl_df_mul(distance,v[1]));v[2]=mkl_df_add(v[2],mkl_df_mul(distance,v[3]));
 }
 t=mkl_df_mul(t,float2(mkl_exp(-absorption.x),0));reference|=fabs(v[1].x)<=as_type<float>(MKL_GHOST_FIRSTORDER_SLOPE_BITS);reference|=!(all(isfinite(r))&&all(isfinite(t)));for(uint j=0;j<4;j++)reference|=!all(isfinite(v[j]));throughput[2*i]=r;throughput[2*i+1]=t;status[i]=reference?MKL_GHOST_REFERENCE:0;
}

kernel void mkl_ghost_segment_flags(device const uint* keys [[buffer(0)]],device uint* output [[buffer(1)]],constant uint4& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){
 uint i=p.y+tid;if(i>=p.x)return;output[i]=(keys[i]<p.z&&(i==0||keys[i]!=keys[i-1]))?1:0;
}
kernel void mkl_ghost_pixel_segments(device const uint* keys [[buffer(0)]],device const uint* prefix [[buffer(1)]],device uint* offsets [[buffer(2)]],device uint* pixels [[buffer(3)]],device atomic_uint* count [[buffer(4)]],constant uint4& p [[buffer(5)]],uint tid [[thread_position_in_grid]]){
 uint i=p.y+tid;if(i==0){uint total=p.x==0?0:prefix[p.x-1];atomic_store_explicit(count,total,memory_order_relaxed);uint lo=0,hi=p.x;while(lo<hi){uint m=lo+(hi-lo)/2;if(keys[m]<p.z)lo=m+1;else hi=m;}offsets[total]=lo;}
 if(i>=p.x||keys[i]>=p.z||(i>0&&keys[i]==keys[i-1]))return;uint dest=prefix[i]-1;offsets[dest]=i;pixels[dest]=keys[i];uint lo=i,hi=p.x;while(lo<hi){uint m=lo+(hi-lo)/2;if(keys[m]<=keys[i])lo=m+1;else hi=m;}atomic_fetch_max_explicit(count+1,lo-i,memory_order_relaxed);
}
kernel void mkl_ghost_map_scatter(device const float2* values [[buffer(0)]],device const uint* pixels [[buffer(1)]],device float2* output [[buffer(2)]],constant uint4& p [[buffer(3)]],uint tid [[thread_position_in_grid]]){
 uint i=p.y+tid;if(i>=p.x)return;if(p.z==0)output[i]=float2(0);else output[pixels[i]]=values[i];
}

kernel void mkl_ghost_reference_count(device const MklGhostValue* values [[buffer(0)]],device atomic_uint* count [[buffer(1)]],constant uint2& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){uint i=p.y+tid;if(i<p.x&&(values[i].status&MKL_RAY_CODE_MASK)==MKL_GHOST_REFERENCE)atomic_fetch_add_explicit(count,1u,memory_order_relaxed);}

kernel void mkl_ghost_moment_terms(device const float2* hits [[buffer(0)]],device float2* terms [[buffer(1)]],constant uint4& p [[buffer(2)]],uint tid [[thread_position_in_grid]]){uint i=p.y+tid;if(i>=p.x)return;for(uint k=0;k<3;k++)terms[k*p.z+i]=hits[5*i+k];}
