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
    if(dz.x==0&&dz.y==0)return float2(0);
    WVec q=wdir(p,k);float2 gamma=-q.z,r=wdf(p.radius);
    float2 s2=mkl_df_add(mkl_df_mul(q.x,q.x),mkl_df_mul(q.y,q.y));
    float2 dr;
    if(r.x==0) dr=mkl_df_div(mkl_df_mul(dz,s2),mkl_df_add_f32(gamma,1));
    else {
        float2 rd=mkl_df_mul_f32(mkl_df_mul(r,dz),2);
        float2 delta=mkl_df_add(mkl_df_mul(rd,gamma),mkl_df_mul(dz,dz));
        float2 path=mkl_df_sqrt(mkl_df_add(mkl_df_mul(r,r),delta));
        float2 axial=mkl_df_add(r,dz);if(axial.x<0)axial=-axial;
        if(q.x.x==0&&q.y.x==0)return float2(0);
        dr=mkl_df_div(mkl_df_mul(rd,s2),mkl_df_mul(mkl_df_add_f32(gamma,1),mkl_df_add(axial,path)));
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
kernel void mkl_wave_focus(device const float *cov [[buffer(1)]],device const float2 *opd [[buffer(2)]],device const float2 *shifts [[buffer(3)]],device float2 *out [[buffer(4)]],constant MklWaveParams &p [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=id+p.first,N=p.n*p.n;out[k]=cov[k%N]>0?mkl_df_add(opd[k%N],wfocus(p,k%N,shifts[k/N])):opd[k%N];
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
kernel void mkl_wave_focus_correlation(device const float *a [[buffer(0)]],device const float *c [[buffer(1)]],device const float2 *w [[buffer(2)]],device float2 *out [[buffer(4)]],device float2 *acc [[buffer(6)]],constant MklWaveParams &p [[buffer(7)]],device const float *stencil [[buffer(8)]],device const float2 *shifts [[buffer(9)]],device const float2 *cached [[buffer(10)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint o=id+p.first,N=p.n*p.n,frequency=o%p.field_side;
    float2 dz=p.unit_energy?shifts[o/p.field_side]:float2(0);float2 re=p.sample_first?acc[3*id]:float2(0),im=p.sample_first?acc[3*id+1]:float2(0),sum=p.sample_first?acc[3*id+2]:float2(0);
    for(uint k=p.sample_first;k<p.sample_first+p.sample_count;++k) {
        if(c[k]<=0)continue;uint off=(frequency*N+k)*5;
        float ix=stencil[off],iy=stencil[off+1],qz=stencil[off+2],gamma=stencil[off+3],srcz=stencil[off+4];
        float energy=c[k]*a[k]*a[k]/(gamma*srcz);sum=mkl_df_add_f32(sum,energy);
        if(!isfinite(ix)||!isfinite(iy))continue;
        float fx=floor(ix),fy=floor(iy),tx=ix-fx,ty=iy-fy;
        float ar=0,cov=0;float2 phase=0;
        for(uint b=0;b<2;++b)for(uint aa=0;aa<2;++aa) {
            int x=int(fx)+int(aa),y=int(fy)+int(b);if(x<0||y<0||x>=int(p.n)||y>=int(p.n))continue;
            uint j=uint(y)*p.n+uint(x);float wt=(aa?tx:1-tx)*(b?ty:1-ty)*c[j];if(wt==0)continue;
            ar+=wt*a[j]/(p.shifted?-wf(wdir(p,j).z):1.0f);cov+=wt;phase=mkl_df_add(phase,mkl_df_mul_f32((p.reserved[0]?cached[(o/p.field_side)*N+j]:mkl_df_add(w[j],wfocus(p,j,dz))),wt));
        }
        float deficit=(1-c[k])*energy*max(0.0f,1-abs(ix-float(k%p.n)))*max(0.0f,1-abs(iy-float(k/p.n)));
        float2 term=0;
        if(cov>0) {float2 ph=mkl_df_sub(mkl_df_div(phase,float2(cov,0)),(p.reserved[0]?cached[(o/p.field_side)*N+k]:mkl_df_add(w[k],wfocus(p,k,dz))));float m=c[k]*a[k]/gamma*ar/(p.shifted?1.0f:qz);term=m*wcis(ph);}
        re=mkl_df_add_f32(re,term.x+deficit);im=mkl_df_add_f32(im,term.y);
    }
    acc[3*id]=re;acc[3*id+1]=im;acc[3*id+2]=sum;
    if(p.sample_first+p.sample_count==N)out[o]=float2(wf(mkl_df_div(re,sum)),wf(mkl_df_div(im,sum)));
}
// Structured lattice resampling. Triangle IDs retain logical cell order even
// when a cell is clipped. Integer minimum makes overlap ownership deterministic.
static float2 wrcoord(uint i,uint ns) {return mkl_df_add_f32(mkl_df_div(float2(float(2*i+1),0),float2(float(ns),0)),-1);}
static float2 wrdist(device const float2 *v,uint a,uint b) {float2 x=mkl_df_sub(v[4*a],v[4*b]),y=mkl_df_sub(v[4*a+1],v[4*b+1]);return mkl_df_add(mkl_df_mul(x,x),mkl_df_mul(y,y));}
static bool wrbary(device const float2 *v,uint3 ids,float2 u,float2 w,thread float2 *l) {
    float2 x0=v[4*ids.x],y0=v[4*ids.x+1];
    float2 ax=mkl_df_sub(v[4*ids.y],x0),ay=mkl_df_sub(v[4*ids.y+1],y0),bx=mkl_df_sub(v[4*ids.z],x0),by=mkl_df_sub(v[4*ids.z+1],y0);
    float2 det=mkl_df_sub(mkl_df_mul(ax,by),mkl_df_mul(ay,bx));if(det.x==0)return false;
    float2 x=mkl_df_sub(u,x0),y=mkl_df_sub(w,y0);
    l[1]=mkl_df_div(mkl_df_sub(mkl_df_mul(x,by),mkl_df_mul(y,bx)),det);
    l[2]=mkl_df_div(mkl_df_sub(mkl_df_mul(ax,y),mkl_df_mul(ay,x)),det);
    l[0]=mkl_df_sub(mkl_df_sub(float2(1,0),l[1]),l[2]);return true;
}
static bool wrinside(constant MklWaveResampleParams &p,device const float2 *edges,float2 u,float2 v) {
    float2 r=mkl_df_add(mkl_df_mul(u,u),mkl_df_mul(v,v));
    if(p.mask==1)return wf(r)<=1;
    if(p.mask==2)return wf(r)<=1&&wf(r)>=p.aperture[0]*p.aperture[0];
    if(p.mask==3)return abs(wf(u))<=p.aperture[0]&&abs(wf(v))<=p.aperture[1];
    bool inside=false;
    for(uint i=0;i<p.edges;++i) {
        float2 x0=edges[4*i],y0=edges[4*i+1],x1=edges[4*i+2],y1=edges[4*i+3];
        if((mkl_df_sub(y0,v).x>0)!=(mkl_df_sub(y1,v).x>0)) {
            float2 xx=mkl_df_add(x0,mkl_df_div(mkl_df_mul(mkl_df_sub(v,y0),mkl_df_sub(x1,x0)),mkl_df_sub(y1,y0)));
            if(mkl_df_sub(u,xx).x<0)inside=!inside;
        }
    }
    return inside;
}
// Cache the same half-open even-odd crossings once per supersample row.
kernel void mkl_wave_mask_rows(device const float2 *edges [[buffer(2)]],constant MklWaveResampleParams &p [[buffer(9)]],device float2 *crossings [[buffer(10)]],device uint *counts [[buffer(11)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint row=p.first+id,count=0;float2 v=wrcoord(row,p.n*p.supersample);
    for(uint i=0;i<p.edges;++i){float2 x0=edges[4*i],y0=edges[4*i+1],x1=edges[4*i+2],y1=edges[4*i+3];if((mkl_df_sub(y0,v).x>0)!=(mkl_df_sub(y1,v).x>0)){crossings[row*p.edges+count++]=mkl_df_add(x0,mkl_df_div(mkl_df_mul(mkl_df_sub(v,y0),mkl_df_sub(x1,x0)),mkl_df_sub(y1,y0)));}}counts[row]=count;
}
kernel void mkl_wave_owner_clear(device atomic_uint *owner [[buffer(4)]],constant MklWaveResampleParams &p [[buffer(9)]],uint id [[thread_position_in_grid]]) {
    if(id<p.count)atomic_store_explicit(owner+p.first+id,0xffffffffu,memory_order_relaxed);
}
kernel void mkl_wave_triangles(device const float2 *nodes [[buffer(0)]],device const uint *valid [[buffer(1)]],device uint *tris [[buffer(3)]],constant MklWaveResampleParams &p [[buffer(9)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint cell=p.first+id,k=(cell/(p.nx-1))*p.nx+cell%(p.nx-1),ids[4]={k,k+1,k+p.nx+1,k+p.nx};uint v[4],n=0;
    for(uint j=0;j<4;++j)if(valid[ids[j]])v[n++]=ids[j];
    uint3 a=uint3(0xffffffffu),b=a;
    if(n==3)a=uint3(v[0],v[1],v[2]);
    if(n==4) {if(mkl_df_sub(wrdist(nodes,v[0],v[2]),wrdist(nodes,v[1],v[3])).x<=0) {a=uint3(v[0],v[1],v[2]);b=uint3(v[0],v[2],v[3]);}else{a=uint3(v[0],v[1],v[3]);b=uint3(v[1],v[2],v[3]);}}
    tris[6*cell]=a.x;tris[6*cell+1]=a.y;tris[6*cell+2]=a.z;tris[6*cell+3]=b.x;tris[6*cell+4]=b.y;tris[6*cell+5]=b.z;
}
kernel void mkl_wave_raster(device const float2 *nodes [[buffer(0)]],device const uint *tris [[buffer(3)]],device atomic_uint *owner [[buffer(4)]],constant MklWaveResampleParams &p [[buffer(9)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint t=p.first+id;uint3 ids=uint3(tris[3*t],tris[3*t+1],tris[3*t+2]);if(ids.x==0xffffffffu)return;
    uint ns=p.n*p.supersample;float2 lo=float2(INFINITY),hi=float2(-INFINITY);
    for(uint j=0;j<3;++j){float2 q=float2(wf(nodes[4*ids[j]]),wf(nodes[4*ids[j]+1]));lo=min(lo,q);hi=max(hi,q);}
    int2 a=int2(max(ceil((lo+1)*float(ns)*0.5f-0.5f),float2(0))),b=int2(min(floor((hi+1)*float(ns)*0.5f-0.5f),float2(float(ns)-1)));
    for(int y=a.y;y<=b.y;++y)for(int x=a.x;x<=b.x;++x){float2 l[3];if(wrbary(nodes,ids,wrcoord(uint(x),ns),wrcoord(uint(y),ns),l)&&l[0].x>=-1e-12f&&l[1].x>=-1e-12f&&l[2].x>=-1e-12f)atomic_fetch_min_explicit(owner+uint(y)*ns+uint(x),t,memory_order_relaxed);}
}
// Parallel exact neighbours within 256-column blocks. An extrapolation of at
// most 128 subpoints can reach only an adjacent block; its summary is sufficient.
kernel void mkl_wave_nearest_blocks(device const uint *owner [[buffer(4)]],constant MklWaveResampleParams &p [[buffer(9)]],device uint2 *nearest [[buffer(12)]],device uint2 *blocks [[buffer(13)]],uint3 gid [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]){
    uint ns=p.n*p.supersample,columns=(ns+255)/256,block=p.first+gid.x,row=block/columns,x=(block%columns)*256+lane;bool valid=x<ns&&owner[row*ns+x]!=0xffffffffu;threadgroup uint2 v[256];v[lane]=uint2(valid?x+1:0,valid?x:0xffffffffu);threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint stride=1;stride<256;stride*=2){uint2 q=v[lane];if(lane>=stride)q.x=max(q.x,v[lane-stride].x);if(lane+stride<256)q.y=min(q.y,v[lane+stride].y);threadgroup_barrier(mem_flags::mem_threadgroup);v[lane]=q;threadgroup_barrier(mem_flags::mem_threadgroup);}
    if(x<ns)nearest[row*ns+x]=uint2(v[lane].x?v[lane].x-1:0xffffffffu,v[lane].y);if(lane==0)blocks[block]=uint2(v[0].y,v[255].x?v[255].x-1:0xffffffffu);
}
static uint2 wrpair(constant MklWaveResampleParams &p,device const uint2 *nearest,device const uint2 *blocks,uint x,uint y){uint ns=p.n*p.supersample,columns=(ns+255)/256,block=x/256;uint2 pair=nearest[y*ns+x];if(pair.x==0xffffffffu&&block>0)pair.x=blocks[y*columns+block-1].y;if(pair.y==0xffffffffu&&block+1<columns)pair.y=blocks[y*columns+block+1].x;return pair;}
kernel void mkl_wave_resample(device const float2 *nodes [[buffer(0)]],device const float2 *edges [[buffer(2)]],device const uint *tris [[buffer(3)]],device const uint *owner [[buffer(4)]],device float *amp [[buffer(5)]],device float *cov [[buffer(6)]],device float2 *opd [[buffer(7)]],device float *diag [[buffer(8)]],constant MklWaveResampleParams &p [[buffer(9)]],device const float2 *crossings [[buffer(10)]],device const uint *counts [[buffer(11)]],device const uint2 *nearest [[buffer(12)]],device const uint2 *blocks [[buffer(13)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=p.first+id,s=p.supersample,ns=p.n*s,N=p.n*p.n;float2 a=0,w=0;uint count=0,unresolved=0;float gap=0,pitch=1;
    for(uint sy=0;sy<s;++sy)for(uint sx=0;sx<s;++sx) {
        uint x=(k%p.n)*s+sx,y=(k/p.n)*s+sy,q=y*ns+x;float2 u=wrcoord(x,ns),v=wrcoord(y,ns);
        if(p.mask){bool inside;if(p.reserved[0]){inside=false;for(uint j=0;j<counts[y];++j)if(mkl_df_sub(u,crossings[y*p.edges+j]).x<0)inside=!inside;}else inside=wrinside(p,edges,u,v);if(!inside)continue;}bool missing=owner[q]==0xffffffffu;
        if(missing) {
            if(!p.mask)continue;uint2 same=wrpair(p,nearest,blocks,x,y);uint range=min(uint(MKL_WAVE_MAX_EXTRAPOLATION),min(same.x==0xffffffffu?0xffffffffu:x-same.x,same.y==0xffffffffu?0xffffffffu:same.y-x));uint best=range+1,bestrow=0xffffffffu;
            uint lo=y>range?y-range:0,hi=min(y+range,ns-1);
            for(uint yy=lo;yy<=hi;++yy){uint dy=uint(abs(int(yy)-int(y)));if(dy>best)continue;uint2 pair=wrpair(p,nearest,blocks,x,yy);uint dx=min(pair.x==0xffffffffu?0xffffffffu:x-pair.x,pair.y==0xffffffffu?0xffffffffu:pair.y-x),d=max(dx,dy);if(d<best){best=d;bestrow=yy;}}
            if(bestrow==0xffffffffu){++unresolved;continue;}
            q=bestrow*ns+wrpair(p,nearest,blocks,x>best?x-best:0,bestrow).y;
        }
        uint t=owner[q];uint3 ids=uint3(tris[3*t],tris[3*t+1],tris[3*t+2]);float2 l[3];if(!wrbary(nodes,ids,u,v,l))continue;
        float2 ww=0,aa=0;for(uint j=0;j<3;++j){ww=mkl_df_add(ww,mkl_df_mul(l[j],nodes[4*ids[j]+2]));aa=mkl_df_add(aa,mkl_df_mul(l[j],nodes[4*ids[j]+3]));}
        w=mkl_df_add(w,ww);if(aa.x>0)a=mkl_df_add(a,aa);++count;
        if(missing){float dx=float(x)-float(q%ns),dy=float(y)-float(q/ns),g=sqrt(dx*dx+dy*dy)*2/float(ns);float e0=mkl_df_sqrt(wrdist(nodes,ids.x,ids.y)).x,e1=mkl_df_sqrt(wrdist(nodes,ids.y,ids.z)).x,e2=mkl_df_sqrt(wrdist(nodes,ids.z,ids.x)).x;float h=max(min(e0,e1),min(max(e0,e1),e2));if(g/h>gap/pitch){gap=g;pitch=h;}}
    }
    cov[k]=float(count)/float(s*s);amp[k]=count?wf(mkl_df_div(a,float2(float(count),0))):0;opd[k]=count?mkl_df_div(w,float2(float(count),0)):float2(0);
    diag[k]=gap;diag[N+k]=pitch;diag[2*N+k]=float(unresolved);
}
static float2 wzernike(float2 x,float2 y,uint n,int m,bool standard) {
    uint am=uint(abs(m)),degree=(n-am)/2;float b=float(am);float2 one=float2(1,0),t=mkl_df_add_f32(mkl_df_mul_f32(mkl_df_add(mkl_df_mul(x,x),mkl_df_mul(y,y)),2),-1);
    float2 prev=one,cur=mkl_df_add(one,mkl_df_mul_f32(mkl_df_sub(t,one),(b+2)*0.5f));
    for(uint k=1;k<degree;++k){float f=float(k),s=2*f+b,a1=2*(f+1)*(f+b+1)*s,a3=2*f*(f+b)*(s+2);float2 a2=mkl_df_mul_f32(mkl_df_add_f32(mkl_df_mul_f32(t,(s+2)*s),-b*b),s+1);float2 next=mkl_df_div(mkl_df_sub(mkl_df_mul(a2,cur),mkl_df_mul_f32(prev,a3)),float2(a1,0));prev=cur;cur=next;}
    float2 re=one,im=0;for(uint k=0;k<am;++k){float2 r=mkl_df_sub(mkl_df_mul(re,x),mkl_df_mul(im,y)),i=mkl_df_add(mkl_df_mul(re,y),mkl_df_mul(im,x));re=r;im=i;}
    float2 z=mkl_df_mul(m>=0?re:im,degree?cur:one);if(standard)z=mkl_df_mul(z,mkl_df_sqrt(float2(float(n+1)*(m==0?1:2),0)));return z;
}
kernel void mkl_wave_zernike(device const float2 *samples [[buffer(0)]],device const uint2 *terms [[buffer(1)]],device float2 *matrix [[buffer(4)]],device float2 *rhs [[buffer(5)]],constant MklWaveParams &p [[buffer(7)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint row=p.first+id;float2 weight=samples[4*row+3];
    rhs[row]=weight.x>0?mkl_df_mul(samples[4*row+2],weight):float2(0);
    for(uint j=0;j<p.field_side;++j)matrix[row*p.field_side+j]=weight.x>0?mkl_df_mul(weight,wzernike(samples[4*row],samples[4*row+1],terms[j].x,as_type<int>(terms[j].y),p.size!=0)):float2(0);
}

kernel void mkl_wave_diagnostics(device const float *c [[buffer(0)]],device const float2 *w [[buffer(1)]],device const float *d [[buffer(2)]],device float *part [[buffer(3)]],constant uint *p [[buffer(5)]],uint block [[thread_position_in_grid]]) {
    uint n=p[0],nn=n*n;if(block>=p[1])return;
    float a[9]={0,0,0,0,1,float(n),float(n),0,0};float2 area=0;
    for(uint k=block*256;k<min(nn,(block+1)*256);++k) {
        area=mkl_df_add_f32(area,c[k]);
        a[1]+=float(d[2*nn+k]>0);
        if(d[k]/d[nn+k]>a[3]/a[4]){a[3]=d[k];a[4]=d[nn+k];}
        if(c[k]<=0)continue;
        a[0]+=1;uint ix=k%n,iy=k/n;
        a[5]=min(a[5],float(ix));a[6]=min(a[6],float(iy));a[7]=max(a[7],float(ix));a[8]=max(a[8],float(iy));
        uint next[2]={ix+1<n?k+1:k,iy+1<n?k+n:k};
        for(uint t=0;t<2;++t)if(c[next[t]]>0)a[2]=max(a[2],abs(wf(mkl_df_sub(w[next[t]],w[k]))));
    }
    for(uint j=0;j<9;++j)part[block*10+j]=a[j];part[block*10+9]=wf(area);
}
kernel void mkl_wave_diagnostics_finish(device const float *part [[buffer(3)]],device float2 *out [[buffer(4)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id)return;float2 area=0;float a[9]={0,0,0,0,1,float(p[0]),float(p[0]),0,0};
    for(uint i=0;i<p[1];++i){device const float *b=part+10*i;area=mkl_df_add_f32(area,b[9]);
        a[0]+=b[0];a[1]+=b[1];a[2]=max(a[2],b[2]);
        if(b[3]/b[4]>a[3]/a[4]){a[3]=b[3];a[4]=b[4];}
        for(uint j=5;j<7;++j)a[j]=min(a[j],b[j]);for(uint j=7;j<9;++j)a[j]=max(a[j],b[j]);
    }
    for(uint j=0;j<9;++j)out[j]=float2(a[j],0);out[9]=area;
}
kernel void mkl_wave_statistics_prepare(device const float *a [[buffer(0)]],device const float *c [[buffer(1)]],device const float2 *w [[buffer(2)]],device float2 *part [[buffer(3)]],device float2 *samples [[buffer(4)]],constant uint *p [[buffer(6)]],uint block [[thread_position_in_grid]]) {
    uint n=p[0],nn=n*n;if(block>=p[1])return;float2 sine=float2(as_type<float>(p[2]),as_type<float>(p[3]));float2 s[4]={0,0,0,0};
    for(uint k=block*256;k<min(nn,(block+1)*256);++k){
        float2 u=mkl_df_add_f32(mkl_df_div(float2(float(2*(k%n)+1),0),float2(float(n),0)),-1);
        float2 v=mkl_df_add_f32(mkl_df_div(float2(float(2*(k/n)+1),0),float2(float(n),0)),-1);
        float2 gamma=mkl_df_sqrt(mkl_df_sub(float2(1,0),mkl_df_mul(mkl_df_mul(sine,sine),mkl_df_add(mkl_df_mul(u,u),mkl_df_mul(v,v)))));
        float2 weight=c[k]>0?mkl_df_div(mkl_df_mul(float2(c[k],0),float2(a[k],0)),gamma):float2(0);
        samples[4*k]=u;samples[4*k+1]=v;float2 phase=c[k]>0?w[k]:float2(0);samples[4*k+2]=phase;samples[4*k+3]=mkl_df_sqrt(weight);
        float2 z=wcis(phase);float2 vals[4]={weight,mkl_df_mul(weight,phase),mkl_df_mul_f32(weight,z.x),mkl_df_mul_f32(weight,z.y)};
        for(uint j=0;j<4;++j)s[j]=mkl_df_add(s[j],vals[j]);
    }for(uint j=0;j<4;++j)part[6*block+j]=s[j];
}
kernel void mkl_wave_statistics_sum(device const float2 *part [[buffer(3)]],device float2 *s [[buffer(5)]],constant uint *p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    if(id)return;for(uint j=0;j<4;++j){float2 v=0;for(uint i=0;i<p[1];++i)v=mkl_df_add(v,part[6*i+j]);s[j]=v;}
}
static bool wless(float2 a,float2 b){return a.x<b.x||(a.x==b.x&&a.y<b.y);}
kernel void mkl_wave_statistics_residual(device const float2 *samples [[buffer(0)]],device const float2 *s [[buffer(1)]],device const float2 *q [[buffer(2)]],device float2 *part [[buffer(3)]],constant uint *p [[buffer(6)]],uint block [[thread_position_in_grid]]) {
    uint nn=p[0]*p[0];if(block>=p[1])return;float2 mean=mkl_df_div(s[1],s[0]);
    float2 a[6]={float2(0),float2(0),float2(INFINITY,0),float2(-INFINITY,0),float2(INFINITY,0),float2(-INFINITY,0)};
    for(uint k=block*256;k<min(nn,(block+1)*256);++k){device const float2 *r=samples+4*k;if(r[3].x<=0)continue;
        float2 weight=mkl_df_mul(r[3],r[3]),r0=mkl_df_sub(r[2],mean),r1=mkl_df_sub(mkl_df_sub(mkl_df_sub(r[2],q[0]),mkl_df_mul(q[1],r[0])),mkl_df_mul(q[2],r[1]));
        a[0]=mkl_df_add(a[0],mkl_df_mul(mkl_df_mul(weight,r0),r0));a[1]=mkl_df_add(a[1],mkl_df_mul(mkl_df_mul(weight,r1),r1));
        if(wless(r0,a[2]))a[2]=r0;if(wless(a[3],r0))a[3]=r0;if(wless(r1,a[4]))a[4]=r1;if(wless(a[5],r1))a[5]=r1;
    }for(uint j=0;j<6;++j)part[6*block+j]=a[j];
}
kernel void mkl_wave_statistics_finish(device const float2 *s [[buffer(1)]],device const float2 *part [[buffer(3)]],device float2 *out [[buffer(4)]],constant uint *p [[buffer(6)]],uint id [[thread_position_in_grid]]) {
    if(id)return;float2 a[6];for(uint j=0;j<6;++j)a[j]=part[j];
    for(uint i=1;i<p[1];++i){device const float2 *b=part+6*i;a[0]=mkl_df_add(a[0],b[0]);a[1]=mkl_df_add(a[1],b[1]);for(uint j=2;j<6;j+=2){if(wless(b[j],a[j]))a[j]=b[j];if(wless(a[j+1],b[j+1]))a[j+1]=b[j+1];}}
    out[0]=mkl_df_sqrt(mkl_df_div(a[0],s[0]));out[1]=mkl_df_sub(a[3],a[2]);out[2]=mkl_df_sqrt(mkl_df_div(a[1],s[0]));out[3]=mkl_df_sub(a[5],a[4]);out[4]=mkl_df_div(mkl_df_add(mkl_df_mul(s[2],s[2]),mkl_df_mul(s[3],s[3])),mkl_df_mul(s[0],s[0]));
}
static void wsample(device uint *s,float2 x,float2 y,uint field,uint k){s[0]=as_type<uint>(x.x);s[1]=as_type<uint>(x.y);s[2]=as_type<uint>(y.x);s[3]=as_type<uint>(y.y);s[4]=field;s[5]=k;s[6]=0;s[7]=as_type<uint>(1.f);}
kernel void mkl_wave_lattice(device uint *samples [[buffer(3)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]){
    if(id>=p[7])return;uint k=p[6]+id,s=p[0],per=s*s,i=k%per%s,j=k%per/s;
    float2 h=mkl_df_div(float2(2,0),float2(float(s-3),0));float c=float(s-1)*.5f;
    wsample(samples+8*k,mkl_df_mul(mkl_df_sub(float2(float(i),0),float2(c,0)),h),mkl_df_mul(mkl_df_sub(float2(float(j),0),float2(c,0)),h),k/per,k);
}
kernel void mkl_wave_survival(device const float2 *samples [[buffer(0)]],device const uint *status [[buffer(1)]],device uint *out [[buffer(3)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]){
    if(id>=p[7])return;uint k=p[6]+id;float2 x=samples[4*k],y=samples[4*k+1];float2 r=mkl_df_add(mkl_df_mul(x,x),mkl_df_mul(y,y));out[k]=uint(status[k]==0&&wless(r,float2(1,1.0001e-12f)));
}
kernel void mkl_wave_rim(device const float2 *edges [[buffer(0)]],device const float2 *prev [[buffer(1)]],device const uint *status [[buffer(2)]],device float2 *state [[buffer(3)]],device uint *samples [[buffer(4)]],constant uint *p [[buffer(5)]],device const uint *fields [[buffer(6)]],uint id [[thread_position_in_grid]]){
    if(id>=p[7])return;uint k=p[6]+id;device const float2 *e=edges+4*k;float2 a[8]={e[0],e[1],e[2],e[3],float2(0),float2(1,0),float2(0),float2(0)};
    float2 dx=mkl_df_sub(e[2],e[0]),dy=mkl_df_sub(e[3],e[1]),qa=mkl_df_add(mkl_df_mul(dx,dx),mkl_df_mul(dy,dy)),len=mkl_df_sqrt(qa);
    if(p[2]){for(uint j=0;j<8;++j)a[j]=prev[8*k+j];if(a[7].x==0){if(status[16*k]==0)a[4]=a[5];else{float2 lo=a[4],span=mkl_df_sub(a[5],a[4]);uint l=0,h=16;for(uint t=0;t<4;++t){uint m=(l+h)/2;if(status[16*k+m]==0)l=m;else h=m;}a[4]=mkl_df_add(lo,mkl_df_mul_f32(span,float(l)/16));a[5]=mkl_df_add(lo,mkl_df_mul_f32(span,float(h)/16));}float2 span=mkl_df_mul(mkl_df_sub(a[5],a[4]),len),tol=float2(as_type<float>(p[4]),as_type<float>(p[5]));if(!wless(tol,span))a[7]=float2(1,0);}}
    else if(wless(float2(1,0),mkl_df_add(mkl_df_mul(e[2],e[2]),mkl_df_mul(e[3],e[3])))){
        float2 qb=mkl_df_add(mkl_df_mul(e[0],dx),mkl_df_mul(e[1],dy)),qc=mkl_df_add_f32(mkl_df_add(mkl_df_mul(e[0],e[0]),mkl_df_mul(e[1],e[1])),-1);
        float2 disc=mkl_df_sub(mkl_df_mul(qb,qb),mkl_df_mul(qa,qc)),root=disc.x<0?float2(0):mkl_df_sqrt(disc);float2 t=qb.x<=0?mkl_df_div(mkl_df_add(-qb,root),qa):mkl_df_div(-qc,mkl_df_add(qb,root));a[5]=wless(t,float2(0))?float2(0):(wless(float2(1,0),t)?float2(1,0):t);
    }
    uint width=p[3]?1:16;
    for(uint j=0;j<width;++j){float2 t=(p[3]||a[7].x>0)?a[4]:(j==0?a[5]:mkl_df_add(a[4],mkl_df_mul_f32(mkl_df_sub(a[5],a[4]),float(j)/16)));
        float2 xx=mkl_df_add(e[0],mkl_df_mul(dx,t)),yy=mkl_df_add(e[1],mkl_df_mul(dy,t));
        if(!p[2]){float2 rad=mkl_df_sqrt(mkl_df_add(mkl_df_mul(xx,xx),mkl_df_mul(yy,yy)));if(wless(float2(1,0),rad)){xx=mkl_df_div(xx,rad);yy=mkl_df_div(yy,rad);}}
        wsample(samples+8*(width*k+j),xx,yy,fields[k],width*k+j);
    }for(uint j=0;j<8;++j)state[8*k+j]=a[j];
}
static float2 wloadhit(device const float *v,uint n,uint k,uint j){return float2(v[j*n+k],v[(j+9)*n+k]);}
kernel void mkl_wave_place(device const float2 *q [[buffer(0)]],device const uint *mask [[buffer(1)]],device const float *start [[buffer(2)]],device const float *hit [[buffer(3)]],device const MklPolarizationValue *power [[buffer(4)]],device float2 *out [[buffer(5)]],device float2 *radius [[buffer(6)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;for(uint j=0;j<8;++j)out[8*k+j]=0;radius[k]=0;if(!mask[k])return;
    WVec sphere={mkl_df_mul_f32(q[p.n+k],p.sheet_factor),mkl_df_mul_f32(q[2*p.n+k],p.sheet_factor),mkl_df_mul_f32(q[3*p.n+k],p.sheet_factor)};
    WVec dir={wloadhit(hit,p.n,k,3),wloadhit(hit,p.n,k,4),mkl_df_mul_f32(wloadhit(hit,p.n,k,5),p.propagation)};
    uint base=p.infinite?0:3;WVec source={wloadhit(start,p.n,k,base),wloadhit(start,p.n,k,base+1),wloadhit(start,p.n,k,base+2)};
    float2 f[3];for(uint j=0;j<3;++j){WVec v={wdf(p.frame[j][0]),wdf(p.frame[j][1]),wdf(p.frame[j][2])};f[j]=wdot(j==2?dir:sphere,v);}
    out[8*k]=f[0];out[8*k+1]=f[1];out[8*k+2]=q[k];out[8*k+3]=f[2].x<0?-f[2]:f[2];out[8*k+4]=p.power?power[k].power[0]:float2(1,0);
    for(uint j=0;j<2;++j){WVec v={wdf(p.object_axes[j][0]),wdf(p.object_axes[j][1]),wdf(p.object_axes[j][2])};out[8*k+5+j]=wdot(source,v);}out[8*k+7]=wloadhit(start,p.n,k,5);radius[k]=mkl_df_sqrt(mkl_df_add(mkl_df_mul(f[0],f[0]),mkl_df_mul(f[1],f[1])));
}
static bool wderiv(device const float2 *raw,device const uint *mask,uint k,uint axis,uint col,uint side,thread float2 *v){
    uint pos=axis?k/side:k%side,d=axis?side:1;bool plus=pos+1<side&&mask[k+d],minus=pos>0&&mask[k-d];if(!plus&&!minus)return false;
    uint lo=minus?k-d:k,hi=plus?k+d:k;float f=plus&&minus?.5f:1.f;
    for(uint j=0;j<2;++j)v[j]=mkl_df_mul_f32(mkl_df_sub(raw[8*hi+col+j],raw[8*lo+col+j]),f);return true;
}
static float2 wabs(float2 v){return v.x<0?-v:v;}
static float2 wdet(thread const float2 *a,thread const float2 *b){return mkl_df_sub(mkl_df_mul(a[0],b[1]),mkl_df_mul(a[1],b[0]));}
kernel void mkl_wave_node_amplitude(device const float2 *raw [[buffer(0)]],device const uint *mask [[buffer(1)]],device float2 *amp [[buffer(6)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;amp[k]=float2(-1,0);if(!mask[k])return;
    if(p.model==0){amp[k]=float2(1,0);return;}if(p.model==1){amp[k]=mkl_df_sqrt(raw[8*k+4]);return;}
    float2 ux[2],uy[2],ox[2],oy[2];if(!wderiv(raw,mask,k,0,0,p.side,ux)||!wderiv(raw,mask,k,1,0,p.side,uy)||!wderiv(raw,mask,k,0,5,p.side,ox)||!wderiv(raw,mask,k,1,5,p.side,oy))return;
    float2 sine=wdf(p.sin_max),jac=mkl_df_div(wabs(wdet(ux,uy)),mkl_df_mul(sine,sine)),m=wabs(wdet(ox,oy));if(!p.infinite)m=mkl_df_div(m,wabs(raw[8*k+7]));if(jac.x>0)amp[k]=mkl_df_sqrt(mkl_df_div(mkl_df_mul(mkl_df_mul(raw[8*k+4],m),raw[8*k+3]),jac));
}
kernel void mkl_wave_node_norm_blocks(device const uint *mask [[buffer(1)]],device const float2 *amp [[buffer(6)]],device float2 *partial [[buffer(9)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint block=p.first+id;float2 sum=0,maxv=0;uint count=0;
    for(uint k=block*256;k<min(p.n,(block+1)*256);++k)if(mask[k]&&amp[k].x>=0){sum=mkl_df_add(sum,amp[k]);++count;if(wless(maxv,amp[k]))maxv=amp[k];}
    partial[3*block]=sum;partial[3*block+1]=float2(float(count),0);partial[3*block+2]=maxv;
}
kernel void mkl_wave_node_norm_finish(device const uint *mask [[buffer(1)]],device const float2 *amp [[buffer(6)]],device float2 *norm [[buffer(7)]],device const float2 *partial [[buffer(9)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id)return;float2 sum=0,maxv=0;uint count=0;
    for(uint i=0;i<(p.n+255)/256;++i){sum=mkl_df_add(sum,partial[3*i]);count+=uint(partial[3*i+1].x);if(wless(maxv,partial[3*i+2]))maxv=partial[3*i+2];}
    float2 mean=count?mkl_df_div(sum,float2(float(count),0)):float2(1,0);if(wless(maxv,mean))maxv=mean;
    uint center=(p.side/2)*p.side+p.side/2;float2 chief=amp[center].x<0?mean:amp[center];norm[0]=mean;norm[1]=(p.side%2&&mask[center]&&chief.x>0)?chief:maxv;
}
kernel void mkl_wave_nodes(device const float2 *raw [[buffer(0)]],device const uint *mask [[buffer(1)]],device float2 *out [[buffer(5)]],device const float2 *amp [[buffer(6)]],device const float2 *norm [[buffer(7)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;for(uint j=0;j<4;++j)out[4*k+j]=0;if(!mask[k])return;
    out[4*k]=mkl_df_div(raw[8*k],wdf(p.sin_max));out[4*k+1]=mkl_df_div(raw[8*k+1],wdf(p.sin_max));out[4*k+2]=raw[8*k+2];out[4*k+3]=mkl_df_div(amp[k].x<0?norm[0]:amp[k],norm[1]);
}
kernel void mkl_wave_slopes(device const float2 *nodes [[buffer(0)]],device const uint *mask [[buffer(1)]],device float2 *out [[buffer(6)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id,s=p.side,i=k%(s-1),j=k/(s-1),ids[4]={j*s+i,j*s+i+1,(j+1)*s+i+1,(j+1)*s+i};uint tris[12]={0,1,2,0,2,3,0,1,3,1,2,3};float2 g=0;
    for(uint t=0;t<4;++t){uint ia=ids[tris[t*3]],ib=ids[tris[t*3+1]],ic=ids[tris[t*3+2]];if(!mask[ia]||!mask[ib]||!mask[ic])continue;
        device const float2 *a=nodes+4*ia,*b=nodes+4*ib,*c=nodes+4*ic;float2 ux=mkl_df_sub(b[0],a[0]),uy=mkl_df_sub(b[1],a[1]),vx=mkl_df_sub(c[0],a[0]),vy=mkl_df_sub(c[1],a[1]),det=mkl_df_sub(mkl_df_mul(ux,vy),mkl_df_mul(uy,vx));
        if(abs(det.x)>1e-20f){float2 dw=mkl_df_sub(b[2],a[2]),ew=mkl_df_sub(c[2],a[2]);float2 v[2]={wabs(mkl_df_div(mkl_df_sub(mkl_df_mul(dw,vy),mkl_df_mul(ew,uy)),det)),wabs(mkl_df_div(mkl_df_sub(mkl_df_mul(ew,ux),mkl_df_mul(dw,vx)),det))};for(uint q=0;q<2;++q)if(wless(g,v[q]))g=v[q];}
    }out[k]=g;
}
// Largest of n non-negative DF values (radii, slopes) in 256-value blocks.
kernel void mkl_wave_max_blocks(device const float2 *v [[buffer(6)]],constant MklWavePlacementParams &p [[buffer(8)]],device float2 *partial [[buffer(9)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint block=p.first+id;float2 m=0;for(uint k=block*256;k<min(p.n,(block+1)*256);++k)if(wless(m,v[k]))m=v[k];partial[block]=m;
}
kernel void mkl_wave_max_finish(device float2 *out [[buffer(7)]],constant MklWavePlacementParams &p [[buffer(8)]],device const float2 *partial [[buffer(9)]],uint id [[thread_position_in_grid]]){
    if(id)return;float2 m=0;for(uint i=0;i<(p.n+255)/256;++i)if(wless(m,partial[i]))m=partial[i];out[0]=m;
}
kernel void mkl_wave_centroid(device const float *hits [[buffer(0)]],device const MklPolarizationValue *power [[buffer(1)]],device float2 *out [[buffer(5)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;for(uint j=0;j<2;++j)out[j*p.n+k]=mkl_df_sub(wloadhit(hits,p.side,k,j),wdf(p.object_axes[0][j]));out[2*p.n+k]=p.power?power[k].power[0]:float2(1,0);
}

kernel void mkl_wave_spectral(device const float *v0 [[buffer(0)]],device const float *v1 [[buffer(1)]],device const float *v2 [[buffer(2)]],device const float *v3 [[buffer(3)]],device const float *v4 [[buffer(4)]],device const float *v5 [[buffer(5)]],device const float *v6 [[buffer(6)]],device const float *v7 [[buffer(7)]],device const float *v8 [[buffer(8)]],device const float *v9 [[buffer(9)]],device const float *v10 [[buffer(10)]],device const float *v11 [[buffer(11)]],device const float *v12 [[buffer(12)]],device const float *v13 [[buffer(13)]],device const float *v14 [[buffer(14)]],device const float *v15 [[buffer(15)]],device float *out [[buffer(16)]],device const float *frequency [[buffer(17)]],constant MklWaveSpectralParams &p [[buffer(18)]],uint k [[thread_position_in_grid]]){
    if(k>=p.count)return;device const float *v[16]={v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15};uint width=p.complex?2:1;float2 re=0,im=0;
    for(uint j=0;j<p.inputs;++j){float2 z=float2(v[j][k*width],p.complex?v[j][k*width+1]:0);
        if(p.phase){float2 phase=-mkl_df_add(mkl_df_mul_f32(wdf(p.offsets[j][0]),frequency[2*k]),mkl_df_mul_f32(wdf(p.offsets[j][1]),frequency[2*k+1]));float2 q=wcis(phase);z=float2(z.x*q.x-z.y*q.y,z.x*q.y+z.y*q.x);}
        re=mkl_df_add(re,mkl_df_mul_f32(wdf(p.weights[j]),z.x));im=mkl_df_add(im,mkl_df_mul_f32(wdf(p.weights[j]),z.y));
    }out[k*width]=wf(re);if(p.complex)out[k*width+1]=wf(im);
}
kernel void mkl_wave_blur(device const float *a [[buffer(0)]],device const float *c [[buffer(1)]],device const float2 *w [[buffer(2)]],device float2 *out [[buffer(3)]],device uint *mask [[buffer(4)]],constant MklWaveBlurParams &p [[buffer(5)]],uint k [[thread_position_in_grid]]){
    if(k>=p.count)return;for(uint j=0;j<5;++j)out[j*p.count+k]=0;mask[k]=0;uint i=k%p.side*p.step,j=k/p.side*p.step,q=j*p.n+i;if(c[q]<=0||c[q+1]<=0||c[q+p.n]<=0)return;mask[k]=1;
    float2 x=mkl_df_mul(mkl_df_sub(w[q+1],w[q]),wdf(p.scale)),y=mkl_df_mul(mkl_df_sub(w[q+p.n],w[q]),wdf(p.scale));
    for(uint axis=0;axis<2;++axis){float2 v=mkl_df_add(mkl_df_add(mkl_df_mul(wdf(p.inverse[axis][0]),x),mkl_df_mul(wdf(p.inverse[axis][1]),y)),wdf(p.offset[axis]));out[axis*p.count+k]=v;out[(axis+2)*p.count+k]=wabs(v);}out[4*p.count+k]=mkl_df_mul(mkl_df_mul(float2(c[q],0),float2(a[q],0)),float2(a[q],0));
}
static float wj1(float x){float ax=abs(x);if(ax<=3){float t=(x/3)*(x/3);return x*(.5f+t*(-.56249985f+t*(.21093573f+t*(-.03954289f+t*(.00443319f+t*(-.00031761f+t*.00001109f))))));}float y=3/ax,f=.79788456f+y*(.00000156f+y*(.01659667f+y*(.00017105f+y*(-.00249511f+y*(.00113653f-y*.00020033f))))),th=ax-2.35619449f+y*(.12499612f+y*(.00005650f+y*(-.00637879f+y*(.00074348f+y*(.00079824f-y*.00029166f)))));float r=f*cos(th)/sqrt(ax);return x<0?-r:r;}
kernel void mkl_wave_image_complex(device const float *input [[buffer(0)]],device float2 *out [[buffer(2)]],constant MklWaveIntegralParams &p [[buffer(4)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;out[k]=float2(p.kind==3?input[k]:input[2*k]*input[2*k]+input[2*k+1]*input[2*k+1],0);
}
static float wsquarekernel(float f,float r){return f==0?2*r:sin(6.283185307179586f*f*r)/(3.141592653589793f*f);}
kernel void mkl_wave_image_query(device const float *input [[buffer(0)]],device const float2 *queries [[buffer(1)]],device float2 *out [[buffer(2)]],device float2 *acc [[buffer(3)]],constant MklWaveIntegralParams &p [[buffer(4)]],uint id [[thread_position_in_grid]]){
    if(id>=p.queries)return;float2 re=p.first?acc[2*id]:float2(0),im=p.first?acc[2*id+1]:float2(0),qx=queries[2*id],qy=queries[2*id+1];float n=float(p.n);
    for(uint k=p.first;k<p.first+p.count;++k){float2 value;
        if(p.kind==0){float2 x=mkl_df_sub(float2(float(k%p.n),0),wdf(p.center[0])),y=mkl_df_sub(float2(float(k/p.n),0),wdf(p.center[1]));float2 phase=-mkl_df_mul(mkl_df_add(mkl_df_mul(x,qx),mkl_df_mul(y,qy)),wdf(p.spacing));value=input[k]*wcis(phase);}
        else {int ix=int(k%p.n),iy=int(k/p.n);if(2*ix>=int(p.n))ix-=int(p.n);if(2*iy>=int(p.n))iy-=int(p.n);
            float2 inv=mkl_df_div(float2(1,0),mkl_df_mul_f32(wdf(p.spacing),n)),fx=mkl_df_mul_f32(inv,float(ix)),fy=mkl_df_mul_f32(inv,float(iy));
            float2 phase=mkl_df_div(mkl_df_add(mkl_df_mul_f32(wdf(p.center[0]),float(ix)),mkl_df_mul_f32(wdf(p.center[1]),float(iy))),float2(n,0)),z=wcis(phase);float real=input[2*k]*z.x-input[2*k+1]*z.y,r=wf(qx),kw=0;
            if(r>0){if(p.kind==1){float rho=wf(mkl_df_sqrt(mkl_df_add(mkl_df_mul(fx,fx),mkl_df_mul(fy,fy))));kw=rho*r<1e-12f?3.141592653589793f*r*r:r*wj1(6.283185307179586f*rho*r)/rho;}else kw=wsquarekernel(wf(fx),r)*wsquarekernel(wf(fy),r);}value=float2(real*kw,0);
        }re=mkl_df_add_f32(re,value.x);im=mkl_df_add_f32(im,value.y);
    }acc[2*id]=re;acc[2*id+1]=im;if(p.finish){out[id]=p.kind==0&&qx.x==0&&qy.x==0?float2(1,0):float2(wf(mkl_df_div(re,wdf(p.normalization))),wf(mkl_df_div(im,wdf(p.normalization))));}
}
kernel void mkl_wave_image_metrics(device const float *a [[buffer(0)]],device float2 *part [[buffer(1)]],constant uint *p [[buffer(3)]],uint block [[thread_position_in_grid]]){
    if(block>=p[1])return;float2 row[5]={0,0,0,0,0};uint n=p[0];for(uint k=block*256;k<min(n*n,(block+1)*256);++k){float v=a[k];row[0]=mkl_df_add_f32(row[0],v);row[1]=mkl_df_add(row[1],mkl_df_mul_f32(float2(v,0),float(k%n)));row[2]=mkl_df_add(row[2],mkl_df_mul_f32(float2(v,0),float(k/n)));if(k%256==0||v>row[3].x){row[3]=float2(v,0);row[4]=float2(float(k),0);}}for(uint j=0;j<5;++j)part[block*5+j]=row[j];
}
kernel void mkl_wave_image_metrics_finish(device const float2 *part [[buffer(1)]],device float2 *out [[buffer(2)]],constant uint *p [[buffer(3)]],uint id [[thread_position_in_grid]]){
    if(id)return;float2 row[5];for(uint j=0;j<5;++j)row[j]=part[j];for(uint i=1;i<p[1];++i){for(uint j=0;j<3;++j)row[j]=mkl_df_add(row[j],part[i*5+j]);if(part[i*5+3].x>row[3].x){row[3]=part[i*5+3];row[4]=part[i*5+4];}}for(uint j=0;j<5;++j)out[j]=row[j];
}
kernel void mkl_wave_sensor_field(device const float *amp [[buffer(0)]],device const float *cov [[buffer(1)]],device const float2 *opd [[buffer(2)]],device float2 *out [[buffer(3)]],device float *mom [[buffer(4)]],constant MklWaveFftMtfParams &p [[buffer(5)]],uint k [[thread_position_in_grid]]){
    if(k>=p.count)return;out[k]=0;uint i=k%p.side,j=k/p.side;if(i>=p.n||j>=p.n)return;uint node=j*p.n+i,N=p.n*p.n;mom[node]=0;mom[N+node]=0;mom[2*N+node]=0;
    float2 sine=wdf(p.sine),al=mkl_df_mul(sine,mkl_df_add_f32(mkl_df_div(float2(float(2*i+1),0),float2(float(p.n),0)),-1)),be=mkl_df_mul(sine,mkl_df_add_f32(mkl_df_div(float2(float(2*j+1),0),float2(float(p.n),0)),-1));float2 g=mkl_df_sqrt(mkl_df_sub(float2(1,0),mkl_df_mul(sine,sine)));
    float2 x=mkl_df_sub(mkl_df_add(mkl_df_mul(wdf(p.frame[0][0]),al),mkl_df_mul(wdf(p.frame[1][0]),be)),mkl_df_mul(wdf(p.frame[2][0]),g)),y=mkl_df_sub(mkl_df_add(mkl_df_mul(wdf(p.frame[0][1]),al),mkl_df_mul(wdf(p.frame[1][1]),be)),mkl_df_mul(wdf(p.frame[2][1]),g));float2 z2=mkl_df_sub(mkl_df_sub(float2(1,0),mkl_df_mul(x,x)),mkl_df_mul(y,y));if(z2.x<=0)return;WVec q={x,y,-mkl_df_sqrt(z2)};WVec ew={wdf(p.frame[2][0]),wdf(p.frame[2][1]),wdf(p.frame[2][2])};if(wdot(q,ew).x>=0)return;
    float2 scale=mkl_df_div(float2(float(p.n)*.5f,0),sine);WVec eu={wdf(p.frame[0][0]),wdf(p.frame[0][1]),wdf(p.frame[0][2])},ev={wdf(p.frame[1][0]),wdf(p.frame[1][1]),wdf(p.frame[1][2])};float ix=wf(mkl_df_add_f32(mkl_df_mul(wdot(q,eu),scale),float(p.n-1)*.5f)),iy=wf(mkl_df_add_f32(mkl_df_mul(wdot(q,ev),scale),float(p.n-1)*.5f));float fx=floor(ix),fy=floor(iy),tx=ix-fx,ty=iy-fy,a=0,c=0;float2 phase=0;
    for(uint t=0;t<4;++t){int xx=int(fx)+int(t%2),yy=int(fy)+int(t/2);if(xx<0||yy<0||xx>=int(p.n)||yy>=int(p.n))continue;uint at=uint(yy)*p.n+uint(xx);float weight=(t%2?tx:1-tx)*(t/2?ty:1-ty)*cov[at];a+=weight*amp[at];c+=weight;phase=mkl_df_add(phase,mkl_df_mul_f32(opd[at],weight));}
    if(c<=0)return;float qz=-wf(q.z);out[k]=(a/qz)*wcis(mkl_df_div(phase,float2(c,0)));mom[N+node]=a*a/(c*qz*qz);
}
kernel void mkl_wave_otf_normalize(device const float2 *in [[buffer(0)]],device const float2 *sums [[buffer(1)]],device float2 *out [[buffer(3)]],constant MklWaveFftMtfParams &p [[buffer(5)]],uint k [[thread_position_in_grid]]){
    if(k>=p.count)return;uint n=p.n,x=(k%n+n-n/2)%n,y=(k/n+n-n/2)%n;float2 z=in[y*n+x],energy=mkl_df_mul_f32(sums[1],float(n*n));out[k]=float2(wf(mkl_df_div(float2(z.x,0),energy)),wf(mkl_df_div(float2(z.y,0),energy)));if(k==(n/2)*n+n/2)out[k]=float2(1,0);
}
static float2 wcprod(float2 a,float2 b){return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
static float wkeys(float x){x=abs(x);if(x<=1)return(1.5f*x-2.5f)*x*x+1;if(x<2)return((-.5f*x+2.5f)*x-4)*x+2;return 0;}
static float2 wat(device const float2 *a,uint n,int i,int j){return i<0||j<0||i>=int(n)||j>=int(n)?float2(0):a[uint(j)*n+uint(i)];}
kernel void mkl_wave_otf_sample(device const float2 *data [[buffer(0)]],device const float2 *dl [[buffer(1)]],device const float2 *freq [[buffer(2)]],device float2 *out [[buffer(3)]],constant MklWaveFftMtfParams &p [[buffer(5)]],uint k [[thread_position_in_grid]]){
    if(k>=p.count)return;uint n=p.n,c=n/2;float d=wf(wdf(p.spacing));float2 zx=data[c*n+c+1],zy=data[(c+1)*n+c];float cx=length(zx)>1e-9f?-atan2(zx.y,zx.x)/(6.283185307179586f*d):0,cy=length(zy)>1e-9f?-atan2(zy.y,zy.x)/(6.283185307179586f*d):0;float2 nu=freq[k];float nx=wf(mkl_df_add(mkl_df_mul_f32(wdf(p.mapping[0][0]),nu.x),mkl_df_mul_f32(wdf(p.mapping[0][1]),nu.y))),ny=wf(mkl_df_add(mkl_df_mul_f32(wdf(p.mapping[1][0]),nu.x),mkl_df_mul_f32(wdf(p.mapping[1][1]),nu.y)));float x=nx/d+float(c),y=ny/d+float(c),fx=floor(x),fy=floor(y),tx=x-fx,ty=y-fy;float2 sum=0;
    for(int by=-1;by<=2;++by)for(int bx=-1;bx<=2;++bx){int i=int(fx)+bx,j=int(fy)+by;float2 phase=mkl_df_mul(float2(d,0),mkl_df_add(mkl_df_mul_f32(float2(float(i-int(c)),0),cx),mkl_df_mul_f32(float2(float(j-int(c)),0),cy)));float2 z=wcprod(wat(data,n,i,j),wcis(phase))-wat(dl,n,i,j);sum+=z*(wkeys(tx-float(bx))*wkeys(ty-float(by)));}
    for(uint t=0;t<4;++t)sum+=wat(dl,n,int(fx)+int(t%2),int(fy)+int(t/2))*((t%2?tx:1-tx)*(t/2?ty:1-ty));out[k]=wcprod(sum,wcis(-mkl_df_add(mkl_df_mul_f32(float2(nx,0),cx),mkl_df_mul_f32(float2(ny,0),cy))));
}
// Image mapping arithmetic uses unevaluated hi/lo scalars throughout Newton.
struct Wd{float2 v;};
static Wd wd(float x){return{float2(x,0)};}static Wd wd(float2 x){return{x};}
static Wd operator+(Wd a,Wd b){return{mkl_df_add(a.v,b.v)};}static Wd operator-(Wd a,Wd b){return{mkl_df_sub(a.v,b.v)};}static Wd operator-(Wd a){return{-a.v};}static Wd operator*(Wd a,Wd b){return{mkl_df_mul(a.v,b.v)};}static Wd operator/(Wd a,Wd b){return{mkl_df_div(a.v,b.v)};}
static float wv(Wd a){return wf(a.v);}static Wd wdabs(Wd a){return a.v.x<0?-a:a;}static Wd wdroot(Wd a){return{mkl_df_sqrt(a.v)};}static Wd wdclamp(Wd a){if(wless(a.v,float2(0)))return wd(0);if(wless(float2(1,0),a.v))return wd(1);return a;}static Wd wdp(constant float *a){return wd(wdf(a));}
static Wd wwhite(Wd t){Wd v=t*wd(8);float f=floor(v.v.x);Wd r=v-wd(f)-wd(.1f);return(wd(.9f)*wd(f)+(r.v.x>0?r:wd(0)))*wd(.125f);}
static Wd wedge(Wd v,Wd lo,Wd hi){Wd e=wd(.45f)+wd(.1f)*v,a=hi-e,b=lo-e;if(a.v.x<0)a=wd(0);if(b.v.x<0)b=wd(0);return(a*a-b*b)/wd(-.2f);}
kernel void mkl_wave_pattern(device float *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;float x=float(k%p.width),y=float(k/p.width),value=0;Wd w=wd(float(p.width)),h=wd(float(p.height));
    if(p.pattern==0)value=wv((wwhite(wd(x+1)/w)-wwhite(wd(x)/w))*w*(wwhite(wd(y+1)/h)-wwhite(wd(y)/h))*h);
    else if(p.pattern==1){for(uint j=0;j<16;++j)for(uint i=0;i<16;++i){float u=(x+(float(i)+.5f)/16)/float(p.width),v=(y+(float(j)+.5f)/16)/float(p.height);if(cos(atan2(v-.5f,u-.5f)*24)>0)value+=1;}value/=256;}
    else if(p.pattern==2){Wd lo=wd(x)/w,hi=wd(x+1)/w;value=wv((wedge(wd(y+1)/h,lo,hi)-wedge(wd(y)/h,lo,hi))*w*h);}
    else{uint yy=p.height-1-uint(y),scales[3]={4,2,1},scale=scales[min(yy*3/p.height,2u)],gx=uint(x)/scale,gy=yy/scale,font[21]={17,27,21,17,17,17,17,17,18,20,24,20,18,17,16,16,16,16,16,16,31};value=gy%9<7&&gx%6<5&&(font[(gx/6)%3*7+gy%9]&(1u<<(4-gx%6)))?0:1;}out[k]=value;
}
static void wweights(uint grid,thread const Wd *uv,thread uint *ids,thread Wd *w){Wd x=wdclamp(uv[0])*wd(float(grid-1)),y=wdclamp(uv[1])*wd(float(grid-1));uint ix=min(uint(floor(x.v.x)),grid-2),iy=min(uint(floor(y.v.x)),grid-2);Wd a=x-wd(float(ix)),b=y-wd(float(iy));uint k=iy*grid+ix;ids[0]=k;ids[1]=k+1;ids[2]=k+grid;ids[3]=k+grid+1;w[0]=(wd(1)-a)*(wd(1)-b);w[1]=a*(wd(1)-b);w[2]=(wd(1)-a)*b;w[3]=a*b;}
static void wmap(device const float2 *nodes,uint grid,thread const Wd *uv,uint offset,thread Wd *point){uint ids[4];Wd weights[4];wweights(grid,uv,ids,weights);point[0]=wd(0);point[1]=wd(0);for(uint k=0;k<4;++k)for(uint j=0;j<2;++j)point[j]=point[j]+wd(nodes[ids[k]*5+offset+j])*weights[k];}
static bool winverse(device const float2 *nodes,uint grid,thread const Wd *target,thread Wd *uv){for(uint j=0;j<2;++j)uv[j]=wdclamp((target[j]-wd(nodes[j]))/(wd(nodes[(grid*grid-1)*5+j])-wd(nodes[j])));for(uint iter=0;iter<16;++iter){Wd point[2];wmap(nodes,grid,uv,2,point);Wd r[2]={point[0]-target[0],point[1]-target[1]};if(wv(wdroot(r[0]*r[0]+r[1]*r[1]))<1e-9f)return true;uint ids[4];Wd weights[4];wweights(grid,uv,ids,weights);Wd a=wdclamp(uv[0])*wd(float(grid-1))-wd(float(ids[0]%grid)),b=wdclamp(uv[1])*wd(float(grid-1))-wd(float(ids[0]/grid)),du[2],dv[2];for(uint j=0;j<2;++j){Wd q[4];for(uint k=0;k<4;++k)q[k]=wd(nodes[5*ids[k]+2+j]);du[j]=((wd(1)-b)*(q[1]-q[0])+b*(q[3]-q[2]))*wd(float(grid-1));dv[j]=((wd(1)-a)*(q[2]-q[0])+a*(q[3]-q[1]))*wd(float(grid-1));}Wd det=du[0]*dv[1]-dv[0]*du[1];if(abs(det.v.x)<1e-20f)return false;uv[0]=uv[0]-(dv[1]*r[0]-dv[0]*r[1])/det;uv[1]=uv[1]-(du[0]*r[1]-du[1]*r[0])/det;for(uint j=0;j<2;++j){if(wless(uv[j].v,float2(-1e-9f,0))||wless(float2(1,1e-9f),uv[j].v))return false;uv[j]=wdclamp(uv[j]);}}return false;}
static float wraster_at(device const float *r,constant MklWaveSimulationParams &p,uint x,uint y,uint ch){uint k=(y*p.source_width+x)*p.source_channels;return p.source_channels==1?r[k]:(p.channels==1?.2126f*r[k]+.7152f*r[k+1]+.0722f*r[k+2]:r[k+ch]);}
static float wraster(device const float *r,constant MklWaveSimulationParams &p,Wd u,Wd v,uint ch){if(wless(u.v,float2(0))||wless(float2(1,0),u.v)||wless(v.v,float2(0))||wless(float2(1,0),v.v))return 0;float x=clamp(wv(u*wd(float(p.source_width))-wd(.5f)),0.f,float(p.source_width-1)),y=clamp(wv(v*wd(float(p.source_height))-wd(.5f)),0.f,float(p.source_height-1));uint ix=uint(floor(x)),iy=uint(floor(y));float a=x-float(ix),b=y-float(iy);uint nx=min(ix+1,p.source_width-1),ny=min(iy+1,p.source_height-1);return(1-b)*((1-a)*wraster_at(r,p,ix,iy,ch)+a*wraster_at(r,p,nx,iy,ch))+b*((1-a)*wraster_at(r,p,ix,ny,ch)+a*wraster_at(r,p,nx,ny,ch));}
kernel void mkl_wave_warp(device const float *source [[buffer(0)]],device const float2 *nodes [[buffer(1)]],device float *out [[buffer(8)]],device float2 *coords [[buffer(9)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;for(uint ch=0;ch<p.channels;++ch)out[k*p.channels+ch]=0;coords[k*2]=float2(-1,0);coords[k*2+1]=float2(-1,0);Wd target[2]={wdp(p.origin[0])+wdp(p.pitch[0])*wd(float(k%p.width)+.5f),wdp(p.origin[1])+wdp(p.pitch[1])*wd(float(k/p.width)+.5f)},uv[2];if(!winverse(nodes,p.grid,target,uv))return;coords[2*k]=uv[0].v;coords[2*k+1]=uv[1].v;Wd ideal[2];wmap(nodes,p.grid,uv,0,ideal);Wd src[2]={(ideal[0]-wdp(p.origin[0]))/(wdp(p.pitch[0])*wd(float(p.width))),(ideal[1]-wdp(p.origin[1]))/(wdp(p.pitch[1])*wd(float(p.height)))};uint ids[4];Wd weights[4],ri=wd(0);wweights(p.grid,uv,ids,weights);for(uint j=0;j<4;++j)ri=ri+wd(nodes[5*ids[j]+4])*weights[j];for(uint ch=0;ch<p.channels;++ch)out[k*p.channels+ch]=wraster(source,p,src[0],src[1],ch)*wv(ri);
}
static Wd wkernel_at(uint i,uint s,uint axis,constant MklWaveSimulationParams &p){return(wd(float(i)-float(p.source_width/2))*wdp(p.spacing)+wdp(p.aperture[axis])*wd((float(s)+.5f)/float(p.quadrature)-.5f))/wdp(p.pitch[axis])+wd(float(p.radius));}
kernel void mkl_wave_kernel_clear(device float2 *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint k [[thread_position_in_grid]]){if(k<p.canvas*p.canvas)out[k]=0;}
static void wk_bounds(uint out,uint axis,constant MklWaveSimulationParams &p,thread int &lo,thread int &hi){Wd a=((wd(float(out)-float(p.radius)-1)*wdp(p.pitch[axis])-wdp(p.aperture[axis])*wd(.5f))/wdp(p.spacing))+wd(float(p.source_width/2)),b=((wd(float(out)-float(p.radius)+1)*wdp(p.pitch[axis])+wdp(p.aperture[axis])*wd(.5f))/wdp(p.spacing))+wd(float(p.source_width/2));lo=max(int(floor(wv(a))),0);hi=clamp(int(ceil(wv(b))),0,int(p.source_width)-1);}
static Wd wk_weight(uint out,uint source,uint axis,constant MklWaveSimulationParams &p){Wd sum=wd(0);for(uint q=0;q<p.quadrature;++q){Wd v=wd(1)-wdabs(wkernel_at(source,q,axis,p)-wd(float(out)));if(v.v.x>0)sum=sum+v;}return sum;}
kernel void mkl_wave_kernel_horizontal(device const float *psf [[buffer(0)]],device float2 *horizontal [[buffer(9)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id,side=2*p.radius+1,ox=k%side,y=k/side;int lo,hi;wk_bounds(ox,0,p,lo,hi);Wd sum=wd(0);for(int ix=lo;ix<=hi;++ix)sum=sum+wd(psf[y*p.source_width+uint(ix)])*wk_weight(ox,uint(ix),0,p);horizontal[k]=(sum/wdp(p.normalization)/wd(float(p.quadrature))).v;}
kernel void mkl_wave_kernel_vertical(device const float2 *horizontal [[buffer(9)]],device float2 *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id,side=2*p.radius+1,ox=k%side,y=k/side;int lo,hi;wk_bounds(y,1,p,lo,hi);Wd sum=wd(0);for(int iy=lo;iy<=hi;++iy)sum=sum+wd(horizontal[uint(iy)*side+ox])*wk_weight(y,uint(iy),1,p);out[y*p.canvas+ox]=float2(wv(sum/wd(float(p.quadrature))),0);}
static bool wsource(constant MklWaveSimulationParams &p,uint k,thread uint &source){uint hw=p.tile[2]+2*p.radius,hh=p.tile[3]+2*p.radius,width=p.mode==4?p.canvas:hw,x=k%width,y=k/width;if(x>=hw||y>=hh)return false;int sx=int(p.tile[0])+int(x)-int(p.radius),sy=int(p.tile[1])+int(y)-int(p.radius);if(sx<0||sy<0||sx>=int(p.width)||sy>=int(p.height))return false;source=uint(sy)*p.width+uint(sx);return true;}
kernel void mkl_wave_active_clear(device atomic_uint *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint k [[thread_position_in_grid]]){if(k<p.grid*p.grid)atomic_store_explicit(out+k,0,memory_order_relaxed);}
kernel void mkl_wave_active(device const float2 *coords [[buffer(0)]],device atomic_uint *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint source;if(!wsource(p,p.first+id,source)||coords[2*source].x<0)return;Wd uv[2]={wd(coords[2*source]),wd(coords[2*source+1])};uint ids[4];Wd weights[4];wweights(p.grid,uv,ids,weights);for(uint j=0;j<4;++j)if(weights[j].v.x>0)atomic_store_explicit(out+ids[j],1,memory_order_relaxed);}
kernel void mkl_wave_weighted(device const float *image [[buffer(0)]],device const float2 *coords [[buffer(1)]],device float2 *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id;out[k]=0;uint source;if(!wsource(p,k,source)||coords[2*source].x<0)return;Wd uv[2]={wd(coords[2*source]),wd(coords[2*source+1])};uint ids[4];Wd weights[4],w=wd(0);wweights(p.grid,uv,ids,weights);for(uint j=0;j<4;++j)if(ids[j]==p.node)w=w+weights[j];out[k]=float2(wv(w*wd(image[source*p.channels+p.channel])),0);}
kernel void mkl_wave_crop(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],device const float2 *c [[buffer(2)]],device const float *previous [[buffer(3)]],device float *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id,ch=k%p.channels,pixel=k/p.channels,x=pixel%p.width,y=pixel/p.width,j=(y+2*p.radius)*p.canvas+x+2*p.radius;float v=ch==0?a[j].x:(ch==1?b[j].x:c[j].x);out[k]=(p.previous?previous[k]:0)+v;}
kernel void mkl_wave_assemble(device const float *tile [[buffer(0)]],device float *out [[buffer(8)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id,ch=k%p.channels,pixel=k/p.channels,x=pixel%p.tile[2],y=pixel/p.tile[2];out[((y+p.tile[1])*p.width+x+p.tile[0])*p.channels+ch]=tile[k];}
struct WNoise{uint2 key;uint stream;uint draw;uint4 words;};
static float wnext(thread WNoise &r){if((r.draw%4)==0)r.words=mkl_philox4x32_10(uint4(r.draw/4,0,r.stream,0),r.key);uint v=r.words[r.draw%4];++r.draw;return mkl_u32_to_unit(v);}
static Wd wlogarithm(Wd x){int exponent=int((as_type<uint>(x.v.x)>>23)&255)-127;float scale=as_type<float>(uint(127-exponent)<<23);Wd m=x*wd(scale),y=(m-wd(1))/(m+wd(1)),yy=y*y,term=y,sum=y;for(uint k=1;k<=16;++k){term=term*yy;sum=sum+term/wd(float(2*k+1));}return sum*wd(2)+wd(float2(.6931471824645996f,-1.904654323148236e-9f))*wd(float(exponent));}
static Wd wlog_probability(float mean,float k){if(k<10){Wd factorial=wd(0);for(uint i=2;i<=uint(k);++i)factorial=factorial+wlogarithm(wd(float(i)));return -wd(mean)+wd(k)*wlogarithm(wd(mean))-factorial;}Wd t=(wd(mean)-wd(k))/wd(k),dev;if(abs(wv(t))<.125f){Wd term=t*t,sum=-term/wd(2);for(uint n=3;n<=16;++n){term=term*t;sum=n%2?sum+term/wd(float(n)):sum-term/wd(float(n));}dev=wd(k)*sum;}else dev=wd(k)*(wlogarithm(wd(1)+t)-t);Wd inv=wd(1)/wd(k);return dev-wlogarithm(wd(k))*wd(.5f)-wd(float2(.9189385175704956f,1.563417706258452e-8f))-inv/wd(12)+inv*inv*inv/wd(360);}
static float wpoisson(float mean,thread WNoise &rng){if(mean<30){float limit=mkl_exp(-mean),product=1,n=0;while(true){product*=wnext(rng);if(product<=limit)return n;n+=1;}}float b=.931f+2.53f*sqrt(mean),a=-.059f+.02483f*b,inv=1.1239f+1.1328f/(b-3.4f),vr=.9277f-3.6224f/(b-2);while(true){float u=wnext(rng)-.5f,v=wnext(rng),us=.5f-abs(u),k=floor((2*a/us+b)*u+mean+.43f);if(us>=.07f&&v<=vr)return k;if(k<0||(us<.013f&&v>us))continue;Wd lhs=wlogarithm(wd(v*inv/(a/(us*us)+b)));if((lhs-wlog_probability(mean,k)).v.x<=0)return k;}}
kernel void mkl_wave_noise(device const float *image [[buffer(0)]],device float *out [[buffer(8)]],device atomic_uint *invalid [[buffer(9)]],constant MklWaveSimulationParams &p [[buffer(10)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id;if(!isfinite(image[k])||(p.well>0&&(!isfinite(max(image[k],0.f)*p.well)||max(image[k],0.f)*p.well>1e12f))){atomic_fetch_add_explicit(invalid,1,memory_order_relaxed);out[k]=NAN;return;}float v=max(image[k],0.f);WNoise rng={uint2(p.seed[0],p.seed[1]),k,0,uint4(0)};if(p.well>0)v=wpoisson(v*p.well,rng)/p.well;if(p.sigma>0){float u=wnext(rng),w=wnext(rng);v+=p.sigma*sqrt(-2*log(u))*cos(6.283185307179586f*w);}if(!isfinite(v))atomic_fetch_add_explicit(invalid,1,memory_order_relaxed);out[k]=v;}
kernel void mkl_wave_shell_keys(device uint *keys [[buffer(8)]],device uint *ids [[buffer(9)]],constant MklWaveShellParams &p [[buffer(20)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id;int i=int(k%p.size),j=int(k/p.size);if(2*i>=int(p.size))i-=int(p.size);if(2*j>=int(p.size))j-=int(p.size);keys[k]=uint(i*i+j*j);ids[k]=k;}
kernel void mkl_wave_shell_boundaries(device const uint *keys [[buffer(0)]],device uint *mask [[buffer(8)]],device uint *positions [[buffer(9)]],constant MklWaveShellParams &p [[buffer(20)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id;mask[k]=uint(k==0||keys[k]!=keys[k-1]);positions[k]=k;}
kernel void mkl_wave_shell_offsets_clear(device const uint *count [[buffer(1)]],device atomic_uint *summary [[buffer(9)]],uint id [[thread_position_in_grid]]){if(id)return;atomic_store_explicit(summary,count[0],memory_order_relaxed);atomic_store_explicit(summary+1,0,memory_order_relaxed);}
kernel void mkl_wave_shell_offsets(device const uint *positions [[buffer(0)]],device const uint *count [[buffer(1)]],device uint *out [[buffer(8)]],device atomic_uint *summary [[buffer(9)]],constant MklWaveShellParams &p [[buffer(20)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id,n=count[0];out[k]=k<n?positions[k]:p.length;if(k<n)atomic_fetch_max_explicit(summary+1,(k+1<n?positions[k+1]:p.length)-positions[k],memory_order_relaxed);}
kernel void mkl_wave_shell_gather(device const float2 *spectrum [[buffer(0)]],device const uint *order [[buffer(1)]],device float2 *out [[buffer(8)]],constant MklWaveShellParams &p [[buffer(20)]],uint id [[thread_position_in_grid]]){if(id>=p.count)return;uint k=p.first+id,q=order[k];int i=int(q%p.size),j=int(q/p.size);if(2*i>=int(p.size))i-=int(p.size);if(2*j>=int(p.size))j-=int(p.size);float2 phase=mkl_df_div(mkl_df_add(mkl_df_mul_f32(wdf(p.center[0]),float(i)),mkl_df_mul_f32(wdf(p.center[1]),float(j))),float2(float(p.size),0)),z=wcis(phase),v=spectrum[q];out[k]=mkl_df_div(float2(v.x*z.x-v.y*z.y,0),wdf(p.normalization));}

kernel void mkl_wave_shell_query(device const float2 *v0 [[buffer(0)]],device const float2 *v1 [[buffer(1)]],device const float2 *v2 [[buffer(2)]],device const float2 *v3 [[buffer(3)]],device const float2 *v4 [[buffer(4)]],device const float2 *v5 [[buffer(5)]],device const float2 *v6 [[buffer(6)]],device const float2 *v7 [[buffer(7)]],device const float2 *v8 [[buffer(8)]],device const float2 *v9 [[buffer(9)]],device const float2 *v10 [[buffer(10)]],device const float2 *v11 [[buffer(11)]],device const float2 *v12 [[buffer(12)]],device const float2 *v13 [[buffer(13)]],device const float2 *v14 [[buffer(14)]],device const float2 *v15 [[buffer(15)]],device const uint *keys [[buffer(16)]],device const float2 *radii [[buffer(17)]],device float2 *out [[buffer(18)]],device float2 *acc [[buffer(19)]],constant MklWaveShellParams &p [[buffer(20)]],uint id [[thread_position_in_grid]]){
    if(id>=p.query_count)return;uint query=p.query_first+id;device const float2 *v[16]={v0,v1,v2,v3,v4,v5,v6,v7,v8,v9,v10,v11,v12,v13,v14,v15};float2 sums[16];for(uint j=0;j<p.fields;++j)sums[j]=p.first?acc[j*p.queries+query]:float2(0);float r=wf(radii[query]);
    if(r>0)for(uint k=p.first;k<p.first+p.count;++k){float rho=wf(mkl_df_mul(mkl_df_sqrt(float2(float(keys[k]),0)),wdf(p.frequency_step))),weight=rho*r<1e-12f?3.141592653589793f*r*r:r*wj1(6.283185307179586f*rho*r)/rho;for(uint j=0;j<p.fields;++j)sums[j]=mkl_df_add(sums[j],mkl_df_mul_f32(v[j][k],weight));}
    for(uint j=0;j<p.fields;++j){acc[j*p.queries+query]=sums[j];if(p.finish)out[j*p.queries+query]=sums[j];}
}

kernel void mkl_wave_wavelength(device const uint *indices [[buffer(0)]],device const uint *table [[buffer(1)]],device uint *out [[buffer(3)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]){if(id>=p[7])return;uint k=p[6]+id,n=p[1],t=indices[k]*MKL_MAX_WAVELENGTHS+indices[n+k];out[k]=0;out[n+k]=t<p[0]?table[t]:0xffffffffu;}
kernel void mkl_wave_split(device const uint *in [[buffer(0)]],device uint *out [[buffer(3)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]){if(id>=p[7])return;uint k=p[6]+id,j=k/p[3];out[k]=in[p[0]+j*p[1]+k%p[3]];}
kernel void mkl_wave_sensor_domain_clear(device atomic_uint *out [[buffer(6)]],uint id [[thread_position_in_grid]]){if(id==0)atomic_store_explicit(out,0,memory_order_relaxed);}
kernel void mkl_wave_sensor_domain(device const float *coverage [[buffer(0)]],device atomic_uint *out [[buffer(6)]],constant MklWaveFftMtfParams &p [[buffer(5)]],uint k [[thread_position_in_grid]]){if(k>=p.count||coverage[k]<=0)return;float2 a=mkl_df_mul(wdf(p.sine),mkl_df_add_f32(mkl_df_div(float2(float(2*(k%p.n)+1),0),float2(float(p.n),0)),-1)),b=mkl_df_mul(wdf(p.sine),mkl_df_add_f32(mkl_df_div(float2(float(2*(k/p.n)+1),0),float2(float(p.n),0)),-1)),gamma=mkl_df_sqrt(mkl_df_sub(mkl_df_sub(float2(1,0),mkl_df_mul(a,a)),mkl_df_mul(b,b)));float2 z=mkl_df_sub(mkl_df_add(mkl_df_mul(wdf(p.frame[0][0]),a),mkl_df_mul(wdf(p.frame[0][1]),b)),mkl_df_mul(wdf(p.frame[0][2]),gamma));if(!isfinite(z.x)||z.x>=0)atomic_fetch_add_explicit(out,1,memory_order_relaxed);}

// Per-plane sampling diagnostics share cached exact defocus when available.
static float2 wfocused(constant MklWaveParams &p,device const float2 *w,device const float2 *cached,uint plane,uint k,float2 dz){return p.reserved[0]?cached[plane*p.n*p.n+k]:mkl_df_add(w[k],wfocus(p,k,dz));}
kernel void mkl_wave_focus_step_blocks(device const float *c [[buffer(1)]],device const float2 *w [[buffer(2)]],constant MklWaveParams &p [[buffer(7)]],device const float2 *shifts [[buffer(9)]],device const float2 *cached [[buffer(10)]],device float *blocks [[buffer(11)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint n=p.n,N=n*n,B=(N+255)/256,b=id+p.first,plane=b/B,first=(b%B)*256;float2 dz=p.unit_energy?shifts[plane]:float2(0);float step=0;
    for(uint k=first;k<min(first+256,N);++k){if(c[k]<=0)continue;float2 v=wfocused(p,w,cached,plane,k,dz);if(k%n+1<n&&c[k+1]>0)step=max(step,abs(wf(mkl_df_sub(wfocused(p,w,cached,plane,k+1,dz),v))));if(k/n+1<n&&c[k+n]>0)step=max(step,abs(wf(mkl_df_sub(wfocused(p,w,cached,plane,k+n,dz),v))));}blocks[b]=step;
}
kernel void mkl_wave_focus_step_finish(constant MklWaveParams &p [[buffer(7)]],device const float *blocks [[buffer(11)]],device float *steps [[buffer(12)]],uint plane [[thread_position_in_grid]]){if(plane>=p.size)return;uint B=(p.n*p.n+255)/256;float step=0;for(uint b=0;b<B;++b)step=max(step,blocks[plane*B+b]);steps[plane]=step;}
