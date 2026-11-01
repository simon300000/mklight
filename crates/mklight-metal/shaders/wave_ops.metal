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
kernel void mkl_wave_focus_correlation(device const float *a [[buffer(0)]],device const float *c [[buffer(1)]],device const float2 *w [[buffer(2)]],device float2 *out [[buffer(4)]],device float2 *acc [[buffer(6)]],constant MklWaveParams &p [[buffer(7)]],device const float *stencil [[buffer(8)]],device const float2 *shifts [[buffer(9)]],uint id [[thread_position_in_grid]]) {
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
            ar+=wt*a[j]/(p.shifted?-wf(wdir(p,j).z):1.0f);cov+=wt;phase=mkl_df_add(phase,mkl_df_mul_f32(mkl_df_add(w[j],wfocus(p,j,dz)),wt));
        }
        float deficit=(1-c[k])*energy*max(0.0f,1-abs(ix-float(k%p.n)))*max(0.0f,1-abs(iy-float(k/p.n)));
        float2 term=0;
        if(cov>0) {float2 ph=mkl_df_sub(mkl_df_div(phase,float2(cov,0)),mkl_df_add(w[k],wfocus(p,k,dz)));float m=c[k]*a[k]/gamma*ar/(p.shifted?1.0f:qz);term=m*wcis(ph);}
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
kernel void mkl_wave_resample(device const float2 *nodes [[buffer(0)]],device const float2 *edges [[buffer(2)]],device const uint *tris [[buffer(3)]],device const uint *owner [[buffer(4)]],device float *amp [[buffer(5)]],device float *cov [[buffer(6)]],device float2 *opd [[buffer(7)]],device float *diag [[buffer(8)]],constant MklWaveResampleParams &p [[buffer(9)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.count)return;uint k=p.first+id,s=p.supersample,ns=p.n*s,N=p.n*p.n;float2 a=0,w=0;uint count=0,unresolved=0;float gap=0,pitch=1;
    for(uint sy=0;sy<s;++sy)for(uint sx=0;sx<s;++sx) {
        uint x=(k%p.n)*s+sx,y=(k/p.n)*s+sy,q=y*ns+x;float2 u=wrcoord(x,ns),v=wrcoord(y,ns);
        if(p.mask&& !wrinside(p,edges,u,v))continue;bool missing=owner[q]==0xffffffffu;
        if(missing) {
            if(!p.mask)continue;bool found=false;
            for(uint d=1;d<min(ns,uint(MKL_WAVE_MAX_EXTRAPOLATION+1))&&!found;++d) {
                uint ymin=y> d?y-d:0,ymax=min(y+d,ns-1),xmin=x>d?x-d:0,xmax=min(x+d,ns-1);
                for(uint yy=ymin;yy<=ymax&&!found;++yy) {
                    // Only the perimeter is searched, preserving row-major ties.
                    if(abs(int(yy)-int(y))==int(d)) {for(uint xx=xmin;xx<=xmax;++xx)if(owner[yy*ns+xx]!=0xffffffffu){q=yy*ns+xx;found=true;break;}}
                    else {if(x>=d&&owner[yy*ns+x-d]!=0xffffffffu){q=yy*ns+x-d;found=true;}else if(x+d<ns&&owner[yy*ns+x+d]!=0xffffffffu){q=yy*ns+x+d;found=true;}}
                }
            }
            if(!found){++unresolved;continue;}
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
    float a[9]={0,0,0,0,1,float(n),float(n),0,0};
    for(uint k=block*256;k<min(nn,(block+1)*256);++k) {
        a[1]+=float(d[2*nn+k]>0);
        if(d[k]/d[nn+k]>a[3]/a[4]){a[3]=d[k];a[4]=d[nn+k];}
        if(c[k]<=0)continue;
        a[0]+=1;uint ix=k%n,iy=k/n;
        a[5]=min(a[5],float(ix));a[6]=min(a[6],float(iy));a[7]=max(a[7],float(ix));a[8]=max(a[8],float(iy));
        uint next[2]={ix+1<n?k+1:k,iy+1<n?k+n:k};
        for(uint t=0;t<2;++t)if(c[next[t]]>0)a[2]=max(a[2],abs(wf(mkl_df_sub(w[next[t]],w[k]))));
    }
    for(uint j=0;j<9;++j)part[block*9+j]=a[j];
}
kernel void mkl_wave_diagnostics_finish(device const float *part [[buffer(3)]],device float2 *out [[buffer(4)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]) {
    if(id)return;float a[9]={0,0,0,0,1,float(p[0]),float(p[0]),0,0};
    for(uint i=0;i<p[1];++i){device const float *b=part+9*i;
        a[0]+=b[0];a[1]+=b[1];a[2]=max(a[2],b[2]);
        if(b[3]/b[4]>a[3]/a[4]){a[3]=b[3];a[4]=b[4];}
        for(uint j=5;j<7;++j)a[j]=min(a[j],b[j]);for(uint j=7;j<9;++j)a[j]=max(a[j],b[j]);
    }
    for(uint j=0;j<9;++j)out[j]=float2(a[j],0);
}
kernel void mkl_wave_statistics_prepare(device const float *a [[buffer(0)]],device const float *c [[buffer(1)]],device const float2 *w [[buffer(2)]],device float2 *part [[buffer(3)]],device float2 *samples [[buffer(4)]],constant uint *p [[buffer(6)]],uint block [[thread_position_in_grid]]) {
    uint n=p[0],nn=n*n;if(block>=p[1])return;float2 sine=float2(as_type<float>(p[2]),as_type<float>(p[3]));float2 s[4]={0,0,0,0};
    for(uint k=block*256;k<min(nn,(block+1)*256);++k){
        float2 u=mkl_df_add_f32(mkl_df_div(float2(float(2*(k%n)+1),0),float2(float(n),0)),-1);
        float2 v=mkl_df_add_f32(mkl_df_div(float2(float(2*(k/n)+1),0),float2(float(n),0)),-1);
        float2 gamma=mkl_df_sqrt(mkl_df_sub(float2(1,0),mkl_df_mul(mkl_df_mul(sine,sine),mkl_df_add(mkl_df_mul(u,u),mkl_df_mul(v,v)))));
        float2 weight=c[k]>0?mkl_df_div(mkl_df_mul(float2(c[k],0),float2(a[k],0)),gamma):float2(0);
        samples[4*k]=u;samples[4*k+1]=v;samples[4*k+2]=w[k];samples[4*k+3]=mkl_df_sqrt(weight);
        float2 z=wcis(w[k]);float2 vals[4]={weight,mkl_df_mul(weight,w[k]),mkl_df_mul_f32(weight,z.x),mkl_df_mul_f32(weight,z.y)};
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
static void wsample(device uint *s,float2 x,float2 y,uint k){s[0]=as_type<uint>(x.x);s[1]=as_type<uint>(x.y);s[2]=as_type<uint>(y.x);s[3]=as_type<uint>(y.y);s[4]=0;s[5]=k;s[6]=0;s[7]=as_type<uint>(1.f);}
kernel void mkl_wave_lattice(device uint *samples [[buffer(3)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]){
    if(id>=p[7])return;uint k=p[6]+id,s=p[0],i,j;float off=0;
    if(k<s*s){i=k%s;j=k/s;}else{uint t=k-s*s;i=t%(s-1);j=t/(s-1);off=.5f;}
    float2 h=mkl_df_div(float2(2,0),float2(float(s-3),0));float c=float(s-1)*.5f;
    wsample(samples+8*k,mkl_df_mul(mkl_df_sub(float2(float(i)+off,0),float2(c,0)),h),mkl_df_mul(mkl_df_sub(float2(float(j)+off,0),float2(c,0)),h),k);
}
kernel void mkl_wave_survival(device const float2 *samples [[buffer(0)]],device const uint *status [[buffer(1)]],device uint *out [[buffer(3)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]){
    if(id>=p[7])return;uint k=p[6]+id;float2 x=samples[4*k],y=samples[4*k+1];float2 r=mkl_df_add(mkl_df_mul(x,x),mkl_df_mul(y,y));out[k]=uint(status[k]==0&&wless(r,float2(1,1.0001e-12f)));
}
kernel void mkl_wave_rim(device const float2 *edges [[buffer(0)]],device const float2 *prev [[buffer(1)]],device const uint *status [[buffer(2)]],device float2 *state [[buffer(3)]],device uint *samples [[buffer(4)]],constant uint *p [[buffer(5)]],uint id [[thread_position_in_grid]]){
    if(id>=p[7])return;uint k=p[6]+id;device const float2 *e=edges+4*k;float2 a[8]={e[0],e[1],e[2],e[3],float2(0),float2(1,0),float2(0),float2(0)};
    float2 dx=mkl_df_sub(e[2],e[0]),dy=mkl_df_sub(e[3],e[1]),qa=mkl_df_add(mkl_df_mul(dx,dx),mkl_df_mul(dy,dy)),len=mkl_df_sqrt(qa);
    if(p[2]){for(uint j=0;j<8;++j)a[j]=prev[8*k+j];if(a[7].x==0){if(status[16*k]==0)a[4]=a[5];else{float2 lo=a[4],span=mkl_df_sub(a[5],a[4]);uint l=0,h=16;for(uint t=0;t<4;++t){uint m=(l+h)/2;if(status[16*k+m]==0)l=m;else h=m;}a[4]=mkl_df_add(lo,mkl_df_mul_f32(span,float(l)/16));a[5]=mkl_df_add(lo,mkl_df_mul_f32(span,float(h)/16));}float2 span=mkl_df_mul(mkl_df_sub(a[5],a[4]),len),tol=float2(as_type<float>(p[4]),as_type<float>(p[5]));if(!wless(tol,span))a[7]=float2(1,0);}}
    else if(wless(float2(1,0),mkl_df_add(mkl_df_mul(e[2],e[2]),mkl_df_mul(e[3],e[3])))){
        float2 qb=mkl_df_add(mkl_df_mul(e[0],dx),mkl_df_mul(e[1],dy)),qc=mkl_df_add_f32(mkl_df_add(mkl_df_mul(e[0],e[0]),mkl_df_mul(e[1],e[1])),-1);
        a[5]=mkl_df_div(mkl_df_add(-qb,mkl_df_sqrt(mkl_df_sub(mkl_df_mul(qb,qb),mkl_df_mul(qa,qc)))),qa);
    }
    uint width=p[3]?1:16;
    for(uint j=0;j<width;++j){float2 t=(p[3]||a[7].x>0)?a[4]:(j==0?a[5]:mkl_df_add(a[4],mkl_df_mul_f32(mkl_df_sub(a[5],a[4]),float(j)/16)));
        float2 xx=mkl_df_add(e[0],mkl_df_mul(dx,t)),yy=mkl_df_add(e[1],mkl_df_mul(dy,t));
        if(!p[2]){float2 rad=mkl_df_sqrt(mkl_df_add(mkl_df_mul(xx,xx),mkl_df_mul(yy,yy)));if(wless(float2(1,0),rad)){xx=mkl_df_div(xx,rad);yy=mkl_df_div(yy,rad);}}
        wsample(samples+8*(width*k+j),xx,yy,width*k+j);
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
kernel void mkl_wave_node_norm(device const uint *mask [[buffer(1)]],device const float2 *amp [[buffer(6)]],device float2 *norm [[buffer(7)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id)return;float2 sum=0,maxv=0;uint count=0;
    for(uint k=0;k<p.n;++k)if(mask[k]&&amp[k].x>=0){sum=mkl_df_add(sum,amp[k]);++count;if(wless(maxv,amp[k]))maxv=amp[k];}
    float2 mean=count?mkl_df_div(sum,float2(float(count),0)):float2(1,0);if(wless(maxv,mean))maxv=mean;
    uint center=(p.side/2)*p.side+p.side/2;float2 chief=amp[center].x<0?mean:amp[center];
    norm[0]=mean;norm[1]=(p.side%2&&mask[center]&&chief.x>0)?chief:maxv;
}
kernel void mkl_wave_nodes(device const float2 *raw [[buffer(0)]],device const uint *mask [[buffer(1)]],device float2 *out [[buffer(5)]],device const float2 *amp [[buffer(6)]],device const float2 *norm [[buffer(7)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id;for(uint j=0;j<4;++j)out[4*k+j]=0;if(!mask[k])return;
    out[4*k]=mkl_df_div(raw[8*k],wdf(p.sin_max));out[4*k+1]=mkl_df_div(raw[8*k+1],wdf(p.sin_max));out[4*k+2]=raw[8*k+2];out[4*k+3]=mkl_df_div(amp[k].x<0?norm[0]:amp[k],norm[1]);
}
kernel void mkl_wave_slopes(device const float2 *nodes [[buffer(0)]],device const uint *mask [[buffer(1)]],device float2 *out [[buffer(5)]],constant MklWavePlacementParams &p [[buffer(8)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;uint k=p.first+id,s=p.side,i=k%(s-1),j=k/(s-1),ids[4]={j*s+i,j*s+i+1,(j+1)*s+i+1,(j+1)*s+i};uint tris[12]={0,1,2,0,2,3,0,1,3,1,2,3};float2 g=0;
    for(uint t=0;t<4;++t){uint ia=ids[tris[t*3]],ib=ids[tris[t*3+1]],ic=ids[tris[t*3+2]];if(!mask[ia]||!mask[ib]||!mask[ic])continue;
        device const float2 *a=nodes+4*ia,*b=nodes+4*ib,*c=nodes+4*ic;float2 ux=mkl_df_sub(b[0],a[0]),uy=mkl_df_sub(b[1],a[1]),vx=mkl_df_sub(c[0],a[0]),vy=mkl_df_sub(c[1],a[1]),det=mkl_df_sub(mkl_df_mul(ux,vy),mkl_df_mul(uy,vx));
        if(abs(det.x)>1e-20f){float2 dw=mkl_df_sub(b[2],a[2]),ew=mkl_df_sub(c[2],a[2]);float2 v[2]={wabs(mkl_df_div(mkl_df_sub(mkl_df_mul(dw,vy),mkl_df_mul(ew,uy)),det)),wabs(mkl_df_div(mkl_df_sub(mkl_df_mul(ew,ux),mkl_df_mul(dw,vx)),det))};for(uint q=0;q<2;++q)if(wless(g,v[q]))g=v[q];}
    }out[k]=g;
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
            float2 phase=mkl_df_div(mkl_df_add(mkl_df_mul_f32(wdf(p.center[0]),float(ix)),mkl_df_mul_f32(wdf(p.center[1]),float(iy))),float2(n,0)),z=wcis(phase);float real=input[2*k]*z.x-input[2*k+1]*z.y,r=wf(qx),kernel=0;
            if(r>0){if(p.kind==1){float rho=wf(mkl_df_sqrt(mkl_df_add(mkl_df_mul(fx,fx),mkl_df_mul(fy,fy))));kernel=rho*r<1e-12f?3.141592653589793f*r*r:r*wj1(6.283185307179586f*rho*r)/rho;}else kernel=wsquarekernel(wf(fx),r)*wsquarekernel(wf(fy),r);}value=float2(real*kernel,0);
        }re=mkl_df_add_f32(re,value.x);im=mkl_df_add_f32(im,value.y);
    }acc[2*id]=re;acc[2*id+1]=im;if(p.finish){out[id]=p.kind==0&&qx.x==0&&qy.x==0?float2(1,0):float2(wf(mkl_df_div(re,wdf(p.normalization))),wf(mkl_df_div(im,wdf(p.normalization))));}
}
kernel void mkl_wave_image_metrics(device const float *a [[buffer(0)]],device float2 *part [[buffer(1)]],constant uint *p [[buffer(3)]],uint block [[thread_position_in_grid]]){
    if(block>=p[1])return;float2 row[5]={0,0,0,0,0};uint n=p[0];for(uint k=block*256;k<min(n*n,(block+1)*256);++k){float v=a[k];row[0]=mkl_df_add_f32(row[0],v);row[1]=mkl_df_add(row[1],mkl_df_mul_f32(float2(v,0),float(k%n)));row[2]=mkl_df_add(row[2],mkl_df_mul_f32(float2(v,0),float(k/n)));if(k%256==0||v>row[3].x){row[3]=float2(v,0);row[4]=float2(float(k),0);}}for(uint j=0;j<5;++j)part[block*5+j]=row[j];
}
kernel void mkl_wave_image_metrics_finish(device const float2 *part [[buffer(1)]],device float2 *out [[buffer(2)]],constant uint *p [[buffer(3)]],uint id [[thread_position_in_grid]]){
    if(id)return;float2 row[5];for(uint j=0;j<5;++j)row[j]=part[j];for(uint i=1;i<p[1];++i){for(uint j=0;j<3;++j)row[j]=mkl_df_add(row[j],part[i*5+j]);if(part[i*5+3].x>row[3].x){row[3]=part[i*5+3];row[4]=part[i*5+4];}}for(uint j=0;j<5;++j)out[j]=row[j];
}
