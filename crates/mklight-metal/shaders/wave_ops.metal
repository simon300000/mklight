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
