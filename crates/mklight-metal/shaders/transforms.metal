#include "mklight_common.h"
static float2 p2_df(constant float *v){return mkl_df_input(float2(v[0],v[1]));}
static float2 p2_cmul(float2 a,float2 b){return float2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);}
// exp(2πi·a) for double-float turns a, reduced modulo a turn before f32 trigonometry.
static float2 p2_cis(float2 a){
    a=mkl_df_add_f32(a,-rint(a.x));
    a=mkl_df_add_f32(a,-rint(a.x));
    float t=(a.x+a.y)*6.2831853071795864769f;
    return float2(cos(t),sin(t));
}
// 0.5·(a0·x² + 2·a1·x·y + a4·y²) in double-float turns.
static float2 p2_q(constant MklTransformParams &p,float x,float y){
    return mkl_df_mul_f32(mkl_df_add(mkl_df_add(mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[0]),x),x),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[1]),2*x),y)),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[4]),y),y)),0.5f);
}
static float2 p2_pre(constant MklTransformParams &p,float x,float y){
    return mkl_df_add(p2_q(p,x,y),mkl_df_add(mkl_df_mul_f32(p2_df(p.affine[2]),x),mkl_df_mul_f32(p2_df(p.affine[5]),y)));
}
// Elementwise chirp/phase/spectral steps.
kernel void mkl_transform(device const float2 *a [[buffer(0)]],device const float2 *b [[buffer(1)]],device float2 *out [[buffer(2)]],constant MklTransformParams &p [[buffer(3)]],uint id [[thread_position_in_grid]]){
    if(id>=p.count)return;
    uint w=p.width,h=p.height;float scale=as_type<float>(p.scale_bits),sign=float(as_type<int>(p.sign_bits));float2 z=0;
    switch(p.mode){
    case MKL_TRANSFORM_KERNEL:{
        uint ix=id%w,iy=id/w;
        if((ix<p.ox||ix>w-p.nx)&&(iy<p.oy||iy>h-p.ny)){
            float x=ix<p.ox?float(ix):float(ix)-float(w),y=iy<p.oy?float(iy):float(iy)-float(h);
            z=p2_cis(mkl_df_mul_f32(p2_q(p,x,y),-sign));
        }break;
    }
    case MKL_TRANSFORM_PRE:{uint x=id%w,y=id/w;if(x<p.nx&&y<p.ny)z=p2_cmul(a[y*p.nx+x],p2_cis(mkl_df_mul_f32(p2_pre(p,float(x),float(y)),sign)));break;}
    case MKL_TRANSFORM_POST:{uint x=id%p.ox,y=id/p.ox;z=p2_cmul(a[y*w+x],p2_cis(mkl_df_mul_f32(p2_q(p,float(x),float(y)),sign)))*scale;break;}
    case MKL_TRANSFORM_MULTIPLY:{float2 v=b[p.flags&2u?id%w:id];z=p2_cmul(a[id],v)*scale;break;}
    case MKL_TRANSFORM_PHASE:{float x=float(id%w),y=float(id/w);float2 ph=mkl_df_add(mkl_df_add(mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[0]),x),x),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[1]),x),y)),mkl_df_mul_f32(mkl_df_mul_f32(p2_df(p.affine[2]),y),y));
        ph=mkl_df_add(ph,mkl_df_add(mkl_df_add(mkl_df_mul_f32(p2_df(p.affine[3]),x),mkl_df_mul_f32(p2_df(p.affine[4]),y)),p2_df(p.affine[5])));z=p2_cmul(a[id],p2_cis(ph));break;}
    }
    out[id]=z;
}
// 32×32 tiles, 256 threads; the padded column avoids bank conflicts.
kernel void mkl_transform_transpose(device const float2 *a [[buffer(0)]],device float2 *out [[buffer(2)]],constant MklTransformParams &p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint2 group [[threadgroup_position_in_grid]]){
    threadgroup float2 tile[32][33];
    uint w=p.width,h=p.height,tx=tid%32,ty=tid/32,bx=group.x*32,by=group.y*32;float scale=as_type<float>(p.scale_bits);
    for(uint j=ty;j<32;j+=8)if(bx+tx<w&&by+j<h)tile[j][tx]=a[(by+j)*w+bx+tx];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint j=ty;j<32;j+=8)if(bx+j<w&&by+tx<h)out[(bx+j)*h+by+tx]=tile[tx][j]*scale;
}

// ---- Stockham FFT. Twiddles are f64-derived host tables (recipe::twiddles):
// stage span p, radix R, entry p-1+k(R-1)+t-1 = exp(-2πi·t·k/(pR)); sign +1 conjugates.
constant float P2_COS[3][8]={{1,-0.5f,-0.5f,0,0,0,0,0},
    {1,0.309016994374947424f,-0.809016994374947424f,-0.809016994374947424f,0.309016994374947424f,0,0,0},
    {1,0.623489801858733531f,-0.222520933956314404f,-0.900968867902419126f,-0.900968867902419126f,-0.222520933956314404f,0.623489801858733531f,0}};
constant float P2_SIN[3][8]={{0,0.866025403784438647f,-0.866025403784438647f,0,0,0,0,0},
    {0,0.951056516295153572f,0.587785252292473129f,-0.587785252292473129f,-0.951056516295153572f,0,0,0},
    {0,0.781831482468029809f,0.974927912181823607f,0.433883739117558120f,-0.433883739117558120f,-0.974927912181823607f,-0.781831482468029809f,0}};
// v·(s·i)
static float2 p2_rot(float2 v,float s){return float2(-s*v.y,s*v.x);}
static void p2_dft4(thread float2 &u0,thread float2 &u1,thread float2 &u2,thread float2 &u3,float s){
    float2 a0=u0+u2,a1=u0-u2,a2=u1+u3,a3=p2_rot(u1-u3,s);
    u0=a0+a2;u2=a0-a2;u1=a1+a3;u3=a1-a3;
}
// Unnormalised DFT of R values, exp(s·2πi·m·t/R).
template<uint R> static void p2_dft(thread float2 *u,float s){
    if(R==2){float2 a=u[0];u[0]=a+u[1];u[1]=a-u[1];}
    else if(R==4)p2_dft4(u[0],u[1],u[2],u[3],s);
    else if(R==8){
        const float c=0.707106781186547524f;
        p2_dft4(u[0],u[2],u[4],u[6],s);p2_dft4(u[1],u[3],u[5],u[7],s);
        float2 o1=c*float2(u[3].x-s*u[3].y,u[3].y+s*u[3].x),o2=p2_rot(u[5],s),o3=c*float2(-u[7].x-s*u[7].y,s*u[7].x-u[7].y);
        float2 e0=u[0],e1=u[2],e2=u[4],e3=u[6];
        u[0]=e0+u[1];u[4]=e0-u[1];u[1]=e1+o1;u[5]=e1-o1;u[2]=e2+o2;u[6]=e2-o2;u[3]=e3+o3;u[7]=e3-o3;
    }else{
        const uint c=R==3?0:R==5?1:2;float2 o[R];
        for(uint m=0;m<R;m++){float2 acc=u[0];for(uint t=1;t<R;t++){uint q=m*t%R;acc+=p2_cmul(u[t],float2(P2_COS[c][q],s*P2_SIN[c][q]));}o[m]=acc;}
        for(uint m=0;m<R;m++)u[m]=o[m];
    }
}
// Twiddle and butterfly of butterfly b (of n/R) at span p; returns output base j.
template<uint R> static uint p2_butterfly(thread float2 *v,uint b,uint p,device const float2 *tw,float s){
    uint k=b%p;device const float2 *w=tw+p-1+k*(R-1);
    for(uint t=1;t<R;t++){float2 c=w[t-1];v[t]=p2_cmul(v[t],float2(c.x,-s*c.y));}
    p2_dft<R>(v,s);
    return (b-k)*R+k;
}
// One in-place Stockham stage in shared memory. T=ceil(n/E) lanes cover n/R
// butterflies in K=ceil(E/R) rounds; all reads precede the barrier and writes.
template<uint R,uint E> static void p2_stage(threadgroup float2 *d,uint n,uint p,uint lane,uint T,device const float2 *tw,float s){
    constexpr uint K=(E+R-1)/R;float2 v[K][R];uint m=n/R;
    for(uint q=0;q<K;q++){uint b=lane+q*T;if(b<m)for(uint t=0;t<R;t++)v[q][t]=d[b+t*m];}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint q=0;q<K;q++){uint b=lane+q*T;if(b<m){uint j=p2_butterfly<R>(v[q],b,p,tw,s);for(uint t=0;t<R;t++)d[j+t*p]=v[q][t];}}
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
// MIXED=false instantiations carry only radices 2/4/8 (less register pressure).
template<uint E,bool MIXED> static void p2_fft(threadgroup float2 *d,constant MklTransformParams &p,uint lane,uint T,device const float2 *tw,float s){
    uint n=p.width,span=1;
    for(uint i=0;i<p.stages;i++){
        uint r=p.radices[i/8]>>(4*(i%8))&15u;
        switch(r){
        case 2:p2_stage<2,E>(d,n,span,lane,T,tw,s);break;
        case 4:p2_stage<4,E>(d,n,span,lane,T,tw,s);break;
        case 8:p2_stage<8,E>(d,n,span,lane,T,tw,s);break;
        default:if(MIXED)switch(r){
            case 3:p2_stage<3,E>(d,n,span,lane,T,tw,s);break;
            case 5:p2_stage<5,E>(d,n,span,lane,T,tw,s);break;
            default:p2_stage<7,E>(d,n,span,lane,T,tw,s);break;
        }
        }
        span*=r;
    }
}
// Shared layout of a group: G=SH/(E·T) rows of n ≤ SH elements, E elements
// per thread, T=ceil(n/E) threads per row. Rows past the batch (live=false)
// still join every barrier.
#define P2_GROUP(SH) constexpr uint E=SH>2048?16:8;threadgroup float2 sh[SH];\
    uint n=p.width,T=(n+E-1)/E,lane=tid%T,row=group*(SH/(E*T))+tid/T;bool live=row<p.height;\
    threadgroup float2 *d=sh+tid/T*n;
template<uint SH,bool MIXED> kernel void p2_fft_rows(device const float2 *a [[buffer(0)]],device const float2 *tw [[buffer(1)]],device float2 *out [[buffer(2)]],constant MklTransformParams &p [[buffer(3)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]){
    P2_GROUP(SH)
    if(live)for(uint i=lane;i<n;i+=T)d[i]=a[row*n+i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p2_fft<E,MIXED>(d,p,lane,T,tw,float(as_type<int>(p.sign_bits)));
    float scale=as_type<float>(p.scale_bits);
    if(live)for(uint i=lane;i<n;i+=T)out[row*n+i]=d[i]*scale;
}
// Exact chirp-z of a row, fused: pre-chirp, FFT, kernel spectrum, inverse
// FFT, post-chirp. chirp holds nx pre- then ox post-factors (recipe::chirp_factors).
template<uint SH,bool MIXED> kernel void p2_chirp_rows(device const float2 *a [[buffer(0)]],device const float2 *spectrum [[buffer(1)]],device float2 *out [[buffer(2)]],constant MklTransformParams &p [[buffer(3)]],device const float2 *tw [[buffer(4)]],device const float2 *chirp [[buffer(5)]],uint tid [[thread_index_in_threadgroup]],uint group [[threadgroup_position_in_grid]]){
    P2_GROUP(SH)
    for(uint i=lane;i<n;i+=T)d[i]=live&&i<p.nx?p2_cmul(a[row*p.nx+i],chirp[i]):float2(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p2_fft<E,MIXED>(d,p,lane,T,tw,-1);
    for(uint i=lane;i<n;i+=T)d[i]=p2_cmul(d[i],spectrum[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    p2_fft<E,MIXED>(d,p,lane,T,tw,1);
    float scale=as_type<float>(p.scale_bits);
    if(live)for(uint i=lane;i<p.ox;i+=T)out[row*p.ox+i]=p2_cmul(d[i],chirp[p.nx+i])*scale;
}
#define P2_ROWS(name,SH,MIXED) template [[host_name(name)]] kernel void p2_fft_rows<SH,MIXED>(device const float2*,device const float2*,device float2*,constant MklTransformParams&,uint,uint);
#define P2_CHIRP(name,SH,MIXED) template [[host_name(name)]] kernel void p2_chirp_rows<SH,MIXED>(device const float2*,device const float2*,device float2*,constant MklTransformParams&,device const float2*,device const float2*,uint,uint);
P2_ROWS("mkl_fft_rows_1k",1024,false) P2_ROWS("mkl_fft_rows_2k",2048,false) P2_ROWS("mkl_fft_rows_4k",4096,false)
P2_ROWS("mkl_fft_rows_1k_mixed",1024,true) P2_ROWS("mkl_fft_rows_2k_mixed",2048,true) P2_ROWS("mkl_fft_rows_4k_mixed",4096,true)
P2_CHIRP("mkl_chirp_rows_1k",1024,false) P2_CHIRP("mkl_chirp_rows_2k",2048,false) P2_CHIRP("mkl_chirp_rows_4k",4096,false)
P2_CHIRP("mkl_chirp_rows_1k_mixed",1024,true) P2_CHIRP("mkl_chirp_rows_2k_mixed",2048,true) P2_CHIRP("mkl_chirp_rows_4k_mixed",4096,true)
// One global Stockham stage (radix radices[0], span) for rows longer than 4096.
kernel void mkl_fft_stage(device const float2 *a [[buffer(0)]],device const float2 *tw [[buffer(1)]],device float2 *out [[buffer(2)]],constant MklTransformParams &p [[buffer(3)]],uint2 id [[thread_position_in_grid]]){
    uint n=p.width,r=p.radices[0],m=n/r,b=id.x;if(b>=m||id.y>=p.height)return;
    device const float2 *x=a+id.y*n;device float2 *y=out+id.y*n;
    float s=float(as_type<int>(p.sign_bits)),scale=as_type<float>(p.scale_bits);float2 v[8];uint j;
    for(uint t=0;t<r;t++)v[t]=x[b+t*m];
    switch(r){
    case 2:j=p2_butterfly<2>(v,b,p.span,tw,s);break;
    case 3:j=p2_butterfly<3>(v,b,p.span,tw,s);break;
    case 4:j=p2_butterfly<4>(v,b,p.span,tw,s);break;
    case 5:j=p2_butterfly<5>(v,b,p.span,tw,s);break;
    case 7:j=p2_butterfly<7>(v,b,p.span,tw,s);break;
    default:j=p2_butterfly<8>(v,b,p.span,tw,s);break;
    }
    for(uint t=0;t<r;t++)y[j+t*p.span]=v[t]*scale;
}

// ---- Correlation over a tile of frequencies: grid (sample k, frequency first+f).
static float2 p2_dot(thread float2 *a,constant float b[3][2]){return mkl_df_add(mkl_df_add(mkl_df_mul(a[0],p2_df(b[0])),mkl_df_mul(a[1],p2_df(b[1]))),mkl_df_mul(a[2],p2_df(b[2])));}
struct P2Geometry {float x,y,z,gamma,source_z;};
static P2Geometry p2_geometry(constant MklCorrelationParams&p,uint k,float2 nu){
    float2 a=mkl_df_mul_f32(p2_df(p.sin_max),float(2*(k%p.n)+1)/float(p.n)-1.0f);
    float2 b=mkl_df_mul_f32(p2_df(p.sin_max),float(2*(k/p.n)+1)/float(p.n)-1.0f);
    float2 gamma=mkl_df_sqrt(mkl_df_sub(mkl_df_sub(float2(1,0),mkl_df_mul(a,a)),mkl_df_mul(b,b))),q[3];
    for(uint i=0;i<3;i++)q[i]=mkl_df_sub(mkl_df_add(mkl_df_mul(p2_df(p.frame[0][i]),a),mkl_df_mul(p2_df(p.frame[1][i]),b)),mkl_df_mul(p2_df(p.frame[2][i]),gamma));
    float2 sx=mkl_df_add(q[0],mkl_df_mul_f32(p2_df(p.wavelength),nu.x)),sy=mkl_df_add(q[1],mkl_df_mul_f32(p2_df(p.wavelength),nu.y));
    float2 z2=mkl_df_sub(mkl_df_sub(float2(1,0),mkl_df_mul(sx,sx)),mkl_df_mul(sy,sy));
    P2Geometry g={mkl_qnan(),mkl_qnan(),0,gamma.x,-q[2].x};
    if(z2.x<=0||q[2].x>=0||!isfinite(gamma.x))return g;
    float2 sz=-mkl_df_sqrt(z2),shifted[3]={sx,sy,sz};
    if(p2_dot(shifted,p.frame[2]).x>=0)return g;
    float2 scale=mkl_df_div(float2(0.5f*float(p.n),0),p2_df(p.sin_max));float offset=0.5f*float(p.n-1);
    float2 ix=mkl_df_add_f32(mkl_df_mul(p2_dot(shifted,p.frame[0]),scale),offset),iy=mkl_df_add_f32(mkl_df_mul(p2_dot(shifted,p.frame[1]),scale),offset);
    g.x=ix.x+ix.y;g.y=iy.x+iy.y;g.z=-sz.x;return g;
}
kernel void mkl_optical_stencil(device const float*freq[[buffer(0)]],device float*dst[[buffer(1)]],constant MklCorrelationParams&p[[buffer(2)]],uint2 id[[thread_position_in_grid]]){
    uint k=id.x,f=p.first+id.y,count=p.n*p.height;if(k>=count||id.y>=p.frequencies)return;
    P2Geometry g=p2_geometry(p,k,float2(freq[2*f],freq[2*f+1]));uint o=(f*count+k)*5;
    dst[o]=g.x;dst[o+1]=g.y;dst[o+2]=g.z;dst[o+3]=g.gamma;dst[o+4]=g.source_z;
}
// Writes channels (real, imaginary, energy) of frequency f at out[(3f+c)·count+k].
kernel void mkl_correlation_terms(device const float*field[[buffer(0)]],device const float2*opd[[buffer(1)]],device const float*cov[[buffer(2)]],device const float*freq[[buffer(3)]],device const float*stencil[[buffer(4)]],device float*terms[[buffer(5)]],constant MklCorrelationParams&p[[buffer(6)]],uint2 id[[thread_position_in_grid]]){
    uint k=id.x,f=p.first+id.y,count=p.n*p.height;if(k>=count||id.y>=p.frequencies)return;
    device float*out=terms+3*id.y*count;out[k]=0;out[count+k]=0;out[2*count+k]=0;
    float c=p.covered?cov[k]:1.0f;if(c<=0)return;float2 nu=float2(freq[2*f],freq[2*f+1]);P2Geometry g;
    if(p.optical){if(p.prepared){uint o=(f*count+k)*5;g={stencil[o],stencil[o+1],stencil[o+2],stencil[o+3],stencil[o+4]};}else g=p2_geometry(p,k,nu);}
    else{
        float2 sx=mkl_df_add(mkl_df_add(mkl_df_mul_f32(p2_df(p.mapping[0]),nu.x),mkl_df_mul_f32(p2_df(p.mapping[1]),nu.y)),p2_df(p.mapping[2]));
        float2 sy=mkl_df_add(mkl_df_add(mkl_df_mul_f32(p2_df(p.mapping[3]),nu.x),mkl_df_mul_f32(p2_df(p.mapping[4]),nu.y)),p2_df(p.mapping[5]));
        g={float(k%p.n)+sx.x+sx.y,float(k/p.n)+sy.x+sy.y,1,1,1};
    }
    float power=p.optical?field[k]*field[k]:field[2*k]*field[2*k]+field[2*k+1]*field[2*k+1];float energy=c*power/(g.gamma*g.source_z);out[2*count+k]=energy;
    if(!isfinite(g.x)||!isfinite(g.y))return;
    float fx=floor(g.x),fy=floor(g.y),tx=g.x-fx,ty=g.y-fy;
    float ar=0,ai=0,coverage=0;float2 phase=0;
    for(int dy=0;dy<2;dy++)for(int dx=0;dx<2;dx++){
        int xi=int(fx)+dx,yi=int(fy)+dy;if(xi<0||yi<0||xi>=int(p.n)||yi>=int(p.height))continue;
        uint j=uint(yi)*p.n+uint(xi);float w=(dx?tx:1-tx)*(dy?ty:1-ty)*(p.covered?cov[j]:1.0f);if(w==0)continue;
        if(p.optical){ar+=w*field[j];coverage+=w;phase=mkl_df_add(phase,mkl_df_mul_f32(opd[j],w));}
        else{ar+=w*field[2*j];ai+=w*field[2*j+1];}
    }
    float deficit=(1-c)*energy*max(0.0f,1-abs(g.x-float(k%p.n)))*max(0.0f,1-abs(g.y-float(k/p.n)));float2 v=0;
    if(p.optical){if(coverage>0){float2 z=p2_cis(mkl_df_sub(mkl_df_div(phase,float2(coverage,0)),opd[k]));v=(c*field[k]/g.gamma*ar/g.z)*z;}}
    else v=c*float2(ar*field[2*k]+ai*field[2*k+1],ai*field[2*k]-ar*field[2*k+1]);
    out[k]=v.x+deficit;out[count+k]=v.y;
}
kernel void mkl_correlation_normalize(device const float2*sums[[buffer(0)]],device float2*out[[buffer(1)]],constant MklCorrelationParams&p[[buffer(2)]],uint f[[thread_position_in_grid]]){
    if(f>=p.frequencies)return;float2 a=mkl_df_div(sums[3*f],sums[3*f+2]),b=mkl_df_div(sums[3*f+1],sums[3*f+2]);out[p.first+f]=float2(a.x+a.y,b.x+b.y);
}
