// Operation-for-operation twin of mklight-gpu/src/surface_math.rs.
#ifndef MKLIGHT_SURFACE_MATH_H
#define MKLIGHT_SURFACE_MATH_H
#include "mklight_common.h"

// A wrapper prevents accidental componentwise float2 arithmetic.
struct MklDf { float2 v; };
static inline MklDf operator+(MklDf a, MklDf b) { return {mkl_df_add(a.v,b.v)}; }
static inline MklDf operator-(MklDf a) { return {mkl_df_neg(a.v)}; }
static inline MklDf operator-(MklDf a, MklDf b) { return a + -b; }
static inline MklDf operator*(MklDf a, MklDf b) { return {mkl_df_mul(a.v,b.v)}; }
static inline MklDf operator/(MklDf a, MklDf b) { return {mkl_df_div(a.v,b.v)}; }
template<typename R> static inline R mkl_parts(float hi, float lo);
template<> inline float mkl_parts<float>(float hi, float lo) { (void)lo; return hi; }
template<> inline MklDf mkl_parts<MklDf>(float hi, float lo) { return {float2(hi,lo)}; }
template<typename R> static inline R mkl_c(float v) { return mkl_parts<R>(v,0.0f); }
static inline float mkl_sqrt(float a) { return sqrt(a); }
static inline MklDf mkl_sqrt(MklDf a) { return {mkl_df_sqrt(a.v)}; }
static inline bool mkl_le(float a, float b) { return a <= b; }
static inline bool mkl_le(MklDf a, float b) { return mkl_df_add_f32(a.v,-b).x <= 0.0f; }
static inline bool mkl_zero(float a) { return a == 0.0f; }
static inline bool mkl_zero(MklDf a) { return a.v.x == 0.0f; }
template<typename R> static inline bool mkl_le_value(R a, R b) { return mkl_le(a-b,0.0f); }

template<typename R> struct MklJet { R v; R x; R y; };
template<typename R> static inline MklJet<R> mkl_jconstant(R v) { return {v,mkl_c<R>(0),mkl_c<R>(0)}; }
template<typename R> static inline MklJet<R> mkl_jvariable(R v, bool x) { return {v,mkl_c<R>(x?1:0),mkl_c<R>(x?0:1)}; }
template<typename R> static inline MklJet<R> mkl_jchain(MklJet<R> a, R v, R d) { return {v,d*a.x,d*a.y}; }
template<typename R> static inline MklJet<R> mkl_jscale(MklJet<R> a, R b) { return {a.v*b,a.x*b,a.y*b}; }
template<typename R> static inline MklJet<R> mkl_jadd(MklJet<R> a, R b) { return {a.v+b,a.x,a.y}; }
template<typename R> static inline MklJet<R> mkl_jsqrt(MklJet<R> a) { R v=mkl_sqrt(a.v); return mkl_jchain(a,v,mkl_c<R>(0.5f)/v); }
template<typename R> static inline MklJet<R> mkl_jrecip(MklJet<R> a) { R r=mkl_c<R>(1)/a.v; return mkl_jchain(a,r,-r*r); }
template<typename R> static inline MklJet<R> operator+(MklJet<R> a,MklJet<R> b) { return {a.v+b.v,a.x+b.x,a.y+b.y}; }
template<typename R> static inline MklJet<R> operator-(MklJet<R> a) { return mkl_jscale(a,mkl_c<R>(-1)); }
template<typename R> static inline MklJet<R> operator-(MklJet<R> a,MklJet<R> b) { return a + -b; }
template<typename R> static inline MklJet<R> operator*(MklJet<R> a,MklJet<R> b) { return {a.v*b.v,a.x*b.v+a.v*b.x,a.y*b.v+a.v*b.y}; }
template<typename R> static inline MklJet<R> operator/(MklJet<R> a,MklJet<R> b) { return a*mkl_jrecip(b); }
template<typename R> static inline R mkl_load(constant float* pool, uint i) { return mkl_parts<R>(pool[2*i],pool[2*i+1]); }
template<typename R> static inline MklJet<R> mkl_poly(constant float* pool,uint start,uint n,MklJet<R> t) {
    auto p=mkl_jconstant(mkl_c<R>(0));
    for (int j=int(n)-1;j>=0;--j) p=mkl_jadd(p*t,mkl_load<R>(pool,start+uint(j)));
    return p;
}
template<typename R> static inline MklJet<R> mkl_even(constant float* pool,uint start,uint n,MklJet<R> t) { return mkl_poly(pool,start,n,t)*t*t; }
template<typename R> static inline bool mkl_conic_sag_jet(R c,R k,MklJet<R> t,thread MklJet<R>& out) {
    R arg=mkl_c<R>(1)-(mkl_c<R>(1)+k)*c*c*t.v;
    if (!mkl_le_value(mkl_c<R>(0),arg)) return false;
    R w=mkl_sqrt(arg);
    out=mkl_jchain(t,c*t.v/(mkl_c<R>(1)+w),c/(mkl_c<R>(2)*w)); return true;
}
template<typename R> static inline void mkl_scaled_jet(MklJet<R> x,MklJet<R> y,R radius,thread MklJet<R>& u,thread MklJet<R>& v,thread MklJet<R>& s) {
    R inv=mkl_c<R>(1)/radius; u=mkl_jscale(x,inv); v=mkl_jscale(y,inv); s=u*u+v*v;
}
template<typename R> static inline bool mkl_normalized_jet(MklJet<R> x,MklJet<R> y,R radius,thread MklJet<R>& u,thread MklJet<R>& v,thread MklJet<R>& s) {
    mkl_scaled_jet(x,y,radius,u,v,s);
    return mkl_le_value(s.v,mkl_parts<R>(1.0f,2e-12f));
}
template<typename R> static inline MklJet<R> mkl_jacobi(uint n,R b,MklJet<R> t) {
    auto p0=mkl_jconstant(mkl_c<R>(1));
    if (n==0) return p0;
    auto p1=mkl_jadd(mkl_jscale(mkl_jadd(t,mkl_c<R>(-1)),(b+mkl_c<R>(2))/mkl_c<R>(2)),mkl_c<R>(1));
    for (uint ki=1;ki<n;++ki) {
        R k=mkl_c<R>(float(ki)),s=mkl_c<R>(2)*k+b;
        R a1=mkl_c<R>(2)*(k+mkl_c<R>(1))*(k+b+mkl_c<R>(1))*s;
        auto a2=mkl_jscale(mkl_jadd(mkl_jscale(t,(s+mkl_c<R>(2))*s),-(b*b)),s+mkl_c<R>(1));
        R a3=mkl_c<R>(2)*k*(k+b)*(s+mkl_c<R>(2));
        auto next=mkl_jscale(a2*p1-mkl_jscale(p0,a3),mkl_c<R>(1)/a1);
        p0=p1; p1=next;
    }
    return p1;
}
// Q-con recurrence: v = 2k+4, d = 2(k+1)(k+5)v.
template<typename R> static inline R mkl_qcon_d(uint ki,R v) { R k=mkl_c<R>(float(ki)); return mkl_c<R>(2)*(k+mkl_c<R>(1))*(k+mkl_c<R>(5))*v; }
template<typename R> static inline MklJet<R> mkl_qcon_a(uint ki,MklJet<R> s) {
    R v=mkl_c<R>(2)*mkl_c<R>(float(ki))+mkl_c<R>(4),d=mkl_qcon_d(ki,v);
    return mkl_jscale(mkl_jadd(mkl_jscale(mkl_jadd(mkl_jscale(s,mkl_c<R>(2)),mkl_c<R>(-1)),(v+mkl_c<R>(2))*v),mkl_c<R>(-16)),(v+mkl_c<R>(1))/d);
}
template<typename R> static inline R mkl_qcon_b(uint ki) {
    R k=mkl_c<R>(float(ki)),v=mkl_c<R>(2)*k+mkl_c<R>(4);
    return mkl_c<R>(2)*k*(k+mkl_c<R>(4))*(v+mkl_c<R>(2))/mkl_qcon_d(ki,v);
}
template<typename R> static inline MklJet<R> mkl_qcon(constant float* pool,uint n,MklJet<R> s) {
    auto b1=mkl_jconstant(mkl_c<R>(0)),b2=b1;
    for (int k=int(n)-1;k>=0;--k) {
        auto b=mkl_jadd(mkl_qcon_a(uint(k),s)*b1-mkl_jscale(b2,mkl_qcon_b<R>(uint(k+1))),mkl_load<R>(pool,uint(k))); b2=b1; b1=b;
    }
    return b1*s*s;
}
template<typename R> static inline MklJet<R> mkl_qbfs(constant float* pool,uint n,MklJet<R> s) {
    auto a1=mkl_jconstant(mkl_c<R>(0)),a2=a1;
    for (int m=int(n)-1;m>=0;--m) {
        auto a=mkl_jadd(mkl_jadd(mkl_jscale(s,mkl_c<R>(-4)),mkl_c<R>(2))*a1-a2,mkl_load<R>(pool,uint(m))); a2=a1; a1=a;
    }
    return mkl_jscale(a1+a2,mkl_c<R>(2));
}
template<typename R> static inline MklJet<R> mkl_zernike(constant float* pool,uint n,bool standard,MklJet<R> x,MklJet<R> y,MklJet<R> s) {
    auto z=mkl_jconstant(mkl_c<R>(0));
    for (uint j=1;j<=n;++j) {
        R a=mkl_load<R>(pool,j-1); if (mkl_zero(a)) continue;
        uint degree; int m;
        if (standard) {
            degree=0; while ((degree+1)*(degree+2)/2<j) ++degree;
            uint k=j-degree*(degree+1)/2-1;
            int am=int(degree%2==0 ? 2*((k+1)/2) : 2*(k/2)+1);
            m=j%2==0?am:-am;
        } else if (j==37) {degree=12;m=0;} else {
            uint k=0; while ((k+1)*(k+1)<j) ++k;
            uint off=j-1-k*k; int am=int(k-off/2); degree=2*k-uint(am); m=off%2==1?-am:am;
        }
        uint am=uint(abs(m));
        auto re=mkl_jconstant(mkl_c<R>(1)),im=mkl_jconstant(mkl_c<R>(0));
        for (uint i=0;i<am;++i) {auto r=re*x-im*y;im=re*y+im*x;re=r;}
        R scale=standard?mkl_sqrt(mkl_c<R>(float((degree+1)*(m==0?1:2)))):mkl_c<R>(1);
        z=z+mkl_jscale(mkl_jacobi((degree-am)/2,mkl_c<R>(float(am)),mkl_jadd(mkl_jscale(s,mkl_c<R>(2)),mkl_c<R>(-1)))*(m>=0?re:im),a*scale);
    }
    return z;
}
static inline constant float* mkl_surface_data(constant MklPackedSurface& s,constant float* pool) { return pool+2ul*s.coefficient_offset; }
template<typename R> static inline bool mkl_surface_domain(constant MklPackedSurface& s,constant float* pool,R x,R y) {
    MklJet<R> u,v,t;
    return mkl_normalized_jet(mkl_jvariable(x,true),mkl_jvariable(y,false),mkl_load<R>(mkl_surface_data(s,pool),0),u,v,t);
}
template<typename R> static inline MklJet<R> mkl_phase(constant MklPackedSurface& s,constant float* pool,R x,R y) {
    constant float* data=mkl_surface_data(s,pool);
    MklJet<R> u,v,t; mkl_scaled_jet(mkl_jvariable(x,true),mkl_jvariable(y,false),mkl_load<R>(data,0),u,v,t);
    return mkl_poly(data,MKL_COEFF_PREFIX+MKL_MAX_WAVELENGTHS,uint(data[2*MKL_COEFF_NA]),t)*t;
}
template<typename R> static inline bool mkl_extended_sag(constant MklPackedSurface& s,constant float* pool,R xv,R yv,thread MklJet<R>& out) {
    constant float* data=mkl_surface_data(s,pool);
    R radius=mkl_load<R>(data,0);
    uint na=uint(data[2*MKL_COEFF_NA]),nb=uint(data[2*MKL_COEFF_NB]);
    constant float* coeff=data+2*MKL_COEFF_PREFIX;
    auto x=mkl_jvariable(xv,true),y=mkl_jvariable(yv,false),r2=x*x+y*y;
    R c=mkl_parts<R>(s.curvature,s.curvature_lo),k=mkl_parts<R>(s.conic,s.conic_lo);
    if (s.shape==MKL_SHAPE_ODD_ASPHERE) {
        MklJet<R> base; if (!mkl_conic_sag_jet(c,k,r2,base)) return false;
        if (mkl_zero(x.v) && mkl_zero(y.v)) out=base; else {
            MklJet<R> radial;
            if (mkl_zero(r2.v)) {
                R ax=mkl_le(x.v,0)?-x.v:x.v,ay=mkl_le(y.v,0)?-y.v:y.v;
                R scale=mkl_le_value(ax,ay)?ay:ax;
                R u=x.v/scale,v=y.v/scale,w=mkl_sqrt(u*u+v*v);
                radial={scale*w,u/w,v/w};
            } else radial=mkl_jsqrt(r2);
            auto r=mkl_jscale(radial,mkl_c<R>(1)/radius); out=base+mkl_poly(coeff,0,na,r)*r;
        }
    } else if (s.shape==MKL_SHAPE_TOROID || s.shape==MKL_SHAPE_BICONIC) {
        R cx=mkl_load<R>(data,MKL_COEFF_CX),inv=mkl_c<R>(1)/radius;
        auto xx=mkl_jscale(x,inv),yy=mkl_jscale(y,inv);
        if (s.shape==MKL_SHAPE_TOROID) {
            MklJet<R> base; if (!mkl_conic_sag_jet(c,k,y*y,base)) return false;
            auto zy=base+mkl_even(coeff,na,nb,yy*yy);
            auto a=mkl_jadd(mkl_jscale(-zy,cx),mkl_c<R>(1)),q=a*a-mkl_jscale(x*x,cx*cx);
            if (mkl_le(a.v,0) || !mkl_le_value(mkl_c<R>(0),q.v)) return false;
            out=zy+mkl_jscale(x*x,cx)/(a+mkl_jsqrt(q));
        } else {
            R kx=mkl_load<R>(data,MKL_COEFF_KX);
            auto arg=mkl_jadd(mkl_jscale(x*x,-(mkl_c<R>(1)+kx)*cx*cx)+mkl_jscale(y*y,-(mkl_c<R>(1)+k)*c*c),mkl_c<R>(1));
            if (!mkl_le_value(mkl_c<R>(0),arg.v)) return false;
            out=(mkl_jscale(x*x,cx)+mkl_jscale(y*y,c))/mkl_jadd(mkl_jsqrt(arg),mkl_c<R>(1))+mkl_even(coeff,0,na,xx*xx)+mkl_even(coeff,na,nb,yy*yy);
        }
    } else {
        MklJet<R> u,v,t,base;
        if (!mkl_normalized_jet(x,y,radius,u,v,t) || !mkl_conic_sag_jet(c,k,r2,base)) return false;
        if (s.shape==MKL_SHAPE_Q_CON) out=base+mkl_qcon(coeff,na,t);
        else if (s.shape==MKL_SHAPE_Q_BFS) {
            if (na==0) out=base; else {
                auto w=mkl_jsqrt(mkl_jadd(mkl_jscale(r2,-c*c),mkl_c<R>(1)));
                if (mkl_zero(w.v)) return false;
                out=base+t*mkl_jadd(-t,mkl_c<R>(1))*mkl_qbfs(coeff,na,t)/w;
            }
        } else if (s.shape==MKL_SHAPE_XY) {
            uint start=0,degree=1; auto z=base;
            while (start<na) {
                auto row=mkl_jconstant(mkl_c<R>(0)),vn=mkl_jconstant(mkl_c<R>(1));
                for (uint n=0;n<=degree;++n) {
                    R a=start+n<na?mkl_load<R>(coeff,start+n):mkl_c<R>(0);
                    row=row*u+mkl_jscale(vn,a); vn=vn*v;
                }
                z=z+row; start+=degree+1; ++degree;
            }
            out=z;
        } else out=base+mkl_zernike(coeff,na,data[2*MKL_COEFF_KIND]!=0.0f,u,v,t);
    }
    return true;
}
// The base of an overlaid surface may be legacy (or a DOE's legacy base).
template<typename R> static inline bool mkl_legacy_sag(constant MklPackedSurface& s,R xv,R yv,thread MklJet<R>& out) {
    auto x=mkl_jvariable(xv,true),y=mkl_jvariable(yv,false),r2=x*x+y*y;
    MklJet<R> base;
    if (!mkl_conic_sag_jet(mkl_parts<R>(s.curvature,s.curvature_lo),mkl_parts<R>(s.conic,s.conic_lo),r2,base)) return false;
    auto t=mkl_jscale(r2,mkl_c<R>(s.asphere_inv_norm2));
    auto p=mkl_jconstant(mkl_c<R>(0));
    for (int j=int(s.num_asphere)-1;j>=0;--j) p=mkl_jadd(p*t,mkl_parts<R>(s.asphere[j],s.asphere_lo[j]));
    out=base+p*t*t; return true;
}
// Sag jet of an extended surface or, with OVERLAYS, of any surface with
// MKL_SURF_HAS_SAG_OVERLAY. Code without OVERLAYS never sees the overlay.
template<bool OVERLAYS,typename R> static inline bool mkl_surface_eval(constant MklPackedSurface& s,constant float* pool,R xv,R yv,thread MklJet<R>& out) {
    if (!OVERLAYS || (s.flags&MKL_SURF_HAS_SAG_OVERLAY)==0u) return mkl_extended_sag(s,pool,xv,yv,out);
    constant float* data=mkl_surface_data(s,pool);
    uint shape=s.shape==MKL_SHAPE_DIFFRACTIVE?uint(data[2*MKL_COEFF_KIND]):s.shape;
    MklJet<R> base;
    if (!(shape<=MKL_SHAPE_EVEN_ASPHERE?mkl_legacy_sag(s,xv,yv,base):mkl_extended_sag(s,pool,xv,yv,base))) return false;
    constant float* d=data+2ul*(s.coefficient_count-MKL_OVERLAY_COUNT);
    R inv=mkl_load<R>(d,MKL_OVERLAY_INV_RADIUS),c=mkl_load<R>(d,MKL_OVERLAY_COS),sn=mkl_load<R>(d,MKL_OVERLAY_SIN);
    R s4=mkl_load<R>(d,MKL_OVERLAY_S4),two=mkl_c<R>(2);
    R u=xv*inv,v=yv*inv,t=u*u+v*v;
    R q=mkl_load<R>(d,MKL_OVERLAY_S2)+t*s4,w=q+t*s4;
    out=base+MklJet<R>{c*((u-v)*(u+v))+two*sn*(u*v)+t*q,two*inv*(u*(c+w)+sn*v),two*inv*(v*(w-c)+sn*u)};
    return true;
}
#endif
