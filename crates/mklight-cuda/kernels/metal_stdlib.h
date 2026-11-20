// metal_stdlib for CUDA: the subset of the Metal Shading Language the MKLight
// Metal sources use, so that crates/mklight-metal/shaders/*.h compile
// unchanged under NVRTC (provided to it as the header `<metal_stdlib>`).
//
// Numerics: compiled with --fmad=false --prec-div=true --prec-sqrt=true
// --ftz=false and no fast math (kernels.rs), so `+ - * /` and `sqrt` are
// IEEE binary32 round-to-nearest, nothing is contracted, and `fma` is the
// fused hardware operation: the arithmetic the Rust specification
// (mklight_gpu::{df, fmath, trace_kernel}) models in its IEEE environment.
// Functions without an execution-space annotation are __device__
// (--device-as-default-execution-space), as in Metal.
#ifndef MKLIGHT_CUDA_METAL_STDLIB
#define MKLIGHT_CUDA_METAL_STDLIB

typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long long ulong;

// Address spaces carry no meaning here, but the constant address space is
// read-only (so `constant T&` and `constant T*` accept const arguments).
#define constant const
#define device
#define thread

#ifndef INFINITY
#define INFINITY __int_as_float(0x7f800000)
#endif
#ifndef NAN
#define NAN __int_as_float(0x7fc00000)
#endif
#define FLT_MAX __int_as_float(0x7f7fffff)
#define FLT_MIN __int_as_float(0x00800000)
#define FLT_EPSILON __int_as_float(0x34000000)

namespace mkl_msl {

// Components are named members; operations are written per component (no
// loops over a dynamic index), so every vector stays in registers.
template <typename T> struct vec2 {
    T x, y;
    vec2() = default;
    explicit constexpr vec2(T s) : x(s), y(s) {}
    constexpr vec2(T a, T b) : x(a), y(b) {}
    template <typename U> explicit constexpr vec2(vec2<U> v) : x(T(v.x)), y(T(v.y)) {}
    T& operator[](int i) { return (&x)[i]; }
    T operator[](int i) const { return (&x)[i]; }
    template <typename F> static vec2 map(vec2 a, vec2 b, F f) { return vec2(f(a.x, b.x), f(a.y, b.y)); }
};
template <typename T> struct vec3 {
    T x, y, z;
    vec3() = default;
    explicit constexpr vec3(T s) : x(s), y(s), z(s) {}
    constexpr vec3(T a, T b, T c) : x(a), y(b), z(c) {}
    constexpr vec3(vec2<T> a, T c) : x(a.x), y(a.y), z(c) {}
    template <typename U> explicit constexpr vec3(vec3<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)) {}
    T& operator[](int i) { return (&x)[i]; }
    T operator[](int i) const { return (&x)[i]; }
    template <typename F> static vec3 map(vec3 a, vec3 b, F f) {
        return vec3(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z));
    }
};
template <typename T> struct vec4 {
    T x, y, z, w;
    vec4() = default;
    explicit constexpr vec4(T s) : x(s), y(s), z(s), w(s) {}
    constexpr vec4(T a, T b, T c, T d) : x(a), y(b), z(c), w(d) {}
    constexpr vec4(vec2<T> a, vec2<T> b) : x(a.x), y(a.y), z(b.x), w(b.y) {}
    constexpr vec4(vec3<T> a, T d) : x(a.x), y(a.y), z(a.z), w(d) {}
    template <typename U> explicit constexpr vec4(vec4<U> v) : x(T(v.x)), y(T(v.y)), z(T(v.z)), w(T(v.w)) {}
    T& operator[](int i) { return (&x)[i]; }
    T operator[](int i) const { return (&x)[i]; }
    template <typename F> static vec4 map(vec4 a, vec4 b, F f) {
        return vec4(f(a.x, b.x), f(a.y, b.y), f(a.z, b.z), f(a.w, b.w));
    }
};

#define MKL_MSL_VEC_OPS(V)                                                                             template <typename T> V<T> operator+(V<T> a, V<T> b) { return V<T>::map(a, b, [](T p, T q) { return p + q; }); }     template <typename T> V<T> operator-(V<T> a, V<T> b) { return V<T>::map(a, b, [](T p, T q) { return p - q; }); }     template <typename T> V<T> operator*(V<T> a, V<T> b) { return V<T>::map(a, b, [](T p, T q) { return p * q; }); }     template <typename T> V<T> operator/(V<T> a, V<T> b) { return V<T>::map(a, b, [](T p, T q) { return p / q; }); }     template <typename T> V<T> operator^(V<T> a, V<T> b) { return V<T>::map(a, b, [](T p, T q) { return p ^ q; }); }     template <typename T> V<T> operator+(V<T> a, T b) { return a + V<T>(b); }                        template <typename T> V<T> operator-(V<T> a, T b) { return a - V<T>(b); }                        template <typename T> V<T> operator*(V<T> a, T b) { return a * V<T>(b); }                        template <typename T> V<T> operator/(V<T> a, T b) { return a / V<T>(b); }                        template <typename T> V<T> operator+(T a, V<T> b) { return V<T>(a) + b; }                        template <typename T> V<T> operator-(T a, V<T> b) { return V<T>(a) - b; }                        template <typename T> V<T> operator*(T a, V<T> b) { return V<T>(a) * b; }                        template <typename T> V<T> operator/(T a, V<T> b) { return V<T>(a) / b; }                        template <typename T> V<T> operator-(V<T> a) { return V<T>::map(a, a, [](T p, T) { return -p; }); }     template <typename T> V<T>& operator+=(V<T>& a, V<T> b) { a = a + b; return a; }                 template <typename T> V<T>& operator-=(V<T>& a, V<T> b) { a = a - b; return a; }                 template <typename T> V<T>& operator*=(V<T>& a, V<T> b) { a = a * b; return a; }                 template <typename T> V<T>& operator*=(V<T>& a, T b) { a = a * b; return a; }                    template <typename T> V<T>& operator/=(V<T>& a, T b) { a = a / b; return a; }                    template <typename T> V<T> fabs(V<T> a) { return V<T>::map(a, a, [](T p, T) { return ::fabsf(p); }); }     template <typename T> V<T> abs(V<T> a) { return fabs(a); }                                        template <typename T> V<T> select(V<T> a, V<T> b, bool c) { return c ? b : a; }                   template <typename T> V<T> min(V<T> a, V<T> b) { return V<T>::map(a, b, [](T p, T q) { return ::fminf(p, q); }); }     template <typename T> V<T> max(V<T> a, V<T> b) { return V<T>::map(a, b, [](T p, T q) { return ::fmaxf(p, q); }); }     template <typename T> V<T> floor(V<T> a) { return V<T>::map(a, a, [](T p, T) { return ::floorf(p); }); }

MKL_MSL_VEC_OPS(vec2)
MKL_MSL_VEC_OPS(vec3)
MKL_MSL_VEC_OPS(vec4)
#undef MKL_MSL_VEC_OPS

template <typename T> T dot(vec2<T> a, vec2<T> b) { return a.x * b.x + a.y * b.y; }
template <typename T> T dot(vec3<T> a, vec3<T> b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
template <typename T> T dot(vec4<T> a, vec4<T> b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }
template <typename T> T length(vec2<T> a) { return ::sqrtf(dot(a, a)); }
template <typename T> T length(vec3<T> a) { return ::sqrtf(dot(a, a)); }

// as_type<T>(x): reinterpret the bits.
template <typename T, typename U> T as_type(U x) {
    static_assert(sizeof(T) == sizeof(U), "as_type needs equal sizes");
    T r;
    memcpy(&r, &x, sizeof(T));
    return r;
}

// Scalars (MSL semantics; fmin/fmax drop a NaN operand like MSL).
inline float select(float a, float b, bool c) { return c ? b : a; }
inline uint select(uint a, uint b, bool c) { return c ? b : a; }
inline int select(int a, int b, bool c) { return c ? b : a; }
inline float clamp(float x, float lo, float hi) { return ::fminf(::fmaxf(x, lo), hi); }
inline uint clamp(uint x, uint lo, uint hi) { return ::min(::max(x, lo), hi); }
inline int clamp(int x, int lo, int hi) { return ::min(::max(x, lo), hi); }
inline float sign(float x) { return x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : (x != x ? 0.0f : x)); }
inline float step(float edge, float x) { return x < edge ? 0.0f : 1.0f; }
inline uint mulhi(uint a, uint b) { return __umulhi(a, b); }
inline uint popcount(uint x) { return (uint)__popc(x); }
inline uint clz(uint x) { return (uint)__clz((int)x); }
inline uint ctz(uint x) { return x == 0u ? 32u : (uint)(__ffs((int)x) - 1); }

} // namespace mkl_msl

namespace metal { using namespace mkl_msl; }
using namespace mkl_msl;

// CUDA declares its own float2, uint4, ... structs; the MSL names refer to
// these instead from here on.

#define float2 mkl_msl::vec2<float>
#define float3 mkl_msl::vec3<float>
#define float4 mkl_msl::vec4<float>
#define uint2 mkl_msl::vec2<uint>
#define uint3 mkl_msl::vec3<uint>
#define uint4 mkl_msl::vec4<uint>
#define int2 mkl_msl::vec2<int>
#define int3 mkl_msl::vec3<int>
#define int4 mkl_msl::vec4<int>
#define ushort2 mkl_msl::vec2<ushort>

// Metal's precise:: and fast:: namespaces: the IEEE functions here.
namespace precise {
inline float sqrt(float x) { return ::sqrtf(x); }
inline float divide(float a, float b) { return a / b; }
}

#endif // MKLIGHT_CUDA_METAL_STDLIB
