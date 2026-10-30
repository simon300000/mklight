// MKLight layout viewport shaders.
//
// Polylines are drawn as instanced screen-space quads: one instance per
// segment, four vertices (a triangle strip) per instance. The quad is
// expanded in pixels around the projected segment, and the fragment shader
// computes the distance to the segment for analytic anti-aliasing with round
// caps (which also give round joins between consecutive segments). MSAA
// handles the remaining edges (fills, overlaps).
//
// Numerical notes: positions arrive relative to the scene centre (the CPU
// subtracts it in double precision before rounding to float), so float32
// keeps sub-micrometre resolution for lens-sized scenes. Divisions by `w` and
// by the segment length are guarded, and degenerate (zero-length) segments
// fall back to a fixed direction instead of producing NaNs.

#include <metal_stdlib>
using namespace metal;

struct Segment {
    float4 p0;      // xyz: start (scene-centred mm)
    float4 p1;      // xyz: end
    float4 color;   // straight (non-premultiplied) RGBA
    float4 params;  // x: width (px), y: dash period (px, 0 = solid)
};

struct Uniforms {
    float4x4 viewProjection;
    float2 viewportSize;   // drawable size in pixels
    float feather;         // anti-aliasing ramp half-width (px)
    float pad;
};

struct LineOut {
    float4 position [[position]];
    float4 color;
    float2 local;          // x: along the segment from p0 (px), y: across (px)
    float halfWidth [[flat]];
    float length [[flat]];
    float dash [[flat]];
};

static inline float2 toScreen(float4 clip, float2 viewport) {
    // Guard against w → 0 (cannot happen with the orthographic cameras used
    // today, but keeps a perspective camera from producing infinities).
    float w = clip.w > 1e-6f ? clip.w : 1e-6f;
    return fma(clip.xy / w, float2(0.5f), float2(0.5f)) * viewport;
}

vertex LineOut line_vertex(uint vid [[vertex_id]],
                           uint iid [[instance_id]],
                           const device Segment *segments [[buffer(0)]],
                           constant Uniforms &u [[buffer(1)]]) {
    Segment s = segments[iid];
    float4 c0 = u.viewProjection * float4(s.p0.xyz, 1.0f);
    float4 c1 = u.viewProjection * float4(s.p1.xyz, 1.0f);
    float2 a = toScreen(c0, u.viewportSize);
    float2 b = toScreen(c1, u.viewportSize);

    float2 d = b - a;
    float len = length(d);
    float2 dir = len > 1e-4f ? d / len : float2(1.0f, 0.0f);
    float2 nrm = float2(-dir.y, dir.x);

    float halfWidth = 0.5f * max(s.params.x, 0.0f);
    float ext = halfWidth + u.feather;

    bool atEnd = (vid >> 1) != 0;
    float side = (vid & 1) != 0 ? 1.0f : -1.0f;
    float2 base = atEnd ? b : a;
    float along = atEnd ? ext : -ext;
    float2 p = base + dir * along + nrm * (side * ext);

    float4 clip = atEnd ? c1 : c0;
    float w = clip.w > 1e-6f ? clip.w : 1e-6f;
    float2 ndc = fma(p / u.viewportSize, float2(2.0f), float2(-1.0f));

    LineOut out;
    out.position = float4(ndc, clamp(clip.z / w, 0.0f, 1.0f), 1.0f);
    out.color = s.color;
    out.local = float2(atEnd ? len + ext : -ext, side * ext);
    out.halfWidth = halfWidth;
    out.length = len;
    out.dash = s.params.y;
    return out;
}

fragment float4 line_fragment(LineOut in [[stage_in]]) {
    // Distance to the segment [0, length] × {0}: round caps beyond the ends.
    float beyond = max(max(-in.local.x, in.local.x - in.length), 0.0f);
    float dist = length(float2(beyond, in.local.y));
    float alpha = saturate(in.halfWidth + 0.5f - dist) * in.color.a;
    if (in.dash > 0.0f) {
        float phase = fmod(max(in.local.x, 0.0f), in.dash);
        alpha *= phase < 0.6f * in.dash ? 1.0f : 0.0f;
    }
    if (alpha <= 0.0f) {
        discard_fragment();
    }
    return float4(in.color.rgb * alpha, alpha);  // premultiplied
}

struct FillVertex {
    float4 position;  // xyz: scene-centred mm
    float4 color;     // straight RGBA
};

struct FillOut {
    float4 position [[position]];
    float4 color;
};

vertex FillOut fill_vertex(uint vid [[vertex_id]],
                           const device FillVertex *vertices [[buffer(0)]],
                           constant Uniforms &u [[buffer(1)]]) {
    FillOut out;
    float4 c = u.viewProjection * float4(vertices[vid].position.xyz, 1.0f);
    out.position = float4(c.xy, clamp(c.z, 0.0f, 1.0f) * c.w, c.w);
    out.color = vertices[vid].color;
    return out;
}

fragment float4 fill_fragment(FillOut in [[stage_in]]) {
    return float4(in.color.rgb * in.color.a, in.color.a);
}
