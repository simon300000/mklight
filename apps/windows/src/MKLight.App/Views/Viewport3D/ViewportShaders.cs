namespace MKLight.App.Views.Viewport3D;

/// <summary>
/// The layout viewport's HLSL (Shader Model 5), compiled at run time by d3dcompiler_47.
///
/// Polylines are drawn as instanced screen-space quads: one instance per segment, four vertices
/// (a triangle strip) per instance. The quad is expanded in pixels around the projected segment,
/// and the pixel shader computes the distance to the segment for analytic anti-aliasing with
/// round caps (which also give round joins between consecutive segments). MSAA handles the
/// remaining edges (fills, overlaps).
///
/// Numerical notes: positions arrive relative to the scene centre (or the camera at deep zoom),
/// subtracted in double on the CPU before rounding to float. Divisions by w and by the segment
/// length are guarded, and degenerate (zero-length) segments fall back to a fixed direction
/// instead of producing NaNs.
/// </summary>
static class ViewportShaders
{
    public const string Source = """
        struct Segment {
            float4 p0;      // xyz: start (frame mm)
            float4 p1;      // xyz: end
            float4 color;   // straight (non-premultiplied) RGBA
            float4 params;  // x: width (px), y: dash period (px, 0 = solid)
        };

        struct FillVertex {
            float4 position; // xyz: frame mm
            float4 color;    // straight RGBA
        };

        cbuffer Uniforms : register(b0) {
            row_major float4x4 viewProjection;
            float2 viewportSize;   // render target size in pixels
            float feather;         // quad margin beyond the half-width (px), room for the 0.5 px edge ramp
            float pad;
        };

        StructuredBuffer<Segment> segments : register(t0);
        StructuredBuffer<FillVertex> fillVertices : register(t1);

        struct LineOut {
            float4 position : SV_Position;
            float4 color : COLOR0;
            float2 local : TEXCOORD0;                  // x: along the segment from p0 (px), y: across (px)
            nointerpolation float halfWidth : TEXCOORD1;
            nointerpolation float len : TEXCOORD2;
            nointerpolation float dash : TEXCOORD3;
        };

        float2 toScreen(float4 clip) {
            float w = clip.w > 1e-6f ? clip.w : 1e-6f;
            return (clip.xy / w * 0.5f + 0.5f) * viewportSize;
        }

        LineOut line_vs(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
            Segment s = segments[iid];
            float4 c0 = mul(viewProjection, float4(s.p0.xyz, 1.0f));
            float4 c1 = mul(viewProjection, float4(s.p1.xyz, 1.0f));
            float2 a = toScreen(c0);
            float2 b = toScreen(c1);

            float2 d = b - a;
            float len = length(d);
            float2 dir = len > 1e-4f ? d / len : float2(1.0f, 0.0f);
            float2 nrm = float2(-dir.y, dir.x);

            float halfWidth = 0.5f * max(s.params.x, 0.0f);
            float ext = halfWidth + feather;

            bool atEnd = (vid >> 1) != 0;
            float side = (vid & 1) != 0 ? 1.0f : -1.0f;
            float2 base = atEnd ? b : a;
            float along = atEnd ? ext : -ext;
            float2 p = base + dir * along + nrm * (side * ext);

            float4 clip = atEnd ? c1 : c0;
            float w = clip.w > 1e-6f ? clip.w : 1e-6f;
            float2 ndc = p / viewportSize * 2.0f - 1.0f;

            LineOut o;
            o.position = float4(ndc, clamp(clip.z / w, 0.0f, 1.0f), 1.0f);
            o.color = s.color;
            o.local = float2(atEnd ? len + ext : -ext, side * ext);
            o.halfWidth = halfWidth;
            o.len = len;
            o.dash = s.params.y;
            return o;
        }

        float4 line_ps(LineOut i) : SV_Target {
            // Distance to the segment [0, len] x {0}: round caps beyond the ends.
            float beyond = max(max(-i.local.x, i.local.x - i.len), 0.0f);
            float dist = length(float2(beyond, i.local.y));
            float alpha = saturate(i.halfWidth + 0.5f - dist) * i.color.a;
            if (i.dash > 0.0f) {
                float phase = fmod(max(i.local.x, 0.0f), i.dash);
                alpha *= phase < 0.6f * i.dash ? 1.0f : 0.0f;
            }
            if (alpha <= 0.0f) discard;
            return float4(i.color.rgb * alpha, alpha); // premultiplied
        }

        struct FillOut {
            float4 position : SV_Position;
            float4 color : COLOR0;
        };

        FillOut fill_vs(uint vid : SV_VertexID) {
            FillOut o;
            float4 c = mul(viewProjection, float4(fillVertices[vid].position.xyz, 1.0f));
            o.position = float4(c.xy, clamp(c.z, 0.0f, 1.0f) * c.w, c.w);
            o.color = fillVertices[vid].color;
            return o;
        }

        float4 fill_ps(FillOut i) : SV_Target {
            return float4(i.color.rgb * i.color.a, i.color.a);
        }
        """;
}
