using System.Numerics;
using System.Runtime.InteropServices;
using MKLight.Engine;
using Vortice.D3DCompiler;
using Vortice.Direct3D;
using Vortice.Direct3D11;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace MKLight.App.Views.Viewport3D;

/// <summary>
/// Draws an engine <see cref="LayoutScene"/> with Direct3D 11 (the Mac app's Metal
/// ViewportRenderer): anti-aliased wide polylines as instanced screen-space quads, translucent
/// fills, 4× MSAA resolved into the target. Draws on demand. Owns no window: the target is a
/// composition swap chain (<see cref="Viewport3DView"/>) or an offscreen texture (captures).
/// </summary>
public sealed class ViewportRenderer : IDisposable
{
    public ID3D11Device Device { get; }
    readonly ID3D11DeviceContext _context;
    readonly ID3D11VertexShader _lineVS, _fillVS;
    readonly ID3D11PixelShader _linePS, _fillPS;
    readonly ID3D11BlendState _blend;
    readonly ID3D11RasterizerState _raster;
    readonly ID3D11Buffer _uniforms;
    public uint SampleCount { get; }

    ID3D11Texture2D? _msaa;
    ID3D11RenderTargetView? _msaaView;
    (uint W, uint H) _msaaSize;

    ID3D11Buffer? _segmentBuffer, _fillBuffer;
    ID3D11ShaderResourceView? _segmentView, _fillView;
    int _segmentCount, _fillCount;

    LayoutScene? _scene;
    ViewportStyle? _style;
    GeometryFrame? _frame;
    double _depthRadius = 1;

    static readonly FeatureLevel[] Levels = [FeatureLevel.Level_11_1, FeatureLevel.Level_11_0, FeatureLevel.Level_10_1, FeatureLevel.Level_10_0];

    /// <summary>Creates the device (hardware, else WARP), compiles the shaders and the fixed states.</summary>
    public ViewportRenderer()
    {
        var flags = DeviceCreationFlags.BgraSupport;
        if (D3D11.D3D11CreateDevice(IntPtr.Zero, DriverType.Hardware, flags, Levels, out var device, out var context).Failure)
            D3D11.D3D11CreateDevice(IntPtr.Zero, DriverType.Warp, flags, Levels, out device, out context).CheckError();
        Device = device!;
        _context = context!;

        SampleCount = 1;
        foreach (var n in new uint[] { 4, 2 })
        {
            if (Device.CheckMultisampleQualityLevels(Format.B8G8R8A8_UNorm, n) > 0) { SampleCount = n; break; }
        }

        ReadOnlyMemory<byte> Compile(string entry, string profile) =>
            Compiler.Compile(ViewportShaders.Source, entry, "Viewport.hlsl", profile, ShaderFlags.OptimizationLevel3, EffectFlags.None);
        _lineVS = Device.CreateVertexShader(Compile("line_vs", "vs_5_0").Span);
        _linePS = Device.CreatePixelShader(Compile("line_ps", "ps_5_0").Span);
        _fillVS = Device.CreateVertexShader(Compile("fill_vs", "vs_5_0").Span);
        _fillPS = Device.CreatePixelShader(Compile("fill_ps", "ps_5_0").Span);

        // Premultiplied-alpha "over".
        var blend = new BlendDescription();
        blend.RenderTarget[0] = new RenderTargetBlendDescription
        {
            BlendEnable = true,
            SourceBlend = Blend.One, DestinationBlend = Blend.InverseSourceAlpha, BlendOperation = BlendOperation.Add,
            SourceBlendAlpha = Blend.One, DestinationBlendAlpha = Blend.InverseSourceAlpha, BlendOperationAlpha = BlendOperation.Add,
            RenderTargetWriteMask = ColorWriteEnable.All,
        };
        _blend = Device.CreateBlendState(blend);
        _raster = Device.CreateRasterizerState(new RasterizerDescription(CullMode.None, FillMode.Solid)
        {
            MultisampleEnable = SampleCount > 1,
            DepthClipEnable = false,
        });
        _uniforms = Device.CreateBuffer(new BufferDescription((uint)Marshal.SizeOf<Uniforms>(), BindFlags.ConstantBuffer, ResourceUsage.Default));
    }

    /// <summary>The camera and mode the next frame is drawn with.</summary>
    public ViewportCamera Camera { get; set; } = new();
    public ViewportMode Mode { get; set; } = ViewportMode.Orbit3D;

    /// <summary>Replaces the drawn geometry (rebuilt on the next frame).</summary>
    public void SetScene(LayoutScene? scene, ViewportStyle style)
    {
        if (ReferenceEquals(scene, _scene) && Equals(style, _style)) return;
        _scene = scene;
        _style = style;
        _frame = null;
        if (scene is null) ReleaseGeometry();
    }

    void ReleaseGeometry()
    {
        _segmentView?.Dispose(); _segmentView = null;
        _segmentBuffer?.Dispose(); _segmentBuffer = null;
        _fillView?.Dispose(); _fillView = null;
        _fillBuffer?.Dispose(); _fillBuffer = null;
        _segmentCount = _fillCount = 0;
    }

    /// <summary>(Re)builds the GPU buffers when the scene, style or geometry frame changed; the frame follows the camera only at deep zoom.</summary>
    void PrepareGeometry(double width, double height)
    {
        if (_scene is not { } scene || _style is not { } style) return;
        var wanted = SceneGeometry.Frame(scene, Camera, Mode, width, height);
        if (_frame == wanted) return;
        _frame = wanted;
        var b = scene.Bounds;
        var diagonal = (V3.Of(b.Max) - V3.Of(b.Min)).Length;
        _depthRadius = Math.Max(Math.Max(diagonal, SceneGeometry.SceneReach(scene, Camera.Center)), 1);
        var (segs, fills) = SceneGeometry.Build(scene, style with { Mode = Mode }, wanted);
        ReleaseGeometry();
        (_segmentBuffer, _segmentView) = Structured(CollectionsMarshal.AsSpan(segs));
        (_fillBuffer, _fillView) = Structured(CollectionsMarshal.AsSpan(fills));
        _segmentCount = segs.Count;
        _fillCount = fills.Count;
    }

    (ID3D11Buffer?, ID3D11ShaderResourceView?) Structured<T>(Span<T> data) where T : unmanaged
    {
        if (data.Length == 0) return (null, null);
        var stride = (uint)Marshal.SizeOf<T>();
        var desc = new BufferDescription((uint)data.Length * stride, BindFlags.ShaderResource, ResourceUsage.Immutable,
            CpuAccessFlags.None, ResourceOptionFlags.BufferStructured, stride);
        var buffer = Device.CreateBuffer(data, desc);
        var view = Device.CreateShaderResourceView(buffer, new ShaderResourceViewDescription
        {
            Format = Format.Unknown,
            ViewDimension = ShaderResourceViewDimension.Buffer,
            Buffer = new BufferShaderResourceView { FirstElement = 0, NumElements = (uint)data.Length },
        });
        return (buffer, view);
    }

    void EnsureMsaa(uint w, uint h)
    {
        if (_msaa is not null && _msaaSize == (w, h)) return;
        _msaaView?.Dispose();
        _msaa?.Dispose();
        _msaa = Device.CreateTexture2D(new Texture2DDescription(Format.B8G8R8A8_UNorm, w, h, 1, 1, BindFlags.RenderTarget,
            ResourceUsage.Default, CpuAccessFlags.None, SampleCount, 0));
        _msaaView = Device.CreateRenderTargetView(_msaa);
        _msaaSize = (w, h);
    }

    /// <summary>
    /// Draws a frame into <paramref name="target"/> (a single-sampled B8G8R8A8 texture of
    /// <paramref name="pixelWidth"/> × <paramref name="pixelHeight"/>) for a view of
    /// <paramref name="width"/> × <paramref name="height"/> points.
    /// </summary>
    public void Render(ID3D11Texture2D target, uint pixelWidth, uint pixelHeight, double width, double height, Color4 clear)
    {
        if (pixelWidth == 0 || pixelHeight == 0) return;
        EnsureMsaa(pixelWidth, pixelHeight);
        PrepareGeometry(width, height);
        var ctx = _context;
        ctx.OMSetRenderTargets(_msaaView!);
        ctx.ClearRenderTargetView(_msaaView!, clear);
        if (_frame is { } frame && (_segmentCount > 0 || _fillCount > 0))
        {
            var u = new Uniforms
            {
                ViewProjection = ViewportMath.ViewProjection(Camera, Mode, frame, width, height, _depthRadius),
                ViewportSize = new Vector2(pixelWidth, pixelHeight),
                Feather = 0.75f,
            };
            ctx.UpdateSubresource(in u, _uniforms);
            ctx.RSSetViewport(new Viewport(pixelWidth, pixelHeight));
            ctx.RSSetState(_raster);
            ctx.OMSetBlendState(_blend);
            ctx.VSSetConstantBuffer(0, _uniforms);
            ctx.IASetInputLayout(null);
            if (_fillCount > 0)
            {
                ctx.IASetPrimitiveTopology(PrimitiveTopology.TriangleList);
                ctx.VSSetShader(_fillVS);
                ctx.PSSetShader(_fillPS);
                ctx.VSSetShaderResource(1, _fillView!);
                ctx.Draw((uint)_fillCount, 0);
            }
            if (_segmentCount > 0)
            {
                ctx.IASetPrimitiveTopology(PrimitiveTopology.TriangleStrip);
                ctx.VSSetShader(_lineVS);
                ctx.PSSetShader(_linePS);
                ctx.VSSetShaderResource(0, _segmentView!);
                ctx.DrawInstanced(4, (uint)_segmentCount, 0, 0);
            }
        }
        if (SampleCount > 1) ctx.ResolveSubresource(target, 0, _msaa!, 0, Format.B8G8R8A8_UNorm);
        else ctx.CopyResource(target, _msaa!);
        ctx.OMSetRenderTargets((ID3D11RenderTargetView)null!);
    }

    /// <summary>Renders offscreen and reads the pixels back (BGRA, premultiplied, top row first): for captures.</summary>
    public byte[] RenderToPixels(uint pixelWidth, uint pixelHeight, double width, double height, Color4 clear)
    {
        using var target = Device.CreateTexture2D(new Texture2DDescription(Format.B8G8R8A8_UNorm, pixelWidth, pixelHeight, 1, 1,
            BindFlags.RenderTarget, ResourceUsage.Default));
        using var staging = Device.CreateTexture2D(new Texture2DDescription(Format.B8G8R8A8_UNorm, pixelWidth, pixelHeight, 1, 1,
            BindFlags.None, ResourceUsage.Staging, CpuAccessFlags.Read));
        Render(target, pixelWidth, pixelHeight, width, height, clear);
        _context.CopyResource(staging, target);
        var map = _context.Map(staging, 0, MapMode.Read);
        var pixels = new byte[pixelWidth * pixelHeight * 4];
        try
        {
            for (uint y = 0; y < pixelHeight; y++)
                Marshal.Copy(map.DataPointer + (nint)(y * map.RowPitch), pixels, (int)(y * pixelWidth * 4), (int)pixelWidth * 4);
        }
        finally { _context.Unmap(staging, 0); }
        return pixels;
    }

    public void Dispose()
    {
        ReleaseGeometry();
        _msaaView?.Dispose();
        _msaa?.Dispose();
        _uniforms.Dispose();
        _raster.Dispose();
        _blend.Dispose();
        _lineVS.Dispose(); _linePS.Dispose(); _fillVS.Dispose(); _fillPS.Dispose();
        _context.ClearState();
        _context.Flush();
        _context.Dispose();
        Device.Dispose();
    }

    /// <summary>Matches <c>Uniforms</c> in the shader (80 bytes).</summary>
    [StructLayout(LayoutKind.Sequential)]
    struct Uniforms
    {
        public Matrix4x4 ViewProjection;
        public Vector2 ViewportSize;
        public float Feather;
        public float Pad;
    }
}
