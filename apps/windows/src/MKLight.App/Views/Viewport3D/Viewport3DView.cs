using System.Numerics;
using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.Engine;
using Vortice.Direct3D11;
using Vortice.DXGI;
using Vortice.Mathematics;

namespace MKLight.App.Views.Viewport3D;

/// <summary>
/// The layout's 3-D view: a <see cref="SwapChainPanel"/> with a composition swap chain that a
/// <see cref="ViewportRenderer"/> draws into on demand (one frame per change, at the panel's
/// composition scale). Recreates the device when it is lost and frees everything when unloaded.
/// Input is the host's: it sets <see cref="Camera"/> and calls <see cref="Invalidate"/>.
/// </summary>
public sealed partial class Viewport3DView : UserControl
{
    readonly SwapChainPanel _panel = new();
    ViewportRenderer? _renderer;
    IDXGISwapChain1? _swapChain;
    (uint W, uint H) _bufferSize;
    bool _queued;
    LayoutScene? _scene;
    ViewportStyle? _style;

    public Viewport3DView()
    {
        Content = _panel;
        IsTabStop = false;
        Loaded += (_, _) => Invalidate();
        Unloaded += (_, _) => Release();
        _panel.SizeChanged += (_, _) => Invalidate();
        _panel.CompositionScaleChanged += (_, _) => Invalidate();
    }

    /// <summary>Why the view cannot draw (no Direct3D 11 device, a shader that does not compile), or null.</summary>
    public string? Error { get; private set; }
    public event Action? ErrorChanged;

    ViewportCamera _camera = new();
    public ViewportCamera Camera
    {
        get => _camera;
        set { if (_camera == value) return; _camera = value; Invalidate(); }
    }

    public ViewportMode Mode { get; set; } = ViewportMode.Orbit3D;

    /// <summary>The background (the content surface colour of the window's theme).</summary>
    public Color4 ClearColor { get; set; } = new(1, 1, 1, 1);

    /// <summary>Device pixels per point of the panel.</summary>
    public float PixelScale => Math.Max(_panel.CompositionScaleX, 0.1f);

    public void SetScene(LayoutScene? scene, ViewportStyle style)
    {
        _scene = scene;
        _style = style;
        _renderer?.SetScene(scene, style);
        Invalidate();
    }

    /// <summary>Draws a new frame before the next composition pass.</summary>
    public void Invalidate()
    {
        if (_queued) return;
        _queued = true;
        CompositionTarget.Rendering += OnRendering;
    }

    void OnRendering(object? sender, object e)
    {
        CompositionTarget.Rendering -= OnRendering;
        _queued = false;
        if (!IsLoaded || Visibility != Visibility.Visible) return;
        try { Draw(); }
        catch (SharpGen.Runtime.SharpGenException ex) when (IsDeviceLost(ex.ResultCode.Code))
        {
            Release();
            Invalidate();
        }
        catch (Exception ex)
        {
            Release();
            SetError("The 3-D view could not start: " + ex.Message);
        }
    }

    static bool IsDeviceLost(int hr) =>
        hr == Vortice.DXGI.ResultCode.DeviceRemoved.Code || hr == Vortice.DXGI.ResultCode.DeviceReset.Code
        || hr == Vortice.DXGI.ResultCode.DeviceHung.Code;

    void SetError(string? message)
    {
        if (Error == message) return;
        Error = message;
        ErrorChanged?.Invoke();
    }

    void Draw()
    {
        double w = _panel.ActualWidth, h = _panel.ActualHeight;
        float sx = PixelScale, sy = Math.Max(_panel.CompositionScaleY, 0.1f);
        uint pw = (uint)Math.Max(1, Math.Round(w * sx)), ph = (uint)Math.Max(1, Math.Round(h * sy));
        if (w < 1 || h < 1) return;
        if (_renderer is null)
        {
            _renderer = new ViewportRenderer();
            if (_scene is not null && _style is not null) _renderer.SetScene(_scene, _style);
            SetError(null);
        }
        if (_swapChain is null)
        {
            using var dxgiDevice = _renderer.Device.QueryInterface<IDXGIDevice>();
            using var adapter = dxgiDevice.GetAdapter();
            using var factory = adapter.GetParent<IDXGIFactory2>();
            _swapChain = factory.CreateSwapChainForComposition(_renderer.Device, new SwapChainDescription1
            {
                Width = pw, Height = ph, Format = Format.B8G8R8A8_UNorm, SampleDescription = new SampleDescription(1, 0),
                BufferUsage = Usage.RenderTargetOutput, BufferCount = 2, SwapEffect = SwapEffect.FlipSequential,
                AlphaMode = AlphaMode.Ignore, Scaling = Scaling.Stretch,
            });
            _bufferSize = (pw, ph);
            SetPanelSwapChain(_panel, _swapChain.NativePointer);
        }
        else if (_bufferSize != (pw, ph))
        {
            _swapChain.ResizeBuffers(2, pw, ph, Format.B8G8R8A8_UNorm, SwapChainFlags.None).CheckError();
            _bufferSize = (pw, ph);
        }
        // The buffers are in device pixels; the panel maps them back to its size in points.
        using (var chain2 = _swapChain.QueryInterfaceOrNull<IDXGISwapChain2>())
        {
            if (chain2 is not null) chain2.MatrixTransform = Matrix3x2.CreateScale(1 / sx, 1 / sy);
        }
        _renderer.Camera = _camera;
        _renderer.Mode = Mode;
        using (var back = _swapChain.GetBuffer<ID3D11Texture2D>(0))
            _renderer.Render(back, pw, ph, w, h, ClearColor);
        _swapChain.Present(1, PresentFlags.None).CheckError();
    }

    /// <summary>
    /// Renders the current scene offscreen at <paramref name="scale"/> pixels per point (BGRA,
    /// top row first), for window captures that cannot see a swap chain.
    /// </summary>
    public (byte[] Pixels, int Width, int Height)? RenderOffscreen(double scale)
    {
        double w = _panel.ActualWidth, h = _panel.ActualHeight;
        if (w < 1 || h < 1 || _scene is null || _style is null) return null;
        uint pw = (uint)Math.Round(w * scale), ph = (uint)Math.Round(h * scale);
        using var renderer = new ViewportRenderer { Camera = _camera, Mode = Mode };
        renderer.SetScene(_scene, _style with { Scale = (float)scale });
        return (renderer.RenderToPixels(pw, ph, w, h, ClearColor), (int)pw, (int)ph);
    }

    /// <summary>Frees the swap chain and the device (unloaded, device lost).</summary>
    void Release()
    {
        if (_queued) { CompositionTarget.Rendering -= OnRendering; _queued = false; }
        if (_swapChain is not null)
        {
            try { SetPanelSwapChain(_panel, IntPtr.Zero); } catch (COMException) { }
            _swapChain.Dispose();
            _swapChain = null;
        }
        _renderer?.Dispose();
        _renderer = null;
    }

    /// <summary>ISwapChainPanelNative::SetSwapChain on the panel's WinRT object.</summary>
    static unsafe void SetPanelSwapChain(SwapChainPanel panel, IntPtr swapChain)
    {
        var unknown = ((WinRT.IWinRTObject)panel).NativeObject.ThisPtr;
        var iid = new Guid("63aad0b8-7c24-40ff-85a8-640d944cc325");
        Marshal.ThrowExceptionForHR(Marshal.QueryInterface(unknown, in iid, out var native));
        try
        {
            var vtable = *(IntPtr**)native;
            var setSwapChain = (delegate* unmanaged[Stdcall]<IntPtr, IntPtr, int>)vtable[3];
            Marshal.ThrowExceptionForHR(setSwapChain(native, swapChain));
        }
        finally { Marshal.Release(native); }
    }
}
