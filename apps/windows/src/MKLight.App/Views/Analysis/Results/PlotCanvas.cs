using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using Windows.Foundation;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// A Win2D canvas for a result plot: redraws on size and theme change,
/// frees its device resources when unloaded (Win2D's RemoveFromVisualTree)
/// and makes a new canvas if the view is loaded again (a pane moved).
/// </summary>
public sealed partial class PlotCanvas : UserControl
{
    readonly Action<CanvasDrawingSession, Size, PlotStyle> _draw;
    CanvasControl? _canvas;

    public PlotCanvas(Action<CanvasDrawingSession, Size, PlotStyle> draw)
    {
        _draw = draw;
        IsTabStop = false;
        Attach();
        Loaded += (_, _) => { if (_canvas is null) Attach(); };
        Unloaded += (_, _) =>
        {
            _canvas?.RemoveFromVisualTree();
            _canvas = null;
            Content = null;
        };
        ActualThemeChanged += (_, _) => Invalidate();
        SizeChanged += (_, _) => Invalidate();
    }

    /// <summary>Raised when the canvas' device is (re)created, so cached bitmaps are rebuilt.</summary>
    public event Action<CanvasDevice>? DeviceCreated;

    void Attach()
    {
        var c = new CanvasControl { ClearColor = Microsoft.UI.Colors.Transparent };
        c.CreateResources += (s, _) => DeviceCreated?.Invoke(s.Device);
        c.Draw += (s, e) =>
        {
            var size = new Size(s.ActualWidth, s.ActualHeight);
            if (size.Width <= 0 || size.Height <= 0) return;
            _draw(e.DrawingSession, size, PlotStyle.For(ActualTheme));
        };
        _canvas = c;
        Content = c;
    }

    public void Invalidate() => _canvas?.Invalidate();
}
