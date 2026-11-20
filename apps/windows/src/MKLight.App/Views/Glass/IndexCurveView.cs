using System.Numerics;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.Engine;

namespace MKLight.App.Views.Glass;

/// <summary>A glass's refractive index against wavelength (Win2D), with its standard lines as dots in their colours.</summary>
public sealed class IndexCurveView : UserControl
{
    readonly CanvasControl _canvas = new();
    GlassDetails? _details;

    public IndexCurveView()
    {
        Height = 150;
        Content = _canvas;
        _canvas.Draw += OnDraw;
        ActualThemeChanged += (_, _) => _canvas.Invalidate();
        Unloaded += (_, _) => _canvas.RemoveFromVisualTree();
    }

    public GlassDetails? Details
    {
        get => _details;
        set { _details = value; _canvas.Invalidate(); }
    }

    void OnDraw(CanvasControl sender, CanvasDrawEventArgs args)
    {
        if (_details is not { IndexCurve.Count: > 1 } d) return;
        var style = PlotStyle.For(ActualTheme);
        var plot = new LinePlot { XTitle = "λ (µm)", ZeroLine = false, YTickSpacing = 32 };
        plot.Series.Add(new PlotSeries(d.IndexCurve.Select(p => new Vector2((float)p.Um, (float)p.N)).ToList(), Palette.Slot(0, style.Dark), Width: 1.6f));
        plot.FitData(pad: 0.08);
        var rect = plot.Draw(args.DrawingSession, new global::Windows.Foundation.Rect(0, 0, sender.ActualWidth, sender.ActualHeight), style);
        if (rect.IsEmpty) return;
        foreach (var line in d.Lines)
        {
            if (line.Um < plot.XMin || line.Um > plot.XMax) continue;
            var x = rect.X + (line.Um - plot.XMin) / (plot.XMax - plot.XMin) * rect.Width;
            var y = rect.Bottom - (line.N - plot.YMin) / (plot.YMax - plot.YMin) * rect.Height;
            args.DrawingSession.FillCircle((float)x, (float)y, 2.8f, Palette.Wavelength(line.Um, style.Dark));
            args.DrawingSession.DrawCircle((float)x, (float)y, 2.8f, style.Surface, 1);
        }
    }
}
