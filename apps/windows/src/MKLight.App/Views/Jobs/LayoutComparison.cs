using Microsoft.Graphics.Canvas;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.App.Views.Analysis.Results;
using MKLight.App.Views.Layout;
using MKLight.Engine;
using MKLight.Model;
using Windows.Foundation;
using Fit = MKLight.App.ViewModels.Jobs.LayoutThumbnailFit;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The design's and a candidate's layouts side by side (the Mac app's <c>LayoutComparison</c>): the engine's layout
/// of each design, small and plain, on one scale so shape changes read. Each is computed once per design
/// (<see cref="DesignFigures"/>); a comparison that leaves the page drops its unfinished requests.
/// </summary>
public sealed partial class LayoutComparison : UserControl
{
    readonly Workspace _ws;
    readonly (OpticalSystem? System, string Title)[] _sides;
    readonly SectionScene?[] _scenes = new SectionScene?[2];
    readonly PlotCanvas?[] _canvases = new PlotCanvas?[2];
    readonly ProgressRing?[] _rings = new ProgressRing?[2];
    readonly List<DesignFigures.Lease> _leases = [];

    public LayoutComparison(Workspace ws, OpticalSystem? before, OpticalSystem? after, string beforeTitle = "Before", string afterTitle = "Candidate")
    {
        _ws = ws;
        _sides = [(before, beforeTitle), (after, afterTitle)];
        IsTabStop = false;
        var grid = new Grid { ColumnSpacing = 8 };
        for (var i = 0; i < 2; i++)
        {
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            var figure = Figure(i);
            Grid.SetColumn(figure, i);
            grid.Children.Add(figure);
        }
        Content = grid;
        Request();
        Loaded += (_, _) => Request();
        Unloaded += (_, _) => Release();
    }

    /// <summary>The thumbnail's height (the Mac's 96 pt less its caption).</summary>
    public const double ThumbnailHeight = 78;

    FrameworkElement Figure(int i)
    {
        var (_, title) = _sides[i];
        var canvas = new PlotCanvas((ds, size, style) => Draw(ds, size, style, i));
        _canvases[i] = canvas;
        var ring = new ProgressRing { Width = 16, Height = 16, IsActive = true, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center };
        _rings[i] = ring;
        var cell = new Grid();
        cell.Children.Add(canvas);
        cell.Children.Add(ring);
        var card = new Border { Child = cell, Height = ThumbnailHeight, CornerRadius = new CornerRadius(8), BorderThickness = new Thickness(1) };
        JobUI.Bind(card, "ContentSurfaceBrush", b => card.Background = b);
        JobUI.Bind(card, "CardStrokeColorDefaultBrush", b => card.BorderBrush = b);
        var column = JobUI.Column(3, JobUI.Text(title, "CaptionText"), card);
        AutomationProperties.SetName(column, $"{title} layout");
        return column;
    }

    void Request()
    {
        if (_leases.Count > 0) return;
        for (var i = 0; i < 2; i++)
        {
            var k = i;
            if (_sides[k].System is not { } system)
            {
                Show(k, null);
                continue;
            }
            _leases.Add(DesignFigures.Of(_ws).Request(system, "layout", ReadSection, scene => Show(k, scene)));
        }
    }

    void Release()
    {
        foreach (var l in _leases) l.Dispose();
        _leases.Clear();
    }

    static SectionScene? ReadSection(AnalysisOutput output, OpticalSystem _) =>
        LayoutScene.FromOutput(output) is { } scene ? EngineLayout.Section(scene) : null;

    void Show(int i, SectionScene? scene)
    {
        _scenes[i] = scene;
        if (_rings[i] is { } r) { r.IsActive = false; r.Visibility = Visibility.Collapsed; }
        foreach (var c in _canvases) c?.Invalidate(); // the shared scale may have changed
    }

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style, int i)
    {
        if (_scenes[i] is not { } scene) return;
        float w = (float)size.Width, h = (float)size.Height;
        // One scale for both, so shape changes read (while one side computes, the other fits alone).
        if (Fit.Union(_scenes.OfType<SectionScene>().Select(FrameOf)) is not { } frame) return;
        var fit = Fit.Fit(frame, w, h, 8, 6);
        SectionRenderer.Draw(ds, scene, p => Fit.ToScreen(p, fit, w, h), new SectionRenderer.Style(style.Dark, LineScale: 0.7f));
    }

    static Fit.Box FrameOf(SectionScene s)
    {
        var (min, max) = s.Bounds();
        var (lmin, lmax) = s.Bounds(lensOnly: true);
        return Fit.Frame(new Fit.Box(min, max), new Fit.Box(lmin, lmax));
    }
}
