using System.Globalization;
using System.Numerics;
using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using MKLight.App.Support;
using Windows.UI;

namespace MKLight.App.Views.Layout;

public enum RayColoring { Field, Wavelength }

/// <summary>
/// The layout's 2-D section view (Win2D): glass fills, surface profiles,
/// stop, image plane, axis and rays coloured by field or wavelength, with a
/// scale bar, fit, pan (drag) and zoom (wheel, about the pointer). The
/// selected surface is drawn in the accent. Geometry is the engine's.
/// </summary>
public sealed partial class LayoutPanel : UserControl
{
    SectionScene? _scene;
    bool _stale;
    int? _selected;
    RayColoring _coloring = RayColoring.Field;

    // View transform: screen = (p - center) * scale (y up) + viewport centre.
    Vector2 _center;
    float _scale = 1; // pixels per mm
    bool _fitted;
    Vector2? _dragFrom;

    public LayoutPanel()
    {
        InitializeComponent();
        ActualThemeChanged += (_, _) => Canvas.Invalidate();
        SizeChanged += (_, e) => { if (!_fitted || e.PreviousSize.Width == 0) Fit(); else Canvas.Invalidate(); };
        Unloaded += (_, _) => Canvas.RemoveFromVisualTree();
    }

    /// <summary>Raised when the user clicks a surface in the section.</summary>
    public event Action<int>? SurfaceClicked;

    public RayColoring Coloring
    {
        get => _coloring;
        set { _coloring = value; UpdateLegend(); Canvas.Invalidate(); }
    }

    public void SetScene(SectionScene? scene, string? message = null)
    {
        var first = _scene is null;
        _scene = scene;
        Message.Text = scene is null ? message ?? "" : "";
        SectionLabel.Text = scene?.Section == "xz" ? "X–Z" : "Section";
        if (first || !_fitted) Fit();
        UpdateLegend();
        Canvas.Invalidate();
    }

    public void SetStale(bool stale, bool updating)
    {
        _stale = stale;
        StaleCard.Visibility = stale ? Visibility.Visible : Visibility.Collapsed;
        StaleRing.IsActive = updating;
        StaleText.Text = updating ? "Updating" : "Design changed";
        Canvas.Invalidate();
    }

    public void SetSelectedSurface(int? index) { _selected = index; Canvas.Invalidate(); }

    void UpdateLegend()
    {
        if (_scene is null) { ColoringLegend.Text = ""; return; }
        ColoringLegend.Text = _coloring == RayColoring.Field
            ? $"Rays by field · {_scene.FieldCount}"
            : "Rays by wavelength · " + string.Join("  ", _scene.WavelengthsUm.Select(w => w.ToString("0.####", CultureInfo.InvariantCulture)));
    }

    public void Fit()
    {
        if (_scene is null || ActualWidth <= 0 || ActualHeight <= 0) return;
        // Frame everything; a long, slim scene (an object far away) frames its lens.
        var (min, max) = _scene.Bounds();
        var (lmin, lmax) = _scene.Bounds(lensOnly: true);
        var span = max - min;
        if (span.X > 12 * Math.Max(span.Y, 1e-6) && (lmax.X - lmin.X) * 5 < span.X) { min = new Vector2(lmin.X - (lmax.X - lmin.X) * 0.5f, min.Y); max = new Vector2(max.X, max.Y); }
        min.Y = Math.Min(min.Y, -max.Y); max.Y = Math.Max(max.Y, -min.Y);
        var size = max - min;
        const float marginX = 40, marginTop = 56, marginBottom = 44;
        var sx = (float)(ActualWidth - 2 * marginX) / Math.Max(size.X, 1e-6f);
        var sy = (float)(ActualHeight - marginTop - marginBottom) / Math.Max(size.Y, 1e-6f);
        _scale = Math.Max(1e-4f, Math.Min(sx, sy));
        _center = (min + max) / 2;
        // Centre between the floating controls and the scale bar.
        _center.Y += (marginTop - marginBottom) / 2 / _scale;
        _fitted = true;
        Canvas.Invalidate();
    }

    Vector2 ToScreen(Vector2 p) => new((p.X - _center.X) * _scale + (float)ActualWidth / 2, -(p.Y - _center.Y) * _scale + (float)ActualHeight / 2);
    Vector2 ToScene(Vector2 s) => new((s.X - (float)ActualWidth / 2) / _scale + _center.X, -(s.Y - (float)ActualHeight / 2) / _scale + _center.Y);

    static readonly CanvasStrokeStyle Round = new() { LineJoin = CanvasLineJoin.Round, StartCap = CanvasCapStyle.Round, EndCap = CanvasCapStyle.Round };
    static readonly CanvasStrokeStyle AxisDash = new() { DashStyle = CanvasDashStyle.Dash, CustomDashStyle = [8, 5] };
    static readonly CanvasStrokeStyle DoeDash = new() { CustomDashStyle = [2, 2] };

    void OnDraw(CanvasControl sender, CanvasDrawEventArgs args)
    {
        var ds = args.DrawingSession;
        UpdateScaleBar();
        if (_scene is null) return;
        var dark = ActualTheme == ElementTheme.Dark;
        Color ink = dark ? Color.FromArgb(255, 224, 227, 232) : Color.FromArgb(255, 33, 36, 41);
        Color faint = dark ? Color.FromArgb(204, 140, 143, 153) : Color.FromArgb(204, 140, 143, 148);
        var accent = (Color)Application.Current.Resources["SystemAccentColor"];
        if (dark) accent = (Color)Application.Current.Resources["SystemAccentColorLight2"];
        var glass = dark ? Color.FromArgb(56, 89, 153, 242) : Color.FromArgb(64, 115, 173, 242);
        var doeInk = Palette.Slot(6, dark).WithAlpha(0.85);
        var dim = _stale ? 0.35 : 1.0;

        // Glass fills between each element's front and back profiles.
        foreach (var e in _scene.Elements)
        {
            if (e.Front.Length < 2 || e.Front.Length != e.Back.Length) continue;
            using var pb = new CanvasPathBuilder(ds);
            pb.BeginFigure(ToScreen(e.Front[0]));
            for (var k = 1; k < e.Front.Length; k++) pb.AddLine(ToScreen(e.Front[k]));
            for (var k = e.Back.Length - 1; k >= 0; k--) pb.AddLine(ToScreen(e.Back[k]));
            pb.EndFigure(CanvasFigureLoop.Closed);
            using var g = CanvasGeometry.CreatePath(pb);
            ds.FillGeometry(g, glass.WithAlpha(glass.A / 255.0 * (_stale ? 0.6 : 1)));
        }

        // Rays first, then outlines on top (the selected surface last).
        foreach (var l in _scene.Lines.Where(l => l.Kind == SectionKind.Ray))
        {
            var c = _coloring == RayColoring.Field
                ? Palette.Field(l.Field ?? 0, dark)
                : Palette.Wavelength(l.Wavelength is int w && w < _scene.WavelengthsUm.Count ? _scene.WavelengthsUm[w] : 0.55, dark);
            DrawLine(ds, l.Points, c.WithAlpha((l.Vignetted ? 0.35 : 0.9) * dim), 1.1f, Round);
        }
        bool IsSelected(SectionLine l) => l.Surface is not null && l.Surface == _selected && l.Kind != SectionKind.ElementEdge;
        var outlines = _scene.Lines.Where(l => l.Kind != SectionKind.Ray).OrderBy(l => IsSelected(l) ? 1 : 0);
        foreach (var l in outlines)
        {
            var sel = IsSelected(l);
            switch (l.Kind)
            {
                case SectionKind.Axis: DrawLine(ds, l.Points, faint, 0.8f, AxisDash); break;
                case SectionKind.ElementEdge: DrawLine(ds, l.Points, ink, 1.4f, Round); break;
                case SectionKind.SurfaceProfile:
                    DrawLine(ds, l.Points, sel ? accent : ink, sel ? 3.2f : 1.4f, Round);
                    if (l.Doe) DrawLine(ds, l.Points, doeInk, sel ? 4.2f : 3.4f, DoeDash);
                    break;
                case SectionKind.SurfaceRim: DrawLine(ds, l.Points, sel ? accent : faint, sel ? 1.8f : 1f, Round); break;
                case SectionKind.Stop: DrawLine(ds, l.Points, sel ? accent : ink, sel ? 2.8f : 2.2f, Round); break;
                case SectionKind.ImagePlane: DrawLine(ds, l.Points, sel ? accent : ink, sel ? 2.4f : 1.8f, Round); break;
            }
        }
    }

    void DrawLine(CanvasDrawingSession ds, Vector2[] pts, Color color, float width, CanvasStrokeStyle style)
    {
        if (pts.Length < 2) return;
        using var pb = new CanvasPathBuilder(ds);
        var open = false;
        foreach (var p in pts)
        {
            if (!float.IsFinite(p.X) || !float.IsFinite(p.Y)) { if (open) pb.EndFigure(CanvasFigureLoop.Open); open = false; continue; }
            var s = ToScreen(p);
            if (!open) { pb.BeginFigure(s); open = true; } else pb.AddLine(s);
        }
        if (open) pb.EndFigure(CanvasFigureLoop.Open);
        using var g = CanvasGeometry.CreatePath(pb);
        ds.DrawGeometry(g, color, width, style);
    }

    /// <summary>A round length (1, 2, 5 × 10ⁿ mm) about 70–150 px long.</summary>
    void UpdateScaleBar()
    {
        if (_scene is null || _scale <= 0) { ScaleBar.Visibility = Visibility.Collapsed; return; }
        ScaleBar.Visibility = Visibility.Visible;
        var target = 90 / _scale;
        var mag = Math.Pow(10, Math.Floor(Math.Log10(target)));
        var n = target / mag;
        var len = (n < 1.5 ? 1 : n < 3.5 ? 2 : n < 7.5 ? 5 : 10) * mag;
        ScaleLine.Width = len * _scale;
        ScaleText.Text = (len >= 1 ? len.ToString("0", CultureInfo.InvariantCulture) : len.ToString("0.###", CultureInfo.InvariantCulture)) + " mm";
    }

    void OnColoring(object sender, RoutedEventArgs e) => Coloring = _coloring == RayColoring.Field ? RayColoring.Wavelength : RayColoring.Field;
    void OnFit(object sender, RoutedEventArgs e) => Fit();
    void OnDoubleTapped(object sender, DoubleTappedRoutedEventArgs e) => Fit();

    void OnPointerPressed(object sender, PointerRoutedEventArgs e)
    {
        var p = e.GetCurrentPoint(Canvas);
        if (!p.Properties.IsLeftButtonPressed && !p.Properties.IsMiddleButtonPressed) return;
        _dragFrom = p.Position.ToVector2();
        _pressAt = _dragFrom;
        Canvas.CapturePointer(e.Pointer);
        ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.SizeAll);
    }

    Vector2? _pressAt;

    void OnPointerMoved(object sender, PointerRoutedEventArgs e)
    {
        if (_dragFrom is not Vector2 from) return;
        var at = e.GetCurrentPoint(Canvas).Position.ToVector2();
        var d = at - from;
        _center -= new Vector2(d.X, -d.Y) / _scale;
        _dragFrom = at;
        Canvas.Invalidate();
    }

    void OnPointerReleased(object sender, PointerRoutedEventArgs e)
    {
        var at = e.GetCurrentPoint(Canvas).Position.ToVector2();
        if (_pressAt is Vector2 p0 && Vector2.Distance(p0, at) < 3) HitTest(at);
        _dragFrom = null;
        _pressAt = null;
        Canvas.ReleasePointerCapture(e.Pointer);
        ProtectedCursor = null;
    }

    void OnPointerWheel(object sender, PointerRoutedEventArgs e)
    {
        var pt = e.GetCurrentPoint(Canvas);
        var at = pt.Position.ToVector2();
        var before = ToScene(at);
        var factor = MathF.Pow(1.0015f, pt.Properties.MouseWheelDelta);
        _scale = Math.Clamp(_scale * factor, 1e-4f, 1e5f);
        var after = ToScene(at);
        _center += before - after;
        Canvas.Invalidate();
        e.Handled = true;
    }

    /// <summary>A click within 5 px of a surface profile selects that surface.</summary>
    void HitTest(Vector2 at)
    {
        if (_scene is null) return;
        int? best = null;
        var bestD = 5f;
        foreach (var l in _scene.Lines.Where(l => l.Surface is not null && l.Kind is SectionKind.SurfaceProfile or SectionKind.Stop or SectionKind.ImagePlane))
        {
            for (var k = 1; k < l.Points.Length; k++)
            {
                var d = DistanceToSegment(at, ToScreen(l.Points[k - 1]), ToScreen(l.Points[k]));
                if (d < bestD) { bestD = d; best = l.Surface; }
            }
        }
        if (best is int s) SurfaceClicked?.Invoke(s);
    }

    static float DistanceToSegment(Vector2 p, Vector2 a, Vector2 b)
    {
        var ab = b - a;
        var t = ab.LengthSquared() == 0 ? 0 : Math.Clamp(Vector2.Dot(p - a, ab) / ab.LengthSquared(), 0, 1);
        return Vector2.Distance(p, a + t * ab);
    }
}
