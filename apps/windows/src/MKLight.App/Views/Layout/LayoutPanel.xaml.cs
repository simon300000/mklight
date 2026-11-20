using System.Globalization;
using System.Numerics;
using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using MKLight.App.Support;

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

    /// <summary>The layout of a design window: follows the session's layout scene, selection and preview.</summary>
    public LayoutPanel(ViewModels.DesignSession session) : this()
    {
        var layout = session.Workspace.Layout;
        void Update()
        {
            SetScene(session.Section, layout.Phase == ViewModels.PanePhase.Failed ? layout.Message : session.Workspace.ModelErrors.FirstOrDefault());
            SetStale(layout.IsStale, layout.IsRunning);
            SetSelectedSurface(session.SelectedSurface);
            var previewing = session.Workspace.PreviewTitle;
            PreviewChip.Visibility = previewing is null ? Visibility.Collapsed : Visibility.Visible;
            PreviewText.Text = previewing is null ? "" : $"Previewing {previewing} · not applied";
        }
        SurfaceClicked += i => session.SelectedSurface = i;
        session.SelectionChanged += () => SetSelectedSurface(session.SelectedSurface);
        session.LayoutSceneChanged += Update;
        layout.Changed += Update;
        session.Workspace.PreviewChanged += Update;
        Update();
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
        (_scale, _center) = SectionRenderer.Fit(_scene, (float)ActualWidth, (float)ActualHeight, 40, 56, 44);
        _fitted = true;
        Canvas.Invalidate();
    }

    Vector2 ToScreen(Vector2 p) => new((p.X - _center.X) * _scale + (float)ActualWidth / 2, -(p.Y - _center.Y) * _scale + (float)ActualHeight / 2);
    Vector2 ToScene(Vector2 s) => new((s.X - (float)ActualWidth / 2) / _scale + _center.X, -(s.Y - (float)ActualHeight / 2) / _scale + _center.Y);

    void OnDraw(CanvasControl sender, CanvasDrawEventArgs args)
    {
        UpdateScaleBar();
        if (_scene is null) return;
        SectionRenderer.Draw(args.DrawingSession, _scene, ToScreen,
            new SectionRenderer.Style(ActualTheme == ElementTheme.Dark, _coloring, _selected, _stale));
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
