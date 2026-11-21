using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Settings;
using Windows.Foundation;
using Windows.System;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// A small diagram of the normalised field: the field disk, the design's field points and the
/// simulated patch. A click (or a drag) moves the patch centre; it snaps to a field point nearby
/// and to the axis. The arrow keys move it by 0.05 of the field.
/// </summary>
sealed partial class ImageSimulationFieldPicker : UserControl
{
    const double Side = 200;
    /// <summary>Normalised field units shown on each side of the axis.</summary>
    const double Span = 1.15;

    readonly double _extent;
    readonly IReadOnlyList<ImageSimulationSettings.FieldPoint> _points;
    readonly Action<(double X, double Y)> _set;
    readonly PlotCanvas _canvas;
    readonly TextBlock _title, _warning;
    readonly Border _diagram;
    (double X, double Y) _center;
    (double X, double Y)? _dragged;

    public ImageSimulationFieldPicker((double X, double Y) center, double extent, IReadOnlyList<ImageSimulationSettings.FieldPoint> points, Action<(double X, double Y)> set)
    {
        _center = center;
        _extent = extent;
        _points = points;
        _set = set;
        _canvas = new PlotCanvas(Draw) { Width = Side, Height = Side };
        _diagram = new Border { Child = _canvas, IsTabStop = true, UseSystemFocusVisuals = true, CornerRadius = new CornerRadius(4) };
        _title = Kit.Text("", "NumberText", 12);
        _warning = Kit.Caption(Loc.S("Results_ImageSim_PickerOutside"), wrap: true);
        var hint = Kit.Caption(Loc.S("Results_ImageSim_PickerHint"), wrap: true);
        var panel = new StackPanel { Spacing = 8, Width = Side };
        panel.Children.Add(Kit.Text(Loc.S("Results_ImageSim_PickerTitle"), "PanelTitleText", 13));
        panel.Children.Add(_diagram);
        panel.Children.Add(_title);
        panel.Children.Add(_warning);
        panel.Children.Add(hint);
        Content = panel;

        _canvas.PointerPressed += (_, e) =>
        {
            _canvas.CapturePointer(e.Pointer);
            _diagram.Focus(FocusState.Pointer);
            Drag(e.GetCurrentPoint(_canvas).Position);
            e.Handled = true;
        };
        _canvas.PointerMoved += (_, e) => { if (_dragged is not null) Drag(e.GetCurrentPoint(_canvas).Position); };
        _canvas.PointerReleased += (_, e) =>
        {
            _canvas.ReleasePointerCapture(e.Pointer);
            if (_dragged is { } c) Commit(c);
            e.Handled = true;
        };
        _canvas.PointerCaptureLost += (_, _) => { if (_dragged is { } c) Commit(c); };
        _diagram.KeyDown += (_, e) =>
        {
            const double step = 0.05;
            (double dx, double dy) = e.Key switch
            {
                VirtualKey.Left => (-step, 0.0), VirtualKey.Right => (step, 0.0),
                VirtualKey.Up => (0.0, step), VirtualKey.Down => (0.0, -step), _ => (0.0, 0.0),
            };
            if (e.Key == VirtualKey.Home) { Commit((0, 0)); e.Handled = true; return; }
            if (dx == 0 && dy == 0) return;
            Commit((Math.Round((_center.X + dx) * 100) / 100, Math.Round((_center.Y + dy) * 100) / 100));
            e.Handled = true;
        };
        AutomationProperties.SetName(_diagram, Loc.S("Results_ImageSim_PickerDiagram"));
        Update();
    }

    void Drag(Point p)
    {
        _dragged = Field(p);
        Update();
    }

    void Commit((double X, double Y) c)
    {
        _dragged = null;
        _center = c;
        Update();
        _set(c);
    }

    (double X, double Y) Field(Point p)
    {
        var k = Side / 2 / Span;
        var f = (Math.Clamp((p.X - Side / 2) / k, -Span, Span), Math.Clamp((Side / 2 - p.Y) / k, -Span, Span));
        return ImageSimulationSettings.Snap(f, _points, 0.06);
    }

    void Update()
    {
        var c = _dragged ?? _center;
        var title = ImageSimulationSettings.CenterTitle(c, _points);
        _title.Text = char.ToUpperInvariant(title[0]) + title[1..];
        _warning.Foreground = ThemeLookup.Brush(this, "WarningBrush");
        _warning.Visibility = ImageSimulationSettings.OutsideField(c, _extent) ? Visibility.Visible : Visibility.Collapsed;
        AutomationProperties.SetHelpText(_diagram, Loc.F("Results_ImageSim_PickerHelp", title, ImageSimulationSettings.ExtentTitle(_extent), _points.Count - 1));
        _canvas.Invalidate();
    }

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var c = _dragged ?? _center;
        var k = size.Width / 2 / Span;
        (float X, float Y) Pt(double fx, double fy) => ((float)(size.Width / 2 + fx * k), (float)(size.Height / 2 - fy * k));
        var cx = (float)(size.Width / 2);
        var cy = (float)(size.Height / 2);
        var ink = style.Dark ? Color.FromArgb(255, 255, 255, 255) : Color.FromArgb(255, 0, 0, 0);
        Color Ink(byte a) => Color.FromArgb(a, ink.R, ink.G, ink.B);
        ds.FillCircle(cx, cy, (float)k, Ink(0x14));
        ds.DrawCircle(cx, cy, (float)k, Ink(0x80), 1);
        using (var dash = new CanvasStrokeStyle { CustomDashStyle = [3, 3] })
        {
            ds.DrawLine(0, cy, (float)size.Width, cy, Ink(0x50), 0.5f, dash);
            ds.DrawLine(cx, 0, cx, (float)size.Height, Ink(0x50), 0.5f, dash);
        }
        foreach (var p in _points.Skip(1))
        {
            var (x, y) = Pt(p.X, p.Y);
            ds.FillCircle(x, y, 3, Ink(0x99));
        }
        var accent = Kit.Accent(this);
        var lo = Pt(c.X - _extent, c.Y - _extent);
        var hi = Pt(c.X + _extent, c.Y + _extent);
        var patch = new Rect(lo.X, hi.Y, Math.Max(hi.X - lo.X, 3), Math.Max(lo.Y - hi.Y, 3));
        ds.FillRectangle(patch, Color.FromArgb(0x40, accent.R, accent.G, accent.B));
        ds.DrawRectangle(patch, accent, 1.5f);
        var m = Pt(c.X, c.Y);
        ds.FillCircle(m.X, m.Y, 2.5f, accent);
    }
}
