using Microsoft.Graphics.Canvas;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// <c>image_simulation</c>: the engine's linear-light image (sRGB for display, row 0 at the
/// low-y edge) beside the source it was run with (a built-in pattern drawn at the output size
/// with the engine's definition, or the imported image), side by side, split or blinking; an
/// exposure in stops; the run's pixel, patch, PSF window and field mapping.
/// </summary>
public sealed partial class ImageSimulationView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    enum Compare { SideBySide, Split, Blink }

    readonly ImageSimulationData _d;
    readonly PlotCanvas _canvas;
    readonly TextBlock _ev;
    readonly FlowPanel _caption = new() { Spacing = 14, LineSpacing = 2 };
    readonly Choice _compare;
    readonly DispatcherTimer _blink = new() { Interval = TimeSpan.FromMilliseconds(800) };
    LinearImage? _source;
    string _sourceTitle = "Source";
    Compare _mode;
    double _exposure;
    double _split = 0.5;
    bool _blinkSource;
    bool _dragging;
    (CanvasDevice Device, double Gain)? _simKey;
    CanvasBitmap? _sim, _src;
    CanvasDevice? _srcDevice;
    Rect _fit;

    public ImageSimulationView(ResultData data, ImageSimulationData d)
    {
        _d = d;
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw);
        _canvas.DeviceCreated += _ => { _sim = null; _src = null; };
        _compare = new Choice("Compare", ["Side by Side", "Split", "Blink"]);
        _compare.Changed += k => { _mode = (Compare)k; UpdateBlink(); _canvas.Invalidate(); };
        _opts["split"] = () => _compare.Select(1);
        _opts["blink"] = () => _compare.Select(2);

        var exposure = new Slider { Minimum = -4, Maximum = 4, StepFrequency = 0.5, SmallChange = 0.5, Value = 0, Width = 96, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, -4, 0, -4) };
        AutomationProperties.SetName(exposure, "Exposure");
        _ev = Kit.Text("+0.0 EV", "NumberText", 12);
        _ev.Width = 52;
        exposure.ValueChanged += (_, e) => { _exposure = e.NewValue; _ev.Text = Fmt.Signed(_exposure, 1).Trim() + " EV"; _sim = null; UpdateCaption(); _canvas.Invalidate(); };
        var exp = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        exp.Children.Add(Kit.Text("Exposure", "SecondaryText", 12));
        exp.Children.Add(exposure);
        exp.Children.Add(_ev);
        ToolTipService.SetToolTip(exp, "Scales the simulated image's linear light by 2^EV; the source is shown at unit exposure");

        var top = new FlowPanel { Spacing = 14, LineSpacing = 4 };
        top.Children.Add(_compare);
        top.Children.Add(exp);
        if (d.Nodes.Count > 0)
        {
            var mapping = new Button
            {
                Content = Kit.LegendItem(new FontIcon { Glyph = "\uE80A", FontSize = 12 }, Kit.Text("Field Mapping", "BodyText", 12)),
                Padding = new Thickness(8, 2, 8, 3), MinHeight = 26, Background = new Microsoft.UI.Xaml.Media.SolidColorBrush(Color.FromArgb(0, 0, 0, 0)), BorderThickness = new Thickness(0),
            };
            mapping.Flyout = new Flyout { Content = MappingTable(d), Placement = FlyoutPlacementMode.Bottom };
            AutomationProperties.SetName(mapping, "Field mapping");
            top.Children.Add(mapping);
        }

        Content = FieldMapViews.Shell(top, _canvas, _caption);
        // Shell gives the last row the space: the canvas is the middle row here.
        if (Content is Grid g)
        {
            g.RowDefinitions[1].Height = new GridLength(1, GridUnitType.Star);
            g.RowDefinitions[2].Height = GridLength.Auto;
        }
        AutomationProperties.SetName(_canvas, $"Simulated image, {d.Image.Width} by {d.Image.Height} pixels");

        _canvas.PointerPressed += (_, e) =>
        {
            if (_mode == Compare.Split && _source is not null) { _dragging = true; _canvas.CapturePointer(e.Pointer); SplitAt(e.GetCurrentPoint(_canvas).Position.X); }
            else if (_mode == Compare.Blink && !_blink.IsEnabled) { _blinkSource = !_blinkSource; _canvas.Invalidate(); }
        };
        _canvas.PointerMoved += (_, e) => { if (_dragging) SplitAt(e.GetCurrentPoint(_canvas).Position.X); };
        _canvas.PointerReleased += (_, e) => { _dragging = false; _canvas.ReleasePointerCapture(e.Pointer); };
        _blink.Tick += (_, _) => { _blinkSource = !_blinkSource; _canvas.Invalidate(); };
        Loaded += (_, _) =>
        {
            // The source the run used comes with its settings.
            var settings = ResultViews.RunSettings(this);
            var dd = settings is null ? _d : ImageSimulationData.Read(data, settings) ?? _d;
            _source = dd.SourceImage();
            _sourceTitle = dd.SourceTitle;
            _src = null;
            _compare.IsEnabled = _source is not null;
            UpdateCaption(dd);
            UpdateBlink();
            _canvas.Invalidate();
        };
        Unloaded += (_, _) => _blink.Stop();
        UpdateCaption();
    }

    void SplitAt(double x)
    {
        _split = Math.Clamp((x - _fit.X) / Math.Max(_fit.Width, 1), 0, 1);
        _canvas.Invalidate();
    }

    void UpdateBlink()
    {
        // No flashing when the system asks for less motion: a click switches instead.
        var animate = new global::Windows.UI.ViewManagement.UISettings().AnimationsEnabled;
        if (_mode == Compare.Blink && _source is not null && animate) _blink.Start(); else _blink.Stop();
    }

    void UpdateCaption(ImageSimulationData? withSource = null)
    {
        var d = withSource ?? _d;
        _caption.Children.Clear();
        void Add(string s, string? tip = null, bool warn = false)
        {
            var t = Kit.Caption(s);
            if (warn) t.Foreground = ThemeLookup.Brush(this, "WarningBrush");
            if (tip is not null) ToolTipService.SetToolTip(t, tip);
            _caption.Children.Add(t);
        }
        var img = d.Image;
        Add(Kit.Inv($"{img.Width} × {img.Height} px") + (img.Channels == 3 ? ", RGB" : ", monochrome"));
        if (double.IsFinite(d.PixelX)) Add(Math.Abs(d.PixelX - d.PixelY) > 1e-9 * d.PixelX ? $"Pixel {Fmt.Length(d.PixelX)} × {Fmt.Length(d.PixelY)}" : "Pixel " + Fmt.Length(d.PixelX));
        if (d.FieldCenter is var (cx, cy))
        {
            var near = d.Nodes.MinBy(n => (n.Hx - cx) * (n.Hx - cx) + (n.Hy - cy) * (n.Hy - cy));
            var at = cx == 0 && cy == 0 ? "On axis" : "At " + ImageSimulationData.Coordinates(cx, cy);
            Add(at + (d.FieldExtent is double e ? $" · view {Fmt.G3(e * 100)} %" : ""),
                "Patch centered at normalized field " + ImageSimulationData.Coordinates(cx, cy) + (near is not null && double.IsFinite(near.RealX) ? $", image point ({Fmt.Fixed(near.RealX, 3)}, {Fmt.Fixed(near.RealY, 3)}) mm" : ""));
        }
        if (d.PsfSize is int n && double.IsFinite(d.PsfSpacing)) Add(Kit.Inv($"PSF {n} × {Fmt.Length(d.PsfSpacing)}"), "The PSF window and sample spacing this run used");
        var clipped = img.ClippedFraction(Math.Pow(2, _exposure));
        if (clipped > 0.001) Add(Fmt.Fixed(clipped * 100, 1) + "% clipped", "Samples brighter than full scale at this exposure", warn: true);
        if (d.SourceKind == "pattern")
            Add("Pattern area-sampled", "The engine evaluates the built-in pattern at each pixel's exact source point (through the distortion), averaged over the pixel, so the blur is the lens's. The source shown is drawn with the same definition.");
    }

    static FrameworkElement MappingTable(ImageSimulationData d)
    {
        var p = new StackPanel { Spacing = 6, MaxHeight = 380 };
        p.Children.Add(Kit.Text("Field Mapping", "PanelTitleText", 13));
        p.Children.Add(Kit.Caption("PSF grid nodes; the image is mapped bilinearly between them."));
        var g = new Grid { ColumnSpacing = 14, RowSpacing = 2 };
        string[] heads = ["Field (Hx, Hy)", "Ideal (mm)", "Real (mm)", "Distortion", "Rel. illum."];
        for (var c = 0; c < heads.Length; c++)
        {
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var h = Kit.Text(heads[c], "ColumnHeaderText", 11);
            h.HorizontalAlignment = c == 0 ? HorizontalAlignment.Left : HorizontalAlignment.Right;
            Grid.SetColumn(h, c);
            g.Children.Add(h);
        }
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        for (var r = 0; r < d.Nodes.Count; r++)
        {
            var n = d.Nodes[r];
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            string[] cells =
            [
                Fmt.Signed(n.Hx, 2) + ", " + Fmt.Signed(n.Hy, 2),
                Fmt.Fixed(n.IdealX, 4) + ", " + Fmt.Fixed(n.IdealY, 4),
                Fmt.Fixed(n.RealX, 4) + ", " + Fmt.Fixed(n.RealY, 4),
                double.IsFinite(n.DistortionPercent) ? Fmt.Signed(n.DistortionPercent, 3) + " %" : "—",
                double.IsFinite(n.RelativeIllumination) ? Fmt.Fixed(n.RelativeIllumination, 3) : "—",
            ];
            for (var c = 0; c < cells.Length; c++)
            {
                var t = Kit.Text(cells[c], "NumberText", 11);
                t.HorizontalAlignment = c == 0 ? HorizontalAlignment.Left : HorizontalAlignment.Right;
                t.IsTextSelectionEnabled = true;
                Grid.SetRow(t, r + 1); Grid.SetColumn(t, c);
                g.Children.Add(t);
            }
        }
        p.Children.Add(new ScrollViewer { Content = g, MaxHeight = 320, VerticalScrollBarVisibility = ScrollBarVisibility.Auto });
        return p;
    }

    CanvasBitmap? Bitmap(CanvasDrawingSession ds, LinearImage? img, double gain) =>
        img is null || !img.IsValid ? null : CanvasBitmap.CreateFromBytes(ds, img.Bgra8(gain), img.Width, img.Height, global::Windows.Graphics.DirectX.DirectXPixelFormat.B8G8R8A8UIntNormalized);

    static Rect Fitted(double aspect, Rect area)
    {
        var w = Math.Min(area.Width, area.Height * aspect);
        var h = w / Math.Max(aspect, 1e-6);
        return new Rect(area.X + (area.Width - w) / 2, area.Y + (area.Height - h) / 2, Math.Max(w, 0), Math.Max(h, 0));
    }

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var gain = Math.Pow(2, _exposure);
        if (_sim is null || _simKey is not { } k || k.Device != ds.Device || k.Gain != gain)
        {
            _sim?.Dispose();
            _sim = Bitmap(ds, _d.Image, gain);
            _simKey = (ds.Device, gain);
        }
        if (_source is not null && (_src is null || _srcDevice != ds.Device))
        {
            _src?.Dispose();
            _src = Bitmap(ds, _source, 1);
            _srcDevice = ds.Device;
        }
        var area = new Rect(0, 0, size.Width, size.Height);
        var aspect = _d.Aspect;
        var mode = _src is null ? Compare.SideBySide : _mode;
        switch (mode)
        {
            case Compare.SideBySide when _src is not null:
                var wide = size.Width / Math.Max(size.Height, 1) > aspect * 1.2;
                const double gap = 8;
                var a = wide ? new Rect(0, 0, (size.Width - gap) / 2, size.Height) : new Rect(0, 0, size.Width, (size.Height - gap) / 2);
                var b = wide ? new Rect(a.Right + gap, 0, a.Width, size.Height) : new Rect(0, a.Bottom + gap, size.Width, a.Height);
                Image(ds, _src, Fitted(aspect, a), _sourceTitle, style);
                Image(ds, _sim, _fit = Fitted(aspect, b), "Simulated", style);
                break;
            case Compare.Split:
                _fit = Fitted(aspect, area);
                Image(ds, _sim, _fit, null, style);
                var x = _fit.X + _fit.Width * _split;
                using (ds.CreateLayer(1, new Rect(_fit.X, _fit.Y, Math.Max(x - _fit.X, 0), _fit.Height)))
                    ds.DrawImage(_src!, _fit, _src!.Bounds, 1, CanvasImageInterpolation.NearestNeighbor);
                ImageTag(ds, _sourceTitle, _fit.X + 6, _fit.Y + 6, TextAnchor.Start);
                ImageTag(ds, "Simulated", _fit.Right - 6, _fit.Y + 6, TextAnchor.End);
                ds.DrawLine((float)x, (float)_fit.Y, (float)x, (float)_fit.Bottom, Color.FromArgb(0x66, 0, 0, 0), 3);
                ds.DrawLine((float)x, (float)_fit.Y, (float)x, (float)_fit.Bottom, White, 1.5f);
                var cy = (float)(_fit.Y + _fit.Height / 2);
                ds.FillCircle((float)x, cy, 10, Color.FromArgb(0x55, 0, 0, 0));
                ds.FillCircle((float)x, cy, 9, White);
                PlotText.Draw(ds, "◂▸", x, cy, Color.FromArgb(255, 0x20, 0x20, 0x20), 9, bold: true);
                break;
            case Compare.Blink:
                _fit = Fitted(aspect, area);
                if (_blinkSource) Image(ds, _src, _fit, _sourceTitle, style);
                else Image(ds, _sim, _fit, "Simulated", style);
                break;
            default:
                Image(ds, _sim, _fit = Fitted(aspect, area), "Simulated", style);
                break;
        }
    }

    static readonly Color White = Color.FromArgb(255, 255, 255, 255);

    static void Image(CanvasDrawingSession ds, CanvasBitmap? bmp, Rect r, string? title, PlotStyle style)
    {
        if (r.Width < 2 || r.Height < 2) return;
        ds.FillRectangle(r, Color.FromArgb(255, 0, 0, 0));
        if (bmp is not null) ds.DrawImage(bmp, r, bmp.Bounds, 1, CanvasImageInterpolation.NearestNeighbor);
        ds.DrawRectangle(r, style.Grid, 1);
        if (title is not null) ImageTag(ds, title, r.X + 6, r.Y + 6, TextAnchor.Start);
    }

    /// <summary>A label on an image: small and legible on any picture.</summary>
    static void ImageTag(CanvasDrawingSession ds, string text, double x, double y, TextAnchor anchor)
    {
        var s = PlotText.Measure(ds, text, 10, true);
        var left = anchor == TextAnchor.End ? x - s.Width - 10 : x;
        ds.FillRoundedRectangle(new Rect(left, y, s.Width + 10, s.Height + 4), 4, 4, Color.FromArgb(0x8C, 0, 0, 0));
        PlotText.Draw(ds, text, left + 5, y + 2 + s.Height / 2, White, 10, TextAnchor.Start, bold: true);
    }
}
