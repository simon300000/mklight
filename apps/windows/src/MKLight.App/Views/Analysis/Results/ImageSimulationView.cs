using System.Runtime.CompilerServices;
using Microsoft.Graphics.Canvas;
using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;
using Windows.ApplicationModel.DataTransfer;
using Windows.Foundation;
using Windows.Storage;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// <c>image_simulation</c>: the engine's linear-light image (sRGB for display, row 0 at the
/// low-y edge) beside the source it was run with (a built-in pattern drawn at the output size
/// with the engine's definition, or the imported image), side by side, split or blinking; an
/// exposure in stops; zoom (the wheel, about the pointer) and pan (a drag), the same on every
/// image; the patch's field of view, centre and resolution and the source (a pattern, a chosen
/// image file or one dropped on the images); the run's pixel, patch, PSF window and field mapping.
/// </summary>
public sealed partial class ImageSimulationView : UserControl, IViewOptions
{
    readonly Dictionary<string, Action> _opts = [];
    public void Apply(string option) { if (_opts.TryGetValue(option, out var a)) a(); }
    enum Compare { SideBySide, Split, Blink }

    /// <summary>What a pane's view keeps across runs (a new result makes a new view).</summary>
    sealed class State
    {
        public double Zoom = 1, U = 0.5, V = 0.5;
        public int Compare;
    }
    static readonly ConditionalWeakTable<AnalysisPaneModel, State> states = new();
    public const double MaxZoom = 32;

    readonly ImageSimulationData _d;
    readonly PlotCanvas _canvas;
    readonly TextBlock _ev;
    readonly FlowPanel _top = new() { Spacing = 14, LineSpacing = 4 };
    readonly FlowPanel _caption = new() { Spacing = 14, LineSpacing = 2 };
    readonly Choice _compare;
    readonly DispatcherTimer _blink = new() { Interval = TimeSpan.FromMilliseconds(800) };
    readonly TextBlock _zoomText;
    readonly Button _zoomOut, _zoomIn, _fitButton;
    State _s = new();
    AnalysisPaneModel? _pane;
    ImageSimulationData _run;
    LinearImage? _source;
    string _sourceTitle = "Source";
    Compare _mode;
    double _exposure;
    double _split = 0.5;
    bool _blinkSource;
    bool _splitting;
    (Point At, double U, double V, Rect Frame)? _panFrom;
    bool _dropTargeted;
    string? _dropFailed;
    (CanvasDevice Device, double Gain)? _simKey;
    CanvasBitmap? _sim, _src;
    CanvasDevice? _srcDevice;
    Rect _fit;
    readonly List<Rect> _frames = [];

    public ImageSimulationView(ResultData data, ImageSimulationData d)
    {
        _d = _run = d;
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw) { AllowDrop = true };
        _canvas.DeviceCreated += _ => { _sim = null; _src = null; };
        _compare = new Choice("Compare", ["Side by Side", "Split", "Blink"]);
        _compare.Changed += k => { _mode = (Compare)k; _s.Compare = k; UpdateBlink(); Describe(); _canvas.Invalidate(); };
        _opts["split"] = () => _compare.Select(1);
        _opts["blink"] = () => _compare.Select(2);

        var exposure = new Slider { Minimum = -4, Maximum = 4, StepFrequency = 0.5, SmallChange = 0.5, Value = 0, Width = 96, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, -4, 0, -4) };
        AutomationProperties.SetName(exposure, "Exposure");
        _ev = Kit.Text("+0.0 EV", "NumberText", 12);
        _ev.Width = 52;
        _ev.VerticalAlignment = VerticalAlignment.Center;
        exposure.ValueChanged += (_, e) => { _exposure = e.NewValue; _ev.Text = Fmt.Signed(_exposure, 1).Trim() + " EV"; _sim = null; UpdateCaption(); Describe(); _canvas.Invalidate(); };
        var exp = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        var expLabel = Kit.Text("Exposure", "SecondaryText", 12);
        expLabel.VerticalAlignment = VerticalAlignment.Center;
        exp.Children.Add(expLabel);
        exp.Children.Add(exposure);
        exp.Children.Add(_ev);
        ToolTipService.SetToolTip(exp, "Scales the simulated image's linear light by 2^EV; the source is shown at unit exposure");

        // Zoom: out, the factor, in, and Fit (also a double-click on the images).
        _zoomOut = GlyphButton("\uE71F", "Zoom out", () => SetZoom(_s.Zoom / 2));
        _zoomIn = GlyphButton("\uE8A3", "Zoom in", () => SetZoom(_s.Zoom * 2));
        _zoomText = Kit.Text("100%", "NumberText", 12);
        _zoomText.MinWidth = 40;
        _zoomText.TextAlignment = TextAlignment.Center;
        _zoomText.VerticalAlignment = VerticalAlignment.Center;
        _fitButton = Plain(Kit.Text("Fit", "BodyText", 12), "Fit the image and reset the pan (or double-click the image)", Fit);
        AutomationProperties.SetName(_fitButton, "Fit");
        var zoom = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 0 };
        zoom.Children.Add(_zoomOut);
        zoom.Children.Add(_zoomText);
        zoom.Children.Add(_zoomIn);
        zoom.Children.Add(_fitButton);
        ToolTipService.SetToolTip(_zoomText, "Wheel over the images to zoom about the pointer; drag to pan");

        _top.Children.Add(_compare);
        _top.Children.Add(exp);
        _top.Children.Add(zoom);
        if (d.Nodes.Count > 0)
        {
            var mapping = Plain(Kit.LegendItem(new FontIcon { Glyph = "\uE80A", FontSize = 12 }, Kit.Text("Field Mapping", "BodyText", 12)), null, () => { });
            mapping.Flyout = new Flyout { Content = MappingTable(d), Placement = FlyoutPlacementMode.Bottom };
            AutomationProperties.SetName(mapping, "Field mapping");
            _top.Children.Add(mapping);
        }

        Content = FieldMapViews.Shell(_top, _canvas, _caption);
        // Shell gives the last row the space: the canvas is the middle row here.
        if (Content is Grid g)
        {
            g.RowDefinitions[1].Height = new GridLength(1, GridUnitType.Star);
            g.RowDefinitions[2].Height = GridLength.Auto;
        }

        Pointers();
        DragAndDrop();
        _blink.Tick += (_, _) => { _blinkSource = !_blinkSource; _canvas.Invalidate(); };
        Loaded += (_, _) =>
        {
            if (Pane(this) is { } pane && _pane is null)
            {
                _pane = pane;
                _s = states.GetValue(pane, _ => new State());
                _compare.Select(_s.Compare);
                AddPaneControls(pane);
                // The run's samples of an imported file, derived again if the output size or field changed.
                if (ImageSourceImport.NeedsRefresh(pane.Settings, ImageSourceImport.FieldAspect(pane.Output?.Json))) _ = RefreshImportAsync(pane);
            }
            // The source the run used comes with its settings.
            var settings = ResultViews.RunSettings(this);
            _run = settings is null ? _d : ImageSimulationData.Read(data, settings) ?? _d;
            _source = _run.SourceImage();
            _sourceTitle = _run.SourceKind == "buffer" && ImageSourceImport.FileName(_pane?.Settings) is { } file ? file : _run.SourceTitle;
            _src = null;
            _compare.IsEnabled = _source is not null;
            UpdateCaption();
            UpdateZoom();
            UpdateBlink();
            _canvas.Invalidate();
        };
        Unloaded += (_, _) => _blink.Stop();
        UpdateCaption();
        UpdateZoom();
    }

    /// <summary>The pane that shows <paramref name="view"/> (null when it is not in a pane).</summary>
    static AnalysisPaneModel? Pane(FrameworkElement view)
    {
        DependencyObject? e = view;
        while (e is not null)
        {
            if (e is AnalysisPaneView pane) return pane.Pane;
            e = VisualTreeHelper.GetParent(e);
        }
        return null;
    }

    async Task RefreshImportAsync(AnalysisPaneModel pane)
    {
        if (!await ImageSimulationFiles.RefreshAsync(pane))
        {
            _dropFailed = Path.GetFileName(ImageSourceImport.FileName(pane.Settings) ?? "The image");
            UpdateCaption();
        }
    }

    static Button GlyphButton(string glyph, string name, Action click)
    {
        var b = Plain(new FontIcon { Glyph = glyph, FontSize = 13 }, name, click);
        AutomationProperties.SetName(b, name);
        return b;
    }

    /// <summary>A borderless button of the view's control row.</summary>
    static Button Plain(object content, string? tip, Action click)
    {
        var b = new Button
        {
            Content = content, Padding = new Thickness(8, 2, 8, 3), MinHeight = 26, MinWidth = 0,
            Background = new SolidColorBrush(Color.FromArgb(0, 0, 0, 0)), BorderThickness = new Thickness(0),
        };
        if (tip is not null) ToolTipService.SetToolTip(b, tip);
        b.Click += (_, _) => click();
        return b;
    }

    // MARK: Zoom and pan

    /// <summary>Zooms to <paramref name="zoom"/> (1 … 32), about the point (<paramref name="u"/>, <paramref name="v"/>) of the image (0 … 1, top left) when given, else about the centre of the view.</summary>
    public void SetZoom(double zoom, double? u = null, double? v = null)
    {
        _s.Zoom = Math.Clamp(zoom, 1, MaxZoom);
        if (u is double uu) _s.U = uu;
        if (v is double vv) _s.V = vv;
        Clamp();
        UpdateZoom();
        _canvas.Invalidate();
    }

    void Fit()
    {
        _s.Zoom = 1;
        _s.U = _s.V = 0.5;
        UpdateZoom();
        _canvas.Invalidate();
    }

    /// <summary>Keeps the view inside the image.</summary>
    void Clamp()
    {
        var h = 0.5 / _s.Zoom;
        _s.U = Math.Clamp(_s.U, h, 1 - h);
        _s.V = Math.Clamp(_s.V, h, 1 - h);
    }

    /// <summary>Zooms by <paramref name="factor"/> keeping the image point under <paramref name="p"/> in place.</summary>
    void ZoomAt(Point p, double factor)
    {
        var r = FrameAt(p) ?? _fit;
        var z = Math.Clamp(_s.Zoom * factor, 1, MaxZoom);
        if (r.Width < 2 || r.Height < 2 || z == _s.Zoom) { SetZoom(z); return; }
        double px = Math.Clamp((p.X - r.X) / r.Width, 0, 1) - 0.5, py = Math.Clamp((p.Y - r.Y) / r.Height, 0, 1) - 0.5;
        double ix = _s.U + px / _s.Zoom, iy = _s.V + py / _s.Zoom;
        SetZoom(z, ix - px / z, iy - py / z);
    }

    Rect? FrameAt(Point p)
    {
        foreach (var r in _frames) if (r.Contains(p)) return r;
        return null;
    }

    void UpdateZoom()
    {
        _zoomText.Text = Fmt.Fixed(_s.Zoom * 100, 0) + "%";
        _zoomOut.IsEnabled = _s.Zoom > 1;
        _zoomIn.IsEnabled = _s.Zoom < MaxZoom;
        _fitButton.IsEnabled = _s.Zoom != 1 || _s.U != 0.5 || _s.V != 0.5;
        AutomationProperties.SetName(_zoomText, "Zoom " + _zoomText.Text);
        Describe();
    }

    void Pointers()
    {
        _canvas.PointerPressed += (_, e) =>
        {
            var pt = e.GetCurrentPoint(_canvas);
            if (!pt.Properties.IsLeftButtonPressed) return;
            var p = pt.Position;
            if (_mode == Compare.Split && _source is not null && (_s.Zoom <= 1 || NearHandle(p)))
            {
                _splitting = true;
                _canvas.CapturePointer(e.Pointer);
                SplitAt(p.X);
            }
            else if (_s.Zoom > 1 && FrameAt(p) is { } r)
            {
                _panFrom = (p, _s.U, _s.V, r);
                _canvas.CapturePointer(e.Pointer);
            }
        };
        _canvas.PointerMoved += (_, e) =>
        {
            var p = e.GetCurrentPoint(_canvas).Position;
            if (_splitting) SplitAt(p.X);
            else if (_panFrom is { } f)
            {
                _s.U = f.U - (p.X - f.At.X) / (f.Frame.Width * _s.Zoom);
                _s.V = f.V - (p.Y - f.At.Y) / (f.Frame.Height * _s.Zoom);
                Clamp();
                _fitButton.IsEnabled = true;
                _canvas.Invalidate();
            }
            ProtectedCursor = _splitting || _mode == Compare.Split && _source is not null && NearHandle(p) ? InputSystemCursor.Create(InputSystemCursorShape.SizeWestEast)
                : _s.Zoom > 1 && (_panFrom is not null || FrameAt(p) is not null) ? InputSystemCursor.Create(InputSystemCursorShape.SizeAll)
                : null;
        };
        void End()
        {
            _splitting = false;
            if (_panFrom is not null) { _panFrom = null; UpdateZoom(); }
        }
        _canvas.PointerReleased += (_, e) => { End(); _canvas.ReleasePointerCapture(e.Pointer); };
        _canvas.PointerCaptureLost += (_, _) => End();
        _canvas.PointerExited += (_, _) => { if (!_splitting && _panFrom is null) ProtectedCursor = null; };
        _canvas.PointerWheelChanged += (_, e) =>
        {
            var pt = e.GetCurrentPoint(_canvas);
            if (pt.Properties.IsHorizontalMouseWheel || pt.Properties.MouseWheelDelta == 0) return;
            // A notch (120) zooms by √2: two notches double.
            ZoomAt(pt.Position, Math.Pow(2, pt.Properties.MouseWheelDelta / 240.0));
            e.Handled = true;
        };
        _canvas.DoubleTapped += (_, e) => { Fit(); e.Handled = true; };
        _canvas.Tapped += (_, _) =>
        {
            // Blink without motion (the system asks for less): a click switches.
            if (_mode == Compare.Blink && !_blink.IsEnabled && _source is not null) { _blinkSource = !_blinkSource; _canvas.Invalidate(); }
        };
    }

    bool NearHandle(Point p) => Math.Abs(p.X - (_fit.X + _fit.Width * _split)) <= 12 && p.Y >= _fit.Y && p.Y <= _fit.Bottom;

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

    // MARK: Dropping an image file

    void DragAndDrop()
    {
        _canvas.DragOver += (_, e) =>
        {
            if (_pane is null || !e.DataView.Contains(StandardDataFormats.StorageItems)) return;
            e.AcceptedOperation = DataPackageOperation.Copy;
            e.DragUIOverride.Caption = "Use as Source";
            if (!_dropTargeted) { _dropTargeted = true; _canvas.Invalidate(); }
        };
        _canvas.DragLeave += (_, _) => { _dropTargeted = false; _canvas.Invalidate(); };
        _canvas.Drop += async (_, e) =>
        {
            _dropTargeted = false;
            _canvas.Invalidate();
            if (_pane is not { } pane || !e.DataView.Contains(StandardDataFormats.StorageItems)) return;
            var deferral = e.GetDeferral();
            IReadOnlyList<IStorageItem> items;
            try { items = await e.DataView.GetStorageItemsAsync(); }
            finally { deferral.Complete(); }
            if (items.OfType<StorageFile>().FirstOrDefault(f => ImageSourceImport.IsImageFile(f.Path)) is not { } file)
            {
                _dropFailed = items.FirstOrDefault()?.Name;
                UpdateCaption();
                return;
            }
            await UseFileAsync(pane, file.Path);
        };
    }

    async Task UseFileAsync(AnalysisPaneModel pane, string path)
    {
        var ok = await ImageSimulationFiles.UseAsync(pane, path);
        _dropFailed = ok ? null : Path.GetFileName(path);
        UpdateCaption();
    }

    // MARK: Caption and accessibility

    void UpdateCaption()
    {
        var d = _run;
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
            var at = CenterTitle((cx, cy));
            Add((at == "on axis" ? "On axis" : "At " + at) + (d.FieldExtent is double e ? " · view " + ImageSimulationSettings.ExtentTitle(e) : ""),
                "Patch centered at normalized field " + ImageSimulationData.Coordinates(cx, cy) + (near is not null && double.IsFinite(near.RealX) ? $", image point ({Fmt.Fixed(near.RealX, 3)}, {Fmt.Fixed(near.RealY, 3)}) mm" : "") + ".");
        }
        if (d.PsfSize is int n && double.IsFinite(d.PsfSpacing)) Add(Kit.Inv($"PSF {n} × {Fmt.Length(d.PsfSpacing)}"), "The PSF window and sample spacing this run used");
        var clipped = img.ClippedFraction(Math.Pow(2, _exposure));
        if (clipped > 0.001) Add(Fmt.Fixed(clipped * 100, 1) + "% clipped", "Samples brighter than full scale at this exposure", warn: true);
        if (_dropFailed is not null) Add(_dropFailed + " could not be read", "Choose an image file Windows can open (PNG, JPEG, TIFF, BMP, GIF, HEIC, WebP)", warn: true);
        if (d.SourceKind == "pattern")
            Add("Pattern area-sampled", "The engine evaluates the built-in pattern at each pixel's exact source point (through the distortion), averaged over the pixel, so the blur is the lens's. The source shown is drawn with the same definition.");
        Describe();
    }

    string CenterTitle((double X, double Y) c) => ImageSimulationSettings.CenterTitle(c, FieldPoints());

    /// <summary>The canvas' name and help: what the images show, for a screen reader.</summary>
    void Describe()
    {
        var d = _run;
        var name = Kit.Inv($"Simulated image, {d.Image.Width}×{d.Image.Height} px");
        if (d.FieldExtent is double e) name += ", field " + ImageSimulationSettings.ExtentTitle(e);
        if (d.FieldCenter is var (cx, cy)) name += ", centre " + CenterTitle((cx, cy));
        AutomationProperties.SetName(_canvas, name);
        var mode = _source is null ? "the simulated image alone" : _mode switch
        {
            Compare.Split => "split with the source " + _sourceTitle + Kit.Inv($" ({_split * 100:F0} % source)"),
            Compare.Blink => "blinking with the source " + _sourceTitle,
            _ => "side by side with the source " + _sourceTitle,
        };
        AutomationProperties.SetHelpText(_canvas, "Shown " + mode + ", zoom " + Fmt.Fixed(_s.Zoom * 100, 0) + " %, exposure " + _ev.Text
            + ". Wheel to zoom about the pointer, drag to pan, double-click to fit; drop an image file to use it as the source.");
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

    // MARK: Drawing

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
        _frames.Clear();
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
                    Zoomed(ds, _src!, _fit);
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
        if (_dropTargeted)
        {
            var accent = Kit.Accent(this);
            ds.DrawRoundedRectangle(new Rect(1, 1, Math.Max(size.Width - 2, 0), Math.Max(size.Height - 2, 0)), 6, 6, accent, 2);
        }
    }

    static readonly Color White = Color.FromArgb(255, 255, 255, 255);

    /// <summary>The visible part of <paramref name="bmp"/> at the view's zoom and pan, filling <paramref name="r"/>; pixels kept sharp.</summary>
    void Zoomed(CanvasDrawingSession ds, CanvasBitmap bmp, Rect r)
    {
        var b = bmp.Bounds;
        double w = b.Width / _s.Zoom, h = b.Height / _s.Zoom;
        var src = new Rect(b.X + (_s.U * b.Width - w / 2), b.Y + (_s.V * b.Height - h / 2), w, h);
        ds.DrawImage(bmp, r, src, 1, CanvasImageInterpolation.NearestNeighbor);
    }

    void Image(CanvasDrawingSession ds, CanvasBitmap? bmp, Rect r, string? title, PlotStyle style)
    {
        if (r.Width < 2 || r.Height < 2) return;
        _frames.Add(r);
        ds.FillRectangle(r, Color.FromArgb(255, 0, 0, 0));
        if (bmp is not null) Zoomed(ds, bmp, r);
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
