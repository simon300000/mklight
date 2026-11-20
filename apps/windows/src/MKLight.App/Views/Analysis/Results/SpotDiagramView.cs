using Microsoft.Graphics.Canvas;
using Microsoft.Graphics.Canvas.Geometry;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using Windows.Foundation;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// The spot diagram as a field × focus grid (docs/APP.md "Spot diagram"):
/// fields as rows, focus planes as columns, each cell's RMS under it, the
/// Airy disk dashed, markers sized by density and drawn as one group per
/// wavelength. The box scale is common to the grid or per field.
/// </summary>
public sealed partial class SpotDiagramView : UserControl, IViewOptions
{
    CompareSwitch? _compare;
    /// <summary><c>result-view=design</c> / <c>candidate</c>: the Design | Candidate switch (for screenshots).</summary>
    public void Apply(string option)
    {
        if (option is "design" or "candidate") _compare?.Select(option == "candidate");
    }

    readonly SpotGridData _candidate;
    /// <summary>While a candidate is previewed: the design's grid (the switch shows either, on shared boxes); else null.</summary>
    readonly SpotGridData? _design;
    SpotGridData _grid;
    readonly ScrollViewer _scroll;
    readonly PlotCanvas _canvas;
    readonly TextBlock _note;
    readonly Button _common, _perField;
    readonly Border _toggle;
    SpotScale _scale = SpotScale.Grid;
    SpotGridLayout.Metrics _metrics;

    public SpotDiagramView(SpotGridData grid, SpotScale scale = SpotScale.Grid, SpotGridData? design = null)
    {
        _grid = _candidate = grid;
        _design = design;
        _scale = scale;
        IsTabStop = false;
        _canvas = new PlotCanvas(Draw);
        _scroll = new ScrollViewer
        {
            Content = _canvas,
            VerticalScrollBarVisibility = ScrollBarVisibility.Auto,
            HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled,
            HorizontalScrollMode = ScrollMode.Disabled,
        };
        _scroll.SizeChanged += (_, e) => Fit(e.NewSize);

        var legendEntries = new List<LegendEntry>();
        foreach (var w in grid.Wavelengths)
            if (w >= 0 && w < grid.WavelengthsUm.Count && double.IsFinite(grid.WavelengthsUm[w]))
                legendEntries.Add(new LegendEntry(Fmt.Wavelength(grid.WavelengthsUm[w]), SeriesColor.Wavelength(grid.WavelengthsUm[w]), LegendMark.Dot));
        if (double.IsFinite(grid.Airy))
            legendEntries.Add(new LegendEntry("Airy disk " + grid.Length(grid.Airy), SeriesColor.Neutral, LegendMark.DashedCircle));

        // A small segmented control: the chosen scale on a quiet fill (the accent stays for selection elsewhere).
        _common = ScaleButton("Common", SpotScale.Grid);
        _perField = ScaleButton("Per field", SpotScale.Field);
        var segments = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 0 };
        segments.Children.Add(_common);
        segments.Children.Add(_perField);
        _toggle = new Border
        {
            Child = segments, CornerRadius = new CornerRadius(12), BorderThickness = new Thickness(1), Padding = new Thickness(2),
            VerticalAlignment = VerticalAlignment.Top,
            Visibility = grid.Rows.Count > 1 ? Visibility.Visible : Visibility.Collapsed,
        };
        ToolTipService.SetToolTip(_toggle, "Box scale: one for all cells, or one per field");
        ActualThemeChanged += (_, _) => UpdateScale();
        var toggle = _toggle;

        var header = new Grid { ColumnSpacing = 8 };
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        header.Children.Add(new Legend(legendEntries) { VerticalAlignment = VerticalAlignment.Center });
        Grid.SetColumn(toggle, 1);
        header.Children.Add(toggle);

        _note = new TextBlock
        {
            Style = (Style)Application.Current.Resources["CaptionText"],
            TextWrapping = TextWrapping.Wrap,
            TextTrimming = TextTrimming.None,
            Margin = new Thickness(2, 0, 0, 0),
            VerticalAlignment = VerticalAlignment.Center,
        };
        // The note, and while a candidate is previewed the Design | Candidate switch beside it
        // (at the foot, clear of the pane's preview banner over the top of the result).
        var footer = new Grid { ColumnSpacing = 8 };
        footer.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        footer.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        footer.Children.Add(_note);
        if (design is not null)
        {
            var compare = _compare = new CompareSwitch { VerticalAlignment = VerticalAlignment.Bottom };
            compare.Changed += candidate =>
            {
                _grid = candidate ? _candidate : design;
                Fit(new Size(_scroll.ActualWidth, _scroll.ActualHeight));
                UpdateScale();
            };
            Grid.SetColumn(compare, 1);
            footer.Children.Add(compare);
        }

        var root = new Grid { Padding = new Thickness(8, 4, 8, 6), RowSpacing = 6 };
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        root.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        root.Children.Add(header);
        Grid.SetRow(_scroll, 1);
        root.Children.Add(_scroll);
        Grid.SetRow(footer, 2);
        root.Children.Add(footer);
        Content = root;
        UpdateScale();
    }

    SpotGridData? Other => _design is null ? null : ReferenceEquals(_grid, _design) ? _candidate : _design;

    /// <summary>The canvas's spoken summary: what it shows and each field's RMS radius at the base plane.</summary>
    void Describe()
    {
        var who = _design is null ? "" : ReferenceEquals(_grid, _design) ? ", the design" : ", the candidate";
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(_canvas,
            $"Spot diagram{who}, {_grid.Rows.Count} fields by {_grid.Planes.Count} focus planes");
        var rms = _grid.Rows.Select(r => (r.Cells.FirstOrDefault(c => c.Column == _grid.BaseColumn) ?? r.Cells.FirstOrDefault()) is { } c
            ? $"{r.Title} RMS {_grid.Length(c.Rms)}" : null).OfType<string>();
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetHelpText(_canvas,
            string.Join(", ", rms) + (double.IsFinite(_grid.Airy) ? ", Airy radius " + _grid.Length(_grid.Airy) : ""));
    }

    /// <summary>The box scale (the pane may keep it with its layout).</summary>
    public SpotScale BoxScale
    {
        get => _scale;
        set { _scale = value; UpdateScale(); }
    }

    public event Action<SpotScale>? BoxScaleChanged;

    Button ScaleButton(string title, SpotScale scale)
    {
        var b = new Button
        {
            Content = title, FontSize = 11, Padding = new Thickness(9, 1, 9, 2), MinHeight = 20, MinWidth = 0,
            CornerRadius = new CornerRadius(10), BorderThickness = new Thickness(0),
        };
        b.Click += (_, _) =>
        {
            if (_scale == scale) return;
            _scale = scale;
            UpdateScale();
            BoxScaleChanged?.Invoke(scale);
        };
        return b;
    }

    void UpdateScale()
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var ink = dark ? Color.FromArgb(0xE6, 255, 255, 255) : Color.FromArgb(0xE4, 0, 0, 0);
        var quiet = dark ? Color.FromArgb(0x99, 255, 255, 255) : Color.FromArgb(0x8F, 0, 0, 0);
        var fill = dark ? Color.FromArgb(0x24, 255, 255, 255) : Color.FromArgb(0x10, 0, 0, 0);
        _toggle.BorderBrush = new SolidColorBrush(dark ? Color.FromArgb(0x1A, 255, 255, 255) : Color.FromArgb(0x17, 0, 0, 0));
        foreach (var (b, s) in new[] { (_common, SpotScale.Grid), (_perField, SpotScale.Field) })
        {
            var on = s == _scale;
            b.Background = new SolidColorBrush(on ? fill : Color.FromArgb(0, 0, 0, 0));
            b.Foreground = new SolidColorBrush(on ? ink : quiet);
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetItemStatus(b, on ? "Selected" : "");
        }
        _note.Text = _grid.ScaleNote(_scale, Other);
        Describe();
        _canvas.Invalidate();
    }

    void Fit(Size viewport)
    {
        _metrics = SpotGridLayout.Compute(_grid.Planes.Count, _grid.Rows.Count, viewport.Width, viewport.Height);
        var m = _metrics;
        var width = m.RowHeader + _grid.Planes.Count * (m.Cell + SpotGridLayout.Spacing);
        _scroll.HorizontalScrollBarVisibility = m.ScrollsHorizontally ? ScrollBarVisibility.Auto : ScrollBarVisibility.Disabled;
        _scroll.HorizontalScrollMode = m.ScrollsHorizontally ? ScrollMode.Auto : ScrollMode.Disabled;
        _canvas.Width = m.ScrollsHorizontally ? width : double.NaN;
        _canvas.Height = SpotGridLayout.ColumnHeader + SpotGridLayout.Spacing
            + _grid.Rows.Count * (m.Cell + SpotGridLayout.Caption + 2 + SpotGridLayout.Spacing);
        _canvas.Invalidate();
    }

    static readonly CanvasStrokeStyle AiryDash = new() { CustomDashStyle = [3, 2] };

    void Draw(CanvasDrawingSession ds, Size size, PlotStyle style)
    {
        var m = _metrics;
        if (m.Cell <= 0) return;
        var halfWidths = _grid.HalfWidths(_scale, Other);
        var tertiary = style.Label.WithAlpha(style.Label.A / 255.0 * 0.7);
        const double sp = SpotGridLayout.Spacing;

        // Column headers: the plane's defocus; the base plane (the points' own) emphasised.
        if (_grid.Planes.Count > 1) PlotText.Draw(ds, "Defocus", 0, SpotGridLayout.ColumnHeader / 2, tertiary, 10, TextAnchor.Start);
        for (var c = 0; c < _grid.Planes.Count; c++)
        {
            var isBase = c == _grid.BaseColumn && _grid.Planes.Count > 1;
            var x = m.RowHeader + c * (m.Cell + sp) + m.Cell / 2;
            PlotText.Draw(ds, SpotGridData.PlaneLabel(_grid.Planes[c]), x, SpotGridLayout.ColumnHeader / 2,
                isBase ? style.Title : style.Label, 11, bold: isBase);
        }

        var rowHeight = m.Cell + SpotGridLayout.Caption + 2 + sp;
        for (var r = 0; r < _grid.Rows.Count; r++)
        {
            var row = _grid.Rows[r];
            var top = SpotGridLayout.ColumnHeader + sp + r * rowHeight;
            // Row header: the field, its value and, per field, its box half-width.
            PlotText.Draw(ds, row.Title, 0, top + 18, style.Title, 11, TextAnchor.Start, bold: true);
            if (row.Value.Length > 0) PlotText.Draw(ds, Fit(ds, row.Value, m.RowHeader - 4, 10), 0, top + 33, style.Label, 10, TextAnchor.Start);
            if (_scale == SpotScale.Field)
                PlotText.Draw(ds, Fit(ds, "±" + _grid.Length(halfWidths[r]), m.RowHeader - 4, 10), 0, top + 47, tertiary, 10, TextAnchor.Start);
            foreach (var cell in row.Cells)
            {
                var cx = m.RowHeader + cell.Column * (m.Cell + sp);
                DrawCell(ds, cell, new Rect(cx, top, m.Cell, m.Cell), halfWidths[r], style);
                PlotText.Draw(ds, Caption(ds, cell, m.Cell), cx + m.Cell / 2, top + m.Cell + 2 + SpotGridLayout.Caption / 2, style.Label, 10);
            }
        }
    }

    /// <summary>The cell's caption, the longest that fits: "RMS … · GEO …" (base plane), "RMS …", the value.</summary>
    string Caption(CanvasDrawingSession ds, SpotGridData.Cell cell, double width)
    {
        var rms = _grid.Length(cell.Rms);
        var options = new List<string>();
        if (cell.Column == _grid.BaseColumn) options.Add($"RMS {rms} · GEO {_grid.Length(cell.Geo)}");
        options.Add("RMS " + rms);
        options.Add(rms);
        foreach (var o in options)
            if (PlotText.Measure(ds, o, 10).Width <= width) return o;
        return rms;
    }

    static string Fit(CanvasDrawingSession ds, string text, double width, float size)
    {
        if (PlotText.Measure(ds, text, size).Width <= width) return text;
        for (var n = text.Length - 1; n > 1; n--)
        {
            var t = text[..n] + "…";
            if (PlotText.Measure(ds, t, size).Width <= width) return t;
        }
        return text;
    }

    void DrawCell(CanvasDrawingSession ds, SpotGridData.Cell cell, Rect box, double halfWidth, PlotStyle style)
    {
        var s = Math.Min(box.Width, box.Height) / 2 / halfWidth;
        float cx = (float)(box.X + box.Width / 2), cy = (float)(box.Y + box.Height / 2);
        var frame = style.Label.WithAlpha(0.22);
        ds.DrawRectangle(box, frame, 0.75f);
        ds.DrawLine(cx, (float)box.Y, cx, (float)box.Bottom, style.Grid, 1);
        ds.DrawLine((float)box.X, cy, (float)box.Right, cy, style.Grid, 1);
        using (ds.CreateLayer(1, box))
        {
            if (double.IsFinite(_grid.Airy))
                ds.DrawCircle(cx, cy, (float)(_grid.Airy * s), style.Label, 1, AiryDash);
            var (radius, opacity) = SpotGridData.MarkerStyle(cell.PointCount, Math.Min(box.Width, box.Height));
            // One group per wavelength, composited once, so overlapping markers keep one opacity.
            foreach (var g in cell.Groups)
            {
                var um = g.Wavelength >= 0 && g.Wavelength < _grid.WavelengthsUm.Count ? _grid.WavelengthsUm[g.Wavelength] : 0.55;
                var color = Palette.Wavelength(um, style.Dark);
                using (ds.CreateLayer((float)opacity))
                {
                    for (var i = 0; i + 1 < g.Xy.Length; i += 2)
                    {
                        double x = g.Xy[i], y = g.Xy[i + 1];
                        if (!double.IsFinite(x) || !double.IsFinite(y)) continue;
                        ds.FillCircle((float)(cx + x * s), (float)(cy - y * s), (float)radius, color);
                    }
                }
            }
        }
    }
}
