using System.Text.Encodings.Web;
using System.Text.Json;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Plotting;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Model;
using Windows.Foundation;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// Any result shown as data, never as an error (the Mac app's
/// GenericResultView): scalars as a two-column list, arrays of records as
/// compact tables, number arrays and 1-D buffers as curves, [N, 2] buffers
/// as points, 2-D buffers as maps, and the whole JSON at the end.
/// </summary>
public sealed partial class GenericResultView : UserControl
{
    const int MaxPlots = 16;
    const int MaxJsonChars = 200_000;
    int _plots;

    public GenericResultView(ResultData data)
    {
        IsTabStop = false;
        var stack = new StackPanel { Spacing = 16, Padding = new Thickness(12, 8, 12, 12) };
        try
        {
            Build(stack, data);
        }
        catch (Exception e)
        {
            stack.Children.Add(Text(Loc.F("Results_Generic_ReadError", e.Message), "SecondaryText"));
        }
        Content = new ScrollViewer { Content = stack, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled };
    }

    void Build(StackPanel stack, ResultData data)
    {
        var scalars = GenericData.Scalars(data.Json);
        if (scalars.Count > 0) stack.Children.Add(Section(Loc.S("Results_Generic_Values"), KeyValues(scalars)));
        foreach (var t in GenericData.RecordTables(data.Json)) stack.Children.Add(Section(EngineText.T(GenericData.Title(t.Name)), Table(t)));
        foreach (var (name, values) in GenericData.Curves(data.Json))
            if (Curve(values) is FrameworkElement plot) stack.Children.Add(Section(EngineText.T(GenericData.Title(name)), plot));
        foreach (var b in data.Buffers) stack.Children.Add(Section(b.Name, Buffer(b)));
        if (data.Json is not null)
        {
            var text = data.Json.ToJsonString(new JsonSerializerOptions { WriteIndented = true, Encoder = JavaScriptEncoder.UnsafeRelaxedJsonEscaping });
            if (text.Length > MaxJsonChars) text = text[..MaxJsonChars] + "\n…";
            var json = new TextBlock
            {
                Text = text, FontFamily = new FontFamily("Cascadia Mono, Consolas"), FontSize = 11,
                IsTextSelectionEnabled = true, TextWrapping = TextWrapping.NoWrap,
            };
            var scroll = new ScrollViewer { Content = json, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled };
            stack.Children.Add(new Expander { Header = Loc.S("Results_Generic_Data"), Content = scroll, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Left });
        }
    }

    static TextBlock Text(string s, string style) => new() { Text = s, Style = (Style)Application.Current.Resources[style] };

    static FrameworkElement Section(string title, FrameworkElement content)
    {
        var p = new StackPanel { Spacing = 6 };
        p.Children.Add(Text(title, "PanelTitleText"));
        p.Children.Add(content);
        return p;
    }

    static FrameworkElement KeyValues(IReadOnlyList<(string Key, string Value)> rows)
    {
        var g = new Grid { ColumnSpacing = 16, RowSpacing = 3, HorizontalAlignment = HorizontalAlignment.Left };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        for (var i = 0; i < rows.Count; i++)
        {
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var k = Text(rows[i].Key, "SecondaryText");
            var v = Text(rows[i].Value, "NumberText");
            v.IsTextSelectionEnabled = true;
            Grid.SetRow(k, i);
            Grid.SetRow(v, i);
            Grid.SetColumn(v, 1);
            g.Children.Add(k);
            g.Children.Add(v);
        }
        return g;
    }

    static FrameworkElement Table(GenericData.Table t)
    {
        var g = new Grid { ColumnSpacing = 14, RowSpacing = 2 };
        foreach (var _ in t.Columns) g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        for (var c = 0; c < t.Columns.Count; c++)
        {
            var h = Text(EngineText.T(GenericData.Title(t.Columns[c])), "ColumnHeaderText");
            h.FontSize = 11;
            h.HorizontalAlignment = HorizontalAlignment.Right;
            Grid.SetColumn(h, c);
            g.Children.Add(h);
        }
        for (var r = 0; r < t.Rows.Count; r++)
        {
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            for (var c = 0; c < t.Columns.Count; c++)
            {
                var cell = Text(t.Rows[r][c], "NumberText");
                cell.FontSize = 12;
                cell.HorizontalAlignment = HorizontalAlignment.Right;
                cell.IsTextSelectionEnabled = true;
                Grid.SetRow(cell, r + 1);
                Grid.SetColumn(cell, c);
                g.Children.Add(cell);
            }
        }
        var p = new StackPanel { Spacing = 4 };
        p.Children.Add(new ScrollViewer
        {
            Content = g, HorizontalScrollBarVisibility = ScrollBarVisibility.Auto, VerticalScrollBarVisibility = ScrollBarVisibility.Disabled,
            HorizontalScrollMode = ScrollMode.Auto, VerticalScrollMode = ScrollMode.Disabled,
        });
        if (t.TotalRows > t.Rows.Count) p.Children.Add(Text(Loc.F("Results_Generic_MoreRows", t.TotalRows - t.Rows.Count), "CaptionText"));
        return p;
    }

    FrameworkElement? Curve(double[] values)
    {
        if (_plots++ >= MaxPlots) return null;
        var panel = new ChartPanel { XTitle = Loc.S("Results_Generic_Index"), ZeroLine = false };
        panel.Series.Add(new ChartSeries(values.Select((_, i) => (double)i).ToArray(), values, SeriesColor.Field(0)));
        return new PlotCanvas((ds, size, style) => LineChartView.DrawPanel(ds, panel, new Rect(0, 0, size.Width, size.Height), style)) { Height = 130 };
    }

    FrameworkElement Buffer(IResultArray b)
    {
        var p = new StackPanel { Spacing = 4 };
        var (lo, hi, missing) = GenericData.Range(b);
        var range = double.IsFinite(lo)
            ? $"[{string.Join(" × ", b.Shape)}] · {Fmt.Short(lo)} … {Fmt.Short(hi)}" + (missing > 0 ? Loc.F("Results_Generic_Missing", missing) : "")
            : Loc.F("Results_Generic_NoValues", $"[{string.Join(" × ", b.Shape)}]");
        p.Children.Add(Text(range, "CaptionText"));
        var kind = GenericData.KindOf(b);
        if (kind != GenericData.BufferKind.Summary && _plots++ < MaxPlots)
        {
            switch (kind)
            {
                case GenericData.BufferKind.Curve:
                    var v = Json.Doubles(b);
                    var panel = new ChartPanel { XTitle = Loc.S("Results_Generic_Index"), ZeroLine = false };
                    panel.Series.Add(new ChartSeries(v.Select((_, i) => (double)i).ToArray(), v, SeriesColor.Field(0)));
                    p.Children.Add(new PlotCanvas((ds, size, style) => LineChartView.DrawPanel(ds, panel, new Rect(0, 0, size.Width, size.Height), style)) { Height = 130 });
                    break;
                case GenericData.BufferKind.Points:
                    p.Children.Add(new PlotCanvas((ds, size, style) => DrawPoints(ds, b, size, style)) { Height = 220 });
                    break;
                case GenericData.BufferKind.Map:
                    var map = new HeatMap(b);
                    p.Children.Add(new PlotCanvas((ds, size, style) => map.Draw(ds, new Rect(0, 0, size.Width, size.Height), style)) { Height = 220 });
                    break;
            }
        }
        return p;
    }

    /// <summary>[N, 2] points as dots on a light frame.</summary>
    static void DrawPoints(Microsoft.Graphics.Canvas.CanvasDrawingSession ds, IResultArray b, Size size, PlotStyle style)
    {
        var n = b.Count / 2;
        var xs = new double[n];
        var ys = new double[n];
        for (var i = 0; i < n; i++) { xs[i] = b[2 * i]; ys[i] = b[2 * i + 1]; }
        var plot = new LinePlot { XTitle = "x", YTitle = "y", ZeroLine = true };
        (plot.XMin, plot.XMax) = Ticks.Range(xs);
        (plot.YMin, plot.YMax) = Ticks.Range(ys);
        var r = plot.Draw(ds, new Rect(0, 0, size.Width, size.Height), style);
        if (r.IsEmpty) return;
        var color = Palette.Field(0, style.Dark).WithAlpha(0.75);
        using (ds.CreateLayer(1, r))
        {
            for (var i = 0; i < n; i++)
            {
                if (!double.IsFinite(xs[i]) || !double.IsFinite(ys[i])) continue;
                var x = r.X + (xs[i] - plot.XMin) / (plot.XMax - plot.XMin) * r.Width;
                var y = r.Bottom - (ys[i] - plot.YMin) / (plot.YMax - plot.YMin) * r.Height;
                ds.FillCircle((float)x, (float)y, 1.3f, color);
            }
        }
    }
}
