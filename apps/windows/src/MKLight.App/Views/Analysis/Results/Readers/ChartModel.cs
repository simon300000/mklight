namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>What a series' colour means; the view maps it to the palette (Support/Palette.cs).</summary>
public enum ColorRole
{
    /// <summary>Categorical slot of field (or curve) <c>Index</c>.</summary>
    Field,
    /// <summary>The spectral slot of wavelength <c>Um</c>.</summary>
    Wavelength,
    /// <summary>Secondary ink: reference curves and limits (the diffraction limit).</summary>
    Neutral,
}

public readonly record struct SeriesColor(ColorRole Role, int Index = 0, double Um = double.NaN)
{
    public static SeriesColor Field(int i) => new(ColorRole.Field, i);
    public static SeriesColor Wavelength(double um) => new(ColorRole.Wavelength, 0, double.IsFinite(um) ? um : 0.55);
    public static readonly SeriesColor Neutral = new(ColorRole.Neutral);
}

/// <summary>A curve: x and y of equal length; non-finite points break the line.</summary>
public sealed record ChartSeries(double[] X, double[] Y, SeriesColor Color, bool Dashed = false);

public enum LegendMark { Line, Dashed, Dot, DashedCircle, Text }

public sealed record LegendEntry(string Label, SeriesColor Color, LegendMark Mark = LegendMark.Line);

/// <summary>A vertical rule at <c>X</c> (Nyquist, best focus, ±Airy), optionally labelled.</summary>
public sealed record ChartRule(double X, string? Label = null, bool Dashed = true);

/// <summary>One cartesian plot.</summary>
public sealed class ChartPanel
{
    public string? Title { get; init; }
    public string XTitle { get; init; } = "";
    public string YTitle { get; init; } = "";
    public List<ChartSeries> Series { get; } = [];
    /// <summary>Fixed axis ranges; null fits the data.</summary>
    public (double Min, double Max)? XDomain { get; set; }
    public (double Min, double Max)? YDomain { get; set; }
    public bool ZeroLine { get; init; } = true;
    public List<ChartRule> Rules { get; } = [];
}

/// <summary>A row of panels side by side, under an optional label (a field of a fan plot).</summary>
public sealed class ChartGroup
{
    public string? Label { get; init; }
    public List<ChartPanel> Panels { get; } = [];
}

/// <summary>
/// A line-plot result: a legend over groups stacked vertically, each a row
/// of panels. One model serves every line-plot analysis, so one view draws
/// them all.
/// </summary>
public sealed class ChartModel
{
    public List<LegendEntry> Legend { get; } = [];
    public List<ChartGroup> Groups { get; } = [];
    /// <summary>Smallest height of a group before the plot scrolls (fans: 120).</summary>
    public double MinGroupHeight { get; init; } = 160;

    public static ChartModel Single(ChartPanel panel, double minHeight = 160)
    {
        var m = new ChartModel { MinGroupHeight = minHeight };
        var g = new ChartGroup();
        g.Panels.Add(panel);
        m.Groups.Add(g);
        return m;
    }

    public IEnumerable<ChartPanel> Panels => Groups.SelectMany(g => g.Panels);
}
