using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.Windows.Storage.Pickers;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using static MKLight.App.Views.Analysis.Settings.FormControls;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// The run-local spectrum of an analysis (the Mac app's SpectrumControl):
/// the design's wavelengths, or a photopic, scotopic, blackbody, explicit or
/// sensor spectrum with its sampling, and a small chart of the weights the
/// run will use: resolved by the engine (a <c>system_summary</c> run with
/// the spectrum) when it answers, else a local preview labelled approximate.
/// Writes the engine's JSON; the design's wavelengths remove the setting.
/// </summary>
public sealed partial class SpectrumControl : UserControl
{
    IReadOnlyList<SpectrumSample> _system;
    int? _systemPrimary;
    readonly Action<JsonNode?> _write;
    readonly StackPanel _root = new() { Spacing = 6 };
    JsonNode? _value;
    SpectrumCsv? _report;
    (JsonNode Key, IReadOnlyList<SpectrumSample> Samples)? _resolved;
    CancellationTokenSource? _resolving;
    bool _rebuildQueued;
    static (JsonNode? Value, SpectrumCsv Report)? s_lastImport;
    static (JsonNode Key, IReadOnlyList<SpectrumSample> Samples)? s_lastResolved;

    /// <param name="value">The <c>spectrum</c> setting; null is the design's wavelengths.</param>
    /// <param name="system">
    /// The design's wavelengths (shown for "Design wavelengths", the seed of an explicit list);
    /// once shown, the control takes their weights from its window's design.
    /// </param>
    /// <param name="systemPrimary">The design's primary wavelength.</param>
    /// <param name="write">Stores a new value (null: the design's wavelengths).</param>
    public SpectrumControl(JsonNode? value, IReadOnlyList<SpectrumSample> system, int? systemPrimary, Action<JsonNode?> write)
    {
        _value = value?.DeepClone();
        // The pane rebuilds its form for every new result: the import report stays with the table it made.
        if (s_lastImport is var (table, report) && Json.Same(table, _value)) _report = report;
        _resolved = s_lastResolved;
        _system = system;
        _systemPrimary = systemPrimary;
        _write = write;
        IsTabStop = false;
        Content = _root;
        Loaded += (_, _) =>
        {
            if (SettingsServices.Window(this)?.Session.Document.System.Wavelengths is { } w)
            {
                _system = w.Wavelengths.Select(x => new SpectrumSample(x.Um, x.Weight)).ToList();
                _systemPrimary = w.Primary;
            }
            Build();
            Resolve();
        };
        Unloaded += (_, _) => _resolving?.Cancel();
        ActualThemeChanged += (_, _) => Build();
    }

    SpectrumSetting? Setting => SpectrumSetting.Read(_value);

    void Write(SpectrumSetting s)
    {
        var j = s.ToJson();
        if (Json.Same(j, _value)) return;
        _value = j?.DeepClone();
        _write(j);
        QueueRebuild();
        Resolve();
    }

    /// <summary>Rebuilds after the control's own event has finished (a box must not vanish under its handler).</summary>
    void QueueRebuild()
    {
        if (_rebuildQueued) return;
        _rebuildQueued = true;
        DispatcherQueue.TryEnqueue(() => { _rebuildQueued = false; Build(); });
    }

    // MARK: Layout

    void Build()
    {
        _root.Children.Clear();
        var setting = Setting;
        // The kind, then its sampling, in one grid so the labels line up.
        var grid = FormGrid();
        AddRow(grid, "Spectrum", KindMenu(setting));
        _root.Children.Add(grid);
        if (setting is null)
        {
            var t = Label("A spectrum this control cannot edit: " + (_value?.ToJsonString() ?? "null"), style: "CaptionText");
            t.TextWrapping = TextWrapping.Wrap;
            t.MaxLines = 3;
            _root.Children.Add(t);
            return;
        }
        if (Editor(setting, grid) is UIElement editor) _root.Children.Add(editor);
        // A file that gave no table leaves the spectrum as it was: say why.
        else if (_report is { IsUsable: false } failed) _root.Children.Add(ImportReport(failed));
        if (Preview(setting) is UIElement preview) _root.Children.Add(preview);
        foreach (var issue in setting.Issues) _root.Children.Add(Warning(issue));
    }

    UIElement Warning(string text)
    {
        var glyph = new FontIcon { Glyph = "\uE7BA", FontSize = 11, Foreground = ThemeLookup.Brush(this, "WarningBrush"), VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) };
        var t = Label(text, style: "CaptionText");
        t.Foreground = ThemeLookup.Brush(this, "WarningBrush");
        t.TextWrapping = TextWrapping.Wrap;
        t.TextTrimming = TextTrimming.None;
        var row = Row(glyph, t);
        row.VerticalAlignment = VerticalAlignment.Top;
        return row;
    }

    // MARK: Kind

    UIElement KindMenu(SpectrumSetting? setting)
    {
        var menu = new MenuFlyout();
        menu.Items.Add(KindItem(SpectrumKind.Design, "\uE8FD"));
        menu.Items.Add(new MenuFlyoutSeparator());
        menu.Items.Add(KindItem(SpectrumKind.Photopic, "\uE706"));
        menu.Items.Add(KindItem(SpectrumKind.Scotopic, "\uE708"));
        menu.Items.Add(KindItem(SpectrumKind.Blackbody, "\uEA80", "…"));
        menu.Items.Add(KindItem(SpectrumKind.Explicit, "\uE9E9"));
        var csv = new MenuFlyoutItem { Text = "Sensor Response from CSV…", Icon = new FontIcon { Glyph = "\uE722" } };
        csv.Click += async (_, _) => await ImportAsync();
        menu.Items.Add(csv);
        var title = setting is null ? "Custom" : SpectrumSetting.Title(setting.Kind);
        var button = Menu(title, menu, "The wavelengths this analysis runs with: the design's, or a spectrum sampled for this run only.");
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(button, "Spectrum");
        return button;
    }

    MenuFlyoutItem KindItem(SpectrumKind k, string glyph, string suffix = "")
    {
        var i = Item(SpectrumSetting.Title(k) + suffix, () => Choose(k));
        i.Icon = new FontIcon { Glyph = glyph };
        return i;
    }

    void Choose(SpectrumKind k)
    {
        _report = null;
        var current = Setting;
        var s = current?.Sampling ?? SpectralSampling.Default;
        switch (k)
        {
            case SpectrumKind.Design: Write(SpectrumSetting.Design); break;
            case SpectrumKind.Photopic: Write(new(k, s.Clamped(SpectrumSetting.PhotopicLo, SpectrumSetting.PhotopicHi))); break;
            case SpectrumKind.Scotopic: Write(new(k, s.Clamped(SpectrumSetting.ScotopicLo, SpectrumSetting.ScotopicHi))); break;
            case SpectrumKind.Blackbody: Write(new(k, s, current?.Kind == SpectrumKind.Blackbody ? current.TemperatureK : 5500)); break;
            case SpectrumKind.Explicit:
                if (current?.Kind == SpectrumKind.Explicit) return;
                var seed = CurrentSamples(current) ?? _system;
                Write(new(k, Samples: seed.Count == 0 ? [new SpectrumSample(0.5876, 1)] : seed.ToList()));
                break;
        }
    }

    // MARK: Editors

    /// <summary>The kind's rows in <paramref name="g"/>, and what goes under the grid.</summary>
    UIElement? Editor(SpectrumSetting s, Grid g)
    {
        switch (s.Kind)
        {
            case SpectrumKind.Design: return null;
            case SpectrumKind.Explicit: return ExplicitEditor(s.List);
            case SpectrumKind.Sensor:
                var summary = Label(TableSummary(s.List), style: "SecondaryNumberText");
                summary.FontSize = 12;
                var import = new Button { Content = "Import CSV…", FontSize = 12, MinHeight = 26, Padding = new Thickness(10, 2, 10, 3), CornerRadius = new CornerRadius(13) };
                ToolTipService.SetToolTip(import, "Two columns, wavelength (µm or nm) and response; a header line is optional.");
                import.Click += async (_, _) => await ImportAsync();
                AddRow(g, "Response", Row(summary, import));
                SamplingRows(g, s);
                return _report is { } r ? ImportReport(r) : null;
            default:
                SamplingRows(g, s);
                return null;
        }
    }

    /// <summary>A two-column grid like the form's: labels on the right of their column.</summary>
    static Grid FormGrid()
    {
        var g = new Grid { ColumnSpacing = 10, RowSpacing = 6 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        return g;
    }

    static void AddRow(Grid g, string label, UIElement control)
    {
        var l = Label(label);
        l.HorizontalAlignment = HorizontalAlignment.Right;
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        var r = g.RowDefinitions.Count - 1;
        Grid.SetRow(l, r);
        Grid.SetRow((FrameworkElement)control, r);
        Grid.SetColumn((FrameworkElement)control, 1);
        g.Children.Add(l);
        g.Children.Add(control);
    }

    /// <summary>A number box (finite numbers; whole for <paramref name="integer"/>).</summary>
    static CommitBox Number(double value, double width, string name, Action<double> commit, bool integer = false)
    {
        var box = new CommitBox(
            exact => integer ? NumberText.Integer(value) : exact ? NumberText.Exact(value) : NumberText.Short(value),
            text => NumberText.Parse(text) is double d && double.IsFinite(d) ? new Edit(integer ? Math.Round(d) : d) : null,
            e => commit(Json.Number(e.Value) ?? value), width);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(box, name);
        return box;
    }

    /// <summary>Sampling, and a blackbody's temperature in the same grid so the fields line up.</summary>
    void SamplingRows(Grid g, SpectrumSetting s)
    {
        var sampling = s.Sampling ?? SpectralSampling.Default;
        if (s.Kind == SpectrumKind.Blackbody)
        {
            var row = Row(Number(s.TemperatureK, 64, "Temperature in kelvin", v => Write(s with { TemperatureK = v })), Label("K"));
            foreach (var p in SpectrumSetting.BlackbodyPresets)
            {
                var b = new Button
                {
                    Content = SpectrumSetting.Kelvin(p), FontSize = 11, MinHeight = 24, MinWidth = 0, Padding = new Thickness(8, 1, 8, 2),
                    CornerRadius = new CornerRadius(12), IsEnabled = p != s.TemperatureK, Margin = new Thickness(p == SpectrumSetting.BlackbodyPresets[0] ? 6 : 0, 0, 0, 0),
                };
                Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(b, FontNumeralAlignment.Tabular);
                ToolTipService.SetToolTip(b, p == 5500 ? "Daylight, 5500 K" : "Tungsten halogen, 3200 K");
                var temperature = p;
                b.Click += (_, _) => Write(s with { TemperatureK = temperature });
                row.Children.Add(b);
            }
            AddRow(g, "Temperature", row);
        }
        var spacing = new ComboBox { FontSize = 12, MinHeight = 28, MinWidth = 0, Padding = new Thickness(8, 3, 6, 4) };
        foreach (var sp in new[] { SamplingSpacing.EqualWeight, SamplingSpacing.Uniform }) spacing.Items.Add(SpectralSampling.SpacingTitle(sp));
        spacing.SelectedIndex = sampling.Spacing == SamplingSpacing.EqualWeight ? 0 : 1;
        spacing.SelectionChanged += (_, _) => Write(s.WithSampling(sampling with { Spacing = spacing.SelectedIndex == 1 ? SamplingSpacing.Uniform : SamplingSpacing.EqualWeight }));
        ToolTipService.SetToolTip(spacing, "Equal energy: bins of equal energy, each sample at its centroid. Uniform: evenly spaced samples weighted by their energy.");
        AddRow(g, "Samples", Row(
            Number(sampling.Samples, 52, "Number of samples", v => Write(s.WithSampling(sampling with
            {
                Samples = (int)Math.Clamp(v, SpectralSampling.MinSamples, SpectralSampling.MaxSamples),
            })), integer: true),
            spacing));
        AddRow(g, "Range", Row(
            Number(sampling.LowerUm, 64, "Shortest wavelength in micrometres", v => Write(s.WithSampling(sampling with { LowerUm = v }))),
            Label("–"),
            Number(sampling.UpperUm, 64, "Longest wavelength in micrometres", v => Write(s.WithSampling(sampling with { UpperUm = v }))),
            Label("µm")));
    }

    UIElement ExplicitEditor(IReadOnlyList<SpectrumSample> list)
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var g = new Grid { ColumnSpacing = 8, RowSpacing = 4 };
        foreach (var w in new[] { 84.0, 64, 28 }) g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(w) });
        void Cell(UIElement e, int row, int col)
        {
            Grid.SetRow((FrameworkElement)e, row);
            Grid.SetColumn((FrameworkElement)e, col);
            g.Children.Add(e);
        }
        g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        var h1 = Label("λ (µm)", style: "CaptionText");
        var h2 = Label("Weight", style: "CaptionText");
        h1.HorizontalAlignment = h2.HorizontalAlignment = HorizontalAlignment.Right;
        Cell(h1, 0, 0);
        Cell(h2, 0, 1);
        for (var k = 0; k < list.Count; k++)
        {
            var i = k;
            g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var dot = new Microsoft.UI.Xaml.Shapes.Ellipse { Width = 7, Height = 7, Fill = new SolidColorBrush(Palette.Wavelength(list[k].Um, dark)), VerticalAlignment = VerticalAlignment.Center };
            var um = Number(list[k].Um, 70, $"Wavelength {k + 1} in micrometres", v => Write(new(SpectrumKind.Explicit, Samples: Replace(list, i, list[i] with { Um = v }))));
            var weight = Number(list[k].Weight, 64, $"Weight of wavelength {k + 1}", v => Write(new(SpectrumKind.Explicit, Samples: Replace(list, i, list[i] with { Weight = v }))));
            var lambda = Row(dot, um);
            lambda.HorizontalAlignment = HorizontalAlignment.Right;
            Cell(lambda, k + 1, 0);
            Cell(weight, k + 1, 1);
            var remove = new Button { Content = new FontIcon { Glyph = "\uE738", FontSize = 11 }, Style = (Style)Application.Current.Resources["CircleButton"], Width = 24, Height = 24, IsEnabled = list.Count > 1 };
            ToolTipService.SetToolTip(remove, "Remove this wavelength");
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(remove, $"Remove wavelength {k + 1}");
            remove.Click += (_, _) => Write(new(SpectrumKind.Explicit, Samples: list.Where((_, j) => j != i).ToList()));
            Cell(remove, k + 1, 2);
        }
        var add = LinkButton("\uE710", "Add", () =>
        {
            var last = list.Count > 0 ? list[^1].Um : 0.55;
            Write(new(SpectrumKind.Explicit, Samples: [.. list, new SpectrumSample(Math.Round((last + 0.05) * 10_000) / 10_000, 1)]));
        });
        var design = LinkButton(null, "Design Wavelengths", () => Write(new(SpectrumKind.Explicit, Samples: _system.ToList())));
        design.IsEnabled = _system.Count > 0 && !_system.SequenceEqual(list);
        ToolTipService.SetToolTip(design, "Start again from the design's wavelengths and weights");
        var p = new StackPanel { Spacing = 4 };
        p.Children.Add(g);
        p.Children.Add(Row(add, design));
        return p;
    }

    static List<SpectrumSample> Replace(IReadOnlyList<SpectrumSample> list, int i, SpectrumSample s)
    {
        var l = list.ToList();
        l[i] = s;
        return l;
    }

    static Button LinkButton(string? glyph, string text, Action click)
    {
        var content = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
        if (glyph is not null) content.Children.Add(new FontIcon { Glyph = glyph, FontSize = 11 });
        content.Children.Add(new TextBlock { Text = text, FontSize = 12 });
        var b = new Button { Content = content, Padding = new Thickness(8, 3, 8, 4), MinHeight = 26, MinWidth = 0, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0) };
        b.Click += (_, _) => click();
        return b;
    }

    static string TableSummary(IReadOnlyList<SpectrumSample> t) =>
        t.Count == 0 ? "No table" : string.Create(CultureInfo.InvariantCulture, $"{t.Count} rows · {t[0].Um:0.000}–{t[^1].Um:0.000} µm");

    UIElement ImportReport(SpectrumCsv r)
    {
        var p = new StackPanel { Spacing = 2 };
        if (r.Table.Count > 0)
            p.Children.Add(Label($"Imported {r.Table.Count.ToString(CultureInfo.InvariantCulture)} rows in {r.Unit}" + (r.Header is null ? "" : ", header skipped"), style: "CaptionText"));
        if (!r.IsUsable && r.BadRows.Count == 0) p.Children.Add(Warning("The file has no table of at least two rows."));
        if (r.BadRows.Count == 0) return p;
        var lines = new StackPanel { Spacing = 1 };
        foreach (var b in r.BadRows.Take(20))
            lines.Children.Add(Label(b.Line == 0 ? $"{b.Text}: {b.Reason}" : $"Line {b.Line.ToString(CultureInfo.InvariantCulture)}: {b.Reason} — {b.Text}", style: "CaptionText"));
        if (r.BadRows.Count > 20) lines.Children.Add(Label($"… and {(r.BadRows.Count - 20).ToString(CultureInfo.InvariantCulture)} more", style: "CaptionText"));
        var n = r.BadRows.Count;
        var expander = new Expander
        {
            Header = Warning(r.Table.Count == 0 && r.BadRows[0].Line == 0 ? "The file could not be read" : $"{n.ToString(CultureInfo.InvariantCulture)} row{(n == 1 ? "" : "s")} skipped"),
            Content = lines, Padding = new Thickness(12, 4, 8, 4), MinHeight = 0, HorizontalAlignment = HorizontalAlignment.Stretch,
        };
        p.Children.Add(expander);
        return p;
    }

    /// <summary>Picks a CSV file and makes it the sensor response (sampling over the table's range).</summary>
    async Task ImportAsync()
    {
        if (SettingsServices.Window(this) is not { } window) return;
        var picker = new FileOpenPicker(window.AppWindow.Id) { SuggestedStartLocation = PickerLocationId.DocumentsLibrary };
        foreach (var ext in new[] { ".csv", ".tsv", ".txt" }) picker.FileTypeFilter.Add(ext);
        picker.FileTypeFilter.Add("*");
        var file = await picker.PickSingleFileAsync();
        if (file?.Path is not { } path) return;
        await ImportFileAsync(path);
    }

    /// <summary>Reads <paramref name="path"/> as a sensor response table (also the <c>spectrum-csv</c> launch action).</summary>
    public async Task ImportFileAsync(string path)
    {
        string text;
        try { text = await File.ReadAllTextAsync(path); }
        catch (Exception) { _report = SpectrumCsv.Unreadable(Path.GetFileName(path)); QueueRebuild(); return; }
        var r = SpectrumCsv.Parse(text);
        _report = r;
        if (r.IsUsable && r.RangeUm is var (lo, hi))
        {
            Write(new(SpectrumKind.Sensor, (Setting?.Sampling ?? SpectralSampling.Default) with { LowerUm = lo, UpperUm = hi }, Samples: r.Table));
            s_lastImport = (_value?.DeepClone(), r);
        }
        QueueRebuild();
    }

    // MARK: Preview

    IReadOnlyList<SpectrumSample>? EngineSamples =>
        _resolved is { } r && _value is not null && Json.Same(r.Key, _value) ? r.Samples : null;

    IReadOnlyList<SpectrumSample>? CurrentSamples(SpectrumSetting? s) => s is null ? null
        : s.Kind == SpectrumKind.Design ? _system
        : EngineSamples ?? s.PreviewSamples();

    UIElement? Preview(SpectrumSetting s)
    {
        if (CurrentSamples(s) is not { Count: > 0 } samples) return null;
        var source = s.Kind switch
        {
            SpectrumKind.Design => SpectrumPreview.Source.Design,
            SpectrumKind.Explicit => SpectrumPreview.Source.Exact,
            _ => EngineSamples is not null ? SpectrumPreview.Source.Engine : SpectrumPreview.Source.Approximate,
        };
        int? primary = source == SpectrumPreview.Source.Design ? _systemPrimary is int p && p >= 0 && p < samples.Count ? p : null : null;
        return new SpectrumPreview(samples, s.PreviewCurve(), source, primary);
    }

    /// <summary>Asks the engine what the current spectrum resolves to, after a short pause; the local preview stays until it answers.</summary>
    void Resolve()
    {
        _resolving?.Cancel();
        if (_value is not { } value || Setting is not { } s || s.Issues.Count > 0 || s.Kind == SpectrumKind.Explicit) return;
        if (EngineSamples is not null) return;
        var cts = _resolving = new CancellationTokenSource();
        var key = value.DeepClone();
        _ = ResolveAsync(key, cts.Token);
    }

    async Task ResolveAsync(JsonNode key, CancellationToken cancel)
    {
        try
        {
            await Task.Delay(250, cancel);
            if (SettingsServices.Window(this) is not { } window) return;
            var samples = await SettingsServices.ResolveAsync(window, key, cancel);
            if (cancel.IsCancellationRequested || samples is null) return;
            DispatcherQueue.TryEnqueue(() =>
            {
                if (cancel.IsCancellationRequested) return;
                _resolved = s_lastResolved = (key, samples);
                Build();
            });
        }
        catch (Exception)
        {
            // The local preview stays (labelled approximate).
        }
    }
}
