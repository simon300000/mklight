using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;
using Windows.ApplicationModel.DataTransfer;
using Windows.System;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Tolerance page (the Mac app's <c>TolerancePanel</c>): the tolerance set grouped by element (each row's kind,
/// surface, limits and unit; distribution and symmetry in its context menu), the Default Tolerances generator,
/// the test wavelength, compensators, the criterion and the analysis; while running the stage, elapsed time and
/// time left; then the results: nominal, RSS prediction, levels and yield, the sensitivities by share (an eye
/// previews a build at its worse limit), the Monte-Carlo histogram with the worst trials (a click previews one).
/// </summary>
public sealed partial class TolerancePanel : UserControl
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;
    readonly ToleranceSession _s;
    readonly DesignSession _session;
    readonly JobPanelFrame _frame = new("Tolerance", "\uE9E9");
    readonly HashSet<Guid> _selection = [];
    readonly DispatcherTimer _clock = new() { Interval = TimeSpan.FromSeconds(1) };
    bool _queued;
    JobPhase? _builtPhase;

    public TolerancePanel(DesignSession session)
    {
        _session = session;
        _s = DesignJobs.Of(session).Tolerance;
        Content = _frame;
        _s.Changed += Queue;
        session.Workspace.PreviewChanged += Queue;
        _clock.Tick += (_, _) => { if (_s.IsRunning) Queue(); else _clock.Stop(); };
        Loaded += (_, _) => Rebuild();
    }

    void Queue()
    {
        if (_queued) return;
        _queued = true;
        DispatcherQueue.TryEnqueue(() => { _queued = false; Rebuild(); });
    }

    void Rebuild()
    {
        var focused = XamlRoot is null ? null : FocusManager.GetFocusedElement(XamlRoot) as NumberField;
        var key = focused?.Key;
        var offset = _builtPhase == _s.Phase ? _frame.Scroller.VerticalOffset : 0;
        _builtPhase = _s.Phase;
        var n = _s.Settings.Rows.Count;
        _frame.Subtitle = $"{n} tolerance{(n == 1 ? "" : "s")} · {(_session.Document.System.Name.Length > 0 ? _session.Document.System.Name : "the design")}";
        if (_s.IsRunning && !_clock.IsEnabled) _clock.Start();
        _frame.Body = _s.Phase switch
        {
            JobPhase.Running => Running(),
            JobPhase.Finished => Results(),
            JobPhase.Failed => JobUI.Failure("Tolerancing Failed", _s.Failure ?? ""),
            _ => Setup(),
        };
        Actions();
        if (offset > 0) DispatcherQueue.TryEnqueue(() => _frame.Scroller.ChangeView(null, offset, null, disableAnimation: true));
        if (key is not null) DispatcherQueue.TryEnqueue(() => Find(_frame.Body as DependencyObject, key)?.Focus(FocusState.Programmatic));
    }

    static NumberField? Find(DependencyObject? root, string key)
    {
        if (root is null) return null;
        if (root is NumberField f && f.Key == key) return f;
        var n = VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
            if (Find(VisualTreeHelper.GetChild(root, i), key) is { } hit) return hit;
        return null;
    }

    void Edit(string name, Action<ToleranceSettings> change) => _s.Edit(name, change);

    // ---- Setup ----

    UIElement Setup()
    {
        var st = _s.Settings;
        var sys = _session.Document.System;
        var ctx = _s.Context();
        var issues = st.Issues(ctx);
        var p = new StackPanel { Spacing = 16, Padding = new Thickness(0, 0, 0, 8) };
        if (_s.Result is not null)
        {
            var last = new HyperlinkButton { Content = "Show the last results", Padding = new Thickness(0, 2, 0, 2), FontSize = 12.5 };
            last.Click += (_, _) => _s.ShowResult();
            p.Children.Add(last);
        }

        // Tolerances.
        var t = new StackPanel { Spacing = 6 };
        var defaults = JobUI.Secondary("Default Tolerances…", "\uE8FD", () => { }, "Generate a set from a class of manufacture");
        defaults.Flyout = GeneratorFlyout();
        var add = new DropDownButton { Content = new FontIcon { Glyph = "\uE710", FontSize = 12 }, Flyout = AddMenu(), MinHeight = 32, CornerRadius = new CornerRadius(16) };
        ToolTipService.SetToolTip(add, "Add a tolerance");
        AutomationProperties.SetName(add, "Add a tolerance");
        var tools = JobUI.Row(4, defaults, add,
            JobUI.Icon("\uE8C8", "Duplicate the selected tolerances", () =>
            {
                var ids = _selection.ToHashSet();
                List<Guid> copies = [];
                Edit("Duplicate Tolerance", x => copies = x.Duplicate(ids));
                _selection.Clear();
                _selection.UnionWith(copies);
            }, _selection.Count > 0, 30),
            JobUI.Icon("\uE738", "Delete the selected tolerances (Delete)", DeleteSelected, _selection.Count > 0, 30));
        t.Children.Add(tools);
        t.Children.Add(JobUI.Labeled("Test fringes at", JobUI.Row(5, new NumberField("test-wl", () => st.TestWavelengthUm * 1000,
            v => Edit("Change Test Wavelength", x => x.TestWavelengthUm = v is > 0 ? v.Value / 1000 : ToleranceSettings.DefaultTestWavelengthUm), 70, "632.8", name: "Test wavelength in nm"),
            JobUI.Text("nm", "SecondaryText")), "Power and irregularity in test fringes are counted at this wavelength"));
        if (st.Rows.Count == 0)
            t.Children.Add(JobUI.Hint("No tolerances. Generate a default set, or add one with +."));
        else t.Children.Add(Table(st, sys, issues));
        p.Children.Add(JobUI.Group("Tolerances", t, JobUI.Text($"{st.Rows.Count}", "CaptionText")));

        // Compensators.
        var c = new StackPanel { Spacing = 4 };
        if (st.Compensators.Count == 0) c.Children.Add(JobUI.Hint("None: every build is judged as it is."));
        foreach (var comp in st.Compensators)
        {
            var row = new Grid { ColumnSpacing = 6 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.Children.Add(JobUI.Text(comp.Title(sys)));
            var id = comp.Id;
            var range = JobUI.Row(4, JobUI.Text("±", "SecondaryText"), new NumberField($"comp-{id}", () => comp.Range,
                v => Edit("Change Compensator Range", x => { if (x.Compensators.FirstOrDefault(k => k.Id == id) is { } k) k.Range = v is > 0 ? v : null; }), 64, "free",
                name: $"Range of {comp.Title(sys)}"), JobUI.Text(comp.Param.Unit(sys), "CaptionText"));
            Grid.SetColumn(range, 1);
            row.Children.Add(range);
            var remove = JobUI.Icon("\uE711", "Remove the compensator", () => Edit("Remove Compensator", x => x.Compensators.RemoveAll(k => k.Id == id)), true, 24);
            Grid.SetColumn(remove, 2);
            row.Children.Add(remove);
            c.Children.Add(row);
        }
        var addComp = new DropDownButton { Content = "Add Compensator", Flyout = CompensatorMenu(sys), FontSize = 12.5, MinHeight = 28 };
        c.Children.Add(addComp);
        p.Children.Add(JobUI.Group("Compensators", c));

        // Criterion.
        var merit = _s.Merit;
        var cr = new StackPanel { Spacing = 6 };
        var kinds = ToleranceCriterionKinds.All;
        cr.Children.Add(JobUI.Labeled("Criterion", JobUI.Choice(kinds.Select(k => k.Title()).ToList(), kinds.ToList().IndexOf(st.Criterion.Kind),
            k => Edit("Change Criterion", x => x.Criterion.Kind = kinds[k]), 150)));
        var display = st.Criterion.Display(merit);
        cr.Children.Add(JobUI.Labeled("Pass limit", JobUI.Row(5, new NumberField("threshold", () => st.Criterion.Threshold,
            v => Edit("Change Pass Limit", x => x.Criterion.Threshold = v), 76, "none", name: "Pass limit"), JobUI.Text(display.Unit(), "CaptionText")),
            display.HigherIsBetter() ? "The lowest value a build may have; gives the yield" : "The highest value a build may have; gives the yield"));
        cr.Children.Add(JobUI.Hint(st.Criterion.Kind == ToleranceCriterionKind.DesignMerit
            ? "Each build is judged by the Optimize page's merit function."
            : st.Criterion.Kind == ToleranceCriterionKind.Mtf ? "The MTF at the frequencies of the merit settings, as a shortfall from 1." : "Over the design's fields and wavelengths, by their weights.", 11.5));
        p.Children.Add(JobUI.Group("Criterion", cr));

        // Analysis.
        var a = new StackPanel { Spacing = 6 };
        var modes = ToleranceModes.All;
        var modeSeg = JobUI.Segments(modes.Select(m => m.Title()).ToList(), modes.ToList().IndexOf(st.Mode), k => Edit("Change Analysis", x => x.Mode = modes[k]));
        a.Children.Add(modeSeg);
        if (st.Mode != ToleranceMode.Sensitivity)
        {
            a.Children.Add(JobUI.Labeled("Trials", new NumberField("trials", () => st.Trials, v => Edit("Change Trials", x => x.Trials = Math.Max(2, (int)Math.Round(v ?? 200))), 80, "200", MeritFormat.Short, optional: false, name: "Trials")));
            a.Children.Add(JobUI.Labeled("Seed", new NumberField("tol-seed", () => st.Seed, v => Edit("Change Seed", x => x.Seed = Math.Max(0, (long)Math.Round(v ?? 0))), 80, "0", MeritFormat.Short, optional: false, name: "Seed")));
            a.Children.Add(JobUI.Labeled("Levels", PercentileField(st), "Percentiles of the builds, in 0…100"));
            a.Children.Add(JobUI.Labeled("Keep trials", new NumberField("record", () => st.RecordTrials, v => Edit("Change Kept Trials", x => x.RecordTrials = Math.Clamp((int)Math.Round(v ?? 200), 0, ToleranceSettings.MaxRecordTrials)), 80, "200", MeritFormat.Short, optional: false, name: "Kept trials"),
                "How many trials keep their perturbations and compensators (to preview them)"));
            if (_s.Result?.SecondsPerTrial is > 0 and { } spt)
                a.Children.Add(JobUI.Hint($"Last run: {Duration(spt)} per trial, so {st.Trials.ToString(C)} trials take about {Duration(spt * st.Trials)}.", 11.5));
        }
        p.Children.Add(JobUI.Group("Analysis", a));
        if (_s.RunProblem() is { } why) p.Children.Add(JobUI.Warning(why));
        return p;
    }

    TextBox PercentileField(ToleranceSettings st)
    {
        string Text() => string.Join(", ", st.Percentiles.Select(MeritFormat.Short));
        var box = new TextBox { Width = 110, MinWidth = 0, MinHeight = 28, FontSize = 12.5, Text = Text() };
        void Commit()
        {
            if (box.Text == Text()) return;
            var values = box.Text.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries).Select(LensValueParser.Number).ToList();
            if (values.Any(v => v is null) || ToleranceSettings.ValidPercentiles(values.Select(v => v!.Value)) is not { } ok) { box.Text = Text(); return; }
            Edit("Change Levels", x => x.Percentiles = ok);
        }
        box.LostFocus += (_, _) => Commit();
        box.KeyDown += (_, e) => { if (e.Key == VirtualKey.Enter) { Commit(); e.Handled = true; } };
        AutomationProperties.SetName(box, "Levels in percent");
        return box;
    }

    static string Duration(double seconds) =>
        seconds < 1 ? (seconds * 1000).ToString("F0", C) + " ms" : seconds < 120 ? seconds.ToString(seconds < 10 ? "F1" : "F0", C) + " s" : (seconds / 60).ToString("F0", C) + " min";

    void DeleteSelected()
    {
        if (_selection.Count == 0) return;
        var ids = _selection.ToHashSet();
        Edit(ids.Count == 1 ? "Delete Tolerance" : "Delete Tolerances", x => x.Delete(ids));
        _selection.Clear();
    }

    /// <summary>The rows grouped by element (<c>L1 · N-SK16</c>, <c>Air after L1</c>, <c>Stop S6</c>).</summary>
    UIElement Table(ToleranceSettings st, OpticalSystem sys, Dictionary<Guid, ToleranceIssue> issues)
    {
        var groups = LensElements.Groups(sys);
        var list = new StackPanel();
        string? current = null;
        var k = 0;
        foreach (var row in st.Rows)
        {
            var title = LensElements.GroupTitle(row, sys, groups);
            if (title != current)
            {
                current = title;
                var head = JobUI.Text(title, "ColumnHeaderText");
                head.Margin = new Thickness(8, list.Children.Count == 0 ? 6 : 10, 0, 2);
                list.Children.Add(head);
            }
            list.Children.Add(Row(row, sys, issues.GetValueOrDefault(row.Id), k++ % 2 == 1));
        }
        var frame = new Border { Child = list, CornerRadius = new CornerRadius(6), BorderThickness = new Thickness(1), Padding = new Thickness(0, 0, 0, 4) };
        JobUI.Bind(frame, "CardStrokeColorDefaultBrush", b => frame.BorderBrush = b);
        JobUI.Bind(frame, "ContentSurfaceBrush", b => frame.Background = b);
        frame.KeyDown += (_, e) => { if (e.Key == VirtualKey.Delete && e.OriginalSource is not TextBox) { DeleteSelected(); e.Handled = true; } };
        return frame;
    }

    UIElement Row(ToleranceRow row, OpticalSystem sys, ToleranceIssue? issue, bool alternate)
    {
        var id = row.Id;
        var selected = _selection.Contains(id);
        var g = new Grid { ColumnSpacing = 4, Padding = new Thickness(8, 2, 6, 2) };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(40) });
        var kind = JobUI.Text(row.Kind.ShortTitle());
        var surface = JobUI.Text(row.Kind == ToleranceKind.Other ? "" : row.SurfaceLabel(sys), "SecondaryNumberText");
        var name = JobUI.Row(6, kind, surface);
        name.VerticalAlignment = VerticalAlignment.Center;
        ToolTipService.SetToolTip(name, row.Label(sys) + " · " + row.Distribution.Title);
        g.Children.Add(name);
        void SetLimit(bool plus, double? v)
        {
            var x = Math.Abs(v ?? 0);
            Edit("Change Tolerance", s =>
            {
                if (s.Rows.FirstOrDefault(r => r.Id == id) is not { } r) return;
                if (r.Minus == r.Plus) { r.Minus = x; r.Plus = x; }
                else if (plus) r.Plus = x; else r.Minus = x;
            });
        }
        FrameworkElement limits = row.Minus == row.Plus
            ? JobUI.Row(2, JobUI.Text("±", "SecondaryText"), new NumberField($"lim-{id}", () => row.Plus, v => SetLimit(true, v), 62, "0", name: $"{row.Label(sys)} limit"))
            : JobUI.Row(2, new NumberField($"lim-{id}-m", () => -row.Minus, v => SetLimit(false, v), 58, "0", name: $"{row.Label(sys)} minus"),
                new NumberField($"lim-{id}-p", () => row.Plus, v => SetLimit(true, v), 58, "0", name: $"{row.Label(sys)} plus"));
        Grid.SetColumn(limits, 1);
        g.Children.Add(limits);
        var units = row.Kind.Units();
        FrameworkElement unit;
        if (units.Count > 1)
        {
            var menu = new MenuFlyout();
            foreach (var u in units)
            {
                var item = new RadioMenuFlyoutItem { Text = u.Title(), IsChecked = u == row.Unit, GroupName = "unit" };
                item.Click += (_, _) => ConvertUnit(id, u);
                menu.Items.Add(item);
            }
            var b = new Button { Content = row.Unit.Label(), Flyout = menu, Padding = new Thickness(4, 1, 4, 2), MinWidth = 0, MinHeight = 0, FontSize = 12, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0) };
            ToolTipService.SetToolTip(b, "The unit of the limits (changing it converts them)");
            unit = b;
        }
        else unit = JobUI.Text(row.Unit.Label(), "CaptionText");
        unit.VerticalAlignment = VerticalAlignment.Center;
        Grid.SetColumn(unit, 2);
        g.Children.Add(unit);
        var content = new StackPanel { Children = { g } };
        if (issue is not null)
        {
            var t = JobUI.Text(issue.Text, "CaptionText");
            t.TextWrapping = TextWrapping.Wrap;
            t.TextTrimming = TextTrimming.None;
            t.Margin = new Thickness(8, 0, 8, 3);
            JobUI.Bind(t, "ErrorBrush", b => t.Foreground = b);
            content.Children.Add(t);
        }
        var border = new Border { Child = content };
        void Paint()
        {
            if (selected)
            {
                var accent = (global::Windows.UI.Color)Support.ThemeLookup.Resource(border.ActualTheme, "SystemAccentColor");
                border.Background = new SolidColorBrush(global::Windows.UI.Color.FromArgb(border.ActualTheme == ElementTheme.Dark ? (byte)56 : (byte)36, accent.R, accent.G, accent.B));
            }
            else border.Background = alternate ? Support.ThemeLookup.Brush(border, "RowStripeBrush") : new SolidColorBrush(Microsoft.UI.Colors.Transparent);
        }
        border.Loaded += (_, _) => Paint();
        border.ActualThemeChanged += (_, _) => Paint();
        border.Tapped += (_, e) =>
        {
            if (e.OriginalSource is TextBox) return;
            var ctrl = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Control).HasFlag(global::Windows.UI.Core.CoreVirtualKeyStates.Down);
            if (!ctrl) _selection.Clear();
            if (!_selection.Remove(id)) _selection.Add(id);
            Queue();
        };
        border.ContextFlyout = RowMenu(row);
        AutomationProperties.SetName(border, $"{row.Label(sys)}, {row.LimitsText}");
        return border;
    }

    void ConvertUnit(Guid id, ToleranceUnit unit)
    {
        var ctx = _s.Context();
        var row = _s.Settings.Rows.FirstOrDefault(r => r.Id == id);
        if (row?.ConvertedTo(unit, ctx) is not { } converted)
        {
            _session.ShowToast("This unit needs the surface's clear aperture or the glass's Abbe number, which the design does not give yet.");
            return;
        }
        Edit("Change Tolerance Unit", s => { var i = s.Rows.FindIndex(r => r.Id == id); if (i >= 0) { converted.Id = id; s.Rows[i] = converted; } });
    }

    MenuFlyout RowMenu(ToleranceRow row)
    {
        var id = row.Id;
        var menu = new MenuFlyout();
        var dist = new MenuFlyoutSubItem { Text = "Distribution" };
        foreach (var d in ToleranceDistribution.Choices)
        {
            var item = new RadioMenuFlyoutItem { Text = d.Title, IsChecked = d == row.Distribution, GroupName = "dist" };
            item.Click += (_, _) => Edit("Change Distribution", s => { if (s.Rows.FirstOrDefault(r => r.Id == id) is { } r) r.Distribution = d; });
            dist.Items.Add(item);
        }
        menu.Items.Add(dist);
        var symmetric = new ToggleMenuFlyoutItem { Text = "Symmetric Limits", IsChecked = row.Minus == row.Plus };
        symmetric.Click += (_, _) => Edit(row.Minus == row.Plus ? "Asymmetric Limits" : "Symmetric Limits", s =>
        {
            if (s.Rows.FirstOrDefault(r => r.Id == id) is not { } r) return;
            if (r.Minus == r.Plus) r.Minus = r.Plus * 0.5 is var h && h > 0 ? h : r.Plus;
            else r.Minus = r.Plus = Math.Max(r.Minus, r.Plus);
        });
        menu.Items.Add(symmetric);
        menu.Items.Add(new MenuFlyoutSeparator());
        var dup = new MenuFlyoutItem { Text = "Duplicate" };
        dup.Click += (_, _) => Edit("Duplicate Tolerance", s => s.Duplicate(new HashSet<Guid> { id }));
        menu.Items.Add(dup);
        var del = new MenuFlyoutItem { Text = "Delete" };
        del.Click += (_, _) => { Edit("Delete Tolerance", s => s.Delete(new HashSet<Guid> { id })); _selection.Remove(id); };
        menu.Items.Add(del);
        return menu;
    }

    MenuFlyout AddMenu()
    {
        var sys = _session.Document.System;
        var menu = new MenuFlyout();
        var lens = LensElements.LensSurfaces(sys);
        foreach (var kind in ToleranceKinds.Addable)
        {
            var sub = new MenuFlyoutSubItem { Text = kind.Title() };
            foreach (var i in lens)
            {
                var sid = sys.Surfaces[i].Id;
                var item = new MenuFlyoutItem { Text = $"Surface {i}" + (sys.Surfaces[i].Label.Length > 0 ? $" · {sys.Surfaces[i].Label}" : "") };
                item.Click += (_, _) =>
                {
                    List<Guid> added = [];
                    var amount = kind switch
                    {
                        ToleranceKind.Radius => 3, ToleranceKind.Thickness => 0.05, ToleranceKind.DecenterX or ToleranceKind.DecenterY => 0.025,
                        ToleranceKind.Index => 0.0005, ToleranceKind.Abbe => 0.5, ToleranceKind.TiltX or ToleranceKind.TiltY => 1, _ => 0.5,
                    };
                    Edit("Add Tolerance", s => added = s.Add([new ToleranceRow(kind, sid, amount)], _selection.FirstOrDefault() is var f && f != Guid.Empty ? f : null));
                    _selection.Clear();
                    _selection.UnionWith(added);
                };
                sub.Items.Add(item);
            }
            menu.Items.Add(sub);
        }
        menu.Items.Add(new MenuFlyoutSeparator());
        var groups = LensElements.Groups(sys);
        foreach (var kind in ToleranceKinds.AddableElement)
        {
            var sub = new MenuFlyoutSubItem { Text = kind.Title() };
            foreach (var grp in groups)
            {
                var first = sys.Surfaces[grp.Surfaces[0]].Id;
                var last = sys.Surfaces[grp.Surfaces[^1]].Id;
                var item = new MenuFlyoutItem { Text = grp.Title };
                item.Click += (_, _) =>
                {
                    var amount = kind is ToleranceKind.ElementDecenterX or ToleranceKind.ElementDecenterY ? 0.03 : 1;
                    Edit("Add Tolerance", s => s.Add([new ToleranceRow(kind, first, amount, last: last)]));
                };
                sub.Items.Add(item);
            }
            menu.Items.Add(sub);
        }
        return menu;
    }

    MenuFlyout CompensatorMenu(OpticalSystem sys)
    {
        var menu = new MenuFlyout();
        var vars = DesignVariables.Of(_session.Document).Select(v => v.Param);
        foreach (var (group, items) in ToleranceCompensator.Candidates(sys, vars))
        {
            var sub = new MenuFlyoutSubItem { Text = group };
            foreach (var param in items)
            {
                var item = new MenuFlyoutItem { Text = ToleranceCompensator.IsBackFocus(param, sys) ? "Back focus" : param.Label(sys) };
                var p = param;
                item.Click += (_, _) => Edit("Add Compensator", s =>
                {
                    if (s.Compensators.Any(c => sys.Canonical(c.Param) == sys.Canonical(p))) return;
                    s.Compensators.Add(new ToleranceCompensator(p, p is ParamPath.Thickness ? 0.5 : null));
                });
                sub.Items.Add(item);
            }
            menu.Items.Add(sub);
        }
        return menu;
    }

    /// <summary>Default Tolerances: a class of manufacture, every value editable (0 leaves a kind out), then Generate.</summary>
    Flyout GeneratorFlyout()
    {
        var flyout = new Flyout();
        var gen = _s.Settings.Generator.Clone();
        var body = new StackPanel { Spacing = 6, Width = 300 };
        void Fill()
        {
            body.Children.Clear();
            body.Children.Add(JobUI.Text("Default Tolerances", "PanelTitleText"));
            var presets = ToleranceGeneratorPresets.All;
            body.Children.Add(JobUI.Labeled("Class", JobUI.Choice(presets.Select(x => x.Title()).ToList(), gen.Preset is { } pr ? presets.ToList().IndexOf(pr) : -1,
                k => { gen.Apply(presets[k]); Fill(); }, 140)));
            (string Title, string Unit, Func<double> Get, Action<double> Set)[] values =
            [
                ("Power", "fr", () => gen.RadiusFringes, v => gen.RadiusFringes = v),
                ("Irregularity", "fr", () => gen.IrregularityFringes, v => gen.IrregularityFringes = v),
                ("Thickness", "mm", () => gen.ThicknessMM, v => gen.ThicknessMM = v),
                ("Surface decenter", "mm", () => gen.DecentreMM, v => gen.DecentreMM = v),
                ("Surface tilt", "′", () => gen.TiltArcmin, v => gen.TiltArcmin = v),
                ("Element decenter", "mm", () => gen.ElementDecentreMM, v => gen.ElementDecentreMM = v),
                ("Element tilt", "′", () => gen.ElementTiltArcmin, v => gen.ElementTiltArcmin = v),
                ("Index", "Δn", () => gen.Index, v => gen.Index = v),
                ("Abbe number", "%", () => gen.AbbePercent, v => gen.AbbePercent = v),
            ];
            foreach (var (title, unit, get, set) in values)
                body.Children.Add(JobUI.Labeled(title, JobUI.Row(5, new NumberField($"gen-{title}", () => get(), v => { set(Math.Max(0, v ?? 0)); gen.Preset = null; }, 70, "0", name: title),
                    new TextBlock { Text = unit, Width = 22, Style = JobUI.Style("CaptionText") })));
            body.Children.Add(JobUI.Hint("0 leaves a kind out. Generating replaces the tolerances; the compensators stay.", 11.5));
            var go = JobUI.Primary("Generate", null, () =>
            {
                flyout.Hide();
                Edit("Default Tolerances", s => { s.Generator = gen.Clone(); s.Generate(_session.Document.System); });
                _selection.Clear();
            });
            go.HorizontalAlignment = HorizontalAlignment.Stretch;
            body.Children.Add(go);
        }
        flyout.Opening += (_, _) => { gen = _s.Settings.Generator.Clone(); Fill(); };
        flyout.Content = body;
        return flyout;
    }

    // ---- Running ----

    UIElement Running()
    {
        var p = new StackPanel { Spacing = 14 };
        var timing = _s.Timing();
        var elapsed = _s.StartedAt is { } s ? (DateTime.UtcNow - s).TotalSeconds : 0;
        p.Children.Add(JobUI.Row(24, JobUI.BigNumber("Elapsed", Duration(elapsed)),
            JobUI.BigNumber("Time left", timing.SecondsLeft is { } left ? "≈ " + Duration(left) : "—")));
        if (timing.TrialsDone is { } done) p.Children.Add(JobUI.BigNumber("Trials", $"{done.ToString(C)} / {_s.Settings.Trials.ToString(C)}"));
        if (timing.SecondsPerTrial is { } spt) p.Children.Add(JobUI.Hint($"{Duration(spt)} per trial", 12));
        p.Children.Add(new ProgressBar { Value = _s.Progress.Fraction * 100, Maximum = 100, IsIndeterminate = _s.Progress.Fraction <= 0 });
        p.Children.Add(JobUI.Text(_s.Progress.Stage.Length > 0 ? _s.Progress.Stage : "Running", "SecondaryText"));
        p.Children.Add(JobUI.Hint("Stop keeps the sensitivities and the trials completed. Cancel discards the run. You can close this panel; the job continues."));
        return p;
    }

    // ---- Results ----

    UIElement Results()
    {
        var p = new StackPanel { Spacing = 14 };
        if (_s.Result is not { } r) return p;
        var d = r.Display;
        var sys = _s.RunSystem ?? _session.Document.System;
        if (r.Stopped) p.Children.Add(JobUI.Hint("Stopped early: the results of what completed."));
        var big = JobUI.Row(24);
        if (r.HasNominal) big.Children.Add(JobUI.BigNumber("Nominal", d.FormatWithUnit(d.Value(r.Nominal))));
        if (r.HasSensitivity && double.IsFinite(r.Predicted)) big.Children.Add(JobUI.BigNumber("RSS prediction", d.FormatWithUnit(d.Value(r.Predicted))));
        p.Children.Add(big);
        if (r.HasMonteCarlo)
        {
            var levels = new StackPanel { Spacing = 2 };
            foreach (var pc in r.Percentiles)
            {
                var row = new Grid();
                row.Children.Add(JobUI.Text($"{MeritFormat.Short(pc.Level)} % of builds", "SecondaryText"));
                var v = JobUI.Text((d.HigherIsBetter() ? "≥ " : "≤ ") + d.FormatWithUnit(d.Value(pc.Value)), "NumberText");
                v.HorizontalAlignment = HorizontalAlignment.Right;
                row.Children.Add(v);
                levels.Children.Add(row);
            }
            if (r.Yield is { } y)
            {
                var row = new Grid();
                row.Children.Add(JobUI.Text($"Yield (pass limit {(r.Threshold is { } th ? d.FormatWithUnit(d.Value(th)) : "—")})", "SecondaryText"));
                var v = JobUI.Text(MeritFormat.Percent(y), "NumberText");
                v.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
                v.HorizontalAlignment = HorizontalAlignment.Right;
                row.Children.Add(v);
                levels.Children.Add(row);
            }
            levels.Children.Add(JobUI.Hint($"{r.Trials.ToString(C)} trials" + (r.Failures > 0 ? $", {r.Failures.ToString(C)} failed" : "") + $", seed {r.Seed.ToString(C)}"
                + (r.SecondsPerTrial is { } spt ? $" · {Duration(spt)} per trial" : ""), 11.5));
            p.Children.Add(JobUI.Card(levels, 8));
        }
        if (_s.Previewed is not null) p.Children.Add(PreviewCard(r, d));
        if (r.HasSensitivity && r.Sensitivities.Count > 0) p.Children.Add(Sensitivities(r, d, sys));
        if (r.HasMonteCarlo && r.Criteria.Count > 0) p.Children.Add(Histogram(r, d));
        var copy = JobUI.Secondary("Copy Summary", "\uE8C8", () =>
        {
            var package = new DataPackage();
            package.SetText(ToleranceReport.Summary(r, _s.ResultRows, sys, _session.Document.System.Name));
            Clipboard.SetContent(package);
            _session.ShowToast("The tolerance summary is on the clipboard.");
        }, "Copy a text summary of the results");
        copy.HorizontalAlignment = HorizontalAlignment.Left;
        p.Children.Add(copy);
        return p;
    }

    UIElement PreviewCard(ViewModels.Jobs.ToleranceResult r, CriterionDisplay d)
    {
        var p = new StackPanel { Spacing = 4 };
        var head = new Grid();
        var title = JobUI.Text(_s.Previewed!.Title(r, _s.RunSystem, _s.ResultRows), "PanelTitleText");
        JobUI.Bind(title, "AccentTextFillColorPrimaryBrush", b => title.Foreground = b);
        head.Children.Add(title);
        var end = new HyperlinkButton { Content = "End Preview", Padding = new Thickness(0), FontSize = 12, HorizontalAlignment = HorizontalAlignment.Right };
        end.Click += (_, _) => _s.EndPreview();
        head.Children.Add(end);
        p.Children.Add(head);
        if (_s.PreviewError is { } err) p.Children.Add(JobUI.Warning(err));
        else if (_s.Preview is not { } pv) p.Children.Add(JobUI.Hint("Rebuilding the build in the engine…"));
        else
        {
            p.Children.Add(JobUI.Text($"Criterion {d.FormatWithUnit(d.Value(pv.Criterion))}" + (r.HasNominal ? $" (nominal {d.FormatWithUnit(d.Value(r.Nominal))})" : ""), "NumberText"));
            for (var j = 0; j < Math.Min(pv.Compensators.Count, r.Compensators.Count); j++)
            {
                var nominal = j < r.NominalCompensators.Count ? r.NominalCompensators[j] : double.NaN;
                p.Children.Add(JobUI.Text($"{r.Compensators[j].Param.Label(_s.RunSystem ?? _session.Document.System)}: {MeritFormat.Short(pv.Compensators[j])} ({MeritFormat.Signed(pv.Compensators[j] - nominal)})", "CaptionText"));
            }
            var changes = ToleranceBuildChange.Changes(pv.Perturbations, ToleranceBuildChange.RunLimits(_s.RunSettings), _s.ResultRows, _s.RunSystem ?? _session.Document.System);
            foreach (var c in changes.OrderByDescending(c => Math.Abs(c.Fraction)).Take(8))
            {
                var row = new Grid();
                row.Children.Add(JobUI.Text(c.Label, "CaptionText"));
                var v = JobUI.Text(c.Text, "CaptionText");
                v.HorizontalAlignment = HorizontalAlignment.Right;
                row.Children.Add(v);
                p.Children.Add(row);
            }
        }
        var card = new Border { Child = p, Padding = new Thickness(10), CornerRadius = new CornerRadius(8), BorderThickness = new Thickness(1) };
        JobUI.Bind(card, "AccentFillColorDefaultBrush", b => card.BorderBrush = b);
        JobUI.Bind(card, "SubtleFillColorSecondaryBrush", b => card.Background = b);
        return card;
    }

    /// <summary>The sensitivities by share of the RSS, with a bar; the worst offenders tinted; an eye previews the build at its worse limit.</summary>
    UIElement Sensitivities(ViewModels.Jobs.ToleranceResult r, CriterionDisplay d, OpticalSystem sys)
    {
        var p = new StackPanel { Spacing = 0 };
        var worst = r.WorstOffenders;
        foreach (var s in r.Sensitivities.OrderByDescending(r.Share).Take(25))
        {
            var share = r.Share(s);
            var g = new Grid { ColumnSpacing = 6, Padding = new Thickness(6, 3, 2, 3) };
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(70) });
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var label = s.Id >= 0 && s.Id < _s.ResultRows.Count ? _s.ResultRows[s.Id].Label(sys) : s.Label;
            var limits = s.Id >= 0 && s.Id < _s.ResultRows.Count ? _s.ResultRows[s.Id].LimitText(s.WorsePlus) : "";
            var name = JobUI.Text(label);
            if (worst.Contains(s.Id)) JobUI.Bind(name, "WarningBrush", b => name.Foreground = b);
            var texts = JobUI.Column(0, name, JobUI.Text($"{limits} → {d.FormatChange(d.Value(s.WorsePlus ? s.CriterionPlus : s.CriterionMinus) - d.Value(r.Nominal))}{(d.Unit().Length > 0 ? " " + d.Unit() : "")}", "CaptionText"));
            g.Children.Add(texts);
            var bar = new Grid { Height = 5, VerticalAlignment = VerticalAlignment.Center };
            var back = new Border { CornerRadius = new CornerRadius(2.5) };
            JobUI.Bind(back, "SubtleFillColorTertiaryBrush", b => back.Background = b);
            var fill = new Border { CornerRadius = new CornerRadius(2.5), HorizontalAlignment = HorizontalAlignment.Left, Width = Math.Max(share > 0 ? 2 : 0, 70 * Math.Min(share, 1)) };
            JobUI.Bind(fill, worst.Contains(s.Id) ? "WarningBrush" : "AccentFillColorDefaultBrush", b => fill.Background = b);
            bar.Children.Add(back);
            bar.Children.Add(fill);
            var right = JobUI.Column(1, bar, JobUI.Text(MeritFormat.Share(share), "CaptionText", 10.5));
            ((FrameworkElement)right.Children[1]).HorizontalAlignment = HorizontalAlignment.Right;
            Grid.SetColumn(right, 1);
            g.Children.Add(right);
            var c = new TolerancePreviewCase.Sensitivity(s.Id, s.WorsePlus);
            var on = Equals(_s.Previewed, c);
            var eye = JobUI.Icon(on ? "\uED1A" : "\uE7B3", on ? "End the preview" : "Preview this build at its worse limit in the window",
                () => _ = _s.PreviewAsync(on ? null : c), true, 26);
            if (on) JobUI.Bind(eye, "AccentTextFillColorPrimaryBrush", b => eye.Foreground = b);
            Grid.SetColumn(eye, 2);
            g.Children.Add(eye);
            AutomationProperties.SetName(g, $"{label}: {MeritFormat.Share(share)} of the RSS");
            p.Children.Add(g);
        }
        var more = r.Sensitivities.Count > 25 ? $" (the largest 25 of {r.Sensitivities.Count.ToString(C)})" : "";
        return JobUI.Group("Sensitivity" + more, p, JobUI.Text("share of RSS", "CaptionText"));
    }

    /// <summary>The Monte-Carlo histogram with the nominal and the levels marked, and the worst trials (a click previews one).</summary>
    UIElement Histogram(ViewModels.Jobs.ToleranceResult r, CriterionDisplay d)
    {
        var values = r.Criteria.Select(x => d.Value(x)).Where(double.IsFinite).ToList();
        if (values.Count == 0) return new StackPanel();
        var p = new StackPanel { Spacing = 6 };
        var canvas = new Canvas { Height = 120 };
        var lo = values.Min();
        var hi = values.Max();
        if (r.HasNominal) { lo = Math.Min(lo, d.Value(r.Nominal)); hi = Math.Max(hi, d.Value(r.Nominal)); }
        if (hi - lo < 1e-12) hi = lo + 1;
        const int Bins = 24;
        var counts = new int[Bins];
        foreach (var v in values) counts[Math.Clamp((int)((v - lo) / (hi - lo) * Bins), 0, Bins - 1)]++;
        var max = Math.Max(1, counts.Max());
        void Draw()
        {
            canvas.Children.Clear();
            var w = canvas.ActualWidth;
            if (w <= 0) return;
            var h = canvas.Height;
            var accent = Support.ThemeLookup.Brush(canvas, "AccentFillColorDefaultBrush");
            for (var i = 0; i < Bins; i++)
            {
                var bh = counts[i] / (double)max * (h - 4);
                var rect = new Rectangle { Width = Math.Max(1, w / Bins - 2), Height = bh, Fill = accent, RadiusX = 1.5, RadiusY = 1.5, Opacity = 0.85 };
                Canvas.SetLeft(rect, i * w / Bins + 1);
                Canvas.SetTop(rect, h - bh);
                canvas.Children.Add(rect);
            }
            void Mark(double v, string key, double opacity)
            {
                var x = (v - lo) / (hi - lo) * w;
                canvas.Children.Add(new Line { X1 = x, X2 = x, Y1 = 0, Y2 = h, Stroke = Support.ThemeLookup.Brush(canvas, key), StrokeThickness = 1.5, StrokeDashArray = [3, 2], Opacity = opacity });
            }
            if (r.HasNominal) Mark(d.Value(r.Nominal), "TextFillColorPrimaryBrush", 0.8);
            foreach (var pc in r.Percentiles) Mark(d.Value(pc.Value), "WarningBrush", 0.9);
            var axis = new Line { X1 = 0, X2 = w, Y1 = h, Y2 = h, Stroke = Support.ThemeLookup.Brush(canvas, "DividerBrush"), StrokeThickness = 1 };
            canvas.Children.Add(axis);
        }
        canvas.SizeChanged += (_, _) => Draw();
        canvas.ActualThemeChanged += (_, _) => Draw();
        p.Children.Add(canvas);
        var scale = new Grid();
        scale.Children.Add(JobUI.Text(d.Format(lo), "CaptionText"));
        var right = JobUI.Text(d.FormatWithUnit(hi), "CaptionText");
        right.HorizontalAlignment = HorizontalAlignment.Right;
        scale.Children.Add(right);
        p.Children.Add(scale);
        p.Children.Add(JobUI.Hint("Dashed: the nominal (gray) and the levels (orange).", 11));
        var trials = r.TrialsWorstFirst.Where(k => r.Record(k) is not null).Take(8).ToList();
        if (trials.Count > 0)
        {
            p.Children.Add(JobUI.Text("Worst trials", "CaptionText"));
            foreach (var k in trials)
            {
                var c = new TolerancePreviewCase.Trial(k);
                var on = Equals(_s.Previewed, c);
                var b = new Button
                {
                    HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch, Padding = new Thickness(6, 2, 6, 2),
                    Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0),
                };
                var g = new Grid();
                var label = JobUI.Text($"Trial {(k + 1).ToString(C)}", on ? "PanelTitleText" : "BodyText");
                if (on) JobUI.Bind(label, "AccentTextFillColorPrimaryBrush", x => label.Foreground = x);
                g.Children.Add(label);
                var v = JobUI.Text(d.FormatWithUnit(d.Value(r.TrialCriteria[k])), "NumberText");
                v.HorizontalAlignment = HorizontalAlignment.Right;
                g.Children.Add(v);
                b.Content = g;
                ToolTipService.SetToolTip(b, on ? "End the preview" : "Preview this build in the window");
                b.Click += (_, _) => _ = _s.PreviewAsync(on ? null : c);
                p.Children.Add(b);
            }
        }
        return JobUI.Group("Monte Carlo", p, JobUI.Text($"{r.Trials.ToString(C)} builds", "CaptionText"));
    }

    // ---- Actions ----

    void Actions()
    {
        var problem = _s.RunProblem();
        switch (_s.Phase)
        {
            case JobPhase.Setup:
                _frame.SetActions(null, JobUI.Primary("Run Tolerancing", "\uE768", () => _s.Start(), problem ?? "Start the run (Ctrl+Enter)", problem is null));
                break;
            case JobPhase.Failed:
                _frame.SetActions(null, JobUI.Secondary("Back", null, _s.EditSettings, "Back to the tolerances"),
                    JobUI.Primary("Try Again", "\uE72C", () => _s.Start(fresh: true), "Run the job again with the same settings", problem is null));
                break;
            case JobPhase.Running:
                _frame.SetActions(JobUI.Progress(_s.Progress),
                    JobUI.Secondary("Cancel", null, _s.Cancel, "End the run and discard it", destructive: true),
                    JobUI.Secondary("Stop", "\uE71A", _s.Stop, "End now and keep the sensitivities and the trials completed", _s.CanStop));
                break;
            default:
                _frame.SetActions(null, JobUI.Primary("Edit Tolerances", null, _s.EditSettings, "Back to the tolerance set (the results are kept)"));
                break;
        }
    }
}
