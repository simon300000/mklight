using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Shapes;
using MKLight.App.Support;
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
/// time left; then the results: nominal, RSS prediction, levels and yield, the Export menu (Copy Summary, the tables
/// as CSV), the sensitivities by share (an eye previews a build at its worse limit), the Monte-Carlo histogram (a
/// click on a bar lists its trials, else the worst; a click on a trial previews it) and each compensator's range used
/// against the range allowed. The table's groups collapse; an element row shows its moving surfaces and pivot.
/// </summary>
public sealed partial class TolerancePanel : UserControl
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;
    readonly ToleranceSession _s;
    readonly DesignSession _session;
    readonly JobPanelFrame _frame = new(Loc.S("Jobs_Tolerance_Panel_Title"), "\uE9E9");
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
        var designName = _session.Document.System.Name.Length > 0 ? _session.Document.System.Name : Loc.S("Jobs_Tolerance_Panel_TheDesign");
        _frame.Subtitle = n == 1 ? Loc.F("Jobs_Tolerance_Panel_Subtitle_One", n, designName) : Loc.F("Jobs_Tolerance_Panel_Subtitle_Many", n, designName);
        if (_s.IsRunning && !_clock.IsEnabled) _clock.Start();
        _frame.Body = _s.Phase switch
        {
            JobPhase.Running => Running(),
            JobPhase.Finished => Results(),
            JobPhase.Failed => JobUI.Failure(Loc.S("Jobs_Tolerance_Panel_FailedTitle"), EngineText.T(_s.Failure ?? "")),
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
            var last = new HyperlinkButton { Content = Loc.S("Jobs_Tolerance_Panel_ShowLastResults"), Padding = new Thickness(0, 2, 0, 2), FontSize = 12.5 };
            last.Click += (_, _) => _s.ShowResult();
            p.Children.Add(last);
        }

        // Tolerances.
        var t = new StackPanel { Spacing = 6 };
        var defaults = JobUI.Secondary(Loc.S("Jobs_Tolerance_Panel_DefaultTolerancesButton"), "\uE8FD", () => { }, Loc.S("Jobs_Tolerance_Panel_DefaultTolerancesHelp"));
        defaults.Flyout = GeneratorFlyout();
        var add = new DropDownButton { Content = new FontIcon { Glyph = "\uE710", FontSize = 12 }, Flyout = AddMenu(), MinHeight = 32, CornerRadius = new CornerRadius(16) };
        ToolTipService.SetToolTip(add, Loc.S("Jobs_Tolerance_Panel_AddTolerance"));
        AutomationProperties.SetName(add, Loc.S("Jobs_Tolerance_Panel_AddTolerance"));
        var tools = JobUI.Row(4, defaults, add,
            JobUI.Icon("\uE8C8", Loc.S("Jobs_Tolerance_Panel_DuplicateSelected"), () =>
            {
                var ids = _selection.ToHashSet();
                List<Guid> copies = [];
                Edit(Loc.S("Jobs_Tolerance_Panel_Undo_DuplicateTolerance"), x => copies = x.Duplicate(ids));
                _selection.Clear();
                _selection.UnionWith(copies);
            }, _selection.Count > 0, 30),
            JobUI.Icon("\uE738", Loc.S("Jobs_Tolerance_Panel_DeleteSelected"), DeleteSelected, _selection.Count > 0, 30));
        t.Children.Add(tools);
        t.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_TestFringesAt"), JobUI.Row(5, new NumberField("test-wl", () => st.TestWavelengthUm * 1000,
            v => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeTestWavelength"), x => x.TestWavelengthUm = v is > 0 ? v.Value / 1000 : ToleranceSettings.DefaultTestWavelengthUm), 70, "632.8", name: Loc.S("Jobs_Tolerance_Panel_TestWavelengthName")),
            JobUI.Text("nm", "SecondaryText")), Loc.S("Jobs_Tolerance_Panel_TestFringesHelp")));
        if (st.Rows.Count == 0)
            t.Children.Add(JobUI.Hint(Loc.S("Jobs_Tolerance_Panel_NoTolerances")));
        else t.Children.Add(Table(st, sys, issues));
        p.Children.Add(JobUI.Group(Loc.S("Jobs_Tolerance_Panel_TolerancesGroup"), t, JobUI.Text($"{st.Rows.Count}", "CaptionText")));

        // Compensators.
        var c = new StackPanel { Spacing = 4 };
        if (st.Compensators.Count == 0) c.Children.Add(JobUI.Hint(Loc.S("Jobs_Tolerance_Panel_NoCompensators")));
        foreach (var comp in st.Compensators)
        {
            var row = new Grid { ColumnSpacing = 6 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.Children.Add(JobUI.Text(comp.Title(sys)));
            var id = comp.Id;
            var range = JobUI.Row(4, JobUI.Text("±", "SecondaryText"), new NumberField($"comp-{id}", () => comp.Range,
                v => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeCompensatorRange"), x => { if (x.Compensators.FirstOrDefault(k => k.Id == id) is { } k) k.Range = v is > 0 ? v : null; }), 64, Loc.S("Jobs_Tolerance_Panel_RangeFree"),
                name: Loc.F("Jobs_Tolerance_Panel_RangeOf", comp.Title(sys))), JobUI.Text(comp.Param.Unit(sys), "CaptionText"));
            Grid.SetColumn(range, 1);
            row.Children.Add(range);
            var remove = JobUI.Icon("\uE711", Loc.S("Jobs_Tolerance_Panel_RemoveCompensatorHelp"), () => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_RemoveCompensator"), x => x.Compensators.RemoveAll(k => k.Id == id)), true, 24);
            Grid.SetColumn(remove, 2);
            row.Children.Add(remove);
            c.Children.Add(row);
        }
        var addComp = new DropDownButton { Content = Loc.S("Jobs_Tolerance_Panel_AddCompensator"), Flyout = CompensatorMenu(sys), FontSize = 12.5, MinHeight = 28 };
        c.Children.Add(addComp);
        p.Children.Add(JobUI.Group(Loc.S("Jobs_Tolerance_Panel_CompensatorsGroup"), c));

        // Criterion.
        var merit = _s.Merit;
        var cr = new StackPanel { Spacing = 6 };
        var kinds = ToleranceCriterionKinds.All;
        cr.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_Criterion"), JobUI.Choice(kinds.Select(k => k.Title()).ToList(), kinds.ToList().IndexOf(st.Criterion.Kind),
            k => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeCriterion"), x => x.Criterion.Kind = kinds[k]), 150)));
        var display = st.Criterion.Display(merit);
        cr.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_PassLimit"), JobUI.Row(5, new NumberField("threshold", () => st.Criterion.Threshold,
            v => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangePassLimit"), x => x.Criterion.Threshold = v), 76, Loc.S("Jobs_Tolerance_Panel_PassLimitNone"), name: Loc.S("Jobs_Tolerance_Panel_PassLimit")), JobUI.Text(display.Unit(), "CaptionText")),
            display.HigherIsBetter() ? Loc.S("Jobs_Tolerance_Panel_PassLimitHelpLowest") : Loc.S("Jobs_Tolerance_Panel_PassLimitHelpHighest")));
        cr.Children.Add(JobUI.Hint(st.Criterion.Kind == ToleranceCriterionKind.DesignMerit
            ? Loc.S("Jobs_Tolerance_Panel_CriterionHintMerit")
            : st.Criterion.Kind == ToleranceCriterionKind.Mtf ? Loc.S("Jobs_Tolerance_Panel_CriterionHintMtf") : Loc.S("Jobs_Tolerance_Panel_CriterionHintFields"), 11.5));
        p.Children.Add(JobUI.Group(Loc.S("Jobs_Tolerance_Panel_Criterion"), cr));

        // Analysis.
        var a = new StackPanel { Spacing = 6 };
        var modes = ToleranceModes.All;
        var modeSeg = JobUI.Segments(modes.Select(m => m.Title()).ToList(), modes.ToList().IndexOf(st.Mode), k => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeAnalysis"), x => x.Mode = modes[k]));
        a.Children.Add(modeSeg);
        if (st.Mode != ToleranceMode.Sensitivity)
        {
            a.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_Trials"), new NumberField("trials", () => st.Trials, v => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeTrials"), x => x.Trials = Math.Max(2, (int)Math.Round(v ?? 200))), 80, "200", MeritFormat.Short, optional: false, name: Loc.S("Jobs_Tolerance_Panel_Trials"))));
            a.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_Seed"), new NumberField("tol-seed", () => st.Seed, v => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeSeed"), x => x.Seed = Math.Max(0, (long)Math.Round(v ?? 0))), 80, "0", MeritFormat.Short, optional: false, name: Loc.S("Jobs_Tolerance_Panel_Seed"))));
            a.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_Levels"), PercentileField(st), Loc.S("Jobs_Tolerance_Panel_LevelsHelp")));
            a.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_KeepTrials"), new NumberField("record", () => st.RecordTrials, v => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeKeptTrials"), x => x.RecordTrials = Math.Clamp((int)Math.Round(v ?? 200), 0, ToleranceSettings.MaxRecordTrials)), 80, "200", MeritFormat.Short, optional: false, name: Loc.S("Jobs_Tolerance_Panel_KeptTrialsName")),
                Loc.S("Jobs_Tolerance_Panel_KeepTrialsHelp")));
            if (_s.Result?.SecondsPerTrial is > 0 and { } spt)
                a.Children.Add(JobUI.Hint(Loc.F("Jobs_Tolerance_Panel_LastRunEstimate", Duration(spt), st.Trials.ToString(C), Duration(spt * st.Trials)), 11.5));
        }
        p.Children.Add(JobUI.Group(Loc.S("Jobs_Tolerance_Panel_AnalysisGroup"), a));
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
            Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeLevels"), x => x.Percentiles = ok);
        }
        box.LostFocus += (_, _) => Commit();
        box.KeyDown += (_, e) => { if (e.Key == VirtualKey.Enter) { Commit(); e.Handled = true; } };
        AutomationProperties.SetName(box, Loc.S("Jobs_Tolerance_Panel_LevelsName"));
        return box;
    }

    static string Duration(double seconds) =>
        seconds < 1 ? (seconds * 1000).ToString("F0", C) + " ms" : seconds < 120 ? seconds.ToString(seconds < 10 ? "F1" : "F0", C) + " s" : (seconds / 60).ToString("F0", C) + " min";

    void DeleteSelected()
    {
        if (_selection.Count == 0) return;
        var ids = _selection.ToHashSet();
        Edit(ids.Count == 1 ? Loc.S("Jobs_Tolerance_Panel_Undo_DeleteTolerance") : Loc.S("Jobs_Tolerance_Panel_Undo_DeleteTolerances"), x => x.Delete(ids));
        _selection.Clear();
    }

    /// <summary>
    /// The rows grouped by element (<c>L1 · N-SK16</c>, <c>Air after L1</c>, <c>Stop S6</c>), each group collapsible (a closed
    /// one shows its count, in red when a row in it cannot run).
    /// </summary>
    UIElement Table(ToleranceSettings st, OpticalSystem sys, Dictionary<Guid, ToleranceIssue> issues)
    {
        var list = new StackPanel();
        var leading = ToleranceTableGroup.LeadingElementRows(st.Rows);
        var k = 0;
        foreach (var group in ToleranceTableGroup.Of(st.Rows, sys))
        {
            var closed = _s.CollapsedGroups.Contains(group.Key);
            list.Children.Add(GroupHeader(group, closed, group.Rows.Count(r => issues.ContainsKey(r.Id)), list.Children.Count == 0));
            if (closed) continue;
            foreach (var row in group.Rows) list.Children.Add(Row(row, sys, issues.GetValueOrDefault(row.Id), k++ % 2 == 1, leading.Contains(row.Id)));
        }
        var frame = new Border { Child = list, CornerRadius = new CornerRadius(6), BorderThickness = new Thickness(1), Padding = new Thickness(0, 0, 0, 4) };
        JobUI.Bind(frame, "CardStrokeColorDefaultBrush", b => frame.BorderBrush = b);
        JobUI.Bind(frame, "ContentSurfaceBrush", b => frame.Background = b);
        frame.KeyDown += (_, e) => { if (e.Key == VirtualKey.Delete && e.OriginalSource is not TextBox) { DeleteSelected(); e.Handled = true; } };
        return frame;
    }

    /// <summary>Whether <paramref name="d"/> lies inside a button (a click there is not a row selection).</summary>
    static bool InButton(DependencyObject? d)
    {
        for (; d is not null; d = VisualTreeHelper.GetParent(d))
        {
            if (d is ButtonBase) return true;
            if (d is Border { Tag: "tolerance-row" }) return false;
        }
        return false;
    }

    /// <summary>A group's title with a disclosure chevron; closed, its count of tolerances.</summary>
    Button GroupHeader(ToleranceTableGroup group, bool closed, int problems, bool first)
    {
        var g = new Grid { ColumnSpacing = 6 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var chevron = new FontIcon { Glyph = closed ? "\uE76C" : "\uE70D", FontSize = 9, VerticalAlignment = VerticalAlignment.Center };
        JobUI.Bind(chevron, "TextFillColorSecondaryBrush", b => chevron.Foreground = b);
        g.Children.Add(chevron);
        var title = JobUI.Text(group.Title, "ColumnHeaderText");
        title.TextTrimming = TextTrimming.CharacterEllipsis;
        Grid.SetColumn(title, 1);
        g.Children.Add(title);
        if (closed)
        {
            var count = JobUI.Text(group.Rows.Count.ToString(C), "SecondaryNumberText", 11.5);
            if (problems > 0) JobUI.Bind(count, "ErrorBrush", b => count.Foreground = b);
            Grid.SetColumn(count, 2);
            g.Children.Add(count);
        }
        var b = new Button
        {
            Content = g, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch,
            Padding = new Thickness(8, 3, 8, 3), Margin = new Thickness(0, first ? 4 : 6, 0, 1), MinHeight = 0,
            Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0),
        };
        b.Click += (_, _) => _s.ToggleGroup(group.Key);
        var n = group.Rows.Count.ToString(C);
        ToolTipService.SetToolTip(b, closed ? Loc.F("Jobs_Tolerance_Panel_Group_Show", n, group.Name) : Loc.F("Jobs_Tolerance_Panel_Group_Hide", group.Name));
        AutomationProperties.SetName(b, group.Title);
        AutomationProperties.SetItemStatus(b, closed ? (problems > 0 ? Loc.F("Jobs_Tolerance_Panel_Group_CollapsedProblems", n, problems.ToString(C)) : Loc.F("Jobs_Tolerance_Panel_Group_Collapsed", n)) : Loc.S("Jobs_Tolerance_Panel_Group_Expanded"));
        AutomationProperties.SetHeadingLevel(b, Microsoft.UI.Xaml.Automation.Peers.AutomationHeadingLevel.Level3);
        return b;
    }

    /// <summary>A borderless button opening a menu (the surface column).</summary>
    static Button MenuButton(string text, MenuFlyout menu, string help)
    {
        var b = new Button
        {
            Content = JobUI.Text(text, "SecondaryNumberText"), Flyout = menu, Padding = new Thickness(4, 1, 4, 2), MinWidth = 0, MinHeight = 0,
            Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0), VerticalAlignment = VerticalAlignment.Center,
        };
        ToolTipService.SetToolTip(b, help);
        AutomationProperties.SetName(b, Loc.F("Jobs_Tolerance_Panel_MenuButtonName", text, help));
        return b;
    }

    static string SurfaceTitle(OpticalSystem sys, int i) => i == sys.Stop ? Loc.F("Jobs_Tolerance_Panel_SurfaceStop", i.ToString(C)) : $"S{i.ToString(C)}";

    static MenuFlyoutItem Item(string text, Action click, bool enabled = true)
    {
        var item = new MenuFlyoutItem { Text = text, IsEnabled = enabled };
        item.Click += (_, _) => click();
        return item;
    }

    /// <summary>
    /// The surface column's menu: a surface row's surface; an element row's element (its range), first and last surface,
    /// the pivot kept where it sits.
    /// </summary>
    MenuFlyout SurfaceMenu(ToleranceRow row, OpticalSystem sys)
    {
        var id = row.Id;
        var menu = new MenuFlyout();
        var last = Math.Max(sys.ImageIndex - 1, 1);
        if (!row.Kind.IsElement())
        {
            for (var i = 1; i <= last && i < sys.Surfaces.Count; i++)
            {
                var sid = sys.Surfaces[i].Id;
                menu.Items.Add(Item(SurfaceTitle(sys, i), () => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeToleranceSurface"), s => { if (s.Rows.FirstOrDefault(r => r.Id == id) is { } r) r.Surface = sid; }), sid != row.Surface));
            }
            return menu;
        }
        var moving = row.MovingSurfaces(sys);
        var pivot = TolerancePivotChoice.Of(row, sys);
        foreach (var g in LensElements.Groups(sys))
        {
            int f = g.Surfaces[0], l = g.Surfaces[^1];
            menu.Items.Add(Item($"{g.Title} · S{f.ToString(C)}–{l.ToString(C)}", () => SetElement(id, f, l, TolerancePivotChoice.First),
                !(moving.Count > 0 && moving[0] == f && moving[^1] == l)));
        }
        menu.Items.Add(new MenuFlyoutSeparator());
        var firstSub = new MenuFlyoutSubItem { Text = Loc.S("Jobs_Tolerance_Panel_FirstSurface") };
        for (var i = 1; i <= last; i++)
        {
            var f = i;
            firstSub.Items.Add(Item(SurfaceTitle(sys, i), () => SetElement(id, f, Math.Max(f, moving.Count > 0 ? moving[^1] : f), pivot), moving.Count == 0 || moving[0] != i));
        }
        menu.Items.Add(firstSub);
        var lastSub = new MenuFlyoutSubItem { Text = Loc.S("Jobs_Tolerance_Panel_LastSurface") };
        for (var i = moving.Count > 0 ? moving[0] : 1; i <= last; i++)
        {
            var l = i;
            lastSub.Items.Add(Item(SurfaceTitle(sys, i), () => SetElement(id, moving.Count > 0 ? moving[0] : l, l, pivot), moving.Count == 0 || moving[^1] != i));
        }
        menu.Items.Add(lastSub);
        return menu;
    }

    void SetElement(Guid id, int first, int last, TolerancePivotChoice pivot, string? name = null) =>
        Edit(name ?? Loc.S("Jobs_Tolerance_Panel_Undo_ChangeElement"), s => pivot.Apply(s, id, _session.Document.System, first, last));

    /// <summary>
    /// "S3 S4 S5 move together · pivot S3, first vertex" under the first element row of a group (the rows starting on one
    /// surface share range and pivot); the pivot a menu.
    /// </summary>
    UIElement ElementLine(ToleranceRow row, OpticalSystem sys)
    {
        var id = row.Id;
        var moving = row.MovingSurfaces(sys);
        var pivotText = TolerancePivotChoice.Text(row.PivotIndex(sys), moving);
        var link = new FontIcon { Glyph = "\uE71B", FontSize = 10, VerticalAlignment = VerticalAlignment.Center };
        JobUI.Bind(link, "TextFillColorTertiaryBrush", b => link.Foreground = b);
        var menu = new MenuFlyout();
        if (moving.Count > 0)
        {
            int f = moving[0], l = moving[^1];
            var current = TolerancePivotChoice.Of(row, sys);
            ToggleMenuFlyoutItem Choice(string text, TolerancePivotChoice c)
            {
                var item = new ToggleMenuFlyoutItem { Text = text, IsChecked = c == current };
                item.Click += (_, _) => SetElement(id, f, l, c, Loc.S("Jobs_Tolerance_Panel_Undo_ChangeElementPivot"));
                return item;
            }
            menu.Items.Add(Choice(Loc.F("Jobs_Tolerance_Panel_PivotFirstVertex", f.ToString(C)), TolerancePivotChoice.First));
            if (l != f) menu.Items.Add(Choice(Loc.F("Jobs_Tolerance_Panel_PivotLastVertex", l.ToString(C)), TolerancePivotChoice.Last));
            var surfaces = new MenuFlyoutSubItem { Text = Loc.S("Jobs_Tolerance_Panel_PivotSurface") };
            for (var i = 1; i <= Math.Max(sys.ImageIndex - 1, 1); i++)
                surfaces.Items.Add(Choice(SurfaceTitle(sys, i), new TolerancePivotChoice(TolerancePivotPlace.Surface, i)));
            menu.Items.Add(new MenuFlyoutSeparator());
            menu.Items.Add(surfaces);
        }
        var label = JobUI.Text(Loc.F("Jobs_Tolerance_Panel_PivotLabel", pivotText), "CaptionText", 11.5);
        JobUI.Bind(label, "AccentTextFillColorPrimaryBrush", b => label.Foreground = b);
        var pivot = new Button
        {
            Content = label, Flyout = menu, Padding = new Thickness(3, 0, 3, 1), MinWidth = 0, MinHeight = 0, VerticalAlignment = VerticalAlignment.Center,
            Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0),
        };
        ToolTipService.SetToolTip(pivot, Loc.F("Jobs_Tolerance_Panel_PivotHelp", (moving.Count > 0 ? moving[0] : 0).ToString(C)));
        AutomationProperties.SetName(pivot, Loc.F("Jobs_Tolerance_Panel_PivotName", pivotText));
        var moves = JobUI.Text(TolerancePivotChoice.MovingText(moving), "CaptionText", 11.5);
        moves.VerticalAlignment = VerticalAlignment.Center;
        var dot = JobUI.Text("·", "CaptionText", 11.5);
        dot.VerticalAlignment = VerticalAlignment.Center;
        var line = JobUI.Row(5, link, moves, dot, pivot);
        line.Margin = new Thickness(10, 0, 8, 2);
        line.Tag = "tolerance-element-line";
        AutomationProperties.SetName(line, Loc.F("Jobs_Tolerance_Panel_ElementLineName", string.Join(Loc.S("Jobs_Tolerance_Panel_ListSeparator"), moving.Select(i => Loc.F("Jobs_Tolerance_Panel_ElementLineSurface", i.ToString(C)))), pivotText));
        return line;
    }

    UIElement Row(ToleranceRow row, OpticalSystem sys, ToleranceIssue? issue, bool alternate, bool elementLine)
    {
        var id = row.Id;
        var selected = _selection.Contains(id);
        var g = new Grid { ColumnSpacing = 4, Padding = new Thickness(8, 2, 6, 2) };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(40) });
        var kind = JobUI.Text(row.Kind.ShortTitle());
        kind.VerticalAlignment = VerticalAlignment.Center;
        UIElement surface = row.Kind == ToleranceKind.Other ? JobUI.Text("", "SecondaryNumberText")
            : MenuButton(row.Kind.IsElement() ? row.ElementTitle(sys) ?? row.SurfaceLabel(sys) : row.SurfaceLabel(sys), SurfaceMenu(row, sys),
                row.Kind.IsElement() ? Loc.S("Jobs_Tolerance_Panel_SurfaceColumnElementHelp") : Loc.S("Jobs_Tolerance_Panel_SurfaceColumnSurfaceHelp"));
        var name = JobUI.Row(2, kind, surface);
        name.VerticalAlignment = VerticalAlignment.Center;
        ToolTipService.SetToolTip(name, row.DisplayLabel(sys) + " · " + row.Distribution.Title);
        g.Children.Add(name);
        void SetLimit(bool plus, double? v)
        {
            var x = Math.Abs(v ?? 0);
            Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeTolerance"), s =>
            {
                if (s.Rows.FirstOrDefault(r => r.Id == id) is not { } r) return;
                if (r.Minus == r.Plus) { r.Minus = x; r.Plus = x; }
                else if (plus) r.Plus = x; else r.Minus = x;
            });
        }
        FrameworkElement limits = row.Minus == row.Plus
            ? JobUI.Row(2, JobUI.Text("±", "SecondaryText"), new NumberField($"lim-{id}", () => row.Plus, v => SetLimit(true, v), 62, "0", name: Loc.F("Jobs_Tolerance_Panel_LimitName", row.DisplayLabel(sys))))
            : JobUI.Row(2, new NumberField($"lim-{id}-m", () => -row.Minus, v => SetLimit(false, v), 58, "0", name: Loc.F("Jobs_Tolerance_Panel_LimitMinusName", row.DisplayLabel(sys))),
                new NumberField($"lim-{id}-p", () => row.Plus, v => SetLimit(true, v), 58, "0", name: Loc.F("Jobs_Tolerance_Panel_LimitPlusName", row.DisplayLabel(sys))));
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
            ToolTipService.SetToolTip(b, Loc.S("Jobs_Tolerance_Panel_UnitHelp"));
            unit = b;
        }
        else unit = JobUI.Text(row.Unit.Label(), "CaptionText");
        unit.VerticalAlignment = VerticalAlignment.Center;
        Grid.SetColumn(unit, 2);
        g.Children.Add(unit);
        var content = new StackPanel { Children = { g } };
        if (elementLine && row.Kind.IsElement()) content.Children.Add(ElementLine(row, sys));
        if (issue is not null)
        {
            var t = JobUI.Text(issue.Text, "CaptionText");
            t.TextWrapping = TextWrapping.Wrap;
            t.TextTrimming = TextTrimming.None;
            t.Margin = new Thickness(8, 0, 8, 3);
            JobUI.Bind(t, "ErrorBrush", b => t.Foreground = b);
            content.Children.Add(t);
        }
        var border = new Border { Child = content, Tag = "tolerance-row" };
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
            if (e.OriginalSource is TextBox || InButton(e.OriginalSource as DependencyObject)) return;
            var ctrl = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Control).HasFlag(global::Windows.UI.Core.CoreVirtualKeyStates.Down);
            if (!ctrl) _selection.Clear();
            if (!_selection.Remove(id)) _selection.Add(id);
            Queue();
        };
        border.ContextFlyout = RowMenu(row);
        AutomationProperties.SetName(border, Loc.F("Jobs_Tolerance_Panel_RowName", row.DisplayLabel(sys), row.LimitsText));
        return border;
    }

    void ConvertUnit(Guid id, ToleranceUnit unit)
    {
        var ctx = _s.Context();
        var row = _s.Settings.Rows.FirstOrDefault(r => r.Id == id);
        if (row?.ConvertedTo(unit, ctx) is not { } converted)
        {
            _session.ShowToast(Loc.S("Jobs_Tolerance_Panel_UnitNeedsData"));
            return;
        }
        Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeToleranceUnit"), s => { var i = s.Rows.FindIndex(r => r.Id == id); if (i >= 0) { converted.Id = id; s.Rows[i] = converted; } });
    }

    MenuFlyout RowMenu(ToleranceRow row)
    {
        var id = row.Id;
        var menu = new MenuFlyout();
        var dist = new MenuFlyoutSubItem { Text = Loc.S("Jobs_Tolerance_Panel_Distribution") };
        foreach (var d in ToleranceDistribution.Choices)
        {
            var item = new RadioMenuFlyoutItem { Text = d.Title, IsChecked = d == row.Distribution, GroupName = "dist" };
            item.Click += (_, _) => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_ChangeDistribution"), s => { if (s.Rows.FirstOrDefault(r => r.Id == id) is { } r) r.Distribution = d; });
            dist.Items.Add(item);
        }
        menu.Items.Add(dist);
        var symmetric = new ToggleMenuFlyoutItem { Text = Loc.S("Jobs_Tolerance_Panel_SymmetricLimits"), IsChecked = row.Minus == row.Plus };
        symmetric.Click += (_, _) => Edit(row.Minus == row.Plus ? Loc.S("Jobs_Tolerance_Panel_Undo_AsymmetricLimits") : Loc.S("Jobs_Tolerance_Panel_Undo_SymmetricLimits"), s =>
        {
            if (s.Rows.FirstOrDefault(r => r.Id == id) is not { } r) return;
            if (r.Minus == r.Plus) r.Minus = r.Plus * 0.5 is var h && h > 0 ? h : r.Plus;
            else r.Minus = r.Plus = Math.Max(r.Minus, r.Plus);
        });
        menu.Items.Add(symmetric);
        menu.Items.Add(new MenuFlyoutSeparator());
        var dup = new MenuFlyoutItem { Text = Loc.S("Jobs_Tolerance_Panel_Duplicate") };
        dup.Click += (_, _) => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_DuplicateTolerance"), s => s.Duplicate(new HashSet<Guid> { id }));
        menu.Items.Add(dup);
        var del = new MenuFlyoutItem { Text = Loc.S("Jobs_Tolerance_Panel_Delete") };
        del.Click += (_, _) => { Edit(Loc.S("Jobs_Tolerance_Panel_Undo_DeleteTolerance"), s => s.Delete(new HashSet<Guid> { id })); _selection.Remove(id); };
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
                var item = new MenuFlyoutItem { Text = sys.Surfaces[i].Label.Length > 0 ? Loc.F("Jobs_Tolerance_Panel_AddSurfaceLabeled", i, sys.Surfaces[i].Label) : Loc.F("Jobs_Tolerance_Panel_AddSurface", i) };
                item.Click += (_, _) =>
                {
                    List<Guid> added = [];
                    var amount = kind switch
                    {
                        ToleranceKind.Radius => 3, ToleranceKind.Thickness => 0.05, ToleranceKind.DecenterX or ToleranceKind.DecenterY => 0.025,
                        ToleranceKind.Index => 0.0005, ToleranceKind.Abbe => 0.5, ToleranceKind.TiltX or ToleranceKind.TiltY => 1, _ => 0.5,
                    };
                    Edit(Loc.S("Jobs_Tolerance_Panel_Undo_AddTolerance"), s => added = s.Add([new ToleranceRow(kind, sid, amount)], _selection.FirstOrDefault() is var f && f != Guid.Empty ? f : null));
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
                    Edit(Loc.S("Jobs_Tolerance_Panel_Undo_AddTolerance"), s => s.Add([new ToleranceRow(kind, first, amount, last: last)]));
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
                var item = new MenuFlyoutItem { Text = ToleranceCompensator.IsBackFocus(param, sys) ? Loc.S("Jobs_Tolerance_Panel_BackFocus") : param.Label(sys) };
                var p = param;
                item.Click += (_, _) => Edit(Loc.S("Jobs_Tolerance_Panel_Undo_AddCompensator"), s =>
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
            body.Children.Add(JobUI.Text(Loc.S("Jobs_Tolerance_Panel_Generator_Title"), "PanelTitleText"));
            var presets = ToleranceGeneratorPresets.All;
            body.Children.Add(JobUI.Labeled(Loc.S("Jobs_Tolerance_Panel_Generator_Class"), JobUI.Choice(presets.Select(x => x.Title()).ToList(), gen.Preset is { } pr ? presets.ToList().IndexOf(pr) : -1,
                k => { gen.Apply(presets[k]); Fill(); }, 140)));
            (string Id, string Title, string Unit, Func<double> Get, Action<double> Set)[] values =
            [
                ("Power", Loc.S("Jobs_Tolerance_Panel_Generator_Power"), "fr", () => gen.RadiusFringes, v => gen.RadiusFringes = v),
                ("Irregularity", Loc.S("Jobs_Tolerance_Panel_Generator_Irregularity"), "fr", () => gen.IrregularityFringes, v => gen.IrregularityFringes = v),
                ("Thickness", Loc.S("Jobs_Tolerance_Panel_Generator_Thickness"), "mm", () => gen.ThicknessMM, v => gen.ThicknessMM = v),
                ("Surface decenter", Loc.S("Jobs_Tolerance_Panel_Generator_SurfaceDecenter"), "mm", () => gen.DecentreMM, v => gen.DecentreMM = v),
                ("Surface tilt", Loc.S("Jobs_Tolerance_Panel_Generator_SurfaceTilt"), "′", () => gen.TiltArcmin, v => gen.TiltArcmin = v),
                ("Element decenter", Loc.S("Jobs_Tolerance_Panel_Generator_ElementDecenter"), "mm", () => gen.ElementDecentreMM, v => gen.ElementDecentreMM = v),
                ("Element tilt", Loc.S("Jobs_Tolerance_Panel_Generator_ElementTilt"), "′", () => gen.ElementTiltArcmin, v => gen.ElementTiltArcmin = v),
                ("Index", Loc.S("Jobs_Tolerance_Panel_Generator_Index"), "Δn", () => gen.Index, v => gen.Index = v),
                ("Abbe number", Loc.S("Jobs_Tolerance_Panel_Generator_AbbeNumber"), "%", () => gen.AbbePercent, v => gen.AbbePercent = v),
            ];
            foreach (var (key, title, unit, get, set) in values)
                body.Children.Add(JobUI.Labeled(title, JobUI.Row(5, new NumberField($"gen-{key}", () => get(), v => { set(Math.Max(0, v ?? 0)); gen.Preset = null; }, 70, "0", name: title),
                    new TextBlock { Text = unit, Width = 22, Style = JobUI.Style("CaptionText") })));
            body.Children.Add(JobUI.Hint(Loc.S("Jobs_Tolerance_Panel_Generator_Hint"), 11.5));
            var go = JobUI.Primary(Loc.S("Jobs_Tolerance_Panel_Generator_Generate"), null, () =>
            {
                flyout.Hide();
                Edit(Loc.S("Jobs_Tolerance_Panel_Undo_DefaultTolerances"), s => { s.Generator = gen.Clone(); s.Generate(_session.Document.System); });
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
        p.Children.Add(JobUI.Row(24, JobUI.BigNumber(Loc.S("Jobs_Tolerance_Panel_Elapsed"), Duration(elapsed)),
            JobUI.BigNumber(Loc.S("Jobs_Tolerance_Panel_TimeLeft"), timing.SecondsLeft is { } left ? "≈ " + Duration(left) : "—")));
        if (timing.TrialsDone is { } done) p.Children.Add(JobUI.BigNumber(Loc.S("Jobs_Tolerance_Panel_Trials"), $"{done.ToString(C)} / {_s.Settings.Trials.ToString(C)}"));
        if (timing.SecondsPerTrial is { } spt) p.Children.Add(JobUI.Hint(Loc.F("Jobs_Tolerance_Panel_PerTrial", Duration(spt)), 12));
        p.Children.Add(new ProgressBar { Value = _s.Progress.Fraction * 100, Maximum = 100, IsIndeterminate = _s.Progress.Fraction <= 0 });
        p.Children.Add(JobUI.Text(_s.Progress.Stage.Length > 0 ? EngineText.T(_s.Progress.Stage) : Loc.S("Jobs_Tolerance_Panel_Running"), "SecondaryText"));
        p.Children.Add(JobUI.Hint(Loc.S("Jobs_Tolerance_Panel_RunningHint")));
        return p;
    }

    // ---- Results ----

    UIElement Results()
    {
        var p = new StackPanel { Spacing = 14 };
        if (_s.Result is not { } r) return p;
        var d = r.Display;
        var sys = _s.RunSystem ?? _session.Document.System;
        var head = new Grid();
        var criterion = JobUI.Text(d.Unit().Length == 0 ? d.Title() : Loc.F("Jobs_Tolerance_Panel_TitleWithUnit", d.Title(), d.Unit()), "PanelTitleText");
        criterion.VerticalAlignment = VerticalAlignment.Center;
        head.Children.Add(criterion);
        var export = new DropDownButton
        {
            Content = JobUI.Row(6, new FontIcon { Glyph = "\uE72D", FontSize = 12 }, new TextBlock { Text = Loc.S("Jobs_Tolerance_Panel_Export") }), Flyout = ExportMenu(r),
            HorizontalAlignment = HorizontalAlignment.Right, MinHeight = 28, FontSize = 12.5, Padding = new Thickness(10, 3, 8, 4),
        };
        ToolTipService.SetToolTip(export, Loc.S("Jobs_Tolerance_Panel_ExportHelp"));
        AutomationProperties.SetName(export, Loc.S("Jobs_Tolerance_Panel_Export"));
        head.Children.Add(export);
        p.Children.Add(head);
        if (r.Stopped) p.Children.Add(JobUI.Hint(Loc.S("Jobs_Tolerance_Panel_StoppedEarly")));
        var big = JobUI.Row(24);
        if (r.HasNominal) big.Children.Add(JobUI.BigNumber(Loc.S("Jobs_Tolerance_Panel_Nominal"), d.FormatWithUnit(d.Value(r.Nominal))));
        if (r.HasSensitivity && double.IsFinite(r.Predicted)) big.Children.Add(JobUI.BigNumber(Loc.S("Jobs_Tolerance_Panel_RssPrediction"), d.FormatWithUnit(d.Value(r.Predicted))));
        p.Children.Add(big);
        if (r.HasMonteCarlo)
        {
            var levels = new StackPanel { Spacing = 2 };
            foreach (var pc in r.Percentiles)
            {
                var row = new Grid();
                row.Children.Add(JobUI.Text(Loc.F("Jobs_Tolerance_Panel_LevelOfBuilds", MeritFormat.Short(pc.Level)), "SecondaryText"));
                var v = JobUI.Text((d.HigherIsBetter() ? "≥ " : "≤ ") + d.FormatWithUnit(d.Value(pc.Value)), "NumberText");
                v.HorizontalAlignment = HorizontalAlignment.Right;
                row.Children.Add(v);
                levels.Children.Add(row);
            }
            if (r.Yield is { } y)
            {
                var row = new Grid();
                row.Children.Add(JobUI.Text(Loc.F("Jobs_Tolerance_Panel_Yield", r.Threshold is { } th ? d.FormatWithUnit(d.Value(th)) : "—"), "SecondaryText"));
                var v = JobUI.Text(MeritFormat.Percent(y), "NumberText");
                v.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
                v.HorizontalAlignment = HorizontalAlignment.Right;
                row.Children.Add(v);
                levels.Children.Add(row);
            }
            var runSummary = r.Failures > 0 ? Loc.F("Jobs_Tolerance_Panel_TrialsSummaryFailed", r.Trials.ToString(C), r.Failures.ToString(C), r.Seed.ToString(C))
                : Loc.F("Jobs_Tolerance_Panel_TrialsSummary", r.Trials.ToString(C), r.Seed.ToString(C));
            levels.Children.Add(JobUI.Hint(r.SecondsPerTrial is { } spt ? Loc.F("Jobs_Tolerance_Panel_TrialsSummaryTimed", runSummary, Duration(spt)) : runSummary, 11.5));
            p.Children.Add(JobUI.Card(levels, 8));
        }
        if (_s.Previewed is not null) p.Children.Add(PreviewCard(r, d));
        if (r.HasSensitivity && r.Sensitivities.Count > 0) p.Children.Add(Sensitivities(r, d, sys));
        if (r.HasMonteCarlo && r.Criteria.Count > 0) p.Children.Add(Histogram(r, d));
        if (r.Compensators.Count > 0) p.Children.Add(Compensators(r, sys));
        return p;
    }

    /// <summary>Copy Summary; Save Sensitivity Table / Monte Carlo Trials as CSV (the trials with their recorded compensators and perturbations).</summary>
    MenuFlyout ExportMenu(ViewModels.Jobs.ToleranceResult r)
    {
        var menu = new MenuFlyout();
        var copy = new MenuFlyoutItem { Text = Loc.S("Jobs_Tolerance_Panel_CopySummary"), Icon = new FontIcon { Glyph = "\uE8C8" } };
        copy.Click += (_, _) =>
        {
            if (_s.SummaryText() is not { } text) return;
            var package = new DataPackage();
            package.SetText(text);
            Clipboard.SetContent(package);
            _session.ShowToast(Loc.S("Jobs_Tolerance_Panel_SummaryCopied"));
        };
        menu.Items.Add(copy);
        menu.Items.Add(new MenuFlyoutSeparator());
        var name = _session.Document.System.Name;
        if (r.HasSensitivity)
        {
            var item = new MenuFlyoutItem { Text = Loc.S("Jobs_Tolerance_Panel_SaveSensitivityCsv") };
            item.Click += (_, _) => _ = SaveCsvAsync(ToleranceExport.FileName(name, "sensitivity"), _s.SensitivityCsv());
            menu.Items.Add(item);
        }
        if (r.HasMonteCarlo)
        {
            var item = new MenuFlyoutItem { Text = Loc.S("Jobs_Tolerance_Panel_SaveTrialsCsv") };
            item.Click += (_, _) => _ = SaveCsvAsync(ToleranceExport.FileName(name, "trials"), _s.TrialsCsv());
            menu.Items.Add(item);
        }
        return menu;
    }

    async Task SaveCsvAsync(string suggested, string? text)
    {
        if (text is null) return;
        var window = global::MKLight.App.App.Windows.OfType<global::MKLight.App.Windows.DesignWindow>().FirstOrDefault(w => w.Session == _session);
        if (window is null) return;
        var picker = new Microsoft.Windows.Storage.Pickers.FileSavePicker(window.AppWindow.Id)
        {
            SuggestedFileName = System.IO.Path.GetFileNameWithoutExtension(suggested), DefaultFileExtension = ".csv",
            SuggestedStartLocation = Microsoft.Windows.Storage.Pickers.PickerLocationId.DocumentsLibrary,
        };
        picker.FileTypeChoices.Add(Loc.S("Jobs_Tolerance_Panel_CsvFileType"), [".csv"]);
        if ((await picker.PickSaveFileAsync())?.Path is not { } path) return;
        try
        {
            await System.IO.File.WriteAllTextAsync(path, text);
            _session.ShowToast(Loc.F("Jobs_Tolerance_Panel_Saved", System.IO.Path.GetFileName(path)));
        }
        catch (Exception e) when (e is System.IO.IOException or UnauthorizedAccessException) { _session.ShowToast(Loc.F("Jobs_Tolerance_Panel_CouldNotSave", e.Message)); }
    }

    /// <summary>Each compensator: the range the sensitivity evaluations used against the range allowed, its nominal and its Monte-Carlo spread.</summary>
    UIElement Compensators(ViewModels.Jobs.ToleranceResult r, OpticalSystem sys)
    {
        var p = new StackPanel { Spacing = 16 };
        for (var j = 0; j < r.Compensators.Count; j++)
        {
            var c = r.Compensators[j];
            var unit = c.Param.Unit(sys);
            var usage = ToleranceCompensatorUsage.Of(r, j);
            var nominal = j < r.NominalCompensators.Count ? r.NominalCompensators[j] : (double?)null;
            var box = new StackPanel { Spacing = 5 };
            var head = new Grid();
            var title = JobUI.Text(new ToleranceCompensator(c.Param, c.Range).Title(sys));
            title.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
            head.Children.Add(title);
            if (c.Range is { } rg)
            {
                var allowed = JobUI.Text(Loc.F("Jobs_Tolerance_Panel_Comp_Allowed", MeritFormat.Short(rg), unit), "CaptionText");
                allowed.HorizontalAlignment = HorizontalAlignment.Right;
                head.Children.Add(allowed);
            }
            box.Children.Add(head);
            if (usage is { Fractions: { } f } u)
            {
                var bar = RangeBar(f.From, f.To, u.AtLimit);
                ToolTipService.SetToolTip(bar, Loc.F("Jobs_Tolerance_Panel_Comp_RangeBarHelp", MeritFormat.Short(u.Low), MeritFormat.Short(u.High), unit, MeritFormat.Short(u.Range!.Value)));
                box.Children.Add(bar);
            }
            var pairs = new Grid { ColumnSpacing = 8, RowSpacing = 2 };
            pairs.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            pairs.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            void Pair(string label, string value, string help)
            {
                pairs.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
                var l = JobUI.Text(label, "SecondaryText", 12);
                ToolTipService.SetToolTip(l, help);
                Grid.SetRow(l, pairs.RowDefinitions.Count - 1);
                pairs.Children.Add(l);
                var v = JobUI.Text(value, "NumberText", 12, wrap: true);
                v.TextAlignment = TextAlignment.Right;
                Grid.SetRow(v, pairs.RowDefinitions.Count - 1);
                Grid.SetColumn(v, 1);
                pairs.Children.Add(v);
            }
            if (nominal is { } n) Pair(Loc.S("Jobs_Tolerance_Panel_Nominal"), $"{MeritFormat.Short(n)} {unit}", Loc.S("Jobs_Tolerance_Panel_Comp_NominalHelp"));
            if (usage is { } used)
                Pair(Loc.S("Jobs_Tolerance_Panel_Comp_RangeUsed"), Loc.F("Jobs_Tolerance_Panel_Comp_RangeUsedValue", ToleranceCompensatorUsage.Signed4(used.Low), ToleranceCompensatorUsage.Signed4(used.High), unit), Loc.S("Jobs_Tolerance_Panel_Comp_RangeUsedHelp"));
            if (j < r.CompensatorMean.Count)
            {
                var mean = r.CompensatorMean[j];
                Pair(Loc.S("Jobs_Tolerance_Panel_Comp_Mean"), nominal is { } n0 ? Loc.F("Jobs_Tolerance_Panel_Comp_MeanWithShift", MeritFormat.Short(mean), unit, ToleranceCompensatorUsage.Signed4(mean - n0)) : $"{MeritFormat.Short(mean)} {unit}", Loc.S("Jobs_Tolerance_Panel_Comp_MeanHelp"));
                var sd = j < r.CompensatorStd.Count ? r.CompensatorStd[j] : double.NaN;
                Pair(Loc.S("Jobs_Tolerance_Panel_Comp_Sigma"), double.IsFinite(sd) ? Loc.F("Jobs_Tolerance_Panel_Comp_SigmaValue", sd.ToString("G4", C), (3 * sd).ToString("G4", C), unit) : "\u2014", Loc.S("Jobs_Tolerance_Panel_Comp_SigmaHelp"));
            }
            box.Children.Add(pairs);
            if (usage is { AtLimit: true }) box.Children.Add(JobUI.Warning(Loc.S("Jobs_Tolerance_Panel_Comp_AtLimit")));
            p.Children.Add(box);
        }
        return JobUI.Group(Loc.S("Jobs_Tolerance_Panel_CompensatorsGroup"), p);
    }

    /// <summary>The used part of a compensator's \u00B1range (fractions \u22121 \u2026 1), centred on nominal; orange when it reached the limit.</summary>
    static Grid RangeBar(double from, double to, bool atLimit)
    {
        var g = new Grid { Height = 10, Margin = new Thickness(0, 1, 0, 1) };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(Math.Max(1 + from, 0), GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(Math.Max(to - from, 0.004), GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(Math.Max(1 - to, 0), GridUnitType.Star) });
        var back = new Border { Height = 5, CornerRadius = new CornerRadius(2.5), VerticalAlignment = VerticalAlignment.Center };
        Grid.SetColumnSpan(back, 3);
        JobUI.Bind(back, "SubtleFillColorTertiaryBrush", b => back.Background = b);
        g.Children.Add(back);
        var fill = new Border { Height = 5, MinWidth = 2, CornerRadius = new CornerRadius(2.5), VerticalAlignment = VerticalAlignment.Center };
        Grid.SetColumn(fill, 1);
        JobUI.Bind(fill, atLimit ? "WarningBrush" : "AccentFillColorDefaultBrush", b => fill.Background = b);
        g.Children.Add(fill);
        var tick = new Border { Width = 1, HorizontalAlignment = HorizontalAlignment.Center };
        Grid.SetColumnSpan(tick, 3);
        JobUI.Bind(tick, "TextFillColorSecondaryBrush", b => tick.Background = b);
        g.Children.Add(tick);
        AutomationProperties.SetAccessibilityView(g, Microsoft.UI.Xaml.Automation.Peers.AccessibilityView.Raw);
        return g;
    }

    UIElement PreviewCard(ViewModels.Jobs.ToleranceResult r, CriterionDisplay d)
    {
        var p = new StackPanel { Spacing = 4 };
        var head = new Grid();
        var title = JobUI.Text(_s.Previewed!.Title(r, _s.RunSystem, _s.ResultRows), "PanelTitleText");
        JobUI.Bind(title, "AccentTextFillColorPrimaryBrush", b => title.Foreground = b);
        head.Children.Add(title);
        var end = new HyperlinkButton { Content = Loc.S("Jobs_Tolerance_Panel_EndPreview"), Padding = new Thickness(0), FontSize = 12, HorizontalAlignment = HorizontalAlignment.Right };
        end.Click += (_, _) => _s.EndPreview();
        head.Children.Add(end);
        p.Children.Add(head);
        if (_s.PreviewError is { } err) p.Children.Add(JobUI.Warning(EngineText.T(err)));
        else if (_s.Preview is not { } pv) p.Children.Add(JobUI.Hint(Loc.S("Jobs_Tolerance_Panel_Rebuilding")));
        else
        {
            p.Children.Add(JobUI.Text(r.HasNominal ? Loc.F("Jobs_Tolerance_Panel_PreviewCriterionNominal", d.FormatWithUnit(d.Value(pv.Criterion)), d.FormatWithUnit(d.Value(r.Nominal))) : Loc.F("Jobs_Tolerance_Panel_PreviewCriterion", d.FormatWithUnit(d.Value(pv.Criterion))), "NumberText"));
            for (var j = 0; j < Math.Min(pv.Compensators.Count, r.Compensators.Count); j++)
            {
                var nominal = j < r.NominalCompensators.Count ? r.NominalCompensators[j] : double.NaN;
                p.Children.Add(JobUI.Text(Loc.F("Jobs_Tolerance_Panel_PreviewCompensator", r.Compensators[j].Param.Label(_s.RunSystem ?? _session.Document.System), MeritFormat.Short(pv.Compensators[j]), MeritFormat.Signed(pv.Compensators[j] - nominal)), "CaptionText"));
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
            var label = s.Id >= 0 && s.Id < _s.ResultRows.Count ? _s.ResultRows[s.Id].DisplayLabel(sys) : EngineText.T(s.Label);
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
            var eye = JobUI.Icon(on ? "\uED1A" : "\uE7B3", on ? Loc.S("Jobs_Tolerance_Panel_EndThePreview") : Loc.S("Jobs_Tolerance_Panel_PreviewWorseLimit"),
                () => _ = _s.PreviewAsync(on ? null : c), true, 26);
            if (on) JobUI.Bind(eye, "AccentTextFillColorPrimaryBrush", b => eye.Foreground = b);
            Grid.SetColumn(eye, 2);
            g.Children.Add(eye);
            AutomationProperties.SetName(g, Loc.F("Jobs_Tolerance_Panel_SensitivityRowName", label, MeritFormat.Share(share)));
            p.Children.Add(g);
        }
        var heading = r.Sensitivities.Count > 25 ? Loc.F("Jobs_Tolerance_Panel_SensitivityGroupLargest", r.Sensitivities.Count.ToString(C)) : Loc.S("Jobs_Tolerance_Panel_SensitivityGroup");
        return JobUI.Group(heading, p, JobUI.Text(Loc.S("Jobs_Tolerance_Panel_ShareOfRss"), "CaptionText"));
    }

    /// <summary>The Monte-Carlo histogram with the nominal and the levels marked, and the worst trials (a click previews one).</summary>
    UIElement Histogram(ViewModels.Jobs.ToleranceResult r, CriterionDisplay d)
    {
        var values = r.Criteria.Select(x => d.Value(x)).Where(double.IsFinite).ToList();
        if (values.Count == 0) return new StackPanel();
        var bins = ToleranceHistogram.Bins(r);
        var selected = _s.SelectedBin is { } sb && sb < bins.Count ? sb : (int?)null;
        var previewed = _s.Previewed is TolerancePreviewCase.Trial pt && pt.K < r.TrialCriteria.Count ? d.Value(r.TrialCriteria[pt.K]) : double.NaN;
        var p = new StackPanel { Spacing = 6 };
        var canvas = new Canvas { Height = 120, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent) };
        var lo = values.Min();
        var hi = values.Max();
        if (r.HasNominal) { lo = Math.Min(lo, d.Value(r.Nominal)); hi = Math.Max(hi, d.Value(r.Nominal)); }
        if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }
        var max = Math.Max(1, bins.Max(b => b.Count));
        void Draw()
        {
            canvas.Children.Clear();
            var w = canvas.ActualWidth;
            if (w <= 0) return;
            var h = canvas.Height - 6;
            double X(double v) => (v - lo) / (hi - lo) * w;
            var dark = canvas.ActualTheme == ElementTheme.Dark;
            var accent = Support.ThemeLookup.Brush(canvas, "AccentFillColorDefaultBrush");
            var bar = new SolidColorBrush(Support.Palette.Field(0, dark));
            for (var i = 0; i < bins.Count; i++)
            {
                if (bins[i].Count == 0) continue;
                var x0 = X(bins[i].Low);
                var x1 = bins.Count == 1 ? x0 + Math.Max(w / 24, 4) : X(bins[i].High);
                var bh = Math.Max(1.5, bins[i].Count / (double)max * (h - 4));
                var on = selected == i;
                var rect = new Rectangle
                {
                    Width = Math.Max(1, x1 - x0 - 2), Height = bh, Fill = on ? accent : bar, RadiusX = 1.5, RadiusY = 1.5,
                    Opacity = on ? 1 : selected is null ? 0.75 : 0.4,
                };
                Canvas.SetLeft(rect, x0 + 1);
                Canvas.SetTop(rect, h - bh);
                canvas.Children.Add(rect);
            }
            void Mark(double v, string key, double opacity, bool dashed)
            {
                var x = X(v);
                var line = new Line { X1 = x, X2 = x, Y1 = 0, Y2 = h, Stroke = Support.ThemeLookup.Brush(canvas, key), StrokeThickness = dashed ? 1 : 1.6, Opacity = opacity };
                if (dashed) line.StrokeDashArray = [3, 3];
                canvas.Children.Add(line);
            }
            foreach (var pc in r.Percentiles) Mark(d.Value(pc.Value), "TextFillColorSecondaryBrush", 0.9, true);
            if (r.HasNominal) Mark(d.Value(r.Nominal), "AccentFillColorDefaultBrush", 1, false);
            canvas.Children.Add(new Line { X1 = 0, X2 = w, Y1 = h, Y2 = h, Stroke = Support.ThemeLookup.Brush(canvas, "DividerBrush"), StrokeThickness = 1 });
            if (double.IsFinite(previewed))
            {
                var x = X(previewed);
                canvas.Children.Add(new Polygon { Points = { new(x, h), new(x - 5, h + 6), new(x + 5, h + 6) }, Fill = accent });
            }
        }
        canvas.SizeChanged += (_, _) => Draw();
        canvas.ActualThemeChanged += (_, _) => Draw();
        canvas.Tapped += (_, e) =>
        {
            var w = canvas.ActualWidth;
            if (w <= 0) return;
            var v = lo + e.GetPosition(canvas).X / w * (hi - lo);
            var hit = ToleranceHistogram.IndexOf(bins, v) is { } k && bins[k].Count > 0 ? k : (int?)null;
            _s.SelectBin(hit == selected ? null : hit);
        };
        ToolTipService.SetToolTip(canvas, Loc.S("Jobs_Tolerance_Panel_HistogramHelp"));
        AutomationProperties.SetName(canvas, Loc.F("Jobs_Tolerance_Panel_HistogramName", values.Count.ToString(C), d.Title()));
        p.Children.Add(canvas);
        var scale = new Grid();
        scale.Children.Add(JobUI.Text(d.Format(lo), "CaptionText"));
        var right = JobUI.Text(d.FormatWithUnit(hi), "CaptionText");
        right.HorizontalAlignment = HorizontalAlignment.Right;
        scale.Children.Add(right);
        p.Children.Add(scale);
        var levels = string.Join(", ", r.Percentiles.Select(x => MeritFormat.Short(x.Level)));
        var legend = r.HasNominal
            ? levels.Length > 0 ? Loc.F("Jobs_Tolerance_Panel_HistogramSolidDashed", d.FormatWithUnit(d.Value(r.Nominal)), levels) : Loc.F("Jobs_Tolerance_Panel_HistogramSolid", d.FormatWithUnit(d.Value(r.Nominal)))
            : levels.Length > 0 ? Loc.F("Jobs_Tolerance_Panel_HistogramDashed", levels) : "";
        p.Children.Add(JobUI.Hint(legend, 11));

        // The selected bar's trials, or the worst.
        var trials = selected is { } bin ? ToleranceHistogram.Trials(r, bins, bin) : r.TrialsWorstFirst;
        if (_limitBin != selected) { _limitBin = selected; _trialLimit = 6; }
        var head = new Grid { Margin = new Thickness(0, 4, 0, 0) };
        var title = JobUI.Text(selected is { } sel ? ToleranceHistogram.Title(d, bins[sel], trials.Count) : Loc.S("Jobs_Tolerance_Panel_WorstTrialsTitle"));
        title.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
        head.Children.Add(title);
        if (selected is not null)
        {
            var worst = new HyperlinkButton { Content = Loc.S("Jobs_Tolerance_Panel_WorstTrialsButton"), Padding = new Thickness(0), FontSize = 12, HorizontalAlignment = HorizontalAlignment.Right };
            worst.Click += (_, _) => _s.SelectBin(null);
            ToolTipService.SetToolTip(worst, Loc.S("Jobs_Tolerance_Panel_WorstTrialsHelp"));
            head.Children.Add(worst);
        }
        p.Children.Add(head);
        if (trials.Count == 0) p.Children.Add(JobUI.Hint(Loc.S("Jobs_Tolerance_Panel_NoTrialInBar")));
        foreach (var k in trials.Take(_trialLimit)) p.Children.Add(TrialRow(r, d, k));
        if (trials.Count > _trialLimit)
        {
            var more = new HyperlinkButton { Content = Loc.F("Jobs_Tolerance_Panel_ShowMore", Math.Min(trials.Count - _trialLimit, 20).ToString(C)), Padding = new Thickness(0), FontSize = 12 };
            more.Click += (_, _) => { _trialLimit += 20; Queue(); };
            p.Children.Add(more);
        }
        p.Children.Add(JobUI.Hint(selected is null ? Loc.S("Jobs_Tolerance_Panel_HistogramHint") : Loc.S("Jobs_Tolerance_Panel_HistogramHintSelected"), 11));
        var group = JobUI.Group(Loc.S("Jobs_Tolerance_Panel_MonteCarloGroup"), p, JobUI.Text(Loc.F("Jobs_Tolerance_Panel_Builds", r.Trials.ToString(C)), "CaptionText"));
        group.Tag = "tolerance-monte-carlo";
        return group;
    }

    int _trialLimit = 6;
    int? _limitBin;

    /// <summary>A trial: its number, value and first compensator's move; a click previews it (or ends its preview).</summary>
    Button TrialRow(ViewModels.Jobs.ToleranceResult r, CriterionDisplay d, int k)
    {
        var c = new TolerancePreviewCase.Trial(k);
        var on = Equals(_s.Previewed, c);
        var b = new Button
        {
            HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch, Padding = new Thickness(6, 2, 6, 2),
            Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0), MinHeight = 0,
        };
        if (on) b.Loaded += (_, _) =>
        {
            var accent = (global::Windows.UI.Color)Support.ThemeLookup.Resource(b.ActualTheme, "SystemAccentColor");
            b.Background = new SolidColorBrush(global::Windows.UI.Color.FromArgb(b.ActualTheme == ElementTheme.Dark ? (byte)56 : (byte)36, accent.R, accent.G, accent.B));
        };
        var g = new Grid { ColumnSpacing = 8 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(64) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.Children.Add(JobUI.Text(Loc.F("Jobs_Tolerance_Panel_Trial", (k + 1).ToString(C)), "NumberText"));
        var value = d.FormatWithUnit(d.Value(r.TrialCriteria[k]));
        var v = JobUI.Text(value, "NumberText");
        Grid.SetColumn(v, 1);
        g.Children.Add(v);
        if (r.Record(k) is { Compensators.Count: > 0 } rec && r.NominalCompensators.Count > 0 && r.Compensators.Count > 0 && _s.RunSystem is { } rs)
        {
            var comp = JobUI.Text(new ToleranceCompensator(r.Compensators[0].Param).Title(rs) + " " + ToleranceCompensatorUsage.Move(rec.Compensators[0] - r.NominalCompensators[0]), "SecondaryNumberText", 11.5);
            comp.TextTrimming = TextTrimming.CharacterEllipsis;
            comp.VerticalAlignment = VerticalAlignment.Center;
            Grid.SetColumn(comp, 2);
            g.Children.Add(comp);
        }
        var eye = new FontIcon { Glyph = "\uE7B3", FontSize = 13 };
        JobUI.Bind(eye, on ? "AccentTextFillColorPrimaryBrush" : "TextFillColorSecondaryBrush", x => eye.Foreground = x);
        Grid.SetColumn(eye, 3);
        g.Children.Add(eye);
        b.Content = g;
        ToolTipService.SetToolTip(b, on ? Loc.S("Jobs_Tolerance_Panel_EndThePreview") : r.Compensators.Count == 0 ? Loc.F("Jobs_Tolerance_Panel_PreviewTrial", (k + 1).ToString(C)) : Loc.F("Jobs_Tolerance_Panel_PreviewTrialCompensated", (k + 1).ToString(C)));
        AutomationProperties.SetName(b, Loc.F("Jobs_Tolerance_Panel_TrialName", (k + 1).ToString(C), value));
        b.Click += (_, _) => _ = _s.PreviewAsync(on ? null : c);
        return b;
    }

    // ---- Actions ----

    void Actions()
    {
        var problem = _s.RunProblem();
        switch (_s.Phase)
        {
            case JobPhase.Setup:
                _frame.SetActions(null, JobUI.Primary(Loc.S("Jobs_Tolerance_Panel_Run"), "\uE768", () => _s.Start(), problem ?? Loc.S("Jobs_Tolerance_Panel_RunHelp"), problem is null));
                break;
            case JobPhase.Failed:
                _frame.SetActions(null, JobUI.Secondary(Loc.S("Jobs_Tolerance_Panel_Back"), null, _s.EditSettings, Loc.S("Jobs_Tolerance_Panel_BackHelp")),
                    JobUI.Primary(Loc.S("Jobs_Tolerance_Panel_TryAgain"), "\uE72C", () => _s.Start(fresh: true), Loc.S("Jobs_Tolerance_Panel_TryAgainHelp"), problem is null));
                break;
            case JobPhase.Running:
                _frame.SetActions(JobUI.Progress(_s.Progress),
                    JobUI.Secondary(Loc.S("Common_Cancel"), null, _s.Cancel, Loc.S("Jobs_Tolerance_Panel_CancelHelp"), destructive: true),
                    JobUI.Secondary(Loc.S("Jobs_Tolerance_Panel_Stop"), "\uE71A", _s.Stop, Loc.S("Jobs_Tolerance_Panel_StopHelp"), _s.CanStop));
                break;
            default:
                _frame.SetActions(null, JobUI.Primary(Loc.S("Jobs_Tolerance_Panel_EditTolerances"), null, _s.EditSettings, Loc.S("Jobs_Tolerance_Panel_EditTolerancesHelp")));
                break;
        }
    }
}
