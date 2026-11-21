using System.Globalization;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;
using Windows.System;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Optimize page's merit function (the Mac app's <c>MeritFunctionSection</c>): Default (the engine's generator:
/// criterion, pupil sampling, fields and wavelengths, thickness limits, squeeze) or Custom (an operand list with a
/// searchable picker and the selected operand's editor), with the merit evaluated live at the design. Every change is
/// one undoable step in the project's <c>merit</c> section.
/// </summary>
public sealed class MeritEditor
{
    readonly OptimizeSession _s;
    readonly StackPanel _readout = new() { HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Center };
    readonly TextBlock _meritValue = new() { FontSize = 14, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Right };
    readonly ProgressRing _ring = new() { Width = 12, Height = 12, IsActive = true, VerticalAlignment = VerticalAlignment.Center };
    readonly StackPanel _error = new();
    StackPanel? _operandList;
    /// <summary>The selected operands (session identities), kept across rebuilds.</summary>
    public HashSet<Guid> Selection { get; } = [];

    public MeritEditor(OptimizeSession session)
    {
        _s = session;
        _meritValue.Style = JobUI.Style("NumberText");
        _meritValue.FontSize = 14;
        _meritValue.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
        var label = JobUI.Text(Loc.S("Jobs_Merit_Readout"), "CaptionText");
        label.HorizontalAlignment = HorizontalAlignment.Right;
        _readout.Children.Add(label);
        _readout.Children.Add(JobUI.Row(5, _ring, _meritValue));
        ((StackPanel)_readout.Children[1]).HorizontalAlignment = HorizontalAlignment.Right;
        AutomationProperties.SetName(_readout, Loc.S("Jobs_Merit_ReadoutName"));
    }

    MeritSettings M => _s.Merit;
    OpticalSystem System => _s.Document.System;

    void Edit(string name, Action<MeritSettings> change) => _s.EditMerit(name, change);

    void Set(string name, Func<DefaultMeritSettings, DefaultMeritSettings> change) => Edit(name, m => m.Defaults = change(m.Defaults));

    /// <summary>The live parts: the readout, the error line and the operands' values (no rebuild of the fields).</summary>
    public void UpdateEvaluation()
    {
        var e = _s.Evaluation;
        _meritValue.Text = e is null ? "—" : MeritFormat.Merit(e.Merit);
        _ring.Visibility = _s.Evaluating ? Visibility.Visible : Visibility.Collapsed;
        _readout.Opacity = _s.Evaluating ? 0.6 : 1;
        ToolTipService.SetToolTip(_readout, e?.Help() ?? Loc.S("Jobs_Merit_ReadoutHelp"));
        AutomationProperties.SetHelpText(_readout, e is null ? Loc.S("Jobs_Merit_NotEvaluated") : MeritFormat.Short(e.Merit));
        _error.Children.Clear();
        if (_s.EvaluationError is { } why && !_s.Evaluating) _error.Children.Add(JobUI.Warning(Loc.F("Jobs_Merit_EvaluationError", EngineText.T(why))));
        if (_operandList is not null) FillOperands(_operandList);
    }

    public UIElement Build()
    {
        var body = new StackPanel { Spacing = 10 };
        // The live parts are kept across rebuilds: take them out of the old page first.
        foreach (var e in new FrameworkElement[] { _readout, _error })
            if (e.Parent is Panel old) old.Children.Remove(e);
        var top = new Grid();
        var mode = JobUI.Segments([Loc.S("Jobs_Merit_Mode_Default"), Loc.S("Jobs_Merit_Mode_Custom")], M.Mode == MeritMode.Default ? 0 : 1,
            k => Edit(Loc.S(k == 1 ? "Jobs_Merit_Undo_UseCustom" : "Jobs_Merit_Undo_UseDefault"), m => m.Mode = k == 1 ? MeritMode.Custom : MeritMode.Default));
        ToolTipService.SetToolTip(mode, Loc.S("Jobs_Merit_ModeHelp"));
        mode.VerticalAlignment = VerticalAlignment.Center;
        top.Children.Add(mode);
        top.Children.Add(_readout);
        body.Children.Add(top);
        _operandList = null;
        if (M.Mode == MeritMode.Default)
        {
            body.Children.Add(DefaultForm());
            var convert = JobUI.Secondary(Loc.S("Jobs_Merit_ConvertToCustom"), "\uE8FD", () => _ = _s.ConvertToCustomAsync(),
                Loc.S("Jobs_Merit_ConvertToCustomHelp"), !_s.Evaluating && _s.Workspace.DesignSnapshot is not null);
            convert.HorizontalAlignment = HorizontalAlignment.Left;
            body.Children.Add(convert);
        }
        else body.Children.Add(OperandTable());
        body.Children.Add(_error);
        body.Children.Add(JobUI.Check(Loc.S("Jobs_Merit_HoldEfl"), M.HoldEfl, v => Edit(Loc.S("Jobs_Merit_Undo_HoldEfl"), m => m.HoldEfl = v),
            Loc.S("Jobs_Merit_HoldEflHelp")));
        UpdateEvaluation();
        return JobUI.Group(Loc.S("Jobs_Merit_Title"), body);
    }

    // ---- Default ----

    UIElement DefaultForm()
    {
        var d = M.Defaults;
        var p = new StackPanel { Spacing = 6 };
        var criteria = Enum.GetValues<MeritCriterion>();
        p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Criterion"), JobUI.Choice(criteria.Select(DefaultMeritSettings.Title).ToList(), Array.IndexOf(criteria, d.Criterion),
            k => Set(Loc.S("Jobs_Merit_Undo_Criterion"), x => x with { Criterion = criteria[k] }))));
        switch (d.Criterion)
        {
            case MeritCriterion.RmsSpot:
                p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Reference"), JobUI.Choice([Loc.S("Jobs_Merit_Reference_Centroid"), Loc.S("Jobs_Merit_Reference_ChiefRay")], d.Reference == "chief" ? 1 : 0,
                    k => Set(Loc.S("Jobs_Merit_Undo_Reference"), x => x with { Reference = k == 1 ? "chief" : "centroid" }))));
                if (d.Reference == "centroid")
                    p.Children.Add(JobUI.Check(Loc.S("Jobs_Merit_WhiteLightCentroid"), d.Scope == "polychromatic",
                        v => Set(Loc.S("Jobs_Merit_Undo_Centroid"), x => x with { Scope = v ? "polychromatic" : "monochromatic" }),
                        Loc.S("Jobs_Merit_WhiteLightCentroidHelp")));
                break;
            case MeritCriterion.Wavefront:
                p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Reference"), JobUI.Choice([Loc.S("Jobs_Merit_Reference_CentroidWavefront"), Loc.S("Jobs_Merit_Reference_ChiefRayWavefront")], d.Reference == "chief" ? 1 : 0,
                    k => Set(Loc.S("Jobs_Merit_Undo_Reference"), x => x with { Reference = k == 1 ? "chief" : "centroid" }), 120)));
                break;
            default:
                p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Frequencies"), JobUI.Row(5, FrequencyField(d), JobUI.Text("cyc/mm", "SecondaryText"))));
                p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Target"), new NumberField("mtf-target", () => d.MtfTarget,
                    v => Set(Loc.S("Jobs_Merit_Undo_MtfTarget"), x => x with { MtfTarget = Math.Clamp(v ?? 0.5, 0, 1) }), 76, "0.5", name: Loc.S("Jobs_Merit_MtfTargetName"))));
                p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Method"), JobUI.Choice([Loc.S("Jobs_Merit_Method_Diffraction"), Loc.S("Jobs_Merit_Method_Geometric")], d.MtfMethod == "geometric" ? 1 : 0,
                    k => Set(Loc.S("Jobs_Merit_Undo_MtfMethod"), x => x with { MtfMethod = k == 1 ? "geometric" : "diffraction" }))));
                p.Children.Add(JobUI.Check(Loc.S("Jobs_Merit_WeakestField"), d.WorstField, v => Set(Loc.S("Jobs_Merit_Undo_MtfFieldRule"), x => x with { WorstField = v }),
                    Loc.S("Jobs_Merit_WeakestFieldHelp")));
                break;
        }
        var sampling = new Grid();
        sampling.Children.Add(JobUI.Stepper(Loc.S("Jobs_Merit_Rings"), d.Rings, 1, 12, n => Set(Loc.S("Jobs_Merit_Undo_PupilRings"), x => x with { Rings = n })));
        var arms = JobUI.Stepper(Loc.S("Jobs_Merit_Arms"), d.Arms, 3, 24, n => Set(Loc.S("Jobs_Merit_Undo_PupilArms"), x => x with { Arms = n }));
        arms.HorizontalAlignment = HorizontalAlignment.Right;
        sampling.Children.Add(arms);
        p.Children.Add(sampling);
        var sys = System;
        var fieldTitles = sys.Fields.Fields.Select((f, k) => $"{k + 1}: " + FieldText(f, sys)).ToList();
        var waveTitles = sys.Wavelengths.Wavelengths.Select(w => w.Um.ToString("0.0000", CultureInfo.InvariantCulture) + " µm").ToList();
        p.Children.Add(JobUI.Row(6,
            IndexChecklist(Loc.S("Jobs_Merit_Fields"), fieldTitles, d.Fields, v => Set(Loc.S("Jobs_Merit_Undo_Fields"), x => x with { Fields = v })),
            IndexChecklist(Loc.S("Jobs_Merit_Wavelengths"), waveTitles, d.Wavelengths, v => Set(Loc.S("Jobs_Merit_Undo_Wavelengths"), x => x with { Wavelengths = v }))));
        p.Children.Add(ThicknessLimits(d));
        var anamorphic = sys.Surfaces.Any(s => s.Shape.Kind is ShapeKind.Toroid or ShapeKind.Biconic);
        if (anamorphic || d.SqueezeTarget is not null)
            p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_SqueezeTarget"), new NumberField("squeeze", () => d.SqueezeTarget,
                v => Set(Loc.S("Jobs_Merit_Undo_SqueezeTarget"), x => x with { SqueezeTarget = v }), 76, Loc.S("Jobs_Merit_None"), name: Loc.S("Jobs_Merit_SqueezeTarget")),
                Loc.S("Jobs_Merit_SqueezeTargetHelp")));
        p.Children.Add(JobUI.Hint(d.Caption, 11.5));
        return p;
    }

    static string FieldText(Field f, OpticalSystem s)
    {
        var unit = s.Fields.Kind == FieldKind.ObjectAngle ? "°" : " mm";
        return f.X == 0 ? MeritFormat.Short(f.Y) + unit : $"({MeritFormat.Short(f.X)}, {MeritFormat.Short(f.Y)}){unit}";
    }

    TextBox FrequencyField(DefaultMeritSettings d)
    {
        string Text() => string.Join(", ", d.MtfFrequencies.Select(MeritFormat.Short));
        var box = new TextBox { Width = 96, MinWidth = 0, MinHeight = 28, FontSize = 12.5, Padding = new Thickness(7, 4, 7, 4), Text = Text(), PlaceholderText = "10, 30" };
        Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(box, FontNumeralAlignment.Tabular);
        AutomationProperties.SetName(box, Loc.S("Jobs_Merit_FrequenciesName"));
        void Commit()
        {
            if (box.Text == Text()) return;
            var parts = box.Text.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries);
            var values = parts.Select(LensValueParser.Number).ToList();
            if (values.Count == 0 || values.Any(v => v is not { } x || !double.IsFinite(x) || x <= 0)) { box.Text = Text(); return; }
            var sorted = values.Select(v => v!.Value).Distinct().Order().ToList();
            Set(Loc.S("Jobs_Merit_Undo_MtfFrequencies"), x => x with { MtfFrequencies = sorted });
        }
        box.LostFocus += (_, _) => Commit();
        box.KeyDown += (_, e) => { if (e.Key == VirtualKey.Enter) { Commit(); e.Handled = true; } };
        return box;
    }

    /// <summary>A menu of checkboxes over indices; every one selected is null (all).</summary>
    static DropDownButton IndexChecklist(string title, IReadOnlyList<string> all, IReadOnlyList<int>? selected, Action<IReadOnlyList<int>?> commit)
    {
        var summary = selected is null || selected.Count >= all.Count ? Loc.S("Jobs_Merit_All") : selected.Count == 1 ? $"{selected[0] + 1}"
            : Loc.F("Jobs_Merit_SomeOf", selected.Count, all.Count);
        var menu = new MenuFlyout();
        var allItem = new MenuFlyoutItem { Text = Loc.S("Jobs_Merit_All") };
        allItem.Click += (_, _) => commit(null);
        menu.Items.Add(allItem);
        menu.Items.Add(new MenuFlyoutSeparator());
        for (var k = 0; k < all.Count; k++)
        {
            var i = k;
            var on = selected?.Contains(k) ?? true;
            var item = new ToggleMenuFlyoutItem { Text = all[k], IsChecked = on };
            item.Click += (_, _) =>
            {
                var set = new SortedSet<int>(selected ?? Enumerable.Range(0, all.Count));
                if (on) set.Remove(i); else set.Add(i);
                if (set.Count == 0) return;
                commit(set.Count == all.Count ? null : set.ToList());
            };
            menu.Items.Add(item);
        }
        return new DropDownButton { Content = Loc.F("Jobs_Merit_ChecklistSummary", title, summary), Flyout = menu, FontSize = 12.5, MinHeight = 28, Padding = new Thickness(9, 3, 7, 4) };
    }

    bool _thicknessOpen;

    UIElement ThicknessLimits(DefaultMeritSettings d)
    {
        var p = new StackPanel { Spacing = 4 };
        var chevron = new Button
        {
            Style = JobUI.Style("CircleButton"), Width = 22, Height = 22, FontSize = 10, Content = _thicknessOpen ? "\uE70D" : "\uE76C",
        };
        AutomationProperties.SetName(chevron, Loc.S(_thicknessOpen ? "Jobs_Merit_HideThickness" : "Jobs_Merit_ShowThickness"));
        chevron.Click += (_, _) => { _thicknessOpen = !_thicknessOpen; var parent = (Panel)p.Parent; var at = parent.Children.IndexOf(p); parent.Children[at] = ThicknessLimits(M.Defaults); };
        p.Children.Add(JobUI.Row(2, chevron, JobUI.Check(Loc.S("Jobs_Merit_ThicknessLimits"), d.Thickness, v => Set(Loc.S("Jobs_Merit_Undo_ThicknessLimits"), x => x with { Thickness = v }))));
        if (!_thicknessOpen) return p;
        var g = new Grid { ColumnSpacing = 6, RowSpacing = 4, Margin = new Thickness(24, 0, 0, 0), IsHitTestVisible = d.Thickness, Opacity = d.Thickness ? 1 : 0.5 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(64) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(64) });
        for (var r = 0; r < 4; r++) g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        void Cell(FrameworkElement e, int row, int col) { Grid.SetRow(e, row); Grid.SetColumn(e, col); g.Children.Add(e); }
        Cell(JobUI.Text(Loc.S("Jobs_Merit_Thickness_Glass"), "CaptionText"), 0, 1);
        Cell(JobUI.Text(Loc.S("Jobs_Merit_Thickness_Air"), "CaptionText"), 0, 2);
        (string Title, Func<DefaultMeritSettings, double?> Glass, Func<DefaultMeritSettings, double?, DefaultMeritSettings> SetGlass,
            Func<DefaultMeritSettings, double?> Air, Func<DefaultMeritSettings, double?, DefaultMeritSettings> SetAir)[] rows =
        [
            (Loc.S("Jobs_Merit_Thickness_MinCenter"), x => x.GlassMinCentre, (x, v) => x with { GlassMinCentre = v }, x => x.AirMinCentre, (x, v) => x with { AirMinCentre = v }),
            (Loc.S("Jobs_Merit_Thickness_MaxCenter"), x => x.GlassMaxCentre, (x, v) => x with { GlassMaxCentre = v }, x => x.AirMaxCentre, (x, v) => x with { AirMaxCentre = v }),
            (Loc.S("Jobs_Merit_Thickness_MinEdge"), x => x.GlassMinEdge, (x, v) => x with { GlassMinEdge = v }, x => x.AirMinEdge, (x, v) => x with { AirMinEdge = v }),
        ];
        for (var i = 0; i < rows.Length; i++)
        {
            var r = rows[i];
            Cell(JobUI.Text(r.Title), i + 1, 0);
            var gf = new NumberField($"thk-g{i}", () => r.Glass(d), v => Set(Loc.S("Jobs_Merit_Undo_ThicknessLimits"), x => r.SetGlass(x, v)), 64, Loc.S("Jobs_Merit_None"), name: Loc.F("Jobs_Merit_Thickness_GlassName", r.Title.ToLowerInvariant())) { Width = double.NaN };
            var af = new NumberField($"thk-a{i}", () => r.Air(d), v => Set(Loc.S("Jobs_Merit_Undo_ThicknessLimits"), x => r.SetAir(x, v)), 64, Loc.S("Jobs_Merit_None"), name: Loc.F("Jobs_Merit_Thickness_AirName", r.Title.ToLowerInvariant())) { Width = double.NaN };
            Cell(gf, i + 1, 1);
            Cell(af, i + 1, 2);
        }
        p.Children.Add(g);
        return p;
    }

    // ---- Custom ----

    UIElement OperandTable()
    {
        var p = new StackPanel { Spacing = 8 };
        var ops = M.Operands;
        int? First() => Selection.Count == 0 ? null : ops.FindIndex(o => Selection.Contains(o.Id)) is var i and >= 0 ? i : null;
        var first = First();

        var add = new Button { Style = JobUI.Style("CircleButton"), Content = "\uE710", Width = 26, Height = 26, FontSize = 12 };
        ToolTipService.SetToolTip(add, Loc.S("Jobs_Merit_AddOperandHelp"));
        AutomationProperties.SetName(add, Loc.S("Jobs_Merit_AddOperand"));
        add.Flyout = OperandPicker(d =>
        {
            Guid id = default;
            Edit(Loc.S("Jobs_Merit_Undo_AddOperand"), m => id = m.Add(d, First() ?? (m.Operands.Count == 0 ? null : m.Operands.Count - 1)));
            Selection.Clear();
            Selection.Add(id);
            _s.ScheduleEvaluation();
        });
        var toolbar = new Grid();
        toolbar.Children.Add(JobUI.Row(0, add,
            JobUI.Icon("\uE738", Loc.S("Jobs_Merit_DeleteSelected"), DeleteSelected, Selection.Count > 0),
            JobUI.Icon("\uE8C8", Loc.S("Jobs_Merit_DuplicateSelected"), () =>
            {
                IReadOnlyList<Guid> copies = [];
                var ids = Selection.ToHashSet();
                Edit(Loc.S("Jobs_Merit_Undo_DuplicateOperand"), m => copies = m.Duplicate(ids));
                Selection.Clear();
                Selection.UnionWith(copies);
            }, Selection.Count > 0),
            JobUI.Icon("\uE70E", Loc.S("Jobs_Merit_MoveUp"), () => MoveSelection(-1), Selection.Count == 1 && first > 0),
            JobUI.Icon("\uE70D", Loc.S("Jobs_Merit_MoveDown"), () => MoveSelection(1), Selection.Count == 1 && first < ops.Count - 1)));
        var count = JobUI.Text(Loc.F(ops.Count == 1 ? "Jobs_Merit_OperandCount_One" : "Jobs_Merit_OperandCount_Many", ops.Count), "CaptionText");
        count.HorizontalAlignment = HorizontalAlignment.Right;
        toolbar.Children.Add(count);
        p.Children.Add(toolbar);

        if (ops.Count == 0)
        {
            var empty = JobUI.Hint(Loc.S("Jobs_Merit_NoOperands"));
            empty.TextAlignment = TextAlignment.Center;
            empty.Margin = new Thickness(0, 16, 0, 16);
            p.Children.Add(empty);
            return p;
        }
        var list = new StackPanel { Spacing = 0 };
        _operandList = list;
        FillOperands(list);
        var frame = new Border { Child = new ScrollViewer { Content = list, MaxHeight = 300, VerticalScrollBarVisibility = ScrollBarVisibility.Auto }, CornerRadius = new CornerRadius(6), BorderThickness = new Thickness(1) };
        JobUI.Bind(frame, "CardStrokeColorDefaultBrush", b => frame.BorderBrush = b);
        JobUI.Bind(frame, "ContentSurfaceBrush", b => frame.Background = b);
        frame.KeyDown += (_, e) =>
        {
            if (e.Key == VirtualKey.Delete) { DeleteSelected(); e.Handled = true; }
        };
        p.Children.Add(frame);
        if (Selection.Count == 1 && ops.FindIndex(o => o.Id == Selection.First()) is var k and >= 0)
            p.Children.Add(OperandDetail(k, ops[k]));
        return p;
    }

    void DeleteSelected()
    {
        if (Selection.Count == 0) return;
        var ids = Selection.ToHashSet();
        Edit(Loc.S(ids.Count == 1 ? "Jobs_Merit_Undo_DeleteOperand" : "Jobs_Merit_Undo_DeleteOperands"), m => m.Delete(ids));
        Selection.Clear();
    }

    void MoveSelection(int step)
    {
        if (Selection.Count != 1) return;
        var i = M.Operands.FindIndex(o => o.Id == Selection.First());
        if (i < 0 || i + step < 0 || i + step >= M.Operands.Count) return;
        Edit(Loc.S("Jobs_Merit_Undo_MoveOperand"), m => m.Move(i, i + step));
    }

    void Update(Guid id, string name, Action<MeritOperand> change) => Edit(name, m =>
    {
        if (m.Operands.FirstOrDefault(o => o.Id == id) is { } op) change(op);
    });

    void FillOperands(StackPanel list)
    {
        list.Children.Clear();
        var ops = M.Operands;
        for (var k = 0; k < ops.Count; k++) list.Children.Add(OperandRow(k, ops[k], k % 2 == 1));
    }

    UIElement OperandRow(int index, MeritOperand op, bool alternate)
    {
        var e = _s.Evaluation?.RowAt(index, op.Type);
        var issues = op.LocalIssues(System.Fields.Fields.Count, System.Wavelengths.Wavelengths.Count);
        var problem = e?.Error is { } err ? Loc.F("Jobs_Merit_OperandError", EngineText.T(err)) : issues.FirstOrDefault();
        var selected = Selection.Contains(op.Id);

        var g = new Grid { ColumnSpacing = 6, Padding = new Thickness(6, 4, 8, 4), Opacity = op.Enabled ? 1 : 0.45 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(20) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(70) });
        var n = JobUI.Text((index + 1).ToString(CultureInfo.InvariantCulture), "CaptionText");
        n.HorizontalAlignment = HorizontalAlignment.Right;
        n.VerticalAlignment = VerticalAlignment.Top;
        n.Margin = new Thickness(0, 2, 0, 0);
        g.Children.Add(n);
        var title = JobUI.Row(4, JobUI.Text(OperandCatalog.Title(op.Type)));
        if (op.IsConstraint)
        {
            var tag = new Border { Padding = new Thickness(5, 0, 5, 1), CornerRadius = new CornerRadius(7), Child = JobUI.Text(Loc.S("Jobs_Merit_ConstraintTag"), "CaptionText", 10) };
            JobUI.Bind(tag, "SubtleFillColorSecondaryBrush", b => tag.Background = b);
            title.Children.Add(tag);
        }
        var detail = JobUI.Text(problem ?? op.Detail(), "CaptionText");
        if (problem is not null) JobUI.Bind(detail, "WarningBrush", b => detail.Foreground = b);
        ToolTipService.SetToolTip(detail, problem ?? op.Detail());
        var texts = JobUI.Column(1, title, detail);
        Grid.SetColumn(texts, 1);
        g.Children.Add(texts);
        var value = JobUI.Text(e is null ? "—" : MeritFormat.Merit(e.Value), "NumberText");
        value.HorizontalAlignment = HorizontalAlignment.Right;
        var bar = ContributionBar(e?.Fraction ?? double.NaN);
        var right = JobUI.Column(2, value, bar);
        right.HorizontalAlignment = HorizontalAlignment.Right;
        Grid.SetColumn(right, 2);
        g.Children.Add(right);

        var row = new Border { Child = g };
        void Paint()
        {
            row.Background = selected ? ThemeLookup(row, "LensSelection") : alternate ? Support.ThemeLookup.Brush(row, "RowStripeBrush") : new SolidColorBrush(Microsoft.UI.Colors.Transparent);
        }
        row.Loaded += (_, _) => Paint();
        row.ActualThemeChanged += (_, _) => Paint();
        row.Tapped += (_, _) =>
        {
            var ctrl = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Control).HasFlag(global::Windows.UI.Core.CoreVirtualKeyStates.Down);
            if (!ctrl) Selection.Clear();
            if (!Selection.Remove(op.Id)) Selection.Add(op.Id);
            _s.NotifyChanged();
        };
        var menu = new MenuFlyout();
        void Item(string text, Action a) { var i = new MenuFlyoutItem { Text = text }; i.Click += (_, _) => a(); menu.Items.Add(i); }
        Item(Loc.S("Jobs_Merit_Menu_Duplicate"), () => { var id = op.Id; Edit(Loc.S("Jobs_Merit_Undo_DuplicateOperand"), m => m.Duplicate(new HashSet<Guid> { id })); });
        Item(Loc.S(op.Enabled ? "Jobs_Merit_Menu_Disable" : "Jobs_Merit_Menu_Enable"),
            () => Update(op.Id, Loc.S(op.Enabled ? "Jobs_Merit_Undo_DisableOperand" : "Jobs_Merit_Undo_EnableOperand"), o => o.Enabled = !o.Enabled));
        menu.Items.Add(new MenuFlyoutSeparator());
        Item(Loc.S("Jobs_Merit_Menu_Delete"), () => { var id = op.Id; Edit(Loc.S("Jobs_Merit_Undo_DeleteOperand"), m => m.Delete(new HashSet<Guid> { id })); Selection.Remove(id); });
        row.ContextFlyout = menu;
        AutomationProperties.SetName(row, Loc.F("Jobs_Merit_OperandRowName", index + 1, OperandCatalog.Title(op.Type), problem ?? op.Detail()));
        return row;
    }

    static Brush ThemeLookup(FrameworkElement e, string _)
    {
        var accent = (global::Windows.UI.Color)Support.ThemeLookup.Resource(e.ActualTheme, "SystemAccentColor");
        return new SolidColorBrush(global::Windows.UI.Color.FromArgb(e.ActualTheme == ElementTheme.Dark ? (byte)56 : (byte)36, accent.R, accent.G, accent.B));
    }

    /// <summary>A small bar with the operand's share of the merit.</summary>
    static StackPanel ContributionBar(double fraction)
    {
        var track = new Grid { Width = 30, Height = 4, VerticalAlignment = VerticalAlignment.Center };
        var back = new Border { CornerRadius = new CornerRadius(2) };
        JobUI.Bind(back, "SubtleFillColorTertiaryBrush", b => back.Background = b);
        track.Children.Add(back);
        if (double.IsFinite(fraction))
        {
            var fill = new Border { CornerRadius = new CornerRadius(2), HorizontalAlignment = HorizontalAlignment.Left, Width = Math.Max(fraction > 0 ? 2 : 0, 30 * Math.Min(fraction, 1)) };
            JobUI.Bind(fill, "AccentFillColorDefaultBrush", b => fill.Background = b);
            track.Children.Add(fill);
        }
        var pct = JobUI.Text(MeritFormat.Share(fraction), "CaptionText", 10);
        pct.Width = 32;
        pct.TextAlignment = TextAlignment.Right;
        var p = JobUI.Row(3, track, pct);
        p.HorizontalAlignment = HorizontalAlignment.Right;
        return p;
    }

    /// <summary>The searchable operand picker, grouped by category.</summary>
    static Flyout OperandPicker(Action<OperandDescriptor> pick)
    {
        var flyout = new Flyout { Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft };
        var search = new TextBox { PlaceholderText = Loc.S("Jobs_Merit_SearchOperands"), Margin = new Thickness(0, 0, 0, 8) };
        var list = new StackPanel { Spacing = 0 };
        void Fill()
        {
            list.Children.Clear();
            foreach (var (category, items) in OperandCatalog.Search(search.Text))
            {
                var head = JobUI.Text(EngineText.T(category), "ColumnHeaderText");
                head.Margin = new Thickness(6, list.Children.Count == 0 ? 0 : 8, 0, 2);
                list.Children.Add(head);
                foreach (var d in items)
                {
                    var title = new Grid();
                    title.Children.Add(JobUI.Text(EngineText.T(d.Title)));
                    if (d.Unit.Length > 0) { var u = JobUI.Text(d.Unit, "CaptionText"); u.HorizontalAlignment = HorizontalAlignment.Right; title.Children.Add(u); }
                    var content = JobUI.Column(0, title);
                    if (d.Summary.Length > 0) content.Children.Add(JobUI.Text(EngineText.T(d.Summary), "CaptionText"));
                    var b = new Button
                    {
                        Content = content, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch,
                        Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0), Padding = new Thickness(6, 3, 6, 4),
                    };
                    b.Click += (_, _) => { flyout.Hide(); pick(d); };
                    list.Children.Add(b);
                }
            }
        }
        search.TextChanged += (_, _) => Fill();
        search.KeyDown += (_, e) =>
        {
            if (e.Key != VirtualKey.Enter) return;
            if (OperandCatalog.Search(search.Text).FirstOrDefault() is { Items.Count: > 0 } g) { flyout.Hide(); pick(g.Items[0]); }
            e.Handled = true;
        };
        Fill();
        var root = new StackPanel { Width = 300 };
        root.Children.Add(search);
        root.Children.Add(new ScrollViewer { Content = list, Height = 340 });
        flyout.Content = root;
        flyout.Opened += (_, _) => search.Focus(FocusState.Programmatic);
        return flyout;
    }

    /// <summary>The selected operand's parameters, field, wavelength, comparison, target, weight, role and enable.</summary>
    UIElement OperandDetail(int index, MeritOperand op)
    {
        var sys = System;
        var id = op.Id;
        var p = new StackPanel { Spacing = 6 };
        var head = new Grid();
        head.Children.Add(JobUI.Text(Loc.F("Jobs_Merit_OperandHeading", index + 1, OperandCatalog.Title(op.Type)), "PanelTitleText"));
        if (op.Descriptor is { Unit.Length: > 0 } dd) { var u = JobUI.Text(dd.Unit, "CaptionText"); u.HorizontalAlignment = HorizontalAlignment.Right; head.Children.Add(u); }
        p.Children.Add(head);
        if (op.Descriptor is { Summary.Length: > 0 } ds) p.Children.Add(JobUI.Hint(EngineText.T(ds.Summary), 11.5));
        foreach (var param in op.Descriptor?.Params ?? []) p.Children.Add(Param(op, param, sys));
        if (op.Descriptor is null) p.Children.Add(JobUI.Hint(Loc.F("Jobs_Merit_ParametersJson", op.Kind.ToJsonString()), 11));

        // Field and wavelength.
        var single = op.Descriptor?.SingleField ?? false;
        var fieldIndex = op.Field["field"].StringOrNull() == "index" ? (int)(op.Field["index"].IntOrNull() ?? 0) : -1;
        var fields = new List<string>();
        if (!single || fieldIndex < 0) fields.Add(Loc.S("Jobs_Merit_AllFieldsWeighted"));
        var offset = fields.Count;
        fields.AddRange(sys.Fields.Fields.Select((f, k) => $"{k + 1}: " + FieldText(f, sys)));
        p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Field"), JobUI.Choice(fields, fieldIndex < 0 ? 0 : fieldIndex + offset, k => Update(id, Loc.S("Jobs_Merit_Undo_OperandField"), o =>
            o.Field = k < offset ? new JsonObject { ["field"] = "all" } : new JsonObject { ["field"] = "index", ["index"] = k - offset }), 130)));
        var w = op.Wavelength["wavelength"].StringOrNull();
        var wIndex = w == "index" ? (int)(op.Wavelength["index"].IntOrNull() ?? 0) + 2 : w == "all" ? 1 : 0;
        var waves = new List<string> { Loc.S("Jobs_Merit_PrimaryWavelength"), Loc.S("Jobs_Merit_AllWavelengths") };
        waves.AddRange(sys.Wavelengths.Wavelengths.Select(x => x.Um.ToString("0.0000", CultureInfo.InvariantCulture) + " µm"));
        p.Children.Add(JobUI.Labeled(Loc.S("Jobs_Merit_Wavelength"), JobUI.Choice(waves, wIndex, k => Update(id, Loc.S("Jobs_Merit_Undo_OperandWavelength"), o =>
            o.Wavelength = k == 0 ? new JsonObject { ["wavelength"] = "primary" } : k == 1 ? new JsonObject { ["wavelength"] = "all" }
                : new JsonObject { ["wavelength"] = "index", ["index"] = k - 2 }), 130)));

        // Comparison and target.
        var comparisons = Enum.GetValues<MeritComparison>();
        var cmp = JobUI.Segments(comparisons.Select(c => c.Symbol()).ToList(), Array.IndexOf(comparisons, op.Comparison),
            k => Update(id, Loc.S("Jobs_Merit_Undo_Comparison"), o => o.Comparison = comparisons[k]));
        ToolTipService.SetToolTip(cmp, Loc.S("Jobs_Merit_ComparisonHelp"));
        var target = new NumberField($"op-target-{id}", () => op.Target, v => Update(id, Loc.S("Jobs_Merit_Undo_Target"), o => o.Target = v ?? 0), 84, "0", name: Loc.S("Jobs_Merit_Target"));
        var targetRow = new Grid();
        targetRow.Children.Add(cmp);
        target.HorizontalAlignment = HorizontalAlignment.Right;
        targetRow.Children.Add(target);
        p.Children.Add(targetRow);

        var roleRow = new Grid();
        roleRow.Children.Add(JobUI.Check(Loc.S("Jobs_Merit_Constraint"), op.IsConstraint, v => Update(id, Loc.S(v ? "Jobs_Merit_Undo_MakeConstraint" : "Jobs_Merit_Undo_MakeResidual"), o => o.IsConstraint = v),
            Loc.S("Jobs_Merit_ConstraintHelp")));
        if (!op.IsConstraint)
        {
            var weight = JobUI.Row(6, JobUI.Text(Loc.S("Jobs_Merit_Weight"), "SecondaryText"),
                new NumberField($"op-weight-{id}", () => op.Weight, v => Update(id, Loc.S("Jobs_Merit_Undo_Weight"), o => o.Weight = Math.Max(v ?? 1, 0)), 70, "1", name: Loc.S("Jobs_Merit_Weight")));
            weight.HorizontalAlignment = HorizontalAlignment.Right;
            roleRow.Children.Add(weight);
        }
        p.Children.Add(roleRow);
        var enabledRow = new Grid();
        enabledRow.Children.Add(JobUI.Check(Loc.S("Jobs_Merit_Enabled"), op.Enabled, v => Update(id, Loc.S(v ? "Jobs_Merit_Undo_EnableOperand" : "Jobs_Merit_Undo_DisableOperand"), o => o.Enabled = v)));
        if (_s.Evaluation?.RowAt(index, op.Type) is { } e)
        {
            var value = JobUI.Text(double.IsFinite(e.Fraction) ? Loc.F("Jobs_Merit_OperandValueShare", MeritFormat.Short(e.Value), MeritFormat.Share(e.Fraction))
                : Loc.F("Jobs_Merit_OperandValue", MeritFormat.Short(e.Value)), "NumberText");
            value.HorizontalAlignment = HorizontalAlignment.Right;
            enabledRow.Children.Add(value);
            if (e.Error is { } err) p.Children.Add(JobUI.Warning(EngineText.T(err)));
        }
        p.Children.Add(enabledRow);
        foreach (var issue in op.LocalIssues(sys.Fields.Fields.Count, sys.Wavelengths.Wavelengths.Count)) p.Children.Add(JobUI.Warning(issue));
        var card = new Border { Child = p, Padding = new Thickness(10), CornerRadius = new CornerRadius(8) };
        JobUI.Bind(card, "SubtleFillColorSecondaryBrush", b => card.Background = b);
        return card;
    }

    UIElement Param(MeritOperand op, OperandParam param, OpticalSystem sys)
    {
        var id = op.Id;
        var v = op.Kind[param.Key];
        var title = EngineText.T(param.Title);
        var name = Loc.F("Jobs_Merit_Undo_Param", title);
        void SetKind(JsonNode? value) => Update(id, name, o => { if (value is null) o.Kind.Remove(param.Key); else o.Kind[param.Key] = value; });
        switch (param.Kind)
        {
            case OperandParamKind.Number:
                var f = new NumberField($"op-{id}-{param.Key}", () => v.DoubleValue(), x => SetKind(x is { } d ? JsonValue.Create(d) : param.Required ? JsonValue.Create(0.0) : null),
                    76, param.Required ? "0" : Loc.S("Jobs_Merit_Auto"), name: title);
                return JobUI.Labeled(title, param.Unit.Length > 0 ? JobUI.Row(5, f, JobUI.Text(param.Unit, "CaptionText")) : f);
            case OperandParamKind.Integer:
                return JobUI.Labeled(title, new NumberField($"op-{id}-{param.Key}", () => v.DoubleValue(),
                    x => SetKind(JsonValue.Create((long)Math.Max(Math.Round(x ?? param.Minimum), param.Minimum))), 76, param.Minimum.ToString(CultureInfo.InvariantCulture), MeritFormat.Short, optional: false, name: title));
            case OperandParamKind.Surface:
                var titles = new List<string>();
                var ids = new List<ulong?>();
                if (!param.Required) { titles.Add(Loc.S(param.Key == "to" ? "Jobs_Merit_Surface_Next" : param.Key == "from" ? "Jobs_Merit_Surface_First" : "Jobs_Merit_Surface_Image")); ids.Add(null); }
                for (var i = 1; i < sys.Surfaces.Count; i++)
                {
                    titles.Add(i == sys.ImageIndex ? Loc.S("Jobs_Merit_Surface_Image")
                        : sys.Surfaces[i].Label.Length == 0 ? Loc.F("Jobs_Merit_Surface", i) : Loc.F("Jobs_Merit_SurfaceLabeled", i, sys.Surfaces[i].Label));
                    ids.Add(sys.Surfaces[i].Id.Value);
                }
                var at = ids.IndexOf(v.UInt64OrNull());
                return JobUI.Labeled(title, JobUI.Choice(titles, Math.Max(at, 0), k => SetKind(ids[k] is { } sid ? JsonValue.Create(sid) : null), 130));
            case OperandParamKind.Wavelength:
                return JobUI.Labeled(title, JobUI.Choice(sys.Wavelengths.Wavelengths.Select(x => x.Um.ToString("0.0000", CultureInfo.InvariantCulture) + " µm").ToList(),
                    (int)(v.IntOrNull() ?? 0), k => SetKind(JsonValue.Create(k)), 130));
            case OperandParamKind.Choice:
                var options = param.Options ?? [];
                var current = Math.Max(0, options.ToList().IndexOf(v.StringOrNull() ?? options.FirstOrDefault() ?? ""));
                return JobUI.Labeled(title, JobUI.Choice(options.Select(OperandCatalog.OptionTitle).ToList(), current,
                    k => SetKind(JsonValue.Create(options[k])), 130));
            case OperandParamKind.Flag:
                return JobUI.Check(title, v.BoolOrNull() ?? false, x => SetKind(JsonValue.Create(x)));
            case OperandParamKind.Sampling:
                var rings = (int)(v?["rings"].IntOrNull() ?? 3);
                var arms = (int)(v?["arms"].IntOrNull() ?? 6);
                JsonObject Sampling(int r, int a)
                {
                    var o = v is JsonObject old ? (JsonObject)old.DeepClone() : [];
                    o["rings"] = r;
                    o["arms"] = a;
                    return o;
                }
                var g = new Grid();
                g.Children.Add(JobUI.Stepper(Loc.S("Jobs_Merit_Rings"), rings, 1, 12, r => SetKind(Sampling(r, arms))));
                var armsStepper = JobUI.Stepper(Loc.S("Jobs_Merit_Arms"), arms, 3, 24, a => SetKind(Sampling(rings, a)));
                armsStepper.HorizontalAlignment = HorizontalAlignment.Right;
                g.Children.Add(armsStepper);
                return g;
            default:
                // Operand numbers, 1-based in the text, 0-based stored.
                var box = new TextBox { Width = 96, MinWidth = 0, MinHeight = 28, FontSize = 12.5, PlaceholderText = "1, 2" };
                string Text() => string.Join(", ", (v.IntArray() ?? []).Select(k => (k + 1).ToString(CultureInfo.InvariantCulture)));
                box.Text = Text();
                void Commit()
                {
                    if (box.Text == Text()) return;
                    var parts = box.Text.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries);
                    var values = parts.Select(x => int.TryParse(x, NumberStyles.Integer, CultureInfo.InvariantCulture, out var n) && n >= 1 ? n - 1 : -1).ToList();
                    if (values.Any(x => x < 0)) { box.Text = Text(); return; }
                    SetKind(new JsonArray([.. values.Select(x => (JsonNode)x)]));
                }
                box.LostFocus += (_, _) => Commit();
                box.KeyDown += (_, e) => { if (e.Key == VirtualKey.Enter) { Commit(); e.Handled = true; } };
                ToolTipService.SetToolTip(box, Loc.S("Jobs_Merit_OperandListHelp"));
                return JobUI.Labeled(title, box);
        }
    }
}
