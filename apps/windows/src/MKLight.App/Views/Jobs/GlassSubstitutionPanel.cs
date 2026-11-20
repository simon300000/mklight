using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Engine;
using MKLight.Model;
using Windows.System;

namespace MKLight.App.Views.Jobs;

/// <summary>
/// The Glass Substitution page (the Mac app's <c>GlassSubstitutionPanel</c>): the lenses whose glass may change and
/// the candidates (for every lens, or per lens), a quick or thorough search that re-optimises the shape for every
/// combination, live progress, then the ranked combinations: the selected one is previewed in the window, compared
/// with the design (glasses, merit, spot numbers, shape changes, the two layouts), shown on the glass map, and applied
/// as one undoable step.
/// </summary>
public sealed partial class GlassSubstitutionPanel : UserControl
{
    readonly GlassSubstitutionSession _s;
    readonly DesignSession _session;
    readonly JobPanelFrame _frame = new("Glass Substitution", "\uEB42");
    IReadOnlyList<CatalogInfo> _catalogs = [];
    bool _queued;
    JobPhase? _builtPhase;
    StackPanel? _comparison;

    public GlassSubstitutionPanel(DesignSession session)
    {
        _session = session;
        _s = DesignJobs.Of(session).Glass;
        Content = _frame;
        _s.Changed += Queue;
        session.Document.Changed += Queue;
        session.Workspace.PreviewChanged += Queue;
        session.Workspace.CatalogsChanged += () => _ = LoadCatalogs();
        session.Workspace.StateChanged += () =>
        {
            if (_comparison is not null && _s.Selected is { } c && _s.Result is { } r) FillComparison(_comparison, c, r);
        };
        Loaded += async (_, _) => { Rebuild(); await LoadCatalogs(); };
    }

    async Task LoadCatalogs()
    {
        try { _catalogs = await _session.Workspace.Engine.CatalogsAsync(); }
        catch (EngineException) { _catalogs = []; }
        Queue();
    }

    void Queue()
    {
        if (_queued) return;
        _queued = true;
        DispatcherQueue.TryEnqueue(() => { _queued = false; Rebuild(); });
    }

    void Rebuild()
    {
        var offset = _builtPhase == _s.Phase ? _frame.Scroller.VerticalOffset : 0;
        _builtPhase = _s.Phase;
        var n = _s.Settings.Selected.Count;
        _frame.Subtitle = n == 0 ? "Snap model glasses to real glasses" : $"{n} glass{(n == 1 ? "" : "es")} may change";
        _comparison = null;
        _frame.Body = _s.Phase switch
        {
            JobPhase.Running => Running(),
            JobPhase.Finished => Results(),
            JobPhase.Failed => JobUI.Failure("Glass Substitution Failed", _s.Failure ?? ""),
            _ => Setup(),
        };
        Actions();
        if (offset > 0) DispatcherQueue.TryEnqueue(() => _frame.Scroller.ChangeView(null, offset, null, disableAnimation: true));
    }

    // ---- Setup ----

    UIElement Setup()
    {
        var st = _s.Settings;
        var p = new StackPanel { Spacing = 16, Padding = new Thickness(0, 0, 0, 8) };

        var slots = _s.Slots;
        var g = new StackPanel { Spacing = 4 };
        if (slots.Count == 0) g.Children.Add(JobUI.Hint("The design has no glass to replace."));
        foreach (var slot in slots) g.Children.Add(SlotRow(slot));
        var all = new HyperlinkButton { Content = "All", Padding = new Thickness(4, 2, 4, 2), FontSize = 12 };
        all.Click += (_, _) => { st.Selected.UnionWith(slots.Select(x => x.Id)); _s.NotifyChanged(); };
        var none = new HyperlinkButton { Content = "None", Padding = new Thickness(4, 2, 4, 2), FontSize = 12 };
        none.Click += (_, _) => { st.Selected.Clear(); _s.NotifyChanged(); };
        g.Children.Add(JobUI.Row(4, all, none));
        if (st.Selected.Count == 0) g.Children.Add(JobUI.Hint("With no glass ticked, every model glass (n_d, V_d) snaps to its nearest real glasses."));
        if (st.OffsetNote(_session.Document.System) is { } note) g.Children.Add(JobUI.Hint(note));
        p.Children.Add(JobUI.Group("Glasses That May Change", g));

        p.Children.Add(JobUI.Group("Candidates for Every Lens", CandidateForm(() => st.Candidates, c => { st.Candidates = c; _s.NotifyChanged(); }, "all")));

        var search = new StackPanel { Spacing = 6 };
        var algs = Enum.GetValues<GlassSearch>();
        search.Children.Add(JobUI.Labeled("Search", JobUI.Choice(algs.Select(GlassSubstitutionSettings.Title).ToList(), Array.IndexOf(algs, st.Algorithm),
            k => { st.Algorithm = algs[k]; _s.NotifyChanged(); }, 150)));
        search.Children.Add(JobUI.Hint(GlassSubstitutionSettings.Detail(st.Algorithm), 11.5));
        search.Children.Add(JobUI.Labeled("Combinations to try", Int("combinations", () => st.MaxCombinations, v => st.MaxCombinations = Math.Max(1, v))));
        search.Children.Add(JobUI.Labeled("Time limit", JobUI.Row(5, new NumberField("time", () => st.MaxSeconds,
            v => { st.MaxSeconds = v is > 0 ? v : null; _s.NotifyChanged(); }, 80, "none", name: "Time limit in seconds"), JobUI.Text("s", "SecondaryText"))));
        search.Children.Add(JobUI.Labeled("Shape iterations each", Int("local", () => st.LocalIterations, v => st.LocalIterations = Math.Max(1, v)),
            "Damped least-squares iterations that re-optimize the shape for each glass combination"));
        search.Children.Add(JobUI.Labeled("Keep the best", Int("top", () => st.TopN, v => st.TopN = Math.Max(1, v))));
        p.Children.Add(JobUI.Group("Search", search));

        var merit = new StackPanel { Spacing = 6 };
        var m = MeritSettings.FromSection(_session.Document.Section("merit"));
        merit.Children.Add(JobUI.Hint((m.Mode == MeritMode.Custom ? $"Custom merit, {m.Operands.Count} operands" : "Default merit: " + Summary(m.Defaults))
            + (m.HoldEfl ? ", focal length held." : "."), 12.5));
        var vars = DesignVariables.Of(_session.Document);
        if (vars.Count == 0) merit.Children.Add(JobUI.Hint("Each combination re-optimizes every curvature and the back focus."));
        else
        {
            var design = new RadioButton { Content = $"Re-optimize the design variables ({vars.Count(v => !v.IsPinned)})", IsChecked = st.UseDesignVariables, GroupName = "gs-vary", MinWidth = 0 };
            var auto = new RadioButton { Content = "Re-optimize every curvature and the back focus", IsChecked = !st.UseDesignVariables, GroupName = "gs-vary", MinWidth = 0 };
            design.Checked += (_, _) => { st.UseDesignVariables = true; };
            auto.Checked += (_, _) => { st.UseDesignVariables = false; };
            merit.Children.Add(JobUI.Column(0, design, auto));
        }
        var edit = new HyperlinkButton { Content = "Edit Merit and Variables…", Padding = new Thickness(0, 2, 0, 2), FontSize = 12.5 };
        edit.Click += (_, _) => _session.ShowInspectorPage("optimize");
        merit.Children.Add(edit);
        p.Children.Add(JobUI.Group("Merit", merit));
        if (_session.Workspace.ModelErrors.Count > 0) p.Children.Add(JobUI.Warning("Fix the design's errors first."));
        return p;
    }

    static string Summary(DefaultMeritSettings d)
    {
        var s = d.Criterion == MeritCriterion.Mtf
            ? $"MTF ≥ {MeritFormat.Short(d.MtfTarget)} at {string.Join(", ", d.MtfFrequencies.Select(MeritFormat.Short))} cyc/mm" + (d.WorstField ? ", weakest field" : "")
            : DefaultMeritSettings.Title(d.Criterion);
        s += $" · {d.Rings} rings × {d.Arms} arms";
        if (d.Thickness) s += " · thickness limits";
        if (d.SqueezeTarget is { } q) s += $" · squeeze {MeritFormat.Short(q)}";
        return s;
    }

    NumberField Int(string key, Func<int> get, Action<int> set) =>
        new(key, () => get(), v => { set((int)Math.Round(v ?? get())); _s.NotifyChanged(); }, 80, "", MeritFormat.Short, optional: false, name: key);

    UIElement SlotRow(GlassSlot slot)
    {
        var st = _s.Settings;
        var head = JobUI.Row(6, JobUI.Text($"S{slot.Index}", "SecondaryNumberText"), JobUI.Text(slot.Material));
        ((TextBlock)head.Children[1]).FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
        if (slot.IsModel)
        {
            var tag = new Border { Padding = new Thickness(5, 0, 5, 1), CornerRadius = new CornerRadius(7), Child = JobUI.Text("model", "CaptionText", 10), VerticalAlignment = VerticalAlignment.Center };
            JobUI.Bind(tag, "SubtleFillColorSecondaryBrush", b => tag.Background = b);
            head.Children.Add(tag);
        }
        if (!slot.Offset.IsZero)
        {
            var o = JobUI.Text(LensFormat.Offset(slot.Offset), "CaptionText");
            ToolTipService.SetToolTip(o, "Melt offset Δn_d / ΔV_d: a replacement glass is installed without it");
            head.Children.Add(o);
        }
        var content = JobUI.Column(0, head);
        if (st.Overrides.TryGetValue(slot.Id, out var own))
        {
            var t = JobUI.Text(own.Summary, "CaptionText");
            JobUI.Bind(t, "AccentTextFillColorPrimaryBrush", b => t.Foreground = b);
            content.Children.Add(t);
        }
        var check = new CheckBox { Content = content, IsChecked = st.Selected.Contains(slot.Id), MinWidth = 0 };
        check.Click += (_, _) =>
        {
            if (check.IsChecked == true) st.Selected.Add(slot.Id); else st.Selected.Remove(slot.Id);
            _s.NotifyChanged();
        };
        var overrideButton = JobUI.Icon(own is null ? "\uE9E9" : "\uE71C", $"Candidates for surface {slot.Index} only", () => { }, true, 26);
        overrideButton.Flyout = OverrideFlyout(slot);
        var g = new Grid();
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.Children.Add(check);
        Grid.SetColumn(overrideButton, 1);
        g.Children.Add(overrideButton);
        return g;
    }

    Flyout OverrideFlyout(GlassSlot slot)
    {
        var st = _s.Settings;
        var flyout = new Flyout();
        var body = new StackPanel { Spacing = 10, Width = 300 };
        void Fill()
        {
            body.Children.Clear();
            body.Children.Add(JobUI.Text($"Candidates for S{slot.Index} · {slot.Material}", "PanelTitleText"));
            body.Children.Add(JobUI.Check("Own candidates for this lens", st.Overrides.ContainsKey(slot.Id), on =>
            {
                if (on) st.Overrides[slot.Id] = st.Candidates; else st.Overrides.Remove(slot.Id);
                Fill();
            }));
            if (st.Overrides.ContainsKey(slot.Id))
                body.Children.Add(CandidateForm(() => st.Overrides.GetValueOrDefault(slot.Id) ?? st.Candidates, c => { st.Overrides[slot.Id] = c; Fill(); }, $"s{slot.Index}"));
            else body.Children.Add(JobUI.Hint("Uses the candidates for every lens."));
        }
        Fill();
        flyout.Content = body;
        flyout.Closed += (_, _) => _s.NotifyChanged();
        return flyout;
    }

    /// <summary>Which catalogue glasses are candidates, in plain terms.</summary>
    StackPanel CandidateForm(Func<GlassCandidates> get, Action<GlassCandidates> set, string key)
    {
        var c = get();
        var p = new StackPanel { Spacing = 6 };

        var catalogs = new MenuFlyout();
        var allItem = new MenuFlyoutItem { Text = "All loaded" };
        allItem.Click += (_, _) => set(get() with { Catalogs = [] });
        catalogs.Items.Add(allItem);
        catalogs.Items.Add(new MenuFlyoutSeparator());
        foreach (var cat in _catalogs)
        {
            var on = c.Catalogs.Contains(cat.Name);
            var item = new ToggleMenuFlyoutItem { Text = $"{cat.Name} ({cat.GlassCount.ToString(CultureInfo.InvariantCulture)})", IsChecked = on };
            var name = cat.Name;
            item.Click += (_, _) =>
            {
                var now = get();
                set(now with { Catalogs = on ? now.Catalogs.Where(x => x != name).ToList() : [.. now.Catalogs, name] });
            };
            catalogs.Items.Add(item);
        }
        p.Children.Add(JobUI.Labeled("Catalogs", Menu(c.Catalogs.Count == 0 ? "All loaded" : string.Join(", ", c.Catalogs), catalogs)));

        var statuses = new MenuFlyout();
        foreach (var s in GlassCandidates.AllStatuses)
        {
            var on = c.Statuses.Contains(s);
            var item = new ToggleMenuFlyoutItem { Text = Title(s), IsChecked = on };
            item.Click += (_, _) =>
            {
                var now = get();
                var next = new HashSet<string>(now.Statuses);
                if (on && next.Count > 1) next.Remove(s); else next.Add(s);
                set(now with { Statuses = next });
            };
            statuses.Items.Add(item);
        }
        p.Children.Add(JobUI.Labeled("Status", Menu(string.Join(", ", GlassCandidates.AllStatuses.Where(c.Statuses.Contains).Select(Title)), statuses)));

        p.Children.Add(JobUI.Labeled("Choose", JobUI.Choice(["The nearest glasses", "Glasses within a window"], c.Rule == CandidateRule.Nearest ? 0 : 1,
            k => set(get() with { Rule = k == 0 ? CandidateRule.Nearest : CandidateRule.Window }), 150)));
        if (c.Rule == CandidateRule.Nearest)
            p.Children.Add(JobUI.Stepper("Nearest", c.Nearest, 1, 200, n => set(get() with { Nearest = n })));
        else
            p.Children.Add(JobUI.Row(6,
                JobUI.Text("n_d ±", "SecondaryText"), new NumberField($"{key}-wnd", () => c.WindowNd, v => set(get() with { WindowNd = Math.Max(v ?? 0.05, 1e-4) }), 64, name: "n_d window"),
                JobUI.Text("V_d ±", "SecondaryText"), new NumberField($"{key}-wvd", () => c.WindowVd, v => set(get() with { WindowVd = Math.Max(v ?? 5, 0.1) }), 56, name: "V_d window")));
        p.Children.Add(JobUI.Labeled("Always include", Names(c.Include, "N-BK7, SCHOTT:F2", v => set(get() with { Include = v }))));
        p.Children.Add(JobUI.Labeled("Never use", Names(c.Exclude, "names", v => set(get() with { Exclude = v }))));
        var limits = new Grid { ColumnSpacing = 8, RowSpacing = 4 };
        limits.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        limits.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        (string Title, string Help, double? Value, Func<GlassCandidates, double?, GlassCandidates> With)[] rows =
        [
            ("Min. transmission", "Internal transmission at the shortest wavelength through the lens's center thickness (0…1)", c.MinTransmission, (x, v) => x with { MinTransmission = v }),
            ("Max. relative cost", "Catalog relative cost", c.MaxRelativeCost, (x, v) => x with { MaxRelativeCost = v }),
            ("Max. climate class", "Climate resistance class (CR)", c.MaxClimateClass, (x, v) => x with { MaxClimateClass = v }),
            ("Max. stain class", "Stain resistance class (FR)", c.MaxStainClass, (x, v) => x with { MaxStainClass = v }),
        ];
        for (var i = 0; i < rows.Length; i++)
        {
            var r = rows[i];
            limits.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var t = JobUI.Text(r.Title);
            ToolTipService.SetToolTip(t, r.Help);
            Grid.SetRow(t, i);
            limits.Children.Add(t);
            var f = new NumberField($"{key}-lim{i}", () => r.Value, v => set(r.With(get(), v)), 70, "none", name: r.Title);
            Grid.SetRow(f, i);
            Grid.SetColumn(f, 1);
            limits.Children.Add(f);
        }
        var expander = new Expander { Header = "Limits", Content = limits, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch, Padding = new Thickness(10, 6, 10, 6), MinHeight = 0 };
        p.Children.Add(expander);
        return p;
    }

    static string Title(string status) => char.ToUpperInvariant(status[0]) + status[1..];

    static DropDownButton Menu(string title, MenuFlyout flyout) => new()
    {
        Content = new TextBlock { Text = title, MaxWidth = 130, TextTrimming = TextTrimming.CharacterEllipsis }, Flyout = flyout, FontSize = 12.5, MinHeight = 28, Padding = new Thickness(9, 3, 7, 4),
    };

    static TextBox Names(IReadOnlyList<string> names, string placeholder, Action<IReadOnlyList<string>> commit)
    {
        string Text() => string.Join(", ", names);
        var box = new TextBox { Width = 140, MinWidth = 0, MinHeight = 28, FontSize = 12.5, Text = Text(), PlaceholderText = placeholder };
        void Commit() { if (box.Text != Text()) commit(GlassCandidates.Names(box.Text)); }
        box.LostFocus += (_, _) => Commit();
        box.KeyDown += (_, e) => { if (e.Key == VirtualKey.Enter) { Commit(); e.Handled = true; } };
        return box;
    }

    // ---- Running ----

    UIElement Running()
    {
        var p = new StackPanel { Spacing = 14 };
        var live = _s.Live;
        p.Children.Add(JobUI.Row(28,
            JobUI.BigNumber("Combinations", live is null ? "—" : $"{live.Combinations} / {_s.Settings.MaxCombinations}"),
            JobUI.BigNumber("Best merit", live is null ? "—" : MeritFormat.Merit(live.BestMerit))));
        if (live is { Glasses.Count: > 0 })
        {
            var slots = _s.Slots.Where(x => _s.Settings.Selected.Contains(x.Id)).ToList();
            if (slots.Count == 0) slots = _s.Slots.Where(x => x.IsModel).ToList();
            var chips = new VariableSizedWrapGrid { Orientation = Orientation.Horizontal, ItemHeight = 24, ItemWidth = 92 };
            foreach (var (slot, name) in slots.Zip(live.Glasses))
            {
                var chip = new Border { CornerRadius = new CornerRadius(10), Padding = new Thickness(7, 2, 7, 2), Margin = new Thickness(0, 0, 4, 4), Child = JobUI.Text($"S{slot.Index} {name}", "NumberText", 11.5) };
                JobUI.Bind(chip, "SubtleFillColorSecondaryBrush", b => chip.Background = b);
                chips.Children.Add(chip);
            }
            p.Children.Add(JobUI.Column(4, JobUI.Text("Best glasses so far", "CaptionText"), chips));
        }
        p.Children.Add(new Sparkline(110) { Values = _s.Trace.ToList() });
        p.Children.Add(JobUI.Hint("Stop keeps the combinations ranked so far. Cancel discards the run. You can close this panel; the search continues."));
        return p;
    }

    // ---- Results ----

    UIElement Results()
    {
        var p = new StackPanel { Spacing = 14 };
        if (_s.Result is not { } r) return p;
        if (_s.Applied) p.Children.Add(JobUI.AppliedNotice());
        p.Children.Add(JobUI.Hint(r.StatusText(_s.Applied)));
        if (r.Ranked.Count == 0)
        {
            p.Children.Add(JobUI.Empty("\uE9CE", "No Combinations", "No combination could be evaluated."));
            return p;
        }
        var list = new StackPanel();
        for (var k = 0; k < r.Ranked.Count; k++) list.Children.Add(RankRow(r.Ranked[k], r, k % 2 == 1));
        var frame = new Border { Child = new ScrollViewer { Content = list, MaxHeight = 280 }, CornerRadius = new CornerRadius(6), BorderThickness = new Thickness(1), IsHitTestVisible = !_s.Applied };
        JobUI.Bind(frame, "CardStrokeColorDefaultBrush", b => frame.BorderBrush = b);
        JobUI.Bind(frame, "ContentSurfaceBrush", b => frame.Background = b);
        p.Children.Add(frame);
        if (_s.Selected is { } c)
        {
            p.Children.Add(Comparison(c, r));
            if (_s.StartSystem is { } start)
                p.Children.Add(new CandidateGlassMap(_session.Workspace, start, c, () => _s.MapPgf, on => _s.MapPgf = on));
        }
        return p;
    }

    UIElement RankRow(GlassCombination c, GlassSubstitutionResult r, bool alternate)
    {
        var selected = c.Rank == _s.SelectedRank;
        var g = new Grid { ColumnSpacing = 8, Padding = new Thickness(6, 5, 8, 5) };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(18) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var rank = JobUI.Text(c.Rank.ToString(CultureInfo.InvariantCulture), "NumberText");
        rank.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
        rank.HorizontalAlignment = HorizontalAlignment.Right;
        rank.VerticalAlignment = VerticalAlignment.Top;
        g.Children.Add(rank);
        var names = JobUI.Text(string.Join(" · ", c.Glasses.Select(x => x.Name)));
        ToolTipService.SetToolTip(names, names.Text);
        var texts = JobUI.Column(1, names, JobUI.Text(c.ChangeSummary, "CaptionText"));
        Grid.SetColumn(texts, 1);
        g.Children.Add(texts);
        var merit = JobUI.Text(MeritFormat.Merit(c.Merit), "NumberText");
        merit.HorizontalAlignment = HorizontalAlignment.Right;
        var right = JobUI.Column(1, merit);
        if (!c.Feasible)
        {
            var t = JobUI.Text("constraints not met", "CaptionText");
            JobUI.Bind(t, "WarningBrush", b => t.Foreground = b);
            right.Children.Add(t);
        }
        else if (r.StartingMerit > 0 && double.IsFinite(c.Merit))
        {
            var t = JobUI.Text(MeritFormat.Percent(1 - c.Merit / r.StartingMerit, 0), "CaptionText");
            t.HorizontalAlignment = HorizontalAlignment.Right;
            if (c.Merit < r.StartingMerit) JobUI.Bind(t, "SystemFillColorSuccessBrush", b => t.Foreground = b);
            right.Children.Add(t);
        }
        Grid.SetColumn(right, 2);
        g.Children.Add(right);
        var row = new Border { Child = g };
        void Paint()
        {
            if (selected)
            {
                var accent = (global::Windows.UI.Color)Support.ThemeLookup.Resource(row.ActualTheme, "SystemAccentColor");
                row.Background = new SolidColorBrush(global::Windows.UI.Color.FromArgb(row.ActualTheme == ElementTheme.Dark ? (byte)56 : (byte)36, accent.R, accent.G, accent.B));
            }
            else row.Background = alternate ? Support.ThemeLookup.Brush(row, "RowStripeBrush") : new SolidColorBrush(Microsoft.UI.Colors.Transparent);
        }
        row.Loaded += (_, _) => Paint();
        row.ActualThemeChanged += (_, _) => Paint();
        row.Tapped += (_, _) => _s.Select(c.Rank);
        AutomationProperties.SetName(row, $"Rank {c.Rank}, merit {MeritFormat.Merit(c.Merit)}, glasses {string.Join(", ", c.Glasses.Select(x => x.Name))}");
        return row;
    }

    /// <summary>Before / after: glasses per surface, merit, the spot numbers and the shape changes.</summary>
    UIElement Comparison(GlassCombination c, GlassSubstitutionResult r)
    {
        var sys = _session.Document.System;
        var p = new StackPanel { Spacing = 8 };
        var head = new Grid();
        head.Children.Add(JobUI.Text($"Candidate {c.Rank}", "PanelTitleText"));
        if (!_s.Applied)
        {
            var t = JobUI.Text(_session.Workspace.IsPreviewing ? "Previewed in the window" : "Not previewed", "CaptionText");
            JobUI.Bind(t, "AccentTextFillColorPrimaryBrush", b => t.Foreground = b);
            t.HorizontalAlignment = HorizontalAlignment.Right;
            head.Children.Add(t);
        }
        p.Children.Add(head);
        var glasses = new Grid { ColumnSpacing = 6, RowSpacing = 3 };
        glasses.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        glasses.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        glasses.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        glasses.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        for (var i = 0; i < c.Glasses.Count; i++)
        {
            var x = c.Glasses[i];
            glasses.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            var idx = sys.SurfaceIndexOf(x.Surface) is { } k ? k.ToString(CultureInfo.InvariantCulture) : "?";
            var after = JobUI.Text(x.Name);
            after.FontWeight = Microsoft.UI.Text.FontWeights.SemiBold;
            ToolTipService.SetToolTip(after, $"n_d {x.Nd.ToString("F4", CultureInfo.InvariantCulture)} · V_d {x.Vd.ToString("F1", CultureInfo.InvariantCulture)}");
            var cells = new FrameworkElement[]
            {
                JobUI.Text($"S{idx}", "SecondaryNumberText"), JobUI.Text(_s.StartGlasses.GetValueOrDefault(x.Surface) ?? "—"),
                new FontIcon { Glyph = "\uE72A", FontSize = 10, VerticalAlignment = VerticalAlignment.Center },
                JobUI.Column(0, after, JobUI.Text($"{x.Nd.ToString("F4", CultureInfo.InvariantCulture)} / {x.Vd.ToString("F1", CultureInfo.InvariantCulture)}", "CaptionText", 10.5)),
            };
            for (var col = 0; col < cells.Length; col++)
            {
                Grid.SetRow(cells[col], i);
                Grid.SetColumn(cells[col], col);
                glasses.Children.Add(cells[col]);
            }
        }
        p.Children.Add(glasses);
        _comparison = new StackPanel { Spacing = 4 };
        FillComparison(_comparison, c, r);
        p.Children.Add(JobUI.Card(_comparison, 8));
        if (c.Moved.Count > 0)
        {
            var moves = new StackPanel { Spacing = 2 };
            foreach (var ch in c.Moved)
            {
                var row = new Grid();
                row.Children.Add(JobUI.Text(ch.Param?.Label(sys) ?? "?", "CaptionText"));
                var v = JobUI.Text($"{MeritFormat.Short(OptimizeResult.Displayed(ch.Start, ch.Param))} → {MeritFormat.Short(OptimizeResult.Displayed(ch.Value, ch.Param))}", "CaptionText");
                v.HorizontalAlignment = HorizontalAlignment.Right;
                row.Children.Add(v);
                moves.Children.Add(row);
            }
            p.Children.Add(new Expander { Header = $"Shape changes ({c.Moved.Count})", Content = moves, HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Stretch });
        }
        // The design's layout and the candidate's (once applied, the design is the candidate's).
        p.Children.Add(new LayoutComparison(_session.Workspace, _s.StartSystem, c.System, afterTitle: _s.Applied ? "Applied" : "Candidate"));
        var card = new Border { Child = p, Padding = new Thickness(10), CornerRadius = new CornerRadius(10) };
        JobUI.Bind(card, "SubtleFillColorSecondaryBrush", b => card.Background = b);
        return card;
    }

    void FillComparison(StackPanel card, GlassCombination c, GlassSubstitutionResult r) =>
        PreviewComparison.Fill(card, _session.Workspace, _session.Workspace.DesignSystem ?? _session.Document.System, $"Candidate {c.Rank}",
            [("Merit", MeritFormat.Merit(r.StartingMerit), MeritFormat.Merit(c.Merit))]);

    // ---- Actions ----

    void Actions()
    {
        switch (_s.Phase)
        {
            case JobPhase.Setup:
                _frame.SetActions(null, JobUI.Primary("Find Glasses", "\uE768", () => _s.Start(), "Start the search (Ctrl+Enter)", !_s.CannotRun));
                break;
            case JobPhase.Failed:
                _frame.SetActions(null, JobUI.Secondary("Back", null, _s.Reset, "Back to the job's settings"),
                    JobUI.Primary("Try Again", "\uE72C", () => _s.Start(fresh: true), "Run the job again with the same settings", !_s.CannotRun));
                break;
            case JobPhase.Running:
                _frame.SetActions(JobUI.Progress(_s.Progress, _s.Live is { } l ? $"{l.Combinations} combinations" : "Preparing"),
                    JobUI.Secondary("Cancel", null, _s.Cancel, "End the run and discard it", destructive: true),
                    JobUI.Secondary("Stop", "\uE71A", _s.Stop, "End now and keep the combinations ranked so far", _s.CanStop));
                break;
            default:
                if (_s.Applied) _frame.SetActions(null, JobUI.Primary("Done", null, _s.Reset, "Back to the glasses and the search settings"));
                else
                    _frame.SetActions(null, JobUI.Secondary("Discard", null, _s.Reset, "Drop this result; the design stays as it is"),
                        JobUI.Primary("Apply to Design", null, _s.Apply, "Apply the previewed combination as one step you can undo", _s.Selected?.System is not null));
                break;
        }
    }
}
