using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using Microsoft.UI.Xaml.Media.Animation;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Panels;
using Windows.Foundation;

namespace MKLight.App.Views.Analysis;

/// <summary>
/// The pane's chrome around the result: the facts it decides from, the
/// updating pulse, folding the header on narrow panes, the More menu and
/// the automation names and ids.
/// </summary>
public sealed partial class AnalysisPaneView
{
    readonly ScalarTransition? _fade;
    Storyboard? _pulse;
    TextBlock? _measure;
    bool _compact;

    /// <summary>Launch actions: show the chrome for these facts instead of the pane's own (null: the pane's).</summary>
    internal PaneFacts? ForcedFacts { get; set; }

    PaneFacts Facts()
    {
        if (ForcedFacts is { } forced) return forced;
        var ws = _session.Workspace;
        return new PaneFacts
        {
            HasOutput = _pane.Output is not null,
            Running = _pane.IsRunning,
            Failed = _pane.Phase == PanePhase.Failed,
            Unavailable = _pane.Phase == PanePhase.Unavailable,
            Cancelled = _pane.Phase == PanePhase.Cancelled,
            AutoUpdate = _pane.AutoUpdate,
            Message = _pane.Message,
            Stage = _pane.Progress.Stage,
            DesignHasErrors = ws.Issues.Any(i => i.Severity == "error"),
            ModelError = ws.ModelErrors.FirstOrDefault(),
        };
    }

    /// <summary>Re-reads the pane (launch actions after forcing a state).</summary>
    internal void Refresh() => Update();

    protected override AutomationPeer OnCreateAutomationPeer() => new AnalysisPaneGroupPeer(this, "AnalysisPane");

    // ---- The updating pulse ----

    /// <summary>While a result updates it breathes gently; with Reduce Motion it stays still.</summary>
    void SetPulse(bool on)
    {
        if (on == _pulse is not null) return;
        if (on)
        {
            var a = new DoubleAnimation
            {
                From = 1, To = 0.72, Duration = TimeSpan.FromSeconds(0.9), AutoReverse = true,
                RepeatBehavior = RepeatBehavior.Forever, EasingFunction = new SineEase { EasingMode = EasingMode.EaseInOut },
            };
            Storyboard.SetTarget(a, PulseHost);
            Storyboard.SetTargetProperty(a, "Opacity");
            _pulse = new Storyboard();
            _pulse.Children.Add(a);
            _pulse.Begin();
        }
        else
        {
            _pulse!.Stop();
            _pulse = null;
            PulseHost.Opacity = 1;
        }
    }

    /// <summary>Windows' animation setting changed (any thread): a running pulse stops, or starts again.</summary>
    void OnMotionChanged() => DispatcherQueue.TryEnqueue(Update);

    /// <summary>Whether the pulse runs now (launch actions check Reduce Motion with it).</summary>
    internal bool IsPulsing => _pulse is not null;

    // ---- Narrow panes ----

    bool HasSettings => _session.Workspace.Descriptor(_pane.Analysis) is not { } d || d.Available && d.SettingsFields.Count > 0;

    double TextWidth(string text)
    {
        _measure ??= new TextBlock { Style = TitleText.Style };
        _measure.Text = text;
        _measure.Measure(new Size(double.PositiveInfinity, double.PositiveInfinity));
        // Off the tree a text may not measure: an estimate of the title's semibold 13 px.
        return _measure.DesiredSize.Width > 0 ? _measure.DesiredSize.Width : text.Length * 7.2;
    }

    /// <summary>
    /// As the pane narrows (the Mac's order): the cluster folds into Run and More (always below
    /// <see cref="AnalysisPaneChrome.FoldWidth"/>), then the title takes its short form.
    /// </summary>
    void UpdateFold()
    {
        var full = AnalysisCatalog.Title(_pane.Analysis, _session.Workspace.Descriptor(_pane.Analysis)?.Title);
        var shortTitle = AnalysisCatalog.ShortTitle(_pane.Analysis);
        var titled = TitleText.Visibility == Visibility.Visible; // as a tab, the tab carries the title
        var hasSettings = HasSettings;
        const double button = 28;
        // Header padding 16, column spacing 6, the title button's padding 6, glyph 14, two spacings 14, chevron 9.
        const double chrome = 16 + 6 + 6 + 14 + 14 + 9 + 3;
        var (compact, useShort) = AnalysisPaneChrome.Fold(ActualWidth,
            titled ? TextWidth(full) : 0, titled ? TextWidth(shortTitle) : 0,
            button * (hasSettings ? 4 : 3), button * 2, chrome);
        _compact = compact;
        AutoButton.Visibility = compact ? Visibility.Collapsed : Visibility.Visible;
        SettingsButton.Visibility = compact || !hasSettings ? Visibility.Collapsed : Visibility.Visible;
        TitleText.Text = useShort ? shortTitle : full;
    }

    // ---- More ----

    /// <summary>
    /// More: the toggles of the cluster (the only place for them on a narrow pane), Duplicate Pane,
    /// Copy Result as JSON, then the panel's own commands: move to another window, close.
    /// </summary>
    void OnMoreOpening(object? sender, object e)
    {
        MoreMenu.Items.Clear();
        foreach (var item in MoreItems()) MoreMenu.Items.Add(item);
    }

    List<MenuFlyoutItemBase> MoreItems()
    {
        var items = new List<MenuFlyoutItemBase>();
        var a = _pane.Analysis;
        var auto = new ToggleMenuFlyoutItem { Text = Loc.S("Analysis_More_AutoUpdate"), IsChecked = _pane.AutoUpdate, Icon = new FontIcon { Glyph = "\uE945" } };
        auto.Click += (_, _) => SetAutoUpdate(auto.IsChecked);
        Identify(auto, a, "More.AutoUpdate");
        items.Add(auto);
        if (HasSettings)
        {
            var settings = new ToggleMenuFlyoutItem { Text = Loc.S("Analysis_More_ShowSettings"), IsChecked = _pane.ShowsSettings, Icon = new FontIcon { Glyph = "\uE9E9" } };
            settings.Click += (_, _) => SetShowsSettings(settings.IsChecked);
            Identify(settings, a, "More.ShowSettings");
            items.Add(settings);
        }
        items.Add(new MenuFlyoutSeparator());
        items.Add(Identify(Item(Loc.S("Analysis_More_Duplicate"), Duplicate), a, "More.Duplicate"));
        var copy = Item(Loc.S("Analysis_More_CopyJson"), CopyJson);
        copy.IsEnabled = _pane.Output is not null;
        items.Add(Identify(copy, a, "More.CopyJson"));
        items.Add(new MenuFlyoutSeparator());
        var panel = PanelId.Analysis(_pane.Id);
        if (Host is { } host)
        {
            if (host.Arrangement.IsDetached(panel)) items.Add(Identify(Item(Loc.S("Analysis_More_MoveToMain"), () => host.MoveToMainWindow(panel)), a, "More.MoveToMain"));
            if (!host.Arrangement.IsAloneInWindow(panel)) items.Add(Identify(Item(Loc.S("Analysis_More_MoveToNewWindow"), () => host.MoveToNewWindow(panel)), a, "More.MoveToNewWindow"));
        }
        items.Add(Identify(Item(Loc.S("Analysis_More_Close"), Close), a, "More.Close"));
        return items;
    }

    static MenuFlyoutItem Item(string text, Action click)
    {
        var item = new MenuFlyoutItem { Text = text };
        item.Click += (_, _) => click();
        return item;
    }

    /// <summary>The panel host the pane is shown by (its tree's, in the main or a separate window).</summary>
    PanelHost? Host
    {
        get
        {
            for (DependencyObject? e = this; e is not null; e = VisualTreeHelper.GetParent(e))
                if (e is PanelTreeView tree) return tree.Host;
            return null;
        }
    }

    /// <summary>
    /// Launch actions: the More menu as it opens, for a window capture. A menu is a popup that a capture of the
    /// window leaves out (constrained to the window or not), so the capture shows the same items in a menu
    /// presenter placed in the pane, under the More button.
    /// </summary>
    internal async Task ShowMoreMenuForCaptureAsync()
    {
        var presenter = new MenuFlyoutPresenter
        {
            HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Top,
            Margin = new Thickness(0, Header.ActualHeight - 2, 4, 0), MinWidth = 200,
            Translation = new System.Numerics.Vector3(0, 0, 32), Shadow = new ThemeShadow(),
        };
        // What the flyout sets when it opens: the in-app acrylic, and room for the checks.
        presenter.Background = ThemeLookup.Brush(this, "AcrylicInAppFillColorDefaultBrush");
        var items = MoreItems();
        foreach (var item in items) presenter.Items.Add(item);
        Grid.SetRowSpan(presenter, 5);
        Root.Children.Add(presenter);
        VisualStateManager.GoToState(MoreButton, "PointerOver", false);
        await Task.Delay(200);
        foreach (var item in items.OfType<MenuFlyoutItem>()) VisualStateManager.GoToState(item, "CheckPlaceholder", false);
        await Task.Delay(300);
    }
    /// <summary>Whether the header's cluster is folded into More.</summary>
    internal bool IsCompact => _compact;

    // ---- Automation ----

    static T Identify<T>(T element, string analysis, string part) where T : DependencyObject
    {
        AutomationProperties.SetAutomationId(element, AnalysisPaneChrome.AutomationId(analysis, part));
        return element;
    }

    /// <summary>Names and stable ids (<c>Pane.{analysis}.{part}</c>) for screen readers and UI tests.</summary>
    void UpdateAutomation(string title, string? preview)
    {
        var a = _pane.Analysis;
        AutomationProperties.SetName(this, AnalysisPaneChrome.AccessibleName(title, preview));
        AutomationProperties.SetAutomationId(this, "Pane." + a);
        Identify(Header, a, "Header");
        AutomationProperties.SetName(Header, Loc.F("Analysis_Pane_HeaderName", title));
        Identify(TitleButton, a, "Title");
        AutomationProperties.SetName(TitleButton, title);
        AutomationProperties.SetHelpText(TitleButton, Loc.S("Analysis_Pane_TitleHelp"));
        Identify(StateText, a, "State");
        Identify(RunButton, a, "Run");
        Identify(AutoButton, a, "AutoUpdate");
        Identify(SettingsButton, a, "Settings");
        Identify(MoreButton, a, "More");
        Identify(MessagePanel, a, "Placeholder");
        Identify(PreviewBanner, a, "PreviewBanner");
        Identify(StaleUpdate, a, "Update");
        Identify(FailedCapsule, a, "RerunFailed");
        Identify(BackendBadge, a, "Backend");
    }
}

/// <summary>A grid that UI automation sees as a named group (a pane's header, placeholder, preview banner).</summary>
public sealed partial class AnalysisPaneGroup : Grid
{
    protected override AutomationPeer OnCreateAutomationPeer() => new AnalysisPaneGroupPeer(this, "AnalysisPaneGroup");
}

/// <summary>The automation peer of a pane and its groups: a group control with its name and id.</summary>
public sealed partial class AnalysisPaneGroupPeer(FrameworkElement owner, string className) : FrameworkElementAutomationPeer(owner)
{
    protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.Group;
    protected override string GetClassNameCore() => className;
    protected override bool IsControlElementCore() => true;
}
