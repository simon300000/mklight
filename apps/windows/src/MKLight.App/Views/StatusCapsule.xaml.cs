using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.Model;

namespace MKLight.App.Views;

/// <summary>A validation issue as the capsule lists it.</summary>
public sealed record IssueItem(string Severity, string Message, string? Where, int? Surface = null);

/// <summary>
/// The status capsule in the title bar's centre (the Mac app's DesignStatus):
/// activity, the key first-order numbers and the issues button.
/// </summary>
public sealed partial class StatusCapsule : UserControl
{
    public StatusCapsule()
    {
        InitializeComponent();
        ToolTipService.SetToolTip(NumbersButton, Loc.S("Shell_Status_NumbersTip"));
        ToolTipService.SetToolTip(BackendPicker, Loc.S("Shell_Status_BackendTip"));
    }

    /// <summary>Raised when an issue that names a surface is clicked.</summary>
    public event Action<IssueItem>? IssueInvoked;

    Action? _activityOpen;

    /// <summary>What a click on the activity does (a running job: open its page).</summary>
    public void SetActivityAction(Action? open)
    {
        _activityOpen = open;
        if (!_activityHooked)
        {
            _activityHooked = true;
            Activity.Tapped += (_, _) => _activityOpen?.Invoke();
        }
        ToolTipService.SetToolTip(Activity, open is null ? null : Loc.S("Shell_Status_ShowRunningJob"));
    }
    bool _activityHooked;

    public void SetActivity(string? text)
    {
        Activity.Visibility = text is null ? Visibility.Collapsed : Visibility.Visible;
        ActivityText.Text = text ?? "";
        ActivityText.Visibility = string.IsNullOrEmpty(text) ? Visibility.Collapsed : Visibility.Visible;
    }

    /// <summary>The GPU kernel preparation: "Preparing GPU n%" with a spinner, "GPU ready", or nothing (null).</summary>
    public void SetGpu(string? text, bool preparing)
    {
        GpuBadge.Visibility = text is null ? Visibility.Collapsed : Visibility.Visible;
        GpuText.Text = text ?? "";
        GpuRing.IsActive = text is not null && preparing;
        GpuRing.Visibility = GpuRing.IsActive ? Visibility.Visible : Visibility.Collapsed;
        ToolTipService.SetToolTip(GpuBadge, preparing ? ViewModels.GpuPreparationMonitor.BadgeHelp : null);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(GpuBadge, text ?? "");
    }

    /// <summary>Opens the details flyout (launch actions for screenshots).</summary>
    /// <param name="keepOpenFor">Shows it again when it closes within this time (another window taking the focus while a screenshot waits).</param>
    public void ShowDetails(TimeSpan keepOpenFor = default)
    {
        var flyout = NumbersButton.Flyout;
        var until = DateTime.UtcNow + keepOpenFor;
        void Reopen(object? s, object e)
        {
            if (DateTime.UtcNow < until) DispatcherQueue.TryEnqueue(() => flyout.ShowAt(NumbersButton));
            else flyout.Closed -= Reopen;
        }
        if (keepOpenFor > TimeSpan.Zero) flyout.Closed += Reopen;
        flyout.ShowAt(NumbersButton);
    }

    /// <summary>The key numbers: (label, value) pairs, or a single caption when there are none; a stale result is dimmed.</summary>
    public void SetNumbers(IReadOnlyList<(string Label, string Value)> numbers, string? caption = null, bool stale = false)
    {
        Numbers.Opacity = stale ? 0.5 : 1;
        Numbers.Children.Clear();
        foreach (var (label, value) in numbers)
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
            row.Children.Add(new TextBlock { Text = label, Style = (Style)Application.Current.Resources["SecondaryText"], FontSize = 12 });
            var number = new TextBlock { Text = value, Style = (Style)Application.Current.Resources["NumberText"], FontSize = 12.5 };
            // The id stays English in every language (the squeeze label is translated).
            var id = label switch { "EFL" or "EFL y" => "EFL", "F/#" => "FNumber", _ when label == Loc.S("Shell_Status_Squeeze") => "Squeeze", _ => label };
            Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(number, "Status." + id);
            row.Children.Add(number);
            Numbers.Children.Add(row);
        }
        if (numbers.Count == 0)
            Numbers.Children.Add(new TextBlock { Text = caption ?? "", Style = (Style)Application.Current.Resources["SecondaryText"], FontSize = 12 });
    }

    static Style StyleOf(string key) => (Style)Application.Current.Resources[key];

    /// <summary>The details flyout: sections, rows, the x / y table of an anamorphic design, notes and warnings.</summary>
    public void SetDetails(IEnumerable<StatusDetail> rows)
    {
        Details.Children.Clear();
        foreach (var row in rows)
        {
            FrameworkElement e;
            switch (row.Kind)
            {
                case StatusDetail.Kinds.Header:
                    e = new TextBlock { Text = row.Label, Style = StyleOf("PanelTitleText"), Margin = new Thickness(0, Details.Children.Count == 0 ? 0 : 10, 0, 2) };
                    break;
                case StatusDetail.Kinds.Note:
                    e = new TextBlock { Text = row.Label, Style = StyleOf("CaptionText"), TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None, IsTextSelectionEnabled = true };
                    break;
                case StatusDetail.Kinds.Warning:
                {
                    var g = new Grid { ColumnSpacing = 6 };
                    g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                    g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                    g.Children.Add(new FontIcon { Glyph = "\uE7BA", FontSize = 12, Foreground = ThemeLookup.Brush(this, "WarningBrush"), VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) });
                    var t = new TextBlock { Text = row.Label, Style = StyleOf("CaptionText"), TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None, IsTextSelectionEnabled = true };
                    Grid.SetColumn(t, 1);
                    g.Children.Add(t);
                    e = g;
                    break;
                }
                case StatusDetail.Kinds.XyHeader:
                case StatusDetail.Kinds.Xy:
                {
                    var head = row.Kind == StatusDetail.Kinds.XyHeader;
                    var g = new Grid { ColumnSpacing = 12 };
                    g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                    g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(90) });
                    g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(90) });
                    g.Children.Add(new TextBlock { Text = row.Label, Style = StyleOf("SecondaryText") });
                    var col = 1;
                    foreach (var value in new[] { row.Value, row.Value2 })
                    {
                        var v = new TextBlock
                        {
                            Text = value ?? "", Style = StyleOf(head ? "CaptionText" : "NumberText"), TextAlignment = TextAlignment.Right,
                            HorizontalAlignment = HorizontalAlignment.Right, IsTextSelectionEnabled = !head,
                        };
                        Grid.SetColumn(v, col++);
                        g.Children.Add(v);
                    }
                    if (head) Microsoft.UI.Xaml.Automation.AutomationProperties.SetAccessibilityView(g, Microsoft.UI.Xaml.Automation.Peers.AccessibilityView.Raw);
                    else Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(g, Loc.F("Shell_Status_XyName", row.Help ?? row.Label, row.Value, row.Value2));
                    e = g;
                    break;
                }
                default:
                {
                    var g = new Grid { ColumnSpacing = 12 };
                    g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
                    g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                    g.Children.Add(new TextBlock { Text = row.Label, Style = StyleOf("SecondaryText") });
                    var v = new TextBlock { Text = row.Value ?? "", Style = StyleOf("NumberText"), TextAlignment = TextAlignment.Right, IsTextSelectionEnabled = true, MaxWidth = 230, TextWrapping = TextWrapping.Wrap };
                    Grid.SetColumn(v, 1);
                    g.Children.Add(v);
                    e = g;
                    break;
                }
            }
            if (row.Help is not null) ToolTipService.SetToolTip(e, row.Help);
            Details.Children.Add(e);
        }
    }

    IReadOnlyList<IssueItem> _issues = [];

    public void SetIssues(IReadOnlyList<IssueItem> issues)
    {
        var changed = !issues.SequenceEqual(_issues);
        _issues = issues;
        if (changed && FlyoutBase.GetAttachedFlyout(IssuesButton) is { IsOpen: true }) FillIssues();
        var errors = issues.Count(i => i.Severity == "error");
        if (issues.Count == 0)
        {
            IssuesGlyph.Glyph = "\uE930";
            IssuesGlyph.Foreground = ThemeLookup.Brush(this, "TextFillColorTertiaryBrush");
            IssuesCount.Visibility = Visibility.Collapsed;
            ToolTipService.SetToolTip(IssuesButton, Loc.S("Shell_Status_NoProblemsTip"));
        }
        else
        {
            IssuesGlyph.Glyph = errors > 0 ? "\uEA39" : "\uE7BA";
            var brush = ThemeLookup.Brush(this, errors > 0 ? "ErrorBrush" : "WarningBrush");
            IssuesGlyph.Foreground = brush;
            IssuesCount.Foreground = brush;
            IssuesCount.Text = issues.Count.ToString(System.Globalization.CultureInfo.InvariantCulture);
            IssuesCount.Visibility = Visibility.Visible;
            ToolTipService.SetToolTip(IssuesButton, Loc.F(issues.Count == 1 ? "Shell_Status_IssuesTipOne" : "Shell_Status_IssuesTipMany", issues.Count));
        }
    }

    Action? _endPreview;

    /// <summary>The previewed candidate's title (null: the design is shown) and how to end the preview.</summary>
    public void SetPreview(string? title, Action? endPreview)
    {
        _endPreview = endPreview;
        PreviewBadge.Visibility = title is null ? Visibility.Collapsed : Visibility.Visible;
        // The capsule stays compact (the panes and the layout carry the full title); the title is the tooltip.
        PreviewText.Text = title is null ? "" : Loc.S("Shell_Status_Previewing");
        ToolTipService.SetToolTip(PreviewBadge, title is null ? null : Loc.F("Shell_Status_PreviewingTip", title));
    }

    void OnEndPreview(object sender, RoutedEventArgs e) => _endPreview?.Invoke();

    Action<int>? _backendChanged;
    bool _settingBackend;

    IReadOnlyList<MKLight.Engine.ComputeBackendOption> _backends = [];

    /// <summary>
    /// The compute backend picker in the details flyout: the engine's backends by title, the
    /// unavailable ones disabled with their reason; a note under it says why a chosen backend is not used.
    /// </summary>
    public void SetBackends(IReadOnlyList<MKLight.Engine.ComputeBackendOption> options, int selected, Action<int> changed)
    {
        _settingBackend = true;
        _backendChanged = changed;
        if (!options.SequenceEqual(_backends))
        {
            _backends = options;
            BackendPicker.Items.Clear();
            foreach (var o in options)
            {
                // The option texts come from the engine library (MKLight.Engine): through the engine text lookup.
                var title = EngineText.T(o.Title);
                var reason = o.Reason is null ? null : EngineText.T(o.Reason);
                var content = new StackPanel { Spacing = 0 };
                content.Children.Add(new TextBlock { Text = title, Style = StyleOf("NumberText") });
                if (reason is not null)
                    content.Children.Add(new TextBlock { Text = reason, Style = StyleOf("CaptionText"), MaxWidth = 280, TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None });
                var item = new ComboBoxItem { Content = content, IsEnabled = o.Enabled };
                ToolTipService.SetToolTip(item, reason ?? EngineText.T(o.Help));
                Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(item, reason is null ? title : Loc.F("Shell_Status_BackendNotAvailableName", title, reason));
                BackendPicker.Items.Add(item);
            }
        }
        BackendPicker.SelectedIndex = selected;
        var chosen = selected >= 0 && selected < options.Count ? options[selected] : null;
        BackendNote.Text = chosen is { Enabled: false } ? Loc.F("Shell_Status_BackendNote", chosen.Reason is null ? "" : EngineText.T(chosen.Reason)) : "";
        BackendNote.Visibility = BackendNote.Text.Length > 0 ? Visibility.Visible : Visibility.Collapsed;
        _settingBackend = false;
    }

    void OnBackendChanged(object sender, SelectionChangedEventArgs e)
    {
        if (!_settingBackend && BackendPicker.SelectedIndex >= 0) _backendChanged?.Invoke(BackendPicker.SelectedIndex);
    }

    public void ShowIssues() => OnIssuesClick(IssuesButton, new RoutedEventArgs());

    void OnIssuesClick(object sender, RoutedEventArgs e)
    {
        FillIssues();
        FlyoutBase.ShowAttachedFlyout(IssuesButton);
    }

    /// <summary>The issues list (again while it is open, when the issues change: <c>--show-issues</c> opens it before validation).</summary>
    void FillIssues()
    {
        IssuesList.Children.Clear();
        if (_issues.Count == 0)
            IssuesList.Children.Add(new TextBlock { Text = Loc.S("Shell_Status_NoIssues"), Style = (Style)Application.Current.Resources["SecondaryText"] });
        foreach (var issue in _issues.OrderBy(i => i.Severity == "error" ? 0 : i.Severity == "warning" ? 1 : 2))
        {
            var row = new Grid { ColumnSpacing = 10 };
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            var isError = issue.Severity == "error";
            row.Children.Add(new FontIcon
            {
                Glyph = isError ? "\uEA39" : issue.Severity == "warning" ? "\uE7BA" : "\uE946", FontSize = 14,
                Foreground = ThemeLookup.Brush(this, isError ? "ErrorBrush" : issue.Severity == "warning" ? "WarningBrush" : "TextFillColorSecondaryBrush"),
                VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0),
            });
            var text = new StackPanel { Spacing = 1 };
            text.Children.Add(new TextBlock { Text = issue.Message, Style = (Style)Application.Current.Resources["BodyText"], TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None });
            if (issue.Where is not null)
                text.Children.Add(new TextBlock { Text = issue.Where, Style = (Style)Application.Current.Resources["CaptionText"] });
            Grid.SetColumn(text, 1);
            row.Children.Add(text);
            var button = new Button { Content = row, Background = new SolidColorBrush(Microsoft.UI.Colors.Transparent), BorderThickness = new Thickness(0), HorizontalAlignment = HorizontalAlignment.Stretch, HorizontalContentAlignment = HorizontalAlignment.Left, Padding = new Thickness(6) };
            button.Click += (_, _) => IssueInvoked?.Invoke(issue);
            IssuesList.Children.Add(button);
        }
    }
}
