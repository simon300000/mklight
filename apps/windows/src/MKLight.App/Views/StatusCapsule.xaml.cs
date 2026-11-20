using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;

namespace MKLight.App.Views;

/// <summary>A validation issue as the capsule lists it.</summary>
public sealed record IssueItem(string Severity, string Message, string? Where, int? Surface = null);

/// <summary>
/// The status capsule in the title bar's centre (the Mac app's DesignStatus):
/// activity, the key first-order numbers and the issues button.
/// </summary>
public sealed partial class StatusCapsule : UserControl
{
    public StatusCapsule() { InitializeComponent(); }

    /// <summary>Raised when an issue that names a surface is clicked.</summary>
    public event Action<IssueItem>? IssueInvoked;

    public void SetActivity(string? text)
    {
        Activity.Visibility = text is null ? Visibility.Collapsed : Visibility.Visible;
        ActivityText.Text = text ?? "";
        ActivityText.Visibility = string.IsNullOrEmpty(text) ? Visibility.Collapsed : Visibility.Visible;
    }

    /// <summary>The key numbers: (label, value) pairs, or a single caption when there are none.</summary>
    public void SetNumbers(IReadOnlyList<(string Label, string Value)> numbers, string? caption = null)
    {
        Numbers.Children.Clear();
        foreach (var (label, value) in numbers)
        {
            var row = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5 };
            row.Children.Add(new TextBlock { Text = label, Style = (Style)Application.Current.Resources["SecondaryText"], FontSize = 12 });
            row.Children.Add(new TextBlock { Text = value, Style = (Style)Application.Current.Resources["NumberText"], FontSize = 12.5 });
            Numbers.Children.Add(row);
        }
        if (numbers.Count == 0)
            Numbers.Children.Add(new TextBlock { Text = caption ?? "", Style = (Style)Application.Current.Resources["SecondaryText"], FontSize = 12 });
    }

    /// <summary>Rows of the details flyout: (section or label, value); a null value is a section header.</summary>
    public void SetDetails(IEnumerable<(string Label, string? Value)> rows)
    {
        Details.Children.Clear();
        foreach (var (label, value) in rows)
        {
            if (value is null)
            {
                Details.Children.Add(new TextBlock
                {
                    Text = label, Style = (Style)Application.Current.Resources["PanelTitleText"],
                    Margin = new Thickness(0, Details.Children.Count == 0 ? 0 : 10, 0, 2),
                });
                continue;
            }
            var g = new Grid { ColumnSpacing = 12 };
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            g.Children.Add(new TextBlock { Text = label, Style = (Style)Application.Current.Resources["SecondaryText"] });
            var v = new TextBlock { Text = value, Style = (Style)Application.Current.Resources["NumberText"], TextAlignment = TextAlignment.Right, IsTextSelectionEnabled = true };
            Grid.SetColumn(v, 1);
            g.Children.Add(v);
            Details.Children.Add(g);
        }
    }

    IReadOnlyList<IssueItem> _issues = [];

    public void SetIssues(IReadOnlyList<IssueItem> issues)
    {
        _issues = issues;
        var errors = issues.Count(i => i.Severity == "error");
        if (issues.Count == 0)
        {
            IssuesGlyph.Glyph = "\uE930";
            IssuesGlyph.Foreground = ThemeLookup.Brush(this, "TextFillColorTertiaryBrush");
            IssuesCount.Visibility = Visibility.Collapsed;
            ToolTipService.SetToolTip(IssuesButton, "The engine found no problems in the design");
        }
        else
        {
            IssuesGlyph.Glyph = errors > 0 ? "\uEA39" : "\uE7BA";
            var brush = ThemeLookup.Brush(this, errors > 0 ? "ErrorBrush" : "WarningBrush");
            IssuesGlyph.Foreground = brush;
            IssuesCount.Foreground = brush;
            IssuesCount.Text = issues.Count.ToString(System.Globalization.CultureInfo.InvariantCulture);
            IssuesCount.Visibility = Visibility.Visible;
            ToolTipService.SetToolTip(IssuesButton, $"{issues.Count} issue{(issues.Count == 1 ? "" : "s")}: show the design's validation issues");
        }
    }

    public void ShowIssues() => OnIssuesClick(IssuesButton, new RoutedEventArgs());

    void OnIssuesClick(object sender, RoutedEventArgs e)
    {
        IssuesList.Children.Clear();
        if (_issues.Count == 0)
            IssuesList.Children.Add(new TextBlock { Text = "No issues. The engine found no problems in the design.", Style = (Style)Application.Current.Resources["SecondaryText"] });
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
        FlyoutBase.ShowAttachedFlyout(IssuesButton);
    }
}
