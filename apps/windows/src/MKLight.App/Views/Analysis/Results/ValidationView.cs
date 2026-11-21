using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// <c>validate</c>: the design's issues, errors first, each with its severity glyph, message, code,
/// and the surface (a link that selects it in the lens data) and field it concerns; "No Issues"
/// when the design passed every check.
/// </summary>
public sealed partial class ValidationView : UserControl
{
    readonly ValidationData _d;
    readonly List<(FontIcon Icon, ValidationData.Severity Severity)> _icons = [];

    public ValidationView(ValidationData d)
    {
        _d = d;
        IsTabStop = false;
        Content = d.Issues.Count == 0 ? NoIssues() : List();
        AutomationProperties.SetName(this, Loc.F("Results_Validation_Name", d.Summary()));
        ActualThemeChanged += (_, _) => Paint();
        Loaded += (_, _) => Paint();
    }

    static FrameworkElement NoIssues()
    {
        var s = new StackPanel { Spacing = 6, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center, Padding = new Thickness(16) };
        var icon = new FontIcon { Glyph = "\uE930", FontSize = 30, HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(0, 0, 0, 4) };
        s.Children.Add(icon);
        var title = Kit.Text(Loc.S("Results_Validation_NoIssuesTitle"), "PanelTitleText", 15);
        title.HorizontalAlignment = HorizontalAlignment.Center;
        s.Children.Add(title);
        var note = Kit.Caption(Loc.S("Results_Validation_Passed"));
        note.HorizontalAlignment = HorizontalAlignment.Center;
        s.Children.Add(note);
        AutomationProperties.SetName(s, Loc.S("Results_Validation_PassedName"));
        return s;
    }

    FrameworkElement List()
    {
        var stack = new StackPanel { Spacing = 2, Padding = new Thickness(10, 8, 12, 12) };
        var head = Kit.Text(_d.Summary(), "PanelTitleText", 12);
        head.Margin = new Thickness(4, 0, 0, 6);
        AutomationProperties.SetHeadingLevel(head, Microsoft.UI.Xaml.Automation.Peers.AutomationHeadingLevel.Level2);
        stack.Children.Add(head);
        foreach (var i in _d.Issues) stack.Children.Add(Row(i));
        return Kit.Page(stack);
    }

    FrameworkElement Row(ValidationData.Issue issue)
    {
        var g = new Grid { ColumnSpacing = 10, Padding = new Thickness(4, 5, 4, 5) };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        var icon = new FontIcon
        {
            Glyph = issue.Severity switch { ValidationData.Severity.Error => "\uEB90", ValidationData.Severity.Warning => "\uE7BA", _ => "\uE946" },
            FontSize = 14, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0),
        };
        _icons.Add((icon, issue.Severity));
        g.Children.Add(icon);
        var text = new StackPanel { Spacing = 2 };
        var message = Kit.Text(issue.Text, "BodyText", 12.5);
        message.TextWrapping = TextWrapping.Wrap;
        message.TextTrimming = TextTrimming.None;
        message.IsTextSelectionEnabled = true;
        text.Children.Add(message);
        var meta = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 10 };
        if (issue.Code.Length > 0)
        {
            var code = Kit.Caption(issue.Code);
            code.FontFamily = new FontFamily("Cascadia Mono, Consolas");
            meta.Children.Add(code);
        }
        if (issue.Surface is int s)
        {
            var link = new HyperlinkButton { Content = Loc.F("Results_Validation_Surface", s), FontSize = 11.5, Padding = new Thickness(0), MinHeight = 0 };
            link.Click += (_, _) => Select(s);
            ToolTipService.SetToolTip(link, Loc.F("Results_Validation_SelectSurface", s));
            meta.Children.Add(link);
        }
        if (issue.Field is int f) meta.Children.Add(Kit.Caption(Loc.F("Results_Chart_FieldNumber", f + 1)));
        if (meta.Children.Count > 0) text.Children.Add(meta);
        Grid.SetColumn(text, 1);
        g.Children.Add(text);
        var severity = issue.Severity switch
        {
            ValidationData.Severity.Error => Loc.S("Results_Validation_SeverityError"),
            ValidationData.Severity.Warning => Loc.S("Results_Validation_SeverityWarning"),
            _ => Loc.S("Results_Validation_SeverityInfo"),
        };
        AutomationProperties.SetName(g, issue.Surface is int t ? Loc.F("Results_Validation_IssueNameSurface", severity, issue.Text, t) : Loc.F("Results_Validation_IssueName", severity, issue.Text));
        return g;
    }

    void Select(int surface)
    {
        if (SettingsServices.Window(this) is { } host) host.Session.SelectedSurface = surface;
    }

    void Paint()
    {
        foreach (var (icon, severity) in _icons)
            icon.Foreground = ThemeLookup.Brush(this, severity switch
            {
                ValidationData.Severity.Error => "SystemFillColorCriticalBrush",
                ValidationData.Severity.Warning => "SystemFillColorCautionBrush",
                _ => "TextFillColorSecondaryBrush",
            });
        if (Content is StackPanel { Children: [FontIcon ok, ..] }) ok.Foreground = ThemeLookup.Brush(this, "SystemFillColorSuccessBrush");
    }
}
