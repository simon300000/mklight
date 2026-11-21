using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using Windows.UI;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>
/// While a candidate design is previewed, which of the two a result view shows: a small
/// segmented "Design | Candidate" control, the candidate (selected at first) in the accent the
/// preview is marked with everywhere else, the design on a quiet fill.
/// </summary>
public sealed partial class CompareSwitch : UserControl
{
    readonly Button _design, _candidate;
    readonly Border _box;

    public CompareSwitch()
    {
        IsTabStop = false;
        VerticalAlignment = VerticalAlignment.Center;
        _design = Segment(Loc.S("Results_Compare_Design"), false);
        _candidate = Segment(Loc.S("Results_Compare_Candidate"), true);
        var row = new StackPanel { Orientation = Orientation.Horizontal };
        row.Children.Add(_design);
        row.Children.Add(_candidate);
        _box = new Border { Child = row, CornerRadius = new CornerRadius(12), BorderThickness = new Thickness(1), Padding = new Thickness(2) };
        Content = _box;
        AutomationProperties.SetName(this, Loc.S("Results_Compare_Name"));
        ToolTipService.SetToolTip(this, Loc.S("Results_Compare_ToolTip"));
        ActualThemeChanged += (_, _) => Restyle();
        Loaded += (_, _) => Restyle();
    }

    /// <summary>True while the candidate is shown (the start).</summary>
    public bool ShowsCandidate { get; private set; } = true;

    public event Action<bool>? Changed;

    public void Select(bool candidate)
    {
        if (candidate == ShowsCandidate) return;
        ShowsCandidate = candidate;
        Restyle();
        Changed?.Invoke(candidate);
    }

    Button Segment(string title, bool candidate)
    {
        var b = new Button
        {
            Content = title, FontSize = 11, Padding = new Thickness(9, 1, 9, 2), MinHeight = 20, MinWidth = 0,
            CornerRadius = new CornerRadius(10), BorderThickness = new Thickness(0),
        };
        b.Click += (_, _) => Select(candidate);
        AutomationProperties.SetName(b, title);
        return b;
    }

    void Restyle()
    {
        var dark = ActualTheme == ElementTheme.Dark;
        var ink = dark ? Color.FromArgb(0xE6, 255, 255, 255) : Color.FromArgb(0xE4, 0, 0, 0);
        var quiet = dark ? Color.FromArgb(0x99, 255, 255, 255) : Color.FromArgb(0x8F, 0, 0, 0);
        var fill = dark ? Color.FromArgb(0x24, 255, 255, 255) : Color.FromArgb(0x10, 0, 0, 0);
        _box.BorderBrush = new SolidColorBrush(dark ? Color.FromArgb(0x1A, 255, 255, 255) : Color.FromArgb(0x17, 0, 0, 0));
        var clear = new SolidColorBrush(Color.FromArgb(0, 0, 0, 0));
        _design.Background = ShowsCandidate ? clear : new SolidColorBrush(fill);
        _design.Foreground = new SolidColorBrush(ShowsCandidate ? quiet : ink);
        _candidate.Background = ShowsCandidate ? ThemeLookup.Brush(this, "AccentFillColorDefaultBrush") : clear;
        _candidate.Foreground = ThemeLookup.Brush(this, ShowsCandidate ? "TextOnAccentFillColorPrimaryBrush" : "AccentTextFillColorPrimaryBrush");
        AutomationProperties.SetItemStatus(_design, ShowsCandidate ? "" : Loc.S("Results_A11y_Selected"));
        AutomationProperties.SetItemStatus(_candidate, ShowsCandidate ? Loc.S("Results_A11y_Selected") : "");
    }
}
