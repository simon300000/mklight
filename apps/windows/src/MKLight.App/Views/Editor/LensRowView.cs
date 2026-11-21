using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;

namespace MKLight.App.Views;

/// <summary>
/// One row of the lens grid: the surface label, the type menu and the value cells, laid out in the
/// grid's shared column widths. Values are text; the one text field is the panel's editor, placed over
/// a cell's <see cref="Slots">slot</see> while it edits.
/// </summary>
sealed partial class LensRowView : Grid
{
    readonly Border _background;
    readonly Grid _grid = new();
    readonly Border _pill;
    readonly Func<string, Style> _style;
    readonly ILensCellActions _actions;
    /// <summary>The element in each grid column (measured to fit a column to its content).</summary>
    readonly List<FrameworkElement> _cells = [];

    public LensRowView(Func<string, Style> style, ILensCellActions actions)
    {
        _style = style;
        _actions = actions;
        Height = 26;
        _background = new Border { Style = style("LensRow") };
        _pill = new Border { Style = style("LensSelectionPill"), Visibility = Visibility.Collapsed };
        Children.Add(_background);
        Children.Add(_grid);
        Children.Add(_pill);
    }

    public LensRow? Row { get; private set; }

    protected override Microsoft.UI.Xaml.Automation.Peers.AutomationPeer OnCreateAutomationPeer() => new LensRowAutomationPeer(this, _actions);

    /// <summary>The element showing each value (an x section has its own), by column: hit target and editor anchor.</summary>
    public Dictionary<LensColumn, Slot> Slots { get; } = [];

    /// <summary>Where a value is: the area it spans, and its text (hidden under the editor).</summary>
    public sealed record Slot(Border Area, TextBlock Text);

    /// <summary>The value under <paramref name="p"/> (in this row's coordinates), if any.</summary>
    public LensColumn? ColumnAt(global::Windows.Foundation.Point p)
    {
        foreach (var (column, slot) in Slots)
        {
            var origin = slot.Area.TransformToVisual(this).TransformPoint(default);
            if (p.X >= origin.X && p.X < origin.X + slot.Area.ActualWidth) return column;
        }
        return null;
    }

    /// <summary>The type menu was asked for (the panel builds it).</summary>
    public event Action<LensRowView, Button>? TypeMenuRequested;

    public void SetWidths(double[] widths)
    {
        while (_grid.ColumnDefinitions.Count < widths.Length) _grid.ColumnDefinitions.Add(new ColumnDefinition());
        for (var i = 0; i < widths.Length; i++) _grid.ColumnDefinitions[i].Width = new GridLength(widths[i]);
    }

    /// <summary>The width grid column <paramref name="k"/> needs to show its content in full (padding included).</summary>
    public double ContentWidth(int k)
    {
        if (k < 0 || k >= _cells.Count) return 0;
        var cell = _cells[k];
        cell.Measure(new global::Windows.Foundation.Size(double.PositiveInfinity, Height));
        var width = cell.DesiredSize.Width;
        // The next layout pass measures it in its column again.
        _grid.InvalidateMeasure();
        return width;
    }

    public void SetSelection(bool selected, bool emphasized, bool alternate)
    {
        _background.Style = _style(selected ? (emphasized ? "LensSelectedRow" : "LensSelectedInactiveRow") : alternate ? "LensStripeRow" : "LensRow");
        _pill.Visibility = selected && emphasized ? Visibility.Visible : Visibility.Collapsed;
    }

    /// <summary>Shows <paramref name="row"/>. <paramref name="editing"/> is the column under the editor (its text is hidden).</summary>
    public void Update(LensRow row, IReadOnlyList<IssueItem> issues, LensColumn? editing)
    {
        Row = row;
        _grid.Children.Clear();
        _cells.Clear();
        Slots.Clear();
        var k = 0;
        foreach (var column in LensColumns.Visible)
        {
            FrameworkElement cell = column switch
            {
                LensColumn.Surface => SurfaceCell(row, issues),
                LensColumn.Type => TypeCell(row),
                _ => ValueCell(row, column),
            };
            cell.Margin = new Thickness(column == LensColumn.Type ? 2 : 6, 0, column == LensColumn.Type ? 2 : 6, 0);
            Grid.SetColumn(cell, k++);
            _grid.Children.Add(cell);
            _cells.Add(cell);
        }
        if (editing is { } c && Slots.TryGetValue(c, out var slot)) slot.Text.Opacity = 0;
        AutomationProperties.SetName(this, LensCellView.RowName(row));
        AutomationProperties.SetAutomationId(this, $"LensRow.{row.Index}");
    }

    /// <summary><c>OBJ</c>, <c>1</c>, <c>STO</c> with its marker, …, <c>IMA</c>; an issue badge before the label.</summary>
    FrameworkElement SurfaceCell(LensRow row, IReadOnlyList<IssueItem> issues)
    {
        var panel = new StackPanel { Orientation = Orientation.Horizontal, VerticalAlignment = VerticalAlignment.Center };
        var shown = issues.Where(i => i.Severity is "error" or "warning").ToList();
        if (shown.Count > 0)
        {
            var badge = new FontIcon { Style = _style(shown.Any(i => i.Severity == "error") ? "LensErrorGlyph" : "LensWarningGlyph") };
            var messages = string.Join("\n", shown.Select(i => i.Message));
            ToolTipService.SetToolTip(badge, messages);
            AutomationProperties.SetName(badge, messages);
            panel.Children.Add(badge);
        }
        var label = new TextBlock
        {
            Text = row.SurfaceText,
            Style = _style(row.Role switch { LensRowRole.Surface => "LensSurfaceLabel", LensRowRole.Stop => "LensStopLabel", _ => "LensRoleLabel" }),
        };
        AutomationProperties.SetAccessibilityView(label, AccessibilityView.Raw);
        panel.Children.Add(label);
        if (row.Role == LensRowRole.Stop)
        {
            var stop = new FontIcon { Style = _style("LensStopGlyph") };
            ToolTipService.SetToolTip(stop, Loc.S("Lens_Row_StopTip"));
            AutomationProperties.SetAccessibilityView(stop, AccessibilityView.Raw);
            panel.Children.Add(stop);
        }
        var cell = new LensCellView(row, LensColumn.Surface, _actions);
        cell.Children.Add(panel);
        return cell;
    }

    /// <summary>The surface type: a compact menu button (glyph, title, chevron); plain text on the object and image.</summary>
    FrameworkElement TypeCell(LensRow row)
    {
        var entry = SurfaceTypeCatalog.EntryFor(row.TypeChoice);
        var cell = new LensCellView(row, LensColumn.Type, _actions);
        if (row.Role is LensRowRole.Object or LensRowRole.Image)
        {
            var text = new TextBlock { Text = row.TypeTitle, Style = _style("LensSecondaryText"), Margin = new Thickness(6, 0, 0, 0) };
            AutomationProperties.SetAccessibilityView(text, AccessibilityView.Raw);
            cell.Children.Add(text);
            return cell;
        }
        var content = new Grid();
        content.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        content.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        content.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        content.Children.Add(new FontIcon { Glyph = entry.Glyph, Style = _style("LensTypeGlyph") });
        var title = new TextBlock { Text = row.TypeTitle, Style = _style("LensText") };
        Grid.SetColumn(title, 1);
        content.Children.Add(title);
        var chevron = new FontIcon { Style = _style("LensChevron") };
        Grid.SetColumn(chevron, 2);
        content.Children.Add(chevron);
        var button = new Button { Style = _style("LensTypeButton"), Content = content };
        ToolTipService.SetToolTip(button, entry.Title);
        // The cell is the automation element (value: the type; invoke: the menu).
        AutomationProperties.SetAccessibilityView(button, AccessibilityView.Raw);
        button.Click += (_, _) => TypeMenuRequested?.Invoke(this, button);
        cell.Children.Add(button);
        return cell;
    }

    /// <summary>The type cell's menu button, if the row has one.</summary>
    public Button? TypeButton => _cells.Count > 1 && _cells[1] is LensCellView { Children: [Button b] } ? b : null;

    /// <summary>A value cell: one value, or the y and x sections of an anamorphic surface side by side.</summary>
    FrameworkElement ValueCell(LensRow row, LensColumn column)
    {
        var shown = row.Shown(column);
        if (shown.Count == 1) return Section(row, shown[0]);
        var pair = new Grid { ColumnSpacing = 8 };
        pair.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        pair.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        for (var i = 0; i < shown.Count; i++)
        {
            var section = Section(row, shown[i]);
            Grid.SetColumn(section, i);
            pair.Children.Add(section);
        }
        return pair;
    }

    /// <summary>
    /// One value. Numbers are right-aligned with their markers and axis hint before them, so the digits
    /// line up down the column; text is left-aligned with a melt offset and markers after it. The slot
    /// (the whole cell) is what a click edits and what the editor covers.
    /// </summary>
    FrameworkElement Section(LensRow row, LensColumn c)
    {
        var numeric = c.IsNumeric();
        var editable = row.Editable.Contains(c);
        var text = new TextBlock
        {
            Text = row.Text(c),
            Style = _style(numeric
                ? (!editable ? "LensSecondaryNumber" : row.IsDimmed(c) ? "LensDimNumber" : "LensNumber")
                : (!editable ? "LensSecondaryText" : "LensText")),
        };
        // The cell's automation value is the text.
        AutomationProperties.SetAccessibilityView(text, AccessibilityView.Raw);
        var marker = row.Markers.TryGetValue(c, out var m) ? Markers(m, leading: numeric) : null;
        var cell = new LensCellView(row, c, _actions);
        var slot = new Border();
        cell.Children.Add(slot);
        Slots[c] = new Slot(slot, text);
        if (numeric)
        {
            var line = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
            if (marker is not null) line.Children.Add(marker);
            if (row.Hints.TryGetValue(c, out var hint))
            {
                // The axis sits right before its value (the cell's name carries it).
                var h = new TextBlock { Text = hint, Style = _style("LensHint") };
                AutomationProperties.SetAccessibilityView(h, AccessibilityView.Raw);
                ToolTipService.SetToolTip(h, HintHelp(row, c, hint));
                line.Children.Add(h);
            }
            line.Children.Add(text);
            cell.Children.Add(line);
            return cell;
        }
        var note = row.Notes.TryGetValue(c, out var n) ? new TextBlock { Text = n, Style = _style("LensNote") } : null;
        // Text alone trims; with a melt offset the glass name shows in full and the offset trims.
        var star = new GridLength(1, GridUnitType.Star);
        cell.ColumnDefinitions.Add(new ColumnDefinition { Width = note is null && marker is null ? star : GridLength.Auto });
        cell.ColumnDefinitions.Add(new ColumnDefinition { Width = note is null ? GridLength.Auto : star });
        cell.ColumnDefinitions.Add(new ColumnDefinition { Width = note is null && marker is not null ? star : GridLength.Auto });
        Grid.SetColumnSpan(slot, 3);
        cell.Children.Add(text);
        if (note is not null)
        {
            ToolTipService.SetToolTip(note, Loc.S("Lens_Row_MeltTip"));
            Grid.SetColumn(note, 1);
            cell.Children.Add(note);
        }
        if (marker is not null)
        {
            Grid.SetColumn(marker, note is null ? 1 : 2);
            cell.Children.Add(marker);
        }
        return cell;
    }

    /// <summary><c>V</c> (variable) and a solve's letter, before a number or after a glass.</summary>
    FrameworkElement Markers(CellMarker m, bool leading)
    {
        var panel = new StackPanel
        {
            Orientation = Orientation.Horizontal, Spacing = 3, VerticalAlignment = VerticalAlignment.Center,
            Margin = leading ? new Thickness(0, 0, 6, 0) : new Thickness(6, 0, 0, 0),
        };
        if (m.SolveLetter is { } letter)
        {
            var b = new Border { Style = _style("LensSolveBadge"), Child = new TextBlock { Text = letter, Style = _style("LensBadgeText") } };
            ToolTipService.SetToolTip(b, Loc.F("Lens_Row_SolveTip", m.SolveTitle));
            AutomationProperties.SetName(b, Loc.F("Lens_Row_SolveName", m.SolveTitle));
            panel.Children.Add(b);
        }
        if (m.IsVariable)
        {
            // A solve sets it: the optimizer leaves the variable out.
            var ignored = m.SolveLetter is not null;
            var v = new Grid();
            v.Children.Add(new TextBlock { Text = "V", Style = _style(ignored ? "LensVariableIgnoredText" : m.Bounds == VariableBounds.Pinned ? "LensBadgeText" : "LensVariableText") });
            if (!ignored && m.Bounds == VariableBounds.Bounded)
                v.Children.Add(new Border { Style = _style("LensBoundMark"), Width = 7, Height = 1.5, CornerRadius = new CornerRadius(0.75), VerticalAlignment = VerticalAlignment.Bottom, HorizontalAlignment = HorizontalAlignment.Center, Margin = new Thickness(0, 0, 0, 1.5) });
            if (!ignored && m.Bounds == VariableBounds.Pinned)
                v.Children.Add(new Border { Style = _style("LensBoundMark"), Width = 4, Height = 4, CornerRadius = new CornerRadius(2), VerticalAlignment = VerticalAlignment.Top, HorizontalAlignment = HorizontalAlignment.Right, Margin = new Thickness(0, -2, -2, 0) });
            var b = new Border { Style = _style(ignored ? "LensVariableIgnoredBadge" : "LensVariableBadge"), Child = v };
            var help = ignored ? Loc.S("Lens_Row_VariableIgnored")
                : m.Bounds == VariableBounds.Pinned ? Loc.S("Lens_Row_VariablePinned") : m.Bounds == VariableBounds.Bounded ? Loc.S("Lens_Row_VariableBounded") : Loc.S("Lens_Row_Variable");
            ToolTipService.SetToolTip(b, help);
            AutomationProperties.SetName(b, help);
            panel.Children.Add(b);
        }
        return panel;
    }

    static string HintHelp(LensRow row, LensColumn c, string hint)
    {
        var what = c.Base() == LensColumn.Radius ? Loc.S("Lens_Row_HintRadius") : Loc.S("Lens_Row_HintConic");
        return row.IsFlatSection(c) ? Loc.F("Lens_Row_HintFlat", hint, what) : Loc.F("Lens_Row_HintAnamorphic", hint, what);
    }
}
