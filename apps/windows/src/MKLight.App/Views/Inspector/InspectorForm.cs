using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Markup;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using Windows.UI;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// The inspector's form styles: section titles over cards of rows (labels left, values right, tabular figures),
/// on the Mica of the inspector column. Values sit in borderless fields that show their frame on hover and focus,
/// as in the Mac app's grouped forms. Loaded once from XAML so every brush is a theme resource of the window.
/// </summary>
static class InspectorStyles
{
    const string Xaml = """
        <ResourceDictionary xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
                            xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml">
            <SolidColorBrush x:Key="TextControlBackground" Color="Transparent" />
            <SolidColorBrush x:Key="TextControlBorderBrush" Color="Transparent" />
            <SolidColorBrush x:Key="ComboBoxBackground" Color="Transparent" />
            <SolidColorBrush x:Key="ComboBoxBorderBrush" Color="Transparent" />
            <Style x:Key="InspectorCard" TargetType="Border">
                <Setter Property="Background" Value="{ThemeResource CardBackgroundFillColorDefaultBrush}" />
                <Setter Property="BorderBrush" Value="{ThemeResource CardStrokeColorDefaultBrush}" />
                <Setter Property="BorderThickness" Value="1" />
                <Setter Property="CornerRadius" Value="8" />
            </Style>
            <Style x:Key="InspectorDivider" TargetType="Border">
                <Setter Property="Height" Value="1" />
                <Setter Property="Margin" Value="12,0,0,0" />
                <Setter Property="Background" Value="{ThemeResource DividerStrokeColorDefaultBrush}" />
            </Style>
            <Style x:Key="InspectorSectionTitle" TargetType="TextBlock">
                <Setter Property="FontSize" Value="12" />
                <Setter Property="FontWeight" Value="SemiBold" />
                <Setter Property="Foreground" Value="{ThemeResource TextFillColorSecondaryBrush}" />
                <Setter Property="Margin" Value="4,14,0,6" />
            </Style>
            <Style x:Key="InspectorLabel" TargetType="TextBlock">
                <Setter Property="FontSize" Value="13" />
                <Setter Property="Foreground" Value="{ThemeResource TextFillColorPrimaryBrush}" />
                <Setter Property="VerticalAlignment" Value="Center" />
                <Setter Property="TextTrimming" Value="CharacterEllipsis" />
                <Setter Property="Typography.NumeralAlignment" Value="Tabular" />
            </Style>
            <Style x:Key="InspectorSecondary" TargetType="TextBlock" BasedOn="{StaticResource InspectorLabel}">
                <Setter Property="Foreground" Value="{ThemeResource TextFillColorSecondaryBrush}" />
            </Style>
            <Style x:Key="InspectorCaption" TargetType="TextBlock">
                <Setter Property="FontSize" Value="11" />
                <Setter Property="Foreground" Value="{ThemeResource TextFillColorSecondaryBrush}" />
                <Setter Property="TextWrapping" Value="Wrap" />
                <Setter Property="Typography.NumeralAlignment" Value="Tabular" />
            </Style>
            <Style x:Key="InspectorTitle" TargetType="TextBlock">
                <Setter Property="FontSize" Value="17" />
                <Setter Property="FontWeight" Value="SemiBold" />
                <Setter Property="Foreground" Value="{ThemeResource TextFillColorPrimaryBrush}" />
                <Setter Property="VerticalAlignment" Value="Center" />
            </Style>
            <Style x:Key="InspectorField" TargetType="TextBox" BasedOn="{StaticResource DefaultTextBoxStyle}">
                <Setter Property="FontSize" Value="13" />
                <Setter Property="MinHeight" Value="28" />
                <Setter Property="MinWidth" Value="0" />
                <Setter Property="Padding" Value="6,4,3,4" />
                <Setter Property="VerticalAlignment" Value="Center" />
            </Style>
            <Style x:Key="InspectorSuggestText" TargetType="TextBox" BasedOn="{StaticResource AutoSuggestBoxTextBoxStyle}">
                <Setter Property="FontSize" Value="13" />
                <Setter Property="MinHeight" Value="28" />
                <Setter Property="Padding" Value="6,4,3,4" />
                <Setter Property="TextAlignment" Value="Right" />
            </Style>
            <Style x:Key="InspectorCombo" TargetType="ComboBox" BasedOn="{StaticResource DefaultComboBoxStyle}">
                <Setter Property="FontSize" Value="13" />
                <Setter Property="MinHeight" Value="28" />
                <Setter Property="MinWidth" Value="0" />
                <Setter Property="Padding" Value="8,3,0,4" />
                <Setter Property="HorizontalAlignment" Value="Right" />
                <Setter Property="VerticalAlignment" Value="Center" />
            </Style>
            <Style x:Key="InspectorLink" TargetType="Button" BasedOn="{StaticResource DefaultButtonStyle}">
                <Setter Property="Background" Value="Transparent" />
                <Setter Property="BorderThickness" Value="0" />
                <Setter Property="FontSize" Value="13" />
                <Setter Property="Padding" Value="6,4,6,5" />
                <Setter Property="Foreground" Value="{ThemeResource AccentTextFillColorPrimaryBrush}" />
                <Setter Property="HorizontalAlignment" Value="Left" />
            </Style>
            <Style x:Key="InspectorMenu" TargetType="DropDownButton">
                <Setter Property="Background" Value="Transparent" />
                <Setter Property="BorderThickness" Value="0" />
                <Setter Property="FontSize" Value="13" />
                <Setter Property="Padding" Value="6,4,6,5" />
                <Setter Property="Foreground" Value="{ThemeResource AccentTextFillColorPrimaryBrush}" />
            </Style>
            <Style x:Key="InspectorIconButton" TargetType="Button" BasedOn="{StaticResource DefaultButtonStyle}">
                <Setter Property="Background" Value="Transparent" />
                <Setter Property="BorderThickness" Value="0" />
                <Setter Property="Width" Value="22" />
                <Setter Property="Height" Value="22" />
                <Setter Property="Padding" Value="0" />
                <Setter Property="CornerRadius" Value="11" />
                <Setter Property="FontFamily" Value="Segoe Fluent Icons, Segoe MDL2 Assets" />
                <Setter Property="FontSize" Value="12" />
                <Setter Property="Foreground" Value="{ThemeResource TextFillColorSecondaryBrush}" />
                <Setter Property="VerticalAlignment" Value="Center" />
            </Style>
            <Style x:Key="InspectorVariable" TargetType="ToggleButton" BasedOn="{StaticResource DefaultToggleButtonStyle}">
                <Setter Property="Width" Value="16" />
                <Setter Property="Height" Value="16" />
                <Setter Property="MinWidth" Value="0" />
                <Setter Property="MinHeight" Value="0" />
                <Setter Property="Padding" Value="0" />
                <Setter Property="CornerRadius" Value="3" />
                <Setter Property="FontSize" Value="9" />
                <Setter Property="FontWeight" Value="Bold" />
                <Setter Property="Background" Value="Transparent" />
                <Setter Property="BorderThickness" Value="1" />
                <Setter Property="BorderBrush" Value="{ThemeResource ControlStrokeColorDefaultBrush}" />
                <Setter Property="Foreground" Value="{ThemeResource TextFillColorTertiaryBrush}" />
                <Setter Property="VerticalAlignment" Value="Center" />
                <Setter Property="Margin" Value="4,0,0,0" />
            </Style>
            <Style x:Key="InspectorBadge" TargetType="Border">
                <Setter Property="Width" Value="16" />
                <Setter Property="Height" Value="16" />
                <Setter Property="CornerRadius" Value="3" />
                <Setter Property="Margin" Value="4,0,0,0" />
                <Setter Property="VerticalAlignment" Value="Center" />
                <Setter Property="Background" Value="{ThemeResource ControlFillColorSecondaryBrush}" />
            </Style>
        </ResourceDictionary>
        """;

    static ResourceDictionary? _dictionary;
    public static ResourceDictionary Dictionary => _dictionary ??= NewDictionary();

    /// <summary>A dictionary of its own for a form's resources (one dictionary cannot be merged into two forms).</summary>
    public static ResourceDictionary NewDictionary() => (ResourceDictionary)XamlReader.Load(Xaml);

    public static Style Get(string key) => (Style)Dictionary[key];
}

/// <summary>What builds the form's rows: the toast for rejected text.</summary>
sealed record FormContext(Action<string> Reject);

/// <summary>
/// One row of an inspector form, described as data. A row is created once per key and then updated in place
/// (its text field keeps the focus and the user's typing across the design's changes).
/// </summary>
abstract record FormRow(string Key)
{
    internal abstract FrameworkElement Create(FormContext ctx);
    internal abstract void Update(FrameworkElement view, FormContext ctx);

    /// <summary>A label-left, content-right row of the card.</summary>
    internal static Grid LabeledGrid(out TextBlock title, out TextBlock detail, FrameworkElement content)
    {
        var g = new Grid { MinHeight = 36, Padding = new Thickness(12, 2, 8, 2), ColumnSpacing = 8 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        title = new TextBlock { Style = InspectorStyles.Get("InspectorLabel") };
        detail = new TextBlock { Style = InspectorStyles.Get("InspectorSecondary"), Margin = new Thickness(6, 0, 0, 0) };
        var label = new StackPanel { Orientation = Orientation.Horizontal, VerticalAlignment = VerticalAlignment.Center };
        label.Children.Add(title);
        label.Children.Add(detail);
        g.Children.Add(label);
        Grid.SetColumn(content, 1);
        g.Children.Add(content);
        return g;
    }

    internal static void SetText(TextBlock t, string? text)
    {
        text ??= "";
        if (t.Text != text) t.Text = text;
        t.Visibility = text.Length == 0 ? Visibility.Collapsed : Visibility.Visible;
    }

    internal static void SetHelp(FrameworkElement e, string? help) => ToolTipService.SetToolTip(e, string.IsNullOrEmpty(help) ? null : help);

    /// <summary>
    /// The row's stable automation id, <c>Inspector.Row.{key}</c>, on its value control (the element
    /// assistive technology and UI tests reach; its name is the row's label).
    /// </summary>
    internal void SetRowId(DependencyObject valueControl) =>
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(valueControl, "Inspector.Row." + Key);
}

/// <summary>Whether a value is a variable (the <c>V</c> toggle), and the solve that sets it.</summary>
sealed record VariableSpec(bool IsOn, string? Solve, Action<bool> Set);

/// <summary>A trailing small icon button (remove a term, a field, a solve).</summary>
sealed record TrailingButton(string Glyph, string Help, Action Invoke, bool Visible = true);

/// <summary>A number with its title, unit, variable toggle and solve badge.</summary>
sealed record NumberRow(string Key, string Title, double? Value, Action<double> Commit) : FormRow(Key)
{
    public string Detail { get; init; } = "";
    public string Unit { get; init; } = "";
    public string Help { get; init; } = "";
    public Func<string, Parsed<double>>? Parse { get; init; }
    public object? Owner { get; init; }
    public VariableSpec? Variable { get; init; }
    public string? Solve { get; init; }
    public TrailingButton? Trailing { get; init; }

    internal abstract class Handle(NumberEntry entry)
    {
        public readonly NumberEntry Entry = entry;
        public abstract void ToggleVariable();
    }

    sealed class View(Grid root, TextBlock title, TextBlock detail, NumberEntry box, TextBlock unit, Border solve, ToggleButton variable, Button trailing) : Handle(box)
    {
        public readonly Grid Root = root;
        public readonly TextBlock Title = title, Detail = detail, Unit = unit;
        public readonly NumberEntry Box = box;
        public readonly Border Solve = solve;
        public readonly ToggleButton Variable = variable;
        public readonly Button Trailing = trailing;
        public VariableSpec? VariableSpec;
        public TrailingButton? TrailingSpec;
        public bool Updating;

        public override void ToggleVariable()
        {
            if (VariableSpec is { } v && !(v.Solve is not null && !v.IsOn)) v.Set(!v.IsOn);
        }
    }

    internal override FrameworkElement Create(FormContext ctx)
    {
        var box = new NumberEntry(InspectorStyles.Get("InspectorField")) { };
        box.Box.Width = 104;
        var unit = new TextBlock { Style = InspectorStyles.Get("InspectorSecondary"), MinWidth = 24, Margin = new Thickness(3, 0, 0, 0) };
        var solve = new Border
        {
            Style = InspectorStyles.Get("InspectorBadge"),
            Child = new TextBlock { Text = "S", FontSize = 9, FontWeight = Microsoft.UI.Text.FontWeights.Bold, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center, Style = InspectorStyles.Get("InspectorSecondary") },
        };
        var variable = new ToggleButton { Style = InspectorStyles.Get("InspectorVariable"), Content = "V" };
        var trailing = new Button { Style = InspectorStyles.Get("InspectorIconButton"), Margin = new Thickness(2, 0, 0, 0) };
        var values = new StackPanel { Orientation = Orientation.Horizontal, HorizontalAlignment = HorizontalAlignment.Right };
        values.Children.Add(box.Box);
        values.Children.Add(unit);
        values.Children.Add(solve);
        values.Children.Add(variable);
        values.Children.Add(trailing);
        var root = LabeledGrid(out var t, out var d, values);
        SetRowId(box.Box);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetAutomationId(variable, "Inspector.Row." + Key + ".Variable");
        var view = new View(root, t, d, box, unit, solve, variable, trailing);
        variable.Click += (_, _) =>
        {
            if (view.Updating || view.VariableSpec is not { } v) return;
            var refused = v.Solve is not null && !v.IsOn;
            variable.IsChecked = v.IsOn;
            if (!refused) v.Set(!v.IsOn);
        };
        trailing.Click += (_, _) => view.TrailingSpec?.Invoke();
        root.Tag = view;
        Update(root, ctx);
        return root;
    }

    internal override void Update(FrameworkElement element, FormContext ctx)
    {
        var view = (View)element.Tag;
        view.Updating = true;
        SetText(view.Title, Title);
        SetText(view.Detail, Detail);
        SetHelp(view.Title, Help);
        view.Unit.Text = Unit;
        view.Unit.Visibility = Unit.Length == 0 && Variable is null && Solve is null && Trailing is null ? Visibility.Collapsed : Visibility.Visible;
        view.Box.Set(Value, Owner, Parse, Commit, ctx.Reject);
        AutomationName(view.Box.Box, Title);
        view.Solve.Visibility = Solve is null ? Visibility.Collapsed : Visibility.Visible;
        SetHelp(view.Solve, Solve is null ? null : Loc.F("Lens_Form_SolveBadge", Solve));
        view.VariableSpec = Variable;
        view.Variable.Visibility = Variable is null ? Visibility.Collapsed : Visibility.Visible;
        if (Variable is { } v)
        {
            view.Variable.IsChecked = v.IsOn;
            var refused = v.Solve is not null && !v.IsOn;
            view.Variable.Opacity = refused ? 0.45 : 1;
            SetHelp(view.Variable, v.Solve is { } s
                ? v.IsOn ? Loc.F("Lens_Form_VariableSolvedOn", s) : Loc.F("Lens_Form_VariableSolvedOff", s)
                : v.IsOn ? Loc.S("Lens_Form_VariableOn") : Loc.S("Lens_Form_VariableOff"));
        }
        view.TrailingSpec = Trailing;
        view.Trailing.Visibility = Trailing is null ? Visibility.Collapsed : Visibility.Visible;
        if (Trailing is { } tr)
        {
            view.Trailing.Content = tr.Glyph;
            view.Trailing.Opacity = tr.Visible ? 1 : 0;
            view.Trailing.IsHitTestVisible = tr.Visible;
            view.Trailing.IsTabStop = tr.Visible;
            SetHelp(view.Trailing, tr.Help);
            AutomationName(view.Trailing, tr.Help);
        }
        view.Updating = false;
    }

    internal static void AutomationName(DependencyObject d, string name) =>
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetName(d, name);
}

/// <summary>A text value (a comment, a glass, a name).</summary>
sealed record TextRow(string Key, string Title, string Value, Action<string> Commit) : FormRow(Key)
{
    public string Prompt { get; init; } = "";
    public object? Owner { get; init; }
    public IReadOnlyList<string>? Suggestions { get; init; }
    public bool Multiline { get; init; }

    internal override FrameworkElement Create(FormContext ctx)
    {
        var field = new TextEntry(InspectorStyles.Get("InspectorField"), Suggestions is not null, TextAlignment.Right);
        field.View.HorizontalAlignment = HorizontalAlignment.Stretch;
        field.View.VerticalAlignment = VerticalAlignment.Center;
        if (field.View is TextBox tb && Multiline) { tb.AcceptsReturn = true; tb.TextWrapping = TextWrapping.Wrap; tb.TextAlignment = TextAlignment.Left; tb.MaxHeight = 120; }
        var root = LabeledGrid(out var title, out _, field.View);
        SetRowId(field.View);
        root.Tag = (field, title);
        Update(root, ctx);
        return root;
    }

    internal override void Update(FrameworkElement view, FormContext ctx)
    {
        var (field, title) = ((TextEntry, TextBlock))view.Tag;
        SetText(title, Title);
        field.SetPrompt(Prompt);
        field.Set(Value, Owner, Commit, Suggestions);
        NumberRow.AutomationName(field.View, Title);
    }
}

/// <summary>A choice from a few options (a picker).</summary>
sealed record ChoiceRow(string Key, string Title, IReadOnlyList<(string Id, string Title)> Options, string Selected, Action<string> Commit) : FormRow(Key)
{
    public string Help { get; init; } = "";

    internal override FrameworkElement Create(FormContext ctx)
    {
        var combo = new ComboBox { Style = InspectorStyles.Get("InspectorCombo") };
        var root = LabeledGrid(out var title, out _, combo);
        SetRowId(combo);
        var state = new ChoiceState(combo, title);
        combo.SelectionChanged += (_, _) =>
        {
            if (state.Updating || combo.SelectedIndex < 0 || state.Spec is not { } spec || combo.SelectedIndex >= spec.Options.Count) return;
            var id = spec.Options[combo.SelectedIndex].Id;
            if (id != spec.Selected) spec.Commit(id);
        };
        root.Tag = state;
        Update(root, ctx);
        return root;
    }

    internal abstract class Handle
    {
        public abstract void Choose(string id);
    }

    sealed class ChoiceState(ComboBox combo, TextBlock title) : Handle
    {
        public override void Choose(string id)
        {
            if (Spec?.Options.Select(o => o.Id).ToList().IndexOf(id) is int i and >= 0) Combo.SelectedIndex = i;
        }

        public readonly ComboBox Combo = combo;
        public readonly TextBlock Title = title;
        public ChoiceRow? Spec;
        public bool Updating;
    }

    internal override void Update(FrameworkElement view, FormContext ctx)
    {
        var s = (ChoiceState)view.Tag;
        s.Updating = true;
        s.Spec = this;
        SetText(s.Title, Title);
        SetHelp(s.Title, Help);
        var titles = Options.Select(o => o.Title).ToList();
        if (!s.Combo.Items.Cast<object>().Select(o => o as string).SequenceEqual(titles))
        {
            s.Combo.Items.Clear();
            foreach (var t in titles) s.Combo.Items.Add(t);
        }
        var index = Options.Select(o => o.Id).ToList().IndexOf(Selected);
        if (s.Combo.SelectedIndex != index) s.Combo.SelectedIndex = index;
        NumberRow.AutomationName(s.Combo, Title);
        s.Updating = false;
    }
}

/// <summary>An on/off setting.</summary>
sealed record ToggleRow(string Key, string Title, bool IsOn, Action<bool> Commit) : FormRow(Key)
{
    sealed class State(ToggleSwitch toggle, TextBlock title)
    {
        public readonly ToggleSwitch Toggle = toggle;
        public readonly TextBlock Title = title;
        public ToggleRow? Spec;
        public bool Updating;
    }

    internal override FrameworkElement Create(FormContext ctx)
    {
        var toggle = new ToggleSwitch { OnContent = null, OffContent = null, MinWidth = 0, HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Center };
        var root = LabeledGrid(out var title, out _, toggle);
        SetRowId(toggle);
        var state = new State(toggle, title);
        toggle.Toggled += (_, _) =>
        {
            if (!state.Updating && state.Spec is { } spec && toggle.IsOn != spec.IsOn) spec.Commit(toggle.IsOn);
        };
        root.Tag = state;
        Update(root, ctx);
        return root;
    }

    internal override void Update(FrameworkElement view, FormContext ctx)
    {
        var state = (State)view.Tag;
        state.Updating = true;
        state.Spec = this;
        SetText(state.Title, Title);
        if (state.Toggle.IsOn != IsOn) state.Toggle.IsOn = IsOn;
        NumberRow.AutomationName(state.Toggle, Title);
        state.Updating = false;
    }
}

/// <summary>A read-only value (a computed semi-diameter, a catalogue's glass count), with optional captions.</summary>
sealed record InfoRow(string Key, string Title, string Value) : FormRow(Key)
{
    /// <summary>A caption under the title.</summary>
    public string Caption { get; init; } = "";
    /// <summary>A caption under the value.</summary>
    public string ValueCaption { get; init; } = "";
    public bool Secondary { get; init; } = true;
    public bool Emphasis { get; init; }
    public string Help { get; init; } = "";
    public TrailingButton? Trailing { get; init; }

    internal override FrameworkElement Create(FormContext ctx)
    {
        var g = new Grid { MinHeight = 36, Padding = new Thickness(12, 5, 8, 5), ColumnSpacing = 8 };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var trailing = new Button { Style = InspectorStyles.Get("InspectorIconButton") };
        trailing.Click += (_, _) => ((InfoRow)trailing.Tag).Trailing?.Invoke();
        Grid.SetColumn(trailing, 2);
        g.Children.Add(trailing);
        var left = new StackPanel { VerticalAlignment = VerticalAlignment.Center };
        var right = new StackPanel { VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Right };
        var title = new TextBlock { Style = InspectorStyles.Get("InspectorLabel") };
        var caption = new TextBlock { Style = InspectorStyles.Get("InspectorCaption") };
        var value = new TextBlock { HorizontalAlignment = HorizontalAlignment.Right };
        var valueCaption = new TextBlock { Style = InspectorStyles.Get("InspectorCaption"), HorizontalAlignment = HorizontalAlignment.Right };
        left.Children.Add(title);
        left.Children.Add(caption);
        right.Children.Add(value);
        right.Children.Add(valueCaption);
        SetRowId(value);
        g.Children.Add(left);
        Grid.SetColumn(right, 1);
        g.Children.Add(right);
        g.Tag = (title, caption, value, valueCaption, trailing);
        Update(g, ctx);
        return g;
    }

    internal override void Update(FrameworkElement view, FormContext ctx)
    {
        var (title, caption, value, valueCaption, trailing) = ((TextBlock, TextBlock, TextBlock, TextBlock, Button))view.Tag;
        trailing.Tag = this;
        trailing.Visibility = Trailing is null ? Visibility.Collapsed : Visibility.Visible;
        if (Trailing is { } tr)
        {
            trailing.Content = tr.Glyph;
            SetHelp(trailing, tr.Help);
            NumberRow.AutomationName(trailing, tr.Help);
        }
        SetText(title, Title);
        SetText(caption, Caption);
        value.Style = InspectorStyles.Get(Secondary && !Emphasis ? "InspectorSecondary" : "InspectorLabel");
        value.FontWeight = Emphasis ? Microsoft.UI.Text.FontWeights.SemiBold : Microsoft.UI.Text.FontWeights.Normal;
        SetText(value, Value);
        SetText(valueCaption, ValueCaption);
        SetHelp(view, Help);
        // Read as "label, value": the label names the value (its text pattern and help text carry the value).
        NumberRow.AutomationName(value, Title);
        Microsoft.UI.Xaml.Automation.AutomationProperties.SetHelpText(value, ValueCaption.Length == 0 ? Value : $"{Value}, {ValueCaption}");
    }
}

/// <summary>A borderless action ("Add Decenter or Tilt…", "Add Field").</summary>
sealed record ButtonRow(string Key, string Text, Action Invoke) : FormRow(Key)
{
    public string Glyph { get; init; } = "";
    public string Help { get; init; } = "";
    /// <summary>A second action on the same row, right-aligned (the wavelength presets).</summary>
    public FormRow? Right { get; init; }

    sealed class State(Button button, ContentControl holder)
    {
        public readonly Button Button = button;
        public readonly ContentControl Holder = holder;
        public ButtonRow? Spec;
        public (string, string)? Label;
        public FrameworkElement? RightView;
        public Type? RightKind;
    }

    internal override FrameworkElement Create(FormContext ctx)
    {
        var g = new Grid { MinHeight = 36, Padding = new Thickness(6, 2, 6, 2) };
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
        g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        var b = new Button { Style = InspectorStyles.Get("InspectorLink"), VerticalAlignment = VerticalAlignment.Center };
        SetRowId(b);
        var holder = new ContentControl { HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Center, IsTabStop = false };
        Grid.SetColumn(holder, 1);
        g.Children.Add(b);
        g.Children.Add(holder);
        var state = new State(b, holder);
        b.Click += (_, _) => state.Spec?.Invoke();
        g.Tag = state;
        Update(g, ctx);
        return g;
    }

    internal override void Update(FrameworkElement view, FormContext ctx)
    {
        var state = (State)view.Tag;
        state.Spec = this;
        if (state.Label != (Glyph, Text))
        {
            state.Label = (Glyph, Text);
            state.Button.Content = Label(Glyph, Text);
        }
        state.Button.Visibility = Text.Length == 0 ? Visibility.Collapsed : Visibility.Visible;
        SetHelp(state.Button, Help);
        if (Right is null)
        {
            state.Holder.Content = state.RightView = null;
            state.RightKind = null;
        }
        else if (state.RightView is { } existing && state.RightKind == Right.GetType())
        {
            Right.Update(existing, ctx);
        }
        else
        {
            state.Holder.Content = state.RightView = Right.Create(ctx);
            state.RightKind = Right.GetType();
        }
    }

    internal static object Label(string glyph, string text)
    {
        if (glyph.Length == 0) return text;
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6 };
        p.Children.Add(new FontIcon { Glyph = glyph, FontSize = 12 });
        p.Children.Add(new TextBlock { Text = text });
        return p;
    }
}

/// <summary>A menu of actions (Add Term, Add Solve, Make Cylinder, wavelength presets).</summary>
sealed record MenuRow(string Key, string Text, IReadOnlyList<(string Title, Action Invoke)> Items) : FormRow(Key)
{
    public string Glyph { get; init; } = "";
    public string Help { get; init; } = "";
    /// <summary>Whether the menu stands alone on its row (else it is the right part of a <see cref="ButtonRow"/>).</summary>
    public bool Row { get; init; } = true;

    sealed class State(DropDownButton button)
    {
        public readonly DropDownButton Button = button;
        public MenuRow? Spec;
        public (string, string)? Label;
        public List<string>? Titles;
    }

    internal override FrameworkElement Create(FormContext ctx)
    {
        var d = new DropDownButton { Style = InspectorStyles.Get("InspectorMenu"), VerticalAlignment = VerticalAlignment.Center };
        SetRowId(d);
        FrameworkElement root = d;
        if (Row)
        {
            var g = new Grid { MinHeight = 36, Padding = new Thickness(6, 2, 6, 2) };
            g.Children.Add(d);
            root = g;
        }
        root.Tag = new State(d);
        Update(root, ctx);
        return root;
    }

    internal override void Update(FrameworkElement view, FormContext ctx)
    {
        var state = (State)view.Tag;
        state.Spec = this;
        var d = state.Button;
        if (state.Label != (Glyph, Text))
        {
            state.Label = (Glyph, Text);
            d.Content = ButtonRow.Label(Glyph, Text);
        }
        d.HorizontalAlignment = Row ? HorizontalAlignment.Left : HorizontalAlignment.Right;
        SetHelp(d, Help);
        // The menu is rebuilt only when its entries change; a click runs the current spec's action.
        var titles = Items.Select(i => i.Title).ToList();
        if (state.Titles is { } old && old.SequenceEqual(titles)) return;
        state.Titles = titles;
        var menu = new MenuFlyout { Placement = FlyoutPlacementMode.BottomEdgeAlignedLeft };
        for (var k = 0; k < titles.Count; k++)
        {
            if (titles[k] == "-") { menu.Items.Add(new MenuFlyoutSeparator()); continue; }
            var item = new MenuFlyoutItem { Text = titles[k] };
            var index = k;
            item.Click += (_, _) => { if (state.Spec is { } spec && index < spec.Items.Count) spec.Items[index].Invoke(); };
            menu.Items.Add(item);
        }
        d.Flyout = menu;
    }
}

/// <summary>A caption inside a card (a limit reached, a hint).</summary>
sealed record NoteRow(string Key, string Text) : FormRow(Key)
{
    public bool Warning { get; init; }

    internal override FrameworkElement Create(FormContext ctx)
    {
        var t = new TextBlock { Style = InspectorStyles.Get("InspectorCaption"), Margin = new Thickness(12, 8, 12, 8) };
        t.Tag = t;
        Update(t, ctx);
        if (Warning) Support.ThemeLookup.Follow(t, () => t.Foreground = Support.ThemeLookup.Brush(t, "WarningBrush"));
        return t;
    }

    internal override void Update(FrameworkElement view, FormContext ctx) => ((TextBlock)view).Text = Text;
}

/// <summary>A row the page draws itself (the title, a shape preview, the field and wavelength grids).</summary>
sealed record CustomRow(string Key, Func<FrameworkElement> Make, Action<FrameworkElement> Refresh) : FormRow(Key)
{
    internal override FrameworkElement Create(FormContext ctx)
    {
        var e = Make();
        e.Tag ??= this;
        Refresh(e);
        return e;
    }

    internal override void Update(FrameworkElement view, FormContext ctx) => Refresh(view);
}

/// <summary>A titled card of rows, with an optional footer.</summary>
sealed record FormSection(string Key, string? Title, IReadOnlyList<FormRow> Rows)
{
    public string? Footer { get; init; }
}

/// <summary>
/// A scrolling form of sections. <see cref="Show"/> takes the form as data: rows are kept by key and updated
/// in place; the tree is rebuilt only when the rows' keys change (another surface type), and then the focused
/// field keeps the focus and its typing.
/// </summary>
sealed class FormHost : UserControl
{
    readonly ScrollViewer _scroll;
    readonly StackPanel _body = new() { Padding = new Thickness(12, 0, 12, 20) };
    readonly Dictionary<string, FrameworkElement> _rows = [];
    readonly Dictionary<string, (TextBlock Title, TextBlock Footer)> _headers = [];
    string? _signature;
    readonly FormContext _ctx;

    public FormHost(FormContext ctx)
    {
        _ctx = ctx;
        Resources.MergedDictionaries.Add(InspectorStyles.NewDictionary());
        _scroll = new ScrollViewer { Content = _body, HorizontalScrollBarVisibility = ScrollBarVisibility.Disabled, VerticalScrollBarVisibility = ScrollBarVisibility.Auto };
        Content = _scroll;
        IsTabStop = false;
    }

    public ScrollViewer Scroller => _scroll;

    /// <summary>
    /// A scripted edit of the row with <paramref name="key"/> (launch actions): a number row gets the text typed
    /// and Enter (or <c>V</c>: its variable toggle is clicked), a choice row the option with that id. False when
    /// there is no such row.
    /// </summary>
    public bool Script(string key, string value)
    {
        if (!_rows.TryGetValue(key, out var view)) return false;
        switch (view.Tag)
        {
            case NumberRow.Handle h when value == "V":
                h.ToggleVariable();
                return true;
            case NumberRow.Handle h:
                h.Entry.TypeAndEnter(value);
                return true;
            case ChoiceRow.Handle c:
                c.Choose(value);
                return true;
            default:
                return false;
        }
    }

    /// <summary>A view built from scratch (a placeholder state): replaces the form.</summary>
    public void ShowEmpty(FrameworkElement view)
    {
        _signature = null;
        foreach (var old in _stacks) old.Children.Clear();
        _stacks.Clear();
        _rows.Clear();
        _headers.Clear();
        _body.Children.Clear();
        _body.Children.Add(view);
    }

    public void Show(IReadOnlyList<FormSection> sections)
    {
        var signature = string.Join("|", sections.Select(s =>
            s.Key + (s.Title is null ? "" : "T") + (s.Footer is null ? "" : "F") + "[" + string.Join(",", s.Rows.Select(r => r.Key + ":" + r.GetType().Name)) + "]"));
        if (signature == _signature)
        {
            foreach (var s in sections)
            {
                var (t, f) = _headers[s.Key];
                if (s.Title is not null) t.Text = s.Title;
                if (s.Footer is not null) f.Text = s.Footer;
                foreach (var r in s.Rows) r.Update(_rows[r.Key], _ctx);
            }
            return;
        }

        var focusedKey = FocusedRow();
        NumberEntry.Suspended = true;
        try
        {
            var keep = new HashSet<string>();
            _body.Children.Clear();
            // Free the kept rows from the cards they were in.
            foreach (var old in _stacks) old.Children.Clear();
            _stacks.Clear();
            foreach (var s in sections)
            {
                if (!_headers.TryGetValue(s.Key, out var h))
                {
                    h = (new TextBlock { Style = InspectorStyles.Get("InspectorSectionTitle") },
                         new TextBlock { Style = InspectorStyles.Get("InspectorCaption"), Margin = new Thickness(4, 6, 4, 0) });
                    _headers[s.Key] = h;
                }
                if (s.Title is not null) { h.Title.Text = s.Title; _body.Children.Add(h.Title); }
                var stack = new StackPanel();
                _stacks.Add(stack);
                var first = true;
                foreach (var r in s.Rows)
                {
                    keep.Add(r.Key);
                    if (_rows.TryGetValue(r.Key, out var view) && Same(view, r))
                    {
                        r.Update(view, _ctx);
                    }
                    else
                    {
                        view = r.Create(_ctx);
                        _rows[r.Key] = view;
                        _kinds[r.Key] = r.GetType();
                    }
                    if (!first) stack.Children.Add(new Border { Style = InspectorStyles.Get("InspectorDivider") });
                    stack.Children.Add(view);
                    first = false;
                }
                if (s.Rows.Count > 0)
                {
                    var card = new Border { Style = InspectorStyles.Get("InspectorCard"), Child = stack };
                    if (s.Title is null && _body.Children.Count == 0) card.Margin = new Thickness(0, 8, 0, 0);
                    else if (s.Title is null) card.Margin = new Thickness(0, 12, 0, 0);
                    _body.Children.Add(card);
                }
                if (s.Footer is not null) { h.Footer.Text = s.Footer; _body.Children.Add(h.Footer); }
            }
            foreach (var key in _rows.Keys.Where(k => !keep.Contains(k)).ToList()) { _rows.Remove(key); _kinds.Remove(key); }
            foreach (var key in _headers.Keys.Where(k => !sections.Any(s => s.Key == k)).ToList()) _headers.Remove(key);
            _signature = signature;
        }
        finally
        {
            NumberEntry.Suspended = false;
        }
        if (focusedKey is not null && _rows.TryGetValue(focusedKey, out var again) && FirstFocusable(again) is Control c)
            c.Focus(FocusState.Programmatic);
    }

    readonly Dictionary<string, Type> _kinds = [];
    readonly List<StackPanel> _stacks = [];
    bool Same(FrameworkElement view, FormRow r) => _kinds.TryGetValue(r.Key, out var t) && t == r.GetType();

    string? FocusedRow()
    {
        if (XamlRoot is null) return null;
        for (var d = FocusManager.GetFocusedElement(XamlRoot) as DependencyObject; d is not null; d = VisualTreeHelper.GetParent(d))
            foreach (var (key, view) in _rows)
                if (ReferenceEquals(view, d)) return key;
        return null;
    }

    static Control? FirstFocusable(DependencyObject d)
    {
        if (d is TextBox or AutoSuggestBox) return (Control)d;
        for (var i = 0; i < VisualTreeHelper.GetChildrenCount(d); i++)
            if (FirstFocusable(VisualTreeHelper.GetChild(d, i)) is { } c) return c;
        return null;
    }
}

/// <summary>Colour helpers for views built in code.</summary>
static class InspectorColors
{
    public static Color Accent(FrameworkElement e) =>
        (Support.ThemeLookup.Resource(e.ActualTheme, "AccentFillColorDefaultBrush") as SolidColorBrush)?.Color ?? Color.FromArgb(255, 0, 103, 192);
}
