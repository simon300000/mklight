using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Glass;

/// <summary>
/// Edits one user glass: name and catalog, a model glass or a dispersion formula, range,
/// physical data and status. The engine checks every change (after a short pause) and the
/// dialog shows the n_d and V_d it computes; Save is enabled only for a glass it accepts.
/// </summary>
public sealed class GlassEditorDialog
{
    readonly GlassEditorModel _e;
    readonly GlassBrowserModel _browser;
    readonly ContentDialog _dialog;
    readonly StackPanel _dispersionFields = new() { Spacing = 8 };
    readonly TextBlock _hint = new() { TextWrapping = TextWrapping.Wrap };
    readonly TextBlock _status = new() { TextWrapping = TextWrapping.Wrap };
    readonly FontIcon _statusIcon = new() { Glyph = "\uE7BA", FontSize = 14, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 8, 0) };
    readonly DispatcherTimer _debounce = new() { Interval = TimeSpan.FromMilliseconds(150) };
    readonly FrameworkElement _owner;

    static Style S(string key) => (Style)Application.Current.Resources[key];

    GlassEditorDialog(GlassEditorModel editor, GlassBrowserModel browser, FrameworkElement owner)
    {
        _e = editor;
        _browser = browser;
        _owner = owner;
        _hint.Style = S("CaptionText");
        _status.Style = S("SecondaryNumberText");
        _status.TextWrapping = TextWrapping.Wrap;
        _debounce.Tick += (_, _) => { _debounce.Stop(); Check(); };

        var form = new StackPanel { Spacing = 16, Width = 480, Padding = new Thickness(0, 0, 14, 0) };

        // Name and catalog.
        var names = new Grid { ColumnSpacing = 12 };
        names.ColumnDefinitions.Add(new ColumnDefinition());
        names.ColumnDefinitions.Add(new ColumnDefinition());
        var name = new TextBox { Header = "Name", Text = editor.Name };
        name.TextChanged += (_, _) => { editor.Name = name.Text; Changed(); };
        names.Children.Add(name);
        var catalog = new ComboBox { Header = "Catalog", HorizontalAlignment = HorizontalAlignment.Stretch, IsEnabled = editor.OriginalName is null };
        foreach (var c in browser.Library.UserCatalogs) catalog.Items.Add(c.Name);
        catalog.SelectedItem = browser.Library.UserCatalogNamed(editor.Catalog)?.Name;
        catalog.SelectionChanged += (_, _) => { if (catalog.SelectedItem is string s) { editor.Catalog = s; Changed(); } };
        Grid.SetColumn(catalog, 1);
        names.Children.Add(catalog);
        form.Children.Add(names);

        // Dispersion: a model glass or an AGF formula.
        var dispersion = new StackPanel { Spacing = 8 };
        dispersion.Children.Add(Heading("Dispersion"));
        var kind = new SelectorBar();
        var model = new SelectorBarItem { Text = "Model Glass" };
        var formula = new SelectorBarItem { Text = "Dispersion Formula" };
        kind.Items.Add(model);
        kind.Items.Add(formula);
        kind.SelectedItem = editor.Dispersion == GlassEditorModel.Kind.Model ? model : formula;
        kind.SelectionChanged += (_, _) =>
        {
            editor.Dispersion = kind.SelectedItem == model ? GlassEditorModel.Kind.Model : GlassEditorModel.Kind.Formula;
            BuildDispersion();
            Changed();
        };
        dispersion.Children.Add(kind);
        dispersion.Children.Add(_dispersionFields);
        dispersion.Children.Add(_hint);
        form.Children.Add(dispersion);
        BuildDispersion();

        // Range and data.
        var data = new StackPanel { Spacing = 8 };
        data.Children.Add(Heading("Range and Data"));
        data.Children.Add(Pairs(
            Optional("Shortest λ (µm)", editor.MinUm, v => editor.MinUm = v),
            Optional("Longest λ (µm)", editor.MaxUm, v => editor.MaxUm = v),
            Optional("Density (g/cm³)", editor.Density, v => editor.Density = v),
            Optional("TCE (10⁻⁶/K)", editor.Tce, v => editor.Tce = v),
            Optional("Relative cost", editor.RelativeCost, v => editor.RelativeCost = v)));
        form.Children.Add(data);

        // Availability.
        var availability = new StackPanel { Spacing = 8 };
        availability.Children.Add(Heading("Availability"));
        var row = new Grid { ColumnSpacing = 12 };
        row.ColumnDefinitions.Add(new ColumnDefinition());
        row.ColumnDefinitions.Add(new ColumnDefinition());
        var status = new ComboBox { Header = "Status", HorizontalAlignment = HorizontalAlignment.Stretch };
        foreach (var s in GlassStatus.Choices) status.Items.Add(s.Title);
        status.SelectedIndex = Math.Max(0, GlassStatus.Choices.ToList().IndexOf(editor.Status));
        status.SelectionChanged += (_, _) => { if (status.SelectedIndex >= 0) { editor.Status = GlassStatus.Choices[status.SelectedIndex]; Changed(); } };
        row.Children.Add(status);
        var exclude = new CheckBox { Content = "Exclude from glass substitution", IsChecked = editor.ExcludeSubstitution, VerticalAlignment = VerticalAlignment.Bottom };
        exclude.Click += (_, _) => { editor.ExcludeSubstitution = exclude.IsChecked == true; Changed(); };
        Grid.SetColumn(exclude, 1);
        row.Children.Add(exclude);
        availability.Children.Add(row);
        var comment = new TextBox { Header = "Comment", Text = editor.Comment, AcceptsReturn = true, TextWrapping = TextWrapping.Wrap, MaxHeight = 72 };
        comment.TextChanged += (_, _) => { editor.Comment = comment.Text; Changed(); };
        availability.Children.Add(comment);
        form.Children.Add(availability);

        // The engine's answer: n_d, V_d, or why not.
        var answer = new Grid { Padding = new Thickness(2, 0, 0, 0) };
        answer.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        answer.ColumnDefinitions.Add(new ColumnDefinition());
        answer.Children.Add(_statusIcon);
        Grid.SetColumn(_status, 1);
        answer.Children.Add(_status);

        var content = new Grid { RowSpacing = 12 };
        content.RowDefinitions.Add(new RowDefinition { Height = new GridLength(1, GridUnitType.Star) });
        content.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
        content.Children.Add(new ScrollViewer { Content = form, MaxHeight = 520, HorizontalScrollMode = ScrollMode.Disabled });
        Grid.SetRow(answer, 1);
        content.Children.Add(answer);

        _dialog = new ContentDialog
        {
            Title = editor.Title, Content = content,
            PrimaryButtonText = "Save", CloseButtonText = "Cancel", DefaultButton = ContentDialogButton.Primary,
            XamlRoot = owner.XamlRoot, RequestedTheme = owner.ActualTheme,
        };
        _dialog.Resources["ContentDialogMaxWidth"] = 560.0;
        _dialog.PrimaryButtonClick += (_, args) =>
        {
            _debounce.Stop();
            if (!_browser.SaveEditor(_e)) { args.Cancel = true; ShowAnswer(); }
        };
        Check();
    }

    /// <summary>Shows the editor; returns whether the glass was saved.</summary>
    public static async Task<bool> ShowAsync(GlassEditorModel editor, GlassBrowserModel browser, FrameworkElement owner)
    {
        var d = new GlassEditorDialog(editor, browser, owner);
        return await d._dialog.ShowAsync() == ContentDialogResult.Primary;
    }

    void BuildDispersion()
    {
        _dispersionFields.Children.Clear();
        if (_e.Dispersion == GlassEditorModel.Kind.Model)
        {
            _dispersionFields.Children.Add(Pairs(
                Required("nd", _e.Nd, v => _e.Nd = v),
                Required("Vd", _e.Vd, v => _e.Vd = v),
                Required("ΔPg,F", _e.Dpgf, v => _e.Dpgf = v)));
            _hint.Text = "A Cauchy fit through nd, the F–C dispersion and ΔPg,F from the normal line (valid 0.35–1.1 µm unless set below).";
            return;
        }
        var formula = new ComboBox { Header = "Formula", HorizontalAlignment = HorizontalAlignment.Stretch };
        foreach (var (f, n) in GlassEditorModel.FormulaNames) formula.Items.Add($"{f}: {n}");
        formula.SelectedIndex = GlassEditorModel.FormulaNames.Select(x => x.Formula).ToList().IndexOf(_e.Formula);
        formula.SelectionChanged += (_, _) =>
        {
            if (formula.SelectedIndex < 0) return;
            _e.Formula = GlassEditorModel.FormulaNames[formula.SelectedIndex].Formula;
            BuildDispersion();
            Changed();
        };
        _dispersionFields.Children.Add(formula);
        var names = GlassEditorModel.CoefficientNames(_e.Formula);
        var fields = new List<FrameworkElement>();
        for (var i = 0; i < names.Count; i++)
        {
            var k = i;
            fields.Add(Required(names[i], i < _e.Coefficients.Length ? _e.Coefficients[i] : 0, v => { if (k < _e.Coefficients.Length) _e.Coefficients[k] = v; }));
        }
        _dispersionFields.Children.Add(Pairs([.. fields]));
        _hint.Text = "AGF formula coefficients, λ in µm.";
    }

    static TextBlock Heading(string text) => new() { Text = text, Style = S("ColumnHeaderText") };

    /// <summary>Fields two (or three) to a row.</summary>
    static Grid Pairs(params FrameworkElement[] fields)
    {
        var columns = fields.Length == 3 ? 3 : 2;
        var g = new Grid { ColumnSpacing = 12, RowSpacing = 8 };
        for (var c = 0; c < columns; c++) g.ColumnDefinitions.Add(new ColumnDefinition());
        for (var i = 0; i < fields.Length; i++)
        {
            if (i % columns == 0) g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
            Grid.SetRow(fields[i], i / columns);
            Grid.SetColumn(fields[i], i % columns);
            g.Children.Add(fields[i]);
        }
        return g;
    }

    TextBox Required(string header, double value, Action<double> set) =>
        NumberField(header, value, null, v => set(v ?? double.NaN));

    TextBox Optional(string header, double? value, Action<double?> set) =>
        NumberField(header, value, "none", set);

    /// <summary>A number in the app's format (decimal point, no grouping); empty is "none" for optional values.</summary>
    TextBox NumberField(string header, double? value, string? placeholder, Action<double?> set)
    {
        var box = new TextBox { Header = header, Text = GlassFormat.Editable(value), TextAlignment = TextAlignment.Right, PlaceholderText = placeholder ?? "" };
        box.TextChanged += (_, _) =>
        {
            var text = box.Text.Trim();
            var v = text.Length == 0 ? null : NumberDisplay.Parse(text);
            box.Foreground = text.Length > 0 && v is null ? ThemeLookup.Brush(_owner, "ErrorBrush") : ThemeLookup.Brush(_owner, "TextFillColorPrimaryBrush");
            set(text.Length > 0 && v is null ? double.NaN : v);
            Changed();
        };
        return box;
    }

    void Changed()
    {
        _debounce.Stop();
        _debounce.Start();
    }

    void Check()
    {
        if (_browser.Engine is { } engine) _e.Check(engine);
        ShowAnswer();
    }

    void ShowAnswer()
    {
        if (_e.Problem is { } p)
        {
            _statusIcon.Visibility = Visibility.Visible;
            _statusIcon.Foreground = ThemeLookup.Brush(_owner, "WarningBrush");
            _status.Text = p;
            _status.FontWeight = FontWeights.Normal;
        }
        else if (_e.PreviewNumbers is { } n)
        {
            _statusIcon.Visibility = Visibility.Collapsed;
            _status.Text = n.Nd > 0 ? $"nd {GlassFormat.Fixed(n.Nd, 5)}    Vd {GlassFormat.Fixed(n.Vd, 2)}" : "No nd, Vd (the d line is outside the range)";
        }
        else
        {
            _statusIcon.Visibility = Visibility.Collapsed;
            _status.Text = "";
        }
        _dialog.IsPrimaryButtonEnabled = _e.Preview is not null;
    }
}
