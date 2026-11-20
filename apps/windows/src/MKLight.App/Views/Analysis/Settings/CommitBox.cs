using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Windows.System;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// A text field with the app's one text-field rule (docs/APP.md): compact
/// text at rest and the exact value with the focus; only edited text is
/// committed, on Enter and when the focus leaves; text that is not a value
/// reverts; Escape reverts.
/// </summary>
public sealed partial class CommitBox : TextBox
{
    readonly Func<bool, string> _text;
    readonly Func<string, Edit?> _parse;
    readonly Action<Edit> _commit;
    string _shown = "";

    /// <param name="text">The current value's text, exact (focused) or compact.</param>
    /// <param name="parse">Typed text to an edit; null when it is not a value.</param>
    /// <param name="commit">Stores the edit (an equal value is ignored by the form).</param>
    public CommitBox(Func<bool, string> text, Func<string, Edit?> parse, Action<Edit> commit, double width, bool number = true)
    {
        _text = text;
        _parse = parse;
        _commit = commit;
        Width = width;
        MinWidth = 0;
        MinHeight = 28;
        FontSize = 12;
        Padding = new Thickness(7, 4, 7, 4);
        VerticalAlignment = VerticalAlignment.Center;
        if (number)
        {
            TextAlignment = TextAlignment.Right;
            Microsoft.UI.Xaml.Documents.Typography.SetNumeralAlignment(this, FontNumeralAlignment.Tabular);
        }
        Show(false);
        GotFocus += (_, _) => Show(true);
        LostFocus += (_, _) => { Commit(); Show(false); };
        KeyDown += OnKey;
    }

    void OnKey(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter) { Commit(); Show(true); SelectAll(); e.Handled = true; }
        else if (e.Key == VirtualKey.Escape && Text != _shown) { Show(true); e.Handled = true; }
    }

    void Show(bool exact)
    {
        _shown = _text(exact);
        Text = _shown;
    }

    /// <summary>Commits edited text; the form ignores a value equal to the current one.</summary>
    public void Commit()
    {
        if (Text == _shown) return;
        var edit = _parse(Text);
        if (edit is not null) _commit(edit);
    }
}

/// <summary>Small helpers for building the form's controls.</summary>
static class FormControls
{
    public static TextBlock Label(string text, string? help = null, string style = "SecondaryText")
    {
        var t = new TextBlock { Text = text, Style = (Style)Application.Current.Resources[style], FontSize = 12 };
        if (!string.IsNullOrEmpty(help)) ToolTipService.SetToolTip(t, help);
        return t;
    }

    public static StackPanel Row(params UIElement[] children)
    {
        var p = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 5, VerticalAlignment = VerticalAlignment.Center };
        foreach (var c in children) p.Children.Add(c);
        return p;
    }

    public static DropDownButton Menu(string title, MenuFlyout flyout, string? help)
    {
        var b = new DropDownButton { Content = title, Flyout = flyout, FontSize = 12, MinHeight = 28, Padding = new Thickness(8, 3, 6, 4) };
        if (!string.IsNullOrEmpty(help)) ToolTipService.SetToolTip(b, help);
        return b;
    }

    public static MenuFlyoutItem Item(string text, Action click)
    {
        var i = new MenuFlyoutItem { Text = text };
        i.Click += (_, _) => click();
        return i;
    }

    public static JsonNode? Clone(JsonNode? n) => n?.DeepClone();
}
