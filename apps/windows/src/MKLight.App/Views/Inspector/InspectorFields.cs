using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using MKLight.App.ViewModels;
using Windows.System;

namespace MKLight.App.Views.Inspector;

/// <summary>
/// A text box that edits a number by the one text-field rule (<see cref="FieldEdit{T}"/>, docs/APP.md): the
/// formatted value at rest, every digit while focused; only edited text commits (Enter or focus loss), a value
/// equal to the current one is not committed, Esc reverts, rejected text reverts with the reason in a toast.
/// Text typed for one owner (a surface) is committed to that owner even when the field moved on to another.
/// </summary>
sealed class NumberEntry
{
    public readonly TextBox Box;
    readonly FieldEdit<double?> _edit;
    Func<string, Parsed<double>> _parse = LensValueParser.Finite;
    Action<double> _commit = _ => { };
    Action<string>? _reject;
    bool _programmatic;

    /// <summary>Set while the form rebuilds its tree: the box loses the focus but the edit goes on.</summary>
    public static bool Suspended;

    public NumberEntry(Style? style)
    {
        Box = new TextBox { Style = style, TextAlignment = TextAlignment.Right, IsSpellCheckEnabled = false };
        _edit = new FieldEdit<double?>(null, new FieldFormat<double?>(
            v => v is { } x ? InspectorFormat.Exact(x) : "",
            v => v is { } x ? InspectorFormat.Short(x) : "",
            text => _parse(text) is { Ok: true } p ? new Optional<double?>(p.Value) : Optional<double?>.None));
        Box.TextChanged += (_, _) => { if (!_programmatic) _edit.Text = Box.Text; };
        Box.GotFocus += (_, _) =>
        {
            if (Suspended) return;
            _edit.BeginEditing();
            Show();
            Box.SelectAll();
        };
        Box.LostFocus += (_, _) =>
        {
            if (Suspended) return;
            Apply(_edit.EndEditing(), _commit);
            Show();
        };
        Box.KeyDown += OnKey;
    }

    void OnKey(object sender, KeyRoutedEventArgs e)
    {
        if (e.Key == VirtualKey.Enter)
        {
            Apply(_edit.Submit(), _commit);
            Show();
            Box.SelectAll();
            e.Handled = true;
        }
        else if (e.Key == VirtualKey.Escape && _edit.IsEdited)
        {
            _edit.Revert();
            Show();
            Box.SelectAll();
            e.Handled = true;
        }
    }

    /// <summary>Types <paramref name="text"/> and presses Enter (the launch actions' scripted edits).</summary>
    public void TypeAndEnter(string text)
    {
        _edit.BeginEditing();
        _programmatic = true;
        Box.Text = text;
        _programmatic = false;
        _edit.Text = text;
        Apply(_edit.Submit(), _commit);
        Show();
    }

    /// <summary>The value (null: empty), its owner, how typed text parses, and what a commit does.</summary>
    public void Set(double? value, object? owner, Func<string, Parsed<double>>? parse, Action<double> commit, Action<string>? reject)
    {
        var earlierCommit = _commit;
        _parse = parse ?? LensValueParser.Finite;
        _reject = reject;
        var earlier = _edit.ValueChanged(value, owner, out _);
        _commit = commit;
        Apply(earlier, earlierCommit);
        Show();
    }

    void Apply(FieldCommit<double?> c, Action<double> commit)
    {
        switch (c)
        {
            case FieldCommit<double?>.Value { NewValue: { } v }:
                commit(v);
                break;
            case FieldCommit<double?>.Invalid bad:
                _reject?.Invoke(_parse(bad.Text).Error ?? LensValueParser.NotANumber(bad.Text));
                _edit.Revert();
                break;
        }
    }

    void Show()
    {
        if (Box.Text == _edit.Text) return;
        _programmatic = true;
        Box.Text = _edit.Text;
        _programmatic = false;
    }
}

/// <summary>A text box that edits a string by the one text-field rule; with suggestions (glass names) it is an <see cref="AutoSuggestBox"/>.</summary>
sealed class TextEntry
{
    public readonly FrameworkElement View;
    readonly TextBox? _box;
    readonly AutoSuggestBox? _suggest;
    readonly FieldEdit<string> _edit = new("", FieldFormat<string>.Identity);
    Action<string> _commit = _ => { };
    IReadOnlyList<string> _suggestions = [];
    bool _programmatic;

    public TextEntry(Style? style, bool suggestions, TextAlignment alignment)
    {
        if (suggestions)
        {
            _suggest = new AutoSuggestBox { MinWidth = 120, UpdateTextOnSelect = true, TextBoxStyle = InspectorStyles.Get("InspectorSuggestText") };
            _suggest.TextChanged += (s, e) =>
            {
                if (_programmatic || e.Reason != AutoSuggestionBoxTextChangeReason.UserInput) { _edit.Text = s.Text; return; }
                _edit.Text = s.Text;
                var q = s.Text.Trim().ToUpperInvariant();
                s.ItemsSource = q.Length == 0 ? null : _suggestions.Where(n => n.ToUpperInvariant().StartsWith(q, StringComparison.Ordinal) && n.ToUpperInvariant() != q).Take(12).ToList();
            };
            _suggest.QuerySubmitted += (s, e) =>
            {
                if (e.ChosenSuggestion is string chosen) { _edit.Text = chosen; }
                Apply(_edit.Submit(), _commit);
                Show();
            };
            _suggest.GotFocus += (_, _) => { if (!NumberEntry.Suspended) _edit.BeginEditing(); };
            _suggest.LostFocus += (_, _) =>
            {
                // The focus moves into the box's own text field and list too.
                if (NumberEntry.Suspended || IsWithin()) return;
                Apply(_edit.EndEditing(), _commit);
                Show();
            };
            View = _suggest;
        }
        else
        {
            _box = new TextBox { Style = style, TextAlignment = alignment, IsSpellCheckEnabled = false };
            _box.TextChanged += (_, _) => { if (!_programmatic) _edit.Text = _box.Text; };
            _box.GotFocus += (_, _) => { if (!NumberEntry.Suspended) _edit.BeginEditing(); };
            _box.LostFocus += (_, _) =>
            {
                if (NumberEntry.Suspended) return;
                Apply(_edit.EndEditing(), _commit);
                Show();
            };
            _box.KeyDown += (_, e) =>
            {
                if (e.Key == VirtualKey.Enter && !_box.AcceptsReturn) { Apply(_edit.Submit(), _commit); Show(); e.Handled = true; }
                else if (e.Key == VirtualKey.Escape && _edit.IsEdited) { _edit.Revert(); Show(); e.Handled = true; }
            };
            View = _box;
        }
    }

    bool IsWithin()
    {
        if (View.XamlRoot is null) return false;
        var focused = FocusManager.GetFocusedElement(View.XamlRoot) as DependencyObject;
        for (var d = focused; d is not null; d = Microsoft.UI.Xaml.Media.VisualTreeHelper.GetParent(d))
            if (ReferenceEquals(d, View)) return true;
        return false;
    }

    public void SetPrompt(string prompt)
    {
        if (_box is not null) _box.PlaceholderText = prompt;
        if (_suggest is not null) _suggest.PlaceholderText = prompt;
    }

    public void Set(string value, object? owner, Action<string> commit, IReadOnlyList<string>? suggestions = null)
    {
        var earlierCommit = _commit;
        _suggestions = suggestions ?? [];
        var earlier = _edit.ValueChanged(value, owner, out _);
        _commit = commit;
        Apply(earlier, earlierCommit);
        Show();
    }

    static void Apply(FieldCommit<string> c, Action<string> commit)
    {
        if (c is FieldCommit<string>.Value v) commit(v.NewValue);
    }

    void Show()
    {
        var current = _box?.Text ?? _suggest!.Text;
        if (current == _edit.Text) return;
        _programmatic = true;
        if (_box is not null) _box.Text = _edit.Text; else _suggest!.Text = _edit.Text;
        _programmatic = false;
    }
}
