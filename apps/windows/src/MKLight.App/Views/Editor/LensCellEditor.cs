using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Controls.Primitives;
using Microsoft.UI.Xaml.Input;
using MKLight.App.ViewModels;
using MKLight.Model;
using Windows.System;

namespace MKLight.App.Views;

/// <summary>A cell of the lens table: the surface (by identity, so typed text goes to it wherever it moved) and the value.</summary>
readonly record struct LensCell(SurfaceID Row, LensColumn Column);

enum LensMove { Up, Down, Left, Right }

/// <summary>
/// The lens table's one text field, placed over the cell it edits (the Mac app's <c>LensCell</c> field).
/// It follows the one text-field rule (<see cref="FieldEdit{T}"/>): the exact value while editing, only
/// edited text commits (Enter, Tab, ↑/↓ and focus loss), a value equal to the current one is not, and
/// Esc reverts. The material column suggests glass names as you type.
/// </summary>
sealed class LensCellEditor
{
    readonly TextBox _box;
    readonly Popup _popup;
    readonly ListView _suggestions;
    FieldEdit<string>? _edit;
    string _exact = "";
    IReadOnlyList<string> _completions = [];

    public LensCellEditor(Canvas overlay, Func<string, Style> style)
    {
        _box = new TextBox { Style = style("LensEditor"), Visibility = Visibility.Collapsed };
        _box.TextChanged += (_, _) => OnTextChanged();
        _box.PreviewKeyDown += OnKeyDown;
        _box.LostFocus += (_, _) => OnLostFocus();
        // A table cell has no clear button: Esc reverts, and the width is the value's.
        _box.GotFocus += (_, _) =>
        {
            // (Its visibility belongs to the template's visual states; a zero-width, transparent button stays out of the way.)
            if (Descendant(_box, "DeleteButton") is FrameworkElement { Width: not 0 } clear)
            {
                clear.Width = 0;
                clear.Opacity = 0;
                clear.IsHitTestVisible = false;
            }
        };
        overlay.Children.Add(_box);

        _suggestions = new ListView
        {
            SelectionMode = ListViewSelectionMode.Single,
            IsItemClickEnabled = true,
            // A click on a suggestion leaves the focus in the field.
            AllowFocusOnInteraction = false,
            MaxHeight = 260,
            Padding = new Thickness(0, 4, 0, 4),
            ItemContainerStyle = style("LensSuggestionItem"),
        };
        _suggestions.ItemClick += (_, e) =>
        {
            if (e.ClickedItem is string name) Accept(name, finish: true);
        };
        _popup = new Popup
        {
            Child = new Border { Style = style("LensSuggestions"), Child = _suggestions },
        };
        overlay.Children.Add(_popup);
    }

    static DependencyObject? Descendant(DependencyObject d, string name)
    {
        for (var i = 0; i < Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChildrenCount(d); i++)
        {
            var c = Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChild(d, i);
            if (c is FrameworkElement { Name: var n } && n == name) return c;
            if (Descendant(c, name) is { } found) return found;
        }
        return null;
    }

    /// <summary>The cell being edited, or null.</summary>
    public LensCell? Active { get; private set; }

    /// <summary>Commits typed text for a cell; false when the text was rejected (the editor then shows the value again).</summary>
    public required Func<LensCell, string, bool> Commit { get; init; }
    /// <summary>Moves the editor to the next cell (after committing).</summary>
    public required Action<LensCell, LensMove> Move { get; init; }
    /// <summary>Esc: editing ends, the focus goes back to the table.</summary>
    public required Action Cancelled { get; init; }
    /// <summary>The editor closed (the focus left, or <see cref="Close"/>).</summary>
    public required Action Closed { get; init; }

    /// <summary>Edits <paramref name="cell"/>, starting from its exact text; completions for the material column.</summary>
    public void Begin(LensCell cell, string exact, bool numeric, IReadOnlyList<string> completions)
    {
        Active = cell;
        _exact = exact;
        _completions = completions;
        _edit = new FieldEdit<string>(exact, FieldFormat<string>.Identity, cell);
        _edit.BeginEditing();
        _box.TextAlignment = numeric ? TextAlignment.Right : TextAlignment.Left;
        SetText(_edit.Text);
        _box.Visibility = Visibility.Visible;
        _box.Focus(FocusState.Programmatic);
        _box.SelectAll();
        ClosePopup();
    }

    /// <summary>Places the field over the value's area (in the overlay's coordinates).</summary>
    public void Place(global::Windows.Foundation.Rect r)
    {
        Canvas.SetLeft(_box, r.X);
        Canvas.SetTop(_box, r.Y + (r.Height - _box.Height) / 2);
        _box.Width = Math.Max(r.Width, 40);
        Canvas.SetLeft(_popup, r.X);
        Canvas.SetTop(_popup, r.Y + r.Height);
    }

    /// <summary>The cell's value changed (an engine refresh, undo): text that was not edited follows it.</summary>
    public void ValueChanged(string exact)
    {
        if (_edit is null) return;
        _exact = exact;
        _edit.ValueChanged(exact);
        if (_box.Text != _edit.Text) SetText(_edit.Text);
    }

    /// <summary>Commits edited text; false when it was rejected (the field then shows the value again).</summary>
    public bool Finish()
    {
        if (_edit is null || Active is not { } cell) return true;
        if (_edit.Submit() is not FieldCommit<string>.Value v || Commit(cell, v.NewValue)) return true;
        _edit.ValueChanged(_exact);
        SetText(_edit.Text);
        _box.SelectAll();
        return false;
    }

    /// <summary>Ends editing without committing (the caller committed or reverted first).</summary>
    public void Close()
    {
        if (Active is null) return;
        Active = null;
        _edit = null;
        ClosePopup();
        _box.Visibility = Visibility.Collapsed;
        Closed();
    }

    void SetText(string text)
    {
        _box.Text = text;
        _box.SelectionStart = text.Length;
    }

    void OnTextChanged()
    {
        if (_edit is null) return;
        _edit.Text = _box.Text;
        if (Active?.Column == LensColumn.Material && _edit.IsEdited) ShowSuggestions();
    }

    void OnLostFocus()
    {
        // The focus left the table's field (a click elsewhere): commit, then close.
        if (Active is null) return;
        Finish();
        Close();
    }

    void OnKeyDown(object sender, KeyRoutedEventArgs e)
    {
        var shift = Microsoft.UI.Input.InputKeyboardSource.GetKeyStateForCurrentThread(VirtualKey.Shift)
            .HasFlag(global::Windows.UI.Core.CoreVirtualKeyStates.Down);
        e.Handled = HandleKey(e.Key, shift);
    }

    /// <summary>The field's keys: Enter, Tab / Shift+Tab, ↑/↓ (or the suggestions), Esc. Returns whether the key was used.</summary>
    internal bool HandleKey(VirtualKey key, bool shift)
    {
        if (Active is not { } cell) return false;
        switch (key)
        {
            case VirtualKey.Enter:
                if (_popup.IsOpen && _suggestions.SelectedItem is string pick) Accept(pick, finish: false);
                if (Finish()) Move(cell, LensMove.Down);
                return true;
            case VirtualKey.Tab:
                if (Finish()) Move(cell, shift ? LensMove.Left : LensMove.Right);
                return true;
            case VirtualKey.Up or VirtualKey.Down when _popup.IsOpen:
                var n = _suggestions.Items.Count;
                if (n > 0) _suggestions.SelectedIndex = key == VirtualKey.Down
                    ? Math.Min(_suggestions.SelectedIndex + 1, n - 1)
                    : Math.Max(_suggestions.SelectedIndex - 1, -1);
                if (_suggestions.SelectedItem is { } item) _suggestions.ScrollIntoView(item);
                return true;
            case VirtualKey.Up or VirtualKey.Down:
                if (Finish()) Move(cell, key == VirtualKey.Up ? LensMove.Up : LensMove.Down);
                return true;
            case VirtualKey.Escape:
                if (_popup.IsOpen)
                {
                    ClosePopup();
                    return true;
                }
                _edit?.Revert();
                Active = null;
                ClosePopup();
                _box.Visibility = Visibility.Collapsed;
                Closed();
                Cancelled();
                return true;
            default:
                return false;
        }
    }

    void Accept(string name, bool finish)
    {
        SetText(name);
        ClosePopup();
        if (finish) Finish();
    }

    void ShowSuggestions()
    {
        var q = _box.Text.Trim().ToUpperInvariant();
        var matches = q.Length == 0 ? [] : _completions
            .Where(n => n.ToUpperInvariant().StartsWith(q, StringComparison.Ordinal) && n.ToUpperInvariant() != q)
            .Take(12).ToList();
        if (matches.Count == 0)
        {
            ClosePopup();
            return;
        }
        _suggestions.ItemsSource = matches;
        _suggestions.SelectedIndex = -1;
        ((FrameworkElement)_popup.Child).MinWidth = Math.Max(160, _box.Width);
        _popup.IsOpen = true;
    }

    void ClosePopup()
    {
        _popup.IsOpen = false;
    }
}
