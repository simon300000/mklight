using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;
using Microsoft.UI.Xaml.Media;
using MKLight.App.ViewModels;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;
using Windows.System;
using Windows.UI.Core;

namespace MKLight.App.Views;

/// <summary>
/// The lens data table (the Mac app's <c>LensTableView</c>): one row per surface with what every surface
/// has (surface, type, comment, radius, thickness, material, semi-diameter, conic), edited in place.
/// <para>
/// Keyboard: ↑/↓ select rows; Enter, F2 or a double click edits (the radius, or the comment where there is
/// none); in a cell, Tab / Shift+Tab move across, ↑/↓ and Enter move between rows committing, Esc reverts.
/// Ctrl+Shift+I / Ctrl+Alt+I insert before / after, Ctrl+Delete deletes, Ctrl+Shift+T makes the stop,
/// Ctrl+Shift+V makes the edited (or last clicked) value a variable or fixes it again.
/// </para>
/// Every commit is one undoable <see cref="DesignDocument.Edit"/>; the window sets <see cref="Host"/>.
/// </summary>
public sealed partial class LensDataPanel : UserControl
{
    const double RowHeight = 26;
    const double Inset = 8;

    readonly List<LensRowView> _views = [];
    readonly LensCellEditor _editor;
    IReadOnlyList<LensRow> _rows = [];
    IReadOnlyList<LensGridColumn> _columns = LensGridLayout.Columns(false);
    double[] _widths = [];
    ILensDataHost? _host;
    DesignDocument? _document;
    bool _focusWithin;
    /// <summary>The row of the last press, and whether it was selected before (the next click on a selected row edits).</summary>
    (int Row, bool WasSelected) _press = (-1, false);
    /// <summary>The value last pressed (the variable toggle's target outside an edit).</summary>
    LensColumn? _pressedColumn;

    public LensDataPanel()
    {
        InitializeComponent();
        _editor = new LensCellEditor(Overlay, Res)
        {
            Commit = Commit,
            Move = Move,
            Cancelled = () => Focus(FocusState.Programmatic),
            Closed = OnEditorClosed,
        };
        Loaded += (_, _) => Attach();
        Unloaded += (_, _) => Detach();
        GotFocus += (_, _) => UpdateFocusWithin();
        LostFocus += (_, _) => UpdateFocusWithin();
        KeyDown += OnKeyDown;
        RowsHost.LayoutUpdated += (_, _) => PlaceEditor();
        BuildHeader();
    }

    /// <summary>The design window's session: document, selection, engine data. Set by the window.</summary>
    public ILensDataHost? Host
    {
        get => _host;
        set
        {
            if (ReferenceEquals(_host, value)) return;
            Detach();
            _host = value;
            if (IsLoaded) Attach();
        }
    }

    // MARK: Commands (also for the window's menus)

    /// <summary>Inserts a surface before the selection (before the image when nothing is selected) and selects it.</summary>
    public void InsertBefore() => Insert(after: false);

    /// <summary>Inserts a surface after the selection (before the image at the latest) and selects it.</summary>
    public void InsertAfter() => Insert(after: true);

    /// <summary>Deletes the selected surface (not the object or the image) and selects the one that took its place.</summary>
    public void DeleteSelected()
    {
        if (_document is not { } doc || Selected is not { } i || !LensCommands.CanDelete(doc.System, i)) return;
        CommitEditing();
        if (doc.Edit("Delete Surface", s => new LensEdit.Delete(i).Apply(s)))
            Select(LensCommands.SelectionAfterDelete(doc.System, i));
    }

    /// <summary>Makes the selected surface the aperture stop.</summary>
    public void MakeSelectedStop()
    {
        if (_document is not { } doc || Selected is not { } i || !LensCommands.CanMakeStop(doc.System, i)) return;
        var edit = new LensEdit.MakeStop(i);
        doc.Edit(edit.ActionName, edit.Apply);
    }

    /// <summary>Edits the selected row's radius (its comment where it has none), or <paramref name="column"/>.</summary>
    public void BeginEdit(LensColumn? column = null)
    {
        if (Selected is not { } i || i >= _rows.Count) return;
        var row = _rows[i];
        BeginEdit(i, column ?? (row.Editable.Contains(LensColumn.Radius) ? LensColumn.Radius : LensColumn.Comment));
    }

    /// <summary>
    /// Makes the value of <paramref name="column"/> in the selected row a variable, or fixes it again (one undoable
    /// step); a value a solve sets is refused with the reason in a toast. Without a column: the edited value, else
    /// the one last clicked, else the radius.
    /// </summary>
    public void ToggleVariable(LensColumn? column = null)
    {
        if (_document is not { } doc || Selected is not { } i || i >= doc.System.Surfaces.Count) return;
        column ??= _editor.Active?.Column ?? _pressedColumn ?? LensColumn.Radius;
        if (VariablePath(doc.System, i, column.Value) is not { } path)
        {
            _host?.ShowToast($"The {column.Value.Title().ToLowerInvariant()} of this surface cannot be a variable.");
            return;
        }
        // Typed text goes to its value first; the field stays open.
        if (_editor.Active is not null && !_editor.Finish()) return;
        if (DesignVariables.Toggle(doc, path) is { } refusal) _host?.ShowToast(refusal);
    }

    /// <summary>The parameter a variable of this value would vary (a catalog glass's index offset for the material), or null.</summary>
    static ParamPath? VariablePath(OpticalSystem s, int i, LensColumn column)
    {
        var sf = s.Surfaces[i];
        if (i == 0 || i == s.ImageIndex && column != LensColumn.SemiDiameter) return null;
        if (column == LensColumn.Material)
            return sf.Interaction == Interaction.Refract && sf.Material is MaterialRef.CatalogGlass ? new ParamPath.MaterialDnD(sf.Id) : null;
        return column.Param(sf);
    }

    /// <summary>Gives the table the keyboard focus (row selection with ↑/↓).</summary>
    public void FocusTable() => Focus(FocusState.Programmatic);

    void Insert(bool after)
    {
        if (_document is not { } doc) return;
        CommitEditing();
        var s = doc.System;
        var edit = after ? LensCommands.InsertAfter(s, Selected ?? s.ImageIndex - 1) : LensCommands.InsertBefore(s, Selected ?? s.ImageIndex);
        if (doc.Edit(edit.ActionName, edit.Apply) && edit is LensEdit.Insert { At: var at }) Select(at);
    }

    /// <summary>Commits what the editor holds before a structural edit, so the typed text goes to its surface.</summary>
    void CommitEditing()
    {
        if (_editor.Active is null) return;
        _editor.Finish();
        _editor.Close();
    }

    // MARK: Host

    void Attach()
    {
        if (_host is null || _document is not null) return;
        _document = _host.Document;
        _document.Changed += Refresh;
        _host.SelectionChanged += OnHostSelectionChanged;
        _host.EngineDataChanged += Refresh;
        Refresh();
    }

    void Detach()
    {
        if (_document is not null) _document.Changed -= Refresh;
        if (_host is not null)
        {
            _host.SelectionChanged -= OnHostSelectionChanged;
            _host.EngineDataChanged -= Refresh;
        }
        _document = null;
    }

    int? Selected => _host?.SelectedSurface;

    void Select(int index)
    {
        if (_host is null || _rows.Count == 0) return;
        var i = Math.Clamp(index, 0, _rows.Count - 1);
        if (_host.SelectedSurface != i) _host.SelectedSurface = i;
        UpdateSelection();
        BringIntoView(i);
    }

    void OnHostSelectionChanged()
    {
        UpdateSelection();
        if (Selected is { } i) BringIntoView(i);
    }

    void BringIntoView(int i)
    {
        if (i < 0 || i >= _views.Count) return;
        var top = i * RowHeight;
        var offset = RowScroller.VerticalOffset;
        var height = RowScroller.ViewportHeight;
        if (top < offset) RowScroller.ChangeView(null, top, null, disableAnimation: true);
        else if (top + RowHeight + Inset > offset + height) RowScroller.ChangeView(null, top + RowHeight + Inset - height, null, disableAnimation: true);
    }

    // MARK: Rows

    void Refresh()
    {
        if (_host is null || _document is null) return;
        var doc = _document;
        var system = doc.System;
        var variables = doc.Project.Sections.TryGetValue("variables", out var section)
            ? LensRows.Variables(section) : new Dictionary<ParamPath, VariableBounds>();
        _rows = LensRows.Build(system, _host.AutoSemiDiameters, variables);
        Caption.Text = LensRows.Caption(system);
        VariablesButton.Visibility = variables.Count > 0 ? Visibility.Visible : Visibility.Collapsed;
        VariablesText.Text = variables.Count == 1 ? "1 variable" : $"{variables.Count} variables";

        var columns = LensGridLayout.Columns(_rows.Any(r => r.Notes.ContainsKey(LensColumn.Material)));
        if (!columns.SequenceEqual(_columns))
        {
            _columns = columns;
            Layout();
        }
        while (_views.Count < _rows.Count) _views.Add(NewRowView());
        while (_views.Count > _rows.Count)
        {
            RowsPanel.Children.Remove(_views[^1]);
            _views.RemoveAt(_views.Count - 1);
        }
        var issues = _host.IssuesBySurface;
        for (var i = 0; i < _rows.Count; i++)
        {
            var row = _rows[i];
            _views[i].Update(row, issues.TryGetValue(i, out var list) ? list : [], _editor.Active is { } a && a.Row == row.Id ? a.Column : null);
            _views[i].SetWidths(_widths);
        }
        UpdateSelection();
        UpdateButtons();

        // The editor follows its surface: unedited text takes the new value; a surface that went away ends the edit.
        if (_editor.Active is { } cell)
        {
            var r = _rows.FirstOrDefault(x => x.Id == cell.Row);
            if (r is null || !r.Editable.Contains(cell.Column)) _editor.Close();
            else _editor.ValueChanged(r.ExactText(cell.Column));
        }
    }

    LensRowView NewRowView()
    {
        var view = new LensRowView(Res);
        view.AddHandler(PointerPressedEvent, new PointerEventHandler(OnRowPressed), handledEventsToo: true);
        view.Tapped += OnRowTapped;
        view.DoubleTapped += OnRowDoubleTapped;
        view.ContextRequested += OnRowContextRequested;
        view.TypeMenuRequested += ShowTypeMenu;
        RowsPanel.Children.Add(view);
        return view;
    }

    int IndexOf(object sender) => sender is LensRowView v ? _views.IndexOf(v) : -1;

    /// <summary>The row of surface <paramref name="id"/> (its index), or -1.</summary>
    int RowIndex(SurfaceID id)
    {
        for (var i = 0; i < _rows.Count; i++)
            if (_rows[i].Id == id) return i;
        return -1;
    }

    void UpdateSelection()
    {
        var selected = Selected;
        for (var i = 0; i < _views.Count; i++) _views[i].SetSelection(i == selected, _focusWithin, i % 2 == 1);
    }

    void UpdateButtons()
    {
        var s = _document?.System;
        InsertButton.IsEnabled = s is not null;
        DeleteButton.IsEnabled = s is not null && Selected is { } i && LensCommands.CanDelete(s, i);
    }

    void UpdateFocusWithin()
    {
        var focused = XamlRoot is null ? null : FocusManager.GetFocusedElement(XamlRoot) as DependencyObject;
        var within = false;
        for (var e = focused; e is not null; e = VisualTreeHelper.GetParent(e))
        {
            if (ReferenceEquals(e, this))
            {
                within = true;
                break;
            }
        }
        if (within == _focusWithin) return;
        _focusWithin = within;
        UpdateSelection();
    }

    // MARK: Layout

    void OnScrollerSizeChanged(object sender, SizeChangedEventArgs e) => Layout();

    /// <summary>Shares the width among the columns (the grid scrolls sideways below its minimum width).</summary>
    void Layout()
    {
        var available = HorizontalScroller.ActualWidth;
        if (available <= 0) return;
        var width = Math.Max(available, LensGridLayout.MinimumWidth(_columns) + 2 * Inset);
        Table.Width = width;
        _widths = LensGridLayout.Widths(_columns, width - 2 * Inset);
        for (var i = 0; i < _widths.Length; i++) ColumnHeader.ColumnDefinitions[i].Width = new GridLength(_widths[i]);
        foreach (var v in _views) v.SetWidths(_widths);
    }

    /// <summary>The column titles, pinned over the rows, with a short divider between titles.</summary>
    void BuildHeader()
    {
        var k = 0;
        foreach (var column in LensColumns.Visible)
        {
            ColumnHeader.ColumnDefinitions.Add(new ColumnDefinition());
            var title = new TextBlock
            {
                Text = column.Title(),
                Style = (Style)Application.Current.Resources["ColumnHeaderText"],
                Margin = new Thickness(column == LensColumn.Type ? 7 : 6, 0, 6, 0),
            };
            Grid.SetColumn(title, k);
            ColumnHeader.Children.Add(title);
            if (k < LensColumns.Visible.Length - 1)
            {
                var divider = new Border { Style = Res("LensHeaderDivider") };
                Grid.SetColumn(divider, k);
                ColumnHeader.Children.Add(divider);
            }
            k++;
        }
    }

    // MARK: Pointer

    void OnRowPressed(object sender, PointerRoutedEventArgs e)
    {
        var i = IndexOf(sender);
        if (i < 0) return;
        _press = (i, Selected == i);
        _pressedColumn = ((LensRowView)sender).ColumnAt(e.GetCurrentPoint((UIElement)sender).Position) ?? _pressedColumn;
        Select(i);
        // A press elsewhere ends the edit (the field commits as it loses the focus).
        if (e.OriginalSource is not TextBox) Focus(FocusState.Pointer);
    }

    void OnRowTapped(object sender, TappedRoutedEventArgs e)
    {
        var i = IndexOf(sender);
        // Like a table: the first click selects the row, the next one edits the value under it.
        if (i < 0 || _press != (i, true)) return;
        if (((LensRowView)sender).ColumnAt(e.GetPosition((UIElement)sender)) is { } c && _rows[i].Editable.Contains(c)) BeginEdit(i, c);
    }

    void OnRowDoubleTapped(object sender, DoubleTappedRoutedEventArgs e)
    {
        var i = IndexOf(sender);
        if (i < 0) return;
        var c = ((LensRowView)sender).ColumnAt(e.GetPosition((UIElement)sender));
        if (c is { } col && _rows[i].Editable.Contains(col)) BeginEdit(i, col);
        else if (_editor.Active is null) BeginEdit(i, _rows[i].Editable.Contains(LensColumn.Radius) ? LensColumn.Radius : LensColumn.Comment);
    }

    // MARK: Editing

    void BeginEdit(int i, LensColumn column)
    {
        if (i < 0 || i >= _rows.Count || !_rows[i].Editable.Contains(column)) return;
        var row = _rows[i];
        if (_editor.Active is { } a && a == new LensCell(row.Id, column)) return;
        // Text typed in the cell being left is committed first (or, when rejected, the edit stays there).
        if (_editor.Active is not null && !_editor.Finish()) return;
        row = _rows[i];
        Select(i);
        ShowSlotText(_editor.Active);
        _editor.Begin(new LensCell(row.Id, column), row.ExactText(column), column.IsNumeric(),
            column == LensColumn.Material ? _host?.MaterialNames ?? [] : []);
        if (_views[i].Slots.TryGetValue(column, out var slot)) slot.Text.Opacity = 0;
        PlaceEditor();
    }

    /// <summary>Puts the field over the edited value, the text inset as in the cell.</summary>
    void PlaceEditor()
    {
        if (_editor.Active is not { } cell) return;
        var i = RowIndex(cell.Row);
        if (i < 0 || !_views[i].Slots.TryGetValue(cell.Column, out var slot) || slot.Area.ActualWidth <= 0) return;
        var r = slot.Area.TransformToVisual(Overlay).TransformBounds(new global::Windows.Foundation.Rect(0, 0, slot.Area.ActualWidth, slot.Area.ActualHeight));
        _editor.Place(new global::Windows.Foundation.Rect(r.X - 6, r.Y, r.Width + 12, r.Height));
    }

    void ShowSlotText(LensCell? cell)
    {
        if (cell is not { } c) return;
        var i = RowIndex(c.Row);
        if (i >= 0 && _views[i].Slots.TryGetValue(c.Column, out var slot)) slot.Text.Opacity = 1;
    }

    void OnEditorClosed()
    {
        foreach (var v in _views)
            foreach (var slot in v.Slots.Values) slot.Text.Opacity = 1;
    }

    /// <summary>
    /// Commits typed text to its surface (wherever the selection is now) as one undoable step. Rejected text
    /// is explained in a toast; the field then shows the value again.
    /// </summary>
    bool Commit(LensCell cell, string text)
    {
        if (_document is not { } doc || _rows.FirstOrDefault(r => r.Id == cell.Row) is not { } row) return true;
        if (row.Commit(cell.Column, text) is not { } parsed) return true;
        if (!parsed.Ok)
        {
            _host?.ShowToast(parsed.Error!);
            return false;
        }
        doc.Edit(parsed.Value.ActionName, parsed.Value.Apply);
        return true;
    }

    /// <summary>Tab / Shift+Tab across the row's values; ↑/↓ and Enter to the same value of the next row that has one.</summary>
    void Move(LensCell from, LensMove dir)
    {
        var r = RowIndex(from.Row);
        if (r < 0)
        {
            EndEditing();
            return;
        }
        switch (dir)
        {
            case LensMove.Up or LensMove.Down:
                // An x section continues in the next row's x section, or its column.
                LensColumn? Target(LensRow row) =>
                    row.Editable.Contains(from.Column) ? from.Column : row.Editable.Contains(from.Column.Base()) ? from.Column.Base() : null;
                var step = dir == LensMove.Up ? -1 : 1;
                var k = r + step;
                while (k >= 0 && k < _rows.Count && Target(_rows[k]) is null) k += step;
                if (k >= 0 && k < _rows.Count) BeginEdit(k, Target(_rows[k])!.Value);
                else EndEditing();
                break;
            default:
                var cols = LensColumns.Editable;
                var c = Array.IndexOf(cols, from.Column);
                do c += dir == LensMove.Left ? -1 : 1;
                while (c >= 0 && c < cols.Length && !_rows[r].Editable.Contains(cols[c]));
                if (c >= 0 && c < cols.Length) BeginEdit(r, cols[c]);
                break;
        }
    }

    /// <summary>Leaves the cell, back to row selection.</summary>
    void EndEditing()
    {
        _editor.Close();
        Focus(FocusState.Programmatic);
    }

    // MARK: Keyboard

    static bool IsDown(VirtualKey key) =>
        InputKeyboardSource.GetKeyStateForCurrentThread(key).HasFlag(CoreVirtualKeyStates.Down);

    void OnKeyDown(object sender, KeyRoutedEventArgs e) =>
        e.Handled = HandleKey(e.Key, IsDown(VirtualKey.Control), IsDown(VirtualKey.Shift), IsDown(VirtualKey.Menu), fromEditor: false);

    /// <summary>
    /// A key pressed in the table: the editor's keys first (they reach it before the table), then row
    /// selection, editing and the surface commands. Returns whether the key was used.
    /// </summary>
    internal bool HandleKey(VirtualKey key, bool ctrl, bool shift, bool alt, bool fromEditor = true)
    {
        if (fromEditor && !ctrl && !alt && _editor.HandleKey(key, shift)) return true;
        var editing = _editor.Active is not null;
        switch (key)
        {
            case VirtualKey.I when ctrl && shift && !alt:
                InsertBefore();
                break;
            case VirtualKey.I when ctrl && alt && !shift:
                InsertAfter();
                break;
            case VirtualKey.T when ctrl && shift:
                MakeSelectedStop();
                break;
            case VirtualKey.V when ctrl && shift && !alt:
                ToggleVariable();
                break;
            case VirtualKey.Delete when ctrl && !editing:
                DeleteSelected();
                break;
            case VirtualKey.Up or VirtualKey.Down when !editing && !ctrl:
                Select(Selected is { } i ? i + (key == VirtualKey.Up ? -1 : 1) : key == VirtualKey.Down ? 0 : _rows.Count - 1);
                break;
            case VirtualKey.Home or VirtualKey.End when !editing:
                Select(key == VirtualKey.Home ? 0 : _rows.Count - 1);
                break;
            case VirtualKey.Enter or VirtualKey.F2 when !editing:
                BeginEdit();
                break;
            default:
                return false;
        }
        return true;
    }

    // MARK: Menus

    void OnInsertClick(object sender, RoutedEventArgs e) => InsertBefore();

    void OnDeleteClick(object sender, RoutedEventArgs e) => DeleteSelected();

    void OnVariablesClick(object sender, RoutedEventArgs e) => (_host as DesignSession)?.ShowInspectorPage("optimize");

    void OnRowContextRequested(UIElement sender, ContextRequestedEventArgs e)
    {
        var i = IndexOf(sender);
        if (i < 0 || _document is null) return;
        Select(i);
        var s = _document.System;
        var menu = new MenuFlyout();
        MenuFlyoutItem Item(string text, string glyph, string keys, bool enabled, Action action)
        {
            var item = new MenuFlyoutItem { Text = text, IsEnabled = enabled, KeyboardAcceleratorTextOverride = keys };
            if (glyph.Length > 0) item.Icon = new FontIcon { Glyph = glyph };
            item.Click += (_, _) => action();
            return item;
        }
        menu.Items.Add(Item("Insert Surface Before", "\uE710", "Ctrl+Shift+I", true, InsertBefore));
        menu.Items.Add(Item("Insert Surface After", "", "Ctrl+Alt+I", true, InsertAfter));
        menu.Items.Add(Item("Delete Surface", "\uE738", "Ctrl+Delete", LensCommands.CanDelete(s, i), DeleteSelected));
        menu.Items.Add(new MenuFlyoutSeparator());
        menu.Items.Add(Item("Make Aperture Stop", "\uE7C8", "Ctrl+Shift+T", LensCommands.CanMakeStop(s, i), MakeSelectedStop));
        menu.Items.Add(new MenuFlyoutSeparator());
        var under = e.TryGetPosition(sender, out var at) ? ((LensRowView)sender).ColumnAt(at) : null;
        if (under is { } col && VariablePath(s, i, col) is { } own)
        {
            var on = DesignVariables.IsVariable(_document, own);
            var refusal = on ? null : DesignVariables.Refusal(s, own);
            var item = Item(on ? "Remove Variable" : "Make Variable", "", "Ctrl+Shift+V", refusal is null, () => ToggleVariable(col));
            if (refusal is not null) ToolTipService.SetToolTip(item, refusal);
            menu.Items.Add(item);
        }
        menu.Items.Add(VariablesMenu(s, i));
        if (e.TryGetPosition(sender, out var p)) menu.ShowAt(sender, p);
        else menu.ShowAt((FrameworkElement)sender);
        e.Handled = true;
    }

    /// <summary>
    /// The row's Variables submenu: each value of the surface that can vary, checked when it is a variable; a value a
    /// solve sets is disabled with the solve named in its tooltip.
    /// </summary>
    MenuFlyoutSubItem VariablesMenu(OpticalSystem s, int i)
    {
        var sub = new MenuFlyoutSubItem { Text = "Variables", Icon = new FontIcon { Glyph = "\uE9D9" } };
        var seen = new HashSet<ParamPath>();
        foreach (var column in LensColumns.Editable)
        {
            if (VariablePath(s, i, column) is not { } path || s.ValueAt(path) is null || !seen.Add(s.Canonical(path))) continue;
            var on = _document is { } doc && DesignVariables.IsVariable(doc, path);
            var refusal = on ? null : DesignVariables.Refusal(s, path);
            var item = new ToggleMenuFlyoutItem { Text = path.Label(s), IsChecked = on, IsEnabled = refusal is null };
            if (refusal is not null) ToolTipService.SetToolTip(item, refusal);
            var c = column;
            item.Click += (_, _) => ToggleVariable(c);
            sub.Items.Add(item);
        }
        sub.IsEnabled = sub.Items.Count > 0;
        return sub;
    }

    /// <summary>The type menu, grouped Spherical, Aspheric, Freeform, Anamorphic, Diffractive, Special; the current type checked.</summary>
    void ShowTypeMenu(LensRowView view, Button button)
    {
        if (view.Row is not { } row || _document is not { } doc) return;
        var menu = new MenuFlyout { Placement = Microsoft.UI.Xaml.Controls.Primitives.FlyoutPlacementMode.BottomEdgeAlignedLeft };
        foreach (var group in Enum.GetValues<SurfaceTypeGroup>())
        {
            if (menu.Items.Count > 0) menu.Items.Add(new MenuFlyoutSeparator());
            menu.Items.Add(new MenuFlyoutItem { Text = SurfaceTypeCatalog.Title(group), IsEnabled = false, FontSize = 12 });
            foreach (var entry in SurfaceTypeCatalog.All.Where(x => x.Group == group))
            {
                var item = new RadioMenuFlyoutItem
                {
                    Text = entry.Title, GroupName = "type", IsChecked = entry.Choice == row.TypeChoice,
                    Icon = new FontIcon { Glyph = entry.Glyph },
                };
                var id = row.Id;
                item.Click += (_, _) =>
                {
                    var edit = new LensEdit.SetType(id, entry.Choice);
                    if (entry.Choice != _rows.FirstOrDefault(r => r.Id == id)?.TypeChoice) doc.Edit(edit.ActionName, edit.Apply);
                };
                menu.Items.Add(item);
            }
        }
        menu.ShowAt(button);
    }

    Style Res(string key) => (Style)Resources[key];
}
