using Microsoft.UI.Input;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Automation;
using Microsoft.UI.Xaml.Automation.Peers;
using Microsoft.UI.Xaml.Automation.Provider;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;

namespace MKLight.App.Views;

/// <summary>What a lens cell asks of its table (the panel): edits go through the same path as typing.</summary>
interface ILensCellActions
{
    /// <summary>Commits <paramref name="text"/> as if typed into the cell; throws <see cref="ArgumentException"/> with the reason when it is rejected.</summary>
    void SetValue(LensRow row, LensColumn column, string text);

    /// <summary>Begins editing the cell (the type cell opens its menu).</summary>
    void Invoke(LensRow row, LensColumn column, FrameworkElement cell);

    bool IsSelected(LensRow row);
    void Select(LensRow row);
}

/// <summary>A lens row for automation: a data item (named "Surface 3", "Object", …) that can be selected.</summary>
sealed partial class LensRowAutomationPeer(LensRowView owner, ILensCellActions actions) : FrameworkElementAutomationPeer(owner), ISelectionItemProvider
{
    LensRow? Row => ((LensRowView)Owner).Row;

    protected override object? GetPatternCore(PatternInterface pattern) =>
        pattern == PatternInterface.SelectionItem ? this : base.GetPatternCore(pattern);

    protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.DataItem;
    protected override string GetClassNameCore() => nameof(LensRowView);
    protected override bool IsControlElementCore() => true;

    public bool IsSelected => Row is { } r && actions.IsSelected(r);
    public IRawElementProviderSimple? SelectionContainer => null;

    public void Select()
    {
        if (Row is { } r) actions.Select(r);
    }

    public void AddToSelection() => Select();

    public void RemoveFromSelection()
    {
        // One row is always selected once one was: nothing to remove.
    }
}

/// <summary>
/// A cell of the lens grid (one value, an x section, the surface label or the type): the hit target the editor
/// covers, and an automation element with the Value pattern (read the value, set it as typing would) and the
/// Invoke pattern (begin editing). Its automation id is <c>LensCell.{row}.{column}</c> with the column's stable key.
/// </summary>
sealed partial class LensCellView : Grid
{
    public LensCellView(LensRow row, LensColumn column, ILensCellActions actions)
    {
        Row = row;
        Column = column;
        Actions = actions;
        AutomationProperties.SetAutomationId(this, AutomationId(row.Index, column));
        AutomationProperties.SetName(this, Loc.F("Lens_Cell_Name", Title(row, column), RowName(row)));
        var status = new List<string>();
        if (column == LensColumn.SemiDiameter && row.SemiDiameterIsAuto) status.Add(Loc.S("Lens_Cell_Automatic"));
        if (row.IsFlatSection(column)) status.Add(Loc.S("Lens_Cell_FlatSection"));
        if (row.Notes.TryGetValue(column, out var note)) status.Add(Loc.F("Lens_Cell_MeltOffset", note));
        if (status.Count > 0) AutomationProperties.SetItemStatus(this, string.Join(", ", status));
    }

    public LensRow Row { get; }
    public LensColumn Column { get; }
    internal ILensCellActions Actions { get; }

    /// <summary>A value typed text edits (the type takes a type's title).</summary>
    public bool IsEditable => Column == LensColumn.Type ? Row.Role is not (LensRowRole.Object or LensRowRole.Image) : Row.Editable.Contains(Column);

    public static string AutomationId(int row, LensColumn column) => $"LensCell.{row}.{LensColumnWidths.Key(column)}";

    /// <summary>The row as a screen reader names it.</summary>
    public static string RowName(LensRow row) => row.Role switch
    {
        LensRowRole.Object => Loc.S("Lens_Row_Object"),
        LensRowRole.Image => Loc.S("Lens_Row_Image"),
        LensRowRole.Stop => Loc.F("Lens_Row_SurfaceStop", row.Index),
        _ => Loc.F("Lens_Row_Surface", row.Index),
    };

    /// <summary>The value's title: "Radius", or "Radius y" / "Radius x" for the sections of an anamorphic surface.</summary>
    static string Title(LensRow row, LensColumn c) =>
        row.Hints.TryGetValue(c, out var hint) ? Loc.F("Lens_Cell_SectionTitle", c.Base().Title(), hint) : c.Title();

    protected override AutomationPeer OnCreateAutomationPeer() => new LensCellAutomationPeer(this);
}

sealed partial class LensCellAutomationPeer(LensCellView owner) : FrameworkElementAutomationPeer(owner), IValueProvider, IInvokeProvider
{
    LensCellView Cell => (LensCellView)Owner;

    protected override object? GetPatternCore(PatternInterface pattern) => pattern switch
    {
        PatternInterface.Value => this,
        PatternInterface.Invoke when Cell.IsEditable => this,
        _ => base.GetPatternCore(pattern),
    };

    protected override AutomationControlType GetAutomationControlTypeCore() =>
        Cell.IsEditable ? AutomationControlType.Edit : AutomationControlType.Text;

    protected override string GetClassNameCore() => nameof(LensCellView);
    protected override string GetLocalizedControlTypeCore() => Loc.S("Lens_Cell_ControlType");
    protected override bool IsControlElementCore() => true;
    protected override bool IsContentElementCore() => true;

    public bool IsReadOnly => !Cell.IsEditable;

    /// <summary>
    /// The text the cell shows (an automatic semi-diameter or a melt offset is the item status); the type cell's is
    /// its type-menu entry ("Sphere / Conic", "Cylinder (power in x)"), which setting the value takes.
    /// </summary>
    public string Value => Cell.Column == LensColumn.Type && Cell.IsEditable
        ? SurfaceTypeCatalog.EntryFor(Cell.Row.TypeChoice).Title
        : Cell.Row.Text(Cell.Column);

    public void SetValue(string value)
    {
        if (IsReadOnly) throw new ElementNotEnabledException();
        Cell.Actions.SetValue(Cell.Row, Cell.Column, value);
    }

    public void Invoke()
    {
        if (!Cell.IsEditable) throw new ElementNotEnabledException();
        Cell.Actions.Invoke(Cell.Row, Cell.Column, Cell);
    }
}

/// <summary>A grid that is one automation element (the panel header, the column header): plain grids have none.</summary>
public sealed partial class LensAutomationGroup : Grid
{
    public AutomationControlType ControlType { get; set; } = AutomationControlType.Group;

    protected override AutomationPeer OnCreateAutomationPeer() => new GroupPeer(this);

    sealed partial class GroupPeer(LensAutomationGroup owner) : FrameworkElementAutomationPeer(owner)
    {
        protected override AutomationControlType GetAutomationControlTypeCore() => ((LensAutomationGroup)Owner).ControlType;
        protected override string GetClassNameCore() => nameof(LensAutomationGroup);
        protected override bool IsControlElementCore() => true;
    }
}

/// <summary>
/// A header divider: drag it to resize the column on its left, double-click it to fit the column to its content.
/// For automation (<c>LensTable.Divider.{column}</c>) it is a range (the width, settable) and Invoke fits.
/// </summary>
sealed partial class LensColumnDivider : Grid
{
    public LensColumnDivider(LensColumn column)
    {
        Column = column;
        ProtectedCursor = InputSystemCursor.Create(InputSystemCursorShape.SizeWestEast);
        AutomationProperties.SetAutomationId(this, $"LensTable.Divider.{LensColumnWidths.Key(column)}");
        AutomationProperties.SetName(this, Loc.F("Lens_Divider_Name", column.Title()));
        AutomationProperties.SetHelpText(this, Loc.S("Lens_Divider_Help"));
    }

    public LensColumn Column { get; }

    /// <summary>The column's width and limits now.</summary>
    public Func<(double Width, LensGridColumn Limits)>? Current { get; set; }
    /// <summary>Sets the column's width (a held width, saved with the project).</summary>
    public Action<double>? Resize { get; set; }
    /// <summary>Fits the column to its content.</summary>
    public Action? Fit { get; set; }

    protected override AutomationPeer OnCreateAutomationPeer() => new LensColumnDividerAutomationPeer(this);
}

sealed partial class LensColumnDividerAutomationPeer(LensColumnDivider owner) : FrameworkElementAutomationPeer(owner), IRangeValueProvider, IInvokeProvider
{
    /// <summary>A finite maximum for columns that can grow without limit.</summary>
    const double Widest = 2000;

    LensColumnDivider Divider => (LensColumnDivider)Owner;

    (double Width, LensGridColumn Limits) Now => Divider.Current?.Invoke() ?? (0, default);

    protected override object? GetPatternCore(PatternInterface pattern) =>
        pattern is PatternInterface.RangeValue or PatternInterface.Invoke ? this : base.GetPatternCore(pattern);

    protected override AutomationControlType GetAutomationControlTypeCore() => AutomationControlType.Thumb;
    protected override string GetClassNameCore() => nameof(LensColumnDivider);
    protected override bool IsControlElementCore() => true;
    protected override bool IsContentElementCore() => false;

    public bool IsReadOnly => Divider.Resize is null;
    public double Value => Now.Width;
    public double Minimum => Now.Limits.Min;
    public double Maximum => Math.Min(Now.Limits.Max, Widest);
    public double SmallChange => 8;
    public double LargeChange => 32;

    public void SetValue(double value)
    {
        if (Divider.Resize is not { } resize) throw new ElementNotEnabledException();
        if (!double.IsFinite(value)) throw new ArgumentOutOfRangeException(nameof(value));
        resize(value);
    }

    public void Invoke() => Divider.Fit?.Invoke();
}
