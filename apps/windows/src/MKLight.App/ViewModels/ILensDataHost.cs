using MKLight.App.Views;

namespace MKLight.App.ViewModels;

/// <summary>
/// What the lens data table reads from and writes to its design window:
/// the document (edits go through <see cref="DesignDocument.Edit"/>, one
/// undoable step each), the selection, the engine's validation issues and
/// automatic semi-diameters, glass names for completion, and toasts.
/// Implemented by the window's DesignSession.
/// </summary>
public interface ILensDataHost
{
    DesignDocument Document { get; }

    /// <summary>The selected surface index (0 = object), or null.</summary>
    int? SelectedSurface { get; set; }
    event Action? SelectionChanged;

    /// <summary>Validation issues by surface index (issues without a surface are not listed here).</summary>
    IReadOnlyDictionary<int, IReadOnlyList<IssueItem>> IssuesBySurface { get; }

    /// <summary>Semi-diameters the engine resolved for automatic apertures, by surface index (from the layout).</summary>
    IReadOnlyDictionary<int, double> AutoSemiDiameters { get; }

    /// <summary>Glass names of the session's catalogs, for completion in the material column.</summary>
    IReadOnlyList<string> MaterialNames { get; }

    /// <summary>Raised when issues, automatic semi-diameters or material names change.</summary>
    event Action? EngineDataChanged;

    /// <summary>A transient message (glass toast), e.g. why typed input was rejected.</summary>
    void ShowToast(string message);
}
