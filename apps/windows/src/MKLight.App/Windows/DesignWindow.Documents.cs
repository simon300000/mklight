using MKLight.App.Support;

namespace MKLight.App.Windows;

/// <summary>
/// The design window's document commands: Revert to Saved, Duplicate, Rename, Replace with Sample,
/// autosave and dropped files. The menu bar calls these.
/// </summary>
public sealed partial class DesignWindow
{
    /// <summary>Called once from the constructor (drag and drop, the autosave timer).</summary>
    void InitDocuments()
    {
    }

    /// <summary>Revert to Saved is possible: a saved design with changes.</summary>
    public bool CanRevert => _session.Document.Path is not null && _session.Document.IsDirty;

    /// <summary>Rename is possible: the design has a file.</summary>
    public bool CanRename => _session.Document.Path is not null;

    public Task RevertAsync() => Task.CompletedTask;

    public Task DuplicateAsync() => Task.CompletedTask;

    public Task RenameAsync() => Task.CompletedTask;

    /// <summary>Lens ▸ Replace with Sample: the sample's system replaces the design's (undoable).</summary>
    public void ReplaceWithSample(string sampleId)
    {
    }
}
