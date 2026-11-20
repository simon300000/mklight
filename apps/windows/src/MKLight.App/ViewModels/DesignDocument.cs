using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// The design state of one open document (the Mac app's DesignDocument):
/// the <see cref="OpticalSystem"/> value and the project's other sections.
/// Every edit replaces the system with a new value and keeps the previous
/// one on the undo stack, so undo and redo restore whole design states;
/// edits that invalidate overrides, solves, mechanics or variables drop
/// them in the same step (<see cref="OpticalSystem.DropReferencesInvalidated"/>).
/// </summary>
public sealed class DesignDocument
{
    sealed record Step(OpticalSystem System, string ActionName, int Revision);

    readonly Stack<Step> _undo = new();
    readonly Stack<Step> _redo = new();
    int _savedRevision;
    int _revision;
    int _nextRevision;

    public DesignDocument(ProjectFile project, string? path = null)
    {
        Project = project;
        Path = path;
    }

    public static DesignDocument FromSample(string id)
    {
        var system = SampleSystems.Get(id) ?? throw new ArgumentException($"No sample named “{id}”.");
        return new DesignDocument(new ProjectFile(system));
    }

    public static DesignDocument Open(string path) => new(ProjectFile.Read(path), path);

    public ProjectFile Project { get; private set; }
    public OpticalSystem System => Project.System;

    /// <summary>The file the design was opened from or last saved to; null while untitled.</summary>
    public string? Path { get; private set; }

    public string DisplayName => Path is null ? "Untitled" : global::System.IO.Path.GetFileName(Path);
    public bool IsDirty => _revision != _savedRevision;

    public bool CanUndo => _undo.Count > 0;
    public bool CanRedo => _redo.Count > 0;
    public string? UndoActionName => _undo.Count > 0 ? _undo.Peek().ActionName : null;
    public string? RedoActionName => _redo.Count > 0 ? _redo.Peek().ActionName : null;

    /// <summary>Raised after every change of the system (edit, undo, redo, replace).</summary>
    public event Action? Changed;

    /// <summary>
    /// Apply an edit as one undoable step. <paramref name="change"/> returns the
    /// new system, or null when the edit is refused; an edit that changes
    /// nothing is not recorded. Returns whether the design changed.
    /// </summary>
    public bool Edit(string actionName, Func<OpticalSystem, OpticalSystem?> change)
    {
        var before = System;
        var after = change(before);
        if (after is null || after == before) return false;
        after = after.DropReferencesInvalidated(before);
        _undo.Push(new Step(before, actionName, _revision));
        _redo.Clear();
        SetSystem(after, ++_nextRevision);
        return true;
    }

    /// <summary>Replace the whole system (a solve or optimisation result) as one undoable step.</summary>
    public void Replace(OpticalSystem system, string actionName) => Edit(actionName, _ => system);

    public void Undo()
    {
        if (_undo.Count == 0) return;
        var step = _undo.Pop();
        _redo.Push(step with { System = System, Revision = _revision });
        SetSystem(step.System, step.Revision);
    }

    public void Redo()
    {
        if (_redo.Count == 0) return;
        var step = _redo.Pop();
        _undo.Push(step with { System = System, Revision = _revision });
        SetSystem(step.System, step.Revision);
    }

    void SetSystem(OpticalSystem system, int revision)
    {
        Project = Project with { System = system };
        _revision = revision;
        Changed?.Invoke();
    }

    /// <summary>Replace a project section (views, …) without an undo step; saved with the next save.</summary>
    public void SetSection(string key, JsonTree value) => Project = Project with { Sections = Project.Sections.Set(key, value) };

    public void Save(string? path = null, string? engineVersion = null)
    {
        var target = path ?? Path ?? throw new InvalidOperationException("An untitled design needs a path.");
        var version = typeof(DesignDocument).Assembly.GetName().Version?.ToString(3) ?? "0.1.0";
        Project.StampMetadata("MKLight for Windows", version, engineVersion).Write(target);
        Path = target;
        _savedRevision = _revision;
    }
}
