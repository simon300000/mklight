using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// The design state of one open document (the Mac app's DesignDocument):
/// the <see cref="OpticalSystem"/> value and the project's other sections.
/// Every edit replaces the system (or a section) with a new value and keeps
/// the previous project on the undo stack, so undo and redo restore whole
/// design states; edits that invalidate overrides, solves, mechanics or
/// variables drop them in the same step
/// (<see cref="OpticalSystem.DropReferencesInvalidated"/>).
/// <para>
/// Undoable: the system and every section except the window layout
/// (<c>views</c>), which <see cref="SetSection"/> writes without an undo step
/// and undo leaves as it is (the Mac app's rule).
/// </para>
/// </summary>
public sealed class DesignDocument
{
    sealed record Step(ProjectFile Project, string ActionName, int Revision);

    /// <summary>Sections that are window state, not design: kept across undo and redo.</summary>
    static readonly string[] NotUndoable = ["views"];

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

    /// <summary>Raised after every undoable change (system or section), undo and redo.</summary>
    public event Action? Changed;

    /// <summary>Raised when the system changed (the workspace takes a new snapshot).</summary>
    public event Action? SystemChanged;

    /// <summary>Raised with a section's key when that section changed (by an edit, undo, redo or <see cref="SetSection"/>).</summary>
    public event Action<string>? SectionChanged;

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
        Commit(Project with { System = after }, actionName);
        return true;
    }

    /// <summary>Replace the whole system (a solve or optimisation result) as one undoable step.</summary>
    public void Replace(OpticalSystem system, string actionName) => Edit(actionName, _ => system);

    /// <summary>The section's current value (<see cref="JsonTree.Null"/> when absent).</summary>
    public JsonTree Section(string key) => Project.Sections.TryGetValue(key, out var v) ? v : JsonTree.Null;

    /// <summary>
    /// Replace a design section (<c>variables</c>, <c>merit</c>, <c>tolerances</c>, <c>catalogs</c>,
    /// <c>metadata</c>) as one undoable step; an equal value is not recorded. Returns whether it changed.
    /// </summary>
    public bool EditSection(string key, JsonTree value, string actionName)
    {
        if (Section(key) == value) return false;
        Commit(Project with { Sections = Project.Sections.Set(key, value) }, actionName);
        return true;
    }

    /// <summary>
    /// Edit the system and sections together as one undoable step (a solve result with its variables, a
    /// glass substitution with its catalogs). <paramref name="change"/> returns the new project or null.
    /// </summary>
    public bool EditProject(string actionName, Func<ProjectFile, ProjectFile?> change)
    {
        var after = change(Project);
        if (after is null || after == Project) return false;
        if (after.System != Project.System) after = after with { System = after.System.DropReferencesInvalidated(Project.System) };
        Commit(after, actionName);
        return true;
    }

    void Commit(ProjectFile after, string actionName)
    {
        _undo.Push(new Step(Project, actionName, _revision));
        _redo.Clear();
        Restore(after, ++_nextRevision);
    }

    public void Undo()
    {
        if (_undo.Count == 0) return;
        var step = _undo.Pop();
        _redo.Push(step with { Project = Project, Revision = _revision });
        Restore(step.Project, step.Revision);
    }

    public void Redo()
    {
        if (_redo.Count == 0) return;
        var step = _redo.Pop();
        _undo.Push(step with { Project = Project, Revision = _revision });
        Restore(step.Project, step.Revision);
    }

    /// <summary>Make <paramref name="project"/> current, keeping the window-state sections as they are now.</summary>
    void Restore(ProjectFile project, int revision)
    {
        var before = Project;
        foreach (var key in NotUndoable)
            project = before.Sections.TryGetValue(key, out var v) ? project with { Sections = project.Sections.Set(key, v) } : project;
        Project = project;
        _revision = revision;
        if (project.System != before.System) SystemChanged?.Invoke();
        foreach (var key in project.Sections.Keys.Union(before.Sections.Keys).Distinct())
        {
            project.Sections.TryGetValue(key, out var a);
            before.Sections.TryGetValue(key, out var b);
            if (a != b) SectionChanged?.Invoke(key);
        }
        Changed?.Invoke();
    }

    /// <summary>Replace a window-state section (<c>views</c>) without an undo step; saved with the next save.</summary>
    public void SetSection(string key, JsonTree value)
    {
        if (Section(key) == value) return;
        Project = Project with { Sections = Project.Sections.Set(key, value) };
        SectionChanged?.Invoke(key);
    }

    public void Save(string? path = null, string? engineVersion = null)
    {
        var target = path ?? Path ?? throw new InvalidOperationException("An untitled design needs a path.");
        var version = typeof(DesignDocument).Assembly.GetName().Version?.ToString(3) ?? "0.1.0";
        Project.StampMetadata("MKLight for Windows", version, engineVersion).Write(target);
        Path = target;
        _savedRevision = _revision;
    }
}
