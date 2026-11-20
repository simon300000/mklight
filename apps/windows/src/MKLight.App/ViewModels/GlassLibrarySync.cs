using System.Runtime.CompilerServices;
using MKLight.Engine;

namespace MKLight.App.ViewModels;

/// <summary>
/// Gives an engine session the library's catalog files and availability overrides (the Mac
/// app's <c>Workspace.syncLibrary</c>). Unchanged files are not read again (the engine compares
/// modification time and size), and overrides are sent only when they differ from the ones the
/// session has. Free of WinUI types.
/// </summary>
public static class GlassLibrarySync
{
    sealed class State
    {
        public int Revision = -1;
        public IReadOnlyList<GlassOverride>? Overrides;
        public IReadOnlyList<string> Warnings = [];
    }

    static readonly ConditionalWeakTable<IEngineClient, State> States = new();

    /// <summary>
    /// Syncs <paramref name="engine"/> with <paramref name="library"/> when the library changed
    /// since the session's last sync (so a catalog file changed on disk is read again only then,
    /// and Reload can report what changed). Returns whether the session's glass library changed
    /// (then its snapshots must be taken again).
    /// </summary>
    public static async Task<bool> SyncAsync(IEngineClient engine, GlassLibrary library)
    {
        var state = States.GetOrCreateValue(engine);
        if (state.Revision == library.Revision) return false;
        var revision = library.Revision;
        var overrides = library.Overrides.ToArray();
        var warnings = new List<string>();
        var sync = await engine.SetLibraryCatalogsAsync(library.CatalogFiles);
        state.Revision = revision; // only a finished sync counts, so a failed one is tried again
        var changed = sync.Changed;
        if (state.Overrides is null || !state.Overrides.SequenceEqual(overrides))
        {
            try
            {
                await engine.SetGlassOverridesAsync(overrides);
                state.Overrides = overrides;
                changed = true;
            }
            catch (EngineException e)
            {
                warnings.Add($"Glass status settings: {e.Message}");
            }
        }
        warnings.InsertRange(0, sync.Errors.Select(e =>
            $"{(string.IsNullOrEmpty(e.Path) ? "Glass library" : Path.GetFileName(e.Path))}: {e.Message}"));
        state.Warnings = warnings;
        return changed;
    }

    /// <summary>Problems of the last sync of <paramref name="engine"/> (unreadable files, duplicate names).</summary>
    public static IReadOnlyList<string> Warnings(IEngineClient engine) =>
        States.TryGetValue(engine, out var s) ? s.Warnings : [];
}
