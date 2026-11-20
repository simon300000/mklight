using System.Runtime.CompilerServices;
using MKLight.App.ViewModels;
using MKLight.Model;

namespace MKLight.App.Views.Panels;

/// <summary>
/// The panes and the panel arrangement in the project's <c>views</c> section
/// (docs/APP.md "Layout persistence"): written without an undo step whenever
/// they change, read back when the project opens.
/// </summary>
public static class PanelPersistence
{
    static readonly ConditionalWeakTable<DesignSession, PanelHost> Hosts = [];

    internal static void Register(DesignSession session, PanelHost host) => Hosts.AddOrUpdate(session, host);

    /// <summary>
    /// The session's start: the panes saved in the project's <c>views</c> (analysis, settings, automatic
    /// update, settings shown) and the arrangement of every window. Returns false when the project saved
    /// no layout (a new design or an import): the session then adds its default panes.
    /// </summary>
    public static bool RestorePanes(DesignSession session)
    {
        var host = Hosts.TryGetValue(session, out var h) ? h : null;
        var layout = WorkspaceLayout.Read(session.Document.Section("views").Node);
        if (!layout.IsSaved)
        {
            host?.MarkReady();
            return false;
        }
        var ws = session.Workspace;
        foreach (var p in layout.Panes)
        {
            if (ws.Panes.Any(x => x.Id == p.Key)) continue;
            var pane = new AnalysisPaneModel(p.Analysis, p.Settings, p.AutoUpdate, id: p.Key) { ShowsSettings = p.ShowsSettings, Workspace = ws };
            ws.Panes.Add(pane);
        }
        ws.FocusedPaneId = ws.Panes.FirstOrDefault()?.Id;
        host?.ApplySaved(layout);
        // Automatic panes run now; the others wait for Run.
        foreach (var pane in ws.Panes.Where(p => p.AutoUpdate)) ws.Run(pane, force: false);
        return true;
    }

    /// <summary>Writes the panes and the arrangement into <c>views</c>, keeping the layout entry's other keys.</summary>
    internal static void Write(DesignSession session, PanelArrangement arrangement)
    {
        var doc = session.Document;
        var layout = WorkspaceLayout.Read(doc.Section("views").Node);
        layout.Panes.Clear();
        foreach (var p in session.Workspace.Panes)
            layout.Panes.Add(new WorkspaceLayout.Pane(p.Id, p.Analysis, (System.Text.Json.Nodes.JsonObject)p.Settings.DeepClone(), p.AutoUpdate, p.ShowsSettings));
        layout.Arrangement = arrangement.ToJson();
        doc.SetSection("views", new JsonTree(layout.ToViews()));
    }
}
