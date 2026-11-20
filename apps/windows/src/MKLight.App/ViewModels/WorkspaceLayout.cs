using System.Text.Json.Nodes;

namespace MKLight.App.ViewModels;

/// <summary>
/// The saved layout of a design's windows, stored as the project's <c>views</c> section
/// (<c>[{key, analysis, settings, ui}]</c>, docs/FFI.md): one entry per analysis pane plus a
/// <c>layout</c> entry for the viewport and the panel arrangement (<c>ui.arrangement</c>,
/// <see cref="PanelArrangement.ToJson"/>). The layout entry's other <c>ui</c> keys (viewport mode,
/// ray colouring, stretch) and its settings are kept as read, for the layout's own code.
/// </summary>
public sealed class WorkspaceLayout
{
    public sealed record Pane(string Key, string Analysis, JsonObject Settings, bool AutoUpdate, bool ShowsSettings);

    /// <summary>
    /// The layout viewport's saved state (<c>ui</c> of the layout entry, the Mac app's keys): the view
    /// (<c>2D</c>, <c>2D-xz</c> or <c>3D</c>), the ray colouring (<c>field</c> or <c>wavelength</c>) and the
    /// vertical stretch when one was chosen.
    /// </summary>
    public sealed record ViewState(string Mode, string RayColoring, double? Stretch)
    {
        public static ViewState Read(JsonObject ui) => new(
            Text(ui["mode"]) is ("3D" or "2D-xz") and { } m ? m : "2D",
            Text(ui["ray_coloring"]) == "wavelength" ? "wavelength" : "field",
            ui["stretch"] is JsonValue v && v.TryGetValue<double>(out var s) && double.IsFinite(s) && s >= 1 ? s : null);

        /// <summary>Writes the state into <paramref name="ui"/>, keeping its other keys.</summary>
        public void WriteTo(JsonObject ui)
        {
            ui["mode"] = Mode;
            ui["ray_coloring"] = RayColoring;
            if (Stretch is { } s) ui["stretch"] = s; else ui.Remove("stretch");
        }
    }

    public List<Pane> Panes { get; } = [];
    /// <summary>Where the panels are, in every window (null: the standard arrangement).</summary>
    public JsonObject? Arrangement { get; set; }
    /// <summary>The layout entry's <c>ui</c> without the arrangement (mode, ray_coloring, stretch, …).</summary>
    public JsonObject LayoutUi { get; set; } = [];
    /// <summary>The layout entry's settings (the engine's <c>layout</c> analysis).</summary>
    public JsonObject LayoutSettings { get; set; } = [];
    /// <summary>Whether the views held a saved layout at all (a <c>layout</c> entry).</summary>
    public bool IsSaved { get; private set; }

    public static WorkspaceLayout Read(JsonNode? views)
    {
        var l = new WorkspaceLayout();
        var keys = new HashSet<string>();
        foreach (var v in views as JsonArray ?? [])
        {
            if (v is not JsonObject o || Text(o["key"]) is not { } key) continue;
            var ui = o["ui"] as JsonObject ?? [];
            var settings = o["settings"]?.DeepClone() as JsonObject ?? [];
            if (key == "layout")
            {
                l.IsSaved = true;
                l.Arrangement = ui["arrangement"]?.DeepClone() as JsonObject;
                l.LayoutUi = (JsonObject)ui.DeepClone();
                l.LayoutUi.Remove("arrangement");
                l.LayoutSettings = settings;
            }
            else if (Text(o["analysis"]) is { } analysis && keys.Add(key))
            {
                l.Panes.Add(new Pane(key, analysis, settings, Bool(ui["auto_update"]) ?? true, Bool(ui["settings_visible"]) ?? false));
            }
        }
        return l;
    }

    public JsonArray ToViews()
    {
        var ui = (JsonObject)LayoutUi.DeepClone();
        if (Arrangement is not null) ui["arrangement"] = Arrangement.DeepClone();
        var views = new JsonArray
        {
            new JsonObject { ["key"] = "layout", ["analysis"] = "layout", ["settings"] = LayoutSettings.DeepClone(), ["ui"] = ui },
        };
        foreach (var p in Panes)
            views.Add(new JsonObject
            {
                ["key"] = p.Key, ["analysis"] = p.Analysis, ["settings"] = Views.Analysis.Results.ImageSourceImport.Persistable(p.Settings), // an imported image keeps only its path
                ["ui"] = new JsonObject { ["auto_update"] = p.AutoUpdate, ["settings_visible"] = p.ShowsSettings },
            });
        return views;
    }

    /// <summary>The saved panel arrangement, or the standard one when none is saved.</summary>
    public PanelArrangement MakeArrangement()
    {
        var keys = Panes.Select(p => p.Key).ToList();
        var known = new HashSet<PanelId>([PanelId.LensData, PanelId.Layout, .. keys.Select(PanelId.Analysis)]);
        return PanelArrangement.Decode(Arrangement, known) ?? PanelArrangement.Standard(keys);
    }

    static string? Text(JsonNode? n) => n is JsonValue v && v.TryGetValue<string>(out var s) ? s : null;
    static bool? Bool(JsonNode? n) => n is JsonValue v && v.TryGetValue<bool>(out var b) ? b : null;
}
