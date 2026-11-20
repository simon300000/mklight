using System.Runtime.CompilerServices;
using System.Text.Json;
using System.Text.Json.Nodes;
using MKLight.App.Support;

namespace MKLight.App.Views.Analysis.Results;

/// <summary>A result view with options a launch action can set (the choices a user would click).</summary>
interface IViewOptions
{
    void Apply(string option);
}

/// <summary>
/// Launch actions of the result views (<c>--do name[=arg]</c>):
/// <list type="bullet">
/// <item><c>image-simulation-start</c>: give every Image Simulation pane the starting settings
/// (a 10 % patch at 256 px on the axis, the Siemens star) and run it.</item>
/// <item><c>result-view=&lt;option&gt;</c>: set a view option on every result view that has it
/// (<c>log</c>, <c>full</c>, <c>split</c>, <c>blink</c>, <c>map</c>, <c>pgf</c>, <c>unit</c>, <c>chromatic</c>), for screenshots.</item>
/// <item><c>dump-results=&lt;dir&gt;</c>: write every pane's result (JSON, run settings and buffers)
/// to <c>&lt;dir&gt;/&lt;analysis&gt;.json</c>, the input of the readers' engine tests
/// (MKLIGHT_RESULTS_DIR).</item>
/// </list>
/// </summary>
static class ResultLaunchActions
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.LaunchActions["image-simulation-start"] = (w, arg) =>
        {
            foreach (var p in w.Session.Workspace.Panes.Where(p => p.Analysis == "image_simulation"))
            {
                var s = (JsonObject)p.Settings.DeepClone();
                foreach (var (k, v) in ImageSimulationStart()) s[k] = v.DeepClone();
                p.SetSettings(s);
            }
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["result-view"] = (w, arg) =>
        {
            foreach (var v in Descendants(w.RootElement).OfType<IViewOptions>()) v.Apply(arg ?? "");
            return Task.CompletedTask;
        };
        AppHooks.LaunchActions["dump-results"] = (w, arg) =>
        {
            var dir = Path.GetFullPath(string.IsNullOrEmpty(arg) ? "results" : arg);
            Directory.CreateDirectory(dir);
            foreach (var p in w.Session.Workspace.Panes)
            {
                if (p.Output is not { } o) continue;
                var buffers = new JsonArray();
                foreach (var b in o.Buffers)
                {
                    var data = new JsonArray();
                    for (var i = 0; i < b.Count; i++)
                    {
                        var v = b[i];
                        data.Add(double.IsFinite(v) ? JsonValue.Create(v) : null);
                    }
                    buffers.Add(new JsonObject { ["name"] = b.Name, ["shape"] = new JsonArray(b.Shape.Select(s => (JsonNode)s).ToArray()), ["data"] = data });
                }
                var doc = new JsonObject
                {
                    ["analysis"] = p.Analysis,
                    ["result"] = o.Json?.DeepClone(),
                    ["settings"] = o.RunInfo.Settings?.DeepClone(),
                    ["buffers"] = buffers,
                };
                File.WriteAllText(Path.Combine(dir, p.Analysis + ".json"), doc.ToJsonString(new JsonSerializerOptions { WriteIndented = false }));
            }
            return Task.CompletedTask;
        };
    }

    static IEnumerable<Microsoft.UI.Xaml.DependencyObject> Descendants(Microsoft.UI.Xaml.DependencyObject root)
    {
        var n = Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
        {
            var c = Microsoft.UI.Xaml.Media.VisualTreeHelper.GetChild(root, i);
            yield return c;
            foreach (var d in Descendants(c)) yield return d;
        }
    }

    /// <summary>
    /// The Image Simulation starting settings (the Mac app's): a patch of 10 % of the field at
    /// 256 px on the axis with the Siemens star, so the blur shows (the engine's default view has
    /// pixels larger than the spot).
    /// </summary>
    public static JsonObject ImageSimulationStart() => new()
    {
        ["size"] = new JsonArray(256, 256),
        ["optics"] = new JsonObject { ["field_extent"] = 0.1 },
        ["source"] = new JsonObject { ["kind"] = "pattern", ["pattern"] = "siemens_star" },
    };
}
