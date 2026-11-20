using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>The image simulation's built-in source patterns (<c>{"kind": "pattern", "pattern": id}</c>).</summary>
public static class ImageSourcePatterns
{
    public static readonly IReadOnlyList<(string Id, string Title)> All =
        [("grid", "Grid"), ("siemens_star", "Siemens star"), ("slanted_edge", "Slanted edge"), ("text_chart", "Text chart")];

    public static string Title(string id) => All.FirstOrDefault(p => p.Id == id).Title ?? id;

    public static JsonObject Source(string id) => new() { ["kind"] = "pattern", ["pattern"] = id };
}

/// <summary>
/// The image simulation's optics sampling (<c>optics</c>): one key changed,
/// or one key of its <c>pupil</c>; a reset key takes the default back. Other keys are kept.
/// </summary>
public static class ImageOptics
{
    /// <param name="current">The setting's value (its own, else the default).</param>
    /// <param name="defaults">The schema's default object.</param>
    public static JsonObject With(JsonNode? current, JsonNode? defaults, string key, Edit e)
    {
        var o = current?.DeepClone() as JsonObject ?? new JsonObject();
        o[key] = e.Remove ? defaults?[key]?.DeepClone() : e.Value?.DeepClone();
        return o;
    }

    public static JsonObject WithPupil(JsonNode? current, JsonNode? defaults, string key, Edit e)
    {
        var pupil = current?["pupil"]?.DeepClone() as JsonObject ?? defaults?["pupil"]?.DeepClone() as JsonObject ?? new JsonObject();
        pupil[key] = e.Remove ? defaults?["pupil"]?[key]?.DeepClone() : e.Value?.DeepClone();
        return With(current, defaults, "pupil", new Edit(pupil));
    }
}
