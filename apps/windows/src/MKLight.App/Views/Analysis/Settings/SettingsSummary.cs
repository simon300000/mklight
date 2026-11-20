using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>The pane header's one-line summary of the settings that differ from the defaults ("±0.193 mm · 3 planes").</summary>
public static class SettingsSummary
{
    /// <param name="descriptor">An element of <c>mk_engine_list_analyses</c>.</param>
    /// <param name="current">The pane's settings (may be partial).</param>
    /// <param name="last">The pane's last result, for the values automatic settings took.</param>
    public static string Describe(JsonObject descriptor, JsonObject current, ResultData? last = null)
    {
        var analysis = Json.Str(descriptor["name"]) ?? "";
        var defaults = Json.Obj(descriptor["default_settings"]);
        var parts = new List<string>();
        foreach (var f in SettingsField.Of(descriptor))
        {
            if (!current.TryGetPropertyValue(f.Name, out var v)) continue;
            if ((f.HasDefault && Json.Same(v, f.Default)) || (defaults is not null && defaults.TryGetPropertyValue(f.Name, out var dv) && Json.Same(v, dv)))
                continue;
            if (Part(f, v, analysis, last) is string p) parts.Add(p);
        }
        return string.Join(" · ", parts);
    }

    static string? Part(SettingsField f, JsonNode? v, string analysis, ResultData? last)
    {
        var label = SettingEntry.Label(f);
        if (f.Name == "spectrum")
            return v is null ? null : "Spectrum " + (Json.Str(Json.At(v, "kind")) is string k ? SettingEntry.Title(k).ToLowerInvariant() : "custom");
        if (f.Name == "pupil" && Json.Has(v, "grid"))
            return Json.Int(Json.At(v, "grid")) is int g && g != 0 ? "Pupil " + g.ToString(CultureInfo.InvariantCulture) : null;
        if (f.Name == "through_focus" && analysis == "spot_diagram")
            return v is null ? "Single plane" : SettingEntry.ThroughFocusSummary(v, AutoSettings.Used("through_focus", analysis, last));
        if (AutoSettings.Sentinel(f.Name, analysis) is AutoSentinel a && AutoSettings.IsAuto(v, a)) return label + " auto";
        if (v is JsonArray { Count: 2 } pair && Json.Number(pair[0]) is double x && Json.Number(pair[1]) is double y)
        {
            var (before, between) = SettingEntry.PairWords(f);
            return before is not null ? $"{label} ({Fmt.Short(x)}, {Fmt.Short(y)})" : $"{label} {Fmt.Short(x)} {between} {Fmt.Short(y)}";
        }
        // Field and wavelength indices are numbered from 1, as everywhere in the app.
        if (f.Name is "field" or "wavelength" or "long_wavelength" or "short_wavelength" && Json.Int(v) is int index)
            return $"{label} {(index + 1).ToString(CultureInfo.InvariantCulture)}";
        if (f.VariantOf(v) is SettingsField.Variant variant) return $"{label} {variant.Title.ToLowerInvariant()}";
        string value = Json.Str(Json.At(v, "type")) switch
        {
            "all" => "all",
            "primary" => "primary",
            "index" => ((Json.Int(Json.At(v, "index")) ?? 0) + 1).ToString(CultureInfo.InvariantCulture),
            "list" => (Json.Arr(Json.At(v, "indices"))?.Count ?? 0).ToString(CultureInfo.InvariantCulture),
            _ => v switch
            {
                null => "off",
                JsonObject => "on",
                JsonArray arr => arr.Count.ToString(CultureInfo.InvariantCulture),
                _ when Json.Bool(v) is bool b => b ? "on" : "off",
                _ when Json.Str(v) is string s => SettingEntry.Title(s),
                _ => Json.Display(v),
            },
        };
        return $"{label} {value}";
    }
}
