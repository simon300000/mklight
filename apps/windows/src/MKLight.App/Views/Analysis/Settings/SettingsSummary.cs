using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.Model;

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
            return SpectrumSetting.Read(v) is { } s ? s.Kind == SpectrumKind.Design ? null : Loc.F("Analysis_Summary_Spectrum", s.Summary) : Loc.S("Analysis_Summary_SpectrumCustom");
        if (f.Name == "sensor_limit" && analysis == "chief_ray_angle")
            return SensorLimit.Rows(v).Count switch
            {
                0 => Loc.S("Analysis_Summary_SensorLimitNone"), 1 => Loc.S("Analysis_Summary_SensorLimit_One"), var n => Loc.F("Analysis_Summary_SensorLimit_Many", n),
            };
        if (f.Name == "source" && Json.Str(Json.At(v, "kind")) is string kind)
            return Loc.F("Analysis_Summary_Source", kind switch
            {
                "pattern" => ImageSourcePatterns.Title(Json.Str(Json.At(v, "pattern")) ?? "grid").ToLowerInvariant(),
                "buffer" => Loc.S("Analysis_Summary_SourceImage"),
                _ => Path.GetFileName(Json.Str(Json.At(v, "path")) ?? ""),
            });
        if (f.Name == "pupil" && Json.Has(v, "grid"))
            return Json.Int(Json.At(v, "grid")) is int g && g != 0 ? Loc.F("Analysis_Summary_Pupil", g) : null;
        if (f.Name == "through_focus" && analysis == "spot_diagram")
            return v is null ? Loc.S("Analysis_Summary_SinglePlane") : SettingEntry.ThroughFocusSummary(v, AutoSettings.Used("through_focus", analysis, last));
        if (AutoSettings.Sentinel(f.Name, analysis) is AutoSentinel a && AutoSettings.IsAuto(v, a)) return Loc.F("Analysis_Summary_Auto", label);
        if (v is JsonArray { Count: 2 } pair && Json.Number(pair[0]) is double x && Json.Number(pair[1]) is double y)
        {
            var (before, between) = SettingEntry.PairWords(f);
            return before is not null ? Loc.F("Analysis_Summary_Point", label, Fmt.Short(x), Fmt.Short(y)) : Loc.F("Analysis_Summary_Pair", label, Fmt.Short(x), between, Fmt.Short(y));
        }
        // Field and wavelength indices are numbered from 1, as everywhere in the app.
        if (f.Name is "field" or "wavelength" or "long_wavelength" or "short_wavelength" && Json.Int(v) is int index)
            return Loc.F("Analysis_Summary_Value", label, (index + 1).ToString(CultureInfo.InvariantCulture));
        // A variant's title is engine text.
        if (f.VariantOf(v) is SettingsField.Variant variant) return Loc.F("Analysis_Summary_Value", label, EngineText.T(variant.Title).ToLowerInvariant());
        string value = Json.Str(Json.At(v, "type")) switch
        {
            "all" => Loc.S("Analysis_Summary_All"),
            "primary" => Loc.S("Analysis_Summary_Primary"),
            "index" => ((Json.Int(Json.At(v, "index")) ?? 0) + 1).ToString(CultureInfo.InvariantCulture),
            "list" => (Json.Arr(Json.At(v, "indices"))?.Count ?? 0).ToString(CultureInfo.InvariantCulture),
            _ => v switch
            {
                null => Loc.S("Analysis_Summary_Off"),
                JsonObject => Loc.S("Analysis_Summary_On"),
                JsonArray arr => arr.Count.ToString(CultureInfo.InvariantCulture),
                _ when Json.Bool(v) is bool b => Loc.S(b ? "Analysis_Summary_On" : "Analysis_Summary_Off"),
                _ when Json.Str(v) is string s => SettingEntry.Title(s),
                _ => Json.Display(v),
            },
        };
        return Loc.F("Analysis_Summary_Value", label, value);
    }
}
