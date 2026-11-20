using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests;

/// <summary>Settings schemas shaped as the engine publishes them (docs/RESULTS.md, docs/FFI.md).</summary>
static class Schemas
{
    public const string Spot = """
    { "name": "spot_diagram", "title": "Spot diagram", "category": "geometric",
      "settings_schema": { "type": "object", "properties": {
        "defocus": {"type": "number", "default": 0.0, "unit": "mm", "description": "Defocus from the image surface."},
        "fields": {"type": "object", "default": {"type": "all"}, "description": "Field selection"},
        "include_points": {"type": "boolean", "default": true},
        "pattern": {"type": "string", "default": "hexapolar", "enum": ["hexapolar", "square", "random", "dithered"]},
        "reference": {"type": "string", "default": "chief", "enum": ["chief", "centroid"]},
        "rings": {"type": "integer", "default": 6, "minimum": 1, "maximum": 512},
        "seed": {"type": "integer", "default": 1, "minimum": 0},
        "spectrum": {"type": ["object", "null"]},
        "through_focus": {"type": ["object", "null"], "default": null, "properties": {
          "max": {"type": ["number", "null"], "default": null, "unit": "mm"},
          "min": {"type": ["number", "null"], "default": null, "unit": "mm"},
          "steps": {"type": "integer", "default": 3, "minimum": 1, "maximum": 100000} } },
        "wavelengths": {"type": "object", "default": {"type": "all"}} } },
      "default_settings": {"defocus": 0.0, "fields": {"type": "all"}, "include_points": true, "pattern": "hexapolar", "reference": "chief",
                           "rings": 6, "seed": 1, "through_focus": null, "wavelengths": {"type": "all"}} }
    """;

    public const string Wave = """
    { "name": "mtf",
      "settings_schema": { "type": "object", "properties": {
        "configuration": {"type": "integer", "default": 0, "minimum": 0},
        "detector": {"type": ["object", "null"], "default": null, "properties": {
          "aperture_mm": {"type": ["array", "null"], "default": null, "unit": "mm"},
          "fill_factor": {"type": "number", "default": 1.0, "exclusiveMinimum": 0.0, "maximum": 1.0},
          "pitch_mm": {"type": "array", "default": [0.005, 0.005], "unit": "mm"} } },
        "max_frequency": {"type": ["number", "null"], "default": null, "unit": "cycles/mm", "exclusiveMinimum": 0.0},
        "method": {"type": "string", "default": "direct", "enum": ["direct", "fft", "huygens"]},
        "pupil": {"type": "object", "default": {"amplitude": "radiometric", "grid": 0, "rays": 65},
          "properties": {"grid": {"type": "integer", "default": 0, "minimum": 0, "maximum": 4096},
                         "rays": {"type": "integer", "default": 65, "minimum": 8},
                         "amplitude": {"type": "string", "default": "radiometric", "enum": ["uniform", "radiometric"]} } },
        "center": {"type": "object", "default": {"type": "centroid"}, "oneOf": [
          {"title": "Centroid", "properties": {"type": {"const": "centroid"}}},
          {"title": "Point", "properties": {"type": {"const": "point"}, "x": {"type": "number", "default": 0.0}, "y": {"type": "number", "default": 0.0}}} ] },
        "frequencies": {"type": "array", "default": [], "items": {"type": "number", "minimum": 0.0}, "unit": "cycles/mm"},
        "range": {"type": ["array", "null"], "default": null, "items": {"type": "number"}, "unit": "mm"},
        "field_center": {"type": "array", "default": [0.0, 0.0]},
        "map_bins": {"type": "array", "default": [64, 64], "items": {"type": "integer", "minimum": 1}},
        "wavelengths": {"type": "array", "default": [], "items": {"type": "integer", "minimum": 0}},
        "wavelength": {"type": ["integer", "null"], "default": null, "minimum": 0},
        "long_wavelength": {"type": ["integer", "null"], "default": null},
        "surface": {"type": "integer", "default": 1},
        "tolerances": {"type": "array", "default": []},
        "name": {"type": "string", "default": "N-BK7"},
        "catalog": {"type": ["string", "null"], "default": null} } } }
    """;

    public static JsonObject Descriptor(string json) => TestData.Obj(json);
    public static SettingsField Field(string descriptor, string name) => SettingsField.Of(Descriptor(descriptor)).Single(f => f.Name == name);
}

public class SettingControlTests
{
    static SettingControl Spot(string name, string? pattern = "hexapolar") =>
        SettingControls.For(Schemas.Field(Schemas.Spot, name), "spot_diagram", pattern: pattern);

    static SettingControl Wave(string name, bool configurations = false) =>
        SettingControls.For(Schemas.Field(Schemas.Wave, name), "mtf", configurations);

    [Fact]
    public void ScalarsGetFieldsTogglesAndPickers()
    {
        Assert.Equal(new SettingControl(ControlKind.Number), Spot("defocus"));
        Assert.Equal(new SettingControl(ControlKind.Number, Integer: true), Spot("rings"));
        Assert.Equal(ControlKind.Toggle, Spot("include_points").Kind);
        Assert.Equal(ControlKind.Picker, Spot("pattern").Kind);
        Assert.Equal(ControlKind.Text, Wave("name").Kind);
        Assert.Equal(new SettingControl(ControlKind.Text, Nullable: true), Wave("catalog"));
        Assert.Equal(new SettingControl(ControlKind.Number, Nullable: true), Wave("max_frequency"));
    }

    [Fact]
    public void SelectionsAndIndicesAreMenusOfTheSystem()
    {
        Assert.Equal(ControlKind.FieldSelection, Spot("fields").Kind);
        Assert.Equal(ControlKind.WavelengthSelection, Spot("wavelengths").Kind);
        Assert.Equal(ControlKind.WavelengthIndex, Wave("wavelength").Kind);
        Assert.Equal(ControlKind.WavelengthIndex, Wave("long_wavelength").Kind);
        Assert.Equal(ControlKind.SurfaceIndex, Wave("surface").Kind);
        Assert.Equal(new SettingControl(ControlKind.IndexList, Domain: IndexDomain.Wavelengths), Wave("wavelengths"));
    }

    [Fact]
    public void ArraysArePairsOrLists()
    {
        Assert.Equal(new SettingControl(ControlKind.Pair, Integer: true), Wave("map_bins"));
        Assert.Equal(new SettingControl(ControlKind.Pair), Wave("field_center"));
        Assert.Equal(new SettingControl(ControlKind.Pair, Nullable: true), Wave("range"));
        Assert.Equal(ControlKind.NumberList, Wave("frequencies").Kind);
        Assert.Equal(ControlKind.Hidden, Wave("tolerances").Kind);
    }

    [Fact]
    public void ObjectsAreGroupsUnionsAndDedicatedControls()
    {
        Assert.Equal(new SettingControl(ControlKind.Group, Nullable: true), Wave("detector"));
        Assert.Equal(ControlKind.Variants, Wave("center").Kind);
        Assert.Equal(ControlKind.Pupil, Wave("pupil").Kind);
        Assert.Equal(ControlKind.ThroughFocus, Spot("through_focus").Kind);
        Assert.Equal(ControlKind.Hidden, Spot("spectrum").Kind);
        // Nested, a "pupil" is an ordinary group.
        Assert.Equal(ControlKind.Group, SettingControls.For(Schemas.Field(Schemas.Wave, "pupil"), "mtf", nested: true).Kind);
        var pitch = Schemas.Field(Schemas.Wave, "detector").Properties.Single(p => p.Name == "pitch_mm");
        Assert.Equal(ControlKind.Pair, SettingControls.For(pitch, "mtf", nested: true).Kind);
    }

    [Fact]
    public void SeedAndConfigurationShowOnlyWhenTheyMatter()
    {
        Assert.Equal(ControlKind.Hidden, Spot("seed", "hexapolar").Kind);
        Assert.Equal(ControlKind.Number, Spot("seed", "random").Kind);
        Assert.Equal(ControlKind.Hidden, Wave("configuration").Kind);
        Assert.Equal(ControlKind.Number, Wave("configuration", configurations: true).Kind);
    }

    [Fact]
    public void ReadsTheSchemaOfASetting()
    {
        var f = Schemas.Field(Schemas.Wave, "max_frequency");
        Assert.True(f.Nullable);
        Assert.Equal("cycles/mm", f.Unit);
        Assert.Equal(0.0, f.Minimum);
        Assert.True(f.MinimumExclusive);
        Assert.True(f.HasDefault);
        Assert.Null(f.Default);
        var center = Schemas.Field(Schemas.Wave, "center");
        Assert.Equal(["Centroid", "Point"], center.Variants.Select(v => v.Title));
        Assert.Equal("Point", center.VariantOf(JsonNode.Parse("""{"type": "point", "x": 1}"""))!.Title);
        Assert.Null(center.VariantOf(JsonNode.Parse("""{"type": "other"}""")));
        // Properties inferred from a default object when the schema lists none.
        var inferred = new SettingsField("huygens", TestData.Obj("""{"type": "object", "default": {"grid_size": 128, "padding": 4.0}}"""));
        Assert.Equal(["grid_size", "padding"], inferred.Properties.Select(p => p.Name));
        Assert.True(inferred.Properties[0].IsInteger);
        Assert.Equal("number", inferred.Properties[1].Types.Single());
    }

    [Fact]
    public void SwitchingOnAndBetweenVariants()
    {
        var detector = Schemas.Field(Schemas.Wave, "detector");
        var on = (JsonObject)SettingControls.EnabledValue(detector);
        Assert.True(Json.Same(on, JsonNode.Parse("""{"aperture_mm": null, "fill_factor": 1.0, "pitch_mm": [0.005, 0.005]}""")));
        var center = Schemas.Field(Schemas.Wave, "center");
        var point = SettingControls.ChooseVariant(center.Variants[1], JsonNode.Parse("""{"type": "centroid"}"""));
        Assert.True(Json.Same(point, JsonNode.Parse("""{"type": "point", "x": 0.0, "y": 0.0}""")));
        var kept = SettingControls.ChooseVariant(center.Variants[1], JsonNode.Parse("""{"type": "point", "x": 2.5, "y": 1}"""));
        Assert.Equal(2.5, Json.Nf(kept["x"]));
        Assert.Equal("point", Json.Str(kept["type"]));
    }
}

public class SettingEntryTests
{
    static SettingsField Number(string schema) => new("n", TestData.Obj(schema));

    [Fact]
    public void NumbersAcceptADecimalCommaAndHoldToTheirRange()
    {
        var f = Number("""{"type": "number", "minimum": 0, "maximum": 10}""");
        Assert.Equal(1.5, Json.Nf(SettingEntry.ParseNumber("1,5", f, null)!.Value));
        Assert.Equal(10.0, Json.Nf(SettingEntry.ParseNumber("25", f, null)!.Value));
        Assert.Equal(0.0, Json.Nf(SettingEntry.ParseNumber("−3", f, null)!.Value));
        Assert.Null(SettingEntry.ParseNumber("abc", f, null));
        Assert.Null(SettingEntry.ParseNumber("1,000.5", f, null));
        Assert.Equal(Edit.Reset, SettingEntry.ParseNumber("  ", f, null));
        var integer = Number("""{"type": "integer", "minimum": 1}""");
        Assert.Equal("7", SettingEntry.ParseNumber("7.9", integer, null)!.Value!.ToJsonString());
        var exclusive = Number("""{"type": "number", "exclusiveMinimum": 0}""");
        Assert.Null(SettingEntry.ParseNumber("0", exclusive, null));
        Assert.Equal(0.5, Json.Nf(SettingEntry.ParseNumber("0.5", exclusive, null)!.Value));
    }

    [Fact]
    public void AutomaticNumbersClearToTheirSentinel()
    {
        var grid = Number("""{"type": "integer", "minimum": 0}""");
        Assert.Equal("0", SettingEntry.ParseNumber("", grid, AutoSentinel.Zero)!.Value!.ToJsonString());
        Assert.Equal("0", SettingEntry.ParseNumber("Auto (282)", grid, AutoSentinel.Zero)!.Value!.ToJsonString());
        var limit = Number("""{"type": ["number", "null"]}""");
        var cleared = SettingEntry.ParseNumber("auto", limit, AutoSentinel.Null)!;
        Assert.False(cleared.Remove);
        Assert.Null(cleared.Value);
        Assert.Equal("", SettingEntry.NumberDisplay(JsonValue.Create(0), true, AutoSentinel.Zero, false));
        Assert.Equal("128", SettingEntry.NumberDisplay(JsonValue.Create(128), true, AutoSentinel.Zero, false));
        Assert.Equal("Auto (282)", AutoSettings.Display("282"));
        Assert.Equal("Auto", AutoSettings.Display(null));
    }

    [Fact]
    public void NumbersShowCompactAtRestAndExactWithTheFocus()
    {
        var third = 1.0 / 3;
        Assert.Equal("0.333333", SettingEntry.NumberDisplay(JsonValue.Create(third), false, null, exact: false));
        Assert.Equal(third, NumberText.Parse(SettingEntry.NumberDisplay(JsonValue.Create(third), false, null, exact: true)));
        Assert.Equal("0.1", NumberText.Exact(0.1));
        Assert.Equal("-2.5", NumberText.Short(-2.5));
        Assert.Equal(double.PositiveInfinity, NumberText.Parse("∞"));
        Assert.Null(NumberText.Parse("nan"));
    }

    [Fact]
    public void ListsAndPairs()
    {
        var freqs = Schemas.Field(Schemas.Wave, "frequencies");
        var e = SettingEntry.ParseList("30, 10; 20 10", freqs, false, AutoSentinel.Empty)!;
        Assert.Equal("[10,20,30]", e.Value!.ToJsonString());
        Assert.Equal("[]", SettingEntry.ParseList("auto", freqs, false, AutoSentinel.Empty)!.Value!.ToJsonString());
        Assert.Null(SettingEntry.ParseList("10, x", freqs, false, null));
        Assert.Equal("10, 20.5", SettingEntry.ListDisplay(JsonNode.Parse("[10, 20.5]"), false, false));

        Assert.Equal("[64,32]", SettingEntry.Pair(64, 32, true, false)!.Value!.ToJsonString());
        Assert.Null(SettingEntry.Pair(1, null, false, true));
        Assert.Null(SettingEntry.Pair(null, null, false, true)!.Value);
        Assert.Equal(1.0, SettingEntry.ParsePairPart("0", true, (1, 101)));
        var bins = new SettingsField("grid", TestData.Obj("""{"type": "array", "default": [5, 5], "description": "Samples per axis, 1..101."}"""));
        Assert.Equal((1.0, 101.0), SettingEntry.PairBounds(bins));
        Assert.Equal(("Hx", "Hy"), SettingEntry.PairWords(Schemas.Field(Schemas.Wave, "field_center")));
        Assert.Equal((null, "to"), SettingEntry.PairWords(Schemas.Field(Schemas.Wave, "range")));
        Assert.Equal((null, "×"), SettingEntry.PairWords(Schemas.Field(Schemas.Wave, "map_bins")));
    }

    [Fact]
    public void LabelsAndTitles()
    {
        Assert.Equal("Max Frequency", SettingEntry.Label(Schemas.Field(Schemas.Wave, "max_frequency")));
        Assert.Equal("Pitch", SettingEntry.Label(new SettingsField("pitch_mm", TestData.Obj("""{"type": "array"}"""))));
        Assert.Equal("Custom Title", SettingEntry.Label(new SettingsField("x", TestData.Obj("""{"title": "Custom Title"}"""))));
        Assert.Equal("FFT", SettingEntry.Title(JsonValue.Create("fft")));
        Assert.Equal("Primary Chief", SettingEntry.Title(JsonValue.Create("primary_chief")));
        Assert.Equal("N-SK16", SettingEntry.Title(JsonValue.Create("N-SK16")));
    }

    [Fact]
    public void MenuTitles()
    {
        var ctx = new SettingsContext { FieldLabels = ["0°", "10°", "14°"], WavelengthsUm = [0.4861, 0.5876, 0.6563], PrimaryWavelength = 1, SurfaceLabels = ["Object", "1", "2 (stop)", "Image"] };
        Assert.Equal("1: 0°", ctx.FieldOptions[0].Title);
        Assert.Equal("0.5876 µm (primary)", ctx.WavelengthOptions[1].Title);
        Assert.Equal([1, 2, 3], ctx.SurfaceOptions.Select(o => o.Index));
        Assert.Equal("All", SettingEntry.SelectionTitle(JsonNode.Parse("""{"type": "all"}"""), ctx.FieldOptions));
        Assert.Equal("Primary", SettingEntry.SelectionTitle(JsonNode.Parse("""{"type": "primary"}"""), ctx.WavelengthOptions));
        Assert.Equal("2", SettingEntry.SelectionTitle(JsonNode.Parse("""{"type": "index", "index": 1}"""), ctx.FieldOptions));
        Assert.Equal("2 selected", SettingEntry.SelectionTitle(JsonNode.Parse("""{"type": "list", "indices": [0, 2]}"""), ctx.FieldOptions));
        Assert.Equal("Primary", SettingEntry.IndexTitle(null, ctx.WavelengthOptions, "Primary"));
        Assert.Equal("2 (stop)", SettingEntry.IndexTitle(JsonValue.Create(2), ctx.SurfaceOptions, null));
        Assert.Equal("All", SettingEntry.IndexListTitle(JsonNode.Parse("[]"), ctx.WavelengthOptions));
        Assert.Equal("All", SettingEntry.IndexListTitle(JsonNode.Parse("[0, 1, 2]"), ctx.WavelengthOptions));
        Assert.Equal("2 of 3", SettingEntry.IndexListTitle(JsonNode.Parse("[0, 2]"), ctx.WavelengthOptions));
        Assert.Equal("[0,2]", SettingEntry.Toggle(JsonNode.Parse("[2]"), 0, true).ToJsonString());
        Assert.Equal("[]", SettingEntry.Toggle(JsonNode.Parse("[2]"), 2, false).ToJsonString());
    }
}

public class SettingsSummaryTests
{
    [Fact]
    public void NamesWhatDiffersFromTheDefaults()
    {
        var d = Schemas.Descriptor(Schemas.Spot);
        Assert.Equal("", SettingsSummary.Describe(d, new JsonObject()));
        Assert.Equal("", SettingsSummary.Describe(d, TestData.Obj("""{"rings": 6, "pattern": "hexapolar", "through_focus": null}""")));
        var s = TestData.Obj("""{"rings": 8, "pattern": "square", "through_focus": {"min": null, "max": null, "steps": 3}, "fields": {"type": "index", "index": 1}}""");
        Assert.Equal("Fields 2 · Pattern Square · Rings 8 · Auto range · 3 planes", SettingsSummary.Describe(d, s));
        Assert.Equal("±0.08 mm · 5 planes", SettingsSummary.Describe(d, TestData.Obj("""{"through_focus": {"min": -0.08, "max": 0.08, "steps": 5}}""")));
    }

    [Fact]
    public void AutomaticRangesReadTheLastRun()
    {
        var d = Schemas.Descriptor(Schemas.Spot);
        var last = TestData.Result("spot_diagram", """
        { "fields": [ {"through_focus": [ {"defocus": -0.193}, {"defocus": 0}, {"defocus": 0.193} ]} ] }
        """);
        var s = TestData.Obj("""{"through_focus": {"min": null, "max": null, "steps": 3}}""");
        Assert.Equal("±0.193 mm · 3 planes", SettingsSummary.Describe(d, s, last));
        Assert.Equal("±0.193", AutoSettings.Used("through_focus", "spot_diagram", last));
    }

    [Fact]
    public void WaveSettingsSummaries()
    {
        var d = Schemas.Descriptor(Schemas.Wave);
        var s = TestData.Obj("""
        { "detector": {"pitch_mm": [0.004, 0.004]}, "pupil": {"grid": 256, "rays": 65}, "center": {"type": "point", "x": 0, "y": 0},
          "range": [-1, 1], "map_bins": [32, 32], "wavelength": 2, "method": "fft", "frequencies": [10, 20, 30] }
        """);
        var text = SettingsSummary.Describe(d, s);
        Assert.Contains("Detector on", text);
        Assert.Contains("Pupil 256", text);
        Assert.Contains("Center point", text);
        Assert.Contains("Range -1 to 1", text);
        Assert.Contains("Map Bins 32 × 32", text);
        Assert.Contains("Wavelength 3", text);
        Assert.Contains("Method FFT", text);
        Assert.Contains("Frequencies 3", text);
        Assert.Equal("", SettingsSummary.Describe(d, TestData.Obj("""{"pupil": {"grid": 0, "rays": 33}}""")));
    }

    [Fact]
    public void UsedValuesOfAutomaticWaveSettings()
    {
        var psf = TestData.Result("psf", """{"psfs": [ {"psf": {"size": 254, "spacing_mm": 0.0004}}, {"psf": {"size": 371, "spacing_mm": 0.0004}} ]}""");
        Assert.Equal("254–371", AutoSettings.Used("grid_size", "psf", psf));
        Assert.Equal("0.4 µm", AutoSettings.Used("spacing_mm", "psf", psf));
        var mtf = TestData.Result("mtf", """{"frequencies": "frequencies", "fields": [ {"pupil_sampling": [{"grid": 128}, {"grid": 192}]} ]}""", TestData.Vector("frequencies", 0, 100, 312.4));
        Assert.Equal("312", AutoSettings.Used("max_frequency", "mtf", mtf));
        Assert.Equal("128–192", AutoSettings.Used("pupil.grid", "mtf", mtf));
        Assert.Null(AutoSettings.Used("max_frequency", "psf", mtf));
        var vf = TestData.Result("mtf_vs_field", """{"pupil_sampling": [ [{"grid": 96}], [], [{"grid": 160}] ]}""");
        Assert.Equal("96–160", AutoSettings.Used("pupil.grid", "mtf_vs_field", vf));
        Assert.Equal(AutoSentinel.Zero, AutoSettings.Sentinel("pupil.grid", "mtf"));
        Assert.Null(AutoSettings.Sentinel("grid_size", "spot_diagram"));
        Assert.True(AutoSettings.IsAuto(JsonNode.Parse("[]"), AutoSentinel.Empty));
        Assert.False(AutoSettings.IsAuto(JsonValue.Create(3), AutoSentinel.Zero));
    }
}
