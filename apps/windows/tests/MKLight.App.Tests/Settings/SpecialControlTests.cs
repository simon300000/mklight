using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests.Settings;

/// <summary>The dedicated controls' choice and their pure parts: row tables, the sensor limit, image simulation, automatic values.</summary>
public class SpecialControlTests
{
    static SettingsField Field(string name, string schema) => new(name, TestData.Obj(schema));

    [Fact]
    public void DedicatedControlsByName()
    {
        Assert.Equal(ControlKind.Spectrum, SettingControls.For(Field("spectrum", """{"type": ["object", "null"]}"""), "mtf").Kind);
        Assert.Equal(ControlKind.SensorLimit, SettingControls.For(Field("sensor_limit", """{"type": "array", "default": []}"""), "chief_ray_angle").Kind);
        Assert.Equal(ControlKind.ImageSource,
            SettingControls.For(Field("source", """{"type": "object", "default": {"kind": "pattern", "pattern": "grid"}}"""), "image_simulation").Kind);
        Assert.Equal(ControlKind.ImageOptics,
            SettingControls.For(Field("optics", """{"type": "object", "default": {"psf_size": 0, "field_grid": 3}, "properties": {"psf_size": {"type": "integer"}}}"""), "image_simulation").Kind);
        // stray_light's source is a tagged union, not the image source.
        Assert.Equal(ControlKind.Variants, SettingControls.For(Field("source",
            """{"type": "object", "default": {"type": "system_field"}, "oneOf": [{"title": "Field", "properties": {"type": {"const": "system_field"}}}]}"""), "stray_light").Kind);
    }

    [Fact]
    public void RowTablesFromItemsDefaultsOrTheDescription()
    {
        // ghost_irradiance.points / system_transmission.fields: keys named in the description.
        var points = SettingControls.For(Field("points", """{"type": "array", "default": [], "description": "Extra normalised field points [{\"hx\", \"hy\"}], weight 1 each."}"""), "ghost_irradiance");
        Assert.Equal(ControlKind.Table, points.Kind);
        Assert.Equal(["hx", "hy"], points.Columns!.Select(c => c.Name));
        Assert.False(points.RowsAreArrays);
        var pairs = SettingControls.For(Field("limits", """{"type": "array", "default": [[1.0, 30.0]]}"""), "x");
        Assert.True(pairs.RowsAreArrays);
        Assert.Equal(["0", "1"], pairs.Columns!.Select(c => c.Name));
        var items = SettingControls.For(Field("rows", """{"type": "array", "default": [], "items": {"properties": {"name": {"type": "string"}, "x": {"type": "number"}}}}"""), "x");
        Assert.Equal(["name", "x"], items.Columns!.Select(c => c.Name));
        // Objects with nested objects are not tables.
        Assert.Equal(ControlKind.Hidden, SettingControls.For(Field("vars", """{"type": "array", "default": [], "items": {"properties": {"c": {"type": "object"}}}}"""), "x").Kind);
        Assert.Equal(ControlKind.Hidden, SettingControls.For(Field("operands", """{"type": "array", "default": [], "description": "Operands appended to the merit."}"""), "x").Kind);
    }

    [Fact]
    public void TableRowsAndValue()
    {
        var cols = new[] { Field("hx", """{"type": "number", "default": 0}"""), Field("hy", """{"type": "number", "default": 0}""") };
        var rows = TableSetting.Rows(JsonNode.Parse("""[{"hx": 0, "hy": 1}, {"hx": 0.5}]"""), cols, false);
        Assert.Equal(2, rows.Count);
        Assert.Null(rows[1][1]);
        // A row with a number missing is not part of the setting until it is filled.
        Assert.Equal("""[{"hx":0,"hy":1}]""", TableSetting.Value(rows, cols, false).ToJsonString());
        rows[1][1] = 0.25;
        Assert.Equal("""[[0,1],[0.5,0.25]]""", TableSetting.Value(rows, cols, true).ToJsonString());
        Assert.Equal(0.5, Json.Number(TableSetting.Parse("0,5", cols[0])!.Value));
        Assert.Null(TableSetting.Parse("x", cols[0]));
        Assert.Null(TableSetting.Parse(" ", cols[0])!.Value);
        Assert.Equal(3L, TableSetting.Parse("3.7", Field("n", """{"type": "integer"}"""))!.Value!.GetValue<long>());
        Assert.True(Json.Bool(TableSetting.Parse("yes", Field("on", """{"type": "boolean"}"""))!.Value));
    }

    [Fact]
    public void TableTitles()
    {
        var f = Field("fields", """{"type": "array", "default": [], "description": "Normalised field points [{\"hx\", \"hy\"}]; empty = the system fields (with their vignetting factors)."}""");
        Assert.Equal("The system fields", TableSetting.ButtonTitle(f, new JsonArray()));
        Assert.Equal("1 row", TableSetting.ButtonTitle(f, JsonNode.Parse("""[{"hx": 0, "hy": 0}]""")));
        Assert.Equal("2 rows", TableSetting.ButtonTitle(f, JsonNode.Parse("""[{}, {}]""")));
        Assert.Equal("None", TableSetting.EmptyTitle(Field("x", """{"type": "array"}""")));
        Assert.Equal("Automatic", TableSetting.EmptyTitle(Field("x", """{"type": ["array", "null"]}""")));
    }

    [Fact]
    public void SensorLimitIsSortedUniqueAndEmptyMeansNone()
    {
        var v = SensorLimit.Value([(10, 28), (0, 20), (10, 30), (20, 32)])!;
        Assert.Equal("[[0,20],[10,28],[20,32]]", v.ToJsonString());
        Assert.Null(SensorLimit.Value([]));
        Assert.Equal("Sensor limit: 3 points", SensorLimit.Title(v));
        Assert.Equal("Sensor limit: 20.0°", SensorLimit.Title(JsonNode.Parse("[[0, 20]]")));
        Assert.Equal("Sensor limit: None", SensorLimit.Title(new JsonArray()));
        Assert.Equal((21.0, 32.0), SensorLimit.Next(SensorLimit.Rows(v)));
        Assert.Equal((0.0, 30.0), SensorLimit.Next([]));
    }

    [Fact]
    public void ImageSimulationOpticsKeepOtherKeys()
    {
        var defaults = JsonNode.Parse("""{"psf_size": 0, "field_grid": 3, "pupil": {"grid": 0, "rays": 65, "rim": true}}""");
        var o = ImageOptics.WithPupil(defaults, defaults, "grid", new Edit(48));
        Assert.Equal("""{"psf_size":0,"field_grid":3,"pupil":{"grid":48,"rays":65,"rim":true}}""", o.ToJsonString());
        Assert.Equal(65, Json.Int(ImageOptics.WithPupil(ImageOptics.WithPupil(o, defaults, "rays", new Edit(9)), defaults, "rays", Edit.Reset)["pupil"]!["rays"]));
        Assert.Equal(129, Json.Int(ImageOptics.With(o, defaults, "psf_size", new Edit(129))["psf_size"]));
        Assert.Equal("Siemens star", ImageSourcePatterns.Title("siemens_star"));
        Assert.Equal("""{"kind":"pattern","pattern":"text_chart"}""", ImageSourcePatterns.Source("text_chart").ToJsonString());
    }

    [Fact]
    public void ImageSimulationAutoValuesFromTheResult()
    {
        Assert.Equal(AutoSentinel.Null, AutoSettings.Sentinel("optics.psf_spacing_mm", "image_simulation"));
        var r = TestData.Result("image_simulation", """{"psf_size": 129, "psf_spacing_mm": 0.00155}""");
        Assert.Equal("129", AutoSettings.Used("optics.psf_size", "image_simulation", r));
        Assert.Equal("1.55 µm", AutoSettings.Used("optics.psf_spacing_mm", "image_simulation", r));
    }

    [Fact]
    public void SpectrumSummaryInTheStateLine()
    {
        var d = TestData.Obj("""{"name": "mtf", "settings_schema": {"type": "object", "properties": {"spectrum": {"type": ["object", "null"]}}}}""");
        Assert.Equal("Spectrum blackbody 5500 K", SettingsSummary.Describe(d, TestData.Obj("""{"spectrum": {"kind": "blackbody", "temperature_k": 5500}}""")));
        Assert.Equal("Spectrum custom", SettingsSummary.Describe(d, TestData.Obj("""{"spectrum": {"kind": "laser"}}""")));
        Assert.Equal("", SettingsSummary.Describe(d, TestData.Obj("""{"spectrum": null}""")));
    }
}
