using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests;

/// <summary>
/// The settings mapping over every analysis the engine lists. Runs when
/// MKLIGHT_ANALYSES_JSON names the output of <c>mklight --json analyses</c>
/// (or a JSON array of analysis descriptors); skipped otherwise.
/// </summary>
public class EngineSchemaTests
{
    static JsonArray? Analyses()
    {
        var path = Environment.GetEnvironmentVariable("MKLIGHT_ANALYSES_JSON");
        if (string.IsNullOrEmpty(path) || !File.Exists(path)) return null;
        var root = JsonNode.Parse(File.ReadAllText(path));
        return root as JsonArray ?? root?["analyses"] as JsonArray;
    }

    [Fact]
    public void EverySettingMapsAndDefaultsSummariseAsNothing()
    {
        var list = Analyses();
        Assert.SkipWhen(list is null, "Set MKLIGHT_ANALYSES_JSON to the engine's analysis list to run this test.");
        var hidden = new List<string>();
        foreach (var a in list!.OfType<JsonObject>())
        {
            var name = a["name"]!.GetValue<string>();
            foreach (var f in SettingsField.Of(a))
            {
                var c = SettingControls.For(f, name, hasConfigurations: true);
                if (c.Kind == ControlKind.Hidden) hidden.Add($"{name}.{f.Name}");
                if (c.Kind is ControlKind.Group or ControlKind.Variants)
                    foreach (var p in f.Properties.Concat(f.Variants.SelectMany(v => v.Properties)))
                        SettingControls.For(p, name, nested: true);
            }
            Assert.Equal("", SettingsSummary.Describe(a, new JsonObject()));
            if (a["default_settings"] is JsonObject defaults)
                Assert.Equal("", SettingsSummary.Describe(a, (JsonObject)defaults.DeepClone()));
        }
        TestContext.Current.SendDiagnosticMessage("Hidden: " + string.Join(", ", hidden));
        var spot = list.OfType<JsonObject>().First(a => a["name"]!.GetValue<string>() == "spot_diagram");
        Assert.Equal(ControlKind.ThroughFocus, SettingControls.For(SettingsField.Of(spot).Single(f => f.Name == "through_focus"), "spot_diagram").Kind);
        var mtf = list.OfType<JsonObject>().First(a => a["name"]!.GetValue<string>() == "mtf");
        var mtfFields = SettingsField.Of(mtf);
        Assert.Equal(ControlKind.Pupil, SettingControls.For(mtfFields.Single(f => f.Name == "pupil"), "mtf").Kind);
        Assert.Equal(ControlKind.Group, SettingControls.For(mtfFields.Single(f => f.Name == "detector"), "mtf").Kind);
        Assert.Equal(ControlKind.FieldSelection, SettingControls.For(mtfFields.Single(f => f.Name == "fields"), "mtf").Kind);
    }
}
