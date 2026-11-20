using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests.Results;

/// <summary>
/// The readers over real engine output. Runs when MKLIGHT_RESULTS_DIR names a folder of
/// results written by the app's <c>--do dump-results=&lt;dir&gt;</c> launch action (one
/// <c>&lt;analysis&gt;.json</c> per pane: result, run settings and buffers); skipped otherwise.
/// Generated output is never committed.
/// </summary>
public class EngineResultTests
{
    static Dictionary<string, (ResultData Data, JsonNode? Settings)>? Results()
    {
        var dir = Environment.GetEnvironmentVariable("MKLIGHT_RESULTS_DIR");
        if (string.IsNullOrEmpty(dir) || !Directory.Exists(dir)) return null;
        var o = new Dictionary<string, (ResultData, JsonNode?)>();
        foreach (var path in Directory.EnumerateFiles(dir, "*.json", SearchOption.AllDirectories))
        {
            if (JsonNode.Parse(File.ReadAllText(path)) is not JsonObject doc || doc["analysis"]?.GetValue<string>() is not { } a) continue;
            var buffers = (doc["buffers"] as JsonArray ?? []).OfType<JsonObject>().Select(b => (IResultArray)new FakeArray(
                b["name"]!.GetValue<string>(),
                b["shape"]!.AsArray().Select(s => s!.GetValue<int>()).ToArray(),
                b["data"]!.AsArray().Select(v => v is null ? double.NaN : v.GetValue<double>()).ToArray())).ToList();
            var um = (doc["result"]?["system"]?["wavelengths_um"] as JsonArray)?.Select(v => v!.GetValue<double>()).ToList() ?? [0.4861, 0.5876, 0.6563];
            o[a + "|" + Path.GetFileName(Path.GetDirectoryName(path))] = (new ResultData
            {
                Analysis = a, Json = doc["result"], Buffers = buffers, WavelengthsUm = um, FieldLabels = ["0°", "10°", "14°"],
            }, doc["settings"]);
        }
        return o;
    }

    [Fact]
    public void EveryDedicatedReaderReadsTheEnginesOutput()
    {
        var all = Results();
        Assert.SkipWhen(all is null, "Set MKLIGHT_RESULTS_DIR to a folder written by --do dump-results to run this test.");
        var read = new List<string>();
        foreach (var (key, (d, settings)) in all!)
        {
            object? r = d.Analysis switch
            {
                "first_order" => FirstOrderData.Read(d.Json),
                "seidel" => SeidelData.Read(d.Json),
                "psf" => PsfData.Read(d),
                "wavefront_map" => WavefrontData.Read(d),
                "pupil_aberration" or "relative_illumination_map" or "chief_ray_angle" => FieldMapData.Read(d),
                "glass_map" => GlassMapData.Read(d),
                "glass_info" => GlassInfoData.Read(d),
                "image_simulation" => ImageSimulationData.Read(d, settings),
                "grid_distortion" => GridDistortionData.Read(d),
                "footprint" => FootprintData.Read(d),
                "ghost_paths" => GhostPathsData.Read(d),
                "ghost_irradiance" => SensorMapsData.GhostIrradiance(d),
                "stray_light" => SensorMapsData.StrayLight(d),
                "polarization_pupil_map" => PolarizationData.Read(d),
                "system_transmission" => TransmissionData.Read(d),
                _ => "skip",
            };
            if (r is "skip") continue;
            Assert.True(r is not null, $"{key}: the reader did not read the engine's output");
            read.Add(key);
            Check(d, r!, settings);
            // Every dedicated analysis has key numbers for the footer.
            if (d.Analysis != "stray_light") Assert.True(KeyNumbers.Of(d).Count > 0, $"{key}: no key numbers");
        }
        TestContext.Current.SendDiagnosticMessage("Read: " + string.Join(", ", read.Order()));
    }

    static void Check(ResultData d, object r, JsonNode? settings)
    {
        switch (r)
        {
            case FirstOrderData f:
                Assert.Contains(f.Entries, e => e.Key == "efl" && double.IsFinite(e.Value));
                break;
            case SeidelData s when s.NotApplicable is null:
                Assert.Equal(s.Surfaces.Count, s.Values["s1"].Length);
                Assert.Equal("Σ", s.Surfaces[^1]);
                break;
            case PsfData p:
                foreach (var it in p.Items)
                {
                    Assert.True(it.Peak > 0);
                    Assert.True(it.Grid.X0 < 0 && it.Grid.X1 > 0, "the grid spans its centre");
                    var focus = PsfData.Focus(it.Grid);
                    Assert.True(focus.Nx <= it.Grid.Nx && focus.Range()!.Value.Hi == it.Peak, "the core keeps the peak");
                }
                break;
            case WavefrontData w:
                foreach (var m in w.Maps)
                {
                    Assert.True(m.Grid.Range() is not null);
                    Assert.True(m.Zernike.Length > 0);
                    Assert.All(m.Rim.SelectMany(l => l), q => Assert.True(Math.Abs(q.U) <= 1.0001 && Math.Abs(q.V) <= 1.0001));
                }
                break;
            case FieldMapData fm:
                Assert.Equal(fm.Nx * fm.Ny * fm.Columns.Count, fm.Values.Length);
                if (fm.Layout == FieldMapData.Layouts.Map)
                {
                    var (x0, x1) = fm.FieldRangeX;
                    Assert.True(x0 < 0 && x1 > 0);
                }
                if (d.Analysis == "relative_illumination_map" && fm.ColumnIndex("relative_illumination") is int ri)
                    Assert.Equal(1, fm.Value(fm.Count / 2, ri), 9); // on axis
                if (d.Analysis == "chief_ray_angle")
                    Assert.Equal(FieldMapData.SensorLimit(settings?["sensor_limit"]).Length, fm.LimitTable(settings).Length);
                break;
            case GlassMapData g:
                Assert.Contains(g.Glasses, x => x.InDesign && x.Surfaces.Count > 0);
                Assert.All(g.Glasses.Where(x => x.Kind == "catalog"), x => Assert.True(x.Nd > 1 && x.Vd > 10));
                break;
            case ImageSimulationData im:
                Assert.True(im.Image.IsValid);
                Assert.NotEmpty(im.Nodes);
                if (settings is not null) Assert.NotNull(im.SourceImage());
                break;
            case SensorMapsData sm:
                Assert.All(sm.Maps, m => Assert.True(m.Grid.X1 > m.Grid.X0));
                break;
            case PolarizationData pol:
                Assert.NotNull(pol.Map("Intensity"));
                Assert.Equal(pol.Pupil.Length, pol.Map("Intensity")!.Values.Count(double.IsFinite));
                break;
        }
    }
}
