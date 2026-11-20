using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests.Settings;

/// <summary>The run-local spectrum setting in the engine's format (docs/RESULTS.md, the analyses' <c>spectrum</c> schema).</summary>
public class SpectrumSettingTests
{
    static SpectrumSetting Read(string json) => SpectrumSetting.Read(JsonNode.Parse(json))!;

    [Fact]
    public void AbsentIsTheDesignAndWritesNothing()
    {
        Assert.Equal(SpectrumKind.Design, SpectrumSetting.Read(null)!.Kind);
        Assert.Null(SpectrumSetting.Design.ToJson());
    }

    [Fact]
    public void ReadsEveryKind()
    {
        Assert.Equal(SpectrumKind.Photopic, Read("""{"kind": "photopic"}""").Kind);
        Assert.Equal(SpectralSampling.Default, Read("""{"kind": "scotopic"}""").Sampling);
        var b = Read("""{"kind": "blackbody", "temperature_k": 3200, "sampling": {"samples": 5, "range_um": [0.4, 0.7], "spacing": "uniform"}}""");
        Assert.Equal(3200, b.TemperatureK);
        Assert.Equal(new SpectralSampling(5, 0.4, 0.7, SamplingSpacing.Uniform), b.Sampling);
        var e = Read("""{"kind": "explicit", "wavelengths": [{"um": 0.55, "weight": 2}, {"um": 0.65}]}""");
        Assert.Equal([new(0.55, 2), new(0.65, 1)], e.List);
        var s = Read("""{"kind": "sensor", "table": [[0.4, 0.1], [0.7, 0.9]]}""");
        Assert.Equal([new(0.4, 0.1), new(0.7, 0.9)], s.List);
    }

    [Theory]
    [InlineData("""{"kind": "rainbow"}""")]
    [InlineData("""{"kind": "photopic", "colour": 1}""")]
    [InlineData("""{"kind": "photopic", "sampling": {"samples": 2.5}}""")]
    [InlineData("""{"kind": "photopic", "sampling": {"range_um": [0.4]}}""")]
    [InlineData("""{"kind": "blackbody"}""")]
    [InlineData("""{"kind": "explicit", "wavelengths": [{"weight": 1}]}""")]
    [InlineData("""{"kind": "sensor", "table": [[0.4, 0.1, 3]]}""")]
    [InlineData("""[1, 2]""")]
    public void RejectsWhatTheEngineRejects(string json) => Assert.Null(SpectrumSetting.Read(JsonNode.Parse(json)));

    [Fact]
    public void WritesOnlyNonDefaultSampling()
    {
        Assert.Equal("""{"kind":"photopic"}""", new SpectrumSetting(SpectrumKind.Photopic, SpectralSampling.Default).ToJson()!.ToJsonString());
        var j = new SpectrumSetting(SpectrumKind.Blackbody, SpectralSampling.Default with { Samples = 15 }, 5500).ToJson()!;
        Assert.Equal(5500, Json.Number(j["temperature_k"]));
        Assert.Equal("""{"samples":15}""", j["sampling"]!.ToJsonString());
        foreach (var json in new[]
        {
            """{"kind": "sensor", "table": [[0.4, 0.1], [0.7, 0.9]], "sampling": {"range_um": [0.4, 0.7]}}""",
            """{"kind": "explicit", "wavelengths": [{"um": 0.55, "weight": 2}]}""",
            """{"kind": "scotopic", "sampling": {"spacing": "uniform"}}""",
        })
            Assert.True(Json.Same(JsonNode.Parse(json), Read(json).ToJson()), json);
    }

    [Fact]
    public void IssuesBeforeARun()
    {
        Assert.Empty(new SpectrumSetting(SpectrumKind.Photopic, SpectralSampling.Default).Issues);
        Assert.Contains("Samples must be between 1 and 4096.", new SpectrumSetting(SpectrumKind.Photopic, SpectralSampling.Default with { Samples = 0 }).Issues);
        Assert.Contains("The range must lie inside the scotopic table (0.380–0.780 µm).",
            new SpectrumSetting(SpectrumKind.Scotopic, SpectralSampling.Default with { UpperUm = 0.8 }).Issues);
        Assert.Contains("The temperature must be positive.", new SpectrumSetting(SpectrumKind.Blackbody, SpectralSampling.Default, -1).Issues);
        Assert.Contains("Add at least one wavelength.", new SpectrumSetting(SpectrumKind.Explicit, Samples: []).Issues);
        Assert.Contains("Wavelengths must increase.", new SpectrumSetting(SpectrumKind.Sensor, SpectralSampling.Default, Samples: [new(0.7, 1), new(0.4, 1)]).Issues);
        Assert.Contains("The range must lie inside the response table (0.400–0.700 µm).",
            new SpectrumSetting(SpectrumKind.Sensor, SpectralSampling.Default, Samples: [new(0.4, 1), new(0.7, 1)]).Issues);
    }

    [Fact]
    public void Summaries()
    {
        Assert.Equal("photopic", Read("""{"kind": "photopic"}""").Summary);
        Assert.Equal("blackbody 5500 K ×15", Read("""{"kind": "blackbody", "temperature_k": 5500, "sampling": {"samples": 15}}""").Summary);
        Assert.Equal("blackbody 3200.5 K", Read("""{"kind": "blackbody", "temperature_k": 3200.5}""").Summary);
        Assert.Equal("0.55 µm", Read("""{"kind": "explicit", "wavelengths": [{"um": 0.55}]}""").Summary);
        Assert.Equal("2 wavelengths", Read("""{"kind": "explicit", "wavelengths": [{"um": 0.55}, {"um": 0.6}]}""").Summary);
        Assert.Equal("sensor (2 rows)", Read("""{"kind": "sensor", "table": [[0.4, 0.1], [0.7, 0.9]]}""").Summary);
    }

    [Fact]
    public void ClampsIntoTheCieTables()
    {
        Assert.Equal((0.38, 0.78), (SpectralSampling.Default with { LowerUm = 0.3, UpperUm = 0.9 }).Clamped(0.38, 0.78) is var s ? (s.LowerUm, s.UpperUm) : default);
        // Nothing left inside: the whole table.
        var t = (SpectralSampling.Default with { LowerUm = 0.9, UpperUm = 1.0 }).Clamped(0.38, 0.78);
        Assert.Equal((0.38, 0.78), (t.LowerUm, t.UpperUm));
    }

    [Fact]
    public void PrimaryIsNearestTheWeightedMean()
    {
        Assert.Equal(1, SpectrumSetting.PrimaryIndex([new(0.45, 1), new(0.55, 1), new(0.65, 1)]));
        Assert.Equal(2, SpectrumSetting.PrimaryIndex([new(0.45, 0), new(0.55, 1), new(0.65, 3)]));
        Assert.Null(SpectrumSetting.PrimaryIndex([]));
    }

    [Fact]
    public void LocalPreviewSamplesTheCurveLikeTheEngine()
    {
        var equal = new SpectrumSetting(SpectrumKind.Photopic, SpectralSampling.Default).PreviewSamples()!;
        Assert.Equal(9, equal.Count);
        Assert.All(equal, w => Assert.Equal(equal[0].Weight, w.Weight, 6)); // equal energy: equal weights
        Assert.True(equal.Zip(equal.Skip(1)).All(p => p.First.Um < p.Second.Um));
        var uniform = new SpectrumSetting(SpectrumKind.Photopic, SpectralSampling.Default with { Spacing = SamplingSpacing.Uniform }).PreviewSamples()!;
        Assert.Equal(0.38, uniform[0].Um, 9);
        Assert.Equal(0.78, uniform[^1].Um, 9);
        Assert.Equal(4, uniform.Select((w, i) => (w, i)).MaxBy(p => p.w.Weight).i); // peak near 0.56 µm
        var curve = new SpectrumSetting(SpectrumKind.Blackbody, SpectralSampling.Default, 5500).PreviewCurve()!;
        Assert.Equal(1, curve.Max(p => p.Weight), 9);
        Assert.Null(SpectrumSetting.Design.PreviewSamples());
    }

    [Fact]
    public void ReadsTheEngineResolution()
    {
        // system_summary result (docs/RESULTS.md): wavelengths_um and wavelength_weights.
        var j = JsonNode.Parse("""{"wavelengths_um": [0.45, 0.55, 0.65], "wavelength_weights": [0.5, 1, 0.25], "primary_wavelength": 1}""");
        Assert.Equal([new(0.45, 0.5), new(0.55, 1), new(0.65, 0.25)], SpectrumSetting.ResolvedSamples(j));
        Assert.Equal([new(0.5, 1)], SpectrumSetting.ResolvedSamples(JsonNode.Parse("""{"wavelengths_um": [0.5]}""")));
        Assert.Null(SpectrumSetting.ResolvedSamples(JsonNode.Parse("""{"wavelengths_um": [0.5], "wavelength_weights": [1, 2]}""")));
    }
}

public class SpectrumCsvTests
{
    [Fact]
    public void ReadsNanometresWithAHeaderAndListsBadRows()
    {
        var r = SpectrumCsv.Parse("wavelength (nm),response\n400,0.1\n# a comment\n\n450,0.5\noops,1\n500\n550,-1\n0,1\n500,0.9\n450,0.6\n");
        Assert.True(r.Nanometres);
        Assert.Equal("wavelength (nm),response", r.Header);
        Assert.Equal([new(0.4, 0.1), new(0.45, 0.5), new(0.5, 0.9)], r.Table);
        Assert.Equal(
            [(6, "not a number"), (7, "needs two columns"), (8, "response must not be negative"), (9, "wavelength must be positive"), (11, "repeats a wavelength")],
            r.BadRows.Select(b => (b.Line, b.Reason)));
        Assert.Equal((0.4, 0.5), r.RangeUm);
        Assert.True(r.IsUsable);
    }

    [Fact]
    public void MicrometresSemicolonsTabsAndDecimalCommas()
    {
        var a = SpectrumCsv.Parse("0,40;0,2\r\n0,70;0,8");
        Assert.False(a.Nanometres);
        Assert.Null(a.Header);
        Assert.Equal([new(0.4, 0.2), new(0.7, 0.8)], a.Table);
        var b = SpectrumCsv.Parse("﻿\"λ µm\"\t\"QE\"\n0.5\t1\n0.4\t0.5");
        Assert.False(b.Nanometres);
        Assert.Equal([new(0.4, 0.5), new(0.5, 1)], b.Table);
        // A header naming nm wins over magnitude.
        Assert.True(SpectrumCsv.Parse("nm,QE\n0.4,1\n0.5,1").Nanometres);
    }

    [Fact]
    public void TooFewRowsIsNotUsable()
    {
        Assert.False(SpectrumCsv.Parse("0.5,1").IsUsable);
        Assert.False(SpectrumCsv.Unreadable("x.csv").IsUsable);
        Assert.Equal("cannot be read", SpectrumCsv.Unreadable("x.csv").BadRows[0].Reason);
    }
}
