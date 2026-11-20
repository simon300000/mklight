using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests;

public class GenericDataTests
{
    const string Json = """
    { "converged": true, "count": 3, "label": "ok", "missing": null, "big": "inf",
      "nested": {"a": 0.25, "deep": {"x": 1}, "list": [1, 2]},
      "rows": [ {"surface": 1, "power": 0.01}, {"surface": 2, "power": -0.02, "note": "x"} ],
      "curve": [1, 2, 3, 4, 5, null],
      "short": [1, 2],
      "pairs": [[0, 1], [1, 2]] }
    """;

    [Fact]
    public void ScalarsComeFromTheTopLevelAndOneLevelDown()
    {
        var s = GenericData.Scalars(JsonNode.Parse(Json));
        Assert.Equal(["big", "converged", "count", "label", "missing", "nested.a"], s.Select(x => x.Key));
        Assert.Equal("∞", s[0].Value);
        Assert.Equal("true", s[1].Value);
        Assert.Equal("—", s[4].Value);
        Assert.Equal("0.25", s[5].Value);
        Assert.Equal([("value", "2.5")], GenericData.Scalars(JsonNode.Parse("2.5")));
    }

    [Fact]
    public void ArraysOfRecordsAreTablesAndNumberArraysCurves()
    {
        var t = GenericData.RecordTables(JsonNode.Parse(Json)).Single();
        Assert.Equal("rows", t.Name);
        Assert.Equal(["note", "power", "surface"], t.Columns);
        Assert.Equal(["", "0.01", "1"], t.Rows[0]);
        Assert.Equal(2, t.TotalRows);
        var c = GenericData.Curves(JsonNode.Parse(Json)).Single();
        Assert.Equal("curve", c.Name);
        Assert.Equal(6, c.Values.Length);
        Assert.True(double.IsNaN(c.Values[5]));
        Assert.Equal("Max Radius Mm", GenericData.Title("max_radius_mm"));
    }

    [Fact]
    public void BuffersArePlottedByShape()
    {
        Assert.Equal(GenericData.BufferKind.Curve, GenericData.KindOf(TestData.Vector("v", 1, 2, 3)));
        Assert.Equal(GenericData.BufferKind.Points, GenericData.KindOf(TestData.Buffer("p", [2, 2], 0, 0, 1, 1)));
        Assert.Equal(GenericData.BufferKind.Map, GenericData.KindOf(TestData.Buffer("m", [2, 4], new double[8])));
        Assert.Equal(GenericData.BufferKind.Summary, GenericData.KindOf(TestData.Buffer("t", [2, 3], new double[6])));
        Assert.Equal(GenericData.BufferKind.Summary, GenericData.KindOf(TestData.Buffer("q", [1, 1, 2, 2], new double[4])));
        var (lo, hi, missing) = GenericData.Range(TestData.Vector("r", 3, double.NaN, -1, 2));
        Assert.Equal((-1.0, 3.0, 1), (lo, hi, missing));
    }

    [Fact]
    public void SignedMapsAreDivergingAboutZeroOthersUseTheirRange()
    {
        var s = HeatScale.For(-0.2, 0.5);
        Assert.True(s.Diverging);
        Assert.Equal((-0.5, 0.5), (s.Min, s.Max));
        Assert.Equal(-0.4, s.Position(-0.2), 9);
        var p = HeatScale.For(0.85, 1);
        Assert.False(p.Diverging);
        Assert.Equal(0.0, p.Position(0.85), 9);
        Assert.Equal(1.0, p.Position(1), 9);
        Assert.True(double.IsNaN(p.Position(double.NaN)));
        var flat = HeatScale.For(2, 2);
        Assert.True(flat.Max > flat.Min);
    }

    [Fact]
    public void KeyNumbersOfOtherAnalyses()
    {
        var fo = TestData.Result("first_order", """
        { "entries": [ {"key": "efl", "label": "EFL", "value": 50.0, "unit": "mm"}, {"key": "working_f_number", "label": "F/#", "value": 4.02, "unit": ""} ] }
        """);
        Assert.Equal("EFL 50 mm  ·  F/# 4.02", KeyNumbers.Text(fo));
        Assert.Equal("Strehl 0.812", KeyNumbers.Text(TestData.Result("psf", """{"psfs": [{"strehl": 0.8123}]}""")));
        Assert.Equal("", KeyNumbers.Text(TestData.Result("mtf", "{}")));
        Assert.Equal("", KeyNumbers.Text(TestData.Result("seidel", """{"status": "not_applicable", "sums": {"s1": 1}}""")));
    }
}
