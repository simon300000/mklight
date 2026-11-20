using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests;

public class JsonTests
{
    [Fact]
    public void ReadsNfNumbersNullAndInfinityStrings()
    {
        var j = JsonNode.Parse("""{"a": 1.5, "b": null, "c": "inf", "d": "-inf", "e": "nan", "f": "text", "g": 3}""");
        Assert.Equal(1.5, Json.Nf(j!["a"]));
        Assert.True(double.IsNaN(Json.Nf(j["b"])));
        Assert.Equal(double.PositiveInfinity, Json.Nf(j["c"]));
        Assert.Equal(double.NegativeInfinity, Json.Nf(j["d"]));
        Assert.True(double.IsNaN(Json.Nf(j["e"])));
        Assert.Null(Json.Number(j["f"]));
        Assert.True(double.IsNaN(Json.Nf(j["missing"])));
        Assert.Equal(3, Json.Int(j["g"]));
        Assert.Null(Json.Int(j["a"]));
    }

    [Fact]
    public void ComparesNumbersAsNumbersAndKeysInAnyOrder()
    {
        Assert.True(Json.Same(JsonNode.Parse("""{"a": 3, "b": [1, 2.0]}"""), JsonNode.Parse("""{"b": [1.0, 2], "a": 3.0}""")));
        Assert.False(Json.Same(JsonNode.Parse("""{"a": 3}"""), JsonNode.Parse("""{"a": 3, "b": null}""")));
        Assert.True(Json.Same(null, null));
        Assert.False(Json.Same(JsonNode.Parse("0"), null));
    }

    [Fact]
    public void ReadsNumbersInlineOrFromANamedBuffer()
    {
        var d = TestData.Result("mtf", """{"inline": [1, 2], "named": "freq"}""", TestData.Vector("freq", 0, 10, 20));
        Assert.Equal([1.0, 2.0], Json.Numbers(d, d.Json!["inline"])!);
        Assert.Equal([0.0, 10, 20], Json.Numbers(d, d.Json!["named"])!);
        Assert.Null(Json.Numbers(d, d.Json!["missing"]));
    }
}

public class FmtTests
{
    [Fact]
    public void ShortUsesSixSignificantDigitsAndScientificAtTheEnds()
    {
        Assert.Equal("0", Fmt.Short(0));
        Assert.Equal("0.5", Fmt.Short(0.5));
        Assert.Equal("123.457", Fmt.Short(123.456789));
        Assert.Equal("1.000e-05", Fmt.Short(1e-5));
        Assert.Equal("1.235e+06", Fmt.Short(1234567));
        Assert.Equal("—", Fmt.Short(double.NaN));
        Assert.Equal("−∞", Fmt.Short(double.NegativeInfinity));
    }

    [Fact]
    public void LengthsBelowATenthOfAMillimetreAreInMicrometres()
    {
        Assert.Equal("25.92 µm", Fmt.Length(0.02592));
        Assert.Equal("0.1234 mm", Fmt.Length(0.1234));
        Assert.Equal("0.00 µm", Fmt.Length(-0.000001));
    }

    [Fact]
    public void SignedNeverShowsMinusZero()
    {
        Assert.Equal("+0.0123", Fmt.Signed(0.0123));
        Assert.Equal("-1.2000", Fmt.Signed(-1.2));
        Assert.Equal(" 0.0000", Fmt.Signed(-0.00001));
    }

    [Fact]
    public void TheNumberFormatIgnoresTheSystemRegion()
    {
        var saved = CultureInfo.CurrentCulture;
        try
        {
            foreach (var name in new[] { "de-DE", "zh-CN", "fr-FR" })
            {
                CultureInfo.CurrentCulture = new CultureInfo(name);
                Assert.Equal("1234.5", Fmt.Short(1234.5));
                Assert.Equal("12.3400 mm", Fmt.Length(12.34));
                Assert.Equal("0.5876 µm", Fmt.Wavelength(0.5876));
                Assert.Equal(1.5, NumberText.Parse("1,5"));
                Assert.Equal(1234.5, NumberText.Parse("1234.5"));
                Assert.Equal("0.1", NumberText.Exact(0.1));
            }
        }
        finally
        {
            CultureInfo.CurrentCulture = saved;
        }
    }

    [Fact]
    public void FieldTitlesUseTheSystemsLabelElseTheValue()
    {
        var d = TestData.Result("x", "{}", [0.55], ["0°", "14°"]);
        Assert.Equal("Field 2 · 14°", Fmt.FieldTitle(d, 1, null));
        Assert.Equal("Field 3 · (3, 4)", Fmt.FieldTitle(d, 2, JsonNode.Parse("[3, 4]")));
        Assert.Equal("Field 4 · 2", Fmt.FieldTitle(d, 3, JsonNode.Parse("[0, 2]")));
    }
}
