using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Settings;

namespace MKLight.App.Tests.Settings;

/// <summary>Field, wavelength and surface selections: their two encodings, the check-mark menu's toggles and titles, the text fallback.</summary>
public class IndexSelectionTests
{
    static readonly SettingsContext Ctx = new()
    {
        FieldLabels = ["0°", "7°", "10°", "14°", "20°"],
        WavelengthsUm = [0.4861, 0.5876, 0.6563],
        PrimaryWavelength = 1,
        SurfaceLabels = ["Object", "1", "2 (stop)", "3", "Image"],
    };

    static IndexSelection Tagged(string json) => IndexSelection.Read(JsonNode.Parse(json), SelectionEncoding.Tagged);
    static IndexSelection List(string json) => IndexSelection.Read(JsonNode.Parse(json), SelectionEncoding.List);

    [Fact]
    public void ReadsBothEncodings()
    {
        Assert.Equal(SelectionKind.All, Tagged("""{"type": "all"}""").Kind);
        Assert.Equal(SelectionKind.All, IndexSelection.Read(null, SelectionEncoding.Tagged).Kind);
        Assert.Equal(SelectionKind.Primary, Tagged("""{"type": "primary"}""").Kind);
        Assert.Equal([2], Tagged("""{"type": "index", "index": 2}""").Indices);
        Assert.Equal([0, 3], Tagged("""{"type": "list", "indices": [3, 0, 3]}""").Indices);
        Assert.Equal(SelectionKind.All, Tagged("""{"type": "list", "indices": []}""").Kind);
        // Empty or unset: all.
        Assert.Equal(SelectionKind.All, List("[]").Kind);
        Assert.Equal(SelectionKind.All, IndexSelection.Read(null, SelectionEncoding.List).Kind);
        Assert.Equal([1, 2], List("[2, 1]").Indices);
    }

    [Fact]
    public void WritesBothEncodings()
    {
        Assert.Equal("""{"type":"all"}""", IndexSelection.All.Write(SelectionEncoding.Tagged).ToJsonString());
        Assert.Equal("""{"type":"primary"}""", IndexSelection.Primary.Write(SelectionEncoding.Tagged).ToJsonString());
        Assert.Equal("""{"type":"index","index":2}""", IndexSelection.Of([2]).Write(SelectionEncoding.Tagged).ToJsonString());
        Assert.Equal("""{"type":"list","indices":[0,2]}""", IndexSelection.Of([2, 0]).Write(SelectionEncoding.Tagged).ToJsonString());
        Assert.Equal("[]", IndexSelection.All.Write(SelectionEncoding.List).ToJsonString());
        Assert.Equal("[0,2]", IndexSelection.Of([2, 0]).Write(SelectionEncoding.List).ToJsonString());
        // Every option checked stays an explicit list: "all" may add axis fields, and an
        // empty ghost field list means no system fields when points are given.
        Assert.Equal("[0,1,2]", IndexSelection.Of([0, 1, 2]).Write(SelectionEncoding.List).ToJsonString());
    }

    [Fact]
    public void TogglesKeepAllAndPrimaryExclusive()
    {
        // From All or Primary an option is chosen alone.
        Assert.Equal([3], IndexSelection.All.Toggle(3).Indices);
        Assert.Equal([0], IndexSelection.Primary.Toggle(0).Indices);
        // Options add and remove; the list stays sorted.
        var s = IndexSelection.Of([2]).Toggle(0);
        Assert.Equal([0, 2], s.Indices);
        Assert.Equal([2], s.Toggle(0).Indices);
        // Unchecking the last option is all (none checked: all).
        Assert.Equal(SelectionKind.All, IndexSelection.Of([2]).Toggle(2).Kind);
        // Check marks: none under All or Primary.
        Assert.False(IndexSelection.All.IsChecked(0));
        Assert.False(IndexSelection.Primary.IsChecked(1));
        Assert.True(s.IsChecked(2));
        Assert.False(s.IsChecked(1));
    }

    [Fact]
    public void Titles()
    {
        var fields = Ctx.FieldOptions;
        var waves = Ctx.WavelengthOptions;
        Assert.Equal("All", IndexSelection.All.Title(fields));
        Assert.Equal("Primary", IndexSelection.Primary.Title(waves));
        Assert.Equal("2", IndexSelection.Of([1]).Title(fields));
        Assert.Equal("1, 3", IndexSelection.Of([0, 2]).Title(fields));
        Assert.Equal("1, 3, 4", IndexSelection.Of([0, 2, 3]).Title(fields));
        Assert.Equal("4 of 5", IndexSelection.Of([0, 1, 2, 3]).Title(fields));
        Assert.Equal("All", IndexSelection.Of([0, 1, 2, 3, 4]).Title(fields));
        Assert.Equal("0.5876 µm", IndexSelection.Of([1]).Title(waves));
        Assert.Equal("2 of 3", IndexSelection.Of([0, 2]).Title(waves));
        Assert.Equal("2 (stop), 3", IndexSelection.Of([2, 3]).Title(Ctx.SurfaceOptions));
        Assert.Equal("1, 7", IndexSelection.Of([0, 6]).Title([]));
        Assert.Equal("4 selected", IndexSelection.Of([0, 1, 2, 6]).Title([]));
        Assert.Equal("1, 3 of 5", IndexSelection.Of([0, 2]).AccessibleValue(fields));
        Assert.Equal("All 5", IndexSelection.All.AccessibleValue(fields));
        Assert.Equal("Primary", IndexSelection.Primary.AccessibleValue(waves));
    }

    [Fact]
    public void TextFallback()
    {
        Assert.Equal("1, 3", IndexSelection.Of([0, 2]).Text());
        Assert.Equal("", IndexSelection.All.Text());
        Assert.Equal("2, 3", IndexSelection.Of([2, 3]).Text(offset: 0));
        Assert.Equal([0, 2], IndexSelection.Parse(" 3, 1 1", allowsPrimary: false)!.Indices);
        Assert.Equal(SelectionKind.All, IndexSelection.Parse("", false)!.Kind);
        Assert.Equal(SelectionKind.All, IndexSelection.Parse("All", false)!.Kind);
        Assert.Equal(SelectionKind.Primary, IndexSelection.Parse("primary", true)!.Kind);
        Assert.Null(IndexSelection.Parse("primary", false));
        Assert.Null(IndexSelection.Parse("0", false));
        Assert.Null(IndexSelection.Parse("1.5", false));
        Assert.Equal([0], IndexSelection.Parse("0", false, offset: 0)!.Indices);
    }
}
