using System.Numerics;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;
using Fit = MKLight.App.ViewModels.Jobs.LayoutThumbnailFit;

namespace MKLight.Tests.Jobs;

public class JobFigureTests
{
    static Fit.Box Box(float x0, float y0, float x1, float y1) => new(new Vector2(x0, y0), new Vector2(x1, y1));

    [Fact]
    public void FrameIsSymmetricAboutTheAxis()
    {
        var f = Fit.Frame(Box(0, -3, 50, 10), Box(0, -3, 20, 10));
        Assert.Equal(Box(0, -10, 50, 10), f);
    }

    [Fact]
    public void ALongSlimLayoutIsFramedOnItsLens()
    {
        // An object 1000 mm away: the lens is 20 mm long, everything is 1100 mm long and 20 mm tall.
        var f = Fit.Frame(Box(-1000, -10, 100, 10), Box(0, -8, 20, 8));
        Assert.Equal(-10, f.Min.X);
        Assert.Equal(100, f.Max.X);
    }

    [Fact]
    public void TwoLayoutsShareOneScaleAndCentre()
    {
        var before = Fit.Frame(Box(0, -10, 60, 10), Box(0, -10, 20, 10));
        var after = Fit.Frame(Box(0, -12, 80, 12), Box(0, -12, 24, 12));
        var union = Fit.Union([before, after])!.Value;
        Assert.Equal(Box(0, -12, 80, 12), union);
        var fit = Fit.Fit(union, 136, 78, 8, 6);
        // Width-limited: (136 - 16) / 80 = 1.5 px/mm (height would allow 66 / 24 = 2.75).
        Assert.Equal(1.5f, fit.Scale, 5);
        Assert.Equal(new Vector2(40, 0), fit.Center);
        // The axis is the thumbnail's middle; the shared frame's ends meet the margins.
        Assert.Equal(new Vector2(8, 39), Fit.ToScreen(new Vector2(0, 0), fit, 136, 78));
        Assert.Equal(new Vector2(128, 39), Fit.ToScreen(new Vector2(80, 0), fit, 136, 78));
        Assert.Equal(39 - 15, Fit.ToScreen(new Vector2(40, 10), fit, 136, 78).Y, 4);
    }

    [Fact]
    public void UnionOfNothingIsNull() => Assert.Null(Fit.Union([]));

    [Fact]
    public void CandidateGlassesAreMarkedOncePerGlassWithTheirSurfaces()
    {
        var s = SampleSystems.Get("cooke_triplet")!;
        var slots = GlassSlot.All(s);
        SubstitutedGlass G(int slot, string catalog, string name) => new(slots[slot].Id, catalog, name, 1.6, 50);
        var marks = CandidateGlassMarks.Marks([G(2, "SCHOTT", "N-SK16"), G(1, "SCHOTT", "SF5"), G(0, "SCHOTT", "N-SK16"), G(0, "SCHOTT", "N-SK16")], s.SurfaceIndexOf);
        Assert.Equal(["SCHOTT/N-SK16", "SCHOTT/SF5"], marks.Select(m => m.Id));
        Assert.Equal(["S1, S5", "S3"], marks.Select(m => m.Label));
        Assert.Equal("N-SK16", marks[0].Name);
    }

    [Fact]
    public void GlassesOfSurfacesNotInTheDesignAreLeftOut()
    {
        var s = SampleSystems.Get("cooke_triplet")!;
        var marks = CandidateGlassMarks.Marks([new SubstitutedGlass(new SurfaceID(987654), "SCHOTT", "F2", 1.62, 36.4)], s.SurfaceIndexOf);
        Assert.Empty(marks);
    }

    [Fact]
    public void TheOriginalGlassIsTheLenssOwnPointBeforeItsCatalogueGlass()
    {
        (string, string, IReadOnlyList<int>)[] design =
        [
            ("SCHOTT/N-SK16", "catalog", [1, 5]), ("offset:SCHOTT/N-SK16:5", "offset", [5]), ("model:1.62/36:3", "model", [3]),
        ];
        Assert.Equal("SCHOTT/N-SK16", CandidateGlassMarks.Original(design, 1));
        Assert.Equal("offset:SCHOTT/N-SK16:5", CandidateGlassMarks.Original(design, 5));
        Assert.Equal("model:1.62/36:3", CandidateGlassMarks.Original(design, 3));
        Assert.Null(CandidateGlassMarks.Original(design, 7));
    }
}
