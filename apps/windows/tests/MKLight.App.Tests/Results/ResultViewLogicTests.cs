using MKLight.App.Views.Analysis;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Tests.Results;

/// <summary>The logic behind the round-3 result views: PSF legibility, the design/candidate comparison, validation, summaries.</summary>
public class ResultViewLogicTests
{
    [Fact]
    public void PsfFocusFollowsTheEnergyOfAnAberratedPsf()
    {
        // A weak peak over a wide, low halo: every cell is above 0.1 % of the peak, but 95 % of the
        // energy lies within ±10 cells, so the core is cropped to about that.
        const int n = 81;
        var v = new double[n * n];
        for (var j = 0; j < n; j++)
            for (var i = 0; i < n; i++)
            {
                var r = Math.Max(Math.Abs(i - 40), Math.Abs(j - 40));
                v[j * n + i] = r <= 10 ? 1 : 0.0015;
            }
        v[40 * n + 40] = 1.2;
        var g = new MapGrid(n, n, v) { X0 = -40, X1 = 40, Y0 = -40, Y1 = 40 };
        var f = PsfData.Focus(g);
        Assert.True(f.Nx < n);
        Assert.InRange(f.Nx, 2 * 10 + 1, 2 * 16 + 1); // the energy square, widened by 15 % and 2 cells
    }

    [Fact]
    public void PsfStartsOnTheLogScaleWhenTheStrehlRatioIsLow()
    {
        var g = new MapGrid(1, 1, [1.0]);
        PsfData.Item Item(double strehl) => new("Field 1", g, strehl, 0.001, 1, 1, "strehl");
        Assert.True(PsfData.PrefersLog(Item(0.026)));
        Assert.False(PsfData.PrefersLog(Item(0.8)));
        Assert.False(PsfData.PrefersLog(Item(double.NaN)));
    }

    [Fact]
    public void FirstOrderTooltipsCarryTheFullLabelAndItsMeaning()
    {
        var t = FirstOrderHelp.Tooltip("entrance_pupil_position", "Entrance pupil position from surface 1", "mm");
        Assert.StartsWith("Entrance pupil position from surface 1 (mm)\n", t);
        Assert.Contains("first surface", t);
        Assert.Equal("Something new", FirstOrderHelp.Tooltip("unknown_key", "Something new", ""));
        Assert.Null(FirstOrderHelp.Description("unknown_key"));
    }

    static ResultData Spot(double rms) => TestData.Result("spot_diagram",
        $$"""{ "fields": [ {"rms_radius": {{rms.ToString(System.Globalization.CultureInfo.InvariantCulture)}} } ], "airy_radius": 0.0029 }""");

    [Fact]
    public void FooterGivesTheDesignsNumbersBesideTheCandidates()
    {
        var design = Spot(0.02592);
        var candidate = Spot(0.01840);
        var previewed = new ResultData { Analysis = candidate.Analysis, Json = candidate.Json, Baseline = design };
        var text = KeyNumbers.Compared(previewed);
        Assert.Contains("Max RMS 18.40 µm (design 25.92 µm)", text);
        Assert.Contains("Airy 2.90 µm", text);
        Assert.DoesNotContain("Airy 2.90 µm (design", text); // unchanged numbers are not repeated
        Assert.Equal(KeyNumbers.Text(candidate), KeyNumbers.Compared(candidate));
    }

    [Fact]
    public void ValidationIssuesAreOrderedBySeverityWithTheirSurfaceAndField()
    {
        var d = ValidationData.Read(TestData.Obj("""
            { "ok": false, "errors": 1, "warnings": 1, "issues": [
              {"severity": "info", "code": "note", "message": "A note"},
              {"severity": "warning", "code": "chief_ray_missed", "message": "The chief ray of field 2 misses surface 4", "surface": 4},
              {"severity": "error", "code": "negative_thickness", "message": "Thickness after surface 3 is negative", "surface": 3} ] }
            """))!;
        Assert.Equal([ValidationData.Severity.Error, ValidationData.Severity.Warning, ValidationData.Severity.Info], d.Issues.Select(i => i.Severity));
        Assert.Equal(3, d.Issues[0].Surface);
        Assert.Equal(2, d.Issues[1].Field);
        Assert.Null(d.Issues[2].Field);
        Assert.Equal("1 error, 1 warning, 1 note", d.Summary());
        Assert.Equal("No issues", ValidationData.Read(TestData.Obj("""{"issues": []}"""))!.Summary());
        Assert.Null(ValidationData.Read(TestData.Obj("{}")));
        Assert.Equal(12, ValidationData.FieldIndex("Field index 12 is outside"));
    }

    [Fact]
    public void ChartSummarySaysWhatThePlotShows()
    {
        var p = new ChartPanel { Title = "MTF", XTitle = "Spatial frequency (cycles/mm)", YTitle = "Modulation" };
        p.Series.Add(new ChartSeries([0, 20, 40], [1, 0.62, 0.3], SeriesColor.Field(0)));
        p.Series.Add(new ChartSeries([0, 20, 40], [1, 0.58, 0.2], SeriesColor.Field(0), Dashed: true));
        var s = ChartSummary.Describe(p);
        Assert.Equal("MTF: 2 curves, spatial frequency (cycles/mm) 0 to 40, modulation 0.2 to 1; at 20: 0.62, 0.58", s);
        Assert.Equal(0.5, ChartSummary.At(new ChartSeries([2, 0], [1, 0], SeriesColor.Neutral), 1));
        Assert.True(double.IsNaN(ChartSummary.At(new ChartSeries([0, 1], [0, 1], SeriesColor.Neutral), 2)));
    }

    [Fact]
    public void GlassMapListsTheGlassesTheEngineOmitted()
    {
        var r = TestData.Result("glass_map", """
            { "glasses": [
              {"name": "N-BK7", "catalog": "SCHOTT", "nd": 1.5168, "vd": 64.17, "pgf": 0.5349, "surfaces": [1]} ],
              "omitted": [
              {"catalog": "SCHOTT", "name": "IRG24", "reason": "range 0.9-12 um misses the d, F and C lines", "surfaces": []},
              {"catalog": "SCHOTT", "name": "IRG26", "reason": "range 0.9-12 um misses the d, F and C lines", "surfaces": [3, 4]} ] }
            """);
        var d = GlassMapData.Read(r)!;
        Assert.Equal(["N-BK7"], d.Glasses.Select(g => g.Name));
        // The design's omitted glasses come first.
        Assert.Equal(["IRG26", "IRG24"], d.Omitted.Select(o => o.Name));
        Assert.True(d.Omitted[0].InDesign);
        Assert.Equal("S3, S4", d.Omitted[0].SurfaceLabel);
        Assert.StartsWith("range 0.9", d.Omitted[1].Reason);
        Assert.Equal(["SCHOTT"], d.Catalogs);
        Assert.Equal([("Glasses", "1"), ("In design", "2"), ("Not on map", "2")], KeyNumbers.Of(r));
    }

    [Fact]
    public void RayGridBinsEachFieldsHitsAboutItsCentroid()
    {
        // 2 fields × 1 wavelength × 3 pupil rays; field 1's third ray did not arrive.
        double nan = double.NaN;
        var d = TestData.Result("ray_grid", """
            { "fields": [0, 1], "field_values": [[0, 0], [0, 10]], "wavelengths": [1], "wavelengths_um": [0.5876],
              "pupil_count": 3, "arrived": 5, "launched": 6 }
            """, new FakeArray("points", [2, 1, 3, 2], [1, 1, 3, 1, 2, 1, 0, 5, 0, 7, nan, nan]));
        var g = RayGridData.Read(d)!;
        Assert.Equal(2, g.Fields.Count);
        Assert.Equal((5, 6), (g.Arrived, g.Launched));
        // Field 0 about its centroid (2, 1): x −1, +1, 0.
        Assert.Equal([-1.0, 0, 1, 0, 0, 0], g.Fields[0].Xy[0]);
        Assert.Equal(1.0, g.Fields[0].HalfWidth);
        Assert.Equal(2, g.Fields[1].Arrived);
        Assert.False(g.Micrometres);
        var bins = RayGridData.Bin(g.Fields[0], 4);
        Assert.Equal(3, bins[0].Sum());
        // +y is up: field 1's upper hit lands in the top half.
        var b1 = RayGridData.Bin(g.Fields[1], 4);
        Assert.Equal(1, b1[0][..8].Sum());
    }
}
