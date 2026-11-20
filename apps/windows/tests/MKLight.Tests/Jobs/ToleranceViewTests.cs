using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.ViewModels.Jobs;
using MKLight.Model;
using MKLight.Tests.Engine;

namespace MKLight.Tests.Jobs;

/// <summary>The tolerance views' pure parts: histogram bars and their trials, compensator usage, table groups, element pivots, exports.</summary>
public class ToleranceViewTests
{
    static OpticalSystem DoubleGauss => SampleSystems.Get("double_gauss")!;

    static SurfaceID Id(OpticalSystem s, int index) => s.Surfaces[index].Id;

    /// <summary>A Monte Carlo of merits 0, 1, 2, 3, 4, 10, 4.5 and a failed trial; a back-focus compensator.</summary>
    static ToleranceResult Result(double? range = 0.5, string criteria = "[0, 1, 2, 3, 4, 10, 4.5, null]") => new(JsonNode.Parse($$"""
        {
          "compensators": [ { "param": { "param": "thickness", "surface": 11 }{{(range is { } r ? $", \"range\": {r.ToString(CultureInfo.InvariantCulture)}" : "")}} } ],
          "sensitivity": {
            "nominal": { "criterion": 1, "compensators": [ 28.9 ] },
            "operands": [ { "index": 0, "label": "Radius S1", "minus": 0.001, "plus": 0.001, "criterion_minus": 1.2, "criterion_plus": 1.1,
                            "change_minus": 0.2, "change_plus": 0.1, "worst": 0.2, "expected_change": 0.1, "ok": true,
                            "compensators_minus": [ 28.95 ], "compensators_plus": [ 28.4 ] } ],
            "ranking": [ 0 ], "rss_change": 0.2, "predicted_criterion": 1.1
          },
          "monte_carlo": { "trials": 8, "failures": 1, "seed": 1, "criteria": {{criteria}} }
        }
        """), DoubleGauss, CriterionDisplay.Merit);

    [Fact]
    public void HistogramBarsCountEveryCompletedTrialOnce()
    {
        var r = Result();
        var bins = ToleranceHistogram.Bins(r);
        Assert.Equal(10, bins.Count); // under 60 trials
        Assert.Equal([1, 1, 1, 1, 2, 0, 0, 0, 0, 1], bins.Select(b => b.Count));
        Assert.Equal((0.0, 1.0, 0.5), (bins[0].Low, bins[0].High, bins[0].Center));
        Assert.Equal(10, bins[^1].High);
        Assert.Equal(7, bins.Sum(b => b.Count));
        Assert.Equal((10, 16, 24), (ToleranceHistogram.BarCount(59), ToleranceHistogram.BarCount(60), ToleranceHistogram.BarCount(200)));

        // A value on a shared edge counts in the bar above it, as the counts do; outside the bars, none.
        Assert.Equal(1, ToleranceHistogram.IndexOf(bins, 1));
        Assert.Equal(9, ToleranceHistogram.IndexOf(bins, 10));
        Assert.Null(ToleranceHistogram.IndexOf(bins, -0.1));
        Assert.Null(ToleranceHistogram.IndexOf(bins, double.NaN));

        // A bar's trials, worst first; every trial in exactly the bar that counts it.
        Assert.Equal([6, 4], ToleranceHistogram.Trials(r, bins, 4));
        Assert.Equal([5], ToleranceHistogram.Trials(r, bins, 9));
        Assert.Empty(ToleranceHistogram.Trials(r, bins, 7));
        Assert.Empty(ToleranceHistogram.Trials(r, bins, 12));
        for (var k = 0; k < bins.Count; k++) Assert.Equal(bins[k].Count, ToleranceHistogram.Trials(r, bins, k).Count);
        Assert.Equal(4, ToleranceHistogram.BinOfTrial(r, bins, 6));
        Assert.Null(ToleranceHistogram.BinOfTrial(r, bins, 7)); // failed
        Assert.Null(ToleranceHistogram.BinOfTrial(r, bins, 99));
        Assert.StartsWith("2 trials at ", ToleranceHistogram.Title(CriterionDisplay.Merit, bins[4], 2));
        Assert.StartsWith("1 trial at ", ToleranceHistogram.Title(CriterionDisplay.Merit, bins[0], 1));

        // Every trial alike: one bar; none completed: no bars.
        var same = Result(criteria: "[3, 3, 3]");
        var one = Assert.Single(ToleranceHistogram.Bins(same));
        Assert.Equal(3, one.Count);
        Assert.Equal([0, 1, 2], ToleranceHistogram.Trials(same, [one], 0).Order());
        Assert.Empty(ToleranceHistogram.Bins(Result(criteria: "[null]")));
    }

    [Fact]
    public void CompensatorUsageAgainstTheAllowedRange()
    {
        // The sensitivity moved back focus from 28.4 to 28.95 about 28.9: −0.5 … +0.05.
        var used = ToleranceCompensatorUsage.Of(Result(0.5), 0)!.Value;
        Assert.Equal(-0.5, used.Low, 9);
        Assert.Equal(0.05, used.High, 9);
        var f = used.Fractions!.Value;
        Assert.Equal(-1, f.From, 9);
        Assert.Equal(0.1, f.To, 9);
        Assert.True(used.AtLimit);

        var roomy = ToleranceCompensatorUsage.Of(Result(1), 0)!.Value;
        Assert.Equal(-0.5, roomy.Fractions!.Value.From, 9);
        Assert.False(roomy.AtLimit);
        Assert.True(new ToleranceCompensatorUsage(-0.2, 0.496, 0.5).AtLimit); // 99 % of the range

        var free = ToleranceCompensatorUsage.Of(Result(null), 0)!.Value;
        Assert.Null(free.Fractions);
        Assert.False(free.AtLimit);
        Assert.Null(ToleranceCompensatorUsage.Of(Result(), 1));

        Assert.Equal("+0.170", ToleranceCompensatorUsage.Move(0.1704));
        Assert.Equal("−0.042", ToleranceCompensatorUsage.Move(-0.042));
        Assert.Equal("+0.000", ToleranceCompensatorUsage.Move(-0.0004));
        Assert.Equal("−0.1260", ToleranceCompensatorUsage.Signed4(-0.126049));
        Assert.Equal("—", ToleranceCompensatorUsage.Signed4(double.NaN));
    }

    [Fact]
    public void TableGroupsFollowTheSurfacesAndCollapseByKey()
    {
        var s = DoubleGauss;
        var rows = ToleranceSettings.Defaults(s).Rows;
        var groups = ToleranceTableGroup.Of(rows, s);
        Assert.Equal(["L1", "Air after L1", "L2", "Air after L2"], groups.Take(4).Select(g => g.Key));
        Assert.StartsWith("L1 · ", groups[0].Title);
        Assert.Equal("Air after L1", groups[1].Title);
        Assert.Equal(rows.Count, groups.Sum(g => g.Rows.Count));
        // In surface order whatever the rows' order; rows keep their order within a group.
        var reversed = ToleranceTableGroup.Of([.. Enumerable.Reverse(rows)], s);
        Assert.Equal(groups.Select(g => g.Key), reversed.Select(g => g.Key));
        Assert.Equal(groups[0].Rows.Reverse().Select(r => r.Id), reversed[0].Rows.Select(r => r.Id));

        // One element line per first surface: four elements.
        var leading = ToleranceTableGroup.LeadingElementRows(rows);
        Assert.Equal(4, leading.Count);
        Assert.All(rows.Where(r => leading.Contains(r.Id)), r => Assert.True(r.Kind.IsElement()));
        Assert.Equal(4, rows.Where(r => leading.Contains(r.Id)).Select(r => r.Surface).Distinct().Count());
    }

    [Fact]
    public void ElementPivotAndRangeMoveTheGroupTogether()
    {
        var s = DoubleGauss;
        var t = new ToleranceSettings();
        var a = new ToleranceRow(ToleranceKind.ElementTiltX, Id(s, 3), 1, last: Id(s, 5));
        var b = new ToleranceRow(ToleranceKind.ElementDecenterY, Id(s, 3), 0.03, last: Id(s, 5));
        t.Add([a, b]);
        Assert.Equal(TolerancePivotChoice.First, TolerancePivotChoice.Of(a, s));
        Assert.Equal("S3, first vertex", TolerancePivotChoice.Text(a.PivotIndex(s), a.MovingSurfaces(s)));
        Assert.Equal("S3 S4 S5 move together", TolerancePivotChoice.MovingText(a.MovingSurfaces(s)));
        Assert.Equal("S3 moves", TolerancePivotChoice.MovingText([3]));

        // The last vertex: both rows follow.
        Assert.True(TolerancePivotChoice.Last.Apply(t, a.Id, s, 3, 5));
        Assert.Equal((Id(s, 5), Id(s, 5)), (a.Pivot!.Value, b.Pivot!.Value));
        Assert.Equal(TolerancePivotChoice.Last, TolerancePivotChoice.Of(b, s));
        Assert.Equal("S5, last vertex", TolerancePivotChoice.Text(b.PivotIndex(s), b.MovingSurfaces(s)));

        // A shorter range keeps the pivot where it sits relative to it (the last vertex is now S4).
        Assert.True(TolerancePivotChoice.Of(a, s).Apply(t, a.Id, s, 3, 4));
        Assert.Equal((Id(s, 4), Id(s, 4)), (b.Last!.Value, b.Pivot!.Value));

        // Any surface; back to the first vertex, stored as none.
        var mid = new TolerancePivotChoice(TolerancePivotPlace.Surface, 7);
        Assert.True(mid.Apply(t, b.Id, s, 3, 5));
        Assert.Equal(mid, TolerancePivotChoice.Of(a, s));
        Assert.Equal("S7", TolerancePivotChoice.Text(7, a.MovingSurfaces(s)));
        Assert.Equal(7, mid.Index(3, 5));
        Assert.True(TolerancePivotChoice.First.Apply(t, a.Id, s, 3, 5));
        Assert.Null(b.Pivot);
        Assert.False(TolerancePivotChoice.First.Apply(t, a.Id, s, 5, 3));
        Assert.False(TolerancePivotChoice.First.Apply(t, a.Id, s, 3, 99));
        Assert.Equal("?", TolerancePivotChoice.Text(null, []));

        // Rows starting on one surface that disagree are flagged, both of them.
        var c = new ToleranceContext(s) { SemiDiameters = s.Surfaces.ToDictionary(x => x.Id, _ => 10.0) };
        Assert.Empty(t.Issues(c));
        b.Last = Id(s, 4);
        var issues = t.Issues(c);
        Assert.Equal(2, issues.Count);
        Assert.Contains("move one group", issues[a.Id].Text);
        Assert.Contains("S3", issues[b.Id].Text);
    }

    [Fact]
    public void ExportsAreInvariantWhateverTheCulture()
    {
        var culture = CultureInfo.CurrentCulture;
        try
        {
            CultureInfo.CurrentCulture = new CultureInfo("de-DE");
            var s = DoubleGauss;
            var r = Result();
            var rows = new List<ToleranceRow> { new(ToleranceKind.Radius, Id(s, 1), 3) };
            var csv = ToleranceReport.SensitivityCsv(r, rows, s);
            Assert.DoesNotContain("\r", csv);
            Assert.EndsWith("\n", csv);
            Assert.StartsWith("1,Radius S1,3,3,fr,", csv.Split('\n')[1]);
            Assert.Contains(",1.2,1.1,", csv);
            var trials = ToleranceReport.TrialsCsv(r, rows, s).Split('\n');
            Assert.Equal("7,4.5,4.5", trials[7]);
            Assert.Equal("8,,", trials[8]);
            Assert.Contains("Monte Carlo: 8 trials, seed 1, 1 failed", ToleranceReport.Summary(r, rows, s, "Double Gauss"));
        }
        finally { CultureInfo.CurrentCulture = culture; }

        Assert.Equal("Double Gauss trials.csv", ToleranceExport.FileName("Double Gauss", "trials"));
        Assert.Equal("a-b-c sensitivity.csv", ToleranceExport.FileName(" a/b:c ", "sensitivity"));
        Assert.Equal("Tolerance trials.csv", ToleranceExport.FileName("  ", "trials"));
    }
}
