using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

/// <summary>The state every design job shares: its settings page, running, a result to review, or a failure.</summary>
public enum JobPhase { Setup, Running, Finished, Failed }

/// <summary>One point of a running optimisation (from the job's stage text <c>"&lt;algorithm&gt; &lt;iteration&gt;: merit &lt;best&gt;"</c>).</summary>
public readonly record struct MeritPoint(int Iteration, double Merit)
{
    /// <summary>Parses the engine's progress stage, e.g. <c>lm 12: merit 3.4e-4</c>.</summary>
    public static MeritPoint? Parse(string stage)
    {
        var colon = stage.IndexOf(':');
        if (colon < 0) return null;
        var left = stage[..colon].Split(' ', StringSplitOptions.RemoveEmptyEntries);
        if (left.Length == 0 || !int.TryParse(left[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out var it)) return null;
        var rhs = stage[(colon + 1)..].Trim();
        if (!rhs.StartsWith("merit", StringComparison.Ordinal)) return null;
        return double.TryParse(rhs[5..].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out var m) ? new MeritPoint(it, m) : null;
    }
}

/// <summary>The parts of an <c>optimize</c> result the page shows.</summary>
public sealed record OptimizeResult(string Status, bool Stopped, bool Improved, double StartMerit, double FinalMerit, int Iterations, int Evaluations,
    bool Feasible, IReadOnlyList<OptimizeResult.Variable> Variables, IReadOnlyList<MeritPoint> History, OpticalSystem? System, string EvaluatedOn)
{
    public sealed record Variable(string Label, ParamPath? Param, double Start, double Value, bool AtBound);

    public static OptimizeResult? FromJson(JsonNode? json)
    {
        if (json is not JsonObject o) return null;
        var status = o["status"].StringOrNull() ?? "";
        OpticalSystem? system = null;
        try { if (o["system"] is { } s) system = OpticalSystem.FromJson(s); }
        catch (Exception e) when (e is global::System.Text.Json.JsonException or InvalidOperationException or FormatException or ArgumentException) { }
        var vars = (o["variables"] as JsonArray ?? []).Select(v =>
        {
            ParamPath? p = null;
            try { if (v?["param"] is { } pj) p = ParamPath.FromJson(pj); }
            catch (Exception e) when (e is global::System.Text.Json.JsonException or InvalidOperationException or FormatException) { }
            return new Variable(v?["label"].StringOrNull() ?? "", p, v?["start"].Nf() ?? double.NaN, v?["value"].Nf() ?? double.NaN, v?["at_bound"].BoolOrNull() ?? false);
        }).ToList();
        var history = (o["history"] as JsonArray ?? []).Select(h =>
        {
            var it = h?["iteration"].IntOrNull();
            var m = (h?["best_merit"] ?? h?["merit"]).Nf();
            return it is { } i && m is { } mm && double.IsFinite(mm) ? new MeritPoint((int)i, mm) : (MeritPoint?)null;
        }).OfType<MeritPoint>().ToList();
        return new OptimizeResult(status, o["stopped"].BoolOrNull() ?? status == "stopped", o["improved"].BoolOrNull() ?? false,
            o["start"]?["merit"].Nf() ?? double.NaN, o["final"]?["merit"].Nf() ?? double.NaN,
            (int)(o["iterations"].IntOrNull() ?? 0), (int)(o["evaluations"].IntOrNull() ?? 0), o["final"]?["feasible"].BoolOrNull() ?? true,
            vars, history, system, o["backend"]?["evaluated_on"].StringOrNull() ?? "");
    }

    public double Improvement => StartMerit > 0 ? 1 - FinalMerit / StartMerit : double.NaN;

    /// <summary>"7 iterations, 51 evaluations. Stopped early: …"</summary>
    public string StatusText()
    {
        var s = $"{Iterations} iterations, {Evaluations} evaluations. ";
        if (Stopped) s += "Stopped early: this is the best design found so far. ";
        if (!Improved) s += "No better design was found; the result equals the start. ";
        if (!Feasible) s += "Some constraints are not met. ";
        return s.TrimEnd();
    }

    /// <summary>The preview title, naming the candidate precisely: <c>Optimize result · merit 0.0123 (−42 %)</c>.</summary>
    public string PreviewTitle()
    {
        var s = (Stopped ? "Optimize result (stopped)" : "Optimize result") + " · merit " + MeritFormat.Merit(FinalMerit);
        if (double.IsFinite(Improvement)) s += " (" + (Improvement > 0 ? "−" : "+") + MeritFormat.Percent(Math.Abs(Improvement), 0) + ")";
        return s;
    }

    /// <summary>A variable's value as shown (curvatures as radii).</summary>
    public static double Displayed(double v, ParamPath? p) => p?.IsCurvature == true ? RadiusDisplay.RadiusFromCurvature(v) : v;
}

/// <summary>Dry run and commit of the design's solves (the <c>solves</c> report, docs/SOLVES.md).</summary>
public sealed record SolveOutcomeRow(int Index, string Target, string Kind, double? Previous, double Value, double Residual)
{
    public bool Changes => Previous != Value;

    /// <summary>The rows of a <c>solves</c> report; curvatures are shown as radii, targets labelled in <paramref name="system"/>.</summary>
    public static IReadOnlyList<SolveOutcomeRow> Read(JsonNode? report, OpticalSystem? system) =>
        (report?["outcomes"] as JsonArray ?? []).Select((o, k) =>
        {
            ParamPath? path = null;
            try { if (o?["target"] is { } t) path = ParamPath.FromJson(t); }
            catch (Exception e) when (e is global::System.Text.Json.JsonException or InvalidOperationException or FormatException) { }
            var kind = o?["kind"].StringOrNull() ?? "";
            var previous = o.NfAt("previous") is { } p && !double.IsNaN(p) ? p : (double?)null;
            var value = o?["value"].Nf() ?? double.NaN;
            if (path?.IsCurvature == true)
            {
                previous = previous is { } pv ? RadiusDisplay.RadiusFromCurvature(pv) : null;
                value = RadiusDisplay.RadiusFromCurvature(value);
            }
            var title = Wire.Parse<SolveType>(kind) is { } st ? st.Title() : kind.Replace('_', ' ');
            return new SolveOutcomeRow((int)(o?["index"].IntOrNull() ?? k), path is null ? "?" : system is null ? path.ToString() : path.Label(system),
                title, previous, value, o?["residual"].Nf() ?? double.NaN);
        }).ToList();

    public static IReadOnlyList<string> Warnings(JsonNode? report) =>
        (report?["warnings"] as JsonArray ?? []).Select(w => w.StringOrNull()).OfType<string>().ToList();
}
