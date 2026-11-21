using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.ViewModels.Jobs;

// The pure parts of the Mac app's ToleranceSession and ToleranceReport: previews of toleranced builds, the run's timing,
// the perturbations of a build as its rows read them, the parsed `tolerance` result (docs/OPTIMIZATION.md) and the
// text / CSV exports. The session controller itself is not here.

/// <summary>A toleranced build the engine can rebuild (<c>preview_case</c>): a Monte-Carlo trial, or one sensitivity operand at its − or + limit.</summary>
public abstract record TolerancePreviewCase
{
    /// <summary>Monte-Carlo trial <paramref name="K"/> (zero-based).</summary>
    public sealed record Trial(int K) : TolerancePreviewCase;

    /// <summary>Sensitivity operand <paramref name="Operand"/> (zero-based) at its + or − limit.</summary>
    public sealed record Sensitivity(int Operand, bool Plus) : TolerancePreviewCase;

    /// <summary>The engine's <c>preview_case</c> (zero-based).</summary>
    public JsonObject ToJson() => this switch
    {
        Trial t => new JsonObject { ["type"] = "trial", ["trial"] = t.K },
        Sensitivity s => new JsonObject { ["type"] = "sensitivity", ["operand"] = s.Operand, ["positive"] = s.Plus },
        _ => throw new InvalidOperationException(),
    };

    /// <summary>"Tolerance: trial 17, compensated", "Tolerance: Decentre y S3 at +0.02 mm" (in the UI language).</summary>
    public string Title(ToleranceResult? result, OpticalSystem? runSystem, IReadOnlyList<ToleranceRow> resultRows)
    {
        var compensated = result is not null && result.Compensators.Count > 0;
        switch (this)
        {
            case Trial t:
                return compensated ? Loc.F("Jobs_Tolerance_Result_PreviewTrialCompensated", t.K + 1) : Loc.F("Jobs_Tolerance_Result_PreviewTrial", t.K + 1);
            case Sensitivity s:
                var what = runSystem is not null && s.Operand >= 0 && s.Operand < resultRows.Count
                    ? Loc.F("Jobs_Tolerance_Result_PreviewAtLimit", resultRows[s.Operand].DisplayLabel(runSystem), resultRows[s.Operand].LimitText(s.Plus))
                    : result?.Sensitivities.FirstOrDefault(x => x.Id == s.Operand)?.Label is { } label ? EngineText.T(label) : Loc.F("Jobs_Tolerance_Result_PreviewOperand", s.Operand + 1);
                return compensated ? Loc.F("Jobs_Tolerance_Result_PreviewCaseCompensated", what) : Loc.F("Jobs_Tolerance_Result_PreviewCase", what);
            default: throw new InvalidOperationException();
        }
    }

    /// <summary>Whether a window preview title is one of <see cref="Title"/>'s (in the UI language, or English).</summary>
    public static bool IsTitle(string? title)
    {
        if (title is null) return false;
        foreach (var format in new[] { Loc.S("Jobs_Tolerance_Result_PreviewCase"), Loc.English("Jobs_Tolerance_Result_PreviewCase") ?? "Tolerance: {0}" })
        {
            var prefix = format.Split("{0}")[0];
            if (prefix.Length > 0 && title.StartsWith(prefix, StringComparison.Ordinal)) return true;
        }
        return false;
    }
}

/// <summary>
/// The engine's rebuild of a toleranced build: the perturbed system with its compensators re-optimised, the criterion,
/// the compensator values and the perturbation of every operand (parameter units, run order).
/// </summary>
public sealed record TolerancePreview(
    TolerancePreviewCase Case, OpticalSystem System, double Criterion, IReadOnlyList<double> Compensators, IReadOnlyList<double> Perturbations)
{
    /// <summary>Reads the <c>preview</c> object of a result; <c>null</c> when it has no system.</summary>
    public static TolerancePreview? FromJson(JsonNode? json, TolerancePreviewCase previewCase)
    {
        if (json is not JsonObject || json["system"] is not { } s) return null;
        OpticalSystem system;
        try { system = OpticalSystem.FromJson(s); }
        catch (Exception e) when (e is global::System.Text.Json.JsonException or InvalidOperationException or FormatException or ArgumentException) { return null; }
        return new TolerancePreview(
            previewCase, system, json.At("criterion").Nf() ?? double.NaN, json.At("compensators").NfArray() ?? [], json.At("perturbations").NfArray() ?? []);
    }
}

/// <summary>Where a run stands and how long the rest should take, from its progress.</summary>
/// <param name="TrialsDone">Monte-Carlo trials done.</param>
/// <param name="SecondsPerTrial">The wall time per trial (measured, or the last run's, or the sensitivity's time per evaluation).</param>
/// <param name="SecondsLeft">The time left.</param>
public readonly record struct ToleranceTiming(int? TrialsDone = null, double? SecondsPerTrial = null, double? SecondsLeft = null)
{
    /// <summary>The share of the job's progress the sensitivity takes (the engine gives it a fifth when both run).</summary>
    public static double SensitivityShare(ToleranceMode mode) => mode switch
    {
        ToleranceMode.Sensitivity => 1,
        ToleranceMode.MonteCarlo => 0,
        _ => 0.2,
    };

    /// <param name="elapsed">Seconds since the job started.</param>
    /// <param name="monteCarloElapsed">Seconds since its Monte Carlo began, if it has.</param>
    /// <param name="fraction">The job's progress, 0 … 1.</param>
    /// <param name="operands">The sensitivity's operands.</param>
    public static ToleranceTiming Estimate(
        double elapsed, double? monteCarloElapsed, double fraction, ToleranceMode mode, int trials, int operands, double? lastSecondsPerTrial)
    {
        var f = Math.Min(Math.Max(fraction, 0), 1);
        var split = SensitivityShare(mode);
        var t = new ToleranceTiming();
        if (mode != ToleranceMode.Sensitivity && (f > split || monteCarloElapsed is not null))
        {
            var mcf = split < 1 ? Math.Min(Math.Max((f - split) / (1 - split), 0), 1) : 0;
            var done = (int)Math.Round(mcf * trials, MidpointRounding.AwayFromZero);
            t = t with { TrialsDone = done };
            t = t with { SecondsPerTrial = monteCarloElapsed is { } mce && done >= 2 && mce > 0 ? mce / done : lastSecondsPerTrial };
            if (t.SecondsPerTrial is { } spt) t = t with { SecondsLeft = Math.Max(trials - done, 0) * spt };
            return t;
        }
        // In the sensitivity: its own rest by its rate, then the trials.
        var sf = split > 0 ? f / split : 0;
        double evaluations = 2 * Math.Max(operands, 0) + 1;
        if (!(sf >= 0.05))
        {
            if (mode != ToleranceMode.Sensitivity && lastSecondsPerTrial is { } last)
                t = t with { SecondsPerTrial = last, SecondsLeft = evaluations * last + trials * last };
            return t;
        }
        var sensLeft = elapsed * (1 - sf) / sf;
        var perEvaluation = elapsed / (sf * evaluations);
        if (mode == ToleranceMode.Sensitivity) return t with { SecondsLeft = sensLeft };
        var perTrial = lastSecondsPerTrial ?? perEvaluation;
        return t with { SecondsPerTrial = perTrial, SecondsLeft = sensLeft + trials * perTrial };
    }
}

/// <summary>One perturbation of a build as its row reads it.</summary>
/// <param name="Id">The operand (row position in the run).</param>
/// <param name="Fraction">The perturbation over the limit on its side (−1 … 1).</param>
/// <param name="Text"><c>+0.84′</c>, <c>−1.9 fr</c> in the row's unit (a radius in ΔR or % is scaled from its limit).</param>
/// <param name="Limits">The row's limits, <c>±1′</c>.</param>
public sealed record ToleranceBuildChange(int Id, string Label, double Fraction, string Text, string Limits)
{
    /// <summary>The engine limits of the run's operands (parameter units), from the run's job settings.</summary>
    public static IReadOnlyList<(double Minus, double Plus)> RunLimits(IReadOnlyDictionary<string, JsonNode?>? runSettings) =>
        [.. (runSettings?.GetValueOrDefault("tolerances") as JsonArray ?? []).Select(o => (o.At("minus").Nf() ?? double.NaN, o.At("plus").Nf() ?? double.NaN))];

    /// <summary>
    /// A build's perturbations (parameter units, run order) as the rows read them, largest share of its limit first; zero
    /// ones left out. <paramref name="limits"/> and <paramref name="rows"/> are the run's (see <see cref="RunLimits"/>).
    /// </summary>
    public static List<ToleranceBuildChange> Changes(
        IReadOnlyList<double> perturbations, IReadOnlyList<(double Minus, double Plus)> limits, IReadOnlyList<ToleranceRow> rows, OpticalSystem? system)
    {
        var output = new List<ToleranceBuildChange>();
        for (var k = 0; k < perturbations.Count; k++)
        {
            var d = perturbations[k];
            if (d == 0 || !double.IsFinite(d) || k >= limits.Count || k >= rows.Count) continue;
            var row = rows[k];
            var lim = d > 0 ? limits[k].Plus : limits[k].Minus;
            if (!(lim > 0)) continue;
            var f = d / lim;
            var shown = (d > 0 ? row.Plus : row.Minus) * Math.Abs(f);
            output.Add(new ToleranceBuildChange(
                k, system is null ? row.Kind.Title() : row.DisplayLabel(system), f, (d > 0 ? "+" : "−") + MeritFormat.Short(Tidy(shown)) + row.Unit.Suffix(),
                row.LimitsText));
        }
        return [.. output.OrderByDescending(c => Math.Abs(c.Fraction))];
    }

    /// <summary>Three significant digits for a perturbation.</summary>
    internal static double Tidy(double v)
    {
        if (v == 0 || !double.IsFinite(v)) return v;
        var q = Math.Pow(10, Math.Floor(Math.Log10(Math.Abs(v))) - 2);
        return Math.Round(v / q, MidpointRounding.AwayFromZero) * q;
    }
}

/// <summary>The parts of a <c>tolerance</c> result the panel shows (docs/OPTIMIZATION.md).</summary>
public sealed class ToleranceResult
{
    public sealed record Sensitivity(
        int Id, string Label, ParamPath? Param, double Minus, double Plus, double CriterionMinus, double CriterionPlus, double ChangeMinus,
        double ChangePlus, double Worst, double ExpectedChange, bool Ok, IReadOnlyList<double> CompensatorsMinus, IReadOnlyList<double> CompensatorsPlus)
    {
        /// <summary>The limit whose change is the larger.</summary>
        public bool WorsePlus => !(ChangeMinus > ChangePlus);
    }

    public sealed record Compensator(ParamPath Param, double? Range);

    /// <summary>One recorded Monte-Carlo trial: its criterion, compensator values and the perturbation of every operand (parameter units).</summary>
    public sealed record TrialRecord(int Trial, double Criterion, bool Ok, IReadOnlyList<double> Compensators, IReadOnlyList<double> Perturbations);

    public readonly record struct Percentile(double Level, double Value);

    public string Status { get; }
    public bool Stopped { get; }
    public CriterionDisplay Display { get; }
    public IReadOnlyList<Compensator> Compensators { get; }
    /// <summary>The design as it is with the compensators re-optimised (the sensitivity's, else the Monte Carlo's).</summary>
    public double Nominal { get; }
    public IReadOnlyList<double> NominalCompensators { get; }
    // Sensitivity
    public bool HasSensitivity { get; }
    /// <summary>Worst first (the engine's ranking).</summary>
    public IReadOnlyList<Sensitivity> Sensitivities { get; }
    public double RssChange { get; }
    /// <summary>Predicted mean criterion: nominal plus the expected changes (nominal + RSS before engine result version 3).</summary>
    public double Predicted { get; }
    public bool SensitivityCancelled { get; }
    // Monte Carlo
    public int Trials { get; }
    public int Failures { get; }
    public long Seed { get; }
    public double Mean { get; }
    public double StdDev { get; }
    public double Min { get; }
    public double Max { get; }
    /// <summary>The engine's percentiles of the criterion (percent, value).</summary>
    public IReadOnlyList<Percentile> Percentiles { get; }
    /// <summary>The criterion of every trial by trial number (not finite: failed).</summary>
    public IReadOnlyList<double> TrialCriteria { get; }
    /// <summary>Finite criteria of the trials.</summary>
    public IReadOnlyList<double> Criteria { get; }
    public IReadOnlyList<TrialRecord> Records { get; }
    /// <summary>Wall time per completed trial.</summary>
    public double? SecondsPerTrial { get; }
    public double? Yield { get; }
    public double? Threshold { get; }
    public IReadOnlyList<double> CompensatorMean { get; }
    public IReadOnlyList<double> CompensatorStd { get; }

    public ToleranceResult(JsonNode? json, OpticalSystem? system, CriterionDisplay display = CriterionDisplay.Merit)
    {
        Display = display;
        Status = json.At("status").StringOrNull() ?? "";
        Stopped = json.At("stopped").BoolOrNull() ?? Status == "stopped";
        Compensators = [.. (json.At("compensators") as JsonArray ?? []).Select(c =>
            ToleranceRow.TryPath(c.At("param")) is { } p ? new Compensator(p, c.At("range").Nf() is { } r && double.IsFinite(r) ? r : null) : null).OfType<Compensator>()];
        var s = json.At("sensitivity");
        var mc = json.At("monte_carlo");
        HasSensitivity = s is not null;
        var nominalJson = s.At("nominal") ?? mc.At("nominal");
        Nominal = nominalJson.At("criterion").Nf() ?? double.NaN;
        NominalCompensators = nominalJson.At("compensators").NfArray() ?? [];
        var ops = (s.At("operands") as JsonArray ?? []).Select((o, k) =>
        {
            var path = ToleranceRow.TryPath(o.At("param"));
            var label = o.At("label").StringOrNull() ?? "";
            if ((label.Length == 0 || label.StartsWith('{')) && path is not null && system is not null) label = path.EnglishLabel(system);
            return new Sensitivity(
                (int)(o.At("index").IntOrNull() ?? k), label, path, Num(o, "minus"), Num(o, "plus"), Num(o, "criterion_minus"), Num(o, "criterion_plus"),
                Num(o, "change_minus"), Num(o, "change_plus"), Num(o, "worst"), Num(o, "expected_change"), o.At("ok").BoolOrNull() ?? true,
                o.At("compensators_minus").NfArray() ?? [], o.At("compensators_plus").NfArray() ?? []);
        }).ToList();
        var ranked = (s.At("ranking").IntArray() ?? []).Select(r => ops.FirstOrDefault(o => o.Id == r)).OfType<Sensitivity>().ToList();
        Sensitivities = ranked.Count == ops.Count ? ranked : [.. ops.OrderByDescending(o => double.IsFinite(o.Worst) ? o.Worst : double.NegativeInfinity)];
        RssChange = Num(s, "rss_change");
        Predicted = Num(s, "predicted_criterion");
        SensitivityCancelled = s.At("cancelled").BoolOrNull() ?? false;
        Trials = (int)(mc.At("trials").IntOrNull() ?? 0);
        Failures = (int)(mc.At("failures").IntOrNull() ?? 0);
        Seed = mc.At("seed").IntOrNull() ?? 0;
        Mean = Num(mc, "mean");
        StdDev = Num(mc, "std_dev");
        Min = Num(mc, "min");
        Max = Num(mc, "max");
        Percentiles = [.. (mc.At("percentiles") as JsonArray ?? []).Select(p =>
            p.At("percentile").Nf() is { } q && p.At("value").Nf() is { } v ? new Percentile(q, v) : (Percentile?)null).OfType<Percentile>()];
        TrialCriteria = mc.At("criteria").NfArray() ?? [];
        Criteria = [.. TrialCriteria.Where(double.IsFinite)];
        Records = [.. (mc.At("records") as JsonArray ?? []).Select(r =>
            r.At("trial").IntOrNull() is { } t
                ? new TrialRecord((int)t, Num(r, "criterion"), r.At("ok").BoolOrNull() ?? true, r.At("compensators").NfArray() ?? [], r.At("perturbations").NfArray() ?? [])
                : null).OfType<TrialRecord>()];
        SecondsPerTrial = mc.At("seconds_per_trial").Nf() is { } spt && double.IsFinite(spt) && spt >= 0 ? spt : null;
        Yield = mc.At("yield_fraction").Nf() is { } y && double.IsFinite(y) ? y : null;
        Threshold = mc.At("threshold").Nf() is { } th && double.IsFinite(th) ? th : null;
        CompensatorMean = mc.At("compensator_mean").NfArray() ?? [];
        CompensatorStd = mc.At("compensator_std").NfArray() ?? [];
    }

    /// <summary>A number under <paramref name="key"/>, NaN when missing or not a number.</summary>
    static double Num(JsonNode? o, string key) => o.At(key).Nf() ?? double.NaN;

    public bool HasMonteCarlo => Trials > 0;
    public bool HasNominal => double.IsFinite(Nominal);

    /// <summary>The record of trial <paramref name="k"/>, if it was kept.</summary>
    public TrialRecord? Record(int k) => Records.FirstOrDefault(r => r.Trial == k);

    /// <summary>
    /// Operand <paramref name="s"/>'s share of the predicted change: its expected change over the sum of the (non-negative)
    /// expected changes; for a result without expected changes, worst² / Σ worst² (its share of the RSS).
    /// </summary>
    public double Share(Sensitivity s)
    {
        var expected = Sensitivities.Any(x => double.IsFinite(x.ExpectedChange));
        double Part(Sensitivity x) =>
            expected ? (double.IsFinite(x.ExpectedChange) ? Math.Max(x.ExpectedChange, 0) : 0) : (double.IsFinite(x.Worst) ? Math.Max(x.Worst, 0) * Math.Max(x.Worst, 0) : 0);
        var total = Sensitivities.Sum(Part);
        return total > 0 ? Part(s) / total : 0;
    }

    /// <summary>The worst offenders: the largest shares that make up half of the predicted change (at most five, each above 5 %).</summary>
    public IReadOnlySet<int> WorstOffenders
    {
        get
        {
            var output = new HashSet<int>();
            var sum = 0.0;
            foreach (var (id, f) in Sensitivities.Select(s => (s.Id, Share(s))).OrderByDescending(x => x.Item2))
            {
                if (!(f > 0.05 && sum < 0.5 && output.Count < 5)) break;
                output.Add(id);
                sum += f;
            }
            return output;
        }
    }

    /// <summary>The fraction of trials whose criterion is at most <paramref name="c"/>.</summary>
    public double Cumulative(double c) => Criteria.Count == 0 ? double.NaN : (double)Criteria.Count(x => x <= c) / Criteria.Count;

    /// <summary>The trials whose shown value lies in <c>[low, high]</c>, worst first.</summary>
    public List<int> TrialsShownIn(double low, double high) => WorstFirst(Enumerable.Range(0, TrialCriteria.Count).Where(k =>
    {
        var v = Display.Value(TrialCriteria[k]);
        return double.IsFinite(v) && v >= low && v <= high;
    }));

    /// <summary>Every completed trial, worst first.</summary>
    public List<int> TrialsWorstFirst => WorstFirst(Enumerable.Range(0, TrialCriteria.Count).Where(k => double.IsFinite(TrialCriteria[k])));

    /// <summary>Lower merits are better in every display.</summary>
    List<int> WorstFirst(IEnumerable<int> ks) => [.. ks.OrderByDescending(k => TrialCriteria[k])];

    /// <summary>Range of a compensator over the sensitivity evaluations, relative to its nominal value: (lowest, highest).</summary>
    public (double Low, double High)? CompensatorRange(int j)
    {
        if (!HasSensitivity || j < 0 || j >= NominalCompensators.Count) return null;
        var n = NominalCompensators[j];
        var v = Sensitivities.SelectMany(s => new[] { s.CompensatorsMinus, s.CompensatorsPlus }.Where(a => j < a.Count).Select(a => a[j])).Where(double.IsFinite).ToList();
        if (v.Count == 0) return null;
        return (Math.Min(v.Min() - n, 0), Math.Max(v.Max() - n, 0));
    }
}

/// <summary>The result as text and CSV: a summary to paste into a report, the sensitivity table and the Monte-Carlo trials.</summary>
public static class ToleranceReport
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;

    /// <summary>A plain-text summary: criterion, nominal, prediction, percentiles, yield, the largest contributors and the compensators.</summary>
    public static string Summary(ToleranceResult r, IReadOnlyList<ToleranceRow> rows, OpticalSystem system, string title)
    {
        var d = r.Display;
        var lines = new List<string> { $"Tolerance analysis: {title}", $"Criterion: {d.EnglishTitle()}{(d.Unit().Length == 0 ? "" : $" ({d.Unit()})")}" };
        if (r.Stopped) lines.Add("Stopped early: statistics of the completed evaluations.");
        if (r.HasSensitivity)
        {
            lines.Add($"Nominal: {d.Format(d.Value(r.Nominal))}");
            lines.Add($"Predicted mean (nominal + expected changes): {d.Format(d.Value(r.Predicted))}");
        }
        if (r.HasMonteCarlo)
        {
            lines.Add($"Monte Carlo: {r.Trials} trials, seed {r.Seed}{(r.Failures > 0 ? $", {r.Failures} failed" : "")}");
            if (r.SecondsPerTrial is { } t) lines.Add($"  {MeritFormat.Short(t)} s per trial");
            foreach (var p in r.Percentiles)
                lines.Add($"  {MeritFormat.Short(p.Level)} % of builds {(d.HigherIsBetter() ? "at least" : "at most")}: {d.Format(d.Value(p.Value))}");
            if (r.Yield is { } y) lines.Add($"  Yield: {MeritFormat.Percent(y)}");
        }
        if (r.HasSensitivity)
        {
            lines.Add("Largest contributors:");
            foreach (var s in r.Sensitivities.Take(5))
                lines.Add($"  {Label(s, rows, system)} {Limits(s, rows)}: {MeritFormat.Percent(r.Share(s), 0)} of the predicted change");
        }
        for (var j = 0; j < r.Compensators.Count; j++)
        {
            var c = r.Compensators[j];
            var line = "Compensator " + new ToleranceCompensator(c.Param, c.Range).EnglishTitle(system);
            if (r.CompensatorRange(j) is { } range) line += $": moved {MeritFormat.Short(range.Low)} to +{MeritFormat.Short(range.High)} {c.Param.Unit(system)}";
            if (c.Range is { } rg) line += $" of ±{MeritFormat.Short(rg)}";
            if (j < r.CompensatorStd.Count) line += $"; Monte Carlo σ {MeritFormat.Short(r.CompensatorStd[j])}";
            lines.Add(line);
        }
        return string.Join("\n", lines);
    }

    /// <summary>One line per sensitivity operand, worst first.</summary>
    public static string SensitivityCsv(ToleranceResult r, IReadOnlyList<ToleranceRow> rows, OpticalSystem system)
    {
        var d = r.Display;
        var n = Name(d);
        var output = new List<string> { $"rank,tolerance,minus,plus,unit,parameter,criterion_minus,criterion_plus,{n}_minus,{n}_plus,change_minus,change_plus,share" };
        for (var k = 0; k < r.Sensitivities.Count; k++)
        {
            var s = r.Sensitivities[k];
            var row = s.Id >= 0 && s.Id < rows.Count ? rows[s.Id] : null;
            string[] fields =
            [
                (k + 1).ToString(C), Csv(Label(s, rows, system)), Num(row?.Minus ?? s.Minus), Num(row?.Plus ?? s.Plus), Csv(row?.Unit.Label() ?? ""),
                Csv(s.Param?.EnglishLabel(system) ?? s.Label), Num(s.CriterionMinus), Num(s.CriterionPlus), Num(d.Value(s.CriterionMinus)), Num(d.Value(s.CriterionPlus)),
                Num(d.Value(s.CriterionMinus) - d.Value(r.Nominal)), Num(d.Value(s.CriterionPlus) - d.Value(r.Nominal)), Num(r.Share(s)),
            ];
            output.Add(string.Join(",", fields));
        }
        return string.Join("\n", output) + "\n";
    }

    /// <summary>
    /// One line per Monte-Carlo trial, in order (a failed trial's values empty), with the compensators and perturbations
    /// (parameter units) of the recorded trials.
    /// </summary>
    public static string TrialsCsv(ToleranceResult r, IReadOnlyList<ToleranceRow>? rows = null, OpticalSystem? system = null)
    {
        rows ??= [];
        var d = r.Display;
        var recorded = r.Records.Count > 0;
        var comps = recorded ? Enumerable.Range(1, r.Compensators.Count).Select(j => $"compensator_{j}").ToList() : [];
        var perts = recorded ? rows.Select(row => Csv(system is null ? row.Kind.EnglishTitle() : row.Label(system))).ToList() : [];
        var output = new List<string> { string.Join(",", ["trial", "criterion", Name(d), .. comps, .. perts]) };
        for (var k = 0; k < r.TrialCriteria.Count; k++)
        {
            var c = r.TrialCriteria[k];
            var f = new List<string> { (k + 1).ToString(C), Num(c), Num(d.Value(c)) };
            if (recorded)
            {
                var rec = r.Record(k);
                f.AddRange(comps.Select((_, j) => rec is not null && j < rec.Compensators.Count ? Num(rec.Compensators[j]) : ""));
                f.AddRange(perts.Select((_, j) => rec is not null && j < rec.Perturbations.Count ? Num(rec.Perturbations[j]) : ""));
            }
            output.Add(string.Join(",", f));
        }
        return string.Join("\n", output) + "\n";
    }

    static string Name(CriterionDisplay d) => d switch
    {
        CriterionDisplay.Spot => "rms_spot_um",
        CriterionDisplay.Wavefront => "rms_wavefront_waves",
        CriterionDisplay.Mtf => "mtf",
        _ => "merit",
    };

    static string Label(ToleranceResult.Sensitivity s, IReadOnlyList<ToleranceRow> rows, OpticalSystem system) =>
        s.Id >= 0 && s.Id < rows.Count ? rows[s.Id].Label(system) : s.Label;

    static string Limits(ToleranceResult.Sensitivity s, IReadOnlyList<ToleranceRow> rows) =>
        s.Id >= 0 && s.Id < rows.Count ? rows[s.Id].LimitsText : "±" + MeritFormat.Short(Math.Max(s.Minus, s.Plus));

    /// <summary>A number as <c>%.10g</c> writes it; empty when not finite.</summary>
    static string Num(double v) => double.IsFinite(v) ? v.ToString("G10", C).Replace('E', 'e') : "";

    static string Csv(string s) => s.Contains(',') || s.Contains('"') ? "\"" + s.Replace("\"", "\"\"") + "\"" : s;
}
