using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>One weighted wavelength (µm); in a sensor table the weight is the response there.</summary>
public readonly record struct SpectrumSample(double Um, double Weight);

public enum SamplingSpacing { EqualWeight, Uniform }

/// <summary>
/// How the engine samples a continuous spectrum (<c>SpectralSampling</c>):
/// <c>samples</c> wavelengths over <c>range_um</c>, in bins of equal energy
/// (each at its bin's centroid, equal weights) or evenly spaced with the
/// energy of their cells as weights.
/// </summary>
public sealed record SpectralSampling(int Samples = 9, double LowerUm = 0.38, double UpperUm = 0.78, SamplingSpacing Spacing = SamplingSpacing.EqualWeight)
{
    /// <summary>The engine's defaults: 9 samples over 0.38–0.78 µm, equal energy.</summary>
    public static readonly SpectralSampling Default = new();
    public const int MinSamples = 1, MaxSamples = 4096;

    public static string SpacingTitle(SamplingSpacing s) => s == SamplingSpacing.EqualWeight ? "Equal energy" : "Uniform";

    /// <summary>
    /// Reads <c>{samples, range_um, spacing}</c>; absent or null is the
    /// default. False for unknown keys or wrong types (the engine rejects them too).
    /// </summary>
    public static bool TryRead(JsonNode? json, out SpectralSampling sampling)
    {
        sampling = Default;
        if (json is null) return true;
        if (json is not JsonObject o || o.Any(p => p.Key is not ("samples" or "range_um" or "spacing"))) return false;
        var s = Default;
        if (o.TryGetPropertyValue("samples", out var n))
        {
            if (Json.Int(n) is not int k) return false;
            s = s with { Samples = k };
        }
        if (o.TryGetPropertyValue("range_um", out var r))
        {
            if (r is not JsonArray { Count: 2 } a || Json.Number(a[0]) is not double lo || Json.Number(a[1]) is not double hi) return false;
            s = s with { LowerUm = lo, UpperUm = hi };
        }
        if (o.TryGetPropertyValue("spacing", out var sp))
        {
            var spacing = Json.Str(sp) switch { "equal_weight" => SamplingSpacing.EqualWeight, "uniform" => SamplingSpacing.Uniform, _ => (SamplingSpacing?)null };
            if (spacing is null) return false;
            s = s with { Spacing = spacing.Value };
        }
        sampling = s;
        return true;
    }

    /// <summary>Only what differs from the defaults (null when nothing does), so the run's settings stay short.</summary>
    public JsonObject? ToJson()
    {
        var o = new JsonObject();
        if (Samples != Default.Samples) o["samples"] = Samples;
        if (LowerUm != Default.LowerUm || UpperUm != Default.UpperUm) o["range_um"] = new JsonArray(LowerUm, UpperUm);
        if (Spacing != Default.Spacing) o["spacing"] = Spacing == SamplingSpacing.Uniform ? "uniform" : "equal_weight";
        return o.Count == 0 ? null : o;
    }

    /// <summary>The engine's checks (<c>validate_sampling</c>).</summary>
    public IReadOnlyList<string> Issues
    {
        get
        {
            var out_ = new List<string>();
            if (Samples is < MinSamples or > MaxSamples) out_.Add("Samples must be between 1 and 4096.");
            if (!(LowerUm > 0 && UpperUm > LowerUm && double.IsFinite(UpperUm))) out_.Add("The range must be positive and increasing.");
            return out_;
        }
    }

    /// <summary>The range moved inside [lo, hi] (for the CIE tables); the whole table when nothing is left.</summary>
    public SpectralSampling Clamped(double lo, double hi)
    {
        double a = Math.Clamp(LowerUm, lo, hi), b = Math.Clamp(UpperUm, lo, hi);
        return b <= a ? this with { LowerUm = lo, UpperUm = hi } : this with { LowerUm = a, UpperUm = b };
    }
}

public enum SpectrumKind { Design, Photopic, Scotopic, Blackbody, Explicit, Sensor }

/// <summary>
/// The <c>spectrum</c> setting of every wavelength-using analysis: absent
/// (the design's wavelengths) or a run-local spectrum in the engine's format,
/// tagged by <c>kind</c>:
/// <code>
/// {"kind": "explicit", "wavelengths": [{"um": 0.55, "weight": 1}]}
/// {"kind": "photopic" | "scotopic", "sampling": {...}}
/// {"kind": "blackbody", "temperature_k": 5500, "sampling": {...}}
/// {"kind": "sensor", "table": [[0.4, 0.1], ...], "sampling": {...}}
/// </code>
/// The engine places the primary at the sample nearest the energy-weighted mean wavelength.
/// </summary>
/// <param name="Samples">The explicit wavelengths, or the sensor's response table.</param>
public sealed record SpectrumSetting(SpectrumKind Kind, SpectralSampling? Sampling = null, double TemperatureK = 5500, IReadOnlyList<SpectrumSample>? Samples = null)
{
    public static readonly SpectrumSetting Design = new(SpectrumKind.Design);

    /// <summary>The CIE tables' ranges in the engine (CIE 018:2019): sampling ranges must lie inside them.</summary>
    public const double PhotopicLo = 0.36, PhotopicHi = 0.83, ScotopicLo = 0.38, ScotopicHi = 0.78;
    /// <summary>Quick picks for the blackbody temperature (K): daylight and tungsten halogen.</summary>
    public static readonly double[] BlackbodyPresets = [5500, 3200];

    public IReadOnlyList<SpectrumSample> List => Samples ?? [];

    public static string Title(SpectrumKind k) => k switch
    {
        SpectrumKind.Design => "Design wavelengths",
        SpectrumKind.Photopic => "Photopic V(λ)",
        SpectrumKind.Scotopic => "Scotopic V′(λ)",
        SpectrumKind.Blackbody => "Blackbody",
        SpectrumKind.Explicit => "Explicit wavelengths",
        _ => "Sensor response",
    };

    /// <summary>The same spectrum with other sampling (no change for design and explicit wavelengths).</summary>
    public SpectrumSetting WithSampling(SpectralSampling s) => Kind is SpectrumKind.Design or SpectrumKind.Explicit ? this : this with { Sampling = s };

    // MARK: JSON

    /// <summary>
    /// Reads the setting; null is the design. Null for anything the engine
    /// would reject structurally (unknown kind or key, wrong types).
    /// </summary>
    public static SpectrumSetting? Read(JsonNode? json)
    {
        if (json is null) return Design;
        if (json is not JsonObject o || Json.Str(o["kind"]) is not string kind) return null;
        bool Keys(params string[] allowed) => o.All(p => p.Key == "kind" || allowed.Contains(p.Key));
        SpectralSampling? Sampling() => SpectralSampling.TryRead(o["sampling"], out var s) ? s : null;
        switch (kind)
        {
            case "photopic" or "scotopic":
                if (!Keys("sampling") || Sampling() is not { } ps) return null;
                return new(kind == "photopic" ? SpectrumKind.Photopic : SpectrumKind.Scotopic, ps);
            case "blackbody":
                if (!Keys("temperature_k", "sampling") || Json.Number(o["temperature_k"]) is not double t || Sampling() is not { } bs) return null;
                return new(SpectrumKind.Blackbody, bs, t);
            case "explicit":
                if (!Keys("wavelengths") || o["wavelengths"] is not JsonArray list) return null;
                var samples = new List<SpectrumSample>();
                foreach (var w in list)
                {
                    if (w is not JsonObject wo || wo.Any(p => p.Key is not ("um" or "weight")) || Json.Number(wo["um"]) is not double um) return null;
                    var weight = 1.0; // the engine's default
                    if (wo.TryGetPropertyValue("weight", out var wv))
                    {
                        if (Json.Number(wv) is not double d) return null;
                        weight = d;
                    }
                    samples.Add(new(um, weight));
                }
                return new(SpectrumKind.Explicit, Samples: samples);
            case "sensor":
                if (!Keys("table", "sampling") || o["table"] is not JsonArray rows || Sampling() is not { } ss) return null;
                var table = new List<SpectrumSample>();
                foreach (var r in rows)
                {
                    if (r is not JsonArray { Count: 2 } p || Json.Number(p[0]) is not double x || Json.Number(p[1]) is not double y) return null;
                    table.Add(new(x, y));
                }
                return new(SpectrumKind.Sensor, ss, Samples: table);
            default:
                return null;
        }
    }

    /// <summary>The engine's JSON; null for the design (the setting is left out).</summary>
    public JsonNode? ToJson()
    {
        JsonObject With(string kind, JsonObject o)
        {
            o["kind"] = kind;
            if ((Sampling ?? SpectralSampling.Default).ToJson() is { } sj) o["sampling"] = sj;
            return o;
        }
        return Kind switch
        {
            SpectrumKind.Design => null,
            SpectrumKind.Photopic => With("photopic", new JsonObject()),
            SpectrumKind.Scotopic => With("scotopic", new JsonObject()),
            SpectrumKind.Blackbody => With("blackbody", new JsonObject { ["temperature_k"] = TemperatureK }),
            SpectrumKind.Explicit => new JsonObject
            {
                ["kind"] = "explicit",
                ["wavelengths"] = new JsonArray(List.Select(w => (JsonNode)new JsonObject { ["um"] = w.Um, ["weight"] = w.Weight }).ToArray()),
            },
            _ => With("sensor", new JsonObject { ["table"] = new JsonArray(List.Select(w => (JsonNode)new JsonArray(w.Um, w.Weight)).ToArray()) }),
        };
    }

    // MARK: Validation

    /// <summary>What the engine would refuse, in words, before a run; empty when valid (energy in range is the engine's check).</summary>
    public IReadOnlyList<string> Issues
    {
        get
        {
            var s = Sampling ?? SpectralSampling.Default;
            switch (Kind)
            {
                case SpectrumKind.Photopic: return [.. s.Issues, .. RangeIssue(s, PhotopicLo, PhotopicHi, "the photopic table")];
                case SpectrumKind.Scotopic: return [.. s.Issues, .. RangeIssue(s, ScotopicLo, ScotopicHi, "the scotopic table")];
                case SpectrumKind.Blackbody: return [.. TemperatureK > 0 && double.IsFinite(TemperatureK) ? [] : new[] { "The temperature must be positive." }, .. s.Issues];
                case SpectrumKind.Explicit:
                    var e = new List<string>();
                    if (List.Count == 0) e.Add("Add at least one wavelength.");
                    if (List.Any(w => !(w.Um > 0 && double.IsFinite(w.Um)))) e.Add("Wavelengths must be positive.");
                    if (List.Any(w => !(w.Weight >= 0 && double.IsFinite(w.Weight)))) e.Add("Weights must not be negative.");
                    return e;
                case SpectrumKind.Sensor:
                    var o = new List<string>(s.Issues);
                    if (TableIssue(List) is string problem) o.Add(problem);
                    else o.AddRange(RangeIssue(s, List[0].Um, List[^1].Um, "the response table"));
                    return o;
                default: return [];
            }
        }
    }

    public static string? TableIssue(IReadOnlyList<SpectrumSample> t)
    {
        if (t.Count < 2) return "The response table needs at least two rows.";
        if (t.Any(w => !(w.Um > 0 && double.IsFinite(w.Um) && w.Weight >= 0 && double.IsFinite(w.Weight)))) return "Responses must not be negative.";
        for (var i = 1; i < t.Count; i++) if (t[i - 1].Um >= t[i].Um) return "Wavelengths must increase.";
        return null;
    }

    static IEnumerable<string> RangeIssue(SpectralSampling s, double lo, double hi, string what)
    {
        if (!(s.LowerUm > 0 && s.UpperUm > s.LowerUm)) yield break;
        if (s.LowerUm < lo - 1e-12 || s.UpperUm > hi + 1e-12)
            yield return string.Create(CultureInfo.InvariantCulture, $"The range must lie inside {what} ({lo:0.000}–{hi:0.000} µm).");
    }

    // MARK: Text

    /// <summary>A short description for summaries: "photopic", "blackbody 5500 K", "3 wavelengths", "sensor (42 rows)".</summary>
    public string Summary => Kind switch
    {
        SpectrumKind.Design => "design",
        SpectrumKind.Photopic => "photopic" + SamplesSuffix,
        SpectrumKind.Scotopic => "scotopic" + SamplesSuffix,
        SpectrumKind.Blackbody => "blackbody " + Kelvin(TemperatureK) + SamplesSuffix,
        SpectrumKind.Explicit => List.Count == 1 ? List[0].Um.ToString("G4", CultureInfo.InvariantCulture) + " µm" : $"{List.Count.ToString(CultureInfo.InvariantCulture)} wavelengths",
        _ => $"sensor ({List.Count.ToString(CultureInfo.InvariantCulture)} rows)" + SamplesSuffix,
    };

    string SamplesSuffix => (Sampling ?? SpectralSampling.Default).Samples is var n && n != SpectralSampling.Default.Samples ? " ×" + n.ToString(CultureInfo.InvariantCulture) : "";

    public static string Kelvin(double t) =>
        Math.Round(t) == t && Math.Abs(t) < 1e15 ? ((long)t).ToString(CultureInfo.InvariantCulture) + " K" : t.ToString("0.0", CultureInfo.InvariantCulture) + " K";

    // MARK: Preview

    /// <summary>
    /// The continuous response the spectrum samples, on its own range, for a
    /// preview drawing (peak 1). Photopic and scotopic use Gaussian fits
    /// (V(λ) ≈ 1.019·exp(−285.4(λ − 0.559)²), V′(λ) ≈ 0.992·exp(−321.9(λ − 0.503)²))
    /// instead of the CIE tables the engine samples: a display
    /// approximation. The blackbody is Planck's law. Null for design and explicit wavelengths.
    /// </summary>
    public IReadOnlyList<SpectrumSample>? PreviewCurve(int points = 161)
    {
        if (Sampling is not { } s || !(s.LowerUm > 0 && s.UpperUm > s.LowerUm && double.IsFinite(s.UpperUm)) || points < 2) return null;
        Func<double, double> f;
        switch (Kind)
        {
            case SpectrumKind.Photopic: f = l => 1.019 * Math.Exp(-285.4 * (l - 0.559) * (l - 0.559)); break;
            case SpectrumKind.Scotopic: f = l => 0.992 * Math.Exp(-321.9 * (l - 0.503) * (l - 0.503)); break;
            case SpectrumKind.Blackbody:
                if (!(TemperatureK > 0 && double.IsFinite(TemperatureK))) return null;
                var t = TemperatureK;
                // Planck: B_λ ∝ λ⁻⁵ / (exp(c₂/λT) − 1), c₂ = 14387.77 µm·K.
                f = l => Math.Exp(-5 * Math.Log(l) - LogExpm1(14387.768775039337 / (l * t)));
                break;
            case SpectrumKind.Sensor:
                if (TableIssue(List) is not null) return null;
                var table = List;
                f = l => Interpolate(table, l);
                break;
            default: return null;
        }
        var xs = Enumerable.Range(0, points).Select(i => s.LowerUm + (s.UpperUm - s.LowerUm) * i / (points - 1)).ToArray();
        var ys = xs.Select(f).ToArray();
        var peak = ys.Where(double.IsFinite).DefaultIfEmpty(0).Max();
        if (!(peak > 0)) return null;
        return xs.Select((x, i) => new SpectrumSample(x, double.IsFinite(ys[i]) ? ys[i] / peak : 0)).ToArray();
    }

    /// <summary>
    /// The weighted wavelengths a run would use, approximated locally (the
    /// engine's sampling on <see cref="PreviewCurve"/>). Explicit wavelengths
    /// are exact; the design returns null (the caller knows the design's wavelengths).
    /// </summary>
    public IReadOnlyList<SpectrumSample>? PreviewSamples()
    {
        switch (Kind)
        {
            case SpectrumKind.Design: return null;
            case SpectrumKind.Explicit: return List;
            default:
                if (Sampling is not { } s || s.Issues.Count > 0 || PreviewCurve(801) is not { } curve) return null;
                return Sample(curve, s);
        }
    }

    /// <summary>The index the engine makes primary: nearest the energy-weighted mean wavelength, the first on ties.</summary>
    public static int? PrimaryIndex(IReadOnlyList<SpectrumSample> samples)
    {
        if (samples.Count == 0) return null;
        var total = samples.Sum(w => w.Weight);
        var mean = total > 0 ? samples.Sum(w => w.Um * w.Weight) / total : samples[0].Um;
        var best = 0;
        for (var i = 1; i < samples.Count; i++) if (Math.Abs(samples[i].Um - mean) < Math.Abs(samples[best].Um - mean)) best = i;
        return best;
    }

    /// <summary>The run's weighted wavelengths as the engine resolves them, from a <c>system_summary</c> result (<c>wavelengths_um</c>, <c>wavelength_weights</c>).</summary>
    public static IReadOnlyList<SpectrumSample>? ResolvedSamples(JsonNode? json)
    {
        if (Json.NfArray(Json.At(json, "wavelengths_um")) is not { Length: > 0 } um) return null;
        var w = Json.NfArray(Json.At(json, "wavelength_weights")) ?? Enumerable.Repeat(1.0, um.Length).ToArray();
        return w.Length == um.Length ? um.Select((x, i) => new SpectrumSample(x, w[i])).ToArray() : null;
    }

    static double LogExpm1(double x) => x > 50 ? x : Math.Log(Math.Exp(x) - 1);

    static double Interpolate(IReadOnlyList<SpectrumSample> t, double x)
    {
        if (t.Count == 0 || x < t[0].Um || x > t[^1].Um) return 0;
        var hi = 0;
        while (hi < t.Count - 1 && t[hi].Um < x) hi++;
        hi = Math.Clamp(hi, 1, t.Count - 1);
        var (a, b) = (t[hi - 1], t[hi]);
        return a.Weight + (b.Weight - a.Weight) * (x - a.Um) / (b.Um - a.Um);
    }

    /// <summary>The engine's two spacings on a finely tabulated curve (trapezoids).</summary>
    static IReadOnlyList<SpectrumSample>? Sample(IReadOnlyList<SpectrumSample> curve, SpectralSampling s)
    {
        if (curve.Count < 2) return null;
        // Cumulative energy and first moment at every node.
        var energy = new double[curve.Count];
        var moment = new double[curve.Count];
        for (var i = 1; i < curve.Count; i++)
        {
            var (p, q) = (curve[i - 1], curve[i]);
            var e = (q.Um - p.Um) * (p.Weight + q.Weight) / 2;
            energy[i] = energy[i - 1] + e;
            moment[i] = moment[i - 1] + e * (p.Um + q.Um) / 2;
        }
        var total = energy[^1];
        if (!(total > 0)) return null;
        // Energy and moment from the start to x (linear inside a step).
        (double E, double M) Cumulative(double x)
        {
            var i = 0;
            while (i < curve.Count - 1 && curve[i].Um < x) i++;
            i = Math.Clamp(i, 1, curve.Count - 1);
            var (a, b) = (curve[i - 1], curve[i]);
            var f = (x - a.Um) / (b.Um - a.Um);
            var vx = a.Weight + (b.Weight - a.Weight) * f;
            var e = (x - a.Um) * (a.Weight + vx) / 2;
            return (energy[i - 1] + e, moment[i - 1] + e * (a.Um + x) / 2);
        }
        var n = s.Samples;
        var result = new List<SpectrumSample>(n);
        if (s.Spacing == SamplingSpacing.Uniform)
        {
            var d = (s.UpperUm - s.LowerUm) / (Math.Max(n, 2) - 1);
            for (var i = 0; i < n; i++)
            {
                var um = n == 1 ? (s.LowerUm + s.UpperUm) / 2 : s.LowerUm + d * i;
                var half = n == 1 ? (s.UpperUm - s.LowerUm) / 2 : d / 2;
                result.Add(new(um, Cumulative(Math.Min(um + half, s.UpperUm)).E - Cumulative(Math.Max(um - half, s.LowerUm)).E));
            }
            return result;
        }
        var left = s.LowerUm;
        for (var i = 0; i < n; i++)
        {
            var target = total * (i + 1) / n;
            // Bisection on the cumulative energy.
            double lo = left, hi = s.UpperUm;
            if (i + 1 < n)
                for (var k = 0; k < 60; k++)
                {
                    var mid = (lo + hi) / 2;
                    if (Cumulative(mid).E < target) lo = mid; else hi = mid;
                }
            var right = i + 1 == n ? s.UpperUm : hi;
            var (e0, m0) = Cumulative(left);
            var (e1, m1) = Cumulative(right);
            var e = e1 - e0;
            result.Add(new(e > 0 ? (m1 - m0) / e : (left + right) / 2, e));
            left = right;
        }
        return result;
    }
}
