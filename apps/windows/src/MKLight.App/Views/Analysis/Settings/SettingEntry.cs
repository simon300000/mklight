using System.Globalization;
using System.Text.Json.Nodes;
using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// A typed setting: a new value, or <see cref="Reset"/> to take the
/// schema default again (the key is removed, or put back to its default
/// inside an object).
/// </summary>
public sealed record Edit(JsonNode? Value, bool Remove = false)
{
    public static readonly Edit Reset = new(null, true);
}

/// <summary>Number text in the app's one format: decimal point, no grouping, invariant culture.</summary>
public static class NumberText
{
    static readonly CultureInfo C = CultureInfo.InvariantCulture;

    /// <summary>
    /// A number, accepting "∞"/"inf", a Unicode minus sign, a decimal comma
    /// ("1,5", as typed in regions that write one) and surrounding space. NaN is rejected.
    /// The app never groups digits, so a lone comma is always the decimal comma: "1,000"
    /// is 1, as in the lens table, the model's parser and the Mac app.
    /// </summary>
    public static double? Parse(string text)
    {
        var t = text.Trim().Replace('−', '-').Replace("∞", "inf");
        if (!t.Contains('.') && t.Count(c => c == ',') == 1) t = t.Replace(',', '.');
        if (t.Length == 0) return null;
        switch (t.ToLowerInvariant())
        {
            case "inf" or "+inf" or "infinity" or "+infinity": return double.PositiveInfinity;
            case "-inf" or "-infinity": return double.NegativeInfinity;
        }
        return double.TryParse(t, NumberStyles.Float, C, out var v) && !double.IsNaN(v) ? v : null;
    }

    /// <summary>Compact text (<see cref="Fmt.Short"/>, with an ASCII minus so it reads back).</summary>
    public static string Short(double v) => Fmt.Short(v).Replace('−', '-').Replace("∞", "inf");

    /// <summary>Text that reads back as exactly <paramref name="v"/>: the compact form when it does, else every digit.</summary>
    public static string Exact(double v)
    {
        var s = Short(v);
        return Parse(s) == v ? s : v.ToString("R", C);
    }

    public static string Integer(double v) =>
        double.IsFinite(v) ? ((long)Math.Clamp(Math.Truncate(v), long.MinValue, long.MaxValue)).ToString(C) : "";
}

/// <summary>
/// The text of each kind of setting and how typed text becomes a value
/// (docs/APP.md "One text-field rule": only edited text is committed, an
/// equal value is not). Pure, so the mapping is unit-tested.
/// </summary>
public static class SettingEntry
{
    // MARK: Labels

    /// <summary>The row label: the title, or the name in words without its unit suffix ("Max Radius").</summary>
    public static string Label(SettingsField f)
    {
        var name = f.Name;
        foreach (var suffix in new[] { "_mm", "_um", "_deg" })
            if (name.EndsWith(suffix, StringComparison.Ordinal)) { name = name[..^suffix.Length]; break; }
        var t = f.Title == f.Name ? Words(name) : f.Title;
        return t.Length > 22 ? t[..20] + "…" : t;
    }

    static string Words(string snake) =>
        string.Join(' ', snake.Split('_', StringSplitOptions.RemoveEmptyEntries).Select(w => char.ToUpperInvariant(w[0]) + w[1..].ToLowerInvariant()));

    /// <summary>An enum value's title: "FFT", "Primary Chief", "N-SK16" (names with capitals as written).</summary>
    public static string Title(JsonNode? v)
    {
        if (Json.Str(v) is not string s) return Json.Display(v);
        if (s.Length <= 3) return s.ToUpperInvariant();
        if (s.Any(char.IsUpper)) return s;
        return Words(s);
    }

    // MARK: Numbers

    /// <summary>The text of a number setting; empty for an automatic one.</summary>
    public static string NumberDisplay(JsonNode? v, bool integer, AutoSentinel? auto, bool exact)
    {
        if (auto is AutoSentinel a && AutoSettings.IsAuto(v, a)) return "";
        if (Json.Number(v) is not double d) return "";
        return integer ? NumberText.Integer(d) : exact ? NumberText.Exact(d) : NumberText.Short(d);
    }

    /// <summary>
    /// A typed number: nothing, "auto", or the 0 of a setting that is
    /// automatic at 0 make it automatic again (or take the default); else the
    /// number, held to the setting's range. Null when the text is not a number.
    /// </summary>
    public static Edit? ParseNumber(string text, SettingsField f, AutoSentinel? auto)
    {
        var t = text.Trim();
        var typed = NumberText.Parse(t);
        if (t.Length == 0 || (auto is not null && t.StartsWith("auto", StringComparison.OrdinalIgnoreCase)) || (auto == AutoSentinel.Zero && typed == 0))
        {
            return auto switch
            {
                AutoSentinel.Zero => new Edit(0),
                AutoSentinel.Null => new Edit(null),
                _ => f.Nullable ? new Edit(null) : Edit.Reset,
            };
        }
        if (typed is not double d || !double.IsFinite(d)) return null;
        if (f.Minimum is double lo)
        {
            if (f.MinimumExclusive && d <= lo) return null;
            d = Math.Max(d, lo);
        }
        if (f.Maximum is double hi)
        {
            if (f.MaximumExclusive && d >= hi) return null;
            d = Math.Min(d, hi);
        }
        return f.IsInteger ? new Edit((long)Math.Truncate(d)) : new Edit(d);
    }

    // MARK: Text

    /// <summary>Typed text: empty restores the default, or null when allowed.</summary>
    public static Edit ParseText(string text, SettingsField f)
    {
        var t = text.Trim();
        if (t.Length > 0) return new Edit(t);
        return f.Nullable ? new Edit(null) : Edit.Reset;
    }

    // MARK: Number lists

    /// <summary>"10, 20, 30": compact, or exact (a focused field).</summary>
    public static string ListDisplay(JsonNode? v, bool integer, bool exact)
    {
        var values = (Json.Arr(v) ?? []).Select(Json.Number).OfType<double>();
        return string.Join(", ", values.Select(x => integer ? NumberText.Integer(x) : exact ? NumberText.Exact(x) : NumberText.Short(x)));
    }

    /// <summary>The finite numbers of "10, 20 30" (commas, semicolons or spaces) in the order typed; null when a part is not one.</summary>
    public static double[]? Numbers(string text)
    {
        var parts = text.Split([',', ';', ' '], StringSplitOptions.RemoveEmptyEntries);
        var v = parts.Select(NumberText.Parse).ToList();
        return v.All(x => x is double d && double.IsFinite(d)) ? v.Select(x => x!.Value).ToArray() : null;
    }

    /// <summary>
    /// A typed list: empty (or "auto" for an automatic list) is the empty list
    /// (null when allowed); frequencies of an automatic list are positive,
    /// sorted and unique.
    /// </summary>
    public static Edit? ParseList(string text, SettingsField f, bool integer, AutoSentinel? auto)
    {
        var t = text.Trim();
        if (t.Length == 0 || (auto is not null && t.StartsWith("auto", StringComparison.OrdinalIgnoreCase)))
            return new Edit(f.Nullable && auto is null ? null : new JsonArray());
        if (Numbers(t) is not double[] v) return null;
        if (auto == AutoSentinel.Empty)
        {
            if (v.Any(x => x <= 0)) return null;
            v = [.. v.Distinct().Order()];
        }
        return new Edit(new JsonArray(v.Select(x => integer ? (JsonNode)(long)Math.Truncate(x) : x).ToArray()));
    }

    // MARK: Pairs

    /// <summary>
    /// The bounds of both numbers: the schema's minimum and maximum (of the
    /// setting or its items), else those the description states ("1..101").
    /// </summary>
    public static (double Lo, double Hi) PairBounds(SettingsField f)
    {
        var lo = f.Minimum ?? Json.Number(Json.At(f.Items, "minimum"));
        var hi = f.Maximum ?? Json.Number(Json.At(f.Items, "maximum"));
        if ((lo is not null || hi is not null) && (lo ?? double.NegativeInfinity) <= (hi ?? double.PositiveInfinity))
            return (lo ?? double.NegativeInfinity, hi ?? double.PositiveInfinity);
        var m = System.Text.RegularExpressions.Regex.Match(f.Description, @"(\d+)\.\.(\d+)");
        if (m.Success && double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture) is var a
            && double.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture) is var b && a <= b)
            return (a, b);
        return (double.NegativeInfinity, double.PositiveInfinity);
    }

    /// <summary>One number of a pair, held to the bounds; null when the text is not a number.</summary>
    public static double? ParsePairPart(string text, bool integer, (double Lo, double Hi) bounds)
    {
        if (NumberText.Parse(text) is not double d || !double.IsFinite(d)) return null;
        d = Math.Clamp(d, bounds.Lo, bounds.Hi);
        return integer ? Math.Truncate(d) : d;
    }

    /// <summary>
    /// The pair from its two boxes' numbers (null: an empty box): both empty
    /// is null for a nullable pair; half a pair is not a value (null result).
    /// </summary>
    public static Edit? Pair(double? x, double? y, bool integer, bool nullable)
    {
        if (x is null && y is null) return nullable ? new Edit(null) : null;
        if (x is null || y is null) return null;
        JsonNode N(double v) => integer ? (long)v : v;
        return new Edit(new JsonArray(N(x.Value), N(y.Value)));
    }

    /// <summary>Text between the two boxes of a pair: "Hx … Hy" field points, "to" ranges, "×" otherwise.</summary>
    public static (string? Before, string Between) PairWords(SettingsField f)
    {
        if (f.Name.EndsWith("center", StringComparison.Ordinal) || f.Name.EndsWith("centre", StringComparison.Ordinal)) return ("Hx", "Hy");
        if (f.Name.Contains("range") || f.Description.Contains("[min, max]", StringComparison.OrdinalIgnoreCase)) return (null, "to");
        return (null, "×");
    }

    // MARK: Menus

    /// <summary>The short title of an option ("2" of "2: 14°").</summary>
    static string Short(string title) => title.Split(':')[0];

    /// <summary>A <c>{"type": …}</c> selection's title: "All", "Primary", "2", "3 selected".</summary>
    public static string SelectionTitle(JsonNode? v, IReadOnlyList<(int Index, string Title)> options)
    {
        switch (Json.Str(Json.At(v, "type")))
        {
            case "primary": return "Primary";
            case "index":
                var i = Json.Int(Json.At(v, "index")) ?? 0;
                return options.Where(o => o.Index == i).Select(o => Short(o.Title)).FirstOrDefault() ?? (i + 1).ToString(CultureInfo.InvariantCulture);
            case "list": return $"{(Json.Arr(Json.At(v, "indices"))?.Count ?? 0).ToString(CultureInfo.InvariantCulture)} selected";
            default: return "All";
        }
    }

    public static JsonObject Selection(string type, int? index = null, IEnumerable<int>? indices = null)
    {
        var o = new JsonObject { ["type"] = type };
        if (index is int i) o["index"] = i;
        if (indices is not null) o["indices"] = new JsonArray(indices.Select(k => (JsonNode)k).ToArray());
        return o;
    }

    /// <summary>An index setting's title: the option's short title, or the null title ("Primary", "Auto").</summary>
    public static string IndexTitle(JsonNode? v, IReadOnlyList<(int Index, string Title)> options, string? nullTitle)
    {
        if (Json.Int(v) is not int i) return nullTitle ?? "—";
        return options.Where(o => o.Index == i).Select(o => Short(o.Title)).FirstOrDefault() ?? i.ToString(CultureInfo.InvariantCulture);
    }

    /// <summary>An index list's title: "All" (empty or every option), one option's short title, or "2 of 5".</summary>
    public static string IndexListTitle(JsonNode? v, IReadOnlyList<(int Index, string Title)> options)
    {
        var s = (Json.Arr(v) ?? []).Select(Json.Int).OfType<int>().ToHashSet();
        if (s.Count == 0 || (s.Count == options.Count && options.Count > 0)) return "All";
        if (s.Count == 1 && options.FirstOrDefault(o => o.Index == s.First()) is { Title: not null } one) return Short(one.Title);
        return FormattableString.Invariant($"{s.Count} of {options.Count}");
    }

    /// <summary>The index list with <paramref name="index"/> switched on or off, sorted.</summary>
    public static JsonArray Toggle(JsonNode? v, int index, bool on)
    {
        var s = (Json.Arr(v) ?? []).Select(Json.Int).OfType<int>().ToHashSet();
        if (on) s.Add(index); else s.Remove(index);
        return new JsonArray(s.Order().Select(k => (JsonNode)k).ToArray());
    }

    // MARK: Through focus (spot diagram)

    /// <summary>"±0.08 mm · 3 planes", "−0.1 to 0.2 mm · 5 planes", or the range the last run used when automatic.</summary>
    public static string ThroughFocusSummary(JsonNode? v, string? used)
    {
        var steps = Json.Int(Json.At(v, "steps")) ?? 3;
        var planes = steps == 1 ? "1 plane" : FormattableString.Invariant($"{steps} planes");
        if (Json.Number(Json.At(v, "min")) is double a && Json.Number(Json.At(v, "max")) is double b)
        {
            if (Math.Abs(a + b) <= 1e-12 * Math.Max(1, Math.Abs(b))) return $"±{Fmt.Short(b)} mm · {planes}";
            return $"{Fmt.Short(a)} to {Fmt.Short(b)} mm · {planes}";
        }
        return used is not null ? $"{used} mm · {planes}" : $"Auto range · {planes}";
    }

    /// <summary>The through-focus object with new bounds or steps.</summary>
    public static JsonObject ThroughFocus(double? min, double? max, int steps) => new()
    {
        ["min"] = min is double a ? a : null,
        ["max"] = max is double b ? b : null,
        ["steps"] = steps,
    };
}
