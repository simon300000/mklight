using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary><c>validate</c>: the design's issues, errors first, then warnings, then notes (each in the engine's order).</summary>
public sealed class ValidationData
{
    public enum Severity { Error, Warning, Info }

    /// <param name="Surface">The surface the issue concerns, if any.</param>
    /// <param name="Field">The field (0-based) a field or chief-ray issue names in its message, if any.</param>
    public sealed record Issue(Severity Severity, string Code, string Message, int? Surface, int? Field);

    public required IReadOnlyList<Issue> Issues { get; init; }

    public int Count(Severity s) => Issues.Count(i => i.Severity == s);

    public static ValidationData? Read(JsonNode? j)
    {
        if (Json.Arr(Json.At(j, "issues")) is not { } list) return null;
        var issues = list.Select(i =>
        {
            var message = Json.Str(Json.At(i, "message")) ?? "";
            var severity = Json.Str(Json.At(i, "severity")) switch { "error" => Severity.Error, "info" => Severity.Info, _ => Severity.Warning };
            return new Issue(severity, Json.Str(Json.At(i, "code")) ?? "", message, Json.Int(Json.At(i, "surface")), FieldIndex(message));
        });
        return new ValidationData { Issues = issues.OrderBy(i => i.Severity).ToList() };
    }

    /// <summary>"field 2" or "field index 2" in a message: 2 (the system's field index, as written).</summary>
    public static int? FieldIndex(string message)
    {
        var words = System.Text.RegularExpressions.Regex.Split(message.ToLowerInvariant(), @"[^\p{L}\p{N}]+").Where(w => w.Length > 0).ToList();
        for (var k = 0; k < words.Count; k++)
        {
            if (words[k] != "field") continue;
            foreach (var next in words.Skip(k + 1).Take(2))
                if (int.TryParse(next, System.Globalization.NumberStyles.None, System.Globalization.CultureInfo.InvariantCulture, out var i)) return i;
        }
        return null;
    }

    /// <summary>"2 errors, 1 warning" (or "No issues").</summary>
    public string Summary()
    {
        static string N(int n, string one) => n == 1 ? "1 " + one : $"{n} {one}s";
        var parts = new List<string>();
        if (Count(Severity.Error) is > 0 and var e) parts.Add(N(e, "error"));
        if (Count(Severity.Warning) is > 0 and var w) parts.Add(N(w, "warning"));
        if (Count(Severity.Info) is > 0 and var i) parts.Add(N(i, "note"));
        return parts.Count == 0 ? "No issues" : string.Join(", ", parts);
    }
}
