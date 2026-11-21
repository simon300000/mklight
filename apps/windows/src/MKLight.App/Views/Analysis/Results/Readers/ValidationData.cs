using System.Text.Json.Nodes;
using MKLight.App.Support;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary><c>validate</c>: the design's issues, errors first, then warnings, then notes (each in the engine's order).</summary>
public sealed class ValidationData
{
    public enum Severity { Error, Warning, Info }

    /// <param name="Surface">The surface the issue concerns, if any.</param>
    /// <param name="Field">The field (0-based) a field or chief-ray issue names in its message, if any.</param>
    /// <param name="Args">The message's values by placeholder name, when the engine gives them (<c>args</c>).</param>
    public sealed record Issue(Severity Severity, string Code, string Message, int? Surface, int? Field,
        IReadOnlyDictionary<string, string>? Args = null)
    {
        /// <summary>The message in the UI language: the engine text catalog's template for the code, else the English.</summary>
        public string Text => MKLight.Model.EngineText.Diagnostic(Code, Message, Args);
    }

    public required IReadOnlyList<Issue> Issues { get; init; }

    public int Count(Severity s) => Issues.Count(i => i.Severity == s);

    public static ValidationData? Read(JsonNode? j)
    {
        if (Json.Arr(Json.At(j, "issues")) is not { } list) return null;
        var issues = list.Select(i =>
        {
            var message = Json.Str(Json.At(i, "message")) ?? "";
            var severity = Json.Str(Json.At(i, "severity")) switch { "error" => Severity.Error, "info" => Severity.Info, _ => Severity.Warning };
            var args = Json.At(i, "args") is JsonObject a
                ? a.Where(p => p.Value is JsonValue).ToDictionary(p => p.Key, p => p.Value!.ToString(), StringComparer.Ordinal)
                : null;
            return new Issue(severity, Json.Str(Json.At(i, "code")) ?? "", message, Json.Int(Json.At(i, "surface")), FieldIndex(message), args);
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
        static string N(int n, string one, string many) => n == 1 ? Loc.S(one) : Loc.F(many, n);
        var parts = new List<string>();
        if (Count(Severity.Error) is > 0 and var e) parts.Add(N(e, "Results_Validation_Errors_One", "Results_Validation_Errors_Many"));
        if (Count(Severity.Warning) is > 0 and var w) parts.Add(N(w, "Results_Validation_Warnings_One", "Results_Validation_Warnings_Many"));
        if (Count(Severity.Info) is > 0 and var i) parts.Add(N(i, "Results_Validation_Notes_One", "Results_Validation_Notes_Many"));
        return parts.Count == 0 ? Loc.S("Results_Validation_NoIssues") : string.Join(Loc.S("Results_Validation_ListSeparator"), parts);
    }
}
