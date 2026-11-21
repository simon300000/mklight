using System.Globalization;

namespace MKLight.Model;

/// <summary>
/// UI text the model builds (parameter labels such as <c>Radius 3</c>): the app's resources look the
/// <c>Model_*</c> keys up through <see cref="Lookup"/> (set to the app's <c>Loc</c> at launch); without it, or for
/// a key it lacks, the English format given at the call. Numbers keep invariant formatting.
/// </summary>
public static class ModelText
{
    public static Func<string, string?>? Lookup { get; set; }

    public static string Format(string key, string english, params object?[] args) =>
        string.Format(CultureInfo.InvariantCulture, Lookup?.Invoke(key) is { Length: > 0 } t ? t : english, args);
}
