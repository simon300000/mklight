using Microsoft.Windows.ApplicationModel.Resources;
using MKLight.Model;

namespace MKLight.App.Support;

/// <summary>
/// Applies the UI language at launch, before any XAML loads (docs/WINDOWS.md "Languages"): the language
/// <see cref="Languages.Resolve"/> picks becomes MRT Core's <c>PrimaryLanguageOverride</c> (x:Uid text and the
/// XAML language, which picks the CJK font fallback), the lookup behind <see cref="Loc"/>, and the engine text
/// catalog (<see cref="EngineText.Current"/>). The app is unpackaged, so the override is set on every launch
/// from the saved choice rather than persisted by Windows; a new choice applies on the next launch.
/// </summary>
public static class UiLanguage
{
    /// <summary>The language this run shows (<see cref="Languages.English"/> or <see cref="Languages.SimplifiedChinese"/>).</summary>
    public static string Current { get; private set; } = Languages.English;

    public static void Apply(IReadOnlyList<string> args)
    {
        string? choice = null;
        for (var i = 0; i < args.Count - 1; i++)
            if (args[i] == "--language") choice = args[i + 1];
        try { Current = Languages.Resolve(choice ?? AppSettings.Language, SystemLanguages()); }
        catch (ArgumentException) { Current = Languages.Resolve(AppSettings.Language, SystemLanguages()); } // LaunchOptions reports it
        try { Microsoft.Windows.Globalization.ApplicationLanguages.PrimaryLanguageOverride = Current; }
        catch (Exception e) { App.Log($"language override: {e.Message}"); }
        try
        {
            var manager = new ResourceManager();
            var context = manager.CreateResourceContext();
            context.QualifierValues["Language"] = Current;
            var map = manager.MainResourceMap.GetSubtree("Resources");
            Loc.Provider = key => map.TryGetValue(key, context)?.ValueAsString;
        }
        catch (Exception e) { App.Log($"resources: {e.Message}"); }
        EngineText.Current = Current == Languages.English ? EngineText.None : EngineText.Load(EngineText.PathFor(AppContext.BaseDirectory, Current));
    }

    static IEnumerable<string> SystemLanguages()
    {
        try { return global::Windows.System.UserProfile.GlobalizationPreferences.Languages; }
        catch (Exception) { return [System.Globalization.CultureInfo.CurrentUICulture.Name]; }
    }
}
