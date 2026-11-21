using Microsoft.Windows.ApplicationModel.Resources;
using MKLight.Model;

namespace MKLight.App.Support;

/// <summary>
/// The UI language (docs/WINDOWS.md "Languages"): the language <see cref="Languages.Resolve"/> picks from the
/// command line (<c>--language</c>), the saved choice and the system's languages. It is resolved once, on first
/// use: by <see cref="Loc"/> (which may be in a <c>[ModuleInitializer]</c> registration, before <c>App</c>
/// starts) or by <see cref="Apply"/> at launch, before any XAML loads. It becomes MRT Core's
/// <c>PrimaryLanguageOverride</c> (x:Uid text and the XAML language, which picks the CJK font fallback), the
/// lookup behind <see cref="Loc"/>, and the engine text catalog (<see cref="EngineText.Current"/>). The app is
/// unpackaged, so the override is set on every launch from the saved choice rather than kept by Windows; a new
/// choice applies on the next launch.
/// </summary>
public static class UiLanguage
{
    static string? _current;
    static Func<string, string?>? _lookup;

    /// <summary>The language this run shows (<see cref="Languages.English"/> or <see cref="Languages.SimplifiedChinese"/>).</summary>
    public static string Current => _current ??= Resolve(Environment.GetCommandLineArgs());

    /// <summary>MRT Core's lookup for <see cref="Current"/> (null when the resources cannot be read: English).</summary>
    internal static Func<string, string?>? Lookup()
    {
        if (_lookup is not null) return _lookup;
        try
        {
            var manager = new ResourceManager();
            var context = manager.CreateResourceContext();
            context.QualifierValues["Language"] = Current;
            var map = manager.MainResourceMap.GetSubtree("Resources");
            return _lookup = key => map.TryGetValue(key, context)?.ValueAsString;
        }
        catch (Exception) { return null; }
    }

    /// <summary>At launch, before any XAML loads: the language override and the engine text catalog.</summary>
    public static void Apply()
    {
        try { Microsoft.Windows.Globalization.ApplicationLanguages.PrimaryLanguageOverride = Current; }
        catch (Exception e) { App.Log($"language override: {e.Message}"); }
        // A lookup that failed on first use (in a module initializer) gets another chance here.
        if (Lookup() is { } lookup) { if (Loc.Provider is null) Loc.Provider = lookup; }
        else App.Log("resources: the UI text resources could not be read; showing English");
        ModelText.Lookup = key => Loc.Provider?.Invoke(key);
        EngineText.Current = Current == Languages.English ? EngineText.None : EngineText.Load(EngineText.PathFor(AppContext.BaseDirectory, Current));
    }

    /// <summary>
    /// The language for these command-line arguments: <c>--language</c>, else the saved choice in the support
    /// folder this run uses (<c>--support-dir</c> is read here too, as it is applied later), else the system's.
    /// An invalid <c>--language</c> falls back to the saved choice (LaunchOptions reports it).
    /// </summary>
    static string Resolve(IReadOnlyList<string> args)
    {
        string? choice = null;
        for (var i = 0; i < args.Count - 1; i++)
        {
            if (args[i] == "--language") choice = args[i + 1];
            if (args[i] == "--support-dir" && Environment.GetEnvironmentVariable("MKLIGHT_SUPPORT_DIRECTORY") is null or "")
                Environment.SetEnvironmentVariable("MKLIGHT_SUPPORT_DIRECTORY", Path.GetFullPath(args[i + 1]));
        }
        try { return Languages.Resolve(choice ?? AppSettings.Language, SystemLanguages()); }
        catch (ArgumentException) { return Languages.Resolve(AppSettings.Language, SystemLanguages()); }
    }

    internal static IEnumerable<string> SystemLanguages()
    {
        try { return global::Windows.System.UserProfile.GlobalizationPreferences.Languages; }
        catch (Exception) { return [System.Globalization.CultureInfo.CurrentUICulture.Name]; }
    }
}

/// <summary>The app's half of <see cref="Loc"/>: MRT Core for the UI language, made on first use.</summary>
public static partial class Loc
{
    static partial void CreateProvider(ref Func<string, string?>? provider) => provider = UiLanguage.Lookup();
}
