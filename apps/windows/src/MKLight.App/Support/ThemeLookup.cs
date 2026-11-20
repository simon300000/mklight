using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;

namespace MKLight.App.Support;

/// <summary>
/// Theme resources looked up in code for an element's own theme. The app
/// lets each window choose light or dark at run time (RequestedTheme on its
/// root), and <c>Application.Current.Resources[key]</c> answers for the app's
/// theme instead, so code that picks a brush asks here.
/// </summary>
public static class ThemeLookup
{
    public static Brush Brush(FrameworkElement element, string key) => (Brush)Resource(element.ActualTheme, key);

    /// <summary>
    /// Runs <paramref name="paint"/> now, once the element is loaded (its theme is the window's then) and
    /// whenever its theme changes: for brushes picked in code.
    /// </summary>
    public static void Follow(FrameworkElement element, Action paint)
    {
        paint();
        element.Loaded += (_, _) => paint();
        element.ActualThemeChanged += (_, _) => paint();
    }

    static readonly Lazy<bool> HighContrast = new(() => new global::Windows.UI.ViewManagement.AccessibilitySettings().HighContrast);

    public static object Resource(ElementTheme theme, string key)
    {
        // A high-contrast theme takes its own dictionary (system colours), whatever the light/dark choice.
        var name = HighContrast.Value ? "HighContrast" : theme == ElementTheme.Dark ? "Dark" : "Light";
        return Find(Application.Current.Resources, name, key)
            ?? (theme == ElementTheme.Dark ? Find(Application.Current.Resources, "Default", key) : null)
            ?? Application.Current.Resources[key];
    }

    static object? Find(ResourceDictionary d, string theme, string key)
    {
        if (d.ThemeDictionaries.TryGetValue(theme, out var t) && t is ResourceDictionary td && td.TryGetValue(key, out var v)) return v;
        foreach (var m in d.MergedDictionaries)
            if (Find(m, theme, key) is { } found) return found;
        return null;
    }
}
