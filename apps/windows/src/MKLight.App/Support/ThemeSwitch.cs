using System.Runtime.CompilerServices;
using Microsoft.UI.Dispatching;
using Microsoft.UI.Xaml;
using Windows.UI.ViewManagement;

namespace MKLight.App.Support;

/// <summary>
/// The appearance (System, Light or Dark) of every window, switched live.
/// <para>
/// A window root given a theme other than the app's before it is loaded
/// keeps the app's theme for the theme resources of styles outside the app's
/// own dictionaries (a control's Resources, dictionaries loaded with
/// XamlReader: the lens table's dimmed numbers, the inspector's variable
/// boxes), even after later switches; set once the root is loaded, the same
/// theme resolves everywhere and follows every later switch. The app's own
/// theme (<c>Application.RequestedTheme</c>) can only be set at launch. So a
/// window joins with <see cref="Attach"/>: its root follows the appearance
/// from its Loaded on, and System is the system's light or dark, followed as
/// it changes.
/// </para>
/// </summary>
public static class ThemeSwitch
{
    sealed record Entry(Window Window, FrameworkElement Root, Action? Changed);

    static readonly List<Entry> _windows = [];
    static UISettings? _system;

    /// <summary>The light or dark the windows show for the chosen appearance.</summary>
    public static ElementTheme Effective => AppSettings.Theme is ElementTheme.Default ? SystemTheme() : AppSettings.Theme;

    /// <summary>
    /// The window's <paramref name="root"/> follows the appearance, the caption buttons its theme;
    /// <paramref name="changed"/> runs whenever the root's theme changed (e.g. to refresh a menu).
    /// </summary>
    public static void Attach(Window window, FrameworkElement root, Action? changed = null)
    {
        var entry = new Entry(window, root, changed);
        _windows.Add(entry);
        window.Closed += (_, _) => _windows.Remove(entry);
        root.ActualThemeChanged += (_, _) => Sync(entry);
        if (root.IsLoaded) Apply(entry);
        else root.Loaded += (_, _) => Apply(entry);
        Sync(entry);
        FollowSystem();
    }

    /// <summary>Saves the appearance and shows it in every window.</summary>
    public static void Set(ElementTheme theme)
    {
        AppSettings.Theme = theme;
        ApplyAll();
    }

    static void ApplyAll()
    {
        foreach (var e in _windows.ToList()) Apply(e);
    }

    static void Apply(Entry e)
    {
        if (!e.Root.IsLoaded) return;
        e.Root.RequestedTheme = Effective;
        Sync(e);
    }

    /// <summary>The caption buttons in the content's theme, then the window's own refresh.</summary>
    static void Sync(Entry e)
    {
        var bar = e.Window.AppWindow.TitleBar;
        var dark = (e.Root.IsLoaded ? e.Root.ActualTheme : Effective) == ElementTheme.Dark;
        bar.ButtonForegroundColor = dark ? Microsoft.UI.Colors.White : Microsoft.UI.Colors.Black;
        bar.ButtonHoverForegroundColor = bar.ButtonForegroundColor;
        e.Changed?.Invoke();
    }

    static ElementTheme SystemTheme()
    {
        var bg = (_system ??= new UISettings()).GetColorValue(UIColorType.Background);
        return bg.R + bg.G + bg.B < 384 ? ElementTheme.Dark : ElementTheme.Light;
    }

    static bool _following;

    /// <summary>System follows the system's light or dark as it changes (the event comes off the UI thread).</summary>
    static void FollowSystem()
    {
        if (_following) return;
        _following = true;
        var queue = DispatcherQueue.GetForCurrentThread();
        _system ??= new UISettings();
        _system.ColorValuesChanged += (_, _) => queue.TryEnqueue(() => { if (AppSettings.Theme is ElementTheme.Default) ApplyAll(); });
    }

    [ModuleInitializer]
    internal static void Register()
    {
        // --do theme=light|dark|system: switch the appearance during a scripted run (not saved).
        AppHooks.LaunchActions["theme"] = (_, arg) =>
        {
            AppSettings.Override = arg switch { "light" => ElementTheme.Light, "dark" => ElementTheme.Dark, _ => ElementTheme.Default };
            ApplyAll();
            return Task.CompletedTask;
        };
        // --do welcome[=no-recent]: opens the welcome window beside the design and captures it.
        AppHooks.LaunchActions["welcome"] = (_, arg) =>
        {
            App.ShowWelcome();
            var welcome = App.Windows.OfType<MKLight.App.Windows.WelcomeWindow>().FirstOrDefault();
            if (arg == "no-recent") welcome?.HideRecent();
            AppHooks.ScreenshotTarget = welcome;
            return Task.CompletedTask;
        };
    }
}
