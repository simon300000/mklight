using Microsoft.UI.Xaml;
using MKLight.App.Windows;
using Windows.System;

namespace MKLight.App.Support;

/// <summary>
/// A command a feature adds to the design window's menu bar, with an optional shortcut. <paramref name="Menu"/>
/// names the top-level menu by its English name (File, Edit, View, Lens, Design, Analysis or Help; null is File);
/// the command goes above that menu's final group, in <paramref name="Order"/>.
/// </summary>
public sealed record WindowCommand(string Text, string Glyph, Action<DesignWindow> Invoke,
    VirtualKey? Key = null, VirtualKeyModifiers Modifiers = VirtualKeyModifiers.None, int Order = 100, string? Menu = null);

/// <summary>
/// Extension points so features add launch actions, commands and window
/// behaviour from their own files (registered with <c>[ModuleInitializer]</c>),
/// without editing <c>App.xaml.cs</c>, <c>LaunchOptions.cs</c> or <c>DesignWindow</c>:
/// <code>
/// static class GlassCatalogsRegistration
/// {
///     [ModuleInitializer]
///     internal static void Register()
///     {
///         AppHooks.Commands.Add(new("Glass Catalogs…", "\uEB42", w => GlassCatalogsWindow.Show(w.Session),
///             VirtualKey.G, VirtualKeyModifiers.Control | VirtualKeyModifiers.Menu));
///         AppHooks.LaunchActions["glass-catalogs"] = (w, arg) => { GlassCatalogsWindow.Show(w.Session, arg); return Task.CompletedTask; };
///     }
/// }
/// </code>
/// </summary>
public static class AppHooks
{
    /// <summary>
    /// Launch actions by name: <c>--do name</c> or <c>--do name=arg</c> (repeatable, in order) runs
    /// them in the design window once its panes have their results, before <c>--screenshot</c>; the
    /// screenshot waits again until the panes are idle. An action may open other windows; to capture
    /// one of those instead, it sets <see cref="ScreenshotTarget"/>.
    /// </summary>
    public static Dictionary<string, Func<DesignWindow, string?, Task>> LaunchActions { get; } = [];

    /// <summary>Commands in the menu bar (and their shortcuts) of every design window.</summary>
    public static List<WindowCommand> Commands { get; } = [];

    /// <summary>Raised for every design window after it is built.</summary>
    public static event Action<DesignWindow>? WindowCreated;
    internal static void RaiseWindowCreated(DesignWindow w) => WindowCreated?.Invoke(w);

    /// <summary>
    /// Run when a design's session starts, before its project catalogs and first snapshot (the app-wide
    /// glass library syncs its catalogs and overrides into every session here).
    /// </summary>
    public static List<Func<ViewModels.DesignSession, Task>> SessionStarting { get; } = [];

    /// <summary>The window <c>--screenshot</c> captures when a launch action opened another one.</summary>
    public static Window? ScreenshotTarget { get; set; }
}
