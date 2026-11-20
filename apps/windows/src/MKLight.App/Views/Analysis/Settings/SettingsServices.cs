using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.App.Views.Panels;
using MKLight.App.Windows;
using MKLight.Engine;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// What the settings controls need beyond the <see cref="SettingsContext"/>:
/// the design window that hosts a control (its session, for the design's
/// wavelength weights and the engine; its window id, for file pickers).
/// Windows are found by the control's <see cref="XamlRoot"/>.
/// </summary>
public static class SettingsServices
{
    static readonly List<WeakReference<DesignWindow>> windows = [];

    [ModuleInitializer]
    internal static void Register() => AppHooks.WindowCreated += w =>
    {
        lock (windows)
        {
            windows.RemoveAll(r => !r.TryGetTarget(out _));
            windows.Add(new(w));
        }
    };

    /// <summary>The session a control belongs to and the window it is shown in (for file pickers).</summary>
    public sealed record Host(DesignSession Session, WindowId WindowId);

    /// <summary>
    /// The window <paramref name="element"/> is shown in, once it is in a window: its design window or one
    /// of the design's separate panel windows (its session's design window when the panel window is gone).
    /// </summary>
    public static Host? Window(UIElement element)
    {
        if (element.XamlRoot is not { } root) return null;
        List<DesignWindow> open;
        lock (windows) open = windows.Select(r => r.TryGetTarget(out var w) ? w : null).OfType<DesignWindow>().ToList();
        foreach (var w in open)
        {
            if (w.Content?.XamlRoot == root) return new(w.Session, w.AppWindow.Id);
            if (w.Panels.TreeOf(root) is { } tree)
                return new(w.Session, (tree.HostWindow as PanelWindow)?.AppWindow.Id ?? w.AppWindow.Id);
        }
        return null;
    }

    /// <summary>
    /// The weighted wavelengths <paramref name="spectrum"/> resolves to on the
    /// design: a <c>system_summary</c> run with it. Null when there is no
    /// snapshot or the engine says nothing.
    /// </summary>
    public static async Task<IReadOnlyList<SpectrumSample>?> ResolveAsync(Host window, JsonNode spectrum, CancellationToken cancel)
    {
        var w = window.Session.Workspace;
        if ((w.Snapshot ?? w.DesignSnapshot) is not { } snapshot) return null;
        var job = w.Engine.Submit(new EngineAnalysisRequest("system_summary", [new("spectrum", spectrum.DeepClone())]), snapshot);
        using var reg = cancel.Register(job.Cancel);
        using var output = await job.ResultAsync(cancel);
        return SpectrumSetting.ResolvedSamples(output.Json);
    }
}
