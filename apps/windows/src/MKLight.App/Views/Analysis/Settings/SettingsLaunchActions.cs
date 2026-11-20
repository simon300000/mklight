using System.Runtime.CompilerServices;
using System.Text.Json.Nodes;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.Windows;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>
/// Launch actions for the settings form and the run details (UI testing and screenshots):
/// <list type="bullet">
/// <item><c>--do spectrum=&lt;kind&gt;</c>: every pane's spectrum (<c>photopic</c>, <c>scotopic</c>,
/// <c>blackbody</c>, <c>blackbody:3200</c>, <c>explicit</c>, <c>design</c>, or the setting's JSON).</item>
/// <item><c>--do spectrum-csv=&lt;file&gt;</c>: imports a sensor response into every open spectrum control.</item>
/// <item><c>--do settings-scroll=end</c>: scrolls every open settings form to its end (the spectrum).</item>
/// <item><c>--do run-details[=&lt;analysis&gt;]</c>: opens the run details of the first pane (of that analysis).</item>
/// <item><c>--do popup=&lt;setting&gt;</c>: opens the popup table of that setting (sensor limit, row tables).</item>
/// <item><c>--do settings-open-multiselect=&lt;setting&gt;[:1,3]</c>: opens the check-mark menu of a field,
/// wavelength or surface selection and toggles those options (numbered from 1).</item>
/// </list>
/// </summary>
static class SettingsLaunchActions
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.LaunchActions["spectrum"] = SetSpectrum;
        AppHooks.LaunchActions["spectrum-csv"] = async (w, arg) =>
        {
            foreach (var c in Descendants<SpectrumControl>(w.RootElement).ToList())
                if (arg is not null) await c.ImportFileAsync(Path.GetFullPath(arg));
            await Task.Delay(400);
        };
        AppHooks.LaunchActions["settings-scroll"] = async (w, _) =>
        {
            await Task.Delay(300);
            foreach (var form in Descendants<SettingsForm>(w.RootElement))
                if (Ancestor<ScrollViewer>(form) is { } scroller) scroller.ChangeView(null, scroller.ScrollableHeight, null, true);
            await Task.Delay(200);
        };
        AppHooks.LaunchActions["run-details"] = async (w, arg) =>
        {
            foreach (var pane in w.Panels.PaneViews)
            {
                if (arg is not null && pane.Pane.Analysis != arg) continue;
                if (Descendants<Button>(pane).FirstOrDefault(b => b.Name == "BackendBadge") is { Flyout: { } flyout } badge)
                {
                    flyout.ShowAt(badge);
                    // As a pointer opens it: focus without the keyboard focus ring.
                    await Task.Delay(300);
                    VisualTreeHelper.GetOpenPopupsForXamlRoot(w.RootElement.XamlRoot)
                        .SelectMany(p => p.Child is { } c ? Descendants<Control>(c) : []).FirstOrDefault(c => c.IsTabStop)?.Focus(FocusState.Pointer);
                    break;
                }
            }
            await Task.Delay(500);
        };
        // Clicks the first remedy of the open run details (in process, through the button's automation peer).
        AppHooks.LaunchActions["remedy"] = async (w, _) =>
        {
            var button = VisualTreeHelper.GetOpenPopupsForXamlRoot(w.RootElement.XamlRoot)
                .SelectMany(p => p.Child is { } c ? Descendants<Button>(c) : [])
                .FirstOrDefault(b => b.Content is string s && (s.StartsWith("Sample the Pupil", StringComparison.Ordinal) || s.StartsWith("Size the", StringComparison.Ordinal)));
            if (button is not null) new Microsoft.UI.Xaml.Automation.Peers.ButtonAutomationPeer(button).Invoke();
            await Task.Delay(500);
        };
        // settings-open-multiselect=fields[:1,3]: opens the menu of that selection, then toggles
        // those options (numbered from 1) through their automation peers, as clicks do.
        AppHooks.LaunchActions["settings-open-multiselect"] = async (w, arg) =>
        {
            var parts = (arg ?? "fields").Split(':', 2);
            await Task.Delay(300);
            foreach (var form in Descendants<SettingsForm>(w.RootElement))
                if (form.MultiSelectFor(parts[0]) is { Flyout: MenuFlyout menu } button)
                {
                    menu.ShowAt(button);
                    await Task.Delay(300);
                    var options = menu.Items.OfType<ToggleMenuFlyoutItem>().Where(i => i.Text is not ("All" or "Primary")).ToList();
                    foreach (var n in parts.Length > 1 ? parts[1].Split(',', StringSplitOptions.RemoveEmptyEntries) : [])
                        if (int.TryParse(n, System.Globalization.CultureInfo.InvariantCulture, out var k) && k >= 1 && k <= options.Count)
                        {
                            new Microsoft.UI.Xaml.Automation.Peers.ToggleMenuFlyoutItemAutomationPeer(options[k - 1]).Toggle();
                            await Task.Delay(150);
                        }
                    break;
                }
            await Task.Delay(500);
        };
        AppHooks.LaunchActions["popup"] = async (w, arg) =>
        {
            foreach (var form in Descendants<SettingsForm>(w.RootElement))
                if (form.PopupFor(arg ?? "") is { Flyout: { } flyout } button)
                {
                    flyout.ShowAt(button);
                    break;
                }
            await Task.Delay(500);
        };
    }

    static async Task SetSpectrum(DesignWindow w, string? arg)
    {
        JsonNode? spectrum = arg switch
        {
            null or "design" => null,
            "photopic" or "scotopic" => new JsonObject { ["kind"] = arg },
            "explicit" => new SpectrumSetting(SpectrumKind.Explicit,
                Samples: [.. w.Session.Document.System.Wavelengths.Wavelengths.Select(x => new SpectrumSample(x.Um, x.Weight))]).ToJson(),
            _ when arg.StartsWith("blackbody", StringComparison.Ordinal) => new JsonObject
            {
                ["kind"] = "blackbody",
                ["temperature_k"] = arg.Split(':') is [_, var t] && double.TryParse(t, System.Globalization.CultureInfo.InvariantCulture, out var k) ? k : 5500,
            },
            _ => JsonNode.Parse(arg),
        };
        foreach (var view in w.Panels.PaneViews)
        {
            var settings = (JsonObject)view.Pane.Settings.DeepClone();
            if (spectrum is null) settings.Remove("spectrum"); else settings["spectrum"] = spectrum.DeepClone();
            // The form is rebuilt from the new settings.
            if (Descendants<ContentPresenter>(view).FirstOrDefault(c => c.Name == "SettingsContent") is { } host) host.Content = null;
            view.Pane.SetSettings(settings);
        }
        await Task.Delay(300);
    }

    public static IEnumerable<T> Descendants<T>(DependencyObject root) where T : DependencyObject
    {
        var n = VisualTreeHelper.GetChildrenCount(root);
        for (var i = 0; i < n; i++)
        {
            var child = VisualTreeHelper.GetChild(root, i);
            if (child is T t) yield return t;
            foreach (var d in Descendants<T>(child)) yield return d;
        }
    }

    static T? Ancestor<T>(DependencyObject e) where T : DependencyObject
    {
        for (var p = VisualTreeHelper.GetParent(e); p is not null; p = VisualTreeHelper.GetParent(p))
            if (p is T t) return t;
        return null;
    }
}
