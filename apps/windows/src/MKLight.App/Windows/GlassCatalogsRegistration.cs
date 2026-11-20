using System.Globalization;
using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using Windows.System;

namespace MKLight.App.Windows;

/// <summary>
/// The glass library and the Glass Catalogs window in the app: the library syncs into every
/// design session before its first snapshot; File ▸ Glass Catalogs… (Ctrl+Alt+G) and Add Glass
/// Catalog to Design…; the window follows the frontmost design window; launch actions for checks.
/// </summary>
static class GlassCatalogsRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        AppHooks.SessionStarting.Add(async session => await GlassLibrarySync.SyncAsync(session.Workspace.Engine, GlassCatalogsWindow.Library));
        AppHooks.Commands.Add(new("Glass Catalogs…", "\uE8F1", w => GlassCatalogsWindow.Show(w),
            VirtualKey.G, VirtualKeyModifiers.Control | VirtualKeyModifiers.Menu, Order: 40, Menu: "Lens"));
        AppHooks.Commands.Add(new("Add Glass Catalog to Design…", "\uE8E5", w => _ = GlassCatalogsWindow.AddCatalogToDesignAsync(w), Order: 41, Menu: "Lens"));
        AppHooks.WindowCreated += w =>
        {
            w.Activated += (_, e) =>
            {
                if (e.WindowActivationState != WindowActivationState.Deactivated) GlassCatalogsWindow.Instance?.Follow(w);
            };
            w.Closed += (_, _) => GlassCatalogsWindow.Instance?.Forget(w);
        };

        // Launch actions (`--do name[=arg]`): open the window and set what it shows, for checks and screenshots.
        // The scratch library for checks: the MKLIGHT_SUPPORT_DIRECTORY environment variable.
        var actions = AppHooks.LaunchActions;
        actions["glass-catalogs"] = async (w, _) => await Open(w);
        actions["glass-catalogs-search"] = async (w, arg) => (await Open(w)).Model.Search = arg ?? "";
        actions["glass-catalogs-scope"] = async (w, arg) =>
        {
            var g = await Open(w);
            // "My_Glasses" names "My Glasses" (launch arguments have no spaces).
            g.Model.Scope = arg is null or "" or "all" ? null : g.Model.Catalog(arg) is null && g.Model.Catalog(arg.Replace('_', ' ')) is { } c ? c.Name : arg;
            await Settle();
        };
        actions["glass-catalogs-filter"] = async (w, arg) =>
            (await Open(w)).Model.Filter = Enum.TryParse<GlassFilter>(arg, true, out var f) ? f : GlassFilter.All;
        actions["glass-catalogs-select"] = async (w, arg) =>
        {
            var g = await Open(w);
            var row = g.Model.VisibleRows.FirstOrDefault(r => r.Id == arg) ?? g.Model.VisibleRows.FirstOrDefault(r => string.Equals(r.Name, arg, StringComparison.OrdinalIgnoreCase));
            g.Model.Selection = row?.Id;
            await Settle();
        };
        // Dialogs stay open for the screenshot: these do not wait for them to close.
        actions["glass-catalogs-reload"] = async (w, arg) =>
        {
            var g = await Open(w);
            if (g.Model.Catalog(arg ?? "") is { } c) _ = g.ReloadAsync(c);
            await Settle();
        };
        actions["glass-catalogs-new-glass"] = async (w, arg) => { _ = (await Open(w)).NewGlassAsync(); await Settle(); };
        actions["glass-catalogs-edit"] = async (w, arg) =>
        {
            var g = await Open(w);
            if (g.Model.SelectedRow is { } row)
                _ = g.Model.Library.IsUserCatalog(row.Catalog) ? g.EditSelectedAsync() : g.DuplicateAsync(row);
            await Settle();
        };
        actions["glass-catalogs-add-to-design"] = async (w, arg) =>
        {
            if (arg is { Length: > 0 }) await (await Open(w)).Model.AddToDesignAsync(Path.GetFullPath(arg));
        };
        // Uses the selected glass for the design's selected surface; the screenshot shows the design.
        actions["glass-catalogs-use"] = async (w, arg) =>
        {
            (await Open(w)).Model.UseSelected();
            AppHooks.ScreenshotTarget = w;
            w.Activate();
        };
        actions["glass-catalogs-pause"] = (_, arg) =>
            Task.Delay(TimeSpan.FromSeconds(double.TryParse(arg, NumberStyles.Float, CultureInfo.InvariantCulture, out var s) ? s : 1));
    }

    static async Task<GlassCatalogsWindow> Open(DesignWindow w)
    {
        var g = GlassCatalogsWindow.Show(w);
        AppHooks.ScreenshotTarget = g;
        await g.Model.RefreshAsync();
        return g;
    }

    static Task Settle() => Task.Delay(500);
}
