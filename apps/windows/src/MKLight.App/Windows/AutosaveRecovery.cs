using System.Globalization;
using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Windows;

/// <summary>
/// At launch, designs a previous run left with unsaved changes (their autosaves, <see cref="Autosaves"/>):
/// one question each over the frontmost window. Recover opens it with the changes as one undoable step
/// (so it shows as changed, and undo goes back to the saved file); Discard deletes the autosave; Later
/// keeps it for the next launch.
/// </summary>
static class AutosaveRecovery
{
    public static async Task OfferAsync()
    {
        IReadOnlyList<Autosaves.Entry> entries;
        try { entries = Autosaves.Find(AppSettings.SupportDirectory, AppSettings.Recent); }
        catch (Exception e) { App.Log($"autosave recovery: {e.Message}"); return; }
        foreach (var entry in entries)
        {
            if (await HostAsync() is not { } root) return;
            var dialog = new ContentDialog
            {
                Title = Loc.S("Shell_Recover_Title"),
                Content = Body(entry),
                PrimaryButtonText = Loc.S("Shell_Recover_Recover"), SecondaryButtonText = Loc.S("Shell_Recover_Discard"), CloseButtonText = Loc.S("Shell_Recover_Later"),
                DefaultButton = ContentDialogButton.Primary, XamlRoot = root.XamlRoot, RequestedTheme = root.ActualTheme,
            };
            ContentDialogResult answer;
            try { answer = await dialog.ShowAsync(); }
            catch (Exception e) { App.Log($"autosave recovery: {e.Message}"); continue; } // another dialog is open: offered again next launch
            if (answer == ContentDialogResult.Primary) Recover(entry);
            else if (answer == ContentDialogResult.Secondary) Autosaves.Delete(entry.Path);
        }
    }

    static FrameworkElement Body(Autosaves.Entry entry)
    {
        var text = new StackPanel { Spacing = 6, MaxWidth = 440 };
        var when = entry.Modified.ToString("yyyy-MM-dd HH:mm", CultureInfo.InvariantCulture);
        text.Children.Add(new TextBlock
        {
            Text = entry.Original is null
                ? Loc.F("Shell_Recover_UntitledBody", when)
                : Loc.F("Shell_Recover_Body", entry.Title, when),
            Style = (Style)Application.Current.Resources["BodyText"], TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None,
        });
        text.Children.Add(new TextBlock
        {
            Text = entry.Original ?? Loc.S("Shell_Recover_UntitledNote"),
            Style = (Style)Application.Current.Resources["CaptionText"], TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None,
        });
        return text;
    }

    /// <summary>The frontmost window's content once it is loaded.</summary>
    static async Task<FrameworkElement?> HostAsync()
    {
        for (var i = 0; i < 100; i++)
        {
            if (App.Windows.LastOrDefault(w => w is WelcomeWindow or DesignWindow)?.Content is FrameworkElement { XamlRoot: not null, IsLoaded: true } root) return root;
            await Task.Delay(100);
        }
        return null;
    }

    /// <summary>
    /// The recovered window's note on the engine's <c>open_warnings</c>: the catalogs it could not load by name
    /// (<c>glass catalog 'NAME' could not be loaded (…)</c>), else the warnings as they are; null without any.
    /// </summary>
    public static string? MissingCatalogsNote(IReadOnlyList<string> warnings)
    {
        if (warnings.Count == 0) return null;
        var names = warnings.Select(w => System.Text.RegularExpressions.Regex.Match(w, "^glass catalog '(.+?)' could not be loaded"))
            .Where(m => m.Success).Select(m => m.Groups[1].Value).ToList();
        if (names.Count != warnings.Count) return Loc.F("Shell_Recover_WithWarnings", string.Join("; ", warnings.Select(EngineText.T)));
        return names.Count == 1
            ? Loc.F("Shell_Recover_CatalogMissingOne", names[0])
            : Loc.F("Shell_Recover_CatalogMissingMany", string.Join(", ", names));
    }

    /// <summary>Opens the autosave: the design's file with the autosaved changes applied, or a new untitled design.</summary>
    public static DesignWindow? Recover(Autosaves.Entry entry)
    {
        try
        {
            if (App.Engine is not FFIEngine engine) return null;
            var (recovered, warnings) = Autosaves.Read(engine, entry.Path);
            var original = entry.Original is { } o && File.Exists(o) ? o : null;
            var doc = new DesignDocument(Autosaves.RecoveryBase(entry with { Original = original }, recovered), original);
            doc.EditProject(Loc.S("Shell_Action_RecoverUnsaved"), _ => recovered);
            // The window's layout is not undoable, so the edit kept the base's: take the autosave's panes and arrangement.
            if (recovered.Sections.TryGetValue("views", out var views)) doc.SetSection("views", views);
            var window = App.OpenDocument(doc);
            window.AdoptAutosave(entry.Path);
            // Catalogs the engine could not load are kept as references (saved again); say which.
            if (MissingCatalogsNote(warnings) is { } note) window.Session.ShowToast(note);
            foreach (var w in warnings) App.Log($"recover {entry.Path}: {w}");
            return window;
        }
        catch (Exception e)
        {
            App.Log($"recover {entry.Path}: {e.Message}");
            return null;
        }
    }
}

/// <summary>Launch actions of the document commands (<c>--do name[=arg]</c>), for checks.</summary>
static class DocumentsRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        var a = AppHooks.LaunchActions;
        a["revert"] = (w, _) => w.RevertAsync();
        a["duplicate"] = async (w, _) =>
        {
            await w.DuplicateAsync();
            if (AppHooks.ScreenshotTarget is DesignWindow copy) await App.WaitIdleAsync(copy.Session);
        };
        a["rename"] = async (w, arg) => await w.RenameToAsync(arg ?? "");
        a["replace-sample"] = (w, arg) => { w.ReplaceWithSample(arg ?? ""); return Task.CompletedTask; };
        a["autosave"] = (w, _) => { App.Log($"autosaved: {w.AutosaveNow()}"); return Task.CompletedTask; };
        // Recovers the newest autosave (as Recover in the launch question does) and captures its window.
        a["recover-autosave"] = async (w, _) =>
        {
            if (Autosaves.Find(AppSettings.SupportDirectory, AppSettings.Recent).FirstOrDefault() is not { } entry) { App.Log("no autosave to recover"); return; }
            if (AutosaveRecovery.Recover(entry) is { } r) { AppHooks.ScreenshotTarget = r; await App.WaitIdleAsync(r.Session); }
        };
        a["import-catalog"] =async (w, arg) =>
        {
            if (arg is { Length: > 0 }) await w.LoadCatalogAndImportAgainAsync(Path.GetFullPath(arg));
        };
    }
}
