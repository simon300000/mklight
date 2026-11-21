using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;
using MKLight.Model;

namespace MKLight.App.Views.Glass;

/// <summary>The Glass Catalogs window's small dialogs: the reload report, rename, confirmations, lists.</summary>
public static class GlassDialogs
{
    static Style S(string key) => (Style)Application.Current.Resources[key];

    static ContentDialog Dialog(FrameworkElement owner, string title, object content) => new()
    {
        Title = title, Content = content, XamlRoot = owner.XamlRoot, RequestedTheme = owner.ActualTheme,
        DefaultButton = ContentDialogButton.Primary,
    };

    /// <summary>What reloading a catalog changed: added, removed and changed glasses with n_d and V_d before → after.</summary>
    public static async Task ShowChangesAsync(FrameworkElement owner, CatalogChanges changes)
    {
        var p = new StackPanel { Spacing = 12, Width = 440 };
        p.Children.Add(new TextBlock
        {
            Text = changes.IsEmpty
                ? Loc.F("Glass_Reload_Same", changes.Unchanged)
                : Loc.F("Glass_Reload_Summary", GlassFormat.ChangeSummary(changes), changes.Unchanged),
            Style = S("SecondaryText"), TextWrapping = TextWrapping.Wrap,
        });
        if (!changes.IsEmpty)
        {
            var list = new StackPanel { Spacing = 14 };
            void Group(string title, string glyph, IEnumerable<(string Name, string? Detail)> items)
            {
                var entries = items.ToList();
                if (entries.Count == 0) return;
                var g = new StackPanel { Spacing = 2 };
                var head = new StackPanel { Orientation = Orientation.Horizontal, Spacing = 6, Margin = new Thickness(0, 0, 0, 4) };
                head.Children.Add(new FontIcon { Glyph = glyph, FontSize = 12, Foreground = ThemeLookup.Brush(owner, "TextFillColorSecondaryBrush") });
                head.Children.Add(new TextBlock { Text = Loc.F("Glass_Reload_Group", title, entries.Count), Style = S("ColumnHeaderText") });
                g.Children.Add(head);
                foreach (var (name, detail) in entries)
                {
                    var e = new StackPanel { Padding = new Thickness(18, 2, 0, 2) };
                    e.Children.Add(new TextBlock { Text = name, Style = S("BodyText") });
                    if (detail is { Length: > 0 }) e.Children.Add(new TextBlock { Text = detail, Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap });
                    g.Children.Add(e);
                }
                list.Children.Add(g);
            }
            Group(Loc.S("Glass_Reload_Added"), "\uE710", changes.Added.Select(n => (n, (string?)null)));
            Group(Loc.S("Glass_Reload_Removed"), "\uE738", changes.Removed.Select(n => (n, (string?)null)));
            Group(Loc.S("Glass_Reload_Changed"), "\uE70F", changes.Changed.Select(c => (c.Name, (string?)GlassFormat.Describe(c))));
            p.Children.Add(new Border
            {
                Background = ThemeLookup.Brush(owner, "CardBackgroundFillColorDefaultBrush"),
                BorderBrush = ThemeLookup.Brush(owner, "CardStrokeColorDefaultBrush"), BorderThickness = new Thickness(1),
                CornerRadius = new CornerRadius(8), Padding = new Thickness(12, 10, 12, 12),
                Child = new ScrollViewer { Content = list, MaxHeight = 320 },
            });
        }
        if (changes.Warnings.Count > 0)
            p.Children.Add(new TextBlock
            {
                Text = changes.Warnings.Count == 1
                    ? Loc.F("Glass_Reload_NotesOne", changes.Warnings.Count, FirstNotes(changes))
                    : Loc.F("Glass_Reload_NotesMany", changes.Warnings.Count, FirstNotes(changes)),
                Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap,
            });
        var d = Dialog(owner, Loc.F("Glass_Reload_Title", changes.Catalog), p);
        d.CloseButtonText = Loc.S("Glass_Dialog_Done");
        d.DefaultButton = ContentDialogButton.Close;
        await d.ShowAsync();
    }

    /// <summary>The first three load notes, joined (an ellipsis when there are more).</summary>
    static string FirstNotes(CatalogChanges changes)
    {
        var separator = Loc.S("Glass_List_Separator");
        return string.Join(separator, changes.Warnings.Take(3).Select(w => EngineText.T(w.Message))) + (changes.Warnings.Count > 3 ? separator + "…" : "");
    }

    /// <summary>Asks for a new name; null when cancelled.</summary>
    public static async Task<string?> AskNameAsync(FrameworkElement owner, string title, string current, string note, string? action = null)
    {
        var box = new TextBox { Header = Loc.S("Glass_Label_Name"), Text = current };
        box.SelectAll();
        var p = new StackPanel { Spacing = 10, Width = 360 };
        p.Children.Add(box);
        p.Children.Add(new TextBlock { Text = note, Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap });
        var d = Dialog(owner, title, p);
        d.PrimaryButtonText = action ?? Loc.S("Glass_Rename_Button");
        d.CloseButtonText = Loc.S("Common_Cancel");
        box.TextChanged += (_, _) => d.IsPrimaryButtonEnabled = GlassLibrary.NameProblem(box.Text) is null;
        return await d.ShowAsync() == ContentDialogResult.Primary ? box.Text.Trim() : null;
    }

    /// <summary>A destructive confirmation; returns whether confirmed.</summary>
    public static async Task<bool> ConfirmAsync(FrameworkElement owner, string title, string message, string action)
    {
        var d = Dialog(owner, title, new TextBlock { Text = message, Style = S("BodyText"), TextWrapping = TextWrapping.Wrap, Width = 360 });
        d.PrimaryButtonText = action;
        d.CloseButtonText = Loc.S("Common_Cancel");
        d.DefaultButton = ContentDialogButton.Close;
        return await d.ShowAsync() == ContentDialogResult.Primary;
    }

    /// <summary>A list of notes (AGF export, catalog load warnings).</summary>
    public static async Task ShowListAsync(FrameworkElement owner, string title, string intro, IEnumerable<string> lines)
    {
        var p = new StackPanel { Spacing = 10, Width = 440 };
        p.Children.Add(new TextBlock { Text = intro, Style = S("SecondaryText"), TextWrapping = TextWrapping.Wrap });
        var list = new StackPanel { Spacing = 4 };
        foreach (var l in lines) list.Children.Add(new TextBlock { Text = l, Style = S("BodyText"), TextWrapping = TextWrapping.Wrap });
        p.Children.Add(new ScrollViewer { Content = list, MaxHeight = 360 });
        var d = Dialog(owner, title, p);
        d.CloseButtonText = Loc.S("Glass_Dialog_Done");
        d.DefaultButton = ContentDialogButton.Close;
        await d.ShowAsync();
    }
}
