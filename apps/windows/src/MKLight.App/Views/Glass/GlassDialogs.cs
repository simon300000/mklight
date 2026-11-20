using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;

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
                ? $"The file has the same glasses as before ({changes.Unchanged})."
                : $"{changes.Summary}; {changes.Unchanged} unchanged.",
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
                head.Children.Add(new TextBlock { Text = $"{title} ({entries.Count})", Style = S("ColumnHeaderText") });
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
            Group("Added", "\uE710", changes.Added.Select(n => (n, (string?)null)));
            Group("Removed", "\uE738", changes.Removed.Select(n => (n, (string?)null)));
            Group("Changed", "\uE70F", changes.Changed.Select(c => (c.Name, (string?)GlassFormat.Describe(c))));
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
                Text = $"{changes.Warnings.Count} note{(changes.Warnings.Count == 1 ? "" : "s")} while reading the file: " + string.Join("; ", changes.Warnings.Take(3).Select(w => w.Message)) + (changes.Warnings.Count > 3 ? "; …" : ""),
                Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap,
            });
        var d = Dialog(owner, $"{changes.Catalog} Reloaded", p);
        d.CloseButtonText = "Done";
        d.DefaultButton = ContentDialogButton.Close;
        await d.ShowAsync();
    }

    /// <summary>Asks for a new name; null when cancelled.</summary>
    public static async Task<string?> AskNameAsync(FrameworkElement owner, string title, string current, string note, string action = "Rename")
    {
        var box = new TextBox { Header = "Name", Text = current };
        box.SelectAll();
        var p = new StackPanel { Spacing = 10, Width = 360 };
        p.Children.Add(box);
        p.Children.Add(new TextBlock { Text = note, Style = S("CaptionText"), TextWrapping = TextWrapping.Wrap });
        var d = Dialog(owner, title, p);
        d.PrimaryButtonText = action;
        d.CloseButtonText = "Cancel";
        box.TextChanged += (_, _) => d.IsPrimaryButtonEnabled = GlassLibrary.NameProblem(box.Text) is null;
        return await d.ShowAsync() == ContentDialogResult.Primary ? box.Text.Trim() : null;
    }

    /// <summary>A destructive confirmation; returns whether confirmed.</summary>
    public static async Task<bool> ConfirmAsync(FrameworkElement owner, string title, string message, string action)
    {
        var d = Dialog(owner, title, new TextBlock { Text = message, Style = S("BodyText"), TextWrapping = TextWrapping.Wrap, Width = 360 });
        d.PrimaryButtonText = action;
        d.CloseButtonText = "Cancel";
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
        d.CloseButtonText = "Done";
        d.DefaultButton = ContentDialogButton.Close;
        await d.ShowAsync();
    }
}
