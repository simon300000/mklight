using System.Globalization;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using Windows.ApplicationModel.DataTransfer;

namespace MKLight.App.Windows;

/// <summary>
/// The dialog a design window shows once after importing a lens prescription file (the Mac app's
/// <c>ImportReportSheet</c>): what was left out or changed, grouped by meaning, each item with its
/// surface and line in the file; Copy Report, Load Glass Catalog… (when glasses were not found and
/// the file can be imported again) and Done.
/// <code>
/// ┌──────────────────────────────────────────────┐
/// │ ⬇  Imported “lens.zmx”                        │
/// │    3 items could not be imported exactly. …   │
/// │    6 surfaces · .zmx · UTF-16                 │
/// │ ┌──────────────────────────────────────────┐ │
/// │ │ ⚠ Glasses                              2 │ │
/// │ │   Glass X is not in the loaded catalogs  │ │
/// │ │   Surface 1 · Line 51                    │ │
/// │ └──────────────────────────────────────────┘ │
/// │      Copy Report   Load Glass Catalog…  Done │
/// └──────────────────────────────────────────────┘
/// </code>
/// </summary>
static class ImportReportDialog
{
    /// <summary>The report for <paramref name="root"/>'s window; its Secondary result is Load Glass Catalog….</summary>
    public static ContentDialog Create(FrameworkElement root, ImportReport report, bool canLoadCatalog)
    {
        var dialog = new ContentDialog
        {
            Content = Body(report),
            PrimaryButtonText = Loc.S("Shell_Import_CopyReport"),
            CloseButtonText = Loc.S("Shell_Import_Done"),
            DefaultButton = ContentDialogButton.Close,
            XamlRoot = root.XamlRoot,
            RequestedTheme = root.ActualTheme,
        };
        dialog.Resources["ContentDialogMaxWidth"] = 720.0;
        dialog.Resources["ContentDialogMaxHeight"] = 760.0;
        if (canLoadCatalog && report.HasUnknownGlasses) dialog.SecondaryButtonText = Loc.S("Shell_Import_LoadCatalog");
        dialog.PrimaryButtonClick += (d, e) =>
        {
            e.Cancel = true; // copying keeps the report open
            var data = new DataPackage();
            data.SetText(report.PlainText);
            Clipboard.SetContent(data);
            d.PrimaryButtonText = Loc.S("Shell_Import_Copied");
        };
        return dialog;
    }

    static Style Style(string key) => (Style)Application.Current.Resources[key];

    static FrameworkElement Body(ImportReport report)
    {
        var body = new StackPanel { Width = 620, Spacing = 16 };

        // Header: the import glyph in the accent colour, the file, what it means, the facts.
        var header = new Grid { ColumnSpacing = 14 };
        header.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
        header.ColumnDefinitions.Add(new ColumnDefinition());
        var icon = new FontIcon { Glyph = "\uE896", FontSize = 32, VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0) };
        ThemeLookup.Follow(icon, () => icon.Foreground = ThemeLookup.Brush(icon, "AccentTextFillColorPrimaryBrush"));
        header.Children.Add(icon);
        var text = new StackPanel { Spacing = 4 };
        text.Children.Add(new TextBlock
        {
            Text = Loc.F("Shell_Import_Title", report.FileName), Style = Style("BodyText"), FontSize = 20, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold,
            FontFamily = (FontFamily)Application.Current.Resources["UIDisplayFont"],
        });
        text.Children.Add(new TextBlock { Text = report.Headline, Style = Style("SecondaryText"), TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None });
        var summary = new TextBlock { Text = report.Summary, Style = Style("CaptionText") };
        ThemeLookup.Follow(summary, () => summary.Foreground = ThemeLookup.Brush(summary, "TextFillColorTertiaryBrush"));
        text.Children.Add(summary);
        Grid.SetColumn(text, 1);
        header.Children.Add(text);
        body.Children.Add(header);

        // The warnings by group on a plain card.
        var list = new StackPanel { Spacing = 18, Padding = new Thickness(16, 14, 16, 16) };
        foreach (var group in report.Groups)
        {
            var section = new StackPanel { Spacing = 10 };
            var title = new Grid { ColumnSpacing = 8 };
            title.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            title.ColumnDefinitions.Add(new ColumnDefinition());
            title.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            var glyph = new FontIcon { Glyph = ImportReport.Glyph(group.Category), FontSize = 13, Width = 16 };
            ThemeLookup.Follow(glyph, () => glyph.Foreground = ThemeLookup.Brush(glyph, "TextFillColorSecondaryBrush"));
            title.Children.Add(glyph);
            var name = new TextBlock { Text = ImportReport.Title(group.Category), Style = Style("ColumnHeaderText") };
            Grid.SetColumn(name, 1);
            title.Children.Add(name);
            var count = new TextBlock { Text = group.Warnings.Count.ToString(CultureInfo.InvariantCulture), Style = Style("SecondaryNumberText"), FontSize = 12 };
            Grid.SetColumn(count, 2);
            title.Children.Add(count);
            section.Children.Add(title);
            foreach (var w in group.Warnings)
            {
                var row = new StackPanel { Spacing = 2, Margin = new Thickness(24, 0, 0, 0) };
                row.Children.Add(new TextBlock
                {
                    Text = ImportReport.Sentence(w), Style = Style("BodyText"), TextWrapping = TextWrapping.Wrap, TextTrimming = TextTrimming.None,
                    IsTextSelectionEnabled = true,
                });
                if (ImportReport.Location(w) is { } at) row.Children.Add(new TextBlock { Text = at, Style = Style("CaptionText") });
                section.Children.Add(row);
            }
            list.Children.Add(section);
        }
        var card = new Border
        {
            CornerRadius = new CornerRadius(8), BorderThickness = new Thickness(1),
            Child = new ScrollViewer { Content = list, MaxHeight = 340, VerticalScrollBarVisibility = ScrollBarVisibility.Auto },
        };
        ThemeLookup.Follow(card, () =>
        {
            card.Background = ThemeLookup.Brush(card, "ContentSurfaceBrush");
            card.BorderBrush = ThemeLookup.Brush(card, "CardStrokeColorDefaultBrush");
        });
        body.Children.Add(card);
        return body;
    }
}
