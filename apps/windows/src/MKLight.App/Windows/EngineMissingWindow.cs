using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;

namespace MKLight.App.Windows;

/// <summary>Shown instead of the app when the engine DLL is missing or incompatible.</summary>
public sealed class EngineMissingWindow : Window
{
    public EngineMissingWindow(string reason)
    {
        Title = "MKLight";
        SystemBackdrop = new MicaBackdrop();
        var close = new Button { Content = Loc.S("Shell_EngineMissing_Quit"), Style = (Style)Application.Current.Resources["AccentCapsuleButton"], HorizontalAlignment = HorizontalAlignment.Right };
        close.Click += (_, _) => Close();
        Content = new StackPanel
        {
            Spacing = 12, Padding = new Thickness(32), VerticalAlignment = VerticalAlignment.Center,
            Children =
            {
                new TextBlock { Text = Loc.S("Shell_EngineMissing_Title"), FontSize = 20, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold },
                new TextBlock { Text = reason, TextWrapping = TextWrapping.Wrap, Foreground = (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"] },
                new TextBlock { Text = Loc.S("Shell_EngineMissing_Build"), TextWrapping = TextWrapping.Wrap },
                close,
            },
        };
        App.Size(this, 560, 300);
    }
}
