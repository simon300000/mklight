using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Media;

namespace MKLight.App.Windows;

/// <summary>Shown instead of the app when the engine DLL is missing or incompatible.</summary>
public sealed class EngineMissingWindow : Window
{
    public EngineMissingWindow(string reason)
    {
        Title = "MKLight";
        SystemBackdrop = new MicaBackdrop();
        var close = new Button { Content = "Quit", Style = (Style)Application.Current.Resources["AccentCapsuleButton"], HorizontalAlignment = HorizontalAlignment.Right };
        close.Click += (_, _) => Close();
        Content = new StackPanel
        {
            Spacing = 12, Padding = new Thickness(32), VerticalAlignment = VerticalAlignment.Center,
            Children =
            {
                new TextBlock { Text = "The optical engine is not available", FontSize = 20, FontWeight = Microsoft.UI.Text.FontWeights.SemiBold },
                new TextBlock { Text = reason, TextWrapping = TextWrapping.Wrap, Foreground = (Brush)Application.Current.Resources["TextFillColorSecondaryBrush"] },
                new TextBlock { Text = "Build it with scripts/build-windows.ps1, which places mklight_ffi.dll next to the app.", TextWrapping = TextWrapping.Wrap },
                close,
            },
        };
        App.Size(this, 560, 300);
    }
}
