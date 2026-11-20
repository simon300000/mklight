using System.Runtime.CompilerServices;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace MKLight.App.Views.Inspector;

/// <summary>Registers the Surface and System pages (placeholders until the inspector lands).</summary>
static class InspectorRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        InspectorPages.Add(new("surface", "Surface", "Surface", 10, _ => Placeholder("Surface")));
        InspectorPages.Add(new("system", "System", "System", 11, _ => Placeholder("System")));
    }

    internal static UIElement Placeholder(string title) => new Border
    {
        CornerRadius = new CornerRadius(8),
        Background = (Microsoft.UI.Xaml.Media.Brush)Application.Current.Resources["CardBackgroundFillColorDefaultBrush"],
        Child = new TextBlock
        {
            Text = title, Style = (Style)Application.Current.Resources["SecondaryText"],
            HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center,
        },
    };
}
