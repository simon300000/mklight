using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace MKLight.App.Windows;

public sealed partial class DesignWindow : Window
{
    public DesignWindow() { InitializeComponent(); ExtendsContentIntoTitleBar = true; SetTitleBar(AppTitleBar); }
    void OnPaneToggleRequested(TitleBar sender, object args) => Nav.IsPaneOpen = !Nav.IsPaneOpen;
    void OnUndo(object sender, RoutedEventArgs e) { }
    void OnRedo(object sender, RoutedEventArgs e) { }
    void OnNew(object sender, RoutedEventArgs e) { }
    void OnOpen(object sender, RoutedEventArgs e) { }
    void OnSave(object sender, RoutedEventArgs e) { }
    void OnSaveAs(object sender, RoutedEventArgs e) { }
    void OnTheme(object sender, RoutedEventArgs e) { }
    void OnInspectorToggle(object sender, RoutedEventArgs e) { }
    void OnInspectorTab(object sender, RoutedEventArgs e) { }
    void OnSidebarSelectionChanged(NavigationView sender, NavigationViewSelectionChangedEventArgs args) { }
    void OnSidebarSearchChanged(AutoSuggestBox sender, AutoSuggestBoxTextChangedEventArgs args) { }
}
