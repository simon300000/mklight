using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;

namespace MKLight.App.Views.Analysis;

public sealed partial class AnalysisPaneView : UserControl
{
    public AnalysisPaneView() { InitializeComponent(); }
    void OnRun(object sender, RoutedEventArgs e) { }
    void OnAutoUpdate(object sender, RoutedEventArgs e) { }
    void OnSettingsToggle(object sender, RoutedEventArgs e) { }
    void OnDefaults(object sender, RoutedEventArgs e) { }
    void OnCopyJson(object sender, RoutedEventArgs e) { }
    void OnClose(object sender, RoutedEventArgs e) { }
    void OnSettingsDone(object sender, RoutedEventArgs e) { }
}
