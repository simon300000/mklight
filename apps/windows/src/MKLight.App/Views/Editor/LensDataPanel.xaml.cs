using Microsoft.UI.Xaml.Controls;

namespace MKLight.App.Views;

/// <summary>The lens data table (placeholder until the table lands).</summary>
public sealed partial class LensDataPanel : UserControl
{
    public LensDataPanel() { InitializeComponent(); }
    public MKLight.App.ViewModels.ILensDataHost? Host { get; set; }
}
