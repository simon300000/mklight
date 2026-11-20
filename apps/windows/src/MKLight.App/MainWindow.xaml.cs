using Microsoft.Graphics.Canvas.UI.Xaml;
using Microsoft.UI.Xaml;

namespace MKLight.App;

public sealed partial class MainWindow : Window
{
    public MainWindow() { InitializeComponent(); SystemBackdrop = new Microsoft.UI.Xaml.Media.MicaBackdrop(); }
    private void OnDraw(CanvasControl sender, CanvasDrawEventArgs args)
        => args.DrawingSession.DrawText("MKLight", 20, 20, Microsoft.UI.Colors.SteelBlue);
}
