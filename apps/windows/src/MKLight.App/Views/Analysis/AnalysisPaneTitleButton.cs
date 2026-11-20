using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Controls;
using Microsoft.UI.Xaml.Input;

namespace MKLight.App.Views.Analysis;

/// <summary>
/// The pane's title: the handle the panel is dragged by, and the button that
/// opens the analysis menu. A plain button captures the pointer on press, so
/// the panel drag (<c>PanelHandles</c>, attached by the panel host to the
/// element named <c>TitleButton</c>) could not take it over; this one does
/// not capture: a press and release without a drag opens the menu, a drag past
/// the threshold moves the panel (the drag captures the pointer, so the
/// release never reaches the title). Keyboard and UI automation invoke it as a
/// button.
/// </summary>
public sealed partial class AnalysisPaneTitleButton : Button
{
    uint? _pressed;
    global::Windows.Foundation.Point _start;

    protected override void OnPointerPressed(PointerRoutedEventArgs e)
    {
        // No base call: no capture, no click on release. The event bubbles on (the pane takes the focus).
        var pt = e.GetCurrentPoint(this);
        if (!pt.Properties.IsLeftButtonPressed) return;
        _pressed = e.Pointer.PointerId;
        _start = pt.Position;
        VisualStateManager.GoToState(this, "Pressed", true);
    }

    protected override void OnPointerMoved(PointerRoutedEventArgs e)
    {
        base.OnPointerMoved(e);
        // Past the drag threshold the press is a drag (the panel host starts it), not a click.
        var p = e.GetCurrentPoint(this).Position;
        if (_pressed == e.Pointer.PointerId && (Math.Abs(p.X - _start.X) >= 6 || Math.Abs(p.Y - _start.Y) >= 6)) _pressed = null;
    }

    protected override void OnPointerReleased(PointerRoutedEventArgs e)
    {
        var click = _pressed == e.Pointer.PointerId;
        _pressed = null;
        VisualStateManager.GoToState(this, "PointerOver", true);
        if (!click) return;
        e.Handled = true;
        Flyout?.ShowAt(this);
    }

    protected override void OnPointerCaptureLost(PointerRoutedEventArgs e)
    {
        // A drag took the pointer.
        _pressed = null;
        VisualStateManager.GoToState(this, "Normal", true);
        base.OnPointerCaptureLost(e);
    }

    protected override void OnPointerExited(PointerRoutedEventArgs e)
    {
        base.OnPointerExited(e);
        _pressed = null;
    }

    protected override void OnPointerCanceled(PointerRoutedEventArgs e)
    {
        base.OnPointerCanceled(e);
        _pressed = null;
    }
}
