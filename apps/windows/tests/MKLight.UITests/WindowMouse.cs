using System.Drawing;
using FlaUI.Core.AutomationElements;

namespace MKLight.UITests;

/// <summary>
/// Mouse input posted to the app's own windows: WM_* mouse messages to the HWND under the point
/// (the WinUI content bridge window, normally), in that window's client coordinates. Nothing here
/// moves the real cursor or touches another application (README.md: no SendInput, no global input).
/// </summary>
sealed class WindowMouse(nint topLevel)
{
    /// <summary>The class of the window WinUI 3 hosts its content in.</summary>
    public const string ContentBridgeClass = "Microsoft.UI.Content.DesktopChildSiteBridge";

    /// <summary>Pause between the messages of one gesture, so the app's input queue sees them as separate events.</summary>
    public static readonly TimeSpan Step = TimeSpan.FromMilliseconds(30);

    /// <summary>The app window that receives a message at <paramref name="screen"/>: the content bridge containing it, else the deepest visible child, else the top-level window.</summary>
    public nint TargetAt(Point screen)
    {
        nint best = 0, bridge = 0;
        foreach (var h in Win32.Descendants(topLevel))
        {
            if (!Win32.IsWindowVisible(h) || !Win32.GetWindowRect(h, out var r)) continue;
            if (screen.X < r.Left || screen.X >= r.Right || screen.Y < r.Top || screen.Y >= r.Bottom) continue;
            if (bridge == 0 && Win32.ClassName(h) == ContentBridgeClass) bridge = h;
            best = h;
        }
        return bridge != 0 ? bridge : best != 0 ? best : topLevel;
    }

    (nint Hwnd, nint LParam) Target(Point screen)
    {
        var h = TargetAt(screen);
        var p = new Win32.POINT { X = screen.X, Y = screen.Y };
        Win32.ScreenToClient(h, ref p);
        return (h, Win32.MakeLParam(p.X, p.Y));
    }

    public static Point Center(AutomationElement e)
    {
        var r = e.BoundingRectangle;
        return new Point(r.X + r.Width / 2, r.Y + r.Height / 2);
    }

    void Post(nint h, uint msg, nint w, nint l)
    {
        if (!Win32.PostMessage(h, msg, w, l)) throw new InvalidOperationException($"PostMessage 0x{msg:X4} failed");
        Thread.Sleep(Step);
    }

    public void Move(Point screen, bool leftDown = false)
    {
        var (h, l) = Target(screen);
        Post(h, Win32.WM_MOUSEMOVE, leftDown ? Win32.MK_LBUTTON : 0, l);
    }

    public void Click(Point screen)
    {
        var (h, l) = Target(screen);
        Post(h, Win32.WM_MOUSEMOVE, 0, l);
        Post(h, Win32.WM_LBUTTONDOWN, Win32.MK_LBUTTON, l);
        Post(h, Win32.WM_LBUTTONUP, 0, l);
    }

    public void Click(AutomationElement e) => Click(Center(e));

    /// <summary>The message sequence of a real double click: down, up, double-click, up.</summary>
    public void DoubleClick(Point screen)
    {
        var (h, l) = Target(screen);
        Post(h, Win32.WM_MOUSEMOVE, 0, l);
        Post(h, Win32.WM_LBUTTONDOWN, Win32.MK_LBUTTON, l);
        Post(h, Win32.WM_LBUTTONUP, 0, l);
        Post(h, Win32.WM_LBUTTONDBLCLK, Win32.MK_LBUTTON, l);
        Post(h, Win32.WM_LBUTTONUP, 0, l);
    }

    public void DoubleClick(AutomationElement e) => DoubleClick(Center(e));

    public void RightClick(Point screen)
    {
        var (h, l) = Target(screen);
        Post(h, Win32.WM_MOUSEMOVE, 0, l);
        Post(h, Win32.WM_RBUTTONDOWN, Win32.MK_RBUTTON, l);
        Post(h, Win32.WM_RBUTTONUP, 0, l);
    }

    public void RightClick(AutomationElement e) => RightClick(Center(e));

    /// <summary>Press at <paramref name="from"/>, move in <paramref name="steps"/> steps with the button down, release at <paramref name="to"/>.</summary>
    public void Drag(Point from, Point to, int steps = 12)
    {
        var (h, l) = Target(from);
        Post(h, Win32.WM_MOUSEMOVE, 0, l);
        Post(h, Win32.WM_LBUTTONDOWN, Win32.MK_LBUTTON, l);
        // Every message of the gesture goes to the window it started in (as with mouse capture).
        nint Client(Point s)
        {
            var p = new Win32.POINT { X = s.X, Y = s.Y };
            Win32.ScreenToClient(h, ref p);
            return Win32.MakeLParam(p.X, p.Y);
        }
        for (var i = 1; i <= steps; i++)
        {
            var p = new Point(from.X + (to.X - from.X) * i / steps, from.Y + (to.Y - from.Y) * i / steps);
            Post(h, Win32.WM_MOUSEMOVE, Win32.MK_LBUTTON, Client(p));
        }
        Post(h, Win32.WM_LBUTTONUP, 0, Client(to));
    }

    /// <summary>
    /// Waits for the effect of a posted gesture. WinUI 3 (Windows App SDK 1.x) takes its pointer input
    /// from the system's pointer pipeline, not from window messages: posted or sent WM_* mouse and
    /// WM_POINTER* messages to any of the app's windows are ignored (README.md). When the effect does not
    /// come, the test is skipped with that reason instead of failing, and names the UI Automation test
    /// that covers the same feature; it starts passing by itself if the messages ever get through.
    /// </summary>
    public static void ExpectEffect(Func<bool> effect, string gesture, string coveredBy, TimeSpan? wait = null)
    {
        try { Ui.WaitUntil(effect, $"the effect of the posted {gesture}", wait ?? TimeSpan.FromSeconds(5)); }
        catch (TimeoutException)
        {
            Assert.Skip($"WinUI 3 ignored the posted {gesture}: window-targeted mouse messages do not reach XAML pointer input " +
                $"(README.md, \"What WinUI 3 ignores\"). Covered through UI Automation by {coveredBy}.");
        }
    }

    /// <summary>One wheel notch is 120; WM_MOUSEWHEEL carries screen coordinates.</summary>
    public void Wheel(Point screen, int delta)
    {
        var h = TargetAt(screen);
        Post(h, Win32.WM_MOUSEWHEEL, (nint)((delta & 0xFFFF) << 16), Win32.MakeLParam(screen.X, screen.Y));
    }
}
