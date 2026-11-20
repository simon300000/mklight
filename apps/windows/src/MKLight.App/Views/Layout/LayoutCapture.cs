using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;

namespace MKLight.App.Views.Layout;

/// <summary>
/// A window capture with its open menus: WinUI opens a menu flyout in a popup window of its own,
/// which a capture of the design window (<see cref="Support.WindowCapture"/>) leaves out. This
/// draws the window (PrintWindow, as composed) and then each visible popup window of this process
/// over it at its place. Window-scoped only: nothing but the app's own windows is read.
/// </summary>
static partial class LayoutCapture
{
    public static void SaveWithPopups(Window window, string path)
    {
        var main = WinRT.Interop.WindowNative.GetWindowHandle(window);
        if (DwmGetWindowAttribute(main, 9 /* DWMWA_EXTENDED_FRAME_BOUNDS */, out var frame, Marshal.SizeOf<RECT>()) != 0) GetWindowRect(main, out frame);
        using var canvas = new Bitmap(frame.Right - frame.Left, frame.Bottom - frame.Top, PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(canvas))
        {
            Draw(g, main, frame, popup: false);
            foreach (var popup in Popups()) Draw(g, popup, frame, popup: true);
        }
        Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(path))!);
        canvas.Save(path, ImageFormat.Png);
    }

    /// <summary>Draws a window as composed at its place relative to <paramref name="origin"/>.</summary>
    static void Draw(Graphics g, nint hwnd, RECT origin, bool popup)
    {
        GetWindowRect(hwnd, out var r);
        int w = r.Right - r.Left, h = r.Bottom - r.Top;
        if (w <= 0 || h <= 0) return;
        using var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
        using (var bg = Graphics.FromImage(bmp))
        {
            var hdc = bg.GetHdc();
            PrintWindow(hwnd, hdc, 2 /* PW_RENDERFULLCONTENT */);
            bg.ReleaseHdc(hdc);
        }
        // A popup's window is larger than its menu (room for the shadow), and what is not menu prints black.
        if (popup) bmp.MakeTransparent(Color.Black);
        g.DrawImageUnscaled(bmp, r.Left - origin.Left, r.Top - origin.Top);
    }

    /// <summary>The visible popup windows (menus, flyouts) of this process.</summary>
    static List<nint> Popups()
    {
        var pid = (uint)Environment.ProcessId;
        var found = new List<nint>();
        EnumWindows((h, _) =>
        {
            GetWindowThreadProcessId(h, out var p);
            if (p != pid || !IsWindowVisible(h)) return true;
            var cls = new char[128];
            var n = GetClassNameW(h, cls, cls.Length);
            if (new string(cls, 0, n).Contains("Popup", StringComparison.Ordinal)) found.Add(h);
            return true;
        }, 0);
        // EnumWindows lists front to back: draw back to front.
        found.Reverse();
        return found;
    }

    delegate bool EnumProc(nint hwnd, nint lParam);

    [StructLayout(LayoutKind.Sequential)]
    struct RECT { public int Left, Top, Right, Bottom; }

    [DllImport("user32.dll")]
    static extern bool EnumWindows(EnumProc callback, nint lParam);

    [DllImport("user32.dll")]
    static extern uint GetWindowThreadProcessId(nint hwnd, out uint pid);

    [DllImport("user32.dll")]
    static extern bool IsWindowVisible(nint hwnd);

    [DllImport("user32.dll", CharSet = CharSet.Unicode)]
    static extern int GetClassNameW(nint hwnd, char[] name, int count);

    [DllImport("user32.dll")]
    static extern bool GetWindowRect(nint hwnd, out RECT rect);

    [DllImport("user32.dll")]
    static extern bool PrintWindow(nint hwnd, nint hdc, uint flags);

    [DllImport("dwmapi.dll")]
    static extern int DwmGetWindowAttribute(nint hwnd, int attribute, out RECT value, int size);
}
