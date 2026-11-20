using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;

namespace MKLight.App.Support;

/// <summary>
/// Renders a window, as composed on screen (Mica, acrylic and Win2D content
/// included), to a PNG: PrintWindow with PW_RENDERFULLCONTENT, which asks DWM
/// for the composed frame, so the window need not be on top.
/// </summary>
public static partial class WindowCapture
{
    public static void SavePng(Window window, string path)
    {
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(window);
        GetWindowRect(hwnd, out var r);
        // Leave out the invisible resize border DWM reports around the frame.
        if (DwmGetWindowAttribute(hwnd, 9 /* DWMWA_EXTENDED_FRAME_BOUNDS */, out var frame, Marshal.SizeOf<RECT>()) != 0) frame = r;
        int w = r.Right - r.Left, h = r.Bottom - r.Top;
        using var bmp = new System.Drawing.Bitmap(w, h, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        using (var g = System.Drawing.Graphics.FromImage(bmp))
        {
            var hdc = g.GetHdc();
            PrintWindow(hwnd, hdc, 2 /* PW_RENDERFULLCONTENT */);
            g.ReleaseHdc(hdc);
        }
        var crop = new System.Drawing.Rectangle(frame.Left - r.Left, frame.Top - r.Top, frame.Right - frame.Left, frame.Bottom - frame.Top);
        crop.Intersect(new System.Drawing.Rectangle(0, 0, w, h));
        using var cropped = bmp.Clone(crop, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        cropped.Save(path, System.Drawing.Imaging.ImageFormat.Png);
    }

    [StructLayout(LayoutKind.Sequential)]
    struct RECT { public int Left, Top, Right, Bottom; }

    [LibraryImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool GetWindowRect(IntPtr hwnd, out RECT rect);

    [LibraryImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool PrintWindow(IntPtr hwnd, IntPtr hdc, uint flags);

    [LibraryImport("dwmapi.dll")]
    private static partial int DwmGetWindowAttribute(IntPtr hwnd, int attribute, out RECT value, int size);
}
