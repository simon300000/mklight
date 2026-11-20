using System.Runtime.InteropServices;
using Microsoft.UI.Xaml;

namespace MKLight.App.Support;

/// <summary>
/// Lets a window grow beyond its display (for <c>--size</c> screenshots on a
/// small or highly scaled display): Windows caps a window's size at the
/// screen's through WM_GETMINMAXINFO, which this subclass raises.
/// </summary>
public static partial class WindowSizing
{
    const uint WM_GETMINMAXINFO = 0x0024;
    static readonly Dictionary<IntPtr, SubclassProc> Procs = [];

    delegate IntPtr SubclassProc(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam, UIntPtr id, UIntPtr data);

    [StructLayout(LayoutKind.Sequential)]
    struct POINT { public int X, Y; }

    [StructLayout(LayoutKind.Sequential)]
    struct MINMAXINFO { public POINT Reserved, MaxSize, MaxPosition, MinTrackSize, MaxTrackSize; }

    public static void AllowOversize(Window window)
    {
        var hwnd = WinRT.Interop.WindowNative.GetWindowHandle(window);
        if (Procs.ContainsKey(hwnd)) return;
        SubclassProc proc = (h, msg, w, l, _, _) =>
        {
            var r = DefSubclassProc(h, msg, w, l);
            if (msg == WM_GETMINMAXINFO)
            {
                var info = Marshal.PtrToStructure<MINMAXINFO>(l);
                info.MaxTrackSize = new POINT { X = 16000, Y = 16000 };
                Marshal.StructureToPtr(info, l, false);
            }
            return r;
        };
        Procs[hwnd] = proc; // keep the delegate alive
        SetWindowSubclass(hwnd, Marshal.GetFunctionPointerForDelegate(proc), 1, 0);
    }

    [DllImport("comctl32.dll")]
    static extern bool SetWindowSubclass(IntPtr hwnd, IntPtr proc, nuint id, nuint data);

    [DllImport("comctl32.dll")]
    static extern IntPtr DefSubclassProc(IntPtr hwnd, uint msg, IntPtr wParam, IntPtr lParam);
}
