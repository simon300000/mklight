using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace MKLight.UITests;

/// <summary>
/// The test process works in physical pixels, like the app: without this, Windows scales the
/// automation rectangles and window rects it reports to a DPI-unaware process, and the client
/// coordinates of posted mouse messages and window captures would be off by the display scale.
/// </summary>
static partial class DpiAwareness
{
    [ModuleInitializer]
    internal static void Init() => SetProcessDpiAwarenessContext(-4 /* PER_MONITOR_AWARE_V2 */);

    [LibraryImport("user32.dll")]
    [return: MarshalAs(UnmanagedType.Bool)]
    private static partial bool SetProcessDpiAwarenessContext(nint value);
}
