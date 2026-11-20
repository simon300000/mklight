using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using System.Text;
using FlaUI.Core.AutomationElements;

namespace MKLight.UITests;

/// <summary>
/// The app window as composed (PrintWindow with PW_RENDERFULLCONTENT, as the app's own
/// Support/WindowCapture does): never the screen, so it works with the window covered.
/// </summary>
static class WindowImage
{
    public static Bitmap Capture(nint hwnd)
    {
        Win32.GetWindowRect(hwnd, out var r);
        if (Win32.DwmGetWindowAttribute(hwnd, 9 /* DWMWA_EXTENDED_FRAME_BOUNDS */, out var frame, Marshal.SizeOf<Win32.RECT>()) != 0) frame = r;
        var bmp = new Bitmap(r.Width, r.Height, PixelFormat.Format32bppArgb);
        using (var g = Graphics.FromImage(bmp))
        {
            var hdc = g.GetHdc();
            Win32.PrintWindow(hwnd, hdc, 2 /* PW_RENDERFULLCONTENT */);
            g.ReleaseHdc(hdc);
        }
        var crop = new Rectangle(frame.Left - r.Left, frame.Top - r.Top, frame.Width, frame.Height);
        crop.Intersect(new Rectangle(0, 0, r.Width, r.Height));
        using (bmp) return bmp.Clone(crop, PixelFormat.Format32bppArgb);
    }

    /// <summary>The part of a window capture under <paramref name="screen"/> (screen pixels).</summary>
    public static Bitmap CaptureRegion(nint hwnd, Rectangle screen)
    {
        Win32.GetWindowRect(hwnd, out var r);
        if (Win32.DwmGetWindowAttribute(hwnd, 9, out var frame, Marshal.SizeOf<Win32.RECT>()) != 0) frame = r;
        using var all = Capture(hwnd);
        var local = new Rectangle(screen.X - frame.Left, screen.Y - frame.Top, screen.Width, screen.Height);
        local.Intersect(new Rectangle(0, 0, all.Width, all.Height));
        return all.Clone(local, PixelFormat.Format32bppArgb);
    }

    /// <summary>The share of pixels (0…1) that differ by more than <paramref name="tolerance"/> in any channel.</summary>
    public static double Difference(Bitmap a, Bitmap b, int tolerance = 24)
    {
        if (a.Size != b.Size) return 1;
        var ra = a.LockBits(new Rectangle(Point.Empty, a.Size), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        var rb = b.LockBits(new Rectangle(Point.Empty, b.Size), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
        try
        {
            var pa = new byte[ra.Stride * a.Height];
            var pb = new byte[rb.Stride * b.Height];
            Marshal.Copy(ra.Scan0, pa, 0, pa.Length);
            Marshal.Copy(rb.Scan0, pb, 0, pb.Length);
            long changed = 0;
            for (var i = 0; i + 3 < pa.Length; i += 4)
                if (Math.Abs(pa[i] - pb[i]) > tolerance || Math.Abs(pa[i + 1] - pb[i + 1]) > tolerance || Math.Abs(pa[i + 2] - pb[i + 2]) > tolerance) changed++;
            return changed / (double)(a.Width * a.Height);
        }
        finally
        {
            a.UnlockBits(ra);
            b.UnlockBits(rb);
        }
    }
}

/// <summary>A readable dump of an automation subtree (failure artefacts, and finding names while writing tests).</summary>
static class TreeDump
{
    public static string Of(AutomationElement root, int maxDepth = 40)
    {
        var sb = new StringBuilder();
        var walker = root.Automation.TreeWalkerFactory.GetControlViewWalker();
        void Walk(AutomationElement e, int depth)
        {
            sb.Append(' ', depth * 2).AppendLine(Describe(e));
            if (depth >= maxDepth) return;
            for (var c = walker.GetFirstChild(e); c is not null; c = walker.GetNextSibling(c)) Walk(c, depth + 1);
        }
        Walk(root, 0);
        return sb.ToString();
    }

    public static string Describe(AutomationElement e)
    {
        string Try(Func<string?> f) { try { return f() ?? ""; } catch (Exception) { return "?"; } }
        var type = Try(() => e.Properties.ControlType.ValueOrDefault.ToString());
        var name = Try(() => e.Properties.Name.ValueOrDefault);
        var id = Try(() => e.Properties.AutomationId.ValueOrDefault);
        var cls = Try(() => e.Properties.ClassName.ValueOrDefault);
        var status = Try(() => e.Properties.ItemStatus.ValueOrDefault);
        var help = Try(() => e.Properties.HelpText.ValueOrDefault);
        var r = Try(() => e.Properties.BoundingRectangle.ValueOrDefault.ToString());
        var value = Try(() => e.Patterns.Value.IsSupported ? e.Patterns.Value.Pattern.Value.ValueOrDefault : null);
        return $"{type} \"{name}\" id={id} class={cls}" + (status.Length > 0 ? $" status=\"{status}\"" : "") + (help.Length > 0 ? $" help=\"{help.Replace('\n', ' ')}\"" : "")
            + (value.Length > 0 ? $" value=\"{value}\"" : "") + $" {r}";
    }
}
