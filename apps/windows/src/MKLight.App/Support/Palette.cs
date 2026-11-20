using Windows.UI;

namespace MKLight.App.Support;

/// <summary>
/// Data colours for plots and the layout, the Mac app's validated Palette.
/// Series use a fixed categorical order (light and dark steps of the same
/// eight hues), never cycled hues. Wavelengths keep their physical meaning:
/// each maps to the categorical slot nearest its spectral colour, so F is
/// blue, d is yellow and C is red everywhere. Magnitude maps use one hue
/// (blue); signed maps use blue ↔ neutral ↔ red.
/// </summary>
public static class Palette
{
    // Categorical slots: blue, orange, aqua, yellow, magenta, green, violet, red.
    static readonly uint[] LightSlots = [0x2a78d6, 0xeb6834, 0x1baf7a, 0xeda100, 0xe87ba4, 0x008300, 0x6250d6, 0xe34948];
    static readonly uint[] DarkSlots = [0x3987e5, 0xd95926, 0x199e70, 0xc98500, 0xd55181, 0x008300, 0x9085e9, 0xe66767];
    static readonly uint[] BlueRamp = [0xcde2fb, 0x9ec5f4, 0x6da7ec, 0x3987e5, 0x256abf, 0x184f95, 0x0d366b];
    static readonly uint[] PerceptualRamp = [0x482475, 0x414487, 0x355f8d, 0x2a788e, 0x21918c, 0x22a884, 0x44bf70, 0x7ad151, 0xbddf26, 0xfde725];

    public static Color Rgb(uint hex, byte a = 255) => Color.FromArgb(a, (byte)(hex >> 16), (byte)(hex >> 8), (byte)hex);

    /// <summary>Categorical slot i (clamped into the eight).</summary>
    public static Color Slot(int i, bool dark)
    {
        var slots = dark ? DarkSlots : LightSlots;
        return Rgb(slots[Math.Clamp(i, 0, slots.Length - 1)]);
    }

    /// <summary>Field i's colour (fields take the slots in order).</summary>
    public static Color Field(int i, bool dark) => Slot(((i % 8) + 8) % 8, dark);

    /// <summary>The slot nearest the wavelength's spectral colour.</summary>
    public static int WavelengthSlot(double um) => (um * 1000) switch
    {
        < 450 => 6, // violet (g line and below)
        < 500 => 0, // blue (F, F')
        < 540 => 2, // aqua
        < 565 => 5, // green (e)
        < 600 => 3, // yellow (d)
        < 630 => 1, // orange
        _ => 7,     // red (C, C') and the infrared
    };

    public static Color Wavelength(double um, bool dark) => Slot(WavelengthSlot(um), dark);

    /// <summary>Magnitude t in 0…1: near zero recedes into the plot surface.</summary>
    public static Color Sequential(double t, bool dark)
    {
        var stops = new List<Color> { dark ? Rgb(0x1a1a19) : Rgb(0xfcfcfb) };
        var ramp = BlueRamp.Select(h => Rgb(h));
        stops.AddRange(dark ? ramp.Reverse() : ramp);
        return Interpolate(stops, t);
    }

    /// <summary>The perceptual ramp (viridis from its 0.1 step).</summary>
    public static Color Perceptual(double t) => Interpolate(PerceptualRamp.Select(h => Rgb(h)).ToList(), t);

    /// <summary>Signed t in −1…1: blue (negative) ↔ neutral ↔ red (positive).</summary>
    public static Color Diverging(double t, bool dark)
    {
        var mid = dark ? Rgb(0x383835) : Rgb(0xf0efec);
        Color[] blue = dark ? [Rgb(0x3987e5), Rgb(0x184f95)] : [Rgb(0x6da7ec), Rgb(0x184f95)];
        Color[] red = dark ? [Rgb(0xe66767), Rgb(0x9b2a29)] : [Rgb(0xee8f8e), Rgb(0xa52b2a)];
        var x = double.IsFinite(t) ? Math.Clamp(t, -1, 1) : 0;
        return x < 0 ? Interpolate((Color[])[mid, .. blue], -x) : Interpolate((Color[])[mid, .. red], x);
    }

    public static Color Interpolate(IReadOnlyList<Color> stops, double t)
    {
        var x = (double.IsFinite(t) ? Math.Clamp(t, 0, 1) : 0) * (stops.Count - 1);
        var i = Math.Min((int)x, stops.Count - 2);
        var f = x - i;
        Color a = stops[i], b = stops[i + 1];
        return Color.FromArgb(255,
            (byte)Math.Round(a.R + (b.R - a.R) * f),
            (byte)Math.Round(a.G + (b.G - a.G) * f),
            (byte)Math.Round(a.B + (b.B - a.B) * f));
    }

    public static Color WithAlpha(this Color c, double alpha) => Color.FromArgb((byte)Math.Round(Math.Clamp(alpha, 0, 1) * 255), c.R, c.G, c.B);
}
