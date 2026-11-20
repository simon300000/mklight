using System.Numerics;

namespace MKLight.App.ViewModels.Jobs;

/// <summary>
/// Frames the layouts of a design and its candidate on one scale (the job pages' before / after thumbnails), so a
/// thicker lens or a longer back focus shows as such: each layout's frame (everything, or a long slim layout's lens
/// with half its length either side; symmetric about the axis), their union, and the scale and centre that fit the
/// union into a thumbnail. In mm, X along the axis, Y up.
/// </summary>
public static class LayoutThumbnailFit
{
    public readonly record struct Box(Vector2 Min, Vector2 Max)
    {
        public Vector2 Size => Max - Min;
        public Vector2 Center => (Min + Max) / 2;
    }

    /// <summary>
    /// One layout's frame from the bounds of everything drawn and of the lens alone: a layout far longer than tall
    /// whose lens is a small part of it (an object far away) is framed on its lens; the frame is symmetric about the axis.
    /// </summary>
    public static Box Frame(Box all, Box lens)
    {
        var min = all.Min;
        var max = all.Max;
        var span = all.Size;
        var lensLength = lens.Max.X - lens.Min.X;
        if (span.X > 12 * MathF.Max(span.Y, 1e-6f) && lensLength * 5 < span.X)
            min = new Vector2(lens.Min.X - lensLength * 0.5f, min.Y);
        var h = MathF.Max(MathF.Abs(min.Y), MathF.Abs(max.Y));
        return new Box(new Vector2(min.X, -h), new Vector2(max.X, h));
    }

    /// <summary>The smallest box holding every frame (null for none).</summary>
    public static Box? Union(IEnumerable<Box> frames)
    {
        Box? u = null;
        foreach (var f in frames)
            u = u is { } b ? new Box(Vector2.Min(b.Min, f.Min), Vector2.Max(b.Max, f.Max)) : f;
        return u;
    }

    /// <summary>The scale (px per mm) and centre (mm) that fit <paramref name="frame"/> into a width × height thumbnail with margins.</summary>
    public static (float Scale, Vector2 Center) Fit(Box frame, float width, float height, float marginX, float marginY)
    {
        var size = frame.Size;
        var sx = MathF.Max(width - 2 * marginX, 1) / MathF.Max(size.X, 1e-6f);
        var sy = MathF.Max(height - 2 * marginY, 1) / MathF.Max(size.Y, 1e-6f);
        return (MathF.Max(1e-4f, MathF.Min(sx, sy)), frame.Center);
    }

    /// <summary>A point in mm to the thumbnail's pixels (y down), for a fit.</summary>
    public static Vector2 ToScreen(Vector2 p, (float Scale, Vector2 Center) fit, float width, float height) =>
        new((p.X - fit.Center.X) * fit.Scale + width / 2, -(p.Y - fit.Center.Y) * fit.Scale + height / 2);
}
