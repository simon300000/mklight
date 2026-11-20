namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// The <c>ray_grid</c> result (docs/RESULTS.md): image-surface hits of a fields × wavelengths × pupil ray grid,
/// <c>points[F, W, P, 2]</c> (NaN where a ray did not arrive), read per field for a density image.
/// </summary>
public sealed class RayGridData
{
    /// <summary>One field: its title and every arrived hit relative to the hits' centroid (mm), by wavelength.</summary>
    public sealed record Field(string Title, IReadOnlyList<double[]> Xy, double HalfWidth, int Arrived);

    public required IReadOnlyList<Field> Fields { get; init; }
    public required IReadOnlyList<double> WavelengthsUm { get; init; }
    public int Arrived { get; init; }
    public int Launched { get; init; }
    public int PupilCount { get; init; }

    /// <summary>The fields' hits in one length unit: µm unless a half-width reaches 1 mm.</summary>
    public bool Micrometres => Fields.Select(f => f.HalfWidth).Where(double.IsFinite).DefaultIfEmpty(0).Max() < 1;

    public string Length(double mm) => Fmt.Length(mm, Micrometres);

    public static RayGridData? Read(ResultData d)
    {
        var j = d.Json;
        var p = Json.Int(Json.At(j, "pupil_count")) ?? 0;
        var um = Json.NfArray(Json.At(j, "wavelengths_um")) ?? [];
        var fieldIndices = Json.IntArray(Json.At(j, "fields")) ?? [];
        var fieldValues = Json.Arr(Json.At(j, "field_values"));
        var nf = Math.Max(fieldIndices.Length, fieldValues?.Count ?? 0);
        var nw = um.Length;
        if (d.Buffer("points") is not { } pts || p < 1 || nf < 1 || nw < 1 || pts.Count < (long)nf * nw * p * 2) return null;
        var fields = new List<Field>();
        for (var f = 0; f < nf; f++)
        {
            // Centred on the centroid of the field's arrived hits.
            double sx = 0, sy = 0;
            var n = 0;
            for (var i = f * nw * p; i < (f + 1) * nw * p; i++)
            {
                double x = pts[2 * i], y = pts[2 * i + 1];
                if (double.IsFinite(x) && double.IsFinite(y)) { sx += x; sy += y; n++; }
            }
            var (cx, cy) = n > 0 ? (sx / n, sy / n) : (0.0, 0.0);
            var half = 0.0;
            var byWavelength = new List<double[]>();
            for (var w = 0; w < nw; w++)
            {
                var xy = new List<double>(2 * p);
                for (var k = 0; k < p; k++)
                {
                    var i = (f * nw + w) * p + k;
                    double x = pts[2 * i] - cx, y = pts[2 * i + 1] - cy;
                    if (!double.IsFinite(x) || !double.IsFinite(y)) continue;
                    xy.Add(x);
                    xy.Add(y);
                    half = Math.Max(half, Math.Max(Math.Abs(x), Math.Abs(y)));
                }
                byWavelength.Add(xy.ToArray());
            }
            var index = f < fieldIndices.Length ? fieldIndices[f] : f;
            fields.Add(new Field(Fmt.FieldTitle(d, index, fieldValues is { } fv && f < fv.Count ? fv[f] : null), byWavelength,
                n > 0 ? Math.Max(half, 1e-9) : double.NaN, n));
        }
        return new RayGridData
        {
            Fields = fields, WavelengthsUm = um, PupilCount = p,
            Arrived = Json.Int(Json.At(j, "arrived")) ?? fields.Sum(f => f.Arrived),
            Launched = Json.Int(Json.At(j, "launched")) ?? nf * nw * p,
        };
    }

    /// <summary>
    /// The field's hits binned on a <paramref name="bins"/> × <paramref name="bins"/> grid over ±<see cref="Field.HalfWidth"/>
    /// (row 0 at the top, +y up), counts per wavelength: <c>[w][row * bins + col]</c>.
    /// </summary>
    public static int[][] Bin(Field field, int bins)
    {
        var counts = new int[field.Xy.Count][];
        var scale = bins / (2 * field.HalfWidth * 1.04);
        for (var w = 0; w < field.Xy.Count; w++)
        {
            var c = counts[w] = new int[bins * bins];
            var xy = field.Xy[w];
            for (var i = 0; i + 1 < xy.Length; i += 2)
            {
                var col = (int)Math.Floor(xy[i] * scale + bins / 2.0);
                var row = (int)Math.Floor(bins / 2.0 - xy[i + 1] * scale);
                if (col >= 0 && col < bins && row >= 0 && row < bins) c[row * bins + col]++;
            }
        }
        return counts;
    }
}
