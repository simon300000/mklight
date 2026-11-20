namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary><c>psf</c>: one PSF image per field (and wavelength when monochromatic), in µm about its grid centre.</summary>
public sealed class PsfData
{
    public sealed record Item(string Title, MapGrid Grid, double Strehl, double SpacingMm, int Size, double Peak, string Normalization);

    public required IReadOnlyList<Item> Items { get; init; }
    public ImageUnits Units { get; init; }

    public static PsfData? Read(ResultData d)
    {
        var list = Json.Arr(Json.At(d.Json, "psfs"));
        if (list is null || list.Count == 0) return null;
        var items = new List<Item>();
        foreach (var p in list)
        {
            var psf = Json.At(p, "psf");
            var h = Json.Nf(Json.At(psf, "spacing_mm"));
            var buffer = Json.BufferRef(d, Json.At(psf, "data"));
            if (buffer is null || !(h > 0)) continue;
            var ny = buffer.Shape.Count >= 2 ? buffer.Shape[^2] : 0;
            var nx = buffer.Shape.Count >= 2 ? buffer.Shape[^1] : 0;
            double cx = 0, cy = 0;
            if (Json.NfArray(Json.At(psf, "center_mm")) is [var ax, var ay])
            {
                if (double.IsFinite(ax)) cx = ax;
                if (double.IsFinite(ay)) cy = ay;
            }
            // Coordinate i is centre + (i − floor(n/2)) × spacing; the grid spans the cells' centres (µm).
            var x0 = (cx - nx / 2 * h) * 1000;
            var y0 = (cy - ny / 2 * h) * 1000;
            var grid = MapGrid.From(buffer, 0, x0, x0 + (nx - 1) * h * 1000, y0, y0 + (ny - 1) * h * 1000);
            if (grid is null) continue;
            var field = Json.Int(Json.At(p, "field")) ?? 0;
            var wl = Json.Number(Json.At(psf, "wavelength_um"));
            var title = Fmt.FieldTitle(d, field, Json.At(p, "field_value"))
                + (wl is double w && double.IsFinite(w) ? " · " + Fmt.Wavelength(w) : " · polychromatic");
            var peak = grid.Range()?.Hi ?? double.NaN;
            items.Add(new Item(title, grid, Json.Nf(Json.At(p, "strehl")), h, Json.Int(Json.At(psf, "size")) ?? nx, peak,
                Json.Str(Json.At(psf, "normalization")) ?? "strehl"));
        }
        return items.Count == 0 ? null : new PsfData { Items = items, Units = ImageUnits.Of(d.Json) };
    }

    /// <summary>
    /// The part of a PSF worth looking at: the smallest square about the intensity centroid
    /// holding <paramref name="fraction"/> of the energy, widened by a quarter (and at least
    /// 8 cells each side), so the core and its first rings fill the view instead of a dot in
    /// a wide sampling window. The whole grid when that is not smaller.
    /// </summary>
    public static MapGrid Focus(MapGrid g, double fraction = 0.995)
    {
        double total = 0, sx = 0, sy = 0;
        for (var j = 0; j < g.Ny; j++)
            for (var i = 0; i < g.Nx; i++)
            {
                var v = g.Values[j * g.Nx + i];
                if (!double.IsFinite(v) || v <= 0) continue;
                total += v; sx += v * i; sy += v * j;
            }
        if (!(total > 0)) return g;
        int ci = (int)Math.Round(sx / total), cj = (int)Math.Round(sy / total);
        var maxR = Math.Max(Math.Max(ci, g.Nx - 1 - ci), Math.Max(cj, g.Ny - 1 - cj));
        var ring = new double[maxR + 1];
        for (var j = 0; j < g.Ny; j++)
            for (var i = 0; i < g.Nx; i++)
            {
                var v = g.Values[j * g.Nx + i];
                if (double.IsFinite(v) && v > 0) ring[Math.Max(Math.Abs(i - ci), Math.Abs(j - cj))] += v;
            }
        double acc = 0;
        var h = maxR;
        for (var r = 0; r <= maxR; r++)
        {
            acc += ring[r];
            if (acc >= fraction * total) { h = r; break; }
        }
        var half = Math.Max((int)Math.Ceiling(h * 1.25) + 2, 8);
        int i0 = Math.Max(ci - half, 0), i1 = Math.Min(ci + half, g.Nx - 1), j0 = Math.Max(cj - half, 0), j1 = Math.Min(cj + half, g.Ny - 1);
        if (i1 - i0 + 1 >= g.Nx && j1 - j0 + 1 >= g.Ny) return g;
        int nx = i1 - i0 + 1, ny = j1 - j0 + 1;
        var v2 = new double[nx * ny];
        for (var j = 0; j < ny; j++) Array.Copy(g.Values, (j0 + j) * g.Nx + i0, v2, j * nx, nx);
        double dx = g.Nx > 1 ? (g.X1 - g.X0) / (g.Nx - 1) : 1, dy = g.Ny > 1 ? (g.Y1 - g.Y0) / (g.Ny - 1) : 1;
        return new MapGrid(nx, ny, v2) { X0 = g.X0 + i0 * dx, X1 = g.X0 + i1 * dx, Y0 = g.Y0 + j0 * dy, Y1 = g.Y0 + j1 * dy };
    }
}

/// <summary><c>wavefront_map</c>: OPD maps (waves) over the normalised pupil with their statistics, Zernike fit and rim.</summary>
public sealed class WavefrontData
{
    public sealed record Map(string Title, MapGrid Grid, double Rms, double Pv, double Strehl, double[] Zernike, string Ordering, IReadOnlyList<(double U, double V)[]> Rim);

    public required IReadOnlyList<Map> Maps { get; init; }

    public static readonly string[] FringeNames =
        ["Piston", "Tilt x", "Tilt y", "Defocus", "Astig 0°", "Astig 45°", "Coma x", "Coma y", "Spherical", "Trefoil x", "Trefoil y"];

    public static WavefrontData? Read(ResultData d)
    {
        var list = Json.Arr(Json.At(d.Json, "maps"));
        if (list is null || list.Count == 0) return null;
        var maps = new List<Map>();
        foreach (var m in list)
        {
            var b = Json.BufferRef(d, Json.At(m, "opd_waves"));
            if (b is null || b.Shape.Count != 2) continue;
            // Cell centres over [−1, 1]: the coordinates buffer when present, else uniform cells.
            int nu = b.Shape[1], nv = b.Shape[0];
            var coords = Json.Numbers(d, Json.At(m, "coordinates"));
            double u0 = -1 + 1.0 / nu, u1 = 1 - 1.0 / nu;
            if (coords is { Length: > 1 } && double.IsFinite(coords[0]) && double.IsFinite(coords[^1]) && coords.Length == nu) (u0, u1) = (coords[0], coords[^1]);
            var v0 = nv == nu ? u0 : -1 + 1.0 / nv;
            var v1 = nv == nu ? u1 : 1 - 1.0 / nv;
            var grid = MapGrid.From(b, 0, u0, u1, v0, v1);
            if (grid is null) continue;
            var field = Json.Int(Json.At(m, "field")) ?? 0;
            var title = Fmt.FieldTitle(d, field, Json.At(m, "field_value")) + " · " + Fmt.Wavelength(Json.Nf(Json.At(m, "wavelength_um")));
            var rim = new List<(double, double)[]>();
            foreach (var r in Json.Arr(Json.At(m, "rim")) ?? [])
                if (Json.BufferRef(d, r) is { Shape: [_, 2] } rb)
                    rim.Add(Enumerable.Range(0, rb.Count / 2).Select(k => (rb[2 * k], rb[2 * k + 1])).ToArray());
            maps.Add(new Map(title, grid, Json.Nf(Json.At(m, "rms_waves")), Json.Nf(Json.At(m, "pv_waves")), Json.Nf(Json.At(m, "strehl_at_reference")),
                Json.Numbers(d, Json.At(m, "zernike", "coefficients")) ?? [], Json.Str(Json.At(m, "zernike", "ordering")) ?? "fringe", rim));
        }
        return maps.Count == 0 ? null : new WavefrontData { Maps = maps };
    }
}
