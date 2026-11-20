using System.Text.Json.Nodes;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>
/// Readers of the line-plot analyses (docs/RESULTS.md §8) into one
/// <see cref="ChartModel"/>. Each returns null when the result does not
/// have the documented shape; the pane then shows it generically.
/// </summary>
public static class LineReaders
{
    /// <summary>The analyses this class reads.</summary>
    public static readonly string[] Analyses =
    [
        "ray_fan", "opd_fan", "field_curvature_distortion", "longitudinal_aberration", "chromatic_focal_shift",
        "lateral_colour", "relative_illumination", "encircled_energy", "diffraction_encircled_energy", "geometric_mtf",
        "mtf", "mtf_vs_field", "through_focus_mtf", "line_spread", "best_focus",
    ];

    public static ChartModel? Read(ResultData d) => d.Analysis switch
    {
        "ray_fan" or "opd_fan" => Fan(d),
        "field_curvature_distortion" => FieldCurvature(d),
        "longitudinal_aberration" => Longitudinal(d),
        "chromatic_focal_shift" => ChromaticFocalShift(d),
        "lateral_colour" => LateralColour(d),
        "relative_illumination" => RelativeIllumination(d),
        "encircled_energy" or "diffraction_encircled_energy" => EncircledEnergy(d),
        "geometric_mtf" or "mtf" => Mtf(d),
        "mtf_vs_field" => MtfVsField(d),
        "through_focus_mtf" => ThroughFocusMtf(d),
        "line_spread" => LineSpread(d),
        "best_focus" => BestFocus(d),
        _ => null,
    };

    /// <summary>
    /// The wavelengths (µm) the result's wavelength indices refer to: its
    /// <c>wavelengths_um</c>, else the system's, completed by the
    /// <c>wavelength</c>/<c>wavelength_um</c> of each entry.
    /// </summary>
    public static double[] WavelengthsUm(ResultData d, IEnumerable<JsonNode?>? entries = null)
    {
        if (Json.NfArray(Json.At(d.Json, "wavelengths_um")) is double[] um) return um;
        var list = d.WavelengthsUm.ToList();
        foreach (var e in entries ?? [])
        {
            if (Json.Int(Json.At(e, "wavelength")) is not int w || w < 0 || w > 4096) continue;
            var v = Json.Nf(Json.At(e, "wavelength_um"));
            if (!double.IsFinite(v)) continue;
            while (list.Count <= w) list.Add(double.NaN);
            list[w] = v;
        }
        return [.. list];
    }

    static SeriesColor WavelengthColor(double[] um, int w) => SeriesColor.Wavelength(w >= 0 && w < um.Length ? um[w] : double.NaN);

    static void WavelengthLegend(ChartModel m, double[] um, IEnumerable<int> wavelengths)
    {
        foreach (var w in wavelengths.Distinct().Order())
            if (w >= 0 && w < um.Length && double.IsFinite(um[w]))
                m.Legend.Add(new LegendEntry(Fmt.Wavelength(um[w]), SeriesColor.Wavelength(um[w])));
    }

    static double[] Scale(double[] v, double k) => v.Select(x => x * k).ToArray();

    static ChartSeries Curve(double[] x, double[] y, SeriesColor c, bool dashed = false)
    {
        var n = Math.Min(x.Length, y.Length);
        return new ChartSeries(x[..n], y[..n], c, dashed);
    }

    // MARK: Fans

    /// <summary>Ray fan (εy, εx in µm) and OPD fan (waves): per field, tangential and sagittal side by side.</summary>
    public static ChartModel? Fan(ResultData d)
    {
        if (Json.Arr(Json.At(d.Json, "fields")) is not { Count: > 0 } list) return null;
        var opd = Json.Str(Json.At(d.Json, "quantity")) == "opd" || d.Analysis == "opd_fan";
        var um = WavelengthsUm(d);
        var scale = opd ? 1.0 : 1000.0;
        var m = new ChartModel { MinGroupHeight = 130 };
        var wavelengths = new List<int>();
        var all = new List<double>();
        for (var k = 0; k < list.Count; k++)
        {
            var f = list[k];
            var index = Json.Int(Json.At(f, "field")) ?? k;
            var g = new ChartGroup { Label = Fmt.FieldTitle(d, index, Json.At(f, "field_value")) };
            foreach (var (key, title, x, y) in new[] { ("tangential", "Tangential", "Py", opd ? "W (waves)" : "εy (µm)"), ("sagittal", "Sagittal", "Px", opd ? "W (waves)" : "εx (µm)") })
            {
                var p = new ChartPanel { Title = title, XTitle = x, YTitle = y, XDomain = (-1, 1) };
                foreach (var c in Json.Arr(Json.At(f, key)) ?? [])
                {
                    var pupil = Json.NfArray(Json.At(c, "pupil"));
                    var values = Json.NfArray(Json.At(c, "values"));
                    if (pupil is null || values is null || pupil.Length != values.Length) continue;
                    var w = Json.Int(Json.At(c, "wavelength")) ?? 0;
                    wavelengths.Add(w);
                    var v = Scale(values, scale);
                    all.AddRange(v.Where(double.IsFinite));
                    p.Series.Add(Curve(pupil, v, WavelengthColor(um, w)));
                }
                g.Panels.Add(p);
            }
            m.Groups.Add(g);
        }
        // One symmetric y range for every fan, so fields compare at a glance.
        var max = Math.Max(all.Select(Math.Abs).DefaultIfEmpty(1).Max(), 1e-9) * 1.08;
        foreach (var p in m.Panels) p.YDomain = (-max, max);
        WavelengthLegend(m, um, wavelengths);
        return m;
    }

    // MARK: Field curvature and distortion

    public static ChartModel? FieldCurvature(ResultData d)
    {
        var field = Json.NfArray(Json.At(d.Json, "field_values"));
        if (field is null || Json.Arr(Json.At(d.Json, "curves")) is not { Count: > 0 } list) return null;
        var unit = Json.Str(Json.At(d.Json, "field_unit")) ?? "";
        var um = WavelengthsUm(d, list);
        var yTitle = unit.Length > 0 ? $"Field ({unit})" : "Field";
        var curvature = new ChartPanel { Title = "Field Curvature", XTitle = "Focus shift (mm)", YTitle = yTitle, PadX = true };
        var distortion = new ChartPanel { Title = "Distortion", XTitle = "Distortion (%)", YTitle = yTitle, PadX = true };
        var wavelengths = new List<int>();
        foreach (var c in list)
        {
            var w = Json.Int(Json.At(c, "wavelength")) ?? 0;
            wavelengths.Add(w);
            var color = WavelengthColor(um, w);
            if (Json.NfArray(Json.At(c, "tangential")) is double[] t) curvature.Series.Add(Curve(t, field, color));
            if (Json.NfArray(Json.At(c, "sagittal")) is double[] s) curvature.Series.Add(Curve(s, field, color, dashed: true));
            if (Json.NfArray(Json.At(c, "distortion_percent")) is double[] dp) distortion.Series.Add(Curve(dp, field, color));
        }
        var m = new ChartModel { MinGroupHeight = 150 };
        var g = new ChartGroup();
        g.Panels.Add(curvature);
        g.Panels.Add(distortion);
        m.Groups.Add(g);
        WavelengthLegend(m, um, wavelengths);
        m.Legend.Add(new LegendEntry("Tangential", SeriesColor.Neutral));
        m.Legend.Add(new LegendEntry("Sagittal", SeriesColor.Neutral, LegendMark.Dashed));
        return m;
    }

    // MARK: Longitudinal aberration, chromatic focal shift, lateral colour

    public static ChartModel? Longitudinal(ResultData d)
    {
        var pupil = Json.NfArray(Json.At(d.Json, "pupil"));
        if (pupil is null || Json.Arr(Json.At(d.Json, "curves")) is not { Count: > 0 } list) return null;
        var um = WavelengthsUm(d, list);
        var bothAxes = list.Select(c => Json.Str(Json.At(c, "axis"))).Distinct().Count() > 1;
        var lo = pupil.Where(double.IsFinite).DefaultIfEmpty(0).Min() < 0 ? -1 : 0;
        var p = new ChartPanel { XTitle = "Longitudinal aberration (mm)", YTitle = "Pupil zone", YDomain = (lo, 1), PadX = true };
        var wavelengths = new List<int>();
        foreach (var c in list)
        {
            var w = Json.Int(Json.At(c, "wavelength")) ?? 0;
            wavelengths.Add(w);
            if (Json.NfArray(Json.At(c, "aberration")) is double[] a)
                p.Series.Add(Curve(a, pupil, WavelengthColor(um, w), dashed: bothAxes && Json.Str(Json.At(c, "axis")) == "x"));
        }
        var m = ChartModel.Single(p);
        WavelengthLegend(m, um, wavelengths);
        if (bothAxes)
        {
            m.Legend.Add(new LegendEntry("y fan", SeriesColor.Neutral));
            m.Legend.Add(new LegendEntry("x fan", SeriesColor.Neutral, LegendMark.Dashed));
        }
        return m;
    }

    public static ChartModel? ChromaticFocalShift(ResultData d)
    {
        var w = Json.NfArray(Json.At(d.Json, "wavelengths_um"));
        var s = Json.NfArray(Json.At(d.Json, "focal_shift"));
        if (w is null || s is null || w.Length != s.Length || w.Length == 0) return null;
        var p = new ChartPanel { XTitle = "Focal shift (µm)", YTitle = "Wavelength (µm)", PadX = true };
        p.Series.Add(Curve(Scale(s, 1000), w, SeriesColor.Field(0)));
        if (Json.NfArray(Json.At(d.Json, "focal_shift_x")) is double[] sx && sx.Length == w.Length && !sx.SequenceEqual(s))
        {
            p.Series.Add(Curve(Scale(sx, 1000), w, SeriesColor.Field(0), dashed: true));
            var m2 = ChartModel.Single(p);
            m2.Legend.Add(new LegendEntry("y", SeriesColor.Field(0)));
            m2.Legend.Add(new LegendEntry("x", SeriesColor.Field(0), LegendMark.Dashed));
            return m2;
        }
        return ChartModel.Single(p);
    }

    public static ChartModel? LateralColour(ResultData d)
    {
        var field = Json.NfArray(Json.At(d.Json, "field_values"));
        if (field is null || Json.Arr(Json.At(d.Json, "curves")) is not { } list) return null;
        var unit = Json.Str(Json.At(d.Json, "field_unit")) ?? "";
        var um = WavelengthsUm(d, list);
        var p = new ChartPanel { XTitle = "Lateral color (µm)", YTitle = unit.Length > 0 ? $"Field ({unit})" : "Field", PadX = true };
        var wavelengths = new List<int>();
        foreach (var c in list)
        {
            var w = Json.Int(Json.At(c, "wavelength")) ?? 0;
            wavelengths.Add(w);
            if (Json.NfArray(Json.At(c, "offset")) is double[] o) p.Series.Add(Curve(Scale(o, 1000), field, WavelengthColor(um, w)));
        }
        var m = ChartModel.Single(p);
        WavelengthLegend(m, um, wavelengths);
        var airy = Json.Nf(Json.At(d.Json, "airy_radius"));
        if (double.IsFinite(airy))
        {
            p.Rules.Add(new ChartRule(airy * 1000));
            p.Rules.Add(new ChartRule(-airy * 1000));
            m.Legend.Add(new LegendEntry("± Airy radius", SeriesColor.Neutral, LegendMark.Dashed));
        }
        return m;
    }

    // MARK: Relative illumination

    public static ChartModel? RelativeIllumination(ResultData d)
    {
        var field = Json.NfArray(Json.At(d.Json, "field_values"));
        var ri = Json.NfArray(Json.At(d.Json, "relative_illumination"));
        if (field is null || ri is null) return null;
        var unit = Json.Str(Json.At(d.Json, "field_unit")) ?? "";
        var p = new ChartPanel { XTitle = unit.Length > 0 ? $"Field ({unit})" : "Field", YTitle = "Fraction", YDomain = (0, 1.05), ZeroLine = false };
        p.Series.Add(Curve(field, ri, SeriesColor.Field(0)));
        var m = ChartModel.Single(p);
        m.Legend.Add(new LegendEntry("Relative illumination", SeriesColor.Field(0)));
        if (Json.NfArray(Json.At(d.Json, "vignetting_fraction")) is { Length: > 0 } v)
        {
            p.Series.Add(Curve(field, v, SeriesColor.Field(1), dashed: true));
            m.Legend.Add(new LegendEntry("Vignetted fraction", SeriesColor.Field(1), LegendMark.Dashed));
        }
        return m;
    }

    // MARK: Encircled energy (geometric and diffraction)

    public static ChartModel? EncircledEnergy(ResultData d)
    {
        if (Json.Arr(Json.At(d.Json, "fields")) is not { Count: > 0 } list) return null;
        var units = ImageUnits.Of(d.Json);
        var radius = Json.Numbers(d, Json.At(d.Json, "radius")) ?? Json.Numbers(d, Json.At(d.Json, "radius_mm"));
        if (radius is null) return null;
        var x = Scale(radius, 1000);
        var p = new ChartPanel { XTitle = $"Radius ({units.Micro})", YTitle = "Encircled energy", YDomain = (0, 1.02), ZeroLine = false };
        var m = ChartModel.Single(p);
        double[]? limit = null;
        for (var k = 0; k < list.Count; k++)
        {
            var f = list[k];
            var enc = Json.Numbers(d, Json.At(f, "encircled"));
            if (enc is null || enc.Length != radius.Length) continue;
            var i = Json.Int(Json.At(f, "field")) ?? k;
            p.Series.Add(Curve(x, enc, SeriesColor.Field(i)));
            var r80 = Json.Number(Json.At(f, "r80")) ?? Json.Nf(Json.At(f, "r80_mm"));
            m.Legend.Add(new LegendEntry(Fmt.FieldTitle(d, i, Json.At(f, "field_value")) + " · r80 " + units.Length(r80), SeriesColor.Field(i)));
            limit ??= Json.Numbers(d, Json.At(f, "diffraction_limit"));
        }
        if (p.Series.Count == 0) return null;
        if (limit is { Length: > 0 })
        {
            p.Series.Insert(0, Curve(x, limit, SeriesColor.Neutral, dashed: true));
            m.Legend.Add(new LegendEntry("Diffraction limit", SeriesColor.Neutral, LegendMark.Dashed));
        }
        return m;
    }

    // MARK: MTF (geometric and wave)

    public static ChartModel? Mtf(ResultData d)
    {
        var j = d.Json;
        var units = ImageUnits.Of(j);
        var freq = Json.Numbers(d, Json.At(j, "frequencies"));
        if (freq is not { Length: > 0 } || !(freq[^1] >= 0)) return null;
        var p = new ChartPanel
        {
            XTitle = $"Spatial frequency ({units.Frequency})", YTitle = "Modulation",
            XDomain = (0, freq[^1] > 0 ? freq[^1] : 1), YDomain = (0, 1.02), ZeroLine = false,
        };
        var m = ChartModel.Single(p, 140);
        var curves = new SortedDictionary<int, (string Label, double[]? T, double[]? S)>();
        double[]? limit = null;
        if (Json.Arr(Json.At(j, "curves")) is JsonArray geometric)
        {
            foreach (var c in geometric)
            {
                var f = Json.Int(Json.At(c, "field")) ?? 0;
                var cur = curves.TryGetValue(f, out var e) ? e : ($"Field {f + 1}", null, null);
                var mod = Json.NfArray(Json.At(c, "modulation"));
                if (Json.Str(Json.At(c, "orientation")) == "sagittal") cur.S = mod; else cur.T = mod;
                curves[f] = cur;
            }
            var values = Json.Arr(Json.At(j, "field_values"));
            var k = 0;
            foreach (var f in curves.Keys.ToList())
            {
                if (values is not null && k < values.Count) curves[f] = curves[f] with { Label = Fmt.FieldTitle(d, f, values[k]) };
                k++;
            }
        }
        else if (Json.Arr(Json.At(j, "fields")) is JsonArray wave)
        {
            for (var k = 0; k < wave.Count; k++)
            {
                var f = wave[k];
                var i = Json.Int(Json.At(f, "field")) ?? k;
                curves[i] = (Fmt.FieldTitle(d, i, Json.At(f, "field_value")), Json.Numbers(d, Json.At(f, "tangential")), Json.Numbers(d, Json.At(f, "sagittal")));
                limit ??= Json.Numbers(d, Json.At(f, "diffraction_limit"));
            }
        }
        else
        {
            return null;
        }
        if (curves.Count == 0) return null;
        if (limit is { Length: > 0 }) p.Series.Add(Curve(freq, limit, SeriesColor.Neutral));
        foreach (var (f, c) in curves)
        {
            if (c.T is not null) p.Series.Add(Curve(freq, c.T, SeriesColor.Field(f)));
            if (c.S is not null) p.Series.Add(Curve(freq, c.S, SeriesColor.Field(f), dashed: true));
            m.Legend.Add(new LegendEntry(c.Label, SeriesColor.Field(f)));
        }
        m.Legend.Add(new LegendEntry("T solid, S dashed", SeriesColor.Neutral, LegendMark.Text));
        if (limit is { Length: > 0 }) m.Legend.Add(new LegendEntry("Diffraction limit", SeriesColor.Neutral));
        var nyquist = Json.NfArray(Json.At(j, "nyquist"))?.FirstOrDefault(double.IsFinite);
        if (nyquist is double n && double.IsFinite(n) && n > 0) p.Rules.Add(new ChartRule(n, "Nyquist"));
        return m;
    }

    static void FrequencyLegend(ChartModel m, IEnumerable<double> frequencies, ImageUnits units)
    {
        var k = 0;
        foreach (var f in frequencies) m.Legend.Add(new LegendEntry(Fmt.Short(f) + " " + units.Frequency, SeriesColor.Field(k++)));
        m.Legend.Add(new LegendEntry("T solid, S dashed", SeriesColor.Neutral, LegendMark.Text));
    }

    public static ChartModel? MtfVsField(ResultData d)
    {
        var j = d.Json;
        var units = ImageUnits.Of(j);
        var field = Json.Numbers(d, Json.At(j, "normalized_field"));
        if (field is null || Json.Arr(Json.At(j, "curves")) is not { Count: > 0 } list) return null;
        var p = new ChartPanel { XTitle = "Normalized field", YTitle = "Modulation", XDomain = (0, 1), YDomain = (0, 1.02), ZeroLine = false };
        var freqs = new List<double>();
        for (var k = 0; k < list.Count; k++)
        {
            var c = list[k];
            freqs.Add(Json.Nf(Json.At(c, "frequency")));
            if (Json.Numbers(d, Json.At(c, "tangential")) is double[] t) p.Series.Add(Curve(field, t, SeriesColor.Field(k)));
            if (Json.Numbers(d, Json.At(c, "sagittal")) is double[] s) p.Series.Add(Curve(field, s, SeriesColor.Field(k), dashed: true));
        }
        var m = ChartModel.Single(p, 140);
        FrequencyLegend(m, freqs, units);
        return m;
    }

    /// <summary>Through-focus MTF: one plot per field (stacked), curves per frequency, the image surface marked.</summary>
    public static ChartModel? ThroughFocusMtf(ResultData d)
    {
        var j = d.Json;
        var units = ImageUnits.Of(j);
        var focus = Json.Numbers(d, Json.At(j, "focus_mm"));
        if (focus is null || Json.Arr(Json.At(j, "fields")) is not { Count: > 0 } list) return null;
        var finite = focus.Where(double.IsFinite).ToArray();
        (double, double)? domain = finite.Length > 1 && finite.Max() > finite.Min() ? (finite.Min(), finite.Max()) : null;
        var m = new ChartModel { MinGroupHeight = 150 };
        var freqs = new List<double>();
        for (var k = 0; k < list.Count; k++)
        {
            var f = list[k];
            var i = Json.Int(Json.At(f, "field")) ?? k;
            var p = new ChartPanel { XTitle = units.FocusAxis, YTitle = "Modulation", XDomain = domain, YDomain = (0, 1.02), ZeroLine = false };
            p.Rules.Add(new ChartRule(0, "Image"));
            var curves = Json.Arr(Json.At(f, "curves")) ?? [];
            for (var c = 0; c < curves.Count; c++)
            {
                if (k == 0) freqs.Add(Json.Nf(Json.At(curves[c], "frequency")));
                if (Json.Numbers(d, Json.At(curves[c], "tangential")) is double[] t) p.Series.Add(Curve(focus, t, SeriesColor.Field(c)));
                if (Json.Numbers(d, Json.At(curves[c], "sagittal")) is double[] s) p.Series.Add(Curve(focus, s, SeriesColor.Field(c), dashed: true));
            }
            var g = new ChartGroup { Label = Fmt.FieldTitle(d, i, Json.At(f, "field_value")) };
            g.Panels.Add(p);
            m.Groups.Add(g);
        }
        FrequencyLegend(m, freqs, units);
        return m;
    }

    // MARK: Line spread

    public static ChartModel? LineSpread(ResultData d)
    {
        if (Json.Arr(Json.At(d.Json, "fields")) is not { Count: > 0 } list) return null;
        var units = ImageUnits.Of(d.Json);
        var p = new ChartPanel { XTitle = $"Position ({units.Micro})", YTitle = $"Line spread ({units.Density})", ZeroLine = false };
        var m = ChartModel.Single(p);
        for (var k = 0; k < list.Count; k++)
        {
            var f = list[k];
            if (Json.Numbers(d, Json.At(f, "positions_mm")) is not double[] pos) continue;
            var i = Json.Int(Json.At(f, "field")) ?? k;
            var x = Scale(pos, 1000);
            if (Json.Numbers(d, Json.At(f, "x")) is double[] lx) p.Series.Add(Curve(x, lx, SeriesColor.Field(i)));
            if (Json.Numbers(d, Json.At(f, "y")) is double[] ly) p.Series.Add(Curve(x, ly, SeriesColor.Field(i), dashed: true));
            m.Legend.Add(new LegendEntry(Fmt.FieldTitle(d, i, Json.At(f, "field_value")), SeriesColor.Field(i)));
        }
        if (p.Series.Count == 0) return null;
        m.Legend.Add(new LegendEntry("x solid, y dashed", SeriesColor.Neutral, LegendMark.Text));
        return m;
    }

    // MARK: Best focus

    /// <summary>The through-focus merit with the best focus marked; null without a through-focus curve.</summary>
    public static ChartModel? BestFocus(ResultData d)
    {
        var x = Json.NfArray(Json.At(d.Json, "through_focus_defocus"));
        var y = Json.NfArray(Json.At(d.Json, "through_focus_merit"));
        if (!Json.Has(d.Json, "defocus") || x is null || y is null || x.Length < 3 || x.Length != y.Length) return null;
        var p = new ChartPanel { XTitle = "Defocus (mm)", YTitle = "RMS radius (µm)", ZeroLine = false };
        p.Series.Add(Curve(x, Scale(y, 1000), SeriesColor.Field(0)));
        p.Rules.Add(new ChartRule(Json.Nf(Json.At(d.Json, "defocus")), "Best"));
        return ChartModel.Single(p);
    }
}
