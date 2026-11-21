using MKLight.App.Support;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>The pane footer's key numbers of a result (as the Mac app shows them).</summary>
public static class KeyNumbers
{
    public static IReadOnlyList<(string Label, string Value)> Of(ResultData d)
    {
        var j = d.Json;
        var out_ = new List<(string, string)>();
        switch (d.Analysis)
        {
            case "first_order":
                foreach (var (label, key) in new[] { ("EFL", "efl"), ("F/#", "working_f_number"), ("BFL", "bfl") })
                {
                    var e = (Json.Arr(Json.At(j, "entries")) ?? []).FirstOrDefault(e => Json.Str(Json.At(e, "key")) == key);
                    if (e is not null && Json.Number(Json.At(e, "value")) is double v)
                        out_.Add((label, Fmt.Value(v, Json.Str(Json.At(e, "unit")) ?? "")));
                }
                break;
            case "spot_diagram":
                var rms = (Json.Arr(Json.At(j, "fields")) ?? []).Select(f => Json.Nf(Json.At(f, "rms_radius"))).Where(double.IsFinite).ToList();
                if (rms.Count > 0) out_.Add((Loc.S("Results_KeyNumbers_MaxRms"), Fmt.Length(rms.Max())));
                var airy = Json.Nf(Json.At(j, "airy_radius"));
                if (double.IsFinite(airy)) out_.Add((Loc.S("Results_KeyNumbers_Airy"), Fmt.Length(airy)));
                break;
            case "field_curvature_distortion":
                var dist = (Json.Arr(Json.At(j, "curves")) ?? []).SelectMany(c => Json.NfArray(Json.At(c, "distortion_percent")) ?? [])
                    .Where(double.IsFinite).Select(Math.Abs).ToList();
                if (dist.Count > 0) out_.Add((Loc.S("Results_KeyNumbers_MaxDistortion"), Fmt.Fixed(dist.Max(), 3) + " %"));
                break;
            case "grid_distortion":
                if (Json.Number(Json.At(j, "max_abs_distortion_percent")) is double g) out_.Add((Loc.S("Results_KeyNumbers_MaxDistortion"), Fmt.Fixed(g, 3) + " %"));
                break;
            case "seidel":
                if (Json.Str(Json.At(j, "status")) == "not_applicable") break;
                foreach (var (label, key) in new[] { ("Σ SI", "s1"), ("Σ SII", "s2"), ("Σ SIII", "s3") })
                    if (Json.Number(Json.At(j, "sums", key)) is double s) out_.Add((label, Fmt.Short(s)));
                break;
            case "chromatic_focal_shift":
                if (Json.Number(Json.At(j, "range")) is double r) out_.Add((Loc.S("Results_KeyNumbers_Range"), Fmt.Length(r)));
                break;
            case "wavefront_map":
                var m = Json.Arr(Json.At(j, "maps"))?.FirstOrDefault();
                if (Json.Number(Json.At(m, "rms_waves")) is double wr) out_.Add(("RMS", Fmt.Fixed(wr, 4) + " λ"));
                if (Json.Number(Json.At(m, "pv_waves")) is double pv) out_.Add(("P-V", Fmt.Fixed(pv, 4) + " λ"));
                if (Json.Number(Json.At(m, "strehl_at_reference")) is double st) out_.Add((Loc.S("Results_KeyNumbers_Strehl"), Fmt.Fixed(st, 3)));
                break;
            case "psf":
                if (Json.Number(Json.At(Json.Arr(Json.At(j, "psfs"))?.FirstOrDefault(), "strehl")) is double ps) out_.Add((Loc.S("Results_KeyNumbers_Strehl"), Fmt.Fixed(ps, 3)));
                break;
            case "ghost_irradiance":
                if (Json.Number(Json.At(j, "total_flux_ratio")) is double gf) out_.Add((Loc.S("Results_KeyNumbers_GhostPrimaryFlux"), Fmt.Short(gf)));
                break;
            case "stray_light":
                if (Json.Number(Json.At(j, "stray_fraction")) is double sf) out_.Add((Loc.S("Results_KeyNumbers_StrayFraction"), Fmt.Short(sf)));
                break;
            case "best_focus":
                if (Json.Number(Json.At(j, "defocus")) is double bf) out_.Add((Loc.S("Results_KeyNumbers_BestDefocus"), Fmt.Length(bf)));
                break;
            case "pupil_aberration":
                if (FieldMapData.Read(d) is { } pa)
                {
                    if (pa.ColumnIndex("stop_error_x") is int ex && pa.ColumnIndex("stop_error_y") is int ey)
                    {
                        var e = Enumerable.Range(0, pa.Count).Select(k => Math.Sqrt(Sq(pa.Value(k, ex)) + Sq(pa.Value(k, ey)))).Where(double.IsFinite).ToList();
                        if (e.Count > 0) out_.Add((Loc.S("Results_KeyNumbers_MaxStopError"), Fmt.Length(e.Max())));
                    }
                    var vs = new[] { "variation_x", "variation_y" }.Select(pa.ColumnIndex).OfType<int>()
                        .SelectMany(pa.ColumnValues).Where(double.IsFinite).ToList();
                    if (vs.Count > 0) out_.Add((Loc.S("Results_KeyNumbers_MaxVariation"), Fmt.Signed(vs.MaxBy(Math.Abs) * 100, 2).Trim() + " %"));
                }
                break;
            case "relative_illumination_map":
                if (FieldMapData.Read(d) is { } ri)
                {
                    if (ri.ColumnIndex("relative_illumination") is int c && ri.Range(c) is var (lo, _)) out_.Add((Loc.S("Results_KeyNumbers_MinRi"), Fmt.Fixed(lo, 3)));
                    if (ri.ColumnIndex("vignetting_fraction") is int vc && ri.Range(vc) is var (_, hi) && hi > 0) out_.Add((Loc.S("Results_KeyNumbers_MaxVignetting"), Fmt.Percent(hi)));
                }
                break;
            case "chief_ray_angle":
                if (FieldMapData.Read(d) is { } cra && cra.ColumnIndex("incidence") is int ic && cra.ArgMax(ic) is int k)
                {
                    out_.Add((Loc.S("Results_KeyNumbers_MaxCra"), Fmt.Fixed(cra.Value(k, ic), 2) + "°"));
                    out_.Add((Loc.S("Results_KeyNumbers_At"), Fmt.G3(cra.ImageRadius[k]) + " mm"));
                    if (cra.ColumnIndex("margin") is int mc && cra.Range(mc) is var (mlo, _)) out_.Add((Loc.S("Results_KeyNumbers_MinMargin"), Fmt.Signed(mlo, 2).Trim() + "°"));
                }
                break;
            case "glass_map":
                if (GlassMapData.Read(d) is { } gm)
                {
                    out_.Add((Loc.S("Results_KeyNumbers_Glasses"), gm.Glasses.Count(x => x.Kind == "catalog").ToString(System.Globalization.CultureInfo.InvariantCulture)));
                    out_.Add((Loc.S("Results_KeyNumbers_InDesign"), (gm.Glasses.Count(x => x.InDesign) + gm.Omitted.Count(x => x.InDesign)).ToString(System.Globalization.CultureInfo.InvariantCulture)));
                    if (gm.Omitted.Count > 0) out_.Add((Loc.S("Results_KeyNumbers_NotOnMap"), gm.Omitted.Count.ToString(System.Globalization.CultureInfo.InvariantCulture)));
                }
                break;
            case "glass_info":
                if (GlassInfoData.Read(d) is { } gi)
                {
                    out_.Add(("nd", Fmt.Fixed(gi.Nd, 6)));
                    out_.Add(("Vd", Fmt.Fixed(gi.Vd, 2)));
                }
                break;
            case "image_simulation":
                var nodes = ImageSimulationData.ReadNodes(j);
                if (Json.NfArray(Json.At(j, "pixel_size_mm")) is [var px, ..] && double.IsFinite(px)) out_.Add((Loc.S("Results_KeyNumbers_Pixel"), Fmt.Length(px)));
                if (ImageSimulationData.MaxDistortionPercent(nodes) is double md) out_.Add((Loc.S("Results_KeyNumbers_MaxDistortion"), Fmt.Signed(md, 3).Trim() + " %"));
                if (ImageSimulationData.MinRelativeIllumination(nodes) is double mr) out_.Add((Loc.S("Results_KeyNumbers_MinRi"), Fmt.Fixed(mr, 3)));
                break;
            case "polarization_pupil_map":
                if (Json.Number(Json.At(j, "mean_intensity")) is double mi) out_.Add((Loc.S("Results_KeyNumbers_MeanIntensity"), Fmt.Fixed(mi, 4)));
                if (Json.Number(Json.At(j, "max_diattenuation")) is double mxd) out_.Add((Loc.S("Results_KeyNumbers_MaxDiattenuation"), Fmt.Short(mxd)));
                break;
            case "system_transmission":
                var poly = Json.NfArray(Json.At(j, "polychromatic"))?.Where(double.IsFinite).ToList();
                if (poly is { Count: > 0 }) out_.Add((Loc.S("Results_KeyNumbers_MinTransmission"), Fmt.Percent(poly.Min())));
                break;
            case "ghost_paths":
                var first = Json.Arr(Json.At(j, "ghosts"))?.FirstOrDefault();
                if (first is not null) out_.Add((Loc.S("Results_KeyNumbers_Strongest"), (Json.Str(Json.At(first, "label")) ?? "") + " " + GhostPathsData.Sci(Json.Nf(Json.At(first, "relative_irradiance")))));
                break;
            case "footprint":
                if (Json.Number(Json.At(j, "max_radius")) is double fr) out_.Add((Loc.S("Results_KeyNumbers_MaxRadius"), Fmt.Value(fr, "mm")));
                if (Json.Number(Json.At(j, "semi_diameter")) is double sd) out_.Add((Loc.S("Results_KeyNumbers_Clear"), Fmt.Value(sd, "mm")));
                break;
            case "validate":
                out_.Add((Loc.S("Results_KeyNumbers_Errors"), (Json.Int(Json.At(j, "errors")) ?? 0).ToString(System.Globalization.CultureInfo.InvariantCulture)));
                out_.Add((Loc.S("Results_KeyNumbers_Warnings"), (Json.Int(Json.At(j, "warnings")) ?? 0).ToString(System.Globalization.CultureInfo.InvariantCulture)));
                break;
        }
        return out_;
    }

    static double Sq(double v) => v * v;

    /// <summary>The footer line: "Max RMS 25.92 µm · Airy 2.87 µm".</summary>
    public static string Text(ResultData d) => string.Join("  ·  ", Of(d).Select(k => k.Label + " " + k.Value));

    /// <summary>
    /// The footer line while a candidate is previewed: each number of the candidate with the
    /// design's beside it where it differs ("Max RMS 18.4 µm (design 25.92 µm)  ·  Airy 2.87 µm");
    /// <see cref="Text"/> without a baseline.
    /// </summary>
    public static string Compared(ResultData d)
    {
        if (d.Baseline is not { } design) return Text(d);
        var before = Of(design);
        return string.Join("  ·  ", Of(d).Select(k =>
        {
            var was = before.FirstOrDefault(b => b.Label == k.Label).Value;
            return was is null || was == k.Value ? k.Label + " " + k.Value : Loc.F("Results_KeyNumbers_Compared", k.Label, k.Value, was);
        }));
    }
}
