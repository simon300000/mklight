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
                if (rms.Count > 0) out_.Add(("Max RMS", Fmt.Length(rms.Max())));
                var airy = Json.Nf(Json.At(j, "airy_radius"));
                if (double.IsFinite(airy)) out_.Add(("Airy", Fmt.Length(airy)));
                break;
            case "field_curvature_distortion":
                var dist = (Json.Arr(Json.At(j, "curves")) ?? []).SelectMany(c => Json.NfArray(Json.At(c, "distortion_percent")) ?? [])
                    .Where(double.IsFinite).Select(Math.Abs).ToList();
                if (dist.Count > 0) out_.Add(("Max distortion", Fmt.Fixed(dist.Max(), 3) + " %"));
                break;
            case "grid_distortion":
                if (Json.Number(Json.At(j, "max_abs_distortion_percent")) is double g) out_.Add(("Max distortion", Fmt.Fixed(g, 3) + " %"));
                break;
            case "seidel":
                if (Json.Str(Json.At(j, "status")) == "not_applicable") break;
                foreach (var (label, key) in new[] { ("Σ SI", "s1"), ("Σ SII", "s2"), ("Σ SIII", "s3") })
                    if (Json.Number(Json.At(j, "sums", key)) is double s) out_.Add((label, Fmt.Short(s)));
                break;
            case "chromatic_focal_shift":
                if (Json.Number(Json.At(j, "range")) is double r) out_.Add(("Range", Fmt.Length(r)));
                break;
            case "wavefront_map":
                var m = Json.Arr(Json.At(j, "maps"))?.FirstOrDefault();
                if (Json.Number(Json.At(m, "rms_waves")) is double wr) out_.Add(("RMS", Fmt.Fixed(wr, 4) + " λ"));
                if (Json.Number(Json.At(m, "pv_waves")) is double pv) out_.Add(("P-V", Fmt.Fixed(pv, 4) + " λ"));
                if (Json.Number(Json.At(m, "strehl_at_reference")) is double st) out_.Add(("Strehl", Fmt.Fixed(st, 3)));
                break;
            case "psf":
                if (Json.Number(Json.At(Json.Arr(Json.At(j, "psfs"))?.FirstOrDefault(), "strehl")) is double ps) out_.Add(("Strehl", Fmt.Fixed(ps, 3)));
                break;
            case "ghost_irradiance":
                if (Json.Number(Json.At(j, "total_flux_ratio")) is double gf) out_.Add(("Ghost/primary flux", Fmt.Short(gf)));
                break;
            case "stray_light":
                if (Json.Number(Json.At(j, "stray_fraction")) is double sf) out_.Add(("Stray fraction", Fmt.Short(sf)));
                break;
            case "best_focus":
                if (Json.Number(Json.At(j, "defocus")) is double bf) out_.Add(("Best defocus", Fmt.Length(bf)));
                break;
            case "validate":
                out_.Add(("Errors", (Json.Int(Json.At(j, "errors")) ?? 0).ToString(System.Globalization.CultureInfo.InvariantCulture)));
                out_.Add(("Warnings", (Json.Int(Json.At(j, "warnings")) ?? 0).ToString(System.Globalization.CultureInfo.InvariantCulture)));
                break;
        }
        return out_;
    }

    /// <summary>The footer line: "Max RMS 25.92 µm · Airy 2.87 µm".</summary>
    public static string Text(ResultData d) => string.Join("  ·  ", Of(d).Select(k => k.Label + " " + k.Value));
}
