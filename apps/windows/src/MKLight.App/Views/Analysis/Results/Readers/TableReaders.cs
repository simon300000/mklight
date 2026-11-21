using System.Text.Json.Nodes;
using MKLight.App.Support;
using MKLight.Model;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary><c>first_order</c>: the keyed report and, when the engine sends it, the x / y first order.</summary>
public sealed class FirstOrderData
{
    public sealed record Entry(string Key, string Label, double Value, string Unit)
    {
        public string Formatted => Fmt.Value(Value, Unit);
    }

    public sealed record DoePower(int Surface, double Power);

    public sealed record XyRow(int Wavelength, double WavelengthUm, double EflX, double EflY, double BflX, double BflY, double Squeeze, IReadOnlyList<DoePower> Doe);

    public required IReadOnlyList<Entry> Entries { get; init; }
    public double WavelengthUm { get; init; } = double.NaN;
    /// <summary>"rotational", "meridional" or "none"; null when the engine does not say.</summary>
    public string? Symmetry { get; init; }
    public int? PrimaryWavelength { get; init; }
    public string? SqueezeDefinition { get; init; }
    public IReadOnlyList<string> Warnings { get; init; } = [];
    public IReadOnlyList<XyRow> Rows { get; init; } = [];

    public static FirstOrderData? Read(JsonNode? j)
    {
        if (Json.Arr(Json.At(j, "entries")) is not { } list) return null;
        var entries = new List<Entry>();
        foreach (var e in list)
        {
            if (Json.Str(Json.At(e, "key")) is not { } key || Json.Number(Json.At(e, "value")) is not double v) continue;
            entries.Add(new Entry(key, Json.Str(Json.At(e, "label")) ?? key, v, Json.Str(Json.At(e, "unit")) ?? ""));
        }
        if (entries.Count == 0) return null;
        var xy = Json.At(j, "paraxial_xy");
        var rows = new List<XyRow>();
        var k = 0;
        foreach (var w in Json.Arr(Json.At(xy, "wavelengths")) ?? [])
        {
            var i = k++;
            if (Json.Number(Json.At(w, "efl_y")) is not double efly) continue;
            var eflx = Json.Nf(Json.At(w, "efl_x"));
            var doe = (Json.Arr(Json.At(w, "doe_power")) ?? [])
                .Where(d => Json.Int(Json.At(d, "surface")) is not null && Json.Number(Json.At(d, "power")) is not null)
                .Select(d => new DoePower(Json.Int(Json.At(d, "surface"))!.Value, Json.Nf(Json.At(d, "power")))).ToList();
            rows.Add(new XyRow(Json.Int(Json.At(w, "wavelength")) ?? i, Json.Nf(Json.At(w, "wavelength_um")), eflx, efly,
                Json.Nf(Json.At(w, "bfl_x")), Json.Nf(Json.At(w, "bfl_y")),
                Json.Number(Json.At(w, "squeeze")) ?? (eflx != 0 ? efly / eflx : double.NaN), doe));
        }
        return new FirstOrderData
        {
            Entries = entries,
            WavelengthUm = Json.Nf(Json.At(j, "wavelength_um")),
            Symmetry = Json.Str(Json.At(xy, "symmetry")) ?? Json.Str(Json.At(j, "symmetry")),
            PrimaryWavelength = Json.Int(Json.At(xy, "primary_wavelength")),
            SqueezeDefinition = Json.Str(Json.At(xy, "squeeze_definition")),
            Warnings = (Json.Arr(Json.At(xy, "warnings")) ?? []).Select(Json.Str).OfType<string>().ToList(),
            Rows = rows,
        };
    }

    /// <summary>Whether the design lacks rotational symmetry (x and y first order differ); false when the engine does not say.</summary>
    public bool IsAnamorphic => Symmetry is not null and not "rotational";

    public XyRow? Primary => Rows.FirstOrDefault(r => r.Wavelength == PrimaryWavelength) ?? Rows.FirstOrDefault();
    public IEnumerable<XyRow> Others => Rows.Where(r => r != Primary);

    public string SymmetryTitle => Symmetry switch
    {
        "rotational" => Loc.S("Results_FirstOrder_SymmetryRotational"),
        "meridional" => Loc.S("Results_FirstOrder_SymmetryMeridional"),
        "none" => Loc.S("Results_FirstOrder_SymmetryNone"),
        _ => EngineText.T(Symmetry ?? ""),
    };

    /// <summary>"1.5000×".</summary>
    public static string SqueezeText(double v) => double.IsFinite(v) ? Fmt.Fixed(v, 4) + "×" : Fmt.Short(v);
}

/// <summary><c>seidel</c>: per-surface and summed coefficients (mm), wavefront coefficients and the Petzval radius.</summary>
public sealed class SeidelData
{
    /// <summary>The terms with <c>_</c> subscripts (set as real subscripts), by key.</summary>
    public static (string Key, string Label)[] Terms =>
    [
        ("s1", Loc.S("Results_Seidel_S1")), ("s2", Loc.S("Results_Seidel_S2")), ("s3", Loc.S("Results_Seidel_S3")),
        ("s4", Loc.S("Results_Seidel_S4")), ("s5", Loc.S("Results_Seidel_S5")),
    ];
    public static (string Key, string Label)[] Chromatic => [("cl", Loc.S("Results_Seidel_Cl")), ("ct", Loc.S("Results_Seidel_Ct"))];

    public required IReadOnlyList<string> Surfaces { get; init; }
    /// <summary>Values[key][k]: surface k's coefficient; the last entry is the sum (Σ).</summary>
    public required IReadOnlyDictionary<string, double[]> Values { get; init; }
    public IReadOnlyList<(string Key, double Value)> Wavefront { get; init; } = [];
    public double PetzvalRadius { get; init; } = double.NaN;
    /// <summary>Why the design has no coefficients (not rotationally symmetric); null when it has them.</summary>
    public string? NotApplicable { get; init; }

    public static SeidelData? Read(JsonNode? j)
    {
        if (Json.Str(Json.At(j, "status")) == "not_applicable")
            return new SeidelData { Surfaces = [], Values = new Dictionary<string, double[]>(), NotApplicable = Json.Str(Json.At(j, "reason")) ?? "" };
        if (Json.Arr(Json.At(j, "surfaces")) is not { } list || Json.Obj(Json.At(j, "sums")) is not { } sums) return null;
        var surfaces = list.Select((s, k) => (Json.Int(Json.At(s, "surface")) ?? k).ToString(System.Globalization.CultureInfo.InvariantCulture)).Append("Σ").ToList();
        var values = new Dictionary<string, double[]>();
        foreach (var (key, _) in Terms.Concat(Chromatic))
            values[key] = list.Select(s => Json.Nf(Json.At(s, key))).Append(Json.Nf(sums[key])).ToArray();
        var wf = (Json.Obj(Json.At(j, "wavefront_waves")) ?? [])
            .OrderBy(p => p.Key, StringComparer.Ordinal)
            .Where(p => Json.Number(p.Value) is not null)
            .Select(p => (WavefrontName(p.Key), Json.Nf(p.Value))).ToList();
        return new SeidelData { Surfaces = surfaces, Values = values, Wavefront = wf, PetzvalRadius = Json.Nf(Json.At(j, "petzval_radius")) };
    }

    /// <summary>"w220p" → "W_220P".</summary>
    static string WavefrontName(string key) => key.Length > 1 && key[0] == 'w' ? "W_" + key[1..].ToUpperInvariant() : key.ToUpperInvariant();
}

/// <summary><c>glass_info</c>: one glass's catalogue data, indices and dispersion curve.</summary>
public sealed class GlassInfoData
{
    public sealed record Index(double Um, double N, double NCatalog, double AbsorptionPerMm, bool InRange);

    public required string Name { get; init; }
    public string Catalog { get; init; } = "";
    public double Nd { get; init; }
    public double Ne { get; init; }
    public double Vd { get; init; }
    public double Pgf { get; init; }
    public double CatalogNd { get; init; }
    public double CatalogVd { get; init; }
    public double Dnd { get; init; }
    public double Dvd { get; init; }
    public double MinUm { get; init; }
    public double MaxUm { get; init; }
    public string? Formula { get; init; }
    public bool HasThermalData { get; init; }
    public double? Cte { get; init; }
    public double? Density { get; init; }
    public double? RelativeCost { get; init; }
    public string? Comment { get; init; }
    public double TemperatureC { get; init; }
    public double PressureAtm { get; init; }
    public IReadOnlyList<Index> Indices { get; init; } = [];
    /// <summary>(λ µm, n) over the glass's range; empty when not requested.</summary>
    public IReadOnlyList<(double Um, double N)> Curve { get; init; } = [];

    public bool HasOffset => Dnd != 0 || Dvd != 0;

    public static GlassInfoData? Read(ResultData d)
    {
        var j = d.Json;
        if (Json.Str(Json.At(j, "name")) is not { } name || Json.Number(Json.At(j, "nd")) is not double nd) return null;
        double? Defined(string k) => Json.Number(Json.At(j, k)) is double v && double.IsFinite(v) ? v : null;
        var curve = new List<(double, double)>();
        if (d.Buffer("dispersion_curve") is { Shape: [_, 2] } b)
            for (var i = 0; i + 1 < b.Count; i += 2) curve.Add((b[i], b[i + 1]));
        var comment = Json.Str(Json.At(j, "comment"));
        return new GlassInfoData
        {
            Name = name,
            Catalog = Json.Str(Json.At(j, "catalog")) ?? "",
            Nd = nd,
            Ne = Json.Nf(Json.At(j, "ne")),
            Vd = Json.Nf(Json.At(j, "vd")),
            Pgf = Json.Nf(Json.At(j, "pgf")),
            CatalogNd = Json.Nf(Json.At(j, "catalog_nd")),
            CatalogVd = Json.Nf(Json.At(j, "catalog_vd")),
            Dnd = Json.Number(Json.At(j, "offset", "dn_d")) ?? 0,
            Dvd = Json.Number(Json.At(j, "offset", "dv_d")) ?? 0,
            MinUm = Json.Nf(Json.At(j, "min_um")),
            MaxUm = Json.Nf(Json.At(j, "max_um")),
            Formula = Json.Str(Json.At(j, "dispersion", "formula")) is { } f ? FormulaTitle(f) : null,
            HasThermalData = Json.Bool(Json.At(j, "has_thermal_data")) ?? false,
            Cte = Defined("cte"),
            Density = Defined("density"),
            RelativeCost = Defined("relative_cost"),
            Comment = string.IsNullOrWhiteSpace(comment) ? null : comment.Trim(),
            TemperatureC = Json.Nf(Json.At(j, "environment", "temperature_c")),
            PressureAtm = Json.Nf(Json.At(j, "environment", "pressure_atm")),
            Indices = (Json.Arr(Json.At(j, "indices")) ?? [])
                .Where(r => Json.Number(Json.At(r, "wavelength_um")) is not null)
                .Select(r => new Index(Json.Nf(Json.At(r, "wavelength_um")), Json.Nf(Json.At(r, "n")), Json.Nf(Json.At(r, "n_catalog")),
                    Json.Nf(Json.At(r, "absorption_per_mm")), Json.Bool(Json.At(r, "in_range")) ?? true)).ToList(),
            Curve = curve,
        };
    }

    public static string FormulaTitle(string key) => key switch
    {
        "schott" => "Schott",
        "sellmeier" => "Sellmeier",
        "sellmeier2" => "Sellmeier 2",
        "herzberger" => "Herzberger",
        "conrady" => "Conrady",
        "extended" => Loc.S("Results_Glass_FormulaExtended"),
        "cauchy" => "Cauchy",
        "tabulated" => Loc.S("Results_Glass_FormulaTabulated"),
        _ => EngineText.T(GenericData.Title(key)),
    };
}
