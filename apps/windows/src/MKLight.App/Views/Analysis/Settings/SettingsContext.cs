using MKLight.App.Views.Analysis.Results.Readers;

namespace MKLight.App.Views.Analysis.Settings;

/// <summary>What the generated settings controls need from the pane and the system.</summary>
public sealed class SettingsContext
{
    /// <summary>The analysis name; empty takes the descriptor's <c>name</c>.</summary>
    public string Analysis { get; init; } = "";
    /// <summary>Each system field as specified ("0°", "14°", "(3, 4) mm"), by field index.</summary>
    public IReadOnlyList<string> FieldLabels { get; init; } = [];
    /// <summary>The system's wavelengths in µm, by wavelength index.</summary>
    public IReadOnlyList<double> WavelengthsUm { get; init; } = [];
    public int PrimaryWavelength { get; init; }
    /// <summary>
    /// The menu title of each surface by surface index ("1", "3 (stop)",
    /// "Image"); index 0, the object, is not offered.
    /// </summary>
    public IReadOnlyList<string> SurfaceLabels { get; init; } = [];
    /// <summary>The system has configurations (the <c>configuration</c> index is hidden otherwise).</summary>
    public bool HasConfigurations { get; init; }
    /// <summary>The pane's last result, for "Auto (n)": the values automatic settings took.</summary>
    public ResultData? LastResult { get; init; }

    public IReadOnlyList<(int Index, string Title)> FieldOptions =>
        FieldLabels.Select((l, k) => (k, $"{k + 1}: {l}")).ToList();

    public IReadOnlyList<(int Index, string Title)> WavelengthOptions =>
        WavelengthsUm.Select((w, k) => (k, Fmt.Wavelength(w) + (k == PrimaryWavelength ? " (primary)" : ""))).ToList();

    public IReadOnlyList<(int Index, string Title)> SurfaceOptions =>
        SurfaceLabels.Select((l, k) => (k, l)).Skip(1).ToList();

    public IReadOnlyList<(int Index, string Title)> Options(IndexDomain d) => d switch
    {
        IndexDomain.Fields => FieldOptions,
        IndexDomain.Wavelengths => WavelengthOptions,
        _ => SurfaceOptions,
    };
}
