using MKLight.App.Support;

namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>What each first-order entry means, for the label's tooltip (keyed by the engine's stable entry key).</summary>
public static class FirstOrderHelp
{
    /// <summary>The resource key of each entry's description.</summary>
    static readonly Dictionary<string, string> Descriptions = new()
    {
        ["efl"] = "Results_FirstOrderHelp_Efl",
        ["bfl"] = "Results_FirstOrderHelp_Bfl",
        ["ffl"] = "Results_FirstOrderHelp_Ffl",
        ["image_distance"] = "Results_FirstOrderHelp_ImageDistance",
        ["object_distance"] = "Results_FirstOrderHelp_ObjectDistance",
        ["entrance_pupil_diameter"] = "Results_FirstOrderHelp_EntrancePupilDiameter",
        ["entrance_pupil_position"] = "Results_FirstOrderHelp_EntrancePupilPosition",
        ["exit_pupil_diameter"] = "Results_FirstOrderHelp_ExitPupilDiameter",
        ["exit_pupil_position"] = "Results_FirstOrderHelp_ExitPupilPosition",
        ["image_f_number"] = "Results_FirstOrderHelp_ImageFNumber",
        ["working_f_number"] = "Results_FirstOrderHelp_WorkingFNumber",
        ["object_na"] = "Results_FirstOrderHelp_ObjectNa",
        ["image_na"] = "Results_FirstOrderHelp_ImageNa",
        ["magnification"] = "Results_FirstOrderHelp_Magnification",
        ["lagrange_invariant"] = "Results_FirstOrderHelp_LagrangeInvariant",
        ["total_track"] = "Results_FirstOrderHelp_TotalTrack",
        ["stop_semi_diameter"] = "Results_FirstOrderHelp_StopSemiDiameter",
        ["max_field"] = "Results_FirstOrderHelp_MaxField",
        ["paraxial_image_height"] = "Results_FirstOrderHelp_ParaxialImageHeight",
        ["image_height_per_tan"] = "Results_FirstOrderHelp_ImageHeightPerTan",
        ["petzval_radius"] = "Results_FirstOrderHelp_PetzvalRadius",
        ["airy_radius"] = "Results_FirstOrderHelp_AiryRadius",
    };

    /// <summary>The tooltip of an entry: its full label, then what it means when known.</summary>
    public static string Tooltip(string key, string label, string unit)
    {
        var head = string.IsNullOrEmpty(unit) ? label : $"{label} ({unit})";
        return Description(key) is { } d ? head + "\n" + d : head;
    }

    /// <summary>The description of an entry, or null when it has none.</summary>
    public static string? Description(string key) => Descriptions.TryGetValue(key, out var k) ? Loc.S(k) : null;
}
