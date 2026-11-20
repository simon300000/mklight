namespace MKLight.App.Views.Analysis.Results.Readers;

/// <summary>What each first-order entry means, for the label's tooltip (keyed by the engine's stable entry key).</summary>
public static class FirstOrderHelp
{
    static readonly Dictionary<string, string> Descriptions = new()
    {
        ["efl"] = "Effective focal length (of the y section): the distance from the rear principal plane to the paraxial focus for an object at infinity.",
        ["bfl"] = "Back focal length: from the last surface to the paraxial focus.",
        ["ffl"] = "Where the front focal point lies, measured from the first surface.",
        ["image_distance"] = "From the last surface to the paraxial image of the object.",
        ["object_distance"] = "From the object to the first surface.",
        ["entrance_pupil_diameter"] = "Diameter of the image of the stop seen from object space.",
        ["entrance_pupil_position"] = "Where the entrance pupil lies, measured from the first surface.",
        ["exit_pupil_diameter"] = "Diameter of the image of the stop seen from image space.",
        ["exit_pupil_position"] = "Where the exit pupil lies, measured from the image surface.",
        ["image_f_number"] = "Image-space F-number: effective focal length over entrance pupil diameter.",
        ["working_f_number"] = "Paraxial working F-number: 1 / (2 n′ sin θ′) of the paraxial marginal ray at the image, for the actual object distance.",
        ["object_na"] = "Object-space numerical aperture n sin θ of the marginal ray (0 for an object at infinity).",
        ["image_na"] = "Image-space numerical aperture n′ sin θ′ of the marginal ray.",
        ["magnification"] = "Paraxial lateral magnification: image height over object height (0 for an object at infinity).",
        ["lagrange_invariant"] = "The optical invariant of the paraxial marginal and chief rays, a measure of the étendue the system passes.",
        ["total_track"] = "From the first surface to the image surface.",
        ["stop_semi_diameter"] = "Semi-diameter of the aperture stop.",
        ["max_field"] = "The largest field of the design, in its field unit.",
        ["paraxial_image_height"] = "Height of the paraxial chief ray of the largest field at the image.",
        ["image_height_per_tan"] = "Paraxial image height per unit tangent of the field angle.",
        ["petzval_radius"] = "Radius of the Petzval surface, the field curvature of an astigmatism-free system.",
        ["airy_radius"] = "Radius of the first dark ring of the Airy pattern, 1.22 λ × working F-number.",
    };

    /// <summary>The tooltip of an entry: its full label, then what it means when known.</summary>
    public static string Tooltip(string key, string label, string unit)
    {
        var head = string.IsNullOrEmpty(unit) ? label : $"{label} ({unit})";
        return Descriptions.TryGetValue(key, out var d) ? head + "\n" + d : head;
    }

    /// <summary>The description of an entry, or null when it has none.</summary>
    public static string? Description(string key) => Descriptions.GetValueOrDefault(key);
}
