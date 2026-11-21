using MKLight.App.Support;
using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// Display names of the model's kinds that the lens table and the inspector show, in the UI language. The model's
/// own English titles (<c>SolveType.Title()</c> and the like) stay as they are for files, logs and identifiers.
/// </summary>
public static class LensText
{
    public static string Solve(SolveType t) => t switch
    {
        SolveType.MarginalRayHeight => Loc.S("Lens_Solve_MarginalRayHeight"),
        SolveType.MarginalRayAngle => Loc.S("Lens_Solve_MarginalRayAngle"),
        SolveType.ChiefRayHeight => Loc.S("Lens_Solve_ChiefRayHeight"),
        SolveType.ChiefRayAngle => Loc.S("Lens_Solve_ChiefRayAngle"),
        SolveType.FNumber => Loc.S("Lens_Solve_FNumber"),
        SolveType.Pickup => Loc.S("Lens_Solve_Pickup"),
        SolveType.Position => Loc.S("Lens_Solve_Position"),
        SolveType.CenterOfCurvature => Loc.S("Lens_Solve_CenterOfCurvature"),
        _ => Loc.S("Lens_Solve_EdgeThickness"),
    };

    public static string Aperture(ApertureKind k) => k switch
    {
        ApertureKind.EntrancePupilDiameter => Loc.S("Lens_Aperture_EntrancePupilDiameter"),
        ApertureKind.ImageSpaceFNumber => Loc.S("Lens_Aperture_ImageSpaceFNumber"),
        ApertureKind.ObjectSpaceNA => Loc.S("Lens_Aperture_ObjectSpaceNA"),
        ApertureKind.ParaxialWorkingFNumber => Loc.S("Lens_Aperture_ParaxialWorkingFNumber"),
        _ => Loc.S("Lens_Aperture_FloatByStopSize"),
    };

    public static string Field(FieldKind k) => k switch
    {
        FieldKind.ObjectAngle => Loc.S("Lens_FieldKind_ObjectAngle"),
        FieldKind.ObjectHeight => Loc.S("Lens_FieldKind_ObjectHeight"),
        FieldKind.ParaxialImageHeight => Loc.S("Lens_FieldKind_ParaxialImageHeight"),
        _ => Loc.S("Lens_FieldKind_RealImageHeight"),
    };

    /// <summary>A shape parameter's name; the symbols (<c>R_x</c>, <c>k_y</c>) stay symbols.</summary>
    public static string Parameter(ShapeParameter p) => p switch
    {
        ShapeParameter.NormRadius => Loc.S("Lens_Param_NormRadius"),
        ShapeParameter.AstigmatismAngleDeg => Loc.S("Lens_Param_AstigmatismAxis"),
        ShapeParameter.DesignOrder => Loc.S("Lens_Param_DesignOrder"),
        ShapeParameter.DesignWavelengthUm => Loc.S("Lens_Param_DesignWavelength"),
        ShapeParameter.TraceOrder => Loc.S("Lens_Param_TraceOrder"),
        _ => p.Title(),
    };

    public static string Group(SurfaceTypeGroup g) => g switch
    {
        SurfaceTypeGroup.Spherical => Loc.S("Lens_TypeGroup_Spherical"),
        SurfaceTypeGroup.Aspheric => Loc.S("Lens_TypeGroup_Aspheric"),
        SurfaceTypeGroup.Freeform => Loc.S("Lens_TypeGroup_Freeform"),
        SurfaceTypeGroup.Anamorphic => Loc.S("Lens_TypeGroup_Anamorphic"),
        SurfaceTypeGroup.Diffractive => Loc.S("Lens_TypeGroup_Diffractive"),
        _ => Loc.S("Lens_TypeGroup_Special"),
    };
}
