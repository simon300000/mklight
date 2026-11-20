using MKLight.Model;

namespace MKLight.App.ViewModels;

/// <summary>
/// A user edit of the lens data, addressed by surface id where possible (the Mac app's <c>LensEdit</c>).
/// Applying it is pure model manipulation: <see cref="Apply"/> returns the new system, or <c>null</c> when
/// the edit is refused. The table applies it as one undoable step,
/// <c>Document.Edit(edit.ActionName, edit.Apply)</c>.
/// </summary>
public abstract record LensEdit
{
    public sealed record SetCurvature(SurfaceID Surface, double Curvature) : LensEdit;
    public sealed record SetThickness(SurfaceID Surface, double Thickness) : LensEdit;
    public sealed record SetConic(SurfaceID Surface, double Conic) : LensEdit;
    public sealed record SetSemiDiameter(SurfaceID Surface, SemiDiameter Value) : LensEdit;
    public sealed record SetMaterial(SurfaceID Surface, MaterialEdit Material) : LensEdit;
    public sealed record SetLabel(SurfaceID Surface, string Label) : LensEdit;
    /// <summary>A type-menu choice: a model type, or the cylinder preset of a biconic.</summary>
    public sealed record SetType(SurfaceID Surface, SurfaceTypeChoice Choice) : LensEdit;
    public sealed record SetShapeParameter(SurfaceID Surface, ShapeParameter Parameter, double Value) : LensEdit;
    /// <summary>Inserts a plane air surface before index <c>At</c> (1 … image).</summary>
    public sealed record Insert(int At) : LensEdit;
    public sealed record Delete(int At) : LensEdit;
    public sealed record MakeStop(int At) : LensEdit;

    public string ActionName => this switch
    {
        SetCurvature => "Change Radius",
        SetThickness => "Change Thickness",
        SetConic => "Change Conic",
        SetSemiDiameter => "Change Semi-Diameter",
        SetMaterial => "Change Material",
        SetLabel => "Change Label",
        SetType => "Change Surface Type",
        SetShapeParameter p => "Change " + p.Parameter.Title(),
        Insert => "Insert Surface",
        Delete => "Delete Surface",
        _ => "Set Stop",
    };

    public OpticalSystem? Apply(OpticalSystem s)
    {
        OpticalSystem? At(SurfaceID id, Func<Surface, Surface?> edit)
        {
            if (s.SurfaceIndexOf(id) is not { } i || edit(s.Surfaces[i]) is not { } changed) return null;
            return s.WithSurface(i, _ => changed);
        }
        static Surface? Shape(Surface sf, SurfaceShape? shape) => shape is null ? null : sf with { Shape = shape };
        switch (this)
        {
            case SetCurvature e: return At(e.Surface, sf => Shape(sf, sf.Shape.WithCurvature(e.Curvature)));
            case SetThickness e: return At(e.Surface, sf => sf with { Thickness = e.Thickness });
            case SetConic e: return At(e.Surface, sf => Shape(sf, sf.Shape.WithConic(e.Conic)));
            case SetSemiDiameter e: return At(e.Surface, sf => sf with { SemiDiameter = e.Value });
            case SetMaterial e:
                return At(e.Surface, sf => sf with
                {
                    Interaction = e.Material.Interaction,
                    Material = e.Material.Material is { } m ? LensValueParser.Replace(sf.Material, m) : sf.Material,
                });
            case SetLabel e: return At(e.Surface, sf => sf with { Label = e.Label });
            case SetShapeParameter e: return At(e.Surface, sf => Shape(sf, sf.Shape.WithShapeParameter(e.Parameter, e.Value)));
            case SetType e:
                if (s.SurfaceIndexOf(e.Surface) is not { } i) return null;
                if (e.Choice.Cylinder is not { } axis) return s.SetShapeKind(e.Choice.Kind, i);
                var c = s.Surfaces[i].Shape.Curvature;
                var biconic = s.SetShapeKind(ShapeKind.Biconic, i);
                return biconic.WithSurface(i, sf => sf with { Shape = SurfaceShape.Cylinder(c, axis) });
            case Insert e: return s.InsertSurface(e.At).System;
            case Delete e: return s.DeleteSurface(e.At);
            case MakeStop e: return s.SetStop(e.At);
            default: return null;
        }
    }
}

/// <summary>The table's structural commands, with the selection they leave (the Mac app's <c>LensEditor</c>).</summary>
public static class LensCommands
{
    /// <summary>Insert before the surface at <paramref name="index"/>: the edit (clamped to 1 … image).</summary>
    public static LensEdit InsertBefore(OpticalSystem s, int index) => new LensEdit.Insert(Math.Clamp(index, 1, s.ImageIndex));

    /// <summary>Insert after the surface at <paramref name="index"/> (before the image at the latest).</summary>
    public static LensEdit InsertAfter(OpticalSystem s, int index) => new LensEdit.Insert(Math.Clamp(index + 1, 1, s.ImageIndex));

    public static bool CanDelete(OpticalSystem s, int index) => index > 0 && index < s.ImageIndex && s.Surfaces.Count > 3;

    public static bool CanMakeStop(OpticalSystem s, int index) => index > 0 && index < s.ImageIndex && index != s.Stop;

    /// <summary>The row selected after deleting the surface at <paramref name="index"/>: the one that took its place.</summary>
    public static int SelectionAfterDelete(OpticalSystem after, int index) => Math.Min(index, after.ImageIndex);
}
