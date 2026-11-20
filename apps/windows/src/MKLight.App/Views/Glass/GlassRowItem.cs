using Microsoft.UI.Text;
using Microsoft.UI.Xaml;
using Microsoft.UI.Xaml.Media;
using MKLight.App.Support;
using MKLight.App.ViewModels;
using MKLight.Engine;
using Windows.UI.Text;

namespace MKLight.App.Views.Glass;

/// <summary>One row of the glass table, as the cells show it (compiled bindings, no reflection).</summary>
public sealed class GlassRowItem(GlassRow row, bool dark, Brush ink, Brush dimBrush)
{
    public GlassRow Row { get; } = row;
    public string Id => Row.Id;
    public string Name => Row.Name;
    public string Catalog => Row.Catalog;
    public string Nd => GlassFormat.Nd(Row);
    public string Vd => GlassFormat.Vd(Row);
    public string Dpgf => GlassFormat.Dpgf(Row);
    public string Density => GlassFormat.Density(Row);
    public string Cost => GlassFormat.Cost(Row);
    public string Tce => GlassFormat.Tce(Row);
    public string Lambda80 => GlassFormat.Lambda80(Row);
    public string Range => GlassFormat.Range(Row);
    public string StatusTitle => Row.Status.Title;

    /// <summary>The IR glass's tooltip on n_d and V_d (the values its catalog lists).</summary>
    public string? IrTip => GlassFormat.IrTip(Row);
    /// <summary>The "—" of an IR glass is dimmed; numbers keep the primary ink.</summary>
    public Brush NdBrush => Row.Nd is null ? dimBrush : ink;
    public Brush VdBrush => Row.Vd is null ? dimBrush : ink;

    public FontWeight NameWeight => Row.Status == GlassStatus.Preferred ? FontWeights.SemiBold : FontWeights.Normal;
    public Visibility ExcludedMark => Row.ExcludeSubstitution ? Visibility.Visible : Visibility.Collapsed;
    public Visibility OverriddenMark => Row.Overridden ? Visibility.Visible : Visibility.Collapsed;

    public Brush StatusBrush => new SolidColorBrush(StatusColor(Row.Status, dark));
    public Brush StatusStroke => new SolidColorBrush(Row.Status == GlassStatus.Obsolete ? StatusColor(Row.Status, dark) : Microsoft.UI.Colors.Transparent);
    /// <summary>Obsolete glasses get a hollow dot, so status does not rely on colour alone (as on the glass map).</summary>
    public Brush StatusFill => Row.Status == GlassStatus.Obsolete ? new SolidColorBrush(Microsoft.UI.Colors.Transparent) : StatusBrush;

    /// <summary>The glass map's status colours: preferred green, standard blue, special violet, melt yellow, obsolete red, others grey.</summary>
    public static global::Windows.UI.Color StatusColor(GlassStatus s, bool dark) => s.Code switch
    {
        1 => Palette.Slot(5, dark),
        0 => Palette.Slot(0, dark),
        3 => Palette.Slot(6, dark),
        4 => Palette.Slot(3, dark),
        2 => Palette.Slot(7, dark),
        _ => dark ? Palette.Rgb(0x8a8a86) : Palette.Rgb(0x8f8f8a),
    };
}
