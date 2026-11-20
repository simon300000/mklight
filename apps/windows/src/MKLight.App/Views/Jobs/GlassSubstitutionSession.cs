using MKLight.App.ViewModels;

namespace MKLight.App.Views.Jobs;

/// <summary>The Glass Substitution job (filled in below).</summary>
public sealed class GlassSubstitutionSession(DesignSession session, OptimizeSession optimize)
{
    public DesignSession Session => session;
    public OptimizeSession Optimize => optimize;
    public void Cancel() { }
}
