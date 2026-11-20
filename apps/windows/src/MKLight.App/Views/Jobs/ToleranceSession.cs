using MKLight.App.ViewModels;

namespace MKLight.App.Views.Jobs;

/// <summary>The Tolerance job (filled in below).</summary>
public sealed class ToleranceSession(DesignSession session, OptimizeSession optimize)
{
    public DesignSession Session => session;
    public OptimizeSession Optimize => optimize;
    public void Cancel() { }
}
