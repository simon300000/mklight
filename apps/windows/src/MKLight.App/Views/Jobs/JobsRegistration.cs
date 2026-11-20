using System.Runtime.CompilerServices;
using MKLight.App.Views.Inspector;

namespace MKLight.App.Views.Jobs;

/// <summary>Registers the design-job pages of the inspector (placeholders until the jobs land).</summary>
static class JobsRegistration
{
    [ModuleInitializer]
    internal static void Register()
    {
        InspectorPages.Add(new("optimize", "Optimize", "\uE9D9", 20, _ => InspectorRegistration.Placeholder("Optimize")));
        InspectorPages.Add(new("glass_substitution", "Glass Substitution", "\uEB42", 21, _ => InspectorRegistration.Placeholder("Glass Substitution")));
        InspectorPages.Add(new("tolerance", "Tolerance", "\uE9E9", 22, _ => InspectorRegistration.Placeholder("Tolerance")));
        InspectorPages.Add(new("solves", "Apply Solves", "\uE943", 23, _ => InspectorRegistration.Placeholder("Apply Solves")));
    }
}
