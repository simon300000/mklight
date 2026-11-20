using Windows.UI.ViewManagement;

namespace MKLight.App.Support;

/// <summary>
/// Reduce Motion: Windows' "Animation effects" setting (Settings, Accessibility,
/// Visual effects). When it is off, the app's own motion stops: the updating
/// pulse, blinking, sliding and fading transitions show their end state at once.
/// Views read <see cref="Enabled"/> before starting an animation and listen to
/// <see cref="Changed"/> (raised on the thread the setting changed on: marshal
/// to the view's dispatcher) to stop one that runs.
/// </summary>
public static class Motion
{
    // UISettings raises its events only while this instance lives.
    static readonly UISettings Settings = new();
    static bool? _override;

    static Motion()
    {
        try { Settings.AnimationsEnabledChanged += (_, _) => Changed?.Invoke(); }
        catch { /* older Windows: the setting is read on demand only */ }
    }

    /// <summary>Whether the app may animate (Windows' "Animation effects" is on).</summary>
    public static bool Enabled
    {
        get
        {
            if (_override is { } o) return o;
            try { return Settings.AnimationsEnabled; } catch { return true; }
        }
    }

    /// <summary>Launch actions and tests: force motion on or off (null follows Windows).</summary>
    public static void Override(bool? enabled)
    {
        _override = enabled;
        Changed?.Invoke();
    }

    /// <summary>The setting changed (any thread).</summary>
    public static event Action? Changed;
}
