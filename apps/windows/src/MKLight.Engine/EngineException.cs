namespace MKLight.Engine;

/// <summary>Status codes of the engine's C ABI (<c>MKStatus</c>, docs/FFI.md).</summary>
public enum EngineErrorCode
{
    InvalidArgument = 1,
    Parse = 2,
    InvalidModel = 3,
    NotFound = 4,
    Io = 5,
    Unsupported = 6,
    Cancelled = 7,
    Timeout = 8,
    NotReady = 9,
    JobFailed = 10,
    Compute = 11,
    Panic = 12,
    Internal = 13,
}

/// <summary>The cases of an <see cref="EngineException"/>.</summary>
public enum EngineErrorKind
{
    Cancelled,
    /// <summary>The model failed validation; <see cref="EngineException.Issues"/> lists the problems.</summary>
    InvalidModel,
    /// <summary>The engine refused the analysis settings.</summary>
    InvalidSettings,
    Unsupported,
    /// <summary>A job ended with an internal error (status <c>JobFailed</c>).</summary>
    Failed,
    /// <summary>Any other engine status, with the engine's message.</summary>
    Engine,
    /// <summary>The engine library is missing, or its ABI major version differs from the one this app was built for.</summary>
    LibraryUnavailable,
}

/// <summary>An error of the engine or of the boundary to it.</summary>
public sealed class EngineException : Exception
{
    public EngineErrorKind Kind { get; }

    /// <summary>The engine status code this error corresponds to (<c>null</c> for <see cref="EngineErrorKind.LibraryUnavailable"/>).</summary>
    public EngineErrorCode? Code { get; }

    /// <summary>The engine's message (for invalid models and settings: the issues joined by "; ").</summary>
    public string Detail { get; }

    /// <summary>The issues of an invalid model or invalid settings (one entry for other kinds).</summary>
    public IReadOnlyList<string> Issues { get; }

    EngineException(EngineErrorKind kind, EngineErrorCode? code, IReadOnlyList<string> issues, string message)
        : base(message)
    {
        Kind = kind;
        Code = code;
        Issues = issues;
        Detail = string.Join("; ", issues);
    }

    public static EngineException Cancelled() =>
        new(EngineErrorKind.Cancelled, EngineErrorCode.Cancelled, ["cancelled"], "Cancelled");

    public static EngineException InvalidModel(IEnumerable<string> issues)
    {
        var list = issues.ToArray();
        return new(EngineErrorKind.InvalidModel, EngineErrorCode.InvalidModel, list, "Invalid model: " + string.Join("; ", list));
    }

    public static EngineException InvalidSettings(IEnumerable<string> issues)
    {
        var list = issues.ToArray();
        return new(EngineErrorKind.InvalidSettings, EngineErrorCode.InvalidArgument, list,
            "Invalid analysis settings: " + string.Join("; ", list));
    }

    public static EngineException Unsupported(string message) =>
        new(EngineErrorKind.Unsupported, EngineErrorCode.Unsupported, [message], "Unsupported: " + message);

    public static EngineException Failed(string message) =>
        new(EngineErrorKind.Failed, EngineErrorCode.JobFailed, [message], message);

    public static EngineException Engine(EngineErrorCode code, string message) =>
        new(EngineErrorKind.Engine, code, [message], message.Length == 0 ? $"Engine error: {Describe(code)}" : message);

    public static EngineException LibraryUnavailable(string message) =>
        new(EngineErrorKind.LibraryUnavailable, null, [message], "Engine library unavailable: " + message);

    /// <summary>
    /// Maps an engine status and message to the most specific case: cancellation,
    /// invalid model and unsupported get their own cases, everything else keeps its
    /// code. A failed job carries the status of its own error; <c>JobFailed</c>
    /// (<see cref="EngineErrorKind.Failed"/>) is a job that ended with an internal error.
    /// </summary>
    public static EngineException From(EngineErrorCode code, string message) => code switch
    {
        EngineErrorCode.Cancelled => Cancelled(),
        EngineErrorCode.InvalidModel => InvalidModel([message]),
        EngineErrorCode.Unsupported => Unsupported(message),
        EngineErrorCode.JobFailed => Failed(message),
        _ => Engine(code, message),
    };

    /// <summary>A short description of a status code ("not found", "I/O error", …).</summary>
    public static string Describe(EngineErrorCode code) => code switch
    {
        EngineErrorCode.InvalidArgument => "invalid argument",
        EngineErrorCode.Parse => "parse error",
        EngineErrorCode.InvalidModel => "invalid model",
        EngineErrorCode.NotFound => "not found",
        EngineErrorCode.Io => "I/O error",
        EngineErrorCode.Unsupported => "unsupported",
        EngineErrorCode.Cancelled => "cancelled",
        EngineErrorCode.Timeout => "timeout",
        EngineErrorCode.NotReady => "not ready",
        EngineErrorCode.JobFailed => "job failed",
        EngineErrorCode.Compute => "computation failed",
        EngineErrorCode.Panic => "internal panic",
        EngineErrorCode.Internal => "internal error",
        _ => code.ToString(),
    };
}
