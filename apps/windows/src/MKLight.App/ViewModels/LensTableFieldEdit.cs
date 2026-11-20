namespace MKLight.App.ViewModels;

/// <summary>How a value reads as text and text as a value.</summary>
/// <param name="Exact">The text while the field has the focus: it parses back to exactly the value.</param>
/// <param name="Display">The text at rest.</param>
/// <param name="Parse">The value of typed text; <c>null</c> when the text is not one.</param>
public sealed record FieldFormat<T>(Func<T, string> Exact, Func<T, string> Display, Func<string, Optional<T>> Parse)
{
    /// <summary>Text that is its own value (the lens table's cells commit the typed text; the row parses it).</summary>
    public static FieldFormat<string> Identity { get; } = new(s => s, s => s, s => new Optional<string>(s));
}

/// <summary>A value or none, for value and reference types alike.</summary>
public readonly record struct Optional<T>(T Value)
{
    public bool HasValue { get; } = true;
    public static Optional<T> None => default;
}

/// <summary>What the end of an edit asks of the owner.</summary>
public abstract record FieldCommit<T>
{
    /// <summary>Nothing to write: the text was not edited, or it is the current value.</summary>
    public sealed record Unchanged : FieldCommit<T>;
    /// <summary>Write this value: typed, and not the current one.</summary>
    public sealed record Value(T NewValue) : FieldCommit<T>;
    /// <summary>The edited text is not a value; the owner reverts it or leaves it to be corrected.</summary>
    public sealed record Invalid(string Text) : FieldCommit<T>;
}

/// <summary>
/// The state of one text field that edits a value, and the one rule by which typed text becomes a value
/// (the Mac app's <c>FieldEdit</c>, docs/APP.md "One text-field rule"):
/// <list type="bullet">
/// <item>only text the user edited is committed: text left as the field showed it stands for the value it was shown for;</item>
/// <item>a value equal to the current one is not committed;</item>
/// <item>text that was not edited follows the value;</item>
/// <item>edited text is replaced only by its own commit, or by <see cref="Revert"/>;</item>
/// <item>text typed for one owner goes to that owner, even when the field moved on to another.</item>
/// </list>
/// No UI: the lens table's cell editor shows <see cref="Text"/> and reports focus, Enter and value changes.
/// </summary>
public sealed class FieldEdit<T>
{
    readonly FieldFormat<T> _format;
    string _base;

    public FieldEdit(T value, FieldFormat<T> format, object? owner = null)
    {
        _format = format;
        Value = value;
        Owner = owner;
        Text = format.Display(value);
        _base = Text;
    }

    /// <summary>The text the field shows. The user's typing sets it.</summary>
    public string Text { get; set; }
    /// <summary>The current value: the reported one, or the one just committed.</summary>
    public T Value { get; private set; }
    /// <summary>Whose value it is, where one field shows several objects in turn.</summary>
    public object? Owner { get; private set; }
    public bool IsFocused { get; private set; }

    /// <summary>Whether the text is the user's and not the value's.</summary>
    public bool IsEdited => Text != _base;

    /// <summary>The field took the focus: text that was not edited shows every digit of the value.</summary>
    public void BeginEditing()
    {
        IsFocused = true;
        Rebase();
    }

    /// <summary>
    /// The owner's value is <paramref name="value"/>. Text that was not edited follows it; edited text stays.
    /// With another owner the edit ends here: the result is the commit of the text typed for the earlier
    /// owner (whose owner it was is <paramref name="earlierOwner"/>), and the field shows the new one.
    /// </summary>
    public FieldCommit<T> ValueChanged(T value, object? owner, out object? earlierOwner)
    {
        FieldCommit<T> earlier = new FieldCommit<T>.Unchanged();
        earlierOwner = Owner;
        if (!Equals(owner, Owner))
        {
            earlier = Submit();
            Text = _base;
            Owner = owner;
        }
        Value = value;
        Rebase();
        return earlier;
    }

    /// <summary>The value changed for the same owner.</summary>
    public void ValueChanged(T value) => ValueChanged(value, Owner, out _);

    /// <summary>Enter: commits edited text; the field keeps the focus and shows every digit of what was typed.</summary>
    public FieldCommit<T> Submit()
    {
        if (!IsEdited) return new FieldCommit<T>.Unchanged();
        var typed = _format.Parse(Text);
        if (!typed.HasValue) return new FieldCommit<T>.Invalid(Text);
        var changed = !EqualityComparer<T>.Default.Equals(typed.Value, Value);
        Value = typed.Value;
        _base = Shown(typed.Value);
        Text = _base;
        return changed ? new FieldCommit<T>.Value(typed.Value) : new FieldCommit<T>.Unchanged();
    }

    /// <summary>The focus left: commits edited text and shows the value at rest. Text that is not a value stays as typed.</summary>
    public FieldCommit<T> EndEditing()
    {
        var commit = Submit();
        IsFocused = false;
        Rebase();
        return commit;
    }

    /// <summary>Gives the edit up: the field shows the value again.</summary>
    public void Revert()
    {
        _base = Shown(Value);
        Text = _base;
    }

    string Shown(T v) => IsFocused ? _format.Exact(v) : _format.Display(v);

    /// <summary>Takes the value's text for the present focus as the base; text that was not edited becomes it.</summary>
    void Rebase()
    {
        var edited = IsEdited;
        _base = Shown(Value);
        if (!edited) Text = _base;
    }
}
