using System.Collections;
using System.Collections.Immutable;
using System.Runtime.CompilerServices;

namespace MKLight.Model;

/// <summary>
/// An immutable list with value equality (elements compared in order), the
/// counterpart of Swift's array in a struct: a design that holds lists can be
/// compared and kept whole for undo.
/// </summary>
[CollectionBuilder(typeof(ValueList), nameof(ValueList.Create))]
public sealed class ValueList<T> : IReadOnlyList<T>, IEquatable<ValueList<T>>
{
    private readonly T[] items;

    public static readonly ValueList<T> Empty = new([]);

    internal ValueList(T[] items) => this.items = items;

    public int Count => items.Length;
    public T this[int index] => items[index];
    public bool IsValidIndex(int index) => (uint)index < (uint)items.Length;

    public IEnumerator<T> GetEnumerator() => ((IEnumerable<T>)items).GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => items.GetEnumerator();

    public ValueList<T> Add(T item) => new([.. items, item]);

    public ValueList<T> AddRange(IEnumerable<T> more) => new([.. items, .. more]);

    public ValueList<T> Insert(int index, T item)
    {
        var a = new List<T>(items);
        a.Insert(index, item);
        return new(a.ToArray());
    }

    public ValueList<T> RemoveAt(int index)
    {
        var a = new List<T>(items);
        a.RemoveAt(index);
        return new(a.ToArray());
    }

    public ValueList<T> RemoveAll(Predicate<T> match) => new(items.Where(x => !match(x)).ToArray());

    /// <summary>The list with element <paramref name="index"/> replaced.</summary>
    public ValueList<T> With(int index, T item)
    {
        var a = (T[])items.Clone();
        a[index] = item;
        return new(a);
    }

    /// <summary>The list with element <paramref name="index"/> replaced by <paramref name="edit"/> of it.</summary>
    public ValueList<T> With(int index, Func<T, T> edit) => With(index, edit(items[index]));

    public bool Equals(ValueList<T>? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        if (items.Length != other.items.Length) return false;
        var c = EqualityComparer<T>.Default;
        for (var i = 0; i < items.Length; i++)
            if (!c.Equals(items[i], other.items[i])) return false;
        return true;
    }

    public override bool Equals(object? obj) => Equals(obj as ValueList<T>);

    public override int GetHashCode()
    {
        var h = new HashCode();
        foreach (var x in items) h.Add(x);
        return h.ToHashCode();
    }

    public static bool operator ==(ValueList<T>? a, ValueList<T>? b) => a is null ? b is null : a.Equals(b);
    public static bool operator !=(ValueList<T>? a, ValueList<T>? b) => !(a == b);

    public static implicit operator ValueList<T>(T[] items) => new((T[])items.Clone());

    public override string ToString() => "[" + string.Join(", ", items) + "]";
}

public static class ValueList
{
    public static ValueList<T> Create<T>(ReadOnlySpan<T> items) => new(items.ToArray());

    public static ValueList<T> ToValueList<T>(this IEnumerable<T> source) => new(source.ToArray());
}

/// <summary>An immutable string-keyed map with value equality, enumerated in key order.</summary>
public sealed class ValueMap<T> : IReadOnlyDictionary<string, T>, IEquatable<ValueMap<T>>
{
    private readonly ImmutableSortedDictionary<string, T> map;

    public static readonly ValueMap<T> Empty = new(ImmutableSortedDictionary.Create<string, T>(StringComparer.Ordinal));

    private ValueMap(ImmutableSortedDictionary<string, T> map) => this.map = map;

    public static ValueMap<T> From(IEnumerable<KeyValuePair<string, T>> pairs) =>
        new(ImmutableSortedDictionary.CreateRange(StringComparer.Ordinal, pairs));

    public int Count => map.Count;
    public T this[string key] => map[key];
    public IEnumerable<string> Keys => map.Keys;
    public IEnumerable<T> Values => map.Values;
    public bool ContainsKey(string key) => map.ContainsKey(key);
    public bool TryGetValue(string key, out T value) => map.TryGetValue(key, out value!);
    public IEnumerator<KeyValuePair<string, T>> GetEnumerator() => map.GetEnumerator();
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public ValueMap<T> Set(string key, T value) => new(map.SetItem(key, value));
    public ValueMap<T> Remove(string key) => new(map.Remove(key));

    public bool Equals(ValueMap<T>? other)
    {
        if (other is null) return false;
        if (ReferenceEquals(this, other)) return true;
        if (map.Count != other.map.Count) return false;
        var c = EqualityComparer<T>.Default;
        foreach (var (k, v) in map)
            if (!other.map.TryGetValue(k, out var o) || !c.Equals(v, o)) return false;
        return true;
    }

    public override bool Equals(object? obj) => Equals(obj as ValueMap<T>);

    public override int GetHashCode()
    {
        var h = 0;
        foreach (var (k, v) in map) h ^= HashCode.Combine(k, v);
        return h;
    }

    public static bool operator ==(ValueMap<T>? a, ValueMap<T>? b) => a is null ? b is null : a.Equals(b);
    public static bool operator !=(ValueMap<T>? a, ValueMap<T>? b) => !(a == b);
}
