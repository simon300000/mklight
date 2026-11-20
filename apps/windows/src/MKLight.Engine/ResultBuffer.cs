using System.Runtime.InteropServices;

namespace MKLight.Engine;

/// <summary>Element type of a result buffer (<c>MKDType</c>).</summary>
public enum BufferDType
{
    F64 = 1,
    F32 = 2,
    I32 = 3,
    U32 = 4,
    I64 = 5,
    U64 = 6,
    U8 = 7,
}

public static class BufferDTypeExtensions
{
    /// <summary>The engine's name: <c>f64</c>, <c>f32</c>, …</summary>
    public static string WireName(this BufferDType d) => d.ToString().ToLowerInvariant();

    public static BufferDType? FromName(string name) =>
        Enum.GetValues<BufferDType>().Select(d => (BufferDType?)d).FirstOrDefault(d => d!.Value.WireName() == name);

    public static int ByteWidth(this BufferDType d) => d switch
    {
        BufferDType.F64 or BufferDType.I64 or BufferDType.U64 => 8,
        BufferDType.U8 => 1,
        _ => 4,
    };

    /// <summary>The dtype of a C# element type, or <c>null</c> for types the engine does not use.</summary>
    public static BufferDType? Of<T>() where T : unmanaged =>
        typeof(T) == typeof(double) ? BufferDType.F64
        : typeof(T) == typeof(float) ? BufferDType.F32
        : typeof(T) == typeof(int) ? BufferDType.I32
        : typeof(T) == typeof(uint) ? BufferDType.U32
        : typeof(T) == typeof(long) ? BufferDType.I64
        : typeof(T) == typeof(ulong) ? BufferDType.U64
        : typeof(T) == typeof(byte) ? BufferDType.U8
        : null;
}

/// <summary>Keeps native buffer memory alive; released once per buffer.</summary>
internal abstract class BufferOwner
{
    public abstract void Release();
}

/// <summary>
/// A named, row-major typed array from an engine result.
/// <para>
/// Engine buffers are <b>not copied</b>: the buffer points into memory owned by the
/// engine library, shared by every buffer of one result. <see cref="Dispose"/> releases
/// this buffer's share; the engine's array is freed (<c>mk_buffer_free</c>) exactly once,
/// when the last buffer of the result is disposed, or by a finalizer once none of them is
/// reachable, so forgetting to dispose leaks nothing. The memory is immutable, so a buffer
/// may be read from any thread.
/// </para>
/// <para>
/// A span from <see cref="AsSpan{T}"/> or <see cref="Bytes"/> points into that memory:
/// keep the buffer referenced (and undisposed) while the span is in use.
/// </para>
/// </summary>
public sealed unsafe class ResultBuffer : IDisposable, IEquatable<ResultBuffer>
{
    readonly byte[]? managed;
    readonly byte* native;
    readonly BufferOwner? owner;
    readonly int[] shape;
    int disposed;

    public string Name { get; }
    public BufferDType DType { get; }
    /// <summary>Extents, row-major (their product is <see cref="Count"/>).</summary>
    public IReadOnlyList<int> Shape => shape;
    /// <summary>Number of elements.</summary>
    public int Count { get; }

    /// <summary>Wraps engine memory without copying; <paramref name="owner"/> keeps it valid and is released once on dispose.</summary>
    internal ResultBuffer(string name, BufferDType dtype, int[] shape, int count, byte* data, BufferOwner owner)
    {
        Name = name;
        DType = dtype;
        this.shape = shape;
        Count = count;
        native = count > 0 ? data : null;
        this.owner = owner;
    }

    ResultBuffer(string name, BufferDType dtype, int[] shape, byte[] bytes)
    {
        Name = name;
        DType = dtype;
        this.shape = shape;
        Count = bytes.Length / dtype.ByteWidth();
        managed = bytes;
    }

    /// <summary>A buffer holding a copy of <paramref name="values"/> (tests and engines without native memory).</summary>
    public static ResultBuffer Create<T>(string name, ReadOnlySpan<T> values, int[]? shape = null) where T : unmanaged
    {
        var dtype = BufferDTypeExtensions.Of<T>() ?? throw new ArgumentException($"{typeof(T).Name} is not a buffer element type");
        var s = shape ?? [values.Length];
        if (s.Aggregate(1L, (p, n) => p * n) != values.Length) throw new ArgumentException("shape does not match the element count");
        return new ResultBuffer(name, dtype, s, MemoryMarshal.AsBytes(values).ToArray());
    }

    /// <summary>A buffer of doubles (a copy).</summary>
    public static ResultBuffer FromDoubles(string name, double[] values, int[]? shape = null) => Create<double>(name, values, shape);

    /// <summary>The raw bytes (<c>Count × DType.ByteWidth()</c>).</summary>
    public ReadOnlySpan<byte> Bytes
    {
        get
        {
            ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed) != 0, this);
            return managed ?? new ReadOnlySpan<byte>(native, Count * DType.ByteWidth());
        }
    }

    /// <summary>Every element converted to double into <paramref name="a"/> (at least <see cref="Count"/> long), in one pass over its typed span.</summary>
    public void CopyTo(Span<double> a)
    {
        var n = Count;
        switch (DType)
        {
            case BufferDType.F64: AsSpan<double>().CopyTo(a); break;
            case BufferDType.F32: { var s = AsSpan<float>(); for (var k = 0; k < n; k++) a[k] = s[k]; break; }
            case BufferDType.I32: { var s = AsSpan<int>(); for (var k = 0; k < n; k++) a[k] = s[k]; break; }
            case BufferDType.U32: { var s = AsSpan<uint>(); for (var k = 0; k < n; k++) a[k] = s[k]; break; }
            case BufferDType.I64: { var s = AsSpan<long>(); for (var k = 0; k < n; k++) a[k] = s[k]; break; }
            case BufferDType.U64: { var s = AsSpan<ulong>(); for (var k = 0; k < n; k++) a[k] = s[k]; break; }
            case BufferDType.U8: { var s = AsSpan<byte>(); for (var k = 0; k < n; k++) a[k] = s[k]; break; }
            default: for (var k = 0; k < n; k++) a[k] = this[k]; break;
        }
        GC.KeepAlive(this);
    }

    /// <summary>Typed access without copying; <typeparamref name="T"/> must match <see cref="DType"/>.</summary>
    public bool TryGetSpan<T>(out ReadOnlySpan<T> span) where T : unmanaged
    {
        if (BufferDTypeExtensions.Of<T>() != DType)
        {
            span = default;
            return false;
        }
        span = MemoryMarshal.Cast<byte, T>(Bytes);
        return true;
    }

    /// <summary>Typed access without copying; throws <see cref="InvalidOperationException"/> when <typeparamref name="T"/> does not match <see cref="DType"/>.</summary>
    public ReadOnlySpan<T> AsSpan<T>() where T : unmanaged =>
        TryGetSpan<T>(out var span) ? span
        : throw new InvalidOperationException($"buffer '{Name}' holds {DType.WireName()}, not {typeof(T).Name}");

    /// <summary>Element <paramref name="i"/> converted to <see cref="double"/>.</summary>
    public double this[int i]
    {
        get
        {
            if ((uint)i >= (uint)Count) throw new IndexOutOfRangeException($"buffer index {i} out of range 0..<{Count}");
            var value = Element(Bytes, i);
            GC.KeepAlive(this);
            return value;
        }
    }

    /// <summary>Element at a 2-D index (row-major).</summary>
    public double this[int i, int j] => this[Flat([i, j])];

    /// <summary>Element at a multi-index (row-major).</summary>
    public double this[params int[] index] => this[Flat(index)];

    int Flat(ReadOnlySpan<int> index)
    {
        if (index.Length != shape.Length) throw new ArgumentException("wrong number of indices");
        int flat = 0;
        for (int k = 0; k < index.Length; k++)
        {
            if ((uint)index[k] >= (uint)shape[k]) throw new IndexOutOfRangeException($"index {index[k]} out of range on axis {k}");
            flat = flat * shape[k] + index[k];
        }
        return flat;
    }

    double Element(ReadOnlySpan<byte> b, int i) => DType switch
    {
        BufferDType.F64 => MemoryMarshal.Cast<byte, double>(b)[i],
        BufferDType.F32 => MemoryMarshal.Cast<byte, float>(b)[i],
        BufferDType.I32 => MemoryMarshal.Cast<byte, int>(b)[i],
        BufferDType.U32 => MemoryMarshal.Cast<byte, uint>(b)[i],
        BufferDType.I64 => MemoryMarshal.Cast<byte, long>(b)[i],
        BufferDType.U64 => MemoryMarshal.Cast<byte, ulong>(b)[i],
        _ => b[i],
    };

    /// <summary>All elements converted to <see cref="double"/> (a copy).</summary>
    public double[] ToDoubleArray()
    {
        var b = Bytes;
        var output = new double[Count];
        for (int i = 0; i < output.Length; i++) output[i] = Element(b, i);
        GC.KeepAlive(this);
        return output;
    }

    /// <summary>The number of rows (first axis).</summary>
    public int Rows => shape.Length > 0 ? shape[0] : Count;

    /// <summary>Row <paramref name="i"/> of a 2-D (or higher) buffer as doubles.</summary>
    public double[] Row(int i)
    {
        if (shape.Length < 2) return [this[i]];
        int k = 1;
        for (int a = 1; a < shape.Length; a++) k *= shape[a];
        var output = new double[k];
        for (int j = 0; j < k; j++) output[j] = this[i * k + j];
        return output;
    }

    /// <summary>Releases this buffer's share of the engine memory. Idempotent.</summary>
    public void Dispose()
    {
        if (Interlocked.Exchange(ref disposed, 1) == 0) owner?.Release();
    }

    internal BufferOwner? Owner => owner;

    public override string ToString() => $"{Name}: {DType.WireName()}[{string.Join("×", shape)}]";

    public bool Equals(ResultBuffer? other)
    {
        if (ReferenceEquals(this, other)) return true;
        if (other is null || Name != other.Name || DType != other.DType || !shape.AsSpan().SequenceEqual(other.shape)) return false;
        var equal = Bytes.SequenceEqual(other.Bytes);
        GC.KeepAlive(this);
        GC.KeepAlive(other);
        return equal;
    }

    public override bool Equals(object? obj) => Equals(obj as ResultBuffer);

    public override int GetHashCode()
    {
        var h = new HashCode();
        h.Add(Name);
        h.Add(DType);
        foreach (var n in shape) h.Add(n);
        return h.ToHashCode();
    }
}
