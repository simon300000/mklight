"""Arrays (buffer protocol, numpy / list fallback) and GIL release."""

import sys
import threading
import time

import numpy as np
import pytest

import mklight


def test_numpy_arrays_are_zero_copy_and_read_only(achromat):
    r = achromat.run("rng_stream", {"count": 1000}, seed=1)
    u = r.arrays["uniform"]
    assert isinstance(u, np.ndarray) and not u.flags.writeable
    with pytest.raises(ValueError):
        u[0] = 1.0
    (buf,) = r.buffers()
    assert isinstance(buf, mklight.Array)
    assert buf.name == "uniform" and buf.dtype == "float64" and buf.shape == (1000,)
    # Two views of the same buffer share memory: no copy was made.
    assert np.shares_memory(np.asarray(buf), u)
    assert np.asarray(buf).tolist() == buf.tolist()
    m = memoryview(buf)
    assert m.readonly and m.format == "d" and m.shape == (1000,)
    assert buf.flat(3) == u[3]


def test_multidimensional_buffer_protocol(achromat):
    g = mklight.trace_grid(achromat, pupil=4)
    p = g["point"]
    f, w, n, _ = p.shape
    assert p.flags.c_contiguous and not p.flags.writeable
    assert p.strides == (w * n * 24, n * 24, 24, 8)
    # Element (f, w, p, k) is where the row-major layout says it is.
    flat = np.frombuffer(memoryview(p).cast("B"), dtype=np.float64)
    assert flat[((1 * w + 2) * n + 3) * 3 + 1] == p[1, 2, 3, 1]


@pytest.mark.skipif(sys.version_info < (3, 12), reason="__buffer__ needs Python 3.12")
def test_fortran_contiguous_requests_are_refused_unless_layouts_coincide(achromat):
    import inspect

    bufs = {b.name: b for b in achromat.run("system_summary").buffers()}
    a = bufs["n_after"]
    assert a.ndim == 2 and min(a.shape) > 1
    F = inspect.BufferFlags
    with pytest.raises(BufferError):
        a.__buffer__(F.F_CONTIGUOUS | F.FORMAT)
    for ok in (F.C_CONTIGUOUS, F.ANY_CONTIGUOUS, F.STRIDES, F.FULL_RO):
        m = a.__buffer__(ok | F.FORMAT)
        assert m.c_contiguous and m.shape == a.shape
    # A 1-d array is both C and Fortran contiguous.
    u = achromat.run("rng_stream", {"count": 8}, seed=1).buffers()[0]
    assert u.__buffer__(F.F_CONTIGUOUS | F.FORMAT).f_contiguous


def test_lists_without_numpy(achromat):
    mklight.set_numpy_enabled(False)
    assert not mklight.numpy_enabled()
    r = achromat.run("system_summary")
    n_after = r.arrays["n_after"]
    assert isinstance(n_after, list) and isinstance(n_after[0], list)
    g = mklight.trace_grid(achromat, fields=0, wavelengths=0, pupil=[[0.0, 0.0]])
    assert g["status"] == [[[0]]]
    mklight.set_numpy_enabled(True)
    assert mklight.numpy_enabled()
    assert np.allclose(np.array(n_after), achromat.run("system_summary").arrays["n_after"])


def _ticks_during(call):
    """Run `call` while a Python thread records timestamps; return the call's
    (start, end) and the timestamps."""
    stamps = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            stamps.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    time.sleep(0.02)
    t0 = time.perf_counter()
    call()
    t1 = time.perf_counter()
    stop.set()
    t.join()
    return t0, t1, stamps


def test_gil_released_during_trace_grid(engine):
    s = engine.new(sample="double_gauss")
    t0, t1, stamps = _ticks_during(lambda: mklight.trace_grid(s, pupil=900))
    assert t1 - t0 > 0.1, "trace too short to test"
    inside = [x for x in stamps if t0 + 0.02 < x < t1 - 0.02]
    # A thread holding the GIL for the whole trace would allow none.
    assert len(inside) > 10


def test_gil_released_during_run(achromat):
    t0, t1, stamps = _ticks_during(
        lambda: achromat.run("rng_stream", {"count": 30, "chunk": 1, "delay_ms": 10.0})
    )
    inside = [x for x in stamps if t0 + 0.02 < x < t1 - 0.02]
    assert t1 - t0 > 0.2 and len(inside) > 20


def test_concurrent_sessions_from_threads(engine):
    out = {}

    def work(name):
        s = engine.new(sample=name)
        out[name] = mklight.trace_grid(s, pupil=64)["status"].shape

    names = mklight.samples()
    threads = [threading.Thread(target=work, args=(n,)) for n in names]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert set(out) == set(names)
