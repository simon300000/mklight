"""Analyses through the engine: registry, results, progress, cancellation,
determinism, timeouts and supersession."""

import _thread
import threading
import time

import numpy as np
import pytest

import mklight


def test_every_runnable_registered_analysis(engine, achromat):
    ran = 0
    for a in engine.analyses():
        if not a["available"]:
            with pytest.raises(mklight.UnsupportedError):
                achromat.run(a["name"])
            continue
        if a["default_settings"] is None:
            continue  # needs required settings (glass_info), tested below
        r = achromat.run(a["name"])
        assert r.analysis == a["name"]
        assert r.run_info["settings"] == a["default_settings"]
        assert r.run_info["engine_version"] == mklight.ENGINE_VERSION
        ran += 1
    if ran == 0:
        pytest.skip("only placeholder analyses are registered")


def test_system_summary_result(achromat):
    r = achromat.run("system_summary")
    n = len(achromat.system["surfaces"])
    w = len(achromat.system["wavelengths"]["wavelengths"])
    assert r.arrays["n_after"].shape == (n, w)
    assert r.arrays["vertex_global"].shape == (n, 3)
    # N-BK7 after surface 1 at the primary (d) wavelength.
    assert r.arrays["n_after"][1, 1] == pytest.approx(1.5168, abs=2e-4)
    assert r["surfaces"][1]["material"] is not None
    env = r.envelope()
    assert [b["name"] for b in env["buffers"]] == ["n_after", "vertex_global"]
    assert env["run_info"]["snapshot_hash"] == achromat.snapshot_hash


def test_glass_info_needs_its_setting(achromat):
    with pytest.raises(mklight.InvalidSettingsError):
        achromat.run("glass_info")
    r = achromat.run("glass_info", {"name": "N-BK7", "curve_samples": 50})
    assert r.data["vd"] == pytest.approx(64.17, abs=0.05)
    assert r.arrays["dispersion_curve"].shape == (50, 2)


def test_unknown_analysis_and_backend(achromat):
    with pytest.raises(mklight.NotFoundError):
        achromat.run("nope")
    with pytest.raises(mklight.UnsupportedError):
        achromat.run("validate", backend="gpu:nonexistent")
    r = achromat.run("validate", backend="gpu:nonexistent", allow_backend_fallback=True)
    assert r.run_info["backend"] == "cpu_reference" and r.warnings
    assert achromat.run("validate", backend="cpu").run_info["backend"] == "cpu_reference"


def test_seed_determinism(achromat):
    a = achromat.run("rng_stream", {"count": 5000}, seed=42)
    b = achromat.run("rng_stream", {"count": 5000}, seed=42)
    c = achromat.run("rng_stream", {"count": 5000}, seed=43)
    assert a.seed == 42 and a.run_info["settings"]["seed"] == 42
    assert np.array_equal(a.arrays["uniform"], b.arrays["uniform"])
    assert not np.array_equal(a.arrays["uniform"], c.arrays["uniform"])
    assert a.data == b.data
    u = a.arrays["uniform"]
    assert abs(u.mean() - 0.5) < 0.02 and np.all((u >= 0) & (u < 1))
    with pytest.raises(mklight.InvalidSettingsError):
        achromat.run("validate", seed=1)  # deterministic analysis


def test_progress_callback_is_monotone(achromat):
    seen = []
    achromat.run(
        "rng_stream",
        {"count": 40, "chunk": 1, "delay_ms": 5.0},
        progress=lambda f, stage: seen.append((f, stage)),
    )
    fr = [f for f, _ in seen]
    assert len(fr) >= 3, seen
    assert all(0.0 <= f <= 1.0 for f in fr)
    assert all(b >= a for a, b in zip(fr, fr[1:]))
    assert fr[-1] == 1.0
    assert any(s == "drawing" for _, s in seen)


def test_exception_in_progress_cancels(achromat):
    class Stop(Exception):
        pass

    def cb(f, stage):
        raise Stop

    t0 = time.monotonic()
    with pytest.raises(Stop):
        achromat.run("rng_stream", {"count": 1000, "chunk": 1, "delay_ms": 10.0}, progress=cb)
    assert time.monotonic() - t0 < 5.0


def test_submit_and_cancel(achromat):
    job = achromat.submit("rng_stream", {"count": 1000, "chunk": 1, "delay_ms": 10.0})
    deadline = time.monotonic() + 5
    while job.progress == 0.0 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert job.state == "running"
    job.cancel()
    with pytest.raises(mklight.CancelledError):
        job.result(timeout=10)
    assert job.state == "cancelled" and job.done()


def test_async_job_result_and_status(achromat):
    job = achromat.submit("rng_stream", {"count": 2000}, seed=7)
    assert job.analysis == "rng_stream" and job.id > 0
    r = job.result(timeout=30)
    assert r.seed == 7 and job.state == "done" and job.progress == 1.0
    # The result can be fetched again.
    assert np.array_equal(job.result().arrays["uniform"], r.arrays["uniform"])
    assert "Job" in repr(job)


def test_result_timeout_leaves_job_running(achromat):
    job = achromat.submit("rng_stream", {"count": 200, "chunk": 1, "delay_ms": 10.0})
    with pytest.raises(mklight.JobTimeoutError) as e:
        job.result(timeout=0.05)
    assert isinstance(e.value, TimeoutError)
    assert not job.done()
    assert job.wait(0.01) is False
    job.cancel()
    assert job.wait(10) is True


def test_view_key_supersedes(achromat):
    a = achromat.submit("rng_stream", {"count": 500, "chunk": 1, "delay_ms": 10.0}, view_key="w")
    b = achromat.submit("rng_stream", {"count": 10}, view_key="w")
    with pytest.raises(mklight.CancelledError):
        a.result(timeout=10)
    assert b.result(timeout=10).data["count"] == 10


def test_keyboard_interrupt_cancels_running_job(engine, achromat):
    timer = threading.Timer(0.2, _thread.interrupt_main)
    timer.start()
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        achromat.run("rng_stream", {"count": 1000, "chunk": 1, "delay_ms": 10.0})
    assert time.monotonic() - t0 < 3.0
    timer.join()
    # The interrupted job does not keep running on the worker pool.
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        running = [j for j in engine.jobs() if j["state"] in ("queued", "running")]
        if not running:
            break
        time.sleep(0.02)
    assert not running


def test_keyboard_interrupt_cancels_trace_grid(engine):
    s = engine.new(sample="double_gauss")
    timer = threading.Timer(0.05, _thread.interrupt_main)
    timer.start()
    t0 = time.monotonic()
    with pytest.raises(KeyboardInterrupt):
        mklight.trace_grid(s, pupil=3000)  # ~60 M rays: far longer than 0.05 s
    timer.join()
    assert time.monotonic() - t0 < 5.0


def test_result_is_current_until_edit(achromat):
    r = achromat.run("validate")
    assert r.snapshot_hash == achromat.snapshot_hash
    sid = achromat.system["surfaces"][3]["id"]
    achromat.set_param({"param": "thickness", "surface": sid}, 50.0)
    assert r.snapshot_hash != achromat.snapshot_hash
