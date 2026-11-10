"""Low-level reference tracing: mklight.trace and mklight.trace_grid."""

import math

import numpy as np
import pytest

import mklight


def test_trace_grid_shapes_and_dtypes(achromat):
    g = mklight.trace_grid(achromat, pupil=8)
    F, W = 3, 3
    P = g["pupil"].shape[0]
    assert P == 32  # 8 x 8 grid clipped to the unit disk
    assert g["point"].shape == (F, W, P, 3) and g["point"].dtype == np.float64
    assert g["dir"].shape == (F, W, P, 3) and g["dir"].dtype == np.float64
    for k in ("opl", "transmittance"):
        assert g[k].shape == (F, W, P) and g[k].dtype == np.float64
    assert g["status"].shape == (F, W, P) and g["status"].dtype == np.uint8
    assert g["surface"].shape == (F, W, P) and g["surface"].dtype == np.int32
    assert g["field"].shape == (F, 2)
    assert g["wavelength_um"].shape == (W,)
    assert g["wavelength_index"].tolist() == [0, 1, 2]
    assert np.all(g["status"] == 0) and np.all(g["surface"] == -1)
    # Unit direction cosines.
    assert np.allclose(np.linalg.norm(g["dir"], axis=-1), 1.0, atol=1e-12)


def test_trace_grid_agrees_with_single_reference_traces(achromat):
    pupil = np.array([[0.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.3, 0.8]])
    g = mklight.trace_grid(achromat, fields=[0, 2, (0.0, 0.5)], wavelengths=[1, 0], pupil=pupil)
    assert g["point"].shape == (3, 2, 4, 3)
    fields = [0, 2, (0.0, 0.5)]
    for fi, f in enumerate(fields):
        for wi, w in enumerate([1, 0]):
            for pi, (px, py) in enumerate(pupil):
                r = mklight.trace(achromat, f, w, px, py, records=False)
                assert r["status"] == mklight.TRACE_STATUS[g["status"][fi, wi, pi]]
                assert tuple(g["point"][fi, wi, pi]) == r["point"]
                assert tuple(g["dir"][fi, wi, pi]) == r["dir"]
                assert g["opl"][fi, wi, pi] == r["opl"]
    assert g["field"][2].tolist() == [0.0, 0.5]


def test_known_answers_achromat(achromat):
    p = mklight.paraxial(achromat)
    # On-axis chief ray lands on the axis.
    r = mklight.trace(achromat, 0, None, 0.0, 0.0)
    assert r["ok"] and abs(r["point"][0]) < 1e-12 and abs(r["point"][1]) < 1e-12
    # Near-paraxial marginal ray focuses on the axis at the paraxial image.
    r = mklight.trace(achromat, 0, None, 0.0, 1e-3)
    assert abs(r["point"][1]) < 1e-7
    # Real chief-ray height ~ f tan(theta) for the 2 deg field (tiny distortion).
    r = mklight.trace(achromat, 2, None, 0.0, 0.0)
    theta = math.radians(achromat.system["fields"]["fields"][2]["y"])
    assert r["point"][1] == pytest.approx(p["efl"] * math.tan(theta), rel=2e-3)
    assert r["point"][1] == pytest.approx(p["paraxial_image_height"], rel=2e-3)


def test_records_match_engine_layout(achromat):
    # The on-axis chief ray crosses every vertex: its global z per surface
    # equals the vertex positions reported by the system_summary analysis.
    r = mklight.trace(achromat, 0, None, 0.0, 0.0, records=True)
    rec = r["records"]
    n = len(achromat.system["surfaces"])
    assert rec["point"].shape == (n, 3)
    assert tuple(rec["point"][-1]) == r["point"]
    vg = achromat.run("system_summary").arrays["vertex_global"]
    assert np.allclose(rec["global_point"][1:, 2], vg[1:, 2], atol=1e-9)
    # Record 0 is the starting ray (on the entrance plane at surface 1's
    # vertex for an object at infinity); the path grows from there.
    assert rec["opl"][0] == 0.0
    assert np.all(np.diff(rec["opl"][1:]) > 0)
    assert np.all((rec["cos_incidence"] >= 0) & (rec["cos_incidence"] <= 1))


def test_failures_are_reported_not_raised(achromat):
    r = mklight.trace(achromat, 0, None, 0.0, 30.0)
    assert not r["ok"] and r["status"] != "ok"
    g = mklight.trace_grid(achromat, fields=0, wavelengths=0, pupil=[[0.0, 30.0], [0.0, 0.0]])
    assert g["status"][0, 0, 0] != 0 and g["surface"][0, 0, 0] >= 0
    assert g["status"][0, 0, 1] == 0


def test_bad_arguments(achromat):
    with pytest.raises(mklight.InvalidArgumentError):
        mklight.trace(achromat, 5)
    with pytest.raises(mklight.InvalidArgumentError):
        mklight.trace_grid(achromat, wavelengths=[3])
    with pytest.raises(ValueError):
        mklight.trace_grid(achromat, pupil=[[0.0]])
    # A grid size outside 1..4096 is refused before anything is allocated.
    for n in (0, -1, 4097, 10**6, 10**10):
        with pytest.raises(mklight.InvalidArgumentError, match="pupil grid size"):
            mklight.trace_grid(achromat, pupil=n)


def test_pupil_patterns_and_progress(achromat):
    seen = []
    g = mklight.trace_grid(
        achromat,
        pupil={"type": "hexapolar", "rings": 4},
        progress=lambda f, stage: seen.append(f),
    )
    assert g["pupil"].shape == (61, 2)
    assert seen and seen[-1] == 1.0
    assert all(b >= a for a, b in zip(seen, seen[1:]))


def test_tracer_follows_edits(achromat):
    sid = achromat.system["surfaces"][3]["id"]
    before = mklight.trace(achromat, 0, None, 0.0, 1.0)["point"][1]
    achromat.set_param({"param": "thickness", "surface": sid}, 80.0)
    after = mklight.trace(achromat, 0, None, 0.0, 1.0)["point"][1]
    # 15 mm before focus the F/5 marginal ray is ~1.5 mm off axis.
    assert abs(before) < 0.05 and abs(after) > 1.0


def test_trace_and_trace_grid_read_field_arguments_alike(achromat):
    # A tuple of two numbers is one normalised point in both functions,
    # whether the numbers are ints or floats.
    primary = mklight.paraxial(achromat)["primary_wavelength"]
    for field in [(0, 1), (0, 0.5), (0.0, 1.0)]:
        r = mklight.trace(achromat, field, primary, records=False)
        g = mklight.trace_grid(achromat, fields=field, wavelengths=primary, pupil=[[0.0, 0.0]])
        assert g["field"].tolist() == [[float(field[0]), float(field[1])]]
        assert g["point"][0, 0, 0].tolist() == list(r["point"])
        lst = mklight.trace_grid(achromat, fields=[field], wavelengths=primary, pupil=[[0.0, 0.0]])
        assert lst["point"].tolist() == g["point"].tolist()
    # A list is a list of fields: system fields 0 and 1.
    g = mklight.trace_grid(achromat, fields=[0, 1], wavelengths=primary, pupil=[[0.0, 0.0]])
    sys_fields = mklight.trace_grid(achromat, wavelengths=primary, pupil=[[0.0, 0.0]])["field"]
    assert g["field"].tolist() == sys_fields[:2].tolist()
    assert g["field"].tolist() != [[0.0, 1.0]]
    with pytest.raises(mklight.InvalidArgumentError, match="tuple"):
        mklight.trace_grid(achromat, fields=[0.0, 0.5])


def test_negative_indices_raise_argument_errors(achromat):
    for bad in (-1, 2**64, True):
        with pytest.raises(mklight.InvalidArgumentError):
            mklight.trace(achromat, 0, bad)
        with pytest.raises(mklight.InvalidArgumentError):
            mklight.paraxial(achromat, bad)
    with pytest.raises(mklight.InvalidArgumentError):
        mklight.trace_grid(achromat, wavelengths=-1)
    assert mklight.trace(achromat, 0, np.int64(1))["wavelength_index"] == 1
