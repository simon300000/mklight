"""Sessions: samples, system dicts, edits, files, catalogs, errors."""

import copy
import math
from pathlib import Path

import pytest

import mklight


@pytest.mark.parametrize("name", mklight.samples())
def test_every_sample_opens_validates_and_round_trips(engine, tmp_path, name):
    s = engine.new(sample=name)
    assert s.system["name"] == mklight.sample(name)["name"]
    r = s.run("validate")
    assert r.data["ok"] is True, r.data["issues"]
    assert r.snapshot_hash == s.snapshot_hash

    # Saving and re-opening gives the same snapshot (content hash).
    path = tmp_path / f"{name}.mklight"
    s.save(path)
    assert not s.dirty
    s2 = engine.open(path)
    assert s2.snapshot_hash == s.snapshot_hash
    assert s2.path == path
    assert s2.system == s.system


def test_samples_list_is_complete():
    names = mklight.samples()
    assert {"singlet", "achromat", "cooke_triplet", "double_gauss"} <= set(names)
    with pytest.raises(mklight.NotFoundError, match="achromat"):
        mklight.sample("no-such-sample")


def test_system_dict_round_trip_keeps_hash(achromat):
    h = achromat.snapshot_hash
    assert len(h) == 64
    assert achromat.set_system(achromat.system) == h
    # Object at infinity: the model writes the infinite thickness as "inf".
    assert achromat.system["surfaces"][0]["thickness"] == "inf"


def test_set_param_changes_system_and_hash(achromat):
    h0 = achromat.snapshot_hash
    sid = achromat.system["surfaces"][3]["id"]
    h1 = achromat.set_param({"param": "thickness", "surface": sid}, 90.0)
    assert h1 != h0 and h1 == achromat.snapshot_hash
    assert achromat.system["surfaces"][3]["thickness"] == 90.0
    assert achromat.dirty
    # Atomic multi-edit: one bad edit leaves everything unchanged.
    with pytest.raises(mklight.MKLightError):
        achromat.set_params(
            [
                {"path": {"param": "thickness", "surface": sid}, "value": 80.0},
                {"path": {"param": "thickness", "surface": 999}, "value": 1.0},
            ]
        )
    assert achromat.snapshot_hash == h1
    with pytest.raises(mklight.InvalidArgumentError):
        achromat.set_param({"param": "no_such_param"}, 1.0)


def test_invalid_system_is_rejected_and_session_unchanged(achromat):
    h = achromat.snapshot_hash
    bad = copy.deepcopy(achromat.system)
    bad["stop"] = 99
    with pytest.raises(mklight.InvalidModelError):
        achromat.set_system(bad)
    with pytest.raises(ValueError):  # InvalidModelError is a ValueError
        achromat.system = bad
    assert achromat.snapshot_hash == h
    with pytest.raises(mklight.ParseError):
        achromat.set_system({"surfaces": "nope"})


def test_new_from_dict_and_empty(engine):
    sysd = mklight.sample("cooke_triplet")
    s = engine.new(system=sysd)
    assert s.system == sysd
    empty = engine.new()
    assert len(empty.system["surfaces"]) == 3
    with pytest.raises(mklight.InvalidArgumentError):
        engine.new(sample="singlet", system=sysd)


def test_open_missing_file_raises_file_error(engine, tmp_path):
    with pytest.raises(mklight.FileError) as e:
        engine.open(tmp_path / "missing.mklight")
    assert isinstance(e.value, OSError)


AGF = """CC test catalog
NM MYBK7 2 517642 1.5168 64.17 0 0 0
CD 1.03961212 0.00600069867 0.231792344 0.0200179144 1.01046945 103.560653
LD 0.3 2.5
"""


def test_load_catalog_changes_library_hash(achromat, tmp_path):
    p = tmp_path / "mycat.agf"
    p.write_text(AGF)
    lib0 = achromat.library_hash
    info = achromat.load_catalog(p)
    assert info["name"] == "mycat" and info["glass_count"] == 1
    assert achromat.library_hash != lib0
    names = [c["name"] for c in achromat.catalogs()]
    assert names[0] == "BUILTIN" and "mycat" in names
    g = achromat.glasses("mycat")
    assert g[0]["name"] == "MYBK7" and math.isclose(g[0]["nd"], 1.5168)
    # The glass is usable by analyses through the session's library.
    r = achromat.run("glass_info", {"name": "MYBK7", "catalog": "mycat"})
    assert r.data["nd"] == pytest.approx(1.5168, abs=2e-4)
    assert r.library_hash == achromat.library_hash
    with pytest.raises(mklight.NotFoundError):
        achromat.glasses("nope")


def test_info_and_repr(achromat):
    info = achromat.info
    assert info["snapshot_hash"] == achromat.snapshot_hash
    assert info["path"] is None
    assert "Session" in repr(achromat)
    assert "Engine" in repr(achromat.engine)


def test_default_engine_helpers():
    s = mklight.new(sample="singlet")
    assert s.engine.config == mklight.default_engine().config


FIXTURES = Path(__file__).resolve().parents[2] / "mklight-core" / "tests" / "fixtures" / "prescriptions"


def test_import_prescription_replaces_the_system(engine):
    s = engine.new()
    h0 = s.snapshot_hash
    report = s.import_prescription(FIXTURES / "cooke_triplet.zmx")
    assert report["format"] == "zmx" and report["encoding"] == "utf16_le"
    assert report["system"]["name"] == "Cooke triplet f50 F/4"
    assert s.snapshot_hash != h0
    assert s.system["stop"] == 4
    assert mklight.paraxial(s)["efl"] == pytest.approx(50.0, abs=0.5)
    awkward = s.import_prescription(FIXTURES / "awkward.zmx")
    kinds = {w["kind"] for w in awkward["warnings"]}
    assert {"unknown_glass", "non_sequential", "unsupported_surface"} <= kinds
    with pytest.raises(mklight.UnsupportedError):
        s.import_prescription(FIXTURES / "missing.txt")
