"""First-order data from mklight.paraxial."""

import math

import pytest

import mklight


def test_achromat_efl_is_100mm(achromat):
    p = mklight.paraxial(achromat)
    assert p["efl"] == pytest.approx(100.0, abs=0.05)
    assert p["wavelength_index"] == p["primary_wavelength"]
    # F/5 image-space F-number by construction.
    assert p["image_f_number"] == pytest.approx(5.0, rel=1e-9)
    # The image surface sits at the paraxial focus (samples are focused).
    last_t = achromat.system["surfaces"][3]["thickness"]
    assert p["image_distance"] == pytest.approx(last_t, abs=1e-5)
    n = len(achromat.system["surfaces"])
    assert len(p["marginal"]["y"]) == n
    assert p["marginal"]["y"][-1] == pytest.approx(0.0, abs=1e-6)


def test_paraxial_per_wavelength_and_colour(achromat):
    efls = [mklight.paraxial(achromat, w)["efl"] for w in range(3)]
    # An achromat has nearly the same EFL at F and C.
    assert abs(efls[0] - efls[2]) < 0.1
    with pytest.raises(mklight.InvalidArgumentError):
        mklight.paraxial(achromat, 7)


def test_infinite_values_stay_floats(engine):
    # A plane-parallel plate is afocal: EFL, BFL and the image-space
    # F-number are infinite and must come back as float infinities (not the
    # model's "inf" strings, and not None).
    s = engine.new(sample="singlet")
    finite = mklight.paraxial(s)
    assert finite["efl"] == pytest.approx(100.0, rel=0.01)
    sys = s.system
    for surf in sys["surfaces"][1:3]:
        surf["shape"]["curvature"] = 0.0
    s.set_system(sys)
    p = mklight.paraxial(s)
    for key in ("efl", "bfl", "image_f_number"):
        assert isinstance(p[key], float), key
        assert math.isinf(p[key]), (key, p[key])
    for k, v in p.items():
        if isinstance(v, float):
            assert not math.isnan(v), k
