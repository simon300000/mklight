"""The exception hierarchy."""

import pytest

import mklight


@pytest.mark.parametrize(
    "cls,builtin",
    [
        (mklight.InvalidArgumentError, ValueError),
        (mklight.InvalidSettingsError, ValueError),
        (mklight.ParseError, ValueError),
        (mklight.InvalidModelError, ValueError),
        (mklight.NotFoundError, LookupError),
        (mklight.FileError, OSError),
        (mklight.UnsupportedError, NotImplementedError),
        (mklight.JobTimeoutError, TimeoutError),
        (mklight.CancelledError, None),
        (mklight.JobFailedError, None),
        (mklight.ComputeError, None),
        (mklight.InternalError, None),
    ],
)
def test_hierarchy(cls, builtin):
    assert issubclass(cls, mklight.MKLightError)
    assert issubclass(mklight.MKLightError, Exception)
    assert cls.__module__ == "mklight"
    if builtin is not None:
        assert issubclass(cls, builtin)
    assert cls.__doc__


def test_settings_error_is_an_argument_error():
    assert issubclass(mklight.InvalidSettingsError, mklight.InvalidArgumentError)


def test_engine_config_is_validated():
    with pytest.raises(mklight.InvalidArgumentError):
        mklight.Engine(worker_threads="many")
    e = mklight.Engine({"worker_threads": 1}, app_name="pytest")
    assert e.config["worker_threads"] == 1 and e.config["app_name"] == "pytest"
