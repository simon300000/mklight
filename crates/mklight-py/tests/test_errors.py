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


def test_engine_config_rejects_unknown_keys():
    with pytest.raises(mklight.InvalidArgumentError, match="worker_thread"):
        mklight.Engine(worker_thread=2)
    with pytest.raises(mklight.InvalidArgumentError, match="bogus"):
        mklight.Engine({"bogus": 1})


def test_engine_default_backend_takes_the_run_forms():
    assert mklight.Engine(default_backend="cpu").default_backend == {"kind": "cpu_reference"}
    assert mklight.Engine(default_backend="auto").config["default_backend"] == {
        "kind": "auto"
    }
    assert mklight.Engine(default_backend={"kind": "cpu_reference"}).config[
        "default_backend"
    ] == {"kind": "cpu_reference"}
    with pytest.raises(mklight.InvalidArgumentError):
        mklight.Engine(default_backend=3)


def test_python_engines_retain_results_until_collected():
    assert mklight.Engine().config["max_retained_jobs"] == 2**64 - 1
    assert mklight.Engine(max_retained_jobs=5).config["max_retained_jobs"] == 5
