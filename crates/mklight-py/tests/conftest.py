import pytest

import mklight


@pytest.fixture(scope="session")
def engine() -> mklight.Engine:
    return mklight.Engine(worker_threads=2)


@pytest.fixture
def achromat(engine: mklight.Engine) -> mklight.Session:
    return engine.new(sample="achromat")


@pytest.fixture(autouse=True)
def _numpy_on():
    mklight.set_numpy_enabled(True)
    yield
    mklight.set_numpy_enabled(True)
