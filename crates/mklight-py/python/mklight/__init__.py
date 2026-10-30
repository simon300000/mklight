"""MKLight: sequential optical design and analysis.

The engine is written in Rust; this package is a thin, typed layer over its
facade, so Python, the command line and the macOS app share one code path.

Quick start::

    import mklight

    engine = mklight.Engine()
    session = engine.new(sample="achromat")
    print(mklight.paraxial(session)["efl"])          # ~100 mm
    result = session.run("system_summary")
    print(result.data["surfaces"][1]["radius"], result.arrays["n_after"].shape)
    grid = mklight.trace_grid(session, pupil=32)       # numpy arrays (F, W, P, ...)

Units and signs follow the engine conventions: millimetres, micrometre
wavelengths, degrees in settings, normalised field (Hx, Hy) and pupil
(Px, Py) coordinates. See docs/PYTHON.md in the repository.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from ._mklight import (
    ENGINE_VERSION,
    TRACE_STATUS,
    Array,
    CancelledError,
    ComputeError,
    Engine,
    FileError,
    InternalError,
    InvalidArgumentError,
    InvalidModelError,
    InvalidSettingsError,
    Job,
    JobFailedError,
    JobTimeoutError,
    MKLightError,
    NotFoundError,
    ParseError,
    Result,
    Session,
    UnsupportedError,
    __version__,
    numpy_enabled,
    paraxial,
    sample,
    samples,
    set_numpy_enabled,
    trace,
    trace_grid,
)

__all__ = [
    "ENGINE_VERSION",
    "TRACE_STATUS",
    "Array",
    "CancelledError",
    "ComputeError",
    "Engine",
    "FileError",
    "InternalError",
    "InvalidArgumentError",
    "InvalidModelError",
    "InvalidSettingsError",
    "Job",
    "JobFailedError",
    "JobTimeoutError",
    "MKLightError",
    "NotFoundError",
    "ParseError",
    "Result",
    "Session",
    "UnsupportedError",
    "__version__",
    "default_engine",
    "new",
    "numpy_enabled",
    "open",
    "paraxial",
    "sample",
    "samples",
    "set_numpy_enabled",
    "trace",
    "trace_grid",
]

_default_engine: Engine | None = None
_default_lock = threading.Lock()


def default_engine() -> Engine:
    """The process-wide engine used by :func:`open` and :func:`new`."""
    global _default_engine
    with _default_lock:
        if _default_engine is None:
            _default_engine = Engine()
        return _default_engine


def open(path: str | os.PathLike[str]) -> Session:  # noqa: A001 - mirrors Engine.open
    """Open a ``.mklight`` project with the default engine."""
    return default_engine().open(path)


def new(sample: str | None = None, system: dict[str, Any] | None = None) -> Session:
    """A new session on the default engine (see :meth:`Engine.new`)."""
    return default_engine().new(sample=sample, system=system)
