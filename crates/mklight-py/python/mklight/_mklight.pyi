"""Type stubs for the MKLight Rust extension module."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Callable, Sequence
from typing import Any, Literal, TypeAlias

ENGINE_VERSION: str
__version__: str
TRACE_STATUS: list[str]
"""Names of ``trace_grid``'s ``status`` codes: ok, miss, clipped,
total_internal_reflection, failed."""

JSON: TypeAlias = Any
"""A JSON-compatible value (dict / list / str / int / float / bool / None)."""
Backend: TypeAlias = str | dict[str, str] | None
"""``"auto"`` (automatic selection), ``None`` (engine default), ``"cpu"``, ``"metal"``, or
``{"kind": "gpu", "name": "metal"}``."""
Progress: TypeAlias = Callable[[float, str], object]
"""``progress(fraction, stage)``; fractions are monotone in [0, 1]."""
FieldArg: TypeAlias = int | tuple[float, float]
"""A system field index, or a normalised ``(hx, hy)`` point (a tuple)."""
FieldsArg: TypeAlias = int | tuple[float, float] | Sequence[int | tuple[float, float]]
"""``trace_grid`` fields: an index, one ``(hx, hy)`` tuple (read exactly as
``trace`` reads it), or a list of those. A list is always a list of fields:
``[0, 1]`` means system fields 0 and 1, ``(0, 1)`` the point hx=0, hy=1."""
ArrayLike: TypeAlias = Any
"""A numpy array (read-only, zero copy) or nested lists without numpy."""

class MKLightError(Exception): ...
class InvalidArgumentError(MKLightError, ValueError): ...
class InvalidSettingsError(InvalidArgumentError): ...
class ParseError(MKLightError, ValueError): ...
class InvalidModelError(MKLightError, ValueError): ...
class NotFoundError(MKLightError, LookupError): ...
class FileError(MKLightError, OSError): ...
class UnsupportedError(MKLightError, NotImplementedError): ...
class CancelledError(MKLightError): ...
class JobTimeoutError(MKLightError, TimeoutError): ...
class JobFailedError(MKLightError): ...
class ComputeError(MKLightError): ...
class InternalError(MKLightError): ...

class Array:
    """Read-only typed n-d array exporting the buffer protocol
    (``numpy.asarray(a)`` does not copy)."""

    @property
    def name(self) -> str: ...
    @property
    def dtype(self) -> str: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    def __len__(self) -> int: ...
    def tolist(self) -> list[Any]: ...
    def flat(self, index: int) -> float | int: ...
    def __buffer__(self, flags: int, /) -> memoryview: ...

class Engine:
    """Worker pool, analysis registry and compute backends."""

    def __init__(self, config: dict[str, JSON] | None = None, **kwargs: JSON) -> None:
        """Keys: ``worker_threads``, ``max_retained_jobs`` (default: unlimited;
        results are kept until collected or their Job is dropped),
        ``resolved_cache_capacity``, ``analysis_graph_bytes``,
        ``progress_interval_ms``, ``app_name``, ``app_version``,
        ``default_backend`` (a :data:`Backend`), ``aim_stop``
        (``"conditioned"`` or ``"requested"``). Unknown keys raise
        ``InvalidArgumentError``."""
    def new(self, sample: str | None = None, system: dict[str, JSON] | None = None) -> Session:
        """A new session from a built-in sample, a system dict, or empty."""
    def open(self, path: str | os.PathLike[str]) -> Session:
        """Open a ``.mklight`` project (only the current schema version)."""
    def analyses(self) -> list[dict[str, JSON]]: ...
    def backends(self) -> list[dict[str, JSON]]: ...
    def jobs(self) -> list[dict[str, JSON]]: ...
    def cancel_all(self) -> None: ...
    @property
    def config(self) -> dict[str, JSON]: ...
    @property
    def default_backend(self) -> dict[str, JSON]: ...

class Session:
    """One open document: system snapshots, glass catalogs, analyses."""

    @property
    def system(self) -> dict[str, JSON]:
        """The ``OpticalSystem`` model (a fresh copy on every access)."""
    @system.setter
    def system(self, value: dict[str, JSON]) -> None: ...
    def set_system(self, system: dict[str, JSON]) -> str:
        """Replace and validate the system; returns the snapshot hash."""
    def set_param(self, path: dict[str, JSON], value: float | str) -> str:
        """Set one parameter (``{"param": "thickness", "surface": 3}``)."""
    def set_params(self, edits: Sequence[dict[str, JSON]]) -> str:
        """Apply ``{"path": ..., "value": ...}`` edits atomically."""
    @property
    def snapshot_hash(self) -> str: ...
    @property
    def library_hash(self) -> str: ...
    @property
    def id(self) -> int: ...
    @property
    def path(self) -> pathlib.Path | None: ...
    @property
    def dirty(self) -> bool: ...
    @property
    def info(self) -> dict[str, JSON]: ...
    @property
    def engine(self) -> Engine: ...
    def save(self, path: str | os.PathLike[str] | None = None) -> None: ...
    def load_catalog(self, path: str | os.PathLike[str]) -> dict[str, JSON]: ...
    def import_prescription(self, path: str | os.PathLike[str]) -> dict[str, JSON]: ...
    def catalogs(self) -> list[dict[str, JSON]]: ...
    def glasses(self, catalog: str | None = None) -> list[dict[str, JSON]]: ...
    def set_project_section(self, section: str, value: JSON) -> None: ...
    def run(
        self,
        analysis: str,
        settings: dict[str, JSON] | None = None,
        seed: int | None = None,
        backend: Backend = None,
        progress: Progress | None = None,
        allow_backend_fallback: bool = False,
        timeout: float | None = None,
    ) -> Result:
        """Run an analysis and wait (GIL released; Ctrl-C cancels)."""
    def submit(
        self,
        analysis: str,
        settings: dict[str, JSON] | None = None,
        seed: int | None = None,
        backend: Backend = None,
        allow_backend_fallback: bool = False,
        view_key: str | None = None,
    ) -> Job:
        """Queue an analysis; returns at once."""

class Job:
    """A queued analysis. Dropping an unfinished Job cancels it."""

    @property
    def id(self) -> int: ...
    @property
    def analysis(self) -> str: ...
    @property
    def state(self) -> Literal["queued", "running", "done", "failed", "cancelled", "evicted"]:
        """``evicted``: the engine discarded the outcome before it was
        collected (only with an explicit ``Engine(max_retained_jobs=...)``);
        ``result()`` then raises ``NotFoundError``."""
    @property
    def progress(self) -> float:
        """Monotone fraction in [0, 1]."""
    @property
    def stage(self) -> str: ...
    def status(self) -> dict[str, JSON] | None: ...
    def done(self) -> bool: ...
    def stop(self) -> None:
        """Stop optimize/tolerance and keep the best-so-far result (stopped=True)."""
        ...
    def cancel(self) -> None: ...
    def wait(self, timeout: float | None = None) -> bool:
        """True once finished; False after ``timeout`` seconds (``None`` or
        ``inf`` wait forever; NaN or negative raise InvalidArgumentError)."""
    def result(self, timeout: float | None = None, progress: Progress | None = None) -> Result:
        """Wait for and return the result (raises CancelledError,
        JobTimeoutError, or the analysis' error)."""

class Result:
    """A finished analysis."""

    @property
    def data(self) -> dict[str, JSON]: ...
    @property
    def run_info(self) -> dict[str, JSON]: ...
    @property
    def arrays(self) -> dict[str, ArrayLike]: ...
    def buffers(self) -> list[Array]: ...
    @property
    def analysis(self) -> str: ...
    @property
    def snapshot_hash(self) -> str: ...
    @property
    def library_hash(self) -> str: ...
    @property
    def seed(self) -> int | None: ...
    @property
    def warnings(self) -> list[str]: ...
    def envelope(self) -> dict[str, JSON]: ...
    def __getitem__(self, key: str) -> JSON: ...

def samples() -> list[str]: ...
def sample(name: str) -> dict[str, JSON]: ...
def set_numpy_enabled(enabled: bool) -> None: ...
def numpy_enabled() -> bool: ...
def trace(
    session: Session,
    field: FieldArg | None = None,
    wavelength: int | None = None,
    px: float = 0.0,
    py: float = 0.0,
    records: bool = True,
    clip: bool = True,
) -> dict[str, Any]:
    """Trace one ray with the CPU f64 reference tracer."""
def trace_grid(
    session: Session,
    fields: FieldsArg | None = None,
    wavelengths: int | Sequence[int] | None = None,
    pupil: int | dict[str, JSON] | Sequence[Sequence[float]] | ArrayLike | None = None,
    clip: bool = True,
    progress: Progress | None = None,
) -> dict[str, ArrayLike]:
    """Trace fields x wavelengths x pupil points. ``pupil``: an int ``n``
    (``n x n`` grid inside the unit disk, 1 to 4096; default 16), a pattern
    dict or ``(P, 2)`` points."""
def paraxial(session: Session, wavelength: int | None = None) -> dict[str, Any]:
    """First-order properties at one wavelength (default: primary)."""
