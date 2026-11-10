//! `mklight.Array`: a read-only, typed, row-major n-d array that holds an
//! engine `Buffer` and exports it through the Python buffer protocol.
//!
//! The buffer shares the payload the solver allocated (with the result it
//! came from and, for a stored result, the engine's analysis graph), so
//! `numpy.asarray(array)` wraps the engine's memory without a copy (the
//! numpy array is read-only and keeps the `Array` alive). Results
//! hand out numpy arrays when numpy is importable (and not disabled with
//! `mklight.set_numpy_enabled(False)`), otherwise nested lists.

use std::ffi::{CStr, c_int, c_void};
use std::sync::atomic::{AtomicBool, Ordering};

use pyo3::exceptions::{PyBufferError, PyIndexError};
use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyList, PyTuple};

use mklight_engine::{Buffer, BufferData, DType};

/// numpy dtype name of an engine dtype.
pub fn dtype_name(d: DType) -> &'static str {
    match d {
        DType::F64 => "float64",
        DType::F32 => "float32",
        DType::I32 => "int32",
        DType::U32 => "uint32",
        DType::I64 => "int64",
        DType::U64 => "uint64",
        DType::U8 => "uint8",
    }
}

/// `struct`-module format character (native size and alignment).
fn format(d: DType) -> &'static CStr {
    match d {
        DType::F64 => c"d",
        DType::F32 => c"f",
        DType::I32 => c"i",
        DType::U32 => c"I",
        DType::I64 => c"q",
        DType::U64 => c"Q",
        DType::U8 => c"B",
    }
}

/// Row-major strides in bytes.
pub fn c_strides(shape: &[usize], itemsize: usize) -> Vec<isize> {
    let mut s = vec![0isize; shape.len()];
    let mut acc = itemsize as isize;
    for i in (0..shape.len()).rev() {
        s[i] = acc;
        acc *= shape[i].max(1) as isize;
    }
    s
}

/// Whether a row-major array of `shape` is also column-major (Fortran)
/// contiguous: true when at most one axis has more than one element, or
/// the array is empty.
pub fn is_f_contiguous(shape: &[usize]) -> bool {
    shape.contains(&0) || shape.iter().filter(|&&n| n > 1).count() <= 1
}

#[pyclass(frozen, module = "mklight", name = "Array")]
pub struct Array {
    buf: Buffer,
    shape: Vec<isize>,
    strides: Vec<isize>,
}

impl Array {
    pub fn new(buf: Buffer) -> Self {
        let shape = buf.shape.iter().map(|&n| n as isize).collect();
        let strides = c_strides(&buf.shape, buf.dtype().size_bytes());
        Array {
            buf,
            shape,
            strides,
        }
    }

    fn item<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyAny>> {
        Ok(match self.buf.data() {
            BufferData::F64(v) => v[i].into_pyobject(py)?.into_any(),
            BufferData::F32(v) => (v[i] as f64).into_pyobject(py)?.into_any(),
            BufferData::I32(v) => v[i].into_pyobject(py)?.into_any(),
            BufferData::U32(v) => v[i].into_pyobject(py)?.into_any(),
            BufferData::I64(v) => v[i].into_pyobject(py)?.into_any(),
            BufferData::U64(v) => v[i].into_pyobject(py)?.into_any(),
            BufferData::U8(v) => v[i].into_pyobject(py)?.into_any(),
        })
    }

    fn nested<'py>(
        &self,
        py: Python<'py>,
        dim: usize,
        offset: usize,
    ) -> PyResult<Bound<'py, PyAny>> {
        let shape = &self.buf.shape;
        if dim == shape.len() {
            return self.item(py, offset);
        }
        let inner: usize = shape[dim + 1..].iter().product();
        let items = (0..shape[dim])
            .map(|k| self.nested(py, dim + 1, offset + k * inner))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(PyList::new(py, items)?.into_any())
    }
}

#[pymethods]
impl Array {
    /// Buffer name (as listed in the analysis result).
    #[getter]
    fn name(&self) -> &str {
        &self.buf.name
    }

    /// numpy dtype name (`"float64"`, `"int32"`, ...).
    #[getter]
    fn dtype(&self) -> &'static str {
        dtype_name(self.buf.dtype())
    }

    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, &self.buf.shape)
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.buf.shape.len()
    }

    /// Total number of elements.
    #[getter]
    fn size(&self) -> usize {
        self.buf.data().len()
    }

    fn __len__(&self) -> usize {
        self.buf.shape.first().copied().unwrap_or(0)
    }

    /// Nested Python lists (like `numpy.ndarray.tolist`).
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        if self.buf.shape.is_empty() {
            return Ok(PyList::empty(py).into_any());
        }
        self.nested(py, 0, 0)
    }

    /// Element at a flat (row-major) index.
    fn flat<'py>(&self, py: Python<'py>, index: usize) -> PyResult<Bound<'py, PyAny>> {
        if index >= self.buf.data().len() {
            return Err(PyIndexError::new_err("flat index out of range"));
        }
        self.item(py, index)
    }

    fn __repr__(&self) -> String {
        format!(
            "<mklight.Array '{}' {} shape={:?}>",
            self.buf.name,
            dtype_name(self.buf.dtype()),
            self.buf.shape
        )
    }

    /// Read-only buffer export (zero copy).
    unsafe fn __getbuffer__(
        slf: Bound<'_, Self>,
        view: *mut ffi::Py_buffer,
        flags: c_int,
    ) -> PyResult<()> {
        if view.is_null() {
            return Err(PyBufferError::new_err("view is null"));
        }
        // The buffer protocol wants `view.obj` NULL when the request fails.
        // SAFETY: `view` is a valid `Py_buffer` supplied by the interpreter.
        unsafe { (*view).obj = std::ptr::null_mut() };
        if (flags & ffi::PyBUF_WRITABLE) == ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("mklight arrays are read-only"));
        }
        let this = slf.get();
        // The data are row-major: C and "any" contiguity requests are
        // satisfied as is; a Fortran-order request only when the layouts
        // coincide (the buffer protocol forbids granting it otherwise).
        if (flags & ffi::PyBUF_F_CONTIGUOUS) == ffi::PyBUF_F_CONTIGUOUS
            && !is_f_contiguous(&this.buf.shape)
        {
            return Err(PyBufferError::new_err(
                "mklight arrays are C (row-major) contiguous, not Fortran contiguous",
            ));
        }
        let dtype = this.buf.dtype();
        let itemsize = dtype.size_bytes() as isize;
        // SAFETY: `view` is a valid `Py_buffer` supplied by the interpreter.
        // The data, shape and strides live in `this`, which is frozen
        // (never mutated) and kept alive by the strong reference stored in
        // `view.obj` until the buffer is released.
        unsafe {
            (*view).buf = this.buf.data().as_ptr() as *mut c_void;
            (*view).len = this.buf.data().len() as isize * itemsize;
            (*view).readonly = 1;
            (*view).itemsize = itemsize;
            (*view).format = if (flags & ffi::PyBUF_FORMAT) == ffi::PyBUF_FORMAT {
                format(dtype).as_ptr().cast_mut()
            } else {
                std::ptr::null_mut()
            };
            if (flags & ffi::PyBUF_ND) == ffi::PyBUF_ND {
                (*view).ndim = this.shape.len() as c_int;
                (*view).shape = this.shape.as_ptr().cast_mut();
            } else {
                (*view).ndim = 1;
                (*view).shape = std::ptr::null_mut();
            }
            (*view).strides = if (flags & ffi::PyBUF_STRIDES) == ffi::PyBUF_STRIDES {
                this.strides.as_ptr().cast_mut()
            } else {
                std::ptr::null_mut()
            };
            (*view).suboffsets = std::ptr::null_mut();
            (*view).internal = std::ptr::null_mut();
            (*view).obj = slf.into_any().into_ptr();
        }
        Ok(())
    }

    unsafe fn __releasebuffer__(&self, _view: *mut ffi::Py_buffer) {}
}

static USE_NUMPY: AtomicBool = AtomicBool::new(true);
static NUMPY: PyOnceLock<Option<Py<PyModule>>> = PyOnceLock::new();

/// Enable or disable numpy output (lists are returned when disabled or when
/// numpy is not installed).
pub fn set_numpy_enabled(on: bool) {
    USE_NUMPY.store(on, Ordering::Relaxed);
}

/// The numpy module, if it is importable and enabled.
pub fn numpy(py: Python<'_>) -> Option<Bound<'_, PyModule>> {
    if !USE_NUMPY.load(Ordering::Relaxed) {
        return None;
    }
    NUMPY
        .get_or_init(py, || py.import("numpy").ok().map(Bound::unbind))
        .as_ref()
        .map(|m| m.bind(py).clone())
}

/// A buffer as a numpy array (zero copy) or nested lists.
pub fn export<'py>(py: Python<'py>, buf: Buffer) -> PyResult<Bound<'py, PyAny>> {
    let arr = Bound::new(py, Array::new(buf))?;
    match numpy(py) {
        Some(np) => np.call_method1("asarray", (arr,)),
        None => arr.get().tolist(py),
    }
}

/// Convenience: export an owned buffer.
pub fn export_owned<'py>(
    py: Python<'py>,
    name: &str,
    shape: Vec<usize>,
    data: BufferData,
) -> PyResult<Bound<'py, PyAny>> {
    export(py, Buffer::new(name, shape, data))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strides_are_row_major() {
        assert_eq!(c_strides(&[2, 3, 4], 8), vec![96, 32, 8]);
        assert_eq!(c_strides(&[5], 4), vec![4]);
        assert_eq!(c_strides(&[], 8), Vec::<isize>::new());
    }

    #[test]
    fn fortran_contiguity() {
        assert!(is_f_contiguous(&[]));
        assert!(is_f_contiguous(&[7]));
        assert!(is_f_contiguous(&[1, 5, 1]));
        assert!(is_f_contiguous(&[0, 3]));
        assert!(!is_f_contiguous(&[5, 3]));
        assert!(!is_f_contiguous(&[2, 1, 2]));
    }

    #[test]
    fn dtype_names_match_sizes() {
        for (d, n) in [
            (DType::F64, 8),
            (DType::F32, 4),
            (DType::I32, 4),
            (DType::U32, 4),
            (DType::I64, 8),
            (DType::U64, 8),
            (DType::U8, 1),
        ] {
            assert_eq!(d.size_bytes(), n);
            let bits: usize = dtype_name(d)
                .trim_start_matches(char::is_alphabetic)
                .parse()
                .unwrap();
            assert_eq!(bits, 8 * n);
        }
    }
}
