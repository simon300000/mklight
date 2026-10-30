//! JSON values ↔ Python objects.
//!
//! The engine's structured data is serde JSON (`docs/FFI.md`), so Python
//! sees exactly the documented model: dicts, lists, `str`, `int`, `float`,
//! `bool` and `None`. JSON cannot hold infinities; the model writes `±∞` as
//! the strings `"inf"` / `"-inf"` (CONVENTIONS §2), so a Python `float`
//! infinity is converted to that string on the way in, and NaN to `None`.

use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};

/// A finite `f64` as a JSON number; `±∞` as `"inf"`/`"-inf"`; NaN as null.
pub fn f64_to_value(x: f64) -> Value {
    if x.is_finite() {
        Number::from_f64(x).map_or(Value::Null, Value::Number)
    } else if x.is_nan() {
        Value::Null
    } else if x > 0.0 {
        Value::String("inf".into())
    } else {
        Value::String("-inf".into())
    }
}

/// JSON → Python.
pub fn to_py<'py>(py: Python<'py>, v: &Value) -> PyResult<Bound<'py, PyAny>> {
    Ok(match v {
        Value::Null => py.None().into_bound(py),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.into_pyobject(py)?.into_any()
            } else if let Some(u) = n.as_u64() {
                u.into_pyobject(py)?.into_any()
            } else {
                PyFloat::new(py, n.as_f64().unwrap_or(f64::NAN)).into_any()
            }
        }
        Value::String(s) => PyString::new(py, s).into_any(),
        Value::Array(a) => {
            let items = a
                .iter()
                .map(|x| to_py(py, x))
                .collect::<PyResult<Vec<_>>>()?;
            PyList::new(py, items)?.into_any()
        }
        Value::Object(o) => {
            let d = PyDict::new(py);
            for (k, x) in o {
                d.set_item(k, to_py(py, x)?)?;
            }
            d.into_any()
        }
    })
}

/// Python → JSON. Accepts dicts with string keys, lists, tuples, `str`,
/// `bool`, `int`, `float`, `None`, and anything with `__index__`,
/// `__float__` or `tolist()` (numpy scalars and arrays).
pub fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    from_py_depth(obj, 0)
}

fn from_py_depth(obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<Value> {
    if depth > 256 {
        return Err(PyTypeError::new_err("structure nested too deeply"));
    }
    if obj.is_none() {
        return Ok(Value::Null);
    }
    // `bool` is a subclass of `int`: test it first.
    if let Ok(b) = obj.cast::<PyBool>() {
        return Ok(Value::Bool(b.is_true()));
    }
    if obj.is_instance_of::<PyInt>() {
        if let Ok(i) = obj.extract::<i64>() {
            return Ok(Value::from(i));
        }
        if let Ok(u) = obj.extract::<u64>() {
            return Ok(Value::from(u));
        }
        return Err(PyTypeError::new_err("integer out of 64-bit range"));
    }
    if let Ok(f) = obj.cast::<PyFloat>() {
        return Ok(f64_to_value(f.value()));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        return Ok(Value::String(s.to_str()?.to_owned()));
    }
    if let Ok(d) = obj.cast::<PyDict>() {
        let mut m = Map::new();
        for (k, v) in d.iter() {
            let key = k
                .cast::<PyString>()
                .map_err(|_| PyTypeError::new_err(format!("dict keys must be str, got {k:?}")))?
                .to_str()?
                .to_owned();
            m.insert(key, from_py_depth(&v, depth + 1)?);
        }
        return Ok(Value::Object(m));
    }
    if let Ok(l) = obj.cast::<PyList>() {
        return l
            .iter()
            .map(|x| from_py_depth(&x, depth + 1))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    if let Ok(t) = obj.cast::<PyTuple>() {
        return t
            .iter()
            .map(|x| from_py_depth(&x, depth + 1))
            .collect::<PyResult<Vec<_>>>()
            .map(Value::Array);
    }
    // numpy and friends.
    if obj.hasattr("tolist")? {
        let l = obj.call_method0("tolist")?;
        if !l.is(obj) {
            return from_py_depth(&l, depth + 1);
        }
    }
    if obj.hasattr("__index__")? {
        let i = obj.call_method0("__index__")?;
        return from_py_depth(&i, depth + 1);
    }
    if obj.hasattr("__float__")? {
        let f: f64 = obj.call_method0("__float__")?.extract()?;
        return Ok(f64_to_value(f));
    }
    Err(PyTypeError::new_err(format!(
        "cannot convert {} to JSON",
        obj.get_type().name()?
    )))
}

/// Serialize any serde value to Python through JSON.
pub fn ser_to_py<'py, T: serde::Serialize>(py: Python<'py>, v: &T) -> PyResult<Bound<'py, PyAny>> {
    let v = serde_json::to_value(v).map_err(|e| crate::errors::core_err(e.into()))?;
    to_py(py, &v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_finite_floats_follow_the_model_convention() {
        assert_eq!(f64_to_value(f64::INFINITY), Value::String("inf".into()));
        assert_eq!(
            f64_to_value(f64::NEG_INFINITY),
            Value::String("-inf".into())
        );
        assert_eq!(f64_to_value(f64::NAN), Value::Null);
        assert_eq!(f64_to_value(2.5), serde_json::json!(2.5));
    }
}
