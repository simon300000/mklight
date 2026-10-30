//! Typed bulk-data buffers that travel next to an analysis' JSON result (ray
//! coordinates, maps, curves). They cross the C ABI as `pointer + length +
//! dtype` without any copy or text encoding.

use serde::Serialize;

/// Element type. The numeric codes are part of the C ABI (`MKDType`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(i32)]
pub enum DType {
    F64 = 1,
    F32 = 2,
    I32 = 3,
    U32 = 4,
    I64 = 5,
    U64 = 6,
    U8 = 7,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::U8 => 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum BufferData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    I64(Vec<i64>),
    U64(Vec<u64>),
    U8(Vec<u8>),
}

impl BufferData {
    pub fn dtype(&self) -> DType {
        match self {
            BufferData::F64(_) => DType::F64,
            BufferData::F32(_) => DType::F32,
            BufferData::I32(_) => DType::I32,
            BufferData::U32(_) => DType::U32,
            BufferData::I64(_) => DType::I64,
            BufferData::U64(_) => DType::U64,
            BufferData::U8(_) => DType::U8,
        }
    }
    /// Number of elements.
    pub fn len(&self) -> usize {
        match self {
            BufferData::F64(v) => v.len(),
            BufferData::F32(v) => v.len(),
            BufferData::I32(v) => v.len(),
            BufferData::U32(v) => v.len(),
            BufferData::I64(v) => v.len(),
            BufferData::U64(v) => v.len(),
            BufferData::U8(v) => v.len(),
        }
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A named, row-major n-dimensional array. `shape` multiplies out to the
/// element count. The meaning of the axes is documented by each analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct Buffer {
    pub name: String,
    pub shape: Vec<usize>,
    pub data: BufferData,
}

/// JSON description of a buffer, listed in the result so front ends can find
/// buffers by name.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BufferDesc {
    pub index: usize,
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<usize>,
}

impl Buffer {
    /// Panics if `shape` does not match the data length: that is a bug in the
    /// analysis, not a user error.
    pub fn new(name: impl Into<String>, shape: Vec<usize>, data: BufferData) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "buffer shape {shape:?} does not match data");
        Buffer {
            name: name.into(),
            shape,
            data,
        }
    }
    /// A 1-D f64 buffer.
    pub fn f64_1d(name: impl Into<String>, v: Vec<f64>) -> Self {
        let n = v.len();
        Buffer::new(name, vec![n], BufferData::F64(v))
    }
    pub fn dtype(&self) -> DType {
        self.data.dtype()
    }
    pub fn desc(&self, index: usize) -> BufferDesc {
        BufferDesc {
            index,
            name: self.name.clone(),
            dtype: self.dtype(),
            shape: self.shape.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_and_dtype() {
        let b = Buffer::new("m", vec![2, 3], BufferData::F32(vec![0.0; 6]));
        assert_eq!(b.dtype(), DType::F32);
        assert_eq!(b.dtype().size_bytes(), 4);
        assert_eq!(
            serde_json::to_value(b.desc(0)).unwrap(),
            serde_json::json!({"index": 0, "name": "m", "dtype": "f32", "shape": [2, 3]})
        );
    }

    #[test]
    #[should_panic]
    fn shape_mismatch_panics() {
        let _ = Buffer::new("m", vec![2, 2], BufferData::U8(vec![0; 3]));
    }
}
