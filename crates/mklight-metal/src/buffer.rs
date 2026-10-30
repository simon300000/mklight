//! Typed shared-storage Metal buffers.
//!
//! `MTLStorageModeShared` buffers live in memory visible to both CPU and GPU
//! (unified memory on Apple Silicon), so uploads and readbacks are plain
//! memcpys. Host access goes through `&self`/`&mut self` slices. GPU work is
//! encoded only by the crate-private `MetalContext::run`, which waits for the
//! command buffer; a buffer a kernel writes must be bound through a `&mut`
//! borrow that lasts until then, so no host slice of it (on any thread) can
//! exist while the GPU writes it.

use std::marker::PhantomData;
use std::ptr::NonNull;

use bytemuck::Pod;
use mklight_core::{MkError, MkResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

pub struct SharedBuffer<T: Pod> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    _t: PhantomData<T>,
}

// SAFETY: Metal resource objects may be referenced from any thread (Apple's
// Metal threading guide: devices, queues, resources and pipeline states are
// thread-safe; only command buffers and encoders are not). Access to the
// buffer *contents* follows Rust borrows: shared reads via `&self`, writes
// via `&mut self`; GPU writes only through a `&mut` borrow held by
// `MetalContext::run` until the command buffer completes, GPU reads through a
// shared borrow held equally long.
unsafe impl<T: Pod + Send> Send for SharedBuffer<T> {}
unsafe impl<T: Pod + Sync> Sync for SharedBuffer<T> {}

impl<T: Pod> SharedBuffer<T> {
    /// A zero-initialised buffer of `len` elements (at least 16 bytes are
    /// allocated, since Metal rejects empty buffers).
    pub fn zeroed(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> MkResult<Self> {
        let bytes = len
            .checked_mul(size_of::<T>())
            .ok_or_else(|| MkError::Backend("buffer size overflow".into()))?;
        if bytes as u64 > device.maxBufferLength() as u64 {
            return Err(MkError::Backend(format!(
                "buffer of {bytes} bytes exceeds the device limit"
            )));
        }
        let raw = device
            .newBufferWithLength_options(
                bytes.max(16),
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeTracked,
            )
            .ok_or_else(|| MkError::Backend(format!("allocation of {bytes} bytes failed")))?;
        // Shared buffers are documented to be zero-filled on creation; make it
        // explicit so results never depend on it.
        let mut b = SharedBuffer {
            raw,
            len,
            _t: PhantomData,
        };
        b.as_mut_slice().fill(T::zeroed());
        Ok(b)
    }

    pub fn from_slice(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> MkResult<Self> {
        let mut b = Self::zeroed(device, data.len())?;
        b.as_mut_slice().copy_from_slice(data);
        Ok(b)
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn byte_len(&self) -> usize {
        self.len * size_of::<T>()
    }

    fn ptr(&self) -> NonNull<T> {
        self.raw.contents().cast()
    }

    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `contents` points to at least `len * size_of::<T>()` bytes of
        // CPU-visible memory, suitably aligned (Metal buffers are page
        // aligned), and `T: Pod` accepts any bit pattern. GPU writes only
        // happen inside `MetalContext::run` through a `&mut` borrow held until
        // the command buffer has completed, which excludes this `&self`.
        unsafe { std::slice::from_raw_parts(self.ptr().as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: as above; `&mut self` guarantees unique host access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr().as_ptr(), self.len) }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}
