//! Typed shared-storage Metal buffers.
//!
//! `MTLStorageModeShared` buffers live in memory visible to both CPU and GPU
//! (unified memory on Apple Silicon), so fills and read views access the same pages. Host access goes through
//! `&self`/`&mut self` slices. GPU work is
//! encoded only by the crate-private `MetalContext::run`, which waits for the
//! command buffer; a buffer a kernel writes must be bound through a `&mut`
//! borrow that lasts until then, so no host slice of it (on any thread) can
//! exist while the GPU writes it.

use mklight_gpu::resident::HostAllocation;
use std::{cell::Cell, ptr::NonNull};
use std::{marker::PhantomData, sync::Arc};

use bytemuck::Pod;
use mklight_core::{MkError, MkResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

thread_local! {
    // Resident encoding is synchronous on one worker thread. Thread-local
    // accounting keeps concurrent jobs from charging one another's copies.
    static COPIED_BYTES: Cell<u64> = const { Cell::new(0) };
}
pub(crate) fn copied_bytes() -> u64 {
    COPIED_BYTES.get()
}

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
        let mut b = Self::allocate(device, len)?;
        // Shared buffers are documented to be zero-filled on creation; make it
        // explicit so results never depend on it.
        b.as_mut_slice().fill(T::zeroed());
        Ok(b)
    }

    fn allocate(device: &ProtocolObject<dyn MTLDevice>, len: usize) -> MkResult<Self> {
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
        Ok(SharedBuffer {
            raw,
            len,
            _t: PhantomData,
        })
    }

    /// A buffer of `len` elements written by `fill`, which must overwrite
    /// every element (anything it skips holds the zeros Metal initialises
    /// shared buffers with). Saves the explicit zeroing pass of
    /// [`SharedBuffer::zeroed`] for large uploads.
    pub fn filled(
        device: &ProtocolObject<dyn MTLDevice>,
        len: usize,
        fill: impl FnOnce(&mut [T]),
    ) -> MkResult<Self> {
        let mut b = Self::allocate(device, len)?;
        fill(b.as_mut_slice());
        Ok(b)
    }

    pub fn from_slice(device: &ProtocolObject<dyn MTLDevice>, data: &[T]) -> MkResult<Self> {
        let b = Self::filled(device, data.len(), |out| out.copy_from_slice(data))?;
        COPIED_BYTES.set(COPIED_BYTES.get() + size_of_val(data) as u64);
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

impl SharedBuffer<u32> {
    /// Transfer initialized page-owned storage to Metal. The block retains its
    /// owner until Metal releases the last buffer reference, including on error.
    pub fn from_host(
        device: &ProtocolObject<dyn MTLDevice>,
        data: HostAllocation,
    ) -> MkResult<Self> {
        let len = data.len();
        let bytes = data.allocation_bytes();
        if bytes as u64 > device.maxBufferLength() as u64 {
            return Err(MkError::Backend(
                "host allocation exceeds device limit".into(),
            ));
        }
        let owner = Arc::new(data);
        let ptr = NonNull::new(owner.as_ptr().cast_mut().cast()).unwrap();
        let release = block2::RcBlock::new(move |_: NonNull<std::ffi::c_void>, _: usize| {
            // Metal retains this block. Its captured owner frees the pages when
            // the block is released; the callback never exposes the allocation.
            let _keep_alive = &owner;
        });
        // SAFETY: HostAllocation supplies page-aligned initialized pages with a
        // page-multiple capacity. Ownership moved exclusively into the retained
        // deallocator block; no CPU alias remains. Tracked shared storage is
        // governed by the same synchronous GPU borrows as ordinary buffers.
        let raw = unsafe {
            device.newBufferWithBytesNoCopy_length_options_deallocator(
                ptr,
                bytes,
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeTracked,
                Some(&release),
            )
        }
        .ok_or_else(|| MkError::Backend("no-copy buffer allocation failed".into()))?;
        Ok(Self {
            raw,
            len,
            _t: PhantomData,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn staging_counter_has_a_copying_control() {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("host Metal required");
        let before = copied_bytes();
        let copied = SharedBuffer::from_slice(&device, &[1u32, 2, 3, 4]).unwrap();
        assert_eq!(copied_bytes() - before, 16);
        let filled = SharedBuffer::<u32>::filled(&device, 4, |v| v.fill(7)).unwrap();
        let pages = SharedBuffer::from_host(&device, HostAllocation::new(4).unwrap()).unwrap();
        assert_eq!(copied_bytes() - before, 16);
        assert_eq!(copied.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(filled.as_slice(), &[7; 4]);
        assert_eq!(pages.as_slice(), &[0; 4]);
    }
}
