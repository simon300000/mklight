//! Typed shared-storage Metal buffers.
//!
//! `MTLStorageModeShared` buffers live in memory visible to both CPU and GPU
//! (unified memory on Apple Silicon), so host fills and read views touch the
//! device's pages. Host access goes through `&self`/`&mut self` slices. GPU work is
//! encoded by synchronous commands or a worker-local resident batch. Deferred
//! bindings retain access guards; host slices reject pending writes (and host
//! mutation rejects any pending access). The resident executor pins recycled
//! leases through completion and publishes only completed immutable storage.

use mklight_gpu::resident::HostAllocation;
use std::{
    cell::Cell,
    marker::PhantomData,
    ptr::NonNull,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use bytemuck::Pod;
use mklight_core::{MkError, MkResult};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

thread_local! {
    // Bytes copied by `from_slice` on this thread. A resident plan encodes
    // synchronously on one worker, so its delta excludes concurrent jobs.
    static COPIED_BYTES: Cell<u64> = const { Cell::new(0) };
}
pub(crate) fn copied_bytes() -> u64 {
    COPIED_BYTES.get()
}

pub struct SharedBuffer<T: Pod> {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
    len: usize,
    _t: PhantomData<T>,
    access: Arc<AccessState>,
}

// SAFETY: Metal resource objects may be referenced from any thread (Apple's
// Metal threading guide: devices, queues, resources and pipeline states are
// thread-safe; only command buffers and encoders are not). Access to the
// buffer contents follow Rust borrows and the deferred device-access guards.
// Mutable plan allocations have a single worker; published storage is complete
// and immutable. Imported storage is only bound for device reads.
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
            access: Arc::default(),
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

    /// Panics if encoded, uncompleted device work may write this storage (or,
    /// for `write`, access it at all). Host slices and synchronous commands
    /// must not observe or race a deferred resident batch.
    pub(crate) fn assert_settled(&self, write: bool) {
        let pending = if write {
            &self.access.users
        } else {
            &self.access.writers
        };
        assert_eq!(
            pending.load(Ordering::Acquire),
            0,
            "buffer {} before resident flush",
            if write { "write" } else { "read" }
        );
    }

    pub fn as_slice(&self) -> &[T] {
        self.assert_settled(false);
        // SAFETY: `contents` points to at least `len * size_of::<T>()` bytes of
        // CPU-visible memory, suitably aligned (Metal buffers are page
        // aligned), and `T: Pod` accepts any bit pattern. GPU writes only
        // happen through exclusive unpublished storage, with the access guard
        // above excluding host slices until completion or abandonment.
        unsafe { std::slice::from_raw_parts(self.ptr().as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [T] {
        self.assert_settled(true);
        // SAFETY: as above; `&mut self` guarantees unique host access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr().as_ptr(), self.len) }
    }

    pub fn to_vec(&self) -> Vec<T> {
        self.as_slice().to_vec()
    }

    pub(crate) fn available(&self) -> bool {
        self.access.users.load(Ordering::Acquire) == 0
    }
    pub(crate) fn device_access(&self, write: bool) -> DeviceAccess {
        self.access.users.fetch_add(1, Ordering::AcqRel);
        if write {
            self.access.writers.fetch_add(1, Ordering::AcqRel);
        }
        DeviceAccess {
            state: self.access.clone(),
            write,
        }
    }
    pub fn raw(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

impl SharedBuffer<u32> {
    /// Wrap `data`'s pages without copying. The deallocator block owns `data`,
    /// so the pages are freed exactly once, when Metal releases the block
    /// after the buffer's last reference (or here, if creation fails).
    pub fn from_host(
        device: &ProtocolObject<dyn MTLDevice>,
        mut data: HostAllocation,
    ) -> MkResult<Self> {
        let len = data.len();
        let bytes = data.allocation_bytes();
        let ptr = NonNull::new(data.as_mut_ptr()).unwrap().cast();
        let release = block2::RcBlock::new(move |_: NonNull<std::ffi::c_void>, _: usize| {
            let _owner = &data;
        });
        // SAFETY: `ptr` and `bytes` are page-aligned and cover exactly the
        // owned allocation, which lives as long as the block. `data` was moved
        // in, so the only access is through the returned buffer's borrows.
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
            access: Arc::default(),
        })
    }
}

#[derive(Default)]
struct AccessState {
    users: AtomicUsize,
    writers: AtomicUsize,
}
/// Prevents host slices while an encoded command may access storage. The
/// command buffer retains the Metal allocation; the executor pins pool leases.
pub(crate) struct DeviceAccess {
    state: Arc<AccessState>,
    write: bool,
}
impl Drop for DeviceAccess {
    fn drop(&mut self) {
        if self.write {
            self.state.writers.fetch_sub(1, Ordering::AcqRel);
        }
        self.state.users.fetch_sub(1, Ordering::AcqRel);
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

    /// Deferred device accesses exclude conflicting host slices and
    /// synchronous bindings until their guards drop (completion/abandonment).
    #[test]
    fn pending_device_access_excludes_host_and_synchronous_use() {
        use std::panic::{AssertUnwindSafe, catch_unwind};
        let device = objc2_metal::MTLCreateSystemDefaultDevice().expect("host Metal required");
        let mut b = SharedBuffer::<u32>::filled(&device, 4, |v| v.fill(1)).unwrap();
        let read = b.device_access(false);
        assert!(!b.available());
        assert_eq!(b.as_slice(), &[1; 4]);
        b.assert_settled(false);
        assert!(catch_unwind(AssertUnwindSafe(|| b.as_mut_slice()[0] = 2)).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| b.assert_settled(true))).is_err());
        let write = b.device_access(true);
        assert!(catch_unwind(AssertUnwindSafe(|| b.as_slice().len())).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| b.assert_settled(false))).is_err());
        drop((read, write));
        assert!(b.available());
        b.as_mut_slice()[0] = 2;
        assert_eq!(b.as_slice(), &[2, 1, 1, 1]);
    }
}
