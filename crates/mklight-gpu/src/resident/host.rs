//! Page-owned host storage, transferable to Metal without copying.
use super::{MkResult, bad};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// Alignment and capacity granule. A multiple of every Apple page size (4 KiB
/// x86_64, 16 KiB arm64), as `newBufferWithBytesNoCopy` requires.
pub const PAGE: usize = 16384;

/// Zero-initialised, `PAGE`-aligned words with a `PAGE`-multiple capacity.
pub struct HostAllocation {
    ptr: NonNull<u32>,
    words: usize,
    layout: Layout,
}
// SAFETY: a unique owner of plain words; access follows Rust borrows.
unsafe impl Send for HostAllocation {}
unsafe impl Sync for HostAllocation {}
impl HostAllocation {
    pub fn new(words: usize) -> MkResult<Self> {
        let layout = words
            .checked_mul(4)
            .and_then(|n| n.max(1).checked_next_multiple_of(PAGE))
            .and_then(|n| Layout::from_size_align(n, PAGE).ok())
            .ok_or_else(|| bad("host allocation size"))?;
        // SAFETY: nonzero size; zeroed memory is a valid [u32].
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| bad("host allocation failed"))?
            .cast();
        Ok(Self { ptr, words, layout })
    }
    /// Whole pages owned (the length a no-copy wrapper must declare).
    pub fn allocation_bytes(&self) -> usize {
        self.layout.size()
    }
}
impl Deref for HostAllocation {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        // SAFETY: `words` initialised u32s, live until drop.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.words) }
    }
}
impl DerefMut for HostAllocation {
    fn deref_mut(&mut self) -> &mut [u32] {
        // SAFETY: as above; `&mut self` is the only access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.words) }
    }
}
impl Drop for HostAllocation {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout; dropped once.
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) }
    }
}
