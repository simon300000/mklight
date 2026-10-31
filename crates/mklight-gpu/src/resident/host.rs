//! Page-owned host storage, transferable to Metal without copying.
use super::{MkResult, bad};
use std::{
    alloc::{Layout, alloc_zeroed, dealloc},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

/// An initialized, page-aligned allocation with a page-multiple capacity.
/// Moving this owner into `import_buffer` relinquishes all mutable CPU access.
/// The allocation is freed exactly once, after the last backend reference ends.
pub struct HostAllocation {
    ptr: NonNull<u32>,
    words: usize,
    layout: Layout,
}
// SAFETY: access to initialized words follows exclusive/shared Rust borrows.
unsafe impl Send for HostAllocation {}
unsafe impl Sync for HostAllocation {}
impl HostAllocation {
    pub fn new(words: usize) -> MkResult<Self> {
        #[cfg(unix)]
        // SAFETY: sysconf has no pointer arguments.
        let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
        #[cfg(not(unix))]
        let page = 65536;
        let bytes = words
            .checked_mul(4)
            .and_then(|n| n.max(1).checked_next_multiple_of(page))
            .ok_or_else(|| bad("host allocation size overflow"))?;
        let layout =
            Layout::from_size_align(bytes, page).map_err(|_| bad("host allocation layout"))?;
        // SAFETY: nonzero valid layout; all u32 bit patterns are valid.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) })
            .ok_or_else(|| bad("host allocation failed"))?
            .cast();
        Ok(Self { ptr, words, layout })
    }
    pub fn allocation_bytes(&self) -> usize {
        self.layout.size()
    }
}
impl Deref for HostAllocation {
    type Target = [u32];
    fn deref(&self) -> &[u32] {
        // SAFETY: allocation owns at least words initialized u32s until drop.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.words) }
    }
}
impl DerefMut for HostAllocation {
    fn deref_mut(&mut self) -> &mut [u32] {
        // SAFETY: the unique borrow excludes other CPU/GPU access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.words) }
    }
}
impl Drop for HostAllocation {
    fn drop(&mut self) {
        // SAFETY: original pointer and layout, with unique ownership.
        unsafe { dealloc(self.ptr.as_ptr().cast(), self.layout) }
    }
}
