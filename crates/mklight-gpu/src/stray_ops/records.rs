//! Numeric records shared with mklight_stray_ops_types.h. Coordinates, bounds
//! and random uniforms retain both words, including close-hit comparisons.
use crate::{ComputeBackend, Df32, resident::*};
use bytemuck::{Pod, Zeroable};
use mklight_core::{JobContext, MkResult};

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Source {
    pub kind: u32,
    pub wavelength_count: u32,
    pub center: [Df32; 3],
    pub axis: [Df32; 3],
    pub tangent: [[Df32; 3]; 2],
    pub radius: Df32,
    pub inner_radius: Df32,
    /// Point: 1-cos(alpha); Lambertian: sin(theta_max).
    pub cone: Df32,
    pub distance: Df32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Ray {
    pub origin: [Df32; 3],
    pub direction: [Df32; 3],
    pub id: [u32; 2],
    pub draws: u32,
    pub wavelength: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Primitive {
    /// 0 plane, 1 conic, 2 open cylinder.
    pub kind: u32,
    /// 0 none, 1 circle, 2 annulus, 3 rectangle, 4 ellipse; bit 8 inverts.
    pub mask: u32,
    pub origin: [Df32; 3],
    /// Row-major local-to-world rotation.
    pub rotation: [Df32; 9],
    pub lower: [Df32; 3],
    pub upper: [Df32; 3],
    pub curvature: Df32,
    pub conic: Df32,
    pub inner_radius: Df32,
    pub outer_radius: Df32,
    pub aperture: [Df32; 2],
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Node {
    pub lower: [Df32; 3],
    pub upper: [Df32; 3],
    /// Preorder escape index; a leaf has a primitive ID, an inner node MAX.
    pub escape: u32,
    pub primitive: u32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct Hit {
    pub point: [Df32; 3],
    pub normal: [Df32; 3],
    pub distance: Df32,
    /// Local coordinates/normal; MAX primitive and all-zero geometry on miss.
    pub primitive: u32,
    pub reserved: u32,
}
pub fn record_desc<T: Pod>(n: usize) -> BufferDesc {
    BufferDesc::new(ElementType::U32, [n, std::mem::size_of::<T>() / 4])
}
/// Produce each numeric record directly into shared storage.
pub fn fill_records<T: Pod>(
    b: &dyn ComputeBackend,
    n: usize,
    ctx: &JobContext,
    mut f: impl FnMut(usize) -> T,
) -> MkResult<BufferHandle> {
    try_fill_records(b, n, ctx, |i| Ok(f(i)))
}
/// Fallible direct producer; errors return the unpublished allocation to the pool.
pub fn try_fill_records<T: Pod>(
    b: &dyn ComputeBackend,
    n: usize,
    ctx: &JobContext,
    mut f: impl FnMut(usize) -> MkResult<T>,
) -> MkResult<BufferHandle> {
    b.fill_buffer(record_desc::<T>(n), &mut |mut dst| {
        let words = dst.as_mut_slice::<u32>()?;
        for (i, row) in bytemuck::cast_slice_mut::<u32, T>(words)
            .iter_mut()
            .enumerate()
        {
            if i % 256 == 0 {
                ctx.check()?;
            }
            *row = f(i)?;
        }
        Ok(())
    })
}
/// Borrow aligned records from a typed shared-buffer view without a copy.
pub fn read_records<T: Pod>(view: &ReadView) -> MkResult<&[T]> {
    bytemuck::try_cast_slice(view.as_slice::<u32>()?).map_err(|_| bad("stray record alignment"))
}
