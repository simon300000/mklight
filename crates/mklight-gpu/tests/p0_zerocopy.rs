use mklight_core::{JobContext, MkError, Snapshot};
use mklight_gpu::{resident::*, *};

/// Zero-copy contracts shared by CPU-f32 and Metal.
pub fn suite(b: &dyn ComputeBackend) {
    let ctx = JobContext::new();
    let before = b.resident_cache_stats().copied_bytes;

    // Imported pages feed a device sum in place: 3334 cancelling triplets, then 1e8.
    let mut pages = HostAllocation::new(10003).unwrap();
    for (i, v) in pages.iter_mut().enumerate() {
        *v = [1e8f32, 1., -1e8][i % 3].to_bits();
    }
    let address = pages.as_ptr();
    let imported = b
        .import_buffer(BufferDesc::new(ElementType::F32, [1, 10003]), pages)
        .unwrap();
    let mut view = imported.view();
    view.offset = 2;
    view.desc.shape = vec![3];
    let guard = b.read_view(view.clone()).unwrap();
    assert_eq!(guard.words().as_ptr(), address.wrapping_add(2));
    assert_eq!(guard.as_slice::<f32>().unwrap(), &[-1e8, 1e8, 1.]);
    assert!(guard.as_slice::<u32>().is_err());
    view.stride = 2;
    assert!(b.read_view(view.clone()).is_err());
    let strided = b.readback(view, &ctx).unwrap();
    assert_eq!(strided.data, BufferData::F32(vec![-1e8, 1., 1e8]));
    assert_eq!(strided.copied_bytes, 12);

    let mut p = ResidentPlan::new(Snapshot::new(testutil::singlet()).unwrap());
    let id = p.import(imported.view());
    let out = p.sum(p.view(id).unwrap(), ReduceMode::DoubleFloat);
    p.outputs = vec![out];
    drop(imported);
    let r = b.submit_plan(p, &ctx).unwrap().wait().unwrap();
    assert_eq!(r.stats.copied_bytes, 0);
    let pinned = b.read_view(r.outputs[0].view()).unwrap();
    assert_eq!(pinned.as_slice::<Df32>().unwrap()[0].to_f64(), 100003334.);
    drop(r);
    // Same-size fills must never recycle storage a guard still pins.
    for _ in 0..12 {
        b.fill_buffer(
            BufferDesc::new(ElementType::DoubleFloat, [1]),
            &mut |mut f| {
                f.as_mut_slice::<Df32>()?[0] = Df32::from_f32(-7.);
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(pinned.as_slice::<Df32>().unwrap()[0].to_f64(), 100003334.);
    }
    // The import guard outlived its handle and the plan that read it.
    assert_eq!(guard.as_slice::<f32>().unwrap(), &[-1e8, 1e8, 1.]);
    // Only the strided readback copied.
    assert_eq!(b.resident_cache_stats().copied_bytes - before, 12);

    // The copying conveniences count their bytes and preserve bits.
    let before = b.resident_cache_stats().copied_bytes;
    let copied = b
        .upload_buffer(
            BufferDesc::new(ElementType::F32, [2]),
            BufferData::F32(vec![-0., f32::from_bits(0x7fc00042)]),
        )
        .unwrap();
    let owned = b.readback(copied.view(), &ctx).unwrap();
    assert_eq!(owned.data.words(), &[0x80000000, 0x7fc00042]);
    assert_eq!(b.resident_cache_stats().copied_bytes - before, 16);

    let empty = b
        .fill_buffer(BufferDesc::new(ElementType::DoubleFloat, [0]), &mut |_| {
            Ok(())
        })
        .unwrap();
    let empty = b.read_view(empty.view()).unwrap();
    assert!(empty.as_slice::<Df32>().unwrap().is_empty());
    assert!(matches!(
        b.fill_buffer(BufferDesc::new(ElementType::F32, [16]), &mut |_| {
            Err(MkError::Cancelled)
        }),
        Err(MkError::Cancelled)
    ));
}
#[test]
fn cpu_shared_views() {
    suite(&CpuF32Backend::new());
}
#[test]
fn guard_outlives_backend() {
    let b = CpuF32Backend::new();
    let h = b
        .fill_buffer(
            BufferDesc::new(ElementType::Complex32, [2]),
            &mut |mut f| {
                f.as_mut_slice::<[f32; 2]>()?
                    .copy_from_slice(&[[1., 2.], [3., 4.]]);
                Ok(())
            },
        )
        .unwrap();
    let g = b.read_view(h.view()).unwrap();
    drop(h);
    drop(b);
    assert_eq!(g.as_slice::<[f32; 2]>().unwrap(), &[[1., 2.], [3., 4.]]);
}
