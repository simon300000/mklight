use mklight_core::{JobContext, MkError, Snapshot};
use mklight_gpu::{resident::*, *};

pub fn suite(b: &dyn ComputeBackend) {
    let ctx = JobContext::new();
    let before = b.resident_cache_stats().copied_bytes;
    let h = b
        .fill_buffer(
            BufferDesc::new(ElementType::F32, [1, 10003]),
            &ctx,
            &mut |mut fill| {
                for (i, v) in fill.as_mut_slice::<f32>()?.iter_mut().enumerate() {
                    *v = match i % 3 {
                        0 => 1e8,
                        1 => 1.,
                        _ => -1e8,
                    };
                }
                Ok(())
            },
        )
        .unwrap();
    let mut p = ResidentPlan::new(Snapshot::new(testutil::singlet()).unwrap());
    let id = p.import(h.view());
    let out = p.sum(p.view(id).unwrap(), ReduceMode::DoubleFloat);
    p.outputs = vec![out];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(r.stats.copied_bytes, 0);
    let pinned = b
        .read_view(r.outputs[0].view(), &JobContext::new())
        .unwrap();
    assert_eq!(pinned.copied_bytes, 0);
    // Independent exact integer sum: 3334 complete triplets, then 1e8.
    assert_eq!(pinned.as_slice::<Df32>().unwrap()[0].to_f64(), 100003334.);
    assert!(pinned.as_slice::<f32>().is_err());
    drop(r);
    // Repeated writes at the same size must never recycle the viewed storage.
    for _ in 0..12 {
        let replacement = b
            .fill_buffer(
                BufferDesc::new(ElementType::DoubleFloat, [1]),
                &JobContext::new(),
                &mut |mut f| {
                    f.as_mut_slice::<Df32>()?[0] = Df32::from_f32(-7.);
                    Ok(())
                },
            )
            .unwrap();
        drop(replacement);
        assert_eq!(pinned.as_slice::<Df32>().unwrap()[0].to_f64(), 100003334.);
    }
    assert_eq!(b.resident_cache_stats().copied_bytes, before);

    let mut pages = HostAllocation::new(10).unwrap();
    for (i, v) in pages.iter_mut().enumerate() {
        *v = i as u32 + 1;
    }
    let address = pages.as_ptr();
    let imported = b
        .import_buffer(BufferDesc::new(ElementType::U32, [10]), pages)
        .unwrap();
    let mut view = imported.view();
    view.offset = 2;
    view.desc.shape = vec![3];
    let guard = b.read_view(view.clone(), &JobContext::new()).unwrap();
    assert_eq!(guard.words().as_ptr(), address.wrapping_add(2));
    assert_eq!(guard.as_slice::<u32>().unwrap(), &[3, 4, 5]);
    view.stride = 2;
    assert!(b.read_view(view.clone(), &JobContext::new()).is_err());
    let copied = b.readback(view, &JobContext::new()).unwrap();
    assert_eq!(copied.data, BufferData::U32(vec![3, 5, 7]));
    assert_eq!(copied.copied_bytes, 12);
    assert_eq!(b.resident_cache_stats().copied_bytes - before, 12);
    drop(imported);
    assert_eq!(guard.as_slice::<u32>().unwrap(), &[3, 4, 5]);

    let before = b.resident_cache_stats().copied_bytes;
    let copied_input = b
        .upload_buffer(
            BufferDesc::new(ElementType::F32, [2]),
            BufferData::F32(vec![-0., f32::from_bits(0x7fc00042)]),
        )
        .unwrap();
    assert_eq!(b.resident_cache_stats().copied_bytes - before, 8);
    let owned = b.readback(copied_input.view(), &JobContext::new()).unwrap();
    assert_eq!(owned.data.words(), &[0x80000000, 0x7fc00042]);
    assert_eq!(owned.copied_bytes, 8);
    assert_eq!(b.resident_cache_stats().copied_bytes - before, 16);
    for ty in [
        ElementType::F32,
        ElementType::U32,
        ElementType::DoubleFloat,
        ElementType::Complex32,
    ] {
        let empty = b
            .fill_buffer(
                BufferDesc::new(ty, [0]),
                &JobContext::new(),
                &mut |mut fill| {
                    match ty {
                        ElementType::F32 => assert!(fill.as_mut_slice::<f32>()?.is_empty()),
                        ElementType::U32 => assert!(fill.as_mut_slice::<u32>()?.is_empty()),
                        ElementType::DoubleFloat => {
                            assert!(fill.as_mut_slice::<Df32>()?.is_empty())
                        }
                        ElementType::Complex32 => {
                            assert!(fill.as_mut_slice::<[f32; 2]>()?.is_empty())
                        }
                    }
                    Ok(())
                },
            )
            .unwrap();
        assert!(
            b.read_view(empty.view(), &JobContext::new())
                .unwrap()
                .words()
                .is_empty()
        );
    }
    let cancelled = JobContext::new();
    cancelled.cancel.cancel();
    assert!(matches!(
        b.fill_buffer(
            BufferDesc::new(ElementType::F32, [16]),
            &cancelled,
            &mut |_| panic!("cancelled fill ran")
        ),
        Err(MkError::Cancelled)
    ));
    assert!(matches!(
        b.read_view(h.view(), &cancelled),
        Err(MkError::Cancelled)
    ));
    let h = b.fill_buffer(
        BufferDesc::new(ElementType::F32, [16]),
        &JobContext::new(),
        &mut |_| Err(MkError::Cancelled),
    );
    assert!(matches!(h, Err(MkError::Cancelled)));
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
            &JobContext::new(),
            &mut |mut f| {
                f.as_mut_slice::<[f32; 2]>()?
                    .copy_from_slice(&[[1., 2.], [3., 4.]]);
                Ok(())
            },
        )
        .unwrap();
    let g = b.read_view(h.view(), &JobContext::new()).unwrap();
    drop(h);
    drop(b);
    assert_eq!(g.as_slice::<[f32; 2]>().unwrap(), &[[1., 2.], [3., 4.]]);
}
