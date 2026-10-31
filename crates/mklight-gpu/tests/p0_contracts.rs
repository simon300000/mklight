use mklight_core::{JobContext, Snapshot};
use mklight_gpu::{resident::*, *};
fn snapshot() -> Snapshot {
    Snapshot::new(testutil::singlet()).unwrap()
}
#[test]
fn views_dependencies_and_leases() {
    let a = CpuF32Backend::new();
    let b = CpuF32Backend::new();
    let h = a
        .upload_buffer(
            BufferDesc::new(ElementType::F32, vec![6]),
            BufferData::F32(vec![1., 2., 3., 4., 5., 6.]),
        )
        .unwrap();
    let mut v = h.view();
    v.offset = 1;
    v.stride = 2;
    v.desc.shape = vec![3];
    assert_eq!(
        a.readback(v.clone(), &JobContext::new()).unwrap().data,
        BufferData::F32(vec![2., 4., 6.])
    );
    assert!(b.readback(v.clone(), &JobContext::new()).is_err());
    v.offset = 2;
    assert!(a.readback(v, &JobContext::new()).is_err());
    let mut p = ResidentPlan::new(snapshot());
    let id = p.import(h.view());
    let mut input = p.view(id).unwrap();
    input.desc.shape = vec![2, 3];
    let sum = p.sum(input, ReduceMode::DoubleFloat);
    p.outputs = vec![sum];
    assert!(b.submit_plan(p.clone(), &JobContext::new()).is_err());
    let r = a
        .submit_plan(p.clone(), &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    assert_eq!(
        a.readback(r.outputs[0].view(), &JobContext::new())
            .unwrap()
            .data,
        BufferData::DoubleFloat(vec![Df32::from_f32(6.), Df32::from_f32(15.)])
    );
    assert_eq!(r.run_info.precision, mklight_core::job::Precision::F32Gpu);
    assert_eq!(r.run_info.snapshot_hash, p.snapshot.hash().to_hex());
    p.operations.push(p.operations[0].clone());
    assert!(a.submit_plan(p, &JobContext::new()).is_err());
    let mut p = ResidentPlan::new(snapshot());
    let id = p.scratch(BufferDesc::new(ElementType::F32, vec![1, 4]));
    let sum = p.sum(p.view(id).unwrap(), ReduceMode::DoubleFloat);
    p.outputs = vec![sum];
    assert!(a.submit_plan(p, &JobContext::new()).is_err());
}
#[test]
fn ill_conditioned_sums_and_reuse() {
    let b = CpuF32Backend::new();
    let values: Vec<f32> = (0..10003)
        .map(|i| match i % 3 {
            0 => 1e8,
            1 => 1.,
            _ => -1e8,
        })
        .collect();
    let exact = values.iter().map(|&v| v as f64).sum::<f64>();
    let h = b
        .upload_buffer(
            BufferDesc::new(ElementType::F32, vec![1, values.len()]),
            BufferData::F32(values),
        )
        .unwrap();
    let mut plan = ResidentPlan::new(snapshot());
    let id = plan.import(h.view());
    let sum = plan.sum(plan.view(id).unwrap(), ReduceMode::DoubleFloat);
    plan.outputs = vec![sum];
    for _ in 0..3 {
        let r = b
            .submit_plan(plan.clone(), &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let BufferData::DoubleFloat(v) = b
            .readback(r.outputs[0].view(), &JobContext::new())
            .unwrap()
            .data
        else {
            panic!()
        };
        assert_eq!(v[0].hi as f64 + v[0].lo as f64, exact);
    }
    let ctx = JobContext::new();
    ctx.cancel.cancel();
    assert!(matches!(
        b.submit_plan(plan, &ctx).and_then(Pending::wait),
        Err(mklight_core::MkError::Cancelled)
    ));
}
#[test]
fn reserved_operation_is_explicitly_unsupported() {
    let b = CpuF32Backend::new();
    let mut p = ResidentPlan::new(snapshot());
    let id = p.scratch(BufferDesc::new(ElementType::Complex32, vec![4]));
    p.operations.push(ResidentOp::Transforms(
        mklight_gpu::transforms::Request::Transform(mklight_gpu::transforms::TransformPlan {
            kind: mklight_gpu::transforms::TransformKind::Fft,
            input: p.view(id).unwrap(),
            output: id,
            dimensions: vec![4],
            output_dimensions: vec![4],
            sign: 1,
            scale: 1.,
            affine: [Df32::ZERO; 6],
        }),
    ));
    assert!(matches!(
        b.submit_plan(p, &JobContext::new()),
        Err(mklight_core::MkError::Unsupported(_))
    ));
}
#[test]
fn immutable_cache_eviction_preserves_live_leases() {
    use mklight_gpu::resident::immutable::{ImmutableCache, OperationCacheKey};
    use std::sync::Arc;
    let b = CpuF32Backend::new();
    let key = |name: &str| OperationCacheKey {
        system: SystemCacheKey {
            snapshot: snapshot().hash().to_hex(),
            library: "test".into(),
            configuration: "base".into(),
            spectrum: "all".into(),
            layout: consts::MKL_LAYOUT_VERSION,
            kernel_build: b.precision().kernel_build.clone(),
            operation_settings: "fixed".into(),
        },
        operation: name.into(),
        plan_version: PLAN_VERSION,
    };
    let mut cache = ImmutableCache::new(8);
    let live = Arc::new([7u32]);
    cache.insert(key("a"), 4, live.clone());
    cache.insert(key("b"), 4, Arc::new([11u32]));
    cache.insert(key("c"), 4, Arc::new([13u32]));
    assert!(cache.get::<[u32; 1]>(&key("a")).unwrap().is_none());
    assert_eq!(*live, [7]);
    assert_eq!(cache.bytes(), 8);
    assert!(cache.get::<f32>(&key("b")).is_err());
    assert_eq!(*cache.get::<[u32; 1]>(&key("b")).unwrap().unwrap(), [11]);
    assert_eq!(cache.bytes(), 8);
}
