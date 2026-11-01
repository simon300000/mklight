use mklight_core::{JobContext, Snapshot, samples};
use mklight_gpu::{collectives as c, ghost_ops as g, resident::*, *};
pub fn fixed_hits(b: &dyn ComputeBackend) {
    let data = [
        [-0.5, -0.5, 0.1, 0., 0.],
        [0.5, 0.5, 0.2, 0., 0.],
        [2., 0., 0.3, 0., 0.],
        [-0.5, -0.5, 0.4, 1., 0.],
        [0.5, 0.5, 0.5, 1., 0.],
        [0., 0., 0.6, 0., 1.],
        [0.5, -0.5, 0.7, 1., 0.],
    ];
    let input = b
        .fill_buffer(
            BufferDesc::new(ElementType::DoubleFloat, [7, 5]),
            &mut |mut f| {
                for (v, x) in f
                    .as_mut_slice::<Df32>()?
                    .iter_mut()
                    .zip(data.iter().flatten())
                {
                    *v = Df32::from_f64(*x);
                }
                Ok(())
            },
        )
        .unwrap();
    let mut p = ResidentPlan::new(Snapshot::new(samples::singlet()).unwrap());
    let input = p.import(input.view());
    let terms = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [3, 7]));
    p.operations
        .push(ResidentOp::Ghost(g::Request::MomentTerms {
            hits: p.view(input).unwrap(),
            output: terms,
        }));
    let mo = b
        .fill_buffer(BufferDesc::new(ElementType::U32, [2]), &mut |mut f| {
            f.as_mut_slice::<u32>()?.copy_from_slice(&[0, 7]);
            Ok(())
        })
        .unwrap();
    let mo = p.import(mo.view());
    let moments = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 10]));
    let ms = p.scratch(BufferDesc::new(ElementType::U32, [1]));
    p.operations
        .push(ResidentOp::Collectives(c::Request::SegmentReduce(
            c::SegmentReduce {
                values: p.view(terms).unwrap(),
                offsets: p.view(mo).unwrap(),
                mask: None,
                mask_value: 0,
                weights: None,
                max_segment_len: 7,
                reference: None,
                mode: ReduceMode::DoubleFloat,
                kind: c::SegmentKind::Moments2,
                output: moments,
                status: ms,
            },
        )));
    let k = p.scratch(BufferDesc::new(ElementType::U32, [7]));
    let ids = p.scratch(BufferDesc::new(ElementType::U32, [7]));
    let w = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [7]));
    p.operations.push(ResidentOp::Ghost(g::Request::PixelKeys {
        hits: p.view(input).unwrap(),
        bins: [2, 2],
        maps: 2,
        half: [Df32::from_f32(1.); 2],
        keys: k,
        ids,
        weights: w,
    }));
    let sk = p.scratch(BufferDesc::new(ElementType::U32, [7]));
    let si = p.scratch(BufferDesc::new(ElementType::U32, [7]));
    let sw = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [7]));
    p.operations
        .push(ResidentOp::Collectives(c::Request::StableSort(
            c::StableSort {
                keys: p.view(k).unwrap(),
                ids: p.view(ids).unwrap(),
                values: Some(p.view(w).unwrap()),
                output_keys: sk,
                output_ids: si,
                output_values: Some(sw),
            },
        )));
    let flags = p.scratch(BufferDesc::new(ElementType::U32, [7]));
    p.operations
        .push(ResidentOp::Ghost(g::Request::SegmentFlags {
            keys: p.view(sk).unwrap(),
            bins: 8,
            output: flags,
        }));
    let prefix = p.scratch(BufferDesc::new(ElementType::U32, [7]));
    p.operations
        .push(ResidentOp::Collectives(c::Request::Scan(c::Scan {
            values: p.view(flags).unwrap(),
            inclusive: true,
            output: prefix,
        })));
    let os = p.scratch(BufferDesc::new(ElementType::U32, [8]));
    let px = p.scratch(BufferDesc::new(ElementType::U32, [7]));
    let count = p.scratch(BufferDesc::new(ElementType::U32, [2]));
    p.operations
        .push(ResidentOp::Ghost(g::Request::PixelSegments {
            keys: p.view(sk).unwrap(),
            prefix: p.view(prefix).unwrap(),
            bins: 8,
            offsets: os,
            pixels: px,
            count,
        }));
    let mut offsets = p.view(os).unwrap();
    offsets.desc.shape = vec![6];
    let mut pixels = p.view(px).unwrap();
    pixels.desc.shape = vec![5];
    let pixels_id = pixels.clone();
    let mut values = p.view(sw).unwrap();
    values.desc.shape = vec![1, 7];
    let sums = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 5]));
    let status = p.scratch(BufferDesc::new(ElementType::U32, [5]));
    p.operations
        .push(ResidentOp::Collectives(c::Request::SegmentReduce(
            c::SegmentReduce {
                values,
                offsets,
                mask: None,
                mask_value: 0,
                weights: None,
                max_segment_len: 2,
                reference: None,
                mode: ReduceMode::DoubleFloat,
                kind: c::SegmentKind::Sum,
                output: sums,
                status,
            },
        )));
    let out = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [8]));
    p.operations.push(ResidentOp::Ghost(g::Request::MapScatter {
        values: p.view(sums).unwrap(),
        pixels: pixels_id,
        output: out,
    }));
    p.outputs = vec![out, count, os, sk, moments];
    let r = b
        .submit_plan(p, &JobContext::new())
        .unwrap()
        .wait()
        .unwrap();
    let m = b.read_view(r.outputs[4].view()).unwrap();
    let m = m.as_slice::<Df32>().unwrap();
    assert_eq!(m[0].to_f64(), 7.);
    assert!((m[1].to_f64() - 2.8).abs() < 1e-13);
    assert!((m[2].to_f64() - 0.375).abs() < 1e-13);
    assert!((m[3].to_f64() + 0.25 / 2.8).abs() < 1e-13);
    let central = 2.15 - 2.8 * (0.375f64.powi(2) + (0.25f64 / 2.8).powi(2));
    assert!((m[5].to_f64() + m[7].to_f64() - central).abs() < 1e-12);
    let map = b.read_view(r.outputs[0].view()).unwrap();
    for (a, e) in map
        .as_slice::<Df32>()
        .unwrap()
        .iter()
        .zip([0.7, 0., 0., 0.2, 0.4, 0.7, 0., 0.5])
    {
        assert!((a.to_f64() - e).abs() < 1e-13);
    }
    let counts = b.read_view(r.outputs[1].view()).unwrap();
    assert_eq!(counts.as_slice::<u32>().unwrap(), &[5, 2]);
    let offsets = b.read_view(r.outputs[2].view()).unwrap();
    assert_eq!(
        &offsets.as_slice::<u32>().unwrap()[..6],
        &[0, 2, 3, 4, 5, 6]
    );
    let keys = b.read_view(r.outputs[3].view()).unwrap();
    assert_eq!(keys.as_slice::<u32>().unwrap(), &[0, 0, 3, 4, 5, 7, 8]);
}
#[test]
fn fixed_hit_conservation() {
    fixed_hits(&CpuF32Backend::new());
}
