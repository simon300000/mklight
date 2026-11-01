use mklight_core::{JobContext, Snapshot, samples};
use mklight_gpu::{collectives as c, ghost_ops as g, resident::*, *};
pub fn fixed_hits(b: &dyn ComputeBackend) {
    // The first hit lies just inside bin 0: its scaled coordinate rounds to
    // an integral hi with a negative lo.
    let data = [
        [-1e-12, -0.5, 0.1, 0., 0.],
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
    assert!((m[2].to_f64() - 1.1 / 2.8).abs() < 1e-13);
    assert!((m[3].to_f64() + 0.25 / 2.8).abs() < 1e-13);
    let central = 2.125 - 2.8 * ((1.1f64 / 2.8).powi(2) + (0.25f64 / 2.8).powi(2));
    assert!((m[5].to_f64() + m[7].to_f64() - central).abs() < 1e-12);
    let map = b.read_view(r.outputs[0].view()).unwrap();
    let map = map.as_slice::<Df32>().unwrap();
    assert_eq!(map.len(), 8);
    for (a, e) in map.iter().zip([0.7, 0., 0., 0.2, 0.4, 0.7, 0., 0.5]) {
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

pub fn first_order(b: &dyn ComputeBackend) -> Vec<u32> {
    use bytemuck::Zeroable;
    use mklight_gpu::{consts::*, radiometry::CoatingValue};
    let mut bits = Vec::new();
    for mirror in [false, true] {
        let mut p = ResidentPlan::new(Snapshot::new(samples::singlet()).unwrap());
        let steps = b
            .fill_buffer(g::record_desc::<g::FirstOrderStep>(2), &mut |mut f| {
                let rows =
                    bytemuck::cast_slice_mut::<_, g::FirstOrderStep>(f.as_mut_slice::<u32>()?);
                rows[0] = g::FirstOrderStep {
                    power: Df32::from_f64(0.1),
                    index: Df32::from_f32(1.),
                    distance: Df32::from_f32(10.),
                    efficiency: Df32::from_f32(1.),
                    absorption: Df32::ZERO,
                    coating: 0,
                    channel: MKL_GHOST_AXIAL_TRANSMIT,
                };
                rows[1] = g::FirstOrderStep {
                    power: Df32::from_f64(-0.2),
                    index: Df32::from_f32(-1.),
                    distance: Df32::ZERO,
                    efficiency: Df32::from_f64(0.8),
                    absorption: Df32::from_f64(0.1),
                    coating: mirror as u32,
                    channel: if mirror {
                        MKL_GHOST_AXIAL_MIRROR
                    } else {
                        MKL_GHOST_AXIAL_REFLECT
                    },
                };
                Ok(())
            })
            .unwrap();
        let offsets = b
            .fill_buffer(BufferDesc::new(ElementType::U32, [2]), &mut |mut f| {
                f.as_mut_slice::<u32>()?.copy_from_slice(&[0, 2]);
                Ok(())
            })
            .unwrap();
        let initial = b
            .fill_buffer(
                BufferDesc::new(ElementType::DoubleFloat, [1, 4]),
                &mut |mut f| {
                    for (v, x) in f.as_mut_slice::<Df32>()?.iter_mut().zip([1., 0., 2., 0.1]) {
                        *v = Df32::from_f64(x);
                    }
                    Ok(())
                },
            )
            .unwrap();
        let coats = b
            .fill_buffer(g::record_desc::<CoatingValue>(2), &mut |mut f| {
                for (i, v) in bytemuck::cast_slice_mut::<_, CoatingValue>(f.as_mut_slice::<u32>()?)
                    .iter_mut()
                    .enumerate()
                {
                    *v = CoatingValue::zeroed();
                    v.powers = [
                        0.04,
                        0.04,
                        if i == 0 { 0.96 } else { 0. },
                        if i == 0 { 0.96 } else { 0. },
                        if i == 0 { 0. } else { 0.96 },
                        if i == 0 { 0. } else { 0.96 },
                    ]
                    .map(Df32::from_f64);
                }
                Ok(())
            })
            .unwrap();
        let steps = p.import(steps.view());
        let offsets = p.import(offsets.view());
        let initial = p.import(initial.view());
        let coats = p.import(coats.view());
        let output = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [2, 4]));
        let throughput = p.scratch(BufferDesc::new(ElementType::DoubleFloat, [1, 2]));
        let status = p.scratch(BufferDesc::new(ElementType::U32, [1]));
        p.operations.push(ResidentOp::Ghost(g::Request::FirstOrder {
            steps: p.view(steps).unwrap(),
            offsets: p.view(offsets).unwrap(),
            initial: p.view(initial).unwrap(),
            coatings: p.view(coats).unwrap(),
            sensor_reflectance: Df32::ZERO,
            output,
            throughput,
            status,
        }));
        p.outputs = vec![output, throughput, status];
        let r = b
            .submit_plan(p, &JobContext::new())
            .unwrap()
            .wait()
            .unwrap();
        let data = b.read_view(r.outputs[0].view()).unwrap();
        assert_eq!(data.as_slice::<Df32>().unwrap().len(), 8);
        for (v, e) in data
            .as_slice::<Df32>()
            .unwrap()
            .iter()
            .zip([1., -0.1, 2., -0.1, 0., -0.1, 1., 0.1])
        {
            assert!((v.to_f64() - e).abs() < 1e-12);
        }
        let rt = b.read_view(r.outputs[1].view()).unwrap();
        let v = rt.as_slice::<Df32>().unwrap();
        assert!((v[0].to_f64() - if mirror { 1. } else { 0.04 }).abs() < 1e-13);
        let expect = 0.96 * 0.8 * (-0.1f64).exp() * if mirror { 0.04 } else { 1. };
        assert!((v[1].to_f64() - expect).abs() < 1e-6 * expect);
        let flags = b.read_view(r.outputs[2].view()).unwrap();
        assert_eq!(flags.as_slice::<u32>().unwrap(), &[0]);
        bits.extend_from_slice(data.words());
        bits.extend_from_slice(rt.words());
    }
    bits
}
#[test]
fn first_order_known_answers() {
    first_order(&CpuF32Backend::new());
}
