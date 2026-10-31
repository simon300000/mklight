mod p3_cases;
use mklight_gpu::*;
#[test]
fn scan_known_answers() {
    p3_cases::scan_answers(&CpuF32Backend::new());
}
#[test]
fn order_and_compact() {
    let b = CpuF32Backend::new();
    p3_cases::sort_answers(&b);
    p3_cases::compact_answers(&b);
}
#[test]
fn segments_known_answers() {
    p3_cases::segment_answers(&CpuF32Backend::new());
}
#[test]
fn qr_reference() {
    p3_cases::qr_answers(&CpuF32Backend::new());
}

#[test]
fn energy_conservation() {
    p3_cases::energy_chain(&CpuF32Backend::new());
}
#[test]
fn conditioned_qr() {
    p3_cases::conditioned_qr(&CpuF32Backend::new());
}

#[test]
fn mapped_tree_preserves_legacy_geometry() {
    use mklight_core::JobContext;
    use mklight_gpu::{
        fenv::{Ftz, Ieee},
        pending, reduce,
    };
    let (_pending, _completer, ctl) = pending::pending::<()>(&JobContext::new());
    for n in [0, 1, 103, 4097, 100_003] {
        let values: Vec<_> = (0..n).map(|i| [1e8f32, 1., -1e8, 1e-39][i % 4]).collect();
        for mode in [
            ReduceMode::Plain,
            ReduceMode::Kahan,
            ReduceMode::DoubleFloat,
        ] {
            let a = reduce::tree_in::<Ieee>(
                reduce::Accumulator::Sum(mode),
                Df32::ZERO,
                n,
                |i| Df32::from_f32(values[i]),
                &ctl,
            )
            .unwrap();
            let b = reduce::reduce_channel_in::<Ieee>(mode, &values);
            assert_eq!(
                [a.hi.to_bits(), a.lo.to_bits()],
                [b.hi.to_bits(), b.lo.to_bits()]
            );
            let a = reduce::tree_in::<Ftz>(
                reduce::Accumulator::Sum(mode),
                Df32::ZERO,
                n,
                |i| Df32::from_f32(values[i]),
                &ctl,
            )
            .unwrap();
            let b = reduce::reduce_channel_in::<Ftz>(mode, &values);
            assert_eq!(
                [a.hi.to_bits(), a.lo.to_bits()],
                [b.hi.to_bits(), b.lo.to_bits()]
            );
        }
    }
}

#[test]
fn nonfinite_qr() {
    p3_cases::nonfinite_qr(&CpuF32Backend::new());
}
