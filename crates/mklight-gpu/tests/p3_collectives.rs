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
