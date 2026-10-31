#[path = "p7_support.rs"]
mod support;
#[test]
fn coating_reference() {
    support::check_coatings(&mklight_gpu::CpuF32Backend::new());
}
#[test]
fn sample_transport_reference() {
    support::check_samples(&mklight_gpu::CpuF32Backend::new());
}
#[test]
fn coated_and_fallback_jobs() {
    support::fixtures(&mklight_gpu::CpuF32Backend::new());
}
#[test]
fn analytic_coating_answers() {
    support::known_answers(&mklight_gpu::CpuF32Backend::new());
}
#[test]
fn malus_and_vector_channels() {
    support::malus_and_channels(&mklight_gpu::CpuF32Backend::new());
}
