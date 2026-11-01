//! The owned numeric record must agree with the C header, independently of
//! Rust/Metal's numerical tests that consume it.
use mklight_gpu::{Df32, geometry::SurfaceSample};
use std::{
    mem::{offset_of, size_of},
    process::Command,
};
#[test]
fn geometry_sample_matches_c_layout() {
    let dir = std::env::temp_dir().join(format!("mklight-p5-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("probe.c");
    let binary = dir.join("probe");
    std::fs::write(&source,"#include <stddef.h>\n#include <stdio.h>\n#include \"mklight_gpu_types.h\"\nint main(void) {printf(\"%zu %zu %zu %zu\\n\",sizeof(struct MklGeometrySample),offsetof(struct MklGeometrySample,system),offsetof(struct MklGeometrySample,surface),offsetof(struct MklGeometrySample,xy));return 0;}\n").unwrap();
    let output = Command::new("cc")
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror"])
        .arg("-I")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/include"))
        .arg(&source)
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let output = Command::new(&binary).output().unwrap();
    assert!(output.status.success());
    let values: Vec<usize> = String::from_utf8(output.stdout)
        .unwrap()
        .split_whitespace()
        .map(|v| v.parse().unwrap())
        .collect();
    assert_eq!(
        values,
        [
            size_of::<SurfaceSample>(),
            offset_of!(SurfaceSample, system),
            offset_of!(SurfaceSample, surface),
            offset_of!(SurfaceSample, xy)
        ]
    );
    assert_eq!(
        size_of::<SurfaceSample>(),
        4 * size_of::<u32>() + 2 * size_of::<Df32>()
    );
    std::fs::remove_dir_all(dir).unwrap();
}
