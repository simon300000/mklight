//! The integration tests reuse the WebGPU backend's test files (the CUDA
//! backend runs the same host operations and kernel interface): each file of
//! `../mklight-webgpu/tests` but `main.rs` and `proxy.rs` is copied into
//! `OUT_DIR/webgpu_tests/` with the placement and backend names changed from
//! `webgpu` to `cuda` and without its inner doc comment, for `tests/main.rs`
//! to include. The files their `#[path = "..."]` modules name (the shared
//! suites of `mklight-gpu/tests`) are copied alongside and the attribute
//! points at the copy, since the copies live in `OUT_DIR`.

use std::{
    env, fs,
    path::{Path, PathBuf},
};

/// Tests of the shared suites that do not apply here, ignored in the copies.
/// `portable_parameter_layout` compiles the C layout header with the host's
/// `cc` (absent on Windows); it checks the host ABI, not the device, and
/// runs with `mklight-gpu`'s own tests. The CUDA units check the same layout
/// with the header's static assertions under NVRTC.
const IGNORED: &[&str] = &["portable_parameter_layout"];

fn copy(src: &Path, out: &Path) -> String {
    println!("cargo:rerun-if-changed={}", src.display());
    let dir = src.parent().unwrap();
    let mut body = String::new();
    for l in fs::read_to_string(src).unwrap().lines() {
        if l.trim_start().starts_with("//!") {
            continue;
        }
        if let Some(rel) = l.trim().strip_prefix("#[path = \"").and_then(|r| r.strip_suffix("\"]")) {
            let target = copy(&dir.join(rel), out);
            body += &format!("#[path = \"{}\"]\n", target.replace('\\', "/"));
            continue;
        }
        if IGNORED.iter().any(|t| l.trim() == format!("fn {t}() {{")) {
            body += "#[ignore = \"not applicable to the CUDA test program (build.rs)\"]\n";
        }
        body += &l
            .replace("\"webgpu:", "\"cuda:")
            .replace("Backend::Gpu(\"webgpu\".into())", "Backend::Gpu(\"cuda\".into())")
            .replace("kernel_build.contains(\"WGSL\")", "kernel_build.contains(\"CUDA\")");
        body += "\n";
    }
    let dst = out.join(src.file_name().unwrap());
    fs::write(&dst, body).unwrap();
    dst.to_string_lossy().into_owned()
}

fn main() {
    let dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../mklight-webgpu/tests");
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("webgpu_tests");
    fs::create_dir_all(&out).unwrap();
    println!("cargo:rerun-if-changed={}", dir.display());
    for e in fs::read_dir(&dir).unwrap() {
        let p = e.unwrap().path();
        let name = p.file_name().unwrap().to_string_lossy().into_owned();
        if !name.ends_with(".rs") || name == "main.rs" || name == "proxy.rs" {
            continue;
        }
        copy(&p, &out);
    }
}
