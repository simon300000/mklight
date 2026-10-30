//! Layout contract: the Rust mirrors in `layout.rs` must match the C header
//! byte for byte. A C compiler prints `sizeof`/`offsetof` for every field
//! and the test compares them with `std::mem::offset_of!`. Skipped (with a
//! message) when no C compiler is available.

use std::path::PathBuf;
use std::process::Command;

use mklight_gpu::layout::{header_offsets, surface_offsets};
use mklight_gpu::{INCLUDE_DIR, PackedHeader, PackedSurface};

fn probe_source() -> String {
    let mut s = String::from(
        "#include <stdio.h>\n#include <stddef.h>\n#include \"mklight_gpu_types.h\"\nint main(void){\n",
    );
    s += "printf(\"header.size %zu\\n\", sizeof(struct MklPackedHeader));\n";
    s += "printf(\"surface.size %zu\\n\", sizeof(struct MklPackedSurface));\n";
    s += "printf(\"header.align %zu\\n\", _Alignof(struct MklPackedHeader));\n";
    s += "printf(\"surface.align %zu\\n\", _Alignof(struct MklPackedSurface));\n";
    for (f, _) in header_offsets() {
        s += &format!("printf(\"header.{f} %zu\\n\", offsetof(struct MklPackedHeader, {f}));\n");
    }
    for (f, _) in surface_offsets() {
        s += &format!("printf(\"surface.{f} %zu\\n\", offsetof(struct MklPackedSurface, {f}));\n");
    }
    s += "printf(\"layout.version %u\\n\", MKL_LAYOUT_VERSION);\n";
    s += "return 0;}\n";
    s
}

fn compiler() -> Option<String> {
    [
        std::env::var("CC").ok(),
        Some("cc".into()),
        Some("clang".into()),
    ]
    .into_iter()
    .flatten()
    .find(|cc| {
        Command::new(cc)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    })
}

#[test]
fn rust_layout_matches_c_header() {
    let Some(cc) = compiler() else {
        eprintln!("SKIP: no C compiler found");
        return;
    };
    let dir: PathBuf = std::env::temp_dir().join(format!("mklight-layout-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("probe.c");
    let exe = dir.join("probe");
    std::fs::write(&src, probe_source()).unwrap();
    let out = Command::new(&cc)
        .args([
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Wpadded",
            "-Werror",
            "-pedantic",
            "-I",
        ])
        .arg(INCLUDE_DIR)
        .arg(&src)
        .arg("-o")
        .arg(&exe)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "C compile failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let run = Command::new(&exe).output().unwrap();
    let text = String::from_utf8(run.stdout).unwrap();
    let get = |k: &str| -> usize {
        text.lines()
            .find_map(|l| l.strip_prefix(k).and_then(|r| r.strip_prefix(' ')))
            .unwrap_or_else(|| panic!("missing {k}"))
            .trim()
            .parse()
            .unwrap()
    };
    assert_eq!(get("layout.version"), 5);
    assert_eq!(get("surface.coefficient_offset"), 616);
    assert_eq!(get("surface.coefficient_count"), 620);
    assert_eq!(get("header.size"), size_of::<PackedHeader>());
    assert_eq!(get("surface.size"), size_of::<PackedSurface>());
    assert_eq!(get("header.align"), align_of::<PackedHeader>());
    assert_eq!(get("surface.align"), align_of::<PackedSurface>());
    for (f, off) in header_offsets() {
        assert_eq!(get(&format!("header.{f}")), off, "header.{f}");
    }
    for (f, off) in surface_offsets() {
        assert_eq!(get(&format!("surface.{f}")), off, "surface.{f}");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// The header must also be valid C++ (MSL and CUDA are C++ dialects).
#[test]
fn header_compiles_as_cpp() {
    let Some(cc) = compiler() else {
        eprintln!("SKIP: no C compiler found");
        return;
    };
    let dir = std::env::temp_dir().join(format!("mklight-layout-cpp-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("probe.cpp");
    std::fs::write(
        &src,
        "#include \"mklight_gpu_types.h\"\nint main() { return 0; }\n",
    )
    .unwrap();
    let out = Command::new(&cc)
        .args([
            "-x",
            "c++",
            "-std=c++14",
            "-Wall",
            "-Werror",
            "-fsyntax-only",
            "-I",
        ])
        .arg(INCLUDE_DIR)
        .arg(&src)
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "C++ compile failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
