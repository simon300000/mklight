//! Compiles `tests/c/smoke.c` with the system C compiler, links it against
//! the `libmklight_ffi.a` that cargo built for this test run, and runs it.
//! Skipped (with a message) when no C compiler is available.

use std::path::{Path, PathBuf};
use std::process::Command;

/// The static library from this build. `cargo test` writes it to
/// `target/<profile>/deps/libmklight_ffi-<hash>.a` (only `cargo build`
/// copies it to `target/<profile>/`), so take the most recent candidate.
fn staticlib() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    let deps = exe.parent().unwrap();
    let profile = deps.parent().unwrap();
    let is_lib = |n: &str| n.starts_with("libmklight_ffi") && n.ends_with(".a");
    let mut cands: Vec<PathBuf> = std::fs::read_dir(deps)
        .unwrap()
        .chain(std::fs::read_dir(profile).unwrap())
        .filter_map(|e| e.ok())
        .filter(|e| is_lib(&e.file_name().to_string_lossy()))
        .map(|e| e.path())
        .collect();
    cands.sort_by_key(|p| std::fs::metadata(p).and_then(|m| m.modified()).ok());
    cands
        .pop()
        .expect("libmklight_ffi.a not found; run `cargo build -p mklight-ffi` first")
}

/// Native libraries the Rust standard library needs when linked statically
/// (`rustc --print native-static-libs`).
fn native_libs() -> &'static [&'static str] {
    if cfg!(target_os = "macos") {
        // The engine links the Metal backend (docs/FFI.md, "Building and
        // linking").
        &[
            "-lm",
            "-liconv",
            "-lobjc",
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-framework",
            "CoreFoundation",
        ]
    } else {
        &["-lpthread", "-ldl", "-lm", "-lc"]
    }
}

fn compiler() -> Option<String> {
    let cc = std::env::var("CC").unwrap_or_else(|_| "clang".into());
    Command::new(&cc)
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|_| cc)
}

#[test]
fn c_smoke_test() {
    let Some(cc) = compiler() else {
        eprintln!("c_smoke_test skipped: no C compiler (set CC)");
        return;
    };
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("c-smoke");
    let _ = std::fs::remove_dir_all(&out_dir);
    std::fs::create_dir_all(&out_dir).unwrap();
    let exe = out_dir.join("smoke");

    let status = Command::new(&cc)
        .args(["-std=c11", "-Wall", "-Wextra", "-Werror", "-pedantic", "-g"])
        .arg("-I")
        .arg(crate_dir.join("include"))
        .arg(crate_dir.join("tests/c/smoke.c"))
        .arg(staticlib())
        .args(native_libs())
        .arg("-o")
        .arg(&exe)
        .status()
        .expect("run C compiler");
    assert!(status.success(), "compiling smoke.c failed");

    let out = Command::new(&exe)
        .arg(&out_dir)
        .output()
        .expect("run smoke");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success(),
        "smoke failed ({:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
        out.status
    );
    assert!(stdout.contains("checks passed"), "{stdout}");
    // No stray temporary files from atomic saves.
    let leftovers: Vec<_> = std::fs::read_dir(&out_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|n| n.contains(".tmp-"))
        .collect();
    assert!(leftovers.is_empty(), "{leftovers:?}");
}

/// Swift imports the header through this module map; check it names the
/// header and the library.
#[test]
fn module_map_is_consistent() {
    let crate_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mm = std::fs::read_to_string(crate_dir.join("include/module.modulemap")).unwrap();
    assert!(mm.contains("module CMKLight"));
    assert!(mm.contains("header \"mklight.h\""));
    assert!(mm.contains("link \"mklight_ffi\""));
    // The static library contains the Metal backend: Swift autolinks these.
    for f in ["Metal", "Foundation", "CoreFoundation"] {
        assert!(mm.contains(&format!("link framework \"{f}\"")), "{f}");
    }
    for l in ["objc", "iconv"] {
        assert!(mm.contains(&format!("link \"{l}\"")), "{l}");
    }
}
