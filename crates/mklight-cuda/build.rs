//! Two jobs. First, the kernels are compiled ahead of time and embedded
//! (`embed_kernels`, below). Second, the integration tests reuse the WebGPU backend's test files (the CUDA
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
        if let Some(rel) = l
            .trim()
            .strip_prefix("#[path = \"")
            .and_then(|r| r.strip_suffix("\"]"))
        {
            let target = copy(&dir.join(rel), out);
            body += &format!("#[path = \"{}\"]\n", target.replace('\\', "/"));
            continue;
        }
        if IGNORED.iter().any(|t| l.trim() == format!("fn {t}() {{")) {
            body += "#[ignore = \"not applicable to the CUDA test program (build.rs)\"]\n";
        }
        body += &l
            .replace("\"webgpu:", "\"cuda:")
            .replace(
                "Backend::Gpu(\"webgpu\".into())",
                "Backend::Gpu(\"cuda\".into())",
            )
            .replace(
                "kernel_build.contains(\"WGSL\")",
                "kernel_build.contains(\"CUDA\")",
            );
        body += "\n";
    }
    let dst = out.join(src.file_name().unwrap());
    fs::write(&dst, body).unwrap();
    dst.to_string_lossy().into_owned()
}

fn main() {
    embed_kernels(&PathBuf::from(env::var("OUT_DIR").unwrap()));
    let dir =
        PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap()).join("../mklight-webgpu/tests");
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

// ---------------------------------------------------------------------------
// Kernels compiled ahead of time.
// ---------------------------------------------------------------------------

#[allow(dead_code)]
#[path = "src/sources.rs"]
mod sources;

#[cfg(any(windows, target_os = "linux"))]
#[allow(dead_code)]
#[path = "src/nvrtc.rs"]
mod nvrtc;

/// Targets compiled by default: CUBINs for Turing (7.5), Ampere (8.0, which
/// the 8.6 parts run too), Ada (8.9) and Hopper (9.0), and PTX for later
/// GPUs (compiled by their driver on first use, about a minute).
/// `MKLIGHT_CUDA_ARCHS` overrides it, e.g. `89` while developing on an Ada
/// GPU, or `none`.
const DEFAULT_TARGETS: &str = "75,80,89,90,ptx90";

/// Added to the default targets when the build machine's NVRTC knows them
/// (12.8 or later): CUBINs for Blackwell (10.0, 12.0), so those GPUs start
/// without compiling PTX.
#[cfg(any(windows, target_os = "linux"))]
const BLACKWELL_TARGETS: &str = "100,120";

/// `MKLIGHT_CUDA_REQUIRE_KERNELS=1` (release builds of the apps): a build
/// that would have no built-in kernels fails instead of warning, since its
/// users' CUDA would then need NVRTC to start.
fn require_kernels() -> bool {
    println!("cargo:rerun-if-env-changed=MKLIGHT_CUDA_REQUIRE_KERNELS");
    env::var("MKLIGHT_CUDA_REQUIRE_KERNELS").is_ok_and(|v| v == "1")
}

/// Write `OUT_DIR/embedded.rs`: the `EMBEDDED` table of `(unit, target,
/// deflated image)` for every unit and target, compiled by NVRTC when the build
/// machine has it (in parallel), else empty: the crate then compiles at run
/// time. The images go to `OUT_DIR/cuda/` and are `include_bytes!`ed.
fn embed_kernels(out: &Path) {
    println!("cargo:rerun-if-env-changed=MKLIGHT_CUDA_ARCHS");
    for dir in [
        "kernels",
        "src/sources.rs",
        "../mklight-metal/shaders",
        "../mklight-gpu/include",
    ] {
        println!("cargo:rerun-if-changed={dir}");
    }
    let table = out.join("embedded.rs");
    let os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let cuda_target = os == "windows" || os == "linux";
    let empty = |why: &str| {
        if cuda_target && require_kernels() {
            panic!("mklight-cuda: no built-in kernels ({why}) and MKLIGHT_CUDA_REQUIRE_KERNELS=1");
        }
        if cuda_target {
            println!(
                "cargo:warning=mklight-cuda: no built-in kernels ({why}); they compile at run time with NVRTC"
            );
        }
        fs::write(&table, "&[]").unwrap();
    };
    let explicit = env::var("MKLIGHT_CUDA_ARCHS").ok();
    let targets: Vec<String> = explicit
        .clone()
        .unwrap_or_else(|| DEFAULT_TARGETS.into())
        .split(',')
        .map(str::trim)
        .filter(|t| !t.is_empty() && *t != "none")
        .map(|t| match t.strip_prefix("ptx") {
            Some(v) => format!("compute_{v}"),
            None => format!("sm_{t}"),
        })
        .collect();
    if !cuda_target {
        return fs::write(&table, "&[]").unwrap();
    }
    if targets.is_empty() {
        return empty("MKLIGHT_CUDA_ARCHS=none");
    }
    #[cfg(not(any(windows, target_os = "linux")))]
    {
        empty("NVRTC runs on Windows and Linux build machines");
    }
    #[cfg(any(windows, target_os = "linux"))]
    {
        let Some(version) = nvrtc::version() else {
            return empty("NVRTC not found on the build machine");
        };
        let mut targets = targets;
        if explicit.is_none() && version >= (12, 8) {
            targets.extend(BLACKWELL_TARGETS.split(',').map(|t| format!("sm_{t}")));
        }
        let dir = out.join("cuda");
        fs::create_dir_all(&dir).unwrap();
        // The units that inline the trace take the longest: first.
        // Each image is named after a digest of what it is compiled from, so
        // a rerun of this script (any watched file changed) compiles only
        // the images whose inputs changed.
        let mut common = format!("{version:?}");
        for (n, h) in sources::HEADERS {
            common += n;
            common += h;
        }
        common += &sources::OPTIONS.join(" ");
        let mut jobs: Vec<(String, String, String, String)> = Vec::new();
        for (name, source) in sources::units() {
            for t in &targets {
                let mut h = blake3::Hasher::new();
                for part in [&common, &name, &source, t] {
                    h.update(part.as_bytes());
                }
                let file = format!("{name}.{t}.{}", &h.finalize().to_hex()[..16]);
                jobs.push((name.clone(), source.clone(), t.clone(), file));
            }
        }
        jobs.sort_by_key(|(_, s, _, _)| !s.contains("trace_bindings.h"));
        let next = std::sync::atomic::AtomicUsize::new(0);
        let failed = std::sync::Mutex::new(Vec::new());
        let threads = std::thread::available_parallelism().map_or(4, |n| n.get());
        std::thread::scope(|scope| {
            for _ in 0..threads.min(jobs.len()) {
                scope.spawn(|| {
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        let Some((name, source, target, file)) = jobs.get(i) else {
                            return;
                        };
                        if dir.join(file).is_file() {
                            continue;
                        }
                        match nvrtc::compile(name, source, target) {
                            // Deflated: the images shrink 4-6x (52 to 11 MB).
                            Ok((image, _)) => {
                                let tmp = dir.join(format!("{file}.tmp"));
                                fs::write(&tmp, miniz_oxide::deflate::compress_to_vec(&image, 8))
                                    .unwrap();
                                fs::rename(&tmp, dir.join(file)).unwrap();
                            }
                            Err(e) => failed.lock().unwrap().push(e),
                        }
                    }
                });
            }
        });
        let failed = failed.into_inner().unwrap();
        if !failed.is_empty() {
            panic!(
                "mklight-cuda: kernels failed to compile:\n{}",
                failed.join("\n")
            );
        }
        // Images of earlier sources go.
        for e in fs::read_dir(&dir).unwrap().flatten() {
            let n = e.file_name().to_string_lossy().into_owned();
            if !jobs.iter().any(|(_, _, _, f)| *f == n) {
                let _ = fs::remove_file(e.path());
            }
        }
        let mut rs = String::from("&[\n");
        for (name, _, target, file) in &jobs {
            let file = dir.join(file);
            rs += &format!(
                "    ({name:?}, {target:?}, include_bytes!({:?})),\n",
                file.to_string_lossy().replace('\\', "/")
            );
        }
        rs += "]";
        fs::write(&table, rs).unwrap();
    }
}
