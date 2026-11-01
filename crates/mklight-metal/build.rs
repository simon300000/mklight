//! Compiles `shaders/*.metal` into one `.metallib` with the offline Metal
//! toolchain (`xcrun -sdk macosx metal` → `.air` → `metallib`) and embeds it.
//! The preprocessed-include source is always embedded too, so the backend can
//! fall back to runtime compilation (`newLibraryWithSource`) when the metallib
//! does not load. A shader that fails offline fails the build; only a debug
//! build on a machine without the toolchain falls back to runtime compilation.
//!
//! Fast-math and FP contraction are disabled: they break the error-free
//! transforms of the double-float arithmetic (see shaders/mklight_common.h).

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::{env, fs};

/// Shipping shaders always use the complete strict numerics/warning contract.
const FLAGS: &[&str] = &[
    "-std=metal3.0",
    "-mmacosx-version-min=13.0",
    "-fno-fast-math",
    "-fmetal-math-mode=safe",
    "-fmetal-math-fp32-functions=precise",
    "-ffp-contract=off",
    "-Wall",
    "-Werror",
];

/// Inline local `#include "..."` directives (recursively), so the runtime
/// compiler, which has no include path, sees one self-contained source.
fn inline_includes(path: &Path, dirs: &[PathBuf], out: &mut String, depth: usize) {
    assert!(depth < 16, "include depth exceeded at {}", path.display());
    let text = fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    for line in text.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("#include \"") {
            let name = rest.split('"').next().unwrap();
            let found = dirs
                .iter()
                .map(|d| d.join(name))
                .find(|p| p.exists())
                .unwrap_or_else(|| panic!("include \"{name}\" not found"));
            println!("cargo:rerun-if-changed={}", found.display());
            inline_includes(&found, dirs, out, depth + 1);
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
}

/// How to invoke the offline compiler (`metal`) and linker (`metallib`).
struct Toolchain {
    metal: Vec<String>,
    metallib: Vec<String>,
}

impl Toolchain {
    fn cmd(argv: &[String]) -> Command {
        let mut c = Command::new(&argv[0]);
        c.args(&argv[1..]);
        c
    }
    fn works(&self) -> bool {
        Self::cmd(&self.metal)
            .arg("--version")
            .output()
            .is_ok_and(|o| o.status.success())
    }
}

/// Candidates in order: an explicit toolchain directory
/// (`MKLIGHT_METAL_TOOLCHAIN_BIN`), `xcrun`, and the downloadable Metal
/// toolchain component mounted by the system (which `xcrun` does not always
/// resolve).
fn find_toolchain() -> Option<Toolchain> {
    let direct = |dir: PathBuf| Toolchain {
        metal: vec![dir.join("metal").display().to_string()],
        metallib: vec![dir.join("metallib").display().to_string()],
    };
    let mut candidates = Vec::new();
    if let Some(d) = env::var_os("MKLIGHT_METAL_TOOLCHAIN_BIN") {
        candidates.push(direct(PathBuf::from(d)));
    }
    let xcrun = |tool: &str| ["xcrun", "-sdk", "macosx", tool].map(String::from).to_vec();
    candidates.push(Toolchain {
        metal: xcrun("metal"),
        metallib: xcrun("metallib"),
    });
    let mounts = Path::new("/var/run/com.apple.security.cryptexd/mnt");
    if let Ok(rd) = fs::read_dir(mounts) {
        let mut dirs: Vec<PathBuf> = rd
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().contains("MetalToolchain"))
            })
            .map(|p| p.join("Metal.xctoolchain/usr/bin"))
            .collect();
        dirs.sort();
        candidates.extend(dirs.into_iter().rev().map(direct));
    }
    candidates.into_iter().find(Toolchain::works)
}

fn try_offline(
    shaders: &[PathBuf],
    inc: &[PathBuf],
    out_dir: &Path,
) -> Result<(PathBuf, String), String> {
    let tc = find_toolchain().ok_or("no working Metal compiler found")?;
    let mut airs = Vec::new();
    for src in shaders {
        let air = out_dir.join(src.file_stem().unwrap()).with_extension("air");
        let mut cmd = Toolchain::cmd(&tc.metal);
        cmd.arg("-c").args(FLAGS);
        for d in inc {
            cmd.arg("-I").arg(d);
        }
        let o = cmd
            .arg(src)
            .arg("-o")
            .arg(&air)
            .output()
            .map_err(|e| format!("metal not runnable: {e}"))?;
        if !o.status.success() {
            return Err(format!(
                "{}: {}",
                src.display(),
                String::from_utf8_lossy(&o.stderr)
            ));
        }
        airs.push(air);
    }
    let lib = out_dir.join("mklight.metallib");
    let o = Toolchain::cmd(&tc.metallib)
        .args(&airs)
        .arg("-o")
        .arg(&lib)
        .output()
        .map_err(|e| format!("metallib not runnable: {e}"))?;
    if !o.status.success() {
        return Err(format!(
            "metallib failed: {}",
            String::from_utf8_lossy(&o.stderr)
        ));
    }
    Ok((lib, FLAGS.join(" ")))
}

fn main() {
    let manifest = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let shader_dir = manifest.join("shaders");
    let gpu_include = manifest.join("../mklight-gpu/include");
    let inc = vec![shader_dir.clone(), gpu_include.clone()];
    println!("cargo:rerun-if-changed={}", shader_dir.display());
    println!(
        "cargo:rerun-if-changed={}",
        gpu_include.join("mklight_gpu_types.h").display()
    );
    println!("cargo:rerun-if-env-changed=MKLIGHT_METAL_SKIP_OFFLINE");
    println!("cargo:rerun-if-env-changed=MKLIGHT_METAL_TOOLCHAIN_BIN");

    let mut generated = String::from("// @generated by build.rs\n");
    if env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("macos") {
        fs::write(out_dir.join("shaders.rs"), generated).unwrap();
        return;
    }

    let mut shaders: Vec<PathBuf> = fs::read_dir(&shader_dir)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "metal"))
        .collect();
    shaders.sort();
    for s in &shaders {
        println!("cargo:rerun-if-changed={}", s.display());
    }

    // Runtime-compilation fallback: all sources, includes inlined.
    let mut source = String::new();
    for s in &shaders {
        writeln!(
            source,
            "// ---- {} ----",
            s.file_name().unwrap().to_string_lossy()
        )
        .unwrap();
        inline_includes(s, &inc, &mut source, 0);
    }
    let src_path = out_dir.join("mklight_all.metal");
    fs::write(&src_path, &source).unwrap();
    writeln!(
        generated,
        "pub const SOURCE_HASH: &str = {:?};",
        blake3::hash(source.as_bytes()).to_hex().to_string()
    )
    .unwrap();
    writeln!(
        generated,
        "pub const SOURCE: &str = include_str!({:?});",
        src_path.display().to_string()
    )
    .unwrap();

    let offline = if env::var_os("MKLIGHT_METAL_SKIP_OFFLINE").is_some() {
        Err("skipped by MKLIGHT_METAL_SKIP_OFFLINE".to_string())
    } else {
        try_offline(&shaders, &inc, &out_dir)
    };
    match offline {
        Ok((lib, flags)) => {
            writeln!(
                generated,
                "pub const METALLIB: Option<&[u8]> = Some(include_bytes!({:?}));",
                lib.display().to_string()
            )
            .unwrap();
            writeln!(generated, "pub const OFFLINE_FLAGS: &str = {flags:?};").unwrap();
        }
        Err(e) => {
            // Shipped builds embed the offline metallib: with a working
            // compiler a failure is a shader error, and a release build
            // without one would silently ship runtime compilation.
            let skipped = env::var_os("MKLIGHT_METAL_SKIP_OFFLINE").is_some();
            if env::var("PROFILE").as_deref() == Ok("release")
                || (!skipped && find_toolchain().is_some())
            {
                panic!(
                    "offline Metal compile failed (install it with `xcodebuild -downloadComponent MetalToolchain`):\n{e}"
                );
            }
            println!(
                "cargo:warning=offline Metal compile unavailable, using runtime compilation: {}",
                e.lines().next().unwrap_or("")
            );
            generated.push_str("pub const METALLIB: Option<&[u8]> = None;\n");
            generated.push_str("pub const OFFLINE_FLAGS: &str = \"\";\n");
        }
    }
    fs::write(out_dir.join("shaders.rs"), generated).unwrap();
}
