//! Invalidate full-chain cost evidence when host code or build settings change.
use std::{env, fs, path::Path, process::Command};

fn sources(path: &Path, root: &Path, hash: &mut blake3::Hasher) {
    let mut entries: Vec<_> = fs::read_dir(path)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            sources(&path, root, hash);
        } else if path.extension().is_some_and(|e| e == "rs") {
            file(&path, root, hash);
        }
    }
}
fn file(path: &Path, root: &Path, hash: &mut blake3::Hasher) {
    println!("cargo:rerun-if-changed={}", path.display());
    hash.update(
        path.strip_prefix(root)
            .unwrap()
            .to_str()
            .unwrap()
            .as_bytes(),
    );
    hash.update(&fs::read(path).unwrap());
}
fn main() {
    let manifest = env::var_os("CARGO_MANIFEST_DIR").unwrap();
    let root = Path::new(&manifest).parent().unwrap().parent().unwrap();
    let mut hash = blake3::Hasher::new();
    for name in ["Cargo.toml", "Cargo.lock", "crates/mklight-engine/build.rs"] {
        file(&root.join(name), root, &mut hash);
    }
    let mut crates: Vec<_> = fs::read_dir(root.join("crates"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    crates.sort();
    for path in crates {
        if path.join("src").is_dir() {
            file(&path.join("Cargo.toml"), root, &mut hash);
            println!("cargo:rerun-if-changed={}", path.join("src").display());
            sources(&path.join("src"), root, &mut hash);
        }
    }
    for name in [
        "TARGET",
        "PROFILE",
        "OPT_LEVEL",
        "DEBUG",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={name}");
        hash.update(env::var(name).unwrap_or_default().as_bytes());
    }
    hash.update(
        &Command::new(env::var_os("RUSTC").unwrap())
            .arg("--version")
            .output()
            .unwrap()
            .stdout,
    );
    println!(
        "cargo:rustc-env=MKLIGHT_COST_BUILD={}",
        hash.finalize().to_hex()
    );
}
