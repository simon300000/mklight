//! CUDA kernels: which modules exist, their block sizes, and the images
//! compiled ahead of time.
//!
//! Every unit of [`sources`] is compiled by NVRTC with [`OPTIONS`] into a
//! CUBIN for each supported architecture, plus PTX for later ones, by the
//! build script when NVRTC is available at build time, and embedded
//! ([`embedded`]). Without an embedded image for the device (a build without
//! NVRTC, or a GPU older than the oldest target) the unit is compiled at run
//! time by NVRTC, loaded dynamically like the driver, and cached on disk.
//!
//! Kernel interface: the WebGPU backend's (entry names, parameter blocks and
//! bindings; `kernels/mkl_cuda.h`). Block sizes are the WGSL entries'
//! `@workgroup_size`, read from the WGSL sources.

pub use crate::sources::*;

/// `(unit, target, deflated image)` of every unit compiled ahead of time;
/// `target` is `sm_XY` (a CUBIN) or `compute_XY` (NUL-terminated PTX). Empty when the
/// build had no NVRTC or `MKLIGHT_CUDA_ARCHS=none`.
pub static EMBEDDED: &[(&str, &str, &[u8])] = include!(concat!(env!("OUT_DIR"), "/embedded.rs"));

/// The embedded image of `unit` for a device of compute capability
/// `(major, minor)`: its own CUBIN, else the newest CUBIN of the same major
/// version not newer than it (binary compatible), else PTX of a target not
/// newer than it (compiled by the driver).
pub fn embedded(unit: &str, (major, minor): (i32, i32)) -> Option<(&'static str, &'static [u8])> {
    let cc = |t: &str, p: &str| {
        t.strip_prefix(p)
            .and_then(|v| v.parse::<i32>().ok())
            .map(|v| (v / 10, v % 10))
    };
    let mine = EMBEDDED.iter().filter(|(u, _, _)| *u == unit);
    let cubin = mine
        .clone()
        .filter_map(|(_, t, b)| cc(t, "sm_").map(|c| (c, *t, *b)))
        .filter(|(c, _, _)| c.0 == major && c.1 <= minor)
        .max_by_key(|(c, _, _)| *c);
    let ptx = || {
        mine.clone()
            .filter_map(|(_, t, b)| cc(t, "compute_").map(|c| (c, *t, *b)))
            .filter(|(c, _, _)| *c <= (major, minor))
            .max_by_key(|(c, _, _)| *c)
    };
    cubin.or_else(ptx).map(|(_, t, b)| (t, b))
}

/// The targets embedded at build time (`kernel_build`).
pub fn embedded_targets() -> Vec<&'static str> {
    let mut t: Vec<&str> = EMBEDDED.iter().map(|(_, t, _)| *t).collect();
    t.sort();
    t.dedup();
    t
}

/// The module a WebGPU operation family needs (`Gpu::ports`).
pub fn ported(family: &str) -> bool {
    let module = match family {
        "sum" => "reduce",
        other => other,
    };
    MODULES.iter().any(|(m, _)| *m == module)
}

/// Block sizes of the entries without a WGSL twin (`block_size`).
pub const CUDA_ENTRIES: &[(&str, &str, [u32; 3])] = &[
    ("fft_mtf", "mkl_fft_mtf_domain", [256, 1, 1]),
    ("fft_mtf", "mkl_fft_mtf_sensor", [256, 1, 1]),
    ("fft_mtf", "mkl_fft_mtf_normalize", [256, 1, 1]),
    ("fft_mtf", "mkl_fft_mtf_sample", [256, 1, 1]),
];

/// The block size of a WGSL entry point: its `@workgroup_size(x[, y[, z]])`,
/// whose terms are literals or `const`s of the module (the generated layout
/// constants included); for CUDA-only entries, [`CUDA_ENTRIES`].
pub fn block_size(module: &str, entry: &str) -> Option<[u32; 3]> {
    if let Some(&(_, _, b)) = CUDA_ENTRIES
        .iter()
        .find(|(m, e, _)| *m == module && *e == entry)
    {
        return Some(b);
    }
    let wgsl = mklight_webgpu::kernels::source(module)?;
    let at = wgsl.find(&format!("fn {entry}("))?;
    let before = &wgsl[..at];
    let ws = before.rfind("@workgroup_size(")?;
    let args = &before[ws + "@workgroup_size(".len()..];
    let args = &args[..args.find(')')?];
    let mut dims = [1u32; 3];
    for (k, term) in args.split(',').enumerate().take(3) {
        dims[k] = wgsl_value(&wgsl, term.trim())?;
    }
    Some(dims)
}

fn wgsl_value(wgsl: &str, term: &str) -> Option<u32> {
    let lit = |t: &str| {
        let t = t.trim().trim_end_matches(['u', 'i']);
        match t.strip_prefix("0x") {
            Some(h) => u32::from_str_radix(h, 16).ok(),
            None => t.parse::<u32>().ok(),
        }
    };
    if let Some(v) = lit(term) {
        return Some(v);
    }
    let decl = format!("const {term}");
    let at = wgsl.find(&decl)?;
    let rest = &wgsl[at + decl.len()..];
    let eq = rest.find('=')?;
    let end = rest.find(';')?;
    let value = rest[eq + 1..end].trim();
    lit(value).or_else(|| wgsl_value(wgsl, value))
}

#[cfg(test)]
mod tests {
    #[test]
    fn block_sizes_resolve() {
        assert_eq!(
            super::block_size("numerics", "mkl_numerics_probe"),
            Some([64, 1, 1])
        );
        assert_eq!(
            super::block_size("reduce", "mkl_reduce_pass1"),
            Some([256, 1, 1])
        );
        assert_eq!(
            super::block_size("geometry", "mkl_geometry_coordinates").map(|b| b[0] > 0),
            Some(true)
        );
    }
}
