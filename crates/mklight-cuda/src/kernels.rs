//! CUDA kernel sources and their compilation.
//!
//! Every module is one translation unit compiled at run time by NVRTC (loaded
//! dynamically, like the driver) into a CUBIN for the device's architecture,
//! and cached on disk by a digest of the sources, the options, the NVRTC
//! version and the architecture, so a second start only loads the CUBIN.
//!
//! The units include the shared layout headers of `mklight-gpu`
//! (`include/*.h`) and the Metal kernels' numerics and surface code
//! (`crates/mklight-metal/shaders`) as in-memory headers, the latter
//! through the MSL subset of `kernels/metal_stdlib.h`, so the arithmetic is
//! written once for Metal and CUDA.
//!
//! Kernel interface: the WebGPU backend's (entry names, parameter blocks and
//! bindings; `kernels/mkl_cuda.h`). Block sizes are the WGSL entries'
//! `@workgroup_size`, read from the WGSL sources.

/// Options of every compilation (GPU.md §3.2, §7): IEEE division and square
/// root, no contraction, subnormals preserved, no fast math.
pub const OPTIONS: &[&str] = &[
    "--std=c++17",
    "--fmad=false",
    "--prec-div=true",
    "--prec-sqrt=true",
    "--ftz=false",
    "--device-as-default-execution-space",
    "-DMKL_CUDA=1",
];

/// `(name, contents)` of every header the units may include.
pub const HEADERS: &[(&str, &str)] = &[
    ("metal_stdlib", include_str!("../kernels/metal_stdlib.h")),
    ("mkl_cuda.h", include_str!("../kernels/mkl_cuda.h")),
    // The layout header includes <stdint.h> outside Metal and GLSL.
    (
        "stdint.h",
        "#pragma once\ntypedef unsigned int uint32_t;\ntypedef int int32_t;\n\
         typedef unsigned long long uint64_t;\ntypedef long long int64_t;\n\
         typedef unsigned short uint16_t;\ntypedef unsigned char uint8_t;\n",
    ),
    (
        "mklight_gpu_types.h",
        include_str!("../../mklight-gpu/include/mklight_gpu_types.h"),
    ),
    (
        "mklight_aiming_types.h",
        include_str!("../../mklight-gpu/include/mklight_aiming_types.h"),
    ),
    (
        "mklight_collectives_types.h",
        include_str!("../../mklight-gpu/include/mklight_collectives_types.h"),
    ),
    (
        "mklight_geometry_types.h",
        include_str!("../../mklight-gpu/include/mklight_geometry_types.h"),
    ),
    (
        "mklight_transforms_types.h",
        include_str!("../../mklight-gpu/include/mklight_transforms_types.h"),
    ),
    (
        "mklight_wave_ops_types.h",
        include_str!("../../mklight-gpu/include/mklight_wave_ops_types.h"),
    ),
    (
        "mklight_radiometry_types.h",
        include_str!("../../mklight-gpu/include/mklight_radiometry_types.h"),
    ),
    (
        "mklight_ghost_ops_types.h",
        include_str!("../../mklight-gpu/include/mklight_ghost_ops_types.h"),
    ),
    (
        "mklight_optim_ops_types.h",
        include_str!("../../mklight-gpu/include/mklight_optim_ops_types.h"),
    ),
    (
        "mklight_stray_ops_types.h",
        include_str!("../../mklight-gpu/include/mklight_stray_ops_types.h"),
    ),
    (
        "mklight_common.h",
        include_str!("../../mklight-metal/shaders/mklight_common.h"),
    ),
    (
        "surface_math.h",
        include_str!("../../mklight-metal/shaders/surface_math.h"),
    ),
    (
        "trace_shared.h",
        include_str!("../../mklight-metal/shaders/trace_shared.h"),
    ),
    ("trace_bindings.h", include_str!("../kernels/trace_bindings.h")),
];

/// `(module, source)` of every CUDA module: the WebGPU backend's module
/// names, for the families ported so far ([`ported`]).
pub const MODULES: &[(&str, &str)] = &[
    ("numerics", include_str!("../kernels/numerics.cu")),
    ("reduce", include_str!("../kernels/reduce.cu")),
    ("geometry", include_str!("../kernels/geometry.cu")),
    ("collectives", include_str!("../kernels/collectives.cu")),
    ("wave", include_str!("../kernels/wave.cu")),
    ("transforms", include_str!("../kernels/transforms.cu")),
    // trace.cu twice: without and with sag overlays.
    (
        "trace",
        concat!("#define MKL_OVERLAYS false\n", include_str!("../kernels/trace.cu")),
    ),
    (
        "trace_overlays",
        concat!("#define MKL_OVERLAYS true\n", include_str!("../kernels/trace.cu")),
    ),
    // aiming.cu likewise (the WGSL modules append the aim solver to the
    // trace; here it calls trace_shared.h's core).
    (
        "aiming",
        concat!("#define MKL_OVERLAYS false\n", include_str!("../kernels/aiming.cu")),
    ),
    (
        "aiming_overlays",
        concat!("#define MKL_OVERLAYS true\n", include_str!("../kernels/aiming.cu")),
    ),
    // FftMtf (CUDA only): after wave.cu, whose helpers it uses.
    (
        "fft_mtf",
        concat!(include_str!("../kernels/wave.cu"), include_str!("../kernels/fft_mtf.cu")),
    ),
];

/// Block sizes of the entries without a WGSL twin (`block_size`).
const CUDA_ENTRIES: &[(&str, &str, [u32; 3])] = &[
    ("fft_mtf", "mkl_fft_mtf_domain", [256, 1, 1]),
    ("fft_mtf", "mkl_fft_mtf_sensor", [256, 1, 1]),
    ("fft_mtf", "mkl_fft_mtf_normalize", [256, 1, 1]),
    ("fft_mtf", "mkl_fft_mtf_sample", [256, 1, 1]),
];

/// The module a WebGPU operation family needs (`Gpu::ports`).
pub fn ported(family: &str) -> bool {
    let module = match family {
        "sum" => "reduce",
        other => other,
    };
    MODULES.iter().any(|(m, _)| *m == module)
}

/// The compilation unit of `entry` of `module`: `(name, source)`. A module
/// whose source contains `MKL_ONE_ENTRY` is compiled per entry point (the
/// unit defines `MKL_ONE_ENTRY` and `MKL_ENTRY_<entry>`, and the source
/// keeps only that entry): the trace is inlined into every entry, and one
/// unit with all of them took about a minute to compile. Other modules are
/// one unit.
pub fn unit(module: &str, entry: &str) -> Option<(String, String)> {
    let src = source(module)?;
    Some(if src.contains("MKL_ONE_ENTRY") {
        (
            format!("{module}.{entry}"),
            format!("#define MKL_ONE_ENTRY 1
#define MKL_ENTRY_{entry} 1
{src}"),
        )
    } else {
        (module.to_string(), src.to_string())
    })
}

/// The source of `module`.
pub fn source(module: &str) -> Option<&'static str> {
    MODULES.iter().find(|(m, _)| *m == module).map(|(_, s)| *s)
}

/// A digest of every source and header (part of `kernel_build`).
pub fn digest() -> String {
    let mut h = blake3::Hasher::new();
    for (n, s) in HEADERS.iter().chain(MODULES) {
        h.update(n.as_bytes());
        h.update(s.as_bytes());
    }
    for o in OPTIONS {
        h.update(o.as_bytes());
    }
    h.finalize().to_hex()[..16].to_string()
}

/// The block size of a WGSL entry point: its `@workgroup_size(x[, y[, z]])`,
/// whose terms are literals or `const`s of the module (the generated layout
/// constants included); for CUDA-only entries, `CUDA_ENTRIES`.
pub fn block_size(module: &str, entry: &str) -> Option<[u32; 3]> {
    if let Some(&(_, _, b)) = CUDA_ENTRIES.iter().find(|(m, e, _)| *m == module && *e == entry) {
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
