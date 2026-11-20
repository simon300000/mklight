//! The CUDA kernel sources: compilation options, in-memory headers, modules
//! and their compilation units. Shared by the crate and its build script
//! (which compiles the units ahead of time, `build.rs`), so it depends on
//! nothing but `blake3`.
//!
//! The units include the shared layout headers of `mklight-gpu`
//! (`include/*.h`) and the Metal kernels' numerics and surface code
//! (`crates/mklight-metal/shaders`), the latter through the MSL subset of
//! `kernels/metal_stdlib.h`, so the arithmetic is written once for Metal and
//! CUDA.

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
    (
        "trace_bindings.h",
        include_str!("../kernels/trace_bindings.h"),
    ),
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
        concat!(
            "#define MKL_OVERLAYS false\n",
            include_str!("../kernels/trace.cu")
        ),
    ),
    (
        "trace_overlays",
        concat!(
            "#define MKL_OVERLAYS true\n",
            include_str!("../kernels/trace.cu")
        ),
    ),
    // aiming.cu likewise (the WGSL modules append the aim solver to the
    // trace; here it calls trace_shared.h's core).
    (
        "aiming",
        concat!(
            "#define MKL_OVERLAYS false\n",
            include_str!("../kernels/aiming.cu")
        ),
    ),
    (
        "aiming_overlays",
        concat!(
            "#define MKL_OVERLAYS true\n",
            include_str!("../kernels/aiming.cu")
        ),
    ),
    // FftMtf (CUDA only): after wave.cu, whose helpers it uses.
    (
        "fft_mtf",
        concat!(
            include_str!("../kernels/wave.cu"),
            include_str!("../kernels/fft_mtf.cu")
        ),
    ),
];

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
            format!("#define MKL_ONE_ENTRY 1\n#define MKL_ENTRY_{entry} 1\n{src}"),
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

/// Every compilation unit `(name, source)`: one per module, or one per
/// `MKL_KERNEL` entry point of a module compiled per entry ([`unit`]).
pub fn units() -> Vec<(String, String)> {
    let mut out = Vec::new();
    for (module, src) in MODULES {
        if src.contains("MKL_ONE_ENTRY") {
            for part in src.split("MKL_KERNEL ").skip(1) {
                let entry: String = part
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !entry.is_empty() && !out.iter().any(|(n, _)| *n == format!("{module}.{entry}"))
                {
                    out.extend(unit(module, &entry));
                }
            }
        } else {
            out.push((module.to_string(), src.to_string()));
        }
    }
    out
}
