//! Occupancy of the specialised transport kernels (ignored; prints a
//! table): maximum threads per threadgroup (register pressure), SIMD width
//! and the pipeline build time, for the stray wavefront's closest-hit and
//! interaction kernels and the ghost trace.
#![cfg(target_os = "macos")]
use mklight_gpu::consts::*;
use mklight_metal::{MetalContext, MetalOptions, PipelineKey};
use objc2_metal::MTLComputePipelineState;
use std::time::Instant;

#[test]
#[ignore]
fn kernel_occupancy() {
    let Some(ctx) = MetalContext::new(&MetalOptions::default()).unwrap() else {
        return;
    };
    let report = |label: String, key: PipelineKey| {
        let t = Instant::now();
        let pso = ctx.pipeline(&key).unwrap();
        println!(
            "{label:<52} max threads {:>4}  width {}  build {:6.2} s",
            pso.maxTotalThreadsPerThreadgroup(),
            pso.threadExecutionWidth(),
            t.elapsed().as_secs_f64()
        );
    };
    for (name, features) in [
        ("weighted plain", 1),
        ("weighted, Lambertian scatter", 1 | 32),
        ("weighted, Gaussian/ABg scatter", 1 | 32 | 64),
        ("weighted, scatter, films", 1 | 32 | 4),
        ("weighted, scatter, aspheres", 1 | 32 | 8),
        ("weighted, scatter, DOE", 1 | 32 | 16),
        ("weighted, scatter, kinoform DOE", 1 | 32 | 16 | 128),
        ("split, scatter", 2 | 32),
        ("analog, scatter", 32),
        ("everything (split)", 0xfe),
    ] {
        for function in ["mkl_stray_wave_cast", "mkl_stray_wave_interact"] {
            report(
                format!("{function} / {name}"),
                PipelineKey::with_u32(function, MKL_STRAY_FEATURES_CONSTANT as u16, features),
            );
        }
    }
    for (name, features) in [
        ("geometry only", 0),
        ("coatings", MKL_GHOST_FEATURE_COATINGS),
        (
            "coatings, films",
            MKL_GHOST_FEATURE_COATINGS | MKL_GHOST_FEATURE_FILMS,
        ),
    ] {
        report(
            format!("mkl_ghost_trace / {name}"),
            PipelineKey::with_u32(
                "mkl_ghost_trace",
                MKL_GHOST_FEATURES_CONSTANT as u16,
                features,
            ),
        );
    }
}
