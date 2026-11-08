//! Resident pattern, inverse distortion/RI, halo convolution, detector and noise.
use super::traced::accumulate;
use super::*;
use crate::image_simulation::{
    FieldMapping, ImageGridSettings, ImageSimulationResult, ImageSimulationSettings, ImageSource,
    TestPattern,
};
use mklight_core::job::RunTimer;
use mklight_gpu::wave_ops::simulation::Simulation as Op;
use mklight_trace::{FieldPoint, SequentialTracer};
use rayon::prelude::*;
use std::{collections::BTreeMap, sync::Arc};
#[derive(Clone)]
pub struct ResidentPsfGrid {
    mapping: Vec<FieldMapping>,
    size: usize,
    psf_size: usize,
    spacing_mm: f64,
    kernels: Vec<BufferHandle>,
    kernel_sums: Vec<f64>,
    warnings: Vec<String>,
    identity: String,
}
impl ResidentPsfGrid {
    pub fn mapping(&self) -> &[FieldMapping] {
        &self.mapping
    }
    pub fn size(&self) -> usize {
        self.size
    }
    pub fn psf_size(&self) -> usize {
        self.psf_size
    }
    pub fn spacing_mm(&self) -> f64 {
        self.spacing_mm
    }
    pub fn warnings(&self) -> &[String] {
        &self.warnings
    }
}

/// Host rule of a node PSF's transform (the PSF rule of `gpu::analysis`).
const IMAGE_HOST: super::split::HostRule = super::split::HostRule {
    limit: 1e8,
    ratio: 4.,
};
fn invalid(s: &str) -> MkError {
    MkError::InvalidParam(format!("image simulation: {s}"))
}
fn identity(t: &SequentialTracer, s: &ImageGridSettings) -> String {
    let mut h = blake3::Hasher::new();
    h.update(t.system().snapshot.hash().to_hex().as_bytes());
    h.update(&serde_json::to_vec(s).unwrap());
    for c in t.system().library.catalogs() {
        let a = serde_json::to_vec(c.as_ref()).unwrap();
        h.update(&(a.len() as u64).to_le_bytes());
        h.update(&a);
    }
    h.finalize().to_hex().to_string()
}
fn p5_stats(v: &serde_json::Value, stats: &mut PlanStats) {
    if let Some(s) = v.get("plan_stats")
        && let Ok(s) = serde_json::from_value::<PlanStats>(s.clone())
    {
        accumulate(stats, &s);
    }
    if let Some(v) = v.get("resident_execution") {
        p5_stats(v, stats);
    }
    if let Some(v) = v.get("extra_plans").and_then(|v| v.as_array()) {
        for v in v {
            p5_stats(v, stats);
        }
    }
}
/// Build the optical grid: P5 distortion, host f64 relative illumination,
/// pupils from the run's work-split design (`split`), and resident PSFs.
pub fn build_psf_grid(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &ImageGridSettings,
    ctx: &JobContext,
) -> MkResult<WaveResult<ResidentPsfGrid>> {
    build_psf_grid_shared(b, t, s, None, ctx)
}
/// [`build_psf_grid`] sharing host lattices and pupils through `memo`.
pub fn build_psf_grid_shared(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &ImageGridSettings,
    memo: Option<&dyn mklight_core::memo::Memo>,
    ctx: &JobContext,
) -> MkResult<WaveResult<ResidentPsfGrid>> {
    use super::analysis::{Source, Staged, gather};
    let timer = RunTimer::start();
    if !(2..=33).contains(&s.field_grid)
        || !s.field_extent.is_finite()
        || s.field_extent <= 0.
        || s.field_center.iter().any(|v| !v.is_finite())
        || (s.psf_size != 0 && (!(3..=2047).contains(&s.psf_size) || s.psf_size.is_multiple_of(2)))
        || s.psf_spacing_mm.is_some_and(|v| !v.is_finite() || v <= 0.)
    {
        return Err(invalid("invalid field/PSF grid or spacing"));
    }
    let mut stats = PlanStats::default();
    let d = mklight_analysis::gpu::grid_distortion_job(
        b,
        t,
        &mklight_analysis::curvature::GridDistortionSettings {
            grid_size: s.field_grid,
            half_width: s.field_extent,
            field_center: s.field_center,
            ..Default::default()
        },
        &ctx.subrange(0., 0.05),
    )?;
    p5_stats(&d.run_info.settings, &mut stats);
    let evidence = vec![d.run_info.clone()];
    let mut warnings = d.run_info.warnings;
    let coordinates: Vec<_> = d.points.iter().map(|p| p.field).collect();
    if d.points
        .iter()
        .any(|p| p.real.iter().chain(&p.paraxial).any(|v| !v.is_finite()))
    {
        return Err(invalid("field grid contains unusable chief"));
    }
    let wls = crate::analysis::wavelengths(t, &Default::default())?;
    let src = Source::new(b, t, memo, ctx)?;
    // Relative illumination of the grid nodes is host f64 preparation, the
    // CPU grid's own rows: the device projected-solid-angle kernel runs the
    // nested object/aim solves of every pupil node inside one command and
    // took ≈1.4 s per wavelength on a 29-surface lens (9 nodes × 576 pupil
    // samples) against ≈0.2 s on the host for all five wavelengths. It runs
    // beside device work: during device tracing, or (designs whose host
    // traces) during the device PSFs.
    let illumination = || -> MkResult<(Vec<f64>, f64, Vec<String>, f64)> {
        let clock = std::time::Instant::now();
        let mut warnings = vec![];
        let mut irradiance = vec![0.; coordinates.len()];
        let mut axial = 0.;
        // Wavelengths side by side; sums and warnings in wavelength order
        // (as the CPU grid's).
        let per_wavelength: Vec<_> = wls
            .par_iter()
            .enumerate()
            .map(|(i, &(wl, _))| {
                let mut w = vec![];
                let r = mklight_analysis::illumination::illumination_rows(
                    t,
                    &coordinates,
                    wl,
                    s.illumination_radial,
                    s.illumination_azimuth,
                    &mut w,
                    &ctx.subrange(
                        0.05 + 0.15 * i as f64 / wls.len() as f64,
                        0.05 + 0.15 * (i + 1) as f64 / wls.len() as f64,
                    ),
                );
                (w, r)
            })
            .collect();
        for ((w, r), &(_, weight)) in per_wavelength.into_iter().zip(&wls) {
            warnings.extend(w);
            let (rows, omega) = r?;
            for (sum, row) in irradiance.iter_mut().zip(rows) {
                *sum += weight * row[0];
            }
            axial += weight * omega;
        }
        Ok((irradiance, axial, warnings, clock.elapsed().as_secs_f64()))
    };
    let prepared = d
        .points
        .par_iter()
        .map(|p| t.prepare_field(FieldPoint::normalized(p.field[0], p.field[1])))
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<MkResult<Vec<_>>>()?;
    let requests: Vec<_> = d
        .points
        .iter()
        .zip(&prepared)
        .enumerate()
        .map(|(i, (p, f))| (i, p.field, f))
        .collect();
    let host_traces = src.design.host_lattice();
    let (per, early) = std::thread::scope(|scope| {
        let early = (!host_traces).then(|| scope.spawn(illumination));
        let per = src.trace(
            &requests,
            &wls,
            &s.pupil,
            &ctx.subrange(0.2, 0.6),
            &mut stats,
            &mut warnings,
        );
        (per, early.map(|h| h.join().expect("illumination rows")))
    });
    let per = per?.into_iter().collect::<MkResult<Vec<_>>>()?;
    let dx = s.psf_spacing_mm.unwrap_or_else(|| {
        per.iter()
            .flat_map(|f| &f.pupils)
            .map(|(_, _, p)| p.pupil.lambda_f() / 2.5)
            .fold(f64::INFINITY, f64::min)
    });
    let mut size = s.psf_size;
    if size == 0 {
        for r in concurrent(&per, |_, f| {
            let mut st = PlanStats::default();
            src.image_grid(
                f,
                0,
                dx,
                crate::analysis::PsfMethod::Fft,
                &ctx.subrange(0.6, 0.6),
                &mut st,
            )
            .map(|g| (g, st))
        }) {
            let (g, st) = r?;
            accumulate(&mut stats, &st);
            size = size.max(g);
        }
        size = (size | 1).min(2047);
    }
    let kernel_bytes = (size * size * 4) as u64 * per.len() as u64;
    let budget = b
        .info()
        .recommended_working_set_bytes
        .unwrap_or(1024 * 1024 * 1024)
        / 2;
    if kernel_bytes > budget {
        return Err(MkError::Unsupported(format!(
            "resident PSF grid needs {kernel_bytes} bytes, exceeding its {budget}-byte working-set budget"
        )));
    }
    let count = d.points.len();
    let range = |i: usize| {
        ctx.subrange(
            0.6 + 0.4 * i as f64 / count as f64,
            0.6 + 0.4 * (i + 1) as f64 / count as f64,
        )
    };
    let ((results, ran), late) = std::thread::scope(|scope| {
        let late = host_traces.then(|| scope.spawn(illumination));
        let results = super::split::schedule_with(
            src.design,
            per,
            |i, mut f| {
                let mut stats = PlanStats::default();
                let warnings = src.ensure_window(
                    &mut f,
                    size as f64 * dx,
                    &s.pupil,
                    &range(i).subrange(0., 0.1),
                    &mut stats,
                )?;
                Ok(Staged::Done {
                    value: f,
                    stats,
                    warnings,
                })
            },
            |i, staged| {
                let sub = range(i);
                staged.then(|f, stats, warnings| {
                    let psf = polychromatic_psf(
                        b,
                        t.system().snapshot.clone(),
                        &f.weighted(),
                        &crate::PolyPsfSettings {
                            grid_size: size,
                            spacing_mm: dx,
                            normalization: PsfNormalization::UnitEnergy,
                            transform: PsfTransform::Matrix,
                        },
                        false,
                        &sub.subrange(0.1, 0.9),
                    )?;
                    accumulate(stats, &psf.stats);
                    let metrics = psf.value.metrics(
                        b,
                        t.system().snapshot.clone(),
                        &sub.subrange(0.9, 1.),
                    )?;
                    accumulate(stats, &metrics.stats);
                    if metrics.value.energy_in_grid < crate::analysis::MIN_ENERGY_IN_GRID {
                        warnings.push(
                            WaveWarning::PsfTruncated {
                                energy_in_grid: metrics.value.energy_in_grid,
                            }
                            .to_string(),
                        );
                    }
                    warnings.extend(psf.value.warnings.iter().map(ToString::to_string));
                    Ok((
                        psf.value.image,
                        metrics.value.energy_in_grid / (dx * dx),
                        f.sampling(),
                        f.reference,
                    ))
                })
            },
            src.reference(
                &|i, staged: Staged<super::analysis::Field>| {
                    let sub = range(i);
                    staged.then(|f, _, warnings| {
                        // The reference's kernel, written into a shared
                        // allocation for the device convolution.
                        let host = f.host_weighted().ok_or_else(|| {
                            MkError::Unsupported("host transform without host pupils".into())
                        })?;
                        let psf = crate::analysis::psf::compute_psf(
                            &host,
                            crate::analysis::PsfMethod::Fft,
                            size,
                            dx,
                            PsfNormalization::UnitEnergy,
                            &sub.subrange(0.1, 1.),
                        )?;
                        warnings.extend(psf.warnings.iter().map(ToString::to_string));
                        let kernel = b.fill_buffer(
                            BufferDesc::new(ElementType::F32, [size, size]),
                            &mut |mut a| {
                                for (d, v) in a.as_mut_slice::<f32>()?.iter_mut().zip(&psf.data) {
                                    *d = *v as f32;
                                }
                                Ok(())
                            },
                        )?;
                        Ok((
                            kernel,
                            psf.energy_in_grid() / (dx * dx),
                            f.sampling(),
                            f.reference,
                        ))
                    })
                },
                &|staged| match staged {
                    Staged::Done { value, .. } => value
                        .pupils
                        .iter()
                        .map(|(_, _, p)| {
                            let n = p.pupil.n as f64;
                            n * size as f64 * (n + size as f64)
                        })
                        .sum(),
                    Staged::Skipped { .. } => 0.,
                },
                IMAGE_HOST,
            ),
        );
        (results, late.map(|h| h.join().expect("illumination rows")))
    });
    src.record(ran);
    let nodes = gather(results, &mut stats, &mut warnings)?;
    if nodes.len() != count {
        return Err(invalid("field grid contains an unusable pupil"));
    }
    let (irradiance, axial, illumination_warnings, illumination_seconds) =
        early.or(late).expect("illumination rows ran")?;
    warnings.extend(illumination_warnings);
    let mut kernels = vec![];
    let mut kernel_sums = vec![];
    let mut mapping = vec![];
    for (i, (p, (kernel, sum, sampling, reference))) in d.points.iter().zip(nodes).enumerate() {
        kernels.push(kernel);
        kernel_sums.push(sum);
        mapping.push(FieldMapping {
            pupil_sampling: sampling,
            field: p.field,
            ideal_mm: p.paraxial,
            real_mm: reference,
            relative_illumination: irradiance[i] / axial,
        });
    }
    let mut run_info = src.info(timer, "image_psf_grid", s, &stats, warnings.clone());
    run_info.settings["p5_evidence"] = serde_json::to_value(evidence).unwrap();
    run_info.settings["illumination"] =
        serde_json::json!({"backend":"cpu_reference","seconds":illumination_seconds});
    Ok(WaveResult {
        value: ResidentPsfGrid {
            mapping,
            size: s.field_grid,
            psf_size: size,
            spacing_mm: dx,
            kernels,
            kernel_sums,
            warnings,
            identity: identity(t, s),
        },
        run_info,
        stats,
    })
}
fn raster(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &ImageSimulationSettings,
    ctx: &JobContext,
    stats: &mut PlanStats,
) -> MkResult<BufferHandle> {
    let fill = |w: usize, h: usize, c: usize, data: &[f64]| -> MkResult<BufferHandle> {
        if w == 0
            || h == 0
            || w > 8192
            || h > 8192
            || ![1, 3].contains(&c)
            || data.len() != w * h * c
        {
            return Err(invalid("inconsistent linear-light buffer"));
        }
        b.fill_buffer(
            BufferDesc::new(ElementType::F32, [h, w, c]),
            &mut |mut a| {
                for (i, (dst, v)) in a.as_mut_slice::<f32>()?.iter_mut().zip(data).enumerate() {
                    if i % 4096 == 0 {
                        ctx.check()?;
                    }
                    if !v.is_finite() || *v < 0. || !(*v as f32).is_finite() {
                        return Err(invalid("invalid linear-light sample"));
                    }
                    *dst = *v as f32;
                }
                Ok(())
            },
        )
    };
    match &s.source {
        ImageSource::Buffer {
            width,
            height,
            channels,
            data,
        } => fill(*width, *height, *channels, data),
        ImageSource::Path { path } => {
            let r = crate::image_simulation::read_png(path)?;
            fill(r.w, r.h, r.c, &r.data)
        }
        ImageSource::Pattern { pattern } => {
            let mut p = ResidentPlan::new(t.system().snapshot.clone());
            let output = p.scratch(BufferDesc::new(ElementType::F32, [s.size[1], s.size[0], 1]));
            p.operations
                .push(ResidentOp::Wave(w::Request::Simulation(Op::Pattern {
                    size: s.size,
                    kind: match pattern {
                        TestPattern::Grid => 0,
                        TestPattern::SiemensStar => 1,
                        TestPattern::SlantedEdge => 2,
                        TestPattern::TextChart => 3,
                    },
                    output,
                })));
            p.outputs = vec![output];
            let r = b.submit_plan(p, ctx)?.wait()?;
            accumulate(stats, &r.stats);
            Ok(r.outputs[0].clone())
        }
    }
}
/// Full image simulation, with reusable immutable optical grids. File decoding
/// and final owned result assembly stay on the host.
pub fn image_simulation_job(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &ImageSimulationSettings,
    grid: Option<Arc<ResidentPsfGrid>>,
    ctx: &JobContext,
) -> MkResult<ImageSimulationResult> {
    render(b, t, s, grid, true, ctx)
}

fn render(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    s: &ImageSimulationSettings,
    grid: Option<Arc<ResidentPsfGrid>>,
    prefer_whole_image: bool,
    ctx: &JobContext,
) -> MkResult<ImageSimulationResult> {
    let timer = RunTimer::start();
    if s.size.iter().any(|n| !(2..=4096).contains(n))
        || !(4..=512).contains(&s.tile_size)
        || !s.read_noise_sigma.is_finite()
        || s.read_noise_sigma < 0.
        || s.full_well_electrons
            .is_some_and(|v| !v.is_finite() || v <= 0. || v > 1e12)
        || s.detector_aperture_mm
            .is_some_and(|v| v.iter().any(|v| !v.is_finite() || *v <= 0.))
    {
        return Err(invalid("invalid image/tile/detector/noise settings"));
    }
    let mut stats = PlanStats::default();
    let mut optical_evidence = None;
    let grid = if let Some(g) = grid {
        if g.identity != identity(t, &s.optics) {
            return Err(invalid("cached PSF grid optical identity differs"));
        }
        g
    } else {
        let r = build_psf_grid(b, t, &s.optics, &ctx.subrange(0., 0.65))?;
        accumulate(&mut stats, &r.stats);
        optical_evidence = Some(r.run_info);
        Arc::new(r.value)
    };
    let [width, height] = s.size;
    let channels = if s.monochrome { 1 } else { 3 };
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in &grid.mapping {
        for j in 0..2 {
            lo[j] = lo[j].min(p.ideal_mm[j]);
            hi[j] = hi[j].max(p.ideal_mm[j]);
        }
    }
    let pitch = [
        (hi[0] - lo[0]) / width as f64,
        (hi[1] - lo[1]) / height as f64,
    ];
    if pitch.iter().any(|v| *v <= 0.) {
        return Err(invalid("non-positive first-order image extent"));
    }
    if s.detector_aperture_mm
        .is_some_and(|a| a[0] > 4. * pitch[0] || a[1] > 4. * pitch[1])
    {
        return Err(invalid("pixel aperture exceeds four output pixels"));
    }
    if grid.spacing_mm * grid.psf_size as f64 / pitch[0].min(pitch[1]) > 1024. {
        return Err(invalid("PSF exceeds 1024 output pixels"));
    }
    let source = raster(b, t, s, &ctx.subrange(0.65, 0.66), &mut stats)?;
    let nodes = b.fill_buffer(
        BufferDesc::new(ElementType::DoubleFloat, [grid.mapping.len(), 5]),
        &mut |mut a| {
            for (row, p) in a.as_mut_slice::<Df32>()?.chunks_mut(5).zip(&grid.mapping) {
                row.copy_from_slice(
                    &[
                        p.ideal_mm[0],
                        p.ideal_mm[1],
                        p.real_mm[0],
                        p.real_mm[1],
                        p.relative_illumination,
                    ]
                    .map(Df32::from_f64),
                );
            }
            Ok(())
        },
    )?;
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    let source = p.import(source.view());
    let mapping = p.import(nodes.view());
    let warped = p.scratch(BufferDesc::new(ElementType::F32, [height, width, channels]));
    let uv = p.scratch(BufferDesc::new(
        ElementType::DoubleFloat,
        [height, width, 2],
    ));
    p.operations
        .push(ResidentOp::Wave(w::Request::Simulation(Op::Warp {
            source: p.view(source)?,
            mapping: p.view(mapping)?,
            size: s.size,
            channels,
            origin: lo.map(Df32::from_f64),
            pitch: pitch.map(Df32::from_f64),
            output: warped,
            uv,
        })));
    p.outputs = vec![warped, uv];
    let r = b.submit_plan(p, &ctx.subrange(0.66, 0.72))?.wait()?;
    accumulate(&mut stats, &r.stats);
    let warped = r.outputs[0].clone();
    let uv = r.outputs[1].clone();
    let physical_radius = grid.psf_size as f64 * 0.5 * grid.spacing_mm
        + s.detector_aperture_mm.map_or(0., |a| a[0].max(a[1]) * 0.5);
    let radius = (physical_radius / pitch[0].min(pitch[1])).ceil() as usize + 1;
    let whole_canvas = crate::fft::next_fast_len(width.max(height) + 4 * radius);
    let budget = b
        .info()
        .recommended_working_set_bytes
        .unwrap_or(1024 * 1024 * 1024)
        / 4;
    let estimated = whole_canvas as u64 * whole_canvas as u64 * 8 * (12 + 3 * channels) as u64;
    let tile = if prefer_whole_image && whole_canvas <= 8192 && estimated <= budget {
        width.max(height)
    } else {
        s.tile_size
    };
    let nx = width.div_ceil(tile);
    let ny = height.div_ceil(tile);
    let mut tiles = vec![];
    let mut kernel_cache = BTreeMap::<(usize, usize), BufferHandle>::new();
    for i in 0..nx * ny {
        let sub = ctx.subrange(
            0.72 + 0.25 * i as f64 / (nx * ny) as f64,
            0.72 + 0.25 * (i + 1) as f64 / (nx * ny) as f64,
        );
        let x0 = i % nx * tile;
        let y0 = i / nx * tile;
        let tw = tile.min(width - x0);
        let th = tile.min(height - y0);
        let bounds = [x0, y0, tw, th];
        let canvas = crate::fft::next_fast_len(tw.max(th) + 4 * radius);
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let v = p.import(uv.view());
        let active = p.scratch(BufferDesc::new(ElementType::U32, [grid.size * grid.size]));
        p.operations
            .push(ResidentOp::Wave(w::Request::Simulation(Op::Active {
                uv: p.view(v)?,
                grid: grid.size,
                tile: bounds,
                radius,
                output: active,
            })));
        p.outputs = vec![active];
        let r = b.submit_plan(p, &sub.subrange(0., 0.05))?.wait()?;
        accumulate(&mut stats, &r.stats);
        let guard = b.read_view(r.outputs[0].view())?;
        let active: Vec<_> = guard
            .as_slice::<u32>()?
            .iter()
            .enumerate()
            .filter_map(|(i, v)| (*v != 0).then_some(i))
            .collect();
        let mut previous: Option<BufferId> = None;
        let mut p = ResidentPlan::new(t.system().snapshot.clone());
        let image = p.import(warped.view());
        let coords = p.import(uv.view());
        let mut cached_outputs = vec![];
        for node in active.iter().copied() {
            let kernel = if let Some(k) = kernel_cache.get(&(node, canvas)) {
                p.import(k.view())
            } else {
                let psf = p.import(grid.kernels[node].view());
                let output = p.scratch(BufferDesc::new(ElementType::Complex32, [canvas, canvas]));
                p.operations
                    .push(ResidentOp::Wave(w::Request::Simulation(Op::Kernel {
                        psf: p.view(psf)?,
                        spacing: Df32::from_f64(grid.spacing_mm),
                        normalization: Df32::from_f64(grid.kernel_sums[node]),
                        pitch: pitch.map(Df32::from_f64),
                        aperture: s.detector_aperture_mm.map(|v| v.map(Df32::from_f64)),
                        radius,
                        canvas,
                        output,
                    })));
                output
            };
            let mut planes = vec![];
            for channel in 0..channels {
                let weighted = p.scratch(BufferDesc::new(ElementType::Complex32, [canvas, canvas]));
                p.operations
                    .push(ResidentOp::Wave(w::Request::Simulation(Op::Weighted {
                        image: p.view(image)?,
                        uv: p.view(coords)?,
                        grid: grid.size,
                        node,
                        channel,
                        tile: bounds,
                        radius,
                        canvas,
                        output: weighted,
                    })));
                let output = p.scratch(BufferDesc::new(ElementType::Complex32, [canvas, canvas]));
                p.operations
                    .push(ResidentOp::Transforms(tr::Request::Convolve {
                        left: p.view(weighted)?,
                        right: p.view(kernel)?,
                        output,
                    }));
                planes.push(p.view(output)?);
            }
            let prev = previous;
            let output = p.scratch(BufferDesc::new(ElementType::F32, [th, tw, channels]));
            p.operations
                .push(ResidentOp::Wave(w::Request::Simulation(Op::Crop {
                    planes,
                    previous: prev.map(|i| p.view(i).unwrap()),
                    size: [tw, th],
                    radius,
                    output,
                })));
            previous = Some(output);
            let bytes = (canvas * canvas * 8) as u64;
            if !kernel_cache.contains_key(&(node, canvas))
                && bytes * (cached_outputs.len() as u64 + 1) <= 64 * 1024 * 1024
            {
                cached_outputs.push((node, kernel));
            }
        }
        let output = if let Some(output) = previous {
            p.outputs = vec![output];
            p.outputs.extend(cached_outputs.iter().map(|(_, id)| *id));
            let r = b.submit_plan(p, &sub.subrange(0.05, 1.))?.wait()?;
            accumulate(&mut stats, &r.stats);
            for (i, (node, _)) in cached_outputs.iter().enumerate() {
                let bytes = (canvas * canvas * 8) as u64;
                while kernel_cache
                    .values()
                    .map(|h| h.desc().len().unwrap() as u64 * 8)
                    .sum::<u64>()
                    + bytes
                    > 64 * 1024 * 1024
                {
                    kernel_cache.pop_first();
                }
                kernel_cache.insert((*node, canvas), r.outputs[i + 1].clone());
            }
            Some(r.outputs[0].clone())
        } else {
            None
        };
        let output = if let Some(h) = output {
            h
        } else {
            b.fill_buffer(
                BufferDesc::new(ElementType::F32, [th, tw, channels]),
                &mut |mut a| {
                    a.as_mut_slice::<f32>()?.fill(0.);
                    Ok(())
                },
            )?
        };
        tiles.push(output);
    }
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    let views: Vec<_> = tiles
        .iter()
        .map(|h| {
            let i = p.import(h.view());
            p.view(i).unwrap()
        })
        .collect();
    let assembled = p.scratch(BufferDesc::new(ElementType::F32, [height, width, channels]));
    p.operations
        .push(ResidentOp::Wave(w::Request::Simulation(Op::Assemble {
            tiles: views,
            size: s.size,
            channels,
            tile_size: tile,
            output: assembled,
        })));
    let output = p.scratch(BufferDesc::new(ElementType::F32, [height, width, channels]));
    p.operations
        .push(ResidentOp::Wave(w::Request::Simulation(Op::Noise {
            input: p.view(assembled)?,
            seed: s.seed,
            well: s.full_well_electrons.map(|v| v as f32),
            sigma: s.read_noise_sigma as f32,
            output,
        })));
    p.seed = Some(s.seed);
    p.outputs = vec![output];
    let r = b.submit_plan(p, &ctx.subrange(0.97, 0.99))?.wait()?;
    accumulate(&mut stats, &r.stats);
    let guard = b.read_view(r.outputs[0].view())?;
    let mut image = Vec::with_capacity(width * height * channels);
    for block in guard.as_slice::<f32>()?.chunks(4096) {
        ctx.check()?;
        if block.iter().any(|v| !v.is_finite()) {
            return Err(MkError::Unsupported(
                "image values exceed the finite f32 noise/convolution domain".into(),
            ));
        }
        image.extend(block.iter().map(|v| *v as f64));
    }
    let mut run_info = super::analysis::info(
        b,
        t,
        timer,
        "image_simulation",
        s,
        &stats,
        grid.warnings.clone(),
    );
    run_info.seed = Some(s.seed);
    run_info.settings["optical_grid_identity"] = grid.identity.clone().into();
    run_info.settings["image_plan_version"] = 2.into();
    run_info.settings["effective_tile_size"] = tile.into();
    if let Some(r) = optical_evidence {
        run_info.settings["optical_grid_execution"] = serde_json::to_value(r).unwrap();
    }
    ctx.report(1., "image simulation");
    Ok(ImageSimulationResult {
        size: s.size,
        channels,
        pixel_size_mm: pitch,
        origin_mm: lo,
        field_grid: grid.size,
        field_mapping: grid.mapping.clone(),
        psf_size: grid.psf_size,
        psf_spacing_mm: grid.spacing_mm,
        image,
        run_info,
    })
}

#[cfg(all(test, target_os = "macos"))]
mod tests {
    use super::*;
    #[test]
    fn tiled_and_whole_image_match_direct_impulse_deposition() {
        impulse_suite(
            32,
            0.1,
            &[[16, 16], [1, 16]],
            &[(16, false), (23, false), (32, true)],
        );
    }
    #[test]
    #[ignore = "host image-size and halo-support caps"]
    fn image_size_and_halo_caps() {
        impulse_suite(4096, 0.001, &[[2048, 2048]], &[(512, true)]);
        impulse_suite(64, 6.4, &[[32, 32]], &[(64, true)]);
    }
    fn impulse_suite(n: usize, spacing: f64, centres: &[[usize; 2]], partitions: &[(usize, bool)]) {
        for origin in [
            mklight_metal::LibraryChoice::Precompiled,
            mklight_metal::LibraryChoice::RuntimeSource,
        ] {
            let b = mklight_metal::MetalBackend::with_options(&mklight_metal::MetalOptions {
                library: origin,
                ..Default::default()
            })
            .unwrap()
            .expect("host Metal required");
            let ctx = JobContext::new();
            let t = mklight_analysis::tracer_for(mklight_core::samples::singlet()).unwrap();
            let mut s = ImageSimulationSettings {
                size: [n, n],
                ..Default::default()
            };
            s.optics.field_grid = 2;
            let kernel = b
                .fill_buffer(BufferDesc::new(ElementType::F32, [5, 5]), &mut |mut a| {
                    a.as_mut_slice::<f32>()?.fill(0.04);
                    Ok(())
                })
                .unwrap();
            let mapping = vec![[-1., -1.], [1., -1.], [-1., 1.], [1., 1.]]
                .into_iter()
                .map(|p| FieldMapping {
                    pupil_sampling: vec![],
                    field: p,
                    ideal_mm: p,
                    real_mm: p,
                    relative_illumination: 1.,
                })
                .collect();
            let grid = Arc::new(ResidentPsfGrid {
                mapping,
                size: 2,
                psf_size: 5,
                spacing_mm: spacing,
                kernels: vec![kernel; 4],
                kernel_sums: vec![1.; 4],
                warnings: vec![],
                identity: identity(&t, &s.optics),
            });
            for &centre in centres {
                let mut data = vec![0.; n * n];
                data[centre[1] * n + centre[0]] = 1.;
                s.source = ImageSource::Buffer {
                    width: n,
                    height: n,
                    channels: 1,
                    data,
                };
                let mut expected = vec![0.; n * n];
                let pitch = 2. / n as f64;
                for j in 0..5 {
                    for i in 0..5 {
                        let x = centre[0] as f64 + (i as f64 - 2.) * spacing / pitch;
                        let y = centre[1] as f64 + (j as f64 - 2.) * spacing / pitch;
                        let ix = x.floor() as isize;
                        let iy = y.floor() as isize;
                        for dy in 0..2 {
                            for dx in 0..2 {
                                let a = ix + dx;
                                let b = iy + dy;
                                if a >= 0 && b >= 0 && a < n as isize && b < n as isize {
                                    expected[b as usize * n + a as usize] += 0.04
                                        * (1. - (x - a as f64).abs())
                                        * (1. - (y - b as f64).abs());
                                }
                            }
                        }
                    }
                }
                for &(tile, whole) in partitions {
                    s.tile_size = tile;
                    let result = render(&b, &t, &s, Some(grid.clone()), whole, &ctx).unwrap();
                    eprintln!(
                        "P6_IMAGE_CAP n={n} spacing={spacing} requested_tile={tile} effective_tile={} resident={}",
                        result.run_info.settings["effective_tile_size"],
                        result.run_info.settings["resident"]["plan_stats"]
                    );
                    let error = result
                        .image
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a - b).abs())
                        .sum::<f64>();
                    assert!(
                        error < if n > 32 { 2e-3 } else { 2e-5 },
                        "impulse {centre:?}, tile {tile}, whole {whole}: {error}"
                    );
                    assert!(
                        (result.image.iter().sum::<f64>() - expected.iter().sum::<f64>()).abs()
                            < if n > 32 { 1e-3 } else { 1e-5 }
                    );
                }
            }
        }
    }
}
