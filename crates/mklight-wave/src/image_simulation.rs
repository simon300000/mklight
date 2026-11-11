//! Space-varying incoherent image simulation (WAVE.md, image simulation).
//!
//! A square grid of normalised fields carries polychromatic PSFs, the
//! paraxial and real image points of each PSF's reference, and the
//! spectrally weighted relative illumination; between nodes all of them are
//! bilinear in the field. Each output pixel is mapped back through the real
//! mapping to its field, then through the paraxial mapping to the source,
//! and scaled by the relative illumination. Each warped pixel then spreads
//! with its own interpolated PSF, `out = Σ_k K_k ∗ (w_k · I)` with `w_k` the
//! pixel's bilinear weight of node `k`. This is evaluated per tile by FFT
//! linear convolution over a halo of the kernel radius, so tiles are exactly
//! seamless.
use crate::analysis::{field_pupils, wavelengths};
use crate::fft::{Dir, fft2, next_fast_len};
use crate::{AmplitudeModel, PolyPsfSettings, PsfNormalization, PupilSettings};
use mklight_analysis::curvature::{GridDistortionSettings, grid_distortion_job};
use mklight_analysis::illumination::illumination_rows;
use mklight_core::job::{Progress, RunTimer};
use mklight_core::rng::Philox;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::{FieldPoint, SequentialTracer};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TestPattern {
    #[default]
    Grid,
    SiemensStar,
    SlantedEdge,
    TextChart,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum ImageSource {
    Pattern {
        pattern: TestPattern,
    },
    /// Linear-light samples, row-major interleaved, row zero at low y.
    /// One or three channels.
    Buffer {
        width: usize,
        height: usize,
        channels: usize,
        data: Vec<f64>,
    },
    /// An 8- or 16-bit PNG file; code values are decoded as sRGB to linear
    /// light and alpha is ignored.
    Path {
        path: String,
    },
}
impl Default for ImageSource {
    fn default() -> Self {
        Self::Pattern {
            pattern: TestPattern::Grid,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageSimulationSettings {
    pub source: ImageSource,
    pub size: [usize; 2],
    pub monochrome: bool,
    pub optics: ImageGridSettings,
    pub tile_size: usize,
    /// Active pixel aperture in mm. None = point-sampled detector.
    pub detector_aperture_mm: Option<[f64; 2]>,
    /// Gaussian read noise in linear-light units (Philox + Box–Muller).
    pub read_noise_sigma: f64,
    /// Full-scale photoelectrons. None = no Poisson shot noise.
    pub full_well_electrons: Option<f64>,
    pub seed: u64,
}
impl Default for ImageSimulationSettings {
    fn default() -> Self {
        Self {
            source: Default::default(),
            size: [64, 64],
            monochrome: true,
            optics: Default::default(),
            tile_size: 32,
            detector_aperture_mm: None,
            read_noise_sigma: 0.0,
            full_well_electrons: None,
            seed: 1,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ImageGridSettings {
    pub field_grid: usize,
    pub field_extent: f64,
    pub field_center: [f64; 2],
    pub pupil: PupilSettings,
    pub psf_size: usize,
    pub psf_spacing_mm: Option<f64>,
    pub illumination_radial: usize,
    pub illumination_azimuth: usize,
}
impl Default for ImageGridSettings {
    fn default() -> Self {
        Self {
            field_grid: 3,
            field_extent: std::f64::consts::FRAC_1_SQRT_2,
            field_center: [0.0; 2],
            pupil: PupilSettings {
                amplitude: AmplitudeModel::Radiometric,
                ..Default::default()
            },
            psf_size: 0,
            psf_spacing_mm: None,
            illumination_radial: 12,
            illumination_azimuth: 48,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldMapping {
    pub pupil_sampling: Vec<crate::analysis::PupilSampling>,
    pub field: [f64; 2],
    /// Paraxial chief image point.
    pub ideal_mm: [f64; 2],
    /// Real image point of the PSF reference (the primary chief ray unless
    /// `pupil.reference` is the centroid): the PSF centre.
    pub real_mm: [f64; 2],
    pub relative_illumination: f64,
}
/// Immutable reusable grid, including PSF kernels; engine cache keys include
/// snapshot, material-library hash and every optical setting.
#[derive(Clone)]
pub struct PsfGrid {
    pub mapping: Vec<FieldMapping>,
    pub size: usize,
    pub psf_size: usize,
    pub spacing_mm: f64,
    pub kernels: Vec<Vec<f64>>,
    pub warnings: Vec<String>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageSimulationResult {
    pub size: [usize; 2],
    pub channels: usize,
    pub pixel_size_mm: [f64; 2],
    pub origin_mm: [f64; 2],
    pub field_grid: usize,
    pub field_mapping: Vec<FieldMapping>,
    pub psf_size: usize,
    pub psf_spacing_mm: f64,
    pub image: Vec<f64>,
    pub run_info: RunInfo,
}
fn invalid(s: &str) -> MkError {
    MkError::InvalidParam(format!("image simulation: {s}"))
}

pub fn build_psf_grid(
    t: &SequentialTracer,
    s: &ImageGridSettings,
    ctx: &JobContext,
) -> MkResult<PsfGrid> {
    if !(2..=33).contains(&s.field_grid)
        || !(s.field_extent > 0.0 && s.field_extent.is_finite())
        || s.field_center.iter().any(|v| !v.is_finite())
        || (s.psf_size != 0 && (!(3..=2047).contains(&s.psf_size) || s.psf_size.is_multiple_of(2)))
        || s.psf_spacing_mm
            .is_some_and(|x| !(x > 0.0 && x.is_finite()))
    {
        return Err(invalid("invalid field/PSF grid or spacing"));
    }
    let d = grid_distortion_job(
        t,
        &GridDistortionSettings {
            grid_size: s.field_grid,
            half_width: s.field_extent,
            field_center: s.field_center,
            ..Default::default()
        },
        &ctx.subrange(0.0, 0.05),
    )?;
    let mut warnings = d.run_info.warnings;
    if d.points
        .iter()
        .any(|p| p.real.iter().chain(&p.paraxial).any(|x| !x.is_finite()))
    {
        return Err(invalid("field grid contains an unusable chief ray"));
    }
    let coordinates: Vec<_> = d.points.iter().map(|p| p.field).collect();
    let wls = wavelengths(t, &Default::default())?;
    // Irradiance Σ w_λ Ω_p,λ(H) relative to its on-axis value.
    // Wavelengths side by side; sums and warnings in wavelength order.
    let mut irradiance = vec![0.0; coordinates.len()];
    let mut axial = 0.0;
    let nw = wls.len() as f64;
    let rows: Vec<_> = wls
        .par_iter()
        .enumerate()
        .map(|(i, &(wl, _))| {
            let mut w = Vec::new();
            let r = illumination_rows(
                t,
                &coordinates,
                wl,
                s.illumination_radial,
                s.illumination_azimuth,
                &mut w,
                &ctx.subrange(
                    0.05 + 0.15 * i as f64 / nw,
                    0.05 + 0.15 * (i + 1) as f64 / nw,
                ),
            );
            (w, r)
        })
        .collect();
    for ((w, r), &(_, weight)) in rows.into_iter().zip(&wls) {
        warnings.extend(w);
        let (rows, omega0) = r?;
        for (sum, row) in irradiance.iter_mut().zip(rows) {
            *sum += weight * row[0];
        }
        axial += weight * omega0;
    }
    // All nodes set the common spacing; using only the first can undersample
    // a slightly faster cone elsewhere in the patch.
    let n = d.points.len() as f64;
    let mut per: Vec<_> = d
        .points
        .par_iter()
        .enumerate()
        .map(|(i, p)| {
            let f = t.prepare_field(FieldPoint::normalized(p.field[0], p.field[1]))?;
            field_pupils(
                t,
                &f,
                "image simulation",
                &wls,
                &s.pupil,
                None,
                &ctx.subrange(0.2 + 0.4 * i as f64 / n, 0.2 + 0.4 * (i + 1) as f64 / n),
            )
        })
        .collect::<Vec<_>>()
        .into_iter()
        .collect::<MkResult<_>>()?;
    for pupils in &per {
        warnings.extend(pupils.warnings.iter().cloned());
    }
    let dx = s.psf_spacing_mm.unwrap_or_else(|| {
        per.iter()
            .map(|fp| fp.min_lambda_f() / 2.5)
            .fold(f64::INFINITY, f64::min)
    });
    let size = if s.psf_size == 0 {
        (per.par_iter()
            .map(|fp| crate::sampling::image_grid(&fp.weighted(), dx))
            .max()
            .unwrap()
            | 1)
        .min(2047)
    } else {
        s.psf_size
    };
    // Every node's PSF side by side, merged in node order.
    type Node = (Vec<String>, Vec<f64>, FieldMapping);
    let nodes: Vec<MkResult<Node>> = d
        .points
        .par_iter()
        .zip(per.par_iter_mut())
        .enumerate()
        .map(|(i, (p, pupils))| -> MkResult<Node> {
            let mut warnings = Vec::new();
            let sub = ctx.subrange(0.6 + 0.4 * i as f64 / n, 0.6 + 0.4 * (i + 1) as f64 / n);
            warnings.extend(pupils.ensure_window(
                size as f64 * dx,
                &s.pupil,
                None,
                &sub.subrange(0.0, 0.1),
            )?);
            let psf = crate::polychromatic_psf(
                &pupils.weighted(),
                &PolyPsfSettings {
                    grid_size: size,
                    spacing_mm: dx,
                    normalization: PsfNormalization::UnitEnergy,
                },
                &sub.subrange(0.1, 1.0),
            )?;
            let energy = psf.energy_in_grid() / psf.total_energy;
            if energy < crate::analysis::MIN_ENERGY_IN_GRID {
                let w = crate::WaveWarning::PsfTruncated {
                    energy_in_grid: energy,
                };
                warnings.push(format!("field {:?}: {w}", p.field));
            }
            warnings.extend(psf.warnings.iter().map(ToString::to_string));
            let sum: f64 = psf.data.iter().sum();
            let kernel = psf.data.iter().map(|v| v / sum).collect();
            let mapping = FieldMapping {
                pupil_sampling: pupils.sampling(),
                field: p.field,
                ideal_mm: p.paraxial,
                real_mm: pupils.reference_mm,
                relative_illumination: irradiance[i] / axial,
            };
            Ok((warnings, kernel, mapping))
        })
        .collect();
    let mut kernels = Vec::new();
    let mut mapping = Vec::new();
    for node in nodes {
        let (w, kernel, map) = node?;
        warnings.extend(w);
        kernels.push(kernel);
        mapping.push(map);
    }
    Ok(PsfGrid {
        mapping,
        size: s.field_grid,
        psf_size: size,
        spacing_mm: dx,
        kernels,
        warnings,
    })
}

#[derive(Clone)]
pub(crate) struct Raster {
    pub(crate) w: usize,
    pub(crate) h: usize,
    pub(crate) c: usize,
    pub(crate) data: Vec<f64>,
}
/// What the warp samples: a raster (bilinear) or a built-in pattern
/// (analytic, over each output pixel's footprint).
enum Scene {
    Raster(Raster),
    Pattern(TestPattern),
}
fn source(s: &ImageSimulationSettings) -> MkResult<Scene> {
    match &s.source {
        ImageSource::Buffer {
            width,
            height,
            channels,
            data,
        } => {
            if *width == 0
                || *height == 0
                || *width > 8192
                || *height > 8192
                || ![1, 3].contains(channels)
                || data.len() != width * height * channels
                || data.iter().any(|v| !v.is_finite() || *v < 0.0)
            {
                return Err(invalid("inconsistent linear-light image buffer"));
            }
            Ok(Scene::Raster(Raster {
                w: *width,
                h: *height,
                c: *channels,
                data: data.clone(),
            }))
        }
        ImageSource::Path { path } => read_png(path).map(Scene::Raster),
        ImageSource::Pattern { pattern } => Ok(Scene::Pattern(*pattern)),
    }
}
/// Decode an 8- or 16-bit grey or RGB PNG (alpha dropped, palettes and low
/// bit depths expanded) from sRGB code values to linear light.
pub(crate) fn read_png(path: &str) -> MkResult<Raster> {
    let io = |e: std::io::Error| MkError::Io(format!("{path}: {e}"));
    let bad = |e: png::DecodingError| invalid(&format!("{path}: {e}"));
    let file = std::io::BufReader::new(std::fs::File::open(path).map_err(io)?);
    let mut decoder = png::Decoder::new(file);
    decoder.set_transformations(png::Transformations::EXPAND);
    let mut reader = decoder.read_info().map_err(bad)?;
    let mut buf = vec![
        0;
        reader
            .output_buffer_size()
            .ok_or_else(|| invalid("PNG too large"))?
    ];
    let info = reader.next_frame(&mut buf).map_err(bad)?;
    let (w, h) = (info.width as usize, info.height as usize);
    if w > 8192 || h > 8192 {
        return Err(invalid("PNG larger than 8192 pixels"));
    }
    let (stored, c) = match info.color_type {
        png::ColorType::Grayscale => (1, 1),
        png::ColorType::GrayscaleAlpha => (2, 1),
        png::ColorType::Rgb => (3, 3),
        png::ColorType::Rgba => (4, 3),
        png::ColorType::Indexed => return Err(invalid("unexpanded PNG palette")),
    };
    let code: Vec<f64> = match info.bit_depth {
        png::BitDepth::Sixteen => buf[..info.buffer_size()]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&b| u16::from_be_bytes(b) as f64 / 65535.0)
            .collect(),
        _ => buf[..info.buffer_size()]
            .iter()
            .map(|&b| b as f64 / 255.0)
            .collect(),
    };
    let data = code
        .chunks_exact(w * stored)
        .rev()
        .flat_map(|row| row.chunks_exact(stored))
        .flat_map(|px| px[..c].iter())
        .map(|&x| {
            if x <= 0.04045 {
                x / 12.92
            } else {
                ((x + 0.055) / 1.055).powf(2.4)
            }
        })
        .collect();
    Ok(Raster { w, h, c, data })
}
/// Pixel `(x, y)` of a built-in pattern drawn on a `w × h` grid,
/// area-averaged over the pixel.
#[cfg(test)]
fn pattern_value(p: TestPattern, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let (w_, h_) = (w as f64, h as f64);
    pattern_box(
        p,
        [x as f64 / w_, (x + 1) as f64 / w_],
        [y as f64 / h_, (y + 1) as f64 / h_],
        w,
        h,
    )
}
/// Mean of a built-in pattern over the box `u × v` of source coordinates
/// (0…1 across the pattern, row zero at low y). The patterns are analytic,
/// so the box can sit anywhere: the warp evaluates them over each output
/// pixel's footprint at its exact source position, with no raster in
/// between. Grid and slanted edge are integrated exactly; the Siemens star
/// by 16 × 16 midpoint samples; the text chart's glyphs are cells of a
/// `w × h` grid, integrated exactly by overlap.
fn pattern_box(p: TestPattern, u: [f64; 2], v: [f64; 2], w: usize, h: usize) -> f64 {
    let (du, dv) = (u[1] - u[0], v[1] - v[0]);
    match p {
        TestPattern::Grid => {
            // Integral of the white portion of the periodic bar pattern.
            let white = |t: f64| {
                let v = 8.0 * t;
                (0.9 * v.floor() + (v.fract() - 0.1).max(0.0)) / 8.0
            };
            (white(u[1]) - white(u[0])) / du * (white(v[1]) - white(v[0])) / dv
        }
        TestPattern::SiemensStar => {
            // 16×16 midpoint supersampling of the 24 spokes.
            let mut sum = 0.0;
            for j in 0..16 {
                for i in 0..16 {
                    let uu = u[0] + du * (i as f64 + 0.5) / 16.0;
                    let vv = v[0] + dv * (j as f64 + 0.5) / 16.0;
                    if ((vv - 0.5).atan2(uu - 0.5) * 24.0).cos() > 0.0 {
                        sum += 1.0;
                    }
                }
            }
            sum / 256.0
        }
        TestPattern::SlantedEdge => {
            // Exact area to the right of a line in the box.
            let primitive = |t: f64| {
                let edge = 0.45 + 0.1 * t;
                ((u[1] - edge).max(0.0).powi(2) - (u[0] - edge).max(0.0).powi(2)) / -0.2
            };
            (primitive(v[1]) - primitive(v[0])) / (du * dv)
        }
        TestPattern::TextChart => {
            // Repeated 5x7 bitmap glyphs M K L, at three spatial scales.
            const FONT: [[u8; 7]; 3] = [
                [17, 27, 21, 17, 17, 17, 17],
                [17, 18, 20, 24, 20, 18, 17],
                [16, 16, 16, 16, 16, 16, 31],
            ];
            let cell = |x: usize, y: usize| {
                // Glyph bitmaps and scale bands are written top to bottom.
                let y = h - 1 - y;
                let band = (y as f64 * 3.0 / h as f64) as usize;
                let scale = [4, 2, 1][band.min(2)];
                let (gx, gy) = (x / scale, y / scale);
                let glyph = (gx / 6) % 3;
                if gy % 9 < 7 && gx % 6 < 5 && FONT[glyph][gy % 9] & (1 << (4 - gx % 6)) != 0 {
                    0.0
                } else {
                    1.0
                }
            };
            // Overlap of the box with each cell it touches.
            let span = |t: [f64; 2], n: usize| {
                let (a, b) = (t[0] * n as f64, t[1] * n as f64);
                let first = a.floor().clamp(0.0, (n - 1) as f64) as usize;
                let last = (b.ceil() - 1.0).clamp(0.0, (n - 1) as f64) as usize;
                (first..=last.max(first)).filter_map(move |i| {
                    // Cells beyond the edges repeat the edge cell.
                    let lo = if i == 0 { a.min(0.0) } else { i as f64 };
                    let hi = if i == n - 1 {
                        b.max(n as f64)
                    } else {
                        (i + 1) as f64
                    };
                    let o = b.min(hi) - a.max(lo);
                    (o > 0.0).then_some((i, o / (b - a)))
                })
            };
            span(v, h)
                .flat_map(|(y, wy)| span(u, w).map(move |(x, wx)| wx * wy * cell(x, y)))
                .sum()
        }
    }
}
/// A built-in pattern at source point `uv`, averaged over a box of one
/// output pixel (`1/w × 1/h` of the source) centred there; black outside
/// the source, as for rasters.
fn pattern_at(p: TestPattern, uv: [f64; 2], w: usize, h: usize) -> f64 {
    if !(0.0..=1.0).contains(&uv[0]) || !(0.0..=1.0).contains(&uv[1]) {
        return 0.0;
    }
    let (hu, hv) = (0.5 / w as f64, 0.5 / h as f64);
    pattern_box(p, [uv[0] - hu, uv[0] + hu], [uv[1] - hv, uv[1] + hv], w, h)
}
fn raster_sample(r: &Raster, u: f64, v: f64, c: usize, mono: bool) -> f64 {
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return 0.0;
    }
    let x = (u * r.w as f64 - 0.5).clamp(0.0, (r.w - 1) as f64);
    let y = (v * r.h as f64 - 0.5).clamp(0.0, (r.h - 1) as f64);
    let (ix, iy) = (x.floor() as usize, y.floor() as usize);
    let (a, b) = (x - ix as f64, y - iy as f64);
    let value = |x: usize, y: usize| {
        let k = (y * r.w + x) * r.c;
        if r.c == 1 {
            r.data[k]
        } else if mono {
            0.2126 * r.data[k] + 0.7152 * r.data[k + 1] + 0.0722 * r.data[k + 2]
        } else {
            r.data[k + c]
        }
    };
    (1.0 - b) * ((1.0 - a) * value(ix, iy) + a * value((ix + 1).min(r.w - 1), iy))
        + b * ((1.0 - a) * value(ix, (iy + 1).min(r.h - 1))
            + a * value((ix + 1).min(r.w - 1), (iy + 1).min(r.h - 1)))
}
/// Grid cell `(ix, iy)` of field coordinates `uv` (0…1 across the grid)
/// and the fractions `(a, b)` within it, clamped to the edge cells.
fn cell(n: usize, uv: [f64; 2]) -> (usize, usize, f64, f64) {
    let x = uv[0].clamp(0.0, 1.0) * (n - 1) as f64;
    let y = uv[1].clamp(0.0, 1.0) * (n - 1) as f64;
    let ix = (x.floor() as usize).min(n - 2);
    let iy = (y.floor() as usize).min(n - 2);
    (ix, iy, x - ix as f64, y - iy as f64)
}
/// Bilinear weights of the four grid nodes around `uv`.
fn weights(n: usize, uv: [f64; 2]) -> [(usize, f64); 4] {
    let (ix, iy, a, b) = cell(n, uv);
    let k = iy * n + ix;
    [
        (k, (1.0 - a) * (1.0 - b)),
        (k + 1, a * (1.0 - b)),
        (k + n, (1.0 - a) * b),
        (k + n + 1, a * b),
    ]
}
fn map(grid: &PsfGrid, uv: [f64; 2], real: bool) -> [f64; 2] {
    let mut out = [0.0; 2];
    for (k, w) in weights(grid.size, uv) {
        let m = &grid.mapping[k];
        let p = if real { m.real_mm } else { m.ideal_mm };
        out[0] += w * p[0];
        out[1] += w * p[1];
    }
    out
}
/// Invert the interpolated real mapping by Newton's method. `None` when the
/// target lies outside the mapped field or the mapping is singular: no
/// source pixel images there, and the pixel stays black.
pub fn inverse_field(grid: &PsfGrid, target: [f64; 2]) -> Option<[f64; 2]> {
    let n = grid.size;
    let lo = grid.mapping[0].ideal_mm;
    let hi = grid.mapping[n * n - 1].ideal_mm;
    let mut uv = [0, 1].map(|j| ((target[j] - lo[j]) / (hi[j] - lo[j])).clamp(0.0, 1.0));
    for _ in 0..16 {
        let p = map(grid, uv, true);
        let r = [p[0] - target[0], p[1] - target[1]];
        if r[0].hypot(r[1]) < 1e-9 {
            return Some(uv);
        }
        let (ix, iy, a, b) = cell(n, uv);
        let k = iy * n + ix;
        let q = [k, k + 1, k + n, k + n + 1].map(|k| grid.mapping[k].real_mm);
        let s = (n - 1) as f64;
        // ∂p/∂u and ∂p/∂v of the bilinear cell.
        let du = [0, 1].map(|j| s * ((1.0 - b) * (q[1][j] - q[0][j]) + b * (q[3][j] - q[2][j])));
        let dv = [0, 1].map(|j| s * ((1.0 - a) * (q[2][j] - q[0][j]) + a * (q[3][j] - q[1][j])));
        let det = du[0] * dv[1] - dv[0] * du[1];
        if det.abs() < 1e-20 {
            return None;
        }
        uv[0] -= (dv[1] * r[0] - dv[0] * r[1]) / det;
        uv[1] -= (du[0] * r[1] - du[1] * r[0]) / det;
        if uv.iter().any(|x| !(-1e-9..=1.0 + 1e-9).contains(x)) {
            return None;
        }
        uv = uv.map(|x| x.clamp(0.0, 1.0));
    }
    None
}
/// Half-width (output pixels) of a kernel that holds every node's PSF and
/// the pixel aperture.
fn kernel_radius(grid: &PsfGrid, pitch: [f64; 2], aperture: Option<[f64; 2]>) -> usize {
    let radius = grid.psf_size as f64 * 0.5 * grid.spacing_mm
        + aperture.map_or(0.0, |a| a[0].max(a[1]) * 0.5);
    (radius / pitch[0].min(pitch[1])).ceil() as usize + 1
}
/// Largest total size of the output-lattice kernels shared by the tiles;
/// beyond it every tile deposits its own.
const SHARED_KERNEL_BYTES: usize = 256 << 20;
/// Deposit node `node`'s PSF, with the optional rectangular pixel aperture
/// (4×4 midpoint quadrature, adequate up to a few output pixels), bilinearly
/// onto a `(2r+1)²` output-lattice kernel, conserving discrete energy.
fn kernel(
    grid: &PsfGrid,
    node: usize,
    pitch: [f64; 2],
    aperture: Option<[f64; 2]>,
    r: usize,
) -> Vec<f64> {
    let n = 2 * r + 1;
    let mut out = vec![0.0; n * n];
    let (count, ap) = aperture.map_or((1, [0.0; 2]), |a| (4, a));
    let m = grid.psf_size;
    let half = (m / 2) as f64;
    // Kernel coordinate of PSF sample `i`, aperture sub-sample `s`, axis `j`.
    let at = |i: usize, s: usize, j: usize| {
        ((i as f64 - half) * grid.spacing_mm + ap[j] * ((s as f64 + 0.5) / count as f64 - 0.5))
            / pitch[j]
            + r as f64
    };
    for y in 0..m {
        for x in 0..m {
            let v = grid.kernels[node][y * m + x] / (count * count) as f64;
            for sy in 0..count {
                for sx in 0..count {
                    let (px, py) = (at(x, sx, 0), at(y, sy, 1));
                    let (ix, iy) = (px.floor() as usize, py.floor() as usize);
                    let (a, b) = (px - ix as f64, py - iy as f64);
                    out[iy * n + ix] += v * (1.0 - a) * (1.0 - b);
                    out[iy * n + ix + 1] += v * a * (1.0 - b);
                    out[(iy + 1) * n + ix] += v * (1.0 - a) * b;
                    out[(iy + 1) * n + ix + 1] += v * a * b;
                }
            }
        }
    }
    out
}
/// Linear convolution of `w × h` planes with one `k × k` kernel (centred),
/// zero padded so nothing wraps.
fn convolve(
    planes: &[Vec<f64>],
    w: usize,
    h: usize,
    kernel: &[f64],
    k: usize,
    ctx: &JobContext,
) -> MkResult<Vec<Vec<f64>>> {
    let n = next_fast_len(w.max(h) + k - 1);
    let mut b = vec![Complex64::default(); n * n];
    for y in 0..k {
        for x in 0..k {
            b[y * n + x].re = kernel[y * k + x];
        }
    }
    fft2(&mut b, n, Dir::Forward, k, ctx)?;
    planes
        .iter()
        .map(|data| {
            let mut a = vec![Complex64::default(); n * n];
            for y in 0..h {
                for x in 0..w {
                    a[y * n + x].re = data[y * w + x];
                }
            }
            fft2(&mut a, n, Dir::Forward, h, ctx)?;
            for (x, y) in a.iter_mut().zip(&b) {
                *x *= y;
            }
            fft2(&mut a, n, Dir::Inverse, n, ctx)?;
            let scale = 1.0 / (n * n) as f64;
            Ok((0..h)
                .flat_map(|y| (0..w).map(move |x| (y, x)))
                .map(|(y, x)| a[(y + k / 2) * n + x + k / 2].re * scale)
                .collect())
        })
        .collect()
}
fn normal(rng: &mut Philox) -> f64 {
    (-2.0 * rng.uniform().ln()).sqrt() * (std::f64::consts::TAU * rng.uniform()).cos()
}
fn poisson(mean: f64, rng: &mut Philox) -> f64 {
    if mean < 30.0 {
        let limit = (-mean).exp();
        let mut product = 1.0;
        let mut n = 0;
        loop {
            product *= rng.uniform();
            if product <= limit {
                return n as f64;
            }
            n += 1;
        }
    }
    // Transformed rejection (Hörmann, 1993, DOI 10.1016/0167-6687(93)90998-4).
    let b = 0.931 + 2.53 * mean.sqrt();
    let a = -0.059 + 0.02483 * b;
    let inv = 1.1239 + 1.1328 / (b - 3.4);
    let vr = 0.9277 - 3.6224 / (b - 2.0);
    loop {
        let u = rng.uniform() - 0.5;
        let v = rng.uniform();
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + mean + 0.43).floor();
        if us >= 0.07 && v <= vr {
            return k;
        }
        if k < 0.0 || (us < 0.013 && v > us) {
            continue;
        }
        let log_factorial = if k < 10.0 {
            (2..=k as usize).map(|v| (v as f64).ln()).sum()
        } else {
            (k + 0.5) * k.ln() - k + 0.9189385332046727 + 1.0 / (12.0 * k)
                - 1.0 / (360.0 * k.powi(3))
        };
        if (v * inv / (a / (us * us) + b)).ln() <= -mean + k * mean.ln() - log_factorial {
            return k;
        }
    }
}

pub fn image_simulation_job(
    t: &SequentialTracer,
    s: &ImageSimulationSettings,
    grid: Option<Arc<PsfGrid>>,
    ctx: &JobContext,
) -> MkResult<ImageSimulationResult> {
    let timer = RunTimer::start();
    if s.size.iter().any(|n| !(2..=4096).contains(n))
        || !(4..=512).contains(&s.tile_size)
        || !(s.read_noise_sigma >= 0.0 && s.read_noise_sigma.is_finite())
        || s.full_well_electrons
            .is_some_and(|x| !(x > 0.0 && x <= 1e12))
        || s.detector_aperture_mm
            .is_some_and(|a| a.iter().any(|v| !(*v > 0.0 && v.is_finite())))
    {
        return Err(invalid("invalid image size, tile size, detector or noise"));
    }
    let grid = match grid {
        Some(g) => g,
        None => Arc::new(build_psf_grid(t, &s.optics, &ctx.subrange(0.0, 0.65))?),
    };
    let scene = source(s)?;
    ctx.report(0.65, "warp source");
    let mut lo = [f64::INFINITY; 2];
    let mut hi = [f64::NEG_INFINITY; 2];
    for p in &grid.mapping {
        for j in 0..2 {
            lo[j] = lo[j].min(p.ideal_mm[j]);
            hi[j] = hi[j].max(p.ideal_mm[j]);
        }
    }
    let [w, h] = s.size;
    let c = if s.monochrome { 1 } else { 3 };
    let pitch = [(hi[0] - lo[0]) / w as f64, (hi[1] - lo[1]) / h as f64];
    if pitch.iter().any(|x| !(*x > 0.0)) {
        return Err(invalid("image field has non-positive first-order extent"));
    }
    if s.detector_aperture_mm
        .is_some_and(|a| a[0] > 4.0 * pitch[0] || a[1] > 4.0 * pitch[1])
    {
        return Err(invalid("pixel aperture exceeds four output pixels"));
    }
    // Bound FFT halo storage at the user-input boundary.
    if grid.spacing_mm * grid.psf_size as f64 / pitch[0].min(pitch[1]) > 1024.0 {
        return Err(invalid(
            "PSF exceeds 1024 output pixels; enlarge field or reduce PSF window",
        ));
    }
    // Each pixel's field coordinates and relatively illuminated source value.
    let mut uvs = vec![None; w * h];
    let mut warped = vec![0.0; w * h * c];
    warped
        .par_chunks_mut(w * c)
        .zip(uvs.par_chunks_mut(w))
        .enumerate()
        .try_for_each(|(y, (row, uv_row))| -> MkResult<()> {
            ctx.check()?;
            for x in 0..w {
                let target = [
                    lo[0] + (x as f64 + 0.5) * pitch[0],
                    lo[1] + (y as f64 + 0.5) * pitch[1],
                ];
                let Some(uv) = inverse_field(&grid, target) else {
                    continue;
                };
                uv_row[x] = Some(uv);
                let ideal = map(&grid, uv, false);
                let src = [0, 1].map(|j| (ideal[j] - lo[j]) / (hi[j] - lo[j]));
                let ri: f64 = weights(grid.size, uv)
                    .iter()
                    .map(|&(k, v)| v * grid.mapping[k].relative_illumination)
                    .sum();
                match &scene {
                    Scene::Raster(r) => {
                        for ch in 0..c {
                            row[x * c + ch] =
                                raster_sample(r, src[0], src[1], ch, s.monochrome) * ri;
                        }
                    }
                    Scene::Pattern(p) => {
                        let v = pattern_at(*p, src, w, h) * ri;
                        row[x * c..(x + 1) * c].fill(v);
                    }
                }
            }
            Ok(())
        })?;
    ctx.report(0.72, "FFT tiles");
    let r = kernel_radius(&grid, pitch, s.detector_aperture_mm);
    let tile = s.tile_size;
    let (nx, ny) = (w.div_ceil(tile), h.div_ceil(tile));
    // Tiles report their own progress; the FFTs only check cancellation.
    let quiet = JobContext::with(ctx.cancel.clone(), Progress::default());
    let finished = AtomicUsize::new(0);
    // Each node's output-lattice kernel is deposited once, by the first tile
    // that uses it.
    let shared = if grid.kernels.len() * (2 * r + 1).pow(2) * 8 <= SHARED_KERNEL_BYTES {
        grid.kernels.len()
    } else {
        0
    };
    let kernels: Vec<OnceLock<Vec<f64>>> = (0..shared).map(|_| OnceLock::new()).collect();
    let tiles = (0..nx * ny)
        .into_par_iter()
        .map(|i| -> MkResult<_> {
            ctx.check()?;
            let (x0, y0) = ((i % nx) * tile, (i / nx) * tile);
            let (tw, th) = (tile.min(w - x0), tile.min(h - y0));
            let (hw, hh) = (tw + 2 * r, th + 2 * r);
            let sources: Vec<Option<usize>> = (0..hw * hh)
                .map(|j| {
                    let sx = (x0 + j % hw).checked_sub(r)?;
                    let sy = (y0 + j / hw).checked_sub(r)?;
                    (sx < w && sy < h).then_some(sy * w + sx)
                })
                .collect();
            // Bilinear weight planes of the PSF nodes the haloed sources use.
            let mut nodes: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
            for (j, p) in sources.iter().enumerate() {
                let Some(uv) = p.and_then(|p| uvs[p]) else {
                    continue;
                };
                for (node, wt) in weights(grid.size, uv) {
                    if wt > 0.0 {
                        nodes.entry(node).or_insert_with(|| vec![0.0; hw * hh])[j] = wt;
                    }
                }
            }
            let mut out = vec![0.0; tw * th * c];
            for (node, wts) in nodes {
                let planes: Vec<Vec<f64>> = (0..c)
                    .map(|ch| {
                        sources
                            .iter()
                            .zip(&wts)
                            .map(|(p, wt)| p.map_or(0.0, |p| wt * warped[p * c + ch]))
                            .collect()
                    })
                    .collect();
                let deposit = || kernel(&grid, node, pitch, s.detector_aperture_mm, r);
                let own;
                let kernel = match kernels.get(node) {
                    Some(k) => k.get_or_init(deposit),
                    None => {
                        own = deposit();
                        &own
                    }
                };
                let blurred = convolve(&planes, hw, hh, kernel, 2 * r + 1, &quiet)?;
                for (ch, b) in blurred.iter().enumerate() {
                    for y in 0..th {
                        for x in 0..tw {
                            out[(y * tw + x) * c + ch] += b[(y + r) * hw + x + r];
                        }
                    }
                }
            }
            // FFT round-off can leave tiny negative values.
            out.iter_mut().for_each(|v| *v = v.max(0.0));
            let done = finished.fetch_add(1, Ordering::Relaxed) + 1;
            ctx.report(0.72 + 0.25 * done as f64 / (nx * ny) as f64, "FFT tiles");
            Ok((x0, y0, tw, th, out))
        })
        .collect::<MkResult<Vec<_>>>()?;
    let mut image = vec![0.0; w * h * c];
    for (x0, y0, tw, th, data) in tiles {
        for y in 0..th {
            image[((y0 + y) * w + x0) * c..((y0 + y) * w + x0 + tw) * c]
                .copy_from_slice(&data[y * tw * c..(y + 1) * tw * c]);
        }
    }
    let noisy = s.full_well_electrons.is_some() || s.read_noise_sigma > 0.0;
    image
        .par_chunks_mut(w * c)
        .enumerate()
        .filter(|_| noisy)
        .try_for_each(|(y, row)| -> MkResult<()> {
            ctx.check()?;
            // One counter-based stream per sample: independent of threads.
            for (i, v) in row.iter_mut().enumerate() {
                let mut rng = Philox::new(s.seed, (y * w * c + i) as u64);
                if let Some(well) = s.full_well_electrons {
                    *v = poisson(*v * well, &mut rng) / well;
                }
                if s.read_noise_sigma > 0.0 {
                    *v += s.read_noise_sigma * normal(&mut rng);
                }
            }
            Ok(())
        })?;
    ctx.report(1.0, "image simulation");
    let mut run_info =
        crate::analysis::run_info(timer, t, "image_simulation", s, grid.warnings.clone());
    run_info.seed = Some(s.seed);
    Ok(ImageSimulationResult {
        size: s.size,
        channels: c,
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

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::{
        ApertureSpec, Field, MaterialRef, OpticalSystem, Surface, SurfaceShape, Wavelength,
        WavelengthSet,
    };
    fn ideal() -> SequentialTracer {
        let mut s = OpticalSystem::new_empty();
        s.aperture = ApertureSpec::EntrancePupilDiameter { value: 5.0 };
        s.fields.fields = vec![Field::new(0.0, 0.0), Field::new(0.0, 0.05)];
        s.wavelengths = WavelengthSet {
            wavelengths: vec![Wavelength {
                um: 0.55,
                weight: 1.0,
            }],
            primary: 0,
        };
        s.surfaces = vec![
            Surface::new(0, SurfaceShape::Plane, f64::INFINITY, MaterialRef::Air),
            Surface::new(
                1,
                SurfaceShape::Paraxial {
                    focal_length: 100.0,
                },
                100.0,
                MaterialRef::Air,
            ),
            Surface::new(2, SurfaceShape::Plane, 0.0, MaterialRef::Air),
        ];
        s.stop = 1;
        mklight_analysis::tracer_for(s).unwrap()
    }
    #[test]
    fn built_in_pixels_integrate_area() {
        for n in [1, 7, 31, 64] {
            let mean = (0..n)
                .flat_map(|y| (0..n).map(move |x| pattern_value(TestPattern::Grid, x, y, n, n)))
                .sum::<f64>()
                / (n * n) as f64;
            assert!((mean - 0.81).abs() < 1e-12);
            let edge = (0..n)
                .flat_map(|y| {
                    (0..n).map(move |x| pattern_value(TestPattern::SlantedEdge, x, y, n, n))
                })
                .sum::<f64>()
                / (n * n) as f64;
            assert!((edge - 0.5).abs() < 1e-10);
        }
        // The central pixel straddles both sides of the edge; point sampling
        // would return black. The star also contains both colours in one pixel.
        assert!((pattern_value(TestPattern::SlantedEdge, 3, 3, 7, 7) - 0.5).abs() < 1e-12);
        assert!((pattern_value(TestPattern::SiemensStar, 0, 0, 1, 1) - 0.5).abs() < 0.04);
    }
    /// The warp evaluates patterns over a pixel box at any source point: on
    /// the grid it is the pixel's area mean, and half a pixel off it is the
    /// mean of the two pixels it straddles (exactly, for the linear-in-area
    /// patterns), not a bilinear blend of a raster.
    #[test]
    fn patterns_are_sampled_off_the_grid() {
        let n = 24;
        for p in [
            TestPattern::Grid,
            TestPattern::SiemensStar,
            TestPattern::SlantedEdge,
            TestPattern::TextChart,
        ] {
            for (x, y) in [(0, 0), (5, 17), (11, 12), (23, 9)] {
                let centre = [(x as f64 + 0.5) / n as f64, (y as f64 + 0.5) / n as f64];
                let d = pattern_at(p, centre, n, n) - pattern_value(p, x, y, n, n);
                assert!(d.abs() < 1e-12, "{p:?} ({x}, {y}): {d}");
            }
        }
        for p in [
            TestPattern::Grid,
            TestPattern::SlantedEdge,
            TestPattern::TextChart,
        ] {
            for (x, y) in [(3, 4), (11, 12), (17, 2)] {
                let between = [(x as f64 + 1.0) / n as f64, (y as f64 + 0.5) / n as f64];
                // The box covers the right half of pixel x and the left half
                // of pixel x + 1: half-pixels of a grid twice as fine in x
                // (text chart cells are uniform, so whole cells there).
                let mean = if let TestPattern::TextChart = p {
                    0.5 * (pattern_value(p, x, y, n, n) + pattern_value(p, x + 1, y, n, n))
                } else {
                    0.5 * (pattern_value(p, 2 * x + 1, y, 2 * n, n)
                        + pattern_value(p, 2 * x + 2, y, 2 * n, n))
                };
                let d = pattern_at(p, between, n, n) - mean;
                assert!(d.abs() < 1e-12, "{p:?} ({x}, {y}): {d}");
            }
        }
        // The star's spokes stay sharp at any resolution: far from the
        // centre a pixel inside a spoke is pure black or white.
        let spoke = [
            0.5 + 0.4 * (0.25f64 * std::f64::consts::PI / 24.0).cos(),
            0.5,
        ];
        assert_eq!(pattern_at(TestPattern::SiemensStar, spoke, 4096, 4096), 1.0);
    }

    #[test]
    fn off_axis_patch_maps_chiefs_and_illumination_at_its_own_fields() {
        let t = ideal();
        let s = ImageGridSettings {
            field_center: [0.2, 0.5],
            field_extent: 0.1,
            psf_size: 65,
            pupil: PupilSettings {
                grid: 32,
                rays: 33,
                ..Default::default()
            },
            ..Default::default()
        };
        let grid = build_psf_grid(&t, &s, &JobContext::new()).unwrap();
        for (k, m) in grid.mapping.iter().enumerate() {
            let expected = [0.1 + 0.1 * (k % 3) as f64, 0.4 + 0.1 * (k / 3) as f64];
            assert!((m.field[0] - expected[0]).hypot(m.field[1] - expected[1]) < 1e-14);
            let f = t
                .prepare_field(FieldPoint::normalized(m.field[0], m.field[1]))
                .unwrap();
            let c = t.chief(&f, 0).ray.pos;
            assert!((m.real_mm[0] - c.x).hypot(m.real_mm[1] - c.y) < 1e-12);
            let mut warnings = Vec::new();
            let (row, axial) = illumination_rows(
                &t,
                &[m.field],
                0,
                s.illumination_radial,
                s.illumination_azimuth,
                &mut warnings,
                &JobContext::new(),
            )
            .unwrap();
            assert!((m.relative_illumination - row[0][0] / axial).abs() < 1e-12);
        }
        let origin = grid.mapping[0].real_mm;
        let uv = inverse_field(&grid, grid.mapping[4].real_mm).unwrap();
        assert!((uv[0] - 0.5).hypot(uv[1] - 0.5) < 1e-7, "{uv:?}");
        assert!(origin[0] > 0.0 && origin[1] > 0.0);
    }

    #[test]
    fn fft_halo_convolution_matches_direct_sum() {
        let (w, h, k) = (17, 13, 5);
        let data: Vec<_> = (0..w * h).map(|i| (i as f64 * 0.37).sin()).collect();
        let kernel: Vec<_> = (0..k * k).map(|i| 1.0 / (i + 1) as f64).collect();
        let out = convolve(
            std::slice::from_ref(&data),
            w,
            h,
            &kernel,
            k,
            &JobContext::new(),
        )
        .unwrap();
        let out = &out[0];
        for y in 0..h {
            for x in 0..w {
                let mut expected = 0.0;
                for j in 0..k {
                    for i in 0..k {
                        let xx = x as isize + k as isize / 2 - i as isize;
                        let yy = y as isize + k as isize / 2 - j as isize;
                        if xx >= 0 && yy >= 0 && xx < w as isize && yy < h as isize {
                            expected += kernel[j * k + i] * data[yy as usize * w + xx as usize];
                        }
                    }
                }
                assert!((out[y * w + x] - expected).abs() < 1e-12);
            }
        }
    }
    #[test]
    fn ideal_lens_impulse_matches_airy_and_direct_image_convolution() {
        let t = ideal();
        let dx = 0.00275;
        let settings = ImageGridSettings {
            pupil: PupilSettings {
                grid: 64,
                rays: 65,
                amplitude: AmplitudeModel::Uniform,
                ..Default::default()
            },
            psf_size: 33,
            psf_spacing_mm: Some(dx),
            ..Default::default()
        };
        let grid = Arc::new(build_psf_grid(&t, &settings, &JobContext::new()).unwrap());
        let p = &grid.kernels[4];
        let center = 16;
        let peak = p[center * 33 + center];
        for i in 0..12 {
            let exact = crate::special::annular_airy(
                std::f64::consts::PI * i as f64 * dx / (0.55e-3 * 20.0),
                0.0,
            );
            assert!((p[center * 33 + center + i] / peak - exact).abs() < 1e-3);
        }
        let n = 65;
        let mut data = vec![0.0; n * n];
        data[(n / 2) * n + n / 2] = 1.0;
        let s = ImageSimulationSettings {
            source: ImageSource::Buffer {
                width: n,
                height: n,
                channels: 1,
                data,
            },
            size: [n, n],
            optics: settings,
            tile_size: 128,
            ..Default::default()
        };
        let result = image_simulation_job(&t, &s, Some(grid.clone()), &JobContext::new()).unwrap();
        let r = kernel_radius(&grid, result.pixel_size_mm, None);
        let (k, ks) = (kernel(&grid, 4, result.pixel_size_mm, None, r), 2 * r + 1);
        for y in 0..ks {
            for x in 0..ks {
                let image = result.image[(n / 2 + y - ks / 2) * n + n / 2 + x - ks / 2];
                assert!(
                    (image - k[y * ks + x]).abs() < 2e-5,
                    "{image} {}",
                    k[y * ks + x]
                );
            }
        }
        assert!((result.image.iter().sum::<f64>() - 1.0).abs() < 1e-4);
    }
    #[test]
    fn distortion_maps_grid_point_to_predicted_chief_position() {
        let t = mklight_analysis::tracer_for(mklight_core::samples::cooke_triplet()).unwrap();
        let s = ImageGridSettings {
            field_extent: 0.5,
            pupil: PupilSettings {
                grid: 16,
                rays: 17,
                ..Default::default()
            },
            psf_size: 17,
            ..Default::default()
        };
        let grid = build_psf_grid(&t, &s, &JobContext::new()).unwrap();
        let point = &grid.mapping[5];
        let uv = inverse_field(&grid, point.real_mm).unwrap();
        let (k, w) = weights(grid.size, uv)
            .into_iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap();
        assert_eq!(k, 5);
        assert!((w - 1.0).abs() < 1e-7);
        let f = t
            .prepare_field(FieldPoint::normalized(point.field[0], point.field[1]))
            .unwrap();
        let chief = t.chief(&f, t.system().primary_wavelength);
        assert!(
            (chief.ray.pos.x - point.real_mm[0]).abs() < 1e-12
                && (chief.ray.pos.y - point.real_mm[1]).abs() < 1e-12
        );
        assert!((point.real_mm[0] - point.ideal_mm[0]).abs() > 1e-6);

        // Isolate geometric mapping with delta PSFs and a sampled grid mark.
        // The chief geometry still comes from the real optical system.
        let mut geometry = grid.clone();
        for kernel in &mut geometry.kernels {
            kernel.fill(0.0);
            kernel[(geometry.psf_size / 2) * geometry.psf_size + geometry.psf_size / 2] = 1.0;
        }
        for p in &mut geometry.mapping {
            p.relative_illumination = 1.0;
        }
        let uv = [0.75, 0.5];
        let ideal = map(&geometry, uv, false);
        let predicted = map(&geometry, uv, true);
        let lo = geometry.mapping[0].ideal_mm;
        let hi = geometry.mapping.last().unwrap().ideal_mm;
        let n = 129;
        let pitch = [(hi[0] - lo[0]) / n as f64, (hi[1] - lo[1]) / n as f64];
        let data = (0..n)
            .flat_map(|y| {
                (0..n).map(move |x| {
                    let dx = (lo[0] + (x as f64 + 0.5) * pitch[0] - ideal[0]) / pitch[0];
                    let dy = (lo[1] + (y as f64 + 0.5) * pitch[1] - ideal[1]) / pitch[1];
                    (-0.5 * (dx * dx + dy * dy) / 4.0).exp()
                })
            })
            .collect();
        let image = image_simulation_job(
            &t,
            &ImageSimulationSettings {
                source: ImageSource::Buffer {
                    width: n,
                    height: n,
                    channels: 1,
                    data,
                },
                size: [n, n],
                optics: s,
                ..Default::default()
            },
            Some(Arc::new(geometry)),
            &JobContext::new(),
        )
        .unwrap();
        let energy: f64 = image.image.iter().sum();
        let centroid: [f64; 2] = std::array::from_fn(|axis| {
            image
                .image
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    v * (lo[axis]
                        + (if axis == 0 {
                            (i % n) as f64
                        } else {
                            (i / n) as f64
                        } + 0.5)
                            * pitch[axis])
                })
                .sum::<f64>()
                / energy
        });
        for axis in 0..2 {
            assert!(
                (centroid[axis] - predicted[axis]).abs() < 0.02 * pitch[axis],
                "{centroid:?} vs {predicted:?}"
            );
        }
    }
    #[test]
    fn shot_and_read_noise_have_known_moments_and_repeat() {
        let mut rng = Philox::new(7, 0);
        let n = 100_000;
        let mut sum = 0.0;
        let mut sum2 = 0.0;
        for _ in 0..n {
            let x = poisson(50.0, &mut rng);
            sum += x;
            sum2 += x * x;
        }
        let mean = sum / n as f64;
        assert!((mean - 50.0).abs() < 0.15);
        assert!((sum2 / n as f64 - mean * mean - 50.0).abs() < 0.7);
        let mut a = Philox::new(5, 9);
        let mut b = Philox::new(5, 9);
        assert_eq!(normal(&mut a), normal(&mut b));
    }
    fn cooke_grid(
        reference: crate::ReferencePoint,
    ) -> (SequentialTracer, ImageGridSettings, Arc<PsfGrid>) {
        let t = mklight_analysis::tracer_for(mklight_core::samples::cooke_triplet()).unwrap();
        let s = ImageGridSettings {
            field_extent: 1.0,
            pupil: PupilSettings {
                grid: 32,
                rays: 33,
                reference,
                ..Default::default()
            },
            psf_size: 33,
            ..Default::default()
        };
        let grid = Arc::new(build_psf_grid(&t, &s, &JobContext::new()).unwrap());
        (t, s, grid)
    }
    /// Each source pixel spreads with its own interpolated PSF, so the tile
    /// size only partitions the work. Tile-constant PSFs used to leave 2%
    /// steps at tile edges of a Siemens star on the Cooke triplet.
    #[test]
    fn tiles_are_seamless() {
        let (t, optics, grid) = cooke_grid(Default::default());
        let run = |tile_size| {
            let s = ImageSimulationSettings {
                source: ImageSource::Pattern {
                    pattern: TestPattern::SiemensStar,
                },
                size: [96, 96],
                optics: optics.clone(),
                tile_size,
                ..Default::default()
            };
            image_simulation_job(&t, &s, Some(grid.clone()), &JobContext::new())
                .unwrap()
                .image
        };
        let reference = run(128);
        for tile in [4, 16, 40] {
            let d = run(tile)
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f64::max);
            assert!(d < 1e-12, "tile {tile}: {d}");
        }
    }
    /// PSFs are centred on their reference point (`FieldPupils`), so that is
    /// where the mapping must put them. A centroid-referenced PSF used to be
    /// placed at the chief ray, 1.5 µm off at the Cooke triplet's corner.
    #[test]
    fn psf_centre_is_its_reference_point() {
        let (t, s, grid) = cooke_grid(crate::ReferencePoint::Centroid);
        let m = &grid.mapping[8];
        let f = t
            .prepare_field(FieldPoint::normalized(s.field_extent, s.field_extent))
            .unwrap();
        let wls = wavelengths(&t, &Default::default()).unwrap();
        let pupils = field_pupils(&t, &f, "", &wls, &s.pupil, None, &JobContext::new()).unwrap();
        assert_eq!(m.real_mm, pupils.reference_mm);
        let chief = t.chief(&f, t.system().primary_wavelength).ray.pos;
        assert!((m.real_mm[1] - chief.y).abs() > 1e-3);
    }
    #[test]
    fn asymmetric_sources_keep_their_orientation_through_identity_mapping() {
        let t = ideal();
        let n = 108;
        // Identity image-plane mapping and delta PSFs isolate source orientation.
        let mapping = (0..2)
            .flat_map(|y| {
                (0..2).map(move |x| {
                    let point = [x as f64, y as f64];
                    FieldMapping {
                        pupil_sampling: Vec::new(),
                        field: point,
                        ideal_mm: point,
                        real_mm: point,
                        relative_illumination: 1.0,
                    }
                })
            })
            .collect();
        let grid = Arc::new(PsfGrid {
            mapping,
            size: 2,
            psf_size: 3,
            spacing_mm: 1.0 / n as f64,
            kernels: vec![vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]; 4],
            warnings: Vec::new(),
        });
        let run = |source| {
            image_simulation_job(
                &t,
                &ImageSimulationSettings {
                    source,
                    size: [n, n],
                    ..Default::default()
                },
                Some(grid.clone()),
                &JobContext::new(),
            )
            .unwrap()
            .image
        };
        let text = run(ImageSource::Pattern {
            pattern: TestPattern::TextChart,
        });
        // The L at the top has a vertical stem and a bottom horizontal bar.
        assert!((text[(n - 1) * n + 52] - 1.0).abs() < 1e-12);
        assert!(text[(n - 1 - 24) * n + 52].abs() < 1e-12);

        let path =
            std::env::temp_dir().join(format!("mklight-orientation-{}.png", std::process::id()));
        let mut top_down = vec![0u8; n * n];
        top_down[7] = 255;
        {
            let mut encoder =
                png::Encoder::new(std::fs::File::create(&path).unwrap(), n as u32, n as u32);
            encoder.set_color(png::ColorType::Grayscale);
            encoder.set_depth(png::BitDepth::Eight);
            encoder
                .write_header()
                .unwrap()
                .write_image_data(&top_down)
                .unwrap();
        }
        let png = run(ImageSource::Path {
            path: path.to_str().unwrap().into(),
        });
        std::fs::remove_file(path).unwrap();
        let mut bottom_up = vec![0.0; n * n];
        bottom_up[(n - 1) * n + 7] = 1.0;
        let buffer = run(ImageSource::Buffer {
            width: n,
            height: n,
            channels: 1,
            data: bottom_up,
        });
        assert!((png[(n - 1) * n + 7] - 1.0).abs() < 1e-12);
        for (a, b) in png.iter().zip(buffer) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn png_sources_decode_srgb_to_linear() {
        let path = std::env::temp_dir().join(format!("mklight-im-{}.png", std::process::id()));
        let decode = |color, depth, data: &[u8]| {
            let mut e = png::Encoder::new(std::fs::File::create(&path).unwrap(), 2, 1);
            e.set_color(color);
            e.set_depth(depth);
            e.write_header().unwrap().write_image_data(data).unwrap();
            let r = read_png(path.to_str().unwrap()).unwrap();
            std::fs::remove_file(&path).unwrap();
            r
        };
        let rgba = [0, 128, 255, 9, 255, 0, 10, 9];
        let r = decode(png::ColorType::Rgba, png::BitDepth::Eight, &rgba);
        assert_eq!((r.w, r.h, r.c), (2, 1, 3));
        let expected = [0.0, 0.2158605, 1.0, 1.0, 0.0, 0.0030353];
        for (v, e) in r.data.iter().zip(expected) {
            assert!((v - e).abs() < 1e-6, "{v} {e}");
        }
        let r = decode(
            png::ColorType::Grayscale,
            png::BitDepth::Sixteen,
            &[128, 0, 255, 255],
        );
        assert_eq!(r.c, 1);
        assert!((r.data[0] - 0.2140482).abs() < 1e-6 && r.data[1] == 1.0);
    }
}
#[cfg(test)]
mod star_resolution {
    use super::*;
    use std::f64::consts::PI;
    /// The Siemens star shows the lens's resolution: the contrast of its 24
    /// spokes on a ring of radius r is the on-axis MTF at the spokes'
    /// frequency 24 / (2πr), times the pixel's sampling (the pixel box of
    /// the source, the PSF's bilinear deposit and this test's bilinear
    /// read-out, ≈ sinc⁴), down to rings a few pixels across.
    #[test]
    fn star_contrast_follows_the_mtf() {
        let t = mklight_analysis::tracer_for(mklight_core::samples::cooke_triplet()).unwrap();
        let optics = ImageGridSettings {
            field_extent: 0.02,
            ..Default::default()
        };
        let grid = Arc::new(build_psf_grid(&t, &optics, &JobContext::new()).unwrap());
        let n = 256;
        let s = ImageSimulationSettings {
            source: ImageSource::Pattern {
                pattern: TestPattern::SiemensStar,
            },
            size: [n, n],
            optics,
            ..Default::default()
        };
        let r = image_simulation_job(&t, &s, Some(grid.clone()), &JobContext::new()).unwrap();
        let pitch = r.pixel_size_mm[0];
        // The MTF of the centre node's PSF, averaged over directions.
        let (m, k) = (grid.psf_size, &grid.kernels[grid.kernels.len() / 2]);
        let total: f64 = k.iter().sum();
        let mtf = |f: f64| {
            (0..8)
                .map(|d| {
                    let a = d as f64 * PI / 8.0;
                    let (mut re, mut im) = (0.0, 0.0);
                    for (i, v) in k.iter().enumerate() {
                        let (x, y) = (
                            (i % m) as f64 - (m / 2) as f64,
                            (i / m) as f64 - (m / 2) as f64,
                        );
                        let phase = 2.0 * PI * f * grid.spacing_mm * (x * a.cos() + y * a.sin());
                        re += v * phase.cos();
                        im += v * phase.sin();
                    }
                    (re * re + im * im).sqrt() / total
                })
                .sum::<f64>()
                / 8.0
        };
        let at = |x: f64, y: f64| {
            let (ix, iy) = (x.floor() as usize, y.floor() as usize);
            let (a, b) = (x - ix as f64, y - iy as f64);
            let v = |x: usize, y: usize| r.image[y * n + x];
            (1.0 - b) * ((1.0 - a) * v(ix, iy) + a * v(ix + 1, iy))
                + b * ((1.0 - a) * v(ix, iy + 1) + a * v(ix + 1, iy + 1))
        };
        let sinc = |x: f64| (PI * x).sin() / (PI * x);
        for radius in [12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0] {
            // The 24th azimuthal harmonic relative to the mean; a 0/1
            // square wave's fundamental is 4/π of its mean.
            let samples = 2048;
            let (mut re, mut im, mut mean) = (0.0, 0.0, 0.0);
            for j in 0..samples {
                let th = 2.0 * PI * j as f64 / samples as f64;
                let v = at(
                    n as f64 / 2.0 + radius * th.cos() - 0.5,
                    n as f64 / 2.0 + radius * th.sin() - 0.5,
                );
                re += v * (24.0 * th).cos();
                im += v * (24.0 * th).sin();
                mean += v;
            }
            let contrast = 2.0 * (re * re + im * im).sqrt() / mean / (4.0 / PI);
            let f = 24.0 / (2.0 * PI * radius * pitch);
            let predicted = mtf(f) * sinc(f * pitch).powi(4);
            assert!(
                (contrast - predicted).abs() < 0.03,
                "r {radius} px, {f:.0} c/mm: contrast {contrast:.3}, MTF × sampling {predicted:.3}"
            );
        }
    }
}
