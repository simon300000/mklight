//! Space-varying incoherent image simulation. Each output tile uses a
//! bilinearly interpolated, polychromatic PSF from a cached field grid.
//! Inverse chief-ray mapping warps the source; projected solid angle models
//! relative illumination. Linear FFT convolution uses halos (no wraparound).
use crate::analysis::{field_pupils, wavelengths};
use crate::fft::{Dir, fft2, next_fast_len};
use crate::{AmplitudeModel, PolyPsfSettings, PsfNormalization, PsfTransform, PupilSettings};
use mklight_analysis::curvature::{GridDistortionSettings, grid_distortion_job};
use mklight_analysis::field_maps::illumination_samples;
use mklight_core::job::RunTimer;
use mklight_core::rng::Philox;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::{FieldPoint, SequentialTracer};
use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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
    /// Linear-light samples, row-major interleaved; one or three channels.
    Buffer {
        width: usize,
        height: usize,
        channels: usize,
        data: Vec<f64>,
    },
    /// PNG/JPEG/PNM. RGB code values are converted from sRGB to linear light.
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
            field_extent: 1.0,
            pupil: PupilSettings {
                grid: 32,
                rays: 33,
                amplitude: AmplitudeModel::Radiometric,
                ..Default::default()
            },
            psf_size: 33,
            psf_spacing_mm: None,
            illumination_radial: 4,
            illumination_azimuth: 16,
        }
    }
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FieldMapping {
    pub field: [f64; 2],
    pub ideal_mm: [f64; 2],
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
    ctx.check()?;
    if !(2..=33).contains(&s.field_grid)
        || !(s.field_extent > 0.0 && s.field_extent.is_finite())
        || !(3..=513).contains(&s.psf_size)
        || s.psf_size % 2 == 0
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
            ..Default::default()
        },
        &ctx.subrange(0.0, 0.05),
    )?;
    let mut coordinates: Vec<_> = d.points.iter().map(|p| p.field).collect();
    coordinates.push([0.0, 0.0]);
    let wls = wavelengths(t, &Default::default())?;
    let mut irradiance = vec![0.0; coordinates.len()];
    for (i, &(wl, weight)) in wls.iter().enumerate() {
        let ri = illumination_samples(
            t,
            &coordinates,
            wl,
            s.illumination_radial,
            s.illumination_azimuth,
            &ctx.subrange(
                0.05 + 0.15 * i as f64 / wls.len() as f64,
                0.05 + 0.15 * (i + 1) as f64 / wls.len() as f64,
            ),
        )?;
        for (sum, row) in irradiance.iter_mut().zip(ri) {
            *sum += weight * row[1];
        }
    }
    let axial = *irradiance.last().unwrap();
    let mut kernels = Vec::new();
    let mut mapping = Vec::new();
    let mut warnings = d.run_info.warnings;
    let mut spacing = s.psf_spacing_mm;
    for (i, p) in d.points.iter().enumerate() {
        ctx.check()?;
        if p.real
            .iter()
            .chain(p.paraxial.iter())
            .any(|x| !x.is_finite())
        {
            return Err(invalid("field grid contains an unusable chief ray"));
        }
        let f = t.prepare_field(FieldPoint::normalized(p.field[0], p.field[1]))?;
        let sub = ctx.subrange(
            0.2 + 0.8 * i as f64 / d.points.len() as f64,
            0.2 + 0.8 * (i + 1) as f64 / d.points.len() as f64,
        );
        let pupils = field_pupils(
            t,
            &f,
            "image simulation",
            &wls,
            &s.pupil,
            &sub.subrange(0.0, 0.7),
        )?;
        warnings.extend(pupils.warnings.clone());
        let dx = *spacing.get_or_insert_with(|| pupils.min_lambda_f() * 0.5);
        let ps = PolyPsfSettings {
            grid_size: s.psf_size,
            spacing_mm: dx,
            normalization: PsfNormalization::UnitEnergy,
            transform: PsfTransform::Fft,
        };
        let psf = crate::polychromatic_psf(&pupils.weighted(), &ps, &sub.subrange(0.7, 1.0))?;
        warnings.extend(psf.warnings.iter().map(ToString::to_string));
        // Keep discrete convolution flux normalised. Truncation is explicit.
        let sum: f64 = psf.data.iter().sum();
        let energy = sum * dx * dx;
        if energy < 0.95 {
            warnings.push(format!(
                "psf_truncated: field {:?}, sampled energy {energy:.5}",
                p.field
            ));
        }
        kernels.push(psf.data.iter().map(|v| v / sum).collect());
        mapping.push(FieldMapping {
            field: p.field,
            ideal_mm: p.paraxial,
            real_mm: p.real,
            relative_illumination: irradiance[i] / axial,
        });
    }
    Ok(PsfGrid {
        mapping,
        size: s.field_grid,
        psf_size: s.psf_size,
        spacing_mm: spacing.unwrap(),
        kernels,
        warnings,
    })
}

#[derive(Clone)]
struct Raster {
    w: usize,
    h: usize,
    c: usize,
    data: Vec<f64>,
}
fn source(s: &ImageSimulationSettings) -> MkResult<Raster> {
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
            Ok(Raster {
                w: *width,
                h: *height,
                c: *channels,
                data: data.clone(),
            })
        }
        ImageSource::Path { path } => {
            let mut reader = image::ImageReader::open(path)
                .map_err(|e| MkError::Io(e.to_string()))?
                .with_guessed_format()
                .map_err(|e| MkError::Io(e.to_string()))?;
            let mut limits = image::Limits::default();
            limits.max_image_width = Some(8192);
            limits.max_image_height = Some(8192);
            reader.limits(limits);
            let img = reader
                .decode()
                .map_err(|e| invalid(&e.to_string()))?
                .to_rgb32f();
            let data = img
                .as_raw()
                .iter()
                .map(|&x| {
                    let x = x as f64;
                    if x <= 0.04045 {
                        x / 12.92
                    } else {
                        ((x + 0.055) / 1.055).powf(2.4)
                    }
                })
                .collect();
            Ok(Raster {
                w: img.width() as usize,
                h: img.height() as usize,
                c: 3,
                data,
            })
        }
        ImageSource::Pattern { pattern } => {
            let [w, h] = s.size;
            let data = (0..h)
                .flat_map(|y| (0..w).map(move |x| pattern_value(*pattern, x, y, w, h)))
                .collect();
            Ok(Raster { w, h, c: 1, data })
        }
    }
}
fn pattern_value(p: TestPattern, x: usize, y: usize, w: usize, h: usize) -> f64 {
    let (u, v) = ((x as f64 + 0.5) / w as f64, (y as f64 + 0.5) / h as f64);
    match p {
        TestPattern::Grid => {
            if (u * 8.0).fract() < 0.1 || (v * 8.0).fract() < 0.1 {
                0.0
            } else {
                1.0
            }
        }
        TestPattern::SiemensStar => {
            if ((v - 0.5).atan2(u - 0.5) * 24.0).cos() > 0.0 {
                1.0
            } else {
                0.0
            }
        }
        TestPattern::SlantedEdge => {
            if u > 0.5 + 0.1 * (v - 0.5) {
                1.0
            } else {
                0.0
            }
        }
        TestPattern::TextChart => {
            // Repeated 5x7 bitmap glyphs M K L, at three spatial scales.
            const FONT: [[u8; 7]; 3] = [
                [17, 27, 21, 17, 17, 17, 17],
                [17, 18, 20, 24, 20, 18, 17],
                [16, 16, 16, 16, 16, 16, 31],
            ];
            let band = (v * 3.0) as usize;
            let scale = [4, 2, 1][band.min(2)];
            let (gx, gy) = (x / scale, y / scale);
            let glyph = (gx / 6) % 3;
            if gy % 9 < 7 && gx % 6 < 5 && FONT[glyph][gy % 9] & (1 << (4 - gx % 6)) != 0 {
                0.0
            } else {
                1.0
            }
        }
    }
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
/// Bilinear field-grid interpolation, clamped to the edge cell.
fn weights(n: usize, uv: [f64; 2]) -> [(usize, f64); 4] {
    let x = uv[0].clamp(0.0, 1.0) * (n - 1) as f64;
    let y = uv[1].clamp(0.0, 1.0) * (n - 1) as f64;
    let ix = (x.floor() as usize).min(n - 2);
    let iy = (y.floor() as usize).min(n - 2);
    let (a, b) = (x - ix as f64, y - iy as f64);
    [
        (iy * n + ix, (1.0 - a) * (1.0 - b)),
        (iy * n + ix + 1, a * (1.0 - b)),
        ((iy + 1) * n + ix, (1.0 - a) * b),
        ((iy + 1) * n + ix + 1, a * b),
    ]
}
fn map(grid: &PsfGrid, uv: [f64; 2], real: bool) -> [f64; 2] {
    let mut out = [0.0; 2];
    for (k, w) in weights(grid.size, uv) {
        let p = if real {
            grid.mapping[k].real_mm
        } else {
            grid.mapping[k].ideal_mm
        };
        for j in 0..2 {
            out[j] += w * p[j];
        }
    }
    out
}
/// Invert the interpolated chief-ray mapping. A singular mapping
/// cannot image a unique source pixel and yields a black pixel.
pub fn inverse_field(grid: &PsfGrid, target: [f64; 2]) -> Option<[f64; 2]> {
    let lo = grid.mapping[0].ideal_mm;
    let hi = grid.mapping.last()?.ideal_mm;
    let mut uv = [
        ((target[0] - lo[0]) / (hi[0] - lo[0])).clamp(0.0, 1.0),
        ((target[1] - lo[1]) / (hi[1] - lo[1])).clamp(0.0, 1.0),
    ];
    for _ in 0..16 {
        let p = map(grid, uv, true);
        let r = [p[0] - target[0], p[1] - target[1]];
        if r[0].hypot(r[1]) < 1e-9 {
            return (uv.iter().all(|x| *x >= -1e-9 && *x <= 1.0 + 1e-9)).then_some(uv);
        }
        let n = grid.size;
        let scale = (n - 1) as f64;
        let x = uv[0] * scale;
        let y = uv[1] * scale;
        let ix = (x.floor() as usize).min(n - 2);
        let iy = (y.floor() as usize).min(n - 2);
        let (a, b) = (x - ix as f64, y - iy as f64);
        let q = [
            iy * n + ix,
            iy * n + ix + 1,
            (iy + 1) * n + ix,
            (iy + 1) * n + ix + 1,
        ]
        .map(|k| grid.mapping[k].real_mm);
        let mut j = [[0.0; 2]; 2];
        for axis in 0..2 {
            j[axis][0] =
                scale * ((1.0 - b) * (q[1][axis] - q[0][axis]) + b * (q[3][axis] - q[2][axis]));
            j[axis][1] =
                scale * ((1.0 - a) * (q[2][axis] - q[0][axis]) + a * (q[3][axis] - q[1][axis]));
        }
        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        if det.abs() < 1e-20 {
            return None;
        }
        uv[0] -= (j[1][1] * r[0] - j[0][1] * r[1]) / det;
        uv[1] -= (-j[1][0] * r[0] + j[0][0] * r[1]) / det;
        if uv.iter().any(|x| *x < -1e-9 || *x > 1.0 + 1e-9) {
            return None;
        }
        uv = uv.map(|x| x.clamp(0.0, 1.0));
    }
    None
}
/// Deposit a physical PSF and optional rectangular pixel aperture onto the
/// output lattice, conserving discrete energy. Aperture uses 4x4 midpoint
/// quadrature, adequate when its width is comparable to one output pixel.
fn kernel(
    grid: &PsfGrid,
    uv: [f64; 2],
    pitch: [f64; 2],
    aperture: Option<[f64; 2]>,
) -> (Vec<f64>, usize) {
    let radius = grid.psf_size as f64 * 0.5 * grid.spacing_mm;
    let ap = aperture.unwrap_or([0.0; 2]);
    let r = ((radius + ap[0].max(ap[1]) * 0.5) / pitch[0].min(pitch[1])).ceil() as usize + 1;
    let n = 2 * r + 1;
    let mut out = vec![0.0; n * n];
    let count = if aperture.is_some() { 4 } else { 1 };
    for (k, w) in weights(grid.size, uv) {
        for y in 0..grid.psf_size {
            for x in 0..grid.psf_size {
                let value = grid.kernels[k][y * grid.psf_size + x] * w / (count * count) as f64;
                for sy in 0..count {
                    for sx in 0..count {
                        let px = ((x as f64 - (grid.psf_size / 2) as f64) * grid.spacing_mm
                            + ap[0] * ((sx as f64 + 0.5) / count as f64 - 0.5))
                            / pitch[0]
                            + r as f64;
                        let py = ((y as f64 - (grid.psf_size / 2) as f64) * grid.spacing_mm
                            + ap[1] * ((sy as f64 + 0.5) / count as f64 - 0.5))
                            / pitch[1]
                            + r as f64;
                        let (ix, iy) = (px.floor() as usize, py.floor() as usize);
                        let (a, b) = (px - ix as f64, py - iy as f64);
                        for (dx, dy, v) in [
                            (0, 0, (1.0 - a) * (1.0 - b)),
                            (1, 0, a * (1.0 - b)),
                            (0, 1, (1.0 - a) * b),
                            (1, 1, a * b),
                        ] {
                            if ix + dx < n && iy + dy < n {
                                out[(iy + dy) * n + ix + dx] += value * v;
                            }
                        }
                    }
                }
            }
        }
    }
    (out, n)
}
/// Linear convolution of one haloed tile; zero padding prevents cyclic wrap.
fn convolve(
    data: &[f64],
    w: usize,
    h: usize,
    kernel: &[f64],
    k: usize,
    ctx: &JobContext,
) -> MkResult<Vec<f64>> {
    let n = next_fast_len(w.max(h) + k - 1);
    let mut a = vec![Complex64::default(); n * n];
    let mut b = a.clone();
    for y in 0..h {
        for x in 0..w {
            a[y * n + x].re = data[y * w + x];
        }
    }
    for y in 0..k {
        for x in 0..k {
            b[y * n + x].re = kernel[y * k + x];
        }
    }
    fft2(&mut a, n, Dir::Forward, h, &ctx.subrange(0.0, 0.3))?;
    fft2(&mut b, n, Dir::Forward, k, &ctx.subrange(0.3, 0.6))?;
    for (x, y) in a.iter_mut().zip(b) {
        *x *= y;
    }
    fft2(&mut a, n, Dir::Inverse, n, &ctx.subrange(0.6, 1.0))?;
    Ok((0..h)
        .flat_map(|y| {
            let a = &a;
            (0..w).map(move |x| a[(y + k / 2) * n + x + k / 2].re / (n * n) as f64)
        })
        .collect())
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
    ctx.check()?;
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
    let raster = source(s)?;
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
    let mut warped = vec![0.0; w * h * c];
    warped
        .par_chunks_mut(w * c)
        .enumerate()
        .try_for_each(|(y, row)| -> MkResult<()> {
            ctx.check()?;
            for x in 0..w {
                let target = [
                    lo[0] + (x as f64 + 0.5) * pitch[0],
                    lo[1] + (y as f64 + 0.5) * pitch[1],
                ];
                if let Some(uv) = inverse_field(&grid, target) {
                    let ideal = map(&grid, uv, false);
                    let src = [
                        (ideal[0] - lo[0]) / (hi[0] - lo[0]),
                        (ideal[1] - lo[1]) / (hi[1] - lo[1]),
                    ];
                    let ri: f64 = weights(grid.size, uv)
                        .iter()
                        .map(|&(k, v)| v * grid.mapping[k].relative_illumination)
                        .sum();
                    for ch in 0..c {
                        row[x * c + ch] =
                            raster_sample(&raster, src[0], src[1], ch, s.monochrome) * ri;
                    }
                }
            }
            Ok(())
        })?;
    ctx.report(0.72, "FFT tiles");
    let tile = s.tile_size;
    let nx = w.div_ceil(tile);
    let ny = h.div_ceil(tile);
    let tiles: Vec<_> = (0..nx * ny)
        .into_par_iter()
        .map(|i| -> MkResult<_> {
            ctx.check()?;
            let (x0, y0) = ((i % nx) * tile, (i / nx) * tile);
            let (tw, th) = (tile.min(w - x0), tile.min(h - y0));
            let target = [
                lo[0] + (x0 as f64 + tw as f64 * 0.5) * pitch[0],
                lo[1] + (y0 as f64 + th as f64 * 0.5) * pitch[1],
            ];
            let uv = inverse_field(&grid, target).unwrap_or([0.5, 0.5]);
            let (kernel, k) = kernel(&grid, uv, pitch, s.detector_aperture_mm);
            let halo = k / 2;
            let (hw, hh) = (tw + 2 * halo, th + 2 * halo);
            let mut out = vec![0.0; tw * th * c];
            for ch in 0..c {
                let mut data = vec![0.0; hw * hh];
                for y in 0..hh {
                    for x in 0..hw {
                        let sx = x0 as isize + x as isize - halo as isize;
                        let sy = y0 as isize + y as isize - halo as isize;
                        if sx >= 0 && sy >= 0 && sx < w as isize && sy < h as isize {
                            data[y * hw + x] = warped[(sy as usize * w + sx as usize) * c + ch];
                        }
                    }
                }
                let blurred = convolve(&data, hw, hh, &kernel, k, &ctx.subrange(0.72, 0.97))?;
                for y in 0..th {
                    for x in 0..tw {
                        out[(y * tw + x) * c + ch] = blurred[(y + halo) * hw + x + halo].max(0.0);
                    }
                }
            }
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
    image
        .par_chunks_mut(w * c)
        .enumerate()
        .try_for_each(|(y, row)| -> MkResult<()> {
            ctx.check()?;
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
    fn fft_halo_convolution_matches_direct_sum() {
        let (w, h, k) = (17, 13, 5);
        let data: Vec<_> = (0..w * h).map(|i| (i as f64 * 0.37).sin()).collect();
        let kernel: Vec<_> = (0..k * k).map(|i| 1.0 / (i + 1) as f64).collect();
        let out = convolve(&data, w, h, &kernel, k, &JobContext::new()).unwrap();
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
        let (k, ks) = kernel(&grid, [0.5, 0.5], result.pixel_size_mm, None);
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
}
