//! Real-ray ghost irradiance on the sensor.
//!
//! # Sampling
//!
//! For each selected field and wavelength, starting rays are generated once
//! on a square grid of `n × n` cell centres over the aim plane of the
//! reference tracer (the paraxial entrance-pupil plane, or object-space
//! tangents for a telecentric object), covering a disc of `overfill` times
//! the entrance-pupil radius around the chief ray's aim point. Ghost light
//! can enter outside the entrance pupil (a reflection before the stop may
//! still pass the stop afterwards), so the overfilled beam is traced and
//! the model's apertures ([`ClipMode`](crate::ClipMode)) decide what
//! survives, for the primary path exactly as for the ghosts.
//!
//! Each ray carries the flux `w_f w_λ · h²/π`: the field and wavelength
//! weights (normalised over the selection; wavelength weights include the
//! sensor's spectral response) times the fraction of the unit entrance
//! pupil its cell covers (`h` = cell size in pupil units). Flux is therefore
//! relative to the flux entering the paraxial entrance pupil, and the
//! primary flux is ≈ 1 minus Fresnel, absorption and vignetting losses.
//!
//! # Maps and ratios
//!
//! Hits are binned into an `nx × ny` map over the sensor (the model's
//! [`Sensor`](mklight_core::Sensor) rectangle, else the square of the image
//! clear semi-diameter, else an explicit size), in the image surface's local
//! `x`, `y`. Values are irradiance: flux per mm² of bin. The primary map is
//! computed by the same code from the empty path. Per path and field:
//! detected flux, peak irradiance, centroid and RMS radius; per path the
//! ghost-to-signal ratios
//!
//! * `flux_ratio` = ghost flux on the map / primary flux on the map, and
//! * `peak_ratio` = max over fields of (ghost peak / primary peak) of that
//!   field. A point source's primary peak depends on the bin size, so this
//!   ratio is meaningful for bins of about a pixel.
//!
//! The total ghost map is the sum of all path maps.
//!
//! # Determinism, parallelism, cancellation
//!
//! Rays are traced in parallel (rayon) in blocks; results are collected in
//! the fixed order `(field, wavelength, pupil)` and accumulated sequentially
//! with Neumaier compensation, so maps are bit-identical run to run. The
//! [`JobContext`] is checked before every block; progress is monotone and
//! ends at 1.
//!
//! # GPU port
//!
//! The work decomposes into independent rays indexed by
//! `(path, field, wavelength, pupil)`; starting rays depend only on
//! `(field, wavelength, pupil)` and are shared by all paths. The kernel
//! interface needed from a GPU backend is described in `docs/GHOST.md`.

use mklight_core::job::RunTimer;
use mklight_core::math::KahanSum;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::Ray;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::model::GhostModel;
use crate::paraxial::run_info;
use crate::path::{EnumerationSettings, GhostPath, Step, enumerate};
use crate::signature::GhostCoverage;

/// Rays per cancellation / progress block.
const BLOCK: usize = 16384;

/// Settings of [`ghost_irradiance`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IrradianceSettings {
    pub enumeration: EnumerationSettings,
    /// Explicit paths instead of the enumeration.
    #[serde(default)]
    pub paths: Option<Vec<GhostPath>>,
    /// System field indices; empty = all fields.
    #[serde(default)]
    pub fields: Vec<usize>,
    /// System wavelength indices; empty = all wavelengths.
    #[serde(default)]
    pub wavelengths: Vec<usize>,
    /// Grid cells across the overfilled pupil square.
    pub pupil_samples: usize,
    /// Sampled pupil radius in entrance-pupil radii (≥ 1).
    pub overfill: f64,
    /// Map bins `[nx, ny]`.
    pub map_bins: [usize; 2],
    /// Map half-size `[x, y]` (mm); `None` = sensor or image clear aperture.
    #[serde(default)]
    pub map_half_size: Option<[f64; 2]>,
    /// Keep every path's map (memory: paths × bins).
    #[serde(default)]
    pub keep_path_maps: bool,
}

impl Default for IrradianceSettings {
    fn default() -> Self {
        IrradianceSettings {
            enumeration: EnumerationSettings::default(),
            paths: None,
            fields: Vec::new(),
            wavelengths: Vec::new(),
            pupil_samples: 128,
            overfill: 1.5,
            map_bins: [64, 64],
            map_half_size: None,
            keep_path_maps: false,
        }
    }
}

/// An irradiance map over the sensor, row-major with `y` outer, both axes
/// ascending from `−half_size` to `+half_size`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IrradianceMap {
    pub bins: [usize; 2],
    pub half_size: [f64; 2],
    /// Flux per mm² (flux relative to the entrance-pupil input).
    pub values: Vec<f64>,
}

impl IrradianceMap {
    pub fn new(bins: [usize; 2], half_size: [f64; 2]) -> Self {
        IrradianceMap {
            bins,
            half_size,
            values: vec![0.0; bins[0] * bins[1]],
        }
    }
    /// Area of one bin (mm²).
    pub fn bin_area(&self) -> f64 {
        (2.0 * self.half_size[0] / self.bins[0] as f64)
            * (2.0 * self.half_size[1] / self.bins[1] as f64)
    }
    /// Bin index of a point, `None` outside the map.
    pub fn bin_of(&self, x: f64, y: f64) -> Option<usize> {
        let fx = (x + self.half_size[0]) / (2.0 * self.half_size[0]) * self.bins[0] as f64;
        let fy = (y + self.half_size[1]) / (2.0 * self.half_size[1]) * self.bins[1] as f64;
        if !(fx >= 0.0 && fy >= 0.0) {
            return None;
        }
        let (ix, iy) = (fx as usize, fy as usize);
        (ix < self.bins[0] && iy < self.bins[1]).then_some(iy * self.bins[0] + ix)
    }
    pub fn peak(&self) -> f64 {
        self.values.iter().copied().fold(0.0, f64::max)
    }
    /// Total flux on the map (compensated).
    pub fn flux(&self) -> f64 {
        let a = self.bin_area();
        self.values
            .iter()
            .map(|v| v * a)
            .collect::<KahanSum>()
            .value()
    }
}

/// Statistics of one path at one field (all selected wavelengths).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldIrradiance {
    /// System field index.
    pub field: usize,
    /// Detected flux, on or off the map.
    pub flux: f64,
    pub flux_on_map: f64,
    /// Peak irradiance of this field's map.
    pub peak: f64,
    pub centroid: [f64; 2],
    pub rms_radius: f64,
    pub rays_traced: usize,
    pub rays_detected: usize,
}

/// Result of one path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathIrradiance {
    pub path: GhostPath,
    pub label: String,
    pub fields: Vec<FieldIrradiance>,
    /// Ghost flux on the map / primary flux on the map (0 for the primary
    /// itself: see `flux`).
    pub flux_ratio: f64,
    /// Max over fields of ghost peak / primary peak.
    pub peak_ratio: f64,
    /// Flux on the map, all fields.
    pub flux: f64,
    #[serde(default)]
    pub map: Option<IrradianceMap>,
}

/// Result of [`ghost_irradiance`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GhostIrradiance {
    pub run: RunInfo,
    /// The primary image, computed the same way (always with its map).
    pub primary: PathIrradiance,
    /// Ranked by `peak_ratio`, strongest first.
    pub ghosts: Vec<PathIrradiance>,
    pub total_ghost_map: IrradianceMap,
    /// Total ghost peak / primary peak, and total ghost flux / primary flux.
    pub total_peak_ratio: f64,
    pub total_flux_ratio: f64,
    pub truncated: bool,
    pub coverage: GhostCoverage,
}

/// One prepared starting ray.
#[derive(Clone, Copy, Debug)]
struct StartRay {
    ray: Ray,
    wl: usize,
    weight: f64,
}

fn ratio(a: f64, b: f64) -> f64 {
    if b > 0.0 {
        a / b
    } else if a > 0.0 {
        f64::INFINITY
    } else {
        0.0
    }
}

/// Starting rays per selected field (module docs).
fn start_rays(
    model: &GhostModel,
    fields: &[usize],
    wls: &[usize],
    settings: &IrradianceSettings,
) -> MkResult<Vec<Vec<StartRay>>> {
    let tracer = model.tracer();
    let sys = model.system();
    let osys = sys.system();
    let n = settings.pupil_samples;
    let k = settings.overfill;
    let pupil = tracer.pupil();
    let scale = if pupil.telecentric {
        pupil.u_marginal.abs()
    } else {
        pupil.r_ep
    };
    let fw: Vec<f64> = fields
        .iter()
        .map(|&f| osys.fields.fields[f].weight)
        .collect();
    let ww: Vec<f64> = wls
        .iter()
        .map(|&w| {
            sys.wavelength_weights[w]
                * osys
                    .sensor
                    .as_ref()
                    .map_or(1.0, |s| s.response(sys.wavelengths_um[w]))
        })
        .collect();
    let (sf, sw) = (fw.iter().sum::<f64>(), ww.iter().sum::<f64>());
    if !(sf > 0.0 && sw > 0.0) {
        return Err(MkError::InvalidModel(
            "selected fields and wavelengths have zero total weight".into(),
        ));
    }
    let h = 2.0 * k / n as f64;
    let cell = h * h / std::f64::consts::PI;
    let mut out = Vec::with_capacity(fields.len());
    for (fi, &f) in fields.iter().enumerate() {
        let field = tracer.field(f);
        let mut rays = Vec::new();
        if field.is_valid() {
            for (wi, &w) in wls.iter().enumerate() {
                let weight = fw[fi] / sf * ww[wi] / sw * cell;
                let c = field.aims[w].chief;
                for iy in 0..n {
                    let py = -k + h * (iy as f64 + 0.5);
                    for ix in 0..n {
                        let px = -k + h * (ix as f64 + 0.5);
                        if px.mul_add(px, py * py) > k * k {
                            continue;
                        }
                        let a = [px.mul_add(scale, c[0]), py.mul_add(scale, c[1])];
                        if let Some(ray) = tracer.start_ray(&field.source, a) {
                            rays.push(StartRay { ray, wl: w, weight });
                        }
                    }
                }
            }
        }
        out.push(rays);
    }
    Ok(out)
}

/// Trace one path for all start rays; per-field maps and statistics.
fn run_path(
    model: &GhostModel,
    steps: &[Step],
    starts: &[Vec<StartRay>],
    fields: &[usize],
    template: &IrradianceMap,
    ctx: &JobContext,
    progress: (f64, f64),
) -> MkResult<(Vec<IrradianceMap>, Vec<FieldIrradiance>)> {
    let total: usize = starts.iter().map(Vec::len).sum();
    let offsets: Vec<usize> = starts
        .iter()
        .scan(0, |acc, v| {
            let o = *acc;
            *acc += v.len();
            Some(o)
        })
        .collect();
    let locate = |i: usize| {
        let f = offsets.partition_point(|&o| o <= i) - 1;
        (f, i - offsets[f])
    };
    let mut hits: Vec<Option<(f64, f64, f64)>> = Vec::with_capacity(total);
    let mut start = 0;
    while start < total {
        ctx.check()?;
        let end = (start + BLOCK).min(total);
        let mut block: Vec<Option<(f64, f64, f64)>> = (start..end)
            .into_par_iter()
            .with_min_len(64)
            .map(|i| {
                let (f, j) = locate(i);
                let s = &starts[f][j];
                let r = model.trace_steps(&s.ray, s.wl, steps, None);
                r.status
                    .is_ok()
                    .then(|| (r.ray.pos.x, r.ray.pos.y, s.weight * r.weight()))
            })
            .collect();
        hits.append(&mut block);
        start = end;
        let f = start as f64 / total as f64;
        ctx.report(
            progress.0 + f * (progress.1 - progress.0),
            "tracing ghost paths",
        );
    }
    let area = template.bin_area();
    let mut maps = Vec::with_capacity(fields.len());
    let mut stats = Vec::with_capacity(fields.len());
    for (fi, &field) in fields.iter().enumerate() {
        let lo = offsets[fi];
        let field_hits = &hits[lo..lo + starts[fi].len()];
        let mut bins = vec![KahanSum::default(); template.values.len()];
        let (mut sw, mut swx, mut swy, mut on) = (
            KahanSum::default(),
            KahanSum::default(),
            KahanSum::default(),
            KahanSum::default(),
        );
        let mut detected = 0usize;
        for &(x, y, w) in field_hits.iter().flatten() {
            detected += 1;
            sw.add(w);
            swx.add(w * x);
            swy.add(w * y);
            if let Some(b) = template.bin_of(x, y) {
                bins[b].add(w);
                on.add(w);
            }
        }
        let flux = sw.value();
        let (cx, cy) = if flux > 0.0 {
            (swx.value() / flux, swy.value() / flux)
        } else {
            (0.0, 0.0)
        };
        // Second moment about the centroid (two-pass, no cancellation).
        let m2: KahanSum = field_hits
            .iter()
            .flatten()
            .map(|&(x, y, w)| w * (x - cx).mul_add(x - cx, (y - cy) * (y - cy)))
            .collect();
        let mut map = template.clone();
        for (v, b) in map.values.iter_mut().zip(&bins) {
            *v = b.value() / area;
        }
        stats.push(FieldIrradiance {
            field,
            flux,
            flux_on_map: on.value(),
            peak: map.peak(),
            centroid: [cx, cy],
            rms_radius: if flux > 0.0 {
                (m2.value() / flux).sqrt()
            } else {
                0.0
            },
            rays_traced: starts[fi].len(),
            rays_detected: detected,
        });
        maps.push(map);
    }
    Ok((maps, stats))
}

fn sum_maps(maps: &[IrradianceMap], template: &IrradianceMap) -> IrradianceMap {
    let mut out = template.clone();
    for (i, v) in out.values.iter_mut().enumerate() {
        *v = maps
            .iter()
            .map(|m| m.values[i])
            .collect::<KahanSum>()
            .value();
    }
    out
}

/// Real-ray ghost irradiance maps and ghost-to-signal ratios (module docs).
pub fn ghost_irradiance(
    model: &GhostModel,
    settings: &IrradianceSettings,
    ctx: &JobContext,
) -> MkResult<GhostIrradiance> {
    let timer = RunTimer::start();
    let sys = model.system();
    let osys = sys.system();
    let image = sys.image_index();
    if settings.pupil_samples == 0 || settings.map_bins[0] == 0 || settings.map_bins[1] == 0 {
        return Err(MkError::InvalidModel(
            "pupil samples and map bins must be positive".into(),
        ));
    }
    if !(settings.overfill >= 1.0 && settings.overfill.is_finite()) {
        return Err(MkError::InvalidModel("overfill must be ≥ 1".into()));
    }
    let fields: Vec<usize> = if settings.fields.is_empty() {
        (0..osys.fields.fields.len()).collect()
    } else {
        settings.fields.clone()
    };
    let wls: Vec<usize> = if settings.wavelengths.is_empty() {
        (0..sys.wavelengths_um.len()).collect()
    } else {
        settings.wavelengths.clone()
    };
    if fields.iter().any(|&f| f >= osys.fields.fields.len())
        || wls.iter().any(|&w| w >= sys.wavelengths_um.len())
    {
        return Err(MkError::InvalidModel(
            "field or wavelength index out of range".into(),
        ));
    }
    let half = match settings.map_half_size {
        Some(h) => h,
        None => match &osys.sensor {
            Some(s) => [s.half_width(), s.half_height()],
            None => {
                let r = model.semi_diameters().values[image].max(1e-3);
                [r, r]
            }
        },
    };
    if !(half[0] > 0.0 && half[1] > 0.0 && half[0].is_finite() && half[1].is_finite()) {
        return Err(MkError::InvalidModel("map size must be positive".into()));
    }
    let template = IrradianceMap::new(settings.map_bins, half);

    let (paths, truncated, coverage) = match &settings.paths {
        Some(p) => (
            p.clone(),
            false,
            GhostCoverage {
                interfaces: model.interface_kinds().to_vec(),
                paths: p.iter().cloned().collect(),
            },
        ),
        None => {
            let e = enumerate(model, &settings.enumeration)?;
            (
                e.paths.into_iter().map(|p| p.path).collect(),
                e.truncated,
                e.coverage,
            )
        }
    };
    let paths: Vec<GhostPath> = paths
        .into_iter()
        .filter(|p| !p.is_primary() && p.ends_at_image())
        .collect();
    let path_steps = paths
        .iter()
        .map(|p| model.check_path(p))
        .collect::<MkResult<Vec<_>>>()?;

    ctx.report(0.0, "preparing rays");
    ctx.check()?;
    let starts = start_rays(model, &fields, &wls, settings)?;
    let n_jobs = (paths.len() + 1) as f64;

    let pstep = model.check_path(&GhostPath::primary())?;
    let (pmaps, pstats) = run_path(
        model,
        &pstep,
        &starts,
        &fields,
        &template,
        ctx,
        (0.0, 1.0 / n_jobs),
    )?;
    let primary_map = sum_maps(&pmaps, &template);
    let primary_flux = primary_map.flux();
    let primary = PathIrradiance {
        path: GhostPath::primary(),
        label: GhostPath::primary().label(image),
        fields: pstats.clone(),
        flux_ratio: 1.0,
        peak_ratio: 1.0,
        flux: primary_flux,
        map: Some(primary_map.clone()),
    };

    let mut total = template.clone();
    let mut total_acc = vec![KahanSum::default(); template.values.len()];
    let mut ghosts = Vec::with_capacity(paths.len());
    for (k, (path, steps)) in paths.iter().zip(&path_steps).enumerate() {
        let span = ((k + 1) as f64 / n_jobs, (k + 2) as f64 / n_jobs);
        let (maps, stats) = run_path(model, steps, &starts, &fields, &template, ctx, span)?;
        let map = sum_maps(&maps, &template);
        for (acc, v) in total_acc.iter_mut().zip(&map.values) {
            acc.add(*v);
        }
        let peak_ratio = stats
            .iter()
            .zip(&pstats)
            .map(|(g, p)| ratio(g.peak, p.peak))
            .fold(0.0, f64::max);
        let flux = map.flux();
        ghosts.push(PathIrradiance {
            label: path.label(image),
            path: path.clone(),
            fields: stats,
            flux_ratio: ratio(flux, primary_flux),
            peak_ratio,
            flux,
            map: settings.keep_path_maps.then_some(map),
        });
    }
    for (v, acc) in total.values.iter_mut().zip(&total_acc) {
        *v = acc.value();
    }
    ghosts.sort_by(|a, b| {
        b.peak_ratio
            .total_cmp(&a.peak_ratio)
            .then_with(|| a.path.cmp(&b.path))
    });
    ctx.check()?;
    ctx.report(1.0, "done");
    let total_flux_ratio = ratio(total.flux(), primary_flux);
    let total_peak_ratio = ratio(total.peak(), primary_map.peak());
    let warnings = model.tracer().warnings().to_vec();
    Ok(GhostIrradiance {
        run: run_info(model, timer, "ghost.irradiance", settings, warnings),
        primary,
        ghosts,
        total_ghost_map: total,
        total_peak_ratio,
        total_flux_ratio,
        truncated,
        coverage,
    })
}
