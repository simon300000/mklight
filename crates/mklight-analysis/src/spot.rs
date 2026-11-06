//! Spot diagrams: where the rays of each field and wavelength land on the
//! image surface.
//!
//! # Sampling and weighting
//!
//! Each selected field is traced at each selected wavelength over the same
//! pupil pattern ([`PupilPattern`]; clipping on, the tracer's aiming mode).
//! Every ray that reaches the image carries the weight of its wavelength
//! (`Wavelength::weight`), so a polychromatic statistic is the weighted mean
//! over all arrived rays. Rays that are clipped, miss, undergo TIR or cannot
//! be aimed count as *vignetted*; the vignetted fraction of a wavelength is
//! `1 − arrived/launched`, and the polychromatic one is weighted the same way.
//!
//! # References and statistics
//!
//! Positions are in the image surface's **local** frame (mm). The *chief* of
//! a field is its unclipped chief ray at the primary wavelength (traced even
//! when the primary wavelength is not selected). Every point is first
//! expressed relative to that chief hit (one subtraction), and all sums are
//! compensated (CONVENTIONS §7.5):
//!
//! * centroid `μ = Σwq / Σw`;
//! * RMS radius about the centroid `√(Σw|q − μ|² / Σw)` (two-pass) and about
//!   the chief `√(Σw|q|² / Σw)`;
//! * GEO radius: the largest distance from the centroid / chief.
//!
//! Per-wavelength statistics use that wavelength's own chief ray. The plotted
//! points and `rms_radius`/`geo_radius` are relative to the reference chosen
//! in the settings ([`SpotReference`]).
//!
//! # Defocus and through-focus
//!
//! A defocus `δ` moves every image point along its ray by an axial (local
//! `z`) distance `δ`: `p(δ) = p + δ·(d_x, d_y)/d_z`. For a plane image surface
//! this is exactly the spot on the image plane translated by `δ`; for a curved
//! image surface it is the first-order translation of the surface, and a
//! warning says so. Positive `δ` is along the image frame's `+z`.
//!
//! A through-focus request samples `steps` planes from `min` to `max`; a
//! `null` bound makes the range automatic and symmetric, `±Δ` with
//! `Δ = max(8 λ N_w², 4 N_w · RMS)` (the depth of focus or the geometric
//! blur, whichever is larger; `RMS` is the largest polychromatic RMS radius
//! about the centroid over the fields; 0.1 mm when neither is finite). With
//! the points, [`FieldSpot::through_focus`] then carries the sampled planes,
//! and each packed point has its slope relative to the plotted reference
//! ([`PackedSpotPoints::slope`]), so a point at plane `d` is
//! `q + (d − defocus) · slope`: a client draws every plane from one trace.
//!
//! # Batches from other backends
//!
//! The statistics are a reduction of a [`RayBatch`]. [`FieldRays`] holds
//! the arrived rays of one field relative to its primary chief ray, and
//! [`FieldRays::from_image_rays`] builds it from plain image hits.
//! [`spot_from_rays`] reduces a batch traced elsewhere (a GPU backend, an
//! optimiser operand) with exactly the same code as [`spot_diagram`], and
//! records the given [`BackendInfo`] in the `RunInfo`.
//!
//! The Airy radius is `1.22 λ N_w` with the primary wavelength and the
//! paraxial working F/#.

use mklight_core::job::RunTimer;
use mklight_core::{JobContext, MkError, MkResult, RunInfo, Vec3};
use mklight_trace::{Grid, SequentialTracer, trace_grid_job};
use serde::{Deserialize, Serialize};

use crate::util::{
    BackendInfo, FieldSelection, Moments, PupilPattern, WavelengthSelection, check_count,
    check_finite, curved_image_warning, field_value, linspace, nf, pupil_points, run_info_backend,
};

/// Reference point of the plotted spot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SpotReference {
    /// The primary-wavelength chief ray.
    #[default]
    Chief,
    /// The polychromatic (weighted) centroid.
    Centroid,
}

/// Through-focus sampling: `steps` defocus values from `min` to `max` (mm).
/// A `None` bound makes the range automatic (module docs).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThroughFocus {
    #[serde(default)]
    pub min: Option<f64>,
    #[serde(default)]
    pub max: Option<f64>,
    #[serde(default = "default_steps")]
    pub steps: usize,
}

fn default_steps() -> usize {
    3
}

impl Default for ThroughFocus {
    fn default() -> Self {
        ThroughFocus {
            min: None,
            max: None,
            steps: 3,
        }
    }
}

impl ThroughFocus {
    /// A symmetric range `±half`.
    pub fn symmetric(half: f64, steps: usize) -> Self {
        ThroughFocus {
            min: Some(-half),
            max: Some(half),
            steps,
        }
    }

    /// The automatic half-range of the module docs.
    pub fn auto_half_range(wavelength_um: f64, working_f_number: f64, max_rms: f64) -> f64 {
        let n = working_f_number;
        let depth = 8.0 * wavelength_um * 1e-3 * n * n;
        let blur = 4.0 * n * max_rms;
        let half = [depth, blur]
            .into_iter()
            .filter(|v| v.is_finite() && *v > 0.0)
            .fold(0.0, f64::max);
        if half > 0.0 { half } else { 0.1 }
    }

    /// The sampled planes (mm), with `auto_half` for an automatic bound.
    pub fn planes(&self, auto_half: f64) -> Vec<f64> {
        let (min, max) = match (self.min, self.max) {
            (Some(a), Some(b)) => (a, b),
            _ => (-auto_half, auto_half),
        };
        linspace(min, max, self.steps)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SpotDiagramSettings {
    pub pattern: PupilPattern,
    /// Density of the pattern (see [`PupilPattern`]).
    pub rings: usize,
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub reference: SpotReference,
    /// Philox seed of the random and dithered patterns.
    pub seed: u64,
    /// Defocus from the image surface (mm, module docs).
    pub defocus: f64,
    pub through_focus: Option<ThroughFocus>,
    /// Include the individual ray points (statistics are always computed).
    pub include_points: bool,
}

impl Default for SpotDiagramSettings {
    fn default() -> Self {
        SpotDiagramSettings {
            pattern: PupilPattern::Hexapolar,
            rings: 6,
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            reference: SpotReference::Chief,
            seed: 1,
            defocus: 0.0,
            through_focus: None,
            include_points: true,
        }
    }
}

/// One ray point relative to the field's reference (mm).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpotPoint {
    #[serde(with = "nf")]
    pub x: f64,
    #[serde(with = "nf")]
    pub y: f64,
    /// System wavelength index.
    pub wavelength: usize,
}

/// Statistics of a set of rays (module docs). NaN (JSON `null`) when no ray
/// arrived.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpotStats {
    pub rays_launched: usize,
    pub rays_arrived: usize,
    #[serde(with = "nf")]
    pub vignetted_fraction: f64,
    /// Centroid relative to the chief ray (mm).
    #[serde(with = "nf::arr2")]
    pub centroid: [f64; 2],
    #[serde(with = "nf")]
    pub rms_radius_centroid: f64,
    #[serde(with = "nf")]
    pub rms_radius_chief: f64,
    #[serde(with = "nf")]
    pub geo_radius_centroid: f64,
    #[serde(with = "nf")]
    pub geo_radius_chief: f64,
    /// RMS of the x / y components about the centroid.
    #[serde(with = "nf")]
    pub rms_x: f64,
    #[serde(with = "nf")]
    pub rms_y: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WavelengthSpot {
    pub wavelength: usize,
    pub wavelength_um: f64,
    pub weight: f64,
    /// This wavelength's chief ray on the image surface (local mm).
    #[serde(with = "nf::arr2")]
    pub chief: [f64; 2],
    /// Statistics about this wavelength's chief ray and centroid.
    pub stats: SpotStats,
}

/// One through-focus sample.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct FocusSample {
    pub defocus: f64,
    #[serde(with = "nf")]
    pub rms_radius_centroid: f64,
    #[serde(with = "nf")]
    pub rms_radius_chief: f64,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldSpot {
    pub field: usize,
    /// Field value as specified (degrees or mm).
    #[serde(with = "nf::arr2")]
    pub field_value: [f64; 2],
    /// Ray points relative to `reference` (empty unless `include_points`).
    pub points: Vec<SpotPoint>,
    /// Reference point on the image surface (local mm).
    #[serde(with = "nf::arr2")]
    pub reference: [f64; 2],
    /// Primary-wavelength chief ray on the image surface (local mm).
    #[serde(with = "nf::arr2")]
    pub chief: [f64; 2],
    /// Polychromatic centroid (local mm).
    #[serde(with = "nf::arr2")]
    pub centroid: [f64; 2],
    /// Polychromatic RMS radius about the reference (mm).
    #[serde(with = "nf")]
    pub rms_radius: f64,
    /// Largest distance from the reference (mm).
    #[serde(with = "nf")]
    pub geo_radius: f64,
    /// Polychromatic statistics about the primary chief ray.
    pub polychromatic: SpotStats,
    pub per_wavelength: Vec<WavelengthSpot>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub through_focus: Vec<FocusSample>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SpotDiagram {
    pub fields: Vec<FieldSpot>,
    /// All system wavelengths (µm), indexed by `SpotPoint::wavelength`.
    pub wavelengths_um: Vec<f64>,
    /// The selected system wavelengths.
    pub wavelengths: Vec<usize>,
    /// Airy radius `1.22 λ N_w` at the primary wavelength (mm); ∞ when afocal.
    #[serde(with = "nf")]
    pub airy_radius: f64,
    pub defocus: f64,
    pub run_info: RunInfo,
    /// The ray points of every field in buffer form, set instead of
    /// `FieldSpot::points` by [`spot_diagram_packed_job`]; never serialised.
    #[serde(skip)]
    pub packed_points: Option<PackedSpotPoints>,
}

/// Ray points of all fields as flat typed arrays (the engine's `points`,
/// `point_field` and `point_wavelength` buffers), filled once by the solver
/// without per-point structs. Values follow the JSON conventions of
/// [`SpotPoint`]: NaN is canonical, infinities are kept.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PackedSpotPoints {
    /// `[x, y]` per point, relative to its field's reference (mm).
    pub xy: Vec<f64>,
    /// Position of the point's field in `SpotDiagram::fields`.
    pub field: Vec<u32>,
    /// System wavelength index of the point.
    pub wavelength: Vec<u32>,
    /// `[dx/dz, dy/dz]` per point relative to the plotted reference (mm per
    /// mm of defocus); empty unless through-focus was requested.
    pub slope: Vec<f64>,
    /// `[offset, count]` of each field's rows, in `SpotDiagram::fields` order.
    pub ranges: Vec<[usize; 2]>,
}

/// Points between cancellation checks while packing.
pub(crate) const POINT_CHUNK: usize = 1 << 16;

impl PackedSpotPoints {
    pub(crate) fn with_capacity(n: usize, fields: usize) -> Self {
        PackedSpotPoints {
            xy: Vec::with_capacity(2 * n),
            field: Vec::with_capacity(n),
            wavelength: Vec::with_capacity(n),
            slope: Vec::new(),
            ranges: Vec::with_capacity(fields),
        }
    }
    #[inline]
    pub(crate) fn push(&mut self, field: usize, x: f64, y: f64, wavelength: usize) {
        let canonical = |v: f64| if v.is_nan() { f64::NAN } else { v };
        self.xy.extend([canonical(x), canonical(y)]);
        self.field.push(field as u32);
        self.wavelength.push(wavelength as u32);
    }
    /// The slope of the point just pushed (through-focus requests).
    #[inline]
    pub(crate) fn push_slope(&mut self, sx: f64, sy: f64) {
        let canonical = |v: f64| if v.is_nan() { f64::NAN } else { v };
        self.slope.extend([canonical(sx), canonical(sy)]);
    }
    /// Close the current field, whose rows start at `start`.
    pub(crate) fn close(&mut self, start: usize) {
        self.ranges.push([start, self.field.len() - start]);
    }
    /// Number of points.
    pub fn len(&self) -> usize {
        self.field.len()
    }
    pub fn is_empty(&self) -> bool {
        self.field.is_empty()
    }
}

// ----- Shared ray data -----------------------------------------------------------

/// A ray on the image surface: hit point (image local frame, mm) and unit
/// direction (image local frame).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageHit {
    pub point: [f64; 2],
    pub dir: Vec3,
}

/// One launched ray of a batch: its system wavelength and its image hit,
/// `None` when it was vignetted, missed or failed.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ImageRay {
    pub wavelength: usize,
    pub hit: Option<ImageHit>,
}

/// One arrived ray, relative to its field's primary chief ray.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    /// Position in the batch's list of wavelengths.
    pub wl_pos: usize,
    /// `p − c` (mm) at zero defocus.
    pub q: [f64; 2],
    /// `t − t_c`: image-space slope `(d_x, d_y)/d_z` minus the chief's.
    pub dt: [f64; 2],
    pub weight: f64,
}

impl RayHit {
    /// Position relative to the chief at defocus `delta`.
    #[inline]
    pub fn at(&self, delta: f64) -> [f64; 2] {
        [
            delta.mul_add(self.dt[0], self.q[0]),
            delta.mul_add(self.dt[1], self.q[1]),
        ]
    }
}

/// The rays of one field, reduced to the form every spot statistic uses:
/// arrived rays relative to the primary chief ray (removing the large
/// common image height before any sum), with their slopes for defocus.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldRays {
    pub field: usize,
    /// Absolute primary chief hit (image local), or the first arrived ray
    /// when the chief failed (`chief_ok = false`).
    pub origin: [f64; 2],
    pub origin_slope: [f64; 2],
    pub chief_ok: bool,
    /// Chief of each batch wavelength relative to `origin`: `(q, dt)`, NaN
    /// when it failed.
    pub wl_chief: Vec<([f64; 2], [f64; 2])>,
    pub rays: Vec<RayHit>,
    /// Launched and arrived rays per batch wavelength.
    pub launched: Vec<usize>,
    pub arrived: Vec<usize>,
}

/// Image-plane ray data of several fields over one list of wavelengths: the
/// input of the reductions [`spot_from_rays`] and
/// `crate::energy::{encircled_energy, geometric_mtf, best_focus}_from_rays`,
/// whatever backend traced the rays.
#[derive(Clone, Debug, PartialEq)]
pub struct RayBatch {
    /// System wavelength indices; `RayHit::wl_pos` indexes this list.
    pub wavelengths: Vec<usize>,
    pub fields: Vec<FieldRays>,
}

impl RayBatch {
    /// Check the internal consistency (counts and indices) of a batch built
    /// outside this crate.
    pub fn validate(&self, num_fields: usize, num_wavelengths: usize) -> MkResult<()> {
        let bad = |m: String| Err(MkError::InvalidModel(format!("ray batch: {m}")));
        let nw = self.wavelengths.len();
        if nw == 0 || self.wavelengths.iter().any(|&w| w >= num_wavelengths) {
            return bad("wavelength list empty or out of range".into());
        }
        for f in &self.fields {
            if f.field >= num_fields {
                return bad(format!("field {} does not exist", f.field));
            }
            if f.wl_chief.len() != nw || f.launched.len() != nw || f.arrived.len() != nw {
                return bad(format!(
                    "field {}: per-wavelength lists of wrong length",
                    f.field
                ));
            }
            let mut count = vec![0usize; nw];
            for r in &f.rays {
                if r.wl_pos >= nw {
                    return bad(format!(
                        "field {}: wavelength position out of range",
                        f.field
                    ));
                }
                count[r.wl_pos] += 1;
            }
            if count != f.arrived || f.arrived.iter().zip(&f.launched).any(|(a, l)| a > l) {
                return bad(format!("field {}: arrived counts inconsistent", f.field));
            }
        }
        Ok(())
    }
}

impl FieldRays {
    /// Polychromatic moments about the primary chief at defocus `delta`.
    pub fn moments(&self, delta: f64) -> Moments {
        Moments::of(self.rays.iter().map(move |r| (r.at(delta), r.weight)))
    }

    /// Moments of one batch wavelength (position `k`) about its own chief.
    pub fn moments_wl(&self, k: usize, delta: f64) -> Moments {
        let (cq, cdt) = self.wl_chief[k];
        let c = [delta.mul_add(cdt[0], cq[0]), delta.mul_add(cdt[1], cq[1])];
        Moments::of(
            self.rays
                .iter()
                .filter(move |r| r.wl_pos == k)
                .map(move |r| {
                    let p = r.at(delta);
                    ([p[0] - c[0], p[1] - c[1]], 1.0)
                }),
        )
    }

    /// Reduce the image hits of one field (from any backend). `weights` are
    /// the spectral weights of `wavelengths` (system indices);
    /// `wavelength_chiefs[k]` is the chief ray of `wavelengths[k]`, and
    /// `primary_chief` the primary-wavelength chief, which is the origin of
    /// every position (the first arrived ray, with a warning, if it failed).
    /// A ray whose wavelength is not in `wavelengths` is an error.
    pub fn from_image_rays(
        field: usize,
        wavelengths: &[usize],
        weights: &[f64],
        primary_chief: Option<ImageHit>,
        wavelength_chiefs: &[Option<ImageHit>],
        rays: &[ImageRay],
        warnings: &mut Vec<String>,
    ) -> MkResult<FieldRays> {
        Self::from_hits(
            field,
            wavelengths,
            weights,
            primary_chief,
            wavelength_chiefs,
            rays.iter().map(|r| (r.wavelength, r.hit)),
            warnings,
        )
    }

    /// [`FieldRays::from_image_rays`] over `(system wavelength, hit)` pairs
    /// read in place, without an intermediate list of rays.
    fn from_hits<I: Iterator<Item = (usize, Option<ImageHit>)> + Clone>(
        field: usize,
        wavelengths: &[usize],
        weights: &[f64],
        primary_chief: Option<ImageHit>,
        wavelength_chiefs: &[Option<ImageHit>],
        rays: I,
        warnings: &mut Vec<String>,
    ) -> MkResult<FieldRays> {
        if weights.len() != wavelengths.len() || wavelength_chiefs.len() != wavelengths.len() {
            return Err(MkError::InvalidModel(
                "ray batch: per-wavelength lists of wrong length".into(),
            ));
        }
        let hit = |h: &ImageHit| slope(h.dir).map(|t| (h.point, t));
        let pc = primary_chief.as_ref().and_then(hit);
        let first = rays.clone().find_map(|(_, h)| h.as_ref().and_then(hit));
        let (origin, origin_slope, chief_ok) = match (pc, first) {
            (Some((p, t)), _) => (p, t, true),
            (None, Some((p, t))) => {
                warnings.push(format!(
                    "field {field}: primary chief ray failed; statistics about the chief are undefined"
                ));
                (p, t, false)
            }
            (None, None) => ([0.0, 0.0], [0.0, 0.0], false),
        };
        let rel = |p: [f64; 2], t: [f64; 2]| {
            (
                [p[0] - origin[0], p[1] - origin[1]],
                [t[0] - origin_slope[0], t[1] - origin_slope[1]],
            )
        };
        let wl_chief = wavelength_chiefs
            .iter()
            .map(|c| match c.as_ref().and_then(hit) {
                Some((p, t)) if chief_ok => rel(p, t),
                _ => ([f64::NAN; 2], [f64::NAN; 2]),
            })
            .collect();
        let mut hits = Vec::with_capacity(rays.clone().filter(|(_, h)| h.is_some()).count());
        let mut launched = vec![0; wavelengths.len()];
        let mut arrived = vec![0; wavelengths.len()];
        for (wavelength, h) in rays {
            let k = wavelengths
                .iter()
                .position(|&w| w == wavelength)
                .ok_or_else(|| {
                    MkError::InvalidModel(format!(
                        "ray batch: wavelength {wavelength} not in the batch"
                    ))
                })?;
            launched[k] += 1;
            let Some((p, t)) = h.as_ref().and_then(hit) else {
                continue;
            };
            arrived[k] += 1;
            let (q, dt) = rel(p, t);
            hits.push(RayHit {
                wl_pos: k,
                q,
                dt,
                weight: weights[k],
            });
        }
        Ok(FieldRays {
            field,
            origin,
            origin_slope,
            chief_ok,
            wl_chief,
            rays: hits,
            launched,
            arrived,
        })
    }
}

/// Image-space slope `(d_x/d_z, d_y/d_z)`; `None` for a ray parallel to the
/// image frame's `xy` plane or non-finite data.
#[inline]
pub(crate) fn slope(d: Vec3) -> Option<[f64; 2]> {
    let t = [d.x / d.z, d.y / d.z];
    (d.z != 0.0 && t[0].is_finite() && t[1].is_finite()).then_some(t)
}

/// Trace `fields × wavelengths × pupil` (clipped) and the chief rays with
/// the CPU reference tracer, and reduce them with
/// [`FieldRays::from_image_rays`].
pub fn trace_field_rays(
    tracer: &SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    pupil: &[[f64; 2]],
    ctx: &JobContext,
    warnings: &mut Vec<String>,
) -> MkResult<RayBatch> {
    let sys = tracer.system();
    let primary = sys.primary_wavelength;
    let weights: Vec<f64> = wls.iter().map(|&w| sys.wavelength_weights[w]).collect();
    let nf = fields.len();
    let mut out = Vec::with_capacity(nf);
    // One field at a time: only that field's traced rays are alive while
    // they are reduced (identical per-ray results in the same order).
    for (fi, &f) in fields.iter().enumerate() {
        ctx.check()?;
        let pf = tracer.field(f);
        let grid = Grid {
            fields: std::slice::from_ref(pf),
            wavelengths: wls,
            pupil,
        };
        let span = ctx.subrange(
            0.95 * fi as f64 / nf as f64,
            0.95 * (fi + 1) as f64 / nf as f64,
        );
        let rays = trace_grid_job(tracer, &grid, &tracer.options(), &span)?;
        let chief = |wl: usize| {
            let c = tracer.chief(pf, wl);
            c.status.is_ok().then_some(ImageHit {
                point: [c.ray.pos.x, c.ray.pos.y],
                dir: c.ray.dir,
            })
        };
        let pc = chief(primary);
        let wl_chiefs: Vec<_> = wls
            .iter()
            .map(|&w| if w == primary { pc } else { chief(w) })
            .collect();
        let hits = rays.iter().map(|r| {
            let hit = r.status.is_ok().then_some(ImageHit {
                point: [r.point.x, r.point.y],
                dir: r.dir,
            });
            (r.wavelength as usize, hit)
        });
        out.push(FieldRays::from_hits(
            f, wls, &weights, pc, &wl_chiefs, hits, warnings,
        )?);
    }
    Ok(RayBatch {
        wavelengths: wls.to_vec(),
        fields: out,
    })
}

fn stats(
    m: &Moments,
    launched: f64,
    arrived: usize,
    missing_weight: f64,
    total: f64,
    chief_ok: bool,
) -> SpotStats {
    let nan_if = |v: f64| if chief_ok { v } else { f64::NAN };
    SpotStats {
        rays_launched: launched as usize,
        rays_arrived: arrived,
        vignetted_fraction: if total > 0.0 {
            missing_weight / total
        } else {
            f64::NAN
        },
        centroid: [nan_if(m.centroid[0]), nan_if(m.centroid[1])],
        rms_radius_centroid: m.rms_centroid,
        rms_radius_chief: nan_if(m.rms_origin),
        geo_radius_centroid: m.geo_centroid,
        geo_radius_chief: nan_if(m.geo_origin),
        rms_x: m.rms_x,
        rms_y: m.rms_y,
    }
}

pub(crate) fn validate_sampling(pattern: PupilPattern, rings: usize) -> MkResult<()> {
    let _ = pattern;
    check_count("rings", rings, 1, 512)
}

/// Spot diagram (module docs). Equivalent to [`spot_diagram_job`] with a
/// fresh context.
pub fn spot_diagram(
    tracer: &SequentialTracer,
    settings: &SpotDiagramSettings,
) -> MkResult<SpotDiagram> {
    spot_diagram_job(tracer, settings, &JobContext::new())
}

pub(crate) fn validate_spot_settings(settings: &SpotDiagramSettings) -> MkResult<()> {
    validate_sampling(settings.pattern, settings.rings)?;
    check_finite("defocus", settings.defocus)?;
    if let Some(tf) = &settings.through_focus {
        check_count("through-focus steps", tf.steps, 1, 100_000)?;
        if let Some(v) = tf.min {
            check_finite("through-focus range", v)?;
        }
        if let Some(v) = tf.max {
            check_finite("through-focus range", v)?;
        }
    }
    Ok(())
}

/// [`spot_diagram`] with cancellation and progress.
pub fn spot_diagram_job(
    tracer: &SequentialTracer,
    settings: &SpotDiagramSettings,
    ctx: &JobContext,
) -> MkResult<SpotDiagram> {
    spot_job(tracer, settings, ctx, false)
}

/// [`spot_diagram_job`] with the ray points in [`SpotDiagram::packed_points`]
/// (always set, empty without `include_points`) instead of
/// `FieldSpot::points`.
pub fn spot_diagram_packed_job(
    tracer: &SequentialTracer,
    settings: &SpotDiagramSettings,
    ctx: &JobContext,
) -> MkResult<SpotDiagram> {
    spot_job(tracer, settings, ctx, true)
}

fn spot_job(
    tracer: &SequentialTracer,
    settings: &SpotDiagramSettings,
    ctx: &JobContext,
    packed: bool,
) -> MkResult<SpotDiagram> {
    let fields_tracer = crate::util::default_fields(tracer, &settings.fields)?;
    let tracer = fields_tracer.as_ref();
    let timer = RunTimer::start();
    let sys = tracer.system();
    validate_spot_settings(settings)?;
    let fields = settings.fields.resolve(tracer.fields().len())?;
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let pupil = pupil_points(settings.pattern, settings.rings, settings.seed);
    let mut warnings = Vec::new();
    let batch = trace_field_rays(tracer, &fields, &wls, &pupil, ctx, &mut warnings)?;
    let out = spot_reduce(
        timer,
        tracer,
        settings,
        &batch,
        &BackendInfo::cpu_reference(),
        warnings,
        ctx,
        packed,
    )?;
    ctx.report(1.0, "spot diagram");
    Ok(out)
}

/// The spot diagram of a ray batch traced by any backend (module docs).
/// The batch's fields and wavelengths replace the settings' selections. The
/// reference, defocus, through-focus and point options apply as in
/// [`spot_diagram`], and `backend` is recorded in the `RunInfo`.
pub fn spot_from_rays(
    tracer: &SequentialTracer,
    settings: &SpotDiagramSettings,
    batch: &RayBatch,
    backend: &BackendInfo,
    warnings: Vec<String>,
) -> MkResult<SpotDiagram> {
    let timer = RunTimer::start();
    validate_spot_settings(settings)?;
    let sys = tracer.system();
    batch.validate(tracer.fields().len(), sys.wavelengths_um.len())?;
    spot_reduce(
        timer,
        tracer,
        settings,
        batch,
        backend,
        warnings,
        &JobContext::new(),
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn spot_reduce(
    timer: RunTimer,
    tracer: &SequentialTracer,
    settings: &SpotDiagramSettings,
    batch: &RayBatch,
    backend: &BackendInfo,
    mut warnings: Vec<String>,
    ctx: &JobContext,
    packed: bool,
) -> MkResult<SpotDiagram> {
    let sys = tracer.system();
    let wls = &batch.wavelengths;
    let mut packed = packed.then(|| {
        let n = if settings.include_points {
            batch.fields.iter().map(|f| f.rays.len()).sum()
        } else {
            0
        };
        PackedSpotPoints::with_capacity(n, batch.fields.len())
    });
    let delta = settings.defocus;
    if delta != 0.0 || settings.through_focus.is_some() {
        warnings.extend(curved_image_warning(tracer));
    }
    let weights: Vec<f64> = wls.iter().map(|&w| sys.wavelength_weights[w]).collect();
    let fo = tracer.paraxial().primary();
    let planes = settings.through_focus.map(|tf| {
        let max_rms = batch
            .fields
            .iter()
            .map(|f| f.moments(delta).rms_centroid)
            .filter(|v| v.is_finite())
            .fold(0.0, f64::max);
        tf.planes(ThroughFocus::auto_half_range(
            sys.wavelengths_um[sys.primary_wavelength],
            fo.working_f_number,
            max_rms,
        ))
    });

    let mut out = Vec::with_capacity(batch.fields.len());
    for (k, fr) in batch.fields.iter().enumerate() {
        ctx.check()?;
        let m = fr.moments(delta);
        let total: f64 = fr
            .launched
            .iter()
            .zip(&weights)
            .map(|(&n, w)| n as f64 * w)
            .sum();
        let missing: f64 = fr
            .launched
            .iter()
            .zip(&fr.arrived)
            .zip(&weights)
            .map(|((&n, &a), w)| (n - a) as f64 * w)
            .sum();
        let poly = stats(
            &m,
            fr.launched.iter().sum::<usize>() as f64,
            fr.rays.len(),
            missing,
            total,
            fr.chief_ok,
        );
        // Absolute chief/centroid at this defocus.
        let c_abs = [
            delta.mul_add(fr.origin_slope[0], fr.origin[0]),
            delta.mul_add(fr.origin_slope[1], fr.origin[1]),
        ];
        let chief = if fr.chief_ok { c_abs } else { [f64::NAN; 2] };
        let centroid = [c_abs[0] + m.centroid[0], c_abs[1] + m.centroid[1]];
        let (ref_rel, rms, geo) = match settings.reference {
            SpotReference::Chief if fr.chief_ok => ([0.0, 0.0], m.rms_origin, m.geo_origin),
            SpotReference::Chief => ([f64::NAN; 2], f64::NAN, f64::NAN),
            SpotReference::Centroid => (m.centroid, m.rms_centroid, m.geo_centroid),
        };
        let reference = [c_abs[0] + ref_rel[0], c_abs[1] + ref_rel[1]];
        // The reference's own slope: the chief's is the origin of every
        // slope difference; the centroid's is the weighted mean.
        let ref_slope = match settings.reference {
            SpotReference::Centroid => Moments::of(fr.rays.iter().map(|r| (r.dt, r.weight))).centroid,
            SpotReference::Chief => [0.0, 0.0],
        };
        let points = if let Some(p) = packed.as_mut() {
            let start = p.len();
            if settings.include_points {
                let with_slope = planes.is_some();
                for chunk in fr.rays.chunks(POINT_CHUNK) {
                    ctx.check()?;
                    for r in chunk {
                        let q = r.at(delta);
                        p.push(k, q[0] - ref_rel[0], q[1] - ref_rel[1], wls[r.wl_pos]);
                        if with_slope {
                            p.push_slope(r.dt[0] - ref_slope[0], r.dt[1] - ref_slope[1]);
                        }
                    }
                }
            }
            p.close(start);
            Vec::new()
        } else if settings.include_points {
            fr.rays
                .iter()
                .map(|r| {
                    let p = r.at(delta);
                    SpotPoint {
                        x: p[0] - ref_rel[0],
                        y: p[1] - ref_rel[1],
                        wavelength: wls[r.wl_pos],
                    }
                })
                .collect()
        } else {
            Vec::new()
        };
        let per_wavelength = wls
            .iter()
            .enumerate()
            .map(|(k, &w)| {
                let mw = fr.moments_wl(k, delta);
                let (cq, cdt) = fr.wl_chief[k];
                let ok = cq[0].is_finite();
                let n = fr.launched[k];
                WavelengthSpot {
                    wavelength: w,
                    wavelength_um: sys.wavelengths_um[w],
                    weight: sys.wavelength_weights[w],
                    chief: [
                        c_abs[0] + delta.mul_add(cdt[0], cq[0]),
                        c_abs[1] + delta.mul_add(cdt[1], cq[1]),
                    ],
                    stats: stats(
                        &mw,
                        n as f64,
                        fr.arrived[k],
                        (n - fr.arrived[k]) as f64,
                        n as f64,
                        ok,
                    ),
                }
            })
            .collect();
        let through_focus = planes
            .as_ref()
            .map(|planes| {
                planes
                    .iter()
                    .map(|&d| {
                        let m = fr.moments(d);
                        FocusSample {
                            defocus: d,
                            rms_radius_centroid: m.rms_centroid,
                            rms_radius_chief: if fr.chief_ok { m.rms_origin } else { f64::NAN },
                        }
                    })
                    .collect()
            })
            .unwrap_or_default();
        out.push(FieldSpot {
            field: fr.field,
            field_value: field_value(tracer, fr.field),
            points,
            reference,
            chief,
            centroid,
            rms_radius: rms,
            geo_radius: geo,
            polychromatic: poly,
            per_wavelength,
            through_focus,
        });
    }
    let airy = 1.22 * sys.wavelengths_um[sys.primary_wavelength] * 1e-3 * fo.working_f_number;
    let seed = matches!(
        settings.pattern,
        PupilPattern::Random | PupilPattern::Dithered
    )
    .then_some(settings.seed);
    Ok(SpotDiagram {
        fields: out,
        wavelengths_um: sys.wavelengths_um.clone(),
        wavelengths: wls.clone(),
        airy_radius: airy,
        defocus: delta,
        run_info: run_info_backend(
            timer,
            tracer,
            "spot_diagram",
            settings,
            seed,
            warnings,
            backend,
        ),
        packed_points: packed,
    })
}

impl SpotDiagram {
    /// The spot of system field `field`, if it was computed.
    pub fn field(&self, field: usize) -> Option<&FieldSpot> {
        self.fields.iter().find(|f| f.field == field)
    }
}

impl FieldSpot {
    /// Per-wavelength data of system wavelength `w`, if selected.
    pub fn wavelength(&self, w: usize) -> Option<&WavelengthSpot> {
        self.per_wavelength.iter().find(|s| s.wavelength == w)
    }
}
