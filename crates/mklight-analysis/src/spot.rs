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
//! image surface it is the first-order translation of the surface. Positive
//! `δ` is along the image frame's `+z`.
//!
//! The Airy radius is `1.22 λ N_w` with the primary wavelength and the
//! paraxial working F/#.

use mklight_core::job::RunTimer;
use mklight_core::{JobContext, MkResult, RunInfo};
use mklight_trace::{Grid, SequentialTracer, trace_grid_job};
use serde::{Deserialize, Serialize};

use crate::util::{
    FieldSelection, Moments, PupilPattern, WavelengthSelection, check_count, check_finite,
    field_value, linspace, nf, pupil_points, run_info,
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
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ThroughFocus {
    pub min: f64,
    pub max: f64,
    pub steps: usize,
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
}

// ----- Shared ray data -----------------------------------------------------------

/// One arrived ray, relative to its field's primary chief ray.
#[derive(Clone, Copy, Debug)]
pub(crate) struct RayHit {
    /// Position in the list of selected wavelengths.
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

/// Traced rays of one field.
#[derive(Clone, Debug)]
pub(crate) struct FieldRays {
    pub field: usize,
    /// Absolute primary chief hit (image local), or the first arrived ray
    /// when the chief failed (`chief_ok = false`).
    pub origin: [f64; 2],
    pub origin_slope: [f64; 2],
    pub chief_ok: bool,
    /// Chief of each selected wavelength relative to `origin`: `(q, dt)`,
    /// NaN when it failed.
    pub wl_chief: Vec<([f64; 2], [f64; 2])>,
    pub rays: Vec<RayHit>,
    pub launched: Vec<usize>,
    pub arrived: Vec<usize>,
}

impl FieldRays {
    /// Polychromatic moments about the primary chief at defocus `delta`.
    pub fn moments(&self, delta: f64) -> Moments {
        Moments::of(self.rays.iter().map(move |r| (r.at(delta), r.weight)))
    }

    /// Moments of one selected wavelength (position `k`) about its own chief.
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
}

/// Image-space slope `(d_x/d_z, d_y/d_z)`; `None` for a ray parallel to the
/// image frame's `xy` plane or non-finite data.
#[inline]
pub(crate) fn slope(d: mklight_core::Vec3) -> Option<[f64; 2]> {
    let t = [d.x / d.z, d.y / d.z];
    (d.z != 0.0 && t[0].is_finite() && t[1].is_finite()).then_some(t)
}

/// Trace `fields × wavelengths × pupil` (clipped) and the chief rays, and
/// collect the arrived rays relative to each field's primary chief ray.
pub(crate) fn trace_field_rays(
    tracer: &SequentialTracer,
    fields: &[usize],
    wls: &[usize],
    pupil: &[[f64; 2]],
    ctx: &JobContext,
    warnings: &mut Vec<String>,
) -> MkResult<Vec<FieldRays>> {
    let sys = tracer.system();
    let primary = sys.primary_wavelength;
    let prepared: Vec<_> = fields.iter().map(|&f| tracer.field(f).clone()).collect();
    let grid = Grid {
        fields: &prepared,
        wavelengths: wls,
        pupil,
    };
    let rays = trace_grid_job(tracer, &grid, &tracer.options(), &ctx.subrange(0.0, 0.95))?;
    let np = pupil.len();
    let mut out = Vec::with_capacity(fields.len());
    for (fi, &f) in fields.iter().enumerate() {
        ctx.check()?;
        let pf = &prepared[fi];
        let chief_hit = |wl: usize| {
            let c = tracer.chief(pf, wl);
            if !c.status.is_ok() {
                return None;
            }
            slope(c.ray.dir).map(|t| ([c.ray.pos.x, c.ray.pos.y], t))
        };
        let pc = chief_hit(primary);
        let block = &rays[fi * wls.len() * np..(fi + 1) * wls.len() * np];
        let first = block
            .iter()
            .find(|r| r.status.is_ok() && slope(r.dir).is_some());
        let (origin, origin_slope, chief_ok) = match (pc, first) {
            (Some((p, t)), _) => (p, t, true),
            (None, Some(r)) => {
                warnings.push(format!(
                    "field {f}: primary chief ray failed; statistics about the chief are undefined"
                ));
                (
                    [r.point.x, r.point.y],
                    slope(r.dir).unwrap_or([0.0, 0.0]),
                    false,
                )
            }
            (None, None) => ([0.0, 0.0], [0.0, 0.0], false),
        };
        let rel = |p: [f64; 2], t: [f64; 2]| {
            (
                [p[0] - origin[0], p[1] - origin[1]],
                [t[0] - origin_slope[0], t[1] - origin_slope[1]],
            )
        };
        let wl_chief = wls
            .iter()
            .map(|&w| {
                let c = if w == primary { pc } else { chief_hit(w) };
                match c {
                    Some((p, t)) if chief_ok || w != primary => rel(p, t),
                    _ => ([f64::NAN; 2], [f64::NAN; 2]),
                }
            })
            .collect();
        let mut hits = Vec::with_capacity(block.len());
        let mut launched = vec![0; wls.len()];
        let mut arrived = vec![0; wls.len()];
        for r in block {
            let k = wls
                .iter()
                .position(|&w| w as u32 == r.wavelength)
                .expect("grid wavelength");
            launched[k] += 1;
            if !r.status.is_ok() {
                continue;
            }
            let Some(t) = slope(r.dir) else { continue };
            arrived[k] += 1;
            let (q, dt) = rel([r.point.x, r.point.y], t);
            hits.push(RayHit {
                wl_pos: k,
                q,
                dt,
                weight: sys.wavelength_weights[wls[k]],
            });
        }
        out.push(FieldRays {
            field: f,
            origin,
            origin_slope,
            chief_ok,
            wl_chief,
            rays: hits,
            launched,
            arrived,
        });
    }
    Ok(out)
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

/// [`spot_diagram`] with cancellation and progress.
pub fn spot_diagram_job(
    tracer: &SequentialTracer,
    settings: &SpotDiagramSettings,
    ctx: &JobContext,
) -> MkResult<SpotDiagram> {
    let timer = RunTimer::start();
    let sys = tracer.system();
    validate_sampling(settings.pattern, settings.rings)?;
    check_finite("defocus", settings.defocus)?;
    if let Some(tf) = &settings.through_focus {
        check_count("through-focus steps", tf.steps, 1, 100_000)?;
        check_finite("through-focus range", tf.min)?;
        check_finite("through-focus range", tf.max)?;
    }
    let fields = settings.fields.resolve(tracer.fields().len())?;
    let wls = settings
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let pupil = pupil_points(settings.pattern, settings.rings, settings.seed);
    let mut warnings = Vec::new();
    let data = trace_field_rays(tracer, &fields, &wls, &pupil, ctx, &mut warnings)?;
    let delta = settings.defocus;
    let weights: Vec<f64> = wls.iter().map(|&w| sys.wavelength_weights[w]).collect();

    let mut out = Vec::with_capacity(data.len());
    for fr in &data {
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
        let points = if settings.include_points {
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
        let through_focus = settings
            .through_focus
            .map(|tf| {
                linspace(tf.min, tf.max, tf.steps)
                    .into_iter()
                    .map(|d| {
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
    ctx.report(1.0, "spot diagram");
    let fo = tracer.paraxial().primary();
    let airy = 1.22 * sys.wavelengths_um[sys.primary_wavelength] * 1e-3 * fo.working_f_number;
    let seed = matches!(
        settings.pattern,
        PupilPattern::Random | PupilPattern::Dithered
    )
    .then_some(settings.seed);
    Ok(SpotDiagram {
        fields: out,
        wavelengths_um: sys.wavelengths_um.clone(),
        wavelengths: wls,
        airy_radius: airy,
        defocus: delta,
        run_info: run_info(timer, tracer, "spot_diagram", settings, seed, warnings),
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
