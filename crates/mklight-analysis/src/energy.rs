//! Geometric encircled / ensquared energy, geometric MTF and best focus, all
//! from spot-diagram ray data (diffraction-free).
//!
//! Rays are traced once per field over a pupil pattern (as in
//! [`crate::spot`]); every arrived ray carries its wavelength's weight, and
//! each pattern point stands for an equal share of the pupil area (use the
//! `Square`, `Random` or `Dithered` patterns for area-uniform sampling;
//! `Hexapolar` is nearly uniform). Energies are normalised to the weight of
//! the rays that **arrive** (vignetted rays carry no energy). A defocus `δ`
//! moves each image point along its ray, as in [`crate::spot`].
//!
//! # Encircled and ensquared energy
//!
//! `EE(r) = Σ w [|q − c| ≤ r] / Σ w` about the reference `c` (centroid or
//! chief ray); ensquared energy uses the square of half-width `r`,
//! `max(|Δx|, |Δy|) ≤ r`. The radii containing 50 / 80 / 90 % are the
//! exact order statistics of the weighted distances (smallest distance at
//! which the cumulative weight reaches the fraction).
//!
//! # Geometric MTF
//!
//! The geometric optical transfer function is the Fourier transform of the
//! spot's ray density: `MTF(ν) = |Σ w e^{−2πiν q·e}| / Σ w` with `e` the
//! tangential direction (the field's radial direction in the image plane,
//! `+y` on axis; CONVENTIONS §4) or the sagittal one perpendicular to it.
//! Positions are taken relative to the centroid and summed with compensation.
//! For a uniform disk of radius `R` (pure defocus of an aberration-free
//! lens) this is `2 J₁(2πνR)/(2πνR)`. No diffraction is included.
//!
//! # Best focus
//!
//! The merit is the field-weighted mean of the polychromatic RMS spot radii,
//! `M(δ) = Σ_f w_f RMS_f(δ) / Σ_f w_f`, about the centroid (default) or the
//! chief ray. Each `RMS_f(δ)` is the norm of an affine function of `δ`, so
//! `M` is convex and its minimum lies between the smallest and largest
//! per-field minimisers `−B_f/C_f` (with `RMS_f² = A_f + 2B_f δ + C_f δ²`):
//! that interval (padded) is the default search range. Brent's method
//! (golden section with parabolic steps) then minimises `M`, evaluated from
//! the rays with two-pass moments (the quadratic form would cancel near a
//! sharp focus). The minimiser of
//! the mean *square* merit, `−Σw_f B_f / Σw_f C_f`, is reported as a
//! cross-check. The image is moved by transfer along the rays, which is
//! exact for a plane image surface (and first order for a curved one).

use std::f64::consts::PI;

use mklight_core::job::RunTimer;
use mklight_core::math::KahanSum;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::SequentialTracer;
use serde::{Deserialize, Serialize};

use crate::spot::{FieldRays, SpotReference, trace_field_rays, validate_sampling};
use crate::util::{
    FieldSelection, PupilPattern, WavelengthSelection, brent_minimize, check_count, check_finite,
    field_value, linspace, nf, pupil_points, run_info,
};

fn seed_of(pattern: PupilPattern, seed: u64) -> Option<u64> {
    matches!(pattern, PupilPattern::Random | PupilPattern::Dithered).then_some(seed)
}

/// Common ray sampling of the energy analyses.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RaySampling {
    pub pattern: PupilPattern,
    pub rings: usize,
    pub fields: FieldSelection,
    pub wavelengths: WavelengthSelection,
    pub seed: u64,
}

impl Default for RaySampling {
    fn default() -> Self {
        RaySampling {
            pattern: PupilPattern::Square,
            rings: 32,
            fields: FieldSelection::All,
            wavelengths: WavelengthSelection::All,
            seed: 1,
        }
    }
}

fn trace(
    tracer: &SequentialTracer,
    s: &RaySampling,
    ctx: &JobContext,
    warnings: &mut Vec<String>,
) -> MkResult<(Vec<FieldRays>, Vec<usize>)> {
    validate_sampling(s.pattern, s.rings)?;
    let sys = tracer.system();
    let fields = s.fields.resolve(tracer.fields().len())?;
    let wls = s
        .wavelengths
        .resolve(sys.wavelengths_um.len(), sys.primary_wavelength)?;
    let pupil = pupil_points(s.pattern, s.rings, s.seed);
    let data = trace_field_rays(tracer, &fields, &wls, &pupil, ctx, warnings)?;
    Ok((data, wls))
}

/// Points relative to the reference at defocus `delta`, with weights.
fn relative_points(fr: &FieldRays, delta: f64, reference: SpotReference) -> Vec<([f64; 2], f64)> {
    let c = match reference {
        SpotReference::Centroid => fr.moments(delta).centroid,
        SpotReference::Chief => [0.0, 0.0],
    };
    fr.rays
        .iter()
        .map(|r| {
            let p = r.at(delta);
            ([p[0] - c[0], p[1] - c[1]], r.weight)
        })
        .collect()
}

// ----- Encircled / ensquared energy ---------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct EncircledEnergySettings {
    #[serde(flatten)]
    pub sampling: RaySampling,
    pub reference: SpotReference,
    pub defocus: f64,
    /// Radius samples from 0 to `max_radius`.
    pub samples: usize,
    /// Largest plotted radius (mm); the largest GEO radius over the fields
    /// when `None`.
    pub max_radius: Option<f64>,
}

impl Default for EncircledEnergySettings {
    fn default() -> Self {
        EncircledEnergySettings {
            sampling: RaySampling::default(),
            reference: SpotReference::Centroid,
            defocus: 0.0,
            samples: 101,
            max_radius: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldEnergy {
    pub field: usize,
    #[serde(with = "nf::arr2")]
    pub field_value: [f64; 2],
    /// Fraction of the arrived energy inside radius `radius[i]`.
    pub encircled: Vec<f64>,
    /// Fraction inside the square of half-width `radius[i]`.
    pub ensquared: Vec<f64>,
    /// Radii (mm) enclosing 50 / 80 / 90 % (encircled).
    #[serde(with = "nf")]
    pub r50: f64,
    #[serde(with = "nf")]
    pub r80: f64,
    #[serde(with = "nf")]
    pub r90: f64,
    /// Half-width (mm) of the square enclosing 80 %.
    #[serde(with = "nf")]
    pub half_width80: f64,
    pub rays_arrived: usize,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EncircledEnergyResult {
    /// Radii / half-widths (mm), shared by every field.
    pub radius: Vec<f64>,
    pub fields: Vec<FieldEnergy>,
    pub wavelengths: Vec<usize>,
    pub run_info: RunInfo,
}

/// Sorted `(distance, weight)` and the smallest distance whose cumulative
/// weight reaches each fraction.
struct Cumulative {
    d: Vec<f64>,
    cum: Vec<f64>,
}

impl Cumulative {
    fn new(mut v: Vec<(f64, f64)>) -> Self {
        v.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut s = KahanSum::default();
        let mut d = Vec::with_capacity(v.len());
        let mut cum = Vec::with_capacity(v.len());
        for (x, w) in v {
            s.add(w);
            d.push(x);
            cum.push(s.value());
        }
        Cumulative { d, cum }
    }
    fn total(&self) -> f64 {
        self.cum.last().copied().unwrap_or(0.0)
    }
    /// Fraction of the weight at distance `≤ r`.
    fn fraction(&self, r: f64) -> f64 {
        let k = self.d.partition_point(|&x| x <= r);
        if k == 0 || self.total() <= 0.0 {
            0.0
        } else {
            self.cum[k - 1] / self.total()
        }
    }
    fn quantile(&self, p: f64) -> f64 {
        let t = self.total();
        if t <= 0.0 {
            return f64::NAN;
        }
        let k = self.cum.partition_point(|&c| c < p * t * (1.0 - 1e-15));
        self.d.get(k).copied().unwrap_or(f64::NAN)
    }
}

/// Encircled and ensquared energy (module docs).
pub fn encircled_energy(
    tracer: &SequentialTracer,
    settings: &EncircledEnergySettings,
) -> MkResult<EncircledEnergyResult> {
    encircled_energy_job(tracer, settings, &JobContext::new())
}

/// [`encircled_energy`] with cancellation and progress.
pub fn encircled_energy_job(
    tracer: &SequentialTracer,
    settings: &EncircledEnergySettings,
    ctx: &JobContext,
) -> MkResult<EncircledEnergyResult> {
    let timer = RunTimer::start();
    check_count("radius samples", settings.samples, 2, 100_001)?;
    check_finite("defocus", settings.defocus)?;
    let mut warnings = Vec::new();
    let (data, wls) = trace(tracer, &settings.sampling, ctx, &mut warnings)?;
    let delta = settings.defocus;
    let per: Vec<(Cumulative, Cumulative, usize, usize)> = data
        .iter()
        .map(|fr| {
            let pts = relative_points(fr, delta, settings.reference);
            let circ = pts.iter().map(|(q, w)| (q[0].hypot(q[1]), *w)).collect();
            let sq = pts
                .iter()
                .map(|(q, w)| (q[0].abs().max(q[1].abs()), *w))
                .collect();
            (
                Cumulative::new(circ),
                Cumulative::new(sq),
                fr.field,
                fr.rays.len(),
            )
        })
        .collect();
    let rmax = match settings.max_radius {
        Some(r) if r > 0.0 && r.is_finite() => r,
        Some(_) => {
            return Err(MkError::InvalidModel(
                "maximum radius must be positive".into(),
            ));
        }
        None => per
            .iter()
            .filter_map(|p| p.0.d.last().copied())
            .fold(0.0, f64::max),
    };
    let radius = linspace(0.0, rmax, settings.samples);
    let fields = per
        .iter()
        .map(|(c, s, f, n)| FieldEnergy {
            field: *f,
            field_value: field_value(tracer, *f),
            encircled: radius.iter().map(|&r| c.fraction(r)).collect(),
            ensquared: radius.iter().map(|&r| s.fraction(r)).collect(),
            r50: c.quantile(0.5),
            r80: c.quantile(0.8),
            r90: c.quantile(0.9),
            half_width80: s.quantile(0.8),
            rays_arrived: *n,
        })
        .collect();
    ctx.report(1.0, "encircled energy");
    Ok(EncircledEnergyResult {
        radius,
        fields,
        wavelengths: wls,
        run_info: run_info(
            timer,
            tracer,
            "encircled_energy",
            settings,
            seed_of(settings.sampling.pattern, settings.sampling.seed),
            warnings,
        ),
    })
}

// ----- Geometric MTF --------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GeometricMtfSettings {
    #[serde(flatten)]
    pub sampling: RaySampling,
    pub defocus: f64,
    /// Highest frequency (cycles/mm); the diffraction cutoff `1/(λ N_w)` at
    /// the primary wavelength when `None` (100 cycles/mm if that is undefined).
    pub max_frequency: Option<f64>,
    pub frequency_samples: usize,
}

impl Default for GeometricMtfSettings {
    fn default() -> Self {
        GeometricMtfSettings {
            sampling: RaySampling::default(),
            defocus: 0.0,
            max_frequency: None,
            frequency_samples: 41,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MtfOrientation {
    Tangential,
    Sagittal,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MtfCurve {
    pub field: usize,
    pub orientation: MtfOrientation,
    #[serde(with = "nf::vec")]
    pub modulation: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometricMtfResult {
    /// Spatial frequencies (cycles/mm).
    pub frequencies: Vec<f64>,
    pub curves: Vec<MtfCurve>,
    /// Always `None`: the geometric MTF has no diffraction limit.
    pub diffraction_limit: Option<Vec<f64>>,
    /// Incoherent diffraction cutoff `1/(λ N_w)` at the primary wavelength,
    /// for reference (cycles/mm).
    #[serde(with = "nf")]
    pub cutoff_frequency: f64,
    /// Field value of each computed field, in the order of `curves`.
    #[serde(with = "nf::vec2")]
    pub field_values: Vec<[f64; 2]>,
    pub run_info: RunInfo,
}

impl GeometricMtfResult {
    pub fn curve(&self, field: usize, orientation: MtfOrientation) -> Option<&MtfCurve> {
        self.curves
            .iter()
            .find(|c| c.field == field && c.orientation == orientation)
    }
}

/// `|Σ w e^{−2πiν x}| / Σ w` for every `ν`.
fn modulation(x: &[(f64, f64)], freqs: &[f64]) -> Vec<f64> {
    let wt: KahanSum = x.iter().map(|p| p.1).collect();
    let wt = wt.value();
    freqs
        .iter()
        .map(|&nu| {
            if !(wt > 0.0) {
                return f64::NAN;
            }
            let mut re = KahanSum::default();
            let mut im = KahanSum::default();
            for &(v, w) in x {
                let (s, c) = (2.0 * PI * nu * v).sin_cos();
                re.add(w * c);
                im.add(w * s);
            }
            re.value().hypot(im.value()) / wt
        })
        .collect()
}

/// Geometric MTF (module docs).
pub fn geometric_mtf(
    tracer: &SequentialTracer,
    settings: &GeometricMtfSettings,
) -> MkResult<GeometricMtfResult> {
    geometric_mtf_job(tracer, settings, &JobContext::new())
}

/// [`geometric_mtf`] with cancellation and progress.
pub fn geometric_mtf_job(
    tracer: &SequentialTracer,
    settings: &GeometricMtfSettings,
    ctx: &JobContext,
) -> MkResult<GeometricMtfResult> {
    let timer = RunTimer::start();
    check_count("frequency samples", settings.frequency_samples, 2, 100_001)?;
    check_finite("defocus", settings.defocus)?;
    let sys = tracer.system();
    let fo = tracer.paraxial().primary();
    let cutoff = 1.0 / (sys.wavelengths_um[sys.primary_wavelength] * 1e-3 * fo.working_f_number);
    let fmax = match settings.max_frequency {
        Some(f) if f > 0.0 && f.is_finite() => f,
        Some(_) => {
            return Err(MkError::InvalidModel(
                "maximum frequency must be positive".into(),
            ));
        }
        None if cutoff.is_finite() && cutoff > 0.0 => cutoff,
        None => 100.0,
    };
    let mut warnings = Vec::new();
    let (data, _) = trace(
        tracer,
        &settings.sampling,
        &ctx.subrange(0.0, 0.8),
        &mut warnings,
    )?;
    let freqs = linspace(0.0, fmax, settings.frequency_samples);
    let mut curves = Vec::with_capacity(2 * data.len());
    let mut field_values = Vec::with_capacity(data.len());
    for (k, fr) in data.iter().enumerate() {
        ctx.check()?;
        let p = tracer.field(fr.field).point;
        let h = p.hx.hypot(p.hy);
        let et = if h > 0.0 {
            [p.hx / h, p.hy / h]
        } else {
            [0.0, 1.0]
        };
        let es = [et[1], -et[0]];
        let pts = relative_points(fr, settings.defocus, SpotReference::Centroid);
        for (e, o) in [
            (et, MtfOrientation::Tangential),
            (es, MtfOrientation::Sagittal),
        ] {
            let x: Vec<(f64, f64)> = pts
                .iter()
                .map(|(q, w)| (q[0].mul_add(e[0], q[1] * e[1]), *w))
                .collect();
            curves.push(MtfCurve {
                field: fr.field,
                orientation: o,
                modulation: modulation(&x, &freqs),
            });
        }
        field_values.push(field_value(tracer, fr.field));
        ctx.report(
            0.8 + 0.2 * (k + 1) as f64 / data.len() as f64,
            "geometric MTF",
        );
    }
    Ok(GeometricMtfResult {
        frequencies: freqs,
        curves,
        diffraction_limit: None,
        cutoff_frequency: cutoff,
        field_values,
        run_info: run_info(
            timer,
            tracer,
            "geometric_mtf",
            settings,
            seed_of(settings.sampling.pattern, settings.sampling.seed),
            warnings,
        ),
    })
}

// ----- Best focus -------------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct BestFocusSettings {
    #[serde(flatten)]
    pub sampling: RaySampling,
    pub reference: SpotReference,
    /// Defocus search range (mm); automatic when `None` (module docs).
    pub range: Option<[f64; 2]>,
    /// Absolute tolerance on the defocus (mm).
    pub tolerance: f64,
    pub max_evaluations: usize,
    /// Samples of the through-focus merit curve over the range (0: none).
    pub through_focus_samples: usize,
}

impl Default for BestFocusSettings {
    fn default() -> Self {
        BestFocusSettings {
            sampling: RaySampling {
                pattern: PupilPattern::Hexapolar,
                rings: 8,
                ..Default::default()
            },
            reference: SpotReference::Centroid,
            range: None,
            tolerance: 1e-8,
            max_evaluations: 200,
            through_focus_samples: 0,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BestFocusResult {
    /// Best defocus from the image surface (mm, image local `+z`).
    pub defocus: f64,
    /// Field-weighted mean RMS radius at the best focus (mm).
    pub merit: f64,
    /// The merit at the current image surface (mm).
    pub merit_at_zero: f64,
    pub fields: Vec<usize>,
    /// Per-field RMS radius at the best focus and at zero defocus (mm).
    #[serde(with = "nf::vec")]
    pub field_rms: Vec<f64>,
    #[serde(with = "nf::vec")]
    pub field_rms_at_zero: Vec<f64>,
    /// Minimiser of the field-weighted mean *square* RMS (closed form).
    pub mean_square_defocus: f64,
    pub range: [f64; 2],
    pub evaluations: usize,
    pub converged: bool,
    /// Through-focus merit curve (empty unless requested).
    pub through_focus_defocus: Vec<f64>,
    pub through_focus_merit: Vec<f64>,
    pub run_info: RunInfo,
}

/// Quadratic coefficients `(A, B, C)` of `RMS²(δ) = A + 2Bδ + Cδ²`.
fn quadratic(fr: &FieldRays, reference: SpotReference) -> Option<(f64, f64, f64)> {
    let mut sw = KahanSum::default();
    let mut s = [KahanSum::default(); 4];
    for r in &fr.rays {
        sw.add(r.weight);
        s[0].add(r.weight * r.q[0]);
        s[1].add(r.weight * r.q[1]);
        s[2].add(r.weight * r.dt[0]);
        s[3].add(r.weight * r.dt[1]);
    }
    let w = sw.value();
    if !(w > 0.0) {
        return None;
    }
    let (mq, mt) = match reference {
        SpotReference::Centroid => (
            [s[0].value() / w, s[1].value() / w],
            [s[2].value() / w, s[3].value() / w],
        ),
        SpotReference::Chief => ([0.0; 2], [0.0; 2]),
    };
    let (mut a, mut b, mut c) = (
        KahanSum::default(),
        KahanSum::default(),
        KahanSum::default(),
    );
    for r in &fr.rays {
        let q = [r.q[0] - mq[0], r.q[1] - mq[1]];
        let t = [r.dt[0] - mt[0], r.dt[1] - mt[1]];
        a.add(r.weight * q[0].mul_add(q[0], q[1] * q[1]));
        b.add(r.weight * q[0].mul_add(t[0], q[1] * t[1]));
        c.add(r.weight * t[0].mul_add(t[0], t[1] * t[1]));
    }
    Some((a.value() / w, b.value() / w, c.value() / w))
}

/// Best focus (module docs).
pub fn best_focus(
    tracer: &SequentialTracer,
    settings: &BestFocusSettings,
) -> MkResult<BestFocusResult> {
    best_focus_job(tracer, settings, &JobContext::new())
}

/// [`best_focus`] with cancellation and progress.
pub fn best_focus_job(
    tracer: &SequentialTracer,
    settings: &BestFocusSettings,
    ctx: &JobContext,
) -> MkResult<BestFocusResult> {
    let timer = RunTimer::start();
    check_finite("tolerance", settings.tolerance)?;
    if !(settings.tolerance > 0.0) {
        return Err(MkError::InvalidModel("tolerance must be positive".into()));
    }
    check_count("evaluations", settings.max_evaluations, 3, 100_000)?;
    let mut warnings = Vec::new();
    let (data, _) = trace(
        tracer,
        &settings.sampling,
        &ctx.subrange(0.0, 0.9),
        &mut warnings,
    )?;
    let osys = tracer.system().system();
    if !matches!(
        tracer.system().surfaces[tracer.system().image_index()].shape,
        mklight_core::SurfaceShape::Plane
    ) {
        warnings
            .push("curved image surface: defocus is a first-order transfer along the rays".into());
    }
    // Field weights and quadratic coefficients of the fields with rays.
    let mut used = Vec::new();
    for fr in &data {
        if let Some(q) = quadratic(fr, settings.reference) {
            used.push((fr, osys.fields.fields[fr.field].weight, q));
        } else {
            warnings.push(format!("field {}: no rays arrived", fr.field));
        }
    }
    let wsum: f64 = used.iter().map(|u| u.1).sum();
    if used.is_empty() || !(wsum > 0.0) {
        return Err(MkError::Numerical(
            "best focus: no field with arrived rays and positive weight".into(),
        ));
    }
    // RMS from the rays (two-pass moments), not from the quadratic, which
    // would cancel near a sharp focus.
    let rms = |fr: &FieldRays, d: f64| {
        let m = fr.moments(d);
        match settings.reference {
            SpotReference::Centroid => m.rms_centroid,
            SpotReference::Chief => m.rms_origin,
        }
    };
    let merit = |d: f64| {
        used.iter()
            .map(|u| u.1 * rms(u.0, d))
            .collect::<KahanSum>()
            .value()
            / wsum
    };
    // Mean-square minimiser and the per-field minimisers.
    let (num, den) = used
        .iter()
        .fold((0.0, 0.0), |(n, d), u| (n + u.1 * u.2.1, d + u.1 * u.2.2));
    let ms = if den > 0.0 { -num / den } else { 0.0 };
    let range = match settings.range {
        Some([a, b]) => {
            if !(a.is_finite() && b.is_finite() && a < b) {
                return Err(MkError::InvalidModel(
                    "defocus range must be finite and ascending".into(),
                ));
            }
            [a, b]
        }
        None => {
            let mins: Vec<f64> = used
                .iter()
                .filter(|u| u.2.2 > 0.0 && u.1 > 0.0)
                .map(|u| -u.2.1 / u.2.2)
                .collect();
            let lo = mins.iter().copied().fold(ms, f64::min);
            let hi = mins.iter().copied().fold(ms, f64::max);
            let pad = (0.05 * (hi - lo)).max(1e-6 * (1.0 + lo.abs().max(hi.abs())));
            [lo - pad, hi + pad]
        }
    };
    ctx.check()?;
    let br = brent_minimize(
        |d| Ok(merit(d)),
        range[0],
        range[1],
        1e-12,
        settings.tolerance,
        settings.max_evaluations,
    )?;
    let edge = (br.x - range[0]).min(range[1] - br.x) <= 2.0 * settings.tolerance;
    if edge && settings.range.is_some() {
        warnings.push("best focus is at the edge of the search range".into());
    }
    let (tfd, tfm) = if settings.through_focus_samples > 0 {
        let d = linspace(range[0], range[1], settings.through_focus_samples.max(2));
        let m = d.iter().map(|&x| merit(x)).collect();
        (d, m)
    } else {
        (Vec::new(), Vec::new())
    };
    ctx.report(1.0, "best focus");
    Ok(BestFocusResult {
        defocus: br.x,
        merit: br.fx,
        merit_at_zero: merit(0.0),
        fields: used.iter().map(|u| u.0.field).collect(),
        field_rms: used.iter().map(|u| rms(u.0, br.x)).collect(),
        field_rms_at_zero: used.iter().map(|u| rms(u.0, 0.0)).collect(),
        mean_square_defocus: ms,
        range,
        evaluations: br.evaluations,
        converged: br.converged,
        through_focus_defocus: tfd,
        through_focus_merit: tfm,
        run_info: run_info(
            timer,
            tracer,
            "best_focus",
            settings,
            seed_of(settings.sampling.pattern, settings.sampling.seed),
            warnings,
        ),
    })
}
