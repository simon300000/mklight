//! The stray-light analysis entry point used by the engine: settings in,
//! per-class irradiance maps with Monte-Carlo errors, totals, the strongest
//! path signatures and `RunInfo` out.
//!
//! [`analyze`] sizes the system with the sequential tracer and builds the
//! scene ([`StraySystem`]), resolves the source ([`StraySource`]), runs the
//! Monte Carlo ([`crate::tracer::run`]) and assembles a
//! [`StrayAnalysisResult`]. [`analyze_system`] reuses a built
//! [`StraySystem`] (for several sources on one design).
//!
//! # Classes and double counting
//!
//! `classes` selects the classes that make up the reported *stray total*
//! and the maps; every class is still tallied and listed in
//! [`StrayAnalysisResult::totals`]. With `exclude_ghost_class` (set it when
//! the explicit ghost solver reports ghosts for the same run), `Ghost` is
//! removed from the total, so a front end can add ghost-solver and
//! stray-solver irradiance without counting any path twice
//! (`docs/STRAY.md`, "Partition"). Pass the ghost run's coverage record
//! (`ghost_coverage`) as well: the shared apertures are then checked, and
//! the `Ghost`-class paths the ghost run did not trace (higher orders than
//! its `max_reflections`, paths it filtered out) stay in the stray total
//! (`EnergyReport::detected_uncovered_ghost`), so that every ghost order is
//! counted exactly once. Without it the whole class is removed and those
//! orders are counted by neither solver.
//!
//! # Reproducibility
//!
//! The result depends only on the snapshot, the settings and the seed: the
//! sizing traces, the scene and the transport are all deterministic and
//! independent of the number of threads (tested). Progress runs from 0 to
//! 1 over the system build (sizing, scene and obstruction check, 10 %), the
//! source's obstruction check (2 %) and transport; cancellation is checked
//! between the build stages, per grid row of the obstruction checks and
//! every 64 rays of transport.
//!
//! # Validation
//!
//! Settings are JSON with every field optional; unknown keys are rejected
//! (also inside `semi_diameters`). The detector
//! map is limited to [`crate::system::MAX_DETECTOR_PIXELS`], source strengths
//! must be finite, and a fraction of a zero reference flux is `None`.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use mklight_core::error::{MkError, MkResult};
use mklight_core::job::{JobContext, Precision, RunInfo, RunTimer};
use mklight_core::resolved::ResolvedSystem;
use mklight_trace::FieldPoint;

use crate::path::{ClassMask, PathClass};
use crate::reference::primary_reference;
use crate::scene::{DetectorSpec, SceneOptions};
use crate::source::SourceSpec;
use crate::system::{Beam, OBSTRUCTION_WARNING, StraySystem, SystemOptions};
use crate::tally::{EnergyReport, ImageMoments, SignatureFlux};
use crate::tracer::{SplitMode, StraySettings, run};

/// Where the light comes from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum StraySource {
    /// A normalised field point `(Hx, Hy)` (CONVENTIONS §3): a beam that
    /// fills the entrance pupil with a relative `margin`
    /// ([`StraySystem::field_source`]). `strength`: irradiance normal to the
    /// beam (W/mm²) at infinity, intensity (W/sr) for a finite object.
    Field {
        hx: f64,
        hy: f64,
        margin: f64,
        strength: f64,
    },
    /// The system's field `index` (vignetting factors ignored).
    SystemField {
        index: usize,
        margin: f64,
        strength: f64,
    },
    /// A collimated source outside the field of view (for example the sun)
    /// whose beam covers the whole scene ([`StraySystem::out_of_field_source`]).
    OutOfField {
        angle_x_deg: f64,
        angle_y_deg: f64,
        margin: f64,
        /// W/mm² normal to the beam.
        irradiance: f64,
    },
    /// Any source; `reference_flux` normalises the reported fractions.
    Custom {
        spec: SourceSpec,
        reference_flux: f64,
    },
}

impl Default for StraySource {
    fn default() -> Self {
        StraySource::SystemField {
            index: 0,
            margin: 0.1,
            strength: 1.0,
        }
    }
}

impl StraySource {
    /// Resolves the source on a built system.
    pub fn resolve(&self, sys: &StraySystem) -> MkResult<Beam> {
        match self {
            StraySource::Field {
                hx,
                hy,
                margin,
                strength,
            } => sys.field_source(FieldPoint::normalized(*hx, *hy), *margin, *strength),
            StraySource::SystemField {
                index,
                margin,
                strength,
            } => sys.system_field_source(*index, *margin, *strength),
            StraySource::OutOfField {
                angle_x_deg,
                angle_y_deg,
                margin,
                irradiance,
            } => sys.out_of_field_source(*angle_x_deg, *angle_y_deg, *margin, *irradiance),
            StraySource::Custom {
                spec,
                reference_flux,
            } => {
                spec.validate()?;
                if !(reference_flux.is_finite() && *reference_flux >= 0.0) {
                    return Err(MkError::InvalidModel(
                        "custom source: reference flux must be finite and ≥ 0".into(),
                    ));
                }
                Ok(Beam {
                    spec: spec.clone(),
                    reference_flux: *reference_flux,
                    description: "custom source".into(),
                })
            }
        }
    }
}

/// Transport controls (see [`StraySettings`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct TransportOptions {
    pub mode: SplitMode,
    pub max_events: u32,
    pub roulette_threshold: f64,
    pub split_ceiling: f64,
    pub scatter_reflection: bool,
    pub scatter_transmission: bool,
    pub chunk_rays: u32,
}

impl Default for TransportOptions {
    fn default() -> Self {
        let s = StraySettings::default();
        TransportOptions {
            mode: s.mode,
            max_events: s.max_events,
            roulette_threshold: s.roulette_threshold,
            split_ceiling: s.split_ceiling,
            scatter_reflection: s.scatter_reflection,
            scatter_transmission: s.scatter_transmission,
            chunk_rays: s.chunk_rays,
        }
    }
}

/// Settings of [`analyze`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct StrayAnalysisSettings {
    pub source: StraySource,
    /// Source rays.
    pub rays: u64,
    pub seed: u64,
    /// Classes in the stray total and the maps (default: everything but
    /// `Primary`).
    pub classes: ClassMask,
    /// Remove `Ghost` from `classes` because the explicit ghost solver owns
    /// those paths (those of `ghost_coverage` when given, module docs).
    pub exclude_ghost_class: bool,
    /// The coverage record of the ghost run this analysis is combined with.
    /// When given, the analysis fails unless the scene clips at exactly the
    /// ghost run's apertures ([`StraySystem::check_ghost_coverage`]), and
    /// with `exclude_ghost_class` the ghost paths outside it are kept.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ghost_coverage: Option<mklight_ghost::GhostCoverage>,
    /// Detector map size `(nx, ny)`.
    pub detector_pixels: (u32, u32),
    /// Number of path signatures reported.
    pub top_signatures: usize,
    /// Scene construction (its `detector_pixels` is replaced by the field
    /// above).
    pub scene: SceneOptions,
    pub semi_diameters: mklight_trace::SemiDiameterSettings,
    /// Relative margin of the detector over the traced image height
    /// (automatic image surface without a sensor).
    pub detector_margin: f64,
    pub transport: TransportOptions,
}

impl Default for StrayAnalysisSettings {
    fn default() -> Self {
        let sys = SystemOptions::default();
        StrayAnalysisSettings {
            source: StraySource::default(),
            rays: 100_000,
            seed: 1,
            classes: ClassMask::ALL_STRAY,
            exclude_ghost_class: false,
            ghost_coverage: None,
            detector_pixels: (128, 128),
            top_signatures: 20,
            scene: sys.scene,
            semi_diameters: sys.semi_diameters,
            detector_margin: sys.detector_margin,
            transport: TransportOptions::default(),
        }
    }
}

impl StrayAnalysisSettings {
    /// The classes actually included in the totals.
    pub fn effective_classes(&self) -> ClassMask {
        if self.exclude_ghost_class {
            ClassMask(self.classes.0 & !PathClass::Ghost.bit())
        } else {
            self.classes
        }
    }

    pub fn system_options(&self) -> SystemOptions {
        let mut scene = self.scene.clone();
        scene.detector_pixels = self.detector_pixels;
        SystemOptions {
            scene,
            semi_diameters: self.semi_diameters,
            detector_margin: self.detector_margin,
        }
    }

    /// The transport settings of the Monte-Carlo run.
    pub fn stray_settings(&self) -> StraySettings {
        let t = &self.transport;
        StraySettings {
            rays: self.rays,
            seed: self.seed,
            max_events: t.max_events,
            roulette_threshold: t.roulette_threshold,
            split_ceiling: t.split_ceiling,
            mode: t.mode,
            scatter_reflection: t.scatter_reflection,
            scatter_transmission: t.scatter_transmission,
            chunk_rays: t.chunk_rays,
            owned_classes: self.effective_classes(),
            start_medium: None,
            signatures: self.top_signatures > 0,
            // Ghost orders the ghost run did not trace stay with the stray
            // solver (CONVENTIONS §10).
            covered_ghosts: match (&self.ghost_coverage, self.exclude_ghost_class) {
                (Some(c), true) => Some(c.paths.clone()),
                _ => None,
            },
        }
    }

    pub fn validate(&self) -> MkResult<()> {
        if self.classes.0 & !ClassMask::ALL.0 != 0 {
            return Err(MkError::InvalidModel("unknown path class bits".into()));
        }
        self.system_options().validate()?;
        self.stray_settings().validate()
    }
}

/// One class's detector map, row-major `ny × nx` in the detector's local
/// `(x, y)` (row 0 at `y = −half_y`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassMap {
    pub class: PathClass,
    /// W/mm².
    pub irradiance: Vec<f64>,
    /// Monte-Carlo standard error of each pixel (W/mm²).
    pub irradiance_std_error: Vec<f64>,
}

/// Flux of one class on the detector.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ClassTotal {
    pub class: PathClass,
    /// W.
    pub flux: f64,
    pub std_error: f64,
    /// `flux / reference_flux` (`None` when the reference flux is 0).
    pub fraction: Option<f64>,
    /// Whole class selected; an uncovered Ghost subset can be included
    /// separately even when this is false.
    pub included: bool,
    pub hits: u64,
    pub moments: ImageMoments,
}

/// Output of [`analyze`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StrayAnalysisResult {
    pub source: Beam,
    /// Emitted power (W).
    pub emitted: f64,
    /// Whole classes in the stray total; `maps` may also contain the
    /// uncovered subset of Ghost.
    pub classes: ClassMask,
    pub detector: DetectorSpec,
    /// One map per included class, in `PathClass` order. With ghost coverage,
    /// the Ghost row contains only uncovered paths; `totals` still reports
    /// the full class for comparison.
    pub maps: Vec<ClassMap>,
    /// Sum of the included classes (W/mm²), with its own standard error.
    pub total_irradiance: Vec<f64>,
    pub total_irradiance_std_error: Vec<f64>,
    /// Stray total (W) and its standard error.
    pub stray_flux: f64,
    pub stray_std_error: f64,
    /// `stray_flux / reference_flux` (`None` when the reference flux is 0).
    pub stray_fraction: Option<f64>,
    /// Field sources: fraction of the designed (sequential) flux that
    /// non-sequential geometry obstructs (module docs); `None` for other
    /// sources.
    pub primary_obstruction: Option<f64>,
    /// Every class, included or not.
    pub totals: Vec<ClassTotal>,
    /// The strongest `top_signatures` (class, path signature) pairs among
    /// the included classes, largest flux first.
    pub top_signatures: Vec<SignatureFlux>,
    pub energy: EnergyReport,
    /// Clear semi-diameters the scene used, per surface (`None`: the
    /// surface's own fixed value or aperture).
    pub clear_semi_diameters: Vec<Option<f64>>,
    pub run_info: RunInfo,
    pub warnings: Vec<String>,
}

impl StrayAnalysisResult {
    pub fn total_of(&self, class: PathClass) -> &ClassTotal {
        &self.totals[class.index()]
    }
}

#[derive(Serialize)]
struct RunSettings<'a> {
    settings: &'a StrayAnalysisSettings,
    /// The options the scene and detector were actually built with.
    system_options: &'a SystemOptions,
    source: &'a Beam,
    scene_fingerprint: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    resident: Option<&'a RunInfo>,
}

/// Runs a stray-light analysis on a resolved system (module docs).
pub fn analyze(
    rs: Arc<ResolvedSystem>,
    settings: &StrayAnalysisSettings,
    ctx: &JobContext,
) -> MkResult<StrayAnalysisResult> {
    settings.validate()?;
    let sys = StraySystem::build(rs, &settings.system_options(), &ctx.subrange(0.0, 0.1))?;
    ctx.report(0.1, "stray light: scene built");
    analyze_system(&sys, settings, &ctx.subrange(0.1, 1.0))
}

/// As [`analyze`], also returning the flux of **every** (class, path
/// signature) pair the run detected, included in the total or not, largest
/// first (signature tallying is on whatever `top_signatures` says; it does
/// not change any other number). A front end that partitions against an
/// explicit ghost run can use it to report the number of uncovered
/// signatures; ownership and errors are already tallied per ray.
pub fn analyze_with_signatures(
    rs: Arc<ResolvedSystem>,
    settings: &StrayAnalysisSettings,
    ctx: &JobContext,
) -> MkResult<(StrayAnalysisResult, Vec<SignatureFlux>)> {
    settings.validate()?;
    let sys = StraySystem::build(rs, &settings.system_options(), &ctx.subrange(0.0, 0.1))?;
    ctx.report(0.1, "stray light: scene built");
    analyze_system_impl(&sys, settings, true, None, false, &ctx.subrange(0.1, 1.0))
}

/// Grid size of the per-source obstruction check.
const OBSTRUCTION_GRID: usize = 64;

/// As [`analyze`] on a built system. The scene options of `settings` are
/// not reapplied (the system already holds its scene); a difference is
/// reported as a warning and `RunInfo` records the system's options.
pub fn analyze_system(
    sys: &StraySystem,
    settings: &StrayAnalysisSettings,
    ctx: &JobContext,
) -> MkResult<StrayAnalysisResult> {
    analyze_system_impl(sys, settings, false, None, false, ctx).map(|r| r.0)
}

/// A device backend, its transport options and the division of the work.
pub(crate) type Resident<'a> = (
    &'a dyn mklight_gpu::ComputeBackend,
    crate::gpu::TransportOptions,
    crate::gpu::Design,
);

/// [`analyze_system`]; with `all_signatures`, signature tallying is forced
/// on and every tallied signature is returned too. With `deferred_checks`
/// (`sys` from `StraySystem::build_scene`), the obstruction checks of the
/// system fields run here, beside the device transport like the source's
/// primary reference, and complete the warnings as `StraySystem::build`
/// would.
pub(crate) fn analyze_system_impl(
    sys: &StraySystem,
    settings: &StrayAnalysisSettings,
    all_signatures: bool,
    resident: Option<Resident<'_>>,
    deferred_checks: bool,
    ctx: &JobContext,
) -> MkResult<(StrayAnalysisResult, Vec<SignatureFlux>)> {
    settings.validate()?;
    if let Some(cov) = &settings.ghost_coverage {
        sys.check_ghost_coverage(cov)?;
    }
    let timer = RunTimer::start();
    ctx.check()?;
    let beam = settings.source.resolve(sys)?;
    let mut ss = settings.stray_settings();
    ss.signatures |= all_signatures;
    // CPU work independent of the transport: the source's primary
    // reference and (deferred) the system fields' obstruction checks.
    // Beside a device transport it reports no progress of its own.
    let mut primary_obstruction = None;
    let mut checks = None;
    let mut side = |ctx: &JobContext| -> MkResult<()> {
        if primary_obstruction.is_none() {
            primary_obstruction = Some(match settings.source {
                StraySource::Field { .. } | StraySource::SystemField { .. } => Some(
                    primary_reference(sys, &beam.spec, OBSTRUCTION_GRID, &ctx.subrange(0.0, 0.02))?
                        .obstruction(),
                ),
                _ => None,
            });
        }
        if deferred_checks && checks.is_none() {
            checks = Some(
                sys.obstruction_checks(&JobContext::with(ctx.cancel.clone(), Default::default()))?,
            );
        }
        Ok(())
    };
    let mut r = match resident {
        Some((b, options, design)) => {
            let quiet = JobContext::with(ctx.cancel.clone(), Default::default());
            let r = crate::gpu::run_design_with(
                b,
                sys.resolved().snapshot.clone(),
                &sys.scene,
                &beam.spec,
                &ss,
                options,
                design,
                &mut || side(&quiet),
                &ctx.subrange(0.02, 0.98),
            )?;
            side(&quiet)?;
            r
        }
        None => {
            side(ctx)?;
            ctx.report(0.02, "stray light: transport");
            run(&sys.scene, &beam.spec, &ss, &ctx.subrange(0.02, 1.0))?
        }
    };
    drop(side);
    let primary_obstruction = primary_obstruction.flatten();
    // The warnings `StraySystem::build` would have collected.
    let mut sys_warnings = sys.warnings.clone();
    if let Some(c) = checks {
        sys_warnings.extend(c.warnings);
        for w in &sys.scene.warnings {
            if !sys_warnings.contains(w) {
                sys_warnings.push(w.clone());
            }
        }
    }
    let owned = ss.owned_classes;
    let det = &r.detector;
    let area = det.spec.pixel_area();
    let reference = beam.reference_flux;
    let frac = |f: f64| (reference > 0.0).then(|| f / reference);
    let maps = PathClass::ALL
        .iter()
        .filter_map(|&c| {
            let (irradiance, error) = if owned.contains(c) {
                (&det.irradiance[c.index()], &det.flux_std_error[c.index()])
            } else if c == PathClass::Ghost && ss.covered_ghosts.is_some() {
                // Only the uncovered ghosts are owned.
                (
                    &det.uncovered_ghost_irradiance,
                    &det.uncovered_ghost_flux_std_error,
                )
            } else {
                return None;
            };
            Some(ClassMap {
                class: c,
                irradiance: irradiance.clone(),
                irradiance_std_error: error.iter().map(|e| e / area).collect(),
            })
        })
        .collect();
    let e = &r.energy;
    let totals = PathClass::ALL
        .iter()
        .map(|c| ClassTotal {
            class: *c,
            flux: e.detected_in(*c),
            std_error: e.std_error_in(*c),
            fraction: frac(e.detected_in(*c)),
            included: owned.contains(*c),
            hits: e.detected_hits_by_class[c.index()].1,
            moments: det.moments[c.index()],
        })
        .collect();
    let top_signatures = r
        .signatures
        .iter()
        .filter(|s| ss.owns(s.class, &s.signature))
        .take(settings.top_signatures)
        .cloned()
        .collect();
    let all = if all_signatures {
        std::mem::take(&mut r.signatures)
    } else {
        Vec::new()
    };
    let mut warnings = r.warnings.clone();
    for w in &sys_warnings {
        if !warnings.contains(w) {
            warnings.push(w.clone());
        }
    }
    if settings.system_options() != sys.options {
        warnings.push(
            "the settings' scene, semi-diameter or detector options differ from those the system was built with; the system's options were used".into(),
        );
    }
    if let Some(o) = primary_obstruction
        && o > OBSTRUCTION_WARNING
    {
        warnings.push(format!(
            "{:.2} % of this source's designed flux is obstructed by non-sequential geometry; a purely sequential Primary overcounts it",
            100.0 * o
        ));
    }
    let snapshot = &sys.resolved().snapshot;
    let execution = resident.and(r.run_info.as_ref());
    let run_info = timer.finish(
        snapshot,
        "stray_light",
        &RunSettings {
            settings,
            system_options: &sys.options,
            source: &beam,
            scene_fingerprint: sys.scene.fingerprint(),
            resident: execution,
        },
        execution.map_or("cpu-f64", |i| i.backend.as_str()),
        execution.map_or(Precision::F64Reference, |i| i.precision),
        Some(settings.seed),
        warnings.clone(),
    );
    let result = StrayAnalysisResult {
        emitted: e.emitted,
        classes: owned,
        detector: det.spec.clone(),
        maps,
        total_irradiance: det.owned_irradiance.clone(),
        total_irradiance_std_error: det.owned_flux_std_error.iter().map(|x| x / area).collect(),
        stray_flux: e.detected_owned,
        stray_std_error: e.detected_owned_std_error,
        stray_fraction: frac(e.detected_owned),
        primary_obstruction,
        totals,
        top_signatures,
        energy: r.energy,
        clear_semi_diameters: sys.clear_semi_diameters.clone(),
        run_info,
        warnings,
        source: beam,
    };
    ctx.check()?;
    ctx.report(1.0, "stray light: done");
    Ok((result, all))
}
