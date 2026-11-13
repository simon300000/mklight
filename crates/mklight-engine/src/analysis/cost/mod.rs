//! `auto` backend routing by a calibrated stage-level cost model (GPU.md
//! §9.2). [`CostModel::decide`] is the router's single entry point.
//!
//! * Accuracy exclusions: analyses whose native output fails its f64
//!   budget on the current evidence run on the CPU reference at every size
//!   and on every device ([`accuracy_exclusion`]).
//! * A run whose native division gives the device no part (a stray run
//!   without a device share) is the reference by rule: the router settles
//!   it before it asks the model (`routing::auto_route`).
//! * Everything else on a calibrated device class is predicted per stage
//!   for both backends: host f64 preparation, aiming and tracing (rays ×
//!   (surfaces + aim traces × stop surfaces)), transforms, output bytes,
//!   the device's seed-table build of large aimed generations, device
//!   setup when cold (opening it, and an analysis' first native use unless
//!   the kernels were prepared ahead), and the risk of a native fallback.
//!   Cache state enters as stage terms that vanish when shared CPU
//!   intermediates (graph `tracer`, `pupil_lattice` nodes) or resident
//!   device content are already built; a stored graph result is answered
//!   before routing and costs nothing on either backend.
//! * `auto` picks Metal only for a meaningful predicted advantage:
//!   [`RATIO`] and [`MIN_GAIN_MS`]. No universal GPU overhead is assumed:
//!   each analysis has its own measured fixed cost on each backend.
//! * Online refinement: observed complete calls correct each analysis'
//!   prediction per backend, design family and workload decade (bounded,
//!   clamped, inspectable in [`CostModel::stats`]). A decision depends only
//!   on the state when it is made and is recorded in the backend report.
//! * Repeated requests: an identical request (snapshot, library, analysis,
//!   settings) takes the route its first decision chose
//!   ([`CostModel::repeated`]). Refinement, the load prior and warm caches
//!   change the prediction between calls, and the backends agree only
//!   within the accuracy contracts, so a fresh decision could change the
//!   result of the same request (a seeded Monte Carlo, an optimisation).
//!
//! Devices without a calibration table (other GPU families, emulations)
//! use qualified Metal; the decision says so.
use std::collections::{BTreeMap, BTreeSet};
use std::hash::{Hash, Hasher};
use std::sync::Mutex;

use mklight_core::OpticalSystem;
use mklight_gpu::{BackendKind, DeviceInfo};
use serde_json::{Value, json};

use crate::lru::LruCache;

pub mod model;
pub mod profile;
pub mod webgpu;

pub use model::{AnalysisCalibration, Calibration, FEATURES, Features, Fit, Mode};
pub use profile::{AIM_TRACES, Profile, SystemShape, profile};

/// Version of the routing model and its features.
pub const VERSION: u32 = 5;

/// Metal must be predicted at least this many times faster…
pub const RATIO: f64 = 1.3;
/// …and save at least this much wall time. Below it the gain is within the
/// run-to-run spread of small calls and not worth a device round trip.
pub const MIN_GAIN_MS: f64 = 5.;

/// Analyses whose native output fails its f64 contract on the current
/// evidence, with the finding: `auto` uses the reference on every device
/// and at every size. Explicit Metal requests still run natively. GPU.md
/// §9.2 has the evidence of each entry (`metal_accuracy`'s 16 samples and
/// a complex lens, warm and edited) and of the analyses routed by cost.
const EXCLUSIONS: &[(&str, &str)] = &[(
    "glass_substitution",
    "the GPU search returns another finalist glass than the f64 reference (finalist merit 3x outside its gate)",
)];

/// The reported reason of an accuracy exclusion.
pub(crate) const EXCLUDED_NOTE: &str = "auto: this analysis is not accurate enough on the GPU";

/// The accuracy exclusion of `analysis`, with its evidence.
pub(crate) fn accuracy_exclusion(analysis: &str) -> Option<&'static str> {
    EXCLUSIONS
        .iter()
        .find(|(a, _)| *a == analysis)
        .map(|(_, evidence)| *evidence)
}

/// The device a decision is for.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Device {
    /// An Apple GPU: the calibrated family.
    Apple,
    /// A device without a calibration table.
    Other,
    /// The WebGPU backend: the Apple table's Metal sets scaled by the
    /// measured WebGPU/Metal ratio of each analysis (`webgpu_factor`).
    WebGpu,
    /// Not open yet, and its family is not known in advance.
    Unknown,
}

impl Device {
    pub(crate) fn of(info: &DeviceInfo) -> Self {
        if info.kind == BackendKind::WebGpu {
            Device::WebGpu
        } else if is_apple(info) {
            Device::Apple
        } else {
            Device::Other
        }
    }

    pub(crate) fn hinted(apple: Option<bool>, webgpu: bool) -> Self {
        if webgpu {
            return Device::WebGpu;
        }
        match apple {
            Some(true) => Device::Apple,
            Some(false) => Device::Other,
            None => Device::Unknown,
        }
    }
}

pub(crate) fn is_apple(info: &DeviceInfo) -> bool {
    info.kind == BackendKind::Metal && info.vendor == "Apple"
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Route {
    Cpu,
    Metal,
    /// The decision needs the opened device.
    Undecided,
}

#[derive(Clone)]
pub(crate) struct Decision {
    pub route: Route,
    /// Recorded as `BackendReport::cost_decision`.
    pub json: Value,
}

impl Decision {
    /// The reported reason for a CPU route.
    pub(crate) fn note(&self) -> &'static str {
        if self.json["rule"] == "accuracy exclusion" {
            EXCLUDED_NOTE
        } else if self.json["rule"] == "known failure" {
            "auto: this request already failed natively on this device"
        } else {
            "auto: the CPU reference is predicted faster for this workload"
        }
    }
}

/// What the caller knows about the request's state when it decides.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct State<'a> {
    /// Snapshot content hash and material library hash.
    pub snapshot: Option<&'a [u8; 32]>,
    pub library: &'a str,
    /// The device is open (no setup to pay).
    pub device_open: bool,
    /// The graph holds the snapshot's reference tracer.
    pub tracer_warm: bool,
    /// The device's deterministic-failure cache holds this request.
    pub known_failure: bool,
    /// The device's kernels were created ahead (`Engine::prepare_gpu`
    /// finished): no analysis pays pipeline creation on its first native
    /// use.
    pub kernels_prepared: bool,
}

/// Exponentially weighted mean.
#[derive(Clone, Copy, Debug, Default)]
struct Ewma {
    mean: f64,
    n: u32,
}

impl Ewma {
    fn add(&mut self, x: f64, alpha: f64) {
        self.mean = if self.n == 0 {
            x
        } else {
            self.mean + alpha * (x - self.mean)
        };
        self.n = self.n.saturating_add(1);
    }
}

/// Bounded online state (module docs).
struct Online {
    /// ln(observed / predicted) per (analysis, backend, family, decade).
    corrections: LruCache<(String, &'static str, u64, i32), Ewma>,
    /// CPU pupil lattices built per (snapshot, library, pupil request).
    lattices: LruCache<[u8; 32], ()>,
    /// Snapshots whose content is resident on the device.
    resident: LruCache<[u8; 32], ()>,
    /// Snapshots that completed any call (resolved and prepared).
    seen: LruCache<[u8; 32], ()>,
    /// Decisions per (analysis, design family): a device's one-time setup
    /// is spread over the calls that are likely to follow.
    calls: LruCache<(String, u64), u32>,
    /// Analyses that ran natively in this process (first use paid).
    native: BTreeSet<String>,
    /// Transient native failure rate per analysis.
    transient: BTreeMap<String, Ewma>,
    /// Automatic wave pupil grids decided by earlier calls, per design
    /// family and pupil request (an automatic grid depends on the
    /// aberrations; small edits rarely change it).
    grids: LruCache<(u64, String), f64>,
    /// Host contention of the CPU reference: ln(observed / predicted) over
    /// every CPU call, what other processes cost this one beyond the
    /// load-average prior of the moment. (The device is not shared with
    /// the host's runnable threads: Metal has no such factor, its
    /// residuals stay per analysis.)
    contention: Ewma,
    /// The settled route of each request (`routing::request_key`).
    routes: LruCache<[u8; 32], Decision>,
    decisions: u64,
    repeats: u64,
    observations: u64,
}

/// Largest per-analysis correction refinement may apply (either way).
const MAX_CORRECTION: f64 = 10.;
/// Largest share of the CPU's host contention learned beyond the
/// load-average prior (either way).
const HOST_EXCESS: f64 = 3.;
/// Range of the CPU's host contention factor. A loaded host slows the CPU
/// reference far more than the device, which has none.
const CONTENTION: (f64, f64) = (1. / 3., 30.);
const ALPHA: f64 = 0.5;
/// Calls a one-time device setup is spread over.
const SETUP_CALLS: u32 = 4;

/// The engine's cost model: a calibration table plus bounded online state.
pub struct CostModel {
    table: Calibration,
    source: String,
    /// Online refinement; without it, routes depend only on the table and
    /// the cache state.
    refine: bool,
    online: Mutex<Online>,
    /// The WebGPU device's calibration, once measured or installed.
    webgpu: Mutex<Option<webgpu::WebGpuCalibration>>,
}

impl Default for CostModel {
    /// The table of [`Calibration::load`], refining unless
    /// `MKLIGHT_COST_REFINEMENT=0`.
    fn default() -> Self {
        let (table, source) = Calibration::load();
        let refine = std::env::var("MKLIGHT_COST_REFINEMENT").map_or(true, |v| v != "0");
        Self::with_table(table, source, refine)
    }
}

/// Design family of a system: what its cost scales with, stable under
/// small edits.
fn family(system: &OpticalSystem) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    let shape = SystemShape::of(system);
    (
        system.surfaces.len(),
        system.stop,
        shape.aimed,
        system.fields.fields.len(),
        system.wavelengths.wavelengths.len(),
    )
        .hash(&mut h);
    h.finish()
}

/// Identity of the CPU pupil lattices a wave request builds.
fn lattice_key(analysis: &str, s: &Value, snapshot: &[u8; 32], library: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    let kind = if analysis == "mtf_vs_field" {
        format!("mtf_vs_field/{}", s["field_samples"])
    } else {
        "system fields".into()
    };
    for part in [
        &snapshot[..],
        library.as_bytes(),
        kind.as_bytes(),
        s["pupil"].to_string().as_bytes(),
        s["fields"].to_string().as_bytes(),
        s["wavelengths"].to_string().as_bytes(),
    ] {
        h.update(&(part.len() as u64).to_le_bytes());
        h.update(part);
    }
    *h.finalize().as_bytes()
}

fn resident_key(snapshot: &[u8; 32], library: &str) -> [u8; 32] {
    let mut h = blake3::Hasher::new();
    h.update(snapshot);
    h.update(library.as_bytes());
    *h.finalize().as_bytes()
}

fn decade(ms: f64) -> i32 {
    ms.max(1e-3).log10().floor() as i32
}

fn rounded(x: f64) -> f64 {
    if x == 0. || !x.is_finite() {
        return x;
    }
    let scale = 10f64.powi(3 - x.abs().log10().floor() as i32);
    (x * scale).round() / scale
}

fn stages_json(s: &Features) -> Value {
    Value::Object(
        FEATURES
            .iter()
            .zip(s)
            .map(|(k, v)| (k.to_string(), json!(rounded(*v))))
            .collect(),
    )
}

impl CostModel {
    /// Install (or with `None` remove) the WebGPU device's calibration
    /// (`webgpu` module); until then WebGPU routes by the prior.
    pub fn set_webgpu_calibration(&self, c: Option<webgpu::WebGpuCalibration>) {
        *self.webgpu.lock().unwrap_or_else(|e| e.into_inner()) = c;
        // Routes repeated from decisions of another calibration are stale.
        self.lock().routes.clear();
    }

    /// The installed WebGPU calibration.
    pub fn webgpu_calibration(&self) -> Option<webgpu::WebGpuCalibration> {
        self.webgpu
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .clone()
    }

    /// A model over `table` (read from `source`), with online refinement
    /// when `refine`.
    pub fn with_table(table: Calibration, source: String, refine: bool) -> Self {
        CostModel {
            table,
            source,
            refine,
            online: Mutex::new(Online {
                corrections: LruCache::new(1024),
                lattices: LruCache::new(256),
                resident: LruCache::new(64),
                seen: LruCache::new(64),
                calls: LruCache::new(256),
                native: BTreeSet::new(),
                transient: BTreeMap::new(),
                contention: Ewma::default(),
                grids: LruCache::new(256),
                routes: LruCache::new(4096),
                decisions: 0,
                repeats: 0,
                observations: 0,
            }),
            webgpu: Mutex::new(None),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Online> {
        self.online.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The route an identical earlier request settled on, recorded as
    /// repeated (module docs).
    pub(crate) fn repeated(&self, key: &[u8; 32]) -> Option<Decision> {
        let mut on = self.lock();
        let mut d = on.routes.get(key)?;
        on.repeats += 1;
        d.json["repeated"] = json!(true);
        Some(d)
    }

    /// Records the route a request settled on.
    pub(crate) fn settle(&self, key: [u8; 32], d: &Decision) {
        if d.route != Route::Undecided {
            self.lock().routes.insert(key, d.clone());
        }
    }

    /// The automatic route for `analysis` with normalised `settings` on
    /// `system`, and its record.
    pub(crate) fn decide(
        &self,
        analysis: &str,
        settings: &Value,
        system: &OpticalSystem,
        device: Device,
        state: &State<'_>,
    ) -> Decision {
        let decision = |route, json| Decision { route, json };
        if let Some(evidence) = accuracy_exclusion(analysis) {
            return decision(
                Route::Cpu,
                json!({"decision":"cpu_reference","rule":"accuracy exclusion","evidence":evidence,"version":VERSION}),
            );
        }
        // On every device, calibrated or not.
        if state.known_failure {
            return decision(
                Route::Cpu,
                json!({"decision":"cpu_reference","rule":"known failure","version":VERSION}),
            );
        }
        match device {
            Device::Unknown => return decision(Route::Undecided, Value::Null),
            Device::Other => {
                return decision(
                    Route::Metal,
                    json!({"decision":"qualified_metal","reason":"no cost calibration for this device","version":VERSION}),
                );
            }
            Device::WebGpu => {
                let cal = self
                    .webgpu
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .clone();
                return webgpu::decide(analysis, settings, system, cal.as_ref());
            }
            Device::Apple => (),
        }
        let fam = family(system);
        let decided = self.lock().grids.get(&(fam, settings["pupil"].to_string()));
        let (Some(cal), Some(work)) = (
            self.table.analyses.get(analysis),
            profile::profile_with(analysis, settings, system, decided),
        ) else {
            return decision(
                Route::Metal,
                json!({"decision":"qualified_metal","reason":"no calibration for this analysis","version":VERSION}),
            );
        };
        let (Some(cpu_fit), Some(warm_fit)) = (&cal.cpu, &cal.metal) else {
            return decision(
                Route::Metal,
                json!({"decision":"qualified_metal","reason":"no calibration for this analysis","version":VERSION}),
            );
        };
        let shape = SystemShape::of(system);
        let mut on = self.lock();
        on.decisions += 1;
        let features = work.features(&shape);
        let lattice_warm = work.lattice
            && state.snapshot.is_some_and(|s| {
                on.lattices
                    .contains(&lattice_key(analysis, settings, s, state.library))
            });
        let resident = state
            .snapshot
            .is_some_and(|s| on.resident.contains(&resident_key(s, state.library)));
        let seen = state
            .snapshot
            .is_some_and(|s| on.seen.contains(&resident_key(s, state.library)));
        let calls = on.calls.get(&(analysis.to_string(), fam)).unwrap_or(0);
        on.calls
            .insert((analysis.to_string(), fam), calls.saturating_add(1));
        // CPU: shared intermediates already in the graph cost nothing. A
        // new snapshot pays its resolution and preparation.
        let cpu_mode = if seen { Mode::Cpu } else { Mode::CpuEdit };
        let mut cpu_f = cpu_mode.features(features);
        if state.tracer_warm || lattice_warm {
            cpu_f[1] = 0.;
        }
        if lattice_warm {
            cpu_f[2] = 0.;
        }
        let mut cpu = cal.fit(cpu_mode).unwrap_or(cpu_fit).stages(&cpu_f);
        let threads = rayon::current_num_threads().max(1) as f64;
        let scale = (self.table.cpu_threads.max(1) as f64 / threads).clamp(0.25, 4.);
        for c in &mut cpu[2..5] {
            *c *= scale;
        }
        // Metal: device caches (the content-cached seed table among them)
        // warm for a resident snapshot, cold after an edit; host f64
        // preparation is shared with the CPU through the graph.
        let metal_mode = if resident {
            Mode::Metal
        } else {
            Mode::MetalEdit
        };
        let mut metal_f = metal_mode.features(features);
        if state.tracer_warm {
            metal_f[1] = 0.;
        }
        let metal = cal.fit(metal_mode).unwrap_or(warm_fit).stages(&metal_f);
        // One-time setup (device open; first native use of the analysis,
        // unless its kernels were prepared ahead), charged in full to the
        // first call of an analysis and design family and spread over up to
        // SETUP_CALLS calls after that.
        let setup = (if state.device_open {
            0.
        } else {
            self.table.device_open_ms
        } + if state.kernels_prepared || on.native.contains(analysis) {
            0.
        } else {
            cal.first_use_ms
        }) / f64::from(calls.min(SETUP_CALLS - 1) + 1);
        let raw_cpu: f64 = cpu.iter().sum();
        let raw_metal: f64 = metal.iter().sum();
        let refine = self.refine;
        let host = if refine {
            self.contention(&on).exp()
        } else {
            1.
        };
        let mut correction = |backend: &'static str, raw: f64| {
            let key = (analysis.to_string(), backend, fam, decade(raw));
            let c = if refine {
                on.corrections.get(&key).map_or(1., |e| {
                    // Shrink young estimates toward no correction.
                    let w = e.n as f64 / (e.n as f64 + 1.);
                    (w * e.mean).exp()
                })
            } else {
                1.
            };
            (c, key.3)
        };
        let (cpu_c, cpu_decade) = correction("cpu", raw_cpu);
        let (metal_c, metal_decade) = correction("metal", raw_metal);
        let cpu_c = cpu_c * host;
        let p_fail = on.transient.get(analysis).map_or(0., |e| e.mean);
        drop(on);
        let t_cpu = raw_cpu * cpu_c;
        let t_run = raw_metal * metal_c;
        let risk = p_fail * (t_run + t_cpu);
        let t_metal = t_run + setup + risk;
        let gain = t_cpu - t_metal;
        let metal_wins = t_cpu >= RATIO * t_metal && gain >= MIN_GAIN_MS;
        decision(
            if metal_wins { Route::Metal } else { Route::Cpu },
            json!({
                "decision": if metal_wins { "qualified_metal" } else { "cpu_reference" },
                "rule": "cost model",
                "predicted_ms": {"cpu": rounded(t_cpu), "metal": rounded(t_metal)},
                "advantage": rounded(t_cpu / t_metal.max(1e-6)),
                "required": {"ratio": RATIO, "gain_ms": MIN_GAIN_MS},
                "stages_ms": {
                    "cpu": stages_json(&cpu),
                    "metal": stages_json(&metal),
                    "metal_setup": rounded(setup),
                    "metal_fallback_risk": rounded(risk),
                },
                "state": {
                    "device_open": state.device_open,
                    "kernels_prepared": state.kernels_prepared,
                    "tracer_warm": state.tracer_warm,
                    "lattice_warm": lattice_warm,
                    "resident": resident,
                    "snapshot_seen": seen,
                    "calls": calls,
                    "transient_failure_rate": rounded(p_fail),
                    "decided_pupil_grid": decided,
                },
                "refinement": {
                    "family": format!("{fam:016x}"),
                    "cpu": {"decade": cpu_decade, "factor": rounded(cpu_c), "contention": rounded(host)},
                    "metal": {"decade": metal_decade, "factor": rounded(metal_c)},
                },
                "calibration": {"device_class": self.table.device_class, "source": self.source},
                "version": VERSION,
            }),
        )
    }

    /// Names the opened device in a cost-model decision when it is not the
    /// device class the table was measured on: the prediction applies that
    /// class's coefficients to it.
    pub(crate) fn note_device(&self, d: &mut Decision, info: &DeviceInfo) {
        if d.json["rule"] == "cost model" && info.name != self.table.device_class {
            d.json["calibration"]["note"] = json!(format!(
                "calibrated on {}; this device is {}",
                self.table.device_class, info.name
            ));
        }
    }

    /// Learn from a completed (not graph-served) run: residency, shared CPU
    /// intermediates, transient failures and, with refinement on, the ratio
    /// of measured to predicted time.
    pub(crate) fn observe(
        &self,
        analysis: &str,
        settings: &Value,
        system: &OpticalSystem,
        snapshot: &[u8; 32],
        library: &str,
        result: &Observation<'_>,
    ) {
        let native = result.used.starts_with("gpu:");
        let failed = result.failed;
        let Some(work) = profile(analysis, settings, system) else {
            return;
        };
        let fam = family(system);
        // The result's JSON is walked before the model is locked. Only the
        // wave pupils (lattice work) report a decided grid.
        let grid = work
            .lattice
            .then(|| decided_grid(result.json))
            .flatten()
            .map(|n| ((fam, settings["pupil"].to_string()), n));
        let mut on = self.lock();
        on.observations += 1;
        on.seen.insert(resident_key(snapshot, library), ());
        if let Some((key, n)) = grid {
            on.grids.insert(key, n);
        }
        if native {
            on.resident.insert(resident_key(snapshot, library), ());
            on.native.insert(analysis.to_string());
        } else if work.lattice && !result.any_attempt {
            on.lattices
                .insert(lattice_key(analysis, settings, snapshot, library), ());
        }
        let tried_native = native || failed;
        if tried_native {
            on.transient
                .entry(analysis.to_string())
                .or_default()
                .add(f64::from(u8::from(failed)), ALPHA);
        }
        if !self.refine || failed {
            return;
        }
        // A replayed decision (`repeated`) predicted the state of its first
        // request, not of this run: nothing to learn from the difference.
        let Some(d) = result
            .decision
            .filter(|d| d["rule"] == "cost model" && d["repeated"] != true)
        else {
            return;
        };
        let backend: &'static str = if native { "metal" } else { "cpu" };
        let (Some(predicted), Some(decade)) = (
            d["stages_ms"][backend]
                .as_object()
                .map(|o| o.values().filter_map(Value::as_f64).sum::<f64>()),
            d["refinement"][backend]["decade"].as_i64(),
        ) else {
            return;
        };
        let setup = if native {
            d["stages_ms"]["metal_setup"].as_f64().unwrap_or(0.)
        } else {
            0.
        };
        let measured = result.duration_ms - setup;
        if predicted < 0.05 || measured <= 0. {
            return;
        }
        // A CPU call splits ln(measured / predicted) into the host
        // contention shared by every CPU call and this analysis' own
        // residual; a native call's is all residual.
        let ratio = (measured / predicted).ln();
        let key = (analysis.to_string(), backend, fam, decade as i32);
        let mut e = on.corrections.get(&key).unwrap_or_default();
        let host = if native {
            0.
        } else {
            let (lo, hi) = CONTENTION;
            let host = (ratio - e.mean).clamp(lo.ln(), hi.ln()) - self.prior();
            on.contention.add(host, ALPHA);
            self.contention(&on)
        };
        e.add(
            (ratio - host).clamp(-MAX_CORRECTION.ln(), MAX_CORRECTION.ln()),
            ALPHA,
        );
        on.corrections.insert(key, e);
    }

    /// ln of the CPU's current host contention: a prior from the load
    /// average now, √(load per core / 4) and at least 1 (measured: the
    /// reference keeps its speed up to about 4 runnable threads per core
    /// and slows 5–10× near 25), times what observed calls learned beyond
    /// that prior. The prior follows the load even while the CPU is not
    /// chosen, so a stale observation from a busier moment does not keep
    /// it away.
    fn contention(&self, on: &Online) -> f64 {
        let (lo, hi) = CONTENTION;
        let excess = on
            .contention
            .mean
            .clamp(-HOST_EXCESS.ln(), HOST_EXCESS.ln());
        (self.prior() + excess).clamp(lo.ln(), hi.ln())
    }

    /// ln of the CPU's load-average prior.
    fn prior(&self) -> f64 {
        (load_per_core() / 4.).sqrt().max(1.).ln()
    }

    /// The table's identity and the online state, for diagnostics.
    pub fn stats(&self) -> Value {
        let on = self.lock();
        json!({
            "version": VERSION,
            "calibration": {"device_class": self.table.device_class, "source": self.source,
                "measured": self.table.measured, "analyses": self.table.analyses.len()},
            "refinement": self.refine,
            "decisions": on.decisions,
            "repeated_routes": on.repeats,
            "observations": on.observations,
            "corrections": on.corrections.len(),
            "cpu_lattices": on.lattices.len(),
            "resident_snapshots": on.resident.len(),
            "native_analyses": on.native,
            "transient_failure_rates": on.transient.iter().map(|(k, e)| (k.clone(), json!(e.mean))).collect::<BTreeMap<_, _>>(),
            "host_contention": {"cpu": self.contention(&on).exp(), "observed": on.contention.n},
        })
    }
}

/// The root-mean-square pupil grid a wave result reports
/// (`pupil_sampling[].grid`, nested per field), if any.
pub fn decided_grid(json: &Value) -> Option<f64> {
    fn walk(v: &Value, inside: bool, acc: &mut (f64, f64)) {
        match v {
            Value::Object(m) => {
                if inside && let Some(g) = m.get("grid").and_then(Value::as_f64) {
                    acc.0 += g * g;
                    acc.1 += 1.;
                }
                for (k, x) in m {
                    walk(x, inside || k == "pupil_sampling", acc);
                }
            }
            Value::Array(a) => a.iter().for_each(|x| walk(x, inside, acc)),
            _ => {}
        }
    }
    let mut acc = (0., 0.);
    walk(json, false, &mut acc);
    (acc.1 > 0.).then(|| (acc.0 / acc.1).sqrt())
}

/// One-minute load average per logical core (0 when unknown).
pub fn load_per_core() -> f64 {
    mklight_core::platform::load_per_core()
}

/// What the model learns from one completed run.
pub(crate) struct Observation<'a> {
    /// `BackendReport::used`.
    pub used: &'a str,
    /// A native attempt failed (and the reference completed the job).
    pub failed: bool,
    /// Any attempt did not produce the result.
    pub any_attempt: bool,
    pub duration_ms: f64,
    pub decision: Option<&'a Value>,
    /// The result's JSON (sampling decisions).
    pub json: &'a Value,
}

impl<'a> From<&'a super::AnalysisResult> for Observation<'a> {
    fn from(r: &'a super::AnalysisResult) -> Self {
        Observation {
            used: &r.backend.used,
            failed: r
                .run_info
                .failed_attempts
                .iter()
                .any(|a| a.outcome == "failed"),
            any_attempt: !r.run_info.failed_attempts.is_empty(),
            duration_ms: r.run_info.duration_ms,
            decision: r.backend.cost_decision.as_ref(),
            json: &r.json,
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn webgpu_routes_by_measured_break_even() {
        let dg = mklight_core::samples::double_gauss();
        let m = CostModel::with_table(Calibration::builtin(), "builtin".into(), false);
        let st = State {
            snapshot: None,
            library: "",
            device_open: true,
            tracer_warm: false,
            known_failure: false,
            kernels_prepared: true,
        };
        let d = |a: &str, s: Value| m.decide(a, &s, &dg, Device::WebGpu, &st);
        assert_eq!(d("ray_grid", json!({"rings": 128})).route, Route::Metal);
        assert_eq!(d("ray_grid", json!({"rings": 32})).route, Route::Cpu);
        assert_eq!(d("spot_diagram", json!({"rings": 250})).route, Route::Metal);
        assert_eq!(d("spot_diagram", json!({"rings": 64})).route, Route::Cpu);
        let mtf = d("mtf", json!({}));
        assert_eq!(mtf.route, Route::Cpu);
        assert_eq!(
            mtf.json["reason"],
            "no measured WebGPU gain for this analysis"
        );
        // Analyses the rule lists are WebGPU analyses.
        for (a, _) in webgpu::WEBGPU_MIN_RAYS {
            assert!(crate::analysis::gpu::WEBGPU_ANALYSES.contains(a));
        }
    }

    use super::*;
    use mklight_core::samples;

    fn table() -> Calibration {
        let fit = |c: Features| Fit {
            coef: c,
            points: 3,
            rms_log_error: 0.,
        };
        let mut analyses = BTreeMap::new();
        analyses.insert(
            "ray_grid".to_string(),
            AnalysisCalibration {
                // CPU: 1 ms + 1 µs per ray-surface; Metal: 8 ms + 0.1 µs.
                cpu: Some(fit([1., 0., 1e-3, 0., 0., 0.])),
                cpu_edit: None,
                metal: Some(fit([8., 0., 1e-4, 0., 0., 0.])),
                metal_edit: Some(fit([20., 0., 1e-4, 0., 0., 0.])),
                first_use_ms: 30.,
            },
        );
        Calibration {
            format: model::FORMAT,
            model_version: VERSION,
            device_class: "test".into(),
            measured: Value::Null,
            cpu_threads: rayon::current_num_threads(),
            device_open_ms: 40.,
            analyses,
        }
    }

    #[test]
    fn metal_needs_a_meaningful_predicted_advantage() {
        // Routes from the table alone (no host-load seed).
        let m = CostModel::with_table(table(), "test".into(), false);
        let singlet = samples::singlet();
        let spot = |rings: u32| json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"square","rings":rings});
        let snap = [7u8; 32];
        let warm = State {
            snapshot: Some(&snap),
            device_open: true,
            ..Default::default()
        };
        let d = |rings, st: &State| m.decide("ray_grid", &spot(rings), &singlet, Device::Apple, st);
        // Small: CPU (Metal's fixed cost is its own, not a global floor).
        let small = d(6, &warm);
        assert_eq!(small.route, Route::Cpu, "{}", small.json);
        assert_eq!(small.json["rule"], "cost model");
        // Large: Metal on an open device…
        let large = d(512, &warm);
        assert_eq!(large.route, Route::Metal, "{}", large.json);
        assert!(large.json["advantage"].as_f64().unwrap() >= RATIO);
        // …and the cold device's setup is part of the prediction.
        let cold = d(
            512,
            &State {
                device_open: false,
                ..warm
            },
        );
        let setup = cold.json["stages_ms"]["metal_setup"].as_f64().unwrap();
        // Device open and first use, spread over the calls so far.
        let calls = cold.json["state"]["calls"].as_f64().unwrap();
        let spread = 70. / (calls.min(f64::from(SETUP_CALLS - 1)) + 1.);
        assert!((setup - spread).abs() < 1e-2, "{setup} {spread}");
        // Kernels prepared ahead leave only the device open to pay.
        let prepared = d(
            512,
            &State {
                device_open: false,
                kernels_prepared: true,
                ..warm
            },
        );
        let calls = prepared.json["state"]["calls"].as_f64().unwrap();
        let spread = 40. / (calls.min(f64::from(SETUP_CALLS - 1)) + 1.);
        let setup = prepared.json["stages_ms"]["metal_setup"].as_f64().unwrap();
        assert!((setup - spread).abs() < 1e-2, "{setup} {spread}");
        assert_eq!(prepared.json["state"]["kernels_prepared"], true);
        assert_eq!(cold.json["state"]["kernels_prepared"], false);
        // A known deterministic failure, and an accuracy exclusion.
        let known = d(
            512,
            &State {
                known_failure: true,
                ..warm
            },
        );
        assert_eq!(
            (known.route, known.json["rule"].as_str()),
            (Route::Cpu, Some("known failure"))
        );
        let excluded = m.decide(
            "glass_substitution",
            &json!({}),
            &singlet,
            Device::Unknown,
            &warm,
        );
        assert_eq!(excluded.route, Route::Cpu);
        assert_eq!(excluded.note(), EXCLUDED_NOTE);
        // Unknown and uncalibrated devices.
        assert_eq!(d(6, &warm).route, Route::Cpu);
        assert_eq!(
            m.decide("ray_grid", &spot(6), &singlet, Device::Unknown, &warm)
                .route,
            Route::Undecided
        );
        assert_eq!(
            m.decide("ray_grid", &spot(6), &singlet, Device::Other, &warm)
                .route,
            Route::Metal
        );
        // A known failure is not selected again on an uncalibrated device
        // either.
        let known = m.decide(
            "ray_grid",
            &spot(6),
            &singlet,
            Device::Other,
            &State {
                known_failure: true,
                ..warm
            },
        );
        assert_eq!(
            (known.route, known.json["rule"].as_str()),
            (Route::Cpu, Some("known failure"))
        );
    }

    #[test]
    fn refinement_is_bounded_and_can_be_disabled() {
        let fixed = CostModel::with_table(table(), "test".into(), false);
        let refining = CostModel::with_table(table(), "test".into(), true);
        let singlet = samples::singlet();
        let settings = json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"square","rings":64});
        let snap = [3u8; 32];
        let st = State {
            snapshot: Some(&snap),
            device_open: true,
            ..Default::default()
        };
        let first = fixed.decide("ray_grid", &settings, &singlet, Device::Apple, &st);
        let predicted = first.json["predicted_ms"]["cpu"].as_f64().unwrap();
        // The CPU keeps running 100× slower than predicted: host contention
        // and the analysis' residual saturate at their bounds.
        let r = Observation {
            used: "cpu_reference",
            failed: false,
            any_attempt: false,
            duration_ms: predicted * 100.,
            decision: Some(&first.json),
            json: &Value::Null,
        };
        for m in [&fixed, &refining] {
            for _ in 0..20 {
                m.observe("ray_grid", &settings, &singlet, &snap, "", &r);
            }
        }
        let later = refining.decide("ray_grid", &settings, &singlet, Device::Apple, &st);
        let factor = later.json["refinement"]["cpu"]["factor"].as_f64().unwrap();
        let bound = (MAX_CORRECTION * HOST_EXCESS * 4.).min(MAX_CORRECTION * CONTENTION.1);
        assert!(factor > 7. && factor <= bound * (1. + 1e-3), "{factor}");
        let host = later.json["refinement"]["cpu"]["contention"]
            .as_f64()
            .unwrap();
        assert!(host > 2.5 && host <= CONTENTION.1 * (1. + 1e-3), "{host}");
        // Without refinement the same observations change nothing.
        let off = fixed.decide("ray_grid", &settings, &singlet, Device::Apple, &st);
        assert_eq!(off.json["predicted_ms"]["cpu"].as_f64().unwrap(), predicted);
        assert_eq!(fixed.stats()["corrections"], 0);
    }

    /// A decision on a device other than the table's class says so; the
    /// route is the table's either way.
    #[test]
    fn decisions_name_a_device_the_table_was_not_measured_on() {
        let m = CostModel::with_table(table(), "test".into(), false);
        let singlet = samples::singlet();
        let settings = json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"square","rings":512});
        let st = State {
            device_open: true,
            ..Default::default()
        };
        let decided = m.decide("ray_grid", &settings, &singlet, Device::Apple, &st);
        use mklight_gpu::ComputeBackend;
        let mut info = mklight_gpu::CpuF32Backend::new().info().clone();
        info.name = "test".into();
        let mut same = decided.clone();
        m.note_device(&mut same, &info);
        assert_eq!(same.json, decided.json);
        info.name = "Apple M9 Ultra".into();
        let mut other = decided.clone();
        m.note_device(&mut other, &info);
        assert_eq!(other.route, decided.route);
        assert_eq!(
            other.json["calibration"]["note"],
            "calibrated on test; this device is Apple M9 Ultra"
        );
        other.json["calibration"]
            .as_object_mut()
            .unwrap()
            .remove("note");
        assert_eq!(other.json, decided.json);
        // Only a prediction is the table's.
        let mut excluded = m.decide(
            "glass_substitution",
            &json!({}),
            &singlet,
            Device::Apple,
            &st,
        );
        let before = excluded.json.clone();
        m.note_device(&mut excluded, &info);
        assert_eq!(excluded.json, before);
    }

    /// A replayed decision predicted its first request's state (cold
    /// caches, an unseen snapshot), so the run it routes corrects nothing.
    #[test]
    fn replayed_decisions_do_not_refine() {
        let m = CostModel::with_table(table(), "test".into(), true);
        let singlet = samples::singlet();
        let settings = json!({"fields":{"type":"all"},"wavelengths":{"type":"all"},"pattern":"square","rings":64});
        let snap = [5u8; 32];
        let st = State {
            snapshot: Some(&snap),
            device_open: true,
            ..Default::default()
        };
        let first = m.decide("ray_grid", &settings, &singlet, Device::Apple, &st);
        let key = [9u8; 32];
        m.settle(key, &first);
        let replayed = m.repeated(&key).unwrap();
        assert_eq!(replayed.json["repeated"], true);
        let run = |decision: &Value| {
            m.observe(
                "ray_grid",
                &settings,
                &singlet,
                &snap,
                "",
                &Observation {
                    used: "cpu_reference",
                    failed: false,
                    any_attempt: false,
                    duration_ms: 1e3,
                    decision: Some(decision),
                    json: &Value::Null,
                },
            );
            let stats = m.stats();
            (
                stats["corrections"].as_u64().unwrap(),
                stats["host_contention"]["observed"].as_u64().unwrap(),
            )
        };
        assert_eq!(run(&replayed.json), (0, 0));
        assert_eq!(run(&first.json), (1, 1));
        assert_eq!(m.stats()["observations"], 2);
    }
}
