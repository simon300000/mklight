//! Real-ray ghost irradiance on the sensor.
//!
//! # Sampling
//!
//! The fields are the selected system fields (with their weights and
//! vignetting factors) followed by arbitrary normalised [`FieldPoint`]s
//! (weight 1 each, e.g. a bright source outside the field of view), so an
//! off-grid source needs no model edit (which would change the snapshot).
//! For each field and wavelength, starting rays are generated once
//! on a square grid of `n × n` cell centres over the aim plane of the
//! reference tracer (the paraxial entrance-pupil plane, or object-space
//! tangents for a telecentric object), covering a disc of `overfill` times
//! the entrance-pupil radius around the chief ray's aim point. Ghost light
//! can enter outside the entrance pupil (a reflection before the stop may
//! still pass the stop afterwards), so the overfilled beam is traced and
//! the model's apertures ([`ClipMode`](crate::ClipMode)) decide what
//! survives, for the primary path exactly as for the ghosts. Cells whose
//! centre and corners disagree on survival or on-map detection are split
//! into four children, up to four levels; each child carries one quarter
//! of its parent's flux. This resolves clipping boundaries without giving
//! every cell the weight of its centre's binary acceptance.
//!
//! All centres and corners lie on an integer lattice of 32 units per base
//! cell, converted to pupil coordinates in one place ([`pupil`]). Per path,
//! each lattice point (and field and wavelength) is traced once: refinement
//! proceeds level by level, tracing the new centres, then the new corners of
//! the cells at that level, in parallel batches; shared corners and the
//! centres and corners of parents are looked up, not traced again. Paths
//! adjacent in step order are refined together, and a lattice point that
//! several of them request is traced once along their shared step prefix
//! (`PathGroup`); each path's results are those of tracing it alone.
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
//! Hits are binned into an `nx × ny` map in the image surface's local `x`,
//! `y`, over an explicit `map_half_size` if given, else the model's
//! [`Sensor`](mklight_core::Sensor) rectangle, else the square of the image
//! clear semi-diameter. Values are irradiance: flux per mm² of bin. The primary map is
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
//! With a sensor reflectance, detection counts the flux the sensor absorbs,
//! `(1 − R_sensor)` × incident, for the primary and every ghost alike; the
//! reflected part is the source of the `[image, j]` ghosts.
//!
//! The primary path obeys the virtual-segment rule of the sequence tracer
//! like every ghost. Primary rays it loses (rays crossing a physically
//! impossible element, which the reference tracer would pass) are counted
//! and reported as a `RunInfo` warning with their flux, because the ratios
//! are then normalised to a smaller signal than the sequential solver's.
//!
//! # Determinism, parallelism, cancellation
//!
//! Probes are traced in parallel (rayon) in blocks; a probe's result depends
//! only on its lattice point, and leaves are collected depth first in the
//! fixed order `(field, wavelength, base cell)` and accumulated sequentially
//! with Neumaier compensation, so maps are bit-identical run to run. The
//! [`JobContext`] is checked before every block; progress is monotone and
//! ends at 1.
//!
//! # GPU port
//!
//! The work decomposes into independent rays indexed by
//! `(path, field, wavelength, lattice point)`. The kernel interface needed
//! from a GPU backend is described in `docs/GHOST.md`.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::hash::BuildHasherDefault;

use mklight_core::job::RunTimer;
use mklight_core::math::KahanSum;
use mklight_core::{JobContext, MkError, MkResult, RunInfo};
use mklight_trace::{FieldPoint, PreparedField};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::model::GhostModel;
use crate::paraxial::run_info;
use crate::path::{EnumerationSettings, GhostPath, Step, enumerate};
use crate::sequence::SequenceStatus;
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
    /// System field indices; empty = all system fields, unless `points` is
    /// non-empty.
    #[serde(default)]
    pub fields: Vec<usize>,
    /// Arbitrary normalised field points (weight 1 each), analysed after
    /// the system fields (module docs).
    #[serde(default)]
    pub points: Vec<FieldPoint>,
    /// System wavelength indices; empty = all wavelengths.
    #[serde(default)]
    pub wavelengths: Vec<usize>,
    /// Grid cells across the overfilled pupil square.
    pub pupil_samples: usize,
    /// Sampled pupil radius in entrance-pupil radii (≥ 1).
    pub overfill: f64,
    /// Map bins `[nx, ny]`.
    pub map_bins: [usize; 2],
    /// Map half-size `[x, y]` (mm); takes precedence; `None` = sensor
    /// rectangle, else image clear aperture.
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
            points: Vec::new(),
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

/// A field of an irradiance run.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum FieldRef {
    /// System field index.
    System { index: usize },
    /// An arbitrary normalised field point (`IrradianceSettings::points`).
    Point { point: FieldPoint },
}

/// Statistics of one path at one field (all selected wavelengths).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FieldIrradiance {
    pub field: FieldRef,
    /// Detected flux, on or off the map.
    pub flux: f64,
    pub flux_on_map: f64,
    /// Peak irradiance of this field's map.
    pub peak: f64,
    pub centroid: [f64; 2],
    pub rms_radius: f64,
    /// Quadrature leaf samples (including zero-weight misses, excluding
    /// corner probes); refinement makes their cell areas unequal.
    pub rays_traced: usize,
    pub rays_detected: usize,
}

/// Result of one path.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathIrradiance {
    pub path: GhostPath,
    pub label: String,
    pub fields: Vec<FieldIrradiance>,
    /// Ghost flux on the map / primary flux on the map (1 for the primary
    /// itself).
    pub flux_ratio: f64,
    /// Max over fields of ghost peak / primary peak (1 for the primary).
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

/// Resolve discontinuities at the pupil rim, clipping apertures and map
/// boundary. A cell-centre grid can alias a small ghost footprint badly;
/// refine cells whose centre and corners disagree, to 1/16 of a base cell.
const BOUNDARY_DEPTH: u32 = 4;

/// Half a base cell in lattice units.
const HALF: i32 = 1 << BOUNDARY_DEPTH;

/// Pupil coordinate of lattice coordinate `u` (0 at the chief ray, `±n·HALF`
/// at `±overfill`).
fn pupil(u: i32, n: usize, overfill: f64) -> f64 {
    f64::from(u) * overfill / (n as f64 * f64::from(HALF))
}

/// A lattice probe: field index, wavelength index and lattice point.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Probe {
    field: u32,
    wl: u32,
    x: i32,
    y: i32,
}

impl Probe {
    /// The four probes at `±d` in both axes, `y` outer.
    fn around(self, d: i32) -> [Probe; 4] {
        [(-d, -d), (d, -d), (-d, d), (d, d)].map(|(dx, dy)| Probe {
            x: self.x + dx,
            y: self.y + dy,
            ..self
        })
    }
}

/// Multiplicative hash of a probe's integers; SipHash dominated the
/// sequential cache bookkeeping.
#[derive(Default)]
struct ProbeHasher(u64);

impl std::hash::Hasher for ProbeHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u32(b.into());
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.0 = (self.0.rotate_left(5) ^ u64::from(i)).wrapping_mul(0x517c_c1b7_2722_0a95);
    }
    fn write_i32(&mut self, i: i32) {
        self.write_u32(i as u32);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

/// A base cell: its centre probe and its flux weight.
#[derive(Clone, Copy, Debug)]
struct Cell {
    probe: Probe,
    weight: f64,
}

/// One selected field: the system field or a prepared point, and its weight.
struct SelectedField<'a> {
    id: FieldRef,
    field: std::borrow::Cow<'a, PreparedField>,
    weight: f64,
}

/// The fields of a run (module docs); unpreparable points give a warning
/// and no rays.
fn select_fields<'a>(
    model: &'a GhostModel,
    settings: &IrradianceSettings,
    warnings: &mut Vec<String>,
) -> MkResult<Vec<SelectedField<'a>>> {
    let tracer = model.tracer();
    let osys = model.system().system();
    let nf = osys.fields.fields.len();
    let system: Vec<usize> = if settings.fields.is_empty() && settings.points.is_empty() {
        (0..nf).collect()
    } else {
        settings.fields.clone()
    };
    if system.iter().any(|&f| f >= nf) {
        return Err(MkError::InvalidModel("field index out of range".into()));
    }
    let mut out: Vec<SelectedField> = system
        .into_iter()
        .map(|f| SelectedField {
            id: FieldRef::System { index: f },
            field: std::borrow::Cow::Borrowed(tracer.field(f)),
            weight: osys.fields.fields[f].weight,
        })
        .collect();
    for p in &settings.points {
        if !(p.hx.is_finite() && p.hy.is_finite()) {
            return Err(MkError::InvalidModel("field point must be finite".into()));
        }
        let field = match tracer.prepare_field(*p) {
            Ok(f) => f,
            Err(e) => {
                warnings.push(format!(
                    "field point ({}, {}) could not be prepared: {e}",
                    p.hx, p.hy
                ));
                let mut f = tracer.field(0).clone();
                f.error = Some(e.to_string());
                f
            }
        };
        out.push(SelectedField {
            id: FieldRef::Point { point: *p },
            field: std::borrow::Cow::Owned(field),
            weight: 1.0,
        });
    }
    Ok(out)
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

/// Base cells per selected field (module docs).
fn base_cells(
    model: &GhostModel,
    fields: &[SelectedField],
    wls: &[usize],
    settings: &IrradianceSettings,
    ctx: &JobContext,
) -> MkResult<Vec<Vec<Cell>>> {
    let sys = model.system();
    let osys = sys.system();
    let n = settings.pupil_samples;
    let k = settings.overfill;
    let fw: Vec<f64> = fields.iter().map(|f| f.weight).collect();
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
    let area = h * h / std::f64::consts::PI;
    let mut out = Vec::with_capacity(fields.len());
    for (fi, sel) in fields.iter().enumerate() {
        let mut cells = Vec::new();
        if sel.field.is_valid() {
            for (wi, &w) in wls.iter().enumerate() {
                let weight = fw[fi] / sf * ww[wi] / sw * area;
                for iy in 0..n {
                    ctx.check()?;
                    for ix in 0..n {
                        let [x, y] = [ix, iy].map(|i| (2 * i as i32 + 1 - n as i32) * HALF);
                        // Skip cells entirely outside the sampled disc.
                        let [nx, ny] = [x, y].map(|u| pupil((u.abs() - HALF).max(0), n, k));
                        if nx.mul_add(nx, ny * ny) > k * k {
                            continue;
                        }
                        cells.push(Cell {
                            probe: Probe {
                                field: fi as u32,
                                wl: w as u32,
                                x,
                                y,
                            },
                            weight,
                        });
                    }
                }
            }
        }
        out.push(cells);
    }
    Ok(out)
}

/// Per-path results: per-field maps and statistics, and the number and
/// flux of rays lost to the virtual-segment rule.
#[derive(Default)]
pub(crate) struct LeafSummary {
    traced: usize,
    virtual_rays: usize,
    virtual_flux: KahanSum,
}
struct PathRun {
    maps: Vec<IrradianceMap>,
    stats: Vec<FieldIrradiance>,
    virtual_rays: usize,
    virtual_flux: f64,
}

/// Outcome of one ray: detected `(x, y, w)`, or lost as virtual with `w`.
pub(crate) type RayOut = Result<Option<(f64, f64, f64)>, f64>;

/// One path's probes on the pupil lattice.
struct PathTrace<'a> {
    model: &'a GhostModel,
    steps: &'a [Step],
    fields: &'a [SelectedField<'a>],
    n: usize,
    overfill: f64,
    map: &'a IrradianceMap,
}

impl PathTrace<'_> {
    /// Resident counterpart of [`Self::leaves`]: detected leaves keep their
    /// resident probe location; virtual and lost leaves only count.
    fn resident_leaves(
        &self,
        probes: &Probes,
        centre: Probe,
        half: i32,
        weight: f64,
        map: u32,
        out: &mut (Vec<mklight_gpu::ghost_ops::Leaf>, LeafSummary),
    ) {
        if self.splits(probes, centre, half) {
            for c in centre.around(half / 2) {
                self.resident_leaves(probes, c, half / 2, weight * 0.25, map, out);
            }
            return;
        }
        let summary = &mut out.1;
        summary.traced += 1;
        let id = probes.index[&centre];
        let (x, y) = match probes.outs[id] {
            Ok(Some((x, y, _))) => (x, y),
            Ok(None) => return,
            Err(w) => {
                summary.virtual_rays += 1;
                summary.virtual_flux.add(weight * w);
                return;
            }
        };
        let [batch, row] = probes.locations[id].unwrap();
        out.0.push(mklight_gpu::ghost_ops::Leaf {
            batch: batch as u32,
            row: row as u32,
            weight: mklight_gpu::Df32::from_f64(weight),
            destination: 0,
            map,
            // The reference bin decides inside the pixel-edge band.
            pixel: if pixel_edge(self.map, x, y) {
                self.map.bin_of(x, y).unwrap_or(self.map.values.len()) as u32 + 1
            } else {
                0
            },
            reserved: 0,
        });
    }
    /// The starting ray of a lattice probe; `None` outside the sampled disc
    /// or when the source cannot reach the aim point. Independent of the path.
    fn start(&self, probe: Probe) -> Option<mklight_trace::Ray> {
        let p = [probe.x, probe.y].map(|u| pupil(u, self.n, self.overfill));
        if p[0].mul_add(p[0], p[1] * p[1]) > self.overfill * self.overfill {
            return None;
        }
        let tracer = self.model.tracer();
        let pupil = tracer.pupil();
        let scale = if pupil.telecentric {
            pupil.u_marginal.abs()
        } else {
            pupil.r_ep
        };
        let (field, wl) = (&self.fields[probe.field as usize].field, probe.wl as usize);
        let c = field.aims[wl].chief;
        let a = [p[0].mul_add(scale, c[0]), p[1].mul_add(scale, c[1])];
        tracer.start_ray(&field.source, a)
    }
    fn trace(&self, probe: Probe) -> RayOut {
        let Some(ray) = self.start(probe) else {
            return Ok(None);
        };
        ray_out(
            &self
                .model
                .trace_steps(&ray, probe.wl as usize, self.steps, None),
        )
    }

    fn region(&self, sample: RayOut) -> u8 {
        match sample {
            Ok(Some((x, y, _))) if self.map.bin_of(x, y).is_some() => 0,
            Ok(Some(_)) => 1,
            Ok(None) => 2,
            Err(_) => 3,
        }
    }

    /// Whether the cell of half-width `half` around `centre` is refined; its
    /// centre and corners must be cached.
    fn splits(&self, probes: &Probes, centre: Probe, half: i32) -> bool {
        let region = self.region(probes.get(centre));
        half > 1
            && centre
                .around(half)
                .into_iter()
                .any(|c| self.region(probes.get(c)) != region)
    }

    /// Weighted leaves of a cell, depth first.
    fn leaves(
        &self,
        probes: &Probes,
        centre: Probe,
        half: i32,
        weight: f64,
        out: &mut Vec<RayOut>,
    ) {
        if self.splits(probes, centre, half) {
            for c in centre.around(half / 2) {
                self.leaves(probes, c, half / 2, weight * 0.25, out);
            }
        } else {
            out.push(weighted(probes.get(centre), weight));
        }
    }
}

/// Outcome of a traced sequence on the sensor.
fn ray_out(r: &crate::sequence::SequenceResult) -> RayOut {
    match r.status {
        SequenceStatus::Ok => Ok(Some((r.ray.pos.x, r.ray.pos.y, r.weight()))),
        SequenceStatus::Virtual { .. } => Err(r.weight()),
        _ => Ok(None),
    }
}

/// Sort key of a step; paths are ordered lexicographically by their steps,
/// so neighbours in that order share their longest common prefixes.
fn step_key(s: &Step) -> (u32, u8, u8, i64) {
    (
        s.surface,
        s.direction as u8,
        s.action as u8,
        s.order.map_or(i64::MIN, i64::from),
    )
}

/// Paths whose probes are traced together: a lattice point requested by
/// several of them is traced once along every shared step prefix (the
/// forward trace to the first reflection, a shared backward run). A copied
/// trace state continues bit-identically, so each path's results are those
/// of tracing it alone.
struct PathGroup<'a, 'b> {
    /// Ordered lexicographically by steps.
    paths: &'b [PathTrace<'a>],
}

impl PathGroup<'_, '_> {
    /// Trace one lattice point for `members` (indices into `paths`,
    /// ascending), appending one outcome per member.
    fn walk(
        &self,
        probe: Probe,
        members: impl Iterator<Item = usize>,
        stack: &mut Vec<crate::sequence::TraceState>,
        out: &mut Vec<RayOut>,
    ) {
        let first = &self.paths[0];
        let ray = first.start(probe);
        let model = first.model;
        let wl = probe.wl as usize;
        let start = ray.map(|r| model.begin(0, &r));
        stack.clear();
        // The steps the stack belongs to, and the step at which they ended
        // (with that result), if they did not complete.
        let mut prev: &[Step] = &[];
        let mut ended: Option<(usize, RayOut)> = None;
        for m in members {
            let steps = self.paths[m].steps;
            let state0 = match &start {
                None => {
                    out.push(Ok(None));
                    continue;
                }
                Some(Err(r)) => {
                    out.push(ray_out(r));
                    continue;
                }
                Some(Ok(s)) => *s,
            };
            let lcp = prev.iter().zip(steps).take_while(|(a, b)| a == b).count();
            prev = steps;
            if let Some((step, result)) = ended
                && lcp > step
            {
                out.push(result);
                continue;
            }
            if stack.is_empty() {
                stack.push(state0);
            }
            stack.truncate(lcp + 1);
            ended = None;
            let mut state = *stack.last().unwrap();
            let mut result = None;
            for (k, st) in steps.iter().enumerate().skip(lcp) {
                if let Some(r) = model.advance(&mut state, wl, st, None) {
                    let r = ray_out(&r);
                    ended = Some((k, r));
                    result = Some(r);
                    break;
                }
                stack.push(state);
            }
            out.push(result.unwrap_or_else(|| ray_out(&state.finish())));
        }
    }
}

fn weighted(sample: RayOut, weight: f64) -> RayOut {
    match sample {
        Ok(Some((x, y, w))) => Ok(Some((x, y, weight * w))),
        Ok(None) => Ok(None),
        Err(w) => Err(weight * w),
    }
}

/// Traced probes of one path.
#[derive(Default)]
struct Probes {
    index: HashMap<Probe, usize, BuildHasherDefault<ProbeHasher>>,
    outs: Vec<RayOut>,
    locations: Vec<Option<[usize; 2]>>,
}

impl Probes {
    fn get(&self, p: Probe) -> RayOut {
        self.outs[self.index[&p]]
    }

    /// Trace the probes not yet cached, in parallel.
    fn add(
        &mut self,
        probes: impl IntoIterator<Item = Probe>,
        path: &PathTrace,
        ctx: &JobContext,
    ) -> MkResult<()> {
        let mut new = Vec::new();
        for p in probes {
            if let Entry::Vacant(e) = self.index.entry(p) {
                e.insert(self.outs.len() + new.len());
                new.push(p);
            }
        }
        for block in new.chunks(BLOCK) {
            ctx.check()?;
            self.outs
                .par_extend(block.par_iter().with_min_len(64).map(|&p| path.trace(p)));
        }
        Ok(())
    }

    /// Trace every probe the refinement of `cells` needs, level by level.
    fn refine(
        cells: &[Cell],
        path: &PathTrace,
        ctx: &JobContext,
        progress: (f64, f64),
    ) -> MkResult<Probes> {
        let mut probes = Probes::default();
        probes.index.reserve(4 * cells.len());
        let mut level: Vec<Probe> = cells.iter().map(|c| c.probe).collect();
        let mut half = HALF;
        loop {
            probes.add(level.iter().copied(), path, ctx)?;
            let done = (HALF / half).ilog2() as f64 / BOUNDARY_DEPTH as f64;
            ctx.report(
                progress.0 + done * (progress.1 - progress.0),
                "tracing ghost paths",
            );
            if half == 1 {
                return Ok(probes);
            }
            probes.add(level.iter().flat_map(|c| c.around(half)), path, ctx)?;
            level = level
                .par_iter()
                .filter(|&&c| path.splits(&probes, c, half))
                .flat_map_iter(|c| c.around(half / 2))
                .collect();
            half /= 2;
        }
    }
}

/// Lattice points traced per cancellation / progress block of a group.
const GROUP_BLOCK: usize = 4096;

impl PathGroup<'_, '_> {
    /// Every probe each member's refinement of `cells` needs, level by level
    /// exactly as [`Probes::refine`] per path, with the lattice points that
    /// several members request at a level traced together.
    fn refine(
        &self,
        cells: &[Cell],
        ctx: &JobContext,
        progress: (f64, f64),
    ) -> MkResult<Vec<Probes>> {
        let n = self.paths.len();
        let mut caches: Vec<Probes> = (0..n)
            .map(|_| {
                let mut p = Probes::default();
                p.index.reserve(4 * cells.len());
                p
            })
            .collect();
        let mut levels: Vec<Vec<Probe>> = vec![cells.iter().map(|c| c.probe).collect(); n];
        let mut half = HALF;
        loop {
            self.add(&mut caches, |m| levels[m].to_vec(), ctx)?;
            let done = (HALF / half).ilog2() as f64 / BOUNDARY_DEPTH as f64;
            ctx.report(
                progress.0 + done * (progress.1 - progress.0),
                "tracing ghost paths",
            );
            if half == 1 {
                return Ok(caches);
            }
            self.add(
                &mut caches,
                |m| levels[m].iter().flat_map(|c| c.around(half)).collect(),
                ctx,
            )?;
            levels = levels
                .iter()
                .zip(&caches)
                .enumerate()
                .map(|(m, (level, cache))| {
                    level
                        .par_iter()
                        .filter(|&&c| self.paths[m].splits(cache, c, half))
                        .flat_map_iter(|c| c.around(half / 2))
                        .collect()
                })
                .collect();
            half /= 2;
        }
    }

    /// Trace the requested probes each member has not cached yet.
    fn add(
        &self,
        caches: &mut [Probes],
        requests: impl Fn(usize) -> Vec<Probe> + Sync,
        ctx: &JobContext,
    ) -> MkResult<()> {
        // (probe, member, cache slot); members own disjoint caches.
        let mut entries = caches
            .par_iter_mut()
            .enumerate()
            .map(|(m, cache)| {
                let mut new = Vec::new();
                for p in requests(m) {
                    if let Entry::Vacant(e) = cache.index.entry(p) {
                        let id = cache.outs.len();
                        e.insert(id);
                        cache.outs.push(Ok(None));
                        new.push((p, m as u32, id as u32));
                    }
                }
                new
            })
            .collect::<Vec<_>>()
            .concat();
        // Keys are unique, so the unstable sort is deterministic.
        entries.par_sort_unstable_by_key(|&(p, m, _)| (p.field, p.wl, p.x, p.y, m));
        let mut groups = Vec::new();
        let mut a = 0;
        for b in 1..=entries.len() {
            if b == entries.len() || entries[b].0 != entries[a].0 {
                groups.push((a, b));
                a = b;
            }
        }
        for block in groups.chunks(GROUP_BLOCK) {
            ctx.check()?;
            // Chunks of lattice points share their output and state stack.
            let outs = block
                .par_chunks(32)
                .map(|chunk| {
                    let (mut out, mut stack) = (Vec::new(), Vec::new());
                    for &(a, b) in chunk {
                        self.walk(
                            entries[a].0,
                            entries[a..b].iter().map(|e| e.1 as usize),
                            &mut stack,
                            &mut out,
                        );
                    }
                    out
                })
                .collect::<Vec<_>>();
            let (lo, hi) = (block[0].0, block[block.len() - 1].1);
            for (&(_, m, id), out) in entries[lo..hi].iter().zip(outs.into_iter().flatten()) {
                caches[m as usize].outs[id as usize] = out;
            }
        }
        Ok(())
    }
}

/// Paths per [`PathGroup`]: bounds the probe caches held at once.
const GROUP_PATHS: usize = 16;

/// Trace one path for all base cells; per-field maps and statistics.
#[allow(clippy::too_many_arguments)]
fn run_path(
    model: &GhostModel,
    steps: &[Step],
    cells: &[Vec<Cell>],
    fields: &[SelectedField],
    settings: &IrradianceSettings,
    template: &IrradianceMap,
    ctx: &JobContext,
    progress: (f64, f64),
    gpu_maps: Option<(&crate::gpu::MapData, usize)>,
    traced: Option<Probes>,
) -> MkResult<PathRun> {
    let path = PathTrace {
        model,
        steps,
        fields,
        n: settings.pupil_samples,
        overfill: settings.overfill,
        map: template,
    };
    if let Some((data, path_index)) = gpu_maps {
        let (mut maps, mut stats) = (Vec::new(), Vec::new());
        let (mut virtual_rays, mut virtual_flux) = (0, KahanSum::default());
        for (fi, field) in fields.iter().enumerate() {
            ctx.check()?;
            let id = path_index * fields.len() + fi;
            let summary = &data.summaries[id];
            let m = data.moments[id];
            let flux = m[1];
            let mut map = template.clone();
            map.values.copy_from_slice(
                &data.values[id * template.values.len()..(id + 1) * template.values.len()],
            );
            stats.push(FieldIrradiance {
                field: field.id,
                flux,
                flux_on_map: map.flux(),
                peak: map.peak(),
                centroid: if flux > 0. { [m[2], m[3]] } else { [0.; 2] },
                rms_radius: if flux > 0. {
                    ((m[5] + m[7]) / flux).sqrt()
                } else {
                    0.
                },
                rays_traced: summary.traced,
                rays_detected: m[0] as usize,
            });
            maps.push(map);
            virtual_rays += summary.virtual_rays;
            virtual_flux.add(summary.virtual_flux.value());
        }
        return Ok(PathRun {
            maps,
            stats,
            virtual_rays,
            virtual_flux: virtual_flux.value(),
        });
    }
    let all: Vec<Cell> = cells.iter().flatten().copied().collect();
    let probes = match traced {
        Some(p) => p,
        None => Probes::refine(&all, &path, ctx, progress)?,
    };
    let leaves: Vec<Vec<RayOut>> = all
        .par_iter()
        .with_min_len(64)
        .map(|c| {
            let mut out = Vec::new();
            path.leaves(&probes, c.probe, HALF, c.weight, &mut out);
            out
        })
        .collect();
    let (mut virtual_rays, mut virtual_flux) = (0usize, KahanSum::default());
    let hits: Vec<Vec<Option<(f64, f64, f64)>>> = leaves
        .into_iter()
        .map(|cell| {
            cell.into_iter()
                .map(|o| {
                    o.unwrap_or_else(|w| {
                        virtual_rays += 1;
                        virtual_flux.add(w);
                        None
                    })
                })
                .collect()
        })
        .collect();
    let offsets: Vec<usize> = cells
        .iter()
        .scan(0, |acc, v| {
            let o = *acc;
            *acc += v.len();
            Some(o)
        })
        .collect();
    let area = template.bin_area();
    let mut maps = Vec::with_capacity(fields.len());
    let mut stats = Vec::with_capacity(fields.len());
    for (fi, field) in fields.iter().enumerate() {
        let lo = offsets[fi];
        let field_hits = &hits[lo..lo + cells[fi].len()];
        let mut bins = vec![KahanSum::default(); template.values.len()];
        let (mut sw, mut swx, mut swy, mut on) = (
            KahanSum::default(),
            KahanSum::default(),
            KahanSum::default(),
            KahanSum::default(),
        );
        let mut detected = 0usize;
        for &(x, y, w) in field_hits.iter().flatten().flatten() {
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
            .flatten()
            .map(|&(x, y, w)| w * (x - cx).mul_add(x - cx, (y - cy) * (y - cy)))
            .collect();
        let mut map = template.clone();
        for (v, b) in map.values.iter_mut().zip(&bins) {
            *v = b.value() / area;
        }
        stats.push(FieldIrradiance {
            field: field.id,
            flux,
            flux_on_map: on.value(),
            peak: map.peak(),
            centroid: [cx, cy],
            rms_radius: if flux > 0.0 {
                (m2.value() / flux).sqrt()
            } else {
                0.0
            },
            rays_traced: field_hits.iter().map(Vec::len).sum(),
            rays_detected: detected,
        });
        maps.push(map);
    }
    Ok(PathRun {
        maps,
        stats,
        virtual_rays,
        virtual_flux: virtual_flux.value(),
    })
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
    ghost_irradiance_with_backend(model, settings, ctx, None)
}
pub(crate) fn ghost_irradiance_with_backend(
    model: &GhostModel,
    settings: &IrradianceSettings,
    ctx: &JobContext,
    backend: Option<&dyn mklight_gpu::ComputeBackend>,
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
    let mut warnings = model.tracer().warnings().to_vec();
    let selected = select_fields(model, settings, &mut warnings)?;
    let wls: Vec<usize> = if settings.wavelengths.is_empty() {
        (0..sys.wavelengths_um.len()).collect()
    } else {
        settings.wavelengths.clone()
    };
    if wls.iter().any(|&w| w >= sys.wavelengths_um.len()) {
        return Err(MkError::InvalidModel(
            "wavelength index out of range".into(),
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
        Some(p) => (p.clone(), false, model.coverage(p.iter().cloned())),
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
    let cells = base_cells(model, &selected, &wls, settings, ctx)?;
    let n_jobs = (paths.len() + 1) as f64;

    let pstep = model.check_path(&GhostPath::primary())?;
    let gpu = match backend {
        Some(b) => {
            let steps = std::iter::once(pstep.as_slice())
                .chain(path_steps.iter().map(Vec::as_slice))
                .collect::<Vec<_>>();
            Some(resident_maps(
                b, model, &paths, &steps, &cells, &selected, settings, &template, ctx,
            )?)
        }
        None => None,
    };
    let gpu_maps = gpu.as_ref().map(|(_, maps)| maps);
    // Reference: the primary (index 0) and the ghosts, traced in groups of
    // paths adjacent in step order, so that shared step prefixes are traced
    // once per lattice point. Results are those of tracing each path alone.
    let mut runs: Vec<Option<PathRun>> = Vec::new();
    if gpu.is_none() {
        let all_steps = std::iter::once(pstep.as_slice())
            .chain(path_steps.iter().map(Vec::as_slice))
            .collect::<Vec<_>>();
        let mut order = (0..all_steps.len()).collect::<Vec<_>>();
        order.sort_by(|&a, &b| {
            all_steps[a]
                .iter()
                .map(step_key)
                .cmp(all_steps[b].iter().map(step_key))
        });
        let all_cells = cells.iter().flatten().copied().collect::<Vec<_>>();
        runs = (0..all_steps.len()).map(|_| None).collect();
        let groups = order.len().div_ceil(GROUP_PATHS) as f64;
        for (g, group) in order.chunks(GROUP_PATHS).enumerate() {
            let members = group
                .iter()
                .map(|&i| PathTrace {
                    model,
                    steps: all_steps[i],
                    fields: &selected,
                    n: settings.pupil_samples,
                    overfill: settings.overfill,
                    map: &template,
                })
                .collect::<Vec<_>>();
            let span = (g as f64 / groups, (g + 1) as f64 / groups);
            let traced = PathGroup { paths: &members }.refine(&all_cells, ctx, span)?;
            for (&i, probes) in group.iter().zip(traced) {
                runs[i] = Some(run_path(
                    model,
                    all_steps[i],
                    &cells,
                    &selected,
                    settings,
                    &template,
                    ctx,
                    span,
                    None,
                    Some(probes),
                )?);
            }
        }
    }
    let prun = match runs.get_mut(0).and_then(Option::take) {
        Some(run) => run,
        None => run_path(
            model,
            &pstep,
            &cells,
            &selected,
            settings,
            &template,
            ctx,
            if gpu.is_some() {
                (0.9, 0.9)
            } else {
                (0.0, 1.0 / n_jobs)
            },
            gpu_maps.map(|v| (v, 0)),
            None,
        )?,
    };
    let (pmaps, pstats) = (prun.maps, prun.stats);
    if prun.virtual_rays > 0 {
        // Flux the lost rays carried when rejected, relative to the signal
        // they would have joined.
        let detected: f64 = pstats.iter().map(|f| f.flux).sum();
        warnings.push(format!(
            "{} primary rays (≈{:.3e} of the primary signal) cross a physically \
             impossible segment (negative edge thickness or overlapping \
             elements) and are excluded from the signal, although the \
             sequential trace passes them",
            prun.virtual_rays,
            prun.virtual_flux / (prun.virtual_flux + detected)
        ));
    }
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
        let span = if gpu.is_some() {
            let v = 0.9 + 0.09 * k as f64 / paths.len() as f64;
            (v, v)
        } else {
            ((k + 1) as f64 / n_jobs, (k + 2) as f64 / n_jobs)
        };
        let PathRun { maps, stats, .. } = match runs.get_mut(k + 1).and_then(Option::take) {
            Some(run) => run,
            None => run_path(
                model,
                steps,
                &cells,
                &selected,
                settings,
                &template,
                ctx,
                span,
                gpu_maps.map(|v| (v, k + 1)),
                None,
            )?,
        };
        if gpu.is_some() {
            ctx.report(
                0.9 + 0.09 * (k + 1) as f64 / paths.len() as f64,
                "assembling ghost maps",
            );
        }
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
    let total_flux_ratio = ratio(total.flux(), primary_flux);
    let total_peak_ratio = ratio(total.peak(), primary_map.peak());
    let mut run = run_info(model, timer, "ghost.irradiance", settings, warnings);
    if let Some((g, _)) = &gpu {
        g.stamp(&mut run);
    }
    let result = GhostIrradiance {
        run,
        primary,
        ghosts,
        total_ghost_map: total,
        total_peak_ratio,
        total_flux_ratio,
        truncated,
        coverage,
    };
    ctx.check()?;
    ctx.report(1., "done");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::sync::Arc;

    use mklight_core::materials::MaterialLibrary;
    use mklight_core::{ResolvedSystem, Snapshot, samples};

    use super::*;
    use crate::GhostSettings;

    /// The recursion without a cache: traces every centre and corner it
    /// visits, recording each.
    fn uncached(
        path: &PathTrace,
        centre: Probe,
        half: i32,
        weight: f64,
        out: &mut Vec<RayOut>,
        traced: &mut Vec<Probe>,
    ) {
        traced.push(centre);
        let sample = path.trace(centre);
        let mut split = false;
        if half > 1 {
            for c in centre.around(half) {
                traced.push(c);
                split |= path.region(path.trace(c)) != path.region(sample);
            }
        }
        if split {
            for c in centre.around(half / 2) {
                uncached(path, c, half / 2, weight * 0.25, out, traced);
            }
        } else {
            out.push(weighted(sample, weight));
        }
    }

    #[test]
    fn cache_traces_each_lattice_point_once_and_matches_uncached() {
        let snapshot = Snapshot::new(samples::double_gauss()).unwrap();
        let system =
            ResolvedSystem::new(&snapshot, Arc::new(MaterialLibrary::with_builtin())).unwrap();
        let model = GhostModel::from_system(Arc::new(system), GhostSettings::default()).unwrap();
        let template = IrradianceMap::new([24, 24], [12.0, 12.0]);
        // A repeated wavelength shares its probes.
        let wls = [0, 1, 1];
        for (n, overfill) in [(12, 1.5), (21, 1.3)] {
            let settings = IrradianceSettings {
                fields: vec![0, 2],
                pupil_samples: n,
                overfill,
                ..Default::default()
            };
            let fields = select_fields(&model, &settings, &mut Vec::new()).unwrap();
            let cells = base_cells(&model, &fields, &wls, &settings, &JobContext::new())
                .unwrap()
                .concat();
            for path in [
                GhostPath::primary(),
                GhostPath::new(vec![9, 7]),
                GhostPath::new(vec![11, 1]),
            ] {
                let steps = model.check_path(&path).unwrap();
                let trace = PathTrace {
                    model: &model,
                    steps: &steps,
                    fields: &fields,
                    n,
                    overfill,
                    map: &template,
                };
                let probes =
                    Probes::refine(&cells, &trace, &JobContext::new(), (0.0, 1.0)).unwrap();
                let (mut cached, mut expected, mut requested) = (vec![], vec![], vec![]);
                for c in &cells {
                    trace.leaves(&probes, c.probe, HALF, c.weight, &mut cached);
                    uncached(
                        &trace,
                        c.probe,
                        HALF,
                        c.weight,
                        &mut expected,
                        &mut requested,
                    );
                }
                assert_eq!(cached, expected, "n = {n}, path {path:?}");
                let distinct: HashSet<Probe> = requested.iter().copied().collect();
                assert!(distinct.len() < requested.len());
                assert_eq!(probes.outs.len(), distinct.len());
                assert_eq!(
                    probes.index.keys().copied().collect::<HashSet<_>>(),
                    distinct
                );
            }
        }
    }

    /// Prefix-shared group tracing reproduces every path's own probes bit
    /// for bit, including paths that end (clipped, virtual, missed) inside a
    /// shared prefix and the primary path.
    #[test]
    fn grouped_paths_match_paths_traced_alone() {
        for (sample, reject_virtual) in [
            (samples::double_gauss(), true),
            (samples::cooke_triplet(), false),
        ] {
            let snapshot = Snapshot::new(sample).unwrap();
            let system =
                ResolvedSystem::new(&snapshot, Arc::new(MaterialLibrary::with_builtin())).unwrap();
            let settings = GhostSettings {
                reject_virtual,
                ..GhostSettings::default()
            };
            let model = GhostModel::from_system(Arc::new(system), settings).unwrap();
            let template = IrradianceMap::new([16, 16], [8.0, 8.0]);
            let irr = IrradianceSettings {
                pupil_samples: 10,
                ..Default::default()
            };
            let fields = select_fields(&model, &irr, &mut Vec::new()).unwrap();
            let wls = (0..model.system().wavelengths_um.len()).collect::<Vec<_>>();
            let cells = base_cells(&model, &fields, &wls, &irr, &JobContext::new())
                .unwrap()
                .concat();
            let mut paths = enumerate(&model, &EnumerationSettings::default())
                .unwrap()
                .paths
                .into_iter()
                .map(|p| p.path)
                .collect::<Vec<_>>();
            paths.insert(0, GhostPath::primary());
            let steps = paths
                .iter()
                .map(|p| model.check_path(p).unwrap())
                .collect::<Vec<_>>();
            let mut order = (0..steps.len()).collect::<Vec<_>>();
            order.sort_by(|&a, &b| {
                steps[a]
                    .iter()
                    .map(step_key)
                    .cmp(steps[b].iter().map(step_key))
            });
            let traces = order
                .iter()
                .map(|&i| PathTrace {
                    model: &model,
                    steps: &steps[i],
                    fields: &fields,
                    n: irr.pupil_samples,
                    overfill: irr.overfill,
                    map: &template,
                })
                .collect::<Vec<_>>();
            assert!(traces.len() > 3);
            let grouped = PathGroup { paths: &traces }
                .refine(&cells, &JobContext::new(), (0.0, 1.0))
                .unwrap();
            let mut ended = 0;
            for (trace, group) in traces.iter().zip(&grouped) {
                let alone = Probes::refine(&cells, trace, &JobContext::new(), (0.0, 1.0)).unwrap();
                assert_eq!(alone.index.len(), group.index.len());
                for (probe, &id) in &alone.index {
                    let (a, g) = (alone.outs[id], group.get(*probe));
                    let bits = |o: RayOut| match o {
                        Ok(Some((x, y, w))) => [1, x.to_bits(), y.to_bits(), w.to_bits()],
                        Ok(None) => [2, 0, 0, 0],
                        Err(w) => [3, w.to_bits(), 0, 0],
                    };
                    assert_eq!(bits(a), bits(g), "{probe:?} along {:?}", trace.steps);
                    ended += matches!(a, Ok(None) | Err(_)) as usize;
                }
            }
            assert!(ended > 0);
        }
    }
}

/// Explicit resident route: breadth-first refinement, then one batched map
/// chain over every path and field. The transport keeps the run counters.
#[allow(clippy::too_many_arguments)]
fn resident_maps<'a>(
    backend: &'a dyn mklight_gpu::ComputeBackend,
    model: &'a GhostModel,
    paths: &[GhostPath],
    steps: &[&[Step]],
    cells: &[Vec<Cell>],
    fields: &[SelectedField],
    settings: &IrradianceSettings,
    template: &IrradianceMap,
    ctx: &JobContext,
) -> MkResult<(crate::gpu::GhostTransport<'a>, crate::gpu::MapData)> {
    let sys = model.system();
    // Every object segment would leave the qualified segment domain and
    // need reference completion; that run belongs to the reference route.
    if !sys.object_at_infinity
        && sys.surfaces[0].thickness.abs() > mklight_gpu::consts::MKL_GHOST_MAX_SEGMENT_MM as f64
    {
        return Err(MkError::Unsupported(
            "ghost object distance beyond the qualified segment length".into(),
        ));
    }
    let all_paths = std::iter::once(GhostPath::primary())
        .chain(paths.iter().cloned())
        .collect::<Vec<_>>();
    let mut g =
        crate::gpu::GhostTransport::new(backend, model, &all_paths, &ctx.subrange(0., 0.01))?;
    let points = fields
        .iter()
        .map(|f| match f.id {
            FieldRef::System { index } => FieldPoint::system(sys.system(), index),
            FieldRef::Point { point } => point,
        })
        .collect::<Vec<_>>();
    g.prepare_fields(&points, &ctx.subrange(0.01, 0.03))?;
    let caches = refine_gpu(
        &mut g,
        model,
        steps,
        cells,
        fields,
        settings,
        template,
        &ctx.subrange(0.03, 0.8),
    )?;
    let (mut leaves, mut summaries) = (Vec::new(), Vec::new());
    for (pi, (steps, cache)) in steps.iter().zip(&caches).enumerate() {
        let path = PathTrace {
            model,
            steps,
            fields,
            n: settings.pupil_samples,
            overfill: settings.overfill,
            map: template,
        };
        for (fi, field_cells) in cells.iter().enumerate() {
            ctx.check()?;
            let map = (pi * fields.len() + fi) as u32;
            let parts = field_cells
                .par_chunks(256)
                .map(|block| {
                    let mut part = (Vec::new(), LeafSummary::default());
                    for c in block {
                        path.resident_leaves(cache, c.probe, HALF, c.weight, map, &mut part);
                    }
                    part
                })
                .collect::<Vec<_>>();
            let mut summary = LeafSummary::default();
            for (part, s) in parts {
                leaves.extend(part);
                summary.traced += s.traced;
                summary.virtual_rays += s.virtual_rays;
                summary.virtual_flux.add(s.virtual_flux.value());
            }
            summaries.push(summary);
        }
    }
    let maps = g.map(leaves, template, summaries, &ctx.subrange(0.8, 0.9))?;
    Ok((g, maps))
}

/// Host controls the five breadth-first levels; every batch spans paths,
/// fields and wavelengths. Cache insertion and leaf traversal keep their
/// original integer IDs and depth-first order.
#[allow(clippy::too_many_arguments)]
fn refine_gpu(
    gpu: &mut crate::gpu::GhostTransport,
    model: &GhostModel,
    steps: &[&[Step]],
    cells: &[Vec<Cell>],
    fields: &[SelectedField],
    settings: &IrradianceSettings,
    template: &IrradianceMap,
    ctx: &JobContext,
) -> MkResult<Vec<Probes>> {
    use mklight_gpu::{Df32, consts::*, ghost_ops::LaunchSample};
    let paths = steps
        .iter()
        .map(|s| PathTrace {
            model,
            steps: s,
            fields,
            n: settings.pupil_samples,
            overfill: settings.overfill,
            map: template,
        })
        .collect::<Vec<_>>();
    let all = cells.iter().flatten().map(|c| c.probe).collect::<Vec<_>>();
    let mut levels = vec![all; steps.len()];
    let mut caches = (0..steps.len())
        .map(|_| {
            let mut p = Probes::default();
            p.index.reserve(4 * levels[0].len());
            p
        })
        .collect::<Vec<_>>();
    let nw = model.system().wavelengths_um.len() as u32;
    let mut half = HALF;
    for stage in 0..=BOUNDARY_DEPTH {
        for corners in [false, true] {
            if corners && half == 1 {
                break;
            }
            // Paths own disjoint caches: deduplicate them in parallel, then
            // order new keys by (path, field, wavelength, x, y). Keys are
            // unique, so the unstable sort is the stable order.
            ctx.check()?;
            let new = levels
                .par_iter()
                .zip(&mut caches)
                .enumerate()
                .map(|(pi, (level, cache))| {
                    let mut new = Vec::new();
                    for &centre in level {
                        let around = centre.around(half);
                        let requested = if corners {
                            &around[..]
                        } else {
                            std::slice::from_ref(&centre)
                        };
                        for &p in requested {
                            if let Entry::Vacant(e) = cache.index.entry(p) {
                                let id = cache.outs.len();
                                e.insert(id);
                                cache.outs.push(Ok(None));
                                cache.locations.push(None);
                                let xy = [p.x, p.y]
                                    .map(|u| pupil(u, settings.pupil_samples, settings.overfill));
                                if xy[0].mul_add(xy[0], xy[1] * xy[1])
                                    <= settings.overfill * settings.overfill
                                {
                                    new.push((pi, p, id, xy));
                                }
                            }
                        }
                    }
                    new.sort_unstable_by_key(|&(_, p, _, _)| (p.field, p.wl, p.x, p.y));
                    new
                })
                .collect::<Vec<_>>()
                .concat();
            ctx.check()?;
            // A memory tile of probe records (every path, field and
            // wavelength together), not a new plan per path.
            for block in new.chunks(crate::gpu::PROBE_TILE) {
                ctx.check()?;
                let values = gpu.lattice(
                    block.len(),
                    |i| {
                        let (pi, p, _, xy) = block[i];
                        Ok(LaunchSample {
                            pupil: xy.map(Df32::from_f64),
                            lattice: [p.x, p.y],
                            field: p.field * nw + p.wl,
                            path: pi as u32,
                            wavelength: p.wl,
                            reserved: 0,
                        })
                    },
                    true,
                    &ctx.subrange(stage as f64 / 5., stage as f64 / 5.),
                )?;
                // Numerical failures, flagged conditioning and pixel-edge hits
                // are completed with the f64 reference, in parallel.
                let outcomes = block
                    .par_iter()
                    .zip(&values[..])
                    .with_min_len(256)
                    .map(|(&(pi, p, _, _), v)| {
                        let geometric = crate::gpu::ray_out(*v);
                        let edge =
                            matches!(geometric, Ok(Some((x, y, _))) if pixel_edge(template, x, y));
                        let code = v.status & MKL_RAY_CODE_MASK;
                        let numerical = (MKL_RAY_NONFINITE..=MKL_RAY_EVANESCENT).contains(&code);
                        if code == MKL_GHOST_REFERENCE || numerical || edge {
                            let start = std::time::Instant::now();
                            let reference = paths[pi].trace(p);
                            (geometric, edge, Some((reference, start.elapsed())))
                        } else {
                            (geometric, false, None)
                        }
                    })
                    .collect::<Vec<_>>();
                ctx.check()?;
                let batch = gpu.resident_probes.len() - 1;
                let mut replacements = Vec::new();
                for (row, (&(pi, _, id, _), (geometric, edge, reference))) in
                    block.iter().zip(outcomes).enumerate()
                {
                    let result = match reference {
                        Some((reference, seconds)) => {
                            gpu.fallback_probes += 1;
                            gpu.reference_seconds += seconds.as_secs_f64();
                            if edge {
                                gpu.boundary_probes += 1;
                                if let (Ok(Some((x, y, _))), Ok(Some((rx, ry, _)))) =
                                    (geometric, reference)
                                {
                                    gpu.pixel_migrations +=
                                        (template.bin_of(x, y) != template.bin_of(rx, ry)) as usize;
                                }
                            }
                            replacements.push((row, reference));
                            reference
                        }
                        None => geometric,
                    };
                    caches[pi].outs[id] = result;
                    caches[pi].locations[id] = Some([batch, row]);
                }
                if !replacements.is_empty() {
                    let batch = gpu.reference_batch(&replacements, ctx)?;
                    for (row, &(original, _)) in replacements.iter().enumerate() {
                        let (pi, _, id, _) = block[original];
                        caches[pi].locations[id] = Some([batch, row]);
                    }
                }
            }
        }
        ctx.check()?;
        ctx.report((stage + 1) as f64 / 5., "refining ghost probes");
        if half == 1 {
            break;
        }
        for (pi, level) in levels.iter_mut().enumerate() {
            *level = level
                .par_iter()
                .with_min_len(1024)
                .filter(|&&p| paths[pi].splits(&caches[pi], p, half))
                .flat_map_iter(|p| p.around(half / 2))
                .collect();
            ctx.check()?;
        }
        half /= 2;
    }
    gpu.lattice_keys = caches.iter().map(|c| c.outs.len()).sum();
    Ok(caches)
}

/// Point binning is discontinuous: a geometrically qualified hit can move
/// between pixels. Resolve only this uncertainty band with the reference,
/// retaining its bin decision (including f64's rounding at an exact edge).
fn pixel_edge(map: &IrradianceMap, x: f64, y: f64) -> bool {
    [x, y].into_iter().enumerate().any(|(k, x)| {
        let width = 2. * map.half_size[k] / map.bins[k] as f64;
        let f = (x + map.half_size[k]) / width;
        f >= -1. && f <= map.bins[k] as f64 + 1. && (f - f.round()).abs() * width <= 2e-5
    })
}

#[cfg(all(test, target_os = "macos"))]
#[test]
fn p9_dense_achromat_probe_accuracy() {
    use mklight_core::{ResolvedSystem, Snapshot, materials::MaterialLibrary, samples};
    use mklight_gpu::{ComputeBackend, Df32, consts::*, ghost_ops::Probe as Input};
    use std::sync::Arc;
    let snapshot = Snapshot::new(samples::achromatic_doublet()).unwrap();
    let sys = ResolvedSystem::new(&snapshot, Arc::new(MaterialLibrary::with_builtin())).unwrap();
    let model = GhostModel::from_system(Arc::new(sys), crate::GhostSettings::default()).unwrap();
    let settings = IrradianceSettings {
        pupil_samples: 128,
        fields: vec![1],
        ..Default::default()
    };
    let fields = select_fields(&model, &settings, &mut Vec::new()).unwrap();
    let wls = (0..model.system().wavelengths_um.len()).collect::<Vec<_>>();
    let cells = base_cells(&model, &fields, &wls, &settings, &JobContext::new())
        .unwrap()
        .concat();
    let path = GhostPath::new(vec![2, 1]);
    let steps = model.check_path(&path).unwrap();
    let map = IrradianceMap::new([64, 64], [12.; 2]);
    let trace = PathTrace {
        model: &model,
        steps: &steps,
        fields: &fields,
        n: 128,
        overfill: 1.5,
        map: &map,
    };
    let cache = Probes::refine(&cells, &trace, &JobContext::new(), (0., 1.)).unwrap();
    let b = mklight_metal::MetalBackend::new()
        .unwrap()
        .expect("P9 requires host Metal");
    let transport = crate::gpu::GhostTransport::new(
        &b,
        &model,
        std::slice::from_ref(&path),
        &JobContext::new(),
    )
    .unwrap();
    let mut keys = cache
        .index
        .keys()
        .copied()
        .filter(|p| {
            let xy = [p.x, p.y].map(|u| pupil(u, 128, 1.5));
            xy[0].mul_add(xy[0], xy[1] * xy[1]) <= 2.25
        })
        .collect::<Vec<_>>();
    keys.sort_by_key(|p| (p.wl, p.x, p.y));
    let (mut max_position, mut max_weight, mut reference) = (0.0f64, 0.0f64, 0usize);
    let mut flagged_inputs = Vec::new();
    let mut flagged_values = Vec::new();
    let mut by_surface = std::collections::BTreeMap::<u32, usize>::new();
    for block in keys.chunks(8192) {
        let inputs = block
            .iter()
            .map(|p| {
                let xy = [p.x, p.y].map(|u| pupil(u, 128, 1.5));
                let f = &fields[0].field;
                let c = f.aims[p.wl as usize].chief;
                let scale = model.tracer().pupil().r_ep;
                let ray = model
                    .tracer()
                    .start_ray(
                        &f.source,
                        [xy[0].mul_add(scale, c[0]), xy[1].mul_add(scale, c[1])],
                    )
                    .unwrap();
                Input {
                    position: [ray.pos.x, ray.pos.y, ray.pos.z].map(Df32::from_f64),
                    direction: [ray.dir.x, ray.dir.y, ray.dir.z].map(Df32::from_f64),
                    path: 0,
                    wavelength: p.wl,
                }
            })
            .collect::<Vec<_>>();
        let result = transport.trace(&inputs, &JobContext::new()).unwrap();
        for (i, (&p, v)) in block.iter().zip(result.values.iter()).enumerate() {
            if v.status & MKL_RAY_CODE_MASK == MKL_GHOST_REFERENCE {
                reference += 1;
                let count = by_surface
                    .entry(v.status >> MKL_RAY_SURFACE_SHIFT)
                    .or_default();
                *count += 1;
                if *count <= 16 {
                    flagged_inputs.push(inputs[i]);
                    flagged_values.push(*v);
                }
                continue;
            }
            let actual = crate::gpu::ray_out(*v);
            let expected = cache.get(p);
            assert_eq!(
                matches!(actual, Ok(Some(_))),
                matches!(expected, Ok(Some(_))),
                "{p:?} code {}",
                v.status
            );
            if let (Ok(Some((x, y, w))), Ok(Some((rx, ry, rw)))) = (actual, expected) {
                let error = (x - rx).abs().max((y - ry).abs());
                max_position = max_position.max(error);
                max_weight = max_weight.max((w - rw).abs() / rw.max(1e-10));
                assert!(error <= 2e-5, "{p:?} hit error {error:e}, hit ({rx},{ry})");
                assert!(
                    (w - rw).abs() <= 1e-5 * rw.abs() + 1e-10,
                    "{p:?} weight error {}",
                    w - rw
                );
            }
        }
    }
    eprintln!(
        "dense achromat probes={} reference={reference} max_position={max_position:e} max_weight_relative={max_weight:e}",
        keys.len()
    );
    assert!(reference > 0);
    let cpu = mklight_gpu::CpuF32Backend::matching(&b.precision().probe);
    let emulated = crate::gpu::GhostTransport::new(
        &cpu,
        &model,
        std::slice::from_ref(&path),
        &JobContext::new(),
    )
    .unwrap()
    .trace(&flagged_inputs, &JobContext::new())
    .unwrap();
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&flagged_values),
        bytemuck::cast_slice::<_, u32>(&emulated.values)
    );
    eprintln!("reference flags by surface={by_surface:?}");
    let mut gpu = crate::gpu::GhostTransport::new(
        &b,
        &model,
        std::slice::from_ref(&path),
        &JobContext::new(),
    )
    .unwrap();
    gpu.prepare_fields(
        &[FieldPoint::system(model.system().system(), 1)],
        &JobContext::new(),
    )
    .unwrap();
    let native = refine_gpu(
        &mut gpu,
        &model,
        &[steps.as_slice()],
        std::slice::from_ref(&cells),
        &fields,
        &settings,
        &map,
        &JobContext::new(),
    )
    .unwrap();
    fn leaf_keys(
        trace: &PathTrace,
        cache: &Probes,
        p: Probe,
        half: i32,
        out: &mut Vec<(Probe, i32)>,
    ) {
        if trace.splits(cache, p, half) {
            for child in p.around(half / 2) {
                leaf_keys(trace, cache, child, half / 2, out);
            }
        } else {
            out.push((p, half));
        }
    }
    let (mut expected_leaves, mut actual_leaves) = (Vec::new(), Vec::new());
    for cell in &cells {
        leaf_keys(&trace, &cache, cell.probe, HALF, &mut expected_leaves);
        leaf_keys(&trace, &native[0], cell.probe, HALF, &mut actual_leaves);
    }
    assert_eq!(expected_leaves, actual_leaves);
    assert_eq!(
        cache
            .index
            .keys()
            .copied()
            .collect::<std::collections::HashSet<_>>(),
        native[0].index.keys().copied().collect()
    );
    eprintln!(
        "native adaptive leaf set={} lattice_keys={} transport_probes={} reference_probes={}",
        actual_leaves.len(),
        gpu.lattice_keys,
        gpu.distinct_probes,
        gpu.fallback_probes
    );
}

/// Clear apertures sized by the edge field's rim rays put lattice points of
/// that field exactly on an aperture edge. The native refinement must make
/// the reference's decision there (aperture ties), so every path refines
/// the same lattice and counts the same leaves as the CPU f64 reference.
#[cfg(all(test, target_os = "macos"))]
#[test]
fn native_refinement_matches_reference_at_aperture_edges() {
    use mklight_core::{ResolvedSystem, Snapshot, materials::MaterialLibrary, samples};
    use std::sync::Arc;
    for (name, sys) in [
        ("singlet", samples::singlet()),
        ("achromat", samples::achromatic_doublet()),
        ("petzval", samples::petzval_asphere()),
        ("newtonian", samples::newtonian_telescope()),
        ("q_con", samples::q_asphere_lens()),
    ] {
        let snapshot = Snapshot::new(sys).unwrap();
        let rs = ResolvedSystem::new(&snapshot, Arc::new(MaterialLibrary::with_builtin())).unwrap();
        let model = GhostModel::from_system(Arc::new(rs), crate::GhostSettings::default()).unwrap();
        let settings = IrradianceSettings {
            pupil_samples: 48,
            ..Default::default()
        };
        let fields = select_fields(&model, &settings, &mut Vec::new()).unwrap();
        let wls = (0..model.system().wavelengths_um.len()).collect::<Vec<_>>();
        let cells = base_cells(&model, &fields, &wls, &settings, &JobContext::new()).unwrap();
        let osys = model.system().system();
        let image = model.system().image_index();
        let half = match &osys.sensor {
            Some(s) => [s.half_width(), s.half_height()],
            None => {
                let r = model.semi_diameters().values[image].max(1e-3);
                [r, r]
            }
        };
        let map = IrradianceMap::new([64, 64], half);
        let mut paths = enumerate(&model, &settings.enumeration)
            .unwrap()
            .paths
            .into_iter()
            .map(|p| p.path)
            .filter(|p| p.ends_at_image())
            .collect::<Vec<_>>();
        paths.insert(0, GhostPath::primary());
        let steps = paths
            .iter()
            .map(|p| model.check_path(p).unwrap())
            .collect::<Vec<_>>();
        let b = mklight_metal::MetalBackend::new().unwrap().unwrap();
        let mut gpu =
            crate::gpu::GhostTransport::new(&b, &model, &paths, &JobContext::new()).unwrap();
        let points = fields
            .iter()
            .map(|f| match f.id {
                FieldRef::System { index } => FieldPoint::system(osys, index),
                FieldRef::Point { point } => point,
            })
            .collect::<Vec<_>>();
        gpu.prepare_fields(&points, &JobContext::new()).unwrap();
        let sl = steps.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let native = refine_gpu(
            &mut gpu,
            &model,
            &sl,
            &cells,
            &fields,
            &settings,
            &map,
            &JobContext::new(),
        )
        .unwrap();
        let all = cells.concat();
        for (pi, st) in steps.iter().enumerate() {
            let trace = PathTrace {
                model: &model,
                steps: st,
                fields: &fields,
                n: 48,
                overfill: 1.5,
                map: &map,
            };
            let cpu = Probes::refine(&all, &trace, &JobContext::new(), (0., 1.)).unwrap();
            for (p, &id) in &cpu.index {
                let g = native[pi].index.get(p).map(|&g| native[pi].outs[g]);
                let c = cpu.outs[id];
                assert_eq!(
                    g.map(|g| trace.region(g)),
                    Some(trace.region(c)),
                    "{name} path {:?} {p:?}: reference {c:?}, native {g:?}",
                    paths[pi]
                );
            }
            assert_eq!(
                cpu.index.len(),
                native[pi].index.len(),
                "{name} {:?}",
                paths[pi]
            );
        }
    }
}
