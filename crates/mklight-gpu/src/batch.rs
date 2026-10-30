//! Structure-of-arrays ray batches exchanged with trace kernels.
//!
//! Rays enter on a chosen surface, expressed in that surface's local frame
//! (CONVENTIONS §7.1): the host (the f64 reference code: field/pupil mapping,
//! ray aiming) produces them, as f32 or as double-float hi + lo pairs
//! ([`RayBatch::push_f64`]), and the kernel carries them from surface to
//! surface with relative transforms only. Optical path is carried in
//! double-float (hi/lo pairs). A trace runs a path program
//! ([`PathStep`]): the sequential train, or an explicit list of transmit /
//! reflect steps forward or backward (ghost paths).

use mklight_core::{MkError, MkResult};
use serde::{Deserialize, Serialize};

use crate::backend::SystemHandle;
use crate::consts::*;
use crate::df::Df32;
use crate::layout::{PackedSurface, PackedSystemSet, TraceParams};
use crate::trace_kernel::{DEFAULT_MAX_ITERATIONS, DEFAULT_TOL_ABS, DEFAULT_TOL_REL};

/// Input rays (SoA; every vector has the same length). Construct with
/// `RayBatch::default()` and fill the fields; the struct may gain optional
/// per-ray fields (empty by default), so it is `non_exhaustive`.
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct RayBatch {
    /// Position (mm) in the local frame of `TraceRequest::start_surface`.
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    /// Unit direction cosines in the same frame.
    pub l: Vec<f32>,
    pub m: Vec<f32>,
    pub n: Vec<f32>,
    /// Optical path already accumulated (mm), double-float.
    pub opl_hi: Vec<f32>,
    pub opl_lo: Vec<f32>,
    /// Packed wavelength index (0..num_wavelengths) per ray.
    pub wavelength: Vec<u32>,
    /// System index within the uploaded set per ray; empty means all 0.
    pub system: Vec<u32>,
    /// Flux weight per ray (relative units or W, CONVENTIONS §1); empty
    /// means all 1. Carried through the trace and multiplied by the path
    /// transmittance (see [`Radiometry`]).
    pub weight: Vec<f32>,
    /// Optional double-float residuals of the position and direction
    /// (`x + x_lo` …): all six empty (f32 input), or all of the batch's
    /// length. The kernel works in double-float from the start, so an f64
    /// ray given with its residuals ([`RayBatch::push_f64`]) is traced
    /// without input rounding (to ~2⁻⁴⁸).
    pub x_lo: Vec<f32>,
    pub y_lo: Vec<f32>,
    pub z_lo: Vec<f32>,
    pub l_lo: Vec<f32>,
    pub m_lo: Vec<f32>,
    pub n_lo: Vec<f32>,
}

impl RayBatch {
    /// An empty batch with room for `n` rays.
    pub fn with_capacity(n: usize) -> Self {
        let f = || Vec::with_capacity(n);
        RayBatch {
            x: f(),
            y: f(),
            z: f(),
            l: f(),
            m: f(),
            n: f(),
            opl_hi: f(),
            opl_lo: f(),
            wavelength: Vec::with_capacity(n),
            system: Vec::new(),
            weight: Vec::new(),
            x_lo: f(),
            y_lo: f(),
            z_lo: f(),
            l_lo: f(),
            m_lo: f(),
            n_lo: f(),
        }
    }

    /// Append a ray given in f64 (the reference generator's output) as
    /// double-float position and direction (`hi = fl32(v)`, `lo = fl32(v −
    /// hi)`), with zero starting optical path. A batch must be filled either
    /// only with `push_f64` or with all residual planes left empty.
    pub fn push_f64(&mut self, pos: [f64; 3], dir: [f64; 3], wavelength: u32) {
        let split = |v: f64| {
            let d = Df32::from_f64(v);
            (d.hi, d.lo)
        };
        let (x, xl) = split(pos[0]);
        let (y, yl) = split(pos[1]);
        let (z, zl) = split(pos[2]);
        let (l, ll) = split(dir[0]);
        let (m, ml) = split(dir[1]);
        let (n, nl) = split(dir[2]);
        self.x.push(x);
        self.y.push(y);
        self.z.push(z);
        self.l.push(l);
        self.m.push(m);
        self.n.push(n);
        self.x_lo.push(xl);
        self.y_lo.push(yl);
        self.z_lo.push(zl);
        self.l_lo.push(ll);
        self.m_lo.push(ml);
        self.n_lo.push(nl);
        self.opl_hi.push(0.0);
        self.opl_lo.push(0.0);
        self.wavelength.push(wavelength);
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
    pub fn validate(&self) -> MkResult<()> {
        let n = self.len();
        let lens = [
            self.y.len(),
            self.z.len(),
            self.l.len(),
            self.m.len(),
            self.n.len(),
            self.opl_hi.len(),
            self.opl_lo.len(),
            self.wavelength.len(),
        ];
        let optional_ok = |k: usize| k == 0 || k == n;
        let lo = [
            self.x_lo.len(),
            self.y_lo.len(),
            self.z_lo.len(),
            self.l_lo.len(),
            self.m_lo.len(),
            self.n_lo.len(),
        ];
        let lo_ok = lo.iter().all(|&k| k == 0) || lo.iter().all(|&k| k == n);
        if lens.iter().any(|&k| k != n)
            || !optional_ok(self.system.len())
            || !optional_ok(self.weight.len())
            || !lo_ok
        {
            return Err(MkError::Backend("ray batch arrays differ in length".into()));
        }
        if u32::try_from(n).is_err() {
            return Err(MkError::Backend("ray batch: more than 2^32 rays".into()));
        }
        Ok(())
    }
}

/// What the trace kernel records besides the final ray state.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RecordMode {
    /// Only the state on `end_surface`.
    #[default]
    Final,
    /// The full ray state and status on every surface from start to end (for
    /// layout drawings, footprints, auto semi-diameters, per-surface
    /// throughput and ghost bookkeeping).
    AllSurfaces,
}

/// How the trace carries flux (CONVENTIONS §5). The enum is
/// `non_exhaustive`: a polarised mode (a Jones vector or Stokes state per ray,
/// with extra SoA fields in [`RayBatch`]/[`TraceResult`]) is added as a new
/// variant without changing the existing ones.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum Radiometry {
    /// Geometry only: weights pass through unchanged.
    Off,
    /// Unpolarised flux: the weight is multiplied by the internal
    /// transmittance `exp(−α·L)` of every medium crossed (`alpha_after`) and
    /// by the unpolarised transmittance (or reflectance) of every interface:
    /// bare Fresnel (`mklight_core::materials::coating::fresnel`, the model
    /// every solver uses) on refracting surfaces, `R = 1`, `T = 0` on bare
    /// mirrors, the constant `R`/`T` of ideal coatings; ideal lenses are
    /// lossless unless coated. Thin-film and tabulated coatings are not
    /// available on the GPU (`MkError::Unsupported`), so `T + R + A = 1`
    /// bookkeeping never silently changes.
    #[default]
    Unpolarized,
}

/// Number of f32 values recorded per ray per surface with
/// [`RecordMode::AllSurfaces`], in this order: x, y, z, l, m, n, opl_hi,
/// opl_lo, weight.
pub const RECORD_FLOATS: usize = MKL_TRACE_RECORD_FLOATS as usize;

/// What a path step does at its surface (`MKL_STEP_*`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepMode {
    /// The surface's own interaction, exactly as `mklight_trace::trace_ray`
    /// (forward steps only; the image surface never interacts).
    Sequential,
    /// Refraction into the medium on the far side (a pass for dummy surfaces
    /// and the image; an ideal lens acts as a lens). Not allowed at mirrors:
    /// the sequential model has no medium behind a mirror.
    Transmit,
    /// Specular reflection (not allowed at coordinate breaks). At a mirror
    /// it is the mirror's own reflection (a mirror-type ideal lens keeps
    /// its power) and the ray continues in the same direction through the
    /// surface sequence; at any other surface it is a ghost reflection
    /// and the direction reverses.
    Reflect,
}

/// One step of a path program: move from the current surface to `surface`
/// (one index forward, or one backward) and interact there.
///
/// A sequential trace from `s` to `e` is `(s+1, Sequential) … (e,
/// Sequential)`; a ghost path reflecting at surfaces `j > i` is
/// `… (j, Reflect), (j−1, Transmit, backward) … (i, Reflect), (i+1,
/// Transmit) …`; a mirror on the way is passed with a `Reflect` step that
/// keeps the direction (see [`PathStep::validate_program`]). The gap
/// between surfaces `j` and `j + 1` always has the medium after surface
/// `j`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PathStep {
    pub surface: u32,
    pub mode: StepMode,
    pub backward: bool,
}

impl PathStep {
    pub fn forward(surface: u32, mode: StepMode) -> Self {
        PathStep {
            surface,
            mode,
            backward: false,
        }
    }
    pub fn backward(surface: u32, mode: StepMode) -> Self {
        PathStep {
            surface,
            mode,
            backward: true,
        }
    }

    /// The kernel's step word.
    pub fn encode(self) -> u32 {
        let mode = match self.mode {
            StepMode::Sequential => MKL_STEP_SEQUENTIAL,
            StepMode::Transmit => MKL_STEP_TRANSMIT,
            StepMode::Reflect => MKL_STEP_REFLECT,
        };
        (self.surface & MKL_STEP_SURFACE_MASK)
            | (mode << MKL_STEP_MODE_SHIFT)
            | if self.backward { MKL_STEP_BACKWARD } else { 0 }
    }

    pub fn decode(word: u32) -> Option<Self> {
        let mode = match (word >> MKL_STEP_MODE_SHIFT) & MKL_STEP_MODE_MASK {
            MKL_STEP_SEQUENTIAL => StepMode::Sequential,
            MKL_STEP_TRANSMIT => StepMode::Transmit,
            MKL_STEP_REFLECT => StepMode::Reflect,
            _ => return None,
        };
        let known =
            MKL_STEP_SURFACE_MASK | (MKL_STEP_MODE_MASK << MKL_STEP_MODE_SHIFT) | MKL_STEP_BACKWARD;
        if word & !known != 0 {
            return None;
        }
        Some(PathStep {
            surface: word & MKL_STEP_SURFACE_MASK,
            mode,
            backward: word & MKL_STEP_BACKWARD != 0,
        })
    }

    /// The sequential program from `start` to `end` (inclusive).
    pub fn sequential(start: u32, end: u32) -> Vec<PathStep> {
        (start + 1..=end)
            .map(|s| PathStep::forward(s, StepMode::Sequential))
            .collect()
    }

    /// Check a program starting on surface `start` of a system whose
    /// packed surfaces are `surfaces`: 1 ≤ length ≤ `MKL_MAX_STEPS`; every
    /// step moves one surface in its stated direction and stays in range;
    /// sequential steps go forward; no reflection at a coordinate break and
    /// no transmission through a mirror; and the direction of travel
    /// through the surface sequence follows the physics of each step:
    ///
    /// * a transmission (and any step at a coordinate break) keeps it;
    /// * a reflection at a refracting, dummy or image surface reverses it
    ///   (a ghost reflection);
    /// * a reflection at a mirror (`MKL_INTERACT_REFLECT`, including a
    ///   mirror-type ideal lens) keeps it, in either direction: in the
    ///   sequential model the gaps on both sides of a mirror are its
    ///   reflecting side, so the light continues to the next surface of
    ///   the sequence (forward, as in a sequential trace) or to the
    ///   previous one (backward, a ghost travelling back through a
    ///   catadioptric system).
    pub fn validate_program(
        steps: &[PathStep],
        start: u32,
        surfaces: &[PackedSurface],
    ) -> MkResult<()> {
        let bad = |m: String| Err(MkError::Backend(format!("path program: {m}")));
        let num_surfaces = surfaces.len() as u32;
        if steps.is_empty() || steps.len() > MKL_MAX_STEPS as usize {
            return bad(format!("{} steps (1..={MKL_MAX_STEPS})", steps.len()));
        }
        if start >= num_surfaces {
            return bad(format!("start surface {start} of {num_surfaces}"));
        }
        let mut cur = start;
        let mut heading: Option<bool> = None; // Some(backward)
        for (k, st) in steps.iter().enumerate() {
            let want = if st.backward {
                cur.checked_sub(1)
            } else {
                Some(cur + 1)
            };
            if want != Some(st.surface) || st.surface >= num_surfaces {
                return bad(format!(
                    "step {k} goes from surface {cur} to {} ({})",
                    st.surface,
                    if st.backward { "backward" } else { "forward" }
                ));
            }
            if st.mode == StepMode::Sequential && st.backward {
                return bad(format!("step {k}: sequential steps go forward"));
            }
            if let Some(h) = heading
                && h != st.backward
            {
                return bad(format!("step {k}: direction does not follow the path"));
            }
            let surf = &surfaces[st.surface as usize];
            let cb = surf.shape == MKL_SHAPE_COORD_BREAK;
            let mirror = !cb && surf.interaction == MKL_INTERACT_REFLECT;
            if st.mode == StepMode::Reflect && cb {
                return bad(format!("step {k}: reflection at a coordinate break"));
            }
            if st.mode == StepMode::Transmit && mirror {
                return bad(format!(
                    "step {k}: transmission through mirror surface {}",
                    st.surface
                ));
            }
            heading = Some(match st.mode {
                StepMode::Sequential => false,
                StepMode::Transmit => st.backward,
                StepMode::Reflect => st.backward != !mirror,
            });
            cur = st.surface;
        }
        Ok(())
    }
}

/// Asphere Newton settings of the trace kernel (f32 counterparts of
/// `mklight_trace::IntersectSettings`): converged when the sag residual is
/// at most `tolerance_abs + tolerance_rel·(|t| + |z| + |conic sag| +
/// |polynomial|)` (the f32 resolution of the residual; the converged root is
/// then refined once in double-float), classified as the reference does
/// after `max_iterations` steps (default 40, the reference's cap; only rays
/// that need many steps pay for it). With a cap below the reference's, a
/// search the cap stops is `NOT_CONVERGED`, never `MISS`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewtonSettings {
    pub max_iterations: u32,
    pub tolerance_abs: f32,
    pub tolerance_rel: f32,
}

impl Default for NewtonSettings {
    fn default() -> Self {
        NewtonSettings {
            max_iterations: DEFAULT_MAX_ITERATIONS,
            tolerance_abs: DEFAULT_TOL_ABS,
            tolerance_rel: DEFAULT_TOL_REL,
        }
    }
}

/// A batch of rays to trace through an uploaded system. Build with
/// [`TraceRequest::new`] and adjust the public fields; new fields get
/// defaults, so the struct is `non_exhaustive`.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct TraceRequest {
    pub systems: SystemHandle,
    pub rays: RayBatch,
    pub start_surface: u32,
    /// Last surface to reach (inclusive), usually the image.
    pub end_surface: u32,
    pub record: RecordMode,
    /// Clip rays at apertures (vignetting). Off for unvignetted analyses.
    pub apply_apertures: bool,
    pub radiometry: Radiometry,
    /// A path program instead of the sequential trace from `start_surface`
    /// to `end_surface` (`end_surface` is then unused). Records hold one
    /// slot per step.
    pub path: Option<Vec<PathStep>>,
    pub newton: NewtonSettings,
}

impl TraceRequest {
    /// Trace `rays` from `start_surface` to `end_surface` with apertures
    /// applied, unpolarised radiometry and only the final state recorded.
    pub fn new(
        systems: SystemHandle,
        rays: RayBatch,
        start_surface: u32,
        end_surface: u32,
    ) -> Self {
        TraceRequest {
            systems,
            rays,
            start_surface,
            end_surface,
            record: RecordMode::Final,
            apply_apertures: true,
            radiometry: Radiometry::Unpolarized,
            path: None,
            newton: NewtonSettings::default(),
        }
    }

    /// Trace `rays` (in `start_surface`'s frame) along a path program.
    pub fn path(
        systems: SystemHandle,
        rays: RayBatch,
        start_surface: u32,
        steps: Vec<PathStep>,
    ) -> Self {
        let end = steps.last().map_or(start_surface, |s| s.surface);
        let mut r = TraceRequest::new(systems, rays, start_surface, end);
        r.path = Some(steps);
        r
    }

    /// The program this request runs.
    pub fn program(&self) -> Vec<PathStep> {
        match &self.path {
            Some(p) => p.clone(),
            None => PathStep::sequential(self.start_surface, self.end_surface),
        }
    }

    /// Backend-independent checks (surface ranges against a specific system
    /// are checked by the backend against the stored set).
    pub fn validate(&self) -> MkResult<()> {
        self.rays.validate()?;
        if self.path.is_none() && self.start_surface >= self.end_surface {
            return Err(MkError::Backend(
                "trace: start_surface must precede end_surface".into(),
            ));
        }
        if let Some(&s) = self
            .rays
            .system
            .iter()
            .find(|&&s| s >= self.systems.count())
        {
            return Err(MkError::Backend(format!(
                "trace: ray system index {s} out of range"
            )));
        }
        Ok(())
    }
}

/// Trace output (SoA, same order as the input).
#[derive(Clone, Debug, Default, PartialEq)]
#[non_exhaustive]
pub struct TraceResult {
    pub x: Vec<f32>,
    pub y: Vec<f32>,
    pub z: Vec<f32>,
    pub l: Vec<f32>,
    pub m: Vec<f32>,
    pub n: Vec<f32>,
    pub opl_hi: Vec<f32>,
    pub opl_lo: Vec<f32>,
    /// Input weight × path transmittance (see [`Radiometry`]); the lost
    /// fraction `1 − weight/input` is the path's reflection + absorption.
    pub weight: Vec<f32>,
    /// Internal (bulk absorption) transmittance `exp(−Σ α·t)` of the path,
    /// whatever the radiometry mode (as `mklight_trace::TraceResult`).
    pub transmittance: Vec<f32>,
    /// `MKL_RAY_*` code | surface << `MKL_RAY_SURFACE_SHIFT`.
    pub status: Vec<u32>,
    /// Surface of the reported state: the last surface on success, the
    /// failing surface otherwise.
    pub last_surface: Vec<u32>,
    /// Path steps completed (the program length on success).
    pub steps: Vec<u32>,
    /// With [`RecordMode::AllSurfaces`]: `[slot][ray][k]` for
    /// `k < RECORD_FLOATS` (x, y, z, l, m, n, opl_hi, opl_lo, weight),
    /// flattened, with `steps + 1` slots: slot 0 the input, slot `j + 1` the
    /// state after step `j` (for a sequential trace, slot `i` is surface
    /// `start + i`). A failed ray repeats its final state in later slots.
    /// Empty otherwise.
    pub recorded: Vec<f32>,
    /// With [`RecordMode::AllSurfaces`]: `[slot][ray]` status word
    /// (`MKL_RAY_OK` while the ray is alive). Empty otherwise.
    pub recorded_status: Vec<u32>,
}

impl TraceResult {
    pub fn len(&self) -> usize {
        self.x.len()
    }
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
    /// Decoded status of ray `i`.
    pub fn ray_status(&self, i: usize) -> RayStatus {
        RayStatus::decode(self.status[i])
    }
    /// Optical path of ray `i` as an exact f64 (`hi + lo`).
    pub fn opl_f64(&self, i: usize) -> f64 {
        self.opl_hi[i] as f64 + self.opl_lo[i] as f64
    }

    /// Unpack the kernel's output planes (`MKL_TRACE_OUT_*`, `MKL_TRACE_OUTU_*`,
    /// each `stride` long) and records into a result of `n` rays.
    pub fn from_planes(
        out_f: &[f32],
        out_u: &[u32],
        n: usize,
        stride: usize,
        rec_f: Vec<f32>,
        rec_u: Vec<u32>,
    ) -> Self {
        let pf = |k: u32| out_f[k as usize * stride..k as usize * stride + n].to_vec();
        let pu = |k: u32| out_u[k as usize * stride..k as usize * stride + n].to_vec();
        TraceResult {
            x: pf(MKL_TRACE_OUT_X),
            y: pf(MKL_TRACE_OUT_Y),
            z: pf(MKL_TRACE_OUT_Z),
            l: pf(MKL_TRACE_OUT_L),
            m: pf(MKL_TRACE_OUT_M),
            n: pf(MKL_TRACE_OUT_N),
            opl_hi: pf(MKL_TRACE_OUT_OPL_HI),
            opl_lo: pf(MKL_TRACE_OUT_OPL_LO),
            weight: pf(MKL_TRACE_OUT_WEIGHT),
            transmittance: pf(MKL_TRACE_OUT_TRANSMITTANCE),
            status: pu(MKL_TRACE_OUTU_STATUS),
            last_surface: pu(MKL_TRACE_OUTU_LAST_SURFACE),
            steps: pu(MKL_TRACE_OUTU_STEPS),
            recorded: rec_f,
            recorded_status: rec_u,
        }
    }
}

/// A [`TraceRequest`] checked against the uploaded system set: the encoded
/// program and kernel parameters every backend launches with.
#[derive(Clone, Debug)]
pub struct PreparedTrace {
    /// Encoded path program (`MKL_STEP_*` words).
    pub steps: Vec<u32>,
    /// Kernel parameters; `num_rays` and `stride` are set per launch.
    pub params: TraceParams,
    /// Record slots per ray (`steps + 1`), 0 without records.
    pub record_slots: usize,
}

impl PreparedTrace {
    /// Validate `req` against `set` (the stored systems, never the handle's
    /// claims): the program against every system a ray uses, each ray's
    /// wavelength against its system, the Newton settings, and radiometry
    /// support of every surface the program touches (thin-film and
    /// tabulated coatings are not available on the GPU: `Unsupported`).
    pub fn new(req: &TraceRequest, set: &PackedSystemSet) -> MkResult<Self> {
        req.validate()?;
        let program = req.program();
        let rays = &req.rays;
        let mut used = vec![false; set.len()];
        if rays.system.is_empty() {
            if set.is_empty() {
                return Err(MkError::Backend("trace: empty system set".into()));
            }
            used[0] = true;
        } else {
            for &s in &rays.system {
                *used.get_mut(s as usize).ok_or_else(|| {
                    MkError::Backend(format!("trace: ray system index {s} out of range"))
                })? = true;
            }
        }
        for (k, h) in set.headers.iter().enumerate().filter(|(k, _)| used[*k]) {
            let surf = &set.surfaces[h.surface_offset as usize..][..h.num_surfaces as usize];
            PathStep::validate_program(&program, req.start_surface, surf)
                .map_err(|e| MkError::Backend(format!("system {k}: {e}")))?;
            if req.radiometry != Radiometry::Off
                && let Some(st) = program
                    .iter()
                    .find(|st| surf[st.surface as usize].coating == MKL_COATING_TABLE)
            {
                return Err(MkError::Unsupported(format!(
                    "system {k} surface {}: thin-film / tabulated coatings are not \
                     available in GPU radiometry (trace with Radiometry::Off or on the CPU)",
                    st.surface
                )));
            }
        }
        for (i, &w) in rays.wavelength.iter().enumerate() {
            let sys = rays.system.get(i).copied().unwrap_or(0) as usize;
            if w >= set.headers[sys].num_wavelengths {
                return Err(MkError::Backend(format!(
                    "trace: ray {i} wavelength {w} out of range"
                )));
            }
        }
        let nw = req.newton;
        if nw.max_iterations > 64
            || !(nw.tolerance_abs.is_finite() && nw.tolerance_abs >= 0.0)
            || !(nw.tolerance_rel.is_finite() && (0.0..1.0).contains(&nw.tolerance_rel))
        {
            return Err(MkError::Backend(format!("trace: newton settings {nw:?}")));
        }
        let mut flags = 0;
        if req.apply_apertures {
            flags |= MKL_TRACE_CLIP;
        }
        if req.radiometry != Radiometry::Off {
            flags |= MKL_TRACE_RADIOMETRY;
        }
        if nw.max_iterations < mklight_trace::IntersectSettings::default().max_iterations {
            // The reference would iterate on: a search stopped by the cap
            // is not evidence of a miss.
            flags |= MKL_TRACE_NEWTON_CAP_NOT_CONVERGED;
        }
        let record_slots = if req.record == RecordMode::AllSurfaces {
            flags |= MKL_TRACE_RECORD;
            program.len() + 1
        } else {
            0
        };
        Ok(PreparedTrace {
            steps: program.iter().map(|s| s.encode()).collect(),
            params: TraceParams {
                num_rays: 0,
                stride: 0,
                num_steps: program.len() as u32,
                start_surface: req.start_surface,
                flags,
                max_iterations: nw.max_iterations,
                tolerance_abs: nw.tolerance_abs,
                tolerance_rel: nw.tolerance_rel,
            },
            record_slots,
        })
    }

    /// The kernel's input planes for `rays`: `MKL_TRACE_IN_FLOATS` float
    /// planes and `MKL_TRACE_IN_UINTS` uint planes of `rays.len()` values
    /// (empty weights become 1, empty system indices and residuals 0).
    pub fn input_planes(rays: &RayBatch) -> (Vec<f32>, Vec<u32>) {
        let n = rays.len();
        let mut f = vec![0.0; n * MKL_TRACE_IN_FLOATS as usize];
        let mut u = vec![0; n * MKL_TRACE_IN_UINTS as usize];
        Self::write_input_planes(rays, &mut f, &mut u);
        (f, u)
    }

    /// [`PreparedTrace::input_planes`] written into caller-provided storage
    /// (e.g. a mapped device buffer), every element overwritten.
    ///
    /// # Panics
    ///
    /// When `f` or `u` is not exactly the planes' size.
    pub fn write_input_planes(rays: &RayBatch, f: &mut [f32], u: &mut [u32]) {
        let n = rays.len();
        assert_eq!(f.len(), n * MKL_TRACE_IN_FLOATS as usize);
        assert_eq!(u.len(), n * MKL_TRACE_IN_UINTS as usize);
        let fplanes: [(u32, &Vec<f32>, f32); 15] = [
            (MKL_TRACE_IN_X, &rays.x, 0.0),
            (MKL_TRACE_IN_Y, &rays.y, 0.0),
            (MKL_TRACE_IN_Z, &rays.z, 0.0),
            (MKL_TRACE_IN_L, &rays.l, 0.0),
            (MKL_TRACE_IN_M, &rays.m, 0.0),
            (MKL_TRACE_IN_N, &rays.n, 0.0),
            (MKL_TRACE_IN_OPL_HI, &rays.opl_hi, 0.0),
            (MKL_TRACE_IN_OPL_LO, &rays.opl_lo, 0.0),
            (MKL_TRACE_IN_WEIGHT, &rays.weight, 1.0),
            (MKL_TRACE_IN_X_LO, &rays.x_lo, 0.0),
            (MKL_TRACE_IN_Y_LO, &rays.y_lo, 0.0),
            (MKL_TRACE_IN_Z_LO, &rays.z_lo, 0.0),
            (MKL_TRACE_IN_L_LO, &rays.l_lo, 0.0),
            (MKL_TRACE_IN_M_LO, &rays.m_lo, 0.0),
            (MKL_TRACE_IN_N_LO, &rays.n_lo, 0.0),
        ];
        for (k, src, empty) in fplanes {
            let dst = &mut f[k as usize * n..(k as usize + 1) * n];
            if src.is_empty() {
                dst.fill(empty);
            } else {
                dst.copy_from_slice(src);
            }
        }
        for (k, src) in [
            (MKL_TRACE_INU_WAVELENGTH, &rays.wavelength),
            (MKL_TRACE_INU_SYSTEM, &rays.system),
        ] {
            let dst = &mut u[k as usize * n..(k as usize + 1) * n];
            if src.is_empty() {
                dst.fill(0);
            } else {
                dst.copy_from_slice(src);
            }
        }
    }
}

/// Decoded per-ray status word (meanings as `mklight_trace::TraceStatus`,
/// see `docs/GPU.md` §4.2).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RayStatus {
    Ok,
    Miss { surface: u32 },
    TotalInternalReflection { surface: u32 },
    Vignetted { surface: u32 },
    NonFinite { surface: u32 },
    NotConverged { surface: u32 },
    OutsideDomain { surface: u32 },
    Degenerate { surface: u32 },
    EvanescentOrder { surface: u32 },
    Unknown { code: u32, surface: u32 },
}

impl RayStatus {
    pub fn decode(word: u32) -> Self {
        let surface = word >> MKL_RAY_SURFACE_SHIFT;
        match word & MKL_RAY_CODE_MASK {
            MKL_RAY_OK => RayStatus::Ok,
            MKL_RAY_MISS => RayStatus::Miss { surface },
            MKL_RAY_TIR => RayStatus::TotalInternalReflection { surface },
            MKL_RAY_VIGNETTED => RayStatus::Vignetted { surface },
            MKL_RAY_NONFINITE => RayStatus::NonFinite { surface },
            MKL_RAY_NOT_CONVERGED => RayStatus::NotConverged { surface },
            MKL_RAY_OUTSIDE_DOMAIN => RayStatus::OutsideDomain { surface },
            MKL_RAY_DEGENERATE => RayStatus::Degenerate { surface },
            MKL_RAY_EVANESCENT => RayStatus::EvanescentOrder { surface },
            code => RayStatus::Unknown { code, surface },
        }
    }
    pub fn encode(self) -> u32 {
        let (code, s) = match self {
            RayStatus::Ok => (MKL_RAY_OK, 0),
            RayStatus::Miss { surface } => (MKL_RAY_MISS, surface),
            RayStatus::TotalInternalReflection { surface } => (MKL_RAY_TIR, surface),
            RayStatus::Vignetted { surface } => (MKL_RAY_VIGNETTED, surface),
            RayStatus::NonFinite { surface } => (MKL_RAY_NONFINITE, surface),
            RayStatus::NotConverged { surface } => (MKL_RAY_NOT_CONVERGED, surface),
            RayStatus::OutsideDomain { surface } => (MKL_RAY_OUTSIDE_DOMAIN, surface),
            RayStatus::Degenerate { surface } => (MKL_RAY_DEGENERATE, surface),
            RayStatus::EvanescentOrder { surface } => (MKL_RAY_EVANESCENT, surface),
            RayStatus::Unknown { code, surface } => (code & MKL_RAY_CODE_MASK, surface),
        };
        code | (s << MKL_RAY_SURFACE_SHIFT)
    }
    pub fn is_ok(self) -> bool {
        self == RayStatus::Ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_roundtrip() {
        for s in [
            RayStatus::Ok,
            RayStatus::Miss { surface: 3 },
            RayStatus::TotalInternalReflection { surface: 17 },
            RayStatus::Vignetted { surface: 1 },
            RayStatus::NonFinite { surface: 0 },
            RayStatus::NotConverged { surface: 999 },
            RayStatus::OutsideDomain { surface: 4 },
            RayStatus::Degenerate { surface: 2 },
        ] {
            assert_eq!(RayStatus::decode(s.encode()), s);
        }
        assert_eq!(
            RayStatus::decode(MKL_RAY_VIGNETTED | (5 << 8)),
            RayStatus::Vignetted { surface: 5 }
        );
    }

    #[test]
    fn batch_validation() {
        let mut b = RayBatch {
            x: vec![0.0; 2],
            y: vec![0.0; 2],
            z: vec![0.0; 2],
            l: vec![0.0; 2],
            m: vec![0.0; 2],
            n: vec![1.0; 2],
            opl_hi: vec![0.0; 2],
            opl_lo: vec![0.0; 2],
            wavelength: vec![0; 2],
            system: vec![],
            weight: vec![],
            ..Default::default()
        };
        assert!(b.validate().is_ok());
        b.weight = vec![1.0, 0.5];
        assert!(b.validate().is_ok());
        b.system = vec![0];
        assert!(b.validate().is_err());
        b.system = vec![0, 1];
        let req = TraceRequest::new(SystemHandle::new(1, 1, 1), b.clone(), 0, 3);
        assert!(req.validate().is_err(), "system 1 of a 1-system set");
        let mut req = TraceRequest::new(SystemHandle::new(1, 1, 2), b.clone(), 0, 3);
        assert!(req.validate().is_ok());
        assert_eq!(req.radiometry, Radiometry::Unpolarized);
        req.end_surface = 0;
        assert!(req.validate().is_err());
        b.weight = vec![1.0];
        assert!(b.validate().is_err());
    }
}
