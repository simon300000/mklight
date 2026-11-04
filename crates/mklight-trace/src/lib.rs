//! MKLight CPU double-precision **reference** sequential ray tracer and
//! paraxial optics.
//!
//! Every GPU backend and every analysis is validated against this crate, so
//! the per-ray core is written as pure functions on plain data with
//! documented formulas that GPU kernels mirror one-to-one. Conventions are
//! those of `docs/CONVENTIONS.md`: millimetres, micrometre wavelengths,
//! positive indices, signed thicknesses, local surface frames.
//!
//! # Layers
//!
//! | module | contents |
//! |---|---|
//! | [`surface`] | intersections (plane, conic, even asphere, ideal lens, coordinate break) and analytic normals in the surface's local frame |
//! | [`interact`] | stable vector Snell refraction with TIR detection, reflection, ideal thin lens |
//! | [`trace`] | [`trace_ray`]: one ray through a [`ResolvedSystem`] using only relative surface-to-surface transforms; status, compensated OPL, internal transmittance, optional per-surface records. [`trace_rays`]: the same for packets of [`PACKET`] rays (SIMD lane kernel for planes and conics), bitwise identical |
//! | [`paraxial`] | y–nu trace on the unfolded equivalent (signed reduced indices for mirrors), [`FirstOrder`] properties per wavelength |
//! | [`tracer`] | [`SequentialTracer`]: fields → object sources, pupils → starting rays, ray aiming (paraxial or real), per-(field, wavelength) aim cache; `generate_many` / `trace_many` aim and trace many rays together, bitwise identical to one by one |
//! | [`opd`] | OPD against the exit-pupil reference sphere (CONVENTIONS §4) |
//! | [`semidia`] | automatic clear semi-diameters |
//! | [`solves`] | solves and pickups: [`apply_solves`] returns a new system with every solve-controlled parameter set (closed-form y–nu updates, fixed-point and bracketed Brent iteration, per configuration) |
//! | [`batch`] | rayon batches over fields × wavelengths × pupil samples (64-ray packet chunks per task), deterministic ordering, cancellable job variants |
//! | [`pupil`] | deterministic pupil sampling patterns |
//! | [`diff`] | analytic differential tracing: the tangent of a traced ray with respect to two launch parameters, carried along its records (aiming Jacobians, chief-ray pencils) |
//! | [`numeric`] | error-free transforms (TwoSum/TwoProd), [`Compensated`] sums, accurate discriminant, 2-D Newton with analytic or finite-difference Jacobians and Broyden updates (also as a resumable state machine for batched aiming) |
//! | [`polarization`] | polarisation ray tracing: 3×3 PRT matrices along recorded rays, Jones pupils in double-pole bases, diattenuation / retardance, energy budgets, vector pupils for wave optics |
//! | [`radiometry`] | per-ray throughput (unpolarised or polarised), pupil-averaged system transmission vs field and wavelength, polarisation pupil maps |
//!
//! # Usage
//!
//! ```
//! use std::sync::Arc;
//! use mklight_core::{materials::MaterialLibrary, samples, ResolvedSystem, Snapshot};
//! use mklight_trace::{SequentialTracer, TraceOptions};
//!
//! let snap = Snapshot::new(samples::cooke_triplet()).unwrap();
//! let sys = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
//! let tracer = SequentialTracer::new(Arc::new(sys)).unwrap();
//! let efl = tracer.paraxial().primary().efl;
//! let field = tracer.field(2);
//! let wl = tracer.system().primary_wavelength;
//! let ray = tracer.trace(field, wl, 0.0, 1.0, &TraceOptions::default());
//! assert!(ray.status.is_ok());
//! let opd = tracer.opd(field, wl, 0.0, 1.0, &TraceOptions::default()).unwrap();
//! # let _ = (efl, opd);
//! ```
//!
//! # Ray model
//!
//! A [`Ray`] is a point and a unit direction in one frame. Traces start in the
//! *starting frame*: the object surface's local frame for a finite object, or
//! the entrance (global) frame for an object at infinity. Inside the trace a
//! ray only ever moves between neighbouring surface frames
//! (`ResolvedSurface::to_next`, or an exact `−t` translation on the axial fast
//! path). Global coordinates are only used to *set up* starting rays and for
//! display ([`trace::to_global`]).
//!
//! Propagation distances are signed (virtual segments are legal in a
//! sequential trace). The optical path is `Σ n_before · t`, accumulated with
//! exact products and TwoSum into a [`Compensated`] `hi + lo` pair, so OPD is
//! formed as a difference before rounding.
//!
//! # Status
//!
//! [`TraceStatus`]: `Ok`, `Miss { surface }` (no intersection with the vertex
//! sheet), `Clipped { surface }` (outside a fixed semi-diameter or explicit
//! aperture; disabled by [`TraceOptions::clip`]), `TotalInternalReflection
//! { surface }`, `Failed { surface, reason }` (asphere non-convergence, outside
//! the sag domain, non-finite values including a non-finite input ray, a
//! degenerate ideal-lens ray, a ray that real aiming could not generate, or
//! any ray of a field that could not be resolved). An asphere that the ray
//! does not cross is a `Miss`, like a conic (see [`surface`] for the exact
//! classification GPU backends must reproduce).
//!
//! # Ideal lenses
//!
//! A `Paraxial` surface is free of ray aberrations at every conjugate and
//! adds the optical path that makes it perfect in phase for the conjugate
//! of the axial object point ([`interact::ideal_lens`],
//! [`paraxial::ideal_lens_object_vergence`]), so wave analyses of
//! finite-conjugate systems with ideal lenses are exact. With a `Dummy`
//! interaction it is a pass-through, like any other dummy surface.

pub mod batch;
pub mod diff;
pub mod interact;
pub mod numeric;
pub mod opd;
pub mod paraxial;
pub mod polarization;
pub mod pupil;
pub mod radiometry;
pub mod semidia;
pub mod solves;
pub mod surface;
pub mod trace;
pub mod tracer;

pub use batch::{BatchRay, Grid, opd_grid, opd_grid_job, trace_grid, trace_grid_job};
pub use diff::{RayTangent, TangentError, TangentPlan, differentiate, trace_ray_tangent};
pub use numeric::Compensated;
pub use opd::OpdReference;
pub use paraxial::{
    FirstOrder, Paraxial, ParaxialAxis, ParaxialModel, ParaxialRayTable, ParaxialXY,
};
pub use polarization::{
    JonesMatrix, JonesVector, PolarizationProperties, PolarizationState, PolarizedRay, PrtMatrix,
    Stokes, VectorPupilSample, trace_ray_polarized, vector_pupil,
};
pub use pupil::PupilSampling;
pub use radiometry::{
    PolarizationMapSettings, PolarizationPupilMap, PupilSamples, RadiometryModel, RayThroughput,
    TransmissionResult, TransmissionSettings, polarization_pupil_map, system_transmission,
};
pub use semidia::{
    CANCEL_CHUNK, SemiDiameterSettings, SemiDiameterSource, SemiDiameters, fix_semi_diameters,
};
pub use solves::{
    SolveMethod, SolveOutcome, SolveReport, SolveSettings, apply_solves, apply_solves_all,
    apply_solves_configuration, apply_solves_with, solved_params,
};
pub use surface::{IntersectError, IntersectSettings};
pub use trace::{
    FailReason, PACKET, Ray, SurfaceRecord, TraceOptions, TraceResult, TraceStatus, trace_ray,
    trace_ray_recorded, trace_rays, trace_rays_recorded, transfer_vector,
};
pub use tracer::{
    AimBudget, AimConditioning, AimData, AimReport, AimStats, AimStop, FieldPoint, GeneratedRay,
    ObjectSource, PreparedField, PupilGeometry, RayRequest, SequentialTracer, TracerSettings,
};

pub use mklight_core::ResolvedSystem;
