//! Low-level access to the CPU f64 reference tracer (`mklight-trace`):
//! single rays with per-surface records, vectorised grids, and first-order
//! (paraxial) data. These bypass the analysis registry on purpose; they use
//! the session's current snapshot and glass library, resolved exactly as
//! analyses resolve them.
//!
//! Coordinates follow `docs/CONVENTIONS.md`: millimetres, unit direction
//! cosines, normalised field `(Hx, Hy)` and pupil `(Px, Py)`.

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use serde_json::Value;

use mklight_core::JobContext;
use mklight_engine::BufferData;
use mklight_trace::pupil::grid;
use mklight_trace::trace::to_global;
use mklight_trace::{
    BatchRay, FieldPoint, FirstOrder, Grid, Paraxial, PreparedField, PupilSampling,
    SequentialTracer, SurfaceRecord, TraceOptions, TraceStatus, trace_grid_job,
};

use crate::array::export_owned;
use crate::convert::{from_py, ser_to_py};
use crate::errors::{ErrorKind, core_err, engine_err, new_err};
use crate::session::{PySession, u64_arg};
use crate::wait::run_interruptible;

/// Status codes of `trace_grid`'s `status` array, by name.
pub const TRACE_STATUS: [&str; 5] = [
    "ok",
    "miss",
    "clipped",
    "total_internal_reflection",
    "failed",
];

/// Numeric code of a trace status (index into [`TRACE_STATUS`]).
pub fn status_code(s: &TraceStatus) -> u8 {
    match s {
        TraceStatus::Ok => 0,
        TraceStatus::Miss { .. } => 1,
        TraceStatus::Clipped { .. } => 2,
        TraceStatus::TotalInternalReflection { .. } => 3,
        TraceStatus::Failed { .. } => 4,
    }
}

/// A field argument: a system field index or a normalised `(Hx, Hy)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FieldSpec {
    System(usize),
    Normalized(f64, f64),
}

fn field_spec(v: &Value, n_fields: usize) -> Result<FieldSpec, String> {
    match v {
        Value::Number(n) => {
            let i = n
                .as_u64()
                .ok_or_else(|| format!("field index must be a non-negative integer, got {n}"))?
                as usize;
            if i >= n_fields {
                return Err(format!("field index {i} out of range (0..{n_fields})"));
            }
            Ok(FieldSpec::System(i))
        }
        Value::Array(a) if a.len() == 2 => match (a[0].as_f64(), a[1].as_f64()) {
            (Some(hx), Some(hy)) if hx.is_finite() && hy.is_finite() => {
                Ok(FieldSpec::Normalized(hx, hy))
            }
            _ => Err(format!(
                "normalised field must be two finite numbers, got {v}"
            )),
        },
        other => Err(format!(
            "a field is an index or a normalised (hx, hy) pair, got {other}"
        )),
    }
}

/// `None` → every system field; an index → that one; a list → each entry
/// (an index or an `(hx, hy)` pair). A top-level list is always a list of
/// fields, so `[0, 1]` means system fields 0 and 1; a single normalised
/// point arrives here already wrapped (see [`fields_value`]).
pub fn parse_fields(v: &Value, n_fields: usize) -> Result<Vec<FieldSpec>, String> {
    match v {
        Value::Null => Ok((0..n_fields).map(FieldSpec::System).collect()),
        Value::Number(_) => Ok(vec![field_spec(v, n_fields)?]),
        Value::Array(a) => a
            .iter()
            .map(|x| field_spec(x, n_fields))
            .collect::<Result<_, _>>()
            .map_err(|e| {
                if a.len() == 2 && a.iter().all(Value::is_number) {
                    format!(
                        "{e} (a list of fields; write one normalised field point as a \
                         tuple (hx, hy))"
                    )
                } else {
                    e
                }
            }),
        other => Err(format!("invalid fields {other}")),
    }
}

/// The `fields` argument of `trace_grid` as JSON. A Python `tuple` of two
/// numbers is one normalised `(hx, hy)` point, read exactly as `trace`
/// reads its `field` argument (whether the numbers are ints or floats);
/// it is wrapped into a one-element list. Everything else converts as is.
pub fn fields_value(obj: Option<&Bound<'_, PyAny>>) -> PyResult<Value> {
    let Some(obj) = obj else {
        return Ok(Value::Null);
    };
    let v = from_py(obj)?;
    match &v {
        Value::Array(a)
            if obj.is_instance_of::<PyTuple>()
                && a.len() == 2
                && a.iter().all(Value::is_number) =>
        {
            Ok(Value::Array(vec![v]))
        }
        _ => Ok(v),
    }
}

/// An optional wavelength index argument: `None` → `None`; otherwise an
/// integer in range (anything else raises `InvalidArgumentError`).
fn wavelength_arg(obj: Option<&Bound<'_, PyAny>>, n: usize) -> PyResult<Option<usize>> {
    let Some(obj) = obj.filter(|o| !o.is_none()) else {
        return Ok(None);
    };
    let w = u64_arg(obj, "wavelength index", ErrorKind::InvalidArgument)?;
    if w >= n as u64 {
        return Err(invalid(format!(
            "wavelength index {w} out of range (0..{n})"
        )));
    }
    Ok(Some(w as usize))
}

/// `None` → every system wavelength; an index or a list of indices.
pub fn parse_wavelengths(v: &Value, n: usize) -> Result<Vec<usize>, String> {
    let one = |x: &Value| -> Result<usize, String> {
        let i = x
            .as_u64()
            .ok_or_else(|| format!("wavelength index must be a non-negative integer, got {x}"))?
            as usize;
        if i >= n {
            return Err(format!("wavelength index {i} out of range (0..{n})"));
        }
        Ok(i)
    };
    match v {
        Value::Null => Ok((0..n).collect()),
        Value::Array(a) => a.iter().map(one).collect(),
        x => Ok(vec![one(x)?]),
    }
}

/// An int `n` → `n × n` grid inside the unit disk; a pattern dict
/// (`{"type": "hexapolar", "rings": 6}`, see `PupilSampling`); or a
/// sequence of `(px, py)` pairs.
pub fn parse_pupil(v: &Value) -> Result<Vec<[f64; 2]>, String> {
    match v {
        Value::Number(n) => {
            let n = n
                .as_u64()
                .ok_or_else(|| format!("pupil grid size must be a positive integer, got {n}"))?;
            Ok(grid(n as usize))
        }
        Value::Object(_) => serde_json::from_value::<PupilSampling>(v.clone())
            .map(|p| p.points())
            .map_err(|e| format!("invalid pupil pattern {v}: {e}")),
        Value::Array(a) => a
            .iter()
            .map(|p| match p.as_array().map(Vec::as_slice) {
                Some([x, y]) => match (x.as_f64(), y.as_f64()) {
                    (Some(x), Some(y)) if x.is_finite() && y.is_finite() => Ok([x, y]),
                    _ => Err(format!("pupil point must be two finite numbers, got {p}")),
                },
                _ => Err(format!("pupil point must be a (px, py) pair, got {p}")),
            })
            .collect(),
        other => Err(format!("invalid pupil {other}")),
    }
}

/// Prepare fields for tracing (system fields come from the tracer's cache).
pub fn prepare_fields(
    tracer: &SequentialTracer,
    specs: &[FieldSpec],
) -> mklight_core::MkResult<Vec<PreparedField>> {
    specs
        .iter()
        .map(|s| match *s {
            FieldSpec::System(i) => Ok(tracer.field(i).clone()),
            FieldSpec::Normalized(hx, hy) => tracer.prepare_field(FieldPoint::normalized(hx, hy)),
        })
        .collect()
}

/// `trace_grid` output in flat row-major arrays, `(F, W, P)` leading axes.
#[derive(Debug, Default)]
pub struct GridArrays {
    pub shape: [usize; 3],
    pub point: Vec<f64>,
    pub dir: Vec<f64>,
    pub opl: Vec<f64>,
    pub transmittance: Vec<f64>,
    pub status: Vec<u8>,
    pub surface: Vec<i32>,
}

impl GridArrays {
    /// Lay out a batch (`mklight_trace::batch` ordering is already
    /// `(f · W + w) · P + p`, i.e. row-major `(F, W, P)`).
    pub fn from_batch(shape: [usize; 3], rays: &[BatchRay]) -> Self {
        let n = rays.len();
        debug_assert_eq!(n, shape.iter().product::<usize>());
        let mut g = GridArrays {
            shape,
            point: Vec::with_capacity(3 * n),
            dir: Vec::with_capacity(3 * n),
            opl: Vec::with_capacity(n),
            transmittance: Vec::with_capacity(n),
            status: Vec::with_capacity(n),
            surface: Vec::with_capacity(n),
        };
        for r in rays {
            g.point.extend([r.point.x, r.point.y, r.point.z]);
            g.dir.extend([r.dir.x, r.dir.y, r.dir.z]);
            g.opl.push(r.opl.hi + r.opl.lo);
            g.transmittance.push(r.transmittance);
            g.status.push(status_code(&r.status));
            g.surface.push(r.status.surface().map_or(-1, |s| s as i32));
        }
        g
    }
}

/// Trace a grid with cancellation and progress.
pub fn trace_grid_arrays(
    tracer: &SequentialTracer,
    fields: &[PreparedField],
    wavelengths: &[usize],
    pupil: &[[f64; 2]],
    opts: &TraceOptions,
    ctx: &JobContext,
) -> mklight_core::MkResult<GridArrays> {
    let g = Grid {
        fields,
        wavelengths,
        pupil,
    };
    let rays = trace_grid_job(tracer, &g, opts, ctx)?;
    Ok(GridArrays::from_batch(
        [fields.len(), wavelengths.len(), pupil.len()],
        &rays,
    ))
}

fn invalid(msg: String) -> PyErr {
    new_err(ErrorKind::InvalidArgument, msg)
}

fn tuple3<'py>(py: Python<'py>, v: mklight_core::Vec3) -> PyResult<Bound<'py, PyTuple>> {
    PyTuple::new(py, [v.x, v.y, v.z])
}

fn records_dict<'py>(
    py: Python<'py>,
    tracer: &SequentialTracer,
    recs: &[SurfaceRecord],
) -> PyResult<Bound<'py, PyDict>> {
    let n = recs.len();
    let sys = tracer.system();
    let mut point = Vec::with_capacity(3 * n);
    let mut dir = Vec::with_capacity(3 * n);
    let mut normal = Vec::with_capacity(3 * n);
    let mut gpoint = Vec::with_capacity(3 * n);
    let mut gdir = Vec::with_capacity(3 * n);
    for (i, r) in recs.iter().enumerate() {
        point.extend([r.point.x, r.point.y, r.point.z]);
        dir.extend([r.dir.x, r.dir.y, r.dir.z]);
        normal.extend([r.normal.x, r.normal.y, r.normal.z]);
        let g = to_global(sys, i, &mklight_trace::Ray::new(r.point, r.dir));
        gpoint.extend([g.pos.x, g.pos.y, g.pos.z]);
        gdir.extend([g.dir.x, g.dir.y, g.dir.z]);
    }
    let d = PyDict::new(py);
    let v3 = |name: &str, v: Vec<f64>| export_owned(py, name, vec![n, 3], BufferData::F64(v));
    let v1 = |name: &str, v: Vec<f64>| export_owned(py, name, vec![n], BufferData::F64(v));
    d.set_item("point", v3("point", point)?)?;
    d.set_item("dir", v3("dir", dir)?)?;
    d.set_item("normal", v3("normal", normal)?)?;
    d.set_item("global_point", v3("global_point", gpoint)?)?;
    d.set_item("global_dir", v3("global_dir", gdir)?)?;
    d.set_item(
        "cos_incidence",
        v1(
            "cos_incidence",
            recs.iter().map(|r| r.cos_incidence).collect(),
        )?,
    )?;
    d.set_item("opl", v1("opl", recs.iter().map(|r| r.opl).collect())?)?;
    d.set_item(
        "transmittance",
        v1(
            "transmittance",
            recs.iter().map(|r| r.transmittance).collect(),
        )?,
    )?;
    Ok(d)
}

/// Trace one ray through the session's current system with the CPU f64
/// reference tracer.
///
/// * `field`: system field index, or a normalised `(hx, hy)` pair.
/// * `wavelength`: system wavelength index (`None`: the primary).
/// * `px`, `py`: normalised pupil coordinates.
/// * `records`: also return per-surface data (`records[i]` is surface `i`,
///   index 0 the starting ray) in each surface's local frame plus global
///   coordinates.
/// * `clip`: apply fixed semi-diameters and apertures.
///
/// Returns a dict: `status` (`ok`, `miss`, `clipped`,
/// `total_internal_reflection`, `failed`), `reason` (failures), `surface`
/// (where the ray stopped; the image on success), `point` and `dir` (in that
/// surface's local frame), `opl` (mm), `transmittance` (bulk absorption),
/// `wavelength_um`, `records`.
#[pyfunction]
#[pyo3(signature = (session, field=None, wavelength=None, px=0.0, py=0.0, records=true, clip=true))]
#[allow(clippy::too_many_arguments)]
pub fn trace<'py>(
    pyo: Python<'py>,
    session: &PySession,
    field: Option<Bound<'py, PyAny>>,
    wavelength: Option<Bound<'py, PyAny>>,
    px: f64,
    py: f64,
    records: bool,
    clip: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let tracer = session.tracer(pyo)?;
    let fv = field.as_ref().map_or(Ok(Value::from(0)), from_py)?;
    let spec = field_spec(&fv, tracer.fields().len()).map_err(invalid)?;
    let wl = wavelength_arg(wavelength.as_ref(), tracer.num_wavelengths())?
        .unwrap_or(tracer.system().primary_wavelength);
    let opts = TraceOptions {
        clip,
        ..tracer.options()
    };
    let t = tracer.clone();
    let (result, recs) = pyo
        .detach(move || -> mklight_core::MkResult<_> {
            let f = prepare_fields(&t, &[spec])?.remove(0);
            let mut recs = Vec::new();
            let r = t.trace_with(&f, wl, px, py, &opts, records.then_some(&mut recs));
            Ok((r, recs))
        })
        .map_err(core_err)?;
    let d = PyDict::new(pyo);
    let st = serde_json::to_value(result.status).unwrap_or(Value::Null);
    d.set_item("status", st.get("status").and_then(Value::as_str))?;
    d.set_item("reason", st.get("reason").and_then(Value::as_str))?;
    d.set_item("ok", result.status.is_ok())?;
    d.set_item("surface", result.surface)?;
    d.set_item("point", tuple3(pyo, result.ray.pos)?)?;
    d.set_item("dir", tuple3(pyo, result.ray.dir)?)?;
    d.set_item("opl", result.opl.hi + result.opl.lo)?;
    d.set_item("transmittance", result.transmittance)?;
    d.set_item("wavelength_index", wl)?;
    d.set_item("wavelength_um", tracer.system().wavelengths_um[wl])?;
    if records {
        d.set_item("records", records_dict(pyo, &tracer, &recs)?)?;
    } else {
        d.set_item("records", pyo.None())?;
    }
    Ok(d)
}

/// Trace fields × wavelengths × pupil points in parallel (GIL released,
/// Ctrl-C cancels) with the CPU f64 reference tracer.
///
/// * `fields`: `None` (all system fields), an index, `(hx, hy)`, or a list.
/// * `wavelengths`: `None` (all), an index, or a list of indices.
/// * `pupil`: int `n` (an `n × n` grid clipped to the unit disk), a pattern
///   dict such as `{"type": "hexapolar", "rings": 6}`, or `(P, 2)` points.
///
/// Returns a dict of arrays with leading axes `(F, W, P)`: `point` and
/// `dir` `(F, W, P, 3)` float64 (image-surface local frame on success),
/// `opl` and `transmittance` `(F, W, P)` float64, `status` uint8 (index
/// into `mklight.TRACE_STATUS`), `surface` int32 (where a failed ray
/// stopped, -1 on success), plus `pupil` `(P, 2)`, `field` `(F, 2)`
/// normalised field points, `wavelength_index` and `wavelength_um` `(W,)`.
#[pyfunction]
#[pyo3(signature = (session, fields=None, wavelengths=None, pupil=None, clip=true, progress=None))]
pub fn trace_grid<'py>(
    py: Python<'py>,
    session: &PySession,
    fields: Option<Bound<'py, PyAny>>,
    wavelengths: Option<Bound<'py, PyAny>>,
    pupil: Option<Bound<'py, PyAny>>,
    clip: bool,
    progress: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let tracer = session.tracer(py)?;
    let fv = fields_value(fields.as_ref())?;
    let specs = parse_fields(&fv, tracer.fields().len()).map_err(invalid)?;
    let wv = wavelengths.as_ref().map_or(Ok(Value::Null), from_py)?;
    let wls = parse_wavelengths(&wv, tracer.num_wavelengths()).map_err(invalid)?;
    let pv = pupil.as_ref().map_or(Ok(Value::from(16)), from_py)?;
    let pts = parse_pupil(&pv).map_err(invalid)?;
    let opts = TraceOptions {
        clip,
        ..tracer.options()
    };
    let ctx = JobContext::new();
    let t: Arc<SequentialTracer> = tracer.clone();
    let (prepared, g) = run_interruptible(py, &ctx, progress.as_ref(), |ctx| {
        let prepared = prepare_fields(&t, &specs)?;
        let g = trace_grid_arrays(&t, &prepared, &wls, &pts, &opts, ctx)?;
        Ok((prepared, g))
    })?;
    let [f, w, p] = g.shape;
    let d = PyDict::new(py);
    d.set_item(
        "point",
        export_owned(py, "point", vec![f, w, p, 3], BufferData::F64(g.point))?,
    )?;
    d.set_item(
        "dir",
        export_owned(py, "dir", vec![f, w, p, 3], BufferData::F64(g.dir))?,
    )?;
    d.set_item(
        "opl",
        export_owned(py, "opl", vec![f, w, p], BufferData::F64(g.opl))?,
    )?;
    d.set_item(
        "transmittance",
        export_owned(
            py,
            "transmittance",
            vec![f, w, p],
            BufferData::F64(g.transmittance),
        )?,
    )?;
    d.set_item(
        "status",
        export_owned(py, "status", vec![f, w, p], BufferData::U8(g.status))?,
    )?;
    d.set_item(
        "surface",
        export_owned(py, "surface", vec![f, w, p], BufferData::I32(g.surface))?,
    )?;
    d.set_item(
        "pupil",
        export_owned(
            py,
            "pupil",
            vec![p, 2],
            BufferData::F64(pts.iter().flatten().copied().collect()),
        )?,
    )?;
    d.set_item(
        "field",
        export_owned(
            py,
            "field",
            vec![f, 2],
            BufferData::F64(
                prepared
                    .iter()
                    .flat_map(|pf| [pf.point.hx, pf.point.hy])
                    .collect(),
            ),
        )?,
    )?;
    d.set_item(
        "wavelength_index",
        export_owned(
            py,
            "wavelength_index",
            vec![w],
            BufferData::I64(wls.iter().map(|&i| i as i64).collect()),
        )?,
    )?;
    d.set_item(
        "wavelength_um",
        export_owned(
            py,
            "wavelength_um",
            vec![w],
            BufferData::F64(
                wls.iter()
                    .map(|&i| tracer.system().wavelengths_um[i])
                    .collect(),
            ),
        )?,
    )?;
    Ok(d)
}

fn first_order_dict<'py>(py: Python<'py>, fo: &FirstOrder) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    // Scalars are set directly (not through JSON) so that infinities, such
    // as a telecentric pupil position, stay `float('inf')`.
    macro_rules! put {
        ($($f:ident),*) => { $( d.set_item(stringify!($f), fo.$f)?; )* };
    }
    put!(
        wavelength_um,
        efl,
        bfl,
        ffl,
        image_distance,
        entrance_pupil_position,
        entrance_pupil_diameter,
        exit_pupil_position,
        exit_pupil_diameter,
        image_f_number,
        working_f_number,
        object_na,
        image_na,
        magnification,
        lagrange_invariant,
        total_track,
        stop_semi_diameter,
        paraxial_image_height,
        image_height_per_tan,
        image_height_per_object_height,
        marginal_object_slope
    );
    for (name, t) in [("marginal", &fo.marginal), ("chief", &fo.chief)] {
        let r = PyDict::new(py);
        r.set_item("y", PyList::new(py, &t.y)?)?;
        r.set_item("u", PyList::new(py, &t.u)?)?;
        r.set_item("nu", PyList::new(py, &t.nu)?)?;
        d.set_item(name, r)?;
    }
    Ok(d)
}

/// First-order (paraxial) properties of the session's current system at
/// one wavelength (`None`: the primary): `efl`, `bfl`, `ffl`,
/// `image_distance`, pupil positions and diameters, F-numbers, NAs,
/// magnification, Lagrange invariant, total track, paraxial image height,
/// and the `marginal` / `chief` ray tables (`y`, `u`, `nu` per surface).
/// `warnings` lists caveats (e.g. tilted systems).
#[pyfunction]
#[pyo3(signature = (session, wavelength=None))]
pub fn paraxial<'py>(
    py: Python<'py>,
    session: &PySession,
    wavelength: Option<Bound<'py, PyAny>>,
) -> PyResult<Bound<'py, PyDict>> {
    let par = py
        .detach(|| -> Result<Paraxial, mklight_engine::EngineError> {
            let resolved = session.lock().resolved()?;
            Ok(Paraxial::new(&resolved)?)
        })
        .map_err(engine_err)?;
    let wl = wavelength_arg(wavelength.as_ref(), par.wavelengths.len())?.unwrap_or(par.primary);
    let d = first_order_dict(py, par.at(wl))?;
    d.set_item("wavelength_index", wl)?;
    d.set_item("primary_wavelength", par.primary)?;
    d.set_item("warnings", ser_to_py(py, &par.warnings)?)?;
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mklight_core::materials::MaterialLibrary;
    use mklight_core::{ResolvedSystem, Snapshot, samples};
    use serde_json::json;

    fn tracer(sys: mklight_core::OpticalSystem) -> SequentialTracer {
        let snap = Snapshot::new(sys).unwrap();
        let r = ResolvedSystem::new(&snap, Arc::new(MaterialLibrary::with_builtin())).unwrap();
        SequentialTracer::new(Arc::new(r)).unwrap()
    }

    #[test]
    fn argument_parsing() {
        assert_eq!(
            parse_fields(&Value::Null, 3).unwrap(),
            vec![
                FieldSpec::System(0),
                FieldSpec::System(1),
                FieldSpec::System(2)
            ]
        );
        // A top-level list is a list of fields, whatever the numbers.
        assert_eq!(
            parse_fields(&json!([0, 1]), 3).unwrap(),
            vec![FieldSpec::System(0), FieldSpec::System(1)]
        );
        let e = parse_fields(&json!([0.0, 0.5]), 3).unwrap_err();
        assert!(e.contains("tuple (hx, hy)"), "{e}");
        assert_eq!(
            parse_fields(&json!([[0, 1]]), 3).unwrap(),
            vec![FieldSpec::Normalized(0.0, 1.0)]
        );
        assert_eq!(
            parse_fields(&json!([2, [0.0, 1.0]]), 3).unwrap(),
            vec![FieldSpec::System(2), FieldSpec::Normalized(0.0, 1.0)]
        );
        assert!(parse_fields(&json!(3), 3).is_err());
        assert!(parse_fields(&json!("x"), 3).is_err());
        assert_eq!(parse_wavelengths(&Value::Null, 2).unwrap(), vec![0, 1]);
        assert_eq!(parse_wavelengths(&json!(1), 2).unwrap(), vec![1]);
        assert!(parse_wavelengths(&json!([0, 2]), 2).is_err());
        assert_eq!(parse_pupil(&json!(3)).unwrap().len(), 5); // centre + 4 edge midpoints
        assert_eq!(
            parse_pupil(&json!({"type": "hexapolar", "rings": 2}))
                .unwrap()
                .len(),
            19
        );
        assert_eq!(
            parse_pupil(&json!([[0.0, 1.0], [0.5, -0.5]])).unwrap(),
            vec![[0.0, 1.0], [0.5, -0.5]]
        );
        assert!(parse_pupil(&json!([[0.0]])).is_err());
    }

    #[test]
    fn grid_layout_matches_single_ray_traces() {
        // Cross-check the flattened (F, W, P) arrays against individual
        // reference traces, including a non-system normalised field.
        let t = tracer(samples::cooke_triplet());
        let specs = [FieldSpec::System(2), FieldSpec::Normalized(0.3, 0.6)];
        let fields = prepare_fields(&t, &specs).unwrap();
        let wls = [0usize, 2];
        let pupil = [[0.0, 0.0], [0.0, 1.0], [0.7, -0.2], [0.0, 40.0]];
        let opts = t.options();
        let g = trace_grid_arrays(&t, &fields, &wls, &pupil, &opts, &JobContext::new()).unwrap();
        assert_eq!(g.shape, [2, 2, 4]);
        let mut seen_fail = false;
        for (fi, f) in fields.iter().enumerate() {
            for (wi, &wl) in wls.iter().enumerate() {
                for (pi, p) in pupil.iter().enumerate() {
                    let k = (fi * 2 + wi) * 4 + pi;
                    let r = t.trace(f, wl, p[0], p[1], &opts);
                    assert_eq!(g.status[k], status_code(&r.status));
                    assert_eq!(g.surface[k], r.status.surface().map_or(-1, |s| s as i32));
                    seen_fail |= !r.status.is_ok();
                    if r.status.is_ok() {
                        assert_eq!(g.point[3 * k], r.ray.pos.x);
                        assert_eq!(g.point[3 * k + 1], r.ray.pos.y);
                        assert_eq!(g.dir[3 * k + 2], r.ray.dir.z);
                        assert_eq!(g.opl[k], r.opl.hi + r.opl.lo);
                    }
                }
            }
        }
        // Py = 40 is far outside the pupil: the ray cannot pass.
        assert!(seen_fail);
    }

    #[test]
    fn grid_trace_is_cancellable() {
        let t = tracer(samples::double_gauss());
        let fields = prepare_fields(&t, &[FieldSpec::System(0)]).unwrap();
        let pupil = grid(64);
        let ctx = JobContext::new();
        ctx.cancel.cancel();
        let r = trace_grid_arrays(&t, &fields, &[0], &pupil, &t.options(), &ctx);
        assert_eq!(r.unwrap_err(), mklight_core::MkError::Cancelled);
    }

    #[test]
    fn status_codes_index_the_name_table() {
        let cases = [
            (TraceStatus::Ok, "ok"),
            (TraceStatus::Miss { surface: 1 }, "miss"),
            (TraceStatus::Clipped { surface: 1 }, "clipped"),
            (
                TraceStatus::TotalInternalReflection { surface: 1 },
                "total_internal_reflection",
            ),
        ];
        for (s, name) in cases {
            assert_eq!(TRACE_STATUS[status_code(&s) as usize], name);
            let v = serde_json::to_value(s).unwrap();
            assert_eq!(v["status"], name);
        }
    }
}
