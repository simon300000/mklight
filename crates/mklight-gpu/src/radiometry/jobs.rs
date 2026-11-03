//! Host boundary and final assembly for resident P1 -> P7 -> P3 jobs.
use super::*;
use crate::{ComputeBackend, Df32, PackedSystem, aiming};
use mklight_core::{JobContext, job::RunInfo};
use mklight_trace::{FieldPoint, PolarizationState, PreparedField, SequentialTracer};

/// Completed backend rows, borrowed in place unless reference completion
/// requires a mutable working copy. The view pins its storage independently
/// of the backend and plan lifetimes.
pub struct ResultSlice<T: bytemuck::Pod> {
    storage: ResultStorage<T>,
}
enum ResultStorage<T: bytemuck::Pod> {
    Resident(ReadView, usize),
    Reference(Vec<T>),
}
impl<T: bytemuck::Pod> ResultSlice<T> {
    pub(super) fn resident(view: ReadView, count: usize) -> Self {
        Self {
            storage: ResultStorage::Resident(view, count),
        }
    }
    pub(super) fn owned(rows: Vec<T>) -> Self {
        Self {
            storage: ResultStorage::Reference(rows),
        }
    }
    pub fn is_resident(&self) -> bool {
        matches!(self.storage, ResultStorage::Resident(..))
    }
}
impl<T: bytemuck::Pod> std::ops::Deref for ResultSlice<T> {
    type Target = [T];
    fn deref(&self) -> &[T] {
        match &self.storage {
            ResultStorage::Resident(view, count) => {
                let rows: &[T] = bytemuck::cast_slice(view.words());
                &rows[..*count]
            }
            ResultStorage::Reference(rows) => rows,
        }
    }
}
impl<T: bytemuck::Pod> AsRef<[T]> for ResultSlice<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

pub struct PupilResult {
    pub values: ResultSlice<PolarizationValue>,
    pub trace_status: ResultSlice<u32>,
    /// P3 summary [channel][sum,count,min,max], channels as append_summary.
    pub summary: [[f64; 4]; 10],
    pub run_info: RunInfo,
    pub stats: PlanStats,
}
fn import<T: bytemuck::Pod>(
    b: &dyn ComputeBackend,
    p: &mut ResidentPlan,
    rows: &[T],
) -> MkResult<PlanView> {
    let buffer = b.fill_buffer(record_desc::<T>(rows.len()), &mut |mut f| {
        bytemuck::cast_slice_mut::<u32, T>(f.as_mut_slice::<u32>()?).copy_from_slice(rows);
        Ok(())
    })?;
    let id = p.import(buffer.view());
    p.view(id)
}
/// Build a primary-path job. The caller owns system-handle lifetime and may
/// retain prepared material leases across fields, wavelengths and pupil grids.
#[allow(clippy::too_many_arguments)]
pub fn pupil_plan(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    prepared: &PreparedRadiometry,
    systems: crate::SystemHandle,
    field: &PreparedField,
    wl: usize,
    points: &[[f64; 2]],
    state: &PolarizationState,
    clip: bool,
    ctx: &JobContext,
) -> MkResult<ResidentPlan> {
    ctx.check()?;
    if wl >= t.num_wavelengths() || points.is_empty() || points.len() > u32::MAX as usize {
        return Err(mklight_core::MkError::InvalidModel(
            "polarization wavelength or empty pupil".into(),
        ));
    }
    let mut p = ResidentPlan::new(t.system().snapshot.clone());
    p.identity.library_hash = prepared.identity.clone();
    p.settings = serde_json::json!({"operation":"radiometry_pupil","version":crate::consts::MKL_RAD_VERSION,"field":field.point,"wavelength":wl,"state":state,"clip":clip,"point_count":points.len()});
    let spec = aiming::FieldSpec::new(
        t.system(),
        t.paraxial(),
        field.point,
        0,
        wl as u32,
        t.aiming(),
    )?;
    let fields = import(b, &mut p, &[spec])?;
    // The chief row comes from the prepared field (f64 source and chief).
    let start = std::time::Instant::now();
    let seed = aiming::ChiefSeed::host(t, field, wl);
    p.host.add(&crate::resident::HostPreparation {
        seed_rows: 1,
        reused_fields: 1,
        solved_fields: 0,
        seconds: start.elapsed().as_secs_f64(),
    });
    let seeds = import(b, &mut p, &[seed])?;
    let samples = b.fill_buffer(record_desc::<aiming::Sample>(points.len()), &mut |mut f| {
        let dest: &mut [aiming::Sample] = bytemuck::cast_slice_mut(f.as_mut_slice::<u32>()?);
        for (i, (point, out)) in points.iter().zip(dest).enumerate() {
            if i % 4096 == 0 {
                ctx.check()?;
            }
            if !point.iter().all(|v| v.is_finite()) {
                return Err(mklight_core::MkError::InvalidModel(
                    "nonfinite pupil point".into(),
                ));
            }
            *out = aiming::Sample::new(*point, 0, i as u64);
        }
        Ok(())
    })?;
    let samples = p.import(samples.view());
    let rays = aiming::rays(&mut p, points.len());
    let tolerance = match t.aiming() {
        mklight_core::RayAiming::Real { tolerance } => tolerance as f32,
        _ => 1e-10,
    };
    p.operations
        .push(ResidentOp::Aiming(aiming::Request::GenerateAim(
            aiming::GenerateAim {
                systems: SystemSet::Uploaded(systems),
                samples: p.view(samples)?,
                sampling: aiming::Sampling::Explicit,
                unresolved: None,
                field_wavelength: fields,
                field_key: Some(blake3::hash(bytemuck::bytes_of(&spec)).to_hex().to_string()),
                seeds: Some(seeds),
                tolerance,
                max_iterations: 40,
                output: rays.clone(),
                options: Default::default(),
            },
        )));
    let steps = t.system().image_index();
    let program = b.fill_buffer(BufferDesc::new(ElementType::U32, [steps]), &mut |mut f| {
        for (i, v) in f.as_mut_slice::<u32>()?.iter_mut().enumerate() {
            *v = (i + 1) as u32;
        }
        Ok(())
    })?;
    let program = p.import(program.view());
    let hits = aiming::hits(&mut p, points.len(), steps + 1);
    p.operations
        .push(ResidentOp::Aiming(aiming::Request::TraceResident(
            aiming::TraceResident {
                systems: SystemSet::Uploaded(systems),
                input: rays,
                steps: p.view(program)?,
                start_surface: 0,
                apply_apertures: clip,
                radiometry: 0,
                newton: Default::default(),
                output: hits.clone(),
            },
        )));
    let values = prepared.append(&mut p, &hits, state)?;
    let summary = append_summary(b, &mut p, values, hits.status, ctx)?;
    p.outputs = vec![values, hits.status, summary];
    Ok(p)
}
/// End-to-end pupil job. Only final values/status/statistics are read on the
/// host. Reference-required rows are retraced in f64, explicitly counted and
/// recorded; cancellation never enters the reference path.
#[allow(clippy::too_many_arguments)]
pub fn pupil(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    field: FieldPoint,
    wl: usize,
    points: &[[f64; 2]],
    state: &PolarizationState,
    clip: bool,
    ctx: &JobContext,
) -> MkResult<PupilResult> {
    pupil_mode(b, t, field, wl, points, state, clip, false, true, ctx)
}
#[allow(clippy::too_many_arguments)]
fn pupil_mode(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    field: FieldPoint,
    wl: usize,
    points: &[[f64; 2]],
    state: &PolarizationState,
    clip: bool,
    scalar: bool,
    retain_values: bool,
    ctx: &JobContext,
) -> MkResult<PupilResult> {
    ctx.check()?;
    let field = t.prepare_field(field)?;
    let prepared = PreparedRadiometry::new(b, t.system(), &ctx.subrange(0.0, 0.05))?;
    let packed = PackedSystem::pack_all(t.system())?;
    let cache_key = SystemCacheKey {
        snapshot: t.system().snapshot.hash().to_hex(),
        library: prepared.identity.clone(),
        configuration: "primary".into(),
        spectrum: "all".into(),
        layout: crate::consts::MKL_LAYOUT_VERSION,
        kernel_build: b.precision().kernel_build.clone(),
        operation_settings: "radiometry geometry v1".into(),
    };
    let handle = b.upload_systems_cached(&cache_key, std::slice::from_ref(&packed))?;
    let result = (|| {
        let mut p = pupil_plan(
            b, t, &prepared, handle, &field, wl, points, state, clip, ctx,
        )?;
        if scalar {
            for op in &mut p.operations {
                if let ResidentOp::Radiometry(Request::Polarization(r)) = op {
                    r.scalar = true;
                }
            }
            p.settings["scalar"] = serde_json::json!(true);
        }
        b.submit_plan(p, &ctx.subrange(0.05, 0.9))?.wait()
    })();
    b.release_systems(handle)?;
    let mut result = result?;
    ctx.check()?;
    let sum = b.read_view(result.outputs[2].view())?;
    let sums = sum.as_slice::<Df32>()?;
    let mut summary = std::array::from_fn(|c| std::array::from_fn(|k| sums[4 * c + k].to_f64()));
    let status_view = b.read_view(result.outputs[1].view())?;
    let status = &status_view.as_slice::<u32>()?[..points.len()];
    let aim_failed =
        |s: u32| (s & crate::consts::MKL_RAY_CODE_MASK) == crate::consts::MKL_AIM_FAILED;
    // Unresolved device aiming is completed in f64 whether or not other rows
    // need reference evaluation; the failed-count channel gates the scan.
    if result.stats.fallback_count == 0
        && (summary[9][0] == 0. || !status.iter().any(|&s| aim_failed(s)))
    {
        let (values, trace_status) = if retain_values {
            (
                ResultSlice::resident(b.read_view(result.outputs[0].view())?, points.len()),
                ResultSlice::resident(status_view, points.len()),
            )
        } else {
            (ResultSlice::owned(vec![]), ResultSlice::owned(vec![]))
        };
        ctx.check()?;
        ctx.report(1., "radiometry complete");
        return Ok(PupilResult {
            values,
            trace_status,
            summary,
            run_info: result.run_info,
            stats: result.stats,
        });
    }
    let guard = b.read_view(result.outputs[0].view())?;
    let rows: &[PolarizationValue] = bytemuck::cast_slice(guard.words());
    let mut values = Vec::with_capacity(rows.len());
    let mut trace_status = status.to_vec();
    let mut fallback = 0;
    for (i, v) in rows.iter().enumerate() {
        if i % 256 == 0 {
            ctx.check()?;
            ctx.report(
                0.9 + 0.09 * i as f64 / rows.len() as f64,
                "polarization assembly",
            );
        }
        if v.status == crate::consts::MKL_RAD_REFERENCE || aim_failed(trace_status[i]) {
            ctx.check()?;
            let ray = t.trace_polarized(
                &field,
                wl,
                points[i][0],
                points[i][1],
                &mklight_trace::TraceOptions {
                    clip,
                    ..t.options()
                },
            );
            let value = reference_value(&ray, state);
            if !value
                .jones
                .iter()
                .chain(&value.proper)
                .chain(&value.fields)
                .chain(&value.power)
                .chain(&value.properties)
                .all(|v| v.is_finite())
            {
                return Err(bad("nonfinite f64 polarization fallback"));
            }
            values.push(value);
            trace_status[i] = reference_status(&ray.trace.status);
            fallback += 1;
        } else {
            values.push(*v);
        }
    }
    if fallback > 0 {
        // Rebuild the final summary in fixed input order only after reference
        // rows changed. This small fallback stage is reported, never disguised
        // as device reduction; the ordinary path uses P3's resident fixed tree.
        summary = [[0.; 4]; 10];
        let mut sums = [mklight_core::math::KahanSum::default(); 10];
        for c in &mut summary {
            c[2] = f64::INFINITY;
            c[3] = f64::NEG_INFINITY;
        }
        for (i, v) in values.iter().enumerate() {
            if i % 256 == 0 {
                ctx.check()?;
            }
            for (c, x) in super::summary_channels(*v, trace_status[i])
                .into_iter()
                .enumerate()
            {
                let x = x.to_f64();
                sums[c].add(x);
                summary[c][1] += 1.;
                summary[c][2] = summary[c][2].min(x);
                summary[c][3] = summary[c][3].max(x);
            }
        }
        for c in 0..10 {
            summary[c][0] = sums[c].value();
        }
        result.run_info.warnings.push(format!("P7 reference evaluated {fallback} pupil rows (coating/retarder conditioning or unresolved aiming); final summary rebuilt in f64"));
        result.run_info.settings["p7_reference_rows"] = serde_json::json!(fallback);
        result.run_info.settings["reference_read_copy_bytes"] = serde_json::json!(
            (points.len() - fallback) * std::mem::size_of::<PolarizationValue>()
                + std::mem::size_of_val(trace_status.as_slice())
        );
    }
    ctx.check()?;
    ctx.report(1., "polarization complete");
    Ok(PupilResult {
        values: ResultSlice::owned(values),
        trace_status: ResultSlice::owned(trace_status),
        summary,
        run_info: result.run_info,
        stats: result.stats,
    })
}
fn reference_status(s: &mklight_trace::TraceStatus) -> u32 {
    use crate::consts::*;
    use mklight_trace::TraceStatus;
    match s {
        TraceStatus::Ok => 0,
        TraceStatus::Miss { surface } => MKL_RAY_MISS | (*surface as u32) << 8,
        TraceStatus::Clipped { surface } => MKL_RAY_VIGNETTED | (*surface as u32) << 8,
        TraceStatus::TotalInternalReflection { surface } => MKL_RAY_TIR | (*surface as u32) << 8,
        TraceStatus::Failed { surface, reason } => {
            use mklight_trace::FailReason::*;
            let code = match reason {
                EvanescentOrder => MKL_RAY_EVANESCENT,
                NoConvergence => MKL_RAY_NOT_CONVERGED,
                OutsideDomain => MKL_RAY_OUTSIDE_DOMAIN,
                NonFinite => MKL_RAY_NONFINITE,
                Degenerate => MKL_RAY_DEGENERATE,
                AimNotConverged | FieldUnreachable => MKL_AIM_FAILED,
            };
            code | (*surface as u32) << 8
        }
    }
}
fn reference_value(
    ray: &mklight_trace::PolarizedRay,
    state: &PolarizationState,
) -> PolarizationValue {
    use bytemuck::Zeroable;
    let mut out = PolarizationValue::zeroed();
    if !ray.is_ok() {
        out.status = crate::consts::MKL_RAD_BLOCKED;
        return out;
    }
    let j = ray.jones_amplitude();
    let proper = ray.proper_jones().scale(ray.internal_transmittance.sqrt());
    for k in 0..4 {
        for (a, m) in [(&mut out.jones, j), (&mut out.proper, proper)] {
            a[2 * k] = Df32::from_f64(m.m[k / 2][k % 2].re);
            a[2 * k + 1] = Df32::from_f64(m.m[k / 2][k % 2].im);
        }
    }
    for (a, v) in ray.image_fields().iter().enumerate() {
        for (k, v) in v.iter().enumerate() {
            out.fields[6 * a + 2 * k] = Df32::from_f64(v.re);
            out.fields[6 * a + 2 * k + 1] = Df32::from_f64(v.im);
        }
    }
    let dir = ray.end_rot.transpose().mul_vec(ray.k_out);
    out.direction = [dir.x, dir.y, dir.z].map(Df32::from_f64);
    let budget = ray.energy_budget(state);
    let scalar = mklight_trace::radiometry::unpolarized_interface_product(&ray.steps)
        * ray.internal_transmittance;
    let sp = ray.steps.iter().fold([1., 1.], |v, s| {
        let (a, b) = s.path_sp();
        [v[0] * a, v[1] * b]
    });
    let s0 = state.stokes().s0;
    out.power = [
        scalar,
        (sp[0] + sp[1]) * 0.5 * ray.internal_transmittance,
        ray.intensity(state),
        j.t_unpolarized(),
        budget.surfaces.iter().map(|s| s.diverted).sum::<f64>() / s0,
        budget.surfaces.iter().map(|s| s.absorbed).sum::<f64>() / s0,
        budget.bulk_absorbed / s0,
    ]
    .map(Df32::from_f64);
    let props = proper.properties();
    let jp = j.properties();
    out.properties = [
        props.diattenuation,
        props.retardance,
        props.diattenuation_axis[0],
        props.diattenuation_axis[1],
        props.diattenuation_axis[2],
        props.fast_axis[0],
        props.fast_axis[1],
        props.fast_axis[2],
        jp.retardance,
    ]
    .map(Df32::from_f64);
    out.defined = props.retarder_defined as u32 + 2 * jp.retarder_defined as u32;
    out
}

/// Transmission versus field/wavelength, with the reference's scalar and
/// polarised models and failed-ray denominators. DOE efficiency is the same
/// immutable per-wavelength factor used by the reference radiometry jobs.
pub fn system_transmission(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    settings: &mklight_trace::TransmissionSettings,
    ctx: &JobContext,
) -> MkResult<mklight_trace::TransmissionResult> {
    use mklight_trace::radiometry::TransmissionPoint;
    use mklight_trace::{PupilSamples, RadiometryModel, TransmissionResult};
    let timer = mklight_core::job::RunTimer::start();
    ctx.check()?;
    let fields = if settings.fields.is_empty() {
        t.fields().iter().map(|f| f.point).collect()
    } else {
        settings.fields.clone()
    };
    let wavelengths = if settings.wavelengths.is_empty() {
        (0..t.num_wavelengths()).collect()
    } else {
        settings.wavelengths.clone()
    };
    if wavelengths.iter().any(|&w| w >= t.num_wavelengths()) {
        return Err(mklight_core::MkError::InvalidModel(
            "transmission wavelength".into(),
        ));
    }
    let state = match settings.model {
        RadiometryModel::Unpolarized => PolarizationState::Unpolarized,
        RadiometryModel::Polarized { state } => state,
    };
    let samples = settings.pupil.points();
    if samples.is_empty() {
        return Err(mklight_core::MkError::InvalidModel(
            "transmission empty pupil".into(),
        ));
    }
    let channel = if matches!(settings.model, RadiometryModel::Unpolarized) {
        0
    } else {
        1
    };
    let mut points = vec![];
    let mut warnings = t.warnings().to_vec();
    let mut runs = vec![];
    for (f, field) in fields.iter().enumerate() {
        for (wi, &wl) in wavelengths.iter().enumerate() {
            let index = f * wavelengths.len() + wi;
            let total = fields.len() * wavelengths.len();
            let r = pupil_mode(
                b,
                t,
                *field,
                wl,
                &samples,
                &state,
                settings.clip,
                matches!(settings.model, RadiometryModel::Unpolarized),
                false,
                &ctx.subrange(
                    index as f64 / total as f64 * 0.99,
                    (index + 1) as f64 / total as f64 * 0.99,
                ),
            )?;
            let sum = r.summary[channel][0]
                * mklight_trace::radiometry::diffraction_product(t.system(), wl);
            let usable = r.summary[5][0] as usize;
            let ok = r.summary[6][0] as usize;
            let failed = r.summary[9][0] as usize;
            if failed > 0 {
                warnings.push(format!("field {f}, wavelength {wl}: {failed} of {} pupil samples failed and were excluded",samples.len()));
            }
            warnings.extend(r.run_info.warnings);
            runs.push(serde_json::json!({"field":f,"wavelength":wl,"stats":r.stats,"backend":r.run_info.backend,"settings":r.run_info.settings}));
            points.push(TransmissionPoint {
                field: f,
                wavelength: wl,
                wavelength_um: t.system().wavelengths_um[wl],
                transmission: if usable > 0 { sum / usable as f64 } else { 0. },
                mean_unvignetted: if ok > 0 { sum / ok as f64 } else { 0. },
                unvignetted_fraction: if usable > 0 {
                    ok as f64 / usable as f64
                } else {
                    0.
                },
                samples: usable,
                failed,
            });
        }
    }
    let polychromatic = (0..fields.len())
        .map(|f| {
            let mut num = mklight_core::math::KahanSum::default();
            let mut den = num;
            for (wi, &wl) in wavelengths.iter().enumerate() {
                let w = t.system().wavelength_weights[wl];
                num.add(w * points[f * wavelengths.len() + wi].transmission);
                den.add(w);
            }
            if den.value() > 0. {
                num.value() / den.value()
            } else {
                0.
            }
        })
        .collect();
    let seed = match settings.pupil {
        PupilSamples::Random { seed, .. } => Some(seed),
        _ => None,
    };
    let mut run_info = timer.finish(
        &t.system().snapshot,
        "system_transmission",
        settings,
        &b.info().name,
        b.precision().model,
        seed,
        warnings,
    );
    run_info.settings["p7_runs"] = serde_json::json!(runs);
    ctx.check()?;
    ctx.report(1., "transmission complete");
    Ok(TransmissionResult {
        fields,
        wavelengths,
        points,
        polychromatic,
        run_info,
    })
}

/// Assemble the reference pupil-map schema from device matrices/properties and
/// P3 statistics. This only widens/repackages final values; retarder extraction
/// and all pupil reductions have already run on the backend.
pub fn polarization_pupil_map(
    b: &dyn ComputeBackend,
    t: &SequentialTracer,
    settings: &mklight_trace::PolarizationMapSettings,
    ctx: &JobContext,
) -> MkResult<mklight_trace::PolarizationPupilMap> {
    use crate::consts::*;
    use mklight_core::materials::coating::Complex64 as C;
    use mklight_trace::{
        FailReason, JonesMatrix, PolarizationProperties, PolarizationPupilMap, TraceStatus,
        radiometry::PupilPolarization,
    };
    let timer = mklight_core::job::RunTimer::start();
    let points = mklight_trace::PupilSamples::Grid { n: settings.grid }.points();
    let r = pupil(
        b,
        t,
        settings.field,
        settings.wavelength,
        &points,
        &settings.state,
        settings.clip,
        &ctx.subrange(0., 0.95),
    )?;
    let eta = mklight_trace::radiometry::diffraction_product(t.system(), settings.wavelength);
    let amp = eta.sqrt();
    let mut samples = Vec::with_capacity(points.len());
    for (i, (v, point)) in r.values.iter().zip(points).enumerate() {
        if i % 256 == 0 {
            ctx.check()?;
        }
        let complex = |a: &[Df32; 8], k: usize| C::new(a[2 * k].to_f64(), a[2 * k + 1].to_f64());
        let jones = JonesMatrix {
            m: std::array::from_fn(|row| {
                std::array::from_fn(|col| complex(&v.jones, 2 * row + col) * amp)
            }),
        };
        let u = v.power[3].to_f64() * eta;
        let d = v.properties[0].to_f64();
        let top = u * (1. + d);
        let det = (complex(&v.proper, 0) * complex(&v.proper, 3)
            - complex(&v.proper, 1) * complex(&v.proper, 2))
            * eta;
        let props = PolarizationProperties {
            t_unpolarized: u,
            t_max: top,
            t_min: if top > 0. { det.norm_sqr() / top } else { 0. },
            diattenuation: d,
            diattenuation_axis: std::array::from_fn(|k| v.properties[2 + k].to_f64()),
            retardance: v.properties[1].to_f64(),
            fast_axis: std::array::from_fn(|k| v.properties[5 + k].to_f64()),
            retarder_defined: v.defined & 1 != 0 && eta > 0.,
        };
        let props = if eta > 0. {
            props
        } else {
            JonesMatrix {
                m: [[C::new(0., 0.); 2]; 2],
            }
            .properties()
        };
        let word = r.trace_status[i];
        let surface = (word >> MKL_RAY_SURFACE_SHIFT) as usize;
        let status = match word & MKL_RAY_CODE_MASK {
            MKL_RAY_OK => TraceStatus::Ok,
            MKL_RAY_MISS => TraceStatus::Miss { surface },
            MKL_RAY_TIR => TraceStatus::TotalInternalReflection { surface },
            MKL_RAY_VIGNETTED => TraceStatus::Clipped { surface },
            code => TraceStatus::Failed {
                surface,
                reason: match code {
                    MKL_RAY_NOT_CONVERGED => FailReason::NoConvergence,
                    MKL_RAY_OUTSIDE_DOMAIN => FailReason::OutsideDomain,
                    MKL_RAY_DEGENERATE => FailReason::Degenerate,
                    MKL_RAY_EVANESCENT => FailReason::EvanescentOrder,
                    MKL_AIM_FAILED => FailReason::AimNotConverged,
                    _ => FailReason::NonFinite,
                },
            },
        };
        samples.push(PupilPolarization {
            pupil: point,
            status,
            jones,
            intensity: v.power[2].to_f64() * eta,
            properties: props,
            retardance_jones: if eta > 0. {
                v.properties[8].to_f64()
            } else {
                0.
            },
        });
    }
    let ok = r.summary[6][0];
    let defined = if eta > 0. { r.summary[7][0] } else { 0. };
    let undefined = (ok - defined) as usize;
    let mut warnings = t.warnings().to_vec();
    warnings.extend(r.run_info.warnings);
    if undefined > 0 {
        warnings.push(format!("{undefined} pupil samples have a singular Jones matrix (no defined retarder) and are excluded from the retardance statistics"));
    }
    let mut run_info = timer.finish(
        &t.system().snapshot,
        "polarization_pupil_map",
        settings,
        &r.run_info.backend,
        r.run_info.precision,
        None,
        warnings,
    );
    run_info.settings["p7_plan"] = r.run_info.settings;
    let out = PolarizationPupilMap {
        samples,
        mean_intensity: r.summary[1][0] / ok.max(1.) * eta,
        max_diattenuation: if eta > 0. { r.summary[3][3] } else { 0. },
        max_retardance: if defined > 0. {
            r.summary[4][3].sqrt()
        } else {
            0.
        },
        rms_retardance: if defined > 0. {
            (r.summary[4][0] / defined).sqrt()
        } else {
            0.
        },
        undefined_retarder: undefined,
        run_info,
    };
    ctx.check()?;
    ctx.report(1., "polarization map complete");
    Ok(out)
}
