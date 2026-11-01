//! Per-chain publication gate. A geometry verdict alone never qualifies a
//! transform, a reduction, or a changed set of analysis settings.
use super::{Analysis, AnalysisInput, AnalysisOutput, BackendUse, gpu};
use crate::{backend::BackendInfo, error::EngineResult};
use mklight_core::{JobContext, MkError, job::Precision};
use serde_json::{Value, json};

pub(crate) const VERSION: u32 = 1;

pub(crate) fn run(
    analysis: &dyn Analysis,
    input: &AnalysisInput<'_>,
    settings: &Value,
    ctx: &JobContext,
) -> EngineResult<AnalysisOutput> {
    let Some(device) = gpu::device(input) else {
        return analysis.run(input, settings, ctx);
    };
    let reference = AnalysisInput::with_resolver(
        input.snapshot.clone(),
        input.library.clone(),
        BackendInfo::cpu_reference(),
        Box::new(|| input.resolved()),
    )
    .with_stop_token(input.stop.clone())
    .with_library_hash(input.library_hash().into());
    let cost = if crate::backend::Backend::is_auto_request(&input.requested_backend) {
        super::cost::decision(input, analysis.name(), settings, &device)
    } else {
        json!({"decision":"explicit_backend","reason":"cost exceptions bypassed"})
    };
    *input.cost_decision.borrow_mut() = Some(cost.clone());
    if cost["decision"] == "cpu_reference" {
        input.report_backend(BackendUse {
            id: "cpu_reference".into(),
            precision: Precision::F64Reference,
            device: None,
            fallback_reason: Some("auto: recorded full-chain CPU cost advantage".into()),
            qualification: None,
            deliberate: true,
        });
        return analysis.run(&reference, settings, ctx);
    }
    // Candidate solvers qualify every derived system and operand themselves.
    if matches!(
        analysis.name(),
        "optimize" | "merit_evaluation" | "tolerance" | "glass_substitution"
    ) {
        return analysis.run(input, settings, ctx);
    }
    let key = serde_json::to_string(&json!({
        "chain": analysis.name(), "version": VERSION,
        "snapshot": input.snapshot.hash().to_hex(), "library": input.library_hash(),
        "settings": settings, "precision": device.backend.precision(),
        "geometry_settings": device.qualification,
    }))?;
    let cached = device.chain_qualification(&key);
    let fallback = |reason: String, qualification: Value| {
        input.report_backend(BackendUse {
            id: "cpu_reference".into(),
            precision: Precision::F64Reference,
            device: None,
            fallback_reason: Some(reason),
            qualification: Some(qualification),
            deliberate: false,
        });
    };
    if let Some(q) = &cached
        && q["passed"] == false
    {
        fallback(q["reason"].as_str().unwrap().into(), q.clone());
        return analysis.run(&reference, settings, ctx);
    }
    let tracer = mklight_trace::SequentialTracer::new(input.resolved()?)?;
    let geometry_key = format!("{}:{}", input.snapshot.hash(), input.library_hash());
    let q = device.qualify(&geometry_key, &tracer, &ctx.subrange(0., 0.15))?;
    if !q.passed {
        fallback(
            q.reason.clone().unwrap_or_default(),
            serde_json::to_value(q.as_ref())?,
        );
        return analysis.run(&reference, settings, &ctx.subrange(0.15, 1.));
    }
    let out = match analysis.run(input, settings, &ctx.subrange(0.15, 0.65)) {
        Ok(out) => out,
        Err(e) if e.is_cancelled() => return Err(e),
        Err(e) => {
            fallback(
                format!("resident {} failed: {e}", analysis.name()),
                json!({"passed":false,"geometry":q}),
            );
            return analysis.run(&reference, settings, &ctx.subrange(0.65, 1.));
        }
    };
    ctx.check()?;
    let completed_on_reference = input
        .backend_used
        .borrow()
        .as_ref()
        .filter(|used| used.precision == Precision::F64Reference)
        .map(|used| {
            used.fallback_reason
                .clone()
                .unwrap_or_else(|| "native adapter required reference completion".into())
        });
    if let Some(reason) = completed_on_reference {
        let qualification = json!({"passed":false,"chain":analysis.name(),"version":VERSION,
            "reason":reason,"mode":"whole_result_reference_completion","geometry":q});
        device.cache_chain_qualification(key, qualification.clone());
        fallback(reason, qualification);
        return Ok(out);
    }
    let qualification = match cached {
        Some(q) => q,
        None => {
            let cpu = analysis.run(&reference, settings, &ctx.subrange(0.65, 0.95))?;
            let report = compare(&cpu, &out, analysis.name(), ctx)?;
            ctx.check()?;
            let qualification = json!({
                "passed": report.reason.is_none(), "reason": report.reason,
                "chain": analysis.name(), "version": VERSION,
                "mode": "full_requested_grid_against_independent_f64",
                "margin": 1.0/3.0, "geometry": q,
                "compared_values": report.compared,
                "maximum_absolute_error": report.maximum,
                "maximum_image_l1_relative":report.image_l1_relative,
                "maximum_image_energy_relative":report.energy_relative,
                "gates": "GPU-9.4-G-OPL-R-E-T-D-v1",
            });
            device.cache_chain_qualification(key, qualification.clone());
            if qualification["passed"] == false {
                fallback(
                    qualification["reason"].as_str().unwrap().into(),
                    qualification,
                );
                return Ok(cpu);
            }
            qualification
        }
    };
    if let Some(used) = input.backend_used.borrow_mut().as_mut() {
        used.qualification = Some(qualification);
    } else {
        return Err(
            MkError::Backend("resident route did not report its actual backend".into()).into(),
        );
    }
    Ok(out)
}

#[derive(Default)]
struct Comparison {
    compared: usize,
    maximum: f64,
    image_l1_relative: f64,
    energy_relative: f64,
    quantile_interval_mm: Option<f64>,
    energy_flux: Option<f64>,
    reason: Option<String>,
}
impl Comparison {
    fn fail(&mut self, path: &str, reason: &str) {
        if self.reason.is_none() {
            self.reason = Some(format!("chain qualification at {path}: {reason}"));
        }
    }
    fn number(&mut self, a: f64, b: f64, path: &str) {
        self.compared += 1;
        if a.is_nan() && b.is_nan() || a == b {
            return;
        }
        if !a.is_finite() || !b.is_finite() {
            self.fail(path, "undefined/survival masks differ");
            return;
        }
        let error = (a - b).abs();
        self.maximum = self.maximum.max(error);
        // GPU §9.4 publication budgets, with the sampled one-third margin.
        // Undefined masks and discrete values are compared separately.
        let mut budget = if let Some(flux) = self
            .energy_flux
            .filter(|_| path.starts_with("result.energy."))
        {
            1e-5 * flux + 1e-10
        } else if path.contains("direction") {
            2e-6
        } else if path.contains("opd") || path.contains("wavefront") || path.contains("zernike") {
            5e-3
        } else if path.contains("modulation")
            || path.contains("mtf")
            || path.contains("strehl")
            || path.contains("encircled")
            || path.contains("ensquared")
            || path.contains("buffer.tangential")
            || path.contains("buffer.sagittal")
            || path.contains("diffraction_limit")
        {
            2e-3
        } else if path.contains("angle") || path.contains("incidence") {
            2e-4
        } else if path.contains("magnification") || path.contains("variation") {
            (1e-3 * a.abs()).max(1e-5)
        } else if path.contains("rms") || path.contains("radius") || path.contains("merit") {
            (0.005 * a.abs()).max(2e-5)
        } else if path.contains("retardance") {
            1e-3
        } else if path.contains("transmission") || path.contains("flux") || path.contains("weight")
        {
            1e-5 * a.abs() + 1e-10
        } else {
            2e-5
        };
        if ["r50", "r80", "r90", "half_width80"]
            .iter()
            .any(|v| path.contains(v))
            && let Some(step) = self.quantile_interval_mm
        {
            budget = budget.min(step);
        }
        if error > budget / 3. {
            self.fail(
                path,
                &format!("error {error:e} exceeds sampled margin {:e}", budget / 3.),
            );
        }
    }
    fn image(
        &mut self,
        a: &[f64],
        b: &[f64],
        path: &str,
        ctx: &JobContext,
    ) -> mklight_core::MkResult<()> {
        let mut norm = 0.;
        let mut delta = 0.;
        let mut energy = 0.;
        for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
            if i % 4096 == 0 {
                ctx.check()?;
            }
            self.compared += 1;
            if !a.is_finite() || !b.is_finite() {
                if !(a.is_nan() && b.is_nan()) && a != b {
                    self.fail(path, "image masks differ");
                }
                continue;
            }
            norm += a.abs();
            delta += (a - b).abs();
            energy += a - b;
            self.maximum = self.maximum.max((a - b).abs());
        }
        if delta > norm * 2e-3 / 3. || energy.abs() > norm * 1e-3 / 3. {
            self.fail(path, "image L1 or integrated-energy gate failed");
        }
        self.image_l1_relative =
            self.image_l1_relative
                .max(if norm > 0. { delta / norm } else { 0. });
        self.energy_relative =
            self.energy_relative
                .max(if norm > 0. { energy.abs() / norm } else { 0. });
        Ok(())
    }
    fn matrix_part(
        &mut self,
        a: &[f64],
        b: &[f64],
        path: &str,
        ctx: &JobContext,
    ) -> mklight_core::MkResult<()> {
        for (i, (a, b)) in a
            .as_chunks::<4>()
            .0
            .iter()
            .zip(b.as_chunks::<4>().0)
            .enumerate()
        {
            if i % 1024 == 0 {
                ctx.check()?;
            }
            let norm = a.iter().map(|v| v * v).sum::<f64>().sqrt();
            let error = a
                .iter()
                .zip(b)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
            self.compared += 4;
            if a.iter().zip(b).any(|(a, b)| {
                a.is_nan() != b.is_nan()
                    || a.is_infinite() != b.is_infinite()
                    || (a.is_infinite() && a != b)
            }) {
                self.fail(path, "matrix masks differ");
            }
            if error > (1e-4 * norm + 1e-7 / std::f64::consts::SQRT_2) / 3. {
                self.fail(path, "Jones norm gate failed");
            }
            self.maximum = self.maximum.max(error);
        }
        Ok(())
    }
    fn json(&mut self, a: &Value, b: &Value, path: &str) {
        match (a, b) {
            (Value::Object(a), Value::Object(b)) => {
                for (key, value) in a {
                    if matches!(
                        key.as_str(),
                        "run_info"
                            | "run"
                            | "warnings"
                            | "evaluations"
                            | "iterations"
                            | "duration_ms"
                    ) {
                        continue;
                    }
                    match b.get(key) {
                        Some(v) => self.json(value, v, &format!("{path}.{key}")),
                        None => self.fail(path, &format!("missing result key {key}")),
                    }
                }
            }
            (Value::Array(a), Value::Array(b)) => {
                if a.len() != b.len() {
                    self.fail(path, "result lengths differ");
                    return;
                }
                for (i, (a, b)) in a.iter().zip(b).enumerate() {
                    self.json(a, b, &format!("{path}[{i}]"));
                }
            }
            (Value::Number(a), Value::Number(b)) if a.is_f64() || b.is_f64() => {
                self.number(a.as_f64().unwrap(), b.as_f64().unwrap(), path)
            }
            _ if a == b => (),
            _ => self.fail(path, "discrete values or undefined masks differ"),
        }
    }
}
fn compare(
    cpu: &AnalysisOutput,
    gpu: &AnalysisOutput,
    analysis: &str,
    ctx: &JobContext,
) -> EngineResult<Comparison> {
    use crate::buffer::BufferData;
    let radius: Vec<f64> = cpu
        .buffers
        .iter()
        .find(|b| b.name == "radius_mm")
        .and_then(|b| match &b.data {
            BufferData::F64(v) => Some(v.clone()),
            _ => None,
        })
        .unwrap_or_else(|| {
            cpu.json["radius"]
                .as_array()
                .map(|v| v.iter().filter_map(Value::as_f64).collect())
                .unwrap_or_default()
        });
    let mut c = Comparison {
        energy_flux: cpu.json["energy"]["emitted"].as_f64(),
        quantile_interval_mm: radius
            .windows(2)
            .map(|v| (v[1] - v[0]).abs())
            .filter(|v| *v > 0.)
            .reduce(f64::min),
        ..Default::default()
    };
    c.json(&cpu.json, &gpu.json, "result");
    if cpu.buffers.len() != gpu.buffers.len() {
        c.fail("buffers", "buffer count differs");
        return Ok(c);
    }
    for (a, b) in cpu.buffers.iter().zip(&gpu.buffers) {
        if a.name != b.name || a.shape != b.shape {
            c.fail(&a.name, "buffer identity/shape differs");
            continue;
        }
        let path = format!("buffer.{}", a.name);
        match (&a.data, &b.data) {
            (BufferData::F32(a), BufferData::F32(b)) => {
                for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
                    if index % 4096 == 0 {
                        ctx.check()?;
                    }
                    c.number(a as f64, b as f64, &path);
                }
            }
            (BufferData::F64(a), BufferData::F64(b))
                if (analysis == "psf" && path.contains("data_"))
                    || (analysis == "line_spread"
                        && (path.starts_with("buffer.x_") || path.starts_with("buffer.y_")))
                    || path.contains("intensity")
                    || path.contains("irradiance")
                    || path == "buffer.image"
                    || path.contains("line_spread") =>
            {
                c.image(a, b, &path, ctx)?
            }
            (BufferData::F64(a), BufferData::F64(b))
                if path.contains("jones_real") || path.contains("jones_imag") =>
            {
                c.matrix_part(a, b, &path, ctx)?
            }
            (BufferData::F64(a), BufferData::F64(b)) => {
                let columns = cpu.json["columns"].as_array();
                for (index, (&a, &b)) in a.iter().zip(b).enumerate() {
                    if index % 4096 == 0 {
                        ctx.check()?;
                    }
                    if path == "buffer.values"
                        && let Some(columns) = columns
                    {
                        let name = columns[index % columns.len()]["name"].as_str().unwrap();
                        c.number(a, b, &format!("{path}.{name}"));
                        continue;
                    }
                    c.number(a, b, &path);
                }
            }
            _ if a.data == b.data => (),
            _ => c.fail(&a.name, "typed buffer values differ"),
        }
    }
    Ok(c)
}
