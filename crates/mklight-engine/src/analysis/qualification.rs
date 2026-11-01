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
    // Candidate solvers qualify every derived system and operand themselves.
    if matches!(
        analysis.name(),
        "optimize" | "merit_evaluation" | "tolerance" | "glass_substitution"
    ) {
        return analysis.run(input, settings, ctx);
    }
    let reference = AnalysisInput::with_resolver(
        input.snapshot.clone(),
        input.library.clone(),
        BackendInfo::cpu_reference(),
        Box::new(|| input.resolved()),
    )
    .with_stop_token(input.stop.clone())
    .with_library_hash(input.library_hash().into());
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
    if let Some(q) = &cached {
        if q["passed"] == false {
            fallback(q["reason"].as_str().unwrap().into(), q.clone());
            return analysis.run(&reference, settings, ctx);
        }
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
    let qualification = match cached {
        Some(q) => q,
        None => {
            let cpu = analysis.run(&reference, settings, &ctx.subrange(0.65, 0.95))?;
            let report = compare(&cpu, &out)?;
            ctx.check()?;
            let qualification = json!({
                "passed": report.reason.is_none(), "reason": report.reason,
                "chain": analysis.name(), "version": VERSION,
                "mode": "full_requested_grid_against_independent_f64",
                "margin": 1.0/3.0, "geometry": q,
                "compared_values": report.compared,
                "maximum_absolute_error": report.maximum,
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
        // Conservative scalar sentinel, tighter than the optical publication
        // gates. Unit-specific gates apply to phase, directions and radiometry.
        let budget = if path.contains("direction") {
            2e-6
        } else if path.contains("opd") || path.contains("wavefront") || path.contains("zernike") {
            5e-3
        } else if path.contains("modulation")
            || path.contains("mtf")
            || path.contains("strehl")
            || path.contains("encircled")
            || path.contains("ensquared")
        {
            2e-3
        } else if path.contains("retardance") {
            1e-3
        } else if path.contains("transmission") || path.contains("flux") || path.contains("weight")
        {
            1e-5 * a.abs() + 1e-10
        } else {
            (1e-7 * a.abs().max(1.)).min(2e-5)
        };
        if error > budget / 3. {
            self.fail(
                path,
                &format!("error {error:e} exceeds sampled margin {:e}", budget / 3.),
            );
        }
    }
    fn image(&mut self, a: &[f64], b: &[f64], path: &str) {
        let mut norm = 0.;
        let mut delta = 0.;
        let mut energy = 0.;
        for (&a, &b) in a.iter().zip(b) {
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
fn compare(cpu: &AnalysisOutput, gpu: &AnalysisOutput) -> EngineResult<Comparison> {
    use crate::buffer::BufferData;
    let mut c = Comparison::default();
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
                for (&a, &b) in a.iter().zip(b) {
                    c.number(a as f64, b as f64, &path);
                }
            }
            (BufferData::F64(a), BufferData::F64(b))
                if path.contains("intensity")
                    || path.contains("irradiance")
                    || path == "buffer.image"
                    || path.contains("line_spread") =>
            {
                c.image(a, b, &path)
            }
            (BufferData::F64(a), BufferData::F64(b)) => {
                for (&a, &b) in a.iter().zip(b) {
                    c.number(a, b, &path);
                }
            }
            _ if a.data == b.data => (),
            _ => c.fail(&a.name, "typed buffer values differ"),
        }
    }
    Ok(c)
}
