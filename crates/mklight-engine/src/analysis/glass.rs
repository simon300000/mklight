//! Glass substitution job, sharing optimize's merit and backend machinery.

use super::*;
use mklight_core::MkResult;
use mklight_optim::discrete::{DiscreteSettings, LocallyRefined, NearestChoices, tabu_search};
use mklight_optim::glass::{
    CandidateSet, DISTANCE_WEIGHTS, GlassChoice, MaterialVariable, model_variables,
};
use mklight_optim::mixed::{MixedHammerSettings, RefinedSearch, compare, mixed_hammer};

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct SearchBudget {
    max_combinations: usize,
    max_seconds: Option<f64>,
}
impl Default for SearchBudget {
    fn default() -> Self {
        Self {
            max_combinations: 100,
            max_seconds: None,
        }
    }
}

#[derive(Clone, Copy, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
struct LocalBudget {
    max_iterations: usize,
    max_evaluations: usize,
}
impl Default for LocalBudget {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            max_evaluations: 10_000,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    variables: Vec<VariableSpec>,
    material_variables: Vec<MaterialVariable>,
    hold_efl: bool,
    merit: Value,
    default_merit: Value,
    operands: Vec<Operand>,
    constraints: Vec<Operand>,
    algorithm: String,
    budget: SearchBudget,
    local: LocalBudget,
    hammer: MixedHammerSettings,
    neighbours: usize,
    pair_moves: usize,
    top_n: usize,
    fd_scheme: FdScheme,
    fd_relative_step: Option<f64>,
    seed: u64,
    backend: BackendChoice,
    gpu_batch: usize,
    feasibility_tolerance: f64,
    stop_key: Option<String>,
}

/// Immutable prescription and candidate space shared by parallel local solves.
#[derive(Clone)]
struct LensSpace {
    base: OpticalSystem,
    variables: Vec<VariableSpec>,
    materials: Vec<MaterialVariable>,
    candidates: Vec<Vec<GlassChoice>>,
    indices: Vec<usize>,
    merit: MeritFunctionSpec,
}

impl LensSpace {
    fn problem<E: OperandEvaluator>(
        &self,
        assignment: &[usize],
        evaluator: E,
    ) -> MkResult<OpticalProblem<E>> {
        let mut system = self.base.clone();
        let mut spec = self.merit.clone();
        for (v, &c) in assignment.iter().enumerate() {
            let glass = &self.candidates[v][c];
            system.surfaces[self.indices[v]].material = glass.material();
            if let Some(max) = glass.max_thickness_mm {
                // Preserve the transmission filter while changing thickness,
                // including the thicknesses used by multiple configurations.
                for config in 0..system.configurations.len().max(1) {
                    let mut o = Operand::new(OperandKind::CentreThickness {
                        surface: self.materials[v].surface,
                        to: None,
                    })
                    .target(max)
                    .as_constraint();
                    o.comparison = Comparison::LessThan;
                    o.config = config;
                    o.label = "glass internal transmission thickness limit".into();
                    spec.operands.push(o);
                }
            }
        }
        OpticalProblem::new(
            system,
            self.variables.clone(),
            spec,
            evaluator,
            &OperandRegistry::builtin(),
        )
    }

    fn glasses(&self, assignment: &[usize]) -> Vec<Value> {
        assignment
            .iter()
            .enumerate()
            .map(|(v, &c)| {
                let g = &self.candidates[v][c];
                json!({"surface": self.materials[v].surface, "catalog": g.catalog, "name": g.name,
                   "nd": g.nd, "vd": g.vd, "dpgf": g.dpgf})
            })
            .collect()
    }
}

pub struct GlassSubstitution;

impl Analysis for GlassSubstitution {
    fn name(&self) -> &str {
        "glass_substitution"
    }
    fn title(&self) -> &str {
        "Glass substitution"
    }
    fn category(&self) -> &str {
        "optimization"
    }
    fn description(&self) -> &str {
        "Searches real catalogue glasses with parallel LM re-optimisation for every combination. \
         Tabu search or mixed glass/shape basin hopping; returns ranked complete prescriptions. \
         Empty material variables snap all model glasses to their nearest real candidates."
    }
    fn gpu_path(&self) -> bool {
        true
    }

    fn settings_schema(&self) -> Value {
        use super::super::schema::*;
        let mut schema = Optimize.settings_schema();
        let props = schema["properties"].as_object_mut().unwrap();
        for key in ["algorithm", "options", "max_iterations"] {
            props.remove(key);
        }
        let candidate_schema = settings_from_default::<CandidateSet>(json!({
            "catalogs": {"items": {"type": "string"}, "description": "Empty: all user catalogues, or BUILTIN if none loaded. Names are case-insensitive."},
            "statuses": {"items": {"enum": ["standard", "preferred", "obsolete", "special", "melt"]}, "description": "AGF codes 0, 1, 2, 3, 4. Default excludes obsolete and special; exclude-substitution is always honoured."},
            "window": {"type": ["object", "null"], "properties": {"nd": {"type":"number", "minimum":0}, "vd": {"type":"number", "minimum":0}}, "required":["nd","vd"], "description": "Absolute nd/Vd half-widths about the starting glass, before nearest selection."},
            "nearest": {"type": ["integer", "null"], "minimum": 1, "description": "K nearest in nd, Vd/100, delta PgF; squared-distance weights 1, 1, 100. null: unlimited."},
            "include": {"items":{"type":"string"}, "description": "Names or catalog:name added beyond proximity limits; status/quality/exclude filters still apply."},
            "exclude": {"items":{"type":"string"}, "description": "Names or catalog:name always excluded."},
            "min_transmission": {"type":["number","null"], "minimum":0, "maximum":1, "description":"Internal transmission at the shortest system wavelength through the element. Missing data passes. LM also enforces the resulting thickness limit."},
            "max_relative_cost": {"type":["number","null"], "minimum":0},
            "max_climate_class": {"type":["number","null"], "minimum":0},
            "max_stain_class": {"type":["number","null"], "minimum":0}
        }));
        let mut material = obj_opt(
            &[
                (
                    "surface",
                    int("Stable id of the surface before the glass gap."),
                ),
                ("candidates", candidate_schema),
            ],
            &["candidates"],
        );
        material["properties"]["candidates"]["default"] = json!(CandidateSet::default());
        props.insert("material_variables".into(), json!({"type":"array", "default":[], "items":material,
            "description":"Material variables; cemented elements are separate variables. Empty snaps every model glass with K=10."}));
        props.insert(
            "algorithm".into(),
            json!({"type":"string", "enum":["tabu","hammer"], "default":"tabu"}),
        );
        let mut budget = settings_from_default::<SearchBudget>(json!({
            "max_combinations":{"minimum":1,"description":"Maximum local solves. Tabu: distinct assignments; hammer: includes different shapes for the same glasses."},
            "max_seconds":{"type":["number","null"],"exclusiveMinimum":0,"description":"Search wall-time limit; retains best partial refinements. null: no time limit. Timed runs are not reproducible."}
        }));
        budget["default"] = json!(SearchBudget::default());
        props.insert("budget".into(), budget);
        let mut local = settings_from_default::<LocalBudget>(
            json!({"max_iterations":{"minimum":1}, "max_evaluations":{"minimum":1}}),
        );
        local["default"] = json!(LocalBudget::default());
        props.insert("local".into(), local);
        let mut h = settings_from_default::<MixedHammerSettings>(json!({
            "rounds":{"minimum":1}, "batch":{"minimum":1}, "step":{"exclusiveMinimum":0},
            "swap_probability":{"minimum":0,"maximum":1},"temperature":{"minimum":0},"stall_rounds":{"minimum":1}
        }));
        h["default"] = json!(MixedHammerSettings::default());
        props.insert("hammer".into(), h);
        props.insert("neighbours".into(), json!({"type":"integer","minimum":1,"default":4,"description":"Nearest neighbours per glass; seeded restarts can reach other candidates."}));
        props.insert("pair_moves".into(), json!({"type":"integer","minimum":0,"default":4,"description":"Sampled simultaneous swaps per tabu iteration or hammer round."}));
        props.insert(
            "top_n".into(),
            json!({"type":"integer","minimum":1,"default":10}),
        );
        props.get_mut("hold_efl").unwrap()["description"] = json!(
            "Hold the starting EFL as an explicit constraint unless the merit already has an efl operand."
        );
        props.get_mut("feasibility_tolerance").unwrap()["description"] = json!(
            "Total normalised constraint violation regarded as feasible for ranking and results[].feasible."
        );
        props.get_mut("stop_key").unwrap()["description"] =
            json!("Key for request_stop: stop and keep ranked results. One running job per key.");
        schema
    }

    fn result_schema(&self) -> Value {
        use super::super::schema::*;
        let glasses = arr(
            obj(&[
                ("surface", int("Stable surface id.")),
                ("catalog", string("")),
                ("name", string("")),
                (
                    "nd",
                    num(
                        "",
                        "d-line index from the dispersion coefficients at catalogue conditions.",
                    ),
                ),
                ("vd", num("", "Abbe number.")),
                ("dpgf", num("", "Relative partial-dispersion anomaly.")),
            ]),
            "One glass per material variable, in settings order.",
        );
        result(
            &[
                (
                    "status",
                    enumeration(&["finished", "stopped"], "Cancel discards the result."),
                ),
                ("stopped", boolean("Stop retained the ranked results.")),
                (
                    "termination",
                    string("Search termination; time_limit when its wall-time budget expired."),
                ),
                ("algorithm", enumeration(&["tabu", "hammer"], "")),
                ("starting_merit", nf()),
                ("starting_violation", nf()),
                (
                    "results",
                    arr(
                        obj(&[
                            (
                                "rank",
                                int(
                                    "One-based rank: feasible first; infeasible by violation, otherwise merit; deterministic ties.",
                                ),
                            ),
                            ("merit", nf_u("", "CPU f64 reference merit.")),
                            ("violation", nf()),
                            ("feasible", boolean("")),
                            ("glasses", glasses.clone()),
                            (
                                "variables",
                                arr(
                                    obj(&[
                                        ("param", opaque("ParamPath.")),
                                        ("start", nf()),
                                        ("value", nf()),
                                        ("change", nf()),
                                    ]),
                                    "Continuous changes relative to the input prescription.",
                                ),
                            ),
                            (
                                "system",
                                opaque(
                                    "Complete optimised OpticalSystem for preview or application in one undoable step.",
                                ),
                            ),
                            ("system_hash", string("Hash of the optimised prescription.")),
                            (
                                "local_termination",
                                string(
                                    "Local solve termination, including cancelled for a retained partial solve.",
                                ),
                            ),
                        ]),
                        "Up to top_n unique glass combinations, re-ranked on the CPU reference.",
                    ),
                ),
                (
                    "candidate_sets",
                    arr(
                        obj(&[
                            ("surface", int("")),
                            (
                                "glasses",
                                arr(
                                    obj(&[
                                        ("catalog", string("")),
                                        ("name", string("")),
                                        ("nd", num("", "")),
                                        ("vd", num("", "")),
                                        ("dpgf", num("", "")),
                                        (
                                            "max_thickness_mm",
                                            nullable(num(
                                                "mm",
                                                "Transmission constraint; null if absent or without absorption data.",
                                            )),
                                        ),
                                    ]),
                                    "Nearest first.",
                                ),
                            ),
                        ]),
                        "Resolved candidates, including model-glass snapping proposals.",
                    ),
                ),
                (
                    "statistics",
                    obj(&[
                        (
                            "combinations",
                            int("Distinct assignments evaluated, including failed refinements."),
                        ),
                        (
                            "local_runs",
                            int(
                                "Attempted local solves, including failures and interrupted solves.",
                            ),
                        ),
                        (
                            "continuous_evaluations",
                            int("Search merit evaluations, excluding final CPU reporting."),
                        ),
                        (
                            "cache_hits",
                            int(
                                "Exact refinement cache hits; tabu also caches scalar assignments.",
                            ),
                        ),
                        ("failed", int("Local solves without a usable point.")),
                    ]),
                ),
                (
                    "history",
                    arr(
                        obj(&[
                            ("combination", int("Local-run count.")),
                            ("evaluations", int("Continuous evaluations.")),
                            ("merit", nf()),
                            ("best_merit", nf()),
                            ("violation", nf()),
                        ]),
                        "Search evaluator history in deterministic input order.",
                    ),
                ),
                ("backend", evaluation_backend_schema()),
                (
                    "merit_spec",
                    opaque(
                        "Effective shared merit; transmission thickness constraints are added per assignment.",
                    ),
                ),
            ],
            vec![],
        )
    }

    fn run(
        &self,
        input: &AnalysisInput<'_>,
        settings: &Value,
        ctx: &JobContext,
    ) -> EngineResult<AnalysisOutput> {
        // Nested schemas validate limits and reject misspelled candidate/budget keys.
        super::super::schema::validate(&self.settings_schema(), settings, true).map_err(invalid)?;
        let mut s: Settings = parse(settings, "settings")?;
        let time_limit = s
            .budget
            .max_seconds
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(invalid)?;
        let stop = StopRequest::register(s.stop_key.as_deref(), input.stop.clone())?;
        let system = input.snapshot.system().clone();
        if s.variables.is_empty() {
            s.variables = auto_variables(&system);
        }
        if s.material_variables.is_empty() {
            s.material_variables = model_variables(&system);
        }
        if s.material_variables.is_empty() {
            return Err(invalid(
                "no material variables or model glasses to substitute",
            ));
        }
        let mut seen = std::collections::BTreeSet::new();
        for m in &s.material_variables {
            if !seen.insert(m.surface) {
                return Err(invalid("duplicate material variable"));
            }
            let conflicting = |p: &ParamPath| {
                matches!(
                    p,
                    ParamPath::ModelNd { .. }
                        | ParamPath::ModelVd { .. }
                        | ParamPath::ModelDpgf { .. }
                ) && p.surface() == Some(m.surface)
            };
            if s.variables.iter().any(|v| conflicting(&v.param))
                || system
                    .configurations
                    .iter()
                    .any(|c| c.overrides.iter().any(|(p, _)| conflicting(p)))
            {
                return Err(invalid(
                    "a substituted material cannot also have continuous model-glass variables or overrides",
                ));
            }
        }
        let candidates = s
            .material_variables
            .iter()
            .map(|v| v.candidates.resolve(&system, v.surface, &input.library))
            .collect::<MkResult<Vec<_>>>()?;
        let cpu = TraceEvaluator::new(input.library.clone());
        let reg = OperandRegistry::builtin();
        let default_merit = patched(
            &DefaultMeritSpec::default(),
            &s.default_merit,
            "default_merit",
        )?;
        let mut spec = build_merit(&system, &s.merit, &default_merit, &s.variables)?;
        let mut warnings = Vec::new();
        if s.merit.is_null() {
            warnings.extend(relax_fixed_boundaries(
                &system,
                &mut spec,
                &s.variables,
                &cpu,
                &reg,
                ctx,
            )?);
        }
        spec.operands.extend(s.operands);
        spec.operands.extend(s.constraints.into_iter().map(|mut o| {
            o.role = OperandRole::Constraint;
            o
        }));
        if s.hold_efl && !has_efl(&spec) {
            match cpu
                .evaluate_system(&system, &[&Operand::new(OperandKind::Efl)], ctx)?
                .remove(0)
            {
                Ok(f) if f.is_finite() && f != 0.0 => spec
                    .operands
                    .push(Operand::new(OperandKind::Efl).target(f).as_constraint()),
                _ => warnings.push("the starting system has no finite EFL to hold".into()),
            }
        }
        let reference = OpticalProblem::new(
            system.clone(),
            s.variables.clone(),
            spec.clone(),
            cpu.clone(),
            &reg,
        )?;
        let x0 = reference.start_point()?;
        let start = cpu_scores(&reference, &[x0.clone()], ctx)?[0];
        let local = LsqSettings {
            max_iterations: s.local.max_iterations,
            max_evaluations: s.local.max_evaluations,
            jacobian: JacobianSettings {
                scheme: s.fd_scheme,
                relative_step: s.fd_relative_step,
                use_analytic: true,
            },
            constraints: AugLagSettings {
                max_inner_iterations: Some(INNER_ITERATIONS),
                ..Default::default()
            },
            ..Default::default()
        };
        let batch = x0.len()
            * if s.fd_scheme == FdScheme::Central {
                2
            } else {
                1
            };
        let work = Workload::new(&cpu, &system, &spec, batch);
        let mut ev = Evaluation::new(s.backend, input, s.gpu_batch, work, ctx)?;
        let space = Arc::new(LensSpace {
            indices: s
                .material_variables
                .iter()
                .map(|v| system.surface_index(v.surface).unwrap())
                .collect(),
            base: system,
            variables: s.variables,
            materials: s.material_variables,
            candidates,
            merit: spec,
        });
        let coords: Vec<_> = space
            .candidates
            .iter()
            .map(|g| g.iter().map(GlassChoice::coordinates).collect())
            .collect();
        let mut neighbours =
            NearestChoices::from_coordinates(&coords, &DISTANCE_WEIGHTS, s.neighbours);
        neighbours.pair_moves = s.pair_moves;
        let initial = vec![0; space.materials.len()];
        let build_space = space.clone();
        let gpu = ev.active_gpu().cloned();
        let builder_cpu = cpu.clone();
        let local = LocallyRefined {
            choices: space.candidates.iter().map(Vec::len).collect(),
            settings: local,
            build: Box::new(move |a| {
                let p: Box<dyn Problem + Send> = match &gpu {
                    Some(g) => Box::new(build_space.problem(a, g.clone().composite())?),
                    None => Box::new(build_space.problem(a, builder_cpu.clone())?),
                };
                Ok((p, x0.clone()))
            }),
        };
        let ((run, archive), why) = interruptible_for(ctx, &stop, time_limit, |inner| {
            let mut mon = Monitor::new(JobContext::with(inner.cancel.clone(), Progress::default()));
            mon.feasibility_tol = s.feasibility_tolerance;
            let best = mon.best_assignment.clone();
            let progress = inner.clone();
            let labels = space.clone();
            let limit = s.budget.max_combinations;
            mon = mon.with_callback(move |r| {
                if r.method != "glass_refinement" {
                    return;
                }
                let glasses = best.get().map(|b| labels.glasses(&b.assignment));
                let stage = format!(
                    "glass_substitution: combinations {}; best merit {:.6e}; glasses {}",
                    r.iteration,
                    r.best_merit,
                    json!(glasses)
                );
                progress.report(r.iteration as f64 / (limit as f64 + 1.0), &stage);
            });
            let search = RefinedSearch::new(local, mon);
            let run = if s.algorithm == "hammer" {
                mixed_hammer(
                    &search,
                    &neighbours,
                    &initial,
                    &s.hammer,
                    s.budget.max_combinations,
                    s.seed,
                    inner,
                )
            } else {
                tabu_search(
                    &search,
                    &neighbours,
                    &initial,
                    &DiscreteSettings {
                        max_evaluations: s.budget.max_combinations,
                        seed: s.seed,
                        ..Default::default()
                    },
                    &search.monitor,
                )
                .map(|r| r.termination)
            };
            (run, search.archive())
        });
        ctx.check()?; // Cancel discards; Stop only cancelled the inner context.
        let termination = match run {
            Ok(t) => t,
            Err(MkError::Cancelled) if why != Interrupt::None => Termination::Cancelled,
            Err(e) => return Err(e.into()),
        };
        ctx.report(
            ctx.progress.fraction(),
            "ranking glass combinations on the CPU reference",
        );
        let mut ranked = Vec::new();
        for (assignment, r) in &archive.results {
            ctx.check()?;
            let p = space.problem(assignment, cpu.clone())?;
            let mut baseline = p.start_point()?;
            mklight_optim::problem::project(p.variables(), &mut baseline);
            let values = cpu_scores(&p, &[baseline.clone(), r.x.clone()], ctx)?;
            let k = if values[0].better_than(&values[1], s.feasibility_tolerance) {
                0
            } else {
                1
            };
            if values[k].status.is_usable()
                && values[k].merit.is_finite()
                && values[k].violation.is_finite()
            {
                let x = if k == 0 { baseline } else { r.x.clone() };
                ranked.push((assignment, x, values[k], r.termination));
            }
        }
        ranked.sort_by(|a, b| compare(a.2, b.2, s.feasibility_tolerance).then(a.0.cmp(b.0)));
        ranked.truncate(s.top_n);
        let mut results = Vec::new();
        let mut best_snapshot = None;
        for (rank, (a, x, value, local_termination)) in ranked.into_iter().enumerate() {
            let p = space.problem(a, cpu.clone())?;
            let snapshot = Snapshot::new(p.apply(&x)?)?;
            if rank == 0 {
                best_snapshot = Some(snapshot.clone());
            }
            let variables = space.variables.iter().zip(&x).map(|(v, &x)| {
                let start = v.param.get(&space.base).expect("validated continuous variable");
                json!({"param": v.param, "start": num(start), "value": num(x), "change": num(x-start)})
            }).collect::<Vec<_>>();
            results.push(json!({"rank": rank+1, "merit": num(value.merit), "violation": num(value.violation),
                "feasible": value.violation <= s.feasibility_tolerance, "glasses": space.glasses(a), "variables": variables,
                "system": snapshot.system(), "system_hash": snapshot.hash().to_hex(), "local_termination": local_termination}));
        }
        ev.finish(input, s.backend, best_snapshot.as_ref(), ctx)?;
        // All published scores, including an empty ranking, use the reference.
        ev.report["reported_on"] = json!(TraceEvaluator::BACKEND);
        warnings.extend(ev.warnings);
        if results.is_empty() {
            warnings
                .push("no usable glass combination was retained within the search budget".into());
        }
        let json = json!({
            "status": if why == Interrupt::Stopped { "stopped" } else { "finished" }, "stopped": why == Interrupt::Stopped,
            "termination": if why == Interrupt::TimeLimit { json!("time_limit") } else { json!(termination) },
            "algorithm": s.algorithm, "starting_merit": num(start.merit), "starting_violation": num(start.violation),
            "results": results,
            "candidate_sets": space.materials.iter().zip(&space.candidates).map(|(v,c)| json!({"surface": v.surface, "glasses":c})).collect::<Vec<_>>(),
            "statistics": {"combinations": archive.assignments.len(), "local_runs": archive.local_runs,
                "continuous_evaluations": archive.evaluations, "cache_hits": archive.cache_hits, "failed": archive.failed},
            "history": archive.history.iter().map(|h| json!({"combination": h.iteration, "evaluations": h.evaluations,
                "merit":num(h.merit), "best_merit": num(h.best_merit), "violation":num(h.violation)})).collect::<Vec<_>>(),
            "backend":ev.report, "merit_spec": space.merit,
        });
        Ok(AnalysisOutput {
            json,
            buffers: Vec::new(),
            warnings,
        })
    }
}

fn cpu_scores(
    p: &OpticalProblem<TraceEvaluator>,
    points: &[Vec<f64>],
    ctx: &JobContext,
) -> MkResult<Vec<ScalarEval>> {
    let e = p.evaluate_batch(
        &CandidateBatch::from_points(p.num_variables(), points.iter().map(Vec::as_slice)),
        ctx,
    )?;
    Ok((0..points.len())
        .map(|i| ScalarEval {
            merit: merit(p.weights(), e.residuals(i), e.status[i]),
            violation: total_violation(p.constraints(), e.constraints(i), e.status[i]),
            status: e.status[i],
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisResult, run_analysis};
    use crate::backend::BackendInfo;
    use mklight_core::materials::{GlassCatalog, MaterialLibrary, agf};
    use mklight_core::{MaterialRef, SurfaceId, SurfaceShape, samples};
    use mklight_optim::merit::operand::OperandKind;

    fn doublet() -> (OpticalSystem, MaterialLibrary, Value) {
        let mut cat = GlassCatalog {
            name: "PAIR".into(),
            ..Default::default()
        };
        for (name, nd, vd) in [
            ("C_GOOD", 1.52, 60.0),
            ("C_BAD", 1.52, 30.0),
            ("F_GOOD", 1.62, 30.0),
            ("F_BAD", 1.62, 60.0),
        ] {
            cat.glasses
                .insert(name.into(), agf::model_glass(name, nd, vd, 0.0).unwrap());
        }
        // Exercise the actual AGF ingestion route with a synthetic catalogue.
        let text = agf::write_agf(&agf::AgfCatalog::from_catalog(cat)).0;
        let cat = agf::parse_agf("PAIR", &text).unwrap();
        let mut lib = MaterialLibrary::with_builtin();
        lib.add_catalog(Arc::new(cat));
        let mut system = samples::achromatic_doublet();
        system.surfaces[1].material = MaterialRef::glass("C_BAD");
        system.surfaces[2].material = MaterialRef::glass("F_BAD");
        system.surfaces[1].thickness = 3.0;
        system.surfaces[2].thickness = 2.0;
        system.surfaces[1].shape = SurfaceShape::sphere(50.0);
        system.surfaces[2].shape = SurfaceShape::sphere(-60.0);
        system.surfaces[3].shape = SurfaceShape::Plane;
        let vars = vec![
            VariableSpec::bounded(
                ParamPath::Curvature {
                    surface: SurfaceId(1),
                },
                0.005,
                0.06,
            ),
            VariableSpec::bounded(
                ParamPath::Curvature {
                    surface: SurfaceId(2),
                },
                -0.04,
                -0.001,
            ),
        ];
        let merit = MeritFunctionSpec {
            operands: vec![
                Operand::new(OperandKind::Efl).target(100.0),
                Operand::new(OperandKind::AxialColour { short: 0, long: 2 }).weight(10.0),
            ],
            ..Default::default()
        };
        let settings = json!({"variables":vars,"merit":merit,"hold_efl":false,"backend":"cpu", "seed":31,
            "material_variables":[
                {"surface":1,"candidates":{"catalogs":["PAIR"],"exclude":["F_GOOD","F_BAD"],"nearest":null}},
                {"surface":2,"candidates":{"catalogs":["PAIR"],"exclude":["C_GOOD","C_BAD"],"nearest":null}}
            ], "budget":{"max_combinations":4}, "local":{"max_iterations":100}, "neighbours":2,"pair_moves":4});
        (system, lib, settings)
    }

    fn run(system: OpticalSystem, library: MaterialLibrary, settings: &Value) -> AnalysisResult {
        let input = AnalysisInput::new(
            Snapshot::new(system).unwrap(),
            Arc::new(library),
            BackendInfo::cpu_reference(),
        );
        run_analysis(
            &GlassSubstitution,
            &input,
            settings,
            vec![],
            &JobContext::new(),
        )
        .unwrap()
    }

    #[test]
    fn cemented_doublet_finds_achromatic_pair_after_rebending() {
        let (system, lib, settings) = doublet();
        let r = run(system, lib.clone(), &settings);
        let rows = r.json["results"].as_array().unwrap();
        assert_eq!(rows.len(), 4);
        let best = &rows[0];
        assert_eq!(best["glasses"][0]["name"], "C_GOOD");
        assert_eq!(best["glasses"][1]["name"], "F_GOOD");
        assert!(best["merit"].as_f64().unwrap() < 1e-12);
        assert!(best["merit"].as_f64().unwrap() < r.json["starting_merit"].as_f64().unwrap());
        let system: OpticalSystem = serde_json::from_value(best["system"].clone()).unwrap();
        let ops = [
            Operand::new(OperandKind::Efl),
            Operand::new(OperandKind::AxialColour { short: 0, long: 2 }),
        ];
        let values = TraceEvaluator::new(Arc::new(lib))
            .evaluate_system(&system, &ops.iter().collect::<Vec<_>>(), &JobContext::new())
            .unwrap();
        assert!((values[0].as_ref().unwrap() - 100.0).abs() < 1e-6);
        assert!(
            values[1].as_ref().unwrap().abs() < 1e-6,
            "F-C focus: {:?}",
            values[1]
        );
        // Independent y / n*u refraction and translation for the three
        // cemented interfaces. The expected pair has positive crown power,
        // negative flint power and V_crown > V_flint; the other three pairs
        // cannot achromatise at positive power within these bending bounds.
        let bfl = |wavelength| {
            let mut y = 1.0;
            let mut nu = 0.0;
            let mut n = 1.0;
            for i in 1..=3 {
                let sf = &system.surfaces[i];
                let next = if i == 3 {
                    1.0
                } else {
                    let name = if i == 1 { "C_GOOD" } else { "F_GOOD" };
                    agf::model_glass(
                        name,
                        if i == 1 { 1.52 } else { 1.62 },
                        if i == 1 { 60.0 } else { 30.0 },
                        0.0,
                    )
                    .unwrap()
                    .catalog_index(wavelength)
                };
                let c = ParamPath::Curvature { surface: sf.id }
                    .get(&system)
                    .unwrap();
                nu -= y * c * (next - n);
                if i < 3 {
                    y += sf.thickness * nu / next;
                }
                n = next;
            }
            -y / nu
        };
        assert!(
            (bfl(mklight_core::materials::lines::F) - bfl(mklight_core::materials::lines::C)).abs()
                < 1e-6
        );
        assert!(
            best["variables"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["change"].as_f64().unwrap().abs() > 1e-4)
        );
        let keys: std::collections::BTreeSet<_> =
            rows.iter().map(|v| v["glasses"].to_string()).collect();
        assert_eq!(keys.len(), rows.len());
        assert!(
            rows.windows(2)
                .all(|w| w[0]["merit"].as_f64() <= w[1]["merit"].as_f64())
        );
        super::super::super::schema::validate(&GlassSubstitution.result_schema(), &r.json, true)
            .unwrap();
        super::super::super::schema::validate(
            &GlassSubstitution.settings_schema(),
            &r.run_info.settings,
            true,
        )
        .unwrap();
        assert_eq!(r.run_info.seed, Some(31));
        assert_eq!(r.run_info.analysis, "glass_substitution");
    }

    #[test]
    fn deterministic_across_thread_counts_and_top_n() {
        let (system, lib, settings) = doublet();
        let run_in = |threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| run(system.clone(), lib.clone(), &settings))
        };
        assert_eq!(run_in(1).json, run_in(4).json);
        let mut top = settings;
        top["top_n"] = json!(2);
        let r = run(system, lib, &top);
        assert_eq!(r.json["results"].as_array().unwrap().len(), 2);
        assert_eq!(r.json["statistics"]["local_runs"], 4);
    }

    #[test]
    fn mixed_hammer_searches_glasses_and_shapes_deterministically() {
        let (system, lib, mut settings) = doublet();
        settings["algorithm"] = json!("hammer");
        settings["budget"] = json!({"max_combinations":25});
        settings["hammer"] = json!({"rounds":10,"batch":4,"swap_probability":1.0});
        let a = run(system.clone(), lib.clone(), &settings);
        let b = run(system, lib, &settings);
        assert_eq!(a.json, b.json);
        assert_eq!(a.json["results"][0]["glasses"][0]["name"], "C_GOOD");
        assert_eq!(a.json["results"][0]["glasses"][1]["name"], "F_GOOD");
        assert!(a.json["results"][0]["merit"].as_f64().unwrap() < 1e-12);
        assert!(a.json["statistics"]["local_runs"].as_u64().unwrap() <= 25);
        assert!(
            a.json["statistics"]["local_runs"].as_u64()
                > a.json["statistics"]["combinations"].as_u64()
        );
    }
    #[test]
    fn stop_keeps_ranked_results_cancel_discards() {
        use crate::scheduler::JobState;
        use crate::session::AnalysisRequest;
        use crate::{Engine, EngineConfig};
        use std::time::Instant;
        for cancel in [false, true] {
            let engine = Engine::new(EngineConfig {
                worker_threads: 1,
                ..Default::default()
            });
            let mut session = engine.new_session();
            let (system, lib, mut settings) = doublet();
            session.add_catalog(lib.catalogs()[1].as_ref().clone(), None);
            session.set_system(system).unwrap();
            settings["algorithm"] = json!("hammer");
            settings["budget"] = json!({"max_combinations":100_000});
            settings["hammer"] = json!({"rounds":100_000,"stall_rounds":100_000,"batch":2});
            let id = session
                .submit_analysis(&AnalysisRequest::new("glass_substitution", settings))
                .unwrap();
            let t = Instant::now();
            let mut fraction = 0.0;
            loop {
                let status = engine.job_status(id).unwrap();
                assert!(
                    !status.state.is_terminal(),
                    "ended before interruption: {status:?}"
                );
                assert!(status.progress >= fraction);
                fraction = status.progress;
                if status.stage.starts_with("glass_substitution: combinations") {
                    assert!(status.stage.contains("glasses [{"));
                    break;
                }
                assert!(t.elapsed() < Duration::from_secs(30));
                std::thread::sleep(Duration::from_millis(1));
            }
            if cancel {
                engine.cancel_job(id).unwrap();
            } else {
                engine.stop_job(id).unwrap();
            }
            let state = engine.wait_job(id, Some(Duration::from_secs(30))).unwrap();
            if cancel {
                assert_eq!(state, JobState::Cancelled);
                assert!(engine.take_result(id).is_err());
            } else {
                assert_eq!(state, JobState::Done);
                let result = engine.take_result(id).unwrap();
                assert_eq!(result.json["status"], "stopped");
                assert!(!result.json["results"].as_array().unwrap().is_empty());
                assert!(result.json["statistics"]["local_runs"].as_u64().unwrap() < 100_000);
            }
        }
    }

    #[test]
    fn schemas_reject_invalid_nested_settings() {
        let (system, lib, settings) = doublet();
        for patch in [
            json!({"budget":{"max_combinations":0}}),
            json!({"local":{"max_iterations":0}}),
            json!({"hammer":{"swap_probability":2}}),
            json!({"material_variables":[{"surface":1,"candidates":{"neerest":2}}]}),
            json!({"material_variables":[{"surface":1},{"surface":1}]}),
            json!({"material_variables":[{"surface":999}]}),
        ] {
            let mut s = settings.clone();
            overlay(&mut s, &patch);
            let input = AnalysisInput::new(
                Snapshot::new(system.clone()).unwrap(),
                Arc::new(lib.clone()),
                BackendInfo::cpu_reference(),
            );
            assert!(
                run_analysis(&GlassSubstitution, &input, &s, vec![], &JobContext::new()).is_err(),
                "accepted {patch}"
            );
        }
    }

    #[test]
    fn gpu_refinements_share_evaluator_and_report_reference_values() {
        let mut devices = vec![Arc::new(GpuDevice::new(
            "gpu:cpu_f32",
            Arc::new(mklight_gpu::CpuF32Backend::new()),
        ))];
        if let Some(metal) = mklight_metal::MetalBackend::new().unwrap() {
            devices.push(Arc::new(GpuDevice::new("gpu:metal", Arc::new(metal))));
        } else {
            eprintln!("SKIP Metal: no device; CPU-f32 emulation still tested");
        }
        for device in devices {
            let input = AnalysisInput::new(
                Snapshot::new(samples::singlet()).unwrap(),
                Arc::new(MaterialLibrary::with_builtin()),
                BackendInfo::cpu_reference(),
            )
            .with_device(Some(device));
            let settings = json!({"backend":"gpu", "hold_efl":false,
            "default_merit":{"fields":[0],"wavelengths":[1],"thickness":null},
            "material_variables":[{"surface":1,"candidates":{"nearest":2}}],
            "budget":{"max_combinations":2},"local":{"max_iterations":2}});
            let r = run_analysis(
                &GlassSubstitution,
                &input,
                &settings,
                vec![],
                &JobContext::new(),
            )
            .unwrap();
            assert_eq!(r.json["statistics"]["combinations"], 2);
            assert_eq!(r.json["backend"]["precision"], "f32_gpu");
            assert_eq!(r.json["backend"]["qualification"]["passed"], true);
            assert_eq!(r.json["backend"]["final_qualification"]["passed"], true);
            assert_eq!(r.json["backend"]["reported_on"], TraceEvaluator::BACKEND);
            assert!(
                r.json["backend"]["gpu_stats"]["dispatches"]
                    .as_u64()
                    .unwrap()
                    > 1
            );
            assert_eq!(r.json["results"].as_array().unwrap().len(), 2);
            super::super::super::schema::validate(
                &GlassSubstitution.result_schema(),
                &r.json,
                true,
            )
            .unwrap();
            eprintln!(
                "glass refinement backend: {}",
                r.json["backend"]["evaluated_on"]
            );
        }
    }

    #[test]
    #[ignore = "Cooke triplet glass-substitution throughput benchmark; run in release mode"]
    fn benchmark_cooke_three_glasses() {
        let settings = json!({"material_variables":[
            {"surface":1,"candidates":{"nearest":3}},
            {"surface":3,"candidates":{"nearest":3}},
            {"surface":5,"candidates":{"nearest":3}}],
            "budget":{"max_combinations":12},"local":{"max_iterations":8,"max_evaluations":500},
            "default_merit":{"fields":[0,2],"wavelengths":[1]},"backend":"cpu","seed":17});
        let r = run(
            samples::cooke_triplet(),
            MaterialLibrary::with_builtin(),
            &settings,
        );
        let count = r.json["statistics"]["local_runs"].as_u64().unwrap();
        assert_eq!(count, 12);
        assert!(!r.json["results"].as_array().unwrap().is_empty());
        eprintln!(
            "Cooke: {} threads, {} ms, {} local runs, {} continuous evaluations, {:.2} combinations/s, best merit {}",
            rayon::current_num_threads(),
            r.run_info.duration_ms,
            count,
            r.json["statistics"]["continuous_evaluations"],
            count as f64 * 1000.0 / r.run_info.duration_ms,
            r.json["results"][0]["merit"]
        );
    }

    #[test]
    fn transmission_limit_survives_continuous_thickness_optimisation() {
        let mut cat = mklight_core::materials::builtin::catalog();
        cat.glasses.get_mut("N-BK7").unwrap().transmittance.push(
            mklight_core::materials::TransmittancePoint {
                um: 0.48,
                tau: 0.9,
                thickness_mm: 10.0,
            },
        );
        let mut lib = MaterialLibrary::new();
        lib.add_catalog(Arc::new(cat));
        let merit = MeritFunctionSpec {
            operands: vec![
                Operand::new(OperandKind::ParamValue {
                    param: ParamPath::Thickness {
                        surface: SurfaceId(1),
                    },
                })
                .target(20.0),
            ],
            ..Default::default()
        };
        let r = run(
            samples::singlet(),
            lib,
            &json!({
                "variables":[VariableSpec::bounded(ParamPath::Thickness { surface:SurfaceId(1) }, 1.0,30.0)],
                "material_variables":[{"surface":1,"candidates":{"nearest":1,"min_transmission":0.9}}],
                "merit":merit,"hold_efl":false,"backend":"cpu","budget":{"max_combinations":1},
            }),
        );
        let best = &r.json["results"][0];
        let thickness = best["system"]["surfaces"][1]["thickness"].as_f64().unwrap();
        assert!((thickness - 10.0).abs() < 1e-6, "{best}");
        assert_eq!(best["feasible"], true);
    }

    #[test]
    fn wall_time_limit_finishes_instead_of_cancelling() {
        let (system, lib, mut settings) = doublet();
        settings["algorithm"] = json!("hammer");
        settings["budget"] = json!({"max_combinations":100_000, "max_seconds":1e-9});
        settings["hammer"] = json!({"rounds":100_000, "stall_rounds":100_000});
        let result = run(system, lib, &settings);
        assert_eq!(result.json["status"], "finished");
        assert_eq!(result.json["termination"], "time_limit");
        assert_eq!(result.json["stopped"], false);
    }
}
