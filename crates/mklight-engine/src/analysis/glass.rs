//! The `glass_substitution` job: tabu search (or mixed glass/shape basin
//! hopping) over catalogue glasses, with an LM re-optimisation of the
//! continuous variables for every combination. Shares `optimize`'s solved
//! start and merit preparation, backend and Stop machinery; every
//! combination is re-optimised from the substituted design with its solves
//! applied, as a solve of its own on a device.

use super::*;
use mklight_core::MkResult;
use mklight_optim::discrete::{DiscreteSettings, LocallyRefined, NearestChoices, tabu_search};
use mklight_optim::glass::{
    CandidateSet, DISTANCE_WEIGHTS, GlassChoice, MaterialVariable, model_variables,
};
use mklight_optim::mixed::{MixedHammerSettings, RefinedSearch, mixed_hammer};
use rayon::prelude::*;

/// Nearest glasses (by [`DISTANCE_WEIGHTS`]) a search move can swap to, and
/// sampled simultaneous two-glass swaps per tabu iteration or hammer round.
const NEIGHBOURS: usize = 4;
const PAIR_MOVES: usize = 4;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Settings {
    material_variables: Vec<MaterialVariable>,
    variables: Vec<VariableSpec>,
    hold_efl: bool,
    merit: Value,
    default_merit: Value,
    operands: Vec<Operand>,
    constraints: Vec<Operand>,
    algorithm: String,
    max_combinations: usize,
    max_seconds: Option<f64>,
    local_iterations: usize,
    top_n: usize,
    fd_scheme: FdScheme,
    fd_relative_step: Option<f64>,
    seed: u64,
    backend: BackendChoice,
    feasibility_tolerance: f64,
}

/// The input prescription (its solves applied) and candidate glasses,
/// shared by the parallel local solves. Every assignment is built from
/// `base`.
struct LensSpace {
    base: OpticalSystem,
    variables: Vec<VariableSpec>,
    materials: Vec<MaterialVariable>,
    candidates: Vec<Vec<GlassChoice>>,
    indices: Vec<usize>,
    merit: MeritFunctionSpec,
    registry: OperandRegistry,
    /// Resolves the materials of a substituted design's solves.
    library: Arc<MaterialLibrary>,
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
                // Keep the transmission filter met while thickness changes,
                // in every configuration.
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
        // A substituted design is the start of its own re-optimisation: its
        // solves are applied to it, as `optimize` applies them to its start,
        // and the re-optimisation does not re-solve.
        if !system.solves.is_empty() {
            system = mklight_trace::apply_solves_with(
                &system,
                &self.library,
                &Default::default(),
                &JobContext::new(),
            )?
            .0;
        }
        OpticalProblem::new(
            system,
            self.variables.clone(),
            spec,
            evaluator,
            &self.registry,
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
    fn result_version(&self) -> u32 {
        2
    }
    fn name(&self) -> &str {
        "glass_substitution"
    }
    fn supports_stop(&self) -> bool {
        true
    }
    fn title(&self) -> &str {
        "Glass substitution"
    }
    fn category(&self) -> &str {
        "optimization"
    }
    fn description(&self) -> &str {
        "Replaces the glasses of chosen lenses with real catalogue glasses: searches the \
         combinations (tabu search, or basin hopping over glasses and shape together), \
         re-optimises the continuous variables by LM for every combination, and returns the \
         best combinations ranked, each with its complete optimised system. With no material \
         variables, every model glass is snapped to its nearest real glasses."
    }

    fn settings_schema(&self) -> Value {
        use super::super::schema::*;
        let mut schema = Optimize.settings_schema();
        let props = schema["properties"].as_object_mut().unwrap();
        for key in ["algorithm", "options", "max_iterations"] {
            props.remove(key);
        }
        let mut candidates = settings_from_default::<CandidateSet>(json!({
            "catalogs": {"items": {"type": "string"}, "title": "Catalogues",
                "description": "Loaded catalogues to draw from (case-insensitive). Empty: every loaded catalogue but BUILTIN, or BUILTIN when it is the only one."},
            "statuses": {"items": {"enum": ["standard", "preferred", "obsolete", "special", "melt"]}, "title": "Glass status",
                "description": "Allowed AGF status codes (0 standard, 1 preferred, 2 obsolete, 3 special, 4 melt). Glasses flagged exclude-substitution are never used."},
            "window": {"type": ["object", "null"], "title": "Window",
                "properties": {"nd": {"type": "number", "minimum": 0}, "vd": {"type": "number", "minimum": 0}},
                "required": ["nd", "vd"],
                "description": "Half-widths {nd, vd} around the current glass including its offsets. Replacements clear offsets. null: no window."},
            "nearest": {"type": ["integer", "null"], "minimum": 1, "title": "Nearest",
                "description": "Keep the K glasses nearest the current one in (nd, Vd/100, delta PgF), squared-distance weights 1, 1, 100. null: no limit."},
            "include": {"items": {"type": "string"}, "title": "Include",
                "description": "Names or catalog:name kept beyond the window and nearest limits (the other filters still apply)."},
            "exclude": {"items": {"type": "string"}, "title": "Exclude",
                "description": "Names or catalog:name never used."},
            "min_transmission": {"type": ["number", "null"], "minimum": 0, "maximum": 1, "title": "Minimum transmission",
                "description": "Internal transmission at the shortest system wavelength through the element's centre thickness, also held as a thickness limit during re-optimisation. Glasses without data pass."},
            "max_relative_cost": {"type": ["number", "null"], "minimum": 0, "title": "Maximum relative cost",
                "description": "Glasses without data pass."},
            "max_climate_class": {"type": ["number", "null"], "minimum": 0, "title": "Maximum climate class",
                "description": "AGF climate resistance (CR). Glasses without data pass."},
            "max_stain_class": {"type": ["number", "null"], "minimum": 0, "title": "Maximum stain class",
                "description": "AGF stain resistance (FR). Glasses without data pass."}
        }));
        candidates["default"] = json!(CandidateSet::default());
        let material = obj_opt(
            &[
                (
                    "surface",
                    int("Stable id of the surface in front of the glass."),
                ),
                ("candidates", candidates),
            ],
            &["candidates"],
        );
        props.insert("material_variables".into(), json!({"type": "array", "items": material, "default": [],
            "title": "Glasses to substitute",
            "description": "One entry per glass that may change; a cemented doublet is two entries. Empty: every model glass, with the default candidates."}));
        props.insert("algorithm".into(), json!({"type": "string", "enum": ["tabu", "hammer"], "default": "tabu",
            "title": "Search",
            "description": "tabu: search glass combinations, every one re-optimised from the input shape. hammer: basin hopping over glasses and shape together, to escape local minima."}));
        props.insert("max_combinations".into(), json!({"type": "integer", "minimum": 1, "default": 100,
            "title": "Combinations",
            "description": "Local re-optimisations (tabu: distinct glass combinations; hammer: also the same glasses from another shape)."}));
        props.insert("max_seconds".into(), json!({"type": ["number", "null"], "exclusiveMinimum": 0, "default": null,
            "title": "Time limit", "unit": "s",
            "description": "Search wall-time limit, keeping the results so far (termination time_limit; not reproducible). null: none."}));
        props.insert(
            "local_iterations".into(),
            json!({"type": "integer", "minimum": 1, "default": 100,
            "title": "LM iterations", "description": "LM iterations per re-optimisation."}),
        );
        props.insert(
            "top_n".into(),
            json!({"type": "integer", "minimum": 1, "default": 10,
            "title": "Results", "description": "Best glass combinations returned."}),
        );
        props["hold_efl"]["description"] = json!(
            "Hold the EFL at its start value, as a constraint, unless the merit already has an efl operand."
        );
        props["feasibility_tolerance"]["description"] = json!(
            "Total normalised constraint violation regarded as feasible (ranking, results[].feasible)."
        );
        schema
    }

    fn result_schema(&self) -> Value {
        use super::super::schema::*;
        let glass = [
            ("catalog", string("")),
            ("name", string("")),
            ("nd", num("", "d-line index from the dispersion formula.")),
            ("vd", num("", "Abbe number.")),
            ("dpgf", num("", "Partial-dispersion anomaly delta PgF.")),
        ];
        let mut per_surface = vec![("surface", int("Stable surface id."))];
        per_surface.extend(glass.iter().cloned());
        let mut candidate = glass.to_vec();
        candidate.push((
            "max_thickness_mm",
            nullable(num(
                "mm",
                "Thickness limit from min_transmission; null without one.",
            )),
        ));
        result(
            &[
                (
                    "status",
                    enumeration(&["finished", "stopped"], "Cancel discards the result."),
                ),
                ("stopped", boolean("Stopped; the results so far are kept.")),
                (
                    "termination",
                    string("Search termination; time_limit when max_seconds expired."),
                ),
                ("algorithm", enumeration(&["tabu", "hammer"], "")),
                ("starting_merit", nf()),
                ("starting_violation", nf()),
                (
                    "results",
                    arr(
                        obj(&[
                            ("rank", int("1 = best.")),
                            ("merit", nf_u("", "CPU f64 reference merit.")),
                            ("violation", nf()),
                            ("feasible", boolean("")),
                            (
                                "glasses",
                                arr(
                                    obj(&per_surface),
                                    "One per material variable, in settings order.",
                                ),
                            ),
                            (
                                "variables",
                                arr(
                                    obj(&[
                                        ("param", opaque("ParamPath.")),
                                        ("start", nf()),
                                        ("value", nf()),
                                        ("change", nf()),
                                    ]),
                                    "Continuous variables against the input system.",
                                ),
                            ),
                            (
                                "system",
                                opaque(
                                    "The complete optimised system, to preview or apply as one edit.",
                                ),
                            ),
                            ("system_hash", string("")),
                            (
                                "local_termination",
                                string(
                                    "LM termination; cancelled for a solve cut short by Stop or the time limit.",
                                ),
                            ),
                        ]),
                        "Up to top_n distinct glass combinations: feasible first, then by merit (infeasible by violation).",
                    ),
                ),
                (
                    "candidate_sets",
                    arr(
                        obj(&[
                            ("surface", int("")),
                            ("glasses", arr(obj(&candidate), "Nearest first.")),
                        ]),
                        "The resolved candidates of each material variable.",
                    ),
                ),
                (
                    "statistics",
                    obj(&[
                        (
                            "combinations",
                            int("Distinct glass combinations re-optimised."),
                        ),
                        ("local_runs", int("LM re-optimisations.")),
                        (
                            "continuous_evaluations",
                            int("Merit evaluations of the re-optimisations."),
                        ),
                        ("failed", int("Re-optimisations without a usable point.")),
                    ]),
                ),
                (
                    "history",
                    arr(
                        obj(&[
                            ("combination", int("Local-run count.")),
                            ("evaluations", int("Continuous evaluations so far.")),
                            ("merit", nf()),
                            ("best_merit", nf()),
                            ("violation", nf()),
                        ]),
                        "One row per re-optimisation.",
                    ),
                ),
                ("backend", evaluation_backend_schema()),
                (
                    "merit_spec",
                    opaque("The merit; transmission thickness limits are added per combination."),
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
        // Also checks the nested candidate sets.
        validate_settings(&self.settings_schema(), settings)?;
        let mut s: Settings = parse(settings, "settings")?;
        let time_limit = s
            .max_seconds
            .map(Duration::try_from_secs_f64)
            .transpose()
            .map_err(invalid)?;
        let stop = input.stop.clone();
        let start = SolvedStart::new(input, &ctx.subrange(0.0, 0.0))?;
        let variables = start.parameters(
            "variables",
            std::mem::take(&mut s.variables),
            auto_variables,
            |v| &v.param,
        )?;
        let system = &start.system;
        if s.material_variables.is_empty() {
            s.material_variables = model_variables(system);
            if s.material_variables.is_empty() {
                return Err(invalid(
                    "'material_variables': none given, and the system has no model glasses",
                ));
            }
        }
        let mut seen = std::collections::BTreeSet::new();
        if !s.material_variables.iter().all(|m| seen.insert(m.surface)) {
            return Err(invalid("'material_variables': a surface is listed twice"));
        }
        // The glass of a substituted surface belongs to the search alone.
        let of_substituted_glass = |p: &ParamPath| {
            matches!(
                p,
                ParamPath::ModelNd { .. }
                    | ParamPath::ModelVd { .. }
                    | ParamPath::ModelDpgf { .. }
                    | ParamPath::MaterialDnD { .. }
                    | ParamPath::MaterialDvD { .. }
            ) && p.surface().is_some_and(|id| seen.contains(&id))
        };
        if variables.iter().any(|v| of_substituted_glass(&v.param)) {
            return Err(invalid(
                "a substituted surface cannot also have continuous material variables",
            ));
        }
        if start.solved.iter().any(of_substituted_glass) {
            return Err(invalid(
                "'material_variables': a solve controls the glass of a substituted surface",
            ));
        }
        let candidates = s
            .material_variables
            .iter()
            .map(|v| v.candidates.resolve(system, v.surface, &input.library))
            .collect::<MkResult<Vec<_>>>()?;

        // The merit, as `optimize` builds it for LM.
        let m = PreparedMerit::new(
            input,
            start,
            MeritSettings {
                variables,
                merit: s.merit.clone(),
                default_merit: patched(
                    &DefaultMeritSpec::default(),
                    &s.default_merit,
                    "default_merit",
                )?,
                relax_boundaries: true,
                operands: s.operands,
                constraints: s.constraints,
                hold_efl: s.hold_efl,
                derivative_free: None,
            },
            &ctx.subrange(0.0, 0.0),
        )?;
        let reference = m.problem(m.cpu.clone())?;
        let PreparedMerit {
            snapshot,
            variables,
            cpu,
            reg,
            spec,
            mut warnings,
            ..
        } = m;
        let system = snapshot.system().clone();
        let x0 = reference.start_point()?;
        let start = reference_score(&reference, &x0, ctx)?;
        let lsq = LsqSettings {
            max_iterations: s.local_iterations,
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
        let jacobian_batch = x0.len()
            * if s.fd_scheme == FdScheme::Central {
                2
            } else {
                1
            };
        let work = Workload::new(&cpu.trace, &system, &spec, jacobian_batch);
        let mut ev = Evaluation::new(s.backend, input, &snapshot, work, ctx)?;
        let lsq = LsqSettings {
            speculative_jacobian: ev.speculative(),
            ..lsq
        };
        let space = Arc::new(LensSpace {
            indices: s
                .material_variables
                .iter()
                .map(|v| system.surface_index(v.surface).unwrap())
                .collect(),
            base: system,
            variables,
            materials: s.material_variables,
            candidates,
            merit: spec,
            registry: reg,
            library: input.library.clone(),
        });
        let coords: Vec<_> = space
            .candidates
            .iter()
            .map(|g| g.iter().map(GlassChoice::coordinates).collect())
            .collect();
        let mut neighbours =
            NearestChoices::from_coordinates(&coords, &DISTANCE_WEIGHTS, NEIGHBOURS);
        neighbours.pair_moves = PAIR_MOVES;
        // Candidates are nearest first, so 0 is the current glass (or, for a
        // model glass, the nearest real one).
        let initial = vec![0; space.materials.len()];
        let build_space = space.clone();
        let gpu = ev.active_gpu().cloned();
        let build_cpu = cpu.clone();
        let start_x = x0.clone();
        let local = LocallyRefined {
            choices: space.candidates.iter().map(Vec::len).collect(),
            settings: lsq,
            // Every combination starts from the input shape, and its solve
            // is one of its own on the device: what it learns about its rays
            // stays with it.
            build: Box::new(move |a| {
                let p: Box<dyn Problem + Send> = match &gpu {
                    Some(g) => Box::new(build_space.problem(a, g.for_solve())?),
                    None => Box::new(build_space.problem(a, build_cpu.clone())?),
                };
                Ok((p, start_x.clone()))
            }),
        };

        let ((run, archive), why) = interruptible_for(ctx, &stop, time_limit, |inner| {
            let mut mon = Monitor::new(JobContext::with(inner.cancel.clone(), Progress::default()));
            mon.feasibility_tol = s.feasibility_tolerance;
            let best = mon.best_assignment.clone();
            let progress = inner.clone();
            let labels = space.clone();
            let limit = s.max_combinations;
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
                    &MixedHammerSettings::default(),
                    s.max_combinations,
                    s.seed,
                    inner,
                )
            } else {
                let settings = DiscreteSettings {
                    max_evaluations: s.max_combinations,
                    seed: s.seed,
                    ..Default::default()
                };
                tabu_search(&search, &neighbours, &initial, &settings, &search.monitor)
                    .map(|r| r.termination)
            };
            (run, search.archive())
        });
        ctx.check()?; // Cancel discards; Stop and the time limit cancelled only the search.
        let termination = match run {
            Ok(t) => t,
            Err(MkError::Cancelled) if why != Interrupt::None => Termination::Cancelled,
            Err(e) => return Err(e.into()),
        };

        // Rank every combination's best answer on the CPU reference.
        ctx.report(
            ctx.progress.fraction(),
            "ranking glass combinations on the CPU reference",
        );
        let entries: Vec<_> = archive.results.iter().collect();
        let scored: Vec<_> = entries
            .par_iter()
            .map(|&(assignment, r)| -> EngineResult<_> {
                ctx.check()?;
                let p = space.problem(assignment, cpu.clone())?;
                let value = reference_score(&p, &r.x, ctx)?;
                let usable = value.status.is_usable()
                    && value.merit.is_finite()
                    && value.violation.is_finite();
                Ok(if usable {
                    Some((assignment, p.apply(&r.x)?, &r.x, value, r.termination))
                } else {
                    None
                })
            })
            .collect();
        let mut ranked = Vec::new();
        for row in scored {
            ranked.extend(row?);
        }
        // `scored` keeps the assignment order of `archive.results`, and the
        // sort is stable: ties keep that order.
        ranked.sort_by(|a, b| a.3.compare(&b.3, s.feasibility_tolerance));
        ranked.truncate(s.top_n);
        let mut results = Vec::new();
        for (rank, (a, system, x, value, local_termination)) in ranked.into_iter().enumerate() {
            let snapshot = Snapshot::new(system)?;
            let variables: Vec<_> = space
                .variables
                .iter()
                .zip(x.iter().zip(&x0))
                .map(|(v, (&x, &start))| {
                    json!({"param": v.param, "start": json_nf(start), "value": json_nf(x), "change": json_nf(x - start)})
                })
                .collect();
            results.push(json!({"rank": rank + 1, "merit": json_nf(value.merit), "violation": json_nf(value.violation),
                "feasible": value.violation <= s.feasibility_tolerance, "glasses": space.glasses(a),
                "variables": variables, "system": snapshot.system(), "system_hash": snapshot.hash().to_hex(),
                "local_termination": local_termination}));
        }
        // Every published score is on the reference.
        ev.finish(input, s.backend, Reported::Reference)?;
        warnings.extend(ev.warnings);
        if results.is_empty() {
            warnings
                .push("no usable glass combination was retained within the search budget".into());
        }
        let stopped = why == Interrupt::Stopped;
        let json = json!({
            "status": job_status(stopped, None), "stopped": stopped,
            "termination": if why == Interrupt::TimeLimit { json!("time_limit") } else { json!(termination) },
            "algorithm": s.algorithm, "starting_merit": json_nf(start.merit), "starting_violation": json_nf(start.violation),
            "results": results,
            "candidate_sets": space.materials.iter().zip(&space.candidates)
                .map(|(v, c)| json!({"surface": v.surface, "glasses": c})).collect::<Vec<_>>(),
            "statistics": {"combinations": archive.assignments.len(), "local_runs": archive.local_runs,
                "continuous_evaluations": archive.evaluations, "failed": archive.failed},
            "history": archive.history.iter().map(|h| json!({"combination": h.iteration, "evaluations": h.evaluations,
                "merit": json_nf(h.merit), "best_merit": json_nf(h.best_merit), "violation": json_nf(h.violation)})).collect::<Vec<_>>(),
            "backend": ev.report, "merit_spec": space.merit,
        });
        Ok(AnalysisOutput {
            json,
            buffers: Vec::new(),
            warnings,
        })
    }
}

/// The CPU f64 reference merit and violation at `x`.
fn reference_score(
    p: &OpticalProblem<BuiltinEvaluator>,
    x: &[f64],
    ctx: &JobContext,
) -> EngineResult<ScalarEval> {
    let r = reference_values(p, &[x], ctx)?;
    Ok(ScalarEval {
        merit: r.merit[0],
        violation: r.violation[0],
        status: r.status[0],
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{AnalysisResult, run_analysis};
    use crate::backend::BackendInfo;
    use crate::scheduler::JobState;
    use crate::session::AnalysisRequest;
    use crate::{Engine, EngineConfig};
    use mklight_core::materials::{GlassCatalog, MaterialLibrary, agf, lines};
    use mklight_core::platform::Instant;
    use mklight_core::{MaterialRef, SurfaceId, SurfaceShape, samples};

    const PAIR: [(&str, f64, f64); 4] = [
        ("C_GOOD", 1.52, 60.0),
        ("C_BAD", 1.52, 30.0),
        ("F_GOOD", 1.62, 30.0),
        ("F_BAD", 1.62, 60.0),
    ];

    /// A cemented doublet on the bad pair (crown and flint with swapped Abbe
    /// numbers), bending variables, and an EFL + F–C focal shift merit. Only
    /// C_GOOD / F_GOOD can achromatise at positive power within the bounds.
    fn doublet() -> (OpticalSystem, MaterialLibrary, Value) {
        let mut cat = GlassCatalog {
            name: "PAIR".into(),
            ..Default::default()
        };
        for (name, nd, vd) in PAIR {
            cat.glasses
                .insert(name.into(), agf::model_glass(name, nd, vd, 0.0).unwrap());
        }
        // Through the AGF reader, as a loaded catalogue would be.
        let text = agf::write_agf(&agf::AgfCatalog::from_catalog(cat)).0;
        let mut lib = MaterialLibrary::with_builtin();
        lib.add_catalog(Arc::new(agf::parse_agf("PAIR", &text).unwrap()));
        let mut system = samples::achromatic_doublet();
        system.surfaces[1].material = MaterialRef::glass("C_BAD");
        system.surfaces[2].material = MaterialRef::glass("F_BAD");
        system.surfaces[1].thickness = 3.0;
        system.surfaces[2].thickness = 2.0;
        system.surfaces[1].shape = SurfaceShape::sphere(50.0);
        system.surfaces[2].shape = SurfaceShape::sphere(-60.0);
        system.surfaces[3].shape = SurfaceShape::Plane;
        let curvature = |id, lo, hi| {
            VariableSpec::bounded(
                ParamPath::Curvature {
                    surface: SurfaceId(id),
                },
                lo,
                hi,
            )
        };
        let merit = MeritFunctionSpec {
            operands: vec![
                Operand::new(OperandKind::Efl).target(100.0),
                Operand::new(OperandKind::AxialColour { short: 0, long: 2 }).weight(10.0),
            ],
            ..Default::default()
        };
        let settings = json!({"variables": [curvature(1, 0.005, 0.06), curvature(2, -0.04, -0.001)],
        "merit": merit, "hold_efl": false, "backend": "cpu", "seed": 31,
        "material_variables": [
            {"surface": 1, "candidates": {"exclude": ["F_GOOD", "F_BAD"]}},
            {"surface": 2, "candidates": {"exclude": ["C_GOOD", "C_BAD"]}}
        ]});
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

    fn names(row: &Value) -> Vec<&str> {
        row["glasses"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g["name"].as_str().unwrap())
            .collect()
    }

    #[test]
    fn cemented_doublet_finds_achromatic_pair_after_rebending() {
        let (system, lib, settings) = doublet();
        let r = run(system, lib, &settings);
        let rows = r.json["results"].as_array().unwrap();
        assert_eq!(rows.len(), 4);
        let best = &rows[0];
        assert_eq!(names(best), ["C_GOOD", "F_GOOD"]);
        assert!(best["merit"].as_f64().unwrap() < 1e-12);
        assert!(r.json["starting_merit"].as_f64().unwrap() > 1.0);
        assert!(
            best["variables"]
                .as_array()
                .unwrap()
                .iter()
                .any(|v| v["change"].as_f64().unwrap().abs() > 1e-4),
            "re-bent"
        );
        // F–C focal shift from an independent paraxial y-nu trace of the
        // returned system.
        let system: OpticalSystem = serde_json::from_value(best["system"].clone()).unwrap();
        let glass = |i: usize| {
            agf::model_glass("", 1.52 + 0.1 * (i - 1) as f64, 90.0 - 30.0 * i as f64, 0.0).unwrap()
        };
        let bfl = |wavelength| {
            let (mut y, mut nu, mut n) = (1.0, 0.0, 1.0);
            for i in 1..=3 {
                let sf = &system.surfaces[i];
                let next = if i == 3 {
                    1.0
                } else {
                    glass(i).catalog_index(wavelength)
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
        assert!((bfl(lines::F) - bfl(lines::C)).abs() < 1e-6);
        super::super::super::schema::validate(&GlassSubstitution.result_schema(), &r.json, true)
            .unwrap();
        assert_eq!(r.run_info.seed, Some(31));
    }

    #[test]
    fn ranking_is_sorted_unique_and_deterministic() {
        let (system, lib, settings) = doublet();
        let run_in = |threads, settings: &Value| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| run(system.clone(), lib.clone(), settings))
        };
        let r = run_in(1, &settings);
        assert_eq!(r.json, run_in(4, &settings).json);
        let rows = r.json["results"].as_array().unwrap();
        let keys: std::collections::BTreeSet<_> = rows.iter().map(names).collect();
        assert_eq!(keys.len(), rows.len());
        assert!(
            rows.windows(2)
                .all(|w| w[0]["merit"].as_f64() <= w[1]["merit"].as_f64())
        );
        let mut top = settings;
        top["top_n"] = json!(2);
        let r2 = run_in(2, &top);
        assert_eq!(r2.json["results"].as_array().unwrap()[..], rows[..2]);
    }

    #[test]
    fn mixed_hammer_searches_glasses_and_shapes_deterministically() {
        let (system, lib, mut settings) = doublet();
        settings["algorithm"] = json!("hammer");
        settings["max_combinations"] = json!(25);
        let a = run(system.clone(), lib.clone(), &settings);
        assert_eq!(a.json, run(system, lib, &settings).json);
        assert_eq!(names(&a.json["results"][0]), ["C_GOOD", "F_GOOD"]);
        assert!(a.json["results"][0]["merit"].as_f64().unwrap() < 1e-12);
        let stats = &a.json["statistics"];
        assert!(stats["local_runs"].as_u64().unwrap() <= 25);
        // Combinations are revisited from other shapes.
        assert!(stats["local_runs"].as_u64() > stats["combinations"].as_u64());
    }

    #[test]
    fn defaults_need_only_the_glasses_to_change() {
        let r = run(
            samples::achromatic_doublet(),
            MaterialLibrary::with_builtin(),
            &json!({"material_variables": [{"surface": 1}, {"surface": 2}], "max_combinations": 6,
                "default_merit": {"fields": [0], "wavelengths": [0, 1, 2]}}),
        );
        let rows = r.json["results"].as_array().unwrap();
        assert!(!rows.is_empty());
        assert!(rows[0]["feasible"].as_bool().unwrap());
        assert_eq!(r.json["candidate_sets"][0]["glasses"][0]["name"], "N-BK7");
        assert_eq!(r.json["candidate_sets"][1]["glasses"][0]["name"], "F2");
        assert_eq!(
            r.json["candidate_sets"][0]["glasses"]
                .as_array()
                .unwrap()
                .len(),
            10
        );
    }

    /// Regression: a re-optimisation that ends on its iteration budget short
    /// of feasibility must not displace a better (feasible) point it saw,
    /// such as the input design itself.
    #[test]
    fn short_refinements_keep_the_best_point_seen() {
        let r = run(
            samples::cooke_triplet(),
            MaterialLibrary::with_builtin(),
            &json!({"material_variables": [{"surface": 1, "candidates": {"nearest": 1}}],
                "local_iterations": 2, "default_merit": {"fields": [0], "wavelengths": [1]},
                "backend": "cpu"}),
        );
        let best = &r.json["results"][0];
        assert_eq!(r.json["starting_violation"], 0.0);
        assert_eq!(best["feasible"], true, "{best}");
        assert!(best["merit"].as_f64() <= r.json["starting_merit"].as_f64());
    }

    #[test]
    fn stop_keeps_ranked_results_cancel_discards() {
        for cancel in [false, true] {
            let engine = Engine::new(EngineConfig {
                worker_threads: 1,
                ..Default::default()
            });
            let mut session = engine.new_session();
            let (system, lib, mut settings) = doublet();
            session.add_catalog(lib.catalogs()[1].as_ref().clone(), None);
            session.set_system(system).unwrap();
            let builtin = json!({"catalogs": ["BUILTIN"], "nearest": null});
            settings["material_variables"] = json!([{"surface": 1, "candidates": builtin}, {"surface": 2, "candidates": builtin}]);
            settings["max_combinations"] = json!(100_000);
            let id = session
                .submit_analysis(&AnalysisRequest::new("glass_substitution", settings))
                .unwrap();
            let t = Instant::now();
            loop {
                let status = engine.job_status(id).unwrap();
                assert!(!status.state.is_terminal(), "ended early: {status:?}");
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
            }
        }
    }

    #[test]
    fn wall_time_limit_finishes_instead_of_cancelling() {
        let (system, lib, mut settings) = doublet();
        settings["max_combinations"] = json!(100_000);
        settings["max_seconds"] = json!(1e-9);
        let result = run(system, lib, &settings);
        assert_eq!(result.json["status"], "finished");
        assert_eq!(result.json["termination"], "time_limit");
    }

    #[test]
    fn invalid_nested_settings_are_rejected() {
        let (system, lib, settings) = doublet();
        for patch in [
            json!({"local_iterations": 0}),
            json!({"material_variables": [{"surface": 1, "candidates": {"neerest": 2}}]}),
            json!({"material_variables": [{"surface": 1, "candidates": {"min_transmission": 2}}]}),
            json!({"material_variables": [{"surface": 1}, {"surface": 1}]}),
            json!({"material_variables": [{"surface": 999}]}),
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
        let thickness = ParamPath::Thickness {
            surface: SurfaceId(1),
        };
        let merit = MeritFunctionSpec {
            operands: vec![
                Operand::new(OperandKind::ParamValue {
                    param: thickness.clone(),
                })
                .target(20.0),
            ],
            ..Default::default()
        };
        let r = run(
            samples::singlet(),
            lib,
            &json!({
                "variables": [VariableSpec::bounded(thickness, 1.0, 30.0)],
                "material_variables": [{"surface": 1, "candidates": {"nearest": 1, "min_transmission": 0.9}}],
                "merit": merit, "hold_efl": false, "backend": "cpu", "max_combinations": 1,
            }),
        );
        // tau(10 mm) = 0.9 is the limit.
        let best = &r.json["results"][0];
        let thickness = best["system"]["surfaces"][1]["thickness"].as_f64().unwrap();
        assert!((thickness - 10.0).abs() < 1e-6, "{best}");
        assert_eq!(best["feasible"], true);
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
            let settings = json!({"backend": "gpu", "hold_efl": false,
                "default_merit": {"fields": [0], "wavelengths": [1], "thickness": null},
                "material_variables": [{"surface": 1, "candidates": {"nearest": 2}}],
                "max_combinations": 2, "local_iterations": 2});
            let r = run_analysis(
                &GlassSubstitution,
                &input,
                &settings,
                vec![],
                &JobContext::new(),
            )
            .unwrap();
            let backend = &r.json["backend"];
            assert_eq!(r.json["statistics"]["combinations"], 2);
            assert_eq!(backend["precision"], "f32_gpu", "{backend}");
            assert_eq!(backend["reported_on"], TraceEvaluator::BACKEND);
            assert!(backend["gpu_stats"]["gpu_candidates"].as_u64().unwrap() > 0);
            if r.backend.used == "gpu:metal" {
                assert!(backend["gpu_stats"]["dispatches"].as_u64().unwrap() > 0);
            }
            assert_eq!(r.json["results"].as_array().unwrap().len(), 2);
        }
    }

    #[test]
    #[ignore = "Cooke triplet glass-substitution throughput benchmark; run in release mode"]
    fn benchmark_cooke_three_glasses() {
        let settings = json!({"material_variables": [
            {"surface": 1, "candidates": {"nearest": 3}},
            {"surface": 3, "candidates": {"nearest": 3}},
            {"surface": 5, "candidates": {"nearest": 3}}],
            "max_combinations": 12, "local_iterations": 8,
            "default_merit": {"fields": [0, 2], "wavelengths": [1]}, "backend": "cpu", "seed": 17});
        let r = run(
            samples::cooke_triplet(),
            MaterialLibrary::with_builtin(),
            &settings,
        );
        let count = r.json["statistics"]["local_runs"].as_u64().unwrap();
        assert_eq!(count, 12);
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
}
