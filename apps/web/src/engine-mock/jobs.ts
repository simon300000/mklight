// MOCK ENGINE — the design jobs: optimize, merit_evaluation, glass_substitution
// and tolerance. Real (small) searches on the mock optics: they minimise the
// mock's real-ray RMS spot merit, so the results respond to the design, but they
// are not the engine's optimiser. Each run stays within about two seconds and
// yields to the event loop between steps; `stop()` keeps the best so far.
import { EngineError, type Json, type JsonObject } from '../engine/types'
import { CATALOG, canonicalJson, sha256Hex, type Obj } from './model'
import { addBuf, newOut, nf, type Out } from './analyses'
import { asPath, clone, defaultVariables, getParamValue, paramLabel, setParamValue, surfaceOf, type Path } from './params'
import { Lm, buildMerit, evaluateMerit, operandReport, readVariables, type Variable } from './merit'
import { dpgfOf, nearestGlasses, type Candidate } from './glass'

export interface JobCtx {
  report(progress: number, stage: string, detail?: Json): void
  stopped(): boolean
  /** Throws `cancelled` once the job was cancelled. */
  check(): void
  /** Yields to the event loop (and notices a cancel). */
  tick(): Promise<void>
  elapsedMs(): number
  /** Wall-clock budget of the search. */
  budgetMs: number
}

const num = (v: unknown, d: number) => (typeof v === 'number' && Number.isFinite(v) ? v : d)
const backend = (rays: number): JsonObject => ({
  requested: 'auto',
  evaluated_on: 'mock',
  reported_on: 'mock',
  precision: 'f64_reference',
  workload: { batch: 1, rays_per_candidate: rays },
})
const meritStatus: Json = { status: 'ok' }

async function hashOf(system: Obj): Promise<string> {
  return sha256Hex(canonicalJson(system))
}

function variablesFor(s: Obj, system: Obj): Variable[] {
  return readVariables(s.variables, system, defaultVariables(system as JsonObject), (p) => paramLabel(system, p))
}

function variableRows(vars: Variable[], x: number[]): JsonObject[] {
  return vars.map((v, i) => {
    const [lo, hi] = [v.lower, v.upper]
    const eps = 1e-9 * Math.max(Math.abs(x[i]), 1e-3)
    return {
      label: v.label,
      param: v.path as JsonObject,
      start: v.start,
      value: x[i],
      change: x[i] - v.start,
      lower: lo,
      upper: hi,
      at_bound: (lo !== null && x[i] <= lo + eps) || (hi !== null && x[i] >= hi - eps) || (lo === null && v.path.param === 'thickness' && x[i] <= 0.05 + eps),
    }
  })
}

// --- merit_evaluation --------------------------------------------------------

export function meritEvaluation(s: Obj, system: Obj): Out {
  const setup = buildMerit(s, system)
  const e = evaluateMerit(system, setup)
  const rows = operandReport(setup, e, e)
  return newOut({
    merit: nf(e.merit),
    violation: nf(e.violation),
    status: meritStatus,
    operands: rows,
    merit_spec: setup.spec,
    backend: backend(19),
  })
}

// --- optimize ------------------------------------------------------------------

export async function optimize(s: Obj, system: Obj, ctx: JobCtx): Promise<Out> {
  const setup = buildMerit(s, system)
  const vars = variablesFor(s, system)
  const work = clone(system)
  const maxIter = Math.max(0, Math.floor(num(s.max_iterations, 30)))
  const lm = new Lm(work, vars, setup)
  const start = lm.eval
  const startMerit = start.merit
  let best = start
  const history: JsonObject[] = []
  const flat: number[] = []
  const record = (iteration: number, stage: string) => {
    history.push({
      iteration,
      evaluations: lm.evaluations,
      merit: nf(lm.eval.merit),
      best_merit: nf(best.merit),
      violation: nf(lm.eval.violation),
      stage,
    })
    flat.push(iteration, lm.evaluations, lm.eval.merit, best.merit)
  }
  record(0, 'start')
  ctx.report(0, 'Optimizing', { iteration: 0, evaluations: lm.evaluations, merit: startMerit, best_merit: startMerit })

  let status: 'converged' | 'finished' | 'stopped' = 'finished'
  let termination = 'max_iterations'
  let iterations = 0
  if (vars.length === 0) {
    status = 'converged'
    termination = 'no_variables'
  }
  while (vars.length > 0) {
    ctx.check()
    if (ctx.stopped()) {
      status = 'stopped'
      termination = 'cancelled'
      break
    }
    if (iterations >= maxIter) break
    if (ctx.elapsedMs() > ctx.budgetMs) {
      termination = 'time_limit'
      break
    }
    const st = lm.step()
    iterations++
    if (lm.eval.merit < best.merit) best = lm.eval
    record(iterations, 'lm')
    ctx.report(iterations / Math.max(maxIter, 1), 'Optimizing', {
      iteration: iterations,
      evaluations: lm.evaluations,
      merit: lm.eval.merit,
      best_merit: best.merit,
    })
    await ctx.tick()
    if (st.converged) {
      status = 'converged'
      termination = st.accepted ? 'small_merit_change' : 'small_step'
      break
    }
  }
  const final = lm.eval
  const improved = final.merit < startMerit - 1e-15 || final.violation < start.violation - 1e-12
  const out = newOut({
    status,
    stopped: status === 'stopped',
    termination,
    algorithm: String(s.algorithm ?? 'lm'),
    iterations,
    evaluations: lm.evaluations,
    improved,
    improvement: startMerit > 0 ? 1 - final.merit / startMerit : null,
    failed_variables: [],
    multipliers: null,
    start: { merit: nf(startMerit), violation: nf(start.violation), status: meritStatus },
    final: { merit: nf(final.merit), violation: nf(final.violation), feasible: final.violation <= num(s.feasibility_tolerance, 1e-6), status: meritStatus },
    search: { merit: nf(final.merit), violation: nf(final.violation), reference_merit: nf(final.merit), reference_violation: nf(final.violation) },
    variables: variableRows(vars, lm.x),
    operands: operandReport(setup, start, final),
    merit_spec: setup.spec,
    history,
    backend: backend(19),
    system: work as JsonObject,
    system_hash: await hashOf(work),
  })
  addBuf(out, 'merit_history', Float64Array.from(flat), [history.length, 4])
  return out
}

// --- glass_substitution ----------------------------------------------------------

interface GlassVar {
  surface: number
  current: { nd: number; vd: number; dpgf: number }
  candidates: Candidate[]
}

function currentGlass(surf: Obj): { nd: number; vd: number; dpgf: number } | null {
  const m = surf.material as Obj | undefined
  if (!m) return null
  if (m.type === 'model') return { nd: Number(m.nd), vd: Number(m.vd), dpgf: Number(m.dpgf ?? 0) }
  if (m.type === 'catalog') {
    const g = CATALOG[String(m.name).toUpperCase()]
    return g ? { nd: g.nd + Number(m.offset?.dn_d ?? 0), vd: g.vd + Number(m.offset?.dv_d ?? 0), dpgf: dpgfOf(String(m.name)) } : null
  }
  return null
}

export async function glassSubstitution(s: Obj, system: Obj, ctx: JobCtx): Promise<Out> {
  const setup = buildMerit(s, system)
  const surfaces: Obj[] = system.surfaces ?? []
  const requested: Obj[] = Array.isArray(s.material_variables) ? (s.material_variables as Obj[]) : []
  let ids: { id: number; cand: Obj }[] = requested
    .filter((r) => typeof r.surface === 'number')
    .map((r) => ({ id: r.surface as number, cand: (r.candidates ?? {}) as Obj }))
  if (ids.length === 0) {
    const model = surfaces.filter((x) => x.material?.type === 'model')
    // Mock shortcut: with no model glasses, every catalogue glass may change.
    const pool = model.length ? model : surfaces.filter((x) => x.material?.type === 'catalog')
    ids = pool.map((x) => ({ id: x.id as number, cand: {} }))
  }
  const gvars: GlassVar[] = []
  for (const { id, cand } of ids) {
    const surf = surfaceOf(system, { param: 'x', surface: id })
    if (!surf) throw new EngineError('invalidSettings', `material variable: no surface with id ${id}`)
    const cur = currentGlass(surf)
    if (!cur) throw new EngineError('invalidSettings', `material variable: surface ${id} has no glass`)
    const candidates = nearestGlasses(cur, {
      nearest: cand.nearest === null ? null : num(cand.nearest, 10),
      include: (cand.include as string[]) ?? [],
      exclude: (cand.exclude as string[]) ?? [],
      catalogs: (cand.catalogs as string[]) ?? [],
    })
    if (candidates.length === 0) throw new EngineError('invalidSettings', `material variable: no candidate glasses for surface ${id}`)
    gvars.push({ surface: id, current: cur, candidates })
  }
  const vars = variablesFor(s, system)
  const startEval = evaluateMerit(system, setup)
  const maxComb = Math.max(1, Math.floor(num(s.max_combinations, 100)))
  const topN = Math.max(1, Math.floor(num(s.top_n, 10)))
  const localIters = Math.min(Math.max(1, Math.floor(num(s.local_iterations, 3))), 3)

  interface Done {
    combo: number[]
    merit: number
    violation: number
    termination: string
    system: Obj
    x: number[]
  }
  const done: Done[] = []
  const history: JsonObject[] = []
  const visited = new Set<string>()
  let frontier: { combo: number[]; parent: number; cost: number }[] = []
  let evaluations = 0
  let failed = 0
  let best = Infinity
  let termination = 'exhausted'

  const push = (combo: number[], parent: number) => {
    const key = combo.join(',')
    if (visited.has(key)) return
    visited.add(key)
    const cost = combo.reduce((a, j, i) => a + gvars[i].candidates[j].distance, 0)
    frontier.push({ combo, parent, cost })
  }
  if (gvars.length) push(gvars.map(() => 0), startEval.merit)

  const runCombo = (combo: number[]): Done => {
    const w = clone(system)
    combo.forEach((j, i) => {
      const c = gvars[i].candidates[j]
      const surf = surfaceOf(w, { param: 'x', surface: gvars[i].surface })!
      surf.material = { type: 'catalog', catalog: c.catalog === 'MOCK' ? null : c.catalog, name: c.name }
    })
    // The substituted design's variables start from the input shape.
    const lm = new Lm(w, vars, setup)
    let term = 'max_iterations'
    for (let k = 0; k < localIters; k++) {
      const st = lm.step()
      if (st.converged) {
        term = 'small_merit_change'
        break
      }
    }
    evaluations += lm.evaluations
    return { combo, merit: lm.eval.merit, violation: lm.eval.violation, termination: term, system: w, x: lm.x }
  }

  while (gvars.length && frontier.length && done.length < maxComb) {
    ctx.check()
    if (ctx.stopped()) {
      termination = 'cancelled'
      break
    }
    if (ctx.elapsedMs() > ctx.budgetMs) {
      termination = 'time_limit'
      break
    }
    frontier.sort((a, b) => a.parent - b.parent || a.cost - b.cost)
    const next = frontier.shift()!
    const r = runCombo(next.combo)
    if (!Number.isFinite(r.merit) || r.merit > 1e5) failed++
    done.push(r)
    best = Math.min(best, r.merit)
    history.push({ combination: done.length, evaluations, merit: nf(r.merit), best_merit: nf(best), violation: nf(r.violation) })
    // Moves: each glass to its nearest candidates and to its neighbours.
    next.combo.forEach((j, i) => {
      const n = gvars[i].candidates.length
      for (const t of new Set([0, 1, 2, 3, j - 1, j + 1])) if (t >= 0 && t < n && t !== j) push(next.combo.map((v, k) => (k === i ? t : v)), r.merit)
    })
    ctx.report(done.length / maxComb, 'Glass substitution', { combinations: done.length, best_merit: best })
    await ctx.tick()
    if (done.length >= maxComb) termination = 'max_combinations'
  }
  frontier = []

  const feasTol = num(s.feasibility_tolerance, 1e-6)
  const ranked = [...done].sort((a, b) => {
    const fa = a.violation <= feasTol
    const fb = b.violation <= feasTol
    if (fa !== fb) return fa ? -1 : 1
    return fa ? a.merit - b.merit : a.violation - b.violation
  })
  const top = ranked.slice(0, topN)
  const results: JsonObject[] = []
  for (const [k, r] of top.entries()) {
    results.push({
      rank: k + 1,
      feasible: r.violation <= feasTol,
      merit: nf(r.merit),
      violation: nf(r.violation),
      local_termination: r.termination,
      glasses: r.combo.map((j, i) => {
        const c = gvars[i].candidates[j]
        return { surface: gvars[i].surface, catalog: c.catalog, name: c.name, nd: c.nd, vd: c.vd, dpgf: c.dpgf }
      }),
      variables: variableRows(vars, r.x).map((v) => ({ param: v.param, start: v.start, value: v.value, change: v.change })),
      system: r.system as JsonObject,
      system_hash: await hashOf(r.system),
    })
  }
  const stopped = termination === 'cancelled'
  return newOut({
    algorithm: String(s.algorithm ?? 'tabu'),
    status: stopped ? 'stopped' : 'finished',
    stopped,
    termination,
    starting_merit: nf(startEval.merit),
    starting_violation: nf(startEval.violation),
    statistics: { combinations: done.length, continuous_evaluations: evaluations, failed, local_runs: done.length },
    candidate_sets: gvars.map((g) => ({
      surface: g.surface,
      glasses: g.candidates.map((c) => ({ catalog: c.catalog, name: c.name, nd: c.nd, vd: c.vd, dpgf: c.dpgf, max_thickness_mm: null })),
    })),
    results,
    history,
    merit_spec: setup.spec,
    backend: backend(19),
  })
}

// --- tolerance ---------------------------------------------------------------------

function rngOf(seed: number): () => number {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}

interface Tol {
  path: Path
  minus: number
  plus: number
  distribution: string
  label: string
}

function readTolerances(list: unknown, system: Obj): Tol[] {
  const entries: Obj[] = Array.isArray(list) ? (list as Obj[]) : []
  const out: Tol[] = []
  if (entries.length) {
    for (const e of entries) {
      const p = asPath(e.param)
      if (!p) continue
      if (getParamValue(system, p) === null) throw new EngineError('invalidSettings', `tolerance: the system has no parameter ${paramLabel(system, p)}`)
      out.push({ path: p, minus: Math.abs(num(e.minus, 0.05)), plus: Math.abs(num(e.plus, 0.05)), distribution: String(e.distribution ?? 'uniform'), label: typeof e.label === 'string' ? e.label : '' })
    }
    return out
  }
  const surfaces: Obj[] = system.surfaces ?? []
  surfaces.forEach((surf, i) => {
    if (i === 0 || i >= surfaces.length - 1) return
    if (i < surfaces.length - 2 && typeof surf.thickness === 'number' && surf.thickness > 0)
      out.push({ path: { param: 'thickness', surface: surf.id }, minus: 0.05, plus: 0.05, distribution: 'uniform', label: '' })
    const c = getParamValue(system, { param: 'curvature', surface: surf.id })
    if (c !== null && c !== 0 && surf.interaction !== 'dummy')
      out.push({ path: { param: 'curvature', surface: surf.id }, minus: 0.005 * Math.abs(c), plus: 0.005 * Math.abs(c), distribution: 'uniform', label: '' })
  })
  return out
}

function gauss(r: () => number): number {
  return Math.sqrt(-2 * Math.log(Math.max(r(), 1e-12))) * Math.cos(2 * Math.PI * r())
}

export async function tolerance(s: Obj, system: Obj, ctx: JobCtx): Promise<Out> {
  const criterionSpec = s.criterion && typeof s.criterion === 'object' ? (s.criterion as Obj) : null
  const setup = buildMerit({ merit: criterionSpec, hold_efl: false, default_merit: s.default_merit }, system)
  const tols = readTolerances(s.tolerances, system)
  const compSpecs: { path: Path; range: number }[] = []
  for (const c of Array.isArray(s.compensators) ? (s.compensators as Obj[]) : []) {
    const p = asPath(c.param)
    if (!p) continue
    if (getParamValue(system, p) === null) throw new EngineError('invalidSettings', `compensator: the system has no parameter ${paramLabel(system, p)}`)
    compSpecs.push({ path: p, range: Math.abs(num(c.range, 1)) })
  }
  const compVars: Variable[] = compSpecs.map((c) => {
    const v = getParamValue(system, c.path)!
    return {
      path: c.path,
      start: v,
      lower: v - c.range,
      upper: v + c.range,
      fd: 1e-4 * Math.max(Math.abs(v), c.path.param === 'curvature' ? 0.01 : 1),
      label: paramLabel(system, c.path),
    }
  })
  const compIters = Math.min(Math.max(1, Math.floor(num(s.compensator_iterations, 20))), 3)
  const work = clone(system)
  const base = tols.map((t) => getParamValue(system, t.path)!)
  const seed = Math.max(0, Math.floor(num(s.seed, 0)))
  const nominalComps = compVars.map((v) => v.start)

  /** Criterion of the system with these perturbations, compensators re-optimised. */
  const evalCase = (deltas: number[]): { criterion: number; comps: number[]; ok: boolean } => {
    tols.forEach((t, i) => setParamValue(work, t.path, base[i] + deltas[i]))
    compVars.forEach((v) => setParamValue(work, v.path, v.start))
    if (compVars.length === 0) {
      const e = evaluateMerit(work, setup)
      return { criterion: e.merit, comps: [], ok: e.ok }
    }
    const lm = new Lm(work, compVars, setup)
    for (let k = 0; k < compIters; k++) if (lm.step().converged) break
    return { criterion: lm.eval.merit, comps: lm.x.slice(), ok: lm.eval.ok }
  }
  const zero = tols.map(() => 0)
  const nominal = evalCase(zero)
  const nominalJson = { criterion: nf(nominal.criterion), ok: nominal.ok, compensators: nominalComps }
  const draws = (trial: number): number[] => {
    const r = rngOf(seed * 7919 + trial * 104729 + 17)
    return tols.map((t) => {
      if (t.distribution === 'normal' || t.distribution === 'gaussian') {
        const sigma = (t.minus + t.plus) / 4
        return Math.max(-t.minus, Math.min(t.plus, gauss(r) * sigma))
      }
      return -t.minus + (t.minus + t.plus) * r()
    })
  }
  const compensatorsOut = (Array.isArray(s.compensators) ? s.compensators : []) as Json

  // A single case: the perturbed system, compensated.
  if (s.preview_case && typeof s.preview_case === 'object') {
    const pc = s.preview_case as Obj
    let deltas: number[]
    if (pc.type === 'trial') deltas = draws(Math.max(0, Math.floor(num(pc.trial, 0))))
    else {
      const k = Math.floor(num(pc.operand, 0))
      if (k < 0 || k >= tols.length) throw new EngineError('invalidSettings', 'preview operand out of range')
      deltas = tols.map((t, i) => (i === k ? (pc.positive === false ? -t.minus : t.plus) : 0))
    }
    const r = evalCase(deltas)
    return newOut({
      status: 'finished',
      stopped: false,
      sensitivity: null,
      monte_carlo: null,
      compensators: compensatorsOut,
      criterion_spec: setup.spec,
      backend: backend(19),
      preview: {
        system: clone(work) as JsonObject,
        system_hash: await hashOf(work),
        criterion: nf(r.criterion),
        compensators: r.comps,
        perturbations: deltas,
      },
    })
  }

  const mode = String(s.mode ?? 'both')
  let stopped = false
  let sensitivity: JsonObject | null = null
  if (mode !== 'monte_carlo') {
    const ops: JsonObject[] = []
    const worst: number[] = []
    let evals = 1
    for (const [k, t] of tols.entries()) {
      ctx.check()
      if (ctx.stopped()) {
        stopped = true
        break
      }
      const minus = evalCase(tols.map((_, i) => (i === k ? -t.minus : 0)))
      const plus = evalCase(tols.map((_, i) => (i === k ? t.plus : 0)))
      evals += 2
      const cm = minus.criterion - nominal.criterion
      const cp = plus.criterion - nominal.criterion
      const w = Math.max(Math.abs(cm), Math.abs(cp))
      worst.push(w)
      ops.push({
        index: k,
        label: t.label || paramLabel(system, t.path),
        param: t.path as JsonObject,
        minus: t.minus,
        plus: t.plus,
        criterion_minus: nf(minus.criterion),
        criterion_plus: nf(plus.criterion),
        change_minus: nf(cm),
        change_plus: nf(cp),
        worst: nf(w),
        ok: minus.ok && plus.ok,
        compensators_minus: minus.comps,
        compensators_plus: plus.comps,
      })
      ctx.report(((k + 1) / Math.max(tols.length, 1)) * (mode === 'both' ? 0.3 : 1), 'Sensitivity', { stage: 'sensitivity', trials: tols.length * 2, done: (k + 1) * 2 })
      await ctx.tick()
    }
    const rss = Math.sqrt(worst.reduce((a, w) => a + w * w, 0))
    sensitivity = {
      nominal: nominalJson,
      operands: ops,
      ranking: worst.map((w, i) => [w, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).map((x) => x[1]),
      rss_change: nf(rss),
      predicted_criterion: nf(nominal.criterion + rss),
      evaluations: evals,
      cancelled: stopped,
    }
  }

  let monteCarlo: JsonObject | null = null
  const criteria: number[] = []
  if (mode !== 'sensitivity' && !stopped) {
    const trials = Math.max(1, Math.floor(num(s.trials, 100)))
    const recordN = Math.max(0, Math.floor(num(s.record_trials, 0)))
    const records: JsonObject[] = []
    const compSum = compVars.map(() => 0)
    const compSq = compVars.map(() => 0)
    let failures = 0
    const t0 = ctx.elapsedMs()
    for (let t = 0; t < trials; t++) {
      ctx.check()
      if (ctx.stopped()) {
        stopped = true
        break
      }
      if (ctx.elapsedMs() > ctx.budgetMs * 1.4) break
      const d = draws(t)
      const r = evalCase(d)
      if (!r.ok) failures++
      criteria.push(r.ok ? r.criterion : Infinity)
      r.comps.forEach((c, i) => {
        compSum[i] += c
        compSq[i] += c * c
      })
      if (t < recordN) records.push({ trial: t, criterion: nf(criteria[t]), ok: r.ok, compensators: r.comps, perturbations: d })
      if (t % 4 === 3 || t === trials - 1) {
        ctx.report((mode === 'both' ? 0.3 : 0) + ((mode === 'both' ? 0.7 : 1) * (t + 1)) / trials, 'Monte Carlo', { stage: 'monte_carlo', trials, done: t + 1 })
        await ctx.tick()
      }
    }
    const secs = criteria.length ? (ctx.elapsedMs() - t0) / 1000 / criteria.length : 0
    const good = criteria.filter(Number.isFinite).sort((a, b) => a - b)
    const pct = (p: number): number => {
      if (!good.length) return NaN
      const x = Math.min(Math.max(p / 100, 0), 1) * (good.length - 1)
      const i = Math.floor(x)
      const f = x - i
      return i + 1 < good.length ? good[i] * (1 - f) + good[i + 1] * f : good[i]
    }
    const levels = Array.isArray(s.percentiles) ? (s.percentiles as number[]) : [50, 90, 97.7]
    const mean = good.length ? good.reduce((a, b) => a + b, 0) / good.length : NaN
    const sd = good.length ? Math.sqrt(good.reduce((a, b) => a + (b - mean) ** 2, 0) / good.length) : NaN
    const thr = typeof s.threshold === 'number' ? s.threshold : null
    const n = Math.max(criteria.length, 1)
    monteCarlo = {
      trials: criteria.length,
      failures,
      seed,
      nominal: nominalJson,
      seconds_per_trial: secs,
      records,
      mean: nf(mean),
      std_dev: nf(sd),
      min: nf(good[0] ?? NaN),
      max: nf(good[good.length - 1] ?? NaN),
      percentiles: levels.map((p) => ({ percentile: p, value: nf(pct(p)) })),
      yield_fraction: thr === null || !criteria.length ? null : criteria.filter((c) => c <= thr).length / criteria.length,
      threshold: thr,
      cancelled: stopped,
      criteria: criteria.map((c) => (Number.isFinite(c) ? c : 'inf')),
      compensator_mean: compSum.map((v) => v / n),
      compensator_std: compSum.map((v, i) => Math.sqrt(Math.max(compSq[i] / n - (v / n) ** 2, 0))),
    }
  }
  const out = newOut({
    status: stopped ? 'stopped' : 'finished',
    stopped,
    sensitivity,
    monte_carlo: monteCarlo,
    compensators: compensatorsOut,
    criterion_spec: setup.spec,
    backend: backend(19),
  })
  if (monteCarlo && criteria.length) addBuf(out, 'monte_carlo_criteria', Float64Array.from(criteria), [criteria.length])
  return out
}

